"""C-MAPSS preprocessing: piecewise RUL, feature selection, scaling, windowing."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

COLUMN_NAMES = (
    ["engine_id", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

# Per-subset constant-sensor drop lists. Derived empirically from the
# train splits (std == 0 → drop). FD001 follows Zheng et al. 2017 (also
# drops s6 because it's binary-jumpy with near-zero variance). FD003 keeps
# s6 + s10 — both move meaningfully with the second fault mode.
DROP_SENSORS = {
    "FD001": [1, 5, 6, 10, 16, 18, 19],   # 14 keep
    "FD003": [1, 5, 16, 18, 19],          # 16 keep
}

# Backward-compat names (used by existing modules and saved artifacts)
DROP_SENSORS_FD001 = DROP_SENSORS["FD001"]
KEEP_SENSORS_FD001 = [i for i in range(1, 22) if i not in DROP_SENSORS_FD001]
FEATURE_COLS_FD001 = [f"sensor_{i}" for i in KEEP_SENSORS_FD001]

SUPPORTED_SUBSETS: tuple[str, ...] = tuple(DROP_SENSORS.keys())

RUL_CAP = 125  # piecewise linear clip (Heimes 2008; Zheng 2017)


def feature_cols_for(subset: str) -> list[str]:
    """Return the list of sensor column names to use as features for ``subset``."""
    subset = subset.upper()
    if subset not in DROP_SENSORS:
        raise ValueError(f"unsupported subset: {subset!r} "
                         f"(supported: {SUPPORTED_SUBSETS})")
    keep = [i for i in range(1, 22) if i not in DROP_SENSORS[subset]]
    return [f"sensor_{i}" for i in keep]


def load_fd(subset: str, data_dir: str | Path = "data/raw"
            ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Generic loader for any C-MAPSS subset. Returns (train, test, rul)."""
    subset = subset.upper()
    if subset not in DROP_SENSORS:
        raise ValueError(f"unsupported subset: {subset!r} "
                         f"(supported: {SUPPORTED_SUBSETS})")
    data_dir = Path(data_dir)
    train = _read_fd(data_dir / f"train_{subset}.txt")
    test = _read_fd(data_dir / f"test_{subset}.txt")
    rul = pd.read_csv(data_dir / f"RUL_{subset}.txt", header=None).iloc[:, 0]
    return train, test, rul


def load_fd001(data_dir: str | Path = "data/raw") -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Backward-compat alias for :func:`load_fd` with ``subset='FD001'``."""
    return load_fd("FD001", data_dir)


def _read_fd(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    df = df.loc[:, ~df.isna().all(axis=0)]
    df.columns = COLUMN_NAMES[: df.shape[1]]
    return df.sort_values(["engine_id", "cycle"]).reset_index(drop=True)


def add_piecewise_rul(train: pd.DataFrame, cap: int = RUL_CAP) -> pd.DataFrame:
    train = train.copy()
    max_cycle = train.groupby("engine_id")["cycle"].transform("max")
    train["RUL"] = (max_cycle - train["cycle"]).clip(upper=cap)
    return train


def fit_scaler(train: pd.DataFrame, feature_cols: list[str]) -> MinMaxScaler:
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(train[feature_cols].values)
    return scaler


def make_sequences(
    df: pd.DataFrame,
    feature_cols: list[str],
    scaler: MinMaxScaler,
    window: int = 30,
    include_label: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (X, y, engine_ids) where X is (n, window, n_features).

    Sliding window per engine. For engines with fewer cycles than `window`,
    we left-pad with zeros so they produce a single sample.
    """
    X_list, y_list, eid_list = [], [], []
    feats = df[feature_cols].values.astype(np.float32)
    feats = scaler.transform(feats).astype(np.float32)
    df = df.assign(__fidx=np.arange(len(df)))

    for eid, grp in df.groupby("engine_id", sort=False):
        idx = grp["__fidx"].values
        n = len(idx)
        arr = feats[idx]
        if include_label:
            y = grp["RUL"].values.astype(np.float32)
        if n < window:
            pad = np.zeros((window - n, arr.shape[1]), dtype=np.float32)
            arr = np.vstack([pad, arr])
            X_list.append(arr)
            if include_label:
                y_list.append(y[-1])
            eid_list.append(eid)
            continue
        for end in range(window, n + 1):
            X_list.append(arr[end - window : end])
            if include_label:
                y_list.append(y[end - 1])
            eid_list.append(eid)
    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.float32) if include_label else np.empty(0)
    return X, y, np.array(eid_list)


def make_test_sequences(
    test: pd.DataFrame,
    feature_cols: list[str],
    scaler: MinMaxScaler,
    window: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    """Last `window` cycles per engine — standard C-MAPSS test protocol.

    Returns (X, engine_ids) in sorted engine_id order matching RUL_FD001.txt.
    """
    X_list, eid_list = [], []
    feats = scaler.transform(test[feature_cols].values.astype(np.float32)).astype(np.float32)
    test = test.assign(__fidx=np.arange(len(test)))
    for eid, grp in test.groupby("engine_id", sort=True):
        idx = grp["__fidx"].values
        arr = feats[idx]
        n = len(arr)
        if n >= window:
            arr = arr[-window:]
        else:
            pad = np.zeros((window - n, arr.shape[1]), dtype=np.float32)
            arr = np.vstack([pad, arr])
        X_list.append(arr)
        eid_list.append(eid)
    return np.stack(X_list, axis=0), np.array(eid_list)
