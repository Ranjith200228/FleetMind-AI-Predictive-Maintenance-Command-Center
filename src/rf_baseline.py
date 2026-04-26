"""Random Forest baseline for C-MAPSS RUL.

Uses hand-engineered per-cycle features: raw sensor values + rolling mean/std
over the last `window` cycles. A strong but transparent baseline for the LSTM
to beat.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


def _rolling_features(
    df: pd.DataFrame, feature_cols: list[str], window: int
) -> pd.DataFrame:
    """Per-engine rolling mean/std aligned to each row (last `window` cycles)."""
    grouped = df.groupby("engine_id")[feature_cols]
    means = grouped.rolling(window=window, min_periods=1).mean().reset_index(level=0, drop=True)
    stds = grouped.rolling(window=window, min_periods=1).std().reset_index(level=0, drop=True).fillna(0.0)
    means.columns = [f"{c}_mean{window}" for c in feature_cols]
    stds.columns = [f"{c}_std{window}" for c in feature_cols]
    return pd.concat([means, stds], axis=1)


def build_rf_features(
    df: pd.DataFrame, feature_cols: list[str], window: int = 30
) -> pd.DataFrame:
    roll = _rolling_features(df, feature_cols, window)
    base = df[feature_cols].reset_index(drop=True)
    roll = roll.reset_index(drop=True)
    return pd.concat([base, roll, df[["cycle"]].reset_index(drop=True)], axis=1)


def train_rf(
    train: pd.DataFrame,
    feature_cols: list[str],
    window: int = 30,
    n_estimators: int = 300,
    max_depth: int | None = 14,
    seed: int = 42,
) -> tuple[RandomForestRegressor, list[str]]:
    X = build_rf_features(train, feature_cols, window)
    y = train["RUL"].values
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=seed,
    )
    model.fit(X.values, y)
    return model, list(X.columns)


def predict_rf_last(
    model: RandomForestRegressor,
    test: pd.DataFrame,
    feature_cols: list[str],
    rf_columns: list[str],
    window: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict one RUL per engine using the last-cycle row's features."""
    X = build_rf_features(test, feature_cols, window)
    X = X.reindex(columns=rf_columns, fill_value=0.0)
    last_idx = test.groupby("engine_id").tail(1).index
    eids = test.loc[last_idx, "engine_id"].values
    preds = model.predict(X.iloc[last_idx].values)
    order = np.argsort(eids)
    return preds[order], eids[order]


def save_rf(model: RandomForestRegressor, rf_columns: list[str], out_path: str | Path) -> None:
    import joblib
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "columns": rf_columns}, out_path)
