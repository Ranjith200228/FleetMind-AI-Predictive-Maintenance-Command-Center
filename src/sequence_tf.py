import numpy as np
import pandas as pd

SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]
SETTING_COLS = [f"setting_{i}" for i in range(1, 4)]

def make_windows(df: pd.DataFrame, window: int = 30, stride: int = 1,
                 feature_cols=None, target_col="RUL"):
    """
    Create sliding windows per engine (NO leakage).
    Returns:
      X: (N, window, F)
      y: (N,)
    """
    if feature_cols is None:
        feature_cols = SETTING_COLS + SENSOR_COLS

    X_list, y_list = [], []
    for _, g in df.sort_values(["engine_id", "cycle"]).groupby("engine_id"):
        g = g.reset_index(drop=True)
        values = g[feature_cols].to_numpy(dtype=np.float32)
        target = g[target_col].to_numpy(dtype=np.float32)

        for end in range(window - 1, len(g), stride):
            start = end - window + 1
            X_list.append(values[start:end + 1])
            y_list.append(target[end])

    X = np.stack(X_list)
    y = np.array(y_list, dtype=np.float32)
    return X, y
