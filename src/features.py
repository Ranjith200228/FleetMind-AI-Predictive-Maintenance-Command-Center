import pandas as pd

SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]

def add_rolling_features(
    df: pd.DataFrame,
    window: int = 5
) -> pd.DataFrame:
    """
    Add rolling mean and std for sensor signals.
    Computed per engine to avoid data leakage.
    """
    df = df.copy()

    for col in SENSOR_COLS:
        df[f"{col}_roll_mean"] = (
            df.groupby("engine_id")[col]
            .rolling(window, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )

        df[f"{col}_roll_std"] = (
            df.groupby("engine_id")[col]
            .rolling(window, min_periods=1)
            .std()
            .reset_index(level=0, drop=True)
            .fillna(0)
        )

    return df
def add_delta_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add first-order difference (trend / wear rate).
    """
    df = df.copy()

    for col in SENSOR_COLS:
        df[f"{col}_delta"] = (
            df.groupby("engine_id")[col]
            .diff()
            .fillna(0)
        )

    return df

from sklearn.preprocessing import MinMaxScaler
import numpy as np

def add_health_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a single Health Index (0–1).
    1 = healthy, 0 = near failure.
    """
    df = df.copy()

    scaler = MinMaxScaler()

    sensor_data = df[SENSOR_COLS]
    scaled = scaler.fit_transform(sensor_data)

    # Health index = inverse mean degradation
    df["health_index"] = 1 - np.mean(scaled, axis=1)

    return df
