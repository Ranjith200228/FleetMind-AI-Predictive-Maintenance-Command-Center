import numpy as np
import pandas as pd

def split_by_engine(
    df: pd.DataFrame,
    test_size: float = 0.2,
    seed: int = 42
):
    rng = np.random.default_rng(seed)
    engines = df["engine_id"].unique()
    rng.shuffle(engines)

    n_test = int(len(engines) * test_size)
    test_engines = set(engines[:n_test])

    train_df = df[~df["engine_id"].isin(test_engines)].reset_index(drop=True)
    val_df = df[df["engine_id"].isin(test_engines)].reset_index(drop=True)
    return train_df, val_df
