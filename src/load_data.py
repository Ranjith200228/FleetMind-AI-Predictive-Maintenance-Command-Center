import pandas as pd

COLUMN_NAMES = (
    ["engine_id", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

def load_turbofan_txt(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=" ", header=None)
    df = df.loc[:, ~df.isna().all(axis=0)]  # drop empty columns
    df.columns = COLUMN_NAMES
    df = df.sort_values(["engine_id", "cycle"]).reset_index(drop=True)
    return df

def load_train(path="data/raw/train_FD001.txt"):
    return load_turbofan_txt(path)

def load_test(path="data/raw/test_FD001.txt"):
    return load_turbofan_txt(path)

def load_rul(path="data/raw/RUL_FD001.txt"):
    return pd.read_csv(path, header=None)[0]
