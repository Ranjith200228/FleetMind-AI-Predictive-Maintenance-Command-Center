import pandas as pd

def get_feature_cols(df: pd.DataFrame):
    drop_cols = {"engine_id", "cycle", "RUL"}
    return [c for c in df.columns if c not in drop_cols]
