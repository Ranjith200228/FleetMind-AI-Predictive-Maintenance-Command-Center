import pandas as pd

def add_rul(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add Remaining Useful Life (RUL) to training data.
    RUL = max_cycle_per_engine - current_cycle
    """
    df = df.copy()
    max_cycle = df.groupby("engine_id")["cycle"].transform("max")
    df["RUL"] = max_cycle - df["cycle"]
    return df
