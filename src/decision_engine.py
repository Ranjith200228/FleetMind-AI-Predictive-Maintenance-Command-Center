import pandas as pd

def decision_from_rul(rul_pred: float) -> str:
    """
    Tesla-style actionable decision rules.
    Tune thresholds later using cost / risk.
    """
    if rul_pred <= 15:
        return "SERVICE_NOW"
    elif rul_pred <= 40:
        return "MONITOR"
    else:
        return "OK"

def add_decisions(df: pd.DataFrame, rul_col: str = "RUL_pred") -> pd.DataFrame:
    df = df.copy()
    df["decision"] = df[rul_col].apply(decision_from_rul)
    return df
