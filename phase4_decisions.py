import joblib
import pandas as pd

from src.load_data import load_train
from src.rul_model import add_rul
from src.features import add_rolling_features, add_delta_features, add_health_index
from src.split import split_by_engine
from src.train_utils import get_feature_cols
from src.models import train_random_forest, evaluate
from src.decision_engine import add_decisions

# 1) Load + features
df = load_train()
df = add_rul(df)
df = add_rolling_features(df, window=5)
df = add_delta_features(df)
df = add_health_index(df)

# 2) Split by engine (NO leakage)
train_df, val_df = split_by_engine(df, test_size=0.2, seed=42)

feature_cols = get_feature_cols(train_df)
X_train, y_train = train_df[feature_cols], train_df["RUL"]
X_val, y_val = val_df[feature_cols], val_df["RUL"]

# 3) Train RF + evaluate
rf = train_random_forest(X_train, y_train)
pred_val = rf.predict(X_val)
print("RF:", evaluate(y_val, pred_val, "RandomForest"))

# 4) Save model + feature list (so inference is consistent)
joblib.dump({"model": rf, "features": feature_cols}, "models/rf_rul_model.joblib")
print("Saved: models/rf_rul_model.joblib")

# 5) Add predictions + decisions
out = val_df[["engine_id", "cycle", "RUL", "health_index"]].copy()
out["RUL_pred"] = pred_val
out = add_decisions(out, rul_col="RUL_pred")

# 6) Show decision counts
print("\nDecision counts:")
print(out["decision"].value_counts())

# 7) Save validation decisions for dashboard
out.to_csv("data/processed/val_predictions_decisions.csv", index=False)
print("\nSaved: data/processed/val_predictions_decisions.csv")
