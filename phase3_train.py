from src.load_data import load_train
from src.rul_model import add_rul
from src.features import add_rolling_features, add_delta_features, add_health_index
from src.split import split_by_engine
from src.train_utils import get_feature_cols
from src.models import train_ridge, train_random_forest, evaluate

# 1) Load + labels
df = load_train()
df = add_rul(df)

# 2) Features (same as Phase 2)
df = add_rolling_features(df, window=5)
df = add_delta_features(df)
df = add_health_index(df)

# 3) Split by engine (NO leakage)
train_df, val_df = split_by_engine(df, test_size=0.2, seed=42)

feature_cols = get_feature_cols(train_df)

X_train = train_df[feature_cols]
y_train = train_df["RUL"]

X_val = val_df[feature_cols]
y_val = val_df["RUL"]

# 4) Baseline model: Ridge
ridge = train_ridge(X_train, y_train)
pred_ridge = ridge.predict(X_val)
res_ridge = evaluate(y_val, pred_ridge, "Ridge Baseline")

# 5) Stronger model: RandomForest
rf = train_random_forest(X_train, y_train)
pred_rf = rf.predict(X_val)
res_rf = evaluate(y_val, pred_rf, "RandomForest")

print("=== Validation Results (Split by Engine) ===")
print(f"{res_ridge.name} | MAE={res_ridge.mae:.2f} | RMSE={res_ridge.rmse:.2f}")
print(f"{res_rf.name}    | MAE={res_rf.mae:.2f} | RMSE={res_rf.rmse:.2f}")

# 6) Tesla-style: Evaluate near failure only (RUL <= 30)
mask = y_val <= 30
if mask.sum() > 0:
    near_mae = abs(y_val[mask] - pred_rf[mask]).mean()
    print(f"\nNear-failure MAE (RUL<=30) for RF: {near_mae:.2f} (n={mask.sum()})")
