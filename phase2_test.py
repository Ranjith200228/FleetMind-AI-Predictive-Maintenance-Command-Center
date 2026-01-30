from src.load_data import load_train
from src.rul_model import add_rul
from src.features import (
    add_rolling_features,
    add_delta_features,
    add_health_index
)

df = load_train()
df = add_rul(df)
df = add_rolling_features(df, window=5)
df = add_delta_features(df)
df = add_health_index(df)

print(df[[
    "engine_id",
    "cycle",
    "RUL",
    "health_index"
]].head())

print("\nFeature count:", df.shape[1])
print("Health index range:",
      df["health_index"].min(),
      df["health_index"].max())
