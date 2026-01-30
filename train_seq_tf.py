import numpy as np
import joblib
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from src.load_data import load_train
from src.rul_model import add_rul
from src.split import split_by_engine
from src.sequence_tf import make_windows, SENSOR_COLS, SETTING_COLS
from src.lstm_tf import build_lstm_model

# Reproducibility (optional)
tf.random.set_seed(42)
np.random.seed(42)

WINDOW = 30
feature_cols = SETTING_COLS + SENSOR_COLS

# 1) Load + label
df = add_rul(load_train())

# 2) Engine split (no leakage)
train_df, val_df = split_by_engine(df, test_size=0.2, seed=42)

# 3) Scale using TRAIN only
scaler = StandardScaler()
scaler.fit(train_df[feature_cols])

train_df = train_df.copy()
val_df = val_df.copy()
train_df[feature_cols] = scaler.transform(train_df[feature_cols])
val_df[feature_cols] = scaler.transform(val_df[feature_cols])

# 4) Create windows
Xtr, ytr = make_windows(train_df, window=WINDOW, stride=1, feature_cols=feature_cols, target_col="RUL")
Xva, yva = make_windows(val_df, window=WINDOW, stride=1, feature_cols=feature_cols, target_col="RUL")

print("Train windows:", Xtr.shape, " Val windows:", Xva.shape)

# 5) Build model
model = build_lstm_model(n_features=len(feature_cols), window=WINDOW)
model.summary()

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.5, min_lr=1e-5),
]

# 6) Train
history = model.fit(
    Xtr, ytr,
    validation_data=(Xva, yva),
    epochs=15,
    batch_size=256,
    callbacks=callbacks,
    verbose=1
)

# 7) Save model + scaler + metadata
model.save("models/lstm_rul.keras")
joblib.dump(
    {"scaler": scaler, "feature_cols": feature_cols, "window": WINDOW},
    "models/lstm_meta.joblib"
)

print("Saved: models/lstm_rul.keras")
print("Saved: models/lstm_meta.joblib")
