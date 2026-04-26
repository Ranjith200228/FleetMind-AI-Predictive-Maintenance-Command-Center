"""Train the LSTM RUL model on C-MAPSS FD001.

Architecture and hyperparameters chosen to be competitive with published
baselines on FD001 (target RMSE 13-16, Score 300-500).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf


def build_lstm(window: int, n_features: int) -> tf.keras.Model:
    inp = tf.keras.Input(shape=(window, n_features))
    x = tf.keras.layers.Masking(mask_value=0.0)(inp)
    x = tf.keras.layers.LSTM(128, return_sequences=True)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.LSTM(64, return_sequences=False)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    out = tf.keras.layers.Dense(1, activation="linear")(x)
    model = tf.keras.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="mse",
        metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse"),
                 tf.keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model


def train_lstm(
    X: np.ndarray,
    y: np.ndarray,
    window: int,
    out_path: str | Path,
    val_frac: float = 0.1,
    epochs: int = 60,
    batch_size: int = 256,
    seed: int = 42,
) -> tf.keras.Model:
    tf.keras.utils.set_random_seed(seed)
    n_features = X.shape[2]
    model = build_lstm(window, n_features)

    n = X.shape[0]
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n_val = int(n * val_frac)
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_rmse", mode="min", patience=8,
            restore_best_weights=True, verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_rmse", mode="min", factor=0.5, patience=4, min_lr=1e-5,
        ),
    ]

    model.fit(
        X[tr_idx], y[tr_idx],
        validation_data=(X[val_idx], y[val_idx]),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=2,
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(out_path)
    return model
