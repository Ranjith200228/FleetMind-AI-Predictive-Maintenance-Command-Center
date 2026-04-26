"""Evaluation metrics for C-MAPSS RUL prediction.

The C-MAPSS scoring function (Saxena & Goebel, 2008) penalizes late
predictions more heavily than early ones, because a late prediction means
a missed maintenance window in practice.
"""
from __future__ import annotations

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    return float(np.mean(np.abs(y_true - y_pred)))


def cmapss_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Asymmetric C-MAPSS score.

    d = y_pred - y_true
    if d < 0:  score += exp(-d/13) - 1      (early, mild)
    else:      score += exp( d/10) - 1      (late, severe)
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    d = y_pred - y_true
    s = np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)
    return float(np.sum(s))


def score_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "cmapss_score": cmapss_score(y_true, y_pred),
        "n": int(np.asarray(y_true).size),
    }
