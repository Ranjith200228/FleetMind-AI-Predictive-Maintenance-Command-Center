"""Cached resource loaders and shared helpers for the Streamlit app.

All heavyweight objects (LSTM, retriever, agent, scaler) are wrapped with
``st.cache_resource`` so the app cold-starts in well under 10 seconds and
re-renders are instant.

Resources that depend on the active C-MAPSS subset (engine backend, fleet
predictions) take a ``subset`` argument; ``st.cache_resource`` and
``st.cache_data`` automatically key the cache on the argument value, so
switching between FD001 and FD003 in the dashboard is a one-frame swap.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.agent import FleetMindAgent
from src.preprocess import load_fd
from src.rag import Retriever
from src.tools import EngineDataBackend


@st.cache_resource(show_spinner="Loading C-MAPSS data…")
def get_fd_data(subset: str = "FD001") -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    return load_fd(subset, "data/raw")


@st.cache_resource(show_spinner="Loading preprocessing artifacts…")
def get_preprocess(subset: str = "FD001") -> dict[str, Any]:
    return joblib.load(f"models/preprocess_{subset.lower()}.joblib")


@st.cache_resource(show_spinner="Loading retrieval index…")
def get_retriever() -> Retriever:
    return Retriever.load("data/rag_index")


@st.cache_resource(show_spinner="Loading engine backend…")
def get_engine_backend(subset: str = "FD001") -> EngineDataBackend:
    return EngineDataBackend.load("data/raw", "models", subset=subset)


@st.cache_resource(show_spinner="Initialising Copilot…")
def get_agent(subset: str = "FD001",
              model: str = "gpt-4o-mini") -> FleetMindAgent:
    return FleetMindAgent(get_retriever(), get_engine_backend(subset),
                          model=model)


# Backward-compat alias used by older entry points (always returns FD001).
def get_fd001() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    return get_fd_data("FD001")


@st.cache_data(show_spinner="Predicting fleet RUL…")
def predict_fleet_rul(subset: str = "FD001") -> pd.DataFrame:
    """Run the LSTM on the last 30 cycles of every test engine and return a DF."""
    backend = get_engine_backend(subset)
    test = backend.test_df
    pre = get_preprocess(subset)
    scaler = pre["scaler"]
    feature_cols = pre["feature_cols"]
    window = pre["window"]
    cap = pre["rul_cap"]

    eids: list[int] = []
    Xs: list[np.ndarray] = []
    for eid, grp in test.groupby("engine_id", sort=True):
        feats = grp[feature_cols].values.astype(np.float32)
        feats = scaler.transform(feats).astype(np.float32)
        if feats.shape[0] >= window:
            x = feats[-window:]
        else:
            pad = np.zeros((window - feats.shape[0], feats.shape[1]), dtype=np.float32)
            x = np.vstack([pad, feats])
        Xs.append(x)
        eids.append(int(eid))
    X = np.stack(Xs, axis=0)
    preds = backend.lstm.predict(X, verbose=0).ravel()
    preds = np.clip(preds, 0, cap)
    actions = [_band(p) for p in preds]
    return pd.DataFrame({
        "engine_id": eids,
        "predicted_rul": preds.round(2),
        "action": actions,
    }).sort_values("engine_id").reset_index(drop=True)


def _band(rul: float) -> str:
    if rul < 10:
        return "REPLACE"
    if rul < 40:
        return "REPAIR"
    if rul < 80:
        return "INSPECT"
    return "MONITOR"
