"""The single tool exposed to the Copilot agent: query_engine_history.

Given a test-set engine id, return:
  * the last `window` cycles of the 14 informative sensors (raw values)
  * per-sensor summary statistics (last value, mean, std, linear trend slope)
  * the current LSTM RUL prediction for this engine
  * a recommended maintenance band (inspect / repair / replace) based on RUL

This is the *only* tool the agent calls. Everything else (background knowledge,
fault modes, citations) comes from the RAG retriever.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

DEFAULT_DATA_DIR = "data/raw"
DEFAULT_MODELS_DIR = "models"


# ---------------------------------------------------------------------------
# Data backend
# ---------------------------------------------------------------------------

@dataclass
class EngineDataBackend:
    """Holds the test-set data + scaler + fitted LSTM in memory for one C-MAPSS subset.

    Lazy-loads the LSTM only on the first prediction request to keep cold
    start under the 10 s HF Space requirement.
    """
    subset: str                  # "FD001" or "FD003"
    test_df: pd.DataFrame
    scaler: Any
    feature_cols: list[str]
    window: int
    rul_cap: int
    models_dir: Path
    _lstm: Any = None

    @classmethod
    def load(cls, data_dir: str | Path = DEFAULT_DATA_DIR,
             models_dir: str | Path = DEFAULT_MODELS_DIR,
             subset: str = "FD001") -> "EngineDataBackend":
        """Load the test data, fitted scaler and (lazily) the LSTM for ``subset``."""
        from src.preprocess import load_fd
        subset = subset.upper()
        _, test, _ = load_fd(subset, data_dir)
        pre = joblib.load(Path(models_dir) / f"preprocess_{subset.lower()}.joblib")
        return cls(
            subset=subset,
            test_df=test,
            scaler=pre["scaler"],
            feature_cols=pre["feature_cols"],
            window=pre["window"],
            rul_cap=pre["rul_cap"],
            models_dir=Path(models_dir),
        )

    @property
    def lstm(self):
        if self._lstm is None:
            import tensorflow as tf
            self._lstm = tf.keras.models.load_model(
                self.models_dir / f"lstm_{self.subset.lower()}.keras"
            )
        return self._lstm

    def engine_ids(self) -> list[int]:
        return sorted(self.test_df["engine_id"].unique().tolist())


# ---------------------------------------------------------------------------
# Public tool function
# ---------------------------------------------------------------------------

def query_engine_history(
    engine_id: int,
    backend: EngineDataBackend,
    include_predictions: bool = True,
) -> dict[str, Any]:
    """Return the last-window slice of an engine + summary stats + RUL prediction.

    Designed to fit inside an LLM tool response: bounded size, JSON-serialisable,
    no raw numpy arrays (lists only).
    """
    df = backend.test_df
    if engine_id not in df["engine_id"].unique():
        return {
            "ok": False,
            "engine_id": int(engine_id),
            "error": f"engine_id {engine_id} not found in {backend.subset} test set "
                     f"(valid range: {df['engine_id'].min()}..{df['engine_id'].max()})",
        }

    sub = df[df["engine_id"] == engine_id].sort_values("cycle")
    n_observed = int(len(sub))
    last = sub.tail(backend.window).copy()
    n_used = int(len(last))

    feats = last[backend.feature_cols].values.astype(np.float32)

    # Summary stats per sensor (over the last window)
    sensors_summary: dict[str, dict[str, float]] = {}
    cycles = last["cycle"].values.astype(float)
    for j, col in enumerate(backend.feature_cols):
        v = feats[:, j].astype(float)
        slope = _linfit_slope(cycles, v)
        sensors_summary[col] = {
            "last": float(v[-1]),
            "mean": float(v.mean()),
            "std": float(v.std()),
            "trend_per_cycle": float(slope),
        }

    out: dict[str, Any] = {
        "ok": True,
        "engine_id": int(engine_id),
        "n_observed_cycles": n_observed,
        "window_cycles_used": n_used,
        "last_cycle": int(last["cycle"].iloc[-1]),
        "feature_cols": list(backend.feature_cols),
        "sensors_summary": sensors_summary,
        "recent_window_raw": [
            {"cycle": int(r["cycle"]), **{c: float(r[c]) for c in backend.feature_cols}}
            for _, r in last.iterrows()
        ],
    }

    if include_predictions:
        # Build the model input window with proper scaling and left-padding.
        x = backend.scaler.transform(feats).astype(np.float32)
        if x.shape[0] < backend.window:
            pad = np.zeros((backend.window - x.shape[0], x.shape[1]), dtype=np.float32)
            x = np.vstack([pad, x])
        x = x[None, ...]  # (1, window, n_features)
        rul = float(backend.lstm.predict(x, verbose=0)[0, 0])
        rul = max(0.0, min(rul, float(backend.rul_cap)))
        band = _action_band(rul)
        out["lstm_prediction"] = {
            "rul_cycles": round(rul, 2),
            "rul_cap": backend.rul_cap,
            "recommended_action": band["action"],
            "action_rationale": band["rationale"],
        }
    return out


# ---------------------------------------------------------------------------
# OpenAI tool schema
# ---------------------------------------------------------------------------

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "query_engine_history",
        "description": (
            "Retrieve the last 30 cycles of sensor data for a specific engine "
            "in the currently active C-MAPSS test set (FD001 or FD003), along "
            "with per-sensor summary statistics (last value, mean, std, linear "
            "trend per cycle) and the current LSTM RUL prediction with a "
            "recommended maintenance action band (inspect / repair / replace). "
            "Use this tool whenever the user asks about a specific engine by "
            "id or asks for the current health of a particular engine. Engine "
            "ids are integers 1..100. Do not call this tool for general "
            "questions about C-MAPSS, fault modes, or modelling — those are "
            "answered from retrieved documentation only."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "engine_id": {
                    "type": "integer",
                    "description": "Engine id in the current C-MAPSS test set, 1 to 100.",
                    "minimum": 1,
                    "maximum": 100,
                }
            },
            "required": ["engine_id"],
            "additionalProperties": False,
        },
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _linfit_slope(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return 0.0
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    xm = x.mean()
    ym = y.mean()
    denom = ((x - xm) ** 2).sum()
    if denom == 0:
        return 0.0
    return float(((x - xm) * (y - ym)).sum() / denom)


def _action_band(rul: float) -> dict[str, str]:
    if rul < 10:
        return {
            "action": "REPLACE",
            "rationale": (
                "Predicted RUL is under 10 cycles; replace the affected module "
                "before next operational dispatch. The C-MAPSS asymmetric "
                "scoring function heavily penalises operating past the failure "
                "threshold, so do not defer."
            ),
        }
    if rul < 40:
        return {
            "action": "REPAIR",
            "rationale": (
                "Predicted RUL is 10-40 cycles; schedule a targeted repair at "
                "the next available maintenance slot. Confirm the affected "
                "module via the sensor trend (HPC degradation on FD001 shows "
                "rising T30, T50, P30, Ps30 and coolant bleeds W31/W32)."
            ),
        }
    if rul < 80:
        return {
            "action": "INSPECT",
            "rationale": (
                "Predicted RUL is 40-80 cycles. Borescope inspection at the "
                "next scheduled stop is appropriate; no immediate action "
                "required. Continue monitoring sensor trends."
            ),
        }
    return {
        "action": "MONITOR",
        "rationale": (
            "Predicted RUL exceeds 80 cycles; continue routine monitoring. "
            "The first 50-100 cycles of an engine's life typically show no "
            "degradation signal, so high RUL predictions early in life are "
            "expected."
        ),
    }


def call_tool(name: str, arguments: dict[str, Any] | str,
              backend: EngineDataBackend) -> str:
    """Dispatch a tool call by name. Returns a JSON string for the LLM."""
    if isinstance(arguments, str):
        arguments = json.loads(arguments) if arguments else {}
    if name == "query_engine_history":
        result = query_engine_history(int(arguments["engine_id"]), backend)
        return json.dumps(result)
    return json.dumps({"ok": False, "error": f"unknown tool: {name}"})
