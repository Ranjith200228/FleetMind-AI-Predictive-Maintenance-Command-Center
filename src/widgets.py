"""Smart inline widgets for the Copilot.

The agent answers in plain prose; this module post-processes the text to
attach compact, glanceable telemetry cards for any ``engine N`` mention.

Cards are pure HTML+SVG (see :mod:`src.sparkline`) so they can be embedded
inside ``st.markdown(..., unsafe_allow_html=True)`` without re-rendering or
extra round-trips to the server.
"""
from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd

from src.sparkline import degradation_series, sparkline_svg
from src.tools import EngineDataBackend, query_engine_history

ENGINE_RE = re.compile(r"\bengine\s+(\d{1,3})\b", re.IGNORECASE)

ACTION_TINT = {
    "REPLACE": ("#E81922", "rgba(232,25,34,0.10)"),
    "REPAIR":  ("#ff8c42", "rgba(255,140,66,0.10)"),
    "INSPECT": ("#ffd166", "rgba(255,209,102,0.10)"),
    "MONITOR": ("#00E5FF", "rgba(0,229,255,0.08)"),
}


def find_engine_mentions(text: str, valid_ids: Iterable[int]) -> list[int]:
    """Return unique engine ids mentioned in the text, in first-seen order."""
    valid = set(int(i) for i in valid_ids)
    seen: list[int] = []
    for m in ENGINE_RE.finditer(text):
        eid = int(m.group(1))
        if eid in valid and eid not in seen:
            seen.append(eid)
    return seen


def telemetry_card_html(engine_id: int, backend: EngineDataBackend,
                        signal_col: str = "sensor_4") -> str:
    """Render a compact telemetry card for one engine as inline HTML."""
    history = backend.test_df[backend.test_df["engine_id"] == engine_id]
    history = history.sort_values("cycle")
    if signal_col not in history.columns:
        signal_col = next(c for c in history.columns if c.startswith("sensor_"))
    series = degradation_series(history[signal_col].values, n_points=30)

    pred = query_engine_history(int(engine_id), backend)["lstm_prediction"]
    rul = float(pred["rul_cycles"])
    action = pred["recommended_action"]
    n_cycles = int(history["cycle"].max())
    stroke, tint = ACTION_TINT.get(action, ACTION_TINT["MONITOR"])
    spark = sparkline_svg(series, width=120, height=28, stroke=stroke,
                          fill=tint, stroke_width=1.4)

    return f'''
<div class="telemetry-card" style="border-left: 2px solid {stroke};">
  <div class="telemetry-card-row">
    <span class="telemetry-card-id">ENGINE {engine_id:03d}</span>
    <span class="telemetry-card-action" style="color:{stroke}">{action}</span>
  </div>
  <div class="telemetry-card-spark">{spark}</div>
  <div class="telemetry-card-row">
    <span class="telemetry-card-meta">CYCLES&nbsp;{n_cycles}</span>
    <span class="telemetry-card-rul">RUL&nbsp;{rul:.1f}</span>
  </div>
</div>'''


def cards_for_answer(answer: str, backend: EngineDataBackend) -> str:
    """Build a side-by-side row of telemetry cards for engines mentioned in
    ``answer``. Returns empty string if none found."""
    eids = find_engine_mentions(answer, backend.engine_ids())
    if not eids:
        return ""
    cards = "".join(telemetry_card_html(e, backend) for e in eids[:3])
    return f'<div class="telemetry-row">{cards}</div>'
