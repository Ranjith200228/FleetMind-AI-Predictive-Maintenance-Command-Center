"""Per-stage health score for a turbofan engine, derived from sensor trends.

The C-MAPSS FD001 fault mode is HPC degradation: as the high-pressure
compressor wears, downstream temperatures and pressures drift upward and
turbine coolant bleeds rise. We approximate per-stage health by aggregating
absolute z-scores of the relevant sensors, computed against the early-life
baseline of the same engine (cycles 1-20).

Stages and the sensors that reflect them. The mapping is the union across
the FD001 (HPC fault) and FD003 (HPC + Fan faults) subsets — the
``stage_healths`` aggregator filters to whichever sensors are actually
present in the engine's history, so the same map works for both subsets.

  Fan       : sensor_6  (P15, bypass duct pressure — FD003 only),
              sensor_8  (Nf, fan speed),
              sensor_13 (NRf, corrected fan speed),
              sensor_15 (BPR, bypass ratio)
  LPC       : sensor_2  (T24, LPC outlet temp)
  HPC       : sensor_3  (T30), sensor_7  (P30), sensor_9  (Nc),
              sensor_11 (Ps30), sensor_14 (NRc)
  Combustor : sensor_12 (phi), sensor_17 (htBleed)
  HPT       : sensor_4  (T50, LPT outlet temp),
              sensor_10 (epr, engine pressure ratio — FD003 only),
              sensor_20 (W31, HPT coolant bleed)
  LPT       : sensor_21 (W32, LPT coolant bleed)

Score interpretation: 0 ~ healthy, 1 ~ moderate drift, 2+ ~ degraded.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

STAGE_SENSORS = {
    "Fan":       ["sensor_6", "sensor_8", "sensor_13", "sensor_15"],
    "LPC":       ["sensor_2"],
    "HPC":       ["sensor_3", "sensor_7", "sensor_9", "sensor_11", "sensor_14"],
    "Combustor": ["sensor_12", "sensor_17"],
    "HPT":       ["sensor_4", "sensor_10", "sensor_20"],
    "LPT":       ["sensor_21"],
}

STAGE_ORDER = ["Fan", "LPC", "HPC", "Combustor", "HPT", "LPT"]


@dataclass
class StageHealth:
    name: str
    score: float       # 0 = healthy, higher = more degraded
    color: str         # hex, green->amber->red gradient
    contributing_sensors: list[str]
    sensor_zscores: dict[str, float]


def _baseline_stats(history: pd.DataFrame, baseline_cycles: int = 20
                    ) -> tuple[pd.Series, pd.Series]:
    """Mean / std of each sensor over the first `baseline_cycles` cycles."""
    base = history.sort_values("cycle").head(baseline_cycles)
    sensor_cols = [c for c in history.columns if c.startswith("sensor_")]
    return base[sensor_cols].mean(), base[sensor_cols].std(ddof=0).replace(0, np.nan)


def stage_healths(history: pd.DataFrame, recent_cycles: int = 5,
                  baseline_cycles: int = 20) -> list[StageHealth]:
    """Compute degradation z-scores per stage for one engine's history.

    history: a DataFrame for a single engine, sorted by cycle, containing
             at least a 'cycle' column and columns named 'sensor_<i>'.
    recent_cycles:   number of recent cycles to average for the "current" reading.
    baseline_cycles: number of early cycles used as the per-engine baseline.
    """
    if "cycle" not in history.columns:
        raise ValueError("history must include a 'cycle' column")
    sensor_cols = [c for c in history.columns if c.startswith("sensor_")]
    if not sensor_cols:
        raise ValueError("history has no sensor columns")
    sorted_h = history.sort_values("cycle")
    n = len(sorted_h)

    if n < max(recent_cycles, 5):
        recent_cycles = max(2, min(recent_cycles, n))
    if n < baseline_cycles:
        baseline_cycles = max(2, min(baseline_cycles, n - recent_cycles))
        if baseline_cycles < 2:
            baseline_cycles = max(2, n // 3)

    base_mean, base_std = _baseline_stats(sorted_h, baseline_cycles)
    recent_mean = sorted_h.tail(recent_cycles)[sensor_cols].mean()
    z = ((recent_mean - base_mean) / base_std).abs()
    z = z.fillna(0.0)

    out: list[StageHealth] = []
    for stage in STAGE_ORDER:
        sensors = [s for s in STAGE_SENSORS[stage] if s in z.index]
        if not sensors:
            score = 0.0
            contrib: dict[str, float] = {}
        else:
            contrib = {s: float(z[s]) for s in sensors}
            score = float(np.mean(list(contrib.values())))
        out.append(StageHealth(
            name=stage,
            score=score,
            color=_score_to_color(score),
            contributing_sensors=sensors,
            sensor_zscores=contrib,
        ))
    return out


# ---------------------------------------------------------------------------

def _score_to_color(score: float, hot: float = 2.5) -> str:
    """Map a score in [0, hot] to a green->amber->red hex color.

    Below 0.5 -> deep teal-green (healthy).
    0.5..1.5 -> amber (drifting).
    1.5+     -> red (degraded). Saturates at `hot`.
    """
    s = max(0.0, min(score / hot, 1.0))
    # waypoints:
    # 0.0 -> #18d4a4 (teal-green)
    # 0.5 -> #ffd166 (amber)
    # 1.0 -> #ff3860 (red)
    if s < 0.5:
        t = s / 0.5
        c = _lerp_hex("#18d4a4", "#ffd166", t)
    else:
        t = (s - 0.5) / 0.5
        c = _lerp_hex("#ffd166", "#ff3860", t)
    return c


def _lerp_hex(a: str, b: str, t: float) -> str:
    ar, ag, ab = int(a[1:3], 16), int(a[3:5], 16), int(a[5:7], 16)
    br, bg, bb = int(b[1:3], 16), int(b[3:5], 16), int(b[5:7], 16)
    r = int(round(ar + (br - ar) * t))
    g = int(round(ag + (bg - ag) * t))
    bb_ = int(round(ab + (bb - ab) * t))
    return f"#{r:02x}{g:02x}{bb_:02x}"
