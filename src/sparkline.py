"""Inline SVG micro-sparklines.

These are rendered as raw HTML inside Streamlit (no JS, no Plotly overhead),
so the 100-engine status grid stays under a few KB and paints instantly.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np


def sparkline_svg(values: Iterable[float],
                  *,
                  width: int = 88,
                  height: int = 22,
                  stroke: str = "#00E5FF",
                  stroke_width: float = 1.2,
                  fill: str | None = None,
                  ) -> str:
    """Return an inline ``<svg>`` micro-sparkline for the given series.

    Values are normalised to the SVG viewport. NaNs are dropped. Empty input
    returns an empty SVG of the requested size so the grid stays aligned.
    """
    arr = np.asarray(list(values), dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size < 2:
        return f'<svg width="{width}" height="{height}"></svg>'

    pad = 1.5
    n = arr.size
    lo, hi = float(arr.min()), float(arr.max())
    rng = hi - lo if hi - lo > 1e-9 else 1.0
    xs = np.linspace(pad, width - pad, n)
    ys = (height - pad) - ((arr - lo) / rng) * (height - 2 * pad)

    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))

    fill_part = ""
    if fill:
        # Build a closed polygon for a soft area fill under the line.
        first_x, last_x = xs[0], xs[-1]
        baseline = height - pad
        area = f"{first_x:.1f},{baseline:.1f} " + pts + f" {last_x:.1f},{baseline:.1f}"
        fill_part = (f'<polygon points="{area}" fill="{fill}" '
                     f'stroke="none" />')

    return (
        f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        f'xmlns="http://www.w3.org/2000/svg" style="display:block">'
        f'{fill_part}'
        f'<polyline points="{pts}" fill="none" stroke="{stroke}" '
        f'stroke-width="{stroke_width}" stroke-linejoin="round" '
        f'stroke-linecap="round" />'
        f'</svg>'
    )


def degradation_series(history_values: np.ndarray, n_points: int = 30) -> np.ndarray:
    """Take the last ``n_points`` of a sensor history. Pads with the first value
    if shorter, so every cell gets the same-width sparkline."""
    arr = np.asarray(history_values, dtype=float).ravel()
    if arr.size == 0:
        return np.zeros(n_points, dtype=float)
    if arr.size >= n_points:
        return arr[-n_points:]
    pad = np.full(n_points - arr.size, arr[0], dtype=float)
    return np.concatenate([pad, arr])
