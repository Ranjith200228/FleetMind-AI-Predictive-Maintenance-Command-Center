"""Plotly 3D model of a high-bypass turbofan, coloured by per-stage health.

Each stage (Fan, LPC, HPC, Combustor, HPT, LPT, Nozzle) is rendered as a
parametric surface — a frustum sweep with smooth radius interpolation —
so the whole engine reads as one continuous body. The colour of each stage
is set by the per-stage health score from src.health.

The figure is dark-themed to match the rest of the dashboard, with smooth
shading, a horizontal turbofan axis, and an outer translucent fan cowling.
Fan blades are added as a thin disk with radial vanes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import plotly.graph_objects as go

from src.health import STAGE_ORDER, StageHealth


# ---------------------------------------------------------------------------
# Geometry definitions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StageGeom:
    name: str
    z0: float   # axial start
    z1: float   # axial end
    r0: float   # radius at z0
    r1: float   # radius at z1


# Approximate axial layout of a high-bypass turbofan (relative units).
DEFAULT_GEOMETRY: list[StageGeom] = [
    StageGeom("Fan",       z0=0.00, z1=0.55, r0=1.00, r1=0.95),
    StageGeom("LPC",       z0=0.55, z1=1.30, r0=0.55, r1=0.45),
    StageGeom("HPC",       z0=1.30, z1=2.30, r0=0.45, r1=0.30),
    StageGeom("Combustor", z0=2.30, z1=2.75, r0=0.40, r1=0.40),
    StageGeom("HPT",       z0=2.75, z1=3.20, r0=0.40, r1=0.50),
    StageGeom("LPT",       z0=3.20, z1=4.00, r0=0.55, r1=0.65),
    StageGeom("Nozzle",    z0=4.00, z1=4.55, r0=0.55, r1=0.40),
]

NACELLE: StageGeom = StageGeom("Nacelle", z0=-0.05, z1=4.10, r0=1.10, r1=1.00)

# Theme palette (mission-control)
BG = "#050505"
GRID = "#1a1a1f"
TEXT = "#e5e7eb"
TEXT_DIM = "#6b7280"
ACCENT = "#00E5FF"    # electric blue (telemetry / AI)
CRITICAL = "#E81922"  # Tesla red (reserved for REPLACE alerts)


# ---------------------------------------------------------------------------
# Surface helpers
# ---------------------------------------------------------------------------

def _frustum_surface(stage: StageGeom, n_theta: int = 48, n_axial: int = 14
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta = np.linspace(0, 2 * np.pi, n_theta)
    z = np.linspace(stage.z0, stage.z1, n_axial)
    Theta, Z = np.meshgrid(theta, z)
    # smooth-step radius interpolation for a hint of curvature
    t = (Z - stage.z0) / max(stage.z1 - stage.z0, 1e-9)
    t_smooth = t * t * (3 - 2 * t)
    R = stage.r0 + (stage.r1 - stage.r0) * t_smooth
    X = R * np.cos(Theta)
    Y = R * np.sin(Theta)
    return X, Y, Z


def _ring_disk(z: float, r_inner: float, r_outer: float, n_theta: int = 48
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A thin annular disk at axial position z (used for fan front face)."""
    theta = np.linspace(0, 2 * np.pi, n_theta)
    R = np.array([r_inner, r_outer])
    Theta, Rg = np.meshgrid(theta, R)
    X = Rg * np.cos(Theta)
    Y = Rg * np.sin(Theta)
    Z = np.full_like(X, z)
    return X, Y, Z


def _fan_blades_lines(z: float, r_inner: float, r_outer: float, n_blades: int = 22
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Radial lines representing fan blades at axial position z."""
    angles = np.linspace(0, 2 * np.pi, n_blades, endpoint=False)
    xs, ys, zs = [], [], []
    for a in angles:
        # add a slight pitch by offsetting end-z by ~5% of disk thickness
        xs += [r_inner * np.cos(a), r_outer * np.cos(a), None]
        ys += [r_inner * np.sin(a), r_outer * np.sin(a), None]
        zs += [z, z + 0.04, None]
    return np.array(xs, dtype=object), np.array(ys, dtype=object), np.array(zs, dtype=object)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_engine_figure(
    healths: list[StageHealth] | None = None,
    *,
    title: str | None = None,
    show_nacelle: bool = True,
    hud: dict | None = None,
) -> go.Figure:
    """Build a Plotly 3D Figure of the turbofan, coloured by stage health.

    ``hud`` is an optional dict with optional keys
    ``engine_id``, ``cycle_count``, ``degradation_index``, ``status``,
    rendered as thin-lined HUD text in the corners of the 3D viewport.
    """
    healths_by_name = {h.name: h for h in (healths or [])}
    fig = go.Figure()

    # Outer translucent nacelle cowling
    if show_nacelle:
        Xn, Yn, Zn = _frustum_surface(NACELLE, n_theta=72, n_axial=24)
        fig.add_trace(go.Surface(
            x=Xn, y=Yn, z=Zn,
            colorscale=[[0, "#3a3a44"], [1, "#1a1a20"]],
            showscale=False, opacity=0.18, lighting=dict(ambient=0.55, diffuse=0.7),
            hoverinfo="skip", name="Nacelle",
        ))

    # Stages
    for stage in DEFAULT_GEOMETRY:
        X, Y, Z = _frustum_surface(stage)
        h = healths_by_name.get(stage.name)
        if h is None:
            color = "#3aa0ff"
            hover = stage.name
            score = 0.0
        else:
            color = h.color
            score = h.score
            zlines = "<br>".join(f"  {s}: {z:+.2f}" for s, z in h.sensor_zscores.items())
            hover = (f"<b>{stage.name}</b><br>health score: {score:.2f}<br>"
                     f"contributing sensors:<br>{zlines}")

        # Lighting + colour: solid colour per stage via a flat colorscale
        fig.add_trace(go.Surface(
            x=X, y=Y, z=Z,
            colorscale=[[0, color], [1, color]],
            cmin=0, cmax=1,
            showscale=False,
            lighting=dict(ambient=0.42, diffuse=0.85, specular=0.45,
                          roughness=0.5, fresnel=0.15),
            lightposition=dict(x=2, y=2, z=4),
            hoverinfo="text", hovertext=hover,
            name=stage.name,
        ))

    # Fan disk + blades (visual flourish at engine front)
    Xd, Yd, Zd = _ring_disk(z=0.0, r_inner=0.10, r_outer=0.92)
    fig.add_trace(go.Surface(
        x=Xd, y=Yd, z=Zd,
        colorscale=[[0, "#2a2a30"], [1, "#4a4a55"]],
        showscale=False, opacity=0.95,
        lighting=dict(ambient=0.6, diffuse=0.5),
        hoverinfo="skip", name="Fan disk",
    ))
    bx, by, bz = _fan_blades_lines(z=0.005, r_inner=0.12, r_outer=0.90, n_blades=22)
    fig.add_trace(go.Scatter3d(
        x=bx, y=by, z=bz, mode="lines",
        line=dict(color="#cfd0d6", width=2),
        hoverinfo="skip", name="Fan blades", showlegend=False,
    ))
    # Centre cone (spinner)
    spinner = StageGeom("Spinner", z0=-0.05, z1=0.18, r0=0.0, r1=0.12)
    Xs, Ys, Zs = _frustum_surface(spinner, n_theta=36, n_axial=8)
    fig.add_trace(go.Surface(
        x=Xs, y=Ys, z=Zs,
        colorscale=[[0, "#cfd0d6"], [1, "#9a9aa3"]],
        showscale=False, lighting=dict(ambient=0.7, diffuse=0.7, specular=0.6),
        hoverinfo="skip", name="Spinner",
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        title=dict(text=title or "", x=0.02, y=0.97,
                   font=dict(color=TEXT, size=16)) if title else None,
        scene=dict(
            xaxis=dict(visible=False, showbackground=False),
            yaxis=dict(visible=False, showbackground=False),
            zaxis=dict(visible=False, showbackground=False),
            bgcolor="rgba(0,0,0,0)",
            aspectmode="data",
            camera=dict(eye=dict(x=1.6, y=1.4, z=0.9),
                        up=dict(x=0, y=0, z=1),
                        center=dict(x=0, y=0, z=0)),
        ),
        margin=dict(l=0, r=0, t=10, b=0),
        showlegend=False,
        hoverlabel=dict(bgcolor="#0a0a0a", font_color=TEXT,
                        bordercolor="#1a1a1f"),
    )
    if hud:
        _apply_hud(fig, hud)
    return fig


# ---------------------------------------------------------------------------
# HUD overlay
# ---------------------------------------------------------------------------

def _hud_block(label: str, value: str) -> str:
    return (
        f"<span style='color:{TEXT_DIM}; letter-spacing:0.18em; "
        f"font-size:9px;'>{label}</span><br>"
        f"<span style='color:{TEXT}; letter-spacing:0.04em; "
        f"font-size:13px;'>{value}</span>"
    )


def _apply_hud(fig: go.Figure, hud: dict) -> None:
    """Add four corner HUD blocks anchored to the paper (the figure container).

    The font-family is enforced via ``font=dict(family=...)`` so the HUD looks
    like instrument readout regardless of the host page's CSS.
    """
    eid = hud.get("engine_id")
    cyc = hud.get("cycle_count")
    deg = hud.get("degradation_index")
    status = hud.get("status")

    items = []
    if eid is not None:
        items.append(("ENGINE ID", f"E-{int(eid):03d}",
                      0.01, 0.99, "left", "top"))
    if cyc is not None:
        items.append(("CYCLES", f"{int(cyc):04d}",
                      0.99, 0.99, "right", "top"))
    if deg is not None:
        items.append(("DEGRADATION INDEX", f"{float(deg):+.2f}σ",
                      0.01, 0.04, "left", "bottom"))
    if status is not None:
        # status colour: red for REPLACE, electric blue otherwise
        sc = CRITICAL if status == "REPLACE" else ACCENT
        items.append((
            "STATUS",
            f"<span style='color:{sc}'>{status}</span>",
            0.99, 0.04, "right", "bottom",
        ))

    annotations = list(fig.layout.annotations or [])
    for label, value, x, y, xanchor, yanchor in items:
        annotations.append(dict(
            xref="paper", yref="paper", x=x, y=y,
            xanchor=xanchor, yanchor=yanchor,
            showarrow=False,
            text=_hud_block(label, value),
            align="left" if xanchor == "left" else "right",
            font=dict(family="JetBrains Mono, IBM Plex Mono, ui-monospace, monospace",
                      color=TEXT, size=11),
            bgcolor="rgba(0,0,0,0)",
            borderpad=0,
        ))

    # Thin corner ticks: four small line "brackets" using shapes for that
    # cockpit-instrument feel.
    shapes = list(fig.layout.shapes or [])
    bracket_len = 0.022
    bracket_color = "rgba(229, 231, 235, 0.35)"
    for cx, cy in [(0.0, 1.0), (1.0, 1.0), (0.0, 0.0), (1.0, 0.0)]:
        dx = bracket_len if cx == 0.0 else -bracket_len
        dy = -bracket_len if cy == 1.0 else bracket_len
        shapes.append(dict(type="line", xref="paper", yref="paper",
                           x0=cx, y0=cy, x1=cx + dx, y1=cy,
                           line=dict(color=bracket_color, width=1)))
        shapes.append(dict(type="line", xref="paper", yref="paper",
                           x0=cx, y0=cy, x1=cx, y1=cy + dy,
                           line=dict(color=bracket_color, width=1)))

    fig.update_layout(annotations=annotations, shapes=shapes)


def build_fleet_grid_figure(eids: list[int], ruls: list[float],
                             rul_cap: float = 125.0) -> go.Figure:
    """Compact 10×10 grid heatmap of predicted RUL across the 100-engine fleet."""
    n = len(eids)
    side = int(np.ceil(np.sqrt(n)))
    grid = np.full((side, side), np.nan)
    label = np.full((side, side), "", dtype=object)
    for i, (e, r) in enumerate(zip(eids, ruls)):
        row, col = divmod(i, side)
        grid[row, col] = r
        label[row, col] = f"engine {e}<br>RUL {r:.1f}"
    fig = go.Figure(data=go.Heatmap(
        z=grid, text=label, hoverinfo="text",
        colorscale=[
            [0.00, "#ff3860"],
            [0.20, "#ff8c42"],
            [0.45, "#ffd166"],
            [0.75, "#7fdfa0"],
            [1.00, "#18d4a4"],
        ],
        zmin=0, zmax=rul_cap,
        colorbar=dict(title=dict(text="RUL (cycles)", font=dict(color=TEXT)),
                      tickfont=dict(color=TEXT), outlinewidth=0,
                      thickness=14, len=0.8),
        xgap=2, ygap=2,
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(visible=False), yaxis=dict(visible=False, autorange="reversed"),
        height=420,
    )
    return fig
