"""FleetMind AI — mission-control dashboard.

Three views:
  1. Fleet Status     — KPI bar + 100-cell high-density status grid w/ sparklines.
  2. Engine Detail    — transparent 3D turbofan with corner HUD + per-stage health.
  3. Neural Copilot   — glassmorphism overlay, neural-pulse orb, smart widgets.

Design language: deep black (#050505), single accent (electric blue #00E5FF),
slate-gray + white text, monospace for telemetry, Inter for headings.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.health import STAGE_ORDER, stage_healths  # noqa: E402
from src.sparkline import degradation_series, sparkline_svg  # noqa: E402
from src.streamlit_helpers import (  # noqa: E402
    get_agent, get_engine_backend, get_fd001, get_retriever, predict_fleet_rul,
)
from src.tools import query_engine_history  # noqa: E402
from src.viz3d import (  # noqa: E402
    ACCENT, BG, CRITICAL, GRID, TEXT, TEXT_DIM,
    build_engine_figure, build_fleet_grid_figure,
)
from src.widgets import cards_for_answer  # noqa: E402
from src.report import ReportInputs, build_executive_pdf  # noqa: E402

# ---------------------------------------------------------------------------
# Tiny formatting helpers (used by the chat history renderer)
# ---------------------------------------------------------------------------

def _escape(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def _md_to_html(s: str) -> str:
    """Tiny markdown-to-HTML for the agent's output. Handles **bold**, line
    breaks, and bullet markers — keeps citations intact."""
    import re
    s = _escape(s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"^[\-\*]\s+(.+)$", r"• \1", s, flags=re.MULTILINE)
    s = s.replace("\n\n", "<br><br>").replace("\n", "<br>")
    return s


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="FleetMind — Mission Control",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Initialise the C-MAPSS subset selector early so the brand bar can read it
# on the first render. The actual radio widget below uses the same key.
if "subset_selector" not in st.session_state:
    st.session_state.subset_selector = "FD001"

# Subset metadata used by the brand bar HTML.
SUBSET_META = {
    "FD001": {"fault_modes": "HPC", "test_engines": 100},
    "FD003": {"fault_modes": "HPC + FAN", "test_engines": 100},
}

# ---------------------------------------------------------------------------
# CSS — mission-control / Tesla cockpit
# ---------------------------------------------------------------------------

st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">

<style>
:root {
  --bg:        #050505;
  --bg-1:      #0a0a0a;
  --bg-2:      #0f0f12;
  --line:      rgba(255,255,255,0.06);
  --line-2:    rgba(255,255,255,0.10);
  --slate:     #6b7280;
  --slate-2:   #9ca3af;
  --slate-dim: #4b5563;
  --text:      #e5e7eb;
  --text-hi:   #ffffff;
  --accent:        #00E5FF;
  --accent-soft:   rgba(0,229,255,0.16);
  --accent-glow:   rgba(0,229,255,0.45);
  --critical:      #E81922;
  --critical-soft: rgba(232,25,34,0.14);

  --mono: 'JetBrains Mono', 'IBM Plex Mono', ui-monospace, Menlo, monospace;
  --sans: 'Inter', 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
}

html, body, [class*="st-"], .stApp, .main, .block-container {
  background-color: var(--bg) !important;
  color: var(--text) !important;
  font-family: var(--sans);
  font-feature-settings: "ss01", "cv11";
  letter-spacing: -0.005em;
}
.main .block-container { padding: 1.2rem 2rem 4rem 2rem; max-width: 1480px; }

/* ----- typography ------------------------------------------------------- */
h1, h2, h3, h4, h5 {
  color: var(--text-hi);
  font-family: var(--sans);
  font-weight: 500;
  letter-spacing: -0.015em;
}
.tracking-wide { letter-spacing: 0.18em; text-transform: uppercase;
                 font-size: 0.68rem; color: var(--slate); font-weight: 500; }
.mono { font-family: var(--mono); font-variant-numeric: tabular-nums; }

/* ----- header / brand bar ---------------------------------------------- */
.brand-bar {
  display: flex; align-items: center; justify-content: space-between;
  padding: 0.4rem 0 1.0rem 0; margin-bottom: 1.2rem;
  border-bottom: 1px solid var(--line);
}
.brand-left { display: flex; align-items: center; gap: 0.9rem; }
.brand-mark {
  width: 24px; height: 24px; border-radius: 50%;
  background: radial-gradient(circle at 35% 35%, var(--accent), transparent 65%);
  box-shadow: 0 0 14px var(--accent-glow), inset 0 0 6px rgba(0,229,255,0.4);
}
.brand-name {
  font-family: var(--sans); font-weight: 600; font-size: 1.1rem;
  letter-spacing: 0.32em; color: var(--text-hi);
}
.brand-tag {
  font-family: var(--mono); font-size: 0.72rem;
  color: var(--slate); letter-spacing: 0.06em;
}
.brand-right { display: flex; gap: 1.6rem; align-items: center; }
.brand-stat .k { font-family: var(--mono); font-size: 0.65rem;
                 color: var(--slate-dim); letter-spacing: 0.16em; }
.brand-stat .v { font-family: var(--mono); font-size: 0.92rem;
                 color: var(--text-hi); letter-spacing: 0.04em; }
.live-dot {
  display: inline-block; width: 6px; height: 6px; border-radius: 50%;
  background: var(--accent); margin-right: 6px;
  box-shadow: 0 0 6px var(--accent-glow);
  animation: live 1.6s ease-in-out infinite;
}
@keyframes live { 0%,100% { opacity: 1; } 50% { opacity: 0.35; } }

/* ----- KPI strip -------------------------------------------------------- */
.kpi-strip { display: grid; grid-template-columns: repeat(6, 1fr); gap: 1px;
             background: var(--line); border: 1px solid var(--line);
             border-radius: 4px; overflow: hidden; margin-bottom: 1.2rem; }
.kpi {
  background: var(--bg-1); padding: 0.85rem 1.05rem;
  display: flex; flex-direction: column; gap: 0.35rem;
}
.kpi-label { font-family: var(--mono); font-size: 0.62rem;
             color: var(--slate); letter-spacing: 0.18em;
             text-transform: uppercase; }
.kpi-value { font-family: var(--mono); font-size: 1.55rem;
             color: var(--text-hi); font-weight: 500;
             font-variant-numeric: tabular-nums; }
.kpi-suffix { font-family: var(--mono); font-size: 0.7rem;
              color: var(--slate); margin-left: 0.35rem; letter-spacing: 0.04em; }
.kpi-accent .kpi-value { color: var(--accent); }
.kpi-critical .kpi-value { color: var(--critical); }

/* ----- tabs ------------------------------------------------------------ */
.stTabs [data-baseweb="tab-list"] {
  gap: 0; border-bottom: 1px solid var(--line); margin-bottom: 1.2rem;
}
.stTabs [data-baseweb="tab"] {
  background: transparent !important; border-radius: 0;
  padding: 0.75rem 1.4rem 0.75rem 0; margin-right: 1.4rem;
  color: var(--slate) !important; font-family: var(--sans);
  font-size: 0.78rem; letter-spacing: 0.22em; text-transform: uppercase;
  font-weight: 500;
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
  color: var(--text-hi) !important;
  border-bottom: 1px solid var(--accent);
}
.stTabs [data-baseweb="tab-highlight"] { background: var(--accent) !important; }

/* ----- status grid ----------------------------------------------------- */
.status-grid {
  display: grid;
  grid-template-columns: repeat(10, 1fr);
  gap: 1px;
  background: var(--line);
  border: 1px solid var(--line);
  border-radius: 4px;
  padding: 1px;
}
.status-cell {
  background: var(--bg-1);
  padding: 0.55rem 0.6rem 0.5rem 0.6rem;
  display: flex; flex-direction: column; gap: 0.35rem;
  position: relative; min-height: 88px;
  cursor: default;
  transition: background 120ms ease, box-shadow 120ms ease;
}
.status-cell:hover { background: var(--bg-2); }
.status-cell::before {
  content: ""; position: absolute; left: 0; top: 0; bottom: 0;
  width: 2px; background: var(--accent);
}
.status-cell.s-REPLACE::before { background: var(--critical); }
.status-cell.s-REPAIR::before  { background: #ff8c42; }
.status-cell.s-INSPECT::before { background: #ffd166; }
.status-cell.s-MONITOR::before { background: var(--accent); }

.cell-row { display: flex; justify-content: space-between; align-items: baseline; }
.cell-id { font-family: var(--mono); font-size: 0.66rem;
           color: var(--slate); letter-spacing: 0.10em; }
.cell-rul { font-family: var(--mono); font-size: 1.0rem;
            color: var(--text-hi); font-variant-numeric: tabular-nums; }
.cell-action { font-family: var(--mono); font-size: 0.56rem;
               color: var(--slate-2); letter-spacing: 0.18em; }
.status-cell.s-REPLACE .cell-action { color: var(--critical); }
.status-cell.s-REPLACE .cell-rul { color: var(--critical); }
.cell-spark { line-height: 0; opacity: 0.85; }

/* ----- panels (engine detail) ----------------------------------------- */
.panel {
  background: var(--bg-1); border: 1px solid var(--line);
  border-radius: 4px; padding: 1.2rem 1.3rem;
}
.panel-title { font-family: var(--mono); font-size: 0.62rem;
               color: var(--slate); letter-spacing: 0.22em;
               text-transform: uppercase; margin-bottom: 0.8rem; }

.stage-row { display: grid; grid-template-columns: 14px 88px 1fr 60px;
             align-items: center; gap: 0.7rem; padding: 0.4rem 0;
             border-bottom: 1px solid var(--line); }
.stage-row:last-child { border-bottom: none; }
.stage-dot { width: 8px; height: 8px; border-radius: 50%; }
.stage-name { font-family: var(--sans); font-size: 0.85rem;
              color: var(--text); font-weight: 500; }
.stage-bar { height: 3px; background: var(--line-2); border-radius: 2px;
             overflow: hidden; }
.stage-bar-fill { height: 100%; background: var(--accent); }
.stage-score { font-family: var(--mono); font-size: 0.78rem;
               color: var(--slate-2); text-align: right;
               font-variant-numeric: tabular-nums; }

/* ----- 3D viewport (transparent) -------------------------------------- */
.viewport-3d {
  position: relative; background: radial-gradient(
    ellipse at center, rgba(0,229,255,0.04) 0%, rgba(0,0,0,0) 70%);
  border: 1px solid var(--line); border-radius: 4px;
  overflow: hidden;
}

/* ----- action pills --------------------------------------------------- */
.action-pill { display: inline-block; padding: 0.22rem 0.7rem;
               border-radius: 2px; font-family: var(--mono);
               font-size: 0.68rem; letter-spacing: 0.18em;
               text-transform: uppercase; font-weight: 500;
               border: 1px solid; }
.action-MONITOR { color: var(--accent); border-color: var(--accent-soft);
                  background: rgba(0,229,255,0.05); }
.action-INSPECT { color: #ffd166; border-color: rgba(255,209,102,0.2);
                  background: rgba(255,209,102,0.05); }
.action-REPAIR  { color: #ff8c42; border-color: rgba(255,140,66,0.2);
                  background: rgba(255,140,66,0.05); }
.action-REPLACE { color: var(--critical); border-color: var(--critical-soft);
                  background: rgba(232,25,34,0.05); }

/* ----- COPILOT — neural overlay -------------------------------------- */
.copilot-shell {
  position: relative;
  background:
    linear-gradient(160deg, rgba(15,15,18,0.7), rgba(5,5,5,0.4));
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 6px;
  backdrop-filter: blur(20px) saturate(120%);
  -webkit-backdrop-filter: blur(20px) saturate(120%);
  box-shadow:
    inset 0 1px 0 rgba(255,255,255,0.04),
    0 30px 60px rgba(0,0,0,0.4),
    0 0 60px rgba(0,229,255,0.06);
  padding: 1.6rem 1.6rem 1.4rem 1.6rem;
  overflow: hidden;
}

/* Neural Pulse — breathing orb */
.neural-pulse {
  display: flex; align-items: center; gap: 0.9rem; margin-bottom: 1.2rem;
}
.neural-orb {
  position: relative; width: 18px; height: 18px; border-radius: 50%;
  background: radial-gradient(circle at 35% 35%,
    rgba(255,255,255,0.95), rgba(0,229,255,0.6) 50%, rgba(0,229,255,0.0) 75%);
  box-shadow: 0 0 18px rgba(0,229,255,0.55), 0 0 4px rgba(255,255,255,0.6);
  animation: breathe 2.6s ease-in-out infinite;
}
.neural-orb::after {
  content: ""; position: absolute; inset: -10px; border-radius: 50%;
  border: 1px solid rgba(0,229,255,0.18);
  animation: ring 2.6s ease-out infinite;
}
@keyframes breathe {
  0%,100% { transform: scale(1);   opacity: 0.85; }
  50%     { transform: scale(1.15); opacity: 1.0;  }
}
@keyframes ring {
  0%   { transform: scale(0.6); opacity: 0.7; }
  100% { transform: scale(1.6); opacity: 0;   }
}
.neural-label-l { font-family: var(--sans); font-size: 0.7rem;
                  color: var(--slate); letter-spacing: 0.32em;
                  text-transform: uppercase; font-weight: 500; }
.neural-label-r { font-family: var(--mono); font-size: 0.72rem;
                  color: var(--accent); letter-spacing: 0.12em;
                  margin-left: auto; }

/* fade-to-black scroll mask + custom scrollbars */
.copilot-history {
  max-height: 460px; overflow-y: auto; padding: 0.2rem 0.4rem 1.0rem 0.2rem;
  -webkit-mask-image: linear-gradient(to bottom,
    transparent 0, #000 36px, #000 calc(100% - 36px), transparent 100%);
          mask-image: linear-gradient(to bottom,
    transparent 0, #000 36px, #000 calc(100% - 36px), transparent 100%);
  scrollbar-width: thin; scrollbar-color: rgba(255,255,255,0.08) transparent;
}
.copilot-history::-webkit-scrollbar { width: 4px; }
.copilot-history::-webkit-scrollbar-thumb {
  background: rgba(255,255,255,0.08); border-radius: 2px; }
.copilot-history::-webkit-scrollbar-track { background: transparent; }

/* messages */
.msg { padding: 0.55rem 0; }
.msg-user {
  font-family: var(--sans); font-size: 0.92rem; font-weight: 400;
  color: var(--slate-2); letter-spacing: -0.005em; line-height: 1.55;
}
.msg-user::before { content: "› "; color: var(--slate-dim); }
.msg-assistant {
  font-family: var(--sans); font-size: 0.96rem; font-weight: 400;
  color: var(--text-hi); letter-spacing: -0.01em; line-height: 1.6;
  padding-left: 1.1rem; border-left: 1px solid rgba(0,229,255,0.18);
}
.msg-assistant strong { font-weight: 600; color: var(--text-hi); }

/* citations */
.citations { margin-top: 0.6rem; display: flex; flex-wrap: wrap;
             gap: 0.4rem; padding-left: 1.1rem; }
.citation {
  display: inline-flex; align-items: center; gap: 0.4rem;
  padding: 0.18rem 0.55rem; border: 1px solid rgba(0,229,255,0.22);
  border-radius: 2px; font-family: var(--mono); font-size: 0.66rem;
  color: var(--accent); letter-spacing: 0.06em; background: rgba(0,229,255,0.04);
}
.citation .ttl { color: var(--slate-2); }

/* smart telemetry cards inside chat */
.telemetry-row { display: flex; gap: 0.5rem; margin: 0.7rem 0 0.2rem 1.1rem;
                 flex-wrap: wrap; }
.telemetry-card {
  background: rgba(255,255,255,0.025); border: 1px solid var(--line-2);
  border-left-width: 2px; border-radius: 3px;
  padding: 0.5rem 0.7rem; min-width: 150px;
  display: flex; flex-direction: column; gap: 0.3rem;
  font-family: var(--mono);
  backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px);
}
.telemetry-card-row { display: flex; justify-content: space-between;
                      align-items: baseline; }
.telemetry-card-id { font-size: 0.62rem; color: var(--slate);
                     letter-spacing: 0.18em; }
.telemetry-card-action { font-size: 0.62rem; letter-spacing: 0.18em;
                         font-weight: 600; }
.telemetry-card-meta { font-size: 0.62rem; color: var(--slate);
                       letter-spacing: 0.10em; }
.telemetry-card-rul { font-size: 0.86rem; color: var(--text-hi);
                      letter-spacing: 0.04em; }
.telemetry-card-spark { line-height: 0; }

/* tool-call detail */
.tool-call {
  margin: 0.6rem 0 0.2rem 1.1rem; font-family: var(--mono);
  font-size: 0.7rem; color: var(--slate);
}
details.tool-call summary { cursor: pointer; letter-spacing: 0.14em;
                             text-transform: uppercase; padding: 0.2rem 0; }
details.tool-call[open] summary { color: var(--accent); }
details.tool-call pre {
  background: rgba(0,0,0,0.4); border: 1px solid var(--line);
  border-radius: 2px; padding: 0.6rem; color: var(--slate-2);
  font-size: 0.68rem; overflow-x: auto;
}

/* suggestion chips */
.suggestion-row { display: flex; flex-wrap: wrap; gap: 0.45rem;
                  margin-bottom: 1.0rem; }
div.stButton > button {
  background: rgba(255,255,255,0.025) !important;
  color: var(--slate-2) !important;
  border: 1px solid var(--line-2) !important;
  border-radius: 2px !important;
  font-family: var(--mono) !important;
  font-size: 0.7rem !important;
  letter-spacing: 0.06em !important;
  padding: 0.35rem 0.8rem !important;
  font-weight: 400 !important;
  transition: border-color 120ms ease, color 120ms ease;
}
div.stButton > button:hover {
  border-color: var(--accent) !important;
  color: var(--accent) !important;
  background: rgba(0,229,255,0.04) !important;
}

/* control row — subset selector + PDF download as one cockpit band */
.control-row { /* visual anchor; the actual columns are Streamlit-managed */ }
.ctrl-label {
  font-family: var(--mono); font-size: 0.62rem; color: var(--slate);
  letter-spacing: 0.22em; text-transform: uppercase;
  margin-bottom: 0.45rem;
}

/* radio selector → cockpit pill toggle */
[data-testid="stRadio"] [role="radiogroup"] {
  gap: 6px !important;
  padding: 0 !important;
}
[data-testid="stRadio"] label {
  background: rgba(255,255,255,0.025) !important;
  border: 1px solid var(--line-2) !important;
  border-radius: 2px !important;
  padding: 0.45rem 1.1rem !important;
  margin: 0 !important;
  cursor: pointer;
  transition: border-color 120ms ease, background 120ms ease, color 120ms ease;
}
[data-testid="stRadio"] label:hover {
  border-color: var(--accent) !important;
  background: rgba(0,229,255,0.05) !important;
}
[data-testid="stRadio"] label[data-checked="true"],
[data-testid="stRadio"] label:has(input:checked) {
  background: rgba(0,229,255,0.10) !important;
  border-color: var(--accent) !important;
}
[data-testid="stRadio"] label > div:first-child { display: none !important; }
[data-testid="stRadio"] label p,
[data-testid="stRadio"] label div {
  font-family: var(--mono) !important;
  font-size: 0.74rem !important;
  font-weight: 500 !important;
  letter-spacing: 0.18em !important;
  text-transform: uppercase !important;
  color: var(--slate-2) !important;
  margin: 0 !important;
}
[data-testid="stRadio"] label:has(input:checked) p,
[data-testid="stRadio"] label:has(input:checked) div {
  color: var(--accent) !important;
}

/* download button — primary CTA (button sits ~4 wrappers deep when help= tooltip is set) */
[data-testid="stDownloadButton"] button {
  background: rgba(0,229,255,0.08) !important;
  color: var(--accent) !important;
  border: 1px solid rgba(0,229,255,0.35) !important;
  border-radius: 2px !important;
  font-family: var(--mono) !important;
  font-size: 0.72rem !important;
  font-weight: 500 !important;
  letter-spacing: 0.18em !important;
  padding: 0.55rem 1.0rem !important;
  text-transform: uppercase !important;
  transition: background 120ms ease, border-color 120ms ease,
              box-shadow 200ms ease, color 120ms ease;
}
[data-testid="stDownloadButton"] button:hover {
  background: rgba(0,229,255,0.16) !important;
  border-color: var(--accent) !important;
  color: #ffffff !important;
  box-shadow: 0 0 16px rgba(0,229,255,0.25);
}
[data-testid="stDownloadButton"] button:focus {
  box-shadow: 0 0 0 1px var(--accent) !important;
  outline: none !important;
}
[data-testid="stDownloadButton"] button p,
[data-testid="stDownloadButton"] button div {
  color: inherit !important;
  font-family: inherit !important;
  letter-spacing: inherit !important;
}

/* chat input — make it look like a HUD prompt */
[data-testid="stChatInput"] {
  background: rgba(255,255,255,0.02) !important;
  border: 1px solid var(--line-2) !important;
  border-radius: 3px !important;
  margin-top: 0.6rem;
}
[data-testid="stChatInput"] textarea {
  background: transparent !important; color: var(--text-hi) !important;
  font-family: var(--sans) !important;
}

/* selectbox styling */
[data-baseweb="select"] > div {
  background: var(--bg-1) !important;
  border: 1px solid var(--line-2) !important;
  border-radius: 3px !important;
  font-family: var(--mono) !important; font-size: 0.85rem !important;
}

/* hide deploy / hamburger / footer for cleaner cockpit feel */
header[data-testid="stHeader"] { display: none; }
footer { display: none; }
#MainMenu { display: none; }

/* misc */
hr { border-color: var(--line); margin: 1.2rem 0; }
.muted { color: var(--slate); font-family: var(--mono); font-size: 0.78rem; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Header / brand bar
# ---------------------------------------------------------------------------

_active_subset = st.session_state.get("subset_selector", "FD001")
_active_meta = SUBSET_META[_active_subset]
st.markdown(f"""
<div class="brand-bar">
  <div class="brand-left">
    <div class="brand-mark"></div>
    <div>
      <div class="brand-name">FLEETMIND</div>
      <div class="brand-tag">INTELLIGENT PREDICTIVE MAINTENANCE</div>
    </div>
  </div>
  <div class="brand-right">
    <div class="brand-stat">
      <div class="k">DATASET</div>
      <div class="v">C-MAPSS · {_active_subset}</div>
    </div>
    <div class="brand-stat">
      <div class="k">FAULT MODES</div>
      <div class="v">{_active_meta['fault_modes']}</div>
    </div>
    <div class="brand-stat">
      <div class="k">STATUS</div>
      <div class="v"><span class="live-dot"></span>NOMINAL</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# KPI strip
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Subset selector (FD001 | FD003) + Executive PDF download
# ---------------------------------------------------------------------------
# The selector sits on the same row as the PDF button so the dashboard has
# one clear "control row" between the brand bar and the KPI strip.

SUPPORTED_SUBSETS = ["FD001", "FD003"]


def _metrics_for(subset: str) -> dict:
    """Read the per-subset metrics file with a legacy fallback for FD001."""
    p_per = Path(f"reports/metrics_{subset}.json")
    if p_per.exists():
        return json.loads(p_per.read_text())
    if subset == "FD001":
        return json.loads(Path("reports/metrics.json").read_text())
    raise FileNotFoundError(f"no metrics file for {subset}")


@st.cache_data
def _kpi_data(subset: str) -> dict:
    fleet = predict_fleet_rul(subset)
    counts = fleet["action"].value_counts().to_dict()
    metrics = _metrics_for(subset)
    retrieval = {}
    for p in ["reports/retrieval_metrics_openai.json", "reports/retrieval_metrics.json"]:
        if Path(p).exists():
            retrieval = json.loads(Path(p).read_text())
            break
    return {
        "subset": subset,
        "n_engines": len(fleet),
        "avg_rul": float(fleet["predicted_rul"].mean()),
        "min_rul": float(fleet["predicted_rul"].min()),
        "n_replace": int(counts.get("REPLACE", 0)),
        "n_repair": int(counts.get("REPAIR", 0)),
        "n_inspect": int(counts.get("INSPECT", 0)),
        "n_monitor": int(counts.get("MONITOR", 0)),
        "test_rmse": float(metrics["lstm"]["rmse"]),
        "test_score": float(metrics["lstm"]["cmapss_score"]),
        "n_features": int(metrics.get("n_features", 14)),
        "hit_at_1": float(retrieval.get("hit@1", 0)),
        "hit_at_3": float(retrieval.get("hit@3", 0)),
        "retrieval_backend": retrieval.get("backend", "n/a"),
    }


@st.cache_data(show_spinner=False)
def _report_bytes(subset: str) -> bytes:
    """Build the executive PDF lazily for the given subset; cached per subset."""
    fleet = predict_fleet_rul(subset)
    metrics = _metrics_for(subset)
    retrieval = {}
    for p in ["reports/retrieval_metrics_openai.json", "reports/retrieval_metrics.json"]:
        if Path(p).exists():
            retrieval = json.loads(Path(p).read_text())
            break
    return build_executive_pdf(ReportInputs(
        fleet=fleet, metrics=metrics, retrieval=retrieval, subset=subset,
    ))


from datetime import datetime as _dt  # noqa: E402

# Wrap the selector + PDF row in a styled container so it looks like
# part of the cockpit chrome rather than a stray Streamlit widget.
st.markdown('<div class="control-row">', unsafe_allow_html=True)
ctrl_l, ctrl_r = st.columns([4, 1.2])
with ctrl_l:
    st.markdown('<div class="ctrl-label">C-MAPSS SUBSET</div>',
                unsafe_allow_html=True)
    SUBSET = st.radio(
        "C-MAPSS subset",
        options=SUPPORTED_SUBSETS,
        index=0,
        horizontal=True,
        label_visibility="collapsed",
        key="subset_selector",
    )
with ctrl_r:
    _pdf_filename = (f"fleetmind-{SUBSET.lower()}-executive-report-"
                     f"{_dt.utcnow().strftime('%Y%m%d')}.pdf")
    st.download_button(
        label="↓  EXECUTIVE PDF",
        data=_report_bytes(SUBSET),
        file_name=_pdf_filename,
        mime="application/pdf",
        use_container_width=True,
        help="Download the executive report for the active subset: "
             "risk summary, maintenance schedule, and cost impact projection.",
        key="dl-pdf",
    )
st.markdown('</div>', unsafe_allow_html=True)


kpi = _kpi_data(SUBSET)


def _kpi_html(label: str, value: str, suffix: str = "",
              cls: str = "") -> str:
    return (
        f'<div class="kpi {cls}">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}<span class="kpi-suffix">{suffix}</span></div>'
        f'</div>'
    )


st.markdown(
    '<div class="kpi-strip">' +
    _kpi_html("Fleet Size",      f"{kpi['n_engines']}",        "engines") +
    _kpi_html("Avg RUL",         f"{kpi['avg_rul']:.1f}",      "cycles") +
    _kpi_html("Replace",         f"{kpi['n_replace']:02d}",    "now",
              cls="kpi-critical" if kpi['n_replace'] else "") +
    _kpi_html("Repair",          f"{kpi['n_repair']:02d}",     "soon") +
    _kpi_html("LSTM RMSE",       f"{kpi['test_rmse']:.2f}",    "cycles",
              cls="kpi-accent") +
    _kpi_html("Retrieval Hit@1", f"{kpi['hit_at_1']:.2f}",
              kpi['retrieval_backend'], cls="kpi-accent") +
    '</div>',
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_fleet, tab_engine, tab_copilot = st.tabs([
    "Fleet Status", "Engine Detail", "Neural Copilot",
])


# ---------------------------------------------------------------------------
# FLEET STATUS — high-density status grid w/ micro-sparklines
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Building fleet status grid…")
def _build_status_cells(subset: str) -> str:
    """Render a 10×10 status grid as a single HTML string (one paint)."""
    fleet = predict_fleet_rul(subset)
    backend = get_engine_backend(subset)
    cells: list[str] = []
    for _, row in fleet.iterrows():
        eid = int(row["engine_id"])
        rul = float(row["predicted_rul"])
        act = row["action"]
        hist = backend.test_df[backend.test_df["engine_id"] == eid]
        # use sensor_4 (HPT exit temp) — monotonic-ish degradation signal
        signal = degradation_series(hist["sensor_4"].values, n_points=30)
        spark_color = {
            "REPLACE": "#E81922",
            "REPAIR":  "#ff8c42",
            "INSPECT": "#ffd166",
            "MONITOR": "#00E5FF",
        }[act]
        spark = sparkline_svg(signal, width=88, height=20,
                              stroke=spark_color, stroke_width=1.1)
        cells.append(
            f'<div class="status-cell s-{act}">'
            f'  <div class="cell-row">'
            f'    <span class="cell-id">E-{eid:03d}</span>'
            f'    <span class="cell-rul">{rul:.0f}</span>'
            f'  </div>'
            f'  <div class="cell-spark">{spark}</div>'
            f'  <div class="cell-action">{act}</div>'
            f'</div>'
        )
    return f'<div class="status-grid">{"".join(cells)}</div>'


with tab_fleet:
    left, right = st.columns([3, 1.3])
    with left:
        st.markdown('<div class="tracking-wide">FLEET TELEMETRY · 100 ENGINES · LIVE</div>',
                    unsafe_allow_html=True)
        st.markdown(_build_status_cells(SUBSET), unsafe_allow_html=True)

    with right:
        fleet = predict_fleet_rul(SUBSET)
        st.markdown('<div class="tracking-wide">ACTION DISTRIBUTION</div>',
                    unsafe_allow_html=True)
        counts = fleet["action"].value_counts().reindex(
            ["REPLACE", "REPAIR", "INSPECT", "MONITOR"], fill_value=0)
        bar = go.Figure(go.Bar(
            x=counts.values, y=counts.index, orientation="h",
            marker=dict(color=["#E81922", "#ff8c42", "#ffd166", "#00E5FF"],
                        line=dict(width=0)),
            text=[f"{v:02d}" for v in counts.values],
            textposition="outside",
            textfont=dict(color=TEXT, family="JetBrains Mono", size=11),
            hoverinfo="skip",
        ))
        bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False,
                       range=[0, max(counts.values) * 1.3]),
            yaxis=dict(color=TEXT_DIM, gridcolor="rgba(0,0,0,0)",
                       zeroline=False, tickfont=dict(family="JetBrains Mono", size=10)),
            margin=dict(l=10, r=20, t=4, b=4), height=180,
            bargap=0.55,
        )
        st.plotly_chart(bar, use_container_width=True)

        st.markdown('<div class="tracking-wide" style="margin-top:1.2rem">'
                    'PRIORITY QUEUE</div>', unsafe_allow_html=True)
        priority = fleet[fleet["action"].isin(["REPLACE", "REPAIR"])]\
            .sort_values("predicted_rul").head(8)
        if len(priority) == 0:
            st.markdown('<p class="muted">No engines flagged.</p>',
                        unsafe_allow_html=True)
        else:
            rows = []
            for _, r in priority.iterrows():
                color = "#E81922" if r["action"] == "REPLACE" else "#ff8c42"
                rows.append(
                    f'<tr><td class="mono" style="color:var(--slate)">E-{int(r["engine_id"]):03d}</td>'
                    f'<td class="mono" style="color:var(--text-hi); text-align:right">{r["predicted_rul"]:.1f}</td>'
                    f'<td class="mono" style="color:{color}; text-align:right; '
                    f'font-size:0.66rem; letter-spacing:0.18em">{r["action"]}</td></tr>'
                )
            st.markdown(
                '<table style="width:100%; border-collapse:collapse; font-size:0.78rem">'
                + "".join(rows)
                + '</table>',
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# ENGINE DETAIL — transparent 3D w/ HUD + per-stage health
# ---------------------------------------------------------------------------

with tab_engine:
    backend = get_engine_backend(SUBSET)
    eids_all = backend.engine_ids()
    default_idx = eids_all.index(17) if 17 in eids_all else 0

    sel_col, _ = st.columns([1, 5])
    with sel_col:
        engine_id = st.selectbox(
            "ENGINE", eids_all, index=default_idx,
            format_func=lambda i: f"E-{int(i):03d}",
            label_visibility="visible",
        )

    history = backend.test_df[backend.test_df["engine_id"] == engine_id]\
        .sort_values("cycle")
    healths = stage_healths(history)
    tool_out = query_engine_history(int(engine_id), backend)
    pred = tool_out["lstm_prediction"]
    rul = pred["rul_cycles"]
    action = pred["recommended_action"]
    n_cycles = int(history["cycle"].max())
    # degradation index = mean stage z-score across non-Nozzle stages
    deg_idx = float(np.mean([h.score for h in healths]))

    left, right = st.columns([3, 2])
    with left:
        st.markdown('<div class="tracking-wide">3D AIRFRAME · PER-STAGE DEGRADATION</div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="viewport-3d">', unsafe_allow_html=True)
        st.plotly_chart(
            build_engine_figure(
                healths,
                hud=dict(engine_id=engine_id, cycle_count=n_cycles,
                         degradation_index=deg_idx, status=action),
            ),
            use_container_width=True, theme=None,
            config=dict(displayModeBar=False),
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">RECOMMENDED ACTION</div>',
                    unsafe_allow_html=True)
        st.markdown(
            f'<div style="display:flex; align-items:baseline; gap:0.9rem; margin-bottom:0.4rem">'
            f'  <span class="action-pill action-{action}">{action}</span>'
            f'  <span class="mono" style="font-size:1.4rem; color:var(--text-hi)">'
            f'    {rul:.1f}<span style="font-size:0.7rem; color:var(--slate); margin-left:0.3rem">cycles</span>'
            f'  </span>'
            f'</div>'
            f'<p class="muted">{pred["action_rationale"]}</p>',
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="panel" style="margin-top:0.8rem">',
                    unsafe_allow_html=True)
        st.markdown('<div class="panel-title">HEALTH BY STAGE · z-score</div>',
                    unsafe_allow_html=True)
        max_score = max((abs(h.score) for h in healths), default=1.0)
        for h in healths:
            pct = min(100, abs(h.score) / max(max_score, 1e-6) * 100)
            st.markdown(
                f'<div class="stage-row">'
                f'  <span class="stage-dot" style="background:{h.color}"></span>'
                f'  <span class="stage-name">{h.name}</span>'
                f'  <div class="stage-bar"><div class="stage-bar-fill" '
                f'       style="width:{pct:.0f}%; background:{h.color}"></div></div>'
                f'  <span class="stage-score">{h.score:+.2f}σ</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

    # sensor timelines
    st.markdown('<div class="tracking-wide" style="margin-top:1.2rem">'
                'SENSOR TIMELINES · HPC SECTION</div>',
                unsafe_allow_html=True)
    cols = ["sensor_3", "sensor_7", "sensor_11", "sensor_4", "sensor_20", "sensor_21"]
    fig = go.Figure()
    palette = ["#00E5FF", "#7dd3fc", "#a78bfa", "#ffd166", "#ff8c42", "#E81922"]
    for c, color in zip(cols, palette):
        if c in history.columns:
            fig.add_trace(go.Scatter(
                x=history["cycle"], y=history[c],
                mode="lines", name=c.replace("sensor_", "S"),
                line=dict(color=color, width=1.2),
                hovertemplate=f"{c} · cycle %{{x}} · %{{y:.2f}}<extra></extra>",
            ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="cycle", color=TEXT_DIM, gridcolor="rgba(255,255,255,0.04)",
                   zeroline=False, tickfont=dict(family="JetBrains Mono", size=10)),
        yaxis=dict(color=TEXT_DIM, gridcolor="rgba(255,255,255,0.04)",
                   zeroline=False, tickfont=dict(family="JetBrains Mono", size=10)),
        legend=dict(font=dict(color=TEXT_DIM, family="JetBrains Mono", size=10),
                    orientation="h", y=1.12, bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=10, r=10, t=10, b=10), height=300,
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# COPILOT — glassmorphism neural overlay
# ---------------------------------------------------------------------------

with tab_copilot:
    api_set = bool(os.getenv("OPENAI_API_KEY"))
    backend = get_engine_backend(SUBSET)
    agent = get_agent(SUBSET)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.markdown('<div class="copilot-shell">', unsafe_allow_html=True)
    st.markdown(f"""
      <div class="neural-pulse">
        <div class="neural-orb"></div>
        <div class="neural-label-l">NEURAL COPILOT</div>
        <div class="neural-label-r">
          {"● ONLINE · gpt-4o-mini" if api_set else "◌ OFFLINE · deterministic mode"}
        </div>
      </div>
    """, unsafe_allow_html=True)

    if not api_set:
        st.markdown(
            '<p class="muted" style="margin-bottom:0.8rem">'
            'OPENAI_API_KEY not set — agent runs against deterministic mock; '
            'tool calls and citations are still real.</p>',
            unsafe_allow_html=True,
        )

    suggestions = [
        "What is the C-MAPSS dataset?",
        "Why is RUL clipped at 125 cycles?",
        "What's the current RUL for engine 17 and what should we do?",
        "Should we replace engine 100 immediately?",
        "Which sensors react to HPC degradation?",
    ]
    st.markdown('<div class="suggestion-row">', unsafe_allow_html=True)
    sc = st.columns(len(suggestions))
    for s, c in zip(suggestions, sc):
        if c.button(s, use_container_width=True, key=f"sug-{hash(s)}"):
            st.session_state.pending_q = s
    st.markdown('</div>', unsafe_allow_html=True)

    # render chat history (we render markup ourselves, not st.chat_message,
    # so we get the fade-to-black mask + smart widgets in-line)
    history_html: list[str] = []
    for entry in st.session_state.chat_history:
        if entry["role"] == "user":
            history_html.append(
                f'<div class="msg msg-user">{_escape(entry["content"])}</div>'
            )
        else:
            history_html.append(
                f'<div class="msg msg-assistant">{_md_to_html(entry["content"])}</div>'
            )
            if entry.get("widgets_html"):
                history_html.append(entry["widgets_html"])
            if entry.get("citations"):
                cites = "".join(
                    f'<span class="citation"><span>[{c["id"]}]</span>'
                    f'<span class="ttl">{c["title"]}</span></span>'
                    for c in entry["citations"]
                )
                history_html.append(f'<div class="citations">{cites}</div>')
            if entry.get("tool_calls"):
                tc = entry["tool_calls"][0]
                payload = json.dumps(tc, indent=2, default=str)
                history_html.append(
                    f'<details class="tool-call">'
                    f'<summary>▸ tool call · {tc["name"]}</summary>'
                    f'<pre>{_escape(payload)}</pre></details>'
                )

    if history_html:
        st.markdown(
            '<div class="copilot-history">' + "".join(history_html) + '</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="copilot-history"><p class="muted" '
            'style="padding:1rem 0">Awaiting first transmission. '
            'Try a suggestion above, or ask anything about the fleet, '
            'an engine, or the model.</p></div>',
            unsafe_allow_html=True,
        )

    st.markdown('</div>', unsafe_allow_html=True)  # /copilot-shell

    user_q = st.chat_input("Transmit query…")
    if "pending_q" in st.session_state and st.session_state.pending_q:
        user_q = st.session_state.pending_q
        st.session_state.pending_q = ""

    if user_q:
        st.session_state.chat_history.append({"role": "user", "content": user_q})
        with st.spinner(""):  # blank spinner — neural pulse covers it visually
            result = agent.chat(user_q)
        widgets_html = cards_for_answer(result.answer, backend)
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": result.answer,
            "citations": result.citations,
            "tool_calls": result.tool_calls,
            "widgets_html": widgets_html,
        })
        st.rerun()




