from __future__ import annotations

from pathlib import Path
import io
import os
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import tensorflow as tf


# ---------------------------
# Plotly styling + safe rendering (unique keys)
# ---------------------------
def style_plotly(fig, title: str | None = None):
    fig.update_layout(
        title=title,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E9EEF6", size=13),
        margin=dict(l=10, r=10, t=60, b=10),
        legend=dict(font=dict(color="#E9EEF6")),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.08)", zerolinecolor="rgba(255,255,255,0.08)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.08)", zerolinecolor="rgba(255,255,255,0.08)")
    return fig


def show_plot(fig, key: str, title: str | None = None):
    """Render plotly chart with unique key to avoid DuplicateElementId crashes."""
    fig = style_plotly(fig, title=title)
    st.plotly_chart(fig, use_container_width=True, key=key)


# ---------------------------
# Page config
# ---------------------------
st.set_page_config(page_title="Fleet Reliability Dashboard", page_icon="⚡", layout="wide")

# -------------------- DEBUG (collapsed) --------------------
SHOW_DEBUG = st.sidebar.toggle("Show Debug", value=False)
if SHOW_DEBUG:
    with st.expander("🔧 Debug (dataset check)", expanded=False):
        st.write("Working dir:", os.getcwd())


# src/app/dashboard.py -> parents[2] == project root
BASE_DIR = Path(__file__).resolve().parents[2]
VAL_PATH = BASE_DIR / "data" / "processed" / "val_predictions_decisions.csv"
MODEL_PATH = BASE_DIR / "models" / "rf_rul_model.joblib"
TRAIN_PATH = BASE_DIR / "data" / "raw" / "train_FD001.txt"

LSTM_MODEL_PATH = BASE_DIR / "models" / "lstm_rul.keras"
LSTM_META_PATH = BASE_DIR / "models" / "lstm_meta.joblib"

lstm_model = None
lstm_meta = None
if LSTM_MODEL_PATH.exists() and LSTM_META_PATH.exists():
    lstm_model = tf.keras.models.load_model(LSTM_MODEL_PATH)
    lstm_meta = joblib.load(LSTM_META_PATH)

model_choice = st.sidebar.selectbox(
    "Prediction model",
    ["RandomForest (Baseline)", "LSTM (Sequence)"]
)

# ---------------------------
# Tesla Premium Styling (Readable + Premium)
# ---------------------------
st.markdown(
    """
<style>
/* -----------------------------
   PREMIUM TESLA-STYLE THEME
------------------------------ */

/* Base app background: premium but readable */
.stApp {
  background:
    radial-gradient(1200px 800px at 18% 8%, rgba(255, 60, 60, 0.16), transparent 60%),
    radial-gradient(950px 650px at 82% 22%, rgba(30, 190, 255, 0.16), transparent 55%),
    radial-gradient(900px 650px at 45% 90%, rgba(0, 255, 180, 0.12), transparent 60%),
    linear-gradient(180deg, #0B1220 0%, #070B12 45%, #05070B 100%);
  color: #F2F6FF;
}

/* Typography */
html, body, [class*="css"] {
  font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
  color: #F2F6FF !important;
}

/* Headings */
h1, h2, h3, h4 {
  color: #FFFFFF !important;
  letter-spacing: 0.2px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
  background: rgba(10, 14, 22, 0.88) !important;
  border-right: 1px solid rgba(255,255,255,0.10);
  box-shadow: 20px 0 60px rgba(0,0,0,0.35);
}

/* Selects */
div[data-baseweb="select"] > div {
  background: rgba(255,255,255,0.06) !important;
  border: 1px solid rgba(255,255,255,0.10) !important;
  border-radius: 14px !important;
}

/* Tabs */
button[data-baseweb="tab"] {
  color: rgba(242,246,255,0.75) !important;
  font-weight: 800;
  border-radius: 12px !important;
  padding: 10px 14px !important;
  background: transparent !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
  color: #FFFFFF !important;
  background: rgba(255,255,255,0.06) !important;
  border: 1px solid rgba(255,255,255,0.10) !important;
}

/* Container spacing */
.block-container {
  padding-top: 1.1rem;
  padding-bottom: 2.0rem;
}

/* Premium card */
.tesla-card {
  background: linear-gradient(180deg, rgba(255,255,255,0.07), rgba(255,255,255,0.04));
  border: 1px solid rgba(255,255,255,0.10);
  box-shadow:
    0 30px 90px rgba(0,0,0,0.50),
    inset 0 1px 0 rgba(255,255,255,0.10);
  border-radius: 20px;
  padding: 18px 18px;
}

/* KPI typography */
.kpi-title {
  font-size: 12px;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: rgba(242,246,255,0.75);
  margin-bottom: 10px;
}
.kpi-value {
  font-size: 40px;
  font-weight: 900;
  color: #FFFFFF;
  line-height: 1.0;
}

/* Pills */
.pill {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 7px 12px;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 900;
  border: 1px solid rgba(255,255,255,0.12);
  margin-top: 12px;
}
.pill-ok { background: rgba(0, 255, 180, 0.12); color: #85FFE0; }
.pill-monitor { background: rgba(255, 210, 0, 0.12); color: #FFE28A; }
.pill-now { background: rgba(255, 60, 60, 0.12); color: #FFB0B0; }


/* --- Dark premium DataFrame (Streamlit / AG-Grid-like) --- */
div[data-testid="stDataFrame"] * {
  color: #F2F6FF !important;
}
div[data-testid="stDataFrame"] {
  background: rgba(255,255,255,0.05) !important;
  border: 1px solid rgba(255,255,255,0.10) !important;
  border-radius: 16px !important;
  overflow: hidden !important;
}
div[data-testid="stDataFrame"] [role="grid"] {
  background: transparent !important;
}
div[data-testid="stDataFrame"] [role="columnheader"] {
  background: rgba(255,255,255,0.06) !important;
  color: #FFFFFF !important;
  font-weight: 800 !important;
}
div[data-testid="stDataFrame"] [role="row"] {
  background: rgba(255,255,255,0.02) !important;
}
div[data-testid="stDataFrame"] [role="row"]:hover {
  background: rgba(255,255,255,0.06) !important;
}


/* Expander */
[data-testid="stExpander"] {
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 16px;
}

/* Code / JSON / exceptions readable */
pre, code, .stCodeBlock, [data-testid="stJson"], [data-testid="stException"] {
  color: #E9EEF6 !important;
  background: rgba(15, 20, 30, 0.70) !important;
  border: 1px solid rgba(255,255,255,0.10) !important;
  border-radius: 14px !important;
}

/* Plotly transparent */
.js-plotly-plot, .plotly, .plot-container {
  background: transparent !important;
}
</style>
""",
    unsafe_allow_html=True
)


# ---------------------------
# Helpers
# ---------------------------
def require_file(path: Path, hint: str):
    if not path.exists():
        st.error(f"Missing file: {path}\n\n{hint}")
        st.stop()


@st.cache_data(show_spinner=False)
def load_val_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    needed = {"engine_id", "cycle", "RUL", "RUL_pred", "health_index", "decision"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")
    return df


@st.cache_data(show_spinner=False)
def load_raw_train(path: Path) -> pd.DataFrame:
    col_names = (
        ["engine_id", "cycle"]
        + [f"setting_{i}" for i in range(1, 4)]
        + [f"sensor_{i}" for i in range(1, 22)]
    )
    df = pd.read_csv(path, sep=r"\s+", header=None)
    df = df.loc[:, ~df.isna().all(axis=0)]
    df.columns = col_names
    return df


@st.cache_resource(show_spinner=False)
def load_model_bundle(path: Path):
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or "model" not in bundle or "features" not in bundle:
        raise ValueError("Model bundle must be a dict with keys: 'model', 'features'.")
    return bundle


def decision_from_rul(rul_pred: float, service_thr: int, monitor_thr: int) -> str:
    if rul_pred <= service_thr:
        return "SERVICE_NOW"
    if rul_pred <= monitor_thr:
        return "MONITOR"
    return "OK"


def fleet_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.sort_values(["engine_id", "cycle"])
          .groupby("engine_id")
          .tail(1)
          .reset_index(drop=True)
    )


def early_warning_lead_time_by_engine(df_engine: pd.DataFrame, service_thr: int) -> int | None:
    df_engine = df_engine.sort_values("cycle")
    end_cycle = int(df_engine["cycle"].max())
    trigger_rows = df_engine[df_engine["RUL_pred"] <= service_thr]
    if trigger_rows.empty:
        return None
    first_trigger_cycle = int(trigger_rows["cycle"].iloc[0])
    return end_cycle - first_trigger_cycle


def compute_anomaly_flags(df_engine: pd.DataFrame, sensor_cols: list[str], z_threshold: float = 3.0, window: int = 20) -> pd.DataFrame:
    df_engine = df_engine.sort_values("cycle").copy()
    df_engine["anomaly_score"] = 0.0

    for col in sensor_cols:
        if col not in df_engine.columns:
            continue
        delta = df_engine[col].diff()
        mu = delta.rolling(window, min_periods=5).mean()
        sigma = delta.rolling(window, min_periods=5).std()
        z = (delta - mu) / sigma.replace(0, np.nan)
        z = z.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        df_engine["anomaly_score"] = np.maximum(df_engine["anomaly_score"], z.abs())

    df_engine["is_anomaly"] = df_engine["anomaly_score"] >= z_threshold
    return df_engine


def get_rf_feature_importance(model_pipeline, feature_names: list[str]) -> pd.DataFrame:
    rf = model_pipeline.named_steps.get("rf", None)
    if rf is None or not hasattr(rf, "feature_importances_"):
        return pd.DataFrame(columns=["feature", "importance"])
    imp = rf.feature_importances_
    out = pd.DataFrame({"feature": feature_names, "importance": imp})
    return out.sort_values("importance", ascending=False).reset_index(drop=True)


# ---------------------------
# Load artifacts
# ---------------------------
require_file(VAL_PATH, "Run `phase4_decisions.py` to generate `data/processed/val_predictions_decisions.csv`.")
require_file(MODEL_PATH, "Run `phase4_decisions.py` to generate `models/rf_rul_model.joblib`.")
require_file(TRAIN_PATH, "Put `train_FD001.txt` into data/raw/.")

val_df = load_val_df(VAL_PATH)
raw_train = load_raw_train(TRAIN_PATH)

# Merge raw sensors into predictions (for anomaly + drilldowns)
val_df = val_df.merge(raw_train, on=["engine_id", "cycle"], how="left")

with st.expander("🔧 Debug (dataset check)", expanded=False):
    sensor_cols = [c for c in val_df.columns if c.startswith("sensor_")]
    st.write("VAL rows:", val_df.shape[0], "| cols:", val_df.shape[1])
    st.write("Sensors found:", len(sensor_cols))
    st.write("First sensors:", sensor_cols[:8])
    st.write("Null % sensor_1:", float(val_df["sensor_1"].isna().mean()) if "sensor_1" in val_df.columns else "n/a")

bundle = load_model_bundle(MODEL_PATH)
rf_model = bundle["model"]
model_features = bundle["features"]

# NOTE: For this dashboard, we use RF for explainability/importance.
# You can extend LSTM later for separate inference paths.
if model_choice.startswith("LSTM") and (lstm_model is None or lstm_meta is None):
    st.sidebar.warning("LSTM artifacts not found. Falling back to RandomForest.")
    model_choice = "RandomForest (Baseline)"


# ---------------------------
# Sidebar controls
# ---------------------------
st.sidebar.markdown("## ⚙️ Controls")
service_thr = st.sidebar.slider("SERVICE_NOW threshold (RUL ≤)", 5, 60, 15, 1)
monitor_thr = st.sidebar.slider("MONITOR threshold (RUL ≤)", 10, 150, 40, 1)
decision_filter = st.sidebar.selectbox("Fleet filter", ["ALL", "SERVICE_NOW", "MONITOR", "OK"])

min_cycle = int(val_df["cycle"].min())
max_cycle = int(val_df["cycle"].max())
cycle_range = st.sidebar.slider("Cycle range", min_cycle, max_cycle, (min_cycle, max_cycle))

anomaly_z = st.sidebar.slider("Anomaly z-threshold", 2.0, 6.0, 3.0, 0.1)
anomaly_window = st.sidebar.slider("Anomaly rolling window", 10, 60, 20, 1)
topk_sensors = st.sidebar.slider("Top sensors for anomaly check", 3, 15, 8, 1)

# Apply filters
df = val_df[(val_df["cycle"] >= cycle_range[0]) & (val_df["cycle"] <= cycle_range[1])].copy()
df["decision_live"] = df["RUL_pred"].apply(lambda x: decision_from_rul(x, service_thr, monitor_thr))

fleet = fleet_snapshot(df).copy()
fleet["decision"] = fleet["decision_live"]

# KPIs
engines = int(fleet["engine_id"].nunique())
n_service = int((fleet["decision"] == "SERVICE_NOW").sum())
n_monitor = int((fleet["decision"] == "MONITOR").sum())
n_ok = int((fleet["decision"] == "OK").sum())


# ---------------------------
# Header
# ---------------------------
status = "OK" if n_service == 0 else ("MONITOR" if n_monitor > 0 else "SERVICE_NOW")
st.markdown("## ⚡ Fleet Reliability & Predictive Maintenance")
st.caption("RUL prediction • Health Index • Fleet decisions • Explainability • Anomaly signals • Live inference upload")

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(
        f"""<div class="tesla-card"><div class="kpi-title">Engines</div><div class="kpi-value">{engines}</div></div>""",
        unsafe_allow_html=True,
    )
with k2:
    st.markdown(
        f"""<div class="tesla-card"><div class="kpi-title">SERVICE_NOW</div><div class="kpi-value">{n_service}</div>
        <span class="pill pill-now">⚠ High risk</span></div>""",
        unsafe_allow_html=True,
    )
with k3:
    st.markdown(
        f"""<div class="tesla-card"><div class="kpi-title">MONITOR</div><div class="kpi-value">{n_monitor}</div>
        <span class="pill pill-monitor">👀 Watchlist</span></div>""",
        unsafe_allow_html=True,
    )
with k4:
    st.markdown(
        f"""<div class="tesla-card"><div class="kpi-title">OK</div><div class="kpi-value">{n_ok}</div>
        <span class="pill pill-ok">✅ Healthy</span></div>""",
        unsafe_allow_html=True,
    )


# Tabs
tab_overview, tab_explain, tab_anom, tab_ew, tab_upload = st.tabs(
    ["📊 Overview", "🧠 Feature Importance", "🚨 Anomaly Timeline", "⏱ Early-Warning Metric", "📥 Upload + Live Inference"]
)

# ---------------------------
# TAB 1: Overview
# ---------------------------
with tab_overview:
    left, right = st.columns([1.2, 1.0])

    fleet_view = fleet.copy()
    if decision_filter != "ALL":
        fleet_view = fleet_view[fleet_view["decision"] == decision_filter]

    with left:
        fig_hist = px.histogram(
            fleet_view, x="RUL_pred", nbins=30,
            hover_data=["engine_id", "decision"],
        )
        show_plot(fig_hist, key="overview_hist", title="Predicted RUL Distribution (Fleet Snapshot)")

    with right:
        mix = fleet_view["decision"].value_counts().reset_index()
        mix.columns = ["decision", "count"]
        fig_pie = px.pie(mix, names="decision", values="count", hole=0.55)
        show_plot(fig_pie, key="overview_pie", title="Decision Mix")

    fig_scatter = px.scatter(
        fleet_view, x="health_index", y="RUL_pred", color="decision",
        hover_data=["engine_id", "cycle"],
    )
    show_plot(fig_scatter, key="overview_scatter", title="Health Index vs Predicted RUL (Fleet Snapshot)")

    st.markdown("### Top Risk Engines (Latest Cycle)")
    decision_order = {"SERVICE_NOW": 0, "MONITOR": 1, "OK": 2}
    fleet_view = fleet_view.copy()
    fleet_view["rank"] = fleet_view["decision"].map(decision_order)
    fleet_sorted = fleet_view.sort_values(["rank", "RUL_pred", "health_index"], ascending=[True, True, True])

    st.dataframe(
        fleet_sorted[["engine_id", "cycle", "RUL_pred", "health_index", "decision"]].head(30).reset_index(drop=True),
        use_container_width=True
    )

    st.download_button(
        "⬇️ Download Fleet Snapshot CSV",
        data=fleet_sorted.drop(columns=["rank"]).to_csv(index=False).encode("utf-8"),
        file_name="fleet_snapshot.csv",
        mime="text/csv",
    )

    st.markdown("### Engine Drill-down")
    engine_id = st.selectbox("Select engine_id", sorted(df["engine_id"].unique()), key="engine_overview")
    e = df[df["engine_id"] == engine_id].sort_values("cycle").copy()
    e["decision"] = e["decision_live"]

    c1, c2 = st.columns([1.25, 0.75])
    with c1:
        fig_rul = go.Figure()
        fig_rul.add_trace(go.Scatter(x=e["cycle"], y=e["RUL"], mode="lines", name="RUL (Actual)"))
        fig_rul.add_trace(go.Scatter(x=e["cycle"], y=e["RUL_pred"], mode="lines", name="RUL (Predicted)"))
        show_plot(fig_rul, key="overview_rul", title="RUL over Cycles")

    with c2:
        latest = e.tail(1).iloc[0]
        st.markdown(
            f"""
            <div class="tesla-card">
              <div class="kpi-title">Latest engine state</div>
              <div style="font-size:20px; font-weight:900; color:#FFF;">Engine {int(latest["engine_id"])}</div>
              <div style="margin-top:10px;">
                <span class="pill {"pill-ok" if latest["decision"]=="OK" else ("pill-monitor" if latest["decision"]=="MONITOR" else "pill-now")}">
                  {latest["decision"]}
                </span>
              </div>
              <div style="margin-top:14px; color: rgba(242,246,255,0.85); line-height:1.65;">
                <div><b>Cycle:</b> {int(latest["cycle"])}</div>
                <div><b>Predicted RUL:</b> {float(latest["RUL_pred"]):.2f}</div>
                <div><b>Health Index:</b> {float(latest["health_index"]):.3f}</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    fig_hi = px.line(e, x="cycle", y="health_index")
    show_plot(fig_hi, key="overview_health", title="Health Index over Cycles")

    st.markdown("### Recent (last 30 cycles)")
    st.dataframe(
        e[["cycle", "RUL", "RUL_pred", "health_index", "decision"]].tail(30).reset_index(drop=True),
        use_container_width=True
    )

# ---------------------------
# TAB 2: Feature importance (Top Sensors clean)
# ---------------------------
with tab_explain:
    st.markdown("### Model Explainability: Top Drivers (RandomForest Feature Importance)")

    imp_df = get_rf_feature_importance(rf_model, model_features)
    if imp_df.empty:
        st.warning("Could not extract feature importances (expected a Pipeline with a RandomForest step named 'rf').")
        st.stop()

    topn = st.slider("Show top N features", 10, 60, 25, 1, key="top_features_n")
    top_imp = imp_df.head(topn).copy()

    fig_imp = px.bar(top_imp[::-1], x="importance", y="feature", orientation="h")
    show_plot(fig_imp, key="explain_feat_imp", title=f"Top {topn} Feature Importances")

    st.markdown("### Top Sensors (Clean Summary)")
    st.caption(
        "We aggregate importance across engineered variants (roll_mean / roll_std / delta) into one score per sensor."
    )

    sensor_features = imp_df[imp_df["feature"].str.startswith("sensor_")].copy()

    def base_sensor(name: str) -> str:
        return "_".join(name.split("_")[:2])  # sensor_7_roll_mean -> sensor_7

    sensor_features["sensor"] = sensor_features["feature"].apply(base_sensor)

    sensor_imp = (
        sensor_features.groupby("sensor", as_index=False)["importance"]
        .sum()
        .sort_values("importance", ascending=False)
    )

    top_s = st.slider("Top sensors to display", 5, 21, 10, 1, key="top_sensors_display")
    top_sensor_imp = sensor_imp.head(top_s).copy()

    c1, c2 = st.columns([1.0, 1.2], gap="large")
    with c1:
        st.markdown('<div class="tesla-card">', unsafe_allow_html=True)
        st.markdown("#### Sensor Ranking")
        ranked = top_sensor_imp.copy()
        ranked.insert(0, "rank", range(1, len(ranked) + 1))
        ranked["importance"] = ranked["importance"].round(6)
        st.dataframe(ranked, use_container_width=True, height=420, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="tesla-card">', unsafe_allow_html=True)
        st.markdown("#### Sensor Contribution (Aggregated Importance)")
        fig_sensor = px.bar(
            top_sensor_imp.sort_values("importance"),
            x="importance",
            y="sensor",
            orientation="h"
        )
        show_plot(fig_sensor, key="explain_top_sensors", title=f"Top {top_s} Sensors")
        st.markdown("</div>", unsafe_allow_html=True)

    st.success(
        "✅ Interpretation: These sensors are the strongest drivers of RUL predictions (aggregated across rolling stats + deltas)."
    )

# ---------------------------
# TAB 3: Anomaly timeline
# ---------------------------
with tab_anom:
    st.markdown("### Anomaly Timeline (sensor spikes / instability)")

    # Use top sensors from importance list (base sensors)
    imp_df = get_rf_feature_importance(rf_model, model_features)
    base_sensors = []
    for f in imp_df["feature"].tolist():
        if f.startswith("sensor_"):
            base = "_".join(f.split("_")[:2])
            if base not in base_sensors:
                base_sensors.append(base)
        if len(base_sensors) >= topk_sensors:
            break

    engine_id = st.selectbox("Select engine for anomaly view", sorted(df["engine_id"].unique()), key="engine_anom")
    e = df[df["engine_id"] == engine_id].sort_values("cycle").copy()

    available = []
    for s in base_sensors:
        if s in e.columns:
            available.append(s)
        elif f"{s}_delta" in e.columns:
            available.append(f"{s}_delta")

    if not available:
        st.warning("No sensor columns found in the dashboard dataset to compute anomalies.")
        st.stop()

    st.caption("Using: " + ", ".join(available[:min(len(available), 8)]) + (" ..." if len(available) > 8 else ""))

    e_an = compute_anomaly_flags(e, sensor_cols=available, z_threshold=anomaly_z, window=anomaly_window)

    fig_score = px.line(e_an, x="cycle", y="anomaly_score")
    show_plot(fig_score, key="anom_score", title="Anomaly Score over Cycles (max |z| across sensors)")

    anomalies = e_an[e_an["is_anomaly"]]
    st.markdown("#### Flagged anomaly cycles")
    st.dataframe(
        anomalies[["cycle", "anomaly_score", "RUL", "RUL_pred", "health_index"]].tail(50),
        use_container_width=True
    )

    col_choice = st.selectbox("Inspect one sensor/feature", available, key="anom_col")
    fig_one = px.line(e_an, x="cycle", y=col_choice)
    show_plot(fig_one, key="anom_one_sensor", title=f"{col_choice} over Cycles")

# ---------------------------
# TAB 4: Early-warning lead time metric
# ---------------------------
with tab_ew:
    st.markdown("### Early-Warning Metric: Lead Time Before Failure (Median across engines)")
    st.caption(
        "Definition: First cycle where predicted RUL ≤ SERVICE_NOW threshold. "
        "Lead time = (last cycle) - (first trigger cycle)."
    )

    lead_times = []
    per_engine_rows = []
    for eid, g in df.groupby("engine_id"):
        g = g.sort_values("cycle").copy()
        lt = early_warning_lead_time_by_engine(g, service_thr=service_thr)
        if lt is not None:
            lead_times.append(lt)

        end_row = g.tail(1).iloc[0]
        per_engine_rows.append({
            "engine_id": int(eid),
            "last_cycle": int(end_row["cycle"]),
            "last_RUL_pred": float(end_row["RUL_pred"]),
            "last_health_index": float(end_row["health_index"]),
            "lead_time_cycles": lt
        })

    per_engine = pd.DataFrame(per_engine_rows)
    median_lt = float(np.median(lead_times)) if lead_times else float("nan")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f"""<div class="tesla-card"><div class="kpi-title">Median lead time (cycles)</div>
            <div class="kpi-value">{0 if np.isnan(median_lt) else int(median_lt)}</div></div>""",
            unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            f"""<div class="tesla-card"><div class="kpi-title">Engines with alert</div>
            <div class="kpi-value">{len(lead_times)}</div></div>""",
            unsafe_allow_html=True
        )
    with c3:
        st.markdown(
            f"""<div class="tesla-card"><div class="kpi-title">SERVICE threshold</div>
            <div class="kpi-value">{service_thr}</div></div>""",
            unsafe_allow_html=True
        )

    fig_lt = px.histogram(pd.DataFrame({"lead_time_cycles": lead_times}), x="lead_time_cycles", nbins=25)
    show_plot(fig_lt, key="ew_leadtime_hist", title="Lead Time Distribution (cycles)")

    st.markdown("#### Engine lead-time table (latest state + lead time)")
    st.dataframe(
        per_engine.sort_values(["lead_time_cycles"], ascending=True).reset_index(drop=True).head(40),
        use_container_width=True
    )

    st.download_button(
        "⬇️ Download Early-Warning Report CSV",
        data=per_engine.to_csv(index=False).encode("utf-8"),
        file_name="early_warning_report.csv",
        mime="text/csv"
    )

# ---------------------------
# TAB 5: Upload + live inference
# ---------------------------
with tab_upload:
    st.markdown("### Upload New Data + Run Live Inference")
    st.caption(
        "Upload a turbofan file in the same format as FD001 (space-separated columns, no header), "
        "then we’ll run the saved model and produce a fleet snapshot."
    )

    uploaded = st.file_uploader(
        "Upload a turbofan TXT file (e.g., test_FD001.txt or similar)",
        type=["txt"]
    )

    if uploaded is None:
        st.info("Upload a file to run inference.")
    else:
        raw_bytes = uploaded.getvalue()
        text = raw_bytes.decode("utf-8", errors="ignore")
        buf = io.StringIO(text)

        col_names = (
            ["engine_id", "cycle"]
            + [f"setting_{i}" for i in range(1, 4)]
            + [f"sensor_{i}" for i in range(1, 22)]
        )

        raw_df = pd.read_csv(buf, sep=r"\s+", header=None)
        raw_df = raw_df.loc[:, ~raw_df.isna().all(axis=0)]
        raw_df.columns = col_names
        raw_df = raw_df.sort_values(["engine_id", "cycle"]).reset_index(drop=True)

        st.success(f"Loaded upload: {raw_df.shape[0]} rows, {raw_df['engine_id'].nunique()} engines")

        SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]

        def add_rolling_features_local(df_in: pd.DataFrame, window: int = 5) -> pd.DataFrame:
            df2 = df_in.copy()
            for col in SENSOR_COLS:
                df2[f"{col}_roll_mean"] = (
                    df2.groupby("engine_id")[col].rolling(window, min_periods=1).mean()
                    .reset_index(level=0, drop=True)
                )
                df2[f"{col}_roll_std"] = (
                    df2.groupby("engine_id")[col].rolling(window, min_periods=1).std()
                    .reset_index(level=0, drop=True)
                    .fillna(0)
                )
            return df2

        def add_delta_features_local(df_in: pd.DataFrame) -> pd.DataFrame:
            df2 = df_in.copy()
            for col in SENSOR_COLS:
                df2[f"{col}_delta"] = df2.groupby("engine_id")[col].diff().fillna(0)
            return df2

        def add_health_index_local(df_in: pd.DataFrame) -> pd.DataFrame:
            df2 = df_in.copy()
            s = df2[SENSOR_COLS]
            s_min = s.min()
            s_max = s.max()
            denom = (s_max - s_min).replace(0, np.nan)
            scaled = (s - s_min) / denom
            scaled = scaled.fillna(0.0)
            df2["health_index"] = 1 - scaled.mean(axis=1)
            return df2

        fe = add_rolling_features_local(raw_df, window=5)
        fe = add_delta_features_local(fe)
        fe = add_health_index_local(fe)

        missing = [c for c in model_features if c not in fe.columns]
        if missing:
            st.error(f"Uploaded data missing {len(missing)} required features. Example: {missing[:10]}")
            st.stop()

        X = fe[model_features].copy().replace([np.inf, -np.inf], np.nan).fillna(0)
        preds = rf_model.predict(X)

        fe_out = fe[["engine_id", "cycle", "health_index"]].copy()
        fe_out["RUL_pred"] = preds
        fe_out["decision"] = fe_out["RUL_pred"].apply(lambda x: decision_from_rul(x, service_thr, monitor_thr))

        upl_fleet = fleet_snapshot(fe_out)

        st.markdown("#### Fleet Snapshot (Upload)")
        st.dataframe(
            upl_fleet.sort_values(["decision", "RUL_pred"], ascending=[True, True])[
                ["engine_id", "cycle", "RUL_pred", "health_index", "decision"]
            ].reset_index(drop=True),
            use_container_width=True
        )

        fig_u = px.histogram(upl_fleet, x="RUL_pred", nbins=30)
        show_plot(fig_u, key="upload_hist", title="Uploaded Fleet: Predicted RUL Distribution")

        st.download_button(
            "⬇️ Download Upload Predictions CSV",
            data=fe_out.to_csv(index=False).encode("utf-8"),
            file_name="upload_predictions.csv",
            mime="text/csv"
        )
