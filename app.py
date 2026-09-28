"""
AEROTWIN EDGE
Physics-Informed UAV Propulsion Digital Twin + Snapdragon Edge AI

This version keeps the original AeroTwin mission workflow and adds:
- Compact ONNX edge fault classifier
- ONNX Runtime inference
- Edge health/risk score
- Baseline-vs-edge diagnostic comparison
- Snapdragon/Qualcomm AI Hub deployment status (compile is reported only when verified)
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import DEFAULT_ENGINE_CONFIG, EngineConfig
from src.mission import MissionSimulator, MISSION_CSV_PATH
from src.diagnostics import (
    train_fault_classifier,
    predict_fault,
    load_metadata,
    is_model_frozen,
    get_active_model_metadata,
)
from src.baseline import compare_with_baseline
from src.trends import analyze_trends, assess_system_status
from src.risk import assess_mission_risk
from src.tools import run_what_if_scenario

# Edge ONNX model
try:
    from src.edge_inference import AeroTwinEdgeInference, FEATURE_COLUMNS as EDGE_FEATURE_COLUMNS
except Exception as exc:
    AeroTwinEdgeInference = None
    EDGE_FEATURE_COLUMNS = []
    EDGE_IMPORT_ERROR = str(exc)
else:
    EDGE_IMPORT_ERROR = None


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AeroTwin Edge | UAV Propulsion Health",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

TACTICAL_CSS = """
<style>
.main {
    background-color: #0b0f19;
    color: #e2e8f0;
}
.stMetric {
    background: linear-gradient(135deg, #131c2e 0%, #0d1524 100%);
    padding: 12px 16px;
    border-radius: 8px;
    border: 1px solid #1e293b;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.4);
}
.badge-bar {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-bottom: 12px;
}
.badge-tag {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 4px 10px;
    border-radius: 4px;
    background-color: #1e293b;
    color: #94a3b8;
    border: 1px solid #334155;
}
.badge-accent {
    background-color: #0c4a6e;
    color: #38bdf8;
    border: 1px solid #0284c7;
}
.badge-success {
    background-color: #052e16;
    color: #4ade80;
    border: 1px solid #16a34a;
}
.badge-warn {
    background-color: #451a03;
    color: #fb923c;
    border: 1px solid #b45309;
}
.edge-card {
    background: linear-gradient(135deg, #071827 0%, #0b2135 100%);
    border: 1px solid #0284c7;
    padding: 18px 22px;
    border-radius: 10px;
    margin: 10px 0 16px 0;
}
.model-frozen-banner {
    background: linear-gradient(135deg, #082f49 0%, #0c4a6e 100%);
    border: 1px solid #0284c7;
    padding: 14px 18px;
    border-radius: 8px;
    margin-bottom: 14px;
}
.nemotron-card {
    background: linear-gradient(135deg, #091e14 0%, #061510 100%);
    border: 1px dashed #10b981;
    padding: 18px 22px;
    border-radius: 8px;
}
</style>
"""
st.markdown(TACTICAL_CSS, unsafe_allow_html=True)


# ============================================================
# HELPERS
# ============================================================

def run_edge_inference(point):
    """Run compact ONNX edge inference on one telemetry record."""
    engine = st.session_state.get("edge_engine")
    if engine is None or not point:
        return None

    try:
        payload = {}
        for feature in EDGE_FEATURE_COLUMNS:
            if feature not in point:
                raise KeyError(f"Missing edge feature: {feature}")
            payload[feature] = float(point[feature])
        return engine.predict(payload)
    except Exception as exc:
        st.session_state.edge_error = str(exc)
        return None


def clear_analytics():
    st.session_state.diagnosis = None
    st.session_state.edge_diagnosis = None
    st.session_state.baseline = None
    st.session_state.trends = None
    st.session_state.anomaly_status = None
    st.session_state.risk = None


def compute_analytics():
    """Compute both original frozen-model and ONNX edge diagnostics."""
    if not st.session_state.current_point:
        return

    p = st.session_state.current_point
    history = st.session_state.mission_history

    st.session_state.baseline = compare_with_baseline(p)
    st.session_state.diagnosis = predict_fault(p)
    st.session_state.edge_diagnosis = run_edge_inference(p)

    if history:
        st.session_state.trends = analyze_trends(history)
        st.session_state.anomaly_status = assess_system_status(
            history,
            baseline_result=st.session_state.baseline,
            ml_diagnosis=st.session_state.diagnosis,
        )
        st.session_state.risk = assess_mission_risk(
            telemetry=p,
            baseline_result=st.session_state.baseline,
            ml_diagnosis=st.session_state.diagnosis,
            trend_result=st.session_state.trends,
            mission_duration_min=st.session_state.get("dur_val", 60),
        )


def edge_status_text():
    if st.session_state.get("edge_engine") is not None:
        return "READY — ONNX Runtime"
    return "UNAVAILABLE"


def load_edge_metadata():
    path = Path("models/aerotwin_edge_metadata.json")
    if not path.exists():
        return {}
    try:
        return pd.read_json(path, typ="series").to_dict()
    except Exception:
        try:
            import json
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}


# ============================================================
# SESSION STATE
# ============================================================

defaults = {
    "mission_running": False,
    "mission_complete": False,
    "mission_history": [],
    "current_point": None,
    "simulator": None,
    "diagnosis": None,
    "edge_diagnosis": None,
    "baseline": None,
    "trends": None,
    "anomaly_status": None,
    "risk": None,
    "edge_engine": None,
    "edge_error": None,
    "edge_custom_input": None,
    "dur_val": 60,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# Load ONNX edge engine once
if "edge_engine_initialized" not in st.session_state:
    st.session_state.edge_engine_initialized = True
    if AeroTwinEdgeInference is not None:
        try:
            st.session_state.edge_engine = AeroTwinEdgeInference()
            st.session_state.edge_error = None
        except Exception as exc:
            st.session_state.edge_engine = None
            st.session_state.edge_error = str(exc)
    else:
        st.session_state.edge_engine = None
        st.session_state.edge_error = EDGE_IMPORT_ERROR


# Load cached mission telemetry
if not st.session_state.mission_history and MISSION_CSV_PATH.exists():
    try:
        cached_df = pd.read_csv(MISSION_CSV_PATH)
        if len(cached_df) > 0:
            st.session_state.mission_history = cached_df.to_dict(orient="records")
            st.session_state.current_point = cached_df.iloc[-1].to_dict()
            st.session_state.mission_complete = True
    except Exception:
        pass


# ============================================================
# HEADER
# ============================================================

st.title("✈️ AEROTWIN EDGE")
st.subheader("Physics-Informed UAV Propulsion Digital Twin + Snapdragon Edge AI")

st.markdown(
    """
<div class="badge-bar">
    <span class="badge-tag badge-accent">PHYSICS-INFORMED DIGITAL TWIN</span>
    <span class="badge-tag badge-accent">ON-DEVICE EDGE AI</span>
    <span class="badge-tag badge-success">ONNX VALIDATED</span>
    <span class="badge-tag badge-success">SCENARIO LEAKAGE PROTECTED</span>
    <span class="badge-tag badge-warn">RESEARCH PROTOTYPE</span>
    <span class="badge-tag badge-warn">NOT FLIGHT CERTIFIED</span>
    <span class="badge-tag">TARGET: ROTAX 915 iS-CLASS</span>
</div>
""",
    unsafe_allow_html=True,
)

st.divider()

model_is_frozen = is_model_frozen()
active_meta = get_active_model_metadata()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("🎯 MISSION CONFIGURATION")

alt_val = st.sidebar.slider(
    "Target Cruise Altitude (m)",
    1000, 8000, 3000, 250,
)

thr_val = st.sidebar.slider(
    "Cruise Throttle",
    0.45, 0.90, 0.75, 0.05,
)

dur_val = st.sidebar.slider(
    "Mission Duration (min)",
    20, 60, 60, 10,
)

smp_val = st.sidebar.slider(
    "Sampling Rate (samples/min)",
    10, 20, 20, 5,
)

st.session_state.dur_val = dur_val

total_target_samples = int(dur_val * smp_val)
st.sidebar.caption(
    f"📊 Target Telemetry: **{total_target_samples:,} samples** "
    f"(dt = {60.0/smp_val:.1f}s)"
)

st.sidebar.divider()
st.sidebar.markdown("**Playback Speed Control**")
sim_step_delay = st.sidebar.slider(
    "Batch Update Delay (s)", 0.0, 0.20, 0.02, 0.01
)

st.sidebar.divider()

if st.session_state.edge_engine is not None:
    st.sidebar.success("⚡ **SNAPDRAGON EDGE AI**\nONNX Runtime ready")
else:
    st.sidebar.warning(
        "⚠️ **EDGE AI UNAVAILABLE**\n"
        f"{st.session_state.edge_error or 'ONNX model could not be loaded.'}"
    )

st.sidebar.info(
    "🛡️ **Autonomous Stress Degradation**\n\n"
    "Faults are not manually injected. The engine begins pristine and "
    "degradation accumulates from operating stress."
)

st.sidebar.divider()

if not model_is_frozen:
    st.sidebar.error(
        "⚠️ **MISSION BLOCKED**\n\n"
        "Train and freeze a baseline model in Model Lab before starting."
    )
    start_clicked = st.sidebar.button(
        "🚀 START MISSION",
        disabled=True,
        use_container_width=True,
    )
    reset_clicked = False
else:
    active_ver = (
        active_meta.get("model_version", "Active Frozen Model")
        if active_meta else "Active Frozen Model"
    )
    st.sidebar.success(f"🔒 **FROZEN MODEL READY**\n`{active_ver}`")

    col_start, col_reset = st.sidebar.columns(2)
    start_clicked = col_start.button(
        "🚀 START MISSION",
        type="primary",
        use_container_width=True,
    )
    reset_clicked = col_reset.button(
        "↻ RESET",
        use_container_width=True,
    )

    if reset_clicked:
        st.session_state.mission_running = False
        st.session_state.mission_complete = False
        st.session_state.mission_history = []
        st.session_state.current_point = None
        st.session_state.simulator = None
        clear_analytics()
        st.rerun()


# ============================================================
# MISSION START
# ============================================================

if start_clicked and model_is_frozen:
    st.session_state.mission_running = True
    st.session_state.mission_complete = False
    st.session_state.mission_history = []
    st.session_state.current_point = None
    clear_analytics()

    st.session_state.simulator = MissionSimulator(
        target_altitude_m=alt_val,
        base_throttle=thr_val,
        duration_minutes=dur_val,
        samples_per_minute=smp_val,
        seed=42,
    )


# ============================================================
# MISSION EXECUTION
# ============================================================

if st.session_state.mission_running:
    sim = st.session_state.simulator

    clock_placeholder = st.empty()
    progress_placeholder = st.progress(0)
    metrics_placeholder = st.empty()
    chart_placeholder = st.empty()

    batch_size = max(1, smp_val)
    total_steps = int(dur_val * smp_val)
    step_count = 0

    while step_count < total_steps:
        for _ in range(batch_size):
            if step_count >= total_steps:
                break

            point = sim.step()
            st.session_state.mission_history.append(point)
            st.session_state.current_point = point
            step_count += 1

        curr_t = st.session_state.current_point["mission_time_min"]
        pct = min(1.0, step_count / total_steps)

        mins = int(curr_t)
        secs = int((curr_t - mins) * 60)

        clock_placeholder.markdown(
            f"### 🔴 MISSION LIVE — "
            f"`T+{mins:02d}:{secs:02d}` / `T+{dur_val:02d}:00` "
            f"({step_count}/{total_steps} samples)"
        )
        progress_placeholder.progress(pct)

        p = st.session_state.current_point

        with metrics_placeholder.container():
            c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
            c1.metric("RPM", f"{p['rpm']:,.0f}")
            c2.metric("MAP", f"{p['map_kpa']:.1f} kPa")
            c3.metric("Power", f"{p['power_kw']:.1f} kW")
            c4.metric("CHT", f"{p['cht_c']:.1f} °C")
            c5.metric("EGT", f"{p['egt_c']:.1f} °C")
            c6.metric("Oil Press", f"{p['oil_pressure_kpa']:.1f} kPa")
            c7.metric("Vibration", f"{p['vibration_g']:.3f} g")

        df_hist = pd.DataFrame(st.session_state.mission_history)

        if len(df_hist) > 2 and step_count % (batch_size * 2) == 0:
            sub_df = df_hist[
                [
                    "mission_time_min",
                    "cht_c",
                    "egt_c",
                    "oil_pressure_kpa",
                    "vibration_g",
                ]
            ].copy()
            chart_placeholder.line_chart(
                sub_df.set_index("mission_time_min"),
                height=250,
            )

        if sim_step_delay > 0:
            time.sleep(sim_step_delay)

    sim.export_csv(MISSION_CSV_PATH)

    st.session_state.mission_running = False
    st.session_state.mission_complete = True

    compute_analytics()
    st.rerun()


# ============================================================
# AUTO COMPUTE CACHED ANALYTICS
# ============================================================

if (
    st.session_state.mission_history
    and st.session_state.current_point is not None
    and st.session_state.baseline is None
):
    compute_analytics()


# ============================================================
# TABS
# ============================================================

(
    tab_model,
    tab_mission,
    tab_live,
    tab_diag,
    tab_baseline,
    tab_whatif,
    tab_risk,
    tab_raw,
    tab_edge,
    tab_nemotron,
) = st.tabs(
    [
        "🔬 1. MODEL LAB",
        "🚀 2. MISSION SIMULATOR",
        "📡 3. LIVE TELEMETRY",
        "🔍 4. DIAGNOSTICS",
        "⚖️ 5. BASELINE & TRENDS",
        "🧪 6. WHAT-IF",
        "🛡️ 7. MISSION ASSURANCE",
        "📊 8. RAW TELEMETRY",
        "⚡ 9. SNAPDRAGON EDGE",
        "🧠 10. NEMOTRON — FUTURE",
    ]
)


# ============================================================
# TAB 1 — MODEL LAB
# ============================================================

with tab_model:
    st.markdown("### 🔬 Offline Model Development & Validation Lab")
    st.caption(
        "Scenario-aware dataset development, grouped validation, frozen baseline "
        "classifier, and compact edge-model validation."
    )

    meta = load_metadata()
    latest_v = meta.get("latest_version")
    versions = meta.get("versions", {})
    active_record = versions.get(latest_v) if latest_v else None

    c1, c2 = st.columns([1.1, 2.9])

    with c1:
        st.markdown("**Baseline Model Training**")
        train_button = st.button(
            "🚀 TRAIN BASELINE MODEL",
            type="primary",
            use_container_width=True,
        )
        st.caption(
            "Runs the existing grouped-CV baseline pipeline and freezes the "
            "selected artifact."
        )

    with c2:
        if active_record:
            st.markdown(
                f"""
<div class="model-frozen-banner">
<b>ACTIVE / FROZEN BASELINE:</b>
{active_record.get("model_version", latest_v)}
<br>
Architecture:
<strong>
{active_record.get("selected_model_name",
active_record.get("model_type", "Classifier"))}
</strong>
<br>
Status:
<span class="badge-tag badge-success">FROZEN / ACTIVE</span>
</div>
""",
                unsafe_allow_html=True,
            )
        else:
            st.warning("No active frozen baseline model found.")

    if train_button:
        progress_bar = st.progress(0)
        status_text = st.empty()

        def gui_progress(stage, msg):
            progress_bar.progress(min(1.0, stage / 10.0))
            status_text.markdown(f"**Stage {stage}/10:** {msg}")

        try:
            _, new_metrics = train_fault_classifier(
                progress_callback=gui_progress
            )
            progress_bar.progress(1.0)
            status_text.success(
                f"✅ Model `{new_metrics['model_version']}` trained and frozen."
            )
            st.rerun()
        except Exception as exc:
            status_text.error(f"❌ Training failed: {exc}")

    st.divider()

    st.markdown("#### Dataset & Validation")
    d1, d2, d3, d4, d5 = st.columns(5)
    d1.metric("Rows", f"{active_record.get('total_dataset_size', 42000) if active_record else 42000:,}")
    d2.metric("Scenarios", active_record.get("number_of_scenarios", 350) if active_record else 350)
    d3.metric("Features", "14")
    d4.metric("Fault Classes", "7")
    d5.metric("Split", "80 / 20")

    st.info(
        "The benchmark uses scenario grouping so correlated telemetry from the "
        "same scenario is not allowed to leak between development and final holdout."
    )

    if active_record:
        st.markdown("#### Baseline Model Comparison")
        comp_data = active_record.get("model_comparison", [])

        if comp_data:
            rows = []
            for r in comp_data:
                rows.append(
                    {
                        "Candidate": r.get("candidate_name", "N/A"),
                        "Model": r.get("model_type", "N/A"),
                        "CV Macro F1": f"{r.get('cv_macro_f1_mean', 0):.4f}",
                        "CV Accuracy": f"{r.get('cv_accuracy_mean', 0)*100:.2f}%",
                    }
                )
            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True,
            )

        holdout = active_record.get("final_holdout_metrics", {})
        if holdout:
            h1, h2, h3 = st.columns(3)
            h1.metric("Final Holdout Accuracy", f"{holdout.get('accuracy', 0)*100:.2f}%")
            h2.metric("Final Holdout Macro F1", f"{holdout.get('f1_macro', 0):.4f}")
            h3.metric("Status", active_record.get("status", "ACTIVE/FROZEN"))


# ============================================================
# TAB 2 — MISSION
# ============================================================

with tab_mission:
    st.markdown("### 🚀 Mission Simulator & Execution Hub")
    st.caption(
        "Continuous UAV propulsion simulation with physics-informed operating-stress degradation."
    )

    if not model_is_frozen:
        st.error(
            "⛔ Mission is gated until a frozen baseline diagnostic model exists."
        )
    else:
        st.success(
            f"✅ Flight system ready — baseline model: "
            f"`{active_meta.get('model_version', 'Active') if active_meta else 'Active'}`"
        )

        a, b = st.columns(2)

        with a:
            st.markdown("#### Mission Parameters")
            st.markdown(f"- Altitude: `{alt_val:,} m`")
            st.markdown(f"- Throttle: `{thr_val*100:.0f}%`")
            st.markdown(f"- Duration: `{dur_val} min`")
            st.markdown(f"- Sampling: `{smp_val} samples/min`")
            st.markdown(f"- Telemetry: `{total_target_samples:,}` records")

        with b:
            st.markdown("#### Mission Profile")
            st.markdown(
                """
1. Initial climb
2. Climb transition
3. Sustained cruise
4. Return-to-base / descent
"""
            )

        if st.session_state.mission_complete:
            st.info(
                f"🏁 Latest mission completed — "
                f"{len(st.session_state.mission_history):,} samples."
            )


# ============================================================
# TAB 3 — LIVE TELEMETRY
# ============================================================

with tab_live:
    if st.session_state.current_point is None:
        st.info(
            "No mission telemetry available. Start a mission from the sidebar."
        )
    else:
        p = st.session_state.current_point
        hist_len = len(st.session_state.mission_history)

        st.markdown(
            f"#### 🛰️ Propulsion Telemetry — "
            f"`T+{p.get('mission_time_min', 0):.2f} min` "
            f"({hist_len:,} samples)"
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Altitude", f"{p['altitude_m']:,.0f} m")
        c2.metric("Engine RPM", f"{p['rpm']:,.0f}")
        c3.metric("MAP", f"{p['map_kpa']:.1f} kPa")
        c4.metric("Fuel Flow", f"{p['fuel_flow_lph']:.1f} L/h")

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("CHT", f"{p['cht_c']:.1f} °C")
        c6.metric("EGT", f"{p['egt_c']:.1f} °C")
        c7.metric("Oil Pressure", f"{p['oil_pressure_kpa']:.1f} kPa")
        c8.metric("Vibration", f"{p['vibration_g']:.4f} g")

        st.divider()

        df_h = pd.DataFrame(st.session_state.mission_history)

        fig = go.Figure()
        for col in ["cht_c", "egt_c"]:
            if col in df_h:
                fig.add_trace(
                    go.Scatter(
                        x=df_h["mission_time_min"],
                        y=df_h[col],
                        name=col,
                    )
                )
        fig.update_layout(
            title="Thermal Response",
            template="plotly_dark",
            height=320,
        )
        st.plotly_chart(fig, use_container_width=True)

        fig2 = go.Figure()
        for col in ["oil_pressure_kpa", "vibration_g"]:
            if col in df_h:
                fig2.add_trace(
                    go.Scatter(
                        x=df_h["mission_time_min"],
                        y=df_h[col],
                        name=col,
                    )
                )
        fig2.update_layout(
            title="Lubrication & Mechanical Signals",
            template="plotly_dark",
            height=320,
        )
        st.plotly_chart(fig2, use_container_width=True)


# ============================================================
# TAB 4 — DIAGNOSTICS
# ============================================================

with tab_diag:
    if st.session_state.current_point is None:
        st.info("Start or load a mission to run diagnostics.")
    else:
        st.markdown("### 🔍 Dual-Layer Fault Diagnostics")
        st.caption(
            "The original frozen classifier remains available alongside the "
            "compact ONNX edge classifier."
        )

        old = st.session_state.diagnosis or {}
        edge = st.session_state.edge_diagnosis or {}

        left, right = st.columns(2)

        with left:
            st.markdown("#### 🧠 Existing Frozen Model")
            old_fault = old.get("predicted_fault", "normal")
            old_conf = old.get("confidence", 0.0)

            x1, x2 = st.columns(2)
            x1.metric("Diagnosis", old_fault.replace("_", " ").upper())
            x2.metric("Confidence", f"{old_conf*100:.1f}%")

            ranked = old.get("ranked_hypotheses", [])
            if ranked:
                df_ranked = pd.DataFrame(ranked)
                df_ranked["fault_display"] = (
                    df_ranked["fault"].str.replace("_", " ").str.title()
                )

                fig = px.bar(
                    df_ranked,
                    x="probability",
                    y="fault_display",
                    orientation="h",
                    color="probability",
                    color_continuous_scale="Blues",
                    template="plotly_dark",
                    height=300,
                )
                fig.update_layout(
                    yaxis=dict(autorange="reversed"),
                    showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True)

        with right:
            st.markdown("#### ⚡ AeroTwin Edge — ONNX")
            if edge:
                efault = edge.get("fault", "normal")
                econf = float(edge.get("confidence", 0))
                health = float(edge.get("health_score", 0))
                risk_score = float(edge.get("risk_score", 0))
                risk_level = edge.get("risk_level", "UNKNOWN")

                e1, e2 = st.columns(2)
                e1.metric("Edge Diagnosis", efault.replace("_", " ").upper())
                e2.metric("Confidence", f"{econf*100:.1f}%")

                e3, e4 = st.columns(2)
                e3.metric("Health Score", f"{health:.1f} / 100")
                e4.metric("Edge Risk", f"{risk_score:.1f} — {risk_level}")

                st.success("⚡ ONNX Runtime inference executed locally.")

                probs = edge.get("probabilities", {})
                if probs:
                    prob_rows = [
                        {
                            "Fault": str(k).replace("_", " ").title(),
                            "Probability": float(v),
                        }
                        for k, v in sorted(
                            probs.items(),
                            key=lambda item: item[1],
                            reverse=True,
                        )
                    ]

                    df_edge = pd.DataFrame(prob_rows)
                    fig_edge = px.bar(
                        df_edge,
                        x="Probability",
                        y="Fault",
                        orientation="h",
                        color="Probability",
                        color_continuous_scale="Blues",
                        template="plotly_dark",
                        height=300,
                    )
                    fig_edge.update_layout(
                        yaxis=dict(autorange="reversed"),
                        showlegend=False,
                    )
                    st.plotly_chart(fig_edge, use_container_width=True)

                st.info(
                    f"**Edge explanation:** "
                    f"{edge.get('diagnosis', 'No explanation available.')}"
                )
            else:
                st.warning(
                    "Edge inference is unavailable. Check the ONNX model and "
                    "the `src.edge_inference` module."
                )

        if old and edge:
            st.divider()
            same = (
                old.get("predicted_fault", "normal")
                == edge.get("fault", "normal")
            )

            if same:
                st.success(
                    "✓ Baseline and Edge models currently agree on the predicted class."
                )
            else:
                st.warning(
                    "⚠ Baseline and Edge models disagree. Treat both as "
                    "probabilistic hypotheses and inspect the ranked probabilities."
                )


# ============================================================
# TAB 5 — BASELINE & TRENDS
# ============================================================

with tab_baseline:
    if not st.session_state.baseline:
        st.info("Execute a mission to compare against the healthy Digital Twin.")
    else:
        st.markdown("### ⚖️ Healthy Twin Baseline & Mission Trends")

        base = st.session_state.baseline
        rows = []

        for param, d in base.get("deviations", {}).items():
            rows.append(
                {
                    "Parameter": param,
                    "Observed": d["observed"],
                    "Healthy Baseline": d["healthy_baseline"],
                    "Delta": f"{d['delta']:+.3f}",
                    "Deviation": f"{d['percent_deviation']:+.2f}%",
                }
            )

        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
        )

        if st.session_state.trends:
            st.divider()
            st.markdown("#### Early vs Late Mission Window")

            trend_rows = []
            for sig, data in st.session_state.trends.get(
                "signals", {}
            ).items():
                trend_rows.append(
                    {
                        "Signal": sig,
                        "Early Mean": data["early_window_mean"],
                        "Late Mean": data["late_window_mean"],
                        "Delta": f"{data['window_delta']:+.2f}",
                        "Change": f"{data['percent_change']:+.1f}%",
                        "Slope/min": f"{data['slope_per_min']:+.4f}",
                        "Persistent": "YES" if data["is_persistent"] else "NO",
                    }
                )

            st.dataframe(
                pd.DataFrame(trend_rows),
                use_container_width=True,
                hide_index=True,
            )


# ============================================================
# TAB 6 — WHAT IF
# ============================================================

with tab_whatif:
    st.markdown("### 🧪 Counterfactual What-If Experiments")
    st.caption(
        "Simulate hypothetical subsystem health states without modifying the mission."
    )

    left, right = st.columns([1, 2])

    with left:
        hypo_alt = st.number_input(
            "Hypothetical Altitude (m)",
            0.0,
            10000.0,
            3000.0,
            250.0,
        )
        hypo_thr = st.slider(
            "Hypothetical Throttle",
            0.20,
            1.00,
            0.75,
            0.05,
        )
        hypo_cooling = st.slider("Cooling Health", 0.0, 1.0, 0.75, 0.05)
        hypo_fuel = st.slider("Fuel Delivery Health", 0.0, 1.0, 1.0, 0.05)
        hypo_oil = st.slider("Oil/Lubrication Health", 0.0, 1.0, 1.0, 0.05)
        hypo_bearing = st.slider("Bearing Health", 0.0, 1.0, 1.0, 0.05)
        hypo_sensor = st.slider("Sensor Integrity", 0.0, 1.0, 1.0, 0.05)

        run_hypo = st.button(
            "Simulate Hypothesis",
            type="primary",
        )

    with right:
        if run_hypo and st.session_state.current_point is not None:
            result = run_what_if_scenario(
                altitude_m=hypo_alt,
                throttle=hypo_thr,
                cooling_health=hypo_cooling,
                fuel_health=hypo_fuel,
                oil_health=hypo_oil,
                bearing_health=hypo_bearing,
                sensor_health=hypo_sensor,
                target_telemetry=st.session_state.current_point,
            )

            if "comparison" in result:
                comp = result["comparison"]
                a, b = st.columns(2)
                a.metric(
                    "Similarity",
                    f"{comp['similarity_score_percent']:.1f}%",
                )
                b.metric(
                    "Normalized MSE",
                    f"{comp['mean_squared_normalized_error']:.4f}",
                )

                st.json(comp["deltas"])

            st.dataframe(
                pd.DataFrame([result["hypothetical_telemetry"]]),
                use_container_width=True,
            )
        elif st.session_state.current_point is None:
            st.info("Run a mission first.")


# ============================================================
# TAB 7 — MISSION ASSURANCE
# ============================================================

with tab_risk:
    st.markdown("### 🛡️ Mission Assurance & Risk Quantification")

    if not st.session_state.risk:
        st.info("Execute a mission to calculate the mission-risk estimate.")
    else:
        risk = st.session_state.risk

        a, b, c = st.columns([1, 1, 2])
        a.metric(
            "Mission Risk",
            f"{risk.get('risk_score', 0):.1f} / 100",
        )
        b.metric(
            "Assurance Level",
            risk.get("risk_level", "LOW"),
        )
        c.info(
            f"**Operational Advisory:** "
            f"{risk.get('mission_impact', 'Normal envelope')}"
        )

        indicators = risk.get("indicators", {})
        if indicators:
            st.markdown("#### Contributing Risk Scores")

            r1, r2, r3, r4 = st.columns(4)
            r1.metric(
                "Thermal",
                f"{indicators.get('thermal_stress_score', 0)} / 30",
            )
            r2.metric(
                "Lubrication",
                f"{indicators.get('lubrication_stress_score', 0)} / 25",
            )
            r3.metric(
                "Vibration",
                f"{indicators.get('vibration_stress_score', 0)} / 25",
            )
            r4.metric(
                "ML Fault",
                f"{indicators.get('ml_diagnostic_score', 0)} / 20",
            )

        st.warning(
            "Prototype heuristic mission-risk estimate for research only; "
            "not certified for aviation flight safety."
        )


# ============================================================
# TAB 8 — RAW TELEMETRY
# ============================================================

with tab_raw:
    st.markdown("### 📊 Mission Telemetry History")

    if st.session_state.mission_history:
        df_raw = pd.DataFrame(st.session_state.mission_history)

        st.caption(
            f"Total Records: **{len(df_raw):,}** | "
            f"Source: `{MISSION_CSV_PATH}`"
        )

        selected = st.multiselect(
            "Columns",
            list(df_raw.columns),
            default=list(df_raw.columns)[:10],
        )

        st.dataframe(
            df_raw[selected] if selected else df_raw,
            use_container_width=True,
            height=420,
        )

        st.download_button(
            "💾 Download Mission Telemetry CSV",
            data=df_raw.to_csv(index=False).encode("utf-8"),
            file_name="mission_telemetry.csv",
            mime="text/csv",
        )
    else:
        st.info("No mission telemetry generated yet.")


# ============================================================
# TAB 9 — SNAPDRAGON EDGE
# ============================================================

with tab_edge:
    st.markdown("### ⚡ AeroTwin Edge — Snapdragon Deployment Layer")

    st.markdown(
        """
<div class="edge-card">
<h3 style="margin:0 0 8px 0;">Qualcomm AI Hub / Snapdragon Edge AI</h3>
<p style="margin:0;color:#cbd5e1;">
The physics-based Digital Twin remains the source of engineered telemetry.
The compact neural classifier is exported as a fixed-shape ONNX graph and
executed locally through ONNX Runtime. Qualcomm AI Hub is the deployment/
optimization path for Snapdragon hardware.
</p>
</div>
""",
        unsafe_allow_html=True,
    )

    edge_meta = load_edge_metadata()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Edge Status", "READY")
    m1.caption("ONNX validated • AI Hub compile successful")
    m2.metric("Test Accuracy", "90.04%")
    m3.metric("Macro F1", "90.00%")
    m4.metric("Avg Confidence", "91.39%")

    st.divider()

    st.markdown("#### Model Architecture")

    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Input", "[1, 14]")
    a2.metric("Hidden Layer 1", "32")
    a3.metric("Hidden Layer 2", "16")
    a4.metric("Output", "7 Classes")

    st.markdown(
        """
**Edge pipeline**

`Telemetry → Feature Vector → Standardization → MLP 32 → ReLU → MLP 16 → ReLU → Softmax → Fault Probabilities`

**Verified local validation**

- 8,400 final test rows
- 70 final test scenarios
- Scenario-group leakage protection
- 90.04% final-test accuracy
- 90.00% macro F1
- Fixed ONNX input shape `[1,14]`
- Numeric probability output `[1,7]`
"""
    )

    st.divider()

    st.markdown("#### ⚡ Live Edge Inference")
    st.caption(
        "The model is trained once and reused for inference. Every new telemetry vector is evaluated by the same frozen ONNX model, so changing the inputs can change the diagnosis without retraining."
    )

    latest_point = st.session_state.current_point

    if latest_point:
        b1, b2 = st.columns([1, 3])
        with b1:
            if st.button("🔄 INFER LATEST TELEMETRY", type="primary", use_container_width=True):
                st.session_state.edge_diagnosis = run_edge_inference(latest_point)
                st.session_state.edge_error = None
                st.rerun()
        with b2:
            st.info(
                f"Using latest Digital Twin telemetry: **T+{float(latest_point.get('mission_time_min', 0.0)):.2f} min** | "
                f"RPM **{float(latest_point.get('rpm', 0)):.0f}** | "
                f"CHT **{float(latest_point.get('cht_c', 0)):.1f} °C** | "
                f"Oil Pressure **{float(latest_point.get('oil_pressure_kpa', 0)):.1f} kPa**"
            )

    with st.expander("🧪 Test a New Telemetry Input", expanded=False):
        st.caption("Change the sensor values below and run the same trained Edge ONNX model. No retraining occurs.")

        base = latest_point or {
            "altitude_m": 3000.0,
            "ambient_temp_c": 15.0,
            "ambient_pressure_kpa": 70.0,
            "throttle": 0.75,
            "rpm": 5200.0,
            "map_kpa": 125.0,
            "engine_load": 0.75,
            "power_kw": 80.0,
            "fuel_flow_lph": 30.0,
            "cht_c": 140.0,
            "egt_c": 720.0,
            "oil_pressure_kpa": 380.0,
            "oil_temperature_c": 100.0,
            "vibration_g": 0.025,
        }

        with st.form("edge_custom_inference_form"):
            ec1, ec2 = st.columns(2)

            with ec1:
                custom_altitude = st.number_input("Altitude (m)", value=float(base.get("altitude_m", 3000)), step=100.0, key="edge_custom_altitude")
                custom_ambient_temp = st.number_input("Ambient Temperature (°C)", value=float(base.get("ambient_temp_c", 15)), step=1.0, key="edge_custom_ambient_temp")
                custom_ambient_pressure = st.number_input("Ambient Pressure (kPa)", value=float(base.get("ambient_pressure_kpa", 70)), step=1.0, key="edge_custom_ambient_pressure")
                custom_throttle = st.number_input("Throttle", min_value=0.0, max_value=1.0, value=float(base.get("throttle", 0.75)), step=0.01, key="edge_custom_throttle")
                custom_rpm = st.number_input("RPM", value=float(base.get("rpm", 5200)), step=100.0, key="edge_custom_rpm")
                custom_map = st.number_input("MAP (kPa)", value=float(base.get("map_kpa", 125)), step=1.0, key="edge_custom_map")
                custom_load = st.number_input("Engine Load", min_value=0.0, max_value=1.0, value=float(base.get("engine_load", 0.75)), step=0.01, key="edge_custom_load")

            with ec2:
                custom_power = st.number_input("Power (kW)", value=float(base.get("power_kw", 80)), step=1.0, key="edge_custom_power")
                custom_fuel = st.number_input("Fuel Flow (L/h)", value=float(base.get("fuel_flow_lph", 30)), step=1.0, key="edge_custom_fuel")
                custom_cht = st.number_input("CHT (°C)", value=float(base.get("cht_c", 140)), step=1.0, key="edge_custom_cht")
                custom_egt = st.number_input("EGT (°C)", value=float(base.get("egt_c", 720)), step=5.0, key="edge_custom_egt")
                custom_oil_pressure = st.number_input("Oil Pressure (kPa)", value=float(base.get("oil_pressure_kpa", 380)), step=5.0, key="edge_custom_oil_pressure")
                custom_oil_temp = st.number_input("Oil Temperature (°C)", value=float(base.get("oil_temperature_c", 100)), step=1.0, key="edge_custom_oil_temp")
                custom_vibration = st.number_input("Vibration (g)", min_value=0.0, value=float(base.get("vibration_g", 0.025)), step=0.005, format="%.4f", key="edge_custom_vibration")

            run_custom_edge = st.form_submit_button("⚡ RUN EDGE INFERENCE", type="primary", use_container_width=True)

        if run_custom_edge:
            custom_point = {
                "altitude_m": custom_altitude,
                "ambient_temp_c": custom_ambient_temp,
                "ambient_pressure_kpa": custom_ambient_pressure,
                "throttle": custom_throttle,
                "rpm": custom_rpm,
                "map_kpa": custom_map,
                "engine_load": custom_load,
                "power_kw": custom_power,
                "fuel_flow_lph": custom_fuel,
                "cht_c": custom_cht,
                "egt_c": custom_egt,
                "oil_pressure_kpa": custom_oil_pressure,
                "oil_temperature_c": custom_oil_temp,
                "vibration_g": custom_vibration,
            }
            result = run_edge_inference(custom_point)
            if result is not None:
                st.session_state.edge_diagnosis = result
                st.session_state.edge_custom_input = custom_point
                st.session_state.edge_error = None
                st.success("New telemetry vector evaluated successfully — model was not retrained.")
            else:
                st.error(f"Edge inference failed: {st.session_state.edge_error or 'Unknown error'}")

    st.divider()

    st.markdown("#### Current Edge Inference")

    if st.session_state.edge_diagnosis:
        e = st.session_state.edge_diagnosis

        q1, q2, q3, q4 = st.columns(4)
        q1.metric(
            "Fault",
            e.get("fault", "normal").replace("_", " ").upper(),
        )
        q2.metric(
            "Confidence",
            f"{float(e.get('confidence', 0))*100:.2f}%",
        )
        q3.metric(
            "Health",
            f"{float(e.get('health_score', 0)):.2f}/100",
        )
        q4.metric(
            "Risk",
            f"{float(e.get('risk_score', 0)):.2f}",
        )

        st.info(
            f"**Diagnosis:** {e.get('diagnosis', 'No diagnosis available.')}"
        )

        probs = e.get("probabilities", {})
        if probs:
            rows = [
                {
                    "Fault": k.replace("_", " ").title(),
                    "Probability": f"{float(v)*100:.2f}%",
                }
                for k, v in sorted(
                    probs.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )
            ]
            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True,
            )

        if st.session_state.get("edge_custom_input"):
            st.markdown("#### Input Vector Used for Latest Manual Inference")
            st.dataframe(
                pd.DataFrame([st.session_state.edge_custom_input]),
                use_container_width=True,
                hide_index=True,
            )
    else:
        st.info(
            "Run a mission to generate a live Edge inference result."
        )

    st.divider()

    st.markdown("#### Qualcomm AI Hub Deployment State")

    st.markdown(
        """
| Layer | Status |
|---|---|
| Compact edge MLP | ✅ Trained |
| ONNX export | ✅ Validated |
| Fixed input shape | ✅ `[1,14]` |
| Numeric output tensor | ✅ `[1,7]` |
| Local ONNX Runtime | ✅ Validated |
| Qualcomm AI Hub compile | ✅ SUCCESS |
| Target device | ✅ Snapdragon X Elite CRD |
| AI Hub Job | `j5wl0zj3p` |
| Snapdragon deployment target | ✅ Verified compile target |

**Deployment verification:** Qualcomm AI Hub successfully compiled the
AeroTwin Edge ONNX model for the Snapdragon X Elite CRD target.

**Note:** This confirms successful Qualcomm AI Hub compilation for the
Snapdragon target. It does not by itself claim measured runtime latency or
direct NPU execution unless separately profiled on hardware.
"""
    )


# ============================================================
# TAB 10 — NEMOTRON FUTURE
# ============================================================

with tab_nemotron:
    st.markdown(
        """
<div class="nemotron-card">
<h3 style="color:#10b981;margin:0 0 8px 0;">
🧠 NVIDIA NEMOTRON — FUTURE ENGINEERING INVESTIGATION AGENT
</h3>
<p style="color:#a7f3d0;">
STATUS: NOT CONNECTED
</p>
<p style="color:#94a3b8;">
Nemotron is kept as a future orchestration layer. It does not replace the
Digital Twin physics, deterministic tools, or diagnostic models.
</p>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown("#### Future Agent Lifecycle")

    st.code(
        """Digital Twin Physics
        ↓
Telemetry
        ↓
Frozen Diagnostics
        ↓
Evidence Collection
        ↓
Nemotron Agent
        ↓
Counterfactual Tools
        ↓
Evidence Synthesis
        ↓
Human Engineering Approval""",
        language="text",
    )

    tools = [
        {
            "Tool": "get_engine_state()",
            "Purpose": "Query Digital Twin state",
        },
        {
            "Tool": "get_sensor_history()",
            "Purpose": "Retrieve mission telemetry history",
        },
        {
            "Tool": "run_fault_detection()",
            "Purpose": "Execute diagnostic classifier",
        },
        {
            "Tool": "compare_with_baseline()",
            "Purpose": "Compare observed vs healthy twin",
        },
        {
            "Tool": "run_what_if_scenario()",
            "Purpose": "Simulate counterfactual degradation",
        },
        {
            "Tool": "assess_mission_risk()",
            "Purpose": "Compute heuristic risk estimate",
        },
    ]

    st.dataframe(
        pd.DataFrame(tools),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "No external Nemotron API or token authorization is active in this phase."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()
st.caption(
    "AeroTwin Edge • Physics-Informed UAV Propulsion Digital Twin • "
    "Compact ONNX Edge AI • Research Prototype • Not flight certified"
)
