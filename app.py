"""
AEROTWIN-NEMOTRON: Tactical Digital Twin & Mission Assurance Dashboard
Physics-Informed UAV Propulsion Digital Twin with Autonomous Engineering Investigation

Architecture Workflow:
1. MODEL LAB: Generate/Validate Synthetic Benchmark -> 5-Fold Grouped CV -> Model Selection (Macro F1) -> Final Untouched Holdout (20%) -> Freeze Model
2. MISSION SIMULATOR: 60-min / 1200+ sample mission (Gated behind Frozen Model)
3. LIVE TELEMETRY: Multi-channel sensor telemetry trajectories
4. DIAGNOSTICS: Frozen ML Inference & Ranked Fault Hypotheses
5. BASELINE & TRENDS: Pristine Twin Comparison & Early vs Late Window Analysis
6. WHAT-IF: Counterfactual Virtual Experiments
7. MISSION ASSURANCE: Heuristic Mission Risk Quantification (0-100)
8. RAW TELEMETRY: Complete Flight History & CSV Export
9. NEMOTRON — FUTURE: Autonomous Engineering Investigation Agent Architecture
"""

import time
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from src.config import DEFAULT_ENGINE_CONFIG, EngineConfig
from src.mission import MissionSimulator, MISSION_CSV_PATH, REQUIRED_TELEMETRY_COLUMNS
from src.diagnostics import (
    train_fault_classifier,
    predict_fault,
    load_metadata,
    is_model_frozen,
    get_active_model_metadata,
    get_latest_model_path,
    FEATURE_COLUMNS,
    FAULT_CLASSES,
    DEFAULT_DATASET_PATH,
)
from src.baseline import compare_with_baseline
from src.trends import analyze_trends, assess_system_status
from src.risk import assess_mission_risk
from src.tools import run_what_if_scenario, compare_scenarios

# =====================================================================
# PAGE CONFIGURATION & TACTICAL STYLING
# =====================================================================

st.set_page_config(
    page_title="AeroTwin-Nemotron | UAV Propulsion Health",
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
    .status-card {
        padding: 16px 20px;
        border-radius: 8px;
        margin-bottom: 16px;
        border-left: 5px solid;
    }
    .status-NORMAL {
        background-color: #052e16;
        border-color: #22c55e;
        color: #86efac;
    }
    .status-MONITORING {
        background-color: #082f49;
        border-color: #0ea5e9;
        color: #7dd3fc;
    }
    .status-WARNING {
        background-color: #422006;
        border-color: #f59e0b;
        color: #fde047;
    }
    .status-CRITICAL {
        background-color: #450a0a;
        border-color: #ef4444;
        color: #fca5a5;
    }
    .nemotron-card {
        background: linear-gradient(135deg, #091e14 0%, #061510 100%);
        border: 1px dashed #10b981;
        padding: 18px 22px;
        border-radius: 8px;
        margin-top: 15px;
    }
    .model-frozen-banner {
        background: linear-gradient(135deg, #082f49 0%, #0c4a6e 100%);
        border: 1px solid #0284c7;
        padding: 14px 18px;
        border-radius: 8px;
        margin-bottom: 14px;
    }
</style>
"""
st.markdown(TACTICAL_CSS, unsafe_allow_html=True)

# =====================================================================
# SESSION STATE INITIALIZATION
# =====================================================================

defaults = {
    "mission_running": False,
    "mission_complete": False,
    "mission_history": [],
    "current_point": None,
    "simulator": None,
    "diagnosis": None,
    "baseline": None,
    "trends": None,
    "anomaly_status": None,
    "risk": None,
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# If mission history is empty but mission_telemetry.csv exists, load it as cached inference telemetry
if not st.session_state.mission_history and MISSION_CSV_PATH.exists():
    try:
        cached_df = pd.read_csv(MISSION_CSV_PATH)
        if len(cached_df) > 0:
            st.session_state.mission_history = cached_df.to_dict(orient="records")
            st.session_state.current_point = cached_df.iloc[-1].to_dict()
            st.session_state.mission_complete = True
    except Exception:
        pass

# =====================================================================
# HEADER & PROTOCOL BADGES
# =====================================================================

st.title("✈️ AEROTWIN-NEMOTRON")
st.subheader("Physics-Informed UAV Propulsion Digital Twin & Mission Assurance")

st.markdown("""
<div class="badge-bar">
    <span class="badge-tag badge-accent">PHYSICS-INFORMED DIGITAL TWIN</span>
    <span class="badge-tag badge-accent">SYNTHETIC BENCHMARK</span>
    <span class="badge-tag badge-success">LEAKAGE PROTECTED (GROUP-AWARE CV)</span>
    <span class="badge-tag badge-warn">RESEARCH PROTOTYPE</span>
    <span class="badge-tag badge-warn">NOT FLIGHT CERTIFIED</span>
    <span class="badge-tag">TARGET: ROTAX 915 iS-CLASS (104 kW TURBO)</span>
</div>
""", unsafe_allow_html=True)

st.divider()

# Check Frozen Model Status
model_is_frozen = is_model_frozen()
active_meta = get_active_model_metadata()

# =====================================================================
# SIDEBAR: MISSION CONTROLS & GATING
# =====================================================================

st.sidebar.header("🎯 MISSION CONFIGURATION")

alt_val = st.sidebar.slider(
    "Target Cruise Altitude (m)",
    min_value=1000,
    max_value=8000,
    value=3000,
    step=250,
    help="Rotax 915 iS class ceiling: 12,000m. Nominal cruise: 3,000m.",
)

thr_val = st.sidebar.slider(
    "Cruise Throttle Position",
    min_value=0.45,
    max_value=0.90,
    value=0.75,
    step=0.05,
    help="Nominal cruise throttle: 0.75 to 0.85.",
)

dur_val = st.sidebar.slider(
    "Mission Duration (min)",
    min_value=20,
    max_value=60,
    value=60,
    step=10,
    help="Standard mission duration is 60 minutes.",
)

smp_val = st.sidebar.slider(
    "Sampling Rate (samples/min)",
    min_value=10,
    max_value=20,
    value=20,
    step=5,
    help="20 samples/min yields 1,200 samples for a 60-min mission.",
)

total_target_samples = int(dur_val * smp_val)
st.sidebar.caption(f"📊 Target Telemetry: **{total_target_samples:,} samples** (dt = {60.0/smp_val:.1f}s)")

st.sidebar.divider()
st.sidebar.markdown("**Playback Speed Control**")
sim_step_delay = st.sidebar.slider("Batch Update Delay (s)", 0.0, 0.20, 0.02, 0.01)

st.sidebar.info(
    "🛡️ **Autonomous Stress Degradation**\n\n"
    "Faults are NEVER manually injected. The engine begins 100% pristine. "
    "Internal degradation accumulates gradually from operating stress over the flight."
)

st.sidebar.divider()

# Mission Gating Check
if not model_is_frozen:
    st.sidebar.error("⚠️ **MISSION BLOCKED**\n\nTrain and freeze a baseline model in Model Lab before starting mission simulation.")
    start_clicked = st.sidebar.button("🚀 START MISSION", disabled=True, use_container_width=True)
else:
    active_ver = active_meta.get("model_version", "Active Frozen Model") if active_meta else "Active Frozen Model"
    st.sidebar.success(f"🔒 **FROZEN MODEL READY**\n`{active_ver}`")
    col_start, col_reset = st.sidebar.columns(2)
    start_clicked = col_start.button("🚀 START MISSION", type="primary", use_container_width=True)
    reset_clicked = col_reset.button("↻ RESET", use_container_width=True)

    if reset_clicked:
        st.session_state.mission_running = False
        st.session_state.mission_complete = False
        st.session_state.mission_history = []
        st.session_state.current_point = None
        st.session_state.diagnosis = None
        st.session_state.baseline = None
        st.session_state.trends = None
        st.session_state.anomaly_status = None
        st.session_state.risk = None
        st.rerun()

# =====================================================================
# CONTINUOUS MISSION EXECUTION LOGIC (INFERENCE ONLY)
# =====================================================================

if start_clicked and model_is_frozen:
    st.session_state.mission_running = True
    st.session_state.mission_complete = False
    st.session_state.mission_history = []
    st.session_state.current_point = None
    st.session_state.diagnosis = None
    st.session_state.baseline = None
    st.session_state.trends = None
    st.session_state.anomaly_status = None
    st.session_state.risk = None

    sim = MissionSimulator(
        target_altitude_m=alt_val,
        base_throttle=thr_val,
        duration_minutes=dur_val,
        samples_per_minute=smp_val,
        seed=42,
    )
    st.session_state.simulator = sim

# Mission Execution Loop
if st.session_state.mission_running:
    sim: MissionSimulator = st.session_state.simulator
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
        clock_placeholder.markdown(f"### 🔴 MISSION LIVE — Clock: `T+{mins:02d}:{secs:02d}` / `T+{dur_val:02d}:00` ({step_count}/{total_steps} samples)")
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
            sub_df = df_hist[["mission_time_min", "cht_c", "egt_c", "oil_pressure_kpa", "vibration_g"]].copy()
            chart_placeholder.line_chart(sub_df.set_index("mission_time_min"), height=250)

        if sim_step_delay > 0:
            time.sleep(sim_step_delay)

    # Save mission telemetry for inference & downstream analysis ONLY (never training)
    sim.export_csv(MISSION_CSV_PATH)

    st.session_state.mission_running = False
    st.session_state.mission_complete = True

    # Post-Mission Analytics using Frozen ML Model & Deterministic Physics
    final_point = st.session_state.current_point
    st.session_state.baseline = compare_with_baseline(final_point)
    st.session_state.diagnosis = predict_fault(final_point)
    st.session_state.trends = analyze_trends(st.session_state.mission_history)
    st.session_state.anomaly_status = assess_system_status(
        st.session_state.mission_history,
        baseline_result=st.session_state.baseline,
        ml_diagnosis=st.session_state.diagnosis,
    )
    st.session_state.risk = assess_mission_risk(
        telemetry=final_point,
        baseline_result=st.session_state.baseline,
        ml_diagnosis=st.session_state.diagnosis,
        trend_result=st.session_state.trends,
        mission_duration_min=dur_val,
    )
    st.rerun()

# Auto-compute post-mission analytics if loaded from cache
if st.session_state.mission_history and st.session_state.current_point is not None:
    if st.session_state.baseline is None:
        p = st.session_state.current_point
        st.session_state.baseline = compare_with_baseline(p)
        st.session_state.diagnosis = predict_fault(p)
        st.session_state.trends = analyze_trends(st.session_state.mission_history)
        st.session_state.anomaly_status = assess_system_status(
            st.session_state.mission_history,
            st.session_state.baseline,
            st.session_state.diagnosis,
        )
        st.session_state.risk = assess_mission_risk(
            p,
            st.session_state.baseline,
            st.session_state.diagnosis,
            st.session_state.trends,
            mission_duration_min=dur_val,
        )

# =====================================================================
# DASHBOARD TABS & MAIN DISPLAY (EXACT ENGINEERING WORKFLOW ORDER)
# =====================================================================

tab_model_lab, tab_mission, tab_live, tab_diag, tab_baseline, tab_whatif, tab_risk, tab_raw, tab_nemotron = st.tabs([
    "🔬 1. MODEL LAB",
    "🚀 2. MISSION SIMULATOR",
    "📡 3. LIVE TELEMETRY",
    "🔍 4. DIAGNOSTICS",
    "⚖️ 5. BASELINE & TRENDS",
    "🧪 6. WHAT-IF",
    "🛡️ 7. MISSION ASSURANCE",
    "📊 8. RAW TELEMETRY",
    "⚡ 9. NEMOTRON — FUTURE",
])

# ---------------------------------------------------------------------
# TAB 1: MODEL LAB
# ---------------------------------------------------------------------
with tab_model_lab:
    st.markdown("### 🔬 Offline Model Development & Validation Lab")
    st.caption("Physics-informed synthetic benchmark development: Group-aware CV, model comparison, final holdout evaluation, and frozen model registry.")

    meta = load_metadata()
    latest_v = meta.get("latest_version")
    versions = meta.get("versions", {})
    active_record = versions.get(latest_v) if latest_v else None

    # Top Status & Train Button
    col_lab_btn, col_lab_status = st.columns([1.2, 2.8])

    with col_lab_btn:
        st.markdown("**Model Training Trigger**")
        train_button = st.button("🚀 TRAIN BASELINE MODEL", type="primary", use_container_width=True)
        st.caption("Executes full 10-stage pipeline: 80/20 Grouped Split -> 5-Fold Grouped CV -> RF vs HistGradientBoosting -> Selection by CV Macro F1 -> Untouched Holdout Evaluation -> Frozen Model Persistence.")

    with col_lab_status:
        if active_record:
            st.markdown(f"""
            <div class="model-frozen-banner">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <h4 style="margin:0; color:#38bdf8;">ACTIVE / FROZEN MODEL: {active_record.get('model_version', latest_v)}</h4>
                        <p style="margin:4px 0 0 0; font-size:0.85rem; color:#94a3b8;">
                            Architecture: <strong>{active_record.get('selected_model_name', active_record.get('model_type', 'Classifier'))}</strong> &nbsp;|&nbsp;
                            Created: <strong>{active_record.get('timestamp', 'N/A')}</strong>
                        </p>
                    </div>
                    <div>
                        <span class="badge-tag badge-success">STATUS: FROZEN / ACTIVE</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ **No Active Frozen Model.** Click 'TRAIN BASELINE MODEL' to initialize the offline model development pipeline.")

    # Live Training Progress
    if train_button:
        progress_bar = st.progress(0)
        status_text = st.empty()

        def gui_progress(stage, msg):
            pct = stage / 10.0
            progress_bar.progress(pct)
            status_text.markdown(f"**Stage {stage}/10:** {msg}")

        try:
            model, new_metrics = train_fault_classifier(progress_callback=gui_progress)
            progress_bar.progress(1.0)
            status_text.success(f"✅ Model `{new_metrics['model_version']}` trained, verified, and frozen successfully!")
            st.rerun()
        except Exception as e:
            status_text.error(f"❌ Training failed: {str(e)}")

    st.divider()

    # Section 1: Dataset Overview & Scenario Group Splitting
    st.markdown("#### 1️⃣ Dataset & Scenario-Group Leakage Protection")
    col_d1, col_d2, col_d3, col_d4, col_d5 = st.columns(5)

    dataset_rows = active_record.get("total_dataset_size", 42000) if active_record else 42000
    scenarios_total = active_record.get("number_of_scenarios", 350) if active_record else 350
    dev_scenarios = active_record.get("development_scenarios_count", 280) if active_record else 280
    test_scenarios = active_record.get("final_test_scenarios_count", 70) if active_record else 70

    col_d1.metric("Total Dataset Rows", f"{dataset_rows:,}")
    col_d2.metric("Total Scenarios", f"{scenarios_total}")
    col_d3.metric("Observable Features", "14 Channels")
    col_d4.metric("Fault Classes", "7 Classes")
    col_d5.metric("Leakage Protection", "Disjoint Groups")

    st.markdown(f"""
    <div style="background-color:#131c2e; padding:12px 16px; border-radius:6px; border:1px solid #1e293b; margin-top:8px;">
        <span class="badge-tag badge-accent">SPLIT RATIO</span> <strong>80% Development Data ({dev_scenarios} scenarios)</strong> &nbsp;/&nbsp; <strong>20% Untouched Final Holdout ({test_scenarios} scenarios)</strong>
        <br/><span class="badge-tag badge-success" style="margin-top:6px;">GUARANTEE</span> <em>Scenario-group leakage protected: The same scenario_id NEVER appears across training, validation, or final holdout test sets.</em>
        <br/><span class="badge-tag badge-warn" style="margin-top:6px;">LABEL</span> <em>Physics-informed synthetic benchmark dataset — NOT real-world UAV flight data.</em>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    if active_record:
        # Section 2: Model Comparison (Random Forest vs HistGradientBoosting)
        st.markdown("#### 2️⃣ Model Comparison on 5-Fold Grouped CV (Development Set Only)")
        st.caption("Benchmark candidate architectures evaluated strictly across 5 grouped folds on development scenarios.")

        comp_data = active_record.get("model_comparison", [])
        if comp_data:
            df_comp = pd.DataFrame(comp_data)
            display_comp = []
            for r in comp_data:
                display_comp.append({
                    "Candidate Architecture": r.get("candidate_name", "N/A"),
                    "Model Type": r.get("model_type", "N/A"),
                    "CV Macro F1 (Mean ± Std)": f"{r.get('cv_macro_f1_mean', 0.0):.4f} ± {r.get('cv_macro_f1_std', 0.0):.4f}",
                    "CV Accuracy Mean": f"{r.get('cv_accuracy_mean', 0.0)*100:.2f}%",
                    "CV Precision Macro": f"{r.get('cv_precision_macro_mean', 0.0):.4f}",
                    "CV Recall Macro": f"{r.get('cv_recall_macro_mean', 0.0):.4f}",
                    "CV Weighted F1": f"{r.get('cv_weighted_f1_mean', 0.0):.4f}",
                    "Selection Status": "⭐ SELECTED (Highest Mean CV Macro F1)" if r.get("is_selected", False) else "Candidate",
                })
            st.dataframe(pd.DataFrame(display_comp), use_container_width=True, hide_index=True)
            st.info(f"🏆 **Selected Model:** `{active_record.get('selected_model_name', 'Optimal')}` chosen based strictly on highest mean CV Macro F1.")

        # Section 3: 5-Fold Cross-Validation Metrics Table
        st.markdown("#### 3️⃣ 5-Fold Grouped Cross-Validation Details (Selected Model)")
        fold_metrics = active_record.get("selected_cv_fold_metrics", [])
        mean_cv = active_record.get("selected_cv_mean_metrics", {})
        std_cv = active_record.get("selected_cv_std_metrics", {})

        if fold_metrics:
            cv_rows = []
            for fm in fold_metrics:
                cv_rows.append({
                    "Fold": f"Fold {fm.get('fold', 'N/A')}",
                    "Train Scenarios": fm.get("train_scenarios", 224),
                    "Val Scenarios": fm.get("val_scenarios", 56),
                    "Accuracy": f"{fm.get('accuracy', 0.0)*100:.2f}%",
                    "Precision (Macro)": f"{fm.get('precision_macro', 0.0):.4f}",
                    "Recall (Macro)": f"{fm.get('recall_macro', 0.0):.4f}",
                    "Macro F1": f"{fm.get('f1_macro', 0.0):.4f}",
                    "Weighted F1": f"{fm.get('f1_weighted', 0.0):.4f}",
                })
            # Summary rows
            cv_rows.append({
                "Fold": "📊 MEAN",
                "Train Scenarios": "-",
                "Val Scenarios": "-",
                "Accuracy": f"{mean_cv.get('accuracy', 0.0)*100:.2f}%",
                "Precision (Macro)": f"{mean_cv.get('precision_macro', 0.0):.4f}",
                "Recall (Macro)": f"{mean_cv.get('recall_macro', 0.0):.4f}",
                "Macro F1": f"{mean_cv.get('f1_macro', 0.0):.4f}",
                "Weighted F1": f"{mean_cv.get('f1_weighted', 0.0):.4f}",
            })
            cv_rows.append({
                "Fold": "📐 STD DEV",
                "Train Scenarios": "-",
                "Val Scenarios": "-",
                "Accuracy": f"±{std_cv.get('accuracy', 0.0)*100:.2f}%",
                "Precision (Macro)": f"±{std_cv.get('precision_macro', 0.0):.4f}",
                "Recall (Macro)": f"±{std_cv.get('recall_macro', 0.0):.4f}",
                "Macro F1": f"±{std_cv.get('f1_macro', 0.0):.4f}",
                "Weighted F1": f"±{std_cv.get('f1_weighted', 0.0):.4f}",
            })
            st.dataframe(pd.DataFrame(cv_rows), use_container_width=True, hide_index=True)

        st.divider()

        # Section 4: Final Untouched Holdout Test Evaluation
        st.markdown("#### 4️⃣ Final Untouched Holdout Evaluation (20% Test Set)")
        st.caption("Evaluated exactly ONCE on untouched final test scenarios. Never used during model selection or tuning.")

        holdout_m = active_record.get("final_holdout_metrics", {})
        h1, h2, h3, h4, h5 = st.columns(5)
        h1.metric("Final Holdout Accuracy", f"{holdout_m.get('accuracy', 0.0)*100:.2f}%")
        h2.metric("Final Precision (Macro)", f"{holdout_m.get('precision_macro', 0.0):.4f}")
        h3.metric("Final Recall (Macro)", f"{holdout_m.get('recall_macro', 0.0):.4f}")
        h4.metric("Final Macro F1", f"{holdout_m.get('f1_macro', 0.0):.4f}")
        h5.metric("Final Weighted F1", f"{holdout_m.get('f1_weighted', 0.0):.4f}")

        col_cm, col_fi = st.columns(2)
        with col_cm:
            st.markdown("**Holdout Confusion Matrix (7 Fault Classes)**")
            cm_data = active_record.get("confusion_matrix", [])
            class_labels = active_record.get("class_names", FAULT_CLASSES)
            if cm_data and class_labels:
                fig_cm = px.imshow(
                    cm_data,
                    labels=dict(x="Predicted Class", y="Actual Ground Truth", color="Samples"),
                    x=[c.replace("_", " ").title() for c in class_labels],
                    y=[c.replace("_", " ").title() for c in class_labels],
                    text_auto=True,
                    template="plotly_dark",
                    color_continuous_scale="Viridis",
                    height=360,
                )
                fig_cm.update_layout(margin=dict(l=10, r=10, t=20, b=20))
                st.plotly_chart(fig_cm, use_container_width=True)

        with col_fi:
            st.markdown("**Observable Feature Importance**")
            fi_list = active_record.get("feature_importance", [])
            if fi_list:
                df_fi = pd.DataFrame(fi_list)
                fig_fi = px.bar(
                    df_fi,
                    x="importance",
                    y="feature",
                    orientation="h",
                    template="plotly_dark",
                    color="importance",
                    color_continuous_scale="Blues",
                    height=360,
                )
                fig_fi.update_layout(yaxis=dict(autorange="reversed"), margin=dict(l=10, r=10, t=20, b=20))
                st.plotly_chart(fig_fi, use_container_width=True)

        st.divider()

        # Section 5: Model Version Registry
        st.markdown("#### 5️⃣ Model Version Registry & Artifact Catalog")
        hist_rows = []
        for v_name, v_data in versions.items():
            hist_rows.append({
                "Version": v_name,
                "Timestamp": v_data.get("timestamp", "N/A"),
                "Model Architecture": v_data.get("selected_model_name", v_data.get("model_type", "Classifier")),
                "CV Macro F1": f"{v_data.get('selected_cv_mean_metrics', {}).get('f1_macro', 0.0):.4f}",
                "Holdout Accuracy": f"{v_data.get('final_holdout_metrics', {}).get('accuracy', 0.0)*100:.2f}%",
                "Holdout Macro F1": f"{v_data.get('final_holdout_metrics', {}).get('f1_macro', 0.0):.4f}",
                "Status": v_data.get("status", "ACTIVE/FROZEN"),
            })
        st.dataframe(pd.DataFrame(hist_rows), use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------
# TAB 2: MISSION SIMULATOR
# ---------------------------------------------------------------------
with tab_mission:
    st.markdown("### 🚀 Mission Simulator & Execution Hub")
    st.caption("Continuous 60-minute UAV mission simulation with autonomous operating stress degradation.")

    if not model_is_frozen:
        st.error("⛔ **MISSION BLOCKED: No Active Frozen Model Found**\n\nPlease navigate to **Tab 1: MODEL LAB** and train/freeze a baseline model before initiating flight operations.")
    else:
        st.success(f"✅ **System Ready for Flight:** Frozen diagnostic model `{active_meta.get('model_version', 'Active')}` active.")
        col_m1, col_m2 = st.columns([1.5, 2.5])

        with col_m1:
            st.markdown("#### Mission Parameters")
            st.markdown(f"- **Target Cruise Altitude:** `{alt_val:,} m`")
            st.markdown(f"- **Cruise Throttle:** `{thr_val*100:.0f}%`")
            st.markdown(f"- **Duration:** `{dur_val} minutes`")
            st.markdown(f"- **Sampling Rate:** `{smp_val} samples/min`")
            st.markdown(f"- **Total Telemetry Samples:** `{total_target_samples:,} records`")
            st.markdown(f"- **Timestep (dt):** `{60.0/smp_val:.1f} seconds`")

        with col_m2:
            st.markdown("#### Flight Phases Profile")
            st.markdown("""
            1. **0 – 5 min:** Ground Rollout & Initial Climb (300m -> 40% target alt)
            2. **5 – 15 min:** Climb Transition to Cruise Altitude
            3. **15 – 50 min:** Sustained Cruise with realistic atmospheric perturbations
            4. **50 – 60 min:** Return-to-Base / Descent Vector
            """)

        if st.session_state.mission_complete and st.session_state.current_point is not None:
            st.info(f"🏁 **Latest Mission Status:** Completed successfully ({len(st.session_state.mission_history):,} samples recorded in `data/mission_telemetry.csv`).")

# ---------------------------------------------------------------------
# TAB 3: LIVE TELEMETRY
# ---------------------------------------------------------------------
with tab_live:
    if st.session_state.current_point is not None:
        p = st.session_state.current_point
        hist_len = len(st.session_state.mission_history)
        last_t = p.get("mission_time_min", 0.0)

        st.markdown(f"#### 🛰️ Propulsion Telemetry Status — `T+{last_t:.2f} min` ({hist_len:,} samples recorded)")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Altitude", f"{p['altitude_m']:,.0f} m", delta=f"{p['throttle']*100:.0f}% Throttle")
        c2.metric("Engine RPM", f"{p['rpm']:,.0f} RPM", delta="Governed")
        c3.metric("MAP (Boosted)", f"{p['map_kpa']:.1f} kPa", delta=f"{p['power_kw']:.1f} kW Power")
        c4.metric("Fuel Flow", f"{p['fuel_flow_lph']:.1f} L/h", delta=f"{p['engine_load']*100:.0f}% Load")

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("CHT (Cylinder Head)", f"{p['cht_c']:.1f} °C", delta="Base: 140.0°C", delta_color="inverse")
        c6.metric("EGT (Exhaust Gas)", f"{p['egt_c']:.1f} °C", delta="Nominal ~720°C")
        c7.metric("Oil Pressure", f"{p['oil_pressure_kpa']:.1f} kPa", delta="Normal: 380 kPa")
        c8.metric("Vibration", f"{p['vibration_g']:.4f} g", delta="Baseline: 0.025 g", delta_color="inverse")

        st.divider()
        st.markdown("#### 📈 Multi-Channel Mission Trajectories")

        df_h = pd.DataFrame(st.session_state.mission_history)
        col_c1, col_c2 = st.columns(2)

        with col_c1:
            fig_thermal = go.Figure()
            fig_thermal.add_trace(go.Scatter(x=df_h["mission_time_min"], y=df_h["cht_c"], name="CHT (°C)", line=dict(color="#f97316", width=2)))
            fig_thermal.add_trace(go.Scatter(x=df_h["mission_time_min"], y=df_h["oil_temperature_c"], name="Oil Temp (°C)", line=dict(color="#eab308", width=1.5)))
            fig_thermal.update_layout(title="Thermal Response (CHT & Oil Temp)", template="plotly_dark", height=320, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_thermal, use_container_width=True)

        with col_c2:
            fig_lub = go.Figure()
            fig_lub.add_trace(go.Scatter(x=df_h["mission_time_min"], y=df_h["oil_pressure_kpa"], name="Oil Pressure (kPa)", line=dict(color="#38bdf8", width=2)))
            fig_lub.add_trace(go.Scatter(x=df_h["mission_time_min"], y=df_h["power_kw"], name="Power (kW)", line=dict(color="#4ade80", width=1.5)))
            fig_lub.update_layout(title="Lubrication Pressure & Engine Power", template="plotly_dark", height=320, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_lub, use_container_width=True)

        col_c3, col_c4 = st.columns(2)
        with col_c3:
            fig_vibe = go.Figure()
            fig_vibe.add_trace(go.Scatter(x=df_h["mission_time_min"], y=df_h["vibration_g"], name="Vibration (g)", line=dict(color="#ec4899", width=2)))
            fig_vibe.update_layout(title="Mechanical Vibration Spectrum", template="plotly_dark", height=300, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_vibe, use_container_width=True)

        with col_c4:
            fig_egt = go.Figure()
            fig_egt.add_trace(go.Scatter(x=df_h["mission_time_min"], y=df_h["egt_c"], name="EGT (°C)", line=dict(color="#f43f5e", width=2)))
            fig_egt.update_layout(title="Exhaust Gas Temperature Trajectory", template="plotly_dark", height=300, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_egt, use_container_width=True)

    else:
        st.info("No mission currently active. Configure flight parameters in the sidebar and click **'🚀 START MISSION'** to begin.")

# ---------------------------------------------------------------------
# TAB 4: DIAGNOSTICS
# ---------------------------------------------------------------------
with tab_diag:
    if st.session_state.current_point is not None:
        st.markdown("### 🔍 Frozen ML Fault Diagnostics & Hypotheses")
        st.caption("Diagnostic inference generated strictly by the frozen active model on observable telemetry.")

        diag = st.session_state.diagnosis or {}
        pred_fault = diag.get("predicted_fault", "normal")
        conf = diag.get("confidence", 0.0)
        ver_tag = diag.get("model_version", "Active Frozen Model")

        col_d_p1, col_d_p2, col_d_p3 = st.columns(3)
        col_d_p1.metric("Predicted Fault Condition", pred_fault.replace("_", " ").upper())
        col_d_p2.metric("Estimated Probability", f"{conf*100:.1f}%")
        col_d_p3.metric("Diagnostic Model", ver_tag)

        st.divider()

        col_diag_bar, col_diag_table = st.columns([1.5, 1.0])

        with col_diag_bar:
            st.markdown("#### 📊 Posterior Class Probability Distribution")
            ranked = diag.get("ranked_hypotheses", [])
            if ranked:
                df_ranked = pd.DataFrame(ranked)
                df_ranked["probability_pct"] = df_ranked["probability"].apply(lambda x: f"{x*100:.1f}%")
                df_ranked["fault_display"] = df_ranked["fault"].str.replace("_", " ").str.title()
                fig_bar = px.bar(
                    df_ranked,
                    x="probability",
                    y="fault_display",
                    orientation="h",
                    color="probability",
                    color_continuous_scale="Blues",
                    labels={"probability": "Estimated Probability", "fault_display": "Diagnostic Hypothesis"},
                    template="plotly_dark",
                    height=300,
                )
                fig_bar.update_layout(yaxis=dict(autorange="reversed"), showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig_bar, use_container_width=True)

        with col_diag_table:
            st.markdown("#### 📋 Ranked Diagnostic Hypotheses")
            if ranked:
                st.dataframe(pd.DataFrame(ranked), use_container_width=True, hide_index=True)

        st.caption("⚠️ *Note: Predictions represent probabilistic diagnostic hypotheses derived from synthetic benchmark models and do not constitute absolute ground truth.*")
    else:
        st.info("Start a mission to execute frozen ML fault diagnosis.")

# ---------------------------------------------------------------------
# TAB 5: BASELINE & TRENDS
# ---------------------------------------------------------------------
with tab_baseline:
    if st.session_state.baseline:
        base_data = st.session_state.baseline
        st.markdown("### ⚖️ Healthy Twin Baseline & Trajectory Trends")
        st.caption("Deviations computed dynamically using pristine Digital Twin physics at identical altitude & throttle.")

        # Baseline Comparison Table
        dev_list = []
        for param, d in base_data.get("deviations", {}).items():
            dev_list.append({
                "Parameter": param,
                "Observed Telemetry": d["observed"],
                "Healthy Baseline": d["healthy_baseline"],
                "Delta": f"{d['delta']:+.3f}",
                "Deviation %": f"{d['percent_deviation']:+.2f}%",
                "Significant (>3%)": "⚠️ YES" if abs(d["percent_deviation"]) >= 3.0 else "NOMINAL",
            })

        st.dataframe(pd.DataFrame(dev_list), use_container_width=True, hide_index=True)

        st.divider()

        # Trend Analysis Window Details
        if st.session_state.trends and "signals" in st.session_state.trends:
            st.markdown("#### 📊 Early (First 10%) vs Late (Last 10%) Mission Window Comparison")
            trend_rows = []
            for sig, sdata in st.session_state.trends["signals"].items():
                trend_rows.append({
                    "Signal": sig,
                    "Early Window Mean": sdata["early_window_mean"],
                    "Late Window Mean": sdata["late_window_mean"],
                    "Delta": f"{sdata['window_delta']:+.2f}",
                    "Change %": f"{sdata['percent_change']:+.1f}%",
                    "Slope/min": f"{sdata['slope_per_min']:+.4f}",
                    "Persistent?": "YES" if sdata["is_persistent"] else "NO",
                })
            st.dataframe(pd.DataFrame(trend_rows), use_container_width=True, hide_index=True)

        # Supporting Evidence
        status_info = st.session_state.anomaly_status or {"status": "NORMAL", "evidence": []}
        st.markdown("#### 📋 Quantitative Evidence Items")
        evidence_list = status_info.get("evidence", [])
        if evidence_list:
            for ev in evidence_list:
                st.markdown(f"- 🔸 {ev}")
        else:
            st.markdown("- All parameters nominal.")
    else:
        st.info("Execute a mission run to compare telemetry against the healthy baseline.")

# ---------------------------------------------------------------------
# TAB 6: WHAT-IF COUNTERFACTUAL
# ---------------------------------------------------------------------
with tab_whatif:
    st.markdown("### 🧪 Counterfactual What-If Virtual Experiments")
    st.caption("Evaluate engineering hypotheses: Simulate how the Digital Twin behaves under custom degraded health states without modifying the mission state.")

    col_w1, col_w2 = st.columns([1, 2])

    with col_w1:
        st.markdown("**Hypothetical Subsystem Health**")
        hypo_alt = st.number_input("Hypothetical Altitude (m)", 0.0, 10000.0, 3000.0, 250.0)
        hypo_thr = st.slider("Hypothetical Throttle", 0.20, 1.00, 0.75, 0.05)
        hypo_cooling = st.slider("Cooling Health", 0.0, 1.0, 0.75, 0.05)
        hypo_fuel = st.slider("Fuel Delivery Health", 0.0, 1.0, 1.0, 0.05)
        hypo_oil = st.slider("Oil/Lubrication Health", 0.0, 1.0, 1.0, 0.05)
        hypo_bearing = st.slider("Bearing Mechanical Health", 0.0, 1.0, 1.0, 0.05)
        hypo_sensor = st.slider("Sensor Integrity", 0.0, 1.0, 1.0, 0.05)

        run_hypo = st.button("Simulate Hypothesis", type="primary")

    with col_w2:
        if run_hypo or st.session_state.current_point is not None:
            observed_target = st.session_state.current_point
            whatif_res = run_what_if_scenario(
                altitude_m=hypo_alt,
                throttle=hypo_thr,
                cooling_health=hypo_cooling,
                fuel_health=hypo_fuel,
                oil_health=hypo_oil,
                bearing_health=hypo_bearing,
                sensor_health=hypo_sensor,
                target_telemetry=observed_target,
            )

            st.markdown("#### Hypothetical State vs Observed Mission Telemetry")
            if "comparison" in whatif_res:
                comp = whatif_res["comparison"]
                c_s1, c_s2 = st.columns(2)
                c_s1.metric("Hypothesis Match Similarity", f"{comp['similarity_score_percent']:.1f}%")
                c_s2.metric("Normalized MSE", f"{comp['mean_squared_normalized_error']:.4f}")

                st.markdown("**Signal Deltas (Hypothetical - Observed):**")
                st.json(comp["deltas"])

            st.markdown("**Predicted Telemetry Under Hypothesis:**")
            hypo_t = whatif_res["hypothetical_telemetry"]
            st.dataframe(pd.DataFrame([hypo_t]), use_container_width=True)

# ---------------------------------------------------------------------
# TAB 7: MISSION ASSURANCE
# ---------------------------------------------------------------------
with tab_risk:
    st.markdown("### 🛡️ Mission Assurance & Risk Quantification")
    st.caption("Composite propulsion risk score based on thermal stress, lubrication integrity, vibration, and ML fault probability.")

    if st.session_state.risk:
        risk = st.session_state.risk
        r_col1, r_col2, r_col3 = st.columns([1, 1, 2])
        r_col1.metric("Mission Risk Score", f"{risk.get('risk_score', 0.0):.1f} / 100")
        r_col2.metric("Assurance Level", risk.get("risk_level", "LOW"))
        r_col3.info(f"**Operational Advisory:**\n{risk.get('mission_impact', 'Normal envelope')}")

        if "indicators" in risk:
            ind = risk["indicators"]
            st.markdown("#### Contributing Risk Sub-Scores")
            sc1, sc2, sc3, sc4 = st.columns(4)
            sc1.metric("Thermal Stress", f"{ind.get('thermal_stress_score', 0)} / 30")
            sc2.metric("Lubrication Stress", f"{ind.get('lubrication_stress_score', 0)} / 25")
            sc3.metric("Mechanical Vibration", f"{ind.get('vibration_stress_score', 0)} / 25")
            sc4.metric("ML Fault Factor", f"{ind.get('ml_diagnostic_score', 0)} / 20")

        st.warning("⚠️ **Disclaimer:** *Prototype heuristic mission-risk estimate for research purposes only. Not certified for aviation flight safety or operational deployment.*")
    else:
        st.info("Execute a mission run to assess mission risk.")

# ---------------------------------------------------------------------
# TAB 8: RAW TELEMETRY
# ---------------------------------------------------------------------
with tab_raw:
    st.markdown("### 📊 Mission Telemetry History (1200+ Samples)")
    if st.session_state.mission_history:
        df_raw = pd.DataFrame(st.session_state.mission_history)
        st.caption(f"Total Telemetry Records: **{len(df_raw):,} rows** | File: `{MISSION_CSV_PATH}`")

        selected_cols = st.multiselect(
            "Select Columns to Display",
            options=list(df_raw.columns),
            default=list(df_raw.columns)[:10],
        )

        st.dataframe(df_raw[selected_cols] if selected_cols else df_raw, use_container_width=True, height=400)

        csv_data = df_raw.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="💾 Download Mission Telemetry (CSV)",
            data=csv_data,
            file_name="mission_telemetry.csv",
            mime="text/csv",
        )
    else:
        st.info("No mission telemetry generated yet. Run a mission from the sidebar.")

# ---------------------------------------------------------------------
# TAB 9: NEMOTRON — FUTURE
# ---------------------------------------------------------------------
with tab_nemotron:
    st.markdown("""
    <div class="nemotron-card">
        <h3 style="color:#10b981; margin:0 0 8px 0;">⚡ NVIDIA NEMOTRON — AUTONOMOUS ENGINEERING INVESTIGATION AGENT</h3>
        <p style="font-size:0.95rem; color:#a7f3d0; margin-bottom:12px;">
            <strong>STATUS: NOT CONNECTED</strong> &nbsp;|&nbsp;
            <strong>ARCHITECTURE READY:</strong> Local Deterministic Tools + Evidence Synthesis Pipeline
        </p>
        <p style="font-size:0.85rem; color:#94a3b8; line-height:1.6;">
            Nemotron will NOT replace the Digital Twin physics, ML classifier, or risk engine.
            Instead, Nemotron acts as an autonomous orchestrator and engineering investigator that observes anomalies,
            formulates hypotheses, executes counterfactual what-if simulations, compares physical evidence, and generates
            structured recommendations for human approval.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### 🔄 Autonomous Engineering Agentic Lifecycle")
    st.markdown("""
    ```text
    Digital Twin (Physics)
          ↓
    Observable Sensor Telemetry
          ↓
    Frozen ML Diagnostics (Hypotheses)
          ↓
    Deterministic Evidence Collection
          ↓
    NVIDIA Nemotron Agent (Orchestrator)
          ↓
    [ Tool Selection: get_engine_state() | compare_with_baseline() | run_what_if_scenario() | assess_mission_risk() ]
          ↓
    Evidence Synthesis & Root-Cause Explanation
          ↓
    Human Engineering Approval
    ```
    """)

    st.markdown("#### 🛠️ Available Local Deterministic Tools")
    tool_data = [
        {"Tool Function": "get_engine_state()", "Purpose": "Query Digital Twin state for given flight conditions", "Output": "Observable telemetry JSON"},
        {"Tool Function": "get_sensor_history()", "Purpose": "Retrieve sliding window mission telemetry", "Output": "List of telemetry records"},
        {"Tool Function": "run_fault_detection()", "Purpose": "Execute frozen ML diagnostic classifier", "Output": "Ranked fault hypotheses & probabilities"},
        {"Tool Function": "compare_with_baseline()", "Purpose": "Compare observed telemetry against healthy twin", "Output": "Signal deltas and % deviations"},
        {"Tool Function": "run_what_if_scenario()", "Purpose": "Simulate counterfactual degraded twin state", "Output": "Hypothetical telemetry & similarity match"},
        {"Tool Function": "assess_mission_risk()", "Purpose": "Compute heuristic propulsion risk score (0-100)", "Output": "Risk score, level, and impact advisory"},
    ]
    st.dataframe(pd.DataFrame(tool_data), use_container_width=True, hide_index=True)
    st.caption("🔒 *Note: No external API requests, Nebius Token Factory connections, or LLM token authorizations are active in this phase.*")