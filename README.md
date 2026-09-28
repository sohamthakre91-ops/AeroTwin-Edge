# AeroTwin-Nemotron: Physics-Informed UAV Propulsion Digital Twin & Autonomous Engineering Investigation

> **Offline ML Pipeline • 5-Fold Grouped CV • Model Comparison • Untouched Holdout Test • Frozen Model Gating • Autonomous Engineering Investigation**  
> *Target Class: Rotax 915 iS-class Turbocharged UAV Piston Engine (104 kW / 140 hp)*

---

## 1. Project Overview

**AeroTwin-Nemotron** is a physics-informed digital twin and autonomous engineering assurance platform for high-altitude, long-endurance (HALE) and tactical unmanned aerial vehicles (UAVs). The system represents a UAV aero piston engine using a physics-informed digital twin, generates realistic multi-channel sensor telemetry, performs offline machine learning fault diagnosis using scenario-group leakage protection, tracks healthy digital twin baselines, executes counterfactual "what-if" virtual experiments, quantifies mission risk, and provides a deterministic tool foundation for future autonomous investigation by NVIDIA Nemotron.

### Central Architectural Philosophy:
1. **Model First, Mission Second**: Machine learning models must be developed, cross-validated, selected, evaluated on untouched holdouts, and frozen *before* any mission simulation occurs.
2. **Zero Data Leakage**: The dataset is organized into cohesive flight scenarios (`scenario_id`). Group-aware splitting guarantees that the same scenario never appears across training, cross-validation, and final holdout sets.
3. **Mission Simulation is Inference-Only**: Mission telemetry is purely inference and analysis data. It is never automatically labeled "normal" or leaked into the training set.
4. **Frozen Model Gating**: Mission simulation is strictly gated and will not execute unless an active, frozen baseline model is present.

> [!IMPORTANT]
> **Safety & Research Disclaimer**: This software is a physics-informed research prototype. It is **NOT** flight-certified, is **NOT** connected to real aircraft systems, and must **NOT** be used for real-world aircraft operations or flight control decisions. All telemetry and performance metrics represent a **physics-informed synthetic benchmark**, not real-world UAV flight validation.

---

## 2. System Architecture

```text
+-----------------------------------------------------------------------------------+
|                        OFFLINE MODEL DEVELOPMENT PIPELINE                         |
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼
                 +───────────────────────────────────────────────+
                 |    Synthetic Benchmark Dataset (42k rows)     |
                 |      7 Fault Classes • 350 Scenarios          |
                 +───────────────────────┬───────────────────────+
                                         │ Group-Aware Split (Zero Leakage)
                    ┌────────────────────┴────────────────────┐
                    ▼                                         ▼
    +───────────────────────────────+         +───────────────────────────────+
    | 80% Development Data (280 sc) |         | 20% Untouched Holdout (70 sc) |
    +───────────────┬───────────────+         +───────────────┬───────────────+
                    │                                         │ (Locked during CV & Selection)
                    ▼                                         │
    +───────────────────────────────+                         │
    |  5-Fold Grouped CV Benchmark  |                         │
    |  (StratifiedGroupKFold splits)|                         │
    |  • RandomForestClassifier     |                         │
    |  • HistGradientBoosting       |                         │
    +───────────────┬───────────────+                         │
                    │                                         │
                    ▼                                         │
    +───────────────────────────────+                         │
    |     Model Selection Rule      |                         │
    |     Highest Mean CV Macro F1  |                         │
    +───────────────┬───────────────+                         │
                    │                                         │
                    ▼                                         │
    +───────────────────────────────+                         │
    | Retrain Selected Architecture |                         │
    |    on ALL Development Data    |                         │
    +───────────────┬───────────────+                         │
                    │                                         │
                    └────────────────────┬────────────────────┘
                                         ▼
                 +───────────────────────────────────────────────+
                 |       Single Final Holdout Evaluation         |
                 |  Accuracy • Precision • Recall • Macro F1     |
                 |  Confusion Matrix • Feature Importance        |
                 +───────────────────────┬───────────────────────+
                                         │
                                         ▼
                 +───────────────────────────────────────────────+
                 |               FREEZE MODEL                    |
                 |  models/fault_classifier_v{N}.joblib          |
                 |  models/model_metadata_v{N}.json              |
                 +───────────────────────┬───────────────────────+
                                         │
                                         ▼
+-----------------------------------------------------------------------------------+
|                        ONLINE FLIGHT & INFERENCE PIPELINE                         |
+-----------------------------------------------------------------------------------+
                                         │
                                         ▼
                 +───────────────────────────────────────────────+
                 |            Mission Simulator                  |
                 |  60 min • 20 Hz / 1,200 samples • Stress Wear |
                 |  (GATED: Requires Active Frozen ML Model)     |
                 +───────────────────────┬───────────────────────+
                                         │
                                         ▼
                 +───────────────────────────────────────────────+
                 |          Observable Sensor Telemetry          |
                 |        14 Channels (No Hidden Variables)      |
                 +───────┬───────────────┼───────────────┬───────+
                         │               │               │
                         ▼               ▼               ▼
                 +---------------+ +-----------+ +---------------+
                 |   Frozen ML   | |  Healthy  | | Trend & Window|
                 |  Diagnostics  | | Baseline  | |  Persistence  |
                 |  (Hypotheses) | | Deviations| |   Analysis    |
                 +───────┬───────+ +─────┬─────+ +───────┬───────+
                         │               │               │
                         └───────────────┼───────────────┘
                                         │
                                         ▼
                 +───────────────────────────────────────────────+
                 |      Counterfactual What-If Experiments       |
                 |      & Mission Assurance Risk (0-100)         |
                 +───────────────────────┬───────────────────────+
                                         │
                                         ▼
                 +───────────────────────────────────────────────+
                 |       FUTURE: NVIDIA NEMOTRON AGENT           |
                 |  Observe -> Diagnose -> Simulate -> Explain   |
                 |  -> Evidence Synthesis -> Human Approval      |
                 +───────────────────────────────────────────────+
```

---

## 3. Digital Twin Physics & Observable Features

The Digital Twin simulates a Rotax 915 iS-class 104 kW turbocharged aero piston engine across four physical sub-models:

### 3.1 ISA Atmosphere Model (`src/atmosphere.py`)
Calculates ambient temperature $T_{\text{amb}}$, ambient pressure $P_{\text{amb}}$, and air density $\rho$ strictly obeying the tropospheric barometric lapse rate:
$$T_{\text{std}}(h) = T_0 - 0.0065 \cdot h, \quad P_{\text{std}}(h) = P_0 \left(\frac{T_{\text{std}}}{T_0}\right)^{5.2561}, \quad \rho(h) = \frac{P_{\text{std}}}{R \cdot T_{\text{std}}}$$

### 3.2 Turbocharger & Manifold Dynamics (`src/twin.py`)
Dynamic manifold absolute pressure (MAP) compensation with wastegate ceiling limits:
$$\text{MAP} = \text{clamp}\left( P_{\text{amb}} \cdot (0.35 + 0.65 \cdot \text{Throttle}) + P_{\text{boost}} \cdot (0.20 + 0.80 \cdot \text{Throttle}), 25.0\text{ kPa}, 200.0\text{ kPa} \right)$$

### 3.3 Thermal Inertia Dynamics
Cylinder Head Temperature (CHT) and Oil Temperature evolve via first-order thermal inertia:
$$\text{CHT}_t = \text{CHT}_{t-1} + (1 - e^{-\Delta t / \tau_{\text{cht}}}) (\text{CHT}_{\text{target}} - \text{CHT}_{t-1})$$

### 3.4 Observable Sensor Feature Space (14 Channels)
The ML classifier strictly consumes observable instrumentation telemetry. Internal health states are never exposed:
1. `altitude_m`: Barometric altitude (m)
2. `ambient_temp_c`: Ambient air temperature (°C)
3. `ambient_pressure_kpa`: Static atmospheric pressure (kPa)
4. `throttle`: Autopilot throttle command [0.0 - 1.0]
5. `rpm`: Engine crankshaft rotational speed (RPM)
6. `map_kpa`: Manifold absolute pressure (kPa)
7. `engine_load`: Computed engine volumetric load [0.0 - 1.0]
8. `power_kw`: Effective shaft mechanical power (kW)
9. `fuel_flow_lph`: Fuel delivery volume flow (L/h)
10. `cht_c`: Cylinder head temperature (°C)
11. `egt_c`: Exhaust gas temperature (°C)
12. `oil_pressure_kpa`: Engine lubrication line pressure (kPa)
13. `oil_temperature_c`: Sump oil temperature (°C)
14. `vibration_g`: Mechanical vibration spectral amplitude (g)

---

## 4. Synthetic Benchmark Dataset & Group Leakage Protection

The benchmark dataset (`data/telemetry.csv`) contains:
- **Total Records**: 42,000 rows
- **Scenarios**: 350 cohesive flight scenarios (120 timesteps each, $dt = 0.05\text{ min}$)
- **Fault Classes (7 Classes, 50 scenarios / 6,000 rows each)**:
  1. `normal`: Pristine nominal flight operation
  2. `cooling_degradation`: Radiator/duct restriction resulting in elevated CHT
  3. `fuel_restriction`: Fuel line restriction restricting fuel flow and power
  4. `oil_pressure_degradation`: Pump/line wear causing pressure drop and oil heating
  5. `bearing_degradation`: Mechanical bearing race wear causing harmonic vibration growth
  6. `sensor_drift`: Instrumentation bias and drift across CHT, EGT, and oil pressure
  7. `misfire`: Combustion instability causing torque oscillation and power dips

### Group-Aware Splitting Guarantee:
- **80% Development Data**: 280 scenario groups (~33,600 rows)
- **20% Final Holdout Test Data**: 70 scenario groups (~8,400 rows)
$$\text{Scenarios}_{\text{dev}} \cap \text{Scenarios}_{\text{holdout}} = \emptyset$$
$$\text{For every CV fold } k: \quad \text{Scenarios}_{\text{train}, k} \cap \text{Scenarios}_{\text{val}, k} = \emptyset$$

---

## 5. Offline Model Development & Comparison

Candidate architectures evaluated across 5-Fold Grouped Cross-Validation on development data:
- **Model A Candidates (`RandomForestClassifier`)**:
  - RF-1: 150 trees, max depth 10, min samples leaf 3
  - RF-2: 250 trees, max depth 14, min samples leaf 3
  - RF-3: 250 trees, max depth None, min samples leaf 5
- **Model B Candidates (`HistGradientBoostingClassifier`)**:
  - HGB-1: 100 iterations, max depth 8, min samples leaf 20, learning rate 0.10
  - HGB-2: 150 iterations, max depth 12, min samples leaf 15, learning rate 0.08

### Model Selection Metric:
Primary metric: **CV Macro F1** (Mean across 5 folds).  
The best candidate configuration is selected strictly on development-set CV performance. The 20% holdout test set is evaluated exactly once after retraining the winner on all 80% development data.

---

## 6. Model Versioning & Frozen Lifecycle

Models are serialized under `models/`:
- `models/fault_classifier_v{N}.joblib`: Frozen versioned artifact
- `models/model_metadata_v{N}.json`: Standalone version metadata
- `models/fault_classifier.joblib`: Active model pointer
- `models/model_metadata.json`: Central version catalog
- `models/model_registry.json`: Index catalog

### Model Metadata Schema:
- `model_version`, `timestamp`, `status: ACTIVE/FROZEN`, `is_frozen: true`
- `selected_model_name`, `model_type`, `hyperparameters`
- `total_dataset_size`, `number_of_scenarios`, `development_size`, `final_test_size`
- `number_of_cv_folds: 5`, `cv_strategy: StratifiedGroupKFold`
- `model_comparison`: Candidate rankings and scores
- `selected_cv_fold_metrics`: Per-fold accuracy, precision, recall, macro F1, weighted F1
- `selected_cv_mean_metrics`, `selected_cv_std_metrics`
- `final_holdout_metrics`: Final holdout accuracy, precision, recall, macro F1, weighted F1
- `classification_report`, `confusion_matrix`, `feature_importance`

---

## 7. Mission Simulation & Mission Gating

- **Mission Duration**: 60 minutes
- **Sampling Rate**: 20 samples/min ($dt = 3.0\text{ s}$) $\rightarrow$ 1,200 telemetry samples
- **Autonomous Stress Degradation**: No manual fault dropdowns. Internal degradation accumulates smoothly based on operating stress (throttle, RPM, altitude, temperature).
- **Mission Gating**: Mission simulation is blocked if no valid frozen model exists in `models/`.
- **Inference Only**: Mission telemetry is recorded in `data/mission_telemetry.csv` and used exclusively for inference, trend analysis, baseline comparison, and risk quantification.

---

## 8. Analytical Engines & Deterministic Tools

### 8.1 Healthy Baseline Comparator (`src/baseline.py`)
Compares observed telemetry against a pristine Digital Twin running at identical altitude and throttle. Flags parameter deviations exceeding $\pm 3\%$.

### 8.2 Trend & Window Analysis Engine (`src/trends.py`)
Calculates least-squares slopes and compares the early flight window (first 10%) vs. late flight window (last 10%).

### 8.3 Counterfactual What-If Explorer (`src/tools.py`)
Executes virtual experiments (e.g. hypothetical cooling health degradation) and compares hypothetical states with observed telemetry without modifying mission state.

### 8.4 Mission Assurance & Risk Engine (`src/risk.py`)
Computes a heuristic propulsion risk score (0-100) combining thermal stress (30%), lubrication stress (25%), vibration stress (25%), and ML diagnostic factor (20%).

---

## 9. Future NVIDIA Nemotron Agent Architecture

NVIDIA Nemotron will not replace the Digital Twin physics, ML classifier, or risk engine. Instead, Nemotron serves as an **Autonomous Engineering Investigation Agent** that orchestrates tools:

```text
Observe Anomaly
      ↓
Diagnose Probabilities (Frozen ML)
      ↓
Collect Physical Evidence (Baseline & Trends)
      ↓
Formulate Investigation Hypotheses
      ↓
Execute Counterfactual Simulations (What-If)
      ↓
Synthesize Engineering Explanation
      ↓
Submit Structured Advisory for Human Approval
```

Deterministic tools available for Nemotron:
- `get_engine_state(altitude_m, throttle, ...)`
- `get_sensor_history(limit)`
- `run_fault_detection(telemetry)`
- `compare_with_baseline(telemetry)`
- `run_what_if_scenario(altitude_m, throttle, ...)`
- `assess_mission_risk(telemetry, ...)`

---

## 10. Installation, Testing & Running

### 10.1 Running Tests
```powershell
python -m unittest discover tests -v
```
All 19 unit tests verify:
- Single scenario generation & dataset integrity (42k rows, 7 classes)
- Zero scenario-group leakage in 80/20 split
- Zero scenario-group leakage in all 5 CV folds
- Model comparison & selection by CV Macro F1
- Retraining on dev set & single final holdout evaluation
- Model versioning, metadata schema, and frozen inference
- Mission simulation (1,200 samples) & mission gating
- Tool outputs & counterfactual comparison

### 10.2 Launch Streamlit Application
```powershell
streamlit run app.py
```
Open browser at `http://localhost:8501`.

### Navigation Workflow:
1. **1. MODEL LAB**: Inspect synthetic benchmark dataset, view 5-fold CV results, compare RF vs HistGradientBoosting, review final holdout test evaluation, and train/freeze models.
2. **2. MISSION SIMULATOR**: Configure and launch 60-min autonomous mission (gated by frozen model).
3. **3. LIVE TELEMETRY**: View live multi-channel instrumentation gauges and real-time trajectories.
4. **4. DIAGNOSTICS**: Inspect frozen ML inference, confidence, and posterior class probability distributions.
5. **5. BASELINE & TRENDS**: Compare observed telemetry against healthy twin baseline and early vs late windows.
6. **6. WHAT-IF**: Run counterfactual degraded experiments.
7. **7. MISSION ASSURANCE**: Quantify heuristic propulsion risk (0-100).
8. **8. RAW TELEMETRY**: View and export 1,200+ mission samples.
9. **9. NEMOTRON — FUTURE**: Review agentic investigation architecture and deterministic tool interfaces.
