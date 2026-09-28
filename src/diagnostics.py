"""
AeroTwin-Nemotron: Machine Learning Fault Diagnostics Engine
Offline Model Development Pipeline:
1. Synthetic dataset validation (42k rows, 14 observable features, 7 fault classes)
2. Group-aware 80% Development / 20% Final Holdout Test split
3. 5-Fold Grouped Cross-Validation (StratifiedGroupKFold / GroupKFold)
4. Model comparison (Random Forest vs HistGradientBoosting)
5. Model selection strictly via development CV Macro F1
6. Retrain selected candidate on all development data
7. Single final evaluation on untouched 20% holdout test
8. Model versioning and frozen active artifact persistence
"""

from pathlib import Path
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List, Callable
import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
from sklearn.inspection import permutation_importance

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_DATASET_PATH = DATA_DIR / "telemetry.csv"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
REGISTRY_PATH = MODELS_DIR / "model_registry.json"

FEATURE_COLUMNS = [
    "altitude_m",
    "ambient_temp_c",
    "ambient_pressure_kpa",
    "throttle",
    "rpm",
    "map_kpa",
    "engine_load",
    "power_kw",
    "fuel_flow_lph",
    "cht_c",
    "egt_c",
    "oil_pressure_kpa",
    "oil_temperature_c",
    "vibration_g",
]

TARGET_COLUMN = "fault"
GROUP_COLUMN = "scenario_id"

FAULT_CLASSES = [
    "bearing_degradation",
    "cooling_degradation",
    "fuel_restriction",
    "misfire",
    "normal",
    "oil_pressure_degradation",
    "sensor_drift",
]

DEFAULT_MODEL_CANDIDATES = [
    {
        "name": "Random Forest (150 trees, depth 10, leaf 3)",
        "model_type": "RandomForestClassifier",
        "hyperparameters": {
            "n_estimators": 150,
            "max_depth": 10,
            "min_samples_leaf": 3,
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": 1,
        },
    },
    {
        "name": "Random Forest (250 trees, depth 14, leaf 3)",
        "model_type": "RandomForestClassifier",
        "hyperparameters": {
            "n_estimators": 250,
            "max_depth": 14,
            "min_samples_leaf": 3,
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": 1,
        },
    },
    {
        "name": "Random Forest (250 trees, depth None, leaf 5)",
        "model_type": "RandomForestClassifier",
        "hyperparameters": {
            "n_estimators": 250,
            "max_depth": None,
            "min_samples_leaf": 5,
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": 1,
        },
    },
    {
        "name": "HistGradientBoosting (100 iter, depth 8, leaf 20, lr 0.10)",
        "model_type": "HistGradientBoostingClassifier",
        "hyperparameters": {
            "max_iter": 100,
            "max_depth": 8,
            "min_samples_leaf": 20,
            "learning_rate": 0.10,
            "class_weight": "balanced",
            "random_state": 42,
        },
    },
    {
        "name": "HistGradientBoosting (150 iter, depth 12, leaf 15, lr 0.08)",
        "model_type": "HistGradientBoostingClassifier",
        "hyperparameters": {
            "max_iter": 150,
            "max_depth": 12,
            "min_samples_leaf": 15,
            "learning_rate": 0.08,
            "class_weight": "balanced",
            "random_state": 42,
        },
    },
]


def load_metadata() -> Dict[str, Any]:
    """Load model version registry from models/model_metadata.json."""
    if METADATA_PATH.exists():
        try:
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"latest_version": None, "versions": {}}
    return {"latest_version": None, "versions": {}}


def save_metadata(metadata: Dict[str, Any]):
    """Save model version registry to models/model_metadata.json and model_registry.json."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Maintain lightweight registry index
    registry = {
        "latest_version": metadata.get("latest_version"),
        "total_versions": len(metadata.get("versions", {})),
        "versions": list(metadata.get("versions", {}).keys()),
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)


def get_latest_model_path() -> Optional[Path]:
    """Return path to latest active/frozen model artifact."""
    meta = load_metadata()
    latest_ver = meta.get("latest_version")
    if latest_ver:
        path = MODELS_DIR / f"{latest_ver}.joblib"
        if path.exists():
            return path

    default_joblib = MODELS_DIR / "fault_classifier.joblib"
    if default_joblib.exists():
        return default_joblib

    v1_joblib = MODELS_DIR / "fault_classifier_v1.joblib"
    if v1_joblib.exists():
        return v1_joblib

    return None


def is_model_frozen() -> bool:
    """Return True if an active frozen model artifact and metadata exist."""
    path = get_latest_model_path()
    if path and path.exists():
        meta = load_metadata()
        latest_ver = meta.get("latest_version")
        if latest_ver and latest_ver in meta.get("versions", {}):
            return meta["versions"][latest_ver].get("is_frozen", True)
        return True
    return False


def get_active_model_metadata() -> Optional[Dict[str, Any]]:
    """Return metadata dictionary for the currently active frozen model."""
    meta = load_metadata()
    latest_ver = meta.get("latest_version")
    if latest_ver and "versions" in meta and latest_ver in meta["versions"]:
        return meta["versions"][latest_ver]
    return None


def create_model_instance(model_type: str, hyperparameters: Dict[str, Any]):
    """Instantiate classifier model based on type and hyperparameters."""
    if model_type == "RandomForestClassifier":
        return RandomForestClassifier(**hyperparameters)
    elif model_type == "HistGradientBoostingClassifier":
        return HistGradientBoostingClassifier(**hyperparameters)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Calculate accuracy, macro precision, macro recall, macro F1, and weighted F1."""
    acc = float(accuracy_score(y_true, y_pred))
    p_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    r_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1_mac = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_wt = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    return {
        "accuracy": round(acc, 4),
        "precision_macro": round(p_macro, 4),
        "recall_macro": round(r_macro, 4),
        "f1_macro": round(f1_mac, 4),
        "f1_weighted": round(f1_wt, 4),
    }


def evaluate_candidate_cv(
    candidate: Dict[str, Any],
    X_dev: pd.DataFrame,
    y_dev: pd.Series,
    groups_dev: pd.Series,
    n_splits: int = 5,
    classes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Perform 5-fold grouped cross-validation on development data.
    Uses StratifiedGroupKFold when available, falling back to GroupKFold.
    Guarantees train_groups ∩ val_groups == empty for each fold.
    """
    try:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
        splits = list(splitter.split(X_dev, y_dev, groups=groups_dev))
        cv_method = f"StratifiedGroupKFold(n_splits={n_splits})"
    except Exception:
        splitter = GroupKFold(n_splits=n_splits)
        splits = list(splitter.split(X_dev, y_dev, groups=groups_dev))
        cv_method = f"GroupKFold(n_splits={n_splits})"

    fold_results: List[Dict[str, Any]] = []

    for fold_idx, (train_idx, val_idx) in enumerate(splits, start=1):
        # Explicit group leakage assertion
        tr_scenarios = set(groups_dev.iloc[train_idx].unique())
        val_scenarios = set(groups_dev.iloc[val_idx].unique())
        assert len(tr_scenarios.intersection(val_scenarios)) == 0, (
            f"Fold {fold_idx} CV Data Leakage: train and validation scenarios overlap!"
        )

        X_tr, y_tr = X_dev.iloc[train_idx], y_dev.iloc[train_idx]
        X_val, y_val = X_dev.iloc[val_idx], y_dev.iloc[val_idx]

        model = create_model_instance(candidate["model_type"], candidate["hyperparameters"])
        model.fit(X_tr, y_tr)

        y_pred = model.predict(X_val)
        m = compute_metrics(y_val.to_numpy(), y_pred, classes=classes)
        m["fold"] = fold_idx
        m["train_scenarios"] = len(tr_scenarios)
        m["val_scenarios"] = len(val_scenarios)
        m["train_samples"] = len(X_tr)
        m["val_samples"] = len(X_val)
        fold_results.append(m)

    # Compute mean and standard deviation
    metric_keys = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted"]
    mean_metrics = {}
    std_metrics = {}

    for k in metric_keys:
        vals = [f[k] for f in fold_results]
        mean_metrics[k] = round(float(np.mean(vals)), 4)
        std_metrics[k] = round(float(np.std(vals)), 4)

    return {
        "candidate_name": candidate["name"],
        "model_type": candidate["model_type"],
        "hyperparameters": candidate["hyperparameters"],
        "cv_method": cv_method,
        "n_splits": n_splits,
        "fold_metrics": fold_results,
        "mean_metrics": mean_metrics,
        "std_metrics": std_metrics,
        "cv_macro_f1_mean": mean_metrics["f1_macro"],
        "cv_macro_f1_std": std_metrics["f1_macro"],
    }


def train_fault_classifier(
    dataset_path: Optional[Path] = None,
    dataframe: Optional[pd.DataFrame] = None,
    progress_callback: Optional[Callable[[int, str], None]] = None,
    candidates: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Executes the strict 10-stage offline model development lifecycle:
    Stage 1: Validate and load synthetic dataset (~42k rows, 14 features, 7 classes)
    Stage 2: Scenario-group split: 80% Development / 20% Final Holdout Test
    Stage 3: Setup candidate models (Random Forest and HistGradientBoosting variants)
    Stage 4: Run 5-fold grouped CV on candidate models using development set only
    Stage 5: Compare candidate CV Macro F1 scores (mean & std) and select optimal configuration
    Stage 6: Retrain selected model on ALL 80% development data
    Stage 7: Evaluate selected model ONCE on untouched 20% final test set
    Stage 8: Extract holdout metrics, confusion matrix, classification report, feature importances
    Stage 9: Save versioned model artifact, metadata, and active frozen pointer
    Stage 10: Mark model ACTIVE/FROZEN and return
    """
    def notify(stage: int, message: str):
        if progress_callback:
            progress_callback(stage, message)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    candidate_list = candidates or DEFAULT_MODEL_CANDIDATES

    # Stage 1: Loading & validating dataset
    notify(1, "Validating & loading physics-informed synthetic benchmark dataset...")
    if dataframe is not None:
        df = dataframe.copy()
    else:
        path = dataset_path or DEFAULT_DATASET_PATH
        if not path.exists():
            raise FileNotFoundError(
                f"Synthetic training dataset not found at {path}. "
                "Generate synthetic dataset before training."
            )
        df = pd.read_csv(path)

    # Validate feature columns and target
    missing = [c for c in FEATURE_COLUMNS + [TARGET_COLUMN] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required dataset columns: {missing}")

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()
    classes = sorted(list(y.unique()))

    # Group identifier for scenario-leakage prevention
    if GROUP_COLUMN in df.columns:
        groups = df[GROUP_COLUMN]
    else:
        # Fallback grouping by contiguous 120-step scenario blocks
        groups = pd.Series(range(len(df)), index=df.index) // 120

    total_scenarios = int(groups.nunique())
    total_samples = len(df)

    # Stage 2: Group-aware 80% Development / 20% Final Holdout Split
    notify(2, f"Creating group-aware 80/20 split across {total_scenarios} scenarios (zero leakage)...")
    try:
        outer_splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        dev_idx, test_idx = next(outer_splitter.split(X, y, groups=groups))
    except Exception:
        outer_splitter = GroupKFold(n_splits=5)
        dev_idx, test_idx = next(outer_splitter.split(X, y, groups=groups))

    X_dev, y_dev, groups_dev = X.iloc[dev_idx], y.iloc[dev_idx], groups.iloc[dev_idx]
    X_test, y_test, groups_test = X.iloc[test_idx], y.iloc[test_idx], groups.iloc[test_idx]

    # Explicit assertion: Zero scenario leakage between development and final test sets
    dev_scenarios = set(groups_dev.unique())
    test_scenarios = set(groups_test.unique())
    assert len(dev_scenarios.intersection(test_scenarios)) == 0, (
        "CRITICAL ERROR: Scenario overlap detected between development and final test sets!"
    )

    dev_size = len(X_dev)
    test_size = len(X_test)

    # Stage 3: Setup candidate models
    notify(3, f"Configuring {len(candidate_list)} candidate models for comparison...")

    # Stage 4: 5-Fold Grouped Cross-Validation on Development Data
    notify(4, f"Executing 5-fold grouped cross-validation on {len(dev_scenarios)} development scenarios...")
    candidate_cv_results: List[Dict[str, Any]] = []

    for i, cand in enumerate(candidate_list, start=1):
        notify(4, f"Cross-validating Candidate {i}/{len(candidate_list)}: {cand['name']}...")
        cv_res = evaluate_candidate_cv(cand, X_dev, y_dev, groups_dev, n_splits=5, classes=classes)
        candidate_cv_results.append(cv_res)

    # Stage 5: Model Selection based strictly on mean CV Macro F1
    notify(5, "Selecting optimal configuration based on highest mean CV Macro F1...")
    # Sort candidate results by cv_macro_f1_mean descending
    candidate_cv_results.sort(key=lambda x: x["cv_macro_f1_mean"], reverse=True)
    best_candidate_cv = candidate_cv_results[0]
    best_candidate_name = best_candidate_cv["candidate_name"]
    best_model_type = best_candidate_cv["model_type"]
    best_hyperparameters = best_candidate_cv["hyperparameters"]

    # Stage 6: Retrain selected model on ALL development data (80%)
    notify(6, f"Retraining selected model '{best_candidate_name}' on all {dev_size:,} development samples...")
    final_model = create_model_instance(best_model_type, best_hyperparameters)
    final_model.fit(X_dev, y_dev)

    # Stage 7: Evaluate ONCE on untouched 20% final test set
    notify(7, f"Evaluating selected model ONCE on untouched final test set ({test_size:,} samples)...")
    y_test_pred = final_model.predict(X_test)

    holdout_metrics = compute_metrics(y_test.to_numpy(), y_test_pred, classes=classes)
    holdout_report = classification_report(
        y_test,
        y_test_pred,
        labels=classes,
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )
    holdout_cm = confusion_matrix(y_test, y_test_pred, labels=classes).tolist()

    # Stage 8: Extract feature importance
    notify(8, "Computing feature importances on selected model...")
    feature_importances: List[Dict[str, Any]] = []
    if hasattr(final_model, "feature_importances_"):
        raw_imp = final_model.feature_importances_
        feature_importances = [
            {"feature": feat, "importance": round(float(imp), 4)}
            for feat, imp in sorted(zip(FEATURE_COLUMNS, raw_imp), key=lambda x: x[1], reverse=True)
        ]
    else:
        try:
            perm = permutation_importance(
                final_model, X_test, y_test, n_repeats=3, random_state=42, n_jobs=1
            )
            feature_importances = [
                {"feature": feat, "importance": round(float(imp), 4)}
                for feat, imp in sorted(
                    zip(FEATURE_COLUMNS, perm.importances_mean), key=lambda x: x[1], reverse=True
                )
            ]
        except Exception:
            feature_importances = [{"feature": f, "importance": 0.0} for f in FEATURE_COLUMNS]

    # Stage 9: Model versioning & artifact saving
    notify(9, "Saving versioned model artifact and metadata registry...")
    metadata = load_metadata()
    existing_versions = metadata.get("versions", {})
    version_num = len(existing_versions) + 1
    version_tag = f"fault_classifier_v{version_num}"

    version_model_path = MODELS_DIR / f"{version_tag}.joblib"
    active_model_path = MODELS_DIR / "fault_classifier.joblib"
    version_metadata_path = MODELS_DIR / f"model_metadata_v{version_num}.json"

    artifact = {
        "model": final_model,
        "features": FEATURE_COLUMNS,
        "classes": classes,
        "version": version_tag,
        "model_type": best_model_type,
        "hyperparameters": best_hyperparameters,
        "is_frozen": True,
        "status": "ACTIVE/FROZEN",
        "cv_mean_macro_f1": best_candidate_cv["cv_macro_f1_mean"],
        "final_holdout_accuracy": holdout_metrics["accuracy"],
        "final_holdout_macro_f1": holdout_metrics["f1_macro"],
        "final_holdout_weighted_f1": holdout_metrics["f1_weighted"],
    }

    joblib.dump(artifact, version_model_path)
    joblib.dump(artifact, active_model_path)

    # Compile comprehensive metadata record
    version_record = {
        "model_version": version_tag,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "ACTIVE/FROZEN",
        "is_frozen": True,
        "selected_model_name": best_candidate_name,
        "model_type": best_model_type,
        "hyperparameters": best_hyperparameters,
        "dataset_description": "Physics-informed synthetic benchmark",
        "leakage_protection": "Scenario-group leakage protected (train/val/test scenario sets disjoint)",
        "random_state": 42,
        "total_dataset_size": total_samples,
        "number_of_scenarios": total_scenarios,
        "development_size": dev_size,
        "development_scenarios_count": len(dev_scenarios),
        "development_percentage": round((dev_size / total_samples) * 100.0, 1),
        "final_test_size": test_size,
        "final_test_scenarios_count": len(test_scenarios),
        "final_test_percentage": round((test_size / total_samples) * 100.0, 1),
        "number_of_cv_folds": 5,
        "cv_strategy": best_candidate_cv.get("cv_method", "StratifiedGroupKFold(n_splits=5)"),
        "primary_selection_metric": "CV Macro F1",
        "model_comparison": [
            {
                "candidate_name": c["candidate_name"],
                "model_type": c["model_type"],
                "hyperparameters": c["hyperparameters"],
                "cv_macro_f1_mean": c["cv_macro_f1_mean"],
                "cv_macro_f1_std": c["cv_macro_f1_std"],
                "cv_accuracy_mean": c["mean_metrics"]["accuracy"],
                "cv_precision_macro_mean": c["mean_metrics"]["precision_macro"],
                "cv_recall_macro_mean": c["mean_metrics"]["recall_macro"],
                "cv_weighted_f1_mean": c["mean_metrics"]["f1_weighted"],
                "is_selected": (c["candidate_name"] == best_candidate_name),
            }
            for c in candidate_cv_results
        ],
        "selected_cv_fold_metrics": best_candidate_cv["fold_metrics"],
        "selected_cv_mean_metrics": best_candidate_cv["mean_metrics"],
        "selected_cv_std_metrics": best_candidate_cv["std_metrics"],
        "final_holdout_metrics": holdout_metrics,
        "classification_report": holdout_report,
        "confusion_matrix": holdout_cm,
        "feature_importance": feature_importances,
        "feature_columns": FEATURE_COLUMNS,
        "class_names": classes,
    }

    # Save per-version standalone metadata JSON
    with open(version_metadata_path, "w", encoding="utf-8") as f:
        json.dump(version_record, f, indent=2)

    # Update main metadata file
    metadata["latest_version"] = version_tag
    if "versions" not in metadata:
        metadata["versions"] = {}
    metadata["versions"][version_tag] = version_record
    save_metadata(metadata)

    # Stage 10: Complete and freeze
    notify(10, f"Model {version_tag} successfully frozen and activated for mission inference.")
    return final_model, version_record


def predict_fault(
    telemetry: Dict[str, Any],
    model_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Run inference on observable telemetry dictionary using the frozen active ML model.
    Mission simulation must ONLY load this frozen model and never modify/retrain it.

    Returns:
        predicted_fault: class label
        confidence: probability [0.0 - 1.0]
        fault_probabilities: dict mapping fault -> probability
        ranked_hypotheses: sorted list of {fault, probability}
        model_version: active model version string
        is_frozen: bool
        status: status label
    """
    path = model_path or get_latest_model_path()
    if not path or not path.exists():
        return {
            "error": "No trained and frozen diagnostic model artifact available. Train a baseline model in Model Lab first.",
            "predicted_fault": "unknown",
            "confidence": 0.0,
            "fault_probabilities": {},
            "ranked_hypotheses": [],
            "model_version": "none",
            "is_frozen": False,
        }

    try:
        loaded = joblib.load(path)
        if isinstance(loaded, dict) and "model" in loaded:
            model = loaded["model"]
            features = loaded.get("features", FEATURE_COLUMNS)
            classes = loaded.get("classes", list(model.classes_))
            version_str = loaded.get("version", "frozen_active_v1")
            is_frozen_flag = loaded.get("is_frozen", True)
        else:
            model = loaded
            features = FEATURE_COLUMNS
            classes = list(model.classes_)
            version_str = getattr(model, "version", "fault_classifier_active")
            is_frozen_flag = True

        # Build feature vector for observable signals
        row = []
        for feat in features:
            val = telemetry.get(feat, 0.0)
            try:
                row.append(float(val))
            except (ValueError, TypeError):
                row.append(0.0)

        X_in = pd.DataFrame([row], columns=features)
        pred_class = model.predict(X_in)[0]

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_in)[0]
        elif hasattr(model, "decision_function"):
            raw_scores = model.decision_function(X_in)[0]
            exp_scores = np.exp(raw_scores - np.max(raw_scores))
            probs = exp_scores / np.sum(exp_scores)
        else:
            probs = np.zeros(len(classes))
            if pred_class in classes:
                probs[classes.index(pred_class)] = 1.0

        prob_map = {str(c): float(p) for c, p in zip(classes, probs)}
        ranked = sorted(
            [{"fault": c, "probability": round(float(p), 4)} for c, p in prob_map.items()],
            key=lambda x: x["probability"],
            reverse=True,
        )
        conf = prob_map.get(str(pred_class), 0.0)

        return {
            "predicted_fault": str(pred_class),
            "confidence": round(float(conf), 4),
            "fault_probabilities": prob_map,
            "ranked_hypotheses": ranked,
            "model_version": version_str,
            "is_frozen": is_frozen_flag,
            "status": "ACTIVE/FROZEN",
        }

    except Exception as exc:
        return {
            "error": f"Diagnostic inference failed: {str(exc)}",
            "predicted_fault": "unknown",
            "confidence": 0.0,
            "fault_probabilities": {},
            "ranked_hypotheses": [],
            "model_version": "error",
            "is_frozen": False,
        }