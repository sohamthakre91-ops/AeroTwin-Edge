from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report


ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "model_training_data.csv"
MODEL_DIR = ROOT / "models"

EDGE_MODEL_PATH = MODEL_DIR / "aerotwin_edge_mlp.joblib"
EDGE_METADATA_PATH = MODEL_DIR / "aerotwin_edge_metadata.json"


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


def main():
    print("=" * 60)
    print("AeroTwin Edge - Neural Edge Model Training")
    print("=" * 60)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    # ---------------------------------------------------------
    # 1. Load existing physics-informed dataset
    # ---------------------------------------------------------
    df = pd.read_csv(DATA_PATH)

    print(f"\nDataset rows : {len(df):,}")
    print(f"Scenarios    : {df[GROUP_COLUMN].nunique():,}")
    print(f"Fault classes: {df[TARGET_COLUMN].nunique()}")

    missing = [
        c
        for c in FEATURE_COLUMNS + [TARGET_COLUMN, GROUP_COLUMN]
        if c not in df.columns
    ]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    X = df[FEATURE_COLUMNS].astype(np.float32)
    y = df[TARGET_COLUMN]
    groups = df[GROUP_COLUMN]

    classes = sorted(y.unique().tolist())

    print("\nClasses:")
    for cls in classes:
        print(f"  - {cls}")

    # ---------------------------------------------------------
    # 2. Scenario-aware 80/20 split
    # ---------------------------------------------------------
    splitter = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    dev_idx, test_idx = next(
        splitter.split(X, y, groups=groups)
    )

    X_dev = X.iloc[dev_idx]
    y_dev = y.iloc[dev_idx]
    groups_dev = groups.iloc[dev_idx]

    X_test = X.iloc[test_idx]
    y_test = y.iloc[test_idx]
    groups_test = groups.iloc[test_idx]

    overlap = set(groups_dev.unique()).intersection(
        set(groups_test.unique())
    )

    if overlap:
        raise RuntimeError(
            f"Scenario leakage detected: {len(overlap)} groups"
        )

    print("\nScenario-aware split:")
    print(f"  Development rows : {len(X_dev):,}")
    print(f"  Final test rows  : {len(X_test):,}")
    print(f"  Development scenarios: {groups_dev.nunique()}")
    print(f"  Test scenarios       : {groups_test.nunique()}")
    print("  Leakage: NONE")

    # ---------------------------------------------------------
    # 3. Compact neural network
    # ---------------------------------------------------------
    # 15 input features
    #       ↓
    # 32 neurons
    #       ↓
    # 16 neurons
    #       ↓
    # 7 fault classes
    #
    # This is a new EDGE candidate.
    # Existing v9 model remains untouched.
    # ---------------------------------------------------------

    edge_model = Pipeline([
        (
            "scaler",
            StandardScaler(),
        ),
        (
            "classifier",
            MLPClassifier(
                hidden_layer_sizes=(32, 16),
                activation="relu",
                solver="adam",
                alpha=1e-4,
                batch_size=256,
                learning_rate_init=1e-3,
                max_iter=150,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=15,
                random_state=42,
                verbose=True,
            ),
        ),
    ])

    print("\nTraining compact neural edge model...")
    edge_model.fit(X_dev, y_dev)

    # ---------------------------------------------------------
    # 4. Evaluate on untouched test set
    # ---------------------------------------------------------
    y_pred = edge_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    macro_f1 = f1_score(
        y_test,
        y_pred,
        average="macro",
    )

    print("\n" + "=" * 60)
    print("AEROTWIN EDGE MODEL RESULTS")
    print("=" * 60)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Macro F1 : {macro_f1:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=classes,
            zero_division=0,
        )
    )

    # ---------------------------------------------------------
    # 5. Save model
    # ---------------------------------------------------------
    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        edge_model,
        EDGE_MODEL_PATH,
    )

    metadata = {
        "model_name": "AeroTwin Edge MLP",
        "model_type": "MLPClassifier",
        "purpose": "Compact edge fault classification",
        "architecture": {
            "input_features": len(FEATURE_COLUMNS),
            "hidden_layers": [32, 16],
            "output_classes": len(classes),
            "activation": "relu",
        },
        "dataset": {
            "total_rows": int(len(df)),
            "development_rows": int(len(X_dev)),
            "test_rows": int(len(X_test)),
            "scenarios": int(df[GROUP_COLUMN].nunique()),
            "scenario_leakage_protection": True,
        },
        "features": FEATURE_COLUMNS,
        "classes": classes,
        "metrics": {
            "accuracy": float(accuracy),
            "macro_f1": float(macro_f1),
        },
        "random_state": 42,
        "status": "EDGE_CANDIDATE",
        "baseline_model": "fault_classifier_v9",
        "deployment_target": "Qualcomm AI Hub / Snapdragon",
    }

    with open(
        EDGE_METADATA_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )

    print("\nSaved:")
    print(f"  {EDGE_MODEL_PATH}")
    print(f"  {EDGE_METADATA_PATH}")


if __name__ == "__main__":
    main()