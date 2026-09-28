import pandas as pd
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from src.edge_inference import (
    AeroTwinEdgeInference,
    FEATURE_COLUMNS,
)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(
    "data/model_training_data.csv"
)

print("=" * 80)
print("AeroTwin Edge - Full ONNX Validation")
print("=" * 80)

print(f"\nTotal dataset rows: {len(df):,}")


# ============================================================
# LOAD ONNX MODEL
# ============================================================

engine = AeroTwinEdgeInference()


# ============================================================
# USE THE FINAL TEST SET
#
# Same scenario-aware split used during model development:
# 80% development / 20% final test
# ============================================================

from sklearn.model_selection import StratifiedGroupKFold

X = df[FEATURE_COLUMNS]
y = df["fault"]
groups = df["scenario_id"]


cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

_, test_indices = next(
    cv.split(
        X,
        y,
        groups,
    )
)

test_df = df.iloc[test_indices].copy()


print(
    f"Final test rows: {len(test_df):,}"
)

print(
    f"Test scenarios: "
    f"{test_df['scenario_id'].nunique()}"
)


# ============================================================
# RUN ONNX INFERENCE
# ============================================================

actual = []
predicted = []
confidences = []


print("\nRunning ONNX inference...")


for _, row in test_df.iterrows():

    telemetry = {
        feature: float(row[feature])
        for feature in FEATURE_COLUMNS
    }

    result = engine.predict(
        telemetry
    )

    actual.append(
        row["fault"]
    )

    predicted.append(
        result["fault"]
    )

    confidences.append(
        result["confidence"]
    )


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    actual,
    predicted,
)

macro_f1 = f1_score(
    actual,
    predicted,
    average="macro",
)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 80)
print("FINAL ONNX VALIDATION RESULTS")
print("=" * 80)

print(
    f"\nAccuracy : {accuracy:.4f} "
    f"({accuracy * 100:.2f}%)"
)

print(
    f"Macro F1 : {macro_f1:.4f} "
    f"({macro_f1 * 100:.2f}%)"
)

print(
    f"Average confidence: "
    f"{np.mean(confidences):.2%}"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 80)
print("CLASSIFICATION REPORT")
print("=" * 80)

print(
    classification_report(
        actual,
        predicted,
        digits=4,
        zero_division=0,
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

classes = sorted(
    df["fault"].unique()
)

cm = confusion_matrix(
    actual,
    predicted,
    labels=classes,
)


print("\n" + "=" * 80)
print("CONFUSION MATRIX")
print("=" * 80)

print(
    "\nRows    = Actual"
)

print(
    "Columns = Predicted\n"
)

print(
    pd.DataFrame(
        cm,
        index=classes,
        columns=classes,
    )
)


# ============================================================
# PER-CLASS ACCURACY
# ============================================================

print("\n" + "=" * 80)
print("PER-CLASS ACCURACY")
print("=" * 80)

for i, fault in enumerate(classes):

    total = cm[i].sum()

    correct = cm[i, i]

    class_accuracy = (
        correct / total
        if total > 0
        else 0
    )

    print(
        f"{fault:<30} "
        f"{class_accuracy:.2%}"
    )


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 80)
print("ONNX VALIDATION COMPLETE")
print("=" * 80)