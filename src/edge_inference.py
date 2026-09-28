"""
AeroTwin Edge Inference
-----------------------

Runs the compact AeroTwin Edge MLP ONNX model locally.

Model:
    14 telemetry features
        ↓
    StandardScaler
        ↓
    32 neurons
        ↓
    16 neurons
        ↓
    7 fault classes

ONNX output:
    [1, 7] fault probabilities
"""

from pathlib import Path

import numpy as np
import onnxruntime as ort


# ============================================================
# MODEL PATH
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models" / "aerotwin_edge_mlp.onnx"


# ============================================================
# FEATURE ORDER
# MUST MATCH TRAINING DATASET
# ============================================================

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


# ============================================================
# FAULT CLASSES
# MUST MATCH TRAINED MODEL
# ============================================================

FAULT_CLASSES = [
    "bearing_degradation",
    "cooling_degradation",
    "fuel_restriction",
    "misfire",
    "normal",
    "oil_pressure_degradation",
    "sensor_drift",
]


# ============================================================
# EDGE INFERENCE CLASS
# ============================================================

class AeroTwinEdgeInference:

    def __init__(self, model_path=None):

        if model_path is None:
            model_path = MODEL_PATH

        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"AeroTwin Edge ONNX model not found:\n"
                f"{self.model_path}"
            )

        # ----------------------------------------------------
        # ONNX Runtime
        # ----------------------------------------------------

        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=["CPUExecutionProvider"],
        )

        # ----------------------------------------------------
        # Input information
        # ----------------------------------------------------

        self.input_name = (
            self.session.get_inputs()[0].name
        )

        self.input_shape = (
            self.session.get_inputs()[0].shape
        )

        # ----------------------------------------------------
        # Output information
        # ----------------------------------------------------

        self.output_name = (
            self.session.get_outputs()[0].name
        )

        self.output_shape = (
            self.session.get_outputs()[0].shape
        )

        print("=" * 60)
        print("AeroTwin Edge Inference Engine")
        print("=" * 60)

        print(f"Model : {self.model_path}")
        print(f"Input : {self.input_name} {self.input_shape}")
        print(f"Output: {self.output_name} {self.output_shape}")

        print("=" * 60)


    # ========================================================
    # FEATURE VECTOR
    # ========================================================

    def build_feature_vector(self, telemetry):
        """
        Convert telemetry dictionary into the exact
        14-feature vector expected by the ONNX model.
        """

        missing = [
            feature
            for feature in FEATURE_COLUMNS
            if feature not in telemetry
        ]

        if missing:
            raise ValueError(
                "Missing telemetry features: "
                + ", ".join(missing)
            )

        values = [
            telemetry[feature]
            for feature in FEATURE_COLUMNS
        ]

        return np.asarray(
            [values],
            dtype=np.float32,
        )


    # ========================================================
    # RAW PREDICTION
    # ========================================================

    def predict_probabilities(self, telemetry):
        """
        Returns probability for all 7 fault classes.
        """

        features = self.build_feature_vector(
            telemetry
        )

        output = self.session.run(
            [self.output_name],
            {
                self.input_name: features
            },
        )

        probabilities = np.asarray(
            output[0],
            dtype=np.float32,
        )[0]

        return probabilities


    # ========================================================
    # FULL DIAGNOSIS
    # ========================================================

    def predict(self, telemetry):
        """
        Run edge inference and return a complete
        machine-readable diagnosis.
        """

        probabilities = self.predict_probabilities(
            telemetry
        )

        # ----------------------------------------------------
        # Safety check
        # ----------------------------------------------------

        if len(probabilities) != len(FAULT_CLASSES):
            raise RuntimeError(
                "Unexpected ONNX output size: "
                f"{len(probabilities)}"
            )

        # ----------------------------------------------------
        # Normalize just in case of tiny floating-point error
        # ----------------------------------------------------

        probabilities = np.clip(
            probabilities,
            0.0,
            1.0,
        )

        probability_sum = probabilities.sum()

        if probability_sum > 0:
            probabilities = (
                probabilities / probability_sum
            )

        # ----------------------------------------------------
        # Best class
        # ----------------------------------------------------

        predicted_index = int(
            np.argmax(probabilities)
        )

        predicted_fault = (
            FAULT_CLASSES[predicted_index]
        )

        confidence = float(
            probabilities[predicted_index]
        )

        # ----------------------------------------------------
        # Probability dictionary
        # ----------------------------------------------------

        probability_dict = {
            fault: float(probability)
            for fault, probability in zip(
                FAULT_CLASSES,
                probabilities,
            )
        }

        # ----------------------------------------------------
        # Health score
        #
        # Normal probability represents healthy state.
        # ----------------------------------------------------

        normal_probability = probability_dict[
            "normal"
        ]

        health_score = float(
            normal_probability * 100.0
        )

        # ----------------------------------------------------
        # Risk score
        # ----------------------------------------------------

        risk_score = float(
            (1.0 - normal_probability) * 100.0
        )

        # ----------------------------------------------------
        # Risk level
        # ----------------------------------------------------

        if predicted_fault == "normal":
            risk_level = "LOW"

        elif confidence >= 0.80:
            risk_level = "HIGH"

        elif confidence >= 0.60:
            risk_level = "MEDIUM"

        else:
            risk_level = "LOW"

        # ----------------------------------------------------
        # Human-readable diagnosis
        # ----------------------------------------------------

        diagnosis = self._format_diagnosis(
            predicted_fault
        )

        return {
            "fault": predicted_fault,
            "diagnosis": diagnosis,
            "confidence": confidence,
            "health_score": health_score,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "probabilities": probability_dict,
        }


    # ========================================================
    # HUMAN READABLE FAULT
    # ========================================================

    @staticmethod
    def _format_diagnosis(fault):

        descriptions = {

            "bearing_degradation":
                "Possible bearing degradation detected.",

            "cooling_degradation":
                "Possible cooling system degradation detected.",

            "fuel_restriction":
                "Possible fuel-flow restriction detected.",

            "misfire":
                "Possible engine misfire detected.",

            "normal":
                "Engine operating within the learned normal range.",

            "oil_pressure_degradation":
                "Possible oil-pressure degradation detected.",

            "sensor_drift":
                "Possible sensor drift detected.",
        }

        return descriptions.get(
            fault,
            "Unknown engine condition.",
        )


# ============================================================
# QUICK LOCAL TEST
# ============================================================

if __name__ == "__main__":

    print("\nRunning AeroTwin Edge local test...\n")

    engine = AeroTwinEdgeInference()

    # --------------------------------------------------------
    # Example telemetry
    #
    # These are only for testing ONNX execution.
    # --------------------------------------------------------

    test_telemetry = {

        "altitude_m": 1500.0,

        "ambient_temp_c": 20.0,

        "ambient_pressure_kpa": 84.0,

        "throttle": 0.65,

        "rpm": 5200.0,

        "map_kpa": 92.0,

        "engine_load": 0.68,

        "power_kw": 85.0,

        "fuel_flow_lph": 28.0,

        "cht_c": 115.0,

        "egt_c": 780.0,

        "oil_pressure_kpa": 420.0,

        "oil_temperature_c": 95.0,

        "vibration_g": 0.035,
    }

    result = engine.predict(
        test_telemetry
    )

    print("\n" + "=" * 60)
    print("EDGE AI RESULT")
    print("=" * 60)

    print(
        f"\nFault       : {result['fault']}"
    )

    print(
        f"Diagnosis   : {result['diagnosis']}"
    )

    print(
        f"Confidence  : "
        f"{result['confidence'] * 100:.2f}%"
    )

    print(
        f"Health Score: "
        f"{result['health_score']:.2f}/100"
    )

    print(
        f"Risk Score  : "
        f"{result['risk_score']:.2f}/100"
    )

    print(
        f"Risk Level  : "
        f"{result['risk_level']}"
    )

    print("\nFault probabilities:")

    for fault, probability in (
        result["probabilities"].items()
    ):

        print(
            f"  {fault:<30} "
            f"{probability * 100:6.2f}%"
        )

    print("\n" + "=" * 60)
    print("AeroTwin Edge inference test complete.")
    print("=" * 60)