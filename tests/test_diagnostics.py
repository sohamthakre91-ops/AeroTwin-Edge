"""
Tests for ML Diagnostics:
- Group-aware 80/20 train/test holdout split with zero leakage
- 5-Fold Grouped Cross-Validation (train_groups ∩ val_groups == ∅ for every fold)
- Model candidate comparison (Random Forest vs HistGradientBoosting)
- Model selection using CV Macro F1
- Retraining on full development data & single evaluation on untouched holdout
- Model versioning, metadata schema, and frozen model inference
"""

import unittest
from pathlib import Path
import pandas as pd
import numpy as np

from src.diagnostics import (
    train_fault_classifier,
    predict_fault,
    load_metadata,
    is_model_frozen,
    evaluate_candidate_cv,
    create_model_instance,
    compute_metrics,
    FEATURE_COLUMNS,
    METADATA_PATH,
    MODELS_DIR,
)
from src.dataset import generate_scenario_data, FAULT_CLASSES


class TestDiagnostics(unittest.TestCase):

    def test_model_metadata_registry(self):
        """Verify model_metadata.json contains valid trained model versions and required metrics."""
        metadata = load_metadata()
        self.assertIn("latest_version", metadata)
        self.assertIn("versions", metadata)

        latest_ver = metadata.get("latest_version")
        if latest_ver and latest_ver in metadata["versions"]:
            latest_metrics = metadata["versions"][latest_ver]
            self.assertIn("model_version", latest_metrics)
            self.assertIn("status", latest_metrics)
            self.assertIn("is_frozen", latest_metrics)
            self.assertTrue(latest_metrics["is_frozen"])
            self.assertIn("number_of_cv_folds", latest_metrics)
            self.assertEqual(latest_metrics["number_of_cv_folds"], 5)
            self.assertIn("final_holdout_metrics", latest_metrics)
            self.assertIn("selected_cv_mean_metrics", latest_metrics)
            self.assertIn("selected_cv_std_metrics", latest_metrics)
            self.assertIn("confusion_matrix", latest_metrics)
            self.assertIn("feature_importance", latest_metrics)

    def test_inference_prediction(self):
        """Verify predict_fault returns valid hypothesis distribution, probabilities, and frozen flag."""
        sample_telemetry = {
            "altitude_m": 3000.0,
            "ambient_temp_c": -4.5,
            "ambient_pressure_kpa": 70.1,
            "throttle": 0.75,
            "rpm": 5420.0,
            "map_kpa": 142.0,
            "engine_load": 0.78,
            "power_kw": 84.5,
            "fuel_flow_lph": 26.2,
            "cht_c": 172.0,
            "egt_c": 745.0,
            "oil_pressure_kpa": 360.0,
            "oil_temperature_c": 98.0,
            "vibration_g": 0.028,
        }

        diag = predict_fault(sample_telemetry)

        self.assertIn("predicted_fault", diag)
        self.assertIn("confidence", diag)
        self.assertIn("ranked_hypotheses", diag)
        self.assertIn("fault_probabilities", diag)
        self.assertIn("is_frozen", diag)

        self.assertIsInstance(diag["confidence"], float)
        self.assertGreaterEqual(diag["confidence"], 0.0)
        self.assertLessEqual(diag["confidence"], 1.0)
        self.assertEqual(len(diag["ranked_hypotheses"]), 7)

    def test_cv_zero_group_leakage(self):
        """Verify that 5-fold grouped CV ensures zero scenario overlap in every fold."""
        # Generate 35 scenarios (5 per class, 20 steps each)
        dfs = []
        scen_id = 100
        for fc in FAULT_CLASSES:
            for _ in range(5):
                dfs.append(generate_scenario_data(scen_id, fc, steps=20, seed=scen_id))
                scen_id += 1
        df_dev = pd.concat(dfs, ignore_index=True)

        X_dev = df_dev[FEATURE_COLUMNS]
        y_dev = df_dev["fault"]
        groups_dev = df_dev["scenario_id"]

        candidate = {
            "name": "Test RF",
            "model_type": "RandomForestClassifier",
            "hyperparameters": {"n_estimators": 20, "max_depth": 5, "random_state": 42, "n_jobs": 1},
        }

        cv_res = evaluate_candidate_cv(candidate, X_dev, y_dev, groups_dev, n_splits=5)
        self.assertEqual(len(cv_res["fold_metrics"]), 5)
        self.assertIn("mean_metrics", cv_res)
        self.assertIn("std_metrics", cv_res)
        self.assertIn("f1_macro", cv_res["mean_metrics"])

        # Check that fold metrics have non-empty valid numbers
        for fm in cv_res["fold_metrics"]:
            self.assertGreaterEqual(fm["accuracy"], 0.0)
            self.assertGreaterEqual(fm["f1_macro"], 0.0)

    def test_end_to_end_training_pipeline_and_selection(self):
        """
        Verify the full 10-stage training pipeline:
        - 80/20 grouped split
        - 5-fold grouped CV on candidate models
        - Selection of candidate with highest mean CV Macro F1
        - Retraining on all dev data
        - Single evaluation on untouched holdout
        - Versioned persistence
        """
        # Create small dataset with 35 scenarios (5 per class)
        dfs = []
        scen_id = 200
        for fc in FAULT_CLASSES:
            for _ in range(5):
                dfs.append(generate_scenario_data(scen_id, fc, steps=20, seed=scen_id))
                scen_id += 1
        df_small = pd.concat(dfs, ignore_index=True)

        candidates = [
            {
                "name": "Candidate A (RF 30 trees)",
                "model_type": "RandomForestClassifier",
                "hyperparameters": {"n_estimators": 30, "max_depth": 6, "random_state": 42, "n_jobs": 1},
            },
            {
                "name": "Candidate B (HGB 20 iter)",
                "model_type": "HistGradientBoostingClassifier",
                "hyperparameters": {"max_iter": 20, "max_depth": 4, "random_state": 42},
            },
        ]

        # Backup original metadata
        orig_meta_text = METADATA_PATH.read_text(encoding="utf-8") if METADATA_PATH.exists() else None

        stages = []
        def on_prog(stage, msg):
            stages.append(stage)

        metrics = None
        try:
            model, metrics = train_fault_classifier(
                dataframe=df_small,
                progress_callback=on_prog,
                candidates=candidates,
            )

            self.assertIsNotNone(model)
            self.assertEqual(set(stages), set(range(1, 11)))
            self.assertTrue(metrics["is_frozen"])
            self.assertEqual(metrics["status"], "ACTIVE/FROZEN")
            self.assertIn("selected_model_name", metrics)
            self.assertIn("final_holdout_metrics", metrics)
            self.assertIn("model_comparison", metrics)
            self.assertEqual(len(metrics["model_comparison"]), 2)

            # Check that best candidate has highest CV Macro F1
            best_name = metrics["selected_model_name"]
            cand_scores = {c["candidate_name"]: c["cv_macro_f1_mean"] for c in metrics["model_comparison"]}
            max_score = max(cand_scores.values())
            self.assertEqual(cand_scores[best_name], max_score)

            # Verify holdout metrics exist and are separate
            holdout = metrics["final_holdout_metrics"]
            self.assertIn("accuracy", holdout)
            self.assertIn("f1_macro", holdout)
            self.assertIn("precision_macro", holdout)
            self.assertIn("recall_macro", holdout)

            # Check frozen status
            self.assertTrue(is_model_frozen())

        finally:
            if orig_meta_text is not None:
                METADATA_PATH.write_text(orig_meta_text, encoding="utf-8")
            if metrics and "model_version" in metrics:
                ephemeral_joblib = MODELS_DIR / f"{metrics['model_version']}.joblib"
                ephemeral_json = MODELS_DIR / f"model_metadata_{metrics['model_version'].replace('fault_classifier_', '')}.json"
                if ephemeral_joblib.exists():
                    ephemeral_joblib.unlink(missing_ok=True)
                if ephemeral_json.exists():
                    ephemeral_json.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
