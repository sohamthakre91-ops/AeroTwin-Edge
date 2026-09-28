"""
Tests for continuous MissionSimulator, 1,200 sample generation, CSV export, and mission gating.
"""

import unittest
from pathlib import Path
import pandas as pd

from src.mission import MissionSimulator, REQUIRED_TELEMETRY_COLUMNS, MISSION_CSV_PATH
from src.diagnostics import is_model_frozen, get_latest_model_path


class TestMissionSimulator(unittest.TestCase):

    def test_default_mission_generates_1200_samples(self):
        """A 60-minute mission @ 20 samples/min must generate exactly 1,200 samples."""
        sim = MissionSimulator(
            target_altitude_m=3000.0,
            base_throttle=0.75,
            duration_minutes=60.0,
            samples_per_minute=20,
            seed=42,
        )
        history = sim.run(save_csv=True)
        self.assertEqual(len(history), 1200)

        # Validate start and end timestamps
        self.assertAlmostEqual(history[0]["mission_time_min"], 0.0, places=2)
        self.assertAlmostEqual(history[-1]["mission_time_min"], 59.95, places=2)

    def test_mission_csv_export_and_columns(self):
        """Verify saved CSV exists and contains all 16 required columns."""
        self.assertTrue(MISSION_CSV_PATH.exists())
        df = pd.read_csv(MISSION_CSV_PATH)
        self.assertGreaterEqual(len(df), 1200)

        for col in REQUIRED_TELEMETRY_COLUMNS:
            self.assertIn(col, df.columns)

        # Check for unphysical NaNs
        self.assertEqual(df[REQUIRED_TELEMETRY_COLUMNS].isna().sum().sum(), 0)

        # Check physical ranges
        self.assertTrue((df["rpm"] >= 1800.0).all())
        self.assertTrue((df["rpm"] <= 6000.0).all())
        self.assertTrue((df["oil_pressure_kpa"] > 0.0).all())
        self.assertTrue((df["vibration_g"] > 0.0).all())

    def test_mission_gating_frozen_model_check(self):
        """Verify is_model_frozen returns boolean reflecting actual artifact state."""
        frozen = is_model_frozen()
        model_path = get_latest_model_path()
        if model_path and model_path.exists():
            self.assertTrue(frozen)
        else:
            self.assertFalse(frozen)


if __name__ == "__main__":
    unittest.main()
