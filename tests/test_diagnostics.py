import unittest
import numpy as np
import pandas as pd
from src.causal.diagnostics import compute_treatment_overlap, diagnose_cate_reliability
from src.causal.estimators import DEFAULT_CONFOUNDERS


class TestDiagnostics(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        n = 100
        data = {
            "customer_id": [f"CUST_{i:04d}" for i in range(n)],
            "treatment_arm": np.random.choice([0, 1, 2, 3], size=n),
            "historical_spend": np.random.uniform(50, 10000, size=n),
        }
        for col in DEFAULT_CONFOUNDERS:
            if col not in data:
                data[col] = np.random.uniform(0, 1, size=n)

        self.df = pd.DataFrame(data)

    def test_compute_treatment_overlap_structure_and_bounds(self):
        props_df, summary = compute_treatment_overlap(
            self.df, n_splits=2, random_state=42, overlap_threshold=0.05
        )
        self.assertEqual(len(props_df), len(self.df))
        self.assertIn("min_propensity", props_df.columns)
        self.assertIn("overlap_violation", props_df.columns)
        self.assertTrue(all(props_df["min_propensity"] >= 0.0))
        self.assertTrue(all(props_df["min_propensity"] <= 1.0))
        self.assertIn("violation_rate", summary)
        self.assertFalse(props_df.isnull().any().any())

    def test_compute_treatment_overlap_missing_column(self):
        bad_df = self.df.drop(columns=["historical_spend"])
        with self.assertRaises(ValueError):
            compute_treatment_overlap(bad_df, n_splits=2)

    def test_diagnose_cate_reliability_crosstabs(self):
        props_df, _ = compute_treatment_overlap(self.df, n_splits=2, random_state=42)
        cate_df = pd.DataFrame({
            "cate_arm_3_vs_0": np.random.uniform(-0.3, 0.2, size=len(self.df))
        }, index=self.df.index)

        res = diagnose_cate_reliability(self.df, cate_df, props_df, target_arm=3, extreme_threshold=-0.2)
        self.assertIn("spend_comparison", res)
        self.assertIn("overlap_crosstab", res)
        self.assertFalse(res["spend_comparison"].isnull().any().any())


if __name__ == "__main__":
    unittest.main()
