import unittest
import numpy as np
import pandas as pd
import warnings
from src.causal.estimators import (
    CausalDMLEstimator,
    DEFAULT_CONFOUNDERS,
    DEFAULT_EFFECT_MODIFIERS,
)


class TestCausalDMLEstimator(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        n = 80
        data = {
            "treatment_arm": np.random.choice([0, 1, 2, 3], size=n),
            "purchased": np.random.choice([0, 1], size=n),
        }
        for col in DEFAULT_CONFOUNDERS:
            # Vary scale dramatically to test pipeline scaling robustness
            data[col] = np.random.uniform(10, 50000, size=n) if col == "historical_spend" else np.random.uniform(0, 1, size=n)
        for col in DEFAULT_EFFECT_MODIFIERS:
            if col not in data:
                data[col] = np.random.uniform(0, 1, size=n)

        self.df = pd.DataFrame(data)
        self.estimator = CausalDMLEstimator(cv=2, random_state=42)

    def test_validation_missing_column(self):
        bad_df = self.df.drop(columns=["historical_spend"])
        with self.assertRaises(ValueError):
            self.estimator.validate_inputs(bad_df)

    def test_validation_null_values(self):
        bad_df = self.df.copy()
        bad_df.loc[0, "recency_days"] = np.nan
        with self.assertRaises(ValueError):
            self.estimator.validate_inputs(bad_df)

    def test_validation_invalid_treatment_values(self):
        bad_df = self.df.copy()
        bad_df.loc[0, "treatment_arm"] = 99
        with self.assertRaises(ValueError):
            self.estimator.validate_inputs(bad_df)

    def test_validation_invalid_outcome_values(self):
        bad_df = self.df.copy()
        bad_df.loc[0, "purchased"] = 5
        with self.assertRaises(ValueError):
            self.estimator.validate_inputs(bad_df)

    def test_minimal_fit_and_effect_structure(self):
        self.estimator.fit(self.df)
        effects = self.estimator.estimate_effects(self.df)
        self.assertEqual(effects.shape, (len(self.df), 3))
        self.assertListEqual(
            list(effects.columns),
            ["cate_arm_1_vs_0", "cate_arm_2_vs_0", "cate_arm_3_vs_0"],
        )
        self.assertFalse(effects.isnull().any().any())

    def test_pipeline_scaling_convergence_no_warnings(self):
        with warnings.catch_warnings(record=True) as recorded_warnings:
            warnings.simplefilter("always")
            self.estimator.fit(self.df)
            conv_warnings = [
                w for w in recorded_warnings
                if "TOTAL NO. OF ITERATIONS REACHED LIMIT" in str(w.message)
            ]
            self.assertEqual(len(conv_warnings), 0)


if __name__ == "__main__":
    unittest.main()
