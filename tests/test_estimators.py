import unittest
import numpy as np
import pandas as pd
from src.causal.estimators import (
    CausalDMLEstimator,
    DEFAULT_CONFOUNDERS,
    DEFAULT_EFFECT_MODIFIERS,
)


class TestCausalDMLEstimator(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        n = 120
        data = {
            "treatment_arm": np.random.choice([0, 1, 2, 3], size=n),
            "purchased": np.random.choice([0, 1], size=n),
        }
        for col in DEFAULT_CONFOUNDERS:
            data[col] = np.random.randn(n)
        self.sample_data = pd.DataFrame(data)
        self.estimator = CausalDMLEstimator(cv=2, random_state=42)

    def test_validation_missing_column(self):
        """Test that missing required columns raise ValueError."""
        bad_df = self.sample_data.drop(columns=["purchased"])
        with self.assertRaises(ValueError):
            self.estimator.validate_inputs(bad_df)

    def test_validation_null_values(self):
        """Test that null values raise ValueError."""
        bad_df = self.sample_data.copy()
        bad_df.loc[0, "browsing_frequency"] = np.nan
        with self.assertRaises(ValueError):
            self.estimator.validate_inputs(bad_df)

    def test_validation_invalid_treatment_values(self):
        """Test that unexpected treatment arm values raise ValueError."""
        bad_df = self.sample_data.copy()
        bad_df.loc[0, "treatment_arm"] = 99
        with self.assertRaises(ValueError):
            self.estimator.validate_inputs(bad_df)

    def test_validation_invalid_outcome_values(self):
        """Test that non-binary outcome values raise ValueError."""
        bad_df = self.sample_data.copy()
        bad_df.loc[0, "purchased"] = 5
        with self.assertRaises(ValueError):
            self.estimator.validate_inputs(bad_df)

    def test_minimal_fit_and_effect_structure(self):
        """Test minimal estimator fit and output structure for arm-vs-control effects."""
        self.estimator.fit(self.sample_data)
        self.assertTrue(self.estimator.is_fitted)

        # Estimate effects on first 10 rows
        test_sub = self.sample_data.iloc[:10]
        effects = self.estimator.estimate_effects(test_sub, control_arm=0, active_arms=[1, 2, 3])

        self.assertIsInstance(effects, pd.DataFrame)
        self.assertEqual(effects.shape, (10, 3))
        expected_cols = ["cate_arm_1_vs_0", "cate_arm_2_vs_0", "cate_arm_3_vs_0"]
        self.assertEqual(list(effects.columns), expected_cols)
        self.assertFalse(effects.isnull().any().any())


if __name__ == "__main__":
    unittest.main()
