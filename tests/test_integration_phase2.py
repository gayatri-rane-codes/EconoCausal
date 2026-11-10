import unittest
import numpy as np
import pandas as pd
from src.causal.graph import build_and_identify
from src.causal.estimators import CausalDMLEstimator
from src.causal.evaluation import evaluate_cate_against_ground_truth
from src.causal.diagnostics import compute_treatment_overlap, diagnose_cate_reliability
from src.causal.policy import assign_discount_policy


class TestPhase2Integration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = pd.read_csv("data/raw/econocausal_synthetic_dataset.csv")
        cls.gt = pd.read_csv("data/raw/ground_truth_evaluation.csv")

    def test_full_pipeline_orchestration(self):
        # 1. Graph & Adjustment set
        model, estimand = build_and_identify()
        self.assertIsNotNone(estimand)

        # 2. Fit Estimator on subset to run fast in test suite
        sample_df = self.df.sample(n=1000, random_state=20261012).reset_index(drop=True)
        estimator = CausalDMLEstimator(random_state=20261012, cv=2)
        estimator.fit(sample_df)
        cate_df = estimator.estimate_effects(sample_df)

        self.assertEqual(len(cate_df), 1000)
        self.assertIn("cate_arm_1_vs_0", cate_df.columns)
        self.assertIn("cate_arm_2_vs_0", cate_df.columns)
        self.assertIn("cate_arm_3_vs_0", cate_df.columns)

        # 3. Diagnostics
        overlap_df, summary = compute_treatment_overlap(sample_df, n_splits=2, random_state=20261012)
        self.assertEqual(len(overlap_df), 1000)
        self.assertTrue(np.all(np.isfinite(overlap_df["min_propensity"])))

        # 4. Policy Assignment
        policy_df = assign_discount_policy(
            cate_df=cate_df,
            econ_df=sample_df,
            spend_tail_threshold=60000.0,
            default_baseline_prob=0.35,
        )

        self.assertEqual(len(policy_df), 1000)
        self.assertTrue(set(policy_df["recommended_arm"].unique()).issubset({0, 1, 2, 3}))
        self.assertFalse(policy_df["recommended_arm"].isnull().any())
        self.assertFalse(policy_df["expected_incremental_profit"].isnull().any())
        self.assertTrue(all(policy_df["expected_incremental_profit"] >= 0.0))


if __name__ == "__main__":
    unittest.main()
