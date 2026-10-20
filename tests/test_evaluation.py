import unittest
import numpy as np
import pandas as pd
from src.causal.evaluation import evaluate_cate_against_ground_truth


class TestEvaluationAudit(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        n = 50
        customer_ids = [f"CUST_{i:04d}" for i in range(n)]
        
        self.preds = pd.DataFrame({
            "customer_id": customer_ids,
            "cate_arm_1_vs_0": np.random.uniform(0.01, 0.05, size=n),
            "cate_arm_2_vs_0": np.random.uniform(0.04, 0.09, size=n),
            "cate_arm_3_vs_0": np.random.uniform(0.10, 0.18, size=n),
        })

        self.gt = pd.DataFrame({
            "customer_id": customer_ids,
            "true_ite_purchase_arm1": np.random.uniform(0.02, 0.06, size=n),
            "true_ite_purchase_arm2": np.random.uniform(0.05, 0.10, size=n),
            "true_ite_purchase_arm3": np.random.uniform(0.12, 0.20, size=n),
        })

    def test_alignment_and_metric_calculation(self):
        eval_df = evaluate_cate_against_ground_truth(self.preds, self.gt)
        self.assertEqual(len(eval_df), 3)
        self.assertIn("mae", eval_df.columns)
        self.assertIn("rmse", eval_df.columns)
        self.assertIn("pearson_r", eval_df.columns)
        self.assertTrue(all(np.isfinite(eval_df["mae"])))

    def test_missing_id_col_raises(self):
        bad_preds = self.preds.drop(columns=["customer_id"])
        with self.assertRaises(ValueError):
            evaluate_cate_against_ground_truth(bad_preds, self.gt)

    def test_missing_arm_col_raises(self):
        bad_preds = self.preds.drop(columns=["cate_arm_3_vs_0"])
        with self.assertRaises(KeyError):
            evaluate_cate_against_ground_truth(bad_preds, self.gt)


if __name__ == "__main__":
    unittest.main()
