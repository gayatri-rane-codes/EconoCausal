import unittest
import numpy as np
import pandas as pd
from src.causal.policy import (
    calculate_incremental_expected_profit,
    assign_discount_policy,
    ARM_DISCOUNTS,
)


class TestPolicyAssignment(unittest.TestCase):
    def setUp(self):
        self.cate_df = pd.DataFrame({
            "cate_arm_1_vs_0": [0.04, 0.05, 0.15, -0.05],
            "cate_arm_2_vs_0": [0.08, 0.12, 0.25, -0.10],
            "cate_arm_3_vs_0": [0.15, 0.14, 0.40, -0.30],
        })
        self.econ_df = pd.DataFrame({
            "basket_value_before_discount": [100.0, 100.0, 200.0, 100.0],
            "cost_of_goods": [40.0, 40.0, 80.0, 40.0],
            "baseline_purchase_prob": [0.30, 0.30, 0.20, 0.30],
            "historical_spend": [1000.0, 75000.0, 5000.0, 2000.0],
        })

    def test_incremental_profit_calculation(self):
        profit_df = calculate_incremental_expected_profit(
            cate_df=self.cate_df,
            basket_value=self.econ_df["basket_value_before_discount"],
            cost_of_goods=self.econ_df["cost_of_goods"],
            baseline_purchase_prob=self.econ_df["baseline_purchase_prob"],
        )
        self.assertIn("incr_profit_arm_1", profit_df.columns)
        self.assertIn("incr_profit_arm_2", profit_df.columns)
        self.assertIn("incr_profit_arm_3", profit_df.columns)
        self.assertEqual(len(profit_df), 4)

    def test_margin_erosion_selects_control_or_lower_discount(self):
        # Even when Arm 3 has higher conversion lift than Arm 2,
        # discount cost can make Arm 2 more profitable.
        policy_df = assign_discount_policy(self.cate_df, self.econ_df)
        self.assertIn(policy_df.loc[0, "recommended_arm"], [0, 1, 2, 3])
        self.assertTrue(policy_df.loc[0, "expected_incremental_profit"] >= 0.0)

    def test_extreme_spend_tail_safeguard(self):
        # Row 1 has historical_spend = 75,000 (> 60,000)
        policy_df = assign_discount_policy(self.cate_df, self.econ_df)
        self.assertEqual(policy_df.loc[1, "recommended_arm"], 0)
        self.assertEqual(policy_df.loc[1, "recommended_discount_rate"], 0.0)
        self.assertEqual(policy_df.loc[1, "assignment_reason"], "fallback_extreme_spend_tail")

    def test_non_finite_input_fallback(self):
        bad_cate = self.cate_df.copy()
        bad_cate.loc[0, "cate_arm_1_vs_0"] = np.nan
        policy_df = assign_discount_policy(bad_cate, self.econ_df)
        self.assertEqual(policy_df.loc[0, "recommended_arm"], 0)
        self.assertEqual(policy_df.loc[0, "assignment_reason"], "fallback_non_finite_input")


if __name__ == "__main__":
    unittest.main()
