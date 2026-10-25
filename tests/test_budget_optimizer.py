import unittest
import numpy as np
import pandas as pd
import itertools
from src.optimization.budget import (
    optimize_budget_allocation,
    optimize_greedy_allocation,
    build_candidate_matrix,
)
from src.causal.policy import compute_candidate_policy_table


class TestBudgetOptimizer(unittest.TestCase):
    def setUp(self):
        self.customers = ["cust_1", "cust_2", "cust_3"]
        self.profit_df = pd.DataFrame({
            "arm_0": [0.0, 0.0, 0.0],
            "arm_1": [10.0, 5.0, 2.0],
            "arm_2": [18.0, 12.0, 4.0],
            "arm_3": [22.0, 15.0, 5.0],
        }, index=self.customers)
        self.cost_df = pd.DataFrame({
            "arm_0": [0.0, 0.0, 0.0],
            "arm_1": [5.0, 4.0, 3.0],
            "arm_2": [10.0, 8.0, 6.0],
            "arm_3": [16.0, 14.0, 10.0],
        }, index=self.customers)

    def test_small_hand_calculated_optimal_assignment(self):
        res = optimize_budget_allocation(self.profit_df, self.cost_df, budget=15.0)
        self.assertEqual(res["status"], "success")
        self.assertLessEqual(res["total_budget_spent"], 15.0)
        self.assertEqual(sum(res["arm_counts"].values()), 3)
        self.assertAlmostEqual(res["total_expected_incremental_profit"], 23.0, places=2)

    def test_milp_matches_brute_force_exhaustive_search(self):
        best_profit = -1.0
        best_cost = 999.0
        budget = 15.0
        for a0, a1, a2 in itertools.product([0, 1, 2, 3], repeat=3):
            cost = self.cost_df.iloc[0, a0] + self.cost_df.iloc[1, a1] + self.cost_df.iloc[2, a2]
            profit = self.profit_df.iloc[0, a0] + self.profit_df.iloc[1, a1] + self.profit_df.iloc[2, a2]
            if cost <= budget:
                if profit > best_profit:
                    best_profit = profit
                    best_cost = cost

        res = optimize_budget_allocation(self.profit_df, self.cost_df, budget=budget)
        self.assertAlmostEqual(res["total_expected_incremental_profit"], best_profit, places=3)
        self.assertAlmostEqual(res["total_budget_spent"], best_cost, places=3)

    def test_greedy_baseline_suboptimality_gap(self):
        profit_df = pd.DataFrame({
            "arm_0": [0.0, 0.0],
            "arm_1": [10.0, 17.0],
            "arm_2": [18.0, 0.0],
            "arm_3": [0.0, 0.0],
        }, index=["cA", "cB"])
        cost_df = pd.DataFrame({
            "arm_0": [0.0, 0.0],
            "arm_1": [5.0, 10.0],
            "arm_2": [10.0, 100.0],
            "arm_3": [100.0, 100.0],
        }, index=["cA", "cB"])
        
        milp_res = optimize_budget_allocation(profit_df, cost_df, budget=10.0)
        greedy_res = optimize_greedy_allocation(profit_df, cost_df, budget=10.0)

        self.assertAlmostEqual(milp_res["total_expected_incremental_profit"], 18.0, places=2)
        self.assertAlmostEqual(greedy_res["total_expected_incremental_profit"], 10.0, places=2)
        self.assertGreater(milp_res["total_expected_incremental_profit"], greedy_res["total_expected_incremental_profit"])

    def test_budget_enforcement_and_remaining(self):
        res = optimize_budget_allocation(self.profit_df, self.cost_df, budget=12.0)
        self.assertLessEqual(res["total_budget_spent"], 12.0)
        self.assertAlmostEqual(res["total_budget_spent"] + res["budget_remaining"], 12.0, places=3)

    def test_greedy_budget_enforcement(self):
        res = optimize_greedy_allocation(self.profit_df, self.cost_df, budget=12.0)
        self.assertLessEqual(res["total_budget_spent"], 12.0)
        self.assertAlmostEqual(res["total_budget_spent"] + res["budget_remaining"], 12.0, places=3)

    def test_exactly_one_arm_assigned(self):
        res = optimize_budget_allocation(self.profit_df, self.cost_df, budget=20.0)
        alloc = res["allocation"]
        self.assertEqual(len(alloc), 3)
        self.assertTrue(all(alloc["assigned_arm"].isin([0, 1, 2, 3])))

    def test_zero_budget_fallback(self):
        res = optimize_budget_allocation(self.profit_df, self.cost_df, budget=0.0)
        self.assertEqual(res["status"], "success")
        self.assertEqual(res["total_budget_spent"], 0.0)
        self.assertEqual(res["total_expected_incremental_profit"], 0.0)
        self.assertEqual(res["arm_counts"][0], 3)

        gres = optimize_greedy_allocation(self.profit_df, self.cost_df, budget=0.0)
        self.assertEqual(gres["status"], "success")
        self.assertEqual(gres["total_budget_spent"], 0.0)
        self.assertEqual(gres["arm_counts"][0], 3)

    def test_insufficient_budget_for_expensive_discounts(self):
        res = optimize_budget_allocation(self.profit_df, self.cost_df, budget=2.0)
        self.assertLessEqual(res["total_budget_spent"], 2.0)
        self.assertEqual(res["arm_counts"][0], 3)

    def test_budget_frontier_same_candidates_monotonic(self):
        budgets = [5.0, 10.0, 15.0, 25.0, 50.0]
        prev_profit = -1.0
        for b in budgets:
            res = optimize_budget_allocation(self.profit_df, self.cost_df, budget=b)
            self.assertGreaterEqual(res["total_expected_incremental_profit"], prev_profit - 1e-4)
            self.assertLessEqual(res["total_budget_spent"], b + 1e-4)
            prev_profit = res["total_expected_incremental_profit"]

    def test_rational_unspent_budget_when_further_discounts_unprofitable(self):
        p_df = pd.DataFrame({
            "arm_0": [0.0],
            "arm_1": [5.0],
            "arm_2": [-2.0],
            "arm_3": [-10.0],
        }, index=["cust_only_1"])
        c_df = pd.DataFrame({
            "arm_0": [0.0],
            "arm_1": [10.0],
            "arm_2": [20.0],
            "arm_3": [40.0],
        }, index=["cust_only_1"])
        
        res = optimize_budget_allocation(p_df, c_df, budget=100.0)
        self.assertAlmostEqual(res["total_budget_spent"], 10.0, places=2)
        self.assertAlmostEqual(res["budget_remaining"], 90.0, places=2)
        self.assertEqual(res["allocation"].loc["cust_only_1", "assigned_arm"], 1)

    def test_empty_candidate_matrix_raises_or_handles(self):
        empty_p = pd.DataFrame(columns=["arm_0", "arm_1", "arm_2", "arm_3"])
        empty_c = pd.DataFrame(columns=["arm_0", "arm_1", "arm_2", "arm_3"])
        with self.assertRaises(ValueError):
            optimize_budget_allocation(empty_p, empty_c, budget=10.0)

    def test_lower_discount_preferred_over_larger_conversion(self):
        custom_profit = pd.DataFrame({
            "arm_0": [0.0],
            "arm_1": [5.0],
            "arm_2": [15.0],
            "arm_3": [10.0],
        }, index=["c1"])
        custom_cost = pd.DataFrame({
            "arm_0": [0.0],
            "arm_1": [2.0],
            "arm_2": [5.0],
            "arm_3": [12.0],
        }, index=["c1"])
        res = optimize_budget_allocation(custom_profit, custom_cost, budget=100.0)
        self.assertEqual(res["allocation"].loc["c1", "assigned_arm"], 2)

    def test_invalid_negative_budget(self):
        with self.assertRaises(ValueError):
            optimize_budget_allocation(self.profit_df, self.cost_df, budget=-10.0)
        with self.assertRaises(ValueError):
            optimize_greedy_allocation(self.profit_df, self.cost_df, budget=-10.0)

    def test_duplicate_or_mismatched_customer_ids(self):
        dup_profit = self.profit_df.copy()
        dup_profit.index = ["c1", "c1", "c2"]
        with self.assertRaises(ValueError):
            optimize_budget_allocation(dup_profit, self.cost_df, budget=10.0)
        with self.assertRaises(ValueError):
            optimize_greedy_allocation(dup_profit, self.cost_df, budget=10.0)

    def test_non_finite_inputs_raise_error(self):
        bad_profit = self.profit_df.copy()
        bad_profit.iloc[0, 1] = np.nan
        with self.assertRaises(ValueError):
            optimize_budget_allocation(bad_profit, self.cost_df, budget=10.0)
        with self.assertRaises(ValueError):
            optimize_greedy_allocation(bad_profit, self.cost_df, budget=10.0)

    def test_unconstrained_large_budget(self):
        res = optimize_budget_allocation(self.profit_df, self.cost_df, budget=1000.0)
        self.assertAlmostEqual(res["total_expected_incremental_profit"], 42.0, places=2)
        self.assertEqual(res["arm_counts"][3], 3)

    def test_deterministic_behavior(self):
        res1 = optimize_budget_allocation(self.profit_df, self.cost_df, budget=15.0)
        res2 = optimize_budget_allocation(self.profit_df, self.cost_df, budget=15.0)
        self.assertEqual(res1["total_expected_incremental_profit"], res2["total_expected_incremental_profit"])
        pd.testing.assert_frame_equal(res1["allocation"], res2["allocation"])

    def test_build_candidate_matrix_integration(self):
        cate_df = pd.DataFrame({
            "cate_arm_1_vs_0": [0.05, 0.02],
            "cate_arm_2_vs_0": [0.10, 0.04],
            "cate_arm_3_vs_0": [0.15, 0.08],
        }, index=["u1", "u2"])
        econ_df = pd.DataFrame({
            "basket_value_before_discount": [200.0, 150.0],
            "cost_of_goods": [80.0, 60.0],
            "historical_spend": [1500.0, 70000.0],
        }, index=["u1", "u2"])
        
        p_mat, c_mat = build_candidate_matrix(["u1", "u2"], cate_df, econ_df)
        self.assertEqual(p_mat.shape, (2, 4))
        self.assertEqual(c_mat.shape, (2, 4))
        self.assertEqual(p_mat.loc["u2", "arm_1"], 0.0)
        self.assertEqual(p_mat.loc["u2", "arm_2"], 0.0)
        self.assertEqual(p_mat.loc["u2", "arm_3"], 0.0)
        self.assertEqual(c_mat.loc["u2", "arm_1"], 0.0)


if __name__ == "__main__":
    unittest.main()
