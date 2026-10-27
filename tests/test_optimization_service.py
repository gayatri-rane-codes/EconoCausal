import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

from src.optimization.schemas import (
    OptimizationRequest,
    SolverSettings,
    OptimizationResponse,
)
from src.optimization.cache import (
    OptimizationCache,
    compute_cohort_fingerprint,
    build_cache_key,
)
from src.optimization.service import OptimizationService


class TestOptimizationServiceAndCache(unittest.TestCase):
    def setUp(self):
        self.customers = ["cust_A", "cust_B", "cust_C"]
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
        self.service = OptimizationService(cache_size=4)

    def test_schema_validation(self):
        req = OptimizationRequest(budget=50.0, mode="bounded_gap")
        req.validate()

        with self.assertRaises(ValueError):
            OptimizationRequest(budget=-1.0).validate()

        with self.assertRaises(ValueError):
            OptimizationRequest(budget=float("nan")).validate()

        with self.assertRaises(ValueError):
            OptimizationRequest(budget=10.0, mode="unsupported_mode").validate()

        with self.assertRaises(ValueError):
            OptimizationRequest(budget=10.0, solver_settings=SolverSettings(mip_rel_gap=-0.1)).validate()
        with self.assertRaises(ValueError):
            OptimizationRequest(budget=10.0, solver_settings=SolverSettings(time_limit=0.0)).validate()

    def test_service_agrees_with_optimizer_on_known_optimum(self):
        req = OptimizationRequest(budget=15.0, mode="exact")
        resp = self.service.optimize(req, self.profit_df, self.cost_df, use_cache=False)

        self.assertEqual(resp.status, "success")
        self.assertTrue(resp.is_optimal_proven)
        self.assertTrue(resp.is_feasible)
        self.assertAlmostEqual(resp.metrics.total_expected_incremental_profit, 23.0, places=2)
        self.assertEqual(len(resp.allocations), 3)
        self.assertEqual([a.customer_id for a in resp.allocations], self.customers)

    def test_cache_hit_and_miss_lifecycle(self):
        req = OptimizationRequest(budget=15.0, mode="exact")
        resp1 = self.service.optimize(req, self.profit_df, self.cost_df, use_cache=True)
        self.assertFalse(resp1.cache_hit)
        self.assertEqual(self.service.cache.hits, 0)
        self.assertEqual(self.service.cache.misses, 1)

        resp2 = self.service.optimize(req, self.profit_df, self.cost_df, use_cache=True)
        self.assertTrue(resp2.cache_hit)
        self.assertEqual(self.service.cache.hits, 1)
        self.assertEqual(resp1.metrics.total_expected_incremental_profit, resp2.metrics.total_expected_incremental_profit)
        self.assertLessEqual(resp2.elapsed_seconds, resp1.elapsed_seconds)

    def test_different_cohorts_do_not_collide_in_cache(self):
        fp1 = compute_cohort_fingerprint(self.profit_df, self.cost_df)
        other_profit = self.profit_df.copy()
        other_profit.iloc[0, 1] = 99.0
        fp2 = compute_cohort_fingerprint(other_profit, self.cost_df)
        self.assertNotEqual(fp1, fp2)

        req = OptimizationRequest(budget=15.0, mode="exact")
        resp1 = self.service.optimize(req, self.profit_df, self.cost_df)
        resp2 = self.service.optimize(req, other_profit, self.cost_df)
        self.assertFalse(resp2.cache_hit)
        self.assertNotEqual(
            resp1.metrics.total_expected_incremental_profit,
            resp2.metrics.total_expected_incremental_profit,
        )

    def test_different_budgets_and_modes_do_not_collide(self):
        req_exact = OptimizationRequest(budget=15.0, mode="exact")
        req_greedy = OptimizationRequest(budget=15.0, mode="greedy")
        req_b20 = OptimizationRequest(budget=20.0, mode="exact")

        self.service.optimize(req_exact, self.profit_df, self.cost_df)
        resp_greedy = self.service.optimize(req_greedy, self.profit_df, self.cost_df)
        resp_b20 = self.service.optimize(req_b20, self.profit_df, self.cost_df)

        self.assertFalse(resp_greedy.cache_hit)
        self.assertFalse(resp_b20.cache_hit)

    def test_cache_defensive_copies_prevent_caller_mutation(self):
        req = OptimizationRequest(budget=15.0, mode="exact")
        resp1 = self.service.optimize(req, self.profit_df, self.cost_df)
        resp1.allocations[0].assigned_arm = 999

        resp2 = self.service.optimize(req, self.profit_df, self.cost_df)
        self.assertTrue(resp2.cache_hit)
        self.assertNotEqual(resp2.allocations[0].assigned_arm, 999)

    def test_cache_eviction_and_capacity(self):
        small_service = OptimizationService(cache_size=2)
        r1 = OptimizationRequest(budget=10.0, mode="exact")
        r2 = OptimizationRequest(budget=12.0, mode="exact")
        r3 = OptimizationRequest(budget=14.0, mode="exact")

        small_service.optimize(r1, self.profit_df, self.cost_df)
        small_service.optimize(r2, self.profit_df, self.cost_df)
        self.assertEqual(small_service.cache.size(), 2)

        small_service.optimize(r3, self.profit_df, self.cost_df)
        self.assertEqual(small_service.cache.size(), 2)
        self.assertEqual(small_service.cache.evictions, 1)

        res_r1 = small_service.optimize(r1, self.profit_df, self.cost_df)
        self.assertFalse(res_r1.cache_hit)

    def test_cache_clear(self):
        req = OptimizationRequest(budget=15.0, mode="exact")
        self.service.optimize(req, self.profit_df, self.cost_df)
        self.assertEqual(self.service.cache.size(), 1)
        self.service.cache.clear()
        self.assertEqual(self.service.cache.size(), 0)

    def test_json_serializability(self):
        req = OptimizationRequest(budget=15.0, mode="exact")
        resp = self.service.optimize(req, self.profit_df, self.cost_df)
        d = resp.to_dict(include_allocations=True)
        self.assertIn("allocations", d)
        self.assertIn("metrics", d)
        self.assertEqual(d["metrics"]["cohort_size"], 3)

    def test_service_maps_feasible_but_not_optimal_status(self):
        # Bounded gap mode should return is_optimal_proven = False and status = feasible_suboptimal
        req = OptimizationRequest(budget=15.0, mode="bounded_gap", solver_settings=SolverSettings(mip_rel_gap=0.01))
        resp = self.service.optimize(req, self.profit_df, self.cost_df)
        self.assertTrue(resp.is_feasible)
        self.assertFalse(resp.is_optimal_proven)
        self.assertEqual(resp.status, "feasible_suboptimal")

    def test_service_handles_solver_failure_without_caching(self):
        with patch("src.optimization.service.optimize_budget_allocation") as mock_opt:
            mock_alloc = pd.DataFrame({
                "assigned_arm": [0, 0, 0],
                "discount_rate": [0.0, 0.0, 0.0],
                "expected_incremental_profit": [0.0, 0.0, 0.0],
                "allocated_budget_cost": [0.0, 0.0, 0.0],
            }, index=self.customers)
            mock_opt.return_value = {
                "status": "solver_failed_fallback",
                "method": "milp_exact",
                "solver_message": "Internal error",
                "is_feasible": False,
                "is_optimal_proven": False,
                "elapsed_seconds": 0.01,
                "total_expected_incremental_profit": 0.0,
                "total_budget_spent": 0.0,
                "budget_remaining": 15.0,
                "arm_counts": {0: 3, 1: 0, 2: 0, 3: 0},
                "allocation": mock_alloc,
            }

            req = OptimizationRequest(budget=15.0, mode="exact")
            resp = self.service.optimize(req, self.profit_df, self.cost_df, use_cache=True)

            self.assertFalse(resp.is_feasible)
            self.assertFalse(resp.is_optimal_proven)
            self.assertEqual(resp.status, "solver_failed_fallback")
            # Ensure failed result was NOT cached
            self.assertEqual(self.service.cache.size(), 0)


if __name__ == "__main__":
    unittest.main()
