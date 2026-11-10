"""
Comprehensive tests for budget frontier precomputation, persistence, validation,
provenance tracking, status semantics, and public package exports (Phase 3 Day 5).
"""

import os
import shutil
import tempfile
import unittest
import json
import numpy as np
import pandas as pd

from src.optimization import (
    OptimizationRequest,
    OptimizationResponse,
    SolverSettings,
    OptimizationService,
    optimize_budget_allocation,
    optimize_greedy_allocation,
    OptimizationCache,
    compute_cohort_fingerprint,
    build_cache_key,
    save_frontier_result,
    load_frontier_result,
)


class TestFrontierPrecomputationAndExports(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        np.random.seed(42)
        n = 10
        self.customer_ids = [f"cust_{i:04d}" for i in range(n)]
        
        # Treatment arms: arm_0 (control), arm_1 (10%), arm_2 (20%), arm_3 (30%)
        p_data = {
            "arm_0": [0.0] * n,
            "arm_1": [12.0, 15.0, 8.0, 20.0, 14.0, 9.0, 18.0, 11.0, 16.0, 13.0],
            "arm_2": [22.0, 28.0, 14.0, 35.0, 25.0, 16.0, 31.0, 19.0, 29.0, 24.0],
            "arm_3": [30.0, 38.0, 19.0, 48.0, 34.0, 22.0, 42.0, 26.0, 39.0, 32.0],
        }
        c_data = {
            "arm_0": [0.0] * n,
            "arm_1": [5.0, 6.0, 4.0, 7.0, 5.5, 4.5, 6.5, 5.0, 6.0, 5.2],
            "arm_2": [11.0, 13.0, 9.0, 15.0, 12.0, 10.0, 14.0, 11.0, 13.0, 11.5],
            "arm_3": [18.0, 21.0, 15.0, 24.0, 19.5, 16.0, 22.5, 17.5, 21.0, 18.5],
        }
        self.p_mat = pd.DataFrame(p_data, index=self.customer_ids)
        self.c_mat = pd.DataFrame(c_data, index=self.customer_ids)
        self.cohort_fp = compute_cohort_fingerprint(self.p_mat, self.c_mat)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_public_package_exports(self):
        """Verify all mandatory Day 5 interfaces are exported via __all__."""
        import src.optimization as opt
        expected = [
            "optimize_budget_allocation",
            "optimize_greedy_allocation",
            "OptimizationRequest",
            "SolverSettings",
            "OptimizationResponse",
            "AggregateMetrics",
            "CustomerAllocationItem",
            "OptimizationCache",
            "compute_cohort_fingerprint",
            "build_cache_key",
            "OptimizationService",
            "save_frontier_result",
            "load_frontier_result",
        ]
        for symbol in expected:
            self.assertTrue(hasattr(opt, symbol), f"Missing public symbol: {symbol}")
            self.assertIn(symbol, opt.__all__, f"Symbol {symbol} not in __all__")

    def test_save_and_load_valid_artifact_preserves_customer_ids(self):
        """Verify customer-ID preservation and validation across save/load."""
        service = OptimizationService(cache_size=10, persistence_dir=self.temp_dir)
        req = OptimizationRequest(budget=50.0, mode="bounded_gap", solver_settings=SolverSettings(mip_rel_gap=0.01))
        resp = service.optimize(req, self.p_mat, self.c_mat, use_cache=False, check_persistence=False)

        saved_path = save_frontier_result(
            output_dir=self.temp_dir,
            cohort_fingerprint=self.cohort_fp,
            cohort_id="default_cohort",
            budget=50.0,
            mode="bounded_gap",
            solver_settings={"mip_rel_gap": 0.01},
            response_dict=resp.to_dict(include_allocations=True),
            policy_version="v1",
        )
        self.assertTrue(os.path.exists(saved_path))

        loaded_resp = load_frontier_result(
            filepath=saved_path,
            expected_fingerprint=self.cohort_fp,
            expected_budget=50.0,
            expected_mode="bounded_gap",
            expected_policy_version="v1",
        )
        self.assertIsNotNone(loaded_resp)
        loaded_cust_ids = [item["customer_id"] for item in loaded_resp["allocations"]]
        self.assertEqual(loaded_cust_ids, self.customer_ids)

    def test_reject_mismatched_cohort_budget_or_mode(self):
        """Reject persisted artifacts when fingerprint, budget, or mode differs."""
        service = OptimizationService(cache_size=10, persistence_dir=self.temp_dir)
        req = OptimizationRequest(budget=30.0, mode="greedy")
        resp = service.optimize(req, self.p_mat, self.c_mat, use_cache=False, check_persistence=False)

        saved_path = save_frontier_result(
            output_dir=self.temp_dir,
            cohort_fingerprint=self.cohort_fp,
            cohort_id="default_cohort",
            budget=30.0,
            mode="greedy",
            solver_settings={},
            response_dict=resp.to_dict(include_allocations=True),
        )

        # Mismatched fingerprint
        res_bad_fp = load_frontier_result(saved_path, expected_fingerprint="bad_fp_hash", expected_budget=30.0, expected_mode="greedy")
        self.assertIsNone(res_bad_fp)

        # Mismatched budget
        res_bad_b = load_frontier_result(saved_path, expected_fingerprint=self.cohort_fp, expected_budget=40.0, expected_mode="greedy")
        self.assertIsNone(res_bad_b)

        # Mismatched mode
        res_bad_m = load_frontier_result(saved_path, expected_fingerprint=self.cohort_fp, expected_budget=30.0, expected_mode="bounded_gap")
        self.assertIsNone(res_bad_m)

    def test_reject_corrupted_or_incomplete_file(self):
        """Reject corrupted JSON or incomplete payload without crashing."""
        corrupt_path = os.path.join(self.temp_dir, "frontier_corrupt_b50_greedy.json")
        with open(corrupt_path, "w") as f:
            f.write("{ invalid json")
        self.assertIsNone(load_frontier_result(corrupt_path, self.cohort_fp, 50.0, "greedy"))

        incomplete_path = os.path.join(self.temp_dir, "frontier_incomplete_b50_greedy.json")
        with open(incomplete_path, "w") as f:
            json.dump({"cohort_fingerprint": "abc"}, f)
        self.assertIsNone(load_frontier_result(incomplete_path, self.cohort_fp, 50.0, "greedy"))

    def test_provenance_lifecycle_live_persisted_cache(self):
        """Verify provenance tags across live execution, memory cache, and persistence lookup."""
        service = OptimizationService(cache_size=10, persistence_dir=self.temp_dir)
        req = OptimizationRequest(budget=50.0, mode="bounded_gap", solver_settings=SolverSettings(mip_rel_gap=0.01))
        
        # 1. Live run (cold)
        live_resp = service.optimize(req, self.p_mat, self.c_mat, use_cache=True, check_persistence=True)
        self.assertFalse(live_resp.cache_hit)
        self.assertIn(live_resp.provenance, ["live_solver", "live_optimization"])

        # 2. In-memory cache hit
        cached_resp = service.optimize(req, self.p_mat, self.c_mat, use_cache=True, check_persistence=True)
        self.assertTrue(cached_resp.cache_hit)
        self.assertEqual(cached_resp.provenance, "in_memory_cache")

        # 3. Save artifact to persistence_dir
        save_frontier_result(
            output_dir=self.temp_dir,
            cohort_fingerprint=self.cohort_fp,
            cohort_id="default_cohort",
            budget=50.0,
            mode="bounded_gap",
            solver_settings={"mip_rel_gap": 0.01},
            response_dict=live_resp.to_dict(include_allocations=True),
        )

        # 4. Fresh service instance (cold in-memory cache, warm disk)
        fresh_service = OptimizationService(cache_size=10, persistence_dir=self.temp_dir)
        persisted_resp = fresh_service.optimize(req, self.p_mat, self.c_mat, use_cache=True, check_persistence=True)
        self.assertFalse(persisted_resp.cache_hit)
        self.assertEqual(persisted_resp.provenance, "persisted_precomputation")
        self.assertEqual(persisted_resp.request_budget, 50.0)

    def test_canonical_budget_grid_progression(self):
        """Verify deterministic profit progression across monotonic budget grid."""
        service = OptimizationService(cache_size=10)
        grid = [10.0, 25.0, 50.0, 75.0, 100.0]
        prev_profit = -1.0
        
        for b in grid:
            req = OptimizationRequest(budget=b, mode="greedy")
            resp = service.optimize(req, self.p_mat, self.c_mat, use_cache=False)
            self.assertTrue(resp.is_feasible)
            self.assertLessEqual(resp.metrics.total_budget_spent, b + 1e-5)
            self.assertGreaterEqual(resp.metrics.total_expected_incremental_profit, prev_profit)
            prev_profit = resp.metrics.total_expected_incremental_profit


if __name__ == "__main__":
    unittest.main()
