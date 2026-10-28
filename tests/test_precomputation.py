"""
Unit tests for offline budget frontier precomputation, persistence, and module exports.
"""

import os
import shutil
import tempfile
import unittest
import pandas as pd
import numpy as np

# Verify direct exports from public package root
from src.optimization import (
    OptimizationService,
    OptimizationRequest,
    SolverSettings,
    OptimizationResponse,
    save_frontier_result,
    load_frontier_result,
    compute_cohort_fingerprint,
)


class TestFrontierPrecomputationAndExports(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        np.random.seed(42)
        self.n_cust = 20
        self.cust_ids = [f"c_{i}" for i in range(self.n_cust)]
        arm_cols = ["arm_0", "arm_1", "arm_2", "arm_3"]
        self.p_mat = pd.DataFrame(
            np.random.uniform(5, 50, size=(self.n_cust, 4)),
            index=self.cust_ids,
            columns=arm_cols,
        )
        self.c_mat = pd.DataFrame(
            np.random.uniform(2, 20, size=(self.n_cust, 4)),
            index=self.cust_ids,
            columns=arm_cols,
        )
        self.p_mat["arm_0"] = 0.0
        self.c_mat["arm_0"] = 0.0
        self.fingerprint = compute_cohort_fingerprint(self.p_mat, self.c_mat)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_save_and_load_valid_artifact(self):
        service = OptimizationService(persistence_dir=self.temp_dir)
        req = OptimizationRequest(budget=50.0, mode="greedy", cohort_id="test_cohort")
        resp = service.optimize(req, self.p_mat, self.c_mat, use_cache=False, check_persistence=False)

        saved_path = save_frontier_result(
            output_dir=self.temp_dir,
            cohort_fingerprint=self.fingerprint,
            cohort_id="test_cohort",
            budget=50.0,
            mode="greedy",
            solver_settings={},
            response_dict=resp.to_dict(include_allocations=True),
        )
        self.assertTrue(os.path.isfile(saved_path))

        loaded = load_frontier_result(
            filepath=saved_path,
            expected_fingerprint=self.fingerprint,
            expected_budget=50.0,
            expected_mode="greedy",
        )
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["request_budget"], 50.0)
        self.assertEqual(loaded["optimization_mode"], "greedy")
        self.assertEqual(len(loaded["allocations"]), self.n_cust)
        self.assertEqual(loaded["allocations"][0]["customer_id"], self.cust_ids[0])

    def test_reject_mismatched_fingerprint_or_budget(self):
        service = OptimizationService(persistence_dir=self.temp_dir)
        req = OptimizationRequest(budget=50.0, mode="greedy")
        resp = service.optimize(req, self.p_mat, self.c_mat, use_cache=False, check_persistence=False)

        saved_path = save_frontier_result(
            output_dir=self.temp_dir,
            cohort_fingerprint=self.fingerprint,
            cohort_id="test_cohort",
            budget=50.0,
            mode="greedy",
            solver_settings={},
            response_dict=resp.to_dict(),
        )

        # Mismatched fingerprint
        bad_fp = load_frontier_result(
            filepath=saved_path,
            expected_fingerprint="0000000000000000",
            expected_budget=50.0,
            expected_mode="greedy",
        )
        self.assertIsNone(bad_fp)

        # Mismatched budget
        bad_b = load_frontier_result(
            filepath=saved_path,
            expected_fingerprint=self.fingerprint,
            expected_budget=75.0,
            expected_mode="greedy",
        )
        self.assertIsNone(bad_b)

        # Mismatched mode
        bad_m = load_frontier_result(
            filepath=saved_path,
            expected_fingerprint=self.fingerprint,
            expected_budget=50.0,
            expected_mode="exact",
        )
        self.assertIsNone(bad_m)

    def test_corrupted_or_invalid_schema_file(self):
        bad_file = os.path.join(self.temp_dir, "corrupt.json")
        with open(bad_file, "w") as f:
            f.write("{invalid_json: true")

        res = load_frontier_result(bad_file, self.fingerprint, 50.0, "greedy")
        self.assertIsNone(res)

    def test_service_retrieves_persisted_artifact(self):
        service = OptimizationService(persistence_dir=self.temp_dir)
        req = OptimizationRequest(budget=40.0, mode="greedy")
        live_resp = service.optimize(req, self.p_mat, self.c_mat, use_cache=False, check_persistence=False)

        save_frontier_result(
            output_dir=self.temp_dir,
            cohort_fingerprint=self.fingerprint,
            cohort_id="default",
            budget=40.0,
            mode="greedy",
            solver_settings={},
            response_dict=live_resp.to_dict(include_allocations=True),
        )

        # Retrieve through service
        fresh_service = OptimizationService(persistence_dir=self.temp_dir)
        retrieved_resp = fresh_service.optimize(req, self.p_mat, self.c_mat, use_cache=False, check_persistence=True)

        self.assertEqual(retrieved_resp.provenance, "persisted_precomputation")
        self.assertAlmostEqual(retrieved_resp.metrics.total_expected_incremental_profit, live_resp.metrics.total_expected_incremental_profit, places=2)
        self.assertEqual(len(retrieved_resp.allocations), len(live_resp.allocations))


if __name__ == "__main__":
    unittest.main()
