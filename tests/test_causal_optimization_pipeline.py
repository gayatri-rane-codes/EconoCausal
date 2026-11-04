import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def _make_dummy_customers(n=10):
    customers = []
    for i in range(n):
        customers.append({
            "customer_id": f"cust_{i:04d}",
            "browsing_frequency": 12.0 + (i % 5),
            "cart_abandonment_rate": 0.25,
            "competitor_price_index": 1.05,
            "historical_spend": 500.0 + (i * 20.0),
            "loyalty_member": bool(i % 2),
            "previous_discount_response_rate": 0.4,
            "recency_days": 10 + (i % 15),
            "seasonality_index": 1.0,
            "basket_value": 120.0,
            "cost_of_goods": 45.0,
            "baseline_purchase_prob": 0.25,
        })
    return customers


class TestCausalOptimizationPipeline:

    @patch("src.api.causal_service.causal_service")
    def test_pipeline_success_end_to_end(self, mock_causal_srv):
        # Setup mock causal estimator output
        n_cust = 5
        customers = _make_dummy_customers(n_cust)
        mock_causal_srv.is_ready = True
        mock_effects = pd.DataFrame({
            "cate_arm_1_vs_0": [0.05, 0.08, 0.02, 0.06, 0.04],
            "cate_arm_2_vs_0": [0.08, 0.12, 0.04, 0.09, 0.06],
            "cate_arm_3_vs_0": [0.10, 0.15, 0.05, 0.12, 0.08],
        }, index=[c["customer_id"] for c in customers])
        mock_causal_srv.estimate_cate.return_value = mock_effects

        payload = {
            "customers": customers,
            "budget": 50.0,
            "mode": "bounded_gap",
            "cohort_id": "test_pipeline_cohort",
            "include_allocations": True,
            "use_cache": False,
            "check_persistence": False,
        }

        response = client.post("/optimization/pipeline", json=payload)
        assert response.status_code == 200, response.text
        data = response.json()

        assert data["status"] == "success"
        assert data["cohort_id"] == "test_pipeline_cohort"
        assert data["total_customers"] == n_cust
        assert data["is_feasible"] is True
        assert data["metrics"]["total_budget_spent"] <= 50.0 + 1e-4
        assert len(data["allocations"]) == n_cust
        for alloc in data["allocations"]:
            assert alloc["assigned_arm"] in [0, 1, 2, 3]

    @patch("src.api.causal_service.causal_service")
    def test_pipeline_greedy_mode(self, mock_causal_srv):
        n_cust = 8
        customers = _make_dummy_customers(n_cust)
        mock_causal_srv.is_ready = True
        mock_effects = pd.DataFrame({
            "cate_arm_1_vs_0": np.random.uniform(0.01, 0.1, n_cust),
            "cate_arm_2_vs_0": np.random.uniform(0.02, 0.15, n_cust),
            "cate_arm_3_vs_0": np.random.uniform(0.03, 0.2, n_cust),
        }, index=[c["customer_id"] for c in customers])
        mock_causal_srv.estimate_cate.return_value = mock_effects

        payload = {
            "customers": customers,
            "budget": 30.0,
            "mode": "greedy",
            "cohort_id": "greedy_cohort",
            "include_allocations": False,
            "use_cache": False,
            "check_persistence": False,
        }

        response = client.post("/optimization/pipeline", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["optimization_mode"] == "greedy"
        assert data["allocations"] is None
        assert data["metrics"]["total_budget_spent"] <= 30.0 + 1e-4

    def test_pipeline_validation_error_missing_customers(self):
        payload = {
            "customers": [],
            "budget": 50.0,
            "mode": "bounded_gap",
        }
        response = client.post("/optimization/pipeline", json=payload)
        assert response.status_code == 422

    def test_pipeline_validation_error_negative_budget(self):
        customers = _make_dummy_customers(2)
        payload = {
            "customers": customers,
            "budget": -10.0,
            "mode": "bounded_gap",
        }
        response = client.post("/optimization/pipeline", json=payload)
        assert response.status_code == 422

    def test_pipeline_no_ground_truth_leakage(self):
        """Strict isolation check: ensure ground_truth_evaluation.csv is never accessed"""
        import os
        forbidden_file = "ground_truth_evaluation.csv"
        with open("src/api/routers/optimization.py", "r", encoding="utf-8") as f:
            src = f.read()
        assert forbidden_file not in src, "Evaluation ground truth leaked into optimization router!"
