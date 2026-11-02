import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from src.api.main import app
from src.optimization.schemas import OptimizationResponse, AggregateMetrics, CustomerAllocationItem
import pathlib

client = TestClient(app)

def test_allocate_endpoint_success():
    payload = {
        "budget": 300.0,
        "mode": "bounded_gap",
        "cohort_id": "test_cohort",
        "use_cache": False,
        "check_persistence": False,
        "include_allocations": True,
        "solver_settings": {
            "mip_rel_gap": 0.05,
            "time_limit": 10.0,
            "allow_greedy_fallback": True,
        },
    }
    resp = client.post("/optimization/allocate", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["request_budget"] == 300.0
    assert "metrics" in data
    assert data["metrics"]["total_budget_spent"] <= 300.0
    assert len(data["allocations"]) == 100

def test_optimize_endpoint_backward_compatible_alias():
    payload = {"budget": 100.0}
    resp = client.post("/optimization/optimize", json=payload)
    assert resp.status_code == 200

def test_allocate_negative_budget_validation():
    payload = {"budget": -50.0}
    resp = client.post("/optimization/allocate", json=payload)
    assert resp.status_code == 422
    assert "greater_than_equal" in str(resp.json())

def test_allocate_forbid_extra_fields():
    payload = {"budget": 100.0, "unknown_field": "disallowed"}
    resp = client.post("/optimization/allocate", json=payload)
    assert resp.status_code == 422
    assert "extra_forbidden" in str(resp.json())

def test_allocate_unsupported_mode():
    payload = {"budget": 100.0, "mode": "unsupported_magic_mode"}
    resp = client.post("/optimization/allocate", json=payload)
    assert resp.status_code == 422

def test_allocate_custom_candidate_matrices():
    profit_matrix = {
        "cust_1": {"arm_0": 0.0, "arm_1": 10.0, "arm_2": 15.0, "arm_3": 18.0},
        "cust_2": {"arm_0": 0.0, "arm_1": 5.0, "arm_2": 8.0, "arm_3": 10.0},
    }
    cost_matrix = {
        "cust_1": {"arm_0": 0.0, "arm_1": 5.0, "arm_2": 10.0, "arm_3": 15.0},
        "cust_2": {"arm_0": 0.0, "arm_1": 2.0, "arm_2": 4.0, "arm_3": 6.0},
    }
    payload = {
        "budget": 10.0,
        "candidate_profit_matrix": profit_matrix,
        "candidate_cost_matrix": cost_matrix,
        "include_allocations": True,
        "use_cache": False,
        "check_persistence": False,
    }
    resp = client.post("/optimization/allocate", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["metrics"]["total_budget_spent"] <= 10.0
    assert len(data["allocations"]) == 2

def test_allocate_service_unavailable_503():
    with patch("src.api.routers.optimization.optimization_service", None):
        resp = client.post("/optimization/allocate", json={"budget": 100.0})
        assert resp.status_code == 503
        assert "unavailable" in resp.json()["detail"].lower()

def test_allocate_solver_failure_handling():
    with patch("src.api.routers.optimization.optimization_service.optimize", side_effect=RuntimeError("Solver core failure")):
        resp = client.post("/optimization/allocate", json={"budget": 100.0})
        assert resp.status_code == 500
        assert "Optimization failed due to an internal error" in resp.json()["detail"]

def test_allocate_greedy_fallback_behavior():
    with patch("src.api.routers.optimization.optimization_service.optimize") as mock_opt:
        mock_opt.return_value = OptimizationResponse(
            allocations=[CustomerAllocationItem(customer_id="cust_1", assigned_arm=0, discount_rate=0.0, expected_incremental_profit=0.0, allocated_budget_cost=0.0)],
            metrics=AggregateMetrics(total_expected_incremental_profit=0.0, total_budget_spent=0.0, budget_remaining=100.0, arm_counts={"0": 1}, profit_per_customer=0.0, cohort_size=1),
            status="greedy_fallback",
            solver_message="Time limit exceeded; greedy fallback triggered",
            request_budget=100.0,
            optimization_mode="bounded_gap",
            method_executed="greedy_heuristic",
            is_feasible=True,
            is_optimal_proven=False,
            provenance="live_optimization",
            elapsed_seconds=0.1,
            solver_time_seconds=0.1,
            mip_gap=None,
            mip_dual_bound=None,
            cache_hit=False,
        )
        resp = client.post("/optimization/allocate", json={"budget": 100.0})
        assert resp.status_code == 200
        assert resp.json()["status"] == "greedy_fallback"
        assert resp.json()["method_executed"] == "greedy_heuristic"

def test_ground_truth_isolation():
    ground_truth = pathlib.Path("data/raw/ground_truth_evaluation.csv")
    if ground_truth.exists():
        mtime_before = ground_truth.stat().st_mtime
        client.post("/optimization/allocate", json={"budget": 100.0})
        mtime_after = ground_truth.stat().st_mtime
        assert mtime_before == mtime_after, "Ground truth file should remain isolated and untouched"
