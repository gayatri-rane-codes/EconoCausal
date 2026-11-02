"""API test suite for /optimization router endpoints."""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_optimize_endpoint_default_cohort():
    """Verify optimization endpoint succeeds using default candidate matrices."""
    payload = {
        "budget": 250.0,
        "mode": "bounded_gap",
        "cohort_id": "test_default_cohort",
        "use_cache": False,
        "check_persistence": False,
        "include_allocations": True,
    }
    response = client.post("/optimization/optimize", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["request_budget"] == 250.0
    assert data["is_feasible"] is True
    assert "metrics" in data
    assert data["metrics"]["cohort_size"] == 100
    assert data["metrics"]["total_budget_spent"] <= 250.0
    assert len(data["allocations"]) == 100


def test_optimize_endpoint_custom_candidate_matrices():
    """Verify optimization endpoint with caller-supplied profit and cost matrices."""
    candidate_profit = {
        "cust_1": {"arm_0": 0.0, "arm_1": 10.0, "arm_2": 20.0, "arm_3": 25.0},
        "cust_2": {"arm_0": 0.0, "arm_1": 5.0, "arm_2": 12.0, "arm_3": 14.0},
        "cust_3": {"arm_0": 0.0, "arm_1": 2.0, "arm_2": 4.0, "arm_3": 6.0},
    }
    candidate_cost = {
        "cust_1": {"arm_0": 0.0, "arm_1": 4.0, "arm_2": 8.0, "arm_3": 12.0},
        "cust_2": {"arm_0": 0.0, "arm_1": 3.0, "arm_2": 6.0, "arm_3": 9.0},
        "cust_3": {"arm_0": 0.0, "arm_1": 1.0, "arm_2": 2.0, "arm_3": 4.0},
    }

    payload = {
        "budget": 15.0,
        "mode": "greedy",
        "cohort_id": "custom_3_cust",
        "use_cache": False,
        "check_persistence": False,
        "include_allocations": True,
        "candidate_profit_matrix": candidate_profit,
        "candidate_cost_matrix": candidate_cost,
    }
    response = client.post("/optimization/optimize", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["is_feasible"] is True
    assert data["metrics"]["cohort_size"] == 3
    assert data["metrics"]["total_budget_spent"] <= 15.0
    assert len(data["allocations"]) == 3
    cust_ids = {item["customer_id"] for item in data["allocations"]}
    assert cust_ids == {"cust_1", "cust_2", "cust_3"}


def test_optimize_endpoint_negative_budget_validation():
    """Verify strict validation rejects negative budgets."""
    payload = {
        "budget": -50.0,
        "mode": "bounded_gap",
    }
    response = client.post("/optimization/optimize", json=payload)
    assert response.status_code == 422


def test_optimize_endpoint_forbid_extra_fields():
    """Verify strict validation rejects unexpected fields under ConfigDict(extra='forbid')."""
    payload = {
        "budget": 100.0,
        "mode": "bounded_gap",
        "unexpected_field": "disallowed",
    }
    response = client.post("/optimization/optimize", json=payload)
    assert response.status_code == 422
