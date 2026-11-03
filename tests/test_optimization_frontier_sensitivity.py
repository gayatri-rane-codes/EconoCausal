import pytest
import pathlib
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_frontier_endpoint_success():
    payload = {
        "budgets": [50.0, 150.0, 300.0],
        "mode": "bounded_gap",
        "cohort_id": "cohort_frontier_test",
        "use_cache": False,
        "check_persistence": False,
    }
    resp = client.post("/optimization/frontier", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["num_points"] == 3
    assert len(data["frontier"]) == 3
    # Check monotonicity of profit curve
    profits = [p["total_expected_incremental_profit"] for p in data["frontier"]]
    assert profits[0] <= profits[1] <= profits[2]

def test_frontier_negative_budget_validation():
    payload = {"budgets": [100.0, -20.0]}
    resp = client.post("/optimization/frontier", json=payload)
    assert resp.status_code == 422

def test_frontier_empty_budgets_validation():
    payload = {"budgets": []}
    resp = client.post("/optimization/frontier", json=payload)
    assert resp.status_code == 422

def test_sensitivity_endpoint_success():
    payload = {
        "base_budget": 200.0,
        "mode": "bounded_gap",
        "budget_multipliers": [0.8, 1.2],
        "cost_multipliers": [0.9, 1.1],
    }
    resp = client.post("/optimization/sensitivity", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["base_budget"] == 200.0
    assert len(data["scenarios"]) == 4
    for scenario in data["scenarios"]:
        assert "profit_delta" in scenario
        assert "spend_delta" in scenario
        assert scenario["is_feasible"] is True

def test_multi_cohort_endpoint_success_and_isolation():
    payload = {
        "cohorts": [
            {
                "cohort_id": "cohort_A",
                "budget": 100.0,
                "mode": "bounded_gap",
                "use_cache": False,
                "check_persistence": False,
            },
            {
                "cohort_id": "cohort_B",
                "budget": 250.0,
                "mode": "bounded_gap",
                "use_cache": False,
                "check_persistence": False,
            },
        ]
    }
    resp = client.post("/optimization/multi-cohort", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_cohorts"] == 2
    assert data["successful_cohorts"] == 2
    assert data["failed_cohorts"] == 0
    assert "cohort_A" in data["results"]
    assert "cohort_B" in data["results"]

def test_multi_cohort_partial_failure_handling():
    payload = {
        "cohorts": [
            {
                "cohort_id": "valid_cohort",
                "budget": 100.0,
            },
            {
                "cohort_id": "failing_cohort",
                "budget": 100.0,
                "candidate_profit_matrix": {"cust_1": {"invalid_arm": 10.0}},
                "candidate_cost_matrix": {"cust_1": {"invalid_arm": 5.0}},
            },
        ]
    }
    resp = client.post("/optimization/multi-cohort", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["successful_cohorts"] == 1
    assert data["failed_cohorts"] == 1
    assert "valid_cohort" in data["results"]
    assert len(data["errors"]) == 1
    assert data["errors"][0]["cohort_id"] == "failing_cohort"

def test_ground_truth_isolation_across_day4_endpoints():
    ground_truth = pathlib.Path("data/raw/ground_truth_evaluation.csv")
    if ground_truth.exists():
        mtime_before = ground_truth.stat().st_mtime
        client.post("/optimization/frontier", json={"budgets": [50.0, 100.0]})
        client.post("/optimization/sensitivity", json={"base_budget": 100.0})
        mtime_after = ground_truth.stat().st_mtime
        assert mtime_before == mtime_after, "Ground truth dataset must remain untouched"
