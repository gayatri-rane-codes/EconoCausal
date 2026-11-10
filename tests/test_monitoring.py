"""
Tests for Phase 4 Day 6: Automated Data Drift Detection and Monitoring.
Verifies KS-test for continuous features, Chi-Square for categorical features,
threshold tuning, edge cases, error handling, and API integration.
"""

import pytest
from fastapi.testclient import TestClient
from src.api.main import app
from src.monitoring.drift import (
    calculate_numerical_drift,
    calculate_categorical_drift,
    detect_dataset_drift,
    CONFOUNDER_NUMERICAL_FEATURES,
    CONFOUNDER_CATEGORICAL_FEATURES,
)

client = TestClient(app)


def make_sample_record(
    browsing=10.0,
    cart_abandon=0.2,
    comp_price=1.0,
    spend=100.0,
    loyalty=1,
    disc_rate=0.3,
    recency=15.0,
    seasonality=1.1,
):
    return {
        "browsing_frequency": browsing,
        "cart_abandonment_rate": cart_abandon,
        "competitor_price_index": comp_price,
        "historical_spend": spend,
        "loyalty_member": loyalty,
        "previous_discount_response_rate": disc_rate,
        "recency_days": recency,
        "seasonality_index": seasonality,
    }


class TestDriftDetectionEngine:
    def test_numerical_drift_identical_distributions(self):
        ref = [float(i) for i in range(50)]
        curr = [float(i) for i in range(50)]
        res = calculate_numerical_drift(ref, curr, alpha=0.05)
        assert res["drift_detected"] is False
        assert res["p_value"] == 1.0
        assert res["statistic"] == 0.0
        assert res["warning"] is None

    def test_numerical_drift_significant_shift(self):
        ref = [float(i) for i in range(100)]
        curr = [float(i + 1000) for i in range(100)]  # Complete distribution shift
        res = calculate_numerical_drift(ref, curr, alpha=0.05)
        assert res["drift_detected"] is True
        assert res["p_value"] < 0.001
        assert res["statistic"] > 0.9

    def test_categorical_drift_identical_distributions(self):
        ref = [0] * 50 + [1] * 50
        curr = [0] * 50 + [1] * 50
        res = calculate_categorical_drift(ref, curr, alpha=0.05)
        assert res["drift_detected"] is False
        assert res["drift_detected"] is False

    def test_categorical_drift_significant_shift(self):
        ref = [0] * 90 + [1] * 10
        curr = [0] * 10 + [1] * 90  # Severe category shift
        res = calculate_categorical_drift(ref, curr, alpha=0.05)
        assert res["drift_detected"] is True
        assert res["p_value"] < 0.001

    def test_small_sample_size_safe_handling(self):
        ref = [1.0, 2.0, 3.0]  # < 5 samples
        curr = [10.0, 20.0, 30.0]
        res_num = calculate_numerical_drift(ref, curr)
        assert res_num["drift_detected"] is False
        assert "Insufficient sample size" in res_num["warning"]

        res_cat = calculate_categorical_drift([1, 0], [0, 1])
        assert res_cat["drift_detected"] is False
        assert "Insufficient sample size" in res_cat["warning"]

    def test_missing_and_nan_values_handled_safely(self):
        ref = [1.0, None, float("nan"), 4.0, 5.0, 6.0, 7.0]
        curr = [1.0, 2.0, None, 4.0, 5.0, 6.0, 7.0]
        res = calculate_numerical_drift(ref, curr)
        assert res["sample_size_reference"] >= 5
        assert res["sample_size_current"] >= 5
        assert res["drift_detected"] is False

    def test_dataset_drift_summary_and_threshold(self):
        # 50 records identical
        ref = [make_sample_record(spend=50.0 + i) for i in range(50)]
        curr = [make_sample_record(spend=50.0 + i) for i in range(50)]
        res_stable = detect_dataset_drift(ref, curr, drift_share_threshold=0.3)
        assert res_stable["overall_drift_detected"] is False
        assert res_stable["drifted_features_count"] == 0
        assert "STABLE" in res_stable["summary"]

        # Shift 4 out of 8 features (50% > 30% threshold)
        curr_shifted = [
            make_sample_record(
                spend=5000.0 + i,
                browsing=500.0 + i,
                cart_abandon=0.99,
                loyalty=0,
            )
            for i in range(50)
        ]
        res_drift = detect_dataset_drift(ref, curr_shifted, drift_share_threshold=0.3)
        assert res_drift["overall_drift_detected"] is True
        assert res_drift["drift_share"] >= 0.3
        assert "DETECTED" in res_drift["summary"]


class TestDataDriftAPI:
    def test_api_data_drift_stable_success(self):
        ref_records = [make_sample_record(spend=float(i + 10)) for i in range(20)]
        curr_records = [make_sample_record(spend=float(i + 10)) for i in range(20)]

        payload = {
            "reference_data": ref_records,
            "current_data": curr_records,
            "alpha": 0.05,
            "drift_share_threshold": 0.25,
        }
        response = client.post("/monitoring/data-drift", json=payload)
        assert response.status_code == 200
        data = response.json()

        assert data["overall_drift_detected"] is False
        assert data["total_features_evaluated"] == 8
        assert len(data["feature_metrics"]) == 8
        for feat in CONFOUNDER_NUMERICAL_FEATURES:
            assert data["feature_metrics"][feat]["test_type"] == "kolmogorov_smirnov"
        for feat in CONFOUNDER_CATEGORICAL_FEATURES:
            assert data["feature_metrics"][feat]["test_type"] == "chi_square"

    def test_api_data_drift_detected_success(self):
        ref_records = [make_sample_record(recency=float(i), comp_price=1.0) for i in range(30)]
        # Severe shift across recency and competitor_price
        curr_records = [make_sample_record(recency=float(i + 500), comp_price=10.0) for i in range(30)]

        payload = {
            "reference_data": ref_records,
            "current_data": curr_records,
            "alpha": 0.05,
            "drift_share_threshold": 0.2,
        }
        response = client.post("/monitoring/data-drift", json=payload)
        assert response.status_code == 200
        data = response.json()

        assert data["feature_metrics"]["recency_days"]["drift_detected"] is True
        assert data["overall_drift_detected"] is True

    def test_api_empty_dataset_validation(self):
        payload = {
            "reference_data": [],
            "current_data": [make_sample_record()],
        }
        response = client.post("/monitoring/data-drift", json=payload)
        assert response.status_code == 422

    def test_api_invalid_feature_value_bounds(self):
        invalid_rec = make_sample_record()
        invalid_rec["cart_abandonment_rate"] = 1.5  # le=1.0 bound violation
        payload = {
            "reference_data": [make_sample_record()],
            "current_data": [invalid_rec],
        }
        response = client.post("/monitoring/data-drift", json=payload)
        assert response.status_code == 422

    def test_api_alpha_threshold_bounds(self):
        payload = {
            "reference_data": [make_sample_record()],
            "current_data": [make_sample_record()],
            "alpha": 1.5,  # lt=1.0 bound violation
        }
        response = client.post("/monitoring/data-drift", json=payload)
        assert response.status_code == 422

    def test_ground_truth_isolation_maintained(self):
        """Verifies ground_truth_evaluation.csv is never accessed during monitoring."""
        import inspect
        from src.monitoring import drift
        from src.api.routers import monitoring

        drift_source = inspect.getsource(drift)
        router_source = inspect.getsource(monitoring)
        assert "ground_truth_evaluation.csv" not in drift_source
        assert "ground_truth_evaluation.csv" not in router_source
