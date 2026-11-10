"""
Unit and integration tests for EconoCausal API Causal Router (/causal/estimate and /causal/predict).
"""
import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
import pandas as pd
import numpy as np

from src.api.main import app
from src.api.causal_service import causal_service, CausalService
from src.causal.estimators import CausalDMLEstimator, DEFAULT_CONFOUNDERS


class TestCausalRouter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        
        # Build minimal synthetic dataset to fit a fast test estimator (cv=2)
        np.random.seed(42)
        n = 60
        data = {
            "treatment_arm": np.random.choice([0, 1, 2, 3], size=n),
            "purchased": np.random.choice([0, 1], size=n),
        }
        for col in DEFAULT_CONFOUNDERS:
            data[col] = np.random.uniform(10, 500, size=n) if col == "historical_spend" else np.random.uniform(0, 1, size=n)
        
        train_df = pd.DataFrame(data)
        cls.test_estimator = CausalDMLEstimator(random_state=42, cv=2)
        cls.test_estimator.fit(train_df)

        cls.sample_customer = {
            "customer_id": "cust_test_001",
            "browsing_frequency": 5.2,
            "cart_abandonment_rate": 0.35,
            "competitor_price_index": 1.02,
            "historical_spend": 450.0,
            "loyalty_member": 1,
            "previous_discount_response_rate": 0.40,
            "recency_days": 12.0,
            "seasonality_index": 1.15,
            "basket_value": 150.0,
            "baseline_purchase_prob": 0.30,
        }

    def setUp(self):
        # Inject fast test estimator into singleton service
        causal_service._estimator = self.test_estimator
        causal_service._is_ready = True

    def test_estimate_cate_endpoint_success(self):
        payload = {"customers": [self.sample_customer]}
        response = self.client.post("/causal/estimate", json=payload)
        self.assertEqual(response.status_code, 200)
        
        body = response.json()
        self.assertEqual(body["count"], 1)
        res = body["results"][0]
        self.assertEqual(res["customer_id"], "cust_test_001")
        self.assertIn("cate_arm_1_vs_0", res)
        self.assertIn("cate_arm_2_vs_0", res)
        self.assertIn("cate_arm_3_vs_0", res)
        self.assertIsInstance(res["cate_arm_1_vs_0"], float)

    def test_predict_policy_endpoint_success(self):
        payload = {"customers": [self.sample_customer]}
        response = self.client.post("/causal/predict", json=payload)
        self.assertEqual(response.status_code, 200)

        body = response.json()
        self.assertEqual(body["count"], 1)
        rec = body["recommendations"][0]
        self.assertEqual(rec["customer_id"], "cust_test_001")
        self.assertIn(rec["recommended_arm"], [0, 1, 2, 3])
        self.assertGreaterEqual(rec["expected_incremental_profit"], 0.0)

    def test_estimate_validation_error_missing_feature(self):
        bad_customer = self.sample_customer.copy()
        del bad_customer["cart_abandonment_rate"]
        response = self.client.post("/causal/estimate", json={"customers": [bad_customer]})
        self.assertEqual(response.status_code, 422)

    def test_estimate_validation_error_out_of_range(self):
        bad_customer = self.sample_customer.copy()
        bad_customer["cart_abandonment_rate"] = 1.5  # must be <= 1.0
        response = self.client.post("/causal/estimate", json={"customers": [bad_customer]})
        self.assertEqual(response.status_code, 422)

    def test_uninitialized_service_503(self):
        with patch.object(causal_service, "initialize", side_effect=Exception("Data missing")):
            causal_service._is_ready = False
            causal_service._estimator = None
            response = self.client.post("/causal/estimate", json={"customers": [self.sample_customer]})
            self.assertEqual(response.status_code, 503)

    def test_data_isolation_ground_truth_never_accessed(self):
        with patch("pandas.read_csv", side_effect=pd.read_csv) as mock_read:
            payload = {"customers": [self.sample_customer]}
            self.client.post("/causal/estimate", json=payload)
            for call_args in mock_read.call_args_list:
                called_path = str(call_args[0][0])
                self.assertNotIn("ground_truth_evaluation.csv", called_path)


if __name__ == "__main__":
    unittest.main()
