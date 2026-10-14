import unittest
import pandas as pd
from src.causal.graph import (
    get_causal_graph_dot,
    verify_acyclicity,
    build_and_identify,
)

class TestCausalGraphValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_path = "data/raw/econocausal_synthetic_dataset.csv"
        cls.df = pd.read_csv(cls.data_path)
        cls.dot_graph = get_causal_graph_dot()
        cls.model, cls.estimand = build_and_identify(data_path=cls.data_path)

    def test_graph_is_acyclic(self):
        """Verify the causal graph is a strict Directed Acyclic Graph (DAG)."""
        is_acyclic = verify_acyclicity(self.dot_graph)
        self.assertTrue(is_acyclic)

    def test_treatment_and_outcome_in_dataset(self):
        """Verify treatment and outcome columns exist in the raw dataset."""
        self.assertIn("treatment_arm", self.df.columns)
        self.assertIn("purchased", self.df.columns)

    def test_four_treatment_discount_arms_exist(self):
        """Verify all 4 discount levels (0%, 5%, 10%, 20%) exist without binary reduction."""
        unique_arms = set(self.df["treatment_arm"].unique())
        self.assertEqual(unique_arms, {0, 1, 2, 3}, f"Expected arms {{0, 1, 2, 3}}, got {unique_arms}")
        
        if "discount_rate" in self.df.columns:
            unique_rates = set(self.df["discount_rate"].round(2).unique())
            self.assertEqual(unique_rates, {0.00, 0.05, 0.10, 0.20})

    def test_confounders_exist_in_dataset(self):
        """Verify all 8 identified backdoor confounders exist in the raw dataset."""
        expected_confounders = [
            "browsing_frequency",
            "cart_abandonment_rate",
            "competitor_price_index",
            "historical_spend",
            "loyalty_member",
            "previous_discount_response_rate",
            "recency_days",
            "seasonality_index",
        ]
        for col in expected_confounders:
            self.assertIn(col, self.df.columns)

    def test_no_prohibited_variables_in_adjustment_set(self):
        """Verify no post-treatment, leakage, or ID variables leak into the adjustment set."""
        prohibited = {
            "customer_id", "revenue", "discount_amount", "net_profit",
            "cost_of_goods", "marketing_cost", "basket_value_before_discount",
            "assigned_arm", "targeting_score",
        }
        backdoor_vars = set(self.estimand.get_backdoor_variables())
        intersection = backdoor_vars.intersection(prohibited)
        self.assertEqual(len(intersection), 0, f"Prohibited variables found: {intersection}")

    def test_dowhy_identification_successful(self):
        """Verify DoWhy successfully computes backdoor estimand with exactly 8 variables."""
        backdoor_vars = self.estimand.get_backdoor_variables()
        self.assertEqual(len(backdoor_vars), 8)

if __name__ == "__main__":
    unittest.main()
