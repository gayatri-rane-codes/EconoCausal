import pytest
import numpy as np
import pandas as pd

from src.causal.estimators import (
    CausalDMLEstimator,
    TLearnerEstimator,
    DEFAULT_CONFOUNDERS,
    DEFAULT_EFFECT_MODIFIERS,
    DEFAULT_CATEGORIES,
)
from src.causal.comparison import (
    split_training_and_evaluation,
    evaluate_estimator_predictions,
    evaluate_policy_on_ground_truth,
    compare_estimators_on_budget_frontier,
)


@pytest.fixture
def mock_dataset():
    np.random.seed(20261012)
    n = 200
    data = {
        "customer_id": [f"cust_{i:04d}" for i in range(n)],
        "treatment_arm": np.random.choice([0, 1, 2, 3], size=n),
        "purchased": np.random.choice([0, 1], size=n, p=[0.7, 0.3]),
        "basket_value_before_discount": np.random.uniform(50, 150, size=n),
        "cost_of_goods": np.random.uniform(20, 60, size=n),
        "baseline_purchase_prob": np.random.uniform(0.1, 0.3, size=n),
    }
    for col in DEFAULT_CONFOUNDERS:
        data[col] = np.random.uniform(0.1, 10.0, size=n)

    df = pd.DataFrame(data)
    return df


@pytest.fixture
def mock_ground_truth(mock_dataset):
    df = pd.DataFrame({
        "customer_id": mock_dataset["customer_id"],
        "true_ite_purchase_arm1": np.random.uniform(0.01, 0.15, size=len(mock_dataset)),
        "true_ite_purchase_arm2": np.random.uniform(0.03, 0.25, size=len(mock_dataset)),
        "true_ite_purchase_arm3": np.random.uniform(0.05, 0.35, size=len(mock_dataset)),
        "true_incr_net_profit_arm1": np.random.uniform(-5.0, 20.0, size=len(mock_dataset)),
        "true_incr_net_profit_arm2": np.random.uniform(-10.0, 35.0, size=len(mock_dataset)),
        "true_incr_net_profit_arm3": np.random.uniform(-15.0, 45.0, size=len(mock_dataset)),
        "expected_promo_spend_arm1": np.random.uniform(5.0, 15.0, size=len(mock_dataset)),
        "expected_promo_spend_arm2": np.random.uniform(10.0, 25.0, size=len(mock_dataset)),
        "expected_promo_spend_arm3": np.random.uniform(15.0, 35.0, size=len(mock_dataset)),
    })
    return df


def test_split_training_and_evaluation_integrity(mock_dataset):
    train_df, test_df = split_training_and_evaluation(mock_dataset, test_size=0.30, random_state=42)
    assert len(train_df) + len(test_df) == len(mock_dataset)
    assert len(test_df) == int(len(mock_dataset) * 0.30)

    # Check zero customer ID overlap
    train_ids = set(train_df["customer_id"])
    test_ids = set(test_df["customer_id"])
    assert len(train_ids.intersection(test_ids)) == 0


def test_split_training_rejects_ground_truth_leakage(mock_dataset):
    leaky_df = mock_dataset.copy()
    leaky_df["true_ite_purchase_arm1"] = 0.05
    with pytest.raises(ValueError, match="Ground-truth columns found in dataset"):
        split_training_and_evaluation(leaky_df)


def test_tlearner_fit_and_predict_interface(mock_dataset):
    tlearner = TLearnerEstimator(n_estimators=10, max_depth=3)
    tlearner.fit(mock_dataset)
    assert tlearner.is_fitted

    preds = tlearner.predict_cate(mock_dataset)
    assert isinstance(preds, pd.DataFrame)
    assert list(preds.columns) == ["cate_arm_1_vs_0", "cate_arm_2_vs_0", "cate_arm_3_vs_0"]
    assert len(preds) == len(mock_dataset)


def test_tlearner_raises_on_missing_arm(mock_dataset):
    # Filter out arm 3
    df_missing_arm = mock_dataset[mock_dataset["treatment_arm"] != 3].copy()
    tlearner = TLearnerEstimator(n_estimators=10)
    with pytest.raises(ValueError, match="missing samples for required treatment arms"):
        tlearner.fit(df_missing_arm)


def test_tlearner_rejects_ground_truth_leakage(mock_dataset):
    leaky_df = mock_dataset.copy()
    leaky_df["true_ite_arm1"] = 0.10
    tlearner = TLearnerEstimator()
    with pytest.raises(ValueError, match="Ground-truth evaluation columns detected"):
        tlearner.fit(leaky_df)


def test_evaluate_estimator_predictions_metrics(mock_dataset, mock_ground_truth):
    pred_df = pd.DataFrame({
        "customer_id": mock_dataset["customer_id"],
        "cate_arm_1_vs_0": mock_ground_truth["true_ite_purchase_arm1"] + 0.01,
        "cate_arm_2_vs_0": mock_ground_truth["true_ite_purchase_arm2"] - 0.01,
        "cate_arm_3_vs_0": mock_ground_truth["true_ite_purchase_arm3"],
    })

    metrics = evaluate_estimator_predictions(pred_df, mock_ground_truth)
    assert metrics["evaluation_sample_size"] == len(mock_dataset)
    assert metrics["excluded_records"] == 0
    assert "arm_1" in metrics["arm_metrics"]
    assert "arm_2" in metrics["arm_metrics"]
    assert "arm_3" in metrics["arm_metrics"]
    assert metrics["arm_metrics"]["arm_3"]["mae"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["arm_metrics"]["arm_1"]["mae"] == pytest.approx(0.01, abs=1e-5)


def test_evaluate_policy_on_ground_truth(mock_ground_truth):
    allocations = [
        {"customer_id": "cust_0000", "assigned_arm": 0, "cost": 0.0},
        {"customer_id": "cust_0001", "assigned_arm": 1, "cost": 10.0},
    ]
    res = evaluate_policy_on_ground_truth(allocations, mock_ground_truth)
    expected_profit = float(mock_ground_truth.loc[mock_ground_truth["customer_id"] == "cust_0001", "true_incr_net_profit_arm1"].values[0])
    assert res["evaluation_count"] == 2
    assert res["realized_incremental_profit"] == pytest.approx(round(expected_profit, 2))