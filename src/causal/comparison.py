from typing import Any, Dict, List, Optional, Tuple
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from src.causal.estimators import (
    CausalDMLEstimator,
    TLearnerEstimator,
    DEFAULT_CONFOUNDERS,
    DEFAULT_EFFECT_MODIFIERS,
)
from src.causal.policy import compute_candidate_policy_table
from src.optimization.schemas import OptimizationRequest, SolverSettings
# OptimizationService imported lazily inside functions to prevent circular import


def split_training_and_evaluation(
    df: pd.DataFrame,
    test_size: float = 0.30,
    random_state: int = 20261012,
    id_col: str = "customer_id",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Performs a deterministic, leakage-safe train/test split.
    Guarantees no customer overlap and strictly preserves customer ordering.
    """
    if id_col in df.columns:
        if df[id_col].duplicated().any():
            raise ValueError(f"Duplicate customer IDs detected in dataset: {df[id_col].duplicated().sum()}")

    gt_cols = [c for c in df.columns if c.startswith("true_") or c.startswith("expected_")]
    if gt_cols:
        raise ValueError(f"Ground-truth columns found in dataset prior to splitting: {gt_cols}")

    shuffled_indices = df.sample(frac=1.0, random_state=random_state).index
    split_idx = int(len(df) * (1.0 - test_size))

    train_indices = shuffled_indices[:split_idx]
    test_indices = shuffled_indices[split_idx:]

    train_df = df.loc[train_indices].copy().reset_index(drop=True)
    test_df = df.loc[test_indices].copy().reset_index(drop=True)
    return train_df, test_df


def evaluate_estimator_predictions(
    predictions_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    id_col: str = "customer_id",
    active_arms: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Calculates leakage-safe CATE accuracy metrics on held-out customers.
    """
    active_arms = active_arms or [1, 2, 3]
    if id_col not in predictions_df.columns:
        raise ValueError(f"id_col '{id_col}' missing from predictions_df")
    if id_col not in ground_truth_df.columns:
        raise ValueError(f"id_col '{id_col}' missing from ground_truth_df")

    merged = pd.merge(
        predictions_df,
        ground_truth_df,
        on=id_col,
        how="inner",
        suffixes=("_pred", "_gt"),
    )

    arm_metrics = {}
    all_abs_errors = []
    all_sq_errors = []

    for arm in active_arms:
        pred_col = f"cate_arm_{arm}_vs_0"
        gt_col = f"true_ite_purchase_arm{arm}"

        if pred_col not in merged.columns:
            raise KeyError(f"Predicted column '{pred_col}' not found.")
        if gt_col not in merged.columns:
            raise KeyError(f"Ground-truth column '{gt_col}' not found.")

        pred = merged[pred_col].to_numpy(dtype=float)
        true = merged[gt_col].to_numpy(dtype=float)

        mae = float(np.mean(np.abs(pred - true)))
        rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
        bias = float(np.mean(pred) - np.mean(true))

        r = float(pearsonr(pred, true)[0]) if np.std(pred) > 1e-9 and np.std(true) > 1e-9 else 0.0
        rho = float(spearmanr(pred, true)[0]) if np.std(pred) > 1e-9 and np.std(true) > 1e-9 else 0.0

        all_abs_errors.extend(np.abs(pred - true))
        all_sq_errors.extend((pred - true) ** 2)

        arm_metrics[f"arm_{arm}"] = {
            "treatment_arm": arm,
            "true_ate": float(np.mean(true)),
            "est_ate": float(np.mean(pred)),
            "bias": bias,
            "mae": mae,
            "rmse": rmse,
            "pearson_r": r,
            "spearman_rho": rho,
            "pred_min": float(np.min(pred)),
            "pred_max": float(np.max(pred)),
        }

    return {
        "evaluation_sample_size": len(merged),
        "excluded_records": len(predictions_df) - len(merged),
        "overall_mae": float(np.mean(all_abs_errors)),
        "overall_rmse": float(np.sqrt(np.mean(all_sq_errors))),
        "arm_metrics": arm_metrics,
    }


def evaluate_policy_on_ground_truth(
    allocations: List[Dict[str, Any]],
    ground_truth_df: pd.DataFrame,
    id_col: str = "customer_id",
) -> Dict[str, float]:
    """
    Evaluates customer-level policy assignments using quarantined ground truth.
    Realized incremental profit is 0.0 for arm 0, else true_incr_net_profit_arm{a}.
    """
    alloc_df = pd.DataFrame(allocations)
    if id_col not in alloc_df.columns or id_col not in ground_truth_df.columns:
        raise ValueError(f"id_col '{id_col}' must be present in allocations and ground truth")

    alloc_df[id_col] = alloc_df[id_col].astype(str)
    # Exclude any allocation-specific columns that might pre-exist in ground truth
    gt_cols_to_drop = [c for c in ['assigned_arm', 'cost'] if c in ground_truth_df.columns]
    gt_eval = ground_truth_df.drop(columns=gt_cols_to_drop).copy()
    gt_eval[id_col] = gt_eval[id_col].astype(str)

    merged = pd.merge(alloc_df, gt_eval, on=id_col, how="inner")
    if len(merged) != len(alloc_df):
        raise ValueError(f"Customer alignment mismatch: {len(alloc_df)} allocations vs {len(merged)} matches")

    realized_incr_profit = 0.0
    realized_promo_spend = 0.0

    for arm in [1, 2, 3]:
        arm_mask = merged["assigned_arm"] == arm
        profit_col = f"true_incr_net_profit_arm{arm}"
        spend_col = f"expected_promo_spend_arm{arm}"

        if arm_mask.any():
            if profit_col in merged.columns:
                realized_incr_profit += float(merged.loc[arm_mask, profit_col].sum())
            if spend_col in merged.columns:
                realized_promo_spend += float(merged.loc[arm_mask, spend_col].sum())
            elif "cost" in merged.columns:
                realized_promo_spend += float(merged.loc[arm_mask, "cost"].sum())

    return {
        "realized_incremental_profit": round(realized_incr_profit, 2),
        "realized_promo_spend": round(realized_promo_spend, 2),
        "evaluation_count": len(merged),
    }


def compare_estimators_on_budget_frontier(
    test_df: pd.DataFrame,
    dml_cate_df: pd.DataFrame,
    tlearner_cate_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    budgets: Optional[List[float]] = None,
    mode: str = "bounded_gap",
    mip_gap: float = 0.01,
    time_limit_secs: float = 30.0,
) -> Dict[str, Any]:
    """
    Compares Causal Forest DML and T-Learner across the canonical budget frontier
    evaluating both predicted and ground-truth realized metrics.
    """
    budgets = budgets or [10000.0, 25000.0, 50000.0, 75000.0, 100000.0]
    service = OptimizationService()

    dml_profit, dml_cost = compute_candidate_policy_table(dml_cate_df, test_df)
    tlearner_profit, tlearner_cost = compute_candidate_policy_table(tlearner_cate_df, test_df)

    frontier_comparison = []

    for budget in budgets:
        point_data = {"budget": budget, "mode": mode}

        for est_name, (profit_mat, cost_mat) in [
            ("causal_dml", (dml_profit, dml_cost)),
            ("t_learner", (tlearner_profit, tlearner_cost)),
        ]:
            req = OptimizationRequest(
                budget=budget,
                mode=mode,
                solver_settings=SolverSettings(
                    time_limit=time_limit_secs,
                    mip_rel_gap=mip_gap,
                ),
            )
            res = service.optimize(
                request=req,
                candidate_profit_matrix=profit_mat,
                candidate_cost_matrix=cost_mat,
            )

            allocations = [
                {"customer_id": item.customer_id, "assigned_arm": item.assigned_arm, "cost": item.allocated_budget_cost}
                for item in res.allocations
            ]
            realized_metrics = evaluate_policy_on_ground_truth(allocations, ground_truth_df)

            utilization_pct = (res.metrics.total_budget_spent / budget * 100.0) if budget > 0 else 0.0
            point_data[est_name] = {
                "predicted_incremental_profit": round(res.metrics.total_expected_incremental_profit, 2),
                "predicted_spend": round(res.metrics.total_budget_spent, 2),
                "budget_utilization_pct": round(utilization_pct, 2),
                "solver_status": res.status,
                "is_optimal_proven": res.is_optimal_proven,
                "arm_distribution": res.metrics.arm_counts,
                "realized_incremental_profit": realized_metrics["realized_incremental_profit"],
                "realized_promo_spend": realized_metrics["realized_promo_spend"],
            }

        point_data["profit_delta_dml_minus_tlearner"] = round(
            point_data["causal_dml"]["realized_incremental_profit"]
            - point_data["t_learner"]["realized_incremental_profit"],
            2,
        )
        frontier_comparison.append(point_data)

    return {
        "budgets_evaluated": budgets,
        "mode": mode,
        "customer_count": len(test_df),
        "frontier_comparison": frontier_comparison,
    }