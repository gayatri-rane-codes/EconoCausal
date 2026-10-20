"""
Evaluation and diagnostic auditing module for CATE estimators.
Computes ATE bias, MAE, RMSE, and correlation against quarantined ground truth.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def evaluate_cate_against_ground_truth(
    predictions_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    active_arms: Optional[List[int]] = None,
    id_col: str = "customer_id",
) -> pd.DataFrame:
    """
    Evaluates individual treatment effect predictions against quarantined ground truth.
    Strictly joins on customer_id to guarantee alignment.
    """
    if id_col not in predictions_df.columns:
        raise ValueError(f"id_col '{id_col}' must be present in predictions_df")
    if id_col not in ground_truth_df.columns:
        raise ValueError(f"id_col '{id_col}' must be present in ground_truth_df")

    active_arms = active_arms or [1, 2, 3]
    records = []

    merged = pd.merge(
        predictions_df,
        ground_truth_df,
        on=id_col,
        suffixes=("_pred", "_gt")
    )

    for arm in active_arms:
        pred_col = f"cate_arm_{arm}_vs_0"
        gt_col = f"true_ite_purchase_arm{arm}"

        if pred_col not in merged.columns:
            raise KeyError(f"Expected prediction column '{pred_col}' not found.")
        if gt_col not in merged.columns:
            raise KeyError(f"Expected ground-truth column '{gt_col}' not found.")

        pred = merged[pred_col].to_numpy(dtype=float)
        true = merged[gt_col].to_numpy(dtype=float)

        mae = float(np.mean(np.abs(pred - true)))
        rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
        bias = float(np.mean(pred) - np.mean(true))
        r, _ = pearsonr(pred, true)
        rho, _ = spearmanr(pred, true)

        records.append({
            "treatment_arm": arm,
            "comparison": f"Arm {arm} (discount) vs Arm 0 (control)",
            "true_ate": float(np.mean(true)),
            "est_ate": float(np.mean(pred)),
            "bias": bias,
            "mae": mae,
            "rmse": rmse,
            "pearson_r": float(r),
            "spearman_rho": float(rho),
            "pred_min": float(np.min(pred)),
            "pred_max": float(np.max(pred)),
        })

    return pd.DataFrame(records)
