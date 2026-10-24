"""
Economic policy assignment module.
Transforms estimated Heterogeneous Treatment Effects (CATE) into
prescriptive, profit-maximizing discount recommendations per customer.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd


ARM_DISCOUNTS = {
    0: 0.00,
    1: 0.05,
    2: 0.10,
    3: 0.20
}


def calculate_incremental_expected_profit(
    cate_df: pd.DataFrame,
    basket_value: Union[pd.Series, np.ndarray, float],
    cost_of_goods: Union[pd.Series, np.ndarray, float],
    baseline_purchase_prob: Union[pd.Series, np.ndarray, float],
    marketing_cost_diff: Optional[Union[pd.Series, np.ndarray, Dict[int, float]]] = None,
    active_arms: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Computes incremental expected profit for each discount arm relative to the 0% control arm.

    Formula:
      Delta_Profit(a) = tau_a * (Basket - COGS) - (P_0 + tau_a) * (d_a * Basket) - Delta_Marketing_Cost(a)
    """
    active_arms = active_arms or [1, 2, 3]
    bv = np.asarray(basket_value, dtype=float)
    cogs = np.asarray(cost_of_goods, dtype=float)
    p0 = np.asarray(baseline_purchase_prob, dtype=float)
    margin_pre = bv - cogs

    profit_dict = {}

    for arm in active_arms:
        cate_col = f"cate_arm_{arm}_vs_0"
        if cate_col not in cate_df.columns:
            raise KeyError(f"Missing required CATE column: {cate_col}")
        
        tau_a = cate_df[cate_col].to_numpy(dtype=float)
        p_a = np.clip(p0 + tau_a, 0.0, 1.0)
        d_a = ARM_DISCOUNTS[arm]

        if marketing_cost_diff is None:
            delta_mkt = 0.0
        elif isinstance(marketing_cost_diff, dict):
            delta_mkt = marketing_cost_diff.get(arm, 0.0)
        else:
            delta_mkt = np.asarray(marketing_cost_diff, dtype=float)

        expected_profit_arm = p_a * (bv * (1.0 - d_a) - cogs) - delta_mkt
        expected_profit_ctrl = p0 * margin_pre
        incr_profit = expected_profit_arm - expected_profit_ctrl

        profit_dict[f"incr_profit_arm_{arm}"] = incr_profit

    return pd.DataFrame(profit_dict, index=cate_df.index)


def compute_candidate_policy_table(
    cate_df: pd.DataFrame,
    econ_df: pd.DataFrame,
    spend_tail_threshold: float = 60000.0,
    default_baseline_prob: float = 0.23,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Computes candidate incremental profit and promotional budget cost matrices
    across all 4 treatment arms (0, 1, 2, 3) for each customer.
    """
    n = len(cate_df)
    index = cate_df.index if "customer_id" not in cate_df.columns else cate_df["customer_id"].values

    bv = econ_df["basket_value_before_discount"].to_numpy(dtype=float) if "basket_value_before_discount" in econ_df else np.full(n, 100.0)
    cogs = econ_df["cost_of_goods"].to_numpy(dtype=float) if "cost_of_goods" in econ_df else np.full(n, 40.0)
    p0 = econ_df["baseline_purchase_prob"].to_numpy(dtype=float) if "baseline_purchase_prob" in econ_df else np.full(n, default_baseline_prob)
    spend = econ_df["historical_spend"].to_numpy(dtype=float) if "historical_spend" in econ_df else np.zeros(n)

    margin_pre = bv - cogs

    profit_dict = {"arm_0": np.zeros(n, dtype=float)}
    cost_dict = {"arm_0": np.zeros(n, dtype=float)}

    for arm in [1, 2, 3]:
        cate_col = f"cate_arm_{arm}_vs_0"
        if cate_col not in cate_df.columns:
            raise KeyError(f"Required CATE column '{cate_col}' not found.")
        
        tau_a = cate_df[cate_col].to_numpy(dtype=float)
        p_a = np.clip(p0 + tau_a, 0.0, 1.0)
        d_a = ARM_DISCOUNTS[arm]

        incr_profit = tau_a * margin_pre - p_a * (d_a * bv)
        budget_cost = p_a * (d_a * bv)

        # Safeguard: extreme spend tail (> 60k) falls back to control
        extreme_mask = spend > spend_tail_threshold
        incr_profit[extreme_mask] = 0.0
        budget_cost[extreme_mask] = 0.0

        # Non-finite values fall back to 0
        invalid_mask = ~np.isfinite(incr_profit) | ~np.isfinite(budget_cost)
        incr_profit[invalid_mask] = 0.0
        budget_cost[invalid_mask] = 0.0

        profit_dict[f"arm_{arm}"] = incr_profit
        cost_dict[f"arm_{arm}"] = np.maximum(budget_cost, 0.0)

    profit_df = pd.DataFrame(profit_dict, index=index)
    cost_df = pd.DataFrame(cost_dict, index=index)
    profit_df.index.name = "customer_id"
    cost_df.index.name = "customer_id"

    return profit_df, cost_df


def assign_discount_policy(
    cate_df: pd.DataFrame,
    econ_df: pd.DataFrame,
    spend_tail_threshold: float = 60000.0,
    min_profit_threshold: float = 0.0,
    default_baseline_prob: float = 0.35,
) -> pd.DataFrame:
    """
    Assigns the optimal discount arm based on positive incremental expected profit.
    Applies reliability safeguards for extreme feature support tails (e.g. historical spend > 60k),
    non-finite values, and negative economic returns.
    """
    n_rows = len(cate_df)
    c_ids = cate_df.index if "customer_id" not in cate_df.columns else cate_df["customer_id"].values

    bv = econ_df["basket_value_before_discount"].values if "basket_value_before_discount" in econ_df else np.full(n_rows, 100.0)
    cogs = econ_df["cost_of_goods"].values if "cost_of_goods" in econ_df else np.full(n_rows, 40.0)
    p0 = econ_df["baseline_purchase_prob"].values if "baseline_purchase_prob" in econ_df else np.full(n_rows, default_baseline_prob)
    spend = econ_df["historical_spend"].values if "historical_spend" in econ_df else np.zeros(n_rows)

    # Check for non-finite inputs in CATE
    cate_arm_cols = [c for c in cate_df.columns if c.startswith("cate_arm_")]
    has_non_finite_cate = cate_df[cate_arm_cols].isnull().any(axis=1).values | ~np.isfinite(cate_df[cate_arm_cols].values).all(axis=1)

    incr_profit_df = calculate_incremental_expected_profit(
        cate_df=cate_df,
        basket_value=bv,
        cost_of_goods=cogs,
        baseline_purchase_prob=p0,
    )

    recom_arms = []
    recom_discounts = []
    recom_profits = []
    recom_reasons = []

    for i in range(n_rows):
        # Safeguard 1: Non-finite inputs
        if has_non_finite_cate[i]:
            recom_arms.append(0)
            recom_discounts.append(ARM_DISCOUNTS[0])
            recom_profits.append(0.0)
            recom_reasons.append("fallback_non_finite_input")
            continue

        # Safeguard 2: Extreme spend tail
        if spend[i] > spend_tail_threshold:
            recom_arms.append(0)
            recom_discounts.append(ARM_DISCOUNTS[0])
            recom_profits.append(0.0)
            recom_reasons.append("fallback_extreme_spend_tail")
            continue

        best_arm = 0
        best_profit = min_profit_threshold

        for arm in [1, 2, 3]:
            profit_arm = incr_profit_df.iloc[i][f"incr_profit_arm_{arm}"]
            if np.isfinite(profit_arm) and profit_arm > best_profit:
                best_profit = profit_arm
                best_arm = arm

        if best_arm == 0:
            recom_arms.append(0)
            recom_discounts.append(ARM_DISCOUNTS[0])
            recom_profits.append(0.0)
            recom_reasons.append("no_discount_economically_viable")
        else:
            recom_arms.append(best_arm)
            recom_discounts.append(ARM_DISCOUNTS[best_arm])
            recom_profits.append(best_profit)
            recom_reasons.append(f"arm_{best_arm}_maximizes_incremental_profit")

    return pd.DataFrame({
        "customer_id": c_ids,
        "recommended_arm": recom_arms,
        "recommended_discount_rate": recom_discounts,
        "expected_incremental_profit": recom_profits,
        "assignment_reason": recom_reasons,
    }, index=cate_df.index)
