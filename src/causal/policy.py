"""
Policy assignment and economic-value evaluation module for dynamic discount pricing.
Translates CATE risk differences into incremental expected profit and assigns optimal treatment arms.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


ARM_DISCOUNTS: Dict[int, float] = {
    0: 0.00,
    1: 0.05,
    2: 0.10,
    3: 0.20,
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
        Incremental Profit(a) = P_a * (Basket * (1 - d_a) - COGS) - P_0 * (Basket - COGS) - Delta_Marketing_Cost(a)
    which decomposes into:
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
            raise KeyError(f"Expected treatment effect column '{cate_col}' not found.")

        tau_a = cate_df[cate_col].to_numpy(dtype=float)
        p_a = np.clip(p0 + tau_a, 0.0, 1.0)
        d_a = ARM_DISCOUNTS[arm]

        # Handling marketing cost differences
        if marketing_cost_diff is None:
            delta_mkt = 0.0
        elif isinstance(marketing_cost_diff, dict):
            delta_mkt = marketing_cost_diff.get(arm, 0.0)
        else:
            delta_mkt = np.asarray(marketing_cost_diff, dtype=float)

        # Expected profit under arm a minus expected profit under control arm 0
        expected_profit_arm = p_a * (bv * (1.0 - d_a) - cogs) - delta_mkt
        expected_profit_ctrl = p0 * margin_pre
        incr_profit = expected_profit_arm - expected_profit_ctrl

        profit_dict[f"incr_profit_arm_{arm}"] = incr_profit

    return pd.DataFrame(profit_dict, index=cate_df.index)


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
    
    # Extract economic inputs with defaults
    bv = econ_df["basket_value_before_discount"].to_numpy(dtype=float) if "basket_value_before_discount" in econ_df else np.full(n_rows, 100.0)
    cogs = econ_df["cost_of_goods"].to_numpy(dtype=float) if "cost_of_goods" in econ_df else np.full(n_rows, 40.0)
    p0 = econ_df["baseline_purchase_prob"].to_numpy(dtype=float) if "baseline_purchase_prob" in econ_df else np.full(n_rows, default_baseline_prob)
    spend = econ_df["historical_spend"].to_numpy(dtype=float) if "historical_spend" in econ_df else np.zeros(n_rows)

    incr_profit_df = calculate_incremental_expected_profit(
        cate_df=cate_df,
        basket_value=bv,
        cost_of_goods=cogs,
        baseline_purchase_prob=p0,
    )

    selected_arms = np.zeros(n_rows, dtype=int)
    recommended_discounts = np.zeros(n_rows, dtype=float)
    expected_values = np.zeros(n_rows, dtype=float)
    reasons = [""] * n_rows

    arms = [1, 2, 3]

    for i in range(n_rows):
        # 1. Safeguard: Non-finite inputs
        row_profits = [incr_profit_df.at[i, f"incr_profit_arm_{a}"] for a in arms]
        if not np.all(np.isfinite(row_profits)) or not np.isfinite(spend[i]):
            selected_arms[i] = 0
            recommended_discounts[i] = 0.0
            expected_values[i] = 0.0
            reasons[i] = "fallback_non_finite_input"
            continue

        # 2. Safeguard: Extreme spend tail sparsity
        if spend[i] > spend_tail_threshold:
            selected_arms[i] = 0
            recommended_discounts[i] = 0.0
            expected_values[i] = 0.0
            reasons[i] = "fallback_extreme_spend_tail"
            continue

        # 3. Economic optimization
        best_arm = 0
        best_val = 0.0

        for a in arms:
            val = incr_profit_df.at[i, f"incr_profit_arm_{a}"]
            if val > best_val and val > min_profit_threshold:
                best_val = val
                best_arm = a

        selected_arms[i] = best_arm
        recommended_discounts[i] = ARM_DISCOUNTS[best_arm]
        expected_values[i] = best_val

        if best_arm == 0:
            reasons[i] = "no_discount_economically_viable"
        else:
            reasons[i] = f"arm_{best_arm}_maximizes_incremental_profit"

    result_df = pd.DataFrame({
        "recommended_arm": selected_arms,
        "recommended_discount_rate": recommended_discounts,
        "expected_incremental_profit": expected_values,
        "assignment_reason": reasons,
    }, index=cate_df.index)

    return pd.concat([result_df, incr_profit_df], axis=1)
