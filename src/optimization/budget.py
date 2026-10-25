"""
Constrained budget optimizer module for dynamic discount pricing.
Allocates exactly one treatment arm per customer to maximize total expected incremental profit
subject to a campaign promotional budget ceiling using Mixed-Integer Linear Programming (MILP)
or a Greedy Cost-Efficiency (ROI) baseline heuristic.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csc_matrix
from src.causal.policy import (
    ARM_DISCOUNTS,
    compute_candidate_policy_table,
)

TREATMENT_ARMS = [0, 1, 2, 3]


def build_candidate_matrix(
    customer_ids: Union[pd.Series, np.ndarray, List[Any]],
    cate_df: pd.DataFrame,
    econ_df: pd.DataFrame,
    spend_tail_threshold: float = 60000.0,
    default_baseline_prob: float = 0.23,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Constructs candidate incremental profit (Delta Pi) and budget cost (c) matrices
    delegating directly to the causal policy candidate evaluator.
    """
    cate_with_ids = cate_df.copy()
    cate_with_ids["customer_id"] = np.asarray(customer_ids)
    return compute_candidate_policy_table(
        cate_df=cate_with_ids,
        econ_df=econ_df,
        spend_tail_threshold=spend_tail_threshold,
        default_baseline_prob=default_baseline_prob,
    )


def optimize_budget_allocation(
    profit_matrix: pd.DataFrame,
    cost_matrix: pd.DataFrame,
    budget: float,
    tolerance: float = 1e-5,
) -> Dict[str, Any]:
    """
    Solves the binary constrained budget allocation problem using Mixed-Integer Linear Programming (scipy.optimize.milp):
        Maximize sum_{i, a} x_{ia} * Delta_Pi_{ia}
        Subject to:
            sum_a x_{ia} = 1  for all i
            sum_{i, a} x_{ia} * c_{ia} <= budget
            x_{ia} in {0, 1}
    """
    if budget < 0:
        raise ValueError(f"Budget must be non-negative, got {budget}")

    if len(profit_matrix) == 0 or len(cost_matrix) == 0:
        raise ValueError("Input candidate matrices cannot be empty")

    if len(profit_matrix) != len(cost_matrix):
        raise ValueError("profit_matrix and cost_matrix must have the same number of rows")

    if not profit_matrix.index.equals(cost_matrix.index):
        raise ValueError("Customer IDs in profit_matrix and cost_matrix must match exactly")

    if profit_matrix.index.duplicated().any():
        raise ValueError("Customer IDs must be unique")

    expected_cols = [f"arm_{a}" for a in TREATMENT_ARMS]
    for col in expected_cols:
        if col not in profit_matrix.columns or col not in cost_matrix.columns:
            raise KeyError(f"Missing required arm column '{col}'")

    n_customers = len(profit_matrix)
    n_arms = len(TREATMENT_ARMS)
    n_vars = n_customers * n_arms

    profit_values = profit_matrix[expected_cols].to_numpy(dtype=float)
    cost_values = cost_matrix[expected_cols].to_numpy(dtype=float)

    if not np.all(np.isfinite(profit_values)) or not np.all(np.isfinite(cost_values)):
        raise ValueError("Profit and cost matrices must contain finite numerical values")

    # Fast path: Budget == 0.0 -> all assigned to Arm 0
    if budget <= tolerance:
        assigned_arms = np.zeros(n_customers, dtype=int)
        assigned_discounts = np.zeros(n_customers, dtype=float)
        assigned_profit = np.zeros(n_customers, dtype=float)
        assigned_cost = np.zeros(n_customers, dtype=float)

        allocation_df = pd.DataFrame(
            {
                "assigned_arm": assigned_arms,
                "discount_rate": assigned_discounts,
                "expected_incremental_profit": assigned_profit,
                "allocated_budget_cost": assigned_cost,
            },
            index=profit_matrix.index,
        )

        return {
            "status": "success",
            "method": "milp",
            "solver_message": "Zero budget; assigned all-control fallback",
            "total_expected_incremental_profit": 0.0,
            "total_budget_spent": 0.0,
            "budget_remaining": float(budget),
            "arm_counts": {0: n_customers, 1: 0, 2: 0, 3: 0},
            "allocation": allocation_df,
        }

    # Objective: scipy.optimize.milp MINIMIZES c_obj^T x, so c_obj = -profit
    c_obj = -profit_values.flatten()

    # Constraint 1: Exactly one arm per customer
    row_indices = []
    col_indices = []
    data_vals = []
    for i in range(n_customers):
        for a in range(n_arms):
            row_indices.append(i)
            col_indices.append(i * n_arms + a)
            data_vals.append(1.0)

    # Constraint 2: Total budget expenditure <= budget
    budget_row_idx = n_customers
    for idx, c_val in enumerate(cost_values.flatten()):
        row_indices.append(budget_row_idx)
        col_indices.append(idx)
        data_vals.append(c_val)

    A_con = csc_matrix(
        (data_vals, (row_indices, col_indices)),
        shape=(n_customers + 1, n_vars),
    )

    lhs_bounds = np.concatenate([np.ones(n_customers), [-np.inf]])
    rhs_bounds = np.concatenate([np.ones(n_customers), [budget]])

    constraints = LinearConstraint(A_con, lhs_bounds, rhs_bounds)
    integrality = np.ones(n_vars)
    variable_bounds = Bounds(0.0, 1.0)

    res = milp(
        c=c_obj,
        integrality=integrality,
        constraints=constraints,
        bounds=variable_bounds,
    )

    if not res.success or res.x is None:
        assigned_arms = np.zeros(n_customers, dtype=int)
        assigned_discounts = np.zeros(n_customers, dtype=float)
        assigned_profit = np.zeros(n_customers, dtype=float)
        assigned_cost = np.zeros(n_customers, dtype=float)
        allocation_df = pd.DataFrame(
            {
                "assigned_arm": assigned_arms,
                "discount_rate": assigned_discounts,
                "expected_incremental_profit": assigned_profit,
                "allocated_budget_cost": assigned_cost,
            },
            index=profit_matrix.index,
        )

        return {
            "status": "solver_failed_fallback",
            "method": "milp",
            "solver_message": str(res.status),
            "total_expected_incremental_profit": 0.0,
            "total_budget_spent": 0.0,
            "budget_remaining": float(budget),
            "arm_counts": {0: n_customers, 1: 0, 2: 0, 3: 0},
            "allocation": allocation_df,
        }

    sol_matrix = np.round(res.x.reshape((n_customers, n_arms))).astype(int)
    assigned_arms = np.argmax(sol_matrix, axis=1)

    assigned_discounts = np.array([ARM_DISCOUNTS[a] for a in assigned_arms])
    assigned_profit = profit_values[np.arange(n_customers), assigned_arms]
    assigned_cost = cost_values[np.arange(n_customers), assigned_arms]

    tot_profit = float(np.sum(assigned_profit))
    tot_cost = float(np.sum(assigned_cost))

    counts_series = pd.Series(assigned_arms).value_counts().to_dict()
    arm_counts = {a: counts_series.get(a, 0) for a in TREATMENT_ARMS}

    allocation_df = pd.DataFrame(
        {
            "assigned_arm": assigned_arms,
            "discount_rate": assigned_discounts,
            "expected_incremental_profit": assigned_profit,
            "allocated_budget_cost": assigned_cost,
        },
        index=profit_matrix.index,
    )

    return {
        "status": "success",
        "method": "milp",
        "solver_message": "Optimal solution found",
        "total_expected_incremental_profit": round(tot_profit, 4),
        "total_budget_spent": round(tot_cost, 4),
        "budget_remaining": round(float(budget - tot_cost), 4),
        "arm_counts": arm_counts,
        "allocation": allocation_df,
    }


def optimize_greedy_allocation(
    profit_matrix: pd.DataFrame,
    cost_matrix: pd.DataFrame,
    budget: float,
    tolerance: float = 1e-5,
) -> Dict[str, Any]:
    """
    Greedy heuristic baseline optimizer.
    Ranks candidate discount assignments (arms 1, 2, 3) by efficiency ratio:
        ROI = Delta_Pi_{ia} / c_{ia}
    and greedily assigns them to customers under budget constraint B.
    
    Guarantees:
        - Exactly one arm per customer (default fallback is Arm 0).
        - Expenditure strictly <= budget.
        - Unprofitable candidates (Delta_Pi <= 0) are never selected.
        - Handled zero-cost candidates without division by zero.
    """
    if budget < 0:
        raise ValueError(f"Budget must be non-negative, got {budget}")

    if len(profit_matrix) == 0 or len(cost_matrix) == 0:
        raise ValueError("Input candidate matrices cannot be empty")

    if len(profit_matrix) != len(cost_matrix):
        raise ValueError("profit_matrix and cost_matrix must have the same number of rows")

    if not profit_matrix.index.equals(cost_matrix.index):
        raise ValueError("Customer IDs in profit_matrix and cost_matrix must match exactly")

    if profit_matrix.index.duplicated().any():
        raise ValueError("Customer IDs must be unique")

    expected_cols = [f"arm_{a}" for a in TREATMENT_ARMS]
    for col in expected_cols:
        if col not in profit_matrix.columns or col not in cost_matrix.columns:
            raise KeyError(f"Missing required arm column '{col}'")

    n_customers = len(profit_matrix)

    profit_values = profit_matrix[expected_cols].to_numpy(dtype=float)
    cost_values = cost_matrix[expected_cols].to_numpy(dtype=float)

    if not np.all(np.isfinite(profit_values)) or not np.all(np.isfinite(cost_values)):
        raise ValueError("Profit and cost matrices must contain finite numerical values")

    # Fast path: Budget == 0.0 -> all assigned to Arm 0
    if budget <= tolerance:
        assigned_arms = np.zeros(n_customers, dtype=int)
        assigned_discounts = np.zeros(n_customers, dtype=float)
        assigned_profit = np.zeros(n_customers, dtype=float)
        assigned_cost = np.zeros(n_customers, dtype=float)

        allocation_df = pd.DataFrame(
            {
                "assigned_arm": assigned_arms,
                "discount_rate": assigned_discounts,
                "expected_incremental_profit": assigned_profit,
                "allocated_budget_cost": assigned_cost,
            },
            index=profit_matrix.index,
        )

        return {
            "status": "success",
            "method": "greedy_heuristic",
            "solver_message": "Zero budget; assigned all-control fallback",
            "total_expected_incremental_profit": 0.0,
            "total_budget_spent": 0.0,
            "budget_remaining": float(budget),
            "arm_counts": {0: n_customers, 1: 0, 2: 0, 3: 0},
            "allocation": allocation_df,
        }

    # Collect candidate upgrades for arms 1, 2, 3 where Delta Pi > 0
    candidates = []
    for i in range(n_customers):
        for arm in [1, 2, 3]:
            p = profit_values[i, arm]
            c = cost_values[i, arm]
            if p > tolerance and c >= 0.0:
                if c <= tolerance:
                    ratio = 1e12
                else:
                    ratio = p / c
                candidates.append((ratio, p, c, i, arm))

    candidates.sort(key=lambda x: (-x[0], -x[1], x[2], x[3], x[4]))

    assigned_arms = np.zeros(n_customers, dtype=int)
    assigned_profit = np.zeros(n_customers, dtype=float)
    assigned_cost = np.zeros(n_customers, dtype=float)
    customer_assigned = set()

    spent_budget = 0.0

    for ratio, p, c, cust_idx, arm in candidates:
        if cust_idx in customer_assigned:
            continue
        if spent_budget + c <= budget + tolerance:
            assigned_arms[cust_idx] = arm
            assigned_profit[cust_idx] = p
            assigned_cost[cust_idx] = c
            spent_budget += c
            customer_assigned.add(cust_idx)

    assigned_discounts = np.array([ARM_DISCOUNTS[a] for a in assigned_arms])
    tot_profit = float(np.sum(assigned_profit))
    tot_cost = float(np.sum(assigned_cost))

    counts_series = pd.Series(assigned_arms).value_counts().to_dict()
    arm_counts = {a: counts_series.get(a, 0) for a in TREATMENT_ARMS}

    allocation_df = pd.DataFrame(
        {
            "assigned_arm": assigned_arms,
            "discount_rate": assigned_discounts,
            "expected_incremental_profit": assigned_profit,
            "allocated_budget_cost": assigned_cost,
        },
        index=profit_matrix.index,
    )

    return {
        "status": "success",
        "method": "greedy_heuristic",
        "solver_message": "Greedy allocation completed",
        "total_expected_incremental_profit": round(tot_profit, 4),
        "total_budget_spent": round(tot_cost, 4),
        "budget_remaining": round(float(budget - tot_cost), 4),
        "arm_counts": arm_counts,
        "allocation": allocation_df,
    }
