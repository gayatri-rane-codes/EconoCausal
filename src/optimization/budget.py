"""
Constrained budget optimizer module for dynamic discount pricing.
Allocates exactly one treatment arm per customer to maximize total expected incremental profit
subject to a campaign promotional budget ceiling using Mixed-Integer Linear Programming (MILP)
or a Greedy Cost-Efficiency (ROI) baseline heuristic. Supports configurable execution limits,
relative optimality tolerances (mip_rel_gap), time limits, and verified fallback paths.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import time
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


def validate_candidate_inputs(
    profit_matrix: pd.DataFrame,
    cost_matrix: pd.DataFrame,
    budget: float,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Validates structure, finite numerical values, customer ID indexing, and budget sanity."""
    if budget < 0:
        raise ValueError(f"Budget must be non-negative, got {budget}")
    if not np.isfinite(budget):
        raise ValueError(f"Budget must be a finite numerical value, got {budget}")

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

    profit_values = profit_matrix[expected_cols].to_numpy(dtype=float)
    cost_values = cost_matrix[expected_cols].to_numpy(dtype=float)

    if not np.all(np.isfinite(profit_values)) or not np.all(np.isfinite(cost_values)):
        raise ValueError("Profit and cost matrices must contain finite numerical values")

    return profit_values, cost_values, len(profit_matrix), len(TREATMENT_ARMS)


def _build_zero_budget_allocation(
    profit_matrix: pd.DataFrame,
    budget: float,
    method: str = "milp_exact",
) -> Dict[str, Any]:
    """Fast-path zero-budget fallback assigning all customers to Arm 0."""
    n_customers = len(profit_matrix)
    allocation_df = pd.DataFrame(
        {
            "assigned_arm": np.zeros(n_customers, dtype=int),
            "discount_rate": np.zeros(n_customers, dtype=float),
            "expected_incremental_profit": np.zeros(n_customers, dtype=float),
            "allocated_budget_cost": np.zeros(n_customers, dtype=float),
        },
        index=profit_matrix.index,
    )

    return {
        "status": "success",
        "method": method,
        "solver_message": "Zero budget; assigned all-control fallback",
        "is_feasible": True,
        "is_optimal_proven": True,
        "elapsed_seconds": 0.0,
        "mip_gap": 0.0,
        "mip_dual_bound": 0.0,
        "total_expected_incremental_profit": 0.0,
        "total_budget_spent": 0.0,
        "budget_remaining": float(budget),
        "arm_counts": {0: n_customers, 1: 0, 2: 0, 3: 0},
        "allocation": allocation_df,
    }


def optimize_budget_allocation(
    profit_matrix: pd.DataFrame,
    cost_matrix: pd.DataFrame,
    budget: float,
    mip_rel_gap: Optional[float] = None,
    time_limit: Optional[float] = None,
    allow_greedy_fallback: bool = False,
    tolerance: float = 1e-5,
) -> Dict[str, Any]:
    """
    Solves the binary constrained budget allocation problem using scipy.optimize.milp:
        Maximize sum_{i, a} x_{ia} * Delta_Pi_{ia}
        Subject to:
            sum_a x_{ia} = 1  for all i
            sum_{i, a} x_{ia} * c_{ia} <= budget
            x_{ia} in {0, 1}

    Parameters
    ----------
    mip_rel_gap : Optional[float]
        Relative MIP optimality gap tolerance (e.g. 0.005 for 0.5%).
    time_limit : Optional[float]
        Maximum solver wall-clock time limit in seconds.
    allow_greedy_fallback : bool
        If True, falls back to validated greedy heuristic when MILP cannot produce
        a feasible solution within constraints/limits.
    """
    t_start = time.time()
    profit_values, cost_values, n_customers, n_arms = validate_candidate_inputs(
        profit_matrix, cost_matrix, budget
    )

    solver_options: Dict[str, Any] = {"presolve": True}
    if mip_rel_gap is not None:
        if not np.isfinite(mip_rel_gap) or mip_rel_gap < 0.0:
            raise ValueError(f"mip_rel_gap must be non-negative and finite, got {mip_rel_gap}")
        solver_options["mip_rel_gap"] = float(mip_rel_gap)

    if time_limit is not None:
        if not np.isfinite(time_limit) or time_limit <= 0.0:
            raise ValueError(f"time_limit must be positive and finite, got {time_limit}")
        solver_options["time_limit"] = float(time_limit)

    method_tag = "milp_exact"
    if mip_rel_gap is not None and time_limit is not None:
        method_tag = "milp_bounded_gap_and_time"
    elif mip_rel_gap is not None:
        method_tag = "milp_bounded_gap"
    elif time_limit is not None:
        method_tag = "milp_time_limited"

    if budget <= tolerance:
        return _build_zero_budget_allocation(profit_matrix, budget, method=method_tag)

    n_vars = n_customers * n_arms
    c_obj = -profit_values.flatten()

    row_indices = []
    col_indices = []
    data_vals = []
    for i in range(n_customers):
        for a in range(n_arms):
            row_indices.append(i)
            col_indices.append(i * n_arms + a)
            data_vals.append(1.0)

    budget_row_idx = n_customers
    for idx, c_val in enumerate(cost_values.flatten()):
        row_indices.append(budget_row_idx)
        col_indices.append(idx)
        data_vals.append(c_val)

    A_con = csc_matrix((data_vals, (row_indices, col_indices)), shape=(n_customers + 1, n_vars))
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
        options=solver_options,
    )
    elapsed_time = round(time.time() - t_start, 4)

    is_feasible = False
    is_optimal_proven = False
    sol_matrix = None

    if res.x is not None:
        sol_candidate = np.round(res.x.reshape((n_customers, n_arms))).astype(int)
        row_sums = np.sum(sol_candidate, axis=1)
        tot_c = float(np.sum(sol_candidate * cost_values))
        # Verify integrality, uniqueness (sum = 1), and budget ceiling
        if (
            np.all(np.isin(sol_candidate, [0, 1]))
            and np.all(row_sums == 1)
            and (tot_c <= budget + tolerance)
        ):
            is_feasible = True
            sol_matrix = sol_candidate

    if res.success and res.status == 0:
        if mip_rel_gap is None or mip_rel_gap == 0.0:
            is_optimal_proven = True

    # Case A: Feasible assignment found by MILP
    if is_feasible and sol_matrix is not None:
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

        reported_gap = getattr(res, "mip_gap", None)
        dual_bnd = getattr(res, "mip_dual_bound", None)

        return {
            "status": "success" if is_optimal_proven else "feasible_suboptimal",
            "method": method_tag,
            "solver_message": res.message,
            "is_feasible": True,
            "is_optimal_proven": is_optimal_proven,
            "elapsed_seconds": elapsed_time,
            "mip_gap": float(reported_gap) if reported_gap is not None and np.isfinite(reported_gap) else None,
            "mip_dual_bound": float(-dual_bnd) if dual_bnd is not None and np.isfinite(dual_bnd) else None,
            "total_expected_incremental_profit": round(tot_profit, 4),
            "total_budget_spent": round(tot_cost, 4),
            "budget_remaining": round(float(budget - tot_cost), 4),
            "arm_counts": arm_counts,
            "allocation": allocation_df,
        }

    # Case B: MILP failed to find a valid feasible solution
    if allow_greedy_fallback:
        greedy_res = optimize_greedy_allocation(profit_matrix, cost_matrix, budget, tolerance=tolerance)
        # Independently validate greedy solution
        if (
            greedy_res.get("is_feasible", False)
            and greedy_res["total_budget_spent"] <= budget + tolerance
            and sum(greedy_res["arm_counts"].values()) == n_customers
        ):
            greedy_res["solver_message"] = f"MILP failed ({res.message}); fell back to verified greedy heuristic"
            greedy_res["method"] = "greedy_fallback"
            greedy_res["is_feasible"] = True
            greedy_res["is_optimal_proven"] = False
            greedy_res["elapsed_seconds"] = elapsed_time
            return greedy_res
        else:
            raise RuntimeError(
                "Optimization failed: neither MILP nor fallback greedy heuristic could produce a valid feasible assignment."
            )

    # Infeasible or failed outcome without fallback
    assigned_arms = np.zeros(n_customers, dtype=int)
    allocation_df = pd.DataFrame(
        {
            "assigned_arm": assigned_arms,
            "discount_rate": np.zeros(n_customers, dtype=float),
            "expected_incremental_profit": np.zeros(n_customers, dtype=float),
            "allocated_budget_cost": np.zeros(n_customers, dtype=float),
        },
        index=profit_matrix.index,
    )

    status_tag = "infeasible" if getattr(res, "status", None) == 2 else "solver_failed_fallback"

    return {
        "status": status_tag,
        "method": method_tag,
        "solver_message": res.message if hasattr(res, "message") else str(res.status),
        "is_feasible": False,
        "is_optimal_proven": False,
        "elapsed_seconds": elapsed_time,
        "mip_gap": None,
        "mip_dual_bound": None,
        "total_expected_incremental_profit": 0.0,
        "total_budget_spent": 0.0,
        "budget_remaining": float(budget),
        "arm_counts": {0: n_customers, 1: 0, 2: 0, 3: 0},
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
    """
    t_start = time.time()
    profit_values, cost_values, n_customers, n_arms = validate_candidate_inputs(
        profit_matrix, cost_matrix, budget
    )

    if budget <= tolerance:
        res = _build_zero_budget_allocation(profit_matrix, budget, method="greedy_heuristic")
        res["elapsed_seconds"] = round(time.time() - t_start, 4)
        return res

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
        "is_feasible": True,
        "is_optimal_proven": False,
        "elapsed_seconds": round(time.time() - t_start, 4),
        "mip_gap": None,
        "mip_dual_bound": None,
        "total_expected_incremental_profit": round(tot_profit, 4),
        "total_budget_spent": round(tot_cost, 4),
        "budget_remaining": round(float(budget - tot_cost), 4),
        "arm_counts": arm_counts,
        "allocation": allocation_df,
    }
