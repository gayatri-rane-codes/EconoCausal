"""
Optimization service layer orchestrating candidate policy data,
safe caching, solver execution, and structured API contracts.
"""

import time
from typing import Optional, Dict, Any, Tuple
import pandas as pd
import numpy as np

from src.optimization.budget import (
    optimize_budget_allocation,
    optimize_greedy_allocation,
    validate_candidate_inputs,
)
from src.optimization.schemas import (
    OptimizationRequest,
    OptimizationResponse,
    AggregateMetrics,
    CustomerAllocationItem,
)
from src.optimization.cache import (
    OptimizationCache,
    compute_cohort_fingerprint,
    build_cache_key,
)


class OptimizationService:
    """Service layer coordinating optimization runs, caching, and payload contracts."""

    def __init__(self, cache_size: int = 64):
        self.cache = OptimizationCache(maxsize=cache_size)

    def optimize(
        self,
        request: OptimizationRequest,
        profit_matrix: pd.DataFrame,
        cost_matrix: pd.DataFrame,
        use_cache: bool = True,
    ) -> OptimizationResponse:
        t_service_start = time.time()
        request.validate()

        # Validate inputs directly to catch mismatch or invalid candidate values early
        validate_candidate_inputs(profit_matrix, cost_matrix, request.budget)

        # Compute stable fingerprint of cohort
        cohort_fp = compute_cohort_fingerprint(profit_matrix, cost_matrix)

        # Map requested mode to solver options
        mip_rel_gap = None
        time_limit = None
        if request.mode == "bounded_gap":
            mip_rel_gap = request.solver_settings.mip_rel_gap or 0.005
        elif request.mode == "bounded_time":
            time_limit = request.solver_settings.time_limit or 15.0
            mip_rel_gap = request.solver_settings.mip_rel_gap

        cache_key = build_cache_key(
            cohort_fingerprint=cohort_fp,
            budget=request.budget,
            mode=request.mode,
            mip_rel_gap=mip_rel_gap,
            time_limit=time_limit,
        )

        # 1. Check cache
        if use_cache:
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                total_elapsed = round(time.time() - t_service_start, 6)
                cached_result["cache_hit"] = True
                cached_result["elapsed_seconds"] = total_elapsed
                return self._dict_to_response(cached_result, request.budget, request.mode)

        # 2. Cache miss -> Dispatch solver
        t_solver_start = time.time()
        if request.mode == "greedy":
            raw_res = optimize_greedy_allocation(
                profit_matrix=profit_matrix,
                cost_matrix=cost_matrix,
                budget=request.budget,
            )
        else:
            raw_res = optimize_budget_allocation(
                profit_matrix=profit_matrix,
                cost_matrix=cost_matrix,
                budget=request.budget,
                mip_rel_gap=mip_rel_gap,
                time_limit=time_limit,
                allow_greedy_fallback=request.solver_settings.allow_greedy_fallback,
            )
        solver_elapsed = round(time.time() - t_solver_start, 4)
        total_elapsed = round(time.time() - t_service_start, 4)

        raw_res["cache_hit"] = False
        raw_res["solver_time_seconds"] = solver_elapsed
        raw_res["elapsed_seconds"] = total_elapsed

        # 3. Store valid feasible results in cache
        if use_cache and raw_res.get("is_feasible", False):
            self.cache.put(cache_key, raw_res)

        return self._dict_to_response(raw_res, request.budget, request.mode)

    def _dict_to_response(
        self,
        res_dict: Dict[str, Any],
        budget: float,
        mode: str,
    ) -> OptimizationResponse:
        alloc_df = res_dict["allocation"]
        n_cust = len(alloc_df)
        tot_profit = res_dict["total_expected_incremental_profit"]

        alloc_items = []
        for cust_id, row in alloc_df.iterrows():
            alloc_items.append(
                CustomerAllocationItem(
                    customer_id=str(cust_id),
                    assigned_arm=int(row["assigned_arm"]),
                    discount_rate=float(row["discount_rate"]),
                    expected_incremental_profit=float(row["expected_incremental_profit"]),
                    allocated_budget_cost=float(row["allocated_budget_cost"]),
                )
            )

        metrics = AggregateMetrics(
            total_expected_incremental_profit=tot_profit,
            total_budget_spent=res_dict["total_budget_spent"],
            budget_remaining=res_dict["budget_remaining"],
            arm_counts=res_dict["arm_counts"],
            profit_per_customer=round(tot_profit / n_cust, 4) if n_cust > 0 else 0.0,
            cohort_size=n_cust,
        )

        return OptimizationResponse(
            request_budget=budget,
            optimization_mode=mode,
            method_executed=res_dict.get("method", mode),
            status=res_dict.get("status", "unknown"),
            solver_message=res_dict.get("solver_message", ""),
            is_feasible=res_dict.get("is_feasible", False),
            is_optimal_proven=res_dict.get("is_optimal_proven", False),
            cache_hit=res_dict.get("cache_hit", False),
            elapsed_seconds=res_dict.get("elapsed_seconds", 0.0),
            solver_time_seconds=res_dict.get("solver_time_seconds", 0.0),
            mip_gap=res_dict.get("mip_gap"),
            mip_dual_bound=res_dict.get("mip_dual_bound"),
            metrics=metrics,
            allocations=alloc_items,
        )
