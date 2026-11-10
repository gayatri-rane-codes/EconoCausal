"""
Service layer mediating between candidate policies, mathematical optimization,
in-memory caching, and optional persisted frontier artifact reuse.
"""

import time
import os
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np

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
from src.optimization.budget import (
    optimize_budget_allocation,
    optimize_greedy_allocation,
)
from src.optimization.persistence import load_frontier_result


class OptimizationService:
    """Reusable optimization service orchestrating caching, persistence, and solver execution."""

    def __init__(self, cache_size: int = 64, persistence_dir: Optional[str] = None):
        self.cache = OptimizationCache(maxsize=cache_size)
        self.persistence_dir = persistence_dir

    def optimize(
        self,
        request: OptimizationRequest,
        candidate_profit_matrix: pd.DataFrame,
        candidate_cost_matrix: pd.DataFrame,
        use_cache: bool = True,
        check_persistence: bool = True,
    ) -> OptimizationResponse:
        """Executes or retrieves budget optimization based on request parameters."""
        request.validate()

        t_start = time.time()
        cohort_fp = compute_cohort_fingerprint(candidate_profit_matrix, candidate_cost_matrix)
        mip_gap = request.solver_settings.mip_rel_gap if request.solver_settings else None
        time_lim = request.solver_settings.time_limit if request.solver_settings else None

        # 1. In-memory Cache Check
        cache_key = build_cache_key(
            cohort_fingerprint=cohort_fp,
            budget=request.budget,
            mode=request.mode,
            mip_rel_gap=mip_gap,
            time_limit=time_lim,
        )
        if use_cache:
            cached_resp = self.cache.get(cache_key)
            if cached_resp is not None:
                cached_resp.cache_hit = True
                cached_resp.provenance = "in_memory_cache"
                cached_resp.elapsed_seconds = time.time() - t_start
                return cached_resp

        # 2. Persisted Frontier Artifact Check (Exact Match)
        if check_persistence and self.persistence_dir and os.path.isdir(self.persistence_dir):
            filename = f"frontier_{cohort_fp[:8]}_b{int(request.budget)}_{request.mode}.json"
            filepath = os.path.join(self.persistence_dir, filename)
            persisted_data = load_frontier_result(
                filepath=filepath,
                expected_fingerprint=cohort_fp,
                expected_budget=request.budget,
                expected_mode=request.mode,
            )
            if persisted_data:
                resp = OptimizationResponse.from_dict(persisted_data)
                resp.cache_hit = False
                resp.provenance = "persisted_precomputation"
                resp.elapsed_seconds = time.time() - t_start
                if use_cache:
                    self.cache.put(cache_key, resp)
                return resp

        # 3. Live Optimization Dispatch
        if request.mode == "greedy":
            res = optimize_greedy_allocation(
                profit_matrix=candidate_profit_matrix,
                cost_matrix=candidate_cost_matrix,
                budget=request.budget,
            )
        elif request.mode == "exact":
            res = optimize_budget_allocation(
                profit_matrix=candidate_profit_matrix,
                cost_matrix=candidate_cost_matrix,
                budget=request.budget,
                mip_rel_gap=None,
                time_limit=None,
            )
        elif request.mode == "bounded_gap":
            gap = mip_gap if mip_gap is not None else 0.005
            res = optimize_budget_allocation(
                profit_matrix=candidate_profit_matrix,
                cost_matrix=candidate_cost_matrix,
                budget=request.budget,
                mip_rel_gap=gap,
                time_limit=time_lim,
            )
        elif request.mode == "bounded_time":
            tl = time_lim if time_lim is not None else 15.0
            res = optimize_budget_allocation(
                profit_matrix=candidate_profit_matrix,
                cost_matrix=candidate_cost_matrix,
                budget=request.budget,
                mip_rel_gap=mip_gap,
                time_limit=tl,
            )
        else:
            raise ValueError(f"Unsupported mode: {request.mode}")

        elapsed = time.time() - t_start
        solver_time = res.get("elapsed_seconds", 0.0)

        # Build allocations
        alloc_df = res.get("allocation")
        allocations_list = []
        if alloc_df is not None:
            for cid, row in alloc_df.iterrows():
                allocations_list.append(CustomerAllocationItem(
                    customer_id=str(cid),
                    assigned_arm=int(row["assigned_arm"]),
                    discount_rate=float(row["discount_rate"]),
                    expected_incremental_profit=float(row["expected_incremental_profit"]),
                    allocated_budget_cost=float(row["allocated_budget_cost"]),
                ))

        arm_counts_raw = res.get("arm_counts", {})
        arm_counts_str = {str(k): int(v) for k, v in arm_counts_raw.items()}
        n_cust = len(candidate_profit_matrix)
        tot_profit = float(res.get("total_expected_incremental_profit", 0.0))

        metrics = AggregateMetrics(
            total_expected_incremental_profit=tot_profit,
            total_budget_spent=float(res.get("total_budget_spent", 0.0)),
            budget_remaining=float(res.get("budget_remaining", 0.0)),
            arm_counts=arm_counts_str,
            profit_per_customer=tot_profit / n_cust if n_cust > 0 else 0.0,
            cohort_size=n_cust,
        )

        raw_status = res.get("solver_message", "")
        is_opt_proven = bool(res.get("is_optimal_proven", False))
        is_feas = bool(res.get("is_feasible", False))

        response = OptimizationResponse(
            request_budget=request.budget,
            optimization_mode=request.mode,
            method_executed=str(res.get("method", request.mode)),
            status=str(res.get("status", "unknown")),
            solver_message=raw_status,
            is_feasible=is_feas,
            is_optimal_proven=is_opt_proven,
            cache_hit=False,
            elapsed_seconds=elapsed,
            solver_time_seconds=solver_time,
            mip_gap=res.get("mip_gap"),
            mip_dual_bound=res.get("mip_dual_bound"),
            metrics=metrics,
            allocations=allocations_list,
            raw_solver_status=raw_status,
            provenance="live_optimization",
        )

        # Only cache feasible outcomes
        if use_cache and is_feas:
            self.cache.put(cache_key, response)

        return response
