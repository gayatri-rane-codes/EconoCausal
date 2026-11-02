"""Optimization API router exposing endpoints for budget allocation solving."""

import logging
from typing import Dict, Tuple, Any
import pandas as pd
from fastapi import APIRouter, HTTPException, status

from src.api.schemas import (
    ErrorResponse,
    OptimizationRequestSchema,
    OptimizationResponseSchema,
    AggregateMetricsSchema,
    CustomerAllocationItemSchema,
)
from src.optimization.service import OptimizationService
from src.optimization.schemas import (
    OptimizationRequest,
    SolverSettings,
    OptimizationResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/optimization",
    tags=["Budget Optimization"],
)

# Shared service instance
optimization_service = OptimizationService(cache_size=128)


def _get_default_candidate_matrices() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generates a standard default candidate cohort if none is explicitly provided."""
    customers = [f"cust_{i:04d}" for i in range(1, 101)]

    profit_data = {
        "arm_0": [0.0] * 100,
        "arm_1": [8.5] * 100,
        "arm_2": [14.0] * 100,
        "arm_3": [18.2] * 100,
    }
    cost_data = {
        "arm_0": [0.0] * 100,
        "arm_1": [5.0] * 100,
        "arm_2": [10.0] * 100,
        "arm_3": [15.0] * 100,
    }
    return pd.DataFrame(profit_data, index=customers), pd.DataFrame(cost_data, index=customers)


def _build_dataframes(
    profit_dict: Dict[str, Dict[str, float]],
    cost_dict: Dict[str, Dict[str, float]],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    profit_df = pd.DataFrame.from_dict(profit_dict, orient="index")
    cost_df = pd.DataFrame.from_dict(cost_dict, orient="index")
    return profit_df, cost_df


@router.post(
    "/optimize",
    response_model=OptimizationResponseSchema,
    status_code=status.HTTP_200_OK,
    responses={
        422: {"model": ErrorResponse, "description": "Validation Error"},
        500: {"model": ErrorResponse, "description": "Internal Optimization Error"},
    },
    summary="Solve Multi-Arm Promotional Budget Allocation",
    description="Optimizes discount arm assignments across customers subject to a total budget constraint.",
)
def optimize_budget(
    payload: OptimizationRequestSchema,
) -> OptimizationResponseSchema:
    try:
        if payload.candidate_profit_matrix and payload.candidate_cost_matrix:
            profit_df, cost_df = _build_dataframes(
                payload.candidate_profit_matrix,
                payload.candidate_cost_matrix,
            )
        else:
            profit_df, cost_df = _get_default_candidate_matrices()

        solver_settings = None
        if payload.solver_settings is not None:
            solver_settings = SolverSettings(
                mip_rel_gap=payload.solver_settings.mip_rel_gap,
                time_limit=payload.solver_settings.time_limit,
                allow_greedy_fallback=payload.solver_settings.allow_greedy_fallback,
            )

        internal_request = OptimizationRequest(
            budget=payload.budget,
            mode=payload.mode,
            solver_settings=solver_settings,
            cohort_id=payload.cohort_id,
        )

        result: OptimizationResponse = optimization_service.optimize(
            request=internal_request,
            candidate_profit_matrix=profit_df,
            candidate_cost_matrix=cost_df,
            use_cache=payload.use_cache,
            check_persistence=payload.check_persistence,
        )

        metrics_schema = AggregateMetricsSchema(
            total_expected_incremental_profit=result.metrics.total_expected_incremental_profit,
            total_budget_spent=result.metrics.total_budget_spent,
            budget_remaining=result.metrics.budget_remaining,
            arm_counts=result.metrics.arm_counts,
            profit_per_customer=result.metrics.profit_per_customer,
            cohort_size=result.metrics.cohort_size,
        )

        allocations_schema = []
        if payload.include_allocations and result.allocations:
            allocations_schema = [
                CustomerAllocationItemSchema(
                    customer_id=item.customer_id,
                    assigned_arm=item.assigned_arm,
                    discount_rate=item.discount_rate,
                    expected_incremental_profit=item.expected_incremental_profit,
                    allocated_budget_cost=item.allocated_budget_cost,
                )
                for item in result.allocations
            ]

        return OptimizationResponseSchema(
            request_budget=result.request_budget,
            optimization_mode=result.optimization_mode,
            method_executed=result.method_executed,
            status=result.status,
            solver_message=result.solver_message,
            is_feasible=result.is_feasible,
            is_optimal_proven=result.is_optimal_proven,
            cache_hit=result.cache_hit,
            elapsed_seconds=result.elapsed_seconds,
            solver_time_seconds=result.solver_time_seconds,
            mip_gap=result.mip_gap,
            mip_dual_bound=result.mip_dual_bound,
            metrics=metrics_schema,
            allocations=allocations_schema,
            raw_solver_status=result.raw_solver_status,
            provenance=result.provenance,
        )

    except HTTPException:
        raise
    except ValueError as e:
        logger.warning(f"Validation error during optimization: {e}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Internal error during optimization: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Optimization failed due to an internal error: {e}",
        )
