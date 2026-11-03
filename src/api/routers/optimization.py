"""Optimization API router exposing endpoints for budget allocation solving."""

import logging
from typing import Dict, Tuple, Any
import pandas as pd
from fastapi import APIRouter, HTTPException, status

from src.api.schemas import (
    BudgetFrontierRequestSchema,
    BudgetFrontierResponseSchema,
    FrontierPointSchema,
    SensitivityAnalysisRequestSchema,
    SensitivityAnalysisResponseSchema,
    SensitivityScenarioResultSchema,
    MultiCohortOptimizationRequestSchema,
    MultiCohortOptimizationResponseSchema,
    CohortErrorSchema,

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



def _map_solver_settings(settings_schema):
    if settings_schema is None:
        return None
    return SolverSettings(
        mip_rel_gap=settings_schema.mip_rel_gap,
        time_limit=settings_schema.time_limit,
        allow_greedy_fallback=settings_schema.allow_greedy_fallback,
    )


def _map_domain_response_to_schema(result: OptimizationResponse, include_allocations: bool = False) -> OptimizationResponseSchema:
    metrics_schema = AggregateMetricsSchema(
        total_expected_incremental_profit=result.metrics.total_expected_incremental_profit,
        total_budget_spent=result.metrics.total_budget_spent,
        budget_remaining=result.metrics.budget_remaining,
        arm_counts=result.metrics.arm_counts,
        profit_per_customer=result.metrics.profit_per_customer,
        cohort_size=result.metrics.cohort_size,
    )

    allocations_schema = []
    if include_allocations and result.allocations:
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

def _build_dataframes(
    profit_dict: Dict[str, Dict[str, float]],
    cost_dict: Dict[str, Dict[str, float]],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    profit_df = pd.DataFrame.from_dict(profit_dict, orient="index")
    cost_df = pd.DataFrame.from_dict(cost_dict, orient="index")
    return profit_df, cost_df


@router.post(
    "/allocate",
    response_model=OptimizationResponseSchema,
    status_code=status.HTTP_200_OK,
    responses={
        422: {"model": ErrorResponse, "description": "Validation Error"},
        500: {"model": ErrorResponse, "description": "Internal Optimization Error"},
        503: {"model": ErrorResponse, "description": "Optimization Service Unavailable"},
    },
    summary="Allocate Multi-Arm Promotional Budget",
    description="Allocates promotional discount arms to customers under budget constraints.",
)
@router.post(
    "/optimize",
    response_model=OptimizationResponseSchema,
    status_code=status.HTTP_200_OK,
    include_in_schema=False,
)
def optimize_budget(
    payload: OptimizationRequestSchema,
) -> OptimizationResponseSchema:
    try:
        if optimization_service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Optimization service is currently unavailable.",
            )
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


@router.post(
    "/frontier",
    response_model=BudgetFrontierResponseSchema,
    status_code=status.HTTP_200_OK,
    responses={
        422: {"model": ErrorResponse, "description": "Validation Error"},
        500: {"model": ErrorResponse, "description": "Internal Optimization Error"},
        503: {"model": ErrorResponse, "description": "Optimization Service Unavailable"},
    },
    summary="Generate Budget-Profit Frontier",
    description="Evaluates optimal spend and expected incremental profit across a range of budget limits.",
)
def compute_budget_frontier(
    payload: BudgetFrontierRequestSchema,
) -> BudgetFrontierResponseSchema:
    if optimization_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Optimization service is currently unavailable.",
        )

    for b in payload.budgets:
        if b < 0:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid budget {b}: budgets must be non-negative.",
            )

    try:
        if payload.candidate_profit_matrix and payload.candidate_cost_matrix:
            profit_df, cost_df = _build_dataframes(
                payload.candidate_profit_matrix,
                payload.candidate_cost_matrix,
            )
        else:
            profit_df, cost_df = _get_default_candidate_matrices()

        frontier_points = []
        for b in sorted(payload.budgets):
            domain_settings = _map_solver_settings(payload.solver_settings)
            domain_req = OptimizationRequest(
                budget=float(b),
                mode=payload.mode,
                solver_settings=domain_settings,
                cohort_id=payload.cohort_id,
            )
            resp = optimization_service.optimize(
                request=domain_req,
                candidate_profit_matrix=profit_df,
                candidate_cost_matrix=cost_df,
                use_cache=payload.use_cache,
                check_persistence=payload.check_persistence,
            )
            frontier_points.append(
                FrontierPointSchema(
                    budget=float(b),
                    total_expected_incremental_profit=resp.metrics.total_expected_incremental_profit,
                    total_budget_spent=resp.metrics.total_budget_spent,
                    budget_remaining=resp.metrics.budget_remaining,
                    is_feasible=resp.is_feasible,
                    status=resp.status,
                    method_executed=resp.method_executed,
                    solver_time_seconds=resp.solver_time_seconds,
                    mip_gap=resp.mip_gap,
                    cache_hit=resp.cache_hit,
                )
            )

        return BudgetFrontierResponseSchema(
            cohort_id=payload.cohort_id,
            mode=payload.mode,
            num_points=len(frontier_points),
            frontier=frontier_points,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Frontier generation failed due to an internal error: {str(exc)}",
        )


@router.post(
    "/sensitivity",
    response_model=SensitivityAnalysisResponseSchema,
    status_code=status.HTTP_200_OK,
    responses={
        422: {"model": ErrorResponse, "description": "Validation Error"},
        500: {"model": ErrorResponse, "description": "Internal Optimization Error"},
        503: {"model": ErrorResponse, "description": "Optimization Service Unavailable"},
    },
    summary="Perform Budget and Cost Sensitivity Analysis",
    description="Assesses the impact of varying budgets and cost assumptions on expected profit and allocation feasibility.",
)
def compute_sensitivity_analysis(
    payload: SensitivityAnalysisRequestSchema,
) -> SensitivityAnalysisResponseSchema:
    if optimization_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Optimization service is currently unavailable.",
        )

    try:
        if payload.candidate_profit_matrix and payload.candidate_cost_matrix:
            base_profit_df, base_cost_df = _build_dataframes(
                payload.candidate_profit_matrix,
                payload.candidate_cost_matrix,
            )
        else:
            base_profit_df, base_cost_df = _get_default_candidate_matrices()

        domain_settings = _map_solver_settings(payload.solver_settings)

        base_req = OptimizationRequest(
            budget=float(payload.base_budget),
            mode=payload.mode,
            solver_settings=domain_settings,
            cohort_id=payload.cohort_id,
        )
        base_resp = optimization_service.optimize(
            request=base_req,
            candidate_profit_matrix=base_profit_df,
            candidate_cost_matrix=base_cost_df,
            use_cache=False,
            check_persistence=False,
        )
        base_profit = base_resp.metrics.total_expected_incremental_profit
        base_spend = base_resp.metrics.total_budget_spent

        scenarios = []

        for bm in payload.budget_multipliers:
            s_budget = max(0.0, float(payload.base_budget * bm))
            s_req = OptimizationRequest(
                budget=s_budget,
                mode=payload.mode,
                solver_settings=domain_settings,
                cohort_id=payload.cohort_id,
            )
            s_resp = optimization_service.optimize(
                request=s_req,
                candidate_profit_matrix=base_profit_df,
                candidate_cost_matrix=base_cost_df,
                use_cache=False,
                check_persistence=False,
            )
            p = s_resp.metrics.total_expected_incremental_profit
            s = s_resp.metrics.total_budget_spent
            scenarios.append(
                SensitivityScenarioResultSchema(
                    scenario_name=f"budget_scale_{bm:.2f}",
                    parameter_varied="budget_multiplier",
                    parameter_value=bm,
                    total_expected_incremental_profit=p,
                    total_budget_spent=s,
                    budget_remaining=s_resp.metrics.budget_remaining,
                    profit_delta=round(p - base_profit, 4),
                    spend_delta=round(s - base_spend, 4),
                    is_feasible=s_resp.is_feasible,
                    status=s_resp.status,
                    solver_time_seconds=s_resp.solver_time_seconds,
                )
            )

        for cm in payload.cost_multipliers:
            scaled_cost_df = base_cost_df * cm
            s_req = OptimizationRequest(
                budget=float(payload.base_budget),
                mode=payload.mode,
                solver_settings=domain_settings,
                cohort_id=payload.cohort_id,
            )
            s_resp = optimization_service.optimize(
                request=s_req,
                candidate_profit_matrix=base_profit_df,
                candidate_cost_matrix=scaled_cost_df,
                use_cache=False,
                check_persistence=False,
            )
            p = s_resp.metrics.total_expected_incremental_profit
            s = s_resp.metrics.total_budget_spent
            scenarios.append(
                SensitivityScenarioResultSchema(
                    scenario_name=f"cost_scale_{cm:.2f}",
                    parameter_varied="cost_multiplier",
                    parameter_value=cm,
                    total_expected_incremental_profit=p,
                    total_budget_spent=s,
                    budget_remaining=s_resp.metrics.budget_remaining,
                    profit_delta=round(p - base_profit, 4),
                    spend_delta=round(s - base_spend, 4),
                    is_feasible=s_resp.is_feasible,
                    status=s_resp.status,
                    solver_time_seconds=s_resp.solver_time_seconds,
                )
            )

        return SensitivityAnalysisResponseSchema(
            base_budget=payload.base_budget,
            mode=payload.mode,
            baseline_profit=base_profit,
            baseline_spend=base_spend,
            scenarios=scenarios,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Sensitivity analysis failed due to an internal error: {str(exc)}",
        )


@router.post(
    "/multi-cohort",
    response_model=MultiCohortOptimizationResponseSchema,
    status_code=status.HTTP_200_OK,
    responses={
        422: {"model": ErrorResponse, "description": "Validation Error"},
        500: {"model": ErrorResponse, "description": "Internal Optimization Error"},
        503: {"model": ErrorResponse, "description": "Optimization Service Unavailable"},
    },
    summary="Batch Multi-Cohort Budget Allocation",
    description="Executes optimization independently across multiple customer cohorts with isolated error handling.",
)
def compute_multi_cohort_optimization(
    payload: MultiCohortOptimizationRequestSchema,
) -> MultiCohortOptimizationResponseSchema:
    if optimization_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Optimization service is currently unavailable.",
        )

    results = {}
    errors = []

    for item in payload.cohorts:
        try:
            if item.candidate_profit_matrix and item.candidate_cost_matrix:
                profit_df, cost_df = _build_dataframes(
                    item.candidate_profit_matrix,
                    item.candidate_cost_matrix,
                )
            else:
                profit_df, cost_df = _get_default_candidate_matrices()

            domain_settings = _map_solver_settings(item.solver_settings)
            domain_req = OptimizationRequest(
                budget=float(item.budget),
                mode=item.mode,
                solver_settings=domain_settings,
                cohort_id=item.cohort_id,
            )
            opt_resp = optimization_service.optimize(
                request=domain_req,
                candidate_profit_matrix=profit_df,
                candidate_cost_matrix=cost_df,
                use_cache=item.use_cache,
                check_persistence=item.check_persistence,
            )

            results[item.cohort_id] = _map_domain_response_to_schema(
                opt_resp,
                include_allocations=item.include_allocations,
            )
        except Exception as exc:
            errors.append(CohortErrorSchema(cohort_id=item.cohort_id, error=str(exc)))

    return MultiCohortOptimizationResponseSchema(
        total_cohorts=len(payload.cohorts),
        successful_cohorts=len(results),
        failed_cohorts=len(errors),
        results=results,
        errors=errors,
    )
