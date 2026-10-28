"""
Public module interfaces for the EconoCausal optimization subsystem.
Exposes optimization services, algorithms, caching abstractions, and API schemas.
"""

from src.optimization.budget import (
    optimize_budget_allocation,
    optimize_greedy_allocation,
)
from src.optimization.schemas import (
    OptimizationRequest,
    SolverSettings,
    OptimizationResponse,
    AggregateMetrics,
    CustomerAllocationItem,
)
from src.optimization.cache import (
    OptimizationCache,
    compute_cohort_fingerprint,
    build_cache_key,
)
from src.optimization.service import (
    OptimizationService,
)
from src.optimization.persistence import (
    save_frontier_result,
    load_frontier_result,
)

__all__ = [
    "optimize_budget_allocation",
    "optimize_greedy_allocation",
    "OptimizationRequest",
    "SolverSettings",
    "OptimizationResponse",
    "AggregateMetrics",
    "CustomerAllocationItem",
    "OptimizationCache",
    "compute_cohort_fingerprint",
    "build_cache_key",
    "OptimizationService",
    "save_frontier_result",
    "load_frontier_result",
]
