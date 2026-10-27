"""
API request and response schemas for the EconoCausal optimization service layer.
Framework-independent dataclasses supporting strict type validation,
range checks, and JSON serialization.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
import numpy as np


@dataclass
class SolverSettings:
    mip_rel_gap: Optional[float] = None
    time_limit: Optional[float] = None
    allow_greedy_fallback: bool = True

    def validate(self) -> None:
        if self.mip_rel_gap is not None:
            if not np.isfinite(self.mip_rel_gap) or self.mip_rel_gap < 0.0:
                raise ValueError(f"mip_rel_gap must be non-negative and finite, got {self.mip_rel_gap}")
        if self.time_limit is not None:
            if not np.isfinite(self.time_limit) or self.time_limit <= 0.0:
                raise ValueError(f"time_limit must be positive and finite, got {self.time_limit}")


@dataclass
class OptimizationRequest:
    budget: float
    mode: str = "bounded_gap"  # "exact", "bounded_gap", "bounded_time", "greedy"
    solver_settings: SolverSettings = field(default_factory=SolverSettings)
    cohort_id: Optional[str] = "default_full_population"

    def validate(self) -> None:
        if not np.isfinite(self.budget):
            raise ValueError(f"Budget must be a finite numerical value, got {self.budget}")
        if self.budget < 0.0:
            raise ValueError(f"Budget cannot be negative, got {self.budget}")
        valid_modes = {"exact", "bounded_gap", "bounded_time", "greedy"}
        if self.mode not in valid_modes:
            raise ValueError(f"Invalid mode '{self.mode}'. Supported modes: {sorted(list(valid_modes))}")
        self.solver_settings.validate()


@dataclass
class CustomerAllocationItem:
    customer_id: str
    assigned_arm: int
    discount_rate: float
    expected_incremental_profit: float
    allocated_budget_cost: float


@dataclass
class AggregateMetrics:
    total_expected_incremental_profit: float
    total_budget_spent: float
    budget_remaining: float
    arm_counts: Dict[int, int]
    profit_per_customer: float
    cohort_size: int


@dataclass
class OptimizationResponse:
    request_budget: float
    optimization_mode: str
    method_executed: str
    status: str
    solver_message: str
    is_feasible: bool
    is_optimal_proven: bool
    cache_hit: bool
    elapsed_seconds: float
    solver_time_seconds: float
    mip_gap: Optional[float]
    mip_dual_bound: Optional[float]
    metrics: AggregateMetrics
    allocations: List[CustomerAllocationItem] = field(default_factory=list)

    def to_dict(self, include_allocations: bool = True) -> Dict[str, Any]:
        data = asdict(self)
        if not include_allocations:
            data.pop("allocations", None)
        return data
