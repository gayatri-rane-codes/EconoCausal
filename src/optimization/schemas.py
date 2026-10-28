"""
Strict dataclass schemas for the optimization service layer.
Defines public API contracts, validation invariants, and serialization helpers.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
import math


@dataclass
class SolverSettings:
    """Solver tuning parameters."""
    mip_rel_gap: Optional[float] = None
    time_limit: Optional[float] = None
    allow_greedy_fallback: bool = True

    def validate(self) -> None:
        if self.mip_rel_gap is not None:
            if not isinstance(self.mip_rel_gap, (int, float)) or math.isnan(self.mip_rel_gap) or math.isinf(self.mip_rel_gap) or self.mip_rel_gap < 0.0:
                raise ValueError(f"mip_rel_gap must be non-negative and finite, got {self.mip_rel_gap}")
        if self.time_limit is not None:
            if not isinstance(self.time_limit, (int, float)) or math.isnan(self.time_limit) or math.isinf(self.time_limit) or self.time_limit <= 0.0:
                raise ValueError(f"time_limit must be strictly positive and finite, got {self.time_limit}")


@dataclass
class OptimizationRequest:
    """Request contract for initiating a budget optimization run."""
    budget: float
    mode: str = "bounded_gap"
    solver_settings: Optional[SolverSettings] = None
    cohort_id: str = "default"

    def validate(self) -> None:
        if not isinstance(self.budget, (int, float)) or math.isnan(self.budget) or math.isinf(self.budget):
            raise ValueError(f"Budget must be a finite numerical value, got {self.budget}")
        if self.budget < 0.0:
            raise ValueError(f"Budget cannot be negative, got {self.budget}")
        
        valid_modes = ["exact", "bounded_gap", "bounded_time", "greedy"]
        if self.mode not in valid_modes:
            raise ValueError(f"Invalid mode '{self.mode}'. Supported modes: {valid_modes}")

        if self.solver_settings is not None:
            self.solver_settings.validate()


@dataclass
class AggregateMetrics:
    """Summary metrics of the optimization run."""
    total_expected_incremental_profit: float
    total_budget_spent: float
    budget_remaining: float
    arm_counts: Dict[str, int]
    profit_per_customer: float
    cohort_size: int


@dataclass
class CustomerAllocationItem:
    """Decision item for an individual customer."""
    customer_id: str
    assigned_arm: int
    discount_rate: float
    expected_incremental_profit: float
    allocated_budget_cost: float


@dataclass
class OptimizationResponse:
    """Structured response contract for optimization results."""
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
    raw_solver_status: Optional[str] = None
    provenance: str = "live_optimization"  # "live_optimization", "in_memory_cache", or "persisted_precomputation"

    def to_dict(self, include_allocations: bool = True) -> Dict[str, Any]:
        data = asdict(self)
        if not include_allocations:
            data.pop("allocations", None)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OptimizationResponse":
        m_dict = data.get("metrics", {})
        metrics = AggregateMetrics(**m_dict) if isinstance(m_dict, dict) else m_dict

        allocs = []
        for a in data.get("allocations", []):
            if isinstance(a, dict):
                allocs.append(CustomerAllocationItem(**a))
            else:
                allocs.append(a)

        return cls(
            request_budget=float(data["request_budget"]),
            optimization_mode=str(data["optimization_mode"]),
            method_executed=str(data["method_executed"]),
            status=str(data["status"]),
            solver_message=str(data["solver_message"]),
            is_feasible=bool(data["is_feasible"]),
            is_optimal_proven=bool(data["is_optimal_proven"]),
            cache_hit=bool(data.get("cache_hit", False)),
            elapsed_seconds=float(data.get("elapsed_seconds", 0.0)),
            solver_time_seconds=float(data.get("solver_time_seconds", 0.0)),
            mip_gap=data.get("mip_gap"),
            mip_dual_bound=data.get("mip_dual_bound"),
            metrics=metrics,
            allocations=allocs,
            raw_solver_status=data.get("raw_solver_status"),
            provenance=data.get("provenance", "live_optimization"),
        )
