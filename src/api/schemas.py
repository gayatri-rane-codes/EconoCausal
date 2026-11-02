"""
Pydantic schemas for EconoCausal API request and response contracts.
"""
from datetime import datetime, timezone
from typing import Dict, List, Optional, Literal
from pydantic import BaseModel, Field, ConfigDict


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: str = Field(default="healthy", description="Service status indicator")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="Current UTC timestamp in ISO 8601",
    )
    version: str = Field(..., description="Application version")
    services: Dict[str, str] = Field(
        default_factory=dict,
        description="Readiness status of internal services",
    )


class RootResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(..., description="Service name")
    version: str = Field(..., description="Service semantic version")
    description: str = Field(..., description="Brief service description")
    docs_url: str = Field(..., description="Swagger UI documentation endpoint")
    openapi_url: str = Field(..., description="OpenAPI schema JSON endpoint")
    health_url: str = Field(default="/health", description="Health check endpoint")


class ErrorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    detail: str = Field(..., description="Human-readable description of error")


# --- Causal Inference Schemas ---

class CustomerFeatureRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    customer_id: Optional[str] = Field(default=None, description="Optional customer identifier")
    browsing_frequency: float = Field(..., ge=0.0, description="Browsing frequency score or count")
    cart_abandonment_rate: float = Field(..., ge=0.0, le=1.0, description="Cart abandonment rate in [0, 1]")
    competitor_price_index: float = Field(..., ge=0.0, description="Competitor price index ratio")
    historical_spend: float = Field(..., ge=0.0, description="Customer historical lifetime spend")
    loyalty_member: int = Field(..., ge=0, le=1, description="Loyalty program indicator (0 or 1)")
    previous_discount_response_rate: float = Field(..., ge=0.0, le=1.0, description="Historical response rate in [0, 1]")
    recency_days: float = Field(..., ge=0.0, description="Days since last purchase")
    seasonality_index: float = Field(..., ge=0.0, description="Seasonality demand index")

    basket_value: Optional[float] = Field(default=100.0, ge=0.0, description="Projected basket value in currency units")
    cost_of_goods: Optional[float] = Field(default=None, ge=0.0, description="Cost of goods for margin calculation")
    baseline_purchase_prob: Optional[float] = Field(default=0.35, ge=0.0, le=1.0, description="Baseline conversion probability")


class CausalBatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    customers: List[CustomerFeatureRecord] = Field(..., min_length=1, description="List of customer feature records")


class CATEEffect(BaseModel):
    model_config = ConfigDict(extra="forbid")
    customer_id: Optional[str] = Field(default=None)
    cate_arm_1_vs_0: float = Field(..., description="Estimated CATE uplift for 5% discount (Arm 1 vs Control 0)")
    cate_arm_2_vs_0: float = Field(..., description="Estimated CATE uplift for 10% discount (Arm 2 vs Control 0)")
    cate_arm_3_vs_0: float = Field(..., description="Estimated CATE uplift for 20% discount (Arm 3 vs Control 0)")


class CATEEstimateResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    results: List[CATEEffect]
    count: int


class PolicyRecommendation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    customer_id: Optional[str] = Field(default=None)
    cate_arm_1_vs_0: float
    cate_arm_2_vs_0: float
    cate_arm_3_vs_0: float
    recommended_arm: int = Field(..., ge=0, le=3, description="Optimal discount arm: 0 (0%), 1 (5%), 2 (10%), 3 (20%)")
    expected_incremental_profit: float = Field(..., description="Estimated incremental expected profit from recommendation")


class PolicyPredictResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    recommendations: List[PolicyRecommendation]
    count: int

# -----------------------------------------------------------------------
# Optimization Schemas
# ----------------------------------------------------------------------

class OptimizationSolverSettingsSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mip_rel_gap: Optional[float] = Field(
        default=0.01,
        ge=0.0,
        description="Target relative optimality gap for MIP solver",
    )
    time_limit: Optional[float] = Field(
        default=30.0,
        gt=0.0,
        description="Solver maximum wall-clock time in seconds",
    )
    warm_start: bool = Field(
        default=True,
        description="Whether to use greedy heuristic solution as warm start",
    )
    verbose: bool = Field(
        default=False,
        description="Whether to output detailed solver log",
    )


class OptimizationRequestSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    budget: float = Field(
        ...,
        ge=0.0,
        description="Maximum budget allocated across all customers",
    )
    mode: Literal["exact", "bounded_gap", "bounded_time", "greedy"] = Field(
        default="bounded_gap",
        description="Solver execution mode",
    )
    solver_settings: Optional[OptimizationSolverSettingsSchema] = Field(
        default=None,
        description="Fine-grained solver convergence parameters",
    )
    cohort_id: str = Field(
        default="default",
        description="Cohort key used for lookup and caching",
    )
    use_cache: bool = Field(
        default=True,
        description="Enable in-memory LRU cache lookup",
    )
    check_persistence: bool = Field(
        default=True,
        description="Enable disk precomputed cache lookup",
    )
    include_allocations: bool = Field(
        default=True,
        description="Whether to return per-customer decision allocations",
    )
    candidate_profit_matrix: Optional[Dict[str, Dict[str, float]]] = Field(
        default=None,
        description="Optional customer x arm profit matrix override",
    )
    candidate_cost_matrix: Optional[Dict[str, Dict[str, float]]] = Field(
        default=None,
        description="Optional customer x arm cost matrix override",
    )


class AggregateMetricsSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total_expected_incremental_profit: float = Field(
        ..., description="Sum of expected incremental profit under assignment"
    )
    total_budget_spent: float = Field(
        ..., description="Total promotional budget expended"
    )
    budget_remaining: float = Field(
        ..., description="Unspent budget"
    )
    arm_counts: Dict[str, int] = Field(
        ..., description="Count of customers assigned to each treatment arm"
    )
    profit_per_customer: float = Field(
        ..., description="Mean incremental profit per cohort member"
    )
    cohort_size: int = Field(
        ..., description="Total number of evaluated customers"
    )


class CustomerAllocationItemSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    customer_id: str = Field(..., description="Unique customer identifier")
    assigned_arm: int = Field(..., ge=0, le=3, description="Assigned treatment arm index")
    discount_rate: float = Field(..., ge=0.0, le=1.0, description="Nominal discount rate applied")
    expected_incremental_profit: float = Field(..., description="Expected incremental profit")
    allocated_budget_cost: float = Field(..., ge=0.0, description="Cost of promo discount")


class OptimizationResponseSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

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
    mip_gap: Optional[float] = None
    mip_dual_bound: Optional[float] = None
    metrics: AggregateMetricsSchema
    allocations: List[CustomerAllocationItemSchema] = Field(default_factory=list)
    raw_solver_status: Optional[str] = None
    provenance: str

