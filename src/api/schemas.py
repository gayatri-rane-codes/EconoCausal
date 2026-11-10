"""
Pydantic schemas for EconoCausal API request and response contracts.
"""
from datetime import datetime, timezone
from typing import Annotated, Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field, field_validator, ConfigDict


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
    cate_arm_3_vs_0: float = Field(..., description="Estimated CATE uplift for 15% discount (Arm 3 vs Control 0)")


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
    recommended_arm: int = Field(..., ge=0, le=3, description="Optimal discount arm: 0 (0%), 1 (5%), 2 (10%), 3 (15%)")
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
    allow_greedy_fallback: bool = Field(
        default=True,
        description="Whether to fall back to greedy heuristic if MIP solver exceeds time limit",
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



# --- Phase 4 Day 4 Schemas ---

class FrontierPointSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")
    budget: float
    total_expected_incremental_profit: float
    total_budget_spent: float
    budget_remaining: float
    is_feasible: bool
    status: str
    method_executed: str
    solver_time_seconds: Optional[float] = None
    mip_gap: Optional[float] = None
    cache_hit: bool = False

class BudgetFrontierRequestSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")
    budgets: List[float] = Field(..., min_length=1, description="List of budget values to sweep")
    mode: Literal["exact", "bounded_gap", "greedy"] = "bounded_gap"
    cohort_id: Optional[str] = None
    use_cache: bool = True
    check_persistence: bool = True
    solver_settings: Optional[OptimizationSolverSettingsSchema] = None
    candidate_profit_matrix: Optional[Dict[str, Dict[str, float]]] = None
    candidate_cost_matrix: Optional[Dict[str, Dict[str, float]]] = None

class BudgetFrontierResponseSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")
    cohort_id: Optional[str] = None
    mode: str
    num_points: int
    frontier: List[FrontierPointSchema]

class SensitivityScenarioResultSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scenario_name: str
    parameter_varied: str
    parameter_value: Any
    total_expected_incremental_profit: float
    total_budget_spent: float
    budget_remaining: float
    profit_delta: float
    spend_delta: float
    is_feasible: bool
    status: str
    solver_time_seconds: Optional[float] = None

class SensitivityAnalysisRequestSchema(BaseModel):
    @field_validator("budget_multipliers", "cost_multipliers")
    @classmethod
    def validate_multipliers(cls, values):
        if values is not None:
            for v in values:
                if v < 0.0:
                    raise ValueError(f"Multiplier values must be non-negative, got {v}")
        return values

    model_config = ConfigDict(extra="forbid")
    base_budget: float = Field(..., ge=0.0)
    mode: Literal["exact", "bounded_gap", "greedy"] = "bounded_gap"
    cohort_id: Optional[str] = None
    budget_multipliers: List[float] = Field(default_factory=lambda: [0.8, 1.0, 1.2])
    cost_multipliers: List[float] = Field(default_factory=lambda: [0.9, 1.0, 1.1])
    solver_settings: Optional[OptimizationSolverSettingsSchema] = None
    candidate_profit_matrix: Optional[Dict[str, Dict[str, float]]] = None
    candidate_cost_matrix: Optional[Dict[str, Dict[str, float]]] = None

class SensitivityAnalysisResponseSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")
    base_budget: float
    mode: str
    baseline_profit: float
    baseline_spend: float
    scenarios: List[SensitivityScenarioResultSchema]

class CohortOptimizationItemSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")
    cohort_id: str
    budget: float = Field(..., ge=0.0)
    mode: Literal["exact", "bounded_gap", "greedy"] = "bounded_gap"
    use_cache: bool = True
    check_persistence: bool = True
    include_allocations: bool = False
    solver_settings: Optional[OptimizationSolverSettingsSchema] = None
    candidate_profit_matrix: Optional[Dict[str, Dict[str, float]]] = None
    candidate_cost_matrix: Optional[Dict[str, Dict[str, float]]] = None

class MultiCohortOptimizationRequestSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")
    cohorts: List[CohortOptimizationItemSchema] = Field(..., min_length=1)

class CohortErrorSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")
    cohort_id: str
    error: str

class MultiCohortOptimizationResponseSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")
    total_cohorts: int
    successful_cohorts: int
    failed_cohorts: int
    results: Dict[str, OptimizationResponseSchema]
    errors: List[CohortErrorSchema]

class CausalOptimizationPipelineRequestSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")
    customers: List[CustomerFeatureRecord] = Field(..., min_length=1)
    budget: float = Field(..., ge=0.0)
    mode: Literal["exact", "bounded_gap", "greedy"] = "bounded_gap"
    cohort_id: Optional[str] = "cohort_default"
    include_allocations: bool = True
    use_cache: bool = True
    check_persistence: bool = True
    solver_settings: Optional[OptimizationSolverSettingsSchema] = None
    default_baseline_prob: float = Field(0.23, ge=0.0, le=1.0)
    spend_tail_threshold: float = Field(60000.0, ge=0.0)


class CausalOptimizationPipelineResponseSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: str
    cohort_id: str
    total_customers: int
    optimization_mode: str
    is_feasible: bool
    metrics: AggregateMetricsSchema
    allocations: Optional[List[CustomerAllocationItemSchema]] = None
    solver_status: Optional[str] = None
    solver_time_seconds: Optional[float] = None
    cache_hit: bool = False
    provenance: str = "pipeline_dispatch"

# ============================================================================
# Data Drift Detection Schemas (Phase 4 - Day 6)
# ============================================================================

class FeatureDriftMetricSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")
    test_type: str = Field(..., description="Statistical test used (kolmogorov_smirnov or chi_square)")
    statistic: float = Field(..., description="Test statistic value")
    p_value: float = Field(..., description="P-value from two-sample test")
    drift_detected: bool = Field(..., description="True if feature has statistically significant drift")
    sample_size_reference: int = Field(..., ge=0, description="Clean sample count in reference dataset")
    sample_size_current: int = Field(..., ge=0, description="Clean sample count in current dataset")
    warning: Optional[str] = Field(default=None, description="Optional diagnostic warning or edge-case notes")


class DataDriftRequestSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reference_data: List[CustomerFeatureRecord] = Field(
        ..., min_length=1, description="Baseline reference customer records (e.g. training distribution)"
    )
    current_data: List[CustomerFeatureRecord] = Field(
        ..., min_length=1, description="New / live candidate customer records to test for drift"
    )
    alpha: float = Field(
        default=0.05, gt=0.0, lt=1.0, description="Significance level alpha for hypothesis tests (default 0.05)"
    )
    drift_share_threshold: float = Field(
        default=0.3, gt=0.0, le=1.0, description="Proportion of drifted features required to flag overall dataset drift"
    )


class DataDriftResponseSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")
    overall_drift_detected: bool = Field(..., description="Overall decision whether the dataset has drifted")
    drift_share: float = Field(..., ge=0.0, le=1.0, description="Ratio of drifted features to total evaluated features")
    drift_share_threshold: float = Field(..., description="Configured drift share threshold applied")
    alpha_significance_level: float = Field(..., description="Significance level alpha applied")
    total_features_evaluated: int = Field(..., ge=0, description="Total confounder features evaluated")
    drifted_features_count: int = Field(..., ge=0, description="Number of confounder features with detected drift")
    feature_metrics: Dict[str, FeatureDriftMetricSchema] = Field(
        ..., description="Feature-by-feature drift metrics and test statistics"
    )
    summary: str = Field(..., description="Human-readable assessment summary")

# ============================================================================
# Data Drift Detection Schemas (Phase 4 - Day 6)
# ============================================================================

class FeatureDriftMetricSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")
    test_type: str = Field(..., description="Statistical test used (kolmogorov_smirnov or chi_square)")
    statistic: float = Field(..., description="Test statistic value")
    p_value: float = Field(..., description="P-value from two-sample test")
    drift_detected: bool = Field(..., description="True if feature has statistically significant drift")
    sample_size_reference: int = Field(..., ge=0, description="Clean sample count in reference dataset")
    sample_size_current: int = Field(..., ge=0, description="Clean sample count in current dataset")
    warning: Optional[str] = Field(default=None, description="Optional diagnostic warning or edge-case notes")


class DataDriftRequestSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reference_data: List[CustomerFeatureRecord] = Field(
        ..., min_length=1, description="Baseline reference customer records (e.g. training distribution)"
    )
    current_data: List[CustomerFeatureRecord] = Field(
        ..., min_length=1, description="New / live candidate customer records to test for drift"
    )
    alpha: float = Field(
        default=0.05, gt=0.0, lt=1.0, description="Significance level alpha for hypothesis tests (default 0.05)"
    )
    drift_share_threshold: float = Field(
        default=0.3, gt=0.0, le=1.0, description="Proportion of drifted features required to flag overall dataset drift"
    )


class DataDriftResponseSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")
    overall_drift_detected: bool = Field(..., description="Overall decision whether the dataset has drifted")
    drift_share: float = Field(..., ge=0.0, le=1.0, description="Ratio of drifted features to total evaluated features")
    drift_share_threshold: float = Field(..., description="Configured drift share threshold applied")
    alpha_significance_level: float = Field(..., description="Significance level alpha applied")
    total_features_evaluated: int = Field(..., ge=0, description="Total confounder features evaluated")
    drifted_features_count: int = Field(..., ge=0, description="Number of confounder features with detected drift")
    feature_metrics: Dict[str, FeatureDriftMetricSchema] = Field(
        ..., description="Feature-by-feature drift metrics and test statistics"
    )
    summary: str = Field(..., description="Human-readable assessment summary")

