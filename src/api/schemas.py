"""
Pydantic schemas for EconoCausal API request and response contracts.
"""
from datetime import datetime, timezone
from typing import Dict, List, Optional
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
