"""Pydantic Schemas for API Requests, Responses, and Errors."""
from datetime import datetime, timezone
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field

class HealthResponse(BaseModel):
    status: str = Field(default="healthy", description="Current service health status")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 UTC timestamp of the health check"
    )
    version: str = Field(description="API version")
    services: Dict[str, str] = Field(
        default_factory=lambda: {
            "causal_engine": "ready",
            "optimization_service": "ready"
        },
        description="Status of internal subsystem dependencies"
    )

class RootResponse(BaseModel):
    name: str = Field(description="API service name")
    version: str = Field(description="API version")
    description: str = Field(description="Service description")
    docs_url: str = Field(description="Swagger UI documentation URL")
    health_url: str = Field(description="Health check endpoint URL")

class ErrorResponse(BaseModel):
    detail: str = Field(description="Human-readable error description")
    error_code: Optional[str] = Field(default=None, description="Domain-specific error code")
