"""API Configuration and Metadata Settings."""
from dataclasses import dataclass

@dataclass(frozen=True)
class Settings:
    PROJECT_NAME: str = "EconoCausal API"
    VERSION: str = "1.0.0"
    DESCRIPTION: str = (
        "Dynamic Pricing and Causal Uplift Optimization Service "
        "leveraging Double Machine Learning (LinearDML) and MIP Budget Allocation."
    )
    DOCS_URL: str = "/docs"
    REDOC_URL: str = "/redoc"
    OPENAPI_URL: str = "/openapi.json"

settings = Settings()
