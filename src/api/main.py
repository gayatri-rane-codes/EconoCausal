"""
Main application factory for the EconoCausal FastAPI backend.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.config import settings
from src.api.schemas import HealthResponse, RootResponse
from src.api.routers.causal import router as causal_router
from src.api.routers.optimization import router as optimization_router

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=settings.DESCRIPTION,
    docs_url=settings.DOCS_URL,
    redoc_url=settings.REDOC_URL,
    openapi_url=settings.OPENAPI_URL,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(causal_router)
app.include_router(optimization_router)


@app.get("/", response_model=RootResponse, tags=["Diagnostic & Info"])
def get_root() -> RootResponse:
    return RootResponse(
        name=settings.PROJECT_NAME,
        version=settings.VERSION,
        description=settings.DESCRIPTION,
        docs_url=settings.DOCS_URL,
        openapi_url=settings.OPENAPI_URL,
        health_url="/health",
    )


@app.get("/health", response_model=HealthResponse, tags=["Diagnostic & Info"])
def get_health() -> HealthResponse:
    services_status = {
        "causal_engine": "ready",
        "optimization_service": "ready",
    }
    return HealthResponse(
        status="healthy",
        version=settings.VERSION,
        services=services_status,
    )
