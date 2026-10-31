"""FastAPI Application Entry Point for EconoCausal."""
from fastapi import FastAPI, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from src.api.config import settings
from src.api.schemas import HealthResponse, RootResponse, ErrorResponse

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=settings.DESCRIPTION,
    docs_url=settings.DOCS_URL,
    redoc_url=settings.REDOC_URL,
    openapi_url=settings.OPENAPI_URL,
)

# Standard CORS Middleware for future frontend/microservice communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get(
    "/",
    response_model=RootResponse,
    status_code=status.HTTP_200_OK,
    summary="Root service metadata",
    tags=["General"]
)
def root() -> RootResponse:
    """Return API metadata and reference links for documentation and health checks."""
    return RootResponse(
        name=settings.PROJECT_NAME,
        version=settings.VERSION,
        description=settings.DESCRIPTION,
        docs_url=settings.DOCS_URL,
        health_url="/health"
    )

@app.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Service health check",
    tags=["Diagnostics"]
)
def health_check() -> HealthResponse:
    """Check API operational status and internal service availability."""
    return HealthResponse(
        status="healthy",
        version=settings.VERSION
    )
