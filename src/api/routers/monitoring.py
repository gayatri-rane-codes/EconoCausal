"""
FastAPI Router for Automated Covariate and Data Drift Monitoring.
Exposes POST /monitoring/data-drift without retraining models or touching evaluation data.
"""

from fastapi import APIRouter, HTTPException, status
from src.api.schemas import (
    DataDriftRequestSchema,
    DataDriftResponseSchema,
    ErrorResponse,
)
from src.monitoring.drift import detect_dataset_drift

router = APIRouter(prefix="/monitoring", tags=["Monitoring & Diagnostics"])


@router.post(
    "/data-drift",
    response_model=DataDriftResponseSchema,
    status_code=status.HTTP_200_OK,
    responses={
        400: {"model": ErrorResponse, "description": "Validation or statistical execution error"},
        422: {"model": ErrorResponse, "description": "Unprocessable entity payload"},
    },
    summary="Detect Covariate Data Drift",
    description=(
        "Compares reference baseline records against current customer records across the 8 causal "
        "confounders using Kolmogorov-Smirnov (continuous) and Chi-Square (categorical) tests. "
        "Operates under strict ground-truth isolation without triggering automated retraining."
    ),
)
def evaluate_data_drift(request: DataDriftRequestSchema) -> DataDriftResponseSchema:
    try:
        ref_dicts = [cust.model_dump() for cust in request.reference_data]
        curr_dicts = [cust.model_dump() for cust in request.current_data]

        result = detect_dataset_drift(
            reference_records=ref_dicts,
            current_records=curr_dicts,
            alpha=request.alpha,
            drift_share_threshold=request.drift_share_threshold,
        )

        return DataDriftResponseSchema(**result)

    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid data distribution payload: {str(ve)}",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Data drift evaluation error: {str(exc)}",
        )
