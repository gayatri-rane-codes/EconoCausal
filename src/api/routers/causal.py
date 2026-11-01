"""
Causal estimation and policy prediction router for EconoCausal API.
"""
import logging
from typing import List
from fastapi import APIRouter, HTTPException, status

from src.api.schemas import (
    CausalBatchRequest,
    CATEEstimateResponse,
    CATEEffect,
    PolicyPredictResponse,
    PolicyRecommendation,
    ErrorResponse,
)
from src.api.causal_service import causal_service

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/causal",
    tags=["Causal Inference & Policy"],
)


@router.post(
    "/estimate",
    response_model=CATEEstimateResponse,
    status_code=status.HTTP_200_OK,
    responses={
        422: {"model": ErrorResponse, "description": "Validation Error"},
        503: {"model": ErrorResponse, "description": "Estimator Not Ready"},
        500: {"model": ErrorResponse, "description": "Internal Inference Error"},
    },
    summary="Estimate Conditional Average Treatment Effects (CATE)",
    description="Computes treatment uplift (tau_a vs control) for discount arms 1 (5%), 2 (10%), and 3 (20%).",
)
def estimate_cate(payload: CausalBatchRequest) -> CATEEstimateResponse:
    if not causal_service.is_ready:
        try:
            causal_service.initialize()
        except Exception as e:
            logger.error(f"Failed to initialize causal service: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Causal estimator service is currently uninitialized or unavailable.",
            )

    try:
        records = [c.model_dump() for c in payload.customers]
        effects_df = causal_service.estimate_cate(records)

        results: List[CATEEffect] = []
        for i, c in enumerate(payload.customers):
            results.append(
                CATEEffect(
                    customer_id=c.customer_id,
                    cate_arm_1_vs_0=float(effects_df.iloc[i]["cate_arm_1_vs_0"]),
                    cate_arm_2_vs_0=float(effects_df.iloc[i]["cate_arm_2_vs_0"]),
                    cate_arm_3_vs_0=float(effects_df.iloc[i]["cate_arm_3_vs_0"]),
                )
            )
        return CATEEstimateResponse(results=results, count=len(results))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Inference failure in /causal/estimate: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to compute causal estimates due to an internal processing error.",
        )


@router.post(
    "/predict",
    response_model=PolicyPredictResponse,
    status_code=status.HTTP_200_OK,
    responses={
        422: {"model": ErrorResponse, "description": "Validation Error"},
        503: {"model": ErrorResponse, "description": "Estimator Not Ready"},
        500: {"model": ErrorResponse, "description": "Internal Inference Error"},
    },
    summary="Predict Optimal Treatment Arm and Expected Incremental Profit",
    description="Computes CATE uplift and prescribes optimal discount assignment maximizing expected incremental profit.",
)
def predict_policy(payload: CausalBatchRequest) -> PolicyPredictResponse:
    if not causal_service.is_ready:
        try:
            causal_service.initialize()
        except Exception as e:
            logger.error(f"Failed to initialize causal service: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Causal estimator service is currently uninitialized or unavailable.",
            )

    try:
        records = [c.model_dump() for c in payload.customers]
        pred_df = causal_service.predict_policy(records)

        recommendations: List[PolicyRecommendation] = []
        for i, c in enumerate(payload.customers):
            recommendations.append(
                PolicyRecommendation(
                    customer_id=c.customer_id,
                    cate_arm_1_vs_0=float(pred_df.iloc[i]["cate_arm_1_vs_0"]),
                    cate_arm_2_vs_0=float(pred_df.iloc[i]["cate_arm_2_vs_0"]),
                    cate_arm_3_vs_0=float(pred_df.iloc[i]["cate_arm_3_vs_0"]),
                    recommended_arm=int(pred_df.iloc[i]["recommended_arm"]),
                    expected_incremental_profit=float(pred_df.iloc[i]["expected_incremental_profit"]),
                )
            )
        return PolicyPredictResponse(recommendations=recommendations, count=len(recommendations))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Inference failure in /causal/predict: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to compute policy predictions due to an internal processing error.",
        )
