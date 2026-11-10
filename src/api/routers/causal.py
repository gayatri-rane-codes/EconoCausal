from typing import Optional
"""
Causal estimation and policy prediction router for EconoCausal API.
"""
import logging
from typing import List
from fastapi import APIRouter, Form, HTTPException, status

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

from fastapi import File, Form, UploadFile
import pandas as pd
import io

@router.post("/evaluate-batch")
async def evaluate_batch_policy(file: UploadFile = File(...), budget: Optional[float] = Form(None)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only .csv files are supported."
        )

    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))

        # Default customer_id if not present in the CSV
        if "customer_id" not in df.columns:
            df["customer_id"] = [f"CUST_{i+1:05d}" for i in range(len(df))]

        # Pass dictionary records directly to your causal engine
        records = df.to_dict(orient="records")
        pred_df = causal_service.predict_policy(records)

        # Map arm numbers to clear business discount labels
        arm_labels = {
            0: "0% (No Discount)",
            1: "5% Discount",
            2: "10% Discount",
            3: "15% Discount"
        }
        pred_df["discount_tier"] = pred_df["recommended_arm"].map(arm_labels).fillna("0% (No Discount)")

                # Optional budget enforcement
        if budget is not None and budget > 0:
            discount_cost_map = {0: 0.0, 1: 5.0, 2: 10.0, 3: 15.0}
            pred_df["arm_cost"] = pred_df["recommended_arm"].map(discount_cost_map).fillna(0.0)
            
            # Sort by expected incremental profit descending to prioritize highest ROI customers
            pred_df = pred_df.sort_values(by="expected_incremental_profit", ascending=False).reset_index(drop=True)
            pred_df["cum_spend"] = pred_df["arm_cost"].cumsum()
            
            # Customers exceeding the budget cap get downgraded to 0% Control
            exceeded_mask = pred_df["cum_spend"] > budget
            pred_df.loc[exceeded_mask, "recommended_arm"] = 0
            pred_df.loc[exceeded_mask, "expected_incremental_profit"] = 0.0
            
            # Recompute discount tier label after budget cap
            pred_df["discount_tier"] = pred_df["recommended_arm"].map(arm_labels).fillna("0% (No Discount)")

    # Aggregate portfolio-level metrics
        total_customers = len(pred_df)
        dist = pred_df["discount_tier"].value_counts().to_dict()
        zero_discount_count = int((pred_df["recommended_arm"] == 0).sum())

        total_incremental_profit = float(pred_df["expected_incremental_profit"].sum()) if "expected_incremental_profit" in pred_df.columns else 0.0
        budget_saved = round(zero_discount_count * 15.0, 2)

        # Include first 100 rows for display in the UI table
        preview_data = pred_df.head(100).to_dict(orient="records")

        return {
            "status": "success",
            "total_customers": total_customers,
            "distribution": dist,
            "budget_saved_estimate": budget_saved,
            "total_incremental_profit": round(total_incremental_profit, 2),
            "preview": preview_data
        }
    except Exception as e:
        logger.error(f"Batch processing error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process CSV cohort: {str(e)}"
        )


