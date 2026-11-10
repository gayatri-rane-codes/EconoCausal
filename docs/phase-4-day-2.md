# Phase 4, Day 2: Causal API Router Documentation

## 1. Overview and Endpoint Specifications
The Phase 4 Day 2 deliverable introduces dedicated FastAPI routes for Double Machine Learning (LinearDML) causal inference and prescriptive discount pricing under `src/api/routers/causal.py`.

### Endpoints
* **`POST /causal/estimate`**: Computes Conditional Average Treatment Effects (CATE) across all active treatment arms relative to the control group (Arm 0 = 0% discount) for a batch of customer profiles.
* **`POST /causal/predict`**: Computes CATE uplift estimates and returns the economically optimal discount recommendation (`recommended_arm` in {0, 1, 2, 3}) and estimated incremental profit using threshold-based policy prescription.

---

## 2. Request and Response Examples

### POST /causal/estimate
**Request Body (`CausalBatchRequest`)|**
ŽÊ'
{
  "customers": [
    {
      "customer_id": "cust_101",
      "browsing_frequency": 6.4,
      "cart_abandonment_rate": 0.28,
      "competitor_price_index": 1.05,
      "historical_spend": 520.0,
      "loyalty_member": 1,
      "previous_discount_response_rate": 0.45,
      "recency_days": 8.0,
      "seasonality_index": 1.20
    }
  ]
}


**Response Body (`CATEEstimateResponse`):**
``jon
{
  "results": [
    {
      "customer_id": "cust_101",
      "cate_arm_1_vs_0": 0.0421,
      "cate_arm_2_vs_0": 0.0894,
      "cate_arm_3_vs_0": 0.1243
    }
  ],
  "count": 1
}
```

### POST /causal/predict
**Request Body (`CausalBatchRequest`)|**
ŽÊ'
{
  "customers": [
    {
      "customer_id": "cust_101",
      "browsing_frequency": 6.4,
      "cart_abandonment_rate": 0.28,
      "competitor_price_index": 1.05,
      "historical_spend": 520.0,
      "loyalty_member": 1,
      "previous_discount_response_rate": 0.45,
      "recency_days": 8.0,
      "seasonality_index": 1.20,
      "basket_value": 150.0,
      "baseline_purchase_prob": 0.35
    }
  ]
}
```

**Response Body (`PolicyPredictResponse`)|**
ŽÊ'
{
  "recommendations": [
    {
      "customer_id": "cust_101",
      "cate_arm_1_vs_0": 0.0421,
      "cate_arm_2_vs_0": 0.0894,
      "cate_arm_3_vs_0": 0.1243,
      "recommended_arm": 2,
      "expected_incremental_profit": 14.85
    }
  ],
  "count": 1
}
```

---

## 3. Reuse of Existing Causal Modules
* **`src.causal.estimators.CausalDMLEstimator`*: Used directly for DML-based CATE estimation using `DEFAULT_CONFOUNDERS`.
* **
src.causal.policy.prescribe_policy`**: Reused directly to map estimated treatment effects into optimal treatment allocations and expected profit gains.
* **No Logic Duplication**: Router functions parse validated Pydantic records into Pandas DataFrames and hand off computation directly to `CausalService`.

---

## 4. Model Loading and Initialization Strategy
* Managed via singleton `CausalService` in `src/api/causal_service.py`.
* Pre-loads/ptrains the estimator on observational training data (`data/raw/synthetic_retail_pricing_dataset.csv`).
* **Data Isolation**: Strictly isolates observational data from ground-truth evaluation tables (`ground_truth_evaluation.csv` is never referenced or accessed by API routines).

---

## 5. Validation and Error Handling
* **HTTP 422 Unprocessable Entity**: Triggered via Pydantic when required covariates are missing, out of range, or unauthorized extra fields are provided (`extra='forbid'`).
* **HTTP 503 Service Unavailable**: Returned if the singleton `causal_service` is uninitialized or fails its dataset load.
* **HTTP 500 Internal Server Error**: Catch-all boundary that logs internal stack traces while returning sanitized failure notices to API consumers.

---

## 6. Test Results Summary
* `tests/test_causal_router.py`: 6/6 tests passing (estimate endpoint, predict endpoint, missing features, out-of-range bounds, 503 service unavailable, and ground-truth data isolation).
* Full regression test suite: 75/75 passing across all existing unit, integration, and policy test suites.

---

## 7. Known Limitations
* Inference latency on cold-start fits can be noticeable if the pre-trained estimator is not cached to disk via serialization.
* Batch requests are synchronous; very large arrays should eventually be handled via background task queues.
