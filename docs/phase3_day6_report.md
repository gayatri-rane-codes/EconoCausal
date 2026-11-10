# Phase 3 Day 6 Report: Causal Estimator Comparison and Benchmark Reports

## 1. Executive Summary
Phase 3 Day 6 established a reproducible, leakage-safe benchmarking framework comparing two causal estimators:
1. Double Machine Learning / Causal Forest (CausalDMLEstimator)
2. Multi-Arm T-Learner baseline (TLearnerEstimator)

Both estimators were evaluated on out-of-sample CATE estimation accuracy and downstream policy optimization performance across the canonical budget grid ($10,000 to $100,000).

---

## 2. Files Created and Modified
- src/causal/estimators.py: Added TLearnerEstimator with multi-arm support, per-arm gradient boosted regression trees, and treatment leakage protections.
- src/causal/comparison.py: Implemented leakage-safe train/test splitting (70/30 split), metric evaluation (CATE MAE, RMSE, Pearson r, bias), candidate policy generation, and multi-budget frontier evaluation.
- src/causal/__init__.py: Exported comparison routines and TLearnerEstimator.
- scripts/benchmark_estimators.py: End-to-end benchmark execution script generating JSON artifacts and Markdown summaries.
- tests/test_estimator_comparison.py: 7 dedicated unit tests verifying split integrity, leakage rejection, missing arm exceptions, and metric calculations.
- reports/estimator_metrics_report.json: Machine-readable CATE estimation metrics.
- reports/budget_frontier_comparison_report.json: Machine-readable frontier evaluation comparing predicted vs. realized metrics.
- reports/estimator_comparison_summary.md: Human-readable Markdown summary tables.

---

## 3. Estimator Implementations & Leakage-Safe Strategy
- Backdoor Adjustment Set: Controlled for 8 covariates: browsing_frequency, cart_abandonment_rate, competitor_price_index, historical_spend, loyalty_member, previous_discount_response_rate, recency_days, and seasonality_index.
- Leakage Prevention:
  - Evaluation ground truth (ground_truth_effects.csv) is quarantined and used strictly post-hoc.
  - Features and training targets exclude true individual treatment effects.
  - Train/test splitting uses a fixed random seed (seed=42) with strict customer ID separation (6,000 held-out evaluation customers).

---

## 4. Measured Results

### CATE Accuracy (Held-Out Test Set: 6,000 Customers)
- Double ML Overall MAE: 0.0886 | RMSE: 0.1299
- T-Learner Overall MAE: 0.0903 | RMSE: 0.1189
- Treatment Effect Pearson Correlation (r):
  - Arm 1 (5% discount): DML r = 0.182 | T-Learner r = 0.289
  - Arm 2 (10% discount): DML r = 0.249 | T-Learner r = 0.564
  - Arm 3 (20% discount): DML r = 0.357 | T-Learner r = 0.748

### Policy Optimization Frontier Across Budgets
- $10,000 Budget: DML Realized Profit = -$2,075.04 | TL Realized Profit = $2,200.75 | Delta (DML - TL) = -$4,275.79
- $25,000 Budget: DML Realized Profit = -$3,302.14 | TL Realized Profit = $3,264.84 | Delta (DML - TL) = -$6,566.98
- $50,000 Budget: DML Realized Profit = -$750.23 | TL Realized Profit = $8,740.93 | Delta (DML - TL) = -$9,491.16
- $75,000 Budget: DML Realized Profit = $5,455.53 | TL Realized Profit = $11,718.65 | Delta (DML - TL) = -$6,263.12
- $100,000 Budget: DML Realized Profit = $6,263.22 | TL Realized Profit = $13,533.19 | Delta (DML - TL) = -$7,269.97

---

## 5. Test Suite Verification
- unittest discover: Ran 65 tests in 2.277s - OK
- pytest tests/test_estimator_comparison.py: 7 passed in 3.64s - 100%
- pytest full suite: 72 passed, 8 warnings in 5.38s - 100%

---

## 6. Known Limitations
- The synthetic DGP induces higher ranking correlation in T-Learner than Causal Forest DML, translating into higher realized policy profits for T-Learner.
- Optimizers curse is evident in both models: predicted incremental profit is consistently higher than realized incremental profit due to selection on positive estimation noise.
