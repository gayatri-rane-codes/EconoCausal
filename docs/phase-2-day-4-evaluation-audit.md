# Phase 2, Day 4: Ground-Truth Audit and Model Reliability Report

## 1. Executive Summary
On Day 4 of Phase 2, we performed an empirical audit of the Double Machine Learning (DML) estimator against the quarantined evaluation dataset (`data/raw/ground_truth_evaluation.csv`).
We verified customer alignment, reproduced CATE metrics, assessed correlation with true individual effects, and executed a controlled benchmark comparison against `CausalForestDML`.

## 2. Customer ID Alignment & Ground-Truth Schema
* **Training Dataset**: `data/raw/econocausal_synthetic_dataset.csv` (20,000 rows, 45 features).
* **Evaluation Dataset**: `data/raw/ground_truth_evaluation.csv` (20,000 rows, 69 features).
* **Alignment Verification**:
  * `customer_id` is unique in both files.
  * Direct equality verification `(df["customer_id"] == gt["customer_id"]).all()` evaluates to `True`.
  * The ground truth columns `true_ite_purchase_arm1`, `true_ite_purchase_arm2`, and `true_ite_purchase_arm3` represent true individual risk differences on purchase conversion relative to Arm 0 (0% discount baseline).

## 3. Empirical Accuracy: LinearDML Audit
| Comparison | True ATE | Est ATE | Bias | MAE (pp) | RMSE | Pearson r | Spearman rho |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Arm 1 (5% vs 0%)** | +0.0382 | +0.0281 | -0.0100 | 0.0348 (3.48 pp) | 0.0480 | 0.1825 | 0.2123 |
| **Arm 2 (10% vs 0%)** | +0.0788 | +0.0598 | -0.0190 | 0.0742 (7.42 pp) | 0.1000 | 0.2568 | 0.2629 |
| **Arm 3 (20% vs 0%)** | +0.1534 | +0.1388 | -0.0146 | 0.1556 (15.56 pp) | 0.1947 | 0.3401 | 0.3633 |

### Average-Effect vs. Individual-Effect Accuracy
* **Average Treatment Effect (ATE)**: The model displays low macro bias (underestimating cohort conversion lift by only 1.0 to 1.9 percentage points).
* **Individual Treatment Effect (ITE)**: Individual accuracy is moderate to weak. An MAE of 0.1556 in Arm 3 indicates an average error of 15.56 percentage points in predicting an individual customer's conversion lift. Pearson correlations range between 0.18 and 0.34.
* **Takeaway**: The model is reliable for cohort-level directional decisions and average policy simulation, but should not be treated as a high-precision deterministic predictor for individual customers.

## 4. Extreme Prediction Diagnostic
* In Arm 3, predictions ranged from -0.6016 to +0.2658.
* Exactly 18 customers out of 20,000 (0.09%) produced predictions < -0.20.
* All 18 customers are extreme right-tail spenders (mean historical spend of $76,876 vs general population average of $6,575) and 100% loyalty club members.
* `LinearDML` enforces $\tau(X) = \theta^\top X$. For covariate vectors located far from the support center, linear extrapolation yields unconstrained negative tails.

## 5. Controlled Estimator Benchmark: CausalForestDML
We evaluated `CausalForestDML` on the identical feature specification under 3-fold cross-fitting:
* **Arm 1 (5% vs 0%)**: Est Mean = 0.0265, Bias = -0.0116, MAE = 0.0382, Pearson r = 0.1648, Min = -0.1292, Max = 0.1738
* **Arm 2 (10% vs 0%)**: Est Mean = 0.0601, Bias = -0.0187, MAE = 0.0746, Pearson r = 0.2532, Min = -0.1219, Max = 0.2149
* **Arm 3 (20% vs 0%)**: Est Mean = 0.1390, Bias = -0.0144, MAE = 0.1559, Pearson r = 0.3302, Min = -0.1953, Max = 0.3613
* **Comparison Finding**: `CausalForestDML` constrains the extreme prediction tail (Arm 3 minimum is -0.1953 vs -0.6016 in LinearDML). However, individual rank correlations (r = 0.33) and MAE (0.1559) remain essentially identical, demonstrating that the remaining variance is intrinsic to binary-outcome residualization rather than estimator form.

## 6. Recommendations & Governance
1. Retain `LinearDML` as our baseline primary pipeline due to computational efficiency and transparent linear coefficients, while noting its tail sensitivity.
2. Maintain strict separation of evaluation ground truth from training workflows.
