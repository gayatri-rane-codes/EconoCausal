# Phase 2, Day 5: Treatment Overlap and CATE Reliability Diagnostics Report

## 1. Executive Summary & Diagnostics Objective
On Day 5 of Phase 2, we investigated whether the weak individual-level CATE rank accuracy (MAE ~15.56 pp for Arm 3) and extreme negative predictions (minimum -0.6016) were driven by treatment arm imbalance, propensity overlap violations, or sparse feature support.

## 2. Treatment Arm Distribution & Representation
* **Arm 0 (0% Control)**: 7,584 customers (37.92%)
* **Arm 1 (5% Discount)**: 5,395 customers (26.98%)
* **Arm 2 (10% Discount)**: 4,266 customers (21.33%)
* **Arm 3 (20% Discount)**: 2,755 customers (13.78%)
All 4 treatment categories are well-represented across the 20,000 customers.

## 3. Generalized Propensity Score Overlap Audit
Using 5-fold cross-fitting with `Pipeline(StandardScaler(), LogisticRegression())` across the 8 backdoor confounders:
* **Global Minimum Propensity**: 0.0299 across all units.
* **Propensity < 0.01**: Exactly 0 customers (0.00%).
* **Propensity < 0.05**: Exactly 161 customers (0.80%).
* **Empirical Finding**: Treatment overlap (the Positivity assumption) is well-satisfied across the vast majority of the population. There is no systemic failure of common support.

## 4. Root Cause of Extreme CATE Predictions
* **Extreme Negative Predictions**: Exactly 18 customers had Arm 3 CATE estimates < -0.20 (reaching down to -0.6016).
* **Propensity Overlap Check**: For these 18 customers, their propensity for Arm 3 ranges from 0.0525 to 0.2163 (mean 0.1126), and Arm 0 ranges from 0.1944 to 0.5606. **None of them violate common support ($e_a(W) > 0.05$)**.
* **Feature Support Diagnosis**: All 18 customers are extreme right-tail spenders with `historical_spend` averaging $77,363 (min $60,710, max $143,464), compared to the general population mean of $6,512.
* **Conclusion**: The extreme negative predictions are caused by **support sparsity in effect-modifier space ($X$)** combined with linear extrapolation in `LinearDML` ($\tau(X) = \theta^\top X$), rather than propensity overlap collapse.

## 5. Confirmed Findings vs. Hypotheses
* **Confirmed Finding 1 (Evidence)**: Positivity is empirically satisfied for 99.2% of the dataset; overlap collapse is NOT the cause of low individual CATE ranking correlations.
* **Confirmed Finding 2 (Evidence)**: Extreme negative predictions are strictly confined to the upper 0.1% extreme spend tail ($> $60,000).
* **Confirmed Finding 3 (Evidence)**: Irreducible binary residualization variance is the primary driver of high individual MAE (corroborated by CausalForestDML exhibiting similar MAE).
* **Hypothesis**: Winsorizing/log-transforming spend or clustering customers into discrete macro segments before policy assignment will stabilize pricing recommendations without unconstrained tail extrapolation.

## 6. Recommended Safeguards for Policy Development
1. **Segment-Level Policy Assignment**: Base discount rules on customer segment groups (e.g. loyalty status x spend tercile) rather than individual continuous point estimates.
2. **Support Restrictions**: Apply fallback baseline rules (e.g., standard 0% or low discount) for customers in extreme covariate tails ($> 99.5$th percentile of spend).
3. **Margin-Aware Policy Rules**: Evaluate policies using business value (incremental revenue minus discount cost) rather than conversion probability in isolation.

## 7. Commands to Reproduce
```powershell
.\.venv\Scripts\Activate.ps1
python -m unittest discover -s tests -p "test_*.py"
