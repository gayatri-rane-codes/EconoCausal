# Dataset Validation Report — EconoCausal

Computed across `econocausal_synthetic_dataset.csv` and `ground_truth_evaluation.csv`.

**Status: 68/68 Verification Checks Passed**

## Summary of Results
- **Main Shape**: 20,000 rows × 45 columns.
- **Ground Truth Shape**: 20,000 rows × 69 columns.
- **Data Completeness**: 0 missing values, 0 duplicate IDs.
- **Treatment Arm Breakdown**:
  - Arm 0 (0%): 7,584 (37.92%)
  - Arm 1 (5%): 5,395 (26.98%)
  - Arm 2 (10%): 4,266 (21.33%)
  - Arm 3 (20%): 2,755 (13.78%)
- **Purchase Conversion Rate**: 32.72% overall.

## Financial Consistency Audit
- Non-buyers: Verified `discount_amount = 0.0`, `revenue = 0.0`, `cost_of_goods = 0.0`, and `net_profit = -marketing_cost`.
- Buyers: Verified `revenue = basket - discount_amount` and `net_profit = revenue - cost_of_goods - marketing_cost`.
- Marketing Cost: Verified channel delivery cost + coupon issue cost (6.00 INR for arms > 0).

## Confounding & Effect Recovery Audit
- **Arm 5% vs 0%**: Naive diff = -0.0068 | True ATE = +0.0382
- **Arm 10% vs 0%**: Naive diff = +0.0072 | True ATE = +0.0788
- **Arm 20% vs 0%**: Naive diff = +0.0808 | True ATE = +0.1534
- Standardized Mean Differences (SMD) before adjustment show substantial imbalance (max raw SMD = 0.525), successfully resolved after propensity weighting (max SMD < 0.06).
