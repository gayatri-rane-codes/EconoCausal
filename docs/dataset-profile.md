# Dataset Profile: EconoCausal Synthetic Promotional Campaign Data

## 1. Provenance & Scope
- **Source**: Synthetic e-commerce promotional campaign dataset generated via NumPy random streams (Seed: 20261012).
- **Scope**: Designed for Causal Machine Learning (DML) and budget-constrained dynamic pricing. Contains no real personal identifiable information.
- **Components**:
  - Main Dataset: 20,000 observations x 45 columns (the single input for model training).
  - Ground Truth: 20,000 observations x 69 columns (reserved exclusively for counterfactual evaluation, true propensities, and oracle benchmarks).

## 2. Verified Dimensions & Data Hygiene
- **Total Rows**: 20,000 unique customer interactions.
- **Missing Values**: 0 across all columns.
- **Duplicate Customer IDs**: 0 duplicates (contiguous range 100001 to 120000).
- **Treatment Arm Counts**:
  - Arm 0 (0% discount): 7,584 (37.92%)
  - Arm 1 (5% discount): 5,395 (26.98%)
  - Arm 2 (10% discount): 4,266 (21.33%)
  - Arm 3 (20% discount): 2,755 (13.78%)
- **Overall Purchase Rate**: 32.72% (Arm 0: 31.63%, Arm 1: 30.95%, Arm 2: 32.35%, Arm 3: 39.71%).

## 3. Variable Roles for Causal Identification
- **Identifier**: customer_id
- **Treatment (T)**: treatment_arm (discrete: 0, 1, 2, 3), discount_pct (0, 5, 10, 20), discount_rate (0.00, 0.05, 0.10, 0.20).
- **Outcome Variables (Y)**:
  - purchased: Binary indicator (0/1) for conversion.
  - revenue: Realized post-discount revenue (INR).
  - net_profit: Bottom-line margin after deducting COGS and marketing costs.
- **Pre-treatment Covariates / Confounders (W) & Effect Modifiers (X)**:
  - Recency, browsing frequency, cart abandonment rate, email open rate, loyalty status, previous discount response rate, historical purchases, competitor price index, customer segment.

## 4. Verified Financial Logic
- **Non-Buyers**: discount_amount = 0.0, revenue = 0.0, cost_of_goods = 0.0, and net_profit = -marketing_cost.
- **Buyers**:
  - discount_amount = round(basket_value_before_discount * discount_rate, 2)
  - revenue = basket_value_before_discount - discount_amount
  - net_profit = revenue - cost_of_goods - marketing_cost
- **Marketing Cost**: channel_cost (email: 4, push: 3, sms: 8, web: 6) + 6.00 coupon issue cost (for discount arms). Correctly incurred by all customers regardless of purchase, preventing discount double-deductions.

## 5. Causal Confounding & Downstream Suitability
- Naive comparison between 5% and 0% discount indicates a negative purchase difference (-0.68%), whereas the true Average Treatment Effect (ATE) is positive (+3.82%).
- Confounding is driven by simulated business policy: higher-risk and less engaged customers were assigned higher discounts.
- Positivity / overlap holds across all segments with minimum propensity > 0.04.
- Suitable for DoWhy DAG formulation (backdoor blocking) and EconML DML heterogenous effect estimation.
