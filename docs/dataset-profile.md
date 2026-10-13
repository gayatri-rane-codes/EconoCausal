# Dataset Profile: EconoCausal Synthetic Promotional Campaign Data

## 1. Provenance & Scope
- **Source**: Synthetic e-commerce promotional campaign dataset generated via Python/NumPy simulation (Seed: 20261012).
- **Scope**: Designed for Causal Machine Learning (DML), Propensity Score Matching (PSM), and budget-constrained dynamic pricing.
- **Data Files**:
  - data/raw/econocausal_synthetic_dataset.csv (20,000 rows x 45 columns): The single input file for model training. Excluded from Git tracking via .gitignore.
  - data/raw/ground_truth_evaluation.csv (20,000 rows x 69 columns): Evaluation and oracle benchmarking file. Excluded from Git tracking via .gitignore.
- **Scripts**:
  - src/data/generate_dataset.py: Reproducible generator script.
  - src/data/validate_dataset.py: Automated local validation script.

## 2. Verified Dimensions & Data Hygiene (Local Execution)
- **Observations**: Exactly 20,000 rows x 45 columns (main) and 20,000 rows x 69 columns (ground truth).
- **Missing Values**: 0 nulls across all fields.
- **Duplicate IDs**: 0 duplicate customer IDs (contiguous range 100001 to 120000).
- **ID Alignment**: Exactly identical customer ID ordering across both CSVs.

## 3. Treatment Specification & Decision
- **Discount Structure**: Four discrete treatment levels:
  - Arm 0: 0% discount (7,584 customers, 37.92%)
  - Arm 1: 5% discount (5,395 customers, 26.98%)
  - Arm 2: 10% discount (4,266 customers, 21.33%)
  - Arm 3: 20% discount (2,755 customers, 13.78%)
- **Scope Decision (Percentage vs Fixed Dollar)**: The working dataset adopts percentage discounts (0%, 5%, 10%, 20%) rather than flat dollar discounts (, , ) to realistically scale with basket list prices (100–15,000 INR) and prevent negative purchase price artifacts.

## 4. Verified Financial Consistency
- **Non-Buyers**: discount_amount = 0.0, evenue = 0.0, cost_of_goods = 0.0, and 
et_profit = -marketing_cost (exact match across all 13,457 non-buyers).
- **Buyers**:
  - discount_amount = round(basket_value_before_discount * discount_rate, 2)
  - evenue = basket_value_before_discount - discount_amount
  - 
et_profit = revenue - cost_of_goods - marketing_cost
- **Marketing Outlay**: Channel delivery cost (email: 4, push: 3, sms: 8, web: 6) plus 6.00 coupon creation cost for discount arms. Accurately incurred by all targeted users, avoiding coupon double-counting.

## 5. Local Confounding & Treatment Effect Verification
Comparison of naive observational differences against true ground-truth ATE:
- **5% vs 0%**: Naive diff = -0.0068 (-0.68%) vs True ATE = +0.0382 (+3.82%)
- **10% vs 0%**: Naive diff = +0.0072 (+0.72%) vs True ATE = +0.0788 (+7.88%)
- **20% vs 0%**: Naive diff = +0.0808 (+8.08%) vs True ATE = +0.1534 (+15.34%)

The severe negative bias in naive estimates confirms substantial confounding in observational assignment, satisfying the requirements for causal DAG formulation (DoWhy) and Double Machine Learning (EconML).
