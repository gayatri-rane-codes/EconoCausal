# EconoCausal Synthetic Dataset Profile & Validation Audit

## 1. Provenance & Dataset Source
- **Origin**: Fully synthetic benchmark generated specifically for causal dynamic pricing and heterogeneous treatment effect estimation.
- **Generator Script**: `src/data/generate_econocausal_dataset.py`
- **Validation Script**: `src/data/validate_econocausal_dataset.py`
- **Execution Date**: Locally executed in Python 3.11.9 (`.venv`)
- **Validation Status**: **68 / 68 checks passed (100%)**

## 2. Dataset Dimensions & Architecture
- **Main Dataset**: `data/raw/econocausal_synthetic_dataset.csv`
  - Shape: 20,000 rows × 45 columns
  - Missing Values: 0 nulls across all cells
  - Customer IDs: Unique, contiguous integers from 100,001 to 120,000
- **Evaluation Ground Truth**: `data/raw/ground_truth_evaluation.csv`
  - Shape: 20,000 rows × 69 columns
  - Stored strictly isolated from observed operational features to prevent data leakage.

## 3. Treatment and Outcome Definitions
- **Treatment Arms ($T$)**:
  - Arm 0: 0% discount (Control) — 7,584 observations (37.9%)
  - Arm 1: 5% discount — 5,395 observations (27.0%)
  - Arm 2: 10% discount — 4,266 observations (21.3%)
  - Arm 3: 20% discount — 2,755 observations (13.8%)
- **Primary Observed Outcomes ($Y$)**:
  - `purchased`: Binary indicator ($Y \in \{0, 1\}$), baseline purchase rate 32.71%
  - `revenue`: Monetary amount realized ($0 \le \text{revenue} \le \text{basket\_value}$)
  - `discount_amount`: Monetary concession ($0$ for non-buyers, $\text{basket} \times \text{rate}$ for buyers)
  - `net_profit`: $\text{revenue} - \text{cost\_of\_goods} - \text{marketing\_cost}$

## 4. Confounders & Backdoor Paths
- **Backdoor Confounders ($W$)**: Customer engagement, recency, cart abandonment rate, loyalty status, and past discount responsiveness.
- Naive gap for 5% vs 0% discount: $-0.68\%$ (biased downward due to targeting of discount-sensitive or higher-abandonment users).
- True ATE for 5% vs 0%: $+3.82\%$.
- Maximum raw Standardized Mean Difference (SMD): $0.525$ (strong confounding).
- Maximum IPW SMD using true propensities: $0.0544$ (complete covariate balance achieved, $< 0.1$).

## 5. Overlap & Positivity Verification
- Positivity condition: For all units $i$ and arms $k$, $P(T_i = k \mid X_i) > 0$.
- Minimum true propensity observed: $0.0424$ (safely exceeds the $0.02$ theoretical floor).
- Fraction of customers with any propensity below $0.05$: $1.36\%$ (well below the $5\%$ limit).

## 6. Financial Integrity Rules
- Zero-out identity: Non-buyers strictly yield $\text{discount\_amount} = \text{revenue} = \text{cost\_of\_goods} = 0$.
- Non-buyer net profit: Strictly equals $-\text{marketing\_cost}$.
- List-price conservation: $E[\text{revenue}_k] + E[\text{discount}_k] = p_k \times \text{basket\_value}$.

## 7. Downstream Compatibility Smoke Tests
All verified locally with seeded scikit-learn and SciPy models:
- **Doubly Robust / DML-style ATE**: Maximum error against true ATE is $0.0084$ across all arms (tolerance $< 0.02$).
- **Propensity Score Matching (PSM)**: Max post-match SMD $< 0.08$.
- **Uplift / Qini Curve**: Monotonically positive cumulative incremental gain over random baseline.
- **SciPy LP Budget Optimization**: Feasible budget allocations successfully obtained under marketing expenditure constraints.
