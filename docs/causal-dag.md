# Causal Directed Acyclic Graph (DAG) & Identification Specification

## 1. Causal Research Question
**Primary Question**: What is the causal effect of varying discount pricing levels (`treatment_arm` $\in \{0, 1, 2, 3\}$, corresponding to discount rates $0\%$, $5\%$, $10\%$, and $20\%$) on customer conversion (`purchased` $\in \{0, 1\}$)?

**Secondary Objective**: Estimate conditional average treatment effects across customer segments without introducing confounding bias, collider bias, or data leakage.

---

## 2. Core Variables

### Treatment ($T$)
* `treatment_arm`: Discrete discount tier $\{0: 0\%, 1: 5\%, 2: 10\%, 3: 20\%\}$.
* Continuous alternative: `discount_rate` $\in \{0.00, 0.05, 0.10, 0.20\}$.

### Outcome ($Y$)
* `purchased`: Binary indicator ($1$ if transaction completed, $0$ otherwise).

### Backdoor Confounders ($W$)
Pre-treatment variables that influence both treatment assignment probabilities and baseline conversion:
1. `cart_abandonment_rate`: Historical proportion of shopping sessions abandoned prior to checkout.
2. `previous_discount_response_rate`: Customer responsiveness to historical discount offers.
3. `historical_spend`: Total cumulative customer expenditure (USD).
4. `recency_days`: Days since customer last visited or made a transaction.
5. `browsing_frequency`: Frequency of site visits per tracking period.
6. `loyalty_member`: Binary indicator ($1$ for rewards member, $0$ otherwise).
7. `competitor_price_index`: External market price index relative to catalog benchmark.
8. `seasonality_index`: Cyclical demand multiplier based on purchase timing.

### Excluded / Prohibited Variables (Collider & Leakage Prevention)
* **Post-Treatment Metrics**: `revenue`, `discount_amount`, `cost_of_goods`, `net_profit`, `basket_value_before_discount`. These variables are generated downstream of both treatment and purchase. Conditioning on them would block causal mechanisms or open spurious non-causal paths (collider bias).
* **Ground-Truth Identifiers**: `customer_id`, `assigned_arm`, `targeting_score`, `true_baseline_logit`, `true_effect_beta_per10pp`, `true_effect_kappa_per10pp2`, and counterfactual outcomes (`purchased_arm0...3`). These columns are strictly reserved for downstream estimator benchmarking.

---

## 3. Directed Acyclic Graph Structure

```mermaid
graph TD
    CAR[cart_abandonment_rate] --> T[treatment_arm]
    CAR --> Y[purchased]

    PDR[previous_discount_response_rate] --> T
    PDR --> Y

    HS[historical_spend] --> T
    HS --> Y

    RD[recency_days] --> T
    RD --> Y

    BF[browsing_frequency] --> T
    BF --> Y

    LM[loyalty_member] --> T
    LM --> Y

    CPI[competitor_price_index] --> T
    CPI --> Y

    SI[seasonality_index] --> T
    SI --> Y

    T --> Y
