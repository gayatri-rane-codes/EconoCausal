# EconoCausal Causal Directed Acyclic Graph (DAG) & Identification Report

## 1. Purpose of the Causal DAG
In dynamic promotional pricing, estimating the effect of discounts by naive comparison introduces confounding bias: customers with high cart abandonment rates or discount-seeking tendencies are systematically offered higher promotions. 
The Directed Acyclic Graph (DAG) formalizes our structural domain assumptions about cause-and-effect relationships, allowing us to:
1. Identify the minimal adjustment set that blocks backdoor paths (confounding).
2. Avoid collider bias and mediator blocking by excluding post-treatment variables.
3. Formally verify mathematical identifiability before applying estimators like Double Machine Learning (DML).

## 2. Treatment & Primary Outcome
- **Treatment ($T$)**: `treatment_arm` (Discrete: 0 = 0%, 1 = 5%, 2 = 10%, 3 = 20%) / `discount_rate` (Continuous: 0.0, 0.05, 0.10, 0.20).
- **Primary Outcome ($Y$)**: `purchased` (Binary: $Y \in \{0, 1\}$).
- *Downstream Business Metrics*: `revenue`, `discount_amount`, and `net_profit` are deterministic functions realized conditional on $Y = 1$.

## 3. Selected Confounders & Adjustment Set ($W$)
The following 8 pre-treatment variables were identified as common causes affecting both treatment assignment and conversion:
1. **`cart_abandonment_rate`**: Algorithmic rules trigger higher discounts for cart abandoners; abandoners also possess lower baseline purchase propensity.
2. **`previous_discount_response_rate`**: Users responsive to prior sales receive more promotional targeting and have higher baseline deal affinity.
3. **`historical_spend`**: High-value spenders receive targeted loyalty discounts and have distinct conversion dynamics.
4. **`recency_days`**: Inactive/slipping customers are targeted for win-back campaigns, but exhibit lower baseline conversion.
5. **`browsing_frequency`**: High-frequency users trigger intent-based promotional logic and show elevated spontaneous purchases.
6. **`loyalty_member`**: Loyalty tier membership grants access to exclusive pricing brackets and increases brand trust.
7. **`competitor_price_index`**: Lower competitor prices trigger dynamic promotional adjustments and negatively impact sales if unaddressed.
8. **`seasonality_index`**: Peak promotional seasons dictate campaign intensity and elevate market-wide conversion rates.

## 4. Excluded Variables & Justifications
- **Identifiers**: `customer_id` (pure surrogate key, no causal mechanism).
- **Alternative Treatment Encodings**: `discount_pct`, `discount_rate` (redundant representations of $T$; adjusting for them alongside $T$ causes extreme collinearity).
- **Post-Treatment Variables & Colliders**: `discount_amount`, `revenue`, `cost_of_goods`, and `net_profit` are realized after the purchase event. Conditioning on them would create post-treatment conditioning bias and block the causal mechanism.
- **Pure Predictors of Outcome**: `age`, `income_bracket` primarily drive baseline affordability/intent; omitting them from the backdoor set does not introduce bias, but they can be included in Phase 2 ML nuisance models to reduce outcome variance.

## 5. Visual Graph Representation (Mermaid)
Graph edges defined:
- cart_abandonment_rate -> treatment_arm, purchased
- previous_discount_response_rate -> treatment_arm, purchased
- historical_spend -> treatment_arm, purchased
- recency_days -> treatment_arm, purchased
- browsing_frequency -> treatment_arm, purchased
- loyalty_member -> treatment_arm, purchased
- competitor_price_index -> treatment_arm, purchased
- seasonality_index -> treatment_arm, purchased
- age -> purchased
- income_bracket -> historical_spend, purchased
- treatment_arm -> purchased

## 6. Formal Identification Results (DoWhy)
- **Identification Criterion**: Backdoor Criterion.
- **Identified Estimand**:
  $$\frac{\partial}{\partial T} \mathbb{E}[Y \mid W]$$
  where $W = \{\text{browsing\_frequency}, \text{cart\_abandonment\_rate}, \text{competitor\_price\_index}, \text{historical\_spend}, \text{loyalty\_member}, \text{previous\_discount\_response\_rate}, \text{recency\_days}, \text{seasonality\_index}\}$.
- **Identification Status**: Formally Identifiable.

## 7. What the DAG Proves vs. Assumes
- **What it Proves**: If the assumed graph accurately represents reality, controlling for $W$ eliminates all backdoor confounding, enabling unbiased causal estimation of discount effects.
- **What it Assumes**:
  1. **Conditional Unconfoundedness**: No unobserved variable $U$ influences both discount assignment and purchase ($Y \perp T \mid W$).
  2. **Positivity / Overlap**: Every customer segment has a non-zero probability of receiving each discount arm ($0 < P(T = t \mid W) < 1$).
  3. **No Interference (SUTVA)**: One customer's discount assignment does not alter another customer's decision.
- **Limitations**: DoWhy's identification check validates graph mathematics, not empirical ground truth. Causal validity rests entirely on the substantive accuracy of our acyclic assumptions.
