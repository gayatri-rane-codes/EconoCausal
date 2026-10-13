# EconoCausal Causal DAG and Identification Report

## 1. Purpose
Identifies minimal adjustment sets and prevents confounding and collider bias.

## 2. Multi-Arm Treatment and Outcome
- Treatment (T): treatment_arm (0: 0%, 1: 5%, 2: 10%, 3: 20%) and discount_rate.
- Potential Outcomes: Y(t) independent of T given W for each arm t.
- Primary Outcome (Y): purchased (binary).
- Excluded: revenue, discount_amount, net_profit (post-treatment).

## 3. Confounders (W)
1. cart_abandonment_rate
2. previous_discount_response_rate
3. historical_spend
4. recency_days
5. browsing_frequency
6. loyalty_member
7. competitor_price_index
8. seasonality_index

## 4. Mermaid DAG
`mermaid
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
  AGE[age] --> Y
  INC[income_bracket] --> HS
  INC --> Y
  T --> Y
`

## 5. Identification
Backdoor criterion verified via DoWhy. Graph verified strictly acyclic via NetworkX.
