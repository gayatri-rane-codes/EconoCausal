# Causal Graph Validation & Identification Report (Phase 1, Day 5)

## 1. Executive Summary
This report formalizes and validates the Directed Acyclic Graph (DAG) for the EconoCausal pricing engine. The primary goal is unbiased identification of the causal effect of discount interventions (`treatment_arm` / `discount_rate`) on customer purchase propensity (`purchased`) without introducing collider bias, mediation conditioning, or data leakage.

---

## 2. Structural Causal Model & Variables

* **Treatment Variable ($T$)**: `treatment_arm` $\in \{0, 1, 2, 3\}$ (`discount_rate` $\in [0.00, 0.05, 0.10, 0.20]$).
* **Primary Outcome ($Y$)**: `purchased` $\in \{0, 1\}$.
* **Validated Backdoor Confounders ($W$)**:
  1. `browsing_frequency`
  2. `cart_abandonment_rate`
  3. `competitor_price_index`
  4. `historical_spend`
  5. `loyalty_member`
  6. `previous_discount_response_rate`
  7. `recency_days`
  8. `seasonality_index`

### Prohibited / Excluded Variables (Collider & Leakage Prevention)
* **Post-treatment Business Metrics**: `revenue`, `discount_amount`, `cost_of_goods`, `marketing_cost`, `net_profit`, and `basket_value_before_discount` are generated downstream of treatment and purchase events. Conditioning on these induces collider and mediation bias.
* **Ground-Truth Evaluation Columns**: `customer_id`, `assigned_arm`, `targeting_score`, `true_baseline_logit`, `true_effect_beta_per10pp`, `true_effect_kappa_per10pp2`, and counterfactual indicators (`purchased_arm{0..3}`) are strictly reserved for downstream benchmark evaluation.

---

## 3. Causal Identification via DoWhy

* **Graph Topography**: 12 nodes, 20 directed edges.
* **Acyclicity Check**: Verified acyclic using NetworkX (`nx.is_directed_acyclic_graph == True`).
* **Identification Strategy**: Backdoor Criterion.
* **Estimand Expression**:
  $$\frac{d}{d[\text{treatment\_arm}]} \mathbb{E}[\text{purchased} \mid W]$$
  where $W$ consists solely of the 8 pre-treatment confounders listed above.
* **Frontdoor / Instrumental Identification**: Not required; unconfoundedness holds conditionally given $W$.

---

## 4. Empirical Overlap & Positivity Verification

Empirical support verified across all 20,000 synthetic observations:
* Arm 0 (0% discount): 7,584 records
* Arm 1 (5% discount): 5,395 records
* Arm 2 (10% discount): 4,266 records
* Arm 3 (20% discount): 2,755 records

Positivity ($P(T = t \mid W) > 0$) is supported across all arms.

---

## 5. Automated Regression Test Suite

Automated verification tests are maintained in `tests/test_causal_graph.py` and run via Python's `unittest` runner:
* `test_graph_is_acyclic`: Verifies DAG structure.
* `test_treatment_and_outcome_in_dataset`: Verifies treatment and outcome column parity.
* `test_confounders_exist_in_dataset`: Verifies all 8 adjustment variables exist in the schema.
* `test_no_prohibited_variables_in_adjustment_set`: Asserts no post-treatment metrics or identifiers leak into $W$.
* `test_dowhy_identification_successful`: Confirms DoWhy backdoor identification completes with 8 adjustment variables.

---

## 6. Generated Visual Artifacts
* `reports/figures/causal_dag.dot`
* `reports/figures/causal_dag.png`
