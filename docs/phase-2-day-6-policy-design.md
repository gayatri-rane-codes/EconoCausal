# Phase 2, Day 6: Policy Assignment and Economic Value Evaluation Documentation

## 1. Executive Summary & Objective
On Day 6 of Phase 2, we designed and implemented the economic value layer (`src/causal/policy.py`) to translate CATE purchase conversion lifts into actionable, profit-maximizing discount recommendations across four treatment tiers (0%, 5%, 10%, 20%).

## 2. Economic Value Formulation
Conversion probability lift ($\Delta P$) does not equal incremental profit ($\Delta \Pi$). A discount erodes profit margins across all units that would have converted anyway under the 0% control.

### Mathematical Formulation
Let:
* $d_a \in \{0.00, 0.05, 0.10, 0.20\}$ be the discount rate for arm $a$.
* $B$ be `basket_value_before_discount`.
* $C$ be `cost_of_goods` (COGS).
* $M_{\text{pre}} = B - C$ be the pre-discount contribution margin.
* $P_0$ be baseline conversion probability under control.
* $\tau_a = \Delta P_a = P_a - P_0$ be the CATE lift from the DML estimator.
* $P_a = \min(\max(P_0 + \tau_a, 0), 1)$ be the estimated conversion probability under arm $a$.

Expected Profit under Arm $a$:
$$\mathbb{E}[\Pi_a] = P_a \times \left(B(1 - d_a) - C\right) - \text{MarketingCost}_a$$

Incremental Expected Profit relative to Control (Arm 0):
$$\Delta \mathbb{E}[\Pi_a] = \mathbb{E}[\Pi_a] - \mathbb{E}[\Pi_0]$$
$$\Delta \mathbb{E}[\Pi_a] = \tau_a \times (B - C) - P_a \times (d_a \times B) - \Delta \text{MarketingCost}_a$$

## 3. Decision Rule & Safeguards
1. **Profit Maximization**:
   $$\text{Arm}^* = \arg\max_{a \in \{0, 1, 2, 3\}} \Delta \mathbb{E}[\Pi_a]$$
   subject to $\Delta \mathbb{E}[\Pi_{\text{Arm}^*}] > 0$. If no discount yields $\Delta \mathbb{E}[\Pi_a] > 0$, the customer is assigned Arm 0 (`no_discount_economically_viable`).
2. **Extreme Spend Tail Safeguard**:
   Customers with `historical_spend` $> \$60,000$ (upper $0.1\%$ feature tail) receive Arm 0 fallback (`fallback_extreme_spend_tail`), preventing linear extrapolation artifacts from affecting high-spend VIPs.
3. **Data Integrity Safeguard**:
   Any record with missing, NaN, or non-finite inputs falls back to Arm 0 (`fallback_non_finite_input`).

## 4. Integration with SciPy Optimizer
This policy module decouples causal estimation from downstream constrained optimization. Phase 3's SciPy budget optimizer will take the output matrix of candidate incremental profits and maximize aggregate net return under total discount campaign budget constraints.
