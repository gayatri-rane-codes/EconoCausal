# Phase 3, Day 1: Constrained Budget Optimizer Design and Implementation

## 1. Resolution of Critical Economic and Evaluation Discrepancies
An empirical investigation was conducted to resolve discrepancies between earlier reports, the Phase 2 final report, and ground-truth values:

### A. Ground-Truth Population Economic Returns
* An audit of `data/raw/ground_truth_evaluation.csv` reveals population-wide average incremental profits of:
  - Arm 1 (5% discount): **-$8.30**
  - Arm 2 (10% discount): **-$14.37**
  - Arm 3 (20% discount): **-$42.19**
* Across the unselected population, blanket discounts are value-destroying because margin erosion ($P_a \times d_a \times B$) dominates conversion lift ($\tau_a \times (B - C)$).
* For the vast majority of shoppers, Arm 0 (0% discount) is the true profit-maximizing decision (oracle best arm median is 0.00).

### B. Explanation of the $52.01 Mean Incremental Profit
* The $52.01 figure reported in Phase 2 was the conditional mean under the max-profit policy assignment: when no discount was economically viable, customers were routed to Arm 0 ($0.00 incremental profit).
* The mean was computed exclusively over selected assignments where negative outcomes were zeroed out by falling back to Arm 0.
* In `calculate_incremental_expected_profit`, using the global empirical baseline purchase probability ($P_0 \approx 0.23$) versus the default fallback ($P_0 = 0.35$) alters the margin erosion calculation: higher baseline purchase probabilities inflate the cost of subsidizing organic conversions, reducing estimated incremental profit.

### C. Variations in Estimated Average Treatment Effects (ATE)
* Earlier exploratory reports and the Phase 2 final report exhibited slight variations in estimated CATE/ATE (e.g., Arm 1 ATE ranging from +0.0382 to +0.0392).
* These differences arise from nuisance model cross-fitting:
  - LinearDML cross-validation fold counts (`cv=3` vs `cv=5`).
  - Cross-fitting random seeds (e.g., `random_state=20261012` vs unseeded `KFold`).
  - Feature scaling: applying `StandardScaler` inside nuisance pipelines stabilizes regularization but slightly shifts individual weight penalties.

---

## 2. Mathematical Optimization Formulation
The customer-level constrained assignment problem allocates exactly one treatment arm $a \in \{0, 1, 2, 3\}$ to each customer $i \in \{1, \dots, N\}$ to maximize aggregate incremental profit while respecting a total campaign promotional budget $B$:

$$\max_{x_{ia}} \sum_{i=1}^N \sum_{a=0}^3 x_{ia} \Delta \Pi_{ia}$$

Subject to:
1. **Assignment Uniqueness**: Exactly one arm assigned per customer:
   $$\sum_{a=0}^3 x_{ia} = 1 \quad \forall i \in \{1, \dots, N\}$$
2. **Campaign Budget Ceiling**: Total promotional expenditure cannot exceed budget $B$:
   $$\sum_{i=1}^N \sum_{a=0}^3 x_{ia} c_{ia} \le B$$
3. **Binary Integrality**: Strict zero-one assignment (no fractional coupons):
   $$x_{ia} \in \{0, 1\} \quad \forall i, a$$

---

## 3. Economic Definitions and Budget Cost Structure
* **Incremental Profit ($\Delta \Pi_{ia}$)**:
  $$\Delta \Pi_{ia} = \tau_{ia}(B_i - C_i) - P_{ia}(d_a \times B_i) - \Delta \text{MarketingCost}_{ia}$$
  where:
  - $B_i$: Basket value before discount
  - $C_i$: Cost of goods sold (COGS)
  - $P_{ia} = \text{clip}(P_{i0} + \tau_{ia}, 0, 1)$: Expected conversion probability under arm $a$
  - $d_a \in \{0.00, 0.05, 0.10, 0.20\}$: Discount rate
  - For Arm 0, $\Delta \Pi_{i0} \equiv 0.00$.
* **Budget Expenditure ($c_{ia}$)**:
  $$c_{ia} = P_{ia}(d_a \times B_i)$$
  representing the expected promotional discount subsidy allocated. For Arm 0 (control), $c_{i0} \equiv 0.00$.
* **No Double-Counting**: Margin erosion is incorporated directly into the objective $\Delta \Pi_{ia}$. The budget constraint is purely a resource limitation and is not subtracted a second time from the objective.

---

## 4. Policy Layer Integration & Architecture
* `src/causal/policy.py` exposes `compute_candidate_policy_table(...)` to generate candidate profit and cost matrices for all 4 arms simultaneously without pre-filtering.
* `src/optimization/budget.py` consumes this table and invokes `scipy.optimize.milp` using `scipy.sparse.csc_matrix`.
* **Safeguards Preserved**:
  - Customers with historical spend $> \$60,000$ are forced to Arm 0 ($0.00 profit, $0.00 cost) to prevent linear extrapolation in the sparse feature tail.
  - Non-finite inputs ($NaN$, $\pm\infty$) and zero-budget conditions ($B \le 10^{-5}$) cleanly fall back to the all-control allocation.

---

## 5. Hand-Checkable Reference Example & Solver Validation
3 customers, 4 discount arms, budget $B = \$15.00$:
* **Customer 1**: Arm 0 (\$0, \$0), Arm 1 (\$5, \$10), Arm 2 (\$10, \$18), Arm 3 (\$16, \$22)
* **Customer 2**: Arm 0 (\$0, \$0), Arm 1 (\$4, \$5), Arm 2 (\$8, \$12), Arm 3 (\$14, \$15)
* **Customer 3**: Arm 0 (\$0, \$0), Arm 1 (\$3, \$2), Arm 2 (\$6, \$4), Arm 3 (\$10, \$5)

**Optimal Solution**:
* Customer 1 $\to$ Arm 2 (Cost: \$10, Profit: \$18)
* Customer 2 $\to$ Arm 1 (Cost: \$4, Profit: \$5)
* Customer 3 $\to$ Arm 0 (Cost: \$0, Profit: \$0)
* **Total Cost**: $\$10 + \$4 + \$0 = \$14.00 \le \$15.00$ (Budget Remaining: \$1.00)
* **Total Incremental Profit**: $\$18 + \$5 + \$0 = \$23.00$
* **Solver Output**: Verified exact match with status `"success"`.

---

## 6. Full Test Suite Execution
* Command: `python -m unittest discover -s tests -p "test_*.py"`
* Result: **34 passing tests** across 7 test modules in **~3.85s** (0 failures, 0 errors).
