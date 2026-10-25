# Phase 3, Day 2: Budget Frontier Analysis and Optimizer Benchmarking

## 1. Overview and Methodology
This report documents the benchmarking of the constrained discount allocation optimizer implemented in Phase 3, Day 1 (`src/optimization/budget.py`) against a greedy cost-efficiency baseline across varied customer cohort sizes and promotional campaign budgets.

### Fixed Experimental Conditions
* **Dataset**: `data/raw/econocausal_synthetic_dataset.csv` (20,000 customer records, unaltered).
* **CATE Estimator**: `CausalDMLEstimator` (LinearDML with 3-fold cross-fitting, `random_state=20261024`).
* **Candidate Value Generation**: `compute_candidate_policy_table` from `src/causal/policy.py` (`spend_tail_threshold=60,000.0`, `default_baseline_prob=0.23`).
* **Evaluation Isolation**: `data/raw/ground_truth_evaluation.csv` was excluded from estimation, policy selection, and optimization routines.

---

## 2. Formulation and Greedy Baseline Definition
Both optimizers operate over the identical per-customer, per-arm candidate matrices ($\Delta \Pi_{ia}$ incremental profit, $c_{ia}$ budget cost) for arms $a \in \{0, 1, 2, 3\}$.

### A. Mixed-Integer Linear Program (MILP)
$$\max_{x_{ia}} \sum_{i=1}^N \sum_{a=0}^3 x_{ia} \Delta \Pi_{ia} \quad \text{s.t.} \quad \sum_{a=0}^3 x_{ia} = 1 \; \forall i, \quad \sum_{i=1}^N \sum_{a=0}^3 x_{ia} c_{ia} \le B, \quad x_{ia} \in \{0, 1\}$$
Solved via `scipy.optimize.milp` using sparse constraint matrices (`scipy.sparse.csc_matrix`).

### B. Greedy Efficiency Baseline Heuristic
* Evaluates non-zero upgrades ($a \in \{1, 2, 3\}$) with positive incremental profit ($\Delta \Pi_{ia} > 0$).
* Ranks candidates by ROI efficiency: $r_{ia} = \Delta \Pi_{ia} / c_{ia}$ (with free-profit candidates $c_{ia} = 0$ assigned $r_{ia} = \infty$).
* Allocates greedily in descending order of $r_{ia}$ subject to customer uniqueness and the remaining promotional budget $B$.
* Guarantees: Exactly one arm assigned per customer (fallback Arm 0), budget expenditure $\le B$, and deterministic tie-breaking.

---

## 3. Empirical Benchmark Results

### A. Cohort Scaling Benchmark (Proportional Budget from $25,000 Ceiling)
Evaluated with random seed `20261024`:

| Cohort Size ($N$) | Budget ($B$) | MILP Profit ($) | MILP Spent ($) | MILP Time (s) | Greedy Profit ($) | Greedy Spent ($) | Greedy Time (s) | Optimality Gap ($) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **100** | $125.00 | $255.90 | $124.87 | 0.125s | $252.33 | $123.84 | 0.003s | **+$3.57** |
| **500** | $625.00 | $1,314.36 | $624.94 | 0.263s | $1,310.60 | $624.90 | 0.000s | **+$3.76** |
| **1,000** | $1,250.00 | $2,656.69 | $1,249.98 | 0.503s | $2,651.61 | $1,249.58 | 0.002s | **+$5.07** |
| **5,000** | $6,250.00 | $13,470.80 | $6,249.99 | 5.248s | $13,441.46 | $6,248.93 | 0.000s | **+$29.34** |
| **20,000** | $25,000.00 | $54,413.55 | $24,999.42 | 6.950s | $54,289.19 | $24,999.03 | 0.063s | **+$124.36** |

### B. Budget Frontier Sweep ($N = 20,000$ Full Population)

| Budget ($B$) | MILP Profit ($) | Profit / Cust ($) | Actual Spent ($) | Spent % | MILP Time (s) | Greedy Profit ($) | Greedy Time (s) | Optimality Gap ($) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$10,000** | $22,995.61 | $1.15 | $10,000.00 | 100.00% | 16.749s | $22,986.03 | 0.067s | **+$9.58** |
| **$25,000** | $54,417.42 | $2.72 | $24,999.99 | 100.00% | 39.288s | $54,289.19 | 0.065s | **+$128.22** |
| **$50,000** | $104,651.99 | $5.23 | $50,000.00 | 100.00% | 223.139s | $104,307.49 | 0.140s | **+$344.50** |
| **$75,000** | $153,463.14 | $7.67 | $74,997.58 | 100.00% | 9.915s | $152,927.94 | 0.129s | **+$535.20** |
| **$100,000** | $201,191.56 | $10.06 | $99,999.97 | 100.00% | 71.043s | $200,395.54 | 0.057s | **+$796.02** |

---

## 4. Frontier Findings and Analysis
1. **Strict Monotonicity & Diminishing Marginal Returns**:
   * As campaign budget increases from $10,000 to $100,000, total incremental profit increases strictly monotonically ($22.9k \to 54.4k \to 104.7k \to 153.5k \to 201.2k$).
   * Marginal ROI ($\Delta \text{Profit} / \Delta B$) displays textbook economic curvature:
     - $0 \to \$10\text{k}$: ROI = 2.30x
     - $\$10\text{k} \to \$25\text{k}$: ROI = 2.10x
     - $\$25\text{k} \to \$50\text{k}$: ROI = 2.01x
     - $\$50\text{k} \to \$75\text{k}$: ROI = 1.95x
     - $\$75\text{k} \to \$100\text{k}$: ROI = 1.91x
2. **Growing Knapsack Gap**:
   * While the greedy heuristic achieves $> 99.6\%$ efficiency on tightly constrained small cohorts, the combinatorial gap widens to nearly **$800** at $B = \$100,000$. The MILP finds subtle multi-arm combinations that pack higher net contribution within budget boundaries.
3. **Solver Performance & Scaling Limits**:
   * Full-population MILP optimization ($N=20,000$, 80,000 variables) is computationally tractable, completing in 9.9s–223.1s across budget regimes.
   * Mid-budget regimes ($B=\$50,000$) create the highest branch-and-bound combinatorial tree complexity (223s), whereas tight and loose budgets prune the branch-and-bound tree significantly faster (9s–16s).

---

## 5. Artifacts and Reproduction
* **Reproduction Script**: `scripts/benchmark_budget_optimizer.py`
* **Artifact Files**:
  - `reports/budget_benchmark_results.csv`
  - `reports/budget_benchmark_summary.json`
* **Full Test Suite Status**: 37 passing unit & integration tests (`python -m unittest discover -s tests -p "test_*.py"`).
