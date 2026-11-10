# Phase 3, Day 5: Offline Budget-Frontier Precomputation & Optimization Service Integration

## 1. Overview & Objective
Phase 3, Day 5 implements a reproducible offline job to precompute the budget-profit frontier across a canonical budget grid for default customer cohorts. It persists validated, tamper-evident optimization artifacts and integrates them cleanly with OptimizationService for sub-millisecond retrieval without re-running MILP solvers.

---

## 2. Investigation & Resolution of Day 4 Discrepancies

### A. Solver Status Semantics Discrepancy
* **Issue Observed in Day 4 Documentation/Payloads**: 
  A solver response logged solver_message: "Optimization terminated successfully. (HiGHS Status 7: Optimal)", yet set status: "feasible_suboptimal" and is_optimal_proven: false.
* **Root Cause & Code Audit**:
  In Day 4's service implementation, when executing under bounded_gap mode, the service unilaterally tagged the outcome as feasible_suboptimal and forced is_optimal_proven = False based solely on the requested mode name, ignoring the underlying solver termination code.
* **Resolution**:
  HiGHS returns status 7 when a solution meets global optimality within numerical tolerances. Under Day 5 semantics:
  - If the solver returns status 7 (or proves optimality within floating-point tolerance <= 1e-5), the service sets status = "optimal_proven" and is_optimal_proven = True.
  - If the solver terminates strictly due to reaching the non-zero relative MIP gap tolerance (mip_rel_gap) without proving global optimality, it is accurately tagged as status = "feasible_bounded" and is_optimal_proven = False.
  - Greedy heuristics are tagged as status = "feasible_greedy" and is_optimal_proven = False.
  - Raw solver status codes, termination messages, requested tolerances, and achieved gaps are preserved verbatim.

### B. Cold-Request Latency Discrepancy (83.488 ms)
* **Issue Observed in Day 4 Reports**:
  reports/day4_cache_benchmark_summary.json recorded a cold cache miss latency of 83.488 ms, leading to potential confusion that a 20,000-customer full-population MILP was solved in sub-100ms.
* **Audit of scripts/benchmark_cache_service.py**:
  Inspection of lines 17-18 revealed:
  cohort_A = df.iloc[:500].copy()
  cohort_B = df.iloc[500:1000].copy()
* **Measurement Scope Resolution**:
  - The 83.488 ms latency was strictly measured on a 500-customer slice, not the 20,000-customer cohort.
  - The candidate policy matrices (p_mat, c_mat) were pre-computed ahead of time before starting the timer.
  - Therefore, 83.488 ms represents only the in-memory wrapper dispatch and a 500-customer MILP solve.
  - A cold MILP solve on the full 20,000 customer population requires ~8-12 seconds.

---

## 3. Precomputation CLI & Workflow

The CLI script scripts/precompute_budget_frontier.py generates deterministic frontier artifacts across the canonical budget grid.

### Canonical Budget Grid
- $10,000.00
- $25,000.00
- $50,000.00
- $75,000.00
- $100,000.00

### CLI Usage Examples
```bash
python scripts/precompute_budget_frontier.py --mode bounded_gap --mip-gap 0.005
python scripts/precompute_budget_frontier.py --mode exact --output-dir reports/precomputed_frontiers
python scripts/precompute_budget_frontier.py --mode greedy
