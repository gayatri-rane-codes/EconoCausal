# Phase 3, Day 4: Optimization Service Layer, Frontier Caching, and API Contracts

## 1. Resolution of Day 3 Benchmark Inconsistencies
* **,000 Budget Uniform Profit (,463.14 across Exact, Bounded-Gap, and Bounded-Time)**:
  - *Finding*: At  = \,000$, the exact MILP solved to global mathematical optimality in **11.46 seconds**.
  - Because 11.46s is under the 15-second cutoff, bounded_time_15s completed the full branch-and-cut tree before expiring, and was properly marked is_optimal_proven: True.
  - For bounded_gap_0.5pct, HiGHS discovered this exact optimal basis within 13.33 seconds, achieving the exact optimal objective.
  - *Unspent Budget (,997.58 vs ,000.00)*: With discrete treatment arms and non-linear economics, there was no incremental arm swap offering positive incremental profit within the remaining .42 budget. The optimizer rationally conserved budget.
* **Time-Limit Overrun Behavior (15s limit taking 25–47s)**:
  - *Root Cause*: HiGHS inspects wall-clock limits at discrete solver boundaries (simplex pivot iterations and branch-and-bound node completions).
  - Additionally, Day 3 elapsed time included Python matrix formatting, CSR sparse matrix assembly for 80,000 variables, and post-solve allocation reconstruction.
  - *Resolution*: The Day 4 service layer isolates solver_time_seconds from total service lifecycle time (elapsed_seconds).

---

## 2. Service Architecture and Design
src/optimization/service.py defines OptimizationService, mediating between:
1. Candidate Policy Generation: Reuses compute_candidate_policy_table.
2. Deterministic Fingerprinting & Caching: Manages OptimizationCache using SHA-256 matrix digests.
3. Solver Dispatch: Executes Exact MILP, Bounded Gap, Bounded Time, or Greedy baseline.
4. Structured API Schemas: Converts raw outputs into validated OptimizationResponse objects.

---

## 3. Request and Response API Contracts
Implemented in src/optimization/schemas.py:
* OptimizationRequest:
  - budget (float, finite, non-negative)
  - mode ("exact", "bounded_gap", "bounded_time", "greedy")
  - solver_settings (mip_rel_gap, time_limit, allow_greedy_fallback)
  - cohort_id (string reference)
* OptimizationResponse:
  - request_budget, optimization_mode, method_executed, status, solver_message
  - is_feasible, is_optimal_proven, cache_hit
  - elapsed_seconds, solver_time_seconds
  - mip_gap, mip_dual_bound
  - metrics (AggregateMetrics with spent, remaining, arm counts, profit/customer)
  - allocations (List of CustomerAllocationItem)

---

## 4. Cache Mechanics: Safety, Keys, and Eviction
* Safe Deterministic Key:
  opt:v1:{cohort_sha256[:16]}:b_{budget:.2f}:m_{mode}:gap_{gap}:tl_{tl}
  - Guarantees different cohorts or candidate values never collide.
  - Differentiates exact from early-stopped bounded runs.
* LRU Bounded Eviction:
  - Configurable maxsize (default 64). Evicts least recently accessed entries.
* Defensive Copying:
  - Uses copy.deepcopy upon storage and retrieval, preventing callers from mutating internal state.
* Safety Invariant:
  - Only valid feasible outcomes (is_feasible: True) are admitted to cache. Exceptions and partial/failed states are never stored.

---

## 5. Empirical Service & Cache Benchmarks (5 Core Scenarios)
1. Cold Cache Miss (First Request): Full solver execution (~150–250 ms for 500-customer cohort; ~8–12 s for 20k full population).
2. Warm Cache Hit (Repeated Request): < 1.0 ms (sub-millisecond retrieval with complete payload deserialization, providing >900x speedup).
3. Different Cohort with Same Budget: Triggers deterministic SHA-256 cache miss and runs independent solve without collision.
4. Different Mode/Settings with Same Cohort: Triggers cache miss and dispatches requested mode (e.g., greedy baseline).
5. Cache Eviction and Subsequent Recomputation: Demonstrates that overflowing LRU capacity evicts the oldest entry cleanly and triggers recomputation on re-query without corrupted state.

---

## 6. Representative API JSON Examples

### Valid Optimization Request
{
  "budget": 25000.0,
  "mode": "bounded_gap",
  "solver_settings": {
    "mip_rel_gap": 0.005,
    "time_limit": null,
    "allow_greedy_fallback": true
  },
  "cohort_id": "default_full_population"
}

### Valid Optimization Response
{
  "request_budget": 25000.0,
  "optimization_mode": "bounded_gap",
  "method_executed": "milp_bounded_gap",
  "status": "feasible_suboptimal",
  "solver_message": "Optimization terminated successfully. (HiGHS Status 7: Optimal)",
  "is_feasible": true,
  "is_optimal_proven": false,
  "cache_hit": true,
  "elapsed_seconds": 0.000412,
  "solver_time_seconds": 8.494,
  "mip_gap": 0.00312,
  "mip_dual_bound": 54450.12,
  "metrics": {
    "total_expected_incremental_profit": 54334.77,
    "total_budget_spent": 24964.48,
    "budget_remaining": 35.52,
    "arm_counts": {
      "0": 13420,
      "1": 3120,
      "2": 2180,
      "3": 1280
    },
    "profit_per_customer": 2.7167,
    "cohort_size": 20000
  },
  "allocations": [
    {
      "customer_id": "cust_00001",
      "assigned_arm": 2,
      "discount_rate": 0.15,
      "expected_incremental_profit": 18.42,
      "allocated_budget_cost": 8.12
    }
  ]
}

### Invalid Request Payloads & Validation Errors

#### A. Negative Budget
{
  "budget": -5000.0,
  "mode": "bounded_gap"
}
Error Response:
{
  "error": "ValueError",
  "message": "Budget cannot be negative, got -5000.0"
}

#### B. Unsupported Optimization Mode
{
  "budget": 10000.0,
  "mode": "genetic_algorithm"
}
Error Response:
{
  "error": "ValueError",
  "message": "Invalid mode 'genetic_algorithm'. Supported modes: ['bounded_gap', 'bounded_time', 'exact', 'greedy']"
}

#### C. Invalid Solver Option Values
{
  "budget": 10000.0,
  "mode": "bounded_gap",
  "solver_settings": {
    "mip_rel_gap": -0.05
  }
}
Error Response:
{
  "error": "ValueError",
  "message": "mip_rel_gap must be non-negative and finite, got -0.05"
}
