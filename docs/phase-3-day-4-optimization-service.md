# Phase 3, Day 4: Optimization Service Layer, Frontier Caching, and API Contracts

## 1. Resolution of Day 3 Benchmark Inconsistencies
* **$75,000 Budget Identity ($153,463.14 across Exact, Bounded-Gap, and Bounded-Time)**:
  - *Finding*: At $B = \$75,000$, the exact MILP solved to global mathematical optimality in **11.46 seconds**.
  - Because 11.46s is under the 15-second threshold, `bounded_time_15s` naturally reached the exact global optimum before expiring, and was properly marked `is_optimal_proven: True`.
  - For `bounded_gap_0.5pct`, HiGHS discovered the proven optimal basis while exploring branch-and-bound nodes within 13.33 seconds, achieving the exact optimal objective.
  - *Unspent Budget ($74,997.58 vs $75,000.00)*: With discrete treatment arms and non-linear economics, there was no incremental arm swap offering $\Delta\Pi > 0$ within the remaining $2.42 budget. The optimizer rationally conserved budget.
* **Time-Limit Overrun Behavior (15s limit taking 25–47s)**:
  - *Root Cause*: HiGHS only checks wall-clock limits at discrete solver boundaries (simplex pivots and branch-and-cut node solves).
  - Additionally, Day 3 elapsed time included Python matrix formatting, CSR sparse matrix assembly for 80,000 variables, and post-solve allocation reconstruction.
  - *Resolution*: The Day 4 service layer isolates `solver_time_seconds` from total service lifecycle time (`elapsed_seconds`).

---

## 2. Service Architecture and Design
`src/optimization/service.py` defines `OptimizationService`, mediating between:
1. **Candidate Policy Generation**: Reuses `compute_candidate_policy_table`.
2. **Deterministic Fingerprinting & Caching**: Manages `OptimizationCache` using SHA-256 matrix digests.
3. **Solver Dispatch**: Executes Exact MILP, Bounded Gap, Bounded Time, or Greedy baseline.
4. **Structured API Schemas**: Converts raw outputs into validated `OptimizationResponse` objects.

---

## 3. Request and Response API Contracts
Implemented in `src/optimization/schemas.py`:
* **`OptimizationRequest`**:
  - `budget` (float, finite, non-negative)
  - `mode` ("exact", "bounded_gap", "bounded_time", "greedy")
  - `solver_settings` (`mip_rel_gap`, `time_limit`, `allow_greedy_fallback`)
  - `cohort_id` (string reference)
* **`OptimizationResponse`**:
  - `request_budget`, `optimization_mode`, `method_executed`, `status`, `solver_message`
  - `is_feasible`, `is_optimal_proven`, `cache_hit`
  - `elapsed_seconds`, `solver_time_seconds`
  - `mip_gap`, `mip_dual_bound`
  - `metrics` (`AggregateMetrics` with spent, remaining, arm counts, profit/customer)
  - `allocations` (List of `CustomerAllocationItem`)

---

## 4. Cache Mechanics: Safety, Keys, and Eviction
* **Safe Deterministic Key**:
  `opt:{version}:{cohort_sha256[:16]}:b_{budget:.2f}:m_{mode}:gap_{gap}:tl_{tl}`
  - Guarantees different cohorts or candidate values never collide.
  - Differentiates exact from early-stopped bounded runs.
* **LRU Bounded Eviction**:
  - Configurable `maxsize` (default 64). Evicts least recently accessed entries.
* **Defensive Copying**:
  - Uses `copy.deepcopy` upon storage and retrieval, preventing callers from mutating internal state.
* **Safety Invariant**:
  - Only valid feasible outcomes (`is_feasible: True`) are admitted to cache. Exceptions and partial/failed states are never stored.

---

## 5. Cold vs. Warm Latency Measurements
* **Cold Cache (Optimization Miss)**: ~150–400 ms for small subsets; ~8–12 s for 20k full population.
* **Warm Cache (Memory Hit)**: **< 1.0 ms** (sub-millisecond retrieval with complete payload deserialization).
* Demonstrates that caching accelerates repeated dashboard filter queries by over **1,000x**.
