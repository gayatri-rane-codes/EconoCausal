# Phase 4, Day 3: Budget Optimization API Documentation

## Endpoint Purpose
Exposes the budget optimization service through a dedicated REST API endpoint:
- Primary: `POST /optimization/allocate`
- Backward Compatibility Alias: `POST/optimization/optimize`

Accepts overall budget constraints, optional custom candidate profit/cost matrices, and solver configuration settings to return optimal promotional arm assignments across customers.

3# Request & Response Contracts

### Request Payload Example (`POST /optimization/allocate`)
``jon
{
  "budget": 300.0,
  "mode": "bounded_gap",
  "cohort_id": "cohort_2026_q4",
  "use_cache": true,
  "check_persistence": true,
  "include_allocations": true,
  "solver_settings": {
    "mip_rel_gap": 0.05,
    "time_limit": 10.0,
    "allow_greedy_fallback": true
  }
}
```

### Response Payload Example (`200 OK`)
``json
{
  "request_budget": 300.0,
  "optimization_mode": "bounded_gap",
  "method_executed": "milp_bounded_gap_and_time",
  "status": "feasible_suboptimal",
  "solver_message": "Optimization terminated successfully. (HiGHS Status 7: Optimal)",
  "is_feasible": true,
  "is_optimal_proven": false,
  "cache_hit": false,
  "elapsed_seconds": 0.069,
  "solver_time_seconds": 0.055,
  "mip_gap": 0.0,
  "mip_dual_bound": 510.0,
  "metrics": {
    "total_expected_incremental_profit": 510.0,
    "total_budget_spent": 300.0,
    "budget_remaining": 0.0,
    "arm_counts": {
      "0": 40,
      "1": 60,
      "2": 0,
      "3": 0
    },
    "profit_per_customer": 5.1,
    "cohort_size": 100
  },
  "allocations": [
    {
      "customer_id": "cust_0001",
      "assigned_arm": 1,
      "discount_rate": 0.1,
      "expected_incremental_profit": 8.5,
      "allocated_budget_cost": 5.0
    }
  ],
  "raw_solver_status": "Optimal",
  "provenance": "live_optimization"
  }
```

## How the Optimizer is Integrated
- Built on top of `src.optimization.service.OptimizationService` (`optimize()`).
- Reuses `src.optimization.budget.solve_budget_constrained_allocation`.
- Pydantic models in `src/api/schemas.py` enforce strict request validation with `extra="forbid"`.
- Inter-layer contract alignment guarantees that API input models map directly to domain dataclasses (`SolverSettings`, `OptimizationRequest`).

## Supported Optimization Modes
- `exact`: MILP optimization solving to global optimality within time limits.
- `bounded_gap`: MIP optimization targeting relative gap tolerance (`mip_rel_gap`).
- `greedy`: Fast fractional density heuristic with discrete rounding for latency-sensitive allocations.

3# Budget Validation and Error Handling
- **Negative or Invalid Budget**: HTTP 422 Unprocessable Entity (field validator `ge=0.0`).
- **Disallowed Extra Fields**: HTTP 422 Unprocessable Entity (`extra="forbid"`).
- **Unsupported Optimization Mode**: HTTP 422 Unprocessable Entity.
-$*Service Unavailability**: HTTP 503 Service Unavailable when the optimization service is uninitialized or unavailable.
- **Solver Core Failures**: HTTP 500 Internal Server Error without leaking internal stack traces.

3# Solver Metadata & Causal Isolation Rules
- Causal predictions represent counterfactual expectations; no ground-truth datasets (`ground_truth_evaluation.csv`) are touched or evaluated during live inference.
- Solver metadata (`mip_gap`, `mip_dual_bound`, `solver_time_seconds`, `is_optimal_proven`) is transparently surfaced without fabricating mathematical guarantees.

## Test Results
- Router tests (`tests/test_optimization_router.py`): 10 passed.
- Pytest full suite: 92 passed.
- Unittest discovery suite: 75 passed.

## Known Limitations
- Custom candidate matrices require keys formatted with standard arm column names (`arm_0` through `arm_3`).
- Budget frontiers and multi-cohort batch queries are scheduled for subsequent phases.
