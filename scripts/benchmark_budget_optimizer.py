"""
Phase 3 Day 3 Benchmark: Exact vs. Bounded Optimization vs. Greedy Baseline.
Compares:
  1. Exact MILP (proven optimal)
  2. Bounded Gap MILP (mip_rel_gap = 0.005 / 0.5%)
  3. Bounded Time MILP (time_limit = 15.0s)
  4. Greedy Cost-Efficiency Heuristic Baseline
Across the promotional budget frontier ($10k - $100k) on the full 20,000 customer cohort.
"""

import os
import sys
import time
import json

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
from typing import Dict, List, Any

from src.causal.estimators import CausalDMLEstimator
from src.causal.policy import compute_candidate_policy_table
from src.optimization.budget import (
    optimize_budget_allocation,
    optimize_greedy_allocation,
)

RANDOM_SEED = 20261024
BUDGET_SWEEP = [10000.0, 25000.0, 50000.0, 75000.0, 100000.0]


def run_benchmarks():
    print("=" * 75)
    print("ECONOCAUSAL PHASE 3 DAY 3: BOUNDED MILP BENCHMARK & PERFORMANCE COMPARISON")
    print("=" * 75)

    data_path = os.path.join("data", "raw", "econocausal_synthetic_dataset.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    print(f"Loading dataset: {data_path}...")
    df = pd.read_csv(data_path)
    n_total = len(df)
    print(f"Loaded {n_total:,} customer records.")

    print("Fitting CausalDMLEstimator (cv=3) for CATE estimation...")
    t0 = time.time()
    estimator = CausalDMLEstimator(random_state=RANDOM_SEED, cv=3)
    estimator.fit(df)
    cate_df = estimator.estimate_effects(df)
    print(f"CATE estimation completed in {time.time() - t0:.2f}s.")

    print("Building candidate policy tables in strict canonical order...")
    full_profit_matrix, full_cost_matrix = compute_candidate_policy_table(
        cate_df=cate_df,
        econ_df=df,
        spend_tail_threshold=60000.0,
        default_baseline_prob=0.23,
    )

    records: List[Dict[str, Any]] = []

    # Benchmark configurations
    configs = [
        {"name": "greedy", "kwargs": {}, "fn": "greedy"},
        {"name": "bounded_gap_0.5pct", "kwargs": {"mip_rel_gap": 0.005}, "fn": "milp"},
        {"name": "bounded_time_15s", "kwargs": {"time_limit": 15.0, "allow_greedy_fallback": True}, "fn": "milp"},
        {"name": "exact_milp", "kwargs": {}, "fn": "milp"},
    ]

    for b in BUDGET_SWEEP:
        print(f"\n--- Budget Level: ${b:,.0f} ---")
        for cfg in configs:
            m_name = cfg["name"]
            fn_type = cfg["fn"]
            kwargs = cfg["kwargs"]

            t_start = time.time()
            if fn_type == "greedy":
                res = optimize_greedy_allocation(full_profit_matrix, full_cost_matrix, budget=b)
            else:
                res = optimize_budget_allocation(full_profit_matrix, full_cost_matrix, budget=b, **kwargs)
            elapsed = time.time() - t_start

            tot_p = res["total_expected_incremental_profit"]
            spent = res["total_budget_spent"]
            c0 = res["arm_counts"][0]
            c1 = res["arm_counts"][1]
            c2 = res["arm_counts"][2]
            c3 = res["arm_counts"][3]

            print(
                f"[{m_name:18s}] Profit: ${tot_p:10.2f} | Spent: ${spent:9.2f} | "
                f"Status: {res['status']:19s} | Time: {elapsed:6.3f}s | "
                f"Feasible: {res['is_feasible']} | OptimalProven: {res['is_optimal_proven']}"
            )

            records.append({
                "budget": b,
                "configuration": m_name,
                "method_tag": res["method"],
                "status": res["status"],
                "solver_message": res["solver_message"],
                "is_feasible": res["is_feasible"],
                "is_optimal_proven": res["is_optimal_proven"],
                "mip_gap_reported": res.get("mip_gap"),
                "mip_dual_bound": res.get("mip_dual_bound"),
                "elapsed_seconds": round(elapsed, 4),
                "incremental_profit": tot_p,
                "profit_per_customer": round(tot_p / n_total, 4),
                "budget_spent": spent,
                "budget_remaining": res["budget_remaining"],
                "arm_0_count": c0,
                "arm_1_count": c1,
                "arm_2_count": c2,
                "arm_3_count": c3,
                "arm_0_pct": round(c0 / n_total * 100.0, 2),
                "arm_1_pct": round(c1 / n_total * 100.0, 2),
                "arm_2_pct": round(c2 / n_total * 100.0, 2),
                "arm_3_pct": round(c3 / n_total * 100.0, 2),
            })

    # Save to reports/
    res_df = pd.DataFrame(records)
    csv_out = os.path.join("reports", "day3_optimizer_performance_results.csv")
    json_out = os.path.join("reports", "day3_optimizer_performance_summary.json")

    res_df.to_csv(csv_out, index=False)
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    print(f"\nSaved Day 3 benchmark outputs to:\n  - {csv_out}\n  - {json_out}")
    print("=" * 75)


if __name__ == "__main__":
    run_benchmarks()
