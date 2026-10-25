"""
Benchmark script for constrained budget optimization.
Compares MILP (scipy.optimize.milp) against a Greedy ROI baseline heuristic
across cohort sample sizes and a promotional budget frontier sweep ($10k - $100k).
Saves outputs to reports/.
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
COHORT_SIZES = [100, 500, 1000, 5000, 20000]


def run_benchmarks():
    print("=" * 70)
    print("ECONOCAUSAL PHASE 3 DAY 2: BUDGET OPTIMIZER BENCHMARK")
    print("=" * 70)

    # 1. Load data
    data_path = os.path.join("data", "raw", "econocausal_synthetic_dataset.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    print(f"Loading dataset: {data_path}...")
    df = pd.read_csv(data_path)
    n_total = len(df)
    print(f"Loaded {n_total:,} customer records.")

    # 2. Fit CATE estimator to obtain real customer-level treatment effects
    print("Fitting CausalDMLEstimator (cv=3) to estimate candidate CATEs...")
    t0 = time.time()
    estimator = CausalDMLEstimator(random_state=RANDOM_SEED, cv=3)
    estimator.fit(df)
    cate_df = estimator.estimate_effects(df)
    print(f"CATE estimation completed in {time.time() - t0:.2f}s.")

    # 3. Compute per-customer, per-arm candidate matrices for the full cohort
    print("Building full candidate policy table...")
    full_profit_matrix, full_cost_matrix = compute_candidate_policy_table(
        cate_df=cate_df,
        econ_df=df,
        spend_tail_threshold=60000.0,
        default_baseline_prob=0.23,
    )

    records: List[Dict[str, Any]] = []

    # 4. Cohort Scaling Benchmark (fixed global budget rate = $25,000 for 20k)
    fixed_budget = 25000.0
    print("\n--- 1. COHORT SCALING BENCHMARK (Proportional Budget from $25,000 ceiling) ---")

    np.random.seed(RANDOM_SEED)
    shuffled_indices = np.random.permutation(n_total)

    for n_sub in COHORT_SIZES:
        sub_idx = shuffled_indices[:n_sub]
        p_sub = full_profit_matrix.iloc[sub_idx]
        c_sub = full_cost_matrix.iloc[sub_idx]

        sub_budget = fixed_budget * (n_sub / n_total)

        # A. MILP Solver
        t_start = time.time()
        milp_res = optimize_budget_allocation(p_sub, c_sub, budget=sub_budget)
        milp_time = time.time() - t_start

        # B. Greedy Baseline
        t_start = time.time()
        greedy_res = optimize_greedy_allocation(p_sub, c_sub, budget=sub_budget)
        greedy_time = time.time() - t_start

        opt_gap = (
            round(milp_res["total_expected_incremental_profit"] - greedy_res["total_expected_incremental_profit"], 2)
            if milp_res["status"] == "success"
            else None
        )

        print(
            f"Cohort N={n_sub:5d} | "
            f"MILP Profit: ${milp_res['total_expected_incremental_profit']:9.2f} (spent ${milp_res['total_budget_spent']:8.2f}, {milp_time:6.3f}s) | "
            f"Greedy Profit: ${greedy_res['total_expected_incremental_profit']:9.2f} (spent ${greedy_res['total_budget_spent']:8.2f}, {greedy_time:6.3f}s) | "
            f"Gap: ${opt_gap if opt_gap is not None else 0.0:6.2f}"
        )

        for name, res, elapsed in [("milp", milp_res, milp_time), ("greedy", greedy_res, greedy_time)]:
            tot_p = res["total_expected_incremental_profit"]
            c0 = res["arm_counts"][0]
            c1 = res["arm_counts"][1]
            c2 = res["arm_counts"][2]
            c3 = res["arm_counts"][3]
            records.append({
                "experiment": "cohort_scaling",
                "cohort_size": n_sub,
                "budget": sub_budget,
                "method": name,
                "status": res["status"],
                "solver_message": res["solver_message"],
                "elapsed_seconds": round(elapsed, 4),
                "incremental_profit": tot_p,
                "profit_per_customer": round(tot_p / n_sub, 4),
                "budget_spent": res["total_budget_spent"],
                "budget_remaining": res["budget_remaining"],
                "arm_0_count": c0,
                "arm_1_count": c1,
                "arm_2_count": c2,
                "arm_3_count": c3,
                "arm_0_pct": round(c0 / n_sub * 100.0, 2),
                "arm_1_pct": round(c1 / n_sub * 100.0, 2),
                "arm_2_pct": round(c2 / n_sub * 100.0, 2),
                "arm_3_pct": round(c3 / n_sub * 100.0, 2),
            })

    # 5. Budget Frontier Sweep on Full 20,000 Cohort
    print("\n--- 2. BUDGET FRONTIER SWEEP (N = 20,000 Full Cohort) ---")
    for b in BUDGET_SWEEP:
        # A. MILP Solver
        t_start = time.time()
        milp_res = optimize_budget_allocation(full_profit_matrix, full_cost_matrix, budget=b)
        milp_time = time.time() - t_start

        # B. Greedy Baseline
        t_start = time.time()
        greedy_res = optimize_greedy_allocation(full_profit_matrix, full_cost_matrix, budget=b)
        greedy_time = time.time() - t_start

        opt_gap = (
            round(milp_res["total_expected_incremental_profit"] - greedy_res["total_expected_incremental_profit"], 2)
            if milp_res["status"] == "success"
            else None
        )

        print(
            f"Budget: ${b:8,.0f} | "
            f"MILP Profit: ${milp_res['total_expected_incremental_profit']:10.2f} (spent ${milp_res['total_budget_spent']:9.2f}, {milp_time:6.3f}s) | "
            f"Greedy Profit: ${greedy_res['total_expected_incremental_profit']:10.2f} (spent ${greedy_res['total_budget_spent']:9.2f}, {greedy_time:6.3f}s) | "
            f"Gap: ${opt_gap if opt_gap is not None else 0.0:7.2f}"
        )

        for name, res, elapsed in [("milp", milp_res, milp_time), ("greedy", greedy_res, greedy_time)]:
            tot_p = res["total_expected_incremental_profit"]
            c0 = res["arm_counts"][0]
            c1 = res["arm_counts"][1]
            c2 = res["arm_counts"][2]
            c3 = res["arm_counts"][3]
            records.append({
                "experiment": "budget_frontier",
                "cohort_size": n_total,
                "budget": b,
                "method": name,
                "status": res["status"],
                "solver_message": res["solver_message"],
                "elapsed_seconds": round(elapsed, 4),
                "incremental_profit": tot_p,
                "profit_per_customer": round(tot_p / n_total, 4),
                "budget_spent": res["total_budget_spent"],
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

    # Save results to reports/
    res_df = pd.DataFrame(records)
    csv_out = os.path.join("reports", "budget_benchmark_results.csv")
    json_out = os.path.join("reports", "budget_benchmark_summary.json")

    res_df.to_csv(csv_out, index=False)
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    print(f"\nSaved benchmark outputs to:\n  - {csv_out}\n  - {json_out}")
    print("=" * 70)


if __name__ == "__main__":
    run_benchmarks()
