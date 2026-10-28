"""
Offline budget-frontier precomputation CLI script.
Generates verified frontier points across canonical budgets ($10k-$100k)
and persists validated artifacts for fast retrieval by OptimizationService.
"""

import os
import sys
import time
import argparse
from typing import Optional
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.optimization.schemas import OptimizationRequest, SolverSettings
from src.optimization.service import OptimizationService
from src.optimization.persistence import save_frontier_result, load_frontier_result
from src.optimization.cache import compute_cohort_fingerprint
from src.causal.estimators import CausalDMLEstimator
from src.causal.policy import compute_candidate_policy_table

CANONICAL_BUDGET_GRID = [10000.0, 25000.0, 50000.0, 75000.0, 100000.0]
DEFAULT_OUTPUT_DIR = os.path.join("reports", "precomputed_frontiers")


def run_precomputation(
    sample_size: Optional[int] = None,
    mode: str = "bounded_gap",
    mip_rel_gap: float = 0.005,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    force_recompute: bool = False,
):
    print("=" * 75)
    print("ECONOCAUSAL PHASE 3 DAY 5: OFFLINE FRONTIER PRECOMPUTATION")
    print(f"Mode: {mode} | MIP Gap: {mip_rel_gap} | Target Dir: {output_dir}")
    print("=" * 75)

    # Set seed for reproducible cross-fitting splits
    np.random.seed(42)

    data_path = os.path.join("data", "raw", "econocausal_synthetic_dataset.csv")
    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"Raw data file not found at {data_path}")

    df = pd.read_csv(data_path)
    if sample_size is not None and sample_size > 0:
        df = df.iloc[:sample_size].copy()
        cohort_id = f"sample_{sample_size}"
    else:
        cohort_id = "full_population_20k"

    print(f"Loaded cohort '{cohort_id}' with {len(df)} customers.")

    # 1. Estimate CATE and generate candidate policy table once
    print("Fitting CausalDMLEstimator to generate candidate policy tables...")
    t0 = time.time()
    estimator = CausalDMLEstimator(random_state=42, cv=2)
    estimator.fit(df)
    cate_df = estimator.estimate_effects(df)
    p_mat, c_mat = compute_candidate_policy_table(
        cate_df=cate_df,
        econ_df=df,
        spend_tail_threshold=60000.0,
        default_baseline_prob=0.23,
    )
    prep_time = time.time() - t0
    print(f"Candidate table generated in {prep_time:.2f}s.")

    cohort_fp = compute_cohort_fingerprint(p_mat, c_mat)
    print(f"Cohort Fingerprint SHA-256: {cohort_fp}")

    service = OptimizationService(cache_size=32, persistence_dir=output_dir)

    # 2. Iterate through canonical budget grid
    for budget in CANONICAL_BUDGET_GRID:
        print("-" * 75)
        print(f"Evaluating Frontier Point: Budget = ${budget:,.2f}")

        if not force_recompute:
            filename = f"frontier_{cohort_fp[:8]}_b{int(budget)}_{mode}.json"
            filepath = os.path.join(output_dir, filename)
            existing = load_frontier_result(
                filepath=filepath,
                expected_fingerprint=cohort_fp,
                expected_budget=budget,
                expected_mode=mode,
            )
            if existing:
                print(f"[REUSED] Verified artifact already exists at {filepath}. Skipping recomputation.")
                continue

        req = OptimizationRequest(
            budget=budget,
            mode=mode,
            solver_settings=SolverSettings(mip_rel_gap=mip_rel_gap),
            cohort_id=cohort_id,
        )

        resp = service.optimize(req, p_mat, c_mat, use_cache=True, check_persistence=False)

        if not resp.is_feasible:
            print(f"[ERROR] Optimization failed for budget ${budget:,.2f}: {resp.solver_message}")
            continue

        if resp.metrics.total_budget_spent > budget + 1e-4:
            raise ValueError(f"Infeasible budget expenditure: Spent ${resp.metrics.total_budget_spent} > ${budget}")

        saved_path = save_frontier_result(
            output_dir=output_dir,
            cohort_fingerprint=cohort_fp,
            cohort_id=cohort_id,
            budget=budget,
            mode=mode,
            solver_settings={"mip_rel_gap": mip_rel_gap},
            response_dict=resp.to_dict(include_allocations=True),
        )
        print(f"[SAVED] Profit: ${resp.metrics.total_expected_incremental_profit:,.2f} | Spent: ${resp.metrics.total_budget_spent:,.2f} | Path: {saved_path}")

    print("=" * 75)
    print("Precomputation run completed successfully.")
    print("=" * 75)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Precompute budget frontier grid.")
    parser.add_argument("--sample-size", type=int, default=None, help="Customer sample size (default: full cohort).")
    parser.add_argument("--mode", type=str, default="bounded_gap", choices=["bounded_gap", "exact", "greedy"])
    parser.add_argument("--gap", type=float, default=0.005, help="Relative MIP gap.")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Destination directory.")
    parser.add_argument("--force", action="store_true", help="Force recomputation of existing artifacts.")
    args = parser.parse_args()

    run_precomputation(
        sample_size=args.sample_size,
        mode=args.mode,
        mip_rel_gap=args.gap,
        output_dir=args.output_dir,
        force_recompute=args.force,
    )
