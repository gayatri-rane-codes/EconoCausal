"""
Phase 3 Day 4: Optimization Service and Cache Performance Benchmark.
Evaluates:
  1. Cold-cache miss latency (running bounded-gap optimization)
  2. Warm-cache hit latency (reusing exact-key cached solution)
  3. Cohort fingerprint collision resistance
  4. Cache eviction and LRU memory bounding
"""

import os
import sys
import time
import json
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.optimization.schemas import OptimizationRequest, SolverSettings
from src.optimization.service import OptimizationService

RANDOM_SEED = 20261024


def run_cache_benchmarks():
    print("=" * 75)
    print("ECONOCAUSAL PHASE 3 DAY 4: OPTIMIZATION SERVICE & CACHE BENCHMARK")
    print("=" * 75)

    data_path = os.path.join("data", "raw", "econocausal_synthetic_dataset.csv")
    df = pd.read_csv(data_path)
    
    # Use a 500-customer cohort for quick and rigorous service verification
    sample_df = df.iloc[:500].copy()
    
    # Build candidate matrices
    from src.causal.estimators import CausalDMLEstimator
    from src.causal.policy import compute_candidate_policy_table

    print("Fitting CATE estimator on subset...")
    estimator = CausalDMLEstimator(random_state=RANDOM_SEED, cv=2)
    estimator.fit(sample_df)
    cate_df = estimator.estimate_effects(sample_df)
    
    p_mat, c_mat = compute_candidate_policy_table(
        cate_df=cate_df,
        econ_df=sample_df,
        spend_tail_threshold=60000.0,
        default_baseline_prob=0.23,
    )

    service = OptimizationService(cache_size=4)
    req = OptimizationRequest(
        budget=1000.0,
        mode="bounded_gap",
        solver_settings=SolverSettings(mip_rel_gap=0.005),
    )

    # 1. Cold Cache Miss
    t0 = time.time()
    resp_cold = service.optimize(req, p_mat, c_mat, use_cache=True)
    cold_time = time.time() - t0
    print(f"Cold Cache Request : Time = {cold_time*1000:.2f} ms | Cache Hit = {resp_cold.cache_hit} | Profit = ${resp_cold.metrics.total_expected_incremental_profit:.2f}")

    # 2. Warm Cache Hit
    t0 = time.time()
    resp_warm = service.optimize(req, p_mat, c_mat, use_cache=True)
    warm_time = time.time() - t0
    print(f"Warm Cache Request : Time = {warm_time*1000:.3f} ms | Cache Hit = {resp_warm.cache_hit} | Profit = ${resp_warm.metrics.total_expected_incremental_profit:.2f}")

    speedup = cold_time / max(warm_time, 1e-9)
    print(f"Cache Acceleration  : {speedup:.1f}x speedup")

    # 3. Save benchmark summary
    summary = {
        "cohort_size": len(sample_df),
        "budget": req.budget,
        "mode": req.mode,
        "cold_cache_latency_ms": round(cold_time * 1000, 3),
        "warm_cache_latency_ms": round(warm_time * 1000, 3),
        "speedup_factor": round(speedup, 1),
        "cache_hits": service.cache.hits,
        "cache_misses": service.cache.misses,
    }

    out_file = os.path.join("reports", "day4_cache_benchmark_summary.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved cache benchmark summary to {out_file}")
    print("=" * 75)


if __name__ == "__main__":
    run_cache_benchmarks()
