"""
Phase 3 Day 4: Comprehensive Optimization Service and Cache Benchmark.
Evaluates all required scenarios from Prompt Section 5:
  1. First request with a cache miss (Cold Cache)
  2. Repeated identical request with a cache hit (Warm Cache)
  3. Same budget but different candidate cohort (Cohort miss / non-collision)
  4. Same cohort but different solver mode or settings (Mode miss / non-collision)
  5. Cache eviction and subsequent recomputation
"""

import os
import sys
import time
import json
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.optimization.schemas import OptimizationRequest, SolverSettings
from src.optimization.service import OptimizationService
from src.causal.estimators import CausalDMLEstimator
from src.causal.policy import compute_candidate_policy_table

RANDOM_SEED = 20261024


def run_cache_benchmarks():
    print("=" * 75)
    print("ECONOCAUSAL PHASE 3 DAY 4: COMPREHENSIVE SERVICE & CACHE BENCHMARK")
    print("=" * 75)

    data_path = os.path.join("data", "raw", "econocausal_synthetic_dataset.csv")
    df = pd.read_csv(data_path)

    # Prepare two distinct cohorts for comparison
    cohort_A = df.iloc[:500].copy()
    cohort_B = df.iloc[500:1000].copy()

    estimator = CausalDMLEstimator(random_state=RANDOM_SEED, cv=2)
    estimator.fit(cohort_A)
    cate_A = estimator.estimate_effects(cohort_A)
    p_mat_A, c_mat_A = compute_candidate_policy_table(cate_A, cohort_A, 60000.0, 0.23)

    cate_B = estimator.estimate_effects(cohort_B)
    p_mat_B, c_mat_B = compute_candidate_policy_table(cate_B, cohort_B, 60000.0, 0.23)

    # Use small capacity cache to test eviction
    service = OptimizationService(cache_size=2)

    results = []

    # Scenario 1: First request with a cache miss
    req_base = OptimizationRequest(budget=1000.0, mode="bounded_gap", solver_settings=SolverSettings(mip_rel_gap=0.005))
    t0 = time.time()
    resp1 = service.optimize(req_base, p_mat_A, c_mat_A, use_cache=True)
    t1 = (time.time() - t0) * 1000
    print(f"1. First Request (Cold Miss)      : {t1:8.3f} ms | CacheHit={resp1.cache_hit} | Profit=${resp1.metrics.total_expected_incremental_profit:.2f}")
    results.append({"scenario": "cold_cache_miss", "latency_ms": round(t1, 3), "cache_hit": resp1.cache_hit})

    # Scenario 2: Repeated identical request with a cache hit
    t0 = time.time()
    resp2 = service.optimize(req_base, p_mat_A, c_mat_A, use_cache=True)
    t2 = (time.time() - t0) * 1000
    print(f"2. Repeated Request (Warm Hit)    : {t2:8.3f} ms | CacheHit={resp2.cache_hit} | Profit=${resp2.metrics.total_expected_incremental_profit:.2f}")
    results.append({"scenario": "warm_cache_hit", "latency_ms": round(t2, 3), "cache_hit": resp2.cache_hit})

    # Scenario 3: Same budget but different candidate cohort
    t0 = time.time()
    resp3 = service.optimize(req_base, p_mat_B, c_mat_B, use_cache=True)
    t3 = (time.time() - t0) * 1000
    print(f"3. Different Cohort (Same Budget) : {t3:8.3f} ms | CacheHit={resp3.cache_hit} | Profit=${resp3.metrics.total_expected_incremental_profit:.2f}")
    results.append({"scenario": "different_cohort_miss", "latency_ms": round(t3, 3), "cache_hit": resp3.cache_hit})

    # Scenario 4: Same cohort but different solver mode (greedy)
    req_greedy = OptimizationRequest(budget=1000.0, mode="greedy")
    t0 = time.time()
    resp4 = service.optimize(req_greedy, p_mat_A, c_mat_A, use_cache=True)
    t4 = (time.time() - t0) * 1000
    print(f"4. Different Mode (Greedy)        : {t4:8.3f} ms | CacheHit={resp4.cache_hit} | Profit=${resp4.metrics.total_expected_incremental_profit:.2f}")
    results.append({"scenario": "different_mode_miss", "latency_ms": round(t4, 3), "cache_hit": resp4.cache_hit})

    # Scenario 5: Cache eviction and subsequent recomputation
    # Cache capacity is 2; adding resp4 evicted req_base on cohort_A
    print(f"Cache Eviction Count: {service.cache.evictions}")
    t0 = time.time()
    resp5 = service.optimize(req_base, p_mat_A, c_mat_A, use_cache=True)
    t5 = (time.time() - t0) * 1000
    print(f"5. Evicted Item (Recomputed Miss) : {t5:8.3f} ms | CacheHit={resp5.cache_hit} | Profit=${resp5.metrics.total_expected_incremental_profit:.2f}")
    results.append({"scenario": "evicted_recomputation_miss", "latency_ms": round(t5, 3), "cache_hit": resp5.cache_hit})

    summary = {
        "benchmarks": results,
        "cache_statistics": {
            "capacity": service.cache.maxsize,
            "hits": service.cache.hits,
            "misses": service.cache.misses,
            "evictions": service.cache.evictions,
        }
    }

    out_file = os.path.join("reports", "day4_cache_benchmark_summary.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved updated benchmark summary to {out_file}")
    print("=" * 75)


if __name__ == "__main__":
    run_cache_benchmarks()
