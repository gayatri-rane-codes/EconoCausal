import os
import time
import pandas as pd
import numpy as np

from src.optimization.service import OptimizationService
from src.optimization.schemas import OptimizationRequest, SolverSettings

def generate_benchmark_matrices(n_customers: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    customer_ids = [f"bench_cust_{i:06d}" for i in range(n_customers)]

    # Arm 0 has 0 incremental profit and 0 budget cost
    # Arms 1, 2, 3 have increasing costs and varied incremental profits
    profit_data = {
        "arm_0": np.zeros(n_customers),
        "arm_1": rng.normal(loc=12.0, scale=4.0, size=n_customers).clip(min=-5.0),
        "arm_2": rng.normal(loc=22.0, scale=7.0, size=n_customers).clip(min=-8.0),
        "arm_3": rng.normal(loc=30.0, scale=10.0, size=n_customers).clip(min=-12.0),
    }

    cost_data = {
        "arm_0": np.zeros(n_customers),
        "arm_1": rng.uniform(low=5.0, high=10.0, size=n_customers),
        "arm_2": rng.uniform(low=12.0, high=22.0, size=n_customers),
        "arm_3": rng.uniform(low=22.0, high=38.0, size=n_customers),
    }

    profit_df = pd.DataFrame(profit_data, index=customer_ids)
    cost_df = pd.DataFrame(cost_data, index=customer_ids)
    return profit_df, cost_df


def run_benchmarks():
    os.makedirs("reports", exist_ok=True)
    service = OptimizationService()
    sample_sizes = [100, 1000, 5000, 20000]
    modes = ["greedy", "bounded_gap"]
    
    benchmark_records = []

    print("=" * 75)
    print("EconoCausal Performance Benchmark: MILP (bounded_gap) vs Greedy")
    print("=" * 75)

    for n in sample_sizes:
        # Scale budget proportionally to test realistic allocation constraints
        budget = float(n * 8.0)
        profit_df, cost_df = generate_benchmark_matrices(n)

        for mode in modes:
            settings = SolverSettings(
                mip_rel_gap=0.01,
                time_limit=120.0,
                allow_greedy_fallback=True,
            )
            req = OptimizationRequest(
                budget=budget,
                mode=mode,
                solver_settings=settings,
                cohort_id=f"benchmark_n{n}_{mode}",
            )

            start_t = time.perf_counter()
            response = service.optimize(
                request=req,
                candidate_profit_matrix=profit_df,
                candidate_cost_matrix=cost_df,
                use_cache=False,
                check_persistence=False,
            )
            elapsed = time.perf_counter() - start_t

            profit = response.metrics.total_expected_incremental_profit
            spent = response.metrics.total_budget_spent
            utilization = (spent / budget) * 100.0 if budget > 0 else 0.0

            record = {
                "sample_size": n,
                "mode": mode,
                "elapsed_seconds": round(elapsed, 4),
                "total_incremental_profit": round(profit, 2),
                "budget_allocated": budget,
                "budget_spent": round(spent, 2),
                "budget_utilization_pct": round(utilization, 2),
                "is_feasible": response.is_feasible,
                "solver_status": response.status,
            }
            benchmark_records.append(record)
            print(
                f"N={n:6d} | Mode={mode:12s} | Time={elapsed:7.3f}s | "
                f"Profit={profit:10.2f} | Spent={spent:9.2f}/{budget:9.2f} ({utilization:5.1f}%)"
            )

    df_results = pd.DataFrame(benchmark_records)
    out_csv = os.path.join("reports", "phase4_day5_benchmark.csv")
    df_results.to_csv(out_csv, index=False)
    print("=" * 75)
    print(f"Benchmark results successfully written to {out_csv}")
    print("=" * 75)


if __name__ == "__main__":
    run_benchmarks()
