import argparse
import json
import logging
from pathlib import Path
import sys
import numpy as np
import pandas as pd

from src.causal.estimators import CausalDMLEstimator, TLearnerEstimator
from src.causal.comparison import (
    split_training_and_evaluation,
    evaluate_estimator_predictions,
    compare_estimators_on_budget_frontier,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("benchmark_estimators")


def run_benchmark(
    data_path: str = "data/synthetic_dataset.csv",
    gt_path: str = "data/ground_truth_evaluation.csv",
    output_dir: str = "reports",
    test_size: float = 0.30,
    random_seed: int = 20261012,
    budgets: list = None,
):
    budgets = budgets or [10000.0, 25000.0, 50000.0, 75000.0, 100000.0]
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading data from {data_path} and ground truth from {gt_path}...")
    df = pd.read_csv(data_path)
    gt_df = pd.read_csv(gt_path)

    logger.info("Executing leakage-safe train/test split (70/30)...")
    train_df, test_df = split_training_and_evaluation(
        df, test_size=test_size, random_state=random_seed, id_col="customer_id"
    )
    logger.info(f"Train samples: {len(train_df)}, Held-out test samples: {len(test_df)}")

    # 1. Fit Causal Forest DML
    logger.info("Fitting CausalDMLEstimator on training cohort...")
    dml = CausalDMLEstimator(random_state=random_seed)
    dml.fit(train_df)
    dml_cate_test = dml.estimate_effects(test_df)
    dml_cate_test["customer_id"] = test_df["customer_id"].values

    # 2. Fit Multi-Arm T-Learner Baseline
    logger.info("Fitting TLearnerEstimator on training cohort...")
    tlearner = TLearnerEstimator(random_state=random_seed)
    tlearner.fit(train_df)
    tlearner_cate_test = tlearner.estimate_effects(test_df)
    tlearner_cate_test["customer_id"] = test_df["customer_id"].values

    # 3. Evaluate CATE Accuracy against Held-Out Ground Truth
    logger.info("Evaluating CATE predictions against quarantined ground truth...")
    dml_eval = evaluate_estimator_predictions(dml_cate_test, gt_df, id_col="customer_id")
    tlearner_eval = evaluate_estimator_predictions(tlearner_cate_test, gt_df, id_col="customer_id")

    metrics_report = {
        "benchmark_metadata": {
            "random_seed": random_seed,
            "test_size": test_size,
            "n_train": len(train_df),
            "n_test": len(test_df),
        },
        "estimators": {
            "causal_dml": dml_eval,
            "t_learner": tlearner_eval,
        },
        "comparison": {
            "mae_delta_tlearner_minus_dml": round(tlearner_eval["overall_mae"] - dml_eval["overall_mae"], 5),
            "rmse_delta_tlearner_minus_dml": round(tlearner_eval["overall_rmse"] - dml_eval["overall_rmse"], 5),
        },
    }

    metrics_file = out_path / "estimator_metrics_report.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics_report, f, indent=2)
    logger.info(f"Saved metrics report to {metrics_file}")

    # 4. Compare Policy Frontiers
    logger.info("Evaluating optimal allocation frontier across canonical budget grid...")
    frontier_report = compare_estimators_on_budget_frontier(
        test_df=test_df,
        dml_cate_df=dml_cate_test,
        tlearner_cate_df=tlearner_cate_test,
        ground_truth_df=gt_df,
        budgets=budgets,
    )

    frontier_file = out_path / "budget_frontier_comparison_report.json"
    with open(frontier_file, "w", encoding="utf-8") as f:
        json.dump(frontier_report, f, indent=2)
    logger.info(f"Saved frontier comparison report to {frontier_file}")

    # 5. Generate Markdown Summary
    md_file = out_path / "estimator_comparison_summary.md"
    generate_markdown_summary(metrics_report, frontier_report, md_file)
    logger.info(f"Saved Markdown summary to {md_file}")

    return metrics_report, frontier_report


def generate_markdown_summary(metrics: dict, frontier: dict, output_path: str) -> str:
    """Format benchmark metrics and budget frontier results into Markdown."""
    lines = []
    lines.append("## 1. CATE Estimation Accuracy (Held-Out Test Set)
")
    
    dml_metrics = metrics["estimators"]["causal_dml"]
    tl_metrics = metrics["estimators"]["t_learner"]
    n_eval = dml_metrics["evaluation_sample_size"]
    
    lines.append(f"- **Evaluation Sample Size**: {n_eval:,} customers")
    lines.append(f"- **Double ML Overall MAE**: {dml_metrics['overall_mae']:.4f} | RMSE: {dml_metrics['overall_rmse']:.4f}")
    lines.append(f"- **T-Learner Overall MAE**: {tl_metrics['overall_mae']:.4f} | RMSE: {tl_metrics['overall_rmse']:.4f}
")
    
    lines.append("### Per-Arm Breakdown
")
    lines.append("| Arm | True ATE | DML Est ATE | DML MAE | DML RMSE | DML r | T-Learner Est ATE | TL MAE | TL RMSE | TL r |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    
    for arm in [1, 2, 3]:
        k = f"arm_{arm}"
        d = dml_metrics["arm_metrics"][k]
        t = tl_metrics["arm_metrics"][k]
        lines.append(
            f"| {arm} | {d['true_ate']:.4f} | {d['est_ate']:.4f} | {d['mae']:.4f} | {d['rmse']:.4f} | {d['pearson_r']:.3f} | "
            f"{t['est_ate']:.4f} | {t['mae']:.4f} | {t['rmse']:.4f} | {t['pearson_r']:.3f} |"
        )
        
    lines.append("
## 2. Policy Optimization Frontier Across Budgets
")
    lines.append("| Budget | DML Pred Profit | DML Realized Profit | DML Realized Spend | TL Pred Profit | TL Realized Profit | TL Realized Spend | Profit Delta (DML - TL) |")
    lines.append("|---|---|---|---|---|---|---|---|")
    
    for pt in frontier.get("frontier_comparison", []):
        b = pt["budget"]
        dml_p = pt["causal_dml"]
        tl_p = pt["t_learner"]
        delta = pt["profit_delta_dml_minus_tlearner"]
        lines.append(
            f"| ${b:,.0f} | "
            f"${dml_p['predicted_incremental_profit']:,.2f} | "
            f"${dml_p['realized_incremental_profit']:,.2f} | "
            f"${dml_p['realized_promo_spend']:,.2f} | "
            f"${tl_p['predicted_incremental_profit']:,.2f} | "
            f"${tl_p['realized_incremental_profit']:,.2f} | "
            f"${tl_p['realized_promo_spend']:,.2f} | "
            f"${delta:,.2f} |"
        )
        
    content = "
".join(lines) + "
"
    with open(output_path, "w", encoding="utf-8") as out_f:
        out_f.write(content)
    return content

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Causal Estimators and Policy Optimization.")
    parser.add_argument("--data", default="data/synthetic_dataset.csv", help="Path to synthetic dataset")
    parser.add_argument("--ground-truth", default="data/ground_truth_evaluation.csv", help="Path to ground truth dataset")
    parser.add_argument("--output-dir", default="reports", help="Output directory for reports")
    parser.add_argument("--test-size", type=float, default=0.30, help="Holdout split ratio")
    parser.add_argument("--seed", type=int, default=20261012, help="Deterministic seed")
    args = parser.parse_args()

    run_benchmark(
        data_path=args.data,
        gt_path=args.ground_truth,
        output_dir=args.output_dir,
        test_size=args.test_size,
        random_seed=args.seed,
    )