"""
Causal estimation singleton service for EconoCausal API.
Strictly isolates observational data from ground-truth evaluation datasets.
"""
from pathlib import Path
from typing import Optional, List, Dict, Any
import pandas as pd

from src.causal.estimators import CausalDMLEstimator
from src.causal.policy import assign_discount_policy

TRAIN_DATA_PATH = Path("data/raw/econocausal_synthetic_dataset.csv")


class CausalService:
    def __init__(self, estimator: Optional[CausalDMLEstimator] = None):
        self._estimator = estimator
        self._is_ready = estimator is not None

    def initialize(self, sample_size: int = 2000, random_state: int = 20261012):
        if self._estimator is not None:
            return
        if not TRAIN_DATA_PATH.exists():
            raise FileNotFoundError(f"Training dataset not found at {TRAIN_DATA_PATH}")

        raw_df = pd.read_csv(TRAIN_DATA_PATH)
        if len(raw_df) > sample_size:
            train_df = raw_df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)
        else:
            train_df = raw_df.copy()

        estimator = CausalDMLEstimator(random_state=random_state, cv=2)
        estimator.fit(train_df)
        self._estimator = estimator
        self._is_ready = True

    @property
    def is_ready(self) -> bool:
        return self._is_ready and self._estimator is not None

    def estimate_cate(self, records: List[Dict[str, Any]]) -> pd.DataFrame:
        if not self.is_ready:
            self.initialize()
        df = pd.DataFrame(records)
        return self._estimator.estimate_effects(df)

    def predict_policy(self, records: List[Dict[str, Any]]) -> pd.DataFrame:
        if not self.is_ready:
            self.initialize()
        df = pd.DataFrame(records)
        cate_df = self._estimator.estimate_effects(df)

        econ_df = pd.DataFrame(index=df.index)
        econ_df["historical_spend"] = df["historical_spend"]
        bv_col = "basket_value_before_discount" if "basket_value_before_discount" in df.columns else ("basket_value" if "basket_value" in df.columns else None)
        bv_vals = df[bv_col] if bv_col is not None else 100.0
        econ_df["basket_value_before_discount"] = bv_vals
        econ_df["basket_value"] = bv_vals
        econ_df["cost_of_goods"] = (
            df["cost_of_goods"] if "cost_of_goods" in df.columns else bv_vals * 0.70
        )
        econ_df["baseline_purchase_prob"] = (
            df["baseline_purchase_prob"] if "baseline_purchase_prob" in df.columns else 0.35
        )

        policy_df = assign_discount_policy(
            cate_df=cate_df,
            econ_df=econ_df,
            spend_tail_threshold=60000.0,
            min_profit_threshold=0.0,
            default_baseline_prob=0.35,
        )
        return pd.concat(
            [cate_df, policy_df[["recommended_arm", "expected_incremental_profit"]]],
            axis=1,
        )


causal_service = CausalService()

