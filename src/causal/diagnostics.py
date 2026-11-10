"""
Diagnostics module for causal inference: treatment overlap and CATE reliability auditing.
Assesses generalized propensity score overlap and identifies extrapolation risk.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from src.causal.estimators import DEFAULT_CONFOUNDERS, DEFAULT_EFFECT_MODIFIERS


def compute_treatment_overlap(
    df: pd.DataFrame,
    treatment_col: str = "treatment_arm",
    confounder_cols: Optional[List[str]] = None,
    n_splits: int = 5,
    random_state: int = 20261012,
    overlap_threshold: float = 0.05,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Computes out-of-fold generalized propensity scores across discrete treatment arms
    using cross-fitting to prevent data leakage. Identifies overlap violations.
    """
    confounder_cols = confounder_cols or DEFAULT_CONFOUNDERS.copy()
    required = [treatment_col] + confounder_cols
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for overlap audit: {sorted(list(missing))}")

    W = df[confounder_cols].values
    T = df[treatment_col].values
    categories = sorted(list(np.unique(T)))

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_propensities = np.zeros((len(df), len(categories)))

    for train_idx, val_idx in skf.split(W, T):
        pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=random_state))
        pipe.fit(W[train_idx], T[train_idx])
        oof_propensities[val_idx] = pipe.predict_proba(W[val_idx])

    prop_col_names = [f"propensity_arm_{c}" for c in categories]
    props_df = pd.DataFrame(oof_propensities, columns=prop_col_names, index=df.index)

    min_propensity = props_df.min(axis=1)
    props_df["min_propensity"] = min_propensity
    props_df["overlap_violation"] = min_propensity < overlap_threshold

    summary = {
        "n_total": float(len(df)),
        "n_overlap_violations": float((min_propensity < overlap_threshold).sum()),
        "violation_rate": float((min_propensity < overlap_threshold).mean()),
        "global_min_propensity": float(min_propensity.min()),
        "global_max_propensity": float(props_df[prop_col_names].max().max()),
    }

    return props_df, summary


def diagnose_cate_reliability(
    df: pd.DataFrame,
    cate_df: pd.DataFrame,
    overlap_df: pd.DataFrame,
    target_arm: int = 3,
    extreme_threshold: float = -0.2,
) -> Dict[str, pd.DataFrame]:
    """
    Examines relationships between extreme CATE predictions, feature support tails,
    and propensity score overlap.
    """
    cate_col = f"cate_arm_{target_arm}_vs_0"
    if cate_col not in cate_df.columns:
        raise KeyError(f"Expected CATE column '{cate_col}' not found in cate_df.")

    merged = pd.concat([df, cate_df[[cate_col]], overlap_df], axis=1)

    # 1. Extreme vs Normal Spenders Comparison
    merged["is_extreme_negative"] = merged[cate_col] < extreme_threshold
    group_spend = merged.groupby("is_extreme_negative")["historical_spend"].agg(
        ["count", "mean", "std", "min", "median", "max"]
    )

    # 2. Overlap vs Extreme CATE cross-tabulation
    crosstab_overlap = pd.crosstab(
        merged["overlap_violation"],
        merged["is_extreme_negative"],
        rownames=["overlap_violation (<0.05)"],
        colnames=["extreme_negative (< -0.2)"],
    )

    return {
        "spend_comparison": group_spend,
        "overlap_crosstab": crosstab_overlap,
    }
