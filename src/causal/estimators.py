"""
Estimator module for CATE estimation using Double Machine Learning (LinearDML).
Handles multi-arm discrete treatments (0%, 5%, 10%, 20% discounts) and binary purchase outcomes.
Includes StandardScaler pipeline in nuisance model_t to ensure stable convergence.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from econml.dml import LinearDML


DEFAULT_TREATMENT_COL = "treatment_arm"
DEFAULT_OUTCOME_COL = "purchased"
DEFAULT_CATEGORIES = [0, 1, 2, 3]

DEFAULT_CONFOUNDERS: List[str] = [
    "browsing_frequency",
    "cart_abandonment_rate",
    "competitor_price_index",
    "historical_spend",
    "loyalty_member",
    "previous_discount_response_rate",
    "recency_days",
    "seasonality_index",
]

DEFAULT_EFFECT_MODIFIERS: List[str] = [
    "historical_spend",
    "recency_days",
    "browsing_frequency",
    "loyalty_member",
]


class CausalDMLEstimator:
    """
    Wrapper around EconML's LinearDML for multi-arm categorical treatment CATE estimation.
    Estimates pairwise risk differences on the probability of purchase relative to 0% control.
    """

    def __init__(
        self,
        treatment_col: str = DEFAULT_TREATMENT_COL,
        outcome_col: str = DEFAULT_OUTCOME_COL,
        confounder_cols: Optional[List[str]] = None,
        effect_modifier_cols: Optional[List[str]] = None,
        categories: Optional[List[int]] = None,
        random_state: int = 20261012,
        cv: int = 5,
    ) -> None:
        self.treatment_col = treatment_col
        self.outcome_col = outcome_col
        self.confounder_cols = confounder_cols or DEFAULT_CONFOUNDERS.copy()
        self.effect_modifier_cols = effect_modifier_cols or DEFAULT_EFFECT_MODIFIERS.copy()
        self.categories = categories or DEFAULT_CATEGORIES.copy()
        self.random_state = random_state
        self.cv = cv

        self.model_y = RandomForestRegressor(
            n_estimators=100,
            max_depth=6,
            random_state=self.random_state,
            n_jobs=-1,
        )
        # Scaled pipeline ensures stable L-BFGS gradient convergence without cross-fold leakage
        self.model_t = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=1000,
                random_state=self.random_state,
            )
        )

        self.estimator = LinearDML(
            model_y=self.model_y,
            model_t=self.model_t,
            discrete_treatment=True,
            discrete_outcome=False,
            categories=self.categories,
            cv=self.cv,
            random_state=self.random_state,
        )
        self.is_fitted = False

    def validate_inputs(self, df: pd.DataFrame) -> None:
        """Validate input DataFrame for required columns, missing values, and valid treatment categories."""
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input data must be a pandas DataFrame.")

        required_cols = set([self.treatment_col, self.outcome_col] + self.confounder_cols + self.effect_modifier_cols)
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns in DataFrame: {sorted(list(missing_cols))}")

        null_counts = df[list(required_cols)].isnull().sum()
        cols_with_nulls = null_counts[null_counts > 0]
        if not cols_with_nulls.empty:
            raise ValueError(f"Null values found in required columns: {cols_with_nulls.to_dict()}")

        unique_treatments = set(df[self.treatment_col].unique())
        invalid_treatments = unique_treatments - set(self.categories)
        if invalid_treatments:
            raise ValueError(
                f"Treatment column contains invalid categories: {invalid_treatments}. Expected subset of {self.categories}"
            )

        unique_outcomes = set(df[self.outcome_col].unique())
        if not unique_outcomes.issubset({0, 1}):
            raise ValueError(f"Outcome column '{self.outcome_col}' must contain binary values (0 or 1). Found: {unique_outcomes}")

    def fit(self, df: pd.DataFrame) -> "CausalDMLEstimator":
        """
        Fit the LinearDML estimator on training data.
        Does NOT accept ground-truth evaluation datasets.
        """
        self.validate_inputs(df)

        Y = df[self.outcome_col].values
        T = df[self.treatment_col].values
        X = df[self.effect_modifier_cols].values
        W = df[self.confounder_cols].values

        self.estimator.fit(Y, T, X=X, W=W)
        self.is_fitted = True
        return self

    def estimate_effects(
        self,
        df: pd.DataFrame,
        control_arm: int = 0,
        active_arms: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """
        Estimate treatment effects (CATE) for each active discount arm relative to the control arm.
        Returns a DataFrame of predicted incremental purchase probability lifts.
        """
        if not self.is_fitted:
            raise RuntimeError("Estimator must be fitted before estimating effects.")

        missing_features = set(self.effect_modifier_cols) - set(df.columns)
        if missing_features:
            raise ValueError(f"Missing effect modifier columns: {sorted(list(missing_features))}")

        X = df[self.effect_modifier_cols].values
        active_arms = active_arms or [arm for arm in self.categories if arm != control_arm]

        effects = {}
        for arm in active_arms:
            col_name = f"cate_arm_{arm}_vs_{control_arm}"
            effects[col_name] = self.estimator.effect(X, T0=control_arm, T1=arm)

        return pd.DataFrame(effects, index=df.index)
