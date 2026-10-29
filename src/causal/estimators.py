"""
Estimator module for CATE estimation using Double Machine Learning (LinearDML).
Handles multi-arm discrete treatments (0%, 5%, 10%, 20% discounts) and binary purchase outcomes.
Includes StandardScaler pipeline in nuisance model_t to ensure stable convergence.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
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

    predict_cate = estimate_effects



class TLearnerEstimator:
    """
    Multi-arm T-Learner baseline for heterogeneous treatment effect estimation.
    Fits a separate outcome model for each treatment arm (0=Control, 1, 2, 3)
    and estimates pairwise CATE relative to the control arm.
    """

    def __init__(
        self,
        treatment_col: str = DEFAULT_TREATMENT_COL,
        outcome_col: str = DEFAULT_OUTCOME_COL,
        feature_cols: Optional[List[str]] = None,
        categories: Optional[List[int]] = None,
        random_state: int = 20261012,
        n_estimators: int = 100,
        max_depth: Optional[int] = 10,
    ) -> None:
        self.treatment_col = treatment_col
        self.outcome_col = outcome_col
        self.feature_cols = feature_cols or (DEFAULT_CONFOUNDERS.copy())
        self.categories = categories or DEFAULT_CATEGORIES.copy()
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.max_depth = max_depth

        self.models: Dict[int, RandomForestClassifier] = {}
        self.is_fitted = False

    def validate_inputs(self, df: pd.DataFrame) -> None:
        # Check for quarantined ground-truth leakage
        gt_indicators = [c for c in df.columns if c.startswith('true_') or c.startswith('expected_')]
        if gt_indicators:
            raise ValueError(f'Ground-truth evaluation columns detected in training data: {gt_indicators}')

        required_cols = set(self.feature_cols) | {self.treatment_col, self.outcome_col}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise ValueError(f'Missing required columns in DataFrame: {sorted(list(missing_cols))}')

        null_counts = df[list(required_cols)].isnull().sum()
        cols_with_nulls = null_counts[null_counts > 0]
        if not cols_with_nulls.empty:
            raise ValueError(f'Null values found in required columns: {cols_with_nulls.to_dict()}')

        unique_treatments = set(df[self.treatment_col].unique())
        invalid_treatments = unique_treatments - set(self.categories)
        if invalid_treatments:
            raise ValueError(
                f'Treatment column contains invalid categories: {invalid_treatments}. Expected subset of {self.categories}'
            )

        # Ensure all expected treatment arms are present in training data
        missing_arms = set(self.categories) - unique_treatments
        if missing_arms:
            raise ValueError(
                f'Training data is missing samples for required treatment arms: {sorted(list(missing_arms))}'
            )

        unique_outcomes = set(df[self.outcome_col].unique())
        if not unique_outcomes.issubset({0, 1}):
            raise ValueError(f"Outcome column '{self.outcome_col}' must contain binary values (0 or 1). Found: {unique_outcomes}")

    def fit(self, df: pd.DataFrame) -> 'TLearnerEstimator':
        self.validate_inputs(df)

        self.models = {}
        for arm in self.categories:
            arm_mask = df[self.treatment_col] == arm
            df_arm = df[arm_mask]
            X_arm = df_arm[self.feature_cols].values
            y_arm = df_arm[self.outcome_col].values

            # Fit outcome model for this arm
            clf = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state + arm,
                n_jobs=-1,
            )
            # Handle edge case where outcome is uniform in an arm sample
            if len(np.unique(y_arm)) < 2:
                # Still fit classifier or wrapper
                clf.fit(X_arm, y_arm)
            else:
                clf.fit(X_arm, y_arm)
            self.models[arm] = clf

        self.is_fitted = True
        return self

    def predict_cate(
        self,
        df: pd.DataFrame,
        control_arm: int = 0,
        active_arms: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError('Estimator must be fitted before estimating effects.')

        missing_features = set(self.feature_cols) - set(df.columns)
        if missing_features:
            raise ValueError(f'Missing feature columns: {sorted(list(missing_features))}')

        if control_arm not in self.models:
            raise ValueError(f'Control arm {control_arm} model is not fitted.')

        X = df[self.feature_cols].values
        active_arms = active_arms or [arm for arm in self.categories if arm != control_arm]

        # Predict control probability P(Y=1 | T=control)
        ctrl_clf = self.models[control_arm]
        if hasattr(ctrl_clf, 'predict_proba') and len(ctrl_clf.classes_) > 1:
            ctrl_idx = list(ctrl_clf.classes_).index(1)
            mu_0 = ctrl_clf.predict_proba(X)[:, ctrl_idx]
        else:
            mu_0 = ctrl_clf.predict(X).astype(float)

        effects = {}
        for arm in active_arms:
            if arm not in self.models:
                raise ValueError(f'Treatment arm {arm} model is not fitted.')
            arm_clf = self.models[arm]
            if hasattr(arm_clf, 'predict_proba') and len(arm_clf.classes_) > 1:
                arm_idx = list(arm_clf.classes_).index(1)
                mu_a = arm_clf.predict_proba(X)[:, arm_idx]
            else:
                mu_a = arm_clf.predict(X).astype(float)

            col_name = f'cate_arm_{arm}_vs_{control_arm}'
            effects[col_name] = mu_a - mu_0

        return pd.DataFrame(effects, index=df.index)

    estimate_effects = predict_cate

