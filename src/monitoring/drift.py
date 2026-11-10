"""
Covariate and Data Drift Detection Engine for EconoCausal.
Monitors the 8 causal confounder features against reference baseline distributions.
"""

from typing import Dict, List, Any, Optional
import math
import numpy as np

CONFOUNDER_NUMERICAL_FEATURES = [
    "browsing_frequency",
    "cart_abandonment_rate",
    "competitor_price_index",
    "historical_spend",
    "previous_discount_response_rate",
    "recency_days",
    "seasonality_index",
]

CONFOUNDER_CATEGORICAL_FEATURES = [
    "loyalty_member",
]

ALL_CONFOUNDER_FEATURES = CONFOUNDER_NUMERICAL_FEATURES + CONFOUNDER_CATEGORICAL_FEATURES


def calculate_numerical_drift(
    ref_vals: List[float],
    curr_vals: List[float],
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Evaluates drift in a continuous numerical feature using the two-sample
    Kolmogorov-Smirnov (KS) test.
    """
    clean_ref = [float(x) for x in ref_vals if x is not None and not (isinstance(x, float) and math.isnan(x))]
    clean_curr = [float(x) for x in curr_vals if x is not None and not (isinstance(x, float) and math.isnan(x))]

    n_ref = len(clean_ref)
    n_curr = len(clean_curr)

    if n_ref < 5 or n_curr < 5:
        return {
            "test_type": "kolmogorov_smirnov",
            "statistic": 0.0,
            "p_value": 1.0,
            "drift_detected": False,
            "sample_size_reference": n_ref,
            "sample_size_current": n_curr,
            "warning": "Insufficient sample size (minimum 5 required per distribution).",
        }

    # Lazy import scipy.stats to guard against memory overhead during app boot
    from scipy.stats import ks_2samp

    res = ks_2samp(clean_ref, clean_curr)
    stat = float(res.statistic)
    p_val = float(res.pvalue)
    drift = bool(p_val < alpha)

    return {
        "test_type": "kolmogorov_smirnov",
        "statistic": round(stat, 5),
        "p_value": round(p_val, 5),
        "drift_detected": drift,
        "sample_size_reference": n_ref,
        "sample_size_current": n_curr,
        "warning": None,
    }


def calculate_categorical_drift(
    ref_vals: List[Any],
    curr_vals: List[Any],
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Evaluates drift in a discrete/categorical feature using the Chi-Square test of independence.
    Falls back gracefully when categories have zero occurrences.
    """
    clean_ref = [x for x in ref_vals if x is not None]
    clean_curr = [x for x in curr_vals if x is not None]

    n_ref = len(clean_ref)
    n_curr = len(clean_curr)

    if n_ref < 5 or n_curr < 5:
        return {
            "test_type": "chi_square",
            "statistic": 0.0,
            "p_value": 1.0,
            "drift_detected": False,
            "sample_size_reference": n_ref,
            "sample_size_current": n_curr,
            "warning": "Insufficient sample size (minimum 5 required per distribution).",
        }

    categories = sorted(list(set(clean_ref) | set(clean_curr)))
    ref_counts = [clean_ref.count(cat) for cat in categories]
    curr_counts = [clean_curr.count(cat) for cat in categories]

    # Contingency table: shape (2, len(categories))
    contingency = np.array([ref_counts, curr_counts])

    # If all values are concentrated in a single category in both sets, distribution is identical
    if len(categories) < 2 or np.all(contingency.sum(axis=0) == 0):
        return {
            "test_type": "chi_square",
            "statistic": 0.0,
            "p_value": 1.0,
            "drift_detected": False,
            "sample_size_reference": n_ref,
            "sample_size_current": n_curr,
            "warning": "Single category distribution observed.",
        }

    from scipy.stats import chi2_contingency

    try:
        chi2, p_val, _, _ = chi2_contingency(contingency)
        stat = float(chi2)
        p_val = float(p_val)
        drift = bool(p_val < alpha)
        warning = None
    except Exception as e:
        stat = 0.0
        p_val = 1.0
        drift = False
        warning = f"Chi-square calculation error: {str(e)}"

    return {
        "test_type": "chi_square",
        "statistic": round(stat, 5),
        "p_value": round(p_val, 5),
        "drift_detected": drift,
        "sample_size_reference": n_ref,
        "sample_size_current": n_curr,
        "warning": warning,
    }


def detect_dataset_drift(
    reference_records: List[Dict[str, Any]],
    current_records: List[Dict[str, Any]],
    alpha: float = 0.05,
    drift_share_threshold: float = 0.3,
) -> Dict[str, Any]:
    """
    Monitors all 8 causal confounders across reference and current datasets.
    """
    if not reference_records or not current_records:
        raise ValueError("Both reference and current customer record datasets must be non-empty.")

    feature_metrics: Dict[str, Any] = {}
    drifted_count = 0
    evaluated_count = 0

    # Continuous confounders
    for feat in CONFOUNDER_NUMERICAL_FEATURES:
        ref_vals = [r.get(feat) for r in reference_records if feat in r and r.get(feat) is not None]
        curr_vals = [r.get(feat) for r in current_records if feat in r and r.get(feat) is not None]
        metrics = calculate_numerical_drift(ref_vals, curr_vals, alpha=alpha)
        feature_metrics[feat] = metrics
        evaluated_count += 1
        if metrics["drift_detected"]:
            drifted_count += 1

    # Categorical confounders
    for feat in CONFOUNDER_CATEGORICAL_FEATURES:
        ref_vals = [r.get(feat) for r in reference_records if feat in r and r.get(feat) is not None]
        curr_vals = [r.get(feat) for r in current_records if feat in r and r.get(feat) is not None]
        metrics = calculate_categorical_drift(ref_vals, curr_vals, alpha=alpha)
        feature_metrics[feat] = metrics
        evaluated_count += 1
        if metrics["drift_detected"]:
            drifted_count += 1

    drift_share = (drifted_count / evaluated_count) if evaluated_count > 0 else 0.0
    overall_drift_detected = drift_share >= drift_share_threshold

    return {
        "overall_drift_detected": overall_drift_detected,
        "drift_share": round(drift_share, 4),
        "drift_share_threshold": drift_share_threshold,
        "alpha_significance_level": alpha,
        "total_features_evaluated": evaluated_count,
        "drifted_features_count": drifted_count,
        "feature_metrics": feature_metrics,
        "summary": (
            f"Dataset drift DETECTED ({drifted_count}/{evaluated_count} features drifted, "
            f"share={drift_share:.1%} >= {drift_share_threshold:.1%})"
            if overall_drift_detected
            else f"Dataset distributions STABLE ({drifted_count}/{evaluated_count} features drifted, "
            f"share={drift_share:.1%} < {drift_share_threshold:.1%})"
        ),
    }
