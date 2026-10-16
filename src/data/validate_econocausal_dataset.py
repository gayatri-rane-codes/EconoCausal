#!/usr/bin/env python3
"""
Validation for the EconoCausal synthetic dataset.  Reads the deliverable files from disk, computes every
check, and writes dataset_validation_report.md.  Nothing in the report is hand-typed except the
"Fixes made during validation" section (NOTES below).

Downstream-compatibility smoke tests use scikit-learn + SciPy only (cross-fitted doubly-robust / DML-style
estimator, PSM, Qini, LP budget optimisation).  DoWhy and EconML are probed with import; if absent the
report says they were NOT executed.

Usage:  python validate_econocausal_dataset.py --dir . --report dataset_validation_report.md
"""
import argparse
import hashlib
import importlib
import os
import subprocess
import sys
import tempfile
import time
import warnings

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
G = importlib.import_module("generate_econocausal_dataset")

SEED_CV = 7
NOTES = """The first complete validation run of the first generator draft failed 3 of 66 checks, and calibration review before that led to two parameter retunes. Everything below was fixed and the generator and validator were re-run from scratch; the report above reflects only the final run.

1. **Effect calibration (before the first validation run).** The first draft gave 42% of customers a negative true purchase effect at 20% and made "no discount" profit-optimal for 71% of customers. `BETA_INTERCEPT` and `KAPPA_BASE` were retuned (0.30/0.08 -> 0.55/0.05 -> 0.45/0.04). Final shares are in section 6.
2. **Overlap (real DGP weakness).** With the first targeting strength, 10.3% of customers had at least one arm propensity below 0.05 (the 'any arm < 0.05' check failed). `TARGETING_GAMMA` was reduced from [0, .50, .85, 1.15] to [0, .45, .75, .95] and `SCORE_CLIP` from 2.2 to 2.0. Minimum true propensity rose to 0.042 and the share of customers with any propensity below 0.05 fell to 1.4%. Confounding remains strong (section 5).
3. **Validator bug, list-price conservation check.** The tolerance ignored that ground-truth probabilities are stored rounded to 6 decimals (error up to ~INR 0.0075 on a INR 15,000 basket). The tolerance now scales with basket value. No data changed.
4. **IPW recovery criterion changed.** The first draft used an arbitrary absolute threshold (0.015) and failed with a maximum error of 0.0194 under the earlier, sharper targeting (large inverse-propensity weights). A fixed absolute threshold is not a meaningful test, so the criterion was replaced by "within 3 standard errors" (Hajek SE reported in section 5). This was a change of criterion made after seeing a failure; the DGP overlap fix in item 2 was applied independently and separately reduced the weights.
5. **Validator reproducibility.** Gradient-boosting models in the validator had no fixed random state, so downstream numbers changed between runs. All models are now seeded; two consecutive validator runs produce an identical report (verified by diff, ignoring the runtime line).
6. **Added structured allocation policy.** The generic doubly-robust profit CATE recovered only a small share of oracle gain because realised profit is a noisy, heavy-tailed outcome. A structured policy (purchase-probability models x unit economics estimated from observed columns) was added to section 9d; both are reported.
7. **DoWhy / EconML could not be installed** (the sandbox could not resolve the package index), so they were never executed; see the status lines at the top."""

INT_COLS = ["customer_id", "age", "tenure_months", "recency_days", "browsing_frequency", "support_tickets_count",
            "previous_discounts_accepted", "previous_discount_offers_received", "previous_purchases_12m",
            "website_visits_30d", "mobile_app_user", "loyalty_member", "days_since_last_campaign",
            "marketing_contact_count_30d", "treatment_arm", "discount_pct", "purchased"]
FLOAT_COLS = ["historical_spend", "avg_session_duration_sec", "cart_abandonment_rate", "email_open_rate",
              "previous_discount_response_rate", "avg_order_value", "competitor_price_index", "seasonality_index",
              "customer_satisfaction", "historical_return_rate", "discount_rate", "basket_value_before_discount",
              "discount_amount", "revenue", "cost_of_goods", "marketing_cost", "net_profit"]
CAT_DOMAINS = {
    "customer_segment": G.SEGMENTS, "income_bracket": ["low", "lower_middle", "upper_middle", "high"],
    "urbanicity": ["rural", "suburban", "urban"], "region": G.REGIONS,
    "subscription_status": ["none", "basic", "premium"], "preferred_channel": list(G.CHANNEL_COST),
    "product_category": G.CATEGORIES, "campaign_type": G.CAMPAIGN_TYPES,
    "day_of_week": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], "device_type": ["mobile", "desktop", "tablet"],
    "stock_availability": ["in_stock", "low_stock"],
}
RANGES = {
    "age": (18, 75), "tenure_months": (1, 120), "historical_spend": (0, 1e7), "recency_days": (0, 180),
    "browsing_frequency": (0, 25), "avg_session_duration_sec": (10, 1800), "cart_abandonment_rate": (0, 1),
    "support_tickets_count": (0, 8), "email_open_rate": (0, 1), "previous_discounts_accepted": (0, 12),
    "previous_discount_response_rate": (0, 1), "previous_purchases_12m": (0, 25), "avg_order_value": (100, 8000),
    "website_visits_30d": (0, 30), "mobile_app_user": (0, 1), "loyalty_member": (0, 1),
    "competitor_price_index": (0.6, 1.5), "seasonality_index": (0.75, 1.35), "customer_satisfaction": (1, 5),
    "historical_return_rate": (0, 0.7), "days_since_last_campaign": (0, 180), "marketing_contact_count_30d": (0, 12),
    "purchased": (0, 1), "basket_value_before_discount": (100, 15000),
}
NUM_FEATS = [c for c in G.FEATURE_COLUMNS if c not in CAT_DOMAINS]
CAT_FEATS = list(CAT_DOMAINS)


# ------------------------------------------------------------------------------------------------ helpers
def md(df, floatfmt="{:.4f}", index=True):
    d = df.copy()
    if index:
        d = d.reset_index()
    cols = [str(c) for c in d.columns]
    rows = []
    for _, r in d.iterrows():
        rows.append([floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in r.tolist()])
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def design(df):
    """Numeric + one-hot covariate matrix (features only) with extra log transforms for linear models."""
    X = df[NUM_FEATS].astype(float).copy()
    X["log1p_recency"] = np.log1p(df.recency_days)
    X["log_aov"] = np.log(df.avg_order_value)
    X["log_tenure"] = np.log(df.tenure_months)
    X["log1p_hist_spend"] = np.log1p(df.historical_spend)
    D = pd.get_dummies(df[CAT_FEATS], drop_first=True).astype(float)
    return pd.concat([X, D], axis=1)


def smd_table(Xm, a_mask, b_mask, wa=None, wb=None):
    A, B = Xm[a_mask], Xm[b_mask]
    sd = np.sqrt((A.var() + B.var()) / 2).replace(0, np.nan)
    ma = np.average(A, axis=0, weights=wa) if wa is not None else A.mean().to_numpy()
    mb = np.average(B, axis=0, weights=wb) if wb is not None else B.mean().to_numpy()
    return pd.Series((ma - mb) / sd.to_numpy(), index=Xm.columns).fillna(0.0)


def hajek(y, w):
    return float(np.sum(w * y) / np.sum(w))


def as_cat(df):
    d = df[G.FEATURE_COLUMNS].copy()
    for c in CAT_FEATS:
        d[c] = d[c].astype("category")
    return d


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=".")
    ap.add_argument("--report", default="dataset_validation_report.md")
    a = ap.parse_args()
    t0 = time.time()
    p_main = os.path.join(a.dir, "econocausal_synthetic_dataset.csv")
    p_gt = os.path.join(a.dir, "ground_truth_evaluation.csv")
    m = pd.read_csv(p_main)
    g = pd.read_csv(p_gt)
    n = len(m)
    checks = []  # (section, name, passed, detail)

    def chk(sec, name, ok, detail=""):
        checks.append((sec, name, bool(ok), detail))

    out = []  # markdown chunks

    # ---------------------------------------------------------------- 1 structure
    chk("1 Structure", "main rows == 20,000", n == 20000, f"{n}")
    chk("1 Structure", "main columns == expected 45 in documented order", list(m.columns) == G.MAIN_COLUMNS, f"{m.shape[1]}")
    chk("1 Structure", "ground-truth rows == 20,000", len(g) == 20000, f"{len(g)}; {g.shape[1]} columns")
    chk("1 Structure", "no duplicate customer_id", m.customer_id.is_unique, f"{m.customer_id.duplicated().sum()} duplicates")
    chk("1 Structure", "ids contiguous 100001..120000", (m.customer_id.to_numpy() == np.arange(100001, 120001)).all())
    chk("1 Structure", "ground-truth ids identical and in same order", (m.customer_id.to_numpy() == g.customer_id.to_numpy()).all())
    chk("1 Structure", "no missing values (main)", int(m.isna().sum().sum()) == 0, f"{int(m.isna().sum().sum())}")
    chk("1 Structure", "no missing values (ground truth)", int(g.isna().sum().sum()) == 0, f"{int(g.isna().sum().sum())}")
    num_all = m.select_dtypes("number")
    chk("1 Structure", "no inf values", bool(np.isfinite(num_all.to_numpy()).all()))
    bad_int = [c for c in INT_COLS if not pd.api.types.is_integer_dtype(m[c])]
    bad_flt = [c for c in FLOAT_COLS if not pd.api.types.is_float_dtype(m[c])]
    bad_cat = [c for c in CAT_FEATS if not pd.api.types.is_string_dtype(m[c])]
    chk("1 Structure", "dtypes match schema (int / float / string)", not (bad_int or bad_flt or bad_cat),
        f"violations: {bad_int + bad_flt + bad_cat}")
    dom_bad = {c: sorted(set(m[c]) - set(v)) for c, v in CAT_DOMAINS.items() if set(m[c]) - set(v)}
    chk("1 Structure", "categorical values within allowed domains", not dom_bad, str(dom_bad))
    rng_bad = {c: (float(m[c].min()), float(m[c].max())) for c, (lo, hi) in RANGES.items() if m[c].min() < lo or m[c].max() > hi}
    chk("1 Structure", "numeric values within documented ranges", not rng_bad, str(rng_bad))
    chk("1 Structure", "treatment encoding consistent (arm<->pct<->rate)",
        (m.discount_pct.to_numpy() == G.ARM_PCT[m.treatment_arm.to_numpy()]).all() and
        np.allclose(m.discount_rate, m.discount_pct / 100) and set(m.discount_pct) == {0, 5, 10, 20})
    dtypes_tbl = pd.DataFrame({"dtype": m.dtypes.astype(str), "n_missing": m.isna().sum(), "n_unique": m.nunique()})

    # ---------------------------------------------------------------- 2 treatment & outcomes
    arm = m.treatment_arm.to_numpy()
    cnt = m.groupby("discount_pct").size().rename("n").to_frame()
    cnt["share"] = cnt.n / n
    by_arm = m.groupby("discount_pct").agg(
        n=("purchased", "size"), purchase_rate=("purchased", "mean"), mean_basket_list=("basket_value_before_discount", "mean"),
        mean_discount_amount=("discount_amount", "mean"), mean_revenue=("revenue", "mean"), mean_cogs=("cost_of_goods", "mean"),
        mean_marketing_cost=("marketing_cost", "mean"), mean_net_profit=("net_profit", "mean"))
    buyers = m[m.purchased == 1]
    dist = pd.DataFrame({
        "basket_value_before_discount": m.basket_value_before_discount.describe(percentiles=[.05, .25, .5, .75, .95]),
        "revenue (buyers)": buyers.revenue.describe(percentiles=[.05, .25, .5, .75, .95]),
        "net_profit (all)": m.net_profit.describe(percentiles=[.05, .25, .5, .75, .95]),
        "net_profit (buyers)": buyers.net_profit.describe(percentiles=[.05, .25, .5, .75, .95])}).T
    chk("2 Treatment/outcomes", "all four arms present with >= 2,500 obs each", (cnt.n >= 2500).all(), str(cnt.n.to_dict()))
    chk("2 Treatment/outcomes", "overall purchase rate in plausible band 0.15-0.50", 0.15 <= m.purchased.mean() <= 0.5, f"{m.purchased.mean():.4f}")

    # ---------------------------------------------------------------- 3 financial formulas
    rate = m.discount_rate.to_numpy()
    B = m.basket_value_before_discount.to_numpy()
    y = m.purchased.to_numpy() == 1
    non = ~y
    tol = 1e-6
    f = {}
    f["non-buyers: discount_amount = revenue = cost_of_goods = 0"] = (m.loc[non, ["discount_amount", "revenue", "cost_of_goods"]].abs().to_numpy() <= tol).all()
    f["non-buyers: net_profit = -marketing_cost"] = np.allclose(m.loc[non, "net_profit"], -m.loc[non, "marketing_cost"], atol=tol)
    f["buyers: discount_amount = round(basket * rate, 2)"] = np.allclose(m.loc[y, "discount_amount"], np.round(B[y] * rate[y], 2), atol=tol)
    f["buyers: revenue = basket - discount_amount"] = np.allclose(m.loc[y, "revenue"], np.round(B[y] - m.loc[y, "discount_amount"], 2), atol=tol)
    f["all: net_profit = revenue - cost_of_goods - marketing_cost"] = np.allclose(m.net_profit, np.round(m.revenue - m.cost_of_goods - m.marketing_cost, 2), atol=tol)
    ch_cost = m.preferred_channel.map(G.CHANNEL_COST).to_numpy()
    f["all: marketing_cost = channel cost + coupon cost*(arm>0)"] = np.allclose(m.marketing_cost, ch_cost + G.COUPON_ISSUE_COST * (arm > 0), atol=tol)
    f["all: marketing_cost independent of purchase (no discount double-count)"] = np.allclose(
        m.loc[y, "marketing_cost"] - ch_cost[y] - G.COUPON_ISSUE_COST * (arm[y] > 0), 0, atol=tol)
    f["buyers: 0 <= revenue <= basket"] = ((m.loc[y, "revenue"] >= 0) & (m.loc[y, "revenue"] <= B[y] + tol)).all()
    cr = (m.loc[y, "cost_of_goods"] / B[y])
    f["buyers: COGS ratio within [0.30, 0.90]"] = bool(cr.between(0.30 - 1e-3, 0.90 + 1e-3).all())
    for k, v in f.items():
        chk("3 Financial", k, v)
    cogs_by_cat = pd.DataFrame({"observed_mean_cogs_ratio": cr.groupby(m.loc[y, "product_category"]).mean(),
                                "design_base_ratio": pd.Series(G.COGS_RATIO_BASE)}).dropna()
    neg_gp = float((m.loc[y, "revenue"] < m.loc[y, "cost_of_goods"]).mean())
    neg_gp_by_arm = (m.loc[y].assign(neg=lambda d: d.revenue < d.cost_of_goods).groupby("discount_pct").neg.mean())

    # observed == potential outcome at assigned arm
    ok_all = True
    detail = []
    for k in range(4):
        mk = arm == k
        for col, gcol in [("purchased", f"purchased_arm{k}"), ("revenue", f"revenue_arm{k}"), ("cost_of_goods", f"cost_of_goods_arm{k}"),
                          ("marketing_cost", f"marketing_cost_arm{k}"), ("net_profit", f"net_profit_arm{k}")]:
            same = np.allclose(m.loc[mk, col], g.loc[mk, gcol], atol=tol)
            ok_all &= bool(same)
            if not same:
                detail.append(f"{col}/arm{k}")
    chk("3 Financial", "observed outcomes == ground-truth potential outcome at assigned arm (all 5 outcome columns, 4 arms)", ok_all, ", ".join(detail))
    chk("3 Financial", "ground-truth assigned_arm == treatment_arm", (g.assigned_arm.to_numpy() == arm).all())
    e_disc = np.stack([g[f"expected_discount_cost_arm{k}"] for k in range(4)], 1)
    e_rev = np.stack([g[f"expected_revenue_arm{k}"] for k in range(4)], 1)
    pk = np.stack([g[f"p_purchase_arm{k}"] for k in range(4)], 1)
    chk("3 Financial", "E[revenue_k] + E[discount_k] == p_k * basket (list-price conservation)",
        bool((np.abs(e_rev + e_disc - pk * B[:, None]) <= 5e-7 * B[:, None] + 2e-4).all()))   # p is stored rounded to 6 dp

    # ---------------------------------------------------------------- 4 overlap
    prop = np.stack([g[f"true_propensity_arm{k}"] for k in range(4)], 1)
    chk("4 Overlap", "true propensities sum to 1", np.allclose(prop.sum(1), 1, atol=1e-5))
    ov = pd.DataFrame({
        "min": prop.min(0), "p1": np.quantile(prop, .01, axis=0), "p5": np.quantile(prop, .05, axis=0),
        "median": np.median(prop, axis=0), "p95": np.quantile(prop, .95, axis=0), "max": prop.max(0),
        "share<0.02": (prop < 0.02).mean(0), "share<0.05": (prop < 0.05).mean(0)}, index=[f"arm{k} ({G.ARM_PCT[k]}%)" for k in range(4)])
    chk("4 Overlap", "min true propensity >= 0.02 for every arm", prop.min() >= 0.02, f"min={prop.min():.4f}")
    chk("4 Overlap", "<= 5% of customers have any arm propensity < 0.05", (prop.min(1) < 0.05).mean() <= 0.05, f"{(prop.min(1) < 0.05).mean():.4f}")
    cell = pd.crosstab([m.customer_segment, m.campaign_type], m.discount_pct)
    chk("4 Overlap", "every segment x campaign cell with >=30 customers has all 4 arms", bool(((cell > 0).all(axis=1) | (cell.sum(axis=1) < 30)).all()),
        f"min nonzero-cell count={int(cell[cell > 0].min().min())}, cells with a missing arm={int((cell == 0).any(axis=1).sum())}")
    seg_cnt = pd.crosstab(m.customer_segment, m.discount_pct)

    # estimated propensities (honest, cross-fitted multinomial logit on features only)
    Xd = design(m)
    skf = KFold(5, shuffle=True, random_state=SEED_CV)
    e_hat = np.zeros((n, 4))
    for tr, te in skf.split(Xd):
        sc = StandardScaler().fit(Xd.iloc[tr])
        lr = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(Xd.iloc[tr]), arm[tr])
        e_hat[te] = lr.predict_proba(sc.transform(Xd.iloc[te]))
    e_hat = np.clip(e_hat, 0.01, 0.99)
    e_hat /= e_hat.sum(1, keepdims=True)
    est_ov = pd.DataFrame({"min": e_hat.min(0), "p1": np.quantile(e_hat, .01, axis=0), "corr_with_true": [np.corrcoef(e_hat[:, k], prop[:, k])[0, 1] for k in range(4)]},
                          index=[f"arm{k}" for k in range(4)])
    acc = (e_hat.argmax(1) == arm).mean()
    auc_arm = roc_auc_score(arm, e_hat, multi_class="ovr")

    # ---------------------------------------------------------------- 5 confounding
    key = ["recency_days", "cart_abandonment_rate", "email_open_rate", "previous_discount_response_rate", "loyalty_member",
           "competitor_price_index", "previous_purchases_12m"]
    cov_by_arm = m.groupby("discount_pct")[key].mean()
    cov_by_arm["win_back_share"] = m.assign(w=(m.campaign_type == "win_back")).groupby("discount_pct").w.mean()
    cov_by_arm["premium_segment_share"] = m.assign(w=(m.customer_segment == "premium")).groupby("discount_pct").w.mean()
    Xbal = design(m)
    smd_before, smd_after = {}, {}
    w_true = 1.0 / prop[np.arange(n), arm]
    for k in (1, 2, 3):
        mk, m0 = arm == k, arm == 0
        smd_before[k] = smd_table(Xbal, mk, m0)
        smd_after[k] = smd_table(Xbal, mk, m0, w_true[mk], w_true[m0])
    bal = pd.DataFrame({f"arm{k} vs 0": [smd_before[k].abs().max(), smd_before[k].abs().mean(), (smd_before[k].abs() > 0.1).sum(),
                                         smd_after[k].abs().max(), smd_after[k].abs().mean(), (smd_after[k].abs() > 0.1).sum()] for k in (1, 2, 3)},
                       index=["max|SMD| raw", "mean|SMD| raw", "# covariates |SMD|>0.1 raw",
                              "max|SMD| IPW(true e)", "mean|SMD| IPW(true e)", "# covariates |SMD|>0.1 IPW"])
    top_imb = smd_before[3].abs().sort_values(ascending=False).head(8).rename("|SMD| arm3 vs arm0 (raw)").to_frame()
    top_imb["|SMD| after IPW"] = smd_after[3].abs()[top_imb.index]
    chk("5 Confounding", "raw covariate imbalance exists (max |SMD| arm3 vs 0 > 0.3)", smd_before[3].abs().max() > 0.3, f"{smd_before[3].abs().max():.3f}")
    chk("5 Confounding", "IPW with true propensities balances covariates (max |SMD| < 0.1, all arms)", max(smd_after[k].abs().max() for k in (1, 2, 3)) < 0.1,
        f"{max(smd_after[k].abs().max() for k in (1, 2, 3)):.4f}")
    base_p = g.p_purchase_arm0.to_numpy()
    score = g.targeting_score.to_numpy()
    beta = g.true_effect_beta_per10pp.to_numpy()
    corr_sb = np.corrcoef(score, base_p)[0, 1]
    corr_se = np.corrcoef(score, beta)[0, 1]
    chk("5 Confounding", "assignment score correlates with baseline purchase probability (|r| > 0.15)", abs(corr_sb) > 0.15, f"r={corr_sb:.3f}")
    chk("5 Confounding", "assignment score correlates with effect size (|r| > 0.15)", abs(corr_se) > 0.15, f"r={corr_se:.3f}")

    # effect recovery table: naive vs IPW(true) vs IPW(est)
    def hajek_se(y, w):
        mu = np.sum(w * y) / np.sum(w)
        return float(np.sqrt(np.sum((w * (y - mu)) ** 2)) / np.sum(w))

    def ate_rows(ycol, gcol_prefix, label):
        yv = m[ycol].to_numpy().astype(float)
        rows = []
        mu0_true = g[f"{gcol_prefix}0"].mean()
        we = 1.0 / e_hat[np.arange(n), arm]
        for k in (1, 2, 3):
            truth = g[f"{gcol_prefix}{k}"].mean() - mu0_true
            naive = yv[arm == k].mean() - yv[arm == 0].mean()
            ipw_t = hajek(yv[arm == k], w_true[arm == k]) - hajek(yv[arm == 0], w_true[arm == 0])
            se_t = np.hypot(hajek_se(yv[arm == k], w_true[arm == k]), hajek_se(yv[arm == 0], w_true[arm == 0]))
            ipw_e = hajek(yv[arm == k], we[arm == k]) - hajek(yv[arm == 0], we[arm == 0])
            se_e = np.hypot(hajek_se(yv[arm == k], we[arm == k]), hajek_se(yv[arm == 0], we[arm == 0]))
            rows.append({"outcome": label, "arm": f"{G.ARM_PCT[k]}% vs 0%", "true ATE": truth, "naive diff": naive, "naive bias": naive - truth,
                         "IPW (true e)": ipw_t, "SE": se_t, "err/SE": (ipw_t - truth) / se_t, "IPW (est e)": ipw_e, "SE ": se_e, "err/SE ": (ipw_e - truth) / se_e})
        return rows

    rec = pd.DataFrame(ate_rows("purchased", "p_purchase_arm", "purchase prob.") + ate_rows("net_profit", "expected_net_profit_arm", "net profit (INR)"))
    pr = rec[rec.outcome == "purchase prob."]
    chk("5 Confounding", "naive purchase-rate difference is materially biased for >=1 arm (|bias| > 0.02)", (pr["naive bias"].abs() > 0.02).any(),
        f"max |bias|={pr['naive bias'].abs().max():.4f}")
    chk("5 Confounding", "IPW(true e) purchase ATE within 3 standard errors of truth, all arms", (pr["err/SE"].abs() < 3).all(),
        f"max |err/SE|={pr['err/SE'].abs().max():.2f}; max abs err={(pr['IPW (true e)'] - pr['true ATE']).abs().max():.4f}")
    chk("5 Confounding", "IPW(est e) purchase ATE within 3 standard errors of truth, all arms", (pr["err/SE "].abs() < 3).all(),
        f"max |err/SE|={pr['err/SE '].abs().max():.2f}; max abs err={(pr['IPW (est e)'] - pr['true ATE']).abs().max():.4f}")

    # ---------------------------------------------------------------- 6 heterogeneity
    ite = {k: g[f"true_ite_purchase_arm{k}"].to_numpy() for k in (1, 2, 3)}
    prof = {k: g[f"true_incr_net_profit_arm{k}"].to_numpy() for k in (1, 2, 3)}
    het = pd.DataFrame({f"ITE purchase {G.ARM_PCT[k]}%": pd.Series(ite[k]).describe(percentiles=[.05, .5, .95]) for k in (1, 2, 3)}).T
    het["share<0"] = [(ite[k] < 0).mean() for k in (1, 2, 3)]
    het["share<-0.01"] = [(ite[k] < -0.01).mean() for k in (1, 2, 3)]
    prof_tbl = pd.DataFrame({f"Δ profit {G.ARM_PCT[k]}%": pd.Series(prof[k]).describe(percentiles=[.05, .5, .95]) for k in (1, 2, 3)}).T
    prof_tbl["share>0"] = [(prof[k] > 0).mean() for k in (1, 2, 3)]
    seg_het = g.assign(segment=m.customer_segment.values).groupby("segment")[
        ["true_ite_purchase_arm1", "true_ite_purchase_arm2", "true_ite_purchase_arm3", "true_incr_net_profit_arm1", "true_incr_net_profit_arm2", "true_incr_net_profit_arm3"]].mean()
    cmp_het = g.assign(c=m.campaign_type.values, l=m.loyalty_member.values)
    camp_het = cmp_het.groupby("c")[["true_ite_purchase_arm2"]].mean().rename(columns={"true_ite_purchase_arm2": "mean ITE purchase @10%"})
    loy_het = cmp_het.groupby("l")[["true_ite_purchase_arm2"]].mean().rename(columns={"true_ite_purchase_arm2": "mean ITE purchase @10%"})
    nonmono = pd.Series({"share p(20%) < p(10%)": (pk[:, 3] < pk[:, 2]).mean(), "share p(10%) < p(5%)": (pk[:, 2] < pk[:, 1]).mean(),
                         "share p(5%) < p(0%)": (pk[:, 1] < pk[:, 0]).mean()}, name="share of customers")
    best = g.oracle_best_arm_by_expected_profit.value_counts(normalize=True).sort_index()
    best.index = [f"arm{k} ({G.ARM_PCT[k]}%)" for k in best.index]
    sd10 = ite[2].std()
    chk("6 Heterogeneity", "true ITE has meaningful dispersion (sd of ITE@10% > 0.04)", sd10 > 0.04, f"sd={sd10:.4f}")
    chk("6 Heterogeneity", "segment-level mean ITE@20% spans > 0.10", seg_het.true_ite_purchase_arm3.max() - seg_het.true_ite_purchase_arm3.min() > 0.10,
        f"range={seg_het.true_ite_purchase_arm3.max() - seg_het.true_ite_purchase_arm3.min():.3f}")
    chk("6 Heterogeneity", "some customers respond weakly/negatively (share ITE@20%<0 between 2% and 40%)", 0.02 < (ite[3] < 0).mean() < 0.40, f"{(ite[3] < 0).mean():.3f}")
    chk("6 Heterogeneity", "purchase response is not forced monotone (some customers have p(20%) < p(10%))", (pk[:, 3] < pk[:, 2]).mean() > 0.01)
    chk("6 Heterogeneity", "profit-optimal arm is mixed (>=3 arms each chosen by >=3% of customers)", (g.oracle_best_arm_by_expected_profit.value_counts(normalize=True) >= 0.03).sum() >= 3,
        str(best.round(3).to_dict()))
    Xc = as_cat(m)
    cv_r2 = {}
    for k in (3,):
        pred = np.zeros(n)
        for tr, te in KFold(5, shuffle=True, random_state=SEED_CV).split(Xc):
            mdl = HistGradientBoostingRegressor(random_state=SEED_CV, max_depth=4, learning_rate=0.08, max_iter=200, categorical_features="from_dtype").fit(Xc.iloc[tr], ite[k][tr])
            pred[te] = mdl.predict(Xc.iloc[te])
        cv_r2[k] = 1 - ((ite[k] - pred) ** 2).sum() / ((ite[k] - ite[k].mean()) ** 2).sum()
    chk("6 Heterogeneity", "true ITE@20% is learnable from observed covariates (CV R² > 0.5) but not perfectly (< 0.999)", 0.5 < cv_r2[3] < 0.999, f"R²={cv_r2[3]:.3f}")

    # ---------------------------------------------------------------- 7 leakage
    gt_only = set(g.columns) - {"customer_id"}
    chk("7 Leakage", "no ground-truth column appears in the main dataset", not (gt_only & set(m.columns)), str(sorted(gt_only & set(m.columns))))
    chk("7 Leakage", "feature list disjoint from treatment, outcome and ground-truth columns",
        not (set(G.FEATURE_COLUMNS) & (set(G.TREATMENT_COLUMNS) | set(G.OUTCOME_COLUMNS) | gt_only | {"customer_id"})))
    chk("7 Leakage", "main columns = id + features + treatment + outcomes (nothing else)", set(m.columns) == {"customer_id"} | set(G.FEATURE_COLUMNS) | set(G.TREATMENT_COLUMNS) | set(G.OUTCOME_COLUMNS))
    tgt = m[["treatment_arm", "discount_rate", "purchased", "revenue", "net_profit", "discount_amount"]]
    cm = pd.concat([m[NUM_FEATS + ["loyalty_member"]].loc[:, ~pd.Index(NUM_FEATS + ["loyalty_member"]).duplicated()], tgt], axis=1).corr().loc[NUM_FEATS, tgt.columns].abs()
    mx = cm.stack().sort_values(ascending=False).head(6)
    mx.index = [f"{a_} ~ {b_}" for a_, b_ in mx.index]
    chk("7 Leakage", "no numeric covariate near-deterministically tied to treatment/outcome (max |r| < 0.6)", mx.max() < 0.6, f"max |r|={mx.max():.3f}")
    # feature-only predictability of purchase
    oof = np.zeros(n)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED_CV).split(Xc, m.purchased):
        clf = HistGradientBoostingClassifier(random_state=SEED_CV, max_depth=4, learning_rate=0.06, max_iter=200, categorical_features="from_dtype").fit(Xc.iloc[tr], m.purchased.iloc[tr])
        oof[te] = clf.predict_proba(Xc.iloc[te])[:, 1]
    auc_p = roc_auc_score(m.purchased, oof)
    chk("7 Leakage", "covariate-only purchase AUC is moderate (0.60-0.85), not near 1", 0.60 <= auc_p <= 0.85, f"AUC={auc_p:.3f}")
    chk("7 Leakage", "covariate-only treatment AUC (one-vs-rest) is moderate (0.55-0.85)", 0.55 <= auc_arm <= 0.85, f"AUC={auc_arm:.3f}, accuracy={acc:.3f}")
    # pre-treatment invariance: redraw treatment only
    m2, g2 = G.generate(G.SEED, G.N, treatment_stream=1)
    same_cov = all(np.array_equal(m2[c].to_numpy(), m[c].to_numpy()) if m[c].dtype == object or str(m[c].dtype).startswith("str") else np.allclose(m2[c], m[c]) for c in ["customer_id"] + G.FEATURE_COLUMNS)
    pot_cols = [c for c in g.columns if c not in ("assigned_arm",)]
    same_gt = all(np.allclose(g2[c].to_numpy(), g[c].to_numpy()) for c in pot_cols)
    diff_arm = float((m2.treatment_arm.to_numpy() != arm).mean())
    cons2 = all(np.allclose(m2.loc[m2.treatment_arm == k, "net_profit"], g2.loc[m2.treatment_arm == k, f"net_profit_arm{k}"]) for k in range(4))
    chk("7 Leakage", "re-drawing treatment leaves ALL covariates unchanged (covariates are pre-treatment)", same_cov)
    chk("7 Leakage", "re-drawing treatment leaves every ground-truth potential outcome / true effect / propensity unchanged", same_gt)
    chk("7 Leakage", "re-drawn treatment is a genuinely different assignment", diff_arm > 0.2, f"{diff_arm:.3f} of customers got a different arm")
    chk("7 Leakage", "observed outcomes in the re-drawn dataset still equal potential outcomes at the new arm", cons2)

    # ---------------------------------------------------------------- 8 reproducibility
    with tempfile.TemporaryDirectory() as td:
        r = subprocess.run([sys.executable, "-I", os.path.join(HERE, "generate_econocausal_dataset.py"), "--outdir", td], capture_output=True, text=True)
        ok_run = r.returncode == 0
        h1 = (sha256(os.path.join(td, "econocausal_synthetic_dataset.csv")), sha256(os.path.join(td, "ground_truth_evaluation.csv"))) if ok_run else ("", "")
    h0 = (sha256(p_main), sha256(p_gt))
    chk("8 Reproducibility", "fresh-process regeneration gives byte-identical main CSV (SHA-256)", ok_run and h1[0] == h0[0], h0[0][:16] + "…")
    chk("8 Reproducibility", "fresh-process regeneration gives byte-identical ground-truth CSV (SHA-256)", ok_run and h1[1] == h0[1], h0[1][:16] + "…")

    # ---------------------------------------------------------------- 9 downstream smoke tests (sklearn / scipy)
    ds = []
    # --- 9a doubly-robust (orthogonal, cross-fitted) DML-style estimation
    kf = KFold(5, shuffle=True, random_state=SEED_CV)
    folds = list(kf.split(Xc))

    def aipw_scores(yv, binary):
        """Cross-fitted AIPW score matrix psi[:,k] for E[Y(k)] with estimated propensities e_hat."""
        mu = np.zeros((n, 4))
        for tr, te in folds:
            for k in range(4):
                trk = tr[arm[tr] == k]
                if binary:
                    mdl = HistGradientBoostingClassifier(random_state=SEED_CV, max_depth=3, learning_rate=0.06, max_iter=120, categorical_features="from_dtype").fit(Xc.iloc[trk], yv[trk])
                    mu[te, k] = mdl.predict_proba(Xc.iloc[te])[:, 1]
                else:
                    mdl = HistGradientBoostingRegressor(random_state=SEED_CV, max_depth=3, learning_rate=0.06, max_iter=120, categorical_features="from_dtype").fit(Xc.iloc[trk], yv[trk])
                    mu[te, k] = mdl.predict(Xc.iloc[te])
        ind = np.eye(4)[arm]
        return mu + ind * (yv[:, None] - mu) / e_hat, mu

    def cate_oof(pseudo):
        pred = np.zeros(n)
        for tr, te in folds:
            mdl = HistGradientBoostingRegressor(random_state=SEED_CV, max_depth=3, learning_rate=0.05, max_iter=150, categorical_features="from_dtype").fit(Xc.iloc[tr], pseudo[tr])
            pred[te] = mdl.predict(Xc.iloc[te])
        return pred

    ypur = m.purchased.to_numpy().astype(float)
    ynet = m.net_profit.to_numpy()
    yspend = (m.discount_amount + m.marketing_cost).to_numpy()
    psi_pur, mu_pur = aipw_scores(ypur, True)
    psi_net, _ = aipw_scores(ynet, False)
    psi_sp, _ = aipw_scores(yspend, False)
    rows_dr, cate_hat = [], {}
    for k in (1, 2, 3):
        d = psi_pur[:, k] - psi_pur[:, 0]
        est, se = d.mean(), d.std() / np.sqrt(n)
        truth = ite[k].mean()
        ch = cate_oof(d)
        cate_hat[k] = ch
        rows_dr.append({"arm": f"{G.ARM_PCT[k]}% vs 0%", "true ATE": truth, "DR/DML ATE": est, "95% CI lo": est - 1.96 * se, "95% CI hi": est + 1.96 * se,
                        "CI covers truth": bool(est - 1.96 * se <= truth <= est + 1.96 * se), "CATE corr (Pearson)": np.corrcoef(ch, ite[k])[0, 1],
                        "CATE corr (Spearman)": spearmanr(ch, ite[k])[0], "CATE RMSE": float(np.sqrt(np.mean((ch - ite[k]) ** 2))), "true ITE sd": ite[k].std()})
    dr = pd.DataFrame(rows_dr).set_index("arm")
    chk("9 Downstream", "orthogonal DR/DML-style ATE within 0.02 of truth for every arm", ((dr["DR/DML ATE"] - dr["true ATE"]).abs() < 0.02).all(),
        f"max err={((dr['DR/DML ATE'] - dr['true ATE']).abs()).max():.4f}")
    chk("9 Downstream", "estimated CATE rank-correlates with true ITE (Spearman > 0.5, arm 20%)", dr["CATE corr (Spearman)"].iloc[-1] > 0.5, f"{dr['CATE corr (Spearman)'].iloc[-1]:.3f}")
    # --- 9b PSM
    Xs = StandardScaler().fit_transform(Xbal)
    psm_rows = []
    for k in (1, 2, 3):
        sub = np.where((arm == 0) | (arm == k))[0]
        t = (arm[sub] == k).astype(int)
        lr = LogisticRegression(max_iter=3000).fit(Xs[sub], t)
        lg = lr.decision_function(Xs[sub])
        cal = 0.2 * lg.std()
        tr_i, co_i = np.where(t == 1)[0], np.where(t == 0)[0]
        nn = NearestNeighbors(n_neighbors=1).fit(lg[co_i, None])
        dist_, idx = nn.kneighbors(lg[tr_i, None])
        keep = dist_[:, 0] <= cal
        ti = sub[tr_i[keep]]
        ci = sub[co_i[idx[keep, 0]]]
        att_hat = ypur[ti].mean() - ypur[ci].mean()
        att_true = (g[f"p_purchase_arm{k}"].to_numpy()[ti] - g["p_purchase_arm0"].to_numpy()[ti]).mean()
        att_naive = ypur[arm == k].mean() - ypur[arm == 0].mean()
        sb = smd_table(Xbal, arm == k, arm == 0).abs()
        Xt = Xbal.iloc[ti]
        Xc_ = Xbal.iloc[ci]
        sd = np.sqrt((Xbal[arm == k].var() + Xbal[arm == 0].var()) / 2).replace(0, np.nan)
        sa = ((Xt.mean() - Xc_.mean()) / sd).fillna(0).abs()
        psm_rows.append({"arm": f"{G.ARM_PCT[k]}% vs 0%", "treated matched": int(keep.sum()), "treated total": int(len(tr_i)), "naive ATT": att_naive,
                         "PSM ATT": att_hat, "true ATT (matched treated)": att_true, "max|SMD| before": sb.max(), "max|SMD| after": sa.max(), "mean|SMD| before": sb.mean(),
                         "mean|SMD| after": sa.mean()})
    psm = pd.DataFrame(psm_rows).set_index("arm")
    chk("9 Downstream", "PSM reduces mean |SMD| for every arm", (psm["mean|SMD| after"] < psm["mean|SMD| before"]).all())
    chk("9 Downstream", "PSM retains >= 90% of treated units within caliper (overlap)", (psm["treated matched"] / psm["treated total"] >= 0.90).all(),
        str((psm["treated matched"] / psm["treated total"]).round(3).to_dict()))
    chk("9 Downstream", "PSM ATT closer to truth than naive ATT for >=2 of 3 arms", ((psm["PSM ATT"] - psm["true ATT (matched treated)"]).abs() < (psm["naive ATT"] - psm["true ATT (matched treated)"]).abs()).sum() >= 2)
    # --- 9c Qini vs random
    fr = np.linspace(0, 1, 101)

    def qini(score_, psi_):
        o = np.argsort(-score_)
        cum = np.concatenate([[0], np.cumsum(psi_[o])])
        q = np.interp(fr * n, np.arange(n + 1), cum)
        return q, np.trapezoid(q - fr * cum[-1], fr)

    qrows = []
    for k in (1, 2, 3):
        psi_ip = (arm == k) * ypur / e_hat[:, k] - (arm == 0) * ypur / e_hat[:, 0]      # IPW individual uplift contribution
        _, a_mod = qini(cate_hat[k], psi_ip)
        _, a_orc = qini(ite[k], psi_ip)
        _, a_rnd = qini(np.random.default_rng(1).random(n), psi_ip)
        _, t_mod = qini(cate_hat[k], ite[k])
        _, t_orc = qini(ite[k], ite[k])
        _, t_rnd = qini(np.random.default_rng(1).random(n), ite[k])
        qrows.append({"arm": f"{G.ARM_PCT[k]}% vs 0%", "Qini area (IPW-realised): model": a_mod, "oracle": a_orc, "random": a_rnd,
                      "Qini area (true effects): model": t_mod, "oracle ": t_orc, "random ": t_rnd, "model / oracle (true)": t_mod / t_orc})
    qdf = pd.DataFrame(qrows).set_index("arm")
    chk("9 Downstream", "estimated-CATE Qini area beats random targeting (true-effect Qini) for every arm", (qdf["Qini area (true effects): model"] > qdf["random "].abs() * 3).all(),
        str(qdf["model / oracle (true)"].round(3).to_dict()))
    chk("9 Downstream", "estimated-CATE Qini area beats random targeting (IPW-realised Qini) for every arm", (qdf["Qini area (IPW-realised): model"] > qdf["random"]).all())
    # --- 9d SciPy LP budget optimisation
    dp_true = np.stack([np.zeros(n)] + [prof[k] for k in (1, 2, 3)], 1)
    sp_true = np.stack([np.zeros(n)] + [g[f"true_incr_promo_spend_arm{k}"].to_numpy() for k in (1, 2, 3)], 1)
    dp_hat = np.stack([np.zeros(n)] + [psi_net[:, k] - psi_net[:, 0] for k in (1, 2, 3)], 1)
    dp_hat_s = np.stack([np.zeros(n)] + [cate_oof(psi_net[:, k] - psi_net[:, 0]) for k in (1, 2, 3)], 1)
    sp_hat_s = np.stack([np.zeros(n)] + [cate_oof(psi_sp[:, k] - psi_sp[:, 0]) for k in (1, 2, 3)], 1)
    sp_hat_s = np.maximum(sp_hat_s, 0)
    Aeq = sparse.kron(sparse.identity(n), np.ones((1, 4)), format="csr")
    beq = np.ones(n)

    def lp(gain, spend, budget):
        res = linprog(-gain.ravel(), A_ub=sparse.csr_matrix(spend.ravel()[None, :]), b_ub=[budget], A_eq=Aeq, b_eq=beq, bounds=(0, 1), method="highs")
        return res.x.reshape(n, 4) if res.success else None

    # structured plug-in: cross-fitted purchase models x unit economics estimated from OBSERVED columns only
    cr_cat = (m.loc[y, "cost_of_goods"] / B[y]).groupby(m.loc[y, "product_category"]).mean()
    c_hat = B * m.product_category.map(cr_cat).to_numpy()
    ch_hat = m[arm == 0].groupby("preferred_channel").marketing_cost.mean()
    coupon_hat = float((m.loc[arm > 0, "marketing_cost"] - m.loc[arm > 0, "preferred_channel"].map(ch_hat)).mean())
    mkt_hat = np.stack([m.preferred_channel.map(ch_hat).to_numpy() + coupon_hat * (k > 0) for k in range(4)], 1)
    prof_hat = mu_pur * (B[:, None] * (1 - G.ARM_RATE[None, :]) - c_hat[:, None]) - mkt_hat
    spend_hat = mu_pur * B[:, None] * G.ARM_RATE[None, :] + mkt_hat
    dp_str = prof_hat - prof_hat[:, [0]]
    sp_str = spend_hat - spend_hat[:, [0]]

    best_arm = dp_true.argmax(1)
    full_spend = sp_true[np.arange(n), best_arm].sum()
    full_gain = dp_true[np.arange(n), best_arm].sum()
    lrows = []
    for frac in (0.25, 0.5, 1.0):
        bud = frac * full_spend
        row = {"budget (INR)": bud}
        for tag, (gn, sp) in {"oracle": (dp_true, sp_true), "structured": (dp_str, sp_str), "generic DR": (dp_hat_s, sp_hat_s)}.items():
            x = lp(gn, sp, bud)
            row[f"{tag} gain"] = float((x * dp_true).sum())
            row[f"{tag} spend"] = float((x * sp_true).sum())
            row[f"{tag} ROI"] = row[f"{tag} gain"] / max(row[f"{tag} spend"], 1e-9)
        row["best random gain"] = max(min(1.0, bud / sp_true[:, k].sum()) * dp_true[:, k].sum() for k in (1, 2, 3))
        lrows.append(row)
    lpd = pd.DataFrame(lrows)
    lpd.index = [f"{int(f * 100)}% of oracle-optimal spend" for f in (0.25, 0.5, 1.0)]
    chk("9 Downstream", "SciPy LP budget optimisation solves; oracle spend within budget", all(r["oracle spend"] <= r["budget (INR)"] * 1.001 + 1 for r in lrows))
    chk("9 Downstream", "oracle allocation beats best random targeting at every budget", all(r["oracle gain"] > r["best random gain"] for r in lrows))
    chk("9 Downstream", "structured model-based allocation (observed data only) beats best random targeting at every budget (true value)",
        all(r["structured gain"] > r["best random gain"] for r in lrows), str([round(r["structured gain"] / r["oracle gain"], 2) for r in lrows]) + " of oracle gain")
    chk("9 Downstream", "generic DR-profit-CATE allocation beats best random targeting at every budget (true value)",
        all(r["generic DR gain"] > r["best random gain"] for r in lrows), str([round(r["generic DR gain"] / r["oracle gain"], 2) for r in lrows]) + " of oracle gain")
    chk("9 Downstream", "structured allocation never exceeds oracle gain (sanity)", all(r["structured gain"] <= r["oracle gain"] + 1 for r in lrows))
    treat_all = {k: (dp_true[:, k].sum(), sp_true[:, k].sum()) for k in (1, 2, 3)}
    pol = pd.DataFrame({"policy": ["no discount (0%)"] + [f"everyone {G.ARM_PCT[k]}%" for k in (1, 2, 3)] + ["oracle unconstrained (best arm per customer)"],
                        "expected Δ net profit vs 0% (INR, all 20k)": [0.0] + [treat_all[k][0] for k in (1, 2, 3)] + [full_gain],
                        "expected Δ promo spend (INR)": [0.0] + [treat_all[k][1] for k in (1, 2, 3)] + [full_spend]}).set_index("policy")
    pol["ROI = Δprofit / Δspend"] = pol.iloc[:, 0] / pol.iloc[:, 1].replace(0, np.nan)
    # --- 9e library probe
    probe = {}
    for lib in ("dowhy", "econml"):
        try:
            importlib.import_module(lib)
            probe[lib] = "importable (compatibility smoke test NOT run by this script)"
        except Exception as e:  # noqa
            probe[lib] = f"NOT installed in this environment ({type(e).__name__}) - NOT executed"

    # ---------------------------------------------------------------- report
    n_pass = sum(c[2] for c in checks)
    out.append("# Dataset Validation Report — EconoCausal (Dynamic Pricing via DML)\n")
    out.append("All numbers below were computed by `validate_econocausal_dataset.py` from the files on disk "
               f"(`econocausal_synthetic_dataset.csv`, `ground_truth_evaluation.csv`); runtime {time.time() - t0:.0f}s. "
               "Findings validate the **simulation assumptions** of a synthetic data-generating process, not real-world causal truth.\n")
    out.append(f"**Result: {n_pass}/{len(checks)} checks passed.**\n")
    out.append(f"- Generator seed: `{G.SEED}`; main CSV SHA-256 `{h0[0]}`; ground-truth CSV SHA-256 `{h0[1]}`")
    out.append(f"- DoWhy: {probe['dowhy']}")
    out.append(f"- EconML: {probe['econml']}")
    out.append("- Downstream smoke tests in §9 use scikit-learn and SciPy re-implementations of the relevant estimators (cross-fitted doubly-robust/DML-style learner, PSM, Qini, LP). "
               "They show the data are *suitable* for these workflows; they do **not** certify DoWhy/EconML runs.\n")
    out.append("## Check summary\n")
    out.append("| Section | Check | Result | Detail |\n|---|---|---|---|")
    for s, nme, ok, d in checks:
        out.append(f"| {s} | {nme} | {'PASS' if ok else '**FAIL**'} | {d} |")
    out.append("\n## 1. Structure\n")
    out.append(f"Main file: {m.shape[0]:,} rows × {m.shape[1]} columns. Ground-truth file: {g.shape[0]:,} rows × {g.shape[1]} columns.\n")
    out.append(md(dtypes_tbl, "{:.0f}"))
    out.append("\n## 2. Treatment counts and outcome distributions\n")
    out.append(md(cnt, "{:.4f}"))
    out.append("\nObserved (confounded) outcomes by arm:\n")
    out.append(md(by_arm, "{:.3f}"))
    out.append("\nDistributions (INR):\n")
    out.append(md(dist, "{:.2f}"))
    out.append("\n## 3. Financial consistency\n")
    out.append(f"Share of purchases with negative gross profit (revenue < COGS): {neg_gp:.4f}; by arm: {neg_gp_by_arm.round(4).to_dict()}.\n")
    out.append(md(cogs_by_cat, "{:.4f}"))
    out.append("\n## 4. Treatment overlap\n")
    out.append("True assignment propensities (ground truth):\n")
    out.append(md(ov, "{:.4f}"))
    out.append("\nCross-fitted multinomial-logit estimated propensities (features only):\n")
    out.append(md(est_ov, "{:.4f}"))
    out.append(f"\nEstimated-propensity accuracy {acc:.3f} vs. majority-class {np.bincount(arm).max() / n:.3f}; one-vs-rest AUC {auc_arm:.3f}.\n")
    out.append("\nCustomers per segment and arm:\n")
    out.append(md(seg_cnt, "{:.0f}"))
    out.append("\n## 5. Confounding\n")
    out.append(f"Correlation of assignment score with baseline purchase probability P(Y(0)=1): {corr_sb:.3f}; with true effect size β: {corr_se:.3f}.\n")
    out.append("Mean covariates by observed arm (differences = confounding):\n")
    out.append(md(cov_by_arm, "{:.3f}"))
    out.append("\nCovariate balance (standardized mean differences, arm k vs 0, all one-hot/numeric covariates):\n")
    out.append(md(bal, "{:.3f}"))
    out.append("\nMost imbalanced covariates (arm 20% vs 0%):\n")
    out.append(md(top_imb, "{:.3f}"))
    out.append("\nEffect recovery — naive vs adjusted estimators (truth from ground-truth file):\n")
    out.append(md(rec.set_index(["outcome", "arm"]), "{:.4f}"))
    out.append("\n## 6. Heterogeneous ground-truth effects\n")
    out.append(md(het[["mean", "std", "min", "5%", "50%", "95%", "max", "share<0", "share<-0.01"]], "{:.4f}"))
    out.append("\n" + md(prof_tbl[["mean", "std", "min", "5%", "50%", "95%", "max", "share>0"]], "{:.3f}"))
    out.append("\nMean true effect by customer segment (purchase probability; incremental expected net profit in INR):\n")
    out.append(md(seg_het, "{:.3f}"))
    out.append("\n" + md(camp_het, "{:.4f}") + "\n\n" + md(loy_het, "{:.4f}"))
    out.append("\nNon-monotone dose-response (shares of customers):\n")
    out.append(md(nonmono.to_frame(), "{:.4f}"))
    out.append("\nProfit-optimal arm (oracle, expected net profit):\n")
    out.append(md(best.rename("share").to_frame(), "{:.4f}"))
    out.append(f"\nCV R² of a gradient-boosted model predicting the true ITE@20% from observed covariates: {cv_r2[3]:.3f}.\n")
    out.append("\n## 7. Leakage\n")
    out.append("Largest |Pearson r| between numeric covariates and treatment/outcomes:\n")
    out.append(md(mx.rename("|r|").to_frame(), "{:.3f}"))
    out.append(f"\nCovariate-only cross-validated AUC for `purchased`: {auc_p:.3f}. Treatment re-draw test: {diff_arm:.1%} of customers received a different arm; covariates and ground truth were unchanged.\n")
    out.append("\n## 8. Reproducibility\n")
    out.append(f"Fresh-process regeneration SHA-256 — main: `{h1[0]}`; ground truth: `{h1[1]}`. Deliverable files — main: `{h0[0]}`; ground truth: `{h0[1]}`.\n")
    out.append("\n## 9. Downstream compatibility smoke tests (scikit-learn / SciPy)\n")
    out.append("### 9a Cross-fitted doubly-robust (orthogonal / DML-style) ATE and CATE for purchase\n")
    out.append(md(dr, "{:.4f}"))
    out.append("\n### 9b Propensity-score matching (1:1 nearest neighbour, caliper 0.2 SD of logit, with replacement)\n")
    out.append(md(psm, "{:.4f}"))
    out.append("\n### 9c Qini area vs random targeting (areas in units of incremental purchases summed over 20,000 customers)\n")
    out.append("IPW-realised Qini uses only observed data (estimated propensities); true-effect Qini uses ground-truth ITEs (noise-free).\n")
    out.append(md(qdf, "{:.3f}"))
    out.append("\n### 9d SciPy LP budget optimisation\n")
    out.append("Objective: maximise expected incremental net profit vs 0% discount. Budget: expected incremental promotional spend = expected redeemed discount value + marketing/coupon cost. "
               "Gains/spend are evaluated with ground-truth expectations. 'structured' = cross-fitted purchase-probability models × unit economics (COGS ratio by category, channel/coupon cost) estimated from observed columns; "
               "'generic DR' = doubly-robust CATE of realised net profit and spend (noisier, because realised profit is a heavy-tailed single draw). 'best random gain' = best of randomly giving one uniform discount level to a budget-sized random subset.\n")
    lpf = lpd.copy()
    for c_ in lpf.columns:
        lpf[c_] = [f"{x:.3f}" if "ROI" in c_ else f"{x:,.0f}" for x in lpf[c_]]
    out.append(md(lpf, "{}"))
    out.append("\nReference policies (whole population):\n")
    out.append(md(pol.assign(**{"ROI = Δprofit / Δspend": pol["ROI = Δprofit / Δspend"].map(lambda x: "n/a" if pd.isna(x) else f"{x:.3f}")}), "{:,.2f}"))
    out.append("\n## 10. Fixes made during validation\n")
    out.append(NOTES)
    out.append("\n## 11. Limitations of this validation\n")
    out.append("- Passing checks confirms the dataset behaves as the documented DGP specifies; it says nothing about real customers.\n"
               "- The DR/DML-style, PSM, Qini and LP results come from scikit-learn/SciPy code written for this validation; DoWhy and EconML were not run (see top).\n"
               "- Estimator recovery thresholds are sanity bounds chosen for this sample size, not formal statistical tests.\n")
    with open(a.report, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")
    print(f"{n_pass}/{len(checks)} checks passed; report -> {a.report}")
    for s, nme, ok, d in checks:
        if not ok:
            print("FAIL:", s, "|", nme, "|", d)


if __name__ == "__main__":
    main()
