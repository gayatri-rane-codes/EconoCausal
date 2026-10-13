# Dataset Validation Report — EconoCausal (Dynamic Pricing via DML)

All numbers below were computed by `validate_econocausal_dataset.py` from the files on disk (`econocausal_synthetic_dataset.csv`, `ground_truth_evaluation.csv`); runtime 29s. Findings validate the **simulation assumptions** of a synthetic data-generating process, not real-world causal truth.

**Result: 68/68 checks passed.**

- Generator seed: `20261012`; main CSV SHA-256 `1e38231d42fa87e9de0cebef58ec3b792736c52105c88ea92e1b4cf4104f3f45`; ground-truth CSV SHA-256 `6aa6af1c7e7c057df9c53ccdedf858c3d3108056fddad5456df2a15b3cca5335`
- DoWhy: importable (compatibility smoke test NOT run by this script)
- EconML: NOT installed in this environment (ModuleNotFoundError) - NOT executed
- Downstream smoke tests in §9 use scikit-learn and SciPy re-implementations of the relevant estimators (cross-fitted doubly-robust/DML-style learner, PSM, Qini, LP). They show the data are *suitable* for these workflows; they do **not** certify DoWhy/EconML runs.

## Check summary

| Section | Check | Result | Detail |
|---|---|---|---|
| 1 Structure | main rows == 20,000 | PASS | 20000 |
| 1 Structure | main columns == expected 45 in documented order | PASS | 45 |
| 1 Structure | ground-truth rows == 20,000 | PASS | 20000; 69 columns |
| 1 Structure | no duplicate customer_id | PASS | 0 duplicates |
| 1 Structure | ids contiguous 100001..120000 | PASS |  |
| 1 Structure | ground-truth ids identical and in same order | PASS |  |
| 1 Structure | no missing values (main) | PASS | 0 |
| 1 Structure | no missing values (ground truth) | PASS | 0 |
| 1 Structure | no inf values | PASS |  |
| 1 Structure | dtypes match schema (int / float / string) | PASS | violations: [] |
| 1 Structure | categorical values within allowed domains | PASS | {} |
| 1 Structure | numeric values within documented ranges | PASS | {} |
| 1 Structure | treatment encoding consistent (arm<->pct<->rate) | PASS |  |
| 2 Treatment/outcomes | all four arms present with >= 2,500 obs each | PASS | {0: 7584, 5: 5395, 10: 4266, 20: 2755} |
| 2 Treatment/outcomes | overall purchase rate in plausible band 0.15-0.50 | PASS | 0.3271 |
| 3 Financial | non-buyers: discount_amount = revenue = cost_of_goods = 0 | PASS |  |
| 3 Financial | non-buyers: net_profit = -marketing_cost | PASS |  |
| 3 Financial | buyers: discount_amount = round(basket * rate, 2) | PASS |  |
| 3 Financial | buyers: revenue = basket - discount_amount | PASS |  |
| 3 Financial | all: net_profit = revenue - cost_of_goods - marketing_cost | PASS |  |
| 3 Financial | all: marketing_cost = channel cost + coupon cost*(arm>0) | PASS |  |
| 3 Financial | all: marketing_cost independent of purchase (no discount double-count) | PASS |  |
| 3 Financial | buyers: 0 <= revenue <= basket | PASS |  |
| 3 Financial | buyers: COGS ratio within [0.30, 0.90] | PASS |  |
| 3 Financial | observed outcomes == ground-truth potential outcome at assigned arm (all 5 outcome columns, 4 arms) | PASS |  |
| 3 Financial | ground-truth assigned_arm == treatment_arm | PASS |  |
| 3 Financial | E[revenue_k] + E[discount_k] == p_k * basket (list-price conservation) | PASS |  |
| 4 Overlap | true propensities sum to 1 | PASS |  |
| 4 Overlap | min true propensity >= 0.02 for every arm | PASS | min=0.0424 |
| 4 Overlap | <= 5% of customers have any arm propensity < 0.05 | PASS | 0.0136 |
| 4 Overlap | every segment x campaign cell with >=30 customers has all 4 arms | PASS | min nonzero-cell count=5, cells with a missing arm=0 |
| 5 Confounding | raw covariate imbalance exists (max |SMD| arm3 vs 0 > 0.3) | PASS | 0.525 |
| 5 Confounding | IPW with true propensities balances covariates (max |SMD| < 0.1, all arms) | PASS | 0.0544 |
| 5 Confounding | assignment score correlates with baseline purchase probability (|r| > 0.15) | PASS | r=-0.677 |
| 5 Confounding | assignment score correlates with effect size (|r| > 0.15) | PASS | r=0.618 |
| 5 Confounding | naive purchase-rate difference is materially biased for >=1 arm (|bias| > 0.02) | PASS | max |bias|=0.0726 |
| 5 Confounding | IPW(true e) purchase ATE within 3 standard errors of truth, all arms | PASS | max |err/SE|=1.24; max abs err=0.0101 |
| 5 Confounding | IPW(est e) purchase ATE within 3 standard errors of truth, all arms | PASS | max |err/SE|=1.19; max abs err=0.0097 |
| 6 Heterogeneity | true ITE has meaningful dispersion (sd of ITE@10% > 0.04) | PASS | sd=0.1011 |
| 6 Heterogeneity | segment-level mean ITE@20% spans > 0.10 | PASS | range=0.333 |
| 6 Heterogeneity | some customers respond weakly/negatively (share ITE@20%<0 between 2% and 40%) | PASS | 0.279 |
| 6 Heterogeneity | purchase response is not forced monotone (some customers have p(20%) < p(10%)) | PASS |  |
| 6 Heterogeneity | profit-optimal arm is mixed (>=3 arms each chosen by >=3% of customers) | PASS | {'arm0 (0%)': 0.622, 'arm1 (5%)': 0.024, 'arm2 (10%)': 0.122, 'arm3 (20%)': 0.232} |
| 6 Heterogeneity | true ITE@20% is learnable from observed covariates (CV R² > 0.5) but not perfectly (< 0.999) | PASS | R²=0.884 |
| 7 Leakage | no ground-truth column appears in the main dataset | PASS | [] |
| 7 Leakage | feature list disjoint from treatment, outcome and ground-truth columns | PASS |  |
| 7 Leakage | main columns = id + features + treatment + outcomes (nothing else) | PASS |  |
| 7 Leakage | no numeric covariate near-deterministically tied to treatment/outcome (max |r| < 0.6) | PASS | max |r|=0.358 |
| 7 Leakage | covariate-only purchase AUC is moderate (0.60-0.85), not near 1 | PASS | AUC=0.741 |
| 7 Leakage | covariate-only treatment AUC (one-vs-rest) is moderate (0.55-0.85) | PASS | AUC=0.598, accuracy=0.392 |
| 7 Leakage | re-drawing treatment leaves ALL covariates unchanged (covariates are pre-treatment) | PASS |  |
| 7 Leakage | re-drawing treatment leaves every ground-truth potential outcome / true effect / propensity unchanged | PASS |  |
| 7 Leakage | re-drawn treatment is a genuinely different assignment | PASS | 0.697 of customers got a different arm |
| 7 Leakage | observed outcomes in the re-drawn dataset still equal potential outcomes at the new arm | PASS |  |
| 8 Reproducibility | fresh-process regeneration gives byte-identical main CSV (SHA-256) | PASS | 1e38231d42fa87e9… |
| 8 Reproducibility | fresh-process regeneration gives byte-identical ground-truth CSV (SHA-256) | PASS | 6aa6af1c7e7c057d… |
| 9 Downstream | orthogonal DR/DML-style ATE within 0.02 of truth for every arm | PASS | max err=0.0084 |
| 9 Downstream | estimated CATE rank-correlates with true ITE (Spearman > 0.5, arm 20%) | PASS | 0.804 |
| 9 Downstream | PSM reduces mean |SMD| for every arm | PASS |  |
| 9 Downstream | PSM retains >= 90% of treated units within caliper (overlap) | PASS | {'5% vs 0%': 1.0, '10% vs 0%': 1.0, '20% vs 0%': 1.0} |
| 9 Downstream | PSM ATT closer to truth than naive ATT for >=2 of 3 arms | PASS |  |
| 9 Downstream | estimated-CATE Qini area beats random targeting (true-effect Qini) for every arm | PASS | {'5% vs 0%': 0.334, '10% vs 0%': 0.641, '20% vs 0%': 0.819} |
| 9 Downstream | estimated-CATE Qini area beats random targeting (IPW-realised Qini) for every arm | PASS |  |
| 9 Downstream | SciPy LP budget optimisation solves; oracle spend within budget | PASS |  |
| 9 Downstream | oracle allocation beats best random targeting at every budget | PASS |  |
| 9 Downstream | structured model-based allocation (observed data only) beats best random targeting at every budget (true value) | PASS | [0.37, 0.42, 0.45] of oracle gain |
| 9 Downstream | generic DR-profit-CATE allocation beats best random targeting at every budget (true value) | PASS | [0.16, 0.22, 0.11] of oracle gain |
| 9 Downstream | structured allocation never exceeds oracle gain (sanity) | PASS |  |

## 1. Structure

Main file: 20,000 rows × 45 columns. Ground-truth file: 20,000 rows × 69 columns.

| index | dtype | n_missing | n_unique |
|---|---|---|---|
| customer_id | int64 | 0 | 20000 |
| customer_segment | str | 0 | 6 |
| age | int64 | 0 | 57 |
| income_bracket | str | 0 | 4 |
| urbanicity | str | 0 | 3 |
| region | str | 0 | 5 |
| tenure_months | int64 | 0 | 120 |
| historical_spend | float64 | 0 | 19244 |
| recency_days | int64 | 0 | 177 |
| browsing_frequency | int64 | 0 | 26 |
| avg_session_duration_sec | float64 | 0 | 5022 |
| cart_abandonment_rate | float64 | 0 | 886 |
| support_tickets_count | int64 | 0 | 9 |
| email_open_rate | float64 | 0 | 934 |
| previous_discounts_accepted | int64 | 0 | 13 |
| previous_discount_offers_received | int64 | 0 | 17 |
| previous_discount_response_rate | float64 | 0 | 58 |
| previous_purchases_12m | int64 | 0 | 22 |
| avg_order_value | float64 | 0 | 19076 |
| website_visits_30d | int64 | 0 | 29 |
| mobile_app_user | int64 | 0 | 2 |
| loyalty_member | int64 | 0 | 2 |
| subscription_status | str | 0 | 3 |
| preferred_channel | str | 0 | 4 |
| product_category | str | 0 | 5 |
| campaign_type | str | 0 | 4 |
| day_of_week | str | 0 | 7 |
| device_type | str | 0 | 3 |
| competitor_price_index | float64 | 0 | 726 |
| stock_availability | str | 0 | 2 |
| seasonality_index | float64 | 0 | 601 |
| customer_satisfaction | float64 | 0 | 348 |
| historical_return_rate | float64 | 0 | 611 |
| days_since_last_campaign | int64 | 0 | 168 |
| marketing_contact_count_30d | int64 | 0 | 10 |
| treatment_arm | int64 | 0 | 4 |
| discount_pct | int64 | 0 | 4 |
| discount_rate | float64 | 0 | 4 |
| purchased | int64 | 0 | 2 |
| basket_value_before_discount | float64 | 0 | 19155 |
| discount_amount | float64 | 0 | 3774 |
| revenue | float64 | 0 | 6431 |
| cost_of_goods | float64 | 0 | 6386 |
| marketing_cost | float64 | 0 | 8 |
| net_profit | float64 | 0 | 6286 |

## 2. Treatment counts and outcome distributions

| discount_pct | n | share |
|---|---|---|
| 0.0000 | 7584.0000 | 0.3792 |
| 5.0000 | 5395.0000 | 0.2697 |
| 10.0000 | 4266.0000 | 0.2133 |
| 20.0000 | 2755.0000 | 0.1378 |

Observed (confounded) outcomes by arm:

| discount_pct | n | purchase_rate | mean_basket_list | mean_discount_amount | mean_revenue | mean_cogs | mean_marketing_cost | mean_net_profit |
|---|---|---|---|---|---|---|---|---|
| 0.000 | 7584.000 | 0.316 | 1279.900 | 0.000 | 400.225 | 236.176 | 4.978 | 159.071 |
| 5.000 | 5395.000 | 0.310 | 1302.569 | 20.449 | 388.535 | 243.172 | 10.986 | 134.378 |
| 10.000 | 4266.000 | 0.323 | 1319.131 | 41.593 | 374.340 | 248.102 | 11.027 | 115.211 |
| 20.000 | 2755.000 | 0.397 | 1297.608 | 97.085 | 388.342 | 294.152 | 11.033 | 83.157 |

Distributions (INR):

| index | count | mean | std | min | 5% | 25% | 50% | 75% | 95% | max |
|---|---|---|---|---|---|---|---|---|---|---|
| basket_value_before_discount | 20000.00 | 1296.82 | 1039.25 | 100.00 | 323.43 | 629.09 | 1012.60 | 1628.99 | 3234.89 | 15000.00 |
| revenue (buyers) | 6543.00 | 1191.85 | 948.59 | 90.00 | 299.39 | 586.30 | 932.90 | 1483.64 | 2936.14 | 14250.00 |
| net_profit (all) | 20000.00 | 132.60 | 285.15 | -171.87 | -14.00 | -10.00 | -6.00 | 190.04 | 692.22 | 4329.33 |
| net_profit (buyers) | 6543.00 | 423.17 | 350.78 | -171.87 | 77.32 | 197.02 | 329.66 | 541.89 | 1089.87 | 4329.33 |

## 3. Financial consistency

Share of purchases with negative gross profit (revenue < COGS): 0.0034; by arm: {0: 0.0, 5: 0.0, 10: 0.0, 20: 0.0201}.

| index | observed_mean_cogs_ratio | design_base_ratio |
|---|---|---|
| beauty | 0.4223 | 0.4200 |
| electronics | 0.7424 | 0.7400 |
| fashion | 0.5210 | 0.5200 |
| home | 0.6016 | 0.6000 |
| sports | 0.5812 | 0.5800 |

## 4. Treatment overlap

True assignment propensities (ground truth):

| index | min | p1 | p5 | median | p95 | max | share<0.02 | share<0.05 |
|---|---|---|---|---|---|---|---|---|
| arm0 (0%) | 0.1210 | 0.1452 | 0.1760 | 0.3813 | 0.5820 | 0.6413 | 0.0000 | 0.0000 |
| arm1 (5%) | 0.2197 | 0.2300 | 0.2405 | 0.2746 | 0.2847 | 0.2848 | 0.0000 | 0.0000 |
| arm2 (10%) | 0.0966 | 0.1074 | 0.1200 | 0.2077 | 0.3055 | 0.3299 | 0.0000 | 0.0000 |
| arm3 (20%) | 0.0424 | 0.0491 | 0.0573 | 0.1284 | 0.2573 | 0.3134 | 0.0000 | 0.0136 |

Cross-fitted multinomial-logit estimated propensities (features only):

| index | min | p1 | corr_with_true |
|---|---|---|---|
| arm0 | 0.0582 | 0.1217 | 0.9729 |
| arm1 | 0.1307 | 0.1891 | 0.4405 |
| arm2 | 0.0361 | 0.0793 | 0.9153 |
| arm3 | 0.0110 | 0.0306 | 0.9222 |

Estimated-propensity accuracy 0.392 vs. majority-class 0.379; one-vs-rest AUC 0.598.


Customers per segment and arm:

| customer_segment | 0 | 5 | 10 | 20 |
|---|---|---|---|---|
| bargain_hunter | 1246 | 956 | 781 | 527 |
| lapsing | 664 | 651 | 623 | 434 |
| loyal_core | 1856 | 1090 | 712 | 401 |
| new_customer | 734 | 556 | 429 | 288 |
| premium | 850 | 514 | 402 | 253 |
| regular | 2234 | 1628 | 1319 | 852 |

## 5. Confounding

Correlation of assignment score with baseline purchase probability P(Y(0)=1): -0.677; with true effect size β: 0.618.

Mean covariates by observed arm (differences = confounding):

| discount_pct | recency_days | cart_abandonment_rate | email_open_rate | previous_discount_response_rate | loyalty_member | competitor_price_index | previous_purchases_12m | win_back_share | premium_segment_share |
|---|---|---|---|---|---|---|---|---|---|
| 0.000 | 18.963 | 0.374 | 0.521 | 0.404 | 0.491 | 1.000 | 5.178 | 0.111 | 0.112 |
| 5.000 | 24.319 | 0.412 | 0.479 | 0.439 | 0.412 | 0.987 | 4.821 | 0.150 | 0.095 |
| 10.000 | 28.444 | 0.433 | 0.451 | 0.470 | 0.355 | 0.979 | 4.592 | 0.160 | 0.094 |
| 20.000 | 32.115 | 0.456 | 0.431 | 0.487 | 0.341 | 0.973 | 4.386 | 0.179 | 0.092 |

Covariate balance (standardized mean differences, arm k vs 0, all one-hot/numeric covariates):

| index | arm1 vs 0 | arm2 vs 0 | arm3 vs 0 |
|---|---|---|---|
| max|SMD| raw | 0.226 | 0.408 | 0.525 |
| mean|SMD| raw | 0.053 | 0.085 | 0.108 |
| # covariates |SMD|>0.1 raw | 13.000 | 19.000 | 21.000 |
| max|SMD| IPW(true e) | 0.036 | 0.034 | 0.054 |
| mean|SMD| IPW(true e) | 0.012 | 0.012 | 0.018 |
| # covariates |SMD|>0.1 IPW | 0.000 | 0.000 | 0.000 |

Most imbalanced covariates (arm 20% vs 0%):

| index | |SMD| arm3 vs arm0 (raw) | |SMD| after IPW |
|---|---|---|
| log1p_recency | 0.525 | 0.018 |
| cart_abandonment_rate | 0.448 | 0.010 |
| email_open_rate | 0.447 | 0.016 |
| recency_days | 0.442 | 0.003 |
| loyalty_member | 0.307 | 0.009 |
| previous_purchases_12m | 0.277 | 0.032 |
| website_visits_30d | 0.274 | 0.006 |
| previous_discount_response_rate | 0.274 | 0.010 |

Effect recovery — naive vs adjusted estimators (truth from ground-truth file):

| outcome | arm | true ATE | naive diff | naive bias | IPW (true e) | SE | err/SE | IPW (est e) | SE  | err/SE  |
|---|---|---|---|---|---|---|---|---|---|---|
| purchase prob. | 5% vs 0% | 0.0382 | -0.0068 | -0.0449 | 0.0483 | 0.0081 | 1.2386 | 0.0479 | 0.0082 | 1.1879 |
| purchase prob. | 10% vs 0% | 0.0788 | 0.0072 | -0.0716 | 0.0859 | 0.0093 | 0.7677 | 0.0869 | 0.0095 | 0.8548 |
| purchase prob. | 20% vs 0% | 0.1534 | 0.0808 | -0.0726 | 0.1459 | 0.0117 | -0.6390 | 0.1512 | 0.0121 | -0.1850 |
| net profit (INR) | 5% vs 0% | -8.2990 | -24.6928 | -16.3938 | 2.5760 | 5.3733 | 2.0239 | 3.1604 | 5.4901 | 2.0873 |
| net profit (INR) | 10% vs 0% | -14.3747 | -43.8597 | -29.4851 | -6.1852 | 5.6420 | 1.4515 | -7.3860 | 5.6216 | 1.2432 |
| net profit (INR) | 20% vs 0% | -42.1919 | -75.9134 | -33.7215 | -45.4469 | 5.1279 | -0.6348 | -44.0143 | 5.3212 | -0.3425 |

## 6. Heterogeneous ground-truth effects

| index | mean | std | min | 5% | 50% | 95% | max | share<0 | share<-0.01 |
|---|---|---|---|---|---|---|---|---|---|
| ITE purchase 5% | 0.0382 | 0.0477 | -0.0174 | -0.0109 | 0.0231 | 0.1379 | 0.3119 | 0.2099 | 0.0612 |
| ITE purchase 10% | 0.0788 | 0.1011 | -0.0448 | -0.0282 | 0.0465 | 0.2899 | 0.5839 | 0.2359 | 0.1537 |
| ITE purchase 20% | 0.1534 | 0.2054 | -0.1286 | -0.0816 | 0.0900 | 0.5692 | 0.8817 | 0.2790 | 0.2386 |

| index | mean | std | min | 5% | 50% | 95% | max | share>0 |
|---|---|---|---|---|---|---|---|---|
| Δ profit 5% | -8.299 | 32.135 | -343.533 | -57.455 | -6.136 | 35.434 | 379.423 | 0.320 |
| Δ profit 10% | -14.375 | 63.797 | -689.082 | -114.328 | -8.330 | 70.701 | 779.881 | 0.355 |
| Δ profit 20% | -42.192 | 118.122 | -1611.507 | -237.684 | -22.137 | 100.434 | 1203.096 | 0.301 |

Mean true effect by customer segment (purchase probability; incremental expected net profit in INR):

| segment | true_ite_purchase_arm1 | true_ite_purchase_arm2 | true_ite_purchase_arm3 | true_incr_net_profit_arm1 | true_incr_net_profit_arm2 | true_incr_net_profit_arm3 |
|---|---|---|---|---|---|---|
| bargain_hunter | 0.075 | 0.161 | 0.323 | 9.370 | 19.959 | 11.972 |
| lapsing | 0.038 | 0.086 | 0.199 | 1.976 | 8.915 | 9.756 |
| loyal_core | 0.026 | 0.048 | 0.075 | -22.839 | -44.839 | -102.364 |
| new_customer | 0.033 | 0.071 | 0.148 | -3.477 | -3.368 | -15.140 |
| premium | 0.005 | 0.005 | -0.010 | -31.257 | -60.239 | -124.018 |
| regular | 0.037 | 0.076 | 0.147 | -6.757 | -11.323 | -35.261 |

| c | mean ITE purchase @10% |
|---|---|
| loyalty | 0.0549 |
| new_product | 0.0725 |
| seasonal | 0.0854 |
| win_back | 0.1085 |

| l | mean ITE purchase @10% |
|---|---|
| 0.0000 | 0.1005 |
| 1.0000 | 0.0489 |

Non-monotone dose-response (shares of customers):

| index | share of customers |
|---|---|
| share p(20%) < p(10%) | 0.3172 |
| share p(10%) < p(5%) | 0.2571 |
| share p(5%) < p(0%) | 0.2099 |

Profit-optimal arm (oracle, expected net profit):

| index | share |
|---|---|
| arm0 (0%) | 0.6216 |
| arm1 (5%) | 0.0236 |
| arm2 (10%) | 0.1223 |
| arm3 (20%) | 0.2325 |

CV R² of a gradient-boosted model predicting the true ITE@20% from observed covariates: 0.884.


## 7. Leakage

Largest |Pearson r| between numeric covariates and treatment/outcomes:

| index | |r| |
|---|---|
| historical_spend ~ revenue | 0.358 |
| avg_order_value ~ revenue | 0.308 |
| historical_spend ~ net_profit | 0.305 |
| email_open_rate ~ purchased | 0.252 |
| avg_order_value ~ net_profit | 0.232 |
| previous_purchases_12m ~ net_profit | 0.224 |

Covariate-only cross-validated AUC for `purchased`: 0.741. Treatment re-draw test: 69.7% of customers received a different arm; covariates and ground truth were unchanged.


## 8. Reproducibility

Fresh-process regeneration SHA-256 — main: `1e38231d42fa87e9de0cebef58ec3b792736c52105c88ea92e1b4cf4104f3f45`; ground truth: `6aa6af1c7e7c057df9c53ccdedf858c3d3108056fddad5456df2a15b3cca5335`. Deliverable files — main: `1e38231d42fa87e9de0cebef58ec3b792736c52105c88ea92e1b4cf4104f3f45`; ground truth: `6aa6af1c7e7c057df9c53ccdedf858c3d3108056fddad5456df2a15b3cca5335`.


## 9. Downstream compatibility smoke tests (scikit-learn / SciPy)

### 9a Cross-fitted doubly-robust (orthogonal / DML-style) ATE and CATE for purchase

| arm | true ATE | DR/DML ATE | 95% CI lo | 95% CI hi | CI covers truth | CATE corr (Pearson) | CATE corr (Spearman) | CATE RMSE | true ITE sd |
|---|---|---|---|---|---|---|---|---|---|
| 5% vs 0% | 0.0382 | 0.0466 | 0.0319 | 0.0613 | True | 0.2764 | 0.2743 | 0.0536 | 0.0477 |
| 10% vs 0% | 0.0788 | 0.0845 | 0.0677 | 0.1014 | True | 0.6138 | 0.6021 | 0.0805 | 0.1011 |
| 20% vs 0% | 0.1534 | 0.1507 | 0.1287 | 0.1727 | True | 0.7739 | 0.8036 | 0.1304 | 0.2054 |

### 9b Propensity-score matching (1:1 nearest neighbour, caliper 0.2 SD of logit, with replacement)

| arm | treated matched | treated total | naive ATT | PSM ATT | true ATT (matched treated) | max|SMD| before | max|SMD| after | mean|SMD| before | mean|SMD| after |
|---|---|---|---|---|---|---|---|---|---|
| 5% vs 0% | 5393 | 5395 | -0.0068 | 0.0604 | 0.0390 | 0.2263 | 0.0588 | 0.0529 | 0.0133 |
| 10% vs 0% | 4266 | 4266 | 0.0072 | 0.1106 | 0.0881 | 0.4085 | 0.0529 | 0.0845 | 0.0155 |
| 20% vs 0% | 2755 | 2755 | 0.0808 | 0.1938 | 0.1981 | 0.5249 | 0.0584 | 0.1079 | 0.0217 |

### 9c Qini area vs random targeting (areas in units of incremental purchases summed over 20,000 customers)

IPW-realised Qini uses only observed data (estimated propensities); true-effect Qini uses ground-truth ITEs (noise-free).

| arm | Qini area (IPW-realised): model | oracle | random | Qini area (true effects): model | oracle  | random  | model / oracle (true) |
|---|---|---|---|---|---|---|---|
| 5% vs 0% | 48.164 | 153.267 | -30.163 | 83.513 | 250.069 | 1.487 | 0.334 |
| 10% vs 0% | 506.897 | 473.708 | -5.408 | 344.007 | 536.952 | 3.849 | 0.641 |
| 20% vs 0% | 872.821 | 1128.604 | 34.281 | 921.143 | 1124.780 | 11.043 | 0.819 |

### 9d SciPy LP budget optimisation

Objective: maximise expected incremental net profit vs 0% discount. Budget: expected incremental promotional spend = expected redeemed discount value + marketing/coupon cost. Gains/spend are evaluated with ground-truth expectations. 'structured' = cross-fitted purchase-probability models × unit economics (COGS ratio by category, channel/coupon cost) estimated from observed columns; 'generic DR' = doubly-robust CATE of realised net profit and spend (noisier, because realised profit is a heavy-tailed single draw). 'best random gain' = best of randomly giving one uniform discount level to a budget-sized random subset.

| index | budget (INR) | oracle gain | oracle spend | oracle ROI | structured gain | structured spend | structured ROI | generic DR gain | generic DR spend | generic DR ROI | best random gain |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 25% of oracle-optimal spend | 162,316 | 219,174 | 162,316 | 1.350 | 80,847 | 133,809 | 0.604 | 34,749 | 149,709 | 0.232 | -46,286 |
| 50% of oracle-optimal spend | 324,632 | 318,793 | 324,632 | 0.982 | 135,245 | 274,712 | 0.492 | 70,387 | 313,379 | 0.225 | -92,572 |
| 100% of oracle-optimal spend | 649,263 | 386,246 | 649,263 | 0.595 | 175,496 | 572,307 | 0.307 | 41,587 | 576,890 | 0.072 | -165,981 |

Reference policies (whole population):

| policy | expected Δ net profit vs 0% (INR, all 20k) | expected Δ promo spend (INR) | ROI = Δprofit / Δspend |
|---|---|---|---|
| no discount (0%) | 0.00 | 0.00 | n/a |
| everyone 5% | -165,980.61 | 517,913.78 | -0.320 |
| everyone 10% | -287,493.36 | 1,008,178.50 | -0.285 |
| everyone 20% | -843,838.20 | 2,230,274.62 | -0.378 |
| oracle unconstrained (best arm per customer) | 386,246.08 | 649,263.34 | 0.595 |

## 10. Fixes made during validation

The first complete validation run of the first generator draft failed 3 of 66 checks, and calibration review before that led to two parameter retunes. Everything below was fixed and the generator and validator were re-run from scratch; the report above reflects only the final run.

1. **Effect calibration (before the first validation run).** The first draft gave 42% of customers a negative true purchase effect at 20% and made "no discount" profit-optimal for 71% of customers. `BETA_INTERCEPT` and `KAPPA_BASE` were retuned (0.30/0.08 -> 0.55/0.05 -> 0.45/0.04). Final shares are in section 6.
2. **Overlap (real DGP weakness).** With the first targeting strength, 10.3% of customers had at least one arm propensity below 0.05 (the 'any arm < 0.05' check failed). `TARGETING_GAMMA` was reduced from [0, .50, .85, 1.15] to [0, .45, .75, .95] and `SCORE_CLIP` from 2.2 to 2.0. Minimum true propensity rose to 0.042 and the share of customers with any propensity below 0.05 fell to 1.4%. Confounding remains strong (section 5).
3. **Validator bug, list-price conservation check.** The tolerance ignored that ground-truth probabilities are stored rounded to 6 decimals (error up to ~INR 0.0075 on a INR 15,000 basket). The tolerance now scales with basket value. No data changed.
4. **IPW recovery criterion changed.** The first draft used an arbitrary absolute threshold (0.015) and failed with a maximum error of 0.0194 under the earlier, sharper targeting (large inverse-propensity weights). A fixed absolute threshold is not a meaningful test, so the criterion was replaced by "within 3 standard errors" (Hajek SE reported in section 5). This was a change of criterion made after seeing a failure; the DGP overlap fix in item 2 was applied independently and separately reduced the weights.
5. **Validator reproducibility.** Gradient-boosting models in the validator had no fixed random state, so downstream numbers changed between runs. All models are now seeded; two consecutive validator runs produce an identical report (verified by diff, ignoring the runtime line).
6. **Added structured allocation policy.** The generic doubly-robust profit CATE recovered only a small share of oracle gain because realised profit is a noisy, heavy-tailed outcome. A structured policy (purchase-probability models x unit economics estimated from observed columns) was added to section 9d; both are reported.
7. **DoWhy / EconML could not be installed** (the sandbox could not resolve the package index), so they were never executed; see the status lines at the top.

## 11. Limitations of this validation

- Passing checks confirms the dataset behaves as the documented DGP specifies; it says nothing about real customers.
- The DR/DML-style, PSM, Qini and LP results come from scikit-learn/SciPy code written for this validation; DoWhy and EconML were not run (see top).
- Estimator recovery thresholds are sanity bounds chosen for this sample size, not formal statistical tests.

