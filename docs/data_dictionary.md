# Data Dictionary — EconoCausal Synthetic Dataset

Files: `econocausal_synthetic_dataset.csv` (20,000 rows × 45 columns, **the only file for model fitting**) and `ground_truth_evaluation.csv` (20,000 rows × 69 columns, **evaluation only**). Both are produced by `generate_dataset.py` (seed `20261012`) and joined on `customer_id`. Row order is identical in both files.

Currency: INR-scaled generic currency units, 2 decimals. Taxes, shipping, returns/refunds and inventory costs are **excluded** (see methodology).

## Column roles

| Role | Meaning | Use as model input? |
|---|---|---|
| `id` | Row key | No |
| `covariate` | Pre-treatment customer / campaign-context variable, measured **before** the discount is assigned | **Yes** (confounders X / effect modifiers) |
| `treatment` | The discount offer assigned to the customer | Treatment variable T only |
| `outcome` | Post-treatment realised result | **Never** as a feature; outcome Y only |
| `ground_truth` | Simulated potential outcomes, true propensities and true effects (separate file) | **Never** (evaluation only) |

Suggested wiring — DoWhy/EconML: `T = treatment_arm` (categorical, 0 is control) or `discount_rate` (continuous); `Y` ? {`purchased`, `revenue`, `net_profit`}; `X`/`W` = all `covariate` columns (use `customer_segment`, `previous_discount_response_rate`, `cart_abandonment_rate`, `loyalty_member`, `product_category`, `campaign_type`, `competitor_price_index` as effect modifiers). `region` was built to shift treatment assignment only (instrument-like); including it as a covariate is harmless. Do **not** pass `discount_pct`/`discount_rate`/`treatment_arm` together as features alongside T.

## Main dataset: `econocausal_synthetic_dataset.csv`

### Identifier
- `customer_id` (int): 100001 … 120000, unique, contiguous. One row per customer-campaign observation.

### Pre-treatment covariates
- `customer_segment` (string): `loyal_core`, `regular`, `bargain_hunter`, `premium`, `lapsing`, `new_customer`.
- `age` (int): 18–75 years.
- `income_bracket` (string): `low`, `lower_middle`, `upper_middle`, `high`.
- `urbanicity` (string): `rural`, `suburban`, `urban`.
- `region` (string): `north`, `south`, `east`, `west`, `central`. Influences discount assignment (pricing practice).
- `tenure_months` (int): 1–120 months.
- `historical_spend` (float): Total spend in prior 12 months (INR).
- `recency_days` (int): Days since last purchase (0–180).
- `browsing_frequency` (int): Weekly browsing sessions (0–25).
- `avg_session_duration_sec` (float): Mean session length (10–1800 sec).
- `cart_abandonment_rate` (float): Past carts abandoned (0.0–1.0).
- `support_tickets_count` (int): Prior support tickets (0–8).
- `email_open_rate` (float): Historical email open rate (0.0–1.0).
- `previous_discounts_accepted` (int): Lifetime discounts redeemed (0–12).
- `previous_discount_offers_received` (int): Lifetime discount offers received (>= accepted).
- `previous_discount_response_rate` (float): `accepted / received` (0.0–1.0).
- `previous_purchases_12m` (int): Orders in prior 12 months (0–25).
- `avg_order_value` (float): Historical average order value (100–8000 INR).
- `website_visits_30d` (int): Site visits in prior 30 days (0–30).
- `mobile_app_user` (int): 1 if customer uses mobile app, 0 otherwise.
- `loyalty_member` (int): 1 if enrolled in loyalty program, 0 otherwise.
- `subscription_status` (string): `none`, `basic`, `premium`.
- `preferred_channel` (string): `email`, `push`, `sms`, `web`.
- `product_category` (string): `fashion`, `home`, `electronics`, `beauty`, `sports`.
- `campaign_type` (string): `seasonal`, `win_back`, `loyalty`, `new_product`.
- `day_of_week` (string): `Mon` … `Sun`.
- `device_type` (string): `mobile`, `desktop`, `tablet`.
- `competitor_price_index` (float): Competitor price / own price (0.6–1.5).
- `stock_availability` (string): `in_stock`, `low_stock`.
- `seasonality_index` (float): Demand seasonality multiplier (0.75–1.35).
- `customer_satisfaction` (float): Latest satisfaction score (1.0–5.0).
- `historical_return_rate` (float): Pre-treatment share of past orders returned (0.0–0.7).
- `days_since_last_campaign` (int): Days since prior campaign (0–180).
- `marketing_contact_count_30d` (int): Contacts in prior 30 days (0–12).

### Treatment
- `treatment_arm` (int): Categorical code: 0 = control (0%), 1 = 5%, 2 = 10%, 3 = 20%.
- `discount_pct` (int): Offered discount in percentage points: 0, 5, 10, 20.
- `discount_rate` (float): `discount_pct / 100` (0.00, 0.05, 0.10, 0.20).

### Outcomes (post-treatment — never features)
- `purchased` (int): 1 if customer bought during campaign window, 0 otherwise.
- `basket_value_before_discount` (float): List-price basket value (100–15,000 INR).
- `discount_amount` (float): Redeemed discount: `round(basket * discount_rate, 2)` if `purchased = 1`, else 0.
- `revenue` (float): `basket - discount_amount` if `purchased = 1`, else 0.
- `cost_of_goods` (float): `round(basket * cogs_ratio, 2)` if `purchased = 1`, else 0.
- `marketing_cost` (float): Delivery channel cost + coupon issue cost (6.00 if `discount_pct > 0`). Incurred by everyone.
- `net_profit` (float): `revenue - cost_of_goods - marketing_cost`.

## Ground-truth file: `ground_truth_evaluation.csv` (evaluation only)
Contains potential outcomes across all 4 arms (`purchased_arm{0..3}`, `revenue_arm{0..3}`, `net_profit_arm{0..3}`), true assignment propensities (`true_propensity_arm{0..3}`), and true individual treatment effects (`true_ite_purchase_arm{1..3}`, `true_incr_net_profit_arm{1..3}`).
