import os
import numpy as np
import pandas as pd

SEED = 20261012
N = 20000

ARM_PCT = np.array([0, 5, 10, 20])
ARM_RATE = ARM_PCT / 100.0
ARM_X = ARM_PCT / 10.0
TARGET_ARM_SHARE = np.array([0.38, 0.27, 0.21, 0.14])
BASELINE_PURCHASE_TARGET = 0.27
TARGETING_GAMMA = np.array([0.0, 0.45, 0.75, 0.95])
SCORE_CLIP = 2.0
BETA_INTERCEPT = 0.45
KAPPA_BASE = 0.04

COGS_RATIO_BASE = {'fashion': 0.52, 'home': 0.60, 'electronics': 0.74, 'beauty': 0.42, 'sports': 0.58}
COGS_RATIO_SIGMA = 0.06
CHANNEL_COST = {'email': 4.0, 'push': 3.0, 'sms': 8.0, 'web': 6.0}
COUPON_ISSUE_COST = 6.0

SEGMENTS = ['loyal_core', 'regular', 'bargain_hunter', 'premium', 'lapsing', 'new_customer']
SEG_P = [0.20, 0.30, 0.18, 0.10, 0.12, 0.10]
SEG_LATENT = {
    'loyal_core': (0.8, -0.3, 0.3),
    'regular': (0.0, 0.0, 0.0),
    'bargain_hunter': (0.0, 1.0, -0.3),
    'premium': (0.3, -0.8, 1.0),
    'lapsing': (-1.0, 0.3, 0.0),
    'new_customer': (0.0, 0.4, -0.2),
}
CAMPAIGN_TYPES = ['seasonal', 'win_back', 'loyalty', 'new_product']
CAMPAIGN_BY_SEG = {
    'loyal_core': [0.30, 0.02, 0.48, 0.20],
    'regular': [0.42, 0.10, 0.18, 0.30],
    'bargain_hunter': [0.45, 0.20, 0.05, 0.30],
    'premium': [0.30, 0.04, 0.30, 0.36],
    'lapsing': [0.30, 0.55, 0.02, 0.13],
    'new_customer': [0.35, 0.05, 0.05, 0.55],
}
CATEGORIES = ['fashion', 'home', 'electronics', 'beauty', 'sports']
CATEGORY_P = [0.26, 0.22, 0.19, 0.18, 0.15]
AOV_CAT_EFFECT = {'fashion': -0.10, 'home': 0.10, 'electronics': 0.35, 'beauty': -0.35, 'sports': -0.05}
RETURN_CAT_FACTOR = {'fashion': 1.5, 'home': 0.8, 'electronics': 0.7, 'beauty': 1.0, 'sports': 1.1}
REGIONS = ['north', 'south', 'east', 'west', 'central']
REGION_T_EFFECT = {'north': 0.0, 'south': 0.0, 'east': 0.25, 'west': -0.15, 'central': 0.0}

FEATURE_COLUMNS = [
    'customer_segment', 'age', 'income_bracket', 'urbanicity', 'region', 'tenure_months',
    'historical_spend', 'recency_days', 'browsing_frequency', 'avg_session_duration_sec',
    'cart_abandonment_rate', 'support_tickets_count', 'email_open_rate',
    'previous_discounts_accepted', 'previous_discount_offers_received', 'previous_discount_response_rate',
    'previous_purchases_12m', 'avg_order_value', 'website_visits_30d', 'mobile_app_user',
    'loyalty_member', 'subscription_status', 'preferred_channel', 'product_category', 'campaign_type',
    'day_of_week', 'device_type', 'competitor_price_index', 'stock_availability', 'seasonality_index',
    'customer_satisfaction', 'historical_return_rate', 'days_since_last_campaign',
    'marketing_contact_count_30d'
]
TREATMENT_COLUMNS = ['treatment_arm', 'discount_pct', 'discount_rate']
OUTCOME_COLUMNS = ['purchased', 'basket_value_before_discount', 'discount_amount', 'revenue',
                   'cost_of_goods', 'marketing_cost', 'net_profit']
MAIN_COLUMNS = ['customer_id'] + FEATURE_COLUMNS + TREATMENT_COLUMNS + OUTCOME_COLUMNS

def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))
def z(x): return (x - x.mean()) / x.std()
def softplus(x): return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)
def pick(rng, labels, prob_matrix):
    cum = np.cumsum(prob_matrix, axis=1)
    u = rng.random(prob_matrix.shape[0])
    idx = (u[:, None] >= cum[:, :-1]).sum(axis=1)
    return np.asarray(labels)[idx]

def make_covariates(rng, n):
    seg = rng.choice(SEGMENTS, size=n, p=SEG_P)
    lat = np.array([SEG_LATENT[s] for s in seg])
    eng = lat[:, 0] + rng.normal(0, 0.85, n)
    ps = lat[:, 1] + rng.normal(0, 0.80, n)
    val = lat[:, 2] + rng.normal(0, 0.90, n)
    category = rng.choice(CATEGORIES, size=n, p=CATEGORY_P)
    region = rng.choice(REGIONS, size=n)
    day = rng.choice(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], size=n)
    age = np.clip(np.round(36 + 2.5 * val - 2.0 * (seg == 'new_customer') + rng.normal(0, 9, n)), 18, 75).astype(int)
    inc_score = 0.8 * val + rng.normal(0, 0.8, n)
    q = np.quantile(inc_score, [0.22, 0.56, 0.85])
    income = np.asarray(['low', 'lower_middle', 'upper_middle', 'high'])[(inc_score[:, None] >= q).sum(axis=1)]
    urb_score = 0.4 * val + rng.normal(0, 1, n)
    q = np.quantile(urb_score, [0.21, 0.52])
    urbanicity = np.asarray(['rural', 'suburban', 'urban'])[(urb_score[:, None] >= q).sum(axis=1)]
    tenure_mean = {'loyal_core': 48, 'regular': 30, 'bargain_hunter': 24, 'premium': 40, 'lapsing': 36}
    tenure = np.empty(n)
    new = seg == 'new_customer'
    tenure[new] = rng.integers(1, 7, new.sum())
    for s, m in tenure_mean.items():
        mask = seg == s
        tenure[mask] = rng.gamma(2.2, m / 2.2, mask.sum())
    tenure = np.clip(np.round(tenure), 1, 120).astype(int)
    n12 = np.clip(rng.poisson(np.exp(1.55 + 0.30 * eng + 0.12 * val - 0.9 * new)), 0, 25).astype(int)
    aov_mu = np.log(1050) + 0.35 * val + pd.Series(category).map(AOV_CAT_EFFECT).to_numpy()
    aov = np.clip(np.exp(aov_mu + rng.normal(0, 0.45, n)), 100, 8000)
    hist_spend = np.where(n12 > 0, n12 * aov * np.exp(rng.normal(-0.02, 0.15, n)), 0.0)
    recency = np.clip(np.round(rng.exponential(22 * np.exp(-0.55 * eng))), 0, 180).astype(int)
    browsing = np.clip(rng.poisson(np.exp(2.0 + 0.22 * eng)), 0, 25).astype(int)
    visits = np.clip(rng.poisson(np.exp(2.1 + 0.20 * eng + 0.03 * (browsing - 8))), 0, 30).astype(int)
    session = np.clip(np.exp(np.log(190) + 0.18 * eng + rng.normal(0, 0.55, n)), 10, 1800)
    cart_ab = np.clip(sigmoid(-0.45 - 0.35 * eng + 0.25 * ps + rng.normal(0, 0.8, n)), 0.001, 0.98)
    app = (rng.random(n) < sigmoid(0.5 + 0.3 * eng - 0.02 * (age - 36))).astype(int)
    ch_p = np.where(app[:, None] == 1, [[0.34, 0.30, 0.20, 0.16]], [[0.42, 0.12, 0.22, 0.24]])
    channel = pick(rng, ['email', 'push', 'sms', 'web'], ch_p)
    email_open = np.clip(sigmoid(-0.2 + 0.5 * eng + 0.2 * (channel == 'email') + rng.normal(0, 0.85, n)), 0.002, 0.98)
    dev_p = np.where(app[:, None] == 1, [[0.80, 0.14, 0.06]], [[0.35, 0.55, 0.10]])
    device = pick(rng, ['mobile', 'desktop', 'tablet'], dev_p)
    sat = np.clip(3.8 + 0.35 * eng + 0.1 * val + rng.normal(0, 0.6, n), 1, 5)
    tickets = np.clip(rng.poisson(np.exp(0.0 - 0.4 * (sat - 3.8))), 0, 8).astype(int)
    accepted = np.clip(rng.poisson(np.exp(0.50 + 0.45 * ps + 0.10 * eng + 0.006 * (tenure - 30))), 0, 12).astype(int)
    received = accepted + rng.poisson(2.2 * np.exp(0.2 * eng))
    resp_rate = np.where(received > 0, accepted / np.maximum(received, 1), 0.0)
    loyalty = (rng.random(n) < sigmoid(-0.9 + 0.7 * eng + 0.4 * val + 0.008 * tenure + 1.0 * (seg == 'loyal_core'))).astype(int)
    sub_score = 0.5 * eng + 0.4 * val + rng.normal(0, 1, n)
    q = np.quantile(sub_score, [0.58, 0.86])
    subscription = np.asarray(['none', 'basic', 'premium'])[(sub_score[:, None] >= q).sum(axis=1)]
    cp = np.array([CAMPAIGN_BY_SEG[s] for s in seg])
    campaign = pick(rng, CAMPAIGN_TYPES, cp)
    cpi = np.clip(1.0 + rng.normal(0, 0.12, n) - 0.06 * (category == 'electronics'), 0.6, 1.5)
    low_stock = rng.random(n) < (0.10 + 0.08 * (category == 'electronics') + 0.03 * (campaign == 'seasonal'))
    seasonality = np.clip(1.03 + 0.10 * (campaign == 'seasonal') + rng.normal(0, 0.14, n), 0.75, 1.35)
    ret = np.clip(rng.beta(2, 12, n) * pd.Series(category).map(RETURN_CAT_FACTOR).to_numpy(), 0, 0.7)
    days_since = np.clip(np.round(rng.exponential(24 * np.exp(-0.10 * eng))), 0, 180).astype(int)
    contacts = np.clip(rng.poisson(np.exp(0.55 + 0.12 * eng + 0.15 * (campaign == 'win_back'))), 0, 12).astype(int)
    return pd.DataFrame({
        'customer_id': np.arange(100001, 100001 + n), 'customer_segment': seg, 'age': age,
        'income_bracket': income, 'urbanicity': urbanicity, 'region': region, 'tenure_months': tenure,
        'historical_spend': np.round(hist_spend, 2), 'recency_days': recency, 'browsing_frequency': browsing,
        'avg_session_duration_sec': np.round(session, 1), 'cart_abandonment_rate': np.round(cart_ab, 3),
        'support_tickets_count': tickets, 'email_open_rate': np.round(email_open, 3),
        'previous_discounts_accepted': accepted, 'previous_discount_offers_received': received.astype(int),
        'previous_discount_response_rate': np.round(resp_rate, 3), 'previous_purchases_12m': n12,
        'avg_order_value': np.round(aov, 2), 'website_visits_30d': visits, 'mobile_app_user': app,
        'loyalty_member': loyalty, 'subscription_status': subscription, 'preferred_channel': channel,
        'product_category': category, 'campaign_type': campaign, 'day_of_week': day, 'device_type': device,
        'competitor_price_index': np.round(cpi, 3), 'stock_availability': np.where(low_stock, 'low_stock', 'in_stock'),
        'seasonality_index': np.round(seasonality, 3), 'customer_satisfaction': np.round(sat, 2),
        'historical_return_rate': np.round(ret, 3), 'days_since_last_campaign': days_since,
        'marketing_contact_count_30d': contacts,
    })

def targeting_score(X):
    s = (0.50 * z(np.log1p(X.recency_days)) + 0.35 * z(X.cart_abandonment_rate) - 0.35 * z(X.email_open_rate)
         + 0.30 * z(X.previous_discount_response_rate) - 0.45 * X.loyalty_member
         - 0.30 * (X.subscription_status == 'premium') - 0.30 * z(X.competitor_price_index)
         + 0.40 * (X.campaign_type == 'win_back') - 0.35 * (X.stock_availability == 'low_stock')
         + 0.15 * z(np.log(X.avg_order_value)) + X.region.map(REGION_T_EFFECT)).to_numpy(dtype=float)
    return SCORE_CLIP * np.tanh(s / SCORE_CLIP)

def assignment_probabilities(X):
    s = targeting_score(X)
    g = TARGETING_GAMMA
    b = np.zeros(4)
    for _ in range(300):
        eta = b[None, :] + g[None, :] * s[:, None]
        p = np.exp(eta - eta.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        b[1:] += 0.7 * np.log(TARGET_ARM_SHARE[1:] / p.mean(0)[1:])
    return p, s

def baseline_logit_raw(X):
    wk = X.day_of_week.isin(['Sat', 'Sun']).to_numpy(float)
    camp = X.campaign_type.map({'loyalty': 0.15, 'new_product': -0.05, 'seasonal': 0.10, 'win_back': -0.25}).to_numpy()
    sub = X.subscription_status.map({'none': 0.0, 'basic': 0.15, 'premium': 0.30}).to_numpy()
    return (0.45 * z(X.email_open_rate) - 0.40 * z(np.log1p(X.recency_days)) + 0.30 * z(X.browsing_frequency)
            - 0.30 * z(X.cart_abandonment_rate) + 0.25 * z(X.previous_purchases_12m) + 0.20 * X.loyalty_member
            + sub + 0.20 * z(X.customer_satisfaction) + 0.8 * (X.competitor_price_index - 1.0)
            - 0.35 * (X.stock_availability == 'low_stock') + 1.0 * (X.seasonality_index - 1.05) + camp
            - 0.06 * np.maximum(X.marketing_contact_count_30d - 3, 0) + 0.08 * wk
            - 0.15 * z(X.historical_return_rate) + 0.10 * z(np.log(X.tenure_months))
            - 0.10 * z(np.log(X.avg_order_value)) + 0.15 * z(X.previous_discount_response_rate)).to_numpy(float)

def effect_params(X, eps):
    inc = X.income_bracket.map({'low': 0, 'lower_middle': 1, 'upper_middle': 2, 'high': 3}).to_numpy(float)
    seg = X.customer_segment
    beta_raw = (BETA_INTERCEPT + 0.45 * z(X.previous_discount_response_rate) + 0.25 * z(X.cart_abandonment_rate)
                - 0.30 * X.loyalty_member - 0.35 * (seg == 'premium') + 0.30 * (seg == 'bargain_hunter')
                + 0.20 * (X.campaign_type == 'win_back') - 1.5 * (X.competitor_price_index - 1.0)
                - 0.15 * (inc - 1.5)).to_numpy(float) + eps
    beta = -0.10 + 0.25 * softplus((beta_raw + 0.10) / 0.25)
    kappa = KAPPA_BASE + 0.02 * (seg == 'premium').to_numpy(float) + 0.02 * (X.loyalty_member == 1).to_numpy(float)
    return beta, kappa

def solve_intercept(raw_plus_u, target):
    lo, hi = -8.0, 8.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if sigmoid(raw_plus_u + mid).mean() < target: lo = mid
        else: hi = mid
    return (lo + hi) / 2

if __name__ == '__main__':
    rng_cov = np.random.default_rng([SEED, 0])
    rng_t = np.random.default_rng([SEED, 1, 0])
    rng_y = np.random.default_rng([SEED, 2])
    X = make_covariates(rng_cov, N)
    P, score = assignment_probabilities(X)
    cum = np.cumsum(P, axis=1)
    u = rng_t.random(N)
    arm = np.minimum((u[:, None] >= cum[:, :-1]).sum(axis=1), 3).astype(int)
    u_base = rng_y.normal(0, 0.5, N)
    eps = rng_y.normal(0, 0.15, N)
    U = rng_y.random(N)
    basket = np.round(np.clip(X.avg_order_value.to_numpy() * np.exp(rng_y.normal(-0.06, 0.35, N)), 100, 15000), 2)
    cogs_ratio = np.clip(X.product_category.map(COGS_RATIO_BASE).to_numpy() * np.exp(rng_y.normal(0, COGS_RATIO_SIGMA, N)), 0.30, 0.90)
    raw = baseline_logit_raw(X) + u_base
    a0 = solve_intercept(raw, BASELINE_PURCHASE_TARGET)
    base = raw + a0
    beta, kappa = effect_params(X, eps)
    logit = base[:, None] + beta[:, None] * ARM_X[None, :] - kappa[:, None] * ARM_X[None, :] ** 2
    p = sigmoid(logit)
    y = (U[:, None] < p).astype(int)
    disc = np.where(y == 1, np.round(basket[:, None] * ARM_RATE[None, :], 2), 0.0)
    rev = np.where(y == 1, np.round(basket[:, None] - disc, 2), 0.0)
    cogs = np.where(y == 1, np.round(basket * cogs_ratio, 2)[:, None], 0.0)
    ch = X.preferred_channel.map(CHANNEL_COST).to_numpy()
    mkt = ch[:, None] + COUPON_ISSUE_COST * (np.arange(4)[None, :] > 0)
    net = np.round(rev - cogs - mkt, 2)
    i = np.arange(N)
    main = X.copy()
    main['treatment_arm'] = arm
    main['discount_pct'] = ARM_PCT[arm]
    main['discount_rate'] = ARM_RATE[arm]
    main['purchased'] = y[i, arm]
    main['basket_value_before_discount'] = basket
    main['discount_amount'] = disc[i, arm]
    main['revenue'] = rev[i, arm]
    main['cost_of_goods'] = cogs[i, arm]
    main['marketing_cost'] = mkt[i, arm]
    main['net_profit'] = net[i, arm]
    main = main[MAIN_COLUMNS]
    os.makedirs('data/raw', exist_ok=True)
    main.to_csv('data/raw/econocausal_synthetic_dataset.csv', index=False)
    print(f'Done: wrote {len(main)} rows to data/raw/econocausal_synthetic_dataset.csv')
