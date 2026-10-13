# Data Generation Methodology — EconoCausal Synthetic Dataset

## 1. Design Overview
One row represents one customer receiving one promotional offer. The generator employs three independent NumPy random streams (`numpy.random.default_rng([seed, stream])` with seed `20261012`):
- **Stream 0**: Pre-treatment covariates X.
- **Stream 1**: Treatment assignment T.
- **Stream 2**: Potential outcomes Y(k), list-price basket values, and COGS noise.

Because streams are decoupled, re-drawing treatment assignments leaves covariates and potential outcome tables unchanged.

## 2. Causal Graph & Backdoor Confounding
- Backdoor Paths: T <- X -> Y. The pricing team intentionally targets higher discounts to customers with higher cart abandonment and longer recency. Because these customers have lower baseline purchase rates, naive differences are severely biased downwards.
- Identification Assumptions:
  1. Consistency / SUTVA: Individual outcomes depend only on individual treatment assignment.
  2. Unconfoundedness: {Y(0), Y(1), Y(2), Y(3)} independent of T given X.
  3. Positivity (Overlap): For all customers i and arms k, P(T_i = k | X_i) > 0.04.

## 3. Potential Outcomes Equations
With discount dose x = discount_pct / 10 in {0, 0.5, 1.0, 2.0}:
logit(P(Y(k) = 1)) = alpha + base(X) + u + beta(X, epsilon) * x_k - kappa(X) * x_k^2
- base(X): Linear combination of standardized customer engagement metrics.
- beta(X, epsilon): Heterogeneous response driven by past discount response rate, segment, cart abandonment, and loyalty status.
- kappa(X): Curvature penalizing extreme discounting (diminishing returns).
