import React from "react";

const PRESETS = {
  balanced: {
    browsing_frequency: 28,
    cart_abandonment_rate: 0.12,
    competitor_price_index: 0.95,
    historical_spend: 420.0,
    loyalty_member: 1,
    previous_discount_response_rate: 0.65,
    recency_days: 4,
    seasonality_index: 1.25,
  },
  loyalist: {
    browsing_frequency: 45,
    cart_abandonment_rate: 0.05,
    competitor_price_index: 1.10,
    historical_spend: 850.0,
    loyalty_member: 1,
    previous_discount_response_rate: 0.30,
    recency_days: 2,
    seasonality_index: 1.40,
  },
  at_risk: {
    browsing_frequency: 10,
    cart_abandonment_rate: 0.45,
    competitor_price_index: 0.85,
    historical_spend: 180.0,
    loyalty_member: 0,
    previous_discount_response_rate: 0.85,
    recency_days: 24,
    seasonality_index: 0.90,
  },
};

export default function CustomerConfounders({ confounders, setConfounders, onEvaluate }) {
  const handleChange = (field, value) => {
    setConfounders((prev) => ({
      ...prev,
      [field]: field === "loyalty_member" ? parseInt(value) : parseFloat(value) || 0,
    }));
  };

  // Radar chart polygon points calculated dynamically from normalized confounders
  const maxVals = {
    browsing_frequency: 50,
    cart_abandonment_rate: 1,
    competitor_price_index: 1.5,
    historical_spend: 1000,
    loyalty_member: 1,
    previous_discount_response_rate: 1,
  };

  const getRadius = (val, max, rMax = 80) => Math.min(Math.max((val / max) * rMax, 15), rMax);

  const angles = [0, 60, 120, 180, 240, 300].map(a => (a * Math.PI) / 180);
  const cx = 130;
  const cy = 120;

  const rValues = [
    getRadius(confounders.browsing_frequency, maxVals.browsing_frequency),
    getRadius(confounders.cart_abandonment_rate, maxVals.cart_abandonment_rate),
    getRadius(confounders.competitor_price_index, maxVals.competitor_price_index),
    getRadius(confounders.historical_spend, maxVals.historical_spend),
    getRadius(confounders.loyalty_member, maxVals.loyalty_member),
    getRadius(confounders.previous_discount_response_rate, maxVals.previous_discount_response_rate),
  ];

  const radarPoints = rValues
    .map((r, i) => `${cx + r * Math.cos(angles[i])},${cy + r * Math.sin(angles[i])}`)
    .join(" ");

  const outerWeb = [80, 55, 30].map(radius =>
    angles.map(a => `${cx + radius * Math.cos(a)},${cy + radius * Math.sin(a)}`).join(" ")
  );

  return (
    <div style={{ display: "grid", gridTemplateColumns: "1.25fr 1fr", gap: "16px" }}>
      {/* Left Column: Form & Sliders */}
      <div className="card" style={{ margin: 0 }}>
        <h2>Customer Confounder Specification</h2>
        <p className="card-subtitle">
          Configure observational customer attributes. These covariates de-bias the causal graph and parameterize heterogeneous treatment effects.
        </p>

        <div style={{ display: "flex", gap: "8px", marginBottom: "18px", flexWrap: "wrap" }}>
          <button className="btn-secondary" onClick={() => setConfounders(PRESETS.balanced)}>
            Default Balanced
          </button>
          <button className="btn-secondary" onClick={() => setConfounders(PRESETS.loyalist)}>
            High-Value Loyalist
          </button>
          <button className="btn-secondary" onClick={() => setConfounders(PRESETS.at_risk)}>
            Price-Sensitive At-Risk
          </button>
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "14px", marginBottom: "20px" }}>
          <div className="form-group">
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <label>Browsing Frequency (monthly)</label>
              <span style={{ fontFamily: "var(--font-mono)", fontWeight: 600 }}>{confounders.browsing_frequency}</span>
            </div>
            <input
              type="range"
              min="0"
              max="60"
              value={confounders.browsing_frequency}
              onChange={(e) => handleChange("browsing_frequency", e.target.value)}
              style={{ accentColor: "#8B5CF6", width: "100%", padding: 0 }}
            />
            <span className="form-hint">Number of product catalog visits</span>
          </div>

          <div className="form-group">
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <label>Cart Abandonment Rate</label>
              <span style={{ fontFamily: "var(--font-mono)", fontWeight: 600 }}>{confounders.cart_abandonment_rate}</span>
            </div>
            <input
              type="range"
              min="0"
              max="1"
              step="0.01"
              value={confounders.cart_abandonment_rate}
              onChange={(e) => handleChange("cart_abandonment_rate", e.target.value)}
              style={{ accentColor: "#8B5CF6", width: "100%", padding: 0 }}
            />
            <span className="form-hint">Ratio of abandoned carts to checkouts</span>
          </div>

          <div className="form-group">
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <label>Competitor Price Index</label>
              <span style={{ fontFamily: "var(--font-mono)", fontWeight: 600 }}>{confounders.competitor_price_index}</span>
            </div>
            <input
              type="range"
              min="0.5"
              max="1.5"
              step="0.01"
              value={confounders.competitor_price_index}
              onChange={(e) => handleChange("competitor_price_index", e.target.value)}
              style={{ accentColor: "#8B5CF6", width: "100%", padding: 0 }}
            />
            <span className="form-hint">Ratio of competitor prices to ours</span>
          </div>

          <div className="form-group">
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <label>Historical Spend ($)</label>
              <span style={{ fontFamily: "var(--font-mono)", fontWeight: 600 }}>{confounders.historical_spend}</span>
            </div>
            <input
              type="range"
              min="0"
              max="1000"
              step="10"
              value={confounders.historical_spend}
              onChange={(e) => handleChange("historical_spend", e.target.value)}
              style={{ accentColor: "#8B5CF6", width: "100%", padding: 0 }}
            />
            <span className="form-hint">Total pre-treatment spending</span>
          </div>

          <div className="form-group">
            <label>Loyalty Tier Member</label>
            <select
              value={confounders.loyalty_member}
              onChange={(e) => handleChange("loyalty_member", e.target.value)}
            >
              <option value={1}>Yes (Active Member)</option>
              <option value={0}>No (Non-Member)</option>
            </select>
            <span className="form-hint">Membership subscription status</span>
          </div>

          <div className="form-group">
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <label>Prior Discount Response Rate</label>
              <span style={{ fontFamily: "var(--font-mono)", fontWeight: 600 }}>{confounders.previous_discount_response_rate}</span>
            </div>
            <input
              type="range"
              min="0"
              max="1"
              step="0.01"
              value={confounders.previous_discount_response_rate}
              onChange={(e) => handleChange("previous_discount_response_rate", e.target.value)}
              style={{ accentColor: "#8B5CF6", width: "100%", padding: 0 }}
            />
            <span className="form-hint">Historical elasticity to promotional coupons</span>
          </div>

          <div className="form-group">
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <label>Recency Days</label>
              <span style={{ fontFamily: "var(--font-mono)", fontWeight: 600 }}>{confounders.recency_days}</span>
            </div>
            <input
              type="range"
              min="1"
              max="60"
              value={confounders.recency_days}
              onChange={(e) => handleChange("recency_days", e.target.value)}
              style={{ accentColor: "#8B5CF6", width: "100%", padding: 0 }}
            />
            <span className="form-hint">Days elapsed since last transaction</span>
          </div>

          <div className="form-group">
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <label>Seasonality Index</label>
              <span style={{ fontFamily: "var(--font-mono)", fontWeight: 600 }}>{confounders.seasonality_index}</span>
            </div>
            <input
              type="range"
              min="0.5"
              max="2.0"
              step="0.05"
              value={confounders.seasonality_index}
              onChange={(e) => handleChange("seasonality_index", e.target.value)}
              style={{ accentColor: "#8B5CF6", width: "100%", padding: 0 }}
            />
            <span className="form-hint">Demand multiplier for current fiscal cycle</span>
          </div>
        </div>

        <div style={{ display: "flex", gap: "10px" }}>
          <button className="btn-primary" onClick={onEvaluate}>
            Submit Confounders & Evaluate Uplift →
          </button>
          <button className="btn-secondary" onClick={() => setConfounders(PRESETS.balanced)}>
            Reset to Default
          </button>
        </div>
      </div>

      {/* Right Column: Radar Chart + Preview Table */}
      <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
        {/* Real-time Covariate Radar */}
        <div className="card" style={{ margin: 0, padding: "18px" }}>
          <h2 style={{ fontSize: "13px", marginBottom: "8px" }}>Real-time Covariate Distribution</h2>
          <div style={{ background: "#080C14", borderRadius: "6px", border: "1px solid var(--border-subtle)", display: "flex", justifyContent: "center", padding: "10px" }}>
            <svg viewBox="0 0 260 240" style={{ width: "240px", height: "210px" }}>
              {/* Outer Webs */}
              {outerWeb.map((pts, i) => (
                <polygon key={i} points={pts} fill="none" stroke="#1E293B" strokeWidth="1" />
              ))}

              {/* Axis lines */}
              {angles.map((a, i) => (
                <line
                  key={i}
                  x1={cx}
                  y1={cy}
                  x2={cx + 80 * Math.cos(a)}
                  y2={cy + 80 * Math.sin(a)}
                  stroke="#1E293B"
                  strokeWidth="1"
                />
              ))}

              {/* Dynamic Customer Polygon */}
              <polygon
                points={radarPoints}
                fill="rgba(139, 92, 246, 0.18)"
                stroke="#8B5CF6"
                strokeWidth="2"
              />

              {/* Axis labels */}
              {[
                { label: "Browse", a: angles[0] },
                { label: "Cart", a: angles[1] },
                { label: "Price", a: angles[2] },
                { label: "Spend", a: angles[3] },
                { label: "Loyalty", a: angles[4] },
                { label: "Promo", a: angles[5] },
              ].map((item, i) => (
                <text
                  key={i}
                  x={cx + 98 * Math.cos(item.a)}
                  y={cy + 98 * Math.sin(item.a) + 3}
                  fill="#64748B"
                  fontSize="9"
                  textAnchor="middle"
                  fontFamily="var(--font-sans)"
                >
                  {item.label}
                </text>
              ))}
            </svg>
          </div>
        </div>

        {/* Active Confounder State Preview */}
        <div className="card" style={{ margin: 0, padding: "16px" }}>
          <h2 style={{ fontSize: "13px", marginBottom: "8px" }}>Active Confounder State Preview</h2>
          <div className="table-container">
            <table>
              <thead>
                <tr>
                  <th>Confounder</th>
                  <th>Value</th>
                  <th>Feature Class</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(confounders).slice(0, 6).map(([key, val]) => (
                  <tr key={key}>
                    <td style={{ fontFamily: "var(--font-mono)", fontSize: "11px" }}>{key}</td>
                    <td style={{ fontSize: "11px" }}><strong>{val}</strong></td>
                    <td>
                      <span className="badge badge-neutral" style={{ fontSize: "9px" }}>
                        {key === "loyalty_member" ? "Categorical" : "Continuous"}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}
