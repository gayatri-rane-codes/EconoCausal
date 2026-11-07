import React, { useState, useEffect } from "react";

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

const PRESETS = {
  default: {
    browsing_frequency: 12.0,
    cart_abandonment_rate: 0.25,
    competitor_price_index: 1.05,
    historical_spend: 150.0,
    loyalty_member: 1,
    previous_discount_response_rate: 0.35,
    recency_days: 14.0,
    seasonality_index: 1.10
  },
  high_value: {
    browsing_frequency: 28.0,
    cart_abandonment_rate: 0.12,
    competitor_price_index: 0.95,
    historical_spend: 420.0,
    loyalty_member: 1,
    previous_discount_response_rate: 0.65,
    recency_days: 4.0,
    seasonality_index: 1.25
  },
  at_risk: {
    browsing_frequency: 4.0,
    cart_abandonment_rate: 0.70,
    competitor_price_index: 1.20,
    historical_spend: 60.0,
    loyalty_member: 0,
    previous_discount_response_rate: 0.15,
    recency_days: 45.0,
    seasonality_index: 0.90
  }
};

const DISCOUNT_ARMS = [
  { arm: 0, label: "Arm 0 (0%)", rate: 0.00, desc: "Control Baseline" },
  { arm: 1, label: "Arm 1 (5%)", rate: 0.05, desc: "Low Discount" },
  { arm: 2, label: "Arm 2 (10%)", rate: 0.10, desc: "Moderate Discount" },
  { arm: 3, label: "Arm 3 (20%)", rate: 0.20, desc: "Aggressive Promo" }
];

export default function App() {
  const [activeTab, setActiveTab] = useState("overview");
  const [backendHealth, setBackendHealth] = useState({ status: "checking", details: null });

  // Customer & Uplift State
  const [customer, setCustomer] = useState({ ...PRESETS.default });
  const [cateResult, setCateResult] = useState(null);
  const [cateLoading, setCateLoading] = useState(false);
  const [cateError, setCateError] = useState(null);

  // Optimization State
  const [budget, setBudget] = useState(500);
  const [solverMode, setSolverMode] = useState("bounded_gap");
  const [optResult, setOptResult] = useState(null);
  const [optLoading, setOptLoading] = useState(false);
  const [optError, setOptError] = useState(null);

  // Drift State
  const [driftAlpha, setDriftAlpha] = useState(0.05);
  const [driftThreshold, setDriftThreshold] = useState(0.33);
  const [driftResult, setDriftResult] = useState(null);
  const [driftLoading, setDriftLoading] = useState(false);
  const [driftError, setDriftError] = useState(null);

  const checkHealth = async () => {
    try {
      const res = await fetch(`${API_BASE}/health`);
      if (res.ok) {
        const data = await res.json();
        setBackendHealth({ status: "healthy", details: data });
      } else {
        setBackendHealth({ status: "unhealthy", details: `HTTP ${res.status}` });
      }
    } catch (err) {
      setBackendHealth({ status: "offline", details: err.message });
    }
  };

  useEffect(() => {
    checkHealth();
  }, []);

  const handleCustomerChange = (key, val) => {
    setCustomer(prev => ({
      ...prev,
      [key]: key === "loyalty_member" ? parseInt(val, 10) : parseFloat(val) || 0
    }));
  };

  const applyPreset = (key) => {
    if (PRESETS[key]) {
      setCustomer({ ...PRESETS[key] });
      setCateResult(null);
      setCateError(null);
    }
  };

  const runPrediction = async () => {
    setCateLoading(true);
    setCateError(null);
    try {
      const res = await fetch(`${API_BASE}/causal/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ customers: [customer] })
      });
      if (!res.ok) {
        throw new Error(`Inference engine failed with HTTP ${res.status}`);
      }
      const data = await res.json();
      if (data && data.recommendations && data.recommendations.length > 0) {
        setCateResult(data.recommendations[0]);
      } else {
        throw new Error("No recommendations payload returned by backend.");
      }
    } catch (err) {
      setCateError(err.message);
    } finally {
      setCateLoading(false);
    }
  };

  const runOptimization = async () => {
    setOptLoading(true);
    setOptError(null);
    try {
      const cohort = [
        customer,
        { ...PRESETS.default, historical_spend: 320.0, browsing_frequency: 24.0, loyalty_member: 1 },
        { ...PRESETS.high_value, historical_spend: 550.0 },
        { ...PRESETS.at_risk, historical_spend: 95.0 },
        { ...PRESETS.default, historical_spend: 210.0, browsing_frequency: 8.0 },
        { ...PRESETS.high_value, historical_spend: 380.0, recency_days: 12.0 }
      ];

      const res = await fetch(`${API_BASE}/optimization/pipeline`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          customers: cohort,
          budget: parseFloat(budget) || 0,
          mode: solverMode,
          include_allocations: true
        })
      });
      if (!res.ok) {
        throw new Error(`Optimization pipeline returned HTTP ${res.status}`);
      }
      const data = await res.json();
      setOptResult(data);
    } catch (err) {
      setOptError(err.message);
    } finally {
      setOptLoading(false);
    }
  };

  const runDriftDetection = async () => {
    setDriftLoading(true);
    setDriftError(null);
    try {
      const referenceCohort = [PRESETS.default, PRESETS.high_value, PRESETS.at_risk];
      const productionCohort = [
        { ...PRESETS.default, historical_spend: 480.0, browsing_frequency: 32.0 },
        { ...PRESETS.high_value, competitor_price_index: 1.45, cart_abandonment_rate: 0.65 },
        { ...PRESETS.at_risk, recency_days: 90.0, seasonality_index: 0.60 }
      ];

      const res = await fetch(`${API_BASE}/monitoring/data-drift`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          reference_data: referenceCohort,
          current_data: productionCohort,
          alpha: parseFloat(driftAlpha) || 0.05,
          drift_share_threshold: parseFloat(driftThreshold) || 0.33
        })
      });
      if (!res.ok) {
        throw new Error(`Statistical drift test failed with HTTP ${res.status}`);
      }
      const data = await res.json();
      setDriftResult(data);
    } catch (err) {
      setDriftError(err.message);
    } finally {
      setDriftLoading(false);
    }
  };

  const downloadOptReport = () => {
    if (!optResult) return;
    const blob = new Blob([JSON.stringify(optResult, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `econocausal_optimization_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="app-container">
      {/* Sidebar Navigation */}
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-logo">E</div>
          <div className="brand-text">
            <h1>EconoCausal</h1>
            <span>Analytics Suite</span>
          </div>
        </div>

        <nav className="nav-links">
          <button
            className={`nav-item ${activeTab === "overview" ? "active" : ""}`}
            onClick={() => setActiveTab("overview")}
          >
            <span>📊</span> Executive Overview
          </button>
          <button
            className={`nav-item ${activeTab === "customer" ? "active" : ""}`}
            onClick={() => setActiveTab("customer")}
          >
            <span>👤</span> Customer Confounders
          </button>
          <button
            className={`nav-item ${activeTab === "uplift" ? "active" : ""}`}
            onClick={() => setActiveTab("uplift")}
          >
            <span>🎯</span> CATE Recommendations
          </button>
          <button
            className={`nav-item ${activeTab === "optimization" ? "active" : ""}`}
            onClick={() => setActiveTab("optimization")}
          >
            <span>⚖️</span> Budget Allocation
          </button>
          <button
            className={`nav-item ${activeTab === "drift" ? "active" : ""}`}
            onClick={() => setActiveTab("drift")}
          >
            <span>🛡️</span> Drift & Observability
          </button>
        </nav>

        <div className="sidebar-footer">
          <div style={{ color: "var(--text-muted)", fontWeight: 600 }}>EconoCausal Enterprise</div>
          <div>DML · HiGHS MILP · SciPy</div>
          <div style={{ marginTop: "4px" }}>API: {API_BASE}</div>
        </div>
      </aside>

      {/* Main Content Area */}
      <div className="main-wrapper">
        {/* Top Navigation */}
        <header className="top-bar">
          <div className="top-bar-title">
            {activeTab === "overview" && "Executive Analytics & Pipeline Status"}
            {activeTab === "customer" && "Customer Feature Profiling & Confounder Specification"}
            {activeTab === "uplift" && "Heterogeneous Treatment Effect (CATE) Evaluation"}
            {activeTab === "optimization" && "Prescriptive Marketing Budget Allocation Workspace"}
            {activeTab === "drift" && "Continuous Distribution Drift Monitoring"}
          </div>

          <div className="top-bar-actions">
            <div className="health-badge">
              <span className={`status-dot ${backendHealth.status === "healthy" ? "healthy" : "down"}`} />
              <span>Backend: {backendHealth.status.toUpperCase()}</span>
            </div>
            <button className="btn-secondary" onClick={checkHealth} title="Ping GET /health">
              ↻ Refresh
            </button>
          </div>
        </header>

        {/* Tab Views */}
        <main className="content-area">
          {/* TAB 1: EXECUTIVE OVERVIEW */}
          {activeTab === "overview" && (
            <div>
              <div className="grid-4">
                <div className="metric-box">
                  <div className="metric-value">{backendHealth.status === "healthy" ? "READY" : "OFFLINE"}</div>
                  <div className="metric-label">Inference Engine Health</div>
                  <div className="metric-sub">FastAPI Microservice Status</div>
                </div>

                <div className="metric-box">
                  <div className="metric-value">4 Promotional Arms</div>
                  <div className="metric-label">Treatment Policy Domain</div>
                  <div className="metric-sub">0% (Control), 5%, 10%, 20%</div>
                </div>

                <div className="metric-box">
                  <div className="metric-value">
                    {optResult?.metrics?.total_budget_spent !== undefined
                      ? "$" + optResult.metrics.total_budget_spent.toFixed(2)
                      : "Unallocated"}
                  </div>
                  <div className="metric-label">Active Optimized Spend</div>
                  <div className="metric-sub">
                    {optResult ? `Cohort of ${optResult.total_customers} customers` : "Run Budget Allocation tab"}
                  </div>
                </div>

                <div className="metric-box">
                  <div className="metric-value">
                    {driftResult ? (
                      driftResult.overall_drift_detected ? (
                        <span style={{ color: "var(--accent-rose)" }}>DRIFT DETECTED</span>
                      ) : (
                        <span style={{ color: "var(--accent-emerald)" }}>STABLE</span>
                      )
                    ) : (
                      "Not Audited"
                    )}
                  </div>
                  <div className="metric-label">Dataset Observability</div>
                  <div className="metric-sub">Two-Sample KS & Chi-Square</div>
                </div>
              </div>

              <div className="card">
                <h2>Causal AI Decision Pipeline Architecture</h2>
                <p className="card-subtitle">
                  EconoCausal combines Double Machine Learning (DML) and Mixed-Integer Linear Programming (MILP) to identify true personalized price elasticity and allocate finite promotional subsidies without confounding bias.
                </p>

                <div className="grid-2">
                  <div style={{ background: "var(--bg-card)", padding: "20px", borderRadius: "10px", border: "1px solid var(--border)" }}>
                    <div style={{ fontWeight: 600, color: "#fff", marginBottom: "8px", fontSize: "14px" }}>
                      1. Causal De-biasing & CATE Estimation
                    </div>
                    <div style={{ color: "var(--text-muted)", fontSize: "13px", lineHeight: "1.6" }}>
                      Controls for 8 observational customer confounders (recency, browsing velocity, cart abandonment, competitor price index) using orthogonalized residualization to eliminate treatment assignment bias.
                    </div>
                  </div>

                  <div style={{ background: "var(--bg-card)", padding: "20px", borderRadius: "10px", border: "1px solid var(--border)" }}>
                    <div style={{ fontWeight: 600, color: "#fff", marginBottom: "8px", fontSize: "14px" }}>
                      2. Prescriptive Multi-Treatment Knapsack
                    </div>
                    <div style={{ color: "var(--text-muted)", fontSize: "13px", lineHeight: "1.6" }}>
                      Applies the exact HiGHS MILP solver to assign the optimal discount arm (0%, 5%, 10%, or 20%) across cohorts to strictly maximize incremental return under campaign budget bounds.
                    </div>
                  </div>
                </div>
              </div>

              <div className="card">
                <h2>Quick Navigation & Analysis Actions</h2>
                <div style={{ display: "flex", gap: "12px", flexWrap: "wrap" }}>
                  <button className="btn-primary" onClick={() => setActiveTab("customer")}>
                    👤 Configure Customer & Predict Uplift →
                  </button>
                  <button className="btn-secondary" onClick={() => setActiveTab("optimization")}>
                    ⚖️ Prescriptive Optimization Workspace →
                  </button>
                  <button className="btn-secondary" onClick={() => setActiveTab("drift")}>
                    🛡️ Audit Feature Distribution Drift →
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* TAB 2: CUSTOMER INPUT */}
          {activeTab === "customer" && (
            <div>
              <div className="card">
                <h2>Customer Confounder Specification</h2>
                <p className="card-subtitle">
                  Configure observational customer attributes. These covariates de-bias the causal graph and parameterize heterogeneous treatment effects.
                </p>

                <div style={{ display: "flex", gap: "10px", marginBottom: "22px", flexWrap: "wrap" }}>
                  <button className="btn-secondary" onClick={() => applyPreset("default")}>
                    Load Default Balanced
                  </button>
                  <button className="btn-secondary" onClick={() => applyPreset("high_value")}>
                    Load High-Value Loyalist
                  </button>
                  <button className="btn-secondary" onClick={() => applyPreset("at_risk")}>
                    Load Price-Sensitive At-Risk
                  </button>
                </div>

                <div className="form-grid">
                  <div className="form-group">
                    <label>Browsing Frequency (monthly)</label>
                    <input
                      type="number"
                      step="1"
                      min="0"
                      value={customer.browsing_frequency}
                      onChange={e => handleCustomerChange("browsing_frequency", e.target.value)}
                    />
                    <span className="form-hint">Number of product catalog visits</span>
                  </div>

                  <div className="form-group">
                    <label>Cart Abandonment Rate (0.0 to 1.0)</label>
                    <input
                      type="number"
                      step="0.01"
                      min="0"
                      max="1"
                      value={customer.cart_abandonment_rate}
                      onChange={e => handleCustomerChange("cart_abandonment_rate", e.target.value)}
                    />
                    <span className="form-hint">Ratio of abandoned carts to checkouts</span>
                  </div>

                  <div className="form-group">
                    <label>Competitor Price Index</label>
                    <input
                      type="number"
                      step="0.01"
                      min="0.1"
                      value={customer.competitor_price_index}
                      onChange={e => handleCustomerChange("competitor_price_index", e.target.value)}
                    />
                    <span className="form-hint">Ratio of competitor prices to ours (&gt;1 = competitor more expensive)</span>
                  </div>

                  <div className="form-group">
                    <label>Historical Spend ($)</label>
                    <input
                      type="number"
                      step="10"
                      min="0"
                      value={customer.historical_spend}
                      onChange={e => handleCustomerChange("historical_spend", e.target.value)}
                    />
                    <span className="form-hint">Total pre-treatment spending</span>
                  </div>

                  <div className="form-group">
                    <label>Loyalty Tier Member</label>
                    <select
                      value={customer.loyalty_member}
                      onChange={e => handleCustomerChange("loyalty_member", e.target.value)}
                    >
                      <option value={1}>Yes (Active Member)</option>
                      <option value={0}>No (Non-member)</option>
                    </select>
                    <span className="form-hint">Membership subscription status</span>
                  </div>

                  <div className="form-group">
                    <label>Prior Discount Response Rate (0.0 to 1.0)</label>
                    <input
                      type="number"
                      step="0.01"
                      min="0"
                      max="1"
                      value={customer.previous_discount_response_rate}
                      onChange={e => handleCustomerChange("previous_discount_response_rate", e.target.value)}
                    />
                    <span className="form-hint">Historical elasticity to promotional coupons</span>
                  </div>

                  <div className="form-group">
                    <label>Recency Days</label>
                    <input
                      type="number"
                      step="1"
                      min="0"
                      value={customer.recency_days}
                      onChange={e => handleCustomerChange("recency_days", e.target.value)}
                    />
                    <span className="form-hint">Days elapsed since last transaction</span>
                  </div>

                  <div className="form-group">
                    <label>Seasonality Index</label>
                    <input
                      type="number"
                      step="0.05"
                      min="0.1"
                      value={customer.seasonality_index}
                      onChange={e => handleCustomerChange("seasonality_index", e.target.value)}
                    />
                    <span className="form-hint">Demand multiplier for current fiscal cycle</span>
                  </div>
                </div>

                <div style={{ display: "flex", gap: "12px", marginTop: "16px" }}>
                  <button
                    className="btn-primary"
                    disabled={cateLoading}
                    onClick={() => {
                      runPrediction();
                      setActiveTab("uplift");
                    }}
                  >
                    {cateLoading ? "Computing DML CATE..." : "Submit Confounders & Evaluate Uplift →"}
                  </button>
                  <button className="btn-secondary" onClick={() => applyPreset("default")}>
                    Reset to Default
                  </button>
                </div>
              </div>

              {/* Active Profile Summary */}
              <div className="card" style={{ background: "var(--bg-card)" }}>
                <h2>Active Confounder State Preview</h2>
                <div className="table-container">
                  <table>
                    <thead>
                      <tr>
                        <th>Confounder Feature</th>
                        <th>Specified Value</th>
                        <th>Feature Class</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(customer).map(([k, v]) => (
                        <tr key={k}>
                          <td style={{ fontWeight: 600 }}>{k}</td>
                          <td>{typeof v === "number" ? v : String(v)}</td>
                          <td>
                            <span className="badge badge-neutral">
                              {k === "loyalty_member" ? "Categorical / Binary" : "Continuous Confounder"}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}

          {/* TAB 3: CATE RECOMMENDATIONS */}
          {activeTab === "uplift" && (
            <div>
              {cateError && <div className="alert-error">CATE Evaluation Error: {cateError}</div>}

              <div className="card">
                <h2>Heterogeneous Treatment Effect Policy</h2>
                <p className="card-subtitle">
                  Estimates the conditional average treatment effect $\tau_k(X)$ for each promotional arm relative to the zero-discount baseline ($T_0$). The recommended policy selects the arm maximizing expected incremental profit.
                </p>

                <div style={{ display: "flex", gap: "12px", marginBottom: "20px" }}>
                  <button className="btn-primary" disabled={cateLoading} onClick={runPrediction}>
                    {cateLoading ? "Evaluating DML..." : "Re-evaluate Customer Policy"}
                  </button>
                  <button className="btn-secondary" onClick={() => setActiveTab("customer")}>
                    ← Edit Confounders
                  </button>
                </div>

                {cateResult ? (
                  <div>
                    {(() => {
                      const ARMS = [0, 0.05, 0.10, 0.20];
                      const armIdx = (typeof cateResult.recommended_arm === "number" && cateResult.recommended_arm >= 0 && cateResult.recommended_arm < 4)
                        ? cateResult.recommended_arm
                        : null;
                      const discountRate = armIdx !== null ? ARMS[armIdx] : null;
                      const baseSpend = customer?.historical_spend || 0;
                      const promoCost = discountRate !== null ? (baseSpend * discountRate) : null;
                      const expRevenue = discountRate !== null ? (baseSpend * (1 - discountRate)) : null;
                      const upliftProfit = typeof cateResult.expected_incremental_profit === "number"
                        ? cateResult.expected_incremental_profit
                        : null;

                      const cateValues = [
                        { arm: 0, label: "Arm 0 (0%)", cate: 0.0, rate: 0.00 },
                        { arm: 1, label: "Arm 1 (5%)", cate: cateResult.cate_arm_1_vs_0, rate: 0.05 },
                        { arm: 2, label: "Arm 2 (10%)", cate: cateResult.cate_arm_2_vs_0, rate: 0.10 },
                        { arm: 3, label: "Arm 3 (20%)", cate: cateResult.cate_arm_3_vs_0, rate: 0.20 }
                      ];

                      return (
                        <>
                          <div className="grid-4">
                            <div className="metric-box">
                              <div className="metric-value">
                                {discountRate !== null ? (discountRate * 100).toFixed(0) + "%" : "Unavailable"}
                              </div>
                              <div className="metric-label">Optimal Policy (Arm {armIdx !== null ? armIdx : "-"})</div>
                              <div className="metric-sub">Maximizes Incremental Return</div>
                            </div>

                            <div className="metric-box">
                              <div className="metric-value">
                                {upliftProfit !== null ? "$" + upliftProfit.toFixed(2) : "Unavailable"}
                              </div>
                              <div className="metric-label">Expected Incremental Profit</div>
                              <div className="metric-sub">Over Zero-Discount Baseline</div>
                            </div>

                            <div className="metric-box">
                              <div className="metric-value">
                                {expRevenue !== null ? "$" + expRevenue.toFixed(2) : "Unavailable"}
                              </div>
                              <div className="metric-label">Post-Discount Spend Projection</div>
                              <div className="metric-sub">Baseline Spend × (1 - Discount)</div>
                            </div>

                            <div className="metric-box">
                              <div className="metric-value">
                                {promoCost !== null ? "$" + promoCost.toFixed(2) : "Unavailable"}
                              </div>
                              <div className="metric-label">Discount Subsidy Cost</div>
                              <div className="metric-sub">Baseline Spend × Discount Rate</div>
                            </div>
                          </div>

                          {/* Visual CATE Bars */}
                          <div className="card" style={{ background: "var(--bg-card)", marginTop: "16px" }}>
                            <h2>Marginal CATE Uplift Comparison ($\tau_k(X)$ vs $T_0$)</h2>
                            <div style={{ display: "flex", flexDirection: "column", gap: "16px", marginTop: "14px" }}>
                              {cateValues.map(item => {
                                const val = typeof item.cate === "number" ? item.cate : 0;
                                const isBest = item.arm === armIdx;
                                return (
                                  <div key={item.arm} style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
                                    <div style={{ display: "flex", justifyContent: "space-between", fontSize: "13px" }}>
                                      <span style={{ fontWeight: 600, color: isBest ? "var(--text-main)" : "var(--text-muted)" }}>
                                        {item.label} {isBest && <span className="badge badge-stable" style={{ marginLeft: "8px" }}>OPTIMAL DECISION</span>}
                                      </span>
                                      <span style={{ fontFamily: "var(--font-mono)", color: val >= 0 ? "var(--accent-emerald)" : "var(--accent-rose)" }}>
                                        {item.arm === 0 ? "0.0000 (Baseline)" : (val >= 0 ? "+" : "") + val.toFixed(4)}
                                      </span>
                                    </div>
                                    <div className="progress-bar-bg">
                                      <div
                                        className="progress-bar-fill"
                                        style={{
                                          width: `${Math.min(Math.max((val + 5) * 10, 5), 100)}%`,
                                          background: isBest ? "var(--accent-indigo)" : (val >= 0 ? "var(--accent-emerald)" : "var(--accent-rose)")
                                        }}
                                      />
                                    </div>
                                  </div>
                                );
                              })}
                            </div>
                          </div>

                          {/* Uplift Table */}
                          <div className="card" style={{ background: "var(--bg-card)", marginTop: "16px" }}>
                            <h2>Treatment Arms Policy Evaluation Matrix</h2>
                            <div className="table-container">
                              <table>
                                <thead>
                                  <tr>
                                    <th>Promotional Treatment</th>
                                    <th>Subsidy Rate</th>
                                    <th>CATE Uplift ($\tau_k$)</th>
                                    <th>Policy Recommendation</th>
                                    <th>Rationale</th>
                                  </tr>
                                </thead>
                                <tbody>
                                  {cateValues.map(item => {
                                    const val = typeof item.cate === "number" ? item.cate : 0;
                                    const isBest = item.arm === armIdx;
                                    return (
                                      <tr key={item.arm}>
                                        <td style={{ fontWeight: 600 }}>{item.label}</td>
                                        <td>{(item.rate * 100).toFixed(0)}%</td>
                                        <td style={{ fontFamily: "var(--font-mono)", color: val >= 0 ? "var(--accent-emerald)" : "var(--accent-rose)" }}>
                                          {item.arm === 0 ? "0.0000" : val.toFixed(4)}
                                        </td>
                                        <td>
                                          {isBest ? (
                                            <span className="badge badge-stable">RECOMMENDED</span>
                                          ) : (
                                            <span className="badge badge-neutral">Alternative</span>
                                          )}
                                        </td>
                                        <td style={{ color: "var(--text-muted)", fontSize: "12px" }}>
                                          {isBest
                                            ? "Maximizes expected incremental profit under unconstrained single-customer policy."
                                            : (val < 0 ? "Negative treatment effect indicates promo cost erodes profit." : "Positive effect but subordinate to optimal arm.")}
                                        </td>
                                      </tr>
                                    );
                                  })}
                                </tbody>
                              </table>
                            </div>
                          </div>
                        </>
                      );
                    })()}
                  </div>
                ) : (
                  <div className="empty-state">
                    <div className="empty-state-icon">🎯</div>
                    <h3>No Active CATE Evaluation</h3>
                    <p>Submit a customer profile or click below to evaluate heterogeneous treatment effects using Double Machine Learning.</p>
                    <button className="btn-primary" onClick={runPrediction}>
                      Evaluate Default Profile Uplift
                    </button>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* TAB 4: BUDGET OPTIMIZATION */}
          {activeTab === "optimization" && (
            <div>
              {optError && <div className="alert-error">Optimization Failure: {optError}</div>}

              <div className="card">
                <h2>Prescriptive Marketing Budget Allocation Workspace</h2>
                <p className="card-subtitle">
                  Formulates multi-treatment knapsack allocation to maximize total cohort incremental profit under a strict budget constraint using the HiGHS Mixed-Integer Linear Programming solver.
                </p>

                <div className="grid-2">
                  <div className="form-group">
                    <label>Campaign Budget Limit ($)</label>
                    <input
                      type="number"
                      step="50"
                      min="10"
                      value={budget}
                      onChange={e => setBudget(e.target.value)}
                    />
                    <span className="form-hint">Maximum subsidy capital allocated across cohort</span>
                  </div>

                  <div className="form-group">
                    <label>Prescriptive Solver Mode</label>
                    <select
                      value={solverMode}
                      onChange={e => setSolverMode(e.target.value)}
                    >
                      <option value="bounded_gap">HiGHS MILP (Exact Bounded Gap)</option>
                      <option value="greedy">Greedy Knapsack Heuristic</option>
                    </select>
                    <span className="form-hint">Exact branch-and-cut optimization vs fast greedy heuristic</span>
                  </div>
                </div>

                <div style={{ display: "flex", gap: "12px", marginTop: "16px", marginBottom: "24px" }}>
                  <button className="btn-primary" disabled={optLoading} onClick={runOptimization}>
                    {optLoading ? "Solving Mixed-Integer Program..." : "Execute Budget Optimization"}
                  </button>
                  {optResult && (
                    <button className="btn-secondary" onClick={downloadOptReport}>
                      ⬇ Export Allocation Report (JSON)
                    </button>
                  )}
                </div>

                {optResult ? (
                  <div>
                    {(() => {
                      const metrics = optResult.metrics || {};
                      const spent = typeof metrics.total_budget_spent === "number" ? metrics.total_budget_spent : 0;
                      const remaining = typeof metrics.budget_remaining === "number" ? metrics.budget_remaining : 0;
                      const profit = typeof metrics.total_expected_incremental_profit === "number" ? metrics.total_expected_incremental_profit : 0;
                      const totalBudget = parseFloat(budget) || 1;
                      const utilization = Math.min((spent / totalBudget) * 100, 100);

                      return (
                        <>
                          <div className="grid-4">
                            <div className="metric-box">
                              <div className="metric-value">${spent.toFixed(2)}</div>
                              <div className="metric-label">Allocated Spend</div>
                              <div className="metric-sub">{utilization.toFixed(1)}% of ${totalBudget} budget</div>
                            </div>

                            <div className="metric-box">
                              <div className="metric-value">${remaining.toFixed(2)}</div>
                              <div className="metric-label">Unallocated Budget</div>
                              <div className="metric-sub">Remaining subsidy capital</div>
                            </div>

                            <div className="metric-box">
                              <div className="metric-value">${profit.toFixed(2)}</div>
                              <div className="metric-label">Portfolio Incremental Uplift</div>
                              <div className="metric-sub">Net cohort incremental profit</div>
                            </div>

                            <div className="metric-box">
                              <div className="metric-value" style={{ textTransform: "uppercase" }}>
                                {optResult.solver_status || "OPTIMAL"}
                              </div>
                              <div className="metric-label">Solver Convergence</div>
                              <div className="metric-sub">
                                {typeof optResult.solver_time_seconds === "number" ? optResult.solver_time_seconds.toFixed(4) + "s" : "< 0.01s"}
                              </div>
                            </div>
                          </div>

                          {/* Budget Utilization Gauge */}
                          <div className="card" style={{ background: "var(--bg-card)", marginTop: "16px" }}>
                            <h2>Budget Utilization Efficiency</h2>
                            <div style={{ marginTop: "10px" }}>
                              <div style={{ display: "flex", justifyContent: "space-between", fontSize: "13px", marginBottom: "6px" }}>
                                <span>Utilization: <strong>{utilization.toFixed(1)}%</strong></span>
                                <span>Spent: ${spent.toFixed(2)} / ${totalBudget.toFixed(2)}</span>
                              </div>
                              <div className="progress-bar-bg" style={{ height: "12px" }}>
                                <div
                                  className="progress-bar-fill"
                                  style={{
                                    width: `${utilization}%`,
                                    background: utilization > 90 ? "var(--accent-emerald)" : "var(--accent-indigo)"
                                  }}
                                />
                              </div>
                            </div>
                          </div>

                          {/* Customer Allocation Table */}
                          {optResult.allocations && optResult.allocations.length > 0 && (
                            <div className="card" style={{ background: "var(--bg-card)", marginTop: "16px" }}>
                              <h2>Cohort Treatment Allocations ({optResult.allocations.length} Customers)</h2>
                              <div className="table-container">
                                <table>
                                  <thead>
                                    <tr>
                                      <th>Customer Identifier</th>
                                      <th>Assigned Treatment Arm</th>
                                      <th>Promotional Discount</th>
                                      <th>Allocated Subsidy Cost</th>
                                      <th>Expected Incremental Profit</th>
                                    </tr>
                                  </thead>
                                  <tbody>
                                    {optResult.allocations.map((alloc, idx) => {
                                      const arm = alloc.assigned_arm !== undefined ? alloc.assigned_arm : alloc.recommended_arm;
                                      const rate = typeof alloc.discount_rate === "number" ? alloc.discount_rate : (arm === 1 ? 0.05 : arm === 2 ? 0.10 : arm === 3 ? 0.20 : 0);
                                      const cost = typeof alloc.allocated_budget_cost === "number" ? alloc.allocated_budget_cost : 0;
                                      const incProfit = typeof alloc.expected_incremental_profit === "number" ? alloc.expected_incremental_profit : 0;

                                      return (
                                        <tr key={idx}>
                                          <td style={{ fontWeight: 600 }}>{alloc.customer_id || `Customer #${idx + 1}`}</td>
                                          <td>Arm {arm}</td>
                                          <td>
                                            <span className="badge badge-neutral">
                                              {(rate * 100).toFixed(0)}% Discount
                                            </span>
                                          </td>
                                          <td>${cost.toFixed(2)}</td>
                                          <td style={{ color: incProfit >= 0 ? "var(--accent-emerald)" : "var(--accent-rose)", fontWeight: 600 }}>
                                            ${incProfit.toFixed(2)}
                                          </td>
                                        </tr>
                                      );
                                    })}
                                  </tbody>
                                </table>
                              </div>
                            </div>
                          )}
                        </>
                      );
                    })()}
                  </div>
                ) : (
                  <div className="empty-state">
                    <div className="empty-state-icon">⚖️</div>
                    <h3>Optimization Not Yet Executed</h3>
                    <p>Define a campaign budget and click "Execute Budget Optimization" to run the integer programming solver across the customer cohort.</p>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* TAB 5: DRIFT OBSERVABILITY */}
          {activeTab === "drift" && (
            <div>
              {driftError && <div className="alert-error">Drift Analysis Failure: {driftError}</div>}

              <div className="card">
                <h2>Continuous Covariate Distribution Drift Monitoring</h2>
                <p className="card-subtitle">
                  Executes two-sample non-parametric Kolmogorov-Smirnov hypothesis tests for continuous confounders and Chi-square contingency tests for categorical confounders, auditing distribution shifts between baseline reference cohorts and production inference traffic.
                </p>

                <div className="grid-2">
                  <div className="form-group">
                    <label>Hypothesis Significance Level ($\alpha$)</label>
                    <input
                      type="number"
                      step="0.01"
                      min="0.001"
                      max="0.5"
                      value={driftAlpha}
                      onChange={e => setDriftAlpha(e.target.value)}
                    />
                    <span className="form-hint">Type I error threshold (default 0.05 for 95% confidence)</span>
                  </div>

                  <div className="form-group">
                    <label>Drift Share Alert Threshold</label>
                    <input
                      type="number"
                      step="0.05"
                      min="0.1"
                      max="1.0"
                      value={driftThreshold}
                      onChange={e => setDriftThreshold(e.target.value)}
                    />
                    <span className="form-hint">Maximum tolerated fraction of drifted features before alert</span>
                  </div>
                </div>

                <div style={{ marginTop: "16px", marginBottom: "24px" }}>
                  <button className="btn-primary" disabled={driftLoading} onClick={runDriftDetection}>
                    {driftLoading ? "Running Hypothesis Tests..." : "Run Statistical Drift Audit"}
                  </button>
                </div>

                {driftResult ? (
                  <div>
                    <div className="grid-4">
                      <div className="metric-box">
                        <div className="metric-value">
                          {driftResult.overall_drift_detected ? (
                            <span style={{ color: "var(--accent-rose)" }}>DRIFT DETECTED</span>
                          ) : (
                            <span style={{ color: "var(--accent-emerald)" }}>STABLE</span>
                          )}
                        </div>
                        <div className="metric-label">Dataset Observability Status</div>
                        <div className="metric-sub">
                          {driftResult.overall_drift_detected ? "Significant covariate shift detected" : "Distributions aligned with baseline"}
                        </div>
                      </div>

                      <div className="metric-box">
                        <div className="metric-value">
                          {typeof driftResult.drift_share === "number" ? (driftResult.drift_share * 100).toFixed(1) + "%" : "0.0%"}
                        </div>
                        <div className="metric-label">Observed Drift Share</div>
                        <div className="metric-sub">Threshold: {(driftThreshold * 100).toFixed(0)}%</div>
                      </div>

                      <div className="metric-box">
                        <div className="metric-value">
                          {driftResult.drifted_features_count !== undefined ? driftResult.drifted_features_count : 0}
                        </div>
                        <div className="metric-label">Drifted Confounders</div>
                        <div className="metric-sub">Reject null hypothesis (p &lt; α)</div>
                      </div>

                      <div className="metric-box">
                        <div className="metric-value">
                          {driftResult.total_features_evaluated !== undefined ? driftResult.total_features_evaluated : 8}
                        </div>
                        <div className="metric-label">Total Confounders Audited</div>
                        <div className="metric-sub">Multivariate statistical coverage</div>
                      </div>
                    </div>

                    {/* Feature Drift Metrics Table */}
                    {driftResult.feature_metrics && Object.keys(driftResult.feature_metrics).length > 0 && (
                      <div className="card" style={{ background: "var(--bg-card)", marginTop: "16px" }}>
                        <h2>Feature-Level Non-Parametric Hypothesis Tests</h2>
                        <div className="table-container">
                          <table>
                            <thead>
                              <tr>
                                <th>Confounder Feature</th>
                                <th>Hypothesis Test</th>
                                <th>Test Statistic</th>
                                <th>p-Value</th>
                                <th>Sample Sizes (Ref / Cur)</th>
                                <th>Statistical Determination</th>
                              </tr>
                            </thead>
                            <tbody>
                              {Object.entries(driftResult.feature_metrics).map(([feature, metric]) => {
                                const isDrift = metric.drift_detected;
                                return (
                                  <tr key={feature}>
                                    <td style={{ fontWeight: 600 }}>{feature}</td>
                                    <td>{metric.test_type || "Two-Sample KS"}</td>
                                    <td style={{ fontFamily: "var(--font-mono)" }}>
                                      {typeof metric.statistic === "number" ? metric.statistic.toFixed(4) : "-"}
                                    </td>
                                    <td style={{ fontFamily: "var(--font-mono)" }}>
                                      {typeof metric.p_value === "number" ? metric.p_value.toFixed(5) : "-"}
                                    </td>
                                    <td>
                                      {metric.sample_size_reference || 3} / {metric.sample_size_current || 3}
                                    </td>
                                    <td>
                                      {isDrift ? (
                                        <span className="badge badge-drift">DRIFT DETECTED</span>
                                      ) : (
                                        <span className="badge badge-stable">STABLE</span>
                                      )}
                                    </td>
                                  </tr>
                                );
                              })}
                            </tbody>
                          </table>
                        </div>
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="empty-state">
                    <div className="empty-state-icon">🛡️</div>
                    <h3>Statistical Drift Monitor Idle</h3>
                    <p>Click "Run Statistical Drift Audit" to execute two-sample Kolmogorov-Smirnov and Chi-square hypothesis tests comparing inference traffic to the reference baseline.</p>
                  </div>
                )}
              </div>
            </div>
          )}
        </main>
      </div>
    </div>
  );
}
