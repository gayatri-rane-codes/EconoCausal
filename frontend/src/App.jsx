import React, { useState, useEffect } from "react";

const DEFAULT_API_BASE = "http://localhost:8000";

const DEFAULT_CUSTOMER = {
  browsing_frequency: 12.0,
  cart_abandonment_rate: 0.25,
  competitor_price_index: 1.05,
  historical_spend: 150.0,
  loyalty_member: 1,
  previous_discount_response_rate: 0.35,
  recency_days: 14.0,
  seasonality_index: 1.10
};

const HIGH_VALUE_CUSTOMER = {
  browsing_frequency: 28.0,
  cart_abandonment_rate: 0.12,
  competitor_price_index: 0.95,
  historical_spend: 420.0,
  loyalty_member: 1,
  previous_discount_response_rate: 0.65,
  recency_days: 4.0,
  seasonality_index: 1.25
};

const AT_RISK_CUSTOMER = {
  browsing_frequency: 4.0,
  cart_abandonment_rate: 0.70,
  competitor_price_index: 1.20,
  historical_spend: 60.0,
  loyalty_member: 0,
  previous_discount_response_rate: 0.15,
  recency_days: 45.0,
  seasonality_index: 0.90
};

const DISCOUNT_ARMS = [
  { arm: 0, label: "Arm 0 (0%)", rate: 0.00, desc: "Control Baseline" },
  { arm: 1, label: "Arm 1 (5%)", rate: 0.05, desc: "Low Discount" },
  { arm: 2, label: "Arm 2 (10%)", rate: 0.10, desc: "Moderate Discount" },
  { arm: 3, label: "Arm 3 (20%)", rate: 0.20, desc: "Aggressive Promo" }
];

export default function App() {
  const [activeTab, setActiveTab] = useState("overview");
  const [apiBase, setApiBase] = useState(DEFAULT_API_BASE);
  const [backendHealth, setBackendHealth] = useState({ status: "checking", details: null });

  // Customer & Uplift State
  const [customer, setCustomer] = useState(DEFAULT_CUSTOMER);
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

  // Check Backend Health on Mount
  const checkHealth = async () => {
    try {
      const res = await fetch(`${apiBase}/health`);
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
  }, [apiBase]);

  // Handle Confounder Input Change
  const handleCustomerChange = (key, val) => {
    setCustomer(prev => ({
      ...prev,
      [key]: key === "loyalty_member" ? parseInt(val, 10) : parseFloat(val) || 0
    }));
  };

  // Run CATE Uplift Prediction
  const runPrediction = async () => {
    setCateLoading(true);
    setCateError(null);
    try {
      const res = await fetch(`${apiBase}/causal/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ customers: [customer] })
      });
      if (!res.ok) {
        throw new Error(`Inference call failed with status HTTP ${res.status}`);
      }
      const data = await res.json();
      if (data && data.recommendations && data.recommendations.length > 0) {
        setCateResult(data.recommendations[0]);
      } else {
        throw new Error("Empty recommendation payload returned by backend.");
      }
    } catch (err) {
      setCateError(err.message);
    } finally {
      setCateLoading(false);
    }
  };

  // Run Prescriptive Optimization Pipeline
  const runOptimization = async () => {
    setOptLoading(true);
    setOptError(null);
    try {
      const testCohort = [
        customer,
        { ...DEFAULT_CUSTOMER, historical_spend: 320.0, browsing_frequency: 24.0, loyalty_member: 1 },
        { ...HIGH_VALUE_CUSTOMER, historical_spend: 550.0 },
        { ...AT_RISK_CUSTOMER, historical_spend: 95.0 },
        { ...DEFAULT_CUSTOMER, historical_spend: 210.0, browsing_frequency: 8.0 }
      ];

      const res = await fetch(`${apiBase}/optimization/pipeline`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          customers: testCohort,
          budget: parseFloat(budget) || 0,
          mode: solverMode
        })
      });
      if (!res.ok) {
        throw new Error(`Optimization pipeline error HTTP ${res.status}`);
      }
      const data = await res.json();
      setOptResult(data);
    } catch (err) {
      setOptError(err.message);
    } finally {
      setOptLoading(false);
    }
  };

  // Run Distribution Drift Monitor
  const runDriftDetection = async () => {
    setDriftLoading(true);
    setDriftError(null);
    try {
      const referenceCohort = [DEFAULT_CUSTOMER, HIGH_VALUE_CUSTOMER, AT_RISK_CUSTOMER];
      const productionCohort = [
        { ...DEFAULT_CUSTOMER, historical_spend: 480.0, browsing_frequency: 32.0 },
        { ...HIGH_VALUE_CUSTOMER, competitor_price_index: 1.45, cart_abandonment_rate: 0.65 },
        { ...AT_RISK_CUSTOMER, recency_days: 90.0, seasonality_index: 0.60 }
      ];

      const res = await fetch(`${apiBase}/monitoring/data-drift`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          reference_data: referenceCohort,
          current_data: productionCohort,
          alpha: parseFloat(driftAlpha) || 0.05,
          threshold: parseFloat(driftThreshold) || 0.33
        })
      });
      if (!res.ok) {
        throw new Error(`Drift detection failed HTTP ${res.status}`);
      }
      const data = await res.json();
      setDriftResult(data);
    } catch (err) {
      setDriftError(err.message);
    } finally {
      setDriftLoading(false);
    }
  };

  return (
    <div className="app-container">
      {/* Sidebar Navigation */}
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-logo">E</div>
          <div className="brand-text">
            <h1>EconoCausal</h1>
            <span>Causal AI Engine</span>
          </div>
        </div>

        <nav className="nav-links">
          <button
            className={`nav-item ${activeTab === "overview" ? "active" : ""}`}
            onClick={() => setActiveTab("overview")}
          >
            📊 Executive Overview
          </button>
          <button
            className={`nav-item ${activeTab === "customer" ? "active" : ""}`}
            onClick={() => setActiveTab("customer")}
          >
            👤 Customer Confounders
          </button>
          <button
            className={`nav-item ${activeTab === "uplift" ? "active" : ""}`}
            onClick={() => setActiveTab("uplift")}
          >
            🎯 CATE Recommendations
          </button>
          <button
            className={`nav-item ${activeTab === "optimization" ? "active" : ""}`}
            onClick={() => setActiveTab("optimization")}
          >
            ⚖️ Budget Allocation
          </button>
          <button
            className={`nav-item ${activeTab === "drift" ? "active" : ""}`}
            onClick={() => setActiveTab("drift")}
          >
            🛡️ Drift & Observability
          </button>
        </nav>

        <div className="sidebar-footer">
          <div style={{ marginBottom: "6px" }}>EconoCausal Suite v1.0.0</div>
          <div>DML · HiGHS MILP · SciPy</div>
        </div>
      </aside>

      {/* Main Content Area */}
      <div className="main-wrapper">
        {/* Top Header */}
        <header className="top-bar">
          <div className="top-bar-title">
            {activeTab === "overview" && "Executive Dashboard & Platform Health"}
            {activeTab === "customer" && "Customer Feature Profiling & Confounder Specification"}
            {activeTab === "uplift" && "Heterogeneous Treatment Effect (CATE) Evaluation"}
            {activeTab === "optimization" && "Prescriptive Marketing Budget Allocation"}
            {activeTab === "drift" && "Continuous Distribution Drift Monitoring"}
          </div>

          <div className="top-bar-actions">
            <div className="health-badge">
              <span className={`status-dot ${backendHealth.status === "healthy" ? "healthy" : "down"}`} />
              <span>Backend: {backendHealth.status.toUpperCase()}</span>
            </div>
            <button className="btn-secondary" style={{ padding: "6px 12px", fontSize: "12px" }} onClick={checkHealth}>
              Refresh Health
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
                  <div className="metric-value">{backendHealth.status === "healthy" ? "ONLINE" : "OFFLINE"}</div>
                  <div className="metric-label">FastAPI Backend Status</div>
                </div>
                <div className="metric-box">
                  <div className="metric-value">4 Arms</div>
                  <div className="metric-label">Evaluation Treatments (0%, 5%, 10%, 20%)</div>
                </div>
                <div className="metric-box">
                  <div className="metric-value">HiGHS MILP</div>
                  <div className="metric-label">Default Prescriptive Solver</div>
                </div>
                <div className="metric-box">
                  <div className="metric-value">Two-Sample KS</div>
                  <div className="metric-label">Non-Parametric Drift Engine</div>
                </div>
              </div>

              <div className="card">
                <h2>Causal AI Decision Pipeline Architecture</h2>
                <p style={{ color: "var(--text-muted)", fontSize: "14px", lineHeight: "1.6", marginBottom: "16px" }}>
                  EconoCausal bridges observational data, Double Machine Learning (DML), and integer programming to solve promotional pricing without confounding bias. Observational customer behavior is evaluated across 4 promotional arms to isolate true marginal treatment uplift ($\tau_k$) from organic baseline purchases.
                </p>
                <div className="grid-2">
                  <div style={{ background: "#151d2f", padding: "16px", borderRadius: "8px", border: "1px solid var(--border)" }}>
                    <div style={{ fontWeight: 600, color: "#fff", marginBottom: "8px" }}>1. Confounder De-biasing</div>
                    <div style={{ color: "var(--text-muted)", fontSize: "13px" }}>
                      Controls for browsing patterns, competitor pricing indices, cart abandonment, recency, and past promotions to prevent treatment selection bias.
                    </div>
                  </div>
                  <div style={{ background: "#151d2f", padding: "16px", borderRadius: "8px", border: "1px solid var(--border)" }}>
                    <div style={{ fontWeight: 600, color: "#fff", marginBottom: "8px" }}>2. Prescriptive Budget Allocation</div>
                    <div style={{ color: "var(--text-muted)", fontSize: "13px" }}>
                      Formulates treatment assignments as an exact bounded Mixed-Integer Linear Program (MILP) or Greedy Knapsack to maximize total profit under strict budgets.
                    </div>
                  </div>
                </div>
              </div>

              <div className="card">
                <h2>Quick Actions</h2>
                <div style={{ display: "flex", gap: "12px", flexWrap: "wrap" }}>
                  <button className="btn-primary" onClick={() => { setActiveTab("customer"); }}>
                    Configure Customer & Predict Uplift →
                  </button>
                  <button className="btn-secondary" onClick={() => { setActiveTab("optimization"); }}>
                    Run Cohort Budget Allocation →
                  </button>
                  <button className="btn-secondary" onClick={() => { setActiveTab("drift"); }}>
                    Inspect Feature Drift →
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
                <p style={{ color: "var(--text-muted)", fontSize: "13px", marginBottom: "20px" }}>
                  Specify the customer profile confounders used to de-bias treatment assignments and compute personalized treatment effects.
                </p>

                <div style={{ display: "flex", gap: "10px", marginBottom: "20px" }}>
                  <button className="btn-secondary" onClick={() => setCustomer(DEFAULT_CUSTOMER)}>
                    Load Default Preset
                  </button>
                  <button className="btn-secondary" onClick={() => setCustomer(HIGH_VALUE_CUSTOMER)}>
                    Load High-Value Loyal Preset
                  </button>
                  <button className="btn-secondary" onClick={() => setCustomer(AT_RISK_CUSTOMER)}>
                    Load Price-Sensitive At-Risk Preset
                  </button>
                </div>

                <div className="form-grid">
                  <div className="form-group">
                    <label>Browsing Frequency (monthly sessions)</label>
                    <input
                      type="number"
                      step="1"
                      value={customer.browsing_frequency}
                      onChange={e => handleCustomerChange("browsing_frequency", e.target.value)}
                    />
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
                  </div>

                  <div className="form-group">
                    <label>Competitor Price Index (e.g. 1.05 = +5% higher)</label>
                    <input
                      type="number"
                      step="0.01"
                      value={customer.competitor_price_index}
                      onChange={e => handleCustomerChange("competitor_price_index", e.target.value)}
                    />
                  </div>

                  <div className="form-group">
                    <label>Historical Spend ($)</label>
                    <input
                      type="number"
                      step="10"
                      value={customer.historical_spend}
                      onChange={e => handleCustomerChange("historical_spend", e.target.value)}
                    />
                  </div>

                  <div className="form-group">
                    <label>Loyalty Member</label>
                    <select
                      value={customer.loyalty_member}
                      onChange={e => handleCustomerChange("loyalty_member", e.target.value)}
                    >
                      <option value={1}>Yes (Active Member)</option>
                      <option value={0}>No (Non-member)</option>
                    </select>
                  </div>

                  <div className="form-group">
                    <label>Previous Discount Response Rate (0.0 to 1.0)</label>
                    <input
                      type="number"
                      step="0.01"
                      min="0"
                      max="1"
                      value={customer.previous_discount_response_rate}
                      onChange={e => handleCustomerChange("previous_discount_response_rate", e.target.value)}
                    />
                  </div>

                  <div className="form-group">
                    <label>Recency Days (days since last purchase)</label>
                    <input
                      type="number"
                      step="1"
                      value={customer.recency_days}
                      onChange={e => handleCustomerChange("recency_days", e.target.value)}
                    />
                  </div>

                  <div className="form-group">
                    <label>Seasonality Index (e.g. 1.10 = peak demand)</label>
                    <input
                      type="number"
                      step="0.05"
                      value={customer.seasonality_index}
                      onChange={e => handleCustomerChange("seasonality_index", e.target.value)}
                    />
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
                    {cateLoading ? "Computing CATE..." : "Submit Profile & Evaluate Uplift →"}
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* TAB 3: CATE RECOMMENDATIONS */}
          {activeTab === "uplift" && (
            <div>
              {cateError && <div className="alert-error">Backend Error: {cateError}</div>}

              <div className="card">
                <h2>CATE Treatment Policy Evaluation</h2>
                <p style={{ color: "var(--text-muted)", fontSize: "13px", marginBottom: "20px" }}>
                  Evaluates conditional treatment effects against the zero-discount baseline ($T_0$). Recommendations reflect the treatment arm maximizing expected incremental profit.
                </p>

                <div style={{ display: "flex", gap: "12px", marginBottom: "20px" }}>
                  <button className="btn-primary" disabled={cateLoading} onClick={runPrediction}>
                    {cateLoading ? "Evaluating DML..." : "Re-evaluate Customer Policy"}
                  </button>
                  <button className="btn-secondary" onClick={() => setActiveTab("customer")}>
                    Edit Confounders
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
                      const baseSpend = (customer && typeof customer.historical_spend === "number") ? customer.historical_spend : 0;
                      const promoCost = discountRate !== null ? (baseSpend * discountRate) : null;
                      const expRevenue = discountRate !== null ? (baseSpend * (1 - discountRate)) : null;
                      const upliftProfit = typeof cateResult.expected_incremental_profit === "number"
                        ? cateResult.expected_incremental_profit
                        : null;

                      return (
                        <>
                          <div className="grid-4">
                            <div className="metric-box">
                              <div className="metric-value">
                                {discountRate !== null ? (discountRate * 100).toFixed(0) + "%" : "Unavailable"}
                              </div>
                              <div className="metric-label">Optimal Policy (Arm {armIdx !== null ? armIdx : "-"})</div>
                            </div>

                            <div className="metric-box">
                              <div className="metric-value">
                                {upliftProfit !== null ? "$" + upliftProfit.toFixed(2) : "Unavailable"}
                              </div>
                              <div className="metric-label">Expected Incremental Profit</div>
                            </div>

                            <div className="metric-box">
                              <div className="metric-value">
                                {expRevenue !== null ? "$" + expRevenue.toFixed(2) : "Unavailable"}
                              </div>
                              <div className="metric-label">Estimated Post-Promo Spend</div>
                            </div>

                            <div className="metric-box">
                              <div className="metric-value">
                                {promoCost !== null ? "$" + promoCost.toFixed(2) : "Unavailable"}
                              </div>
                              <div className="metric-label">Discount Subsidy Cost</div>
                            </div>
                          </div>

                          <div className="card" style={{ background: "#151d2f", marginTop: "16px" }}>
                            <h2>Heterogeneous Treatment Uplift vs Control ($T_0$)</h2>
                            <div className="table-container">
                              <table>
                                <thead>
                                  <tr>
                                    <th>Promotional Arm</th>
                                    <th>Discount Rate</th>
                                    <th>CATE Marginal Uplift ($\tau$)</th>
                                    <th>Policy Decision</th>
                                  </tr>
                                </thead>
                                <tbody>
                                  <tr>
                                    <td>Arm 0 (Control)</td>
                                    <td>0%</td>
                                    <td>0.0000 (Baseline)</td>
                                    <td>{armIdx === 0 ? <span className="badge badge-stable">RECOMMENDED</span> : <span className="badge badge-neutral">Alternative</span>}</td>
                                  </tr>
                                  <tr>
                                    <td>Arm 1 (Low)</td>
                                    <td>5%</td>
                                    <td>{typeof cateResult.cate_arm_1_vs_0 === "number" ? cateResult.cate_arm_1_vs_0.toFixed(4) : "Unavailable"}</td>
                                    <td>{armIdx === 1 ? <span className="badge badge-stable">RECOMMENDED</span> : <span className="badge badge-neutral">Alternative</span>}</td>
                                  </tr>
                                  <tr>
                                    <td>Arm 2 (Moderate)</td>
                                    <td>10%</td>
                                    <td>{typeof cateResult.cate_arm_2_vs_0 === "number" ? cateResult.cate_arm_2_vs_0.toFixed(4) : "Unavailable"}</td>
                                    <td>{armIdx === 2 ? <span className="badge badge-stable">RECOMMENDED</span> : <span className="badge badge-neutral">Alternative</span>}</td>
                                  </tr>
                                  <tr>
                                    <td>Arm 3 (Aggressive)</td>
                                    <td>20%</td>
                                    <td>{typeof cateResult.cate_arm_3_vs_0 === "number" ? cateResult.cate_arm_3_vs_0.toFixed(4) : "Unavailable"}</td>
                                    <td>{armIdx === 3 ? <span className="badge badge-stable">RECOMMENDED</span> : <span className="badge badge-neutral">Alternative</span>}</td>
                                  </tr>
                                </tbody>
                              </table>
                            </div>
                          </div>
                        </>
                      );
                    })()}
                  </div>
                ) : (
                  <div style={{ textAlign: "center", padding: "40px", color: "var(--text-muted)" }}>
                    No CATE prediction evaluated yet. Click "Re-evaluate Customer Policy" or submit a customer profile.
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
                <h2>Prescriptive Budget Allocation Workspace</h2>
                <p style={{ color: "var(--text-muted)", fontSize: "13px", marginBottom: "20px" }}>
                  Solves a multi-treatment knapsack allocation to maximize total cohort uplift under a fixed campaign budget constraint using mixed-integer linear programming.
                </p>

                <div className="grid-2">
                  <div className="form-group">
                    <label>Total Campaign Budget ($)</label>
                    <input
                      type="number"
                      step="50"
                      value={budget}
                      onChange={e => setBudget(e.target.value)}
                    />
                  </div>

                  <div className="form-group">
                    <label>Prescriptive Solver Engine</label>
                    <select
                      value={solverMode}
                      onChange={e => setSolverMode(e.target.value)}
                    >
                      <option value="bounded_gap">HiGHS MILP (Exact Bounded Gap)</option>
                      <option value="greedy">Greedy Knapsack Heuristic</option>
                    </select>
                  </div>
                </div>

                <div style={{ marginTop: "16px", marginBottom: "24px" }}>
                  <button className="btn-primary" disabled={optLoading} onClick={runOptimization}>
                    {optLoading ? "Solving Pipeline..." : "Execute Budget Optimization"}
                  </button>
                </div>

                {optResult && (
                  <div>
                    <div className="grid-4">
                      <div className="metric-box">
                        <div className="metric-value">
                          ${typeof optResult.total_cost === "number" ? optResult.total_cost.toFixed(2) : "0.00"}
                        </div>
                        <div className="metric-label">Allocated Spend</div>
                      </div>

                      <div className="metric-box">
                        <div className="metric-value">
                          ${typeof optResult.remaining_budget === "number" ? optResult.remaining_budget.toFixed(2) : "0.00"}
                        </div>
                        <div className="metric-label">Unallocated Budget</div>
                      </div>

                      <div className="metric-box">
                        <div className="metric-value">
                          {typeof optResult.total_cost === "number" && parseFloat(budget) > 0
                            ? ((optResult.total_cost / parseFloat(budget)) * 100).toFixed(1) + "%"
                            : "0.0%"}
                        </div>
                        <div className="metric-label">Budget Utilization</div>
                      </div>

                      <div className="metric-box">
                        <div className="metric-value">
                          ${typeof optResult.total_incremental_profit === "number"
                            ? optResult.total_incremental_profit.toFixed(2)
                            : (typeof optResult.expected_incremental_profit === "number"
                              ? optResult.expected_incremental_profit.toFixed(2)
                              : "0.00")}
                        </div>
                        <div className="metric-label">Portfolio Incremental Uplift</div>
                      </div>
                    </div>

                    {optResult.allocations && optResult.allocations.length > 0 && (
                      <div className="card" style={{ background: "#151d2f" }}>
                        <h2>Customer Cohort Treatment Allocations</h2>
                        <div className="table-container">
                          <table>
                            <thead>
                              <tr>
                                <th>Customer Index</th>
                                <th>Assigned Discount Arm</th>
                                <th>Discount Rate</th>
                                <th>Allocated Subsidy Cost</th>
                              </tr>
                            </thead>
                            <tbody>
                              {optResult.allocations.map((alloc, idx) => (
                                <tr key={idx}>
                                  <td>Customer #{idx + 1}</td>
                                  <td>Arm {alloc.arm !== undefined ? alloc.arm : alloc.recommended_arm}</td>
                                  <td>
                                    <span className="badge badge-neutral">
                                      {alloc.discount !== undefined
                                        ? (alloc.discount * 100).toFixed(0) + "%"
                                        : (DISCOUNT_ARMS[alloc.recommended_arm]?.rate * 100).toFixed(0) + "%"}
                                    </span>
                                  </td>
                                  <td>${typeof alloc.cost === "number" ? alloc.cost.toFixed(2) : "0.00"}</td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          )}

          {/* TAB 5: DRIFT OBSERVABILITY */}
          {activeTab === "drift" && (
            <div>
              {driftError && <div className="alert-error">Drift Monitor Failure: {driftError}</div>}

              <div className="card">
                <h2>Continuous Covariate Drift Monitoring</h2>
                <p style={{ color: "var(--text-muted)", fontSize: "13px", marginBottom: "20px" }}>
                  Conducts two-sample non-parametric Kolmogorov-Smirnov tests across continuous confounders and Chi-square contingency tests across categorical features to identify distribution drift between baseline reference sets and live inference traffic.
                </p>

                <div className="grid-2">
                  <div className="form-group">
                    <label>Significance Level (Alpha, default 0.05)</label>
                    <input
                      type="number"
                      step="0.01"
                      min="0.001"
                      max="0.5"
                      value={driftAlpha}
                      onChange={e => setDriftAlpha(e.target.value)}
                    />
                  </div>

                  <div className="form-group">
                    <label>Drift Share Alert Threshold (default 0.33)</label>
                    <input
                      type="number"
                      step="0.05"
                      min="0.1"
                      max="1.0"
                      value={driftThreshold}
                      onChange={e => setDriftThreshold(e.target.value)}
                    />
                  </div>
                </div>

                <div style={{ marginTop: "16px", marginBottom: "24px" }}>
                  <button className="btn-primary" disabled={driftLoading} onClick={runDriftDetection}>
                    {driftLoading ? "Running Hypothesis Tests..." : "Run Covariate Drift Analysis"}
                  </button>
                </div>

                {driftResult && (
                  <div>
                    <div className="grid-4">
                      <div className="metric-box">
                        <div className="metric-value">
                          {driftResult.dataset_drift ? (
                            <span style={{ color: "var(--accent-rose)" }}>DRIFT DETECTED</span>
                          ) : (
                            <span style={{ color: "var(--accent-emerald)" }}>STABLE</span>
                          )}
                        </div>
                        <div className="metric-label">Overall Observability Status</div>
                      </div>

                      <div className="metric-box">
                        <div className="metric-value">
                          {typeof driftResult.drift_share === "number" ? (driftResult.drift_share * 100).toFixed(1) + "%" : "0.0%"}
                        </div>
                        <div className="metric-label">Drifted Feature Share</div>
                      </div>

                      <div className="metric-box">
                        <div className="metric-value">
                          {driftResult.number_of_drifted_features !== undefined ? driftResult.number_of_drifted_features : "-"}
                        </div>
                        <div className="metric-label">Drifted Confounders</div>
                      </div>

                      <div className="metric-box">
                        <div className="metric-value">
                          {driftResult.number_of_features !== undefined ? driftResult.number_of_features : "-"}
                        </div>
                        <div className="metric-label">Total Confounders Audited</div>
                      </div>
                    </div>

                    {driftResult.metrics && Object.keys(driftResult.metrics).length > 0 && (
                      <div className="card" style={{ background: "#151d2f" }}>
                        <h2>Feature-Level Non-Parametric Hypothesis Tests</h2>
                        <div className="table-container">
                          <table>
                            <thead>
                              <tr>
                                <th>Feature Confounder</th>
                                <th>Statistical Test</th>
                                <th>Test Statistic</th>
                                <th>p-Value</th>
                                <th>Drift Determination</th>
                              </tr>
                            </thead>
                            <tbody>
                              {Object.entries(driftResult.metrics).map(([feature, metric]) => (
                                <tr key={feature}>
                                  <td style={{ fontWeight: 600 }}>{feature}</td>
                                  <td>{metric.test_name || "Two-Sample KS"}</td>
                                  <td>{typeof metric.statistic === "number" ? metric.statistic.toFixed(4) : "-"}</td>
                                  <td>{typeof metric.p_value === "number" ? metric.p_value.toFixed(5) : "-"}</td>
                                  <td>
                                    {metric.drift_detected ? (
                                      <span className="badge badge-drift">DRIFT DETECTED</span>
                                    ) : (
                                      <span className="badge badge-stable">STABLE</span>
                                    )}
                                  </td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      </div>
                    )}
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
