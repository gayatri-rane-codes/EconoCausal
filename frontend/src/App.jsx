import React, { useState, useEffect } from "react";
import "./App.css";

const API_BASE = "http://localhost:8000";

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

export default function App() {
  const [tab, setTab] = useState("input");
  const [health, setHealth] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [customer, setCustomer] = useState(DEFAULT_CUSTOMER);
  const [rec, setRec] = useState(null);
  const [budget, setBudget] = useState(100.0);
  const [solverMode, setSolverMode] = useState("bounded_gap");
  const [opt, setOpt] = useState(null);
  const [alpha, setAlpha] = useState(0.05);
  const [driftThresh, setDriftThresh] = useState(0.3);
  const [drift, setDrift] = useState(null);

  useEffect(() => {
    fetch(API_BASE + "/health")
      .then(res => res.json())
      .then(d => setHealth(d))
      .catch(() => setHealth({ status: "offline" }));
  }, []);

  const handleChange = (e) => {
    const { name, value, type } = e.target;
    setCustomer(p => ({ ...p, [name]: type === "number" ? parseFloat(value) : parseInt(value, 10) }));
  };
  const runRecommend = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(API_BASE + "/causal/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ customers: [customer] })
      });
      if (!res.ok) throw new Error("Causal prediction failed: " + res.status);
      const d = await res.json();
      setRec(d.recommendations[0]);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const runOptimize = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(API_BASE + "/optimization/pipeline", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          customers: [
            customer,
            { ...customer, historical_spend: 80.0, cart_abandonment_rate: 0.6, loyalty_member: 0 },
            { ...customer, historical_spend: 320.0, browsing_frequency: 24.0, loyalty_member: 1 }
          ],
          budget: parseFloat(budget),
          mode: solverMode
        })
      });
      if (!res.ok) throw new Error("Optimization failed: " + res.status);
      const d = await res.json();
      setOpt(d);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const runDrift = async () => {
    setLoading(true);
    setError(null);
    try {
      const refData = Array(5).fill(DEFAULT_CUSTOMER);
      const currData = Array(5).fill({
        ...DEFAULT_CUSTOMER,
        historical_spend: 400.0,
        browsing_frequency: 2.0,
        recency_days: 90.0
      });
      const res = await fetch(API_BASE + "/monitoring/data-drift", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          reference_data: refData,
          current_data: currData,
          alpha: parseFloat(alpha),
          drift_share_threshold: parseFloat(driftThresh)
        })
      });
      if (!res.ok) throw new Error("Drift check failed: " + res.status);
      const d = await res.json();
      setDrift(d);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };
  return (
    <div className="dashboard-container">
      <div className="header">
        <div>
          <h1>EconoCausal Dashboard</h1>
          <p>Dynamic Pricing & Optimization via Double Machine Learning</p>
        </div>
        <div>
          <span className={"badge " + (health && health.status === "healthy" ? "badge-success" : "badge-danger")}>
            API: {health ? health.status : "Connecting..."}
          </span>
        </div>
      </div>

      <div className="tabs">
        <button className={"tab-btn " + (tab === "input" ? "active" : "")} onClick={() => setTab("input")}>1. Customer Input</button>
        <button className={"tab-btn " + (tab === "rec" ? "active" : "")} onClick={() => setTab("rec")}>2. Discount Recommendations</button>
        <button className={"tab-btn " + (tab === "opt" ? "active" : "")} onClick={() => setTab("opt")}>3. Budget Optimization</button>
        <button className={"tab-btn " + (tab === "drift" ? "active" : "")} onClick={() => setTab("drift")}>4. Drift Monitoring</button>
      </div>

      {error && <div className="error-alert">{error}</div>}

      {tab === "input" && (
        <div className="card">
          <h2>Customer Confounder Features (8 Observed Confounders)</h2>
          <div className="grid-2">
            <div className="form-group">
              <label>Browsing Frequency</label>
              <input type="number" step="0.1" name="browsing_frequency" value={customer.browsing_frequency} onChange={handleChange} />
            </div>
            <div className="form-group">
              <label>Cart Abandonment Rate (0 to 1)</label>
              <input type="number" step="0.01" min="0" max="1" name="cart_abandonment_rate" value={customer.cart_abandonment_rate} onChange={handleChange} />
            </div>
            <div className="form-group">
              <label>Competitor Price Index</label>
              <input type="number" step="0.01" name="competitor_price_index" value={customer.competitor_price_index} onChange={handleChange} />
            </div>
            <div className="form-group">
              <label>Historical Spend ($)</label>
              <input type="number" step="1" name="historical_spend" value={customer.historical_spend} onChange={handleChange} />
            </div>
            <div className="form-group">
              <label>Loyalty Member</label>
              <select name="loyalty_member" value={customer.loyalty_member} onChange={handleChange}>
                <option value={1}>1 - Yes</option>
                <option value={0}>0 - No</option>
              </select>
            </div>
            <div className="form-group">
              <label>Previous Discount Response Rate (0 to 1)</label>
              <input type="number" step="0.01" min="0" max="1" name="previous_discount_response_rate" value={customer.previous_discount_response_rate} onChange={handleChange} />
            </div>
            <div className="form-group">
              <label>Recency Days</label>
              <input type="number" step="1" name="recency_days" value={customer.recency_days} onChange={handleChange} />
            </div>
            <div className="form-group">
              <label>Seasonality Index</label>
              <input type="number" step="0.01" name="seasonality_index" value={customer.seasonality_index} onChange={handleChange} />
            </div>
          </div>
          <button className="btn-primary" onClick={() => { setTab("rec"); runRecommend(); }}>
            Save & Predict Uplift
          </button>
        </div>
      )}

      {tab === "rec" && (
        <div className="card">
          <h2>Causal Uplift & Discount Recommendation</h2>
          <p style={{ color: "var(--text-muted)", fontSize: "13px" }}>
            CATE Uplift evaluation across 4 promotional arms: 0%, 5%, 10%, 20%.
          </p>
          <button className="btn-primary" onClick={runRecommend} disabled={loading} style={{ marginBottom: "16px" }}>
            {loading ? "Evaluating CATE..." : "Evaluate Discount Recommendation"}
          </button>
          {rec && (
            <div className="grid-4">
              <div className="metric-box">
                <div className="metric-value">{(rec.recommended_discount * 100).toFixed(0)}%</div>
                <div className="metric-label">Recommended Arm</div>
              </div>
              <div className="metric-box">
                <div className="metric-value">${rec.expected_revenue ? rec.expected_revenue.toFixed(2) : "N/A"}</div>
                <div className="metric-label">Expected Revenue</div>
              </div>
              <div className="metric-box">
                <div className="metric-value">${rec.expected_cost ? rec.expected_cost.toFixed(2) : "N/A"}</div>
                <div className="metric-label">Promotion Cost</div>
              </div>
              <div className="metric-box">
                <div className="metric-value">${rec.expected_profit ? rec.expected_profit.toFixed(2) : "N/A"}</div>
                <div className="metric-label">Net Uplift Profit</div>
              </div>
            </div>
          )}
        </div>
      )}

      {tab === "opt" && (
        <div className="card">
          <h2>Prescriptive Budget Allocation Pipeline</h2>
          <div className="grid-2" style={{ marginBottom: "16px" }}>
            <div className="form-group">
              <label>Total Marketing Budget ($)</label>
              <input type="number" value={budget} onChange={(e) => setBudget(e.target.value)} />
            </div>
            <div className="form-group">
              <label>Optimization Solver Mode</label>
              <select value={solverMode} onChange={(e) => setSolverMode(e.target.value)}>
                <option value="bounded_gap">HiGHS MILP (bounded_gap)</option>
                <option value="greedy">Greedy Heuristic</option>
              </select>
            </div>
          </div>
          <button className="btn-primary" onClick={runOptimize} disabled={loading}>
            {loading ? "Solving Knapsack..." : "Run Optimization Pipeline"}
          </button>
          {opt && (
            <div style={{ marginTop: "20px" }}>
              <div className="grid-4" style={{ marginBottom: "16px" }}>
                <div className="metric-box">
                  <div className="metric-value">${opt.total_expected_profit ? opt.total_expected_profit.toFixed(2) : "0.00"}</div>
                  <div className="metric-label">Optimized Profit</div>
                </div>
                <div className="metric-box">
                  <div className="metric-value">${opt.total_cost ? opt.total_cost.toFixed(2) : "0.00"} / ${budget}</div>
                  <div className="metric-label">Cost / Budget</div>
                </div>
                <div className="metric-box">
                  <div className="metric-value">{(opt.budget_utilization * 100).toFixed(1)}%</div>
                  <div className="metric-label">Budget Utilization</div>
                </div>
                <div className="metric-box">
                  <div className="metric-value" style={{ fontSize: "18px" }}>
                    <span className="badge badge-success">{opt.solver_status || "optimal"}</span>
                  </div>
                  <div className="metric-label">Solver Status</div>
                </div>
              </div>
              <div className="table-container">
                <table>
                  <thead>
                    <tr>
                      <th>Customer</th>
                      <th>Allocated Discount</th>
                      <th>Expected Cost</th>
                      <th>Expected Profit</th>
                    </tr>
                  </thead>
                  <tbody>
                    {opt.allocations && opt.allocations.map((item, idx) => (
                      <tr key={idx}>
                        <td>Customer #{item.customer_id != null ? item.customer_id : idx + 1}</td>
                        <td>{(item.allocated_discount * 100).toFixed(0)}%</td>
                        <td>${item.cost ? item.cost.toFixed(2) : "0.00"}</td>
                        <td>${item.expected_profit ? item.expected_profit.toFixed(2) : "0.00"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}

      {tab === "drift" && (
        <div className="card">
          <h2>Statistical Data Drift Monitoring</h2>
          <div className="grid-2" style={{ marginBottom: "16px" }}>
            <div className="form-group">
              <label>Alpha Significance Level (alpha)</label>
              <input type="number" step="0.01" value={alpha} onChange={(e) => setAlpha(e.target.value)} />
            </div>
            <div className="form-group">
              <label>Drift Share Threshold (0.0 to 1.0)</label>
              <input type="number" step="0.05" value={driftThresh} onChange={(e) => setDriftThresh(e.target.value)} />
            </div>
          </div>
          <button className="btn-primary" onClick={runDrift} disabled={loading}>
            {loading ? "Evaluating KS & Chi-Square..." : "Run Data Drift Check"}
          </button>
          {drift && (
            <div style={{ marginTop: "20px" }}>
              <div style={{ marginBottom: "16px" }}>
                <span className={"badge " + (drift.overall_drift_detected ? "badge-danger" : "badge-success")} style={{ fontSize: "14px", padding: "6px 12px" }}>
                  {drift.overall_drift_detected ? "ALERT: DATA DRIFT DETECTED" : "SYSTEM STABLE: NO SIGNIFICANT DRIFT"}
                </span>
                <p style={{ marginTop: "8px", color: "var(--text-muted)", fontSize: "13px" }}>{drift.summary}</p>
              </div>
              <div className="table-container">
                <table>
                  <thead>
                    <tr>
                      <th>Feature Name</th>
                      <th>Test Type</th>
                      <th>Statistic</th>
                      <th>p-value</th>
                      <th>Drift Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(drift.feature_metrics || {}).map(([feat, m]) => (
                      <tr key={feat}>
                        <td><strong>{feat}</strong></td>
                        <td>{m.test_type}</td>
                        <td>{m.statistic != null ? m.statistic.toFixed(4) : "N/A"}</td>
                        <td>{m.p_value != null ? m.p_value.toFixed(4) : "N/A"}</td>
                        <td>
                          <span className={"badge " + (m.drift_detected ? "badge-danger" : "badge-success")}>
                            {m.drift_detected ? "Drifted" : "Stable"}
                          </span>
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
  );
}
