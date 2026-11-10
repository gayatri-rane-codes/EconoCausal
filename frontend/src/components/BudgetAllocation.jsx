import React, { useState, useEffect } from "react";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

export default function BudgetAllocation() {
  const [budget, setBudget] = useState(500);
  const [solverMode, setSolverMode] = useState("bounded_gap");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [optData, setOptData] = useState(null);

  const runOptimization = async () => {
    setLoading(true);
    setError(null);
    try {
      const payload = {
        budget: Number(budget),
        mode: solverMode,
        cohort_id: "default",
        include_allocations: true,
      };

      // Try /optimization/optimize then fallback to /optimize if mounted at root
      let res = await fetch(`${API_BASE}/optimization/optimize`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (res.status === 404) {
        res = await fetch(`${API_BASE}/optimize`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
      }

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        let msg = "Optimization failed.";
        if (Array.isArray(errData.detail)) {
          msg = errData.detail.map((d) => `${d.loc ? d.loc.join('.') : ''}: ${d.msg}`).join(" | ");
        } else if (typeof errData.detail === "string") {
          msg = errData.detail;
        }
        throw new Error(msg);
      }

      const data = await res.json();
      setOptData(data);
    } catch (err) {
      setError(err.message || "Optimization request failed.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    runOptimization();
  }, []);

  const metrics = optData?.metrics || {
    total_budget_spent: 0,
    budget_remaining: budget,
    total_expected_incremental_profit: 0,
    arm_counts: {},
  };

  const allocations = optData?.allocations || [];
  const allocatedSpend = metrics.total_budget_spent ?? 0;
  const remainingBudget = metrics.budget_remaining ?? (budget - allocatedSpend);
  const totalProfit = metrics.total_expected_incremental_profit ?? 0;
  const percentSpent = budget > 0 ? ((allocatedSpend / budget) * 100).toFixed(1) : "0.0";

  // Build SVG path for frontier curve
  const chartWidth = 740;
  const chartHeight = 160;
  const padding = { top: 15, right: 30, bottom: 25, left: 50 };
  const innerW = chartWidth - padding.left - padding.right;
  const innerH = chartHeight - padding.top - padding.bottom;

  let cumulativePoints = [];
  if (allocations.length > 0) {
    let cumCost = 0;
    let cumProfit = 0;
    const sorted = [...allocations].sort((a, b) => (b.expected_incremental_profit / (b.allocated_budget_cost || 1)) - (a.expected_incremental_profit / (a.allocated_budget_cost || 1)));
    
    cumulativePoints.push({ x: 0, y: 0 });
    sorted.forEach((item) => {
      cumCost += item.allocated_budget_cost;
      cumProfit += item.expected_incremental_profit;
      cumulativePoints.push({ x: cumCost, y: cumProfit });
    });
  } else {
    cumulativePoints = [
      { x: 0, y: 0 },
      { x: 100, y: 220 },
      { x: 250, y: 480 },
      { x: 400, y: 640 },
      { x: 500, y: 720 },
    ];
  }

  const maxX = Math.max(budget, ...cumulativePoints.map((p) => p.x), 1);
  const maxY = Math.max(...cumulativePoints.map((p) => p.y), 1);

  const svgPath = cumulativePoints
    .map((p, i) => {
      const px = padding.left + (p.x / maxX) * innerW;
      const py = padding.top + innerH - (p.y / maxY) * innerH;
      return `${i === 0 ? "M" : "L"} ${px.toFixed(1)} ${py.toFixed(1)}`;
    })
    .join(" ");

  const areaPath = `${svgPath} L ${(padding.left + innerW).toFixed(1)} ${(padding.top + innerH).toFixed(1)} L ${padding.left} ${(padding.top + innerH).toFixed(1)} Z`;

  return (
    <div>
      {/* Top Configuration Card */}
      <div className="card" style={{ marginBottom: "16px" }}>
        <h2>Prescriptive Marketing Budget Allocation Workspace</h2>
        <p className="card-subtitle">
          Formulates multi-treatment knapsack allocation to maximize total cohort incremental profit under a strict budget constraint using the HiGHS Mixed-Integer Linear Programming solver.
        </p>

        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "20px", marginTop: "16px" }}>
          <div>
            <label style={{ display: "block", fontSize: "12px", color: "var(--text-secondary)", marginBottom: "6px" }}>
              Campaign Budget Limit ($)
            </label>
            <input
              type="number"
              value={budget}
              onChange={(e) => setBudget(Number(e.target.value))}
              style={{
                width: "100%",
                background: "var(--bg-card)",
                border: "1px solid var(--border-color)",
                borderRadius: "4px",
                color: "var(--text-primary)",
                padding: "8px 12px",
                fontFamily: "var(--font-mono)",
                fontSize: "13px",
              }}
            />
            <span style={{ fontSize: "11px", color: "var(--text-tertiary)", marginTop: "4px", display: "block" }}>
              Maximum subsidy capital allocated across cohort
            </span>
          </div>

          <div>
            <label style={{ display: "block", fontSize: "12px", color: "var(--text-secondary)", marginBottom: "6px" }}>
              Prescriptive Solver Mode
            </label>
            <select
              value={solverMode}
              onChange={(e) => setSolverMode(e.target.value)}
              style={{
                width: "100%",
                background: "var(--bg-card)",
                border: "1px solid var(--border-color)",
                borderRadius: "4px",
                color: "var(--text-primary)",
                padding: "8px 12px",
                fontSize: "13px",
              }}
            >
              <option value="bounded_gap">HiGHS MILP (Exact Bounded Gap)</option>
              <option value="exact">HiGHS MILP (Exact Full Optimal)</option>
              <option value="bounded_time">HiGHS MILP (Time Constrained)</option>
              <option value="greedy">Greedy Incremental Heuristic</option>
            </select>
            <span style={{ fontSize: "11px", color: "var(--text-tertiary)", marginTop: "4px", display: "block" }}>
              Exact branch-and-cut optimization vs fast greedy heuristic
            </span>
          </div>
        </div>

        <div style={{ marginTop: "16px" }}>
          <button className="btn-primary" onClick={runOptimization} disabled={loading}>
            {loading ? "Optimizing Portfolio..." : "Execute Budget Optimization"}
          </button>
        </div>

        {error && <div className="alert-error" style={{ marginTop: "14px" }}>{error}</div>}

        {/* 4 Metric Bento Tiles */}
        <div className="grid-4" style={{ marginTop: "18px", marginBottom: 0 }}>
          <div className="metric-box">
            <div className="metric-value">
              ${allocatedSpend.toFixed(2)}
            </div>
            <div className="metric-label">Allocated Spend</div>
            <div className="metric-sub">{percentSpent}% of ${budget} budget</div>
          </div>

          <div className="metric-box">
            <div className="metric-value">
              ${remainingBudget.toFixed(2)}
            </div>
            <div className="metric-label">Unallocated Budget</div>
            <div className="metric-sub">Remaining subsidy capital</div>
          </div>

          <div className="metric-box">
            <div className="metric-value" style={{ color: "var(--accent-emerald)" }}>
              +${totalProfit.toFixed(2)}
            </div>
            <div className="metric-label">Portfolio Incremental Uplift</div>
            <div className="metric-sub">Net cohort incremental profit</div>
          </div>

          <div className="metric-box">
            <div className="metric-value" style={{ color: optData ? "var(--accent-emerald)" : "var(--accent-amber)" }}>
              {optData?.is_optimal_proven ? "OPTIMAL" : optData ? "CONVERGED" : "AWAITING RUN"}
            </div>
            <div className="metric-label">Solver Convergence</div>
            <div className="metric-sub">
              {optData?.solver_time_seconds ? `${(optData.solver_time_seconds * 1000).toFixed(1)}ms execution` : "No active session"}
            </div>
          </div>
        </div>
      </div>

      {/* Knapsack Frontier Chart */}
      <div className="card" style={{ marginBottom: "16px" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <h2>Knapsack Portfolio Frontier (Cumulative Profit vs Subsidy Cost)</h2>
          <span style={{ fontSize: "11px", color: "var(--text-tertiary)", fontFamily: "var(--font-mono)" }}>
            Efficiency Frontier: dProfit/dCost
          </span>
        </div>

        <div style={{ background: "#080C14", borderRadius: "6px", border: "1px solid var(--border-subtle)", padding: "12px", marginTop: "12px" }}>
          <svg viewBox={`0 0 ${chartWidth} ${chartHeight}`} style={{ width: "100%", height: "180px", display: "block" }}>
            {/* Grid lines */}
            {[0, 0.25, 0.5, 0.75, 1].map((pct, idx) => {
              const y = padding.top + innerH * (1 - pct);
              return (
                <g key={idx}>
                  <line x1={padding.left} y1={y} x2={chartWidth - padding.right} y2={y} stroke="#1E293B" strokeDasharray="3 3" />
                  <text x={padding.left - 8} y={y + 3} fill="#64748B" fontSize="9" textAnchor="end" fontFamily="var(--font-mono)">
                    ${Math.round(maxY * pct)}
                  </text>
                </g>
              );
            })}

            {/* Gradient Area Fill */}
            <defs>
              <linearGradient id="frontierGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#7C3AED" stopOpacity="0.3" />
                <stop offset="100%" stopColor="#7C3AED" stopOpacity="0.0" />
              </linearGradient>
            </defs>

            <path d={areaPath} fill="url(#frontierGrad)" />
            <path d={svgPath} fill="none" stroke="#A78BFA" strokeWidth="2.5" />

            {/* Active Points */}
            {cumulativePoints.slice(0, 12).map((p, i) => {
              const px = padding.left + (p.x / maxX) * innerW;
              const py = padding.top + innerH - (p.y / maxY) * innerH;
              return <circle key={i} cx={px} cy={py} r="3" fill="#7C3AED" stroke="#E2E8F0" strokeWidth="1" />;
            })}
          </svg>
        </div>
      </div>

      {/* Allocation Roster Table */}
      <div className="card">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <h2>Prescriptive Customer Allocation Roster</h2>
          <span className="badge badge-neutral">
            {allocations.length > 0 ? `${allocations.length} Allocations` : "No Active Allocations"}
          </span>
        </div>

        <div className="table-container" style={{ marginTop: "12px" }}>
          <table>
            <thead>
              <tr>
                <th>Customer ID</th>
                <th>Assigned Arm</th>
                <th>Discount Rate</th>
                <th>Allocated Subsidy Cost</th>
                <th>Expected Incremental Profit</th>
                <th>Uplift ROI Multiplier</th>
              </tr>
            </thead>
            <tbody>
              {allocations.length > 0 ? (
                allocations.map((item, idx) => {
                  const roi = item.allocated_budget_cost > 0
                    ? (item.expected_incremental_profit / item.allocated_budget_cost).toFixed(2)
                    : "Baseline (N/A)";
                  return (
                    <tr key={idx}>
                      <td style={{ fontFamily: "var(--font-mono)", fontSize: "11px" }}>{item.customer_id}</td>
                      <td>
                        <span className={`badge ${item.assigned_arm > 0 ? "badge-stable" : "badge-neutral"}`}>
                          Arm {item.assigned_arm}
                        </span>
                      </td>
                      <td style={{ fontFamily: "var(--font-mono)" }}>{(item.discount_rate * 100).toFixed(0)}%</td>
                      <td style={{ fontFamily: "var(--font-mono)" }}>${Number(item.allocated_budget_cost).toFixed(2)}</td>
                      <td style={{ fontFamily: "var(--font-mono)", color: "var(--accent-emerald)" }}>
                        +${Number(item.expected_incremental_profit).toFixed(2)}
                      </td>
                      <td style={{ fontFamily: "var(--font-mono)", color: item.allocated_budget_cost > 0 ? "var(--accent-cyan)" : "var(--text-tertiary)" }}>
                        {roi !== "Baseline (N/A)" ? `${roi}x` : roi}
                      </td>
                    </tr>
                  );
                })
              ) : (
                <tr>
                  <td colSpan="6" style={{ textAlign: "center", color: "var(--text-tertiary)", padding: "20px" }}>
                    No cohort allocations returned. Click "Execute Budget Optimization" to dispatch the solver.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

