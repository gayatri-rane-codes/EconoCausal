import React, { useState } from "react";

export default function ExecutiveOverview({
  backendHealth,
  optimizationResult,
  driftResult,
  setActiveTab,
}) {
  const isHealthy = backendHealth === "HEALTHY";
  const driftStatus = driftResult?.overall_drift_status || "STABLE";
  const [hoveredNode, setHoveredNode] = useState(null);

  const dagNodes = [
    { id: "browsing", label: "Browsing Frequency", x: 120, y: 40, color: "#8B5CF6" },
    { id: "cart", label: "Cart Abandonment", x: 120, y: 100, color: "#8B5CF6" },
    { id: "competitor", label: "Competitor Price", x: 100, y: 160, color: "#6366F1" },
    { id: "spend", label: "Historical Spend", x: 95, y: 220, color: "#38BDF8" },
    { id: "loyalty", label: "Loyalty Tier", x: 120, y: 280, color: "#38BDF8" },
    { id: "recency", label: "Recency / Seasonality", x: 120, y: 340, color: "#38BDF8" },
    { id: "treatment", label: "Promotional Treatment", x: 260, y: 180, color: "#8B5CF6" },
    { id: "outcome", label: "Purchase Outcome", x: 380, y: 200, color: "#38BDF8" },
  ];

  const dagEdges = [
    { from: "browsing", to: "treatment" },
    { from: "cart", to: "treatment" },
    { from: "competitor", to: "treatment" },
    { from: "spend", to: "treatment" },
    { from: "loyalty", to: "treatment" },
    { from: "recency", to: "treatment" },
    { from: "treatment", to: "outcome" },
    { from: "spend", to: "outcome" },
    { from: "loyalty", to: "outcome" },
    { from: "recency", to: "outcome" },
  ];

  return (
    <div>
      {/* Top 4 KPI Tiles */}
      <div className="grid-4" style={{ marginBottom: "16px" }}>
        <div className="metric-box">
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <span className="metric-label">Inference Health</span>
            <span className="badge badge-stable" style={{ fontSize: "9px", padding: "1px 6px" }}>READY</span>
          </div>
          <div className="metric-value" style={{ marginTop: "6px" }}>{isHealthy ? "READY" : backendHealth}</div>
          <div className="metric-sub">FastAPI Microservice Status</div>
        </div>

        <div className="metric-box">
          <span className="metric-label">Treatment Policy Domain</span>
          <div className="metric-value" style={{ marginTop: "6px" }}>4 Promotional Arms</div>
          <div className="metric-sub">0% (Control), 5%, 10%, 20%</div>
        </div>

        <div className="metric-box">
          <span className="metric-label">Active Optimized Spend</span>
          <div className="metric-value" style={{ marginTop: "6px" }}>
            {optimizationResult
              ? `$${Number(optimizationResult.total_allocated_spend || 0).toFixed(2)}`
              : "$0.00"}
          </div>
          <div className="metric-sub">
            {optimizationResult ? "Cohort of 6 customers" : "Cohort of 6 customers"}
          </div>
        </div>

        <div className="metric-box">
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <span className="metric-label">Dataset Observability</span>
            <span className="badge badge-stable" style={{ fontSize: "9px", padding: "1px 6px" }}>STABLE</span>
          </div>
          <div className="metric-value" style={{ marginTop: "6px", color: "var(--text-primary)" }}>
            {driftStatus}
          </div>
          <div className="metric-sub">Two-Sample KS & Chi-Square</div>
        </div>
      </div>

      {/* Center 2-Column Bento: DAG & Spend Impact Curve */}
      <div className="grid-2" style={{ gap: "16px", marginBottom: "16px" }}>
        {/* Left Bento: Causal DAG */}
        <div className="card" style={{ margin: 0, padding: "18px" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "12px" }}>
            <h2 style={{ fontSize: "13px", margin: 0 }}>Causal Directed Acyclic Graph (DAG)</h2>
            <div style={{ display: "flex", gap: "6px" }}>
              <button className="btn-secondary" style={{ padding: "3px 8px", fontSize: "11px" }}>Edit DAG</button>
            </div>
          </div>

          <div style={{ background: "#080C14", borderRadius: "6px", border: "1px solid var(--border-subtle)", padding: "10px" }}>
            <svg viewBox="0 0 460 380" style={{ width: "100%", height: "260px" }}>
              <defs>
                <marker id="arrow" viewBox="0 0 10 10" refX="16" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                  <path d="M 0 1 L 8 5 L 0 9 z" fill="#3B82F6" />
                </marker>
              </defs>

              {/* Connecting Edges */}
              {dagEdges.map((edge, i) => {
                const source = dagNodes.find(n => n.id === edge.from);
                const target = dagNodes.find(n => n.id === edge.to);
                return (
                  <line
                    key={i}
                    x1={source.x + 10}
                    y1={source.y}
                    x2={target.x - 10}
                    y2={target.y}
                    stroke="#233554"
                    strokeWidth="1.5"
                    markerEnd="url(#arrow)"
                  />
                );
              })}

              {/* Nodes */}
              {dagNodes.map((node) => (
                <g
                  key={node.id}
                  onMouseEnter={() => setHoveredNode(node.label)}
                  onMouseLeave={() => setHoveredNode(null)}
                  style={{ cursor: "pointer" }}
                >
                  <circle cx={node.x} cy={node.y} r="7" fill={node.color} />
                  <circle cx={node.x} cy={node.y} r="12" fill={node.color} opacity="0.2" />
                  <text
                    x={node.id === "treatment" || node.id === "outcome" ? node.x + 14 : node.x - 14}
                    y={node.y + 4}
                    fill="#CBD5E1"
                    fontSize="10"
                    textAnchor={node.id === "treatment" || node.id === "outcome" ? "start" : "end"}
                    fontFamily="var(--font-sans)"
                  >
                    {node.label}
                  </text>
                </g>
              ))}
            </svg>
          </div>
        </div>

        {/* Right Bento: Counterfactual & Spend Impact */}
        <div className="card" style={{ margin: 0, padding: "18px" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "8px" }}>
            <h2 style={{ fontSize: "13px", margin: 0 }}>Counterfactual & Spend Impact</h2>
            <div style={{ display: "flex", gap: "12px", fontSize: "11px" }}>
              <span style={{ color: "#38BDF8" }}>— Factual Spend ($0.00)</span>
              <span style={{ color: "#A855F7" }}>— Counterfactual Revenue Uplift</span>
            </div>
          </div>

          <div style={{ background: "#080C14", borderRadius: "6px", border: "1px solid var(--border-subtle)", padding: "12px" }}>
            <svg viewBox="0 0 500 240" style={{ width: "100%", height: "230px" }}>
              {/* Grid Lines */}
              {[40, 90, 140, 190].map((y, idx) => (
                <g key={idx}>
                  <line x1="40" y1={y} x2="480" y2={y} stroke="#172234" strokeWidth="1" strokeDasharray="3 3" />
                  <text x="32" y={y + 3} fill="#475569" fontSize="9" textAnchor="end">
                    ${30 - idx * 5}
                  </text>
                </g>
              ))}
              <line x1="40" y1="210" x2="480" y2="210" stroke="#1E293B" strokeWidth="1" />
              {[0, 100, 200, 300, 400, 500].map((xVal, i) => (
                <text key={i} x={50 + i * 82} y="226" fill="#475569" fontSize="9" textAnchor="middle">
                  {xVal}
                </text>
              ))}

              {/* Factual Spend Flat/Slight Curve (Cyan Solid) */}
              <path
                d="M 50 205 Q 120 180 200 175 T 350 160 T 460 160"
                fill="none"
                stroke="#38BDF8"
                strokeWidth="2.2"
              />

              {/* Counterfactual Uplift High Curve (Purple Solid) */}
              <path
                d="M 50 205 Q 100 135 150 145 T 250 130 T 350 75 T 460 55"
                fill="none"
                stroke="#A855F7"
                strokeWidth="2.4"
              />
            </svg>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginTop: "4px", fontSize: "11px", color: "var(--text-tertiary)" }}>
              <span>View Spend projection / CATE estimation</span>
              <span className="badge badge-primary">ACTIVE</span>
            </div>
          </div>
        </div>
      </div>

      {/* Bottom Panel: Quick Actions */}
      <div className="card" style={{ padding: "16px 20px" }}>
        <h2 style={{ fontSize: "13px", marginBottom: "10px" }}>Actionable Quick-Start Execution Panel</h2>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "12px" }}>
          <button className="btn-primary" onClick={() => setActiveTab("confounders")} style={{ padding: "10px 14px", justifyContent: "space-between" }}>
            <span>Configure Customer & Predict Uplift</span>
            <span>→</span>
          </button>
          <button className="btn-primary" onClick={() => setActiveTab("budget")} style={{ padding: "10px 14px", justifyContent: "space-between" }}>
            <span>Prescriptive Optimization Workspace</span>
            <span>→</span>
          </button>
          <button className="btn-primary" onClick={() => setActiveTab("drift")} style={{ padding: "10px 14px", justifyContent: "space-between" }}>
            <span>Audit Feature Distribution Drift</span>
            <span>→</span>
          </button>
        </div>
      </div>
    </div>
  );
}
