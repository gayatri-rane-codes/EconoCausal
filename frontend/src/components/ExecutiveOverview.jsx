import React, { useState } from "react";

export default function ExecutiveOverview({
  cohortData,
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
  ];
  // --- Dynamic Counterfactual & Spend Path Generation ---
  const allocations = optimizationResult?.allocations || [];
  
  // 1. Calculate actual spend from cohortData distribution or backend property
  let calculatedSpend = Number(cohortData?.total_budget_spent ?? optimizationResult?.metrics?.total_budget_spent ?? 0);
  if (!calculatedSpend && cohortData?.distribution) {
    const costMap = { "15%": 15, "10%": 10, "5%": 5, "0%": 0 };
    Object.entries(cohortData.distribution).forEach(([tier, count]) => {
      for (const [key, cost] of Object.entries(costMap)) {
        if (tier.includes(key)) {
          calculatedSpend += cost * count;
          break;
        }
      }
    });
  }
  if (!calculatedSpend && allocations.length > 0) {
    calculatedSpend = allocations.reduce((acc, a) => acc + Number(a.allocated_budget_cost || 0), 0);
  }

  const totalSpend = calculatedSpend;
  const totalUplift = Number(cohortData?.total_incremental_profit ?? optimizationResult?.metrics?.total_expected_incremental_profit ?? 0);

  // 2. Build smooth continuous curves scaled to real values
  let spendPoints = [];
  let upliftPoints = [];

  const maxBudget = Number(cohortData?.budget_cap ?? (totalSpend > 0 ? totalSpend : 1000));
  const effectiveSpend = totalSpend > 0 ? totalSpend : maxBudget;
  const effectiveUplift = totalUplift > 0 ? totalUplift : 1000;

  // Generate 8 parametric points representing cumulative trajectory across budget
  const steps = 8;
  for (let i = 0; i <= steps; i++) {
    const t = i / steps; // 0.0 to 1.0
    const xVal = t * effectiveSpend;
    
    // Spend scales linearly
    const ySpend = t * effectiveSpend;
    // Diminishing returns curve for profit uplift
    const yUplift = Math.sin((t * Math.PI) / 2) * effectiveUplift;
    
    spendPoints.push({ x: xVal, y: ySpend });
    upliftPoints.push({ x: xVal, y: yUplift });
  }

  const maxSpendX = Math.max(...spendPoints.map((p) => p.x), 100);
  const maxUpliftY = Math.max(...upliftPoints.map((p) => p.y), totalUplift || 1, 100);

  const formatShort = (val) => {
    if (val >= 1000000) return `$${(val / 1000000).toFixed(1)}M`;
    if (val >= 1000) return `$${(val / 1000).toFixed(1)}k`;
    return `$${Math.round(val)}`;
  };

  const yTicks = [
    { y: 35, label: formatShort(maxUpliftY) },
    { y: 90, label: formatShort(maxUpliftY * 0.66) },
    { y: 145, label: formatShort(maxUpliftY * 0.33) },
    { y: 200, label: "$0" },
  ];

  const xTicks = [
    { x: 40, label: "$0" },
    { x: 150, label: formatShort(maxSpendX * 0.25) },
    { x: 260, label: formatShort(maxSpendX * 0.5) },
    { x: 370, label: formatShort(maxSpendX * 0.75) },
    { x: 470, label: formatShort(maxSpendX) },
  ];



  const factualSpendPath = spendPoints.map((p, idx) => {
    const px = 40 + (p.x / maxSpendX) * 440;
    const py = 200 - (p.y / maxSpendX) * 160;
    return `${idx === 0 ? 'M' : 'L'} ${px.toFixed(1)} ${py.toFixed(1)}`;
  }).join(' ');

  const upliftPath = upliftPoints.map((p, idx) => {
    const px = 40 + (p.x / maxSpendX) * 440;
    const py = 200 - (p.y / maxUpliftY) * 160;
    return `${idx === 0 ? 'M' : 'L'} ${px.toFixed(1)} ${py.toFixed(1)}`;
  }).join(' ');


  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "20px" }}>
      <div className="section-header">
        <div>
          <h2>Executive Overview</h2>
        </div>
        <div style={{ display: "flex", gap: "10px", alignItems: "center" }}>
          <span className={`badge ${isHealthy ? "badge-healthy" : "badge-critical"}`}>
            ? Backend: {backendHealth}
          </span>
          <button className="btn btn-secondary" onClick={() => window.location.reload()}>
            Refresh
          </button>
        </div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: "16px" }}>
        <div className="metric-box">
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <span className="metric-label">INFERENCE HEALTH</span>
            <span className="badge badge-healthy" style={{ fontSize: "9px", padding: "1px 6px" }}>READY</span>
          </div>
          <div className="metric-value" style={{ marginTop: "6px" }}>{isHealthy ? "READY" : backendHealth}</div>
          <div className="metric-sub">FastAPI Microservice Status</div>
        </div>

        <div className="metric-box">
          <span className="metric-label">TREATMENT POLICY DOMAIN</span>
          <div className="metric-value" style={{ marginTop: "6px" }}>4 Promotional Arms</div>
          <div className="metric-sub">0% (Control), 5%, 10%, 15%</div>
        </div>

        <div className="metric-box">
          <span className="metric-label">ACTIVE COHORT EXPECTED PROFIT</span>
          <div className="metric-value" style={{ marginTop: "6px", color: cohortData ? "#34d399" : "#f8fafc" }}>
            {cohortData
              ? `$${Number(cohortData.total_incremental_profit || 0).toLocaleString(undefined, { minimumFractionDigits: 2 })}`
              : "$0.00"}
          </div>
          <div className="metric-sub">
            {cohortData ? `Evaluated ${cohortData.total_customers?.toLocaleString()} customers` : "Cohort of 6 customers"}
          </div>
        </div>

        <div className="metric-box">
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <span className="metric-label">DATASET OBSERVABILITY</span>
            <span className="badge badge-healthy" style={{ fontSize: "9px", padding: "1px 6px" }}>STABLE</span>
          </div>
          <div className="metric-value" style={{ marginTop: "6px" }}>{driftStatus}</div>
          <div className="metric-sub">Two-Sample KS & Chi-Square</div>
        </div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "20px" }}>
        <div className="card">
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "14px" }}>
            <h3>Causal Directed Acyclic Graph (DAG)</h3>
            <button className="btn btn-secondary" style={{ fontSize: "11px", padding: "4px 8px" }}>
              Edit DAG
            </button>
          </div>
          <div style={{ height: "260px", background: "rgba(10, 15, 30, 0.4)", borderRadius: "8px", position: "relative", overflow: "hidden" }}>
            <svg width="100%" height="100%" viewBox="0 0 450 380">
              <defs>
                <marker id="arrow" viewBox="0 0 10 10" refX="16" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                  <path d="M 0 0 L 10 5 L 0 10 z" fill="#4B5563" />
                </marker>
              </defs>
              {dagEdges.map((edge, idx) => {
                const src = dagNodes.find((n) => n.id === edge.from);
                const tgt = dagNodes.find((n) => n.id === edge.to);
                if (!src || !tgt) return null;
                return (
                  <line
                    key={idx}
                    x1={src.x}
                    y1={src.y}
                    x2={tgt.x}
                    y2={tgt.y}
                    stroke="#374151"
                    strokeWidth="1.5"
                    strokeDasharray={edge.to === "outcome" ? "none" : "3,3"}
                    markerEnd="url(#arrow)"
                  />
                );
              })}
              {dagNodes.map((node) => (
                <g
                  key={node.id}
                  transform={`translate(${node.x},${node.y})`}
                  onMouseEnter={() => setHoveredNode(node.id)}
                  onMouseLeave={() => setHoveredNode(null)}
                  style={{ cursor: "pointer" }}
                >
                  <circle
                    r={hoveredNode === node.id ? 10 : 7}
                    fill={node.color}
                    opacity={hoveredNode && hoveredNode !== node.id ? 0.4 : 1}
                    style={{ transition: "all 0.2s" }}
                  />
                  <text
                    x={node.x > 200 ? 12 : -12}
                    y={4}
                    textAnchor={node.x > 200 ? "start" : "end"}
                    fill="#9CA3AF"
                    fontSize="10"
                    fontWeight="500"
                  >
                    {node.label}
                  </text>
                </g>
              ))}
            </svg>
          </div>
        </div>

        <div className="card">
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "14px" }}>
            <h3>Counterfactual & Spend Impact</h3>
            <div style={{ display: "flex", gap: "12px", fontSize: "11px" }}>
              <span style={{ color: "#38BDF8" }}>? Factual Spend (${totalSpend.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })})</span>
              <span style={{ color: "#8B5CF6" }}>? Counterfactual Uplift (${totalUplift.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })})</span>
            </div>
          </div>
          <div style={{ height: "260px", background: "rgba(10, 15, 30, 0.4)", borderRadius: "8px", position: "relative" }}>
            <svg width="100%" height="100%" viewBox="0 0 500 240">
              <line x1="40" y1="210" x2="480" y2="210" stroke="#1f293d" strokeWidth="1" />
              <line x1="40" y1="20" x2="40" y2="210" stroke="#1f293d" strokeWidth="1" />
              {yTicks.map((t, i) => (
            <text key={`ytick-${i}`} x="35" y={t.y} fill="#64748b" fontSize="10" textAnchor="end">{t.label}</text>
          ))}
          {xTicks.map((t, i) => (
            <text key={`xtick-${i}`} x={t.x} y="222" fill="#64748b" fontSize="10" textAnchor="middle">{t.label}</text>
          ))}
              <path d={factualSpendPath} fill="none" stroke="#38BDF8" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />
              <path d={upliftPath} fill="none" stroke="#8B5CF6" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
            <div style={{ position: "absolute", top: "12px", left: "16px", fontSize: "11px", color: "#64748b" }}>
              View Spend projection / CATE estimation
            </div>
            <div style={{ position: "absolute", bottom: "10px", right: "16px" }}>
              <span className="badge badge-healthy" style={{ fontSize: "10px", padding: "2px 8px" }}>ACTIVE</span>
            </div>
          </div>
        </div>
      </div>

      <div>
        <h4 style={{ fontSize: "13px", fontWeight: "600", color: "#94a3b8", marginBottom: "12px" }}>
          Actionable Quick-Start Execution Panel
        </h4>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "16px" }}>
          <button
            onClick={() => setActiveTab && setActiveTab("confounders")}
            className="btn"
            style={{
              background: "#4f46e5",
              color: "#fff",
              padding: "14px 18px",
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              fontWeight: "600",
              borderRadius: "8px",
            }}
          >
            <span>Configure Customer & Predict Uplift</span>
            <span>?</span>
          </button>

          <button
            onClick={() => setActiveTab && setActiveTab("budget")}
            className="btn"
            style={{
              background: "#4f46e5",
              color: "#fff",
              padding: "14px 18px",
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              fontWeight: "600",
              borderRadius: "8px",
            }}
          >
            <span>Prescriptive Optimization Workspace</span>
            <span>?</span>
          </button>

          <button
            onClick={() => setActiveTab && setActiveTab("drift")}
            className="btn"
            style={{
              background: "#4f46e5",
              color: "#fff",
              padding: "14px 18px",
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              fontWeight: "600",
              borderRadius: "8px",
            }}
          >
            <span>Audit Feature Distribution Drift</span>
            <span>?</span>
          </button>
        </div>
      </div>
    </div>
  );
}

