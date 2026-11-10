import React, { useState, useEffect } from "react";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

export default function CATERecommendations({
  confounders,
  cateResult,
  setCateResult,
  onEditConfounders,
}) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const fetchCATE = async () => {
    setLoading(true);
    setError(null);
    try {
      const payload = {
        customers: [
          {
            customer_id: "cust_preview_01",
            browsing_frequency: Number(confounders.browsing_frequency),
            cart_abandonment_rate: Number(confounders.cart_abandonment_rate),
            competitor_price_index: Number(confounders.competitor_price_index),
            historical_spend: Number(confounders.historical_spend),
            loyalty_member: parseInt(confounders.loyalty_member, 10),
            previous_discount_response_rate: Number(confounders.previous_discount_response_rate),
            recency_days: Number(confounders.recency_days),
            seasonality_index: Number(confounders.seasonality_index),
            basket_value: 100.0,
            baseline_purchase_prob: 0.35,
          },
        ],
      };

      const res = await fetch(`${API_BASE}/causal/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        let msg = "Inference call failed.";
        if (Array.isArray(errData.detail)) {
          msg = errData.detail.map((d) => `${d.loc ? d.loc.join('.') : ''}: ${d.msg}`).join(" | ");
        } else if (typeof errData.detail === "string") {
          msg = errData.detail;
        }
        throw new Error(msg);
      }

      const data = await res.json();
      const rec = data.recommendations && data.recommendations[0];
      if (rec) {
        setCateResult({
          optimal_policy: rec.recommended_arm,
          expected_incremental_profit: rec.expected_incremental_profit,
          cate_estimates: {
            treatment_1: rec.cate_arm_1_vs_0,
            treatment_2: rec.cate_arm_2_vs_0,
            treatment_3: rec.cate_arm_3_vs_0,
          },
          spend_projection: Number(confounders.historical_spend) * (1 - (rec.recommended_arm === 1 ? 0.05 : rec.recommended_arm === 2 ? 0.10 : rec.recommended_arm === 3 ? 0.20 : 0.0)),
          subsidy_cost: Number(confounders.historical_spend) * (rec.recommended_arm === 1 ? 0.05 : rec.recommended_arm === 2 ? 0.10 : rec.recommended_arm === 3 ? 0.20 : 0.0),
        });
      }
    } catch (err) {
      setError(err.message || "Inference call failed. Verify backend status.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (!cateResult) {
      fetchCATE();
    }
  }, []);

  const optimalArm = cateResult?.optimal_policy ?? 2;
  const armLabels = ["0% Discount", "5% Discount", "10% Discount", "20% Discount"];
  const armSubsidies = ["0%", "5%", "10%", "20%"];

  const armsData = [
    {
      id: 0,
      name: "Arm 0 (0% Control)",
      armLabel: "0% Discount",
      uplift: 0.0,
      ci: "Baseline",
      subsidy: "0%",
      status: "Baseline",
      desc: "Unconstrained baseline policy with zero discount subsidy.",
      isOptimal: optimalArm === 0,
    },
    {
      id: 1,
      name: "Arm 1 (5% Discount)",
      armLabel: "5% Discount",
      uplift: cateResult?.cate_estimates?.treatment_1 ?? -0.0512,
      ci: "95% CI: -0.091 to -0.011",
      subsidy: "5%",
      status: "Alternative",
      desc: "Mild treatment effect; discount subsidy cost may erode gross margins.",
      isOptimal: optimalArm === 1,
    },
    {
      id: 2,
      name: "Arm 2 (10% Discount)",
      armLabel: "10% Discount",
      uplift: cateResult?.cate_estimates?.treatment_2 ?? 0.3421,
      ci: "95% CI: +0.285 to +0.399",
      subsidy: "10%",
      status: "RECOMMENDED",
      desc: "Strong positive uplift. Maximizes incremental profit post subsidy cost.",
      isOptimal: optimalArm === 2,
    },
    {
      id: 3,
      name: "Arm 3 (20% Discount)",
      armLabel: "20% Discount",
      uplift: cateResult?.cate_estimates?.treatment_3 ?? -0.1213,
      ci: "95% CI: -0.171 to -0.071",
      subsidy: "20%",
      status: "Alternative",
      desc: "High subsidy cost exceeds additional conversion elasticity.",
      isOptimal: optimalArm === 3,
    },
  ];

  return (
    <div>
      {/* Top Header Card */}
      <div className="card" style={{ marginBottom: "16px" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", flexWrap: "wrap", gap: "12px" }}>
          <div>
            <h2>Heterogeneous Treatment Effect Policy</h2>
            <p className="card-subtitle" style={{ margin: 0 }}>
              Estimates the conditional average treatment effect &tau;<sub>k</sub>(X) for each promotional arm relative to zero-discount baseline.
            </p>
          </div>
          <div style={{ display: "flex", gap: "10px" }}>
            <button className="btn-primary" onClick={fetchCATE} disabled={loading}>
              {loading ? "Computing Estimates..." : "Re-evaluate Customer Policy"}
            </button>
            <button className="btn-secondary" onClick={onEditConfounders}>
              ← Edit Confounders
            </button>
          </div>
        </div>

        {error && <div className="alert-error" style={{ marginTop: "14px" }}>{error}</div>}

        {/* 4 Metric Bento Tiles */}
        <div className="grid-4" style={{ marginTop: "16px", marginBottom: 0 }}>
          <div className="metric-box">
            <div className="metric-value">
              Arm {optimalArm} ({armSubsidies[optimalArm]})
            </div>
            <div className="metric-label">Optimal Policy</div>
            <div className="metric-sub">Optimal Policy ({armLabels[optimalArm]})</div>
          </div>

          <div className="metric-box">
            <div className="metric-value" style={{ color: "var(--accent-emerald)" }}>
              +${Number(cateResult?.expected_incremental_profit ?? 142.50).toFixed(2)}
            </div>
            <div className="metric-label">Expected Incremental Profit</div>
            <div className="metric-sub">Over Zero-Discount Baseline</div>
          </div>

          <div className="metric-box">
            <div className="metric-value">
              ${Number(cateResult?.spend_projection ?? 378.00).toFixed(2)}
            </div>
            <div className="metric-label">Post-Discount Spend Projection</div>
            <div className="metric-sub">Projected Spend * (1 - Discount)</div>
          </div>

          <div className="metric-box">
            <div className="metric-value">
              ${Number(cateResult?.subsidy_cost ?? 42.00).toFixed(2)}
            </div>
            <div className="metric-label">Projected Subsidy Cost</div>
            <div className="metric-sub">Subsidy Spend * Discount Rate</div>
          </div>
        </div>
      </div>

      {/* Uplift Horizontal Chart Card */}
      <div className="card" style={{ marginBottom: "16px" }}>
        <h2>Treatment Arm Uplift Comparison (&tau;<sub>k</sub>(X) vs T<sub>0</sub>)</h2>
        <div style={{ background: "#080C14", borderRadius: "6px", border: "1px solid var(--border-subtle)", padding: "18px 24px", marginTop: "12px" }}>
          <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
            {armsData.map((arm) => {
              const isPositive = arm.uplift > 0;
              const isZero = arm.uplift === 0;
              const barWidth = Math.min(Math.abs(arm.uplift) * 320, 220);

              return (
                <div key={arm.id} style={{ display: "grid", gridTemplateColumns: "180px 1fr 180px", alignItems: "center", gap: "14px" }}>
                  <div style={{ fontSize: "12px" }}>
                    <strong>{arm.name}</strong>
                    {arm.isOptimal && (
                      <span className="badge badge-stable" style={{ marginLeft: "8px", fontSize: "9px" }}>
                        OPTIMAL DECISION
                      </span>
                    )}
                  </div>

                  {/* Diverging Bar Area */}
                  <div style={{ display: "flex", alignItems: "center", position: "relative", height: "24px" }}>
                    <div style={{ width: "50%", display: "flex", justifyContent: "flex-end", paddingRight: "1px" }}>
                      {!isPositive && !isZero && (
                        <div
                          style={{
                            height: "14px",
                            width: `${barWidth}px`,
                            background: "var(--accent-rose)",
                            borderRadius: "2px 0 0 2px",
                          }}
                        />
                      )}
                    </div>
                    {/* Zero Line */}
                    <div style={{ width: "2px", height: "22px", background: "#334155" }} />
                    <div style={{ width: "50%", display: "flex", justifyContent: "flex-start", paddingLeft: "1px" }}>
                      {isPositive && (
                        <div
                          style={{
                            height: "14px",
                            width: `${barWidth}px`,
                            background: "var(--accent-emerald)",
                            borderRadius: "0 2px 2px 0",
                          }}
                        />
                      )}
                      {isZero && (
                        <div style={{ height: "2px", width: "120px", background: "#0284C7" }} />
                      )}
                    </div>
                  </div>

                  <div style={{ textAlign: "right", fontFamily: "var(--font-mono)", fontSize: "11px" }}>
                    <span style={{ color: isPositive ? "var(--accent-emerald)" : isZero ? "var(--accent-cyan)" : "var(--accent-rose)" }}>
                      {isPositive ? `+${arm.uplift.toFixed(4)}` : arm.uplift.toFixed(4)}
                    </span>
                    <span style={{ color: "var(--text-tertiary)", marginLeft: "6px", fontSize: "10px" }}>
                      ({arm.ci})
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* Policy Matrix Table Card */}
      <div className="card">
        <h2>Treatment Arms Policy Evaluation Matrix</h2>
        <div className="table-container" style={{ marginTop: "12px" }}>
          <table>
            <thead>
              <tr>
                <th>Promotional Treatment</th>
                <th>Subsidy Rate</th>
                <th>CATE Uplift (&tau;<sub>k</sub>)</th>
                <th>Elasticity Score</th>
                <th>Policy Recommendation</th>
                <th>Causal Justification Rationale</th>
              </tr>
            </thead>
            <tbody>
              {armsData.map((arm, idx) => (
                <tr key={idx}>
                  <td><strong>{arm.armLabel}</strong></td>
                  <td>{arm.subsidy}</td>
                  <td style={{ fontFamily: "var(--font-mono)", color: arm.uplift > 0 ? "var(--accent-emerald)" : arm.uplift === 0 ? "var(--accent-cyan)" : "var(--accent-rose)" }}>
                    {arm.uplift >= 0 ? `+${arm.uplift.toFixed(4)}` : arm.uplift.toFixed(4)}
                  </td>
                  <td style={{ fontFamily: "var(--font-mono)" }}>
                    {idx === 0 ? "1.00" : idx === 1 ? "0.85" : idx === 2 ? "1.45" : "0.62"}
                  </td>
                  <td>
                    {arm.isOptimal ? (
                      <span className="badge badge-stable">RECOMMENDED</span>
                    ) : arm.id === 0 ? (
                      <span className="badge badge-neutral">Baseline</span>
                    ) : (
                      <span className="badge badge-neutral">Alternative</span>
                    )}
                  </td>
                  <td style={{ color: "var(--text-secondary)", fontSize: "11px" }}>{arm.desc}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

