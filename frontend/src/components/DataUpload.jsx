import React, { useState, useRef } from "react";
import axios from "axios";

const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

export default function DataUpload({ cohortData, setCohortData }) {
  const [file, setFile] = useState(null);
  const [budget, setBudget] = useState("");
  const [uploading, setUploading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const fileInputRef = useRef(null);

  const handleFileChange = (e) => {
    const selected = e.target.files[0];
    setError("");
    if (!selected) return;
    if (!selected.name.toLowerCase().endsWith(".csv")) {
      setError("Please select a valid .csv file.");
      return;
    }
    setFile(selected);
  };

    const handleExport = () => {
    if (!result?.preview || result.preview.length === 0) return;
    const headers = Object.keys(result.preview[0]);
    const csvRows = [
      headers.join(","),
      ...result.preview.map(row => headers.map(fieldName => JSON.stringify(row[fieldName] ?? "")).join(","))
    ];
    const blob = new Blob([csvRows.join("\n")], { type: "text/csv" });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `cohort_policy_recommendations_${new Date().toISOString().slice(0, 10)}.csv`;
    a.click();
    window.URL.revokeObjectURL(url);
  };

  const handleUpload = async () => {
    if (!file) return;
    setUploading(true);
    setError("");
    setResult(null); if (setCohortData) setCohortData(null);

    const formData = new FormData();
    formData.append("file", file);
    if (budget) { formData.append("budget", budget); }

    try {
      const response = await axios.post(`${API_BASE}/causal/evaluate-batch`, formData, {
        headers: { "Content-Type": "multipart/form-data" }
      });
      setResult(response.data); if (setCohortData) setCohortData(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || "Failed to process cohort on server.");
    } finally {
      setUploading(false);
    }
  };

  return (
    <div style={{ padding: "24px", color: "#e2e8f0" }}>
      <h2 style={{ fontSize: "1.5rem", fontWeight: "bold", marginBottom: "8px" }}>
        Cohort Batch Policy Evaluation
      </h2>
      <p style={{ color: "#94a3b8", marginBottom: "20px" }}>
        Upload customer cohort CSVs to run bulk causal policy recommendations and compute expected incremental profit.
      </p>

      {/* File Upload Control */}
      <div style={{ display: "flex", gap: "12px", alignItems: "center", marginBottom: "24px" }}>
        <input
          type="file"
          accept=".csv"
          ref={fileInputRef}
          onChange={handleFileChange}
          style={{ display: "none" }}
        />
        <button
          onClick={() => fileInputRef.current?.click()}
          style={{
            padding: "10px 18px",
            background: "#1e293b",
            border: "1px solid #475569",
            color: "#f8fafc",
            borderRadius: "6px",
            cursor: "pointer"
          }}
        >
          {file ? file.name : "Choose CSV File"}
        </button>
      

      <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
        <span style={{ fontSize: "12px", color: "#94a3b8", fontWeight: "500" }}>Max Budget ($):</span>
        <input
          type="number"
          placeholder="e.g. 30000 (optional)"
          value={budget}
          onChange={(e) => setBudget(e.target.value)}
          style={{
            background: "#0f172a",
            border: "1px solid #334155",
            borderRadius: "6px",
            color: "#f8fafc",
            padding: "8px 12px",
            fontSize: "12px",
            width: "180px",
            outline: "none"
          }}
        />
      </div>
        <button
          onClick={handleUpload}
          disabled={!file || uploading}
          style={{
            padding: "10px 22px",
            background: !file || uploading ? "#334155" : "#3b82f6",
            border: "none",
            color: "#ffffff",
            fontWeight: "600",
            borderRadius: "6px",
            cursor: !file || uploading ? "not-allowed" : "pointer"
          }}
        >
          {uploading ? "Evaluating Policy..." : "Run Policy Engine"}
        </button>
      {result?.preview && (
        <button
          onClick={handleExport}
          style={{
            padding: "10px 18px",
            background: "#10b981",
            border: "none",
            borderRadius: "6px",
            color: "#ffffff",
            fontSize: "13px",
            fontWeight: "600",
            cursor: "pointer"
          }}
        >
          Export CSV
        </button>
      )}
      </div>

      {error && (
        <div style={{ background: "#7f1d1d", color: "#fecaca", padding: "12px 16px", borderRadius: "6px", marginBottom: "20px" }}>
          {error}
        </div>
      )}

      {/* Aggregated KPI Cards */}
      {result && (
        <div>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: "16px", marginBottom: "28px" }}>
            <div style={{ background: "#0f172a", border: "1px solid #1e293b", padding: "16px", borderRadius: "8px" }}>
              <div style={{ color: "#94a3b8", fontSize: "0.85rem" }}>Total Customers</div>
              <div style={{ fontSize: "1.6rem", fontWeight: "bold", color: "#f8fafc" }}>
                {result.total_customers.toLocaleString()}
              </div>
            </div>

            <div style={{ background: "#0f172a", border: "1px solid #1e293b", padding: "16px", borderRadius: "8px" }}>
              <div style={{ color: "#94a3b8", fontSize: "0.85rem" }}>Expected Incremental Profit</div>
              <div style={{ fontSize: "1.6rem", fontWeight: "bold", color: "#10b981" }}>
                ${result.total_incremental_profit.toLocaleString()}
              </div>
            </div>

            <div style={{ background: "#0f172a", border: "1px solid #1e293b", padding: "16px", borderRadius: "8px" }}>
              <div style={{ color: "#94a3b8", fontSize: "0.85rem" }}>Wasted Budget Saved</div>
              <div style={{ fontSize: "1.6rem", fontWeight: "bold", color: "#38bdf8" }}>
                ${result.budget_saved_estimate.toLocaleString()}
              </div>
            </div>
          </div>

          {/* Action Distribution Breakdown */}
          <div style={{ background: "#0f172a", border: "1px solid #1e293b", padding: "18px", borderRadius: "8px", marginBottom: "24px" }}>
            <h3 style={{ fontSize: "1.1rem", marginBottom: "12px", color: "#f1f5f9" }}>Treatment Distribution</h3>
            <div style={{ display: "flex", gap: "24px", flexWrap: "wrap" }}>
              {Object.entries(result.distribution || {}).map(([tier, count]) => (
                <div key={tier} style={{ background: "#1e293b", padding: "8px 14px", borderRadius: "6px" }}>
                  <span style={{ color: "#94a3b8", marginRight: "8px" }}>{tier}:</span>
                  <strong style={{ color: "#f8fafc" }}>{count}</strong>
                </div>
              ))}
            </div>
          </div>

          {/* Preview Table */}
          <h3 style={{ fontSize: "1.1rem", marginBottom: "12px", color: "#f1f5f9" }}>Individual Customer Decisions (Top 100)</h3>
          <div style={{ overflowX: "auto", border: "1px solid #1e293b", borderRadius: "8px" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", textAlign: "left", fontSize: "0.9rem" }}>
              <thead style={{ background: "#1e293b", color: "#cbd5e1" }}>
                <tr>
                  <th style={{ padding: "10px 14px" }}>Customer ID</th>
                  <th style={{ padding: "10px 14px" }}>Recommended Tier</th>
                  <th style={{ padding: "10px 14px" }}>Exp. Incremental Profit</th>
                  <th style={{ padding: "10px 14px" }}>CATE (Arm 1)</th>
                  <th style={{ padding: "10px 14px" }}>CATE (Arm 2)</th>
                  <th style={{ padding: "10px 14px" }}>CATE (Arm 3)</th>
                </tr>
              </thead>
              <tbody>
                {result.preview?.map((row, idx) => (
                  <tr key={idx} style={{ borderBottom: "1px solid #1e293b", background: idx % 2 === 0 ? "#0f172a" : "#131d33" }}>
                    <td style={{ padding: "10px 14px" }}>{row.customer_id || row.user_id || row.id || row.CustomerID || `Cust #${idx + 1}`}</td>
                    <td style={{ padding: "10px 14px", fontWeight: "600", color: row.recommended_arm === 0 ? "#94a3b8" : "#38bdf8" }}>
                      {row.discount_tier}
                    </td>
                    <td style={{ padding: "10px 14px", color: "#10b981" }}>
                      ${Number(row.expected_incremental_profit || 0).toFixed(2)}
                    </td>
                    <td style={{ padding: "10px 14px" }}>{Number(row.cate_arm_1_vs_0 || 0).toFixed(3)}</td>
                    <td style={{ padding: "10px 14px" }}>{Number(row.cate_arm_2_vs_0 || 0).toFixed(3)}</td>
                    <td style={{ padding: "10px 14px" }}>{Number(row.cate_arm_3_vs_0 || 0).toFixed(3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}









