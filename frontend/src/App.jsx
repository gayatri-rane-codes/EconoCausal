import React, { useState, useEffect } from "react";
import axios from "axios";
import ExecutiveOverview from "./components/ExecutiveOverview";
import CustomerConfounders from "./components/CustomerConfounders";
import CATERecommendations from "./components/CATERecommendations";
import BudgetAllocation from "./components/BudgetAllocation";
import DriftObservability from "./components/DriftObservability";

const API_BASE = "http://localhost:8000";

const DEFAULT_CONFOUNDERS = {
  browsing_frequency: 28,
  cart_abandonment_rate: 0.12,
  competitor_price_index: 0.95,
  historical_spend: 420.0,
  loyalty_member: 1,
  previous_discount_response_rate: 0.65,
  recency_days: 4,
  seasonality_index: 1.25,
};

export default function App() {
  const [activeTab, setActiveTab] = useState("overview");
  const [backendHealth, setBackendHealth] = useState("CHECKING");
  const [confounders, setConfounders] = useState(DEFAULT_CONFOUNDERS);
  const [cateResult, setCateResult] = useState(null);
  const [optimizationResult, setOptimizationResult] = useState(null);
  const [driftResult, setDriftResult] = useState(null);

  const checkHealth = async () => {
    try {
      const res = await axios.get(`${API_BASE}/health`);
      if (res.data?.status === "healthy" || res.data?.status === "ok") {
        setBackendHealth("HEALTHY");
      } else {
        setBackendHealth("UNHEALTHY");
      }
    } catch {
      setBackendHealth("DOWN");
    }
  };

  useEffect(() => {
    checkHealth();
    const interval = setInterval(checkHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  const navItems = [
    { id: "overview", label: "Executive Overview", tag: "SYS" },
    { id: "confounders", label: "Customer Confounders", tag: "INP" },
    { id: "cate", label: "CATE Recommendations", tag: "INF" },
    { id: "budget", label: "Budget Allocation", tag: "OPT" },
    { id: "drift", label: "Drift & Observability", tag: "OBS" },
  ];

  return (
    <div className="app-container">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-logo">E</div>
          <div className="brand-text">
            <h1>EconoCausal</h1>
            <span>Analytics Suite</span>
          </div>
        </div>

        <nav className="nav-links">
          {navItems.map((item) => (
            <button
              key={item.id}
              className={`nav-item ${activeTab === item.id ? "active" : ""}`}
              onClick={() => setActiveTab(item.id)}
            >
              <span>{item.label}</span>
            </button>
          ))}
        </nav>

        <div className="sidebar-footer">
          <div>EconoCausal Enterprise</div>
          <div>DML · HiGHS MILP · SciPy</div>
          <div>API: {API_BASE}</div>
        </div>
      </aside>

      <div className="main-wrapper">
        <header className="top-bar">
          <div className="top-bar-title">
            {navItems.find((n) => n.id === activeTab)?.label}
          </div>
          <div className="top-bar-actions">
            <div className="health-badge">
              <span
                className={`status-dot ${
                  backendHealth === "HEALTHY"
                    ? "healthy"
                    : backendHealth === "DOWN"
                    ? "down"
                    : ""
                }`}
              />
              <span>Backend: {backendHealth}</span>
            </div>
            <button className="btn-secondary" onClick={checkHealth}>
              Refresh
            </button>
          </div>
        </header>

        <main className="content-area">
          {activeTab === "overview" && (
            <ExecutiveOverview
              backendHealth={backendHealth}
              optimizationResult={optimizationResult}
              driftResult={driftResult}
              setActiveTab={setActiveTab}
            />
          )}

          {activeTab === "confounders" && (
            <CustomerConfounders
              confounders={confounders}
              setConfounders={setConfounders}
              onEvaluate={() => setActiveTab("cate")}
            />
          )}

          {activeTab === "cate" && (
            <CATERecommendations
              confounders={confounders}
              cateResult={cateResult}
              setCateResult={setCateResult}
              onEditConfounders={() => setActiveTab("confounders")}
            />
          )}

          {activeTab === "budget" && (
            <BudgetAllocation
              optimizationResult={optimizationResult}
              setOptimizationResult={setOptimizationResult}
            />
          )}

          {activeTab === "drift" && (
            <DriftObservability
              driftResult={driftResult}
              setDriftResult={setDriftResult}
            />
          )}
        </main>
      </div>
    </div>
  );
}
