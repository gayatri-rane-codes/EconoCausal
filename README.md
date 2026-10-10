# EconoCausal: Dynamic Pricing via Double Machine Learning

## Project Overview
EconoCausal is an econometrics and causal machine learning system designed to estimate the causal impact of promotional discounts on customer purchase behavior and optimize marketing budget allocation.

Unlike conventional predictive machine learning models that predict correlation (e.g., purchase likelihood), EconoCausal uses Double/Debiased Machine Learning (DML) to estimate counterfactual Individual Treatment Effects (ITE)—identifying customers who purchase *because* of an incentive rather than those who would purchase regardless.

## Core Architecture
- **Module A: Causal Inference Engine**: Formulates the Directed Acyclic Graph (DAG) with DoWhy, adjusts for confounders, and trains Double Machine Learning models (EconML) with diagnostic refutation tests.
- **Module B: Propensity & Budget Optimization**: Checks covariate balance/overlap and runs SciPy constrained optimization to maximize predicted revenue under a strict marketing budget constraint.
- **Module C: Backend & REST API**: FastAPI service exposing data validation, causal inference execution, optimization solvers, and automated data drift monitoring.
- **Module D: Experimentation Dashboard**: Interactive React and Plotly.js interface visualizing uplift curves, Qini metrics, and prescriptive customer discount allocations.

## Implementation Roadmap (4 Weeks)
- **Phase 1 (Week 1)**: Causal Graphing, initial data preparation, and React dashboard scaffolding.
- **Phase 2 (Week 2)**: Double Machine Learning estimator training, ITE scoring, Qini/Uplift evaluation, and DoWhy refutation audits.
- **Phase 3 (Week 3)**: Prescriptive optimization via SciPy and discount allocation matrix generation.
- **Phase 4 (Week 4)**: FastAPI integration, data drift detection, end-to-end testing, and review deliverables.

## Development Status
- **Current Phase**: Phase 1 — Day 1 completed.
- **Environment**: Python 3.11 (virtual environment configured), Node.js v24.19.0, npm 11.17.0, Git version control.
- **Working Branch**: gayatri
- **Dataset & Model Status**: Candidate dataset selection and causal DAG construction pending Day 2+ implementation.
