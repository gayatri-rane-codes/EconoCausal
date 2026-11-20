# EconoCausal – Dynamic Pricing via Double Machine Learning

EconoCausal is an end-to-end causal machine learning and prescriptive optimization platform for e-commerce promotional discount allocation, combining ForestDML and SciPy HiGHS MILP.

## Workflow

1. Customer Ingestion (8 Confounders)
2. CATE Uplift Estimation (ForestDML)
3. Policy Matrix Construction
4. Prescriptive Optimization (SciPy HiGHS MILP / Greedy)
5. Non-Parametric Data Drift Monitoring (KS & Chi-Square)
6. React 19 Experimentation Dashboard

## API Endpoints (10 Routes)

- `GET /`
- `GET /health`
- `POST /causal/estimate`
- `POST /causal/predict`
- `POST /optimization/allocate`
- `POST /optimization/frontier`
- `POST /optimization/sensitivity`
- `POST /optimization/multi-cohort`
- `POST /optimization/pipeline`
- `POST /monitoring/data-drift`

## Setup & Run

**Backend**
```bash
.venv\Scripts\python.exe -m uvicorn src.api.main:app --port 8000
```

**Frontend**
```bash
cd frontend
npm run dev
```

**Tests**
```bash
.venv\Scripts\python.exe -m pytest -v
```
