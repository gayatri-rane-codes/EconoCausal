# EconoCausal: Demonstration Walkthrough

## 1. Objective
Prescriptive discount optimization across 4 arms (0%, 5%, 10%, 20%) using Double Machine Learning (LinearDML/ForestDML) and SciPy HiGHS knapsack under strict budget constraints.

## 2. Walkthrough Steps
1. Backend Tests: .venv\Scripts\python.exe -m pytest -q
2. Backend Server: .venv\Scripts\python.exe -m uvicorn src.api.main:app --port 8000
3. Frontend UI: cd frontend; npm run dev

## 3. Demo Endpoints
- /health: Service readiness
- /causal/predict: Personalized CATE uplifts
- /optimization/pipeline: HiGHS discrete allocations
- /monitoring/data-drift: Kolmogorov-Smirnov and Chi-Square distribution drift tests

## 4. Limitations
- State held client-side in React memory
- Local CORS set to allow all origins
- Docker excluded per project scope
