# Phase 2 Technical Design Document: Double Machine Learning (DML) Formulation

## 1. Executive Formulation & Objective
The primary objective of Phase 2 is to estimate the Conditional Average Treatment Effect (CATE) of multi-level discount pricing on customer purchase probability using Double Machine Learning (DML). 

By leveraging Neyman-orthogonal score functions and cross-fitting, DML strips away confounding bias from observational customer covariates without imposing rigid parametric forms on the high-dimensional relationship between confounders and customer behavior.

---

## 2. Estimator Selection & Rationale

* **Selected Estimator**: `econml.dml.LinearDML`
* **Class Configuration**:
  ```python
  from econml.dml import LinearDML
  from sklearn.ensemble import RandomForestRegressor
  from sklearn.linear_model import LogisticRegression

  dml_estimator = LinearDML(
      model_y=RandomForestRegressor(n_estimators=100, max_depth=6, random_state=20261012),
      model_t=LogisticRegression(multi_class='multinomial', max_iter=1000, random_state=20261012),
      discrete_treatment=True,
      discrete_outcome=False,
      categories=[0, 1, 2, 3],
      cv=5,
      random_state=20261012
  )
