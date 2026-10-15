import React from 'react';

export default function BudgetInput({ budget, setBudget, budgetError, setBudgetError }) {
  const handleChange = (e) => {
    const val = e.target.value;
    setBudget(val);

    if (val === '') {
      setBudgetError('Budget cannot be empty.');
      return;
    }

    const num = Number(val);
    if (isNaN(num)) {
      setBudgetError('Budget must be a valid numeric value.');
    } else if (num <= 0) {
      setBudgetError('Budget must be a positive number greater than 0.');
    } else if (!Number.isFinite(num)) {
      setBudgetError('Budget must be a finite number.');
    } else {
      setBudgetError('');
    }
  };

  return (
    <section className="card">
      <h2 className="card-title">Campaign Budget Constraint</h2>
      <p className="card-description">
        Define the total marketing budget allocated for discount policies. 
        <em> (Default $5,000 is an illustrative example. Actual budget optimization will run in subsequent phases.)</em>
      </p>

      <div className="form-group">
        <label htmlFor="budget-input" className="form-label">
          Budget Constraint ($ USD):
        </label>
        <div className="input-wrapper">
          <span className="currency-symbol">$</span>
          <input
            id="budget-input"
            type="number"
            step="100"
            min="1"
            value={budget}
            onChange={handleChange}
            placeholder="5000"
            className={`form-input ${budgetError ? 'input-error' : ''}`}
            aria-describedby={budgetError ? 'budget-error-msg' : undefined}
          />
        </div>
      </div>

      {budgetError ? (
        <div id="budget-error-msg" className="feedback-banner error-banner" role="alert">
          {budgetError}
        </div>
      ) : (
        <div className="feedback-banner valid-banner">
          Valid budget constraint: ${Number(budget).toLocaleString()} USD
        </div>
      )}
    </section>
  );
}
