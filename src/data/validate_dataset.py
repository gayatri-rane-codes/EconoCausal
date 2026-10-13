import os
import sys
import numpy as np
import pandas as pd

p_main = "data/raw/econocausal_synthetic_dataset.csv"
p_gt = "data/raw/ground_truth_evaluation.csv"

if not os.path.exists(p_main) or not os.path.exists(p_gt):
    print("Error: Dataset files missing in data/raw/")
    sys.exit(1)

m = pd.read_csv(p_main)
g = pd.read_csv(p_gt)
n = len(m)

print(f"--- Dataset Audit (Rows: {n}) ---")
print(f"1. Main Shape: {m.shape} (Expected: 20000, 45)")
print(f"2. GT Shape: {g.shape} (Expected: 20000, 69)")
print(f"3. Main Nulls: {m.isna().sum().sum()} | GT Nulls: {g.isna().sum().sum()}")
print(f"4. Duplicate IDs: {m['customer_id'].duplicated().sum()}")
print(f"5. ID Matching: {(m['customer_id'].values == g['customer_id'].values).all()}")

# Financial consistency check
y = m['purchased'] == 1
non = ~y
tol = 1e-5

non_ok = (m.loc[non, ['discount_amount', 'revenue', 'cost_of_goods']].abs().values <= tol).all()
profit_non_ok = np.allclose(m.loc[non, 'net_profit'], -m.loc[non, 'marketing_cost'], atol=tol)
disc_ok = np.allclose(m.loc[y, 'discount_amount'], np.round(m.loc[y, 'basket_value_before_discount'] * m.loc[y, 'discount_rate'], 2), atol=tol)
rev_ok = np.allclose(m.loc[y, 'revenue'], np.round(m.loc[y, 'basket_value_before_discount'] - m.loc[y, 'discount_amount'], 2), atol=tol)
prof_all_ok = np.allclose(m['net_profit'], np.round(m['revenue'] - m['cost_of_goods'] - m['marketing_cost'], 2), atol=tol)

print(f"6. Non-buyer zeros check: {non_ok}")
print(f"7. Non-buyer profit == -marketing_cost: {profit_non_ok}")
print(f"8. Buyer discount calculation: {disc_ok}")
print(f"9. Buyer revenue calculation: {rev_ok}")
print(f"10. Net profit identity across all rows: {prof_all_ok}")

# Treatment distribution
print("\n--- Treatment Arm Distribution ---")
print(m['discount_pct'].value_counts().sort_index())

# Naive vs True ATE comparison
print("\n--- Confounding & Treatment Effect Check ---")
for k in [1, 2, 3]:
    pct = [0, 5, 10, 20][k]
    naive = m.loc[m['treatment_arm'] == k, 'purchased'].mean() - m.loc[m['treatment_arm'] == 0, 'purchased'].mean()
    true_ate = g[f'true_ite_purchase_arm{k}'].mean()
    print(f"Arm {pct}% vs 0% -> Naive diff: {naive:+.4f} | True ATE: {true_ate:+.4f}")

print("\n--- Validation Execution Finished ---")
