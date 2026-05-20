import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).parent.parent))
from settings import PROCESSED, RESULTS, NW_LAG

T = RESULTS / "tables"
panel = pd.read_parquet(PROCESSED / "panel.parquet")
roll = pd.read_parquet(T / "kappa_rolling.parquet")

assets = ["BTC", "ETH", "BNB", "SOL", "XRP", "DOGE", "ADA", "LTC"]

def rolling_var_delta(window_h):
    return panel["delta_USDT"].rolling(window_h).var()

WINDOW = 720
var_delta = rolling_var_delta(WINDOW)

print("=" * 80)
print("Theorem 1' Jensen-bias test:  γ̂_t  =  α  -  β · σ_δ²_t  +  ε_t")
print(f"Rolling window: {WINDOW}h ({WINDOW/24:.1f}d)")
print("=" * 80)

records = []
for a in assets:
    df_a = roll[roll["asset"] == a].copy()
    if len(df_a) < 30:
        continue
    df_a["t_end"] = pd.to_datetime(df_a["t_end"])
    df_a = df_a.set_index("t_end").sort_index()

    df_a["var_delta"] = var_delta.reindex(df_a.index, method="nearest")
    df_a = df_a.dropna(subset=["gamma", "var_delta"])
    if len(df_a) < 30:
        continue

    res = sm.OLS(df_a["gamma"], sm.add_constant(df_a["var_delta"])).fit(
        cov_type="HAC", cov_kwds={"maxlags": 3})
    alpha = float(res.params["const"])
    beta = float(res.params["var_delta"])
    se_beta = float(res.bse["var_delta"])
    t_beta = beta / se_beta
    p_beta = float(res.pvalues["var_delta"])
    r2 = float(res.rsquared)

    records.append({
        "asset": a,
        "n_windows": len(df_a),
        "alpha_intercept": alpha,
        "beta_var_delta": beta,
        "se_beta": se_beta,
        "t_beta": t_beta,
        "p_beta": p_beta,
        "r2": r2,
        "mean_gamma": float(df_a["gamma"].mean()),
        "mean_var_delta": float(df_a["var_delta"].mean()),
    })
    sign_label = "Q concave (CARA-like)" if beta < 0 else "Q convex"
    sig = "  ***" if p_beta < 0.01 else ("  **" if p_beta < 0.05 else ("  *" if p_beta < 0.10 else ""))
    print(f"  {a}:  α={alpha:+.4f}  β={beta:+.2f}  (t={t_beta:+.2f}, p={p_beta:.4f})  "
          f"R²={r2:.3f}  →  {sign_label}{sig}")

df = pd.DataFrame(records)
df.to_parquet(T / "jensen_bias_test.parquet", index=False)

print()
print("=" * 80)
print("Joint test across 8 assets")
print("=" * 80)
n_neg = (df["beta_var_delta"] < 0).sum()
n_pos = (df["beta_var_delta"] > 0).sum()
n_total = len(df)
print(f"  β < 0 (Q concave) in {n_neg}/{n_total} assets")
print(f"  β > 0 (Q convex)  in {n_pos}/{n_total} assets")
print(f"  mean β = {df['beta_var_delta'].mean():+.2f}")

zs = df["t_beta"].values
from scipy.stats import norm, binom
z_stouffer = float(zs.sum() / np.sqrt(len(zs)))
p_stouffer = float(norm.cdf(z_stouffer))
print(f"  Stouffer Z = {z_stouffer:+.3f}, one-sided p(β<0) = {p_stouffer:.4f}")

p_binomial = float(1 - binom.cdf(n_neg - 1, n_total, 0.5))
print(f"  Binomial sign test (n_neg/n_total): p = {p_binomial:.4f}")

mean_dist = (df["beta_var_delta"] * df["mean_var_delta"]).mean()
print(f"  Average implied Jensen-bias contribution to γ: {mean_dist:+.5f}")
print(f"  Mean γ (across asset rolling): {df['mean_gamma'].mean():+.5f}")
print(f"  Relative size: {abs(mean_dist) / abs(df['mean_gamma'].mean()) * 100:.2f}% of γ")

print(f"\n→ Saved {T / 'jensen_bias_test.parquet'}")
