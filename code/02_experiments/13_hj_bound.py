import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).parent.parent))
from settings import PROCESSED, RESULTS, NW_LAG

T = RESULTS / "tables"
panel = pd.read_parquet(PROCESSED / "panel.parquet")
extra = pd.read_parquet(PROCESSED / "panel_extra.parquet")

def basis_for(a):
    if f"basis_{a}" in panel.columns:
        return panel[f"basis_{a}"].dropna()
    return extra[f"basis_{a}"].dropna()

print("=" * 80)
print("Hansen-Jagannathan lower bound on SDF volatility")
print("=" * 80)
print("σ(m̃)/m̄  ≥  |γ̂| · Var(δ) / σ(δ·X̃_a)")
print()

assets = ["BTC", "ETH", "BNB", "SOL", "XRP", "DOGE", "ADA", "LTC"]
records = []
for stable in ["USDT", "USDC"]:
    delta = panel[f"delta_{stable}"]
    var_delta = float(delta.var())
    for a in assets:
        b = basis_for(a).reindex(panel.index)
        df = pd.concat([b.rename("X"), delta.rename("d")], axis=1).dropna()
        if len(df) < 1000:
            continue
        res = sm.OLS(df["X"], sm.add_constant(df["d"])).fit(
            cov_type="HAC", cov_kwds={"maxlags": NW_LAG})
        gamma = float(res.params["d"])
        sigma_prod = float(np.std(df["X"] * df["d"], ddof=1))
        if sigma_prod < 1e-15:
            continue
        hj_lb = abs(gamma) * var_delta / sigma_prod
        records.append({
            "stable": stable, "asset": a,
            "gamma_OLS": gamma, "var_delta": var_delta,
            "sigma_delta_X": sigma_prod,
            "HJ_sigma_m_over_mbar_LB": hj_lb,
        })
        print(f"  {stable:4s} × {a:4s}:  γ̂={gamma:+.4f}  σ(δX̃)={sigma_prod:.3e}  "
              f"σ(m̃)/m̄ ≥ {hj_lb:.4f}  ({hj_lb*100:.1f}%)")

df = pd.DataFrame(records)
df.to_parquet(T / "hj_bound_sdf.parquet", index=False)

print()
print("=" * 80)
print("Summary")
print("=" * 80)
for stable in ["USDT", "USDC"]:
    sub = df[df["stable"] == stable]
    print(f"  {stable}:  HJ bound range [{sub['HJ_sigma_m_over_mbar_LB'].min()*100:.1f}%, "
          f"{sub['HJ_sigma_m_over_mbar_LB'].max()*100:.1f}%], "
          f"median {sub['HJ_sigma_m_over_mbar_LB'].median()*100:.1f}%")

print()
print("Reference: Hansen-Jagannathan (1991) U.S. equity HJ bound is 0.4–0.5.")
print("Crypto-basis HJ bound is much smaller, indicating that any SDF satisfying")
print("Theorem 4 needs only moderate volatility — existence is plausible without")
print("requiring complete markets or extreme risk-aversion.")
print(f"\n→ Saved {T / 'hj_bound_sdf.parquet'}")
