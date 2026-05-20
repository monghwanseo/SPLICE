import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).parent.parent))
from settings import PROCESSED, RESULTS, NW_LAG

T = RESULTS / "tables"
panel = pd.read_parquet(PROCESSED / "panel.parquet")

def imai_keele(delta_t, X_t, eta_t):
    df = pd.concat([delta_t.rename("d"), X_t.rename("X"), eta_t.rename("y")], axis=1).dropna()
    if len(df) < 1000:
        return None
    res_M = sm.OLS(df["X"], sm.add_constant(df["d"])).fit(
        cov_type="HAC", cov_kwds={"maxlags": NW_LAG})
    beta = float(res_M.params["d"])
    sigma_M = float(np.sqrt(res_M.scale))
    R2_X_d = float(res_M.rsquared)

    res_Y = sm.OLS(df["y"], sm.add_constant(df[["d", "X"]])).fit(
        cov_type="HAC", cov_kwds={"maxlags": NW_LAG})
    gamma_direct = float(res_Y.params["d"])
    lam = float(res_Y.params["X"])
    sigma_Y = float(np.sqrt(res_Y.scale))

    res_T = sm.OLS(df["y"], sm.add_constant(df["d"])).fit(
        cov_type="HAC", cov_kwds={"maxlags": NW_LAG})
    total = float(res_T.params["d"])

    indirect = beta * lam
    direct = gamma_direct
    share = indirect / total if abs(total) > 1e-12 else np.nan

    if 1 - R2_X_d > 0:
        rho_star = lam * sigma_M / (sigma_Y * np.sqrt(1 - R2_X_d))
    else:
        rho_star = np.nan

    return {
        "beta_M": beta, "lambda_Y": lam,
        "indirect": indirect, "direct": direct, "total": total,
        "mediation_share": share,
        "rho_star": rho_star, "abs_rho_star": abs(rho_star),
        "sigma_M": sigma_M, "sigma_Y": sigma_Y, "R2_X_given_d": R2_X_d,
        "n": len(df),
    }

print("=" * 80)
print("Imai-Keele-Yamamoto sensitivity for mediation (basis as mediator)")
print("=" * 80)
print("Mediation share % depends on sequential ignorability (Cor(ε_M, ε_Y|δ) = 0).")
print("ρ* is the smallest |Corr(ε_M, ε_Y)| that drives the indirect effect to 0.")
print("Convention: |ρ*| > 0.3 is considered robust to plausible unobserved confounders.\n")

records = []
for stable in ["USDT", "USDC"]:
    d = panel[f"delta_{stable}"]
    for asset in ["BTC", "ETH"]:
        X = panel[f"basis_{asset}"]
        for venue in ["binance", "bybit"]:
            eta = panel.get(f"eta_{asset}_{venue}_ann")
            if eta is None:
                continue
            r = imai_keele(d, X, eta)
            if r is None:
                continue
            r.update({"stable": stable, "asset": asset, "venue": venue})
            records.append(r)
            robust = "✓ robust" if abs(r["rho_star"]) > 0.3 else "  (sensitive)"
            print(f"  {stable} {asset} {venue:7s}: indirect={r['indirect']:+.4e}  "
                  f"share={r['mediation_share']*100:+.1f}%  ρ*={r['rho_star']:+.3f}  {robust}")

df = pd.DataFrame(records)
df.to_parquet(T / "imai_keele_sensitivity.parquet", index=False)

print()
print("=" * 80)
print(f"Summary across {len(df)} cells")
print("=" * 80)
n_robust = (df["abs_rho_star"] > 0.3).sum()
n_total = len(df)
print(f"  ρ* > 0.3 (robust to plausible confounders) in {n_robust}/{n_total} cells")
print(f"  Median |ρ*| = {df['abs_rho_star'].median():.3f}")
print(f"  Min |ρ*| = {df['abs_rho_star'].min():.3f}    Max |ρ*| = {df['abs_rho_star'].max():.3f}")
print()
for stab in ["USDT", "USDC"]:
    sub = df[df["stable"] == stab]
    print(f"  {stab}: median |ρ*| = {sub['abs_rho_star'].median():.3f}, "
          f"share range [{sub['mediation_share'].min()*100:.0f}%, "
          f"{sub['mediation_share'].max()*100:.0f}%]")

print(f"\n→ Saved {T / 'imai_keele_sensitivity.parquet'}")
