import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).parent.parent))
from settings import PROCESSED, RESULTS, NW_LAG

T = RESULTS / "tables"
panel = pd.read_parquet(PROCESSED / "panel.parquet")

print("=" * 80)
print("(B) Structural GMM joint estimation")
print("=" * 80)

DELTA_PANEL = 1.0
HOURS_YEAR = 8760.0

def gmm_estimate(asset, venue, stable):
    eta = panel[f"eta_{asset}_{venue}_ann"].dropna() / HOURS_YEAR
    X = panel[f"basis_{asset}"].dropna()
    delta = panel[f"delta_{stable}"].dropna()

    df = pd.concat([eta.rename("eta"), X.rename("X"), delta.rename("d")], axis=1).dropna()
    n = len(df)
    if n < 1000:
        return None

    e = df["eta"].values
    x = df["X"].values
    d = df["d"].values

    res_ar = sm.OLS(x[1:], sm.add_constant(x[:-1])).fit()
    phi_hat = float(res_ar.params[1])
    if phi_hat <= 0 or phi_hat >= 1:
        return None
    rho_hat = -np.log(phi_hat)

    def moments(params):
        kappa, lam, gam, c = params
        xi = x - gam * d
        eps = e + lam * x
        u = e - c - (-lam * gam) * d
        struct1 = gam - kappa * DELTA_PANEL / lam
        struct2 = c + lam * gam

        m = np.array([
            xi.mean(),
            (xi * d).mean(),
            eps.mean(),
            (eps * x).mean(),
            struct1,
            struct2,
        ])
        return m

    def gmm_obj(params):
        m = moments(params)
        return float(m @ m)

    res_ss = sm.OLS(x, sm.add_constant(d)).fit()
    gam0 = float(res_ss.params[1])

    res_s1 = sm.OLS(e, x).fit()
    lam0 = -float(res_s1.params[0])

    if lam0 < 0:
        lam0 = max(rho_hat * 0.3, 1e-4)

    kappa0 = gam0 * lam0 / DELTA_PANEL
    if kappa0 <= 0:
        kappa0 = 0.001
    c0 = -lam0 * gam0

    x0 = np.array([kappa0, lam0, gam0, c0])

    bounds = [
        (1e-6, 0.1),
        (1e-6, 1.0),
        (-0.5, 0.5),
        (-0.05, 0.05),
    ]
    res_opt = minimize(gmm_obj, x0=x0, method="L-BFGS-B", bounds=bounds,
                       options={"ftol": 1e-12, "gtol": 1e-10, "maxiter": 5000})
    kappa_h, lam_h, gam_h, c_h = res_opt.x

    m_at = moments(res_opt.x)
    J_stat = n * float(m_at @ m_at)

    return {
        "asset": asset, "venue": venue, "stable": stable,
        "n": n,
        "rho_per_hour": rho_hat,
        "kappa_per_hour": kappa_h,
        "kappa_apy_pct": kappa_h * HOURS_YEAR * 100,
        "lambda_per_hour": lam_h,
        "lambda_ann": lam_h * HOURS_YEAR,
        "gamma": gam_h,
        "c_rf_per_hour": c_h,
        "c_rf_ann": c_h * HOURS_YEAR,
        "implied_lambda_gamma_per_hour": lam_h * gam_h,
        "J_stat_unweighted": J_stat,
        "moments_norm": float(np.linalg.norm(m_at)),
        "converged": bool(res_opt.success),
    }

cells = [(a, v, s) for a in ["BTC", "ETH"] for v in ["binance", "bybit"]
         for s in ["USDT", "USDC"]]

records = []
for a, v, s in cells:
    res = gmm_estimate(a, v, s)
    if res is None:
        continue
    records.append(res)
    print(f"  {a}-{v}-{s}:  κ={res['kappa_per_hour']:.5f}/h  "
          f"κ_APY={res['kappa_apy_pct']:.0f}%  "
          f"λ={res['lambda_ann']:.1f}/yr  "
          f"γ={res['gamma']:+.4f}  c_ann={res['c_rf_ann']:+.2f}  "
          f"||m||={res['moments_norm']:.4e}")

out = pd.DataFrame(records)
out.to_parquet(T / "structural_gmm.parquet", index=False)
print(f"\n→ Saved {T / 'structural_gmm.parquet'}")

print("\n" + "=" * 80)
print("Comparison: GMM joint estimate vs OLS triangulation (paper §5.3)")
print("=" * 80)
e1 = pd.read_parquet(T / "e1_first_stage.parquet")
e2 = pd.read_parquet(T / "e2_second_stage.parquet")
e3 = pd.read_parquet(T / "e3_reduced_form.parquet")

for r in records:
    a, v, s = r["asset"], r["venue"], r["stable"]
    if s == "USDT":
        m3 = e1[(e1["spec"] == "M3_joint") & (e1["asset"] == a)]
        gam_ols = float(m3["delta_USDT_coef"].iloc[0]) if len(m3) else np.nan
    else:
        m3 = e1[(e1["spec"] == "M3_joint") & (e1["asset"] == a)]
        gam_ols = float(m3["delta_USDC_coef"].iloc[0]) if len(m3) else np.nan
    sub_e2 = e2[(e2["asset"] == a) & (e2["venue"] == v)]
    lam_ols_ann = float(sub_e2["lambda"].iloc[0]) if len(sub_e2) else np.nan
    sub_e3 = e3[(e3["asset"] == a) & (e3["venue"] == v) & (e3["stablecoin"] == s)]
    c_ols = float(sub_e3["c"].iloc[0]) if len(sub_e3) else np.nan

    print(f"  {a}-{v}-{s}:")
    print(f"    γ:    GMM={r['gamma']:+.4f}  OLS={gam_ols:+.4f}  diff={r['gamma']-gam_ols:+.5f}")
    print(f"    λ_ann: GMM={r['lambda_ann']:.1f}  OLS={lam_ols_ann:.1f}  diff={r['lambda_ann']-lam_ols_ann:+.2f}")
    print(f"    c_ann: GMM={r['c_rf_ann']:+.2f}  OLS={c_ols:+.2f}  diff={r['c_rf_ann']-c_ols:+.3f}")
