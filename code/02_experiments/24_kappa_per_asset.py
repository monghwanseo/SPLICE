import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).parent.parent))
from settings import PROCESSED, RESULTS, NW_LAG, ANNUALIZATION_8H

T = RESULTS / "tables"
DELTA_HOURS = 1.0
HOURS_PER_YEAR = 8760.0

panel = pd.read_parquet(PROCESSED / "panel.parquet")
extra = pd.read_parquet(PROCESSED / "panel_extra.parquet")
e10 = pd.read_parquet(T / "e10_cross_asset.parquet").set_index("asset")
e1 = pd.read_parquet(T / "e1_first_stage.parquet")
m3 = e1[e1["spec"] == "M3_joint"].set_index("asset")
e2 = pd.read_parquet(T / "e2_second_stage.parquet")
fix5 = pd.read_parquet(T / "fix_5_joint_panel_boot.parquet").set_index("asset")

print("=" * 80)
print("(b) κ_a recovery per asset (Theorem 1 inversion)")
print("=" * 80)

assets_order = ["BTC", "ETH", "BNB", "SOL", "XRP", "DOGE", "ADA", "LTC"]

def basis_series(asset):
    col = f"basis_{asset}"
    if col in panel.columns:
        return panel[col].dropna()
    if col in extra.columns:
        return extra[col].dropna()
    raise KeyError(f"basis_{asset} not found")

def estimate_rho(b):
    b = b.dropna()
    y = b.iloc[1:].values
    x = b.iloc[:-1].values
    X = sm.add_constant(x)
    res = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": NW_LAG})
    phi = float(res.params[1])
    if phi <= 0 or phi >= 1:
        return np.nan, np.nan, len(y)
    rho = -np.log(phi)
    se_rho = float(res.bse[1]) / phi
    return rho, se_rho, len(y)

def estimate_lambda_per_hour(asset):
    if asset in ("BTC", "ETH"):
        sub = e2[e2["asset"] == asset]
        lam_per_hour = float(sub["lambda"].mean()) / (365 * 24 / DELTA_HOURS) * (1 / DELTA_HOURS)
        lam_per_hour = float(sub["lambda"].mean()) / (365 * 24)
        lam_se = float(sub["se"].mean()) / (365 * 24)
        return lam_per_hour, lam_se
    if asset not in e10.index:
        return np.nan, np.nan
    r = e10.loc[asset]
    lam_ann = (float(r["lambda_binance"]) + float(r["lambda_bybit"])) / 2
    lam_per_hour = lam_ann / (365 * 24)
    return lam_per_hour, np.nan

def estimate_gamma(asset):
    if asset in m3.index:
        g = float(m3.loc[asset, "delta_USDT_coef"])
        t = float(m3.loc[asset, "delta_USDT_t"])
        se = abs(g / t) if t != 0 else np.nan
        return g, se, False
    if asset in e10.index:
        g = float(e10.loc[asset, "gamma_USDT"])
        t = float(e10.loc[asset, "t_USDT"])
        se = abs(g / t) if t != 0 else np.nan
        return g, se, False
    if asset in fix5.index:
        return float(fix5.loc[asset, "gamma_median"]), \
               float(fix5.loc[asset, "ci_hi"] - fix5.loc[asset, "ci_lo"]) / (2 * 1.96), \
               True
    return np.nan, np.nan, False

records = []
rho_list = []
lambda_list = []

for asset in assets_order:
    try:
        b = basis_series(asset)
    except KeyError:
        print(f"  {asset}: no basis series — skipping")
        continue

    rho, se_rho, n_basis = estimate_rho(b)
    lam, se_lam = estimate_lambda_per_hour(asset)
    gam, se_gam, used_fix5 = estimate_gamma(asset)

    if any(pd.isna(x) for x in [rho, lam, gam]):
        print(f"  {asset}: missing ρ/λ/γ — skipping")
        continue

    if rho - lam <= 0:
        print(f"  {asset}: ρ-λ ≤ 0 (={rho-lam:.6f}) — Theorem 1 inapplicable")
        continue

    kappa = gam * lam / DELTA_HOURS

    se_lam_safe = se_lam if (se_lam is not None and not np.isnan(se_lam)) else 0.0
    var_kappa = (lam / DELTA_HOURS)**2 * se_gam**2 + \
                (gam / DELTA_HOURS)**2 * se_lam_safe**2
    se_kappa = np.sqrt(var_kappa)
    kappa_t = kappa / se_kappa if se_kappa > 0 else np.nan

    predicted_gamma = kappa * DELTA_HOURS / (rho - lam)
    over_id_resid_pct = abs(gam - predicted_gamma) / abs(gam) * 100

    kappa_apy_pct = kappa * HOURS_PER_YEAR * 100

    records.append({
        "asset": asset,
        "gamma_USDT": gam,
        "gamma_se": se_gam,
        "rho_per_hour": rho,
        "rho_se": se_rho,
        "lambda_per_hour": lam,
        "kappa_per_hour": kappa,
        "kappa_se_delta_method": se_kappa,
        "kappa_t": kappa_t,
        "kappa_apy_pct": kappa_apy_pct,
        "predicted_gamma": predicted_gamma,
        "over_id_residual_pct": over_id_resid_pct,
        "n_basis": n_basis,
        "used_fix5": used_fix5,
    })
    rho_list.append(rho)
    lambda_list.append(lam)

    print(f"  {asset}:  γ={gam:+.4f}  ρ={rho:.4f}/h  λ={lam:.6f}/h  "
          f"κ={kappa:.5f}/h  κ-APY={kappa_apy_pct:.1f}%  t_κ={kappa_t:.2f}")

df = pd.DataFrame(records)
df.to_parquet(T / "kappa_per_asset.parquet", index=False)
print(f"\n→ Saved {T / 'kappa_per_asset.parquet'}")

print("\n" + "=" * 80)
print("Cross-section diagnostics")
print("=" * 80)
n_pos = (df["kappa_per_hour"] > 0).sum()
print(f"κ_a > 0 in {n_pos}/{len(df)} assets")

kappa_pooled = df["kappa_per_hour"].mean()
kappa_pooled_se = df["kappa_per_hour"].std() / np.sqrt(len(df))
print(f"Pooled κ̂ = {kappa_pooled:.5f}/h  (SD across assets: {df['kappa_per_hour'].std():.5f})")
print(f"Pooled κ̂ APY = {kappa_pooled * HOURS_PER_YEAR * 100:.1f}%")

cv = df["kappa_per_hour"].std() / df["kappa_per_hour"].mean() if df["kappa_per_hour"].mean() > 0 else np.nan
print(f"Coefficient of variation (CV) of κ_a: {cv:.3f}")

print("\n" + "=" * 80)
print("(b/2★) Theorem 2★ spectral-radius condition")
print("=" * 80)

r_f_per_hour = 0.05 / HOURS_PER_YEAR
DELTA_EVENT = 8.0

Lambda_diag = np.array(lambda_list) * DELTA_EVENT
rho_spec_Lambda = float(Lambda_diag.max())

print(f"r_f (per hour) = {r_f_per_hour:.6e}    Δ_event = {DELTA_EVENT}h")
print(f"max(λ_a × Δ_event) = ρ_spec(Λ) = {rho_spec_Lambda:.4f}")
print()
print(f"{'β (per hour)':>15}  {'K_β':>14}  {'ρ_spec·K_β':>14}  {'contraction':>11}")
print("-" * 60)

records = []
for beta in [-1.0, -0.1, -0.05, -0.01, 0.0, r_f_per_hour - 1e-7]:
    K_beta = 1.0 / (1.0 - np.exp(-(r_f_per_hour - beta) * DELTA_EVENT))
    prod = rho_spec_Lambda * K_beta
    ok = (prod < 1.0)
    print(f"{beta:>15.6f}  {K_beta:>14.4e}  {prod:>14.4e}  {'PASS' if ok else 'FAIL':>11}")
    records.append({
        "beta": beta,
        "K_beta": K_beta,
        "rho_spec_Lambda": rho_spec_Lambda,
        "product": prod,
        "contraction_OK": ok,
        "r_f_per_hour": r_f_per_hour,
        "Delta_event_hours": DELTA_EVENT,
    })

if rho_spec_Lambda < 1.0:
    beta_sharp = r_f_per_hour - (1.0 / DELTA_EVENT) * np.log(1.0 / (1.0 - rho_spec_Lambda))
    print(f"\nSharp β at boundary (ρ·K_β = 1): β* = {beta_sharp:.6f} per hour")
    print(f"  → For any β < {beta_sharp:.6f}/h, the augmented operator T_δ is a contraction "
          f"on X_β^d (Theorem 2★).")
else:
    beta_sharp = np.nan
    print(f"\nρ_spec(Λ) ≥ 1 — Theorem 2★ contraction CANNOT hold for any β < r_f.")

records.append({
    "beta": beta_sharp,
    "K_beta": np.nan,
    "rho_spec_Lambda": rho_spec_Lambda,
    "product": 1.0 if not np.isnan(beta_sharp) else np.nan,
    "contraction_OK": True if not np.isnan(beta_sharp) else False,
    "r_f_per_hour": r_f_per_hour,
    "Delta_event_hours": DELTA_EVENT,
})

pd.DataFrame(records).to_parquet(T / "spectral_radius_check.parquet", index=False)
print(f"\n→ Saved {T / 'spectral_radius_check.parquet'}")

print("\n" + "=" * 80)
print("Summary")
print("=" * 80)
print(df[["asset", "gamma_USDT", "rho_per_hour", "lambda_per_hour",
          "kappa_per_hour", "kappa_apy_pct"]].to_string(index=False))
