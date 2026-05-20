import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))
from settings import PROCESSED, RESULTS, NW_LAG, BOOT_BLOCK, BOOT_REPS, SEED, EVENTS
from econometrics import log, hr, ols_nw, baron_kenny, block_bootstrap_resample, boot_ci

panel_main = pd.read_parquet(PROCESSED / "panel.parquet")
panel_xtra = pd.read_parquet(PROCESSED / "panel_extra.parquet")
panel = panel_main.copy()
OUT = RESULTS / "tables"

def build_event_ind():
    ind = pd.Series(0, index=panel.index)
    for ev_name, ev_str, _ in EVENTS:
        ev_t = pd.Timestamp(ev_str, tz="UTC")
        if ev_t < panel.index.min() or ev_t > panel.index.max():
            continue
        mask = ((panel.index >= ev_t - pd.Timedelta(hours=72)) &
                (panel.index <= ev_t + pd.Timedelta(hours=72)))
        ind = ind + mask.astype(int)
    return (ind > 0).astype(float)

ev_ind = build_event_ind()

panel["L_proxy_BTC"] = panel["eta_BTC_binance_ann"].abs().rolling(24).mean()
panel["L_proxy_ETH"] = panel["eta_ETH_binance_ann"].abs().rolling(24).mean()

hr()
log("#1: IV + L_t simultaneous control")
hr()
log("""
  Strategy: 2SLS where δ is endogenous (instrument with Z=event indicator),
            and L_t is included as exogenous control.
  If γ_IV remains positive and substantial after L_t control,
  then gamma magnitude survives the L24 critique.
""")

records = []
for stable in ["USDT", "USDC"]:
    d = panel[f"delta_{stable}"]
    for asset in ["BTC", "ETH"]:
        x = panel[f"basis_{asset}"]
        L = panel[f"L_proxy_{asset}"]
        df = pd.concat([x.rename("X"), d.rename("delta"),
                        ev_ind.rename("Z"), L.rename("L_proxy")], axis=1).dropna()

        X1 = sm.add_constant(df[["Z", "L_proxy"]])
        fs = sm.OLS(df["delta"], X1).fit()
        delta_hat = fs.fittedvalues
        X2 = sm.add_constant(pd.DataFrame({"delta_hat": delta_hat, "L_proxy": df["L_proxy"]}, index=df.index))
        ss = sm.OLS(df["X"], X2).fit()
        gamma_iv_with_L = float(ss.params["delta_hat"])
        L_coef = float(ss.params["L_proxy"])

        X1_basic = sm.add_constant(df[["Z"]])
        fs_b = sm.OLS(df["delta"], X1_basic).fit()
        delta_hat_b = fs_b.fittedvalues
        X2_basic = sm.add_constant(pd.DataFrame({"delta_hat": delta_hat_b}, index=df.index))
        ss_b = sm.OLS(df["X"], X2_basic).fit()
        gamma_iv_basic = float(ss_b.params["delta_hat"])

        ols_L = ols_nw(df["X"], df[["delta", "L_proxy"]], lag=NW_LAG)
        gamma_ols_with_L = float(ols_L.params["delta"])

        log(f"\n  {stable} x basis_{asset}:")
        log(f"    OLS (no controls):           γ = +{x.cov(d)/d.var():.4f}")
        log(f"    OLS + L_proxy:               γ = {gamma_ols_with_L:+.4f}")
        log(f"    IV (no L control, E13):      γ = {gamma_iv_basic:+.4f}")
        log(f"    **IV + L_proxy control**:    γ = {gamma_iv_with_L:+.4f}  (L_coef = {L_coef:+.3f})")

        records.append({
            "stable": stable, "asset": asset,
            "ols_with_L": gamma_ols_with_L,
            "iv_basic": gamma_iv_basic,
            "iv_with_L": gamma_iv_with_L,
            "L_coef_in_2nd_stage": L_coef,
            "first_stage_F_with_L": float(fs.fvalue),
        })

pd.DataFrame(records).to_parquet(OUT / "fix_1_iv_with_L.parquet", index=False)

hr()
log("#2: P1 quantitative ratio test")
hr()
log("""
  Theory P1: γ ∝ κ. So γ_USDT/γ_USDC = κ_USDT/κ_USDC (assuming same ρ-λ).
  Compute empirical γ ratio across multiple specs and compare to κ ratio.
""")

e1 = pd.read_parquet(OUT / "e1_first_stage.parquet")

ratio_records = []
for asset in ["BTC", "ETH"]:
    g_T = float(e1[(e1["spec"] == "M3_joint") & (e1["asset"] == asset)]["delta_USDT_coef"].iloc[0])
    g_C = float(e1[(e1["spec"] == "M3_joint") & (e1["asset"] == asset)]["delta_USDC_coef"].iloc[0])
    ratio_OLS = g_T / g_C if g_C != 0 else np.nan

    e13 = pd.read_parquet(OUT / "e13_iv_gmm.parquet")
    g_iv_T = float(e13[(e13["stable"] == "USDT") & (e13["asset"] == asset)]["gamma_IV"].iloc[0])
    g_iv_C = float(e13[(e13["stable"] == "USDC") & (e13["asset"] == asset)]["gamma_IV"].iloc[0])
    ratio_IV = g_iv_T / g_iv_C if g_iv_C != 0 else np.nan

    e21 = pd.read_parquet(OUT / "e21_rigobon.parquet")
    g_R_T = float(e21[(e21["stable"] == "USDT") & (e21["asset"] == asset)]["gamma_Rigobon"].iloc[0])
    g_R_C = float(e21[(e21["stable"] == "USDC") & (e21["asset"] == asset)]["gamma_Rigobon"].iloc[0])
    ratio_R = g_R_T / g_R_C if g_R_C != 0 else np.nan

    log(f"\n  basis_{asset}:")
    log(f"    OLS M3:    γ_USDT/γ_USDC = {g_T:.4f}/{g_C:.4f} = {ratio_OLS:.2f}")
    log(f"    IV:        γ_USDT/γ_USDC = {g_iv_T:.4f}/{g_iv_C:.4f} = {ratio_IV:.2f}")
    log(f"    Rigobon:   γ_USDT/γ_USDC = {g_R_T:.4f}/{g_R_C:.4f} = {ratio_R:.2f}")

    ratio_records.append({
        "asset": asset,
        "ratio_OLS": ratio_OLS, "ratio_IV": ratio_IV, "ratio_Rigobon": ratio_R,
    })

g_fix2 = pd.read_parquet(OUT / "g_fix2_FINAL.parquet")
kappa_T = float(g_fix2[g_fix2["stable"] == "USDT"]["kappa_emp_pct_apy"].iloc[0])
kappa_C = float(g_fix2[g_fix2["stable"] == "USDC"]["kappa_emp_pct_apy"].iloc[0])
ratio_kappa = kappa_T / kappa_C
log(f"\n  Empirical κ ratio (from Aave APY): κ_USDT/κ_USDC = {kappa_T:.2f}/{kappa_C:.2f} = {ratio_kappa:.2f}")

log(f"\n  THEORETICAL PREDICTION: γ ratio should equal κ ratio (P1).")
for rec in ratio_records:
    log(f"  {rec['asset']}: γ_ratio (OLS={rec['ratio_OLS']:.2f}, IV={rec['ratio_IV']:.2f}, R={rec['ratio_Rigobon']:.2f}) "
        f"vs κ_ratio={ratio_kappa:.2f}")
    rec["kappa_ratio"] = ratio_kappa
    rec["match_OLS"] = abs(rec["ratio_OLS"] - ratio_kappa) / ratio_kappa
    rec["match_IV"] = abs(rec["ratio_IV"] - ratio_kappa) / ratio_kappa if not np.isnan(rec["ratio_IV"]) else np.nan

pd.DataFrame(ratio_records).to_parquet(OUT / "fix_2_p1_ratio.parquet", index=False)

hr()
log("#3: Multi-control mediation - is basis still dominant after multiple controls?")
hr()
log("""
  Add candidate confounders/mediators: BTC vol, funding vol, BTC volume,
  cross-venue spread. Test if Π_med (basis) survives.
""")

panel["btc_vol_24h"] = np.log(panel["spot_BTC"]).diff().rolling(24).std()
panel["funding_vol_BTC"] = panel["eta_BTC_binance_ann"].rolling(24).std()
panel["xvenue_spread_BTC"] = (panel["eta_BTC_binance_ann"] - panel["eta_BTC_bybit_ann"]).abs()

records = []
for stable in ["USDT"]:
    d = panel[f"delta_{stable}"]
    for asset in ["BTC", "ETH"]:
        b = panel[f"basis_{asset}"]
        e = panel[f"eta_{asset}_binance_ann"]

        bk0 = baron_kenny(d, b, e, lag=NW_LAG)
        log(f"\n  {stable}-{asset}:")
        log(f"    Baseline:                     c={bk0['c']:+.3f}  c'={bk0['c_prime']:+.3f}  "
            f"prop_med={bk0['prop_mediated']:+.3f}")

        controls = panel[["btc_vol_24h", "funding_vol_BTC", "xvenue_spread_BTC"]].copy()
        common_idx = pd.concat([d, b, e, controls], axis=1).dropna().index

        d_c = d.loc[common_idx]; b_c = b.loc[common_idx]; e_c = e.loc[common_idx]
        ctrl_c = controls.loc[common_idx]

        def resid_on(y, X):
            Xc = sm.add_constant(X)
            res = sm.OLS(y, Xc).fit()
            return y - res.fittedvalues

        d_res = resid_on(d_c, ctrl_c)
        b_res = resid_on(b_c, ctrl_c)
        e_res = resid_on(e_c, ctrl_c)

        bk1 = baron_kenny(d_res, b_res, e_res, lag=NW_LAG)
        log(f"    After residualizing on 3 ctls: c={bk1['c']:+.3f}  c'={bk1['c_prime']:+.3f}  "
            f"prop_med={bk1['prop_mediated']:+.3f}  (Sobel z={bk1['sobel_z']:+.2f})")

        records.append({
            "stable": stable, "asset": asset,
            "prop_med_baseline": bk0["prop_mediated"], "sobel_baseline": bk0["sobel_z"],
            "prop_med_with_controls": bk1["prop_mediated"], "sobel_with_controls": bk1["sobel_z"],
        })

pd.DataFrame(records).to_parquet(OUT / "fix_3_multicontrol_mediation.parquet", index=False)

hr()
log("#6: λγ product test - does -λγ predict reduced-form c?")
hr()
log("""
  Theory: c_δ = -λγ. Test: estimate λ̂, γ̂ separately, compare product to c.
""")

e2 = pd.read_parquet(OUT / "e2_second_stage.parquet")
e3 = pd.read_parquet(OUT / "e3_reduced_form.parquet")

records = []
for stable in ["USDT", "USDC"]:
    for asset in ["BTC", "ETH"]:
        for venue in ["binance", "bybit"]:
            g = float(e1[(e1["spec"] == "M3_joint") & (e1["asset"] == asset)][f"delta_{stable}_coef"].iloc[0])
            l = float(e2[(e2["asset"] == asset) & (e2["venue"] == venue)]["lambda"].iloc[0])
            implied = -l * g

            row = e3[(e3["asset"] == asset) & (e3["venue"] == venue) & (e3["stablecoin"] == stable)]
            actual_c = float(row["c"].iloc[0])
            residual_pct = (actual_c - implied) / actual_c * 100 if actual_c != 0 else np.nan

            log(f"  {stable} x {asset}_{venue}:  -λ×γ = {implied:+.3f}  vs  c (reduced) = {actual_c:+.3f}  residual={residual_pct:+.1f}%")
            records.append({
                "stable": stable, "asset": asset, "venue": venue,
                "lambda": l, "gamma": g, "implied": implied, "actual_c": actual_c,
                "residual_pct": residual_pct,
            })

pd.DataFrame(records).to_parquet(OUT / "fix_6_lambdaGamma.parquet", index=False)

hr()
log("#5: Joint cross-asset panel block bootstrap (preserves cross-asset error correlation)")
hr()

ASSETS = ["BTC", "ETH", "SOL", "XRP", "BNB", "DOGE", "ADA", "LTC"]

def get_basis(asset):
    if asset in ["BTC", "ETH"]:
        return panel_main[f"basis_{asset}"]
    return panel_xtra[f"basis_{asset}"]

common_idx = panel_main.index
basis_panel = pd.DataFrame(index=common_idx)
for a in ASSETS:
    basis_panel[a] = get_basis(a)

delta = panel_main["delta_USDT"]

n_obs = len(common_idx)
rng = np.random.default_rng(SEED)
B = BOOT_REPS
boot_gammas = np.empty((B, len(ASSETS)))

for b in range(B):
    idx = block_bootstrap_resample(n_obs, BOOT_BLOCK, rng)
    d_b = delta.iloc[idx].to_numpy()
    for j, a in enumerate(ASSETS):
        x_b = basis_panel[a].iloc[idx].to_numpy()
        m = ~np.isnan(d_b) & ~np.isnan(x_b)
        if m.sum() < 100:
            boot_gammas[b, j] = np.nan
            continue
        Xm = np.column_stack([np.ones(m.sum()), d_b[m]])
        beta, *_ = np.linalg.lstsq(Xm, x_b[m], rcond=None)
        boot_gammas[b, j] = beta[1]

log(f"\n  Joint panel bootstrap (B={B}, block={BOOT_BLOCK}, seed={SEED}):")
log(f"  {'Asset':6s} {'γ_median':>10s} {'95% CI':>30s}")
records = []
for j, a in enumerate(ASSETS):
    g = boot_gammas[:, j]
    g_clean = g[~np.isnan(g)]
    if len(g_clean) < 10:
        continue
    median = float(np.median(g_clean))
    ci_lo = float(np.quantile(g_clean, 0.025))
    ci_hi = float(np.quantile(g_clean, 0.975))
    log(f"  {a:6s} {median:>+10.4f}  [{ci_lo:+.4f}, {ci_hi:+.4f}]")
    records.append({"asset": a, "gamma_median": median,
                    "ci_lo": ci_lo, "ci_hi": ci_hi,
                    "joint_panel_boot": True})

mean_g_per_boot = np.nanmean(boot_gammas, axis=1)
log(f"\n  Cross-asset POOLED γ (panel mean):  median={np.median(mean_g_per_boot):.4f}  "
    f"95% CI [{np.quantile(mean_g_per_boot, 0.025):+.4f}, {np.quantile(mean_g_per_boot, 0.975):+.4f}]")
records.append({"asset": "POOLED_8", "gamma_median": float(np.median(mean_g_per_boot)),
                "ci_lo": float(np.quantile(mean_g_per_boot, 0.025)),
                "ci_hi": float(np.quantile(mean_g_per_boot, 0.975)),
                "joint_panel_boot": True})

pd.DataFrame(records).to_parquet(OUT / "fix_5_joint_panel_boot.parquet", index=False)
log("\nFinal strengthening done.")
