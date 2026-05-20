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

print("=" * 80)
print("Theorem 4 expanded multi-SDF robustness (11 candidates)")
print("=" * 80)

log_BTC = np.log(panel["spot_BTC"]).diff()
log_ETH = np.log(panel["spot_ETH"]).diff()

extras_logret = {}
for asset in ["BNB", "SOL", "XRP", "DOGE", "ADA", "LTC"]:
    p = Path("data_raw/binance") / f"{asset}USDT_spot_1h.parquet"
    if p.exists():
        df = pd.read_parquet(p)
        df["ts"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
        df = df.sort_values("ts").set_index("ts")
        s = np.log(df["close"]).diff().reindex(panel.index)
        extras_logret[asset] = s

funding_cols = [c for c in panel.columns if c.startswith("eta_") and c.endswith("_ann")]
funding_cols += [c for c in extra.columns if c.startswith("eta_") and c.endswith("_ann")]
df_fund = pd.concat([panel[[c for c in funding_cols if c in panel.columns]],
                     extra[[c for c in funding_cols if c in extra.columns]]],
                    axis=1)
agg_funding = df_fund.mean(axis=1)

basis_cols = ["basis_BTC", "basis_ETH"] + [f"basis_{a}" for a in ["BNB","SOL","XRP","DOGE","ADA","LTC"]]
df_basis = pd.concat([panel[[c for c in basis_cols if c in panel.columns]],
                      extra[[c for c in basis_cols if c in extra.columns]]],
                     axis=1)
agg_abs_basis = df_basis.abs().mean(axis=1)

all_logret = pd.DataFrame({"BTC": log_BTC, "ETH": log_ETH, **extras_logret}).dropna()
print(f"  PC1 panel: {len(all_logret)} obs × {all_logret.shape[1]} assets")
returns_demeaned = all_logret - all_logret.mean()
cov = returns_demeaned.cov()
eigvals, eigvecs = np.linalg.eigh(cov.values)
pc1_loadings = eigvecs[:, -1]
PC1 = (returns_demeaned * pc1_loadings).sum(axis=1)
print(f"  PC1 explained variance: {eigvals[-1] / eigvals.sum() * 100:.1f}%")

fund_8 = pd.DataFrame({
    "BTC": panel["eta_BTC_binance_ann"],
    "ETH": panel["eta_ETH_binance_ann"],
    "BNB": extra["eta_BNB_binance_ann"],
    "SOL": extra["eta_SOL_binance_ann"],
    "XRP": extra["eta_XRP_binance_ann"],
    "DOGE": extra["eta_DOGE_binance_ann"],
    "ADA": extra["eta_ADA_binance_ann"],
    "LTC": extra["eta_LTC_binance_ann"],
})
ret_8 = pd.DataFrame({"BTC": log_BTC, "ETH": log_ETH, **extras_logret})
ranks = fund_8.rank(axis=1)
n_assets = ranks.shape[1]
low_mask = ranks <= 2
high_mask = ranks >= n_assets - 1
carry = (ret_8 * low_mask).sum(axis=1) / 2 - (ret_8 * high_mask).sum(axis=1) / 2

SDFs = {
    "S1_LogUtil_BTCETH":    -0.5 * (log_BTC + log_ETH),
    "S2_VolScaled":         -0.5 * (log_BTC + log_ETH) /
                            (-0.5 * (log_BTC + log_ETH)).rolling(168, min_periods=24).std(),
    "S3_BTC_only":          -log_BTC,
    "S4_RiskReversal":      -log_BTC * np.abs(log_BTC),
    "S5_RealVol_BTC":       log_BTC.rolling(168, min_periods=24).std(),
    "S6_RealVol_ETH":       log_ETH.rolling(168, min_periods=24).std(),
    "S7_AggFunding":        agg_funding - agg_funding.rolling(720, min_periods=24).mean(),
    "S8_AggBasisDisp":      agg_abs_basis - agg_abs_basis.rolling(720, min_periods=24).mean(),
    "S9_ETH_only":          -log_ETH,
    "S10_PC1":              -PC1.reindex(panel.index),
    "S11_FundingCarry":     -carry.reindex(panel.index),
}

def beta_M_test(M, delta):
    df = pd.DataFrame({"M": M, "d": delta}).dropna()
    if df["d"].var() < 1e-12 or len(df) < 100:
        return np.nan, np.nan, np.nan, len(df)
    res = sm.OLS(df["M"], sm.add_constant(df["d"])).fit(
        cov_type="HAC", cov_kwds={"maxlags": NW_LAG})
    slope = float(res.params[1])
    se = float(res.bse[1])
    beta_M = -slope
    return beta_M, se, beta_M / se, len(df)

gammas = {}
for stable in ["USDT", "USDC"]:
    df_ss = pd.DataFrame({"basis_BTC": panel["basis_BTC"],
                          "d": panel[f"delta_{stable}"]}).dropna()
    res = sm.OLS(df_ss["basis_BTC"], sm.add_constant(df_ss[["d"]])).fit(
        cov_type="HAC", cov_kwds={"maxlags": NW_LAG})
    gammas[stable] = float(res.params["d"])

print(f"\n  Direct γ_δ:  USDT={gammas['USDT']:+.4f}  USDC={gammas['USDC']:+.4f}\n")
print(f"  {'SDF':>22}  {'stable':>5}  {'β_M':>11}  {'t':>6}  {'sign':>6}  {'n':>7}")
print("  " + "-" * 70)

records = []
for sdf_name, M in SDFs.items():
    for stable in ["USDT", "USDC"]:
        delta = panel[f"delta_{stable}"]
        beta, se, t, n = beta_M_test(M, delta)
        if np.isnan(beta):
            continue
        match = "MATCH" if np.sign(beta) == np.sign(gammas[stable]) else "FLIP"
        print(f"  {sdf_name:>22}  {stable:>5}  {beta:>+11.5f}  {t:>+6.2f}  {match:>6}  {n:>7}")
        records.append({
            "sdf_proxy": sdf_name,
            "stable": stable,
            "beta_M": beta,
            "se": se,
            "t": t,
            "gamma_direct": gammas[stable],
            "sign_match": match,
            "n": n,
        })

out = pd.DataFrame(records)
out.to_parquet(T / "theorem4_multi_sdf_v2.parquet", index=False)
print(f"\n→ Saved {T / 'theorem4_multi_sdf_v2.parquet'}")

RETURN_BASED = ["S1_LogUtil_BTCETH", "S3_BTC_only", "S4_RiskReversal", "S9_ETH_only", "S10_PC1"]
VOL_FUND_BASED = ["S2_VolScaled", "S5_RealVol_BTC", "S6_RealVol_ETH",
                  "S7_AggFunding", "S8_AggBasisDisp", "S11_FundingCarry"]

print("\n" + "=" * 80)
print("Sign-match summary (11 SDFs total) — pooled & by family")
print("=" * 80)

family_records = []
for stable in ["USDT", "USDC"]:
    sub = out[out["stable"] == stable]
    print(f"\n  {stable}:")

    n_match = (sub["sign_match"] == "MATCH").sum()
    pct = 100 * n_match / len(sub)
    print(f"    All 11 SDFs:                    {n_match}/{len(sub)} sign-match  ({pct:.0f}%)")
    family_records.append({"stable": stable, "family": "all_11",
                           "n_match": int(n_match), "n_total": len(sub),
                           "match_pct": pct})

    rb = sub[sub["sdf_proxy"].isin(RETURN_BASED)]
    n_rb = (rb["sign_match"] == "MATCH").sum()
    pct_rb = 100 * n_rb / len(rb)
    print(f"    Return-based (5):                {n_rb}/{len(rb)} sign-match  ({pct_rb:.0f}%)")
    family_records.append({"stable": stable, "family": "return_based",
                           "n_match": int(n_rb), "n_total": len(rb),
                           "match_pct": pct_rb})

    vf = sub[sub["sdf_proxy"].isin(VOL_FUND_BASED)]
    n_vf = (vf["sign_match"] == "MATCH").sum()
    pct_vf = 100 * n_vf / len(vf)
    print(f"    Vol/funding-based (6):           {n_vf}/{len(vf)} sign-match  ({pct_vf:.0f}%)")
    family_records.append({"stable": stable, "family": "vol_funding_based",
                           "n_match": int(n_vf), "n_total": len(vf),
                           "match_pct": pct_vf})

from scipy.stats import binom
print("\n" + "=" * 80)
print("Binomial test on return-based subfamily (n=5 each)")
print("=" * 80)
for stable in ["USDT", "USDC"]:
    sub = out[(out["stable"] == stable) & out["sdf_proxy"].isin(RETURN_BASED)]
    N = len(sub)
    k_match = int((sub["sign_match"] == "MATCH").sum())
    k_flip = N - k_match
    if stable == "USDT":
        p_one = float(1 - binom.cdf(k_match - 1, N, 0.5))
        print(f"  USDT: P(MATCH ≥ {k_match} of {N} | random sign) = {p_one:.4f}")
    else:
        p_one = float(1 - binom.cdf(k_flip - 1, N, 0.5))
        print(f"  USDC: P(FLIP ≥ {k_flip} of {N} | random sign) = {p_one:.4f}")

sub_T = out[(out["stable"] == "USDT") & out["sdf_proxy"].isin(RETURN_BASED)]
sub_C = out[(out["stable"] == "USDC") & out["sdf_proxy"].isin(RETURN_BASED)]
k1 = int((sub_T["sign_match"] == "MATCH").sum())
k2 = int((sub_C["sign_match"] == "FLIP").sum())
joint_p = float((1 - binom.cdf(k1 - 1, len(sub_T), 0.5)) *
                (1 - binom.cdf(k2 - 1, len(sub_C), 0.5)))
print(f"  Joint (USDT MATCH ≥ {k1} × USDC FLIP ≥ {k2}): p = {joint_p:.5f}")

pd.DataFrame(family_records).to_parquet(T / "theorem4_multi_sdf_v2_summary.parquet", index=False)
