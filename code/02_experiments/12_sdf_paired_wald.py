import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import combine_pvalues, norm

sys.path.insert(0, str(Path(__file__).parent.parent))
from settings import PROCESSED, RESULTS, NW_LAG

T = RESULTS / "tables"
panel = pd.read_parquet(PROCESSED / "panel.parquet")
extra = pd.read_parquet(PROCESSED / "panel_extra.parquet")

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

all_logret = pd.DataFrame({"BTC": log_BTC, "ETH": log_ETH, **extras_logret}).dropna()
returns_demeaned = all_logret - all_logret.mean()
cov = returns_demeaned.cov()
eigvals, eigvecs = np.linalg.eigh(cov.values)
PC1 = (returns_demeaned * eigvecs[:, -1]).sum(axis=1)

SDFs = {
    "S1_LogUtil_BTCETH": -0.5 * (log_BTC + log_ETH),
    "S3_BTC_only":       -log_BTC,
    "S4_RiskReversal":   -log_BTC * np.abs(log_BTC),
    "S9_ETH_only":       -log_ETH,
    "S10_PC1":           -PC1.reindex(panel.index),
}

print("=" * 80)
print("SDF paired test: H0: β_USDT − β_USDC = 0  (Wald test, HAC SE)")
print("=" * 80)

records = []
for sdf_name, M in SDFs.items():
    df = pd.concat([M.rename("M"),
                    panel["delta_USDT"].rename("d_T"),
                    panel["delta_USDC"].rename("d_C")], axis=1).dropna()
    X = sm.add_constant(df[["d_T", "d_C"]])
    res = sm.OLS(df["M"], X).fit(cov_type="HAC", cov_kwds={"maxlags": NW_LAG})

    b_T = float(res.params["d_T"])
    b_C = float(res.params["d_C"])
    diff = b_T - b_C

    R = np.array([[0.0, 1.0, -1.0]])
    cov_b = res.cov_params().values
    se_diff = float(np.sqrt(R @ cov_b @ R.T)[0, 0])
    t_diff = diff / se_diff
    p_diff = 2 * (1 - norm.cdf(abs(t_diff)))

    se_T = float(res.bse["d_T"])
    se_C = float(res.bse["d_C"])

    records.append({
        "sdf_proxy": sdf_name,
        "beta_USDT": b_T, "se_USDT": se_T, "t_USDT": b_T / se_T,
        "beta_USDC": b_C, "se_USDC": se_C, "t_USDC": b_C / se_C,
        "diff": diff, "se_diff": se_diff, "t_diff": t_diff, "p_diff": p_diff,
        "n": int(res.nobs),
    })
    print(f"  {sdf_name:22s}:  β_USDT={b_T:+.5f} (t={b_T/se_T:+.2f})  "
          f"β_USDC={b_C:+.5f} (t={b_C/se_C:+.2f})  "
          f"Δ={diff:+.5f} (t={t_diff:+.2f}, p={p_diff:.4f})")

df = pd.DataFrame(records)
df.to_parquet(T / "sdf_paired_test.parquet", index=False)

print()
print("=" * 80)
print("Joint test across 5 return-based SDFs   (H1: Δ = β_USDT − β_USDC < 0)")
print("=" * 80)
p_one_sided = []
for r in records:
    p1 = norm.cdf(r["t_diff"])
    p_one_sided.append(p1)
    print(f"  {r['sdf_proxy']:22s}: one-sided p(Δ<0) = {p1:.4f}")

chi2, p_fisher = combine_pvalues(p_one_sided, method="fisher")
print(f"\n  Fisher combined χ²(10) = {chi2:.2f}, p = {p_fisher:.4e}")

zs = [r["t_diff"] for r in records]
z_stouffer = sum(zs) / np.sqrt(len(zs))
p_stouffer = norm.cdf(z_stouffer)
print(f"  Stouffer  Z = {z_stouffer:.3f}, one-sided p(Δ<0) = {p_stouffer:.4e}")

n_neg = sum(1 for r in records if r["diff"] < 0)
n_total = len(records)
from scipy.stats import binom
p_sign = 1 - binom.cdf(n_neg - 1, n_total, 0.5)
print(f"  Sign test: Δ < 0 in {n_neg}/{n_total} SDFs, exact binomial p = {p_sign:.4f}")

diffs = np.array([r["diff"] for r in records])
print(f"  mean(Δ) = {diffs.mean():+.5f}  SD(Δ) = {diffs.std(ddof=1):.5f}")

summary = pd.DataFrame([{
    "n_sdfs": n_total,
    "n_diff_negative": n_neg,
    "binomial_p_one_sided_Δ_lt_0": p_sign,
    "fisher_chi2": chi2,
    "fisher_p_Δ_lt_0": p_fisher,
    "stouffer_z": z_stouffer,
    "stouffer_p_Δ_lt_0": p_stouffer,
    "mean_diff": diffs.mean(),
    "sd_diff": diffs.std(ddof=1),
}])
summary.to_parquet(T / "sdf_paired_test_summary.parquet", index=False)

print(f"\n→ Saved {T / 'sdf_paired_test.parquet'}")
print(f"→ Saved {T / 'sdf_paired_test_summary.parquet'}")
