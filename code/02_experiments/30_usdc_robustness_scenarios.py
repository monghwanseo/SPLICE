import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import binom

sys.path.insert(0, str(Path(__file__).parent.parent))
from settings import PROCESSED, RESULTS, RAW, NW_LAG

T = RESULTS / "tables"
panel = pd.read_parquet(PROCESSED / "panel.parquet")

log_BTC = np.log(panel["spot_BTC"]).diff()
log_ETH = np.log(panel["spot_ETH"]).diff()

extras_logret = {}
for asset in ["BNB", "SOL", "XRP", "DOGE", "ADA", "LTC"]:
    p = RAW / "binance" / f"{asset}USDT_spot_1h.parquet"
    if p.exists():
        df = pd.read_parquet(p)
        df["ts"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
        df = df.sort_values("ts").set_index("ts")
        extras_logret[asset] = np.log(df["close"]).diff().reindex(panel.index)

all_logret = pd.DataFrame({"BTC": log_BTC, "ETH": log_ETH, **extras_logret}).dropna()
returns_dm = all_logret - all_logret.mean()
_eigvals, _eigvecs = np.linalg.eigh(returns_dm.cov().values)
pc1_loadings = _eigvecs[:, -1]
PC1 = (returns_dm * pc1_loadings).sum(axis=1).reindex(panel.index)

PROXY_BUILDERS = {
    "S1_EqWt_BTCETH":  lambda: -0.5 * (log_BTC + log_ETH),
    "S3_BTC_only":     lambda: -log_BTC,
    "S9_ETH_only":     lambda: -log_ETH,
    "S4_RiskRev":      lambda: -log_BTC * np.abs(log_BTC),
    "S10_PC1":         lambda: -PC1,
}
PROXY_LABEL = {
    "S1_EqWt_BTCETH":  "Equal-weighted BTC and ETH",
    "S3_BTC_only":     "BTC-only log-return",
    "S9_ETH_only":     "ETH-only log-return",
    "S4_RiskRev":      "Put-call risk reversal",
    "S10_PC1":         "First principal component",
}


def beta_M(M, delta, mask=None):
    df = pd.DataFrame({"M": M, "d": delta})
    if mask is not None:
        df = df[mask]
    df = df.dropna()
    if df["d"].var() < 1e-12 or len(df) < 100:
        return np.nan, len(df)
    res = sm.OLS(df["M"], sm.add_constant(df["d"])).fit(
        cov_type="HAC", cov_kwds={"maxlags": NW_LAG})
    return -float(res.params.iloc[1]), len(df)


_df_full_T = pd.DataFrame({
    "basis_BTC": panel["basis_BTC"],
    "d": panel["delta_USDT"],
}).dropna()
_g_T = sm.OLS(_df_full_T["basis_BTC"], sm.add_constant(_df_full_T[["d"]])).fit(
    cov_type="HAC", cov_kwds={"maxlags": NW_LAG})
GAMMA_USDT_FULL = float(_g_T.params["d"])
SIGN_REF_USDT = np.sign(GAMMA_USDT_FULL)

_df_full_C = pd.DataFrame({
    "basis_BTC": panel["basis_BTC"],
    "d": panel["delta_USDC"],
}).dropna()
_g_C = sm.OLS(_df_full_C["basis_BTC"], sm.add_constant(_df_full_C[["d"]])).fit(
    cov_type="HAC", cov_kwds={"maxlags": NW_LAG})
GAMMA_USDC_FULL = float(_g_C.params["d"])
SIGN_REF_USDC = np.sign(GAMMA_USDC_FULL)

idx = panel.index

USDC_STRESS_DATES = [
    "2023-03-10",
    "2023-03-11",
    "2023-03-12",
    "2023-03-13",
    "2022-05-12",
    "2023-08-01",
]


def build_mask(scenario):
    if scenario == "full_sample":
        return pd.Series(True, index=idx)
    if scenario == "drop_svb_window":
        lo = pd.Timestamp("2023-03-07", tz="UTC")
        hi = pd.Timestamp("2023-03-16", tz="UTC")
        return ~((idx >= lo) & (idx < hi))
    if scenario == "drop_usdc_events":
        mask = pd.Series(True, index=idx)
        lo = pd.Timestamp("2023-03-07", tz="UTC")
        hi = pd.Timestamp("2023-03-16", tz="UTC")
        mask &= ~((idx >= lo) & (idx < hi))
        for d in USDC_STRESS_DATES:
            t = pd.Timestamp(d, tz="UTC")
            mask &= ~((idx >= t - pd.Timedelta(hours=24)) &
                      (idx <= t + pd.Timedelta(hours=24)))
        return mask
    if scenario == "pre_svb_subsample":
        cutoff = pd.Timestamp("2023-03-01", tz="UTC")
        return pd.Series(idx < cutoff, index=idx)
    raise ValueError(scenario)


SCENARIOS = ["full_sample", "drop_svb_window", "drop_usdc_events", "pre_svb_subsample"]
delta_USDT = panel["delta_USDT"]
delta_USDC = panel["delta_USDC"]

print(f"Sign reference (FULL-sample): gamma_USDT = {GAMMA_USDT_FULL:+.5f}, gamma_USDC = {GAMMA_USDC_FULL:+.5f}")
print()

rows = []
for scenario in SCENARIOS:
    mask = build_mask(scenario)
    n_obs = int(mask.sum())
    print(f"--- {scenario}: n={n_obs} ---")
    for proxy_name, builder in PROXY_BUILDERS.items():
        M = builder()
        bT, nT = beta_M(M, delta_USDT, mask=mask)
        bC, nC = beta_M(M, delta_USDC, mask=mask)
        match_T = "MATCH" if not np.isnan(bT) and np.sign(bT) == SIGN_REF_USDT else "FLIP"
        match_C = "MATCH" if not np.isnan(bC) and np.sign(bC) == SIGN_REF_USDC else "FLIP"
        rows.append({
            "scenario": scenario,
            "sdf_proxy": proxy_name,
            "label": PROXY_LABEL[proxy_name],
            "beta_M_USDT": bT,
            "beta_M_USDC": bC,
            "match_USDT": match_T,
            "match_USDC": match_C,
        })

detail = pd.DataFrame(rows)

summary_rows = []
for scenario in SCENARIOS:
    sub = detail[detail["scenario"] == scenario]
    k_T = int((sub["match_USDT"] == "MATCH").sum())
    k_C_flip = int((sub["match_USDC"] == "FLIP").sum())
    n = len(sub)
    joint_p = float((1 - binom.cdf(k_T - 1, n, 0.5)) *
                    (1 - binom.cdf(k_C_flip - 1, n, 0.5)))
    summary_rows.append({
        "scenario": scenario,
        "usdt_matches": k_T,
        "usdc_flips": k_C_flip,
        "n_proxies": n,
        "joint_sign_rand_p": joint_p,
    })

summary = pd.DataFrame(summary_rows)
print()
print(summary.to_string(index=False))

detail.to_parquet(T / "usdc_robustness_scenarios.parquet", index=False)
summary.to_parquet(T / "usdc_robustness_summary.parquet", index=False)
print(f"\nSaved {T / 'usdc_robustness_scenarios.parquet'}")
print(f"Saved {T / 'usdc_robustness_summary.parquet'}")
