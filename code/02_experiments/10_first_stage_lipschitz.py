import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from settings import PROCESSED, RESULTS, EVENTS

T = RESULTS / "tables"
panel = pd.read_parquet(PROCESSED / "panel.parquet")
extra = pd.read_parquet(PROCESSED / "panel_extra.parquet")

kappa = pd.read_parquet(T / "kappa_per_asset.parquet").set_index("asset")
iv_evt = pd.read_parquet(T / "iv_event_individual.parquet")
delta = 8.0

print("=" * 80)
print("MS_E3: Lipschitz stability bound  ↔  IV first-stage F")
print("=" * 80)

def L_per_asset(asset):
    if asset not in kappa.index:
        return np.nan
    k = float(kappa.loc[asset, "kappa_per_hour"])
    r = float(kappa.loc[asset, "rho_per_hour"])
    l = float(kappa.loc[asset, "lambda_per_hour"])
    if r - l <= 0:
        return np.nan
    return k * delta / (r - l)

basis = {}
for a in kappa.index:
    src = panel if a in ("BTC", "ETH") else extra
    if f"basis_{a}" in src.columns:
        basis[a] = src[f"basis_{a}"]

records = []
for ev_name, ev_iso, ev_desc in EVENTS:
    ev_ts = pd.Timestamp(ev_iso, tz="UTC")
    pre = (panel.index >= ev_ts - pd.Timedelta(hours=24)) & (panel.index < ev_ts)
    post = (panel.index >= ev_ts) & (panel.index <= ev_ts + pd.Timedelta(hours=24))

    dT_pre = panel.loc[pre, "delta_USDT"].mean()
    dT_post = panel.loc[post, "delta_USDT"].mean()
    dC_pre = panel.loc[pre, "delta_USDC"].mean()
    dC_post = panel.loc[post, "delta_USDC"].mean()
    delta_dT = float(dT_post - dT_pre)
    delta_dC = float(dC_post - dC_pre)

    for a in basis.keys():
        Xa = basis[a]
        X_pre = Xa[pre].mean() if pre.any() else np.nan
        X_post = Xa[post].mean() if post.any() else np.nan
        delta_X = float(X_post - X_pre)

        L = L_per_asset(a)
        ratio_T = delta_X / delta_dT if abs(delta_dT) > 1e-7 else np.nan
        coverage_T = ratio_T / L if (L and L != 0 and np.isfinite(ratio_T)) else np.nan

        match = iv_evt[(iv_evt["event"] == ev_name) & (iv_evt["asset"] == a) & (iv_evt["stable"] == "USDT")]
        fs_F = float(match["first_stage_F"].iloc[0]) if len(match) else np.nan
        gamma_IV = float(match["gamma_IV"].iloc[0]) if len(match) else np.nan

        records.append({
            "event": ev_name,
            "event_date": ev_iso,
            "asset": a,
            "delta_delta_USDT_bp": delta_dT * 1e4,
            "delta_delta_USDC_bp": delta_dC * 1e4,
            "delta_X_bp": delta_X * 1e4,
            "L_theoretical": L,
            "ratio_observed_USDT": ratio_T,
            "coverage_pct": coverage_T * 100 if np.isfinite(coverage_T) else np.nan,
            "first_stage_F": fs_F,
            "gamma_IV": gamma_IV,
        })

out = pd.DataFrame(records)
out.to_csv(T / "ms_e3_lipschitz_iv.csv", index=False)
print(f"\n→ ms_e3_lipschitz_iv.csv ({len(out)} rows)")

print("\n" + "=" * 80)
print("PER-EVENT SUMMARY  (mean over 8 assets)")
print("=" * 80)
ev_summary = out.groupby("event").agg(
    delta_delta_USDT_bp=("delta_delta_USDT_bp", "first"),
    delta_X_bp=("delta_X_bp", "mean"),
    L_theoretical_avg=("L_theoretical", "mean"),
    ratio_avg=("ratio_observed_USDT", "mean"),
    coverage_pct_avg=("coverage_pct", "mean"),
    first_stage_F=("first_stage_F", "mean"),
).reset_index()
print(ev_summary.to_string(index=False))

ev_summary.to_csv(T / "ms_e3_lipschitz_iv_summary.csv", index=False)
print(f"→ ms_e3_lipschitz_iv_summary.csv")

print("\n" + "=" * 80)
print("Theory-empirics link")
print("=" * 80)
sub = out.dropna(subset=["delta_delta_USDT_bp", "first_stage_F"])
if len(sub) >= 5:
    abs_dd = sub["delta_delta_USDT_bp"].abs()
    f_vals = sub["first_stage_F"]
    rho_p = float(np.corrcoef(abs_dd, f_vals)[0, 1])
    rho_s = float(pd.Series(abs_dd).rank().corr(pd.Series(f_vals).rank()))
    print(f"  corr(|Δδ_event|, first_stage_F) = {rho_p:+.3f}  (Pearson, n={len(sub)})")
    print(f"  corr(rank|Δδ|, rank fs_F)       = {rho_s:+.3f}  (Spearman)")

print("\nCoverage (observed |ΔX/Δδ| / theoretical L) — should be ≤ 1 if bound is tight:")
cov_summary = out.dropna(subset=["coverage_pct"]).groupby("event")["coverage_pct"].agg(["mean", "min", "max"])
print(cov_summary.to_string())
