import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from settings import RAW, RESULTS, ANNUALIZATION_8H, NW_LAG
from econometrics import ols_nw

T = RESULTS / "tables"
PROCESSED = RESULTS.parent / "data_processed"

print("=" * 90)
print("κ decomposition (Edge-1: addressing L2 — quantitative gap closing)")
print("=" * 90)

panel = pd.read_parquet(PROCESSED / "panel.parquet")
print(f"Panel: {panel.shape}; sample n = {len(panel)}")

g2 = pd.read_parquet(T / "g_fix2_FINAL.parquet").set_index("stable")
print("\nReference: G-Fix2 implied κ vs lending κ:")
print(g2[["kappa_implied_per_event", "kappa_emp_per_event", "ratio_emp_over_implied"]])

def apy_pct_to_per_event(apy_pct):
    return apy_pct / 100.0 / ANNUALIZATION_8H

print("\n[C1] Lending opportunity cost (κ_lend)")
kappa_C1 = {
    "USDT": float(g2.loc["USDT", "kappa_emp_per_event"]),
    "USDC": float(g2.loc["USDC", "kappa_emp_per_event"]),
}
for s, v in kappa_C1.items():
    print(f"  κ_lend ({s}) = {v:.6f} per event")

print("\n[C2] Cross-venue funding differential (κ_xvenue)")
kappa_C2 = {}
eta_diff_BTC = (panel["eta_BTC_binance"] - panel["eta_BTC_bybit"]).abs()
for stable in ["USDT", "USDC"]:
    delta_col = f"delta_{stable}"
    if delta_col not in panel.columns:
        kappa_C2[stable] = np.nan
        continue
    df_reg = pd.concat([eta_diff_BTC.rename("disp"),
                        panel[delta_col].rename("delta")], axis=1).dropna()
    if len(df_reg) < 100:
        kappa_C2[stable] = np.nan
        continue
    res = ols_nw(df_reg["disp"], df_reg[["delta"]], lag=NW_LAG)
    coef = float(res.params["delta"])
    t_stat = float(res.tvalues["delta"])
    kappa_C2[stable] = abs(coef)
    print(f"  κ_xvenue ({stable}) = {kappa_C2[stable]:.6f} per event "
          f"(slope of |η_bnc − η_byb| on δ, t = {t_stat:+.2f}, n = {len(df_reg)})")

print("\n[C3] Tail-risk premium (κ_tail) — from E19 risk-premium decomposition")
e19 = pd.read_parquet(T / "e19_risk_premium.parquet")
beta_sigma2 = e19[(e19["asset"] == "BTC") & (e19["venue"] == "binance")].iloc[0]["beta_sigma2"]
print(f"  E19 β_σ² (BTC binance) = {beta_sigma2:.4f}")

e2 = pd.read_parquet(T / "e2_second_stage.parquet")
lam_BTC_bnc = float(e2[(e2["asset"] == "BTC") & (e2["venue"] == "binance")].iloc[0]["lambda"])
print(f"  λ̂ (BTC binance) = {lam_BTC_bnc:.1f}")

kappa_C3 = {}
for stable in ["USDT", "USDC"]:
    delta = panel[f"delta_{stable}"].dropna()
    sd = float(delta.std())
    stress_mask = delta.abs() > sd
    var_stress = float(delta.loc[stress_mask].var()) if stress_mask.sum() > 100 else float(delta.var())
    kappa_C3[stable] = abs(beta_sigma2) * var_stress * lam_BTC_bnc
    print(f"  κ_tail ({stable}) = {kappa_C3[stable]:.6f} per event "
          f"(σ²(δ|stress) = {var_stress:.4e}, n_stress = {int(stress_mask.sum())})")

print("\n[C4] Money-market spread (κ_mm) — Aave APY vs US 3M T-bill")
aave_files = {
    "USDT": RAW / "aave" / "aave_v3_USDT_lendapy_daily.parquet",
    "USDC": RAW / "aave" / "aave_v3_USDC_lendapy_daily.parquet",
}
fred_path = RAW / "fred" / "DGS3MO_daily.parquet"
if fred_path.exists():
    fred = pd.read_parquet(fred_path)
    fred["date"] = pd.to_datetime(fred["date"], utc=True).dt.normalize()
    rf_daily = fred.set_index("date")["rate_pct"]
    print(f"  FRED 3M T-bill loaded: n={len(rf_daily)}, mean={rf_daily.mean():.2f}%")
else:
    rf_daily = None
    print("  ⚠ FRED data missing — using constant proxy r_f = 4.0% APY (sample mean)")

kappa_C4 = {}
for stable in ["USDT", "USDC"]:
    if not aave_files[stable].exists():
        kappa_C4[stable] = np.nan
        continue
    aave = pd.read_parquet(aave_files[stable])
    aave["date"] = pd.to_datetime(aave["ts"], utc=True).dt.normalize()
    aave_d = aave.groupby("date")["lend_apy_pct"].last()

    if rf_daily is not None:
        rf_aligned = rf_daily.reindex(aave_d.index, method="ffill")
        spread = (aave_d - rf_aligned).dropna()
    else:
        spread = aave_d - 4.0

    panel_idx = pd.to_datetime(panel.index, utc=True).normalize()
    delta_d = pd.Series(panel[f"delta_{stable}"].values, index=panel_idx).groupby(level=0).mean()
    delta_d = delta_d.reindex(spread.index, method="ffill")

    df_reg = pd.concat([spread.rename("spread"), delta_d.rename("delta")], axis=1).dropna()
    if len(df_reg) < 100:
        kappa_C4[stable] = np.nan
        continue
    res = ols_nw(df_reg["spread"], df_reg[["delta"]], lag=NW_LAG)
    slope_apy = float(res.params["delta"])
    t_stat = float(res.tvalues["delta"])
    kappa_C4[stable] = apy_pct_to_per_event(abs(slope_apy))
    print(f"  κ_mm ({stable}) = {kappa_C4[stable]:.6f} per event "
          f"(spread slope on δ = {slope_apy:+.2f}% APY/unit, t = {t_stat:+.2f})")

print("\n" + "=" * 90)
print("κ DECOMPOSITION SUMMARY")
print("=" * 90)
rows = []
for stable in ["USDT", "USDC"]:
    implied = float(g2.loc[stable, "kappa_implied_per_event"])
    c1 = kappa_C1.get(stable, np.nan)
    c2 = kappa_C2.get(stable, np.nan)
    c3 = kappa_C3.get(stable, np.nan)
    c4 = kappa_C4.get(stable, np.nan)
    components = [c1, c2, c3, c4]
    total = float(np.nansum(components))
    pct = lambda v: f"{v/implied*100:6.2f}%" if not np.isnan(v) and implied > 0 else "    n/a"
    rows.append({
        "stable": stable,
        "implied_kappa": implied,
        "C1_lending":  c1,
        "C2_xvenue":   c2,
        "C3_tail":     c3,
        "C4_mm":       c4,
        "total_C1234": total,
        "pct_C1": c1 / implied if implied > 0 else np.nan,
        "pct_C2": c2 / implied if implied > 0 else np.nan,
        "pct_C3": c3 / implied if implied > 0 else np.nan,
        "pct_C4": c4 / implied if implied > 0 else np.nan,
        "pct_total": total / implied if implied > 0 else np.nan,
        "L2_old_pct": c1 / implied if implied > 0 else np.nan,
    })
    cons_total = float(np.nansum([c1, c2, c4]))
    print(f"\n[{stable}] κ_implied = {implied:.6f} per event")
    print(f"    C1 lending     = {c1:.6f}  ({pct(c1)})")
    print(f"    C2 xvenue      = {c2:.6f}  ({pct(c2)})")
    print(f"    C3 tail        = {c3:.6f}  ({pct(c3)})  [model-dependent; see caveat]")
    print(f"    C4 mm spread   = {c4:.6f}  ({pct(c4)})")
    print(f"    --- ")
    print(f"    Conservative total (C1+C2+C4) = {cons_total:.6f}  ({pct(cons_total)})")
    print(f"    Full total       (C1+C2+C3+C4) = {total:.6f}  ({pct(total)})")
    print(f"    (Old L2: lending only = {pct(c1)})")
    rows[-1]["conservative_total"] = cons_total
    rows[-1]["pct_conservative"] = cons_total / implied if implied > 0 else np.nan

out = pd.DataFrame(rows)
out.to_parquet(T / "kappa_decomposition.parquet", index=False)
print(f"\n→ Saved {T / 'kappa_decomposition.parquet'}")

print("""
┌──────────────────────────────────────────────────────────────────────────┐
│ INTERPRETATION                                                           │
├──────────────────────────────────────────────────────────────────────────┤
│ L2's old framing: "lending alone explains only 2-4% of κ_implied"        │
│ This decomposition adds C2 (cross-venue), C3 (tail risk), C4 (mm spread).│
│                                                                          │
│ Even when individual components have measurement error, summing them     │
│ tightens the structural identification of κ. The total fraction          │
│ recovered is the empirical analog of                                     │
│   κ = κ_lend + κ_xvenue + κ_tail + κ_mm                                  │
│ — the four-channel decomposition discussed in §3.5 (C1)–(C4).            │
│                                                                          │
│ Caveat: C3 calibration uses E19 β_σ² which is not statistically          │
│ significant at conventional levels (t≈1.3 for BTC binance). Conservative │
│ readers should drop C3 and report C1+C2+C4 only.                         │
└──────────────────────────────────────────────────────────────────────────┘
""")
