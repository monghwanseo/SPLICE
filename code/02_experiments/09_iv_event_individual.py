import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).parent.parent))
from settings import PROCESSED, RESULTS, EVENTS

T = RESULTS / "tables"
panel = pd.read_parquet(PROCESSED / "panel.parquet")

def event_indicator(ev_t, panel_idx, window_h=72):
    mask = (panel_idx >= ev_t - pd.Timedelta(hours=window_h)) & \
           (panel_idx <= ev_t + pd.Timedelta(hours=window_h))
    return pd.Series(mask.astype(float), index=panel_idx)

def two_sls_just_id(y, x, z):
    Z = np.column_stack([np.ones_like(z), z])
    fs = sm.OLS(x, Z).fit()
    fs_t = float(fs.tvalues[1])
    F = fs_t ** 2
    x_hat = np.asarray(fs.fittedvalues)
    Xss = np.column_stack([np.ones_like(y), x_hat])
    beta = np.linalg.solve(Xss.T @ Xss, Xss.T @ y)
    return float(beta[1]), F

print("=" * 80)
print("Individual-event IV (just-id per event)")
print("=" * 80)

records = []
for stable in ["USDT", "USDC"]:
    for asset in ["BTC", "ETH"]:
        b_col = f"basis_{asset}"
        d_col = f"delta_{stable}"
        df_panel = panel[[b_col, d_col]].dropna()
        for ev_name, ev_str, _ in EVENTS:
            ev_t = pd.Timestamp(ev_str, tz="UTC")
            if ev_t < df_panel.index.min() or ev_t > df_panel.index.max():
                continue
            z = event_indicator(ev_t, df_panel.index, 72)
            df = pd.concat([
                df_panel[b_col].rename("X"),
                df_panel[d_col].rename("d"),
                z.rename("Z"),
            ], axis=1).dropna()
            if df["Z"].sum() < 10:
                continue
            y = df["X"].values
            x = df["d"].values
            zv = df["Z"].values
            try:
                gamma_iv, F = two_sls_just_id(y, x, zv)
            except Exception as e:
                print(f"  [debug] {stable} {asset} {ev_name}: {type(e).__name__}: {e}")
                gamma_iv, F = np.nan, np.nan
            records.append({
                "stable": stable, "asset": asset,
                "event": ev_name, "event_date": ev_str,
                "n": len(df), "n_treated": int(df["Z"].sum()),
                "gamma_IV": gamma_iv, "first_stage_F": F,
            })

df = pd.DataFrame(records)
df.to_parquet(T / "iv_event_individual.parquet", index=False)

print(f"\n{'stable':>5}  {'asset':>4}  {'event':>20}  {'n_treated':>9}  {'γ̂_IV':>9}  {'F':>8}")
print("  " + "-" * 65)
for stable in ["USDT", "USDC"]:
    for asset in ["BTC", "ETH"]:
        sub = df[(df["stable"] == stable) & (df["asset"] == asset)]
        for _, r in sub.iterrows():
            sig = "✓" if r["first_stage_F"] >= 10 else "weak"
            print(f"  {r['stable']:>5}  {r['asset']:>4}  {r['event']:>20}  "
                  f"{r['n_treated']:>9}  {r['gamma_IV']:>+9.4f}  {r['first_stage_F']:>8.1f}  {sig}")
        sub_strong = sub[sub["first_stage_F"] >= 10]
        if len(sub_strong) >= 2:
            print(f"  → {stable} {asset}: strong-IV γ̂ range [{sub_strong['gamma_IV'].min():+.4f}, "
                  f"{sub_strong['gamma_IV'].max():+.4f}], median {sub_strong['gamma_IV'].median():+.4f}, "
                  f"strong-IV count {len(sub_strong)}/5")
        print()

print("=" * 80)
print("Per-event identifying power (across all 4 cells)")
print("=" * 80)
for ev_name, ev_str, _ in EVENTS:
    sub = df[df["event"] == ev_name]
    if len(sub) == 0:
        continue
    sig_count = (sub["first_stage_F"] >= 10).sum()
    print(f"  {ev_name:>20}  ({ev_str}):  strong-IV in {sig_count}/{len(sub)} cells, "
          f"median γ̂ = {sub['gamma_IV'].median():+.4f}, "
          f"median F = {sub['first_stage_F'].median():.1f}")

print(f"\n→ Saved {T / 'iv_event_individual.parquet'}")
