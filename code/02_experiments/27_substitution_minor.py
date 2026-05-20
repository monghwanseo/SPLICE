import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from settings import PROCESSED, RESULTS, RAW, NW_LAG
from econometrics import ols_nw

def seeded_ffill(s, grid):
    full_idx = s.index.union(grid)
    return s.reindex(full_idx).ffill().reindex(grid)

T = RESULTS / "tables"
panel = pd.read_parquet(PROCESSED / "panel.parquet")
grid = panel.index

print("=" * 80)
print("(I) Theorem 5 minor-stable regime conditioning")
print("=" * 80)

def load_close(sym):
    p = RAW / "binance" / f"{sym}_spot_1h.parquet"
    if not p.exists():
        return pd.Series(dtype=float)
    df = pd.read_parquet(p)
    if df.empty:
        return pd.Series(dtype=float)
    df["ts"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
    df.index = df["ts"].dt.floor("1h")
    return df[~df.index.duplicated(keep="last")]["close"].sort_index()

tusd_usdt = load_close("TUSDUSDT")
fdusd_usdt = load_close("FDUSDUSDT")
panel["delta_TUSD_rel"] = (seeded_ffill(tusd_usdt, grid) - 1.0)
panel["delta_FDUSD_rel"] = (seeded_ffill(fdusd_usdt, grid) - 1.0)

dai_safe = pd.read_parquet(PROCESSED / "delta_DAI_safe.parquet").set_index("ts")["delta_DAI_safe"]
panel["delta_DAI"] = dai_safe.reindex(grid)

if "regime" not in panel.columns:
    print("  WARN: regime column missing — cannot do regime conditioning")
    sys.exit(1)

windows = [
    ("TUSD",  "delta_TUSD_rel",  "2020-11-01"),
    ("FDUSD", "delta_FDUSD_rel", "2023-08-01"),
    ("DAI",   "delta_DAI",       "2022-01-01"),
]

records = []
for label, d_col, start in windows:
    sub = panel[panel.index >= pd.Timestamp(start, tz="UTC")]
    print(f"\n  {label} (start {start}, n={sub[d_col].notna().sum()})")

    for regime in ["all", "normal", "stress"]:
        if regime == "all":
            df = sub
        else:
            df = sub[sub["regime"] == regime]
        df_clean = df[["basis_BTC", "delta_USDT", d_col]].dropna()
        if len(df_clean) < 200:
            continue

        try:
            res = ols_nw(df_clean["basis_BTC"], df_clean[["delta_USDT", d_col]], lag=NW_LAG)
        except Exception as e:
            print(f"    {regime}: failed — {e}")
            continue
        g_own = float(res.params["delta_USDT"])
        t_own = float(res.tvalues["delta_USDT"])
        g_cross = float(res.params[d_col])
        t_cross = float(res.tvalues[d_col])

        records.append({
            "minor_stable": label,
            "regime": regime,
            "n": len(df_clean),
            "gamma_own_USDT": g_own,
            "t_own": t_own,
            "gamma_cross_minor": g_cross,
            "t_cross_minor": t_cross,
        })
        print(f"    {regime:>6}  n={len(df_clean):>6}  γ_USDT={g_own:+.4f} (t={t_own:+.2f})  "
              f"γ_{label}={g_cross:+.4f} (t={t_cross:+.2f})")

out = pd.DataFrame(records)
out.to_parquet(T / "theorem5_regime_minor.parquet", index=False)
print(f"\n→ Saved {T / 'theorem5_regime_minor.parquet'}")

print("\n" + "=" * 80)
print("Δ(stress − normal) cross-coefficient on minor stable")
print("=" * 80)
for label, _, _ in windows:
    sub = out[out["minor_stable"] == label]
    g_n = sub[sub["regime"] == "normal"]["gamma_cross_minor"]
    g_s = sub[sub["regime"] == "stress"]["gamma_cross_minor"]
    if len(g_n) and len(g_s):
        gn = float(g_n.iloc[0])
        gs = float(g_s.iloc[0])
        print(f"  {label}: γ_normal={gn:+.4f}  γ_stress={gs:+.4f}  Δ={gs-gn:+.4f}")
