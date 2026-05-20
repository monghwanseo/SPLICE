import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from settings import START_DT, END_DT, RAW, PROCESSED, RESULTS, NW_LAG, ANNUALIZATION_8H
from econometrics import log, hr, ols_nw, baron_kenny

EXTRA = ["SOL", "XRP", "BNB", "DOGE", "ADA", "LTC"]

panel = pd.read_parquet(PROCESSED / "panel.parquet")
OUT = RESULTS / "tables"

hr()
log(f"E10: Cross-asset extension {EXTRA}")
hr()

grid = panel.index

def seeded_ffill(s, grid):
    full_idx = s.index.union(grid)
    return s.reindex(full_idx).ffill().reindex(grid)

def load_kline_close(side, sym):
    p = RAW / "binance" / f"{sym}_{side}_1h.parquet"
    df = pd.read_parquet(p)
    df["ts"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
    df.index = df["ts"].dt.floor("1h")
    return df[~df.index.duplicated(keep="last")]["close"].sort_index()

def load_funding(venue, sym):
    p = RAW / venue / f"{sym}_funding_8h.parquet"
    df = pd.read_parquet(p)
    if df.empty:
        return pd.Series(dtype=float)
    df["ts"] = pd.to_datetime(df["funding_ts_ms"], unit="ms", utc=True)
    df.index = df["ts"].dt.floor("1h")
    return df[~df.index.duplicated(keep="last")]["fundingRate"].sort_index()

ext = pd.DataFrame(index=grid)
for asset in EXTRA:
    sym = f"{asset}USDT"
    spot = seeded_ffill(load_kline_close("spot", sym), grid)
    perp = seeded_ffill(load_kline_close("perp", sym), grid)
    ext[f"basis_{asset}"] = (spot - perp) / spot
    for venue in ["binance", "bybit"]:
        fr = seeded_ffill(load_funding(venue, sym), grid)
        ext[f"eta_{asset}_{venue}_ann"] = fr * ANNUALIZATION_8H

ext["delta_USDT"] = panel["delta_USDT"]
ext["delta_USDC"] = panel["delta_USDC"]

ext.to_parquet(PROCESSED / "panel_extra.parquet", compression="snappy")

records = []
for asset in EXTRA:
    b_col = f"basis_{asset}"
    if b_col not in ext.columns:
        continue
    df = ext[[b_col, "delta_USDT", "delta_USDC"]].dropna()
    if len(df) < 1000:
        continue
    res = ols_nw(df[b_col], df[["delta_USDT", "delta_USDC"]], lag=NW_LAG)
    g_t = float(res.params["delta_USDT"])
    g_c = float(res.params["delta_USDC"])
    log(f"\n  {asset}:  gamma_USDT={g_t:+.4f} (t={res.tvalues['delta_USDT']:+.2f})  "
        f"gamma_USDC={g_c:+.4f} (t={res.tvalues['delta_USDC']:+.2f})  R2={res.rsquared:.4f}")
    rec = {"asset": asset, "n": int(res.nobs),
           "gamma_USDT": g_t, "gamma_USDC": g_c,
           "t_USDT": float(res.tvalues["delta_USDT"]),
           "t_USDC": float(res.tvalues["delta_USDC"]),
           "r2": float(res.rsquared)}

    for venue in ["binance", "bybit"]:
        e_col = f"eta_{asset}_{venue}_ann"
        df2 = ext[[e_col, "delta_USDT"]].dropna()
        if len(df2) < 1000:
            continue
        res_r = ols_nw(df2[e_col], df2[["delta_USDT"]], lag=NW_LAG)
        c = float(res_r.params["delta_USDT"])
        log(f"    {venue}: c (USDT) = {c:+.3f}  t={res_r.tvalues['delta_USDT']:+.2f}  R2={res_r.rsquared:.4f}")
        rec[f"c_USDT_{venue}"] = c

        res_l = ols_nw(df2[e_col], ext.loc[df2.index, [b_col]].rename(columns={b_col: "basis"}), lag=NW_LAG)
        lam = -float(res_l.params["basis"])
        log(f"    {venue}: lambda = {lam:+.2f}")
        rec[f"lambda_{venue}"] = lam

    records.append(rec)

pd.DataFrame(records).to_parquet(OUT / "e10_cross_asset.parquet", index=False)
log("\nE10 done.")
