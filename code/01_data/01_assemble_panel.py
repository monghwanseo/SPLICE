import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from settings import (START_DT, END_DT, RAW, PROCESSED, BITFINEX_PAIRS,
                    ANNUALIZATION_8H, DELTA_STRESS_BP, ETA_STRESS_SIGMA,
                    ASSET_VENUE)
from econometrics import log, hr

def seeded_ffill(s: pd.Series, grid: pd.DatetimeIndex) -> pd.Series:
    full_idx = s.index.union(grid)
    s_full = s.reindex(full_idx).ffill()
    return s_full.reindex(grid)

def load_bitfinex_close(symbol: str) -> pd.Series:
    p = RAW / "bitfinex" / f"{symbol}_1h.parquet"
    df = pd.read_parquet(p)
    df["ts"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
    df.index = df["ts"].dt.floor("1h")
    s = df[~df.index.duplicated(keep="last")]["close"].sort_index()
    s.name = symbol
    return s

def load_funding(venue: str, sym: str) -> pd.Series:
    p = RAW / venue / f"{sym}_funding_8h.parquet"
    df = pd.read_parquet(p)
    df["ts"] = pd.to_datetime(df["funding_ts_ms"], unit="ms", utc=True)
    df.index = df["ts"].dt.floor("1h")
    s = df[~df.index.duplicated(keep="last")]["fundingRate"].sort_index()
    s.name = f"{venue}_{sym}"
    return s

def load_klines_close(side: str, sym: str) -> pd.Series:
    p = RAW / "binance" / f"{sym}_{side}_1h.parquet"
    df = pd.read_parquet(p)
    df["ts"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
    df.index = df["ts"].dt.floor("1h")
    s = df[~df.index.duplicated(keep="last")]["close"].sort_index()
    s.name = f"{side}_{sym}"
    return s

def main():
    hr()
    log("Building canonical SPLICE panel")
    hr()

    grid = pd.date_range(START_DT, END_DT, freq="1h", inclusive="both", tz="UTC")
    log(f"  grid: {len(grid)} hours, {grid[0]} ~ {grid[-1]}")

    panel = pd.DataFrame(index=grid)
    panel.index.name = "ts"

    for sym, label in BITFINEX_PAIRS.items():
        s = load_bitfinex_close(sym)
        s_ff = seeded_ffill(s, grid)
        panel[f"delta_{label}"] = s_ff - 1.0
        panel[f"price_{label}"] = s_ff
        log(f"  delta_{label}: raw obs {(s.reindex(grid).notna()).sum()}/{len(grid)}, ffilled to {s_ff.notna().sum()}/{len(grid)}")

    for asset in ["BTC", "ETH"]:
        sym = f"{asset}USDT"
        for venue in ["binance", "bybit"]:
            f = load_funding(venue, sym)
            f_ff = seeded_ffill(f, grid)
            panel[f"eta_{asset}_{venue}"] = f_ff
            panel[f"eta_{asset}_{venue}_ann"] = f_ff * ANNUALIZATION_8H
            panel[f"eta_{asset}_{venue}_raw"] = f.reindex(grid)
            log(f"  eta_{asset}_{venue}: events {f.reindex(grid).notna().sum()}, ffill {f_ff.notna().sum()}/{len(grid)}")

    for asset in ["BTC", "ETH"]:
        sym = f"{asset}USDT"
        spot = load_klines_close("spot", sym)
        perp = load_klines_close("perp", sym)
        spot_g = seeded_ffill(spot, grid)
        perp_g = seeded_ffill(perp, grid)
        panel[f"spot_{asset}"] = spot_g
        panel[f"perp_{asset}"] = perp_g
        panel[f"X_{asset}"] = spot_g - perp_g
        panel[f"basis_{asset}"] = (spot_g - perp_g) / spot_g
        log(f"  basis_{asset}: range [{panel[f'basis_{asset}'].min():+.5f}, {panel[f'basis_{asset}'].max():+.5f}]")

    d_thresh = DELTA_STRESS_BP / 10000.0
    delta_stress = (panel["delta_USDT"].abs() > d_thresh) | (panel["delta_USDC"].abs() > d_thresh)

    eta_stress_per = []
    for asset, venue in ASSET_VENUE:
        s = panel[f"eta_{asset}_{venue}_ann"]
        mu = s.mean()
        sd = s.std()
        eta_stress_per.append((s - mu).abs() > ETA_STRESS_SIGMA * sd)
    eta_stress = pd.concat(eta_stress_per, axis=1).any(axis=1)

    panel["regime"] = np.where(delta_stress | eta_stress, "stress", "normal")
    n_s = (panel["regime"] == "stress").sum()
    log(f"  regime: stress={n_s} ({n_s/len(grid)*100:.2f}%), normal={len(grid)-n_s}")

    analysis_cols = (
        ["delta_USDT", "delta_USDC", "basis_BTC", "basis_ETH", "X_BTC", "X_ETH"] +
        [f"eta_{a}_{v}" for a, v in ASSET_VENUE] +
        [f"eta_{a}_{v}_ann" for a, v in ASSET_VENUE]
    )
    nan_summary = panel[analysis_cols].isna().sum()
    log(f"\n  NaN check across {len(analysis_cols)} analysis columns:")
    for c in analysis_cols:
        if nan_summary[c] > 0:
            log(f"    {c}: {nan_summary[c]} NaN  WARNING")
        else:
            log(f"    {c}: 0 NaN OK")

    out = PROCESSED / "panel.parquet"
    panel.to_parquet(out, compression="snappy")
    log(f"\nSaved panel: {out}, shape {panel.shape}")

if __name__ == "__main__":
    main()
