import urllib.request
import json
import ssl
import time
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from settings import START_MS, END_MS, RAW

CTX = ssl.create_default_context()

def log(m): print(m, flush=True)

def get(url, retries=3):
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30, context=CTX) as r:
                return json.loads(r.read().decode())
        except Exception as e:
            if i == retries - 1:
                raise
            time.sleep(2 * (i + 1))

log("=" * 70)
log("Fetching Aave V3 USDT/USDC lending APY (DefiLlama)")
log("=" * 70)

POOLS = {
    "aave_v3_USDT": "f981a304-bb6c-45b8-b0c5-fd2f515ad23a",
    "aave_v3_USDC": "aa70268e-4b52-42bf-a116-608b370f9501",
}

aave_dir = RAW / "aave"
aave_dir.mkdir(exist_ok=True)

for label, pool_id in POOLS.items():
    chart = get(f"https://yields.llama.fi/chart/{pool_id}")
    rows = chart.get("data", chart) if isinstance(chart, dict) else chart
    if isinstance(rows, dict) and "data" in rows:
        rows = rows["data"]
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df[["ts", "apy", "apyBase", "tvlUsd"]].rename(columns={"apy": "lend_apy_pct"})
    df = df.sort_values("ts").reset_index(drop=True)
    out = aave_dir / f"{label}_lendapy_daily.parquet"
    df.to_parquet(out, compression="snappy", index=False)
    log(f"  {label}: {len(df)} rows, {df['ts'].min()} ~ {df['ts'].max()}")

log("\n" + "=" * 70)
log("Fetching Binance DAI/TUSD/FDUSD spot klines")
log("=" * 70)

def fetch_klines(host, symbol, label, interval="1h"):
    is_spot = "fapi" not in host
    url = f"{host}/api/v3/klines" if is_spot else f"{host}/fapi/v1/klines"
    limit = 1000 if is_spot else 1500
    cur = START_MS
    rows = []
    page = 0
    while cur <= END_MS:
        params = f"?symbol={symbol}&interval={interval}&startTime={cur}&endTime={END_MS}&limit={limit}"
        for attempt in range(5):
            try:
                full_url = url + params
                with urllib.request.urlopen(full_url, timeout=30, context=CTX) as r:
                    data = json.loads(r.read().decode())
                break
            except Exception as e:
                time.sleep(3 * (attempt + 1))
        else:
            raise RuntimeError(f"failed {label}")
        if not data:
            break
        rows.extend(data)
        last_open = data[-1][0]
        page += 1
        if last_open >= END_MS:
            break
        cur = last_open + 1
        time.sleep(0.2)
    if not rows:
        return pd.DataFrame()
    cols = ["open_time", "open", "high", "low", "close", "volume", "close_time",
            "quote_vol", "trades", "tb_base", "tb_quote", "ignore"]
    df = pd.DataFrame(rows, columns=cols)
    df = df[["open_time", "close", "volume", "quote_vol", "trades"]].rename(columns={"open_time": "ts_ms"})
    for c in ["close", "volume", "quote_vol", "trades"]:
        df[c] = pd.to_numeric(df[c])
    df = df.drop_duplicates("ts_ms").sort_values("ts_ms").reset_index(drop=True)
    df = df[(df["ts_ms"] >= START_MS) & (df["ts_ms"] <= END_MS)].reset_index(drop=True)
    return df

for sym in ["DAIUSDT", "TUSDUSDT", "FDUSDUSDT"]:
    df = fetch_klines("https://api.binance.com", sym, f"binance/{sym}")
    out = RAW / "binance" / f"{sym}_spot_1h.parquet"
    df.to_parquet(out, compression="snappy", index=False)
    log(f"  {sym}: {len(df)} rows")

log("\n" + "=" * 70)
log("Re-fetching BTCUSDT/ETHUSDT spot/perp with volume (for liquidity controls)")
log("=" * 70)

for sym in ["BTCUSDT", "ETHUSDT"]:
    for host, side in [("https://api.binance.com", "spot"), ("https://fapi.binance.com", "perp")]:
        df = fetch_klines(host, sym, f"{side}/{sym}_with_vol")
        out = RAW / "binance" / f"{sym}_{side}_1h_full.parquet"
        df.to_parquet(out, compression="snappy", index=False)
        log(f"  {sym} {side}: {len(df)} rows (with volume)")

log("\nDone.")
