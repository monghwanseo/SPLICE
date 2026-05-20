import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))
from settings import START_MS, END_MS, RAW

OUT = RAW
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "SPLICE-fetcher/1.0"})

EXTRA_ASSETS = ["SOL", "XRP", "BNB", "DOGE", "ADA", "LTC"]

def log(m): print(m, flush=True)

def fetch_binance_funding(symbol):
    url = "https://fapi.binance.com/fapi/v1/fundingRate"
    cur = START_MS; rows = []; page = 0
    while cur <= END_MS:
        params = {"symbol": symbol, "startTime": cur, "endTime": END_MS, "limit": 1000}
        for attempt in range(5):
            try:
                r = SESSION.get(url, params=params, timeout=30)
                if r.status_code == 429:
                    time.sleep(8 * (attempt + 1)); continue
                r.raise_for_status()
                data = r.json(); break
            except Exception as e:
                time.sleep(3 * (attempt + 1))
        else:
            raise RuntimeError(f"failed funding {symbol}")
        if not data: break
        rows.extend(data)
        last_ts = data[-1]["fundingTime"]
        page += 1
        if last_ts >= END_MS or len(data) < 1000: break
        cur = last_ts + 1
        time.sleep(0.25)
    if not rows: return pd.DataFrame()
    df = pd.DataFrame(rows).rename(columns={"fundingTime": "funding_ts_ms"})
    df["fundingRate"] = pd.to_numeric(df["fundingRate"])
    df["markPrice"] = pd.to_numeric(df.get("markPrice", pd.NA), errors="coerce")
    df = df[["funding_ts_ms", "fundingRate", "markPrice"]].drop_duplicates("funding_ts_ms").sort_values("funding_ts_ms").reset_index(drop=True)
    df = df[(df["funding_ts_ms"] >= START_MS) & (df["funding_ts_ms"] <= END_MS)].reset_index(drop=True)
    return df

def fetch_bybit_funding(symbol):
    url = "https://api.bybit.com/v5/market/funding/history"
    cur_end = END_MS
    rows = []; page = 0
    while page < 100:
        params = {"category": "linear", "symbol": symbol, "startTime": START_MS, "endTime": cur_end, "limit": 200}
        for attempt in range(5):
            try:
                r = SESSION.get(url, params=params, timeout=30)
                if r.status_code == 429:
                    time.sleep(8 * (attempt + 1)); continue
                r.raise_for_status()
                data = r.json(); break
            except Exception as e:
                time.sleep(3 * (attempt + 1))
        else:
            raise RuntimeError(f"failed bybit {symbol}")
        rs = data.get("result", {}).get("list", [])
        if not rs: break
        rows.extend(rs)
        oldest = min(int(x["fundingRateTimestamp"]) for x in rs)
        page += 1
        if len(rs) < 200 or oldest <= START_MS: break
        cur_end = oldest - 1
        time.sleep(0.25)
    if not rows: return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["funding_ts_ms"] = pd.to_numeric(df["fundingRateTimestamp"]).astype("int64")
    df["fundingRate"] = pd.to_numeric(df["fundingRate"])
    df = df[["funding_ts_ms", "fundingRate"]].drop_duplicates("funding_ts_ms").sort_values("funding_ts_ms").reset_index(drop=True)
    df = df[(df["funding_ts_ms"] >= START_MS) & (df["funding_ts_ms"] <= END_MS)].reset_index(drop=True)
    return df

def fetch_klines(host, symbol, side):
    is_spot = "fapi" not in host
    url = f"{host}/api/v3/klines" if is_spot else f"{host}/fapi/v1/klines"
    limit = 1000 if is_spot else 1500
    cur = START_MS; rows = []; page = 0
    while cur <= END_MS:
        params = {"symbol": symbol, "interval": "1h", "startTime": cur, "endTime": END_MS, "limit": limit}
        for attempt in range(5):
            try:
                r = SESSION.get(url, params=params, timeout=30)
                if r.status_code == 429:
                    time.sleep(8 * (attempt + 1)); continue
                r.raise_for_status()
                data = r.json(); break
            except Exception as e:
                time.sleep(3 * (attempt + 1))
        else:
            raise RuntimeError(f"failed klines {symbol}/{side}")
        if not data: break
        rows.extend(data)
        last_open = data[-1][0]
        page += 1
        if last_open >= END_MS: break
        cur = last_open + 1
        time.sleep(0.2)
    if not rows: return pd.DataFrame()
    cols = ["open_time", "open", "high", "low", "close", "volume", "close_time",
            "quote_vol", "trades", "tb_base", "tb_quote", "ignore"]
    df = pd.DataFrame(rows, columns=cols)
    df = df[["open_time", "close"]].rename(columns={"open_time": "ts_ms"})
    df["close"] = pd.to_numeric(df["close"])
    df = df.drop_duplicates("ts_ms").sort_values("ts_ms").reset_index(drop=True)
    df = df[(df["ts_ms"] >= START_MS) & (df["ts_ms"] <= END_MS)].reset_index(drop=True)
    return df

def main():
    log("=" * 70)
    log("Fetching extra assets for cross-asset robustness (E10)")
    log("=" * 70)
    for asset in EXTRA_ASSETS:
        sym = f"{asset}USDT"
        log(f"\n>>> {asset}")
        df = fetch_binance_funding(sym)
        df.to_parquet(OUT / "binance" / f"{sym}_funding_8h.parquet", compression="snappy", index=False)
        log(f"  binance funding: {len(df)} rows")
        df = fetch_bybit_funding(sym)
        df.to_parquet(OUT / "bybit" / f"{sym}_funding_8h.parquet", compression="snappy", index=False)
        log(f"  bybit funding: {len(df)} rows")
        df = fetch_klines("https://api.binance.com", sym, "spot")
        df.to_parquet(OUT / "binance" / f"{sym}_spot_1h.parquet", compression="snappy", index=False)
        log(f"  spot 1h: {len(df)} rows")
        df = fetch_klines("https://fapi.binance.com", sym, "perp")
        df.to_parquet(OUT / "binance" / f"{sym}_perp_1h.parquet", compression="snappy", index=False)
        log(f"  perp 1h: {len(df)} rows")

if __name__ == "__main__":
    main()
