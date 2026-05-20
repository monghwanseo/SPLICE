import urllib.request
import json
import ssl
import sys
import time
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from settings import START_MS, END_MS, RAW

CTX = ssl.create_default_context()

def log(m): print(m, flush=True)

def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30, context=CTX) as r:
        return json.loads(r.read().decode())

log("=" * 70)
log("Probing Bitstamp DAI/USD launch")
log("=" * 70)

for year in range(2018, 2027):
    ts = int(datetime(year, 1, 1, tzinfo=timezone.utc).timestamp())
    url = f"https://www.bitstamp.net/api/v2/ohlc/daiusd/?step=86400&limit=10&start={ts}"
    try:
        data = get(url)
        ohlc = data.get("data", {}).get("ohlc", [])
        if ohlc:
            first_ts = min(int(c["timestamp"]) for c in ohlc)
            log(f"  {year}: HIT, first bar = {datetime.fromtimestamp(first_ts, tz=timezone.utc).strftime('%Y-%m-%d')}")
            break
        else:
            log(f"  {year}: empty")
    except Exception as e:
        log(f"  {year}: ERR {e}")
    time.sleep(0.3)

log("\n" + "=" * 70)
log("Fetching DAI/USD 1h from Bitstamp")
log("=" * 70)

def fetch_bitstamp(pair, start_ms, end_ms, step=3600, limit=1000):
    all_rows = []
    cur = start_ms // 1000
    end_s = end_ms // 1000
    page = 0
    while cur <= end_s and page < 200:
        url = f"https://www.bitstamp.net/api/v2/ohlc/{pair}/?step={step}&limit={limit}&start={cur}"
        for attempt in range(3):
            try:
                data = get(url)
                break
            except Exception as e:
                time.sleep(2 * (attempt + 1))
        else:
            log(f"  page {page} failed; stopping")
            break
        ohlc = data.get("data", {}).get("ohlc", [])
        if not ohlc:
            break
        all_rows.extend(ohlc)
        last_ts = max(int(c["timestamp"]) for c in ohlc)
        log(f"  page {page}: {len(ohlc)} rows, last={datetime.fromtimestamp(last_ts, tz=timezone.utc)}")
        page += 1
        if last_ts >= end_s or len(ohlc) < limit:
            break
        cur = last_ts + 1
        time.sleep(0.4)
    return all_rows

DAI_START_MS = int(datetime(2022, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
rows = fetch_bitstamp("daiusd", DAI_START_MS, END_MS)
if rows:
    df = pd.DataFrame(rows)
    df["ts_ms"] = pd.to_numeric(df["timestamp"]) * 1000
    df["close"] = pd.to_numeric(df["close"])
    df = df[["ts_ms", "close"]].drop_duplicates("ts_ms").sort_values("ts_ms").reset_index(drop=True)
    df = df[(df["ts_ms"] >= START_MS) & (df["ts_ms"] <= END_MS)].reset_index(drop=True)
    out = RAW / "bitstamp"
    out.mkdir(exist_ok=True)
    df.to_parquet(out / "DAIUSD_1h.parquet", compression="snappy", index=False)
    log(f"\nSaved: {out / 'DAIUSD_1h.parquet'}, {len(df)} rows")
else:
    log("\nNo data fetched")
