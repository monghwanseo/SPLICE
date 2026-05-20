import urllib.request
import sys
import io
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from settings import RAW

print("Fetching FRED DGS3MO (3-month Treasury rate)...")

url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS3MO"
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
with urllib.request.urlopen(req, timeout=30) as r:
    raw = r.read().decode()

df = pd.read_csv(io.StringIO(raw))
print(f"Columns: {df.columns.tolist()}")
df.columns = [c.strip().lower() for c in df.columns]

date_col = "observation_date" if "observation_date" in df.columns else "date"
rate_col = "dgs3mo" if "dgs3mo" in df.columns else df.columns[1]
df["date"] = pd.to_datetime(df[date_col], utc=True)
df["rate_pct"] = pd.to_numeric(df[rate_col], errors="coerce")
df = df[["date", "rate_pct"]].dropna()

df = df[(df["date"] >= pd.Timestamp("2020-11-01", tz="UTC")) &
        (df["date"] <= pd.Timestamp("2026-04-30", tz="UTC"))]

out = RAW / "fred"
out.mkdir(exist_ok=True)
df.to_parquet(out / "DGS3MO_daily.parquet", index=False)
print(f"Saved: {out / 'DGS3MO_daily.parquet'}, {len(df)} rows")
print(f"Range: {df['date'].min()} ~ {df['date'].max()}")
print(f"Rate range: {df['rate_pct'].min():.2f}% to {df['rate_pct'].max():.2f}%")
