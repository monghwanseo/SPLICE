from datetime import datetime, timezone
from pathlib import Path

SEED = 2026

START_ISO = "2020-11-01T00:00:00Z"
END_ISO = "2026-04-30T23:00:00Z"
START_DT = datetime(2020, 11, 1, 0, 0, 0, tzinfo=timezone.utc)
END_DT = datetime(2026, 4, 30, 23, 0, 0, tzinfo=timezone.utc)
START_MS = int(START_DT.timestamp() * 1000)
END_MS = int(END_DT.timestamp() * 1000)

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data_raw"
RESULTS = ROOT / "results"
LOGS = ROOT / "logs"
PAPER = ROOT / "paper"
PROCESSED = ROOT / "data_processed"
for p in [RESULTS, LOGS, PROCESSED, RESULTS / "tables", RESULTS / "intermediate"]:
    p.mkdir(parents=True, exist_ok=True)

BITFINEX_PAIRS = {"tUSTUSD": "USDT", "tUDCUSD": "USDC"}
ASSETS = ["BTC", "ETH"]
VENUES = ["binance", "bybit"]
STABLECOINS = ["USDT", "USDC"]

ASSET_VENUE = [(a, v) for a in ASSETS for v in VENUES]

ANNUALIZATION_8H = 365 * 24 / 8

NW_LAG = 24
BOOT_BLOCK = 168
BOOT_REPS = 5000

DELTA_STRESS_BP = 20
ETA_STRESS_SIGMA = 2.0

EVENTS = [
    ("LUNA_collapse",    "2022-05-12 00:00", "Terra/UST collapse"),
    ("FTX_collapse",     "2022-11-08 00:00", "FTX bankruptcy"),
    ("USDC_depeg",       "2023-03-10 22:00", "Circle SVB exposure"),
    ("BTC_ETF_approval", "2024-01-10 22:00", "Spot BTC ETF approval"),
    ("Ethena_yen",       "2024-08-05 06:00", "Yen carry / Ethena stress"),
]
EVENT_WINDOW_HOURS = 7 * 24

OOS_SPLIT_ISO = "2025-01-01T00:00:00Z"
