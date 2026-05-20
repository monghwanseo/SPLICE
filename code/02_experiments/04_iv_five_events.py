import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from settings import PROCESSED, RESULTS, EVENTS, SEED, NW_LAG
from econometrics import log, hr

import statsmodels.api as sm

panel = pd.read_parquet(PROCESSED / "panel.parquet")
OUT = RESULTS / "tables"

hr()
log("E13: GMM (2SLS) with event-window instruments")
hr()

def build_event_indicator(window_hours=72):
    ind = pd.Series(0, index=panel.index, name="ev")
    for ev_name, ev_str, _ in EVENTS:
        ev_t = pd.Timestamp(ev_str, tz="UTC")
        if ev_t < panel.index.min() or ev_t > panel.index.max():
            continue
        mask = (panel.index >= ev_t - pd.Timedelta(hours=window_hours)) & \
               (panel.index <= ev_t + pd.Timedelta(hours=window_hours))
        ind = ind + mask.astype(int)
    return ind > 0

event_ind = build_event_indicator(72)
log(f"  Event indicator coverage: {event_ind.sum()}/{len(event_ind)} hours ({event_ind.mean()*100:.2f}%)")

records = []
for stable in ["USDT", "USDC"]:
    d_col = f"delta_{stable}"
    for asset in ["BTC", "ETH"]:
        b_col = f"basis_{asset}"
        df = pd.DataFrame({
            "X": panel[b_col],
            "delta": panel[d_col],
            "Z": event_ind.astype(float),
        }).dropna()

        ols_res = sm.OLS(df["X"], sm.add_constant(df[["delta"]])).fit()
        ols_g = float(ols_res.params["delta"])

        fs = sm.OLS(df["delta"], sm.add_constant(df[["Z"]])).fit()
        delta_hat = fs.fittedvalues
        ss = sm.OLS(df["X"], sm.add_constant(pd.DataFrame({"delta_hat": delta_hat}, index=df.index))).fit()
        iv_g = float(ss.params["delta_hat"])
        fs_t = float(fs.tvalues["Z"])
        fs_f = fs_t ** 2

        log(f"  {stable} -> basis_{asset}:  OLS gamma={ols_g:+.4f}  IV gamma={iv_g:+.4f}  "
            f"first-stage F = {fs_f:.2f} (Z->delta t={fs_t:+.2f})")
        records.append({
            "stable": stable, "asset": asset,
            "gamma_OLS": ols_g, "gamma_IV": iv_g,
            "first_stage_F": fs_f, "first_stage_t": fs_t,
            "n": len(df),
        })

pd.DataFrame(records).to_parquet(OUT / "e13_iv_gmm.parquet", index=False)
log("\nE13 done.")
