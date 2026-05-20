import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))
from settings import PROCESSED, RESULTS, ASSET_VENUE, SEED, OOS_SPLIT_ISO, NW_LAG
from econometrics import log, hr

import statsmodels.api as sm
from scipy.stats import norm

panel = pd.read_parquet(PROCESSED / "panel.parquet")
OUT = RESULTS / "tables"

split_t = pd.Timestamp(OOS_SPLIT_ISO, tz="UTC")
hr()
log(f"E9: Out-of-sample forecast | train < {split_t} | test >= {split_t}")
log(f"  train hours: {(panel.index < split_t).sum()},  test hours: {(panel.index >= split_t).sum()}")
hr()

def fit_predict(X_train, y_train, X_test):
    Xt = sm.add_constant(X_train, has_constant="add")
    Xs = sm.add_constant(X_test, has_constant="add")
    res = sm.OLS(y_train, Xt).fit()
    return res.predict(Xs), res.params

def diebold_mariano(d, h=1):
    n = len(d)
    if n < 10:
        return np.nan, np.nan
    mean_d = d.mean()
    lag = max(1, h - 1)
    autoc = np.array([np.cov(d[:-k], d[k:])[0, 1] if k > 0 else d.var() for k in range(lag + 1)])
    weights = 1 - np.arange(lag + 1) / (lag + 1)
    var_d = (autoc[0] + 2 * np.sum(weights[1:] * autoc[1:])) / n
    if var_d <= 0:
        return np.nan, np.nan
    dm = mean_d / np.sqrt(var_d)
    p = 2 * (1 - norm.cdf(abs(dm)))
    return float(dm), float(p)

def clark_west(y, f1, f2, h=1):
    e1 = y - f1
    e2 = y - f2
    f_diff = f1 - f2
    cw = e1 ** 2 - e2 ** 2 + f_diff ** 2
    n = len(cw)
    mean_cw = cw.mean()
    var_cw = cw.var() / n
    if var_cw <= 0:
        return np.nan, np.nan
    z = mean_cw / np.sqrt(var_cw)
    p = 1 - norm.cdf(z)
    return float(z), float(p)

HORIZONS = [1, 4, 8, 24]
records = []
all_preds = []

for asset, venue in ASSET_VENUE:
    e_col = f"eta_{asset}_{venue}_ann"
    b_col = f"basis_{asset}"
    log(f"\n--- {e_col} ---")

    for stable in ["USDT"]:
        d_col = f"delta_{stable}"

        for h in HORIZONS:
            df = pd.DataFrame({
                "y": panel[e_col].shift(-h),
                "eta_t": panel[e_col],
                "delta_t": panel[d_col],
                "basis_t": panel[b_col],
            }).dropna()

            train = df[df.index < split_t]
            test = df[df.index >= split_t]
            if len(test) < 100:
                continue

            yt, ys = train["y"], test["y"]

            f0, _ = fit_predict(train[["eta_t"]], yt, test[["eta_t"]])
            f1, _ = fit_predict(train[["delta_t"]], yt, test[["delta_t"]])
            f2, _ = fit_predict(train[["basis_t"]], yt, test[["basis_t"]])
            f3, _ = fit_predict(train[["eta_t", "delta_t"]], yt, test[["eta_t", "delta_t"]])
            f4, _ = fit_predict(train[["eta_t", "delta_t", "basis_t"]], yt,
                                  test[["eta_t", "delta_t", "basis_t"]])

            mean_y = yt.mean()
            sse_naive = ((ys - mean_y) ** 2).sum()
            for fname, f in [("AR1", f0), ("Delta", f1), ("Basis", f2),
                             ("AR+D", f3), ("SPLICE", f4)]:
                e = (ys - f).to_numpy()
                rmse = np.sqrt(np.mean(e ** 2))
                oos_r2 = 1 - (e ** 2).sum() / sse_naive

                records.append({
                    "asset": asset, "venue": venue, "stable": stable, "h": h,
                    "model": fname, "n_test": len(test),
                    "rmse": rmse, "oos_r2": oos_r2,
                })

            for fname, f in [("Delta", f1), ("Basis", f2), ("AR+D", f3), ("SPLICE", f4)]:
                e_ar = (ys - f0).to_numpy()
                e_m = (ys - f).to_numpy()
                d = e_ar ** 2 - e_m ** 2
                dm, dm_p = diebold_mariano(d, h=h)
                cw, cw_p = clark_west(ys.to_numpy(), f.to_numpy(), f0.to_numpy(), h=h)
                log(f"  h={h:>2d}h  {fname:7s}  vs AR1:  DM={dm:+.2f} (p={dm_p:.2e})  "
                    f"CW={cw:+.2f} (p={cw_p:.2e})")
                records.append({
                    "asset": asset, "venue": venue, "stable": stable, "h": h,
                    "model": f"{fname}_vs_AR1", "n_test": len(test),
                    "DM_stat": dm, "DM_p": dm_p, "CW_stat": cw, "CW_p": cw_p,
                })

pd.DataFrame(records).to_parquet(OUT / "e9_oos_forecast.parquet", index=False)
log("\nE9 done.")
