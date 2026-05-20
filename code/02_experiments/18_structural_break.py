import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))
from settings import PROCESSED, RESULTS
from econometrics import log, hr

panel = pd.read_parquet(PROCESSED / "panel.parquet")
OUT = RESULTS / "tables"

hr()
log("E34: Quandt-Andrews supremum-F structural break test")
hr()

def supremum_F(y, x_only_with_const, trim=0.15):
    n = len(y)
    t_lo = int(n * trim)
    t_hi = int(n * (1 - trim))
    Xc = sm.add_constant(x_only_with_const)
    full = sm.OLS(y, Xc).fit()
    rss_full = ((full.resid) ** 2).sum()
    k = Xc.shape[1]

    best_F = -np.inf
    best_t = -1
    sub_step = max(1, (t_hi - t_lo) // 200)
    for t in range(t_lo, t_hi, sub_step):
        y1, X1 = y[:t], Xc[:t]
        y2, X2 = y[t:], Xc[t:]
        if len(y1) < k + 2 or len(y2) < k + 2:
            continue
        try:
            res1 = sm.OLS(y1, X1).fit()
            res2 = sm.OLS(y2, X2).fit()
            rss_split = ((res1.resid) ** 2).sum() + ((res2.resid) ** 2).sum()
            F = ((rss_full - rss_split) / k) / (rss_split / (n - 2 * k))
            if F > best_F:
                best_F = F
                best_t = t
        except Exception:
            continue
    return best_F, best_t

records = []
for stable in ["USDT", "USDC"]:
    d_col = f"delta_{stable}"
    for asset in ["BTC", "ETH"]:
        b_col = f"basis_{asset}"
        df = panel[[b_col, d_col]].dropna().reset_index(drop=True)
        y = df[b_col].to_numpy()
        x = df[d_col].to_numpy().reshape(-1, 1)

        sup_F, best_t = supremum_F(y, x, trim=0.15)
        idx_dates = panel[[b_col, d_col]].dropna().index
        if 0 <= best_t < len(idx_dates):
            best_date = idx_dates[best_t]
        else:
            best_date = "n/a"

        crit_10 = 8.85
        crit_5 = 11.69
        crit_1 = 16.39
        sig = "***" if sup_F > crit_1 else ("**" if sup_F > crit_5 else ("*" if sup_F > crit_10 else "n.s."))

        log(f"  {stable} -> basis_{asset}:  sup-F = {sup_F:.2f} {sig}  best_break = {best_date}")
        records.append({
            "stable": stable, "asset": asset,
            "sup_F": float(sup_F),
            "best_break_idx": int(best_t),
            "best_break_date": str(best_date),
            "crit_10": crit_10, "crit_5": crit_5, "crit_1": crit_1,
            "significance": sig,
        })

pd.DataFrame(records).to_parquet(OUT / "e34_quandt_andrews.parquet", index=False)
log("\nE34 done.")
