import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from settings import PROCESSED, RESULTS, NW_LAG
from econometrics import log, hr, ols_nw

panel = pd.read_parquet(PROCESSED / "panel.parquet")
OUT = RESULTS / "tables"

stress_mask = panel["regime"] == "stress"

hr()
log("E21: Rigobon (2003) heteroskedasticity-based identification")
hr()

records = []
for stable in ["USDT", "USDC"]:
    d = panel[f"delta_{stable}"]
    for asset in ["BTC", "ETH"]:
        b = panel[f"basis_{asset}"]

        d_s = d[stress_mask]; b_s = b[stress_mask]
        d_n = d[~stress_mask]; b_n = b[~stress_mask]

        cov_xd_s = float(np.cov(b_s, d_s)[0, 1])
        cov_xd_n = float(np.cov(b_n, d_n)[0, 1])
        var_d_s = float(np.var(d_s))
        var_d_n = float(np.var(d_n))

        denom = var_d_s - var_d_n
        if abs(denom) > 1e-12:
            gamma_R = (cov_xd_s - cov_xd_n) / denom
        else:
            gamma_R = np.nan

        ols_res = ols_nw(b, d.rename("d").to_frame(), lag=NW_LAG)
        ols_g = float(ols_res.params["d"])

        s_res = ols_nw(b_s, d_s.rename("d").to_frame(), lag=NW_LAG)
        ols_g_s = float(s_res.params["d"])
        n_res = ols_nw(b_n, d_n.rename("d").to_frame(), lag=NW_LAG)
        ols_g_n = float(n_res.params["d"])

        log(f"\n  {stable} -> basis_{asset}:")
        log(f"    moments:  Var_S(d)={var_d_s:.6e}  Var_N(d)={var_d_n:.6e}  "
            f"Cov_S(X,d)={cov_xd_s:.6e}  Cov_N(X,d)={cov_xd_n:.6e}")
        log(f"    OLS gamma (full)   = {ols_g:+.4f}")
        log(f"    OLS gamma (stress) = {ols_g_s:+.4f}")
        log(f"    OLS gamma (normal) = {ols_g_n:+.4f}")
        log(f"    Rigobon gamma_R    = {gamma_R:+.4f}")

        records.append({
            "stable": stable, "asset": asset,
            "var_delta_stress": var_d_s, "var_delta_normal": var_d_n,
            "cov_X_delta_stress": cov_xd_s, "cov_X_delta_normal": cov_xd_n,
            "gamma_OLS": ols_g, "gamma_OLS_stress": ols_g_s, "gamma_OLS_normal": ols_g_n,
            "gamma_Rigobon": gamma_R,
        })

pd.DataFrame(records).to_parquet(OUT / "e21_rigobon.parquet", index=False)
log("\nE21 done.")
