import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from settings import PROCESSED, RESULTS, ASSET_VENUE, SEED, NW_LAG, BOOT_REPS, BOOT_BLOCK
from econometrics import log, hr, ols_nw, block_bootstrap_resample, boot_ci

panel = pd.read_parquet(PROCESSED / "panel.parquet")
OUT = RESULTS / "tables"

e1 = pd.read_parquet(OUT / "e1_first_stage.parquet")
e2 = pd.read_parquet(OUT / "e2_second_stage.parquet")

hr()
log("E3: Reduced-form  eta_t^{a,v} = alpha + c_c * delta_t^c + u_t")
hr()

records = []
for asset, venue in ASSET_VENUE:
    eta_col = f"eta_{asset}_{venue}_ann"
    for stable in ["USDT", "USDC"]:
        d_col = f"delta_{stable}"
        df = panel[[eta_col, d_col]].dropna()
        res = ols_nw(df[eta_col], df[[d_col]], lag=NW_LAG)
        c = float(res.params[d_col])
        se = float(res.bse[d_col])

        lam = float(e2[(e2["asset"] == asset) & (e2["venue"] == venue)]["lambda"].iloc[0])
        gamma = float(e1[(e1["spec"] == "M3_joint") & (e1["asset"] == asset)
                         ][f"delta_{stable}_coef"].iloc[0])
        implied = -lam * gamma
        residual = c - implied

        log(f"  {asset}_{venue} on {stable}:  c={c:+.3f} SE={se:.3f} t={c/se:+.2f}  "
            f"implied(-lambda*gamma)={implied:+.3f}  residual={residual:+.3f} "
            f"({100*residual/c:+.1f}%)  R2={res.rsquared:.4f}")

        records.append({
            "asset": asset, "venue": venue, "stablecoin": stable,
            "c": c, "se": se, "t": c/se,
            "lambda": lam, "gamma": gamma,
            "implied_lambda_gamma": implied,
            "residual": residual, "residual_pct": 100*residual/c if c != 0 else np.nan,
            "r2": float(res.rsquared), "n": int(res.nobs),
        })

pd.DataFrame(records).to_parquet(OUT / "e3_reduced_form.parquet", index=False)
log("\nE3 done.")
