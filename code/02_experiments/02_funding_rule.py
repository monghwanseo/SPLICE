import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from settings import PROCESSED, RESULTS, ASSET_VENUE, SEED, NW_LAG, BOOT_REPS, BOOT_BLOCK
from econometrics import log, hr, ols_nw, block_bootstrap_resample, boot_ci

panel = pd.read_parquet(PROCESSED / "panel.parquet")

OUT = RESULTS / "tables"

records = []
hr()
log("E2: Stage I  -  eta_t^{a,v} = alpha + (-lambda)_{a,v} * b_t^{a,v} + u_t")
hr()

for asset, venue in ASSET_VENUE:
    eta = panel[f"eta_{asset}_{venue}_ann"]
    basis = panel[f"basis_{asset}"]
    df = pd.concat([eta.rename("eta"), basis.rename("basis")], axis=1).dropna()

    res = ols_nw(df["eta"], df[["basis"]], lag=NW_LAG)
    beta = float(res.params["basis"])
    se = float(res.bse["basis"])
    lam = -beta
    log(f"  {asset}_{venue}:  beta(basis)={beta:+.3f}  SE={se:.3f}  t={beta/se:+.2f}  "
        f"lambda={lam:+.3f}  R2={res.rsquared:.4f}  n={int(res.nobs)}")
    records.append({
        "asset": asset, "venue": venue,
        "beta_basis": beta, "se": se, "t": beta/se, "lambda": lam,
        "r2": float(res.rsquared), "n": int(res.nobs),
    })

log("\nBlock bootstrap CIs for lambda")
rng = np.random.default_rng(SEED)
boot_records = []
for asset, venue in ASSET_VENUE:
    df = panel[[f"eta_{asset}_{venue}_ann", f"basis_{asset}"]].dropna()
    y = df.iloc[:, 0].to_numpy()
    Xmat = np.column_stack([np.ones(len(df)), df.iloc[:, 1].to_numpy()])
    n = len(df)
    boots = np.empty(BOOT_REPS)
    for k in range(BOOT_REPS):
        idx = block_bootstrap_resample(n, BOOT_BLOCK, rng)
        beta, *_ = np.linalg.lstsq(Xmat[idx], y[idx], rcond=None)
        boots[k] = -beta[1]
    ci = boot_ci(boots)
    log(f"  {asset}_{venue}: lambda 95% CI = [{ci[0]:+.4f}, {ci[1]:+.4f}]")
    boot_records.append({
        "asset": asset, "venue": venue,
        "lambda_lo95": ci[0], "lambda_hi95": ci[1],
        "B": BOOT_REPS, "seed": SEED,
    })

pd.DataFrame(records).to_parquet(OUT / "e2_second_stage.parquet", index=False)
pd.DataFrame(boot_records).to_parquet(OUT / "e2_second_stage_bootstrap.parquet", index=False)
log("\nE2 done.")
