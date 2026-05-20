import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from settings import PROCESSED, RESULTS, ASSET_VENUE, SEED, NW_LAG, BOOT_REPS, BOOT_BLOCK
from econometrics import log, hr, ols_nw, block_bootstrap_resample, boot_ci

panel = pd.read_parquet(PROCESSED / "panel.parquet")
log(f"Panel: {panel.shape}")

OUT = RESULTS / "tables"

panel["hod"] = panel.index.hour
panel["dow"] = panel.index.dayofweek

records = []

hr()
log("E1: First-stage Stage II  -  b_{a,v}_t = alpha + gamma_c * delta_t^c + controls + u_t")
hr()

def estimate_one(b_col, X_df, label, n_dummies=0):
    res = ols_nw(panel[b_col], X_df, lag=NW_LAG)
    rec = {
        "spec": label, "b_col": b_col, "n": int(res.nobs),
        "r2": float(res.rsquared), "r2_adj": float(res.rsquared_adj),
    }
    for k in ["delta_USDT", "delta_USDC", "basis_lag", "delta_USDT_lag", "delta_USDC_lag"]:
        if k in res.params.index:
            rec[f"{k}_coef"] = float(res.params[k])
            rec[f"{k}_se"] = float(res.bse[k])
            rec[f"{k}_t"] = float(res.tvalues[k])
            rec[f"{k}_p"] = float(res.pvalues[k])
    return rec, res

for asset in ["BTC", "ETH"]:
    b_col = f"basis_{asset}"
    log(f"\n--- {b_col} ---")

    X = panel[["delta_USDT"]]
    rec, res = estimate_one(b_col, X, "M1_USDT_only")
    rec["asset"] = asset
    log(f"  M1 (USDT only): gamma={rec['delta_USDT_coef']:+.4f}  SE={rec['delta_USDT_se']:.4f}  "
        f"t={rec['delta_USDT_t']:+.2f}  R2={rec['r2']:.4f}")
    records.append(rec)

    X = panel[["delta_USDC"]]
    rec, res = estimate_one(b_col, X, "M2_USDC_only")
    rec["asset"] = asset
    log(f"  M2 (USDC only): gamma={rec['delta_USDC_coef']:+.4f}  SE={rec['delta_USDC_se']:.4f}  "
        f"t={rec['delta_USDC_t']:+.2f}  R2={rec['r2']:.4f}")
    records.append(rec)

    X = panel[["delta_USDT", "delta_USDC"]]
    rec, res = estimate_one(b_col, X, "M3_joint")
    rec["asset"] = asset
    log(f"  M3 (joint):     gamma_USDT={rec['delta_USDT_coef']:+.4f}({rec['delta_USDT_t']:+.1f})  "
        f"gamma_USDC={rec['delta_USDC_coef']:+.4f}({rec['delta_USDC_t']:+.1f})  R2={rec['r2']:.4f}")
    records.append(rec)

    panel[f"basis_{asset}_lag1"] = panel[b_col].shift(1)
    X = panel[["delta_USDT", "delta_USDC", f"basis_{asset}_lag1"]].rename(
        columns={f"basis_{asset}_lag1": "basis_lag"})
    rec, res = estimate_one(b_col, X, "M4_AR_augmented")
    rec["asset"] = asset
    log(f"  M4 (+ lag-1):   gamma_USDT={rec['delta_USDT_coef']:+.4f}({rec['delta_USDT_t']:+.1f})  "
        f"gamma_USDC={rec['delta_USDC_coef']:+.4f}({rec['delta_USDC_t']:+.1f})  "
        f"phi={rec['basis_lag_coef']:.4f}({rec['basis_lag_t']:+.1f})  R2={rec['r2']:.4f}")
    records.append(rec)

    hod = pd.get_dummies(panel["hod"], prefix="hod", drop_first=True).astype(float)
    dow = pd.get_dummies(panel["dow"], prefix="dow", drop_first=True).astype(float)
    X = pd.concat([panel[["delta_USDT", "delta_USDC", f"basis_{asset}_lag1"]].rename(
        columns={f"basis_{asset}_lag1": "basis_lag"}), hod, dow], axis=1)
    rec, res = estimate_one(b_col, X, "M5_FE_AR")
    rec["asset"] = asset
    log(f"  M5 (+ FE):      gamma_USDT={rec['delta_USDT_coef']:+.4f}({rec['delta_USDT_t']:+.1f})  "
        f"gamma_USDC={rec['delta_USDC_coef']:+.4f}({rec['delta_USDC_t']:+.1f})  R2={rec['r2']:.4f}")
    records.append(rec)

hr()
log("\nBlock bootstrap CIs for M3 (joint USDT+USDC) ...")
hr()

rng = np.random.default_rng(SEED)
boot_records = []
for asset in ["BTC", "ETH"]:
    b_col = f"basis_{asset}"
    df = panel[[b_col, "delta_USDT", "delta_USDC"]].dropna()
    y = df[b_col].to_numpy()
    X = np.column_stack([np.ones(len(df)), df["delta_USDT"].to_numpy(), df["delta_USDC"].to_numpy()])
    n = len(df)

    boots = np.empty((BOOT_REPS, 2))
    for k in range(BOOT_REPS):
        idx = block_bootstrap_resample(n, BOOT_BLOCK, rng)
        beta, *_ = np.linalg.lstsq(X[idx], y[idx], rcond=None)
        boots[k] = [beta[1], beta[2]]
    ci_t = boot_ci(boots[:, 0])
    ci_c = boot_ci(boots[:, 1])
    log(f"  basis_{asset}: gamma_USDT 95% CI = [{ci_t[0]:+.4f}, {ci_t[1]:+.4f}]  "
        f"gamma_USDC 95% CI = [{ci_c[0]:+.4f}, {ci_c[1]:+.4f}]")
    boot_records.append({
        "asset": asset,
        "gamma_USDT_lo95": ci_t[0], "gamma_USDT_hi95": ci_t[1],
        "gamma_USDC_lo95": ci_c[0], "gamma_USDC_hi95": ci_c[1],
        "B": BOOT_REPS, "block": BOOT_BLOCK, "seed": SEED,
    })

pd.DataFrame(records).to_parquet(OUT / "e1_first_stage.parquet", index=False)
pd.DataFrame(boot_records).to_parquet(OUT / "e1_first_stage_bootstrap.parquet", index=False)
log("\nE1 done.")
