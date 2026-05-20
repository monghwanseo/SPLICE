import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from settings import PROCESSED, RESULTS, ASSET_VENUE, SEED, NW_LAG, BOOT_REPS, BOOT_BLOCK
from econometrics import log, hr, ols_nw, baron_kenny, block_bootstrap_resample, boot_ci

panel = pd.read_parquet(PROCESSED / "panel.parquet")
OUT = RESULTS / "tables"

stress_mask = panel["regime"] == "stress"

hr()
log("E5: Regime-conditional Stage II + mediation")
hr()

records = []
joint_records = []

for stable in ["USDT", "USDC"]:
    d_col = f"delta_{stable}"
    for asset in ["BTC", "ETH"]:
        b_col = f"basis_{asset}"
        log(f"\n--- delta_{stable} -> basis_{asset} ---")

        for regime_name, mask in [("STRESS", stress_mask), ("NORMAL", ~stress_mask), ("FULL", pd.Series(True, index=panel.index))]:
            sub = panel.loc[mask]
            if len(sub) < 200:
                continue
            res = ols_nw(sub[b_col], sub[[d_col]], lag=NW_LAG)
            gamma = float(res.params[d_col])
            se = float(res.bse[d_col])
            log(f"  {regime_name:6s} (n={len(sub)}): gamma={gamma:+.4f}  SE={se:.4f}  t={gamma/se:+.2f}  R2={res.rsquared:.4f}")
            records.append({
                "stablecoin": stable, "asset": asset, "regime": regime_name,
                "gamma": gamma, "se": se, "t": gamma/se,
                "r2": float(res.rsquared), "n": int(res.nobs),
            })

        df = panel[[b_col, d_col]].copy()
        df["stress"] = stress_mask.astype(float)
        df["delta_x_stress"] = df[d_col] * df["stress"]
        df = df.dropna()

        res_int = ols_nw(df[b_col], df[[d_col, "stress", "delta_x_stress"]], lag=NW_LAG)
        beta_main = float(res_int.params[d_col])
        beta_int = float(res_int.params["delta_x_stress"])
        se_int = float(res_int.bse["delta_x_stress"])
        log(f"  INTERACTION test:  gamma_normal={beta_main:+.4f}  gamma_stress-normal={beta_int:+.4f}  "
            f"SE={se_int:.4f}  t={beta_int/se_int:+.2f}  p={res_int.pvalues['delta_x_stress']:.3e}")

        gamma_stress = beta_main + beta_int
        gamma_normal = beta_main
        amp_ratio = gamma_stress / gamma_normal if gamma_normal != 0 else np.nan
        log(f"  Implied: gamma_stress={gamma_stress:+.4f}  gamma_normal={gamma_normal:+.4f}  "
            f"amp_ratio={amp_ratio:.2f}x")

        joint_records.append({
            "stablecoin": stable, "asset": asset,
            "gamma_main": beta_main, "gamma_interaction": beta_int,
            "se_interaction": se_int, "t_interaction": beta_int/se_int,
            "p_interaction": float(res_int.pvalues["delta_x_stress"]),
            "gamma_stress": gamma_stress, "gamma_normal": gamma_normal,
            "amp_ratio": amp_ratio,
        })

pd.DataFrame(records).to_parquet(OUT / "e5_regime_gamma.parquet", index=False)
pd.DataFrame(joint_records).to_parquet(OUT / "e5_regime_interaction.parquet", index=False)

hr()
log("\nE5b: Per-regime mediation (Baron-Kenny)")
hr()

bk_records = []
for stable in ["USDT", "USDC"]:
    d_col = f"delta_{stable}"
    for asset, venue in ASSET_VENUE:
        b_col = f"basis_{asset}"
        e_col = f"eta_{asset}_{venue}_ann"

        log(f"\n--- {stable} -> basis_{asset} -> eta_{asset}_{venue} ---")
        for regime_name, mask in [("STRESS", stress_mask), ("NORMAL", ~stress_mask)]:
            sub = panel.loc[mask]
            if len(sub) < 200:
                continue
            try:
                bk = baron_kenny(sub[d_col], sub[b_col], sub[e_col], lag=NW_LAG)
                log(f"  {regime_name:6s} (n={len(sub)}):  c={bk['c']:+.4f} c'={bk['c_prime']:+.4f} "
                    f"a={bk['a']:+.6f}  b={bk['b']:+.4f}  prop_med={bk['prop_mediated']:+.4f}  "
                    f"Sobel z={bk['sobel_z']:+.2f}")
                bk_records.append({"stablecoin": stable, "asset": asset, "venue": venue,
                                   "regime": regime_name, **bk})
            except Exception as ex:
                log(f"  {regime_name}: ERROR {ex}")

pd.DataFrame(bk_records).to_parquet(OUT / "e5_regime_mediation.parquet", index=False)
log("\nE5 done.")
