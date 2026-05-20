import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from settings import PROCESSED, RESULTS, ASSET_VENUE, SEED, NW_LAG, BOOT_REPS, BOOT_BLOCK
from econometrics import log, hr, baron_kenny, baron_kenny_bootstrap_ci

panel = pd.read_parquet(PROCESSED / "panel.parquet")
OUT = RESULTS / "tables"

hr()
log("E4: Mediation Baron-Kenny + Sobel  (mediator = perp basis)")
hr()
log(f"  Bootstrap config: B={BOOT_REPS}, block={BOOT_BLOCK}h, seed={SEED}")

records = []
boot_records = []

for stable in ["USDT", "USDC"]:
    d_col = f"delta_{stable}"
    for asset, venue in ASSET_VENUE:
        b_col = f"basis_{asset}"
        e_col = f"eta_{asset}_{venue}_ann"

        log(f"\n--- {stable} -> basis_{asset} -> eta_{asset}_{venue} ---")

        bk = baron_kenny(panel[d_col], panel[b_col], panel[e_col], lag=NW_LAG)
        log(f"  c (total)     = {bk['c']:+.4f}  SE={bk['sc']:.4f}  t={bk['c']/bk['sc']:+.2f}")
        log(f"  c'(direct)    = {bk['c_prime']:+.4f}")
        log(f"  a (d->b)      = {bk['a']:+.6f}  SE={bk['sa']:.6f}  t={bk['a']/bk['sa']:+.2f}")
        log(f"  b (b->e|d)    = {bk['b']:+.4f}  SE={bk['sb']:.4f}  t={bk['b']/bk['sb']:+.2f}")
        log(f"  indirect=a*b  = {bk['indirect']:+.4f}  SE={bk['se_indirect']:.4f}  Sobel z={bk['sobel_z']:+.2f}")
        log(f"  prop_mediated = {bk['prop_mediated']:+.4f}")

        boot = baron_kenny_bootstrap_ci(panel[d_col], panel[b_col], panel[e_col],
                                         B=BOOT_REPS, block_len=BOOT_BLOCK, seed=SEED)
        log(f"  c       95% CI [{boot['c_ci'][0]:+.4f}, {boot['c_ci'][1]:+.4f}]")
        log(f"  c'      95% CI [{boot['c_prime_ci'][0]:+.4f}, {boot['c_prime_ci'][1]:+.4f}]")
        log(f"  indir   95% CI [{boot['indirect_ci'][0]:+.4f}, {boot['indirect_ci'][1]:+.4f}]")
        log(f"  prop    95% CI [{boot['prop_mediated_ci'][0]:+.4f}, {boot['prop_mediated_ci'][1]:+.4f}]")

        rec = {
            "stablecoin": stable, "asset": asset, "venue": venue,
            **{k: v for k, v in bk.items() if not isinstance(v, dict)},
            "c_lo95": boot["c_ci"][0], "c_hi95": boot["c_ci"][1],
            "c_prime_lo95": boot["c_prime_ci"][0], "c_prime_hi95": boot["c_prime_ci"][1],
            "indirect_lo95": boot["indirect_ci"][0], "indirect_hi95": boot["indirect_ci"][1],
            "prop_med_lo95": boot["prop_mediated_ci"][0], "prop_med_hi95": boot["prop_mediated_ci"][1],
        }
        records.append(rec)

pd.DataFrame(records).to_parquet(OUT / "e4_mediation.parquet", index=False)
log("\nE4 done.")
