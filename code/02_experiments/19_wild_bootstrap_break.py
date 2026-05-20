import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).parent.parent))
from settings import PROCESSED, RESULTS, SEED, BOOT_BLOCK
from econometrics import log, hr

panel = pd.read_parquet(PROCESSED / "panel.parquet")
OUT = RESULTS / "tables"

hr()
log("E47: Wild bootstrap Quandt-Andrews break test")
hr()

def chow_F_at(t, y, X):
    n = len(y)
    k = X.shape[1]
    if t < k + 2 or n - t < k + 2:
        return np.nan
    full = sm.OLS(y, X).fit()
    rss_full = ((full.resid) ** 2).sum()
    res1 = sm.OLS(y[:t], X[:t]).fit()
    res2 = sm.OLS(y[t:], X[t:]).fit()
    rss_split = ((res1.resid) ** 2).sum() + ((res2.resid) ** 2).sum()
    return ((rss_full - rss_split) / k) / (rss_split / (n - 2 * k))

def supF_test(y, X, trim=0.15, sub_step=None):
    n = len(y)
    t_lo = int(n * trim)
    t_hi = int(n * (1 - trim))
    if sub_step is None:
        sub_step = max(1, (t_hi - t_lo) // 200)
    best_F = -np.inf
    best_t = -1
    for t in range(t_lo, t_hi, sub_step):
        F = chow_F_at(t, y, X)
        if not np.isnan(F) and F > best_F:
            best_F = F
            best_t = t
    return best_F, best_t

def wild_bootstrap_QA(y, X, B=200, trim=0.15, seed=2026):
    n = len(y)
    full = sm.OLS(y, X).fit()
    resid = full.resid
    fitted = full.fittedvalues
    rng = np.random.default_rng(seed)
    boot_supF = np.empty(B)
    for b in range(B):
        w = rng.choice([-1.0, 1.0], size=n)
        y_boot = fitted + w * resid
        F_b, _ = supF_test(y_boot, X, trim=trim)
        boot_supF[b] = F_b
    return boot_supF

records = []
for stable in ["USDT", "USDC"]:
    d_col = f"delta_{stable}"
    for asset in ["BTC", "ETH"]:
        b_col = f"basis_{asset}"
        log(f"\n--- {stable} -> {b_col} ---")

        df = panel[[b_col, d_col]].dropna().reset_index(drop=True)
        y = df[b_col].to_numpy()
        X = sm.add_constant(df[d_col].to_numpy().reshape(-1, 1))

        sup_F, best_t = supF_test(y, X)
        log(f"  Observed sup-F = {sup_F:.2f} at t={best_t}")

        log(f"  Running wild bootstrap (B=200, seed={SEED})...")
        boot_F = wild_bootstrap_QA(y, X, B=200, seed=SEED)
        boot_F = boot_F[~np.isnan(boot_F)]
        crit_90 = float(np.quantile(boot_F, 0.90))
        crit_95 = float(np.quantile(boot_F, 0.95))
        crit_99 = float(np.quantile(boot_F, 0.99))
        p_value = float(np.mean(boot_F >= sup_F))
        log(f"  Bootstrap CV: 90%={crit_90:.2f}  95%={crit_95:.2f}  99%={crit_99:.2f}")
        log(f"  Bootstrap p-value = {p_value:.4f}  -> {'REJECT H_0' if p_value < 0.05 else 'fail to reject'}")

        records.append({
            "stable": stable, "asset": asset,
            "sup_F_observed": sup_F, "best_t": int(best_t),
            "boot_crit_90": crit_90, "boot_crit_95": crit_95, "boot_crit_99": crit_99,
            "boot_p_value": p_value,
            "B": len(boot_F),
        })

pd.DataFrame(records).to_parquet(OUT / "e47_wild_bootstrap_qa.parquet", index=False)
log("\nE47 done.")
