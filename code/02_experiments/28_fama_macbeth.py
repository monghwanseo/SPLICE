import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as ss

sys.path.insert(0, str(Path(__file__).parent.parent))
from settings import PROCESSED, RESULTS, NW_LAG

T = RESULTS / "tables"
panel = pd.read_parquet(PROCESSED / "panel.parquet")
extra = pd.read_parquet(PROCESSED / "panel_extra.parquet")

ASSETS = ["BTC", "ETH", "SOL", "XRP", "BNB", "DOGE", "ADA", "LTC"]
VENUES = ["binance", "bybit"]

print("=" * 80)
print("MS_E1: Fama-MacBeth cross-sectional pricing test")
print("       (stablecoin-augmented factor model, n_test = 16/32 cells)")
print("=" * 80)

basis = {}
funding = {}
for a in ASSETS:
    src = panel if a in ("BTC", "ETH") else extra
    basis[a] = src[f"basis_{a}"]
    for v in VENUES:
        col = f"eta_{a}_{v}_ann"
        if col in src.columns:
            funding[(a, v)] = src[col]

dbasis = {a: s.diff() for a, s in basis.items()}

test_eta = {}
for a in ASSETS:
    for v in VENUES:
        if (a, v) in funding:
            test_eta[f"FU_{a}_{v}"] = funding[(a, v)]

test_db = {}
for a in ASSETS:
    for v in VENUES:
        test_db[f"DB_{a}_{v}"] = dbasis[a]

test_stack = {}
for k, s in test_eta.items():
    s0 = (s - s.mean()) / s.std()
    test_stack[k] = s0
for k, s in test_db.items():
    s0 = (s - s.mean()) / s.std()
    test_stack[k] = s0

panels_to_run = [
    ("eta_level", test_eta),
    ("dbasis", test_db),
    ("stacked_32", test_stack),
]

log_BTC = np.log(panel["spot_BTC"]).diff()
log_ETH = np.log(panel["spot_ETH"]).diff()
f_M = 0.5 * (log_BTC + log_ETH)
f_T = panel["delta_USDT"].diff()
f_C = panel["delta_USDC"].diff()
factors = pd.concat([f_M.rename("f_M"), f_T.rename("f_T"), f_C.rename("f_C")], axis=1)

def pass1(series_dict):
    rows = []
    for name, r in series_dict.items():
        df = pd.concat([r.rename("r"), factors], axis=1).dropna()
        if len(df) < 1000:
            continue
        X = sm.add_constant(df[["f_M", "f_T", "f_C"]])
        res = sm.OLS(df["r"], X).fit(cov_type="HAC", cov_kwds={"maxlags": NW_LAG})
        rows.append({
            "portfolio": name,
            "outcome": "funding" if name.startswith("FU_") else "dbasis",
            "asset": name.split("_")[1],
            "venue": name.split("_")[2],
            "alpha": float(res.params["const"]),
            "beta_M": float(res.params["f_M"]),
            "beta_T": float(res.params["f_T"]),
            "beta_C": float(res.params["f_C"]),
            "t_alpha": float(res.tvalues["const"]),
            "t_beta_M": float(res.tvalues["f_M"]),
            "t_beta_T": float(res.tvalues["f_T"]),
            "t_beta_C": float(res.tvalues["f_C"]),
            "r2": float(res.rsquared),
            "n": int(res.nobs),
        })
    return pd.DataFrame(rows)

def fm_stat(series, nw_lag=4):
    s = series.dropna().astype(float)
    n = len(s)
    if n < 5:
        return np.nan, np.nan, np.nan, np.nan, n
    mean = s.mean()
    res = sm.OLS(s.values, np.ones(n)).fit(cov_type="HAC", cov_kwds={"maxlags": nw_lag})
    se = float(res.bse[0])
    t_ = mean / se if se > 0 else np.nan
    p_ = 2 * (1 - ss.t.cdf(abs(t_), n - 1)) if np.isfinite(t_) else np.nan
    return float(mean), float(se), float(t_), float(p_), int(n)

def fm_panel(loadings, series_dict, label):
    R_monthly = pd.DataFrame({k: v.resample("ME").mean() for k, v in series_dict.items()})
    betas = loadings.set_index("portfolio")[["beta_M", "beta_T", "beta_C"]]
    common = R_monthly.columns.intersection(betas.index)
    R_monthly = R_monthly[common]
    betas = betas.loc[common]

    g_M, g_T, g_C, g_0, r2s = [], [], [], [], []
    for m, row in R_monthly.iterrows():
        y = row.values.astype(float)
        valid = np.isfinite(y)
        if valid.sum() < min(6, len(y)):
            continue
        X = sm.add_constant(betas.values)
        res = sm.OLS(y[valid], X[valid]).fit()
        g_0.append(res.params[0]); g_M.append(res.params[1])
        g_T.append(res.params[2]); g_C.append(res.params[3])
        r2s.append(res.rsquared)

    rows = []
    for coef_name, vals in [("const", g_0), ("g_M", g_M), ("g_T", g_T), ("g_C", g_C)]:
        m_, se_, t_, p_, n_ = fm_stat(pd.Series(vals), nw_lag=4)
        rows.append({"panel": label, "coef": coef_name,
                     "lambda_hat": m_, "nw_se": se_, "t_stat": t_, "p_value": p_,
                     "n_months": n_,
                     "avg_cs_r2": float(np.mean(r2s)) if r2s else np.nan})

    grs_F, grs_p = np.nan, np.nan
    try:
        resid = {}
        for name in common:
            r = series_dict[name]
            df = pd.concat([r.rename("r"), factors], axis=1).dropna()
            X_ = sm.add_constant(df[["f_M", "f_T", "f_C"]])
            res = sm.OLS(df["r"], X_).fit()
            resid[name] = res.resid
        resid_df = pd.DataFrame(resid).dropna()
        Sigma = resid_df.cov().values
        f_mu = factors.dropna().mean().values
        Omega = factors.dropna().cov().values
        T_obs = len(resid_df)
        N_test = len(common); K = 3
        alpha_vec = loadings.set_index("portfolio").loc[resid_df.columns, "alpha"].values
        Sigma_inv = np.linalg.pinv(Sigma); Omega_inv = np.linalg.pinv(Omega)
        quad_a = float(alpha_vec @ Sigma_inv @ alpha_vec)
        quad_f = float(f_mu @ Omega_inv @ f_mu)
        grs_F = ((T_obs - N_test - K) / N_test) * (quad_a / (1.0 + quad_f))
        grs_p = float(1.0 - ss.f.cdf(grs_F, N_test, T_obs - N_test - K))
    except Exception:
        pass
    rows.append({"panel": label, "coef": "GRS_F",
                 "lambda_hat": np.nan, "nw_se": np.nan,
                 "t_stat": float(grs_F), "p_value": float(grs_p),
                 "n_months": len(R_monthly), "avg_cs_r2": np.nan})
    return rows

all_loadings = []
all_rows = []
all_avg_ret = []
for label, sdict in panels_to_run:
    print(f"\n--- Panel: {label} ({len(sdict)} test assets) ---")
    L = pass1(sdict)
    L["panel"] = label
    all_loadings.append(L)
    for name, s in sdict.items():
        s0 = s.dropna()
        all_avg_ret.append({"panel": label, "portfolio": name,
                            "avg": float(s0.mean()), "std": float(s0.std()), "n": int(len(s0))})
    rows = fm_panel(L, sdict, label)
    for r in rows:
        print(f"  {r['coef']:8s}: lambda_hat = {r['lambda_hat']:+.6f}  "
              f"NW_SE = {r['nw_se']:.6f}  t = {r['t_stat']:+.3f}  p = {r['p_value']:.4f}")
    all_rows.extend(rows)

loadings_df = pd.concat(all_loadings, ignore_index=True)
loadings_df.to_csv(T / "ms_e1_fm_loadings.csv", index=False)
pd.DataFrame(all_avg_ret).to_csv(T / "ms_e1_fm_avg_returns.csv", index=False)
pd.DataFrame(all_rows).to_csv(T / "ms_e1_fm_pricing.csv", index=False)

print(f"\n→ ms_e1_fm_loadings.csv ({len(loadings_df)} rows)")
print(f"→ ms_e1_fm_avg_returns.csv")
print(f"→ ms_e1_fm_pricing.csv ({len(all_rows)} rows)")

print("\n" + "=" * 80)
print("HEADLINE: Is δ_USDT priced in the cross-section?")
print("=" * 80)
summary_df = pd.DataFrame(all_rows)
for panel_label in ["eta_level", "dbasis", "stacked_32"]:
    sub = summary_df[(summary_df["panel"] == panel_label) & (summary_df["coef"].isin(["g_M", "g_T", "g_C"]))]
    print(f"  [{panel_label}]")
    for _, row in sub.iterrows():
        star = "***" if row["p_value"] < 0.01 else ("**" if row["p_value"] < 0.05 else ("*" if row["p_value"] < 0.10 else ""))
        print(f"    {row['coef']}: lambda = {row['lambda_hat']:+.5e}  t = {row['t_stat']:+.2f}  p = {row['p_value']:.3f}  {star}")
