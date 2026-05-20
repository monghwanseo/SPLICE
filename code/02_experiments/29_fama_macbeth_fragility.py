import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as ss
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).parent.parent))
from settings import PROCESSED, RESULTS, NW_LAG, SEED

rng = np.random.default_rng(SEED)
T = RESULTS / "tables"
panel = pd.read_parquet(PROCESSED / "panel.parquet")
extra = pd.read_parquet(PROCESSED / "panel_extra.parquet")

ASSETS = ["BTC", "ETH", "SOL", "XRP", "BNB", "DOGE", "ADA", "LTC"]
VENUES = ["binance", "bybit"]
LABELS = ["const", "lambda_M", "lambda_T", "lambda_C"]

print("=" * 80)
print("MS_E1 robustness:  monthly-consistent FM + bootstrap + Shanken + SDF-GMM")
print("=" * 80)

funding = {}
for a in ASSETS:
    src = panel if a in ("BTC", "ETH") else extra
    for v in VENUES:
        c = f"eta_{a}_{v}_ann"
        if c in src.columns:
            funding[(a, v)] = src[c]
test = {f"FU_{a}_{v}": s for (a, v), s in funding.items()}
R_panel_h = pd.DataFrame(test)

log_BTC = np.log(panel["spot_BTC"]).diff()
log_ETH = np.log(panel["spot_ETH"]).diff()
F_h = pd.concat([
    (0.5 * (log_BTC + log_ETH)).rename("f_M"),
    panel["delta_USDT"].diff().rename("f_T"),
    panel["delta_USDC"].diff().rename("f_C"),
], axis=1)

R_m = R_panel_h.resample("ME").mean()
F_m = pd.concat([F_h["f_M"].resample("ME").sum(),
                 F_h["f_T"].resample("ME").last() - F_h["f_T"].resample("ME").first().shift(0),
                 F_h["f_C"].resample("ME").last() - F_h["f_C"].resample("ME").first().shift(0)],
                axis=1)
delta_T_m_end = panel["delta_USDT"].resample("ME").last()
delta_C_m_end = panel["delta_USDC"].resample("ME").last()
F_m["f_T"] = delta_T_m_end.diff()
F_m["f_C"] = delta_C_m_end.diff()

mat = pd.concat([R_m, F_m], axis=1).dropna()
R_mat = mat[R_panel_h.columns].values
F_mat = mat[F_m.columns].values
T_m, N = R_mat.shape
K = F_mat.shape[1]
print(f"\nMonthly panel: T={T_m} × N={N} × K={K}")

B = np.zeros((N, K)); A = np.zeros(N); RES = np.zeros((T_m, N))
for i in range(N):
    X = sm.add_constant(F_mat)
    res = sm.OLS(R_mat[:, i], X).fit()
    A[i] = res.params[0]; B[i, :] = res.params[1:]
    RES[:, i] = res.resid

g_ts = np.zeros((T_m, K + 1)); r2 = np.zeros(T_m)
for t in range(T_m):
    res = sm.OLS(R_mat[t, :], sm.add_constant(B)).fit()
    g_ts[t, :] = res.params; r2[t] = res.rsquared
lam_hat = g_ts.mean(axis=0)
nw_se = np.zeros(K + 1); nw_t = np.zeros(K + 1); nw_p = np.zeros(K + 1)
for j in range(K + 1):
    s = g_ts[:, j]
    res = sm.OLS(s, np.ones(T_m)).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
    nw_se[j] = float(res.bse[0]); nw_t[j] = lam_hat[j] / nw_se[j]
    nw_p[j] = 2 * (1 - ss.t.cdf(abs(nw_t[j]), T_m - 1))

print("\n(A) Monthly-consistent FM (Newey-West SE, lag=4 months):")
for j, lab in enumerate(LABELS):
    print(f"  {lab:10s}: lambda={lam_hat[j]:+.5e}  NW_SE={nw_se[j]:.5e}  "
          f"t={nw_t[j]:+.2f}  p={nw_p[j]:.4f}")

reps = 2000; block_len = 4
n_blocks = int(np.ceil(T_m / block_len))
boot_lam = np.zeros((reps, K + 1))
for b in range(reps):
    starts = rng.integers(0, T_m - block_len + 1, size=n_blocks)
    idx = np.concatenate([np.arange(s, s + block_len) for s in starts])[:T_m]
    boot_lam[b, :] = g_ts[idx, :].mean(axis=0)
boot_se = boot_lam.std(axis=0, ddof=1)
boot_t = lam_hat / boot_se
boot_p = 2 * (1 - ss.norm.cdf(np.abs(boot_t)))
ci_lo = np.quantile(boot_lam, 0.025, axis=0); ci_hi = np.quantile(boot_lam, 0.975, axis=0)

print("\n(B) Block-bootstrap (block_len=4, reps=2000):")
for j, lab in enumerate(LABELS):
    print(f"  {lab:10s}: lambda={lam_hat[j]:+.5e}  boot_SE={boot_se[j]:.5e}  "
          f"t={boot_t[j]:+.2f}  p={boot_p[j]:.4f}  CI=[{ci_lo[j]:+.5e},{ci_hi[j]:+.5e}]")

Sigma_f = np.cov(F_mat.T, ddof=1)
try:
    Sigma_f_inv = np.linalg.pinv(Sigma_f)
    lam_k = lam_hat[1:]
    shanken_correction = float(1.0 + lam_k @ Sigma_f_inv @ lam_k)
except Exception:
    shanken_correction = np.nan

shanken_se = nw_se * np.sqrt(shanken_correction)
shanken_t = lam_hat / shanken_se
shanken_p = 2 * (1 - ss.t.cdf(np.abs(shanken_t), T_m - 1))

print(f"\n(C) Shanken EIV correction factor = {shanken_correction:.4f}")
for j, lab in enumerate(LABELS):
    print(f"  {lab:10s}: lambda={lam_hat[j]:+.5e}  Shanken_SE={shanken_se[j]:.5e}  "
          f"t={shanken_t[j]:+.2f}  p={shanken_p[j]:.4f}")

f_demeaned = F_mat - F_mat.mean(axis=0)
R_centered = R_mat

def sdf_moments(theta, R_mat, f_demeaned):
    b = theta[:K]; mu = theta[K]
    T_, N_ = R_mat.shape
    m_t = 1.0 - f_demeaned @ b
    moments = (m_t[:, None] * R_mat) - mu
    return moments

def sdf_obj(theta, R_mat, f_demeaned, W):
    M = sdf_moments(theta, R_mat, f_demeaned)
    g_bar = M.mean(axis=0)
    return float(g_bar @ W @ g_bar)

def nw_S(M, lag):
    T_, q = M.shape
    Mc = M - M.mean(axis=0)
    S = (Mc.T @ Mc) / T_
    for l in range(1, lag + 1):
        w = 1 - l / (lag + 1)
        G = (Mc[l:].T @ Mc[:-l]) / T_
        S += w * (G + G.T)
    return S

theta0 = np.zeros(K + 1); theta0[K] = R_mat.mean()
W1 = np.eye(N)
res1 = minimize(sdf_obj, theta0, args=(R_mat, f_demeaned, W1),
                method="L-BFGS-B", options={"maxiter": 2000, "ftol": 1e-12})
theta1 = res1.x

M1 = sdf_moments(theta1, R_mat, f_demeaned)
S1 = nw_S(M1, lag=4)
try:
    W2 = np.linalg.pinv(S1)
except np.linalg.LinAlgError:
    W2 = np.eye(N)

res2 = minimize(sdf_obj, theta1, args=(R_mat, f_demeaned, W2),
                method="L-BFGS-B", options={"maxiter": 5000, "ftol": 1e-14})
theta_g = res2.x
b_g = theta_g[:K]; mu_g = theta_g[K]

def g_bar(theta):
    return sdf_moments(theta, R_mat, f_demeaned).mean(axis=0)

eps = 1e-7
P = len(theta_g)
G = np.zeros((N, P))
for p_ in range(P):
    tp = theta_g.copy(); tp[p_] += eps
    tm = theta_g.copy(); tm[p_] -= eps
    G[:, p_] = (g_bar(tp) - g_bar(tm)) / (2 * eps)
M2 = sdf_moments(theta_g, R_mat, f_demeaned)
S2 = nw_S(M2, lag=4)
try:
    GWG = G.T @ W2 @ G
    Var_b = np.linalg.pinv(GWG) @ G.T @ W2 @ S2 @ W2 @ G @ np.linalg.pinv(GWG) / T_m
    se_b = np.sqrt(np.maximum(np.diag(Var_b), 0))
except Exception:
    se_b = np.full(P, np.nan)

lam_gmm = Sigma_f @ b_g
try:
    Var_lam = Sigma_f @ Var_b[:K, :K] @ Sigma_f
    se_lam = np.sqrt(np.maximum(np.diag(Var_lam), 0))
except Exception:
    se_lam = np.full(K, np.nan)
t_lam = lam_gmm / se_lam
p_lam = 2 * (1 - ss.norm.cdf(np.abs(t_lam)))

J = float(T_m * res2.fun)
df_J = N - len(theta_g)
J_p = float(1 - ss.chi2.cdf(J, df_J))

print(f"\n(D) SDF-GMM (linear SDF, efficient):")
print(f"  b coefficients:")
for j, lab in enumerate(["b_M", "b_T", "b_C"]):
    print(f"    {lab}: b={b_g[j]:+.4e}  HAC_SE={se_b[j]:.4e}")
print(f"  mu (zero-beta) = {mu_g:+.6f}")
print(f"  Implied lambda (lambda = Sigma_f * b):")
for j, lab in enumerate(["lambda_M", "lambda_T", "lambda_C"]):
    print(f"    {lab}: lambda={lam_gmm[j]:+.5e}  delta-SE={se_lam[j]:.5e}  "
          f"t={t_lam[j]:+.2f}  p={p_lam[j]:.4f}")
print(f"  Hansen J = {J:.2f}  df = {df_J}  p = {J_p:.4f}")

loadings = pd.read_csv(T / "ms_e1_fm_loadings.csv")
load_eta = loadings[loadings["panel"] == "eta_level"].set_index("portfolio")
common = R_m.columns.intersection(load_eta.index)
B_hourly = load_eta.loc[common, ["beta_M", "beta_T", "beta_C"]].values
R_m_sub = R_m[common].values
T_x, N_x = R_m_sub.shape
g_x = np.zeros((T_x, K + 1))
for t in range(T_x):
    mask = np.isfinite(R_m_sub[t, :])
    if mask.sum() < 6:
        g_x[t, :] = np.nan; continue
    res = sm.OLS(R_m_sub[t, mask], sm.add_constant(B_hourly[mask])).fit()
    g_x[t, :] = res.params
mask_valid = ~np.isnan(g_x).any(axis=1)
g_x_v = g_x[mask_valid]
lam_x = g_x_v.mean(axis=0)
nw_se_x = np.zeros(K + 1); nw_t_x = np.zeros(K + 1); nw_p_x = np.zeros(K + 1)
for j in range(K + 1):
    s = g_x_v[:, j]
    res = sm.OLS(s, np.ones(len(s))).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
    nw_se_x[j] = float(res.bse[0]); nw_t_x[j] = lam_x[j] / nw_se_x[j]
    nw_p_x[j] = 2 * (1 - ss.t.cdf(abs(nw_t_x[j]), len(s) - 1))

print("\n(X) Replication: hourly Pass 1, monthly Pass 2 (the 126 spec):")
for j, lab in enumerate(LABELS):
    print(f"  {lab:10s}: lambda={lam_x[j]:+.5e}  NW_SE={nw_se_x[j]:.5e}  "
          f"t={nw_t_x[j]:+.2f}  p={nw_p_x[j]:.4f}")

rows = []
for j, lab in enumerate(LABELS):
    rows.append({"method": "A_FM_monthly_NW", "coef": lab,
                 "lambda_hat": lam_hat[j], "se": nw_se[j],
                 "t_stat": nw_t[j], "p_value": nw_p[j], "ci_lo": np.nan, "ci_hi": np.nan})
for j, lab in enumerate(LABELS):
    rows.append({"method": "B_FM_monthly_block_boot", "coef": lab,
                 "lambda_hat": lam_hat[j], "se": boot_se[j],
                 "t_stat": boot_t[j], "p_value": boot_p[j], "ci_lo": ci_lo[j], "ci_hi": ci_hi[j]})
for j, lab in enumerate(LABELS):
    rows.append({"method": "C_FM_monthly_Shanken", "coef": lab,
                 "lambda_hat": lam_hat[j], "se": shanken_se[j],
                 "t_stat": shanken_t[j], "p_value": shanken_p[j], "ci_lo": np.nan, "ci_hi": np.nan})
for j, lab in enumerate(["lambda_M", "lambda_T", "lambda_C"]):
    rows.append({"method": "D_SDF_GMM_efficient", "coef": lab,
                 "lambda_hat": lam_gmm[j], "se": se_lam[j],
                 "t_stat": t_lam[j], "p_value": p_lam[j], "ci_lo": np.nan, "ci_hi": np.nan})
rows.append({"method": "D_SDF_GMM_efficient", "coef": "Hansen_J",
             "lambda_hat": np.nan, "se": np.nan,
             "t_stat": J, "p_value": J_p, "ci_lo": np.nan, "ci_hi": np.nan})
for j, lab in enumerate(LABELS):
    rows.append({"method": "X_orig_hourly_beta_monthly_FM", "coef": lab,
                 "lambda_hat": lam_x[j], "se": nw_se_x[j],
                 "t_stat": nw_t_x[j], "p_value": nw_p_x[j], "ci_lo": np.nan, "ci_hi": np.nan})

out = pd.DataFrame(rows)
out.to_csv(T / "ms_e1_fm_robustness.csv", index=False)
print(f"\n→ ms_e1_fm_robustness.csv ({len(out)} rows)")

print("\n" + "=" * 80)
print("SUMMARY: lambda_T across specifications")
print("=" * 80)
for m in out["method"].unique():
    sub = out[(out["method"] == m) & (out["coef"] == "lambda_T")]
    if len(sub) > 0:
        r = sub.iloc[0]
        print(f"  {m:35s}: lambda={r['lambda_hat']:+.4e}  t={r['t_stat']:+.2f}  p={r['p_value']:.4f}")
