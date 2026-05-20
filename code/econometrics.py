import numpy as np
import pandas as pd
import statsmodels.api as sm

from settings import SEED, NW_LAG, BOOT_BLOCK, BOOT_REPS

def ols_nw(y: pd.Series, X: pd.DataFrame, lag: int = NW_LAG, add_const: bool = True):
    Xc = sm.add_constant(X) if add_const else X
    df = pd.concat([y, Xc], axis=1).dropna()
    yy = df.iloc[:, 0]
    XX = df.iloc[:, 1:]
    res = sm.OLS(yy, XX).fit(cov_type="HAC", cov_kwds={"maxlags": lag})
    return res

def block_bootstrap_resample(n: int, block_len: int, rng: np.random.Generator) -> np.ndarray:
    n_blocks = (n + block_len - 1) // block_len
    starts = rng.integers(0, n - block_len + 1, size=n_blocks)
    out = np.concatenate([np.arange(s, s + block_len) for s in starts])[:n]
    return out

def block_bootstrap_stat(y: np.ndarray, X: np.ndarray, fn, B: int = BOOT_REPS,
                         block_len: int = BOOT_BLOCK, seed: int = SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(y)
    out = np.empty(B)
    for b in range(B):
        idx = block_bootstrap_resample(n, block_len, rng)
        out[b] = fn(y[idx], X[idx])
    return out

def boot_ci(arr: np.ndarray, alpha: float = 0.05) -> tuple:
    return float(np.quantile(arr, alpha / 2)), float(np.quantile(arr, 1 - alpha / 2))

def baron_kenny(delta: pd.Series, basis: pd.Series, eta: pd.Series, lag: int = NW_LAG):
    df = pd.concat([delta.rename("delta"), basis.rename("basis"), eta.rename("eta")], axis=1).dropna()
    d, b_, e = df["delta"], df["basis"], df["eta"]

    r1 = ols_nw(e, d.rename("delta").to_frame(), lag=lag)
    c = float(r1.params["delta"])
    sc = float(r1.bse["delta"])

    r2 = ols_nw(b_, d.rename("delta").to_frame(), lag=lag)
    a = float(r2.params["delta"])
    sa = float(r2.bse["delta"])

    X3 = pd.concat([d.rename("delta"), b_.rename("basis")], axis=1)
    r3 = ols_nw(e, X3, lag=lag)
    c_prime = float(r3.params["delta"])
    b_coef = float(r3.params["basis"])
    sb = float(r3.bse["basis"])

    indirect = a * b_coef
    se_indirect = np.sqrt(b_coef ** 2 * sa ** 2 + a ** 2 * sb ** 2)
    sobel_z = indirect / se_indirect if se_indirect > 0 else np.nan
    prop_med = indirect / c if c != 0 else np.nan

    return dict(
        n=int(len(df)), c=c, sc=sc, c_prime=c_prime, a=a, sa=sa,
        b=b_coef, sb=sb, indirect=indirect, se_indirect=float(se_indirect),
        sobel_z=float(sobel_z), prop_mediated=float(prop_med),
        r1_r2=float(r1.rsquared), r3_r2=float(r3.rsquared),
    )

def baron_kenny_bootstrap_ci(delta: pd.Series, basis: pd.Series, eta: pd.Series,
                              B: int = BOOT_REPS, block_len: int = BOOT_BLOCK,
                              seed: int = SEED) -> dict:
    df = pd.concat([delta.rename("delta"), basis.rename("basis"), eta.rename("eta")], axis=1).dropna()
    d_arr = df["delta"].to_numpy()
    b_arr = df["basis"].to_numpy()
    e_arr = df["eta"].to_numpy()
    n = len(df)

    rng = np.random.default_rng(seed)
    out = np.empty((B, 4))

    for k in range(B):
        idx = block_bootstrap_resample(n, block_len, rng)
        d_b = d_arr[idx]; b_b = b_arr[idx]; e_b = e_arr[idx]
        Xd = np.column_stack([np.ones(n), d_b])
        beta1, *_ = np.linalg.lstsq(Xd, e_b, rcond=None)
        c = beta1[1]
        beta2, *_ = np.linalg.lstsq(Xd, b_b, rcond=None)
        a = beta2[1]
        Xdb = np.column_stack([np.ones(n), d_b, b_b])
        beta3, *_ = np.linalg.lstsq(Xdb, e_b, rcond=None)
        c_prime = beta3[1]
        b_coef = beta3[2]
        indirect = a * b_coef
        prop_med = indirect / c if c != 0 else np.nan
        out[k] = [c, c_prime, indirect, prop_med]

    return dict(
        c_ci=boot_ci(out[:, 0]),
        c_prime_ci=boot_ci(out[:, 1]),
        indirect_ci=boot_ci(out[:, 2]),
        prop_mediated_ci=boot_ci(out[:, 3]),
        bootstrap_samples=out,
    )

def load_panel(path):
    return pd.read_parquet(path)

def log(m): print(m, flush=True)

def hr(n: int = 78): log("=" * n)
