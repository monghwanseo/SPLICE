import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from settings import RESULTS, SEED

T = RESULTS / "tables"
NAIVE_P = 0.006
N_SIMS = 500_000


def li_ji_effective_M(rho_within, M=5):
    R = np.full((M, M), rho_within)
    np.fill_diagonal(R, 1.0)
    eigvals = np.linalg.eigvalsh(R)
    eigvals = np.clip(np.round(eigvals, 10), 0.0, None)
    return float(sum(1.0 * (lam >= 1.0) + (lam - np.floor(lam)) for lam in eigvals))


def sidak_adjusted_p(p_raw, M_eff):
    return float(1.0 - (1.0 - p_raw) ** M_eff)


def build_correlation_matrix(rho_within, rho_across, M=5):
    n = 2 * M
    R = np.eye(n)
    for i in range(M):
        for j in range(M):
            if i != j:
                R[i, j] = rho_within
                R[i + M, j + M] = rho_within
    for i in range(M):
        R[i, i + M] = rho_across
        R[i + M, i] = rho_across
    for i in range(M):
        for j in range(M):
            if i != j:
                R[i, j + M] = rho_within * rho_across
                R[j + M, i] = rho_within * rho_across
    return R


def mvn_joint_extreme_p(rho_within, rho_across, n_sims=N_SIMS, seed=SEED, M=5):
    R = build_correlation_matrix(rho_within, rho_across, M)
    eigvals, eigvecs = np.linalg.eigh(R)
    eigvals = np.clip(eigvals, 1e-12, None)
    L = eigvecs @ np.diag(np.sqrt(eigvals))
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n_sims, 2 * M)) @ L.T
    usdt_match = (Z[:, :M] >= 0).sum(axis=1)
    usdc_flip = (Z[:, M:] < 0).sum(axis=1)
    joint = (usdt_match >= 4) & (usdc_flip >= 5)
    return float(joint.mean())


records = [
    {"method": "joint_sign_randomisation_naive",
     "assumption": "proxy_independence",
     "p_value": NAIVE_P},
]

for rho in [0.00, 0.50, 0.85, 0.95]:
    M_eff = li_ji_effective_M(rho, M=5)
    p_adj = sidak_adjusted_p(NAIVE_P, M_eff)
    records.append({
        "method": "sidak_li_ji",
        "assumption": f"rho_within={rho:.2f}",
        "M_eff": M_eff,
        "p_value": p_adj,
    })

mvn_configs = [
    (0.30, 0.50),
    (0.50, 0.50),
    (0.70, 0.50),
    (0.85, 0.50),
    (0.95, 0.50),
]
for rho_w, rho_a in mvn_configs:
    p = mvn_joint_extreme_p(rho_w, rho_a)
    records.append({
        "method": "mvn_joint_extreme",
        "assumption": f"rho_within={rho_w:.2f}, rho_across={rho_a:.2f}",
        "p_value": p,
    })

df = pd.DataFrame(records)
print(df.to_string(index=False))

df.to_parquet(T / "dependence_robust_pvalues.parquet", index=False)
print(f"\nSaved {T / 'dependence_robust_pvalues.parquet'}")
