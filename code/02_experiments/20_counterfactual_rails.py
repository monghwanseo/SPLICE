import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from settings import PROCESSED, RESULTS

T = RESULTS / "tables"
panel = pd.read_parquet(PROCESSED / "panel.parquet")
extra = pd.read_parquet(PROCESSED / "panel_extra.parquet")
e1 = pd.read_parquet(T / "e1_first_stage.parquet")
e2 = pd.read_parquet(T / "e2_second_stage.parquet")
kappa = pd.read_parquet(T / "kappa_per_asset.parquet")

ASSETS = ["BTC", "ETH", "SOL", "XRP", "BNB", "DOGE", "ADA", "LTC"]
VENUES = ["binance", "bybit"]

print("=" * 80)
print("MS_E2: Counterfactual — USDT under banking-rails (USDC-like) reserve")
print("=" * 80)

delta_T = panel["delta_USDT"]
delta_C = panel["delta_USDC"]
regime = panel["regime"]

sigma_T = float(delta_T.std())
sigma_C = float(delta_C.std())
mean_T = float(delta_T.mean())
mean_C = float(delta_C.mean())
print(f"\n  σ(δ_USDT) = {sigma_T*1e4:.2f} bp  ;  σ(δ_USDC) = {sigma_C*1e4:.2f} bp")
print(f"  scale ratio  σ_USDC / σ_USDT  =  {sigma_C/sigma_T:.3f}")

mask_s = (regime == "stress")
sigma_T_s = float(delta_T[mask_s].std())
sigma_C_s = float(delta_C[mask_s].std())
print(f"  Stress σ(δ_USDT)  = {sigma_T_s*1e4:.2f} bp ;  stress σ(δ_USDC) = {sigma_C_s*1e4:.2f} bp")

delta_T_cf = mean_T + (delta_T - mean_T) * (sigma_C / sigma_T)
delta_C_cf = mean_C + (delta_C - mean_C) * (sigma_T / sigma_C)

def get_basis(a):
    return panel[f"basis_{a}"] if a in ("BTC", "ETH") else extra[f"basis_{a}"]

def get_eta(a, v):
    col = f"eta_{a}_{v}_ann"
    src = panel if a in ("BTC", "ETH") else extra
    return src[col] if col in src.columns else None

records = []
for a in ASSETS:
    if a in e1["asset"].values:
        row = e1[(e1["spec"] == "M3_joint") & (e1["asset"] == a)]
        if len(row) > 0:
            g_T = float(row["delta_USDT_coef"].iloc[0])
            g_C = float(row["delta_USDC_coef"].iloc[0])
        else:
            g_T = g_C = np.nan
    else:
        X = get_basis(a)
        df = pd.DataFrame({"X": X, "dT": delta_T, "dC": delta_C}).dropna()
        if len(df) > 1000:
            import statsmodels.api as sm
            res = sm.OLS(df["X"], sm.add_constant(df[["dT", "dC"]])).fit()
            g_T = float(res.params["dT"]); g_C = float(res.params["dC"])
        else:
            g_T = g_C = np.nan

    X_a = get_basis(a)
    for v in VENUES:
        eta_av = get_eta(a, v)
        if eta_av is None:
            continue

        lam_row = e2[(e2["asset"] == a) & (e2["venue"] == v)]
        if len(lam_row) > 0:
            lam = float(lam_row["lambda"].iloc[0])
        else:
            import statsmodels.api as sm
            dfL = pd.DataFrame({"y": eta_av, "x": X_a}).dropna()
            res = sm.OLS(dfL["y"], sm.add_constant(dfL["x"])).fit()
            lam = -float(res.params["x"])

        df_all = pd.DataFrame({"X": X_a, "dT": delta_T, "dC": delta_C,
                                "dT_cf": delta_T_cf, "dC_cf": delta_C_cf,
                                "eta": eta_av, "regime": regime}).dropna()
        xi = df_all["X"] - g_T * df_all["dT"] - g_C * df_all["dC"]
        eps_eta = df_all["eta"] - (-lam * df_all["X"])

        X_cf_T = g_T * df_all["dT_cf"] + g_C * df_all["dC"] + xi
        eta_cf_T = -lam * X_cf_T + eps_eta

        X_cf_C = g_T * df_all["dT"] + g_C * df_all["dC_cf"] + xi
        eta_cf_C = -lam * X_cf_C + eps_eta

        def stats(s):
            return float(s.std()), float(np.quantile(np.abs(s.dropna()), 0.99))

        sd_X_act, q_X_act = stats(df_all["X"])
        sd_X_cfT, q_X_cfT = stats(X_cf_T)
        sd_X_cfC, q_X_cfC = stats(X_cf_C)

        sd_e_act, q_e_act = stats(df_all["eta"])
        sd_e_cfT, q_e_cfT = stats(eta_cf_T)
        sd_e_cfC, q_e_cfC = stats(eta_cf_C)

        m_stress = df_all["regime"] == "stress"
        sd_X_act_s = float(df_all.loc[m_stress, "X"].std())
        sd_X_cfT_s = float(X_cf_T[m_stress].std())
        sd_e_act_s = float(df_all.loc[m_stress, "eta"].std())
        sd_e_cfT_s = float(eta_cf_T[m_stress].std())

        records.append({
            "asset": a, "venue": v,
            "gamma_T": g_T, "gamma_C": g_C, "lambda": lam,
            "sigma_X_actual": sd_X_act,
            "sigma_X_cf_USDT_to_C": sd_X_cfT,
            "sigma_X_cf_USDC_to_T": sd_X_cfC,
            "delta_pct_X_USDT_rails": 100 * (sd_X_cfT - sd_X_act) / sd_X_act,
            "delta_pct_X_USDC_rails": 100 * (sd_X_cfC - sd_X_act) / sd_X_act,
            "sigma_eta_actual": sd_e_act,
            "sigma_eta_cf_USDT_to_C": sd_e_cfT,
            "sigma_eta_cf_USDC_to_T": sd_e_cfC,
            "delta_pct_eta_USDT_rails": 100 * (sd_e_cfT - sd_e_act) / sd_e_act,
            "delta_pct_eta_USDC_rails": 100 * (sd_e_cfC - sd_e_act) / sd_e_act,
            "q99_eta_actual": q_e_act,
            "q99_eta_cf_USDT_to_C": q_e_cfT,
            "q99_eta_cf_USDC_to_T": q_e_cfC,
            "stress_sigma_eta_actual": sd_e_act_s,
            "stress_sigma_eta_cf_USDT_to_C": sd_e_cfT_s,
            "stress_delta_pct": 100 * (sd_e_cfT_s - sd_e_act_s) / sd_e_act_s,
            "n_stress": int(m_stress.sum()),
            "n": int(len(df_all)),
        })

out = pd.DataFrame(records)
out.to_csv(T / "ms_e2_counterfactual_rails.csv", index=False)
print(f"\n→ ms_e2_counterfactual_rails.csv ({len(out)} cells)")

agg = pd.DataFrame([{
    "metric": "sigma_X reduction (USDT to USDC rails) %  [mean]",
    "value": float(out["delta_pct_X_USDT_rails"].mean()),
    "median": float(out["delta_pct_X_USDT_rails"].median()),
    "min": float(out["delta_pct_X_USDT_rails"].min()),
    "max": float(out["delta_pct_X_USDT_rails"].max()),
}, {
    "metric": "sigma_eta reduction (USDT to USDC rails) %  [mean]",
    "value": float(out["delta_pct_eta_USDT_rails"].mean()),
    "median": float(out["delta_pct_eta_USDT_rails"].median()),
    "min": float(out["delta_pct_eta_USDT_rails"].min()),
    "max": float(out["delta_pct_eta_USDT_rails"].max()),
}, {
    "metric": "sigma_eta CHANGE (USDC to USDT rails) %  [mean]  (welfare cost)",
    "value": float(out["delta_pct_eta_USDC_rails"].mean()),
    "median": float(out["delta_pct_eta_USDC_rails"].median()),
    "min": float(out["delta_pct_eta_USDC_rails"].min()),
    "max": float(out["delta_pct_eta_USDC_rails"].max()),
}, {
    "metric": "Stress-regime sigma_eta reduction (USDT to USDC rails) %  [mean]",
    "value": float(out["stress_delta_pct"].mean()),
    "median": float(out["stress_delta_pct"].median()),
    "min": float(out["stress_delta_pct"].min()),
    "max": float(out["stress_delta_pct"].max()),
}, {
    "metric": "99 percentile |eta| reduction (USDT to USDC rails) %  [mean]",
    "value": float((100 * (out["q99_eta_cf_USDT_to_C"] - out["q99_eta_actual"]) / out["q99_eta_actual"]).mean()),
    "median": float((100 * (out["q99_eta_cf_USDT_to_C"] - out["q99_eta_actual"]) / out["q99_eta_actual"]).median()),
    "min": float((100 * (out["q99_eta_cf_USDT_to_C"] - out["q99_eta_actual"]) / out["q99_eta_actual"]).min()),
    "max": float((100 * (out["q99_eta_cf_USDT_to_C"] - out["q99_eta_actual"]) / out["q99_eta_actual"]).max()),
}])
agg.to_csv(T / "ms_e2_counterfactual_summary.csv", index=False)
print(f"→ ms_e2_counterfactual_summary.csv")

print("\n" + "=" * 80)
print("AGGREGATE COUNTERFACTUAL EFFECTS  (16 cells)")
print("=" * 80)
print(agg.to_string(index=False))

print("\n" + "=" * 80)
print("PER-CELL RESULTS")
print("=" * 80)
display_cols = ["asset", "venue", "gamma_T", "lambda",
                "delta_pct_X_USDT_rails", "delta_pct_eta_USDT_rails",
                "stress_delta_pct"]
print(out[display_cols].to_string(index=False))
