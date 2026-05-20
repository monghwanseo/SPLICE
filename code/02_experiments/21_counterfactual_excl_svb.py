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

ASSETS = ["BTC", "ETH", "SOL", "XRP", "BNB", "DOGE", "ADA", "LTC"]
VENUES = ["binance", "bybit"]

SVB_TS = pd.Timestamp("2023-03-10 22:00", tz="UTC")

print("=" * 80)
print("MS_E2 robustness:  USDT-rails counterfactual EXCLUDING USDC SVB depeg")
print("=" * 80)

def make_mask(window_days=None):
    if window_days is None:
        return pd.Series(True, index=panel.index)
    lo = SVB_TS - pd.Timedelta(days=window_days)
    hi = SVB_TS + pd.Timedelta(days=window_days)
    return ~((panel.index >= lo) & (panel.index <= hi))

def get_basis(a):
    return panel[f"basis_{a}"] if a in ("BTC", "ETH") else extra[f"basis_{a}"]

def get_eta(a, v):
    c = f"eta_{a}_{v}_ann"
    src = panel if a in ("BTC", "ETH") else extra
    return src[c] if c in src.columns else None

def run_counterfactual(mask, label):
    delta_T = panel.loc[mask, "delta_USDT"]
    delta_C = panel.loc[mask, "delta_USDC"]
    regime = panel.loc[mask, "regime"]

    sigma_T = float(delta_T.std()); sigma_C = float(delta_C.std())
    mean_T = float(delta_T.mean()); mean_C = float(delta_C.mean())
    delta_T_cf = mean_T + (delta_T - mean_T) * (sigma_C / sigma_T)
    delta_C_cf = mean_C + (delta_C - mean_C) * (sigma_T / sigma_C)

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
            X = get_basis(a).loc[mask]
            df = pd.DataFrame({"X": X, "dT": delta_T, "dC": delta_C}).dropna()
            if len(df) > 1000:
                import statsmodels.api as sm
                res = sm.OLS(df["X"], sm.add_constant(df[["dT", "dC"]])).fit()
                g_T = float(res.params["dT"]); g_C = float(res.params["dC"])
            else:
                g_T = g_C = np.nan

        X_a = get_basis(a).loc[mask]
        for v in VENUES:
            eta_av = get_eta(a, v)
            if eta_av is None: continue
            eta_av = eta_av.loc[mask]
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

            sd_X_act = float(df_all["X"].std()); sd_X_cf = float(X_cf_T.std())
            sd_e_act = float(df_all["eta"].std()); sd_e_cf = float(eta_cf_T.std())
            m_s = df_all["regime"] == "stress"
            sd_e_act_s = float(df_all.loc[m_s, "eta"].std()) if m_s.sum() > 0 else np.nan
            sd_e_cf_s = float(eta_cf_T[m_s].std()) if m_s.sum() > 0 else np.nan
            q99_act = float(np.quantile(np.abs(df_all["eta"].dropna()), 0.99))
            q99_cf = float(np.quantile(np.abs(eta_cf_T.dropna()), 0.99))

            records.append({
                "subsample": label,
                "asset": a, "venue": v,
                "n": int(len(df_all)),
                "sigma_delta_USDT_bp": sigma_T * 1e4,
                "sigma_delta_USDC_bp": sigma_C * 1e4,
                "scale_ratio": sigma_C / sigma_T,
                "gamma_T": g_T, "gamma_C": g_C, "lambda": lam,
                "sigma_X_actual": sd_X_act,
                "sigma_X_cf": sd_X_cf,
                "delta_pct_X": 100 * (sd_X_cf - sd_X_act) / sd_X_act,
                "sigma_eta_actual": sd_e_act,
                "sigma_eta_cf": sd_e_cf,
                "delta_pct_eta": 100 * (sd_e_cf - sd_e_act) / sd_e_act,
                "stress_delta_pct": 100 * (sd_e_cf_s - sd_e_act_s) / sd_e_act_s if sd_e_act_s and sd_e_act_s > 0 else np.nan,
                "q99_eta_actual": q99_act, "q99_eta_cf": q99_cf,
                "q99_delta_pct": 100 * (q99_cf - q99_act) / q99_act,
            })
    return pd.DataFrame(records), sigma_T, sigma_C

frames = []
print()
for window, label in [(None, "full_sample"),
                       (7, "exclude_SVB_pm7d"),
                       (30, "exclude_SVB_pm30d")]:
    mask = make_mask(window)
    df, sT, sC = run_counterfactual(mask, label)
    n_obs = int(mask.sum())
    print(f"  {label}: n={n_obs}, sigma_delta_T={sT*1e4:.2f}bp, sigma_delta_C={sC*1e4:.2f}bp, "
          f"ratio C/T={sC/sT:.3f}, mean dpct_eta={df['delta_pct_eta'].mean():+.2f}%")
    frames.append(df)

all_df = pd.concat(frames, ignore_index=True)
all_df.to_csv(T / "ms_e2_counterfactual_no_svb.csv", index=False)
print(f"\n→ ms_e2_counterfactual_no_svb.csv ({len(all_df)} rows)")

agg_rows = []
for label in ["full_sample", "exclude_SVB_pm7d", "exclude_SVB_pm30d"]:
    sub = all_df[all_df["subsample"] == label]
    agg_rows.append({
        "subsample": label,
        "sigma_T_bp": sub["sigma_delta_USDT_bp"].iloc[0],
        "sigma_C_bp": sub["sigma_delta_USDC_bp"].iloc[0],
        "scale_ratio": sub["scale_ratio"].iloc[0],
        "mean_dpct_X": sub["delta_pct_X"].mean(),
        "mean_dpct_eta": sub["delta_pct_eta"].mean(),
        "mean_stress_dpct": sub["stress_delta_pct"].mean(),
        "mean_q99_dpct": sub["q99_delta_pct"].mean(),
        "median_dpct_eta": sub["delta_pct_eta"].median(),
        "min_dpct_eta": sub["delta_pct_eta"].min(),
        "max_dpct_eta": sub["delta_pct_eta"].max(),
    })
agg = pd.DataFrame(agg_rows)
agg.to_csv(T / "ms_e2_counterfactual_no_svb_summary.csv", index=False)
print(f"→ ms_e2_counterfactual_no_svb_summary.csv")

print("\n" + "=" * 80)
print("SUMMARY: USDT-rails counterfactual under three subsamples")
print("=" * 80)
print(agg.to_string(index=False))

print("\n" + "=" * 80)
print("INTERPRETATION")
print("=" * 80)
sub0 = agg[agg["subsample"] == "full_sample"].iloc[0]
sub7 = agg[agg["subsample"] == "exclude_SVB_pm7d"].iloc[0]
sub30 = agg[agg["subsample"] == "exclude_SVB_pm30d"].iloc[0]
print(f"  Full sample: scale ratio C/T = {sub0['scale_ratio']:.3f}, mean dpct_eta = {sub0['mean_dpct_eta']:+.2f}%")
print(f"  +/- 7d SVB excluded: scale ratio = {sub7['scale_ratio']:.3f}, mean dpct_eta = {sub7['mean_dpct_eta']:+.2f}%")
print(f"  +/- 30d SVB excluded: scale ratio = {sub30['scale_ratio']:.3f}, mean dpct_eta = {sub30['mean_dpct_eta']:+.2f}%")
flip = (sub0["mean_dpct_eta"] * sub30["mean_dpct_eta"]) < 0
print(f"\n  Conclusion REVERSES when SVB excluded?  {flip}")
