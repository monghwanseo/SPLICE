import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from settings import RESULTS, ROOT

OUT = ROOT / "paper" / "tables"
OUT.mkdir(parents=True, exist_ok=True)
T = RESULTS / "tables"

for f in OUT.glob("*.tex"):
    f.unlink()
    print(f"  removed legacy {f.name}")

def save_csv(df, name):
    path = OUT / name
    df.to_csv(path, index=False, float_format="%.6g")
    print(f"  ✓ {name}  ({len(df)} rows × {df.shape[1]} cols)")
    return path

def T1_baselines():
    e1 = pd.read_parquet(T / "e1_first_stage.parquet")
    e2 = pd.read_parquet(T / "e2_second_stage.parquet")
    e3 = pd.read_parquet(T / "e3_reduced_form.parquet")

    panels = []

    m3 = e1[e1["spec"] == "M3_joint"].copy()
    for _, r in m3.iterrows():
        panels.append({
            "panel": "A: Stage II (M3 joint)",
            "row": f"basis_{r['asset']}",
            "stat": "gamma",
            "USDT_value": r["delta_USDT_coef"],
            "USDT_se": r["delta_USDT_se"],
            "USDT_t": r["delta_USDT_t"],
            "USDC_value": r["delta_USDC_coef"],
            "USDC_se": r["delta_USDC_se"],
            "USDC_t": r["delta_USDC_t"],
            "r2": r["r2"],
            "n": int(r["n"]),
        })

    for _, r in e2.sort_values(["asset", "venue"]).iterrows():
        panels.append({
            "panel": "B: Stage I (eta on -X)",
            "row": f"{r['asset']}/{r['venue']}",
            "stat": "lambda",
            "USDT_value": r["lambda"],
            "USDT_se": r["se"],
            "USDT_t": r["t"],
            "USDC_value": np.nan,
            "USDC_se": np.nan,
            "USDC_t": np.nan,
            "r2": r["r2"],
            "n": int(r["n"]),
        })

    for _, r in e3.sort_values(["asset", "venue", "stablecoin"]).iterrows():
        panels.append({
            "panel": "C: Reduced form (eta on delta)",
            "row": f"{r['stablecoin']}/{r['asset']}/{r['venue']}",
            "stat": "c_delta",
            "USDT_value": r["c"] if r["stablecoin"] == "USDT" else np.nan,
            "USDT_se": r["se"] if r["stablecoin"] == "USDT" else np.nan,
            "USDT_t": r["t"] if r["stablecoin"] == "USDT" else np.nan,
            "USDC_value": r["c"] if r["stablecoin"] == "USDC" else np.nan,
            "USDC_se": r["se"] if r["stablecoin"] == "USDC" else np.nan,
            "USDC_t": r["t"] if r["stablecoin"] == "USDC" else np.nan,
            "r2": r["r2"],
            "n": int(r["n"]),
            "implied_lambda_gamma": r["implied_lambda_gamma"],
            "c_lo95": r.get("c_lo95"),
            "c_hi95": r.get("c_hi95"),
        })

    df = pd.DataFrame(panels)
    return save_csv(df, "T1_baselines.csv")

def T2_identification():
    e21 = pd.read_parquet(T / "e21_rigobon.parquet")
    e13 = pd.read_parquet(T / "e13_iv_gmm.parquet")
    fix1 = pd.read_parquet(T / "fix_1_iv_with_L.parquet")

    rows = []
    methods = [
        ("OLS_full",     "gamma_OLS",      e13),
        ("OLS_with_L",   "ols_with_L",     fix1),
        ("IV_5events",   "gamma_IV",       e13),
        ("IV_with_L",    "iv_with_L",      fix1),
        ("Rigobon",      "gamma_Rigobon",  e21),
    ]
    for asset in ["BTC", "ETH"]:
        for stable in ["USDT", "USDC"]:
            row = {"asset": asset, "stable": stable}
            for method, col, src in methods:
                sub = src[(src["stable"] == stable) & (src["asset"] == asset)]
                if len(sub):
                    row[f"{method}_gamma"] = float(sub[col].iloc[0])
                else:
                    row[f"{method}_gamma"] = np.nan
                if method == "IV_5events" and "first_stage_F" in sub.columns:
                    row["first_stage_F_basic"] = float(sub["first_stage_F"].iloc[0])
                if method == "IV_with_L" and "first_stage_F_with_L" in sub.columns:
                    row["first_stage_F_with_L"] = float(sub["first_stage_F_with_L"].iloc[0])
            rows.append(row)

    df = pd.DataFrame(rows)
    return save_csv(df, "T2_identification.csv")

def T3_mediation():
    df = pd.read_parquet(T / "e4_mediation.parquet")
    df = df.sort_values(["stablecoin", "asset", "venue"]).reset_index(drop=True)
    cols = ["stablecoin", "asset", "venue", "n",
            "c", "sc", "c_prime", "a", "b", "indirect", "se_indirect",
            "sobel_z", "prop_mediated",
            "c_lo95", "c_hi95", "prop_med_lo95", "prop_med_hi95"]
    df_out = df[[c for c in cols if c in df.columns]].copy()
    return save_csv(df_out, "T3_mediation.csv")

def T4_cross_asset():
    e10 = pd.read_parquet(T / "e10_cross_asset.parquet").set_index("asset")
    e1 = pd.read_parquet(T / "e1_first_stage.parquet")
    fix5 = pd.read_parquet(T / "fix_5_joint_panel_boot.parquet").set_index("asset")
    kappa = pd.read_parquet(T / "kappa_per_asset.parquet").set_index("asset")
    assets_order = ["BTC", "ETH", "BNB", "SOL", "XRP", "DOGE", "ADA", "LTC"]

    m3 = e1[e1["spec"] == "M3_joint"].set_index("asset")

    rows = []
    for a in assets_order:
        r = {"asset": a}
        if a in e10.index:
            e = e10.loc[a]
            r["gamma_USDT"] = float(e["gamma_USDT"])
            r["t_USDT"]     = float(e["t_USDT"])
            r["gamma_USDC"] = float(e["gamma_USDC"])
            r["t_USDC"]     = float(e["t_USDC"])
        elif a in m3.index:
            e = m3.loc[a]
            r["gamma_USDT"] = float(e["delta_USDT_coef"])
            r["t_USDT"]     = float(e["delta_USDT_t"])
            r["gamma_USDC"] = float(e["delta_USDC_coef"])
            r["t_USDC"]     = float(e["delta_USDC_t"])
        else:
            continue
        if a in fix5.index:
            f = fix5.loc[a]
            r["fix5_gamma_median"] = float(f["gamma_median"])
            r["fix5_ci_lo"] = float(f["ci_lo"])
            r["fix5_ci_hi"] = float(f["ci_hi"])
        if a in kappa.index:
            k = kappa.loc[a]
            r["rho_per_hour"] = float(k["rho_per_hour"])
            r["lambda_per_hour"] = float(k["lambda_per_hour"])
            r["kappa_per_hour"] = float(k["kappa_per_hour"])
            r["kappa_apy_pct"] = float(k["kappa_apy_pct"])
        rows.append(r)

    if "POOLED_8" in fix5.index:
        p = fix5.loc["POOLED_8"]
        rows.append({
            "asset": "POOLED_8",
            "fix5_gamma_median": float(p["gamma_median"]),
            "fix5_ci_lo": float(p["ci_lo"]),
            "fix5_ci_hi": float(p["ci_hi"]),
        })

    return save_csv(pd.DataFrame(rows), "T4_cross_asset.csv")

def T5_sdf_asymmetry():
    df = pd.read_parquet(T / "theorem4_multi_sdf_v2.parquet").copy()
    RETURN_BASED = {"S1_LogUtil_BTCETH", "S3_BTC_only", "S4_RiskReversal", "S9_ETH_only", "S10_PC1"}
    df["family"] = df["sdf_proxy"].apply(
        lambda s: "return_based" if s in RETURN_BASED else "vol_funding_based"
    )
    df = df[["family", "sdf_proxy", "stable", "beta_M", "se", "t",
             "gamma_direct", "sign_match", "n"]].sort_values(
        ["family", "stable", "sdf_proxy"]).reset_index(drop=True)
    save_csv(df, "T5_sdf_asymmetry.csv")

    summary = df.groupby(["family", "stable"]).agg(
        n_total=("sign_match", "size"),
        n_match=("sign_match", lambda s: (s == "MATCH").sum()),
    ).reset_index()
    summary["match_pct"] = 100 * summary["n_match"] / summary["n_total"]
    return save_csv(summary, "T5_sdf_asymmetry_summary.csv")

def T6_oos():
    df = pd.read_parquet(T / "e9_oos_forecast.parquet")
    horizons = [1, 4, 8, 24]

    rows = []
    for (a, v, s), grp in df.groupby(["asset", "venue", "stable"]):
        for h in horizons:
            sub = grp[grp["h"] == h]
            if not len(sub):
                continue
            ar1 = sub[sub["model"] == "AR1"]["oos_r2"]
            spl = sub[sub["model"] == "SPLICE"]["oos_r2"]
            vs = sub[sub["model"] == "SPLICE_vs_AR1"]
            row = {
                "asset": a, "venue": v, "stable": s, "h": h,
                "r2_AR1":     float(ar1.iloc[0]) if len(ar1) else np.nan,
                "r2_SPLICE":  float(spl.iloc[0]) if len(spl) else np.nan,
                "delta_r2":   float(spl.iloc[0]) - float(ar1.iloc[0])
                              if len(ar1) and len(spl) else np.nan,
                "DM_t":       float(vs["DM_stat"].iloc[0]) if len(vs) else np.nan,
                "DM_p":       float(vs["DM_p"].iloc[0])    if len(vs) else np.nan,
                "CW_t":       float(vs["CW_stat"].iloc[0]) if len(vs) else np.nan,
                "CW_p":       float(vs["CW_p"].iloc[0])    if len(vs) else np.nan,
                "n_test":     int(sub["n_test"].iloc[0]),
            }
            rows.append(row)

    return save_csv(pd.DataFrame(rows), "T6_oos.csv")

def T7_break():
    e34 = pd.read_parquet(T / "e34_quandt_andrews.parquet")
    e47 = pd.read_parquet(T / "e47_wild_bootstrap_qa.parquet")
    merged = e34.merge(e47, on=["stable", "asset"], how="left",
                       suffixes=("_e34", "_e47"))
    merged["sup_F_over_99cv"] = merged["sup_F"] / merged["boot_crit_99"]
    cols = ["stable", "asset", "sup_F", "best_break_date",
            "crit_5", "crit_10", "crit_1",
            "boot_crit_90", "boot_crit_95", "boot_crit_99",
            "boot_p_value", "B", "sup_F_over_99cv"]
    out = merged[[c for c in cols if c in merged.columns]]
    return save_csv(out, "T7_break.csv")

def T8_gmm_vs_ols():
    gmm = pd.read_parquet(T / "structural_gmm.parquet")
    e1 = pd.read_parquet(T / "e1_first_stage.parquet")
    e2 = pd.read_parquet(T / "e2_second_stage.parquet")
    e3 = pd.read_parquet(T / "e3_reduced_form.parquet")

    rows = []
    for _, r in gmm.iterrows():
        a, v, s = r["asset"], r["venue"], r["stable"]
        m3 = e1[(e1["spec"] == "M3_joint") & (e1["asset"] == a)]
        gam_ols = float(m3[f"delta_{s}_coef"].iloc[0]) if len(m3) else np.nan
        l = e2[(e2["asset"] == a) & (e2["venue"] == v)]
        lam_ols_ann = float(l["lambda"].iloc[0]) if len(l) else np.nan
        crow = e3[(e3["asset"] == a) & (e3["venue"] == v) & (e3["stablecoin"] == s)]
        c_ols = float(crow["c"].iloc[0]) if len(crow) else np.nan

        rows.append({
            "asset": a, "venue": v, "stable": s, "n": int(r["n"]),
            "gamma_GMM":      r["gamma"],
            "gamma_OLS":      gam_ols,
            "lambda_GMM_ann": r["lambda_ann"],
            "lambda_OLS_ann": lam_ols_ann,
            "kappa_GMM_per_hour": r["kappa_per_hour"],
            "kappa_GMM_apy_pct":  r["kappa_apy_pct"],
            "c_GMM_ann":      r["c_rf_ann"],
            "c_OLS":          c_ols,
            "rho_per_hour":   r["rho_per_hour"],
            "moments_norm":   r["moments_norm"],
            "converged":      bool(r["converged"]),
        })

    return save_csv(pd.DataFrame(rows), "T8_gmm.csv")

def T9_kappa_decomposition():
    src = pd.read_parquet(T / "kappa_decomposition.parquet")

    rows = []
    for _, r in src.iterrows():
        stable = r["stable"]
        rows.append({
            "stable": stable,
            "implied_kappa_per_hour": r["implied_kappa"],
            "C1_lending_per_hour": r["C1_lending"],
            "C2_xvenue_per_hour":  r["C2_xvenue"],
            "C3_tail_per_hour":    r["C3_tail"],
            "C4_mm_per_hour":      r["C4_mm"],
            "C1_pct": r["pct_C1"] * 100,
            "C2_pct": r["pct_C2"] * 100,
            "C3_pct": r["pct_C3"] * 100,
            "C4_pct": r["pct_C4"] * 100,
            "L2_old_direct_only_pct":  r["L2_old_pct"] * 100,
            "conservative_total_pct":  r["pct_conservative"] * 100,
            "structural_total_pct":    r["pct_total"] * 100 if r["pct_total"] < 1.0 else float("nan"),
        })

    return save_csv(pd.DataFrame(rows), "T9_kappa_decomposition.csv")

def T10_theorem5_substitution():
    sub = pd.read_parquet(T / "theorem5_substitution.parquet")
    minor = pd.read_parquet(T / "theorem5_regime_minor.parquet")

    cols_sub = ["regime", "stable_pair", "asset",
                "gamma_own", "t_own", "gamma_cross", "t_cross",
                "sign_prediction", "sign_observed",
                "consistent_with_segmentation", "n"]
    sub_out = sub[[c for c in cols_sub if c in sub.columns]].copy()
    sub_out["section"] = "Major USDT-USDC pair (own + cross)"

    minor_out = minor.copy()
    minor_out["section"] = "Minor stables (FDUSD/DAI/TUSD), Theorem 5 friction φ"

    out_cols = ["section"] + [c for c in sub_out.columns if c != "section"]
    sub_out = sub_out[out_cols]

    minor_cols = ["section"] + [c for c in minor_out.columns if c != "section"]
    minor_out = minor_out[minor_cols]

    save_csv(sub_out, "T10a_theorem5_major_pair.csv")
    save_csv(minor_out, "T10b_theorem5_minor_stables.csv")

    combined = pd.concat([
        sub_out.assign(stable=sub_out.get("stable_pair", ""))
                .rename(columns={"asset": "asset_or_minor"}),
        minor_out.rename(columns={"minor_stable": "asset_or_minor"})
                  .assign(stable_pair=minor_out.get("minor_stable", "")),
    ], ignore_index=True, sort=False)
    return save_csv(combined.fillna(""), "T10_theorem5_substitution.csv")

def T_C11_usdc_robustness():
    detail = pd.read_parquet(T / "usdc_robustness_scenarios.parquet")
    summary = pd.read_parquet(T / "usdc_robustness_summary.parquet")
    save_csv(detail, "T_C11_usdc_robustness_detail.csv")
    return save_csv(summary, "T_C11_usdc_robustness_summary.csv")

def T_C12_dependence_robust():
    df = pd.read_parquet(T / "dependence_robust_pvalues.parquet")
    return save_csv(df, "T_C12_dependence_robust_pvalues.csv")

def main():
    print(f"Writing CSV tables to {OUT}\n")
    fns = [T1_baselines, T2_identification, T3_mediation, T4_cross_asset,
           T5_sdf_asymmetry, T6_oos, T7_break, T8_gmm_vs_ols,
           T9_kappa_decomposition, T10_theorem5_substitution,
           T_C11_usdc_robustness, T_C12_dependence_robust]
    for fn in fns:
        try:
            fn()
        except Exception as e:
            print(f"  ✗ {fn.__name__}: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()

    print(f"\nDone — tables saved to {OUT.relative_to(ROOT)}")

if __name__ == "__main__":
    main()
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import kurtosis

sys.path.insert(0, str(Path(__file__).parent))
from plot_style import COLOR, save, setup, subcaption, tighten_y_positive

setup()

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "results" / "tables"
PROC = ROOT / "data_processed"
RAW = ROOT / "data_raw"

ASSETS_8 = ["BTC", "ETH", "BNB", "SOL", "XRP", "DOGE", "ADA", "LTC"]
CELLS_4 = ["USDT-BTC", "USDT-ETH", "USDC-BTC", "USDC-ETH"]

def fig01_delta():
    panel = pd.read_parquet(PROC / "panel.parquet")
    fig, ax = plt.subplots(figsize=(11.0, 4.8), constrained_layout=False)
    fig.subplots_adjust(left=0.09, right=0.97, top=0.94, bottom=0.20)

    USDT_C = "#1f4e79"
    USDC_C = "#c2410c"
    EVENT_C = "#cc0000"

    ax.axhline(0, color=COLOR["neutral"], lw=0.9, alpha=0.7)
    ax.plot(panel.index, panel["delta_USDC"] * 1e4,
            color=USDC_C, lw=0.9, alpha=0.85, label="USDC",
            zorder=2)
    ax.plot(panel.index, panel["delta_USDT"] * 1e4,
            color=USDT_C, lw=0.9, alpha=0.95, label="USDT",
            zorder=3)

    ax.set_xlabel("Date")
    ax.set_ylabel(r"Stablecoin discount  $\delta_t = U_t - 1$  in basis points")
    ax.legend(loc="lower left", frameon=True, framealpha=0.95,
              facecolor="white", edgecolor=COLOR["grid"])

    events = [("2022-05-12", "Terra Luna"),
              ("2022-11-08", "FTX"),
              ("2023-03-10", "USDC depeg"),
              ("2024-01-10", "Bitcoin spot ETF"),
              ("2024-08-05", "Yen carry")]
    ymin, ymax = ax.get_ylim()
    headroom = (ymax - ymin) * 0.10
    ax.set_ylim(ymin, ymax + 2 * headroom)
    label_y = ymax + 1.0 * headroom
    for date, lbl in events:
        d = pd.Timestamp(date, tz="UTC")
        if d < panel.index.min() or d > panel.index.max():
            continue
        ax.axvline(d, color=EVENT_C, lw=0.9, ls=":", alpha=0.65, zorder=1)
        ax.text(d, label_y, lbl,
                rotation=0, va="center", ha="center",
                fontsize=8.5, color=EVENT_C, fontweight="bold")

    fig.text(0.5, 0.025,
             "Stablecoin Discount Time Series",
             ha="center", va="bottom", fontsize=12, fontweight="bold",
             color=COLOR["neutral"])
    save(fig, "figure1")
    plt.close(fig)

def fig02_gamma():
    e10 = pd.read_parquet(T / "e10_cross_asset.parquet")
    e13 = pd.read_parquet(T / "e13_iv_gmm.parquet")
    e21 = pd.read_parquet(T / "e21_rigobon.parquet")
    fix5 = pd.read_parquet(T / "fix_5_joint_panel_boot.parquet")

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4), constrained_layout=False)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.94, bottom=0.18, wspace=0.22)

    ax = axes[0]
    y = np.arange(len(ASSETS_8))[::-1]
    ax.axvspan(0.12, 0.42, color=COLOR["fill_pos"], alpha=0.30,
               label="Convergence band")
    ax.axvline(0, color=COLOR["neutral"], lw=0.9, ls="--", alpha=0.55)

    for i, a in enumerate(ASSETS_8):
        yi = y[i]
        row = e10[e10["asset"].astype(str).str.upper() == a]
        if len(row):
            ax.scatter([float(row["gamma_USDT"].iloc[0])], [yi - 0.30],
                       marker="o", s=36, color=COLOR["neutral"],
                       label="Ordinary least squares" if i == 0 else None, zorder=3)
        f = fix5[fix5["asset"].astype(str).str.upper() == a]
        if len(f):
            m, lo, hi = float(f["gamma_median"].iloc[0]), float(f["ci_lo"].iloc[0]), float(f["ci_hi"].iloc[0])
            ax.errorbar([m], [yi - 0.10], xerr=[[m - lo], [hi - m]],
                        fmt="s", ms=5, color=COLOR["highlight"], capsize=2.5, lw=1.3,
                        label="Joint panel bootstrap" if i == 0 else None, zorder=4)
        iv = e13[(e13["stable"] == "USDT") & (e13["asset"] == a)]
        if len(iv):
            ax.scatter([float(iv["gamma_IV"].iloc[0])], [yi + 0.10],
                       marker="^", s=42, color=COLOR["iv"],
                       label="Instrumental variable" if i == 0 else None, zorder=3)
        rig = e21[(e21["stable"] == "USDT") & (e21["asset"] == a)]
        if len(rig):
            ax.scatter([float(rig["gamma_Rigobon"].iloc[0])], [yi + 0.30],
                       marker="D", s=36, color=COLOR["rigobon"],
                       label="Rigobon heteroskedasticity" if i == 0 else None, zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels(ASSETS_8)
    ax.set_xlabel(r"Estimated transmission coefficient $\hat\gamma_\delta$")
    ax.set_xlim(-0.10, 0.85)
    ax.legend(loc="lower right", fontsize=8.5,
              framealpha=0.95, facecolor="white", edgecolor=COLOR["grid"])

    ax = axes[1]
    fix5_o = fix5[fix5["asset"].astype(str).str.upper().isin(ASSETS_8)].copy()
    fix5_o["asset_u"] = fix5_o["asset"].astype(str).str.upper()
    fix5_o = fix5_o.set_index("asset_u").reindex(ASSETS_8).reset_index()
    y = np.arange(len(ASSETS_8))[::-1]

    for i, (m, lo, hi) in enumerate(zip(fix5_o["gamma_median"], fix5_o["ci_lo"], fix5_o["ci_hi"])):
        ax.barh(y[i], hi - lo, left=lo, height=0.55,
                color=COLOR["fill_pos"], edgecolor=COLOR["usdt"], lw=1.0, alpha=0.85)
    ax.scatter(fix5_o["gamma_median"].values, y, marker="|", s=200,
               color=COLOR["usdt"], lw=2.0, zorder=3, label="Median")
    pooled = float(fix5_o["gamma_median"].median())
    ax.axvline(pooled, color=COLOR["highlight"], lw=1.4, ls="--",
               label="Cross-asset median")
    ax.axvline(0, color=COLOR["neutral"], lw=0.9, ls="--", alpha=0.55)
    ax.set_yticks(y)
    ax.set_yticklabels(ASSETS_8)
    ax.set_xlabel(r"Bootstrap distribution of $\hat\gamma_\delta$")
    ax.set_xlim(-0.05, 0.45)
    ax.legend(loc="lower right", fontsize=8.5,
              framealpha=0.95, facecolor="white", edgecolor=COLOR["grid"])

    fig.text(0.5, 0.025,
             "Identification of the Discount-Funding Transmission Coefficient",
             ha="center", va="bottom", fontsize=12, fontweight="bold",
             color=COLOR["neutral"])
    save(fig, "figure2")
    plt.close(fig)

def fig03_stress_break():
    amp = pd.read_parquet(T / "e5_regime_interaction.parquet").rename(columns={"stablecoin": "stable"})
    wb = pd.read_parquet(T / "e47_wild_bootstrap_qa.parquet")

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.2), constrained_layout=False)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.94, bottom=0.24, wspace=0.26)

    ax = axes[0]
    amp["cell"] = amp["stable"] + "-" + amp["asset"]
    amp_o = amp.set_index("cell").reindex(CELLS_4).reset_index()
    x = np.arange(len(CELLS_4))
    width = 0.36
    bars_n = ax.bar(x - width/2, amp_o["gamma_normal"], width, color=COLOR["neutral"],
                    label="Normal regime", edgecolor="white", lw=1.0)
    bars_s = ax.bar(x + width/2, amp_o["gamma_stress"], width, color=COLOR["stress"],
                    label="Stress regime", edgecolor="white", lw=1.0)
    ax.axhline(0, color=COLOR["neutral"], lw=0.7, alpha=0.5)
    for b, v in zip(bars_n, amp_o["gamma_normal"]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.3f}",
                ha="center", va="bottom", fontsize=8.5, color=COLOR["neutral"])
    for b, v in zip(bars_s, amp_o["gamma_stress"]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.3f}",
                ha="center", va="bottom", fontsize=8.5, color=COLOR["stress"])
    ax.set_xticks(x)
    ax.set_xticklabels(CELLS_4, fontsize=9.5)
    ax.set_ylabel(r"Estimated $\hat\gamma_\delta$")
    ax.set_ylim(0, 0.42)
    ax.legend(loc="upper right", framealpha=0.95, facecolor="white",
              edgecolor=COLOR["grid"])
    subcaption(ax, "a", "Stress versus normal-market subsample estimates", y=-0.12)

    ax = axes[1]
    wb["cell"] = wb["stable"] + "-" + wb["asset"]
    x = np.arange(len(wb))
    bars_f = ax.bar(x, wb["sup_F_observed"], color=COLOR["stress"],
                    edgecolor="white", lw=1.0)
    for b, v, cv99 in zip(bars_f, wb["sup_F_observed"], wb["boot_crit_99"]):
        mult = v / cv99
        ax.text(b.get_x() + b.get_width() / 2, v * 2.0,
                f"{int(round(v))}\n(×{int(round(mult))})",
                ha="center", va="bottom", fontsize=8.5,
                color=COLOR["stress"])
    ax.text(0.02, 0.97,
            r"$B = 200$ wild bootstrap" "\n"
            r"$\times N$ = ratio to 99th-percentile CV",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=8.0, color=COLOR["muted"])
    ax.set_xticks(x)
    ax.set_xticklabels(wb["cell"], fontsize=9.5)
    ax.set_ylabel(r"Quandt-Andrews $\sup F$ statistic")
    ax.set_yscale("log")
    ax.set_ylim(5, 1e6)
    subcaption(ax, "b",
               "Break point estimated at December 5, 2021 (common across all four cells)",
               y=-0.12)

    fig.text(0.5, 0.025,
             "Regime Amplification and the Structural Break",
             ha="center", va="bottom", fontsize=12, fontweight="bold",
             color=COLOR["neutral"])
    save(fig, "figure8")
    plt.close(fig)

def fig04_mediation_ik():
    ik = pd.read_parquet(T / "imai_keele_sensitivity.parquet")

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0), constrained_layout=False)
    fig.subplots_adjust(left=0.10, right=0.97, top=0.94, bottom=0.24, wspace=0.30)

    ik = ik.copy()
    ik["cell"] = ik["stable"] + " · " + ik["asset"] + " · " + ik["venue"].str.title()
    ik = ik.sort_values(["stable", "asset", "venue"]).reset_index(drop=True)
    y = np.arange(len(ik))[::-1]
    colors = [COLOR["usdt"] if r["stable"] == "USDT" else COLOR["usdc"] for _, r in ik.iterrows()]

    ax = axes[0]
    share_pct = ik["mediation_share"] * 100
    ax.scatter(share_pct, y, s=80, c=colors, edgecolors="white", linewidths=1.2, zorder=3)
    ax.axvspan(58, 93, color=COLOR["fill_pos"], alpha=0.20)
    ax.axvline(50, color=COLOR["neutral"], lw=0.7, ls=":", alpha=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(ik["cell"], fontsize=8.5)
    ax.set_xlabel("Share of total effect routed through the basis")
    ax.set_xlim(0, 110)
    subcaption(ax, "a", "Cross-cell mediation shares", y=-0.20)

    ax = axes[1]
    ax.barh(y, ik["abs_rho_star"], color=colors, edgecolor="white", lw=1.0, height=0.7)
    ax.axvline(0.30, color=COLOR["stress"], ls="--", lw=1.2, alpha=0.85)
    ax.text(1.03, 0,
            "Robustness threshold\n" r"($|\rho^*| = 0.30$)",
            color=COLOR["stress"], fontsize=8.0,
            ha="left", va="center", multialignment="left")
    ax.set_yticks(y)
    ax.set_yticklabels(ik["cell"], fontsize=8.5)
    ax.set_xlabel(r"Sensitivity bound $|\rho^*|$")
    ax.set_xlim(0, 1.30)
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    subcaption(ax, "b", "Imai-Keele unobserved-confounder bound", y=-0.20)

    fig.text(0.5, 0.025,
             "Basis Mediation and Imai-Keele Sensitivity",
             ha="center", va="bottom", fontsize=12, fontweight="bold",
             color=COLOR["neutral"])
    save(fig, "figure7")
    plt.close(fig)

def fig05_oos():
    oos = pd.read_parquet(T / "e9_oos_forecast.parquet")
    oos["cell"] = oos["stable"] + " · " + oos["asset"] + " · " + oos["venue"].str.title()
    dm = oos[oos["model"] == "SPLICE_vs_AR1"].copy()
    splice = dm

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.2), constrained_layout=False)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.94, bottom=0.22, wspace=0.30)

    ax = axes[0]
    horizons = sorted(splice["h"].unique())
    cells_for_panel = ["USDT · BTC · Binance", "USDT · BTC · Bybit",
                       "USDT · ETH · Binance", "USDT · ETH · Bybit"]
    width = 0.22
    x = np.arange(len(horizons))
    asset_color = {"BTC": COLOR["usdt"], "ETH": "#a35200"}
    for i, c in enumerate(cells_for_panel):
        sub = splice[splice["cell"] == c].set_index("h").reindex(horizons).reset_index()
        offset = (i - (len(cells_for_panel) - 1) / 2) * width
        asset = "BTC" if "BTC" in c else "ETH"
        col = asset_color[asset]
        alpha_val = 0.95 if "Binance" in c else 0.55
        bars_a = ax.bar(x + offset, sub["DM_stat"], width,
                        color=col, edgecolor="white", lw=0.6, label=c,
                        alpha=alpha_val)
        for b, v in zip(bars_a, sub["DM_stat"]):
            if pd.notna(v):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.25, f"{v:.3f}",
                        ha="center", va="bottom", fontsize=6.0,
                        color=col)
    ax.axhline(1.96, color=COLOR["stress"], ls="--", lw=1.0,
               label="5% significance")
    ax.axhline(0, color=COLOR["neutral"], lw=0.7, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(h)}h" for h in horizons])
    ax.set_xlabel("Forecast horizon")
    ax.set_ylabel(r"Diebold-Mariano $t$-statistic")
    ax.set_ylim(0, splice["DM_stat"].max() * 1.18)
    ax.legend(loc="upper left", fontsize=7.5, ncol=2,
              framealpha=0.95, facecolor="white", edgecolor=COLOR["grid"])
    subcaption(ax, "a", "Predictive accuracy gain over the autoregressive benchmark", y=-0.16)

    ax = axes[1]
    splice24 = oos[(oos["model"] == "SPLICE") & (oos["h"] == 24)][["cell", "oos_r2"]].rename(
        columns={"oos_r2": "r2_splice"})
    ar24 = oos[(oos["model"] == "AR1") & (oos["h"] == 24)][["cell", "oos_r2"]].rename(
        columns={"oos_r2": "r2_ar"})
    merged = splice24.merge(ar24, on="cell", how="left")
    merged["delta_r2"] = merged["r2_splice"] - merged["r2_ar"]
    merged = merged.sort_values("delta_r2", ascending=False).reset_index(drop=True)
    cell_colors = [COLOR["usdt"] if "BTC" in c else "#a35200"
                   for c in merged["cell"]]
    cell_alphas = [0.95 if "Binance" in c else 0.55 for c in merged["cell"]]
    bars_b = ax.bar(np.arange(len(merged)), merged["delta_r2"] * 100,
                    color=cell_colors, edgecolor="white", lw=1.0,
                    width=0.85)
    for bar, a in zip(bars_b, cell_alphas):
        bar.set_alpha(a)
    for b, v, col in zip(bars_b, merged["delta_r2"] * 100, cell_colors):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.15, f"{v:.3f}",
                ha="center", va="bottom", fontsize=8.5,
                color=col, fontweight="bold")
    ax.set_xticks(np.arange(len(merged)))
    ax.set_xticklabels(merged["cell"], rotation=0, fontsize=7.5)
    ax.set_ylabel(r"Out-of-sample $R^2$ improvement  (% points)")
    ax.axhline(0, color=COLOR["neutral"], lw=0.8, alpha=0.5)
    ax.set_ylim(0, max(merged["delta_r2"] * 100) * 1.15)
    subcaption(ax, "b", "Improvement at the 24-hour horizon", y=-0.16)

    fig.text(0.5, 0.025,
             "Out-of-Sample Forecast Performance",
             ha="center", va="bottom", fontsize=12, fontweight="bold",
             color=COLOR["neutral"])
    save(fig, "figure9")
    plt.close(fig)

def fig06_kappa_tail():
    kpa = pd.read_parquet(T / "kappa_per_asset.parquet").sort_values("kappa_per_hour").reset_index(drop=True)
    panel = pd.read_parquet(PROC / "panel.parquet")

    def spot_close(a):
        if f"spot_{a}" in panel.columns:
            return panel[f"spot_{a}"].dropna()
        p = RAW / "binance" / f"{a}USDT_spot_1h.parquet"
        df = pd.read_parquet(p)
        df["ts"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
        return df.sort_values("ts").set_index("ts")["close"]

    kpa["kurt"] = [float(kurtosis(np.log(spot_close(a)).diff().dropna().values, fisher=True))
                   for a in kpa["asset"]]

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0), constrained_layout=False)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.94, bottom=0.24, wspace=0.28)

    ax = axes[0]
    x = np.arange(len(kpa))
    colors = [COLOR["usdt"] if t > 1.96 else COLOR["highlight"] if t > 1.0 else COLOR["muted"]
              for t in kpa["kappa_t"]]
    ax.bar(x, kpa["kappa_per_hour"], color=colors, edgecolor="white", lw=1.0,
           yerr=1.96 * kpa["kappa_se_delta_method"], capsize=3, error_kw=dict(lw=1.0))
    ax.set_xticks(x)
    ax.set_xticklabels(kpa["asset"], fontsize=10)
    ax.set_ylabel(r"Margin sensitivity $\hat\kappa_a$  (per hour)")
    tighten_y_positive(ax)
    subcaption(ax, "a", "Recovered margin sensitivity by asset", y=-0.16)

    ax = axes[1]
    log_kappa = np.log(kpa["kappa_per_hour"])
    ax.scatter(kpa["kurt"], log_kappa, s=72, color=COLOR["usdt"],
               edgecolors="white", linewidths=1.2, zorder=3)
    label_offsets = {
        "SOL":  (8, 5),
        "BNB":  (8, -14),
        "XRP":  (8, -16),
        "BTC":  (8, 7),
        "ETH":  (-26, -12),
        "DOGE": (-32, 6),
        "ADA":  (8, 6),
        "LTC":  (8, -14),
    }
    for _, row in kpa.iterrows():
        dx, dy = label_offsets.get(row["asset"], (8, 5))
        ax.annotate(row["asset"], (row["kurt"], np.log(row["kappa_per_hour"])),
                    xytext=(dx, dy), textcoords="offset points", fontsize=9,
                    color=COLOR["neutral"])
    res = sm.OLS(log_kappa.values, sm.add_constant(kpa["kurt"].values)).fit()
    xs = np.linspace(kpa["kurt"].min() - 5, kpa["kurt"].max() + 5, 50)
    ax.plot(xs, res.params[0] + res.params[1] * xs, color=COLOR["stress"],
            lw=1.6, alpha=0.85, zorder=2, label="Cross-section fit")
    ax.set_xlabel(r"Excess kurtosis of hourly log returns")
    ax.set_ylabel(r"$\log\hat\kappa_a$")
    ax.legend(loc="upper left", fontsize=9.0)
    subcaption(ax, "b", "Kurtosis explains most of the cross-sectional spread", y=-0.16)

    fig.text(0.5, 0.025,
             "Margin Sensitivity and Return Tail Fatness",
             ha="center", va="bottom", fontsize=12, fontweight="bold",
             color=COLOR["neutral"])
    save(fig, "figure10")
    plt.close(fig)

def fig07_sdf():
    multi = pd.read_parquet(T / "theorem4_multi_sdf_v2.parquet")
    paired = pd.read_parquet(T / "sdf_paired_test.parquet")

    return_based = ["S1_LogUtil_BTCETH", "S3_BTC_only", "S4_RiskReversal", "S9_ETH_only", "S10_PC1"]
    vol_based = ["S2_VolScaled", "S5_RealVol_BTC", "S6_RealVol_ETH", "S7_AggFunding",
                 "S8_AggBasisDisp", "S11_FundingCarry"]

    def family_match(df, family):
        sub = df[df["sdf_proxy"].isin(family)]
        return (sub["sign_match"] == "MATCH").sum(), len(sub)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.4), constrained_layout=False)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.94, bottom=0.24, wspace=0.30)

    ax = axes[0]
    labels = ["Return-based", "Volatility and funding-based"]
    usdt_r, usdt_rt = family_match(multi[multi["stable"] == "USDT"], return_based)
    usdc_r, usdc_rt = family_match(multi[multi["stable"] == "USDC"], return_based)
    usdt_v, usdt_vt = family_match(multi[multi["stable"] == "USDT"], vol_based)
    usdc_v, usdc_vt = family_match(multi[multi["stable"] == "USDC"], vol_based)

    x = np.arange(len(labels))
    width = 0.36
    usdt_pct = [usdt_r/usdt_rt*100, usdt_v/usdt_vt*100]
    usdc_pct = [usdc_r/usdc_rt*100, usdc_v/usdc_vt*100]
    bars_t = ax.bar(x - width/2, usdt_pct, width, color=COLOR["usdt"], label="USDT", edgecolor="white", lw=1.0)
    bars_c = ax.bar(x + width/2, usdc_pct, width, color=COLOR["usdc"], label="USDC", edgecolor="white", lw=1.0)
    ax.axhline(50, color=COLOR["neutral"], lw=0.7, ls=":", alpha=0.5)
    for b, v in zip(bars_t, usdt_pct):
        ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.1f}%",
                ha="center", va="bottom", fontsize=9.0,
                color=COLOR["usdt"], fontweight="bold")
    for b, v in zip(bars_c, usdc_pct):
        is_zero = v < 1e-6
        ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.1f}%",
                ha="center", va="bottom",
                fontsize=10.5 if is_zero else 9.0,
                color=COLOR["stress"] if is_zero else COLOR["usdc"],
                fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Sign-match rate (%)")
    ax.set_ylim(0, 110)
    ax.legend(loc="upper right")
    subcaption(ax, "a", "Match rates for two structurally distinct kernel families", y=-0.16)

    ax = axes[1]
    df = paired.copy()
    label_map = {
        "S1_LogUtil_BTCETH": "Log-utility (BTC+ETH)",
        "S3_BTC_only": "BTC-only return",
        "S4_RiskReversal": "Risk reversal",
        "S9_ETH_only": "ETH-only return",
        "S10_PC1": "First principal component",
    }
    df["lbl"] = df["sdf_proxy"].map(label_map).fillna(df["sdf_proxy"])
    n = len(df)
    y = np.arange(n)
    offset = 0.18
    ax.errorbar(df["beta_USDT"], y - offset,
                xerr=1.96 * df["se_USDT"], fmt="o", color=COLOR["usdt"], ms=5, capsize=2.5, lw=1.2,
                label=r"USDT loading")
    ax.errorbar(df["beta_USDC"], y,
                xerr=1.96 * df["se_USDC"], fmt="s", color=COLOR["usdc"], ms=5, capsize=2.5, lw=1.2,
                label=r"USDC loading")
    ax.errorbar(df["diff"], y + offset,
                xerr=1.96 * df["se_diff"], fmt="D", color=COLOR["stress"], ms=5, capsize=2.5, lw=1.2,
                label=r"USDT minus USDC")
    ax.axvline(0, color=COLOR["neutral"], lw=0.8, alpha=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(df["lbl"], fontsize=8.0)
    ax.invert_yaxis()
    ax.set_xlabel("Partial-regression coefficient on stablecoin discount")
    ax.legend(loc="upper right", fontsize=8.5,
              framealpha=0.95, facecolor="white", edgecolor=COLOR["grid"])
    subcaption(ax, "b", "Paired regression across the five return-based kernels", y=-0.16)

    fig.text(0.5, 0.025,
             "Pricing Kernel Asymmetry Between USDT and USDC",
             ha="center", va="bottom", fontsize=12, fontweight="bold",
             color=COLOR["neutral"])
    save(fig, "figure4")
    plt.close(fig)

def fig08_hj():
    hj = pd.read_parquet(T / "hj_bound_sdf.parquet")
    fig, ax = plt.subplots(figsize=(10.0, 4.8), constrained_layout=False)
    fig.subplots_adjust(left=0.10, right=0.97, top=0.94, bottom=0.20)

    n_assets = len(ASSETS_8)
    width = 0.4
    x = np.arange(n_assets)
    usdt = [float(hj[(hj["stable"] == "USDT") & (hj["asset"] == a)]["HJ_sigma_m_over_mbar_LB"].iloc[0]) * 100
            for a in ASSETS_8]
    usdc = [float(hj[(hj["stable"] == "USDC") & (hj["asset"] == a)]["HJ_sigma_m_over_mbar_LB"].iloc[0]) * 100
            for a in ASSETS_8]
    bars_t = ax.bar(x - width/2, usdt, width, color=COLOR["usdt"], label="USDT", edgecolor="white", lw=1.0)
    bars_c = ax.bar(x + width/2, usdc, width, color=COLOR["usdc"], label="USDC", edgecolor="white", lw=1.0)
    for b, v in zip(bars_t, usdt):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.15, f"{v:.3f}",
                ha="center", va="bottom", fontsize=7.5,
                color=COLOR["usdt"], fontweight="bold")
    for b, v in zip(bars_c, usdc):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.15, f"{v:.3f}",
                ha="center", va="bottom", fontsize=7.5,
                color=COLOR["usdc"], fontweight="bold")

    ymax = max(max(usdt), max(usdc))
    ax.set_xticks(x)
    ax.set_xticklabels(ASSETS_8)
    ax.set_ylabel(r"Implied $\sigma(\tilde m)/\bar m$  (%)")
    ax.set_ylim(0, ymax * 1.30)
    ax.legend(loc="upper left", fontsize=9.5)

    fig.text(0.5, 0.025,
             "Hansen-Jagannathan Bound on Pricing Kernel Volatility",
             ha="center", va="bottom", fontsize=12, fontweight="bold",
             color=COLOR["neutral"])
    save(fig, "figure5")
    plt.close(fig)

def fig09_iv_robust():
    robust = pd.read_parquet(T / "iv_overid_robust.parquet")
    events = pd.read_parquet(T / "iv_event_individual.parquet")
    fix1 = pd.read_parquet(T / "fix_1_iv_with_L.parquet")

    robust["cell"] = robust["stable"] + "-" + robust["asset"]
    events["cell"] = events["stable"] + "-" + events["asset"]

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.4), constrained_layout=False)
    fig.subplots_adjust(left=0.06, right=0.97, top=0.94, bottom=0.18, wspace=0.20)

    ax = axes[0]
    estimators = [("g_OLS", "Ordinary least squares", COLOR["neutral"]),
                  ("g_2SLS_5IV", "Two-stage least squares", COLOR["iv"]),
                  ("g_JIVE", "Jackknife IV", COLOR["highlight"]),
                  ("g_LIML", "Limited-information maximum likelihood", COLOR["usdt"]),
                  ("g_Fuller", "Fuller $k = 1$ estimator", COLOR["usdc"])]
    n_est = len(estimators)
    x = np.arange(len(robust))
    width = 0.18
    for i, (col, lbl, c) in enumerate(estimators):
        offset = (i - (n_est - 1) / 2) * width
        bars = ax.bar(x + offset, robust[col], width, color=c, label=lbl,
                      edgecolor="white", lw=0.6)
        for b, v in zip(bars, robust[col]):
            if v >= 0:
                ax.text(b.get_x() + b.get_width() / 2, v + 0.010, f"{v:.3f}",
                        ha="center", va="bottom", fontsize=6.0,
                        color=c, fontweight="bold")
            else:
                ax.text(b.get_x() + b.get_width() / 2, v - 0.010, f"{v:.3f}",
                        ha="center", va="top", fontsize=6.0,
                        color=c, fontweight="bold")
    ax.axhline(0, color=COLOR["neutral"], lw=0.8, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(robust["cell"], fontsize=9.5)
    ax.set_ylabel(r"Estimated $\hat\gamma_\delta$")
    ax.set_ylim(-0.05, 0.45)
    ax.legend(loc="upper right", fontsize=7.5, ncol=2)
    subcaption(ax, "a", "Robustness to estimator choice across five bias-correction methods", y=-0.08)

    ax = axes[1]
    event_order = ["LUNA_collapse", "FTX_collapse", "USDC_depeg", "BTC_ETF_approval", "Ethena_yen"]
    event_lbl = {"LUNA_collapse": "LUNA collapse", "FTX_collapse": "FTX collapse",
                 "USDC_depeg": "USDC depeg", "BTC_ETF_approval": "BTC ETF approval",
                 "Ethena_yen": "Ethena USDe launch"}
    cmap = {"LUNA_collapse": COLOR["muted"], "FTX_collapse": COLOR["stress"],
            "USDC_depeg": COLOR["highlight"], "BTC_ETF_approval": COLOR["usdc"],
            "Ethena_yen": COLOR["usdt"]}
    n_events = len(event_order)
    x = np.arange(len(CELLS_4))
    width = 0.18
    for j, ev in enumerate(event_order):
        sub = events[events["event"] == ev].set_index("cell").reindex(CELLS_4).reset_index()
        offset = (j - (n_events - 1) / 2) * width
        bar_color = cmap[ev]
        bar_colors = [bar_color if F >= 10 else COLOR["muted"] for F in sub["first_stage_F"]]
        clipped = sub["gamma_IV"].clip(-1.5, 2.5)
        bars = ax.bar(x + offset, clipped, width, color=bar_colors,
                      edgecolor="white", lw=0.5,
                      label=event_lbl[ev] if x[0] == 0 else None)
        for b, v_real, v_plot, bc in zip(bars, sub["gamma_IV"], clipped, bar_colors):
            is_clipped = abs(v_real - v_plot) > 1e-6
            text = f"{v_real:.3f}" + (" ↓" if is_clipped else "")
            if v_plot >= 0:
                ax.text(b.get_x() + b.get_width() / 2, v_plot + 0.08, text,
                        ha="center", va="bottom", fontsize=6.0,
                        color=bc, fontweight="bold")
            else:
                ax.text(b.get_x() + b.get_width() / 2, v_plot - 0.08, text,
                        ha="center", va="top", fontsize=6.0,
                        color=bc, fontweight="bold")

    ax.axhline(0, color=COLOR["neutral"], lw=0.8, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(CELLS_4, fontsize=9.5)
    ax.set_ylabel(r"Estimated $\hat\gamma_\delta$ from one event")
    ax.set_ylim(-1.95, 3.0)
    ax.legend(loc="upper right", fontsize=7.5, ncol=2, framealpha=0.95,
              facecolor="white", edgecolor=COLOR["grid"])
    subcaption(ax, "b", "Robustness to instrument choice across five separate events", y=-0.08)

    fig.text(0.5, 0.025,
             "Robustness of Instrumental-Variable Estimates",
             ha="center", va="bottom", fontsize=12, fontweight="bold",
             color=COLOR["neutral"])
    save(fig, "figure3")
    plt.close(fig)

def fig10_jensen():
    df = pd.read_parquet(T / "jensen_bias_test.parquet").sort_values("beta_var_delta").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9.0, 4.6), constrained_layout=False)
    fig.subplots_adjust(left=0.10, right=0.97, top=0.94, bottom=0.20)

    x = np.arange(len(df))
    colors = [COLOR["usdt"] if b < 0 else COLOR["highlight"] for b in df["beta_var_delta"]]
    beta_k = df["beta_var_delta"] / 1000.0
    se_k = df["se_beta"] / 1000.0
    ax.bar(x, beta_k, color=colors, edgecolor="white", lw=1.0,
           yerr=1.96 * se_k, capsize=4)
    ax.axhline(0, color=COLOR["neutral"], lw=0.8, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(df["asset"], fontsize=10)
    ax.set_ylabel(r"Discount-variance loading,  $\hat\beta_{\rm Jensen}$  $(\times 10^{3})$")
    fig.text(0.5, 0.025,
             "Curvature Signature in Rolling Discount-Variance Loadings",
             ha="center", va="bottom", fontsize=12, fontweight="bold",
             color=COLOR["neutral"])
    save(fig, "figure6")
    plt.close(fig)

if __name__ == "__main__":
    for fn in [fig01_delta, fig02_gamma, fig03_stress_break, fig04_mediation_ik,
               fig05_oos, fig06_kappa_tail, fig07_sdf, fig08_hj,
               fig09_iv_robust, fig10_jensen]:
        try:
            fn()
        except Exception as e:
            print(f"  ✗ {fn.__name__}: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()
    print("\nDone — 10 figures emitted.")
