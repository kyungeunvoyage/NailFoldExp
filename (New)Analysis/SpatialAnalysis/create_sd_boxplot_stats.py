"""
create_sd_boxplot_stats.py
==========================
sd_signed_boxplot_1g_mm.png  및  sd_signed_boxplot_26g_mm.png 에 사용된
모든 통계를 추출하여 각 figure 이름으로 CSV 저장.

sd_signed_boxplot_1g_mm.csv
sd_signed_boxplot_26g_mm.csv

section 컬럼:
  descriptive   — 기술통계 (mean, SD, median, Q1/Q3, IQR, whisker, n)  per offset
  inferential   — GEE / Wilcoxon / one-sample tests
  subject_data  — 피험자별 개별 accuracy (scatter dot 1개 = 1 row)
  jnd           — 피험자별 JND (mm) + 그룹 요약
"""

import os, glob, re, warnings, sys
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
from pathlib import Path

warnings.filterwarnings("ignore")

# ── 경로 ─────────────────────────────────────────────────────────────────────
REPO_ROOT   = "/Users/kyungeunjung/NailFoldExp"
SD_PATTERN  = os.path.join(REPO_ROOT, "Data", "(SD)CurData",
                            "P*_SpatialDiscrimination.csv")
OUTPUT_DIR  = os.path.join(REPO_ROOT, "(New)Analysis",
                            "SpatialAnalysis", "SDAnalysis1_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

GRID_SPACING_MM = 1.5
THRESHOLD       = 0.75


# ── Raw data 로드 ────────────────────────────────────────────────────────────
def _parse_grid(s):
    m = re.match(r"g(-?\d+)", str(s).strip())
    return float(m.group(1)) if m else np.nan

def _parse_force(s):
    m = re.match(r"([\d.]+)", str(s).strip())
    return float(m.group(1)) if m else np.nan

sd_files = sorted(glob.glob(SD_PATTERN))
print(f"[Load] {len(sd_files)} participant file(s)")

df_all = pd.concat(
    [pd.read_csv(f, encoding="utf-8-sig") for f in sd_files],
    ignore_index=True,
)
df_all["pos_1st_mm"]       = df_all["Stim_1st"].apply(_parse_grid) * GRID_SPACING_MM
df_all["pos_2nd_mm"]       = df_all["Stim_2nd"].apply(_parse_grid) * GRID_SPACING_MM
df_all["signed_offset_mm"] = df_all["pos_2nd_mm"] - df_all["pos_1st_mm"]
df_all["abs_offset_mm"]    = df_all["signed_offset_mm"].abs()
df_all["force_g"]          = df_all["Force"].apply(_parse_force)
df_all["IsCorrect"]        = pd.to_numeric(df_all["IsCorrect"], errors="coerce")
df_all = df_all.dropna(subset=["IsCorrect", "signed_offset_mm", "force_g"])

print(f"  n_subjects={df_all['Subject'].nunique()}, rows={len(df_all)}")


# ── 피험자별 signed offset 기준 accuracy ────────────────────────────────────
subj_sym = (
    df_all.groupby(["Subject", "force_g", "signed_offset_mm"])
    .agg(accuracy=("IsCorrect", "mean"), n_trials=("IsCorrect", "count"))
    .reset_index()
)

# ── Psychometric fit & JND ────────────────────────────────────────────────────
def _psychometric(x, x50, beta, lapse=0.02):
    return 0.5 + (0.48 - lapse) / (1.0 + np.exp(-beta * (x - x50)))

def _fit_curve(xs, ys):
    try:
        popt, _ = curve_fit(
            _psychometric, xs, ys,
            p0=(3.0, 1.0),
            bounds=([0.1, 0.05], [15.0, 10.0]),
            maxfev=8000,
        )
        return popt, True
    except Exception:
        return (np.nan, np.nan), False

def _jnd_from_fit(popt):
    x_arr = np.linspace(0, 20, 20000)
    y_arr = _psychometric(x_arr, *popt)
    idx   = np.argmin(np.abs(y_arr - THRESHOLD))
    return float(x_arr[idx])

subj_acc = (
    df_all.groupby(["Subject", "force_g", "abs_offset_mm"])
    .agg(accuracy=("IsCorrect", "mean"), n_trials=("IsCorrect", "count"))
    .reset_index()
)
jnd_records = []
for (subj, force), grp in subj_acc.groupby(["Subject", "force_g"]):
    grp_s = grp.sort_values("abs_offset_mm")
    xs, ys = grp_s["abs_offset_mm"].values, grp_s["accuracy"].values
    if len(xs) < 3:
        continue
    popt, ok = _fit_curve(xs, ys)
    jnd_val  = _jnd_from_fit(popt) if ok else np.nan
    jnd_records.append({"Subject": subj, "force_g": force,
                        "jnd_mm": jnd_val, "x50": popt[0], "beta": popt[1], "fit_ok": ok})
jnd_df = pd.DataFrame(jnd_records)

# Wilcoxon JND(1g) vs JND(26g)
wide_jnd = jnd_df[jnd_df["fit_ok"]].pivot(index="Subject", columns="force_g", values="jnd_mm").dropna()
if 1.0 in wide_jnd.columns and 26.0 in wide_jnd.columns:
    w_stat, p_wil = stats.wilcoxon(wide_jnd[1.0], wide_jnd[26.0])
    n_wil = len(wide_jnd)
else:
    w_stat, p_wil, n_wil = np.nan, np.nan, 0
print(f"  Wilcoxon JND(1g vs 26g): W={w_stat:.2f}, p={p_wil:.4f}, n={n_wil}")


# ── GEE results (from saved text file) ──────────────────────────────────────
GEE_COEFFS = {
    "Intercept":           {"coef": -1.8666, "se": 0.277, "z": -6.731, "p":  0.000, "ci_lo": -2.410, "ci_hi": -1.323},
    "abs_offset_mm":       {"coef":  1.0165, "se": 0.123, "z":  8.272, "p":  0.000, "ci_lo":  0.776, "ci_hi":  1.257},
    "force_g":             {"coef":  0.0206, "se": 0.013, "z":  1.536, "p":  0.124, "ci_lo": -0.006, "ci_hi":  0.047},
    "abs_offset_mm:force_g":{"coef":-0.0032, "se": 0.006, "z": -0.552, "p":  0.581, "ci_lo": -0.015, "ci_hi":  0.008},
}
GEE_META = {"n_obs": 1632, "n_subjects": 25, "family": "Binomial",
            "cov_struct": "Exchangeable", "formula": "IsCorrect ~ abs_offset_mm * force_g"}


# ── 헬퍼: boxplot stats ───────────────────────────────────────────────────────
def boxplot_stats(vals):
    arr = np.asarray(vals, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return dict(n=0, mean=np.nan, sd=np.nan, sem=np.nan,
                    median=np.nan, q1=np.nan, q3=np.nan, iqr=np.nan,
                    whisker_lo=np.nan, whisker_hi=np.nan)
    q1, med, q3 = np.percentile(arr, [25, 50, 75])
    iqr = q3 - q1
    lo_f, hi_f = q1 - 1.5*iqr, q3 + 1.5*iqr
    in_r = arr[(arr >= lo_f) & (arr <= hi_f)]
    return dict(
        n=len(arr),
        mean=float(arr.mean()),
        sd=float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        sem=float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0,
        median=float(med),
        q1=float(q1), q3=float(q3), iqr=float(iqr),
        whisker_lo=float(in_r.min()) if len(in_r) else float(arr.min()),
        whisker_hi=float(in_r.max()) if len(in_r) else float(arr.max()),
    )


def sig_star(p):
    if pd.isna(p): return ""
    if p < 0.001:  return "***"
    if p < 0.01:   return "**"
    if p < 0.05:   return "*"
    return "n.s."


# ── Figure별 CSV 생성 ─────────────────────────────────────────────────────────
FORCE_LABEL = {1.0: "1g", 26.0: "26g"}
FIGURE_NAMES = {
    1.0:  "sd_signed_boxplot_1g_mm",
    26.0: "sd_signed_boxplot_26g_mm",
}

signed_vals = sorted(subj_sym["signed_offset_mm"].unique())

for force in [1.0, 26.0]:
    rows_out = []
    fsym = subj_sym[subj_sym["force_g"] == force].copy()
    force_str = FORCE_LABEL[force]

    # ── Descriptive (per signed offset) ─────────────────────────────────────
    for offset in signed_vals:
        sub = fsym[fsym["signed_offset_mm"] == offset]
        vals = sub["accuracy"].values * 100
        if len(vals) == 0:
            continue
        bs = boxplot_stats(vals)
        # One-sample Wilcoxon vs 75% threshold
        try:
            w1s, p1s = stats.wilcoxon(vals - 75.0)
        except Exception:
            w1s, p1s = np.nan, np.nan
        # One-sample t-test vs 50% (chance)
        try:
            t_ch, p_ch = stats.ttest_1samp(vals, 50.0)
        except Exception:
            t_ch, p_ch = np.nan, np.nan

        rows_out.append({
            "figure":         FIGURE_NAMES[force],
            "force_g":        force,
            "section":        "descriptive",
            "signed_offset_mm": offset,
            "n_subjects":     bs["n"],
            "mean_pct":       round(bs["mean"], 4),
            "sd_pct":         round(bs["sd"], 4),
            "sem_pct":        round(bs["sem"], 4),
            "median_pct":     round(bs["median"], 4),
            "q1_pct":         round(bs["q1"], 4),
            "q3_pct":         round(bs["q3"], 4),
            "iqr_pct":        round(bs["iqr"], 4),
            "whisker_lo_pct": round(bs["whisker_lo"], 4),
            "whisker_hi_pct": round(bs["whisker_hi"], 4),
            "vs_chance_t":    round(t_ch, 4)  if not np.isnan(t_ch) else np.nan,
            "vs_chance_p":    round(p_ch, 6)  if not np.isnan(p_ch) else np.nan,
            "vs_chance_sig":  sig_star(p_ch),
            "vs_75pct_W":     round(w1s, 4)   if not np.isnan(w1s) else np.nan,
            "vs_75pct_p":     round(p1s, 6)   if not np.isnan(p1s) else np.nan,
            "vs_75pct_sig":   sig_star(p1s),
        })

    # ── Inferential: GEE (shared across both figures, noted once) ───────────
    for term, c in GEE_COEFFS.items():
        rows_out.append({
            "figure":   FIGURE_NAMES[force],
            "force_g":  force,
            "section":  "inferential",
            "stat_type": "GEE",
            "term":      term,
            "gee_formula": GEE_META["formula"],
            "gee_family":  GEE_META["family"],
            "gee_cov_struct": GEE_META["cov_struct"],
            "n_obs":     GEE_META["n_obs"],
            "coef":      c["coef"],
            "se":        c["se"],
            "z_stat":    c["z"],
            "p_value":   c["p"],
            "ci_lo_95":  c["ci_lo"],
            "ci_hi_95":  c["ci_hi"],
            "significance": sig_star(c["p"]),
        })

    # ── Inferential: Wilcoxon JND(1g vs 26g) ────────────────────────────────
    rows_out.append({
        "figure":    FIGURE_NAMES[force],
        "force_g":   force,
        "section":   "inferential",
        "stat_type": "Wilcoxon signed-rank (paired)",
        "term":      "JND(1g) vs JND(26g)",
        "n_obs":     n_wil,
        "W_stat":    round(w_stat, 4) if not np.isnan(w_stat) else np.nan,
        "p_value":   round(p_wil, 6)  if not np.isnan(p_wil) else np.nan,
        "significance": sig_star(p_wil),
    })

    # ── Subject data (scatter dots) ──────────────────────────────────────────
    for _, r in fsym.iterrows():
        rows_out.append({
            "figure":           FIGURE_NAMES[force],
            "force_g":          force,
            "section":          "subject_data",
            "signed_offset_mm": r["signed_offset_mm"],
            "subject_id":       r["Subject"],
            "accuracy_pct":     round(r["accuracy"] * 100, 4),
            "n_trials":         int(r["n_trials"]),
        })

    # ── JND per subject ──────────────────────────────────────────────────────
    fjnd = jnd_df[jnd_df["force_g"] == force].copy()
    # Group summary
    jnd_ok = fjnd[fjnd["fit_ok"]]
    jnd_bs  = boxplot_stats(jnd_ok["jnd_mm"].values)
    rows_out.append({
        "figure":        FIGURE_NAMES[force],
        "force_g":       force,
        "section":       "jnd",
        "stat_type":     "JND group summary",
        "n_subjects":    jnd_bs["n"],
        "jnd_mean_mm":   round(jnd_bs["mean"], 4),
        "jnd_sd_mm":     round(jnd_bs["sd"], 4),
        "jnd_median_mm": round(jnd_bs["median"], 4),
        "jnd_q1_mm":     round(jnd_bs["q1"], 4),
        "jnd_q3_mm":     round(jnd_bs["q3"], 4),
        "jnd_whisker_lo_mm": round(jnd_bs["whisker_lo"], 4),
        "jnd_whisker_hi_mm": round(jnd_bs["whisker_hi"], 4),
    })
    for _, r in fjnd.iterrows():
        rows_out.append({
            "figure":     FIGURE_NAMES[force],
            "force_g":    force,
            "section":    "jnd",
            "stat_type":  "JND per subject",
            "subject_id": r["Subject"],
            "jnd_mm":     round(r["jnd_mm"], 4) if not np.isnan(r["jnd_mm"]) else np.nan,
            "x50_mm":     round(r["x50"], 4)    if not np.isnan(r["x50"]) else np.nan,
            "beta":       round(r["beta"], 4)   if not np.isnan(r["beta"]) else np.nan,
            "fit_ok":     r["fit_ok"],
        })

    # ── 저장 ─────────────────────────────────────────────────────────────────
    df_out = pd.DataFrame(rows_out)
    out_path = os.path.join(OUTPUT_DIR, f"{FIGURE_NAMES[force]}.csv")
    df_out.to_csv(out_path, index=False, float_format="%.6f")
    print(f"\n완료 → {out_path}  ({len(df_out)} rows)")


print("\n=== 모든 CSV 생성 완료 ===")
