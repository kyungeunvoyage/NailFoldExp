"""
ATD Psychometric Function Fitting  (v2 — threshold-aligned mean curve 추가)
===========================================================================
세 가지 방식의 group-level psychometric curve를 비교합니다:

  (A) Group-mean fit   — 30명 accuracy를 force별로 먼저 평균낸 뒤 한 curve fit
                          → ceiling effect로 인해 threshold가 오른쪽으로 bias됨
  (B) Threshold-aligned mean — 각 피험자 curve를 본인 threshold에 정렬한 뒤 평균
                          → bias 없는 "평균 curve shape"
  (C) Individual threshold scatter — 개인별 threshold 분포

Figure 구성
-----------
  Panel A: 개인 curve(얇) + Group-mean fit(굵) + Threshold-aligned mean(굵,점선)
  Panel B: Group-mean fit vs Threshold-aligned mean 직접 비교 (두 조건 각각)
  Panel C: 개인 threshold 분포 (scatter + mean±SEM)

Expected CSV columns:
  Subject | Force | Condition | Correct  (1=detected, 0=not detected)
"""

import os
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib import rcParams
from scipy.optimize import curve_fit
from scipy import stats as sci_stats

# =============================================================================
# --- Paths ---
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.normpath(os.path.join(SCRIPT_DIR, "../../"))
FILE_PATTERN = os.path.join(
    REPO_ROOT, "Data", "(ATD)CurData", "P*_AbsoluteThresholdDetection.csv"
)
OUT_DIR = os.path.join(SCRIPT_DIR, "atd_c1_outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# =============================================================================
# --- Column / condition names ---
# =============================================================================
COL_SUBJECT   = "Subject"
COL_FORCE     = "Force"
COL_CONDITION = "Condition"
COL_CORRECT   = "Correct"

VAL_INAIR   = "In-air"
VAL_ONTOUCH = "On-touch (Mid)"
CONDITIONS  = [VAL_INAIR, VAL_ONTOUCH]

FORCE_ORDER = [0.07, 0.16, 0.6, 1.0, 1.4]
CRITERION   = 0.80


def normalize_atd_columns(raw: pd.DataFrame) -> pd.DataFrame:
    """Map CurData headers (SubjectID, IsCorrect, 1.0g) to internal column names."""
    out = raw.copy()
    renames = {}
    if COL_SUBJECT not in out.columns:
        if "SubjectID" in out.columns:
            renames["SubjectID"] = COL_SUBJECT
        elif "subject_id" in out.columns:
            renames["subject_id"] = COL_SUBJECT
    if COL_CORRECT not in out.columns:
        if "IsCorrect" in out.columns:
            renames["IsCorrect"] = COL_CORRECT
        elif "is_correct" in out.columns:
            renames["is_correct"] = COL_CORRECT
    if renames:
        out = out.rename(columns=renames)
    missing = [c for c in (COL_SUBJECT, COL_FORCE, COL_CONDITION, COL_CORRECT) if c not in out.columns]
    if missing:
        raise KeyError(
            f"Missing columns {missing}. Found: {list(out.columns)}"
        )
    return out


# =============================================================================
# --- Palette ---
# =============================================================================
SLATE_BLUE = "#56708A"
OLIVE      = "#686F12"
WINE       = "#7F212B"
BLACK      = "#1A1A1A"

COND_CLR = {VAL_INAIR: SLATE_BLUE, VAL_ONTOUCH: OLIVE}

rcParams.update({
    "figure.facecolor":  "#FFFFFF",
    "axes.facecolor":    "#FFFFFF",
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Helvetica", "Arial", "DejaVu Sans"],
    "axes.linewidth":    0.8,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.size":         9,
    "axes.labelsize":    10,
    "axes.titlesize":    11,
    "axes.grid":         True,
    "axes.grid.axis":    "y",
    "grid.alpha":        0.25,
    "grid.linestyle":    "--",
    "grid.color":        SLATE_BLUE,
})

# =============================================================================
# 1.  Load & clean data
# =============================================================================
files = glob.glob(FILE_PATTERN)
if not files:
    raise FileNotFoundError(f"No CSVs found:\n  {FILE_PATTERN}")
print(f"Loaded {len(files)} participant file(s).")

df = pd.concat(
    [pd.read_csv(f, encoding="utf-8-sig") for f in sorted(files)],
    ignore_index=True,
)
df = normalize_atd_columns(df)
df[COL_CONDITION] = df[COL_CONDITION].astype(str).str.strip()
df[COL_CONDITION] = df[COL_CONDITION].replace(
    {"Active": VAL_ONTOUCH, "On-touch (Hard)": VAL_ONTOUCH, "Passive": VAL_INAIR}
)
df = df[df[COL_CONDITION].isin(CONDITIONS)].copy()

_force = df[COL_FORCE].astype(str).str.strip().str.lower().str.replace("g", "", regex=False)
df[COL_FORCE] = pd.to_numeric(_force, errors="coerce")
df[COL_CORRECT] = pd.to_numeric(df[COL_CORRECT], errors="coerce")
df = df.dropna(subset=[COL_FORCE, COL_CORRECT])

n_subjects = df[COL_SUBJECT].nunique()
print(f"Rows: {len(df)}  |  Subjects: {n_subjects}")

# =============================================================================
# 2.  Per-subject accuracy per force × condition
# =============================================================================
subj_acc = (
    df.groupby([COL_SUBJECT, COL_CONDITION, COL_FORCE])[COL_CORRECT]
    .mean()
    .reset_index()
    .rename(columns={COL_CORRECT: "accuracy", COL_FORCE: "force"})
)

# =============================================================================
# 3.  Psychometric function: logistic on log10(force)
# =============================================================================
def logistic(log_x, alpha, beta):
    """
    P(detection) = 1 / (1 + exp(-(log_x - alpha) / beta))
      log_x : log10(force)
      alpha : threshold on log10 scale  (P = 0.5 when log_x == alpha)
      beta  : inverse slope (>0; smaller = steeper)
    """
    return 1.0 / (1.0 + np.exp(-(log_x - alpha) / beta))


def threshold_from_params(alpha, beta, criterion=CRITERION):
    """Solve logistic(log_x) = criterion → force in grams."""
    log_thr = alpha + beta * np.log(criterion / (1.0 - criterion))
    return 10.0 ** log_thr


def fit_subject_psychometric(forces, accuracies):
    """
    Fit logistic to one subject's data on log10(force) scale.
    Returns (alpha, beta, threshold_g) or (nan, nan, nan) on failure.
    """
    log_f = np.log10(np.asarray(forces, dtype=float))
    acc   = np.asarray(accuracies, dtype=float)

    idx0   = np.argmin(np.abs(acc - 0.50))
    alpha0 = log_f[idx0]
    beta0  = max((log_f[-1] - log_f[0]) / 4.0, 0.05)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, _ = curve_fit(
                logistic, log_f, acc,
                p0=[alpha0, beta0],
                bounds=([-3.0, 1e-3], [2.0, 5.0]),
                maxfev=10_000,
            )
        alpha, beta = popt
        return alpha, beta, threshold_from_params(alpha, beta)
    except Exception:
        return np.nan, np.nan, np.nan


# --- Per-subject fits ---
fit_rows = []
for (subj, cond), grp in subj_acc.groupby([COL_SUBJECT, COL_CONDITION]):
    g = grp.sort_values("force")
    alpha, beta, thr = fit_subject_psychometric(
        g["force"].values, g["accuracy"].values
    )
    fit_rows.append(dict(Subject=subj, Condition=cond,
                         alpha=alpha, beta=beta, threshold=thr))

df_fit = pd.DataFrame(fit_rows)

# =============================================================================
# 4.  Group-level stats (mean ± SEM per force × condition)
# =============================================================================
def group_stats(cond):
    sub = subj_acc[subj_acc[COL_CONDITION] == cond]
    rows = []
    for f in sorted(sub["force"].unique()):
        vals = sub.loc[sub["force"] == f, "accuracy"].values
        rows.append({"force": f, "mean": np.mean(vals),
                     "sem": sci_stats.sem(vals), "n": len(vals)})
    return pd.DataFrame(rows)

grp_stats = {c: group_stats(c) for c in CONDITIONS}

# Group-level fit (fit on group-mean accuracy)
grp_fit = {}
for cond in CONDITIONS:
    g = grp_stats[cond]
    alpha, beta, thr = fit_subject_psychometric(
        g["force"].values, g["mean"].values
    )
    grp_fit[cond] = dict(alpha=alpha, beta=beta, threshold=thr)

# =============================================================================
# 5.  Threshold-aligned mean curve  ← NEW
# =============================================================================
def compute_threshold_aligned(df_fit_cond, n_pts=500):
    """
    각 피험자의 logistic curve를 자신의 threshold 위치로 정렬(shift)한 뒤
    평균을 냅니다. 결과를 mean threshold 위치로 다시 shift하여 반환.

    아이디어:
      - 개인 threshold on log scale: thr_log_i = log10(threshold_i)
      - 정렬된 x: x_shifted = log10(force) - thr_log_i
        → x_shifted = 0 에서 모든 피험자가 P = CRITERION을 가짐
      - 정렬된 공간에서 곡선을 평균 → 평균 slope(beta) 추정
      - 원래 force 단위로 복원: x_plot = x_shifted + mean(thr_log)

    Returns
    -------
    x_g    : force 축 (grams), display용
    y_mean : 평균 detection probability (0–1)
    y_sem  : SEM across subjects
    mean_thr_g : mean threshold used for the back-shift
    """
    # x_shifted 범위: 모든 피험자를 커버하도록 넉넉히 설정
    x_shifted = np.linspace(-2.2, 1.8, n_pts)

    curves = []
    thr_logs = []
    for _, row in df_fit_cond.iterrows():
        if not (np.isfinite(row["alpha"]) and np.isfinite(row["beta"])
                and np.isfinite(row["threshold"]) and row["threshold"] > 0):
            continue
        thr_log = np.log10(row["threshold"])
        # curve evaluated at x_shifted: actual log10_force = x_shifted + thr_log_i
        y = logistic(x_shifted + thr_log, row["alpha"], row["beta"])
        curves.append(y)
        thr_logs.append(thr_log)

    if len(curves) < 2:
        return None, None, None, np.nan

    curves = np.array(curves)          # shape: (n_subjects, n_pts)
    y_mean = np.mean(curves, axis=0)
    y_sem  = sci_stats.sem(curves, axis=0)

    # Back-shift to original force scale using the mean threshold
    mean_thr_log = np.mean(thr_logs)
    x_g = 10.0 ** (x_shifted + mean_thr_log)
    mean_thr_g = 10.0 ** mean_thr_log

    return x_g, y_mean, y_sem, mean_thr_g


aligned = {}
for cond in CONDITIONS:
    sub_df = df_fit[df_fit["Condition"] == cond].copy()
    x_g, y_mean, y_sem, mean_thr_g = compute_threshold_aligned(sub_df)
    aligned[cond] = dict(x=x_g, y_mean=y_mean, y_sem=y_sem,
                         mean_thr_g=mean_thr_g)

# =============================================================================
# 6.  Wilcoxon signed-rank test
# =============================================================================
def paired_thresholds(cond_a, cond_b):
    a = df_fit[df_fit["Condition"] == cond_a].set_index("Subject")["threshold"]
    b = df_fit[df_fit["Condition"] == cond_b].set_index("Subject")["threshold"]
    common = a.index.intersection(b.index)
    a, b = a[common].values, b[common].values
    mask = np.isfinite(a) & np.isfinite(b)
    return a[mask], b[mask]

a_thr, b_thr = paired_thresholds(VAL_INAIR, VAL_ONTOUCH)
if len(a_thr) >= 5:
    stat, pval = sci_stats.wilcoxon(a_thr, b_thr)
    test_str = f"Wilcoxon signed-rank: W={stat:.1f}, p={pval:.4f}, n={len(a_thr)}"
else:
    pval, test_str = np.nan, "Too few paired subjects for Wilcoxon test."

# =============================================================================
# 7.  Figure  (3 panels)
# =============================================================================
fig = plt.figure(figsize=(17, 5.5), facecolor="white")
gs  = fig.add_gridspec(1, 3, width_ratios=[1.7, 1.3, 1.0], wspace=0.32)
ax_c  = fig.add_subplot(gs[0])   # Panel A: curves
ax_cmp = fig.add_subplot(gs[1])  # Panel B: group-fit vs aligned comparison
ax_t  = fig.add_subplot(gs[2])   # Panel C: individual thresholds

log_x_fine = np.linspace(np.log10(0.04), np.log10(2.5), 400)
x_fine     = 10.0 ** log_x_fine

# ── Panel A: psychometric curves ─────────────────────────────────────────────
for cond in CONDITIONS:
    clr = COND_CLR[cond]
    g   = grp_stats[cond]
    gf  = grp_fit[cond]
    al  = aligned[cond]
    n_s = df[df[COL_CONDITION] == cond][COL_SUBJECT].nunique()

    # Individual subject curves (thin background)
    for _, row in df_fit[df_fit["Condition"] == cond].iterrows():
        if np.isfinite(row["alpha"]):
            y_sub = logistic(log_x_fine, row["alpha"], row["beta"])
            ax_c.plot(x_fine, y_sub * 100,
                      color=clr, alpha=0.09, linewidth=0.8, zorder=2)

    # Group-mean fit (solid thick)
    if np.isfinite(gf["alpha"]):
        y_grp = logistic(log_x_fine, gf["alpha"], gf["beta"])
        ax_c.plot(x_fine, y_grp * 100,
                  color=clr, linewidth=2.2, zorder=5,
                  label=f"{cond} group-fit (thr={gf['threshold']:.2f} g)")
        thr_g = gf["threshold"]
        ax_c.plot([thr_g, thr_g], [-5, CRITERION * 100],
                  color=clr, linewidth=0.9, linestyle=":", alpha=0.6, zorder=3)
        ax_c.scatter([thr_g], [CRITERION * 100],
                     color=clr, s=55, zorder=8,
                     marker="^", edgecolors=BLACK, linewidths=0.5)

    # Threshold-aligned mean curve (dashed thick)
    if al["x"] is not None:
        mask = (al["x"] >= 0.04) & (al["x"] <= 2.5)
        ax_c.plot(al["x"][mask], al["y_mean"][mask] * 100,
                  color=clr, linewidth=2.0, linestyle=(0, (6, 3)),
                  zorder=6, alpha=0.85,
                  label=f"{cond} aligned-mean (thr={al['mean_thr_g']:.2f} g)")
        # SEM band
        ax_c.fill_between(
            al["x"][mask],
            (al["y_mean"][mask] - al["y_sem"][mask]) * 100,
            (al["y_mean"][mask] + al["y_sem"][mask]) * 100,
            color=clr, alpha=0.07, zorder=1,
        )
        thr_al = al["mean_thr_g"]
        ax_c.scatter([thr_al], [CRITERION * 100],
                     color=clr, s=55, zorder=8,
                     marker="v", edgecolors=BLACK, linewidths=0.5)

    # Raw group mean ± SEM dots
    ax_c.errorbar(
        g["force"].values, g["mean"].values * 100,
        yerr=g["sem"].values * 100,
        fmt="o", color=clr, markersize=6,
        capsize=3, capthick=1.0, linewidth=0,
        markeredgecolor=BLACK, markeredgewidth=0.4,
        zorder=7,
    )

ax_c.axhline(CRITERION * 100, color=WINE, linestyle="--",
             linewidth=1.2, alpha=0.85, label="80 % criterion")
ax_c.axvspan(0.05, 0.20, color=OLIVE, alpha=0.06, zorder=0)
ax_c.text(0.108, 6, "Fingerpad MDT\n(Rolke 2006:\n0.05–0.2 g)",
          ha="center", va="bottom", fontsize=6.5,
          color=OLIVE, style="italic", linespacing=1.35)

# Legend: ▲ = group-fit threshold, ▼ = aligned-mean threshold
extra_handles = [
    Line2D([0], [0], color="gray", lw=2.0,
           label="Group-mean fit (solid)"),
    Line2D([0], [0], color="gray", lw=2.0, linestyle=(0, (6, 3)),
           label="Threshold-aligned mean (dashed)"),
    Line2D([0], [0], marker="^", color="w", markerfacecolor="gray",
           markersize=7, label="▲ Group-fit threshold"),
    Line2D([0], [0], marker="v", color="w", markerfacecolor="gray",
           markersize=7, label="▼ Aligned-mean threshold"),
]
ax_c.legend(handles=extra_handles, fontsize=7.5, frameon=False,
            loc="upper left", ncol=2, columnspacing=0.8)

ax_c.set_xscale("log")
ax_c.set_xticks(FORCE_ORDER)
ax_c.set_xticklabels([str(f) for f in FORCE_ORDER])
ax_c.set_xlim(0.04, 2.5)
ax_c.set_ylim(-5, 115)
ax_c.set_xlabel("Stimulus Force (g, log scale)", fontsize=10, labelpad=6)
ax_c.set_ylabel("Detection Accuracy (%)", fontsize=10)
ax_c.set_title("Psychometric Functions — Periungual ATD",
               fontsize=11, fontweight="bold", pad=10)

# ── Panel B: group-fit vs aligned comparison ─────────────────────────────────
for ci, cond in enumerate(CONDITIONS):
    clr = COND_CLR[cond]
    gf  = grp_fit[cond]
    al  = aligned[cond]

    row_offset = ci * 0.04   # slight vertical shift for readability

    # Group-mean fit curve
    if np.isfinite(gf["alpha"]):
        y_grp = logistic(log_x_fine, gf["alpha"], gf["beta"])
        ax_cmp.plot(x_fine, y_grp * 100 + row_offset,
                    color=clr, linewidth=2.0, zorder=5)
        ax_cmp.axvline(gf["threshold"], color=clr, linewidth=1.0,
                       linestyle=":", alpha=0.5)
        ax_cmp.scatter([gf["threshold"]], [CRITERION * 100 + row_offset],
                       color=clr, s=55, marker="^",
                       edgecolors=BLACK, linewidths=0.5, zorder=8)

    # Threshold-aligned mean curve
    if al["x"] is not None:
        mask = (al["x"] >= 0.04) & (al["x"] <= 2.5)
        ax_cmp.plot(al["x"][mask], al["y_mean"][mask] * 100 + row_offset,
                    color=clr, linewidth=2.0,
                    linestyle=(0, (6, 3)), zorder=6, alpha=0.85)
        ax_cmp.fill_between(
            al["x"][mask],
            (al["y_mean"][mask] - al["y_sem"][mask]) * 100 + row_offset,
            (al["y_mean"][mask] + al["y_sem"][mask]) * 100 + row_offset,
            color=clr, alpha=0.08,
        )
        ax_cmp.axvline(al["mean_thr_g"], color=clr, linewidth=1.0,
                       linestyle="-.", alpha=0.5)
        ax_cmp.scatter([al["mean_thr_g"]], [CRITERION * 100 + row_offset],
                       color=clr, s=55, marker="v",
                       edgecolors=BLACK, linewidths=0.5, zorder=8)

    # Bias annotation arrow
    if (np.isfinite(gf.get("threshold", np.nan)) and
            al.get("mean_thr_g") is not None and np.isfinite(al["mean_thr_g"])):
        thr_gf = gf["threshold"]
        thr_al = al["mean_thr_g"]
        y_arr  = CRITERION * 100 + 6 + ci * 5
        ax_cmp.annotate(
            "", xy=(thr_gf, y_arr), xytext=(thr_al, y_arr),
            arrowprops=dict(arrowstyle="->", color=clr, lw=1.2),
        )
        ax_cmp.text((thr_gf + thr_al) / 2, y_arr + 1.5,
                    f"bias\n+{thr_gf - thr_al:+.2f} g",
                    ha="center", va="bottom",
                    fontsize=6.5, color=clr)

ax_cmp.axhline(CRITERION * 100, color=WINE, linestyle="--",
               linewidth=1.0, alpha=0.8)
ax_cmp.axvspan(0.05, 0.20, color=OLIVE, alpha=0.06, zorder=0)

ax_cmp.set_xscale("log")
ax_cmp.set_xticks(FORCE_ORDER)
ax_cmp.set_xticklabels([str(f) for f in FORCE_ORDER])
ax_cmp.set_xlim(0.04, 2.5)
ax_cmp.set_ylim(-5, 115)
ax_cmp.set_xlabel("Stimulus Force (g, log scale)", fontsize=10, labelpad=6)
ax_cmp.set_ylabel("Detection Accuracy (%)", fontsize=10)
ax_cmp.set_title("Group-fit  vs  Threshold-aligned mean",
                 fontsize=11, fontweight="bold", pad=10)

# Compact legend for Panel B
b_handles = [
    Line2D([0], [0], color=SLATE_BLUE, lw=2.0, label="In-air"),
    Line2D([0], [0], color=OLIVE,      lw=2.0, label="On-touch (Mid)"),
    Line2D([0], [0], color="gray", lw=1.8,
           label="Solid = group-fit"),
    Line2D([0], [0], color="gray", lw=1.8, linestyle=(0, (6, 3)),
           label="Dashed = aligned-mean"),
    Line2D([0], [0], marker="^", color="w", markerfacecolor="gray",
           markersize=6, label="▲ Group-fit thr"),
    Line2D([0], [0], marker="v", color="w", markerfacecolor="gray",
           markersize=6, label="▼ Aligned-mean thr"),
]
ax_cmp.legend(handles=b_handles, fontsize=7.0, frameon=False,
              loc="upper left", ncol=2, columnspacing=0.6)

# ── Panel C: individual threshold distributions ───────────────────────────────
rng = np.random.default_rng(42)

for xi, cond in enumerate(CONDITIONS):
    clr  = COND_CLR[cond]
    thrs = df_fit.loc[
        (df_fit["Condition"] == cond) & np.isfinite(df_fit["threshold"]),
        "threshold"
    ].values

    jx = xi + (rng.random(len(thrs)) - 0.5) * 0.28
    ax_t.scatter(jx, thrs, color=clr, alpha=0.45, s=22, zorder=4)

    m, se = np.mean(thrs), sci_stats.sem(thrs)
    ax_t.errorbar(xi, m, yerr=se,
                  fmt="D", color=clr, markersize=8,
                  capsize=4, capthick=1.3, linewidth=1.4,
                  markeredgecolor=BLACK, markeredgewidth=0.5, zorder=6)
    ax_t.text(xi, m * 1.12, f"{m:.2f} g",
              ha="center", fontsize=8.5, color=clr, fontweight="500")

    # Aligned-mean threshold marker (diamond outline)
    if aligned[cond]["mean_thr_g"] is not None:
        ax_t.scatter([xi], [aligned[cond]["mean_thr_g"]],
                     marker="D", s=55, facecolors="none",
                     edgecolors=clr, linewidths=1.5, zorder=7)

ax_t.axhspan(0.05, 0.20, color=OLIVE, alpha=0.10, zorder=0)
ax_t.text(1.42, 0.115, "Fingerpad MDT\n(Rolke 2006:\n0.05–0.2 g)",
          ha="center", fontsize=6.5, color=OLIVE,
          style="italic", va="center")

if np.isfinite(pval):
    sig = ("***" if pval < 0.001 else
           "**"  if pval < 0.01  else
           "*"   if pval < 0.05  else "n.s.")
else:
    sig = "n.s."

ax_t.set_xticks([0, 1])
ax_t.set_xticklabels([VAL_INAIR, VAL_ONTOUCH], fontsize=9.5)
ax_t.set_xlim(-0.6, 1.8)
ax_t.set_yscale("log")
ax_t.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
ax_t.set_ylabel("Threshold Force (g, log scale)", fontsize=10)
ax_t.set_title("Individual Thresholds (80%)", fontsize=11,
               fontweight="bold", pad=10)

# Wilcoxon bracket
ymax = ax_t.get_ylim()[1]
bracket_y = ymax * 0.80
ax_t.plot([0, 0, 1, 1],
          [bracket_y * 0.85, bracket_y, bracket_y, bracket_y * 0.85],
          color=BLACK, linewidth=1.0, clip_on=False)
ax_t.text(0.5, bracket_y * 1.05, sig,
          ha="center", va="bottom", fontsize=10,
          color="crimson" if (np.isfinite(pval) and pval < 0.05) else "dimgray")

# Filled = individual mean, outline = aligned-mean
c_handles = [
    Line2D([0], [0], marker="D", color="w", markerfacecolor="gray",
           markersize=8, markeredgecolor="gray", markeredgewidth=0.5,
           label="Filled ◆ = individual mean ± SEM"),
    Line2D([0], [0], marker="D", color="w", markerfacecolor="none",
           markersize=8, markeredgecolor="gray", markeredgewidth=1.5,
           label="Outline ◇ = threshold-aligned mean"),
]
ax_t.legend(handles=c_handles, fontsize=7.0, frameon=False, loc="lower right")

# =============================================================================
# 8.  Save figure
# =============================================================================
fig.tight_layout(pad=2.0)
out_path = os.path.join(OUT_DIR, "atd_psychometric_v2.png")
fig.savefig(out_path, dpi=600, facecolor="white", bbox_inches="tight")
plt.close(fig)
print(f"\nSaved → {out_path}")

# =============================================================================
# 9.  Print summary
# =============================================================================
print("\n" + "=" * 60)
print(f"  THRESHOLD SUMMARY  (criterion = {CRITERION*100:.0f} %)")
print("=" * 60)
for cond in CONDITIONS:
    sub = df_fit.loc[
        (df_fit["Condition"] == cond) & np.isfinite(df_fit["threshold"]),
        "threshold"
    ].values
    al_thr = aligned[cond]["mean_thr_g"]
    gf_thr = grp_fit[cond].get("threshold", np.nan)

    print(f"\n{cond}  (n = {len(sub)} valid fits)")
    print(f"  Individual mean ± SEM : {np.mean(sub):.3f} ± {sci_stats.sem(sub):.3f} g")
    print(f"  Individual median     : {np.median(sub):.3f} g")
    print(f"  Threshold-aligned mean: {al_thr:.3f} g  ← bias-corrected")
    print(f"  Group-mean fit        : {gf_thr:.3f} g  ← biased (ceiling effect)")
    if np.isfinite(al_thr) and np.isfinite(gf_thr):
        print(f"  Bias (group – aligned): {gf_thr - al_thr:+.3f} g")

print(f"\n{test_str}")

# Fingerpad comparison
print("\n" + "=" * 60)
print("  FINGERPAD COMPARISON  (Rolke 2006: 0.05–0.2 g)")
print("=" * 60)
fp_lo, fp_hi = 0.05, 0.20
for cond in CONDITIONS:
    sub = df_fit.loc[
        (df_fit["Condition"] == cond) & np.isfinite(df_fit["threshold"]),
        "threshold"
    ].values
    m = np.mean(sub)
    print(f"\n{cond}: mean threshold = {m:.3f} g")
    print(f"  vs fingerpad: {m/fp_hi:.1f}× – {m/fp_lo:.1f}×  less sensitive")

# Save threshold table
csv_path = os.path.join(OUT_DIR, "atd_thresholds.csv")
df_fit[["Subject", "Condition", "threshold"]].to_csv(csv_path, index=False)
print(f"\nThreshold table → {csv_path}")