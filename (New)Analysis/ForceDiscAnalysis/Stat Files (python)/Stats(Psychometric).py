"""
================================================================
Force Discrimination — Extended Weber Analysis
================================================================

Complements the existing FD analysis script by adding:
  (1) Data-driven Weber ratio scatter with per-subject points
  (2) Statistical test of sub-chance accuracy (binomial test vs 0.5)
  (3) Direction-specific analysis (comparison > ref vs comparison < ref)
      → tells us whether the reversal is symmetric or directional
  (4) Per-subject Weber fraction via psychometric fit
  (5) Group-level psychometric curve overlaying both reference groups

Outputs (saved into ./fd_outputs):
  - weber_scatter_data_driven.png
  - direction_specific_accuracy.png
  - per_subject_weber_fraction.csv
  - subchance_test_results.csv
"""

import os
import glob
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.optimize import curve_fit
from scipy import stats

warnings.filterwarnings("ignore")

# =============================================================================
# PNAS style + shared figure palette (Stats GEE / ATD C1_Figure)
# =============================================================================
SLATE_BLUE = "#56708A"
OLIVE      = "#686F12"
WINE       = "#7F212B"
CREAM      = "#EDE2D0"
BLACK      = "#1A1A1A"
REF_LINE   = WINE

REF_PALETTE = {1.0: SLATE_BLUE, 26.0: WINE}
DIR_PALETTE = {"heavier": OLIVE, "lighter": SLATE_BLUE}

from fd_export import FIG_SIZE, SAVE_DPI, save_figure_png

FONT_TICK   = 16
FONT_LABEL  = 14
FONT_TITLE  = 16
FONT_LEGEND = 14
FONT_ANNOT  = 15
FONT_WEIGHT = 900    # extra-bold for in-figure numbers and labels

WSPACE      = 0.06   # horizontal gap between side-by-side panels
FIT_LW      = 3.5
REF_LW      = 2.2
CHANCE_LW   = 1.8
ERR_LW      = 2.0
SCATTER_S   = 28
SCATTER_A   = 0.30


def _style_axes(ax):
    ax.tick_params(axis="both", labelsize=FONT_TICK)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight(FONT_WEIGHT)
    ax.xaxis.label.set_fontweight(FONT_WEIGHT)
    ax.yaxis.label.set_fontweight(FONT_WEIGHT)
    ax.title.set_fontweight(FONT_WEIGHT)


def _annotate_pair(ax, xy, text, color, y_off=8, side=None):
    if side is None:
        side = "left" if " / " in text else "right"
    if side == "left":
        x_off, ha = -8, "right"
    else:
        x_off, ha = 8, "left"
    ax.annotate(
        text, xy,
        xytext=(x_off, y_off),
        textcoords="offset points",
        fontsize=FONT_ANNOT,
        fontweight=FONT_WEIGHT,
        color=color,
        ha=ha,
        zorder=5,
    )

rcParams.update({
    "figure.facecolor":      "#FFFFFF",
    "axes.facecolor":        "#FFFFFF",
    "font.family":           "sans-serif",
    "font.sans-serif":       ["Helvetica", "Arial", "DejaVu Sans"],
    "axes.linewidth":        0.8,
    "axes.spines.top":       False,
    "axes.spines.right":     False,
    "xtick.major.width":     0.8,
    "ytick.major.width":     0.8,
    "xtick.major.size":      3.5,
    "ytick.major.size":      3.5,
    "xtick.direction":       "out",
    "ytick.direction":       "out",
    "legend.frameon":        False,
    "legend.fontsize":       FONT_LEGEND,
    "legend.title_fontsize": FONT_LEGEND,
    "font.size":             12,
    "axes.titlesize":        FONT_TITLE,
    "axes.labelsize":        FONT_LABEL,
    "xtick.labelsize":       FONT_TICK,
    "ytick.labelsize":       FONT_TICK,
    "axes.grid":             True,
    "axes.grid.axis":        "y",
    "grid.alpha":            0.35,
    "grid.linestyle":        "--",
    "grid.color":            SLATE_BLUE,
    "figure.dpi":            SAVE_DPI,
    "savefig.dpi":           SAVE_DPI,
})


# ============================================================
# CONFIG
# ============================================================
FILE_PATTERN = "/Users/kyungeunjung/NailFoldExp/Data/(FD)CurData/P*_ForceDiscrimination.csv"
OUTPUT_DIR = "/Users/kyungeunjung/NailFoldExp/(New)Analysis/ForceDiscAnalysis/Output/Stats(Psychometric)"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 1. LOAD + COMPUTE ACCURACY (same definition as existing code)
# ============================================================
def parse_force_num(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    return float(str(x).strip().lower().replace("g", ""))

def calc_accuracy(row):
    """correct = 'chose the stronger stimulus' (matches your paradigm)."""
    if row["UserChoice"] == 1:
        return 1 if row["FirstStim"] > row["SecondStim"] else 0
    if row["UserChoice"] == 2:
        return 1 if row["SecondStim"] > row["FirstStim"] else 0
    return 0


all_files = glob.glob(FILE_PATTERN)
if not all_files:
    raise FileNotFoundError(f"No files at {FILE_PATTERN}")

raw = pd.concat([pd.read_csv(f) for f in all_files], ignore_index=True)
sub_col = "SubjectID" if "SubjectID" in raw.columns else "Subject"

df = raw.copy()
df["IsCorrect"] = df.apply(calc_accuracy, axis=1)
df["Reference_n"] = df["Reference"].map(parse_force_num)
df["Comparison_n"] = df["Comparison"].map(parse_force_num)
df["FirstStim_n"] = df["FirstStim"].map(parse_force_num)
df["SecondStim_n"] = df["SecondStim"].map(parse_force_num)
df = df.dropna(subset=["Reference_n", "Comparison_n", "IsCorrect"])

# Weber ratio: |comparison - reference| / reference
df["weber_ratio"] = (df["Comparison_n"] - df["Reference_n"]).abs() / df["Reference_n"]
# Direction: did we test a comparison heavier (+) or lighter (-) than reference?
df["direction"] = np.where(df["Comparison_n"] > df["Reference_n"], "heavier", "lighter")

print(f"[1] Loaded {len(df)} trials from {df[sub_col].nunique()} subjects")
print(f"    Reference values: {sorted(df['Reference_n'].unique())}")
print(f"    Weber ratios: {sorted(df['weber_ratio'].round(2).unique())}")


def force_pair_label(ref, wr, source_df=df, comp_filter=None):
    """Return compact label of (min–max) force pairs for a given (ref, weber_ratio).

    comp_filter: None | 'heavier' | 'lighter' to restrict by comparison direction.
    Same `weber_ratio` can map to multiple physical pairs (e.g. ref=1, wr=0.4 →
    {(0.6, 1), (1, 1.4)}), so the label joins them with ' / '.
    """
    sub = source_df[
        (source_df["Reference_n"] == ref)
        & (np.isclose(source_df["weber_ratio"], wr, atol=1e-3))
    ]
    if comp_filter == "heavier":
        sub = sub[sub["Comparison_n"] > sub["Reference_n"]]
    elif comp_filter == "lighter":
        sub = sub[sub["Comparison_n"] < sub["Reference_n"]]
    pairs = set()
    for _, r in sub.iterrows():
        a, b = sorted([r["Reference_n"], r["Comparison_n"]])
        pairs.add((round(a, 2), round(b, 2)))
    return " / ".join(f"{a:g}–{b:g}" for a, b in sorted(pairs))


# ============================================================
# 2. DATA-DRIVEN WEBER SCATTER (replaces hardcoded version)
# ============================================================
# Per-subject accuracy per (reference, weber_ratio)
subj_agg = (df.groupby([sub_col, "Reference_n", "weber_ratio"])["IsCorrect"]
              .agg(["mean", "count"])
              .reset_index()
              .rename(columns={"mean": "accuracy", "count": "n_trials"}))

# Group-level (mean ± SEM across subjects)
grp_agg = (subj_agg.groupby(["Reference_n", "weber_ratio"])
                   .agg(mean_acc=("accuracy", "mean"),
                        sem_acc=("accuracy", lambda x: x.std(ddof=1)/np.sqrt(len(x))),
                        n_subj=("accuracy", "count"))
                   .reset_index())

print("\n[2] Group-level summary:")
print(grp_agg.round(3).to_string(index=False))


fig, ax = plt.subplots(figsize=FIG_SIZE, facecolor="#FFFFFF")
palette = REF_PALETTE

# (a) Per-subject jittered dots (translucent)
for ref, sub in subj_agg.groupby("Reference_n"):
    color = palette.get(ref, SLATE_BLUE)
    jitter = np.random.uniform(-0.015, 0.015, size=len(sub))
    ax.scatter(sub["weber_ratio"] + jitter, sub["accuracy"],
               s=25, alpha=0.25, color=color, edgecolor="none", zorder=2)

# (b) Group mean ± SEM, annotated with the actual force pair(s)
for ref, sub in grp_agg.groupby("Reference_n"):
    color = palette.get(ref, SLATE_BLUE)
    ax.errorbar(sub["weber_ratio"], sub["mean_acc"], yerr=sub["sem_acc"],
                marker="o", color=color, ms=10, lw=2, capsize=4,
                label=f"ref = {ref:g} g", zorder=4)
    for _, row in sub.iterrows():
        lbl = force_pair_label(ref, row["weber_ratio"])
        if not lbl:
            continue
        _annotate_pair(ax, (row["weber_ratio"], row["mean_acc"]), lbl, color)

ax.axhline(0.75, ls="--", color=REF_LINE, alpha=0.8, linewidth=1.0,
           label="75% threshold criterion")
ax.axhline(0.50, ls="-", color=BLACK, alpha=0.8, linewidth=0.8,
           label="Chance (2-AFC)")
ax.set_xlabel("Weber ratio |ΔF| / F_ref", fontsize=FONT_LABEL)
ax.set_ylabel("Accuracy ('chose the stronger')", fontsize=FONT_LABEL)
ax.set_title("Force discrimination as a function of Weber ratio (data-driven)",
              fontsize=FONT_TITLE)
_style_axes(ax)
ax.set_ylim(-0.02, 1.05)
ax.legend(loc="lower right", fontsize=FONT_LEGEND, prop={"weight": FONT_WEIGHT})
plt.tight_layout()
out1 = os.path.join(OUTPUT_DIR, "weber_scatter_data_driven.png")
save_figure_png(fig, out1)
plt.close(fig)
print(f"    Saved {out1}")


# ============================================================
# 3. SUB-CHANCE STATISTICAL TEST
# ============================================================
# For each (reference, weber_ratio), test H0: accuracy = 0.5 (chance)
# Using binomial test on total successes vs total trials (pooled across subjects)
print("\n[3] Binomial test against chance (0.5) per pair:")
test_records = []
for (ref, wr), grp in df.groupby(["Reference_n", "weber_ratio"]):
    n_success = int(grp["IsCorrect"].sum())
    n_total   = len(grp)
    if n_total == 0:
        continue
    # Two-sided binomial test
    try:
        res = stats.binomtest(n_success, n_total, p=0.5, alternative="two-sided")
        p_val = res.pvalue
    except AttributeError:
        # older scipy
        p_val = stats.binom_test(n_success, n_total, p=0.5, alternative="two-sided")
    acc = n_success / n_total
    direction = "above-chance" if acc > 0.5 else "BELOW-chance" if acc < 0.5 else "chance"
    test_records.append({
        "Reference": ref, "Weber_ratio": wr,
        "accuracy": acc, "n_trials": n_total, "n_success": n_success,
        "binomial_p": p_val, "verdict": direction,
        "significant_at_0.05": p_val < 0.05,
    })

test_df = pd.DataFrame(test_records).sort_values(["Reference", "Weber_ratio"])
test_df.to_csv(os.path.join(OUTPUT_DIR, "subchance_test_results.csv"), index=False)
print(test_df.round(4).to_string(index=False))

# Highlight any sub-chance significant cases
sub_chance = test_df[(test_df["accuracy"] < 0.5) & (test_df["significant_at_0.05"])]
if len(sub_chance):
    print("\n    ⚠ Significant sub-chance accuracy detected at:")
    for _, r in sub_chance.iterrows():
        print(f"      ref={r['Reference']:g}g, Weber={r['Weber_ratio']:.2f}: "
              f"acc={r['accuracy']:.2f}, p={r['binomial_p']:.4f}")
    print("    → Investigate direction-specific analysis below.")


# ============================================================
# 4. DIRECTION-SPECIFIC ANALYSIS
# ============================================================
# For each Weber ratio, split trials by direction (heavier vs lighter comparison)
# If reversal is direction-specific → asymmetric pattern
# If reversal is symmetric → general low-force confusion
print("\n[4] Direction-specific accuracy (heavier comp vs lighter comp):")

dir_agg = (df.groupby([sub_col, "Reference_n", "weber_ratio", "direction"])["IsCorrect"]
             .mean().reset_index().rename(columns={"IsCorrect": "accuracy"}))
dir_grp = (dir_agg.groupby(["Reference_n", "weber_ratio", "direction"])
                  .agg(mean_acc=("accuracy", "mean"),
                       sem_acc=("accuracy", lambda x: x.std(ddof=1)/np.sqrt(len(x))),
                       n_subj=("accuracy", "count"))
                  .reset_index())
print(dir_grp.round(3).to_string(index=False))


fig, axes = plt.subplots(1, 2, figsize=FIG_SIZE, sharey=True, facecolor="#FFFFFF")
for ax, ref in zip(axes, sorted(df["Reference_n"].unique())):
    sub = dir_grp[dir_grp["Reference_n"] == ref]
    for direction, dsub in sub.groupby("direction"):
        marker = "o" if direction == "heavier" else "s"
        color = DIR_PALETTE.get(direction, SLATE_BLUE)
        ax.errorbar(dsub["weber_ratio"], dsub["mean_acc"], yerr=dsub["sem_acc"],
                    marker=marker, color=color, ms=9, lw=2, capsize=4,
                    label=f"Comparison {direction} than ref")
        # annotate each dot with the actual force pair for this direction
        y_off = 9 if direction == "heavier" else -14
        for _, row in dsub.iterrows():
            lbl = force_pair_label(ref, row["weber_ratio"], comp_filter=direction)
            if not lbl:
                continue
            _annotate_pair(
                ax, (row["weber_ratio"], row["mean_acc"]), lbl, color, y_off=y_off
            )
    ax.axhline(0.75, ls="--", color=REF_LINE, alpha=0.8, linewidth=1.0)
    ax.axhline(0.50, ls="-", color=BLACK, alpha=0.8, linewidth=0.8)
    ax.set_xlabel("Weber ratio |ΔF| / F_ref", fontsize=FONT_LABEL)
    ax.set_ylabel("Accuracy ('chose the stronger')", fontsize=FONT_LABEL)
    ax.set_title(f"Reference = {ref:g} g", fontsize=FONT_TITLE)
    _style_axes(ax)
    ax.set_ylim(-0.02, 1.05)
    ax.legend(loc="lower right", fontsize=FONT_LEGEND, prop={"weight": FONT_WEIGHT})
fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.14, wspace=WSPACE)
out2 = os.path.join(OUTPUT_DIR, "direction_specific_accuracy.png")
save_figure_png(fig, out2)
plt.close(fig)
print(f"    Saved {out2}")

# Interpretation hint
print("\n    Interpretation guide:")
print("      - If BOTH directions show sub-chance at small Weber ratios → general confusion")
print("      - If ONLY ONE direction shows sub-chance → systematic perceptual bias")
print("      - e.g., if 'lighter' comparisons are misjudged as heavier → tendency to")
print("        report the BASELINE/reference as stronger (anchoring bias)")


# ============================================================
# 5. PSYCHOMETRIC FIT ON WEBER RATIOS
# ============================================================
def psychometric(x, k, slope, lapse=0.02, guess=0.5):
    """k = Weber fraction (75% threshold); slope = sharpness."""
    return guess + (1 - guess - lapse) / (1 + np.exp(-slope * (x - k)))

def fit_psychometric(x, y, p0=(0.5, 5.0), bounds=([0.01, 0.1], [3.0, 100])):
    try:
        popt, _ = curve_fit(psychometric, x, y, p0=p0, bounds=bounds, maxfev=5000)
        return popt
    except Exception:
        return (np.nan, np.nan)


# Group-level fit per reference
print("\n[5] Psychometric fits (group-level):")
fig, ax = plt.subplots(figsize=FIG_SIZE, facecolor="#FFFFFF")
x_smooth = np.linspace(0, 1.5, 200)

group_fits = {}
for ref, sub in subj_agg.groupby("Reference_n"):
    color = REF_PALETTE.get(ref, SLATE_BLUE)
    jitter = np.random.uniform(-0.015, 0.015, size=len(sub))
    ax.scatter(
        sub["weber_ratio"] + jitter, sub["accuracy"],
        s=SCATTER_S, alpha=SCATTER_A, color=color, edgecolor="none", zorder=2,
    )

for ref, sub in grp_agg.groupby("Reference_n"):
    color = REF_PALETTE.get(ref, SLATE_BLUE)
    k, slope = fit_psychometric(sub["weber_ratio"].values, sub["mean_acc"].values)
    group_fits[ref] = (k, slope)
    print(f"    ref={ref:g}g: Weber fraction k={k:.3f}, slope={slope:.2f}")

    ax.errorbar(
        sub["weber_ratio"], sub["mean_acc"], yerr=sub["sem_acc"],
        marker="o", color=color, ms=10, linestyle="none",
        elinewidth=ERR_LW, capsize=5, label=f"ref = {ref:g} g", zorder=4,
    )
    for _, row in sub.iterrows():
        lbl = force_pair_label(ref, row["weber_ratio"])
        if not lbl:
            continue
        _annotate_pair(ax, (row["weber_ratio"], row["mean_acc"]), lbl, color)
    if not np.isnan(k):
        ax.plot(
            x_smooth, psychometric(x_smooth, k, slope),
            color=color, lw=FIT_LW, alpha=0.9, zorder=3,
            label=f"ref={ref:g}g fit (k={k:.2f})",
        )

ax.axhline(0.75, ls="--", color=REF_LINE, alpha=0.85, linewidth=REF_LW, label="75% criterion")
ax.axhline(0.50, ls="-", color=BLACK, alpha=0.85, linewidth=CHANCE_LW, label="Chance")
ax.set_xlabel("Weber ratio |ΔF| / F_ref", fontsize=FONT_LABEL)
ax.set_ylabel("Accuracy", fontsize=FONT_LABEL)
ax.set_title("Psychometric fit per reference (Weber's law check)", fontsize=FONT_TITLE)
_style_axes(ax)
ax.set_ylim(-0.02, 1.05)
ax.legend(loc="lower right", fontsize=FONT_LEGEND, prop={"weight": FONT_WEIGHT})
plt.tight_layout()
out3 = os.path.join(OUTPUT_DIR, "weber_psychometric_fits.png")
save_figure_png(fig, out3)
plt.close(fig)
print(f"    Saved {out3}")


# ============================================================
# 6. PER-SUBJECT WEBER FRACTION
# ============================================================
print("\n[6] Per-subject Weber fraction extraction ...")
subj_records = []
for (subj, ref), grp in subj_agg.groupby([sub_col, "Reference_n"]):
    if grp["weber_ratio"].nunique() < 3:
        continue
    k, slope = fit_psychometric(grp["weber_ratio"].values, grp["accuracy"].values)
    subj_records.append({
        "Subject": subj, "Reference": ref,
        "weber_fraction": k, "slope": slope,
        "n_levels": grp["weber_ratio"].nunique()
    })

subj_df = pd.DataFrame(subj_records)
subj_df.to_csv(os.path.join(OUTPUT_DIR, "per_subject_weber_fraction.csv"), index=False)
print(f"    Saved per_subject_weber_fraction.csv ({len(subj_df)} fits)")

# Paired comparison: same subject, ref=1g vs ref=26g
wide = subj_df.pivot(index="Subject", columns="Reference",
                     values="weber_fraction").dropna()
if len(wide.columns) >= 2 and len(wide) >= 5:
    r1, r2 = sorted(wide.columns)
    try:
        stat, p = stats.wilcoxon(wide[r1], wide[r2])
        print(f"\n    Paired Wilcoxon (within-subject, ref={r1}g vs ref={r2}g):")
        print(f"      W = {stat:.2f}, p = {p:.4f}")
        print(f"      Median Weber fraction @ {r1}g = {wide[r1].median():.3f}")
        print(f"      Median Weber fraction @ {r2}g = {wide[r2].median():.3f}")
        if p >= 0.05:
            print(f"      → No reliable difference → Weber's law SUPPORTED (within subject).")
        else:
            print(f"      → Reference-dependent deviation from Weber's law.")
    except Exception as e:
        print(f"    Paired test failed: {e}")


print("\n=== Extended Weber analysis done. ===")