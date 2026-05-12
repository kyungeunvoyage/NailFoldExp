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
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from scipy import stats

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")


# ============================================================
# CONFIG
# ============================================================
FILE_PATTERN = "/Users/kyungeunjung/NailFoldExp/Data/(FD)CurData/P*_ForceDiscrimination.csv"
OUTPUT_DIR = "./fd_outputs"
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
df = df.dropna(subset=["Reference_n", "Comparison_n", "IsCorrect", "FirstStim_n", "SecondStim_n"])

# Weber ratio: |comparison - reference| / reference
df["weber_ratio"] = (df["Comparison_n"] - df["Reference_n"]).abs() / df["Reference_n"]
# Direction: did we test a comparison heavier (+) or lighter (-) than reference?
df["direction"] = np.where(df["Comparison_n"] > df["Reference_n"], "heavier", "lighter")

# 첫 번째 자극이 강한지/약한지로 분리
df["first_is_heavier"] = (df["FirstStim_n"] > df["SecondStim_n"]).astype(int)
order_check = (
    df.groupby(["weber_ratio", "first_is_heavier"])["IsCorrect"]
    .mean()
    .reset_index()
)
# Note: same displayed weber_ratio can pool trials from different Reference_n (e.g. 1g vs 26g).
print("\n[1b] Mean accuracy by Weber ratio × first stimulus heavier (1) vs lighter (0):")
print(order_check.sort_values(["weber_ratio", "first_is_heavier"]).to_string(index=False))
print(f"\n[1] Loaded {len(df)} trials from {df[sub_col].nunique()} subjects")
print(f"    Reference values: {sorted(df['Reference_n'].unique())}")
print(f"    Weber ratios: {sorted(df['weber_ratio'].round(2).unique())}")


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


fig, ax = plt.subplots(figsize=(9, 5.5))
palette = {1.0: "#1f77b4", 26.0: "#d62728"}

# (a) Per-subject jittered dots (translucent)
for ref, sub in subj_agg.groupby("Reference_n"):
    color = palette.get(ref, "gray")
    jitter = np.random.uniform(-0.015, 0.015, size=len(sub))
    ax.scatter(sub["weber_ratio"] + jitter, sub["accuracy"],
               s=25, alpha=0.25, color=color, edgecolor="none", zorder=2)

# (b) Group mean ± SEM
for ref, sub in grp_agg.groupby("Reference_n"):
    color = palette.get(ref, "gray")
    ax.errorbar(sub["weber_ratio"], sub["mean_acc"], yerr=sub["sem_acc"],
                marker="o", color=color, ms=10, lw=2, capsize=4,
                label=f"ref = {ref:g} g", zorder=4)

ax.axhline(0.75, ls="--", color="red", alpha=0.7, label="75% threshold criterion")
ax.axhline(0.50, ls=":",  color="gray", alpha=0.7, label="Chance (2-AFC)")
ax.set_xlabel("Weber ratio |ΔF| / F_ref", fontsize=12)
ax.set_ylabel("Accuracy ('chose the stronger')", fontsize=12)
ax.set_title("Force discrimination as a function of Weber ratio (data-driven)",
             fontsize=12)
ax.set_ylim(-0.02, 1.05)
ax.legend(fontsize=9, loc="lower right")
ax.grid(alpha=0.3)
plt.tight_layout()
out1 = os.path.join(OUTPUT_DIR, "weber_scatter_data_driven.png")
plt.savefig(out1, dpi=200, bbox_inches="tight")
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


fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
for ax, ref in zip(axes, sorted(df["Reference_n"].unique())):
    sub = dir_grp[dir_grp["Reference_n"] == ref]
    for direction, dsub in sub.groupby("direction"):
        marker = "o" if direction == "heavier" else "s"
        color = "#2ca02c" if direction == "heavier" else "#9467bd"
        ax.errorbar(dsub["weber_ratio"], dsub["mean_acc"], yerr=dsub["sem_acc"],
                    marker=marker, color=color, ms=9, lw=2, capsize=4,
                    label=f"Comparison {direction} than ref")
    ax.axhline(0.75, ls="--", color="red", alpha=0.6)
    ax.axhline(0.50, ls=":",  color="gray", alpha=0.6)
    ax.set_xlabel("Weber ratio |ΔF| / F_ref")
    ax.set_ylabel("Accuracy ('chose the stronger')")
    ax.set_title(f"Reference = {ref:g} g")
    ax.set_ylim(-0.02, 1.05)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
plt.tight_layout()
out2 = os.path.join(OUTPUT_DIR, "direction_specific_accuracy.png")
plt.savefig(out2, dpi=200, bbox_inches="tight")
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
fig, ax = plt.subplots(figsize=(8.5, 5.5))
x_smooth = np.linspace(0, 1.5, 200)

group_fits = {}
for ref, sub in grp_agg.groupby("Reference_n"):
    color = palette.get(ref, "gray")
    k, slope = fit_psychometric(sub["weber_ratio"].values, sub["mean_acc"].values)
    group_fits[ref] = (k, slope)
    print(f"    ref={ref:g}g: Weber fraction k={k:.3f}, slope={slope:.2f}")

    # plot data
    ax.errorbar(sub["weber_ratio"], sub["mean_acc"], yerr=sub["sem_acc"],
                marker="o", color=color, ms=10, lw=0, capsize=4,
                label=f"ref={ref:g}g (data)")
    # plot fit
    if not np.isnan(k):
        ax.plot(x_smooth, psychometric(x_smooth, k, slope),
                color=color, lw=2, alpha=0.8,
                label=f"ref={ref:g}g fit (k={k:.2f})")

ax.axhline(0.75, ls="--", color="red", alpha=0.6, label="75% criterion")
ax.axhline(0.50, ls=":",  color="gray", alpha=0.6, label="Chance")
ax.set_xlabel("Weber ratio |ΔF| / F_ref", fontsize=12)
ax.set_ylabel("Accuracy", fontsize=12)
ax.set_title("Psychometric fit per reference (Weber's law check)", fontsize=12)
ax.set_ylim(-0.02, 1.05)
ax.legend(fontsize=8, loc="lower right")
ax.grid(alpha=0.3)
plt.tight_layout()
out3 = os.path.join(OUTPUT_DIR, "weber_psychometric_fits.png")
plt.savefig(out3, dpi=200, bbox_inches="tight")
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