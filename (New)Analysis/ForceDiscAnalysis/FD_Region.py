"""
================================================================
Force Discrimination — Region × ForcePair Alternative Visualizations
================================================================

Produces multiple complementary visualizations of how force
discrimination accuracy varies across the six periungual
sub-regions (A-F) and seven canonical force pairs.

Key fix vs. earlier heatmap:
  Uses Weber ratio = |Comparison - Reference| / Reference
  (matching the canonical convention used throughout the paper),
  rather than (f_hi - f_lo) / f_lo. This re-orders force pairs
  so that sub-chance pairs (WR ≈ 0.4) and above-criterion pairs
  (WR ≥ 0.6) cluster correctly within each band.

Outputs (saved to OUTPUT_DIR):
  - fd_region_lineplot_by_weber.png       Psychometric curves per region
  - fd_region_bar_per_pair.png            Grouped bars per pair
  - fd_region_violin_per_pair.png         Violin plot per pair
  - fd_region_heatmap_fixed.png           Heatmap with proper Weber ordering
  - fd_region_slope_subjects.png          Per-subject slope plot
  - fd_region_summary_fixed.csv           Summary table
"""

import os
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")


# ============================================================
# 1. Paths
# ============================================================
FILE_PATTERN = "/Users/kyungeunjung/NailFoldExp/Data/(FD)CurData/P*_ForceDiscrimination.csv"
OUTPUT_DIR = "/Users/kyungeunjung/NailFoldExp/(New)Analysis/ForceDiscAnalysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CHANCE_LEVEL = 0.50
JND_CRITERION = 0.75


# ============================================================
# 2. Load & preprocess (with FIXED Weber ratio)
# ============================================================
files = glob.glob(FILE_PATTERN)
if not files:
    raise FileNotFoundError(f"No files found at {FILE_PATTERN}")

raw = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
sub_col = "SubjectID" if "SubjectID" in raw.columns else "Subject"


def calc_accuracy(row):
    if row["UserChoice"] == 1:
        return 1 if row["FirstStim"] > row["SecondStim"] else 0
    if row["UserChoice"] == 2:
        return 1 if row["SecondStim"] > row["FirstStim"] else 0
    return np.nan


df = raw.copy()
df["IsCorrect"] = df.apply(calc_accuracy, axis=1)
df["Reference"] = pd.to_numeric(df["Reference"], errors="coerce")
df["Comparison"] = pd.to_numeric(df["Comparison"], errors="coerce")
df = df.dropna(subset=["IsCorrect", "Reference", "Comparison", "Region"])

# CANONICAL Weber ratio (matching the paper convention)
df["WeberRatio"] = (df["Comparison"] - df["Reference"]).abs() / df["Reference"]

# Force pair label with f_lo--f_hi
df["f_lo"] = df[["Reference", "Comparison"]].min(axis=1)
df["f_hi"] = df[["Reference", "Comparison"]].max(axis=1)
df["ForcePair"] = df.apply(lambda r: f"{r['f_lo']:g}--{r['f_hi']:g}", axis=1)
df["Band"] = np.where(df["Reference"] >= 10, "High (ref=26g)", "Low (ref=1g)")

print(f"[1] Loaded {len(df)} trials from {df[sub_col].nunique()} subjects.")
print(f"    Regions: {sorted(df['Region'].unique())}")
print(f"    ForcePairs and Weber ratios:")
pair_meta = (df.groupby(["Band", "ForcePair", "WeberRatio"]).size()
                .reset_index(name="n_trials")
                .sort_values(["Band", "WeberRatio"]))
print(pair_meta.to_string(index=False))


# ============================================================
# 3. Per-subject accuracy per (Region, ForcePair)
# ============================================================
subj_acc = (
    df.groupby([sub_col, "Region", "ForcePair", "Band", "WeberRatio"])["IsCorrect"]
    .mean()
    .reset_index()
    .rename(columns={"IsCorrect": "accuracy"})
)

grp_summary = (
    subj_acc.groupby(["Region", "ForcePair", "Band", "WeberRatio"])
    .agg(
        mean_acc=("accuracy", "mean"),
        sem_acc=("accuracy", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
        n_subj=("accuracy", "count"),
    )
    .reset_index()
)
grp_summary.to_csv(os.path.join(OUTPUT_DIR, "fd_region_summary_fixed.csv"), index=False)


# ============================================================
# 4. Canonical ordering by Weber ratio (within band)
# ============================================================
pair_order_low = (
    grp_summary[grp_summary["Band"] == "Low (ref=1g)"]
    .drop_duplicates("ForcePair")
    .sort_values("WeberRatio")["ForcePair"]
    .tolist()
)
pair_order_high = (
    grp_summary[grp_summary["Band"] == "High (ref=26g)"]
    .drop_duplicates("ForcePair")
    .sort_values("WeberRatio")["ForcePair"]
    .tolist()
)
pair_order_all = pair_order_low + pair_order_high

print(f"\n[2] Pair order (Low band): {pair_order_low}")
print(f"    Pair order (High band): {pair_order_high}")

region_order = ["A", "B", "C", "D", "E", "F"]
region_palette = {
    "A": "#e74c3c",
    "B": "#e67e22",
    "C": "#27ae60",
    "D": "#16a085",
    "E": "#3498db",
    "F": "#9b59b6",
}


# ============================================================
# Visualization 1 — Line plot: psychometric curves per region
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)

for ax, (band_name, pair_order) in zip(
    axes,
    [("Low (ref=1g)", pair_order_low), ("High (ref=26g)", pair_order_high)],
):
    sub = grp_summary[grp_summary["Band"] == band_name]
    for region in region_order:
        rdata = sub[sub["Region"] == region].sort_values("WeberRatio")
        if len(rdata) == 0:
            continue
        ax.errorbar(
            rdata["WeberRatio"],
            rdata["mean_acc"],
            yerr=rdata["sem_acc"],
            marker="o",
            ms=8,
            lw=2,
            capsize=3,
            color=region_palette[region],
            label=region,
        )
    ax.axhline(JND_CRITERION, color="red", ls="--", lw=1.5, alpha=0.7)
    ax.axhline(CHANCE_LEVEL, color="gray", ls=":", lw=1.2, alpha=0.7)
    ax.set_xlabel(r"Weber ratio $|\Delta F|/F_{\mathrm{ref}}$", fontsize=12)
    ax.set_ylabel("Mean accuracy" if band_name.startswith("Low") else "")
    ax.set_title(band_name, fontsize=13)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.grid(alpha=0.3)
    if band_name.startswith("Low"):
        ax.legend(title="Region", loc="lower right", fontsize=10, ncol=2)

fig.suptitle("Psychometric curves per region (mean ± SEM)", fontsize=14, y=1.02)
plt.tight_layout()
out1 = os.path.join(OUTPUT_DIR, "fd_region_lineplot_by_weber.png")
plt.savefig(out1, dpi=200, bbox_inches="tight")
plt.close()
print(f"\n[3] Saved {out1}")


# ============================================================
# Visualization 2 — Bar plot: grouped bars per pair
# ============================================================
fig, ax = plt.subplots(figsize=(15, 6))

# Need to compute mean per (Region, ForcePair) and add error bars
plot_data = grp_summary.copy()
plot_data["pair_idx"] = plot_data["ForcePair"].map(
    {p: i for i, p in enumerate(pair_order_all)}
)
plot_data = plot_data.sort_values("pair_idx")

bar_width = 0.13
n_pairs = len(pair_order_all)
x_centers = np.arange(n_pairs)

for i, region in enumerate(region_order):
    rdata = plot_data[plot_data["Region"] == region].sort_values("pair_idx")
    offsets = x_centers + (i - len(region_order) / 2 + 0.5) * bar_width
    ax.bar(
        offsets,
        rdata["mean_acc"],
        bar_width,
        yerr=rdata["sem_acc"],
        color=region_palette[region],
        edgecolor="black",
        linewidth=0.5,
        capsize=2,
        label=region,
    )

ax.axhline(JND_CRITERION, color="red", ls="--", lw=2, alpha=0.7,
           label=f"{int(JND_CRITERION*100)}% criterion")
ax.axhline(CHANCE_LEVEL, color="gray", ls=":", lw=1.5, alpha=0.7, label="Chance (50%)")

# Band separator
boundary = len(pair_order_low) - 0.5
ax.axvline(boundary, color="black", lw=2.5)
ax.text((len(pair_order_low) - 1) / 2, 1.08, "Low band (ref=1g)",
         ha="center", fontsize=12, fontweight="bold")
ax.text(len(pair_order_low) + (len(pair_order_high) - 1) / 2, 1.08,
         "High band (ref=26g)", ha="center", fontsize=12, fontweight="bold")

ax.set_xticks(x_centers)
ax.set_xticklabels(pair_order_all)
ax.set_xlabel("Force pair (ordered by Weber ratio within band)", fontsize=12)
ax.set_ylabel("Mean accuracy", fontsize=12)
ax.set_title("Force discrimination accuracy by Region × ForcePair (grouped bars)",
              fontsize=13, pad=20)
ax.set_ylim(0, 1.15)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.legend(title="Region", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=10)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
out2 = os.path.join(OUTPUT_DIR, "fd_region_bar_per_pair.png")
plt.savefig(out2, dpi=200, bbox_inches="tight")
plt.close()
print(f"[4] Saved {out2}")


# ============================================================
# Visualization 3 — Violin plot per pair (distribution shape)
# ============================================================
fig, ax = plt.subplots(figsize=(16, 7))

sns.violinplot(
    data=subj_acc,
    x="ForcePair",
    y="accuracy",
    hue="Region",
    hue_order=region_order,
    order=pair_order_all,
    palette=region_palette,
    inner="quartile",
    cut=0,
    linewidth=0.7,
    ax=ax,
)

ax.axhline(JND_CRITERION, color="red", ls="--", lw=2, alpha=0.7)
ax.axhline(CHANCE_LEVEL, color="gray", ls=":", lw=1.5, alpha=0.7)

boundary = len(pair_order_low) - 0.5
ax.axvline(boundary, color="black", lw=2.5)

ax.set_xlabel("Force pair", fontsize=12)
ax.set_ylabel("Accuracy (per subject)", fontsize=12)
ax.set_title("Per-subject accuracy distribution shape (violin)", fontsize=13)
ax.set_ylim(-0.05, 1.10)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.legend(title="Region", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=10)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
out3 = os.path.join(OUTPUT_DIR, "fd_region_violin_per_pair.png")
plt.savefig(out3, dpi=200, bbox_inches="tight")
plt.close()
print(f"[5] Saved {out3}")


# ============================================================
# Visualization 4 — Heatmap with PROPER Weber ratio ordering
# ============================================================
heatmap_data = (
    grp_summary.pivot(index="Region", columns="ForcePair", values="mean_acc")
    .reindex(region_order)
    .reindex(columns=pair_order_all)
)

fig, ax = plt.subplots(figsize=(13, 5.5))
sns.heatmap(
    heatmap_data,
    annot=True,
    fmt=".2f",
    cmap="RdYlGn",
    vmin=0,
    vmax=1,
    center=0.5,
    cbar_kws={"label": "Mean accuracy"},
    linewidths=0.6,
    linecolor="white",
    ax=ax,
)

# Band boundary
boundary = len(pair_order_low)
ax.axvline(boundary, color="black", lw=3)

# Band labels above heatmap
ax.text(boundary / 2, -0.6, "Low band (ref=1g)",
         ha="center", va="bottom", fontsize=12, fontweight="bold")
ax.text(boundary + len(pair_order_high) / 2, -0.6, "High band (ref=26g)",
         ha="center", va="bottom", fontsize=12, fontweight="bold")

# Add Weber ratio annotations below x-labels
xtick_labels = []
for p in pair_order_all:
    wr = grp_summary[grp_summary["ForcePair"] == p]["WeberRatio"].iloc[0]
    xtick_labels.append(f"{p}\n(WR={wr:.2f})")
ax.set_xticklabels(xtick_labels, rotation=0, fontsize=10)

ax.set_xlabel("Force pair (ordered by Weber ratio within band)", fontsize=12)
ax.set_ylabel("Region", fontsize=12)
ax.set_title("Mean accuracy heatmap: Region × ForcePair (Weber-ratio ordered)",
              fontsize=13, pad=25)
plt.tight_layout()
out4 = os.path.join(OUTPUT_DIR, "fd_region_heatmap_fixed.png")
plt.savefig(out4, dpi=200, bbox_inches="tight")
plt.close()
print(f"[6] Saved {out4}")


# ============================================================
# Visualization 5 — Per-subject slope plots (paired)
# ============================================================
# Show whether individual subjects are consistent across regions
# Use 2 representative pairs: one sub-chance + one above-criterion

fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=True)
axes_flat = axes.flatten()

# Pick representative pairs from each band, one sub-chance and one above
representative_pairs = [
    ("0.6--1", "Low band, sub-chance"),
    ("0.4--1", "Low band, above-criterion"),
    ("15--26", "High band, sub-chance"),
    ("10--26", "High band, above-criterion"),
]

for ax, (pair, title) in zip(axes_flat, representative_pairs):
    sub = subj_acc[subj_acc["ForcePair"] == pair]
    if len(sub) == 0:
        ax.text(0.5, 0.5, f"No data for {pair}",
                ha="center", va="center", transform=ax.transAxes)
        continue

    for subj_id, gr in sub.groupby(sub_col):
        gr_sorted = gr.set_index("Region").reindex(region_order).reset_index()
        ax.plot(
            gr_sorted["Region"],
            gr_sorted["accuracy"],
            "-o",
            color="gray",
            alpha=0.4,
            ms=4,
            lw=0.8,
        )

    # Overlay mean
    mean_by_region = (sub.groupby("Region")["accuracy"]
                         .mean()
                         .reindex(region_order))
    ax.plot(
        mean_by_region.index,
        mean_by_region.values,
        "-s",
        color="red",
        lw=2.5,
        ms=10,
        label="Mean across subjects",
    )

    ax.axhline(JND_CRITERION, color="red", ls="--", lw=1, alpha=0.6)
    ax.axhline(CHANCE_LEVEL, color="gray", ls=":", lw=1, alpha=0.6)
    ax.set_title(f"{pair}\n{title}", fontsize=11)
    ax.set_ylim(-0.05, 1.10)
    ax.set_xlabel("Region")
    ax.set_ylabel("Accuracy" if ax in axes[:, 0] else "")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)

fig.suptitle("Per-subject accuracy across regions (representative pairs)",
              fontsize=14, y=1.00)
plt.tight_layout()
out5 = os.path.join(OUTPUT_DIR, "fd_region_slope_subjects.png")
plt.savefig(out5, dpi=200, bbox_inches="tight")
plt.close()
print(f"[7] Saved {out5}")


# ============================================================
# Console summary
# ============================================================
print("\n" + "=" * 70)
print("=== Alternative visualization suite complete ===")
print("=" * 70)
print(f"Output directory: {OUTPUT_DIR}/")
for f_name in sorted(os.listdir(OUTPUT_DIR)):
    if f_name.startswith("fd_region"):
        print(f"  - {f_name}")