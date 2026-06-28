"""
Region Breakdown -- Is the reversal uniform across regions, or driven
by specific ones (e.g. C/D vs A/F)?
==========================================================================
Uses the same accuracy definition as the GEE pairwise script (Reference,
Comparison, ChoseComparison columns), but keeps Region as a grouping
variable instead of collapsing across all 6 regions.

Produces:
  A) region_pair_heatmap.png   -- Region (rows) x Force pair (cols) mean
                                   accuracy, both bands, diverging colormap
                                   centered on chance (50%) so reversed
                                   pairs jump out visually regardless of
                                   region.
  B) region_breakdown_difficult.png -- per-region boxplots (with subject
                                   dots) for just the 3 anomalous pairs,
                                   to see whether some regions are clearly
                                   better/worse than others within them.

How to read this:
  - If the reversal shows up at roughly the same depth in ALL 6 regions
    (including on-nail C/D and off-nail A/F alike), that's consistent with
    a global procedural cause (e.g. the missing timing cue) that doesn't
    care which skin site was stimulated.
  - If only some regions show the reversal while others look fine, that
    points to something region-specific instead (e.g. a particular site
    being harder to apply the filament to consistently).
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FILE_PATTERN = "/Users/kyungeunjung/NailFoldExp/Data/(FD)CurData/P*_ForceDiscrimination.csv"
OUTPUT_DIR = "/Users/kyungeunjung/NailFoldExp/(New)Analysis/ForceDiscAnalysis/Output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CHANCE_PCT = 50.0
JND_PCT = 75.0

# ── Load ─────────────────────────────────────────────────────────────────
files = glob.glob(FILE_PATTERN)
if not files:
    raise FileNotFoundError(f"No CSV files found matching: {FILE_PATTERN}")
df = pd.concat([pd.read_csv(f, encoding="utf-8-sig") for f in sorted(files)], ignore_index=True)
print(f"Participants loaded: {df['Subject'].nunique()}")
print(f"Total trials        : {len(df)}")

# Same accuracy definition as the GEE script
df["correct"] = np.where(
    df["Comparison"] > df["Reference"],
    df["ChoseComparison"] == 1,
    df["ChoseComparison"] == 0,
).astype(int)

df["pair_label"] = df.apply(
    lambda r: f"{min(r['Reference'], r['Comparison']):g}-{max(r['Reference'], r['Comparison']):g}",
    axis=1,
)
df["band"] = df["Reference"].apply(lambda r: "Low" if r == 1 else "High")

low_order = ["0.4-1", "0.6-1", "1-1.4", "1-2"]
high_order = ["10-26", "15-26", "26-60"]
region_order = ["A", "B", "C", "D", "E", "F"]

DIFFICULT_PAIRS = ["0.6-1", "1-1.4", "15-26"]

# Per-subject, per-region, per-pair accuracy
subj_region_acc = (
    df.groupby(["Subject", "Region", "band", "pair_label"])["correct"]
      .mean()
      .reset_index()
      .rename(columns={"correct": "accuracy"})
)
subj_region_acc["accuracy_pct"] = subj_region_acc["accuracy"] * 100

print("\nTrials per (Subject x Region x pair) cell:")
print(df.groupby(["Region", "pair_label"]).size().unstack(fill_value=0))

# ═══════════════════════════════════════════════════════════════════════════
# A) Region x Pair heatmap (mean accuracy across subjects)
# ═══════════════════════════════════════════════════════════════════════════
all_pairs_order = low_order + high_order
heat = (
    subj_region_acc.groupby(["Region", "pair_label"])["accuracy_pct"]
    .mean()
    .reset_index()
    .pivot(index="Region", columns="pair_label", values="accuracy_pct")
    .reindex(index=region_order, columns=all_pairs_order)
)

fig, ax = plt.subplots(figsize=(10, 6))
im = ax.imshow(heat.values, cmap="RdBu_r", vmin=0, vmax=100, aspect="auto")

for i in range(heat.shape[0]):
    for j in range(heat.shape[1]):
        val = heat.values[i, j]
        if np.isnan(val):
            continue
        color = "white" if (val > 70 or val < 30) else "black"
        ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=10,
                fontweight="bold", color=color)

ax.set_xticks(range(len(all_pairs_order)))
ax.set_xticklabels(all_pairs_order, rotation=45, ha="right", fontsize=10)
ax.set_yticks(range(len(region_order)))
ax.set_yticklabels(region_order, fontsize=11)
ax.set_xlabel("Force pair (g)", fontsize=11)
ax.set_ylabel("Region", fontsize=11)
ax.set_title("Mean Accuracy (%) by Region x Force Pair\n(centered on chance = 50%, red = below chance)",
             fontsize=12, fontweight="bold", pad=12)

# mark the boundary between low/high band visually
ax.axvline(len(low_order) - 0.5, color="black", lw=1.5)

cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
cbar.set_label("Accuracy (%)", fontsize=10)

plt.tight_layout()
heatmap_path = f"{OUTPUT_DIR}/region_pair_heatmap.png"
fig.savefig(heatmap_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved -> {heatmap_path}")

# ═══════════════════════════════════════════════════════════════════════════
# B) Per-region boxplots, difficult pairs only
# ═══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, len(DIFFICULT_PAIRS), figsize=(5.2 * len(DIFFICULT_PAIRS), 5.5), sharey=True)

for ax, pair in zip(axes, DIFFICULT_PAIRS):
    sub = subj_region_acc[subj_region_acc["pair_label"] == pair]
    box_data = [sub.loc[sub["Region"] == r, "accuracy_pct"].values for r in region_order]
    bp = ax.boxplot(box_data, positions=range(len(region_order)), widths=0.55,
                     patch_artist=True, showmeans=False)
    for patch in bp["boxes"]:
        patch.set_facecolor("#dde6f0")
        patch.set_edgecolor("black")
    for median in bp["medians"]:
        median.set_color("#c0392b")
        median.set_linewidth(2)

    for i, r in enumerate(region_order):
        vals = sub.loc[sub["Region"] == r, "accuracy_pct"].values
        jitter = np.random.uniform(-0.1, 0.1, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, color="#2166AC", alpha=0.6, s=20, zorder=3)

        # Count dots at 0, 50, 100 and annotate
        for y_val, v_offset in [(0, 3.5), (50, 3.5), (100, 3.5)]:
            n = np.sum(np.abs(vals - y_val) < 1e-6)
            if n > 0:
                ax.text(i, y_val + v_offset, f"n={n}",
                        ha="center", va="bottom", fontsize=7.5,
                        color="#2166AC", fontweight="bold", zorder=5)

    ax.axhline(CHANCE_PCT, color="gray", ls=":", lw=1.3)
    ax.axhline(JND_PCT, color="black", ls="--", lw=1.3)
    ax.set_xticks(range(len(region_order)))
    ax.set_xticklabels(region_order, fontsize=11)
    ax.set_title(f"Pair {pair}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Region", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)

axes[0].set_ylabel("Accuracy (%)", fontsize=11)
axes[0].set_ylim(0, 105)
fig.suptitle("Accuracy by Region -- Anomalous (Below-Chance) Pairs Only", fontsize=13, fontweight="bold", y=1.02)

plt.tight_layout()
diff_path = f"{OUTPUT_DIR}/region_breakdown_difficult.png"
fig.savefig(diff_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved -> {diff_path}")

# ═══════════════════════════════════════════════════════════════════════════
# Quick numeric summary printed to console
# ═══════════════════════════════════════════════════════════════════════════
print("\nMean accuracy (%) by region, difficult pairs only:")
print(
    subj_region_acc[subj_region_acc["pair_label"].isin(DIFFICULT_PAIRS)]
    .groupby(["pair_label", "Region"])["accuracy_pct"]
    .mean()
    .unstack()
    .reindex(columns=region_order)
    .round(1)
)

on_nail = subj_region_acc[subj_region_acc["Region"].isin(["C", "D"])]
off_nail = subj_region_acc[subj_region_acc["Region"].isin(["A", "F"])]
print("\nOn-nail (C+D) vs Off-nail (A+F) mean accuracy, difficult pairs only:")
for pair in DIFFICULT_PAIRS:
    on_mean = on_nail.loc[on_nail["pair_label"] == pair, "accuracy_pct"].mean()
    off_mean = off_nail.loc[off_nail["pair_label"] == pair, "accuracy_pct"].mean()
    print(f"  {pair:<8} On-nail: {on_mean:5.1f}%   Off-nail: {off_mean:5.1f}%")