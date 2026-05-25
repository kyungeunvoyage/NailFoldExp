"""
Force Discrimination – Perceptual Reversal Visualization
=========================================================
Works with a single CSV or a folder of 25 participant CSVs.

Column assumptions
------------------
Subject, Reference (g), Comparison (g), ChoseComparison (1/0)
  ChoseComparison=1  → participant said Comparison was stronger
  ChoseComparison=0  → participant said Reference was stronger

Accuracy = proportion of trials where the LARGER force was correctly chosen.
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from pathlib import Path

# ── 0. Matplotlib style ─────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
})

# ── 1. Load data ─────────────────────────────────────────────────────────────
FILE_PATTERN = "/Users/kyungeunjung/NailFoldExp/Data/(FD)CurData/P*_ForceDiscrimination.csv"
OUTPUT_DIR = "./fd_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

files = glob.glob(FILE_PATTERN)
if not files:
    raise FileNotFoundError(f"No CSV files found matching: {FILE_PATTERN}")
print(f"Loaded {len(files)} participant file(s).")

df = pd.concat(
    [pd.read_csv(f, encoding="utf-8-sig") for f in sorted(files)],
    ignore_index=True,
)

# ── 2. Derived columns ────────────────────────────────────────────────────────
# Accuracy: did the participant choose the physically larger stimulus?
df["correct"] = np.where(
    df["Comparison"] > df["Reference"],
    df["ChoseComparison"] == 1,
    df["ChoseComparison"] == 0,
).astype(int)

# Canonical pair label  (smaller–larger)
df["pair_label"] = df.apply(
    lambda r: f"{min(r['Reference'], r['Comparison']):g}–"
              f"{max(r['Reference'], r['Comparison']):g}",
    axis=1,
)

# Weber ratio  |ΔF| / F_ref
df["weber_ratio"] = (
    (df["Comparison"] - df["Reference"]).abs() / df["Reference"]
)

# Band
df["band"] = df["Reference"].apply(
    lambda r: "Low band  (ref = 1 g)" if r == 1 else "High band  (ref = 26 g)"
)

# "Chose lighter" = reversal indicator (1 when lighter was called stronger)
df["chose_lighter"] = np.where(
    df["Comparison"] < df["Reference"],
    df["ChoseComparison"] == 1,
    df["ChoseComparison"] == 0,
).astype(int)

# ── 3. Per-subject accuracy per force pair ────────────────────────────────────
subj_acc = (
    df.groupby(["Subject", "band", "pair_label", "weber_ratio"])["correct"]
    .mean()
    .reset_index()
    .rename(columns={"correct": "accuracy"})
)

subj_rev = (
    df.groupby(["Subject", "band", "pair_label", "weber_ratio"])["chose_lighter"]
    .mean()
    .reset_index()
    .rename(columns={"chose_lighter": "reversal_rate"})
)

# ── 4. Ordered pair lists ─────────────────────────────────────────────────────
low_order  = ["0.4–1", "0.6–1", "1–1.4", "1–2"]
high_order = ["10–26", "15–26", "26–60"]

# Colour logic: above JND criterion (≥0.75) → teal
#               reversal zone (<0.50)        → salmon
#               in between                   → steel-blue
TEAL   = "#2a9d8f"
SALMON = "#e76f51"
MID    = "#457b9d"

def pair_color(pair, band_df):
    med = band_df.loc[band_df["pair_label"] == pair, "accuracy"].mean()
    if med >= 0.75:
        return TEAL
    elif med < 0.50:
        return SALMON
    return MID

# ── 5. Figure layout ──────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 14))
fig.suptitle(
    "Perceptual Reversal in Periungual Force Discrimination",
    fontsize=15, fontweight="bold", y=0.98,
)

gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.35)
ax_low  = fig.add_subplot(gs[0, 0])   # box plot – low band
ax_high = fig.add_subplot(gs[0, 1])   # box plot – high band
ax_rev  = fig.add_subplot(gs[1, 0])   # reversal rate bar – both bands
ax_psy  = fig.add_subplot(gs[1, 1])   # accuracy vs Weber ratio (psychometric)


# ── helper: jittered strip ───────────────────────────────────────────────────
def jitter(n, width=0.18):
    return (np.random.default_rng(42).random(n) - 0.5) * width


# ─────────────────────────────────────────────────────────────────────────────
# Plot A & B  –  Accuracy per force pair (Low / High band)
# ─────────────────────────────────────────────────────────────────────────────
for ax, band_label, order in [
    (ax_low,  "Low band  (ref = 1 g)",  low_order),
    (ax_high, "High band  (ref = 26 g)", high_order),
]:
    sub = subj_acc[subj_acc["band"] == band_label].copy()

    for xi, pair in enumerate(order):
        pdata = sub.loc[sub["pair_label"] == pair, "accuracy"].values
        if len(pdata) == 0:
            continue
        color = pair_color(pair, sub)

        # box
        bp = ax.boxplot(
            pdata, positions=[xi], widths=0.45,
            patch_artist=True,
            medianprops=dict(color="white", linewidth=2),
            whiskerprops=dict(color=color, linewidth=1.4),
            capprops=dict(color=color, linewidth=1.4),
            flierprops=dict(marker="o", markerfacecolor=color,
                            markersize=4, alpha=0.5, linestyle="none"),
            boxprops=dict(facecolor=color, alpha=0.6, linewidth=0),
        )

        # mean diamond
        ax.scatter(xi, np.mean(pdata), marker="D",
                   color="crimson", s=55, zorder=5)

        # jittered points
        ax.scatter(np.full(len(pdata), xi) + jitter(len(pdata)),
                   pdata, color=color, alpha=0.55, s=22, zorder=4)

    # reference lines
    ax.axhline(0.75, color="gray",  linestyle="--", linewidth=1.2,
               label="JND criterion (0.75)")
    ax.axhline(0.50, color="black", linestyle="-",  linewidth=1.0,
               label="Chance (0.50)")

    # reversal shading
    ax.axhspan(-0.05, 0.50, color=SALMON, alpha=0.08, label="Reversal zone")

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, fontsize=10)
    ax.set_ylim(-0.05, 1.10)
    ax.set_ylabel("Accuracy (proportion correct)", fontsize=11)
    ax.set_xlabel("Force pair (g)", fontsize=11)
    ax.set_title(band_label, fontsize=12, fontweight="bold")
    ax.legend(fontsize=8.5, loc="upper left")


# ─────────────────────────────────────────────────────────────────────────────
# Plot C  –  Reversal Rate  (proportion chose lighter as stronger)
# ─────────────────────────────────────────────────────────────────────────────
reversal_pairs = {
    "Low band":  ["0.6–1", "1–1.4"],
    "High band": ["15–26"],
}

bar_labels, bar_vals, bar_errs, bar_colors = [], [], [], []

for band_label, pairs in reversal_pairs.items():
    short = "Low" if "Low" in band_label else "High"
    band_key = "Low band  (ref = 1 g)" if "Low" in band_label else "High band  (ref = 26 g)"
    for pair in pairs:
        vals = subj_rev.loc[
            (subj_rev["band"] == band_key) & (subj_rev["pair_label"] == pair),
            "reversal_rate",
        ].values
        bar_labels.append(f"{pair}\n({short})")
        bar_vals.append(np.mean(vals) if len(vals) > 0 else 0)
        bar_errs.append(np.std(vals) / np.sqrt(max(len(vals), 1)))
        bar_colors.append(SALMON)

xs = np.arange(len(bar_labels))
ax_rev.bar(xs, bar_vals, color=bar_colors, alpha=0.75,
           yerr=bar_errs, capsize=5, error_kw=dict(linewidth=1.5))
ax_rev.axhline(0.50, color="black", linestyle="-", linewidth=1.2,
               label="Chance (0.50)")
ax_rev.axhline(0.75, color="gray",  linestyle="--", linewidth=1.2,
               label="75% threshold")

ax_rev.set_xticks(xs)
ax_rev.set_xticklabels(bar_labels, fontsize=10)
ax_rev.set_ylim(0, 1.05)
ax_rev.set_ylabel("Proportion 'chose lighter as stronger'", fontsize=11)
ax_rev.set_title(
    "Reversal Rate at Near-Threshold Pairs\n"
    "(>0.5 = systematic reversal)",
    fontsize=12, fontweight="bold",
)
ax_rev.legend(fontsize=9)

# annotate if > 0.5
for x, v in zip(xs, bar_vals):
    if v > 0.50:
        ax_rev.text(x, v + 0.03, f"{v:.2f}", ha="center",
                    fontsize=10, color=SALMON, fontweight="bold")


# ─────────────────────────────────────────────────────────────────────────────
# Plot D  –  Accuracy vs Weber ratio  (psychometric, both bands overlaid)
# ─────────────────────────────────────────────────────────────────────────────
band_styles = {
    "Low band  (ref = 1 g)":   dict(color="#2196f3", marker="o", label="Low band (ref = 1 g)"),
    "High band  (ref = 26 g)": dict(color="#f44336", marker="s", label="High band (ref = 26 g)"),
}

for band_label, style in band_styles.items():
    sub = subj_acc[subj_acc["band"] == band_label].copy()
    grp = sub.groupby("weber_ratio")["accuracy"].agg(["mean", "sem"]).reset_index()
    grp = grp.sort_values("weber_ratio")

    ax_psy.errorbar(
        grp["weber_ratio"], grp["mean"], yerr=grp["sem"],
        color=style["color"], marker=style["marker"],
        markersize=8, linewidth=2, capsize=5,
        label=style["label"],
    )

# Reversal zone shading
ax_psy.axhspan(-0.05, 0.50, color=SALMON, alpha=0.10)
ax_psy.axhline(0.75, color="gray",  linestyle="--", linewidth=1.2,
               label="JND criterion (0.75)")
ax_psy.axhline(0.50, color="black", linestyle="-",  linewidth=1.0,
               label="Chance (0.50)")

# Annotate reversal zone
ax_psy.text(0.35, 0.10, "Reversal zone\n(sub-chance)",
            ha="left", color=SALMON, fontsize=9.5, style="italic")

ax_psy.set_xlim(0.25, 1.55)
ax_psy.set_ylim(-0.05, 1.10)
ax_psy.set_xlabel("Weber ratio  |ΔF| / F_ref", fontsize=11)
ax_psy.set_ylabel("Accuracy (proportion correct)", fontsize=11)
ax_psy.set_title(
    "Accuracy vs. Weber Ratio — Both Bands\n(Reversal zone highlighted)",
    fontsize=12, fontweight="bold",
)
ax_psy.legend(fontsize=9, loc="upper left")

# ── 6. Global legend ──────────────────────────────────────────────────────────
legend_elements = [
    mpatches.Patch(facecolor=TEAL,   alpha=0.7, label="Above JND criterion (≥0.75)"),
    mpatches.Patch(facecolor=MID,    alpha=0.7, label="Intermediate"),
    mpatches.Patch(facecolor=SALMON, alpha=0.7, label="Reversal zone (<0.50)"),
    Line2D([0], [0], marker="D", color="w", markerfacecolor="crimson",
           markersize=8, label="Mean accuracy"),
]
fig.legend(
    handles=legend_elements,
    loc="lower center", ncol=4,
    fontsize=9.5, framealpha=0.9,
    bbox_to_anchor=(0.5, 0.01),
)

plt.savefig(os.path.join(OUTPUT_DIR, "reversal_plot.png"),
            dpi=180, bbox_inches="tight")
plt.close()
print("Saved: reversal_plot.png")
