"""
Force Discrimination – GEE Pairwise Statistics Box Plot
=========================================================
Creates two-subplot figure (Low band / High band) with:
  - Box plots per force pair, individual subject dots (jittered)
  - Red diamond = mean accuracy
  - JND criterion line at 0.75
  - Significance brackets from GEE pairwise contrasts

Column assumptions
------------------
Subject, Reference (g), Comparison (g), ChoseComparison (1/0)
  ChoseComparison=1  → participant said Comparison was stronger
  ChoseComparison=0  → participant said Reference was stronger

Accuracy = proportion of trials where the LARGER force was correctly chosen.

Statistics priority:
  1. statsmodels GEE (binomial family, subject clustering) — preferred
  2. scipy Wilcoxon signed-rank test (fallback, per-subject means)
  3. Permutation test (final fallback if neither package is available)
"""

import os
import glob
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from pathlib import Path

# ── Stat backend selection ────────────────────────────────────────────────────
USE_GEE = False
USE_WILCOXON = False

try:
    from statsmodels.genmod.generalized_estimating_equations import GEE
    from statsmodels.genmod.families import Binomial
    USE_GEE = True
    print("Using: statsmodels GEE (binomial)")
except ImportError:
    try:
        from scipy.stats import wilcoxon
        USE_WILCOXON = True
        print("statsmodels not found — using: scipy Wilcoxon signed-rank (fallback)")
    except ImportError:
        print("Neither statsmodels nor scipy found — using: permutation test (fallback)")

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

# ── 1. Paths ──────────────────────────────────────────────────────────────────
FILE_PATTERN = "/Users/kyungeunjung/NailFoldExp/Data/(FD)CurData/P*_ForceDiscrimination.csv"
OUTPUT_DIR   = "/Users/kyungeunjung/NailFoldExp/ForceDiscAnalysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 2. Load data ──────────────────────────────────────────────────────────────
files = glob.glob(FILE_PATTERN)
if not files:
    raise FileNotFoundError(f"No CSV files found matching: {FILE_PATTERN}")
print(f"Loaded {len(files)} participant file(s).")

df = pd.concat(
    [pd.read_csv(f, encoding="utf-8-sig") for f in sorted(files)],
    ignore_index=True,
)

# ── 3. Derived columns ────────────────────────────────────────────────────────
df["correct"] = np.where(
    df["Comparison"] > df["Reference"],
    df["ChoseComparison"] == 1,
    df["ChoseComparison"] == 0,
).astype(int)

df["pair_label"] = df.apply(
    lambda r: f"{min(r['Reference'], r['Comparison']):g}–"
              f"{max(r['Reference'], r['Comparison']):g}",
    axis=1,
)

df["band"] = df["Reference"].apply(
    lambda r: "Low" if r == 1 else "High"
)

# ── 4. Per-subject accuracy per pair ─────────────────────────────────────────
subj_acc = (
    df.groupby(["Subject", "band", "pair_label"])["correct"]
    .mean()
    .reset_index()
    .rename(columns={"correct": "accuracy"})
)

# ── 5. GEE pairwise contrasts ─────────────────────────────────────────────────
def _permutation_pval(a, b, n_perm=5000, seed=0):
    """Two-sided permutation test on the difference of means."""
    rng = np.random.default_rng(seed)
    obs = abs(np.mean(a) - np.mean(b))
    combined = np.concatenate([a, b])
    n_a = len(a)
    count = 0
    for _ in range(n_perm):
        rng.shuffle(combined)
        count += abs(np.mean(combined[:n_a]) - np.mean(combined[n_a:])) >= obs
    return count / n_perm


def run_gee_pairwise(band_label, pair_order):
    """
    For a given band, run all pairwise contrasts.
    Uses GEE (trial-level, binomial) → Wilcoxon → permutation, in that order.
    Returns a dict: (pair_a, pair_b) -> p_value
    """
    sub = df[df["band"] == band_label].copy()
    sub = sub[sub["pair_label"].isin(pair_order)]

    # Per-subject mean accuracy (needed for Wilcoxon / permutation fallbacks)
    subj_means = (
        sub.groupby(["Subject", "pair_label"])["correct"]
        .mean()
        .reset_index()
        .rename(columns={"correct": "accuracy"})
    )

    results = {}
    for p1, p2 in itertools.combinations(pair_order, 2):

        if USE_GEE:
            mask = sub["pair_label"].isin([p1, p2])
            chunk = sub[mask].copy()
            if chunk["Subject"].nunique() < 2:
                results[(p1, p2)] = np.nan
                continue
            chunk["pair_dummy"] = (chunk["pair_label"] == p2).astype(int)
            chunk = chunk.rename(columns={"Subject": "subj_id"})
            try:
                model = GEE.from_formula(
                    "correct ~ pair_dummy",
                    groups="subj_id",
                    data=chunk,
                    family=Binomial(),
                )
                fit = model.fit(maxiter=60)
                results[(p1, p2)] = fit.pvalues["pair_dummy"]
            except Exception as e:
                print(f"  GEE failed ({p1} vs {p2}): {e} — falling back to Wilcoxon")
                USE_GEE_local = False
            else:
                continue

        # Wilcoxon signed-rank (paired, per-subject means)
        a_vals = subj_means.loc[subj_means["pair_label"] == p1, "accuracy"].values
        b_vals = subj_means.loc[subj_means["pair_label"] == p2, "accuracy"].values

        # Align subjects
        s_sub = subj_means[subj_means["pair_label"].isin([p1, p2])]
        paired = s_sub.pivot(index="Subject", columns="pair_label", values="accuracy").dropna()
        if len(paired) < 5:
            results[(p1, p2)] = np.nan
            continue

        if USE_WILCOXON:
            try:
                from scipy.stats import wilcoxon
                _, pval = wilcoxon(paired[p1].values, paired[p2].values)
                results[(p1, p2)] = pval
            except Exception:
                results[(p1, p2)] = _permutation_pval(paired[p1].values, paired[p2].values)
        else:
            results[(p1, p2)] = _permutation_pval(paired[p1].values, paired[p2].values)

    return results


# ── 6. Ordered pair lists ─────────────────────────────────────────────────────
low_order  = ["0.4–1", "0.6–1", "1–1.4", "1–2"]
high_order = ["10–26", "15–26", "26–60"]

# Fix potential label mismatches for 25-60 vs 26-60 depending on data
# Check actual labels in data
actual_labels = subj_acc["pair_label"].unique().tolist()
print("Pair labels found in data:", sorted(actual_labels))

# Replace any "25–60" or "60–26" etc. to standard form matching data
def fix_order(order, actual):
    fixed = []
    for p in order:
        if p in actual:
            fixed.append(p)
        else:
            # try to find nearest
            alt = [a for a in actual if set(a.split("–")) == set(p.split("–"))]
            fixed.append(alt[0] if alt else p)
    return fixed

low_order  = fix_order(low_order,  actual_labels)
high_order = fix_order(high_order, actual_labels)
print("Low band pairs:", low_order)
print("High band pairs:", high_order)

low_pvals  = run_gee_pairwise("Low",  low_order)
high_pvals = run_gee_pairwise("High", high_order)

# ── 7. Color logic ────────────────────────────────────────────────────────────
TEAL    = "#2a9d8f"   # ≥ 0.75
SALMON  = "#e76f51"   # < 0.50  (reversal)
MID     = "#457b9d"   # in between
PINK    = "#c77dff"   # special: 1-2g pair

def pair_color(pair, band_subj_df):
    med = band_subj_df.loc[band_subj_df["pair_label"] == pair, "accuracy"].mean()
    if pair == "1–2":
        return PINK
    if med >= 0.75:
        return TEAL
    elif med < 0.50:
        return SALMON
    return MID

# ── helper: jittered strip ────────────────────────────────────────────────────
def jitter(n, width=0.18, seed=42):
    return (np.random.default_rng(seed).random(n) - 0.5) * width

# ── 8. Significance label ─────────────────────────────────────────────────────
def pval_label(p):
    if np.isnan(p):
        return "n.s."
    if p < 0.001:
        return "p<0.001"
    if p < 0.05:
        return f"p={p:.3f}"
    return "n.s."   # non-significant: show as n.s. to reduce clutter

# ── 9. Draw significance bracket ─────────────────────────────────────────────
def draw_bracket(ax, x1, x2, y, label, color="dimgray", linewidth=1.2, fontsize=8.0):
    """Draw a bracket from x1 to x2 at height y with label above."""
    h = 0.018
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y],
            color=color, linewidth=linewidth, clip_on=False)
    ax.text((x1 + x2) / 2, y + h + 0.004, label,
            ha="center", va="bottom", fontsize=fontsize, color=color, clip_on=False)

# ── proper interval-scheduling level assignment ───────────────────────────────
def assign_bracket_levels(combos):
    """
    For each (i1, i2) in combos, assign the minimum level such that
    it doesn't horizontally overlap any already-placed bracket at that level.
    Returns list of (i1, i2, level).
    """
    placed = []   # list of (i1, i2, level)
    for i1, i2 in combos:
        level = 0
        while True:
            conflict = any(
                l == level and not (i2 < a or b < i1)
                for a, b, l in placed
            )
            if not conflict:
                break
            level += 1
        placed.append((i1, i2, level))
    return placed

# ── 10. Plot ──────────────────────────────────────────────────────────────────
fig, (ax_low, ax_high) = plt.subplots(
    2, 1, figsize=(11, 12), constrained_layout=True
)
fig.suptitle(
    "Periungual Force Discrimination — GEE Pairwise Contrasts",
    fontsize=14, fontweight="bold",
)

def plot_band(ax, band_label, order, pvals_dict, title):
    sub = subj_acc[subj_acc["band"] == band_label].copy()

    for xi, pair in enumerate(order):
        pdata = sub.loc[sub["pair_label"] == pair, "accuracy"].values
        if len(pdata) == 0:
            print(f"  WARNING: no data for pair '{pair}' in band '{band_label}'")
            continue
        color = pair_color(pair, sub)

        # box plot
        bp = ax.boxplot(
            pdata, positions=[xi], widths=0.45,
            patch_artist=True,
            medianprops=dict(color="white", linewidth=2.2),
            whiskerprops=dict(color=color, linewidth=1.4),
            capprops=dict(color=color, linewidth=1.4),
            flierprops=dict(marker="o", markerfacecolor=color,
                            markersize=4, alpha=0.5, linestyle="none"),
            boxprops=dict(facecolor=color, alpha=0.60, linewidth=0),
        )

        # mean diamond
        ax.scatter(xi, np.mean(pdata), marker="D",
                   color="crimson", s=60, zorder=6)

        # jittered individual dots
        ax.scatter(
            np.full(len(pdata), xi) + jitter(len(pdata)),
            pdata,
            color=color, alpha=0.55, s=22, zorder=5,
        )

    # reference lines
    ax.axhline(0.75, color="gray",  linestyle="--", linewidth=1.3,
               label="JND criterion (0.75)")
    ax.axhline(0.50, color="black", linestyle="-",  linewidth=1.0,
               label="Chance (0.50)")
    ax.axhspan(-0.05, 0.50, color=SALMON, alpha=0.07, label="Reversal zone")

    # ── GEE pairwise brackets ────────────────────────────────────────────────
    # Sort combos by span (smallest first) then assign non-overlapping levels
    pair_combos = sorted(itertools.combinations(range(len(order)), 2),
                         key=lambda t: t[1] - t[0])
    placed = assign_bracket_levels(pair_combos)
    max_level = max(lv for _, _, lv in placed) if placed else 0

    bracket_base = 1.04
    bracket_step = 0.075

    # Compute required ylim to fit all brackets
    top_y = bracket_base + max_level * bracket_step + 0.018 + 0.04  # h + text
    ylim_top = max(1.15, top_y)

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, fontsize=10.5)
    ax.set_ylim(-0.05, ylim_top)
    ax.set_ylabel("Accuracy (proportion correct)", fontsize=11)
    ax.set_xlabel("Force pair (g)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
    ax.legend(fontsize=8.5, loc="upper right")

    for i1, i2, level in placed:
        p1 = order[i1]
        p2 = order[i2]
        pval = pvals_dict.get((p1, p2), pvals_dict.get((p2, p1), np.nan))
        label = pval_label(pval)
        y = bracket_base + level * bracket_step
        color = "crimson" if (not np.isnan(pval) and pval < 0.05) else "dimgray"
        draw_bracket(ax, i1, i2, y, label, color=color)

plot_band(ax_low,  "Low",  low_order,  low_pvals,
          "Low band (ref = 1 g)")
plot_band(ax_high, "High", high_order, high_pvals,
          "High band (ref = 26 g)")

# ── 11. Global colour legend ──────────────────────────────────────────────────
legend_elements = [
    mpatches.Patch(facecolor=TEAL,   alpha=0.7, label="Above JND criterion (≥0.75)"),
    mpatches.Patch(facecolor=MID,    alpha=0.7, label="Intermediate (0.50–0.75)"),
    mpatches.Patch(facecolor=SALMON, alpha=0.7, label="Reversal zone (<0.50)"),
    mpatches.Patch(facecolor=PINK,   alpha=0.7, label="1–2 g pair"),
    Line2D([0], [0], marker="D", color="w", markerfacecolor="crimson",
           markersize=9, label="Mean accuracy"),
]
fig.legend(
    handles=legend_elements,
    loc="lower center", ncol=5,
    fontsize=9, framealpha=0.9,
    bbox_to_anchor=(0.5, -0.01),
)

# ── 12. Save ──────────────────────────────────────────────────────────────────
out_path = os.path.join(OUTPUT_DIR, "gee_pairwise_plot.png")
plt.savefig(out_path, dpi=180, bbox_inches="tight")
plt.close()
print(f"Saved: {out_path}")