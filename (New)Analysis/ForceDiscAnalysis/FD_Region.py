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
  - fd_region_pooled_by_region.png        All force pairs pooled, accuracy by region
  - fd_region_slope_subjects.png          Per-subject slope plot
  - fd_region_summary_fixed.csv           Summary table
  - fd_region_pooled_summary.csv          Pooled (all pairs) summary by region
  - fd_region_pooled_stats.txt            Omnibus + pairwise LME for pooled regions
"""

import os
import glob
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import statsmodels.formula.api as smf

from fd_export import (
    BLACK,
    FIG_SIZE,
    HEATMAP_CMAP,
    REGION_ORDER,
    REGION_PALETTE,
    SLATE_BLUE,
    WINE,
    save_figure_png,
)

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")


# ============================================================
# 1. Paths
# ============================================================
FILE_PATTERN = "/Users/kyungeunjung/NailFoldExp/Data/(FD)CurData/P*_ForceDiscrimination.csv"
OUTPUT_DIR = "/Users/kyungeunjung/NailFoldExp/(New)Analysis/ForceDiscAnalysis/Output/FD_Region"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CHANCE_LEVEL = 0.50
JND_CRITERION = 0.75
REF_LINE = WINE
SIG_COLOR = WINE


def _star_from_p(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def lme_region_pair_pooled(df_in, sub_col, ref_region, target_region):
    """Trial-level region contrast with all force pairs pooled."""
    sub = df_in[df_in["Region"].isin([ref_region, target_region])].copy()
    sub = sub.dropna(subset=[sub_col, "IsCorrect", "Region"])
    if len(sub) < 10 or sub[sub_col].nunique() < 2 or sub["Region"].nunique() < 2:
        return None
    formula = f"IsCorrect ~ C(Region, Treatment(reference='{ref_region}'))"
    try:
        res = smf.mixedlm(formula, sub, groups=sub[sub_col]).fit()
        col = f"C(Region, Treatment(reference='{ref_region}'))[T.{target_region}]"
        if col not in res.params.index:
            return None
        ci = res.conf_int().loc[col]
        return {
            "ref": ref_region,
            "target": target_region,
            "label": f"{target_region} − {ref_region}",
            "coef": float(res.params[col]),
            "ci_lo": float(ci[0]),
            "ci_hi": float(ci[1]),
            "p": float(res.pvalues[col]),
        }
    except Exception:
        return None


def run_region_pairwise_lme(df_in, sub_col, regions):
    rows = []
    for r1, r2 in combinations(regions, 2):
        result = lme_region_pair_pooled(df_in, sub_col, r1, r2)
        if result is None:
            rows.append({
                "region_a": r1,
                "region_b": r2,
                "contrast": f"{r2} − {r1}",
                "coef": np.nan,
                "ci_lo": np.nan,
                "ci_hi": np.nan,
                "p": np.nan,
                "sig": "LME failed",
            })
        else:
            rows.append({
                "region_a": r1,
                "region_b": r2,
                "contrast": result["label"],
                "coef": result["coef"],
                "ci_lo": result["ci_lo"],
                "ci_hi": result["ci_hi"],
                "p": result["p"],
                "sig": _star_from_p(result["p"]),
            })
    return pd.DataFrame(rows)


def _add_sig_bracket(ax, x_l, x_r, y_base, tick_h=0.018, text="", fontsize=7.5):
    x_center = (x_l + x_r) / 2.0
    y_top = y_base + tick_h
    ax.plot(
        [x_l, x_l, x_r, x_r],
        [y_base, y_top, y_top, y_base],
        color=SIG_COLOR,
        linewidth=0.9,
        clip_on=False,
        zorder=5,
    )
    ax.text(
        x_center,
        y_top + 0.008,
        text,
        ha="center",
        va="bottom",
        fontsize=fontsize,
        color=SIG_COLOR,
        fontweight="bold",
        clip_on=False,
        zorder=6,
    )


def _boxplot_x_center(ax, cat_index):
    if not ax.containers:
        return float(cat_index)
    boxes = ax.containers[0].boxes
    if cat_index >= len(boxes):
        return float(cat_index)
    ext = boxes[cat_index].get_path().get_extents()
    return 0.5 * (ext.xmin + ext.xmax)


def _boxplot_whisker_top(ax, cat_index):
    if not ax.containers:
        return None
    whiskers = ax.containers[0].whiskers
    if cat_index >= len(whiskers) // 2:
        return None
    return max(whiskers[2 * cat_index + 1].get_ydata())


def annotate_region_pairwise_brackets(ax, regions, df_in, sub_col, *, bracket_step=0.045, alpha=0.05):
    """Significant pairwise region LME brackets only (all force pairs pooled)."""
    pair_stats = []
    for i, j in combinations(range(len(regions)), 2):
        r = lme_region_pair_pooled(df_in, sub_col, regions[i], regions[j])
        if r is not None and r["p"] < alpha:
            pair_stats.append((i, j, regions[i], regions[j], r))

    tops = [
        _boxplot_whisker_top(ax, k)
        for k in range(len(regions))
        if _boxplot_whisker_top(ax, k) is not None
    ]
    if not tops or not pair_stats:
        return ax.get_ylim()[1]

    y0 = max(tops) + 0.04
    y_max = y0
    for level, (i, j, _r1, _r2, r) in enumerate(
        sorted(pair_stats, key=lambda t: (t[1] - t[0], t[0]))
    ):
        y_base = y0 + level * bracket_step
        _add_sig_bracket(
            ax,
            _boxplot_x_center(ax, i),
            _boxplot_x_center(ax, j),
            y_base,
            text=f"{_star_from_p(r['p'])} p={r['p']:.3f}",
        )
        y_max = max(y_max, y_base + 0.06)
    return y_max


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
# 4. Canonical ordering by comparison force (f_lo, then f_hi)
# ============================================================
def _pair_sort_key(force_pair):
    lo, hi = force_pair.split("--")
    return (float(lo), float(hi))


def _ordered_pairs(band_name):
    pairs = (
        grp_summary[grp_summary["Band"] == band_name]
        .drop_duplicates("ForcePair")["ForcePair"]
        .tolist()
    )
    return sorted(pairs, key=_pair_sort_key)


pair_order_low = _ordered_pairs("Low (ref=1g)")
pair_order_high = _ordered_pairs("High (ref=26g)")
pair_order_all = pair_order_low + pair_order_high

print(f"\n[2] Pair order (Low band): {pair_order_low}")
print(f"    Pair order (High band): {pair_order_high}")

region_order = REGION_ORDER
region_palette = REGION_PALETTE


# ============================================================
# Visualization 1 — Line plot: psychometric curves per region
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=FIG_SIZE, sharey=True)

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
    ax.axhline(JND_CRITERION, color=REF_LINE, ls="--", lw=1.5, alpha=0.85)
    ax.axhline(CHANCE_LEVEL, color=BLACK, ls=":", lw=1.2, alpha=0.55)
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
save_figure_png(fig, out1)
plt.close()
print(f"\n[3] Saved {out1}")


# ============================================================
# Visualization 2 — Bar plot: grouped bars per pair
# ============================================================
fig, ax = plt.subplots(figsize=FIG_SIZE)

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
        edgecolor=BLACK,
        linewidth=0.5,
        capsize=2,
        label=region,
    )

ax.axhline(JND_CRITERION, color=REF_LINE, ls="--", lw=2, alpha=0.85,
           label=f"{int(JND_CRITERION*100)}% criterion")
ax.axhline(CHANCE_LEVEL, color=BLACK, ls=":", lw=1.5, alpha=0.55, label="Chance (50%)")

# Band separator
boundary = len(pair_order_low) - 0.5
ax.axvline(boundary, color=BLACK, lw=2.5)
ax.text((len(pair_order_low) - 1) / 2, 1.08, "Low band (ref=1g)",
         ha="center", fontsize=15, fontweight="bold")
ax.text(len(pair_order_low) + (len(pair_order_high) - 1) / 2, 1.08,
         "High band (ref=26g)", ha="center", fontsize=15, fontweight="bold")

ax.set_xticks(x_centers)
ax.set_xticklabels(pair_order_all)
ax.set_xlabel("Force pair (ordered by comparison force, ascending)", fontsize=12)
ax.set_ylabel("Mean accuracy", fontsize=12)
ax.set_title("Force discrimination accuracy by Region × ForcePair (grouped bars)",
              fontsize=13, pad=20)
ax.set_ylim(0, 1.15)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.legend(title="Region", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=10)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
out2 = os.path.join(OUTPUT_DIR, "fd_region_bar_per_pair.png")
save_figure_png(fig, out2)
plt.close()
print(f"[4] Saved {out2}")


# ============================================================
# Visualization 3 — Violin plot per pair (distribution shape)
# ============================================================
fig, ax = plt.subplots(figsize=FIG_SIZE)

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

ax.axhline(JND_CRITERION, color=REF_LINE, ls="--", lw=2, alpha=0.85)
ax.axhline(CHANCE_LEVEL, color=BLACK, ls=":", lw=1.5, alpha=0.55)

boundary = len(pair_order_low) - 0.5
ax.axvline(boundary, color=BLACK, lw=2.5)

ax.set_xlabel("Force pair", fontsize=12)
ax.set_ylabel("Accuracy (per subject)", fontsize=12)
ax.set_title("Per-subject accuracy distribution shape (violin)", fontsize=13)
ax.set_ylim(-0.05, 1.10)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.legend(title="Region", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=10)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
out3 = os.path.join(OUTPUT_DIR, "fd_region_violin_per_pair.png")
save_figure_png(fig, out3)
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

fig, ax = plt.subplots(figsize=FIG_SIZE)
sns.heatmap(
    heatmap_data,
    annot=True,
    fmt=".2f",
    cmap=HEATMAP_CMAP,
    vmin=0,
    vmax=1,
    cbar_kws={"label": "Mean accuracy"},
    linewidths=0.6,
    linecolor="white",
    ax=ax,
)

# Band boundary
boundary = len(pair_order_low)
ax.axvline(boundary, color=BLACK, lw=3)

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

ax.set_xlabel("Force pair (ordered by comparison force, ascending)", fontsize=12)
ax.set_ylabel("Region", fontsize=12)
# ax.set_title("Mean accuracy heatmap: Region × ForcePair (Weber-ratio ordered)",
#               fontsize=13, pad=25)
plt.tight_layout()
out4 = os.path.join(OUTPUT_DIR, "fd_region_heatmap_fixed.png")
save_figure_png(fig, out4)
plt.close()
print(f"[6] Saved {out4}")


# ============================================================
# Visualization 5 — All force pairs pooled: accuracy by region
# ============================================================
subj_region_pooled = (
    df.groupby([sub_col, "Region"])["IsCorrect"]
    .mean()
    .reset_index()
    .rename(columns={"IsCorrect": "accuracy"})
)

region_pooled_summary = (
    subj_region_pooled.groupby("Region")
    .agg(
        mean_acc=("accuracy", "mean"),
        sem_acc=("accuracy", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
        n_subj=("accuracy", "count"),
    )
    .reset_index()
    .set_index("Region")
    .reindex(region_order)
    .reset_index()
)
region_pooled_summary.to_csv(
    os.path.join(OUTPUT_DIR, "fd_region_pooled_summary.csv"), index=False
)

print("\n[Pooled region LME | all force pairs aggregated, trial-level, RE=Subject]")
pairwise_lme = run_region_pairwise_lme(df, sub_col, region_order)
pairwise_lme.to_csv(
    os.path.join(OUTPUT_DIR, "fd_region_pooled_pairwise_lme.csv"), index=False
)

stat_lines = [
    "Force Discrimination — Pooled region LME",
    "All force pairs aggregated | trial-level IsCorrect ~ C(Region) | RE=Subject",
    "",
]
try:
    omnibus = smf.mixedlm(
        "IsCorrect ~ C(Region)",
        df,
        groups=df[sub_col],
    ).fit()
    stat_lines.append("Omnibus C(Region):")
    for idx in omnibus.pvalues.index:
        if idx.startswith("C(Region)"):
            stat_lines.append(f"  {idx}: p={omnibus.pvalues[idx]:.4f}")
except Exception as exc:
    stat_lines.append(f"Omnibus C(Region) failed: {exc}")

stat_lines.extend(["", "Pairwise region contrasts (p < 0.05 marked *):"])
for _, row in pairwise_lme.iterrows():
    if np.isnan(row["p"]):
        stat_lines.append(f"  {row['region_a']} vs {row['region_b']}: LME failed")
    else:
        stat_lines.append(
            f"  {row['region_a']} vs {row['region_b']}: "
            f"Δ={row['coef']:.3f} [{row['ci_lo']:.3f}, {row['ci_hi']:.3f}], "
            f"p={row['p']:.4f} ({row['sig']})"
        )

stats_txt = os.path.join(OUTPUT_DIR, "fd_region_pooled_stats.txt")
with open(stats_txt, "w", encoding="utf-8") as fh:
    fh.write("\n".join(stat_lines) + "\n")

for line in stat_lines:
    print(line)

fig, ax = plt.subplots(figsize=FIG_SIZE)
box_palette = {r: region_palette[r] for r in region_order}

sns.boxplot(
    data=subj_region_pooled,
    x="Region",
    y="accuracy",
    order=region_order,
    palette=box_palette,
    width=0.55,
    fliersize=0,
    linewidth=1.0,
    boxprops=dict(alpha=0.35, edgecolor=BLACK),
    whiskerprops=dict(color=BLACK, linewidth=0.8),
    capprops=dict(color=BLACK, linewidth=0.8),
    medianprops=dict(color=BLACK, linewidth=1.5),
    ax=ax,
)
sns.stripplot(
    data=subj_region_pooled,
    x="Region",
    y="accuracy",
    order=region_order,
    palette=box_palette,
    alpha=0.55,
    size=5,
    jitter=0.18,
    linewidth=0,
    ax=ax,
)

for i, row in region_pooled_summary.iterrows():
    ax.text(
        i,
        min(row["mean_acc"] + row["sem_acc"] + 0.04, 1.02),
        f"{row['mean_acc']:.0%}",
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight="bold",
    )

ax.axhline(JND_CRITERION, color=REF_LINE, ls="--", lw=1.5, alpha=0.85,
           label=f"{int(JND_CRITERION*100)}% criterion")
ax.axhline(CHANCE_LEVEL, color=BLACK, ls=":", lw=1.2, alpha=0.55, label="Chance (50%)")

y_top_bp = annotate_region_pairwise_brackets(
    ax, region_order, df, sub_col, bracket_step=0.045
)
y_floor = 0.0
if (subj_region_pooled["accuracy"] < 0.15).any():
    y_floor = max(-0.05, float(subj_region_pooled["accuracy"].min()) - 0.05)
ax.set_ylim(y_floor, min(1.18, y_top_bp + 0.04))

ax.set_xlabel("Region", fontsize=12)
ax.set_ylabel("Mean accuracy (all force pairs pooled)", fontsize=12)
ax.set_title(
    "Force discrimination accuracy by region\n"
    "(all force pairs aggregated; brackets = significant pairwise LME, p < 0.05)",
    fontsize=13,
)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.legend(loc="lower right", fontsize=9)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
out_pooled = os.path.join(OUTPUT_DIR, "fd_region_pooled_by_region.png")
save_figure_png(fig, out_pooled)
plt.close()
print(f"\n[7] Saved {out_pooled}")
print(f"    Saved {stats_txt}")
print(f"    Saved {os.path.join(OUTPUT_DIR, 'fd_region_pooled_pairwise_lme.csv')}")
print("    Pooled summary by region:")
print(region_pooled_summary.to_string(index=False, formatters={
    "mean_acc": "{:.3f}".format,
    "sem_acc": "{:.3f}".format,
}))


# ============================================================
# Visualization 6 — Per-subject slope plots (paired)
# ============================================================
# Show whether individual subjects are consistent across regions
# Use 2 representative pairs: one sub-chance + one above-criterion

fig, axes = plt.subplots(2, 2, figsize=FIG_SIZE, sharey=True)
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
            color=SLATE_BLUE,
            alpha=0.35,
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
        color=WINE,
        lw=2.5,
        ms=10,
        label="Mean across subjects",
    )

    ax.axhline(JND_CRITERION, color=REF_LINE, ls="--", lw=1, alpha=0.75)
    ax.axhline(CHANCE_LEVEL, color=BLACK, ls=":", lw=1, alpha=0.55)
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
save_figure_png(fig, out5)
plt.close()
print(f"[8] Saved {out5}")


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