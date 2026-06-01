"""
Force Discrimination – GEE Pairwise Statistics Box Plot
=========================================================
Creates horizontal (Low | High) and vertical (Low above High) figures with:
  - Box plots per force pair, individual subject dots (jittered)
  - JND criterion line at 0.75
  - Significance brackets from GEE pairwise contrasts (p < 0.05 only)

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
from matplotlib import rcParams
from matplotlib.lines import Line2D
from pathlib import Path

# =============================================================================
# PNAS style + shared figure palette (ATD C1_Figure)
# =============================================================================
SLATE_BLUE = "#56708A"
OLIVE      = "#686F12"
WINE       = "#7F212B"
CREAM      = "#EDE2D0"
BLACK      = "#1A1A1A"
REF_LINE   = WINE

# Semantic mapping for force-discrimination bands
COLOR_ABOVE_JND = OLIVE       # accuracy ≥ 0.75
COLOR_MID       = SLATE_BLUE  # 0.50–0.75
COLOR_REVERSAL  = WINE        # accuracy < 0.50
COLOR_SPECIAL   = CREAM       # 1–2 g pair

from fd_export import FIG_SIZE, SAVE_DPI, save_figure_png

BRACKET_ALPHA = 0.05

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

# ── 0. Matplotlib style (PNAS-like, white background) ───────────────────────
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
    "legend.fontsize":       15,
    "legend.title_fontsize": 15,
    "font.size":             10,
    "axes.titlesize":        12,
    "axes.labelsize":        11,
    "xtick.labelsize":       11,
    "ytick.labelsize":       11,
    "axes.grid":             True,
    "axes.grid.axis":        "y",
    "grid.alpha":            0.35,
    "grid.linestyle":        "--",
    "grid.color":            SLATE_BLUE,
    "figure.dpi":            SAVE_DPI,
    "savefig.dpi":           SAVE_DPI,
})

# ── 1. Paths ──────────────────────────────────────────────────────────────────
FILE_PATTERN = "/Users/kyungeunjung/NailFoldExp/Data/(FD)CurData/P*_ForceDiscrimination.csv"
OUTPUT_DIR   = "/Users/kyungeunjung/NailFoldExp/(New)Analysis/ForceDiscAnalysis/Output/Stats(GEE)"
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
def pair_color(pair, band_subj_df):
    med = band_subj_df.loc[band_subj_df["pair_label"] == pair, "accuracy"].mean()
    if pair == "1–2":
        return COLOR_SPECIAL
    if med >= 0.75:
        return COLOR_ABOVE_JND
    if med < 0.50:
        return COLOR_REVERSAL
    return COLOR_MID

# ── helper: jittered strip ────────────────────────────────────────────────────
def jitter(n, width=0.18, seed=42):
    return (np.random.default_rng(seed).random(n) - 0.5) * width

# ── 8. Significance label ─────────────────────────────────────────────────────
def pval_label(p):
    if np.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


# ── Export plot data for paper ────────────────────────────────────────────────
def _band_summary(band_label, order):
    sub = subj_acc[subj_acc["band"] == band_label]
    rows = []
    for pair in order:
        vals = sub.loc[sub["pair_label"] == pair, "accuracy"].values
        if len(vals) == 0:
            continue
        rows.append({
            "band": band_label,
            "force_pair_g": pair,
            "n_subjects": len(vals),
            "mean_accuracy": np.mean(vals),
            "sem": np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else np.nan,
            "sd": np.std(vals, ddof=1) if len(vals) > 1 else np.nan,
            "median": np.median(vals),
            "min": np.min(vals),
            "max": np.max(vals),
        })
    return rows


def _pairwise_rows(band_label, order, pvals_dict):
    rows = []
    for p1, p2 in itertools.combinations(order, 2):
        pval = pvals_dict.get((p1, p2), pvals_dict.get((p2, p1), np.nan))
        rows.append({
            "band": band_label,
            "pair_a": p1,
            "pair_b": p2,
            "p_value": pval,
            "significant_p05": (not np.isnan(pval)) and pval < BRACKET_ALPHA,
            "sig_label": pval_label(pval),
        })
    return rows


def export_plot_data_txt(path):
    stat_method = (
        "GEE (binomial, subject-clustered trials)"
        if USE_GEE else
        "Wilcoxon signed-rank (per-subject means)" if USE_WILCOXON else
        "Permutation test (per-subject means)"
    )
    n_subj = subj_acc["Subject"].nunique()
    summaries = _band_summary("Low", low_order) + _band_summary("High", high_order)
    pairwise = _pairwise_rows("Low", low_order, low_pvals) + _pairwise_rows("High", high_order, high_pvals)

    lines = [
        "=" * 72,
        "Force Discrimination — GEE Pairwise Plot Data Export",
        "=" * 72,
        "",
        f"N participants     : {n_subj}",
        f"Statistical method   : {stat_method}",
        f"Significance alpha   : {BRACKET_ALPHA}",
        f"Accuracy definition  : proportion correct (chose stronger force)",
        "",
        "Low band  (reference = 1 g)  pairs: " + ", ".join(low_order),
        "High band (reference = 26 g) pairs: " + ", ".join(high_order),
        "",
        "-" * 72,
        "1. Per-subject accuracy (values shown as box-plot dots)",
        "-" * 72,
        f"{'Subject':<12} {'Band':<6} {'Force pair (g)':<14} {'Accuracy':>10}",
    ]
    for _, row in subj_acc.sort_values(["band", "pair_label", "Subject"]).iterrows():
        lines.append(
            f"{str(row['Subject']):<12} {row['band']:<6} {row['pair_label']:<14} {row['accuracy']:>10.4f}"
        )

    lines += [
        "",
        "-" * 72,
        "2. Group summary (box-plot aggregates; mean ± SEM across subjects)",
        "-" * 72,
        f"{'Band':<6} {'Force pair (g)':<14} {'N':>4} {'Mean':>8} {'SEM':>8} {'SD':>8} {'Median':>8} {'Min':>8} {'Max':>8}",
    ]
    for s in summaries:
        lines.append(
            f"{s['band']:<6} {s['force_pair_g']:<14} {s['n_subjects']:>4} "
            f"{s['mean_accuracy']:>8.4f} {s['sem']:>8.4f} {s['sd']:>8.4f} "
            f"{s['median']:>8.4f} {s['min']:>8.4f} {s['max']:>8.4f}"
        )

    lines += [
        "",
        "-" * 72,
        "3. Pairwise contrasts (bracket statistics; significant if p < 0.05)",
        "-" * 72,
        f"{'Band':<6} {'Pair A':<12} {'Pair B':<12} {'p-value':>12} {'Sig.':>6} {'Label':>6}",
    ]
    for r in pairwise:
        pstr = f"{r['p_value']:.6f}" if not np.isnan(r["p_value"]) else "NA"
        sig = "Yes" if r["significant_p05"] else "No"
        lines.append(
            f"{r['band']:<6} {r['pair_a']:<12} {r['pair_b']:<12} {pstr:>12} {sig:>6} {r['sig_label']:>6}"
        )

    lines += ["", "=" * 72]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


export_txt = os.path.join(OUTPUT_DIR, "gee_pairwise_plot_data.txt")
export_plot_data_txt(Path(export_txt))
subj_acc.to_csv(os.path.join(OUTPUT_DIR, "gee_pairwise_subject_accuracy.csv"), index=False)
pd.DataFrame(_band_summary("Low", low_order) + _band_summary("High", high_order)).to_csv(
    os.path.join(OUTPUT_DIR, "gee_pairwise_group_summary.csv"), index=False
)
pd.DataFrame(_pairwise_rows("Low", low_order, low_pvals) + _pairwise_rows("High", high_order, high_pvals)).to_csv(
    os.path.join(OUTPUT_DIR, "gee_pairwise_contrasts.csv"), index=False
)
print(f"Saved: {export_txt}")


# ── 9. Draw significance bracket ─────────────────────────────────────────────
def draw_bracket(ax, x1, x2, y, label, color=SLATE_BLUE, linewidth=1.0, fontsize=9.0):
    """Draw a bracket from x1 to x2 at height y with label above."""
    h = 0.014
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y],
            color=color, linewidth=linewidth, clip_on=False, zorder=7)
    if label:
        ax.text((x1 + x2) / 2, y + h + 0.003, label,
                ha="center", va="bottom", fontsize=fontsize,
                color=color, clip_on=False, zorder=8)

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
def plot_band(ax, band_label, order, pvals_dict, title, show_xlabel=True, show_ylabel=True,
              title_fontsize=12):
    sub = subj_acc[subj_acc["band"] == band_label].copy()
    band_max = 0.0

    for xi, pair in enumerate(order):
        pdata = sub.loc[sub["pair_label"] == pair, "accuracy"].values
        if len(pdata) == 0:
            print(f"  WARNING: no data for pair '{pair}' in band '{band_label}'")
            continue
        color = pair_color(pair, sub)
        band_max = max(band_max, float(np.max(pdata)))

        ax.boxplot(
            pdata, positions=[xi], widths=0.42,
            patch_artist=True,
            medianprops=dict(color=BLACK, linewidth=1.4),
            whiskerprops=dict(color=color, linewidth=1.2),
            capprops=dict(color=color, linewidth=1.2),
            flierprops=dict(marker="o", markerfacecolor=color,
                            markersize=3.5, alpha=0.45, linestyle="none"),
            boxprops=dict(facecolor=color, alpha=0.72, linewidth=0.8,
                          edgecolor=BLACK),
        )

        ax.scatter(
            np.full(len(pdata), xi) + jitter(len(pdata), width=0.14),
            pdata,
            color=color, alpha=0.38, s=16, zorder=5, linewidths=0,
        )

    ax.axhline(0.75, color=REF_LINE, linestyle="--", linewidth=1.0, alpha=0.8)
    ax.axhline(0.50, color=BLACK, linestyle="-", linewidth=0.8, alpha=0.55)
    ax.axhspan(-0.05, 0.50, color=COLOR_REVERSAL, alpha=0.07)

    pair_combos = sorted(
        itertools.combinations(range(len(order)), 2),
        key=lambda t: t[1] - t[0],
    )
    sig_combos = []
    for i1, i2 in pair_combos:
        p1, p2 = order[i1], order[i2]
        pval = pvals_dict.get((p1, p2), pvals_dict.get((p2, p1), np.nan))
        if not np.isnan(pval) and pval < BRACKET_ALPHA:
            sig_combos.append((i1, i2))

    placed = assign_bracket_levels(sig_combos)
    max_level = max((lv for _, _, lv in placed), default=-1)

    bracket_step = 0.055
    bracket_base = max(1.02, band_max + 0.05)
    if placed:
        top_y = bracket_base + max_level * bracket_step + 0.028
    else:
        top_y = max(1.08, band_max + 0.08)
    ylim_top = max(1.10, top_y)

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, fontsize=15)
    ax.tick_params(axis="y", labelsize=15)
    ax.set_ylim(-0.05, ylim_top)
    ax.set_xlim(-0.55, len(order) - 0.45)
    if show_ylabel:
        ax.set_ylabel("Accuracy (proportion correct)", fontsize=11)
    else:
        ax.set_ylabel("")
    if show_xlabel:
        ax.set_xlabel("Force pair (g)", fontsize=11, labelpad=4)
    else:
        ax.set_xlabel("")
    ax.set_title(title, fontsize=title_fontsize, fontweight="bold", pad=30)

    for i1, i2, level in placed:
        p1 = order[i1]
        p2 = order[i2]
        pval = pvals_dict.get((p1, p2), pvals_dict.get((p2, p1), np.nan))
        label = pval_label(pval)
        y = bracket_base + level * bracket_step
        draw_bracket(ax, i1, i2, y, label, color=WINE)


LEGEND_ELEMENTS = [
    mpatches.Patch(facecolor=COLOR_ABOVE_JND, edgecolor=BLACK, linewidth=0.6,
                   alpha=0.75, label="≥ 0.75"),
    mpatches.Patch(facecolor=COLOR_MID, edgecolor=BLACK, linewidth=0.6,
                   alpha=0.75, label="0.50–0.75"),
    mpatches.Patch(facecolor=COLOR_REVERSAL, edgecolor=BLACK, linewidth=0.6,
                   alpha=0.75, label="< 0.50"),
    mpatches.Patch(facecolor=COLOR_SPECIAL, edgecolor=BLACK, linewidth=0.6,
                   alpha=0.75, label="1–2 g"),
    Line2D([0], [0], color=REF_LINE, linestyle="--", linewidth=1.2,
           label="JND (0.75)"),
    Line2D([0], [0], color=BLACK, linestyle="-", linewidth=0.9,
           label="Chance (0.50)"),
]


def _add_legend(fig, ax_low, ax_high):
    p0 = ax_low.get_position()
    p1 = ax_high.get_position()
    legend_x = p0.x0
    legend_w = p1.x1 - p0.x0
    fig.legend(
        handles=LEGEND_ELEMENTS,
        loc="lower center",
        bbox_to_anchor=(legend_x + legend_w / 2, 0.02),
        bbox_transform=fig.transFigure,
        ncol=3,
        fontsize=15,
        frameon=False,
        columnspacing=1.2,
        handletextpad=0.5,
    )


MARGIN_BOTTOM = 0.28   # figure fraction — room for 2-row legend
GAP_BAND_IN   = 1.5    # inches between Low and High panels
TOP_MARGIN_V  = 1.4    # inches — headroom above Low band (vertical canvas)


def _panel_size_inches():
    """Panel size (in) from horizontal layout — used as fixed size for both orientations."""
    tmp, axes = plt.subplots(1, 2, figsize=FIG_SIZE, sharey=True, facecolor="#FFFFFF")
    tmp.subplots_adjust(left=0.07, right=0.98, top=0.86, bottom=MARGIN_BOTTOM, wspace=0.10)
    pos = axes[0].get_position()
    px = pos.x0 * FIG_SIZE[0]
    py = pos.y0 * FIG_SIZE[1]
    pw = pos.width * FIG_SIZE[0]
    ph = pos.height * FIG_SIZE[1]
    plt.close(tmp)
    return px, py, pw, ph


def make_pairwise_figure(orientation):
    """Build GEE pairwise plot. orientation: 'horizontal' (1×2) or 'vertical' (2×1)."""
    px, py, pw, ph = _panel_size_inches()
    left_in = 0.07 * FIG_SIZE[0]
    right_in = (1 - 0.98) * FIG_SIZE[0]
    bottom_in = MARGIN_BOTTOM * FIG_SIZE[1]
    top_in = (1 - 0.86) * FIG_SIZE[1]

    if orientation == "horizontal":
        fig_w = left_in + pw + GAP_BAND_IN + pw + right_in
        fig = plt.figure(figsize=(fig_w, FIG_SIZE[1]), facecolor="#FFFFFF")
        ax_low = fig.add_axes([left_in / fig_w, py / FIG_SIZE[1], pw / fig_w, ph / FIG_SIZE[1]])
        x_high = (left_in + pw + GAP_BAND_IN) / fig_w
        ax_high = fig.add_axes([x_high, py / FIG_SIZE[1], pw / fig_w, ph / FIG_SIZE[1]])
    elif orientation == "vertical":
        fig_h = bottom_in + ph + GAP_BAND_IN + ph + TOP_MARGIN_V
        fig = plt.figure(figsize=(FIG_SIZE[0], fig_h), facecolor="#FFFFFF")
        y_high = bottom_in / fig_h
        y_low = (bottom_in + ph + GAP_BAND_IN) / fig_h
        ax_high = fig.add_axes([px / FIG_SIZE[0], y_high, pw / FIG_SIZE[0], ph / fig_h])
        ax_low = fig.add_axes([px / FIG_SIZE[0], y_low, pw / FIG_SIZE[0], ph / fig_h])
    else:
        raise ValueError(f"Unknown orientation: {orientation!r}")

    plot_band(ax_low, "Low", low_order, low_pvals,
              "Low band (ref = 1 g)", show_xlabel=(orientation == "horizontal"),
              show_ylabel=True, title_fontsize=20)
    plot_band(ax_high, "High", high_order, high_pvals,
              "High band (ref = 26 g)", show_xlabel=True,
              show_ylabel=(orientation == "vertical"), title_fontsize = 20)
    _add_legend(fig, ax_low, ax_high)
    return fig


# ── 11–12. Save both layouts ──────────────────────────────────────────────────
for orientation, filename in (
    ("horizontal", "gee_pairwise_plot_horizontal.png"),
    ("vertical", "gee_pairwise_plot_vertical.png"),
):
    fig = make_pairwise_figure(orientation)
    out_path = os.path.join(OUTPUT_DIR, filename)
    save_figure_png(fig, out_path)
    w, h = fig.get_size_inches()
    px = (int(round(w * SAVE_DPI)), int(round(h * SAVE_DPI)))
    plt.close(fig)
    print(f"Saved: {out_path}  ({w}×{h} in @ {SAVE_DPI} dpi → {px[0]}×{px[1]} px)")