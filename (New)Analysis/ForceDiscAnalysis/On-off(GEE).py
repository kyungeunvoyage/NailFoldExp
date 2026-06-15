"""
Force Discrimination – GEE Pairwise Statistics + Region Analysis
=================================================================
Produces five figures:
  1. gee_pairwise_plot_horizontal  — overall Low | High boxplots (horizontal)
  2. gee_pairwise_plot_vertical    — overall Low / High boxplots (vertical)
  3. gee_region_onnail_vs_offnail  — On-nail vs Off-nail boxplots per force pair
  4. fd_region_slope               — Paired slope plot (On-nail ↔ Off-nail)
  5. fd_region_diff                — Difference strip plot (On-nail − Off-nail)
"""

import os
import glob
import itertools
import numpy as np
import pandas as pd
import io
import shutil
import importlib.util
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import seaborn as sns
from pathlib import Path

# =============================================================================
# ATD C1 figure style
# =============================================================================
_SCRIPT_DIR = Path(__file__).resolve().parent
_ATD_C1_PATH = _SCRIPT_DIR.parent / "ATDAnalysis" / "ATD_C1_Fig(Anika).py"

def _load_atd_c1():
    spec = importlib.util.spec_from_file_location("atd_c1_fig", _ATD_C1_PATH)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

ATD = _load_atd_c1()

# ── Style constants ───────────────────────────────────────────────────────────
SLATE_BLUE         = ATD.SLATE_BLUE
ACCENT_RED         = ATD.ACCENT_RED
CRITERION_COLOR    = ATD.CRITERION_COLOR
REF_LINE_ZORDER    = ATD.REF_LINE_ZORDER
BLACK              = ATD.BLACK
BOX_STROKE         = "#000000"
FONT_TICK          = ATD.FONT_TICK
FONT_LABEL         = ATD.FONT_LABEL
FONT_ANNOT         = ATD.FONT_ANNOT
BOX_LINEWIDTH      = ATD.BOX_LINEWIDTH
CAP_LINEWIDTH      = ATD.CAP_LINEWIDTH
FIG_SIZE           = ATD.FIG_SIZE
SAVE_DPI           = ATD.SAVE_DPI
EXPORT_WIDTHS_PX   = ATD.EXPORT_WIDTHS_PX
STRIP_ALPHA        = ATD.STRIP_ALPHA
SCATTER_HSB_BRIGHTNESS = ATD.SCATTER_HSB_BRIGHTNESS

MEDIAN_LINEWIDTH   = 2.0
MEDIAN_ZORDER      = 15
BOX_PATCH_ZORDER   = 1
WHISKER_ZORDER     = 4
SCATTER_ZORDER     = 5
STRIP_SIZE         = 3.8
JND_PCT            = 75.0
CHANCE_PCT         = 50.0
BRACKET_ALPHA      = 0.05

# Band colors (overall boxplot)
COLOR_LOW_BAND  = "#BAD6EB"
COLOR_HIGH_BAND = "#5B9BD5"
BAND_BOX_COLOR  = {"Low": COLOR_LOW_BAND, "High": COLOR_HIGH_BAND}

# Region colors
COLOR_ON_NAIL   = "#4A90C4"
COLOR_OFF_NAIL  = "#A8C8E0"

# Diff strip colors
COLOR_DIFF_POS  = "#C94040"
COLOR_DIFF_NEG  = "#4A90C4"
COLOR_DIFF_NEU  = "#AAAAAA"

# Layout constants
BOX_WIDTH_REF_N      = 3
BOX_WIDTH_AT_REF     = 0.42
STRIP_JITTER_AT_REF  = 0.14
BRACKET_TEXT_HEIGHT  = 2.5
BRACKET_TIER_GAP     = 0.5
BRACKET_TICK_H       = 2.0   # local override (no vertical ticks drawn)
BRACKET_YLIM_PAD     = 1.5
FIG_PANEL_TOP_FRAC   = 0.80
FIG_LEGEND_ANCHOR_Y  = 0.88
LEGEND_HEADROOM_IN   = 0.55
MARGIN_BOTTOM        = ATD.FIG_LEGEND_BOTTOM
GAP_BAND_IN          = 1.5

# Region viz constants
ALPHA_LINE   = 0.35
ALPHA_MEAN   = 1.00
DOT_SIZE     = 5.0
MEAN_SIZE    = 9.0
DIFF_STRIP   = 5.5
JITTER_W     = 0.06
ZERO_LINE_LW = 1.0
SIDE_OFFSET  = 0.22

# ── Stat backend ──────────────────────────────────────────────────────────────
USE_GEE      = False
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
        print("statsmodels not found — using: scipy Wilcoxon (fallback)")
    except ImportError:
        print("Neither statsmodels nor scipy — using: permutation test")

# =============================================================================
# 1. Paths
# =============================================================================
FILE_PATTERN = "/Users/kyungeunjung/NailFoldExp/Data/(FD)CurData/P*_ForceDiscrimination.csv"
OUTPUT_DIR   = "/Users/kyungeunjung/NailFoldExp/(New)Analysis/ForceDiscAnalysis/Output/Stats(GEE)"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# 2. Load data
# =============================================================================
files = glob.glob(FILE_PATTERN)
if not files:
    raise FileNotFoundError(f"No CSV files found: {FILE_PATTERN}")
print(f"Loaded {len(files)} participant file(s).")

df = pd.concat(
    [pd.read_csv(f, encoding="utf-8-sig") for f in sorted(files)],
    ignore_index=True,
)

# =============================================================================
# 3. Derived columns
# =============================================================================
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

df["band"] = df["Reference"].apply(lambda r: "Low" if r == 1 else "High")

# =============================================================================
# 4. Per-subject accuracy (all regions pooled)
# =============================================================================
subj_acc = (
    df.groupby(["Subject", "band", "pair_label"])["correct"]
    .mean().reset_index()
    .rename(columns={"correct": "accuracy"})
)

# =============================================================================
# 5. Region classification (On-nail / Off-nail)
# =============================================================================
REGION_MAP = {"C": "On-nail", "D": "On-nail", "A": "Off-nail", "F": "Off-nail"}
df_region  = df[df["Region"].isin(REGION_MAP.keys())].copy()
df_region["region_group"] = df_region["Region"].map(REGION_MAP)

subj_acc_region = (
    df_region.groupby(["Subject", "band", "pair_label", "region_group"])["correct"]
    .mean().reset_index()
    .rename(columns={"correct": "accuracy"})
)

# =============================================================================
# 6. Stat helpers
# =============================================================================
def _permutation_pval(a, b, n_perm=5000, seed=0):
    rng = np.random.default_rng(seed)
    obs = abs(np.mean(a) - np.mean(b))
    combined = np.concatenate([a, b])
    n_a = len(a)
    count = sum(
        abs(np.mean(combined[:n_a]) - np.mean(combined[n_a:])) >= obs
        for _ in range(n_perm)
        if not rng.shuffle(combined)  # shuffle in-place, always None
    )
    return count / n_perm


def run_gee_pairwise(band_label, pair_order):
    """All pairwise contrasts across force pairs (overall)."""
    sub = df[df["band"] == band_label].copy()
    sub = sub[sub["pair_label"].isin(pair_order)]
    subj_means = (
        sub.groupby(["Subject", "pair_label"])["correct"]
        .mean().reset_index()
        .rename(columns={"correct": "accuracy"})
    )
    results = {}
    for p1, p2 in itertools.combinations(pair_order, 2):
        if USE_GEE:
            mask  = sub["pair_label"].isin([p1, p2])
            chunk = sub[mask].copy()
            if chunk["Subject"].nunique() < 2:
                results[(p1, p2)] = np.nan
                continue
            chunk["pair_dummy"] = (chunk["pair_label"] == p2).astype(int)
            chunk = chunk.rename(columns={"Subject": "subj_id"})
            try:
                fit = GEE.from_formula(
                    "correct ~ pair_dummy", groups="subj_id",
                    data=chunk, family=Binomial(),
                ).fit(maxiter=60)
                results[(p1, p2)] = fit.pvalues["pair_dummy"]
                continue
            except Exception as e:
                print(f"  GEE failed ({p1} vs {p2}): {e}")

        paired = (
            subj_means[subj_means["pair_label"].isin([p1, p2])]
            .pivot(index="Subject", columns="pair_label", values="accuracy")
            .dropna()
        )
        if len(paired) < 5:
            results[(p1, p2)] = np.nan
        elif USE_WILCOXON:
            from scipy.stats import wilcoxon as _wlx
            try:
                _, pval = _wlx(paired[p1].values, paired[p2].values)
                results[(p1, p2)] = pval
            except Exception:
                results[(p1, p2)] = _permutation_pval(paired[p1].values, paired[p2].values)
        else:
            results[(p1, p2)] = _permutation_pval(paired[p1].values, paired[p2].values)
    return results


def run_gee_region(band_label, pair_order):
    """On-nail vs Off-nail GEE per force pair."""
    sub = df_region[df_region["band"] == band_label].copy()
    sub = sub[sub["pair_label"].isin(pair_order)]
    results = {}
    for pair in pair_order:
        chunk = sub[sub["pair_label"] == pair].copy()
        if chunk["Subject"].nunique() < 2:
            results[pair] = np.nan
            continue
        chunk["region_dummy"] = (chunk["region_group"] == "On-nail").astype(int)
        chunk = chunk.rename(columns={"Subject": "subj_id"})
        if USE_GEE:
            try:
                fit = GEE.from_formula(
                    "correct ~ region_dummy", groups="subj_id",
                    data=chunk, family=Binomial(),
                ).fit(maxiter=60)
                results[pair] = fit.pvalues["region_dummy"]
                continue
            except Exception as e:
                print(f"  GEE region failed ({pair}): {e}")
        pivot = (
            subj_acc_region[
                (subj_acc_region["band"] == band_label) &
                (subj_acc_region["pair_label"] == pair)
            ]
            .pivot(index="Subject", columns="region_group", values="accuracy")
            .dropna()
        )
        if len(pivot) < 5:
            results[pair] = np.nan
        elif USE_WILCOXON:
            from scipy.stats import wilcoxon as _wlx
            try:
                _, pval = _wlx(pivot["On-nail"].values, pivot["Off-nail"].values)
                results[pair] = pval
            except Exception:
                results[pair] = _permutation_pval(
                    pivot["On-nail"].values, pivot["Off-nail"].values)
        else:
            results[pair] = _permutation_pval(
                pivot["On-nail"].values, pivot["Off-nail"].values)
    return results

# =============================================================================
# 7. Ordered pair lists
# =============================================================================
low_order  = ["0.4–1", "0.6–1", "1–1.4", "1–2"]
high_order = ["10–26", "15–26", "26–60"]

actual_labels = subj_acc["pair_label"].unique().tolist()
print("Pair labels found:", sorted(actual_labels))

def fix_order(order, actual):
    fixed = []
    for p in order:
        if p in actual:
            fixed.append(p)
        else:
            alt = [a for a in actual if set(a.split("–")) == set(p.split("–"))]
            fixed.append(alt[0] if alt else p)
    return fixed

low_order  = fix_order(low_order,  actual_labels)
high_order = fix_order(high_order, actual_labels)

low_pvals  = run_gee_pairwise("Low",  low_order)
high_pvals = run_gee_pairwise("High", high_order)

low_region_pvals  = run_gee_region("Low",  low_order)
high_region_pvals = run_gee_region("High", high_order)

# =============================================================================
# 8. Shared drawing helpers
# =============================================================================
def boxplot_width(n_pairs):
    return BOX_WIDTH_AT_REF * n_pairs / BOX_WIDTH_REF_N

def jitter(n, width=0.18, seed=42):
    return (np.random.default_rng(seed).random(n) - 0.5) * width

def pval_label(p):
    if np.isnan(p): return ""
    if p < 0.001:   return "***"
    if p < 0.01:    return "**"
    if p < 0.05:    return "*"
    return ""

def pval_text_full(p):
    if np.isnan(p): return ""
    lbl = pval_label(p)
    tag = lbl if lbl else "n.s."
    return f"{tag}  p={p:.3f}"

def pval_color(p):
    return ACCENT_RED if (not np.isnan(p) and p < 0.05) else "#888888"

def draw_bracket(ax, x1, x2, y, label, tick_h=None):
    if tick_h is None:
        tick_h = BRACKET_TICK_H
    y_top = y + tick_h
    ax.plot([x1, x2], [y_top, y_top],
            color=ACCENT_RED, linewidth=1.5, clip_on=False,
            zorder=ATD.FIG2_BRACKET_ZORDER)
    if label:
        ax.text((x1+x2)/2, y_top, label,
                ha="center", va="bottom", fontsize=FONT_ANNOT + 2,
                color=ACCENT_RED, fontweight="bold", clip_on=False,
                zorder=ATD.FIG2_BRACKET_ZORDER + 1)

def _bracket_intervals_overlap(i1, i2, a, b):
    return i1 <= b and a <= i2

def bracket_tier_step():
    return (BRACKET_TICK_H + ATD.FIG2_BRACKET_TEXT_PAD
            + BRACKET_TEXT_HEIGHT + BRACKET_TIER_GAP)

def bracket_stack_top(y_base, label):
    y_top = y_base + BRACKET_TICK_H
    if label:
        return y_top + ATD.FIG2_BRACKET_TEXT_PAD + BRACKET_TEXT_HEIGHT
    return y_top

def assign_bracket_levels(combos):
    spans = [(i1, i2, i2-i1) for i1, i2 in combos]
    spans.sort(key=lambda t: (t[2], t[0]))
    placed = []
    for i1, i2, _w in spans:
        level = 0
        while any(lv == level and _bracket_intervals_overlap(i1, i2, a, b)
                  for a, b, lv in placed):
            level += 1
        placed.append((i1, i2, level))
    return placed

def finalize_axes(ax, n_x, ylim_top, *, show_ylabel=True, show_xlabel=True,
                  xlabel="Force pair (g)", ylabel="Detection Accuracy (%)"):
    ax.set_ylim(ATD.ACCURACY_YMIN, min(ATD.FIG2_BRACKET_YLIM_CAP, ylim_top))
    ax.set_yticks(ATD.ACCURACY_YTICKS)
    ax.grid(False)
    ax.tick_params(axis="both", which="both", length=0, labelsize=FONT_TICK)
    if show_ylabel:
        ax.set_ylabel(ylabel, fontsize=FONT_LABEL, labelpad=ATD.FIG_AXIS_LABELPAD)
    if show_xlabel:
        ax.set_xlabel(xlabel, fontsize=FONT_LABEL, labelpad=ATD.FIG_AXIS_LABELPAD)
    sns.despine(ax=ax)
    ATD.apply_accuracy_y_spine_bounds(ax)
    ATD.add_inward_tick_guides(ax, n_x)
    ATD.apply_accuracy_y_spine_bounds(ax)

def save_figure(fig, stem):
    from PIL import Image
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=SAVE_DPI, bbox_inches="tight",
                pad_inches=0.02, facecolor="white")
    buf.seek(0)
    master = Image.open(buf).convert("RGB")
    for tag, width_px in EXPORT_WIDTHS_PX:
        height_px = round(width_px * master.height / master.width)
        out = master.resize((width_px, height_px), Image.Resampling.LANCZOS)
        png_path = os.path.join(OUTPUT_DIR, f"{stem}_{tag}.png")
        out.save(png_path)
        print(f"Saved → {png_path}  ({width_px}×{height_px} px)")
    legacy = os.path.join(OUTPUT_DIR, f"{stem}.png")
    master.resize(
        (2102, round(2102 * master.height / master.width)),
        Image.Resampling.LANCZOS,
    ).save(legacy)
    print(f"Saved → {legacy}")

def _panel_size_inches():
    tmp, axes = plt.subplots(1, 2, figsize=FIG_SIZE, sharey=True, facecolor="#FFFFFF")
    tmp.subplots_adjust(left=0.07, right=0.98, top=FIG_PANEL_TOP_FRAC,
                        bottom=MARGIN_BOTTOM, wspace=0.10)
    pos = axes[0].get_position()
    px  = pos.x0 * FIG_SIZE[0]
    py  = pos.y0 * FIG_SIZE[1]
    pw  = pos.width  * FIG_SIZE[0]
    ph  = pos.height * FIG_SIZE[1]
    plt.close(tmp)
    return px, py, pw, ph

def _make_two_panel_fig(fig_h_override=None):
    sns.set_theme(style="white")
    ATD.apply_plot_style()
    px, py, pw, ph = _panel_size_inches()
    left_in  = 0.07 * FIG_SIZE[0]
    right_in = (1 - 0.98) * FIG_SIZE[0]
    fig_w    = left_in + pw + GAP_BAND_IN + pw + right_in
    fig_h    = fig_h_override or (FIG_SIZE[1] + LEGEND_HEADROOM_IN)
    ax_y     = py / fig_h
    ax_h     = ph / fig_h
    fig      = plt.figure(figsize=(fig_w, fig_h), facecolor="#FFFFFF")
    ax_low   = fig.add_axes([left_in / fig_w, ax_y, pw / fig_w, ax_h])
    ax_high  = fig.add_axes([(left_in + pw + GAP_BAND_IN) / fig_w, ax_y,
                              pw / fig_w, ax_h])
    return fig, ax_low, ax_high

def _add_legend(fig, handles):
    fig.legend(
        handles=handles,
        loc="upper center", bbox_to_anchor=(0.5, FIG_LEGEND_ANCHOR_Y),
        bbox_transform=fig.transFigure,
        ncol=len(handles), fontsize=FONT_LABEL, frameon=False,
        columnspacing=2.0, handletextpad=0.5, handlelength=1.6,
    )

# =============================================================================
# 9. Export stats to text / CSV
# =============================================================================
def _band_summary(band_label, order):
    sub = subj_acc[subj_acc["band"] == band_label]
    rows = []
    for pair in order:
        vals = sub.loc[sub["pair_label"] == pair, "accuracy"].values
        if not len(vals): continue
        rows.append(dict(band=band_label, force_pair_g=pair, n_subjects=len(vals),
                         mean_accuracy=np.mean(vals),
                         sem=np.std(vals,ddof=1)/np.sqrt(len(vals)) if len(vals)>1 else np.nan,
                         sd=np.std(vals,ddof=1) if len(vals)>1 else np.nan,
                         median=np.median(vals), min=np.min(vals), max=np.max(vals)))
    return rows

def _pairwise_rows(band_label, order, pvals_dict):
    rows = []
    for p1, p2 in itertools.combinations(order, 2):
        pval = pvals_dict.get((p1,p2), pvals_dict.get((p2,p1), np.nan))
        rows.append(dict(band=band_label, pair_a=p1, pair_b=p2, p_value=pval,
                         significant_p05=(not np.isnan(pval)) and pval < BRACKET_ALPHA,
                         sig_label=pval_label(pval)))
    return rows

summaries  = _band_summary("Low", low_order) + _band_summary("High", high_order)
pairwise_r = (_pairwise_rows("Low", low_order, low_pvals) +
              _pairwise_rows("High", high_order, high_pvals))
pd.DataFrame(summaries).to_csv(
    os.path.join(OUTPUT_DIR, "gee_pairwise_group_summary.csv"), index=False)
pd.DataFrame(pairwise_r).to_csv(
    os.path.join(OUTPUT_DIR, "gee_pairwise_contrasts.csv"), index=False)
subj_acc.to_csv(
    os.path.join(OUTPUT_DIR, "gee_pairwise_subject_accuracy.csv"), index=False)
print("Exported summary CSVs.")

# =============================================================================
# 10. Figure 1 & 2: Overall pairwise boxplots
# =============================================================================
OVERALL_LEGEND = [
    mpatches.Patch(facecolor=ATD.pale_box_face(COLOR_LOW_BAND),
                   edgecolor=BOX_STROKE, linewidth=BOX_LINEWIDTH,
                   label="Low band (ref = 1 g)"),
    mpatches.Patch(facecolor=ATD.pale_box_face(COLOR_HIGH_BAND),
                   edgecolor=BOX_STROKE, linewidth=BOX_LINEWIDTH,
                   label="High band (ref = 26 g)"),
]

def plot_band_overall(ax, band_label, order, pvals_dict,
                      show_xlabel=True, show_ylabel=True):
    sub = subj_acc[subj_acc["band"] == band_label].copy()
    bw  = boxplot_width(len(order))
    jw  = STRIP_JITTER_AT_REF * bw / BOX_WIDTH_AT_REF
    band_max_pct = 0.0

    for xi, pair in enumerate(order):
        pdata = sub.loc[sub["pair_label"] == pair, "accuracy"].values * 100
        if not len(pdata): continue
        band_max_pct = max(band_max_pct, float(np.max(pdata)))
        color = BAND_BOX_COLOR[band_label]
        bp = ax.boxplot([pdata], positions=[xi], widths=bw, patch_artist=True,
                        showfliers=False, capwidths=ATD.CAP_WIDTH,
                        whiskerprops={"linewidth": BOX_LINEWIDTH, "color": BOX_STROKE},
                        capprops={"linewidth": BOX_LINEWIDTH, "color": BOX_STROKE},
                        medianprops={"color": ACCENT_RED, "linewidth": MEDIAN_LINEWIDTH},
                        boxprops={"linewidth": BOX_LINEWIDTH, "edgecolor": BOX_STROKE})
        for patch in bp["boxes"]:
            patch.set_facecolor(ATD.pale_box_face(color))
            patch.set_edgecolor(BOX_STROKE); patch.set_zorder(BOX_PATCH_ZORDER)
        for line in bp["medians"]:
            line.set_color(ACCENT_RED); line.set_linewidth(MEDIAN_LINEWIDTH)
            line.set_zorder(MEDIAN_ZORDER)
        x_strip = np.full(len(pdata), xi) + jitter(len(pdata), width=jw)
        rgba    = ATD._hsb_scatter_rgba(color, SCATTER_HSB_BRIGHTNESS, STRIP_ALPHA)
        ax.scatter(x_strip, pdata, c=[rgba]*len(pdata), s=STRIP_SIZE**2,
                   linewidths=0, edgecolors="none", alpha=STRIP_ALPHA,
                   zorder=SCATTER_ZORDER, clip_on=False)

    ax.axhline(JND_PCT, color=CRITERION_COLOR, linestyle="--",
               linewidth=1.0, alpha=0.85, zorder=REF_LINE_ZORDER)
    ax.axhline(CHANCE_PCT, color=BLACK, linestyle=":", linewidth=0.8,
               alpha=0.5, zorder=1)

    sig_combos = []
    for i1, i2 in sorted(itertools.combinations(range(len(order)), 2),
                          key=lambda t: t[1]-t[0]):
        p1, p2 = order[i1], order[i2]
        pval = pvals_dict.get((p1,p2), pvals_dict.get((p2,p1), np.nan))
        if not np.isnan(pval) and pval < BRACKET_ALPHA:
            sig_combos.append((i1, i2))

    placed     = assign_bracket_levels(sig_combos)
    max_level  = max((lv for _,_,lv in placed), default=-1)
    tier_step  = bracket_tier_step()
    y_base     = max(102.0, band_max_pct + ATD.FIG2_BRACKET_BASE_PAD)
    ylim_top   = ATD.ACCURACY_YLIM_TOP
    if placed:
        ylim_top = max(ylim_top,
                       bracket_stack_top(y_base + max_level*tier_step, "*** p=0.000")
                       + BRACKET_YLIM_PAD)
    ylim_top = min(ATD.FIG2_BRACKET_YLIM_CAP, ylim_top)

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, fontsize=FONT_TICK)
    ax.set_xlim(-0.55, len(order) - 0.45)
    finalize_axes(ax, len(order), ylim_top,
                  show_ylabel=show_ylabel, show_xlabel=show_xlabel)

    for i1, i2, level in placed:
        p1, p2   = order[i1], order[i2]
        pval     = pvals_dict.get((p1,p2), pvals_dict.get((p2,p1), np.nan))
        lbl      = pval_label(pval)
        sig_text = lbl if (lbl and not np.isnan(pval)) else ""
        draw_bracket(ax, i1, i2, y_base + level*tier_step, sig_text)


def make_pairwise_figure(orientation):
    sns.set_theme(style="white"); ATD.apply_plot_style()
    px, py, pw, ph = _panel_size_inches()
    left_in   = 0.07 * FIG_SIZE[0]
    right_in  = (1 - 0.98) * FIG_SIZE[0]
    bottom_in = MARGIN_BOTTOM * FIG_SIZE[1]

    if orientation == "horizontal":
        fig_w = left_in + pw + GAP_BAND_IN + pw + right_in
        fig_h = FIG_SIZE[1] + LEGEND_HEADROOM_IN
        ax_y  = py / fig_h; ax_h = ph / fig_h
        fig   = plt.figure(figsize=(fig_w, fig_h), facecolor="#FFFFFF")
        ax_l  = fig.add_axes([left_in/fig_w, ax_y, pw/fig_w, ax_h])
        ax_r  = fig.add_axes([(left_in+pw+GAP_BAND_IN)/fig_w, ax_y, pw/fig_w, ax_h])
        plot_band_overall(ax_l, "Low",  low_order,  low_pvals,  show_ylabel=True)
        plot_band_overall(ax_r, "High", high_order, high_pvals, show_ylabel=False)
    else:
        fig_h = bottom_in + ph + GAP_BAND_IN + ph + LEGEND_HEADROOM_IN
        ax_h  = ph / fig_h; ax_w = pw / FIG_SIZE[0]; ax_x = px / FIG_SIZE[0]
        fig   = plt.figure(figsize=(FIG_SIZE[0], fig_h), facecolor="#FFFFFF")
        ax_r  = fig.add_axes([ax_x, bottom_in/fig_h, ax_w, ax_h])
        ax_l  = fig.add_axes([ax_x, (bottom_in+ph+GAP_BAND_IN)/fig_h, ax_w, ax_h])
        plot_band_overall(ax_l, "Low",  low_order,  low_pvals,  show_ylabel=True)
        plot_band_overall(ax_r, "High", high_order, high_pvals, show_ylabel=True)

    _add_legend(fig, OVERALL_LEGEND)
    return fig

if not os.getenv("PAPER_RENDER"):
    for ori, stem in [("horizontal", "gee_pairwise_plot_horizontal"),
                      ("vertical",   "gee_pairwise_plot_vertical")]:
        fig = make_pairwise_figure(ori)
        save_figure(fig, stem)
        plt.close(fig)

# =============================================================================
# 11. Figure 3: On-nail vs Off-nail boxplot
# =============================================================================
REGION_BOX_LEGEND = [
    mpatches.Patch(facecolor=ATD.pale_box_face(COLOR_ON_NAIL),
                   edgecolor=BOX_STROKE, linewidth=BOX_LINEWIDTH, label="On-nail (C+D)"),
    mpatches.Patch(facecolor=ATD.pale_box_face(COLOR_OFF_NAIL),
                   edgecolor=BOX_STROKE, linewidth=BOX_LINEWIDTH, label="Off-nail (A+F)"),
]

def plot_region_band_box(ax, band_label, order, region_pvals,
                         show_xlabel=True, show_ylabel=True):
    sub = subj_acc_region[subj_acc_region["band"] == band_label].copy()
    bw  = boxplot_width(len(order)) * 0.45
    band_max_pct = 0.0

    for xi, pair in enumerate(order):
        for gi, grp in enumerate(["On-nail", "Off-nail"]):
            x_pos  = xi + SIDE_OFFSET * (gi - 0.5)
            pdata  = sub.loc[
                (sub["pair_label"] == pair) & (sub["region_group"] == grp),
                "accuracy"].values * 100
            if not len(pdata): continue
            band_max_pct = max(band_max_pct, float(np.max(pdata)))
            color = COLOR_ON_NAIL if grp == "On-nail" else COLOR_OFF_NAIL
            bp = ax.boxplot([pdata], positions=[x_pos], widths=bw, patch_artist=True,
                            showfliers=False, capwidths=ATD.CAP_WIDTH,
                            whiskerprops={"linewidth": BOX_LINEWIDTH, "color": BOX_STROKE},
                            capprops={"linewidth": BOX_LINEWIDTH, "color": BOX_STROKE},
                            medianprops={"color": ACCENT_RED, "linewidth": MEDIAN_LINEWIDTH},
                            boxprops={"linewidth": BOX_LINEWIDTH, "edgecolor": BOX_STROKE})
            for patch in bp["boxes"]:
                patch.set_facecolor(ATD.pale_box_face(color))
                patch.set_edgecolor(BOX_STROKE); patch.set_zorder(BOX_PATCH_ZORDER)
            for line in bp["medians"]:
                line.set_color(ACCENT_RED); line.set_linewidth(MEDIAN_LINEWIDTH)
                line.set_zorder(MEDIAN_ZORDER)
            x_strip = np.full(len(pdata), x_pos) + jitter(len(pdata), width=bw*0.4)
            rgba    = ATD._hsb_scatter_rgba(color, SCATTER_HSB_BRIGHTNESS, STRIP_ALPHA)
            ax.scatter(x_strip, pdata, c=[rgba]*len(pdata), s=STRIP_SIZE**2,
                       linewidths=0, edgecolors="none", alpha=STRIP_ALPHA,
                       zorder=SCATTER_ZORDER, clip_on=False)

        pval = region_pvals.get(pair, np.nan)
        txt  = pval_text_full(pval)
        col  = pval_color(pval)
        y_br = min(band_max_pct + 8, ATD.FIG2_BRACKET_YLIM_CAP - 15)
        draw_bracket(ax, xi-SIDE_OFFSET*0.5, xi+SIDE_OFFSET*0.5, y_br, txt)

    ax.axhline(JND_PCT, color=CRITERION_COLOR, linestyle="--",
               linewidth=1.0, alpha=0.85, zorder=REF_LINE_ZORDER)
    ax.axhline(CHANCE_PCT, color=BLACK, linestyle=":", linewidth=0.8,
               alpha=0.5, zorder=1)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, fontsize=FONT_TICK)
    ax.set_xlim(-0.6, len(order) - 0.4)
    finalize_axes(ax, len(order), min(band_max_pct + 30, ATD.FIG2_BRACKET_YLIM_CAP),
                  show_ylabel=show_ylabel, show_xlabel=show_xlabel)


def make_region_box_figure():
    fig, ax_l, ax_r = _make_two_panel_fig()
    ax_l.set_title("Low Band  (ref = 1 g)",   fontsize=FONT_LABEL, fontweight="bold", pad=6)
    ax_r.set_title("High Band  (ref = 26 g)", fontsize=FONT_LABEL, fontweight="bold", pad=6)
    plot_region_band_box(ax_l, "Low",  low_order,  low_region_pvals,  show_ylabel=True)
    plot_region_band_box(ax_r, "High", high_order, high_region_pvals, show_ylabel=False)
    _add_legend(fig, REGION_BOX_LEGEND)
    return fig

fig = make_region_box_figure()
save_figure(fig, "gee_region_onnail_vs_offnail")
plt.close(fig)

# =============================================================================
# 12. Figure 4: Paired slope plot
# =============================================================================
def _get_paired(band_label, pair_label):
    sub = subj_acc_region[
        (subj_acc_region["band"] == band_label) &
        (subj_acc_region["pair_label"] == pair_label)
    ]
    pivot = (sub.pivot(index="Subject", columns="region_group", values="accuracy")
               .dropna(subset=["On-nail", "Off-nail"]).reset_index())
    pivot["On-nail"]  *= 100
    pivot["Off-nail"] *= 100
    return pivot

def plot_slope_panel(ax, band_label, order, region_pvals, show_ylabel):
    X_ON  = -0.18
    X_OFF =  0.18
    for xi, pair in enumerate(order):
        df_pair = _get_paired(band_label, pair)
        if df_pair.empty: continue

        for _, row in df_pair.iterrows():
            ax.plot([xi+X_ON, xi+X_OFF], [row["On-nail"], row["Off-nail"]],
                    color="#888888", linewidth=0.8, alpha=ALPHA_LINE, zorder=2)

        ax.scatter(np.full(len(df_pair), xi+X_ON),  df_pair["On-nail"].values,
                   color=COLOR_ON_NAIL,  s=DOT_SIZE**2, zorder=4, alpha=0.7,
                   edgecolors="none")
        ax.scatter(np.full(len(df_pair), xi+X_OFF), df_pair["Off-nail"].values,
                   color=COLOR_OFF_NAIL, s=DOT_SIZE**2, zorder=4, alpha=0.7,
                   edgecolors="none")

        for x_pos, col, c in [(xi+X_ON, "On-nail", COLOR_ON_NAIL),
                               (xi+X_OFF, "Off-nail", COLOR_OFF_NAIL)]:
            vals = df_pair[col].values
            m  = np.mean(vals)
            se = np.std(vals, ddof=1) / np.sqrt(len(vals))
            ax.errorbar(x_pos, m, yerr=se, fmt="o", color=c,
                        markeredgecolor=BOX_STROKE, markeredgewidth=0.6,
                        markersize=MEAN_SIZE, capsize=3, capthick=1.0,
                        elinewidth=1.0, ecolor=c, zorder=6)

        pval = region_pvals.get(pair, np.nan)
        txt  = pval_text_full(pval)
        col  = pval_color(pval)
        y_top = max(df_pair[["On-nail","Off-nail"]].values.flatten())
        ax.text(xi, min(108, y_top+6), txt, ha="center", va="bottom",
                fontsize=FONT_ANNOT-0.5, color=col, fontweight="bold")

    ax.axhline(JND_PCT, color=CRITERION_COLOR, linestyle="--",
               linewidth=1.0, alpha=0.85, zorder=1)
    ax.axhline(CHANCE_PCT, color="#333333", linestyle=":",
               linewidth=0.8, alpha=0.5, zorder=1)
    ax.set_ylim(ATD.ACCURACY_YMIN, 120)
    ax.set_yticks(ATD.ACCURACY_YTICKS)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, fontsize=FONT_TICK)
    ax.set_xlim(-0.6, len(order)-0.4)
    ax.tick_params(axis="both", which="both", length=0, labelsize=FONT_TICK)
    if show_ylabel:
        ax.set_ylabel("Detection Accuracy (%)", fontsize=FONT_LABEL,
                      labelpad=ATD.FIG_AXIS_LABELPAD)
    ax.set_xlabel("Force pair (g)", fontsize=FONT_LABEL, labelpad=ATD.FIG_AXIS_LABELPAD)
    ax.grid(False); sns.despine(ax=ax)
    ATD.apply_accuracy_y_spine_bounds(ax)


def make_slope_figure():
    fig, ax_l, ax_r = _make_two_panel_fig()
    ax_l.set_title("Low Band  (ref = 1 g)",   fontsize=FONT_LABEL, fontweight="bold", pad=6)
    ax_r.set_title("High Band  (ref = 26 g)", fontsize=FONT_LABEL, fontweight="bold", pad=6)
    plot_slope_panel(ax_l, "Low",  low_order,  low_region_pvals,  show_ylabel=True)
    plot_slope_panel(ax_r, "High", high_order, high_region_pvals, show_ylabel=False)
    _add_legend(fig, [
        mlines.Line2D([], [], color=COLOR_ON_NAIL,  marker="o", markersize=6,
                      linewidth=0, label="On-nail (C+D)"),
        mlines.Line2D([], [], color=COLOR_OFF_NAIL, marker="o", markersize=6,
                      linewidth=0, label="Off-nail (A+F)"),
        mlines.Line2D([], [], color="#888888", linewidth=0.9, alpha=0.6,
                      label="Individual subject"),
    ])
    return fig

fig = make_slope_figure()
save_figure(fig, "fd_region_slope")
plt.close(fig)

# =============================================================================
# 13. Figure 5: Difference strip plot
# =============================================================================
def plot_diff_panel(ax, band_label, order, region_pvals, show_ylabel):
    rng = np.random.default_rng(42)
    all_diffs = []

    for xi, pair in enumerate(order):
        df_pair = _get_paired(band_label, pair)
        if df_pair.empty: continue
        diffs = df_pair["On-nail"].values - df_pair["Off-nail"].values
        all_diffs.extend(diffs)

        colors  = [COLOR_DIFF_POS if d > 2 else COLOR_DIFF_NEG if d < -2
                   else COLOR_DIFF_NEU for d in diffs]
        x_strip = xi + (rng.random(len(diffs)) - 0.5) * JITTER_W * 2
        ax.scatter(x_strip, diffs, c=colors, s=DIFF_STRIP**2,
                   edgecolors="white", linewidths=0.3, alpha=0.80, zorder=4)

        m  = np.mean(diffs)
        se = np.std(diffs, ddof=1) / np.sqrt(len(diffs))
        ax.errorbar(xi, m, yerr=se, fmt="D", color="#222222",
                    markeredgecolor="#FFFFFF", markeredgewidth=0.5,
                    markersize=6, capsize=3, capthick=1.0, elinewidth=1.2, zorder=6)

        pval = region_pvals.get(pair, np.nan)
        txt  = pval_text_full(pval)
        col  = pval_color(pval)
        y_top = max(abs(diffs)) if len(diffs) else 10
        ax.text(xi, y_top + 5, txt, ha="center", va="bottom",
                fontsize=FONT_ANNOT - 0.5, color=col, fontweight="bold")

    ax.axhline(0, color="#333333", linewidth=1.0, linestyle="-", alpha=0.8, zorder=3)
    ylim = max(55, max(abs(d) for d in all_diffs) + 15) if all_diffs else 55
    ax.set_ylim(-ylim, ylim)
    ticks = [-50, -25, 0, 25, 50]
    ax.set_yticks([t for t in ticks if abs(t) <= ylim])
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, fontsize=FONT_TICK)
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.tick_params(axis="both", which="both", length=0, labelsize=FONT_TICK)
    if show_ylabel:
        ax.set_ylabel("Δ Accuracy: On-nail − Off-nail (%)",
                      fontsize=FONT_LABEL, labelpad=ATD.FIG_AXIS_LABELPAD)
    ax.set_xlabel("Force pair (g)", fontsize=FONT_LABEL, labelpad=ATD.FIG_AXIS_LABELPAD)
    ax.grid(False); sns.despine(ax=ax)


def make_diff_figure():
    fig, ax_l, ax_r = _make_two_panel_fig()
    ax_l.set_title("Low Band  (ref = 1 g)",   fontsize=FONT_LABEL, fontweight="bold", pad=18)
    ax_r.set_title("High Band  (ref = 26 g)", fontsize=FONT_LABEL, fontweight="bold", pad=18)
    plot_diff_panel(ax_l, "Low",  low_order,  low_region_pvals,  show_ylabel=True)
    plot_diff_panel(ax_r, "High", high_order, high_region_pvals, show_ylabel=False)
    _add_legend(fig, [
        mlines.Line2D([], [], color=COLOR_DIFF_POS, marker="o", markersize=6,
                      linewidth=0, label="On-nail > Off-nail"),
        mlines.Line2D([], [], color=COLOR_DIFF_NEG, marker="o", markersize=6,
                      linewidth=0, label="Off-nail > On-nail"),
        mlines.Line2D([], [], color=COLOR_DIFF_NEU, marker="o", markersize=6,
                      linewidth=0, label="Near-zero (±2%)"),
        mlines.Line2D([], [], color="#222222", marker="D", markersize=6,
                      linewidth=0, label="Mean ± SE"),
    ])
    return fig

fig = make_diff_figure()
save_figure(fig, "fd_region_diff")
plt.close(fig)

print("\nAll figures saved to:", OUTPUT_DIR)