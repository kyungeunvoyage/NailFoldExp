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
import io
import shutil
import importlib.util
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

# =============================================================================
# ATD C1 figure style (fonts, axes, export widths — ATD_C1_Fig(Anika).py)
# =============================================================================
_SCRIPT_DIR = Path(__file__).resolve().parent
_ATD_C1_PATH = _SCRIPT_DIR.parent / "ATDAnalysis" / "ATD_C1_Fig(Anika).py"


def _load_atd_c1():
    spec = importlib.util.spec_from_file_location("atd_c1_fig", _ATD_C1_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ATD = _load_atd_c1()

# Semantic FD band colors (box fill only; lines/ticks match ATD)
SLATE_BLUE = ATD.SLATE_BLUE
OLIVE      = "#686F12"
WINE       = "#7F212B"
CREAM      = "#EDE2D0"
BLACK      = ATD.BLACK          # axis / criterion (#1A1A1A)
BOX_STROKE = "#000000"          # box outline — fully opaque (not faded by patch alpha)
MEDIAN_LINEWIDTH = 2.0
MEDIAN_ZORDER = 15
BOX_PATCH_ZORDER = 1
WHISKER_ZORDER = 4
SCATTER_ZORDER = 5
ACCENT_RED = ATD.ACCENT_RED
CRITERION_COLOR = ATD.CRITERION_COLOR
REF_LINE_ZORDER = ATD.REF_LINE_ZORDER
FONT_TICK = ATD.FONT_TICK
FONT_LABEL = ATD.FONT_LABEL
FONT_ANNOT = ATD.FONT_ANNOT
BOX_LINEWIDTH = ATD.BOX_LINEWIDTH
CAP_LINEWIDTH = ATD.CAP_LINEWIDTH
FIG_SIZE = ATD.FIG_SIZE
SAVE_DPI = ATD.SAVE_DPI
EXPORT_WIDTHS_PX = ATD.EXPORT_WIDTHS_PX
JND_PCT = 75.0
CHANCE_PCT = 50.0

COLOR_LOW_BAND = "#BAD6EB"   # all Low-band boxes / points
COLOR_HIGH_BAND = "#D0E4FF"  # all High-band boxes / points
BAND_BOX_COLOR = {"Low": COLOR_LOW_BAND, "High": COLOR_HIGH_BAND}

BRACKET_ALPHA = 0.05
STRIP_ALPHA = ATD.STRIP_ALPHA
SCATTER_HSB_BRIGHTNESS = ATD.SCATTER_HSB_BRIGHTNESS
STRIP_SIZE = 3.8

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

# Box width in data coords: scale so Low/High panels match pixel width (High = ref).
BOX_WIDTH_REF_N = len(high_order)
BOX_WIDTH_AT_REF = 0.42
STRIP_JITTER_AT_REF = 0.14


def boxplot_width(n_pairs):
    """Same visual box width in both panels (fewer x categories → wider axis span)."""
    return BOX_WIDTH_AT_REF * n_pairs / BOX_WIDTH_REF_N


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


def save_gee_figure(fig, stem, export_widths=None):
    """Save PNG at ATD column widths (1col / 1.5col / 2col) + legacy single name."""
    from PIL import Image

    if export_widths is None:
        export_widths = EXPORT_WIDTHS_PX
    buf = io.BytesIO()
    fig.savefig(
        buf, format="png", dpi=SAVE_DPI, bbox_inches="tight",
        pad_inches=0.02, facecolor="white",
    )
    buf.seek(0)
    master = Image.open(buf).convert("RGB")
    for tag, width_px in export_widths:
        height_px = round(width_px * master.height / master.width)
        out = master.resize((width_px, height_px), Image.Resampling.LANCZOS)
        png_path = os.path.join(OUTPUT_DIR, f"{stem}_{tag}.png")
        out.save(png_path)
        print(f"Saved PNG → {png_path}  ({width_px}×{height_px} px)")
    legacy = os.path.join(OUTPUT_DIR, f"{stem}.png")
    master.resize(
        (2102, round(2102 * master.height / master.width)),
        Image.Resampling.LANCZOS,
    ).save(legacy)
    print(f"Saved PNG → {legacy}  (2col alias)")


def finalize_gee_axes(ax, n_x, ylim_top, *, show_ylabel=True, show_xlabel=True,
                      xlabel="Force pair (g)"):
    """ATD C1 axes: 0–100% y, inward ticks, floating 0%, no grid."""
    ax.set_ylim(ATD.ACCURACY_YMIN, min(ATD.FIG2_BRACKET_YLIM_CAP, ylim_top))
    ax.set_yticks(ATD.ACCURACY_YTICKS)
    ax.grid(False)
    ax.tick_params(axis="both", which="both", length=0, labelsize=FONT_TICK)
    if show_ylabel:
        ax.set_ylabel("Detection Accuracy (%)", fontsize=FONT_LABEL,
                      labelpad=ATD.FIG_AXIS_LABELPAD)
    if show_xlabel:
        ax.set_xlabel(xlabel, fontsize=FONT_LABEL, labelpad=ATD.FIG_AXIS_LABELPAD)
    sns.despine(ax=ax)
    ATD.apply_accuracy_y_spine_bounds(ax)
    ATD.add_inward_tick_guides(ax, n_x)
    ATD.apply_accuracy_y_spine_bounds(ax)


# ── 9. Draw significance bracket (ATD Fig2 style) ─────────────────────────────
def draw_bracket(ax, x1, x2, y, label, tick_h=ATD.FIG2_BRACKET_TICK_H):
    """Bracket between pair positions; y in % scale."""
    y_top = y + tick_h
    ax.plot(
        [x1, x1, x2, x2], [y, y_top, y_top, y],
        color=ACCENT_RED, linewidth=0.75, clip_on=False, zorder=ATD.FIG2_BRACKET_ZORDER,
    )
    if label:
        ax.text(
            (x1 + x2) / 2, y_top + ATD.FIG2_BRACKET_TEXT_PAD, label,
            ha="center", va="bottom", fontsize=FONT_ANNOT, color=ACCENT_RED,
            fontweight="bold", clip_on=False, zorder=ATD.FIG2_BRACKET_ZORDER + 1,
        )

# Bracket stacking (ATD_aggregate-style): closed x-intervals, narrow spans first.
BRACKET_TEXT_HEIGHT = 4.5   # label extent in % y-units (fontsize ~10)
BRACKET_TIER_GAP = 1.2      # gap between label top and next tier baseline
BRACKET_YLIM_PAD = 1.5      # headroom above topmost label


def _bracket_intervals_overlap(i1, i2, a, b):
    """True when [i1, i2] and [a, b] share x (touching endpoints counts)."""
    return i1 <= b and a <= i2


def bracket_tier_step():
    """Vertical distance between bracket baselines (line + label + gap)."""
    return (
        ATD.FIG2_BRACKET_TICK_H
        + ATD.FIG2_BRACKET_TEXT_PAD
        + BRACKET_TEXT_HEIGHT
        + BRACKET_TIER_GAP
    )


def bracket_stack_top(y_base, label):
    """Highest y used by one bracket tier (for ylim headroom)."""
    y_top = y_base + ATD.FIG2_BRACKET_TICK_H
    if label:
        return y_top + ATD.FIG2_BRACKET_TEXT_PAD + BRACKET_TEXT_HEIGHT
    return y_top


def assign_bracket_levels(combos):
    """
    Minimum tier per pair so overlapping x-spans are not on the same level.
    Narrow spans are placed first (inner brackets lower, wide spans higher).
    """
    spans = [(i1, i2, i2 - i1) for i1, i2 in combos]
    spans.sort(key=lambda t: (t[2], t[0]))
    placed = []
    for i1, i2, _w in spans:
        level = 0
        while any(
            lv == level and _bracket_intervals_overlap(i1, i2, a, b)
            for a, b, lv in placed
        ):
            level += 1
        placed.append((i1, i2, level))
    return placed

# ── 10. Plot ──────────────────────────────────────────────────────────────────
def plot_band(ax, band_label, order, pvals_dict, show_xlabel=True, show_ylabel=True):
    sub = subj_acc[subj_acc["band"] == band_label].copy()
    band_max_pct = 0.0
    bw = boxplot_width(len(order))
    strip_jitter = STRIP_JITTER_AT_REF * bw / BOX_WIDTH_AT_REF

    for xi, pair in enumerate(order):
        pdata_pct = sub.loc[sub["pair_label"] == pair, "accuracy"].values * 100.0
        if len(pdata_pct) == 0:
            print(f"  WARNING: no data for pair '{pair}' in band '{band_label}'")
            continue
        color = BAND_BOX_COLOR[band_label]
        band_max_pct = max(band_max_pct, float(np.max(pdata_pct)))

        bp = ax.boxplot(
            [pdata_pct], positions=[xi], widths=bw,
            patch_artist=True, showfliers=False,
            capwidths=ATD.CAP_WIDTH,
            whiskerprops={
                "linewidth": BOX_LINEWIDTH, "color": BOX_STROKE,
                "solid_capstyle": "butt",
            },
            capprops={
                "linewidth": BOX_LINEWIDTH, "color": BOX_STROKE,
                "solid_capstyle": "butt",
            },
            medianprops={"color": ACCENT_RED, "linewidth": MEDIAN_LINEWIDTH},
            boxprops={"linewidth": BOX_LINEWIDTH, "edgecolor": BOX_STROKE},
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(ATD.pale_box_face(color))
            patch.set_edgecolor(BOX_STROKE)
            patch.set_linewidth(BOX_LINEWIDTH)
            patch.set_alpha(1.0)  # opacity in pale_box_face hex alpha channel
            patch.set_zorder(BOX_PATCH_ZORDER)
        for key in ("whiskers", "caps"):
            for line in bp[key]:
                line.set_color(BOX_STROKE)
                line.set_linewidth(BOX_LINEWIDTH)
                line.set_alpha(1.0)
                line.set_zorder(WHISKER_ZORDER)
        for line in bp["medians"]:
            line.set_color(ACCENT_RED)
            line.set_linewidth(MEDIAN_LINEWIDTH)
            line.set_alpha(1.0)
            line.set_zorder(MEDIAN_ZORDER)

        x_strip = np.full(len(pdata_pct), xi) + jitter(len(pdata_pct), width=strip_jitter)
        strip_rgba = ATD._hsb_scatter_rgba(color, SCATTER_HSB_BRIGHTNESS, STRIP_ALPHA)
        ax.scatter(
            x_strip, pdata_pct,
            c=[strip_rgba] * len(pdata_pct),
            s=STRIP_SIZE ** 2,
            linewidths=0,
            edgecolors="none",
            alpha=STRIP_ALPHA,
            zorder=SCATTER_ZORDER,
            clip_on=False,
        )

    ax.axhline(
        JND_PCT, color=CRITERION_COLOR, linestyle="--", linewidth=1.0, alpha=0.85,
        zorder=REF_LINE_ZORDER,
    )
    ax.axhline(CHANCE_PCT, color=BLACK, linestyle=":", linewidth=0.8, alpha=0.5, zorder=1)

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
    tier_step = bracket_tier_step()
    bracket_base = max(102.0, band_max_pct + ATD.FIG2_BRACKET_BASE_PAD)
    ylim_top = max(ATD.ACCURACY_YLIM_TOP, band_max_pct + 8.0)
    if placed:
        ylim_top = max(
            ylim_top,
            bracket_stack_top(
                bracket_base + max_level * tier_step, "***  p=0.000",
            ) + BRACKET_YLIM_PAD,
        )
    ylim_top = min(ATD.FIG2_BRACKET_YLIM_CAP, ylim_top)

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, fontsize=FONT_TICK)
    ax.set_xlim(-0.55, len(order) - 0.45)
    finalize_gee_axes(
        ax, len(order), ylim_top,
        show_ylabel=show_ylabel, show_xlabel=show_xlabel,
    )

    for i1, i2, level in placed:
        p1 = order[i1]
        p2 = order[i2]
        pval = pvals_dict.get((p1, p2), pvals_dict.get((p2, p1), np.nan))
        label = pval_label(pval)
        if label:
            sig_text = f"{label}  p={pval:.3f}" if not np.isnan(pval) else label
        else:
            sig_text = ""
        y = bracket_base + level * tier_step
        draw_bracket(ax, i1, i2, y, sig_text)


LEGEND_ELEMENTS = [
    mpatches.Patch(
        facecolor=ATD.pale_box_face(COLOR_LOW_BAND),
        edgecolor=BOX_STROKE, linewidth=BOX_LINEWIDTH, label="Low band (ref = 1 g)",
    ),
    mpatches.Patch(
        facecolor=ATD.pale_box_face(COLOR_HIGH_BAND),
        edgecolor=BOX_STROKE, linewidth=BOX_LINEWIDTH, label="High band (ref = 26 g)",
    ),
]


def _add_legend(fig):
    """Legend in the dedicated top strip of the figure."""
    fig.legend(
        handles=LEGEND_ELEMENTS,
        loc="upper center",
        bbox_to_anchor=(0.5, FIG_LEGEND_ANCHOR_Y),
        bbox_transform=fig.transFigure,
        ncol=2,
        fontsize=FONT_LABEL,
        frameon=False,
        columnspacing=2.0,
        handletextpad=0.5,
        handlelength=1.6,
    )


# Layout: legend strip at top → plot (+ brackets in data space)
FIG_PANEL_TOP_FRAC = 0.80    # axes ymax — headroom for figure legend only
FIG_LEGEND_ANCHOR_Y = 0.975  # top-center of full figure canvas
LEGEND_HEADROOM_IN = 0.55    # extra figure height above panels (inches)
MARGIN_BOTTOM = ATD.FIG_LEGEND_BOTTOM
GAP_BAND_IN = 1.5            # inches between Low and High panels


def _panel_size_inches():
    """Panel size (in) from horizontal layout — used as fixed size for both orientations."""
    tmp, axes = plt.subplots(1, 2, figsize=FIG_SIZE, sharey=True, facecolor="#FFFFFF")
    tmp.subplots_adjust(
        left=0.07, right=0.98, top=FIG_PANEL_TOP_FRAC, bottom=MARGIN_BOTTOM, wspace=0.10,
    )
    pos = axes[0].get_position()
    px = pos.x0 * FIG_SIZE[0]
    py = pos.y0 * FIG_SIZE[1]
    pw = pos.width * FIG_SIZE[0]
    ph = pos.height * FIG_SIZE[1]
    plt.close(tmp)
    return px, py, pw, ph


def make_pairwise_figure(orientation):
    """Build GEE pairwise plot. orientation: 'horizontal' (1×2) or 'vertical' (2×1)."""
    sns.set_theme(style="white")
    ATD.apply_plot_style()
    px, py, pw, ph = _panel_size_inches()
    left_in = 0.07 * FIG_SIZE[0]
    right_in = (1 - 0.98) * FIG_SIZE[0]
    bottom_in = MARGIN_BOTTOM * FIG_SIZE[1]

    if orientation == "horizontal":
        fig_w = left_in + pw + GAP_BAND_IN + pw + right_in
        fig_h = FIG_SIZE[1] + LEGEND_HEADROOM_IN
        ax_y = py / fig_h
        ax_h_frac = ph / fig_h
        fig = plt.figure(figsize=(fig_w, fig_h), facecolor="#FFFFFF")
        ax_low = fig.add_axes([left_in / fig_w, ax_y, pw / fig_w, ax_h_frac])
        x_high = (left_in + pw + GAP_BAND_IN) / fig_w
        ax_high = fig.add_axes([x_high, ax_y, pw / fig_w, ax_h_frac])
    elif orientation == "vertical":
        fig_h = bottom_in + ph + GAP_BAND_IN + ph + LEGEND_HEADROOM_IN
        ax_h_frac = ph / fig_h
        fig = plt.figure(figsize=(FIG_SIZE[0], fig_h), facecolor="#FFFFFF")
        y_high = bottom_in / fig_h
        y_low = (bottom_in + ph + GAP_BAND_IN) / fig_h
        ax_w_frac = pw / FIG_SIZE[0]
        ax_x = px / FIG_SIZE[0]
        ax_high = fig.add_axes([ax_x, y_high, ax_w_frac, ax_h_frac])
        ax_low = fig.add_axes([ax_x, y_low, ax_w_frac, ax_h_frac])
    else:
        raise ValueError(f"Unknown orientation: {orientation!r}")

    plot_band(ax_low, "Low", low_order, low_pvals,
              show_xlabel=(orientation == "horizontal"), show_ylabel=True)
    plot_band(ax_high, "High", high_order, high_pvals,
              show_xlabel=True, show_ylabel=(orientation == "vertical"))
    _add_legend(fig)
    return fig


# ── 11–12. Save both layouts (ATD column widths + legacy alias) ───────────────
for orientation, stem in (
    ("horizontal", "gee_pairwise_plot_horizontal"),
    ("vertical", "gee_pairwise_plot_vertical"),
):
    fig = make_pairwise_figure(orientation)
    save_gee_figure(fig, stem)
    plt.close(fig)

# Legacy path used in repo root / IDE
_root_alias = _SCRIPT_DIR / "gee_pairwise_plot.png"
_h2col = Path(OUTPUT_DIR) / "gee_pairwise_plot_horizontal_2col.png"
if _h2col.exists():
    shutil.copy2(_h2col, _root_alias)
    print(f"Copied → {_root_alias}")

# ── REGION ANALYSIS: On-nail (C+D) vs Off-nail (A+F) ──────────────────────────

# 1. Region 분류
REGION_MAP = {"C": "On-nail", "D": "On-nail", "A": "Off-nail", "F": "Off-nail"}
df_region = df[df["Region"].isin(REGION_MAP.keys())].copy()
df_region["region_group"] = df_region["Region"].map(REGION_MAP)

# 2. Per-subject accuracy by region_group × band × pair
subj_acc_region = (
    df_region.groupby(["Subject", "band", "pair_label", "region_group"])["correct"]
    .mean()
    .reset_index()
    .rename(columns={"correct": "accuracy"})
)

# 3. GEE: On-nail vs Off-nail per force pair, per band
def run_gee_region(band_label, pair_order):
    """
    For each force pair, test On-nail vs Off-nail using trial-level GEE.
    Returns dict: pair_label -> p_value
    """
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
                model = GEE.from_formula(
                    "correct ~ region_dummy",
                    groups="subj_id",
                    data=chunk,
                    family=Binomial(),
                )
                fit = model.fit(maxiter=60)
                results[pair] = fit.pvalues["region_dummy"]
            except Exception as e:
                print(f"  GEE region failed ({pair}): {e}")
                results[pair] = np.nan
        else:
            # Wilcoxon fallback: per-subject means
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
                from scipy.stats import wilcoxon
                try:
                    _, pval = wilcoxon(pivot["On-nail"].values, pivot["Off-nail"].values)
                    results[pair] = pval
                except Exception:
                    results[pair] = _permutation_pval(
                        pivot["On-nail"].values, pivot["Off-nail"].values
                    )
            else:
                results[pair] = _permutation_pval(
                    pivot["On-nail"].values, pivot["Off-nail"].values
                )

    return results


low_region_pvals  = run_gee_region("Low",  low_order)
high_region_pvals = run_gee_region("High", high_order)

# 4. Figure: On-nail vs Off-nail side by side per force pair
COLOR_ON_NAIL  = "#7FB3D3"   # darker blue
COLOR_OFF_NAIL = "#D3E9F5"   # lighter blue

REGION_COLORS = {"On-nail": COLOR_ON_NAIL, "Off-nail": COLOR_OFF_NAIL}
REGION_GROUPS = ["On-nail", "Off-nail"]
SIDE_OFFSET   = 0.22  # x offset between On-nail and Off-nail boxes within a pair


def plot_region_band(ax, band_label, order, region_pvals,
                     show_xlabel=True, show_ylabel=True):
    sub = subj_acc_region[subj_acc_region["band"] == band_label].copy()
    bw = boxplot_width(len(order)) * 0.45  # narrower boxes (two per slot)
    band_max_pct = 0.0

    for xi, pair in enumerate(order):
        for gi, grp in enumerate(REGION_GROUPS):
            x_pos = xi + SIDE_OFFSET * (gi - 0.5)
            pdata = sub.loc[
                (sub["pair_label"] == pair) & (sub["region_group"] == grp),
                "accuracy"
            ].values * 100.0
            if len(pdata) == 0:
                continue

            color = REGION_COLORS[grp]
            band_max_pct = max(band_max_pct, float(np.max(pdata)))

            bp = ax.boxplot(
                [pdata], positions=[x_pos], widths=bw,
                patch_artist=True, showfliers=False,
                capwidths=ATD.CAP_WIDTH,
                whiskerprops={"linewidth": BOX_LINEWIDTH, "color": BOX_STROKE},
                capprops={"linewidth": BOX_LINEWIDTH, "color": BOX_STROKE},
                medianprops={"color": ACCENT_RED, "linewidth": MEDIAN_LINEWIDTH},
                boxprops={"linewidth": BOX_LINEWIDTH, "edgecolor": BOX_STROKE},
            )
            for patch in bp["boxes"]:
                patch.set_facecolor(ATD.pale_box_face(color))
                patch.set_edgecolor(BOX_STROKE)
                patch.set_zorder(BOX_PATCH_ZORDER)
            for line in bp["medians"]:
                line.set_color(ACCENT_RED)
                line.set_linewidth(MEDIAN_LINEWIDTH)
                line.set_zorder(MEDIAN_ZORDER)

            # Jittered dots
            x_strip = np.full(len(pdata), x_pos) + jitter(len(pdata), width=bw * 0.4)
            strip_rgba = ATD._hsb_scatter_rgba(color, SCATTER_HSB_BRIGHTNESS, STRIP_ALPHA)
            ax.scatter(x_strip, pdata, c=[strip_rgba]*len(pdata),
                       s=STRIP_SIZE**2, linewidths=0, edgecolors="none",
                       alpha=STRIP_ALPHA, zorder=SCATTER_ZORDER, clip_on=False)

        # n.s. / sig bracket between On-nail and Off-nail for this pair
        pval = region_pvals.get(pair, np.nan)
        label = pval_label(pval)
        if np.isnan(pval):
            sig_text = ""
        elif label:
            sig_text = f"{label}  p={pval:.3f}"
        else:
            sig_text = f"n.s.  p={pval:.3f}"

        # Draw bracket just above the boxes
        y_bracket = band_max_pct + 8.0
        x1 = xi - SIDE_OFFSET * 0.5
        x2 = xi + SIDE_OFFSET * 0.5
        draw_bracket(ax, x1, x2, y_bracket, sig_text)

    ax.axhline(JND_PCT, color=CRITERION_COLOR, linestyle="--",
               linewidth=1.0, alpha=0.85, zorder=REF_LINE_ZORDER)
    ax.axhline(CHANCE_PCT, color=BLACK, linestyle=":", linewidth=0.8,
               alpha=0.5, zorder=1)

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, fontsize=FONT_TICK)
    ax.set_xlim(-0.6, len(order) - 0.4)
    ylim_top = max(ATD.ACCURACY_YLIM_TOP, band_max_pct + 25.0)
    finalize_gee_axes(ax, len(order), ylim_top,
                      show_ylabel=show_ylabel, show_xlabel=show_xlabel)


REGION_LEGEND_ELEMENTS = [
    mpatches.Patch(facecolor=ATD.pale_box_face(COLOR_ON_NAIL),
                   edgecolor=BOX_STROKE, linewidth=BOX_LINEWIDTH, label="On-nail (C+D)"),
    mpatches.Patch(facecolor=ATD.pale_box_face(COLOR_OFF_NAIL),
                   edgecolor=BOX_STROKE, linewidth=BOX_LINEWIDTH, label="Off-nail (A+F)"),
]


def make_region_figure():
    sns.set_theme(style="white")
    ATD.apply_plot_style()
    px, py, pw, ph = _panel_size_inches()
    left_in  = 0.07 * FIG_SIZE[0]
    right_in = (1 - 0.98) * FIG_SIZE[0]
    bottom_in = MARGIN_BOTTOM * FIG_SIZE[1]

    fig_w = left_in + pw + GAP_BAND_IN + pw + right_in
    fig_h = FIG_SIZE[1] + LEGEND_HEADROOM_IN
    ax_y  = py / fig_h
    ax_h  = ph / fig_h

    fig   = plt.figure(figsize=(fig_w, fig_h), facecolor="#FFFFFF")
    ax_low  = fig.add_axes([left_in / fig_w, ax_y, pw / fig_w, ax_h])
    x_high  = (left_in + pw + GAP_BAND_IN) / fig_w
    ax_high = fig.add_axes([x_high, ax_y, pw / fig_w, ax_h])

    # Titles
    ax_low.set_title("Low Band  (ref = 1g)",  fontsize=FONT_LABEL, fontweight="bold", pad=6)
    ax_high.set_title("High Band  (ref = 26g)", fontsize=FONT_LABEL, fontweight="bold", pad=6)

    plot_region_band(ax_low,  "Low",  low_order,  low_region_pvals,
                     show_xlabel=True, show_ylabel=True)
    plot_region_band(ax_high, "High", high_order, high_region_pvals,
                     show_xlabel=True, show_ylabel=False)

    fig.legend(
        handles=REGION_LEGEND_ELEMENTS,
        loc="upper center",
        bbox_to_anchor=(0.5, FIG_LEGEND_ANCHOR_Y),
        bbox_transform=fig.transFigure,
        ncol=2, fontsize=FONT_LABEL, frameon=False,
        columnspacing=2.0, handletextpad=0.5, handlelength=1.6,
    )
    return fig


fig_region = make_region_figure()
save_gee_figure(fig_region, "gee_region_onnail_vs_offnail")
plt.close(fig_region)
print("Saved: On-nail vs Off-nail regional comparison figure")