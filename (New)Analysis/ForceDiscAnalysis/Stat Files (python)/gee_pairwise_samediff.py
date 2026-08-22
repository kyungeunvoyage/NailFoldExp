"""
gee_pairwise_samediff.py
========================
Plot P76–P90 Same/Different force-discrimination accuracy in the EXACT same
style as gee_pairwise_plot_horizontal_2col(Final).png (Stats(GEE).py), but
using SameDiff data and TRIANGLE scatter points.

Output
------
  Output/SameDiff_GEE/gee_pairwise_samediff_horizontal_2col.png  (2102 px wide)
  Final/gee_pairwise_samediff_horizontal_2col(Final).png
"""

import os
import glob
import itertools
import importlib.util
import shutil

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"]     = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

# ── ATD C1 style ──────────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent

def _load_atd_c1():
    root = _SCRIPT_DIR.parent.parent / "ATDAnalysis"
    for sub in ("Stat files", "Stat files (final) "):
        path = root / sub / "(Final)ATD_C1_Fig(Anika).py"
        if path.is_file():
            spec = importlib.util.spec_from_file_location("atd_c1_fig", path)
            mod  = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    raise FileNotFoundError("Could not find (Final)ATD_C1_Fig(Anika).py")

ATD = _load_atd_c1()

from gee_export_utils import (
    EXPORT_CANVAS,
    EXPORT_WIDTH_2COL,
    EXPORT_HEIGHT_2COL,
    ON_TOUCH_BLUE,
    XLABEL_FORCE_PAIR,
    add_figure_legend,
    horizontal_panel_rects,
    on_touch_box_color,
    on_touch_scatter_rgba,
    save_export_figure,
)

# ── Colors / style (identical to Stats(GEE).py) ───────────────────────────────
BOX_STROKE       = "#000000"
ACCENT_RED       = ATD.ACCENT_RED
CRITERION_COLOR  = ATD.CRITERION_COLOR
REF_LINE_ZORDER  = ATD.REF_LINE_ZORDER
FONT_TICK        = 20
FONT_XTICK       = 14   # force-pair labels; smaller than y ticks (ref screenshot)
FONT_LABEL       = 20
FONT_ANNOT       = ATD.FONT_ANNOT
BOX_LINEWIDTH    = ATD.BOX_LINEWIDTH
MEDIAN_LINEWIDTH = 2.0
MEDIAN_ZORDER    = 15
BOX_PATCH_ZORDER = 1
WHISKER_ZORDER   = 4
SCATTER_ZORDER   = 5
STRIP_ALPHA      = ATD.STRIP_ALPHA
STRIP_SIZE       = 3.8
BLACK            = ATD.BLACK
JND_PCT          = 75.0
CHANCE_PCT       = 50.0
BOX_FILL_COLOR   = "#E3EDF7"
SCATTER_RGBA     = on_touch_scatter_rgba(ATD)

# ── Box width constants (exact copy from Stats(GEE).py) ───────────────────────
BOX_WIDTH_REF_N    = 3          # len(high_order) = 3
BOX_WIDTH_AT_REF   = 0.42
STRIP_JITTER_AT_REF = 0.14

def boxplot_width(n_pairs):
    return BOX_WIDTH_AT_REF * n_pairs / BOX_WIDTH_REF_N

def jitter(n, width=0.1, seed=42):
    rng = np.random.default_rng(seed)
    return rng.uniform(-width / 2, width / 2, n)

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR   = Path("/Users/kyungeunjung/NailFoldExp/Data/(FD)CurData")
OUTPUT_DIR = Path("/Users/kyungeunjung/NailFoldExp/(New)Analysis/ForceDiscAnalysis/Output/SameDiff_GEE")
FINAL_DIR  = Path("/Users/kyungeunjung/NailFoldExp/(New)Analysis/ForceDiscAnalysis/Final")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
low_files  = sorted(DATA_DIR.glob("P*_ForceDiscrimination_SameDiff.csv"))
high_files = sorted(DATA_DIR.glob("P*_ForceDiscrimination_SameDiff_26g.csv"))

if not low_files and not high_files:
    raise FileNotFoundError(f"No SameDiff CSV files found in {DATA_DIR}")

frames = []
for f in low_files:
    df_f = pd.read_csv(f, encoding="utf-8-sig")
    df_f["band"] = "Low"
    frames.append(df_f)
for f in high_files:
    df_f = pd.read_csv(f, encoding="utf-8-sig")
    df_f["band"] = "High"
    frames.append(df_f)

df = pd.concat(frames, ignore_index=True)
print(f"Loaded {len(low_files)} Low-band + {len(high_files)} High-band files "
      f"({df['Subject'].nunique()} subjects)")

# ── Derived columns ────────────────────────────────────────────────────────────
df["correct"] = df["IsCorrect"].astype(int)
df["pair_label"] = df.apply(
    lambda r: f"{min(r['Reference'], r['Comparison']):g}–"
              f"{max(r['Reference'], r['Comparison']):g}",
    axis=1,
)

# ── Per-subject accuracy per pair per band ────────────────────────────────────
subj_acc = (
    df.groupby(["Subject", "band", "pair_label"])["correct"]
    .mean()
    .reset_index()
    .rename(columns={"correct": "accuracy"})
)

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
    return [p for p in fixed if p in actual]

low_order  = fix_order(["0.4–1", "0.6–1", "1–1.4", "1–2"], actual_labels)
high_order = fix_order(["10–26", "15–26", "26–60"],          actual_labels)
print("Low band pairs:", low_order)
print("High band pairs:", high_order)

# ── Axes finalize (exact copy from Stats(GEE).py) ─────────────────────────────
def finalize_gee_axes(ax, n_x, ylim_top, *, show_ylabel=True, show_xlabel=True):
    ax.set_ylim(ATD.ACCURACY_YMIN, min(ATD.FIG2_BRACKET_YLIM_CAP, ylim_top))
    ax.set_yticks(ATD.ACCURACY_YTICKS)
    ax.grid(False)
    ax.tick_params(axis="y", which="both", length=0, labelsize=FONT_TICK)
    ax.tick_params(axis="x", which="both", length=0, labelsize=FONT_XTICK)
    if show_ylabel:
        ax.set_ylabel("Discrimination Accuracy (%)", fontsize=FONT_LABEL,
                      labelpad=ATD.FIG_AXIS_LABELPAD)
    if show_xlabel:
        ax.set_xlabel(XLABEL_FORCE_PAIR, fontsize=FONT_LABEL,
                      labelpad=ATD.FIG_AXIS_LABELPAD)
    sns.despine(ax=ax)
    ATD.apply_accuracy_y_spine_bounds(ax)
    ATD.add_inward_tick_guides(ax, n_x)
    # add_inward_tick_guides resets labelsize — restore y vs x separately
    ax.tick_params(axis="y", which="both", length=0, labelsize=FONT_TICK)
    ax.tick_params(axis="x", which="both", length=0, labelsize=FONT_XTICK)
    ATD.apply_accuracy_y_spine_bounds(ax)

# ── Plot band (same as Stats(GEE).py but marker="^") ─────────────────────────
def plot_band(ax, band_label, order, show_xlabel=True, show_ylabel=True,
              *, ylim_top_override=None):
    sub = subj_acc[subj_acc["band"] == band_label].copy()
    bw          = boxplot_width(len(order))
    ref_bw      = boxplot_width(BOX_WIDTH_REF_N)
    strip_jitter = STRIP_JITTER_AT_REF * bw / ref_bw
    lw   = BOX_LINEWIDTH
    band_max_pct = 0.0

    for xi, pair in enumerate(order):
        pdata_pct = sub.loc[sub["pair_label"] == pair, "accuracy"].values * 100.0
        if len(pdata_pct) == 0:
            print(f"  WARNING: no data for pair '{pair}' in band '{band_label}'")
            continue
        band_max_pct = max(band_max_pct, float(np.max(pdata_pct)))

        bp = ax.boxplot(
            [pdata_pct], positions=[xi], widths=bw,
            patch_artist=True, showfliers=False,
            capwidths=ATD.CAP_WIDTH,
            whiskerprops={"linewidth": lw, "color": BOX_STROKE, "solid_capstyle": "butt"},
            capprops={"linewidth": lw, "color": BOX_STROKE, "solid_capstyle": "butt"},
            medianprops={"color": ACCENT_RED, "linewidth": MEDIAN_LINEWIDTH},
            boxprops={"linewidth": lw, "edgecolor": BOX_STROKE},
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(BOX_FILL_COLOR)
            patch.set_edgecolor(BOX_STROKE)
            patch.set_linewidth(lw)
            patch.set_alpha(1.0)
            patch.set_zorder(BOX_PATCH_ZORDER)
        for key in ("whiskers", "caps"):
            for line in bp[key]:
                line.set_color(BOX_STROKE)
                line.set_linewidth(lw)
                line.set_alpha(1.0)
                line.set_zorder(WHISKER_ZORDER)
        for line in bp["medians"]:
            line.set_color(ACCENT_RED)
            line.set_linewidth(MEDIAN_LINEWIDTH)
            line.set_alpha(1.0)
            line.set_zorder(MEDIAN_ZORDER)

        x_strip = np.full(len(pdata_pct), xi) + jitter(len(pdata_pct), width=strip_jitter)
        ax.scatter(
            x_strip, pdata_pct,
            c=[SCATTER_RGBA] * len(pdata_pct),
            s=STRIP_SIZE ** 2,
            linewidths=0,
            edgecolors="none",
            alpha=STRIP_ALPHA,
            zorder=SCATTER_ZORDER,
            marker="^",          # ← triangle
            clip_on=False,
        )

    ax.axhline(JND_PCT, color=CRITERION_COLOR, linestyle="--", linewidth=1.0,
               alpha=0.85, zorder=REF_LINE_ZORDER)
    ax.axhline(CHANCE_PCT, color=BLACK, linestyle=":", linewidth=0.8,
               alpha=0.5, zorder=1)

    if ylim_top_override is not None:
        ylim_top = ylim_top_override
    else:
        ylim_top = min(ATD.FIG2_BRACKET_YLIM_CAP,
                       max(ATD.ACCURACY_YLIM_TOP, band_max_pct + 8.0))
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, fontsize=FONT_XTICK)
    ax.set_xlim(-0.55, len(order) - 0.45)
    finalize_gee_axes(ax, len(order), ylim_top,
                      show_ylabel=show_ylabel, show_xlabel=show_xlabel)

# ── Legend (same as Stats(GEE).py) ────────────────────────────────────────────
LEGEND_ELEMENTS = [
    mpatches.Patch(facecolor=BOX_FILL_COLOR, edgecolor=BOX_STROKE,
                   linewidth=BOX_LINEWIDTH, label="Low band (ref = 1 g)"),
    mpatches.Patch(facecolor=BOX_FILL_COLOR, edgecolor=BOX_STROKE,
                   linewidth=BOX_LINEWIDTH, label="High band (ref = 26 g)"),
]

# ── Build & save figure ────────────────────────────────────────────────────────
sns.set_theme(style="white")
ATD.apply_plot_style()

fig = plt.figure(figsize=EXPORT_CANVAS, facecolor="#FFFFFF")
# Wider panels for 2640×1042: tighter outer margins, keep inter-panel gap
low_r, high_r = horizontal_panel_rects(
    left=0.10, right=0.04, bottom=0.12, top=0.90, gap_frac=0.14,
)
ax_low  = fig.add_axes(low_r)
ax_high = fig.add_axes(high_r)

all_max_pct = 0.0
for band_label, order in [("Low", low_order), ("High", high_order)]:
    sub = subj_acc[subj_acc["band"] == band_label]
    for pair in order:
        pdata = sub.loc[sub["pair_label"] == pair, "accuracy"].values * 100.0
        if len(pdata):
            all_max_pct = max(all_max_pct, float(np.max(pdata)))
shared_ylim_top = min(ATD.FIG2_BRACKET_YLIM_CAP,
                      max(ATD.ACCURACY_YLIM_TOP, all_max_pct + 8.0))

plot_band(ax_low,  "Low",  low_order,  show_xlabel=False, show_ylabel=True,
          ylim_top_override=shared_ylim_top)
plot_band(ax_high, "High", high_order, show_xlabel=False, show_ylabel=False,
          ylim_top_override=shared_ylim_top)

stem = "gee_pairwise_samediff_horizontal"
save_export_figure(fig, str(OUTPUT_DIR), stem, ATD.EXPORT_WIDTHS_PX)

# Extra wide canvas: 2640 × 1042
EXPORT_W_2640 = 2640
EXPORT_H_2640 = 1042
save_export_figure(
    fig, str(OUTPUT_DIR), stem, (("2640", EXPORT_W_2640),),
    letterbox=True,
    margin_frac_x=0.06,  # tighter outer pad → larger panels on 2640×1042
    margin_frac_y=0.05,
    trim_white=True,
    fixed_height_px=EXPORT_H_2640,
)
plt.close(fig)

# Publish 2col + 2640 versions to Final
for tag, dest_name in (
    ("2col", "(Final)gee_pairwise_samediff_horizontal_2col.png"),
    ("2640", "(Final)gee_pairwise_samediff_horizontal_2640.png"),
):
    src = OUTPUT_DIR / f"{stem}_{tag}.png"
    dst = FINAL_DIR / dest_name
    if src.is_file():
        shutil.copy2(src, dst)
        print(f"Published → {dst}")
    else:
        print(f"WARNING: source not found: {src}")
