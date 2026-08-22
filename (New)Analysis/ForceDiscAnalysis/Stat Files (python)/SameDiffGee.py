"""
Force Discrimination – Same/Different 2AFC GEE Analysis
=========================================================
Adapted from the original "which is stronger" GEE pairwise script.

CSV column assumptions (P*_ForceDiscrimination_SameDiff.csv, P*_ForceDiscrimination_SameDiff_26g.csv)
-------------------------------------------------------------
Subject, Session, Condition, Region, Reference, Comparison,
Rep, TrialType (same_ref / same_comp / diff_rc / diff_cr),
Stim1, Stim2, GroundTruth (SAME / DIFFERENT),
UserChoice (SAME / DIFFERENT), IsCorrect (1 / 0)

Accuracy definition
--------------------
IsCorrect is already computed in the CSV (1 = correct, 0 = incorrect).
Chance level  = 50%   (2AFC: SAME or DIFFERENT)
JND threshold = 75%   (halfway between chance and ceiling)

Additional analyses vs. original script
-----------------------------------------
1. Hit Rate / False Alarm Rate / d′ / criterion c per pair × region group
   (Signal Detection Theory decomposition — separates sensitivity from bias)
2. Order effect check: diff_rc vs diff_cr accuracy compared per pair
   (directly tests whether stimulus presentation order affected performance)
3. Response bias overview (3-panel stacked bars, analogous to response_bias_overview.py)
   — delivered SAME/DIFFERENT, response distribution, incorrect-trial responses

Statistics priority:
  1. statsmodels GEE (binomial family, subject clustering) — preferred
  2. scipy Wilcoxon signed-rank test (fallback)
  3. Permutation test (final fallback)
"""

import os
import glob
import itertools
import importlib.util
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"]     = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.transforms import blended_transform_factory
from matplotlib.ticker import FixedLocator
from pathlib import Path
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ── Stat backend ──────────────────────────────────────────────────────────────
USE_GEE = False
USE_WILCOXON = False

try:
    from statsmodels.genmod.generalized_estimating_equations import GEE
    from statsmodels.genmod.families import Binomial
    USE_GEE = True
except ImportError:
    try:
        from scipy.stats import wilcoxon
        USE_WILCOXON = True
    except ImportError:
        pass

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = "/Users/kyungeunjung/NailFoldExp/Data/(FD)CurData"
FILE_PATTERNS = [
    os.path.join(DATA_DIR, "P*_ForceDiscrimination_SameDiff.csv"),
    os.path.join(DATA_DIR, "P*_ForceDiscrimination_SameDiff_26g.csv"),
]
OUTPUT_DIR = "/Users/kyungeunjung/NailFoldExp/(New)Analysis/ForceDiscAnalysis/Output/SameDiff_GEE"
os.makedirs(OUTPUT_DIR, exist_ok=True)

from gee_export_utils import (
    EXPORT_CANVAS,
    EXPORT_WIDTH_2COL,
    EXPORT_HEIGHT_2COL,
    ON_TOUCH_BLUE,
    XLABEL_FORCE_PAIR,
    horizontal_panel_rects,
    on_touch_box_color,
    on_touch_scatter_rgba,
    POOLED_BOX_REF,
    add_figure_legend,
)


CHANCE_PCT = 50.0
JND_PCT    = 75.0
COMBINED_PANEL_COUNT = 2
STRIP_JITTER_REF = 0.12
BAND_BASE_COLORS = {"Low": ON_TOUCH_BLUE, "High": ON_TOUCH_BLUE}


def _load_atd_c1():
    root = Path(__file__).resolve().parent.parent.parent / "ATDAnalysis"
    for sub in ("Stat files", "Stat files (final) "):
        path = root / sub / "(Final)ATD_C1_Fig(Anika).py"
        if path.is_file():
            spec = importlib.util.spec_from_file_location("atd_c1", path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    raise FileNotFoundError(f"Could not find (Final)ATD_C1_Fig(Anika).py under {root}")


ATD = _load_atd_c1()

ATD_FIG2_REF_N = len(getattr(ATD, "USER_FORCES", [])) or 5
FIG2_DODGED_BOX_WIDTH = ATD.FIG2_DODGED_BOX_WIDTH  # mpl_boxplot_width(ref_n) on full canvas


def band_box_color(band_label=None):
    del band_label
    return on_touch_box_color(ATD)


def band_scatter_rgba(band_label=None):
    del band_label
    return on_touch_scatter_rgba(ATD)

BOX_LINEWIDTH = ATD.BOX_LINEWIDTH
CAP_WIDTH = ATD.CAP_WIDTH
BOX_STROKE = "#000000"
CRITERION_COLOR = ATD.CRITERION_COLOR
REF_LINE_ZORDER = ATD.REF_LINE_ZORDER
ACCENT_RED = ATD.ACCENT_RED
FONT_TICK = ATD.FONT_TICK
FONT_LABEL = ATD.FONT_LABEL
MEDIAN_LINEWIDTH = 2.0
BRACKET_BASE_PAD = 3.0
BRACKET_TIER_STEP = 8.0
BRACKET_YLIM_CAP = 122.0
SAVE_DPI_COMBINED = 600
POOL_ON_NAIL = ON_TOUCH_BLUE
POOL_OFF_NAIL = ON_TOUCH_BLUE
POOL_GROUP_ORDER = ["Off-nail", "On-nail"]
POOL_PALETTE = {"On-nail": POOL_ON_NAIL, "Off-nail": POOL_OFF_NAIL}
POOL_X_LABELS = ["Lateral\n(a+f)", "Proximal\n(c+d)"]
# Within-panel box placement (leave room for two-line xticklabels)
POOL_X_POS = (0.0, 0.68)
POOL_XLIM = (-0.38, 1.06)
POOL_WSPACE = 0.05
POOL_MARGINS = dict(left=0.10, right=0.995, top=0.96, bottom=0.14)
# Target export canvas
ONNAIL_EXPORT_WIDTH_PX = 1385
ONNAIL_EXPORT_HEIGHT_PX = 1042
ONNAIL_EXPORT_DPI = 200
# Box width in data coords (kept at half of earlier wide setting;
# fonts/linewidths below track the 1385×1042 canvas)
ONNAIL_BOX_WIDTH = 0.275
# Fonts / markers sized for 1385×1042
FONT_XTICK = 16
FONT_XTICK_LOW = 14
FONT_PANEL_TITLE = 12
FONT_BRACKET_STAR = 13
POOLED_Y_TICKS = list(ATD.ACCURACY_YTICKS)
POOLED_YLIM_TOP = 105
POOLED_Y_AXIS_TOP = 100
TICK_LEN_AXES = getattr(ATD, "TICK_LEN_AXES", 0.016)
CAP_LINEWIDTH = ATD.CAP_LINEWIDTH
BLACK = ATD.BLACK
PARTIAL_SUBJECTS = getattr(ATD, "_PARTIAL_SUBJ", set())
BRACKET_LINEWIDTH = 1.5
SCATTER_DOT_SIZE = 3.5
ONNAIL_FONT_LABEL = 14
ONNAIL_BOX_LINEWIDTH = 1.15
ONNAIL_MEDIAN_LINEWIDTH = 2.2

GEE_LEGEND_KW = dict(
    frameon=False,
    fontsize=FONT_LABEL,
    columnspacing=2.0,
    handletextpad=0.5,
    handlelength=1.6,
)


def _gee_ax_legend(ax, handles, **kwargs):
    kw = {**GEE_LEGEND_KW, **kwargs}
    return ax.legend(handles=handles, **kw)


def _finalize_gee_accuracy_axes(ax, n_x, ylim_top, *, show_ylabel=True, show_xlabel=True):
    fs_tick = FONT_TICK
    fs_label = FONT_LABEL
    labelpad = ATD.FIG_AXIS_LABELPAD
    ax.set_ylim(ATD.ACCURACY_YMIN, min(ATD.FIG2_BRACKET_YLIM_CAP, ylim_top))
    ax.set_yticks(ATD.ACCURACY_YTICKS)
    ax.grid(False)
    ax.tick_params(axis="both", which="both", length=0, labelsize=fs_tick)
    if show_ylabel:
        ax.set_ylabel(
            "Discrimination Accuracy (%)",
            fontsize=fs_label,
            labelpad=labelpad,
        )
    if show_xlabel:
        ax.set_xlabel(
            XLABEL_FORCE_PAIR,
            fontsize=fs_label,
            labelpad=labelpad,
        )
    sns.despine(ax=ax)
    ATD.apply_accuracy_y_spine_bounds(ax)
    ATD.add_inward_tick_guides(ax, n_x)
    # add_inward_tick_guides resets labelsize to ATD.FONT_TICK (16) — restore
    ax.tick_params(axis="both", which="both", length=0, labelsize=fs_tick)
    ATD.apply_accuracy_y_spine_bounds(ax)


def _pooled_star_label(p):
    if np.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def _pooled_sig_bracket(ax, x_l, x_r, y_base, text="", tick_h=0.5, text_pad=0.0,
                        *, fontsize=FONT_BRACKET_STAR, linewidth=BRACKET_LINEWIDTH):
    y_top = y_base + tick_h
    ax.plot(
        [x_l, x_r], [y_top, y_top],
        color=ACCENT_RED, linewidth=linewidth, clip_on=False, zorder=25,
    )
    if text:
        ax.text(
            (x_l + x_r) / 2, y_top + text_pad, text,
            ha="center", va="bottom", fontsize=fontsize,
            color=ACCENT_RED, fontweight="bold", clip_on=False, zorder=26,
        )


def _pooled_panel_title(ax, pair_label, *, y=0.95, clip_on=False, fontsize=FONT_PANEL_TITLE):
    ax.text(
        0.5, y, f"{pair_label} (g)", transform=ax.transAxes,
        ha="center", va="bottom", fontsize=fontsize,
        fontweight="normal", clip_on=clip_on,
    )


def _pooled_scatter_strip(ax, x_pos, vals, subjects, rgba, jitter_arr, *, dot_size=SCATTER_DOT_SIZE):
    mask = np.array([s in PARTIAL_SUBJECTS for s in subjects])
    kw = dict(linewidths=0, zorder=3, clip_on=False)
    if (~mask).any():
        ax.scatter(
            x_pos + jitter_arr[~mask], vals[~mask],
            c=[rgba] * int((~mask).sum()), s=dot_size ** 2,
            marker="o", **kw,
        )
    if mask.any():
        ax.scatter(
            x_pos + jitter_arr[mask], vals[mask],
            c=[rgba] * int(mask.sum()), s=(dot_size * 1.3) ** 2,
            marker="^", **kw,
        )


def _pooled_inward_ticks(ax, y_ticks):
    frac_x = TICK_LEN_AXES
    frac_y = ATD.y_tick_frac_match_x(ax, frac_x)
    ax.tick_params(axis="both", which="both", length=0)
    x_tr = blended_transform_factory(ax.transData, ax.transAxes)
    y_tr = blended_transform_factory(ax.transAxes, ax.transData)
    kw = dict(color=BLACK, linewidth=1.0, solid_capstyle="butt", clip_on=False, zorder=6)
    for xi in POOL_X_POS:
        ax.plot([xi, xi], [0, frac_x], transform=x_tr, **kw)
    y_lo, y_hi = ax.get_ylim()
    for y in y_ticks:
        if y_lo - 1e-9 <= y <= y_hi + 1e-9:
            ax.plot([0, frac_y], [y, y], transform=y_tr, **kw)


def _draw_pooled_pair_panel(ax, pair, subj_acc_reg, region_pval, rng, *, box_width=0.45,
                            clip_titles=False, show_bracket=True, show_title=True,
                            xtick_fs=None):
    """One force-pair panel — On-nail vs Off-nail on x (ATD pooled layout)."""
    if xtick_fs is None:
        xtick_fs = FONT_XTICK
    tops = {}
    scatter_rgba = on_touch_scatter_rgba(ATD)
    box_fill = on_touch_box_color(ATD)
    jitter_span = STRIP_JITTER_REF * box_width / POOLED_BOX_REF
    for xi, grp in enumerate(POOL_GROUP_ORDER):
        x_pos = POOL_X_POS[xi]
        rows = subj_acc_reg[
            (subj_acc_reg["pair_label"] == pair)
            & (subj_acc_reg["region_group"] == grp)
        ]
        vals = rows["accuracy"].values * 100
        subjects = rows["Subject"].values
        if len(vals) == 0:
            continue
        bp = ax.boxplot(
            [vals], positions=[x_pos], widths=box_width,
            patch_artist=True, showfliers=False, zorder=2,
            whiskerprops=dict(color=BLACK, linewidth=ONNAIL_BOX_LINEWIDTH),
            capprops=dict(color=BLACK, linewidth=ONNAIL_BOX_LINEWIDTH),
            medianprops=dict(color=ACCENT_RED, linewidth=ONNAIL_MEDIAN_LINEWIDTH),
            boxprops=dict(
                facecolor=box_fill,
                edgecolor=BLACK, linewidth=ONNAIL_BOX_LINEWIDTH,
            ),
        )
        if grp == "Off-nail":
            for patch in bp["boxes"]:
                patch.set_hatch("////")
                patch.set_edgecolor((*mcolors.to_rgb(BLACK), 0.30))
                # redraw box outline solid on top
                import matplotlib.patches as _mp
                verts = patch.get_path().vertices
                x0 = verts[:, 0].min(); x1 = verts[:, 0].max()
                y0 = verts[:, 1].min(); y1 = verts[:, 1].max()
                rect = _mp.Rectangle(
                    (x0, y0), x1 - x0, y1 - y0,
                    linewidth=ONNAIL_BOX_LINEWIDTH, edgecolor=BLACK,
                    facecolor="none", zorder=patch.get_zorder() + 0.5,
                )
                patch.axes.add_patch(rect)
        whiskers = [w.get_ydata()[1] for w in bp["whiskers"]]
        tops[grp] = max(whiskers) if whiskers else float(np.max(vals))
        jitter = rng.uniform(-jitter_span, jitter_span, size=len(vals))
        _pooled_scatter_strip(ax, x_pos, vals, subjects, scatter_rgba, jitter)

    if show_bracket and tops and not np.isnan(region_pval):
        star = _pooled_star_label(region_pval)
        if clip_titles:
            y_brk = min(max(max(tops.values()) + 2.0, 82), 88)
            title_y = 0.62
        else:
            y_brk = max(max(tops.values()) + 3.0, 103)
            title_y = 0.95
        _pooled_sig_bracket(ax, POOL_X_POS[0], POOL_X_POS[1], y_brk, text=star)
    else:
        title_y = 0.88 if clip_titles else 0.95

    ax.axhline(
        JND_PCT, color=CRITERION_COLOR, linestyle="--",
        linewidth=1.0, alpha=0.85, zorder=20,
    )
    if show_title:
        _pooled_panel_title(ax, pair, y=title_y, clip_on=clip_titles)
    ax.set_xlim(*POOL_XLIM)
    ax.set_xticks(list(POOL_X_POS))
    ax.set_xticklabels(POOL_X_LABELS, fontsize=xtick_fs)
    ax.set_yticks(POOLED_Y_TICKS)
    ax.yaxis.set_major_locator(FixedLocator(POOLED_Y_TICKS))
    ax.tick_params(axis="y", labelsize=xtick_fs)
    ax.tick_params(axis="x", length=0, labelsize=xtick_fs)
    ax.set_ylim(-5, POOLED_YLIM_TOP)
    ax.spines["left"].set_bounds(-5, POOLED_Y_AXIS_TOP)
    sns.despine(ax=ax)


def save_region_onnail_figure(subj_acc_reg, pair_order, region_pvals, out_path,
                              *, box_width=None, show_title=True,
                              width_px=None, height_px=None, compensate_width=True,
                              xtick_fs=None):
    """Multi-panel On-nail vs Off-nail — layout sized to the export canvas."""
    if xtick_fs is None:
        xtick_fs = FONT_XTICK
    sns.set_theme(style="white")
    ATD.apply_plot_style()
    n_panels = len(pair_order)
    export_w = width_px  if width_px  is not None else ONNAIL_EXPORT_WIDTH_PX
    export_h = height_px if height_px is not None else ONNAIL_EXPORT_HEIGHT_PX
    dpi = ONNAIL_EXPORT_DPI

    if box_width is None:
        # Target ~58 px box width on the export canvas
        usable_px = export_w * (POOL_MARGINS["right"] - POOL_MARGINS["left"])
        panel_px = usable_px / (n_panels + (n_panels - 1) * POOL_WSPACE)
        data_span = POOL_XLIM[1] - POOL_XLIM[0]
        target_box_px = 58.0 * (export_h / 1042.0)  # track canvas height a bit
        box_width = (target_box_px / max(panel_px, 1.0)) * data_span
        box_width = float(np.clip(box_width, 0.18, 0.40))
    if compensate_width:
        box_width = box_width * (ONNAIL_EXPORT_WIDTH_PX / export_w)

    # Exact inch size so savefig(dpi=…) lands on export_w × export_h (no stretch)
    fig_w = export_w / dpi
    fig_h = export_h / dpi
    fig, axes = plt.subplots(1, n_panels, figsize=(fig_w, fig_h), dpi=dpi,
                             facecolor="white")
    if n_panels == 1:
        axes = [axes]
    rng = np.random.default_rng(42)
    for ax, pair in zip(axes, pair_order):
        pval = region_pvals.get(pair, np.nan)
        _draw_pooled_pair_panel(ax, pair, subj_acc_reg, pval, rng,
                                box_width=box_width, show_bracket=False,
                                show_title=show_title, xtick_fs=xtick_fs)
        if ax is axes[0]:
            ax.set_ylabel("Discrimination Accuracy (%)", fontsize=ONNAIL_FONT_LABEL)
        else:
            ax.set_ylabel("")
            ax.tick_params(axis="y", labelleft=False)

    fig.subplots_adjust(wspace=POOL_WSPACE, **POOL_MARGINS)
    fig.canvas.draw()
    for ax in axes:
        _pooled_inward_ticks(ax, POOLED_Y_TICKS)

    # Exact canvas — do not use bbox_inches='tight' (it breaks size adaptation)
    fig.savefig(
        out_path, dpi=dpi, facecolor="white", edgecolor="none",
        bbox_inches=None, pad_inches=0,
    )
    plt.close(fig)

ON_NAIL  = ["C", "D"]
OFF_NAIL = ["A", "F"]

BAND_CONFIG = {
    "Low": {
        "ref": 1,
        "pair_order": ["0.4–1", "0.6–1", "1–1.4", "1–2"],
        "suffix": "_low",
        "title_ref": "1 g",
    },
    "High": {
        "ref": 26,
        "pair_order": ["10–26", "15–26", "26–60"],
        "suffix": "_high",
        "title_ref": "26 g",
    },
}


def band_title_text(band_label, title_ref, n_subjects):
    return f"{band_label} band (ref = {title_ref}, n = {n_subjects})"


def combined_panel_box_width(n_pairs):
    """ATD Fig2 dodged-box pixel width on a half-width 2-col panel."""
    return (
        ATD.mpl_boxplot_width(n_pairs, reference_n=ATD_FIG2_REF_N)
        * COMBINED_PANEL_COUNT
    )


def save_png_at_width(fig, out_path, width_px=EXPORT_WIDTH_2COL, *,
                      height_px=None, pad_inches=0.04, dpi=150):
    import io
    from PIL import Image

    buf = io.BytesIO()
    fig.savefig(
        buf, format="png", dpi=dpi, bbox_inches="tight",
        pad_inches=pad_inches, facecolor="white",
    )
    buf.seek(0)
    master = Image.open(buf).convert("RGB")
    if height_px is None:
        height_px = round(width_px * master.height / master.width)
    master.resize((width_px, height_px), Image.Resampling.LANCZOS).save(out_path)


# ── Load data ─────────────────────────────────────────────────────────────────
import re

all_files = sorted(set(
    f for pat in FILE_PATTERNS for f in glob.glob(pat)
))
if not all_files:
    raise FileNotFoundError(
        "No CSV files found matching:\n  " + "\n  ".join(FILE_PATTERNS)
    )

def subject_number(filepath):
    m = re.search(r"P(\d+)", os.path.basename(filepath))
    return int(m.group(1)) if m else 0

files = sorted([f for f in all_files if subject_number(f) > 73])
if not files:
    raise ValueError(
        f"No files with subject number > 73 found.\n"
        f"Files present: {sorted(os.path.basename(f) for f in all_files)}"
    )
df = pd.concat(
    [pd.read_csv(f, encoding="utf-8-sig") for f in files],
    ignore_index=True,
)

df["correct"] = df["IsCorrect"].astype(int)

df["pair_label"] = df.apply(
    lambda r: f"{min(r['Reference'], r['Comparison']):g}–{max(r['Reference'], r['Comparison']):g}",
    axis=1,
)

df["band"] = df["Reference"].map({1: "Low", 26: "High"})

df["region_group"] = df["Region"].map(
    {r: "On-nail"  for r in ON_NAIL} |
    {r: "Off-nail" for r in OFF_NAIL}
)

def fix_order(order, actual):
    fixed = []
    for p in order:
        if p in actual:
            fixed.append(p)
        else:
            alt = [a for a in actual if set(a.replace("–", "-").split("-")) == set(p.replace("–", "-").split("-"))]
            fixed.append(alt[0] if alt else p)
    return fixed


# ── SDT helper ────────────────────────────────────────────────────────────────
def compute_sdt(sub_df):
    diff_trials = sub_df[sub_df["GroundTruth"] == "DIFFERENT"]
    same_trials = sub_df[sub_df["GroundTruth"] == "SAME"]
    n_diff = len(diff_trials)
    n_same = len(same_trials)
    if n_diff == 0 or n_same == 0:
        return {"H": np.nan, "FA": np.nan, "d_prime": np.nan, "criterion": np.nan}

    hits = (diff_trials["UserChoice"] == "DIFFERENT").sum()
    fas  = (same_trials["UserChoice"]  == "DIFFERENT").sum()
    H  = (hits + 0.5) / (n_diff + 1)
    FA = (fas  + 0.5) / (n_same + 1)

    from scipy.stats import norm
    d_prime   = norm.ppf(H) - norm.ppf(FA)
    criterion = -0.5 * (norm.ppf(H) + norm.ppf(FA))
    return {"H": H, "FA": FA, "d_prime": d_prime, "criterion": criterion}


def _permutation_pval(a, b, n_perm=5000, seed=0):
    rng = np.random.default_rng(seed)
    obs = abs(np.mean(a) - np.mean(b))
    combined = np.concatenate([a, b])
    n_a = len(a)
    count = sum(
        abs(np.mean(combined[:n_a]) - np.mean(combined[n_a:])) >= obs
        for _ in range(n_perm)
        if not rng.shuffle(combined) or True
    )
    return count / n_perm


def run_gee_pairwise(df_band, subj_acc, pair_order):
    results = {}
    subj_means = subj_acc.copy()

    for p1, p2 in itertools.combinations(pair_order, 2):
        if USE_GEE:
            chunk = df_band[df_band["pair_label"].isin([p1, p2])].copy()
            if chunk["Subject"].nunique() < 2:
                results[(p1, p2)] = np.nan
                continue
            chunk["pair_dummy"] = (chunk["pair_label"] == p2).astype(int)
            chunk = chunk.rename(columns={"Subject": "subj_id"})
            try:
                fit = GEE.from_formula("correct ~ pair_dummy", groups="subj_id",
                                        data=chunk, family=Binomial()).fit(maxiter=60)
                results[(p1, p2)] = fit.pvalues["pair_dummy"]
                continue
            except Exception:
                pass

        paired = subj_means[subj_means["pair_label"].isin([p1, p2])]\
                 .pivot(index="Subject", columns="pair_label", values="accuracy").dropna()
        if len(paired) < 5:
            results[(p1, p2)] = np.nan
            continue
        if USE_WILCOXON:
            try:
                from scipy.stats import wilcoxon
                _, pval = wilcoxon(paired[p1].values, paired[p2].values)
                results[(p1, p2)] = pval
                continue
            except Exception:
                pass
        results[(p1, p2)] = _permutation_pval(paired[p1].values, paired[p2].values)

    return results


def run_gee_region(df_band, pair_order):
    df_reg = df_band[df_band["region_group"].notna()].copy()
    results = {}

    for pair in pair_order:
        chunk = df_reg[df_reg["pair_label"] == pair].copy()
        if chunk["Subject"].nunique() < 2:
            results[pair] = np.nan
            continue
        chunk["region_dummy"] = (chunk["region_group"] == "On-nail").astype(int)
        chunk = chunk.rename(columns={"Subject": "subj_id"})

        if USE_GEE:
            try:
                fit = GEE.from_formula("correct ~ region_dummy", groups="subj_id",
                                        data=chunk, family=Binomial()).fit(maxiter=60)
                results[pair] = fit.pvalues["region_dummy"]
                continue
            except Exception:
                pass

        subj_reg = (
            df_reg[df_reg["pair_label"] == pair]
            .groupby(["Subject", "region_group"])["correct"].mean().reset_index()
        )
        pivot = subj_reg.pivot(index="Subject", columns="region_group", values="correct").dropna()
        if len(pivot) < 5:
            results[pair] = np.nan
            continue
        if USE_WILCOXON:
            try:
                from scipy.stats import wilcoxon
                _, pval = wilcoxon(pivot["On-nail"].values, pivot["Off-nail"].values)
                results[pair] = pval
                continue
            except Exception:
                pass
        results[pair] = _permutation_pval(pivot["On-nail"].values, pivot["Off-nail"].values)

    return results


# ── Plotting helpers ──────────────────────────────────────────────────────────
C1 = "#2166AC"
C_ON  = "#7FB3D3"
C_OFF = "#D3E9F5"
RED   = ACCENT_RED
C_SAME = "#5B9BD5"
C_DIFF = "#E07B39"



def pval_label(p):
    if np.isnan(p): return ""
    if p < 0.001:   return "***"
    if p < 0.01:    return "**"
    if p < 0.05:    return "*"
    return f"n.s. p={p:.3f}"


def draw_bracket(ax, x1, x2, y, label, tick_h=3.5):
    ax.plot([x1, x1, x2, x2], [y, y+tick_h, y+tick_h, y],
            color=RED, lw=1.2, clip_on=False)
    if label:
        ax.text((x1+x2)/2, y+tick_h+1.5, label,
                ha="center", va="bottom", fontsize=9, color=RED, fontweight="bold")


def draw_bracket_above_axes(ax, x1, x2, tier, label, tier_step=0.065):
    """Significance bracket in margin above plot — y-axis stays 0–100."""
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    y = 1.02 + tier * tier_step
    tick_h = 0.022
    ax.plot([x1, x1, x2, x2], [y, y + tick_h, y + tick_h, y],
            color=RED, lw=1.2, clip_on=False, transform=trans)
    if label:
        ax.text((x1 + x2) / 2, y + tick_h + 0.012, label, transform=trans,
                ha="center", va="bottom", fontsize=9, color=RED, fontweight="bold", clip_on=False)


def draw_bracket_horizontal_data(ax, x1, x2, y, label, *, linewidth=1.0):
    """Horizontal significance line in data coords (no vertical end ticks)."""
    ax.plot([x1, x2], [y, y], color=RED, lw=linewidth, clip_on=False, zorder=30)
    if label:
        ax.text((x1 + x2) / 2, y + 1.5, label,
                ha="center", va="bottom", fontsize=9, color=RED, fontweight="bold",
                clip_on=False, zorder=31)


def jitter_x(n, width=0.12, seed=42):
    return (np.random.default_rng(seed).random(n) - 0.5) * width


# ── Response bias overview (Same/Different analogue of response_bias_overview.py) ─
def _bias_proportions(df_sub):
    """Return pct SAME / pct DIFFERENT for delivered, all responses, incorrect-only."""
    n_total = len(df_sub)
    if n_total == 0:
        return None

    def _pct(mask):
        return mask.sum() / n_total * 100

    pct_del_same = _pct(df_sub["GroundTruth"] == "SAME")
    pct_del_diff = 100 - pct_del_same

    pct_resp_same = _pct(df_sub["UserChoice"] == "SAME")
    pct_resp_diff = 100 - pct_resp_same

    inc = df_sub[df_sub["correct"] == 0]
    n_incorrect = len(inc)
    if n_incorrect == 0:
        pct_inc_same = pct_inc_diff = np.nan
    else:
        pct_inc_same = (inc["UserChoice"] == "SAME").sum() / n_incorrect * 100
        pct_inc_diff = 100 - pct_inc_same

    return {
        "n_total": n_total,
        "n_incorrect": n_incorrect,
        "del_same": pct_del_same,
        "del_diff": pct_del_diff,
        "resp_same": pct_resp_same,
        "resp_diff": pct_resp_diff,
        "inc_same": pct_inc_same,
        "inc_diff": pct_inc_diff,
    }


def _draw_bias_bar(ax, pct_same, pct_diff, label_same, label_diff, title, n, subtitle=None):
    """Horizontal stacked bar — SAME (blue) vs DIFFERENT (orange)."""
    ax.barh(0, pct_same, height=0.55, color=C_SAME, label=label_same)
    ax.barh(0, pct_diff, height=0.55, left=pct_same, color=C_DIFF, label=label_diff)

    if pct_same > 8:
        ax.text(pct_same / 2, 0, f"{pct_same:.1f}%", ha="center", va="center",
                fontsize=13, fontweight="bold", color="white")
    if pct_diff > 8:
        ax.text(pct_same + pct_diff / 2, 0, f"{pct_diff:.1f}%", ha="center", va="center",
                fontsize=13, fontweight="bold", color="white")

    ax.axvline(50, color="gray", lw=1.2, ls="--", alpha=0.7)
    ax.text(50, 0.34, "50%", ha="center", fontsize=9, color="gray", va="bottom")

    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, 0.55)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=10)
    ax.set_yticks([])
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    if subtitle:
        ax.set_xlabel(subtitle, fontsize=10, labelpad=6)
    ax.legend(
        loc="lower center", bbox_to_anchor=(0.5, -0.38), ncol=2,
        handles=[
            mpatches.Patch(color=C_SAME, label=label_same),
            mpatches.Patch(color=C_DIFF, label=label_diff),
        ],
        **GEE_LEGEND_KW,
    )
    ax.text(0.98, 1.02, f"n = {n:,} trials", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=9, color="#555")


def run_response_bias_overview(df_sub, out_path, suptitle):
    """Three-panel response bias figure (pooled, band, or single-subject)."""
    stats = _bias_proportions(df_sub)
    if stats is None:
        return

    fig, axes = plt.subplots(3, 1, figsize=(9, 7.5))
    fig.suptitle(suptitle, fontsize=13, fontweight="bold", y=0.99)

    _draw_bias_bar(
        axes[0], stats["del_same"], stats["del_diff"],
        "Ground truth: SAME", "Ground truth: DIFFERENT",
        "Plot 1 – Delivered Stimulus Distribution",
        n=stats["n_total"],
        subtitle="Were the two stimuli physically the same or different?",
    )
    _draw_bias_bar(
        axes[1], stats["resp_same"], stats["resp_diff"],
        "Responded: SAME", "Responded: DIFFERENT",
        "Plot 2 – Response Distribution  (all trials)",
        n=stats["n_total"],
        subtitle="What did participants say?",
    )
    _draw_bias_bar(
        axes[2], stats["inc_same"], stats["inc_diff"],
        "Responded: SAME", "Responded: DIFFERENT",
        "Plot 3 – Incorrect-Trial Response Distribution",
        n=stats["n_incorrect"],
        subtitle="Among wrong answers, did they say SAME or DIFFERENT?",
    )

    plt.tight_layout(rect=[0, 0, 1, 0.97], h_pad=2.5)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _pair_response_distribution_rows(df_sub, pair_order):
    rows = []
    for pair in pair_order:
        for gt in ["SAME", "DIFFERENT"]:
            sub = df_sub[(df_sub["pair_label"] == pair) & (df_sub["GroundTruth"] == gt)]
            n = len(sub)
            if n == 0:
                continue
            n_same = int((sub["UserChoice"] == "SAME").sum())
            n_diff = n - n_same
            rows.append({
                "pair_label": pair,
                "ground_truth": gt,
                "n_trials": n,
                "n_resp_same": n_same,
                "n_resp_diff": n_diff,
                "pct_resp_same": n_same / n * 100,
                "pct_resp_diff": n_diff / n * 100,
            })
    return rows


def _label_stacked_bar(ax, x_left, width, y, text, min_width=8):
    if width < min_width:
        return
    ax.text(x_left + width / 2, y, text, ha="center", va="center",
            fontsize=8, fontweight="bold", color="white")


def save_response_distribution_by_pair(df_sub, pair_order, title, out_path, *, csv_path=None):
    """
    Per force pair: response distribution on SAME trials vs DIFFERENT trials.
    Each row = one pair × ground-truth type; stacked bar = responded SAME | DIFFERENT.
    """
    rows = _pair_response_distribution_rows(df_sub, pair_order)
    if not rows:
        return
    if csv_path:
        pd.DataFrame(rows).to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(11, 1.55 * len(pair_order) * 2 + 1.2))
    y = 0
    yticks, ylabels = [], []
    bar_h = 0.38
    group_gap = 0.55

    for pair in pair_order:
        pair_rows = [r for r in rows if r["pair_label"] == pair]
        for r in pair_rows:
            ps, pct_diff = r["pct_resp_same"], r["pct_resp_diff"]
            ax.barh(y, ps, height=bar_h, color=C_SAME, edgecolor="black", linewidth=0.6)
            ax.barh(y, pct_diff, height=bar_h, left=ps, color=C_DIFF, edgecolor="black", linewidth=0.6)
            _label_stacked_bar(ax, 0, ps, y, f"{ps:.1f}%\n(n={r['n_resp_same']})")
            _label_stacked_bar(ax, ps, pct_diff, y, f"{pct_diff:.1f}%\n(n={r['n_resp_diff']})")
            yticks.append(y)
            gt = r["ground_truth"]
            ylabels.append(f"{pair}  ·  {gt} trials  (n={r['n_trials']})")
            y -= 1
        y -= group_gap

    ax.axvline(50, color="gray", ls="--", lw=1.0, alpha=0.7)
    ax.text(50, y + group_gap + 0.5, "50%", ha="center", fontsize=8, color="gray")
    ax.set_xlim(0, 100)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=10)
    ax.set_xlabel("Response distribution (%)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.invert_yaxis()
    ax.spines[["top", "right", "left"]].set_visible(False)
    _gee_ax_legend(ax, [
        mpatches.Patch(facecolor=C_SAME, edgecolor="black", linewidth=BOX_LINEWIDTH, label="Responded: SAME"),
        mpatches.Patch(facecolor=C_DIFF, edgecolor="black", linewidth=BOX_LINEWIDTH, label="Responded: DIFFERENT"),
    ], loc="lower right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _same_trial_response_rows(df_sub, pair_order):
    """Response distribution within SAME trials only, per force pair."""
    rows = []
    for pair in pair_order:
        sub = df_sub[(df_sub["pair_label"] == pair) & (df_sub["GroundTruth"] == "SAME")]
        n = len(sub)
        if n == 0:
            continue
        n_same = int((sub["UserChoice"] == "SAME").sum())
        n_diff = n - n_same
        rows.append({
            "pair_label": pair,
            "n_trials": n,
            "n_resp_same": n_same,
            "n_resp_diff": n_diff,
            "pct_resp_same": n_same / n * 100,
            "pct_resp_diff": n_diff / n * 100,
        })
    return rows


def save_same_trial_response_by_pair(df_sub, pair_order, title, out_path, *, csv_path=None):
    """SAME trials only: how participants responded (SAME vs DIFFERENT), by force pair."""
    rows = _same_trial_response_rows(df_sub, pair_order)
    if not rows:
        return
    if csv_path:
        pd.DataFrame(rows).to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(11, 1.1 * len(pair_order) + 1.8))
    yticks, ylabels = [], []
    bar_h = 0.55

    for yi, r in enumerate(rows):
        ps, pct_diff = r["pct_resp_same"], r["pct_resp_diff"]
        ax.barh(yi, ps, height=bar_h, color=C_SAME, edgecolor="black", linewidth=0.6)
        ax.barh(yi, pct_diff, height=bar_h, left=ps, color=C_DIFF, edgecolor="black", linewidth=0.6)
        _label_stacked_bar(ax, 0, ps, yi, f"{ps:.1f}%\n(n={r['n_resp_same']})")
        _label_stacked_bar(ax, ps, pct_diff, yi, f"{pct_diff:.1f}%\n(n={r['n_resp_diff']})")
        yticks.append(yi)
        ylabels.append(f"{r['pair_label']}  (SAME trials, n={r['n_trials']})")

    ax.axvline(50, color="gray", ls="--", lw=1.0, alpha=0.7)
    ax.text(50, len(rows) - 0.15, "50%", ha="center", fontsize=8, color="gray")
    ax.set_xlim(0, 100)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=10)
    ax.set_xlabel("Response distribution (%)  — ground truth always SAME", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.invert_yaxis()
    ax.spines[["top", "right", "left"]].set_visible(False)
    _gee_ax_legend(ax, [
        mpatches.Patch(facecolor=C_SAME, edgecolor="black", linewidth=BOX_LINEWIDTH, label="Responded: SAME (correct)"),
        mpatches.Patch(facecolor=C_DIFF, edgecolor="black", linewidth=BOX_LINEWIDTH, label="Responded: DIFFERENT (incorrect)"),
    ], loc="lower right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _draw_pair_accuracy_boxplot(ax, pair_order, values_by_pair, *,
                                pairwise_pvals_dict=None, dot_label=None,
                                data_ylim_top=100, show_ylabel=True,
                                bracket_in_data_space=False, box_color="#dde6f0",
                                scatter_rgba=None,
                                box_width=0.42, jitter_width=0.12,
                                finalize_gee_axes=True, scatter_marker="o",
                                style_scale=1.0):
    """Boxplot + scatter; y-axis 0–100 (brackets may extend ylim when in data space)."""
    lw = BOX_LINEWIDTH * style_scale
    med_lw = MEDIAN_LINEWIDTH * style_scale
    band_max = 0.0
    for xi, pair in enumerate(pair_order):
        vals = np.asarray(values_by_pair.get(pair, []), dtype=float) * 100
        if len(vals) == 0:
            continue
        band_max = max(band_max, float(np.nanmax(vals)))
        bp = ax.boxplot(
            [vals], positions=[xi], widths=box_width,
            patch_artist=True, showfliers=False, capwidths=CAP_WIDTH,
            whiskerprops={"linewidth": lw, "color": BOX_STROKE},
            capprops={"linewidth": lw, "color": BOX_STROKE},
            medianprops={"color": ACCENT_RED, "linewidth": med_lw},
            boxprops={"linewidth": lw, "edgecolor": BOX_STROKE},
        )
        bp["boxes"][0].set_facecolor(box_color)
        bp["boxes"][0].set_edgecolor(BOX_STROKE)
        bp["medians"][0].set_color(ACCENT_RED)
        bp["medians"][0].set_linewidth(med_lw)
        jx = xi + jitter_x(len(vals), width=jitter_width)
        dot_rgba = scatter_rgba if scatter_rgba is not None else box_color
        dot_size = (20 if scatter_marker == "o" else round(20 * 1.3 ** 2)) * style_scale ** 2
        ax.scatter(jx, vals, c=[dot_rgba] * len(vals), s=dot_size, zorder=3,
                   linewidths=0, edgecolors="none", marker=scatter_marker)

    ax.axhline(
        JND_PCT, color=CRITERION_COLOR, linestyle="--",
        linewidth=1.0, alpha=0.85, zorder=REF_LINE_ZORDER,
    )

    max_bracket_tier = -1
    ylim_top = data_ylim_top
    if pairwise_pvals_dict:
        pair_combos = sorted(itertools.combinations(range(len(pair_order)), 2),
                             key=lambda t: t[1]-t[0])
        tier_used = []
        for i1, i2 in pair_combos:
            p1, p2 = pair_order[i1], pair_order[i2]
            pval = pairwise_pvals_dict.get((p1, p2), pairwise_pvals_dict.get((p2, p1), np.nan))
            if np.isnan(pval) or pval >= 0.05:
                continue
            level = 0
            while any(l == level and not (i2 < a or b < i1) for a, b, l in tier_used):
                level += 1
            tier_used.append((i1, i2, level))
            label = pval_label(pval)
            if bracket_in_data_space:
                y = max(102.0, band_max + BRACKET_BASE_PAD) + level * BRACKET_TIER_STEP
                draw_bracket_horizontal_data(ax, i1, i2, y, label)
            else:
                draw_bracket_above_axes(ax, i1, i2, level, label)
        if tier_used:
            max_bracket_tier = max(l for _, _, l in tier_used)
            if bracket_in_data_space:
                ylim_top = min(
                    BRACKET_YLIM_CAP,
                    max(102.0, band_max + BRACKET_BASE_PAD)
                    + max_bracket_tier * BRACKET_TIER_STEP + 6.0,
                )
    elif bracket_in_data_space:
        ylim_top = min(
            ATD.FIG2_BRACKET_YLIM_CAP,
            max(ATD.ACCURACY_YLIM_TOP, band_max + 8.0),
        )

    ax.set_xticks(range(len(pair_order)))
    tick_fs = (FONT_TICK if bracket_in_data_space else 11) * style_scale
    label_fs = (FONT_LABEL if bracket_in_data_space else 11) * style_scale
    ax.set_xticklabels(pair_order, fontsize=tick_fs)
    ax.set_xlim(-0.55, len(pair_order) - 0.45)
    if not bracket_in_data_space or not finalize_gee_axes:
        ax.set_ylim(0, ylim_top)
        y_tick_vals = list(range(0, data_ylim_top + 1, 20))
        ax.set_yticks(y_tick_vals)
        ax.tick_params(axis="both", labelsize=tick_fs, length=0)
    if show_ylabel and (not bracket_in_data_space or not finalize_gee_axes):
        ax.set_ylabel(
            "Discrimination Accuracy (%)" if bracket_in_data_space else "Accuracy (%)",
            fontsize=label_fs,
            labelpad=ATD.FIG_AXIS_LABELPAD if bracket_in_data_space else None,
        )
    if not bracket_in_data_space or not finalize_gee_axes:
        ax.set_xlabel(
            XLABEL_FORCE_PAIR,
            fontsize=label_fs,
            labelpad=ATD.FIG_AXIS_LABELPAD if bracket_in_data_space else None,
        )
    ax.spines[["top", "right"]].set_visible(False)
    if bracket_in_data_space and finalize_gee_axes:
        _finalize_gee_accuracy_axes(
            ax, len(pair_order), ylim_top, show_ylabel=show_ylabel,
            style_scale=style_scale,
        )
    if dot_label:
        ax.text(0.02, 0.98, dot_label, transform=ax.transAxes,
                ha="left", va="top", fontsize=9, color="#555")
    if bracket_in_data_space:
        return max_bracket_tier, ylim_top
    return max_bracket_tier


def _apply_accuracy_figure_layout(fig, max_bracket_tier, title, *, panel_titles=None):
    """Title at figure top; brackets in margin below title; axes y-axis 0–100."""
    title_pad = 0.055 if title else 0.02
    panel_pad = 0.045 if panel_titles else 0.0
    bracket_pad = 0.06 + max(0, max_bracket_tier) * 0.052
    axes_top = 1.0 - title_pad - panel_pad - bracket_pad
    fig.subplots_adjust(top=max(axes_top, 0.55), wspace=0.12)
    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold", y=1.0, va="top")
    if panel_titles:
        for ax, panel_title in zip(fig.axes, panel_titles):
            ax.set_title(panel_title, fontsize=11, fontweight="bold", pad=10)


def _draw_accuracy_panel(ax, subj_acc_df, pair_order, pairwise_pvals_dict, *, show_ylabel=True,
                         bracket_in_data_space=False, box_color="#dde6f0",
                         scatter_rgba=None,
                         box_width=0.42, jitter_width=0.12,
                         finalize_gee_axes=True, scatter_marker="o", style_scale=1.0):
    subj_acc_sorted = subj_acc_df[subj_acc_df["pair_label"].isin(pair_order)].copy()
    values_by_pair = {
        pair: subj_acc_sorted.loc[subj_acc_sorted["pair_label"] == pair, "accuracy"].values
        for pair in pair_order
    }
    return _draw_pair_accuracy_boxplot(
        ax, pair_order, values_by_pair,
        pairwise_pvals_dict=pairwise_pvals_dict,
        show_ylabel=show_ylabel,
        bracket_in_data_space=bracket_in_data_space,
        box_color=box_color,
        scatter_rgba=scatter_rgba,
        box_width=box_width,
        jitter_width=jitter_width,
        finalize_gee_axes=finalize_gee_axes,
        scatter_marker=scatter_marker,
        style_scale=style_scale,
    )


def draw_accuracy_band_on_ax(ax, spec, *, show_brackets=False, show_ylabel=True,
                             show_xlabel=True, title=None, scatter_marker="o", title_pad=4):
    """Single-band accuracy-by-pair panel (standalone 2-col figure)."""
    bl = spec["band_label"]
    pair_order = spec["pair_order"]
    box_w = combined_panel_box_width(len(pair_order))
    jitter_w = STRIP_JITTER_REF * box_w / POOLED_BOX_REF
    pvals = spec["pairwise_pvals"] if show_brackets else None
    _, ylim_top = _draw_accuracy_panel(
        ax, spec["subj_acc"], pair_order, pvals,
        show_ylabel=show_ylabel,
        bracket_in_data_space=True,
        box_color=band_box_color(bl),
        scatter_rgba=band_scatter_rgba(bl),
        box_width=box_w,
        jitter_width=jitter_w,
        finalize_gee_axes=False,
        scatter_marker=scatter_marker,
    )
    if title:
        ax.set_title(title, fontsize=FONT_LABEL, pad=title_pad)
    _finalize_gee_accuracy_axes(
        ax, len(pair_order), ylim_top,
        show_ylabel=show_ylabel, show_xlabel=show_xlabel,
    )
    return ylim_top


def _combined_accuracy_legend_handles(band_specs):
    return [
        mpatches.Patch(
            facecolor=band_box_color(spec["band_label"]),
            edgecolor=BOX_STROKE,
            linewidth=BOX_LINEWIDTH,
            label=band_title_text(spec["band_label"], spec["title_ref"], spec["n_subj"]),
        )
        for spec in band_specs
    ]


def save_combined_accuracy_by_pair(band_specs, *, show_brackets=True, scatter_marker="o"):
    """Low | High accuracy-by-pair in one 2-column figure (2102×1298 px)."""
    sns.set_theme(style="white")
    ATD.apply_plot_style()
    low_r, high_r = horizontal_panel_rects()
    fig = plt.figure(figsize=EXPORT_CANVAS, facecolor="#FFFFFF")
    rects = [low_r, high_r][:len(band_specs)]

    axes = []
    panel_n_x = []
    shared_ylim = ATD.ACCURACY_YLIM_TOP
    for rect, spec in zip(rects, band_specs):
        ax = fig.add_axes(rect)
        axes.append(ax)
        panel_n_x.append(len(spec["pair_order"]))
        ylim_top = draw_accuracy_band_on_ax(
            ax, spec, show_brackets=show_brackets,
            show_ylabel=(ax is axes[0]),
            show_xlabel=True,
            title=band_title_text(
                spec["band_label"], spec["title_ref"], spec["n_subj"],
            ),
            scatter_marker=scatter_marker,
        )
        shared_ylim = max(shared_ylim, ylim_top)

    for ax, n_x in zip(axes, panel_n_x):
        ax.set_ylim(ATD.ACCURACY_YMIN, min(ATD.FIG2_BRACKET_YLIM_CAP, shared_ylim))

    add_figure_legend(
        fig, _combined_accuracy_legend_handles(band_specs),
        ncol=len(band_specs), fontsize=FONT_LABEL,
    )
    suffix = "" if show_brackets else "_nobracket"
    if scatter_marker != "o":
        suffix += "_triangle" if scatter_marker == "^" else f"_{scatter_marker}"
    out_path = os.path.join(OUTPUT_DIR, f"sd_accuracy_by_pair_2col{suffix}.png")
    save_png_at_width(
        fig, out_path,
        width_px=EXPORT_WIDTH_2COL,
        height_px=EXPORT_HEIGHT_2COL,
        dpi=SAVE_DPI_COMBINED,
        pad_inches=0.05,
    )
    plt.close(fig)


def _save_subject_accuracy_by_pair(df_sub, subject, pair_order, band_title, out_path):
    """Per-subject sd_accuracy_by_pair-style figure (dots = regions A–F)."""
    region_acc = (
        df_sub.groupby(["pair_label", "Region"])["correct"]
        .mean()
        .reset_index()
    )
    values_by_pair = {
        pair: region_acc.loc[region_acc["pair_label"] == pair, "correct"].values
        for pair in pair_order
    }
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_title = f"{subject} — Overall Accuracy — Same/Different 2AFC ({band_title})"
    max_tier = _draw_pair_accuracy_boxplot(
        ax, pair_order, values_by_pair,
        dot_label="each dot = one region (A–F)",
    )
    _apply_accuracy_figure_layout(fig, max_tier, plot_title)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _draw_subject_bars(ax, pair_order, values_pct, title, bar_color=C1):
    """Single-subject bar chart (no group stats — one person only)."""
    xs = np.arange(len(pair_order))
    bars = ax.bar(xs, values_pct, width=0.55, color=bar_color, edgecolor="black", alpha=0.85)
    for bar, val in zip(bars, values_pct):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 2, f"{val:.0f}%",
                    ha="center", va="bottom", fontsize=9)
    ax.axhline(
        JND_PCT, color=CRITERION_COLOR, linestyle="--",
        linewidth=1.0, alpha=0.85, zorder=REF_LINE_ZORDER,
    )
    ax.set_xticks(xs)
    ax.set_xticklabels(pair_order, fontsize=11)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_xlabel(XLABEL_FORCE_PAIR, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)


def run_subject_analysis(df_band, band_label, pair_order, out_suffix, title_ref):
    """Per-subject accuracy tables and figures (one participant at a time)."""
    subj_root = os.path.join(OUTPUT_DIR, "per_subject")
    os.makedirs(subj_root, exist_ok=True)

    band_title = band_title_text(band_label, title_ref, 1)
    summary_rows = []

    for subject in sorted(df_band["Subject"].unique()):
        df_sub = df_band[df_band["Subject"] == subject].copy()
        subj_dir = os.path.join(subj_root, subject)
        os.makedirs(subj_dir, exist_ok=True)

        overall = (
            df_sub.groupby("pair_label")["correct"]
            .agg(n_trials="count", accuracy="mean")
            .reindex(pair_order)
        )
        same_acc = df_sub[df_sub["GroundTruth"] == "SAME"].groupby("pair_label")["correct"].mean()
        diff_acc = df_sub[df_sub["GroundTruth"] == "DIFFERENT"].groupby("pair_label")["correct"].mean()

        for pair in pair_order:
            row = overall.loc[pair] if pair in overall.index else None
            if row is None or pd.isna(row["n_trials"]):
                continue
            acc_pct = row["accuracy"] * 100
            same_pct = same_acc.get(pair, np.nan) * 100 if pair in same_acc.index else np.nan
            diff_pct = diff_acc.get(pair, np.nan) * 100 if pair in diff_acc.index else np.nan
            summary_rows.append({
                "Subject": subject,
                "band": band_label,
                "pair_label": pair,
                "n_trials": int(row["n_trials"]),
                "accuracy_pct": acc_pct,
                "same_accuracy_pct": same_pct,
                "different_accuracy_pct": diff_pct,
            })

        overall_vals = [
            overall.loc[p, "accuracy"] * 100 if p in overall.index and not pd.isna(overall.loc[p, "accuracy"]) else np.nan
            for p in pair_order
        ]
        same_vals = [same_acc.get(p, np.nan) * 100 if p in same_acc.index else np.nan for p in pair_order]
        diff_vals = [diff_acc.get(p, np.nan) * 100 if p in diff_acc.index else np.nan for p in pair_order]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        _draw_subject_bars(
            ax1, pair_order, overall_vals,
            f"{subject} — Overall ({band_title})",
        )

        xs = np.arange(len(pair_order))
        bw = 0.35
        ax2.bar(xs - bw / 2, same_vals, width=bw, color=C_SAME, edgecolor="black", alpha=0.85, label="SAME")
        ax2.bar(xs + bw / 2, diff_vals, width=bw, color=C_DIFF, edgecolor="black", alpha=0.85, label="DIFFERENT")
        ax2.axhline(
            JND_PCT, color=CRITERION_COLOR, linestyle="--",
            linewidth=1.0, alpha=0.85, zorder=REF_LINE_ZORDER,
        )
        ax2.set_xticks(xs)
        ax2.set_xticklabels(pair_order, fontsize=11)
        ax2.set_ylim(0, 110)
        ax2.set_ylabel("Accuracy (%)", fontsize=11)
        ax2.set_xlabel(XLABEL_FORCE_PAIR, fontsize=11)
        ax2.set_title(f"{subject} — SAME vs DIFFERENT ({band_title})", fontsize=12, fontweight="bold")
        _gee_ax_legend(ax2, [
            mpatches.Patch(facecolor=C_SAME, edgecolor=BOX_STROKE, linewidth=BOX_LINEWIDTH, label="SAME"),
            mpatches.Patch(facecolor=C_DIFF, edgecolor=BOX_STROKE, linewidth=BOX_LINEWIDTH, label="DIFFERENT"),
        ])
        ax2.spines[["top", "right"]].set_visible(False)

        plt.tight_layout()
        out_name = f"sd_overview{out_suffix}.png"
        fig.savefig(os.path.join(subj_dir, out_name), dpi=150, bbox_inches="tight")
        plt.close(fig)

        pair_out = os.path.join(subj_dir, f"sd_accuracy_by_pair{out_suffix}.png")
        _save_subject_accuracy_by_pair(
            df_sub, subject, pair_order, band_title, pair_out,
        )

        bias_out = os.path.join(subj_dir, f"sd_response_bias{out_suffix}.png")
        run_response_bias_overview(
            df_sub, bias_out,
            f"Response Bias — {subject}  ({band_title})",
        )

        dist_out = os.path.join(subj_dir, f"sd_response_by_pair{out_suffix}.png")
        save_response_distribution_by_pair(
            df_sub, pair_order,
            f"{subject} — Response by Pair × Ground Truth ({band_title})",
            dist_out,
        )
        save_same_trial_response_by_pair(
            df_sub, pair_order,
            f"{subject} — Responses on SAME Trials Only ({band_title})",
            os.path.join(subj_dir, f"sd_same_trial_response{out_suffix}.png"),
        )

    if summary_rows:
        summary_path = os.path.join(subj_root, f"accuracy_by_subject{out_suffix}.csv")
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)


def build_band_spec(df_band, band_label, pair_order, title_ref):
    """Compute per-band data needed for accuracy / on-nail figures (no file output)."""
    subj_acc = (
        df_band.groupby(["Subject", "pair_label"])["correct"]
        .mean().reset_index()
        .rename(columns={"correct": "accuracy"})
    )
    pairwise_pvals = run_gee_pairwise(df_band, subj_acc, pair_order)
    region_pvals = run_gee_region(df_band, pair_order)
    df_reg = df_band[df_band["region_group"].notna()]
    subj_acc_reg = (
        df_reg.groupby(["Subject", "pair_label", "region_group"])["correct"]
        .mean().reset_index().rename(columns={"correct": "accuracy"})
    )
    n_subj = df_band["Subject"].nunique()
    return {
        "band_label": band_label,
        "title_ref": title_ref,
        "pair_order": pair_order,
        "subj_acc": subj_acc,
        "pairwise_pvals": pairwise_pvals,
        "n_subj": n_subj,
        "subj_acc_reg": subj_acc_reg,
        "region_pvals": region_pvals,
    }


def run_band_analysis(df_band, band_label, pair_order, out_suffix, title_ref):
    """Run SDT, GEE, and figures for one force band (Low ref=1g or High ref=26g)."""
    spec = build_band_spec(df_band, band_label, pair_order, title_ref)
    subj_acc = spec["subj_acc"]
    pairwise_pvals = spec["pairwise_pvals"]
    region_pvals = spec["region_pvals"]
    subj_acc_reg = spec["subj_acc_reg"]
    n_subj = spec["n_subj"]

    sdt_rows = []
    for (pair, grp), sub in df_band[df_band["region_group"].notna()].groupby(["pair_label", "region_group"]):
        sdt = compute_sdt(sub)
        sdt_rows.append({"pair_label": pair, "region_group": grp, **sdt})
    df_sdt = pd.DataFrame(sdt_rows)
    df_sdt.to_csv(os.path.join(OUTPUT_DIR, f"sdt_summary{out_suffix}.csv"), index=False)

    order_rows = []
    for pair in pair_order:
        for grp in ["On-nail", "Off-nail"]:
            sub = df_band[(df_band["pair_label"] == pair) & (df_band["region_group"] == grp)]
            acc_rc = sub.loc[sub["TrialType"] == "diff_rc", "correct"].mean()
            acc_cr = sub.loc[sub["TrialType"] == "diff_cr", "correct"].mean()
            order_rows.append({
                "pair_label": pair, "region_group": grp,
                "acc_diff_rc": acc_rc * 100 if not np.isnan(acc_rc) else np.nan,
                "acc_diff_cr": acc_cr * 100 if not np.isnan(acc_cr) else np.nan,
                "delta_rc_minus_cr": (acc_rc - acc_cr) * 100 if not (np.isnan(acc_rc) or np.isnan(acc_cr)) else np.nan,
            })
    df_order = pd.DataFrame(order_rows)
    df_order.to_csv(os.path.join(OUTPUT_DIR, f"order_effect{out_suffix}.csv"), index=False)

    subj_acc_split = (
        df_band.groupby(["Subject", "pair_label", "GroundTruth"])["correct"]
        .mean().reset_index()
        .rename(columns={"correct": "accuracy"})
    )

    band_title = band_title_text(band_label, title_ref, n_subj)

    OFFSET = 0.22
    BW_SD  = 0.20
    fig, ax = plt.subplots(figsize=(10, 6))
    band_max = 0.0
    for xi, pair in enumerate(pair_order):
        for gi, (gt, color) in enumerate([("SAME", C_SAME), ("DIFFERENT", C_DIFF)]):
            xp = xi + OFFSET * (gi - 0.5)
            vals = subj_acc_split.loc[
                (subj_acc_split["pair_label"] == pair) &
                (subj_acc_split["GroundTruth"] == gt),
                "accuracy"
            ].values * 100
            if len(vals) == 0:
                continue
            band_max = max(band_max, vals.max())
            bp = ax.boxplot([vals], positions=[xp], widths=BW_SD,
                             patch_artist=True, showfliers=False, capwidths=CAP_WIDTH,
                             whiskerprops={"linewidth": BOX_LINEWIDTH, "color": BOX_STROKE},
                             capprops={"linewidth": BOX_LINEWIDTH, "color": BOX_STROKE},
                             medianprops={"color": ACCENT_RED, "linewidth": MEDIAN_LINEWIDTH},
                             boxprops={"linewidth": BOX_LINEWIDTH, "edgecolor": BOX_STROKE})
            bp["boxes"][0].set_facecolor(color); bp["boxes"][0].set_alpha(0.35)
            bp["boxes"][0].set_edgecolor(BOX_STROKE)
            bp["medians"][0].set_color(ACCENT_RED); bp["medians"][0].set_linewidth(MEDIAN_LINEWIDTH)
            jx = xp + jitter_x(len(vals), width=BW_SD * 0.5)
            ax.scatter(jx, vals, color=color, alpha=0.75, s=22, zorder=3)

    ax.axhline(
        JND_PCT, color=CRITERION_COLOR, linestyle="--",
        linewidth=1.0, alpha=0.85, zorder=REF_LINE_ZORDER,
    )
    ax.set_xticks(range(len(pair_order)))
    ax.set_xticklabels(pair_order, fontsize=11)
    ax.set_ylim(0, min(115, band_max + 20))
    ax.set_ylabel("Discrimination Accuracy (%)", fontsize=11)
    ax.set_xlabel(XLABEL_FORCE_PAIR, fontsize=11)
    ax.set_title(f"SAME vs DIFFERENT Trial Accuracy — {band_title}", fontsize=12, fontweight="bold")
    _gee_ax_legend(ax, [
        mpatches.Patch(facecolor=C_SAME, edgecolor=BOX_STROKE, linewidth=BOX_LINEWIDTH, alpha=0.55, label="SAME trials"),
        mpatches.Patch(facecolor=C_DIFF, edgecolor=BOX_STROKE, linewidth=BOX_LINEWIDTH, alpha=0.55, label="DIFFERENT trials"),
    ], loc="lower right")
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    out_a2 = f"sd_accuracy_split{out_suffix}.png"
    fig.savefig(os.path.join(OUTPUT_DIR, out_a2), dpi=150, bbox_inches="tight")
    plt.close(fig)

    out_b = f"sd_onnail_vs_offnail{out_suffix}.png"
    xtick_fs = FONT_XTICK_LOW if out_suffix == "_low" else FONT_XTICK
    save_region_onnail_figure(
        subj_acc_reg, pair_order, region_pvals,
        os.path.join(OUTPUT_DIR, out_b),
        show_title=False,
        width_px=ONNAIL_EXPORT_WIDTH_PX,
        height_px=ONNAIL_EXPORT_HEIGHT_PX,
        compensate_width=False,
        xtick_fs=xtick_fs,
    )
    # Legacy *_4204px filename — now also exported at 1385×H
    out_b_4col = f"sd_onnail_vs_offnail{out_suffix}_4204px.png"
    save_region_onnail_figure(
        subj_acc_reg, pair_order, region_pvals,
        os.path.join(OUTPUT_DIR, out_b_4col),
        show_title=False,
        width_px=ONNAIL_EXPORT_WIDTH_PX,
        height_px=ONNAIL_EXPORT_HEIGHT_PX,
        compensate_width=False,
        xtick_fs=xtick_fs,
    )

    df_sdt_plot = df_sdt[df_sdt["pair_label"].isin(pair_order)].copy()
    fig, ax = plt.subplots(figsize=(9, 5))
    BW = 0.20
    xs = np.arange(len(pair_order))
    for gi, (grp, color) in enumerate([("On-nail", C_ON), ("Off-nail", C_OFF)]):
        vals = [df_sdt_plot.loc[(df_sdt_plot["pair_label"]==p) & (df_sdt_plot["region_group"]==grp), "d_prime"].values
                for p in pair_order]
        ys = [v[0] if len(v) else np.nan for v in vals]
        ax.bar(xs + OFFSET*(gi-0.5), ys, width=BW*1.1,
               color=color, edgecolor="black", label=grp)
        for x, y in zip(xs + OFFSET*(gi-0.5), ys):
            if not np.isnan(y):
                ax.text(x, y + 0.05, f"{y:.2f}", ha="center", fontsize=8)

    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(xs); ax.set_xticklabels(pair_order, fontsize=11)
    ax.set_ylabel("d′  (sensitivity)", fontsize=11)
    ax.set_xlabel(XLABEL_FORCE_PAIR, fontsize=11)
    ax.set_title(f"Signal Detection Theory — d′ ({band_title})", fontsize=12, fontweight="bold")
    _gee_ax_legend(ax, [
        mpatches.Patch(facecolor=C_ON, edgecolor=BOX_STROKE, linewidth=BOX_LINEWIDTH, label="On-nail"),
        mpatches.Patch(facecolor=C_OFF, edgecolor=BOX_STROKE, linewidth=BOX_LINEWIDTH, label="Off-nail"),
    ])
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    out_c = f"sd_dprime_by_pair_region{out_suffix}.png"
    fig.savefig(os.path.join(OUTPUT_DIR, out_c), dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    df_order_plot = df_order[df_order["pair_label"].isin(pair_order)]
    for gi, (grp, color) in enumerate([("On-nail", C_ON), ("Off-nail", C_OFF)]):
        sub = df_order_plot[df_order_plot["region_group"]==grp]
        xs_pairs = [pair_order.index(p) for p in sub["pair_label"] if p in pair_order]
        rc_vals = sub["acc_diff_rc"].values
        cr_vals = sub["acc_diff_cr"].values
        x_pos = np.array(xs_pairs, dtype=float) + OFFSET*(gi-0.5)
        ax.plot(x_pos, rc_vals, "o-", color=color, lw=2, label=f"{grp} diff_rc")
        ax.plot(x_pos, cr_vals, "s--", color=color, lw=1.5, alpha=0.7, label=f"{grp} diff_cr")

    ax.axhline(
        JND_PCT, color=CRITERION_COLOR, linestyle="--",
        linewidth=1.0, alpha=0.85, zorder=REF_LINE_ZORDER,
    )
    ax.set_xticks(range(len(pair_order))); ax.set_xticklabels(pair_order, fontsize=11)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Accuracy (%)  — DIFFERENT trials only", fontsize=11)
    ax.set_xlabel(XLABEL_FORCE_PAIR, fontsize=11)
    ax.set_title(f"Order Effect: diff_rc vs diff_cr ({band_title})", fontsize=12, fontweight="bold")
    handles, labels = ax.get_legend_handles_labels()
    _gee_ax_legend(ax, handles, ncol=2)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    out_d = f"sd_order_effect{out_suffix}.png"
    fig.savefig(os.path.join(OUTPUT_DIR, out_d), dpi=150, bbox_inches="tight")
    plt.close(fig)

    run_response_bias_overview(
        df_band,
        os.path.join(OUTPUT_DIR, f"sd_response_bias_overview{out_suffix}.png"),
        f"Response Bias Overview — Same/Different 2AFC ({band_title})",
    )
    save_response_distribution_by_pair(
        df_band, pair_order,
        f"Response Distribution by Force Pair — SAME vs DIFFERENT trials ({band_title})",
        os.path.join(OUTPUT_DIR, f"sd_response_by_pair{out_suffix}.png"),
        csv_path=os.path.join(OUTPUT_DIR, f"response_distribution_by_pair{out_suffix}.csv"),
    )
    save_same_trial_response_by_pair(
        df_band, pair_order,
        f"Responses on SAME Trials Only — by Force Pair ({band_title})",
        os.path.join(OUTPUT_DIR, f"sd_same_trial_response{out_suffix}.png"),
        csv_path=os.path.join(OUTPUT_DIR, f"same_trial_response_by_pair{out_suffix}.csv"),
    )
    run_subject_analysis(df_band, band_label, pair_order, out_suffix, title_ref)

    return spec


if __name__ == "__main__":
    accuracy_band_specs = []
    for band_label, cfg in BAND_CONFIG.items():
        df_band = df[df["band"] == band_label].copy()
        if df_band.empty:
            continue
        pair_order = fix_order(cfg["pair_order"], df_band["pair_label"].unique().tolist())
        spec = run_band_analysis(df_band, band_label, pair_order, cfg["suffix"], cfg["title_ref"])
        if spec:
            accuracy_band_specs.append(spec)

    if accuracy_band_specs:
        save_combined_accuracy_by_pair(accuracy_band_specs)
        save_combined_accuracy_by_pair(accuracy_band_specs, show_brackets=False)
        save_combined_accuracy_by_pair(
            accuracy_band_specs, show_brackets=False, scatter_marker="^",
        )

    run_response_bias_overview(
        df,
        os.path.join(OUTPUT_DIR, "sd_response_bias_overview_all.png"),
        f"Response Bias Overview — Same/Different 2AFC (All Bands, n = {df['Subject'].nunique()})",
    )

    print(f"\nAll outputs saved to:\n  {OUTPUT_DIR}")