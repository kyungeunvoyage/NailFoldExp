"""
ATD Comparison Figures
======================
Figure 1: Kao fingerpad No-paint vs. Periungual In-air (this study)
Figure 2: Periungual On-touch vs. In-air (this study)
Figure 3: Kao fingerpad No-paint vs. Periungual On-touch (Mid) (this study)

Kao et al. 2022 No-paint condition values are digitized from published Fig. A (n=5).
"""

import os
import io
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.path import Path
import colorsys
import seaborn as sns
from matplotlib import rcParams

try:
    import statsmodels.formula.api as smf
    HAS_STATSMODELS = True
except ImportError:
    smf = None
    HAS_STATSMODELS = False
from matplotlib.transforms import blended_transform_factory

# =============================================================================
# Palette
# =============================================================================
IN_AIR = "#6A4A3C"       # Periungual In-air (brown)
IN_AIR_LEGACY = "#56708A"  # previous slate blue In-air
SLATE_BLUE = IN_AIR_LEGACY  # backward-compatible alias
ON_TOUCH   = "#10559A"   # Periungual On-touch (teal)
ON_TOUCH_LEGACY = "#295E11"  # previous green On-touch
ACCENT_RED = "#BF2C23"   # median line
REF_LINE_ZORDER = 20     # accuracy criterion dashed line — top layer
BLACK      = "#1A1A1A"
CRITERION_COLOR = BLACK  # 80% reference line & label
KAO_COLOR  = "#5A5A5A"   # Anika Paint — dark gray (matches original paper)

STRIP_ALPHA   = 0.50
SCATTER_HSB_BRIGHTNESS = 0.60  # HSB brightness (HSV V) for scatter points
# Shared pale box fill for periungual conditions (In-air & On-touch)
COND_BOX_BRIGHTNESS = 0.88
COND_BOX_SATURATION_SCALE = 0.40
COND_BOX_ALPHA_HEX = "40"  # ~25% opacity
# backward-compatible aliases
ON_TOUCH_BOX_BRIGHTNESS = COND_BOX_BRIGHTNESS
ON_TOUCH_BOX_SATURATION_SCALE = COND_BOX_SATURATION_SCALE
ON_TOUCH_BOX_ALPHA_HEX = COND_BOX_ALPHA_HEX

FIG_SIZE  = (8.0, 4.5)   # design aspect ratio (w×h inches)
SAVE_DPI  = 600            # master raster before column-width resize
SAVE_SVG  = False          # set True to also export .svg

# Publication column widths (px); height scales with figure aspect ratio
EXPORT_WIDTHS_PX = (
    ("1col",   1028),
    ("1p5col", 1346),
    ("2col",   2102),
)

# Font sizes — change here (applied after sns.set_theme so they are not reset)
FONT_TICK   = 16   # axis tick numbers (0.02, 0.04, …)
FONT_LABEL  = 14
FONT_LEGEND = 12
FONT_ANNOT  = 10
REGION_ONLY_Y = 98        # y for "Fingerpad only" / "Periungual only" labels
REGION_ONLY_X_FRAC = 0.28  # along shaded span (0=left, 0.5=center)

Y_ZERO_GAP_BELOW = 5         # ylim padding: 0% tick floats above x-axis (no stroke in gap)
ACCURACY_YMIN = -Y_ZERO_GAP_BELOW
ACCURACY_YTICKS = (0, 20, 40, 60, 80, 100)
ACCURACY_YSPINE = (ACCURACY_YMIN, 100)   # left spine extends to ylim bottom so it meets x-axis
ACCURACY_YLIM_TOP = 103      # data ylim above 100 so 100% scatter is not clipped

# Inward tick guides drawn manually (seaborn categorical axes hide mpl ticks)
TICK_LEN_AXES = 0.016  # fraction of axis length into the plot
FIG_LEGEND_TOP = 0.92          # subplots_adjust top — raise (e.g. 0.96) = plot closer to legend
FIG_LEGEND_BOTTOM = 0.12
FIG_LEGEND_ABOVE_AXES = 0.01   # gap in axes coords (smaller → tighter; 0 = flush to plot top)
FIG_LEGEND_PAD_PT = 1.0        # borderaxespad (points) between legend box and axes
FIG_LEGEND_TOP_MULTILINE = 0.90  # slightly lower axes top when legend has \\n lines
FIG_AXIS_LABELPAD = 6

BOX_LINEWIDTH = 1.4
KAO_BOX_LINEWIDTH = 1.4   # thicker stroke for hollow Kao boxes
CAP_LINEWIDTH = 0.5
CAP_WIDTH     = 0.10   # matplotlib capwidths — short end ticks (default ~0.5× box)
REGION_BOX_WIDTH = 0.55  # sns ``width`` on Fig2 (5 forces, 2 hues)
FIG2_DODGED_BOX_WIDTH = REGION_BOX_WIDTH / 2  # one In-air / On-touch box (≈0.275 data units)
FIG3_GROUP_HALF_WIDTH = 0.55  # x tolerance when shifting a whole force group to its tick
FIG3_STRIP_JITTER_FRAC = 0.38   # |dx| ≤ this × half box width — strip stays centered in box

# Fig2: In-air vs On-touch LME brackets (pattern from ATD_Stats.py)
FIG2_REF_CONDITION = "In-air"
FIG2_TEST_CONDITION = "On-touch (Mid)"
FIG2_BRACKET_EXCLUDE_FORCES = {0.07}  # no bracket at 0.07 g
FIG2_BRACKET_MAX_P = 0.001            # *** only (0.16 g, 0.6 g)
FIG2_BRACKET_BASE_PAD = 3.0
FIG2_BRACKET_TICK_H = 4.0
FIG2_BRACKET_TEXT_PAD = 2.0
FIG2_BRACKET_YLIM_CAP = 122   # headroom above 100% for brackets; y ticks still 0–100
FIG2_BRACKET_ZORDER = 30     # above 80% criterion line (REF_LINE_ZORDER)


def fig3_hue_dodge_half(n_forces):
    """Half-gap between dodged Kao / Peri boxes at a shared force (e.g. 0.07 g)."""
    return sns_boxplot_width(n_forces) / 3.0


def assign_fig3_plot_x(df_plot, force_to_idx, source_order, n_forces):
    """
    x = tick index (0, 1, 2, …) for each stimulus force.

    Exception: forces with both Kao + Periungual (0.07 g) dodge left/right of the tick.
    """
    half = fig3_hue_dodge_half(n_forces)
    src_counts = df_plot.groupby("Force_Val")["Source"].nunique()
    cats = df_plot["Force_Val"].map(force_to_idx).astype(float)
    dual = df_plot["Force_Val"].map(src_counts) >= 2
    is_first = df_plot["Source"] == source_order[0]
    x = cats.copy()
    x = np.where(dual & is_first, cats - half, x)
    x = np.where(dual & ~is_first, cats + half, x)
    df_plot = df_plot.copy()
    df_plot["Plot_X"] = x
    return df_plot


def _snap_boxplot_line_to_x(line, target_x):
    """Move one whisker, cap, or median line so its x center is *target_x*."""
    xdata = np.asarray(line.get_xdata(), dtype=float)
    ydata = np.asarray(line.get_ydata(), dtype=float)
    if len(xdata) != 2 or len(ydata) != 2:
        return
    if np.isclose(ydata[0], ydata[1]):
        dx = target_x - (xdata[0] + xdata[1]) / 2
    elif np.isclose(xdata[0], xdata[1]):
        dx = target_x - xdata[0]
    else:
        return
    _shift_box_artist_x(line, dx)


def _scores_are_flat(scores, atol=1e-9):
    """True when Q1 = Q3 (same rule as Fig2: no visible box, median line only)."""
    if len(scores) == 0:
        return True
    q1, _, q3 = np.percentile(scores, [25, 50, 75])
    return bool(np.isclose(q1, q3, atol=atol))


def _snap_whisker_caps_to_spine(bp, index, y_floor=0.0, y_ceil=100.0, floor_atol=0.5,
                                ceil_atol=0.5):
    """Snap whisker/cap ends to 0% / 100% so strokes meet the y-axis spine cleanly."""
    for j in (2 * index, 2 * index + 1):
        w = bp["whiskers"][j]
        if not w.get_visible():
            continue
        y = np.asarray(w.get_ydata(), dtype=float)
        if np.min(y) <= floor_atol:
            y[y == np.min(y)] = y_floor
        if np.max(y) >= ceil_atol:
            y[y == np.max(y)] = y_ceil
        w.set_ydata(y)
        cap = bp["caps"][j]
        if not cap.get_visible():
            continue
        cy = np.asarray(cap.get_ydata(), dtype=float)
        if np.isclose(cy[0], cy[1]) and cy[0] <= floor_atol:
            cap.set_ydata([y_floor, y_floor])
        elif np.isclose(cy[0], cy[1]) and cy[0] >= ceil_atol:
            cap.set_ydata([y_ceil, y_ceil])


def _style_flat_boxplot_group(bp, index):
    """Zero-IQR: hide box; black whiskers, caps, and median (not snapped to axes)."""
    bp["boxes"][index].set_visible(False)
    for j in (2 * index, 2 * index + 1):
        for key, lw in (("whiskers", BOX_LINEWIDTH), ("caps", CAP_LINEWIDTH)):
            line = bp[key][j]
            line.set_visible(True)
            line.set_color(BLACK)
            line.set_linewidth(lw)
            line.set_zorder(4)
            line.set_solid_capstyle("butt")
    med = bp["medians"][index]
    med.set_visible(True)
    med.set_color(BLACK)
    med.set_linewidth(2.0)
    med.set_zorder(15)
    med.set_solid_capstyle("butt")


def finish_boxplot_styling(bp, flat_flags, sources, source_order, hollow_source=None):
    """Black whiskers everywhere; flat groups keep whiskers (no axis snap)."""
    if bp is None:
        return
    for i, src in enumerate(sources):
        lw = (KAO_BOX_LINEWIDTH if hollow_source is not None and src == hollow_source
              else BOX_LINEWIDTH)
        for j in (2 * i, 2 * i + 1):
            for key, cap_lw in (("whiskers", lw), ("caps", CAP_LINEWIDTH)):
                line = bp[key][j]
                if not line.get_visible():
                    continue
                line.set_color(BLACK)
                line.set_linewidth(cap_lw)
                line.set_solid_capstyle("butt")
        if flat_flags[i]:
            _style_flat_boxplot_group(bp, i)


def _snap_boxplot_group_to_x(bp, index, target_x):
    """Align box patch + median + whiskers + caps on the same x (tick or dodge)."""
    patch = bp["boxes"][index]
    _shift_box_artist_x(patch, target_x - _patch_x_center(patch))
    _snap_boxplot_line_to_x(bp["medians"][index], target_x)
    for j in (2 * index, 2 * index + 1):
        _snap_boxplot_line_to_x(bp["whiskers"][j], target_x)
        _snap_boxplot_line_to_x(bp["caps"][j], target_x)


def draw_fig3_boxplot(ax, df_plot, combined_forces, source_order, box_palette, box_w,
                      hollow_source=None):
    """Boxplot at tick x; returns *bp*, positions, flat flags, and source per group."""
    groups, positions, faces, sources, flat_flags, force_vals = [], [], [], [], [], []
    for fval in combined_forces:
        for src in source_order:
            sub = df_plot[(df_plot["Force_Val"] == fval) & (df_plot["Source"] == src)]
            if sub.empty:
                continue
            scores = sub["Score"].values
            groups.append(scores)
            flat_flags.append(_scores_are_flat(scores))
            positions.append(float(sub["Plot_X"].iloc[0]))
            faces.append(box_palette[src])
            sources.append(src)
            force_vals.append(fval)
    if not groups:
        return None, [], [], [], []
    bp = ax.boxplot(
        groups,
        positions=positions,
        widths=box_w,
        patch_artist=True,
        showfliers=False,
        capwidths=CAP_WIDTH,
        whiskerprops={"linewidth": BOX_LINEWIDTH, "color": BLACK},
        capprops={"linewidth": CAP_LINEWIDTH, "color": BLACK},
        medianprops={"color": ACCENT_RED, "linewidth": 2.0},
        boxprops={"linewidth": BOX_LINEWIDTH, "edgecolor": BLACK},
    )
    for i, (patch, fc, src) in enumerate(zip(bp["boxes"], faces, sources)):
        if hollow_source is not None and src == hollow_source:
            patch.set_facecolor("none")
            patch.set_edgecolor(KAO_COLOR)
            patch.set_linewidth(KAO_BOX_LINEWIDTH)
        else:
            patch.set_facecolor(fc)
            patch.set_edgecolor(BLACK)
            patch.set_linewidth(BOX_LINEWIDTH)
        _snap_boxplot_group_to_x(bp, i, positions[i])
    return bp, positions, flat_flags, sources, force_vals


def resnap_fig3_boxplot_groups(bp, positions):
    """Re-align whiskers/medians/boxes after flat-box expansion."""
    if bp is None:
        return
    for i, target in enumerate(positions):
        _snap_boxplot_group_to_x(bp, i, target)


def _centered_strip_offsets(n, half_span, rng):
    """Even x offsets in [-half_span, half_span], shuffled (centered strip, seed-stable)."""
    if n <= 0:
        return np.empty(0, dtype=float)
    if n == 1:
        return np.array([0.0])
    off = np.linspace(-half_span, half_span, n)
    rng.shuffle(off)
    return off


def draw_fig3_stripplot(
    ax,
    df_plot,
    source_order,
    source_colors,
    box_width,
    brightness=SCATTER_HSB_BRIGHTNESS,
    alpha=STRIP_ALPHA,
    size=3.8,
    jitter_frac=FIG3_STRIP_JITTER_FRAC,
    seed=0,
    participant_median=False,
    triangle_keys=None,
):
    """Strip points at ``Plot_X``; narrow centered jitter inside each box.

    triangle_keys: set of (source_name, force_val) tuples → use '^' marker.
    """
    rng = np.random.default_rng(seed)
    half_span = (box_width / 2) * jitter_frac
    for src in source_order:
        sub = df_plot[df_plot["Source"] == src]
        if sub.empty:
            continue
        if participant_median and "Participant" in sub.columns:
            grp_cols = ["Participant", "Plot_X"]
            if "Force_Val" in sub.columns:
                grp_cols = ["Participant", "Plot_X", "Force_Val"]
            sub = sub.groupby(grp_cols, as_index=False)["Score"].median()
        rgba = _hsb_scatter_rgba(source_colors[src], brightness)
        xs_circ, ys_circ = [], []
        xs_tri,  ys_tri  = [], []
        for _, grp in sub.groupby("Plot_X", sort=False):
            x0   = float(grp["Plot_X"].iloc[0])
            fval = float(grp["Force_Val"].iloc[0]) if "Force_Val" in grp.columns else None
            y    = grp["Score"].to_numpy(dtype=float)
            offsets = x0 + _centered_strip_offsets(len(y), half_span, rng)
            if triangle_keys is not None and (src, fval) in triangle_keys:
                xs_tri.append(offsets);  ys_tri.append(y)
            else:
                xs_circ.append(offsets); ys_circ.append(y)
        scatter_kw = dict(linewidths=0, edgecolors="none", alpha=alpha,
                          zorder=3, clip_on=False)
        if xs_circ:
            xc = np.concatenate(xs_circ); yc = np.concatenate(ys_circ)
            ax.scatter(xc, yc, c=[rgba]*len(xc), s=size**2, marker="o", **scatter_kw)
        if xs_tri:
            xt = np.concatenate(xs_tri);  yt = np.concatenate(ys_tri)
            ax.scatter(xt, yt, c=[rgba]*len(xt), s=(size*1.3)**2, marker="^", **scatter_kw)


def sns_boxplot_width(n_x_categories, reference_n=None):
    """
    sns ``width`` so each dodged box has the same pixel width as Fig2
    (scale up when there are more x categories on the same fig size).
    """
    ref_n = reference_n if reference_n is not None else len(USER_FORCES)
    if n_x_categories <= 1 or ref_n <= 1:
        return REGION_BOX_WIDTH
    return REGION_BOX_WIDTH * (n_x_categories - 1) / (ref_n - 1)


def mpl_boxplot_width(n_x_categories, reference_n=None):
    """matplotlib ``widths`` for one box per category (matches Fig2 dodge box pixels)."""
    return sns_boxplot_width(n_x_categories, reference_n) / 2

# =============================================================================
# Paths
# =============================================================================
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT    = os.path.normpath(os.path.join(SCRIPT_DIR, "../../"))
FILE_PATTERN = os.path.join(
    REPO_ROOT, "Data", "(ATD)CurData", "P*_AbsoluteThresholdDetection.csv"
)
OUT_DIR = os.path.join(SCRIPT_DIR, "atd_c1_outputs")
os.makedirs(OUT_DIR, exist_ok=True)


def add_legend_outside(
    fig,
    ax,
    handles,
    ncol=2,
    title=None,
    *,
    legend_fontsize=None,
    left=0.11,
    right=0.98,
    top=FIG_LEGEND_TOP,
    bottom=FIG_LEGEND_BOTTOM,
    wspace=None,
    above_axes=FIG_LEGEND_ABOVE_AXES,
):
    """Legend centered just above the plot (axes coords — use constants to tighten gap)."""
    fs = legend_fontsize if legend_fontsize is not None else FONT_LABEL
    labels = [h.get_label() for h in handles]
    multiline = any("\n" in (lab or "") for lab in labels)
    axes_top = FIG_LEGEND_TOP_MULTILINE if multiline else top
    labelspacing = 0.55 if multiline else 0.2
    handleheight = 1.0
    adjust_kw = dict(left=left, right=right, top=axes_top, bottom=bottom)
    if wspace is not None:
        adjust_kw["wspace"] = wspace
    fig.subplots_adjust(**adjust_kw)
    ax.legend(
        handles=handles,
        title=title,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0 + above_axes),
        bbox_transform=ax.transAxes,
        ncol=ncol,
        fontsize=fs,
        title_fontsize=fs,
        frameon=False,
        labelspacing=labelspacing,
        columnspacing=2.0,
        handlelength=1.6,
        handleheight=handleheight,
        borderaxespad=FIG_LEGEND_PAD_PT,
    )


def save_figure(fig, stem, export_widths=None):
    """Save PNG at publication column widths (px)."""
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
        png_path = os.path.join(OUT_DIR, f"{stem}_{tag}.png")
        out.save(png_path)
        print(f"Saved PNG → {png_path}  ({width_px}×{height_px} px)")

    if SAVE_SVG:
        svg_path = os.path.join(OUT_DIR, f"{stem}.svg")
        fig.savefig(
            svg_path,
            format="svg",
            bbox_inches="tight",
            facecolor="white",
            metadata={"Creator": "ATD_C1_Fig(Anika).py"},
        )
        print(f"Saved SVG → {svg_path}  (vector)")


def apply_plot_style():
    """Apply fonts after sns.set_theme (seaborn resets rcParams otherwise)."""
    rcParams.update({
        "figure.facecolor":      "#FFFFFF",
        "axes.facecolor":        "#FFFFFF",
        "font.family":           "sans-serif",
        "font.sans-serif":       ["Helvetica", "Arial", "DejaVu Sans"],
        "axes.linewidth":        1,
        "axes.spines.top":       False,
        "axes.spines.right":     False,
        "xtick.major.width":     1,
        "ytick.major.width":     1,
        "xtick.major.size":      0,
        "ytick.major.size":      0,
        "legend.frameon":        False,
        "legend.fontsize":       FONT_LEGEND,
        "legend.title_fontsize": FONT_LEGEND,
        "font.size":             FONT_LABEL,
        "axes.titlesize":        FONT_LABEL,
        "axes.labelsize":        FONT_LABEL,
        "xtick.labelsize":       FONT_TICK,
        "ytick.labelsize":       FONT_TICK,
        "figure.dpi":            600,
        "savefig.dpi":           SAVE_DPI,
        "svg.fonttype":          "path",
    })


def set_detection_accuracy_ylim(ax, ylim_top=None):
    """Ticks 0–100%; ylim padded below 0 and above 100 (0% tick above x-axis)."""
    top = ACCURACY_YLIM_TOP
    if ylim_top is not None:
        top = min(FIG2_BRACKET_YLIM_CAP, max(ACCURACY_YLIM_TOP, ylim_top))
    ax.set_ylim(ACCURACY_YMIN, top)
    ax.set_yticks(ACCURACY_YTICKS)
    ax.margins(y=0)


def apply_accuracy_y_spine_bounds(ax):
    """Left spine ends at 0% tick so nothing draws between 0 and the x-axis."""
    y0, y1 = ACCURACY_YSPINE
    ax.spines["left"].set_bounds(y0, y1)


def finalize_accuracy_axes(fig, ax, n_x, xticks, xticklabels, leg_handles, ylim_top=None):
    """Shared axis format for Fig1–3 and 10559A export."""
    ax.set_xlabel("Stimulus Force (g)", fontsize=FONT_LABEL, labelpad=FIG_AXIS_LABELPAD)
    ax.set_ylabel("Detection Accuracy (%)", fontsize=FONT_LABEL, labelpad=FIG_AXIS_LABELPAD)
    ax.set_xlim(-0.5, n_x - 0.5)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, fontsize=FONT_TICK)
    add_legend_outside(
        fig, ax, leg_handles, ncol=2,
        top=FIG_LEGEND_TOP, bottom=FIG_LEGEND_BOTTOM,
    )
    set_detection_accuracy_ylim(ax, ylim_top=ylim_top)
    sns.despine(ax=ax)
    apply_accuracy_y_spine_bounds(ax)
    add_inward_tick_guides(ax, n_x)
    apply_accuracy_y_spine_bounds(ax)


def _star_from_p(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def lme_condition_at_force(df_in, force_val, sub_col, ref_cond=FIG2_REF_CONDITION,
                           test_cond=FIG2_TEST_CONDITION):
    """Trial-level LME: Score ~ Condition at one force (random intercept ~ subject)."""
    if not HAS_STATSMODELS:
        raise ImportError(
            "Fig2 LME brackets require statsmodels. Run:\n"
            "  .venv/bin/python \"(New)Analysis/ATDAnalysis/ATD_C1_Fig(Anika).py\""
        )
    sub = df_in[df_in["Force_Val"] == force_val].dropna(
        subset=[sub_col, "Score", "Condition"]
    )
    if sub.empty or sub["Condition"].nunique() < 2 or sub[sub_col].nunique() < 2:
        return None
    formula = f"Score ~ C(Condition, Treatment(reference='{ref_cond}'))"
    try:
        res = smf.mixedlm(formula, sub, groups=sub[sub_col]).fit()
        col = f"C(Condition, Treatment(reference='{ref_cond}'))[T.{test_cond}]"
        if col not in res.pvalues.index:
            return None
        return {"p": float(res.pvalues[col]), "coef": float(res.params[col])}
    except Exception:
        return None


def _condition_group_top_y(df_plot, force_val, cond, bp, bp_index):
    """Whisker/scatter top for bracket base (avoid flat-box underestimates)."""
    sub = df_plot[
        (df_plot["Force_Val"] == force_val) & (df_plot["Condition"] == cond)
    ]["Score"].dropna()
    data_top = float(sub.max()) if len(sub) else 0.0
    whisker_top = _boxplot_group_whisker_top(bp, bp_index)
    if whisker_top is None:
        return data_top
    return max(data_top, whisker_top)


def _boxplot_group_whisker_top(bp, index):
    if bp is None:
        return None
    return float(max(bp["whiskers"][2 * index + 1].get_ydata()))


def _add_condition_sig_bracket(ax, x_l, x_r, y_base, text="", tick_h=FIG2_BRACKET_TICK_H):
    """Bracket between dodged boxes (ATD_Stats._add_sig_bracket style)."""
    x_center = (x_l + x_r) / 2.0
    y_top = y_base + tick_h
    ax.plot(
        [x_l, x_l, x_r, x_r],
        [y_base, y_top, y_top, y_base],
        color=ACCENT_RED,
        linewidth=2.0,
        clip_on=False,
        zorder=FIG2_BRACKET_ZORDER,
    )
    ax.text(
        x_center,
        y_top - 1.5,
        text,
        ha="center",
        va="bottom",
        fontsize=FONT_ANNOT,
        color=ACCENT_RED,
        fontweight="bold",
        clip_on=False,
        zorder=FIG2_BRACKET_ZORDER + 1,
    )
    return y_top + FIG2_BRACKET_TEXT_PAD + 3.0


def annotate_fig2_condition_brackets(
    ax,
    bp,
    positions,
    sources,
    force_vals,
    cond_list,
    df_plot,
    sub_col,
    *,
    ref_cond=FIG2_REF_CONDITION,
    test_cond=FIG2_TEST_CONDITION,
    max_p=FIG2_BRACKET_MAX_P,
):
    """
    Per-force In-air vs On-touch LME brackets (significant pairs only).
    Returns suggested ylim top for bracket headroom.
    """
    if bp is None or len(positions) == 0:
        return ACCURACY_YLIM_TOP

    x_by_force_cond = {}
    idx_by_force_cond = {}
    for i, (x, src, fval) in enumerate(zip(positions, sources, force_vals)):
        x_by_force_cond.setdefault(fval, {})[src] = x
        idx_by_force_cond.setdefault(fval, {})[src] = i

    y_ceiling = ACCURACY_YLIM_TOP
    for fval in sorted(x_by_force_cond.keys()):
        if float(fval) in FIG2_BRACKET_EXCLUDE_FORCES:
            continue
        xs = x_by_force_cond[fval]
        if ref_cond not in xs or test_cond not in xs:
            continue
        stat = lme_condition_at_force(
            df_plot, fval, sub_col, ref_cond=ref_cond, test_cond=test_cond,
        )
        if stat is None or stat["p"] >= max_p:
            continue
        tops = [
            _condition_group_top_y(
                df_plot, fval, c, bp, idx_by_force_cond[fval][c],
            )
            for c in (ref_cond, test_cond)
            if c in idx_by_force_cond[fval]
        ]
        if not tops:
            continue
        y_base = max(tops) + FIG2_BRACKET_BASE_PAD
        sig_text = _star_from_p(stat['p'])
        text_top = _add_condition_sig_bracket(
            ax, xs[ref_cond], xs[test_cond], y_base, text=sig_text,
        )
        y_ceiling = max(y_ceiling, text_top)

    return y_ceiling


def add_inward_tick_guides(ax, n_x):
    """Short inward guides at each x/y label (matplotlib ticks fail on sns categorical axes)."""
    ax.set_xticks(range(n_x))
    ax.grid(False)
    ax.tick_params(axis="both", which="both", length=0, labelsize=FONT_TICK)

    x_trans = blended_transform_factory(ax.transData, ax.transAxes)
    y_trans = blended_transform_factory(ax.transAxes, ax.transData)
    y_lo, y_hi = ax.get_ylim()
    y_vals = [t for t in ax.get_yticks() if y_lo - 1e-9 <= t <= y_hi + 1e-9]

    for xi in range(n_x):
        ax.plot(
            [xi, xi], [0, TICK_LEN_AXES],
            color=BLACK, linewidth=1.0, solid_capstyle="butt",
            transform=x_trans, clip_on=False, zorder=6,
        )
    for y in y_vals:
        ax.plot(
            [0, TICK_LEN_AXES], [y, y],
            color=BLACK, linewidth=1.0, solid_capstyle="butt",
            transform=y_trans, clip_on=False, zorder=6,
        )


def _patch_x_center(patch):
    """Geometric center (box paths have duplicate verts; mean ≠ visual center)."""
    xs = patch.get_path().vertices[:, 0]
    return (float(xs.min()) + float(xs.max())) / 2


def _patch_yspan(patch):
    verts = patch.get_path().vertices
    return float(verts[:, 1].max() - verts[:, 1].min())


def _shift_box_artist_x(artist, dx):
    """Shift a box patch or whisker/median line horizontally by *dx*."""
    if isinstance(artist, mpatches.Patch):
        verts = artist.get_path().vertices.copy()
        verts[:, 0] += dx
        artist.set_path(Path(verts))
        return
    xdata = np.asarray(artist.get_xdata(), dtype=float)
    artist.set_xdata(xdata + dx)


def _hide_boxplot_artists_at_x(ax, xc, match_atol=0.30):
    """Hide whiskers/caps/medians tied to a removed phantom hue slot."""
    for line in ax.lines:
        xdata = np.asarray(line.get_xdata(), dtype=float)
        ydata = np.asarray(line.get_ydata(), dtype=float)
        if len(xdata) != 2 or len(ydata) != 2:
            continue
        xmid = (xdata[0] + xdata[1]) / 2
        if np.isclose(ydata[0], ydata[1]):
            if abs(xmid - xc) < match_atol:
                line.set_visible(False)
        elif np.isclose(xdata[0], xdata[1]) and abs(xdata[0] - xc) < match_atol:
            line.set_visible(False)


def _shift_boxplot_lines_at_x(ax, xc, dx, match_atol=0.30):
    """Shift whiskers, caps, and median lines only (leave box patches unchanged)."""
    for line in ax.lines:
        if not line.get_visible():
            continue
        xdata = np.asarray(line.get_xdata(), dtype=float)
        ydata = np.asarray(line.get_ydata(), dtype=float)
        if len(xdata) != 2 or len(ydata) != 2:
            continue
        xmid = (xdata[0] + xdata[1]) / 2
        if np.isclose(ydata[0], ydata[1]):
            if abs(xmid - xc) < match_atol:
                _shift_box_artist_x(line, dx)
        elif np.isclose(xdata[0], xdata[1]) and abs(xdata[0] - xc) < match_atol:
            _shift_box_artist_x(line, dx)


def _shift_boxplot_group_at_x(ax, xc, dx, match_atol=0.30):
    """Shift a dodged box patch and all of its whiskers, caps, and median lines."""
    for patch in ax.patches:
        if patch.get_visible() and abs(_patch_x_center(patch) - xc) < match_atol:
            _shift_box_artist_x(patch, dx)
    for line in ax.lines:
        if not line.get_visible():
            continue
        xdata = np.asarray(line.get_xdata(), dtype=float)
        ydata = np.asarray(line.get_ydata(), dtype=float)
        if len(xdata) != 2 or len(ydata) != 2:
            continue
        xmid = (xdata[0] + xdata[1]) / 2
        if np.isclose(ydata[0], ydata[1]):
            if abs(xmid - xc) < match_atol:
                _shift_box_artist_x(line, dx)
        elif np.isclose(xdata[0], xdata[1]) and abs(xdata[0] - xc) < match_atol:
            _shift_box_artist_x(line, dx)


def _expand_flat_box_patch(patch, min_yspan=5.0, ylim=(0.0, 100.0)):
    """
    Give zero-IQR boxes a visible height without breaking the box Path.

    Matplotlib box paths are a closed rectangle (6 verts). Splitting verts
    by index (old approach) draws a bow-tie / triangle; rebuild the rect instead.
    """
    verts = patch.get_path().vertices.copy()
    ylo, yhi = float(verts[:, 1].min()), float(verts[:, 1].max())
    if yhi - ylo >= min_yspan:
        return
    x0, x1 = float(verts[:, 0].min()), float(verts[:, 0].max())
    ymid = (ylo + yhi) / 2
    half = min_yspan / 2
    ylo_b = max(ylim[0], ymid - half)
    yhi_b = min(ylim[1], ymid + half)
    if yhi_b - ylo_b < min_yspan:
        if ymid <= (ylim[0] + ylim[1]) / 2:
            yhi_b = min(ylim[1], ylo_b + min_yspan)
        else:
            ylo_b = max(ylim[0], yhi_b - min_yspan)
    new_verts = np.array([
        [x0, ylo_b],
        [x1, ylo_b],
        [x1, yhi_b],
        [x0, yhi_b],
        [x0, ylo_b],
        [x0, ylo_b],
    ])
    patch.set_path(Path(new_verts))


def _singleton_hue_shifts(ax, n_categories):
    """Return {category: (x_center, dx)} for force levels with only one hue."""
    by_cat = {}
    for patch in ax.patches:
        if not patch.get_visible():
            continue
        yspan = float(patch.get_path().vertices[:, 1].max()
                      - patch.get_path().vertices[:, 1].min())
        if yspan < 1e-6:
            continue
        cat = int(round(_patch_x_center(patch)))
        if 0 <= cat < n_categories:
            by_cat.setdefault(cat, []).append(patch)

    shifts = {}
    for cat, patches in by_cat.items():
        if len(patches) != 1:
            continue
        xc = _patch_x_center(patches[0])
        dx = cat - xc
        if abs(dx) >= 0.01:
            shifts[cat] = (xc, dx)
    return shifts


def _snap_vertical_whiskers_at_category(ax, cat, half_width=FIG3_GROUP_HALF_WIDTH):
    """Place vertical whiskers exactly on the tick x (after group shift)."""
    for line in ax.lines:
        if not line.get_visible():
            continue
        xdata = np.asarray(line.get_xdata(), dtype=float)
        ydata = np.asarray(line.get_ydata(), dtype=float)
        if len(xdata) != 2 or len(ydata) != 2:
            continue
        if np.isclose(ydata[0], ydata[1]):
            continue
        if np.isclose(xdata[0], xdata[1]) and abs(xdata[0] - cat) <= half_width:
            line.set_xdata([cat, cat])


def _snap_median_to_category(ax, cat, half_width=FIG3_GROUP_HALF_WIDTH):
    """Center singleton medians on the tick."""
    for line in ax.lines:
        if not line.get_visible():
            continue
        xdata = np.asarray(line.get_xdata(), dtype=float)
        ydata = np.asarray(line.get_ydata(), dtype=float)
        if len(xdata) != 2 or len(ydata) != 2 or not np.isclose(ydata[0], ydata[1]):
            continue
        span = abs(xdata[1] - xdata[0])
        if span < 0.20:
            continue
        xmid = (xdata[0] + xdata[1]) / 2
        if abs(xmid - cat) <= half_width:
            half = span / 2
            line.set_xdata([cat - half, cat + half])


def align_boxplot_groups_to_ticks(
    ax,
    n_categories,
    combined_forces,
    df_plot,
    source_order,
    group_half_width=FIG3_GROUP_HALF_WIDTH,
):
    """
    Align box + whisker group centers with x ticks (0, 1, 2, …).
    Hides phantom hue slots; snaps vertical whiskers onto the tick line.
    """
    shifts = {}
    for cat, fval in enumerate(combined_forces):
        if cat >= n_categories:
            break
        sources = df_plot.loc[df_plot["Force_Val"] == fval, "Source"].unique()
        patches = [
            p for p in ax.patches
            if p.get_visible()
            and abs(_patch_x_center(p) - cat) <= group_half_width
            and _patch_yspan(p) > 0.01
        ]
        if len(sources) == 1 and len(patches) > 1:
            src = sources[0]
            keep = (
                min(patches, key=_patch_x_center)
                if src == source_order[0]
                else max(patches, key=_patch_x_center)
            )
            for patch in patches:
                if patch is keep:
                    continue
                phantom_x = _patch_x_center(patch)
                patch.set_visible(False)
                _hide_boxplot_artists_at_x(ax, phantom_x)
            patches = [keep]
        if not patches:
            continue
        xc = (
            _patch_x_center(patches[0])
            if len(patches) == 1
            else float(np.mean([_patch_x_center(p) for p in patches]))
        )
        dx = cat - xc
        if abs(dx) >= 0.005:
            shifts[cat] = (xc, dx)
            _shift_boxplot_group_at_x(ax, xc, dx, match_atol=group_half_width)
        _snap_vertical_whiskers_at_category(ax, cat, half_width=group_half_width)
        if len(patches) == 1:
            _snap_median_to_category(ax, cat, half_width=group_half_width)
    return shifts


def shift_strip_collections(ax, shifts, match_atol=0.25):
    """Apply box-group shifts to stripplot points (drawn before alignment)."""
    if not shifts:
        return
    for coll in ax.collections:
        offs = coll.get_offsets()
        if len(offs) == 0:
            continue
        xs = offs[:, 0].copy()
        for xc, dx in shifts.values():
            xs[np.abs(xs - xc) < match_atol] += dx
        coll.set_offsets(np.column_stack([xs, offs[:, 1]]))


def finalize_boxplot_lines(ax, median_color, median_lw=2.0, cap_lw=CAP_LINEWIDTH):
    """Style caps thin/short; medians above scatter (span-based; robust to empty boxes)."""
    for line in ax.lines:
        x, y = line.get_xdata(), line.get_ydata()
        if len(x) != 2 or len(y) != 2 or not np.isclose(y[0], y[1]):
            continue
        span = abs(x[1] - x[0])
        if span >= 0.20:
            line.set_color(median_color)
            line.set_linewidth(median_lw)
            line.set_zorder(15)
        elif span >= 0.01:
            line.set_linewidth(cap_lw)
            line.set_zorder(4)
        else:
            line.set_visible(False)


def style_kao_hollow_boxes(ax, combined_forces, kao_forces, edge_color=KAO_COLOR,
                           linewidth=KAO_BOX_LINEWIDTH):
    """Hollow gray stroke for Kao boxes (left hue at each Kao force level)."""
    kao_set = {float(f) for f in kao_forces}
    by_cat = {}
    for patch in ax.patches:
        cat = int(round(_patch_x_center(patch)))
        if 0 <= cat < len(combined_forces):
            by_cat.setdefault(cat, []).append(patch)

    for cat, patches in by_cat.items():
        if float(combined_forces[cat]) not in kao_set:
            continue
        kao_patch = min(patches, key=_patch_x_center)
        if _patch_yspan(kao_patch) < 1e-6:
            _expand_flat_box_patch(kao_patch)
        kao_patch.set_facecolor("none")
        kao_patch.set_edgecolor(edge_color)
        kao_patch.set_linewidth(linewidth)


def style_filled_box_edges(ax, edgecolor=BLACK, linewidth=BOX_LINEWIDTH):
    """Black box outline on filled patches (matches aggregate onnail boxplots)."""
    for patch in ax.patches:
        if not patch.get_visible():
            continue
        if getattr(patch, "_is_force_highlight", False):
            continue
        fc = patch.get_facecolor()
        if len(fc) >= 4 and fc[3] < 0.05:
            continue
        patch.set_edgecolor(edgecolor)
        patch.set_linewidth(linewidth)


def lower_stripplot_zorder(ax, zorder=3):
    for col in ax.collections:
        col.set_zorder(zorder)


def hsb_hex(base_hex, brightness=SCATTER_HSB_BRIGHTNESS, saturation_scale=1.0):
    """Hex color with hue preserved; optional saturation scale and fixed HSB brightness."""
    r, g, b = mcolors.to_rgb(base_hex)
    h, s, _v = colorsys.rgb_to_hsv(r, g, b)
    s = min(1.0, max(0.0, s * saturation_scale))
    rr, gg, bb = colorsys.hsv_to_rgb(h, s, brightness)
    return mcolors.to_hex((rr, gg, bb))


def pale_box_face(color):
    """Pale condition box fill (shared In-air / On-touch styling)."""
    return (hsb_hex(color, COND_BOX_BRIGHTNESS, COND_BOX_SATURATION_SCALE)
            + COND_BOX_ALPHA_HEX)


def pale_vis_color(color):
    """Region labels / axvspan tint for periungual conditions."""
    return hsb_hex(color, COND_BOX_BRIGHTNESS,
                   min(1.0, COND_BOX_SATURATION_SCALE * 1.4))


on_touch_box_face = pale_box_face  # backward-compatible alias
on_touch_vis_color = pale_vis_color


def box_fill_palette(source_colors):
    """Box fill colors with shared pale periungual styling."""
    return {k: pale_box_face(v) for k, v in source_colors.items()}


def _hsb_scatter_rgba(base_hex, brightness=SCATTER_HSB_BRIGHTNESS, alpha=STRIP_ALPHA):
    """Scatter color: keep hue & saturation, fix HSB brightness (HSV V)."""
    r, g, b = mcolors.to_rgb(base_hex)
    h, s, _v = colorsys.rgb_to_hsv(r, g, b)
    rr, gg, bb = colorsys.hsv_to_rgb(h, s, brightness)
    return (rr, gg, bb, alpha)


def _stripplot_collection_hue_plot_x(coll, df_plot, source_col="Source"):
    """Map strip collection → source via nearest ``Plot_X`` (Fig3)."""
    offs = coll.get_offsets()
    if len(offs) == 0:
        return None
    xmid = float(np.median(offs[:, 0]))
    lookup = df_plot.groupby([source_col, "Force_Val"])["Plot_X"].first()
    best_src, best_d = None, np.inf
    for (src, _fval), px in lookup.items():
        d = abs(xmid - float(px))
        if d < best_d:
            best_d, best_src = d, src
    return best_src if best_d < 0.25 else None


def _stripplot_collection_hue(
    coll, hue_order, x_levels, df_plot, x_col, source_col,
    *, force_to_idx=None, hue_half=None, use_plot_x=False,
):
    """Map one stripplot PathCollection to its hue (sns uses one coll per force×hue)."""
    offs = coll.get_offsets()
    if len(offs) == 0:
        return None
    if use_plot_x and "Plot_X" in df_plot.columns:
        return _stripplot_collection_hue_plot_x(coll, df_plot, source_col)
    xmid = float(np.median(offs[:, 0]))
    if force_to_idx is not None and hue_half is not None:
        tol = hue_half + 0.12
        for fval in x_levels:
            cat = force_to_idx[fval]
            sources = df_plot.loc[df_plot[x_col] == fval, source_col].unique()
            if len(sources) == 1:
                if abs(xmid - cat) <= tol:
                    return sources[0]
            elif len(sources) >= 2:
                if abs(xmid - (cat - hue_half)) <= tol:
                    return hue_order[0]
                if abs(xmid - (cat + hue_half)) <= tol:
                    return hue_order[1]
        return None
    cat = int(round(xmid))
    cat = max(0, min(cat, len(x_levels) - 1))
    fval = x_levels[cat]
    sources = df_plot.loc[df_plot[x_col] == fval, source_col].unique()
    if len(sources) == 1:
        return sources[0]
    if len(sources) >= 2:
        return hue_order[0] if xmid < cat else hue_order[1]
    return None


def apply_scatter_hsb_brightness(
    ax,
    hue_order,
    palette,
    brightness=SCATTER_HSB_BRIGHTNESS,
    *,
    x_levels,
    df_plot,
    x_col="Force_Val",
    source_col="Source",
    force_to_idx=None,
    hue_half=None,
    use_plot_x=False,
):
    """Set stripplot colors per force×hue collection (not only the last N collections)."""
    for coll in ax.collections:
        hue = _stripplot_collection_hue(
            coll, hue_order, x_levels, df_plot, x_col, source_col,
            force_to_idx=force_to_idx, hue_half=hue_half, use_plot_x=use_plot_x,
        )
        if hue is None or hue not in palette:
            continue
        rgba = _hsb_scatter_rgba(palette[hue], brightness)
        n = len(coll.get_offsets())
        coll.set_facecolors([rgba] * n)
        coll.set_edgecolors("none")
        coll.set_linewidths(0)

# =============================================================================
# Kao et al. 2022 — No-paint condition (panel A, light grey boxes)
# Digitized from published Fig. A No-paint trace (n=5, index fingerpad).
# Values chosen so matplotlib boxplots match panel box/median/whisker layout.
# y = Percent correct (%)
# =============================================================================
KAO_PAINT_RAW = {
    # force(g) : [P1, P2, P3, P4, P5]
    0.02: [  0,   8,  20,  58,  58],
    0.04: [ 45,  75,  85,  95, 100],
    0.07: [ 35,  78,  80,  83,  95],
    0.40: [ 90,  91,  93, 100, 100],
    1.00: [100, 100, 100, 100, 100],
    1.40: [ 82, 100, 100, 100, 100],
}
KAO_N = 5

# Build tidy DataFrame
kao_rows = []
for force, vals in KAO_PAINT_RAW.items():
    for pid, v in enumerate(vals):
        kao_rows.append({
            "Force_Val":  float(force),
            "Score":      float(v),
            "Source":     "Kao_NoPaint",
            "Participant": f"KP{pid+1}",
        })
df_kao = pd.DataFrame(kao_rows)

# =============================================================================
# Load user data
# =============================================================================
all_files = glob.glob(FILE_PATTERN)
if not all_files:
    raise FileNotFoundError(f"No CSVs found:\n  {FILE_PATTERN}")
print(f"Loaded {len(all_files)} participant file(s).")

df_raw = pd.concat(
    [pd.read_csv(f) for f in sorted(all_files)],
    ignore_index=True,
)
df_raw["Condition"] = df_raw["Condition"].str.strip().replace({
    "Active":          "On-touch (Mid)",
    "On-touch (Hard)": "On-touch (Mid)",
    "Passive":         "In-air",
})
df_raw = df_raw[df_raw["Condition"] != "On-touch (Soft)"]
df_raw = df_raw[df_raw["Area"].isin(["A", "B", "C", "D", "E", "F"])].copy()
df_raw["Force_Val"] = df_raw["Force"].str.extract(r"(\d+\.?\d*)").astype(float)

# P61, P62, P63: only include 0.4 g data (partial-protocol participants)
_PARTIAL_SUBJ = {"P61", "P62", "P63"}
_is_partial = df_raw["SubjectID" if "SubjectID" in df_raw.columns else "Subject"].isin(_PARTIAL_SUBJ)
df_raw = df_raw[~_is_partial | (df_raw["Force_Val"] == 0.4)].copy()
print(f"After partial-subject filter: {len(df_raw)} rows")

SUBJECT_COL = "SubjectID" if "SubjectID" in df_raw.columns else "Subject"
n_subjects = df_raw[SUBJECT_COL].nunique() if SUBJECT_COL in df_raw.columns else len(all_files)


def calc_score(row):
    if row["Target"] == 0:
        return 100.0 if row["Response"] == 0 else 0.0
    return max(0.0, (1 - abs(row["Target"] - row["Response"]) / row["Target"]) * 100.0)


df_raw["Score"] = df_raw.apply(calc_score, axis=1)

USER_FORCES   = sorted(df_raw["Force_Val"].unique())   # [0.07, 0.16, 0.6, 1.0, 1.4]
KAO_FORCE_PLOT_MAX = 0.4   # omit Kao forces >= 0.4 g from Fig1/3

# =============================================================================
# Kao vs Periungual comparison (shared plotting)
# =============================================================================
KAO_LABEL = f"Fingerpad \n(Kao et al. 2022, n={KAO_N})"

df_kao_plot = df_kao[df_kao["Force_Val"] < KAO_FORCE_PLOT_MAX].copy()
df_kao_plot["Source"] = KAO_LABEL
KAO_FORCES = sorted(df_kao_plot["Force_Val"].unique())  # [0.02, 0.04, 0.07]

COMBINED_FORCES = sorted(set(KAO_FORCES) | set(USER_FORCES))


def xspan(forces_sub, all_forces, pad=0.48):
    idxs = [list(all_forces).index(f) for f in forces_sub if f in all_forces]
    return (min(idxs) - pad, max(idxs) + pad) if idxs else None


def force_highlight_xspan(force_val, combined_forces, n_forces, box_w, edge_pad=0.06):
    """x span covering all dodged boxes at one force tick (e.g. 0.07 g Kao + Peri)."""
    forces = list(combined_forces)
    if force_val not in forces:
        return None
    cat = forces.index(force_val)
    half = fig3_hue_dodge_half(n_forces)
    xlo = cat - half - box_w / 2 - edge_pad
    xhi = cat + half + box_w / 2 + edge_pad
    return (xlo, xhi)


def draw_force_highlight_background(ax, highlight_forces, combined_forces, n_forces, box_w,
                                    color="#B0B0B0", alpha=0.10):
    """Light vertical band behind the plot — fill only, no border."""
    if not highlight_forces:
        return
    fill = mcolors.to_rgba(color, alpha)
    band_trans = blended_transform_factory(ax.transData, ax.transAxes)
    with matplotlib.rc_context({"patch.force_edgecolor": False, "patch.linewidth": 0.0}):
        for f in highlight_forces:
            span = force_highlight_xspan(f, combined_forces, n_forces, box_w)
            if not span:
                continue
            xlo, xhi = span
            band = mpatches.Polygon(
                [(xlo, 0.0), (xhi, 0.0), (xhi, 1.0), (xlo, 1.0)],
                closed=True,
                transform=band_trans,
                facecolor=fill,
                edgecolor="none",
                linewidth=0.0,
                antialiased=False,
                snap=False,
                zorder=0,
            )
            band._is_force_highlight = True
            ax.add_patch(band)


def plot_kao_vs_periungual(df_periungual, peri_label, peri_color, save_stem,
                           export_widths=None,
                           scatter_brightness=SCATTER_HSB_BRIGHTNESS,
                           peri_box_brightness=None,
                           peri_box_alpha_hex=None,
                           peri_box_saturation_scale=None,
                           region_background=True,
                           region_labels=True,
                           highlight_forces=None,
                           highlight_force_color="#B0B0B0",
                           highlight_force_alpha=0.10,
                           participant_median=False,
                           kao_df_override=None,
                           triangle_keys=None,
                           extra_leg_handles=None):
    """Box/strip plot: Kao fingerpad No-paint vs one periungual condition.

    kao_df_override: replace module-level df_kao_plot with a custom Kao DataFrame.
    triangle_keys:   set of (source_name, force_val) → triangle scatter marker.
    extra_leg_handles: additional legend patches appended after the default two.
    """
    df_peri = df_periungual.copy()
    df_peri["Source"] = peri_label
    if SUBJECT_COL in df_peri.columns:
        df_peri["Participant"] = df_peri[SUBJECT_COL]

    df_kao_local = kao_df_override if kao_df_override is not None else df_kao_plot
    kao_plot_cols = df_kao_local.columns.tolist()
    peri_plot_cols = ["Force_Val", "Score", "Source", "Participant"]
    peri_available = [c for c in peri_plot_cols if c in df_peri.columns]
    df_plot = pd.concat(
        [df_kao_local[kao_plot_cols],
         df_peri[peri_available]],
        ignore_index=True,
    )

    # Derive forces from actual data (supports custom kao_df_override)
    kao_forces_local  = sorted(df_kao_local["Force_Val"].unique())
    peri_forces_local = sorted(df_peri["Force_Val"].unique())
    combined_forces   = sorted(set(kao_forces_local) | set(peri_forces_local))

    n_forces = len(combined_forces)
    force_to_idx = {f: i for i, f in enumerate(combined_forces)}
    df_plot["Force_idx"] = df_plot["Force_Val"].map(force_to_idx)
    force_idx_order = list(range(n_forces))
    source_order = [KAO_LABEL, peri_label]
    df_plot = assign_fig3_plot_x(df_plot, force_to_idx, source_order, n_forces)
    box_w = mpl_boxplot_width(n_forces)
    source_colors = {KAO_LABEL: KAO_COLOR, peri_label: peri_color}
    peri_b = peri_box_brightness if peri_box_brightness is not None else COND_BOX_BRIGHTNESS
    peri_a = peri_box_alpha_hex if peri_box_alpha_hex is not None else COND_BOX_ALPHA_HEX
    peri_s = peri_box_saturation_scale if peri_box_saturation_scale is not None else COND_BOX_SATURATION_SCALE
    peri_box_face = hsb_hex(peri_color, peri_b, peri_s) + peri_a
    peri_vis_color = hsb_hex(peri_color, peri_b, min(1.0, peri_s * 1.4))
    box_palette = box_fill_palette(source_colors)
    box_palette[peri_label] = peri_box_face

    kao_only  = [f for f in kao_forces_local  if f not in peri_forces_local]
    peri_only = [f for f in peri_forces_local if f not in kao_forces_local]

    apply_plot_style()
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    bp, box_positions, flat_flags, box_sources, box_forces = draw_fig3_boxplot(
        ax, df_plot, combined_forces, source_order, box_palette, box_w,
        hollow_source=KAO_LABEL,
    )
    if bp is not None:
        resnap_fig3_boxplot_groups(bp, box_positions)
    style_filled_box_edges(ax)
    draw_fig3_stripplot(
        ax, df_plot, source_order, source_colors, box_w,
        brightness=scatter_brightness,
        participant_median=participant_median,
        triangle_keys=triangle_keys,
    )
    lower_stripplot_zorder(ax)
    finalize_boxplot_lines(ax, median_color=ACCENT_RED, median_lw=2.0)
    finish_boxplot_styling(
        bp, flat_flags, box_sources, source_order, hollow_source=KAO_LABEL,
    )

    draw_force_highlight_background(
        ax, highlight_forces, combined_forces, n_forces, box_w,
        color=highlight_force_color, alpha=highlight_force_alpha,
    )

    ax.axhline(80, color=CRITERION_COLOR, linestyle="--", linewidth=1.0, alpha=0.85,
               zorder=REF_LINE_ZORDER)

    if region_background or region_labels:
        sp_kao  = xspan(kao_only,  combined_forces)
        sp_peri = xspan(peri_only, combined_forces)
        if region_background and sp_kao:
            ax.axvspan(*sp_kao, color=KAO_COLOR, alpha=0.08, zorder=0)
        if region_labels and sp_kao:
            label_x = sp_kao[0] + (sp_kao[1] - sp_kao[0]) * REGION_ONLY_X_FRAC
            ax.text(label_x, REGION_ONLY_Y, "Fingerpad only",
                    ha="center", va="top", fontsize=FONT_ANNOT, color=KAO_COLOR,
                    fontweight="bold")
        if region_background and sp_peri:
            ax.axvspan(*sp_peri, color=peri_vis_color, alpha=0.08, zorder=0)
        if region_labels and sp_peri:
            label_x = sp_peri[0] + (sp_peri[1] - sp_peri[0]) * REGION_ONLY_X_FRAC
            ax.text(label_x, REGION_ONLY_Y, "Periungual only",
                    ha="center", va="top", fontsize=FONT_ANNOT, color=peri_vis_color,
                    fontweight="bold")

    leg_handles = [
        mpatches.Patch(facecolor="none", edgecolor=KAO_COLOR,
                       linewidth=KAO_BOX_LINEWIDTH, label=KAO_LABEL),
        mpatches.Patch(facecolor=peri_box_face,
                       edgecolor=BLACK, linewidth=BOX_LINEWIDTH, label=peri_label),
    ]
    if extra_leg_handles:
        leg_handles.extend(extra_leg_handles)
    finalize_accuracy_axes(
        fig, ax, n_forces, force_idx_order,
        [str(f) for f in combined_forces], leg_handles,
    )
    save_figure(fig, save_stem, export_widths=export_widths)
    plt.close(fig)


FIG2_COND_LABELS = {
    "In-air": "In-air",
    "On-touch (Mid)": "On-touch",
}

INAIR_LABEL = f"Periungual — In-air\n(this study, n={n_subjects})"
ONTouch_LABEL = f"Periungual: On-touch\n(this study, n={n_subjects})"


def plot_ontouch_vs_inair(df, cond_list, cond_colors, save_stem,
                          export_widths=None,
                          scatter_brightness=SCATTER_HSB_BRIGHTNESS):
    """Fig2: Periungual On-touch vs In-air (matplotlib boxes + centered strip)."""
    df_plot = df.copy()
    df_plot["Source"] = df_plot["Condition"]
    n_forces = len(USER_FORCES)
    force_to_idx = {f: i for i, f in enumerate(USER_FORCES)}
    df_plot = assign_fig3_plot_x(df_plot, force_to_idx, cond_list, n_forces)
    box_w = mpl_boxplot_width(n_forces)
    box_palette = {c: pale_box_face(cond_colors[c]) for c in cond_list}

    apply_plot_style()
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    bp, box_positions, flat_flags, box_sources, box_forces = draw_fig3_boxplot(
        ax, df_plot, USER_FORCES, cond_list, box_palette, box_w,
        hollow_source=None,
    )
    if bp is not None:
        resnap_fig3_boxplot_groups(bp, box_positions)
    style_filled_box_edges(ax)
    draw_fig3_stripplot(
        ax, df_plot, cond_list, cond_colors, box_w,
        brightness=scatter_brightness,
    )
    lower_stripplot_zorder(ax)
    finalize_boxplot_lines(ax, median_color=ACCENT_RED, median_lw=2.0)
    finish_boxplot_styling(bp, flat_flags, box_sources, cond_list, hollow_source=None)

    ax.axhline(80, color=CRITERION_COLOR, linestyle="--", linewidth=1.0, alpha=0.85,
               zorder=REF_LINE_ZORDER)

    bracket_ylim_top = annotate_fig2_condition_brackets(
        ax, bp, box_positions, box_sources, box_forces, cond_list, df_plot, SUBJECT_COL,
    )

    leg_handles = [
        mpatches.Patch(facecolor=pale_box_face(cond_colors[c]),
                       edgecolor=BLACK, linewidth=BOX_LINEWIDTH,
                       label=FIG2_COND_LABELS.get(c, c))
        for c in cond_list
    ]
    finalize_accuracy_axes(
        fig, ax, n_forces, range(n_forces),
        [str(f) for f in USER_FORCES], leg_handles,
        ylim_top=bracket_ylim_top,
    )
    save_figure(fig, save_stem, export_widths=export_widths)
    plt.close(fig)


def run_all_figures(export_widths=None, in_air=IN_AIR, on_touch=ON_TOUCH,
                    scatter_brightness=SCATTER_HSB_BRIGHTNESS):
    """Generate Fig1–3 (default: all column widths, amber In-air + teal On-touch)."""
    if export_widths is None:
        export_widths = EXPORT_WIDTHS_PX

    pale_box_kw = dict(
        peri_box_brightness=COND_BOX_BRIGHTNESS,
        peri_box_alpha_hex=COND_BOX_ALPHA_HEX,
        peri_box_saturation_scale=COND_BOX_SATURATION_SCALE,
    )

    sns.set_theme(style="white")
    plot_kao_vs_periungual(
        df_raw[df_raw["Condition"] == "In-air"],
        INAIR_LABEL,
        in_air,
        "Fig1_fingerpad_nopaint_vs_inair",
        export_widths=export_widths,
        scatter_brightness=scatter_brightness,
        **pale_box_kw,
    )

    plot_kao_vs_periungual(
        df_raw[df_raw["Condition"] == "On-touch (Mid)"],
        ONTouch_LABEL,
        on_touch,
        "Fig3_fingerpad_nopaint_vs_periungual_ontouch",
        export_widths=export_widths,
        scatter_brightness=scatter_brightness,
        **pale_box_kw,
    )

    cond_list = [c for c in ["In-air", "On-touch (Mid)"] if c in df_raw["Condition"].unique()]
    cond_colors = {"In-air": in_air, "On-touch (Mid)": on_touch}

    plot_ontouch_vs_inair(
        df_raw,
        cond_list,
        cond_colors,
        "Fig2_ontouch_vs_inair",
        export_widths=export_widths,
        scatter_brightness=scatter_brightness,
    )

    export_fig3_10559A_2col(scatter_brightness=scatter_brightness, **pale_box_kw)
    export_fig3_10559A_2col_v2(scatter_brightness=scatter_brightness, **pale_box_kw)
    export_fig3_future_0p4g(scatter_brightness=scatter_brightness, **pale_box_kw)


def export_fig3_10559A_2col(scatter_brightness=SCATTER_HSB_BRIGHTNESS, **pale_box_kw):
    """Fig3 10559A 2-col export (no region bg/labels, 0.07 g highlight)."""
    plot_kao_vs_periungual(
        df_raw[df_raw["Condition"] == "On-touch (Mid)"],
        ONTouch_LABEL,
        ON_TOUCH,
        "Fig3_fingerpad_nopaint_vs_periungual_ontouch_10559A",
        export_widths=(("2col", 2102),),
        region_background=False,
        region_labels=False,
        highlight_forces=[0.07],
        scatter_brightness=scatter_brightness,
        **pale_box_kw,
    )


def export_fig3_10559A_2col_v2(scatter_brightness=SCATTER_HSB_BRIGHTNESS, **pale_box_kw):
    """Fig3 v2: scatter collapsed to 1 point per participant (participant median)."""
    plot_kao_vs_periungual(
        df_raw[df_raw["Condition"] == "On-touch (Mid)"],
        ONTouch_LABEL,
        ON_TOUCH,
        "Fig3_fingerpad_nopaint_vs_periungual_ontouch_10559A_v2",
        export_widths=(("2col", 2102),),
        region_background=False,
        region_labels=False,
        highlight_forces=[0.07],
        scatter_brightness=scatter_brightness,
        participant_median=True,
        **pale_box_kw,
    )


def export_fig3_future_0p4g(scatter_brightness=SCATTER_HSB_BRIGHTNESS, **pale_box_kw):
    """Figure 3 extended to 0.4 g: uses real P61/P62/P63 data at 0.4 g (triangles) + Kao 0.4 g."""
    # --- Kao data extended to include 0.4 g ---
    kao_rows_ext = []
    for force, vals in KAO_PAINT_RAW.items():
        if force <= 0.4:
            for pid, v in enumerate(vals):
                kao_rows_ext.append({
                    "Force_Val": float(force),
                    "Score":     float(v),
                    "Source":    KAO_LABEL,
                    "Participant": f"KP{pid + 1}",
                })
    df_kao_ext = pd.DataFrame(kao_rows_ext)

    # Real On-touch data; df_raw already filtered so P61/P62/P63 contribute only 0.4 g
    df_peri_combined = df_raw[df_raw["Condition"] == "On-touch (Mid)"].copy()

    plot_kao_vs_periungual(
        df_peri_combined,
        ONTouch_LABEL,
        ON_TOUCH,
        "Fig3_future_0p4g",
        export_widths=(("2col", 2102),),
        region_background=False,
        region_labels=False,
        highlight_forces=[0.07, 0.4],
        scatter_brightness=scatter_brightness,
        kao_df_override=df_kao_ext,
        triangle_keys={(ONTouch_LABEL, 0.4)},
        participant_median=True,
        **pale_box_kw,
    )


if __name__ == "__main__":
    if not HAS_STATSMODELS:
        raise SystemExit(
            "statsmodels is required for Fig2 LME brackets.\n"
            "Use the project venv:\n"
            "  .venv/bin/python \"(New)Analysis/ATDAnalysis/ATD_C1_Fig(Anika).py\""
        )
    run_all_figures()
    print("\nDone.")