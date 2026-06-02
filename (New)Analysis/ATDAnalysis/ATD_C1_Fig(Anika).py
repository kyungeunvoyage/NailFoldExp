"""
ATD Comparison Figures
======================
Figure 1: Kao fingerpad Paint vs. Periungual In-air (this study)
Figure 2: Periungual On-touch vs. In-air (this study)
Figure 3: Kao fingerpad Paint vs. Periungual On-touch (Mid) (this study)

Kao et al. 2022 Paint condition values are digitized from published Fig. B (n=5).
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
import colorsys
import seaborn as sns
from matplotlib import rcParams
from matplotlib.transforms import blended_transform_factory

# =============================================================================
# Palette
# =============================================================================
SLATE_BLUE = "#56708A"   # In-air
ON_TOUCH   = "#295E11"   # Periungual On-touch
ON_TOUCH_TEAL = "#10559A"  # alternate On-touch (2-col export variant)
ACCENT_RED = "#BF2C23"   # median line & 80% reference
REF_LINE_ZORDER = 20     # accuracy criterion dashed line — top layer
BLACK      = "#1A1A1A"
KAO_COLOR  = "#5A5A5A"   # Anika Paint — dark gray (matches original paper)

BOX_ALPHA_HEX = "99"     # ~60% opacity (lighter fill so scatter reads through)
STRIP_ALPHA   = 0.50
BOX_HSB_BRIGHTNESS     = 0.65  # HSB brightness (HSV V) for box fill — lighter than scatter
SCATTER_HSB_BRIGHTNESS = 0.60  # HSB brightness (HSV V) for scatter points

FIG_SIZE  = (8.0, 4.5)   # design aspect ratio (w×h inches)
SAVE_DPI  = 300            # master raster before column-width resize
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
REGION_ONLY_Y = 110       # y for "Fingerpad only" / "Periungual only" labels
REGION_ONLY_X_FRAC = 0.28  # along shaded span (0=left, 0.5=center)

# Inward tick guides drawn manually (seaborn categorical axes hide mpl ticks)
TICK_LEN_AXES = 0.016  # fraction of axis length into the plot

BOX_LINEWIDTH = 1.4
KAO_BOX_LINEWIDTH = 1.4   # thicker stroke for hollow Kao boxes
CAP_LINEWIDTH = 0.5
CAP_WIDTH     = 0.10   # matplotlib capwidths — short end ticks (default ~0.5× box)

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


def add_legend_outside(fig, handles, ncol=2, title=None):
    """Place legend below the axes, outside the plot area."""
    fig.subplots_adjust(left=0.11, right=0.98, top=0.94, bottom=0.15)
    fig.legend(
        handles=handles,
        title=title,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.02),
        bbox_transform=fig.transFigure,
        ncol=ncol,
        fontsize=FONT_LEGEND,
        title_fontsize=FONT_LEGEND,
        frameon=False,
        labelspacing=0.35,
        columnspacing=1.5,
        handlelength=1.4,
        handleheight=1.0,
        borderaxespad=0.2,
    )


def save_figure(fig, stem, export_widths=None):
    """Save PNG at publication column widths (px)."""
    from PIL import Image

    if export_widths is None:
        export_widths = EXPORT_WIDTHS_PX

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=SAVE_DPI, bbox_inches="tight", facecolor="white")
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
        "figure.dpi":            150,
        "savefig.dpi":           SAVE_DPI,
        "svg.fonttype":          "path",
    })


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
            line.set_color(BLACK)
            line.set_linewidth(cap_lw)
            line.set_zorder(4)
        else:
            line.set_visible(False)


def style_kao_hollow_boxes(ax, combined_forces, kao_forces, box_width=0.55,
                           kao_hue_idx=0, n_hue=2, edge_color=KAO_COLOR,
                           linewidth=KAO_BOX_LINEWIDTH):
    """Hollow gray stroke for Kao boxes only (by x position, not patch index)."""
    kao_set = {float(f) for f in kao_forces}
    dodge = box_width / n_hue
    kao_offset = (kao_hue_idx - (n_hue - 1) / 2) * dodge

    for patch in ax.patches:
        verts = patch.get_path().vertices
        xc = float(verts[:, 0].mean())
        yspan = float(verts[:, 1].max() - verts[:, 1].min())
        cat = int(round(xc))
        if cat < 0 or cat >= len(combined_forces):
            continue
        if yspan < 1e-6:
            patch.set_visible(False)
            continue
        if not np.isclose(xc - cat, kao_offset, atol=dodge * 0.35):
            continue
        if float(combined_forces[cat]) in kao_set:
            patch.set_facecolor("none")
            patch.set_edgecolor(edge_color)
            patch.set_linewidth(linewidth)
        else:
            patch.set_visible(False)


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


def box_fill_palette(source_colors, brightness=BOX_HSB_BRIGHTNESS):
    """Box fill colors at fixed HSB brightness (+ alpha hex suffix)."""
    return {k: hsb_hex(v, brightness) + BOX_ALPHA_HEX for k, v in source_colors.items()}


def _hsb_scatter_rgba(base_hex, brightness=SCATTER_HSB_BRIGHTNESS, alpha=STRIP_ALPHA):
    """Scatter color: keep hue & saturation, fix HSB brightness (HSV V)."""
    r, g, b = mcolors.to_rgb(base_hex)
    h, s, _v = colorsys.rgb_to_hsv(r, g, b)
    rr, gg, bb = colorsys.hsv_to_rgb(h, s, brightness)
    return (rr, gg, bb, alpha)


def apply_scatter_hsb_brightness(ax, hue_order, palette, brightness=SCATTER_HSB_BRIGHTNESS):
    """Set all stripplot points to fixed HSB brightness per hue."""
    cols = [c for c in ax.collections if len(c.get_offsets())]
    if len(cols) < len(hue_order):
        return
    cols = cols[-len(hue_order):]
    for coll, hue in zip(cols, hue_order):
        rgba = _hsb_scatter_rgba(palette[hue], brightness)
        n = len(coll.get_offsets())
        coll.set_facecolors([rgba] * n)
        coll.set_edgecolors("none")
        coll.set_linewidths(0)

def label_accuracy_criterion_left(ax, y=81.5):
    """Place accuracy criterion label at the left of the plot."""
    ax.text(-0.55, y, "accuracy \ncriterion (80%)", ha="left", va="bottom",
            fontsize=FONT_ANNOT, color=ACCENT_RED, fontweight="bold")

# =============================================================================
# Kao et al. 2022 — Paint condition
# Digitized from Fig. B of the paper (n=5, index fingerpad)
# y = Percent correct (%)
# =============================================================================
KAO_PAINT_RAW = {
    # force(g) : [P1, P2, P3, P4, P5]
    0.02: [  0,  30,  47,  65,  68],
    0.04: [ 35,  78,  82,  88,  97],
    0.07: [ 32,  60,  82,  88, 100],
    0.40: [ 32,  78,  80,  87, 100],
    1.00: [ 48, 100, 100, 100, 100],
    1.40: [ 80, 100, 100, 100, 100],
}
KAO_N = 5

# Build tidy DataFrame
kao_rows = []
for force, vals in KAO_PAINT_RAW.items():
    for pid, v in enumerate(vals):
        kao_rows.append({
            "Force_Val":  float(force),
            "Score":      float(v),
            "Source":     "Kao_Paint",
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

n_subjects = df_raw["SubjectID"].nunique() if "SubjectID" in df_raw.columns else len(all_files)


def calc_score(row):
    if row["Target"] == 0:
        return 100.0 if row["Response"] == 0 else 0.0
    return max(0.0, (1 - abs(row["Target"] - row["Response"]) / row["Target"]) * 100.0)


df_raw["Score"] = df_raw.apply(calc_score, axis=1)

USER_FORCES   = sorted(df_raw["Force_Val"].unique())   # [0.07, 0.16, 0.6, 1.0, 1.4]
KAO_FORCES    = sorted(KAO_PAINT_RAW.keys())           # [0.02, 0.04, 0.07, 0.4, 1.0, 1.4]

# =============================================================================
# Kao vs Periungual comparison (shared plotting)
# =============================================================================
KAO_LABEL = f"Fingerpad (Kao et al. 2022, n={KAO_N})"

df_kao_plot = df_kao.copy()
df_kao_plot["Source"] = KAO_LABEL

COMBINED_FORCES = sorted(set(KAO_FORCES) | set(USER_FORCES))


def xspan(forces_sub, all_forces, pad=0.48):
    idxs = [list(all_forces).index(f) for f in forces_sub if f in all_forces]
    return (min(idxs) - pad, max(idxs) + pad) if idxs else None


def plot_kao_vs_periungual(df_periungual, peri_label, peri_color, save_stem,
                           export_widths=None,
                           box_brightness=BOX_HSB_BRIGHTNESS,
                           scatter_brightness=SCATTER_HSB_BRIGHTNESS,
                           peri_box_brightness=None,
                           peri_box_alpha_hex=None,
                           peri_box_saturation_scale=1.0):
    """Box/strip plot: Kao fingerpad Paint vs one periungual condition."""
    df_peri = df_periungual.copy()
    df_peri["Source"] = peri_label

    df_plot = pd.concat(
        [df_kao_plot[["Force_Val", "Score", "Source"]],
         df_peri[["Force_Val", "Score", "Source"]]],
        ignore_index=True,
    )
    source_order = [KAO_LABEL, peri_label]
    source_colors = {KAO_LABEL: KAO_COLOR, peri_label: peri_color}
    peri_b = peri_box_brightness if peri_box_brightness is not None else box_brightness
    peri_a = peri_box_alpha_hex if peri_box_alpha_hex is not None else BOX_ALPHA_HEX
    peri_s = peri_box_saturation_scale
    peri_box_face = hsb_hex(peri_color, peri_b, peri_s) + peri_a
    peri_vis_color = hsb_hex(peri_color, peri_b, min(1.0, peri_s * 1.4))
    box_palette = box_fill_palette(source_colors, box_brightness)
    box_palette[peri_label] = peri_box_face

    kao_only = [f for f in KAO_FORCES if f not in USER_FORCES]
    peri_only = [f for f in USER_FORCES if f not in KAO_FORCES]

    apply_plot_style()
    fig, ax = plt.subplots(figsize=FIG_SIZE)

    sns.boxplot(
        data=df_plot,
        x="Force_Val", y="Score",
        hue="Source", hue_order=source_order,
        order=COMBINED_FORCES,
        palette=box_palette,
        linewidth=BOX_LINEWIDTH, fliersize=0, width=0.55,
        capwidths=CAP_WIDTH,
        medianprops={"color": ACCENT_RED, "linewidth": 2.0},
        whiskerprops={"linewidth": BOX_LINEWIDTH, "color": BLACK},
        capprops={"linewidth": CAP_LINEWIDTH, "color": BLACK},
        boxprops={"linewidth": BOX_LINEWIDTH},
        legend=False, ax=ax,
    )
    style_kao_hollow_boxes(ax, COMBINED_FORCES, KAO_FORCES, box_width=0.55)
    sns.stripplot(
        data=df_plot,
        x="Force_Val", y="Score",
        hue="Source", hue_order=source_order,
        order=COMBINED_FORCES,
        palette=source_colors,
        dodge=True, alpha=STRIP_ALPHA,
        size=3.8, jitter=0.15, linewidth=0,
        legend=False, ax=ax,
    )
    apply_scatter_hsb_brightness(ax, source_order, source_colors, scatter_brightness)
    lower_stripplot_zorder(ax)
    finalize_boxplot_lines(ax, median_color=ACCENT_RED, median_lw=2.0)

    ax.axhline(80, color=ACCENT_RED, linestyle="--", linewidth=1.0, alpha=0.85,
               zorder=REF_LINE_ZORDER)
    label_accuracy_criterion_left(ax)

    sp_kao = xspan(kao_only, COMBINED_FORCES)
    sp_peri = xspan(peri_only, COMBINED_FORCES)
    if sp_kao:
        ax.axvspan(*sp_kao, color=KAO_COLOR, alpha=0.08, zorder=0)
        label_x = sp_kao[0] + (sp_kao[1] - sp_kao[0]) * REGION_ONLY_X_FRAC
        ax.text(label_x, REGION_ONLY_Y, "Fingerpad only",
                ha="center", va="top", fontsize=FONT_ANNOT, color=KAO_COLOR,
                fontweight="bold")
    if sp_peri:
        ax.axvspan(*sp_peri, color=peri_vis_color, alpha=0.08, zorder=0)
        label_x = sp_peri[0] + (sp_peri[1] - sp_peri[0]) * REGION_ONLY_X_FRAC
        ax.text(label_x, REGION_ONLY_Y, "Periungual only",
                ha="center", va="top", fontsize=FONT_ANNOT, color=peri_vis_color,
                fontweight="bold")

    leg_handles = [
        mpatches.Patch(facecolor="none", edgecolor=KAO_COLOR,
                       linewidth=KAO_BOX_LINEWIDTH, label=KAO_LABEL),
        mpatches.Patch(facecolor=peri_box_face,
                       edgecolor=BLACK, linewidth=0.7, label=peri_label),
    ]
    ax.set_xlabel("Stimulus Force (g)", fontsize=FONT_LABEL, labelpad=6)
    ax.set_ylabel("Detection Accuracy (%)", fontsize=FONT_LABEL, labelpad=6)
    ax.set_ylim(-5, 115)
    ax.set_xticks(range(len(COMBINED_FORCES)))
    ax.set_xticklabels([str(f) for f in COMBINED_FORCES], fontsize=FONT_TICK)
    sns.despine(ax=ax)
    add_inward_tick_guides(ax, len(COMBINED_FORCES))
    add_legend_outside(fig, leg_handles, ncol=2)
    save_figure(fig, save_stem, export_widths=export_widths)
    plt.close(fig)


INAIR_LABEL = f"Periungual — In-air\n(this study, n={n_subjects})"
ONTouch_LABEL = f"Periungual: On-touch (this study, n={n_subjects})"


def run_all_figures(export_widths=None, on_touch=ON_TOUCH,
                    box_brightness=BOX_HSB_BRIGHTNESS,
                    scatter_brightness=SCATTER_HSB_BRIGHTNESS):
    """Generate Fig1–3 (default: all column widths, green On-touch)."""
    if export_widths is None:
        export_widths = EXPORT_WIDTHS_PX

    sns.set_theme(style="white")
    plot_kao_vs_periungual(
        df_raw[df_raw["Condition"] == "In-air"],
        INAIR_LABEL,
        SLATE_BLUE,
        "Fig1_fingerpad_paint_vs_inair",
        export_widths=export_widths,
        box_brightness=box_brightness,
        scatter_brightness=scatter_brightness,
    )

    plot_kao_vs_periungual(
        df_raw[df_raw["Condition"] == "On-touch (Mid)"],
        ONTouch_LABEL,
        on_touch,
        "Fig3_fingerpad_paint_vs_periungual_ontouch",
        export_widths=export_widths,
        box_brightness=box_brightness,
        scatter_brightness=scatter_brightness,
    )

    cond_list = [c for c in ["In-air", "On-touch (Mid)"] if c in df_raw["Condition"].unique()]
    cond_colors = {"In-air": SLATE_BLUE, "On-touch (Mid)": on_touch}

    apply_plot_style()
    fig2, ax2 = plt.subplots(figsize=FIG_SIZE)

    sns.boxplot(
        data=df_raw,
        x="Force_Val", y="Score",
        hue="Condition", hue_order=cond_list,
        order=USER_FORCES,
        palette=box_fill_palette(cond_colors, box_brightness),
        linewidth=BOX_LINEWIDTH, fliersize=0, width=0.55,
        capwidths=CAP_WIDTH,
        medianprops={"color": ACCENT_RED, "linewidth": 2.0},
        whiskerprops={"linewidth": BOX_LINEWIDTH, "color": BLACK},
        capprops={"linewidth": CAP_LINEWIDTH, "color": BLACK},
        boxprops={"linewidth": BOX_LINEWIDTH},
        legend=False, ax=ax2,
    )

    sns.stripplot(
        data=df_raw,
        x="Force_Val", y="Score",
        hue="Condition", hue_order=cond_list,
        order=USER_FORCES,
        palette=cond_colors,
        dodge=True, alpha=STRIP_ALPHA,
        size=3.5, jitter=0.18, linewidth=0,
        legend=False, ax=ax2,
    )
    apply_scatter_hsb_brightness(ax2, cond_list, cond_colors, scatter_brightness)
    lower_stripplot_zorder(ax2)
    finalize_boxplot_lines(ax2, median_color=ACCENT_RED, median_lw=2.0)

    if ax2.get_legend() is not None:
        ax2.get_legend().remove()

    ax2.axhline(80, color=ACCENT_RED, linestyle="--", linewidth=1.0, alpha=0.85,
                zorder=REF_LINE_ZORDER)
    label_accuracy_criterion_left(ax2)

    leg_handles2 = [
        mpatches.Patch(facecolor=hsb_hex(cond_colors[c], box_brightness) + BOX_ALPHA_HEX,
                       edgecolor=BLACK, linewidth=0.7, label=c)
        for c in cond_list
    ]
    ax2.set_xlabel("Stimulus Force (g)", fontsize=FONT_LABEL, labelpad=6)
    ax2.set_ylabel("Detection Accuracy (%)", fontsize=FONT_LABEL, labelpad=6)
    ax2.set_ylim(-5, 110)
    ax2.set_xticks(range(len(USER_FORCES)))
    ax2.set_xticklabels([str(f) for f in USER_FORCES], fontsize=FONT_TICK)
    sns.despine(ax=ax2)
    add_inward_tick_guides(ax2, len(USER_FORCES))
    add_legend_outside(fig2, leg_handles2, ncol=2, title="Condition")

    save_figure(fig2, "Fig2_ontouch_vs_inair", export_widths=export_widths)
    plt.close(fig2)


if __name__ == "__main__":
    run_all_figures()
    print("\nDone.")