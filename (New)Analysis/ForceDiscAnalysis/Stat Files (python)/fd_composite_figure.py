"""
fd_composite_figure.py — self-contained FD composite renderer.

Layouts:
  horizontal (default): 3-col × 2-row — plot types as columns, bands as rows
  vertical:             2-col × 3-row — Low | High as columns, plot types as rows

Output:
  ForceDiscAnalysis/Final/fd_composite_grid_2col(Final).png       (horizontal)
  ForceDiscAnalysis/Final/fd_composite_grid_vertical(Final).png   (vertical)
"""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.transforms import blended_transform_factory
from matplotlib.ticker import FixedLocator

from gee_export_utils import (
    EXPORT_WIDTH_2COL,
    EXPORT_HEIGHT_2COL,
    ON_TOUCH_BLUE,
    on_touch_box_color,
    on_touch_scatter_rgba,
    on_touch_hatch_rgba,
    off_nail_box_color,
    off_nail_scatter_rgba,
)

# ── Canvas & layout (pixel-anchored) ─────────────────────────────────────────
ONNAIL_LEGEND_RESERVE_PX = 52   # extra canvas height — row-1 headers above (1,1)–(3,1)
LOW_ROW_HEADER_GAP_PX = 8
COL_TITLE_LIFT_PX = 12          # nudge (1,1)/(2,1) titles slightly above legend row
COL1_ROW1_TITLE = "2AFC"
COL2_ROW1_TITLE = "SAME/DIFFERENT"
ONNAIL_LEGEND_LABELS = ("On-nail (C+D)", "Off-nail (A+F)")

# Horizontal composite: standard 2-col export width + headroom for on-nail legend
GRID_WIDTH_PX_H = EXPORT_WIDTH_2COL
GRID_HEIGHT_PX_H = EXPORT_HEIGHT_2COL + ONNAIL_LEGEND_RESERVE_PX

MARGIN_LEFT_PX = 88    # y tick digits
MARGIN_RIGHT_PX = 22
MARGIN_BOTTOM_PX = 58   # x pair tick labels
MARGIN_TOP_PX = 32
COL_GAP_PX = 42
ROW_GAP_PX = 65
TITLE_RESERVE_PX = 0
XLABEL_RESERVE_PX = 42
LEFT_Y_TICK_RESERVE_PX = 32
LEFT_YLABEL_RESERVE_PX = 48  # vertical layout only
TICK_RESERVE_PX = 20

FONT_SCALE = 0.72
AXIS_TICK_FONT_SCALE = 0.52
AXIS_LABEL_FONT_SCALE = 0.52   # per-panel axis title (vertical layout)
Y_LABEL_FONT_SCALE = 0.6
Y_TICK_FONT_SCALE = FONT_SCALE
COL12_TICK_FONT_SCALE = Y_TICK_FONT_SCALE
COL12_Y_TICK_FONT_SCALE = Y_TICK_FONT_SCALE
COL12_LABEL_FONT_SCALE = 0.28
PAIR_X_TICK_FONT_SCALE = 0.84   # pair labels on x-axis (both rows)
DOT_SCALE = 0.72
LINE_SCALE = 0.90

# Box width — ATD Fig3-like fill per category slot on 2-col sub-panels
POOLED_BOX_REF = 0.45
BOX_WIDTH_SCALE = 4.2
MAX_BOX_FRAC = 0.72
CRITERION_PCT = 75.0

# Scatter marker radius (data-point units) — match Stats(GEE) STRIP_SIZE / ATD strip
SCATTER_RADIUS = 3.8
SCATTER_ROW_SCALE_GEE = {0: 1.0, 1: 1.0}
SCATTER_ROW_SCALE_SD = {0: 1.0, 1: 1.0}

NROWS_H = 2
NCOLS_H = 3
COL_FRACS_H = [0.30, 0.28, 0.42]  # GEE | SD | On-nail — wider col 3 for matched boxes

NROWS_V = 3
NCOLS_V = 2
ROW_FRACS_V = [0.30, 0.30, 0.40]  # GEE | SD | On-nail
COL_FRACS_V = [0.50, 0.50]        # Low | High

GRID_WIDTH_PX_V = 5600
GRID_HEIGHT_PX_V = 10000
# Col 3 on-nail — dodge On/Off at each pair tick (same x scale as SD col)
ONNAIL_DODGE_BOX_FRAC = 0.50   # base: center offset ≥ half box width
ONNAIL_PAIR_INNER_GAP_FRAC = 0.08  # extra gap between On vs Off within each pair
ONNAIL_DODGE_GAP_EXTRA_PX = 2
ONNAIL_BOX_WIDTH_FRAC = 0.93   # (3,1)/(3,2) — slight trim vs GEE-matched width
ONNAIL_ON_TOUCH_HATCH = "//"   # on-touch (On-nail) box fill — diagonal hatch
COL_GEE = 0
COL_ONNAIL = 2
XLABEL_FORCE_PAIR = "Force Stimulus pairs (g)"
YLABEL_ACCURACY = "Discrimination Accuracy (%)"
X_LABEL_FONT_SCALE = AXIS_LABEL_FONT_SCALE  # per-panel fallback (vertical layout)

SCRIPT_DIR = Path(__file__).resolve().parent
FD_ROOT = SCRIPT_DIR.parent
FINAL_DIR = FD_ROOT / "Final"
OUTPUT_STATS = FD_ROOT / "Output" / "Stats(GEE)"
OUTPUT_SD = FD_ROOT / "Output" / "SameDiff_GEE"


# ── ATD style loader ─────────────────────────────────────────────────────────
def _load_atd_c1():
    root = SCRIPT_DIR.parent.parent / "ATDAnalysis"
    for sub in ("Stat files", "Stat files (final) "):
        path = root / sub / "(Final)ATD_C1_Fig(Anika).py"
        if path.is_file():
            spec = importlib.util.spec_from_file_location("atd_c1_fig", path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    raise FileNotFoundError(f"Could not find (Final)ATD_C1_Fig(Anika).py under {root}")


def _box_fill(atd):
    return on_touch_box_color(atd)


def _scatter_rgba(atd):
    return on_touch_scatter_rgba(atd)


# ── Grid geometry ────────────────────────────────────────────────────────────
def _grid_kw(width_px=GRID_WIDTH_PX_H, height_px=GRID_HEIGHT_PX_H, *, margin_top_px=MARGIN_TOP_PX):
    return dict(
        left=MARGIN_LEFT_PX / width_px,
        right=MARGIN_RIGHT_PX / width_px,
        bottom=MARGIN_BOTTOM_PX / height_px,
        top=1.0 - margin_top_px / height_px,
        wspace=COL_GAP_PX / width_px,
        hspace=ROW_GAP_PX / height_px,
    )


def _layout_spec(layout="horizontal"):
    if layout == "vertical":
        w_px, h_px = GRID_WIDTH_PX_V, GRID_HEIGHT_PX_V
        return dict(
            layout=layout,
            nrows=NROWS_V,
            ncols=NCOLS_V,
            col_fracs=COL_FRACS_V,
            row_fracs=ROW_FRACS_V,
            width_px=w_px,
            height_px=h_px,
            canvas=(8.0, round(8.0 * h_px / w_px, 3)),
            output_name="fd_composite_grid_vertical(Final).png",
            band_index=lambda plot_row, band_col: band_col,
            plot_rows={"gee": 0, "sd": 1, "onnail": 2},
            plot_cols={"gee": 0, "sd": 1, "onnail": 2},
            left_band_col=0,
        )
    w_px, h_px = GRID_WIDTH_PX_H, GRID_HEIGHT_PX_H
    return dict(
        layout="horizontal",
        nrows=NROWS_H,
        ncols=NCOLS_H,
        col_fracs=COL_FRACS_H,
        row_fracs=None,
        width_px=w_px,
        height_px=h_px,
        margin_top_px=MARGIN_TOP_PX + ONNAIL_LEGEND_RESERVE_PX,
        canvas=(8.0, round(8.0 * h_px / w_px, 3)),
        output_name="fd_composite_grid_2col(Final).png",
        band_index=lambda plot_row, band_col: plot_row if layout == "vertical" else band_col,
        plot_rows={"gee": 0, "sd": 1, "onnail": 2},
        plot_cols={"gee": 0, "sd": 1, "onnail": 2},
        left_band_col=0,
    )


def _cell_coords(layout_spec, *, band_col, plot_kind):
    """Map band + plot type → ``_panel_rects`` cell key."""
    if layout_spec["layout"] == "vertical":
        return (layout_spec["plot_rows"][plot_kind], band_col)
    return (band_col, layout_spec["plot_cols"][plot_kind])


def _col_width_frac(col, col_fracs, *, ncols, **grid_kw):
    inner = 1.0 - grid_kw["left"] - grid_kw["right"]
    usable = inner - (ncols - 1) * grid_kw["wspace"]
    total = sum(col_fracs)
    return usable * col_fracs[col] / total


def _cell_height_frac(*, nrows, row_fracs=None, **grid_kw):
    inner = grid_kw["top"] - grid_kw["bottom"]
    if row_fracs is None:
        return (inner - (nrows - 1) * grid_kw["hspace"]) / nrows
    usable = inner - (nrows - 1) * grid_kw["hspace"]
    return usable / sum(row_fracs)


def _panel_rects(*, nrows, ncols, col_fracs, row_fracs=None, **grid_kw):
    inner_w = 1.0 - grid_kw["left"] - grid_kw["right"]
    usable_w = inner_w - (ncols - 1) * grid_kw["wspace"]
    col_ws = [usable_w * f / sum(col_fracs) for f in col_fracs]

    inner_h = grid_kw["top"] - grid_kw["bottom"]
    if row_fracs is None:
        row_hs = [_cell_height_frac(nrows=nrows, **grid_kw)] * nrows
    else:
        usable_h = inner_h - (nrows - 1) * grid_kw["hspace"]
        row_hs = [usable_h * f / sum(row_fracs) for f in row_fracs]

    rects = {}
    y_top = grid_kw["top"]
    for row in range(nrows):
        h = row_hs[row]
        y = y_top - h
        x = grid_kw["left"]
        for col in range(ncols):
            rects[(row, col)] = [x, y, col_ws[col], h]
            x += col_ws[col] + grid_kw["wspace"]
        y_top = y - grid_kw["hspace"]
    return rects


def _plot_rect(
    cell_rect, *,
    height_px=GRID_HEIGHT_PX_H,
    width_px=GRID_WIDTH_PX_H,
    left_tick_reserve_px=0,
    bottom_reserve_px=None,
):
    left, bottom, w, h = cell_rect
    bottom_reserve = XLABEL_RESERVE_PX if bottom_reserve_px is None else bottom_reserve_px
    left_frac = left_tick_reserve_px / width_px
    bottom_frac = bottom_reserve / height_px
    plot_left = left + left_frac
    plot_w = max(w - left_frac, 0.05)
    cell_h_px = h * height_px
    plot_h_px = cell_h_px - TITLE_RESERVE_PX - bottom_reserve
    plot_bottom = bottom + bottom_frac
    return [plot_left, plot_bottom, plot_w, max(plot_h_px / height_px, 0.05)]


def _left_axis_reserve_px(*, layout, band_col, plot_kind, spec, show_ylabel):
    if layout == "vertical":
        is_left_col = band_col == spec["left_band_col"]
    else:
        is_left_col = plot_kind == "gee"
    reserve = LEFT_Y_TICK_RESERVE_PX if is_left_col else 0
    if is_left_col and show_ylabel and layout != "horizontal":
        reserve += LEFT_YLABEL_RESERVE_PX
    return reserve


def _panel_show_xlabel(*, layout, plot_row, band_col, bottom_plot_row, nrows):
    if layout == "horizontal":
        return False
    if layout == "vertical":
        return plot_row == bottom_plot_row
    return band_col == nrows - 1


def _panel_show_x_pair_labels(*, layout, band_col, nrows):
    """Pair tick text on each row — horizontal: (1,*)(2,*)(3,*) Low + High."""
    if layout == "horizontal":
        return True
    return band_col == nrows - 1


def _box_width_atd(atd, n_x):
    """Same data-unit width as ATD Fig3, capped so boxes do not overlap neighbors."""
    raw = atd.mpl_boxplot_width(max(n_x, 2)) * BOX_WIDTH_SCALE
    return min(raw, MAX_BOX_FRAC)


def _onnail_col_width_ratio(col_fracs, *, layout):
    """Compensate narrower/wider on-nail column so box pixel width matches col 1 GEE."""
    if layout == "vertical":
        return 1.0
    return col_fracs[COL_GEE] / col_fracs[COL_ONNAIL]


def _plot_width_frac(cell_rect, *, height_px, width_px, left_tick_reserve_px=0):
    return _plot_rect(
        cell_rect,
        height_px=height_px,
        width_px=width_px,
        left_tick_reserve_px=left_tick_reserve_px,
    )[2]


def _onnail_box_width_match_gee_panel(gee_box_w, gee_plot_w, onnail_plot_w):
    """Match each on-nail dodged box pixel width to (1,1) GEE at the same xlim span."""
    return gee_box_w * (gee_plot_w / onnail_plot_w)


def _onnail_dodge_for_box(box_w):
    """On/Off center offset — leaves ONNAIL_PAIR_INNER_GAP_FRAC × box_w between boxes."""
    return box_w * (ONNAIL_DODGE_BOX_FRAC + ONNAIL_PAIR_INNER_GAP_FRAC / 2.0)


def _onnail_dodge_bump_px(plot_w_frac, width_px, xlim_n_pairs, center_dist_extra_px):
    """Convert extra On/Off center distance (px) to data-unit dodge bump."""
    plot_w_px = plot_w_frac * width_px
    if plot_w_px <= 0:
        return 0.0
    span = _onnail_xlim_span(xlim_n_pairs)
    return (center_dist_extra_px / 2.0) * span / plot_w_px


def _onnail_box_width(sd_box_w):
    """Legacy dodged width (fallback only)."""
    return sd_box_w * 0.58


def _onnail_xlim_span(n_pairs):
    """X-axis data span (matches xlim −0.55 … n−1+0.45)."""
    return max(n_pairs - 1, 0) + 1.0


def _onnail_box_width_pixel_match(ref_box_w, ref_n_pairs, n_pairs):
    """Scale data-unit width so on-nail box pixel width matches the ref panel."""
    return ref_box_w * _onnail_xlim_span(n_pairs) / _onnail_xlim_span(ref_n_pairs)


def _onnail_dodge_pixel_match(ref_dodge, ref_n_pairs, n_pairs):
    """Scale On/Off dodge so pixel gap matches the ref panel."""
    return ref_dodge * _onnail_xlim_span(n_pairs) / _onnail_xlim_span(ref_n_pairs)


# ── Drawing helpers ──────────────────────────────────────────────────────────
def _fs_tick(base, *, tick_scale=AXIS_TICK_FONT_SCALE):
    return base * tick_scale


def _fs_axis_label(base, *, label_scale=AXIS_LABEL_FONT_SCALE):
    return base * label_scale


def _jitter(n, width=0.18, seed=42):
    return (np.random.default_rng(seed).random(n) - 0.5) * width


def _draw_criterion_line(ax, atd, *, y=CRITERION_PCT):
    """75% dashed criterion — spans full panel width."""
    z = getattr(atd, "REF_LINE_ZORDER", 20)
    ax.axhline(
        y, color=atd.CRITERION_COLOR, linestyle="--",
        linewidth=1.0, alpha=0.85, zorder=z, clip_on=False,
    )


def _finalize_accuracy_axes(
    ax, atd, n_x, ylim_top, *, show_ylabel=True, show_xlabel=True,
    show_yticklabels=True,
    tick_scale=AXIS_TICK_FONT_SCALE, label_scale=AXIS_LABEL_FONT_SCALE,
    y_tick_scale=None, y_label_scale=Y_LABEL_FONT_SCALE,
):
    fs_tick = _fs_tick(atd.FONT_TICK, tick_scale=tick_scale)
    fs_y_tick = _fs_tick(
        atd.FONT_TICK, tick_scale=y_tick_scale if y_tick_scale is not None else tick_scale,
    )
    fs_xlabel = _fs_axis_label(atd.FONT_LABEL, label_scale=label_scale)
    fs_ylabel = _fs_axis_label(atd.FONT_LABEL, label_scale=y_label_scale)
    xlabelpad = atd.FIG_AXIS_LABELPAD * label_scale
    ylabelpad = atd.FIG_AXIS_LABELPAD * y_label_scale
    ax.set_ylim(atd.ACCURACY_YMIN, min(atd.FIG2_BRACKET_YLIM_CAP, ylim_top))
    ax.set_yticks(atd.ACCURACY_YTICKS)
    ax.grid(False)
    ax.tick_params(axis="both", which="both", length=0, labelsize=fs_tick)
    if show_ylabel:
        ax.set_ylabel(
            YLABEL_ACCURACY, fontsize=fs_ylabel, labelpad=ylabelpad,
        )
    if show_xlabel:
        ax.set_xlabel(XLABEL_FORCE_PAIR, fontsize=fs_xlabel, labelpad=xlabelpad)
    sns.despine(ax=ax)
    atd.apply_accuracy_y_spine_bounds(ax)
    atd.add_inward_tick_guides(ax, n_x)
    atd.apply_accuracy_y_spine_bounds(ax)
    # ATD add_inward_tick_guides() resets labelsize to FONT_TICK (16 pt) — undo that
    ax.tick_params(axis="x", which="both", length=0, labelsize=fs_tick)
    ax.tick_params(axis="y", which="both", length=0, labelsize=fs_y_tick)
    for lbl in ax.get_xticklabels():
        lbl.set_fontsize(fs_tick)
    for lbl in ax.get_yticklabels():
        lbl.set_fontsize(fs_y_tick)
        if not show_yticklabels:
            lbl.set_visible(False)
    if not show_yticklabels:
        ax.tick_params(axis="y", labelleft=False)
    _draw_criterion_line(ax, atd)


def _scatter_marker_area(*, row, panel):
    """Matplotlib scatter ``s`` (area) for GEE / SD box-strip panels."""
    row_scale = SCATTER_ROW_SCALE_GEE[row] if panel == "gee" else SCATTER_ROW_SCALE_SD[row]
    radius = SCATTER_RADIUS * DOT_SCALE * row_scale
    return radius ** 2


def _gee_band_ylim_top(atd, subj_acc, band_label, order):
    """Match GEE panel ylim_top (data units)."""
    sub = subj_acc[subj_acc["band"] == band_label]
    band_max_pct = 0.0
    for pair in order:
        pdata = sub.loc[sub["pair_label"] == pair, "accuracy"].values * 100.0
        if len(pdata) == 0:
            continue
        band_max_pct = max(band_max_pct, float(np.max(pdata)))
    return min(
        atd.FIG2_BRACKET_YLIM_CAP,
        max(atd.ACCURACY_YLIM_TOP, band_max_pct + 8.0),
    )


def _sd_band_ylim_top(atd, sd_spec):
    subj_acc = sd_spec["subj_acc"]
    band_max = 0.0
    for pair in sd_spec["pair_order"]:
        vals = subj_acc.loc[subj_acc["pair_label"] == pair, "accuracy"].values.astype(float) * 100
        if len(vals) == 0:
            continue
        band_max = max(band_max, float(np.nanmax(vals)))
    return min(
        atd.FIG2_BRACKET_YLIM_CAP,
        max(atd.ACCURACY_YLIM_TOP, band_max + 8.0),
    )


def _band_shared_ylim_tops(atd, stats_mod, sd_specs):
    """Per-band ylim shared across GEE / SD / On-nail in the same row."""
    tops = {}
    for band_label, order in [("Low", stats_mod.low_order), ("High", stats_mod.high_order)]:
        top = _gee_band_ylim_top(atd, stats_mod.subj_acc, band_label, order)
        if band_label in sd_specs:
            top = max(top, _sd_band_ylim_top(atd, sd_specs[band_label]))
        tops[band_label] = top
    return tops


def _draw_gee_band(
    ax, atd, subj_acc, band_label, order, *,
    show_ylabel, show_xlabel, box_w, row, ylim_top,
    x_tick_scale=COL12_TICK_FONT_SCALE, show_yticklabels=True,
    show_x_pair_labels=True,
):
    box_stroke = "#000000"
    box_fill = _box_fill(atd)
    scatter_rgba = _scatter_rgba(atd)
    strip_jitter_ref = 0.14
    box_width_ref_n = 5
    marker_area = _scatter_marker_area(row=row, panel="gee")
    lw = atd.BOX_LINEWIDTH * LINE_SCALE
    med_lw = 2.0 * LINE_SCALE

    sub = subj_acc[subj_acc["band"] == band_label].copy()
    band_max_pct = 0.0
    ref_bw = 0.42 * len(order) / box_width_ref_n
    strip_jitter = strip_jitter_ref * box_w / ref_bw

    for xi, pair in enumerate(order):
        pdata_pct = sub.loc[sub["pair_label"] == pair, "accuracy"].values * 100.0
        if len(pdata_pct) == 0:
            continue
        band_max_pct = max(band_max_pct, float(np.max(pdata_pct)))

        bp = ax.boxplot(
            [pdata_pct], positions=[xi], widths=box_w,
            patch_artist=True, showfliers=False,
            capwidths=atd.CAP_WIDTH,
            whiskerprops={"linewidth": lw, "color": box_stroke, "solid_capstyle": "butt"},
            capprops={"linewidth": lw, "color": box_stroke, "solid_capstyle": "butt"},
            medianprops={"color": atd.ACCENT_RED, "linewidth": med_lw},
            boxprops={"linewidth": lw, "edgecolor": box_stroke},
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(box_fill)
            patch.set_edgecolor(box_stroke)
            patch.set_linewidth(lw)
            patch.set_clip_on(False)
        for key in ("whiskers", "caps", "medians"):
            for line in bp[key]:
                line.set_color(box_stroke if key != "medians" else atd.ACCENT_RED)
                line.set_linewidth(lw if key != "medians" else med_lw)
                line.set_clip_on(False)

        x_strip = np.full(len(pdata_pct), xi) + _jitter(len(pdata_pct), width=strip_jitter)
        ax.scatter(
            x_strip, pdata_pct,
            c=[scatter_rgba] * len(pdata_pct),
            s=marker_area,
            linewidths=0, edgecolors="none", alpha=atd.STRIP_ALPHA,
            zorder=5, clip_on=False,
        )

    ax.set_xticks(range(len(order)))
    if show_x_pair_labels:
        ax.set_xticklabels(order, fontsize=_fs_tick(atd.FONT_TICK, tick_scale=x_tick_scale))
    else:
        ax.set_xticklabels([])
    ax.set_xlim(-0.55, len(order) - 0.45)
    _finalize_accuracy_axes(
        ax, atd, len(order), ylim_top,
        show_ylabel=show_ylabel, show_xlabel=show_xlabel,
        show_yticklabels=show_yticklabels,
        tick_scale=x_tick_scale, label_scale=COL12_LABEL_FONT_SCALE,
        y_tick_scale=COL12_Y_TICK_FONT_SCALE,
    )


def _draw_sd_band(
    ax, atd, spec, *,
    show_ylabel, show_xlabel, box_w, row, ylim_top,
    x_tick_scale=COL12_TICK_FONT_SCALE, show_yticklabels=True,
    show_x_pair_labels=True,
):
    bl = spec["band_label"]
    pair_order = spec["pair_order"]
    box_fill = _box_fill(atd)
    scatter_rgba = _scatter_rgba(atd)
    jitter_w = 0.12 * box_w / POOLED_BOX_REF
    lw = atd.BOX_LINEWIDTH * LINE_SCALE
    med_lw = 2.0 * LINE_SCALE
    marker_area = _scatter_marker_area(row=row, panel="sd")

    subj_acc = spec["subj_acc"]
    band_max = 0.0
    for xi, pair in enumerate(pair_order):
        vals = (
            subj_acc.loc[subj_acc["pair_label"] == pair, "accuracy"].values.astype(float) * 100
        )
        if len(vals) == 0:
            continue
        band_max = max(band_max, float(np.nanmax(vals)))
        bp = ax.boxplot(
            [vals], positions=[xi], widths=box_w,
            patch_artist=True, showfliers=False, capwidths=atd.CAP_WIDTH,
            whiskerprops={"linewidth": lw, "color": "#000000"},
            capprops={"linewidth": lw, "color": "#000000"},
            medianprops={"color": atd.ACCENT_RED, "linewidth": med_lw},
            boxprops={"linewidth": lw, "edgecolor": "#000000"},
        )
        bp["boxes"][0].set_facecolor(box_fill)
        bp["boxes"][0].set_edgecolor("#000000")
        bp["boxes"][0].set_clip_on(False)
        jx = xi + _jitter(len(vals), width=jitter_w)
        ax.scatter(
            jx, vals, c=[scatter_rgba] * len(vals), s=marker_area,
            zorder=3, linewidths=0, edgecolors="none", marker="o",
            alpha=atd.STRIP_ALPHA,
        )

    ax.set_xticks(range(len(pair_order)))
    if show_x_pair_labels:
        ax.set_xticklabels(
            pair_order, fontsize=_fs_tick(atd.FONT_TICK, tick_scale=x_tick_scale),
        )
    else:
        ax.set_xticklabels([])
    ax.set_xlim(-0.55, len(pair_order) - 0.45)
    _finalize_accuracy_axes(
        ax, atd, len(pair_order), ylim_top,
        show_ylabel=show_ylabel, show_xlabel=show_xlabel,
        show_yticklabels=show_yticklabels,
        tick_scale=x_tick_scale, label_scale=COL12_LABEL_FONT_SCALE,
        y_tick_scale=COL12_Y_TICK_FONT_SCALE,
    )


def _combined_onnail_inward_ticks(ax, atd, x_positions, y_ticks):
    frac_x = getattr(atd, "TICK_LEN_AXES", 0.016)
    frac_y = atd.y_tick_frac_match_x(ax, frac_x)
    ax.tick_params(axis="both", which="both", length=0)
    x_tr = blended_transform_factory(ax.transData, ax.transAxes)
    y_tr = blended_transform_factory(ax.transAxes, ax.transData)
    kw = dict(color=atd.BLACK, linewidth=1.0, solid_capstyle="butt", clip_on=False, zorder=6)
    for xi in x_positions:
        ax.plot([xi, xi], [0, frac_x], transform=x_tr, **kw)
    y_lo, y_hi = ax.get_ylim()
    for y in y_ticks:
        if y_lo - 1e-9 <= y <= y_hi + 1e-9:
            ax.plot([0, frac_y], [y, y], transform=y_tr, **kw)


def _onnail_pair_centers(pair_order, xlim_n_pairs):
    """Spread pair ticks across the shared xlim when fewer pairs than the ref span."""
    n = len(pair_order)
    if xlim_n_pairs is None or n <= 1 or n >= xlim_n_pairs:
        return [float(i) for i in range(n)]
    span = max(xlim_n_pairs - 1, 1)
    return [i * span / (n - 1) for i in range(n)]


def _style_on_touch_hatch(patch, *, hatch_rgba, edgecolor, linewidth):
    """Diagonal hatch in light on-touch blue; box outline stays black."""
    patch.set_hatch(ONNAIL_ON_TOUCH_HATCH)
    patch.set_edgecolor(edgecolor)
    patch.set_linewidth(linewidth)
    # mpl 3.9: set_edgecolor copies edge into _hatch_color — override after.
    patch._hatch_color = hatch_rgba


def _low_row_header_fontsize(atd):
    return _fs_tick(atd.FONT_TICK, tick_scale=FONT_SCALE)


def _low_row_header_anchor(cell_rect, *, height_px, width_px, left_tick_reserve_px=0):
    """Shared y (and column center x) for row-1 headers — (1,1)(2,1) titles + (3,1) legend."""
    plot_rect = _plot_rect(
        cell_rect,
        height_px=height_px,
        width_px=width_px,
        left_tick_reserve_px=left_tick_reserve_px,
    )
    cx = plot_rect[0] + plot_rect[2] / 2.0
    y = plot_rect[1] + plot_rect[3] + LOW_ROW_HEADER_GAP_PX / height_px
    return cx, y


def _add_low_row_column_title(
    fig, atd, *, cell_rect, height_px, width_px, title, left_tick_reserve_px=0,
):
    cx, y = _low_row_header_anchor(
        cell_rect,
        height_px=height_px,
        width_px=width_px,
        left_tick_reserve_px=left_tick_reserve_px,
    )
    y += COL_TITLE_LIFT_PX / height_px
    fig.text(
        cx, y, title,
        ha="center", va="bottom",
        fontsize=_low_row_header_fontsize(atd),
        transform=fig.transFigure,
    )


def _onnail_legend_handles(atd):
    """On-nail (hatched) vs Off-nail — matches (3,1) panel styling."""
    lw = atd.BOX_LINEWIDTH * LINE_SCALE
    hatch_rgba = on_touch_hatch_rgba(atd)
    on_patch = mpatches.Patch(
        facecolor=on_touch_box_color(atd),
        edgecolor=atd.BLACK,
        linewidth=lw,
        label=ONNAIL_LEGEND_LABELS[0],
    )
    _style_on_touch_hatch(
        on_patch,
        hatch_rgba=hatch_rgba,
        edgecolor=atd.BLACK,
        linewidth=lw,
    )
    off_patch = mpatches.Patch(
        facecolor=off_nail_box_color(atd),
        edgecolor=atd.BLACK,
        linewidth=lw,
        label=ONNAIL_LEGEND_LABELS[1],
    )
    return [on_patch, off_patch]


def _add_onnail_panel_legend(
    fig, atd, *, cell_rect, height_px, width_px, left_tick_reserve_px=0,
):
    """Legend centered above composite panel (3,1) — Low band, On-nail column."""
    cx, y = _low_row_header_anchor(
        cell_rect,
        height_px=height_px,
        width_px=width_px,
        left_tick_reserve_px=left_tick_reserve_px,
    )
    fig.legend(
        handles=_onnail_legend_handles(atd),
        loc="lower center",
        bbox_to_anchor=(cx, y),
        bbox_transform=fig.transFigure,
        ncol=2,
        fontsize=_low_row_header_fontsize(atd),
        frameon=False,
        columnspacing=1.6,
        handletextpad=0.45,
        handlelength=1.4,
        borderaxespad=0.0,
    )


def _add_low_row_headers(fig, atd, spec, cells, *, height_px, width_px, include_onnail_legend):
    """Row-1 column headers: (1,1) 2AFC, (2,1) SAME/DIFFERENT, (3,1) On/Off legend."""
    band_col = 0
    gee_cell = cells[_cell_coords(spec, band_col=band_col, plot_kind="gee")]
    gee_reserve = _left_axis_reserve_px(
        layout="horizontal", band_col=band_col, plot_kind="gee",
        spec=spec, show_ylabel=False,
    )
    _add_low_row_column_title(
        fig, atd,
        cell_rect=gee_cell,
        height_px=height_px,
        width_px=width_px,
        title=COL1_ROW1_TITLE,
        left_tick_reserve_px=gee_reserve,
    )

    sd_cell = cells[_cell_coords(spec, band_col=band_col, plot_kind="sd")]
    sd_reserve = _left_axis_reserve_px(
        layout="horizontal", band_col=band_col, plot_kind="sd",
        spec=spec, show_ylabel=False,
    )
    _add_low_row_column_title(
        fig, atd,
        cell_rect=sd_cell,
        height_px=height_px,
        width_px=width_px,
        title=COL2_ROW1_TITLE,
        left_tick_reserve_px=sd_reserve,
    )

    if include_onnail_legend:
        on_cell = cells[_cell_coords(spec, band_col=band_col, plot_kind="onnail")]
        on_reserve = _left_axis_reserve_px(
            layout="horizontal", band_col=band_col, plot_kind="onnail",
            spec=spec, show_ylabel=False,
        )
        _add_onnail_panel_legend(
            fig, atd,
            cell_rect=on_cell,
            height_px=height_px,
            width_px=width_px,
            left_tick_reserve_px=on_reserve,
        )


def _onnail_pool_style(atd, grp, row):
    """On-nail uses on-touch blue; Off-nail lighter blue in (3,1)/(3,2)."""
    if grp == "Off-nail":
        return off_nail_box_color(atd), off_nail_scatter_rgba(atd)
    return _box_fill(atd), _scatter_rgba(atd)


def _draw_onnail_band(
    ax, atd, spec, *,
    show_ylabel, show_xlabel, box_w, ylim_top, row, dodge_offset, xlim_n_pairs=None,
    x_tick_scale=Y_TICK_FONT_SCALE, show_yticklabels=True,
    show_x_pair_labels=True,
):
    """All force pairs in one axes — On-nail vs Off-nail grouped per pair."""
    pair_order = spec["pair_order"]
    subj_acc_reg = spec["subj_acc_reg"]
    pool_order = ["On-nail", "Off-nail"]
    jitter_span = 0.12 * box_w / POOLED_BOX_REF
    lw = atd.BOX_LINEWIDTH * LINE_SCALE
    cap_lw = atd.CAP_LINEWIDTH * LINE_SCALE
    med_lw = 2.0 * LINE_SCALE
    marker_area = _scatter_marker_area(row=row, panel="sd")
    y_ticks = list(atd.ACCURACY_YTICKS)
    rng = np.random.default_rng(42)

    pair_centers = _onnail_pair_centers(pair_order, xlim_n_pairs)
    tick_pos = []
    tick_labels = []
    box_x = []

    for pi, pair in enumerate(pair_order):
        center = pair_centers[pi]
        tick_pos.append(center)
        tick_labels.append(pair)
        for gi, grp in enumerate(pool_order):
            x_pos = center + (dodge_offset if gi == 1 else -dodge_offset)
            box_x.append(x_pos)
            rows = subj_acc_reg[
                (subj_acc_reg["pair_label"] == pair)
                & (subj_acc_reg["region_group"] == grp)
            ]
            vals = rows["accuracy"].values * 100
            if len(vals) == 0:
                continue
            box_fill, scatter_rgba = _onnail_pool_style(atd, grp, row)
            hatch_rgba = on_touch_hatch_rgba(atd) if grp == "On-nail" else None
            bp = ax.boxplot(
                [vals], positions=[x_pos], widths=box_w,
                patch_artist=True, showfliers=False, zorder=2,
                whiskerprops=dict(color=atd.BLACK, linewidth=lw),
                capprops=dict(color=atd.BLACK, linewidth=cap_lw),
                medianprops=dict(color=atd.ACCENT_RED, linewidth=med_lw),
                boxprops=dict(facecolor=box_fill, edgecolor=atd.BLACK, linewidth=lw),
            )
            for patch in bp["boxes"]:
                patch.set_clip_on(False)
                if grp == "On-nail":
                    _style_on_touch_hatch(
                        patch,
                        hatch_rgba=hatch_rgba,
                        edgecolor=atd.BLACK,
                        linewidth=lw,
                    )
            jitter = rng.uniform(-jitter_span, jitter_span, size=len(vals))
            ax.scatter(
                x_pos + jitter, vals,
                c=[scatter_rgba] * len(vals), s=marker_area,
                marker="o", linewidths=0, zorder=3, clip_on=False,
                alpha=atd.STRIP_ALPHA,
            )

    ax.set_xticks(tick_pos)
    if show_x_pair_labels:
        ax.set_xticklabels(tick_labels, fontsize=_fs_tick(atd.FONT_TICK, tick_scale=x_tick_scale))
    else:
        ax.set_xticklabels([])
    ax.set_ylim(atd.ACCURACY_YMIN, ylim_top)
    ax.set_yticks(y_ticks)
    ax.yaxis.set_major_locator(FixedLocator(y_ticks))
    fs_x_tick = _fs_tick(atd.FONT_TICK, tick_scale=x_tick_scale)
    fs_y_tick = _fs_tick(atd.FONT_TICK, tick_scale=Y_TICK_FONT_SCALE)
    ax.tick_params(axis="y", labelsize=fs_y_tick, labelleft=show_yticklabels)
    ax.tick_params(axis="x", labelsize=fs_x_tick)
    ax.tick_params(axis="x", length=0)
    for lbl in ax.get_xticklabels():
        lbl.set_fontsize(fs_x_tick)
    for lbl in ax.get_yticklabels():
        lbl.set_fontsize(fs_y_tick)
        if not show_yticklabels:
            lbl.set_visible(False)
    sns.despine(ax=ax)
    atd.apply_accuracy_y_spine_bounds(ax)

    ax.set_xlim(-0.55, max((xlim_n_pairs if xlim_n_pairs is not None else len(pair_order)) - 1, 0) + 0.45)

    if show_ylabel:
        ax.set_ylabel(
            YLABEL_ACCURACY,
            fontsize=_fs_axis_label(atd.FONT_LABEL, label_scale=Y_LABEL_FONT_SCALE),
            labelpad=atd.FIG_AXIS_LABELPAD * Y_LABEL_FONT_SCALE,
        )
    if show_xlabel:
        ax.set_xlabel(
            XLABEL_FORCE_PAIR, fontsize=_fs_axis_label(atd.FONT_LABEL),
            labelpad=atd.FIG_AXIS_LABELPAD * AXIS_LABEL_FONT_SCALE,
        )

    ax.figure.canvas.draw()
    _combined_onnail_inward_ticks(ax, atd, tick_pos, y_ticks)
    _draw_criterion_line(ax, atd)


# ── Data loading ─────────────────────────────────────────────────────────────
def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _collect_sd_specs(sd_mod):
    specs = {}
    for band_label, cfg in sd_mod.BAND_CONFIG.items():
        df_band = sd_mod.df[sd_mod.df["band"] == band_label].copy()
        if df_band.empty:
            continue
        pair_order = sd_mod.fix_order(
            cfg["pair_order"],
            df_band["pair_label"].unique().tolist(),
        )
        specs[band_label] = sd_mod.build_band_spec(
            df_band, band_label, pair_order, cfg["title_ref"],
        )
    return specs


def _save_composite_figure(fig, out_path, *, width_px, height_px):
    w_in, _ = fig.get_size_inches()
    dpi = width_px / w_in
    fig.savefig(str(out_path), dpi=dpi, facecolor="white", edgecolor="none", pad_inches=0)
    plt.close(fig)

    from PIL import Image

    img = Image.open(out_path).convert("RGB")
    tw, th = width_px, height_px
    if img.size != (tw, th):
        if abs(img.size[0] - tw) <= 1 and abs(img.size[1] - th) <= 1:
            canvas = Image.new("RGB", (tw, th), "white")
            canvas.paste(img, ((tw - img.size[0]) // 2, (th - img.size[1]) // 2))
            canvas.save(out_path)
        else:
            raise RuntimeError(f"Size mismatch: {img.size} vs ({tw}, {th})")
    print(f"Saved → {out_path}  ({tw}×{th} px)")


# ── Main render ──────────────────────────────────────────────────────────────
def render_composite(stats_mod=None, sd_mod=None, sd_specs=None, *, layout="horizontal"):
    if stats_mod is None:
        stats_mod = _load_module("stats_gee", SCRIPT_DIR / "Stats(GEE).py")
    if sd_mod is None:
        sd_mod = _load_module("samediff_gee", SCRIPT_DIR / "SameDiffGee.py")
    if sd_specs is None:
        sd_specs = _collect_sd_specs(sd_mod)

    spec = _layout_spec(layout)
    atd = _load_atd_c1()
    sns.set_theme(style="white")
    atd.apply_plot_style()

    grid_kw = _grid_kw(
        width_px=spec["width_px"],
        height_px=spec["height_px"],
        margin_top_px=spec.get("margin_top_px", MARGIN_TOP_PX),
    )
    print(
        f"Composite [{layout}] — font: {FONT_SCALE:.2f}, dots: {DOT_SCALE:.2f}"
    )

    fig = plt.figure(figsize=spec["canvas"], facecolor="#FFFFFF")
    cells = _panel_rects(
        nrows=spec["nrows"],
        ncols=spec["ncols"],
        col_fracs=spec["col_fracs"],
        row_fracs=spec["row_fracs"],
        **grid_kw,
    )

    # Per-band shared ylim across GEE / SD / On-nail in the same row
    band_ylim_tops = _band_shared_ylim_tops(atd, stats_mod, sd_specs)

    # GEE col-1: (1,1) Low is reference — (1,2) High box px width matches
    gee_ref_n_pairs = len(stats_mod.low_order)
    gee_ref_box_w = _box_width_atd(atd, gee_ref_n_pairs)

    # On-nail (3,1)/(3,2): fixed box px width from (1,1) GEE + shared xlim span
    onnail_box_w = None
    onnail_dodge = None
    low_sd_spec = sd_specs.get("Low")
    if low_sd_spec is not None:
        gee_ref_cell = cells[_cell_coords(spec, band_col=0, plot_kind="gee")]
        onnail_ref_cell = cells[_cell_coords(spec, band_col=0, plot_kind="onnail")]
        gee_plot_w = _plot_width_frac(
            gee_ref_cell,
            height_px=spec["height_px"],
            width_px=spec["width_px"],
            left_tick_reserve_px=_left_axis_reserve_px(
                layout=layout, band_col=0, plot_kind="gee",
                spec=spec, show_ylabel=layout != "horizontal",
            ),
        )
        onnail_plot_w = _plot_width_frac(
            onnail_ref_cell,
            height_px=spec["height_px"],
            width_px=spec["width_px"],
            left_tick_reserve_px=_left_axis_reserve_px(
                layout=layout, band_col=0, plot_kind="onnail",
                spec=spec, show_ylabel=False,
            ),
        )
        onnail_box_w = _onnail_box_width_match_gee_panel(
            gee_ref_box_w, gee_plot_w, onnail_plot_w,
        )
        onnail_dodge = _onnail_dodge_for_box(onnail_box_w)
        sd_ref_n_pairs = len(low_sd_spec["pair_order"])
        sd_ref_box_w = _box_width_atd(atd, sd_ref_n_pairs)
    else:
        sd_ref_n_pairs = None
        sd_ref_box_w = None

    bottom_plot_row = spec["nrows"] - 1

    for band_col, band_label in [(0, "Low"), (1, "High")]:
        if band_label == "Low":
            order = stats_mod.low_order
        else:
            order = stats_mod.high_order

        band_idx = band_col
        show_ylabel = (
            band_col == spec["left_band_col"]
            if layout == "vertical"
            else band_col == 0
        )
        band_ylim = band_ylim_tops[band_label]
        is_bottom_band = band_col == spec["nrows"] - 1
        show_x_pair_labels = _panel_show_x_pair_labels(
            layout=layout, band_col=band_col, nrows=spec["nrows"],
        )

        def _x_tick_scale(plot_row):
            if layout == "horizontal":
                return PAIR_X_TICK_FONT_SCALE
            return PAIR_X_TICK_FONT_SCALE if plot_row == bottom_plot_row else COL12_TICK_FONT_SCALE

        # GEE pairwise
        plot_row = spec["plot_rows"]["gee"] if layout == "vertical" else band_col
        gee_show_ylabel = layout != "horizontal" and (
            band_col == spec["left_band_col"] if layout == "vertical" else band_col == 0
        )
        cell = cells[_cell_coords(spec, band_col=band_col, plot_kind="gee")]
        ax = fig.add_axes(_plot_rect(
            cell,
            height_px=spec["height_px"],
            width_px=spec["width_px"],
            left_tick_reserve_px=_left_axis_reserve_px(
                layout=layout, band_col=band_col, plot_kind="gee",
                spec=spec, show_ylabel=gee_show_ylabel,
            ),
        ))
        n_pairs = len(order)
        if band_label == "Low":
            gee_box_w = gee_ref_box_w
        else:
            gee_box_w = _onnail_box_width_pixel_match(
                gee_ref_box_w, gee_ref_n_pairs, n_pairs,
            )

        _draw_gee_band(
            ax, atd, stats_mod.subj_acc, band_label, order,
            show_ylabel=gee_show_ylabel,
            show_xlabel=_panel_show_xlabel(
                layout=layout, plot_row=plot_row, band_col=band_col,
                bottom_plot_row=bottom_plot_row, nrows=spec["nrows"],
            ),
            box_w=gee_box_w,
            row=band_idx,
            ylim_top=band_ylim,
            x_tick_scale=_x_tick_scale(plot_row),
            show_x_pair_labels=show_x_pair_labels,
        )

        sd_spec = sd_specs.get(band_label)
        if sd_spec is not None:
            n_sd_pairs = len(sd_spec["pair_order"])
            if band_label == "Low" and sd_ref_box_w is not None:
                sd_box_w = sd_ref_box_w
            elif band_label == "High" and sd_ref_box_w is not None:
                sd_box_w = _onnail_box_width_pixel_match(
                    sd_ref_box_w, sd_ref_n_pairs, n_sd_pairs,
                )
            else:
                sd_box_w = _box_width_atd(atd, n_sd_pairs)

            # SD accuracy
            plot_row = spec["plot_rows"]["sd"] if layout == "vertical" else band_col
            sd_show_ylabel = show_ylabel if layout == "vertical" else False
            cell = cells[_cell_coords(spec, band_col=band_col, plot_kind="sd")]
            ax = fig.add_axes(_plot_rect(
                cell,
                height_px=spec["height_px"],
                width_px=spec["width_px"],
                left_tick_reserve_px=_left_axis_reserve_px(
                    layout=layout, band_col=band_col, plot_kind="sd",
                    spec=spec, show_ylabel=sd_show_ylabel,
                ),
            ))
            _draw_sd_band(
                ax, atd, sd_spec,
                show_ylabel=show_ylabel if layout == "vertical" else False,
                show_xlabel=_panel_show_xlabel(
                    layout=layout, plot_row=plot_row, band_col=band_col,
                    bottom_plot_row=bottom_plot_row, nrows=spec["nrows"],
                ),
                box_w=sd_box_w,
                row=band_idx,
                ylim_top=band_ylim,
                x_tick_scale=_x_tick_scale(plot_row),
                show_yticklabels=False,
                show_x_pair_labels=show_x_pair_labels,
            )

            # On-nail vs Off-nail
            plot_row = spec["plot_rows"]["onnail"] if layout == "vertical" else band_col
            on_show_ylabel = show_ylabel if layout == "vertical" else False
            cell = cells[_cell_coords(spec, band_col=band_col, plot_kind="onnail")]
            ax = fig.add_axes(_plot_rect(
                cell,
                height_px=spec["height_px"],
                width_px=spec["width_px"],
                left_tick_reserve_px=_left_axis_reserve_px(
                    layout=layout, band_col=band_col, plot_kind="onnail",
                    spec=spec, show_ylabel=on_show_ylabel,
                ),
            ))
            n_onnail_pairs = len(sd_spec["pair_order"])
            onnail_plot_w_band = _plot_width_frac(
                cell,
                height_px=spec["height_px"],
                width_px=spec["width_px"],
                left_tick_reserve_px=_left_axis_reserve_px(
                    layout=layout, band_col=band_col, plot_kind="onnail",
                    spec=spec, show_ylabel=on_show_ylabel,
                ),
            )
            if onnail_box_w is not None:
                onnail_box_w_use = onnail_box_w * ONNAIL_BOX_WIDTH_FRAC
                onnail_dodge_use = _onnail_dodge_for_box(onnail_box_w_use)
                onnail_dodge_use += _onnail_dodge_bump_px(
                    onnail_plot_w_band,
                    spec["width_px"],
                    gee_ref_n_pairs,
                    ONNAIL_DODGE_GAP_EXTRA_PX,
                )
            else:
                onnail_box_w_use = _onnail_box_width(sd_box_w) * ONNAIL_BOX_WIDTH_FRAC
                onnail_dodge_use = _onnail_dodge_for_box(onnail_box_w_use)

            _draw_onnail_band(
                ax, atd, sd_spec,
                show_ylabel=show_ylabel if layout == "vertical" else False,
                show_xlabel=_panel_show_xlabel(
                    layout=layout, plot_row=plot_row, band_col=band_col,
                    bottom_plot_row=bottom_plot_row, nrows=spec["nrows"],
                ),
                box_w=onnail_box_w_use,
                ylim_top=band_ylim,
                row=band_idx,
                dodge_offset=onnail_dodge_use,
                xlim_n_pairs=gee_ref_n_pairs,
                x_tick_scale=_x_tick_scale(plot_row),
                show_yticklabels=False,
                show_x_pair_labels=show_x_pair_labels,
            )

            if layout == "horizontal" and band_col == 0:
                _add_low_row_headers(
                    fig, atd, spec, cells,
                    height_px=spec["height_px"],
                    width_px=spec["width_px"],
                    include_onnail_legend=True,
                )

    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FINAL_DIR / spec["output_name"]
    _save_composite_figure(
        fig, out_path,
        width_px=spec["width_px"],
        height_px=spec["height_px"],
    )
    return out_path


def publish_individual_figures():
    copies = [
        (OUTPUT_STATS / "gee_pairwise_plot_horizontal_2col.png",
         FINAL_DIR / "gee_pairwise_plot_horizontal_2col(Final).png"),
        (OUTPUT_SD / "sd_accuracy_by_pair_2col_nobracket.png",
         FINAL_DIR / "sd_accuracy_by_pair_2col_nobracket(Final).png"),
        (OUTPUT_SD / "sd_onnail_vs_offnail_low.png",
         FINAL_DIR / "sd_onnail_vs_offnail_low(final).png"),
        (OUTPUT_SD / "sd_onnail_vs_offnail_high.png",
         FINAL_DIR / "sd_onnail_vs_offnail_high(final).png"),
    ]
    for src, dst in copies:
        if src.is_file():
            shutil.copy2(src, dst)
            print(f"Published → {dst}")


def main():
    print("=" * 60)
    print("FD composite (fd_composite_figure.py)")
    print("=" * 60)
    stats_mod = _load_module("stats_gee", SCRIPT_DIR / "Stats(GEE).py")
    sd_mod = _load_module("samediff_gee", SCRIPT_DIR / "SameDiffGee.py")
    sd_specs = _collect_sd_specs(sd_mod)
    render_composite(stats_mod=stats_mod, sd_mod=sd_mod, sd_specs=sd_specs, layout="horizontal")
    render_composite(stats_mod=stats_mod, sd_mod=sd_mod, sd_specs=sd_specs, layout="vertical")
    publish_individual_figures()
    print("Done.")


if __name__ == "__main__":
    main()
