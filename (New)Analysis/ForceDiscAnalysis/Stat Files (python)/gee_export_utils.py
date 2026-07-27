"""Shared 2-col export canvas (2102×1298) for Force Disc GEE figures."""

import os
import shutil

import matplotlib.colors as mcolors

EXPORT_WIDTH_2COL = 2102
EXPORT_HEIGHT_2COL = 1298
EXPORT_CANVAS = (
    8.0,
    round(8.0 * EXPORT_HEIGHT_2COL / EXPORT_WIDTH_2COL, 3),
)

# Match ATD onnail_vs_offnail_pooled(final): left=0.08, bottom=0.12, tight wspace
LAYOUT_LEFT = 0.11          # extra room so y-axis title is not clipped
LAYOUT_RIGHT = 0.03
LAYOUT_BOTTOM = 0.12
LAYOUT_TOP = 0.88           # headroom for legend above panels
PANEL_GAP_FRAC = 0.06       # ~wspace 0.18 for two equal panels
LEGEND_ANCHOR_Y = 0.975
XLABEL_FORCE_PAIR = "Stimulus force pair (g)"

# ATD Fig2 On-touch — unified box/scatter base color across FD Final figures
ON_TOUCH_BLUE = "#10559A"
OFF_NAIL_LIGHT_BLUE = "#85B1D9"  # lighter blue for Off-nail in (3,1)

# ATD pooled on-nail mini-panel reference box width (data coords)
POOLED_BOX_REF = 0.45


def export_height_px(width_px):
    return round(width_px * EXPORT_HEIGHT_2COL / EXPORT_WIDTH_2COL)


def horizontal_panel_rects(
    *,
    left=LAYOUT_LEFT,
    right=LAYOUT_RIGHT,
    bottom=LAYOUT_BOTTOM,
    top=LAYOUT_TOP,
    gap_frac=PANEL_GAP_FRAC,
):
    inner_w = 1.0 - left - right
    panel_w = (inner_w - gap_frac) / 2
    panel_h = top - bottom
    y0 = bottom
    low = [left, y0, panel_w, panel_h]
    high = [left + panel_w + gap_frac, y0, panel_w, panel_h]
    return low, high


def vertical_panel_rects(
    *,
    left=LAYOUT_LEFT,
    right=LAYOUT_RIGHT,
    bottom=LAYOUT_BOTTOM,
    top=LAYOUT_TOP,
    gap_frac=0.04,
):
    inner_h = top - bottom
    panel_h = (inner_h - gap_frac) / 2
    panel_w = 1.0 - left - right
    high = [left, bottom, panel_w, panel_h]
    low = [left, bottom + panel_h + gap_frac, panel_w, panel_h]
    return low, high


def single_panel_rect(
    *,
    left=LAYOUT_LEFT,
    right=LAYOUT_RIGHT,
    bottom=LAYOUT_BOTTOM,
    top=0.90,
):
    return [left, bottom, 1.0 - left - right, top - bottom]


def on_touch_box_color(atd_module):
    """Pale box fill matching ATD Fig2 On-touch."""
    return atd_module.pale_box_face(ON_TOUCH_BLUE)


def on_touch_scatter_rgba(atd_module):
    """Scatter rgba matching ATD Fig2 On-touch strip."""
    return atd_module._hsb_scatter_rgba(
        ON_TOUCH_BLUE,
        atd_module.SCATTER_HSB_BRIGHTNESS,
        atd_module.STRIP_ALPHA,
    )


# Softer than scatter (V=0.60), darker than pale box fill (V=0.88 @ low sat)
ON_TOUCH_HATCH_BRIGHTNESS = 0.78
ON_TOUCH_HATCH_SATURATION_SCALE = 0.58


def on_touch_hatch_rgba(atd_module):
    """Light on-touch hatch lines on pale On-nail boxes."""
    hex_c = atd_module.hsb_hex(
        ON_TOUCH_BLUE,
        ON_TOUCH_HATCH_BRIGHTNESS,
        ON_TOUCH_HATCH_SATURATION_SCALE,
    )
    r, g, b = mcolors.to_rgb(hex_c)
    return (r, g, b, 1.0)


def off_nail_box_color(atd_module):
    """Lighter pale box fill for Off-nail (Low band col 3)."""
    return atd_module.pale_box_face(OFF_NAIL_LIGHT_BLUE)


def off_nail_scatter_rgba(atd_module):
    """Lighter scatter for Off-nail (Low band col 3)."""
    return atd_module._hsb_scatter_rgba(
        OFF_NAIL_LIGHT_BLUE,
        atd_module.SCATTER_HSB_BRIGHTNESS,
        atd_module.STRIP_ALPHA,
    )


def add_figure_legend(fig, handles, *, ncol=None, anchor_y=LEGEND_ANCHOR_Y, fontsize=12):
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, anchor_y),
        bbox_transform=fig.transFigure,
        ncol=ncol or len(handles),
        fontsize=fontsize,
        frameon=False,
        columnspacing=2.0,
        handletextpad=0.5,
        handlelength=1.6,
    )


def fit_export_canvas(img, target_w, target_h):
    """Uniform scale + center crop — no anisotropic stretch."""
    from PIL import Image

    iw, ih = img.size
    if iw == 0 or ih == 0:
        return img
    src_aspect = iw / ih
    tgt_aspect = target_w / target_h
    if abs(src_aspect - tgt_aspect) < 0.015:
        return img.resize((target_w, target_h), Image.Resampling.LANCZOS)
    scale = max(target_w / iw, target_h / ih)
    nw = max(1, round(iw * scale))
    nh = max(1, round(ih * scale))
    img = img.resize((nw, nh), Image.Resampling.LANCZOS)
    left = max(0, (nw - target_w) // 2)
    top = max(0, (nh - target_h) // 2)
    return img.crop((left, top, left + target_w, top + target_h))


def save_export_figure(fig, out_dir, stem, export_widths_px):
    """Save PNGs at column widths; 2-col uses fixed canvas without stretch."""
    from PIL import Image

    fig.canvas.draw()
    w_in, _ = fig.get_size_inches()
    path_2col = None

    for tag, width_px in export_widths_px:
        height_px = export_height_px(width_px)
        out_path = os.path.join(out_dir, f"{stem}_{tag}.png")
        dpi = width_px / w_in
        fig.savefig(
            out_path, dpi=dpi, facecolor="white",
            edgecolor="none", pad_inches=0,
        )
        img = Image.open(out_path).convert("RGB")
        if img.size != (width_px, height_px):
            img = fit_export_canvas(img, width_px, height_px)
            img.save(out_path)
        print(f"Saved → {out_path}  ({width_px}×{height_px} px)")
        if tag == "2col":
            path_2col = out_path

    if path_2col:
        legacy = os.path.join(out_dir, f"{stem}.png")
        shutil.copy2(path_2col, legacy)
        print(f"Saved → {legacy}")
