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

# Shared Low|High pairwise format — Stats(GEE) + SameDiff must stay identical
PAIRWISE_PANEL = dict(
    left=0.15, right=0.02, bottom=0.08, top=0.94, gap_frac=0.08,
)
# Wide short canvas (2640×1072): each panel x-axis = AXIS_W_2640_PX
EXPORT_W_2640 = 2640
EXPORT_H_2640 = 1072
AXIS_W_2640_PX = 1124


def pairwise_panel_2640(
    *,
    axis_w_px=AXIS_W_2640_PX,
    canvas_w=EXPORT_W_2640,
    left=0.095,
    right=0.018,
    bottom=0.08,   # xticklabels only — flush to canvas edges
    top=1.0,
):
    """Layout so each panel axes width is ``axis_w_px`` on a ``canvas_w`` export."""
    panel = float(axis_w_px) / float(canvas_w)
    gap = 1.0 - left - right - 2.0 * panel
    if gap < 0.01:
        raise ValueError(
            f"axis_w_px={axis_w_px} too wide for canvas {canvas_w} "
            f"with left={left}, right={right} (gap={gap:.4f})"
        )
    return dict(left=left, right=right, bottom=bottom, top=top, gap_frac=gap)


PAIRWISE_PANEL_2640 = pairwise_panel_2640()
PAIRWISE_FONT_TICK = 20       # y-axis numbers
PAIRWISE_FONT_XTICK = 14      # force-pair x labels
PAIRWISE_FONT_LABEL = 20
PAIRWISE_BOX_WIDTH_AT_REF = 0.55
PAIRWISE_X_SPACING = 1.35     # data units between adjacent x ticks
PAIRWISE_XLIM_PAD = 0.55      # pad beyond first/last tick


def pairwise_x_positions(n_pairs):
    """X centers for n force-pair boxes."""
    return [i * PAIRWISE_X_SPACING for i in range(int(n_pairs))]


def pairwise_xlim(n_pairs):
    """X limits with equal pad beyond first/last tick."""
    n = int(n_pairs)
    return (
        -PAIRWISE_XLIM_PAD,
        (n - 1) * PAIRWISE_X_SPACING + PAIRWISE_XLIM_PAD,
    )


def pairwise_box_width(n_pairs, ref_n):
    """Same *pixel* box width in Low/High panels (scales with axis span)."""
    n = int(n_pairs)
    r = int(ref_n)
    span = (n - 1) * PAIRWISE_X_SPACING + 2.0 * PAIRWISE_XLIM_PAD
    span_ref = (r - 1) * PAIRWISE_X_SPACING + 2.0 * PAIRWISE_XLIM_PAD
    return PAIRWISE_BOX_WIDTH_AT_REF * span / span_ref


def add_pairwise_inward_ticks(ax, x_positions, atd):
    """Inward tick guides at custom x positions (ATD helper assumes unit spacing)."""
    from matplotlib.transforms import blended_transform_factory

    x_positions = list(x_positions)
    ax.set_xticks(x_positions)
    ax.grid(False)
    ax.tick_params(axis="both", which="both", length=0, labelsize=atd.FONT_TICK)

    x_trans = blended_transform_factory(ax.transData, ax.transAxes)
    y_trans = blended_transform_factory(ax.transAxes, ax.transData)
    y_lo, y_hi = ax.get_ylim()
    y_vals = [t for t in ax.get_yticks() if y_lo - 1e-9 <= t <= y_hi + 1e-9]
    frac_x = atd.TICK_LEN_AXES
    frac_y = atd.y_tick_frac_match_x(ax, frac_x)

    for xi in x_positions:
        ax.plot(
            [xi, xi], [0, frac_x],
            color=atd.BLACK, linewidth=1.0, solid_capstyle="butt",
            transform=x_trans, clip_on=False, zorder=6,
        )
    for y in y_vals:
        ax.plot(
            [0, frac_y], [y, y],
            color=atd.BLACK, linewidth=1.0, solid_capstyle="butt",
            transform=y_trans, clip_on=False, zorder=6,
        )

# ATD Fig2 On-touch — unified box/scatter base color across FD Final figures
# Optional override: FD_PAIRWISE_BLUE=#RRGGBB (used for Blues sat variants)
ON_TOUCH_BLUE = os.environ.get("FD_PAIRWISE_BLUE", "#10559A")
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


def fit_export_canvas_letterbox(img, target_w, target_h, content_scale=0.82,
                                margin_frac=None, margin_frac_x=None,
                                margin_frac_y=None, equal_px_frac=None,
                                trim_white=False, white_thresh=252,
                                match_canvas_aspect=False,
                                fill_height=False):
    """Uniform scale + white margins (no stretch).

    ``trim_white`` — crop near-white borders to the ink bbox first.
    ``match_canvas_aspect`` — center-crop ink to target W/H (can clip labels).
    ``fill_height`` — scale so content height == target_h (no T/B pad).
    ``margin_frac`` — same fraction of each canvas dimension on all sides.
    """
    from PIL import Image
    import numpy as np

    if trim_white:
        arr = np.asarray(img.convert("RGB"))
        ink = arr.mean(axis=2) < float(white_thresh)
        rows = np.any(ink, axis=1)
        cols = np.any(ink, axis=0)
        if rows.any() and cols.any():
            r0, r1 = int(np.argmax(rows)), int(len(rows) - np.argmax(rows[::-1]))
            c0, c1 = int(np.argmax(cols)), int(len(cols) - np.argmax(cols[::-1]))
            img = img.crop((c0, r0, c1, r1))

    iw, ih = img.size
    if iw == 0 or ih == 0:
        return img

    if match_canvas_aspect and target_w > 0 and target_h > 0:
        target_aspect = target_w / float(target_h)
        cur_aspect = iw / float(ih)
        if cur_aspect > target_aspect:
            new_w = max(1, round(ih * target_aspect))
            left = max(0, (iw - new_w) // 2)
            img = img.crop((left, 0, left + new_w, ih))
        elif cur_aspect < target_aspect:
            new_h = max(1, round(iw / target_aspect))
            top = max(0, (ih - new_h) // 2)
            img = img.crop((0, top, iw, top + new_h))
        iw, ih = img.size

    if fill_height:
        # Fill canvas height exactly; pad only left/right (or crop width).
        scale = target_h / float(ih)
        nw = max(1, round(iw * scale))
        nh = int(target_h)
        scaled = img.resize((nw, nh), Image.Resampling.LANCZOS)
        if nw > target_w:
            left = max(0, (nw - target_w) // 2)
            return scaled.crop((left, 0, left + target_w, nh))
        canvas = Image.new("RGB", (target_w, target_h), "white")
        canvas.paste(scaled, ((target_w - nw) // 2, 0))
        return canvas

    if equal_px_frac is not None:
        m = max(0, round(min(target_w, target_h) * float(equal_px_frac)))
        max_w = max(1, target_w - 2 * m)
        max_h = max(1, target_h - 2 * m)
        scale = min(max_w / iw, max_h / ih)
    elif margin_frac_x is not None or margin_frac_y is not None:
        mfx = float(margin_frac_x if margin_frac_x is not None else margin_frac or 0.12)
        mfy = float(margin_frac_y if margin_frac_y is not None else margin_frac or 0.12)
        max_w = max(1, round(target_w * (1.0 - 2.0 * mfx)))
        max_h = max(1, round(target_h * (1.0 - 2.0 * mfy)))
        scale = min(max_w / iw, max_h / ih)
    elif margin_frac is not None:
        mf = float(margin_frac)
        max_w = max(1, round(target_w * (1.0 - 2.0 * mf)))
        max_h = max(1, round(target_h * (1.0 - 2.0 * mf)))
        scale = min(max_w / iw, max_h / ih)
    else:
        max_w = max(1, round(target_w * float(content_scale)))
        max_h = max(1, round(target_h * float(content_scale)))
        scale = min(max_w / iw, max_h / ih)

    nw = max(1, round(iw * scale))
    nh = max(1, round(ih * scale))
    scaled = img.resize((nw, nh), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (target_w, target_h), "white")
    canvas.paste(scaled, ((target_w - nw) // 2, (target_h - nh) // 2))
    return canvas


def save_export_figure(fig, out_dir, stem, export_widths_px, *, letterbox=False,
                       content_scale=0.82, margin_frac=None,
                       margin_frac_x=None, margin_frac_y=None,
                       equal_px_frac=None, trim_white=False,
                       match_canvas_aspect=False,
                       height_from_content=False,
                       fixed_height_px=None,
                       fill_height=False):
    """Save PNGs at column widths; 2-col uses fixed canvas without stretch.

    ``height_from_content`` — after optional white-trim, set canvas height from
    ink aspect (width fixed).
    ``fixed_height_px`` — force canvas height (overrides default aspect / content).
    ``fill_height`` — letterbox mode: content fills canvas height (no T/B pad).
    """
    from PIL import Image
    import numpy as np

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

        if height_from_content and fixed_height_px is None:
            arr = np.asarray(img)
            ink = arr.mean(axis=2) < 252
            rows = np.any(ink, axis=1)
            cols = np.any(ink, axis=0)
            if rows.any() and cols.any():
                r0 = int(np.argmax(rows))
                r1 = int(len(rows) - np.argmax(rows[::-1]))
                c0 = int(np.argmax(cols))
                c1 = int(len(cols) - np.argmax(cols[::-1]))
                iw, ih = c1 - c0, r1 - r0
                if iw > 0 and ih > 0:
                    height_px = max(1, round(width_px * ih / float(iw)))

        if fixed_height_px is not None:
            height_px = int(fixed_height_px)

        if letterbox or img.size != (width_px, height_px):
            if letterbox:
                img = fit_export_canvas_letterbox(
                    img, width_px, height_px,
                    content_scale=content_scale,
                    margin_frac=margin_frac,
                    margin_frac_x=margin_frac_x,
                    margin_frac_y=margin_frac_y,
                    equal_px_frac=equal_px_frac,
                    trim_white=trim_white,
                    match_canvas_aspect=match_canvas_aspect,
                    fill_height=fill_height,
                )
            else:
                img = fit_export_canvas(img, width_px, height_px)
            img.save(out_path)
        print(f"Saved → {out_path}  ({width_px}×{height_px} px)")
        if tag == "2col":
            path_2col = out_path

    if path_2col:
        legacy = os.path.join(out_dir, f"{stem}.png")
        shutil.copy2(path_2col, legacy)
        print(f"Saved → {legacy}")
