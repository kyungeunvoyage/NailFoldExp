

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

# ----------------------------------------------------------------------
# 스타일 설정
# ----------------------------------------------------------------------
plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size":         11,
    "axes.linewidth":    1,
    "axes.edgecolor":    "#000000",
    "xtick.color":       "#000000",
    "ytick.color":       "#000000",
    "xtick.direction":   "in",
    "ytick.direction":   "in",
    "xtick.major.width": 1,
    "ytick.major.width": 1,
    "axes.grid":         False,
})

METRICS = [
    ("Deformation radius (mm)",     "Deformation radius (mm)",     (0, 2.5),    [0, 0.5, 1.0, 1.5, 2.0, 2.5]),
    ("Penetration depth (mm)",      "Max normal surface\ndisplacement (mm)", (0, 0.15), [0, 0.03, 0.06, 0.09, 0.12, 0.15]),
    ("Max compressive strain (%)",  "Max compressive strain (%)",  (0, 15),   [0, 3, 6, 9, 12,15]),
]

# Region별 대표 색상 (fig1 force blue = proximal; lateral slightly lighter)
# Glabrous (volar) is drawn hollow like ATD Kao — color unused for fill
_NAIL_FOLD_BLUE = mcolors.to_hex(plt.cm.Blues(0.85))      # Proximal (+ fig1)
_LATERAL_BLUE   = mcolors.to_hex(plt.cm.Blues(0.65))      # Lateral — slightly lighter
_VOLAR_SCATTER  = "#C0C0C0"                               # light gray (ATD Kao-like)
REGION_COLORS = {
    "Region A":  _LATERAL_BLUE,    # Lateral nail fold
    "Region CD": _NAIL_FOLD_BLUE,  # Proximal nail fold
    "Glabrous":  _VOLAR_SCATTER,   # Volar finger pad (hollow box + gray scatter)
}

# Region CD × force (fig1): all forces use the same Blues shade as 1.0 g
_FORCE_FILL = _NAIL_FOLD_BLUE
FORCE_SHADES = {
    0.16: _FORCE_FILL,
    0.40: _FORCE_FILL,
    0.60: _FORCE_FILL,
    1.00: _FORCE_FILL,
}

# Trial -> 마커 모양
TRIAL_MARKERS = {1: "o", 2: "^", 3: "s"}


def _darken(hex_color, factor=0.6):
    """박스 테두리·중앙값선용으로 색을 어둡게."""
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    r, g, b = (int(c * factor) for c in (r, g, b))
    return f"#{r:02x}{g:02x}{b:02x}"


def _draw_group(ax, values_by_trial, pos, fill_color, width=0.5, jitter=0.13,
                *, hollow=False):
    """한 그룹(하나의 x위치)에 박스 + 지터 스캐터를 그린다.

    hollow=True → ATD Kao / volar style: outline-only box + light-gray scatter.
    """
    all_vals = np.concatenate(list(values_by_trial.values()))

    # --- 박스플롯 ---
    bp = ax.boxplot(
        all_vals, positions=[pos], widths=width, patch_artist=True,
        showfliers=False, whis=1.5, zorder=1,
    )
    for box in bp["boxes"]:
        if hollow:
            box.set_facecolor("none")
        else:
            box.set_facecolor((*mcolors.to_rgb(fill_color), 0.55))
        box.set_edgecolor("#000000")
        box.set_linewidth(1.2)
    for med in bp["medians"]:
        med.set(color="#CC0000", linewidth=1.6)
    # Allow whiskers/caps to draw slightly past ylim (avoids raising axis limits)
    for whisk in bp["whiskers"]:
        whisk.set(color="#000000", linewidth=1.0, clip_on=False)
    for cap in bp["caps"]:
        cap.set(color="#000000", linewidth=1.0, clip_on=False)

    # --- 지터 스캐터 (Trial별 마커) ---
    scatter_fc = _VOLAR_SCATTER if hollow else fill_color
    scatter_alpha = 0.55 if hollow else 0.50
    rng = np.random.default_rng(42)
    for trial, vals in values_by_trial.items():
        x = pos + rng.uniform(-jitter, jitter, size=len(vals))
        ax.scatter(
            x, vals, s=22, marker=TRIAL_MARKERS.get(trial, "o"),
            facecolor=scatter_fc, edgecolor="none", linewidth=0,
            alpha=scatter_alpha, zorder=3,
        )


def _despine(ax, *, ylim=None):
    """각 subplot: left(y) + bottom(x) 축선만 — top/right 제거."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    if ylim is not None and ylim[0] == 0:
        yticks = [t for t in ax.get_yticks() if ylim[0] - 1e-9 <= t <= ylim[1] + 1e-9]
        if yticks:
            # extend spine down to actual ylim bottom (below 0) so it meets the x-axis
            ax.spines["left"].set_bounds(ax.get_ylim()[0], yticks[-1])
    ax.grid(False)


def _finalize_axis(ax, ylabel, ylim, yticks, xlabel, xticklabels, positions,
                   tick_fs=20, label_fs=20, xtick_fs=None, xlim_pad=None):
    if xtick_fs is None:
        xtick_fs = tick_fs
    if xlim_pad is None:
        xlim_pad = XLIM_PAD
    ax.set_ylabel(ylabel, fontsize=label_fs, labelpad=12)
    ax.yaxis.label.set_linespacing(0.92)
    ax.yaxis.label.set_multialignment("center")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=label_fs)
    else:
        ax.set_xlabel("")
    # pad below 0 only — keep top at ylim so axis scale is unchanged
    y_range = ylim[1] - ylim[0]
    y_pad   = y_range * 0.035
    ax.set_ylim(ylim[0] - y_pad, ylim[1])
    ax.set_yticks(yticks)
    ax.set_xticks(positions)
    ax.set_xticklabels(xticklabels, fontsize=xtick_fs)
    ax.set_yticklabels([f"{t:g}" for t in yticks], fontsize=tick_fs)
    for t in ax.get_xticklabels():
        t.set_linespacing(0.95)
        t.set_multialignment("center")
        t.set_clip_on(False)
    ax.yaxis.label.set_clip_on(False)
    ax.yaxis.label.set_in_layout(True)
    for t in ax.get_yticklabels():
        t.set_clip_on(False)
    ax.set_xlim(positions[0] - xlim_pad, positions[-1] + xlim_pad)
    ax.tick_params(axis="y", labelsize=tick_fs)
    ax.tick_params(axis="x", labelsize=xtick_fs)
    _despine(ax, ylim=ylim)


# Wider canvas + more panel gap for multi-line y-axis titles
FIGSIZE = (14.80, 5.60)
LAYOUT  = dict(left=0.12, right=0.995, top=0.97, bottom=0.22, wspace=0.52)
LAYOUT_REGION = dict(left=0.12, right=0.995, top=0.93, bottom=0.20, wspace=0.52)
LAYOUT_LONG_XTICK = dict(left=0.12, right=0.995, top=0.93, bottom=0.22, wspace=0.52)
SAVE_DPI = 200
# Outer pad so y-axis titles / tick labels are not flush-cropped
SAVE_PAD_INCHES = 0.20
# Resized export heights (px). Edit this list to add/remove sizes, e.g. [800] or [700, 900].
EXPORT_HEIGHTS_PX = [800, 900, 1000]
# Fixed W×H canvas (matches ATD 2-col export). Aspect preserved; white letterbox.
EXPORT_FIXED_PX = (2102, 1298)
# fig1 reference: 4 forces, box_width=0.7, xlim pad=0.7 each side
FIG1_N_CATS = 4
FIG1_BOX_WIDTH = 0.7
XLIM_PAD = 0.7
XLIM_PAD_REGION = 0.55   # moderate side pad for long x labels
# Horizontal gap between category ticks (data units). Larger → ticks farther apart.
X_CAT_GAP = 1.1          # fig1 (force ticks)
X_CAT_GAP_REGION = 1.1  # fig2 / fig3 — just enough to avoid label overlap


def _cat_positions(n, gap=1.0):
    """x positions for n categories with constant gap."""
    return [1.0 + i * gap for i in range(n)]


def _save_fig(fig, path):
    """Crop white canvas to artists (incl. y/x labels), with outer pad."""
    fig.canvas.draw()
    extra = []
    for ax in fig.axes:
        if ax.yaxis.label.get_text():
            extra.append(ax.yaxis.label)
        if ax.xaxis.label.get_text():
            extra.append(ax.xaxis.label)
        extra.extend(ax.get_yticklabels())
        extra.extend(ax.get_xticklabels())
    fig.savefig(
        path, dpi=SAVE_DPI, facecolor="white", edgecolor="none",
        bbox_inches="tight", pad_inches=SAVE_PAD_INCHES,
        bbox_extra_artists=extra,
    )


def _box_width_match_fig1(n_cats, cat_gap=1.0, xlim_pad=None):
    """Scale box width so pixel size matches fig1 (same panel width)."""
    if xlim_pad is None:
        xlim_pad = XLIM_PAD
    # xlim span = (n-1)*gap + 2*xlim_pad
    span_ref = (FIG1_N_CATS - 1) + 2 * XLIM_PAD   # 4.4
    span_cur = (n_cats - 1) * cat_gap + 2 * xlim_pad
    return FIG1_BOX_WIDTH * span_cur / span_ref


def plot_by_force(df, region, forces=None, title=None, save=None, box_width=None):
    """지정한 Region을 여러 Force에 걸쳐 비교 (이미지 1 유형)."""
    sub = df[df["Region"] == region]
    if forces is None:
        forces = sorted(sub["Force"].unique())
    positions = _cat_positions(len(forces), gap=X_CAT_GAP)
    def _fmt_force(f):
        # Keep two decimals only when needed (e.g. 0.16); else 0.4 / 0.6 / 1.0
        return f"{f:.2f}".rstrip("0").rstrip(".") if f != int(f) else f"{f:.1f}"

    xticklabels = [_fmt_force(f) for f in forces]
    if box_width is None:
        box_width = FIG1_BOX_WIDTH

    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE)
    for ax, (col, ylabel, ylim, yticks) in zip(axes, METRICS):
        for pos, f in zip(positions, forces):
            g = sub[sub["Force"] == f]
            vals_by_trial = {t: g[g["Trial"] == t][col].to_numpy()
                             for t in sorted(g["Trial"].unique())}
            _draw_group(ax, vals_by_trial, pos, FORCE_SHADES.get(f, REGION_COLORS[region]),
                        width=box_width, jitter=box_width * 0.25)
        _finalize_axis(
            ax, ylabel, ylim, yticks, "Stimulus Force (g)", xticklabels, positions,
        )

    fig.subplots_adjust(**LAYOUT)
    if save:
        _save_fig(fig, save)
    return fig


def plot_by_region(df, force, regions, title=None, save=None, box_width=None,
                   xtick_labels=None, layout=None):
    """지정한 Force에서 여러 Region을 비교 (이미지 2·3 유형)."""
    sub = df[df["Force"] == force]
    n = len(regions)
    positions = _cat_positions(n, gap=X_CAT_GAP_REGION)
    if box_width is None:
        box_width = _box_width_match_fig1(
            n, cat_gap=X_CAT_GAP_REGION, xlim_pad=XLIM_PAD_REGION,
        )

    _label_map = {"Glabrous": "Volar\nfinger pad", "Region A": "Lateral\nnail fold", "Region CD": "Proximal\nnail fold"}
    if xtick_labels is None:
        xticklabels = [_label_map.get(r, r.replace(" ", "\n")) for r in regions]
    else:
        xticklabels = list(xtick_labels)
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE)
    for ax, (col, ylabel, ylim, yticks) in zip(axes, METRICS):
        for pos, reg in zip(positions, regions):
            g = sub[sub["Region"] == reg]
            vals_by_trial = {t: g[g["Trial"] == t][col].to_numpy()
                             for t in sorted(g["Trial"].unique())}
            _draw_group(ax, vals_by_trial, pos, REGION_COLORS[reg],
                        width=box_width, jitter=box_width * 0.25,
                        hollow=(reg == "Glabrous"))
        _finalize_axis(ax, ylabel, ylim, yticks, "", xticklabels, positions,
                       xtick_fs=16, xlim_pad=XLIM_PAD_REGION)

    fig.subplots_adjust(**(layout or LAYOUT_REGION))
    if save:
        _save_fig(fig, save)
    return fig


def plot_fig5_1p0_plus_0p4(df, save=None):
    """Fig5: per metric — 1.0 g (A / CD / Glabrous) | 0.4 g (CD / Glabrous)."""
    groups = [
        # (force, region, x_label)
        (1.00, "Region A",  "Lateral\nnail fold\n(1.0 g)"),
        (1.00, "Region CD", "Proximal\nnail fold\n(1.0 g)"),
        (1.00, "Glabrous",  "Volar\nfinger pad\n(1.0 g)"),
        (0.40, "Region CD", "Proximal\nnail fold\n(0.4 g)"),
        (0.40, "Glabrous",  "Volar\nfinger pad\n(0.4 g)"),
    ]
    # Gap between 1.0 g block and 0.4 g block
    gap = X_CAT_GAP_REGION
    block_gap = gap * 1.35
    positions = [
        1.0,
        1.0 + gap,
        1.0 + 2 * gap,
        1.0 + 2 * gap + block_gap + gap,          # start of 0.4 g block
        1.0 + 2 * gap + block_gap + 2 * gap,
    ]
    n = len(groups)
    box_width = _box_width_match_fig1(
        n, cat_gap=gap, xlim_pad=XLIM_PAD_REGION,
    )
    # Slightly narrower so 5 cats + gap still read clearly
    box_width *= 0.92

    figsize = (FIGSIZE[0] * 1.28, FIGSIZE[1] + 0.35)
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    xticklabels = [lab for _f, _r, lab in groups]

    for ax, (col, ylabel, ylim, yticks) in zip(axes, METRICS):
        for pos, (force, reg, _lab) in zip(positions, groups):
            g = df[(df["Force"] == force) & (df["Region"] == reg)]
            vals_by_trial = {
                t: g[g["Trial"] == t][col].to_numpy()
                for t in sorted(g["Trial"].unique())
            }
            _draw_group(
                ax, vals_by_trial, pos, REGION_COLORS[reg],
                width=box_width, jitter=box_width * 0.25,
                hollow=(reg == "Glabrous"),
            )
        _finalize_axis(
            ax, ylabel, ylim, yticks, "", xticklabels, positions,
            xtick_fs=12, xlim_pad=XLIM_PAD_REGION,
        )
        # Light divider between 1.0 g and 0.4 g blocks
        mid = 0.5 * (positions[2] + positions[3])
        ax.axvline(mid, color="#B0B0B0", linewidth=0.8, linestyle="--",
                   zorder=0, clip_on=False)

    layout = dict(left=0.10, right=0.995, top=0.93, bottom=0.26, wspace=0.48)
    fig.subplots_adjust(**layout)
    if save:
        _save_fig(fig, save)
    return fig


if __name__ == "__main__":
    import os

    OUT_DIR = "/Users/kyungeunjung/NailFoldExp/(New)Analysis/DIC"
    CSV_PATH = os.path.join(OUT_DIR, "hold_phase_data_Final2.csv")
    df = pd.read_csv(CSV_PATH)

    # 이미지 1: Region CD, 네 가지 Force
    plot_by_force(df, "Region CD", forces=[0.16, 0.40, 0.60, 1.00],
                  title="Region CD",
                  save=os.path.join(OUT_DIR, "fig1_regionCD_by_force.png"))

    # 이미지 2: 0.40 g, Region CD vs Glabrous
    plot_by_region(df, 0.40, ["Region CD", "Glabrous"],
                   title="0.40 g",
                   save=os.path.join(OUT_DIR, "fig2_0p40g_by_region.png"),
                   xtick_labels=["Dorsal\nnail fold",
                                 "Volar \nfinger pad"],
                   layout=LAYOUT_LONG_XTICK)

    # 이미지 3: 1.00 g, Region A vs Region CD vs Glabrous
    plot_by_region(df, 1.00, ["Region A", "Region CD", "Glabrous"],
                   title="1.00 g",
                   save=os.path.join(OUT_DIR, "fig3_1p00g_by_region.png"))

    # 이미지 4: 1.00 g, Region CD vs Glabrous (fig2와 동일 구성, force만 1.0 g)
    plot_by_region(df, 1.00, ["Region CD", "Glabrous"],
                   title="1.00 g",
                   save=os.path.join(OUT_DIR, "fig4_1p00g_CD_vs_glabrous.png"),
                   xtick_labels=["Dorsal\nnail fold",
                                 "Volar \nfinger pad"],
                   layout=LAYOUT_LONG_XTICK)

    # 이미지 5: 1.0 g (A/CD/Glabrous) + 옆에 0.4 g (CD vs Glabrous)
    plot_fig5_1p0_plus_0p4(
        df, save=os.path.join(OUT_DIR, "fig5_1p0_with_0p4_CD_glabrous.png"),
    )

    # generate resized versions
    from PIL import Image
    base_figs = [
        "fig1_regionCD_by_force.png",
        "fig2_0p40g_by_region.png",
        "fig3_1p00g_by_region.png",
        "fig4_1p00g_CD_vs_glabrous.png",
        "fig5_1p0_with_0p4_CD_glabrous.png",
    ]
    for target_h in EXPORT_HEIGHTS_PX:
        for fname in base_figs:
            src = os.path.join(OUT_DIR, fname)
            stem = fname.replace(".png", "")
            dst = os.path.join(OUT_DIR, f"{stem}_{target_h}px.png")
            img = Image.open(src)
            w, h = img.size
            new_w = round(w * target_h / h)
            img.resize((new_w, target_h), Image.Resampling.LANCZOS).save(dst)

    # Fixed W×H canvas: keep aspect ratio, pad with white (centered)
    fw, fh = EXPORT_FIXED_PX
    for fname in base_figs:
        src = os.path.join(OUT_DIR, fname)
        stem = fname.replace(".png", "")
        dst = os.path.join(OUT_DIR, f"{stem}_{fw}x{fh}.png")
        img = Image.open(src).convert("RGBA")
        w, h = img.size
        scale = min(fw / w, fh / h)
        new_w = max(1, round(w * scale))
        new_h = max(1, round(h * scale))
        scaled = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (fw, fh), "white")
        canvas.paste(
            scaled,
            ((fw - new_w) // 2, (fh - new_h) // 2),
            scaled,
        )
        canvas.save(dst)

    print(f"saved {len(base_figs)} figures to {OUT_DIR}")
    print(f"  + fixed {fw}x{fh} px versions")
