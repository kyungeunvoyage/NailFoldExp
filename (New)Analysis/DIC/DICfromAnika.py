"""
Hold-phase DIC 데이터 박스플롯 + 지터 스캐터 재현 스크립트
================================================================
- 3개 지표(변형 반경 / 침투 깊이 / 최대 압축 변형률)를 subplot으로 나란히 표시
- 박스플롯(중앙값·IQR·수염) 위에 프레임별 데이터를 가로 지터 스캐터로 오버레이
- 스캐터 마커 모양은 Trial(1·2·3)로 구분

두 가지 그림 유형을 지원:
  1) 특정 Region을 여러 Force에 걸쳐 비교  -> plot_by_force()
  2) 특정 Force에서 여러 Region을 비교       -> plot_by_region()
"""

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
    ("Penetration depth (mm)",      "Penetration depth (mm)",      (0, 0.15), [0, 0.03, 0.06, 0.09, 0.12, 0.15]),
    ("Max compressive strain (%)",  "Max compressive strain (%)",  (0, 15),   [0, 3, 6, 9, 12,15]),
]

# Region별 대표 색상
REGION_COLORS = {
    "Region A":  "#A9577E",   # 로즈/플럼
    "Region CD": "#5CA894",   # 틸/씨그린
    "Glabrous":  "#7C6BB0",   # 퍼플/바이올렛
}

# Region CD를 Force별로 그릴 때 쓰는 순차 음영(회록 -> 밝은 틸)
FORCE_SHADES = {
    0.16: "#7E938C",
    0.40: "#8FB4A9",
    0.60: "#78AD9D",
    1.00: "#5FBFA6",
}

# Trial -> 마커 모양
TRIAL_MARKERS = {1: "o", 2: "^", 3: "s"}


def _darken(hex_color, factor=0.6):
    """박스 테두리·중앙값선용으로 색을 어둡게."""
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    r, g, b = (int(c * factor) for c in (r, g, b))
    return f"#{r:02x}{g:02x}{b:02x}"


def _draw_group(ax, values_by_trial, pos, fill_color, width=0.5, jitter=0.13):
    """한 그룹(하나의 x위치)에 박스 + 지터 스캐터를 그린다."""
    all_vals = np.concatenate(list(values_by_trial.values()))
    edge = _darken(fill_color, 0.55)

    # --- 박스플롯 ---
    bp = ax.boxplot(
        all_vals, positions=[pos], widths=width, patch_artist=True,
        showfliers=False, whis=1.5, zorder=1,
    )
    face_rgba = (*mcolors.to_rgb(fill_color), 0.55)  # alpha only on face
    for box in bp["boxes"]:
        box.set_facecolor(face_rgba)
        box.set_edgecolor("#000000")
        box.set_linewidth(1.2)
    for med in bp["medians"]:
        med.set(color="#CC0000", linewidth=1.6)
    for whisk in bp["whiskers"]:
        whisk.set(color="#000000", linewidth=1.0)
    for cap in bp["caps"]:
        cap.set(color="#000000", linewidth=1.0)

    # --- 지터 스캐터 (Trial별 마커) ---
    rng = np.random.default_rng(42)
    for trial, vals in values_by_trial.items():
        x = pos + rng.uniform(-jitter, jitter, size=len(vals))
        ax.scatter(
            x, vals, s=22, marker=TRIAL_MARKERS.get(trial, "o"),
            facecolor=fill_color, edgecolor="none", linewidth=0,
            alpha=0.50, zorder=3,
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


def _finalize_axis(ax, ylabel, ylim, yticks, xlabel, xticklabels, positions):
    tick_fs = label_fs = 20
    ax.set_ylabel(ylabel, fontsize=label_fs)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=label_fs)
    else:
        ax.set_xlabel("")
    # add padding below 0 so the spine starts at 0 with visible gap underneath
    y_range = ylim[1] - ylim[0]
    y_pad   = y_range * 0.035  # tighter than before — reclaim vertical plot area
    ax.set_ylim(ylim[0] - y_pad, ylim[1])
    ax.set_yticks(yticks)
    ax.set_xticks(positions)
    ax.set_xticklabels(xticklabels, fontsize=tick_fs)
    ax.set_yticklabels([f"{t:g}" for t in yticks], fontsize=tick_fs)
    for t in ax.get_xticklabels():
        t.set_linespacing(0.95)
        t.set_multialignment("center")
    ax.set_xlim(positions[0] - XLIM_PAD, positions[-1] + XLIM_PAD)
    ax.tick_params(axis="both", labelsize=tick_fs)
    _despine(ax, ylim=ylim)


# Compact canvas: shorter height, tight vertical margins so axes (boxes) stay large
FIGSIZE = (12.50, 4.80)   # was (12.50, 6.00)
LAYOUT  = dict(left=0.07, right=0.995, top=0.98, bottom=0.13, wspace=0.30)
LAYOUT_LONG_XTICK = dict(left=0.07, right=0.995, top=0.98, bottom=0.14, wspace=0.30)
# fig1 reference: 4 forces, box_width=0.7, xlim pad=0.7 each side
FIG1_N_CATS = 4
FIG1_BOX_WIDTH = 0.7
XLIM_PAD = 0.7


def _box_width_match_fig1(n_cats):
    """Scale box width so pixel size matches fig1 (same panel width)."""
    # xlim span = (n-1) + 2*XLIM_PAD  for positions 1..n
    span_ref = (FIG1_N_CATS - 1) + 2 * XLIM_PAD   # 4.4
    span_cur = (n_cats - 1) + 2 * XLIM_PAD
    return FIG1_BOX_WIDTH * span_cur / span_ref


def plot_by_force(df, region, forces=None, title=None, save=None, box_width=None):
    """지정한 Region을 여러 Force에 걸쳐 비교 (이미지 1 유형)."""
    sub = df[df["Region"] == region]
    if forces is None:
        forces = sorted(sub["Force"].unique())
    positions = list(range(1, len(forces) + 1))
    xticklabels = [f"{f:.2f}" for f in forces]
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
        _finalize_axis(ax, ylabel, ylim, yticks, "", xticklabels, positions)

    fig.subplots_adjust(**LAYOUT)
    if save:
        fig.savefig(save, dpi=200)
    return fig


def plot_by_region(df, force, regions, title=None, save=None, box_width=None,
                   xtick_labels=None, layout=None):
    """지정한 Force에서 여러 Region을 비교 (이미지 2·3 유형)."""
    sub = df[df["Force"] == force]
    positions = list(range(1, len(regions) + 1))
    if box_width is None:
        box_width = _box_width_match_fig1(len(regions))

    _label_map = {"Glabrous": "Volar\nfingerpad", "Region A": "LNF", "Region CD": "PNF"}
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
                        width=box_width, jitter=box_width * 0.25)
        _finalize_axis(ax, ylabel, ylim, yticks, "", xticklabels, positions)

    fig.subplots_adjust(**(layout or LAYOUT))
    if save:
        fig.savefig(save, dpi=200)
    return fig


if __name__ == "__main__":
    import os

    OUT_DIR = "/Users/kyungeunjung/NailFoldExp/(New)Analysis/DIC"
    CSV_PATH = os.path.join(OUT_DIR, "hold_phase_data_Final1.csv")
    df = pd.read_csv(CSV_PATH)

    # 이미지 1: Region CD, 네 가지 Force
    plot_by_force(df, "Region CD", forces=[0.16, 0.40, 0.60, 1.00],
                  title="Region CD",
                  save=os.path.join(OUT_DIR, "fig1_regionCD_by_force.png"))

    # 이미지 2: 0.40 g, Region CD vs Glabrous
    plot_by_region(df, 0.40, ["Region CD", "Glabrous"],
                   title="0.40 g",
                   save=os.path.join(OUT_DIR, "fig2_0p40g_by_region.png"),
                   xtick_labels=["Dorsal\nPNF skin",
                                 "Volar \nfingerpad"],
                   layout=LAYOUT_LONG_XTICK)

    # 이미지 3: 1.00 g, Region A vs Region CD vs Glabrous
    plot_by_region(df, 1.00, ["Region A", "Region CD", "Glabrous"],
                   title="1.00 g",
                   save=os.path.join(OUT_DIR, "fig3_1p00g_by_region.png"))

    # generate resized versions
    from PIL import Image
    base_figs = [
        "fig1_regionCD_by_force.png",
        "fig2_0p40g_by_region.png",
        "fig3_1p00g_by_region.png",
    ]
    for target_h in [800, 900, 1000]:
        for fname in base_figs:
            src = os.path.join(OUT_DIR, fname)
            stem = fname.replace(".png", "")
            dst = os.path.join(OUT_DIR, f"{stem}_{target_h}px.png")
            img = Image.open(src)
            w, h = img.size
            new_w = round(w * target_h / h)
            img.resize((new_w, target_h), Image.Resampling.LANCZOS).save(dst)

    print(f"saved 3 figures to {OUT_DIR}")