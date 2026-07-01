"""
Paradigm comparison — Stronger (P21–P59) vs Same/Different (ID > 73)
======================================================================
Two export versions:
  a) Same/Different: all trials (IsCorrect)
  b) Same/Different: DIFFERENT trials only

Exports four figures per trial definition (a/b), two layouts each:
  overlay   — semi-transparent boxes with slight overlap
  sidebyside — Stronger | Same/Different dodged left/right at each force pair

No significance brackets. Style matched to sd_accuracy_by_pair_2col.png.

Style matched to SameDiffGee sd_accuracy_by_pair_2col.png (2102 px wide, inward ticks).
"""

import io
import os
import re
import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.transforms import blended_transform_factory
import numpy as np
import pandas as pd
from PIL import Image

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = "/Users/kyungeunjung/NailFoldExp/Data/(FD)CurData"
OUTPUT_DIR = (
    "/Users/kyungeunjung/NailFoldExp/(New)Analysis/ForceDiscAnalysis/Output/ParadigmCompare"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

LEGACY_SUBJ_RANGE = (21, 59)
SAMEDIFF_MIN_SUBJ_NUM = 73  # match SameDiffGee.py: subject_number > 73

# ── Style (SameDiffGee sd_accuracy_by_pair_2col) ──────────────────────────────
CHANCE_PCT = 50.0
JND_PCT = 75.0
EXPORT_WIDTH_2COL = 2102
EXPORT_HEIGHT_2COL = 1298
ACC_BY_PAIR_FIGSIZE = (14.0, round(14.0 * EXPORT_HEIGHT_2COL / EXPORT_WIDTH_2COL, 3))
ATD_FIG2_REF_N = 5
ATD_FIG2_DODGED_BOX_WIDTH = 0.275
COMBINED_PANEL_COUNT = 2
STRIP_JITTER_REF = 0.12
CAP_WIDTH = 0.10
BOX_LINEWIDTH = 1.0
FIG_PANEL_TOP_FRAC = 0.80
FIG_LEGEND_ANCHOR_Y = 0.975
BLACK = "#1A1A1A"
TICK_LEN_AXES = 0.016
ACCURACY_YSPINE_TOP = 100.0
FONT_TICK = 16
FONT_LABEL = 14
FONT_LEGEND = 12
FIG_AXIS_LABELPAD = 6
AXIS_SPINE_LW = 2.0
PAPER_FD_FIG_H_IN = 4.5
PAPER_FD_OUT_H_PX = 1137
SAVE_DPI_COMBINED = 600
RED = "#c0392b"

COLOR_STRONGER_FACE = "#2166AC"
COLOR_STRONGER_DOT = "#2166AC"
COLOR_SAMEDIFF_FACE = "#E07B39"
COLOR_SAMEDIFF_DOT = "#C65D1A"
BOX_FACE_ALPHA_OVERLAY = 0.42
PARADIGM_OVERLAP_OFFSET = 0.07
PARADIGM_DODGE_OFFSET = 0.09

BAND_CONFIG = {
    "Low": {
        "pair_order": ["0.4–1", "0.6–1", "1–1.4", "1–2"],
        "title_ref": "1 g",
    },
    "High": {
        "pair_order": ["10–26", "15–26", "26–60"],
        "title_ref": "26 g",
    },
}


def _combined_axis_font_size(base_pt):
    fig_h = ACC_BY_PAIR_FIGSIZE[1]
    return round(base_pt * PAPER_FD_OUT_H_PX / EXPORT_HEIGHT_2COL * fig_h / PAPER_FD_FIG_H_IN)


COMBINED_FONT_TICK = _combined_axis_font_size(FONT_TICK)
COMBINED_FONT_LABEL = _combined_axis_font_size(FONT_LABEL)
COMBINED_FONT_LEGEND = _combined_axis_font_size(FONT_LEGEND)
COMBINED_FONT_MEDIAN = _combined_axis_font_size(10)
MEDIAN_ANNOT_PAD = 4.0
MEDIAN_ANNOT_HEADROOM = 12.0
MEDIAN_TIER_STEP = 11.0


def subject_number(filepath):
    m = re.search(r"P(\d+)", os.path.basename(filepath))
    return int(m.group(1)) if m else 0


def fix_order(order, actual):
    fixed = []
    for p in order:
        if p in actual:
            fixed.append(p)
        else:
            alt = [
                a for a in actual
                if set(a.replace("–", "-").split("-")) == set(p.replace("–", "-").split("-"))
            ]
            fixed.append(alt[0] if alt else p)
    return fixed


def pair_label_from_row(reference, comparison):
    return f"{min(reference, comparison):g}–{max(reference, comparison):g}"


def jitter_x(n, width=0.12, seed=42):
    return (np.random.default_rng(seed).random(n) - 0.5) * width


def apply_accuracy_y_spine_bounds(ax, y_top=ACCURACY_YSPINE_TOP):
    ax.spines["left"].set_bounds(0, y_top)


def apply_combined_axis_spines(ax):
    for spine in ("left", "bottom"):
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color(BLACK)
        ax.spines[spine].set_linewidth(AXIS_SPINE_LW)


def add_inward_tick_guides(ax, n_x=None, y_ticks=None, *, labelsize=None,
                           linewidth=AXIS_SPINE_LW):
    ax.grid(False)
    tick_kw = {"axis": "both", "which": "both", "length": 0}
    if labelsize is not None:
        tick_kw["labelsize"] = labelsize
    ax.tick_params(**tick_kw)
    x_trans = blended_transform_factory(ax.transData, ax.transAxes)
    y_trans = blended_transform_factory(ax.transAxes, ax.transData)
    y_lo, y_hi = ax.get_ylim()
    y_vals = y_ticks if y_ticks is not None else [
        t for t in ax.get_yticks() if y_lo - 1e-9 <= t <= y_hi + 1e-9
    ]
    x_positions = range(n_x) if n_x is not None else ax.get_xticks()
    for xi in x_positions:
        ax.plot(
            [xi, xi], [0, TICK_LEN_AXES],
            color=BLACK, linewidth=linewidth, solid_capstyle="butt",
            transform=x_trans, clip_on=False, zorder=6,
        )
    for y in y_vals:
        ax.plot(
            [0, TICK_LEN_AXES], [y, y],
            color=BLACK, linewidth=linewidth, solid_capstyle="butt",
            transform=y_trans, clip_on=False, zorder=6,
        )


def combined_panel_box_width(n_pairs):
    return ATD_FIG2_DODGED_BOX_WIDTH * n_pairs / ATD_FIG2_REF_N * COMBINED_PANEL_COUNT


def overlay_box_width(n_pairs):
    """Single paradigm box width — wide enough to overlap the paired box."""
    return combined_panel_box_width(n_pairs) * 0.72


def sidebyside_box_width(n_pairs):
    """Half-width dodged box at each force pair (no overlap)."""
    return combined_panel_box_width(n_pairs) / 2 * 0.95


def save_png_at_width(fig, out_path, width_px=EXPORT_WIDTH_2COL, *,
                      height_px=None, pad_inches=0.05, dpi=150):
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


def load_legacy_df():
    pattern = os.path.join(DATA_DIR, "P*_ForceDiscrimination.csv")
    files = sorted(
        f for f in glob.glob(pattern)
        if LEGACY_SUBJ_RANGE[0] <= subject_number(f) <= LEGACY_SUBJ_RANGE[1]
    )
    if not files:
        raise FileNotFoundError(f"No legacy files in P{LEGACY_SUBJ_RANGE[0]}–P{LEGACY_SUBJ_RANGE[1]}")
    df = pd.concat([pd.read_csv(f, encoding="utf-8-sig") for f in files], ignore_index=True)
    df["correct"] = np.where(
        df["Comparison"] > df["Reference"],
        df["ChoseComparison"] == 1,
        df["ChoseComparison"] == 0,
    ).astype(int)
    df["pair_label"] = df.apply(
        lambda r: pair_label_from_row(r["Reference"], r["Comparison"]), axis=1,
    )
    df["band"] = np.where(df["Reference"] == 1, "Low", "High")
    return df


def samediff_cohort_label(n_subj, subjects):
    nums = sorted(int(re.sub(r"\D", "", s)) for s in subjects)
    if not nums:
        return f"Same/Different (ID > {SAMEDIFF_MIN_SUBJ_NUM}, n = {n_subj})"
    lo, hi = nums[0], nums[-1]
    span = f"P{lo}–P{hi}" if lo != hi else f"P{lo}"
    return f"Same/Different ({span}, n = {n_subj})"


def load_samediff_df(*, different_only=False):
    patterns = [
        os.path.join(DATA_DIR, "P*_ForceDiscrimination_SameDiff.csv"),
        os.path.join(DATA_DIR, "P*_ForceDiscrimination_SameDiff_26g.csv"),
    ]
    files = sorted(
        set(
            f for pat in patterns for f in glob.glob(pat)
            if subject_number(f) > SAMEDIFF_MIN_SUBJ_NUM
        )
    )
    if not files:
        raise FileNotFoundError(
            f"No Same/Different files with subject number > {SAMEDIFF_MIN_SUBJ_NUM}"
        )
    df = pd.concat([pd.read_csv(f, encoding="utf-8-sig") for f in files], ignore_index=True)
    if different_only:
        df = df[df["GroundTruth"] == "DIFFERENT"].copy()
    df["correct"] = df["IsCorrect"].astype(int)
    df["pair_label"] = df.apply(
        lambda r: pair_label_from_row(r["Reference"], r["Comparison"]), axis=1,
    )
    df["band"] = df["Reference"].map({1: "Low", 26: "High"})
    return df


def subject_accuracy(df, band_label, pair_order):
    sub = df[(df["band"] == band_label) & (df["pair_label"].isin(pair_order))].copy()
    return (
        sub.groupby(["Subject", "pair_label"])["correct"]
        .mean()
        .reset_index()
        .rename(columns={"correct": "accuracy"})
    )


def _draw_single_box(ax, position, values_pct, *, facecolor, dot_color,
                     box_width, jitter_width, zorder_base, face_alpha=1.0):
    vals = np.asarray(values_pct, dtype=float)
    if len(vals) == 0:
        return
    bp = ax.boxplot(
        [vals], positions=[position], widths=box_width,
        patch_artist=True, showfliers=False, capwidths=CAP_WIDTH,
        whiskerprops={"linewidth": BOX_LINEWIDTH, "color": "black"},
        capprops={"linewidth": BOX_LINEWIDTH, "color": "black"},
        medianprops={"color": RED, "linewidth": 2},
        boxprops={"linewidth": BOX_LINEWIDTH, "edgecolor": "black"},
        zorder=zorder_base,
    )
    bp["boxes"][0].set_facecolor(facecolor)
    bp["boxes"][0].set_alpha(face_alpha)
    bp["boxes"][0].set_edgecolor("black")
    bp["boxes"][0].set_zorder(zorder_base)
    bp["medians"][0].set_zorder(zorder_base + 2)
    jx = position + jitter_x(len(vals), width=jitter_width)
    ax.scatter(
        jx, vals, color=dot_color, alpha=0.65, s=20,
        zorder=zorder_base + 3, edgecolors="none",
    )


def _pair_accuracy_pct(acc_df, pair):
    vals = acc_df.loc[acc_df["pair_label"] == pair, "accuracy"].values
    return np.asarray(vals, dtype=float) * 100


def _format_median_change(s_med, d_med):
    delta = d_med - s_med
    sign = "+" if delta >= 0 else "−"
    return f"{s_med:.0f} → {d_med:.0f}\n({sign}{abs(delta):.1f})"


def _median_text_half_width(n_pairs):
    """Approximate half-width of the two-line label in x data coordinates."""
    return min(0.58, 0.46 * ATD_FIG2_REF_N / max(n_pairs, 1))


def _assign_median_label_tiers(entries, *, n_pairs):
    """Raise y tier until horizontal neighbours no longer overlap."""
    half_w = _median_text_half_width(n_pairs)
    placed = []
    for entry in sorted(entries, key=lambda e: e["xi"]):
        tier = 0
        while True:
            y = entry["base_y"] + tier * MEDIAN_TIER_STEP
            conflict = False
            for prev in placed:
                x_close = abs(entry["xi"] - prev["xi"]) < (2 * half_w + 0.04)
                y_close = abs(y - prev["y"]) < MEDIAN_TIER_STEP * 0.95
                if x_close and y_close:
                    conflict = True
                    break
            if not conflict:
                break
            tier += 1
        entry["y"] = y
        entry["tier"] = tier
        placed.append({"xi": entry["xi"], "y": y})
    return entries


def _annotate_median_changes(ax, pair_order, acc_stronger, acc_samediff):
    """Label each force pair with Stronger → Same/Diff median and delta (pp)."""
    entries = []
    for xi, pair in enumerate(pair_order):
        s_vals = _pair_accuracy_pct(acc_stronger, pair)
        d_vals = _pair_accuracy_pct(acc_samediff, pair)
        if len(s_vals) == 0 or len(d_vals) == 0:
            continue
        s_med = float(np.median(s_vals))
        d_med = float(np.median(d_vals))
        entries.append({
            "xi": xi,
            "base_y": max(s_med, d_med) + MEDIAN_ANNOT_PAD,
            "s_med": s_med,
            "d_med": d_med,
        })

    if not entries:
        return ACCURACY_YSPINE_TOP

    entries = _assign_median_label_tiers(entries, n_pairs=len(pair_order))
    y_top = ACCURACY_YSPINE_TOP
    for entry in entries:
        delta = entry["d_med"] - entry["s_med"]
        delta_color = "#2e7d32" if delta >= 0 else RED
        ax.text(
            entry["xi"], entry["y"], _format_median_change(entry["s_med"], entry["d_med"]),
            ha="center", va="bottom",
            fontsize=COMBINED_FONT_MEDIAN,
            color=delta_color,
            fontweight="bold",
            linespacing=0.9,
            clip_on=False,
            zorder=10,
        )
        y_top = max(y_top, entry["y"] + MEDIAN_ANNOT_HEADROOM + entry["tier"] * 2)
    return y_top


def _apply_panel_axes(ax, pair_order, band_label, *, show_ylabel=True, ylim_top=ACCURACY_YSPINE_TOP):
    ax.axhline(JND_PCT, color="black", ls="--", lw=1.2, alpha=0.8, zorder=1)
    ax.axhline(CHANCE_PCT, color="gray", ls=":", lw=0.9, alpha=0.7, zorder=1)

    y_tick_vals = list(range(0, 101, 20))
    ax.set_xticks(range(len(pair_order)))
    ax.set_xticklabels(pair_order, fontsize=COMBINED_FONT_TICK)
    ax.set_xlim(-0.55, len(pair_order) - 0.45)
    ax.set_ylim(0, ylim_top)
    ax.set_yticks(y_tick_vals)
    ax.tick_params(axis="both", labelsize=COMBINED_FONT_TICK, length=0)
    if show_ylabel:
        ax.set_ylabel(
            "Discrimination Accuracy (%)",
            fontsize=COMBINED_FONT_LABEL,
            labelpad=FIG_AXIS_LABELPAD,
        )
    ax.set_xlabel("Force pair (g)", fontsize=COMBINED_FONT_LABEL, labelpad=FIG_AXIS_LABELPAD)
    ax.set_title(
        f"{band_label} band (ref = {BAND_CONFIG[band_label]['title_ref']})",
        fontsize=COMBINED_FONT_LABEL,
        fontweight="bold",
        pad=10,
    )
    ax.spines[["top", "right"]].set_visible(False)
    apply_combined_axis_spines(ax)
    apply_accuracy_y_spine_bounds(ax)
    add_inward_tick_guides(
        ax, n_x=len(pair_order), y_ticks=y_tick_vals, labelsize=COMBINED_FONT_TICK,
    )
    apply_accuracy_y_spine_bounds(ax)
    apply_combined_axis_spines(ax)


def _draw_paradigm_pair_boxes(ax, pair_order, acc_stronger, acc_samediff, *,
                              x_offset, box_width, jitter_width, face_alpha,
                              annotate_medians=False):
    for xi, pair in enumerate(pair_order):
        s_vals = _pair_accuracy_pct(acc_stronger, pair)
        d_vals = _pair_accuracy_pct(acc_samediff, pair)
        _draw_single_box(
            ax, xi - x_offset, s_vals,
            facecolor=COLOR_STRONGER_FACE, dot_color=COLOR_STRONGER_DOT,
            box_width=box_width, jitter_width=jitter_width, zorder_base=2,
            face_alpha=face_alpha,
        )
        _draw_single_box(
            ax, xi + x_offset, d_vals,
            facecolor=COLOR_SAMEDIFF_FACE, dot_color=COLOR_SAMEDIFF_DOT,
            box_width=box_width, jitter_width=jitter_width, zorder_base=4,
            face_alpha=face_alpha,
        )
    if annotate_medians:
        return _annotate_median_changes(ax, pair_order, acc_stronger, acc_samediff)
    return ACCURACY_YSPINE_TOP


def draw_overlay_panel(ax, pair_order, acc_stronger, acc_samediff, band_label, *,
                       show_ylabel=True):
    box_w = overlay_box_width(len(pair_order))
    jitter_w = STRIP_JITTER_REF * box_w / ATD_FIG2_DODGED_BOX_WIDTH
    ylim_top = _draw_paradigm_pair_boxes(
        ax, pair_order, acc_stronger, acc_samediff,
        x_offset=PARADIGM_OVERLAP_OFFSET, box_width=box_w, jitter_width=jitter_w,
        face_alpha=BOX_FACE_ALPHA_OVERLAY, annotate_medians=True,
    )
    _apply_panel_axes(ax, pair_order, band_label, show_ylabel=show_ylabel, ylim_top=ylim_top)
    return ylim_top


def draw_sidebyside_panel(ax, pair_order, acc_stronger, acc_samediff, band_label, *,
                          show_ylabel=True):
    box_w = sidebyside_box_width(len(pair_order))
    jitter_w = STRIP_JITTER_REF * box_w / ATD_FIG2_DODGED_BOX_WIDTH
    ylim_top = _draw_paradigm_pair_boxes(
        ax, pair_order, acc_stronger, acc_samediff,
        x_offset=PARADIGM_DODGE_OFFSET, box_width=box_w, jitter_width=jitter_w,
        face_alpha=1.0, annotate_medians=True,
    )
    _apply_panel_axes(ax, pair_order, band_label, show_ylabel=show_ylabel, ylim_top=ylim_top)
    return ylim_top


def paradigm_legend_handles(n_stronger, n_samediff, *, samediff_subjects, face_alpha=1.0):
    return [
        mpatches.Patch(
            facecolor=COLOR_STRONGER_FACE, edgecolor="black",
            alpha=face_alpha,
            label=f"Which is stronger (P{LEGACY_SUBJ_RANGE[0]}–P{LEGACY_SUBJ_RANGE[1]}, n = {n_stronger})",
        ),
        mpatches.Patch(
            facecolor=COLOR_SAMEDIFF_FACE, edgecolor="black",
            alpha=face_alpha,
            label=samediff_cohort_label(n_samediff, samediff_subjects),
        ),
    ]


def save_comparison_figure(df_legacy, df_samediff, *, out_name, subtitle, draw_panel):
    band_specs = []
    for band_label, cfg in BAND_CONFIG.items():
        actual = sorted(
            set(df_legacy.loc[df_legacy["band"] == band_label, "pair_label"].unique())
            | set(df_samediff.loc[df_samediff["band"] == band_label, "pair_label"].unique())
        )
        pair_order = fix_order(cfg["pair_order"], actual)
        band_specs.append({
            "band_label": band_label,
            "pair_order": pair_order,
            "acc_stronger": subject_accuracy(df_legacy, band_label, pair_order),
            "acc_samediff": subject_accuracy(df_samediff, band_label, pair_order),
        })

    n_stronger = df_legacy["Subject"].nunique()
    n_samediff = df_samediff["Subject"].nunique()

    legend_alpha = BOX_FACE_ALPHA_OVERLAY if draw_panel is draw_overlay_panel else 1.0

    fig, axes = plt.subplots(1, 2, figsize=ACC_BY_PAIR_FIGSIZE, sharey=True)
    ylim_tops = []
    for ax, spec in zip(axes, band_specs):
        ylim_top = draw_panel(
            ax, spec["pair_order"], spec["acc_stronger"], spec["acc_samediff"],
            spec["band_label"], show_ylabel=(ax is axes[0]),
        )
        ylim_tops.append(ylim_top)
    shared_ylim = max(ylim_tops)
    for ax in axes:
        ax.set_ylim(0, shared_ylim)
        apply_accuracy_y_spine_bounds(ax)

    fig.legend(
        handles=paradigm_legend_handles(
            n_stronger, n_samediff,
            samediff_subjects=sorted(df_samediff["Subject"].unique()),
            face_alpha=legend_alpha,
        ),
        loc="upper center",
        bbox_to_anchor=(0.5, FIG_LEGEND_ANCHOR_Y),
        bbox_transform=fig.transFigure,
        ncol=2,
        frameon=False,
        fontsize=COMBINED_FONT_LEGEND,
        columnspacing=2.0,
        handletextpad=0.5,
        handlelength=1.6,
    )
    if subtitle:
        fig.suptitle(subtitle, fontsize=COMBINED_FONT_LABEL, y=0.995, va="top")
    fig.subplots_adjust(
        left=0.07, right=0.98, top=FIG_PANEL_TOP_FRAC, bottom=0.12, wspace=0.10,
    )

    out_path = os.path.join(OUTPUT_DIR, out_name)
    save_png_at_width(
        fig, out_path,
        width_px=EXPORT_WIDTH_2COL,
        height_px=EXPORT_HEIGHT_2COL,
        dpi=SAVE_DPI_COMBINED,
    )
    plt.close(fig)
    return out_path, band_specs, n_stronger, n_samediff


def export_summary_csv(band_specs, n_stronger, n_samediff, out_path):
    rows = []
    for spec in band_specs:
        for paradigm, acc_df, n_subj in (
            ("Stronger", spec["acc_stronger"], n_stronger),
            ("SameDiff", spec["acc_samediff"], n_samediff),
        ):
            for pair in spec["pair_order"]:
                vals = acc_df.loc[acc_df["pair_label"] == pair, "accuracy"].values
                if len(vals) == 0:
                    continue
                rows.append({
                    "paradigm": paradigm,
                    "band": spec["band_label"],
                    "pair_label": pair,
                    "n_subjects": n_subj,
                    "n_with_data": len(vals),
                    "median_accuracy_pct": np.median(vals) * 100,
                    "mean_accuracy_pct": np.mean(vals) * 100,
                    "sem_pct": np.std(vals, ddof=1) / np.sqrt(len(vals)) * 100 if len(vals) > 1 else np.nan,
                })
    df_out = pd.DataFrame(rows)
    if not df_out.empty:
        delta_rows = []
        for (band, pair), grp in df_out.groupby(["band", "pair_label"]):
            s = grp.loc[grp["paradigm"] == "Stronger", "median_accuracy_pct"]
            d = grp.loc[grp["paradigm"] == "SameDiff", "median_accuracy_pct"]
            if len(s) and len(d):
                delta_rows.append({
                    "band": band,
                    "pair_label": pair,
                    "stronger_median_pct": s.iloc[0],
                    "samediff_median_pct": d.iloc[0],
                    "median_delta_pp": d.iloc[0] - s.iloc[0],
                })
        if delta_rows:
            pd.DataFrame(delta_rows).to_csv(
                out_path.replace("_summary.csv", "_median_delta.csv"), index=False,
            )
    df_out.to_csv(out_path, index=False)


def main():
    df_legacy = load_legacy_df()

    layouts = [
        (draw_overlay_panel, "overlay", BOX_FACE_ALPHA_OVERLAY),
        (draw_sidebyside_panel, "sidebyside", 1.0),
    ]

    versions = [
        {
            "different_only": False,
            "base": "paradigm_compare_alltrials",
            "subtitle": "Same/Different accuracy: all trials",
        },
        {
            "different_only": True,
            "base": "paradigm_compare_differentonly",
            "subtitle": "Same/Different accuracy: DIFFERENT trials only",
        },
    ]

    for ver in versions:
        df_samediff = load_samediff_df(different_only=ver["different_only"])
        band_specs = None
        n_stronger = n_samediff = None
        for draw_panel, layout_tag, _ in layouts:
            suffix = "" if layout_tag == "overlay" else f"_{layout_tag}"
            _, band_specs, n_stronger, n_samediff = save_comparison_figure(
                df_legacy, df_samediff,
                out_name=f"{ver['base']}_2col{suffix}.png",
                subtitle=ver["subtitle"],
                draw_panel=draw_panel,
            )
        export_summary_csv(
            band_specs, n_stronger, n_samediff,
            os.path.join(OUTPUT_DIR, f"{ver['base']}_summary.csv"),
        )

    print(f"\nAll outputs saved to:\n  {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
