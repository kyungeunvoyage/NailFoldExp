"""
Same/Different 2AFC — Signal Detection Theory (SDT)
===================================================
Per-subject and pooled SDT metrics for P76+ Same/Different force-discrimination data.

Trial coding
------------
  Hit (H)        : P(respond DIFFERENT | GroundTruth = DIFFERENT)
  False alarm (FA): P(respond DIFFERENT | GroundTruth = SAME)
  d′             : z(H) − z(FA)          — sensitivity
  criterion c    : −½ [z(H) + z(FA)]     — response bias (c > 0 → bias toward SAME)

Rates use log-linear correction (Hautus & Lee, 2006): (count + 0.5) / (n + 1).

Outputs (Output/SameDiff_SDT/)
------------------------------
  sdt_per_subject.csv          — subject × band × pair × region
  sdt_pooled.csv               — pooled across subjects
  sd_dprime_by_pair_2col.png   — per-subject d′ (On-nail vs Off-nail dodged)
  sd_criterion_by_pair_2col.png
  sd_sdt_pooled_regions_2col.png — d′ + criterion, all regions pooled (2×2)
  sd_hit_fa_by_pair_2col.png   — per-subject H and FA (dodged)
  sd_sdt_roc_space_2col.png    — FA vs H scatter (one point per subject × pair)
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
from scipy.stats import norm

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = "/Users/kyungeunjung/NailFoldExp/Data/(FD)CurData"
FILE_PATTERNS = [
    os.path.join(DATA_DIR, "P*_ForceDiscrimination_SameDiff.csv"),
    os.path.join(DATA_DIR, "P*_ForceDiscrimination_SameDiff_26g.csv"),
]
OUTPUT_DIR = (
    "/Users/kyungeunjung/NailFoldExp/(New)Analysis/ForceDiscAnalysis/Output/SameDiff_SDT"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Style (match SameDiffGee / paradigm_compare 2-col export) ─────────────────
EXPORT_WIDTH_2COL = 2102
EXPORT_HEIGHT_2COL = 1298
EXPORT_HEIGHT_SDT_POOLED = 2200  # taller export for 2×2 d′ + c layout
FIG_SIZE = (14.0, round(14.0 * EXPORT_HEIGHT_2COL / EXPORT_WIDTH_2COL, 3))
FIG_SIZE_SDT_POOLED = (
    14.0, round(14.0 * EXPORT_HEIGHT_SDT_POOLED / EXPORT_WIDTH_2COL, 3),
)
SAVE_DPI = 600
ATD_FIG2_REF_N = 5
ATD_FIG2_DODGED_BOX_WIDTH = 0.275
COMBINED_PANEL_COUNT = 2
STRIP_JITTER_REF = 0.12
CAP_WIDTH = 0.10
BOX_LINEWIDTH = 1.0
REGION_OFFSET = 0.135
BLACK = "#1A1A1A"
TICK_LEN_AXES = 0.016
AXIS_SPINE_LW = 2.0
FIG_AXIS_LABELPAD = 6
FIG_PANEL_TOP_FRAC = 0.82
FIG_LEGEND_ANCHOR_Y = 0.975
FONT_TICK = 16
FONT_LABEL = 14
FONT_LEGEND = 12
RED = "#c0392b"
C_ON = "#7FB3D3"
C_OFF = "#D3E9F5"
C_HIT = "#4CAF50"
C_FA = "#E57373"
C_DOT = "#2166AC"
COLOR_LOW_BAND = "#BAD6EB"
COLOR_HIGH_BAND = "#D0E4FF"

ON_NAIL = ["C", "D"]
OFF_NAIL = ["A", "F"]

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
    fig_h = FIG_SIZE[1]
    return round(base_pt * 1137 / EXPORT_HEIGHT_2COL * fig_h / 4.5)


FONT_TICK_PX = _combined_axis_font_size(FONT_TICK)
FONT_LABEL_PX = _combined_axis_font_size(FONT_LABEL)
FONT_LEGEND_PX = _combined_axis_font_size(FONT_LEGEND)


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


def apply_combined_axis_spines(ax):
    for spine in ("left", "bottom"):
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color(BLACK)
        ax.spines[spine].set_linewidth(AXIS_SPINE_LW)


def add_inward_tick_guides(ax, n_x, y_ticks, *, labelsize):
    ax.grid(False)
    ax.tick_params(axis="both", which="both", length=0, labelsize=labelsize)
    x_trans = blended_transform_factory(ax.transData, ax.transAxes)
    y_trans = blended_transform_factory(ax.transAxes, ax.transData)
    y_lo, y_hi = ax.get_ylim()
    y_vals = [t for t in y_ticks if y_lo - 1e-9 <= t <= y_hi + 1e-9]
    for xi in range(n_x):
        ax.plot(
            [xi, xi], [0, TICK_LEN_AXES],
            color=BLACK, linewidth=AXIS_SPINE_LW, solid_capstyle="butt",
            transform=x_trans, clip_on=False, zorder=6,
        )
    for y in y_vals:
        ax.plot(
            [0, TICK_LEN_AXES], [y, y],
            color=BLACK, linewidth=AXIS_SPINE_LW, solid_capstyle="butt",
            transform=y_trans, clip_on=False, zorder=6,
        )


def panel_box_width(n_pairs):
    return ATD_FIG2_DODGED_BOX_WIDTH * n_pairs / ATD_FIG2_REF_N * COMBINED_PANEL_COUNT


def region_dodged_box_width(n_pairs):
    return panel_box_width(n_pairs) / 2 * 0.95


def save_png_at_width(fig, out_path, *, height_px=EXPORT_HEIGHT_2COL):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=SAVE_DPI, bbox_inches="tight",
                pad_inches=0.05, facecolor="white")
    buf.seek(0)
    master = Image.open(buf).convert("RGB")
    master.resize((EXPORT_WIDTH_2COL, height_px), Image.Resampling.LANCZOS).save(out_path)


def compute_sdt(sub_df):
    """Return H, FA, d′, c with log-linear rate correction."""
    diff_trials = sub_df[sub_df["GroundTruth"] == "DIFFERENT"]
    same_trials = sub_df[sub_df["GroundTruth"] == "SAME"]
    n_diff = len(diff_trials)
    n_same = len(same_trials)
    if n_diff == 0 or n_same == 0:
        return {
            "n_diff": n_diff, "n_same": n_same,
            "hits": np.nan, "false_alarms": np.nan,
            "H": np.nan, "FA": np.nan, "d_prime": np.nan, "criterion": np.nan,
        }

    hits = int((diff_trials["UserChoice"] == "DIFFERENT").sum())
    fas = int((same_trials["UserChoice"] == "DIFFERENT").sum())
    H = (hits + 0.5) / (n_diff + 1)
    FA = (fas + 0.5) / (n_same + 1)
    d_prime = norm.ppf(H) - norm.ppf(FA)
    criterion = -0.5 * (norm.ppf(H) + norm.ppf(FA))
    return {
        "n_diff": n_diff, "n_same": n_same,
        "hits": hits, "false_alarms": fas,
        "H": H, "FA": FA, "d_prime": d_prime, "criterion": criterion,
    }


def load_data():
    all_files = sorted(set(f for pat in FILE_PATTERNS for f in glob.glob(pat)))
    files = sorted(f for f in all_files if subject_number(f) > 73)
    if not files:
        raise FileNotFoundError("No Same/Different CSV files with subject ID > 73.")
    df = pd.concat([pd.read_csv(f, encoding="utf-8-sig") for f in files], ignore_index=True)
    df["pair_label"] = df.apply(
        lambda r: pair_label_from_row(r["Reference"], r["Comparison"]), axis=1,
    )
    df["band"] = df["Reference"].map({1: "Low", 26: "High"})
    df["region_group"] = df["Region"].map(
        {r: "On-nail" for r in ON_NAIL} | {r: "Off-nail" for r in OFF_NAIL}
    )
    return df


def build_sdt_tables(df):
    rows = []
    for keys, sub in df.groupby(["Subject", "band", "pair_label", "region_group"]):
        if sub["region_group"].isna().all():
            continue
        subject, band, pair, region = keys
        rows.append({
            "Subject": subject, "band": band,
            "pair_label": pair, "region_group": region,
            **compute_sdt(sub),
        })
    per_subject = pd.DataFrame(rows)

    pooled_rows = []
    for keys, sub in df[df["region_group"].notna()].groupby(
        ["band", "pair_label", "region_group"]
    ):
        band, pair, region = keys
        sdt = compute_sdt(sub)
        pooled_rows.append({
            "band": band, "pair_label": pair, "region_group": region,
            "n_subjects": sub["Subject"].nunique(),
            **sdt,
        })
    pooled = pd.DataFrame(pooled_rows)
    return per_subject, pooled


def build_sdt_tables_all_regions(df):
    """Subject × band × pair with all nail regions (A–F) pooled."""
    rows = []
    for keys, sub in df.groupby(["Subject", "band", "pair_label"]):
        subject, band, pair = keys
        rows.append({
            "Subject": subject, "band": band, "pair_label": pair, **compute_sdt(sub),
        })
    per_subject = pd.DataFrame(rows)

    pooled_rows = []
    for keys, sub in df.groupby(["band", "pair_label"]):
        band, pair = keys
        pooled_rows.append({
            "band": band, "pair_label": pair,
            "n_subjects": sub["Subject"].nunique(),
            **compute_sdt(sub),
        })
    pooled = pd.DataFrame(pooled_rows)
    return per_subject, pooled


def _metric_values_by_pair(per_subject, band, pair_order, metric):
    out = {}
    sub = per_subject[per_subject["band"] == band]
    for pair in pair_order:
        vals = sub.loc[sub["pair_label"] == pair, metric].dropna().values
        out[pair] = vals
    return out


def _draw_pooled_boxes(ax, pair_order, values_by_pair, *, y_label, box_color,
                       ref_line=None, show_ylabel=True):
    box_w = panel_box_width(len(pair_order))
    jitter_w = STRIP_JITTER_REF * box_w / ATD_FIG2_DODGED_BOX_WIDTH

    for xi, pair in enumerate(pair_order):
        vals = np.asarray(values_by_pair.get(pair, []), dtype=float)
        if len(vals) == 0:
            continue
        bp = ax.boxplot(
            [vals], positions=[xi], widths=box_w,
            patch_artist=True, showfliers=False, capwidths=CAP_WIDTH,
            whiskerprops={"linewidth": BOX_LINEWIDTH, "color": "black"},
            capprops={"linewidth": BOX_LINEWIDTH, "color": "black"},
            medianprops={"color": RED, "linewidth": 2},
            boxprops={"linewidth": BOX_LINEWIDTH, "edgecolor": "black"},
        )
        bp["boxes"][0].set_facecolor(box_color)
        bp["boxes"][0].set_edgecolor("black")
        jx = xi + jitter_x(len(vals), width=jitter_w)
        ax.scatter(jx, vals, color=C_DOT, alpha=0.6, s=20, zorder=3, edgecolors="none")

    if ref_line is not None:
        ax.axhline(ref_line, color="gray", ls="--", lw=1.1, alpha=0.8, zorder=1)
    ax.set_xticks(range(len(pair_order)))
    ax.set_xticklabels(pair_order, fontsize=FONT_TICK_PX)
    ax.set_xlim(-0.55, len(pair_order) - 0.45)
    ax.set_xlabel("Force pair (g)", fontsize=FONT_LABEL_PX, labelpad=FIG_AXIS_LABELPAD)
    if show_ylabel:
        ax.set_ylabel(y_label, fontsize=FONT_LABEL_PX, labelpad=FIG_AXIS_LABELPAD)
    ax.spines[["top", "right"]].set_visible(False)
    apply_combined_axis_spines(ax)


def _apply_shared_y_limits(axes, y_lo, y_hi, *, step=0.5, min_pad=0.3):
    pad = max(min_pad, (y_hi - y_lo) * 0.12) if y_hi > y_lo else min_pad
    y_ticks = np.arange(
        np.floor((y_lo - pad) * 2) / 2,
        np.ceil((y_hi + pad) * 2) / 2 + 0.01,
        step,
    )
    for ax in axes:
        ax.set_ylim(y_lo - pad, y_hi + pad)
        ax.set_yticks(y_ticks)
    return y_ticks


def save_sdt_pooled_regions_figure(per_subject_all, band_specs):
    """d′ (top row) and criterion (bottom row), all regions pooled — Low | High."""
    fig_h = FIG_SIZE_SDT_POOLED[1]
    font_tick = round(FONT_TICK * fig_h / FIG_SIZE[1])
    font_label = round(FONT_LABEL * fig_h / FIG_SIZE[1])
    font_legend = round(FONT_LEGEND * fig_h / FIG_SIZE[1])

    fig, axes = plt.subplots(2, 2, figsize=FIG_SIZE_SDT_POOLED, sharex="col")
    band_colors = {"Low": COLOR_LOW_BAND, "High": COLOR_HIGH_BAND}

    dprime_axes = axes[0]
    crit_axes = axes[1]
    d_lo, d_hi = np.inf, -np.inf
    c_lo, c_hi = np.inf, -np.inf

    def draw_row(ax, pair_order, values_by_pair, *, y_label, box_color, show_ylabel):
        box_w = panel_box_width(len(pair_order))
        jitter_w = STRIP_JITTER_REF * box_w / ATD_FIG2_DODGED_BOX_WIDTH
        for xi, pair in enumerate(pair_order):
            vals = np.asarray(values_by_pair.get(pair, []), dtype=float)
            if len(vals) == 0:
                continue
            bp = ax.boxplot(
                [vals], positions=[xi], widths=box_w,
                patch_artist=True, showfliers=False, capwidths=CAP_WIDTH,
                whiskerprops={"linewidth": BOX_LINEWIDTH, "color": "black"},
                capprops={"linewidth": BOX_LINEWIDTH, "color": "black"},
                medianprops={"color": RED, "linewidth": 2},
                boxprops={"linewidth": BOX_LINEWIDTH, "edgecolor": "black"},
            )
            bp["boxes"][0].set_facecolor(box_color)
            bp["boxes"][0].set_edgecolor("black")
            jx = xi + jitter_x(len(vals), width=jitter_w)
            ax.scatter(jx, vals, color=C_DOT, alpha=0.6, s=22, zorder=3, edgecolors="none")
        ax.axhline(0.0, color="gray", ls="--", lw=1.1, alpha=0.8, zorder=1)
        ax.set_xticks(range(len(pair_order)))
        ax.set_xticklabels(pair_order, fontsize=font_tick)
        ax.set_xlim(-0.55, len(pair_order) - 0.45)
        ax.set_xlabel("Force pair (g)", fontsize=font_label, labelpad=FIG_AXIS_LABELPAD)
        if show_ylabel:
            ax.set_ylabel(y_label, fontsize=font_label, labelpad=FIG_AXIS_LABELPAD)
        ax.spines[["top", "right"]].set_visible(False)
        apply_combined_axis_spines(ax)

    for col, spec in enumerate(band_specs):
        band = spec["band_label"]
        pair_order = spec["pair_order"]
        panel_title = _band_panel_title(band, spec["n_subj"])

        d_vals = _metric_values_by_pair(per_subject_all, band, pair_order, "d_prime")
        draw_row(
            dprime_axes[col], pair_order, d_vals,
            y_label="d′  (sensitivity)",
            box_color=band_colors.get(band, "#dde6f0"),
            show_ylabel=(col == 0),
        )
        dprime_axes[col].set_title(panel_title, fontsize=font_label, fontweight="bold", pad=12)
        dprime_axes[col].tick_params(labelbottom=False)
        dprime_axes[col].set_xlabel("")
        dprime_vals = [v for arr in d_vals.values() for v in arr if np.isfinite(v)]
        if dprime_vals:
            d_lo = min(d_lo, min(dprime_vals))
            d_hi = max(d_hi, max(dprime_vals))

        c_vals = _metric_values_by_pair(per_subject_all, band, pair_order, "criterion")
        draw_row(
            crit_axes[col], pair_order, c_vals,
            y_label="criterion c  (response bias)",
            box_color=band_colors.get(band, "#dde6f0"),
            show_ylabel=(col == 0),
        )
        c_vals_flat = [v for arr in c_vals.values() for v in arr if np.isfinite(v)]
        if c_vals_flat:
            c_lo = min(c_lo, min(c_vals_flat))
            c_hi = max(c_hi, max(c_vals_flat))

    d_ticks = _apply_shared_y_limits(list(dprime_axes), d_lo, d_hi, step=0.5, min_pad=0.4)
    c_ticks = _apply_shared_y_limits(list(crit_axes), c_lo, c_hi, step=0.5, min_pad=0.3)
    for ax, spec in zip(dprime_axes, band_specs):
        add_inward_tick_guides(ax, len(spec["pair_order"]), d_ticks, labelsize=font_tick)
    for ax, spec in zip(crit_axes, band_specs):
        add_inward_tick_guides(ax, len(spec["pair_order"]), c_ticks, labelsize=font_tick)

    fig.legend(
        handles=[
            mpatches.Patch(facecolor=COLOR_LOW_BAND, edgecolor="black", label="Low band"),
            mpatches.Patch(facecolor=COLOR_HIGH_BAND, edgecolor="black", label="High band"),
        ],
        loc="upper center", bbox_to_anchor=(0.5, 0.985),
        bbox_transform=fig.transFigure, ncol=2, frameon=False,
        fontsize=font_legend, columnspacing=2.0,
    )
    fig.suptitle(
        "SDT — All Regions Pooled (A–F)",
        fontsize=font_label + 1, fontweight="bold", y=1.0, va="top",
    )
    fig.subplots_adjust(left=0.10, right=0.98, top=0.90, bottom=0.08, hspace=0.38, wspace=0.10)
    save_png_at_width(
        fig, os.path.join(OUTPUT_DIR, "sd_sdt_pooled_regions_2col.png"),
        height_px=EXPORT_HEIGHT_SDT_POOLED,
    )
    plt.close(fig)


def _draw_region_dodged_boxes(ax, pair_order, values_by_pair_region, *, metric_key,
                              y_label, ref_line=None, ref_label=None):
    box_w = region_dodged_box_width(len(pair_order))
    jitter_w = STRIP_JITTER_REF * box_w / ATD_FIG2_DODGED_BOX_WIDTH
    region_specs = [("On-nail", C_ON, -REGION_OFFSET), ("Off-nail", C_OFF, REGION_OFFSET)]

    for xi, pair in enumerate(pair_order):
        for region, color, x_off in region_specs:
            vals = np.asarray(
                values_by_pair_region.get((pair, region), []), dtype=float,
            )
            if len(vals) == 0:
                continue
            pos = xi + x_off
            bp = ax.boxplot(
                [vals], positions=[pos], widths=box_w,
                patch_artist=True, showfliers=False, capwidths=CAP_WIDTH,
                whiskerprops={"linewidth": BOX_LINEWIDTH, "color": "black"},
                capprops={"linewidth": BOX_LINEWIDTH, "color": "black"},
                medianprops={"color": RED, "linewidth": 2},
                boxprops={"linewidth": BOX_LINEWIDTH, "edgecolor": "black"},
            )
            bp["boxes"][0].set_facecolor(color)
            bp["boxes"][0].set_edgecolor("black")
            jx = pos + jitter_x(len(vals), width=jitter_w)
            ax.scatter(jx, vals, color=color, alpha=0.65, s=18, zorder=3, edgecolors="none")

    if ref_line is not None:
        ax.axhline(ref_line, color="gray", ls="--", lw=1.1, alpha=0.8, zorder=1)
    ax.set_xticks(range(len(pair_order)))
    ax.set_xticklabels(pair_order, fontsize=FONT_TICK_PX)
    ax.set_xlim(-0.55, len(pair_order) - 0.45)
    ax.set_xlabel("Force pair (g)", fontsize=FONT_LABEL_PX, labelpad=FIG_AXIS_LABELPAD)
    ax.set_ylabel(y_label, fontsize=FONT_LABEL_PX, labelpad=FIG_AXIS_LABELPAD)
    ax.spines[["top", "right"]].set_visible(False)
    apply_combined_axis_spines(ax)


def _metric_values(per_subject, band, pair_order, metric):
    out = {}
    sub = per_subject[per_subject["band"] == band]
    for pair in pair_order:
        for region in ("On-nail", "Off-nail"):
            vals = sub.loc[
                (sub["pair_label"] == pair) & (sub["region_group"] == region),
                metric,
            ].dropna().values
            out[(pair, region)] = vals
    return out


def _band_panel_title(band_label, n_subj):
    ref = BAND_CONFIG[band_label]["title_ref"]
    return f"{band_label} band (ref = {ref}, n = {n_subj})"


def save_dprime_figure(per_subject, band_specs):
    fig, axes = plt.subplots(1, 2, figsize=FIG_SIZE, sharey=True)
    y_lo, y_hi = np.inf, -np.inf
    for ax, spec in zip(axes, band_specs):
        vals_map = _metric_values(
            per_subject, spec["band_label"], spec["pair_order"], "d_prime",
        )
        _draw_region_dodged_boxes(
            ax, spec["pair_order"], vals_map,
            metric_key="d_prime",
            y_label="d′  (sensitivity)" if ax is axes[0] else "",
            ref_line=0.0,
        )
        ax.set_title(_band_panel_title(spec["band_label"], spec["n_subj"]),
                     fontsize=FONT_LABEL_PX, fontweight="bold", pad=10)
        all_vals = [v for arr in vals_map.values() for v in arr]
        if all_vals:
            y_lo = min(y_lo, min(all_vals))
            y_hi = max(y_hi, max(all_vals))

    pad = max(0.4, (y_hi - y_lo) * 0.12) if y_hi > y_lo else 0.5
    y_ticks = np.arange(np.floor((y_lo - pad) * 2) / 2, np.ceil((y_hi + pad) * 2) / 2 + 0.01, 0.5)
    for ax in axes:
        ax.set_ylim(y_lo - pad, y_hi + pad)
        ax.set_yticks(y_ticks)
        add_inward_tick_guides(ax, len(band_specs[0]["pair_order"]), y_ticks,
                               labelsize=FONT_TICK_PX)

    fig.legend(
        handles=[
            mpatches.Patch(facecolor=C_ON, edgecolor="black", label="On-nail (C+D)"),
            mpatches.Patch(facecolor=C_OFF, edgecolor="black", label="Off-nail (A+F)"),
        ],
        loc="upper center", bbox_to_anchor=(0.5, FIG_LEGEND_ANCHOR_Y),
        bbox_transform=fig.transFigure, ncol=2, frameon=False,
        fontsize=FONT_LEGEND_PX, columnspacing=2.0,
    )
    fig.subplots_adjust(left=0.08, right=0.98, top=FIG_PANEL_TOP_FRAC, bottom=0.12, wspace=0.10)
    save_png_at_width(fig, os.path.join(OUTPUT_DIR, "sd_dprime_by_pair_2col.png"))
    plt.close(fig)


def save_criterion_figure(per_subject, band_specs):
    fig, axes = plt.subplots(1, 2, figsize=FIG_SIZE, sharey=True)
    y_lo, y_hi = np.inf, -np.inf
    for ax, spec in zip(axes, band_specs):
        vals_map = _metric_values(
            per_subject, spec["band_label"], spec["pair_order"], "criterion",
        )
        _draw_region_dodged_boxes(
            ax, spec["pair_order"], vals_map,
            metric_key="criterion",
            y_label="criterion c  (response bias)" if ax is axes[0] else "",
            ref_line=0.0,
        )
        ax.set_title(_band_panel_title(spec["band_label"], spec["n_subj"]),
                     fontsize=FONT_LABEL_PX, fontweight="bold", pad=10)
        all_vals = [v for arr in vals_map.values() for v in arr]
        if all_vals:
            y_lo = min(y_lo, min(all_vals))
            y_hi = max(y_hi, max(all_vals))

    pad = max(0.3, (y_hi - y_lo) * 0.12) if y_hi > y_lo else 0.5
    y_ticks = np.arange(np.floor((y_lo - pad) * 2) / 2, np.ceil((y_hi + pad) * 2) / 2 + 0.01, 0.5)
    for ax in axes:
        ax.set_ylim(y_lo - pad, y_hi + pad)
        ax.set_yticks(y_ticks)
        add_inward_tick_guides(ax, len(band_specs[0]["pair_order"]), y_ticks,
                               labelsize=FONT_TICK_PX)

    fig.legend(
        handles=[
            mpatches.Patch(facecolor=C_ON, edgecolor="black", label="On-nail (C+D)"),
            mpatches.Patch(facecolor=C_OFF, edgecolor="black", label="Off-nail (A+F)"),
        ],
        loc="upper center", bbox_to_anchor=(0.5, FIG_LEGEND_ANCHOR_Y),
        bbox_transform=fig.transFigure, ncol=2, frameon=False,
        fontsize=FONT_LEGEND_PX, columnspacing=2.0,
    )
    fig.subplots_adjust(left=0.08, right=0.98, top=FIG_PANEL_TOP_FRAC, bottom=0.12, wspace=0.10)
    save_png_at_width(fig, os.path.join(OUTPUT_DIR, "sd_criterion_by_pair_2col.png"))
    plt.close(fig)


def save_hit_fa_figure(per_subject, band_specs):
    """H and FA per pair — On/Off nail dodged; H vs FA metric dodged."""
    metric_offset = 0.045
    fig, axes = plt.subplots(1, 2, figsize=FIG_SIZE, sharey=True)
    for ax, spec in zip(axes, band_specs):
        box_w = region_dodged_box_width(len(spec["pair_order"])) * 0.88
        jitter_w = STRIP_JITTER_REF * box_w / ATD_FIG2_DODGED_BOX_WIDTH
        sub = per_subject[per_subject["band"] == spec["band_label"]]

        for xi, pair in enumerate(spec["pair_order"]):
            for region, rcolor, r_off in [
                ("On-nail", C_ON, -REGION_OFFSET), ("Off-nail", C_OFF, REGION_OFFSET),
            ]:
                cell = sub[(sub["pair_label"] == pair) & (sub["region_group"] == region)]
                for mi, (metric, mcolor, m_off) in enumerate([
                    ("H", C_HIT, -metric_offset), ("FA", C_FA, metric_offset),
                ]):
                    vals = cell[metric].dropna().values * 100
                    if len(vals) == 0:
                        continue
                    pos = xi + r_off + m_off
                    bp = ax.boxplot(
                        [vals], positions=[pos], widths=box_w * 0.45,
                        patch_artist=True, showfliers=False, capwidths=CAP_WIDTH * 0.8,
                        whiskerprops={"linewidth": BOX_LINEWIDTH, "color": "black"},
                        capprops={"linewidth": BOX_LINEWIDTH, "color": "black"},
                        medianprops={"color": RED, "linewidth": 2},
                        boxprops={"linewidth": BOX_LINEWIDTH, "edgecolor": "black"},
                    )
                    bp["boxes"][0].set_facecolor(mcolor if mi == 0 else mcolor)
                    bp["boxes"][0].set_alpha(0.55 if mi else 0.75)
                    jx = pos + jitter_x(len(vals), width=jitter_w * 0.6)
                    ax.scatter(jx, vals, color=mcolor, alpha=0.6, s=14, zorder=3)

        ax.axhline(50, color="gray", ls=":", lw=0.9, alpha=0.7)
        ax.set_xticks(range(len(spec["pair_order"])))
        ax.set_xticklabels(spec["pair_order"], fontsize=FONT_TICK_PX)
        ax.set_xlim(-0.55, len(spec["pair_order"]) - 0.45)
        ax.set_ylim(0, 105)
        ax.set_yticks(range(0, 101, 20))
        ax.set_xlabel("Force pair (g)", fontsize=FONT_LABEL_PX, labelpad=FIG_AXIS_LABELPAD)
        if ax is axes[0]:
            ax.set_ylabel("Rate (%)", fontsize=FONT_LABEL_PX, labelpad=FIG_AXIS_LABELPAD)
        ax.set_title(_band_panel_title(spec["band_label"], spec["n_subj"]),
                     fontsize=FONT_LABEL_PX, fontweight="bold", pad=10)
        ax.spines[["top", "right"]].set_visible(False)
        apply_combined_axis_spines(ax)
        add_inward_tick_guides(ax, len(spec["pair_order"]), list(range(0, 101, 20)),
                               labelsize=FONT_TICK_PX)

    fig.legend(
        handles=[
            mpatches.Patch(facecolor=C_HIT, edgecolor="black", alpha=0.75, label="Hit rate (H)"),
            mpatches.Patch(facecolor=C_FA, edgecolor="black", alpha=0.55, label="False-alarm rate (FA)"),
            mpatches.Patch(facecolor=C_ON, edgecolor="black", label="On-nail"),
            mpatches.Patch(facecolor=C_OFF, edgecolor="black", label="Off-nail"),
        ],
        loc="upper center", bbox_to_anchor=(0.5, FIG_LEGEND_ANCHOR_Y),
        bbox_transform=fig.transFigure, ncol=4, frameon=False,
        fontsize=FONT_LEGEND_PX, columnspacing=1.2,
    )
    fig.subplots_adjust(left=0.08, right=0.98, top=FIG_PANEL_TOP_FRAC, bottom=0.12, wspace=0.10)
    save_png_at_width(fig, os.path.join(OUTPUT_DIR, "sd_hit_fa_by_pair_2col.png"))
    plt.close(fig)


def save_roc_space_figure(per_subject, band_specs):
    """FA vs H scatter — one point per subject × pair (regions pooled per subject×pair)."""
    fig, axes = plt.subplots(1, 2, figsize=FIG_SIZE, sharey=True, sharex=True)
    for ax, spec in zip(axes, band_specs):
        sub = per_subject[per_subject["band"] == spec["band_label"]].copy()
        pooled_pts = (
            sub.groupby(["Subject", "pair_label"], as_index=False)
            .agg(H=("H", "mean"), FA=("FA", "mean"))
        )
        colors = [COLOR_LOW_BAND if spec["band_label"] == "Low" else COLOR_HIGH_BAND] * len(spec["pair_order"])
        for pi, pair in enumerate(spec["pair_order"]):
            pts = pooled_pts[pooled_pts["pair_label"] == pair]
            ax.scatter(
                pts["FA"] * 100, pts["H"] * 100,
                s=36, alpha=0.65, color=colors[pi], edgecolors="black", linewidths=0.4,
                label=pair, zorder=3,
            )
        ax.plot([0, 100], [0, 100], color="gray", ls=":", lw=0.8, alpha=0.5, zorder=1)
        ax.set_xlim(-2, 102)
        ax.set_ylim(-2, 102)
        ax.set_xticks(range(0, 101, 20))
        ax.set_yticks(range(0, 101, 20))
        ax.set_xlabel("False-alarm rate FA (%)", fontsize=FONT_LABEL_PX, labelpad=FIG_AXIS_LABELPAD)
        if ax is axes[0]:
            ax.set_ylabel("Hit rate H (%)", fontsize=FONT_LABEL_PX, labelpad=FIG_AXIS_LABELPAD)
        ax.set_title(_band_panel_title(spec["band_label"], spec["n_subj"]),
                     fontsize=FONT_LABEL_PX, fontweight="bold", pad=10)
        ax.spines[["top", "right"]].set_visible(False)
        apply_combined_axis_spines(ax)
        add_inward_tick_guides(ax, 6, list(range(0, 101, 20)), labelsize=FONT_TICK_PX)
        ax.legend(frameon=False, fontsize=FONT_LEGEND_PX * 0.85, loc="lower right",
                  title="Force pair (g)", title_fontsize=FONT_LEGEND_PX * 0.85)

    fig.subplots_adjust(left=0.10, right=0.98, top=FIG_PANEL_TOP_FRAC, bottom=0.12, wspace=0.12)
    save_png_at_width(fig, os.path.join(OUTPUT_DIR, "sd_sdt_roc_space_2col.png"))
    plt.close(fig)


def main():
    df = load_data()
    per_subject, pooled = build_sdt_tables(df)
    per_subject_all, pooled_all = build_sdt_tables_all_regions(df)

    per_subject.to_csv(os.path.join(OUTPUT_DIR, "sdt_per_subject.csv"), index=False)
    pooled.to_csv(os.path.join(OUTPUT_DIR, "sdt_pooled.csv"), index=False)
    per_subject_all.to_csv(
        os.path.join(OUTPUT_DIR, "sdt_per_subject_all_regions.csv"), index=False,
    )
    pooled_all.to_csv(
        os.path.join(OUTPUT_DIR, "sdt_pooled_all_regions.csv"), index=False,
    )

    band_specs = []
    for band_label, cfg in BAND_CONFIG.items():
        df_band = df[df["band"] == band_label]
        if df_band.empty:
            continue
        actual = df_band["pair_label"].unique().tolist()
        pair_order = fix_order(cfg["pair_order"], actual)
        band_specs.append({
            "band_label": band_label,
            "pair_order": pair_order,
            "n_subj": df_band["Subject"].nunique(),
        })

    save_dprime_figure(per_subject, band_specs)
    save_criterion_figure(per_subject, band_specs)
    save_sdt_pooled_regions_figure(per_subject_all, band_specs)
    save_hit_fa_figure(per_subject, band_specs)
    save_roc_space_figure(per_subject, band_specs)

    print(f"\nAll outputs saved to:\n  {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
