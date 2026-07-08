"""
Which-is-Stronger 2AFC — Signal Detection Theory (SDT)
=====================================================
Per-subject and pooled SDT metrics for legacy force-discrimination data (P21–P59).

This task has no SAME trials, and each force pair has a fixed stronger side
(reference or comparison). Per-pair classic H & FA cannot both be computed.

Two analysis levels (parallel to SameDiffSDT outputs)
-----------------------------------------------------
1) Band × region (pooled across force pairs within band)
   Full SDT with both trial types mixed:
     Signal : Comparison > Reference
     Noise  : Reference > Comparison

2) Per force pair (for 2-column figures)
     H      : on comparison-stronger pairs only  (= proportion correct)
     FA     : on reference-stronger pairs only   (= P chose comparison)
     d′     : √2 · z(p_c)  with log-linear correction on proportion correct
              (standard 2AFC sensitivity when catch trials are absent)
     c      : z(P chose comparison) − z(0.5)  — comparison-response bias

Outputs (Output/Stronger_SDT/)
------------------------------
  sdt_per_subject_pair.csv     — subject × band × pair × region
  sdt_pooled_pair.csv
  sdt_per_subject_band.csv     — subject × band × region (full H/FA/d′/c)
  sdt_pooled_band.csv
  sd_dprime_by_pair_2col.png
  sd_criterion_by_pair_2col.png
  sd_hit_fa_by_pair_2col.png
  sd_sdt_roc_space_2col.png
  sd_dprime_by_band_2col.png   — band-pooled full SDT
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

DATA_DIR = "/Users/kyungeunjung/NailFoldExp/Data/(FD)CurData"
FILE_PATTERN = os.path.join(DATA_DIR, "P*_ForceDiscrimination.csv")
OUTPUT_DIR = (
    "/Users/kyungeunjung/NailFoldExp/(New)Analysis/ForceDiscAnalysis/Output/Stronger_SDT"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

LEGACY_SUBJ_RANGE = (21, 59)

EXPORT_WIDTH_2COL = 2102
EXPORT_HEIGHT_2COL = 1298
FIG_SIZE = (14.0, round(14.0 * EXPORT_HEIGHT_2COL / EXPORT_WIDTH_2COL, 3))
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


def region_dodged_box_width(n_pairs):
    return (
        ATD_FIG2_DODGED_BOX_WIDTH * n_pairs / ATD_FIG2_REF_N * COMBINED_PANEL_COUNT / 2 * 0.95
    )


def save_png_at_width(fig, out_path, *, height_px=EXPORT_HEIGHT_2COL):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=SAVE_DPI, bbox_inches="tight",
                pad_inches=0.05, facecolor="white")
    buf.seek(0)
    master = Image.open(buf).convert("RGB")
    master.resize((EXPORT_WIDTH_2COL, height_px), Image.Resampling.LANCZOS).save(out_path)


def _adj_rate(count, n):
    return (count + 0.5) / (n + 1)


def compute_sdt_band(sub_df):
    """Full SDT pooling all force pairs within a band × region cell."""
    signal = sub_df[sub_df["Comparison"] > sub_df["Reference"]]
    noise = sub_df[sub_df["Reference"] > sub_df["Comparison"]]
    n_signal = len(signal)
    n_noise = len(noise)
    if n_signal == 0 or n_noise == 0:
        return {
            "n_signal": n_signal, "n_noise": n_noise,
            "hits": np.nan, "false_alarms": np.nan,
            "H": np.nan, "FA": np.nan, "d_prime": np.nan, "criterion": np.nan,
        }

    hits = int((signal["ChoseComparison"] == 1).sum())
    fas = int((noise["ChoseComparison"] == 1).sum())
    H = _adj_rate(hits, n_signal)
    FA = _adj_rate(fas, n_noise)
    z_h, z_fa = norm.ppf(H), norm.ppf(FA)
    return {
        "n_signal": n_signal, "n_noise": n_noise,
        "hits": hits, "false_alarms": fas,
        "H": H, "FA": FA,
        "d_prime": z_h - z_fa,
        "criterion": -0.5 * (z_h + z_fa),
    }


def compute_sdt_pair(sub_df):
    """Per-pair SDT analog for the fixed-reference stronger design."""
    n = len(sub_df)
    if n == 0:
        return {
            "n_trials": 0, "n_signal": 0, "n_noise": 0,
            "hits": np.nan, "false_alarms": np.nan,
            "pc": np.nan, "H": np.nan, "FA": np.nan,
            "d_prime": np.nan, "criterion": np.nan,
            "pair_type": "",
        }

    correct = np.where(
        sub_df["Comparison"] > sub_df["Reference"],
        sub_df["ChoseComparison"] == 1,
        sub_df["ChoseComparison"] == 0,
    )
    n_correct = int(correct.sum())
    pc = _adj_rate(n_correct, n)
    chose_comp = int((sub_df["ChoseComparison"] == 1).sum())
    p_comp = _adj_rate(chose_comp, n)

    comp_stronger = bool((sub_df["Comparison"] > sub_df["Reference"]).all())
    ref_stronger = bool((sub_df["Reference"] > sub_df["Comparison"]).all())
    pair_type = "comparison-stronger" if comp_stronger else "reference-stronger"

    H = pc if comp_stronger else np.nan
    FA = p_comp if ref_stronger else np.nan
    hits = n_correct if comp_stronger else np.nan
    false_alarms = chose_comp if ref_stronger else np.nan

    return {
        "n_trials": n,
        "n_signal": int((sub_df["Comparison"] > sub_df["Reference"]).sum()),
        "n_noise": int((sub_df["Reference"] > sub_df["Comparison"]).sum()),
        "hits": hits, "false_alarms": false_alarms,
        "pc": pc, "H": H, "FA": FA,
        "d_prime": np.sqrt(2) * norm.ppf(pc),
        "criterion": norm.ppf(p_comp) - norm.ppf(0.5),
        "pair_type": pair_type,
    }


def load_data():
    files = sorted(
        f for f in glob.glob(FILE_PATTERN)
        if LEGACY_SUBJ_RANGE[0] <= subject_number(f) <= LEGACY_SUBJ_RANGE[1]
    )
    if not files:
        raise FileNotFoundError(
            f"No legacy files in P{LEGACY_SUBJ_RANGE[0]}–P{LEGACY_SUBJ_RANGE[1]}"
        )
    df = pd.concat([pd.read_csv(f, encoding="utf-8-sig") for f in files], ignore_index=True)
    df = df[df["Reference"] != df["Comparison"]].copy()
    df["pair_label"] = df.apply(
        lambda r: pair_label_from_row(r["Reference"], r["Comparison"]), axis=1,
    )
    df["band"] = np.where(df["Reference"] == 1, "Low", "High")
    df["region_group"] = df["Region"].map(
        {r: "On-nail" for r in ON_NAIL} | {r: "Off-nail" for r in OFF_NAIL}
    )
    return df


def build_sdt_tables(df):
    pair_rows = []
    for keys, sub in df.groupby(["Subject", "band", "pair_label", "region_group"]):
        if pd.isna(keys[3]):
            continue
        subject, band, pair, region = keys
        pair_rows.append({
            "Subject": subject, "band": band,
            "pair_label": pair, "region_group": region,
            **compute_sdt_pair(sub),
        })
    per_subject_pair = pd.DataFrame(pair_rows)

    pooled_pair_rows = []
    for keys, sub in df[df["region_group"].notna()].groupby(
        ["band", "pair_label", "region_group"]
    ):
        band, pair, region = keys
        pooled_pair_rows.append({
            "band": band, "pair_label": pair, "region_group": region,
            "n_subjects": sub["Subject"].nunique(),
            **compute_sdt_pair(sub),
        })
    pooled_pair = pd.DataFrame(pooled_pair_rows)

    band_rows = []
    for keys, sub in df.groupby(["Subject", "band", "region_group"]):
        if pd.isna(keys[2]):
            continue
        subject, band, region = keys
        band_rows.append({
            "Subject": subject, "band": band, "region_group": region,
            **compute_sdt_band(sub),
        })
    per_subject_band = pd.DataFrame(band_rows)

    pooled_band_rows = []
    for keys, sub in df[df["region_group"].notna()].groupby(["band", "region_group"]):
        band, region = keys
        pooled_band_rows.append({
            "band": band, "region_group": region,
            "n_subjects": sub["Subject"].nunique(),
            **compute_sdt_band(sub),
        })
    pooled_band = pd.DataFrame(pooled_band_rows)

    return per_subject_pair, pooled_pair, per_subject_band, pooled_band


def _draw_region_dodged_boxes(ax, pair_order, values_by_pair_region, *, y_label, ref_line=None):
    box_w = region_dodged_box_width(len(pair_order))
    jitter_w = STRIP_JITTER_REF * box_w / ATD_FIG2_DODGED_BOX_WIDTH
    for xi, pair in enumerate(pair_order):
        for region, color, x_off in [("On-nail", C_ON, -REGION_OFFSET), ("Off-nail", C_OFF, REGION_OFFSET)]:
            vals = np.asarray(values_by_pair_region.get((pair, region), []), dtype=float)
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
                (sub["pair_label"] == pair) & (sub["region_group"] == region), metric,
            ].dropna().values
            out[(pair, region)] = vals
    return out


def _finite_vals(values_by_pair_region):
    return [v for arr in values_by_pair_region.values() for v in arr if np.isfinite(v)]


def _set_symmetric_ylim(axes, y_lo, y_hi, *, default=0.5):
    if not np.isfinite(y_lo) or not np.isfinite(y_hi):
        y_lo, y_hi = -default, default
    pad = max(0.4, (y_hi - y_lo) * 0.12) if y_hi > y_lo else default
    y_ticks = np.arange(
        np.floor((y_lo - pad) * 2) / 2,
        np.ceil((y_hi + pad) * 2) / 2 + 0.01,
        0.5,
    )
    if len(y_ticks) > 30:
        y_ticks = np.linspace(y_lo - pad, y_hi + pad, 7)
    for ax in axes:
        ax.set_ylim(y_lo - pad, y_hi + pad)
        ax.set_yticks(y_ticks)
    return y_ticks


def _band_panel_title(band_label, n_subj):
    ref = BAND_CONFIG[band_label]["title_ref"]
    return f"{band_label} band (ref = {ref}, n = {n_subj})"


def _metric_values_band(per_subject_band, band, metric):
    out = {}
    sub = per_subject_band[per_subject_band["band"] == band]
    for region in ("On-nail", "Off-nail"):
        vals = sub.loc[sub["region_group"] == region, metric].dropna().values
        out[region] = vals
    return out


def save_dprime_figure(per_subject_pair, band_specs):
    fig, axes = plt.subplots(1, 2, figsize=FIG_SIZE, sharey=True)
    y_lo, y_hi = np.inf, -np.inf
    for ax, spec in zip(axes, band_specs):
        vals_map = _metric_values(
            per_subject_pair, spec["band_label"], spec["pair_order"], "d_prime",
        )
        _draw_region_dodged_boxes(
            ax, spec["pair_order"], vals_map,
            y_label="d′  (√2 · z(p_c))" if ax is axes[0] else "", ref_line=0.0,
        )
        ax.set_title(_band_panel_title(spec["band_label"], spec["n_subj"]),
                     fontsize=FONT_LABEL_PX, fontweight="bold", pad=10)
        vals = _finite_vals(vals_map)
        if vals:
            y_lo = min(y_lo, min(vals))
            y_hi = max(y_hi, max(vals))

    y_ticks = _set_symmetric_ylim(axes, y_lo, y_hi)
    for ax in axes:
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


def save_dprime_band_figure(per_subject_band, band_specs):
    """Band-pooled full SDT d′ (signal + noise trials combined)."""
    fig, axes = plt.subplots(1, 2, figsize=FIG_SIZE, sharey=True)
    y_lo, y_hi = np.inf, -np.inf
    for ax, spec in zip(axes, band_specs):
        vals_map = _metric_values_band(per_subject_band, spec["band_label"], "d_prime")
        box_w = region_dodged_box_width(3)
        jitter_w = STRIP_JITTER_REF * box_w / ATD_FIG2_DODGED_BOX_WIDTH
        for xi, (region, color) in enumerate([("On-nail", C_ON), ("Off-nail", C_OFF)]):
            vals = np.asarray(vals_map.get(region, []), dtype=float)
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
            bp["boxes"][0].set_facecolor(color)
            jx = xi + jitter_x(len(vals), width=jitter_w)
            ax.scatter(jx, vals, color=color, alpha=0.65, s=18, zorder=3, edgecolors="none")
            y_lo = min(y_lo, float(np.min(vals)))
            y_hi = max(y_hi, float(np.max(vals)))

        ax.axhline(0.0, color="gray", ls="--", lw=1.1, alpha=0.8)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["On-nail\n(C+D)", "Off-nail\n(A+F)"], fontsize=FONT_TICK_PX)
        ax.set_xlim(-0.55, 1.45)
        ax.set_title(_band_panel_title(spec["band_label"], spec["n_subj"]),
                     fontsize=FONT_LABEL_PX, fontweight="bold", pad=10)
        if ax is axes[0]:
            ax.set_ylabel("d′  (z(H) − z(FA))", fontsize=FONT_LABEL_PX, labelpad=FIG_AXIS_LABELPAD)
        ax.spines[["top", "right"]].set_visible(False)
        apply_combined_axis_spines(ax)

    y_ticks = _set_symmetric_ylim(axes, y_lo, y_hi)
    for ax in axes:
        add_inward_tick_guides(ax, 2, y_ticks, labelsize=FONT_TICK_PX)

    fig.subplots_adjust(left=0.10, right=0.98, top=FIG_PANEL_TOP_FRAC, bottom=0.12, wspace=0.12)
    save_png_at_width(fig, os.path.join(OUTPUT_DIR, "sd_dprime_by_band_2col.png"))
    plt.close(fig)


def save_criterion_figure(per_subject_pair, band_specs):
    fig, axes = plt.subplots(1, 2, figsize=FIG_SIZE, sharey=True)
    y_lo, y_hi = np.inf, -np.inf
    for ax, spec in zip(axes, band_specs):
        vals_map = _metric_values(
            per_subject_pair, spec["band_label"], spec["pair_order"], "criterion",
        )
        _draw_region_dodged_boxes(
            ax, spec["pair_order"], vals_map,
            y_label="comparison bias  (z(p_comp) − z(0.5))" if ax is axes[0] else "",
            ref_line=0.0,
        )
        ax.set_title(_band_panel_title(spec["band_label"], spec["n_subj"]),
                     fontsize=FONT_LABEL_PX, fontweight="bold", pad=10)
        vals = _finite_vals(vals_map)
        if vals:
            y_lo = min(y_lo, min(vals))
            y_hi = max(y_hi, max(vals))

    y_ticks = _set_symmetric_ylim(axes, y_lo, y_hi, default=0.3)
    for ax in axes:
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


def save_hit_fa_figure(per_subject_pair, band_specs):
    """H on comparison-stronger pairs; FA on reference-stronger pairs."""
    fig, axes = plt.subplots(1, 2, figsize=FIG_SIZE, sharey=True)
    for ax, spec in zip(axes, band_specs):
        box_w = region_dodged_box_width(len(spec["pair_order"])) * 0.88
        jitter_w = STRIP_JITTER_REF * box_w / ATD_FIG2_DODGED_BOX_WIDTH
        sub = per_subject_pair[per_subject_pair["band"] == spec["band_label"]]

        for xi, pair in enumerate(spec["pair_order"]):
            pair_type = sub.loc[sub["pair_label"] == pair, "pair_type"].iloc[0]
            metric = "H" if pair_type == "comparison-stronger" else "FA"
            mcolor = C_HIT if metric == "H" else C_FA
            for region, rcolor, r_off in [
                ("On-nail", C_ON, -REGION_OFFSET), ("Off-nail", C_OFF, REGION_OFFSET),
            ]:
                cell = sub[(sub["pair_label"] == pair) & (sub["region_group"] == region)]
                vals = cell[metric].dropna().values * 100
                if len(vals) == 0:
                    continue
                pos = xi + r_off
                bp = ax.boxplot(
                    [vals], positions=[pos], widths=box_w * 0.55,
                    patch_artist=True, showfliers=False, capwidths=CAP_WIDTH * 0.8,
                    whiskerprops={"linewidth": BOX_LINEWIDTH, "color": "black"},
                    capprops={"linewidth": BOX_LINEWIDTH, "color": "black"},
                    medianprops={"color": RED, "linewidth": 2},
                    boxprops={"linewidth": BOX_LINEWIDTH, "edgecolor": "black"},
                )
                bp["boxes"][0].set_facecolor(mcolor)
                bp["boxes"][0].set_alpha(0.7)
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
            mpatches.Patch(facecolor=C_HIT, edgecolor="black", alpha=0.75,
                           label="Hit rate (comparison-stronger pairs)"),
            mpatches.Patch(facecolor=C_FA, edgecolor="black", alpha=0.75,
                           label="FA rate (reference-stronger pairs)"),
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


def save_roc_space_figure(per_subject_band, band_specs):
    """Band-pooled H vs FA (full SDT rates)."""
    fig, axes = plt.subplots(1, 2, figsize=FIG_SIZE, sharey=True, sharex=True)
    for ax, spec in zip(axes, band_specs):
        sub = per_subject_band[per_subject_band["band"] == spec["band_label"]].copy()
        for region, color in [("On-nail", C_ON), ("Off-nail", C_OFF)]:
            pts = sub[sub["region_group"] == region]
            ax.scatter(
                pts["FA"] * 100, pts["H"] * 100,
                s=42, alpha=0.7, color=color, edgecolors="black", linewidths=0.4,
                label=region, zorder=3,
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
        ax.legend(frameon=False, fontsize=FONT_LEGEND_PX, loc="lower right")

    fig.subplots_adjust(left=0.10, right=0.98, top=FIG_PANEL_TOP_FRAC, bottom=0.12, wspace=0.12)
    save_png_at_width(fig, os.path.join(OUTPUT_DIR, "sd_sdt_roc_space_2col.png"))
    plt.close(fig)


def main():
    df = load_data()
    per_subject_pair, pooled_pair, per_subject_band, pooled_band = build_sdt_tables(df)

    per_subject_pair.to_csv(
        os.path.join(OUTPUT_DIR, "sdt_per_subject_pair.csv"), index=False,
    )
    pooled_pair.to_csv(os.path.join(OUTPUT_DIR, "sdt_pooled_pair.csv"), index=False)
    per_subject_band.to_csv(
        os.path.join(OUTPUT_DIR, "sdt_per_subject_band.csv"), index=False,
    )
    pooled_band.to_csv(os.path.join(OUTPUT_DIR, "sdt_pooled_band.csv"), index=False)

    band_specs = []
    for band_label, cfg in BAND_CONFIG.items():
        df_band = df[df["band"] == band_label]
        if df_band.empty:
            continue
        pair_order = fix_order(cfg["pair_order"], df_band["pair_label"].unique().tolist())
        band_specs.append({
            "band_label": band_label,
            "pair_order": pair_order,
            "n_subj": df_band["Subject"].nunique(),
        })

    save_dprime_figure(per_subject_pair, band_specs)
    save_criterion_figure(per_subject_pair, band_specs)
    save_hit_fa_figure(per_subject_pair, band_specs)
    save_dprime_band_figure(per_subject_band, band_specs)
    save_roc_space_figure(per_subject_band, band_specs)

    print(f"\nAll outputs saved to:\n  {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
