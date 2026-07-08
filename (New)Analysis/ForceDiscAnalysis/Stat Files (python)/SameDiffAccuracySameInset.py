"""
Same/Different 2AFC — accuracy with SAME-trial response inset
==============================================================
Professor suggestion: show what participants responded on SAME trials
as a small strip below the main discrimination figure, rather than
giving SAME and DIFFERENT equal visual weight (sd_accuracy_split) or
as a separate full figure (sd_same_trial_response).

Layout (2-col Low | High):
  • Main (~82% height): overall accuracy by force pair (subject boxplots)
  • Inset (~18% height): per pair, stacked bar of SAME-trial responses
      blue  = responded SAME (correct on SAME trials)
      orange = responded DIFFERENT (false alarm on SAME trials)
    Bars show the median per-subject response rate across participants.

Output: Output/SameDiff_GEE/sd_accuracy_same_inset_2col.png
"""

import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from SameDiffGee import (
    BAND_CONFIG,
    CHANCE_PCT,
    JND_PCT,
    OUTPUT_DIR,
    ACC_BY_PAIR_FIGSIZE,
    EXPORT_WIDTH_2COL,
    EXPORT_HEIGHT_2COL,
    SAVE_DPI_COMBINED,
    COMBINED_FONT_TICK,
    COMBINED_FONT_LABEL,
    COMBINED_FONT_LEGEND,
    FIG_LEGEND_ANCHOR_Y,
    COLOR_LOW_BAND,
    COLOR_HIGH_BAND,
    C_SAME,
    C_DIFF,
    band_title_text,
    combined_panel_box_width,
    save_png_at_width,
    fix_order,
    run_gee_pairwise,
    _draw_accuracy_panel,
    _combined_accuracy_legend_handles,
    apply_combined_axis_spines,
    apply_accuracy_y_spine_bounds,
    add_inward_tick_guides,
)

DATA_DIR = "/Users/kyungeunjung/NailFoldExp/Data/(FD)CurData"
FILE_PATTERNS = [
    os.path.join(DATA_DIR, "P*_ForceDiscrimination_SameDiff.csv"),
    os.path.join(DATA_DIR, "P*_ForceDiscrimination_SameDiff_26g.csv"),
]

INSET_HEIGHT_RATIO = 0.22
MAIN_HEIGHT_RATIO = 1.0 - INSET_HEIGHT_RATIO
EXPORT_HEIGHT_INSET = round(EXPORT_HEIGHT_2COL * 1.12)


def _load_df():
    all_files = sorted(set(f for pat in FILE_PATTERNS for f in glob.glob(pat)))
    if not all_files:
        raise FileNotFoundError("No Same/Diff CSV files found.")

    def subject_number(filepath):
        m = re.search(r"P(\d+)", os.path.basename(filepath))
        return int(m.group(1)) if m else 0

    files = sorted(f for f in all_files if subject_number(f) > 73)
    df = pd.concat(
        [pd.read_csv(f, encoding="utf-8-sig") for f in files],
        ignore_index=True,
    )
    df["correct"] = df["IsCorrect"].astype(int)
    df["pair_label"] = df.apply(
        lambda r: f"{min(r['Reference'], r['Comparison']):g}–{max(r['Reference'], r['Comparison']):g}",
        axis=1,
    )
    df["band"] = df["Reference"].map({1: "Low", 26: "High"})
    return df


def _subject_same_trial_response(df_band, pair_order):
    """Per subject × pair: response rates on SAME trials only."""
    rows = []
    for subj, sub in df_band.groupby("Subject"):
        for pair in pair_order:
            trials = sub[(sub["pair_label"] == pair) & (sub["GroundTruth"] == "SAME")]
            if len(trials) == 0:
                continue
            n_same_resp = int((trials["UserChoice"] == "SAME").sum())
            n = len(trials)
            rows.append({
                "Subject": subj,
                "pair_label": pair,
                "n_trials": n,
                "pct_resp_same": n_same_resp / n * 100,
                "pct_resp_diff": (n - n_same_resp) / n * 100,
            })
    return pd.DataFrame(rows)


def _inset_median_rows(subj_same_df, pair_order):
    rows = []
    for pair in pair_order:
        sub = subj_same_df[subj_same_df["pair_label"] == pair]
        if sub.empty:
            continue
        rows.append({
            "pair_label": pair,
            "median_pct_resp_same": float(np.median(sub["pct_resp_same"])),
            "median_pct_resp_diff": float(np.median(sub["pct_resp_diff"])),
            "n_subjects": int(sub["Subject"].nunique()),
        })
    return rows


def _draw_same_trial_inset(ax, pair_order, subj_same_df, *, box_width=0.55, show_ylabel=True):
    """Thin stacked bars: SAME-trial response distribution (median across subjects)."""
    inset_font = max(9, COMBINED_FONT_TICK - 4)
    for xi, pair in enumerate(pair_order):
        sub = subj_same_df[subj_same_df["pair_label"] == pair]
        if sub.empty:
            continue
        med_same = float(np.median(sub["pct_resp_same"]))
        med_diff = float(np.median(sub["pct_resp_diff"]))
        ax.bar(
            xi, med_same, width=box_width, color=C_SAME,
            edgecolor="black", linewidth=0.6, zorder=2,
        )
        ax.bar(
            xi, med_diff, width=box_width, bottom=med_same, color=C_DIFF,
            edgecolor="black", linewidth=0.6, zorder=2,
        )

    ax.axhline(CHANCE_PCT, color="gray", ls=":", lw=0.9, alpha=0.75, zorder=1)
    ax.set_xlim(-0.55, len(pair_order) - 0.45)
    ax.set_ylim(0, 100)
    ax.set_xticks(range(len(pair_order)))
    ax.set_xticklabels(pair_order, fontsize=inset_font)
    ax.set_yticks([0, 50, 100])
    ax.tick_params(axis="both", labelsize=inset_font, length=0)
    if show_ylabel:
        ax.set_ylabel(
            "SAME-trial\nresponses (%)",
            fontsize=inset_font,
            labelpad=4,
        )
    ax.set_xlabel("Force pair (g)", fontsize=COMBINED_FONT_LABEL)
    ax.spines[["top", "right"]].set_visible(False)
    apply_combined_axis_spines(ax)
    apply_accuracy_y_spine_bounds(ax)


def save_accuracy_same_inset_figure(band_specs):
    n_cols = len(band_specs)
    fig_w, fig_h_base = ACC_BY_PAIR_FIGSIZE
    fig_h = fig_h_base * (1.0 + INSET_HEIGHT_RATIO * 0.35)
    fig = plt.figure(figsize=(fig_w, fig_h))
    outer = fig.add_gridspec(1, n_cols, wspace=0.10)

    band_colors = {"Low": COLOR_LOW_BAND, "High": COLOR_HIGH_BAND}
    shared_ylim = 100.0
    inset_summary_rows = []

    for col, spec in enumerate(band_specs):
        inner = outer[col].subgridspec(
            2, 1,
            height_ratios=[MAIN_HEIGHT_RATIO, INSET_HEIGHT_RATIO],
            hspace=0.06,
        )
        ax_main = fig.add_subplot(inner[0])
        ax_inset = fig.add_subplot(inner[1], sharex=ax_main)

        pair_order = spec["pair_order"]
        box_w = combined_panel_box_width(len(pair_order))
        jitter_w = 0.12 * box_w / 0.275

        _, ylim_top = _draw_accuracy_panel(
            ax_main, spec["subj_acc"], pair_order, spec["pairwise_pvals"],
            show_ylabel=(col == 0),
            bracket_in_data_space=True,
            box_color=band_colors.get(spec["band_label"], "#dde6f0"),
            box_width=box_w,
            jitter_width=jitter_w,
        )
        shared_ylim = max(shared_ylim, ylim_top)
        ax_main.set_title(
            band_title_text(spec["band_label"], spec["title_ref"], spec["n_subj"]),
            fontsize=COMBINED_FONT_LABEL,
            fontweight="bold",
            pad=8,
        )
        ax_main.tick_params(labelbottom=False)
        ax_main.set_xlabel("")

        subj_same = _subject_same_trial_response(spec["df_band"], pair_order)
        _draw_same_trial_inset(
            ax_inset, pair_order, subj_same,
            box_width=min(0.55, box_w * 1.15),
            show_ylabel=(col == 0),
        )
        for row in _inset_median_rows(subj_same, pair_order):
            inset_summary_rows.append({
                "band": spec["band_label"],
                **row,
            })

    for ax in fig.axes[::2]:
        ax.set_ylim(0, shared_ylim)

    fig.suptitle(
        "Overall Accuracy with SAME-Trial Response Inset — Same/Different 2AFC",
        fontsize=COMBINED_FONT_LABEL + 1,
        fontweight="bold",
        y=0.995,
        va="top",
    )

    band_handles = _combined_accuracy_legend_handles(band_specs)
    inset_handles = [
        mpatches.Patch(facecolor=C_SAME, edgecolor="black", label="Responded: SAME (on SAME trials)"),
        mpatches.Patch(facecolor=C_DIFF, edgecolor="black", label="Responded: DIFFERENT (false alarm)"),
        mpatches.Patch(facecolor="white", edgecolor="gray", linestyle=":", label="50% reference"),
    ]
    fig.legend(
        handles=band_handles + inset_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, FIG_LEGEND_ANCHOR_Y - 0.02),
        bbox_transform=fig.transFigure,
        ncol=2,
        frameon=False,
        fontsize=COMBINED_FONT_LEGEND,
        columnspacing=1.5,
        handletextpad=0.4,
        handlelength=1.4,
    )

    fig.text(
        0.5, 0.015,
        "Main panel: overall accuracy (all trials).  "
        "Bottom strip: median per-subject response on SAME trials only "
        "(stacked blue + orange = 100%).",
        ha="center", va="bottom", fontsize=max(9, COMBINED_FONT_LEGEND - 2), color="#444",
        transform=fig.transFigure,
    )

    fig.subplots_adjust(left=0.07, right=0.98, top=0.78, bottom=0.10, wspace=0.10)

    out_png = os.path.join(OUTPUT_DIR, "sd_accuracy_same_inset_2col.png")
    save_png_at_width(
        fig, out_png,
        width_px=EXPORT_WIDTH_2COL,
        height_px=EXPORT_HEIGHT_INSET,
        dpi=SAVE_DPI_COMBINED,
        pad_inches=0.05,
    )
    plt.close(fig)

    out_csv = os.path.join(OUTPUT_DIR, "same_inset_summary_2col.csv")
    pd.DataFrame(inset_summary_rows).to_csv(out_csv, index=False)
    return out_png, out_csv


def main():
    df = _load_df()
    band_specs = []

    for band_label, cfg in BAND_CONFIG.items():
        df_band = df[df["band"] == band_label].copy()
        if df_band.empty:
            continue
        pair_order = fix_order(cfg["pair_order"], df_band["pair_label"].unique())
        subj_acc = (
            df_band.groupby(["Subject", "pair_label"])["correct"]
            .mean().reset_index()
            .rename(columns={"correct": "accuracy"})
        )
        band_specs.append({
            "band_label": band_label,
            "title_ref": cfg["title_ref"],
            "pair_order": pair_order,
            "suffix": cfg["suffix"],
            "df_band": df_band,
            "subj_acc": subj_acc,
            "pairwise_pvals": run_gee_pairwise(df_band, subj_acc, pair_order),
            "n_subj": df_band["Subject"].nunique(),
        })

    if len(band_specs) < 2:
        raise ValueError("Need both Low and High bands for 2-col figure.")

    out_png, out_csv = save_accuracy_same_inset_figure(band_specs)
    print(f"Saved: {out_png}")
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
