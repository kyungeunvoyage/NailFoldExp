"""
Absolute Threshold Detection — single-participant figures
=========================================================
Based on ATD C1_Figure.py: same plots (Force × Condition; Force × Region × Condition)
for one subject only (e.g. P29).

Usage:
  python ATD_onebyone.py
  python ATD_onebyone.py --subject P21

Outputs (per subject):
  atd_onebyone_outputs/<SubjectID>/ATD_relative_accuracy_force_condition.png
  atd_onebyone_outputs/<SubjectID>/ATD_facet_region_by_force.png
  atd_onebyone_outputs/<SubjectID>/<SubjectID>_trial_data.csv
"""

import argparse
import os
import glob
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import statsmodels.formula.api as smf
from matplotlib import rcParams

# =============================================================================
# Config — change default subject here or pass --subject on the command line
# =============================================================================
DEFAULT_SUBJECT = "P29"

# PNAS style + palette (matches ATD C1_Figure.py)
SLATE_BLUE = "#56708A"
OLIVE      = "#686F12"
WINE       = "#7F212B"
CREAM      = "#EDE2D0"
IN_AIR        = SLATE_BLUE
ON_TOUCH_MID  = OLIVE
REF_LINE      = WINE
BLACK         = "#1A1A1A"

COND_ORDER = ["In-air", "On-touch (Mid)"]
COND_COLORS = {
    "In-air":         IN_AIR,
    "On-touch (Mid)": ON_TOUCH_MID,
}
STRIP_ALPHA = 0.45
BOX_ALPHA_HEX = "BE"

EXCLUDE_FORCES = set()
FIG_A_SIZE = (10.0, 4.0)
FIG_B_SIZE = (18.0, 4.5)
SAVE_DPI   = 600


def condition_legend_handles():
    return [
        mpatches.Patch(
            facecolor=COND_COLORS[c] + BOX_ALPHA_HEX,
            edgecolor=BLACK, linewidth=0.6, label=c,
        )
        for c in COND_ORDER
    ]


def add_condition_legend(ax, loc="lower right"):
    leg = ax.get_legend()
    if leg is not None:
        leg.remove()
    ax.legend(
        handles=condition_legend_handles(),
        labels=COND_ORDER,
        title="Condition",
        loc=loc,
        fontsize=8,
        title_fontsize=8,
        handlelength=1.2,
        handleheight=0.9,
        frameon=False,
        borderpad=0.5,
        labelspacing=0.3,
    )


def calc_relative_accuracy(row):
    if row["Target"] == 0:
        return 100 if row["Response"] == 0 else 0
    error_ratio = abs(row["Target"] - row["Response"]) / row["Target"]
    return max(0, (1 - error_ratio) * 100)


def get_star_label(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return None


def load_subject_data(subject_id, file_pattern):
    """Load one participant's ATD CSV (by filename or SubjectID column)."""
    subject_id = str(subject_id).strip()
    if not subject_id.upper().startswith("P"):
        subject_id = f"P{subject_id}"

    data_dir = os.path.dirname(file_pattern)
    direct = os.path.join(data_dir, f"{subject_id}_AbsoluteThresholdDetection.csv")

    if os.path.isfile(direct):
        df = pd.read_csv(direct)
        print(f"Loaded: {direct}  ({len(df)} trials)")
        return df, subject_id

    all_files = glob.glob(file_pattern)
    if not all_files:
        raise FileNotFoundError(f"No ATD CSV files at: {file_pattern}")

    df = pd.concat([pd.read_csv(f) for f in all_files], ignore_index=True)
    sub_col = "SubjectID" if "SubjectID" in df.columns else "Subject"
    if sub_col not in df.columns:
        raise ValueError(f"No subject column in data; expected SubjectID or Subject")

    subj_vals = df[sub_col].astype(str).unique()
    match = [s for s in subj_vals if s.strip().upper() == subject_id.upper()]
    if not match:
        raise FileNotFoundError(
            f"Subject {subject_id!r} not found. Available: {sorted(subj_vals)}"
        )
    df = df[df[sub_col].astype(str).str.upper() == match[0].upper()].copy()
    print(f"Filtered to {match[0]}: {len(df)} trials from pooled files")
    return df, match[0]


def preprocess(df):
    df = df.copy()
    df["Condition"] = df["Condition"].str.strip()
    df["Condition"] = df["Condition"].replace("Active", "On-touch (Mid)")
    df["Condition"] = df["Condition"].replace("On-touch (Hard)", "On-touch (Mid)")
    df["Condition"] = df["Condition"].replace("Passive", "In-air")
    df = df[df["Condition"] != "On-touch (Soft)"]
    df = df[df["Area"].isin(["A", "B", "C", "D", "E", "F"])].copy()

    df["Force_Val"] = df["Force"].str.extract(r"(\d+\.?\d*)").astype(float)
    all_forces = sorted(df["Force_Val"].unique())
    force_order = [f for f in all_forces if f not in EXCLUDE_FORCES]
    if not force_order:
        raise ValueError(f"No forces left after exclusions. Available: {all_forces}")

    df["Relative_Score"] = df.apply(calc_relative_accuracy, axis=1)
    df["Region"] = df["Area"]
    df_plot = df[df["Force_Val"].isin(force_order)].copy()
    cond_list = [c for c in COND_ORDER if c in df_plot["Condition"].unique()]
    region_order = sorted(df_plot["Region"].unique())
    return df_plot, force_order, cond_list, region_order


def plot_figure_a(df_plot, force_order, cond_list, palette_list, subject_id, out_path):
    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=FIG_A_SIZE)

    sns.boxplot(
        data=df_plot,
        x="Force_Val",
        y="Relative_Score",
        hue="Condition",
        hue_order=cond_list,
        palette=[c + BOX_ALPHA_HEX for c in palette_list],
        linewidth=0.8,
        fliersize=0,
        width=0.55,
        order=force_order,
        medianprops={"color": BLACK, "linewidth": 1.5},
        whiskerprops={"linewidth": 0.8, "color": BLACK},
        capprops={"linewidth": 0.8, "color": BLACK},
        boxprops={"linewidth": 0.8},
        legend=False,
        ax=ax,
    )
    sns.stripplot(
        data=df_plot,
        x="Force_Val",
        y="Relative_Score",
        hue="Condition",
        hue_order=cond_list,
        dodge=True,
        palette=palette_list,
        alpha=STRIP_ALPHA,
        size=5,
        jitter=0.18,
        ax=ax,
        order=force_order,
        linewidth=0,
        legend=False,
    )

    ax.axhline(80, color=REF_LINE, linestyle="--", linewidth=1.0, alpha=0.8, zorder=0)
    ax.set_xlabel("Stimulus Force (g)", labelpad=6)
    ax.set_ylabel("Relative Accuracy (%)", labelpad=6)
    ax.set_ylim(-5, 108)
    ax.set_title(f"Subject {subject_id} — Force × Condition", fontsize=10, pad=8, loc="left")
    add_condition_legend(ax)
    ax.text(-0.08, 1.08, "A", transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="top", ha="left")
    sns.despine(ax=ax)
    fig.tight_layout(pad=1.2)
    fig.savefig(out_path, dpi=SAVE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_figure_b(df_plot, force_order, cond_list, palette_list, region_order,
                  subject_id, out_path):
    significance_results = []
    for f in force_order:
        for r in region_order:
            sub_data = df_plot[(df_plot["Force_Val"] == f) & (df_plot["Region"] == r)]
            if sub_data["Condition"].nunique() > 1 and len(sub_data) >= 4:
                try:
                    model = smf.ols("Relative_Score ~ C(Condition)", data=sub_data).fit()
                    significance_results.append({
                        "Force": f, "Region": r, "p_val": model.pvalues.iloc[1],
                    })
                except Exception:
                    pass

    sns.set_theme(style="white")
    n_facet_cols = len(force_order)
    g = sns.FacetGrid(
        df_plot,
        col="Force_Val",
        col_order=force_order,
        height=FIG_B_SIZE[1],
        aspect=FIG_B_SIZE[0] / (n_facet_cols * FIG_B_SIZE[1]),
        sharey=True,
    )
    g.map_dataframe(
        sns.boxplot,
        x="Region",
        y="Relative_Score",
        hue="Condition",
        hue_order=cond_list,
        palette=[c + BOX_ALPHA_HEX for c in palette_list],
        linewidth=0.8,
        fliersize=0,
        width=0.65,
        order=region_order,
        medianprops={"color": BLACK, "linewidth": 1.5},
        whiskerprops={"linewidth": 0.8, "color": BLACK},
        capprops={"linewidth": 0.8, "color": BLACK},
        boxprops={"linewidth": 0.8},
        legend=False,
    )
    g.map_dataframe(
        sns.stripplot,
        x="Region",
        y="Relative_Score",
        hue="Condition",
        hue_order=cond_list,
        dodge=True,
        palette=palette_list,
        alpha=STRIP_ALPHA,
        size=4,
        jitter=0.18,
        order=region_order,
        linewidth=0,
        legend=False,
    )

    for ax_i in g.axes.flat:
        title_text = ax_i.get_title()
        current_force = float(title_text.split("=")[1].strip())
        ax_i.set_title(f"{current_force} g", fontsize=9, fontweight="bold", pad=4)
        ax_i.axhline(80, color=REF_LINE, linestyle="--", linewidth=0.9, alpha=0.75, zorder=0)
        ax_i.set_xlabel("")
        ax_i.tick_params(axis="both", labelsize=8)
        for sp in ["top", "right"]:
            ax_i.spines[sp].set_visible(False)
        ax_i.spines["left"].set_linewidth(0.8)
        ax_i.spines["bottom"].set_linewidth(0.8)

        for i, r in enumerate(region_order):
            res = next(
                (item for item in significance_results
                 if item["Force"] == current_force and item["Region"] == r),
                None,
            )
            if res:
                star = get_star_label(res["p_val"])
                if star:
                    y_max = df_plot[
                        (df_plot["Force_Val"] == current_force) & (df_plot["Region"] == r)
                    ]["Relative_Score"].max()
                    y_pos = y_max + 3 if y_max < 97 else 102
                    ax_i.text(i, y_pos, star, ha="center", va="bottom",
                              color=BLACK, fontsize=9, fontweight="bold")

    g.set_axis_labels("Region", "Relative Accuracy (%)")
    g.set(ylim=(-5, 115))
    g.fig.suptitle(f"Subject {subject_id} — Region × Condition by Force",
                   fontsize=10, y=1.02)
    add_condition_legend(g.axes.flat[-1])
    g.axes.flat[0].text(-0.22, 1.10, "B", transform=g.axes.flat[0].transAxes,
                        fontsize=12, fontweight="bold", va="top", ha="left")
    g.fig.set_size_inches(*FIG_B_SIZE)
    plt.subplots_adjust(top=0.84, wspace=0.08)
    g.fig.savefig(out_path, dpi=SAVE_DPI, bbox_inches="tight", facecolor="white")
    plt.close(g.fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="ATD figures for one participant")
    parser.add_argument(
        "--subject", "-s", default=DEFAULT_SUBJECT,
        help=f"Participant ID (default: {DEFAULT_SUBJECT})",
    )
    args = parser.parse_args()
    subject_id = args.subject.strip()
    if not subject_id.upper().startswith("P"):
        subject_id = f"P{subject_id}"

    rcParams.update({
        "figure.facecolor": "#FFFFFF",
        "axes.facecolor": "#FFFFFF",
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "font.size": 9,
    })

    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.normpath(os.path.join(script_dir, "../../"))
    file_pattern = os.path.join(
        repo_root, "Data", "(ATD)CurData", "P*_AbsoluteThresholdDetection.csv"
    )
    out_dir = os.path.join(script_dir, "atd_onebyone_outputs", subject_id)
    os.makedirs(out_dir, exist_ok=True)

    df_raw, subject_id = load_subject_data(subject_id, file_pattern)
    df_plot, force_order, cond_list, region_order = preprocess(df_raw)
    palette_list = [COND_COLORS[c] for c in cond_list]

    print(f"Forces: {force_order}")
    print(f"Conditions: {cond_list}")
    print(f"Regions: {region_order}")
    print(f"Trials: {len(df_plot)}")

    trial_csv = os.path.join(out_dir, f"{subject_id}_trial_data.csv")
    export_cols = [
        c for c in [
            "SubjectID", "Condition", "Region", "Area", "Force", "Force_Val",
            "Target", "Response", "IsCorrect", "Relative_Score",
        ]
        if c in df_plot.columns
    ]
    df_plot[export_cols].to_csv(trial_csv, index=False)
    print(f"Saved: {trial_csv}")

    plot_figure_a(
        df_plot, force_order, cond_list, palette_list, subject_id,
        os.path.join(out_dir, "ATD_relative_accuracy_force_condition.png"),
    )
    plot_figure_b(
        df_plot, force_order, cond_list, palette_list, region_order, subject_id,
        os.path.join(out_dir, "ATD_facet_region_by_force.png"),
    )

    print(f"\nDone — outputs in: {out_dir}")


if __name__ == "__main__":
    main()
