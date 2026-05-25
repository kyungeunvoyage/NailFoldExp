import os
import glob
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive: save fixed-size figures only
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import statsmodels.formula.api as smf
from matplotlib import rcParams

# =============================================================================
# PNAS Style + shared figure palette
# =============================================================================
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
BOX_ALPHA_HEX = "BE"   # ~75 % opacity appended to hex color

# Fixed figure sizes (inches) and export resolution
FIG_A_SIZE = (7.2, 4.0)
FIG_B_SIZE = (18.0, 4.5)
SAVE_DPI   = 200


def condition_legend_handles():
    return [
        mpatches.Patch(
            facecolor=COND_COLORS[c] + BOX_ALPHA_HEX,
            edgecolor=BLACK, linewidth=0.6, label=c,
        )
        for c in COND_ORDER
    ]


def add_condition_legend(ax, loc="lower right"):
    """All figures share the same Condition legend (In-air / On-touch Mid)."""
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

# --- PNAS-like rcParams ---
rcParams.update({
    "figure.facecolor":      "#FFFFFF",
    "axes.facecolor":        "#FFFFFF",
    "font.family":           "sans-serif",
    "font.sans-serif":       ["Helvetica", "Arial", "DejaVu Sans"],
    "axes.linewidth":        0.8,
    "axes.spines.top":       False,
    "axes.spines.right":     False,
    "xtick.major.width":     0.8,
    "ytick.major.width":     0.8,
    "xtick.major.size":      3.5,
    "ytick.major.size":      3.5,
    "xtick.direction":       "out",
    "ytick.direction":       "out",
    "legend.frameon":        False,
    "legend.fontsize":       8,
    "legend.title_fontsize": 8,
    "font.size":             9,
    "axes.titlesize":        10,
    "axes.labelsize":        9,
    "figure.dpi":            150,
})

# --- Paths ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.normpath(os.path.join(SCRIPT_DIR, "../../"))
FILE_PATTERN = os.path.join(
    REPO_ROOT, "Data", "(ATD)CurData", "P*_AbsoluteThresholdDetection.csv"
)
OUT_DIR = os.path.join(SCRIPT_DIR, "atd_c1_outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# =============================================================================
# 1. Load + preprocess
# =============================================================================
all_files = glob.glob(FILE_PATTERN)
print(f"로드된 파일 개수: {len(all_files)}")

if not all_files:
    print("분석할 CSV 파일을 찾을 수 없습니다.")
    raise SystemExit(1)

print(f"발견된 파일 (처음 5개): {all_files[:5]}...")
df_list = [pd.read_csv(f) for f in all_files]
df = pd.concat(df_list, ignore_index=True)

df["Condition"] = df["Condition"].str.strip()
df["Condition"] = df["Condition"].replace("Active", "On-touch (Mid)")
df["Condition"] = df["Condition"].replace("On-touch (Hard)", "On-touch (Mid)")
df["Condition"] = df["Condition"].replace("Passive", "In-air")

df = df[df["Condition"] != "On-touch (Soft)"]
df = df[df["Area"].isin(["A", "B", "C", "D", "E", "F"])].copy()

df["Force_Val"] = df["Force"].str.extract(r"(\d+\.?\d*)").astype(float)
force_order = sorted(df["Force_Val"].unique())


def calc_relative_accuracy(row):
    if row["Target"] == 0:
        return 100 if row["Response"] == 0 else 0
    error_ratio = abs(row["Target"] - row["Response"]) / row["Target"]
    score = (1 - error_ratio) * 100
    return max(0, score)


df["Relative_Score"] = df.apply(calc_relative_accuracy, axis=1)
df["Region"] = df["Area"]

cond_list    = [c for c in COND_ORDER if c in df["Condition"].unique()]
num_subjects = df["SubjectID"].nunique() if "SubjectID" in df.columns else len(all_files)
palette_list = [COND_COLORS[c] for c in cond_list]

# =============================================================================
# 2. Figure A — Force × Condition  (PNAS style)
# =============================================================================
sns.set_theme(style="white")
fig, ax = plt.subplots(figsize=FIG_A_SIZE)

sns.boxplot(
    data=df,
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
    data=df,
    x="Force_Val",
    y="Relative_Score",
    hue="Condition",
    hue_order=cond_list,
    dodge=True,
    palette=palette_list,
    alpha=STRIP_ALPHA,
    size=3.5,
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
#ax.set_title(
#     f"Absolute Threshold Detection: Relative Accuracy  (N = {num_subjects})",
#     fontsize=10, pad=8, loc="left",
# )

add_condition_legend(ax)

# Panel label
ax.text(-0.08, 1.08, "A", transform=ax.transAxes,
        fontsize=12, fontweight="bold", va="top", ha="left")

sns.despine(ax=ax)
fig.tight_layout(pad=1.2)

out_overview = os.path.join(OUT_DIR, "ATD_relative_accuracy_force_condition.png")
fig.savefig(out_overview, dpi=SAVE_DPI, bbox_inches="tight", facecolor="white")
print(f"Saved: {out_overview}  ({FIG_A_SIZE[0]}×{FIG_A_SIZE[1]} in @ {SAVE_DPI} dpi)")
plt.close(fig)
print("분석 및 시각화 (Force × Condition) 완료.")

# =============================================================================
# 3. Figure B — Facet: Force × Region × Condition  (PNAS style)
# =============================================================================
force_order_facet = [0.07, 0.16, 0.6, 1.0, 1.4]
region_order = sorted(df["Region"].unique())


def get_star_label(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return None


significance_results = []
for f in force_order_facet:
    for r in region_order:
        sub_data = df[(df["Force_Val"] == f) & (df["Region"] == r)]
        if len(sub_data["Condition"].unique()) > 1:
            try:
                model = smf.ols("Relative_Score ~ C(Condition)", data=sub_data).fit()
                p_val = model.pvalues.iloc[1]
                significance_results.append({"Force": f, "Region": r, "p_val": p_val})
            except Exception:
                pass

sns.set_theme(style="white")

n_facet_cols = len(force_order_facet)
g = sns.FacetGrid(
    df,
    col="Force_Val",
    col_order=force_order_facet,
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
    size=2.8,
    jitter=0.18,
    order=region_order,
    linewidth=0,
    legend=False,
)

# Style each facet panel
for ax_i in g.axes.flat:
    title_text  = ax_i.get_title()
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
                y_max = df[
                    (df["Force_Val"] == current_force) & (df["Region"] == r)
                ]["Relative_Score"].max()
                y_pos = y_max + 3 if y_max < 97 else 102
                ax_i.text(i, y_pos, star, ha="center", va="bottom",
                          color=BLACK, fontsize=9, fontweight="bold")

g.set_axis_labels("Region", "Relative Accuracy (%)")
g.set(ylim=(-5, 115))

# Condition legend on rightmost facet only
add_condition_legend(g.axes.flat[-1])

# Panel label B
g.axes.flat[0].text(-0.22, 1.10, "B", transform=g.axes.flat[0].transAxes,
                    fontsize=12, fontweight="bold", va="top", ha="left")

g.fig.set_size_inches(*FIG_B_SIZE)
plt.subplots_adjust(top=0.88, wspace=0.08)
# plt.suptitle(
#     f"Detection Accuracy by Force, Region, and Condition  (N = {num_subjects})",
#     fontsize=10, y=0.98,
# )

out_facet = os.path.join(OUT_DIR, "ATD_facet_region_by_force.png")
g.fig.savefig(out_facet, dpi=SAVE_DPI, bbox_inches="tight", facecolor="white")
print(f"Saved: {out_facet}  ({FIG_B_SIZE[0]}×{FIG_B_SIZE[1]} in @ {SAVE_DPI} dpi)")
plt.close(g.fig)

# =============================================================================
# 4. Mixed-effects summary
# =============================================================================
print("\n[Mixed-Effects Model: Relative_Score ~ Force * Condition * Region]")
full_model = smf.mixedlm(
    "Relative_Score ~ C(Force_Val) * C(Condition) * C(Region)",
    groups=df["SubjectID"],
    data=df,
).fit()
print(full_model.summary())