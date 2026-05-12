import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf

# --- Paths ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# CurData: Absolute threshold CSVs
FILE_PATTERN = os.path.join(
    SCRIPT_DIR,
    "..",
    "..",
    "Data",
    "(ATD)CurData",
    "P*_AbsoluteThresholdDetection.csv",
)
FILE_PATTERN = os.path.normpath(FILE_PATTERN)
OUT_DIR = os.path.join(SCRIPT_DIR, "atd_c1_outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# =============================================================================
# 1. Load + preprocess (same logic as your snippet; Area = nail regions A–F)
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

# Facet / mixed model below use "Region" in original script; data column is Area
df["Region"] = df["Area"]

# =============================================================================
# 2. Figure — Force × Condition (your boxplot + stripplot)
# =============================================================================
plt.figure(figsize=(12, 7))
sns.set_theme(style="white")

ax = sns.boxplot(
    data=df,
    x="Force_Val",
    y="Relative_Score",
    hue="Condition",
    palette=["#FFFFFF", "#D3D3D3"],
    linewidth=1.5,
    fliersize=0,
    width=0.6,
    order=force_order,
    medianprops={"color": "red", "linewidth": 2},
)

sns.stripplot(
    data=df,
    x="Force_Val",
    y="Relative_Score",
    hue="Condition",
    dodge=True,
    palette=["#000000", "#000000"],
    alpha=0.4,
    size=5,
    jitter=0.2,
    ax=ax,
    order=force_order,
)

plt.axhline(80, color="red", linestyle="--", linewidth=1.2, alpha=0.7)

num_subjects = (
    df["SubjectID"].nunique() if "SubjectID" in df.columns else len(all_files)
)
plt.title(
    f"Absolute Threshold Detection: Relative Accuracy (N={num_subjects})\n"
    r"(1 - |Error|/Target)",
    fontsize=16,
    pad=20,
)
plt.xlabel("Stimulus Force (g)", fontsize=13)
plt.ylabel("Relative Accuracy (%)", fontsize=13)
plt.ylim(-5, 105)

handles, labels = ax.get_legend_handles_labels()
plt.legend(handles[0:2], labels[0:2], title="Condition", frameon=False, loc="lower right")

sns.despine()
plt.tight_layout()
out_overview = os.path.join(OUT_DIR, "ATD_relative_accuracy_force_condition.png")
plt.savefig(out_overview, dpi=200, bbox_inches="tight")
print(f"Saved: {out_overview}")
plt.show()
print("분석 및 시각화 (Force × Condition) 완료.")

# =============================================================================
# 3. Facet by Force — Region × Condition (original C1 figure)
# =============================================================================
force_order_facet = [0.07, 0.16, 0.6, 1.0, 1.4]
region_order = sorted(df["Region"].unique())
# Facet plot: harmonized pair (cool vs warm), readable on white
cond_list = sorted(df["Condition"].unique())
condition_palette = ["#8EBAE5", "#E8A598"]  # soft blue, soft coral-rose
strip_palette = ["#1E4D6B", "#9A3D4A"]  # deeper mates for points


def get_star_label(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
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

g = sns.FacetGrid(
    df,
    col="Force_Val",
    col_order=force_order_facet,
    height=6,
    aspect=0.8,
    sharey=True,
)

g.map_dataframe(
    sns.boxplot,
    x="Region",
    y="Relative_Score",
    hue="Condition",
    hue_order=cond_list,
    palette=condition_palette,
    linewidth=1.2,
    fliersize=0,
    width=0.7,
    order=region_order,
    medianprops={"color": "#2C3E50", "linewidth": 2},
    boxprops={"alpha": 0.92, "edgecolor": "#5D6D7E", "linewidth": 1.0},
)

g.map_dataframe(
    sns.stripplot,
    x="Region",
    y="Relative_Score",
    hue="Condition",
    hue_order=cond_list,
    dodge=True,
    palette=strip_palette,
    alpha=0.45,
    size=3.2,
    jitter=0.2,
    order=region_order,
)

for ax in g.axes.flat:
    title_text = ax.get_title()
    current_force = float(title_text.split("=")[1].strip())
    ax.set_title(f"Force: {current_force}g", fontsize=14, fontweight="bold")

    ax.axhline(80, color="#C75B5B", linestyle="--", linewidth=1.05, alpha=0.65)

    for i, r in enumerate(region_order):
        res = next(
            (
                item
                for item in significance_results
                if item["Force"] == current_force and item["Region"] == r
            ),
            None,
        )
        if res:
            star = get_star_label(res["p_val"])
            if star:
                y_max = df[
                    (df["Force_Val"] == current_force) & (df["Region"] == r)
                ]["Relative_Score"].max()
                y_pos = y_max + 3 if y_max < 97 else 102
                ax.text(
                    i,
                    y_pos,
                    star,
                    ha="center",
                    va="bottom",
                    color="black",
                    fontsize=12,
                    fontweight="bold",
                )

g.set_axis_labels("Region", "Relative Accuracy (%)")
g.set(ylim=(-5, 115))

plt.legend(
    handles=g.axes.flat[0].get_legend_handles_labels()[0][: len(cond_list)],
    labels=cond_list,
    title="Condition",
    loc="lower right",
    frameon=True,
    framealpha=0.92,
    edgecolor="#E0E0E0",
)

plt.subplots_adjust(top=0.85, wspace=0.1)
plt.suptitle(
    f"Detection Accuracy by Force, Region, and Condition (N={df['SubjectID'].nunique()})",
    fontsize=18,
    y=1.02,
)

out_facet = os.path.join(OUT_DIR, "ATD_facet_region_by_force.png")
plt.savefig(out_facet, dpi=200, bbox_inches="tight")
print(f"Saved: {out_facet}")
plt.show()

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
