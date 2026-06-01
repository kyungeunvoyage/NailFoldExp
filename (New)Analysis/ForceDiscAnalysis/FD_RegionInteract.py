"""
================================================================
Force Discrimination — Region × Band Interaction Analysis
================================================================
질문: 특정 region이 Low band(ref=1g)에서 잘하고, 다른 region은
      High band(ref=26g)에서 더 잘하는 패턴이 있는가?

Outputs:
  - fd_interaction_profile.png    Region별 band 프로파일 + LME 결과
  - fd_interaction_delta.png      WR-matched delta heatmap (High - Low)
  - fd_interaction_wr_facet.png   WR 구간별 region 프로파일 비교
  - fd_interaction_stats.txt      LME 결과 텍스트 요약
================================================================
"""

import os
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import statsmodels.formula.api as smf

from fd_export import FIG_SIZE, save_figure_png

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

# ============================================================
# Paths (기존 코드와 동일)
# ============================================================
FILE_PATTERN = "/Users/kyungeunjung/NailFoldExp/Data/(FD)CurData/P*_ForceDiscrimination.csv"
OUTPUT_DIR   = "/Users/kyungeunjung/NailFoldExp/(New)Analysis/ForceDiscAnalysis/Output2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SLATE_BLUE = "#56708A"
OLIVE      = "#686F12"
WINE       = "#7F212B"
CREAM      = "#EDE2D0"

CHANCE_LEVEL  = 0.50
JND_CRITERION = 0.75

region_order = ["A", "B", "C", "D", "E", "F"]
region_palette = {
    "A": "#e74c3c", "B": "#e67e22", "C": "#27ae60",
    "D": "#16a085", "E": "#3498db", "F": "#9b59b6",
}
band_colors = {"Low (ref=1g)": SLATE_BLUE, "High (ref=26g)": OLIVE}

# ============================================================
# 데이터 로드 (기존 코드와 동일한 전처리)
# ============================================================
files = glob.glob(FILE_PATTERN)
if not files:
    raise FileNotFoundError(f"No files: {FILE_PATTERN}")

raw = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
sub_col = "SubjectID" if "SubjectID" in raw.columns else "Subject"

def calc_accuracy(row):
    if row["UserChoice"] == 1:
        return 1 if row["FirstStim"] > row["SecondStim"] else 0
    if row["UserChoice"] == 2:
        return 1 if row["SecondStim"] > row["FirstStim"] else 0
    return np.nan

df = raw.copy()
df["IsCorrect"]  = df.apply(calc_accuracy, axis=1)
df["Reference"]  = pd.to_numeric(df["Reference"],  errors="coerce")
df["Comparison"] = pd.to_numeric(df["Comparison"], errors="coerce")
df = df.dropna(subset=["IsCorrect", "Reference", "Comparison", "Region"])

df["WeberRatio"] = (df["Comparison"] - df["Reference"]).abs() / df["Reference"]
df["f_lo"]       = df[["Reference", "Comparison"]].min(axis=1)
df["f_hi"]       = df[["Reference", "Comparison"]].max(axis=1)
df["ForcePair"]  = df.apply(lambda r: f"{r['f_lo']:g}--{r['f_hi']:g}", axis=1)
df["Band"]       = np.where(df["Reference"] >= 10, "High (ref=26g)", "Low (ref=1g)")

# WR 그룹 분류 (비교 가능한 구간끼리 묶기)
def wr_group(wr):
    if wr < 0.45:
        return "WR≈0.40 (sub-chance)"
    elif wr < 0.65:
        return "WR≈0.60"
    else:
        return "WR≥1.00"

df["WR_Group"] = df["WeberRatio"].apply(wr_group)

# ============================================================
# Per-subject accuracy 집계
# ============================================================
subj_acc = (
    df.groupby([sub_col, "Region", "ForcePair", "Band", "WeberRatio", "WR_Group"])
    ["IsCorrect"]
    .mean()
    .reset_index()
    .rename(columns={"IsCorrect": "accuracy"})
)

# Band × Region 평균 (pair 전체 풀링)
band_region_subj = (
    subj_acc.groupby([sub_col, "Region", "Band"])["accuracy"]
    .mean()
    .reset_index()
)

# WR Group × Band × Region 평균
wr_band_region_subj = (
    subj_acc.groupby([sub_col, "Region", "Band", "WR_Group"])["accuracy"]
    .mean()
    .reset_index()
)

print(f"Loaded {df[sub_col].nunique()} subjects, {len(df)} trials")
print(f"Regions: {sorted(df['Region'].unique())}")
print(f"Bands:   {sorted(df['Band'].unique())}")

# ============================================================
# LME: accuracy ~ Region * Band + (1|Subject)
# ============================================================
stat_lines = []
stat_lines.append("=" * 60)
stat_lines.append("LME: accuracy ~ C(Region) * C(Band)  |  RE: Subject")
stat_lines.append("=" * 60)

interaction_p_overall = np.nan
interaction_results   = {}

try:
    lme_df = band_region_subj.copy()
    model  = smf.mixedlm(
        "accuracy ~ C(Region) * C(Band)",
        lme_df,
        groups=lme_df[sub_col],
    )
    result = model.fit(reml=True)

    # 전체 interaction term p-values
    int_terms = [c for c in result.pvalues.index
                 if "Region" in c and "Band" in c]
    stat_lines.append("\n[Interaction terms]")
    for t in int_terms:
        p = result.pvalues[t]
        star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        stat_lines.append(f"  {t}: p={p:.4f}  {star}")
        interaction_results[t] = p

    if interaction_results:
        interaction_p_overall = min(interaction_results.values())

    # Region main effect
    region_terms = [c for c in result.pvalues.index
                    if "Region" in c and "Band" not in c]
    stat_lines.append("\n[Region main effect terms]")
    for t in region_terms:
        p = result.pvalues[t]
        star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        stat_lines.append(f"  {t}: p={p:.4f}  {star}")

    # Band main effect
    band_terms = [c for c in result.pvalues.index
                  if "Band" in c and "Region" not in c]
    stat_lines.append("\n[Band main effect terms]")
    for t in band_terms:
        p = result.pvalues[t]
        star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        stat_lines.append(f"  {t}: p={p:.4f}  {star}")

    stat_lines.append(f"\nModel converged: {result.converged}")
    stat_lines.append(f"Overall min interaction p = {interaction_p_overall:.4f}")

except Exception as e:
    stat_lines.append(f"\nLME failed: {e}")
    result = None

# WR group별 LME
stat_lines.append("\n" + "=" * 60)
stat_lines.append("WR-group-specific LME: accuracy ~ C(Region) * C(Band)")
stat_lines.append("=" * 60)

wr_interaction_p = {}
for wr_grp in sorted(wr_band_region_subj["WR_Group"].unique()):
    sub_df = wr_band_region_subj[wr_band_region_subj["WR_Group"] == wr_grp].copy()
    stat_lines.append(f"\n  [{wr_grp}]")
    try:
        m = smf.mixedlm(
            "accuracy ~ C(Region) * C(Band)",
            sub_df,
            groups=sub_df[sub_col],
        )
        r = m.fit(reml=True)
        it = [c for c in r.pvalues.index if "Region" in c and "Band" in c]
        if it:
            ps = [r.pvalues[t] for t in it]
            min_p = min(ps)
            wr_interaction_p[wr_grp] = min_p
            star = "***" if min_p < 0.001 else "**" if min_p < 0.01 else "*" if min_p < 0.05 else "n.s."
            stat_lines.append(f"    min interaction p = {min_p:.4f}  {star}")
        else:
            stat_lines.append("    no interaction term found")
    except Exception as e:
        stat_lines.append(f"    LME failed: {e}")
        wr_interaction_p[wr_grp] = np.nan

stat_out = os.path.join(OUTPUT_DIR, "fd_interaction_stats.txt")
with open(stat_out, "w") as f:
    f.write("\n".join(stat_lines))
print("\n".join(stat_lines))
print(f"\nStats saved → {stat_out}")


# ============================================================
# Figure 1: Region Profile by Band (overall + per WR group)
# ============================================================
wr_groups = sorted(wr_band_region_subj["WR_Group"].unique())
n_panels  = 1 + len(wr_groups)   # overall + per WR group

fig1, axes1 = plt.subplots(1, n_panels, figsize=FIG_SIZE,
                            sharey=True, facecolor="white")

def _plot_profile(ax, data_subj, title, int_p=np.nan):
    """data_subj: [sub_col, Region, Band, accuracy]"""
    grp = (
        data_subj.groupby(["Region", "Band"])["accuracy"]
        .agg(["mean", "sem"])
        .reset_index()
    )
    for band_name, color in band_colors.items():
        sub = grp[grp["Band"] == band_name].set_index("Region").reindex(region_order)
        ax.errorbar(
            region_order,
            sub["mean"].values,
            yerr=sub["sem"].values,
            marker="o", ms=8, lw=2.2, capsize=4,
            color=color, label=band_name,
        )
    ax.axhline(JND_CRITERION, color=WINE, ls="--", lw=1.2, alpha=0.7)
    ax.axhline(CHANCE_LEVEL,  color="gray", ls=":", lw=1.0, alpha=0.6)
    ax.set_xlabel("Region", fontsize=11)
    ax.set_ylabel("Mean accuracy (pooled)", fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.grid(alpha=0.3)

    # p annotation
    if not np.isnan(int_p):
        star = "***" if int_p < 0.001 else "**" if int_p < 0.01 else "*" if int_p < 0.05 else "n.s."
        p_str = f"Interaction {star}\n(p={int_p:.3f})"
        color_txt = WINE if int_p < 0.05 else "gray"
    else:
        p_str = "LME n/a"
        color_txt = "gray"
    ax.text(0.04, 0.97, p_str, transform=ax.transAxes,
            va="top", fontsize=8.5, color=color_txt)
    ax.set_title(title, fontsize=11, fontweight="bold")

# Overall panel
_plot_profile(axes1[0], band_region_subj,
              "Overall\n(all pairs pooled)", interaction_p_overall)
axes1[0].legend(fontsize=8.5, frameon=False, loc="lower left")

# WR-group panels
for i, wr_grp in enumerate(wr_groups):
    sub_wr = wr_band_region_subj[wr_band_region_subj["WR_Group"] == wr_grp]
    _plot_profile(axes1[i + 1], sub_wr,
                  wr_grp, wr_interaction_p.get(wr_grp, np.nan))

fig1.suptitle(
    "Region × Band Interaction: Accuracy profile\n"
    "Do regions differ between Low (ref=1g) and High (ref=26g) bands?",
    fontsize=12, y=1.02,
)
fig1.tight_layout()
out1 = os.path.join(OUTPUT_DIR, "fd_interaction_profile.png")
save_figure_png(fig1, out1)
plt.close(fig1)
print(f"Saved → {out1}")


# ============================================================
# Figure 2: WR-matched Δ Heatmap (High − Low)
# ============================================================
# WR-matched pair 지정 (Weber ratio 기준 대응)
wr_matched_pairs = [
    ("0.6--1",  "15--26",  "WR≈0.40\nsub-chance"),
    ("1--1.4",  "15--26",  "WR≈0.40\n(alt low)"),
    ("0.4--1",  "10--26",  "WR≈0.60"),
    ("1--2",    "26--60",  "WR≥1.00"),
]

grp_pair_region = (
    subj_acc.groupby(["Region", "ForcePair"])["accuracy"]
    .mean()
    .reset_index()
)

delta_rows = []
valid_labels = []
for pair_low, pair_high, label in wr_matched_pairs:
    low_d  = grp_pair_region[grp_pair_region["ForcePair"] == pair_low]
    high_d = grp_pair_region[grp_pair_region["ForcePair"] == pair_high]
    if low_d.empty or high_d.empty:
        continue
    low_d  = low_d.set_index("Region")["accuracy"]
    high_d = high_d.set_index("Region")["accuracy"]
    row = {}
    for region in region_order:
        if region in low_d.index and region in high_d.index:
            row[region] = high_d[region] - low_d[region]
        else:
            row[region] = np.nan
    delta_rows.append(row)
    valid_labels.append(label)

if delta_rows:
    delta_df = pd.DataFrame(delta_rows, index=valid_labels)[region_order]

    fig2, (ax_hm, ax_bar) = plt.subplots(
        1, 2, figsize=FIG_SIZE,
        gridspec_kw={"width_ratios": [1.8, 1]},
        facecolor="white",
    )

    # Heatmap
    sns.heatmap(
        delta_df,
        annot=True, fmt=".2f",
        cmap="RdBu_r", center=0, vmin=-0.35, vmax=0.35,
        linewidths=0.5, linecolor="white",
        cbar_kws={"label": "Δ accuracy (High − Low)", "shrink": 0.85},
        ax=ax_hm,
    )
    ax_hm.set_title(
        "Δ Accuracy: High band − Low band\n(WR-matched pairs, per region)",
        fontsize=12, fontweight="bold",
    )
    ax_hm.set_xlabel("Region", fontsize=11)
    ax_hm.set_ylabel("WR-matched pair group", fontsize=10)
    ax_hm.set_yticklabels(valid_labels, rotation=0, fontsize=9)

    # Bar: mean Δ across WR groups per region
    mean_delta = delta_df.mean(axis=0)
    sem_delta  = delta_df.sem(axis=0)
    colors_bar = [region_palette[r] for r in region_order]
    ax_bar.bar(region_order, mean_delta, yerr=sem_delta,
               color=colors_bar, edgecolor="black", linewidth=0.6,
               capsize=4, width=0.65)
    ax_bar.axhline(0, color="black", lw=1.0)
    ax_bar.set_xlabel("Region", fontsize=11)
    ax_bar.set_ylabel("Mean Δ accuracy (High − Low)", fontsize=10)
    ax_bar.set_title("Average High − Low\nper region (across WR groups)", fontsize=11, fontweight="bold")
    ax_bar.grid(axis="y", alpha=0.3)
    sns.despine(ax=ax_bar)

    # Color bars by sign
    for patch, val in zip(ax_bar.patches, mean_delta):
        patch.set_alpha(0.85)
        patch.set_edgecolor(WINE if val < -0.05 else OLIVE if val > 0.05 else "gray")
        patch.set_linewidth(1.5 if abs(val) > 0.05 else 0.6)

    fig2.suptitle(
        "Region × Band Interaction: Which regions perform differently in Low vs High band?",
        fontsize=12, y=1.03,
    )
    fig2.tight_layout()
    out2 = os.path.join(OUTPUT_DIR, "fd_interaction_delta.png")
    save_figure_png(fig2, out2)
    plt.close(fig2)
    print(f"Saved → {out2}")


# ============================================================
# Figure 3: Per-subject spaghetti — individual Region × Band
# (가장 많은 정보: 개인 패턴 일관성 확인)
# ============================================================
fig3, axes3 = plt.subplots(1, len(wr_groups), figsize=FIG_SIZE,
                            sharey=True, facecolor="white")

for ax, wr_grp in zip(axes3, wr_groups):
    sub_wr = wr_band_region_subj[wr_band_region_subj["WR_Group"] == wr_grp]

    for subj_id, sdf in sub_wr.groupby(sub_col):
        for band_name, color in band_colors.items():
            bdf = sdf[sdf["Band"] == band_name].set_index("Region").reindex(region_order)
            ax.plot(region_order, bdf["accuracy"].values,
                    color=color, alpha=0.25, lw=0.9, marker="o", ms=3)

    # Overlay group mean
    for band_name, color in band_colors.items():
        bdf = (
            sub_wr[sub_wr["Band"] == band_name]
            .groupby("Region")["accuracy"]
            .mean()
            .reindex(region_order)
        )
        ax.plot(region_order, bdf.values,
                color=color, lw=2.5, marker="o", ms=9,
                label=band_name, zorder=5)

    ax.axhline(JND_CRITERION, color=WINE, ls="--", lw=1.2, alpha=0.7)
    ax.axhline(CHANCE_LEVEL,  color="gray", ls=":", lw=1.0, alpha=0.6)
    ax.set_xlabel("Region", fontsize=11)
    ax.set_ylabel("Accuracy" if ax == axes3[0] else "", fontsize=10)
    ax.set_ylim(-0.05, 1.10)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.grid(alpha=0.3)

    int_p = wr_interaction_p.get(wr_grp, np.nan)
    star = ("***" if int_p < 0.001 else "**" if int_p < 0.01
            else "*" if int_p < 0.05 else "n.s.") if not np.isnan(int_p) else "n/a"
    ax.set_title(f"{wr_grp}\nInteraction {star} (p={int_p:.3f})" if not np.isnan(int_p)
                 else wr_grp, fontsize=11, fontweight="bold")

    if ax == axes3[0]:
        ax.legend(fontsize=8.5, frameon=False)

fig3.suptitle(
    "Individual Region profiles: Low vs High band (spaghetti + group mean)\n"
    "Thin lines = individual subjects, thick = group mean",
    fontsize=12, y=1.02,
)
fig3.tight_layout()
out3 = os.path.join(OUTPUT_DIR, "fd_interaction_wr_facet.png")
save_figure_png(fig3, out3)
plt.close(fig3)
print(f"Saved → {out3}")

print("\n=== Region × Band interaction analysis complete ===")