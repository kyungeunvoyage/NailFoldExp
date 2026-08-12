import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ---------------------------------------------------------
# 1. 데이터 로드 및 전처리 (P21 ~ P59)
# ---------------------------------------------------------
data_dir = "/Users/kyungeunjung/NailFoldExp/Data/(FD)CurData"
file_paths = glob.glob(os.path.join(data_dir, "P*.csv"))

# P21 ~ P59 데이터만 필터링
valid_files = []
for f in file_paths:
    filename = os.path.basename(f)
    try:
        sub_num = int(''.join(filter(str.isdigit, filename)))
        if 21 <= sub_num <= 59:
            valid_files.append(f)
    except ValueError:
        continue

df_list = []
for f in valid_files:
    temp_df = pd.read_csv(f)
    df_list.append(temp_df)

if not df_list:
    raise FileNotFoundError("지정한 경로에서 P21~P59 CSV 파일을 찾을 수 없습니다.")

df = pd.concat(df_list, ignore_index=True)

# 지각된 정답 자극 판단 (1st > 2nd -> StrongerStim = 1, else 2)
df['StrongerStim'] = df.apply(lambda r: 1 if r['FirstStim'] > r['SecondStim'] else 2, axis=1)
df['IsCorrect'] = (df['UserChoice'] == df['StrongerStim']).astype(int)
df['ChoseSecond'] = (df['UserChoice'] == 2).astype(int)

# Force Pair 문자열 생성
df['MinForce'] = df[['Reference', 'Comparison']].min(axis=1)
df['MaxForce'] = df[['Reference', 'Comparison']].max(axis=1)
df['PairLabel'] = df.apply(lambda r: f"{r['MinForce']:.1f} - {r['MaxForce']:.1f}g", axis=1)

# Pair 순서 정의 (Low Band -> High Band)
pair_order = ['0.4 - 1.0g', '0.6 - 1.0g', '1.0 - 1.4g', '1.0 - 2.0g',
              '10.0 - 26.0g', '15.0 - 26.0g', '26.0 - 60.0g']
df = df[df['PairLabel'].isin(pair_order)]

# ---------------------------------------------------------
# 2. 참가자별 / Pair별 지표 집계
# ---------------------------------------------------------
# (A) 제시 순서별 정답률 (S1 > S2 vs S2 > S1)
sub_order_acc = df.groupby(['Subject', 'PairLabel', 'StrongerStim'])['IsCorrect'].mean().reset_index()
sub_order_acc['Accuracy_Pct'] = sub_order_acc['IsCorrect'] * 100

# (B) 2번째 자극 선택 비율 P(Choose 2nd)
sub_choice_bias = df.groupby(['Subject', 'PairLabel'])['ChoseSecond'].mean().reset_index()
sub_choice_bias['ChoseSecond_Pct'] = sub_choice_bias['ChoseSecond'] * 100

# ---------------------------------------------------------
# 3. 피겨 스타일 (Width 2102px)
# ---------------------------------------------------------
dpi = 200
width_px = 2102
height_px_legend = 950      # room for top legend
height_px_nolegend = 870   # tighter canvas when no legend
fig_width_in = width_px / dpi

plt.rcParams['font.sans-serif'] = 'Helvetica'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.major.size'] = 5
plt.rcParams['ytick.major.size'] = 5


# Light gray / dark gray palette (shared across appendix panels)
c_light_box = '#E2E2E2'
c_light_pts = '#666666'
c_dark_box = '#9A9A9A'
c_dark_pts = '#333333'

c_s1_box = c_light_box   # S1 > S2
c_s1_pts = c_light_pts
c_s2_box = c_dark_box    # S2 > S1
c_s2_pts = c_dark_pts
c_choice_box = c_light_box
c_choice_pts = c_light_pts
c_fair_box = c_dark_box
c_fair_pts = c_dark_pts

x_coords = np.arange(len(pair_order))
xticklabels = [p.replace('g', '') for p in pair_order]
out_dir = "/Users/kyungeunjung/NailFoldExp/(New)Analysis/ForceDiscAnalysis/Output/Appendix"
os.makedirs(out_dir, exist_ok=True)


def _style_ax(ax, y_spine_top=100):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='both', direction='in',
                   length=5, width=1.0, labelsize=17)
    # Y-axis line ends at 100 (like ATD Fig2), even if ylim goes higher for labels
    y0 = ax.get_ylim()[0]
    ax.spines['left'].set_bounds(y0, y_spine_top)
    ax.set_yticks([0, 20, 40, 60, 80, 100])


def _add_top_legend(ax, handles, ncol=None):
    """Place legend above the panel (ATD Fig2 style)."""
    ax.legend(
        handles=handles,
        loc='lower center',
        bbox_to_anchor=(0.5, 1.02),
        ncol=ncol or len(handles),
        frameon=False,
        fontsize=14,
        columnspacing=2.0,
        handletextpad=0.5,
        handlelength=1.6,
        borderaxespad=0.0,
    )


def _figsize(has_legend=True):
    h_px = height_px_legend if has_legend else height_px_nolegend
    return (fig_width_in, h_px / dpi), h_px


def _finalize_fig(fig, has_legend=True):
    # Keep top margin for legend; reclaim it when none is drawn
    top = 0.90 if has_legend else 0.97
    fig.subplots_adjust(left=0.10, right=0.97, top=top, bottom=0.14)


# ---------------------------------------------------------
# Panel A: Presentation Order Accuracy  (standalone)
# ---------------------------------------------------------
figsize_a, height_px_a = _figsize(has_legend=True)
fig_a, ax1 = plt.subplots(figsize=figsize_a, dpi=dpi)
box_width = 0.28

for idx, pair in enumerate(pair_order):
    data_s1 = sub_order_acc[(sub_order_acc['PairLabel'] == pair) & (sub_order_acc['StrongerStim'] == 1)]['Accuracy_Pct'].values
    data_s2 = sub_order_acc[(sub_order_acc['PairLabel'] == pair) & (sub_order_acc['StrongerStim'] == 2)]['Accuracy_Pct'].values

    pos_s1 = idx - box_width / 2 - 0.02
    pos_s2 = idx + box_width / 2 + 0.02

    if len(data_s1) > 0:
        ax1.boxplot(data_s1, positions=[pos_s1], widths=box_width, patch_artist=True,
                    boxprops=dict(facecolor=c_s1_box, edgecolor='black', linewidth=1),
                    medianprops=dict(color='red', linewidth=2.0),
                    whiskerprops=dict(color='black', linewidth=1),
                    capprops=dict(color='black', linewidth=1), flierprops=dict(marker=''))
        jit1 = np.random.normal(0, 0.03, size=len(data_s1))
        ax1.scatter(pos_s1 + jit1, data_s1, color=c_s1_pts, alpha=0.5, s=16, zorder=3)

    if len(data_s2) > 0:
        ax1.boxplot(data_s2, positions=[pos_s2], widths=box_width, patch_artist=True,
                    boxprops=dict(facecolor=c_s2_box, edgecolor='black', linewidth=1),
                    medianprops=dict(color='red', linewidth=2.0),
                    whiskerprops=dict(color='black', linewidth=1),
                    capprops=dict(color='black', linewidth=1), flierprops=dict(marker=''))
        jit2 = np.random.normal(0, 0.03, size=len(data_s2))
        ax1.scatter(pos_s2 + jit2, data_s2, color=c_s2_pts, alpha=0.5, s=16, zorder=3)

ax1.axhline(50, color='black', linestyle='--', linewidth=1, alpha=0.7)
ax1.axhline(75, color='gray', linestyle=':', linewidth=1, alpha=0.5)
ax1.set_xticks(x_coords)
ax1.set_xticklabels(xticklabels, fontsize=17)
ax1.set_xlabel('Stimulus Force Pair (g)', fontsize=17, fontweight='bold')
ax1.set_ylabel('Discrimination accuracy (%)', fontsize=17, fontweight='bold')
ax1.set_ylim(-5, 105)
_style_ax(ax1)
_add_top_legend(ax1, [
    Patch(facecolor=c_s1_box, edgecolor='black', label='$S_1 > S_2$ (1st Stronger)'),
    Patch(facecolor=c_s2_box, edgecolor='black', label='$S_2 > S_1$ (2nd Stronger)'),
])
_finalize_fig(fig_a, has_legend=True)

out_a = os.path.join(out_dir, "Appendix_Bias_Order_Accuracy.png")
fig_a.savefig(out_a, dpi=dpi)
plt.close(fig_a)
print(f"Saved → {out_a}  ({width_px}×{height_px_a} px)")


# ---------------------------------------------------------
# Panel B: Response Choice Ratio P(Choose 2nd)  (standalone)
# ---------------------------------------------------------
figsize_b, height_px_b = _figsize(has_legend=False)
fig_b, ax2 = plt.subplots(figsize=figsize_b, dpi=dpi)

for idx, pair in enumerate(pair_order):
    data_choice = sub_choice_bias[sub_choice_bias['PairLabel'] == pair]['ChoseSecond_Pct'].values
    if len(data_choice) > 0:
        ax2.boxplot(data_choice, positions=[idx], widths=0.45, patch_artist=True,
                    boxprops=dict(facecolor=c_choice_box, edgecolor='black', linewidth=1),
                    medianprops=dict(color='red', linewidth=2.0),
                    whiskerprops=dict(color='black', linewidth=1),
                    capprops=dict(color='black', linewidth=1), flierprops=dict(marker=''))
        jit = np.random.normal(0, 0.04, size=len(data_choice))
        ax2.scatter(idx + jit, data_choice, color=c_choice_pts, alpha=0.5, s=16, zorder=3)

ax2.axhline(50, color='black', linestyle='--', linewidth=1, alpha=0.7)
ax2.set_xticks(x_coords)
ax2.set_xticklabels(xticklabels, fontsize=17)
ax2.set_xlabel('Stimulus Force Pair (g)', fontsize=17, fontweight='bold')
ax2.set_ylabel('Proportion choosing \n2nd stimulus (%)', fontsize=17, fontweight='bold')
ax2.set_ylim(-5, 105)
_style_ax(ax2)
_finalize_fig(fig_b, has_legend=False)

out_b = os.path.join(out_dir, "Appendix_Bias_Choice_Bias.png")
fig_b.savefig(out_b, dpi=dpi)
plt.close(fig_b)
print(f"Saved → {out_b}  ({width_px}×{height_px_b} px)")


# ---------------------------------------------------------
# Panel C: Was presentation order delivered fairly?
#   For each force pair, was S1>S2 vs S2>S1 delivered ~50/50?
# ---------------------------------------------------------
order_counts = (
    df.groupby(['Subject', 'PairLabel', 'StrongerStim'])
      .size()
      .unstack(fill_value=0)
      .reindex(columns=[1, 2], fill_value=0)
)
order_counts.columns = ['n_S1stronger', 'n_S2stronger']
order_counts['n_total'] = order_counts['n_S1stronger'] + order_counts['n_S2stronger']
order_counts['pct_S1stronger'] = (
    order_counts['n_S1stronger'] / order_counts['n_total'].clip(lower=1) * 100
)
order_counts = order_counts.reset_index()

overall = (
    df.groupby(['PairLabel', 'StrongerStim']).size()
      .unstack(fill_value=0)
      .reindex(index=pair_order, columns=[1, 2], fill_value=0)
)
overall.columns = ['n_S1stronger', 'n_S2stronger']
overall['n_total'] = overall['n_S1stronger'] + overall['n_S2stronger']
overall['pct_S1stronger'] = overall['n_S1stronger'] / overall['n_total'] * 100

summary_csv = os.path.join(out_dir, "Appendix_Bias_Order_Fairness_summary.csv")
overall.to_csv(summary_csv)
print(f"Saved → {summary_csv}")

figsize_c, height_px_c = _figsize(has_legend=False)
fig_c, ax3 = plt.subplots(figsize=figsize_c, dpi=dpi)

for idx, pair in enumerate(pair_order):
    pcts = order_counts.loc[order_counts['PairLabel'] == pair, 'pct_S1stronger'].values
    if len(pcts) == 0:
        continue
    ax3.boxplot(pcts, positions=[idx], widths=0.45, patch_artist=True,
                boxprops=dict(facecolor=c_fair_box, edgecolor='black', linewidth=1),
                medianprops=dict(color='red', linewidth=2.0),
                whiskerprops=dict(color='black', linewidth=1),
                capprops=dict(color='black', linewidth=1), flierprops=dict(marker=''))
    jit = np.random.normal(0, 0.04, size=len(pcts))
    ax3.scatter(idx + jit, pcts, color=c_fair_pts, alpha=0.5, s=16, zorder=3)

ax3.axhline(50, color='black', linestyle='--', linewidth=1, alpha=0.7)
ax3.set_xticks(x_coords)
ax3.set_xticklabels(xticklabels, fontsize=17)
ax3.set_xlabel('Stimulus Force Pair (g)', fontsize=17, fontweight='bold')
ax3.set_ylabel('Proportion choosing \nstronger stimulus first (%)', fontsize=17, fontweight='bold')
ax3.set_ylim(-5, 105)
_style_ax(ax3)
_finalize_fig(fig_c, has_legend=False)

out_c = os.path.join(out_dir, "Appendix_Bias_Order_Fairness.png")
fig_c.savefig(out_c, dpi=dpi)
plt.close(fig_c)
print(f"Saved → {out_c}  ({width_px}×{height_px_c} px)")

print("\n=== Presentation-order fairness (pooled trial counts) ===")
print(overall.to_string())
print("\nIdeal = 50% stronger-first. Numbers above boxes = pooled %.")
