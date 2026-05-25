"""
corr_1_overall.py
=================
ATD 전체 평균 Relative_Score × 손톱 비율 (W:H, L/W, R/W) Correlation

실행: python corr_1_overall.py
"""

import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# ── 경로 ──────────────────────────────────────────────────────────
NAIL_CSV    = '/Users/kyungeunjung/NailFoldExp/nail_measurements.csv'
ATD_PATTERN = '/Users/kyungeunjung/NailFoldExp/Data/(ATD)CurData/P*_AbsoluteThresholdDetection.csv'
OUTPUT_DIR  = '/Users/kyungeunjung/NailFoldExp/plots/correlation/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

NAIL_VARS   = ['ratio_w_h', 'ratio_left_w', 'ratio_right_w']
NAIL_LABELS = {'ratio_w_h': 'W : H', 'ratio_left_w': 'Left / W', 'ratio_right_w': 'Right / W'}

# ── 스타일 ────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor':'#0f0f0f','axes.facecolor':'#1a1a1a',
    'axes.edgecolor':'#333','axes.labelcolor':'#ccc',
    'xtick.color':'#888','ytick.color':'#888','text.color':'#ddd',
    'grid.color':'#2a2a2a','grid.linewidth':0.6,'font.family':'monospace',
    'axes.titlesize':11,'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,
})
ACCENT='#c8f050'; BLUE='#50c8f0'; PINK='#f05090'; ORANGE='#f0a050'

def normalize_pid(pid):
    num = ''.join(filter(str.isdigit, str(pid).strip()))
    return f'P{int(num)}' if num else str(pid)

def star(p):
    return '***' if p<0.001 else ('**' if p<0.01 else ('*' if p<0.05 else 'ns'))

# ── 1. ATD 로드 & Relative_Score 계산 ────────────────────────────
all_files = glob.glob(ATD_PATTERN)
if not all_files:
    raise FileNotFoundError(f"파일 없음: {ATD_PATTERN}")

df_list = []
for f in all_files:
    tmp = pd.read_csv(f)
    if 'SubjectID' not in tmp.columns:
        tmp['SubjectID'] = os.path.basename(f).split('_')[0]
    df_list.append(tmp)

df = pd.concat(df_list, ignore_index=True)
df['Condition'] = df['Condition'].str.strip().replace({'Active':'On-touch (Mid)','On-touch (Hard)':'On-touch (Mid)'})
df = df[df['Condition'] != 'On-touch (Soft)']
df = df[df['Area'].isin(['A','B','C','D','E','F'])].copy()
df['Force_Val'] = df['Force'].str.extract(r'(\d+\.?\d*)').astype(float)

def calc_score(row):
    if row['Target'] == 0:
        return 100 if row['Response'] == 0 else 0
    return max(0, (1 - abs(row['Target']-row['Response'])/row['Target']) * 100)

df['Relative_Score'] = df.apply(calc_score, axis=1)
df['SubjectID'] = df['SubjectID'].apply(normalize_pid)

# 피험자별 전체 평균
atd_overall = (df.groupby('SubjectID')['Relative_Score']
               .mean().reset_index()
               .rename(columns={'Relative_Score':'score_overall'}))

# ── 2. Nail 로드 ──────────────────────────────────────────────────
df_nail = pd.read_csv(NAIL_CSV, encoding='utf-8-sig')
df_nail['subject_id'] = df_nail['subject_id'].apply(normalize_pid)
df_nail = df_nail.drop_duplicates('subject_id', keep='first')
df_nail = df_nail[(df_nail['nail_width_px']>=50) & (df_nail['nail_height_px']>=50)]

# ── 3. 병합 ───────────────────────────────────────────────────────
df_m = atd_overall.merge(df_nail[['subject_id']+NAIL_VARS],
                          left_on='SubjectID', right_on='subject_id', how='inner')
n = len(df_m)
print(f"병합 완료: {n}명 → {sorted(df_m['SubjectID'].tolist())}")

# ══════════════════════════════════════════════════════════════════
# Plot A: Scatter (1×3) — 전체평균 score × 각 nail 비율
# ══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(12, 4))  # 1×3 scatter

for ax, nv in zip(axes, NAIL_VARS):
    tmp = df_m[['score_overall', nv, 'SubjectID']].dropna()
    x, y = tmp[nv].values, tmp['score_overall'].values
    r_p, p_p = stats.pearsonr(x, y)
    r_s, p_s = stats.spearmanr(x, y)

    color = ACCENT if p_p < 0.05 else BLUE
    ax.scatter(x, y, color=color, s=70, alpha=0.85, edgecolors='white', linewidths=0.5, zorder=3)

    # 회귀선 + 95% CI
    m, b = np.polyfit(x, y, 1)
    xline = np.linspace(x.min(), x.max(), 100)
    ax.plot(xline, m*xline+b, color='white', lw=1.5, alpha=0.6, zorder=4)

    # 피험자 ID
    for _, row in tmp.iterrows():
        ax.annotate(row['SubjectID'], (row[nv], row['score_overall']),
                    fontsize=5.5, color='#999', xytext=(3,3), textcoords='offset points')

    s_label = star(p_p)
    title_color = ACCENT if p_p < 0.05 else '#ccc'
    ax.set_title(f'{NAIL_LABELS[nv]}\nPearson r={r_p:+.3f} {s_label}   Spearman ρ={r_s:+.3f} {star(p_s)}',
                 color=title_color, pad=10)
    ax.set_xlabel(NAIL_LABELS[nv])
    ax.set_ylabel('Overall Relative Score (%)')
    ax.grid(zorder=0)

fig.suptitle(f'[1] ATD Overall Mean Score × Nail Ratios  (n={n})\nGreen = significant (p<.05)',
             color=ACCENT, fontsize=13, y=1.03)
plt.tight_layout()
path = os.path.join(OUTPUT_DIR, '1_overall_scatter.png')
fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='#0f0f0f')
print(f"저장 → {path}")
plt.show()

# ══════════════════════════════════════════════════════════════════
# Plot B: Heatmap — 전체평균 score × nail 비율 (r + p값)
# ══════════════════════════════════════════════════════════════════
r_vals, annot_vals = [], []
for nv in NAIL_VARS:
    tmp = df_m[['score_overall', nv]].dropna()
    r_p, p_p = stats.pearsonr(tmp['score_overall'], tmp[nv])
    r_vals.append(r_p)
    s = star(p_p) if p_p < 0.05 else ''
    annot_vals.append(f'{r_p:+.3f}\n{s}')

r_mat   = pd.DataFrame([r_vals],   columns=[NAIL_LABELS[v] for v in NAIL_VARS], index=['Overall Score'])
ann_mat = pd.DataFrame([annot_vals], columns=[NAIL_LABELS[v] for v in NAIL_VARS], index=['Overall Score'])

fig, ax = plt.subplots(figsize=(12, 8))  # heatmap
sns.heatmap(r_mat.astype(float), annot=ann_mat, fmt='', center=0, vmin=-1, vmax=1,
            cmap='RdBu_r', linewidths=1, linecolor='#333',
            annot_kws={'size':13, 'color':'white', 'fontweight':'bold'},
            ax=ax, cbar_kws={'label':'Pearson r','shrink':0.8})
ax.set_title(f'Pearson r — Overall Score × Nail Ratios  (n={n}, * p<.05  ** p<.01  *** p<.001)',
             color=ACCENT, pad=10)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

path2 = os.path.join(OUTPUT_DIR, '1_overall_heatmap.png')
fig.savefig(path2, dpi=150, bbox_inches='tight', facecolor='#0f0f0f')
print(f"저장 → {path2}")
plt.show()
