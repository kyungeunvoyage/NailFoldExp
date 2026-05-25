"""
corr_3_by_region.py
===================
Region(A~F)별 Relative_Score × 손톱 비율 (W:H, L/W, R/W) Correlation

실행: python corr_3_by_region.py
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
NAIL_LABELS = {'ratio_w_h':'W : H', 'ratio_left_w':'Left / W', 'ratio_right_w':'Right / W'}
REGION_LIST = ['A','B','C','D','E','F']

plt.rcParams.update({
    'figure.facecolor':'#0f0f0f','axes.facecolor':'#1a1a1a',
    'axes.edgecolor':'#333','axes.labelcolor':'#ccc',
    'xtick.color':'#888','ytick.color':'#888','text.color':'#ddd',
    'grid.color':'#2a2a2a','grid.linewidth':0.6,'font.family':'monospace',
    'axes.titlesize':10,'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,
})
ACCENT='#c8f050'; BLUE='#50c8f0'

def normalize_pid(pid):
    num = ''.join(filter(str.isdigit, str(pid).strip()))
    return f'P{int(num)}' if num else str(pid)

def star(p):
    return '***' if p<0.001 else ('**' if p<0.01 else ('*' if p<0.05 else 'ns'))

# ── ATD 로드 ──────────────────────────────────────────────────────
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
df = df[df['Area'].isin(REGION_LIST)].copy()
df['Force_Val'] = df['Force'].str.extract(r'(\d+\.?\d*)').astype(float)

def calc_score(row):
    if row['Target'] == 0:
        return 100 if row['Response'] == 0 else 0
    return max(0, (1 - abs(row['Target']-row['Response'])/row['Target']) * 100)

df['Relative_Score'] = df.apply(calc_score, axis=1)
df['SubjectID'] = df['SubjectID'].apply(normalize_pid)
df['Region'] = df['Area']

# Region별 피험자 평균
atd_region = (df.groupby(['SubjectID','Region'])['Relative_Score']
              .mean().reset_index()
              .rename(columns={'Relative_Score':'score'}))

# ── Nail 로드 ─────────────────────────────────────────────────────
df_nail = pd.read_csv(NAIL_CSV, encoding='utf-8-sig')
df_nail['subject_id'] = df_nail['subject_id'].apply(normalize_pid)
df_nail = df_nail.drop_duplicates('subject_id', keep='first')
df_nail = df_nail[(df_nail['nail_width_px']>=50) & (df_nail['nail_height_px']>=50)]

# 병합
df_m = atd_region.merge(df_nail[['subject_id']+NAIL_VARS],
                         left_on='SubjectID', right_on='subject_id', how='inner')
n_subjects = df_m['SubjectID'].nunique()
regions_found = sorted(df_m['Region'].unique())
print(f"병합 완료: {n_subjects}명 | Region: {regions_found}")

# ══════════════════════════════════════════════════════════════════
# Plot A: Scatter grid (Region × nail var)
# ══════════════════════════════════════════════════════════════════
n_regions = len(regions_found)
n_nails   = len(NAIL_VARS)

fig, axes = plt.subplots(n_regions, n_nails,
                          figsize=(5*n_nails, 4*n_regions),  # 행/열 수에 따라 자동
                          squeeze=False)

for row_i, reg in enumerate(regions_found):
    sub = df_m[df_m['Region'] == reg]
    n   = sub['SubjectID'].nunique()

    for col_j, nv in enumerate(NAIL_VARS):
        ax  = axes[row_i][col_j]
        tmp = sub[['score', nv, 'SubjectID']].dropna()
        if len(tmp) < 4:
            ax.set_visible(False)
            continue

        x, y     = tmp[nv].values, tmp['score'].values
        r_p, p_p = stats.pearsonr(x, y)
        r_s, p_s = stats.spearmanr(x, y)
        color    = ACCENT if p_p < 0.05 else BLUE

        ax.scatter(x, y, color=color, s=60, alpha=0.85,
                   edgecolors='white', linewidths=0.4, zorder=3)
        m, b   = np.polyfit(x, y, 1)
        xline  = np.linspace(x.min(), x.max(), 100)
        ax.plot(xline, m*xline+b, color='white', lw=1.2, alpha=0.5, zorder=4)

        for _, row in tmp.iterrows():
            ax.annotate(row['SubjectID'], (row[nv], row['score']),
                        fontsize=5, color='#999',
                        xytext=(2,2), textcoords='offset points')

        tc = ACCENT if p_p < 0.05 else '#ccc'
        ax.set_title(f'Region {reg}  ×  {NAIL_LABELS[nv]}\nr={r_p:+.3f} {star(p_p)}  ρ={r_s:+.3f} {star(p_s)}',
                     color=tc, fontsize=9, pad=6)
        ax.set_xlabel(NAIL_LABELS[nv], fontsize=8)
        ax.set_ylabel(f'Score @ Region {reg} (%)', fontsize=8)
        ax.grid(zorder=0)

fig.suptitle(f'[3] ATD Score by Region × Nail Ratios  (n={n_subjects})\nGreen = significant (p<.05)',
             color=ACCENT, fontsize=13, y=1.005)
plt.tight_layout()
path = os.path.join(OUTPUT_DIR, '3_region_scatter_grid.png')
fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='#0f0f0f')
print(f"저장 → {path}")
plt.show()

# ══════════════════════════════════════════════════════════════════
# Plot B: Heatmap — Region × nail var (r값)
# ══════════════════════════════════════════════════════════════════
r_mat   = pd.DataFrame(index=regions_found,
                        columns=[NAIL_LABELS[v] for v in NAIL_VARS], dtype=float)
ann_mat = pd.DataFrame(index=r_mat.index, columns=r_mat.columns, dtype=object)

for reg in regions_found:
    sub = df_m[df_m['Region'] == reg]
    for nv in NAIL_VARS:
        tmp = sub[['score', nv]].dropna()
        if len(tmp) < 4:
            r_mat.loc[reg, NAIL_LABELS[nv]]   = np.nan
            ann_mat.loc[reg, NAIL_LABELS[nv]] = ''
            continue
        r_p, p_p = stats.pearsonr(tmp['score'], tmp[nv])
        r_mat.loc[reg, NAIL_LABELS[nv]]   = round(r_p, 3)
        s = star(p_p) if p_p < 0.05 else ''
        ann_mat.loc[reg, NAIL_LABELS[nv]] = f'{r_p:+.3f}\n{s}'

fig, ax = plt.subplots(figsize=(12, 8))  # heatmap
sns.heatmap(r_mat.astype(float), annot=ann_mat, fmt='', center=0, vmin=-1, vmax=1,
            cmap='RdBu_r', linewidths=0.6, linecolor='#333',
            annot_kws={'size':11, 'color':'white', 'fontweight':'bold'},
            ax=ax, cbar_kws={'label':'Pearson r','shrink':0.8})
ax.set_title(f'Pearson r — Score by Region × Nail Ratios  (n={n_subjects})\n* p<.05  ** p<.01  *** p<.001',
             color=ACCENT, pad=10)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
ax.set_xticklabels(ax.get_xticklabels(), rotation=15)

path2 = os.path.join(OUTPUT_DIR, '3_region_heatmap.png')
fig.savefig(path2, dpi=150, bbox_inches='tight', facecolor='#0f0f0f')
print(f"저장 → {path2}")
plt.show()

# ══════════════════════════════════════════════════════════════════
# Plot C: r값 라인 플롯 — Region에 따른 correlation 변화
# ══════════════════════════════════════════════════════════════════
nail_colors = {'ratio_w_h': ACCENT, 'ratio_left_w': '#f05090', 'ratio_right_w': '#f0a050'}
x_pos = np.arange(len(regions_found))

fig, ax = plt.subplots(figsize=(12, 8))  # lineplot
for nv in NAIL_VARS:
    r_vals, p_vals = [], []
    for reg in regions_found:
        sub = df_m[df_m['Region'] == reg][['score', nv]].dropna()
        if len(sub) >= 4:
            r_p, p_p = stats.pearsonr(sub['score'], sub[nv])
            r_vals.append(r_p); p_vals.append(p_p)
        else:
            r_vals.append(np.nan); p_vals.append(np.nan)

    c = nail_colors[nv]
    ax.plot(x_pos, r_vals, color=c, lw=2, marker='o', markersize=8,
            label=NAIL_LABELS[nv], zorder=3)

    for xi, (r, p) in enumerate(zip(r_vals, p_vals)):
        if p is not None and not np.isnan(r) and p < 0.05:
            ax.scatter(xi, r, color=c, s=160, zorder=5,
                       edgecolors='white', linewidths=1.5)
            ax.annotate(star(p), (xi, r), xytext=(0,10),
                        textcoords='offset points', ha='center',
                        fontsize=10, color=c, fontweight='bold')

ax.axhline(0, color='white', lw=0.8, ls='--', alpha=0.3)
ax.axhline(0.3,  color='white', lw=0.5, ls=':', alpha=0.2)
ax.axhline(-0.3, color='white', lw=0.5, ls=':', alpha=0.2)
ax.set_xticks(x_pos)
ax.set_xticklabels([f'Region {r}' for r in regions_found])
ax.set_ylabel('Pearson r')
ax.set_title(f'[3] Correlation Change by Region  (n={n_subjects})\nLarge dot = significant (p<.05)',
             color=ACCENT, pad=10)
ax.legend(fontsize=9, facecolor='#1a1a1a', edgecolor='#444')
ax.set_ylim(-1, 1)
ax.grid(zorder=0)

path3 = os.path.join(OUTPUT_DIR, '3_region_r_lineplot.png')
fig.savefig(path3, dpi=150, bbox_inches='tight', facecolor='#0f0f0f')
print(f"저장 → {path3}")
plt.show()
