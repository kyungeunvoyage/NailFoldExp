"""
nail_analysis_plot.py
=====================
손톱 측정 데이터 시각화 스크립트
각 플롯을 개별 PNG 파일로 저장

필요 패키지:
    pip install matplotlib numpy scipy

실행:
    python nail_analysis_plot.py
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

# ── 경로 설정 ─────────────────────────────────────────────────────
CSV_PATH   = '/Users/kyungeunjung/NailFoldExp/nail_measurements.csv'
OUTPUT_DIR = '/Users/kyungeunjung/NailFoldExp/plots/'

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save(fig, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='#0f0f0f')
    print(f"  저장 → {path}")
    plt.close(fig)

# ── 데이터 로드 & 정제 ────────────────────────────────────────────
rows = []
with open(CSV_PATH, encoding='utf-8-sig') as f:
    for row in csv.DictReader(f):
        rows.append(row)

seen, clean = set(), []
for r in rows:
    if r['subject_id'] not in seen:
        seen.add(r['subject_id'])
        clean.append(r)

outliers = [r['subject_id'] for r in clean
            if float(r['nail_width_px']) < 50 or float(r['nail_height_px']) < 50]
valid = [r for r in clean
         if float(r['nail_width_px']) >= 50 and float(r['nail_height_px']) >= 50]

print(f"전체: {len(clean)}명 | Outlier 제거: {outliers} | 분석 대상: {len(valid)}명\n")

ids = [r['subject_id'] for r in valid]
W   = np.array([float(r['nail_width_px'])  for r in valid])
H   = np.array([float(r['nail_height_px']) for r in valid])
L   = np.array([float(r['left_skin_px'])   for r in valid])
R   = np.array([float(r['right_skin_px'])  for r in valid])
WH  = np.array([float(r['ratio_w_h'])      for r in valid])
LW  = np.array([float(r['ratio_left_w'])   for r in valid])
RW  = np.array([float(r['ratio_right_w'])  for r in valid])
n   = len(valid)

# ── 손톱 모양 분류 ────────────────────────────────────────────────
def classify(wh):
    if wh < 0.85:   return 'Oval/Long'
    elif wh < 1.05: return 'Square-ish'
    else:           return 'Wide/Square'

shapes       = [classify(wh) for wh in WH]
shape_cats   = ['Oval/Long', 'Square-ish', 'Wide/Square']
shape_colors = {'Oval/Long': '#7eb8f7', 'Square-ish': '#f7c07e', 'Wide/Square': '#f77e7e'}
pt_colors    = np.array([shape_colors[s] for s in shapes])

# ── 공통 스타일 ───────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0f0f0f',
    'axes.facecolor':   '#1a1a1a',
    'axes.edgecolor':   '#333',
    'axes.labelcolor':  '#ccc',
    'xtick.color':      '#888',
    'ytick.color':      '#888',
    'text.color':       '#ddd',
    'grid.color':       '#2a2a2a',
    'grid.linewidth':   0.6,
    'font.family':      'monospace',
    'axes.titlesize':   12,
    'axes.labelsize':   10,
    'xtick.labelsize':  9,
    'ytick.labelsize':  9,
})

ACCENT = '#c8f050'
BLUE   = '#50c8f0'
PINK   = '#f05090'
ORANGE = '#f0a050'

outlier_str = ', '.join(outliers) if outliers else 'None'

# ═════════════════════════════════════════════════════════════════
# Plot 1: Raw 측정값 Mean ± SD Bar
# ═════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 5))
labels = ['Nail Width (px)', 'Nail Height (px)', 'Left Skin (px)', 'Right Skin (px)']
means  = [W.mean(), H.mean(), L.mean(), R.mean()]
stds   = [W.std(),  H.std(),  L.std(),  R.std()]
colors = [ACCENT, BLUE, PINK, ORANGE]
x      = np.arange(len(labels))

bars = ax.bar(x, means, yerr=stds, color=colors, alpha=0.85,
              error_kw=dict(ecolor='white', elinewidth=1.5, capsize=7, capthick=1.5),
              width=0.55, zorder=3)
for bar, m, s in zip(bars, means, stds):
    ax.text(bar.get_x() + bar.get_width() / 2, m + s + 1.5,
            f'{m:.1f}', ha='center', va='bottom', fontsize=10,
            color='white', fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel('pixels')
ax.set_title(f'Mean +/- SD  —  Raw Measurements (px)\n(n={n}, outliers removed: {outlier_str})',
             color=ACCENT, pad=10)
ax.grid(axis='y', zorder=0)
ax.set_ylim(0, max(means) + max(stds) + 15)
save(fig, '01_mean_sd_raw_px.png')

# ═════════════════════════════════════════════════════════════════
# Plot 2: Ratio Mean ± SD Bar
# ═════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(6, 5))
rlabels = ['W : H', 'Left / W', 'Right / W']
rmeans  = [WH.mean(), LW.mean(), RW.mean()]
rstds   = [WH.std(),  LW.std(),  RW.std()]
rcolors = [ACCENT, PINK, ORANGE]
xr      = np.arange(3)

rbars = ax.bar(xr, rmeans, yerr=rstds, color=rcolors, alpha=0.85,
               error_kw=dict(ecolor='white', elinewidth=1.5, capsize=7, capthick=1.5),
               width=0.5, zorder=3)
for bar, m, s in zip(rbars, rmeans, rstds):
    ax.text(bar.get_x() + bar.get_width() / 2, m + s + 0.012,
            f'{m:.3f}', ha='center', va='bottom', fontsize=10,
            color='white', fontweight='bold')

ax.set_xticks(xr)
ax.set_xticklabels(rlabels)
ax.set_ylabel('ratio')
ax.set_title(f'Mean +/- SD  —  Ratios\n(n={n}, outliers removed: {outlier_str})',
             color=ACCENT, pad=10)
ax.grid(axis='y', zorder=0)
ax.axhline(1.0, color='white', lw=0.8, ls='--', alpha=0.3)
save(fig, '02_mean_sd_ratios.png')

# ═════════════════════════════════════════════════════════════════
# Plot 3: Width vs Height scatter
# ═════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(6, 6))
for s in shape_cats:
    mask = [sh == s for sh in shapes]
    ax.scatter(W[mask], H[mask], c=shape_colors[s], s=65, alpha=0.85,
               edgecolors='white', linewidths=0.4, label=s, zorder=3)

lim_min = min(W.min(), H.min()) - 5
lim_max = max(W.max(), H.max()) + 5
ax.plot([lim_min, lim_max], [lim_min, lim_max], '--', color='white', alpha=0.25, lw=1)
ax.set_xlabel('Nail Width (px)')
ax.set_ylabel('Nail Height (px)')
ax.set_title(f'Width vs Height\n(diagonal = W:H=1, n={n})', color=ACCENT, pad=10)
ax.legend(fontsize=8, facecolor='#1a1a1a', edgecolor='#444')
ax.grid(zorder=0)
save(fig, '03_width_vs_height_scatter.png')

# ═════════════════════════════════════════════════════════════════
# Plot 4: W:H 비율 히스토그램 + KDE
# ═════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 5))
counts, bins, patches_hist = ax.hist(WH, bins=14, alpha=0.75,
                                      edgecolor='white', linewidth=0.5,
                                      density=True, zorder=3)
for patch, left in zip(patches_hist, bins):
    mid = left + (bins[1] - bins[0]) / 2
    patch.set_facecolor(shape_colors[classify(mid)])
    patch.set_alpha(0.8)

kde_x = np.linspace(WH.min() - 0.1, WH.max() + 0.1, 200)
kde   = stats.gaussian_kde(WH)
ax.plot(kde_x, kde(kde_x), color='white', lw=1.8, zorder=4)
ax.axvline(WH.mean(),     color=ACCENT, lw=1.8, ls='--', label=f'Mean {WH.mean():.3f}')
ax.axvline(np.median(WH), color=ORANGE, lw=1.8, ls=':',  label=f'Median {np.median(WH):.3f}')
ax.set_xlabel('W : H ratio')
ax.set_ylabel('density')
ax.set_title(f'W:H Ratio Distribution  (n={n})', color=ACCENT, pad=10)
ax.legend(fontsize=8, facecolor='#1a1a1a', edgecolor='#444')
ax.grid(zorder=0)
save(fig, '04_wh_ratio_distribution.png')

# ═════════════════════════════════════════════════════════════════
# Plot 5: Left vs Right 피부띠 scatter
# ═════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(L, R, c=pt_colors, s=65, alpha=0.85,
           edgecolors='white', linewidths=0.4, zorder=3)
lim = max(L.max(), R.max()) + 3
ax.plot([0, lim], [0, lim], '--', color='white', alpha=0.25, lw=1)
ax.set_xlabel('Left Skin Band (px)')
ax.set_ylabel('Right Skin Band (px)')
ax.set_title(f'Left vs Right Skin Band\n(diagonal = symmetric, n={n})', color=ACCENT, pad=10)
ax.grid(zorder=0)
patches_leg = [mpatches.Patch(color=v, label=k) for k, v in shape_colors.items()]
ax.legend(handles=patches_leg, fontsize=8, facecolor='#1a1a1a', edgecolor='#444')
save(fig, '05_left_vs_right_skin_scatter.png')

# ═════════════════════════════════════════════════════════════════
# Plot 6: 손톱 모양 분류 Pie
# ═════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(6, 6))
shape_counts = {s: shapes.count(s) for s in shape_cats}
wedges, texts, autotexts = ax.pie(
    [shape_counts[s] for s in shape_cats],
    labels=[f"{s}\n(n={shape_counts[s]})" for s in shape_cats],
    colors=[shape_colors[s] for s in shape_cats],
    autopct='%1.1f%%',
    startangle=140,
    textprops={'fontsize': 9},
    wedgeprops={'edgecolor': '#0f0f0f', 'linewidth': 2}
)
for at in autotexts:
    at.set_color('black')
    at.set_fontweight('bold')
ax.set_title(f'Nail Shape Classification  (n={n})\nOval < 0.85  |  Square-ish 0.85–1.05  |  Wide > 1.05',
             color=ACCENT, pad=12, fontsize=11)
save(fig, '06_nail_shape_pie.png')

# ═════════════════════════════════════════════════════════════════
# Plot 7: 피부띠 비율 Boxplot
# ═════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(6, 5))
bp = ax.boxplot([LW, RW], patch_artist=True,
                medianprops=dict(color='white', linewidth=2),
                whiskerprops=dict(color='#888'),
                capprops=dict(color='#888'),
                flierprops=dict(marker='o', color='#888', markersize=5))
bp['boxes'][0].set_facecolor(PINK);   bp['boxes'][0].set_alpha(0.75)
bp['boxes'][1].set_facecolor(ORANGE); bp['boxes'][1].set_alpha(0.75)
ax.set_xticklabels(['Left Skin / W', 'Right Skin / W'])
ax.set_ylabel('ratio')
ax.set_title(f'Skin Band Ratio Distribution  (n={n})', color=ACCENT, pad=10)
ax.grid(axis='y', zorder=0)
ax.axhline(LW.mean(), color=PINK,   lw=1.2, ls='--', alpha=0.7, label=f'L mean {LW.mean():.3f}')
ax.axhline(RW.mean(), color=ORANGE, lw=1.2, ls='--', alpha=0.7, label=f'R mean {RW.mean():.3f}')
ax.legend(fontsize=8, facecolor='#1a1a1a', edgecolor='#444')
save(fig, '07_skin_band_ratio_boxplot.png')

# ═════════════════════════════════════════════════════════════════
# Plot 8: 좌우 비대칭도 (L-R) histogram
# ═════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 5))
asym = L - R
ax.hist(asym, bins=12, color=ACCENT, alpha=0.75,
        edgecolor='white', linewidth=0.5, zorder=3)
ax.axvline(0,            color='white',  lw=1.2, ls='--', alpha=0.5, label='Symmetric (0)')
ax.axvline(asym.mean(),  color=ORANGE, lw=1.8, ls='--', label=f'Mean {asym.mean():.1f} px')
ax.set_xlabel('Left - Right Skin Band (px)')
ax.set_ylabel('count')
ax.set_title(f'Lateral Asymmetry  (L - R)  (n={n})\nPositive = Left skin band wider',
             color=ACCENT, pad=10)
ax.legend(fontsize=8, facecolor='#1a1a1a', edgecolor='#444')
ax.grid(zorder=0)
save(fig, '08_lateral_asymmetry.png')

# ═════════════════════════════════════════════════════════════════
# Plot 9: 개인별 W:H Strip plot
# ═════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(18, 5))
sorted_idx       = np.argsort(WH)
wh_sorted        = WH[sorted_idx]
ids_sorted       = [ids[i] for i in sorted_idx]
pt_colors_sorted = [shape_colors[shapes[i]] for i in sorted_idx]

ax.scatter(range(n), wh_sorted, c=pt_colors_sorted, s=70,
           edgecolors='white', linewidths=0.5, zorder=3)
ax.axhline(WH.mean(),     color=ACCENT, lw=1.5, ls='--', zorder=4, label=f'Mean {WH.mean():.3f}')
ax.axhline(np.median(WH), color=ORANGE, lw=1.2, ls=':',  zorder=4, label=f'Median {np.median(WH):.3f}')
ax.axhline(0.85, color='#7eb8f7', lw=0.8, ls=':', alpha=0.5)
ax.axhline(1.05, color='#f77e7e', lw=0.8, ls=':', alpha=0.5)
ax.fill_between(range(n), 0.85, 1.05, alpha=0.06, color='white')
ax.set_xticks(range(n))
ax.set_xticklabels(ids_sorted, rotation=90, fontsize=8)
ax.set_ylabel('W : H ratio')
ax.set_title(f'Individual W:H Ratio  —  sorted  (shaded band = Square-ish zone, n={n})',
             color=ACCENT, pad=10)
ax.grid(zorder=0)
patches_leg2 = [mpatches.Patch(color=v, label=k) for k, v in shape_colors.items()]
patches_leg2 += [
    mpatches.Patch(color=ACCENT, label=f'Mean {WH.mean():.3f}'),
    mpatches.Patch(color=ORANGE, label=f'Median {np.median(WH):.3f}'),
]
ax.legend(handles=patches_leg2, fontsize=8, facecolor='#1a1a1a', edgecolor='#444', ncol=3)
save(fig, '09_individual_wh_strip.png')

print(f"\n완료! 총 9개 파일 → {OUTPUT_DIR}")