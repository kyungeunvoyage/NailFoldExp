"""
All-participants Position Bias Figures

Figure 1: Per-participant summary scatter
  X = % chose 1st when 1st stim was stronger
  Y = % chose 1st when 2nd stim was stronger
  → ideal unbiased: top-left quadrant (low x, high y would be 2nd bias;
    bottom-right would be 1st bias; diagonal = no position bias)
  → Actually: x high & y low = correct discriminator (chose stronger)
              x high & y high = 1st interval bias
              x low  & y low  = 2nd interval bias
              x low  & y high = inverted/confused

Figure 2: Participant × Pair heatmap
  Color = % chose 1st when 1st was stronger MINUS % chose 1st when 2nd was stronger
  → positive (red) = 1st-leaning correct discriminator
  → near zero (white) = position bias (always same choice)
  → negative (blue) = reversed
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm

# ── Load data ─────────────────────────────────────────────────────────────────
files = sorted(glob.glob('/Users/kyungeunjung/NailFoldExp/Data/(FD)CurData/P*.csv'))
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
df['StrongerInterval'] = np.where(df['FirstStim'] > df['SecondStim'], 1, 2)

subjects = sorted(df['Subject'].unique())
pairs    = [(1, 0.4), (1, 0.6), (1, 1.4), (1, 2.0),
            (26, 10), (26, 15), (26, 60)]
pair_labels = [f'Ref={r}N\nComp={c}N' for r, c in pairs]

# ── Compute per-subject per-pair stats ────────────────────────────────────────
records = []
for subj in subjects:
    sd = df[df['Subject'] == subj]
    for ref, comp in pairs:
        grp = sd[(sd['Reference'] == ref) & (sd['Comparison'] == comp)]

        g1 = grp[grp['StrongerInterval'] == 1]
        g2 = grp[grp['StrongerInterval'] == 2]

        p1_when1st = g1['UserChoice'].eq(1).mean() * 100 if len(g1) > 0 else np.nan
        p1_when2nd = g2['UserChoice'].eq(1).mean() * 100 if len(g2) > 0 else np.nan

        records.append({
            'Subject': subj, 'Reference': ref, 'Comparison': comp,
            'pct_chose1_when1st_stronger': p1_when1st,
            'pct_chose1_when2nd_stronger': p1_when2nd,
            'n_1st_stronger': len(g1),
            'n_2nd_stronger': len(g2),
        })

res = pd.DataFrame(records)

# Overall per-subject (collapsed across pairs)
overall_rows = []
for subj in subjects:
    g = df[df['Subject'] == subj]
    g1 = g[g['StrongerInterval'] == 1]
    g2 = g[g['StrongerInterval'] == 2]
    overall_rows.append({
        'Subject': subj,
        'overall_p1_when1st': g1['UserChoice'].eq(1).mean() * 100,
        'overall_p1_when2nd': g2['UserChoice'].eq(1).mean() * 100,
    })
overall = pd.DataFrame(overall_rows)

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 – Scatter: % chose 1st | 1st stronger  vs  % chose 1st | 2nd stronger
# ═══════════════════════════════════════════════════════════════════════════════
fig1, ax = plt.subplots(figsize=(7, 7))

# quadrant shading
ax.axvspan(50, 100, ymin=0,   ymax=0.5, alpha=0.04, color='blue')   # 1st bias
ax.axvspan(50, 100, ymin=0.5, ymax=1,   alpha=0.06, color='green')  # correct discriminator
ax.axvspan(0,  50,  ymin=0.5, ymax=1,   alpha=0.04, color='orange') # 2nd bias
ax.axvspan(0,  50,  ymin=0,   ymax=0.5, alpha=0.04, color='red')    # inverted

ax.axhline(50, color='gray', lw=1, ls='--', alpha=0.6)
ax.axvline(50, color='gray', lw=1, ls='--', alpha=0.6)
ax.plot([0, 100], [100, 0], color='gray', lw=0.8, ls=':', alpha=0.5)  # anti-diagonal

sc = ax.scatter(overall['overall_p1_when1st'], overall['overall_p1_when2nd'],
                s=90, c='#1976D2', edgecolors='white', linewidths=1, zorder=5)

for _, row in overall.iterrows():
    ax.annotate(row['Subject'],
                (row['overall_p1_when1st'], row['overall_p1_when2nd']),
                textcoords='offset points', xytext=(5, 3),
                fontsize=7.5, color='#333')

# Quadrant labels
ax.text(75, 95, 'Correct\nDiscriminator', ha='center', fontsize=9,
        color='green', fontweight='bold', alpha=0.7)
ax.text(75, 5,  '1st Interval\nBias', ha='center', fontsize=9,
        color='blue', fontweight='bold', alpha=0.7)
ax.text(25, 95, '2nd Interval\nBias', ha='center', fontsize=9,
        color='darkorange', fontweight='bold', alpha=0.7)
ax.text(25, 5,  'Inverted\n(confused)', ha='center', fontsize=9,
        color='red', fontweight='bold', alpha=0.7)

ax.set_xlabel('% chose 1st interval  |  when 1st stim was STRONGER', fontsize=11)
ax.set_ylabel('% chose 1st interval  |  when 2nd stim was STRONGER', fontsize=11)
ax.set_title('All Participants – Position Bias Overview\n(each dot = one participant, overall across all pairs)',
             fontsize=12, fontweight='bold')
ax.set_xlim(0, 100); ax.set_ylim(0, 100)
ax.set_xticks(range(0, 101, 25)); ax.set_yticks(range(0, 101, 25))
ax.spines[['top', 'right']].set_visible(False)
ax.grid(True, linestyle=':', alpha=0.3)

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 – Heatmap: discrimination index per subject × pair
# ═══════════════════════════════════════════════════════════════════════════════
# discrimination index = % chose 1st when 1st stronger  −  % chose 1st when 2nd stronger
# +100 = perfect discriminator, 0 = pure position bias, −100 = inverted
res['disc_index'] = res['pct_chose1_when1st_stronger'] - res['pct_chose1_when2nd_stronger']

hmap_data = res.pivot(index='Subject', columns=['Reference', 'Comparison'], values='disc_index')
hmap_data.columns = [f'Ref={r}N\nComp={c}N' for r, c in hmap_data.columns]
hmap_data = hmap_data.reindex(subjects)

fig2, ax2 = plt.subplots(figsize=(12, 9))
norm = TwoSlopeNorm(vmin=-100, vcenter=0, vmax=100)
im = ax2.imshow(hmap_data.values, cmap='RdYlGn', norm=norm, aspect='auto')

# Annotate cells
for i in range(len(subjects)):
    for j in range(len(pair_labels)):
        val = hmap_data.values[i, j]
        if not np.isnan(val):
            ax2.text(j, i, f'{val:.0f}', ha='center', va='center',
                     fontsize=8.5, fontweight='bold',
                     color='white' if abs(val) > 60 else '#222')

ax2.set_xticks(range(len(pair_labels)))
ax2.set_xticklabels(pair_labels, fontsize=9)
ax2.set_yticks(range(len(subjects)))
ax2.set_yticklabels(subjects, fontsize=9)
ax2.set_title('Discrimination Index per Participant × Pair\n'
              '(+100 = perfect, 0 = position bias, −100 = inverted)',
              fontsize=12, fontweight='bold')

cbar = fig2.colorbar(im, ax=ax2, fraction=0.03, pad=0.02)
cbar.set_label('Discrimination Index\n(% chose 1st | 1st stronger)  −  (% chose 1st | 2nd stronger)',
               fontsize=9)
cbar.set_ticks([-100, -50, 0, 50, 100])

# Vertical separator between 1N and 26N blocks
ax2.axvline(3.5, color='black', lw=2)
ax2.text(1.5, -0.8, 'Reference = 1N', ha='center', va='top',
         fontsize=10, fontweight='bold', color='#555',
         transform=ax2.get_xaxis_transform())
ax2.text(5.0, -0.8, 'Reference = 26N', ha='center', va='top',
         fontsize=10, fontweight='bold', color='#555',
         transform=ax2.get_xaxis_transform())

# ── Save ──────────────────────────────────────────────────────────────────────
out_dir = '/Users/kyungeunjung/NailFoldExp/(New)Analysis/ForceDiscAnalysis/Output'
os.makedirs(out_dir, exist_ok=True)
fig1.savefig(f'{out_dir}/all_position_bias_scatter.png',  dpi=150, bbox_inches='tight')
fig2.savefig(f'{out_dir}/all_position_bias_heatmap.png',  dpi=150, bbox_inches='tight')
print('Saved scatter + heatmap.')
plt.close('all')
