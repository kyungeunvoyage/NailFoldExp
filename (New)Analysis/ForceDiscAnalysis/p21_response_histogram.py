"""
P21 Position Bias Check
For each (Reference × Comparison) pair:
  X-axis group: "1st stim was STRONGER" | "2nd stim was STRONGER"
  Bars: chose 1st (blue) vs chose 2nd (orange)

If unbiased: when 1st is stronger → mostly chose 1st; when 2nd is stronger → mostly chose 2nd
Position bias: chose 1st (or 2nd) regardless of which was actually stronger
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

df = pd.read_csv('/Users/kyungeunjung/NailFoldExp/Data/(FD)CurData/P21_ForceDiscrimination.csv')

# Which interval had the stronger stimulus?
df['StrongerInterval'] = np.where(df['FirstStim'] > df['SecondStim'], 1, 2)

BLUE   = '#1976D2'   # chose 1st
ORANGE = '#F57C00'   # chose 2nd

refs   = sorted(df['Reference'].unique())
n_cols = 4

fig, axes = plt.subplots(
    nrows=len(refs), ncols=n_cols,
    figsize=(n_cols * 3.4, len(refs) * 4.2),
    constrained_layout=True,
)
fig.suptitle('P21 – Position Bias Check\n'
             'Blue = chose 1st interval  |  Orange = chose 2nd interval\n'
             '(ideal unbiased: blue dominates left group, orange dominates right group)',
             fontsize=12, fontweight='bold')

for row_idx, ref in enumerate(refs):
    sub_ref = df[df['Reference'] == ref]
    comps   = sorted(sub_ref['Comparison'].unique())

    for col_idx in range(n_cols):
        ax = axes[row_idx, col_idx]

        if col_idx >= len(comps):
            ax.set_visible(False)
            continue

        comp = comps[col_idx]
        sub  = sub_ref[sub_ref['Comparison'] == comp]

        # Two groups: stronger in 1st interval, stronger in 2nd interval
        x_labels = ['1st stim\nwas stronger', '2nd stim\nwas stronger']
        x_pos    = np.array([0, 1])

        for xi, strong_interval in enumerate([1, 2]):
            grp = sub[sub['StrongerInterval'] == strong_interval]
            n   = len(grp)
            n1  = (grp['UserChoice'] == 1).sum()
            n2  = (grp['UserChoice'] == 2).sum()
            p1  = n1 / n * 100 if n > 0 else 0
            p2  = n2 / n * 100 if n > 0 else 0

            b1 = ax.bar(xi - 0.2, p1, 0.35,
                        color=BLUE,   edgecolor='white', linewidth=0.8, zorder=3)
            b2 = ax.bar(xi + 0.2, p2, 0.35,
                        color=ORANGE, edgecolor='white', linewidth=0.8, zorder=3)

            for bar, val, cnt in [(b1, p1, n1), (b2, p2, n2)]:
                if cnt > 0:
                    ax.text(bar[0].get_x() + bar[0].get_width() / 2,
                            bar[0].get_height() + 1,
                            f'{val:.0f}%\n(n={cnt})',
                            ha='center', va='bottom', fontsize=8, fontweight='bold')

            ax.text(xi, -6,
                    f'n={n}', ha='center', va='top', fontsize=8, color='#777')

        # 50% reference line
        ax.axhline(50, color='gray', linestyle='--', linewidth=1, alpha=0.6, zorder=1)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, fontsize=8.5)
        ax.set_title(f'Ref={ref}N  vs  Comp={comp}N', fontsize=9, fontweight='bold')
        ax.set_ylabel('% of trials', fontsize=8)
        ax.set_ylim(-10, 115)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_yticklabels(['0%', '25%', '50%', '75%', '100%'], fontsize=8)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', linestyle=':', alpha=0.4, zorder=0)
        ax.text(0.97, 0.97, f'total n={len(sub)}',
                transform=ax.transAxes, ha='right', va='top', fontsize=8, color='#555')

# ── Overall summary panel (bottom) ────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(6, 4))
fig2.suptitle('P21 – Overall Position Bias (all pairs combined)',
              fontsize=12, fontweight='bold')

grp1 = df[df['StrongerInterval'] == 1]
grp2 = df[df['StrongerInterval'] == 2]

data = {
    '1st stim\nwas stronger': {
        'Chose 1st': (grp1['UserChoice'] == 1).sum(),
        'Chose 2nd': (grp1['UserChoice'] == 2).sum(),
        'n': len(grp1),
    },
    '2nd stim\nwas stronger': {
        'Chose 1st': (grp2['UserChoice'] == 1).sum(),
        'Chose 2nd': (grp2['UserChoice'] == 2).sum(),
        'n': len(grp2),
    },
}

for xi, (label, vals) in enumerate(data.items()):
    n   = vals['n']
    p1  = vals['Chose 1st'] / n * 100
    p2  = vals['Chose 2nd'] / n * 100
    b1 = ax2.bar(xi - 0.2, p1, 0.35, color=BLUE,   edgecolor='white', label='Chose 1st' if xi==0 else '')
    b2 = ax2.bar(xi + 0.2, p2, 0.35, color=ORANGE, edgecolor='white', label='Chose 2nd' if xi==0 else '')
    for bar, pct, cnt in [(b1, p1, vals['Chose 1st']), (b2, p2, vals['Chose 2nd'])]:
        ax2.text(bar[0].get_x() + bar[0].get_width()/2,
                 bar[0].get_height() + 1,
                 f'{pct:.0f}%\n(n={cnt})', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax2.text(xi, -6, f'n={n}', ha='center', va='top', fontsize=9, color='#777')

ax2.axhline(50, color='gray', linestyle='--', linewidth=1, alpha=0.6)
ax2.set_xticks([0, 1])
ax2.set_xticklabels(list(data.keys()), fontsize=10)
ax2.set_ylabel('% of trials', fontsize=10)
ax2.set_ylim(-10, 115)
ax2.set_yticks([0, 25, 50, 75, 100])
ax2.set_yticklabels(['0%', '25%', '50%', '75%', '100%'])
ax2.spines[['top', 'right']].set_visible(False)
ax2.grid(axis='y', linestyle=':', alpha=0.4)
ax2.legend(fontsize=10)

# ── Save ──────────────────────────────────────────────────────────────────────
out_dir = '/Users/kyungeunjung/NailFoldExp/(New)Analysis/ForceDiscAnalysis/Output'
os.makedirs(out_dir, exist_ok=True)

fig.savefig(f'{out_dir}/p21_response_histogram_by_pair.png',  dpi=150, bbox_inches='tight')
fig2.savefig(f'{out_dir}/p21_position_bias_overall.png',       dpi=150, bbox_inches='tight')
print('Saved both plots.')
plt.close('all')
