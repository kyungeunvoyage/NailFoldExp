"""
All-participants Position Bias Histogram – same style as P21 figure
For each (Reference × Comparison) pair:
  X-axis: "1st stim was stronger" | "2nd stim was stronger"
  Bars: mean % chose 1st (blue) vs mean % chose 2nd (orange) across participants
  Error bars: ±1 SE
  Dots: individual participant values
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ── Load data ─────────────────────────────────────────────────────────────────
files = sorted(glob.glob('/Users/kyungeunjung/NailFoldExp/Data/(FD)CurData/P*.csv'))
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
df['StrongerInterval'] = np.where(df['FirstStim'] > df['SecondStim'], 1, 2)

subjects = sorted(df['Subject'].unique())
pairs    = [(1, 0.4), (1, 0.6), (1, 1.4), (1, 2.0),
            (26, 10), (26, 15), (26, 60)]

# ── Compute per-subject per-pair per-condition stats ──────────────────────────
records = []
for subj in subjects:
    sd = df[df['Subject'] == subj]
    for ref, comp in pairs:
        grp = sd[(sd['Reference'] == ref) & (sd['Comparison'] == comp)]
        for strong_interval in [1, 2]:
            g = grp[grp['StrongerInterval'] == strong_interval]
            n = len(g)
            p1 = g['UserChoice'].eq(1).mean() * 100 if n > 0 else np.nan
            p2 = 100 - p1 if not np.isnan(p1) else np.nan
            records.append({
                'Subject': subj, 'Reference': ref, 'Comparison': comp,
                'StrongerInterval': strong_interval,
                'pct_chose1': p1, 'pct_chose2': p2, 'n': n,
            })

res = pd.DataFrame(records)

# ── Plot ──────────────────────────────────────────────────────────────────────
BLUE   = '#1976D2'
ORANGE = '#F57C00'
DOT_BLUE   = '#90CAF9'
DOT_ORANGE = '#FFCC80'
n_refs = 2
n_cols = 4

fig, axes = plt.subplots(
    nrows=n_refs, ncols=n_cols,
    figsize=(n_cols * 3.6, n_refs * 4.4),
    constrained_layout=True,
)
fig.suptitle('All Participants – Position Bias per Pair\n'
             'Bars = mean ±1 SE  |  dashed line = median  |  dots = individual participants\n'
             'Blue = chose 1st interval  |  Orange = chose 2nd interval',
             fontsize=12, fontweight='bold')

refs_list = [1, 26]

for row_idx, ref in enumerate(refs_list):
    comps = [c for r, c in pairs if r == ref]

    for col_idx in range(n_cols):
        ax = axes[row_idx, col_idx]

        if col_idx >= len(comps):
            ax.set_visible(False)
            continue

        comp = comps[col_idx]
        sub  = res[(res['Reference'] == ref) & (res['Comparison'] == comp)]

        x_labels = ['1st stim\nwas stronger', '2nd stim\nwas stronger']

        for xi, strong_interval in enumerate([1, 2]):
            g = sub[sub['StrongerInterval'] == strong_interval]
            vals1 = g['pct_chose1'].dropna().values
            vals2 = g['pct_chose2'].dropna().values
            n_subj = len(vals1)

            mean1 = vals1.mean(); se1 = vals1.std() / np.sqrt(n_subj)
            mean2 = vals2.mean(); se2 = vals2.std() / np.sqrt(n_subj)
            med1  = np.median(vals1)
            med2  = np.median(vals2)

            # Bars (height = mean)
            b1 = ax.bar(xi - 0.22, mean1, 0.38,
                        color=BLUE,   edgecolor='white', linewidth=0.8,
                        zorder=3, yerr=se1, capsize=4,
                        error_kw=dict(ecolor='#0D47A1', lw=1.5, capthick=1.5))
            b2 = ax.bar(xi + 0.22, mean2, 0.38,
                        color=ORANGE, edgecolor='white', linewidth=0.8,
                        zorder=3, yerr=se2, capsize=4,
                        error_kw=dict(ecolor='#E65100', lw=1.5, capthick=1.5))

            # Mean labels
            for mean_val, se_val, xpos in [(mean1, se1, xi - 0.22), (mean2, se2, xi + 0.22)]:
                ax.text(xpos, mean_val + se_val + 2.5,
                        f'{mean_val:.0f}%',
                        ha='center', va='bottom', fontsize=8, fontweight='bold')

            # Median markers (horizontal white line across bar)
            for med_val, xpos in [(med1, xi - 0.22), (med2, xi + 0.22)]:
                ax.plot([xpos - 0.17, xpos + 0.17], [med_val, med_val],
                        color='white', lw=2.5, zorder=6, solid_capstyle='round')
                ax.plot([xpos - 0.17, xpos + 0.17], [med_val, med_val],
                        color='black', lw=1.2, zorder=7, solid_capstyle='round',
                        linestyle=(0, (4, 2)))

            # Individual dots with jitter
            jitter = (np.random.RandomState(42).rand(n_subj) - 0.5) * 0.15
            ax.scatter(xi - 0.22 + jitter, vals1, s=20, color=DOT_BLUE,
                       edgecolors='#1565C0', linewidths=0.5, zorder=5, alpha=0.8)
            ax.scatter(xi + 0.22 + jitter, vals2, s=20, color=DOT_ORANGE,
                       edgecolors='#BF360C', linewidths=0.5, zorder=5, alpha=0.8)

            ax.text(xi, -8, f'n={n_subj}', ha='center', va='top',
                    fontsize=7.5, color='#777')

        # 50% reference line
        ax.axhline(50, color='gray', linestyle='--', linewidth=1, alpha=0.5, zorder=1)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(x_labels, fontsize=8.5)
        ax.set_title(f'Ref={ref}N  vs  Comp={comp}N', fontsize=9, fontweight='bold')
        ax.set_ylabel('Chosen Percentage (%)', fontsize=8)
        ax.set_ylim(-12, 115)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_yticklabels(['0%', '25%', '50%', '75%', '100%'], fontsize=8)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', linestyle=':', alpha=0.3, zorder=0)

# ── Save ──────────────────────────────────────────────────────────────────────
out_path = ('/Users/kyungeunjung/NailFoldExp/(New)Analysis/ForceDiscAnalysis'
            '/Output/all_response_histogram_by_pair.png')
os.makedirs(os.path.dirname(out_path), exist_ok=True)
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f'Saved → {out_path}')
plt.close()
