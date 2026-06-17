"""
Response Bias Overview – Three summary plots (all participants pooled)

Plot 1: Delivered stimulus distribution
  → Of all trials, what % had First stim stronger vs Second stim stronger

Plot 2: Response distribution
  → Of all trials, what % of responses were "1st stronger" vs "2nd stronger"

Plot 3: Incorrect-only response distribution
  → Among trials answered incorrectly, what % responded "1st stronger" vs "2nd stronger"
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Load all participant data ──────────────────────────────────────────────────
files = sorted(glob.glob('/Users/kyungeunjung/NailFoldExp/Data/(FD)CurData/P*.csv'))
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

# Correct answer: whichever interval had the higher force
df['CorrectAnswer'] = np.where(df['FirstStim'] > df['SecondStim'], 1, 2)
df['IsCorrect'] = df['UserChoice'] == df['CorrectAnswer']
df['DeliveredStronger'] = df['CorrectAnswer']   # 1 = first stim was stronger, 2 = second

n_total = len(df)
n_incorrect = (~df['IsCorrect']).sum()

print(f"Total trials : {n_total}")
print(f"Correct      : {df['IsCorrect'].sum()}  ({df['IsCorrect'].mean()*100:.1f}%)")
print(f"Incorrect    : {n_incorrect}  ({(~df['IsCorrect']).mean()*100:.1f}%)")

# ── Compute proportions ────────────────────────────────────────────────────────
# Plot 1 – Delivered
del_first  = (df['DeliveredStronger'] == 1).sum()
del_second = (df['DeliveredStronger'] == 2).sum()
pct_del_first  = del_first  / n_total * 100
pct_del_second = del_second / n_total * 100

# Plot 2 – Responses
resp_first  = (df['UserChoice'] == 1).sum()
resp_second = (df['UserChoice'] == 2).sum()
pct_resp_first  = resp_first  / n_total * 100
pct_resp_second = resp_second / n_total * 100

# Plot 3 – Incorrect only
inc = df[~df['IsCorrect']]
inc_resp_first  = (inc['UserChoice'] == 1).sum()
inc_resp_second = (inc['UserChoice'] == 2).sum()
pct_inc_first  = inc_resp_first  / n_incorrect * 100
pct_inc_second = inc_resp_second / n_incorrect * 100

print(f"\n── Delivered ──")
print(f"  1st stronger: {del_first} ({pct_del_first:.1f}%)")
print(f"  2nd stronger: {del_second} ({pct_del_second:.1f}%)")
print(f"\n── Responses ──")
print(f"  Chose 1st:    {resp_first} ({pct_resp_first:.1f}%)")
print(f"  Chose 2nd:    {resp_second} ({pct_resp_second:.1f}%)")
print(f"\n── Incorrect responses ──")
print(f"  Chose 1st:    {inc_resp_first} ({pct_inc_first:.1f}%)")
print(f"  Chose 2nd:    {inc_resp_second} ({pct_inc_second:.1f}%)")

# ── Style helpers ──────────────────────────────────────────────────────────────
C1 = '#2166AC'   # blue  – "1st stronger"
C2 = '#D6604D'   # red   – "2nd stronger"

def draw_bar(ax, pct1, pct2, label1, label2, title, n, subtitle=None):
    """Horizontal stacked bar with percentage labels."""
    ax.barh(0, pct1, height=0.55, color=C1, label=label1)
    ax.barh(0, pct2, height=0.55, left=pct1, color=C2, label=label2)

    # Percentage text inside bars
    if pct1 > 8:
        ax.text(pct1 / 2, 0, f'{pct1:.1f}%', ha='center', va='center',
                fontsize=13, fontweight='bold', color='white')
    if pct2 > 8:
        ax.text(pct1 + pct2 / 2, 0, f'{pct2:.1f}%', ha='center', va='center',
                fontsize=13, fontweight='bold', color='white')

    # 50 % reference line
    ax.axvline(50, color='gray', lw=1.2, ls='--', alpha=0.7)
    ax.text(50, 0.34, '50%', ha='center', fontsize=9, color='gray', va='bottom')

    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, 0.55)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'], fontsize=10)
    ax.set_yticks([])
    ax.spines[['top', 'right', 'left']].set_visible(False)

    ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
    if subtitle:
        ax.set_xlabel(subtitle, fontsize=10, labelpad=6)

    leg = ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.38),
                    ncol=2, frameon=False, fontsize=10,
                    handles=[mpatches.Patch(color=C1, label=label1),
                              mpatches.Patch(color=C2, label=label2)])
    ax.text(0.98, 1.02, f'n = {n:,} trials', transform=ax.transAxes,
            ha='right', va='bottom', fontsize=9, color='#555')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE – Three stacked bars
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(3, 1, figsize=(9, 7.5))
fig.suptitle('Response Bias Overview  –  All Participants, All Regions, All Force Pairs',
             fontsize=13, fontweight='bold', y=0.99)

draw_bar(axes[0],
         pct_del_first, pct_del_second,
         '1st stim was stronger', '2nd stim was stronger',
         'Plot 1 – Delivered Stimulus Distribution',
         n=n_total,
         subtitle='Which interval was physically stronger?')

draw_bar(axes[1],
         pct_resp_first, pct_resp_second,
         'Responded: 1st stronger', 'Responded: 2nd stronger',
         'Plot 2 – Response Distribution  (all trials)',
         n=n_total,
         subtitle='What did participants say?')

draw_bar(axes[2],
         pct_inc_first, pct_inc_second,
         'Responded: 1st stronger', 'Responded: 2nd stronger',
         'Plot 3 – Incorrect-Trial Response Distribution',
         n=n_incorrect,
         subtitle='Among wrong answers, which interval did they pick?')

plt.tight_layout(rect=[0, 0, 1, 0.97], h_pad=2.5)

out_dir = '/Users/kyungeunjung/NailFoldExp/(New)Analysis/ForceDiscAnalysis/Output'
os.makedirs(out_dir, exist_ok=True)
out_path = f'{out_dir}/response_bias_overview.png'
fig.savefig(out_path, dpi=150, bbox_inches='tight')
print(f'\nSaved → {out_path}')
plt.close('all')
