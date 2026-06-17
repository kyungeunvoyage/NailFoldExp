"""
Deep-dive: below-chance accuracy in Force Discrimination

Key finding: on near-threshold pairs, participants systematically choose the
WEAKER stimulus as stronger — producing accuracies of 22–34%.

Figures
-------
A  Accuracy by force pair (all participants pooled) — annotated 50% chance line
B  % Chose Comparison by force pair — reveals the perceptual reversal
C  Accuracy by region for each force pair (heatmap)
D  Per-participant accuracy on the 3 below-chance pairs
E  Response split for below-chance pairs: chose-weaker rate broken down by
   which interval the weaker stimulus occupied (interval-bias check)
"""

import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm
import matplotlib.gridspec as gridspec

# ── Load data ──────────────────────────────────────────────────────────────────
files = sorted(glob.glob('/Users/kyungeunjung/NailFoldExp/Data/(FD)CurData/P*.csv'))
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
df['CorrectAnswer']  = np.where(df['FirstStim'] > df['SecondStim'], 1, 2)
df['IsCorrect']      = df['UserChoice'] == df['CorrectAnswer']
df['CompIsStronger'] = df['Comparison'] > df['Reference']      # True if comp > ref
df['ChoseWeaker']    = (                                        # chose the lower-force stim
    np.where(df['CompIsStronger'],
             df['ChoseComparison'] == 0,   # comp>ref: chose weaker = chose ref
             df['ChoseComparison'] == 1))  # comp<ref: chose weaker = chose comp

# Force-ratio: comparison / reference  (>1 = comp stronger, <1 = comp weaker)
df['ForceRatio'] = df['Comparison'] / df['Reference']

def make_label(ref, comp):
    c = f"{comp:g}"
    return f"Ref={int(ref)}N\nComp={c}N"

df['PairLabel'] = df.apply(lambda r: make_label(r['Reference'], r['Comparison']), axis=1)

pairs_meta  = (df.groupby(['Reference','Comparison'])
                 .agg(acc=('IsCorrect','mean'),
                      chose_comp=('ChoseComparison','mean'),
                      chose_weaker=('ChoseWeaker','mean'),
                      n=('IsCorrect','count'))
                 .reset_index())
pairs_meta['ratio'] = pairs_meta['Comparison'] / pairs_meta['Reference']
pairs_meta['PairLabel'] = pairs_meta.apply(
    lambda r: make_label(r['Reference'], r['Comparison']), axis=1)
pairs_meta = pairs_meta.sort_values('ratio').reset_index(drop=True)

# ── Palette ────────────────────────────────────────────────────────────────────
BELOW_CLR  = '#D32F2F'   # red   — below chance
ABOVE_CLR  = '#1565C0'   # blue  — above chance
NEUT_CLR   = '#888'
REGION_ORDER = ['A','B','C','D','E','F']

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE
# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(18, 14))
fig.suptitle('Force Discrimination – Deep-Dive: Why Are Some Pairs Below Chance?',
             fontsize=15, fontweight='bold', y=0.99)

gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.38,
                       left=0.07, right=0.97, top=0.94, bottom=0.06)

ax_A  = fig.add_subplot(gs[0, :2])   # top-left wide: accuracy by pair
ax_B  = fig.add_subplot(gs[0, 2])    # top-right: % chose weaker
ax_C  = fig.add_subplot(gs[1, :])    # middle: region heatmap
ax_D  = fig.add_subplot(gs[2, :2])   # bottom-left wide: per-participant
ax_E  = fig.add_subplot(gs[2, 2])    # bottom-right: interval bias check

# ── Panel A: Accuracy by force pair ───────────────────────────────────────────
n_pairs = len(pairs_meta)
x = np.arange(n_pairs)
colors_A = [BELOW_CLR if a < 0.5 else ABOVE_CLR for a in pairs_meta['acc']]
bars = ax_A.bar(x, pairs_meta['acc'] * 100, color=colors_A,
                edgecolor='white', linewidth=0.8, width=0.6, zorder=3)

ax_A.axhline(50, color='gray', ls='--', lw=1.4, zorder=2, label='Chance (50%)')
ax_A.set_xticks(x)
ax_A.set_xticklabels(pairs_meta['PairLabel'], fontsize=9)
ax_A.set_ylabel('Accuracy (%)', fontsize=11)
ax_A.set_ylim(0, 100)
ax_A.set_title('A – Accuracy by Force Pair  (all participants pooled)', fontweight='bold')
ax_A.spines[['top','right']].set_visible(False)
ax_A.legend(fontsize=9, frameon=False)
ax_A.grid(axis='y', ls=':', alpha=0.4, zorder=1)

# Annotate bars
for i, (acc, n) in enumerate(zip(pairs_meta['acc'], pairs_meta['n'])):
    ax_A.text(i, acc*100 + 1.5, f'{acc*100:.0f}%', ha='center', fontsize=8.5,
              fontweight='bold', color=colors_A[i])

# Add Ref blocks
ax_A.axvline(3.5, color='black', lw=1, alpha=0.4)
for xpos, lbl in [(1.5, 'Reference = 1 N'), (5, 'Reference = 26 N')]:
    ax_A.text(xpos, 98, lbl, ha='center', fontsize=9, color='#444',
              fontweight='bold', va='top',
              bbox=dict(boxstyle='round,pad=0.2', fc='#f5f5f5', ec='#bbb', lw=0.5))

# ── Panel B: % Chose Weaker ────────────────────────────────────────────────────
colors_B = [BELOW_CLR if cw > 0.5 else ABOVE_CLR for cw in pairs_meta['chose_weaker']]
ax_B.barh(x, pairs_meta['chose_weaker'] * 100,
          color=colors_B, edgecolor='white', linewidth=0.8, height=0.6)
ax_B.axvline(50, color='gray', ls='--', lw=1.4)
ax_B.set_xlim(0, 100)
ax_B.set_yticks(x)
ax_B.set_yticklabels(pairs_meta['PairLabel'], fontsize=8.5)
ax_B.set_xlabel('% trials chose\nWEAKER stimulus as stronger', fontsize=10)
ax_B.set_title('B – "Chose Weaker"\nResponse Rate', fontweight='bold', fontsize=11)
ax_B.spines[['top','right']].set_visible(False)
ax_B.grid(axis='x', ls=':', alpha=0.4)
for i, cw in enumerate(pairs_meta['chose_weaker']):
    ax_B.text(cw*100 + 1.5, i, f'{cw*100:.0f}%', va='center', fontsize=8.5,
              fontweight='bold', color=colors_B[i])

# ── Panel C: Region × Pair heatmap ────────────────────────────────────────────
region_pair_acc = (df.groupby(['Region', 'PairLabel'])['IsCorrect']
                     .mean().unstack('PairLabel') * 100)
ordered_cols = pairs_meta['PairLabel'].tolist()
region_pair_acc = region_pair_acc.reindex(index=REGION_ORDER, columns=ordered_cols)
mat = region_pair_acc.values.astype(float)

norm = TwoSlopeNorm(vmin=0, vcenter=50, vmax=100)
im = ax_C.imshow(mat, cmap='RdYlGn', norm=norm, aspect='auto')

for i in range(len(REGION_ORDER)):
    for j in range(len(ordered_cols)):
        val = mat[i, j]
        if not np.isnan(val):
            txt_color = 'white' if val < 25 or val > 82 else '#222'
            ax_C.text(j, i, f'{val:.0f}%', ha='center', va='center',
                      fontsize=9, fontweight='bold', color=txt_color)

ax_C.set_xticks(range(n_pairs))
ax_C.set_xticklabels(pairs_meta['PairLabel'], fontsize=9)
ax_C.set_yticks(range(len(REGION_ORDER)))
ax_C.set_yticklabels(REGION_ORDER, fontsize=10)
ax_C.set_ylabel('Region', fontsize=11)
ax_C.set_title('C – Accuracy by Region × Force Pair  (color: green = above chance, red = below chance)',
               fontweight='bold')

cbar = fig.colorbar(im, ax=ax_C, fraction=0.015, pad=0.01, orientation='vertical')
cbar.set_label('Accuracy (%)', fontsize=9)
cbar.set_ticks([0, 25, 50, 75, 100])

ax_C.axvline(3.5, color='black', lw=1.5)

# ── Panel D: Per-participant accuracy on below-chance pairs ────────────────────
below_pairs = [(1, 0.6), (26, 15), (1, 1.4)]
below_labels = ['Ref=1N / Comp=0.6N\n(comp weaker)',
                'Ref=26N / Comp=15N\n(comp weaker)',
                'Ref=1N / Comp=1.4N\n(comp stronger)']
pair_colors  = ['#C62828', '#AD1457', '#6A1B9A']

subjects = sorted(df['Subject'].unique())
x_subj = np.arange(len(subjects))
width = 0.26

for k, ((ref, comp), lbl, clr) in enumerate(zip(below_pairs, below_labels, pair_colors)):
    sub = df[(df['Reference']==ref) & (df['Comparison']==comp)]
    accs = [sub[sub['Subject']==s]['IsCorrect'].mean() * 100 for s in subjects]
    ax_D.bar(x_subj + (k - 1) * width, accs,
             width=width, label=lbl, color=clr, alpha=0.85,
             edgecolor='white', linewidth=0.5)

ax_D.axhline(50, color='gray', ls='--', lw=1.4, label='Chance (50%)')
ax_D.set_xticks(x_subj)
ax_D.set_xticklabels(subjects, fontsize=8, rotation=45, ha='right')
ax_D.set_ylabel('Accuracy (%)', fontsize=11)
ax_D.set_ylim(0, 100)
ax_D.set_title('D – Per-Participant Accuracy on Below-Chance Pairs', fontweight='bold')
ax_D.spines[['top','right']].set_visible(False)
ax_D.legend(fontsize=8.5, frameon=False, ncol=3, loc='upper right')
ax_D.grid(axis='y', ls=':', alpha=0.4)

# ── Panel E: Interval bias check for below-chance pairs ───────────────────────
# For each below-chance pair: when weaker stim was 1st vs 2nd interval,
# what % chose the weaker stim?  (If interval position matters, we'd see a difference)
interval_data = []
for ref, comp, lbl, clr in zip(
        [r for r,c in below_pairs], [c for r,c in below_pairs],
        below_labels, pair_colors):
    sub = df[(df['Reference']==ref) & (df['Comparison']==comp)]
    # Weaker stim in 1st interval = FirstStim is the weaker one
    weaker_first  = sub[sub['FirstStim'] < sub['SecondStim']]   # weaker was 1st
    weaker_second = sub[sub['FirstStim'] > sub['SecondStim']]   # weaker was 2nd
    # Did they choose the weaker stim?
    pct_w1 = weaker_first['ChoseWeaker'].mean() * 100   if len(weaker_first)  > 0 else np.nan
    pct_w2 = weaker_second['ChoseWeaker'].mean() * 100  if len(weaker_second) > 0 else np.nan
    interval_data.append((lbl.split('\n')[0], pct_w1, pct_w2, len(weaker_first), len(weaker_second), clr))

x_e = np.arange(len(interval_data))
w_e = 0.32
for i, (lbl, pw1, pw2, n1, n2, clr) in enumerate(interval_data):
    ax_E.bar(i - w_e/2, pw1, width=w_e, color=clr, alpha=0.6, label='Weaker = 1st interval' if i==0 else '')
    ax_E.bar(i + w_e/2, pw2, width=w_e, color=clr, alpha=1.0, label='Weaker = 2nd interval' if i==0 else '')

ax_E.axhline(50, color='gray', ls='--', lw=1.4)
ax_E.set_xticks(x_e)
ax_E.set_xticklabels([d[0] for d in interval_data], fontsize=8.5, rotation=12, ha='right')
ax_E.set_ylim(0, 100)
ax_E.set_ylabel('% chose weaker stim', fontsize=10)
ax_E.set_title('E – Interval Position Check\n(Is "chose weaker" the same\nregardless of which interval?)',
               fontweight='bold', fontsize=10)
ax_E.spines[['top','right']].set_visible(False)
ax_E.legend(fontsize=8.5, frameon=False, loc='lower right')
ax_E.grid(axis='y', ls=':', alpha=0.4)

# ── Save ──────────────────────────────────────────────────────────────────────
out_dir  = '/Users/kyungeunjung/NailFoldExp/(New)Analysis/ForceDiscAnalysis/Output'
out_path = f'{out_dir}/response_bias_deepdive.png'
os.makedirs(out_dir, exist_ok=True)
fig.savefig(out_path, dpi=150, bbox_inches='tight')
print(f'Saved → {out_path}')
plt.close('all')
