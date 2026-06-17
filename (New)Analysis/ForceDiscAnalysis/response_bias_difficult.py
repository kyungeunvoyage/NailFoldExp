"""
Response Bias – Difficult vs Easy Force-Pair Comparison + Per-Participant Breakdown
=====================================================================================
Filters trials down to the hardest force pairs (1.4-1, 0.6-1, 15-26),
auto-detects the easiest pairs (highest accuracy) for comparison, and
produces:

  A) response_bias_difficult_combined.png        – 3 difficult pairs pooled, marginal bars
  B) response_bias_difficult_per_pair.png         – 3x3 grid, marginal bars per difficult pair
  C) response_bias_confusion_matrix_difficult.png – Delivered x Response 2x2 matrix, difficult pairs
  D) response_bias_confusion_matrix_easy.png      – same matrix, but for the EASIEST pairs
  E) response_bias_accuracy_by_participant.png    – bar chart: each participant's accuracy
                                                      on difficult-pair trials, sorted
  F) response_bias_confusion_by_participant.png   – small-multiples confusion matrix,
                                                      one panel per participant (difficult
                                                      pairs combined)

E and F answer: is the reversal pattern seen in (C) driven by most participants
consistently, or by just a handful of outlier participants?
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Load all participant data ───────────────────────────────────────────────
files = sorted(glob.glob('/Users/kyungeunjung/NailFoldExp/Data/(FD)CurData/P*.csv'))
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

# Each row already has a 'Subject' column (e.g. "P21") -- use that directly
# rather than parsing the filename.
print(f"Participants loaded: {df['Subject'].nunique()}  -> {sorted(df['Subject'].unique())}")

# Correct answer: whichever interval had the higher force
df['CorrectAnswer'] = np.where(df['FirstStim'] > df['SecondStim'], 1, 2)
df['IsCorrect'] = df['UserChoice'] == df['CorrectAnswer']
df['DeliveredStronger'] = df['CorrectAnswer']

# ── Pair-matching helper ─────────────────────────────────────────────────────
TOL = 1e-3  # tolerance for floating point comparisons

def pair_mask(data, a, b):
    """Rows where {FirstStim, SecondStim} == {a, b}, regardless of order."""
    m1 = np.isclose(data['FirstStim'], a, atol=TOL) & np.isclose(data['SecondStim'], b, atol=TOL)
    m2 = np.isclose(data['FirstStim'], b, atol=TOL) & np.isclose(data['SecondStim'], a, atol=TOL)
    return m1 | m2

def filter_to_pairs(data, pairs):
    """Return rows matching any pair in `pairs`, tagged with a PairLabel column."""
    out = data.copy()
    out['PairLabel'] = None
    for a, b in pairs:
        out.loc[pair_mask(out, a, b), 'PairLabel'] = f'{a}-{b}'
    return out[out['PairLabel'].notna()].copy()

# ── Difficult pairs (manually specified) ─────────────────────────────────────
DIFFICULT_PAIRS = [(1.4, 1), (0.6, 1), (15, 26)]
df_hard = filter_to_pairs(df, DIFFICULT_PAIRS)

print(f"\nTotal trials             : {len(df)}")
print(f"Difficult-pair trials     : {len(df_hard)}")
for a, b in DIFFICULT_PAIRS:
    n_pair = (df_hard['PairLabel'] == f'{a}-{b}').sum()
    print(f"  {a}-{b:<6}: {n_pair} trials")

if df_hard.empty:
    raise ValueError(
        "No trials matched DIFFICULT_PAIRS. Check that FirstStim/SecondStim "
        "values actually equal the pairs listed (units, decimals, etc.)."
    )

# ── Auto-detect the EASIEST pairs (highest accuracy) for comparison ─────────
df['PairKey'] = list(zip(np.minimum(df['FirstStim'], df['SecondStim']).round(3),
                          np.maximum(df['FirstStim'], df['SecondStim']).round(3)))

pair_accuracy = (
    df.groupby('PairKey')
      .agg(n=('IsCorrect', 'size'), accuracy=('IsCorrect', 'mean'))
      .reset_index()
      .sort_values('accuracy', ascending=False)
)

print("\nAll force pairs ranked by accuracy (top 10 shown):")
print(pair_accuracy.head(10).to_string(index=False))

N_EASY = 3
easy_keys = pair_accuracy.head(N_EASY)['PairKey'].tolist()
EASY_PAIRS = [(float(lo), float(hi)) for lo, hi in easy_keys]
print(f"\nAuto-selected easiest pairs (highest accuracy): {EASY_PAIRS}")

df_easy = filter_to_pairs(df, EASY_PAIRS)
print(f"Easy-pair trials          : {len(df_easy)}")
for a, b in EASY_PAIRS:
    n_pair = (df_easy['PairLabel'] == f'{a}-{b}').sum()
    print(f"  {a}-{b:<6}: {n_pair} trials")

# ── Style helpers ────────────────────────────────────────────────────────────
C1 = '#2166AC'   # blue – "1st stronger / chosen"
C2 = '#D6604D'   # red  – "2nd stronger / chosen"


def compute_stats(sub):
    n_total = len(sub)
    n_incorrect = int((~sub['IsCorrect']).sum())

    del_first  = (sub['DeliveredStronger'] == 1).sum()
    del_second = (sub['DeliveredStronger'] == 2).sum()

    resp_first  = (sub['UserChoice'] == 1).sum()
    resp_second = (sub['UserChoice'] == 2).sum()

    inc = sub[~sub['IsCorrect']]
    inc_first  = (inc['UserChoice'] == 1).sum()
    inc_second = (inc['UserChoice'] == 2).sum()

    return dict(
        n_total=n_total,
        n_incorrect=n_incorrect,
        pct_del_first=del_first / n_total * 100 if n_total else 0,
        pct_del_second=del_second / n_total * 100 if n_total else 0,
        pct_resp_first=resp_first / n_total * 100 if n_total else 0,
        pct_resp_second=resp_second / n_total * 100 if n_total else 0,
        pct_inc_first=inc_first / n_incorrect * 100 if n_incorrect else 0,
        pct_inc_second=inc_second / n_incorrect * 100 if n_incorrect else 0,
    )


def draw_bar(ax, pct1, pct2, label1, label2, title, n, subtitle=None, show_legend=True):
    """Horizontal stacked bar with percentage labels."""
    ax.barh(0, pct1, height=0.55, color=C1, label=label1)
    ax.barh(0, pct2, height=0.55, left=pct1, color=C2, label=label2)

    if pct1 > 8:
        ax.text(pct1 / 2, 0, f'{pct1:.1f}%', ha='center', va='center',
                fontsize=12, fontweight='bold', color='white')
    if pct2 > 8:
        ax.text(pct1 + pct2 / 2, 0, f'{pct2:.1f}%', ha='center', va='center',
                fontsize=12, fontweight='bold', color='white')

    ax.axvline(50, color='gray', lw=1.2, ls='--', alpha=0.7)

    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, 0.55)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'], fontsize=9)
    ax.set_yticks([])
    ax.spines[['top', 'right', 'left']].set_visible(False)

    if title:
        ax.set_title(title, fontsize=11, fontweight='bold', pad=8)
    if subtitle:
        ax.set_xlabel(subtitle, fontsize=9, labelpad=5)

    ax.text(0.98, 1.05, f'n = {n:,}', transform=ax.transAxes,
            ha='right', va='bottom', fontsize=8, color='#555')

    if show_legend:
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.38),
                  ncol=2, frameon=False, fontsize=9,
                  handles=[mpatches.Patch(color=C1, label=label1),
                            mpatches.Patch(color=C2, label=label2)])


def confusion_counts(sub):
    """2x2 counts: rows = actual stronger (1,2), cols = response (1,2)."""
    mat = np.zeros((2, 2), dtype=int)
    for i, actual in enumerate([1, 2]):
        for j, resp in enumerate([1, 2]):
            mat[i, j] = ((sub['DeliveredStronger'] == actual) & (sub['UserChoice'] == resp)).sum()
    return mat


def draw_confusion_cell(ax, sub, label, fontsize=12):
    """Draw one 2x2 confusion-matrix panel into the given axis."""
    counts = confusion_counts(sub)
    row_totals = counts.sum(axis=1, keepdims=True)
    row_pct = np.divide(counts, row_totals, out=np.zeros_like(counts, dtype=float),
                         where=row_totals != 0) * 100

    ax.imshow(row_pct, cmap='RdBu_r', vmin=0, vmax=100, aspect='auto')

    for i in range(2):
        for j in range(2):
            txt_color = 'white' if (row_pct[i, j] > 65 or row_pct[i, j] < 35) else 'black'
            ax.text(j, i, f'{row_pct[i, j]:.0f}%\n(n={counts[i, j]})',
                    ha='center', va='center', fontsize=fontsize, fontweight='bold',
                    color=txt_color)

    for i in range(2):
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False,
                                     edgecolor='black', lw=1.8))

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Resp 1st', 'Resp 2nd'], fontsize=max(7, fontsize - 3))
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Actual 1st', 'Actual 2nd'], fontsize=max(7, fontsize - 3))
    ax.set_title(f'{label}   (n={counts.sum()})', fontsize=fontsize, fontweight='bold', pad=10)
    return counts


def plot_confusion_grid(pairs, data, suptitle, savepath):
    """Row-normalized Delivered x Response confusion matrix, one panel per pair (1 row)."""
    n_pairs = len(pairs)
    fig, axes = plt.subplots(1, n_pairs, figsize=(5.3 * n_pairs, 5.2))
    if n_pairs == 1:
        axes = [axes]

    fig.suptitle(suptitle, fontsize=14, fontweight='bold', y=1.03)

    for ax, (a, b) in zip(axes, pairs):
        sub = data[data['PairLabel'] == f'{a}-{b}']
        draw_confusion_cell(ax, sub, f'Pair {a}-{b}', fontsize=12)

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(savepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved -> {savepath}")


def plot_confusion_panels(panels, suptitle, savepath, ncols=5, panel_size=3.4):
    """Small-multiples confusion matrix grid. panels = list of (label, sub_df)."""
    n = len(panels)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(panel_size * ncols, panel_size * nrows))
    axes = np.atleast_1d(axes).flatten()

    fig.suptitle(suptitle, fontsize=15, fontweight='bold', y=1.0 + 0.4 / nrows)

    for ax, (label, sub) in zip(axes, panels):
        draw_confusion_cell(ax, sub, label, fontsize=9)

    for ax in axes[len(panels):]:
        ax.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 1 - 0.35 / nrows])
    fig.savefig(savepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved -> {savepath}")


out_dir = '/Users/kyungeunjung/NailFoldExp/(New)Analysis/ForceDiscAnalysis/Output'
os.makedirs(out_dir, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE A – Combined: all three difficult pairs pooled together (marginal bars)
# ═══════════════════════════════════════════════════════════════════════════
stats = compute_stats(df_hard)

fig, axes = plt.subplots(3, 1, figsize=(9, 10))
fig.suptitle('Response Bias – Difficult Pairs Only  (1.4-1, 0.6-1, 15-26 combined)',
             fontsize=13, fontweight='bold', y=0.99)

draw_bar(axes[0], stats['pct_del_first'], stats['pct_del_second'],
         '1st stim was stronger', '2nd stim was stronger',
         'Plot 1 – Delivered Stimulus Distribution', stats['n_total'],
         'Which interval was physically stronger?')

draw_bar(axes[1], stats['pct_resp_first'], stats['pct_resp_second'],
         'Responded: 1st stronger', 'Responded: 2nd stronger',
         'Plot 2 – Response Distribution (all trials)', stats['n_total'],
         'What did participants say?')

draw_bar(axes[2], stats['pct_inc_first'], stats['pct_inc_second'],
         'Responded: 1st stronger', 'Responded: 2nd stronger',
         'Plot 3 – Incorrect-Trial Response Distribution', stats['n_incorrect'],
         'Among wrong answers, which interval did they pick?')

plt.tight_layout(rect=[0, 0, 1, 0.96], h_pad=6)
combined_path = f'{out_dir}/response_bias_difficult_combined.png'
fig.savefig(combined_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"\nSaved -> {combined_path}")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE B – Per-pair breakdown (rows = pairs, columns = plot type)
# ═══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(len(DIFFICULT_PAIRS), 3, figsize=(16, 9.5))
fig.suptitle('Response Bias by Difficult Pair', fontsize=14, fontweight='bold', y=0.995)

col_titles = ['Delivered Stimulus', 'Response (all trials)', 'Response (incorrect only)']

for row, (a, b) in enumerate(DIFFICULT_PAIRS):
    sub = df_hard[df_hard['PairLabel'] == f'{a}-{b}']
    s = compute_stats(sub)

    draw_bar(axes[row, 0], s['pct_del_first'], s['pct_del_second'],
             '1st stronger', '2nd stronger', '', s['n_total'], show_legend=False)

    draw_bar(axes[row, 1], s['pct_resp_first'], s['pct_resp_second'],
             'Resp: 1st', 'Resp: 2nd', '', s['n_total'], show_legend=False)

    draw_bar(axes[row, 2], s['pct_inc_first'], s['pct_inc_second'],
             'Resp: 1st', 'Resp: 2nd', '', s['n_incorrect'], show_legend=False)

    axes[row, 0].set_ylabel(f'{a}-{b}', fontsize=12, fontweight='bold',
                              rotation=0, labelpad=45, va='center')

for col, t in enumerate(col_titles):
    axes[0, col].set_title(t, fontsize=12, fontweight='bold', pad=20)

fig.legend(handles=[mpatches.Patch(color=C1, label='1st stronger / chosen'),
                     mpatches.Patch(color=C2, label='2nd stronger / chosen')],
           loc='lower center', ncol=2, frameon=False, fontsize=10,
           bbox_to_anchor=(0.5, -0.01))

plt.tight_layout(rect=[0.03, 0.04, 1, 0.96], h_pad=3.5, w_pad=2)
per_pair_path = f'{out_dir}/response_bias_difficult_per_pair.png'
fig.savefig(per_pair_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Saved -> {per_pair_path}")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE C – Confusion matrix: DIFFICULT pairs
# ═══════════════════════════════════════════════════════════════════════════
plot_confusion_grid(
    DIFFICULT_PAIRS, df_hard,
    'Delivered × Response Confusion Matrix — DIFFICULT pairs  (% within row)',
    f'{out_dir}/response_bias_confusion_matrix_difficult.png'
)

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE D – Confusion matrix: EASY pairs (auto-selected, for comparison)
# ═══════════════════════════════════════════════════════════════════════════
plot_confusion_grid(
    EASY_PAIRS, df_easy,
    'Delivered × Response Confusion Matrix — EASY pairs  (% within row)',
    f'{out_dir}/response_bias_confusion_matrix_easy.png'
)

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE E – Per-participant accuracy on difficult-pair trials, sorted
# ═══════════════════════════════════════════════════════════════════════════
acc_by_participant = (
    df_hard.groupby('Subject')['IsCorrect']
           .agg(n='size', accuracy='mean')
           .reset_index()
)
acc_by_participant['accuracy_pct'] = acc_by_participant['accuracy'] * 100
acc_by_participant = acc_by_participant.sort_values('accuracy_pct').reset_index(drop=True)

print("\nPer-participant accuracy on difficult-pair trials:")
print(acc_by_participant[['Subject', 'n', 'accuracy_pct']].to_string(index=False))

fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(acc_by_participant)), 5.5))
bar_colors = [C2 if v < 50 else C1 for v in acc_by_participant['accuracy_pct']]
ax.bar(acc_by_participant['Subject'].astype(str), acc_by_participant['accuracy_pct'],
       color=bar_colors, edgecolor='white')

ax.axhline(50, color='gray', ls='--', lw=1.3)
ax.text(len(acc_by_participant) - 0.5, 51.5, 'chance (50%)', ha='right', fontsize=9, color='#555')

for i, v in enumerate(acc_by_participant['accuracy_pct']):
    ax.text(i, v + (1.5 if v >= 0 else -3), f'{v:.0f}%', ha='center', fontsize=8,
            color='black' if v >= 5 else 'white')

ax.set_ylim(0, max(100, acc_by_participant['accuracy_pct'].max() + 10))
ax.set_ylabel('Accuracy on difficult-pair trials (%)', fontsize=11)
ax.set_title('Per-Participant Accuracy — Difficult Pairs Combined (1.4-1, 0.6-1, 15-26)',
             fontsize=12, fontweight='bold', pad=12)
ax.set_xticks(range(len(acc_by_participant)))
ax.set_xticklabels(acc_by_participant['Subject'].astype(str), rotation=45, ha='right', fontsize=8)
ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
accuracy_path = f'{out_dir}/response_bias_accuracy_by_participant.png'
fig.savefig(accuracy_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Saved -> {accuracy_path}")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE F – Per-participant confusion matrix, small multiples (difficult pairs combined)
# ═══════════════════════════════════════════════════════════════════════════
participants_sorted = acc_by_participant['Subject'].tolist()  # worst accuracy first
panels = [(pid, df_hard[df_hard['Subject'] == pid]) for pid in participants_sorted]

plot_confusion_panels(
    panels,
    'Per-Participant Confusion Matrix — Difficult Pairs Combined (sorted: lowest accuracy first)',
    f'{out_dir}/response_bias_confusion_by_participant.png',
    ncols=5
)