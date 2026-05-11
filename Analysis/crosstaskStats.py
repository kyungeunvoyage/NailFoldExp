"""
================================================================
(A) Cross-Task Correlation: Detection ↔ Force JND ↔ Spatial JND
================================================================

Loads all three experiments, computes per-subject summary metrics,
and tests whether they share a common latent tactile-acuity factor.

Metrics per subject:
  - detection_threshold_g    : force at 80% accuracy (Exp 1)
                                computed per condition (in-air / on-touch)
  - force_jnd_pct            : force-discrimination JND
                                as % of reference force (Exp 2)
  - spatial_jnd_mm           : spatial JND from psychometric fit (Exp 3)
                                computed per force (1.0g / 26.0g)

Outputs:
  - per_subject_metrics.csv  : merged subject-level table
  - correlation_matrix.csv   : pairwise Spearman & Pearson correlations
  - correlation_scatter.png  : scatter plot matrix
"""

import os
import glob
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats

warnings.filterwarnings("ignore")


# ============================================================
# CONFIG — adjust these glob patterns if needed
# ============================================================
PATTERNS = {
    'AT': '/Users/kyungeunjung/NailFoldExp/Data/(AT)CurData/P*_AbsoluteThresholdDetection.csv',
    'FD': '/Users/kyungeunjung/NailFoldExp/Data/(FD)CurData/P*_ForceDiscrimination.csv',
    'SD': '/Users/kyungeunjung/NailFoldExp/Data/(SD)CurData/P*_SpatialDiscrimination.csv',
}
GRID_SPACING_MM = 1.5
DETECTION_CRITERION = 0.80
OUTPUT_DIR = './sd_outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# UTILS
# ============================================================
def parse_force_str(s):
    if pd.isna(s):
        return np.nan
    m = re.search(r'([\d.]+)', str(s))
    return float(m.group(1)) if m else np.nan

def parse_grid(s):
    if pd.isna(s):
        return np.nan
    m = re.match(r'g(-?\d+)', str(s).strip())
    return int(m.group(1)) if m else np.nan

def load_concat(pattern):
    files = glob.glob(pattern)
    if not files:
        print(f"  ! No files matched: {pattern}")
        return None
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

def psychometric(x, jnd, slope, lapse=0.02, guess=0.5):
    return guess + (1 - guess - lapse) / (1 + np.exp(-slope * (x - jnd)))

def fit_psychometric(x, y, p0=(2.0, 1.0), bounds=([0.01, 0.01], [50, 50])):
    try:
        popt, _ = curve_fit(psychometric, x, y, p0=p0,
                            bounds=bounds, maxfev=5000)
        return popt
    except Exception:
        return (np.nan, np.nan)


# ============================================================
# 1. EXP 1 — DETECTION THRESHOLD per subject
# ============================================================
print("[1] Exp 1 — Absolute Threshold Detection")
at_raw = load_concat(PATTERNS['AT'])
det_metrics = pd.DataFrame()

if at_raw is not None:
    at = at_raw.copy()
    at['force_g'] = at['Force'].apply(parse_force_str)
    at['IsCorrect'] = pd.to_numeric(at['IsCorrect'], errors='coerce')
    at = at.dropna(subset=['force_g', 'IsCorrect'])

    # Per subject × condition: accuracy across forces (averaged over regions)
    agg = (at.groupby(['SubjectID', 'Condition', 'force_g'])
             .agg(acc=('IsCorrect', 'mean'))
             .reset_index())

    rows = []
    for (subj, cond), grp in agg.groupby(['SubjectID', 'Condition']):
        grp = grp.sort_values('force_g')
        x = grp['force_g'].values
        y = grp['acc'].values
        # Try to fit a sigmoid; alternatively, interpolate where accuracy = 0.80
        try:
            if (y.max() >= DETECTION_CRITERION) and (y.min() <= DETECTION_CRITERION):
                # interpolate
                thr = np.interp(DETECTION_CRITERION, y, x)
            else:
                # fall back to lowest force with accuracy ≥ criterion
                above = grp[grp['acc'] >= DETECTION_CRITERION]
                thr = above['force_g'].min() if len(above) else np.nan
        except Exception:
            thr = np.nan
        rows.append({'Subject': subj, 'Condition': cond, 'detection_threshold_g': thr})

    det_metrics = pd.DataFrame(rows)
    # Pivot to wide
    det_wide = det_metrics.pivot(index='Subject', columns='Condition',
                                  values='detection_threshold_g')
    det_wide.columns = [f'AT_threshold_{c}' for c in det_wide.columns]
    det_wide = det_wide.reset_index()
    print(f"    {len(det_wide)} subjects, columns: {list(det_wide.columns)}")
else:
    det_wide = pd.DataFrame(columns=['Subject'])


# ============================================================
# 2. EXP 2 — FORCE DISCRIMINATION JND per subject
# ============================================================
print("\n[2] Exp 2 — Force Discrimination")
fd_raw = load_concat(PATTERNS['FD'])
fd_metrics = pd.DataFrame()

if fd_raw is not None:
    fd = fd_raw.copy()
    fd['Reference'] = pd.to_numeric(fd['Reference'], errors='coerce')
    fd['Comparison'] = pd.to_numeric(fd['Comparison'], errors='coerce')
    fd['ChoseComparison'] = pd.to_numeric(fd['ChoseComparison'], errors='coerce')
    fd = fd.dropna(subset=['Reference', 'Comparison', 'ChoseComparison'])

    # Force ratio: how different is comparison from reference (signed)
    fd['ratio'] = (fd['Comparison'] - fd['Reference']) / fd['Reference']
    fd['abs_ratio'] = fd['ratio'].abs()

    # IsCorrect approximation: choosing the LARGER stimulus when it differs
    # ChoseComparison=1 means user chose the comparison stimulus
    # Treat trial as "correct" when user chose the heavier stimulus
    fd['heavier_is_comp'] = (fd['Comparison'] > fd['Reference']).astype(int)
    fd['correct'] = (fd['ChoseComparison'] == fd['heavier_is_comp']).astype(int)

    # Per subject × reference: fit psychometric on (abs_ratio, accuracy)
    rows = []
    for (subj, ref), grp in fd.groupby(['Subject', 'Reference']):
        coll = (grp.groupby('abs_ratio')['correct']
                   .mean().reset_index())
        if len(coll) < 3:
            continue
        jnd, slope = fit_psychometric(coll['abs_ratio'].values,
                                       coll['correct'].values)
        rows.append({'Subject': subj, 'Reference': ref,
                     'force_jnd_pct': jnd * 100 if not np.isnan(jnd) else np.nan,
                     'force_slope': slope})

    fd_metrics = pd.DataFrame(rows)
    fd_wide = fd_metrics.pivot(index='Subject', columns='Reference',
                                values='force_jnd_pct')
    fd_wide.columns = [f'FD_jnd_pct_ref{c}' for c in fd_wide.columns]
    fd_wide = fd_wide.reset_index()
    print(f"    {len(fd_wide)} subjects, columns: {list(fd_wide.columns)}")
else:
    fd_wide = pd.DataFrame(columns=['Subject'])


# ============================================================
# 3. EXP 3 — SPATIAL JND per subject
# ============================================================
print("\n[3] Exp 3 — Spatial Discrimination")
sd_raw = load_concat(PATTERNS['SD'])
sd_metrics = pd.DataFrame()

if sd_raw is not None:
    sd = sd_raw.copy()
    sd['pos_1st'] = sd['Stim_1st'].apply(parse_grid) * GRID_SPACING_MM
    sd['pos_2nd'] = sd['Stim_2nd'].apply(parse_grid) * GRID_SPACING_MM
    sd['force_g'] = sd['Force'].apply(parse_force_str)
    sd['offset_mm'] = np.where(sd['pos_1st'] != 0, sd['pos_1st'], sd['pos_2nd'])
    sd['abs_offset_mm'] = sd['offset_mm'].abs()
    sd['IsCorrect'] = pd.to_numeric(sd['IsCorrect'], errors='coerce')
    sd = sd.dropna(subset=['IsCorrect', 'offset_mm', 'force_g'])

    rows = []
    for (subj, force), grp in sd.groupby(['Subject', 'force_g']):
        coll = (grp.groupby('abs_offset_mm')['IsCorrect']
                   .mean().reset_index())
        if len(coll) < 3:
            continue
        jnd, slope = fit_psychometric(coll['abs_offset_mm'].values,
                                       coll['IsCorrect'].values,
                                       bounds=([0.1, 0.1], [10, 10]))
        rows.append({'Subject': subj, 'force_g': force,
                     'spatial_jnd_mm': jnd, 'spatial_slope': slope})

    sd_metrics = pd.DataFrame(rows)
    sd_wide = sd_metrics.pivot(index='Subject', columns='force_g',
                                values='spatial_jnd_mm')
    sd_wide.columns = [f'SD_jnd_mm_{c}g' for c in sd_wide.columns]
    sd_wide = sd_wide.reset_index()
    print(f"    {len(sd_wide)} subjects, columns: {list(sd_wide.columns)}")
else:
    sd_wide = pd.DataFrame(columns=['Subject'])


# ============================================================
# 4. MERGE & FIND COMMON SUBJECTS
# ============================================================
print("\n[4] Merging subject-level tables ...")

def normalize_id(s):
    """Normalize Subject IDs (e.g., P1 vs P01) for matching."""
    m = re.match(r'P0*(\d+)', str(s))
    return f'P{int(m.group(1)):02d}' if m else str(s)

for d in (det_wide, fd_wide, sd_wide):
    if len(d) and 'Subject' in d.columns:
        d['Subject'] = d['Subject'].apply(normalize_id)

merged = det_wide.copy() if len(det_wide) else pd.DataFrame({'Subject': []})
if len(fd_wide):
    merged = merged.merge(fd_wide, on='Subject', how='outer')
if len(sd_wide):
    merged = merged.merge(sd_wide, on='Subject', how='outer')

merged.to_csv(os.path.join(OUTPUT_DIR, 'per_subject_metrics.csv'), index=False)
print(f"    Saved per_subject_metrics.csv  (n={len(merged)})")

# Show how many subjects overlap across tasks
metric_cols = [c for c in merged.columns if c != 'Subject']
print(f"\n    Coverage per metric:")
for c in metric_cols:
    print(f"      {c}: n={merged[c].notna().sum()}")


# ============================================================
# 5. CORRELATION MATRIX (Pearson + Spearman)
# ============================================================
print("\n[5] Pairwise correlations across tasks:")
results = []
for i, c1 in enumerate(metric_cols):
    for c2 in metric_cols[i+1:]:
        pair = merged[[c1, c2]].dropna()
        n = len(pair)
        if n < 5:
            continue
        try:
            r_p, p_p = stats.pearsonr(pair[c1], pair[c2])
            r_s, p_s = stats.spearmanr(pair[c1], pair[c2])
            results.append({
                'metric_1': c1, 'metric_2': c2, 'n': n,
                'pearson_r': r_p, 'pearson_p': p_p,
                'spearman_r': r_s, 'spearman_p': p_s,
            })
        except Exception as e:
            print(f"    ! {c1} vs {c2}: {e}")

corr_df = pd.DataFrame(results).sort_values('spearman_p')
corr_df.to_csv(os.path.join(OUTPUT_DIR, 'correlation_matrix.csv'), index=False)
print(corr_df.to_string(index=False))


# ============================================================
# 6. SCATTER PLOT MATRIX
# ============================================================
print("\n[6] Generating scatter plot matrix ...")
valid_cols = [c for c in metric_cols if merged[c].notna().sum() >= 5]
if len(valid_cols) >= 2:
    n = len(valid_cols)
    fig, axes = plt.subplots(n, n, figsize=(3*n, 3*n))
    for i, c1 in enumerate(valid_cols):
        for j, c2 in enumerate(valid_cols):
            ax = axes[i, j] if n > 1 else axes
            if i == j:
                vals = merged[c1].dropna()
                ax.hist(vals, bins=15, color='steelblue', alpha=0.7)
                ax.set_title(c1, fontsize=8)
            else:
                pair = merged[[c1, c2]].dropna()
                if len(pair) >= 5:
                    ax.scatter(pair[c2], pair[c1], s=25, alpha=0.7, color='darkred')
                    # add fit line
                    if pair[c2].std() > 0:
                        m, b = np.polyfit(pair[c2], pair[c1], 1)
                        xs = np.linspace(pair[c2].min(), pair[c2].max(), 50)
                        ax.plot(xs, m*xs + b, color='black', lw=1, alpha=0.5)
                    try:
                        r_s, p_s = stats.spearmanr(pair[c1], pair[c2])
                        ax.text(0.05, 0.95,
                                f'ρ={r_s:.2f}\np={p_s:.3f}\nn={len(pair)}',
                                transform=ax.transAxes,
                                fontsize=7, va='top',
                                bbox=dict(boxstyle='round', fc='white', alpha=0.7))
                    except Exception:
                        pass
            ax.tick_params(labelsize=6)
            if i == n-1:
                ax.set_xlabel(c2, fontsize=8)
            if j == 0:
                ax.set_ylabel(c1, fontsize=8)
    plt.tight_layout()
    out_png = os.path.join(OUTPUT_DIR, 'correlation_scatter.png')
    plt.savefig(out_png, dpi=200, bbox_inches='tight')
    print(f"    Saved {out_png}")
else:
    print("    Not enough metrics with sufficient overlap to plot.")

print("\n=== Cross-task correlation analysis done. ===")