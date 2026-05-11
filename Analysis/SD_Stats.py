"""
================================================================
Spatial Discrimination Analysis Pipeline
================================================================

Analyses included:
  1. Data loading & parsing
  2. Summary statistics (per subject, per force, per distance)
  3. Left-right symmetry check
  4. Psychometric function fitting (group-level)
  5. Per-subject JND extraction
  6. Logistic LME with Distance x Force interaction
  7. Visualization

Outputs:
  - summary_table.csv
  - jnd_per_subject.csv
  - psychometric_curves.png
  - lme_results.txt
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
# CONFIG
# ============================================================
FILE_PATTERN = '/Users/kyungeunjung/NailFoldExp/Data/(SD)CurData/P*_SpatialDiscrimination.csv'
GRID_SPACING_MM = 1.5    # 1 grid unit = 1.5 mm
THRESHOLD_CRITERION = 0.75   # 75% accuracy criterion
OUTPUT_DIR = './sd_outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 1. DATA LOADING
# ============================================================
all_files = glob.glob(FILE_PATTERN)

if not all_files:
    raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {FILE_PATTERN}")

print(f"[1] {len(all_files)}개의 파일 발견.")

dfs = []
for f in all_files:
    try:
        df = pd.read_csv(f)
        dfs.append(df)
    except Exception as e:
        print(f"  ! Failed to read {f}: {e}")

raw = pd.concat(dfs, ignore_index=True)
print(f"    Total rows: {len(raw)}")


# ============================================================
# 2. DATA PARSING & CLEANING
# ============================================================
def parse_grid(s):
    """'g3' -> 3, 'g-2' -> -2, 'g0' -> 0"""
    if pd.isna(s):
        return np.nan
    m = re.match(r'g(-?\d+)', str(s).strip())
    return int(m.group(1)) if m else np.nan


def parse_force(s):
    """'26.0g' -> 26.0, '1.0g' -> 1.0"""
    if pd.isna(s):
        return np.nan
    m = re.match(r'([\d.]+)', str(s).strip())
    return float(m.group(1)) if m else np.nan


df = raw.copy()
df['pos_1st'] = df['Stim_1st'].apply(parse_grid) * GRID_SPACING_MM
df['pos_2nd'] = df['Stim_2nd'].apply(parse_grid) * GRID_SPACING_MM
df['force_g'] = df['Force'].apply(parse_force)

# offset = position of the non-center stimulus (signed: + = right, - = left)
df['offset_mm'] = np.where(df['pos_1st'] != 0, df['pos_1st'], df['pos_2nd'])
df['abs_offset_mm'] = df['offset_mm'].abs()

# Make sure IsCorrect is numeric 0/1
df['IsCorrect'] = pd.to_numeric(df['IsCorrect'], errors='coerce')
df = df.dropna(subset=['IsCorrect', 'offset_mm', 'force_g'])

print(f"[2] 정제 후 rows: {len(df)}")
print(f"    Subjects: {df['Subject'].nunique()}")
print(f"    Forces: {sorted(df['force_g'].unique())}")
print(f"    Offsets: {sorted(df['offset_mm'].unique())}")


# ============================================================
# 3. SUMMARY STATISTICS
# ============================================================
# 3-1. Per (Subject, Force, offset)
agg_subj = (df.groupby(['Subject', 'force_g', 'offset_mm', 'abs_offset_mm'])
              .agg(accuracy=('IsCorrect', 'mean'),
                   n_trials=('IsCorrect', 'count'),
                   n_correct=('IsCorrect', 'sum'))
              .reset_index())

# 3-2. Group-level summary (Force x offset)
agg_group = (agg_subj.groupby(['force_g', 'offset_mm', 'abs_offset_mm'])
                     .agg(mean_acc=('accuracy', 'mean'),
                          sem_acc=('accuracy', lambda x: x.std(ddof=1)/np.sqrt(len(x))),
                          n_subj=('accuracy', 'count'))
                     .reset_index())

agg_group.to_csv(os.path.join(OUTPUT_DIR, 'summary_table.csv'), index=False)
print("[3] summary_table.csv 저장 완료.")


# ============================================================
# 4. LEFT-RIGHT SYMMETRY CHECK
# ============================================================
# For each subject and force, compare accuracy at +X vs -X
sym_records = []
for (subj, force), grp in agg_subj.groupby(['Subject', 'force_g']):
    for abs_d in sorted(grp['abs_offset_mm'].unique()):
        if abs_d == 0:
            continue
        left = grp.loc[grp['offset_mm'] == -abs_d, 'accuracy'].values
        right = grp.loc[grp['offset_mm'] == abs_d, 'accuracy'].values
        if len(left) and len(right):
            sym_records.append({
                'Subject': subj, 'force_g': force, 'abs_offset_mm': abs_d,
                'acc_left': float(left[0]), 'acc_right': float(right[0]),
                'diff': float(right[0] - left[0])
            })

sym_df = pd.DataFrame(sym_records)
print("\n[4] Symmetry check (signed-rank test per force):")
for force in sorted(sym_df['force_g'].unique()):
    sub = sym_df[sym_df['force_g'] == force]
    if len(sub) > 5:
        try:
            stat, p = stats.wilcoxon(sub['acc_left'], sub['acc_right'])
            print(f"    Force={force}g: Wilcoxon W={stat:.2f}, p={p:.4f} (n={len(sub)})")
        except Exception as e:
            print(f"    Force={force}g: Wilcoxon failed ({e})")


# ============================================================
# 5. PSYCHOMETRIC FUNCTION FITTING (group-level)
# ============================================================
def psychometric(x, jnd, slope, lapse=0.02, guess=0.5):
    """
    Logistic psychometric function with lapse rate.
    x       : absolute offset (mm)
    jnd     : 75% threshold (mm)
    slope   : steepness of the curve
    lapse   : asymptote ceiling = 1 - lapse
    guess   : chance level (0.5 for 2AFC)
    """
    # location where P=0.75 (between guess and 1-lapse)
    return guess + (1 - guess - lapse) / (1 + np.exp(-slope * (x - jnd)))


def fit_psychometric(x, y, p0=(2.0, 1.0)):
    """Fit psychometric function, return (jnd, slope)."""
    try:
        popt, _ = curve_fit(psychometric, x, y, p0=p0,
                            bounds=([0.1, 0.1], [10, 10]),
                            maxfev=5000)
        return popt
    except Exception as e:
        print(f"    Fit failed: {e}")
        return (np.nan, np.nan)


# Group-level fit per force, using collapsed absolute offset
print("\n[5] Group-level psychometric fits:")
group_fits = {}
for force in sorted(agg_subj['force_g'].unique()):
    sub = agg_subj[agg_subj['force_g'] == force]
    # collapse over sign (assuming symmetry — verified in step 4)
    coll = (sub.groupby('abs_offset_mm')['accuracy']
               .mean().reset_index())
    jnd, slope = fit_psychometric(coll['abs_offset_mm'].values,
                                  coll['accuracy'].values)
    group_fits[force] = {'jnd': jnd, 'slope': slope, 'data': coll}
    print(f"    Force={force}g: JND={jnd:.2f} mm, slope={slope:.2f}")


# ============================================================
# 6. PER-SUBJECT JND EXTRACTION
# ============================================================
print("\n[6] Per-subject psychometric fits ...")
subj_records = []
for (subj, force), grp in agg_subj.groupby(['Subject', 'force_g']):
    coll = (grp.groupby('abs_offset_mm')['accuracy']
               .mean().reset_index())
    if len(coll) < 3:
        continue
    jnd, slope = fit_psychometric(coll['abs_offset_mm'].values,
                                  coll['accuracy'].values)
    subj_records.append({
        'Subject': subj, 'force_g': force,
        'jnd_mm': jnd, 'slope': slope
    })

jnd_df = pd.DataFrame(subj_records)
jnd_df.to_csv(os.path.join(OUTPUT_DIR, 'jnd_per_subject.csv'), index=False)
print(f"    Saved jnd_per_subject.csv (n={len(jnd_df)} fits)")

# Within-subject force comparison (paired test)
print("\n    Per-subject JND comparison (1.0g vs 26.0g):")
wide = jnd_df.pivot(index='Subject', columns='force_g', values='jnd_mm').dropna()
if len(wide.columns) >= 2 and len(wide) > 3:
    f_low, f_high = sorted(wide.columns)
    try:
        stat, p = stats.wilcoxon(wide[f_low], wide[f_high])
        print(f"    Wilcoxon (paired): W={stat:.2f}, p={p:.4f}")
        print(f"    Median JND @ {f_low}g = {wide[f_low].median():.2f} mm")
        print(f"    Median JND @ {f_high}g = {wide[f_high].median():.2f} mm")
    except Exception as e:
        print(f"    Failed: {e}")


# ============================================================
# 7. LOGISTIC LME (Distance x Force interaction)
# ============================================================
print("\n[7] Logistic Mixed-Effects (trial-level)")
try:
    import statsmodels.formula.api as smf
    # Trial-level: IsCorrect ~ abs_offset_mm * force_g + (1|Subject)
    df['force_centered'] = df['force_g'] - df['force_g'].mean()
    df['offset_centered'] = df['abs_offset_mm'] - df['abs_offset_mm'].mean()

    # Random intercept logistic GLM-MM via BinomialBayesMixedGLM proxy:
    # statsmodels has limited mixed logistic; use GEE as practical alt.
    # For a proper logistic LME, you can switch to pymer4/lme4 in R.
    from statsmodels.genmod.generalized_estimating_equations import GEE
    from statsmodels.genmod.cov_struct import Exchangeable
    from statsmodels.genmod.families import Binomial

    model = GEE.from_formula(
        "IsCorrect ~ abs_offset_mm * force_g",
        groups="Subject",
        data=df,
        cov_struct=Exchangeable(),
        family=Binomial()
    )
    result = model.fit()
    with open(os.path.join(OUTPUT_DIR, 'lme_results.txt'), 'w') as f:
        f.write(str(result.summary()))
    print(result.summary().tables[1])
except Exception as e:
    print(f"    LME fitting failed: {e}")
    print("    (R/pymer4 사용을 권장 — Python 환경에서 mixed logistic은 제한적)")


# ============================================================
# 8. VISUALIZATION
# ============================================================
print("\n[8] 시각화 생성 ...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
colors = {1.0: '#1f77b4', 26.0: '#d62728'}

# --- (a) Group psychometric curves ---
ax = axes[0]
x_smooth = np.linspace(0, 6.5, 200)
for force in sorted(group_fits.keys()):
    info = group_fits[force]
    data = info['data']
    color = colors.get(force, 'gray')
    # raw data
    ax.scatter(data['abs_offset_mm'], data['accuracy'],
               color=color, s=80, label=f"{force}g (data)", zorder=3)
    # fit
    if not np.isnan(info['jnd']):
        y_smooth = psychometric(x_smooth, info['jnd'], info['slope'])
        ax.plot(x_smooth, y_smooth, color=color, lw=2,
                label=f"{force}g fit (JND={info['jnd']:.2f}mm)")
        ax.axvline(info['jnd'], color=color, ls=':', alpha=0.5)

ax.axhline(THRESHOLD_CRITERION, color='red', ls='--', alpha=0.6,
           label=f'{int(THRESHOLD_CRITERION*100)}% threshold')
ax.axhline(0.5, color='gray', ls=':', alpha=0.5, label='Chance')
ax.set_xlabel('Absolute offset (mm)')
ax.set_ylabel('Accuracy')
ax.set_ylim(0.3, 1.05)
ax.set_title('Group-level psychometric fits')
ax.legend(loc='lower right', fontsize=8)
ax.grid(alpha=0.3)

# --- (b) Per-subject JND distribution ---
ax = axes[1]
forces = sorted(jnd_df['force_g'].unique())
positions = range(len(forces))
data_to_plot = [jnd_df.loc[jnd_df['force_g'] == f, 'jnd_mm'].dropna()
                for f in forces]
bp = ax.boxplot(data_to_plot, positions=positions, widths=0.5,
                patch_artist=True, showfliers=False)
for patch, f in zip(bp['boxes'], forces):
    patch.set_facecolor(colors.get(f, 'gray'))
    patch.set_alpha(0.5)

# overlay individual points (jittered) + connect by subject
for subj in jnd_df['Subject'].unique():
    sub = jnd_df[jnd_df['Subject'] == subj].sort_values('force_g')
    xs = [forces.index(f) + np.random.uniform(-0.05, 0.05)
          for f in sub['force_g']]
    ys = sub['jnd_mm'].values
    ax.plot(xs, ys, '-', color='gray', alpha=0.3, lw=0.8)
    ax.scatter(xs, ys, color='black', s=20, alpha=0.7, zorder=4)

ax.set_xticks(positions)
ax.set_xticklabels([f'{f}g' for f in forces])
ax.set_xlabel('Force condition')
ax.set_ylabel('JND (mm)')
ax.set_title('Per-subject spatial JND')
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
out_png = os.path.join(OUTPUT_DIR, 'psychometric_curves.png')
plt.savefig(out_png, dpi=200, bbox_inches='tight')
print(f"    저장 완료: {out_png}")

# --- (c) Symmetry visualization ---
fig2, ax = plt.subplots(figsize=(10, 5))
for force in sorted(agg_group['force_g'].unique()):
    sub = agg_group[agg_group['force_g'] == force].sort_values('offset_mm')
    color = colors.get(force, 'gray')
    ax.errorbar(sub['offset_mm'], sub['mean_acc'], yerr=sub['sem_acc'],
                marker='o', color=color, label=f'{force}g', capsize=3)
ax.axhline(THRESHOLD_CRITERION, color='red', ls='--', alpha=0.6)
ax.axhline(0.5, color='gray', ls=':', alpha=0.5)
ax.axvline(0, color='black', ls='-', alpha=0.3)
ax.set_xlabel('Signed offset (mm, negative=left)')
ax.set_ylabel('Accuracy (mean ± SEM)')
ax.set_title('Symmetry of spatial discrimination')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
out_png2 = os.path.join(OUTPUT_DIR, 'symmetry_plot.png')
plt.savefig(out_png2, dpi=200, bbox_inches='tight')
print(f"    저장 완료: {out_png2}")

print("\n=== Done. 모든 결과는 ./sd_outputs/ 에 저장됨. ===")