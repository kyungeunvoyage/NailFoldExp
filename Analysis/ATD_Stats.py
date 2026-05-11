import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from scipy import stats
import statsmodels.formula.api as smf
import warnings

warnings.filterwarnings('ignore')


def build_pairwise_lme_p_matrices(df_input, subject_col, area_order, force_values):
    """Build pairwise LME p-value matrices by changing treatment reference area."""
    all_p_matrices = {}

    for force_val in force_values:
        subset = df_input[df_input['Force_Val'] == force_val].copy()
        p_matrix = pd.DataFrame(np.nan, index=area_order, columns=area_order)

        # 데이터가 부족하면 NaN 행렬 유지
        if subset.empty:
            all_p_matrices[force_val] = p_matrix
            continue

        for ref_area in area_order:
            ref_subset = subset[subset['Area'].isin(area_order)].copy()
            if ref_subset['Area'].nunique() < 2:
                continue

            try:
                formula = f"Relative_Score ~ C(Area, Treatment(reference='{ref_area}'))"
                model = smf.mixedlm(formula, ref_subset, groups=ref_subset[subject_col])
                result = model.fit()

                for target_area in area_order:
                    if ref_area == target_area:
                        p_matrix.loc[ref_area, target_area] = 1.0
                        continue

                    col_name = f"C(Area, Treatment(reference='{ref_area}'))[T.{target_area}]"
                    if col_name in result.pvalues:
                        p_matrix.loc[ref_area, target_area] = result.pvalues[col_name]
            except Exception:
                continue

        all_p_matrices[force_val] = p_matrix

    return all_p_matrices

# 1. 데이터 로드 및 전처리 (기존 로직 유지)
file_pattern = '/Users/kyungeunjung/NailFoldExp/Data/(ATD)CurData/P*_AbsoluteThresholdDetection.csv'
all_files = glob.glob(file_pattern)

if not all_files:
    print("CSV 파일을 찾을 수 없습니다.")
else:
    df_list = [pd.read_csv(f) for f in all_files]
    df = pd.concat(df_list, ignore_index=True)
    sub_col = 'SubjectID' if 'SubjectID' in df.columns else 'Subject'

    df['Force_Val'] = df['Force'].str.extract(r'(\d+\.?\d*)').astype(float)
    df['Condition'] = df['Condition'].str.strip().replace({'Active': 'On-touch (Mid)', 'On-touch (Hard)': 'On-touch (Mid)'})
    
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].fillna('Unknown').astype(str).str.strip().str.upper()

    def calc_relative_score(row):
        if row['Response'] == 0: return 0
        if row['Target'] == 0: return 100 if row['Response'] == 0 else 0
        error_ratio = abs(row['Target'] - row['Response']) / row['Target']
        return max(0, (1 - error_ratio) * 100)

    df['Relative_Score'] = df.apply(calc_relative_score, axis=1)

    target_forces = sorted([0.16, 0.6, 1.0])
    df_analysis = df[(df['Condition'] == 'On-touch (Mid)') & (df['Force_Val'].isin(target_forces))].copy()

    # --- 시각화 설정 ---
    plt.figure(figsize=(12, 8))
    sns.set_theme(style="whitegrid")
    palette = {"M": "#4C72B0", "F": "#C44E52"} 

    # 1. 박스플롯 (배경)
    ax = sns.boxplot(data=df_analysis, x='Force_Val', y='Relative_Score', hue='Gender',
                     palette=palette, width=0.6, boxprops=dict(alpha=0.3), fliersize=0)

    # 2. 개별 데이터 점 (Scatter/Stripplot)
    sns.stripplot(data=df_analysis, x='Force_Val', y='Relative_Score', hue='Gender',
                  dodge=True, palette=palette, alpha=0.4, size=4, ax=ax, legend=False)

    # --- T-Test 수행 및 별표(Asterisk) 표시 ---
    print("\n" + "="*50)
    print(f"{'Force':<10} | {'T-Test P-value':<15} | {'Significance'}")
    print("-" * 50)

    for i, f_val in enumerate(target_forces):
        subset = df_analysis[df_analysis['Force_Val'] == f_val]
        
        m_scores = subset[subset['Gender'] == 'M']['Relative_Score']
        f_scores = subset[subset['Gender'] == 'F']['Relative_Score']
        
        if len(m_scores) > 1 and len(f_scores) > 1:
            # Independent T-test (Welch's t-test)
            t_stat, p_val = stats.ttest_ind(m_scores, f_scores, equal_var=False)
            
            # 유의성 수준에 따른 별표 결정
            if p_val < 0.001: star = '***'
            elif p_val < 0.01: star = '**'
            elif p_val < 0.05: star = '*'
            else: star = 'n.s.'

            print(f"{f_val:<10.2f} | {p_val:<15.4f} | {star}")

            # 중앙값(Median) 표시
            m_med, f_med = m_scores.median(), f_scores.median()
            ax.text(i - 0.2, m_med + 1, f'{m_med:.1f}', color=palette['M'], fontweight='bold', ha='center', fontsize=10)
            ax.text(i + 0.2, f_med + 1, f'{f_med:.1f}', color=palette['F'], fontweight='bold', ha='center', fontsize=10)

            # 그래프 상단 별표(Asterisk) 추가
            if star != 'n.s.':
                y_max = 115
                ax.text(i, y_max, star, ha='center', va='bottom', color='red', fontsize=20, fontweight='bold')
                ax.text(i, y_max-5, f"p={p_val:.3f}", ha='center', va='top', color='black', fontsize=9)
        else:
            print(f"{f_val:<10.2f} | 데이터 부족")

    print("="*50)

    plt.title('Gender Accuracy Comparison (T-Test Based Stars)', fontsize=16, fontweight='bold')
    plt.ylim(-5, 140)
    plt.tight_layout()

    # 2) Area pairwise LME p-value heatmaps
    if 'Area' not in df_analysis.columns:
        print("'Area' 컬럼이 없어 LME heatmap 분석을 건너뜁니다.")
    else:
        areas = ['A', 'B', 'C', 'D', 'E', 'F']
        all_p_matrices = build_pairwise_lme_p_matrices(
            df_input=df_analysis,
            subject_col=sub_col,
            area_order=areas,
            force_values=target_forces,
        )

        fig, axes = plt.subplots(1, len(target_forces), figsize=(22, 6))
        if len(target_forces) == 1:
            axes = [axes]

        for i, f_val in enumerate(target_forces):
            sns.heatmap(
                all_p_matrices[f_val],
                annot=True,
                fmt=".3f",
                cmap="YlGnBu_r",
                ax=axes[i],
                vmin=0,
                vmax=0.1,
            )
            axes[i].set_title(f'Force {f_val}g: Pairwise LME p-values')
            axes[i].set_xlabel('Compared Area')
            axes[i].set_ylabel('Reference Area')

        plt.tight_layout()

    # show는 마지막에 한 번만 호출해서 뒤 플롯까지 모두 그리도록 함
    plt.show()
