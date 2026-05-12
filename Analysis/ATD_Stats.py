import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import statsmodels.formula.api as smf
import warnings

warnings.filterwarnings("ignore")


def build_pairwise_lme_p_matrices(df_input, subject_col, area_order, force_values):
    """Build pairwise LME p-value matrices by changing treatment reference area."""
    all_p_matrices = {}

    for force_val in force_values:
        subset = df_input[df_input["Force_Val"] == force_val].copy()
        p_matrix = pd.DataFrame(np.nan, index=area_order, columns=area_order)

        if subset.empty:
            all_p_matrices[force_val] = p_matrix
            continue

        for ref_area in area_order:
            ref_subset = subset[subset["Area"].isin(area_order)].copy()
            if ref_subset["Area"].nunique() < 2:
                continue

            try:
                formula = f"Relative_Score ~ C(Area, Treatment(reference='{ref_area}'))"
                model = smf.mixedlm(formula, ref_subset, groups=ref_subset[subject_col])
                result = model.fit()

                for target_area in area_order:
                    if ref_area == target_area:
                        p_matrix.loc[ref_area, target_area] = 1.0
                        continue

                    col_name = (
                        f"C(Area, Treatment(reference='{ref_area}'))[T.{target_area}]"
                    )
                    if col_name in result.pvalues:
                        p_matrix.loc[ref_area, target_area] = result.pvalues[col_name]
            except Exception:
                continue

        all_p_matrices[force_val] = p_matrix

    return all_p_matrices


def gender_lme_pvalue(subset, subject_col, score_col="Relative_Score", gender_col="Gender"):
    """
    Linear mixed model: score ~ Gender, random intercept for subject.
    Returns (p_value, ref_level, other_level) for the non-reference gender, or (nan, None, None).
    """
    sub = subset.dropna(subset=[subject_col, gender_col, score_col]).copy()
    if sub.empty or sub[subject_col].nunique() < 2:
        return np.nan, None, None
    levels = sorted(sub[gender_col].astype(str).unique())
    if len(levels) < 2:
        return np.nan, None, None
    ref_g = levels[0]
    other_g = levels[1]
    formula = f"{score_col} ~ C({gender_col}, Treatment(reference='{ref_g}'))"
    try:
        result = smf.mixedlm(formula, sub, groups=sub[subject_col]).fit()
        col = f"C({gender_col}, Treatment(reference='{ref_g}'))[T.{other_g}]"
        if col not in result.pvalues.index:
            return np.nan, ref_g, other_g
        return float(result.pvalues[col]), ref_g, other_g
    except Exception:
        return np.nan, ref_g, other_g


# 1. 데이터 로드 및 전처리
file_pattern = "/Users/kyungeunjung/NailFoldExp/Data/(ATD)CurData/P*_AbsoluteThresholdDetection.csv"
all_files = glob.glob(file_pattern)

if not all_files:
    print("CSV 파일을 찾을 수 없습니다.")
else:
    df_list = [pd.read_csv(f) for f in all_files]
    df = pd.concat(df_list, ignore_index=True)
    sub_col = "SubjectID" if "SubjectID" in df.columns else "Subject"

    df["Force_Val"] = df["Force"].str.extract(r"(\d+\.?\d*)").astype(float)
    df["Condition"] = (
        df["Condition"]
        .str.strip()
        .replace(
            {"Active": "On-touch (Mid)", "On-touch (Hard)": "On-touch (Mid)"}
        )
    )

    if "Gender" in df.columns:
        df["Gender"] = (
            df["Gender"].fillna("Unknown").astype(str).str.strip().str.upper()
        )

    def calc_relative_score(row):
        if row["Response"] == 0:
            return 0
        if row["Target"] == 0:
            return 100 if row["Response"] == 0 else 0
        error_ratio = abs(row["Target"] - row["Response"]) / row["Target"]
        return max(0, (1 - error_ratio) * 100)

    df["Relative_Score"] = df.apply(calc_relative_score, axis=1)

    target_forces = sorted([0.16, 0.6, 1.0])
    df_analysis = df[
        (df["Condition"] == "On-touch (Mid)") & (df["Force_Val"].isin(target_forces))
    ].copy()

    if "Area" not in df_analysis.columns and "Region" in df_analysis.columns:
        df_analysis["Area"] = df_analysis["Region"]
    if "Area" not in df_analysis.columns:
        raise ValueError(
            "ATD_Stats: LME heatmaps require an 'Area' column (or 'Region' to alias as Area)."
        )

    # --- 시각화 설정 ---
    plt.figure(figsize=(12, 8))
    sns.set_theme(style="whitegrid")
    palette = {"M": "#4C72B0", "F": "#C44E52"}

    ax = sns.boxplot(
        data=df_analysis,
        x="Force_Val",
        y="Relative_Score",
        hue="Gender",
        palette=palette,
        width=0.6,
        boxprops=dict(alpha=0.3),
        fliersize=0,
    )

    sns.stripplot(
        data=df_analysis,
        x="Force_Val",
        y="Relative_Score",
        hue="Gender",
        dodge=True,
        palette=palette,
        alpha=0.4,
        size=4,
        ax=ax,
        legend=False,
    )

    # --- LME: Gender effect at each force (random intercept = subject) ---
    print("\n" + "=" * 50)
    print(f"{'Force':<10} | {'LME p (Gender)':<15} | {'Significance'}")
    print("-" * 50)

    for i, f_val in enumerate(target_forces):
        subset = df_analysis[df_analysis["Force_Val"] == f_val]

        m_scores = subset[subset["Gender"] == "M"]["Relative_Score"]
        f_scores = subset[subset["Gender"] == "F"]["Relative_Score"]

        p_val, ref_g, other_g = gender_lme_pvalue(subset, sub_col)

        if not np.isnan(p_val):
            if p_val < 0.001:
                star = "***"
            elif p_val < 0.01:
                star = "**"
            elif p_val < 0.05:
                star = "*"
            else:
                star = "n.s."

            print(f"{f_val:<10.2f} | {p_val:<15.4f} | {star}  (ref={ref_g} vs {other_g})")

            m_med, f_med = m_scores.median(), f_scores.median()
            ax.text(
                i - 0.2,
                m_med + 1,
                f"{m_med:.1f}",
                color=palette["M"],
                fontweight="bold",
                ha="center",
                fontsize=10,
            )
            ax.text(
                i + 0.2,
                f_med + 1,
                f"{f_med:.1f}",
                color=palette["F"],
                fontweight="bold",
                ha="center",
                fontsize=10,
            )

            if star != "n.s.":
                y_max = 115
                ax.text(
                    i,
                    y_max,
                    star,
                    ha="center",
                    va="bottom",
                    color="red",
                    fontsize=20,
                    fontweight="bold",
                )
                ax.text(
                    i,
                    y_max - 5,
                    f"p={p_val:.3f}",
                    ha="center",
                    va="top",
                    color="black",
                    fontsize=9,
                )
        else:
            print(f"{f_val:<10.2f} | LME 불가 (성별/피험자 수준 부족)")

    print("=" * 50)

    plt.title(
        "Gender accuracy (LME: random intercept per subject)",
        fontsize=16,
        fontweight="bold",
    )
    plt.ylim(-5, 140)
    plt.tight_layout()

    # --- Area pairwise LME p-value heatmaps (always LME) ---
    areas = ["A", "B", "C", "D", "E", "F"]
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
        axes[i].set_title(f"Force {f_val}g: Pairwise LME p-values")
        axes[i].set_xlabel("Compared Area")
        axes[i].set_ylabel("Reference Area")

    plt.tight_layout()
    plt.show()
