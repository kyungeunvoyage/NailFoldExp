import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import glob
import statsmodels.formula.api as smf
import warnings

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(SCRIPT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# Shared figure palette
SLATE_BLUE = "#56708A"
OLIVE      = "#686F12"
WINE       = "#7F212B"
CREAM      = "#EDE2D0"
PALETTE_4  = [SLATE_BLUE, OLIVE, WINE, CREAM]
ATD_CMAP   = LinearSegmentedColormap.from_list("atd", [CREAM, SLATE_BLUE, OLIVE, WINE])

ENABLE_FIG1 = False
FIG2_SIZE   = (22.0, 6.0)   # pairwise LME heatmaps
FIG3_SIZE   = (14.5, 5.2)   # lateral / proximal contrasts
SAVE_DPI    = 220


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


def lme_area_pair_contrast(df_in, sub_col, ref_area, target_area):
    """
    Trial-level LME on two areas only:
    Relative_Score ~ C(Area, ref) + C(Force_Val), random intercept ~ Subject.
    Coefficient = mean(target) - mean(ref), conditional on Force effects.
    """
    sub = df_in[df_in["Area"].isin([ref_area, target_area])].copy()
    sub = sub.dropna(subset=[sub_col, "Relative_Score", "Area", "Force_Val"])
    if len(sub) < 10 or sub[sub_col].nunique() < 2 or sub["Area"].nunique() < 2:
        return None
    formula = (
        f"Relative_Score ~ C(Area, Treatment(reference='{ref_area}')) + C(Force_Val)"
    )
    try:
        res = smf.mixedlm(formula, sub, groups=sub[sub_col]).fit()
        col = f"C(Area, Treatment(reference='{ref_area}'))[T.{target_area}]"
        if col not in res.params.index:
            return None
        coef = float(res.params[col])
        pval = float(res.pvalues[col])
        ci_row = res.conf_int().loc[col]
        lo, hi = float(ci_row[0]), float(ci_row[1])
        return {
            "label": f"{target_area} − {ref_area}",
            "coef": coef,
            "ci_lo": lo,
            "ci_hi": hi,
            "p": pval,
        }
    except Exception:
        return None


def _star_from_p(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def subject_mean_accuracy_long(df_in, sub_col, area_first, area_second):
    """Per-subject mean accuracy in each area (trials pooled across forces in df_in)."""
    sub = df_in[df_in["Area"].isin([area_first, area_second])].dropna(
        subset=[sub_col, "Area", "Relative_Score"]
    )
    if sub.empty:
        return pd.DataFrame(columns=[sub_col, "Area", "accuracy", "contrast"])
    g = (
        sub.groupby([sub_col, "Area"], as_index=False)["Relative_Score"]
        .mean()
        .rename(columns={"Relative_Score": "accuracy"})
    )
    g["contrast"] = f"{area_first} vs {area_second}"
    return g


def build_contrast_tables(df_in, sub_col, pair_specs):
    """
    pair_specs: list of (area_first, area_second).
    LME: ref = outer area, coef = inner − outer, + C(Force_Val), RE subject.
    """
    plot_parts = []
    lme_by_label = {}
    for a1, a2 in pair_specs:
        label = f"{a1} vs {a2}"
        plot_parts.append(subject_mean_accuracy_long(df_in, sub_col, a1, a2))
        lme_by_label[label] = lme_area_pair_contrast(df_in, sub_col, a2, a1)
    plot_df = pd.concat(plot_parts, ignore_index=True) if plot_parts else pd.DataFrame()
    return plot_df, lme_by_label


AREA_PALETTE = {
    area: PALETTE_4[i % len(PALETTE_4)]
    for i, area in enumerate(["A", "B", "C", "D", "E", "F"])
}


def plot_accuracy_contrast_panel(ax, plot_df, lme_by_label, title):
    """Y = accuracy (%); grouped boxplot by contrast (two areas); LME p from trial-level model."""
    order = list(lme_by_label.keys())
    if plot_df.empty or not order:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title, fontsize=12)
        return

    hue_order = ["A", "B", "C", "D", "E", "F"]
    pal = {k: v for k, v in AREA_PALETTE.items() if k in plot_df["Area"].unique()}

    sns.boxplot(
        data=plot_df,
        x="contrast",
        y="accuracy",
        hue="Area",
        order=order,
        hue_order=[a for a in hue_order if a in plot_df["Area"].unique()],
        palette=pal,
        width=0.72,
        fliersize=0,
        linewidth=1.1,
        ax=ax,
    )

    ax.axhline(80, color=WINE, linestyle="--", linewidth=1, alpha=0.55)
    ax.set_ylabel("Accuracy (relative score, %)", fontsize=11)
    ax.set_xlabel("Contrast (LME: inner − outer area, + Force)", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylim(-5, 118)
    ax.grid(axis="y", alpha=0.35)
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    h2, l2 = [], []
    for hi, li in zip(handles, labels):
        if li not in seen:
            seen.add(li)
            h2.append(hi)
            l2.append(li)
    ax.legend(h2, l2, title="Area", frameon=False, loc="lower right")
    ax.tick_params(axis="x", rotation=20)

    for i, cv in enumerate(order):
        sub_y = plot_df.loc[plot_df["contrast"] == cv, "accuracy"]
        if sub_y.empty:
            continue
        y_ann = float(sub_y.max()) + 4.0
        r = lme_by_label.get(cv)
        if r is None:
            ax.text(i, y_ann, "LME fail", ha="center", fontsize=8, color=SLATE_BLUE)
        else:
            ax.text(
                i,
                y_ann,
                f"{_star_from_p(r['p'])}\np={r['p']:.3f}",
                ha="center",
                fontsize=8,
                color=WINE,
            )


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

    sns.set_theme(style="whitegrid")
    palette = {"M": SLATE_BLUE, "F": OLIVE}

    # --- Figure 1: Gender (disabled) — LME stats printed only ---
    print("\n" + "=" * 50)
    print(f"{'Force':<10} | {'LME p (Gender)':<15} | {'Significance'}")
    print("-" * 50)

    ax_gender = None
    fig_gender = None
    if ENABLE_FIG1:
        fig_gender = plt.figure(figsize=(12, 8), facecolor="white")
        ax_gender = sns.boxplot(
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
            ax=ax_gender,
            legend=False,
        )

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

            if ax_gender is not None:
                m_med, f_med = m_scores.median(), f_scores.median()
                ax_gender.text(
                    i - 0.2, m_med + 1, f"{m_med:.1f}",
                    color=palette["M"], fontweight="bold", ha="center", fontsize=10,
                )
                ax_gender.text(
                    i + 0.2, f_med + 1, f"{f_med:.1f}",
                    color=palette["F"], fontweight="bold", ha="center", fontsize=10,
                )
                if star != "n.s.":
                    ax_gender.text(
                        i, 115, star, ha="center", va="bottom",
                        color=WINE, fontsize=20, fontweight="bold",
                    )
                    ax_gender.text(
                        i, 110, f"p={p_val:.3f}", ha="center", va="top",
                        color="black", fontsize=9,
                    )
        else:
            print(f"{f_val:<10.2f} | LME 불가 (성별/피험자 수준 부족)")

    print("=" * 50)

    if fig_gender is not None:
        ax_gender.set_title(
            "Gender accuracy (LME: random intercept per subject)",
            fontsize=16, fontweight="bold",
        )
        ax_gender.set_ylim(-5, 140)
        fig_gender.tight_layout()
        out_gender = os.path.join(FIG_DIR, "gender_accuracy.png")
        fig_gender.savefig(out_gender, dpi=SAVE_DPI, bbox_inches="tight", facecolor="white")
        print(f"Saved: {out_gender}")
        plt.close(fig_gender)

    # --- Figure 2: Area pairwise LME p-value heatmaps ---
    areas = ["A", "B", "C", "D", "E", "F"]
    all_p_matrices = build_pairwise_lme_p_matrices(
        df_input=df_analysis,
        subject_col=sub_col,
        area_order=areas,
        force_values=target_forces,
    )

    fig2, axes = plt.subplots(
        1, len(target_forces), figsize=FIG2_SIZE, facecolor="white"
    )
    if len(target_forces) == 1:
        axes = [axes]

    for i, f_val in enumerate(target_forces):
        sns.heatmap(
            all_p_matrices[f_val],
            annot=True,
            fmt=".3f",
            cmap=ATD_CMAP,
            ax=axes[i],
            vmin=0,
            vmax=0.1,
        )
        axes[i].set_title(f"Force {f_val}g: Pairwise LME p-values")
        axes[i].set_xlabel("Compared Area")
        axes[i].set_ylabel("Reference Area")

    fig2.tight_layout()
    out_hm = os.path.join(FIG_DIR, "pairwise_lme_heatmap.png")
    fig2.savefig(out_hm, dpi=SAVE_DPI, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out_hm}  ({FIG2_SIZE[0]}×{FIG2_SIZE[1]} in @ {SAVE_DPI} dpi)")
    plt.close(fig2)

    # --- Figure 3: Lateral / proximal contrasts ---
    pairs_left = [("A", "C"), ("A", "D"), ("B", "C"), ("B", "D")]
    pairs_right = [("E", "C"), ("E", "D"), ("F", "C"), ("F", "D")]

    plot_left, lme_left = build_contrast_tables(df_analysis, sub_col, pairs_left)
    plot_right, lme_right = build_contrast_tables(df_analysis, sub_col, pairs_right)

    print(
        "\n[LME contrasts | trial-level: first − second, + Force_Val, RE=Subject]"
    )
    for lab, d in list(lme_left.items()) + list(lme_right.items()):
        if d is None:
            print(f"  {lab}: LME failed")
        else:
            print(
                f"  {lab}: Δcoef={d['coef']:.3f} [{d['ci_lo']:.3f}, {d['ci_hi']:.3f}], p={d['p']:.4f}"
            )

    fig3, (ax_l, ax_r) = plt.subplots(
        1, 2, figsize=FIG3_SIZE, gridspec_kw={"wspace": 0.28}, facecolor="white"
    )
    fig3.suptitle(
        "Lateral / proximal accuracy by region contrast (LME inference)",
        fontsize=13,
        y=1.02,
    )
    plot_accuracy_contrast_panel(
        ax_l,
        plot_left,
        lme_left,
        "A–C, A–D, B–C, B–D",
    )
    plot_accuracy_contrast_panel(
        ax_r,
        plot_right,
        lme_right,
        "E–C, E–D, F–C, F–D",
    )
    fig3.tight_layout()
    out_lp = os.path.join(FIG_DIR, "lateral_proximal_accuracy.png")
    fig3.savefig(out_lp, dpi=SAVE_DPI, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out_lp}  ({FIG3_SIZE[0]}×{FIG3_SIZE[1]} in @ {SAVE_DPI} dpi)")
    plt.close(fig3)
