import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
BLACK      = "#1A1A1A"
BOX_ALPHA_HEX = "BE"
PALETTE_4  = [SLATE_BLUE, OLIVE, WINE, CREAM]
ATD_CMAP   = LinearSegmentedColormap.from_list("atd", [CREAM, SLATE_BLUE, OLIVE, WINE])

ENABLE_FIG1 = False
#여기서 빼고 싶은거..
EXCLUDE_FORCES = {0.07, 1.0, 1.4}   # omit from all plots / plot-filtered analyses
FIG2_SIZE   = (14.0, 6.0)   # pairwise LME heatmaps (width scales with # forces)
FIG3_SIZE   = (14.5, 5.2)   # lateral / proximal contrasts
FIG5_SIZE   = (9.0, 5.0)    # on-nail contrasts faceted by force
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


def _add_sig_bracket(ax, x_l, x_r, y_base, tick_h=0.5, text=""):
    """Bracket from each box center (x_l, x_r) up to a shared bar; compact vertical size."""
    x_center = (x_l + x_r) / 2.0
    y_top = y_base + tick_h
    ax.plot(
        [x_l, x_l, x_r, x_r],
        [y_base, y_top, y_top, y_base],
        color=WINE,
        linewidth=0.75,
        clip_on=False,
        zorder=5,
    )
    ax.text(
        x_center,
        y_top + 0.35,
        text,
        ha="center",
        va="bottom",
        fontsize=6.5,
        color=WINE,
        fontweight="bold",
        clip_on=False,
        zorder=6,
    )


def _style_boxplot(bp, facecolor):
    for patch in bp["boxes"]:
        patch.set_facecolor(facecolor + BOX_ALPHA_HEX)
        patch.set_edgecolor(BLACK)
        patch.set_linewidth(0.8)
    for w in bp["whiskers"]:
        w.set_color(BLACK)
        w.set_linewidth(0.8)
    for c in bp["caps"]:
        c.set_color(BLACK)
        c.set_linewidth(0.8)
    for m in bp["medians"]:
        m.set_color(BLACK)
        m.set_linewidth(1.5)


def plot_paired_contrast_boxes(
    ax,
    plot_df,
    order_nail,
    contrast_areas,
    palette,
    *,
    bar_w=0.17,
    pair_gap=0.05,
    group_gap=0.14,
    edge_pad_left=0.16,
    edge_pad_right=0.30,
    x_tick_labels=None,
):
    """
    Paired boxplots: equal bar geometry in every group, tight cluster, edge margin.
    Bar centers at xc ± (bar_w + pair_gap) / 2  →  gap between boxes = pair_gap.
    Returns {contrast: (x_center_left, x_center_right, whisker_top)} for brackets.
    """
    rng = np.random.default_rng(0)
    pair_offset = (bar_w + pair_gap) / 2.0
    group_step = 2 * pair_offset + bar_w + group_gap
    n_groups = len(order_nail)

    box_kw = dict(
        widths=bar_w,
        patch_artist=True,
        showfliers=False,
        zorder=2,
    )
    spans = {}
    group_centers = []
    all_edges = []

    for gi, cv in enumerate(order_nail):
        areas = contrast_areas[cv]
        xc = gi * group_step
        group_centers.append(xc)
        positions = [xc - pair_offset, xc + pair_offset]
        xmins, xmaxs, ytops = [], [], []

        for pos, area in zip(positions, areas):
            vals = plot_df.loc[
                (plot_df["contrast"] == cv) & (plot_df["Area"] == area), "accuracy"
            ].dropna().values
            if len(vals) == 0:
                continue
            bp = ax.boxplot([vals], positions=[pos], **box_kw)
            _style_boxplot(bp, palette[area])
            bx0, bx1 = pos - bar_w / 2, pos + bar_w / 2
            xmins.append(bx0)
            xmaxs.append(bx1)
            all_edges.extend([bx0, bx1])
            ytops.append(max(bp["whiskers"][1].get_ydata()))

            jitter = rng.uniform(-bar_w * 0.12, bar_w * 0.12, size=len(vals))
            ax.scatter(
                pos + jitter,
                vals,
                color=palette[area],
                alpha=0.4,
                s=14,
                linewidths=0,
                zorder=3,
            )

        if ytops:
            spans[cv] = (positions[0], positions[1], max(ytops))

    labels = x_tick_labels if x_tick_labels else order_nail
    ax.set_xticks(group_centers)
    ax.set_xticklabels(labels)
    if all_edges:
        ax.set_xlim(
            min(all_edges) - edge_pad_left,
            max(all_edges) + edge_pad_right,
        )
    return spans


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


def subject_mean_accuracy_long_v2(df_in, sub_col, area_first, area_second):
    """Per-subject mean accuracy for two areas (trials pooled across forces)."""
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


def subject_mean_accuracy_by_force_v2(
    df_in, sub_col, area_first, area_second, contrast_label
):
    """Per-subject mean accuracy by area and force (one force level per row)."""
    sub = df_in[df_in["Area"].isin([area_first, area_second])].dropna(
        subset=[sub_col, "Area", "Relative_Score", "Force_Val"]
    )
    if sub.empty:
        return pd.DataFrame(
            columns=[sub_col, "Area", "Force_Val", "accuracy", "contrast"]
        )
    g = (
        sub.groupby([sub_col, "Area", "Force_Val"], as_index=False)["Relative_Score"]
        .mean()
        .rename(columns={"Relative_Score": "accuracy"})
    )
    g["contrast"] = contrast_label
    return g


def lme_force_test(df_in, sub_col, areas=None, include_area=True):
    """
    Trial-level LME for Force_Val (ref = lowest force), RE=Subject.
    Optionally adjust for Area when multiple regions are in the subset.
    """
    sub = df_in.copy()
    if areas is not None:
        sub = sub[sub["Area"].isin(areas)]
    sub = sub.dropna(subset=[sub_col, "Relative_Score", "Force_Val", "Area"])
    forces = sorted(sub["Force_Val"].unique())
    if len(forces) < 2 or sub[sub_col].nunique() < 2:
        return None
    ref_f = forces[0]
    other_f = forces[1]
    if include_area and sub["Area"].nunique() > 1:
        formula = "Relative_Score ~ C(Force_Val) + C(Area)"
    else:
        formula = "Relative_Score ~ C(Force_Val)"
    try:
        res = smf.mixedlm(formula, sub, groups=sub[sub_col]).fit()
        col = f"C(Force_Val)[T.{other_f}]"
        if col not in res.params.index:
            return None
        ci = res.conf_int().loc[col]
        return {
            "label": f"{other_f}g − {ref_f}g",
            "coef": float(res.params[col]),
            "ci_lo": float(ci[0]),
            "ci_hi": float(ci[1]),
            "p": float(res.pvalues[col]),
            "ref_force": ref_f,
            "other_force": other_f,
        }
    except Exception:
        return None


def lme_force_area_interaction(df_in, sub_col, areas):
    """Trial-level LME: Force * Area interaction."""
    sub = df_in[df_in["Area"].isin(areas)].dropna(
        subset=[sub_col, "Relative_Score", "Force_Val", "Area"]
    )
    if sub["Force_Val"].nunique() < 2 or sub["Area"].nunique() < 2:
        return None
    formula = "Relative_Score ~ C(Force_Val) * C(Area)"
    try:
        res = smf.mixedlm(formula, sub, groups=sub[sub_col]).fit()
        interact_cols = [c for c in res.pvalues.index if ":" in c and "Force_Val" in c]
        out = {}
        for col in interact_cols:
            ci = res.conf_int().loc[col]
            out[col] = {
                "coef": float(res.params[col]),
                "ci_lo": float(ci[0]),
                "ci_hi": float(ci[1]),
                "p": float(res.pvalues[col]),
            }
        return out
    except Exception:
        return None


def lme_area_pair_at_force(df_in, sub_col, ref_area, target_area, force_val):
    """Area contrast LME at a single force level (no Force term)."""
    sub = df_in[
        df_in["Area"].isin([ref_area, target_area])
        & np.isclose(df_in["Force_Val"], force_val)
    ].dropna(subset=[sub_col, "Relative_Score", "Area"])
    if len(sub) < 10 or sub[sub_col].nunique() < 2:
        return None
    formula = f"Relative_Score ~ C(Area, Treatment(reference='{ref_area}'))"
    try:
        res = smf.mixedlm(formula, sub, groups=sub[sub_col]).fit()
        col = f"C(Area, Treatment(reference='{ref_area}'))[T.{target_area}]"
        if col not in res.params.index:
            return None
        ci = res.conf_int().loc[col]
        return {
            "coef": float(res.params[col]),
            "p": float(res.pvalues[col]),
        }
    except Exception:
        return None


def lme_area_pair_contrast_v2(df_in, sub_col, ref_area, target_area):
    """
    Trial-level LME on two areas:
    Relative_Score ~ C(Area, ref) + C(Force_Val), random intercept ~ Subject.
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

    all_forces = sorted(df["Force_Val"].unique())
    plot_forces = [f for f in all_forces if f not in EXCLUDE_FORCES]
    if not plot_forces:
        raise ValueError(
            f"No forces left for plotting after excluding {sorted(EXCLUDE_FORCES)}. "
            f"Available: {all_forces}"
        )
    print(f"Plot forces (g): {plot_forces}  |  excluded: {sorted(EXCLUDE_FORCES)}")

    df_analysis = df[
        (df["Condition"] == "On-touch (Mid)") & (df["Force_Val"].isin(plot_forces))
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

    for i, f_val in enumerate(plot_forces):
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
        force_values=plot_forces,
    )

    fig2, axes = plt.subplots(
        1, len(plot_forces), figsize=FIG2_SIZE, facecolor="white"
    )
    if len(plot_forces) == 1:
        axes = [axes]

    for i, f_val in enumerate(plot_forces):
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

    # --- Figure 4: On-Nail (C+D) vs Off-Nail (A, F) + A vs F ---
    df_onnail = df_analysis.copy()
    df_onnail["Area"] = df_onnail["Area"].replace({"C": "On-Nail", "D": "On-Nail"})
    df_onnail["Area"] = df_onnail["Area"].replace(
        {"A": "Off-Nail (A)", "F": "Off-Nail (F)"}
    )

    pairs_nail = [("On-Nail", "Off-Nail (A)"), ("On-Nail", "Off-Nail (F)")]

    plot_parts_nail = []
    lme_nail = {}
    for a_nail, a_off in pairs_nail:
        label = f"{a_nail} vs {a_off}"
        plot_parts_nail.append(
            subject_mean_accuracy_long_v2(df_onnail, sub_col, a_nail, a_off)
        )
        lme_nail[label] = lme_area_pair_contrast_v2(df_onnail, sub_col, a_off, a_nail)

    # Same Off-Nail labels as other panels (not raw "A" / "F")
    plot_af = subject_mean_accuracy_long_v2(
        df_onnail, sub_col, "Off-Nail (A)", "Off-Nail (F)"
    )
    if not plot_af.empty:
        plot_af["contrast"] = "A vs F"
    lme_nail["A vs F"] = lme_area_pair_contrast_v2(
        df_onnail, sub_col, "Off-Nail (F)", "Off-Nail (A)"
    )

    plot_df_nail = pd.concat(plot_parts_nail + [plot_af], ignore_index=True)

    print(
        "\n[On-Nail / Off-Nail / A–F LME contrasts | + Force_Val, RE=Subject]"
    )
    for lab, d in lme_nail.items():
        if d is None:
            print(f"  {lab}: LME failed")
        else:
            print(
                f"  {lab}: Δcoef={d['coef']:.3f} "
                f"[{d['ci_lo']:.3f}, {d['ci_hi']:.3f}], p={d['p']:.4f}"
            )

    NAIL_PALETTE = {
        "On-Nail": SLATE_BLUE,
        "Off-Nail (A)": OLIVE,
        "Off-Nail (F)": WINE,
    }
    REGION_ORDER = ["On-Nail", "Off-Nail (A)", "Off-Nail (F)"]

    order_nail = [
        "On-Nail vs Off-Nail (A)",
        "On-Nail vs Off-Nail (F)",
        "A vs F",
    ]

    CONTRAST_AREAS = {
        "On-Nail vs Off-Nail (A)": ["On-Nail", "Off-Nail (A)"],
        "On-Nail vs Off-Nail (F)": ["On-Nail", "Off-Nail (F)"],
        "A vs F": ["Off-Nail (A)", "Off-Nail (F)"],
    }

    fig4, ax4 = plt.subplots(figsize=(7.0, 5.2), facecolor="white")
    # fig4.suptitle(
    #     "On-Nail (C+D) vs Off-Nail (A, F) and A vs F — Accuracy Contrast\n"
    #     "(LME: first − second in contrast label, + Force, RE=Subject)",
    #     fontsize=11,
    #     y=0.98,
    # )

    x_labels = ["On vs Off (A)", "On vs Off (F)", "A vs F"]

    box_spans = plot_paired_contrast_boxes(
        ax4,
        plot_df_nail,
        order_nail,
        CONTRAST_AREAS,
        NAIL_PALETTE,
        bar_w=0.17,
        pair_gap=0.05,
        group_gap=0.14,
        edge_pad_left=0.14,
        edge_pad_right=0.30,
        x_tick_labels=x_labels,
    )

    ax4.axhline(80, color=WINE, linestyle="--", linewidth=1, alpha=0.55, zorder=0)
    ax4.set_ylabel("Accuracy (relative score, %)", fontsize=11)
    #ax4.set_xlabel("Contrast", fontsize=10)
    ax4.grid(axis="y", alpha=0.35)
    ax4.tick_params(axis="x", rotation=0, labelsize=8.5)

    y_ceiling = 80.0
    for cv in order_nail:
        span = box_spans.get(cv)
        if span is None:
            continue
        x_l, x_r, box_top = span
        y_ceiling = max(y_ceiling, box_top)
        y_bracket = box_top + 1.2
        r = lme_nail.get(cv)
        if r is None:
            sig_text = "LME fail"
        else:
            sig_text = f"{_star_from_p(r['p'])}  p={r['p']:.3f}"
        _add_sig_bracket(ax4, x_l, x_r, y_bracket, text=sig_text)
        y_ceiling = max(y_ceiling, y_bracket + 2.8)

    ax4.set_ylim(-5, min(105, y_ceiling + 3))

    leg_handles = [
        mpatches.Patch(facecolor=NAIL_PALETTE[r] + BOX_ALPHA_HEX, edgecolor=BLACK,
                       linewidth=0.6, label=r)
        for r in REGION_ORDER
    ]
    ax4.legend(
        handles=leg_handles,
        title="Region",
        frameon=False,
        loc="lower right",
        fontsize=8,
        title_fontsize=8,
    )

    fig4.subplots_adjust(left=0.10, right=0.97, top=0.94, bottom=0.14)
    out_nail = os.path.join(FIG_DIR, "onnail_vs_offnail_accuracy.png")
    fig4.savefig(out_nail, dpi=SAVE_DPI, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out_nail}")
    plt.close(fig4)

    # --- Figure 5: Same contrasts, split by force (0.16 g vs 0.6 g) ---
    plot_parts_f5 = []
    for a_nail, a_off in pairs_nail:
        plot_parts_f5.append(
            subject_mean_accuracy_by_force_v2(
                df_onnail, sub_col, a_nail, a_off, f"{a_nail} vs {a_off}"
            )
        )
    plot_af_f5 = subject_mean_accuracy_by_force_v2(
        df_onnail, sub_col, "Off-Nail (A)", "Off-Nail (F)", "A vs F"
    )
    if not plot_af_f5.empty:
        plot_parts_f5.append(plot_af_f5)
    plot_df_f5 = pd.concat(plot_parts_f5, ignore_index=True)

    nail_regions = ["On-Nail", "Off-Nail (A)", "Off-Nail (F)"]

    print(
        "\n[Figure 5 — Force effects | On-touch (Mid), trial-level LME, RE=Subject]"
    )
    r_all = lme_force_test(df_onnail, sub_col, areas=nail_regions, include_area=True)
    if r_all:
        print(
            f"  All regions: {r_all['label']} Δ={r_all['coef']:.3f} "
            f"[{r_all['ci_lo']:.3f}, {r_all['ci_hi']:.3f}], p={r_all['p']:.4f}"
        )
    else:
        print("  All regions: Force LME failed")

    r_no_area = lme_force_test(
        df_onnail, sub_col, areas=nail_regions, include_area=False
    )
    if r_no_area:
        print(f"  All regions (Force only): p={r_no_area['p']:.4f}")

    interact = lme_force_area_interaction(df_onnail, sub_col, nail_regions)
    if interact:
        for col, d in interact.items():
            print(
                f"  Interaction {col}: Δ={d['coef']:.3f}, p={d['p']:.4f}"
            )
    else:
        print("  Force × Area interaction: LME failed")

    for cv, areas_pair in CONTRAST_AREAS.items():
        sub_cv = df_onnail[df_onnail["Area"].isin(areas_pair)]
        r_cv = lme_force_test(sub_cv, sub_col, include_area=True)
        if r_cv:
            print(f"  {cv} — Force|Area: p={r_cv['p']:.4f}")
        for fval in plot_forces:
            r_af = lme_area_pair_at_force(
                df_onnail, sub_col, areas_pair[1], areas_pair[0], fval
            )
            if r_af:
                print(
                    f"    @ {fval:.2f}g Area contrast: p={r_af['p']:.4f}"
                )

    n_force_panels = len(plot_forces)
    fig5, axes5 = plt.subplots(
        1,
        n_force_panels,
        figsize=(FIG5_SIZE[0], FIG5_SIZE[1]),
        sharey=True,
        facecolor="white",
    )
    if n_force_panels == 1:
        axes5 = [axes5]

    for ax5, fval in zip(axes5, plot_forces):
        sub_f = plot_df_f5[np.isclose(plot_df_f5["Force_Val"], fval)].copy()
        spans_f = plot_paired_contrast_boxes(
            ax5,
            sub_f,
            order_nail,
            CONTRAST_AREAS,
            NAIL_PALETTE,
            bar_w=0.17,
            pair_gap=0.05,
            group_gap=0.14,
            edge_pad_left=0.14,
            edge_pad_right=0.22 if fval == plot_forces[-1] else 0.14,
            x_tick_labels=x_labels,
        )
        ax5.axhline(80, color=WINE, linestyle="--", linewidth=1, alpha=0.55, zorder=0)
        ax5.set_title(f"{fval:.2f} g", fontsize=10, fontweight="bold", pad=6)
        ax5.grid(axis="y", alpha=0.35)
        ax5.tick_params(axis="x", labelsize=8.5)

        y_ceil_f = 80.0
        for cv in order_nail:
            areas_pair = CONTRAST_AREAS[cv]
            span = spans_f.get(cv)
            if span is None:
                continue
            x_l, x_r, box_top = span
            y_ceil_f = max(y_ceil_f, box_top)
            y_bracket = box_top + 1.2
            r_af = lme_area_pair_at_force(
                df_onnail, sub_col, areas_pair[1], areas_pair[0], fval
            )
            if r_af is None:
                sig_text = "n.s."
            else:
                sig_text = f"{_star_from_p(r_af['p'])}  p={r_af['p']:.3f}"
            _add_sig_bracket(ax5, x_l, x_r, y_bracket, text=sig_text)
            y_ceil_f = max(y_ceil_f, y_bracket + 2.8)

        ax5.set_ylim(-5, min(105, y_ceil_f + 3))
        if fval == plot_forces[0]:
            ax5.set_ylabel("Accuracy (relative score, %)", fontsize=11)
        else:
            ax5.set_ylabel("")

    leg_handles_f5 = [
        mpatches.Patch(
            facecolor=NAIL_PALETTE[r] + BOX_ALPHA_HEX,
            edgecolor=BLACK,
            linewidth=0.6,
            label=r,
        )
        for r in REGION_ORDER
    ]
    axes5[-1].legend(
        handles=leg_handles_f5,
        title="Region",
        frameon=False,
        loc="lower right",
        fontsize=8,
        title_fontsize=8,
    )

    # fig5.suptitle(
    
    #     "On-Nail vs Off-Nail contrasts by force (On-touch Mid)\n"
    #     "Brackets: area contrast within force; see console for Force LME",
    #     fontsize=10,
    #     y=1.02,
    # )
    fig5.subplots_adjust(left=0.08, right=0.97, top=0.88, bottom=0.14, wspace=0.06)
    out_f5 = os.path.join(FIG_DIR, "onnail_vs_offnail_by_force.png")
    fig5.savefig(out_f5, dpi=SAVE_DPI, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out_f5}")
    plt.close(fig5)