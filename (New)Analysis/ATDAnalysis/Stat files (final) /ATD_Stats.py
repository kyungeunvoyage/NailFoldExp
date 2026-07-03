import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import glob
import statsmodels.formula.api as smf
import warnings
from itertools import combinations

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(SCRIPT_DIR, "Stats_Fig")
os.makedirs(FIG_DIR, exist_ok=True)

# Shared figure palette
SLATE_BLUE = "#56708A"
OLIVE      = "#686F12"
WINE       = "#7F212B"
CREAM      = "#EDE2D0"
BOX_ALPHA_HEX = "BE"
PALETTE_4  = [SLATE_BLUE, OLIVE, WINE, CREAM]
ATD_CMAP   = LinearSegmentedColormap.from_list("atd", [CREAM, SLATE_BLUE, OLIVE, WINE])

ENABLE_FIG1 = False
#0.07, 1.0, 1.4 빼고 싶은거..
#EXCLUDE_FORCES = {0.07, 1.0, 1.4}   # omit from all plots / plot-filtered analyses
EXCLUDE_FORCES = {}   # omit from all plots / plot-filtered analyses
FIG2_PANEL_W = 2.85   # inches per 6×6 heatmap panel
FIG2_HEIGHT  = 4.0    # fixed height for pairwise LME heatmaps
FIG4_SIZE    = (10.2, 4.8)   # force-pooled: heatmap + region boxplot + brackets
FIG3_SIZE   = (14.5, 5.2)   # lateral / proximal contrasts
SAVE_DPI    = 600


def _pairwise_lme_p_matrix(subset, subject_col, area_order):
    """One pairwise LME p-value matrix for a trial subset."""
    p_matrix = pd.DataFrame(np.nan, index=area_order, columns=area_order)
    ref_subset = subset[subset["Area"].isin(area_order)].copy()
    if ref_subset.empty:
        return p_matrix

    for ref_area in area_order:
        if ref_subset["Area"].nunique() < 2:
            continue
        try:
            formula = f"Relative_Score ~ C(Area, Treatment(reference='{ref_area}'))"
            result = smf.mixedlm(formula, ref_subset, groups=ref_subset[subject_col]).fit()
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
    return p_matrix


def build_pairwise_lme_p_matrices(df_input, subject_col, area_order, force_values):
    """Build pairwise LME p-value matrices by changing treatment reference area."""
    all_p_matrices = {}
    for force_val in force_values:
        subset = df_input[df_input["Force_Val"] == force_val].copy()
        all_p_matrices[force_val] = _pairwise_lme_p_matrix(
            subset, subject_col, area_order
        )
    return all_p_matrices


def build_pairwise_lme_p_matrix_pooled(df_input, subject_col, area_order):
    """Pairwise LME p-values with all forces aggregated (no Force term)."""
    return _pairwise_lme_p_matrix(df_input, subject_col, area_order)


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


def _add_sig_bracket(ax, x_l, x_r, y_base, tick_h=0.45, text="", fontsize=6.0):
    """Bracket from each box center; compact for multi-pair stacking."""
    x_center = (x_l + x_r) / 2.0
    y_top = y_base + tick_h
    ax.plot(
        [x_l, x_l, x_r, x_r],
        [y_base, y_top, y_top, y_base],
        color=WINE,
        linewidth=0.7,
        clip_on=False,
        zorder=5,
    )
    ax.text(
        x_center,
        y_top + 0.25,
        text,
        ha="center",
        va="bottom",
        fontsize=fontsize,
        color=WINE,
        fontweight="bold",
        clip_on=False,
        zorder=6,
    )


def lme_area_pair_pooled(df_in, sub_col, ref_area, target_area):
    """Trial-level area contrast with all forces pooled (no Force term)."""
    sub = df_in[df_in["Area"].isin([ref_area, target_area])].copy()
    sub = sub.dropna(subset=[sub_col, "Relative_Score", "Area"])
    if len(sub) < 10 or sub[sub_col].nunique() < 2 or sub["Area"].nunique() < 2:
        return None
    formula = f"Relative_Score ~ C(Area, Treatment(reference='{ref_area}'))"
    try:
        res = smf.mixedlm(formula, sub, groups=sub[sub_col]).fit()
        col = f"C(Area, Treatment(reference='{ref_area}'))[T.{target_area}]"
        if col not in res.params.index:
            return None
        return {"p": float(res.pvalues[col]), "coef": float(res.params[col])}
    except Exception:
        return None


def _boxplot_x_center(ax, cat_index):
    if not ax.containers:
        return float(cat_index)
    boxes = ax.containers[0].boxes
    if cat_index >= len(boxes):
        return float(cat_index)
    ext = boxes[cat_index].get_path().get_extents()
    return 0.5 * (ext.xmin + ext.xmax)


def _boxplot_whisker_top(ax, cat_index):
    if not ax.containers:
        return None
    whiskers = ax.containers[0].whiskers
    if cat_index >= len(whiskers) // 2:
        return None
    return max(whiskers[2 * cat_index + 1].get_ydata())


def annotate_region_pairwise_brackets(ax, areas, df_in, sub_col, *, bracket_step=2.5, alpha=0.05):
    """Significant pairwise region LME brackets only (forces pooled), stacked by span."""
    pair_stats = []
    for i, j in combinations(range(len(areas)), 2):
        r = lme_area_pair_pooled(df_in, sub_col, areas[i], areas[j])
        if r is not None and r["p"] < alpha:
            pair_stats.append((i, j, areas[i], areas[j], r))

    tops = [
        _boxplot_whisker_top(ax, k)
        for k in range(len(areas))
        if _boxplot_whisker_top(ax, k) is not None
    ]
    if not tops or not pair_stats:
        return ax.get_ylim()[1]

    y0 = max(tops) + 0.8
    y_max = y0
    for level, (i, j, _a1, _a2, r) in enumerate(
        sorted(pair_stats, key=lambda t: (t[1] - t[0], t[0]))
    ):
        y_base = y0 + level * bracket_step
        _add_sig_bracket(
            ax,
            _boxplot_x_center(ax, i),
            _boxplot_x_center(ax, j),
            y_base,
            text=f"{_star_from_p(r['p'])} p={r['p']:.3f}",
        )
        y_max = max(y_max, y_base + 2.2)
    return y_max


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


def log_region_accuracy_txt(region_subj, areas, sub_col, out_path):
    """Write per-region accuracy summary and per-subject means to a text file."""
    lines = [
        "Figure 4 — Subject mean accuracy by region",
        "On-touch (Mid) | all forces aggregated",
        "",
        "=== Group summary (subject means, %) ===",
        f"{'Region':<8} {'Mean':>8} {'SD':>8} {'SEM':>8} {'Median':>8} "
        f"{'Min':>8} {'Max':>8} {'N':>4}",
        "-" * 68,
    ]
    for area in areas:
        vals = region_subj.loc[region_subj["Area"] == area, "accuracy"]
        n = len(vals)
        if n == 0:
            lines.append(f"{area:<8} {'—':>8} {'—':>8} {'—':>8} {'—':>8} {'—':>8} {'—':>8} {0:>4}")
            continue
        mean = float(vals.mean())
        sd = float(vals.std(ddof=1)) if n > 1 else 0.0
        sem = sd / np.sqrt(n) if n > 1 else 0.0
        lines.append(
            f"{area:<8} {mean:8.3f} {sd:8.3f} {sem:8.3f} "
            f"{float(vals.median()):8.3f} {float(vals.min()):8.3f} "
            f"{float(vals.max()):8.3f} {n:4d}"
        )

    wide = region_subj.pivot(index=sub_col, columns="Area", values="accuracy")
    wide = wide.reindex(columns=areas)
    lines.extend(["", "=== Per-subject mean accuracy (%) ==="])
    header = f"{'Subject':<12}" + "".join(f"{a:>10}" for a in areas)
    lines.append(header)
    lines.append("-" * len(header))
    for subj, row in wide.sort_index().iterrows():
        cells = "".join(
            f"{row[a]:10.3f}" if pd.notna(row[a]) else f"{'—':>10}" for a in areas
        )
        lines.append(f"{str(subj):<12}{cells}")

    text = "\n".join(lines) + "\n"
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"Saved: {out_path}")
    print("\n  Region accuracy summary (subject means, %):")
    for line in lines[4:11]:
        print(f"  {line}")


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

    n_force_hm = len(plot_forces)
    cbar_col_ratio = 0.07
    fig2_w = n_force_hm * FIG2_PANEL_W + 0.45
    fig2 = plt.figure(figsize=(fig2_w, FIG2_HEIGHT), facecolor="white")
    gs = gridspec.GridSpec(
        1,
        n_force_hm + 1,
        figure=fig2,
        width_ratios=[1.0] * n_force_hm + [cbar_col_ratio],
        wspace=0.14,
    )
    axes = [fig2.add_subplot(gs[0, i]) for i in range(n_force_hm)]
    cax = fig2.add_subplot(gs[0, n_force_hm])

    hm_mappable = None
    for i, f_val in enumerate(plot_forces):
        hm = sns.heatmap(
            all_p_matrices[f_val],
            annot=True,
            fmt=".3f",
            annot_kws={"size": 7},
            cmap=ATD_CMAP,
            ax=axes[i],
            vmin=0,
            vmax=0.1,
            square=True,
            cbar=False,
            linewidths=0.4,
            linecolor="white",
        )
        hm_mappable = hm.collections[0]
        axes[i].set_aspect("equal", adjustable="box")
        axes[i].set_title(f"{f_val:g} g", fontsize=10, fontweight="bold", pad=6)
        axes[i].set_xlabel("Compared Area", fontsize=9)
        if i == 0:
            axes[i].set_ylabel("Reference Area", fontsize=9)
        else:
            axes[i].set_ylabel("")
        axes[i].tick_params(labelsize=8)

    if hm_mappable is not None:
        fig2.colorbar(
            hm_mappable,
            cax=cax,
            label="p-value",
        )
        cax.tick_params(labelsize=8)
        cax.yaxis.label.set_size(9)

    fig2.subplots_adjust(left=0.06, right=0.96, top=0.90, bottom=0.14)
    out_hm = os.path.join(FIG_DIR, "pairwise_lme_heatmap.png")
    fig2.savefig(out_hm, dpi=SAVE_DPI, facecolor="white")
    print(f"Saved: {out_hm}  ({fig2_w:.1f}×{FIG2_HEIGHT} in @ {SAVE_DPI} dpi)")
    plt.close(fig2)

    # --- Figure 4: All forces pooled — region-only pairwise LME + accuracy by region ---
    p_matrix_pooled = build_pairwise_lme_p_matrix_pooled(
        df_analysis, sub_col, areas
    )

    print(
        "\n[Figure 4 — Force-pooled region LME | all forces aggregated, On-touch (Mid)]"
    )
    try:
        area_main = smf.mixedlm(
            "Relative_Score ~ C(Area)",
            df_analysis,
            groups=df_analysis[sub_col],
        ).fit()
        print("  Omnibus C(Area) (mixed model):")
        for idx in area_main.pvalues.index:
            if idx.startswith("C(Area)"):
                print(f"    {idx}: p={area_main.pvalues[idx]:.4f}")
    except Exception as exc:
        print(f"  Omnibus Area LME failed: {exc}")

    region_subj = (
        df_analysis.groupby([sub_col, "Area"], as_index=False)["Relative_Score"]
        .mean()
        .rename(columns={"Relative_Score": "accuracy"})
    )
    out_acc_txt = os.path.join(FIG_DIR, "region_pairwise_pooled_forces_accuracy.txt")
    log_region_accuracy_txt(region_subj, areas, sub_col, out_acc_txt)
    region_pal = {a: AREA_PALETTE[a] for a in areas}

    fig4 = plt.figure(figsize=FIG4_SIZE, facecolor="white")
    gs4 = gridspec.GridSpec(
        1, 2, figure=fig4, width_ratios=[1.0, 1.05], wspace=0.48
    )
    ax_hm4 = fig4.add_subplot(gs4[0, 0])
    ax_bp4 = fig4.add_subplot(gs4[0, 1])

    hm4 = sns.heatmap(
        p_matrix_pooled,
        annot=True,
        fmt=".3f",
        annot_kws={"size": 8},
        cmap=ATD_CMAP,
        ax=ax_hm4,
        vmin=0,
        vmax=0.1,
        square=True,
        cbar=False,
        linewidths=0.4,
        linecolor="white",
    )
    ax_hm4.set_aspect("equal", adjustable="box")
    ax_hm4.set_title("All forces pooled", fontsize=10, fontweight="bold", pad=6)
    ax_hm4.set_xlabel("Compared Area", fontsize=9)
    ax_hm4.set_ylabel("Reference Area", fontsize=9)
    ax_hm4.tick_params(labelsize=8)

    cbar_div = make_axes_locatable(ax_hm4)
    cax4 = cbar_div.append_axes("right", size="5%", pad=0.10)
    cb4 = fig4.colorbar(hm4.collections[0], cax=cax4)
    cb4.set_label("p-value", fontsize=9)
    cax4.tick_params(labelsize=8)

    sns.boxplot(
        data=region_subj,
        x="Area",
        y="accuracy",
        order=areas,
        palette={k: v + BOX_ALPHA_HEX for k, v in region_pal.items()},
        width=0.62,
        fliersize=0,
        linewidth=1.0,
        ax=ax_bp4,
    )
    sns.stripplot(
        data=region_subj,
        x="Area",
        y="accuracy",
        order=areas,
        palette=region_pal,
        alpha=0.45,
        size=4,
        jitter=0.15,
        ax=ax_bp4,
        legend=False,
    )
    ax_bp4.axhline(80, color=WINE, linestyle="--", linewidth=1, alpha=0.55)
    ax_bp4.set_title("Subject mean accuracy by region", fontsize=10, fontweight="bold", pad=6)
    ax_bp4.set_xlabel("Region", fontsize=9)
    ax_bp4.set_ylabel("Accuracy (relative score, %)", fontsize=9, labelpad=8)
    ax_bp4.grid(axis="y", alpha=0.35)
    ax_bp4.tick_params(labelsize=8)

    print("  Pairwise region LME (forces pooled, trial-level):")
    for a1, a2 in combinations(areas, 2):
        r = lme_area_pair_pooled(df_analysis, sub_col, a1, a2)
        if r is None:
            print(f"    {a1} vs {a2}: LME failed")
        else:
            print(f"    {a1} vs {a2}: p={r['p']:.4f} ({_star_from_p(r['p'])})")

    y_top_bp = annotate_region_pairwise_brackets(
        ax_bp4, areas, df_analysis, sub_col, bracket_step=2.6
    )
    y_floor = 30
    if (region_subj["accuracy"] < y_floor).any():
        y_lo = max(-5, float(region_subj["accuracy"].min()) - 3)
    else:
        y_lo = y_floor
    ax_bp4.set_ylim(y_lo, min(115, y_top_bp + 2))

    fig4.suptitle(
        "Region differences with all forces aggregated (On-touch Mid, trial-level LME)",
        fontsize=11,
        y=0.98,
    )
    fig4.subplots_adjust(left=0.08, right=0.97, top=0.86, bottom=0.14)
    out_pooled = os.path.join(FIG_DIR, "region_pairwise_pooled_forces.png")
    fig4.savefig(out_pooled, dpi=SAVE_DPI, facecolor="white")
    print(f"Saved: {out_pooled}  ({FIG4_SIZE[0]}×{FIG4_SIZE[1]} in @ {SAVE_DPI} dpi)")
    plt.close(fig4)

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
