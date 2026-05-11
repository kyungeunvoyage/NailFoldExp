import os
import pandas as pd
import numpy as np
import glob
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns


CHANCE_LEVEL = 0.75

# Export: fixed layout size (inches) + raster DPI (no interactive window unless FD_STATS_SHOW=1)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_EXPORT_DIR = os.path.join(SCRIPT_DIR, "figures")
EXPORT_DPI = 220
# Width/height tuned for axis labels + brackets + two stacked panels
FIGSIZE_BOX = (11.0, 9.25)
FIGSIZE_HEATMAP = (12.0, 5.35)


def export_figure(fig, path, dpi=EXPORT_DPI):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(
        path,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.14,
        facecolor="white",
        edgecolor="none",
    )


def pvalue_to_stars(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def draw_sig_bracket(ax, x_left, x_right, y, text, color="black"):
    """Horizontal bracket with label between two x positions (categorical indices)."""
    h = 0.02
    mid = (x_left + x_right) / 2
    ax.plot(
        [x_left, x_left, x_right, x_right],
        [y, y + h, y + h, y],
        color=color,
        lw=1.2,
        clip_on=False,
    )
    ax.text(
        mid,
        y + h + 0.012,
        text,
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        color=color,
        clip_on=False,
    )


def parse_force_num(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x).strip().lower().replace("g", "")
    return float(s)


def snap_pair_to_band(lo, hi, band_defs, tol):
    """Return (label, code) for closest canonical pair within tol (L1), else (None, None)."""
    lo, hi = float(min(lo, hi)), float(max(lo, hi))
    best_d, best = np.inf, None
    for row in band_defs:
        d = abs(lo - row["lo"]) + abs(hi - row["hi"])
        if d < best_d:
            best_d, best = d, row
    if best is not None and best_d <= tol:
        return best["label"], best["code"]
    return None, None


def compute_pairwise_gee_p_matrix(df_band, sub_col, pair_defs):
    """
    For each reference level, fit Binomial GEE (ForcePair + Region, cluster=subject).
    Fill p_matrix[ref_label, tgt_label] = p-value for tgt vs ref (diagonal 1).
    Same hypothesis as changing reference in Treatment coding (Wald z).
    """
    labels = [p["label"] for p in pair_defs]
    p_matrix = pd.DataFrame(np.nan, index=labels, columns=labels)
    first_result = None
    fam = sm.families.Binomial()
    ind = sm.cov_struct.Exchangeable()

    for ri, ref in enumerate(pair_defs):
        refc, refl = ref["code"], ref["label"]
        formula = (
            f"IsCorrect ~ C(ForcePairCode, Treatment(reference='{refc}')) + C(Region)"
        )
        try:
            result = smf.gee(
                formula,
                data=df_band,
                groups=df_band[sub_col],
                family=fam,
                cov_struct=ind,
            ).fit()
        except Exception:
            continue
        if ri == 0:
            first_result = result

        for tgt in pair_defs:
            if tgt["code"] == refc:
                p_matrix.loc[refl, tgt["label"]] = 1.0
                continue
            col = (
                f"C(ForcePairCode, Treatment(reference='{refc}'))[T.{tgt['code']}]"
            )
            if col in result.pvalues.index:
                p_matrix.loc[refl, tgt["label"]] = float(result.pvalues[col])

    return p_matrix, first_result


def pairwise_min_p(p_matrix, lab_i, lab_j):
    """Conservative p for undirected pair i–j: min of the two directed Wald p-values."""
    a = p_matrix.loc[lab_i, lab_j]
    b = p_matrix.loc[lab_j, lab_i]
    vals = [x for x in (a, b) if not np.isnan(x)]
    if not vals:
        return np.nan
    return float(min(vals))


def run_gee_and_plot_band(
    ax,
    df_band,
    sub_col,
    band_title,
    pair_labels_order,
    pair_defs,
    p_pairwise,
    gee_result_for_summary,
):
    """Boxplot + strip; brackets for any pair with pairwise min p < 0.05."""
    if df_band.empty or df_band["ForcePairCode"].nunique() < 2:
        ax.text(
            0.5,
            0.5,
            f"{band_title}: insufficient data",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title(band_title)
        return

    if gee_result_for_summary is not None:
        print(f"\n--- {band_title} | Binomial GEE (reference = {pair_defs[0]['label']}) ---")
        print(gee_result_for_summary.summary())

    print(f"\n--- {band_title} | Pairwise GEE p-values (rows = reference) ---")
    print(p_pairwise.round(4).to_string())

    subj_acc = (
        df_band.groupby([sub_col, "ForcePairLabel"], as_index=False)["IsCorrect"].mean()
    )

    sns.boxplot(
        x="ForcePairLabel",
        y="IsCorrect",
        hue="ForcePairLabel",
        data=subj_acc,
        order=pair_labels_order,
        hue_order=pair_labels_order,
        palette="Set2",
        width=0.65,
        dodge=False,
        legend=False,
        showmeans=True,
        meanprops={
            "marker": "D",
            "markerfacecolor": "red",
            "markeredgecolor": "red",
        },
        ax=ax,
    )
    sns.stripplot(
        x="ForcePairLabel",
        y="IsCorrect",
        data=subj_acc,
        order=pair_labels_order,
        color="black",
        alpha=0.35,
        jitter=True,
        size=6,
        ax=ax,
    )

    y_bracket = 1.06
    step = 0.055
    n_brackets = 0
    for i, li in enumerate(pair_labels_order):
        for j, lj in enumerate(pair_labels_order):
            if i >= j:
                continue
            p = pairwise_min_p(p_pairwise, li, lj)
            if np.isnan(p) or p >= 0.05:
                continue
            x1, x2 = sorted([i, j])
            y = y_bracket + n_brackets * step
            draw_sig_bracket(ax, x1, x2, y, pvalue_to_stars(p))
            ax.text(
                (x1 + x2) / 2,
                y + 0.045,
                f"p={p:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="dimgray",
                clip_on=False,
            )
            n_brackets += 1

    ax.set_title(band_title, fontsize=13)
    ax.set_ylabel("Accuracy (mean per subject)", fontsize=11)
    ax.set_xlabel("Force pair (canonical)", fontsize=11)
    ax.tick_params(axis="x", rotation=25)
    ax.set_ylim(-0.05, min(1.38, 1.08 + max(0, n_brackets) * step + 0.14))
    ax.axhline(
        y=CHANCE_LEVEL,
        color="gray",
        linestyle="--",
        label=f"Chance ({CHANCE_LEVEL})",
    )
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)


# Canonical pairs: 비교는 각 밴드 안에서만
HIGH_UNIQUE_DEFS = [
    {"lo": 10.0, "hi": 26.0, "label": "10–26", "code": "FPH0"},
    {"lo": 15.0, "hi": 26.0, "label": "15–26", "code": "FPH1"},
    {"lo": 25.0, "hi": 60.0, "label": "25–60", "code": "FPH2"},
]

LOW_BAND_DEFS = [
    {"lo": 0.4, "hi": 1.0, "label": "0.4–1", "code": "FPL0"},
    {"lo": 0.6, "hi": 1.0, "label": "0.6–1", "code": "FPL1"},
    {"lo": 1.0, "hi": 1.4, "label": "1–1.4", "code": "FPL2"},
    {"lo": 1.0, "hi": 2.0, "label": "1–2", "code": "FPL3"},
]
HIGH_BAND_DEFS = [
    HIGH_UNIQUE_DEFS[0],
    HIGH_UNIQUE_DEFS[1],
    {"lo": 25.0, "hi": 60.0, "label": "25–60", "code": "FPH2"},
    {"lo": 26.0, "hi": 60.0, "label": "25–60", "code": "FPH2"},
]
LOW_TOL = 0.35
HIGH_TOL = 2.5

file_pattern = "/Users/kyungeunjung/NailFoldExp/Data/(FD)CurData/P*_ForceDiscrimination.csv"
all_files = glob.glob(file_pattern)

if not all_files:
    print("CSV 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
else:
    df_list = [pd.read_csv(f) for f in all_files]
    df_merged = pd.concat(df_list, ignore_index=True)
    sub_col = "SubjectID" if "SubjectID" in df_merged.columns else "Subject"
    print(f"총 {len(all_files)}명의 피험자 데이터를 로드했습니다.")

    def calc_accuracy(row):
        if row["UserChoice"] == 1:
            return 1 if row["FirstStim"] > row["SecondStim"] else 0
        if row["UserChoice"] == 2:
            return 1 if row["SecondStim"] > row["FirstStim"] else 0
        return 0

    df_merged["IsCorrect"] = df_merged.apply(calc_accuracy, axis=1)

    ref_n = df_merged["Reference"].map(parse_force_num)
    cmp_n = df_merged["Comparison"].map(parse_force_num)
    df_merged["_f_lo"] = np.minimum(ref_n, cmp_n)
    df_merged["_f_hi"] = np.maximum(ref_n, cmp_n)
    df_merged = df_merged.dropna(subset=["_f_lo", "_f_hi"]).copy()

    labels, codes = [], []
    for lo, hi in zip(df_merged["_f_lo"], df_merged["_f_hi"]):
        mx = max(lo, hi)
        mn = min(lo, hi)
        lab, cod = None, None
        if mx <= 4.5:
            lab, cod = snap_pair_to_band(lo, hi, LOW_BAND_DEFS, LOW_TOL)
        elif mn >= 8.0:
            lab, cod = snap_pair_to_band(lo, hi, HIGH_BAND_DEFS, HIGH_TOL)
        else:
            l1, c1 = snap_pair_to_band(lo, hi, LOW_BAND_DEFS, LOW_TOL)
            l2, c2 = snap_pair_to_band(lo, hi, HIGH_BAND_DEFS, HIGH_TOL)
            d1 = min(
                abs(lo - p["lo"]) + abs(hi - p["hi"]) for p in LOW_BAND_DEFS
            )
            d2 = min(
                abs(lo - p["lo"]) + abs(hi - p["hi"]) for p in HIGH_BAND_DEFS
            )
            if l1 is not None and (l2 is None or d1 <= d2):
                lab, cod = l1, c1
            elif l2 is not None:
                lab, cod = l2, c2
        labels.append(lab)
        codes.append(cod)

    df_merged["ForcePairLabel"] = labels
    df_merged["ForcePairCode"] = codes
    unmatched = df_merged["ForcePairLabel"].isna().sum()
    if unmatched:
        print(f"주의: canonical에 맞지 않아 제외한 trial 수 = {unmatched}")
    df_use = df_merged.dropna(subset=["ForcePairLabel", "ForcePairCode"]).copy()

    low_order = [p["label"] for p in LOW_BAND_DEFS[:4]]
    high_order = [p["label"] for p in HIGH_UNIQUE_DEFS]

    df_low = df_use[df_use["ForcePairLabel"].isin(low_order)].copy()
    df_high = df_use[df_use["ForcePairLabel"].isin(high_order)].copy()

    print(f"저역 밴드 trial 수: {len(df_low)}, 고역 밴드 trial 수: {len(df_high)}")

    p_low, gee_low = compute_pairwise_gee_p_matrix(df_low, sub_col, LOW_BAND_DEFS[:4])
    p_high, gee_high = compute_pairwise_gee_p_matrix(
        df_high, sub_col, HIGH_UNIQUE_DEFS
    )

    sns.set_theme(style="whitegrid")
    sns.set_context("paper", font_scale=1.12)

    fig, axes = plt.subplots(2, 1, figsize=FIGSIZE_BOX, sharey=False)

    run_gee_and_plot_band(
        axes[0],
        df_low,
        sub_col,
        "Low band: pairwise GEE (all force-pair contrasts)",
        low_order,
        LOW_BAND_DEFS[:4],
        p_low,
        gee_low,
    )
    run_gee_and_plot_band(
        axes[1],
        df_high,
        sub_col,
        "High band: pairwise GEE (all force-pair contrasts)",
        high_order,
        HIGH_UNIQUE_DEFS,
        p_high,
        gee_high,
    )

    fig.tight_layout()

    # Pairwise p heatmaps (same models as matrix; easy scan of e.g. 0.6–1 vs 1–1.4)
    fig_hm, axes_hm = plt.subplots(1, 2, figsize=FIGSIZE_HEATMAP)
    sns.heatmap(
        p_low,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu_r",
        vmin=0,
        vmax=0.1,
        ax=axes_hm[0],
        cbar_kws={"label": "p"},
    )
    axes_hm[0].set_title("Low band: pairwise GEE p (row = ref)")
    axes_hm[0].set_xlabel("Compared pair")
    axes_hm[0].set_ylabel("Reference pair")

    sns.heatmap(
        p_high,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu_r",
        vmin=0,
        vmax=0.1,
        ax=axes_hm[1],
        cbar_kws={"label": "p"},
    )
    axes_hm[1].set_title("High band: pairwise GEE p (row = ref)")
    axes_hm[1].set_xlabel("Compared pair")
    axes_hm[1].set_ylabel("Reference pair")

    fig_hm.tight_layout()

    os.makedirs(FIG_EXPORT_DIR, exist_ok=True)
    path_box = os.path.join(FIG_EXPORT_DIR, "FD_force_pair_boxplots.png")
    path_hm = os.path.join(FIG_EXPORT_DIR, "FD_force_pair_pairwise_p_heatmap.png")
    export_figure(fig, path_box)
    export_figure(fig_hm, path_hm)
    print(f"\nSaved figures ({EXPORT_DPI} dpi, tight bbox):")
    print(f"  {path_box}")
    print(f"  {path_hm}")

    if os.environ.get("FD_STATS_SHOW", "").strip().lower() in ("1", "true", "yes"):
        plt.show()

    plt.close(fig)
    plt.close(fig_hm)
