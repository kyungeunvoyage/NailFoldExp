import os
import importlib.util
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.transforms import blended_transform_factory
from matplotlib.ticker import FixedLocator
import seaborn as sns
import glob
import statsmodels.formula.api as smf
import warnings

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(SCRIPT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# Shared styling with ATD_C1_Fig(Anika).py
_ANIKA_PATH = os.path.join(SCRIPT_DIR, "ATD_C1_Fig(Anika).py")
_spec = importlib.util.spec_from_file_location("atd_c1", _ANIKA_PATH)
atd_c1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(atd_c1)

IN_AIR = atd_c1.IN_AIR
ON_TOUCH = atd_c1.ON_TOUCH
SLATE_BLUE = atd_c1.SLATE_BLUE
KAO_COLOR = atd_c1.KAO_COLOR
ACCENT_RED = atd_c1.ACCENT_RED
CRITERION_COLOR = atd_c1.CRITERION_COLOR
BLACK = atd_c1.BLACK
STRIP_ALPHA = atd_c1.STRIP_ALPHA
SCATTER_HSB_BRIGHTNESS = atd_c1.SCATTER_HSB_BRIGHTNESS
BOX_LINEWIDTH = atd_c1.BOX_LINEWIDTH
CAP_LINEWIDTH = atd_c1.CAP_LINEWIDTH
CAP_WIDTH = atd_c1.CAP_WIDTH
FONT_TICK = atd_c1.FONT_TICK
FONT_LABEL = atd_c1.FONT_LABEL
FONT_LEGEND = atd_c1.FONT_LEGEND
FONT_ANNOT = atd_c1.FONT_ANNOT
SAVE_DPI = atd_c1.SAVE_DPI
pale_box_face = atd_c1.pale_box_face
pale_vis_color = atd_c1.pale_vis_color
_hsb_scatter_rgba = atd_c1._hsb_scatter_rgba
finalize_boxplot_lines = atd_c1.finalize_boxplot_lines

ATD_CMAP = LinearSegmentedColormap.from_list(
    "atd", ["#FFFFFF", pale_vis_color(IN_AIR), ON_TOUCH, ACCENT_RED]
)
_AREA_BASE = [IN_AIR, ON_TOUCH, KAO_COLOR, SLATE_BLUE, atd_c1.IN_AIR_LEGACY, atd_c1.ON_TOUCH_LEGACY]

ENABLE_FIG1 = False
#여기서 빼고 싶은거..
EXCLUDE_FORCES = {0.07, 1.4}   # omit from plots (1.0 g included)
FIG2_SIZE   = (14.0, 6.0)   # pairwise LME heatmaps (width scales with # forces)
FIG3_SIZE   = (14.5, 5.2)   # lateral / proximal contrasts
FIG5_SIZE = (8.0, 4.5)       # onnail_vs_offnail_by_force (2-col aspect)
EXPORT_WIDTH_2COL = 2102
Y_LABEL = "Detection Accuracy (%)"
FONT_XTICK = 11              # smaller x-axis tick labels on contrast plots
FONT_FIG5_XTICK = 12         # Fig5 region labels (On-nail\n(C+D), …)
FIG5_Y_TICKS = [0, 25, 50, 75, 100]   # tick labels; axis line ends here
FIG5_Y_AXIS_TOP = 100                 # visible y-axis spine stops at 100
FIG5_YLIM_TOP_CAP = 120               # plot headroom for significance brackets
TICK_LEN_AXES = atd_c1.TICK_LEN_AXES
add_legend_outside = atd_c1.add_legend_outside


def add_fig5_legend(fig, handles, ncol=3):
    """Legend above the panels (same anchor pattern as ATD_C1_Fig add_legend_outside)."""
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.99),
        bbox_transform=fig.transFigure,
        ncol=ncol,
        frameon=False,
        fontsize=FONT_LABEL,
        title_fontsize=FONT_LABEL,
        labelspacing=0.2,
        columnspacing=2.0,
        handlelength=1.6,
        handleheight=1.0,
        borderaxespad=atd_c1.FIG_LEGEND_PAD_PT,
    )


def add_inward_tick_guides(ax, x_positions=None, y_ticks=None):
    """Short inward tick marks at each x/y label (same style as ATD_C1_Fig)."""
    ax.tick_params(axis="both", which="both", length=0)
    x_trans = blended_transform_factory(ax.transData, ax.transAxes)
    y_trans = blended_transform_factory(ax.transAxes, ax.transData)
    y_lo, y_hi = ax.get_ylim()
    if y_ticks is None:
        y_vals = [t for t in ax.get_yticks() if y_lo - 1e-9 <= t <= y_hi + 1e-9]
    else:
        y_vals = [t for t in y_ticks if y_lo - 1e-9 <= t <= y_hi + 1e-9]
    if x_positions is None:
        x_positions = ax.get_xticks()
    for xi in x_positions:
        ax.plot(
            [xi, xi], [0, TICK_LEN_AXES],
            color=BLACK, linewidth=1.0, solid_capstyle="butt",
            transform=x_trans, clip_on=False, zorder=6,
        )
    for y in y_vals:
        ax.plot(
            [0, TICK_LEN_AXES], [y, y],
            color=BLACK, linewidth=1.0, solid_capstyle="butt",
            transform=y_trans, clip_on=False, zorder=6,
        )


def save_png_at_width(fig, out_path, width_px=EXPORT_WIDTH_2COL, *, pad_inches=0.04):
    w_in, _ = fig.get_size_inches()
    dpi = width_px / w_in
    fig.savefig(
        out_path, dpi=dpi, bbox_inches="tight",
        pad_inches=pad_inches, facecolor="white",
    )


def _force_panel_title(force_val):
    """Compact force label for facet titles (e.g. 0.16 g, 0.6 g, 1 g)."""
    text = f"{force_val:g}"
    if "." not in text:
        text = f"{text}.0"
    return f"{text} g"


def _set_force_title_above(ax, force_val, y=1.12):
    """Place force title above the axes so brackets do not overlap it."""
    ax.set_title("")
    ax.text(
        0.5, y, _force_panel_title(force_val),
        transform=ax.transAxes,
        ha="center", va="bottom",
        fontsize=FONT_LABEL, fontweight="black",
        clip_on=False,
    )


def _apply_c1_theme():
    sns.set_theme(style="white")
    atd_c1.apply_plot_style()


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


BRACKET_BASE_OFFSET = 2.5
BRACKET_TIER_STEP = 8.0
BRACKET_TEXT_PAD = 5.0


def _assign_bracket_tiers(contrast_specs, region_order):
    """Assign vertical tiers so overlapping x-spans do not share a tier."""
    spans = []
    for a1, a2, key in contrast_specs:
        if a1 not in region_order or a2 not in region_order:
            continue
        x_l, x_r = sorted([region_order.index(a1), region_order.index(a2)])
        spans.append({"x_l": x_l, "x_r": x_r, "w": x_r - x_l, "key": key})
    spans.sort(key=lambda s: (s["w"], s["x_l"]))
    for i, s_i in enumerate(spans):
        tier = 0
        while True:
            conflict = any(
                spans[j]["tier"] == tier
                and s_i["x_l"] <= spans[j]["x_r"]
                and spans[j]["x_l"] <= s_i["x_r"]
                for j in range(i)
            )
            if not conflict:
                s_i["tier"] = tier
                break
            tier += 1
    return spans


def _add_sig_bracket(ax, x_l, x_r, y_base, tick_h=0.45, text=""):
    """Bracket from each box center (x_l, x_r) up to a shared bar."""
    x_center = (x_l + x_r) / 2.0
    y_top = y_base + tick_h
    ax.plot(
        [x_l, x_l, x_r, x_r],
        [y_base, y_top, y_top, y_base],
        color=ACCENT_RED,
        linewidth=1.5,
        clip_on=False,
        zorder=5,
    )
    ax.text(
        x_center,
        y_top + 0.6,
        text,
        ha="center",
        va="bottom",
        fontsize=max(8, FONT_ANNOT - 1),
        color=ACCENT_RED,
        fontweight="bold",
        clip_on=False,
        zorder=6,
    )
    return y_top + 0.6 + BRACKET_TEXT_PAD


def _style_boxplot(bp, base_color):
    face = pale_box_face(base_color)
    for patch in bp["boxes"]:
        patch.set_facecolor(face)
        patch.set_edgecolor(BLACK)
        patch.set_linewidth(BOX_LINEWIDTH)
    for w in bp["whiskers"]:
        w.set_color(BLACK)
        w.set_linewidth(BOX_LINEWIDTH)
    for c in bp["caps"]:
        c.set_color(BLACK)
        c.set_linewidth(CAP_LINEWIDTH)
    for m in bp["medians"]:
        m.set_color(ACCENT_RED)
        m.set_linewidth(2.0)


def subject_mean_accuracy_regions(df_in, sub_col, regions):
    """Per-subject mean accuracy per region (trials pooled across forces)."""
    sub = df_in[df_in["Area"].isin(regions)].dropna(
        subset=[sub_col, "Area", "Relative_Score"]
    )
    if sub.empty:
        return pd.DataFrame(columns=[sub_col, "Area", "accuracy"])
    return (
        sub.groupby([sub_col, "Area"], as_index=False)["Relative_Score"]
        .mean()
        .rename(columns={"Relative_Score": "accuracy"})
    )


def subject_mean_accuracy_regions_by_force(df_in, sub_col, regions):
    """Per-subject mean accuracy per region and force."""
    sub = df_in[df_in["Area"].isin(regions)].dropna(
        subset=[sub_col, "Area", "Relative_Score", "Force_Val"]
    )
    if sub.empty:
        return pd.DataFrame(columns=[sub_col, "Area", "Force_Val", "accuracy"])
    return (
        sub.groupby([sub_col, "Area", "Force_Val"], as_index=False)["Relative_Score"]
        .mean()
        .rename(columns={"Relative_Score": "accuracy"})
    )


def subject_area_pool_as_separate(df_in, sub_col, area_group_map, force_val=None):
    """Per-subject mean accuracy per original area, then relabelled to group.

    Unlike subject_mean_accuracy_regions (which averages C+D together per subject,
    giving n=30), this keeps each area as a separate subject-level observation so
    pooling C and D gives n=60 (30 from C + 30 from D).

    area_group_map: dict mapping original Area label to group name.
                   e.g. {"C": "On-nail", "D": "On-nail", "A": "Off-nail", "F": "Off-nail"}
    """
    areas = list(area_group_map.keys())
    sub = df_in[df_in["Area"].isin(areas)].dropna(
        subset=[sub_col, "Area", "Relative_Score"]
    )
    if force_val is not None:
        sub = sub[np.isclose(sub["Force_Val"], force_val)]
    if sub.empty:
        return pd.DataFrame(columns=[sub_col, "Area", "Group", "accuracy"])
    group_keys = ([sub_col, "Area", "Force_Val"] if force_val is None
                  else [sub_col, "Area"])
    agg = (
        sub.groupby(group_keys, as_index=False)["Relative_Score"]
        .mean()
        .rename(columns={"Relative_Score": "accuracy"})
    )
    agg["Group"] = agg["Area"].map(area_group_map)
    return agg


def lme_two_groups_pooled(df_pooled, sub_col, ref_group, target_group,
                           score_col="accuracy"):
    """LME comparing two pooled groups where each subject may contribute
    multiple observations (e.g. both C and D rows → On-nail).

    Model: score ~ C(Group, Treatment(ref)), random intercept = Subject.
    Returns dict with coef, ci_lo, ci_hi, p, or None on failure.
    """
    sub = df_pooled[df_pooled["Group"].isin([ref_group, target_group])].dropna(
        subset=[sub_col, "Group", score_col]
    )
    if sub[sub_col].nunique() < 2 or sub["Group"].nunique() < 2:
        return None
    formula = (
        f"{score_col} ~ C(Group, Treatment(reference='{ref_group}'))"
    )
    try:
        res = smf.mixedlm(formula, sub, groups=sub[sub_col]).fit(reml=True)
        col = f"C(Group, Treatment(reference='{ref_group}'))[T.{target_group}]"
        if col not in res.params.index:
            return None
        ci = res.conf_int().loc[col]
        return {
            "coef": float(res.params[col]),
            "ci_lo": float(ci[0]),
            "ci_hi": float(ci[1]),
            "p": float(res.pvalues[col]),
        }
    except Exception:
        return None


def subject_mean_of_area_means(df_in, sub_col, area_group_map, force_val=None):
    """Per-subject mean-of-area-means: first compute per-area mean, then average
    across areas within each group.

    e.g. area_group_map = {"C": "On-nail", "D": "On-nail", "A": "Off-nail", "F": "Off-nail"}
    → On-nail per subject  = (C_mean + D_mean) / 2   (n=30)
    → Off-nail per subject = (A_mean + F_mean) / 2   (n=30)
    """
    areas = list(area_group_map.keys())
    sub = df_in[df_in["Area"].isin(areas)].dropna(
        subset=[sub_col, "Area", "Relative_Score"]
    )
    if force_val is not None:
        sub = sub[np.isclose(sub["Force_Val"], force_val)]
    if sub.empty:
        return pd.DataFrame(columns=[sub_col, "Group", "accuracy"])

    group_keys = [sub_col, "Area"] + (["Force_Val"] if force_val is None else [])
    area_means = (
        sub.groupby(group_keys, as_index=False)["Relative_Score"]
        .mean()
        .rename(columns={"Relative_Score": "area_acc"})
    )
    area_means["Group"] = area_means["Area"].map(area_group_map)

    grp_keys = [sub_col, "Group"] + (["Force_Val"] if force_val is None else [])
    result = (
        area_means.groupby(grp_keys, as_index=False)["area_acc"]
        .mean()
        .rename(columns={"area_acc": "accuracy"})
    )
    return result


def plot_region_boxes(
    ax,
    plot_df,
    region_order,
    palette,
    *,
    bar_w=0.55,
    edge_pad=0.35,
    x_tick_labels=None,
):
    """One box per region on the x-axis. Returns {region: whisker_top}."""
    rng = np.random.default_rng(0)
    box_kw = dict(
        widths=bar_w,
        patch_artist=True,
        showfliers=False,
        zorder=2,
    )
    tops = {}
    all_edges = []

    for xi, region in enumerate(region_order):
        vals = plot_df.loc[plot_df["Area"] == region, "accuracy"].dropna().values
        if len(vals) == 0:
            continue
        bp = ax.boxplot([vals], positions=[xi], **box_kw)
        _style_boxplot(bp, palette[region])
        bx0, bx1 = xi - bar_w / 2, xi + bar_w / 2
        all_edges.extend([bx0, bx1])
        tops[region] = max(bp["whiskers"][1].get_ydata())

        jitter = rng.uniform(-bar_w * 0.12, bar_w * 0.12, size=len(vals))
        ax.scatter(
            xi + jitter,
            vals,
            color=_hsb_scatter_rgba(palette[region]),
            s=14,
            linewidths=0,
            zorder=3,
        )

    labels = x_tick_labels if x_tick_labels is not None else region_order
    ax.set_xticks(range(len(region_order)))
    ax.set_xticklabels(labels)
    if all_edges:
        ax.set_xlim(min(all_edges) - edge_pad, max(all_edges) + edge_pad)
    return tops


def add_region_contrast_brackets(ax, region_order, region_tops, contrast_specs, lme_by_key):
    """
    Significance brackets between region pairs.
    contrast_specs: [(area_left, area_right, lme_key), ...]
    """
    base_top = max(region_tops.values()) if region_tops else 80.0
    y_ceiling = 80.0
    for span in _assign_bracket_tiers(contrast_specs, region_order):
        x_l, x_r = span["x_l"], span["x_r"]
        tier = span["tier"]
        y_bracket = base_top + BRACKET_BASE_OFFSET + tier * BRACKET_TIER_STEP
        r = lme_by_key.get(span["key"])
        if r is None:
            sig_text = "LME fail"
        else:
            sig_text = f"{_star_from_p(r['p'])}  p={r['p']:.3f}"
        text_top = _add_sig_bracket(ax, x_l, x_r, y_bracket, text=sig_text)
        y_ceiling = max(y_ceiling, text_top)
    return y_ceiling


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
                color=_hsb_scatter_rgba(palette[area]),
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
    area: _AREA_BASE[i % len(_AREA_BASE)]
    for i, area in enumerate(["A", "B", "C", "D", "E", "F"])
}


def plot_region_accuracy_panel(ax, plot_df, region_order, palette, contrast_specs,
                               lme_by_label, title, *, x_tick_labels=None):
    """One box per region; brackets show pairwise LME contrasts."""
    if plot_df.empty or not region_order:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title, fontsize=FONT_LABEL)
        return 80.0

    region_tops = plot_region_boxes(
        ax, plot_df, region_order, palette,
        x_tick_labels=x_tick_labels,
    )
    ax.axhline(80, color=CRITERION_COLOR, linestyle="--", linewidth=1.0, alpha=0.85,
               zorder=atd_c1.REF_LINE_ZORDER)
    ax.set_ylabel(Y_LABEL, fontsize=FONT_LABEL)
    ax.set_xlabel("Region", fontsize=FONT_LABEL)
    ax.set_title(title, fontsize=FONT_LABEL, fontweight="bold")
    ax.tick_params(axis="x", labelsize=FONT_XTICK)
    ax.tick_params(axis="y", labelsize=FONT_TICK)
    y_ceiling = add_region_contrast_brackets(
        ax, region_order, region_tops, contrast_specs, lme_by_label,
    )
    ax.set_ylim(-5, min(145, y_ceiling + 4))
    return y_ceiling


def plot_accuracy_contrast_panel(ax, plot_df, lme_by_label, title):
    """Deprecated wrapper — use plot_region_accuracy_panel."""
    region_order = sorted(plot_df["Area"].unique())
    contrast_specs = [
        tuple(k.split(" vs ", 1)) + (k,)
        for k in lme_by_label.keys()
        if " vs " in k and all(p in region_order for p in k.split(" vs ", 1))
    ]
    pal = {k: AREA_PALETTE.get(k, SLATE_BLUE) for k in region_order}
    plot_region_accuracy_panel(
        ax, plot_df, region_order, pal, contrast_specs, lme_by_label, title,
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

    # P61, P62, P63: partial-protocol — only 0.4 g data should be included
    _PARTIAL_SUBJ = {"P61", "P62", "P63"}
    _is_partial = df[sub_col].isin(_PARTIAL_SUBJ)
    df = df[~_is_partial | (df["Force_Val"] == 0.4)].copy()
    print(f"After partial-subject filter: {len(df)} rows")

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

    _apply_c1_theme()
    palette = {"M": ON_TOUCH, "F": IN_AIR}

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
                    color=palette["M"], fontweight="bold", ha="center", fontsize=FONT_ANNOT,
                )
                ax_gender.text(
                    i + 0.2, f_med + 1, f"{f_med:.1f}",
                    color=palette["F"], fontweight="bold", ha="center", fontsize=FONT_ANNOT,
                )
                if star != "n.s.":
                    ax_gender.text(
                        i, 115, star, ha="center", va="bottom",
                        color=ACCENT_RED, fontsize=FONT_LABEL, fontweight="bold",
                    )
                    ax_gender.text(
                        i, 110, f"p={p_val:.3f}", ha="center", va="top",
                        color=BLACK, fontsize=FONT_ANNOT,
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
        1, len(plot_forces),
        figsize=(FIG2_SIZE[0] * len(plot_forces) / 2, FIG2_SIZE[1]),
        facecolor="white",
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
        axes[i].set_title(f"Force {f_val}g: Pairwise LME p-values", fontsize=FONT_LABEL)
        axes[i].set_xlabel("Compared Area", fontsize=FONT_LABEL)
        axes[i].set_ylabel("Reference Area", fontsize=FONT_LABEL)
        axes[i].tick_params(labelsize=FONT_TICK)

    fig2.tight_layout()
    out_hm = os.path.join(FIG_DIR, "pairwise_lme_heatmap.png")
    fig2.savefig(out_hm, dpi=SAVE_DPI, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out_hm}  ({FIG2_SIZE[0]}×{FIG2_SIZE[1]} in @ {SAVE_DPI} dpi)")
    plt.close(fig2)

    # --- Figure 3: Lateral / proximal contrasts ---
    LEFT_AREAS = ["A", "B", "C", "D"]
    LEFT_PAIRS = [("A", "C"), ("A", "D"), ("B", "C"), ("B", "D")]
    RIGHT_AREAS = ["E", "F", "C", "D"]
    RIGHT_PAIRS = [("E", "C"), ("E", "D"), ("F", "C"), ("F", "D")]

    plot_left = subject_mean_accuracy_regions(df_analysis, sub_col, LEFT_AREAS)
    plot_right = subject_mean_accuracy_regions(df_analysis, sub_col, RIGHT_AREAS)
    lme_left = {
        f"{a1} vs {a2}": lme_area_pair_contrast(df_analysis, sub_col, a2, a1)
        for a1, a2 in LEFT_PAIRS
    }
    lme_right = {
        f"{a1} vs {a2}": lme_area_pair_contrast(df_analysis, sub_col, a2, a1)
        for a1, a2 in RIGHT_PAIRS
    }
    left_specs = [(a1, a2, f"{a1} vs {a2}") for a1, a2 in LEFT_PAIRS]
    right_specs = [(a1, a2, f"{a1} vs {a2}") for a1, a2 in RIGHT_PAIRS]
    left_pal = {a: AREA_PALETTE[a] for a in LEFT_AREAS}
    right_pal = {a: AREA_PALETTE[a] for a in RIGHT_AREAS}

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
        fontsize=FONT_LABEL,
        y=1.02,
    )
    plot_region_accuracy_panel(
        ax_l,
        plot_left,
        LEFT_AREAS,
        left_pal,
        left_specs,
        lme_left,
        "A–C, A–D, B–C, B–D",
    )
    plot_region_accuracy_panel(
        ax_r,
        plot_right,
        RIGHT_AREAS,
        right_pal,
        right_specs,
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

    NAIL_PALETTE = {
        "On-Nail": "#10559A",
        "Off-Nail (A)": "#7C94B8",
        "Off-Nail (F)": "#B1BBC8",
    }
    REGION_ORDER = ["On-Nail", "Off-Nail (A)", "Off-Nail (F)"]
    NAIL_X_LABELS = ["On-nail\n(C+D)", "Off-nail\n(A)", "Off-nail\n(F)"]
    NAIL_CONTRAST_SPECS = [
        ("On-Nail", "Off-Nail (A)", "On-Nail vs Off-Nail (A)"),
        ("On-Nail", "Off-Nail (F)", "On-Nail vs Off-Nail (F)"),
        ("Off-Nail (A)", "Off-Nail (F)", "A vs F"),
    ]

    lme_nail = {}
    for a_nail, a_off in pairs_nail:
        label = f"{a_nail} vs {a_off}"
        lme_nail[label] = lme_area_pair_contrast_v2(df_onnail, sub_col, a_off, a_nail)
    lme_nail["A vs F"] = lme_area_pair_contrast_v2(
        df_onnail, sub_col, "Off-Nail (F)", "Off-Nail (A)"
    )

    plot_df_nail = subject_mean_accuracy_regions(df_onnail, sub_col, REGION_ORDER)

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

    fig4, ax4 = plt.subplots(figsize=(5.5, 5.2), facecolor="white")

    region_tops4 = plot_region_boxes(
        ax4, plot_df_nail, REGION_ORDER, NAIL_PALETTE, edge_pad=0.40,
    )
    ax4.axhline(80, color=CRITERION_COLOR, linestyle="--", linewidth=1.0, alpha=0.85,
                zorder=atd_c1.REF_LINE_ZORDER)
    ax4.set_ylabel(Y_LABEL, fontsize=FONT_LABEL)
    ax4.tick_params(axis="x", rotation=0, labelsize=FONT_XTICK)
    ax4.tick_params(axis="y", labelsize=FONT_TICK)
    y_ceiling = add_region_contrast_brackets(
        ax4, REGION_ORDER, region_tops4, NAIL_CONTRAST_SPECS, lme_nail,
    )
    ax4.set_ylim(-5, min(120, y_ceiling + 4))

    leg_handles = [
        mpatches.Patch(
            facecolor=pale_box_face(NAIL_PALETTE[r]), edgecolor=BLACK,
            linewidth=BOX_LINEWIDTH,
            label=NAIL_X_LABELS[i].replace("\n", " "),
        )
        for i, r in enumerate(REGION_ORDER)
    ]
    add_legend_outside(fig4, ax4, leg_handles, ncol=len(REGION_ORDER))

    fig4.subplots_adjust(left=0.10, right=0.97, top=0.94, bottom=0.18)
    out_nail = os.path.join(FIG_DIR, "onnail_vs_offnail_accuracy.png")
    fig4.savefig(out_nail, dpi=SAVE_DPI, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out_nail}")
    plt.close(fig4)

    # --- Figure 5: On-nail regions by force (0.16, 0.6, 1.0 g) ---
    plot_df_f5 = subject_mean_accuracy_regions_by_force(
        df_onnail, sub_col, REGION_ORDER,
    )

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

    for a1, a2, key in NAIL_CONTRAST_SPECS:
        sub_cv = df_onnail[df_onnail["Area"].isin([a1, a2])]
        r_cv = lme_force_test(sub_cv, sub_col, include_area=True)
        if r_cv:
            print(f"  {key} — Force|Area: p={r_cv['p']:.4f}")
        for fval in plot_forces:
            r_af = lme_area_pair_at_force(
                df_onnail, sub_col, a2, a1, fval
            )
            if r_af:
                print(
                    f"    @ {fval:.2f}g Area contrast: p={r_af['p']:.4f}"
                )

    n_force_panels = len(plot_forces)
    fig5, axes5 = plt.subplots(
        1,
        n_force_panels,
        figsize=FIG5_SIZE,
        sharey=False,
        facecolor="white",
    )
    if n_force_panels == 1:
        axes5 = [axes5]

    for ax5, fval in zip(axes5, plot_forces):
        sub_f = plot_df_f5[np.isclose(plot_df_f5["Force_Val"], fval)].copy()

        region_tops_f = plot_region_boxes(
            ax5, sub_f, REGION_ORDER, NAIL_PALETTE,
            edge_pad=0.35, bar_w=0.55, x_tick_labels=NAIL_X_LABELS,
        )
        ax5.axhline(80, color=CRITERION_COLOR, linestyle="--", linewidth=1.0, alpha=0.85,
                    zorder=atd_c1.REF_LINE_ZORDER)
        _set_force_title_above(ax5, fval, y=1.04)
        ax5.tick_params(axis="x", labelsize=FONT_FIG5_XTICK)
        ax5.tick_params(axis="y", labelsize=FONT_TICK)

        lme_at_force = {
            key: lme_area_pair_at_force(df_onnail, sub_col, a2, a1, fval)
            for a1, a2, key in NAIL_CONTRAST_SPECS
        }
        y_ceil_f = add_region_contrast_brackets(
            ax5, REGION_ORDER, region_tops_f, NAIL_CONTRAST_SPECS, lme_at_force,
        )
        y_top = min(FIG5_YLIM_TOP_CAP, y_ceil_f + 4)
        ax5.set_ylim(-5, y_top)
        ax5.set_yticks(FIG5_Y_TICKS)
        ax5.yaxis.set_major_locator(FixedLocator(FIG5_Y_TICKS))
        # Spine/ticks end at 100; ylim may extend to ~120 for brackets only
        ax5.spines["left"].set_bounds(-5, FIG5_Y_AXIS_TOP)
        add_inward_tick_guides(
            ax5,
            x_positions=range(len(REGION_ORDER)),
            y_ticks=FIG5_Y_TICKS,
        )
        if fval == plot_forces[0]:
            ax5.set_ylabel(Y_LABEL, fontsize=FONT_LABEL)
        else:
            ax5.set_ylabel("")

    leg_handles_f5 = [
        mpatches.Patch(
            facecolor=pale_box_face(NAIL_PALETTE[r]),
            edgecolor=BLACK,
            linewidth=BOX_LINEWIDTH,
            label=NAIL_X_LABELS[i].replace("\n", " "),
        )
        for i, r in enumerate(REGION_ORDER)
    ]
    add_fig5_legend(fig5, leg_handles_f5, ncol=len(REGION_ORDER))
    # Larger ``top`` → subplots move up → less white gap under the legend
    fig5.subplots_adjust(left=0.08, right=0.97, top=0.88, bottom=0.12, wspace=0.18)
    out_f5 = os.path.join(FIG_DIR, "onnail_vs_offnail_by_force.png")
    save_png_at_width(fig5, out_f5, width_px=EXPORT_WIDTH_2COL)
    print(f"Saved: {out_f5}")
    plt.close(fig5)

    # =========================================================================
    # Figure 6: On-nail (C+D, n=60) vs Off-nail (A+F, n=60) — pooled, by force
    # Each subject contributes one mean per original area (C, D, A, F),
    # giving 60 observations per group. Shown as 3 force panels.
    # =========================================================================
    POOL_GROUP_MAP = {
        "C": "On-nail",
        "D": "On-nail",
        "A": "Off-nail",
        "F": "Off-nail",
    }
    POOL_GROUP_ORDER = ["On-nail", "Off-nail"]
    POOL_PALETTE = {
        "On-nail":  ON_TOUCH,
        "Off-nail": "#7C94B8",   # same as Off-Nail (A) in NAIL_PALETTE
    }
    POOL_X_LABELS = ["On-nail\n(C+D)", "Off-nail\n(A+F)"]

    print("\n[Figure 6 — On-nail(C+D) vs Off-nail(A+F) | pooled area means, by force, RE=Subject]")

    fig6, axes6 = plt.subplots(1, len(plot_forces), figsize=FIG5_SIZE, facecolor="white")
    if len(plot_forces) == 1:
        axes6 = [axes6]

    rng6 = np.random.default_rng(42)

    for ax6, fval in zip(axes6, plot_forces):
        df_pool_f = subject_area_pool_as_separate(
            df_analysis, sub_col, POOL_GROUP_MAP, force_val=fval
        )

        lme_pool_f = lme_two_groups_pooled(
            df_pool_f, sub_col, ref_group="Off-nail", target_group="On-nail"
        )
        if lme_pool_f:
            star_f = ("***" if lme_pool_f["p"] < 0.001 else
                      "**"  if lme_pool_f["p"] < 0.01  else
                      "*"   if lme_pool_f["p"] < 0.05  else "n.s.")
            print(f"  {fval}g  On-nail vs Off-nail: Δ={lme_pool_f['coef']:.3f} "
                  f"[{lme_pool_f['ci_lo']:.3f}, {lme_pool_f['ci_hi']:.3f}], "
                  f"p={lme_pool_f['p']:.4f}  {star_f}")
        else:
            print(f"  {fval}g  LME failed")

        tops_f = {}
        for xi, grp in enumerate(POOL_GROUP_ORDER):
            grp_data = df_pool_f[df_pool_f["Group"] == grp]["accuracy"].dropna().values
            bp = ax6.boxplot(
                [grp_data],
                positions=[xi],
                widths=0.45,
                patch_artist=True,
                showfliers=False,
                zorder=2,
                whiskerprops=dict(color=BLACK, linewidth=BOX_LINEWIDTH),
                capprops=dict(color=BLACK, linewidth=CAP_LINEWIDTH),
                medianprops=dict(color=ACCENT_RED, linewidth=2.0),
                boxprops=dict(
                    facecolor=pale_box_face(POOL_PALETTE[grp]),
                    edgecolor=BLACK,
                    linewidth=BOX_LINEWIDTH,
                ),
            )
            whisker_ys = [w.get_ydata()[1] for w in bp["whiskers"]]
            tops_f[grp] = max(whisker_ys) if whisker_ys else 0.0

            jitter = rng6.uniform(-0.12, 0.12, size=len(grp_data))
            rgba = _hsb_scatter_rgba(POOL_PALETTE[grp])
            ax6.scatter(
                xi + jitter, grp_data,
                c=[rgba] * len(grp_data),
                s=3.5 ** 2,
                linewidths=0,
                zorder=3,
                clip_on=False,
            )

        if lme_pool_f:
            star6 = ("***" if lme_pool_f["p"] < 0.001 else
                     "**"  if lme_pool_f["p"] < 0.01  else
                     "*"   if lme_pool_f["p"] < 0.05  else "n.s.")
            y_brk = max(tops_f.values()) + 4
            _add_sig_bracket(ax6, 0, 1, y_brk, text=star6)

        ax6.axhline(80, color=CRITERION_COLOR, linestyle="--", linewidth=1.0,
                    alpha=0.85, zorder=atd_c1.REF_LINE_ZORDER)
        _set_force_title_above(ax6, fval, y=0.95)
        ax6.set_xticks([0, 1])
        ax6.set_xticklabels(POOL_X_LABELS, fontsize=FONT_FIG5_XTICK)
        ax6.set_yticks(FIG5_Y_TICKS)
        ax6.yaxis.set_major_locator(FixedLocator(FIG5_Y_TICKS))
        ax6.tick_params(axis="y", labelsize=FONT_TICK)
        ax6.tick_params(axis="x", length=0)
        ax6.set_ylim(-5, FIG5_YLIM_TOP_CAP)
        ax6.spines["left"].set_bounds(-5, FIG5_Y_AXIS_TOP)
        sns.despine(ax=ax6)
        add_inward_tick_guides(ax6, x_positions=[0, 1], y_ticks=FIG5_Y_TICKS)
        if fval == plot_forces[0]:
            ax6.set_ylabel(Y_LABEL, fontsize=FONT_LABEL)
        else:
            ax6.set_ylabel("")

    leg_handles_f6 = [
        mpatches.Patch(
            facecolor=pale_box_face(POOL_PALETTE[grp]),
            edgecolor=BLACK,
            linewidth=BOX_LINEWIDTH,
            label=POOL_X_LABELS[i].replace("\n", " "),
        )
        for i, grp in enumerate(POOL_GROUP_ORDER)
    ]
    add_fig5_legend(fig6, leg_handles_f6, ncol=len(POOL_GROUP_ORDER))
    fig6.subplots_adjust(left=0.08, right=0.97, top=1.0, bottom=0.12, wspace=0.18)
    out_f6 = os.path.join(FIG_DIR, "onnail_vs_offnail_pooled.png")
    save_png_at_width(fig6, out_f6, width_px=EXPORT_WIDTH_2COL)
    print(f"Saved: {out_f6}")
    plt.close(fig6)

    # =========================================================================
    # Figure 7: On-nail (C+D trials pooled) vs Off-nail (A+F trials pooled)
    # Same trial-level pooling as current On-nail approach, extended to A+F.
    # Per subject: mean of ALL C+D trials → On-nail (n=30)
    #              mean of ALL A+F trials → Off-nail (n=30)
    # =========================================================================
    # Relabel C,D→On-nail and A,F→Off-nail in the dataframe, then use
    # subject_mean_accuracy_regions_by_force for per-force subject means.
    df_pool2 = df_analysis.copy()
    df_pool2["Area"] = df_pool2["Area"].replace(
        {"C": "On-nail", "D": "On-nail", "A": "Off-nail", "F": "Off-nail"}
    )

    MOA_GROUP_ORDER = ["On-nail", "Off-nail"]
    MOA_PALETTE = {
        "On-nail":  ON_TOUCH,
        "Off-nail": "#7C94B8",
    }
    MOA_X_LABELS = ["On-nail\n(C+D)", "Off-nail\n(A+F)"]

    print("\n[Figure 7 — On-nail (C+D pooled) vs Off-nail (A+F pooled) | trial-level mean, n=30, by force, RE=Subject]")

    df_pool2_by_force = subject_mean_accuracy_regions_by_force(
        df_pool2, sub_col, MOA_GROUP_ORDER
    )

    fig7, axes7 = plt.subplots(1, len(plot_forces), figsize=FIG5_SIZE, facecolor="white")
    if len(plot_forces) == 1:
        axes7 = [axes7]

    rng7 = np.random.default_rng(7)

    for ax7, fval in zip(axes7, plot_forces):
        df_moa_f = df_pool2_by_force[
            np.isclose(df_pool2_by_force["Force_Val"], fval)
        ].rename(columns={"Area": "Group"})

        lme_moa_f = lme_two_groups_pooled(
            df_moa_f, sub_col, ref_group="Off-nail", target_group="On-nail"
        )
        if lme_moa_f:
            star_m = ("***" if lme_moa_f["p"] < 0.001 else
                      "**"  if lme_moa_f["p"] < 0.01  else
                      "*"   if lme_moa_f["p"] < 0.05  else "n.s.")
            print(f"  {fval}g  On-nail vs Off-nail: Δ={lme_moa_f['coef']:.3f} "
                  f"[{lme_moa_f['ci_lo']:.3f}, {lme_moa_f['ci_hi']:.3f}], "
                  f"p={lme_moa_f['p']:.4f}  {star_m}")
        else:
            print(f"  {fval}g  LME failed")

        tops_m = {}
        for xi, grp in enumerate(MOA_GROUP_ORDER):
            grp_data = df_moa_f[df_moa_f["Group"] == grp]["accuracy"].dropna().values
            bp = ax7.boxplot(
                [grp_data],
                positions=[xi],
                widths=0.45,
                patch_artist=True,
                showfliers=False,
                zorder=2,
                whiskerprops=dict(color=BLACK, linewidth=BOX_LINEWIDTH),
                capprops=dict(color=BLACK, linewidth=CAP_LINEWIDTH),
                medianprops=dict(color=ACCENT_RED, linewidth=2.0),
                boxprops=dict(
                    facecolor=pale_box_face(MOA_PALETTE[grp]),
                    edgecolor=BLACK,
                    linewidth=BOX_LINEWIDTH,
                ),
            )
            whisker_ys = [w.get_ydata()[1] for w in bp["whiskers"]]
            tops_m[grp] = max(whisker_ys) if whisker_ys else 0.0

            jitter = rng7.uniform(-0.12, 0.12, size=len(grp_data))
            rgba = _hsb_scatter_rgba(MOA_PALETTE[grp])
            ax7.scatter(
                xi + jitter, grp_data,
                c=[rgba] * len(grp_data),
                s=3.5 ** 2,
                linewidths=0,
                zorder=3,
                clip_on=False,
            )

        if lme_moa_f:
            star7 = ("***" if lme_moa_f["p"] < 0.001 else
                     "**"  if lme_moa_f["p"] < 0.01  else
                     "*"   if lme_moa_f["p"] < 0.05  else "n.s.")
            p_txt = f"p={lme_moa_f['p']:.3f}"
            y_brk = max(tops_m.values()) + 4
            _add_sig_bracket(ax7, 0, 1, y_brk, text=f"{star7}  {p_txt}")

        ax7.axhline(80, color=CRITERION_COLOR, linestyle="--", linewidth=1.0,
                    alpha=0.85, zorder=atd_c1.REF_LINE_ZORDER)
        _set_force_title_above(ax7, fval, y=1.04)
        ax7.set_xticks([0, 1])
        ax7.set_xticklabels(MOA_X_LABELS, fontsize=FONT_FIG5_XTICK)
        ax7.set_yticks(FIG5_Y_TICKS)
        ax7.yaxis.set_major_locator(FixedLocator(FIG5_Y_TICKS))
        ax7.tick_params(axis="y", labelsize=FONT_TICK)
        ax7.tick_params(axis="x", length=0)
        y_top7 = min(FIG5_YLIM_TOP_CAP, max(tops_m.values()) + 20)
        ax7.set_ylim(-5, y_top7)
        ax7.spines["left"].set_bounds(-5, FIG5_Y_AXIS_TOP)
        sns.despine(ax=ax7)
        add_inward_tick_guides(ax7, x_positions=[0, 1], y_ticks=FIG5_Y_TICKS)
        if fval == plot_forces[0]:
            ax7.set_ylabel(Y_LABEL, fontsize=FONT_LABEL)
        else:
            ax7.set_ylabel("")

    leg_handles_f7 = [
        mpatches.Patch(
            facecolor=pale_box_face(MOA_PALETTE[grp]),
            edgecolor=BLACK,
            linewidth=BOX_LINEWIDTH,
            label=MOA_X_LABELS[i].replace("\n", " "),
        )
        for i, grp in enumerate(MOA_GROUP_ORDER)
    ]
    add_fig5_legend(fig7, leg_handles_f7, ncol=len(MOA_GROUP_ORDER))
    fig7.subplots_adjust(left=0.08, right=0.97, top=0.88, bottom=0.12, wspace=0.18)
    out_f7 = os.path.join(FIG_DIR, "onnail_vs_offnail_pooled_n30.png")
    save_png_at_width(fig7, out_f7, width_px=EXPORT_WIDTH_2COL)
    print(f"Saved: {out_f7}")

    # =========================================================================
    # Figure 8a: On-nail (B+C+D+E) vs Off-nail (A+F) — trial-level pooling, n=30
    # Per subject per force: mean of ALL B+C+D+E trials → On-nail (n=30)
    #                        mean of ALL A+F trials     → Off-nail (n=30)
    # Equal sample sizes; On-nail estimate uses 2× more trials (more stable).
    # =========================================================================
    df_pool8 = df_analysis.copy()
    df_pool8["Area"] = df_pool8["Area"].replace({
        "B": "On-nail", "C": "On-nail", "D": "On-nail", "E": "On-nail",
        "A": "Off-nail", "F": "Off-nail",
    })

    F8A_GROUP_ORDER = ["On-nail", "Off-nail"]
    F8A_PALETTE = {
        "On-nail":  ON_TOUCH,
        "Off-nail": "#7C94B8",
    }
    F8A_X_LABELS = ["On-nail\n(B+C+D+E)", "Off-nail\n(A+F)"]

    print("\n[Figure 8a — On-nail(B+C+D+E) vs Off-nail(A+F) | trial-level pool, n=30, by force]")

    df_pool8_by_force = subject_mean_accuracy_regions_by_force(
        df_pool8, sub_col, F8A_GROUP_ORDER
    )

    fig8a, axes8a = plt.subplots(1, len(plot_forces), figsize=FIG5_SIZE, facecolor="white")
    if len(plot_forces) == 1:
        axes8a = [axes8a]

    rng8a = np.random.default_rng(8)

    for ax8a, fval in zip(axes8a, plot_forces):
        df_f8a = df_pool8_by_force[
            np.isclose(df_pool8_by_force["Force_Val"], fval)
        ].rename(columns={"Area": "Group"})

        lme_f8a = lme_two_groups_pooled(
            df_f8a, sub_col, ref_group="Off-nail", target_group="On-nail"
        )
        if lme_f8a:
            star_8a = ("***" if lme_f8a["p"] < 0.001 else
                       "**"  if lme_f8a["p"] < 0.01  else
                       "*"   if lme_f8a["p"] < 0.05  else "n.s.")
            print(f"  {fval}g  On-nail vs Off-nail: Δ={lme_f8a['coef']:.3f} "
                  f"[{lme_f8a['ci_lo']:.3f}, {lme_f8a['ci_hi']:.3f}], "
                  f"p={lme_f8a['p']:.4f}  {star_8a}")
        else:
            print(f"  {fval}g  LME failed")

        tops_8a = {}
        for xi, grp in enumerate(F8A_GROUP_ORDER):
            grp_data = df_f8a[df_f8a["Group"] == grp]["accuracy"].dropna().values
            bp = ax8a.boxplot(
                [grp_data], positions=[xi], widths=0.45,
                patch_artist=True, showfliers=False, zorder=2,
                whiskerprops=dict(color=BLACK, linewidth=BOX_LINEWIDTH),
                capprops=dict(color=BLACK, linewidth=CAP_LINEWIDTH),
                medianprops=dict(color=ACCENT_RED, linewidth=2.0),
                boxprops=dict(facecolor=pale_box_face(F8A_PALETTE[grp]),
                              edgecolor=BLACK, linewidth=BOX_LINEWIDTH),
            )
            tops_8a[grp] = max(w.get_ydata()[1] for w in bp["whiskers"])
            jitter = rng8a.uniform(-0.12, 0.12, size=len(grp_data))
            ax8a.scatter(xi + jitter, grp_data,
                         c=[_hsb_scatter_rgba(F8A_PALETTE[grp])] * len(grp_data),
                         s=3.5 ** 2, linewidths=0, zorder=3, clip_on=False)

        if lme_f8a:
            star8a = ("***" if lme_f8a["p"] < 0.001 else "**" if lme_f8a["p"] < 0.01
                      else "*" if lme_f8a["p"] < 0.05 else "n.s.")
            _add_sig_bracket(ax8a, 0, 1, max(tops_8a.values()) + 4,
                             text=f"{star8a}  p={lme_f8a['p']:.3f}")

        ax8a.axhline(80, color=CRITERION_COLOR, linestyle="--", linewidth=1.0,
                     alpha=0.85, zorder=atd_c1.REF_LINE_ZORDER)
        _set_force_title_above(ax8a, fval, y=1.04)
        ax8a.set_xticks([0, 1])
        ax8a.set_xticklabels(F8A_X_LABELS, fontsize=FONT_FIG5_XTICK)
        ax8a.set_yticks(FIG5_Y_TICKS)
        ax8a.yaxis.set_major_locator(FixedLocator(FIG5_Y_TICKS))
        ax8a.tick_params(axis="y", labelsize=FONT_TICK)
        ax8a.tick_params(axis="x", length=0)
        ax8a.set_ylim(-5, min(FIG5_YLIM_TOP_CAP, max(tops_8a.values()) + 20))
        ax8a.spines["left"].set_bounds(-5, FIG5_Y_AXIS_TOP)
        sns.despine(ax=ax8a)
        add_inward_tick_guides(ax8a, x_positions=[0, 1], y_ticks=FIG5_Y_TICKS)
        if fval == plot_forces[0]:
            ax8a.set_ylabel(Y_LABEL, fontsize=FONT_LABEL)

    leg_f8a = [
        mpatches.Patch(facecolor=pale_box_face(F8A_PALETTE[g]), edgecolor=BLACK,
                       linewidth=BOX_LINEWIDTH, label=F8A_X_LABELS[i].replace("\n", " "))
        for i, g in enumerate(F8A_GROUP_ORDER)
    ]
    add_fig5_legend(fig8a, leg_f8a, ncol=len(F8A_GROUP_ORDER))
    fig8a.subplots_adjust(left=0.08, right=0.97, top=0.88, bottom=0.12, wspace=0.18)
    out_f8a = os.path.join(FIG_DIR, "onnail_bcde_vs_offnail_af_n30.png")
    save_png_at_width(fig8a, out_f8a, width_px=EXPORT_WIDTH_2COL)
    print(f"Saved: {out_f8a}")
    plt.close(fig8a)

    # =========================================================================
    # Figure 8b: On-nail (B+C+D+E) vs Off-nail (A+F) — per-region pooling
    # Each subject contributes 4 obs to On-nail (B,C,D,E means) and
    # 2 obs to Off-nail (A,F means) → n=120 vs n=60 (unequal).
    # LME with random intercept for Subject handles the imbalance.
    # =========================================================================
    F8B_GROUP_MAP = {
        "B": "On-nail", "C": "On-nail", "D": "On-nail", "E": "On-nail",
        "A": "Off-nail", "F": "Off-nail",
    }
    F8B_GROUP_ORDER = ["On-nail", "Off-nail"]
    F8B_PALETTE = {
        "On-nail":  ON_TOUCH,
        "Off-nail": "#7C94B8",
    }
    F8B_X_LABELS = ["On-nail\n(B+C+D+E, n=120)", "Off-nail\n(A+F, n=60)"]

    print("\n[Figure 8b — On-nail(B+C+D+E) vs Off-nail(A+F) | per-region pool, n=120 vs 60, by force]")

    fig8b, axes8b = plt.subplots(1, len(plot_forces), figsize=FIG5_SIZE, facecolor="white")
    if len(plot_forces) == 1:
        axes8b = [axes8b]

    rng8b = np.random.default_rng(9)

    for ax8b, fval in zip(axes8b, plot_forces):
        df_pool8b_f = subject_area_pool_as_separate(
            df_analysis, sub_col, F8B_GROUP_MAP, force_val=fval
        )

        lme_f8b = lme_two_groups_pooled(
            df_pool8b_f, sub_col, ref_group="Off-nail", target_group="On-nail"
        )
        if lme_f8b:
            star_8b = ("***" if lme_f8b["p"] < 0.001 else
                       "**"  if lme_f8b["p"] < 0.01  else
                       "*"   if lme_f8b["p"] < 0.05  else "n.s.")
            print(f"  {fval}g  On-nail vs Off-nail: Δ={lme_f8b['coef']:.3f} "
                  f"[{lme_f8b['ci_lo']:.3f}, {lme_f8b['ci_hi']:.3f}], "
                  f"p={lme_f8b['p']:.4f}  {star_8b}")
        else:
            print(f"  {fval}g  LME failed")

        tops_8b = {}
        for xi, grp in enumerate(F8B_GROUP_ORDER):
            grp_data = df_pool8b_f[df_pool8b_f["Group"] == grp]["accuracy"].dropna().values
            bp = ax8b.boxplot(
                [grp_data], positions=[xi], widths=0.45,
                patch_artist=True, showfliers=False, zorder=2,
                whiskerprops=dict(color=BLACK, linewidth=BOX_LINEWIDTH),
                capprops=dict(color=BLACK, linewidth=CAP_LINEWIDTH),
                medianprops=dict(color=ACCENT_RED, linewidth=2.0),
                boxprops=dict(facecolor=pale_box_face(F8B_PALETTE[grp]),
                              edgecolor=BLACK, linewidth=BOX_LINEWIDTH),
            )
            tops_8b[grp] = max(w.get_ydata()[1] for w in bp["whiskers"])
            jitter = rng8b.uniform(-0.12, 0.12, size=len(grp_data))
            ax8b.scatter(xi + jitter, grp_data,
                         c=[_hsb_scatter_rgba(F8B_PALETTE[grp])] * len(grp_data),
                         s=3.5 ** 2, linewidths=0, zorder=3, clip_on=False)

        if lme_f8b:
            star8b = ("***" if lme_f8b["p"] < 0.001 else "**" if lme_f8b["p"] < 0.01
                      else "*" if lme_f8b["p"] < 0.05 else "n.s.")
            _add_sig_bracket(ax8b, 0, 1, max(tops_8b.values()) + 4,
                             text=f"{star8b}  p={lme_f8b['p']:.3f}")

        ax8b.axhline(80, color=CRITERION_COLOR, linestyle="--", linewidth=1.0,
                     alpha=0.85, zorder=atd_c1.REF_LINE_ZORDER)
        _set_force_title_above(ax8b, fval, y=1.04)
        ax8b.set_xticks([0, 1])
        ax8b.set_xticklabels(F8B_X_LABELS, fontsize=FONT_FIG5_XTICK)
        ax8b.set_yticks(FIG5_Y_TICKS)
        ax8b.yaxis.set_major_locator(FixedLocator(FIG5_Y_TICKS))
        ax8b.tick_params(axis="y", labelsize=FONT_TICK)
        ax8b.tick_params(axis="x", length=0)
        ax8b.set_ylim(-5, min(FIG5_YLIM_TOP_CAP, max(tops_8b.values()) + 20))
        ax8b.spines["left"].set_bounds(-5, FIG5_Y_AXIS_TOP)
        sns.despine(ax=ax8b)
        add_inward_tick_guides(ax8b, x_positions=[0, 1], y_ticks=FIG5_Y_TICKS)
        if fval == plot_forces[0]:
            ax8b.set_ylabel(Y_LABEL, fontsize=FONT_LABEL)

    leg_f8b = [
        mpatches.Patch(facecolor=pale_box_face(F8B_PALETTE[g]), edgecolor=BLACK,
                       linewidth=BOX_LINEWIDTH, label=F8B_X_LABELS[i].replace("\n", " "))
        for i, g in enumerate(F8B_GROUP_ORDER)
    ]
    add_fig5_legend(fig8b, leg_f8b, ncol=len(F8B_GROUP_ORDER))
    fig8b.subplots_adjust(left=0.08, right=0.97, top=0.88, bottom=0.12, wspace=0.18)
    out_f8b = os.path.join(FIG_DIR, "onnail_bcde_vs_offnail_af_pooled.png")
    save_png_at_width(fig8b, out_f8b, width_px=EXPORT_WIDTH_2COL)
    print(f"Saved: {out_f8b}")
    plt.close(fig8b)
    plt.close(fig7)