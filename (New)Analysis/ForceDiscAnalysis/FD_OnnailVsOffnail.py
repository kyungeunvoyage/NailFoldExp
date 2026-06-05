"""
Force Discrimination – On-nail (C+D) vs Off-nail (A+F) by Force Pair
=====================================================================
Three aggregation approaches saved as separate figures:

  Approach 0 (current / n=25):
    Per subject: mean all C+D trials → one accuracy value per group per pair
    On-nail n=25, Off-nail n=25  |  LME with random intercept (Subject)

  Approach A (pooled / n=50):
    Per subject per region: mean C trials, mean D trials kept separate.
    Pool C-means + D-means → On-nail n=50 per pair
    Pool A-means + F-means → Off-nail n=50 per pair
    LME with random intercept (Subject) – handles 2 obs per subject

  Approach B (trial-level GEE):
    Each trial is a binary observation (correct=1 / incorrect=0).
    GEE: Binomial family, Exchangeable working correlation, groups=Subject
    Visualisation: per-subject means shown as scatter; p-value from GEE
"""

import os
import glob
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.cov_struct import Exchangeable
from matplotlib.ticker import FixedLocator

# ── Style from ATD_C1_Fig(Anika).py ──────────────────────────────────────────
_SCRIPT_DIR  = Path(__file__).resolve().parent
_ATD_C1_PATH = _SCRIPT_DIR.parent / "ATDAnalysis" / "ATD_C1_Fig(Anika).py"

spec = importlib.util.spec_from_file_location("atd_c1", _ATD_C1_PATH)
ATD  = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ATD)

BLACK           = ATD.BLACK
ACCENT_RED      = ATD.ACCENT_RED
CRITERION_COLOR = ATD.CRITERION_COLOR
REF_LINE_ZORDER = ATD.REF_LINE_ZORDER
FONT_TICK       = ATD.FONT_TICK
FONT_LABEL      = ATD.FONT_LABEL
FONT_ANNOT      = ATD.FONT_ANNOT
BOX_LINEWIDTH   = ATD.BOX_LINEWIDTH
CAP_LINEWIDTH   = ATD.CAP_LINEWIDTH
STRIP_ALPHA     = ATD.STRIP_ALPHA
pale_box_face   = ATD.pale_box_face
_hsb_scatter_rgba  = ATD._hsb_scatter_rgba
add_legend_outside = ATD.add_legend_outside
FIG_LEGEND_TOP    = ATD.FIG_LEGEND_TOP
FIG_LEGEND_BOTTOM = ATD.FIG_LEGEND_BOTTOM

ON_NAIL_COLOR  = ATD.ON_TOUCH   # "#10559A"  blue
OFF_NAIL_COLOR = "#7C94B8"      # same as Off-Nail (A) in ATD_aggregate

OUTPUT_DIR = str(_SCRIPT_DIR / "Output" / "OnnailVsOffnail")
os.makedirs(OUTPUT_DIR, exist_ok=True)

EXPORT_WIDTH_2COL = 2102
JND_PCT = 75.0

# ── Load data ─────────────────────────────────────────────────────────────────
FILE_PATTERN = "/Users/kyungeunjung/NailFoldExp/Data/(FD)CurData/P*_ForceDiscrimination.csv"
files = glob.glob(FILE_PATTERN)
if not files:
    raise FileNotFoundError(f"No CSV files found: {FILE_PATTERN}")
print(f"Loaded {len(files)} participant file(s).")

df_raw = pd.concat(
    [pd.read_csv(f, encoding="utf-8-sig") for f in sorted(files)],
    ignore_index=True,
)

# ── Derived columns ───────────────────────────────────────────────────────────
df_raw["correct"] = np.where(
    df_raw["Comparison"] > df_raw["Reference"],
    df_raw["ChoseComparison"] == 1,
    df_raw["ChoseComparison"] == 0,
).astype(int)

df_raw["pair_label"] = df_raw.apply(
    lambda r: f"{min(r['Reference'], r['Comparison']):g}–"
              f"{max(r['Reference'], r['Comparison']):g}",
    axis=1,
)
df_raw["band"] = df_raw["Reference"].apply(lambda r: "Low" if r == 1 else "High")

# ── Group mapping: C+D → On-nail,  A+F → Off-nail  (exclude B, E) ────────────
GROUP_MAP = {"C": "On-nail", "D": "On-nail", "A": "Off-nail", "F": "Off-nail"}
df_cd_af = df_raw[df_raw["Region"].isin(GROUP_MAP)].copy()
df_cd_af["Group"] = df_cd_af["Region"].map(GROUP_MAP)

GROUP_ORDER   = ["On-nail", "Off-nail"]
GROUP_PALETTE = {"On-nail": ON_NAIL_COLOR, "Off-nail": OFF_NAIL_COLOR}
GROUP_LABELS  = ["On-nail (C+D)", "Off-nail (A+F)"]

# =============================================================================
#  Aggregated data sets
# =============================================================================

# ── Approach 0: trial-level pooling per subject  (n=25) ─────────────────────
subj_acc = (
    df_cd_af.groupby(["Subject", "band", "pair_label", "Group"], as_index=False)
    ["correct"].mean()
    .rename(columns={"correct": "accuracy"})
)
subj_acc["accuracy"] *= 100

# ── Approach A: per-region subject means pooled  (n=50) ──────────────────────
subj_region_acc = (
    df_cd_af.groupby(
        ["Subject", "band", "pair_label", "Region"], as_index=False
    )["correct"].mean()
    .rename(columns={"correct": "accuracy"})
)
subj_region_acc["accuracy"] *= 100
subj_region_acc["Group"] = subj_region_acc["Region"].map(GROUP_MAP)

# =============================================================================
#  Statistical helpers
# =============================================================================

def lme_two_groups(df_in, ref_group="Off-nail", target_group="On-nail",
                   score_col="accuracy"):
    """LME with random intercept for Subject."""
    sub = df_in[df_in["Group"].isin([ref_group, target_group])].dropna(
        subset=["Subject", "Group", score_col]
    )
    if sub["Subject"].nunique() < 2 or sub["Group"].nunique() < 2:
        return None
    formula = f"{score_col} ~ C(Group, Treatment(reference='{ref_group}'))"
    try:
        res = smf.mixedlm(formula, sub, groups=sub["Subject"]).fit(reml=True)
        col = f"C(Group, Treatment(reference='{ref_group}'))[T.{target_group}]"
        if col not in res.params.index:
            return None
        ci = res.conf_int().loc[col]
        return {"coef": float(res.params[col]),
                "ci_lo": float(ci[0]), "ci_hi": float(ci[1]),
                "p": float(res.pvalues[col])}
    except Exception:
        return None


def gee_two_groups(df_in, ref_group="Off-nail", target_group="On-nail"):
    """
    GEE on trial-level binary data.
    Returns log-odds coef + p-value.
    Visualisation still uses per-subject means (same as Approach 0).
    """
    sub = df_in[df_in["Group"].isin([ref_group, target_group])].dropna(
        subset=["Subject", "Group", "correct"]
    ).copy()
    sub["Group_bin"] = (sub["Group"] == target_group).astype(int)
    if sub["Subject"].nunique() < 2 or sub["Group"].nunique() < 2:
        return None
    try:
        X = sm.add_constant(sub["Group_bin"])
        model = GEE(
            sub["correct"], X,
            groups=sub["Subject"],
            family=Binomial(),
            cov_struct=Exchangeable(),
        )
        res = model.fit()
        p     = float(res.pvalues["Group_bin"])
        coef  = float(res.params["Group_bin"])
        ci    = res.conf_int()
        return {"coef": coef,
                "ci_lo": float(ci.loc["Group_bin", 0]),
                "ci_hi": float(ci.loc["Group_bin", 1]),
                "p": p}
    except Exception as e:
        print(f"  GEE failed: {e}")
        return None


def star(p):
    return ("***" if p < 0.001 else "**" if p < 0.01 else
            "*" if p < 0.05 else "n.s.")


# ── Band configs ──────────────────────────────────────────────────────────────
BANDS = [
    {"name": "Low",  "ref": 1,  "title": "Low band  (ref = 1 g)"},
    {"name": "High", "ref": 26, "title": "High band  (ref = 26 g)"},
]

BOX_W    = 0.28
GAP      = 0.08
TICK_LEN = ATD.TICK_LEN_AXES
Y_TICKS  = [0, 25, 50, 75, 100]
YLIM_BOT = -5
YLIM_TOP_CAP = 130


# =============================================================================
#  Shared figure-drawing function
# =============================================================================

def draw_fd_figure(agg_df, stat_fn, stat_label, fname,
                   scatter_col="accuracy", fig_title_suffix=""):
    """
    Parameters
    ----------
    agg_df      : subject-level data used for box/scatter (must have Group, accuracy)
    stat_fn     : callable(pair_df) → dict with 'p' key  (or None)
    stat_label  : short string printed to console for identification
    fname       : output filename (no path)
    scatter_col : column name for y-axis values
    """
    ATD.apply_plot_style()
    sns.set_theme(style="white")

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.5), facecolor="white")
    rng = np.random.default_rng(42)

    print(f"\n[FD — {stat_label} | On-nail (C+D) vs Off-nail (A+F)]")

    for ax, band_cfg in zip(axes, BANDS):
        band_name  = band_cfg["name"]
        band_title = band_cfg["title"]

        band_df   = agg_df[agg_df["band"] == band_name]
        pairs     = sorted(band_df["pair_label"].unique(),
                           key=lambda s: float(s.split("–")[1]))
        n_pairs   = len(pairs)
        x_centers = np.arange(n_pairs)

        print(f"\n  [{band_name} band]")
        pair_tops = []

        for xi, pair in enumerate(pairs):
            pair_df   = band_df[band_df["pair_label"] == pair]

            # stats on the raw/appropriate df — stat_fn closure captures df_cd_af
            stat_res  = stat_fn(pair, band_name)
            if stat_res:
                print(f"    {pair} g  On- vs Off-nail: "
                      f"coef={stat_res['coef']:.2f} "
                      f"[{stat_res['ci_lo']:.2f}, {stat_res['ci_hi']:.2f}], "
                      f"p={stat_res['p']:.4f}  {star(stat_res['p'])}")
            else:
                print(f"    {pair} g  stat failed")

            whisker_tops = []
            for gi, grp in enumerate(GROUP_ORDER):
                dx   = (gi - 0.5) * (BOX_W * 2 + GAP)
                xpos = xi + dx
                data = pair_df[pair_df["Group"] == grp][scatter_col].dropna().values
                if len(data) == 0:
                    continue

                bp = ax.boxplot(
                    [data], positions=[xpos], widths=BOX_W * 1.6,
                    patch_artist=True, showfliers=False, zorder=2,
                    whiskerprops=dict(color=BLACK, linewidth=BOX_LINEWIDTH),
                    capprops=dict(color=BLACK, linewidth=CAP_LINEWIDTH),
                    medianprops=dict(color=ACCENT_RED, linewidth=2.0),
                    boxprops=dict(facecolor=pale_box_face(GROUP_PALETTE[grp]),
                                  edgecolor=BLACK, linewidth=BOX_LINEWIDTH),
                )
                w_top = max(w.get_ydata()[1] for w in bp["whiskers"])
                whisker_tops.append(w_top)

                jitter = rng.uniform(-BOX_W * 0.5, BOX_W * 0.5, size=len(data))
                rgba   = _hsb_scatter_rgba(GROUP_PALETTE[grp])
                ax.scatter(xpos + jitter, data,
                           c=[rgba] * len(data), s=3.5 ** 2,
                           linewidths=0, zorder=3, clip_on=False)

            pair_top = max(whisker_tops) if whisker_tops else 80.0
            pair_tops.append(pair_top)

            if stat_res:
                x_on  = xi + (0 - 0.5) * (BOX_W * 2 + GAP)
                x_off = xi + (1 - 0.5) * (BOX_W * 2 + GAP)
                y_brk = pair_top + 3
                x_mid = (x_on + x_off) / 2
                y_top_brk = y_brk + 0.5
                ax.plot([x_on, x_on, x_off, x_off],
                        [y_brk, y_top_brk, y_top_brk, y_brk],
                        color=ACCENT_RED, linewidth=0.75, clip_on=False, zorder=5)
                p_txt = f"{star(stat_res['p'])}  p={stat_res['p']:.3f}"
                ax.text(x_mid, y_top_brk + 0.6, p_txt,
                        ha="center", va="bottom",
                        fontsize=max(8, FONT_ANNOT - 1),
                        color=ACCENT_RED, fontweight="bold",
                        clip_on=False, zorder=6)

        ax.axhline(JND_PCT, color=CRITERION_COLOR, linestyle="--",
                   linewidth=1.0, alpha=0.85, zorder=REF_LINE_ZORDER)
        ax.set_title(band_title, fontsize=FONT_LABEL, fontweight="bold", pad=6)
        ax.set_xticks(x_centers)
        ax.set_xticklabels([f"{p} g" for p in pairs], fontsize=FONT_TICK - 2)
        ax.set_yticks(Y_TICKS)
        ax.yaxis.set_major_locator(FixedLocator(Y_TICKS))
        ax.tick_params(axis="y", labelsize=FONT_TICK)
        ax.tick_params(axis="x", length=0)
        y_top = min(YLIM_TOP_CAP, max(pair_tops) + 18) if pair_tops else 120
        ax.set_ylim(YLIM_BOT, y_top)
        ax.spines["left"].set_bounds(YLIM_BOT, 100)
        sns.despine(ax=ax)

        x_trans = ax.get_xaxis_transform()
        for xi in x_centers:
            ax.plot([xi, xi], [0, TICK_LEN], color=BLACK,
                    linewidth=1.0, solid_capstyle="butt",
                    transform=x_trans, clip_on=False, zorder=6)
        y_trans = ax.get_yaxis_transform()
        y_lo, y_hi = ax.get_ylim()
        for y in Y_TICKS:
            if y_lo - 1e-9 <= y <= y_hi + 1e-9:
                ax.plot([0, TICK_LEN], [y, y], color=BLACK,
                        linewidth=1.0, solid_capstyle="butt",
                        transform=y_trans, clip_on=False, zorder=6)

    axes[0].set_ylabel("Accuracy (%)", fontsize=FONT_LABEL)
    axes[1].set_ylabel("")

    leg_handles = [
        mpatches.Patch(facecolor=pale_box_face(GROUP_PALETTE[g]),
                       edgecolor=BLACK, linewidth=BOX_LINEWIDTH,
                       label=GROUP_LABELS[i])
        for i, g in enumerate(GROUP_ORDER)
    ]
    add_legend_outside(fig, axes[0], leg_handles, ncol=2,
                       top=FIG_LEGEND_TOP, bottom=FIG_LEGEND_BOTTOM,
                       left=0.07, right=0.97,
                       above_axes=ATD.FIG_LEGEND_ABOVE_AXES)

    fig.subplots_adjust(left=0.07, right=0.97, top=0.88, bottom=0.12, wspace=0.28)

    out_path = os.path.join(OUTPUT_DIR, fname)
    w_in, _ = fig.get_size_inches()
    fig.savefig(out_path, dpi=EXPORT_WIDTH_2COL / w_in,
                bbox_inches="tight", pad_inches=0.04, facecolor="white")
    print(f"\nSaved: {out_path}")
    plt.close(fig)


# =============================================================================
#  Approach 0 — trial-level pooling per subject (n=25)
# =============================================================================
def _stat0(pair, band):
    df = subj_acc[(subj_acc["pair_label"] == pair) & (subj_acc["band"] == band)]
    return lme_two_groups(df)

draw_fd_figure(
    agg_df=subj_acc,
    stat_fn=_stat0,
    stat_label="Approach 0: LME, trial-level pool per subject (n=25)",
    fname="fd_onnail_vs_offnail_by_pair.png",
)

# =============================================================================
#  Approach A — per-region subject means pooled (n=50 each)
# =============================================================================
def _stat_A(pair, band):
    df = subj_region_acc[
        (subj_region_acc["pair_label"] == pair) &
        (subj_region_acc["band"] == band)
    ]
    return lme_two_groups(df)

draw_fd_figure(
    agg_df=subj_region_acc,
    stat_fn=_stat_A,
    stat_label="Approach A: LME, per-region means pooled (n=50)",
    fname="fd_onnail_vs_offnail_pooled_n50.png",
)

# =============================================================================
#  Approach B — trial-level GEE (binary 0/1, Binomial, Exchangeable)
#  Visualisation: same scatter as Approach 0 (per-subject means, n=25)
#  Statistics   : GEE on raw binary trials
# =============================================================================
def _stat_B(pair, band):
    df = df_cd_af[
        (df_cd_af["pair_label"] == pair) &
        (df_cd_af["band"] == band)
    ]
    return gee_two_groups(df)

draw_fd_figure(
    agg_df=subj_acc,          # same box/scatter as Approach 0
    stat_fn=_stat_B,
    stat_label="Approach B: GEE (binary, Binomial, Exchangeable), viz=subj means",
    fname="fd_onnail_vs_offnail_gee.png",
)
