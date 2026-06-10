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


# =============================================================================
#  Method 1 — Band-level box + scatter (more continuous values)
#  Aggregation: per subject, mean of ALL C+D (or A+F) trials across every
#               force pair within a band.
#               Low  On-nail: 4 pairs × 2 regions × 2 trials = 16 trials → 17 distinct values
#               High On-nail: 3 pairs × 2 regions × 2 trials = 12 trials → 13 distinct values
# =============================================================================

subj_band_acc = (
    df_cd_af.groupby(["Subject", "band", "Group"], as_index=False)
    ["correct"].mean()
    .rename(columns={"correct": "accuracy"})
)
subj_band_acc["accuracy"] *= 100

BAND_ORDER  = ["Low", "High"]
BAND_LABELS = ["Low band\n(ref = 1 g)", "High band\n(ref = 26 g)"]

def lme_band(band_name):
    """LME for On-nail vs Off-nail within a band (all pairs collapsed)."""
    df = subj_band_acc[subj_band_acc["band"] == band_name]
    return lme_two_groups(df)

print("\n[FD — Method 1: Band-level box+scatter (all force pairs collapsed)]")
for b in BAND_ORDER:
    res = lme_band(b)
    if res:
        print(f"  {b} band  On- vs Off-nail: Δ={res['coef']:.1f}% "
              f"[{res['ci_lo']:.1f}, {res['ci_hi']:.1f}], "
              f"p={res['p']:.4f}  {star(res['p'])}")

ATD.apply_plot_style()
sns.set_theme(style="white")

fig_c, ax_c = plt.subplots(1, 1, figsize=(5.5, 4.5), facecolor="white")
rng_c = np.random.default_rng(42)

BOX_W_C = 0.28
GAP_C   = 0.10

band_tops_c = []
for xi, (bname, blabel) in enumerate(zip(BAND_ORDER, BAND_LABELS)):
    band_df = subj_band_acc[subj_band_acc["band"] == bname]
    lme_res = lme_band(bname)

    whisker_tops_c = []
    for gi, grp in enumerate(GROUP_ORDER):
        dx   = (gi - 0.5) * (BOX_W_C * 2 + GAP_C)
        xpos = xi + dx
        data = band_df[band_df["Group"] == grp]["accuracy"].dropna().values
        if len(data) == 0:
            continue

        bp = ax_c.boxplot(
            [data], positions=[xpos], widths=BOX_W_C * 1.6,
            patch_artist=True, showfliers=False, zorder=2,
            whiskerprops=dict(color=BLACK, linewidth=BOX_LINEWIDTH),
            capprops=dict(color=BLACK, linewidth=CAP_LINEWIDTH),
            medianprops=dict(color=ACCENT_RED, linewidth=2.0),
            boxprops=dict(facecolor=pale_box_face(GROUP_PALETTE[grp]),
                          edgecolor=BLACK, linewidth=BOX_LINEWIDTH),
        )
        w_top = max(w.get_ydata()[1] for w in bp["whiskers"])
        whisker_tops_c.append(w_top)

        jitter_c = rng_c.uniform(-BOX_W_C * 0.5, BOX_W_C * 0.5, size=len(data))
        rgba = _hsb_scatter_rgba(GROUP_PALETTE[grp])
        ax_c.scatter(xpos + jitter_c, data,
                     c=[rgba] * len(data), s=3.5 ** 2,
                     linewidths=0, zorder=3, clip_on=False)

    band_top = max(whisker_tops_c) if whisker_tops_c else 80.0
    band_tops_c.append(band_top)

    if lme_res:
        x_on  = xi + (0 - 0.5) * (BOX_W_C * 2 + GAP_C)
        x_off = xi + (1 - 0.5) * (BOX_W_C * 2 + GAP_C)
        y_brk = band_top + 3
        x_mid = (x_on + x_off) / 2
        y_top_brk = y_brk + 0.5
        ax_c.plot([x_on, x_on, x_off, x_off],
                  [y_brk, y_top_brk, y_top_brk, y_brk],
                  color=ACCENT_RED, linewidth=0.75, clip_on=False, zorder=5)
        p_txt = f"{star(lme_res['p'])}  p={lme_res['p']:.3f}"
        ax_c.text(x_mid, y_top_brk + 0.6, p_txt,
                  ha="center", va="bottom",
                  fontsize=max(8, FONT_ANNOT - 1),
                  color=ACCENT_RED, fontweight="bold",
                  clip_on=False, zorder=6)

ax_c.axhline(JND_PCT, color=CRITERION_COLOR, linestyle="--",
             linewidth=1.0, alpha=0.85, zorder=REF_LINE_ZORDER)
ax_c.set_xticks(range(len(BAND_ORDER)))
ax_c.set_xticklabels(BAND_LABELS, fontsize=FONT_TICK)
ax_c.set_yticks(Y_TICKS)
ax_c.yaxis.set_major_locator(FixedLocator(Y_TICKS))
ax_c.tick_params(axis="y", labelsize=FONT_TICK)
ax_c.tick_params(axis="x", length=0)
y_top_c = min(YLIM_TOP_CAP, max(band_tops_c) + 18) if band_tops_c else 120
ax_c.set_ylim(YLIM_BOT, y_top_c)
ax_c.set_ylabel("Accuracy (%)", fontsize=FONT_LABEL)
ax_c.spines["left"].set_bounds(YLIM_BOT, 100)
sns.despine(ax=ax_c)

x_trans_c = ax_c.get_xaxis_transform()
for xi in range(len(BAND_ORDER)):
    ax_c.plot([xi, xi], [0, TICK_LEN], color=BLACK, linewidth=1.0,
              solid_capstyle="butt", transform=x_trans_c, clip_on=False, zorder=6)
y_trans_c = ax_c.get_yaxis_transform()
for y in Y_TICKS:
    if YLIM_BOT - 1e-9 <= y <= y_top_c + 1e-9:
        ax_c.plot([0, TICK_LEN], [y, y], color=BLACK, linewidth=1.0,
                  solid_capstyle="butt", transform=y_trans_c, clip_on=False, zorder=6)

leg_handles_c = [
    mpatches.Patch(facecolor=pale_box_face(GROUP_PALETTE[g]),
                   edgecolor=BLACK, linewidth=BOX_LINEWIDTH, label=GROUP_LABELS[i])
    for i, g in enumerate(GROUP_ORDER)
]
add_legend_outside(fig_c, ax_c, leg_handles_c, ncol=2,
                   top=FIG_LEGEND_TOP, bottom=FIG_LEGEND_BOTTOM,
                   left=0.10, right=0.95,
                   above_axes=ATD.FIG_LEGEND_ABOVE_AXES)

fig_c.subplots_adjust(left=0.10, right=0.95, top=0.88, bottom=0.12)

out_c = os.path.join(OUTPUT_DIR, "fd_onnail_vs_offnail_band.png")
import io as _io
from PIL import Image as _Image
_buf = _io.BytesIO()
fig_c.savefig(_buf, format="png", dpi=600,
              bbox_inches="tight", pad_inches=0.04, facecolor="white")
_buf.seek(0)
_master = _Image.open(_buf).convert("RGB")
_h_px = round(EXPORT_WIDTH_2COL * _master.height / _master.width)
_master.resize((EXPORT_WIDTH_2COL, _h_px), _Image.Resampling.LANCZOS).save(out_c)
print(f"\nSaved: {out_c}  ({EXPORT_WIDTH_2COL}×{_h_px} px @ 600 dpi render)")
plt.close(fig_c)


# =============================================================================
#  Method 3 — Mean ± 95% CI per force pair  (no scatter, fully continuous)
#  X-axis: force pairs within Low / High band panels
#  Error bar: 95% CI = mean ± 1.96 × SEM across subjects
#  Statistics: same GEE as Approach B
# =============================================================================

CI95_MULTIPLIER = 1.96
DOT_SIZE = 7.0       # marker size (pt)
DOT_LW   = 0.0       # no edge line on marker
CAP_W    = 0.06      # error bar cap width in x-data units
ERR_LW   = 1.5       # error bar line width

def draw_fd_meanCI_figure():
    ATD.apply_plot_style()
    sns.set_theme(style="white")

    fig_d, axes_d = plt.subplots(1, 2, figsize=(10.0, 4.5), facecolor="white")
    print("\n[FD — Method 3: Mean ± 95% CI per force pair | GEE stats]")

    for ax, band_cfg in zip(axes_d, BANDS):
        band_name  = band_cfg["name"]
        band_title = band_cfg["title"]

        band_df   = subj_acc[subj_acc["band"] == band_name]
        pairs     = sorted(band_df["pair_label"].unique(),
                           key=lambda s: float(s.split("–")[1]))
        n_pairs   = len(pairs)
        x_centers = np.arange(n_pairs)

        y_max_used = 0.0

        for xi, pair in enumerate(pairs):
            pair_df  = band_df[band_df["pair_label"] == pair]
            stat_res = _stat_B(pair, band_name)   # reuse GEE stat

            for gi, grp in enumerate(GROUP_ORDER):
                dx   = (gi - 0.5) * (BOX_W * 2 + GAP)
                xpos = xi + dx
                data = pair_df[pair_df["Group"] == grp]["accuracy"].dropna().values
                if len(data) == 0:
                    continue
                n     = len(data)
                mean  = float(np.mean(data))
                sem   = float(np.std(data, ddof=1) / np.sqrt(n))
                ci    = CI95_MULTIPLIER * sem
                ci_lo = mean - ci
                ci_hi = mean + ci

                color = GROUP_PALETTE[grp]
                # vertical error bar
                ax.plot([xpos, xpos], [ci_lo, ci_hi],
                        color=color, linewidth=ERR_LW, zorder=2, clip_on=False)
                # caps
                for cap_y in (ci_lo, ci_hi):
                    ax.plot([xpos - CAP_W, xpos + CAP_W], [cap_y, cap_y],
                            color=color, linewidth=ERR_LW, zorder=2, clip_on=False)
                # mean dot
                ax.scatter([xpos], [mean],
                           c=[color], s=DOT_SIZE ** 2, linewidths=DOT_LW,
                           zorder=4, clip_on=False)
                y_max_used = max(y_max_used, ci_hi)

            # significance bracket (GEE)
            if stat_res:
                x_on  = xi + (0 - 0.5) * (BOX_W * 2 + GAP)
                x_off = xi + (1 - 0.5) * (BOX_W * 2 + GAP)
                # find top ci_hi among the two groups for bracket base
                ci_tops = []
                for gi, grp in enumerate(GROUP_ORDER):
                    dx   = (gi - 0.5) * (BOX_W * 2 + GAP)
                    data = pair_df[pair_df["Group"] == grp]["accuracy"].dropna().values
                    if len(data):
                        n    = len(data)
                        ci_h = np.mean(data) + CI95_MULTIPLIER * np.std(data, ddof=1) / np.sqrt(n)
                        ci_tops.append(float(ci_h))
                y_brk     = max(ci_tops) + 2.5 if ci_tops else y_max_used + 2.5
                y_top_brk = y_brk + 0.5
                x_mid     = (x_on + x_off) / 2
                ax.plot([x_on, x_on, x_off, x_off],
                        [y_brk, y_top_brk, y_top_brk, y_brk],
                        color=ACCENT_RED, linewidth=0.75, clip_on=False, zorder=5)
                p_txt = f"{star(stat_res['p'])}  p={stat_res['p']:.3f}"
                ax.text(x_mid, y_top_brk + 0.6, p_txt,
                        ha="center", va="bottom",
                        fontsize=max(8, FONT_ANNOT - 1),
                        color=ACCENT_RED, fontweight="bold",
                        clip_on=False, zorder=6)

            if stat_res:
                print(f"  [{band_name}] {pair} g  GEE p={stat_res['p']:.4f}  "
                      f"{star(stat_res['p'])}")

        ax.axhline(JND_PCT, color=CRITERION_COLOR, linestyle="--",
                   linewidth=1.0, alpha=0.85, zorder=REF_LINE_ZORDER)
        ax.set_title(band_title, fontsize=FONT_LABEL, fontweight="bold", pad=6)
        ax.set_xticks(x_centers)
        ax.set_xticklabels([f"{p} g" for p in pairs], fontsize=FONT_TICK - 2)
        ax.set_yticks(Y_TICKS)
        ax.yaxis.set_major_locator(FixedLocator(Y_TICKS))
        ax.tick_params(axis="y", labelsize=FONT_TICK)
        ax.tick_params(axis="x", length=0)
        y_top = min(YLIM_TOP_CAP, y_max_used + 18)
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

    axes_d[0].set_ylabel("Accuracy (%)", fontsize=FONT_LABEL)
    axes_d[1].set_ylabel("")

    leg_handles_d = [
        plt.Line2D([0], [0], color=GROUP_PALETTE[g], marker="o",
                   markersize=DOT_SIZE * 0.7, linewidth=ERR_LW,
                   label=GROUP_LABELS[i])
        for i, g in enumerate(GROUP_ORDER)
    ]
    add_legend_outside(fig_d, axes_d[0], leg_handles_d, ncol=2,
                       top=FIG_LEGEND_TOP, bottom=FIG_LEGEND_BOTTOM,
                       left=0.07, right=0.97,
                       above_axes=ATD.FIG_LEGEND_ABOVE_AXES)

    fig_d.subplots_adjust(left=0.07, right=0.97, top=0.88, bottom=0.12, wspace=0.28)

    out_d = os.path.join(OUTPUT_DIR, "fd_onnail_vs_offnail_meanCI.png")
    w_in_d, _ = fig_d.get_size_inches()
    fig_d.savefig(out_d, dpi=EXPORT_WIDTH_2COL / w_in_d,
                  bbox_inches="tight", pad_inches=0.04, facecolor="white")
    print(f"\nSaved: {out_d}")
    plt.close(fig_d)


draw_fd_meanCI_figure()


# =============================================================================
#  Region-pair comparison: A-C, A-D, F-C, F-D
#  Two figures:
#    (R1) Band-level box + scatter  (force pairs collapsed → more continuous)
#    (R2) Force-pair level mean ± 95% CI  (4-row × 2-col grid)
# =============================================================================

REGION_PAIRS  = [("A", "C"), ("A", "D"), ("F", "C"), ("F", "D")]
PAIR_LABELS   = ["A vs C", "A vs D", "F vs C", "F vs D"]

# Off-nail regions (A, F) use OFF_NAIL_COLOR; On-nail (C, D) use ON_NAIL_COLOR
REGION_COLOR  = {
    "A": OFF_NAIL_COLOR, "F": OFF_NAIL_COLOR,
    "C": ON_NAIL_COLOR,  "D": ON_NAIL_COLOR,
}

# Band-level per-region per-subject mean (force pairs collapsed within band)
_df_regs = df_raw[df_raw["Region"].isin(["A", "C", "D", "F"])].copy()
subj_region_band = (
    _df_regs.groupby(["Subject", "band", "Region"], as_index=False)
    ["correct"].mean()
    .rename(columns={"correct": "accuracy"})
)
subj_region_band["accuracy"] *= 100

# Force-pair level per-region per-subject mean
subj_region_pair = (
    _df_regs.groupby(["Subject", "band", "pair_label", "Region"], as_index=False)
    ["correct"].mean()
    .rename(columns={"correct": "accuracy"})
)
subj_region_pair["accuracy"] *= 100


def lme_two_regions(df_in, r1, r2, score_col="accuracy"):
    """LME: accuracy ~ region, random intercept for Subject.  r1 = reference."""
    sub = df_in[df_in["Region"].isin([r1, r2])].dropna(
        subset=["Subject", "Region", score_col]
    )
    if sub["Subject"].nunique() < 2 or sub["Region"].nunique() < 2:
        return None
    formula = f"{score_col} ~ C(Region, Treatment(reference='{r1}'))"
    try:
        res = smf.mixedlm(formula, sub, groups=sub["Subject"]).fit(reml=True)
        col = f"C(Region, Treatment(reference='{r1}'))[T.{r2}]"
        if col not in res.params.index:
            return None
        ci = res.conf_int().loc[col]
        return {"coef": float(res.params[col]),
                "ci_lo": float(ci[0]), "ci_hi": float(ci[1]),
                "p": float(res.pvalues[col])}
    except Exception:
        return None


# ── Figure R1: Band-level box + scatter ───────────────────────────────────────
def draw_region_pair_band_figure():
    ATD.apply_plot_style()
    sns.set_theme(style="white")

    fig_r, axes_r = plt.subplots(1, 2, figsize=(10.0, 4.5), facecolor="white")
    rng_r = np.random.default_rng(42)

    BOX_W_R = 0.22
    GAP_R   = 0.06

    print("\n[FD — Region pairs: band-level box+scatter | LME]")

    for ax, band_cfg in zip(axes_r, BANDS):
        band_name  = band_cfg["name"]
        band_title = band_cfg["title"]

        band_df   = subj_region_band[subj_region_band["band"] == band_name]
        pair_tops = []

        for xi, (rp, pl) in enumerate(zip(REGION_PAIRS, PAIR_LABELS)):
            r1, r2 = rp
            lme_res = lme_two_regions(band_df, r1, r2)
            if lme_res:
                print(f"  [{band_name}] {pl}: Δ={lme_res['coef']:.1f}% "
                      f"[{lme_res['ci_lo']:.1f}, {lme_res['ci_hi']:.1f}], "
                      f"p={lme_res['p']:.4f}  {star(lme_res['p'])}")
            else:
                print(f"  [{band_name}] {pl}: LME failed")

            whisker_tops = []
            for gi, r in enumerate([r1, r2]):
                dx   = (gi - 0.5) * (BOX_W_R * 2 + GAP_R)
                xpos = xi + dx
                data = band_df[band_df["Region"] == r]["accuracy"].dropna().values
                if len(data) == 0:
                    continue

                bp = ax.boxplot(
                    [data], positions=[xpos], widths=BOX_W_R * 1.6,
                    patch_artist=True, showfliers=False, zorder=2,
                    whiskerprops=dict(color=BLACK, linewidth=BOX_LINEWIDTH),
                    capprops=dict(color=BLACK, linewidth=CAP_LINEWIDTH),
                    medianprops=dict(color=ACCENT_RED, linewidth=2.0),
                    boxprops=dict(facecolor=pale_box_face(REGION_COLOR[r]),
                                  edgecolor=BLACK, linewidth=BOX_LINEWIDTH),
                )
                w_top = max(w.get_ydata()[1] for w in bp["whiskers"])
                whisker_tops.append(w_top)

                jitter_r = rng_r.uniform(-BOX_W_R * 0.5, BOX_W_R * 0.5, size=len(data))
                rgba = _hsb_scatter_rgba(REGION_COLOR[r])
                ax.scatter(xpos + jitter_r, data,
                           c=[rgba] * len(data), s=3.5 ** 2,
                           linewidths=0, zorder=3, clip_on=False)

            pair_top = max(whisker_tops) if whisker_tops else 80.0
            pair_tops.append(pair_top)

            if lme_res:
                x1 = xi + (0 - 0.5) * (BOX_W_R * 2 + GAP_R)
                x2 = xi + (1 - 0.5) * (BOX_W_R * 2 + GAP_R)
                y_brk = pair_top + 3
                y_top_brk = y_brk + 0.5
                ax.plot([x1, x1, x2, x2],
                        [y_brk, y_top_brk, y_top_brk, y_brk],
                        color=ACCENT_RED, linewidth=0.75, clip_on=False, zorder=5)
                p_txt = f"{star(lme_res['p'])}  p={lme_res['p']:.3f}"
                ax.text((x1 + x2) / 2, y_top_brk + 0.6, p_txt,
                        ha="center", va="bottom",
                        fontsize=max(8, FONT_ANNOT - 1),
                        color=ACCENT_RED, fontweight="bold",
                        clip_on=False, zorder=6)

        ax.axhline(JND_PCT, color=CRITERION_COLOR, linestyle="--",
                   linewidth=1.0, alpha=0.85, zorder=REF_LINE_ZORDER)
        ax.set_title(band_title, fontsize=FONT_LABEL, fontweight="bold", pad=6)
        ax.set_xticks(range(len(REGION_PAIRS)))
        ax.set_xticklabels(PAIR_LABELS, fontsize=FONT_TICK - 1)
        ax.set_yticks(Y_TICKS)
        ax.yaxis.set_major_locator(FixedLocator(Y_TICKS))
        ax.tick_params(axis="y", labelsize=FONT_TICK)
        ax.tick_params(axis="x", length=0)
        y_top = min(YLIM_TOP_CAP, max(pair_tops) + 18) if pair_tops else 120
        ax.set_ylim(YLIM_BOT, y_top)
        ax.spines["left"].set_bounds(YLIM_BOT, 100)
        sns.despine(ax=ax)

        x_trans = ax.get_xaxis_transform()
        for xi in range(len(REGION_PAIRS)):
            ax.plot([xi, xi], [0, TICK_LEN], color=BLACK, linewidth=1.0,
                    solid_capstyle="butt", transform=x_trans, clip_on=False, zorder=6)
        y_trans = ax.get_yaxis_transform()
        y_lo, y_hi = ax.get_ylim()
        for y in Y_TICKS:
            if y_lo - 1e-9 <= y <= y_hi + 1e-9:
                ax.plot([0, TICK_LEN], [y, y], color=BLACK, linewidth=1.0,
                        solid_capstyle="butt", transform=y_trans, clip_on=False, zorder=6)

    axes_r[0].set_ylabel("Accuracy (%)", fontsize=FONT_LABEL)
    axes_r[1].set_ylabel("")

    leg_r = [
        mpatches.Patch(facecolor=pale_box_face(OFF_NAIL_COLOR),
                       edgecolor=BLACK, linewidth=BOX_LINEWIDTH,
                       label="Off-nail regions (A, F)"),
        mpatches.Patch(facecolor=pale_box_face(ON_NAIL_COLOR),
                       edgecolor=BLACK, linewidth=BOX_LINEWIDTH,
                       label="On-nail regions (C, D)"),
    ]
    add_legend_outside(fig_r, axes_r[0], leg_r, ncol=2,
                       top=FIG_LEGEND_TOP, bottom=FIG_LEGEND_BOTTOM,
                       left=0.07, right=0.97,
                       above_axes=ATD.FIG_LEGEND_ABOVE_AXES)

    fig_r.subplots_adjust(left=0.07, right=0.97, top=0.88, bottom=0.12, wspace=0.28)

    out_r = os.path.join(OUTPUT_DIR, "fd_region_pairs_band.png")
    w_in_r, _ = fig_r.get_size_inches()
    fig_r.savefig(out_r, dpi=EXPORT_WIDTH_2COL / w_in_r,
                  bbox_inches="tight", pad_inches=0.04, facecolor="white")
    print(f"\nSaved: {out_r}")
    plt.close(fig_r)


# ── Figure R2: Force-pair level mean ± 95% CI  (4-row × 2-col grid) ──────────
def draw_region_pair_meanCI_figure():
    """4 rows (A-C, A-D, F-C, F-D) × 2 cols (Low/High band). mean ± 95% CI."""
    ATD.apply_plot_style()
    sns.set_theme(style="white")

    n_comps = len(REGION_PAIRS)
    fig_m, axes_m = plt.subplots(
        n_comps, 2, figsize=(10.0, n_comps * 3.2), facecolor="white",
        sharex="col", sharey=True,
    )

    print("\n[FD — Region pairs: force-pair mean ± 95% CI | LME per pair]")

    for row, (rp, pl) in enumerate(zip(REGION_PAIRS, PAIR_LABELS)):
        r1, r2 = rp

        for col, band_cfg in enumerate(BANDS):
            ax = axes_m[row, col]
            band_name = band_cfg["name"]

            band_df   = subj_region_pair[subj_region_pair["band"] == band_name]
            pairs     = sorted(band_df["pair_label"].unique(),
                               key=lambda s: float(s.split("–")[1]))
            n_pairs   = len(pairs)
            x_centers = np.arange(n_pairs)

            y_max_used = 0.0

            for xi, pair in enumerate(pairs):
                pair_df = band_df[band_df["pair_label"] == pair]
                lme_r   = lme_two_regions(pair_df, r1, r2)

                ci_tops = []
                for gi, r in enumerate([r1, r2]):
                    dx   = (gi - 0.5) * (BOX_W * 2 + GAP)
                    xpos = xi + dx
                    data = pair_df[pair_df["Region"] == r]["accuracy"].dropna().values
                    if len(data) == 0:
                        continue
                    n    = len(data)
                    mean = float(np.mean(data))
                    ci   = CI95_MULTIPLIER * float(np.std(data, ddof=1) / np.sqrt(n))
                    color = REGION_COLOR[r]
                    ax.plot([xpos, xpos], [mean - ci, mean + ci],
                            color=color, linewidth=ERR_LW, zorder=2, clip_on=False)
                    for cap_y in (mean - ci, mean + ci):
                        ax.plot([xpos - CAP_W, xpos + CAP_W], [cap_y, cap_y],
                                color=color, linewidth=ERR_LW, zorder=2, clip_on=False)
                    ax.scatter([xpos], [mean],
                               c=[color], s=DOT_SIZE ** 2, linewidths=DOT_LW,
                               zorder=4, clip_on=False)
                    ci_tops.append(float(mean + ci))
                    y_max_used = max(y_max_used, mean + ci)

                if lme_r and ci_tops:
                    y_brk     = max(ci_tops) + 2.5
                    y_top_brk = y_brk + 0.5
                    x1 = xi + (0 - 0.5) * (BOX_W * 2 + GAP)
                    x2 = xi + (1 - 0.5) * (BOX_W * 2 + GAP)
                    ax.plot([x1, x1, x2, x2],
                            [y_brk, y_top_brk, y_top_brk, y_brk],
                            color=ACCENT_RED, linewidth=0.75, clip_on=False, zorder=5)
                    p_txt = f"{star(lme_r['p'])}  p={lme_r['p']:.3f}"
                    ax.text((x1 + x2) / 2, y_top_brk + 0.6, p_txt,
                            ha="center", va="bottom",
                            fontsize=max(7, FONT_ANNOT - 2),
                            color=ACCENT_RED, fontweight="bold",
                            clip_on=False, zorder=6)

            ax.axhline(JND_PCT, color=CRITERION_COLOR, linestyle="--",
                       linewidth=1.0, alpha=0.85, zorder=REF_LINE_ZORDER)

            if row == 0:
                ax.set_title(band_cfg["title"], fontsize=FONT_LABEL,
                             fontweight="bold", pad=6)
            ax.set_xticks(x_centers)
            if row == n_comps - 1:
                ax.set_xticklabels([f"{p} g" for p in pairs],
                                   fontsize=FONT_TICK - 2)
            else:
                ax.set_xticklabels([""] * n_pairs)
            ax.tick_params(axis="x", length=0)
            ax.set_yticks(Y_TICKS)
            ax.yaxis.set_major_locator(FixedLocator(Y_TICKS))
            ax.tick_params(axis="y", labelsize=FONT_TICK)
            y_top = min(YLIM_TOP_CAP, y_max_used + 18)
            ax.set_ylim(YLIM_BOT, y_top)
            ax.spines["left"].set_bounds(YLIM_BOT, 100)
            sns.despine(ax=ax)

            if col == 0:
                ax.set_ylabel(pl, fontsize=FONT_LABEL, labelpad=4)

            x_trans = ax.get_xaxis_transform()
            for xi in range(n_pairs):
                ax.plot([xi, xi], [0, TICK_LEN], color=BLACK, linewidth=1.0,
                        solid_capstyle="butt", transform=x_trans,
                        clip_on=False, zorder=6)
            y_trans = ax.get_yaxis_transform()
            y_lo2, y_hi2 = ax.get_ylim()
            for y in Y_TICKS:
                if y_lo2 - 1e-9 <= y <= y_hi2 + 1e-9:
                    ax.plot([0, TICK_LEN], [y, y], color=BLACK, linewidth=1.0,
                            solid_capstyle="butt", transform=y_trans,
                            clip_on=False, zorder=6)

    leg_m = [
        plt.Line2D([0], [0], color=OFF_NAIL_COLOR, marker="o",
                   markersize=DOT_SIZE * 0.7, linewidth=ERR_LW,
                   label="Off-nail regions (A, F)"),
        plt.Line2D([0], [0], color=ON_NAIL_COLOR, marker="o",
                   markersize=DOT_SIZE * 0.7, linewidth=ERR_LW,
                   label="On-nail regions (C, D)"),
    ]
    add_legend_outside(fig_m, axes_m[0, 0], leg_m, ncol=2,
                       top=FIG_LEGEND_TOP, bottom=FIG_LEGEND_BOTTOM,
                       left=0.08, right=0.97,
                       above_axes=ATD.FIG_LEGEND_ABOVE_AXES)

    fig_m.subplots_adjust(left=0.08, right=0.97, top=0.95, bottom=0.06,
                          wspace=0.22, hspace=0.15)
    fig_m.text(0.5, 0.01, "Force pair (g)", ha="center", fontsize=FONT_LABEL)

    out_m = os.path.join(OUTPUT_DIR, "fd_region_pairs_meanCI.png")
    w_in_m, _ = fig_m.get_size_inches()
    fig_m.savefig(out_m, dpi=EXPORT_WIDTH_2COL / w_in_m,
                  bbox_inches="tight", pad_inches=0.04, facecolor="white")
    print(f"\nSaved: {out_m}")
    plt.close(fig_m)


draw_region_pair_band_figure()
draw_region_pair_meanCI_figure()


# =============================================================================
#  Figure R3 — Trial-level binary scatter (GEE approach)
#  Each trial = one dot at 0% or 100%. 25 subjects × 2 trials = 50 dots/region.
#  Transparency + jitter reveals density (more dots at top → higher accuracy).
#  Horizontal bar = mean. Bracket = GEE (Binomial, Exchangeable) p-value.
#  Layout: 4 rows (A-C, A-D, F-C, F-D) × 2 cols (Low/High band)
# =============================================================================

# Trial-level data for regions of interest (binary 0/1 → multiply by 100 for %)
_df_trial = df_raw[df_raw["Region"].isin(["A", "C", "D", "F"])].copy()
_df_trial["correct_pct"] = _df_trial["correct"] * 100.0   # 0.0 or 100.0

TRIAL_ALPHA  = 0.25   # individual dot transparency (many overlap → density visible)
TRIAL_SIZE   = 4.0    # dot size (pt)
MEAN_LW      = 2.5    # mean bar linewidth
MEAN_W       = 0.18   # half-width of mean bar in x-data units
JITTER_W_T   = 0.14   # jitter width for trial-level dots


def gee_two_regions_trial(df_in, r1, r2):
    """GEE on raw binary trial data: correct ~ region, clusters=Subject."""
    sub = df_in[df_in["Region"].isin([r1, r2])].dropna(
        subset=["Subject", "Region", "correct"]
    ).copy()
    sub["reg_bin"] = (sub["Region"] == r2).astype(int)
    if sub["Subject"].nunique() < 2 or sub["Region"].nunique() < 2:
        return None
    try:
        X = sm.add_constant(sub["reg_bin"])
        model = GEE(
            sub["correct"], X,
            groups=sub["Subject"],
            family=Binomial(),
            cov_struct=Exchangeable(),
        )
        res = model.fit()
        p    = float(res.pvalues["reg_bin"])
        coef = float(res.params["reg_bin"])
        ci   = res.conf_int()
        return {"coef": coef,
                "ci_lo": float(ci.loc["reg_bin", 0]),
                "ci_hi": float(ci.loc["reg_bin", 1]),
                "p": p}
    except Exception as e:
        print(f"  GEE-trial failed: {e}")
        return None


def draw_region_pair_trial_figure():
    ATD.apply_plot_style()
    sns.set_theme(style="white")

    n_comps = len(REGION_PAIRS)
    fig_t, axes_t = plt.subplots(
        n_comps, 2, figsize=(10.0, n_comps * 3.2), facecolor="white",
        sharex="col", sharey=True,
    )
    rng_t = np.random.default_rng(42)

    print("\n[FD — Region pairs: trial-level binary scatter | GEE (Binomial)]")

    for row, (rp, pl) in enumerate(zip(REGION_PAIRS, PAIR_LABELS)):
        r1, r2 = rp

        for col, band_cfg in enumerate(BANDS):
            ax = axes_t[row, col]
            band_name = band_cfg["name"]

            band_trial = _df_trial[_df_trial["band"] == band_name]
            pairs      = sorted(
                band_trial["pair_label"].unique(),
                key=lambda s: float(s.split("–")[1])
            )
            n_pairs   = len(pairs)
            x_centers = np.arange(n_pairs)

            y_max_used = 0.0

            for xi, pair in enumerate(pairs):
                pair_trial = band_trial[band_trial["pair_label"] == pair]
                gee_res    = gee_two_regions_trial(pair_trial, r1, r2)

                mean_tops = []
                for gi, r in enumerate([r1, r2]):
                    dx    = (gi - 0.5) * (BOX_W * 2 + GAP)
                    xpos  = xi + dx
                    color = REGION_COLOR[r]

                    trials = pair_trial[pair_trial["Region"] == r]["correct_pct"].dropna().values
                    if len(trials) == 0:
                        continue

                    jit = rng_t.uniform(-JITTER_W_T, JITTER_W_T, size=len(trials))
                    ax.scatter(
                        xpos + jit, trials,
                        c=[color] * len(trials),
                        s=TRIAL_SIZE ** 2,
                        alpha=TRIAL_ALPHA,
                        linewidths=0,
                        zorder=3, clip_on=False,
                    )

                    mean_val = float(np.mean(trials))
                    ax.plot(
                        [xpos - MEAN_W, xpos + MEAN_W],
                        [mean_val, mean_val],
                        color=color, linewidth=MEAN_LW, zorder=5, clip_on=False,
                        solid_capstyle="butt",
                    )
                    mean_tops.append(mean_val)
                    y_max_used = max(y_max_used, 100.0)

                if gee_res and mean_tops:
                    y_brk     = 103
                    y_top_brk = y_brk + 0.5
                    x1 = xi + (0 - 0.5) * (BOX_W * 2 + GAP)
                    x2 = xi + (1 - 0.5) * (BOX_W * 2 + GAP)
                    ax.plot([x1, x1, x2, x2],
                            [y_brk, y_top_brk, y_top_brk, y_brk],
                            color=ACCENT_RED, linewidth=0.75, clip_on=False, zorder=6)
                    p_txt = f"{star(gee_res['p'])}  p={gee_res['p']:.3f}"
                    ax.text((x1 + x2) / 2, y_top_brk + 0.6, p_txt,
                            ha="center", va="bottom",
                            fontsize=max(7, FONT_ANNOT - 2),
                            color=ACCENT_RED, fontweight="bold",
                            clip_on=False, zorder=7)
                    if gee_res:
                        print(f"  [{band_name}] {pl} | {pair} g: "
                              f"GEE p={gee_res['p']:.4f}  {star(gee_res['p'])}")

            ax.axhline(JND_PCT, color=CRITERION_COLOR, linestyle="--",
                       linewidth=1.0, alpha=0.85, zorder=REF_LINE_ZORDER)

            if row == 0:
                ax.set_title(band_cfg["title"], fontsize=FONT_LABEL,
                             fontweight="bold", pad=6)
            ax.set_xticks(x_centers)
            if row == n_comps - 1:
                ax.set_xticklabels([f"{p} g" for p in pairs],
                                   fontsize=FONT_TICK - 2)
            else:
                ax.set_xticklabels([""] * n_pairs)
            ax.tick_params(axis="x", length=0)
            ax.set_yticks(Y_TICKS)
            ax.yaxis.set_major_locator(FixedLocator(Y_TICKS))
            ax.tick_params(axis="y", labelsize=FONT_TICK)
            ax.set_ylim(YLIM_BOT, min(YLIM_TOP_CAP, 120))
            ax.spines["left"].set_bounds(YLIM_BOT, 100)
            sns.despine(ax=ax)

            if col == 0:
                ax.set_ylabel(pl, fontsize=FONT_LABEL, labelpad=4)

            x_trans = ax.get_xaxis_transform()
            for xi in range(n_pairs):
                ax.plot([xi, xi], [0, TICK_LEN], color=BLACK, linewidth=1.0,
                        solid_capstyle="butt", transform=x_trans,
                        clip_on=False, zorder=6)
            y_trans = ax.get_yaxis_transform()
            y_lo2, y_hi2 = ax.get_ylim()
            for y in Y_TICKS:
                if y_lo2 - 1e-9 <= y <= y_hi2 + 1e-9:
                    ax.plot([0, TICK_LEN], [y, y], color=BLACK, linewidth=1.0,
                            solid_capstyle="butt", transform=y_trans,
                            clip_on=False, zorder=6)

    # legend: scatter dot + mean bar
    leg_t = [
        plt.Line2D([0], [0], color=OFF_NAIL_COLOR, marker="o",
                   markersize=TRIAL_SIZE * 0.9, linewidth=MEAN_LW,
                   alpha=1.0, label="Off-nail regions (A, F)"),
        plt.Line2D([0], [0], color=ON_NAIL_COLOR, marker="o",
                   markersize=TRIAL_SIZE * 0.9, linewidth=MEAN_LW,
                   alpha=1.0, label="On-nail regions (C, D)"),
    ]
    add_legend_outside(fig_t, axes_t[0, 0], leg_t, ncol=2,
                       top=FIG_LEGEND_TOP, bottom=FIG_LEGEND_BOTTOM,
                       left=0.08, right=0.97,
                       above_axes=ATD.FIG_LEGEND_ABOVE_AXES)

    fig_t.subplots_adjust(left=0.08, right=0.97, top=0.95, bottom=0.06,
                          wspace=0.22, hspace=0.15)
    fig_t.text(0.5, 0.01, "Force pair (g)", ha="center", fontsize=FONT_LABEL)

    out_t = os.path.join(OUTPUT_DIR, "fd_region_pairs_trial.png")
    w_in_t, _ = fig_t.get_size_inches()
    fig_t.savefig(out_t, dpi=EXPORT_WIDTH_2COL / w_in_t,
                  bbox_inches="tight", pad_inches=0.04, facecolor="white")
    print(f"\nSaved: {out_t}")
    plt.close(fig_t)


draw_region_pair_trial_figure()


# =============================================================================
#  A+B+C vs D+E+F comparison
#  Band-level (R_band) + Force-pair mean ± 95% CI (R_CI)
#
#  Granularity improvement:
#    Force-pair level: 3 regions × 2 trials = 6 trials/subject → 7 discrete values
#    Band level      : 3 × 4(Low)/3(High) pairs × 2 = 24/18 trials → 25/19 values
# =============================================================================

ABC_COLOR = "#5B7FA6"
DEF_COLOR = "#8B6BAE"

GROUP2_ORDER   = ["A+B+C", "D+E+F"]
GROUP2_PALETTE = {"A+B+C": ABC_COLOR, "D+E+F": DEF_COLOR}

ABC_DEF_MAP = {r: "A+B+C" for r in ["A", "B", "C"]}
ABC_DEF_MAP.update({r: "D+E+F" for r in ["D", "E", "F"]})

_df_all6 = df_raw[df_raw["Region"].isin(["A", "B", "C", "D", "E", "F"])].copy()
_df_all6["Group2"] = _df_all6["Region"].map(ABC_DEF_MAP)

subj_abc_band = (
    _df_all6.groupby(["Subject", "band", "Group2"], as_index=False)
    ["correct"].mean()
    .rename(columns={"correct": "accuracy"})
)
subj_abc_band["accuracy"] *= 100

subj_abc_pair = (
    _df_all6.groupby(["Subject", "band", "pair_label", "Group2"], as_index=False)
    ["correct"].mean()
    .rename(columns={"correct": "accuracy"})
)
subj_abc_pair["accuracy"] *= 100


def lme_abc_def(df_in, score_col="accuracy"):
    sub = df_in.dropna(subset=["Subject", "Group2", score_col])
    if sub["Subject"].nunique() < 2 or sub["Group2"].nunique() < 2:
        return None
    formula = f"{score_col} ~ C(Group2, Treatment(reference='A+B+C'))"
    try:
        res = smf.mixedlm(formula, sub, groups=sub["Subject"]).fit(reml=True)
        col = "C(Group2, Treatment(reference='A+B+C'))[T.D+E+F]"
        if col not in res.params.index:
            return None
        ci = res.conf_int().loc[col]
        return {"coef": float(res.params[col]),
                "ci_lo": float(ci[0]), "ci_hi": float(ci[1]),
                "p": float(res.pvalues[col])}
    except Exception:
        return None


def _save_600dpi(fig, fname):
    import io as _io_s; from PIL import Image as _Img_s
    _buf = _io_s.BytesIO()
    fig.savefig(_buf, format="png", dpi=600,
                bbox_inches="tight", pad_inches=0.04, facecolor="white")
    _buf.seek(0)
    _m = _Img_s.open(_buf).convert("RGB")
    _h = round(EXPORT_WIDTH_2COL * _m.height / _m.width)
    _m.resize((EXPORT_WIDTH_2COL, _h), _Img_s.Resampling.LANCZOS).save(fname)
    print(f"\nSaved: {fname}  ({EXPORT_WIDTH_2COL}×{_h} px @ 600 dpi)")


# ── Band-level box + scatter ──────────────────────────────────────────────────
def draw_abc_def_band_figure():
    ATD.apply_plot_style()
    sns.set_theme(style="white")

    fig_ab, ax_ab = plt.subplots(1, 1, figsize=(5.5, 4.5), facecolor="white")
    rng_ab = np.random.default_rng(42)
    BOX_W_AB = 0.28
    GAP_AB   = 0.10

    print("\n[FD — A+B+C vs D+E+F: band-level box+scatter | LME]")
    band_tops_ab = []

    for xi, (bname, blabel) in enumerate(zip(BAND_ORDER, BAND_LABELS)):
        band_df = subj_abc_band[subj_abc_band["band"] == bname]
        lme_res = lme_abc_def(band_df)
        if lme_res:
            print(f"  {bname} band: Δ={lme_res['coef']:.1f}% "
                  f"[{lme_res['ci_lo']:.1f}, {lme_res['ci_hi']:.1f}], "
                  f"p={lme_res['p']:.4f}  {star(lme_res['p'])}")

        whisker_tops_ab = []
        for gi, grp in enumerate(GROUP2_ORDER):
            dx   = (gi - 0.5) * (BOX_W_AB * 2 + GAP_AB)
            xpos = xi + dx
            data = band_df[band_df["Group2"] == grp]["accuracy"].dropna().values
            if len(data) == 0:
                continue
            bp = ax_ab.boxplot(
                [data], positions=[xpos], widths=BOX_W_AB * 1.6,
                patch_artist=True, showfliers=False, zorder=2,
                whiskerprops=dict(color=BLACK, linewidth=BOX_LINEWIDTH),
                capprops=dict(color=BLACK, linewidth=CAP_LINEWIDTH),
                medianprops=dict(color=ACCENT_RED, linewidth=2.0),
                boxprops=dict(facecolor=pale_box_face(GROUP2_PALETTE[grp]),
                              edgecolor=BLACK, linewidth=BOX_LINEWIDTH),
            )
            w_top = max(w.get_ydata()[1] for w in bp["whiskers"])
            whisker_tops_ab.append(w_top)
            jit = rng_ab.uniform(-BOX_W_AB * 0.5, BOX_W_AB * 0.5, size=len(data))
            rgba = _hsb_scatter_rgba(GROUP2_PALETTE[grp])
            ax_ab.scatter(xpos + jit, data, c=[rgba] * len(data),
                          s=3.5 ** 2, linewidths=0, zorder=3, clip_on=False)

        band_top = max(whisker_tops_ab) if whisker_tops_ab else 80.0
        band_tops_ab.append(band_top)

        if lme_res:
            x1 = xi + (0 - 0.5) * (BOX_W_AB * 2 + GAP_AB)
            x2 = xi + (1 - 0.5) * (BOX_W_AB * 2 + GAP_AB)
            y_brk = band_top + 3
            y_top_brk = y_brk + 0.5
            ax_ab.plot([x1, x1, x2, x2], [y_brk, y_top_brk, y_top_brk, y_brk],
                       color=ACCENT_RED, linewidth=0.75, clip_on=False, zorder=5)
            p_txt = f"{star(lme_res['p'])}  p={lme_res['p']:.3f}"
            ax_ab.text((x1 + x2) / 2, y_top_brk + 0.6, p_txt,
                       ha="center", va="bottom", fontsize=max(8, FONT_ANNOT - 1),
                       color=ACCENT_RED, fontweight="bold", clip_on=False, zorder=6)

    ax_ab.axhline(JND_PCT, color=CRITERION_COLOR, linestyle="--",
                  linewidth=1.0, alpha=0.85, zorder=REF_LINE_ZORDER)
    ax_ab.set_xticks(range(len(BAND_ORDER)))
    ax_ab.set_xticklabels(BAND_LABELS, fontsize=FONT_TICK)
    ax_ab.set_yticks(Y_TICKS)
    ax_ab.yaxis.set_major_locator(FixedLocator(Y_TICKS))
    ax_ab.tick_params(axis="y", labelsize=FONT_TICK)
    ax_ab.tick_params(axis="x", length=0)
    y_top_ab = min(YLIM_TOP_CAP, max(band_tops_ab) + 18) if band_tops_ab else 120
    ax_ab.set_ylim(YLIM_BOT, y_top_ab)
    ax_ab.set_ylabel("Accuracy (%)", fontsize=FONT_LABEL)
    ax_ab.spines["left"].set_bounds(YLIM_BOT, 100)
    sns.despine(ax=ax_ab)

    x_trans = ax_ab.get_xaxis_transform()
    for xi in range(len(BAND_ORDER)):
        ax_ab.plot([xi, xi], [0, TICK_LEN], color=BLACK, linewidth=1.0,
                   solid_capstyle="butt", transform=x_trans, clip_on=False, zorder=6)
    y_trans = ax_ab.get_yaxis_transform()
    for y in Y_TICKS:
        if YLIM_BOT - 1e-9 <= y <= y_top_ab + 1e-9:
            ax_ab.plot([0, TICK_LEN], [y, y], color=BLACK, linewidth=1.0,
                       solid_capstyle="butt", transform=y_trans, clip_on=False, zorder=6)

    leg_ab = [
        mpatches.Patch(facecolor=pale_box_face(GROUP2_PALETTE[g]),
                       edgecolor=BLACK, linewidth=BOX_LINEWIDTH, label=g)
        for g in GROUP2_ORDER
    ]
    add_legend_outside(fig_ab, ax_ab, leg_ab, ncol=2,
                       top=FIG_LEGEND_TOP, bottom=FIG_LEGEND_BOTTOM,
                       left=0.10, right=0.95,
                       above_axes=ATD.FIG_LEGEND_ABOVE_AXES)
    fig_ab.subplots_adjust(left=0.10, right=0.95, top=0.88, bottom=0.12)

    _save_600dpi(fig_ab, os.path.join(OUTPUT_DIR, "fd_abc_vs_def_band.png"))
    plt.close(fig_ab)


# ── Force-pair mean ± 95% CI ──────────────────────────────────────────────────
def draw_abc_def_meanCI_figure():
    ATD.apply_plot_style()
    sns.set_theme(style="white")

    fig_ac, axes_ac = plt.subplots(1, 2, figsize=(10.0, 4.5), facecolor="white")
    print("\n[FD — A+B+C vs D+E+F: force-pair mean ± 95% CI | LME]")

    for ax, band_cfg in zip(axes_ac, BANDS):
        band_name  = band_cfg["name"]
        band_df    = subj_abc_pair[subj_abc_pair["band"] == band_name]
        pairs      = sorted(band_df["pair_label"].unique(),
                            key=lambda s: float(s.split("–")[1]))
        x_centers  = np.arange(len(pairs))
        y_max_used = 0.0

        for xi, pair in enumerate(pairs):
            pair_df = band_df[band_df["pair_label"] == pair]
            lme_res = lme_abc_def(pair_df)
            ci_tops = []

            for gi, grp in enumerate(GROUP2_ORDER):
                dx   = (gi - 0.5) * (BOX_W * 2 + GAP)
                xpos = xi + dx
                data = pair_df[pair_df["Group2"] == grp]["accuracy"].dropna().values
                if len(data) == 0:
                    continue
                n    = len(data)
                mean = float(np.mean(data))
                ci   = CI95_MULTIPLIER * float(np.std(data, ddof=1) / np.sqrt(n))
                color = GROUP2_PALETTE[grp]
                ax.plot([xpos, xpos], [mean - ci, mean + ci],
                        color=color, linewidth=ERR_LW, zorder=2, clip_on=False)
                for cap_y in (mean - ci, mean + ci):
                    ax.plot([xpos - CAP_W, xpos + CAP_W], [cap_y, cap_y],
                            color=color, linewidth=ERR_LW, zorder=2, clip_on=False)
                ax.scatter([xpos], [mean], c=[color], s=DOT_SIZE ** 2,
                           linewidths=DOT_LW, zorder=4, clip_on=False)
                ci_tops.append(float(mean + ci))
                y_max_used = max(y_max_used, mean + ci)

            if lme_res and ci_tops:
                y_brk = max(ci_tops) + 2.5
                y_top_brk = y_brk + 0.5
                x1 = xi + (0 - 0.5) * (BOX_W * 2 + GAP)
                x2 = xi + (1 - 0.5) * (BOX_W * 2 + GAP)
                ax.plot([x1, x1, x2, x2], [y_brk, y_top_brk, y_top_brk, y_brk],
                        color=ACCENT_RED, linewidth=0.75, clip_on=False, zorder=5)
                p_txt = f"{star(lme_res['p'])}  p={lme_res['p']:.3f}"
                ax.text((x1 + x2) / 2, y_top_brk + 0.6, p_txt,
                        ha="center", va="bottom", fontsize=max(8, FONT_ANNOT - 1),
                        color=ACCENT_RED, fontweight="bold", clip_on=False, zorder=6)
                print(f"  [{band_name}] {pair} g: Δ={lme_res['coef']:.1f}%  "
                      f"p={lme_res['p']:.4f}  {star(lme_res['p'])}")

        ax.axhline(JND_PCT, color=CRITERION_COLOR, linestyle="--",
                   linewidth=1.0, alpha=0.85, zorder=REF_LINE_ZORDER)
        ax.set_title(band_cfg["title"], fontsize=FONT_LABEL, fontweight="bold", pad=6)
        ax.set_xticks(x_centers)
        ax.set_xticklabels([f"{p} g" for p in pairs], fontsize=FONT_TICK - 2)
        ax.set_yticks(Y_TICKS)
        ax.yaxis.set_major_locator(FixedLocator(Y_TICKS))
        ax.tick_params(axis="y", labelsize=FONT_TICK)
        ax.tick_params(axis="x", length=0)
        y_top = min(YLIM_TOP_CAP, y_max_used + 18)
        ax.set_ylim(YLIM_BOT, y_top)
        ax.spines["left"].set_bounds(YLIM_BOT, 100)
        sns.despine(ax=ax)

        x_trans = ax.get_xaxis_transform()
        for xi in range(len(pairs)):
            ax.plot([xi, xi], [0, TICK_LEN], color=BLACK, linewidth=1.0,
                    solid_capstyle="butt", transform=x_trans, clip_on=False, zorder=6)
        y_trans = ax.get_yaxis_transform()
        y_lo, y_hi = ax.get_ylim()
        for y in Y_TICKS:
            if y_lo - 1e-9 <= y <= y_hi + 1e-9:
                ax.plot([0, TICK_LEN], [y, y], color=BLACK, linewidth=1.0,
                        solid_capstyle="butt", transform=y_trans, clip_on=False, zorder=6)

    axes_ac[0].set_ylabel("Accuracy (%)", fontsize=FONT_LABEL)
    axes_ac[1].set_ylabel("")

    leg_ac = [
        plt.Line2D([0], [0], color=GROUP2_PALETTE[g], marker="o",
                   markersize=DOT_SIZE * 0.7, linewidth=ERR_LW, label=g)
        for g in GROUP2_ORDER
    ]
    add_legend_outside(fig_ac, axes_ac[0], leg_ac, ncol=2,
                       top=FIG_LEGEND_TOP, bottom=FIG_LEGEND_BOTTOM,
                       left=0.07, right=0.97,
                       above_axes=ATD.FIG_LEGEND_ABOVE_AXES)
    fig_ac.subplots_adjust(left=0.07, right=0.97, top=0.88, bottom=0.12, wspace=0.28)

    _save_600dpi(fig_ac, os.path.join(OUTPUT_DIR, "fd_abc_vs_def_meanCI.png"))
    plt.close(fig_ac)


draw_abc_def_band_figure()
draw_abc_def_meanCI_figure()


# =============================================================================
#  d' (d-prime) analysis — On-nail (C+D) vs Off-nail (A+F)
#  d' = √2 × Φ⁻¹(proportion correct)   [2AFC formula]
#
#  Ceiling/floor correction (loglinear):
#    p_adj = (n_correct + 0.5) / (n_trials + 1)
#  Applied per subject so d' is always finite.
#
#  Two figures:
#    (D1) Band-level d' box + scatter  (most continuous, 16/12 trials per subject)
#    (D2) Force-pair d' mean ± 95% CI  (per force pair, 4 trials/subject → 5 values)
#
#  Criterion line: d' = √2 × Φ⁻¹(0.75) ≈ 0.953 (equivalent to 75% accuracy)
# =============================================================================
from scipy.stats import norm as _norm

DPRIME_CRITERION = np.sqrt(2) * _norm.ppf(0.75)   # ≈ 0.9535

def to_dprime_loglinear(n_correct, n_trials):
    """Loglinear correction → d' for 2AFC."""
    p_adj = (n_correct + 0.5) / (n_trials + 1)
    return float(np.sqrt(2) * _norm.ppf(p_adj))


# ── Build d' datasets ─────────────────────────────────────────────────────────

# Band-level: count correct & total per subject × band × group
_band_counts = (
    df_cd_af.groupby(["Subject", "band", "Group"])["correct"]
    .agg(n_correct="sum", n_trials="count")
    .reset_index()
)
_band_counts["dprime"] = _band_counts.apply(
    lambda r: to_dprime_loglinear(r["n_correct"], r["n_trials"]), axis=1
)

# Force-pair level: count per subject × band × pair_label × group
_pair_counts = (
    df_cd_af.groupby(["Subject", "band", "pair_label", "Group"])["correct"]
    .agg(n_correct="sum", n_trials="count")
    .reset_index()
)
_pair_counts["dprime"] = _pair_counts.apply(
    lambda r: to_dprime_loglinear(r["n_correct"], r["n_trials"]), axis=1
)

print("\n[d' summary — band level]")
for band in ["Low", "High"]:
    for grp in GROUP_ORDER:
        vals = _band_counts[
            (_band_counts["band"] == band) & (_band_counts["Group"] == grp)
        ]["dprime"].values
        print(f"  {band} {grp}: mean={np.mean(vals):.3f}  sd={np.std(vals,ddof=1):.3f}  "
              f"n={len(vals)}  range=[{np.min(vals):.2f}, {np.max(vals):.2f}]")


def lme_dprime(df_in):
    """LME on d' scores: dprime ~ Group, random intercept for Subject."""
    sub = df_in.dropna(subset=["Subject", "Group", "dprime"])
    if sub["Subject"].nunique() < 2 or sub["Group"].nunique() < 2:
        return None
    formula = "dprime ~ C(Group, Treatment(reference='Off-nail'))"
    try:
        res = smf.mixedlm(formula, sub, groups=sub["Subject"]).fit(reml=True)
        col = "C(Group, Treatment(reference='Off-nail'))[T.On-nail]"
        if col not in res.params.index:
            return None
        ci = res.conf_int().loc[col]
        return {"coef": float(res.params[col]),
                "ci_lo": float(ci[0]), "ci_hi": float(ci[1]),
                "p": float(res.pvalues[col])}
    except Exception:
        return None


# ── Figure D1: Band-level d' box + scatter ────────────────────────────────────
def draw_dprime_band_figure():
    ATD.apply_plot_style()
    sns.set_theme(style="white")

    fig_d1, ax_d1 = plt.subplots(1, 1, figsize=(5.5, 4.5), facecolor="white")
    rng_d1 = np.random.default_rng(42)
    BOX_W_D = 0.28
    GAP_D   = 0.10

    print("\n[FD — d' band-level box+scatter | LME]")

    y_ticks_d = [-1, 0, 1, 2, 3]
    ylim_bot_d = -1.5
    ylim_top_d = 3.5
    band_tops_d = []

    for xi, (bname, blabel) in enumerate(zip(BAND_ORDER, BAND_LABELS)):
        band_df = _band_counts[_band_counts["band"] == bname]
        lme_res = lme_dprime(band_df)
        if lme_res:
            print(f"  {bname} band: Δd'={lme_res['coef']:.3f} "
                  f"[{lme_res['ci_lo']:.3f}, {lme_res['ci_hi']:.3f}], "
                  f"p={lme_res['p']:.4f}  {star(lme_res['p'])}")

        whisker_tops_d = []
        for gi, grp in enumerate(GROUP_ORDER):
            dx   = (gi - 0.5) * (BOX_W_D * 2 + GAP_D)
            xpos = xi + dx
            data = band_df[band_df["Group"] == grp]["dprime"].dropna().values
            if len(data) == 0:
                continue
            bp = ax_d1.boxplot(
                [data], positions=[xpos], widths=BOX_W_D * 1.6,
                patch_artist=True, showfliers=False, zorder=2,
                whiskerprops=dict(color=BLACK, linewidth=BOX_LINEWIDTH),
                capprops=dict(color=BLACK, linewidth=CAP_LINEWIDTH),
                medianprops=dict(color=ACCENT_RED, linewidth=2.0),
                boxprops=dict(facecolor=pale_box_face(GROUP_PALETTE[grp]),
                              edgecolor=BLACK, linewidth=BOX_LINEWIDTH),
            )
            w_top = max(w.get_ydata()[1] for w in bp["whiskers"])
            whisker_tops_d.append(w_top)
            jit = rng_d1.uniform(-BOX_W_D * 0.5, BOX_W_D * 0.5, size=len(data))
            rgba = _hsb_scatter_rgba(GROUP_PALETTE[grp])
            ax_d1.scatter(xpos + jit, data, c=[rgba] * len(data),
                          s=3.5 ** 2, linewidths=0, zorder=3, clip_on=False)

        band_top = max(whisker_tops_d) if whisker_tops_d else 1.5
        band_tops_d.append(band_top)

        if lme_res:
            x1 = xi + (0 - 0.5) * (BOX_W_D * 2 + GAP_D)
            x2 = xi + (1 - 0.5) * (BOX_W_D * 2 + GAP_D)
            y_brk = band_top + 0.08
            y_top_brk = y_brk + 0.04
            ax_d1.plot([x1, x1, x2, x2], [y_brk, y_top_brk, y_top_brk, y_brk],
                       color=ACCENT_RED, linewidth=0.75, clip_on=False, zorder=5)
            p_txt = f"{star(lme_res['p'])}  p={lme_res['p']:.3f}"
            ax_d1.text((x1 + x2) / 2, y_top_brk + 0.03, p_txt,
                       ha="center", va="bottom", fontsize=max(8, FONT_ANNOT - 1),
                       color=ACCENT_RED, fontweight="bold", clip_on=False, zorder=6)

    # criterion line at d' = 0.953 (JND)
    ax_d1.axhline(DPRIME_CRITERION, color=CRITERION_COLOR, linestyle="--",
                  linewidth=1.0, alpha=0.85, zorder=REF_LINE_ZORDER,
                  label=f"JND criterion (d'≈{DPRIME_CRITERION:.2f})")
    ax_d1.axhline(0, color=BLACK, linestyle=":", linewidth=0.8, alpha=0.4, zorder=1)

    ax_d1.set_xticks(range(len(BAND_ORDER)))
    ax_d1.set_xticklabels(BAND_LABELS, fontsize=FONT_TICK)
    ax_d1.set_yticks(y_ticks_d)
    ax_d1.tick_params(axis="y", labelsize=FONT_TICK)
    ax_d1.tick_params(axis="x", length=0)
    y_top_plot = min(ylim_top_d, max(band_tops_d) + 0.4) if band_tops_d else ylim_top_d
    ax_d1.set_ylim(ylim_bot_d, y_top_plot)
    ax_d1.set_ylabel("d' (sensitivity)", fontsize=FONT_LABEL)
    ax_d1.spines["left"].set_bounds(ylim_bot_d, max(y_ticks_d))
    sns.despine(ax=ax_d1)

    x_trans = ax_d1.get_xaxis_transform()
    for xi in range(len(BAND_ORDER)):
        ax_d1.plot([xi, xi], [0, TICK_LEN], color=BLACK, linewidth=1.0,
                   solid_capstyle="butt", transform=x_trans, clip_on=False, zorder=6)
    y_trans = ax_d1.get_yaxis_transform()
    for y in y_ticks_d:
        ax_d1.plot([0, TICK_LEN], [y, y], color=BLACK, linewidth=1.0,
                   solid_capstyle="butt", transform=y_trans, clip_on=False, zorder=6)

    leg_d1 = [
        mpatches.Patch(facecolor=pale_box_face(GROUP_PALETTE[g]),
                       edgecolor=BLACK, linewidth=BOX_LINEWIDTH, label=GROUP_LABELS[i])
        for i, g in enumerate(GROUP_ORDER)
    ]
    add_legend_outside(fig_d1, ax_d1, leg_d1, ncol=2,
                       top=FIG_LEGEND_TOP, bottom=FIG_LEGEND_BOTTOM,
                       left=0.10, right=0.95,
                       above_axes=ATD.FIG_LEGEND_ABOVE_AXES)
    fig_d1.subplots_adjust(left=0.10, right=0.95, top=0.88, bottom=0.12)

    _save_600dpi(fig_d1, os.path.join(OUTPUT_DIR, "fd_dprime_band.png"))
    plt.close(fig_d1)


# ── Figure D2: Force-pair d' mean ± 95% CI ───────────────────────────────────
def draw_dprime_meanCI_figure():
    ATD.apply_plot_style()
    sns.set_theme(style="white")

    fig_d2, axes_d2 = plt.subplots(1, 2, figsize=(10.0, 4.5), facecolor="white")
    print("\n[FD — d' force-pair mean ± 95% CI | LME]")

    y_ticks_d2 = [-1, 0, 1, 2, 3]
    ylim_bot_d2 = -1.5

    for ax, band_cfg in zip(axes_d2, BANDS):
        band_name  = band_cfg["name"]
        band_df    = _pair_counts[_pair_counts["band"] == band_name]
        pairs      = sorted(band_df["pair_label"].unique(),
                            key=lambda s: float(s.split("–")[1]))
        x_centers  = np.arange(len(pairs))
        y_max_used = 0.0

        for xi, pair in enumerate(pairs):
            pair_df = band_df[band_df["pair_label"] == pair]
            lme_res = lme_dprime(pair_df)
            ci_tops = []

            for gi, grp in enumerate(GROUP_ORDER):
                dx    = (gi - 0.5) * (BOX_W * 2 + GAP)
                xpos  = xi + dx
                data  = pair_df[pair_df["Group"] == grp]["dprime"].dropna().values
                if len(data) == 0:
                    continue
                n     = len(data)
                mean  = float(np.mean(data))
                ci    = CI95_MULTIPLIER * float(np.std(data, ddof=1) / np.sqrt(n))
                color = GROUP_PALETTE[grp]
                ax.plot([xpos, xpos], [mean - ci, mean + ci],
                        color=color, linewidth=ERR_LW, zorder=2, clip_on=False)
                for cap_y in (mean - ci, mean + ci):
                    ax.plot([xpos - CAP_W, xpos + CAP_W], [cap_y, cap_y],
                            color=color, linewidth=ERR_LW, zorder=2, clip_on=False)
                ax.scatter([xpos], [mean], c=[color], s=DOT_SIZE ** 2,
                           linewidths=DOT_LW, zorder=4, clip_on=False)
                ci_tops.append(float(mean + ci))
                y_max_used = max(y_max_used, mean + ci)

            if lme_res and ci_tops:
                y_brk = max(ci_tops) + 0.08
                y_top_brk = y_brk + 0.04
                x1 = xi + (0 - 0.5) * (BOX_W * 2 + GAP)
                x2 = xi + (1 - 0.5) * (BOX_W * 2 + GAP)
                ax.plot([x1, x1, x2, x2], [y_brk, y_top_brk, y_top_brk, y_brk],
                        color=ACCENT_RED, linewidth=0.75, clip_on=False, zorder=5)
                p_txt = f"{star(lme_res['p'])}  p={lme_res['p']:.3f}"
                ax.text((x1 + x2) / 2, y_top_brk + 0.03, p_txt,
                        ha="center", va="bottom", fontsize=max(8, FONT_ANNOT - 1),
                        color=ACCENT_RED, fontweight="bold", clip_on=False, zorder=6)
                print(f"  [{band_name}] {pair} g: Δd'={lme_res['coef']:.3f}  "
                      f"p={lme_res['p']:.4f}  {star(lme_res['p'])}")

        ax.axhline(DPRIME_CRITERION, color=CRITERION_COLOR, linestyle="--",
                   linewidth=1.0, alpha=0.85, zorder=REF_LINE_ZORDER)
        ax.axhline(0, color=BLACK, linestyle=":", linewidth=0.8, alpha=0.4, zorder=1)
        ax.set_title(band_cfg["title"], fontsize=FONT_LABEL, fontweight="bold", pad=6)
        ax.set_xticks(x_centers)
        ax.set_xticklabels([f"{p} g" for p in pairs], fontsize=FONT_TICK - 2)
        ax.set_yticks(y_ticks_d2)
        ax.tick_params(axis="y", labelsize=FONT_TICK)
        ax.tick_params(axis="x", length=0)
        y_top = min(3.5, y_max_used + 0.4)
        ax.set_ylim(ylim_bot_d2, y_top)
        ax.spines["left"].set_bounds(ylim_bot_d2, max(y_ticks_d2))
        sns.despine(ax=ax)

        x_trans = ax.get_xaxis_transform()
        for xi in range(len(pairs)):
            ax.plot([xi, xi], [0, TICK_LEN], color=BLACK, linewidth=1.0,
                    solid_capstyle="butt", transform=x_trans, clip_on=False, zorder=6)
        y_trans = ax.get_yaxis_transform()
        y_lo, y_hi = ax.get_ylim()
        for y in y_ticks_d2:
            if y_lo - 1e-9 <= y <= y_hi + 1e-9:
                ax.plot([0, TICK_LEN], [y, y], color=BLACK, linewidth=1.0,
                        solid_capstyle="butt", transform=y_trans, clip_on=False, zorder=6)

    axes_d2[0].set_ylabel("d' (sensitivity)", fontsize=FONT_LABEL)
    axes_d2[1].set_ylabel("")

    leg_d2 = [
        plt.Line2D([0], [0], color=GROUP_PALETTE[g], marker="o",
                   markersize=DOT_SIZE * 0.7, linewidth=ERR_LW, label=GROUP_LABELS[i])
        for i, g in enumerate(GROUP_ORDER)
    ]
    add_legend_outside(fig_d2, axes_d2[0], leg_d2, ncol=2,
                       top=FIG_LEGEND_TOP, bottom=FIG_LEGEND_BOTTOM,
                       left=0.07, right=0.97,
                       above_axes=ATD.FIG_LEGEND_ABOVE_AXES)
    fig_d2.subplots_adjust(left=0.07, right=0.97, top=0.88, bottom=0.12, wspace=0.28)

    _save_600dpi(fig_d2, os.path.join(OUTPUT_DIR, "fd_dprime_meanCI.png"))
    plt.close(fig_d2)


draw_dprime_band_figure()
draw_dprime_meanCI_figure()


# =============================================================================
#  B+C+D+E vs A+F comparison — trial-level pooling
#  On-nail  : B, C, D, E  (per subject per force: mean of all B+C+D+E trials)
#  Off-nail : A, F         (per subject per force: mean of all A+F trials)
#  Equal sample size n=25 per group (same as Approach 0).
#
#  Two figures:
#    (BF1) Band-level box + scatter
#    (BF2) Force-pair mean ± 95% CI
# =============================================================================

BCDE_AF_MAP = {
    "B": "On-nail", "C": "On-nail", "D": "On-nail", "E": "On-nail",
    "A": "Off-nail", "F": "Off-nail",
}
BCDE_AF_ORDER   = ["On-nail", "Off-nail"]
BCDE_AF_PALETTE = {"On-nail": ON_NAIL_COLOR, "Off-nail": OFF_NAIL_COLOR}
BCDE_AF_LABELS  = ["On-nail (B+C+D+E)", "Off-nail (A+F)"]

_df_bcde_af = df_raw[df_raw["Region"].isin(BCDE_AF_MAP)].copy()
_df_bcde_af["Group"] = _df_bcde_af["Region"].map(BCDE_AF_MAP)

# Band-level: per subject per band, mean of all trials within group
subj_bcde_band = (
    _df_bcde_af.groupby(["Subject", "band", "Group"], as_index=False)
    ["correct"].mean()
    .rename(columns={"correct": "accuracy"})
)
subj_bcde_band["accuracy"] *= 100

# Force-pair level: per subject per band per pair, mean of all trials within group
subj_bcde_pair = (
    _df_bcde_af.groupby(["Subject", "band", "pair_label", "Group"], as_index=False)
    ["correct"].mean()
    .rename(columns={"correct": "accuracy"})
)
subj_bcde_pair["accuracy"] *= 100


def lme_bcde_af(df_in, score_col="accuracy"):
    sub = df_in.dropna(subset=["Subject", "Group", score_col])
    if sub["Subject"].nunique() < 2 or sub["Group"].nunique() < 2:
        return None
    formula = f"{score_col} ~ C(Group, Treatment(reference='Off-nail'))"
    try:
        res = smf.mixedlm(formula, sub, groups=sub["Subject"]).fit(reml=True)
        col = "C(Group, Treatment(reference='Off-nail'))[T.On-nail]"
        if col not in res.params.index:
            return None
        ci = res.conf_int().loc[col]
        return {"coef": float(res.params[col]),
                "ci_lo": float(ci[0]), "ci_hi": float(ci[1]),
                "p": float(res.pvalues[col])}
    except Exception:
        return None


def draw_bcde_af_band_figure():
    ATD.apply_plot_style()
    sns.set_theme(style="white")

    fig_bf, axes_bf = plt.subplots(1, 2, figsize=(10.0, 4.5), facecolor="white")
    rng_bf = np.random.default_rng(42)

    print("\n[FD — On-nail(B+C+D+E) vs Off-nail(A+F): band-level box+scatter | LME]")

    for ax, band_cfg in zip(axes_bf, BANDS):
        band_name  = band_cfg["name"]
        band_df    = subj_bcde_band[subj_bcde_band["band"] == band_name]
        pairs      = sorted(
            subj_bcde_pair[subj_bcde_pair["band"] == band_name]["pair_label"].unique(),
            key=lambda s: float(s.split("–")[1])
        )
        n_pairs   = len(pairs)
        x_centers = np.arange(n_pairs)

        print(f"\n  [{band_name} band]")
        band_tops_bf = []

        for xi, pair in enumerate(pairs):
            pair_df  = subj_bcde_pair[
                (subj_bcde_pair["band"] == band_name) &
                (subj_bcde_pair["pair_label"] == pair)
            ]
            lme_res = lme_bcde_af(pair_df)
            if lme_res:
                print(f"    {pair} g  On- vs Off-nail: "
                      f"Δ={lme_res['coef']:.2f} "
                      f"[{lme_res['ci_lo']:.2f}, {lme_res['ci_hi']:.2f}], "
                      f"p={lme_res['p']:.4f}  {star(lme_res['p'])}")
            else:
                print(f"    {pair} g  stat failed")

            whisker_tops_bf = []
            for gi, grp in enumerate(BCDE_AF_ORDER):
                dx   = (gi - 0.5) * (BOX_W * 2 + GAP)
                xpos = xi + dx
                data = pair_df[pair_df["Group"] == grp]["accuracy"].dropna().values
                if len(data) == 0:
                    continue
                bp = ax.boxplot(
                    [data], positions=[xpos], widths=BOX_W * 1.6,
                    patch_artist=True, showfliers=False, zorder=2,
                    whiskerprops=dict(color=BLACK, linewidth=BOX_LINEWIDTH),
                    capprops=dict(color=BLACK, linewidth=CAP_LINEWIDTH),
                    medianprops=dict(color=ACCENT_RED, linewidth=2.0),
                    boxprops=dict(facecolor=pale_box_face(BCDE_AF_PALETTE[grp]),
                                  edgecolor=BLACK, linewidth=BOX_LINEWIDTH),
                )
                w_top = max(w.get_ydata()[1] for w in bp["whiskers"])
                whisker_tops_bf.append(w_top)
                jitter = rng_bf.uniform(-BOX_W * 0.5, BOX_W * 0.5, size=len(data))
                rgba   = _hsb_scatter_rgba(BCDE_AF_PALETTE[grp])
                ax.scatter(xpos + jitter, data, c=[rgba] * len(data),
                           s=3.5 ** 2, linewidths=0, zorder=3, clip_on=False)

            pair_top = max(whisker_tops_bf) if whisker_tops_bf else 80.0
            band_tops_bf.append(pair_top)

            if lme_res:
                x_on  = xi + (0 - 0.5) * (BOX_W * 2 + GAP)
                x_off = xi + (1 - 0.5) * (BOX_W * 2 + GAP)
                y_brk = pair_top + 3
                y_top_brk = y_brk + 0.5
                ax.plot([x_on, x_on, x_off, x_off],
                        [y_brk, y_top_brk, y_top_brk, y_brk],
                        color=ACCENT_RED, linewidth=0.75, clip_on=False, zorder=5)
                p_txt = f"{star(lme_res['p'])}  p={lme_res['p']:.3f}"
                ax.text((x_on + x_off) / 2, y_top_brk + 0.6, p_txt,
                        ha="center", va="bottom", fontsize=max(8, FONT_ANNOT - 1),
                        color=ACCENT_RED, fontweight="bold", clip_on=False, zorder=6)

        ax.axhline(JND_PCT, color=CRITERION_COLOR, linestyle="--",
                   linewidth=1.0, alpha=0.85, zorder=REF_LINE_ZORDER)
        ax.set_title(band_cfg["title"], fontsize=FONT_LABEL, fontweight="bold", pad=6)
        ax.set_xticks(x_centers)
        ax.set_xticklabels([f"{p} g" for p in pairs], fontsize=FONT_TICK - 2)
        ax.set_yticks(Y_TICKS)
        ax.yaxis.set_major_locator(FixedLocator(Y_TICKS))
        ax.tick_params(axis="y", labelsize=FONT_TICK)
        ax.tick_params(axis="x", length=0)
        y_top = min(YLIM_TOP_CAP, max(band_tops_bf) + 18) if band_tops_bf else 120
        ax.set_ylim(YLIM_BOT, y_top)
        ax.spines["left"].set_bounds(YLIM_BOT, 100)
        sns.despine(ax=ax)

        x_trans = ax.get_xaxis_transform()
        for xi_t in x_centers:
            ax.plot([xi_t, xi_t], [0, TICK_LEN], color=BLACK, linewidth=1.0,
                    solid_capstyle="butt", transform=x_trans, clip_on=False, zorder=6)
        y_trans = ax.get_yaxis_transform()
        y_lo, y_hi = ax.get_ylim()
        for y in Y_TICKS:
            if y_lo - 1e-9 <= y <= y_hi + 1e-9:
                ax.plot([0, TICK_LEN], [y, y], color=BLACK, linewidth=1.0,
                        solid_capstyle="butt", transform=y_trans, clip_on=False, zorder=6)

    axes_bf[0].set_ylabel("Accuracy (%)", fontsize=FONT_LABEL)
    axes_bf[1].set_ylabel("")

    leg_bf = [
        mpatches.Patch(facecolor=pale_box_face(BCDE_AF_PALETTE[g]),
                       edgecolor=BLACK, linewidth=BOX_LINEWIDTH, label=BCDE_AF_LABELS[i])
        for i, g in enumerate(BCDE_AF_ORDER)
    ]
    add_legend_outside(fig_bf, axes_bf[0], leg_bf, ncol=2,
                       top=FIG_LEGEND_TOP, bottom=FIG_LEGEND_BOTTOM,
                       left=0.07, right=0.97,
                       above_axes=ATD.FIG_LEGEND_ABOVE_AXES)
    fig_bf.subplots_adjust(left=0.07, right=0.97, top=0.88, bottom=0.12, wspace=0.28)

    _save_600dpi(fig_bf, os.path.join(OUTPUT_DIR, "fd_bcde_vs_af_band.png"))
    plt.close(fig_bf)


def draw_bcde_af_meanCI_figure():
    ATD.apply_plot_style()
    sns.set_theme(style="white")

    fig_bf2, axes_bf2 = plt.subplots(1, 2, figsize=(10.0, 4.5), facecolor="white")
    print("\n[FD — On-nail(B+C+D+E) vs Off-nail(A+F): force-pair mean ± 95% CI | LME]")

    for ax, band_cfg in zip(axes_bf2, BANDS):
        band_name  = band_cfg["name"]
        band_df    = subj_bcde_pair[subj_bcde_pair["band"] == band_name]
        pairs      = sorted(band_df["pair_label"].unique(),
                            key=lambda s: float(s.split("–")[1]))
        x_centers  = np.arange(len(pairs))
        y_max_used = 0.0

        for xi, pair in enumerate(pairs):
            pair_df = band_df[band_df["pair_label"] == pair]
            lme_res = lme_bcde_af(pair_df)
            ci_tops = []

            for gi, grp in enumerate(BCDE_AF_ORDER):
                dx   = (gi - 0.5) * (BOX_W * 2 + GAP)
                xpos = xi + dx
                data = pair_df[pair_df["Group"] == grp]["accuracy"].dropna().values
                if len(data) == 0:
                    continue
                n    = len(data)
                mean = float(np.mean(data))
                ci   = CI95_MULTIPLIER * float(np.std(data, ddof=1) / np.sqrt(n))
                color = BCDE_AF_PALETTE[grp]
                ax.plot([xpos, xpos], [mean - ci, mean + ci],
                        color=color, linewidth=ERR_LW, zorder=2, clip_on=False)
                for cap_y in (mean - ci, mean + ci):
                    ax.plot([xpos - CAP_W, xpos + CAP_W], [cap_y, cap_y],
                            color=color, linewidth=ERR_LW, zorder=2, clip_on=False)
                ax.scatter([xpos], [mean], c=[color], s=DOT_SIZE ** 2,
                           linewidths=DOT_LW, zorder=4, clip_on=False)
                ci_tops.append(float(mean + ci))
                y_max_used = max(y_max_used, mean + ci)

            if lme_res and ci_tops:
                y_brk = max(ci_tops) + 2.5
                y_top_brk = y_brk + 0.5
                x1 = xi + (0 - 0.5) * (BOX_W * 2 + GAP)
                x2 = xi + (1 - 0.5) * (BOX_W * 2 + GAP)
                ax.plot([x1, x1, x2, x2], [y_brk, y_top_brk, y_top_brk, y_brk],
                        color=ACCENT_RED, linewidth=0.75, clip_on=False, zorder=5)
                p_txt = f"{star(lme_res['p'])}  p={lme_res['p']:.3f}"
                ax.text((x1 + x2) / 2, y_top_brk + 0.6, p_txt,
                        ha="center", va="bottom", fontsize=max(8, FONT_ANNOT - 1),
                        color=ACCENT_RED, fontweight="bold", clip_on=False, zorder=6)
                print(f"  [{band_name}] {pair} g: Δ={lme_res['coef']:.1f}%  "
                      f"p={lme_res['p']:.4f}  {star(lme_res['p'])}")

        ax.axhline(JND_PCT, color=CRITERION_COLOR, linestyle="--",
                   linewidth=1.0, alpha=0.85, zorder=REF_LINE_ZORDER)
        ax.set_title(band_cfg["title"], fontsize=FONT_LABEL, fontweight="bold", pad=6)
        ax.set_xticks(x_centers)
        ax.set_xticklabels([f"{p} g" for p in pairs], fontsize=FONT_TICK - 2)
        ax.set_yticks(Y_TICKS)
        ax.yaxis.set_major_locator(FixedLocator(Y_TICKS))
        ax.tick_params(axis="y", labelsize=FONT_TICK)
        ax.tick_params(axis="x", length=0)
        y_top = min(YLIM_TOP_CAP, y_max_used + 18)
        ax.set_ylim(YLIM_BOT, y_top)
        ax.spines["left"].set_bounds(YLIM_BOT, 100)
        sns.despine(ax=ax)

        x_trans = ax.get_xaxis_transform()
        for xi_t in range(len(pairs)):
            ax.plot([xi_t, xi_t], [0, TICK_LEN], color=BLACK, linewidth=1.0,
                    solid_capstyle="butt", transform=x_trans, clip_on=False, zorder=6)
        y_trans = ax.get_yaxis_transform()
        y_lo, y_hi = ax.get_ylim()
        for y in Y_TICKS:
            if y_lo - 1e-9 <= y <= y_hi + 1e-9:
                ax.plot([0, TICK_LEN], [y, y], color=BLACK, linewidth=1.0,
                        solid_capstyle="butt", transform=y_trans, clip_on=False, zorder=6)

    axes_bf2[0].set_ylabel("Accuracy (%)", fontsize=FONT_LABEL)
    axes_bf2[1].set_ylabel("")

    leg_bf2 = [
        plt.Line2D([0], [0], color=BCDE_AF_PALETTE[g], marker="o",
                   markersize=DOT_SIZE * 0.7, linewidth=ERR_LW, label=BCDE_AF_LABELS[i])
        for i, g in enumerate(BCDE_AF_ORDER)
    ]
    add_legend_outside(fig_bf2, axes_bf2[0], leg_bf2, ncol=2,
                       top=FIG_LEGEND_TOP, bottom=FIG_LEGEND_BOTTOM,
                       left=0.07, right=0.97,
                       above_axes=ATD.FIG_LEGEND_ABOVE_AXES)
    fig_bf2.subplots_adjust(left=0.07, right=0.97, top=0.88, bottom=0.12, wspace=0.28)

    _save_600dpi(fig_bf2, os.path.join(OUTPUT_DIR, "fd_bcde_vs_af_meanCI.png"))
    plt.close(fig_bf2)


draw_bcde_af_band_figure()
draw_bcde_af_meanCI_figure()
