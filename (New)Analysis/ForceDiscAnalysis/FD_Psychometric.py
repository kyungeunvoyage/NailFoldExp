"""
FD_Psychometric.py — Force Discrimination Psychometric Analysis
================================================================

Three analyses:

  1. Psychometric curves  [Figure PC1]
     Per-region mean accuracy vs Weber fraction (± bootstrap CI)
     Logistic (2AFC) sigmoid fit overlay
     Separate panels: Low band (ref=1g) | High band (ref=26g)

  2. JND (threshold Weber fraction)  [Figure JND2]
     Bootstrap group-level sigmoid → WR at 75% criterion
     On-nail (C+D) vs Off-nail (A+F) per band
     Also shows all 6 regions as individual dots + CI bars

  3. GEE forest plot  [Figure GEE3]
     Population-averaged logistic model
     (GEE, Binomial family, Exchangeable working correlation)
     Odds ratios + 95% CI for all key group contrasts
"""

import os
import glob
import warnings
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.optimize import curve_fit
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.cov_struct import Exchangeable
from matplotlib.ticker import FixedLocator, MultipleLocator
import io as _io
from PIL import Image as _PILImage

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
_SCRIPT_DIR  = Path(__file__).resolve().parent
_ATD_C1_PATH = _SCRIPT_DIR.parent / "ATDAnalysis" / "ATD_C1_Fig(Anika).py"

spec = importlib.util.spec_from_file_location("atd_c1", _ATD_C1_PATH)
ATD  = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ATD)

OUTPUT_DIR = str(_SCRIPT_DIR / "Output" / "Psychometric")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Shared style ──────────────────────────────────────────────────────────────
BLACK           = ATD.BLACK
ACCENT_RED      = ATD.ACCENT_RED
CRITERION_COLOR = ATD.CRITERION_COLOR
REF_LINE_ZORDER = ATD.REF_LINE_ZORDER
FONT_TICK       = ATD.FONT_TICK
FONT_LABEL      = ATD.FONT_LABEL
FONT_LEGEND     = ATD.FONT_LEGEND
FONT_ANNOT      = ATD.FONT_ANNOT
BOX_LINEWIDTH   = ATD.BOX_LINEWIDTH
CAP_LINEWIDTH   = ATD.CAP_LINEWIDTH
pale_box_face   = ATD.pale_box_face
_hsb_scatter_rgba  = ATD._hsb_scatter_rgba
add_legend_outside = ATD.add_legend_outside
FIG_LEGEND_TOP     = ATD.FIG_LEGEND_TOP
FIG_LEGEND_BOTTOM  = ATD.FIG_LEGEND_BOTTOM
TICK_LEN = ATD.TICK_LEN_AXES

ON_NAIL_COLOR  = ATD.ON_TOUCH
OFF_NAIL_COLOR = "#7C94B8"
EXPORT_WIDTH   = 2102

# Per-region palette (A–F)
REGION_COLORS = {
    "A": "#56708A", "B": "#3A7D44", "C": ON_NAIL_COLOR,
    "D": "#A0522D", "E": "#8B4F9E", "F": "#C08040",
}
REGION_ORDER = ["A", "B", "C", "D", "E", "F"]


# ── Save helper ───────────────────────────────────────────────────────────────
def _save(fig, fname):
    buf = _io.BytesIO()
    fig.savefig(buf, format="png", dpi=600,
                bbox_inches="tight", pad_inches=0.04, facecolor="white")
    buf.seek(0)
    img = _PILImage.open(buf).convert("RGB")
    h = round(EXPORT_WIDTH * img.height / img.width)
    img.resize((EXPORT_WIDTH, h), _PILImage.Resampling.LANCZOS).save(fname)
    print(f"Saved: {fname}  ({EXPORT_WIDTH}×{h} px)")


# ── Load & preprocess ─────────────────────────────────────────────────────────
FILE_PATTERN = str(_SCRIPT_DIR.parent.parent / "Data" / "(FD)CurData" /
                   "P*_ForceDiscrimination.csv")
files = glob.glob(FILE_PATTERN)
if not files:
    raise FileNotFoundError(f"No CSV: {FILE_PATTERN}")
print(f"Loaded {len(files)} participant files.")

df_raw = pd.concat([pd.read_csv(f, encoding="utf-8-sig") for f in sorted(files)],
                   ignore_index=True)

sub_col = "SubjectID" if "SubjectID" in df_raw.columns else "Subject"

df_raw["correct"] = np.where(
    df_raw["Comparison"] > df_raw["Reference"],
    df_raw["ChoseComparison"] == 1,
    df_raw["ChoseComparison"] == 0,
).astype(int)

df_raw["WR"]   = (df_raw["Comparison"] - df_raw["Reference"]).abs() / df_raw["Reference"]
df_raw["band"] = df_raw["Reference"].apply(lambda r: "Low" if r == 1 else "High")

BANDS = [
    {"name": "Low",  "ref": 1,  "label": "Low band  (ref = 1 g)"},
    {"name": "High", "ref": 26, "label": "High band  (ref = 26 g)"},
]

# WR levels per band (sorted)
WR_LEVELS = {
    "Low":  sorted(df_raw[df_raw["band"] == "Low"]["WR"].unique()),
    "High": sorted(df_raw[df_raw["band"] == "High"]["WR"].unique()),
}
print("WR levels:", WR_LEVELS)

subjects = sorted(df_raw[sub_col].unique())
N_SUBJ   = len(subjects)

GROUP_MAP = {
    "C": "On-nail", "D": "On-nail",
    "A": "Off-nail", "F": "Off-nail",
}


# =============================================================================
# Helper: sigmoid psychometric function (2AFC, floor = 0.5)
# =============================================================================
def sigmoid_2afc(wr, wr0, k):
    """
    2AFC psychometric function.
    p(WR) = 0.5 + 0.5 * logistic(k * (WR - WR0))
    WR0 = threshold at 75% criterion, k = slope.
    """
    return 0.5 + 0.5 / (1.0 + np.exp(-k * (wr - wr0)))


def fit_sigmoid(wr_vals, acc_vals):
    """
    Fit 2AFC sigmoid. acc_vals in [0, 1].
    Returns (wr0, k) or (nan, nan) on failure.
    """
    try:
        wr_mid = float(np.median(wr_vals))
        popt, _ = curve_fit(
            sigmoid_2afc, wr_vals, acc_vals,
            p0=[wr_mid, 3.0],
            bounds=([0.01, 0.01], [5.0, 100.0]),
            maxfev=5000,
        )
        return float(popt[0]), float(popt[1])
    except Exception:
        return np.nan, np.nan


def bootstrap_jnd(subject_matrix, wr_levels, n_boot=2000, seed=42):
    """
    Bootstrap JND (= WR0 from sigmoid fit) for a group of subjects.
    subject_matrix: shape (n_subjects, n_wr_levels), values in [0, 1].
    Returns (median_jnd, ci_lo_95, ci_hi_95).
    """
    rng   = np.random.default_rng(seed)
    n_sub = len(subject_matrix)
    wr    = np.array(wr_levels, dtype=float)
    jnds  = []
    for _ in range(n_boot):
        idx     = rng.integers(0, n_sub, size=n_sub)
        grp_acc = subject_matrix[idx].mean(axis=0)
        wr0, _  = fit_sigmoid(wr, grp_acc)
        if np.isfinite(wr0):
            jnds.append(wr0)
    if len(jnds) < 50:
        return np.nan, np.nan, np.nan
    return (float(np.median(jnds)),
            float(np.percentile(jnds, 2.5)),
            float(np.percentile(jnds, 97.5)))


# Build per-subject accuracy matrix per (band, region)
def subject_wr_matrix(band_name, region):
    """Returns (subjects_list, wr_levels, matrix n×m) accuracy in [0,1]."""
    wr_lvls = WR_LEVELS[band_name]
    mat = []
    valid_subs = []
    bd = df_raw[(df_raw["band"] == band_name) & (df_raw["Region"] == region)]
    for sub in subjects:
        row = []
        ok  = True
        for wr in wr_lvls:
            trials = bd[(bd[sub_col] == sub) & (np.isclose(bd["WR"], wr))]["correct"]
            if len(trials) == 0:
                ok = False; break
            row.append(float(trials.mean()))
        if ok:
            mat.append(row)
            valid_subs.append(sub)
    return valid_subs, wr_lvls, np.array(mat)


# =============================================================================
# Figure PC1 — Psychometric Curves
# =============================================================================
def draw_psychometric_curves():
    ATD.apply_plot_style()
    sns.set_theme(style="white")

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.0), facecolor="white")
    WR_FINE = np.linspace(0.05, 2.0, 300)

    print("\n[Figure PC1 — Psychometric curves per region]")

    for ax, band_cfg in zip(axes, BANDS):
        band = band_cfg["name"]
        wr   = np.array(WR_LEVELS[band])

        for region in REGION_ORDER:
            _, _, mat = subject_wr_matrix(band, region)
            if len(mat) == 0:
                continue

            group_mean = mat.mean(axis=0) * 100   # %
            group_sem  = (mat.std(axis=0, ddof=1) / np.sqrt(len(mat))) * 100

            color = REGION_COLORS[region]

            # Data points ± SEM
            ax.errorbar(wr, group_mean, yerr=group_sem,
                        fmt="o", color=color, markersize=5,
                        capsize=3, linewidth=1.2, label=f"Region {region}",
                        zorder=4, clip_on=False)

            # Sigmoid fit
            wr0, k = fit_sigmoid(wr, group_mean / 100)
            if np.isfinite(wr0):
                fit_y = sigmoid_2afc(WR_FINE, wr0, k) * 100
                ax.plot(WR_FINE, fit_y, color=color, linewidth=1.0,
                        alpha=0.6, zorder=3)
                print(f"  [{band}] Region {region}: WR0={wr0:.3f}, k={k:.2f}")
            else:
                print(f"  [{band}] Region {region}: fit failed")

        ax.axhline(75, color=CRITERION_COLOR, linestyle="--",
                   linewidth=1.0, alpha=0.85, zorder=2,
                   label="75% criterion" if ax is axes[0] else None)
        ax.axhline(50, color=BLACK, linestyle=":", linewidth=0.8,
                   alpha=0.5, zorder=2,
                   label="Chance (50%)" if ax is axes[0] else None)

        ax.set_title(band_cfg["label"], fontsize=FONT_LABEL,
                     fontweight="bold", pad=6)
        ax.set_xlabel("Weber Fraction  (|Δf| / f_ref)", fontsize=FONT_LABEL)
        ax.set_xlim(0.2, max(wr) * 1.25)
        ax.set_ylim(20, 115)
        ax.set_yticks([25, 50, 75, 100])
        ax.yaxis.set_major_locator(FixedLocator([25, 50, 75, 100]))
        ax.tick_params(axis="both", labelsize=FONT_TICK)
        ax.tick_params(axis="x", length=0)

        # inward y-ticks
        y_trans = ax.get_yaxis_transform()
        for y in [25, 50, 75, 100]:
            ax.plot([0, TICK_LEN], [y, y], color=BLACK, linewidth=1.0,
                    solid_capstyle="butt", transform=y_trans,
                    clip_on=False, zorder=6)

        ax.spines["left"].set_bounds(20, 100)
        sns.despine(ax=ax)

    axes[0].set_ylabel("Accuracy (%)", fontsize=FONT_LABEL)
    axes[1].set_ylabel("")

    # Legend
    leg_handles = (
        [mpatches.Patch(facecolor=REGION_COLORS[r], edgecolor=BLACK,
                        linewidth=0.6, label=f"Region {r}")
         for r in REGION_ORDER]
        + [plt.Line2D([0],[0], color=CRITERION_COLOR, linestyle="--",
                      linewidth=1.0, label="75% criterion"),
           plt.Line2D([0],[0], color=BLACK, linestyle=":",
                      linewidth=0.8, alpha=0.5, label="Chance (50%)")]
    )
    add_legend_outside(fig, axes[0], leg_handles, ncol=4,
                       top=FIG_LEGEND_TOP, bottom=FIG_LEGEND_BOTTOM,
                       left=0.07, right=0.97,
                       above_axes=ATD.FIG_LEGEND_ABOVE_AXES)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.88, bottom=0.13, wspace=0.30)

    _save(fig, os.path.join(OUTPUT_DIR, "fd_psychometric_curves.png"))
    plt.close(fig)


# =============================================================================
# Figure JND2 — JND (threshold WR) comparison
# =============================================================================
def draw_jnd_comparison():
    ATD.apply_plot_style()
    sns.set_theme(style="white")

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.0), facecolor="white")
    print("\n[Figure JND2 — Bootstrap JND per region & On/Off-nail comparison]")

    ERR_LW  = 1.8
    DOT_S   = 7.0
    CAP_W   = 0.15

    for ax, band_cfg in zip(axes, BANDS):
        band  = band_cfg["name"]
        wr    = np.array(WR_LEVELS[band])

        region_jnds = {}   # region → (median, lo, hi)
        for region in REGION_ORDER:
            _, _, mat = subject_wr_matrix(band, region)
            if len(mat) < 5:
                region_jnds[region] = (np.nan, np.nan, np.nan)
                continue
            med, lo, hi = bootstrap_jnd(mat, wr)
            region_jnds[region] = (med, lo, hi)
            print(f"  [{band}] Region {region}: JND={med:.3f} "
                  f"[{lo:.3f}, {hi:.3f}]")

        # Group JNDs: On-nail (C+D averaged), Off-nail (A+F averaged)
        def group_matrix(regions):
            mats = []
            for r in regions:
                _, _, m = subject_wr_matrix(band, r)
                mats.append(m)
            # stack and average across regions per subject
            stacked = np.stack(mats, axis=2)   # (n_sub, n_wr, n_regions)
            return stacked.mean(axis=2)

        mat_on  = group_matrix(["C", "D"])
        mat_off = group_matrix(["A", "F"])
        jnd_on  = bootstrap_jnd(mat_on,  wr)
        jnd_off = bootstrap_jnd(mat_off, wr)
        print(f"  [{band}] On-nail (C+D): JND={jnd_on[0]:.3f} [{jnd_on[1]:.3f}, {jnd_on[2]:.3f}]")
        print(f"  [{band}] Off-nail (A+F): JND={jnd_off[0]:.3f} [{jnd_off[1]:.3f}, {jnd_off[2]:.3f}]")

        # ── Plot individual regions ─────────────────────────────
        for xi, region in enumerate(REGION_ORDER):
            med, lo, hi = region_jnds[region]
            if not np.isfinite(med):
                continue
            color = REGION_COLORS[region]
            ax.plot([xi, xi], [lo, hi], color=color, linewidth=ERR_LW,
                    zorder=3, clip_on=False)
            for cap_y in (lo, hi):
                ax.plot([xi - CAP_W, xi + CAP_W], [cap_y, cap_y],
                        color=color, linewidth=ERR_LW, zorder=3, clip_on=False)
            ax.scatter([xi], [med], c=[color], s=DOT_S ** 2,
                       linewidths=0.5, edgecolors=BLACK, zorder=5, clip_on=False)

        # ── On-nail / Off-nail group summaries ──────────────────
        GROUP_SUMMARY = [
            (6.5,  jnd_on,  ON_NAIL_COLOR,  "On-nail\n(C+D)"),
            (7.5, jnd_off, OFF_NAIL_COLOR, "Off-nail\n(A+F)"),
        ]
        for gx, (med, lo, hi), col, lbl in GROUP_SUMMARY:
            if not np.isfinite(med):
                continue
            ax.plot([gx, gx], [lo, hi], color=col, linewidth=ERR_LW + 0.5,
                    zorder=3, clip_on=False)
            for cap_y in (lo, hi):
                ax.plot([gx - CAP_W, gx + CAP_W], [cap_y, cap_y],
                        color=col, linewidth=ERR_LW + 0.5,
                        zorder=3, clip_on=False)
            ax.scatter([gx], [med], c=[col], s=(DOT_S * 1.3) ** 2,
                       linewidths=0.5, edgecolors=BLACK, zorder=5, clip_on=False)

        # Divider between regions and groups
        ax.axvline(6.0, color=BLACK, linewidth=0.5, linestyle=":", alpha=0.5)

        ax.set_title(band_cfg["label"], fontsize=FONT_LABEL,
                     fontweight="bold", pad=6)
        ax.set_xticks(list(range(6)) + [6.5, 7.5])
        ax.set_xticklabels(REGION_ORDER + ["On-nail\n(C+D)", "Off-nail\n(A+F)"],
                           fontsize=FONT_TICK - 1)
        ax.tick_params(axis="x", length=0)
        ax.tick_params(axis="y", labelsize=FONT_TICK)
        ax.set_xlim(-0.7, 8.2)

        # y-axis
        all_vals = [v[0] for v in region_jnds.values() if np.isfinite(v[0])]
        all_vals += [jnd_on[0], jnd_off[0]]
        all_vals = [v for v in all_vals if np.isfinite(v)]
        ymin = max(0, min(all_vals) * 0.6) if all_vals else 0
        ymax = max(all_vals) * 1.5 if all_vals else 2.0
        ax.set_ylim(ymin, ymax)
        ax.set_ylabel("JND (Weber Fraction at 75%)", fontsize=FONT_LABEL)

        # horizontal reference: chance WR
        ax.axhline(0, color=BLACK, linewidth=0.5, alpha=0.3, linestyle="-")
        sns.despine(ax=ax)

        # inward y ticks
        y_trans = ax.get_yaxis_transform()
        for y in ax.get_yticks():
            if ymin - 1e-6 <= y <= ymax + 1e-6:
                ax.plot([0, TICK_LEN], [y, y], color=BLACK, linewidth=1.0,
                        solid_capstyle="butt", transform=y_trans,
                        clip_on=False, zorder=6)

    # Legend
    leg = (
        [mpatches.Patch(facecolor=REGION_COLORS[r], edgecolor=BLACK,
                        linewidth=0.6, label=f"Region {r}")
         for r in REGION_ORDER]
        + [mpatches.Patch(facecolor=ON_NAIL_COLOR,  edgecolor=BLACK,
                          linewidth=0.6, label="On-nail (C+D)"),
           mpatches.Patch(facecolor=OFF_NAIL_COLOR, edgecolor=BLACK,
                          linewidth=0.6, label="Off-nail (A+F)")]
    )
    add_legend_outside(fig, axes[0], leg, ncol=4,
                       top=FIG_LEGEND_TOP, bottom=FIG_LEGEND_BOTTOM,
                       left=0.07, right=0.97,
                       above_axes=ATD.FIG_LEGEND_ABOVE_AXES)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.88, bottom=0.13, wspace=0.35)

    _save(fig, os.path.join(OUTPUT_DIR, "fd_jnd_comparison.png"))
    plt.close(fig)


# =============================================================================
# Figure GEE3 — Forest plot: odds ratios from GEE (Binomial, Exchangeable)
# =============================================================================
def run_gee_contrast(group_col, ref_label, target_label):
    """
    Run GEE on trial-level binary data.
    Returns odds ratio, 95% CI, and p-value.
    """
    sub = df_raw.dropna(subset=[sub_col, group_col, "correct"]).copy()
    sub = sub[sub[group_col].isin([ref_label, target_label])]
    sub["_bin"] = (sub[group_col] == target_label).astype(int)
    if sub[sub_col].nunique() < 3 or sub[group_col].nunique() < 2:
        return None
    try:
        X   = sm.add_constant(sub["_bin"])
        res = GEE(sub["correct"], X,
                  groups=sub[sub_col],
                  family=Binomial(),
                  cov_struct=Exchangeable()).fit()
        coef = float(res.params["_bin"])
        ci   = res.conf_int()
        lo   = float(ci.loc["_bin", 0])
        hi   = float(ci.loc["_bin", 1])
        p    = float(res.pvalues["_bin"])
        return {
            "coef": coef, "or": np.exp(coef),
            "or_lo": np.exp(lo), "or_hi": np.exp(hi), "p": p
        }
    except Exception as e:
        print(f"  GEE failed ({target_label} vs {ref_label}): {e}")
        return None


def draw_gee_forest():
    ATD.apply_plot_style()
    sns.set_theme(style="white")

    print("\n[Figure GEE3 — GEE forest plot (Binomial, Exchangeable)]")

    # ── Build contrast list ───────────────────────────────────────────────────
    # Add group columns to a working copy
    df_work = df_raw.copy()

    # Group A: On-nail(C+D) vs Off-nail(A+F)
    df_work["grp_cd_af"] = df_work["Region"].map(
        {"C": "On-nail", "D": "On-nail", "A": "Off-nail", "F": "Off-nail"}
    )
    # Group B: On-nail(B+C+D+E) vs Off-nail(A+F)
    df_work["grp_bcde_af"] = df_work["Region"].map(
        {"B": "On-nail", "C": "On-nail", "D": "On-nail",
         "E": "On-nail", "A": "Off-nail", "F": "Off-nail"}
    )
    # Group C: individual region labels (use Region column directly)
    df_work["grp_region"] = df_work["Region"]

    contrasts = []

    def _run_gee(df_in, group_col, ref_label, target_label):
        sub = df_in.dropna(subset=[sub_col, group_col, "correct"]).copy()
        sub = sub[sub[group_col].isin([ref_label, target_label])]
        sub["_bin"] = (sub[group_col] == target_label).astype(int)
        if sub[sub_col].nunique() < 3 or sub[group_col].nunique() < 2:
            return None
        try:
            X   = sm.add_constant(sub["_bin"])
            res = GEE(sub["correct"], X, groups=sub[sub_col],
                      family=Binomial(), cov_struct=Exchangeable()).fit()
            coef = float(res.params["_bin"])
            ci   = res.conf_int()
            lo, hi = float(ci.loc["_bin", 0]), float(ci.loc["_bin", 1])
            p    = float(res.pvalues["_bin"])
            return {"coef": coef, "or": np.exp(coef),
                    "or_lo": np.exp(lo), "or_hi": np.exp(hi), "p": p}
        except Exception as e:
            print(f"  GEE failed ({target_label} vs {ref_label}): {e}")
            return None

    # ── 1. Main grouping contrasts ────────────────────────────────────────────
    for grp_col, ref, tgt, label in [
        ("grp_cd_af",   "Off-nail", "On-nail", "On-nail (C+D)  vs  Off-nail (A+F)"),
        ("grp_bcde_af", "Off-nail", "On-nail", "On-nail (B+C+D+E)  vs  Off-nail (A+F)"),
    ]:
        r = _run_gee(df_work, grp_col, ref, tgt)
        if r:
            contrasts.append({"label": label, "section": "Grouping", **r})

    # ── 2. On-nail individual regions vs Off-nail pooled ─────────────────────
    for on_r in ["C", "D", "B", "E"]:
        df_tmp = df_work[df_work["Region"].isin([on_r, "A", "F"])].copy()
        df_tmp["grp_tmp"] = df_tmp["Region"].map(
            {on_r: "On-nail", "A": "Off-nail", "F": "Off-nail"}
        )
        r = _run_gee(df_tmp, "grp_tmp", "Off-nail", "On-nail")
        if r:
            contrasts.append({
                "label": f"Region {on_r}  vs  Off-nail (A+F)",
                "section": "Region vs Off-nail", **r
            })

    # ── 3. Off-nail individual regions vs On-nail pooled ─────────────────────
    for off_r in ["A", "F"]:
        df_tmp = df_work[df_work["Region"].isin(["C", "D", off_r])].copy()
        df_tmp["grp_tmp"] = df_tmp["Region"].map(
            {"C": "On-nail", "D": "On-nail", off_r: "Off-nail"}
        )
        r = _run_gee(df_tmp, "grp_tmp", "Off-nail", "On-nail")
        if r:
            contrasts.append({
                "label": f"On-nail (C+D)  vs  Region {off_r}",
                "section": "On-nail vs Region", **r
            })

    # ── 4. All pairwise region contrasts ─────────────────────────────────────
    region_pairs = [
        ("A", "C"), ("A", "D"), ("F", "C"), ("F", "D"),
        ("A", "B"), ("A", "E"), ("F", "B"), ("F", "E"),
    ]
    for ref_r, tgt_r in region_pairs:
        df_tmp = df_work[df_work["Region"].isin([ref_r, tgt_r])].copy()
        df_tmp["grp_tmp"] = df_tmp["Region"]
        r = _run_gee(df_tmp, "grp_tmp", ref_r, tgt_r)
        if r:
            contrasts.append({
                "label": f"Region {tgt_r}  vs  Region {ref_r}",
                "section": "Pairwise Regions", **r
            })

    # ── Print results ─────────────────────────────────────────────────────────
    for c in contrasts:
        star = ("***" if c["p"] < 0.001 else "**" if c["p"] < 0.01 else
                "*" if c["p"] < 0.05 else "n.s.")
        print(f"  {c['label']}: OR={c['or']:.3f} "
              f"[{c['or_lo']:.3f}, {c['or_hi']:.3f}], "
              f"p={c['p']:.4f} {star}")

    # ── Build figure ──────────────────────────────────────────────────────────
    n_rows = len(contrasts)
    fig_h  = max(5.5, n_rows * 0.52 + 1.5)
    fig, ax = plt.subplots(1, 1, figsize=(8.5, fig_h), facecolor="white")

    sections = ["Grouping", "Region vs Off-nail", "On-nail vs Region", "Pairwise Regions"]
    sec_colors = {
        "Grouping":            "#1A1A1A",
        "Region vs Off-nail":  "#2E6FA3",
        "On-nail vs Region":   "#C04040",
        "Pairwise Regions":    "#5A7A5A",
    }

    y_pos   = []
    y_tick  = []
    section_breaks = {}
    yi = 0

    prev_sec = None
    for c in reversed(contrasts):  # bottom-up so first is at top
        if c["section"] != prev_sec:
            if prev_sec is not None:
                yi += 0.5   # gap between sections
            section_breaks[c["section"]] = yi + 0.5
            prev_sec = c["section"]

        color = sec_colors.get(c["section"], BLACK)
        star  = ("***" if c["p"] < 0.001 else "**" if c["p"] < 0.01 else
                 "*" if c["p"] < 0.05 else "n.s.")
        sig   = c["p"] < 0.05

        # CI bar
        ax.plot([c["or_lo"], c["or_hi"]], [yi, yi],
                color=color, linewidth=1.5, zorder=2)
        # caps
        cap_h = 0.18
        for x_cap in (c["or_lo"], c["or_hi"]):
            ax.plot([x_cap, x_cap], [yi - cap_h, yi + cap_h],
                    color=color, linewidth=1.5, zorder=2)
        # point
        ax.scatter([c["or"]], [yi],
                   c=[color], s=60, zorder=4,
                   marker="D" if sig else "o",
                   edgecolors=BLACK, linewidths=0.4)

        # p-value annotation
        x_annot = c["or_hi"] + 0.03
        ax.text(x_annot, yi, f"{star}  p={c['p']:.3f}",
                va="center", ha="left",
                fontsize=FONT_ANNOT - 1, color=color,
                fontweight="bold" if sig else "normal")

        y_pos.append(yi)
        y_tick.append(c["label"])
        yi += 1

    # Null line at OR=1
    ax.axvline(1.0, color=BLACK, linewidth=1.0, linestyle="--",
               alpha=0.6, zorder=1)

    # Section labels
    for sec, y_sec in section_breaks.items():
        ax.text(ax.get_xlim()[0] if ax.get_xlim()[0] else 0.3,
                y_sec, sec, fontsize=FONT_ANNOT - 0.5,
                color=sec_colors.get(sec, BLACK), fontstyle="italic",
                va="bottom", ha="left")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_tick, fontsize=FONT_TICK - 1)
    ax.set_xlabel("Odds Ratio  (GEE, Binomial, Exchangeable)",
                  fontsize=FONT_LABEL)
    ax.set_xlim(0.3, 3.5)
    ax.set_ylim(-0.8, yi + 0.8)
    ax.tick_params(axis="x", labelsize=FONT_TICK)
    ax.tick_params(axis="y", length=0)

    ax.set_title("Force Discrimination: GEE Odds Ratios\n"
                 "(Population-averaged logistic model, all trials)",
                 fontsize=FONT_LABEL, fontweight="bold", pad=8)

    sns.despine(ax=ax, left=True)
    fig.tight_layout(rect=[0, 0, 0.88, 1.0])

    _save(fig, os.path.join(OUTPUT_DIR, "fd_gee_forest.png"))
    plt.close(fig)


# =============================================================================
# =============================================================================
# Figure PC2 — Psychometric Curves (4-parameter logistic, all free)
# Model: p(WR) = lam_L + (lam_H - lam_L) * logistic(b*(WR - WR0))
#   lam_L : lower asymptote (floor, free — can go below 0.5)
#   lam_H : upper asymptote (ceiling, free — plateau below 100%)
#   WR0   : inflection point (midpoint between floor and ceiling)
#   b     : slope
#
# JND = WR where p = lam_L + (lam_H - lam_L) * 0.75
#       i.e. 75% of the way from floor to ceiling
# =============================================================================
def logistic_4p(wr, lam_l, lam_h, wr0, b):
    """4-parameter logistic: floor=lam_l (free), ceiling=lam_h (free)."""
    return lam_l + (lam_h - lam_l) / (1.0 + np.exp(-b * (wr - wr0)))


def fit_logistic_unrestricted(wr_vals, acc_vals):
    """
    Fit 4-parameter logistic. Returns (lam_l, lam_h, wr0, b) or (nan,nan,nan,nan).
    Falls back to 2-parameter if 4-param fails.
    """
    wr_vals = np.array(wr_vals, dtype=float)
    acc_vals = np.array(acc_vals, dtype=float)
    lam_l0  = max(0.0,  float(np.min(acc_vals)) - 0.05)
    lam_h0  = min(1.0,  float(np.max(acc_vals)) + 0.05)
    wr0_0   = float(np.median(wr_vals))
    try:
        popt, _ = curve_fit(
            logistic_4p, wr_vals, acc_vals,
            p0=[lam_l0, lam_h0, wr0_0, 8.0],
            bounds=([-0.5, 0.5, 0.0,  0.1],
                    [ 0.5, 1.0, 3.0, 200.0]),
            maxfev=10000,
        )
        return tuple(float(x) for x in popt)
    except Exception:
        # fallback: 2-param logistic
        try:
            wr_mid = float(np.median(wr_vals))
            def log2p(wr, a, b): return 1.0 / (1.0 + np.exp(-(a + b * wr)))
            popt2, _ = curve_fit(log2p, wr_vals, acc_vals,
                                 p0=[-wr_mid * 5.0, 5.0],
                                 bounds=([-50.0, 0.01], [50.0, 200.0]),
                                 maxfev=8000)
            a, b = float(popt2[0]), float(popt2[1])
            lam_l = float(log2p(wr_vals.min(), a, b))
            lam_h = float(log2p(wr_vals.max(), a, b))
            wr0_v = (-a) / b
            return lam_l, lam_h, wr0_v, b
        except Exception:
            return np.nan, np.nan, np.nan, np.nan


def logistic_unrestricted(wr, *params):
    """Wrapper that calls logistic_4p with unpacked params."""
    return logistic_4p(wr, *params)


def jnd_unrestricted(lam_l, lam_h, wr0, b, criterion=0.75):
    """
    WR at which p = lam_l + (lam_h - lam_l)*criterion.
    Inverse of the 4-parameter logistic at the relative criterion level.
    """
    if not all(np.isfinite([lam_l, lam_h, wr0, b])) or b == 0:
        return np.nan
    # logistic(b*(WR-WR0)) = criterion  → b*(WR-WR0) = logit(criterion)
    logit_c = np.log(criterion / (1.0 - criterion))
    return wr0 + logit_c / b


def draw_psychometric_curves_free():
    ATD.apply_plot_style()
    sns.set_theme(style="white")

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.5), facecolor="white")
    WR_FINE = np.linspace(0.05, 2.0, 300)

    print("\n[Figure PC2 — Psychometric curves (unconstrained logistic)]")

    for ax, band_cfg in zip(axes, BANDS):
        band = band_cfg["name"]
        wr   = np.array(WR_LEVELS[band])
        wr_min, wr_max = float(wr.min()), float(wr.max())

        for region in REGION_ORDER:
            _, _, mat = subject_wr_matrix(band, region)
            if len(mat) == 0:
                continue

            group_mean = mat.mean(axis=0) * 100   # %
            group_sem  = (mat.std(axis=0, ddof=1) / np.sqrt(len(mat))) * 100

            color = REGION_COLORS[region]

            # Data points ± SEM
            ax.errorbar(wr, group_mean, yerr=group_sem,
                        fmt="o", color=color, markersize=5,
                        capsize=3, linewidth=1.2, label=f"Region {region}",
                        zorder=4, clip_on=False)

            # 4-parameter logistic fit
            params = fit_logistic_unrestricted(wr, group_mean / 100)
            lam_l, lam_h, wr0_fit, b_fit = params
            if np.isfinite(lam_l):
                fit_y = logistic_4p(WR_FINE, lam_l, lam_h, wr0_fit, b_fit) * 100
                ax.plot(WR_FINE, fit_y, color=color, linewidth=1.0,
                        alpha=0.65, zorder=3)
                thr = jnd_unrestricted(lam_l, lam_h, wr0_fit, b_fit)
                print(f"  [{band}] Region {region}: "
                      f"floor={lam_l*100:.1f}%  ceil={lam_h*100:.1f}%  "
                      f"JND(75%)={thr:.3f}")
            else:
                print(f"  [{band}] Region {region}: fit failed")

        # Reference lines
        ax.axhline(75, color=CRITERION_COLOR, linestyle="--",
                   linewidth=1.0, alpha=0.85, zorder=2)
        ax.axhline(50, color=BLACK, linestyle=":", linewidth=0.8,
                   alpha=0.5, zorder=2)

        ax.set_title(band_cfg["label"], fontsize=FONT_LABEL,
                     fontweight="bold", pad=6)
        ax.set_xlabel("Weber Fraction  (|Δf| / f_ref)", fontsize=FONT_LABEL)
        ax.set_xlim(0.2, max(wr) * 1.25)
        ax.set_ylim(0, 115)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.yaxis.set_major_locator(FixedLocator([0, 25, 50, 75, 100]))
        ax.tick_params(axis="both", labelsize=FONT_TICK)
        ax.tick_params(axis="x", length=0)

        y_trans = ax.get_yaxis_transform()
        for y in [0, 25, 50, 75, 100]:
            ax.plot([0, TICK_LEN], [y, y], color=BLACK, linewidth=1.0,
                    solid_capstyle="butt", transform=y_trans,
                    clip_on=False, zorder=6)

        ax.spines["left"].set_bounds(0, 100)
        sns.despine(ax=ax)

    axes[0].set_ylabel("Accuracy (%)", fontsize=FONT_LABEL)
    axes[1].set_ylabel("")

    axes[0].text(0.02, 0.04,
                 "Model: p = logistic(a + b·WR)\n"
                 "No floor constraint — curve can cross below 50%",
                 transform=axes[0].transAxes,
                 fontsize=FONT_ANNOT - 1, color="#555555",
                 va="bottom", ha="left")

    leg_handles = (
        [mpatches.Patch(facecolor=REGION_COLORS[r], edgecolor=BLACK,
                        linewidth=0.6, label=f"Region {r}")
         for r in REGION_ORDER]
        + [plt.Line2D([0],[0], color=CRITERION_COLOR, linestyle="--",
                      linewidth=1.0, label="75% criterion"),
           plt.Line2D([0],[0], color=BLACK, linestyle=":",
                      linewidth=0.8, alpha=0.5, label="Chance (50%)")]
    )
    add_legend_outside(fig, axes[0], leg_handles, ncol=4,
                       top=FIG_LEGEND_TOP, bottom=FIG_LEGEND_BOTTOM,
                       left=0.07, right=0.97,
                       above_axes=ATD.FIG_LEGEND_ABOVE_AXES)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.88, bottom=0.13, wspace=0.30)

    _save(fig, os.path.join(OUTPUT_DIR, "fd_psychometric_curves_free.png"))
    plt.close(fig)


# =============================================================================
# Figure PC3 — Pooled (all 6 regions) psychometric curve
# =============================================================================
def draw_psychometric_pooled():
    ATD.apply_plot_style()
    sns.set_theme(style="white")

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.5), facecolor="white")
    WR_FINE = np.linspace(0.05, 2.0, 300)
    POOL_COLOR = "#3A3A3A"

    print("\n[Figure PC3 — Pooled all-region psychometric curve]")

    for ax, band_cfg in zip(axes, BANDS):
        band = band_cfg["name"]
        wr   = np.array(WR_LEVELS[band])

        all_mats = []
        for region in REGION_ORDER:
            _, _, mat = subject_wr_matrix(band, region)
            if len(mat) > 0:
                all_mats.append(mat)

        # Per-region mean across subjects, then average across regions
        region_means = np.stack([m.mean(axis=0) for m in all_mats], axis=0)  # (n_regions, n_wr)
        group_mean = region_means.mean(axis=0) * 100   # mean across regions
        group_sem  = (region_means.std(axis=0, ddof=1)
                      / np.sqrt(len(all_mats))) * 100  # SEM across regions

        ax.errorbar(wr, group_mean, yerr=group_sem,
                    fmt="o", color=POOL_COLOR, markersize=6,
                    capsize=3, linewidth=1.4, label="All regions (A–F)",
                    zorder=4, clip_on=False)

        lam_l, lam_h, wr0_fit, b_fit = fit_logistic_unrestricted(wr, group_mean / 100)
        if np.isfinite(lam_l):
            fit_y = logistic_4p(WR_FINE, lam_l, lam_h, wr0_fit, b_fit) * 100
            ax.plot(WR_FINE, fit_y, color=POOL_COLOR, linewidth=1.8,
                    alpha=0.8, zorder=3)
            thr = jnd_unrestricted(lam_l, lam_h, wr0_fit, b_fit)
            print(f"  [{band}] Pooled: floor={lam_l*100:.1f}%  "
                  f"ceil={lam_h*100:.1f}%  JND(75%)={thr:.3f}")

        ax.axhline(75, color=CRITERION_COLOR, linestyle="--",
                   linewidth=1.0, alpha=0.85, zorder=2)
        ax.axhline(50, color=BLACK, linestyle=":", linewidth=0.8,
                   alpha=0.5, zorder=2)

        ax.set_title(band_cfg["label"], fontsize=FONT_LABEL,
                     fontweight="bold", pad=6)
        ax.set_xlabel("Weber Fraction  (|Δf| / f_ref)", fontsize=FONT_LABEL)
        ax.set_xlim(0.2, float(wr.max()) * 1.25)
        ax.set_ylim(0, 115)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.yaxis.set_major_locator(FixedLocator([0, 25, 50, 75, 100]))
        ax.tick_params(axis="both", labelsize=FONT_TICK)
        ax.tick_params(axis="x", length=0)
        y_trans = ax.get_yaxis_transform()
        for y in [0, 25, 50, 75, 100]:
            ax.plot([0, TICK_LEN], [y, y], color=BLACK, linewidth=1.0,
                    solid_capstyle="butt", transform=y_trans,
                    clip_on=False, zorder=6)
        ax.spines["left"].set_bounds(0, 100)
        sns.despine(ax=ax)

    axes[0].set_ylabel("Accuracy (%)", fontsize=FONT_LABEL)
    axes[1].set_ylabel("")
    axes[0].text(0.02, 0.04,
                 "Model: p = logistic(a + b·WR)\nAll 6 regions pooled",
                 transform=axes[0].transAxes,
                 fontsize=FONT_ANNOT - 1, color="#555555", va="bottom")

    leg = [plt.Line2D([0], [0], color=POOL_COLOR, marker="o",
                      markersize=6, linewidth=1.4, label="All regions (A–F)"),
           plt.Line2D([0],[0], color=CRITERION_COLOR, linestyle="--",
                      linewidth=1.0, label="75% criterion"),
           plt.Line2D([0],[0], color=BLACK, linestyle=":", linewidth=0.8,
                      alpha=0.5, label="Chance (50%)")]
    add_legend_outside(fig, axes[0], leg, ncol=3,
                       top=FIG_LEGEND_TOP, bottom=FIG_LEGEND_BOTTOM,
                       left=0.07, right=0.97,
                       above_axes=ATD.FIG_LEGEND_ABOVE_AXES)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.88, bottom=0.13, wspace=0.30)

    _save(fig, os.path.join(OUTPUT_DIR, "fd_psychometric_pooled.png"))
    plt.close(fig)


# =============================================================================
# Figure PC4 — On-nail (C+D) vs Off-nail (A+F) psychometric curves
# =============================================================================
def draw_psychometric_onnail_offnail():
    ATD.apply_plot_style()
    sns.set_theme(style="white")

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.5), facecolor="white")
    WR_FINE = np.linspace(0.05, 2.0, 300)

    GROUPS = [
        {"label": "On-nail (C+D)",  "regions": ["C", "D"], "color": ON_NAIL_COLOR},
        {"label": "Off-nail (A+F)", "regions": ["A", "F"], "color": OFF_NAIL_COLOR},
    ]

    print("\n[Figure PC4 — On-nail (C+D) vs Off-nail (A+F) psychometric curves]")

    for ax, band_cfg in zip(axes, BANDS):
        band = band_cfg["name"]
        wr   = np.array(WR_LEVELS[band])

        for grp in GROUPS:
            mats = []
            for r in grp["regions"]:
                _, _, mat = subject_wr_matrix(band, r)
                if len(mat) > 0:
                    mats.append(mat)
            if not mats:
                continue

            # Average across the two regions per subject → (n_subj, n_wr)
            subj_mean  = np.stack(mats, axis=0).mean(axis=0)
            group_mean = subj_mean.mean(axis=0) * 100
            group_sem  = (subj_mean.std(axis=0, ddof=1)
                          / np.sqrt(len(subj_mean))) * 100

            color = grp["color"]
            ax.errorbar(wr, group_mean, yerr=group_sem,
                        fmt="o", color=color, markersize=6,
                        capsize=3, linewidth=1.4, label=grp["label"],
                        zorder=4, clip_on=False)

            lam_l, lam_h, wr0_fit, b_fit = fit_logistic_unrestricted(wr, group_mean / 100)
            if np.isfinite(lam_l):
                fit_y = logistic_4p(WR_FINE, lam_l, lam_h, wr0_fit, b_fit) * 100
                ax.plot(WR_FINE, fit_y, color=color, linewidth=1.8,
                        alpha=0.8, zorder=3)
                thr = jnd_unrestricted(lam_l, lam_h, wr0_fit, b_fit)
                print(f"  [{band}] {grp['label']}: "
                      f"floor={lam_l*100:.1f}%  ceil={lam_h*100:.1f}%  "
                      f"JND(75%)={thr:.3f}")

        ax.axhline(75, color=CRITERION_COLOR, linestyle="--",
                   linewidth=1.0, alpha=0.85, zorder=2)
        ax.axhline(50, color=BLACK, linestyle=":", linewidth=0.8,
                   alpha=0.5, zorder=2)

        ax.set_title(band_cfg["label"], fontsize=FONT_LABEL,
                     fontweight="bold", pad=6)
        ax.set_xlabel("Weber Fraction  (|Δf| / f_ref)", fontsize=FONT_LABEL)
        ax.set_xlim(0.2, float(wr.max()) * 1.25)
        ax.set_ylim(0, 115)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.yaxis.set_major_locator(FixedLocator([0, 25, 50, 75, 100]))
        ax.tick_params(axis="both", labelsize=FONT_TICK)
        ax.tick_params(axis="x", length=0)
        y_trans = ax.get_yaxis_transform()
        for y in [0, 25, 50, 75, 100]:
            ax.plot([0, TICK_LEN], [y, y], color=BLACK, linewidth=1.0,
                    solid_capstyle="butt", transform=y_trans,
                    clip_on=False, zorder=6)
        ax.spines["left"].set_bounds(0, 100)
        sns.despine(ax=ax)

    axes[0].set_ylabel("Accuracy (%)", fontsize=FONT_LABEL)
    axes[1].set_ylabel("")
    axes[0].text(0.02, 0.04,
                 "Model: p = logistic(a + b·WR)\nC+D averaged → On-nail,  A+F averaged → Off-nail",
                 transform=axes[0].transAxes,
                 fontsize=FONT_ANNOT - 1, color="#555555", va="bottom")

    leg = ([mpatches.Patch(facecolor=g["color"], edgecolor=BLACK,
                           linewidth=0.8, label=g["label"]) for g in GROUPS]
           + [plt.Line2D([0],[0], color=CRITERION_COLOR, linestyle="--",
                         linewidth=1.0, label="75% criterion"),
              plt.Line2D([0],[0], color=BLACK, linestyle=":", linewidth=0.8,
                         alpha=0.5, label="Chance (50%)")])
    add_legend_outside(fig, axes[0], leg, ncol=4,
                       top=FIG_LEGEND_TOP, bottom=FIG_LEGEND_BOTTOM,
                       left=0.07, right=0.97,
                       above_axes=ATD.FIG_LEGEND_ABOVE_AXES)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.88, bottom=0.13, wspace=0.30)

    _save(fig, os.path.join(OUTPUT_DIR, "fd_psychometric_onnail_offnail.png"))
    plt.close(fig)


# =============================================================================
# Run all
# =============================================================================
draw_psychometric_curves()
draw_psychometric_curves_free()
draw_psychometric_pooled()
draw_psychometric_onnail_offnail()
draw_jnd_comparison()
draw_gee_forest()
