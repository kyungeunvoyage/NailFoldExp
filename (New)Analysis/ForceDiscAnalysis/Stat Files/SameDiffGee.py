"""
Force Discrimination – Same/Different 2AFC GEE Analysis
=========================================================
Adapted from the original "which is stronger" GEE pairwise script.

CSV column assumptions (P*_ForceDiscrimination_SameDiff.csv, P*_ForceDiscrimination_SameDiff_26g.csv)
-------------------------------------------------------------
Subject, Session, Condition, Region, Reference, Comparison,
Rep, TrialType (same_ref / same_comp / diff_rc / diff_cr),
Stim1, Stim2, GroundTruth (SAME / DIFFERENT),
UserChoice (SAME / DIFFERENT), IsCorrect (1 / 0)

Accuracy definition
--------------------
IsCorrect is already computed in the CSV (1 = correct, 0 = incorrect).
Chance level  = 50%   (2AFC: SAME or DIFFERENT)
JND threshold = 75%   (halfway between chance and ceiling)

Additional analyses vs. original script
-----------------------------------------
1. Hit Rate / False Alarm Rate / d′ / criterion c per pair × region group
   (Signal Detection Theory decomposition — separates sensitivity from bias)
2. Order effect check: diff_rc vs diff_cr accuracy compared per pair
   (directly tests whether stimulus presentation order affected performance)
3. Response bias overview (3-panel stacked bars, analogous to response_bias_overview.py)
   — delivered SAME/DIFFERENT, response distribution, incorrect-trial responses

Statistics priority:
  1. statsmodels GEE (binomial family, subject clustering) — preferred
  2. scipy Wilcoxon signed-rank test (fallback)
  3. Permutation test (final fallback)
"""

import os
import glob
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.transforms import blended_transform_factory
import warnings
warnings.filterwarnings("ignore")

# ── Stat backend ──────────────────────────────────────────────────────────────
USE_GEE = False
USE_WILCOXON = False

try:
    from statsmodels.genmod.generalized_estimating_equations import GEE
    from statsmodels.genmod.families import Binomial
    USE_GEE = True
    print("Using: statsmodels GEE (binomial)")
except ImportError:
    try:
        from scipy.stats import wilcoxon
        USE_WILCOXON = True
        print("statsmodels not found — using scipy Wilcoxon (fallback)")
    except ImportError:
        print("Neither statsmodels nor scipy — using permutation test (fallback)")

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = "/Users/kyungeunjung/NailFoldExp/Data/(FD)CurData"
FILE_PATTERNS = [
    os.path.join(DATA_DIR, "P*_ForceDiscrimination_SameDiff.csv"),
    os.path.join(DATA_DIR, "P*_ForceDiscrimination_SameDiff_26g.csv"),
]
OUTPUT_DIR = "/Users/kyungeunjung/NailFoldExp/(New)Analysis/ForceDiscAnalysis/Output/SameDiff_GEE"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CHANCE_PCT = 50.0
JND_PCT    = 75.0
EXPORT_WIDTH_2COL = 2102
EXPORT_HEIGHT_2COL = 1298  # match Fig2_ontouch_vs_inair_2col.png
ACC_BY_PAIR_FIGSIZE = (14.0, round(14.0 * EXPORT_HEIGHT_2COL / EXPORT_WIDTH_2COL, 3))
ATD_FIG2_REF_N = 5
ATD_FIG2_DODGED_BOX_WIDTH = 0.275  # one dodged box at 5 forces (ATD Fig2)
COMBINED_PANEL_COUNT = 2
STRIP_JITTER_REF = 0.12
CAP_WIDTH = 0.10
BOX_LINEWIDTH = 1.0
COLOR_LOW_BAND = "#BAD6EB"
COLOR_HIGH_BAND = "#D0E4FF"
FIG_PANEL_TOP_FRAC = 0.80
FIG_LEGEND_ANCHOR_Y = 0.975
BRACKET_BASE_PAD = 3.0
BRACKET_TIER_STEP = 8.0
BRACKET_YLIM_CAP = 122.0
BLACK = "#1A1A1A"
TICK_LEN_AXES = 0.016
ACCURACY_YSPINE_TOP = 100.0
FONT_TICK = 16
FONT_LABEL = 14
FONT_LEGEND = 12
FIG_AXIS_LABELPAD = 6
AXIS_SPINE_LW = 2.0
# paper_fd_onnail_vs_offnail_meanCI: FONT_TICK=16, fig (8×4.5 in), export 2102×1137 px
PAPER_FD_FIG_H_IN = 4.5
PAPER_FD_OUT_H_PX = 1137
SAVE_DPI_COMBINED = 600


def _combined_axis_font_size(base_pt):
    """Match paper FD tick label pixel size on the taller 2-col export canvas."""
    fig_h = ACC_BY_PAIR_FIGSIZE[1]
    return round(base_pt * PAPER_FD_OUT_H_PX / EXPORT_HEIGHT_2COL * fig_h / PAPER_FD_FIG_H_IN)


COMBINED_FONT_TICK = _combined_axis_font_size(FONT_TICK)
COMBINED_FONT_LABEL = _combined_axis_font_size(FONT_LABEL)
COMBINED_FONT_LEGEND = _combined_axis_font_size(FONT_LEGEND)

ON_NAIL  = ["C", "D"]
OFF_NAIL = ["A", "F"]

BAND_CONFIG = {
    "Low": {
        "ref": 1,
        "pair_order": ["0.4–1", "0.6–1", "1–1.4", "1–2"],
        "suffix": "_low",
        "title_ref": "1 g",
    },
    "High": {
        "ref": 26,
        "pair_order": ["10–26", "15–26", "26–60"],
        "suffix": "_high",
        "title_ref": "26 g",
    },
}


def band_title_text(band_label, title_ref, n_subjects):
    return f"{band_label} band (ref = {title_ref}, n = {n_subjects})"


def combined_panel_box_width(n_pairs):
    """Match Fig2 dodged-box pixel width inside each half-width 2-col panel."""
    return (
        ATD_FIG2_DODGED_BOX_WIDTH * n_pairs / ATD_FIG2_REF_N * COMBINED_PANEL_COUNT
    )


def save_png_at_width(fig, out_path, width_px=EXPORT_WIDTH_2COL, *,
                      height_px=None, pad_inches=0.04, dpi=150):
    import io
    from PIL import Image

    buf = io.BytesIO()
    fig.savefig(
        buf, format="png", dpi=dpi, bbox_inches="tight",
        pad_inches=pad_inches, facecolor="white",
    )
    buf.seek(0)
    master = Image.open(buf).convert("RGB")
    if height_px is None:
        height_px = round(width_px * master.height / master.width)
    master.resize((width_px, height_px), Image.Resampling.LANCZOS).save(out_path)


# ── Load data ─────────────────────────────────────────────────────────────────
import re

all_files = sorted(set(
    f for pat in FILE_PATTERNS for f in glob.glob(pat)
))
if not all_files:
    raise FileNotFoundError(
        "No CSV files found matching:\n  " + "\n  ".join(FILE_PATTERNS)
    )

def subject_number(filepath):
    m = re.search(r"P(\d+)", os.path.basename(filepath))
    return int(m.group(1)) if m else 0

files = sorted([f for f in all_files if subject_number(f) > 73])
if not files:
    raise ValueError(
        f"No files with subject number > 73 found.\n"
        f"Files present: {sorted(os.path.basename(f) for f in all_files)}"
    )
print(f"Total files found  : {len(all_files)}")
print(f"Files with ID > 73 : {len(files)}")
print(f"  → {[os.path.basename(f) for f in files]}")

df = pd.concat(
    [pd.read_csv(f, encoding="utf-8-sig") for f in files],
    ignore_index=True,
)

df["correct"] = df["IsCorrect"].astype(int)

df["pair_label"] = df.apply(
    lambda r: f"{min(r['Reference'], r['Comparison']):g}–{max(r['Reference'], r['Comparison']):g}",
    axis=1,
)

df["band"] = df["Reference"].map({1: "Low", 26: "High"})

df["region_group"] = df["Region"].map(
    {r: "On-nail"  for r in ON_NAIL} |
    {r: "Off-nail" for r in OFF_NAIL}
)

def fix_order(order, actual):
    fixed = []
    for p in order:
        if p in actual:
            fixed.append(p)
        else:
            alt = [a for a in actual if set(a.replace("–", "-").split("-")) == set(p.replace("–", "-").split("-"))]
            fixed.append(alt[0] if alt else p)
    return fixed


# ── SDT helper ────────────────────────────────────────────────────────────────
def compute_sdt(sub_df):
    diff_trials = sub_df[sub_df["GroundTruth"] == "DIFFERENT"]
    same_trials = sub_df[sub_df["GroundTruth"] == "SAME"]
    n_diff = len(diff_trials)
    n_same = len(same_trials)
    if n_diff == 0 or n_same == 0:
        return {"H": np.nan, "FA": np.nan, "d_prime": np.nan, "criterion": np.nan}

    hits = (diff_trials["UserChoice"] == "DIFFERENT").sum()
    fas  = (same_trials["UserChoice"]  == "DIFFERENT").sum()
    H  = (hits + 0.5) / (n_diff + 1)
    FA = (fas  + 0.5) / (n_same + 1)

    from scipy.stats import norm
    d_prime   = norm.ppf(H) - norm.ppf(FA)
    criterion = -0.5 * (norm.ppf(H) + norm.ppf(FA))
    return {"H": H, "FA": FA, "d_prime": d_prime, "criterion": criterion}


def _permutation_pval(a, b, n_perm=5000, seed=0):
    rng = np.random.default_rng(seed)
    obs = abs(np.mean(a) - np.mean(b))
    combined = np.concatenate([a, b])
    n_a = len(a)
    count = sum(
        abs(np.mean(combined[:n_a]) - np.mean(combined[n_a:])) >= obs
        for _ in range(n_perm)
        if not rng.shuffle(combined) or True
    )
    return count / n_perm


def run_gee_pairwise(df_band, subj_acc, pair_order):
    results = {}
    subj_means = subj_acc.copy()

    for p1, p2 in itertools.combinations(pair_order, 2):
        if USE_GEE:
            chunk = df_band[df_band["pair_label"].isin([p1, p2])].copy()
            if chunk["Subject"].nunique() < 2:
                results[(p1, p2)] = np.nan
                continue
            chunk["pair_dummy"] = (chunk["pair_label"] == p2).astype(int)
            chunk = chunk.rename(columns={"Subject": "subj_id"})
            try:
                fit = GEE.from_formula("correct ~ pair_dummy", groups="subj_id",
                                        data=chunk, family=Binomial()).fit(maxiter=60)
                results[(p1, p2)] = fit.pvalues["pair_dummy"]
                continue
            except Exception as e:
                print(f"  GEE failed ({p1} vs {p2}): {e}")

        paired = subj_means[subj_means["pair_label"].isin([p1, p2])]\
                 .pivot(index="Subject", columns="pair_label", values="accuracy").dropna()
        if len(paired) < 5:
            results[(p1, p2)] = np.nan
            continue
        if USE_WILCOXON:
            try:
                from scipy.stats import wilcoxon
                _, pval = wilcoxon(paired[p1].values, paired[p2].values)
                results[(p1, p2)] = pval
                continue
            except Exception:
                pass
        results[(p1, p2)] = _permutation_pval(paired[p1].values, paired[p2].values)

    return results


def run_gee_region(df_band, pair_order):
    df_reg = df_band[df_band["region_group"].notna()].copy()
    results = {}

    for pair in pair_order:
        chunk = df_reg[df_reg["pair_label"] == pair].copy()
        if chunk["Subject"].nunique() < 2:
            results[pair] = np.nan
            continue
        chunk["region_dummy"] = (chunk["region_group"] == "On-nail").astype(int)
        chunk = chunk.rename(columns={"Subject": "subj_id"})

        if USE_GEE:
            try:
                fit = GEE.from_formula("correct ~ region_dummy", groups="subj_id",
                                        data=chunk, family=Binomial()).fit(maxiter=60)
                results[pair] = fit.pvalues["region_dummy"]
                continue
            except Exception as e:
                print(f"  GEE region failed ({pair}): {e}")

        subj_reg = (
            df_reg[df_reg["pair_label"] == pair]
            .groupby(["Subject", "region_group"])["correct"].mean().reset_index()
        )
        pivot = subj_reg.pivot(index="Subject", columns="region_group", values="correct").dropna()
        if len(pivot) < 5:
            results[pair] = np.nan
            continue
        if USE_WILCOXON:
            try:
                from scipy.stats import wilcoxon
                _, pval = wilcoxon(pivot["On-nail"].values, pivot["Off-nail"].values)
                results[pair] = pval
                continue
            except Exception:
                pass
        results[pair] = _permutation_pval(pivot["On-nail"].values, pivot["Off-nail"].values)

    return results


# ── Plotting helpers ──────────────────────────────────────────────────────────
C1 = "#2166AC"
C_ON  = "#7FB3D3"
C_OFF = "#D3E9F5"
RED   = "#c0392b"
C_SAME = "#5B9BD5"
C_DIFF = "#E07B39"


def pval_label(p):
    if np.isnan(p): return ""
    if p < 0.001:   return "***"
    if p < 0.01:    return "**"
    if p < 0.05:    return "*"
    return f"n.s. p={p:.3f}"


def draw_bracket(ax, x1, x2, y, label, tick_h=3.5):
    ax.plot([x1, x1, x2, x2], [y, y+tick_h, y+tick_h, y],
            color=RED, lw=1.2, clip_on=False)
    if label:
        ax.text((x1+x2)/2, y+tick_h+1.5, label,
                ha="center", va="bottom", fontsize=9, color=RED, fontweight="bold")


def draw_bracket_above_axes(ax, x1, x2, tier, label, tier_step=0.065):
    """Significance bracket in margin above plot — y-axis stays 0–100."""
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    y = 1.02 + tier * tier_step
    tick_h = 0.022
    ax.plot([x1, x1, x2, x2], [y, y + tick_h, y + tick_h, y],
            color=RED, lw=1.2, clip_on=False, transform=trans)
    if label:
        ax.text((x1 + x2) / 2, y + tick_h + 0.012, label, transform=trans,
                ha="center", va="bottom", fontsize=9, color=RED, fontweight="bold", clip_on=False)


def draw_bracket_horizontal_data(ax, x1, x2, y, label, *, linewidth=AXIS_SPINE_LW):
    """Horizontal significance line in data coords (no vertical end ticks)."""
    ax.plot([x1, x2], [y, y], color=RED, lw=linewidth, clip_on=False, zorder=30)
    if label:
        ax.text((x1 + x2) / 2, y + 1.5, label,
                ha="center", va="bottom", fontsize=9, color=RED, fontweight="bold",
                clip_on=False, zorder=31)


def jitter_x(n, width=0.12, seed=42):
    return (np.random.default_rng(seed).random(n) - 0.5) * width


def apply_accuracy_y_spine_bounds(ax, y_top=ACCURACY_YSPINE_TOP):
    """Left spine ends at the top data tick (100%) even when ylim extends for brackets."""
    ax.spines["left"].set_bounds(0, y_top)


def apply_combined_axis_spines(ax):
    """Thicker bottom/left axis outlines for 2-col export."""
    for spine in ("left", "bottom"):
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color(BLACK)
        ax.spines[spine].set_linewidth(AXIS_SPINE_LW)


def add_inward_tick_guides(ax, n_x=None, y_ticks=None, *, labelsize=None,
                           linewidth=AXIS_SPINE_LW):
    """Short inward tick marks at each x/y label (ATD C1 style)."""
    ax.grid(False)
    tick_kw = {"axis": "both", "which": "both", "length": 0}
    if labelsize is not None:
        tick_kw["labelsize"] = labelsize
    ax.tick_params(**tick_kw)
    x_trans = blended_transform_factory(ax.transData, ax.transAxes)
    y_trans = blended_transform_factory(ax.transAxes, ax.transData)
    y_lo, y_hi = ax.get_ylim()
    if y_ticks is None:
        y_vals = [t for t in ax.get_yticks() if y_lo - 1e-9 <= t <= y_hi + 1e-9]
    else:
        y_vals = [t for t in y_ticks if y_lo - 1e-9 <= t <= y_hi + 1e-9]
    x_positions = range(n_x) if n_x is not None else ax.get_xticks()
    for xi in x_positions:
        ax.plot(
            [xi, xi], [0, TICK_LEN_AXES],
            color=BLACK, linewidth=linewidth, solid_capstyle="butt",
            transform=x_trans, clip_on=False, zorder=6,
        )
    for y in y_vals:
        ax.plot(
            [0, TICK_LEN_AXES], [y, y],
            color=BLACK, linewidth=linewidth, solid_capstyle="butt",
            transform=y_trans, clip_on=False, zorder=6,
        )


# ── Response bias overview (Same/Different analogue of response_bias_overview.py) ─
def _bias_proportions(df_sub):
    """Return pct SAME / pct DIFFERENT for delivered, all responses, incorrect-only."""
    n_total = len(df_sub)
    if n_total == 0:
        return None

    def _pct(mask):
        return mask.sum() / n_total * 100

    pct_del_same = _pct(df_sub["GroundTruth"] == "SAME")
    pct_del_diff = 100 - pct_del_same

    pct_resp_same = _pct(df_sub["UserChoice"] == "SAME")
    pct_resp_diff = 100 - pct_resp_same

    inc = df_sub[df_sub["correct"] == 0]
    n_incorrect = len(inc)
    if n_incorrect == 0:
        pct_inc_same = pct_inc_diff = np.nan
    else:
        pct_inc_same = (inc["UserChoice"] == "SAME").sum() / n_incorrect * 100
        pct_inc_diff = 100 - pct_inc_same

    return {
        "n_total": n_total,
        "n_incorrect": n_incorrect,
        "del_same": pct_del_same,
        "del_diff": pct_del_diff,
        "resp_same": pct_resp_same,
        "resp_diff": pct_resp_diff,
        "inc_same": pct_inc_same,
        "inc_diff": pct_inc_diff,
    }


def _draw_bias_bar(ax, pct_same, pct_diff, label_same, label_diff, title, n, subtitle=None):
    """Horizontal stacked bar — SAME (blue) vs DIFFERENT (orange)."""
    ax.barh(0, pct_same, height=0.55, color=C_SAME, label=label_same)
    ax.barh(0, pct_diff, height=0.55, left=pct_same, color=C_DIFF, label=label_diff)

    if pct_same > 8:
        ax.text(pct_same / 2, 0, f"{pct_same:.1f}%", ha="center", va="center",
                fontsize=13, fontweight="bold", color="white")
    if pct_diff > 8:
        ax.text(pct_same + pct_diff / 2, 0, f"{pct_diff:.1f}%", ha="center", va="center",
                fontsize=13, fontweight="bold", color="white")

    ax.axvline(50, color="gray", lw=1.2, ls="--", alpha=0.7)
    ax.text(50, 0.34, "50%", ha="center", fontsize=9, color="gray", va="bottom")

    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, 0.55)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=10)
    ax.set_yticks([])
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    if subtitle:
        ax.set_xlabel(subtitle, fontsize=10, labelpad=6)
    ax.legend(
        loc="lower center", bbox_to_anchor=(0.5, -0.38), ncol=2, frameon=False, fontsize=10,
        handles=[
            mpatches.Patch(color=C_SAME, label=label_same),
            mpatches.Patch(color=C_DIFF, label=label_diff),
        ],
    )
    ax.text(0.98, 1.02, f"n = {n:,} trials", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=9, color="#555")


def run_response_bias_overview(df_sub, out_path, suptitle, *, print_header=None):
    """Three-panel response bias figure (pooled, band, or single-subject)."""
    stats = _bias_proportions(df_sub)
    if stats is None:
        return

    if print_header:
        print(f"\n{print_header}")
        print(f"  Delivered   SAME {stats['del_same']:.1f}%  |  DIFFERENT {stats['del_diff']:.1f}%")
        print(f"  Responses   SAME {stats['resp_same']:.1f}%  |  DIFFERENT {stats['resp_diff']:.1f}%")
        if stats["n_incorrect"]:
            print(f"  Incorrect   SAME {stats['inc_same']:.1f}%  |  DIFFERENT {stats['inc_diff']:.1f}%  "
                  f"(n={stats['n_incorrect']})")

    fig, axes = plt.subplots(3, 1, figsize=(9, 7.5))
    fig.suptitle(suptitle, fontsize=13, fontweight="bold", y=0.99)

    _draw_bias_bar(
        axes[0], stats["del_same"], stats["del_diff"],
        "Ground truth: SAME", "Ground truth: DIFFERENT",
        "Plot 1 – Delivered Stimulus Distribution",
        n=stats["n_total"],
        subtitle="Were the two stimuli physically the same or different?",
    )
    _draw_bias_bar(
        axes[1], stats["resp_same"], stats["resp_diff"],
        "Responded: SAME", "Responded: DIFFERENT",
        "Plot 2 – Response Distribution  (all trials)",
        n=stats["n_total"],
        subtitle="What did participants say?",
    )
    _draw_bias_bar(
        axes[2], stats["inc_same"], stats["inc_diff"],
        "Responded: SAME", "Responded: DIFFERENT",
        "Plot 3 – Incorrect-Trial Response Distribution",
        n=stats["n_incorrect"],
        subtitle="Among wrong answers, did they say SAME or DIFFERENT?",
    )

    plt.tight_layout(rect=[0, 0, 1, 0.97], h_pad=2.5)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {os.path.relpath(out_path, OUTPUT_DIR)}")


def _pair_response_distribution_rows(df_sub, pair_order):
    rows = []
    for pair in pair_order:
        for gt in ["SAME", "DIFFERENT"]:
            sub = df_sub[(df_sub["pair_label"] == pair) & (df_sub["GroundTruth"] == gt)]
            n = len(sub)
            if n == 0:
                continue
            n_same = int((sub["UserChoice"] == "SAME").sum())
            n_diff = n - n_same
            rows.append({
                "pair_label": pair,
                "ground_truth": gt,
                "n_trials": n,
                "n_resp_same": n_same,
                "n_resp_diff": n_diff,
                "pct_resp_same": n_same / n * 100,
                "pct_resp_diff": n_diff / n * 100,
            })
    return rows


def _label_stacked_bar(ax, x_left, width, y, text, min_width=8):
    if width < min_width:
        return
    ax.text(x_left + width / 2, y, text, ha="center", va="center",
            fontsize=8, fontweight="bold", color="white")


def save_response_distribution_by_pair(df_sub, pair_order, title, out_path, *, csv_path=None):
    """
    Per force pair: response distribution on SAME trials vs DIFFERENT trials.
    Each row = one pair × ground-truth type; stacked bar = responded SAME | DIFFERENT.
    """
    rows = _pair_response_distribution_rows(df_sub, pair_order)
    if not rows:
        return
    if csv_path:
        pd.DataFrame(rows).to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(11, 1.55 * len(pair_order) * 2 + 1.2))
    y = 0
    yticks, ylabels = [], []
    bar_h = 0.38
    group_gap = 0.55

    for pair in pair_order:
        pair_rows = [r for r in rows if r["pair_label"] == pair]
        for r in pair_rows:
            ps, pct_diff = r["pct_resp_same"], r["pct_resp_diff"]
            ax.barh(y, ps, height=bar_h, color=C_SAME, edgecolor="black", linewidth=0.6)
            ax.barh(y, pct_diff, height=bar_h, left=ps, color=C_DIFF, edgecolor="black", linewidth=0.6)
            _label_stacked_bar(ax, 0, ps, y, f"{ps:.1f}%\n(n={r['n_resp_same']})")
            _label_stacked_bar(ax, ps, pct_diff, y, f"{pct_diff:.1f}%\n(n={r['n_resp_diff']})")
            yticks.append(y)
            gt = r["ground_truth"]
            ylabels.append(f"{pair}  ·  {gt} trials  (n={r['n_trials']})")
            y -= 1
        y -= group_gap

    ax.axvline(50, color="gray", ls="--", lw=1.0, alpha=0.7)
    ax.text(50, y + group_gap + 0.5, "50%", ha="center", fontsize=8, color="gray")
    ax.set_xlim(0, 100)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=10)
    ax.set_xlabel("Response distribution (%)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.invert_yaxis()
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.legend(handles=[
        mpatches.Patch(facecolor=C_SAME, edgecolor="black", label="Responded: SAME"),
        mpatches.Patch(facecolor=C_DIFF, edgecolor="black", label="Responded: DIFFERENT"),
    ], loc="lower right", frameon=False, fontsize=10)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {os.path.relpath(out_path, OUTPUT_DIR)}")


def _same_trial_response_rows(df_sub, pair_order):
    """Response distribution within SAME trials only, per force pair."""
    rows = []
    for pair in pair_order:
        sub = df_sub[(df_sub["pair_label"] == pair) & (df_sub["GroundTruth"] == "SAME")]
        n = len(sub)
        if n == 0:
            continue
        n_same = int((sub["UserChoice"] == "SAME").sum())
        n_diff = n - n_same
        rows.append({
            "pair_label": pair,
            "n_trials": n,
            "n_resp_same": n_same,
            "n_resp_diff": n_diff,
            "pct_resp_same": n_same / n * 100,
            "pct_resp_diff": n_diff / n * 100,
        })
    return rows


def save_same_trial_response_by_pair(df_sub, pair_order, title, out_path, *, csv_path=None):
    """SAME trials only: how participants responded (SAME vs DIFFERENT), by force pair."""
    rows = _same_trial_response_rows(df_sub, pair_order)
    if not rows:
        return
    if csv_path:
        pd.DataFrame(rows).to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(11, 1.1 * len(pair_order) + 1.8))
    yticks, ylabels = [], []
    bar_h = 0.55

    for yi, r in enumerate(rows):
        ps, pct_diff = r["pct_resp_same"], r["pct_resp_diff"]
        ax.barh(yi, ps, height=bar_h, color=C_SAME, edgecolor="black", linewidth=0.6)
        ax.barh(yi, pct_diff, height=bar_h, left=ps, color=C_DIFF, edgecolor="black", linewidth=0.6)
        _label_stacked_bar(ax, 0, ps, yi, f"{ps:.1f}%\n(n={r['n_resp_same']})")
        _label_stacked_bar(ax, ps, pct_diff, yi, f"{pct_diff:.1f}%\n(n={r['n_resp_diff']})")
        yticks.append(yi)
        ylabels.append(f"{r['pair_label']}  (SAME trials, n={r['n_trials']})")

    ax.axvline(50, color="gray", ls="--", lw=1.0, alpha=0.7)
    ax.text(50, len(rows) - 0.15, "50%", ha="center", fontsize=8, color="gray")
    ax.set_xlim(0, 100)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=10)
    ax.set_xlabel("Response distribution (%)  — ground truth always SAME", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.invert_yaxis()
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.legend(handles=[
        mpatches.Patch(facecolor=C_SAME, edgecolor="black", label="Responded: SAME (correct)"),
        mpatches.Patch(facecolor=C_DIFF, edgecolor="black", label="Responded: DIFFERENT (incorrect)"),
    ], loc="lower right", frameon=False, fontsize=10)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {os.path.relpath(out_path, OUTPUT_DIR)}")


def _draw_pair_accuracy_boxplot(ax, pair_order, values_by_pair, *,
                                pairwise_pvals_dict=None, dot_label=None,
                                data_ylim_top=100, show_ylabel=True,
                                bracket_in_data_space=False, box_color="#dde6f0",
                                box_width=0.42, jitter_width=0.12):
    """Boxplot + scatter; y-axis 0–100 (brackets may extend ylim when in data space)."""
    band_max = 0.0
    for xi, pair in enumerate(pair_order):
        vals = np.asarray(values_by_pair.get(pair, []), dtype=float) * 100
        if len(vals) == 0:
            continue
        band_max = max(band_max, float(np.nanmax(vals)))
        bp = ax.boxplot(
            [vals], positions=[xi], widths=box_width,
            patch_artist=True, showfliers=False, capwidths=CAP_WIDTH,
            whiskerprops={"linewidth": BOX_LINEWIDTH, "color": "black"},
            capprops={"linewidth": BOX_LINEWIDTH, "color": "black"},
            medianprops={"color": RED, "linewidth": 2},
            boxprops={"linewidth": BOX_LINEWIDTH, "edgecolor": "black"},
        )
        bp["boxes"][0].set_facecolor(box_color); bp["boxes"][0].set_edgecolor("black")
        bp["medians"][0].set_color(RED); bp["medians"][0].set_linewidth(2)
        jx = xi + jitter_x(len(vals), width=jitter_width)
        ax.scatter(jx, vals, color=C1, alpha=0.6, s=20, zorder=3)

    ax.axhline(JND_PCT,    color="black", ls="--", lw=1.2, alpha=0.8)
    ax.axhline(CHANCE_PCT, color="gray",  ls=":",  lw=0.9, alpha=0.7)

    max_bracket_tier = -1
    ylim_top = data_ylim_top
    if pairwise_pvals_dict:
        pair_combos = sorted(itertools.combinations(range(len(pair_order)), 2),
                             key=lambda t: t[1]-t[0])
        tier_used = []
        for i1, i2 in pair_combos:
            p1, p2 = pair_order[i1], pair_order[i2]
            pval = pairwise_pvals_dict.get((p1, p2), pairwise_pvals_dict.get((p2, p1), np.nan))
            if np.isnan(pval) or pval >= 0.05:
                continue
            level = 0
            while any(l == level and not (i2 < a or b < i1) for a, b, l in tier_used):
                level += 1
            tier_used.append((i1, i2, level))
            label = pval_label(pval)
            if bracket_in_data_space:
                y = max(102.0, band_max + BRACKET_BASE_PAD) + level * BRACKET_TIER_STEP
                draw_bracket_horizontal_data(ax, i1, i2, y, label)
            else:
                draw_bracket_above_axes(ax, i1, i2, level, label)
        if tier_used:
            max_bracket_tier = max(l for _, _, l in tier_used)
            if bracket_in_data_space:
                ylim_top = min(
                    BRACKET_YLIM_CAP,
                    max(102.0, band_max + BRACKET_BASE_PAD)
                    + max_bracket_tier * BRACKET_TIER_STEP + 6.0,
                )

    ax.set_xticks(range(len(pair_order)))
    tick_fs = COMBINED_FONT_TICK if bracket_in_data_space else 11
    label_fs = COMBINED_FONT_LABEL if bracket_in_data_space else 11
    ax.set_xticklabels(pair_order, fontsize=tick_fs)
    ax.set_xlim(-0.55, len(pair_order) - 0.45)
    ax.set_ylim(0, ylim_top)
    y_tick_vals = list(range(0, data_ylim_top + 1, 20))
    ax.set_yticks(y_tick_vals)
    ax.tick_params(axis="both", labelsize=tick_fs, length=0)
    if show_ylabel:
        ax.set_ylabel(
            "Discrimination Accuracy (%)" if bracket_in_data_space else "Accuracy (%)",
            fontsize=label_fs,
            labelpad=FIG_AXIS_LABELPAD if bracket_in_data_space else None,
        )
    ax.set_xlabel(
        "Force pair (g)", fontsize=label_fs,
        labelpad=FIG_AXIS_LABELPAD if bracket_in_data_space else None,
    )
    ax.spines[["top", "right"]].set_visible(False)
    if bracket_in_data_space:
        apply_combined_axis_spines(ax)
        apply_accuracy_y_spine_bounds(ax)
        add_inward_tick_guides(ax, n_x=len(pair_order), y_ticks=y_tick_vals, labelsize=tick_fs)
        apply_accuracy_y_spine_bounds(ax)
        apply_combined_axis_spines(ax)
    if dot_label:
        ax.text(0.02, 0.98, dot_label, transform=ax.transAxes,
                ha="left", va="top", fontsize=9, color="#555")
    if bracket_in_data_space:
        return max_bracket_tier, ylim_top
    return max_bracket_tier


def _apply_accuracy_figure_layout(fig, max_bracket_tier, title, *, panel_titles=None):
    """Title at figure top; brackets in margin below title; axes y-axis 0–100."""
    title_pad = 0.055 if title else 0.02
    panel_pad = 0.045 if panel_titles else 0.0
    bracket_pad = 0.06 + max(0, max_bracket_tier) * 0.052
    axes_top = 1.0 - title_pad - panel_pad - bracket_pad
    fig.subplots_adjust(top=max(axes_top, 0.55), wspace=0.12)
    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold", y=1.0, va="top")
    if panel_titles:
        for ax, panel_title in zip(fig.axes, panel_titles):
            ax.set_title(panel_title, fontsize=11, fontweight="bold", pad=10)


def _draw_accuracy_panel(ax, subj_acc_df, pair_order, pairwise_pvals_dict, *, show_ylabel=True,
                         bracket_in_data_space=False, box_color="#dde6f0",
                         box_width=0.42, jitter_width=0.12):
    subj_acc_sorted = subj_acc_df[subj_acc_df["pair_label"].isin(pair_order)].copy()
    values_by_pair = {
        pair: subj_acc_sorted.loc[subj_acc_sorted["pair_label"] == pair, "accuracy"].values
        for pair in pair_order
    }
    return _draw_pair_accuracy_boxplot(
        ax, pair_order, values_by_pair,
        pairwise_pvals_dict=pairwise_pvals_dict,
        show_ylabel=show_ylabel,
        bracket_in_data_space=bracket_in_data_space,
        box_color=box_color,
        box_width=box_width,
        jitter_width=jitter_width,
    )


def _combined_accuracy_legend_handles(band_specs):
    band_colors = {"Low": COLOR_LOW_BAND, "High": COLOR_HIGH_BAND}
    return [
        mpatches.Patch(
            facecolor=band_colors.get(spec["band_label"], "#dde6f0"),
            edgecolor="black",
            label=band_title_text(spec["band_label"], spec["title_ref"], spec["n_subj"]),
        )
        for spec in band_specs
    ]


def save_combined_accuracy_by_pair(band_specs):
    """Low | High accuracy-by-pair in one 2-column figure (2102 px wide)."""
    fig, axes = plt.subplots(
        1, len(band_specs),
        figsize=ACC_BY_PAIR_FIGSIZE,
        sharey=True,
    )
    if len(band_specs) == 1:
        axes = [axes]

    band_colors = {"Low": COLOR_LOW_BAND, "High": COLOR_HIGH_BAND}
    shared_ylim = 100.0
    for ax, spec in zip(axes, band_specs):
        box_w = combined_panel_box_width(len(spec["pair_order"]))
        jitter_w = STRIP_JITTER_REF * box_w / ATD_FIG2_DODGED_BOX_WIDTH
        _, ylim_top = _draw_accuracy_panel(
            ax, spec["subj_acc"], spec["pair_order"], spec["pairwise_pvals"],
            show_ylabel=(ax is axes[0]),
            bracket_in_data_space=True,
            box_color=band_colors.get(spec["band_label"], "#dde6f0"),
            box_width=box_w,
            jitter_width=jitter_w,
        )
        shared_ylim = max(shared_ylim, ylim_top)

    for ax in axes:
        ax.set_ylim(0, shared_ylim)

    fig.legend(
        handles=_combined_accuracy_legend_handles(band_specs),
        loc="upper center",
        bbox_to_anchor=(0.5, FIG_LEGEND_ANCHOR_Y),
        bbox_transform=fig.transFigure,
        ncol=len(band_specs),
        frameon=False,
        fontsize=COMBINED_FONT_LEGEND,
        columnspacing=2.0,
        handletextpad=0.5,
        handlelength=1.6,
    )
    fig.subplots_adjust(
        left=0.07, right=0.98, top=FIG_PANEL_TOP_FRAC, bottom=0.12, wspace=0.10,
    )
    out_path = os.path.join(OUTPUT_DIR, "sd_accuracy_by_pair_2col.png")
    save_png_at_width(
        fig, out_path,
        width_px=EXPORT_WIDTH_2COL,
        height_px=EXPORT_HEIGHT_2COL,
        dpi=SAVE_DPI_COMBINED,
        pad_inches=0.05,
    )
    plt.close(fig)
    print(
        f"Saved → sd_accuracy_by_pair_2col.png  "
        f"({EXPORT_WIDTH_2COL}×{EXPORT_HEIGHT_2COL} px)"
    )


def _save_subject_accuracy_by_pair(df_sub, subject, pair_order, band_title, out_path):
    """Per-subject sd_accuracy_by_pair-style figure (dots = regions A–F)."""
    region_acc = (
        df_sub.groupby(["pair_label", "Region"])["correct"]
        .mean()
        .reset_index()
    )
    values_by_pair = {
        pair: region_acc.loc[region_acc["pair_label"] == pair, "correct"].values
        for pair in pair_order
    }
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_title = f"{subject} — Overall Accuracy — Same/Different 2AFC ({band_title})"
    max_tier = _draw_pair_accuracy_boxplot(
        ax, pair_order, values_by_pair,
        dot_label="each dot = one region (A–F)",
    )
    _apply_accuracy_figure_layout(fig, max_tier, plot_title)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {os.path.relpath(out_path, OUTPUT_DIR)}")


def _draw_subject_bars(ax, pair_order, values_pct, title, bar_color=C1):
    """Single-subject bar chart (no group stats — one person only)."""
    xs = np.arange(len(pair_order))
    bars = ax.bar(xs, values_pct, width=0.55, color=bar_color, edgecolor="black", alpha=0.85)
    for bar, val in zip(bars, values_pct):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 2, f"{val:.0f}%",
                    ha="center", va="bottom", fontsize=9)
    ax.axhline(JND_PCT, color="black", ls="--", lw=1.2, alpha=0.8)
    ax.axhline(CHANCE_PCT, color="gray", ls=":", lw=0.9, alpha=0.7)
    ax.set_xticks(xs)
    ax.set_xticklabels(pair_order, fontsize=11)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_xlabel("Force pair (g)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)


def run_subject_analysis(df_band, band_label, pair_order, out_suffix, title_ref):
    """Per-subject accuracy tables and figures (one participant at a time)."""
    subj_root = os.path.join(OUTPUT_DIR, "per_subject")
    os.makedirs(subj_root, exist_ok=True)

    band_title = band_title_text(band_label, title_ref, 1)
    summary_rows = []

    for subject in sorted(df_band["Subject"].unique()):
        df_sub = df_band[df_band["Subject"] == subject].copy()
        subj_dir = os.path.join(subj_root, subject)
        os.makedirs(subj_dir, exist_ok=True)

        overall = (
            df_sub.groupby("pair_label")["correct"]
            .agg(n_trials="count", accuracy="mean")
            .reindex(pair_order)
        )
        same_acc = df_sub[df_sub["GroundTruth"] == "SAME"].groupby("pair_label")["correct"].mean()
        diff_acc = df_sub[df_sub["GroundTruth"] == "DIFFERENT"].groupby("pair_label")["correct"].mean()

        print(f"\n--- {subject} | {band_title} ---")
        for pair in pair_order:
            row = overall.loc[pair] if pair in overall.index else None
            if row is None or pd.isna(row["n_trials"]):
                continue
            acc_pct = row["accuracy"] * 100
            same_pct = same_acc.get(pair, np.nan) * 100 if pair in same_acc.index else np.nan
            diff_pct = diff_acc.get(pair, np.nan) * 100 if pair in diff_acc.index else np.nan
            same_str = f"{same_pct:5.1f}%" if not np.isnan(same_pct) else "  n/a "
            diff_str = f"{diff_pct:5.1f}%" if not np.isnan(diff_pct) else "  n/a "
            print(f"  {pair:8s}  overall {acc_pct:5.1f}%  "
                  f"SAME {same_str}  DIFF {diff_str}  "
                  f"(n={int(row['n_trials'])})")
            summary_rows.append({
                "Subject": subject,
                "band": band_label,
                "pair_label": pair,
                "n_trials": int(row["n_trials"]),
                "accuracy_pct": acc_pct,
                "same_accuracy_pct": same_pct,
                "different_accuracy_pct": diff_pct,
            })

        overall_vals = [
            overall.loc[p, "accuracy"] * 100 if p in overall.index and not pd.isna(overall.loc[p, "accuracy"]) else np.nan
            for p in pair_order
        ]
        same_vals = [same_acc.get(p, np.nan) * 100 if p in same_acc.index else np.nan for p in pair_order]
        diff_vals = [diff_acc.get(p, np.nan) * 100 if p in diff_acc.index else np.nan for p in pair_order]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        _draw_subject_bars(
            ax1, pair_order, overall_vals,
            f"{subject} — Overall ({band_title})",
        )

        xs = np.arange(len(pair_order))
        bw = 0.35
        ax2.bar(xs - bw / 2, same_vals, width=bw, color=C_SAME, edgecolor="black", alpha=0.85, label="SAME")
        ax2.bar(xs + bw / 2, diff_vals, width=bw, color=C_DIFF, edgecolor="black", alpha=0.85, label="DIFFERENT")
        ax2.axhline(JND_PCT, color="black", ls="--", lw=1.2, alpha=0.8)
        ax2.axhline(CHANCE_PCT, color="gray", ls=":", lw=0.9, alpha=0.7)
        ax2.set_xticks(xs)
        ax2.set_xticklabels(pair_order, fontsize=11)
        ax2.set_ylim(0, 110)
        ax2.set_ylabel("Accuracy (%)", fontsize=11)
        ax2.set_xlabel("Force pair (g)", fontsize=11)
        ax2.set_title(f"{subject} — SAME vs DIFFERENT ({band_title})", fontsize=12, fontweight="bold")
        ax2.legend(frameon=False, fontsize=10)
        ax2.spines[["top", "right"]].set_visible(False)

        plt.tight_layout()
        out_name = f"sd_overview{out_suffix}.png"
        fig.savefig(os.path.join(subj_dir, out_name), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved → per_subject/{subject}/{out_name}")

        pair_out = os.path.join(subj_dir, f"sd_accuracy_by_pair{out_suffix}.png")
        _save_subject_accuracy_by_pair(
            df_sub, subject, pair_order, band_title, pair_out,
        )

        bias_out = os.path.join(subj_dir, f"sd_response_bias{out_suffix}.png")
        run_response_bias_overview(
            df_sub, bias_out,
            f"Response Bias — {subject}  ({band_title})",
            print_header=f"Response bias | {subject} | {band_title}",
        )

        dist_out = os.path.join(subj_dir, f"sd_response_by_pair{out_suffix}.png")
        save_response_distribution_by_pair(
            df_sub, pair_order,
            f"{subject} — Response by Pair × Ground Truth ({band_title})",
            dist_out,
        )
        save_same_trial_response_by_pair(
            df_sub, pair_order,
            f"{subject} — Responses on SAME Trials Only ({band_title})",
            os.path.join(subj_dir, f"sd_same_trial_response{out_suffix}.png"),
        )

    if summary_rows:
        summary_path = os.path.join(subj_root, f"accuracy_by_subject{out_suffix}.csv")
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
        print(f"Saved summary → per_subject/accuracy_by_subject{out_suffix}.csv")


def run_band_analysis(df_band, band_label, pair_order, out_suffix, title_ref):
    """Run SDT, GEE, and figures for one force band (Low ref=1g or High ref=26g)."""
    print(f"\n{'='*60}")
    print(f"Band: {band_label} (ref = {title_ref})  |  trials = {len(df_band)}  |  subjects = {df_band['Subject'].nunique()}")
    print(f"Pair order: {pair_order}")

    subj_acc = (
        df_band.groupby(["Subject", "pair_label"])["correct"]
        .mean().reset_index()
        .rename(columns={"correct": "accuracy"})
    )

    sdt_rows = []
    for (pair, grp), sub in df_band[df_band["region_group"].notna()].groupby(["pair_label", "region_group"]):
        sdt = compute_sdt(sub)
        sdt_rows.append({"pair_label": pair, "region_group": grp, **sdt})
    df_sdt = pd.DataFrame(sdt_rows)
    print("\nSDT summary (pooled across subjects):")
    print(df_sdt.to_string(index=False))
    df_sdt.to_csv(os.path.join(OUTPUT_DIR, f"sdt_summary{out_suffix}.csv"), index=False)

    order_rows = []
    for pair in pair_order:
        for grp in ["On-nail", "Off-nail"]:
            sub = df_band[(df_band["pair_label"] == pair) & (df_band["region_group"] == grp)]
            acc_rc = sub.loc[sub["TrialType"] == "diff_rc", "correct"].mean()
            acc_cr = sub.loc[sub["TrialType"] == "diff_cr", "correct"].mean()
            order_rows.append({
                "pair_label": pair, "region_group": grp,
                "acc_diff_rc": acc_rc * 100 if not np.isnan(acc_rc) else np.nan,
                "acc_diff_cr": acc_cr * 100 if not np.isnan(acc_cr) else np.nan,
                "delta_rc_minus_cr": (acc_rc - acc_cr) * 100 if not (np.isnan(acc_rc) or np.isnan(acc_cr)) else np.nan,
            })
    df_order = pd.DataFrame(order_rows)
    print("\nOrder effect (diff_rc vs diff_cr accuracy %):")
    print(df_order.to_string(index=False))
    df_order.to_csv(os.path.join(OUTPUT_DIR, f"order_effect{out_suffix}.csv"), index=False)

    pairwise_pvals = run_gee_pairwise(df_band, subj_acc, pair_order)
    region_pvals   = run_gee_region(df_band, pair_order)

    subj_acc_split = (
        df_band.groupby(["Subject", "pair_label", "GroundTruth"])["correct"]
        .mean().reset_index()
        .rename(columns={"correct": "accuracy"})
    )

    n_subj = df_band["Subject"].nunique()
    band_title = band_title_text(band_label, title_ref, n_subj)

    OFFSET = 0.22
    BW_SD  = 0.20
    fig, ax = plt.subplots(figsize=(10, 6))
    band_max = 0.0
    for xi, pair in enumerate(pair_order):
        for gi, (gt, color) in enumerate([("SAME", C_SAME), ("DIFFERENT", C_DIFF)]):
            xp = xi + OFFSET * (gi - 0.5)
            vals = subj_acc_split.loc[
                (subj_acc_split["pair_label"] == pair) &
                (subj_acc_split["GroundTruth"] == gt),
                "accuracy"
            ].values * 100
            if len(vals) == 0:
                continue
            band_max = max(band_max, vals.max())
            bp = ax.boxplot([vals], positions=[xp], widths=BW_SD,
                             patch_artist=True, showfliers=False)
            bp["boxes"][0].set_facecolor(color); bp["boxes"][0].set_alpha(0.35)
            bp["boxes"][0].set_edgecolor("black")
            bp["medians"][0].set_color(RED); bp["medians"][0].set_linewidth(2)
            jx = xp + jitter_x(len(vals), width=BW_SD * 0.5)
            ax.scatter(jx, vals, color=color, alpha=0.75, s=22, zorder=3)

    ax.axhline(JND_PCT,    color="black", ls="--", lw=1.2, alpha=0.8)
    ax.axhline(CHANCE_PCT, color="gray",  ls=":",  lw=0.9, alpha=0.7)
    ax.set_xticks(range(len(pair_order)))
    ax.set_xticklabels(pair_order, fontsize=11)
    ax.set_ylim(0, min(115, band_max + 20))
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_xlabel("Force pair (g)", fontsize=11)
    ax.set_title(f"SAME vs DIFFERENT Trial Accuracy — {band_title}", fontsize=12, fontweight="bold")
    ax.legend(handles=[
        mpatches.Patch(facecolor=C_SAME, edgecolor="black", alpha=0.55, label="SAME trials"),
        mpatches.Patch(facecolor=C_DIFF, edgecolor="black", alpha=0.55, label="DIFFERENT trials"),
    ], loc="lower right", frameon=False, fontsize=10)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    out_a2 = f"sd_accuracy_split{out_suffix}.png"
    fig.savefig(os.path.join(OUTPUT_DIR, out_a2), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_a2}")

    df_reg = df_band[df_band["region_group"].notna()]
    subj_acc_reg = (
        df_reg.groupby(["Subject", "pair_label", "region_group"])["correct"]
        .mean().reset_index().rename(columns={"correct": "accuracy"})
    )

    BW = 0.20
    fig, ax = plt.subplots(figsize=(10, 6))
    band_max = 0.0
    for xi, pair in enumerate(pair_order):
        for gi, (grp, color) in enumerate([("On-nail", C_ON), ("Off-nail", C_OFF)]):
            xp = xi + OFFSET * (gi - 0.5)
            vals = subj_acc_reg.loc[
                (subj_acc_reg["pair_label"]==pair) & (subj_acc_reg["region_group"]==grp),
                "accuracy"].values * 100
            if len(vals) == 0: continue
            band_max = max(band_max, vals.max())
            bp = ax.boxplot([vals], positions=[xp], widths=BW,
                             patch_artist=True, showfliers=False)
            bp["boxes"][0].set_facecolor(color); bp["boxes"][0].set_edgecolor("black")
            bp["medians"][0].set_color(RED); bp["medians"][0].set_linewidth(2)
            jx = xp + jitter_x(len(vals), width=BW*0.5)
            ax.scatter(jx, vals,
                       color=C_ON if grp=="On-nail" else "#5b7fa6",
                       alpha=0.6, s=18, zorder=3)

        pval = region_pvals.get(pair, np.nan)
        y_b = band_max + 10
        ax.plot([xi-OFFSET*0.5, xi-OFFSET*0.5, xi+OFFSET*0.5, xi+OFFSET*0.5],
                [y_b, y_b+3, y_b+3, y_b], color=RED, lw=1.2)
        ax.text(xi, y_b+4.5, pval_label(pval), ha="center", va="bottom",
                fontsize=9, color=RED, fontweight="bold")

    ax.axhline(JND_PCT, color="black", ls="--", lw=1.2, alpha=0.8)
    ax.axhline(CHANCE_PCT, color="gray", ls=":", lw=0.9, alpha=0.7)
    ax.set_xticks(range(len(pair_order))); ax.set_xticklabels(pair_order, fontsize=11)
    ax.set_ylim(0, band_max + 28)
    ax.set_ylabel("Discrimination Accuracy (%)", fontsize=11)
    ax.set_xlabel("Force pair (g)", fontsize=11)
    ax.set_title(f"On-nail (C+D) vs Off-nail (A+F) — {band_title}", fontsize=12, fontweight="bold")
    ax.legend(handles=[
        mpatches.Patch(facecolor=C_ON,  edgecolor="black", label="On-nail (C+D)"),
        mpatches.Patch(facecolor=C_OFF, edgecolor="black", label="Off-nail (A+F)"),
    ], loc="upper left", frameon=False, fontsize=10)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    out_b = f"sd_onnail_vs_offnail{out_suffix}.png"
    fig.savefig(os.path.join(OUTPUT_DIR, out_b), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_b}")

    df_sdt_plot = df_sdt[df_sdt["pair_label"].isin(pair_order)].copy()
    fig, ax = plt.subplots(figsize=(9, 5))
    xs = np.arange(len(pair_order))
    for gi, (grp, color) in enumerate([("On-nail", C_ON), ("Off-nail", C_OFF)]):
        vals = [df_sdt_plot.loc[(df_sdt_plot["pair_label"]==p) & (df_sdt_plot["region_group"]==grp), "d_prime"].values
                for p in pair_order]
        ys = [v[0] if len(v) else np.nan for v in vals]
        ax.bar(xs + OFFSET*(gi-0.5), ys, width=BW*1.1,
               color=color, edgecolor="black", label=grp)
        for x, y in zip(xs + OFFSET*(gi-0.5), ys):
            if not np.isnan(y):
                ax.text(x, y + 0.05, f"{y:.2f}", ha="center", fontsize=8)

    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(xs); ax.set_xticklabels(pair_order, fontsize=11)
    ax.set_ylabel("d′  (sensitivity)", fontsize=11)
    ax.set_xlabel("Force pair (g)", fontsize=11)
    ax.set_title(f"Signal Detection Theory — d′ ({band_title})", fontsize=12, fontweight="bold")
    ax.legend(frameon=False, fontsize=10)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    out_c = f"sd_dprime_by_pair_region{out_suffix}.png"
    fig.savefig(os.path.join(OUTPUT_DIR, out_c), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_c}")

    fig, ax = plt.subplots(figsize=(9, 5))
    df_order_plot = df_order[df_order["pair_label"].isin(pair_order)]
    for gi, (grp, color) in enumerate([("On-nail", C_ON), ("Off-nail", C_OFF)]):
        sub = df_order_plot[df_order_plot["region_group"]==grp]
        xs_pairs = [pair_order.index(p) for p in sub["pair_label"] if p in pair_order]
        rc_vals = sub["acc_diff_rc"].values
        cr_vals = sub["acc_diff_cr"].values
        x_pos = np.array(xs_pairs, dtype=float) + OFFSET*(gi-0.5)
        ax.plot(x_pos, rc_vals, "o-", color=color, lw=2, label=f"{grp} diff_rc")
        ax.plot(x_pos, cr_vals, "s--", color=color, lw=1.5, alpha=0.7, label=f"{grp} diff_cr")

    ax.axhline(CHANCE_PCT, color="gray", ls=":", lw=0.9)
    ax.axhline(JND_PCT, color="black", ls="--", lw=1.1, alpha=0.8)
    ax.set_xticks(range(len(pair_order))); ax.set_xticklabels(pair_order, fontsize=11)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Accuracy (%)  — DIFFERENT trials only", fontsize=11)
    ax.set_xlabel("Force pair (g)", fontsize=11)
    ax.set_title(f"Order Effect: diff_rc vs diff_cr ({band_title})", fontsize=12, fontweight="bold")
    ax.legend(frameon=False, fontsize=9, ncol=2)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    out_d = f"sd_order_effect{out_suffix}.png"
    fig.savefig(os.path.join(OUTPUT_DIR, out_d), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_d}")

    run_response_bias_overview(
        df_band,
        os.path.join(OUTPUT_DIR, f"sd_response_bias_overview{out_suffix}.png"),
        f"Response Bias Overview — Same/Different 2AFC ({band_title})",
        print_header=f"Response bias (pooled) | {band_title}",
    )
    save_response_distribution_by_pair(
        df_band, pair_order,
        f"Response Distribution by Force Pair — SAME vs DIFFERENT trials ({band_title})",
        os.path.join(OUTPUT_DIR, f"sd_response_by_pair{out_suffix}.png"),
        csv_path=os.path.join(OUTPUT_DIR, f"response_distribution_by_pair{out_suffix}.csv"),
    )
    rows = _pair_response_distribution_rows(df_band, pair_order)
    print("\nResponse distribution by pair (pooled):")
    for r in rows:
        print(f"  {r['pair_label']:8s}  GT={r['ground_truth']:9s}  "
              f"resp SAME {r['pct_resp_same']:5.1f}% (n={r['n_resp_same']:3d})  "
              f"resp DIFF {r['pct_resp_diff']:5.1f}% (n={r['n_resp_diff']:3d})  "
              f"[trials={r['n_trials']}]")
    save_same_trial_response_by_pair(
        df_band, pair_order,
        f"Responses on SAME Trials Only — by Force Pair ({band_title})",
        os.path.join(OUTPUT_DIR, f"sd_same_trial_response{out_suffix}.png"),
        csv_path=os.path.join(OUTPUT_DIR, f"same_trial_response_by_pair{out_suffix}.csv"),
    )
    same_rows = _same_trial_response_rows(df_band, pair_order)
    print("\nSAME-trial response distribution by pair (pooled):")
    for r in same_rows:
        print(f"  {r['pair_label']:8s}  resp SAME {r['pct_resp_same']:5.1f}% (n={r['n_resp_same']:3d})  "
              f"resp DIFF {r['pct_resp_diff']:5.1f}% (n={r['n_resp_diff']:3d})  "
              f"[SAME trials={r['n_trials']}]")
    run_subject_analysis(df_band, band_label, pair_order, out_suffix, title_ref)

    return {
        "band_label": band_label,
        "title_ref": title_ref,
        "pair_order": pair_order,
        "subj_acc": subj_acc,
        "pairwise_pvals": pairwise_pvals,
        "n_subj": n_subj,
    }


accuracy_band_specs = []
for band_label, cfg in BAND_CONFIG.items():
    df_band = df[df["band"] == band_label].copy()
    if df_band.empty:
        print(f"\nNo data for {band_label} band — skipping")
        continue
    pair_order = fix_order(cfg["pair_order"], df_band["pair_label"].unique().tolist())
    spec = run_band_analysis(df_band, band_label, pair_order, cfg["suffix"], cfg["title_ref"])
    if spec:
        accuracy_band_specs.append(spec)

if accuracy_band_specs:
    save_combined_accuracy_by_pair(accuracy_band_specs)

run_response_bias_overview(
    df,
    os.path.join(OUTPUT_DIR, "sd_response_bias_overview_all.png"),
    f"Response Bias Overview — Same/Different 2AFC (All Bands, n = {df['Subject'].nunique()})",
    print_header="Response bias (all bands pooled)",
)

print(f"\nAll outputs saved to:\n  {OUTPUT_DIR}")