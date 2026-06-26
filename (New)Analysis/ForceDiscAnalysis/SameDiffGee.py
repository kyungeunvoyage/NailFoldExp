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


def jitter_x(n, width=0.12, seed=42):
    return (np.random.default_rng(seed).random(n) - 0.5) * width


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


def _draw_pair_accuracy_boxplot(ax, pair_order, values_by_pair, title, *,
                                pairwise_pvals_dict=None, dot_label=None):
    """Boxplot + jittered scatter per force pair (shared by group & single-subject plots)."""
    band_max = 0.0
    for xi, pair in enumerate(pair_order):
        vals = np.asarray(values_by_pair.get(pair, []), dtype=float) * 100
        if len(vals) == 0:
            continue
        band_max = max(band_max, float(np.nanmax(vals)))
        bp = ax.boxplot([vals], positions=[xi], widths=0.42,
                         patch_artist=True, showfliers=False)
        bp["boxes"][0].set_facecolor("#dde6f0"); bp["boxes"][0].set_edgecolor("black")
        bp["medians"][0].set_color(RED); bp["medians"][0].set_linewidth(2)
        jx = xi + jitter_x(len(vals))
        ax.scatter(jx, vals, color=C1, alpha=0.6, s=20, zorder=3)

    ax.axhline(JND_PCT,    color="black", ls="--", lw=1.2, alpha=0.8)
    ax.axhline(CHANCE_PCT, color="gray",  ls=":",  lw=0.9, alpha=0.7)
    ax.text(len(pair_order)-0.5, JND_PCT+1.5,    f"threshold ({JND_PCT:.0f}%)",
            ha="right", fontsize=9, color="#333")
    ax.text(len(pair_order)-0.5, CHANCE_PCT+1.5, f"chance ({CHANCE_PCT:.0f}%)",
            ha="right", fontsize=9, color="#888")

    y_top = max(105.0, band_max + 8)
    if pairwise_pvals_dict:
        pair_combos = sorted(itertools.combinations(range(len(pair_order)), 2),
                             key=lambda t: t[1]-t[0])
        tier_step = 11
        tier_used = []
        y_bracket = y_top
        for i1, i2 in pair_combos:
            p1, p2 = pair_order[i1], pair_order[i2]
            pval = pairwise_pvals_dict.get((p1, p2), pairwise_pvals_dict.get((p2, p1), np.nan))
            if np.isnan(pval) or pval >= 0.05:
                continue
            level = 0
            while any(l == level and not (i2 < a or b < i1) for a, b, l in tier_used):
                level += 1
            tier_used.append((i1, i2, level))
            draw_bracket(ax, i1, i2, y_bracket + level * tier_step, pval_label(pval))
        y_top = y_bracket + max((l for _, _, l in tier_used), default=-1) * tier_step + 18

    ax.set_xticks(range(len(pair_order)))
    ax.set_xticklabels(pair_order, fontsize=11)
    ax.set_ylim(0, y_top)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_xlabel("Force pair (g)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    if dot_label:
        ax.text(0.02, 0.98, dot_label, transform=ax.transAxes,
                ha="left", va="top", fontsize=9, color="#555")


def _draw_accuracy_panel(ax, subj_acc_df, pair_order, title, pairwise_pvals_dict):
    subj_acc_sorted = subj_acc_df[subj_acc_df["pair_label"].isin(pair_order)].copy()
    values_by_pair = {
        pair: subj_acc_sorted.loc[subj_acc_sorted["pair_label"] == pair, "accuracy"].values
        for pair in pair_order
    }
    _draw_pair_accuracy_boxplot(
        ax, pair_order, values_by_pair, title,
        pairwise_pvals_dict=pairwise_pvals_dict,
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
    _draw_pair_accuracy_boxplot(
        ax, pair_order, values_by_pair,
        f"{subject} — Overall Accuracy — Same/Different 2AFC ({band_title})",
        dot_label="each dot = one region (A–F)",
    )
    plt.tight_layout()
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

    band_title = f"{band_label} band (ref = {title_ref})"
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

    band_title = f"{band_label} band (ref = {title_ref})"

    fig, ax = plt.subplots(figsize=(8, 6))
    _draw_accuracy_panel(
        ax, subj_acc, pair_order,
        f"Overall Accuracy — Same/Different 2AFC ({band_title})",
        pairwise_pvals,
    )
    plt.tight_layout()
    out_a1 = f"sd_accuracy_by_pair{out_suffix}.png"
    fig.savefig(os.path.join(OUTPUT_DIR, out_a1), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_a1}")

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
    run_subject_analysis(df_band, band_label, pair_order, out_suffix, title_ref)


for band_label, cfg in BAND_CONFIG.items():
    df_band = df[df["band"] == band_label].copy()
    if df_band.empty:
        print(f"\nNo data for {band_label} band — skipping")
        continue
    pair_order = fix_order(cfg["pair_order"], df_band["pair_label"].unique().tolist())
    run_band_analysis(df_band, band_label, pair_order, cfg["suffix"], cfg["title_ref"])

run_response_bias_overview(
    df,
    os.path.join(OUTPUT_DIR, "sd_response_bias_overview_all.png"),
    "Response Bias Overview — Same/Different 2AFC (All Bands, All Participants)",
    print_header="Response bias (all bands pooled)",
)

print(f"\nAll outputs saved to:\n  {OUTPUT_DIR}")