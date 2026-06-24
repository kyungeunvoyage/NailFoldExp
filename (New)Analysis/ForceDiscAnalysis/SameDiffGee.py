"""
Force Discrimination – Same/Different 2AFC GEE Analysis
=========================================================
Adapted from the original "which is stronger" GEE pairwise script.

CSV column assumptions (P*_ForceDiscrimination_SameDiff.csv)
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
FILE_PATTERN = "/Users/kyungeunjung/NailFoldExp/Data/(FD)CurData/P*_ForceDiscrimination_SameDiff.csv"
OUTPUT_DIR   = "/Users/kyungeunjung/NailFoldExp/(New)Analysis/ForceDiscAnalysis/Output/SameDiff_GEE"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CHANCE_PCT = 50.0
JND_PCT    = 75.0

# ── Load data ─────────────────────────────────────────────────────────────────
import re

all_files = glob.glob(FILE_PATTERN)
if not all_files:
    raise FileNotFoundError(f"No CSV files found matching:\n  {FILE_PATTERN}")

def subject_number(filepath):
    m = re.search(r'P(\d+)', os.path.basename(filepath))
    return int(m.group(1)) if m else 0

files = sorted([f for f in all_files if subject_number(f) > 74])
if not files:
    raise ValueError(
        f"No files with subject number > 75 found.\n"
        f"Files present: {sorted(os.path.basename(f) for f in all_files)}"
    )
print(f"Total files found  : {len(all_files)}")
print(f"Files with ID > 75 : {len(files)}")
print(f"  → {[os.path.basename(f) for f in files]}")

df = pd.concat(
    [pd.read_csv(f, encoding="utf-8-sig") for f in sorted(files)],
    ignore_index=True,
)

# Use the pre-computed IsCorrect column directly
df["correct"] = df["IsCorrect"].astype(int)

df["pair_label"] = df.apply(
    lambda r: f"{min(r['Reference'], r['Comparison']):g}–{max(r['Reference'], r['Comparison']):g}",
    axis=1,
)

# Region groups
ON_NAIL  = ["C", "D"]
OFF_NAIL = ["A", "F"]
df["region_group"] = df["Region"].map(
    {r: "On-nail"  for r in ON_NAIL} |
    {r: "Off-nail" for r in OFF_NAIL}
)

# Pair order
low_order = ["0.4–1", "0.6–1", "1–1.4", "1–2"]

# Verify labels in data
actual_labels = df["pair_label"].unique().tolist()
print("Pair labels found:", sorted(actual_labels))

def fix_order(order, actual):
    fixed = []
    for p in order:
        if p in actual:
            fixed.append(p)
        else:
            alt = [a for a in actual if set(a.replace("–","-").split("-")) == set(p.replace("–","-").split("-"))]
            fixed.append(alt[0] if alt else p)
    return fixed

low_order = fix_order(low_order, actual_labels)
print("Using pair order:", low_order)

# ── Per-subject accuracy per pair (all regions pooled) ────────────────────────
subj_acc = (
    df.groupby(["Subject", "pair_label"])["correct"]
    .mean().reset_index()
    .rename(columns={"correct": "accuracy"})
)

# ── SDT: Hit Rate, False Alarm Rate, d′, criterion c ─────────────────────────
def compute_sdt(sub_df):
    """
    sub_df: slice of df for one group (e.g. one pair x one region_group).
    Returns dict with H, FA, d_prime, criterion.
    Hit        = P(respond DIFFERENT | ground truth = DIFFERENT)
    False Alarm = P(respond DIFFERENT | ground truth = SAME)
    Hautus (1995) log-linear correction applied to avoid ±inf.
    """
    diff_trials = sub_df[sub_df["GroundTruth"] == "DIFFERENT"]
    same_trials = sub_df[sub_df["GroundTruth"] == "SAME"]
    n_diff = len(diff_trials)
    n_same = len(same_trials)
    if n_diff == 0 or n_same == 0:
        return {"H": np.nan, "FA": np.nan, "d_prime": np.nan, "criterion": np.nan}

    hits = (diff_trials["UserChoice"] == "DIFFERENT").sum()
    fas  = (same_trials["UserChoice"]  == "DIFFERENT").sum()

    # Hautus log-linear correction
    H  = (hits + 0.5) / (n_diff + 1)
    FA = (fas  + 0.5) / (n_same + 1)

    from scipy.stats import norm
    d_prime   = norm.ppf(H) - norm.ppf(FA)
    criterion = -0.5 * (norm.ppf(H) + norm.ppf(FA))
    return {"H": H, "FA": FA, "d_prime": d_prime, "criterion": criterion}

# SDT per pair × region_group
sdt_rows = []
for (pair, grp), sub in df[df["region_group"].notna()].groupby(["pair_label", "region_group"]):
    sdt = compute_sdt(sub)
    sdt_rows.append({"pair_label": pair, "region_group": grp, **sdt})
df_sdt = pd.DataFrame(sdt_rows)
print("\nSDT summary (pooled across subjects):")
print(df_sdt.to_string(index=False))
df_sdt.to_csv(os.path.join(OUTPUT_DIR, "sdt_summary.csv"), index=False)

# ── Order effect: diff_rc vs diff_cr ─────────────────────────────────────────
order_rows = []
for pair in low_order:
    for grp in ["On-nail", "Off-nail"]:
        sub = df[(df["pair_label"] == pair) & (df["region_group"] == grp)]
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
df_order.to_csv(os.path.join(OUTPUT_DIR, "order_effect.csv"), index=False)

# ── GEE helpers ───────────────────────────────────────────────────────────────
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

def run_gee_pairwise(pair_order):
    """All pairwise contrasts across force pairs (accuracy ~ pair_dummy)."""
    results = {}
    subj_means = subj_acc.copy()

    for p1, p2 in itertools.combinations(pair_order, 2):
        if USE_GEE:
            chunk = df[df["pair_label"].isin([p1, p2])].copy()
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

def run_gee_region(pair_order):
    """On-nail vs Off-nail per force pair."""
    df_reg = df[df["region_group"].notna()].copy()
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

pairwise_pvals = run_gee_pairwise(low_order)
region_pvals   = run_gee_region(low_order)

# ── Plotting helpers ──────────────────────────────────────────────────────────
C1 = "#2166AC"
C_ON  = "#7FB3D3"
C_OFF = "#D3E9F5"
RED   = "#c0392b"

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

# ── Per-subject accuracy split by GroundTruth (SAME / DIFFERENT) ─────────────
subj_acc_split = (
    df.groupby(["Subject", "pair_label", "GroundTruth"])["correct"]
    .mean().reset_index()
    .rename(columns={"correct": "accuracy"})
)

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE A1 – Overall accuracy by force pair (SAME + DIFFERENT pooled)
# ═══════════════════════════════════════════════════════════════════════════════
def _draw_accuracy_panel(ax, subj_acc_df, pair_order, title, pairwise_pvals_dict):
    subj_acc_sorted = subj_acc_df[subj_acc_df["pair_label"].isin(pair_order)].copy()
    band_max = 0.0
    for xi, pair in enumerate(pair_order):
        vals = subj_acc_sorted.loc[subj_acc_sorted["pair_label"] == pair, "accuracy"].values * 100
        band_max = max(band_max, vals.max() if len(vals) else 0)
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

    pair_combos = sorted(itertools.combinations(range(len(pair_order)), 2),
                         key=lambda t: t[1]-t[0])
    y_bracket = max(105.0, band_max + 8)
    tier_step  = 11
    tier_used  = []
    for i1, i2 in pair_combos:
        p1, p2 = pair_order[i1], pair_order[i2]
        pval = pairwise_pvals_dict.get((p1,p2), pairwise_pvals_dict.get((p2,p1), np.nan))
        if np.isnan(pval) or pval >= 0.05: continue
        level = 0
        while any(l == level and not (i2 < a or b < i1) for a,b,l in tier_used):
            level += 1
        tier_used.append((i1, i2, level))
        draw_bracket(ax, i1, i2, y_bracket + level*tier_step, pval_label(pval))

    ax.set_xticks(range(len(pair_order)))
    ax.set_xticklabels(pair_order, fontsize=11)
    ax.set_ylim(0, y_bracket + max((l for _,_,l in tier_used), default=-1)*tier_step + 18)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_xlabel("Force pair (g)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.spines[["top","right"]].set_visible(False)

fig, ax = plt.subplots(figsize=(8, 6))
_draw_accuracy_panel(ax, subj_acc, low_order,
                     "Overall Accuracy — Same/Different 2AFC (SAME + DIFFERENT pooled)",
                     pairwise_pvals)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "sd_accuracy_by_pair.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved → sd_accuracy_by_pair.png")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE A2 – SAME vs DIFFERENT trial accuracy side-by-side per force pair
# ═══════════════════════════════════════════════════════════════════════════════
C_SAME = "#5B9BD5"   # blue  – SAME trials
C_DIFF = "#E07B39"   # orange – DIFFERENT trials
OFFSET = 0.22
BW_SD  = 0.20

fig, ax = plt.subplots(figsize=(10, 6))
band_max = 0.0

for xi, pair in enumerate(low_order):
    for gi, (gt, color, label) in enumerate([
            ("SAME",      C_SAME, "SAME trials"),
            ("DIFFERENT", C_DIFF, "DIFFERENT trials"),
    ]):
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
ax.text(len(low_order)-0.5, JND_PCT+1.5,    f"threshold ({JND_PCT:.0f}%)",
        ha="right", fontsize=9, color="#333")
ax.text(len(low_order)-0.5, CHANCE_PCT+1.5, f"chance ({CHANCE_PCT:.0f}%)",
        ha="right", fontsize=9, color="#888")

ax.set_xticks(range(len(low_order)))
ax.set_xticklabels(low_order, fontsize=11)
ax.set_ylim(0, min(115, band_max + 20))
ax.set_ylabel("Accuracy (%)", fontsize=11)
ax.set_xlabel("Force pair (g)", fontsize=11)
ax.set_title("SAME vs DIFFERENT Trial Accuracy — Same/Different 2AFC", fontsize=12, fontweight="bold")
ax.legend(handles=[
    mpatches.Patch(facecolor=C_SAME, edgecolor="black", alpha=0.55, label="SAME trials (correct = say SAME)"),
    mpatches.Patch(facecolor=C_DIFF, edgecolor="black", alpha=0.55, label="DIFFERENT trials (correct = say DIFFERENT)"),
], loc="lower right", frameon=False, fontsize=10)
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "sd_accuracy_split.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved → sd_accuracy_split.png")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE B – On-nail (C+D) vs Off-nail (A+F) side-by-side per pair
# ═══════════════════════════════════════════════════════════════════════════════
df_reg = df[df["region_group"].notna()]
subj_acc_reg = (
    df_reg.groupby(["Subject", "pair_label", "region_group"])["correct"]
    .mean().reset_index().rename(columns={"correct": "accuracy"})
)

fig, ax = plt.subplots(figsize=(10, 6))
OFFSET = 0.22
BW     = 0.20
band_max = 0.0

for xi, pair in enumerate(low_order):
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

    # Region bracket
    pval = region_pvals.get(pair, np.nan)
    y_b = band_max + 10
    ax.plot([xi-OFFSET*0.5, xi-OFFSET*0.5, xi+OFFSET*0.5, xi+OFFSET*0.5],
            [y_b, y_b+3, y_b+3, y_b], color=RED, lw=1.2)
    ax.text(xi, y_b+4.5, pval_label(pval), ha="center", va="bottom",
            fontsize=9, color=RED, fontweight="bold")

ax.axhline(JND_PCT, color="black", ls="--", lw=1.2, alpha=0.8)
ax.axhline(CHANCE_PCT, color="gray", ls=":", lw=0.9, alpha=0.7)
ax.set_xticks(range(len(low_order))); ax.set_xticklabels(low_order, fontsize=11)
ax.set_ylim(0, band_max + 28)
ax.set_ylabel("Discrimination Accuracy (%)", fontsize=11)
ax.set_xlabel("Force pair (g)", fontsize=11)
ax.set_title("On-nail (C+D) vs Off-nail (A+F) — Same/Different 2AFC", fontsize=12, fontweight="bold")
ax.legend(handles=[
    mpatches.Patch(facecolor=C_ON,  edgecolor="black", label="On-nail (C+D)"),
    mpatches.Patch(facecolor=C_OFF, edgecolor="black", label="Off-nail (A+F)"),
], loc="upper left", frameon=False, fontsize=10)
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "sd_onnail_vs_offnail.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved → sd_onnail_vs_offnail.png")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE C – d′ by pair × region group (bar chart)
# ═══════════════════════════════════════════════════════════════════════════════
df_sdt_plot = df_sdt[df_sdt["pair_label"].isin(low_order)].copy()

fig, ax = plt.subplots(figsize=(9, 5))
OFFSET = 0.22
xs = np.arange(len(low_order))
for gi, (grp, color) in enumerate([("On-nail", C_ON), ("Off-nail", C_OFF)]):
    vals = [df_sdt_plot.loc[(df_sdt_plot["pair_label"]==p) & (df_sdt_plot["region_group"]==grp), "d_prime"].values
            for p in low_order]
    ys = [v[0] if len(v) else np.nan for v in vals]
    ax.bar(xs + OFFSET*(gi-0.5), ys, width=BW*1.1,
           color=color, edgecolor="black", label=grp)
    for x, y in zip(xs + OFFSET*(gi-0.5), ys):
        if not np.isnan(y):
            ax.text(x, y + 0.05, f"{y:.2f}", ha="center", fontsize=8)

ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(xs); ax.set_xticklabels(low_order, fontsize=11)
ax.set_ylabel("d′  (sensitivity)", fontsize=11)
ax.set_xlabel("Force pair (g)", fontsize=11)
ax.set_title("Signal Detection Theory — d′ by Force Pair × Region Group", fontsize=12, fontweight="bold")
ax.legend(frameon=False, fontsize=10)
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "sd_dprime_by_pair_region.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved → sd_dprime_by_pair_region.png")

# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE D – Order effect: diff_rc vs diff_cr accuracy
# ═══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 5))
df_order_plot = df_order[df_order["pair_label"].isin(low_order)]
OFFSET = 0.22

for gi, (grp, color) in enumerate([("On-nail", C_ON), ("Off-nail", C_OFF)]):
    sub = df_order_plot[df_order_plot["region_group"]==grp]
    xs_pairs = [low_order.index(p) for p in sub["pair_label"] if p in low_order]
    rc_vals = sub["acc_diff_rc"].values
    cr_vals = sub["acc_diff_cr"].values
    x_pos = np.array(xs_pairs, dtype=float) + OFFSET*(gi-0.5)
    ax.plot(x_pos, rc_vals, "o-", color=color, lw=2, label=f"{grp} diff_rc")
    ax.plot(x_pos, cr_vals, "s--", color=color, lw=1.5, alpha=0.7, label=f"{grp} diff_cr")

ax.axhline(CHANCE_PCT, color="gray", ls=":", lw=0.9)
ax.axhline(JND_PCT, color="black", ls="--", lw=1.1, alpha=0.8)
ax.set_xticks(range(len(low_order))); ax.set_xticklabels(low_order, fontsize=11)
ax.set_ylim(0, 105)
ax.set_ylabel("Accuracy (%)  — DIFFERENT trials only", fontsize=11)
ax.set_xlabel("Force pair (g)", fontsize=11)
ax.set_title("Order Effect: ref→comp (diff_rc) vs comp→ref (diff_cr)", fontsize=12, fontweight="bold")
ax.legend(frameon=False, fontsize=9, ncol=2)
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "sd_order_effect.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved → sd_order_effect.png")

print(f"\nAll outputs saved to:\n  {OUTPUT_DIR}")