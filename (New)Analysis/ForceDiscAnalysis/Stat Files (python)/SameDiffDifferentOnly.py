"""
Same/Different 2AFC — DIFFERENT trials only
============================================
Companion to SameDiffGee.py.

SAME trials (same_ref, same_comp) are excluded. Analysis uses only trials where
GroundTruth == "DIFFERENT" (TrialType: diff_rc, diff_cr).

Accuracy
--------
On each included trial the correct response is DIFFERENT:
    correct = 1  if UserChoice == "DIFFERENT"  (equivalent to IsCorrect on these rows)
    correct = 0  if UserChoice == "SAME"

Accuracy = mean(correct) within the chosen grouping (subject × pair, region, etc.)

Chance  = 50%   (2AFC guess)
Threshold reference line = 75%
"""

import os
import re
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

USE_GEE = False
USE_WILCOXON = False
try:
    from statsmodels.genmod.generalized_estimating_equations import GEE
    from statsmodels.genmod.families import Binomial
    USE_GEE = True
    print("Using: statsmodels GEE (binomial)")
except ImportError:
    try:
        from scipy.stats import wilcoxon  # noqa: F401
        USE_WILCOXON = True
        print("statsmodels not found — using scipy Wilcoxon (fallback)")
    except ImportError:
        print("Neither statsmodels nor scipy — using permutation test (fallback)")

DATA_DIR = "/Users/kyungeunjung/NailFoldExp/Data/(FD)CurData"
FILE_PATTERNS = [
    os.path.join(DATA_DIR, "P*_ForceDiscrimination_SameDiff.csv"),
    os.path.join(DATA_DIR, "P*_ForceDiscrimination_SameDiff_26g.csv"),
]
OUTPUT_DIR = "/Users/kyungeunjung/NailFoldExp/(New)Analysis/ForceDiscAnalysis/Output/SameDiff_DifferentOnly"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CHANCE_PCT = 50.0
JND_PCT = 75.0
ON_NAIL = ["C", "D"]
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

C1 = "#2166AC"
C_ON = "#7FB3D3"
C_OFF = "#D3E9F5"
RED = "#c0392b"


def band_title_text(band_label, title_ref, n_subjects, n_diff_trials=None, n_total_trials=None):
    base = f"{band_label} band (ref = {title_ref}, n = {n_subjects}"
    if n_diff_trials is not None and n_total_trials is not None:
        n_same = n_total_trials - n_diff_trials
        base += f", DIFFERENT trials = {n_diff_trials} / {n_total_trials} total"
        base += f", SAME excluded = {n_same}"
    base += ")"
    return base


def subject_number(filepath):
    m = re.search(r"P(\d+)", os.path.basename(filepath))
    return int(m.group(1)) if m else 0


def fix_order(order, actual):
    fixed = []
    for p in order:
        if p in actual:
            fixed.append(p)
        else:
            alt = [a for a in actual if set(a.replace("–", "-").split("-")) == set(p.replace("–", "-").split("-"))]
            fixed.append(alt[0] if alt else p)
    return fixed


def trial_count_summary(df_all, df_diff):
    """Overall and per-band trial counts before/after removing SAME trials."""
    n_total = len(df_all)
    n_diff = len(df_diff)
    n_same = n_total - n_diff
    rows = [{
        "scope": "all",
        "band": "All",
        "n_total_trials": n_total,
        "n_same_excluded": n_same,
        "n_different_kept": n_diff,
        "pct_different": round(100 * n_diff / n_total, 1) if n_total else np.nan,
    }]
    for band in ["Low", "High"]:
        sub_all = df_all[df_all["band"] == band]
        sub_diff = df_diff[df_diff["band"] == band]
        if len(sub_all) == 0:
            continue
        rows.append({
            "scope": "band",
            "band": band,
            "n_total_trials": len(sub_all),
            "n_same_excluded": len(sub_all) - len(sub_diff),
            "n_different_kept": len(sub_diff),
            "pct_different": round(100 * len(sub_diff) / len(sub_all), 1),
        })
    return pd.DataFrame(rows)


def print_trial_counts(summary_df):
    print("\n── Trial counts (SAME excluded, DIFFERENT kept) ──")
    for _, row in summary_df.iterrows():
        label = row["band"]
        print(f"  {label:5s}  total = {int(row['n_total_trials']):4d}  "
              f"SAME excluded = {int(row['n_same_excluded']):4d}  "
              f"DIFFERENT kept = {int(row['n_different_kept']):4d}  "
              f"({row['pct_different']:.1f}% of total)")


def load_different_trials():
    all_files = sorted(set(f for pat in FILE_PATTERNS for f in glob.glob(pat)))
    if not all_files:
        raise FileNotFoundError("No SameDiff CSV files found.")
    files = sorted(f for f in all_files if subject_number(f) > 73)
    if not files:
        raise ValueError("No files with subject ID > 73.")

    print(f"Files loaded: {[os.path.basename(f) for f in files]}")
    df_all = pd.concat([pd.read_csv(f, encoding="utf-8-sig") for f in files], ignore_index=True)

    df_all["pair_label"] = df_all.apply(
        lambda r: f"{min(r['Reference'], r['Comparison']):g}–{max(r['Reference'], r['Comparison']):g}",
        axis=1,
    )
    df_all["band"] = df_all["Reference"].map({1: "Low", 26: "High"})

    df = df_all[df_all["GroundTruth"] == "DIFFERENT"].copy()
    summary = trial_count_summary(df_all, df)
    print_trial_counts(summary)

    df["correct"] = (df["UserChoice"] == "DIFFERENT").astype(int)
    df["region_group"] = df["Region"].map(
        {r: "On-nail" for r in ON_NAIL} | {r: "Off-nail" for r in OFF_NAIL}
    )
    return df, df_all, summary


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

        paired = subj_acc[subj_acc["pair_label"].isin([p1, p2])]\
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


def pval_label(p):
    if np.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return f"n.s. p={p:.3f}"


def draw_bracket(ax, x1, x2, y, label, tick_h=3.5):
    ax.plot([x1, x1, x2, x2], [y, y + tick_h, y + tick_h, y], color=RED, lw=1.2, clip_on=False)
    if label:
        ax.text((x1 + x2) / 2, y + tick_h + 1.5, label,
                ha="center", va="bottom", fontsize=9, color=RED, fontweight="bold")


def jitter_x(n, width=0.12, seed=42):
    return (np.random.default_rng(seed).random(n) - 0.5) * width


def draw_pair_accuracy_boxplot(ax, pair_order, values_by_pair, title, *,
                               pairwise_pvals_dict=None, dot_label=None, trial_note=None):
    band_max = 0.0
    for xi, pair in enumerate(pair_order):
        vals = np.asarray(values_by_pair.get(pair, []), dtype=float) * 100
        if len(vals) == 0:
            continue
        band_max = max(band_max, float(np.nanmax(vals)))
        bp = ax.boxplot([vals], positions=[xi], widths=0.42, patch_artist=True, showfliers=False)
        bp["boxes"][0].set_facecolor("#dde6f0")
        bp["boxes"][0].set_edgecolor("black")
        bp["medians"][0].set_color(RED)
        bp["medians"][0].set_linewidth(2)
        ax.scatter(xi + jitter_x(len(vals)), vals, color=C1, alpha=0.6, s=20, zorder=3)

    ax.axhline(JND_PCT, color="black", ls="--", lw=1.2, alpha=0.8)
    ax.axhline(CHANCE_PCT, color="gray", ls=":", lw=0.9, alpha=0.7)
    ax.text(len(pair_order) - 0.5, CHANCE_PCT + 1.5, f"chance ({CHANCE_PCT:.0f}%)",
            ha="right", fontsize=9, color="#888")

    y_top = max(105.0, band_max + 8)
    if pairwise_pvals_dict:
        pair_combos = sorted(itertools.combinations(range(len(pair_order)), 2), key=lambda t: t[1] - t[0])
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
        ax.text(0.02, 0.98, dot_label, transform=ax.transAxes, ha="left", va="top", fontsize=9, color="#555")
    if trial_note:
        ax.text(0.98, 0.98, trial_note, transform=ax.transAxes, ha="right", va="top", fontsize=9, color="#555")


def save_subject_accuracy_by_pair(df_sub, subject, pair_order, band_title, out_path, trial_note):
    region_acc = df_sub.groupby(["pair_label", "Region"])["correct"].mean().reset_index()
    values_by_pair = {
        pair: region_acc.loc[region_acc["pair_label"] == pair, "correct"].values
        for pair in pair_order
    }
    fig, ax = plt.subplots(figsize=(8, 6))
    draw_pair_accuracy_boxplot(
        ax, pair_order, values_by_pair,
        f"{subject} — DIFFERENT trials only ({band_title})",
        dot_label="each dot = one region (A–F)",
        trial_note=trial_note,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {os.path.relpath(out_path, OUTPUT_DIR)}")


def run_subject_analysis(df_band, df_all_band, band_label, pair_order, out_suffix, title_ref):
    subj_root = os.path.join(OUTPUT_DIR, "per_subject")
    os.makedirs(subj_root, exist_ok=True)
    summary_rows = []

    for subject in sorted(df_band["Subject"].unique()):
        df_sub = df_band[df_band["Subject"] == subject]
        df_sub_all = df_all_band[df_all_band["Subject"] == subject]
        n_total = len(df_sub_all)
        n_diff = len(df_sub)
        n_same = n_total - n_diff
        band_title = band_title_text(band_label, title_ref, 1, n_diff, n_total)
        trial_note = f"DIFFERENT {n_diff} / total {n_total} (SAME −{n_same})"
        subj_dir = os.path.join(subj_root, subject)
        os.makedirs(subj_dir, exist_ok=True)

        by_pair = df_sub.groupby("pair_label")["correct"].agg(n_trials="count", accuracy="mean").reindex(pair_order)
        print(f"\n--- {subject} | {band_title} ---")
        print(f"  Trials: total {n_total}  |  SAME excluded {n_same}  |  DIFFERENT kept {n_diff}")
        for pair in pair_order:
            if pair not in by_pair.index or pd.isna(by_pair.loc[pair, "n_trials"]):
                continue
            acc = by_pair.loc[pair, "accuracy"] * 100
            n = int(by_pair.loc[pair, "n_trials"])
            print(f"  {pair:8s}  {acc:5.1f}%  (n={n})")
            summary_rows.append({
                "Subject": subject, "band": band_label, "pair_label": pair,
                "n_total_trials": n_total, "n_same_excluded": n_same, "n_different_kept": n_diff,
                "n_trials_pair": n, "accuracy_pct": acc,
            })

        save_subject_accuracy_by_pair(
            df_sub, subject, pair_order, band_title,
            os.path.join(subj_dir, f"diff_accuracy_by_pair{out_suffix}.png"),
            trial_note,
        )

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(
            os.path.join(subj_root, f"diff_accuracy_by_subject{out_suffix}.csv"), index=False
        )
        print(f"Saved summary → per_subject/diff_accuracy_by_subject{out_suffix}.csv")


def run_band_analysis(df_band, df_all_band, band_label, pair_order, out_suffix, title_ref):
    n_subj = df_band["Subject"].nunique()
    n_total = len(df_all_band)
    n_diff = len(df_band)
    n_same = n_total - n_diff
    band_title = band_title_text(band_label, title_ref, n_subj, n_diff, n_total)
    trial_note = f"DIFFERENT {n_diff} / total {n_total} (SAME −{n_same})"
    print(f"\n{'=' * 60}")
    print(f"DIFFERENT only | {band_title}")
    print(f"  Trials: total {n_total}  |  SAME excluded {n_same}  |  DIFFERENT kept {n_diff}")

    subj_acc = (
        df_band.groupby(["Subject", "pair_label"])["correct"]
        .mean().reset_index().rename(columns={"correct": "accuracy"})
    )
    subj_acc.to_csv(os.path.join(OUTPUT_DIR, f"diff_accuracy_subject_pair{out_suffix}.csv"), index=False)

    pairwise_pvals = run_gee_pairwise(df_band, subj_acc, pair_order)
    region_pvals = run_gee_region(df_band, pair_order)

    fig, ax = plt.subplots(figsize=(8, 6))
    values_by_pair = {
        pair: subj_acc.loc[subj_acc["pair_label"] == pair, "accuracy"].values
        for pair in pair_order
    }
    draw_pair_accuracy_boxplot(
        ax, pair_order, values_by_pair,
        f"DIFFERENT-Trial Accuracy — Same/Different 2AFC ({band_title})",
        pairwise_pvals_dict=pairwise_pvals,
        dot_label="each dot = one subject (regions pooled)",
        trial_note=trial_note,
    )
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, f"diff_accuracy_by_pair{out_suffix}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → diff_accuracy_by_pair{out_suffix}.png")

    OFFSET = 0.22
    BW = 0.20
    df_reg = df_band[df_band["region_group"].notna()]
    subj_acc_reg = (
        df_reg.groupby(["Subject", "pair_label", "region_group"])["correct"]
        .mean().reset_index().rename(columns={"correct": "accuracy"})
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    band_max = 0.0
    for xi, pair in enumerate(pair_order):
        for gi, (grp, color) in enumerate([("On-nail", C_ON), ("Off-nail", C_OFF)]):
            xp = xi + OFFSET * (gi - 0.5)
            vals = subj_acc_reg.loc[
                (subj_acc_reg["pair_label"] == pair) & (subj_acc_reg["region_group"] == grp),
                "accuracy",
            ].values * 100
            if len(vals) == 0:
                continue
            band_max = max(band_max, vals.max())
            bp = ax.boxplot([vals], positions=[xp], widths=BW, patch_artist=True, showfliers=False)
            bp["boxes"][0].set_facecolor(color)
            bp["boxes"][0].set_edgecolor("black")
            bp["medians"][0].set_color(RED)
            bp["medians"][0].set_linewidth(2)
            ax.scatter(xp + jitter_x(len(vals), width=BW * 0.5), vals,
                       color=C_ON if grp == "On-nail" else "#5b7fa6", alpha=0.6, s=18, zorder=3)
        pval = region_pvals.get(pair, np.nan)
        y_b = band_max + 10
        ax.plot([xi - OFFSET * 0.5, xi - OFFSET * 0.5, xi + OFFSET * 0.5, xi + OFFSET * 0.5],
                [y_b, y_b + 3, y_b + 3, y_b], color=RED, lw=1.2)
        ax.text(xi, y_b + 4.5, pval_label(pval), ha="center", va="bottom", fontsize=9, color=RED, fontweight="bold")

    ax.axhline(JND_PCT, color="black", ls="--", lw=1.2, alpha=0.8)
    ax.axhline(CHANCE_PCT, color="gray", ls=":", lw=0.9, alpha=0.7)
    ax.set_xticks(range(len(pair_order)))
    ax.set_xticklabels(pair_order, fontsize=11)
    ax.set_ylim(0, band_max + 28)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_xlabel("Force pair (g)", fontsize=11)
    ax.set_title(f"On-nail vs Off-nail — DIFFERENT trials ({band_title})", fontsize=12, fontweight="bold")
    ax.text(0.98, 0.02, trial_note, transform=ax.transAxes, ha="right", va="bottom", fontsize=9, color="#555")
    ax.legend(handles=[
        mpatches.Patch(facecolor=C_ON, edgecolor="black", label="On-nail (C+D)"),
        mpatches.Patch(facecolor=C_OFF, edgecolor="black", label="Off-nail (A+F)"),
    ], loc="upper left", frameon=False, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, f"diff_onnail_vs_offnail{out_suffix}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → diff_onnail_vs_offnail{out_suffix}.png")

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
            })
    df_order = pd.DataFrame(order_rows)
    df_order.to_csv(os.path.join(OUTPUT_DIR, f"diff_order_effect{out_suffix}.csv"), index=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    for gi, (grp, color) in enumerate([("On-nail", C_ON), ("Off-nail", C_OFF)]):
        sub = df_order[df_order["region_group"] == grp]
        xs_pairs = [pair_order.index(p) for p in sub["pair_label"] if p in pair_order]
        x_pos = np.array(xs_pairs, dtype=float) + OFFSET * (gi - 0.5)
        ax.plot(x_pos, sub["acc_diff_rc"].values, "o-", color=color, lw=2, label=f"{grp} diff_rc")
        ax.plot(x_pos, sub["acc_diff_cr"].values, "s--", color=color, lw=1.5, alpha=0.7, label=f"{grp} diff_cr")
    ax.axhline(CHANCE_PCT, color="gray", ls=":", lw=0.9)
    ax.axhline(JND_PCT, color="black", ls="--", lw=1.1, alpha=0.8)
    ax.set_xticks(range(len(pair_order)))
    ax.set_xticklabels(pair_order, fontsize=11)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_xlabel("Force pair (g)", fontsize=11)
    ax.set_title(f"Order effect — DIFFERENT trials ({band_title})", fontsize=12, fontweight="bold")
    ax.text(0.98, 0.02, trial_note, transform=ax.transAxes, ha="right", va="bottom", fontsize=9, color="#555")
    ax.legend(frameon=False, fontsize=9, ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, f"diff_order_effect{out_suffix}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → diff_order_effect{out_suffix}.png")

    run_subject_analysis(df_band, df_all_band, band_label, pair_order, out_suffix, title_ref)


def main():
    df, df_all, summary = load_different_trials()
    summary.to_csv(os.path.join(OUTPUT_DIR, "trial_counts.csv"), index=False)
    print(f"Saved → trial_counts.csv")

    for band_label, cfg in BAND_CONFIG.items():
        df_band = df[df["band"] == band_label].copy()
        df_all_band = df_all[df_all["band"] == band_label].copy()
        if df_band.empty:
            print(f"\nNo DIFFERENT trials for {band_label} band — skipping")
            continue
        pair_order = fix_order(cfg["pair_order"], df_band["pair_label"].unique().tolist())
        run_band_analysis(df_band, df_all_band, band_label, pair_order, cfg["suffix"], cfg["title_ref"])
    print(f"\nAll outputs saved to:\n  {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
