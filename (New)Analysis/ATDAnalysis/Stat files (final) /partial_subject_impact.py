"""
partial_subject_impact.py
=========================
P61+ (partial-protocol) 참가자 추가 전후 비교:
  0.16 g 와 0.6 g 에서 분포가 어떻게 바뀌는지 시각화.

Figure layout (1 row × 2 cols, one per force):
  왼쪽 violin/strip: Original only (P1–P60)
  오른쪽 violin/strip: Original + Partial (P61+)
  세모(▲) = partial-protocol subjects, 원(●) = original
"""

import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

# ── Load ──────────────────────────────────────────────────────────────────────
REPO  = "/Users/kyungeunjung/NailFoldExp"
OUT   = os.path.join(REPO, "(New)Analysis", "ATDAnalysis", "atd_c1_outputs")
files = sorted(glob.glob(os.path.join(REPO, "Data", "(ATD)CurData", "P*.csv")))
df    = pd.concat([pd.read_csv(f, encoding="utf-8-sig") for f in files], ignore_index=True)

scol = "SubjectID" if "SubjectID" in df.columns else "Subject"
df["Force_Val"] = df["Force"].str.extract(r"(\d+\.?\d*)").astype(float)
df["Num"]       = df[scol].str.extract(r"(\d+)")[0].astype(int)
df["is_partial"]= df["Num"] >= 61

df["Condition"] = df["Condition"].str.strip().replace({
    "Active": "On-touch (Mid)", "On-touch (Hard)": "On-touch (Mid)", "Passive": "In-air"})
df = df[df["Condition"] == "On-touch (Mid)"].copy()

def score(row):
    if row["Target"] == 0:
        return 100.0 if row["Response"] == 0 else 0.0
    return max(0.0, (1 - abs(row["Target"] - row["Response"]) / row["Target"]) * 100.0)
df["Score"] = df.apply(score, axis=1)

FORCES = [0.16, 0.6]
BLUE   = "#4A7FB5"
TRI_C  = "#C0392B"

# Per-subject means
subj = df.groupby([scol, "Force_Val", "is_partial"])["Score"].mean().reset_index()
subj.rename(columns={"Score": "acc"}, inplace=True)

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 5.5), sharey=True)
fig.suptitle("Impact of Adding P61+ (Partial-Protocol) Participants\n"
             "On-touch condition · per-subject mean accuracy",
             fontsize=13, fontweight="bold", y=1.01)

def jitter(n, seed=0, span=0.08):
    rng = np.random.default_rng(seed)
    return rng.uniform(-span, span, size=n)

for ax, fval in zip(axes, FORCES):
    sub = subj[subj["Force_Val"] == fval]
    orig    = sub[~sub["is_partial"]]["acc"].values
    partial = sub[ sub["is_partial"]]["acc"].values
    combined= sub["acc"].values

    # ── Violin: original (left half) vs combined (right half) ──────────────
    x_orig = 0.0
    x_comb = 1.2

    for xpos, vals, label in [(x_orig, orig, "Original\n(P1–P60)"),
                               (x_comb, combined, "With P61+\n(all)")]:
        vp = ax.violinplot(vals, positions=[xpos], widths=0.55,
                           showmedians=False, showextrema=False)
        for body in vp["bodies"]:
            body.set_facecolor(BLUE)
            body.set_alpha(0.18)
            body.set_edgecolor(BLUE)
            body.set_linewidth(0.8)

        # Median line
        med = np.median(vals)
        ax.hlines(med, xpos - 0.15, xpos + 0.15,
                  color=BLUE, linewidth=2.2, zorder=5)
        # IQR box
        q1, q3 = np.percentile(vals, [25, 75])
        ax.add_patch(plt.Rectangle((xpos - 0.12, q1), 0.24, q3 - q1,
                                   facecolor=BLUE, alpha=0.25, edgecolor=BLUE,
                                   linewidth=0.8, zorder=4))

    # ── Strip: original dots on both columns ─────────────────────────────────
    j_orig = jitter(len(orig), seed=1)
    ax.scatter(x_orig + j_orig, orig,
               color=BLUE, s=22, alpha=0.65, marker="o", zorder=6,
               linewidths=0, edgecolors="none")
    ax.scatter(x_comb + j_orig, orig,       # same subjects on combined panel
               color=BLUE, s=22, alpha=0.65, marker="o", zorder=6,
               linewidths=0, edgecolors="none")

    # Partial subjects: triangles (only on combined panel)
    j_par = jitter(len(partial), seed=2)
    ax.scatter(x_comb + j_par, partial,
               color=TRI_C, s=32, alpha=0.85, marker="^", zorder=7,
               linewidths=0, edgecolors="none")

    # ── Median annotation ─────────────────────────────────────────────────────
    med_orig = np.median(orig)
    med_comb = np.median(combined)
    delta    = med_comb - med_orig
    sign     = "+" if delta >= 0 else "−"
    ax.text(0.5, 1.03,
            f"Δ median = {sign}{abs(delta):.1f} pp",
            ha="center", va="bottom", transform=ax.transAxes,
            fontsize=11, color="black", fontweight="bold")

    for xpos, med, vals, lbl in [
            (x_orig, med_orig, orig,     f"median {med_orig:.1f}%\n(n={len(orig)})"),
            (x_comb, med_comb, combined, f"median {med_comb:.1f}%\n(n={len(combined)})")]:
        ax.text(xpos, med + 2.5, lbl, ha="center", va="bottom",
                fontsize=9, color=BLUE, fontweight="bold")

    # ── Styling ───────────────────────────────────────────────────────────────
    ax.set_xticks([x_orig, x_comb])
    ax.set_xticklabels(["Original\n(P1–P60, n=30)", f"+ P61+\n(n={len(combined)})"],
                       fontsize=10)
    ax.set_title(f"{fval} g", fontsize=13, fontweight="bold")
    ax.set_xlim(-0.5, 1.7)
    ax.set_ylim(-5, 115)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.axhline(80, color="black", ls="--", lw=1.0, alpha=0.55)
    ax.axhline(50, color="gray",  ls=":",  lw=0.8, alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", ls=":", alpha=0.25)

axes[0].set_ylabel("Detection Accuracy (%)", fontsize=12)

# ── Legend ────────────────────────────────────────────────────────────────────
handles = [
    mlines.Line2D([], [], color=BLUE,  marker="o", lw=0, markersize=7,
                  label="Original participants (P1–P60)"),
    mlines.Line2D([], [], color=TRI_C, marker="^", lw=0, markersize=8,
                  label="Partial-protocol participants (P61+)"),
    mlines.Line2D([], [], color=BLUE,  lw=2.2, label="Median"),
    mpatches.Patch(facecolor=BLUE, alpha=0.25, label="IQR"),
    mlines.Line2D([], [], color="black", ls="--", lw=1.0, label="80% criterion"),
]
fig.legend(handles=handles, loc="lower center",
           bbox_to_anchor=(0.5, -0.06), ncol=5,
           fontsize=9, frameon=False)

plt.tight_layout()
out_path = os.path.join(OUT, "partial_subject_impact.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved → {out_path}")

# ── Console summary ────────────────────────────────────────────────────────────
print("\n── Numeric summary ──────────────────────────────────────────────────")
for fval in FORCES:
    sub     = subj[subj["Force_Val"] == fval]
    orig_v  = sub[~sub["is_partial"]]["acc"]
    comb_v  = sub["acc"]
    par_v   = sub[ sub["is_partial"]]["acc"]
    print(f"\n{fval} g")
    print(f"  Original  (n={len(orig_v)}): "
          f"median={orig_v.median():.1f}%  mean={orig_v.mean():.1f}%  "
          f"IQR=[{orig_v.quantile(.25):.1f}, {orig_v.quantile(.75):.1f}]")
    print(f"  With P61+ (n={len(comb_v)}): "
          f"median={comb_v.median():.1f}%  mean={comb_v.mean():.1f}%  "
          f"IQR=[{comb_v.quantile(.25):.1f}, {comb_v.quantile(.75):.1f}]")
    print(f"  Δ median = {comb_v.median() - orig_v.median():.1f} pp  "
          f"Δ mean = {comb_v.mean() - orig_v.mean():.1f} pp")
    print(f"  P61+ range: [{par_v.min():.1f}, {par_v.max():.1f}]%  "
          f"median={par_v.median():.1f}%")
