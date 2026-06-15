"""
Force Discrimination – Directional Bias Analysis
=================================================
Directly tests the claim:
  "participants systematically judged the LIGHTER stimulus as the HEAVIER one"

For each force pair, computes:
  → "proportion of trials where the lighter stimulus was called heavier"

  If comparison < reference (e.g. 0.6–1 g):
      bias = proportion where ChoseComparison == 1
             (participant said 0.6 g is heavier, but it's actually lighter)

  If comparison > reference (e.g. 1–1.4 g):
      bias = proportion where ChoseComparison == 0
             (participant said 1 g is heavier, but 1.4 g is actually heavier)

  Unified interpretation:
    > 50%  → lighter stimulus systematically called heavier  (structured reversal)
    = 50%  → random guessing (no bias)
    < 50%  → lighter correctly identified as lighter

Two figures:
  Fig 1. Bias strip plot  — per-subject dots + mean±SE, per force pair
  Fig 2. Bias vs accuracy — scatter confirming bias mirrors below-chance accuracy

Run:
    python fd_directional_bias.py
"""

import os, glob, importlib.util
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.font_manager as _fm
import seaborn as sns
from pathlib import Path

# =============================================================================
# 0. ATD style loader
# =============================================================================
_SCRIPT_DIR = Path(__file__).resolve().parent
_ATD_PATH   = _SCRIPT_DIR.parent / "ATDAnalysis" / "ATD_C1_Fig(Anika).py"

def _load_atd():
    spec = importlib.util.spec_from_file_location("atd_c1_fig", _ATD_PATH)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

ATD = _load_atd()

# Override ATD's zero-tick global rcParams for this script
import matplotlib as _mpl
_mpl.rcParams["xtick.major.size"]  = 4
_mpl.rcParams["ytick.major.size"]  = 4
_mpl.rcParams["xtick.major.width"] = 0.8
_mpl.rcParams["ytick.major.width"] = 0.8
_mpl.rcParams["axes.titleweight"]  = "bold"

ACCENT_RED      = ATD.ACCENT_RED
FONT_TICK       = ATD.FONT_TICK  + 4   # 20  (ATD default: 16)
FONT_LABEL      = ATD.FONT_LABEL + 4   # 18  (ATD default: 14)
FONT_ANNOT      = ATD.FONT_ANNOT + 2
FIG_SIZE        = (10.0, ATD.FIG_SIZE[1] * (10.0 / 8.0))   # wider panels, proportional height
SAVE_DPI        = ATD.SAVE_DPI
EXPORT_WIDTHS_PX = ATD.EXPORT_WIDTHS_PX

COLOR_LOW       = "#2166AC"
COLOR_HIGH      = "#C0392B"
COLOR_BIAS_HIGH = "#C94040"   # bias > 50% (reversal)
COLOR_BIAS_LOW  = "#4A90C4"   # bias < 50% (correct)
COLOR_NEUTRAL   = "#AAAAAA"   # near 50%
GAP_IN          = 0.8

# =============================================================================
# 1. Paths
# =============================================================================
REPO_ROOT  = "/Users/kyungeunjung/NailFoldExp"
FD_PATTERN = os.path.join(REPO_ROOT, "Data", "(FD)CurData",
                           "P*_ForceDiscrimination.csv")
OUTPUT_DIR = os.path.join(REPO_ROOT, "(New)Analysis", "ForceDiscAnalysis",
                           "Force_ATD_outputs", "DirectionalBias")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# 2. Load & prepare FD data
# =============================================================================
fd_files = sorted(glob.glob(FD_PATTERN))
if not fd_files:
    raise FileNotFoundError(f"No FD files found: {FD_PATTERN}")
print(f"FD: {len(fd_files)} participant file(s).")

df = pd.concat(
    [pd.read_csv(f, encoding="utf-8-sig") for f in fd_files],
    ignore_index=True,
)

# Standard columns
df["correct"] = np.where(
    df["Comparison"] > df["Reference"],
    df["ChoseComparison"] == 1,
    df["ChoseComparison"] == 0,
).astype(int)

df["pair_label"] = df.apply(
    lambda r: f"{min(r['Reference'], r['Comparison']):g}–"
              f"{max(r['Reference'], r['Comparison']):g}",
    axis=1,
)
df["band"] = df["Reference"].apply(lambda r: "Low" if r == 1 else "High")

df["rel_contrast"] = (
    np.abs(df["Comparison"] - df["Reference"]) / df["Reference"]
)

# =============================================================================
# 3. Directional bias metric
#    "proportion of trials where the LIGHTER stimulus was called HEAVIER"
# =============================================================================
def lighter_called_heavier(row):
    """
    Returns 1 if the lighter stimulus was labelled as the heavier one.
    - comparison < reference  →  lighter = comparison  →  error if ChoseComparison=1
    - comparison > reference  →  lighter = reference   →  error if ChoseComparison=0
    """
    if row["Comparison"] < row["Reference"]:
        return int(row["ChoseComparison"] == 1)   # said comparison (lighter) is heavier
    else:
        return int(row["ChoseComparison"] == 0)   # said reference (lighter) is heavier

df["lighter_as_heavier"] = df.apply(lighter_called_heavier, axis=1)

# Per-subject bias per force pair
subj_bias = (
    df.groupby(["Subject", "band", "pair_label", "rel_contrast"])
    .agg(
        bias=("lighter_as_heavier", "mean"),
        accuracy=("correct", "mean"),
        n_trials=("correct", "count"),
    )
    .reset_index()
)
subj_bias["bias_pct"]     = subj_bias["bias"]     * 100
subj_bias["accuracy_pct"] = subj_bias["accuracy"] * 100

# Group mean ± SE
grp_bias = (
    subj_bias.groupby(["band", "pair_label", "rel_contrast"])
    .agg(
        mean_bias  =("bias_pct",     "mean"),
        se_bias    =("bias_pct",     lambda x: x.std(ddof=1) / np.sqrt(len(x))),
        median_bias=("bias_pct",     "median"),
        q25_bias   =("bias_pct",     lambda x: np.percentile(x, 25)),
        q75_bias   =("bias_pct",     lambda x: np.percentile(x, 75)),
        mean_acc   =("accuracy_pct", "mean"),
        se_acc     =("accuracy_pct", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
        n          =("bias_pct",     "count"),
    )
    .reset_index()
)

# Ordered pair lists
low_order  = ["0.4–1", "0.6–1", "1–1.4", "1–2"]
high_order = ["10–26", "15–26", "26–60"]

actual = grp_bias["pair_label"].unique().tolist()

def fix_order(order):
    fixed = []
    for p in order:
        if p in actual:
            fixed.append(p)
        else:
            alt = [a for a in actual
                   if set(a.split("–")) == set(p.split("–"))]
            if alt:
                fixed.append(alt[0])
    return fixed

low_order  = fix_order(low_order)
high_order = fix_order(high_order)

print("Low pairs:",  low_order)
print("High pairs:", high_order)
print(grp_bias[["band","pair_label","rel_contrast","mean_bias","mean_acc"]].to_string(index=False))

# =============================================================================
# 4. Save helper
# =============================================================================
def save_fig(fig, stem):
    import io
    from PIL import Image
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=SAVE_DPI, bbox_inches="tight",
                pad_inches=0.05, facecolor="white")
    buf.seek(0)
    master = Image.open(buf).convert("RGB")
    for tag, w in EXPORT_WIDTHS_PX:
        h = round(w * master.height / master.width)
        master.resize((w, h), Image.Resampling.LANCZOS).save(
            os.path.join(OUTPUT_DIR, f"{stem}_{tag}.png"))
    legacy = os.path.join(OUTPUT_DIR, f"{stem}.png")
    master.resize(
        (2102, round(2102 * master.height / master.width)),
        Image.Resampling.LANCZOS,
    ).save(legacy)
    print(f"Saved → {legacy}")

# =============================================================================
# 5. Layout helper
# =============================================================================
def make_two_panel(fig_h=None):
    sns.set_theme(style="white")
    ATD.apply_plot_style()
    # Re-apply bold title weight after sns.set_theme resets rcParams
    _mpl.rcParams["axes.titleweight"] = "bold"
    left_in  = 0.09 * FIG_SIZE[0]
    right_in = 0.03 * FIG_SIZE[0]
    pw       = (FIG_SIZE[0] - left_in - right_in - GAP_IN) / 2
    ph       = FIG_SIZE[1] * 0.72
    bot_in   = ATD.FIG_LEGEND_BOTTOM * FIG_SIZE[1]
    fig_h    = fig_h or (ph + bot_in + 0.55 + 0.3)
    ax_y     = bot_in / fig_h
    ax_h     = ph / fig_h
    fig      = plt.figure(figsize=(FIG_SIZE[0], fig_h), facecolor="#FFFFFF")
    ax_l     = fig.add_axes([left_in / FIG_SIZE[0],
                              ax_y, pw / FIG_SIZE[0], ax_h])
    ax_r     = fig.add_axes([(left_in + pw + GAP_IN) / FIG_SIZE[0],
                              ax_y, pw / FIG_SIZE[0], ax_h])
    return fig, ax_l, ax_r

# =============================================================================
# FIGURE 1: Directional Bias Strip Plot
# =============================================================================
BIAS_THRESHOLD = 5.0    # ±5% from 50% → "near neutral"
JITTER_SEED    = 42
DOT_ALPHA      = 0.65
DOT_S          = 28

def dot_color(bias_pct):
    """Color by direction of bias."""
    if bias_pct > 50 + BIAS_THRESHOLD:
        return COLOR_BIAS_HIGH   # reversal: lighter called heavier
    elif bias_pct < 50 - BIAS_THRESHOLD:
        return COLOR_BIAS_LOW    # correct: lighter correctly identified
    return COLOR_NEUTRAL

def bias_panel(ax, band, order, show_ylabel):
    sub = subj_bias[subj_bias["band"] == band]
    grp = grp_bias[grp_bias["band"] == band]
    rng = np.random.default_rng(JITTER_SEED)

    for xi, pair in enumerate(order):
        d   = sub[sub["pair_label"] == pair]["bias_pct"].values
        g   = grp[grp["pair_label"] == pair]
        if not len(d) or g.empty:
            continue

        # Per-subject dots
        x_jitter = xi + (rng.random(len(d)) - 0.5) * 0.22
        colors   = [dot_color(v) for v in d]
        ax.scatter(x_jitter, d, c=colors, s=DOT_S,
                   alpha=DOT_ALPHA, edgecolors="white",
                   linewidths=0.3, zorder=4)

        # Median ± IQR
        m   = g["median_bias"].values[0]
        q25 = g["q25_bias"].values[0]
        q75 = g["q75_bias"].values[0]
        ax.errorbar(xi, m, yerr=[[m - q25], [q75 - m]],
                    fmt="D", color="#111111",
                    markersize=6, markeredgecolor="white",
                    markeredgewidth=0.5,
                    capsize=3, capthick=1.0,
                    elinewidth=1.2, zorder=6)

        # Median value annotation – placed to the right of the diamond
        ax.text(xi + 0.18, m, f"{m:.0f}%",
                ha="left", va="center",
                fontsize=FONT_ANNOT - 0.5,
                color=COLOR_BIAS_HIGH if m > 50 else COLOR_BIAS_LOW,
                fontweight="bold")

    # Reference lines
    ax.axhline(50, color="#333333", lw=1.2, linestyle="-", alpha=0.75, zorder=2)
    ax.fill_between([-0.5, len(order) - 0.5],
                    50 - BIAS_THRESHOLD, 50 + BIAS_THRESHOLD,
                    color="#DDDDDD", alpha=0.40, zorder=1,
                    label=f"Neutral zone (50 ± {BIAS_THRESHOLD:.0f}%)")

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, fontsize=FONT_TICK)
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_ylim(0, 110)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0", "25", "50", "75", "100"],
                        fontsize=FONT_TICK)
    ax.set_xlabel("Force pair (g)", fontsize=FONT_LABEL,
                  labelpad=ATD.FIG_AXIS_LABELPAD)
    if show_ylabel:
        ax.set_ylabel("Proportion (%)",
                      fontsize=FONT_LABEL, labelpad=ATD.FIG_AXIS_LABELPAD)
    ax.set_title("")
    ax.grid(False)
    sns.despine(ax=ax)
    ax.spines["left"].set_bounds(0, 100)
    ax.tick_params(axis="both", which="both", length=0, labelsize=FONT_TICK)
    # Draw outward tick marks in data coordinates
    y_lo, y_hi = ax.get_ylim()
    x_lo, x_hi = ax.get_xlim()
    dy = (y_hi - y_lo) * 0.022   # tick length in data units (y)
    dx = (x_hi - x_lo) * 0.022   # tick length in data units (x)
    for xi in range(len(order)):
        ax.plot([xi, xi], [y_lo, y_lo + dy],
                color="black", linewidth=0.8, solid_capstyle="butt",
                clip_on=False, zorder=10)
    for y in ax.get_yticks():
        if y_lo - 1e-9 <= y <= y_hi + 1e-9:
            ax.plot([x_lo, x_lo + dx], [y, y],
                    color="black", linewidth=0.8, solid_capstyle="butt",
                    clip_on=False, zorder=10)


def make_bias_strip():
    fig, ax_l, ax_r = make_two_panel()
    bias_panel(ax_l, "Low",  low_order,  show_ylabel=True)
    bias_panel(ax_r, "High", high_order, show_ylabel=False)

    handles = [
        mlines.Line2D([], [], color=COLOR_BIAS_HIGH, marker="o",
                      markersize=6, linewidth=0,
                      label="Reversal: \nlighter chose as heavier"),
        mlines.Line2D([], [], color=COLOR_BIAS_LOW, marker="o",
                      markersize=6, linewidth=0,
                      label="Correct: \nlighter identified as lighter"),
    ]
    fig.legend(handles=handles, loc="upper center",
               bbox_to_anchor=(0.5, 0.99),
               bbox_transform=fig.transFigure,
               ncol=2, fontsize=FONT_LABEL, frameon=False,
               columnspacing=1.5, handletextpad=0.5)
    return fig

fig = make_bias_strip()
save_fig(fig, "fd_directional_bias_strip")
plt.close(fig)

# =============================================================================
# FIGURE 2: Bias vs Accuracy Scatter
#   x = accuracy (%), y = bias (%)
#   Each point = one force pair × one band
#   Expected: when accuracy < 50%, bias > 50%  (inverse relationship)
# =============================================================================
def make_bias_vs_accuracy():
    sns.set_theme(style="white")
    ATD.apply_plot_style()

    fig, ax = plt.subplots(
        figsize=(FIG_SIZE[0] * 0.65, FIG_SIZE[1] * 0.95),
        facecolor="#FFFFFF",
    )

    marks = {"Low": "o", "High": "s"}
    colors = {"Low": COLOR_LOW, "High": COLOR_HIGH}

    for band in ["Low", "High"]:
        order = low_order if band == "Low" else high_order
        grp   = grp_bias[(grp_bias["band"] == band) &
                          (grp_bias["pair_label"].isin(order))]
        for _, row in grp.iterrows():
            ax.errorbar(
                row["mean_acc"], row["mean_bias"],
                xerr=row["se_acc"], yerr=row["se_bias"],
                fmt=marks[band], color=colors[band],
                markersize=8, markeredgecolor="white",
                markeredgewidth=0.5,
                capsize=2.5, capthick=0.8, elinewidth=0.9,
                ecolor=colors[band], alpha=0.90, zorder=4,
            )
            ax.text(
                row["mean_acc"] + 1.5, row["mean_bias"] + 1.5,
                row["pair_label"] + " g",
                fontsize=FONT_ANNOT - 1.0,
                color=colors[band], alpha=0.85,
            )

    # Quadrant lines
    ax.axhline(50, color="#333333", lw=1.0, linestyle="-", alpha=0.6, zorder=2)
    ax.axvline(50, color="#333333", lw=1.0, linestyle="-", alpha=0.6, zorder=2)

    # Quadrant labels
    ax.text(18, 85, "Structured\nreversal",
            fontsize=FONT_ANNOT, color=COLOR_BIAS_HIGH,
            fontweight="bold", ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec=COLOR_BIAS_HIGH, alpha=0.7))
    ax.text(82, 40, "Correct\ndiscrimination",
            fontsize=FONT_ANNOT, color=COLOR_BIAS_LOW,
            fontweight="bold", ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec=COLOR_BIAS_LOW, alpha=0.7))

    ax.set_xlim(0, 105)
    ax.set_ylim(0, 105)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0", "25", "50\n(chance)", "75", "100"],
                        fontsize=FONT_TICK)
    ax.set_yticklabels(["0", "25", "50\n(neutral)", "75", "100"],
                        fontsize=FONT_TICK)
    ax.set_xlabel("Discrimination Accuracy (%)",
                  fontsize=FONT_LABEL, labelpad=ATD.FIG_AXIS_LABELPAD)
    ax.set_ylabel("Proportion: lighter judged as heavier (%)",
                  fontsize=FONT_LABEL, labelpad=ATD.FIG_AXIS_LABELPAD)
    ax.set_title("Accuracy vs. Directional Bias per Force Pair",
                 fontsize=FONT_LABEL, fontweight="bold", pad=8)
    ax.tick_params(length=0, labelsize=FONT_TICK)
    ax.grid(False)
    sns.despine(ax=ax)

    handles = [
        mlines.Line2D([], [], color=COLOR_LOW,  marker="o", markersize=7,
                      linewidth=0, label="Low band (ref = 1 g)"),
        mlines.Line2D([], [], color=COLOR_HIGH, marker="s", markersize=7,
                      linewidth=0, label="High band (ref = 26 g)"),
    ]
    ax.legend(handles=handles, fontsize=FONT_LABEL, frameon=False,
              loc="upper right", bbox_to_anchor=(1.0, 1.0),
              bbox_transform=ax.transAxes, handletextpad=0.5)

    fig.tight_layout()
    return fig

fig = make_bias_vs_accuracy()
save_fig(fig, "fd_directional_bias_vs_accuracy")
plt.close(fig)

print("\nDone. Files saved to:", OUTPUT_DIR)
print("  1. fd_directional_bias_strip          — per-subject bias dots + mean±SE")
print("  2. fd_directional_bias_vs_accuracy    — bias vs accuracy scatter (quadrant)")