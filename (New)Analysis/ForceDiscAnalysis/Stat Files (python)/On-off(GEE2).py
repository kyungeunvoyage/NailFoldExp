"""
Force Discrimination – Regional Equivalence Visualizations
==========================================================
Four fully standalone figures (no external script dependencies):

  Fig 1. Identity Scatter   — x=Off-nail, y=On-nail, identity line, per force pair
  Fig 2. Effect Size Forest — Cohen's d per force pair (On-nail vs Off-nail)
  Fig 3. Bland-Altman       — mean vs difference, per force pair
  Fig 4. TOST Equivalence   — equivalence confidence intervals per force pair

Run:
    python fd_region_alt_viz.py
"""

import os, glob, importlib.util
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from scipy import stats

# =============================================================================
# 0. ATD style loader
# =============================================================================
_SCRIPT_DIR = Path(__file__).resolve().parent
def _resolve_atd_path():
    root = _SCRIPT_DIR.parent.parent / "ATDAnalysis"
    for sub in ("Stat files", "Stat files (final) "):
        path = root / sub / "(Final)ATD_C1_Fig(Anika).py"
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"Could not find (Final)ATD_C1_Fig(Anika).py under {root}"
    )


_ATD_PATH   = _resolve_atd_path()

def _load_atd():
    spec = importlib.util.spec_from_file_location("atd_c1_fig", _ATD_PATH)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

ATD = _load_atd()

from gee_export_utils import EXPORT_CANVAS, add_figure_legend, horizontal_panel_rects, save_export_figure

# ── Style constants ───────────────────────────────────────────────────────────
ACCENT_RED      = ATD.ACCENT_RED
SLATE_BLUE      = ATD.SLATE_BLUE
FONT_TICK       = ATD.FONT_TICK
FONT_LABEL      = ATD.FONT_LABEL
FONT_ANNOT      = ATD.FONT_ANNOT
FIG_SIZE        = ATD.FIG_SIZE
SAVE_DPI        = ATD.SAVE_DPI
EXPORT_WIDTHS_PX = ATD.EXPORT_WIDTHS_PX
BOX_STROKE      = "#000000"

COLOR_ON_NAIL   = "#4A90C4"
COLOR_OFF_NAIL  = "#A8C8E0"
COLOR_LOW       = "#2166AC"
COLOR_HIGH      = "#D6604D"
COLOR_ZERO      = "#555555"
ALPHA_DOT       = 0.72

GAP_BAND_IN     = 1.5
LEGEND_H_IN     = 0.55
MARGIN_BOT      = ATD.FIG_LEGEND_BOTTOM

# =============================================================================
# 1. Paths
# =============================================================================
FILE_PATTERN = "/Users/kyungeunjung/NailFoldExp/Data/(FD)CurData/P*_ForceDiscrimination.csv"
OUTPUT_DIR   = "/Users/kyungeunjung/NailFoldExp/(New)Analysis/ForceDiscAnalysis/Output/Stats(GEE)"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# 2. Load & prepare data
# =============================================================================
files = sorted(glob.glob(FILE_PATTERN))
if not files:
    raise FileNotFoundError(f"No files found: {FILE_PATTERN}")
print(f"Loaded {len(files)} participant file(s).")

df = pd.concat([pd.read_csv(f, encoding="utf-8-sig") for f in files], ignore_index=True)

df["correct"] = np.where(
    df["Comparison"] > df["Reference"],
    df["ChoseComparison"] == 1,
    df["ChoseComparison"] == 0,
).astype(int)

df["pair_label"] = df.apply(
    lambda r: f"{min(r['Reference'], r['Comparison']):g}–"
              f"{max(r['Reference'], r['Comparison']):g}", axis=1)

df["band"] = df["Reference"].apply(lambda r: "Low" if r == 1 else "High")

# Region grouping
REGION_MAP = {"C": "On-nail", "D": "On-nail", "A": "Off-nail", "F": "Off-nail"}
df_r = df[df["Region"].isin(REGION_MAP)].copy()
df_r["region_group"] = df_r["Region"].map(REGION_MAP)

# Per-subject accuracy per region group per force pair
subj_reg = (
    df_r.groupby(["Subject", "band", "pair_label", "region_group"])["correct"]
    .mean().reset_index().rename(columns={"correct": "accuracy"})
)
subj_reg["accuracy_pct"] = subj_reg["accuracy"] * 100

# Pivot to wide: columns = On-nail, Off-nail
wide = (
    subj_reg.pivot_table(
        index=["Subject", "band", "pair_label"],
        columns="region_group", values="accuracy_pct"
    ).dropna().reset_index()
)
wide.columns.name = None
wide["diff"] = wide["On-nail"] - wide["Off-nail"]
wide["mean_acc"] = (wide["On-nail"] + wide["Off-nail"]) / 2

# Ordered pair lists
low_order  = ["0.4–1", "0.6–1", "1–1.4", "1–2"]
high_order = ["10–26", "15–26", "26–60"]

actual = wide["pair_label"].unique().tolist()

def fix_order(order):
    fixed = []
    for p in order:
        if p in actual:
            fixed.append(p)
        else:
            alt = [a for a in actual if set(a.split("–")) == set(p.split("–"))]
            if alt: fixed.append(alt[0])
    return fixed

low_order  = fix_order(low_order)
high_order = fix_order(high_order)
ALL_ORDER  = low_order + high_order

print("Force pairs (Low):",  low_order)
print("Force pairs (High):", high_order)

def save_fig(fig, stem):
    save_export_figure(fig, OUTPUT_DIR, stem, EXPORT_WIDTHS_PX)

# =============================================================================
# 4. Two-panel layout helper
# =============================================================================
def make_two_panel(fig_h=None):
    sns.set_theme(style="white")
    ATD.apply_plot_style()
    low_r, high_r = horizontal_panel_rects()
    fig = plt.figure(figsize=EXPORT_CANVAS, facecolor="#FFFFFF")
    ax_l = fig.add_axes(low_r)
    ax_r = fig.add_axes(high_r)
    return fig, ax_l, ax_r

def add_legend(fig, handles, ncol=None):
    add_figure_legend(fig, handles, ncol=ncol, fontsize=FONT_LABEL)

# Pair colors within each band
LOW_PAIR_COLORS  = ["#08519C", "#2171B5", "#6BAED6", "#BDD7E7"]
HIGH_PAIR_COLORS = ["#A50F15", "#DE2D26", "#FC8D59"]

def pair_colors(band):
    return LOW_PAIR_COLORS if band == "Low" else HIGH_PAIR_COLORS

def pair_markers():
    return ["o", "s", "^", "D", "v", "P", "X"]


# =============================================================================
# FIGURE 1: Identity Scatter
# =============================================================================
def make_identity_scatter():
    fig, ax_l, ax_r = make_two_panel()

    for ax, band, order in [(ax_l, "Low", low_order), (ax_r, "High", high_order)]:
        sub   = wide[wide["band"] == band]
        marks = pair_markers()
        colors = pair_colors(band)
        handles = []

        for i, pair in enumerate(order):
            d = sub[sub["pair_label"] == pair]
            ax.scatter(d["Off-nail"], d["On-nail"],
                       color=colors[i], marker=marks[i],
                       s=55, alpha=ALPHA_DOT, zorder=4,
                       edgecolors="white", linewidths=0.4)
            handles.append(mlines.Line2D([], [], color=colors[i], marker=marks[i],
                                         markersize=7, linewidth=0,
                                         label=f"{pair} g"))

        # Identity line
        ax.plot([0, 100], [0, 100], color=COLOR_ZERO, lw=1.1,
                linestyle="--", alpha=0.6, zorder=2, label="y = x  (no difference)")

        # JND / chance lines
        ax.axhline(75, color=ATD.CRITERION_COLOR, linestyle=":", lw=0.8, alpha=0.5)
        ax.axvline(75, color=ATD.CRITERION_COLOR, linestyle=":", lw=0.8, alpha=0.5)
        ax.axhline(50, color="#AAAAAA", linestyle=":", lw=0.7, alpha=0.4)
        ax.axvline(50, color="#AAAAAA", linestyle=":", lw=0.7, alpha=0.4)

        ax.set_xlim(-5, 108)
        ax.set_ylim(-5, 108)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.tick_params(labelsize=FONT_TICK, length=0)
        ax.set_xlabel("Off-nail accuracy (%)", fontsize=FONT_LABEL,
                      labelpad=ATD.FIG_AXIS_LABELPAD)
        ax.set_title(
            f"{'Low' if band=='Low' else 'High'} Band  "
            f"(ref = {'1' if band=='Low' else '26'} g)",
            fontsize=FONT_LABEL, fontweight="bold", pad=6)
        if ax is ax_l:
            ax.set_ylabel("On-nail accuracy (%)", fontsize=FONT_LABEL,
                          labelpad=ATD.FIG_AXIS_LABELPAD)
        ax.grid(False)
        sns.despine(ax=ax)
        ax.set_aspect("equal", adjustable="box")

        # legend per panel
        handles.append(mlines.Line2D([], [], color=COLOR_ZERO, lw=1.1,
                                     linestyle="--", alpha=0.7, label="y = x"))
        ax.legend(handles=handles, fontsize=FONT_ANNOT, frameon=False,
                  loc="upper left", handletextpad=0.4, borderpad=0.3)

    fig.suptitle("On-nail vs Off-nail Accuracy per Subject",
                 fontsize=FONT_LABEL + 1, fontweight="bold", y=1.01)
    return fig

fig = make_identity_scatter()
save_fig(fig, "fd_region_identity_scatter")
plt.close(fig)


# =============================================================================
# FIGURE 2: Effect Size Forest (Cohen's d)
# =============================================================================
def cohens_d(a, b):
    """Paired Cohen's d = mean(diff) / SD(diff)."""
    diff = np.array(a) - np.array(b)
    return np.mean(diff) / (np.std(diff, ddof=1) + 1e-12)

def cohens_d_ci(a, b, conf=0.95, n_boot=5000, seed=0):
    """Bootstrap CI for paired Cohen's d."""
    rng  = np.random.default_rng(seed)
    diff = np.array(a) - np.array(b)
    n    = len(diff)
    ds   = [cohens_d(*[rng.choice(diff, size=n, replace=True) + np.zeros(n),
                       np.zeros(n)]) for _ in range(n_boot)]
    lo, hi = np.percentile(ds, [(1-conf)/2*100, (1+conf)/2*100])
    return lo, hi

def make_forest():
    sns.set_theme(style="white")
    ATD.apply_plot_style()

    rows = []
    for pair in ALL_ORDER:
        d_sub = wide[wide["pair_label"] == pair]
        if len(d_sub) < 4: continue
        a, b  = d_sub["On-nail"].values, d_sub["Off-nail"].values
        d_val = cohens_d(a, b)
        lo, hi = cohens_d_ci(a, b)
        band  = d_sub["band"].iloc[0]
        rows.append(dict(pair=pair, d=d_val, lo=lo, hi=hi, band=band))

    df_es = pd.DataFrame(rows)
    # Order: Low then High, top to bottom
    order_all = low_order + high_order
    df_es["yi"] = df_es["pair"].apply(lambda p: order_all.index(p))
    df_es = df_es.sort_values("yi", ascending=False).reset_index(drop=True)

    n = len(df_es)
    fig = plt.figure(figsize=EXPORT_CANVAS, facecolor="#FFFFFF")
    ax = fig.add_axes([0.28, 0.12, 0.65, 0.76])

    SMALL_D  = 0.2   # conventional small effect threshold

    for i, row in df_es.iterrows():
        color = COLOR_LOW if row["band"] == "Low" else COLOR_HIGH
        # CI bar
        ax.plot([row["lo"], row["hi"]], [i, i],
                color=color, lw=1.8, solid_capstyle="round", zorder=3)
        # Point estimate
        ax.scatter(row["d"], i, color=color, s=65, zorder=5,
                   edgecolors="white", linewidths=0.5)
        # Text: d value
        ax.text(max(row["hi"], SMALL_D) + 0.05, i,
                f"d = {row['d']:+.2f}", va="center",
                fontsize=FONT_ANNOT - 0.5, color=color)

    # Reference lines
    ax.axvline(0, color=COLOR_ZERO, lw=1.2, linestyle="-", alpha=0.7, zorder=1)
    ax.axvspan(-SMALL_D, SMALL_D, color="#EEEEEE", alpha=0.55, zorder=0,
               label=f"Negligible effect (|d| < {SMALL_D})")

    ax.set_yticks(range(n))
    ax.set_yticklabels(df_es["pair"].tolist(), fontsize=FONT_TICK)
    ax.set_xlabel("Cohen's d  (On-nail − Off-nail, paired)", fontsize=FONT_LABEL,
                  labelpad=ATD.FIG_AXIS_LABELPAD)
    ax.set_title("Effect Size: Regional Difference in Force Discrimination",
                 fontsize=FONT_LABEL, fontweight="bold", pad=8)
    ax.tick_params(axis="both", length=0, labelsize=FONT_TICK)
    ax.grid(False)
    sns.despine(ax=ax)

    # Band labels on left margin
    low_ys  = [i for i, r in df_es.iterrows() if r["band"] == "Low"]
    high_ys = [i for i, r in df_es.iterrows() if r["band"] == "High"]
    if low_ys:
        ax.annotate("Low\nBand", xy=(-0.27, np.mean(low_ys)),
                    xycoords=("axes fraction", "data"),
                    fontsize=FONT_ANNOT, color=COLOR_LOW, fontweight="bold",
                    ha="center", va="center")
    if high_ys:
        ax.annotate("High\nBand", xy=(-0.27, np.mean(high_ys)),
                    xycoords=("axes fraction", "data"),
                    fontsize=FONT_ANNOT, color=COLOR_HIGH, fontweight="bold",
                    ha="center", va="center")

    leg_handles = [
        mpatches.Patch(color="#DDDDDD", label=f"Negligible effect  |d| < {SMALL_D}"),
        mlines.Line2D([], [], color=COLOR_LOW,  lw=2, marker="o",
                      markersize=6, label="Low band (ref = 1 g)"),
        mlines.Line2D([], [], color=COLOR_HIGH, lw=2, marker="o",
                      markersize=6, label="High band (ref = 26 g)"),
    ]
    ax.legend(handles=leg_handles, fontsize=FONT_ANNOT, frameon=False,
              loc="lower right", borderpad=0.4)

    return fig

fig = make_forest()
save_fig(fig, "fd_region_effect_forest")
plt.close(fig)


# =============================================================================
# FIGURE 3: Bland-Altman
# =============================================================================
def make_bland_altman():
    fig, ax_l, ax_r = make_two_panel()

    for ax, band, order in [(ax_l, "Low", low_order), (ax_r, "High", high_order)]:
        sub    = wide[wide["band"] == band]
        colors = pair_colors(band)
        marks  = pair_markers()
        handles = []

        all_diffs = sub["diff"].values
        mean_diff = np.mean(all_diffs)
        sd_diff   = np.std(all_diffs, ddof=1)
        loa_lo    = mean_diff - 1.96 * sd_diff
        loa_hi    = mean_diff + 1.96 * sd_diff

        for i, pair in enumerate(order):
            d = sub[sub["pair_label"] == pair]
            ax.scatter(d["mean_acc"], d["diff"],
                       color=colors[i], marker=marks[i],
                       s=55, alpha=ALPHA_DOT, zorder=4,
                       edgecolors="white", linewidths=0.4)
            handles.append(mlines.Line2D([], [], color=colors[i], marker=marks[i],
                                         markersize=7, linewidth=0, label=f"{pair} g"))

        x_min, x_max = -3, 105

        # Mean difference line
        ax.axhline(mean_diff, color=ACCENT_RED, lw=1.4, linestyle="-",
                   zorder=3, label=f"Mean diff = {mean_diff:+.1f}%")
        # Limits of agreement
        ax.axhline(loa_hi, color=ACCENT_RED, lw=0.9, linestyle="--",
                   alpha=0.75, label=f"+1.96 SD = {loa_hi:+.1f}%")
        ax.axhline(loa_lo, color=ACCENT_RED, lw=0.9, linestyle="--",
                   alpha=0.75, label=f"−1.96 SD = {loa_lo:+.1f}%")
        # Zero line
        ax.axhline(0, color=COLOR_ZERO, lw=0.8, linestyle=":", alpha=0.6)

        # Shade between LoA
        ax.fill_between([x_min, x_max], loa_lo, loa_hi,
                        color=ACCENT_RED, alpha=0.05, zorder=0)

        # Annotate LoA and mean
        for y, label in [(mean_diff, f"Mean\n{mean_diff:+.1f}%"),
                          (loa_hi,   f"+1.96SD\n{loa_hi:+.1f}%"),
                          (loa_lo,   f"−1.96SD\n{loa_lo:+.1f}%")]:
            ax.text(x_max + 1, y, label, va="center", ha="left",
                    fontsize=FONT_ANNOT - 1.0, color=ACCENT_RED, clip_on=False)

        ax.set_xlim(x_min, x_max)
        ylim = max(80, abs(loa_lo) + 15, abs(loa_hi) + 15)
        ax.set_ylim(-ylim, ylim)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_yticks([-75, -50, -25, 0, 25, 50, 75])
        ax.tick_params(labelsize=FONT_TICK, length=0)
        ax.set_xlabel("Mean accuracy: (On-nail + Off-nail) / 2  (%)",
                      fontsize=FONT_LABEL, labelpad=ATD.FIG_AXIS_LABELPAD)
        ax.set_title(
            f"{'Low' if band=='Low' else 'High'} Band  "
            f"(ref = {'1' if band=='Low' else '26'} g)",
            fontsize=FONT_LABEL, fontweight="bold", pad=6)
        if ax is ax_l:
            ax.set_ylabel("Difference: On-nail − Off-nail  (%)",
                          fontsize=FONT_LABEL, labelpad=ATD.FIG_AXIS_LABELPAD)
        ax.grid(False)
        sns.despine(ax=ax)
        ax.legend(handles=handles, fontsize=FONT_ANNOT, frameon=False,
                  loc="upper right", handletextpad=0.4)

    fig.suptitle("Bland-Altman Agreement: On-nail vs Off-nail",
                 fontsize=FONT_LABEL + 1, fontweight="bold", y=1.01)
    return fig

fig = make_bland_altman()
save_fig(fig, "fd_region_bland_altman")
plt.close(fig)


# =============================================================================
# FIGURE 4: TOST Equivalence Plot
# =============================================================================
def tost_ci(a, b, conf=0.90):
    """
    Paired t-test mean difference and 90% CI for TOST.
    Returns (mean_diff, lo_90, hi_90, p_tost_lo, p_tost_hi).
    TOST null: |mean diff| >= delta (equivalence margin).
    """
    diff = np.array(a) - np.array(b)
    n    = len(diff)
    m    = np.mean(diff)
    se   = stats.sem(diff)
    t_crit = stats.t.ppf((1 + conf) / 2, df=n - 1)
    lo   = m - t_crit * se
    hi   = m + t_crit * se
    return m, lo, hi

def make_tost(delta_pct=15.0):
    """
    delta_pct: equivalence margin in percentage points.
    If 90% CI is entirely within [-delta, +delta], equivalence is established.
    """
    sns.set_theme(style="white")
    ATD.apply_plot_style()

    rows = []
    for pair in ALL_ORDER:
        d_sub = wide[wide["pair_label"] == pair]
        if len(d_sub) < 4: continue
        a, b = d_sub["On-nail"].values, d_sub["Off-nail"].values
        m, lo, hi = tost_ci(a, b, conf=0.90)
        band = d_sub["band"].iloc[0]
        n    = len(d_sub)
        equiv = (lo >= -delta_pct) and (hi <= delta_pct)
        rows.append(dict(pair=pair, mean=m, lo=lo, hi=hi, band=band,
                         n=n, equiv=equiv))

    df_t = pd.DataFrame(rows)
    order_all = low_order + high_order
    df_t["yi"] = df_t["pair"].apply(lambda p: order_all.index(p))
    df_t = df_t.sort_values("yi", ascending=False).reset_index(drop=True)

    n_rows = len(df_t)
    fig = plt.figure(figsize=EXPORT_CANVAS, facecolor="#FFFFFF")
    ax = fig.add_axes([0.24, 0.12, 0.60, 0.76])

    for i, row in df_t.iterrows():
        color  = COLOR_LOW if row["band"] == "Low" else COLOR_HIGH
        equiv  = row["equiv"]
        lw     = 2.2

        # 90% CI bar
        ax.plot([row["lo"], row["hi"]], [i, i],
                color=color, lw=lw, solid_capstyle="round", zorder=4,
                alpha=1.0)
        # Point estimate
        ms = 80 if equiv else 60
        mk = "D" if equiv else "o"
        ax.scatter(row["mean"], i, color=color, s=ms, zorder=6,
                   marker=mk, edgecolors="white", linewidths=0.5)

        # Equivalence verdict text
        verdict = "✓ Equiv." if equiv else "—"
        v_color = "#228B22" if equiv else "#888888"
        ax.text(delta_pct + 2, i, verdict, va="center", ha="left",
                fontsize=FONT_ANNOT - 0.5, color=v_color, fontweight="bold",
                clip_on=False)
        # n
        ax.text(-delta_pct - 2, i, f"n={row['n']}", va="center", ha="right",
                fontsize=FONT_ANNOT - 1.0, color="#666666", clip_on=False)

    # Equivalence zone shading
    ax.axvspan(-delta_pct, delta_pct, color="#CCEECC", alpha=0.40, zorder=0,
               label=f"Equivalence zone (±{delta_pct:.0f}%)")
    ax.axvline(-delta_pct, color="#228B22", lw=0.9, linestyle="--", alpha=0.7)
    ax.axvline( delta_pct, color="#228B22", lw=0.9, linestyle="--", alpha=0.7)
    ax.axvline(0, color=COLOR_ZERO, lw=1.0, linestyle="-", alpha=0.6)

    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(df_t["pair"].tolist(), fontsize=FONT_TICK)
    ax.set_xlabel(
        "Mean difference: On-nail − Off-nail  (%)\n"
        "90% CI for equivalence test  (TOST)",
        fontsize=FONT_LABEL, labelpad=ATD.FIG_AXIS_LABELPAD)
    ax.set_title("TOST Equivalence: On-nail ≡ Off-nail?",
                 fontsize=FONT_LABEL, fontweight="bold", pad=8)
    ax.tick_params(axis="both", length=0, labelsize=FONT_TICK)

    xlim = delta_pct * 2.0
    ax.set_xlim(-xlim, xlim)

    # Band labels
    low_ys  = [i for i, r in df_t.iterrows() if r["band"] == "Low"]
    high_ys = [i for i, r in df_t.iterrows() if r["band"] == "High"]
    if low_ys:
        ax.annotate("Low Band", xy=(-0.22, np.mean(low_ys)),
                    xycoords=("axes fraction", "data"),
                    fontsize=FONT_ANNOT, color=COLOR_LOW, fontweight="bold",
                    ha="center", va="center", rotation=90)
    if high_ys:
        ax.annotate("High Band", xy=(-0.22, np.mean(high_ys)),
                    xycoords=("axes fraction", "data"),
                    fontsize=FONT_ANNOT, color=COLOR_HIGH, fontweight="bold",
                    ha="center", va="center", rotation=90)

    ax.grid(False)
    sns.despine(ax=ax)

    leg_handles = [
        mpatches.Patch(color="#CCEECC", alpha=0.7,
                       label=f"Equivalence zone  ±{delta_pct:.0f}%  (90% CI)"),
        mlines.Line2D([], [], color=COLOR_LOW,  lw=2, marker="D", markersize=7,
                      label="Low band — equivalent (✓)"),
        mlines.Line2D([], [], color=COLOR_HIGH, lw=2, marker="D", markersize=7,
                      label="High band — equivalent (✓)"),
        mlines.Line2D([], [], color="#888888",  lw=2, marker="o", markersize=7,
                      label="Not yet equivalent"),
    ]
    ax.legend(handles=leg_handles, fontsize=FONT_ANNOT - 0.5, frameon=False,
              loc="lower right", borderpad=0.4, handletextpad=0.5)

    fig.tight_layout()
    return fig

fig = make_tost(delta_pct=15.0)
save_fig(fig, "fd_region_tost")
plt.close(fig)

print("\nAll four figures saved to:", OUTPUT_DIR)
print("  1. fd_region_identity_scatter  — Identity scatter (On vs Off)")
print("  2. fd_region_effect_forest     — Effect size forest (Cohen's d)")
print("  3. fd_region_bland_altman      — Bland-Altman agreement")
print("  4. fd_region_tost              — TOST equivalence CI")