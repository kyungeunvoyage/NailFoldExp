"""
Detection–Discrimination Dissociation Visualizations
=====================================================
Highlights the paradox:
  - 0.6 g is EASIER to detect (ATD > 80%)
  - BUT the 0.6–1 g pair is HARDER to discriminate (FD < chance)
  - 0.4 g is HARDER to detect (ATD ~60%)
  - BUT the 0.4–1 g pair is EASIER to discriminate (FD > criterion)

Two figures:
  Fig 1. Aligned Dual-Panel  — ATD (top) aligned with FD (bottom) on same force x-axis
  Fig 2. Regime Map          — heatmap of FD accuracy in (comparison × reference) space
                               with ATD threshold line overlaid

Run:
    python fd_atd_dissociation.py
"""

import os, glob, importlib.util
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from pathlib import Path
from scipy import stats

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

ACCENT_RED   = ATD.ACCENT_RED
SLATE_BLUE   = ATD.SLATE_BLUE
FONT_TICK    = ATD.FONT_TICK
FONT_LABEL   = ATD.FONT_LABEL
FONT_ANNOT   = ATD.FONT_ANNOT
FIG_SIZE     = ATD.FIG_SIZE
SAVE_DPI     = ATD.SAVE_DPI
EXPORT_WIDTHS_PX = ATD.EXPORT_WIDTHS_PX

# Colors
COLOR_ATD        = "#4A7FB5"     # detection curve
COLOR_FD_LOW     = "#2C6E9E"     # FD low band
COLOR_FD_HIGH    = "#C0392B"     # FD high band
COLOR_HIGHLIGHT  = "#E67E22"     # 0.4g / 0.6g highlight
COLOR_CRITERION  = ATD.CRITERION_COLOR
ALPHA_FILL       = 0.15
BOX_STROKE       = "#000000"

# =============================================================================
# 1. Paths
# =============================================================================
REPO_ROOT = "/Users/kyungeunjung/NailFoldExp"

ATD_PATTERN = os.path.join(
    REPO_ROOT, "Data", "(ATD)CurData", "P*_AbsoluteThresholdDetection.csv"
)
FD_PATTERN = os.path.join(
    REPO_ROOT, "Data", "(FD)CurData", "P*_ForceDiscrimination.csv"
)
OUTPUT_DIR = os.path.join(
    REPO_ROOT, "(New)Analysis", "ForceDiscAnalysis", "Force_ATD_outputs", "Dissociation"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# 2. Load ATD data
# =============================================================================
atd_files = sorted(glob.glob(ATD_PATTERN))
if not atd_files:
    raise FileNotFoundError(f"No ATD files found: {ATD_PATTERN}")
print(f"ATD: {len(atd_files)} participant file(s).")

df_atd = pd.concat(
    [pd.read_csv(f, encoding="utf-8-sig") for f in atd_files], ignore_index=True
)

# Force column: strip 'g' suffix → float
df_atd["force_val"] = df_atd["Force"].str.replace("g", "", regex=False).astype(float)

# On-touch only (matches FD condition)
df_atd_touch = df_atd[df_atd["Condition"].str.contains("On-touch", case=False)].copy()

# P61/P62/P63: partial-protocol — only include their 0.4 g data
_ATD_PARTIAL = {"P61", "P62", "P63"}
_is_partial = df_atd_touch["SubjectID"].isin(_ATD_PARTIAL)
df_atd_touch = df_atd_touch[~_is_partial | (df_atd_touch["force_val"] == 0.4)].copy()
print(f"ATD after partial-subject filter: {df_atd_touch['SubjectID'].nunique()} participants")

# Score metric (consistent with ATD_C1_Fig: partial credit for near-correct responses)
def _atd_score(row):
    if row["Target"] == 0:
        return 100.0 if row["Response"] == 0 else 0.0
    return max(0.0, (1 - abs(row["Target"] - row["Response"]) / row["Target"]) * 100.0)

df_atd_touch["score"] = df_atd_touch.apply(_atd_score, axis=1)

# Per-subject mean score per force level
atd_subj = (
    df_atd_touch.groupby(["SubjectID", "force_val"])["score"]
    .mean().reset_index().rename(columns={"score": "accuracy_pct"})
)

# Group mean ± SE per force level
atd_group = (
    atd_subj.groupby("force_val")["accuracy_pct"]
    .agg(mean="mean", se=lambda x: x.std(ddof=1) / np.sqrt(len(x)), n="count")
    .reset_index()
)

ATD_FORCE_ORDER = [0.07, 0.16, 0.4, 0.6, 1.0, 1.4]
atd_group = atd_group[atd_group["force_val"].isin(ATD_FORCE_ORDER)].copy()
atd_group = atd_group.sort_values("force_val").reset_index(drop=True)
print("ATD forces:", atd_group["force_val"].tolist())
print(atd_group[["force_val","mean","n"]].to_string(index=False))

# =============================================================================
# 3. Load FD data
# =============================================================================
fd_files = sorted(glob.glob(FD_PATTERN))
if not fd_files:
    raise FileNotFoundError(f"No FD files found: {FD_PATTERN}")
print(f"FD: {len(fd_files)} participant file(s).")

df_fd = pd.concat(
    [pd.read_csv(f, encoding="utf-8-sig") for f in fd_files], ignore_index=True
)

df_fd["correct"] = np.where(
    df_fd["Comparison"] > df_fd["Reference"],
    df_fd["ChoseComparison"] == 1,
    df_fd["ChoseComparison"] == 0,
).astype(int)

df_fd["pair_label"] = df_fd.apply(
    lambda r: f"{min(r['Reference'], r['Comparison']):g}–"
              f"{max(r['Reference'], r['Comparison']):g}", axis=1
)
df_fd["band"] = df_fd["Reference"].apply(lambda r: "Low" if r == 1 else "High")

# Relative contrast = |comp - ref| / ref
df_fd["rel_contrast"] = np.abs(df_fd["Comparison"] - df_fd["Reference"]) / df_fd["Reference"]

# Comparison force (lighter of the two for low band, heavier for high band)
# For dual-panel alignment: use the non-reference force as x position
df_fd["comp_force"] = df_fd.apply(
    lambda r: r["Comparison"] if r["Comparison"] != r["Reference"] else r["Reference"],
    axis=1
)

# Per-subject accuracy per force pair
fd_subj = (
    df_fd.groupby(["Subject", "band", "pair_label", "Reference", "comp_force", "rel_contrast"])
    ["correct"].mean().reset_index().rename(columns={"correct": "accuracy"})
)
fd_subj["accuracy_pct"] = fd_subj["accuracy"] * 100

# Group mean ± SE per force pair
fd_group = (
    fd_subj.groupby(["band", "pair_label", "Reference", "comp_force", "rel_contrast"])
    ["accuracy_pct"]
    .agg(mean="mean", se=lambda x: x.std(ddof=1) / np.sqrt(len(x)), n="count")
    .reset_index()
)

# Low band only for aligned panel
fd_low = fd_group[fd_group["band"] == "Low"].copy()
fd_high = fd_group[fd_group["band"] == "High"].copy()
print("FD pairs (Low):",  fd_low["pair_label"].tolist())
print("FD pairs (High):", fd_high["pair_label"].tolist())

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
        Image.Resampling.LANCZOS).save(legacy)
    print(f"Saved → {legacy}")

# =============================================================================
# FIGURE 1: Aligned Dual-Panel
# =============================================================================
# Key forces to highlight
HIGHLIGHT_FORCES = {0.4: "0.4 g", 0.6: "0.6 g"}
HIGHLIGHT_COLOR  = COLOR_HIGHLIGHT

def make_aligned_panel():
    sns.set_theme(style="white")
    ATD.apply_plot_style()

    fig = plt.figure(figsize=(FIG_SIZE[0] * 1.05, FIG_SIZE[1] * 1.7), facecolor="#FFFFFF")

    # Shared x-axis range (force value in g)
    x_min, x_max = 0.03, 2.2

    # ── Top panel: ATD ───────────────────────────────────────────────────────
    ax_atd = fig.add_axes([0.12, 0.54, 0.82, 0.38])

    # Shade highlight forces vertically
    for fv in HIGHLIGHT_FORCES:
        ax_atd.axvspan(fv * 0.85, fv * 1.18, color=HIGHLIGHT_COLOR,
                       alpha=0.10, zorder=0)

    # Individual subject dots
    for _, row in atd_subj.iterrows():
        if row["force_val"] not in ATD_FORCE_ORDER: continue
        ax_atd.scatter(row["force_val"], row["accuracy_pct"],
                       color=COLOR_ATD, s=18, alpha=0.30,
                       edgecolors="none", zorder=3)

    # Mean ± SE line
    ax_atd.plot(atd_group["force_val"], atd_group["mean"],
                color=COLOR_ATD, lw=2.0, zorder=5, marker="o",
                markersize=7, markeredgecolor="white", markeredgewidth=0.5)
    ax_atd.fill_between(atd_group["force_val"],
                        atd_group["mean"] - atd_group["se"],
                        atd_group["mean"] + atd_group["se"],
                        color=COLOR_ATD, alpha=ALPHA_FILL, zorder=2)

    # 80% criterion & 50% chance lines
    ax_atd.axhline(80, color=COLOR_CRITERION, linestyle="--",
                   lw=1.0, alpha=0.8, zorder=1)
    ax_atd.axhline(50, color="#888888", linestyle=":", lw=0.8, alpha=0.5)

    # Annotate 0.4g and 0.6g
    for fv, label in HIGHLIGHT_FORCES.items():
        row = atd_group[atd_group["force_val"] == fv]
        if row.empty: continue
        m = row["mean"].values[0]
        ax_atd.annotate(f"{label}\n({m:.0f}%)",
                        xy=(fv, m), xytext=(fv, m + 12),
                        fontsize=FONT_ANNOT, color=HIGHLIGHT_COLOR,
                        fontweight="bold", ha="center",
                        arrowprops=dict(arrowstyle="-", color=HIGHLIGHT_COLOR,
                                        lw=1.0, alpha=0.7))

    ax_atd.set_xscale("log")
    ax_atd.set_xlim(x_min, x_max)
    ax_atd.set_ylim(-5, 115)
    ax_atd.set_yticks([0, 25, 50, 75, 100])
    ax_atd.set_ylabel("Detection Accuracy (%)", fontsize=FONT_LABEL,
                       labelpad=ATD.FIG_AXIS_LABELPAD)
    ax_atd.set_title("A    Absolute Detection Threshold (On-touch)",
                     fontsize=FONT_LABEL, fontweight="bold", loc="left", pad=6)
    ax_atd.tick_params(labelsize=FONT_TICK, length=0)
    ax_atd.set_xticklabels([])   # hide x labels (shared with bottom)
    ax_atd.grid(False)
    sns.despine(ax=ax_atd)
    ATD.apply_accuracy_y_spine_bounds(ax_atd)

    # ── Bottom panel: FD (Low band) ───────────────────────────────────────────
    ax_fd = fig.add_axes([0.12, 0.10, 0.82, 0.38])

    # Shade highlight forces
    for fv in HIGHLIGHT_FORCES:
        ax_fd.axvspan(fv * 0.85, fv * 1.18, color=HIGHLIGHT_COLOR,
                      alpha=0.10, zorder=0)

    # All FD pairs: plot at comparison force x-position
    for _, row in fd_low.iterrows():
        cf = row["comp_force"]
        m  = row["mean"]
        se = row["se"]
        pair = row["pair_label"]

        ax_fd.errorbar(cf, m, yerr=se,
                       fmt="o", color=COLOR_FD_LOW,
                       markersize=7, markeredgecolor="white",
                       markeredgewidth=0.5,
                       capsize=3, capthick=0.9, elinewidth=1.0,
                       ecolor=COLOR_FD_LOW, zorder=5)
        # Pair label
        ax_fd.text(cf, m - 8, pair + " g", ha="center",
                   fontsize=FONT_ANNOT - 1.5, color=COLOR_FD_LOW, alpha=0.85)

    # Criterion and chance lines
    ax_fd.axhline(75, color=COLOR_CRITERION, linestyle="--",
                  lw=1.0, alpha=0.8, zorder=1)
    ax_fd.axhline(50, color="#888888", linestyle=":", lw=0.8, alpha=0.5, zorder=1)

    # Annotate the two key points
    key_pairs = {"0.4–1": (0.4, "0.4–1 g\n(above criterion)"),
                 "0.6–1": (0.6, "0.6–1 g\n(below chance)")}
    for pair_key, (cf, ann_text) in key_pairs.items():
        match = fd_low[fd_low["pair_label"].str.startswith(pair_key.split("–")[0])]
        if match.empty: continue
        m = match["mean"].values[0]
        dy = 14 if m > 50 else -18
        ax_fd.annotate(ann_text,
                       xy=(cf, m), xytext=(cf, m + dy),
                       fontsize=FONT_ANNOT, color=HIGHLIGHT_COLOR,
                       fontweight="bold", ha="center",
                       arrowprops=dict(arrowstyle="-", color=HIGHLIGHT_COLOR,
                                       lw=1.0, alpha=0.7))

    ax_fd.set_xscale("log")
    ax_fd.set_xlim(x_min, x_max)
    ax_fd.set_ylim(-5, 115)
    ax_fd.set_yticks([0, 25, 50, 75, 100])
    ax_fd.set_ylabel("Discrimination Accuracy (%)", fontsize=FONT_LABEL,
                      labelpad=ATD.FIG_AXIS_LABELPAD)
    ax_fd.set_xlabel("Force (g)", fontsize=FONT_LABEL,
                     labelpad=ATD.FIG_AXIS_LABELPAD)
    ax_fd.set_title("B    Force Discrimination — Low Band  (ref = 1 g)",
                    fontsize=FONT_LABEL, fontweight="bold", loc="left", pad=6)
    ax_fd.tick_params(labelsize=FONT_TICK, length=0)
    ax_fd.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda v, _: f"{v:g}"))
    ax_fd.grid(False)
    sns.despine(ax=ax_fd)
    ATD.apply_accuracy_y_spine_bounds(ax_fd)

    # ── Connecting vertical lines between panels ──────────────────────────────
    for fv, color in [(0.4, HIGHLIGHT_COLOR), (0.6, HIGHLIGHT_COLOR)]:
        # transFigure coordinates
        ax_atd_pos = ax_atd.transData.transform((fv, 0))
        ax_fd_pos  = ax_fd.transData.transform((fv, 115))
        fig_atd    = fig.transFigure.inverted().transform(ax_atd_pos)
        fig_fd     = fig.transFigure.inverted().transform(ax_fd_pos)
        line = plt.Line2D(
            [fig_atd[0], fig_fd[0]], [fig_atd[1], fig_fd[1]],
            transform=fig.transFigure, color=color,
            lw=1.2, linestyle="--", alpha=0.45, zorder=10)
        fig.add_artist(line)

    # ── Legend ───────────────────────────────────────────────────────────────
    handles = [
        mlines.Line2D([], [], color=COLOR_ATD, lw=2, marker="o",
                      markersize=6, label="ATD: On-touch (this study)"),
        mlines.Line2D([], [], color=COLOR_FD_LOW, lw=0, marker="o",
                      markersize=6, label="FD: Low band (ref = 1 g)"),
        mlines.Line2D([], [], color=COLOR_CRITERION, lw=1.2, linestyle="--",
                      label="75/80% criterion"),
        mlines.Line2D([], [], color="#888888", lw=1.0, linestyle=":",
                      label="50% chance"),
        mpatches.Patch(color=HIGHLIGHT_COLOR, alpha=0.25,
                       label="Highlighted forces: 0.4 g & 0.6 g"),
    ]
    fig.legend(handles=handles, loc="upper center",
               bbox_to_anchor=(0.5, 1.00), bbox_transform=fig.transFigure,
               ncol=3, fontsize=FONT_LABEL, frameon=False,
               columnspacing=1.5, handletextpad=0.5)

    return fig

fig = make_aligned_panel()
save_fig(fig, "dissociation_aligned_panel")
plt.close(fig)


# =============================================================================
# FIGURE 1b: Overlay Panel (ATD left-axis + FD right-axis, single panel)
# =============================================================================
COLOR_FD_OVERLAY = "#C0392B"   # red for FD in overlay panel

# Forces to keep in overlay panel
OV_ATD_FORCES = [0.16, 0.4, 0.6, 1.4]   # ATD x positions
OV_FD_COMPS   = [0.4, 0.6, 1.4]          # FD comp_force values (0.4–1, 0.6–1, 1–1.4)
OV_X_TICKS    = [0.16, 0.4, 0.6, 1.4]


def make_overlay_panel():
    sns.set_theme(style="white")
    ATD.apply_plot_style()

    fig, ax_atd = plt.subplots(figsize=(FIG_SIZE[0] * 1.05, FIG_SIZE[1] * 1.0),
                               facecolor="#FFFFFF")
    ax_fd = ax_atd.twinx()

    x_min, x_max = 0.12, 1.80

    # ── Background highlight bands (0.4 g, 0.6 g, 1.4 g) ─────────────────────
    # axvspan ymin/ymax are in axes fraction; start from plot bottom, end at y=100
    _ylim_b, _ylim_t = -5, 115
    _hl_ymax = (100 - _ylim_b) / (_ylim_t - _ylim_b)   # = 0.875
    for fv in [0.4, 0.6, 1.4]:
        ax_atd.axvspan(fv * 0.88, fv * 1.14, ymin=0, ymax=_hl_ymax,
                       color=HIGHLIGHT_COLOR, alpha=0.10, zorder=0)

    # ── ATD: individual subject dots ───────────────────────────────────────────
    atd_subj_ov = atd_subj[atd_subj["force_val"].isin(OV_ATD_FORCES)]
    for _, row in atd_subj_ov.iterrows():
        ax_atd.scatter(row["force_val"], row["accuracy_pct"],
                       color=COLOR_ATD, s=18, alpha=0.28,
                       edgecolors="none", zorder=3)

    # ── ATD: mean ± SE line ────────────────────────────────────────────────────
    atd_ov = atd_group[atd_group["force_val"].isin(OV_ATD_FORCES)].sort_values("force_val")
    ax_atd.plot(atd_ov["force_val"], atd_ov["mean"],
                color=COLOR_ATD, lw=2.0, zorder=5, marker="o",
                markersize=7, markeredgecolor="white", markeredgewidth=0.5)
    ax_atd.fill_between(atd_ov["force_val"],
                        atd_ov["mean"] - atd_ov["se"],
                        atd_ov["mean"] + atd_ov["se"],
                        color=COLOR_ATD, alpha=ALPHA_FILL, zorder=2)

    # n=3 annotation removed

    # ── ATD: criterion lines ───────────────────────────────────────────────────
    ax_atd.axhline(80, color=COLOR_CRITERION, linestyle="--",
                   lw=1.0, alpha=0.8, zorder=1)
    ax_atd.axhline(50, color="#888888", linestyle=":", lw=0.8, alpha=0.5)

    # ── ATD: annotate 0.4 g and 0.6 g ─────────────────────────────────────────
    # atd_annot_cfg = {
    #     0.4: dict(xytext=(0.38, 28), ha="center"),
    #     0.6: dict(xytext=(0.66, 52), ha="center"),
    # }
    # for fv, label in HIGHLIGHT_FORCES.items():
    #     row = atd_ov[atd_ov["force_val"] == fv]
    #     if row.empty: continue
    #     m   = row["mean"].values[0]
    #     cfg = atd_annot_cfg.get(fv)
    #     if cfg is None: continue
        # ax_atd.annotate(f"{label}\nATD: {m:.0f}%",
        #                 xy=(fv, m), xytext=cfg["xytext"],
        #                 fontsize=FONT_ANNOT + 2, color=COLOR_ATD,
        #                 fontweight="bold", ha=cfg["ha"],
        #                 arrowprops=dict(arrowstyle="-", color=COLOR_ATD,
        #                                 lw=0.9, alpha=0.7))

    ax_atd.set_xscale("log")
    ax_atd.set_xlim(x_min, x_max)
    ax_atd.set_ylim(-5, 115)
    ax_atd.set_yticks([0, 25, 50, 75, 100])
    ax_atd.set_ylabel("Detection Accuracy (%)", fontsize=FONT_LABEL + 4,
                       color=COLOR_ATD, labelpad=ATD.FIG_AXIS_LABELPAD)
    ax_atd.tick_params(axis="y", labelsize=FONT_TICK + 4, colors=COLOR_ATD, length=4, direction="in")
    ax_atd.set_xlabel("Force (g)", fontsize=FONT_LABEL + 4,
                      labelpad=ATD.FIG_AXIS_LABELPAD)
    ax_atd.set_xticks(OV_X_TICKS)
    ax_atd.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax_atd.xaxis.set_minor_locator(ticker.NullLocator())
    ax_atd.tick_params(axis="x", labelsize=FONT_TICK + 4, length=0, which="both")
    ax_atd.spines["left"].set_color(COLOR_ATD)
    ax_atd.grid(False)
    sns.despine(ax=ax_atd, right=False)
    ax_atd.spines["right"].set_visible(False)

    # ── FD: pairs at comp_force 0.4, 0.6, 1.4 (all Low band, ref=1) ──────────
    fd_ov = fd_low[fd_low["comp_force"].isin(OV_FD_COMPS)].sort_values("comp_force")
    for _, row in fd_ov.iterrows():
        cf   = row["comp_force"]
        m    = row["mean"]
        se   = row["se"]
        pair = row["pair_label"]

        ax_fd.errorbar(cf, m, yerr=se,
                       fmt="s", color=COLOR_FD_OVERLAY,
                       markersize=7, markeredgecolor="white",
                       markeredgewidth=0.5,
                       capsize=3, capthick=0.9, elinewidth=1.0,
                       ecolor=COLOR_FD_OVERLAY, zorder=5)
        label_dy = -9 if m > 50 else 5
        # ax_fd.text(cf, m + label_dy, pair + " g", ha="center",
        #            fontsize=FONT_ANNOT + 1, color=COLOR_FD_OVERLAY, alpha=0.85)

    # 75% discrimination criterion line removed

    # ── FD: annotate key pairs ─────────────────────────────────────────────────
    _above_pairs = {"0.4"}   # place label above marker for these pairs
    for _, row in fd_ov.iterrows():
        cf   = row["comp_force"]
        m    = row["mean"]
        se   = row["se"]
        pair = row["pair_label"]   # e.g. "0.4–1"
        key  = pair.split("–")[0]
        label_text = f"FD: {pair} g"
        if key in _above_pairs:
            y_pos = m + se + 7
            va = "bottom"
        else:
            y_pos = m - se - 7
            va = "top"
        ax_fd.text(cf, y_pos, label_text, ha="center", va=va,
                   fontsize=FONT_ANNOT + 5, color=COLOR_FD_OVERLAY,
                   fontweight="extra bold")

    ax_fd.set_ylim(-5, 115)
    ax_fd.set_yticks([0, 25, 50, 75, 100])
    ax_fd.set_ylabel("Discrimination Accuracy (%)", fontsize=FONT_LABEL + 4,
                      color=COLOR_FD_OVERLAY, labelpad=ATD.FIG_AXIS_LABELPAD)
    ax_fd.tick_params(axis="y", labelsize=FONT_TICK + 4, colors=COLOR_FD_OVERLAY, length=4, direction="in")
    ax_fd.spines["right"].set_visible(True)
    ax_fd.spines["right"].set_color(COLOR_FD_OVERLAY)

    # ── Combined legend ────────────────────────────────────────────────────────
    handles = [
        mlines.Line2D([], [], color=COLOR_ATD, lw=2, marker="o",
                      markersize=6, label="Absolute threshold detection \n(On-touch)"),
        mlines.Line2D([], [], color=COLOR_FD_OVERLAY, lw=0, marker="s",
                      markersize=6, label="Force discrimination \n(Low band, ref = 1 g)"),
        # mlines.Line2D([], [], color=COLOR_CRITERION, lw=1.2, linestyle="--",
        #               label="80% detection criterion"),
        # mlines.Line2D([], [], color=COLOR_FD_OVERLAY, lw=1.2, linestyle="--",
        #               alpha=0.55, label="75% discrimination criterion"),
        # mlines.Line2D([], [], color="#888888", lw=1.0, linestyle=":",
        #               label="50% chance"),
        # mpatches.Patch(color=HIGHLIGHT_COLOR, alpha=0.25,
        #                label="Highlighted forces: 0.4 g, 0.6 g & 1.4 g"),
    ]
    fig.legend(handles=handles, loc="upper center",
               bbox_to_anchor=(0.5, 1.01), bbox_transform=fig.transFigure,
               ncol=3, fontsize=FONT_LABEL + 3, frameon=False,
               columnspacing=1.2, handletextpad=0.5)

    fig.subplots_adjust(top=0.91)

    # Reinforce left y-axis ticks after twinx setup (twinx can reset the locator)
    ax_atd.yaxis.set_major_locator(ticker.FixedLocator([0, 25, 50, 75, 100]))
    ax_atd.set_yticklabels(["0", "25", "50", "75", "100"],
                            fontsize=FONT_TICK + 4, color=COLOR_ATD)
    # Spine bounds: start at ylim bottom (-5) so the spine meets the x-axis,
    # end at 100 so it doesn't extend past the top tick
    _ylim_bot = -5
    ax_atd.spines["left"].set_bounds(_ylim_bot, 100)
    ax_atd.spines["top"].set_visible(False)

    # Hide ax_fd's extra spines (twinx keeps all 4 by default);
    # only the right spine should show, also bounded same way
    for _sp in ("left", "top", "bottom"):
        ax_fd.spines[_sp].set_visible(False)
    ax_fd.spines["right"].set_bounds(_ylim_bot, 100)

    return fig


if not os.getenv("PAPER_RENDER"):
    fig_ov = make_overlay_panel()
    import io as _io
    from PIL import Image as _PILImage
    _buf = _io.BytesIO()
    fig_ov.savefig(_buf, format="png", dpi=SAVE_DPI, bbox_inches="tight",
                   pad_inches=0.05, facecolor="white")
    _buf.seek(0)
    _master_ov = _PILImage.open(_buf).convert("RGB")
    _OV_W, _OV_H = 2102, 1113
    for _tag, _w in EXPORT_WIDTHS_PX:
        _h = round(_w * _OV_H / _OV_W)
        _master_ov.resize((_w, _h), _PILImage.Resampling.LANCZOS).save(
            os.path.join(OUTPUT_DIR, f"dissociation_overlay_panel_{_tag}.png"))
    _master_ov.resize((_OV_W, _OV_H), _PILImage.Resampling.LANCZOS).save(
        os.path.join(OUTPUT_DIR, "dissociation_overlay_panel.png"))
    print(f"Saved → dissociation_overlay_panel  ({_OV_W}×{_OV_H} px)")
    plt.close(fig_ov)


# =============================================================================
# FIGURE 2: Regime Map (FD heatmap + ATD threshold line)
# =============================================================================
def make_regime_map():
    sns.set_theme(style="white")
    ATD.apply_plot_style()

    # Build pivot: rows = reference force, cols = comparison force
    # All unique (reference, comparison) pairs in FD data
    fd_all = (
        df_fd.groupby(["Reference", "comp_force"])["correct"]
        .mean().reset_index().rename(columns={"correct": "accuracy"})
    )
    fd_all["accuracy_pct"] = fd_all["accuracy"] * 100

    refs  = sorted(fd_all["Reference"].unique())
    comps = sorted(fd_all["comp_force"].unique())
    print("References:", refs)
    print("Comparisons:", comps)

    # All forces for axes
    all_forces = sorted(set(refs) | set(comps))

    # Build matrix: rows=reference, cols=comparison
    mat = pd.DataFrame(np.nan, index=refs, columns=all_forces)
    for _, row in fd_all.iterrows():
        mat.loc[row["Reference"], row["comp_force"]] = row["accuracy_pct"]

    # Color map: below chance = blue, above criterion = red, middle = white
    # Custom diverging: blue(0%) → white(50%) → red(100%)
    cmap = LinearSegmentedColormap.from_list(
        "disc_map",
        [(0.0,  "#3B82C4"),   # deep blue  (below chance)
         (0.45, "#B0C8E0"),   # light blue
         (0.50, "#FFFFFF"),   # white      (50% = chance)
         (0.55, "#F5C0B0"),   # light red
         (1.0,  "#C94040")],  # deep red   (above criterion)
        N=256
    )

    fig = plt.figure(figsize=(FIG_SIZE[0] * 1.0, FIG_SIZE[1] * 1.1),
                     facecolor="#FFFFFF")
    ax  = fig.add_axes([0.14, 0.13, 0.72, 0.76])

    # Heatmap
    n_r, n_c = len(refs), len(all_forces)
    img = ax.imshow(mat.values, aspect="auto", cmap=cmap,
                    vmin=0, vmax=100, origin="lower",
                    extent=[-0.5, n_c - 0.5, -0.5, n_r - 0.5])

    # Text annotations per cell
    for ri, ref in enumerate(refs):
        for ci, comp in enumerate(all_forces):
            val = mat.loc[ref, comp]
            if np.isnan(val): continue
            text_color = "white" if (val < 30 or val > 85) else "#333333"
            ax.text(ci, ri, f"{val:.0f}%",
                    ha="center", va="center",
                    fontsize=FONT_ANNOT, color=text_color, fontweight="bold")

    # ATD threshold line: draw where force = ATD 80% threshold
    # Estimate ATD 80% threshold by interpolation
    atd_sorted = atd_group.sort_values("force_val")
    try:
        # Find where mean crosses 80% on the way up
        above = atd_sorted[atd_sorted["mean"] >= 80]
        if not above.empty:
            atd_threshold = above["force_val"].values[0]
        else:
            atd_threshold = atd_sorted["force_val"].values[-1]
        print(f"ATD 80% threshold ≈ {atd_threshold} g")

        # Draw vertical line at threshold (on comp_force axis)
        if atd_threshold in all_forces:
            ci_thresh = list(all_forces).index(atd_threshold)
        else:
            # Interpolate position
            ci_thresh = np.interp(atd_threshold, all_forces, range(len(all_forces)))

        ax.axvline(ci_thresh, color=COLOR_CRITERION, lw=2.0,
                   linestyle="--", alpha=0.9, zorder=5,
                   label=f"ATD 80% threshold (~{atd_threshold:g} g)")

        ax.text(ci_thresh + 0.08, n_r - 0.6,
                f"ATD\nthreshold\n(~{atd_threshold:g} g)",
                fontsize=FONT_ANNOT, color=COLOR_CRITERION,
                fontweight="bold", va="top", ha="left")
    except Exception as e:
        print(f"Could not draw ATD threshold line: {e}")

    # Highlight 0.4g and 0.6g columns
    for fv in [0.4, 0.6]:
        if fv in all_forces:
            ci = list(all_forces).index(fv)
            ax.add_patch(plt.Rectangle(
                (ci - 0.5, -0.5), 1, n_r,
                fill=False, edgecolor=HIGHLIGHT_COLOR,
                lw=2.5, linestyle="-", zorder=6))
            ax.text(ci, -0.85, f"↑ {fv:g} g",
                    ha="center", va="top",
                    fontsize=FONT_ANNOT, color=HIGHLIGHT_COLOR,
                    fontweight="bold")

    # Axes labels
    ax.set_xticks(range(len(all_forces)))
    ax.set_xticklabels([f"{f:g}" for f in all_forces], fontsize=FONT_TICK)
    ax.set_yticks(range(len(refs)))
    ax.set_yticklabels([f"{r:g}" for r in refs], fontsize=FONT_TICK)
    ax.set_xlabel("Comparison Force (g)", fontsize=FONT_LABEL,
                  labelpad=ATD.FIG_AXIS_LABELPAD)
    ax.set_ylabel("Reference Force (g)", fontsize=FONT_LABEL,
                  labelpad=ATD.FIG_AXIS_LABELPAD)
    ax.set_title("Force Discrimination Accuracy: Regime Map\n"
                 "with ATD threshold overlay",
                 fontsize=FONT_LABEL, fontweight="bold", pad=8)
    ax.tick_params(length=0, labelsize=FONT_TICK)

    # Colorbar
    cbar_ax = fig.add_axes([0.89, 0.13, 0.025, 0.76])
    cb = fig.colorbar(img, cax=cbar_ax)
    cb.set_label("Discrimination Accuracy (%)", fontsize=FONT_LABEL,
                 labelpad=6)
    cb.ax.tick_params(labelsize=FONT_TICK, length=0)
    cb.ax.axhline(50, color="#333333", lw=1.2, linestyle=":")
    cb.ax.axhline(75, color=COLOR_CRITERION, lw=1.2, linestyle="--")
    cb.ax.text(2.5, 50, "Chance", fontsize=FONT_ANNOT - 1,
               va="center", color="#333333")
    cb.ax.text(2.5, 75, "Criterion", fontsize=FONT_ANNOT - 1,
               va="center", color=COLOR_CRITERION)

    # Legend
    handles = [
        mpatches.Patch(color="#3B82C4", label="Below chance  (< 50%)"),
        mpatches.Patch(color="#FFFFFF", edgecolor="#AAAAAA",
                       label="Near chance  (~50%)"),
        mpatches.Patch(color="#C94040", label="Above criterion  (> 75%)"),
        mlines.Line2D([], [], color=COLOR_CRITERION, lw=2, linestyle="--",
                      label="ATD 80% threshold"),
        mpatches.Patch(facecolor="none", edgecolor=HIGHLIGHT_COLOR, lw=2,
                       label="Key forces: 0.4 g & 0.6 g"),
    ]
    fig.legend(handles=handles, loc="upper center",
               bbox_to_anchor=(0.48, 1.00), bbox_transform=fig.transFigure,
               ncol=3, fontsize=FONT_ANNOT, frameon=False,
               columnspacing=1.2, handletextpad=0.4)

    return fig

fig = make_regime_map()
save_fig(fig, "dissociation_regime_map")
plt.close(fig)

print("\nDone. Files saved to:", OUTPUT_DIR)
print("  1. dissociation_aligned_panel  — ATD (top) + FD (bottom), aligned on force axis")
print("  2. dissociation_regime_map     — FD heatmap with ATD threshold overlaid")