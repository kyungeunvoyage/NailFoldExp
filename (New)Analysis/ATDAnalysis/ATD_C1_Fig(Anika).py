"""
ATD Comparison Figures
======================
Figure 1: Anika Paint (fingerpad, Kao et al. 2022) vs. Periungual In-air (this study)
Figure 2: Periungual On-touch vs. In-air (this study)

Kao et al. 2022 Paint condition values are digitized from published Fig. B (n=5).
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib import rcParams

# =============================================================================
# Palette
# =============================================================================
SLATE_BLUE = "#56708A"   # In-air
OLIVE      = "#686F12"   # On-touch
WINE       = "#7F212B"   # 80% line
BLACK      = "#1A1A1A"
KAO_COLOR  = "#5A5A5A"   # Anika Paint — dark gray (matches original paper)

BOX_ALPHA_HEX = "CC"     # ~80% opacity
STRIP_ALPHA   = 0.50

FIG_SIZE  = (8.0, 4.5)
SAVE_DPI  = 300   # PNG raster export
SAVE_SVG  = True  # vector copy (resolution-independent)

# =============================================================================
# Paths
# =============================================================================
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT    = os.path.normpath(os.path.join(SCRIPT_DIR, "../../"))
FILE_PATTERN = os.path.join(
    REPO_ROOT, "Data", "(ATD)CurData", "P*_AbsoluteThresholdDetection.csv"
)
OUT_DIR = os.path.join(SCRIPT_DIR, "atd_c1_outputs")
os.makedirs(OUT_DIR, exist_ok=True)


def save_figure(fig, stem):
    """Save PNG (high-DPI raster) and SVG (vector, max effective resolution)."""
    png_path = os.path.join(OUT_DIR, f"{stem}.png")
    fig.savefig(png_path, dpi=SAVE_DPI, bbox_inches="tight", facecolor="white")
    print(f"Saved PNG → {png_path}  ({FIG_SIZE[0]}×{FIG_SIZE[1]} in @ {SAVE_DPI} dpi)")

    if SAVE_SVG:
        svg_path = os.path.join(OUT_DIR, f"{stem}.svg")
        fig.savefig(
            svg_path,
            format="svg",
            bbox_inches="tight",
            facecolor="white",
            metadata={"Creator": "ATD_C1_Fig(Anika).py"},
        )
        print(f"Saved SVG → {svg_path}  (vector)")


# =============================================================================
# rcParams
# =============================================================================
rcParams.update({
    "figure.facecolor":      "#FFFFFF",
    "axes.facecolor":        "#FFFFFF",
    "font.family":           "sans-serif",
    "font.sans-serif":       ["Helvetica", "Arial", "DejaVu Sans"],
    "axes.linewidth":        0.8,
    "axes.spines.top":       False,
    "axes.spines.right":     False,
    "xtick.major.width":     0.8,
    "ytick.major.width":     0.8,
    "xtick.major.size":      3.5,
    "ytick.major.size":      3.5,
    "xtick.direction":       "out",
    "ytick.direction":       "out",
    "legend.frameon":        False,
    "legend.fontsize":       8.5,
    "legend.title_fontsize": 8.5,
    "font.size":             9,
    "axes.titlesize":        10,
    "axes.labelsize":        9.5,
    "figure.dpi":            150,
    "savefig.dpi":           SAVE_DPI,
    "svg.fonttype":          "path",   # embed fonts as paths (Illustrator-safe)
})

# =============================================================================
# Kao et al. 2022 — Paint condition
# Digitized from Fig. B of the paper (n=5, index fingerpad)
# y = Percent correct (%)
# =============================================================================
KAO_PAINT_RAW = {
    # force(g) : [P1, P2, P3, P4, P5]
    0.02: [  0,  30,  47,  65,  68],
    0.04: [ 35,  78,  82,  88,  97],
    0.07: [ 32,  60,  82,  88, 100],
    0.40: [ 32,  78,  80,  87, 100],
    1.00: [ 48, 100, 100, 100, 100],
    1.40: [ 80, 100, 100, 100, 100],
}
KAO_N = 5

# Build tidy DataFrame
kao_rows = []
for force, vals in KAO_PAINT_RAW.items():
    for pid, v in enumerate(vals):
        kao_rows.append({
            "Force_Val":  float(force),
            "Score":      float(v),
            "Source":     f"Fingerpad — Paint\n(Kao et al. 2022, n={KAO_N})",
            "Participant": f"KP{pid+1}",
        })
df_kao = pd.DataFrame(kao_rows)

# =============================================================================
# Load user data
# =============================================================================
all_files = glob.glob(FILE_PATTERN)
if not all_files:
    raise FileNotFoundError(f"No CSVs found:\n  {FILE_PATTERN}")
print(f"Loaded {len(all_files)} participant file(s).")

df_raw = pd.concat(
    [pd.read_csv(f) for f in sorted(all_files)],
    ignore_index=True,
)
df_raw["Condition"] = df_raw["Condition"].str.strip().replace({
    "Active":          "On-touch (Mid)",
    "On-touch (Hard)": "On-touch (Mid)",
    "Passive":         "In-air",
})
df_raw = df_raw[df_raw["Condition"] != "On-touch (Soft)"]
df_raw = df_raw[df_raw["Area"].isin(["A", "B", "C", "D", "E", "F"])].copy()
df_raw["Force_Val"] = df_raw["Force"].str.extract(r"(\d+\.?\d*)").astype(float)

n_subjects = df_raw["SubjectID"].nunique() if "SubjectID" in df_raw.columns else len(all_files)


def calc_score(row):
    if row["Target"] == 0:
        return 100.0 if row["Response"] == 0 else 0.0
    return max(0.0, (1 - abs(row["Target"] - row["Response"]) / row["Target"]) * 100.0)


df_raw["Score"] = df_raw.apply(calc_score, axis=1)

USER_FORCES   = sorted(df_raw["Force_Val"].unique())   # [0.07, 0.16, 0.6, 1.0, 1.4]
KAO_FORCES    = sorted(KAO_PAINT_RAW.keys())           # [0.02, 0.04, 0.07, 0.4, 1.0, 1.4]

# =============================================================================
# Figure 1 — Anika Paint (fingerpad) vs Periungual In-air
# =============================================================================
INAIR_LABEL = f"Periungual — In-air\n(this study, n= {n_subjects})"
KAO_LABEL   = f"Fingerpad — On-touch\n(Kao et al. 2022, n= {KAO_N})"

df_inair = df_raw[df_raw["Condition"] == "In-air"].copy()
df_inair["Source"] = INAIR_LABEL

df_kao_plot = df_kao.copy()
df_kao_plot["Source"] = KAO_LABEL

# Combined force axis (union, ascending)
COMBINED_FORCES = sorted(set(KAO_FORCES) | set(USER_FORCES))
# = [0.02, 0.04, 0.07, 0.16, 0.4, 0.6, 1.0, 1.4]

df_fig1 = pd.concat(
    [df_kao_plot[["Force_Val", "Score", "Source"]],
     df_inair[["Force_Val",  "Score", "Source"]]],
    ignore_index=True,
)

SOURCE_ORDER  = [KAO_LABEL, INAIR_LABEL]
SOURCE_COLORS = {KAO_LABEL: KAO_COLOR, INAIR_LABEL: SLATE_BLUE}

sns.set_theme(style="white")
fig1, ax1 = plt.subplots(figsize=FIG_SIZE)

# Boxplots
sns.boxplot(
    data=df_fig1,
    x="Force_Val", y="Score",
    hue="Source", hue_order=SOURCE_ORDER,
    order=COMBINED_FORCES,
    palette={s: SOURCE_COLORS[s] + BOX_ALPHA_HEX for s in SOURCE_ORDER},
    linewidth=0.8, fliersize=0, width=0.55,
    medianprops={"color": WINE, "linewidth": 2.0},
    whiskerprops={"linewidth": 0.8, "color": BLACK},
    capprops={"linewidth": 0.8, "color": BLACK},
    boxprops={"linewidth": 0.8},
    legend=False, ax=ax1,
)

# Strip plots
sns.stripplot(
    data=df_fig1,
    x="Force_Val", y="Score",
    hue="Source", hue_order=SOURCE_ORDER,
    order=COMBINED_FORCES,
    palette=SOURCE_COLORS,
    dodge=True, alpha=STRIP_ALPHA,
    size=3.8, jitter=0.15, linewidth=0,
    legend=False, ax=ax1,
)

# 80% reference line
ax1.axhline(80, color=WINE, linestyle="--", linewidth=1.0, alpha=0.85, zorder=0)
ax1.text(len(COMBINED_FORCES) - 0.55, 81.5, "80%",
         ha="right", va="bottom", fontsize=8, color=WINE)

# Shade: forces where ONLY fingerpad has data (0.02, 0.04) → KAO_COLOR tint
kao_only  = [f for f in KAO_FORCES  if f not in USER_FORCES]
user_only = [f for f in USER_FORCES if f not in KAO_FORCES]

def xspan(forces_sub, all_forces, pad=0.48):
    idxs = [list(all_forces).index(f) for f in forces_sub if f in all_forces]
    return (min(idxs) - pad, max(idxs) + pad) if idxs else None

sp_kao  = xspan(kao_only,  COMBINED_FORCES)
sp_user = xspan(user_only, COMBINED_FORCES)

if sp_kao:
    ax1.axvspan(*sp_kao, color=KAO_COLOR, alpha=0.06, zorder=0)
    mid = (sp_kao[0] + sp_kao[1]) / 2
    ax1.text(mid, 105, "Fingerpad\nonly",
             ha="center", va="top", fontsize=7, color=KAO_COLOR, style="italic")

if sp_user:
    ax1.axvspan(*sp_user, color=SLATE_BLUE, alpha=0.06, zorder=0)
    mid = (sp_user[0] + sp_user[1]) / 2
    ax1.text(mid, 105, "Periungual\nonly",
             ha="center", va="top", fontsize=7, color=SLATE_BLUE, style="italic")

# Legend
leg_handles = [
    mpatches.Patch(facecolor=KAO_COLOR + BOX_ALPHA_HEX,
                   edgecolor=BLACK, linewidth=0.7, label=KAO_LABEL),
    mpatches.Patch(facecolor=SLATE_BLUE + BOX_ALPHA_HEX,
                   edgecolor=BLACK, linewidth=0.7, label=INAIR_LABEL),
]
ax1.legend(handles=leg_handles, loc="lower right",
           fontsize=8, frameon=False, labelspacing=0.5)

ax1.set_xlabel("Stimulus Force (g)", labelpad=6)
ax1.set_ylabel("Detection Accuracy (%)", labelpad=6)
ax1.set_ylim(-5, 115)
ax1.set_xticklabels([str(f) for f in COMBINED_FORCES])

sns.despine(ax=ax1)
fig1.tight_layout(pad=1.5)

save_figure(fig1, "Fig1_fingerpad_paint_vs_inair")
plt.close(fig1)

# =============================================================================
# Figure 2 — Periungual On-touch vs In-air
# =============================================================================
COND_ORDER  = ["In-air", "On-touch (Mid)"]
COND_COLORS = {"In-air": SLATE_BLUE, "On-touch (Mid)": OLIVE}
cond_list   = [c for c in COND_ORDER if c in df_raw["Condition"].unique()]

fig2, ax2 = plt.subplots(figsize=FIG_SIZE)

sns.boxplot(
    data=df_raw,
    x="Force_Val", y="Score",
    hue="Condition", hue_order=cond_list,
    order=USER_FORCES,
    palette={c: COND_COLORS[c] + BOX_ALPHA_HEX for c in cond_list},
    linewidth=0.8, fliersize=0, width=0.55,
    medianprops={"color": BLACK, "linewidth": 2.0},
    whiskerprops={"linewidth": 0.8, "color": BLACK},
    capprops={"linewidth": 0.8, "color": BLACK},
    boxprops={"linewidth": 0.8},
    legend=False, ax=ax2,
)

sns.stripplot(
    data=df_raw,
    x="Force_Val", y="Score",
    hue="Condition", hue_order=cond_list,
    order=USER_FORCES,
    palette=COND_COLORS,
    dodge=True, alpha=STRIP_ALPHA,
    size=3.5, jitter=0.18, linewidth=0,
    legend=False, ax=ax2,
)

if ax2.get_legend() is not None:
    ax2.get_legend().remove()

ax2.axhline(80, color=WINE, linestyle="--", linewidth=1.0, alpha=0.85, zorder=0)
ax2.text(len(USER_FORCES) - 0.55, 81.5, "80%",
         ha="right", va="bottom", fontsize=8, color=WINE)

leg_handles2 = [
    mpatches.Patch(facecolor=COND_COLORS[c] + BOX_ALPHA_HEX,
                   edgecolor=BLACK, linewidth=0.7, label=c)
    for c in cond_list
]
ax2.legend(handles=leg_handles2, title="Condition", loc="lower right",
           fontsize=8.5, title_fontsize=8.5, frameon=False,
           handlelength=1.2, handleheight=0.9, labelspacing=0.3)

ax2.set_xlabel("Stimulus Force (g)", labelpad=6)
ax2.set_ylabel("Detection Accuracy (%)", labelpad=6)
ax2.set_ylim(-5, 110)
ax2.set_xticklabels([str(f) for f in USER_FORCES])

sns.despine(ax=ax2)
fig2.tight_layout(pad=1.5)

save_figure(fig2, "Fig2_ontouch_vs_inair")
plt.close(fig2)

print("\nDone.")