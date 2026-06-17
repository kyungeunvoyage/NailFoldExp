"""
================================================================
Spatial Discrimination Analysis  –  Periungual (C–D) Region
================================================================
Task   : 2-AFC  "Was the 2nd stimulus to the LEFT or RIGHT of the 1st?"
Region : C–D zone only (on-nail)
Forces : 1 g  &  26 g  (matched to FD reference forces)
Grid   : 1 unit = 1.5 mm  →  offsets = 1.5 / 3.0 / 4.5 / 6.0 mm

Signed offset = pos_2nd − pos_1st
  negative → 2nd is to the LEFT  (correct answer: "Left")
  positive → 2nd is to the RIGHT (correct answer: "Right")

Figures:
  1. sd_psychometric_curves  — per-subject thin lines + group mean±SE
                               for 1 g and 26 g; JND at 75% marked
  2. sd_jnd_paired           — per-subject JND slope plot (1 g vs 26 g)
  3. sd_symmetry             — accuracy by signed offset (left / right bias)

Statistics:
  • GEE (Binomial, exchangeable)  :  IsCorrect ~ abs_offset_mm × force_g
  • Wilcoxon signed-rank (paired) :  JND(1 g) vs JND(26 g)
================================================================
"""

import os, glob, re, importlib.util, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
from pathlib import Path
from scipy.optimize import curve_fit
from scipy import stats

warnings.filterwarnings("ignore")

# ================================================================
# 0. ATD style loader
# ================================================================
_SCRIPT_DIR = Path(__file__).resolve().parent
_ATD_PATH   = _SCRIPT_DIR.parent / "ATDAnalysis" / "ATD_C1_Fig(Anika).py"

def _load_atd():
    spec = importlib.util.spec_from_file_location("atd_c1_fig", _ATD_PATH)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

ATD = _load_atd()

FONT_TICK        = ATD.FONT_TICK
FONT_LABEL       = ATD.FONT_LABEL
FONT_ANNOT       = ATD.FONT_ANNOT
FIG_SIZE         = ATD.FIG_SIZE
SAVE_DPI         = ATD.SAVE_DPI
EXPORT_WIDTHS_PX = ATD.EXPORT_WIDTHS_PX
GAP_IN           = 1.5

COLOR_LOW  = "#2166AC"   # 1 g  → blue
COLOR_HIGH = "#C0392B"   # 26 g → red
FORCE_COLORS = {1.0: COLOR_LOW, 26.0: COLOR_HIGH}

GRID_SPACING_MM  = 1.5
THRESHOLD        = 0.75   # 75 % criterion for JND

# ================================================================
# 1. Paths
# ================================================================
REPO_ROOT  = "/Users/kyungeunjung/NailFoldExp"
SD_PATTERN = os.path.join(REPO_ROOT, "Data", "(SD)CurData",
                           "P*_SpatialDiscrimination.csv")
OUTPUT_DIR = os.path.join(REPO_ROOT, "(New)Analysis",
                           "SpatialAnalysis", "SDAnalysis1_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================================================================
# 2. Load & parse
# ================================================================
sd_files = sorted(glob.glob(SD_PATTERN))
if not sd_files:
    raise FileNotFoundError(f"No SD files found:\n  {SD_PATTERN}")
print(f"[Load] {len(sd_files)} participant file(s) found.")

def _parse_grid(s):
    """'g3' → 3.0,  'g-2' → -2.0"""
    m = re.match(r"g(-?\d+)", str(s).strip())
    return float(m.group(1)) if m else np.nan

def _parse_force(s):
    """'26.0g' → 26.0,  '1.0g' → 1.0"""
    m = re.match(r"([\d.]+)", str(s).strip())
    return float(m.group(1)) if m else np.nan

df = pd.concat(
    [pd.read_csv(f, encoding="utf-8-sig") for f in sd_files],
    ignore_index=True,
)

# Grid → mm
df["pos_1st_mm"] = df["Stim_1st"].apply(_parse_grid) * GRID_SPACING_MM
df["pos_2nd_mm"] = df["Stim_2nd"].apply(_parse_grid) * GRID_SPACING_MM

# Signed offset: positive = 2nd is to the RIGHT of 1st
df["signed_offset_mm"] = df["pos_2nd_mm"] - df["pos_1st_mm"]
df["abs_offset_mm"]    = df["signed_offset_mm"].abs()

df["force_g"]  = df["Force"].apply(_parse_force)
df["IsCorrect"] = pd.to_numeric(df["IsCorrect"], errors="coerce")
df = df.dropna(subset=["IsCorrect", "signed_offset_mm", "force_g"])

n_subj   = df["Subject"].nunique()
forces   = sorted(df["force_g"].unique())
offsets  = sorted(df["abs_offset_mm"].unique())

print(f"       Subjects : {n_subj}")
print(f"       Forces   : {forces} g")
print(f"       Offsets  : {offsets} mm")
print(f"       Rows     : {len(df)}")

# ================================================================
# 3. Per-subject summary per (force, abs_offset)
# ================================================================
subj_acc = (
    df.groupby(["Subject", "force_g", "abs_offset_mm"])
    .agg(accuracy=("IsCorrect", "mean"),
         n_trials=("IsCorrect", "count"))
    .reset_index()
)

# Group mean ± SE across subjects
grp_acc = (
    subj_acc.groupby(["force_g", "abs_offset_mm"])
    .agg(mean_acc=("accuracy", "mean"),
         se_acc  =("accuracy", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
         n_subj  =("accuracy", "count"))
    .reset_index()
)

# Signed offset summary (for symmetry plot)
subj_sym = (
    df.groupby(["Subject", "force_g", "signed_offset_mm"])
    .agg(accuracy=("IsCorrect", "mean"))
    .reset_index()
)
grp_sym = (
    subj_sym.groupby(["force_g", "signed_offset_mm"])
    .agg(mean_acc=("accuracy", "mean"),
         se_acc  =("accuracy", lambda x: x.std(ddof=1) / np.sqrt(len(x))))
    .reset_index()
)

# ================================================================
# 4. Psychometric function fitting
# ================================================================
def _psychometric(x, x50, beta, lapse=0.02):
    """
    Logistic psychometric with fixed guess=0.5 and free lapse.
    P(x) = 0.5 + (0.48 - lapse) / (1 + exp(-beta*(x - x50)))
    """
    return 0.5 + (0.48 - lapse) / (1.0 + np.exp(-beta * (x - x50)))

def _fit_curve(xs, ys, p0=(3.0, 1.0)):
    """Return (popt, success)."""
    try:
        popt, _ = curve_fit(
            _psychometric, xs, ys,
            p0=p0,
            bounds=([0.1, 0.05], [15.0, 10.0]),
            maxfev=8000,
        )
        return popt, True
    except Exception:
        return (np.nan, np.nan), False

def _jnd_from_fit(popt, target=THRESHOLD):
    """Numerically invert psychometric to find x at P=target."""
    x_arr = np.linspace(0, 20, 20000)
    y_arr = _psychometric(x_arr, *popt)
    idx   = np.argmin(np.abs(y_arr - target))
    return float(x_arr[idx])

# ── Group-level fits ──────────────────────────────────────────
print("\n[Psychometric] Group-level fits:")
group_fits = {}
for force in forces:
    sub  = grp_acc[grp_acc["force_g"] == force].sort_values("abs_offset_mm")
    xs, ys = sub["abs_offset_mm"].values, sub["mean_acc"].values
    popt, ok = _fit_curve(xs, ys)
    jnd  = _jnd_from_fit(popt) if ok else np.nan
    group_fits[force] = {"popt": popt, "ok": ok, "jnd": jnd, "data": sub}
    print(f"  {force:4.1f} g → JND = {jnd:.2f} mm  (x50={popt[0]:.2f}, β={popt[1]:.2f})")

# ── Per-subject fits ──────────────────────────────────────────
print("\n[Psychometric] Per-subject fits:")
subj_jnd_records = []
for (subj, force), grp in subj_acc.groupby(["Subject", "force_g"]):
    grp_s = grp.sort_values("abs_offset_mm")
    xs, ys = grp_s["abs_offset_mm"].values, grp_s["accuracy"].values
    if len(xs) < 3:
        continue
    popt, ok = _fit_curve(xs, ys)
    jnd = _jnd_from_fit(popt) if ok else np.nan
    subj_jnd_records.append({
        "Subject": subj, "force_g": force,
        "jnd_mm": jnd, "x50": popt[0], "beta": popt[1], "fit_ok": ok,
    })

jnd_df = pd.DataFrame(subj_jnd_records)
jnd_df.to_csv(os.path.join(OUTPUT_DIR, "jnd_per_subject.csv"), index=False)
print(f"  Saved jnd_per_subject.csv  (n = {len(jnd_df)} fits)")

# ── Wilcoxon paired test ──────────────────────────────────────
print("\n[Statistics] Wilcoxon signed-rank: JND(1g) vs JND(26g)")
wide = (jnd_df[jnd_df["fit_ok"]]
        .pivot(index="Subject", columns="force_g", values="jnd_mm")
        .dropna())

if 1.0 in wide.columns and 26.0 in wide.columns:
    stat, p_val = stats.wilcoxon(wide[1.0], wide[26.0])
    print(f"  W = {stat:.2f},  p = {p_val:.4f}  (n = {len(wide)})")
    print(f"  Median JND  1g : {wide[1.0].median():.2f} mm")
    print(f"  Median JND 26g : {wide[26.0].median():.2f} mm")
else:
    print("  Not enough data for paired test.")
    stat, p_val = np.nan, np.nan

# ── GEE ──────────────────────────────────────────────────────
print("\n[GEE] IsCorrect ~ abs_offset_mm × force_g  (Binomial, Exchangeable)")
try:
    from statsmodels.genmod.generalized_estimating_equations import GEE
    from statsmodels.genmod.cov_struct import Exchangeable
    from statsmodels.genmod.families import Binomial

    df_gee = df.copy()
    df_gee["force_centered"]  = df_gee["force_g"]       - df_gee["force_g"].mean()
    df_gee["offset_centered"] = df_gee["abs_offset_mm"] - df_gee["abs_offset_mm"].mean()

    gee_model  = GEE.from_formula(
        "IsCorrect ~ abs_offset_mm * force_g",
        groups="Subject",
        data=df_gee,
        cov_struct=Exchangeable(),
        family=Binomial(),
    )
    gee_result = gee_model.fit()
    gee_table  = gee_result.summary().tables[1]
    print(gee_table)
    with open(os.path.join(OUTPUT_DIR, "gee_results.txt"), "w") as f:
        f.write(str(gee_result.summary()))
    print("  Saved gee_results.txt")
except Exception as e:
    print(f"  GEE failed: {e}")
    gee_result = None

# ================================================================
# 5. Save helper (ATD style)
# ================================================================
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
    print(f"  → {legacy}")

# ================================================================
# 6. Layout helpers
# ================================================================
def _despine(ax):
    sns.despine(ax=ax)
    ax.tick_params(length=0, labelsize=FONT_TICK)
    ax.grid(False)

def _two_panel(fig_h=None):
    sns.set_theme(style="white")
    ATD.apply_plot_style()
    left_in  = 0.09 * FIG_SIZE[0]
    right_in = 0.03 * FIG_SIZE[0]
    pw       = (FIG_SIZE[0] - left_in - right_in - GAP_IN) / 2
    ph       = FIG_SIZE[1] * 0.68
    bot_in   = ATD.FIG_LEGEND_BOTTOM * FIG_SIZE[1]
    fig_h    = fig_h or (ph + bot_in + 0.6 + 0.3)
    ax_y     = bot_in / fig_h
    ax_h     = ph / fig_h
    fig      = plt.figure(figsize=(FIG_SIZE[0], fig_h), facecolor="#FFFFFF")
    ax_l = fig.add_axes([left_in / FIG_SIZE[0],
                         ax_y, pw / FIG_SIZE[0], ax_h])
    ax_r = fig.add_axes([(left_in + pw + GAP_IN) / FIG_SIZE[0],
                         ax_y, pw / FIG_SIZE[0], ax_h])
    return fig, ax_l, ax_r

# ================================================================
# FIGURE 1: Psychometric curves
# ================================================================
_x_smooth = np.linspace(0, offsets[-1] + 1.0, 300)
_xticks   = offsets
_xlabels  = [f"{x:g}" for x in _xticks]

def _psych_panel(ax, force, show_ylabel):
    color = FORCE_COLORS[force]
    info  = group_fits[force]
    sub   = subj_acc[subj_acc["force_g"] == force]
    grp   = grp_acc[grp_acc["force_g"] == force].sort_values("abs_offset_mm")

    # Per-subject lines (thin, transparent)
    for subj in sub["Subject"].unique():
        s = sub[sub["Subject"] == subj].sort_values("abs_offset_mm")
        ax.plot(s["abs_offset_mm"], s["accuracy"],
                color=color, alpha=0.15, lw=0.8, zorder=2)

    # Group mean ± SE
    ax.errorbar(grp["abs_offset_mm"], grp["mean_acc"],
                yerr=grp["se_acc"],
                fmt="o", color=color, markersize=6,
                markeredgecolor="white", markeredgewidth=0.5,
                capsize=3, capthick=1.0, elinewidth=1.2,
                zorder=5, label="Mean ± SE")

    # Fitted curve
    if info["ok"]:
        y_s = _psychometric(_x_smooth, *info["popt"])
        ax.plot(_x_smooth, y_s, color=color, lw=2.0, zorder=4,
                label=f"Fit (JND = {info['jnd']:.1f} mm)")
        # JND vertical marker
        ax.axvline(info["jnd"], color=color, lw=1.2,
                   linestyle="--", alpha=0.65, zorder=3)
        ax.scatter([info["jnd"]], [THRESHOLD],
                   color=color, s=55, zorder=6,
                   edgecolors="white", linewidths=0.5)

    # Reference lines
    ax.axhline(THRESHOLD, color="#555555", lw=1.0,
               linestyle=":", alpha=0.7, label=f"{int(THRESHOLD*100)}% criterion")
    ax.axhline(0.5, color="#AAAAAA", lw=0.8,
               linestyle=":", alpha=0.6, label="Chance (50%)")

    # Axes
    ax.set_xlim(0.5, offsets[-1] + 0.8)
    ax.set_ylim(0.30, 1.05)
    ax.set_xticks(_xticks)
    ax.set_xticklabels(_xlabels, fontsize=FONT_TICK)
    ax.set_yticks([0.50, 0.75, 1.00])
    ax.set_yticklabels(["50\n(chance)", "75\n(JND)", "100"],
                        fontsize=FONT_TICK)
    ax.set_xlabel("Absolute offset (mm)", fontsize=FONT_LABEL,
                  labelpad=ATD.FIG_AXIS_LABELPAD)
    if show_ylabel:
        ax.set_ylabel("Proportion correct (%)", fontsize=FONT_LABEL,
                      labelpad=ATD.FIG_AXIS_LABELPAD)
    ax.set_title(f"{force:g} g", fontsize=FONT_LABEL,
                 fontweight="bold", pad=6, color=color)
    ax.legend(fontsize=FONT_ANNOT, frameon=False,
              loc="lower right", handlelength=1.4)
    _despine(ax)


def make_psychometric_fig():
    fig, ax_l, ax_r = _two_panel()
    _psych_panel(ax_l, 1.0,  show_ylabel=True)
    _psych_panel(ax_r, 26.0, show_ylabel=False)
    return fig

print("\n[Figure 1] Psychometric curves ...")
fig = make_psychometric_fig()
save_fig(fig, "sd_psychometric_curves")
plt.close(fig)

# ================================================================
# FIGURE 2: Per-subject JND paired slope plot
# ================================================================
def make_jnd_paired():
    sns.set_theme(style="white")
    ATD.apply_plot_style()

    fw   = FIG_SIZE[0] * 0.52
    fh   = FIG_SIZE[1] * 1.0
    fig, ax = plt.subplots(figsize=(fw, fh), facecolor="#FFFFFF")

    jnd_ok = jnd_df[jnd_df["fit_ok"]].copy()
    forces_sorted = [1.0, 26.0]
    x_pos = {1.0: 0, 26.0: 1}
    rng   = np.random.default_rng(7)

    # Per-subject lines
    for subj in jnd_ok["Subject"].unique():
        sub = jnd_ok[jnd_ok["Subject"] == subj].sort_values("force_g")
        if len(sub) < 2:
            continue
        xs = [x_pos[f] + (rng.random() - 0.5) * 0.06 for f in sub["force_g"]]
        ys = sub["jnd_mm"].values
        ax.plot(xs, ys, color="#888888", alpha=0.35, lw=0.9, zorder=2)
        for xi, yi, f in zip(xs, ys, sub["force_g"]):
            ax.scatter(xi, yi, color=FORCE_COLORS[f], s=28,
                       edgecolors="white", linewidths=0.4,
                       alpha=0.75, zorder=4)

    # Group mean ± SE
    for force in forces_sorted:
        sub  = jnd_ok[jnd_ok["force_g"] == force]["jnd_mm"].dropna()
        m, s = sub.mean(), sub.std(ddof=1) / np.sqrt(len(sub))
        xi   = x_pos[force]
        ax.errorbar(xi, m, yerr=s,
                    fmt="D", color=FORCE_COLORS[force],
                    markersize=9, markeredgecolor="white",
                    markeredgewidth=0.6,
                    capsize=4, capthick=1.2, elinewidth=1.5,
                    zorder=6)
        ax.text(xi, m + s + 0.3, f"{m:.1f} mm",
                ha="center", va="bottom",
                fontsize=FONT_ANNOT, color=FORCE_COLORS[force],
                fontweight="bold")

    # Statistical annotation
    if not np.isnan(p_val):
        p_str = f"p = {p_val:.3f}" if p_val >= 0.001 else "p < 0.001"
        sig   = "n.s." if p_val > 0.05 else ("*" if p_val > 0.01 else "**")
        y_bar = jnd_ok["jnd_mm"].max() + 1.0
        ax.plot([0, 1], [y_bar, y_bar], color="#333333", lw=1.0)
        ax.text(0.5, y_bar + 0.15, f"{p_str}  {sig}",
                ha="center", va="bottom",
                fontsize=FONT_ANNOT, color="#333333")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["1 g", "26 g"], fontsize=FONT_LABEL)
    ax.set_xlim(-0.45, 1.45)
    ax.set_ylabel("JND (mm)", fontsize=FONT_LABEL,
                  labelpad=ATD.FIG_AXIS_LABELPAD)
    ax.set_title("Spatial JND by Force", fontsize=FONT_LABEL,
                 fontweight="bold", pad=8)
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.5))
    _despine(ax)
    fig.tight_layout()
    return fig

print("[Figure 2] JND paired slope plot ...")
fig = make_jnd_paired()
save_fig(fig, "sd_jnd_paired")
plt.close(fig)

# ================================================================
# FIGURE 3: Symmetry check  (signed offset: left vs right)
# ================================================================
def make_symmetry_fig():
    sns.set_theme(style="white")
    ATD.apply_plot_style()

    fw = FIG_SIZE[0] * 0.85
    fh = FIG_SIZE[1] * 0.9
    fig, ax = plt.subplots(figsize=(fw, fh), facecolor="#FFFFFF")

    signed_vals = sorted(grp_sym["signed_offset_mm"].unique())

    for force in forces:
        color = FORCE_COLORS[force]
        sub   = (grp_sym[grp_sym["force_g"] == force]
                 .sort_values("signed_offset_mm"))
        ax.errorbar(sub["signed_offset_mm"], sub["mean_acc"],
                    yerr=sub["se_acc"],
                    fmt="o-", color=color, markersize=6,
                    markeredgecolor="white", markeredgewidth=0.5,
                    capsize=3, capthick=0.9, elinewidth=1.0,
                    lw=1.5, zorder=4,
                    label=f"{force:g} g")

    ax.axvline(0, color="#AAAAAA", lw=1.0, linestyle="-", alpha=0.5)
    ax.axhline(THRESHOLD, color="#555555", lw=1.0, linestyle=":",
               alpha=0.7, label=f"{int(THRESHOLD*100)}% criterion")
    ax.axhline(0.5, color="#AAAAAA", lw=0.8, linestyle=":",
               alpha=0.6, label="Chance (50%)")

    # Symmetry annotation
    ax.text(-0.3, 0.33, "← 2nd to LEFT",
            ha="right", va="center", fontsize=FONT_ANNOT,
            color="#555555", style="italic")
    ax.text(0.3, 0.33, "2nd to RIGHT →",
            ha="left", va="center", fontsize=FONT_ANNOT,
            color="#555555", style="italic")

    ax.set_xlabel("Signed offset (mm)  [negative = 2nd is LEFT]",
                  fontsize=FONT_LABEL, labelpad=ATD.FIG_AXIS_LABELPAD)
    ax.set_ylabel("Proportion correct", fontsize=FONT_LABEL,
                  labelpad=ATD.FIG_AXIS_LABELPAD)
    ax.set_title("Left–Right Symmetry Check", fontsize=FONT_LABEL,
                 fontweight="bold", pad=8)
    ax.set_ylim(0.25, 1.05)
    ax.set_yticks([0.50, 0.75, 1.00])
    ax.set_yticklabels(["50%", "75%", "100%"], fontsize=FONT_TICK)
    ax.legend(fontsize=FONT_ANNOT, frameon=False,
              loc="upper left", handlelength=1.4)
    _despine(ax)
    fig.tight_layout()
    return fig

print("[Figure 3] Symmetry check ...")
fig = make_symmetry_fig()
save_fig(fig, "sd_symmetry")
plt.close(fig)

# ================================================================
# FIGURE 4: Signed-offset boxplot  (accuracy × distance × force)
#   X-axis  : Force group (1 g / 26 g)
#   Clusters : one box per signed distance within each force group
#   Color    : diverging warm→cool  (negative=warm, positive=cool)
# ================================================================

# Diverging colour palette matching the reference figure
DIST_PALETTE = {
    -6.0: "#D73027",   # dark red
    -4.5: "#F46D43",   # orange-red
    -3.0: "#FDAE61",   # light orange
    -1.5: "#FEE090",   # yellow
     1.5: "#D9EF8B",   # light yellow-green
     3.0: "#66BD63",   # green
     4.5: "#1A9850",   # dark green
     6.0: "#4393C3",   # blue
}

def _boxplot_panel(ax, force, signed_vals, show_ylabel=True):
    """Draw one force-condition panel onto the given axes."""
    import matplotlib.patches as mpatches
    bw  = 0.14
    rng = np.random.default_rng(42)
    x_positions = {dist: di * 0.18 for di, dist in enumerate(signed_vals)}

    for dist in signed_vals:
        xp    = x_positions[dist]
        color = DIST_PALETTE.get(dist, "#888888")
        sub   = (subj_sym[
                     (subj_sym["force_g"] == force) &
                     (subj_sym["signed_offset_mm"] == dist)
                 ]["accuracy"].dropna().values * 100)
        if len(sub) == 0:
            continue
        ax.boxplot(
            sub, positions=[xp], widths=bw * 0.82,
            patch_artist=True, showfliers=False,
            medianprops=dict(color="#333333", lw=2.0),
            whiskerprops=dict(color="#555555", lw=1.4),
            capprops=dict(color="#555555",   lw=1.4),
            boxprops=dict(facecolor=color, alpha=0.65,
                          edgecolor="#333333", lw=1.8),
        )
        jitter = rng.uniform(-bw * 0.30, bw * 0.30, len(sub))
        ax.scatter(xp + jitter, sub, color=color, s=28, alpha=0.80,
                   edgecolors="#444444", linewidths=0.4, zorder=5)

    ax.axhline(75, color="#CC0000", lw=1.4, linestyle="--", alpha=0.80, zorder=2)
    ax.axhline(50, color="#888888", lw=0.9, linestyle=":",  alpha=0.60, zorder=2)

    center = np.mean(list(x_positions.values()))
    ax.set_xticks([center])
    ax.set_xticklabels([f"{force:g} g"], fontsize=FONT_LABEL)
    ax.set_xlim(min(x_positions.values()) - bw, max(x_positions.values()) + bw)
    ax.set_ylim(-5, 108)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    if show_ylabel:
        ax.set_yticklabels(["0%", "20%", "40%", "60%", "80%", "100%"], fontsize=FONT_TICK)
        ax.set_ylabel("Accuracy (Correct Rate)", fontsize=FONT_LABEL,
                      labelpad=ATD.FIG_AXIS_LABELPAD)
    else:
        ax.set_yticklabels([""] * 6)
        ax.set_ylabel("")
    ax.set_xlabel("Force Condition", fontsize=FONT_LABEL, labelpad=ATD.FIG_AXIS_LABELPAD)
    ax.set_title(f"{force:g} g", fontsize=FONT_LABEL, fontweight="bold", pad=8)

    dist_handles = [
        mpatches.Patch(facecolor=DIST_PALETTE.get(d, "#888"),
                       edgecolor="#333333", linewidth=1.0, label=f"{d:+.1f} mm")
        for d in signed_vals
    ]
    ref_handles = [
        plt.Line2D([0], [0], color="#CC0000", lw=1.4, linestyle="--", label="75% criterion"),
        plt.Line2D([0], [0], color="#888888", lw=0.9, linestyle=":", label="Chance (50%)"),
    ]
    leg = ax.legend(handles=dist_handles + ref_handles, title="Distance (mm)",
                    title_fontsize=FONT_ANNOT, fontsize=FONT_ANNOT,
                    frameon=True, framealpha=0.9, edgecolor="#DDDDDD",
                    loc="upper right", bbox_to_anchor=(1.0, 1.0), ncol=1)
    leg.get_title().set_fontweight("bold")
    _despine(ax)


def make_signed_boxplot():
    """Two-panel figure: 1 g (left) | 26 g (right) on one canvas."""
    sns.set_theme(style="white")
    ATD.apply_plot_style()
    signed_vals = sorted(subj_sym["signed_offset_mm"].unique())

    fig, (ax_l, ax_r) = plt.subplots(
        1, 2,
        figsize=(FIG_SIZE[0] * 1.55, FIG_SIZE[1] * 0.95),
        facecolor="#FFFFFF",
        gridspec_kw={"wspace": 0.08},
    )
    fig.suptitle("Spatial Discrimination Accuracy by Distance",
                 fontsize=FONT_LABEL, fontweight="bold", y=1.01)

    _boxplot_panel(ax_l, 1.0,  signed_vals, show_ylabel=True)
    _boxplot_panel(ax_r, 26.0, signed_vals, show_ylabel=False)

    fig.tight_layout()
    return fig

def make_signed_boxplot_single(force):
    """Single-panel figure for one force condition – larger boxes, bolder outlines."""
    sns.set_theme(style="white")
    ATD.apply_plot_style()
    import matplotlib.patches as mpatches

    signed_vals = sorted(subj_sym["signed_offset_mm"].unique())
    bw      = 0.14
    gap_in  = 0.18
    rng     = np.random.default_rng(42)

    x_positions = {dist: di * gap_in for di, dist in enumerate(signed_vals)}

    fw = FIG_SIZE[0] * 0.62
    fh = FIG_SIZE[1] * 0.95
    fig, ax = plt.subplots(figsize=(fw, fh), facecolor="#FFFFFF")

    for dist in signed_vals:
        xp    = x_positions[dist]
        color = DIST_PALETTE.get(dist, "#888888")
        sub   = (subj_sym[
                     (subj_sym["force_g"] == force) &
                     (subj_sym["signed_offset_mm"] == dist)
                 ]["accuracy"].dropna().values * 100)
        if len(sub) == 0:
            continue

        ax.boxplot(
            sub, positions=[xp], widths=bw * 0.82,
            patch_artist=True, showfliers=False,
            medianprops=dict(color="#333333", lw=2.0),
            whiskerprops=dict(color="#555555", lw=1.4),
            capprops=dict(color="#555555", lw=1.4),
            boxprops=dict(facecolor=color, alpha=0.65,
                          edgecolor="#333333", lw=1.8),
        )
        jitter = rng.uniform(-bw * 0.30, bw * 0.30, len(sub))
        ax.scatter(xp + jitter, sub, color=color, s=28, alpha=0.80,
                   edgecolors="#444444", linewidths=0.4, zorder=5)

    ax.axhline(75, color="#CC0000", lw=1.4, linestyle="--", alpha=0.80, zorder=2)
    ax.axhline(50, color="#888888", lw=0.9, linestyle=":",  alpha=0.60, zorder=2)

    center = np.mean(list(x_positions.values()))
    ax.set_xticks([center])
    ax.set_xticklabels([f"{force:g} g"], fontsize=FONT_LABEL)
    ax.set_xlim(min(x_positions.values()) - bw, max(x_positions.values()) + bw)
    ax.set_ylim(-5, 108)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_yticklabels(["0%", "20%", "40%", "60%", "80%", "100%"], fontsize=FONT_TICK)
    ax.set_xlabel("Force Condition", fontsize=FONT_LABEL, labelpad=ATD.FIG_AXIS_LABELPAD)
    ax.set_ylabel("Accuracy (Correct Rate)", fontsize=FONT_LABEL, labelpad=ATD.FIG_AXIS_LABELPAD)
    ax.set_title(f"Spatial Discrimination Accuracy – {force:g} g",
                 fontsize=FONT_LABEL, fontweight="bold", pad=8)

    dist_handles = [
        mpatches.Patch(facecolor=DIST_PALETTE.get(d, "#888"),
                       edgecolor="#333333", linewidth=1.0,
                       label=f"{d:+.1f} mm")
        for d in signed_vals
    ]
    ref_handles = [
        plt.Line2D([0], [0], color="#CC0000", lw=1.4, linestyle="--", label="75% criterion"),
        plt.Line2D([0], [0], color="#888888", lw=0.9, linestyle=":", label="Chance (50%)"),
    ]
    leg = ax.legend(handles=dist_handles + ref_handles, title="Distance (mm)",
                    title_fontsize=FONT_ANNOT, fontsize=FONT_ANNOT,
                    frameon=True, framealpha=0.9, edgecolor="#DDDDDD",
                    loc="upper right", bbox_to_anchor=(1.0, 1.0), ncol=1)
    leg.get_title().set_fontweight("bold")
    _despine(ax)
    fig.tight_layout()
    return fig


print("[Figure 4] Signed-offset boxplot ...")
fig = make_signed_boxplot()
save_fig(fig, "sd_signed_boxplot")
plt.close(fig)

print("[Figure 4b] Signed-offset boxplot – 1 g ...")
fig = make_signed_boxplot_single(1.0)
save_fig(fig, "sd_signed_boxplot_1g")
plt.close(fig)

print("[Figure 4c] Signed-offset boxplot – 26 g ...")
fig = make_signed_boxplot_single(26.0)
save_fig(fig, "sd_signed_boxplot_26g")
plt.close(fig)

# ================================================================
# 7. Print summary table
# ================================================================
print("\n=== Group accuracy by force × offset ===")
pivot = grp_acc.pivot(index="abs_offset_mm",
                      columns="force_g",
                      values="mean_acc").round(3)
pivot.columns = [f"{c:g} g" for c in pivot.columns]
print(pivot.to_string())

print("\n=== JND summary ===")
jnd_summary = (jnd_df[jnd_df["fit_ok"]]
               .groupby("force_g")["jnd_mm"]
               .agg(median="median", mean="mean",
                    sd=lambda x: x.std(ddof=1), n="count")
               .round(2))
print(jnd_summary)

print(f"\n=== Done. Outputs → {OUTPUT_DIR} ===")
print("  sd_psychometric_curves  — psychometric functions (per-subj + group)")
print("  sd_jnd_paired           — per-subject JND: 1 g vs 26 g")
print("  sd_symmetry             — left / right bias check")
print("  sd_signed_boxplot       — accuracy × signed distance × force (boxplot)")
print("  jnd_per_subject.csv")
print("  gee_results.txt")