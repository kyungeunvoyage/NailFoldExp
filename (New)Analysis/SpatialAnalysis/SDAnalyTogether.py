"""
================================================================
Spatial Discrimination Analysis  –  JND Scatter & Combined Figure
================================================================
Standalone companion to SDAnalysis1.

Reads per-subject SD trial data, fits psychometric curves at each
force level (1 g, 26 g), extracts JND at the 75% criterion, and
renders two figures:

  1. sd_jnd_scatter            — per-subject JND scatter (1 g vs 26 g)
                                 with identity line, group mean±SE,
                                 and Spearman r annotation
  2. sd_jnd_combined           — two-panel figure combining the JND
                                 scatter (left) with the paired slope
                                 plot (right)

Style matches SDAnalysis1: ATD module fonts, colors, tick style,
inward ticks, save_fig_2col export at 2102 px width.

Outputs are written to SDAnalysis1_outputs so both scripts share
the same directory.
================================================================
"""

import os
import glob
import re
import importlib.util
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"]     = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from scipy import stats

warnings.filterwarnings("ignore")

# ================================================================
# 0. ATD style loader (identical to SDAnalysis1)
# ================================================================
_SCRIPT_DIR = Path(__file__).resolve().parent


def _load_atd():
    root = _SCRIPT_DIR.parent / "ATDAnalysis"
    for sub in ("Stat files", "Stat files (final) "):
        path = root / sub / "(Final)ATD_C1_Fig(Anika).py"
        if path.is_file():
            spec = importlib.util.spec_from_file_location("atd_c1_fig", path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    raise FileNotFoundError(
        f"Could not find (Final)ATD_C1_Fig(Anika).py under {root}"
    )


ATD = _load_atd()

FONT_TICK = ATD.FONT_TICK
FONT_LABEL = ATD.FONT_LABEL
FONT_ANNOT = ATD.FONT_ANNOT
FIG_SIZE = ATD.FIG_SIZE
SAVE_DPI = ATD.SAVE_DPI
EXPORT_WIDTHS_PX = ATD.EXPORT_WIDTHS_PX

COLOR_LOW = "#85B1D9"   # 1 g
COLOR_HIGH = "#3D5F9A"  # 26 g
FORCE_COLORS = {1.0: COLOR_LOW, 26.0: COLOR_HIGH}
SIG_COLOR = ATD.ACCENT_RED

GRID_SPACING_MM = 1.5
THRESHOLD = 0.75  # 75% criterion for JND

PSYCH_EXPORT_HEIGHT_PX = 1200
PSYCH_EXPORT_WIDTH_PX = 2102
PSYCH_WSPACE = 0.24
PSYCH_MARGIN_LEFT = 0.10
PSYCH_MARGIN_RIGHT = 0.98
PSYCH_MARGIN_TOP = 0.90
PSYCH_R_PANEL_SHIFT = 0.012

# ================================================================
# 1. Paths
# ================================================================
REPO_ROOT = "/Users/kyungeunjung/NailFoldExp"
SD_PATTERN = os.path.join(
    REPO_ROOT, "Data", "(SD)CurData", "P*_SpatialDiscrimination.csv"
)
OUTPUT_DIR = os.path.join(
    REPO_ROOT, "(New)Analysis", "SpatialAnalysis", "SDAnalysis1_outputs"
)
os.makedirs(OUTPUT_DIR, exist_ok=True)



# ================================================================
# 2. Load & parse
# ================================================================
def _parse_grid(s):
    m = re.match(r"g(-?\d+)", str(s).strip())
    return float(m.group(1)) if m else np.nan


def _parse_force(s):
    m = re.match(r"([\d.]+)", str(s).strip())
    return float(m.group(1)) if m else np.nan


def load_sd_data():
    sd_files = sorted(glob.glob(SD_PATTERN))
    if not sd_files:
        raise FileNotFoundError(f"No SD files found:\n  {SD_PATTERN}")
    print(f"[Load] {len(sd_files)} participant file(s) found.")

    df = pd.concat(
        [pd.read_csv(f, encoding="utf-8-sig") for f in sd_files],
        ignore_index=True,
    )
    df["pos_1st_mm"] = df["Stim_1st"].apply(_parse_grid) * GRID_SPACING_MM
    df["pos_2nd_mm"] = df["Stim_2nd"].apply(_parse_grid) * GRID_SPACING_MM
    df["signed_offset_mm"] = df["pos_2nd_mm"] - df["pos_1st_mm"]
    df["abs_offset_mm"] = df["signed_offset_mm"].abs()
    df["force_g"] = df["Force"].apply(_parse_force)
    df["IsCorrect"] = pd.to_numeric(df["IsCorrect"], errors="coerce")
    df = df.dropna(subset=["IsCorrect", "signed_offset_mm", "force_g"])

    print(f"       Subjects: {df['Subject'].nunique()}")
    print(f"       Forces  : {sorted(df['force_g'].unique())} g")
    print(f"       Rows    : {len(df)}")
    return df


# ================================================================
# 3. Psychometric fitting
# ================================================================
def _psychometric(x, x50, beta, lapse=0.02):
    return 0.5 + (0.48 - lapse) / (1.0 + np.exp(-beta * (x - x50)))


def _fit_curve(xs, ys, p0=(3.0, 1.0)):
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
    x_arr = np.linspace(0, 20, 20000)
    y_arr = _psychometric(x_arr, *popt)
    idx = np.argmin(np.abs(y_arr - target))
    return float(x_arr[idx])


def fit_per_subject_jnd(df):
    """Return a DataFrame of per-subject JND fits at each force level."""
    subj_acc = (
        df.groupby(["Subject", "force_g", "abs_offset_mm"])
        .agg(accuracy=("IsCorrect", "mean"),
             n_trials=("IsCorrect", "count"))
        .reset_index()
    )

    records = []
    for (subj, force), grp in subj_acc.groupby(["Subject", "force_g"]):
        grp_s = grp.sort_values("abs_offset_mm")
        xs, ys = grp_s["abs_offset_mm"].values, grp_s["accuracy"].values
        if len(xs) < 3:
            continue
        popt, ok = _fit_curve(xs, ys)
        jnd = _jnd_from_fit(popt) if ok else np.nan
        records.append({
            "Subject": subj, "force_g": force,
            "jnd_mm": jnd, "x50": popt[0], "beta": popt[1], "fit_ok": ok,
        })
    return pd.DataFrame(records)


def paired_jnd_wilcoxon(jnd_df):
    """Return Wilcoxon signed-rank p-value comparing 1 g vs 26 g JND."""
    wide = (
        jnd_df[jnd_df["fit_ok"]]
        .pivot(index="Subject", columns="force_g", values="jnd_mm")
        .dropna()
    )
    if 1.0 in wide.columns and 26.0 in wide.columns:
        stat, p = stats.wilcoxon(wide[1.0], wide[26.0])
        return stat, p, len(wide)
    return np.nan, np.nan, 0


# ================================================================
# 4. Layout helpers (matched to SDAnalysis1)
# ================================================================
def _draw_inward_ticks(ax, frac=None, color="#333333", lw=1.2):
    from matplotlib.transforms import blended_transform_factory
    if frac is None:
        frac = ATD.TICK_LEN_AXES
    x_trans = blended_transform_factory(ax.transData, ax.transAxes)
    y_trans = blended_transform_factory(ax.transAxes, ax.transData)
    xlo, xhi = ax.get_xlim()
    ylo, yhi = ax.get_ylim()
    kw = dict(color=color, lw=lw, clip_on=False, zorder=10,
              solid_capstyle="butt")
    for yt in ax.get_yticks():
        if ylo <= yt <= yhi:
            ax.plot([0, frac], [yt, yt], transform=y_trans, **kw)
    for xt in ax.get_xticks():
        if xlo <= xt <= xhi:
            ax.plot([xt, xt], [0, frac], transform=x_trans, **kw)


def _despine(ax):
    sns.despine(ax=ax)
    ax.tick_params(length=0, labelsize=FONT_TICK)
    ax.grid(False)


def _two_panel(fig_h=None):
    sns.set_theme(style="white")
    ATD.apply_plot_style()
    fw = FIG_SIZE[0]
    fh = fig_h or fw * (PSYCH_EXPORT_HEIGHT_PX / PSYCH_EXPORT_WIDTH_PX)
    fig, (ax_l, ax_r) = plt.subplots(
        1, 2,
        figsize=(fw, fh),
        sharex=False,
        sharey=False,
        facecolor="#FFFFFF",
        gridspec_kw={"width_ratios": [1, 1]},
    )
    fig.subplots_adjust(
        left=PSYCH_MARGIN_LEFT,
        right=PSYCH_MARGIN_RIGHT,
        bottom=ATD.FIG_LEGEND_BOTTOM,
        top=PSYCH_MARGIN_TOP,
        wspace=PSYCH_WSPACE,
    )
    return fig, ax_l, ax_r


def _finalize_panels(fig, ax_l, ax_r, titles=None):
    pos_l, pos_r = ax_l.get_position(), ax_r.get_position()
    w, h, y0 = pos_l.width, pos_l.height, pos_l.y0
    gap = pos_r.x0 - pos_l.x1
    ax_l.set_position([pos_l.x0, y0, w, h])
    ax_r.set_position([pos_l.x0 + w + gap - PSYCH_R_PANEL_SHIFT, y0, w, h])

    ax_l.set_title("")
    ax_r.set_title("")

    if titles is not None:
        for ax, title in zip((ax_l, ax_r), titles):
            p = ax.get_position()
            fig.text(
                p.x0 + p.width / 2, p.y1 + 0.015,
                title,
                ha="center", va="bottom",
                fontsize=FONT_LABEL, fontweight="bold",
                color="#333333",
                transform=fig.transFigure,
            )


# ================================================================
# 5. Save helpers
# ================================================================
def save_fig_2col(fig, stem, target_h=None):
    import io
    from PIL import Image
    w_in, _ = fig.get_size_inches()
    w_px = 2102
    dpi = w_px / w_in
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                pad_inches=0.04, facecolor="white")
    buf.seek(0)
    master = Image.open(buf).convert("RGB")
    h = target_h if target_h is not None else round(
        w_px * master.height / master.width
    )
    master = master.resize((w_px, h), Image.Resampling.LANCZOS)
    out_2col = os.path.join(OUTPUT_DIR, f"{stem}_2col.png")
    legacy = os.path.join(OUTPUT_DIR, f"{stem}.png")
    master.save(out_2col)
    master.save(legacy)
    print(f"  → {legacy}  ({w_px}×{h} px)")


# ================================================================
# 6. FIGURE — JND scatter (standalone panel)
# ================================================================
def _draw_jnd_scatter(ax, jnd_df, wilcoxon_p=None):
    """Render the JND scatter (1 g vs 26 g) into `ax`. Return v_max and stats."""
    jnd_ok = jnd_df[jnd_df["fit_ok"]].copy()
    wide = (
        jnd_ok.pivot(index="Subject", columns="force_g", values="jnd_mm")
        .dropna(subset=[1.0, 26.0], how="any")
    )
    if len(wide) < 3:
        return None

    x_vals = wide[1.0].values
    y_vals = wide[26.0].values

    r_sp, p_sp = stats.spearmanr(x_vals, y_vals)
    mean_x, mean_y = x_vals.mean(), y_vals.mean()
    se_x = x_vals.std(ddof=1) / np.sqrt(len(x_vals))
    se_y = y_vals.std(ddof=1) / np.sqrt(len(y_vals))

    v_max = float(np.ceil(np.concatenate([x_vals, y_vals]).max() + 0.5))
    ax.set_xlim(0, v_max)
    ax.set_ylim(0, v_max)

    ax.plot([0, v_max], [0, v_max],
            color="#888888", lw=1.0, linestyle="--", alpha=0.75, zorder=1)

    scatter_rgba = ATD._hsb_scatter_rgba(ATD.ON_TOUCH)
    ax.scatter(
        x_vals, y_vals,
        c=[scatter_rgba] * len(x_vals),
        s=50, edgecolors="#333333", linewidths=0.6,
        alpha=0.85, zorder=4,
    )
    ax.errorbar(
        mean_x, mean_y,
        xerr=se_x, yerr=se_y,
        fmt="D", color=SIG_COLOR,
        markersize=10, markeredgecolor="white", markeredgewidth=0.8,
        capsize=4, capthick=1.4, elinewidth=1.6,
        zorder=6, label="Group mean ± SE",
    )

    p_txt = f"p = {p_sp:.2f}" if p_sp >= 0.001 else "p < 0.001"
    annot_lines = [f"Spearman r = {r_sp:.2f}", p_txt, f"n = {len(wide)}"]
    if wilcoxon_p is not None and not np.isnan(wilcoxon_p):
        w_txt = f"p = {wilcoxon_p:.2f}" if wilcoxon_p >= 0.001 else "p < 0.001"
        annot_lines += ["", f"Wilcoxon (1 g vs 26 g)", w_txt]
    ax.text(
        0.04, 0.96,
        "\n".join(annot_lines),
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=FONT_ANNOT, color="#333333",
        bbox=dict(facecolor="white", edgecolor="#DDDDDD",
                  boxstyle="round,pad=0.4", alpha=0.85),
        zorder=7,
    )

    tick_step = 1.0 if v_max <= 6 else 2.0
    ticks = np.arange(0, v_max + 0.01, tick_step)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks], fontsize=FONT_TICK)
    ax.set_yticklabels([f"{t:g}" for t in ticks], fontsize=FONT_TICK)
    ax.set_xlabel("JND at 1 g (mm)", fontsize=FONT_LABEL,
                  labelpad=ATD.FIG_AXIS_LABELPAD)
    ax.set_ylabel("JND at 26 g (mm)", fontsize=FONT_LABEL,
                  labelpad=ATD.FIG_AXIS_LABELPAD)
    ax.set_aspect("equal", adjustable="box")

    _despine(ax)
    _draw_inward_ticks(ax)

    return {
        "v_max": v_max,
        "ticks": ticks,
        "r_spearman": r_sp,
        "p_spearman": p_sp,
        "n": len(wide),
    }


def make_jnd_scatter(jnd_df):
    """Single-panel JND scatter figure."""
    sns.set_theme(style="white")
    ATD.apply_plot_style()
    fw = FIG_SIZE[0] * 0.55
    fh = FIG_SIZE[1] * 1.10
    fig, ax = plt.subplots(figsize=(fw, fh), facecolor="#FFFFFF")
    result = _draw_jnd_scatter(ax, jnd_df)
    if result is None:
        plt.close(fig)
        return None
    ax.legend(
        fontsize=FONT_ANNOT, frameon=False,
        loc="lower right", bbox_to_anchor=(0.98, 0.02),
        handlelength=1.4,
    )
    fig.tight_layout()
    return fig


# ================================================================
# 7. FIGURE — JND scatter + paired slope (combined)
# ================================================================
def _draw_jnd_paired_slope(ax, jnd_df, wilcoxon_p, y_max_override=None,
                          ticks_override=None):
    """Render the paired slope plot into `ax`."""
    jnd_ok = jnd_df[jnd_df["fit_ok"]].copy()
    forces_sorted = [1.0, 26.0]
    x_pos = {1.0: 0, 26.0: 1}
    rng = np.random.default_rng(7)

    for subj in jnd_ok["Subject"].unique():
        sub = jnd_ok[jnd_ok["Subject"] == subj].sort_values("force_g")
        if len(sub) < 2:
            continue
        xs = [x_pos[f] + (rng.random() - 0.5) * 0.06 for f in sub["force_g"]]
        ys = sub["jnd_mm"].values
        ax.plot(xs, ys, color="#888888", alpha=0.35, lw=0.9, zorder=2)
        for xi, yi, f in zip(xs, ys, sub["force_g"]):
            ax.scatter(xi, yi, color=FORCE_COLORS[f], s=32,
                       edgecolors="white", linewidths=0.4,
                       alpha=0.80, zorder=4)

    for force in forces_sorted:
        sub = jnd_ok[jnd_ok["force_g"] == force]["jnd_mm"].dropna()
        m, s = sub.mean(), sub.std(ddof=1) / np.sqrt(len(sub))
        xi = x_pos[force]
        ax.errorbar(
            xi, m, yerr=s,
            fmt="D", color=FORCE_COLORS[force],
            markersize=10, markeredgecolor="white",
            markeredgewidth=0.6,
            capsize=4, capthick=1.4, elinewidth=1.6,
            zorder=6,
        )

    if not np.isnan(wilcoxon_p):
        p_txt = f"p = {wilcoxon_p:.2f}" if wilcoxon_p >= 0.001 else "p < 0.001"
        y_bar = jnd_ok["jnd_mm"].max() + 0.5
        ax.plot([0, 1], [y_bar, y_bar], color="#333333", lw=1.0)
        ax.text(0.5, y_bar + 0.15, p_txt,
                ha="center", va="bottom",
                fontsize=FONT_ANNOT, color="#333333")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["1 g", "26 g"], fontsize=FONT_LABEL)
    ax.set_xlim(-0.45, 1.45)
    ax.set_ylabel("JND (mm)", fontsize=FONT_LABEL,
                  labelpad=ATD.FIG_AXIS_LABELPAD)

    if y_max_override is not None:
        ax.set_ylim(0, y_max_override)
    if ticks_override is not None:
        ax.set_yticks(ticks_override)
        ax.set_yticklabels([f"{t:g}" for t in ticks_override],
                           fontsize=FONT_TICK)

    _despine(ax)
    _draw_inward_ticks(ax)


def make_jnd_combined(jnd_df, wilcoxon_p):
    """Single-panel figure: Within-subject JND scatter only."""
    sns.set_theme(style="white")
    ATD.apply_plot_style()
    fw = FIG_SIZE[0] * 0.60
    fh = FIG_SIZE[0] * 0.60 * 1.10
    fig, ax = plt.subplots(figsize=(fw, fh), facecolor="#FFFFFF")
    result = _draw_jnd_scatter(ax, jnd_df, wilcoxon_p=wilcoxon_p)
    if result is None:
        plt.close(fig)
        return None
    ax.legend(
        fontsize=FONT_ANNOT, frameon=False,
        loc="lower right", bbox_to_anchor=(0.98, 0.02),
        handlelength=1.4,
    )
    # title
    p = ax.get_position()
    fig.text(
        p.x0 + p.width / 2, p.y1 + 0.015,
        "Within-subject JND",
        ha="center", va="bottom",
        fontsize=FONT_LABEL, fontweight="bold",
        color="#333333",
        transform=fig.transFigure,
    )
    fig.tight_layout()
    return fig


# ================================================================
# 8. Runner
# ================================================================
def run_cohort(df_cohort, tag=""):
    print(f"\n[Cohort {tag or 'full'}] n={df_cohort['Subject'].nunique()}  "
          f"rows={len(df_cohort)}")
    jnd_df = fit_per_subject_jnd(df_cohort)
    jnd_path = os.path.join(OUTPUT_DIR, f"jnd_per_subject_scatter{tag}.csv")
    jnd_df.to_csv(jnd_path, index=False)
    print(f"  Saved {os.path.basename(jnd_path)}")

    stat, p, n_paired = paired_jnd_wilcoxon(jnd_df)
    if n_paired > 0:
        print(f"  Wilcoxon JND(1g vs 26g): W={stat:.2f}, p={p:.4f}, n={n_paired}")
    else:
        print("  Wilcoxon: insufficient paired data")

    print("  [Fig] JND scatter ...")
    fig = make_jnd_scatter(jnd_df)
    if fig is not None:
        save_fig_2col(fig, f"sd_jnd_scatter{tag}", target_h=1200)
        plt.close(fig)

    print("  [Fig] JND scatter + paired combined ...")
    fig = make_jnd_combined(jnd_df, p)
    if fig is not None:
        save_fig_2col(fig, f"sd_jnd_combined{tag}", target_h=1200)
        plt.close(fig)


def main():
    df_all = load_sd_data()
    run_cohort(df_all, tag="")
    print(f"\n=== Done. Outputs → {OUTPUT_DIR} ===")


if __name__ == "__main__":
    main()