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
def _load_atd():
    root = _SCRIPT_DIR.parent / "ATDAnalysis"
    for sub in ("Stat files", "Stat files (final) "):
        path = root / sub / "(Final)ATD_C1_Fig(Anika).py"
        if path.is_file():
            spec = importlib.util.spec_from_file_location("atd_c1_fig", path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    raise FileNotFoundError(f"Could not find (Final)ATD_C1_Fig(Anika).py under {root}")

ATD = _load_atd()

FONT_TICK        = ATD.FONT_TICK
FONT_LABEL       = ATD.FONT_LABEL
FONT_ANNOT       = ATD.FONT_ANNOT
FIG_SIZE         = ATD.FIG_SIZE
SAVE_DPI         = ATD.SAVE_DPI
EXPORT_WIDTHS_PX = ATD.EXPORT_WIDTHS_PX
GAP_IN           = 1.5

COLOR_LOW  = "#85B1D9"   # 1 g  → light blue (mm boxplot 3 mm ramp)
COLOR_HIGH = "#3D5F9A"   # 26 g → dark blue (mm boxplot 4.5 mm ramp)
FORCE_COLORS = {1.0: COLOR_LOW, 26.0: COLOR_HIGH}

GRID_SPACING_MM  = 1.5
THRESHOLD        = 0.75   # 75 % criterion for JND
PSYCH_EXPORT_HEIGHT_PX = 1200   # 2-col psychometric figure target height
PSYCH_EXPORT_WIDTH_PX  = 2102
PSYCH_WSPACE           = 0.24   # gap between panels (fraction of axis width)
PSYCH_MARGIN_LEFT      = 0.10   # room for shared y-axis label
PSYCH_MARGIN_RIGHT     = 0.98
PSYCH_MARGIN_TOP       = 0.90
PSYCH_R_PANEL_SHIFT    = 0.012  # nudge 26g panel left (figure fraction)

# ================================================================
# 1. Paths
# ================================================================
REPO_ROOT  = "/Users/kyungeunjung/NailFoldExp"
SD_PATTERN = os.path.join(REPO_ROOT, "Data", "(SD)CurData",
                           "P*_SpatialDiscrimination.csv")
OUTPUT_DIR = os.path.join(REPO_ROOT, "(New)Analysis",
                           "SpatialAnalysis", "SDAnalysis1_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# SD cohort gender (CurData n=25: 13 M, 12 F)
SD_GENDER = {
    "P24": "M", "P26": "F", "P27": "F", "P28": "F", "P29": "M",
    "P32": "M", "P34": "F", "P36": "M", "P37": "M", "P40": "F",
    "P41": "M", "P44": "F", "P45": "M", "P47": "M", "P48": "M",
    "P49": "M", "P50": "M", "P51": "F", "P52": "F", "P53": "F",
    "P54": "F", "P55": "F", "P56": "F", "P59": "M", "P60": "M",
}
SUBSET_N_M    = 8
SUBSET_N_F    = 7
SUBSET_SEED   = 42
SUBSET_STEM   = "_n15_m8f7"   # filename suffix for balanced subset figures


def select_gender_balanced_subjects(gender_map, n_m, n_f, seed=42):
    """Randomly sample n_m males + n_f females from the SD cohort."""
    rng = np.random.default_rng(seed)
    males   = sorted(s for s, g in gender_map.items() if g == "M")
    females = sorted(s for s, g in gender_map.items() if g == "F")
    if len(males) < n_m or len(females) < n_f:
        raise ValueError(
            f"Not enough subjects for {n_m}M + {n_f}F "
            f"(have {len(males)}M, {len(females)}F)"
        )
    pick_m = list(rng.choice(males, n_m, replace=False))
    pick_f = list(rng.choice(females, n_f, replace=False))
    return sorted(pick_m + pick_f)

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

df_all = pd.concat(
    [pd.read_csv(f, encoding="utf-8-sig") for f in sd_files],
    ignore_index=True,
)

# Grid → mm
df_all["pos_1st_mm"] = df_all["Stim_1st"].apply(_parse_grid) * GRID_SPACING_MM
df_all["pos_2nd_mm"] = df_all["Stim_2nd"].apply(_parse_grid) * GRID_SPACING_MM

# Signed offset: positive = 2nd is to the RIGHT of 1st
df_all["signed_offset_mm"] = df_all["pos_2nd_mm"] - df_all["pos_1st_mm"]
df_all["abs_offset_mm"]    = df_all["signed_offset_mm"].abs()

df_all["force_g"]   = df_all["Force"].apply(_parse_force)
df_all["IsCorrect"] = pd.to_numeric(df_all["IsCorrect"], errors="coerce")
df_all = df_all.dropna(subset=["IsCorrect", "signed_offset_mm", "force_g"])

print(f"       Subjects (all): {df_all['Subject'].nunique()}")
print(f"       Forces         : {sorted(df_all['force_g'].unique())} g")
print(f"       Rows (all)     : {len(df_all)}")

# Cohort globals — populated by recompute_cohort()
df = df_all.copy()
n_subj = forces = offsets = None
subj_acc = grp_acc = subj_sym = grp_sym = None
group_fits = jnd_df = p_val = None
df_cvsd = subj_cvsd = None
_region_mode = ""
_x_smooth = _xticks = _xlabels = None

# ================================================================
# 3–4. Cohort summaries & psychometric fits
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

REGION_C, REGION_D = "C", "D"
REGION_COLORS = {REGION_C: "#10559A", REGION_D: "#85B1D9"}
SIG_COLOR = ATD.ACCENT_RED


def _assign_sd_region(frame):
    """Return per-trial region label (C or D)."""
    if "Area" in frame.columns:
        return frame["Area"].astype(str).str.strip().str.upper()
    if "Region" in frame.columns:
        return frame["Region"].astype(str).str.strip().str.upper()
    return pd.Series(
        np.where(frame["signed_offset_mm"] < 0, REGION_C, REGION_D),
        index=frame.index,
    )


def _star_from_p(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def _lme_cvsd(frame, force_val, ref=REGION_C, target=REGION_D):
    """Trial-level C vs D contrast at one force level."""
    import statsmodels.formula.api as smf

    sub = frame[
        (frame["force_g"] == force_val) & frame["Area"].isin([ref, target])
    ].copy()
    if len(sub) < 10 or sub["Area"].nunique() < 2:
        return None
    formula = f"IsCorrect ~ C(Area, Treatment(reference='{ref}'))"
    try:
        res = smf.mixedlm(formula, sub, groups=sub["Subject"]).fit()
        col = f"C(Area, Treatment(reference='{ref}'))[T.{target}]"
        if col not in res.params.index:
            return None
        ci = res.conf_int().loc[col]
        return {
            "coef": float(res.params[col]),
            "ci_lo": float(ci[0]),
            "ci_hi": float(ci[1]),
            "p": float(res.pvalues[col]),
        }
    except Exception:
        return None


def _write_cvsd_stats(aux_tag=""):
    """Write C vs D summary stats for the current cohort globals."""
    cvsd_stats_lines = [
        "Spatial Discrimination — Region C vs D",
        f"Classification: {_region_mode}",
        "Metric: per-subject mean accuracy (all offset magnitudes pooled)",
        "",
        "=== Group summary (subject means, %) ===",
        f"{'Force':>6}  {'Region':>6}  {'Mean':>7}  {'SD':>7}  {'SEM':>7}  {'Median':>7}  {'N':>4}",
        "-" * 56,
    ]
    for force in forces:
        for region in (REGION_C, REGION_D):
            vals = subj_cvsd.loc[
                (subj_cvsd["force_g"] == force) & (subj_cvsd["Area"] == region),
                "accuracy_pct",
            ]
            if len(vals) == 0:
                continue
            cvsd_stats_lines.append(
                f"{force:5.1f}g  {region:>6}  {vals.mean():7.2f}  "
                f"{vals.std(ddof=1):7.2f}  {vals.sem():7.2f}  "
                f"{vals.median():7.2f}  {len(vals):4d}"
            )
        wide = (
            subj_cvsd[subj_cvsd["force_g"] == force]
            .pivot(index="Subject", columns="Area", values="accuracy_pct")
            .dropna(subset=[REGION_C, REGION_D], how="any")
        )
        if len(wide) >= 5:
            w_stat, p_w = stats.wilcoxon(wide[REGION_C], wide[REGION_D])
            cvsd_stats_lines.append(
                f"  Wilcoxon {force:g}g (C vs D): W={w_stat:.1f}, p={p_w:.4f}, "
                f"n={len(wide)}  ({_star_from_p(p_w)})"
            )
        else:
            cvsd_stats_lines.append(f"  Wilcoxon {force:g}g: insufficient paired data")
        lme = _lme_cvsd(df_cvsd, force)
        if lme:
            cvsd_stats_lines.append(
                f"  LME {force:g}g (D − C): Δ={lme['coef']:.3f} "
                f"[{lme['ci_lo']:.3f}, {lme['ci_hi']:.3f}], "
                f"p={lme['p']:.4f}  ({_star_from_p(lme['p'])})"
            )
        else:
            cvsd_stats_lines.append(f"  LME {force:g}g: failed")

    cvsd_stats_path = os.path.join(OUTPUT_DIR, f"sd_cvsd_stats{aux_tag}.txt")
    with open(cvsd_stats_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(cvsd_stats_lines) + "\n")
    print(f"  Saved {os.path.basename(cvsd_stats_path)}")


def _add_sig_bracket(ax, x_l, x_r, y_base, text="", tick_h=1.8):
    x_center = (x_l + x_r) / 2.0
    y_top = y_base + tick_h
    ax.plot(
        [x_l, x_l, x_r, x_r],
        [y_base, y_top, y_top, y_base],
        color=SIG_COLOR,
        linewidth=0.9,
        clip_on=False,
        zorder=6,
    )
    ax.text(
        x_center,
        y_top + 0.4,
        text,
        ha="center",
        va="bottom",
        fontsize=FONT_ANNOT,
        color=SIG_COLOR,
    )


def recompute_cohort(df_in, cohort_label="cohort", aux_tag="", run_gee=False):
    """Recompute all summary tables / fits for a subject subset."""
    global df, n_subj, forces, offsets
    global subj_acc, grp_acc, subj_sym, grp_sym
    global group_fits, jnd_df, p_val
    global df_cvsd, subj_cvsd, _region_mode
    global _x_smooth, _xticks, _xlabels

    df = df_in.copy()
    n_subj  = df["Subject"].nunique()
    forces  = sorted(df["force_g"].unique())
    offsets = sorted(df["abs_offset_mm"].unique())

    print(f"\n[{cohort_label}] n={n_subj}  forces={forces} g  rows={len(df)}")

    subj_acc = (
        df.groupby(["Subject", "force_g", "abs_offset_mm"])
        .agg(accuracy=("IsCorrect", "mean"),
             n_trials=("IsCorrect", "count"))
        .reset_index()
    )
    grp_acc = (
        subj_acc.groupby(["force_g", "abs_offset_mm"])
        .agg(mean_acc=("accuracy", "mean"),
             se_acc  =("accuracy", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
             n_subj  =("accuracy", "count"))
        .reset_index()
    )
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

    print("[Psychometric] Group-level fits:")
    group_fits = {}
    for force in forces:
        sub  = grp_acc[grp_acc["force_g"] == force].sort_values("abs_offset_mm")
        xs, ys = sub["abs_offset_mm"].values, sub["mean_acc"].values
        popt, ok = _fit_curve(xs, ys)
        jnd  = _jnd_from_fit(popt) if ok else np.nan
        group_fits[force] = {"popt": popt, "ok": ok, "jnd": jnd, "data": sub}
        print(f"  {force:4.1f} g → JND = {jnd:.2f} mm  (x50={popt[0]:.2f}, β={popt[1]:.2f})")

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
    jnd_path = os.path.join(OUTPUT_DIR, f"jnd_per_subject{aux_tag}.csv")
    jnd_df.to_csv(jnd_path, index=False)
    print(f"  Saved {os.path.basename(jnd_path)}  (n = {len(jnd_df)} fits)")

    wide = (jnd_df[jnd_df["fit_ok"]]
            .pivot(index="Subject", columns="force_g", values="jnd_mm")
            .dropna())
    if 1.0 in wide.columns and 26.0 in wide.columns:
        stat, p_val = stats.wilcoxon(wide[1.0], wide[26.0])
        print(f"  Wilcoxon JND(1g vs 26g): W={stat:.2f}, p={p_val:.4f}, n={len(wide)}")
    else:
        p_val = np.nan

    if run_gee:
        print("[GEE] IsCorrect ~ abs_offset_mm × force_g  (Binomial, Exchangeable)")
        try:
            from statsmodels.genmod.generalized_estimating_equations import GEE
            from statsmodels.genmod.cov_struct import Exchangeable
            from statsmodels.genmod.families import Binomial

            df_gee = df.copy()
            gee_model = GEE.from_formula(
                "IsCorrect ~ abs_offset_mm * force_g",
                groups="Subject",
                data=df_gee,
                cov_struct=Exchangeable(),
                family=Binomial(),
            )
            gee_result = gee_model.fit()
            print(gee_result.summary().tables[1])
            gee_path = os.path.join(OUTPUT_DIR, f"gee_results{aux_tag}.txt")
            with open(gee_path, "w") as fh:
                fh.write(str(gee_result.summary()))
            print(f"  Saved {os.path.basename(gee_path)}")
        except Exception as exc:
            print(f"  GEE failed: {exc}")

    _x_smooth = np.linspace(0, offsets[-1] + 1.0, 300)
    _xticks   = offsets
    _xlabels  = [f"{x:g}" for x in _xticks]

    df_cvsd = df[df["signed_offset_mm"] != 0].copy()
    df_cvsd["Area"] = _assign_sd_region(df_cvsd)
    df_cvsd = df_cvsd[df_cvsd["Area"].isin([REGION_C, REGION_D])].copy()
    _region_mode = (
        "Area/Region column"
        if ("Area" in df_in.columns or "Region" in df_in.columns)
        else "directional proxy (signed offset → C/D at C–D zone)"
    )
    subj_cvsd = (
        df_cvsd.groupby(["Subject", "force_g", "Area"], as_index=False)
        .agg(accuracy=("IsCorrect", "mean"), n_trials=("IsCorrect", "count"))
    )
    subj_cvsd["accuracy_pct"] = subj_cvsd["accuracy"] * 100.0
    cvsd_csv = os.path.join(OUTPUT_DIR, f"sd_cvsd_per_subject{aux_tag}.csv")
    subj_cvsd.to_csv(cvsd_csv, index=False)
    _write_cvsd_stats(aux_tag)

# ================================================================
# 5. Save helper (ATD style)
# ================================================================
def save_fig(fig, stem, height_2col_px=None, bbox_inches="tight"):
    import io
    from PIL import Image
    buf = io.BytesIO()
    save_kw = dict(format="png", dpi=SAVE_DPI, facecolor="white")
    if bbox_inches is not None:
        save_kw.update(bbox_inches=bbox_inches, pad_inches=0.05)
    else:
        save_kw.update(bbox_inches=None, pad_inches=0)
    fig.savefig(buf, **save_kw)
    buf.seek(0)
    master = Image.open(buf).convert("RGB")
    for tag, w in EXPORT_WIDTHS_PX:
        if tag == "2col" and height_2col_px:
            h = height_2col_px
        else:
            h = round(w * master.height / master.width)
        master.resize((w, h), Image.Resampling.LANCZOS).save(
            os.path.join(OUTPUT_DIR, f"{stem}_{tag}.png"))
    legacy = os.path.join(OUTPUT_DIR, f"{stem}.png")
    legacy_h = height_2col_px or round(2102 * master.height / master.width)
    master.resize(
        (2102, legacy_h),
        Image.Resampling.LANCZOS,
    ).save(legacy)
    print(f"  → {legacy}  (2102×{legacy_h} px)")


def save_fig_psych(fig, stem):
    """Export psychometric figure at exact 2-col px with no aspect distortion."""
    import io
    from PIL import Image
    w_in, h_in = fig.get_size_inches()
    dpi = PSYCH_EXPORT_WIDTH_PX / w_in
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches=None,
                pad_inches=0, facecolor="white")
    buf.seek(0)
    master = Image.open(buf).convert("RGB")
    aspect = PSYCH_EXPORT_HEIGHT_PX / PSYCH_EXPORT_WIDTH_PX
    for tag, w in EXPORT_WIDTHS_PX:
        h = round(w * aspect)
        out = master.resize((w, h), Image.Resampling.LANCZOS)
        out.save(os.path.join(OUTPUT_DIR, f"{stem}_{tag}.png"))
    legacy = os.path.join(OUTPUT_DIR, f"{stem}.png")
    master.save(legacy)
    print(f"  → {legacy}  ({master.width}×{master.height} px)")


def save_fig_2col(fig, stem):
    """2-col export matching render_final_figures / Fig2 (2102 px wide)."""
    import io
    from PIL import Image
    w_in, _ = fig.get_size_inches()
    w_px = 2102
    dpi  = w_px / w_in
    buf  = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                pad_inches=0.04, facecolor="white")
    buf.seek(0)
    master = Image.open(buf).convert("RGB")
    h = round(w_px * master.height / master.width)
    master = master.resize((w_px, h), Image.Resampling.LANCZOS)
    out_2col = os.path.join(OUTPUT_DIR, f"{stem}_2col.png")
    legacy   = os.path.join(OUTPUT_DIR, f"{stem}.png")
    master.save(out_2col)
    master.save(legacy)
    print(f"  → {legacy}  ({w_px}×{h} px)")

# ================================================================
# 6. Layout helpers
# ================================================================
def _draw_inward_ticks(ax, frac=None, color="#333333", lw=1.2):
    """Draw inward tick marks (same length on x and y, matching ATD Fig2)."""
    from matplotlib.transforms import blended_transform_factory
    if frac is None:
        frac = ATD.TICK_LEN_AXES
    x_trans = blended_transform_factory(ax.transData, ax.transAxes)
    y_trans = blended_transform_factory(ax.transAxes, ax.transData)
    xlo, xhi = ax.get_xlim()
    ylo, yhi = ax.get_ylim()
    kw = dict(color=color, lw=lw, clip_on=False, zorder=10, solid_capstyle="butt")

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
    """Equal-width 1×2 panels; figure aspect matches 2-col export (2102×1200)."""
    sns.set_theme(style="white")
    ATD.apply_plot_style()
    fw = FIG_SIZE[0]
    fh = fig_h or fw * (PSYCH_EXPORT_HEIGHT_PX / PSYCH_EXPORT_WIDTH_PX)
    fig, (ax_l, ax_r) = plt.subplots(
        1, 2,
        figsize=(fw, fh),
        sharex=True,
        sharey=True,
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

# ================================================================
# FIGURE 1: Psychometric curves
# ================================================================
def _psych_panel(ax, force):
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

    # Reference lines (no legend entries)
    ax.axhline(THRESHOLD, color="#555555", lw=1.0,
               linestyle=":", alpha=0.7, zorder=1)
    ax.axhline(0.5, color="#AAAAAA", lw=0.8,
               linestyle=":", alpha=0.6, zorder=1)

    # Axes
    ax.set_xlim(0.5, offsets[-1] + 0.8)
    ax.set_ylim(0.30, 1.05)
    ax.set_xticks(_xticks)
    ax.set_xticklabels(_xlabels, fontsize=FONT_TICK)
    ax.set_yticks([0.50, 0.75, 1.00])
    ax.set_yticklabels(["50", "75", "100"], fontsize=FONT_TICK)
    ax.set_xlabel("Absolute offset (mm)", fontsize=FONT_LABEL,
                  labelpad=ATD.FIG_AXIS_LABELPAD)
    ax.legend(fontsize=FONT_ANNOT, frameon=False,
              loc="lower right", bbox_to_anchor=(0.98, 0.02),
              handlelength=1.4)
    _despine(ax)
    _draw_inward_ticks(ax)


def _finalize_psych_panels(fig, ax_l, ax_r):
    """Lock equal panel boxes; y ticks on left panel only; titles in figure coords."""
    ax_r.tick_params(labelleft=False)

    pos_l, pos_r = ax_l.get_position(), ax_r.get_position()
    w, h, y0 = pos_l.width, pos_l.height, pos_l.y0
    gap = pos_r.x0 - pos_l.x1
    ax_l.set_position([pos_l.x0, y0, w, h])
    ax_r.set_position([pos_l.x0 + w + gap - PSYCH_R_PANEL_SHIFT, y0, w, h])

    ax_l.set_title("")
    ax_r.set_title("")
    for ax, force in ((ax_l, 1.0), (ax_r, 26.0)):
        p = ax.get_position()
        fig.text(
            p.x0 + p.width / 2, p.y1 + 0.015,
            f"{force:g} g",
            ha="center", va="bottom",
            fontsize=FONT_LABEL, fontweight="bold",
            color=FORCE_COLORS[force],
            transform=fig.transFigure,
        )


def make_psychometric_fig():
    fig, ax_l, ax_r = _two_panel()
    _psych_panel(ax_l, 1.0)
    _psych_panel(ax_r, 26.0)
    fig.supylabel(
        "Proportion correct (%)",
        fontsize=FONT_LABEL,
        x=0.04,
        ha="center",
    )
    _finalize_psych_panels(fig, ax_l, ax_r)
    return fig

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
#fig = make_jnd_paired()
#save_fig(fig, "sd_jnd_paired")
#plt.close(fig)

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
               alpha=0.7, zorder=1)
    ax.axhline(0.5, color="#AAAAAA", lw=0.8, linestyle=":",
               alpha=0.6, zorder=1)

    # Symmetry annotation
    ax.text(-0.3, 0.33, "← 2nd to LEFT",
            ha="right", va="center", fontsize=FONT_ANNOT,
            color="#555555", style="italic")
    ax.text(0.3, 0.33, "2nd to RIGHT →",
            ha="left", va="center", fontsize=FONT_ANNOT,
            color="#555555", style="italic")

    ax.set_xlabel("Signed offset (mm)  [negative = 2nd is LEFT]",
                  fontsize=FONT_LABEL, labelpad=ATD.FIG_AXIS_LABELPAD)
    ax.set_ylabel("Proportion correct (%)", fontsize=FONT_LABEL,
                  labelpad=ATD.FIG_AXIS_LABELPAD)
    ax.set_title("Left–Right Symmetry Check", fontsize=FONT_LABEL,
                 fontweight="bold", pad=8)
    ax.set_ylim(0.25, 1.05)
    ax.set_yticks([0.50, 0.75, 1.00])
    ax.set_yticklabels(["50", "75", "100"], fontsize=FONT_TICK)
    ax.legend(fontsize=FONT_ANNOT, frameon=False,
              loc="lower right", bbox_to_anchor=(0.98, 0.02),
              handlelength=1.4)
    _despine(ax)
    _draw_inward_ticks(ax)
    fig.tight_layout()
    return fig

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

# Blue ramp for mm boxplot: larger |offset| → darker blue (force-specific)
ABS_OFFSET_BLUE_1G = {
    1.5: "#C8DAEF",
    3.0: "#85B1D9",
    4.5: "#4A90C2",
    6.0: "#10559A",   # darkest (matches ATD On-touch)
}
ABS_OFFSET_BLUE_26G = {
    1.5: "#B8C9E8",
    3.0: "#6B8FC7",
    4.5: "#3D5F9A",
    6.0: "#1A2F5C",
}
ABS_OFFSET_BLUE_BY_FORCE = {1: ABS_OFFSET_BLUE_1G, 26: ABS_OFFSET_BLUE_26G}


def _color_by_abs_offset(signed_mm, force):
    palette = ABS_OFFSET_BLUE_BY_FORCE.get(force, ABS_OFFSET_BLUE_1G)
    return palette.get(abs(signed_mm), "#888888")

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


def make_signed_boxplot_mm(force):
    """Single-panel figure: x-axis shows signed distance in mm (one tick per distance)."""
    sns.set_theme(style="white")
    ATD.apply_plot_style()

    signed_vals = sorted(subj_sym["signed_offset_mm"].unique())
    bw      = 0.14
    gap_in  = 0.28   # wider spacing so tick labels don't crowd
    rng     = np.random.default_rng(42)

    x_positions = {dist: di * gap_in for di, dist in enumerate(signed_vals)}

    fw = FIG_SIZE[0]   # 8.0 in — same canvas as Fig2 (2-col export width)
    fh = FIG_SIZE[1]   # 4.5 in
    fig, ax = plt.subplots(figsize=(fw, fh), facecolor="#FFFFFF")

    box_fill = ATD.pale_box_face(ATD.ON_TOUCH)
    scatter_rgba = ATD._hsb_scatter_rgba(ATD.ON_TOUCH)

    for dist in signed_vals:
        xp    = x_positions[dist]
        sub   = (subj_sym[
                     (subj_sym["force_g"] == force) &
                     (subj_sym["signed_offset_mm"] == dist)
                 ]["accuracy"].dropna().values * 100)
        if len(sub) == 0:
            continue

        bp = ax.boxplot(
            sub, positions=[xp], widths=bw * 0.82,
            patch_artist=True, showfliers=False,
            medianprops=dict(color="#CC0000", lw=2.0),
            whiskerprops=dict(color="#000000", lw=1.4),
            capprops=dict(color="#000000", lw=1.4),
            boxprops=dict(facecolor=box_fill,
                          edgecolor="#000000", lw=1.8),
        )
        bp["boxes"][0].set_edgecolor("#000000")
        for whisker in bp["whiskers"]:
            whisker.set_color("#000000")
        for cap in bp["caps"]:
            cap.set_color("#000000")
        jitter = rng.uniform(-bw * 0.30, bw * 0.30, len(sub))
        ax.scatter(xp + jitter, sub, c=[scatter_rgba] * len(sub), s=28,
                   edgecolors="none", linewidths=0, zorder=5)

    ax.axhline(
        THRESHOLD * 100,
        color=ATD.CRITERION_COLOR,
        linestyle="--",
        linewidth=1.0,
        alpha=0.85,
        zorder=2,
    )

    # x-axis: one tick per mm value
    ax.set_xticks(list(x_positions.values()))
    ax.set_xticklabels(
        [f"{d:+.1f}" for d in signed_vals],
        fontsize=FONT_TICK, rotation=0,
    )
    ax.set_xlim(min(x_positions.values()) - bw, max(x_positions.values()) + bw)
    ax.set_ylim(ATD.ACCURACY_YMIN, ATD.ACCURACY_YLIM_TOP)
    ax.set_yticks(ATD.ACCURACY_YTICKS)
    ax.set_yticklabels(["0", "20", "40", "60", "80", "100"], fontsize=FONT_TICK)
    y0, y1 = ATD.ACCURACY_YSPINE
    ax.spines["left"].set_bounds(y0, y1)
    ax.set_xlabel("Offset distance (mm)", fontsize=FONT_LABEL + 4,
                  labelpad=ATD.FIG_AXIS_LABELPAD)
    ax.set_ylabel("Spatial accuracy (%)", fontsize=FONT_LABEL + 4,
                  labelpad=ATD.FIG_AXIS_LABELPAD)

    _despine(ax)
    fig.subplots_adjust(
        left=0.11, right=0.98,
        top=0.92, bottom=ATD.FIG_LEGEND_BOTTOM,
    )
    _draw_inward_ticks(ax)
    return fig


def _cvsd_panel(ax, force):
    """Paired C vs D accuracy boxplot for one force level."""
    sub = subj_cvsd[subj_cvsd["force_g"] == force].copy()
    order = [REGION_C, REGION_D]
    rng = np.random.default_rng(11)
    bw = 0.38

    for xi, region in enumerate(order):
        vals = sub.loc[sub["Area"] == region, "accuracy_pct"].values
        color = REGION_COLORS[region]
        ax.boxplot(
            [vals],
            positions=[xi],
            widths=bw,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color=SIG_COLOR, lw=2.0),
            whiskerprops=dict(color="#111111", lw=1.4),
            capprops=dict(color="#111111", lw=1.4),
            boxprops=dict(facecolor=color, alpha=0.65,
                          edgecolor="#111111", lw=1.8),
        )
        jitter = rng.uniform(-0.10, 0.10, len(vals))
        ax.scatter(xi + jitter, vals, color=color, s=28, alpha=0.40,
                   edgecolors="none", zorder=5)

    # Per-subject paired lines
    wide = sub.pivot(index="Subject", columns="Area", values="accuracy_pct")
    for subj, row in wide.iterrows():
        if REGION_C in row and REGION_D in row and pd.notna(row[REGION_C]) and pd.notna(row[REGION_D]):
            ax.plot([0, 1], [row[REGION_C], row[REGION_D]],
                    color="#888888", alpha=0.35, lw=0.9, zorder=3)

    wide_p = wide.dropna(subset=[REGION_C, REGION_D], how="any")
    if len(wide_p) >= 5:
        _, p_w = stats.wilcoxon(wide_p[REGION_C], wide_p[REGION_D])
        y_max = max(wide_p.max().max(), sub["accuracy_pct"].max())
        _add_sig_bracket(ax, 0, 1, y_max + 2.5, text=_star_from_p(p_w))

    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"Region {REGION_C}", f"Region {REGION_D}"],
                       fontsize=FONT_TICK)
    ax.set_xlim(-0.55, 1.55)
    ax.set_ylim(ATD.ACCURACY_YMIN, ATD.ACCURACY_YLIM_TOP)
    ax.set_yticks(ATD.ACCURACY_YTICKS)
    ax.set_yticklabels(["0", "20", "40", "60", "80", "100"], fontsize=FONT_TICK)
    y0, y1 = ATD.ACCURACY_YSPINE
    ax.spines["left"].set_bounds(y0, y1)
    ax.axhline(75, color="#555555", lw=1.0, linestyle=":", alpha=0.7, zorder=1)
    ax.axhline(50, color="#AAAAAA", lw=0.8, linestyle=":", alpha=0.6, zorder=1)
    _despine(ax)
    _draw_inward_ticks(ax)


def make_cvsd_fig():
    sns.set_theme(style="white")
    ATD.apply_plot_style()
    fig, ax_l, ax_r = _two_panel()
    _cvsd_panel(ax_l, 1.0)
    _cvsd_panel(ax_r, 26.0)
    fig.supylabel(
        "Spatial accuracy (%)",
        fontsize=FONT_LABEL,
        x=0.04,
        ha="center",
    )
    _finalize_psych_panels(fig, ax_l, ax_r)
    mode_note = (
        "Area logged per trial"
        if _region_mode.startswith("Area")
        else "C = 2nd left of 1st; D = 2nd right (C–D zone)"
    )
    fig.text(
        0.5, 0.02,
        mode_note,
        ha="center", va="bottom",
        fontsize=FONT_ANNOT, color="#555555", style="italic",
        transform=fig.transFigure,
    )
    return fig


def render_all_figures(stem_suffix=""):
    """Render all active SD figures; stem_suffix e.g. '_n15_m8f7' for subset."""
    tag = stem_suffix
    label = f"n={n_subj}" + (f" ({stem_suffix.lstrip('_')})" if tag else " (full)")
    print(f"\n[Render figures — {label}]")

    print("  [Fig 1] Psychometric curves ...")
    fig = make_psychometric_fig()
    save_fig_psych(fig, f"sd_psychometric_curves{tag}")
    plt.close(fig)

    print("  [Fig 3] Symmetry check ...")
    fig = make_symmetry_fig()
    save_fig(fig, f"sd_symmetry{tag}")
    plt.close(fig)

    print("  [Fig 4d] Signed-offset boxplot – 1 g (mm) ...")
    fig = make_signed_boxplot_mm(1.0)
    save_fig_2col(fig, f"sd_signed_boxplot_1g_mm{tag}")
    plt.close(fig)

    print("  [Fig 4e] Signed-offset boxplot – 26 g (mm) ...")
    fig = make_signed_boxplot_mm(26.0)
    save_fig_2col(fig, f"sd_signed_boxplot_26g_mm{tag}")
    plt.close(fig)

    print("  [Fig 5] Region C vs D paired accuracy ...")
    fig = make_cvsd_fig()
    save_fig_2col(fig, f"sd_cvsd_paired{tag}")
    plt.close(fig)


# ================================================================
# 7. Run cohorts & render
# ================================================================
# Full cohort (n=25)
recompute_cohort(df_all, cohort_label="Full cohort", aux_tag="", run_gee=True)
render_all_figures("")

# Balanced random subset (n=15: 8 M + 7 F)
subset_ids = select_gender_balanced_subjects(
    SD_GENDER, SUBSET_N_M, SUBSET_N_F, SUBSET_SEED,
)
subset_meta = pd.DataFrame({
    "Subject": subset_ids,
    "Gender":  [SD_GENDER[s] for s in subset_ids],
})
subset_csv = os.path.join(OUTPUT_DIR, f"sd_subset{SUBSET_STEM}_subjects.csv")
subset_meta.to_csv(subset_csv, index=False)
print(f"\n[Subset {SUBSET_STEM}] seed={SUBSET_SEED}  "
      f"M={SUBSET_N_M} F={SUBSET_N_F}  →  {', '.join(subset_ids)}")
print(f"  Saved {os.path.basename(subset_csv)}")

df_subset = df_all[df_all["Subject"].isin(subset_ids)].copy()
recompute_cohort(
    df_subset,
    cohort_label=f"Subset {SUBSET_STEM}",
    aux_tag=SUBSET_STEM,
    run_gee=False,
)
render_all_figures(SUBSET_STEM)

# ================================================================
# 8. Print summary table (last cohort = subset)
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
print("  Full n=25  : sd_*.{png,csv,txt}")
print(f"  Subset n=15: sd_*{SUBSET_STEM}.{{png,csv,txt}}  (8M + 7F, seed={SUBSET_SEED})")
print("  sd_subset_n15_m8f7_subjects.csv — selected participant list")