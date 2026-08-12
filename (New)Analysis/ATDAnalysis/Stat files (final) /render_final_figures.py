"""render_final_figures.py
=========================
최종 논문 피규어 3개를 통일된 스타일로 생성하는 스크립트.

  Fig A: On-touch vs In-air
  Fig B: Fingerpad vs Periungual On-touch (0.4 g extended)
  Fig C: On-nail (C+D) vs Off-nail (A+F) pooled

► 스타일을 바꾸려면 UNIFIED_STYLE 딕셔너리(Section 2)만 수정하세요.
► 실행 방법:
      python "(New)Analysis/ATDAnalysis/render_final_figures.py"

출력 파일 (2-col, 600 dpi):
  atd_c1_outputs/Fig2_ontouch_vs_inair(final)_2col.png
  atd_c1_outputs/Fig3_future_0p4g(final)_2col.png
  figures/onnail_vs_offnail_pooled(final).png
"""

import os
import glob
import importlib.util

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["font.family"]     = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]
import matplotlib.patches as mpatches
from matplotlib.ticker import FixedLocator
from matplotlib.transforms import blended_transform_factory
import seaborn as sns
import statsmodels.formula.api as smf

# =============================================================================
# 1. PATHS
# =============================================================================
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
ATD_ROOT    = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
ATD_C1_PATH = os.path.join(SCRIPT_DIR, "(Final)ATD_C1_Fig(Anika).py")
OUT_C1      = os.path.join(ATD_ROOT, "atd_c1_outputs")
OUT_AGG     = os.path.join(ATD_ROOT, "figures")
OUT_FINAL   = os.path.join(ATD_ROOT, "Final")
os.makedirs(OUT_C1, exist_ok=True)
os.makedirs(OUT_AGG, exist_ok=True)
os.makedirs(OUT_FINAL, exist_ok=True)

# Shared On-touch blue — all box plots in Final/ use this base color
ON_TOUCH_BLUE = "#10559A"

# =============================================================================
# 2. UNIFIED STYLE  ← 이 딕셔너리 하나만 수정하면 세 피규어 모두 반영됩니다
# =============================================================================
UNIFIED_STYLE = dict(
    # ── Colors ───────────────────────────────────────────────────────────────
    IN_AIR                    = "#6A4A3C",   # In-air (brown)
    ON_TOUCH                  = ON_TOUCH_BLUE,   # On-touch (blue)
    POOL_ON_NAIL              = ON_TOUCH_BLUE,   # pooled On-nail — same as On-touch
    POOL_OFF_NAIL             = ON_TOUCH_BLUE,   # pooled Off-nail — same as On-touch
    ACCENT_RED                = "#BF2C23",   # median line
    BLACK                     = "#1A1A1A",
    KAO_COLOR                 = "#5A5A5A",   # Fingerpad (gray)

    # ── Scatter ──────────────────────────────────────────────────────────────
    STRIP_ALPHA               = 0.50,
    SCATTER_HSB_BRIGHTNESS    = 0.60,

    # ── Box fill ─────────────────────────────────────────────────────────────
    COND_BOX_BRIGHTNESS       = 0.88,
    COND_BOX_SATURATION_SCALE = 0.40,
    COND_BOX_ALPHA_HEX        = "40",       # ~25% opacity

    # ── Line widths ──────────────────────────────────────────────────────────
    BOX_LINEWIDTH             = 0.8,
    CAP_LINEWIDTH             = 0.5,
    BRACKET_LINEWIDTH         = 1.5,

    # ── Font sizes ───────────────────────────────────────────────────────────
    FONT_TICK                 = 16,
    FONT_LABEL                = 17,
    FONT_LEGEND               = 12,
    FONT_ANNOT                = 10,
    FONT_BRACKET_STAR         = 15,  # significance asterisk size (*** / *)
    FONT_PANEL_TITLE          = 14,  # force panel titles (0.16 g, 0.4 g, …)

    # ── Y-axis  (세 피규어 모두 동일한 tick 간격) ─────────────────────────────
    ACCURACY_YTICKS           = (0, 20, 40, 60, 80, 100),

    # ── Tick lengths (x/y equal — fraction of respective axis dimension) ─────
    TICK_LEN_AXES             = 0.016,
)

# 2-col export (unified canvas: 2102×1298 px)
EXPORT_WIDTH = 2102
EXPORT_HEIGHT = 1298
EXPORT_TAG   = "2col"
POOLED_FIGSIZE = (8.0, round(8.0 * EXPORT_HEIGHT / EXPORT_WIDTH, 3))

# =============================================================================
# 3. Load & patch atd_c1 module
#    (데이터 로딩·전처리도 여기서 수행됨 → atd_c1.df_raw 사용 가능)
# =============================================================================
print("Loading atd_c1 module …")
_spec  = importlib.util.spec_from_file_location("atd_c1", ATD_C1_PATH)
atd_c1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(atd_c1)

# Patch all overridable style constants
for _k, _v in UNIFIED_STYLE.items():
    if hasattr(atd_c1, _k):
        setattr(atd_c1, _k, _v)

# Re-sync derived aliases
atd_c1.CRITERION_COLOR             = atd_c1.BLACK
atd_c1.ON_TOUCH_BOX_BRIGHTNESS     = UNIFIED_STYLE["COND_BOX_BRIGHTNESS"]
atd_c1.ON_TOUCH_BOX_SATURATION_SCALE = UNIFIED_STYLE["COND_BOX_SATURATION_SCALE"]
atd_c1.ON_TOUCH_BOX_ALPHA_HEX      = UNIFIED_STYLE["COND_BOX_ALPHA_HEX"]
atd_c1.OUT_DIR = OUT_C1

# Convenience shortcuts
S             = UNIFIED_STYLE
pale_box_face = atd_c1.pale_box_face

FONT_TICK     = S["FONT_TICK"]
FONT_LABEL    = S["FONT_LABEL"]
BLACK         = S["BLACK"]
ACCENT_RED    = S["ACCENT_RED"]
BOX_LINEWIDTH = S["BOX_LINEWIDTH"]
CAP_LINEWIDTH = S["CAP_LINEWIDTH"]
ON_TOUCH      = S["ON_TOUCH"]

POOLED_Y_TICKS    = list(S["ACCURACY_YTICKS"])
POOLED_YLIM_TOP   = 120
POOLED_Y_AXIS_TOP = 100


# =============================================================================
# 4. Helper: save at 2-col width
# =============================================================================
def save_final(fig, out_path, width_px=EXPORT_WIDTH, height_px=EXPORT_HEIGHT):
    import io
    from PIL import Image

    buf = io.BytesIO()
    fig.savefig(
        buf, format="png", dpi=600, bbox_inches="tight",
        pad_inches=0.04, facecolor="white",
    )
    buf.seek(0)
    master = Image.open(buf).convert("RGB")
    master.resize((width_px, height_px), Image.Resampling.LANCZOS).save(out_path)
    print(f"  Saved → {out_path}  ({width_px}×{height_px} px)")


def publish_to_final(src_path, dest_name):
    """Copy a rendered PNG into ATDAnalysis/Final/."""
    import shutil

    dest = os.path.join(OUT_FINAL, dest_name)
    shutil.copy2(src_path, dest)
    print(f"  Published → {dest}")


# =============================================================================
# 5. Fig A — On-touch vs In-air
# =============================================================================
def generate_fig2():
    print("\n[Fig A] On-touch vs In-air …")
    df_raw = atd_c1.df_raw
    cond_list = [c for c in ["In-air", "On-touch (Mid)"]
                 if c in df_raw["Condition"].unique()]
    cond_colors = {
        "In-air":         S["IN_AIR"],
        "On-touch (Mid)": S["ON_TOUCH"],
    }
    atd_c1.plot_ontouch_vs_inair(
        df_raw,
        cond_list,
        cond_colors,
        "Fig2_ontouch_vs_inair(final)",
        export_widths=((EXPORT_TAG, EXPORT_WIDTH),),
        scatter_brightness=S["SCATTER_HSB_BRIGHTNESS"],
        partial_subjects=atd_c1._PARTIAL_SUBJ,
    )


# =============================================================================
# 6. Fig B — Fingerpad vs Periungual On-touch (0.4 g extended)
# =============================================================================
def generate_fig3():
    print("\n[Fig B] Fingerpad vs Periungual On-touch (0.4 g extended) …")
    # Kao data extended to 0.4 g (same as export_fig3_future_0p4g in atd_c1)
    kao_rows_ext = []
    for force, vals in atd_c1.KAO_PAINT_RAW.items():
        if force <= 0.4:
            for pid, v in enumerate(vals):
                kao_rows_ext.append({
                    "Force_Val":   float(force),
                    "Score":       float(v),
                    "Source":      atd_c1.KAO_LABEL,
                    "Participant": f"KP{pid + 1}",
                })
    df_kao_ext = pd.DataFrame(kao_rows_ext)
    df_peri    = atd_c1.df_raw[atd_c1.df_raw["Condition"] == "On-touch (Mid)"].copy()

    pale_kw = dict(
        peri_box_brightness       = S["COND_BOX_BRIGHTNESS"],
        peri_box_alpha_hex        = S["COND_BOX_ALPHA_HEX"],
        peri_box_saturation_scale = S["COND_BOX_SATURATION_SCALE"],
    )
    atd_c1.plot_kao_vs_periungual(
        df_peri,
        atd_c1.ONTouch_LABEL,
        S["ON_TOUCH"],
        "Fig3_future_0p4g(final)",
        export_widths=((EXPORT_TAG, EXPORT_WIDTH),),
        region_background=False,
        region_labels=False,
        highlight_forces=[0.07, 0.4],
        scatter_brightness=S["SCATTER_HSB_BRIGHTNESS"],
        kao_df_override=df_kao_ext,
        partial_subjects=atd_c1._PARTIAL_SUBJ,
        participant_median=True,
        **pale_kw,
    )


# =============================================================================
# 6b. Fig B (full Kao) — Fingerpad (all forces) vs Periungual On-touch
# =============================================================================
def generate_fig3_full_kao():
    print("\n[Fig B full-Kao] Fingerpad (0.02–1.4 g) vs Periungual On-touch …")
    # All Kao forces included (0.02, 0.04, 0.07, 0.40, 1.00, 1.40 g)
    kao_rows_all = []
    for force, vals in atd_c1.KAO_PAINT_RAW.items():
        for pid, v in enumerate(vals):
            kao_rows_all.append({
                "Force_Val":   float(force),
                "Score":       float(v),
                "Source":      atd_c1.KAO_LABEL,
                "Participant": f"KP{pid + 1}",
            })
    df_kao_all = pd.DataFrame(kao_rows_all)
    df_peri    = atd_c1.df_raw[atd_c1.df_raw["Condition"] == "On-touch (Mid)"].copy()

    pale_kw = dict(
        peri_box_brightness       = S["COND_BOX_BRIGHTNESS"],
        peri_box_alpha_hex        = S["COND_BOX_ALPHA_HEX"],
        peri_box_saturation_scale = S["COND_BOX_SATURATION_SCALE"],
    )
    atd_c1.plot_kao_vs_periungual(
        df_peri,
        atd_c1.ONTouch_LABEL,
        S["ON_TOUCH"],
        "Fig3_future_full_kao(final)",
        export_widths=((EXPORT_TAG, EXPORT_WIDTH),),
        region_background=False,
        region_labels=False,
        highlight_forces=[0.07, 0.4, 1.0, 1.4],
        highlight_force_edge_pad={
            1.0: (0.06, 0.01),
            1.4: (0.01, 0.04),
        },
        highlight_force_min_gap=0.05,
        scatter_brightness=S["SCATTER_HSB_BRIGHTNESS"],
        kao_df_override=df_kao_all,
        partial_subjects=atd_c1._PARTIAL_SUBJ,
        participant_median=True,
        **pale_kw,
    )


# =============================================================================
# 7. Fig C — On-nail (C+D) vs Off-nail (A+F) pooled
#    (atd_c1.df_raw 재사용 — 별도 데이터 로딩 불필요)
# =============================================================================

def _subject_area_pool_v(df_in, sub_col, score_col, area_group_map, force_val):
    """Per-subject mean accuracy per area, relabelled to group (generic score column)."""
    areas = list(area_group_map.keys())
    sub   = df_in[df_in["Area"].isin(areas)].dropna(
        subset=[sub_col, "Area", score_col]
    )
    sub   = sub[np.isclose(sub["Force_Val"], force_val)]
    if sub.empty:
        return pd.DataFrame(columns=[sub_col, "Area", "Group", "accuracy"])
    agg = (
        sub.groupby([sub_col, "Area"], as_index=False)[score_col]
        .mean()
        .rename(columns={score_col: "accuracy"})
    )
    agg["Group"] = agg["Area"].map(area_group_map)
    return agg


def _lme_two_groups(df, sub_col, ref_group, target_group):
    sub = df[df["Group"].isin([ref_group, target_group])].dropna(
        subset=[sub_col, "Group", "accuracy"]
    )
    if sub[sub_col].nunique() < 2 or sub["Group"].nunique() < 2:
        return None
    formula = f"accuracy ~ C(Group, Treatment(reference='{ref_group}'))"
    try:
        res = smf.mixedlm(formula, sub, groups=sub[sub_col]).fit(reml=True)
        col = f"C(Group, Treatment(reference='{ref_group}'))[T.{target_group}]"
        if col not in res.params.index:
            return None
        ci = res.conf_int().loc[col]
        return {
            "coef":  float(res.params[col]),
            "ci_lo": float(ci[0]),
            "ci_hi": float(ci[1]),
            "p":     float(res.pvalues[col]),
        }
    except Exception:
        return None


def _sig_bracket(ax, x_l, x_r, y_base, text="", tick_h=0.5, text_pad=0.0):
    mid   = (x_l + x_r) / 2
    y_top = y_base + tick_h
    ax.plot([x_l, x_r], [y_top, y_top],
            color=ACCENT_RED, linewidth=S["BRACKET_LINEWIDTH"], clip_on=False, zorder=25)
    ax.text(mid, y_top + text_pad, text,
            ha="center", va="bottom", fontsize=S["FONT_BRACKET_STAR"],
            color=ACCENT_RED, fontweight="bold", clip_on=False, zorder=26)


def _force_title(ax, force_val, y=0.95):
    txt = f"{force_val:g}"
    if "." not in txt:
        txt += ".0"
    ax.text(0.5, y, f"{txt} g", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=S["FONT_PANEL_TITLE"],
            fontweight="normal", clip_on=False)


def _scatter_strip(ax, x_pos, vals, subjects, partial_set, color, jitter_arr):
    rgba = atd_c1._hsb_scatter_rgba(
        color,
        brightness=S["SCATTER_HSB_BRIGHTNESS"],
        alpha=S["STRIP_ALPHA"],
    )
    mask = np.array([s in partial_set for s in subjects])
    kw   = dict(linewidths=0, zorder=3, clip_on=False)
    if (~mask).any():
        ax.scatter(x_pos + jitter_arr[~mask], vals[~mask],
                   c=[rgba] * int((~mask).sum()), s=3.5 ** 2, marker="o", **kw)
    if mask.any():
        ax.scatter(x_pos + jitter_arr[mask], vals[mask],
                   c=[rgba] * int(mask.sum()), s=(3.5 * 1.3) ** 2, marker="^", **kw)


def _inward_ticks(ax, x_positions, y_ticks):
    frac_x = S["TICK_LEN_AXES"]
    frac_y = atd_c1.y_tick_frac_match_x(ax, frac_x)
    ax.tick_params(axis="both", which="both", length=0)
    x_tr = blended_transform_factory(ax.transData, ax.transAxes)
    y_tr = blended_transform_factory(ax.transAxes, ax.transData)
    kw   = dict(color=BLACK, linewidth=1.0, solid_capstyle="butt",
                clip_on=False, zorder=6)
    for xi in x_positions:
        ax.plot([xi, xi], [0, frac_x], transform=x_tr, **kw)
    lo, hi = ax.get_ylim()
    for y in y_ticks:
        if lo - 1e-9 <= y <= hi + 1e-9:
            ax.plot([0, frac_y], [y, y], transform=y_tr, **kw)


def generate_pooled():
    print("\n[Fig C] On-nail (C+D) vs Off-nail (A+F) pooled …")

    # atd_c1.df_raw 재사용: Score 컬럼을 Relative_Score로 매핑
    df = atd_c1.df_raw.copy()
    df["Relative_Score"] = df["Score"]
    sub_col    = atd_c1.SUBJECT_COL
    partial_s  = atd_c1._PARTIAL_SUBJ
    EXCLUDE    = {0.07, 1.4}
    plot_forces = sorted(f for f in df["Force_Val"].unique() if f not in EXCLUDE)

    POOL_GROUP_MAP   = {"C": "On-nail", "D": "On-nail",
                        "A": "Off-nail", "F": "Off-nail"}
    POOL_GROUP_ORDER = ["Off-nail", "On-nail"]
    POOL_PALETTE     = {"On-nail": S["POOL_ON_NAIL"], "Off-nail": S["POOL_OFF_NAIL"]}
    POOL_X_LABELS    = ["LNF\n(a+f)", "PNF\n(c+d)"]
    FONT_XTICK       = 12

    fig, axes = plt.subplots(1, len(plot_forces),
                             figsize=POOLED_FIGSIZE, facecolor="white")
    if len(plot_forces) == 1:
        axes = [axes]

    rng = np.random.default_rng(42)

    for ax, fval in zip(axes, plot_forces):
        df_f = _subject_area_pool_v(df, sub_col, "Relative_Score",
                                    POOL_GROUP_MAP, fval)
        lme  = _lme_two_groups(df_f, sub_col,
                               ref_group="Off-nail", target_group="On-nail")

        tops = {}
        for xi, grp in enumerate(POOL_GROUP_ORDER):
            rows  = df_f[df_f["Group"] == grp].dropna(subset=["accuracy"])
            data  = rows["accuracy"].values
            subjs = rows[sub_col].values
            bp = ax.boxplot(
                [data], positions=[xi], widths=0.45,
                patch_artist=True, showfliers=False, zorder=2,
                whiskerprops=dict(color=BLACK, linewidth=BOX_LINEWIDTH),
                capprops    =dict(color=BLACK, linewidth=CAP_LINEWIDTH),
                medianprops =dict(color=ACCENT_RED, linewidth=2.0),
                boxprops    =dict(facecolor=pale_box_face(POOL_PALETTE[grp]),
                                  edgecolor=BLACK, linewidth=BOX_LINEWIDTH),
            )
            whiskers = [w.get_ydata()[1] for w in bp["whiskers"]]
            tops[grp] = max(whiskers) if whiskers else 0.0
            jitter    = rng.uniform(-0.12, 0.12, size=len(data))
            _scatter_strip(ax, xi, data, subjs, partial_s,
                           POOL_PALETTE[grp], jitter)

        if lme:
            star  = ("***" if lme["p"] < 0.001 else
                     "**"  if lme["p"] < 0.01  else
                     "*"   if lme["p"] < 0.05  else "n.s.")
            y_brk = max(max(tops.values()) + 3.0, 103)
            _sig_bracket(ax, 0, 1, y_brk, text=star)

        ax.axhline(80, color=BLACK, linestyle="--",
                   linewidth=1.0, alpha=0.85, zorder=20)
        _force_title(ax, fval)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(POOL_X_LABELS, fontsize=FONT_XTICK)
        ax.set_yticks(POOLED_Y_TICKS)
        ax.yaxis.set_major_locator(FixedLocator(POOLED_Y_TICKS))
        ax.tick_params(axis="y", labelsize=FONT_TICK)
        ax.tick_params(axis="x", length=0)
        ax.set_ylim(-5, POOLED_YLIM_TOP)
        ax.spines["left"].set_bounds(-5, POOLED_Y_AXIS_TOP)
        sns.despine(ax=ax)
        if fval == plot_forces[0]:
            ax.set_ylabel("Detection Accuracy (%)", fontsize=FONT_LABEL)
        else:
            ax.set_ylabel("")
            ax.tick_params(axis="y", labelleft=False)

    fig.subplots_adjust(left=0.08, right=0.97, top=0.96, bottom=0.12, wspace=0.18)
    fig.canvas.draw()
    for ax in axes:
        _inward_ticks(ax, x_positions=[0, 1], y_ticks=POOLED_Y_TICKS)

    out = os.path.join(OUT_AGG, "onnail_vs_offnail_pooled(final).png")
    save_final(fig, out)
    plt.close(fig)
    return out


def publish_final_figures():
    """Sync the three publication figures into ATDAnalysis/Final/."""
    print("\n[Publishing] Copy figures to Final/ …")
    publish_to_final(
        os.path.join(OUT_C1, "Fig2_ontouch_vs_inair(final)_2col.png"),
        "Fig2_ontouch_vs_inair_2col.png",
    )
    publish_to_final(
        os.path.join(OUT_C1, "Fig3_future_full_kao(final)_2col.png"),
        "Fig3_future_full_kao(final)_2col.png",
    )
    publish_to_final(
        os.path.join(OUT_AGG, "onnail_vs_offnail_pooled(final).png"),
        "onnail_vs_offnail_pooled(final).png",
    )


# =============================================================================
# 8. Run all
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Rendering final figures with unified style …")
    print(f"On-touch blue: {ON_TOUCH_BLUE}")
    print(f"OUT_C1   → {OUT_C1}")
    print(f"OUT_AGG  → {OUT_AGG}")
    print(f"OUT_FINAL → {OUT_FINAL}")
    print("=" * 60)
    generate_fig2()
    generate_fig3()
    generate_fig3_full_kao()
    generate_pooled()
    publish_final_figures()
    print("\nDone.")
