"""
create_paper_stats_csv.py
=========================
논문 작성용 통계 CSV — 피겨당 파일 하나씩 생성.

출력 파일 (paper_stats/ 폴더):
  Fig2_ontouch_vs_inair(final).csv
  Fig3_future_full_kao(final).csv
  onnail_vs_offnail_pooled(final).csv

각 파일 구조 (section 컬럼으로 구분):
  descriptive       — 기술통계 (mean, SD, median, Q1, Q3, IQR, whiskers, n)
  inferential       — 추론통계 (p-value, LME coef/CI, Cohen's d, Hedges' g)
  subject_data      — 피험자별 개별 데이터 포인트
  sensitivity       — Sensitivity analysis (primary vs 각 sensitivity cohort 비교)
  kao_raw           — (Fig3 전용) Kao et al. 2022 원본 개별 데이터
"""

import os
import importlib.util
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── 경로 설정 ─────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
ATD_C1_PATH = os.path.join(SCRIPT_DIR,
    "Stat files (final) ", "(Final)ATD_C1_Fig(Anika).py")
OUT_DIR     = os.path.join(SCRIPT_DIR, "paper_stats")
os.makedirs(OUT_DIR, exist_ok=True)

# ── ATD 데이터 로딩 ───────────────────────────────────────────────────────────
print("ATD 데이터 로딩 중 …")
spec = importlib.util.spec_from_file_location("atd_c1", ATD_C1_PATH)
atd  = importlib.util.module_from_spec(spec)
spec.loader.exec_module(atd)

df_raw      = atd.df_raw.copy()
sub_col     = atd.SUBJECT_COL
partial_set = atd._PARTIAL_SUBJ

# ── 기존 통계 CSV 로딩 ────────────────────────────────────────────────────────
STATS_C1  = os.path.join(SCRIPT_DIR, "atd_c1_outputs", "stats")
STATS_AGG = os.path.join(SCRIPT_DIR, "figures", "stats")

def load(path):
    return pd.read_csv(path)

desc2 = load(os.path.join(STATS_C1,  "Fig2_ontouch_vs_inair(final)_descriptives.csv"))
comp2 = load(os.path.join(STATS_C1,  "Fig2_ontouch_vs_inair(final)_comparisons.csv"))
desc3 = load(os.path.join(STATS_C1,  "Fig3_future_full_kao(final)_descriptives.csv"))
comp3 = load(os.path.join(STATS_C1,  "Fig3_future_full_kao(final)_comparisons.csv"))
descC = load(os.path.join(STATS_AGG, "onnail_vs_offnail_pooled(final)_descriptives.csv"))
compC = load(os.path.join(STATS_AGG, "onnail_vs_offnail_pooled(final)_comparisons.csv"))
sens  = load(os.path.join(STATS_C1,  "final_figures_sensitivity_summary.csv"))

POOL_MAP      = {"C": "On-nail (C+D)", "D": "On-nail (C+D)",
                 "A": "Off-nail (A+F)", "F": "Off-nail (A+F)"}
EXCLUDE_FORCE = {0.07, 1.4}


# ─────────────────────────────────────────────────────────────────────────────
# 공통 헬퍼
# ─────────────────────────────────────────────────────────────────────────────
def cohort_label(code):
    return {
        "primary_all":                "Primary: all included subjects (matches final figures)",
        "sensitivity_full_n30":       "Sensitivity: full protocol only (P1–P60, n=30)",
        "sensitivity_partial_n15":    "Sensitivity: partial protocol only (P61–P75, n=15)",
        "sensitivity_trial_pooled_n30": "Sensitivity: trial-pooled subject means, full protocol (n=30 per group)",
    }.get(code, code)


def concat_sections(sections):
    """섹션 리스트(각 df)를 이어붙여 하나의 DataFrame으로 반환."""
    return pd.concat(sections, ignore_index=True, sort=False)


# ─────────────────────────────────────────────────────────────────────────────
# 섹션 빌더 — Descriptive
# ─────────────────────────────────────────────────────────────────────────────
DESC_COLS = [
    "section", "analysis_cohort", "cohort_label",
    "force_g", "group", "group_detail",
    "n_subjects", "n_observations",
    "mean", "sd", "median", "q1", "q3", "iqr", "whisker_lo", "whisker_hi",
]

def build_desc_fig2(desc_df):
    rows = []
    for _, r in desc_df.iterrows():
        if r["n_subjects"] == 0 or pd.isna(r.get("median")):
            continue
        rows.append({
            "section":        "descriptive",
            "analysis_cohort": r["analysis_cohort"],
            "cohort_label":   r["cohort_label"],
            "force_g":        r["force_g"],
            "group":          r["condition"],
            "group_detail":   r["condition_raw"],
            "n_subjects":     int(r["n_subjects"]),
            "n_observations": int(r["n_trials"]),
            "mean":           r["mean"],
            "sd":             r["sd"],
            "median":         r["median"],
            "q1":             r["q1"],
            "q3":             r["q3"],
            "iqr":            r["iqr"],
            "whisker_lo":     r["whisker_lo"],
            "whisker_hi":     r["whisker_hi"],
        })
    return pd.DataFrame(rows)[DESC_COLS]


def build_desc_fig3(desc_df):
    rows = []
    for _, r in desc_df.iterrows():
        if r["n_subjects"] == 0 or pd.isna(r.get("median")):
            continue
        rows.append({
            "section":        "descriptive",
            "analysis_cohort": r["analysis_cohort"],
            "cohort_label":   r["cohort_label"],
            "force_g":        r["force_g"],
            "group":          r["source"],
            "group_detail":   r.get("source_raw", ""),
            "n_subjects":     int(r["n_subjects"]),
            "n_observations": int(r["n_trials"]),
            "mean":           r["mean"],
            "sd":             r["sd"],
            "median":         r["median"],
            "q1":             r["q1"],
            "q3":             r["q3"],
            "iqr":            r["iqr"],
            "whisker_lo":     r["whisker_lo"],
            "whisker_hi":     r["whisker_hi"],
        })
    return pd.DataFrame(rows)[DESC_COLS]


def build_desc_figC(desc_df):
    rows = []
    for _, r in desc_df.iterrows():
        if r["n_subjects"] == 0 or pd.isna(r.get("median")):
            continue
        rows.append({
            "section":        "descriptive",
            "analysis_cohort": r["analysis_cohort"],
            "cohort_label":   r["cohort_label"],
            "force_g":        r["force_g"],
            "group":          r["group_label"],
            "group_detail":   r["pooling_method"],
            "n_subjects":     int(r["n_subjects"]),
            "n_observations": int(r["n_observations"]),
            "mean":           r["mean"],
            "sd":             r["sd"],
            "median":         r["median"],
            "q1":             r["q1"],
            "q3":             r["q3"],
            "iqr":            r["iqr"],
            "whisker_lo":     r["whisker_lo"],
            "whisker_hi":     r["whisker_hi"],
        })
    return pd.DataFrame(rows)[DESC_COLS]


# ─────────────────────────────────────────────────────────────────────────────
# 섹션 빌더 — Inferential
# ─────────────────────────────────────────────────────────────────────────────
INFER_COLS = [
    "section", "analysis_cohort", "cohort_label",
    "force_g", "comparison", "statistical_test",
    "reference_group", "test_group", "pooling_method",
    "n_reference_subjects", "n_test_subjects", "n_paired_subjects",
    "n_reference_observations", "n_test_observations",
    "mean_reference", "mean_test", "mean_difference",
    "lme_coef", "lme_95CI_lo", "lme_95CI_hi",
    "p_value", "significance",
    "cohens_d", "hedges_g", "cohens_d_95CI_lo", "cohens_d_95CI_hi",
    "effect_size_method",
    "shown_on_figure",
]

def build_infer(comp_df):
    rows = []
    for _, r in comp_df.iterrows():
        if pd.isna(r.get("p_value")) and r.get("n_ref_observations", 0) == 0:
            continue
        rows.append({
            "section":                  "inferential",
            "analysis_cohort":          r["analysis_cohort"],
            "cohort_label":             cohort_label(r["analysis_cohort"]),
            "force_g":                  r["force_g"],
            "comparison":               r.get("comparison", ""),
            "statistical_test":         r.get("test", ""),
            "reference_group":          r.get("reference", ""),
            "test_group":               r.get("test_group", ""),
            "pooling_method":           r.get("pooling_method", ""),
            "n_reference_subjects":     r.get("n_ref_subjects", np.nan),
            "n_test_subjects":          r.get("n_test_subjects", np.nan),
            "n_paired_subjects":        r.get("n_paired_subjects", np.nan),
            "n_reference_observations": r.get("n_ref_observations", np.nan),
            "n_test_observations":      r.get("n_test_observations", np.nan),
            "mean_reference":           r.get("mean_ref", np.nan),
            "mean_test":                r.get("mean_test", np.nan),
            "mean_difference":          r.get("mean_diff", np.nan),
            "lme_coef":                 r.get("lme_coef", np.nan),
            "lme_95CI_lo":              r.get("lme_ci_lo", np.nan),
            "lme_95CI_hi":              r.get("lme_ci_hi", np.nan),
            "p_value":                  r.get("p_value", np.nan),
            "significance":             r.get("sig_star", ""),
            "cohens_d":                 r.get("cohens_d", np.nan),
            "hedges_g":                 r.get("hedges_g", np.nan),
            "cohens_d_95CI_lo":         r.get("cohens_d_ci_lo", np.nan),
            "cohens_d_95CI_hi":         r.get("cohens_d_ci_hi", np.nan),
            "effect_size_method":       r.get("effect_size_method", ""),
            "shown_on_figure":          r.get("shown_on_figure", False),
        })
    df = pd.DataFrame(rows)
    for c in INFER_COLS:
        if c not in df.columns:
            df[c] = np.nan
    return df[INFER_COLS]


# ─────────────────────────────────────────────────────────────────────────────
# 섹션 빌더 — Subject-level data
# ─────────────────────────────────────────────────────────────────────────────
SUBJ_COLS = [
    "section", "subject_id", "protocol", "marker",
    "force_g", "group", "area",
    "n_trials", "mean_accuracy_pct",
]

def build_subj_fig2():
    """Fig2: per-subject mean per Condition × Force."""
    rows = []
    for (subj, cond, force), grp in df_raw.groupby([sub_col, "Condition", "Force_Val"]):
        if cond not in ["In-air", "On-touch (Mid)"]:
            continue
        vals = grp["Score"].dropna()
        if len(vals) == 0:
            continue
        rows.append({
            "section":           "subject_data",
            "subject_id":        subj,
            "protocol":          "partial" if subj in partial_set else "full",
            "marker":            "△ (partial)" if subj in partial_set else "● (full)",
            "force_g":           force,
            "group":             atd.FIG2_COND_LABELS.get(cond, cond),
            "area":              "all areas pooled",
            "n_trials":          len(vals),
            "mean_accuracy_pct": round(float(vals.mean()), 4),
        })
    return pd.DataFrame(rows)[SUBJ_COLS]


def build_subj_fig3():
    """Fig3: per-subject mean for On-touch (Mid) × Force (Periungual side)."""
    rows = []
    df_peri = df_raw[df_raw["Condition"] == "On-touch (Mid)"]
    for (subj, force), grp in df_peri.groupby([sub_col, "Force_Val"]):
        vals = grp["Score"].dropna()
        if len(vals) == 0:
            continue
        rows.append({
            "section":           "subject_data",
            "subject_id":        subj,
            "protocol":          "partial" if subj in partial_set else "full",
            "marker":            "△ (partial)" if subj in partial_set else "● (full)",
            "force_g":           force,
            "group":             "Periungual: On-touch (this study)",
            "area":              "all areas pooled",
            "n_trials":          len(vals),
            "mean_accuracy_pct": round(float(vals.mean()), 4),
        })
    return pd.DataFrame(rows)[SUBJ_COLS]


def build_subj_figC():
    """FigC: per-subject × per-area mean → On-nail/Off-nail."""
    df_areas = df_raw[df_raw["Area"].isin(POOL_MAP.keys())].copy()
    rows = []
    for (subj, area, force), grp in df_areas.groupby([sub_col, "Area", "Force_Val"]):
        if force in EXCLUDE_FORCE:
            continue
        vals = grp["Score"].dropna()
        if len(vals) == 0:
            continue
        rows.append({
            "section":           "subject_data",
            "subject_id":        subj,
            "protocol":          "partial" if subj in partial_set else "full",
            "marker":            "△ (partial)" if subj in partial_set else "● (full)",
            "force_g":           force,
            "group":             POOL_MAP[area],
            "area":              area,
            "n_trials":          len(vals),
            "mean_accuracy_pct": round(float(vals.mean()), 4),
        })
    return pd.DataFrame(rows)[SUBJ_COLS]


# ─────────────────────────────────────────────────────────────────────────────
# 섹션 빌더 — Sensitivity analysis
# ─────────────────────────────────────────────────────────────────────────────
SENS_COLS = [
    "section", "force_g", "comparison",
    "sensitivity_analysis", "analysis_cohort",
    "pooling_method_primary", "pooling_method_sensitivity",
    "n_subjects_primary", "n_subjects_sensitivity",
    "p_primary", "significance_primary",
    "p_sensitivity", "significance_sensitivity",
    "lme_coef_primary", "lme_coef_sensitivity",
    "cohens_d_primary", "cohens_d_sensitivity",
    "robustness_verdict",
]

def build_sens(fig_name):
    sub = sens[sens["figure"] == fig_name].copy()
    if sub.empty:
        return pd.DataFrame(columns=SENS_COLS)
    sub["section"] = "sensitivity"
    rename = {
        "sig_primary":     "significance_primary",
        "sig_sensitivity": "significance_sensitivity",
        "n_subjects_primary":     "n_subjects_primary",
        "n_subjects_sensitivity": "n_subjects_sensitivity",
    }
    sub = sub.rename(columns=rename)
    for c in SENS_COLS:
        if c not in sub.columns:
            sub[c] = np.nan
    return sub[SENS_COLS].reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Kao 원본 데이터 (Fig3 전용)
# ─────────────────────────────────────────────────────────────────────────────
KAO_COLS = [
    "section", "subject_id", "force_g", "group", "mean_accuracy_pct",
]

def build_kao_raw():
    rows = []
    for force, vals in atd.KAO_PAINT_RAW.items():
        for pid, v in enumerate(vals):
            rows.append({
                "section":           "kao_raw",
                "subject_id":        f"KP{pid+1}",
                "force_g":           float(force),
                "group":             "Fingerpad (Kao et al. 2022)",
                "mean_accuracy_pct": float(v),
            })
    return pd.DataFrame(rows)[KAO_COLS]


# ─────────────────────────────────────────────────────────────────────────────
# 파일 저장 헬퍼
# ─────────────────────────────────────────────────────────────────────────────
def save(df, filename):
    path = os.path.join(OUT_DIR, filename)
    df.to_csv(path, index=False, float_format="%.6f")
    size_kb = os.path.getsize(path) // 1024
    print(f"  → {filename}  ({len(df)} rows, {size_kb} KB)")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Fig2: On-touch vs In-air
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Fig2] On-touch vs In-air …")

fig2 = concat_sections([
    build_desc_fig2(desc2),
    build_infer(comp2),
    build_subj_fig2(),
    build_sens("Fig2_ontouch_vs_inair(final)"),
])
save(fig2, "Fig2_ontouch_vs_inair(final).csv")


# ─────────────────────────────────────────────────────────────────────────────
# Fig3: Fingerpad (Kao) vs Periungual On-touch
# ─────────────────────────────────────────────────────────────────────────────
print("[Fig3] Fingerpad vs Periungual On-touch …")

fig3 = concat_sections([
    build_desc_fig3(desc3),
    build_infer(comp3),
    build_subj_fig3(),
    build_kao_raw(),
    build_sens("Fig3_future_full_kao(final)"),
])
save(fig3, "Fig3_future_full_kao(final).csv")


# ─────────────────────────────────────────────────────────────────────────────
# FigC: On-nail vs Off-nail (pooled)
# ─────────────────────────────────────────────────────────────────────────────
print("[FigC] On-nail vs Off-nail …")

# compC: subject×area rows만 (trial-pooled sensitivity는 sensitivity 섹션에 포함)
compC_area = compC[compC["pooling_method"].str.contains("subject×area", na=False)]

figC = concat_sections([
    build_desc_figC(descC),
    build_infer(compC_area),
    build_subj_figC(),
    build_sens("onnail_vs_offnail_pooled(final)"),
])
save(figC, "onnail_vs_offnail_pooled(final).csv")


# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("완료! 출력 폴더:", OUT_DIR)
print("="*60)
print("""
각 CSV 파일의 'section' 컬럼 값:
  descriptive   — 기술통계 (mean, SD, median, Q1/Q3, IQR, whiskers, n)
                  ※ primary + sensitivity 코호트 모두 포함
  inferential   — 추론통계 (p-value, LME coef+95%CI, Cohen's d, Hedges' g)
                  ※ primary + sensitivity 코호트 모두 포함
  subject_data  — 피험자별 개별 데이터 포인트 (mean_accuracy_pct)
  sensitivity   — Sensitivity robustness 요약 (primary vs sensitivity 비교)
  kao_raw       — (Fig3 전용) Kao et al. 2022 참가자 5명 개별 데이터
""")
