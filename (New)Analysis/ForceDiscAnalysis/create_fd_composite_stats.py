"""
create_fd_composite_stats.py
============================
fd_composite_grid_2col(Final).png 의 모든 패널 통계를 추출하여
panel 순서대로 하나의 CSV로 저장.

패널 순서 (1,1)→(2,1)→(3,1)→(1,2)→(2,2)→(3,2):
  (1,1) = 2AFC           Low  band  (0.4–1, 0.6–1, 1–1.4, 1–2 g)
  (2,1) = SAME/DIFFERENT Low  band
  (3,1) = On-nail vs Off-nail  Low  band
  (1,2) = 2AFC           High band  (10–26, 15–26, 26–60 mN)
  (2,2) = SAME/DIFFERENT High band
  (3,2) = On-nail vs Off-nail  High band

각 패널에 포함되는 내용:
  section = descriptive   — 기술통계 (mean±SD, median, Q1/Q3, IQR, whisker, n)
  section = inferential   — 추론통계 (GEE p-value, pairwise contrasts)
  section = subject_data  — 피험자별 개별 accuracy (scatter dots)
"""

import os, sys, importlib.util, warnings, itertools
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── 경로 ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR    = os.path.join(SCRIPT_DIR, "Final")
os.makedirs(OUT_DIR, exist_ok=True)

OUT_CSV = os.path.join(OUT_DIR, "fd_composite_grid_2col(Final).csv")

# 기존 CSV 경로
GEE_DIR = os.path.join(SCRIPT_DIR, "Output", "Stats(GEE)")
SD_DIR  = os.path.join(SCRIPT_DIR, "Output", "SameDiff_GEE")
SD_SUBJ = os.path.join(SD_DIR, "per_subject")

# SameDiffGee.py 동적 로딩 (region pvals 계산용)
SAMEDIFF_PY = os.path.join(SCRIPT_DIR, "Stat Files (python)", "SameDiffGee.py")
sys.path.insert(0, os.path.join(SCRIPT_DIR, "Stat Files (python)"))


# ── 헬퍼 ─────────────────────────────────────────────────────────────────────
def boxplot_stats(vals):
    """matplotlib boxplot 1.5×IQR whiskers 기준 기술통계."""
    arr = np.asarray(vals, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return dict(n=0, mean=np.nan, sd=np.nan, median=np.nan,
                    q1=np.nan, q3=np.nan, iqr=np.nan,
                    whisker_lo=np.nan, whisker_hi=np.nan)
    q1, med, q3 = np.percentile(arr, [25, 50, 75])
    iqr = q3 - q1
    lo_f, hi_f = q1 - 1.5*iqr, q3 + 1.5*iqr
    in_r = arr[(arr >= lo_f) & (arr <= hi_f)]
    wlo  = float(in_r.min()) if len(in_r) else float(arr.min())
    whi  = float(in_r.max()) if len(in_r) else float(arr.max())
    return dict(
        n=len(arr),
        mean=float(arr.mean()),
        sd=float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        median=float(med),
        q1=float(q1), q3=float(q3), iqr=float(iqr),
        whisker_lo=wlo, whisker_hi=whi,
    )


def sig_star(p):
    if pd.isna(p): return ""
    if p < 0.001:  return "***"
    if p < 0.01:   return "**"
    if p < 0.05:   return "*"
    return "n.s."


def panel_label(col, row):
    """(col, row) → 논문 패널 표기."""
    labels = {
        (1,1): "(1,1) 2AFC — Low band",
        (2,1): "(2,1) SAME/DIFFERENT — Low band",
        (3,1): "(3,1) On-nail vs Off-nail — Low band",
        (1,2): "(1,2) 2AFC — High band",
        (2,2): "(2,2) SAME/DIFFERENT — High band",
        (3,2): "(3,2) On-nail vs Off-nail — High band",
    }
    return labels[(col, row)]


# ── 공통 row 생성기 ───────────────────────────────────────────────────────────
def make_base(panel_col, panel_row, section, force_pair, group=""):
    return {
        "panel_order": f"({panel_col},{panel_row})",
        "panel_label": panel_label(panel_col, panel_row),
        "panel_col":   panel_col,
        "panel_row":   panel_row,
        "section":     section,
        "force_pair":  force_pair,
        "group":       group,
    }


# ─────────────────────────────────────────────────────────────────────────────
# COLUMN 1 & 2 — GEE / SD 공통 함수
# ─────────────────────────────────────────────────────────────────────────────
gee_group_summary  = pd.read_csv(os.path.join(GEE_DIR, "gee_pairwise_group_summary.csv"))
gee_contrasts      = pd.read_csv(os.path.join(GEE_DIR, "gee_pairwise_contrasts.csv"))
gee_subj           = pd.read_csv(os.path.join(GEE_DIR, "gee_pairwise_subject_accuracy.csv"))

sd_subj_low  = pd.read_csv(os.path.join(SD_SUBJ, "accuracy_by_subject_low.csv"))
sd_subj_high = pd.read_csv(os.path.join(SD_SUBJ, "accuracy_by_subject_high.csv"))

BAND_PAIRS = {
    "Low":  {"pairs": ["0.4–1", "0.6–1", "1–1.4", "1–2"],  "row": 1},
    "High": {"pairs": ["10–26", "15–26", "26–60"],           "row": 2},
}


def build_col1_rows(band_label, panel_row):
    """Column 1: 2AFC — GEE pairwise."""
    pairs     = BAND_PAIRS[band_label]["pairs"]
    rows_out  = []
    band_summ = gee_group_summary[gee_group_summary["band"] == band_label]
    band_subj = gee_subj[gee_subj["band"] == band_label]
    band_cont = gee_contrasts[gee_contrasts["band"] == band_label]

    # ── Descriptive (per pair) ───────────────────────────────────────────────
    for pair in pairs:
        summ = band_summ[band_summ["force_pair_g"] == pair]
        if summ.empty:
            continue
        r = summ.iloc[0]
        subj_vals = band_subj.loc[band_subj["pair_label"] == pair, "accuracy"].values * 100
        bs = boxplot_stats(subj_vals)
        base = make_base(1, panel_row, "descriptive", pair)
        rows_out.append({**base,
            "n_subjects":     int(r["n_subjects"]),
            "mean_pct":       round(r["mean_accuracy"] * 100, 4),
            "sd_pct":         round(r["sd"] * 100, 4),
            "sem_pct":        round(r["sem"] * 100, 4),
            "median_pct":     round(bs["median"], 4),
            "q1_pct":         round(bs["q1"], 4),
            "q3_pct":         round(bs["q3"], 4),
            "iqr_pct":        round(bs["iqr"], 4),
            "whisker_lo_pct": round(bs["whisker_lo"], 4),
            "whisker_hi_pct": round(bs["whisker_hi"], 4),
            "min_pct":        round(r["min"] * 100, 4),
            "max_pct":        round(r["max"] * 100, 4),
        })

    # ── Inferential (pairwise GEE contrasts) ────────────────────────────────
    for _, r in band_cont.iterrows():
        base = make_base(1, panel_row, "inferential", f"{r['pair_a']} vs {r['pair_b']}")
        rows_out.append({**base,
            "statistical_test": "GEE (binomial, subject clustering)",
            "contrast":         f"{r['pair_b']} − {r['pair_a']}",
            "p_value":          r["p_value"],
            "significance":     r["sig_label"],
        })

    # ── Subject data ─────────────────────────────────────────────────────────
    for _, r in band_subj[band_subj["pair_label"].isin(pairs)].iterrows():
        base = make_base(1, panel_row, "subject_data", r["pair_label"], group="all pairs")
        rows_out.append({**base,
            "subject_id":        r["Subject"],
            "accuracy_pct":      round(r["accuracy"] * 100, 4),
        })

    return rows_out


def build_col2_rows(band_label, panel_row):
    """Column 2: SAME/DIFFERENT — SD GEE accuracy."""
    pairs = BAND_PAIRS[band_label]["pairs"]
    sd_subj = sd_subj_low if band_label == "Low" else sd_subj_high
    rows_out = []

    # ── Descriptive (per pair) ───────────────────────────────────────────────
    for pair in pairs:
        sub = sd_subj[sd_subj["pair_label"] == pair]
        if sub.empty:
            continue
        vals = sub["accuracy_pct"].values
        bs = boxplot_stats(vals)
        base = make_base(2, panel_row, "descriptive", pair)
        rows_out.append({**base,
            "n_subjects":            int(len(sub)),
            "mean_pct":              round(float(np.mean(vals)), 4),
            "sd_pct":                round(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0, 4),
            "median_pct":            round(bs["median"], 4),
            "q1_pct":                round(bs["q1"], 4),
            "q3_pct":                round(bs["q3"], 4),
            "iqr_pct":               round(bs["iqr"], 4),
            "whisker_lo_pct":        round(bs["whisker_lo"], 4),
            "whisker_hi_pct":        round(bs["whisker_hi"], 4),
            "mean_same_trial_pct":   round(float(sub["same_accuracy_pct"].mean()), 4),
            "mean_diff_trial_pct":   round(float(sub["different_accuracy_pct"].mean()), 4),
        })

    # ── Inferential (pairwise GEE contrasts from SameDiffGee module) ────────
    sd_spec   = sd_specs[band_label]
    pairwise_pvals = sd_spec["pairwise_pvals"]   # dict {(p1,p2): p_value}
    for (p1, p2) in itertools.combinations(pairs, 2):
        pval = pairwise_pvals.get((p1, p2),
               pairwise_pvals.get((p2, p1), np.nan))
        base = make_base(2, panel_row, "inferential", f"{p1} vs {p2}")
        rows_out.append({**base,
            "statistical_test": "GEE (binomial, subject clustering)",
            "contrast":         f"{p2} − {p1}",
            "p_value":          pval,
            "significance":     sig_star(pval),
        })

    # ── Subject data ─────────────────────────────────────────────────────────
    for _, r in sd_subj[sd_subj["pair_label"].isin(pairs)].iterrows():
        base = make_base(2, panel_row, "subject_data", r["pair_label"], group="overall")
        rows_out.append({**base,
            "subject_id":              r["Subject"],
            "accuracy_pct":            round(r["accuracy_pct"], 4),
            "same_trial_accuracy_pct": round(r["same_accuracy_pct"], 4),
            "diff_trial_accuracy_pct": round(r["different_accuracy_pct"], 4),
            "n_trials":                int(r["n_trials"]),
        })

    return rows_out


# ─────────────────────────────────────────────────────────────────────────────
# COLUMN 3 — On-nail (C+D) vs Off-nail (A+F) per SD band
# subj_acc_reg + region_pvals를 SameDiffGee 모듈에서 동적으로 추출
# ─────────────────────────────────────────────────────────────────────────────
print("SameDiffGee 모듈 로딩 및 On-nail 통계 계산 중 …")
spec = importlib.util.spec_from_file_location("samediff_gee", SAMEDIFF_PY)
sd_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sd_mod)

sd_specs = {}
for band_label, cfg in sd_mod.BAND_CONFIG.items():
    df_band = sd_mod.df[sd_mod.df["band"] == band_label].copy()
    if df_band.empty:
        continue
    pair_order = sd_mod.fix_order(
        cfg["pair_order"],
        df_band["pair_label"].unique().tolist(),
    )
    sd_specs[band_label] = sd_mod.build_band_spec(
        df_band, band_label, pair_order, cfg["title_ref"],
    )
print("  → 완료")


def build_col3_rows(band_label, panel_row):
    """Column 3: On-nail vs Off-nail (SD region pooled)."""
    spec  = sd_specs[band_label]
    pairs = spec["pair_order"]
    subj_acc_reg = spec["subj_acc_reg"]   # Subject, pair_label, region_group, accuracy
    region_pvals = spec["region_pvals"]   # dict {pair: p_value}
    rows_out = []

    for pair in pairs:
        for grp in ["On-nail", "Off-nail"]:
            sub = subj_acc_reg[
                (subj_acc_reg["pair_label"] == pair) &
                (subj_acc_reg["region_group"] == grp)
            ]
            vals = sub["accuracy"].values * 100
            if len(vals) == 0:
                continue
            bs = boxplot_stats(vals)
            base = make_base(3, panel_row, "descriptive", pair, group=grp)
            rows_out.append({**base,
                "n_subjects":     int(len(sub)),
                "mean_pct":       round(float(np.mean(vals)), 4),
                "sd_pct":         round(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0, 4),
                "median_pct":     round(bs["median"], 4),
                "q1_pct":         round(bs["q1"], 4),
                "q3_pct":         round(bs["q3"], 4),
                "iqr_pct":        round(bs["iqr"], 4),
                "whisker_lo_pct": round(bs["whisker_lo"], 4),
                "whisker_hi_pct": round(bs["whisker_hi"], 4),
            })

        # ── Inferential (On-nail vs Off-nail per pair) ───────────────────────
        pval = region_pvals.get(pair, np.nan)
        base = make_base(3, panel_row, "inferential",
                         pair, group="On-nail vs Off-nail")
        rows_out.append({**base,
            "statistical_test": "GEE (binomial, subject clustering)",
            "contrast":         "On-nail − Off-nail",
            "p_value":          pval,
            "significance":     sig_star(pval),
        })

        # ── Subject data ─────────────────────────────────────────────────────
        for grp in ["On-nail", "Off-nail"]:
            sub = subj_acc_reg[
                (subj_acc_reg["pair_label"] == pair) &
                (subj_acc_reg["region_group"] == grp)
            ]
            for _, r in sub.iterrows():
                base = make_base(3, panel_row, "subject_data", pair, group=grp)
                rows_out.append({**base,
                    "subject_id":   r["Subject"],
                    "accuracy_pct": round(r["accuracy"] * 100, 4),
                })

    return rows_out


# ─────────────────────────────────────────────────────────────────────────────
# 전체 조합 → DataFrame
# ─────────────────────────────────────────────────────────────────────────────
print("\n패널 통계 조합 중 …")
all_rows = []

for band_label, info in [("Low", 1), ("High", 2)]:
    panel_row = info
    all_rows += build_col1_rows(band_label, panel_row)
    all_rows += build_col2_rows(band_label, panel_row)
    all_rows += build_col3_rows(band_label, panel_row)

df_out = pd.DataFrame(all_rows)

# 패널 순서 정렬: (col,row) → (1,1),(2,1),(3,1),(1,2),(2,2),(3,2)
sort_key = df_out["panel_col"] * 10 + df_out["panel_row"]
df_out = df_out.iloc[sort_key.argsort(kind="stable")].reset_index(drop=True)

df_out.to_csv(OUT_CSV, index=False, float_format="%.6f")
print(f"\n완료 → {OUT_CSV}  ({len(df_out)} rows)")
print("""
'section' 컬럼 값:
  descriptive  — 기술통계 (mean, SD, median, Q1/Q3, IQR, whisker, n)
  inferential  — GEE p-value, significance (***/**/*/ n.s.)
  subject_data — 피험자별 개별 accuracy (scatter dot 1개 = 1 row)
""")
