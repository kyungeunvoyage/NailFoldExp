"""export_final_figure_stats.py
Export descriptive statistics, inferential tests, and sensitivity analyses
for the three final ATD figures.

Analysis cohorts:
  primary_all              — matches final figures (n=45 at 0.16/0.6 g; n=15 at 0.4 g)
  sensitivity_full_n30     — full protocol only (P1–P60, n=30)
  sensitivity_partial_n15  — partial protocol only (P61–P75, n=15)
  sensitivity_trial_pooled_n30 — on-nail only: trial-pooled subject means, n=30

Outputs:
  atd_c1_outputs/stats/final_figures_stats.xlsx  (multi-tab workbook)
  atd_c1_outputs/stats/final_figures_sensitivity_summary.csv
  per-figure *_descriptives.csv / *_comparisons.csv
"""

import os
import importlib.util

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ATD_C1_PATH = os.path.join(SCRIPT_DIR, "(Final)ATD_C1_Fig(Anika).py")
OUT_STATS_C1 = os.path.join(SCRIPT_DIR, "atd_c1_outputs", "stats")
OUT_STATS_AGG = os.path.join(SCRIPT_DIR, "figures", "stats")
os.makedirs(OUT_STATS_C1, exist_ok=True)
os.makedirs(OUT_STATS_AGG, exist_ok=True)


def _load_atd_c1():
    spec = importlib.util.spec_from_file_location("atd_c1", ATD_C1_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _boxplot_whiskers(values):
    """Match matplotlib boxplot (showfliers=False): whiskers at 1.5×IQR."""
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return dict(n=0, q1=np.nan, median=np.nan, q3=np.nan, iqr=np.nan,
                    whisker_lo=np.nan, whisker_hi=np.nan, mean=np.nan, sd=np.nan)
    q1, med, q3 = np.percentile(arr, [25, 50, 75])
    iqr = q3 - q1
    lo_fence = q1 - 1.5 * iqr
    hi_fence = q3 + 1.5 * iqr
    in_range = arr[(arr >= lo_fence) & (arr <= hi_fence)]
    if len(in_range) == 0:
        wlo, whi = float(arr.min()), float(arr.max())
    else:
        wlo, whi = float(in_range.min()), float(in_range.max())
    return dict(
        n=len(arr),
        q1=float(q1),
        median=float(med),
        q3=float(q3),
        iqr=float(iqr),
        whisker_lo=wlo,
        whisker_hi=whi,
        mean=float(arr.mean()),
        sd=float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
    )


def _star_from_p(p):
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def _hedges_g_paired(d, n):
    if n < 2 or pd.isna(d):
        return np.nan
    correction = 1.0 - 3.0 / (4.0 * (n - 1) - 1.0)
    return float(d * correction)


def _hedges_g_independent(d, n1, n2):
    if n1 < 2 or n2 < 2 or pd.isna(d):
        return np.nan
    df = n1 + n2 - 2
    correction = 1.0 - 3.0 / (4.0 * df - 1.0)
    return float(d * correction)


def _cohens_d_paired(ref_vals, test_vals):
    ref = np.asarray(ref_vals, dtype=float)
    test = np.asarray(test_vals, dtype=float)
    if len(ref) != len(test) or len(ref) < 2:
        return np.nan, np.nan, np.nan
    diff = test - ref
    sd = float(np.std(diff, ddof=1))
    if sd == 0:
        return float(np.mean(diff)), 0.0, np.nan
    d = float(np.mean(diff) / sd)
    return float(np.mean(diff)), d, _hedges_g_paired(d, len(diff))


def _cohens_d_independent(ref_vals, test_vals):
    ref = np.asarray(ref_vals, dtype=float)
    test = np.asarray(test_vals, dtype=float)
    ref = ref[~np.isnan(ref)]
    test = test[~np.isnan(test)]
    n1, n2 = len(ref), len(test)
    if n1 < 2 or n2 < 2:
        return np.nan, np.nan, np.nan
    mean_diff = float(np.mean(test) - np.mean(ref))
    var1 = float(np.var(ref, ddof=1))
    var2 = float(np.var(test, ddof=1))
    pooled = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled == 0:
        return mean_diff, 0.0, np.nan
    d = float(mean_diff / pooled)
    return mean_diff, d, _hedges_g_independent(d, n1, n2)


def _bootstrap_d_ci(ref_vals, test_vals, paired=True, n_boot=5000, seed=0):
    ref = np.asarray(ref_vals, dtype=float)
    test = np.asarray(test_vals, dtype=float)
    if paired and len(ref) != len(test):
        return np.nan, np.nan
    n = len(ref) if paired else min(len(ref), len(test))
    if n < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    ds = []
    for _ in range(n_boot):
        if paired:
            idx = rng.integers(0, n, size=n)
            _, d, _ = _cohens_d_paired(ref[idx], test[idx])
        else:
            idx1 = rng.integers(0, len(ref), size=len(ref))
            idx2 = rng.integers(0, len(test), size=len(test))
            _, d, _ = _cohens_d_independent(ref[idx1], test[idx2])
        if not np.isnan(d):
            ds.append(d)
    if not ds:
        return np.nan, np.nan
    lo, hi = np.percentile(ds, [2.5, 97.5])
    return float(lo), float(hi)


def _subject_condition_means(df, sub_col, score_col, force_val, ref, test):
    """Per-subject mean score in ref and test groups at one force (paired)."""
    sub = df[np.isclose(df["Force_Val"], force_val)].dropna(
        subset=[sub_col, score_col, "Group"]
    )
    sub = sub[sub["Group"].isin([ref, test])]
    wide = (
        sub.groupby([sub_col, "Group"], as_index=False)[score_col]
        .mean()
        .pivot(index=sub_col, columns="Group", values=score_col)
    )
    if ref not in wide.columns or test not in wide.columns:
        return None
    paired = wide[[ref, test]].dropna()
    if paired.empty:
        return None
    return paired


def _effect_size_paired_subject_means(df, sub_col, score_col, force_val, ref, test):
    paired = _subject_condition_means(df, sub_col, score_col, force_val, ref, test)
    if paired is None or len(paired) < 2:
        return None
    ref_vals = paired[ref].values
    test_vals = paired[test].values
    mean_diff, d, g = _cohens_d_paired(ref_vals, test_vals)
    lo, hi = _bootstrap_d_ci(ref_vals, test_vals, paired=True)
    return {
        "n_paired_subjects": len(paired),
        "n_ref_subjects": len(paired),
        "n_test_subjects": len(paired),
        "mean_ref": float(np.mean(ref_vals)),
        "mean_test": float(np.mean(test_vals)),
        "mean_diff": mean_diff,
        "cohens_d": d,
        "hedges_g": g,
        "cohens_d_ci_lo": lo,
        "cohens_d_ci_hi": hi,
        "effect_size_method": "paired Cohen's d on subject-level means (test − ref)",
    }


def _subject_onnail_means(df, sub_col, score_col, force_val, pool_map):
    """Per-subject mean accuracy pooled across On-nail vs Off-nail areas."""
    areas = list(pool_map.keys())
    sub = df[df["Area"].isin(areas)].dropna(subset=[sub_col, "Area", score_col])
    sub = sub[np.isclose(sub["Force_Val"], force_val)]
    if sub.empty:
        return None
    sub = sub.copy()
    sub["Group"] = sub["Area"].map(pool_map)
    wide = (
        sub.groupby([sub_col, "Group"], as_index=False)[score_col]
        .mean()
        .pivot(index=sub_col, columns="Group", values=score_col)
    )
    ref, test = "Off-nail", "On-nail"
    if ref not in wide.columns or test not in wide.columns:
        return None
    paired = wide[[ref, test]].dropna()
    if paired.empty:
        return None
    return paired


def _observation_counts(df, sub_col, score_col, force_val, ref, test):
    sub = df.dropna(subset=[sub_col, score_col, "Group"])
    if force_val is not None and "Force_Val" in sub.columns:
        sub = sub[np.isclose(sub["Force_Val"], force_val)]
    sub = sub[sub["Group"].isin([ref, test])]
    return {
        "n_ref_observations": int((sub["Group"] == ref).sum()),
        "n_test_observations": int((sub["Group"] == test).sum()),
        "n_ref_subjects": int(sub.loc[sub["Group"] == ref, sub_col].nunique()),
        "n_test_subjects": int(sub.loc[sub["Group"] == test, sub_col].nunique()),
    }


def _lme_condition(df, sub_col, score_col, ref, test, force_val=None):
    sub = df.dropna(subset=[sub_col, score_col, "Group"])
    if force_val is not None:
        sub = sub[np.isclose(sub["Force_Val"], force_val)]
    sub = sub[sub["Group"].isin([ref, test])]
    if sub.empty or sub["Group"].nunique() < 2 or sub[sub_col].nunique() < 2:
        return None
    formula = f"{score_col} ~ C(Group, Treatment(reference='{ref}'))"
    try:
        res = smf.mixedlm(formula, sub, groups=sub[sub_col]).fit(reml=True)
        col = f"C(Group, Treatment(reference='{ref}'))[T.{test}]"
        if col not in res.params.index:
            return None
        ci = res.conf_int().loc[col]
        return {
            "lme_coef": float(res.params[col]),
            "lme_ci_lo": float(ci[0]),
            "lme_ci_hi": float(ci[1]),
            "p_value": float(res.pvalues[col]),
            "sig_star": _star_from_p(float(res.pvalues[col])),
        }
    except Exception:
        return None


def _subject_area_pool(df_in, sub_col, score_col, area_group_map, force_val):
    areas = list(area_group_map.keys())
    sub = df_in[df_in["Area"].isin(areas)].dropna(subset=[sub_col, "Area", score_col])
    sub = sub[np.isclose(sub["Force_Val"], force_val)]
    if sub.empty:
        return pd.DataFrame(columns=[sub_col, "Area", "Group", "accuracy"])
    agg = (
        sub.groupby([sub_col, "Area"], as_index=False)[score_col]
        .mean()
        .rename(columns={score_col: "accuracy"})
    )
    agg["Group"] = agg["Area"].map(area_group_map)
    return agg


def _filter_cohort(df, sub_col, partial_subj, cohort):
    if cohort == "primary_all":
        return df
    if cohort in ("sensitivity_full_n30", "sensitivity_trial_pooled_n30"):
        return df[~df[sub_col].isin(partial_subj)].copy()
    if cohort == "sensitivity_partial_n15":
        return df[df[sub_col].isin(partial_subj)].copy()
    raise ValueError(f"Unknown cohort: {cohort}")


COHORT_LABELS = {
    "primary_all": "Primary: all included subjects (matches final figures)",
    "sensitivity_full_n30": "Sensitivity: full protocol only (P1–P60, n=30)",
    "sensitivity_partial_n15": "Sensitivity: partial protocol only (P61–P75, n=15)",
    "sensitivity_trial_pooled_n30": (
        "Sensitivity: trial-pooled subject means, full protocol (n=30 per group)"
    ),
}


def _subject_trial_pool_group(df, sub_col, score_col, pool_map, force_val):
    """One subject-level mean per group (all area trials pooled), for trial-pooled sensitivity."""
    areas = list(pool_map.keys())
    sub = df[df["Area"].isin(areas)].dropna(subset=[sub_col, "Area", score_col])
    sub = sub[np.isclose(sub["Force_Val"], force_val)]
    if sub.empty:
        return pd.DataFrame(columns=[sub_col, "Group", "accuracy"])
    sub = sub.copy()
    sub["Group"] = sub["Area"].map(pool_map)
    return (
        sub.groupby([sub_col, "Group"], as_index=False)[score_col]
        .mean()
        .rename(columns={score_col: "accuracy"})
    )


def _empty_comparison(comp_base, cohort, note=""):
    row = {**comp_base, "analysis_cohort": cohort,
           "cohort_label": COHORT_LABELS.get(cohort, cohort),
           "p_value": np.nan, "sig_star": "", "lme_coef": np.nan,
           "lme_ci_lo": np.nan, "lme_ci_hi": np.nan,
           "n_ref_observations": 0, "n_test_observations": 0,
           "n_ref_subjects": 0, "n_test_subjects": 0, "n_paired_subjects": np.nan,
           "mean_ref": np.nan, "mean_test": np.nan, "mean_diff": np.nan,
           "cohens_d": np.nan, "hedges_g": np.nan,
           "cohens_d_ci_lo": np.nan, "cohens_d_ci_hi": np.nan,
           "effect_size_method": "", "sensitivity_note": note}
    return row


def _comparison_fig2(df, atd, fval, cohort, *, shown_on_figure=False):
    sub_col = atd.SUBJECT_COL
    ref, test = atd.FIG2_REF_CONDITION, atd.FIG2_TEST_CONDITION
    comp_base = {
        "figure": "Fig2_ontouch_vs_inair(final)",
        "force_g": fval,
        "comparison": (
            f"{atd.FIG2_COND_LABELS.get(test, test)} vs "
            f"{atd.FIG2_COND_LABELS.get(ref, ref)}"
        ),
        "test": "LME (trial-level): Score ~ Condition, random intercept ~ subject",
        "reference": ref,
        "test_group": test,
        "shown_on_figure": shown_on_figure,
        "pooling_method": "trial-level (all areas)",
    }
    lme_df = df[df["Force_Val"] == fval].copy()
    if lme_df.empty:
        return _empty_comparison(comp_base, cohort, "No data in this cohort at this force")
    lme_df["Group"] = lme_df["Condition"]
    stat = _lme_condition(lme_df, sub_col, "Score", ref, test)
    n_info = _observation_counts(lme_df, sub_col, "Score", fval, ref, test)
    es = _effect_size_paired_subject_means(lme_df, sub_col, "Score", fval, ref, test)
    comp = _empty_comparison(comp_base, cohort)
    comp.update(n_info)
    if es:
        comp.update(es)
    if stat:
        comp.update(stat)
    comp["sensitivity_note"] = ""
    return comp


def _comparison_fig3_shared(df_plot, df_kao, df_peri, atd, fval, cohort):
    ref_g, test_g = "Fingerpad", "Periungual_On-touch"
    comp_base = {
        "figure": "Fig3_future_full_kao(final)",
        "force_g": fval,
        "comparison": "Periungual: On-touch vs Fingerpad (Kao)",
        "test": "LME (trial-level): Score ~ Group, random intercept ~ participant",
        "reference": "Fingerpad (Kao et al. 2022)",
        "test_group": "Periungual: On-touch",
        "shown_on_figure": False,
        "pooling_method": "trial-level (Kao participant / peri subject trials)",
    }
    lme_df = df_plot[df_plot["Force_Val"] == fval].copy()
    if lme_df.empty or lme_df["Group"].nunique() < 2:
        return _empty_comparison(
            comp_base, cohort,
            "No periungual data in this cohort at this force" if cohort != "primary_all" else "",
        )
    stat = _lme_condition(lme_df, "Participant", "Score", ref_g, test_g)
    n_info = _observation_counts(lme_df, "Participant", "Score", fval, ref_g, test_g)
    kao_vals = (
        df_kao[df_kao["Force_Val"] == fval]
        .groupby("Participant", as_index=False)["Score"].mean()["Score"].values
    )
    peri_vals = (
        df_peri[df_peri["Force_Val"] == fval]
        .groupby(atd.SUBJECT_COL, as_index=False)["Score"].mean()["Score"].values
    )
    mean_diff, d, g = _cohens_d_independent(kao_vals, peri_vals)
    lo, hi = _bootstrap_d_ci(kao_vals, peri_vals, paired=False)
    comp = _empty_comparison(comp_base, cohort)
    comp.update(n_info)
    comp.update({
        "mean_ref": float(np.mean(kao_vals)) if len(kao_vals) else np.nan,
        "mean_test": float(np.mean(peri_vals)) if len(peri_vals) else np.nan,
        "mean_diff": mean_diff,
        "cohens_d": d,
        "hedges_g": g,
        "cohens_d_ci_lo": lo,
        "cohens_d_ci_hi": hi,
        "effect_size_method": (
            "independent Cohen's d on participant/subject means "
            "(Periungual − Fingerpad; bootstrap CI)"
        ),
        "sensitivity_note": "Kao n=5 unchanged across cohorts",
    })
    if stat:
        comp.update(stat)
    return comp


def _comparison_onnail(df, atd, fval, cohort, pool_map, *, trial_pooled=False):
    sub_col = atd.SUBJECT_COL
    ref_g, test_g = "Off-nail", "On-nail"
    comp_base = {
        "figure": "onnail_vs_offnail_pooled(final)",
        "force_g": fval,
        "comparison": "On-nail (C+D) vs Off-nail (A+F)",
        "test": "LME: accuracy ~ Group, random intercept ~ subject",
        "reference": "Off-nail (A+F)",
        "test_group": "On-nail (C+D)",
        "shown_on_figure": False,
        "pooling_method": (
            "trial-pooled subject means (1 value/subject/group)"
            if trial_pooled
            else "subject×area means (C,D,A,F separate)"
        ),
    }
    if trial_pooled:
        df_f = _subject_trial_pool_group(df, sub_col, "Score", pool_map, fval)
        score_col = "accuracy"
    else:
        df_f = _subject_area_pool(df, sub_col, "Score", pool_map, fval)
        score_col = "accuracy"

    if df_f.empty:
        return _empty_comparison(comp_base, cohort, "No data in this cohort at this force")

    stat = _lme_condition(df_f, sub_col, score_col, ref_g, test_g)
    n_info = _observation_counts(df_f, sub_col, score_col, None, ref_g, test_g)
    paired = _subject_onnail_means(df, sub_col, "Score", fval, pool_map)
    es = None
    if paired is not None and len(paired) >= 2:
        ref_vals, test_vals = paired[ref_g].values, paired[test_g].values
        mean_diff, d, g = _cohens_d_paired(ref_vals, test_vals)
        lo, hi = _bootstrap_d_ci(ref_vals, test_vals, paired=True)
        es = {
            "n_paired_subjects": len(paired),
            "mean_ref": float(np.mean(ref_vals)),
            "mean_test": float(np.mean(test_vals)),
            "mean_diff": mean_diff,
            "cohens_d": d,
            "hedges_g": g,
            "cohens_d_ci_lo": lo,
            "cohens_d_ci_hi": hi,
            "effect_size_method": (
                "paired Cohen's d on subject-level trial-pooled area means "
                "(On-nail − Off-nail; bootstrap CI)"
            ),
        }
    comp = _empty_comparison(comp_base, cohort)
    comp.update(n_info)
    if es:
        comp.update(es)
    if stat:
        comp.update(stat)
        if cohort == "primary_all" and not trial_pooled:
            comp["shown_on_figure"] = stat["p_value"] < 0.05
    comp["sensitivity_note"] = ""
    return comp


def build_sensitivity_summary(primary, sensitivity):
    """Side-by-side primary vs sensitivity_full_n30 (and trial-pooled for on-nail)."""
    sens_ids = ["sensitivity_full_n30", "sensitivity_trial_pooled_n30", "sensitivity_partial_n15"]
    rows = []
    for _, pr in primary.iterrows():
        key = (pr["figure"], pr["force_g"], pr["comparison"])
        for sid in sens_ids:
            sr_match = sensitivity[
                (sensitivity["analysis_cohort"] == sid)
                & (sensitivity["figure"] == pr["figure"])
                & (np.isclose(sensitivity["force_g"], pr["force_g"]))
                & (sensitivity["comparison"] == pr["comparison"])
            ]
            if sr_match.empty:
                continue
            sr = sr_match.iloc[0]
            if pr["figure"] == "onnail_vs_offnail_pooled(final)":
                if sid == "sensitivity_full_n30":
                    if sr.get("pooling_method") != pr.get("pooling_method"):
                        continue
                elif sid == "sensitivity_trial_pooled_n30":
                    if sr.get("pooling_method") != (
                        "trial-pooled subject means (1 value/subject/group)"
                    ):
                        continue
            p_pri, p_sen = pr.get("p_value"), sr.get("p_value")
            d_pri, d_sen = pr.get("cohens_d"), sr.get("cohens_d")
            n_sen_obs = sr.get("n_ref_observations", 0) or 0
            if pd.isna(n_sen_obs):
                n_sen_obs = 0
            no_sen_data = (
                n_sen_obs == 0
                or str(sr.get("sensitivity_note", "")).startswith("No data")
            )
            sig_pri = pd.notna(p_pri) and p_pri < 0.05
            sig_sen = pd.notna(p_sen) and p_sen < 0.05
            if no_sen_data:
                verdict = "not applicable (no data in sensitivity cohort)"
            elif pd.isna(p_pri) and pd.isna(p_sen):
                verdict = "insufficient data"
            elif sig_pri and sig_sen:
                verdict = "robust (both p < .05)"
            elif (not sig_pri) and (not sig_sen):
                verdict = "concordant non-significant"
            elif sig_pri and (not sig_sen):
                verdict = "attenuated (primary sig, sensitivity n.s.)"
            else:
                verdict = "sensitivity significant, primary n.s."
            rows.append({
                "figure": pr["figure"],
                "force_g": pr["force_g"],
                "comparison": pr["comparison"],
                "sensitivity_analysis": COHORT_LABELS.get(sid, sid),
                "analysis_cohort": sid,
                "pooling_method_primary": pr.get("pooling_method", ""),
                "pooling_method_sensitivity": sr.get("pooling_method", ""),
                "n_paired_primary": pr.get("n_paired_subjects"),
                "n_paired_sensitivity": sr.get("n_paired_subjects"),
                "n_subjects_primary": pr.get("n_ref_subjects"),
                "n_subjects_sensitivity": sr.get("n_ref_subjects"),
                "p_primary": p_pri,
                "p_sensitivity": p_sen,
                "sig_primary": _star_from_p(p_pri) if pd.notna(p_pri) else "",
                "sig_sensitivity": _star_from_p(p_sen) if pd.notna(p_sen) else "",
                "cohens_d_primary": d_pri,
                "cohens_d_sensitivity": d_sen,
                "lme_coef_primary": pr.get("lme_coef"),
                "lme_coef_sensitivity": sr.get("lme_coef"),
                "robustness_verdict": verdict,
                "sensitivity_note": sr.get("sensitivity_note", ""),
            })
    return pd.DataFrame(rows)


def export_fig2(atd):
    """Fig2: Periungual On-touch vs In-air (trial-level)."""
    sub_col = atd.SUBJECT_COL
    ref = atd.FIG2_REF_CONDITION
    test = atd.FIG2_TEST_CONDITION
    cond_list = [c for c in ["In-air", "On-touch (Mid)"] if c in atd.df_raw["Condition"].unique()]
    cohorts = ["primary_all", "sensitivity_full_n30", "sensitivity_partial_n15"]

    desc_rows, comp_rows = [], []
    for cohort in cohorts:
        df = _filter_cohort(atd.df_raw.copy(), sub_col, atd._PARTIAL_SUBJ, cohort)
        for fval in sorted(atd.df_raw["Force_Val"].unique()):
            for cond in cond_list:
                sub = df[(df["Force_Val"] == fval) & (df["Condition"] == cond)]["Score"].dropna()
                stats = _boxplot_whiskers(sub.values)
                desc_rows.append({
                    "figure": "Fig2_ontouch_vs_inair(final)",
                    "analysis_cohort": cohort,
                    "cohort_label": COHORT_LABELS[cohort],
                    "force_g": fval,
                    "condition": atd.FIG2_COND_LABELS.get(cond, cond),
                    "condition_raw": cond,
                    "n_trials": stats["n"],
                    "n_subjects": df[(df["Force_Val"] == fval) & (df["Condition"] == cond)][sub_col].nunique(),
                    **{k: stats[k] for k in ["q1", "median", "q3", "iqr", "whisker_lo", "whisker_hi", "mean", "sd"]},
                })
            shown = (
                cohort == "primary_all"
                and float(fval) not in atd.FIG2_BRACKET_EXCLUDE_FORCES
            )
            comp = _comparison_fig2(df, atd, fval, cohort, shown_on_figure=False)
            if shown and pd.notna(comp.get("p_value")) and comp["p_value"] < atd.FIG2_BRACKET_MAX_P:
                comp["shown_on_figure"] = True
            comp_rows.append(comp)

    desc = pd.DataFrame(desc_rows)
    comp = pd.DataFrame(comp_rows)
    _save_fig_stats("Fig2_ontouch_vs_inair(final)", desc, comp, OUT_STATS_C1)
    return desc, comp


def _build_fig3_frames(atd, df_peri):
    kao_rows = []
    for force, vals in atd.KAO_PAINT_RAW.items():
        for pid, v in enumerate(vals):
            kao_rows.append({
                "Force_Val": float(force),
                "Score": float(v),
                "Source": atd.KAO_LABEL,
                "Participant": f"KP{pid + 1}",
            })
    df_kao = pd.DataFrame(kao_rows)
    df_peri = df_peri.copy()
    df_peri["Source"] = atd.ONTouch_LABEL
    if atd.SUBJECT_COL in df_peri.columns:
        df_peri["Participant"] = df_peri[atd.SUBJECT_COL]
    df_plot = pd.concat(
        [df_kao, df_peri[["Force_Val", "Score", "Source", "Participant"]]],
        ignore_index=True,
    )
    df_plot["Group"] = np.where(
        df_plot["Source"] == atd.KAO_LABEL, "Fingerpad", "Periungual_On-touch"
    )
    return df_kao, df_peri, df_plot


def export_fig3_full_kao(atd):
    """Fig3 full Kao: Fingerpad (Kao n=5) vs Periungual On-touch."""
    source_order = [atd.KAO_LABEL, atd.ONTouch_LABEL]
    cohorts = ["primary_all", "sensitivity_full_n30", "sensitivity_partial_n15"]
    combined_forces = sorted(set(atd.KAO_PAINT_RAW.keys()) | set(atd.df_raw["Force_Val"].unique()))

    desc_rows, comp_rows = [], []
    for cohort in cohorts:
        df_peri = _filter_cohort(
            atd.df_raw[atd.df_raw["Condition"] == "On-touch (Mid)"].copy(),
            atd.SUBJECT_COL, atd._PARTIAL_SUBJ, cohort,
        )
        df_kao, df_peri, df_plot = _build_fig3_frames(atd, df_peri)
        for fval in combined_forces:
            for src in source_order:
                sub = df_plot[(df_plot["Force_Val"] == fval) & (df_plot["Source"] == src)]["Score"].dropna()
                stats = _boxplot_whiskers(sub.values)
                is_kao = src == atd.KAO_LABEL
                n_subj = (
                    df_kao[df_kao["Force_Val"] == fval]["Participant"].nunique()
                    if is_kao
                    else df_peri[df_peri["Force_Val"] == fval][atd.SUBJECT_COL].nunique()
                )
                desc_rows.append({
                    "figure": "Fig3_future_full_kao(final)",
                    "analysis_cohort": cohort,
                    "cohort_label": COHORT_LABELS[cohort],
                    "force_g": fval,
                    "source": "Fingerpad (Kao et al. 2022)" if is_kao else "Periungual: On-touch (this study)",
                    "source_raw": src,
                    "n_trials": stats["n"],
                    "n_subjects": n_subj,
                    "both_conditions_at_force": (
                        fval in df_kao["Force_Val"].values and fval in df_peri["Force_Val"].values
                    ),
                    **{k: stats[k] for k in ["q1", "median", "q3", "iqr", "whisker_lo", "whisker_hi", "mean", "sd"]},
                })
            if fval in df_kao["Force_Val"].values and fval in df_peri["Force_Val"].values:
                comp_rows.append(_comparison_fig3_shared(df_plot, df_kao, df_peri, atd, fval, cohort))

    desc = pd.DataFrame(desc_rows)
    comp = pd.DataFrame(comp_rows)
    _save_fig_stats("Fig3_future_full_kao(final)", desc, comp, OUT_STATS_C1)
    return desc, comp


def export_pooled_onnail(atd):
    """Fig C: On-nail (C+D) vs Off-nail (A+F)."""
    sub_col = atd.SUBJECT_COL
    pool_map = {"C": "On-nail", "D": "On-nail", "A": "Off-nail", "F": "Off-nail"}
    group_order = ["On-nail", "Off-nail"]
    exclude = {0.07, 1.4}
    plot_forces = sorted(f for f in atd.df_raw["Force_Val"].unique() if f not in exclude)
    cohort_specs = [
        ("primary_all", False),
        ("sensitivity_full_n30", False),
        ("sensitivity_partial_n15", False),
        ("sensitivity_trial_pooled_n30", True),
    ]

    desc_rows, comp_rows = [], []
    for cohort, trial_pooled in cohort_specs:
        df = _filter_cohort(atd.df_raw.copy(), sub_col, atd._PARTIAL_SUBJ, cohort)
        for fval in plot_forces:
            if trial_pooled:
                df_f = _subject_trial_pool_group(df, sub_col, "Score", pool_map, fval)
            else:
                df_f = _subject_area_pool(df, sub_col, "Score", pool_map, fval)
            for grp in group_order:
                vals = df_f[df_f["Group"] == grp]["accuracy"].dropna().values
                stats = _boxplot_whiskers(vals)
                desc_rows.append({
                    "figure": "onnail_vs_offnail_pooled(final)",
                    "analysis_cohort": cohort,
                    "cohort_label": COHORT_LABELS[cohort],
                    "force_g": fval,
                    "group": grp,
                    "group_label": f"{grp} (C+D)" if grp == "On-nail" else f"{grp} (A+F)",
                    "pooling_method": (
                        "trial-pooled subject means (1 value/subject/group)"
                        if trial_pooled
                        else "subject×area means (C,D,A,F separate)"
                    ),
                    "n_observations": stats["n"],
                    "n_subjects": df_f[df_f["Group"] == grp][sub_col].nunique(),
                    **{k: stats[k] for k in ["q1", "median", "q3", "iqr", "whisker_lo", "whisker_hi", "mean", "sd"]},
                })
            comp_rows.append(_comparison_onnail(
                df, atd, fval, cohort, pool_map, trial_pooled=trial_pooled,
            ))

    desc = pd.DataFrame(desc_rows)
    comp = pd.DataFrame(comp_rows)
    _save_fig_stats("onnail_vs_offnail_pooled(final)", desc, comp, OUT_STATS_AGG)
    return desc, comp


def _save_fig_stats(stem, desc, comp, out_dir):
    out_desc = os.path.join(out_dir, f"{stem}_descriptives.csv")
    out_comp = os.path.join(out_dir, f"{stem}_comparisons.csv")
    out_all = os.path.join(out_dir, f"{stem}_stats.csv")
    desc.to_csv(out_desc, index=False, float_format="%.4f")
    comp.to_csv(out_comp, index=False, float_format="%.6f")
    pd.concat([desc.assign(row_type="descriptive"), comp.assign(row_type="comparison")],
              ignore_index=True, sort=False).to_csv(out_all, index=False, float_format="%.6f")
    print(f"  Saved → {out_desc}")
    print(f"  Saved → {out_comp}")


def export_figure_n_summary(atd, d2, d3, dp):
    """Overall sample-size summary per figure (for paper Methods/Results)."""
    sub_col = atd.SUBJECT_COL
    df = atd.df_raw
    n_full = df.loc[~df[sub_col].isin(atd._PARTIAL_SUBJ), sub_col].nunique()
    n_partial = len(atd._PARTIAL_SUBJ)
    n_total = df[sub_col].nunique()

    rows = [
        dict(figure="Fig2_ontouch_vs_inair(final)", metric="n_subjects_full_protocol", value=n_full,
             note="Subjects with all force levels (0.07–1.4 g)"),
        dict(figure="Fig2_ontouch_vs_inair(final)", metric="n_subjects_partial_protocol", value=n_partial,
             note="Subjects with 0.16, 0.4, 0.6 g only (shown as △ in figure)"),
        dict(figure="Fig2_ontouch_vs_inair(final)", metric="n_subjects_total", value=n_total,
             note="Unique subjects contributing to any force in Fig2"),
        dict(figure="Fig3_future_full_kao(final)", metric="n_subjects_fingerpad_kao", value=atd.KAO_N,
             note="Kao et al. 2022 digitized fingerpad (independent sample)"),
        dict(figure="Fig3_future_full_kao(final)", metric="n_subjects_periungual_full", value=n_full,
             note="This study, full protocol (● in figure legend)"),
        dict(figure="Fig3_future_full_kao(final)", metric="n_subjects_periungual_partial", value=n_partial,
             note="This study, partial protocol (△ in figure legend)"),
        dict(figure="onnail_vs_offnail_pooled(final)", metric="n_subjects_full_protocol", value=n_full,
             note="Areas A,C,D,F at 0.16/0.6/1.0 g"),
        dict(figure="onnail_vs_offnail_pooled(final)", metric="n_subjects_partial_protocol", value=n_partial,
             note="Areas A,C,D,F at 0.16/0.4/0.6 g only"),
        dict(figure="onnail_vs_offnail_pooled(final)", metric="n_subjects_total", value=n_total,
             note="Unique subjects in on-nail pooled analysis"),
        dict(figure="ALL", metric="sensitivity_1", value="sensitivity_full_n30",
             note="Exclude partial-protocol subjects P61–P75; primary comparison repeated at n=30"),
        dict(figure="ALL", metric="sensitivity_2", value="sensitivity_partial_n15",
             note="Partial-protocol only; applicable at 0.16/0.4/0.6 g"),
        dict(figure="onnail_vs_offnail_pooled(final)", metric="sensitivity_3", value="sensitivity_trial_pooled_n30",
             note="Alternative aggregation: one trial-pooled mean per subject per group (n=30)"),
    ]
    return pd.DataFrame(rows)


def export_workbook(d2, c2, d3, c3, dp, cp, combined, n_summary, sens_summary, sens_comp):
    """Write all stats tables into one Excel workbook (one sheet per table)."""
    out_xlsx = os.path.join(OUT_STATS_C1, "final_figures_stats.xlsx")
    c2_primary = c2[c2["analysis_cohort"] == "primary_all"]
    c3_primary = c3[c3["analysis_cohort"] == "primary_all"]
    cp_primary = cp[cp["analysis_cohort"] == "primary_all"]
    d2_primary = d2[d2["analysis_cohort"] == "primary_all"]
    d3_primary = d3[d3["analysis_cohort"] == "primary_all"]
    dp_primary = dp[dp["analysis_cohort"] == "primary_all"]
    sheets = [
        ("Figure_n_summary", n_summary),
        ("Sensitivity_summary", sens_summary),
        ("Sensitivity_comparisons", sens_comp),
        ("Fig2_descriptives", d2_primary),
        ("Fig2_comparisons", c2_primary),
        ("Fig2_all_cohorts", c2),
        ("Fig3_descriptives", d3_primary),
        ("Fig3_comparisons", c3_primary),
        ("Fig3_all_cohorts", c3),
        ("Onnail_descriptives", dp_primary),
        ("Onnail_comparisons", cp_primary),
        ("Onnail_all_cohorts", cp),
        ("All_combined", combined),
    ]
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        for name, df in sheets:
            df.to_excel(writer, sheet_name=name, index=False)
    print(f"\nWorkbook (multi-tab) → {out_xlsx}")
    return out_xlsx


def main():
    print("Loading ATD data …")
    atd = _load_atd_c1()

    print("\n[Fig2] On-touch vs In-air …")
    d2, c2 = export_fig2(atd)

    print("\n[Fig3] Fingerpad (full Kao) vs Periungual On-touch …")
    d3, c3 = export_fig3_full_kao(atd)

    print("\n[Pooled] On-nail vs Off-nail …")
    dp, cp = export_pooled_onnail(atd)

    combined = pd.concat([
        d2.assign(row_type="descriptive"),
        c2.assign(row_type="comparison"),
        d3.assign(row_type="descriptive"),
        c3.assign(row_type="comparison"),
        dp.assign(row_type="descriptive"),
        cp.assign(row_type="comparison"),
    ], ignore_index=True, sort=False)
    combined_path = os.path.join(OUT_STATS_C1, "final_figures_all_stats.csv")
    combined.to_csv(combined_path, index=False, float_format="%.6f")
    print(f"\nCombined CSV → {combined_path}")

    n_summary = export_figure_n_summary(atd, d2, d3, dp)
    n_path = os.path.join(OUT_STATS_C1, "final_figures_n_summary.csv")
    n_summary.to_csv(n_path, index=False)
    print(f"N summary → {n_path}")

    all_comp = pd.concat([c2, c3, cp], ignore_index=True, sort=False)
    primary_comp = all_comp[all_comp["analysis_cohort"] == "primary_all"]
    sens_comp = all_comp[all_comp["analysis_cohort"] != "primary_all"]
    sens_summary = build_sensitivity_summary(primary_comp, all_comp)
    sens_path = os.path.join(OUT_STATS_C1, "final_figures_sensitivity_summary.csv")
    sens_summary.to_csv(sens_path, index=False, float_format="%.6f")
    print(f"Sensitivity summary → {sens_path}")

    export_workbook(d2, c2, d3, c3, dp, cp, combined, n_summary, sens_summary, sens_comp)

    # Console summary of key values for quick reference
    print("\n" + "=" * 60)
    print("KEY VALUES — PRIMARY + SENSITIVITY (full n=30)")
    print("=" * 60)
    c2_pri = c2[c2["analysis_cohort"] == "primary_all"]
    c2_sen = c2[c2["analysis_cohort"] == "sensitivity_full_n30"]
    for fig, desc_df, comp_df in [
        ("Fig2 primary", d2[d2["analysis_cohort"] == "primary_all"], c2_pri),
        ("Fig2 sens. n=30", d2[d2["analysis_cohort"] == "sensitivity_full_n30"], c2_sen),
        ("Fig3 primary", d3[d3["analysis_cohort"] == "primary_all"],
         c3[c3["analysis_cohort"] == "primary_all"]),
        ("On-nail primary", dp[dp["analysis_cohort"] == "primary_all"],
         cp[cp["analysis_cohort"] == "primary_all"]),
        ("On-nail sens. n=30", dp[dp["analysis_cohort"] == "sensitivity_full_n30"],
         cp[(cp["analysis_cohort"] == "sensitivity_full_n30") & (cp["pooling_method"].str.contains("subject×area"))]),
    ]:
        print(f"\n--- {fig} ---")
        if "primary" in fig.lower() or "sens" in fig.lower():
            for _, r in comp_df.iterrows():
                if pd.notna(r.get("p_value")):
                    d_str = f", d={r['cohens_d']:.2f}" if pd.notna(r.get("cohens_d")) else ""
                    print(f"  {r['force_g']:>4} g: p={r['p_value']:.2e} ({r['sig_star']}){d_str}  n={r.get('n_paired_subjects', r.get('n_ref_subjects', ''))}")
        else:
            for _, r in desc_df.iterrows():
                label = r.get("condition") or r.get("source") or r.get("group_label")
                n_col = r.get("n_trials", r.get("n_observations", np.nan))
                med = r.get("median", np.nan)
                med_s = f"{med:.1f}" if pd.notna(med) else "—"
                print(f"  {r['force_g']:>4} g  {label}: median={med_s}%  (n={int(n_col) if pd.notna(n_col) else 0})")
            shown = comp_df[comp_df.get("shown_on_figure", False) == True]  # noqa: E712
            if len(shown):
                for _, r in shown.iterrows():
                    d_str = f", d={r['cohens_d']:.2f}" if pd.notna(r.get("cohens_d")) else ""
                    print(f"  * {r['force_g']} g: p={r['p_value']:.2e} ({r['sig_star']}){d_str}")

    print("\n--- Robustness verdicts (primary vs full n=30) ---")
    for _, r in sens_summary[sens_summary["analysis_cohort"] == "sensitivity_full_n30"].iterrows():
        print(f"  {r['figure']} {r['force_g']}g: {r['robustness_verdict']}")


if __name__ == "__main__":
    main()
