"""
Export paper-ready descriptive stats for DIC fig1 / fig2 / fig3.

Stats match the plotted boxplots (hold-phase frames pooled across trials;
matplotlib whis=1.5 fences for whiskers).

Outputs (same folder as this script):
  fig1_regionCD_by_force_stats.csv
  fig2_0p40g_by_region_stats.csv
  fig3_1p00g_by_region_stats.csv
  DIC_fig1_fig2_fig3_stats.csv   (combined)
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(OUT_DIR, "hold_phase_data_Final2.csv")

METRICS = [
    "Deformation radius (mm)",
    "Penetration depth (mm)",
    "Max compressive strain (%)",
]

COLS = [
    "figure", "figure_note", "panel_metric", "x_label", "region_raw", "force_g",
    "n_frames", "n_trials", "median", "q1", "q3", "iqr",
    "whisker_lo", "whisker_hi", "mean", "sd", "min", "max",
]


def _box_stats(vals: np.ndarray) -> dict:
    vals = np.asarray(vals, dtype=float)
    vals = vals[~np.isnan(vals)]
    n = len(vals)
    if n == 0:
        return {k: (0 if k == "n" else np.nan)
                for k in ("n", "mean", "sd", "median", "q1", "q3", "iqr",
                          "whisker_lo", "whisker_hi", "min", "max")}
    q1, med, q3 = np.percentile(vals, [25, 50, 75])
    iqr = q3 - q1
    fence_lo, fence_hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    inside = vals[(vals >= fence_lo) & (vals <= fence_hi)]
    return dict(
        n=n,
        mean=float(np.mean(vals)),
        sd=float(np.std(vals, ddof=1)) if n > 1 else 0.0,
        median=float(med),
        q1=float(q1),
        q3=float(q3),
        iqr=float(iqr),
        whisker_lo=float(inside.min() if len(inside) else vals.min()),
        whisker_hi=float(inside.max() if len(inside) else vals.max()),
        min=float(vals.min()),
        max=float(vals.max()),
    )


def _rows(df: pd.DataFrame, groups, figure: str, figure_note: str, label_fn) -> list[dict]:
    rows = []
    for g in groups:
        sub = df.copy()
        for k, v in g["filters"].items():
            sub = sub[sub[k] == v]
        for metric in METRICS:
            st = _box_stats(sub[metric].values)
            rows.append({
                "figure": figure,
                "figure_note": figure_note,
                "panel_metric": metric,
                "x_label": label_fn(g),
                "region_raw": g["filters"].get("Region", ""),
                "force_g": g["filters"].get("Force", ""),
                "n_frames": st["n"],
                "n_trials": int(sub["Trial"].nunique()) if len(sub) else 0,
                "median": st["median"],
                "q1": st["q1"],
                "q3": st["q3"],
                "iqr": st["iqr"],
                "whisker_lo": st["whisker_lo"],
                "whisker_hi": st["whisker_hi"],
                "mean": st["mean"],
                "sd": st["sd"],
                "min": st["min"],
                "max": st["max"],
            })
    return rows


def export_stats(df: pd.DataFrame | None = None, out_dir: str = OUT_DIR) -> dict[str, str]:
    if df is None:
        df = pd.read_csv(CSV_PATH)

    fig1 = _rows(
        df,
        [{"filters": {"Region": "Region CD", "Force": f}} for f in (0.16, 0.40, 0.60, 1.00)],
        "fig1_regionCD_by_force",
        "Region CD (Proximal) across forces; boxplot over hold-phase frames (all trials)",
        lambda g: f"{g['filters']['Force']:.2f}",
    )
    fig2_labels = {"Region CD": "Dorsal nail fold", "Glabrous": "Volar fingerpad"}
    fig2 = _rows(
        df,
        [
            {"filters": {"Region": "Region CD", "Force": 0.40}},
            {"filters": {"Region": "Glabrous", "Force": 0.40}},
        ],
        "fig2_0p40g_by_region",
        "0.40 g; Dorsal nail fold (Region CD) vs Volar fingerpad (Glabrous)",
        lambda g: fig2_labels[g["filters"]["Region"]],
    )
    fig3_labels = {"Region A": "Lateral", "Region CD": "Proximal", "Glabrous": "Volar"}
    fig3 = _rows(
        df,
        [
            {"filters": {"Region": "Region A", "Force": 1.00}},
            {"filters": {"Region": "Region CD", "Force": 1.00}},
            {"filters": {"Region": "Glabrous", "Force": 1.00}},
        ],
        "fig3_1p00g_by_region",
        "1.00 g; Lateral (Region A) vs Proximal (Region CD) vs Volar (Glabrous)",
        lambda g: fig3_labels[g["filters"]["Region"]],
    )

    paths = {}
    for name, rows in [
        ("fig1_regionCD_by_force_stats.csv", fig1),
        ("fig2_0p40g_by_region_stats.csv", fig2),
        ("fig3_1p00g_by_region_stats.csv", fig3),
        ("DIC_fig1_fig2_fig3_stats.csv", fig1 + fig2 + fig3),
    ]:
        path = os.path.join(out_dir, name)
        pd.DataFrame(rows)[COLS].to_csv(path, index=False, float_format="%.6f")
        paths[name] = path
    return paths


if __name__ == "__main__":
    for name, path in export_stats().items():
        print(f"wrote {path}")
