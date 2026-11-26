"""1.0 g: DIC vs On-touch accuracy by region (C/D, A, volar finger pad)."""
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import dic_vs_detection_regionCD as d

HERE = Path(__file__).resolve().parent
FORCE_G = 1.0

# Kao et al. 2022 No-paint, digitized in ATD_C1_Fig(Anika).py (n=5)
KAO_1G = [100.0, 100.0, 100.0, 100.0, 100.0]

REGIONS = [
    {
        "key": "cd",
        "label": "Proximal nail fold",
        "dic_region": "Region CD",
        "color": "#D55E00",
    },
    {
        "key": "a",
        "label": "Lateral nail fold",
        "dic_region": "Region A",
        "color": "#009E73",
    },
    {
        "key": "volar",
        "label": "Volar finger pad",
        "dic_region": "Glabrous",
        "color": "#5A5A5A",
    },
]
REGION_COLORS = {r["label"]: r["color"] for r in REGIONS}


def load_dic_region_at_force(region, metric, force=FORCE_G):
    df = pd.read_csv(d.DIC_CSV)
    sub = df[df["Region"] == region].copy()
    if sub.empty:
        raise ValueError(f"No DIC rows for {region}")
    if metric not in sub.columns:
        raise KeyError(f"No DIC column {metric!r}")
    sub = sub[np.isclose(sub["Force"].astype(float), force)]
    if sub.empty:
        raise ValueError(f"No DIC rows for {region} @ {force} g")
    trial = sub.groupby(["Force", "Trial"], as_index=False)[metric].mean()
    vals = trial[metric].to_numpy(dtype=float)
    return {
        "dic_mean": float(np.mean(vals)),
        "dic_sem": d._sem(vals),
        "n_trials": int(trial["Trial"].nunique()),
    }


def load_atd_areas_at_force(areas, force=FORCE_G):
    files = sorted(d.DATA_DIR.glob("P*_AbsoluteThresholdDetection.csv"))
    if not files:
        raise FileNotFoundError(f"No ATD CSVs in {d.DATA_DIR}")
    df = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    df["Condition"] = df["Condition"].astype(str).str.strip()
    df["Force_Val"] = df["Force"].astype(str).str.extract(r"(\d+\.?\d*)").astype(float)
    df["Area"] = df["Area"].astype(str).str.strip().str.upper()
    areas = tuple(a.upper() for a in areas)
    df = df[df["Area"].isin(areas)].copy()
    df["Score"] = df.apply(d.calc_score, axis=1)
    sub = df[df["Condition"] == d.CONDITION].copy()
    sub = sub[np.isclose(sub["Force_Val"].astype(float), force)]
    if sub.empty:
        raise ValueError(f"No On-touch rows for {areas} @ {force} g")
    by_sa = sub.groupby(["SubjectID", "Area"], as_index=False)["Score"].mean()
    by_s = by_sa.groupby("SubjectID", as_index=False)["Score"].mean()
    by_s["ratio_correct"] = by_s["Score"] / 100.0
    return by_s.rename(columns={"SubjectID": "subject"})[["subject", "ratio_correct"]]


def load_kao_at_force(force=FORCE_G):
    if not np.isclose(force, 1.0):
        raise ValueError("Kao 1.0 g values are hardcoded; other forces not wired here")
    return pd.DataFrame({
        "subject": [f"KP{i + 1}" for i in range(len(KAO_1G))],
        "ratio_correct": np.asarray(KAO_1G, dtype=float) / 100.0,
    })


def collect_region_points(metric):
    rows = []
    mean_rows = []
    for spec in REGIONS:
        dic = load_dic_region_at_force(spec["dic_region"], metric)
        if spec["key"] == "cd":
            atd = load_atd_areas_at_force(("C", "D"))
        elif spec["key"] == "a":
            atd = load_atd_areas_at_force(("A",))
        else:
            atd = load_kao_at_force()
        for _, a in atd.iterrows():
            rows.append({
                "region": spec["label"],
                "subject": a["subject"],
                "dic_x": dic["dic_mean"],
                "dic_sem": dic["dic_sem"],
                "ratio_correct": float(a["ratio_correct"]),
            })
        mean_rows.append({
            "region": spec["label"],
            "dic_x": dic["dic_mean"],
            "dic_sem": dic["dic_sem"],
            "acc_mean": float(atd["ratio_correct"].mean()),
            "acc_sem": d._sem(atd["ratio_correct"]),
            "n": int(len(atd)),
        })
    points = pd.DataFrame(rows)
    means = pd.DataFrame(mean_rows)
    return points, means


def _region_legend(ax, handles):
    leg = ax.legend(
        handles=handles, loc="lower right",
        bbox_to_anchor=(0.98, 0.065),
        fontsize=15, title="Region", title_fontsize=15,
        frameon=False, ncol=1,
        borderpad=0.15, labelspacing=0.28, handletextpad=0.35,
        handlelength=0.8, handleheight=0.65, borderaxespad=0.0,
        alignment="center",
    )
    if hasattr(leg, "_legend_box"):
        leg._legend_box.sep = 1
    title = leg.get_title()
    title.set_ha("center")
    leg.set_zorder(11)
    return leg


def _draw_region_scatter(ax, points, means, rng, scatter_s=22, mean_s=80):
    xs = means["dic_x"].to_numpy()
    span = max(float(np.ptp(xs)), 0.08 * float(np.mean(np.abs(xs))))
    d._hline_80(ax)
    order = means.sort_values("dic_x")
    ax.plot(order["dic_x"], d._acc(order["acc_mean"]), color=d.LINE_GRAY,
            lw=1.2, zorder=2)
    for spec in REGIONS:
        g = points[points["region"] == spec["label"]]
        jitter = rng.uniform(-0.045 * span, 0.045 * span, size=len(g))
        ax.scatter(
            g["dic_x"].to_numpy() + jitter, d._acc(g["ratio_correct"]),
            s=scatter_s, facecolor=spec["color"], edgecolor="none",
            alpha=0.55, zorder=3, clip_on=True,
        )
    ax.scatter(
        means["dic_x"], d._acc(means["acc_mean"]), s=mean_s,
        facecolor=[REGION_COLORS[r] for r in means["region"]],
        edgecolor="#000000", linewidth=0.8, zorder=4,
    )


def _legend_handles():
    return [
        Line2D([0], [0], marker="o", color="none",
               markerfacecolor=spec["color"], markeredgecolor="none",
               markersize=12, label=spec["label"])
        for spec in REGIONS
    ]


def plot_region_color(points, means, path, xlabel):
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    rng = np.random.default_rng(42)
    _draw_region_scatter(ax, points, means, rng)
    _region_legend(ax, _legend_handles())
    d._axes_labels(ax, xlabel=xlabel)
    ax._nf_xdata = points["dic_x"].to_numpy()
    d._apply_x_axis(ax, ax._nf_xdata)
    d._apply_y_axis(ax)
    d._despine(ax)
    d._fit_legend_panel(ax)
    d._save(fig, path)


def plot_region_three_metrics(metric_data, path):
    n = len(metric_data)
    fig, axes = plt.subplots(1, n, figsize=(5.0 * n, 4.4), sharey=True)
    if n == 1:
        axes = [axes]
    rng = np.random.default_rng(42)
    for ax, (xlabel, points, means) in zip(axes, metric_data):
        _draw_region_scatter(ax, points, means, rng, scatter_s=18, mean_s=70)
        ax.set_xlabel(xlabel, fontsize=15, labelpad=8)
        ax._nf_xdata = points["dic_x"].to_numpy()
        d._apply_x_axis(ax, ax._nf_xdata)
        d._apply_y_axis(ax)
        d._despine(ax)
    axes[0].set_ylabel(d.Y_LABEL, fontsize=18, labelpad=8)
    _region_legend(axes[-1], _legend_handles())
    fig.tight_layout()
    d._fit_legend_panel(axes[-1])
    d._save(fig, path)


if __name__ == "__main__":
    metric_data = []
    paths = []
    for col, label, stem in d.DIC_METRICS:
        points, means = collect_region_points(col)
        out = HERE / f"dic_vs_detection_regions_1g_{stem}.png"
        plot_region_color(points, means, out, xlabel=label)
        csv_path = HERE / f"dic_vs_detection_regions_1g_{stem}.csv"
        points.to_csv(csv_path, index=False, float_format="%.6f")
        print(f"  {stem} table → {csv_path}")
        print(means.assign(metric=col).to_string(index=False))
        metric_data.append((label, points, means))
        paths.append(out)

    three = HERE / "dic_vs_detection_regions_1g_three_metrics.png"
    plot_region_three_metrics(metric_data, three)

    if sys.platform == "darwin":
        subprocess.run(["open", str(three)], check=False)
        for p in paths:
            subprocess.run(["open", str(p)], check=False)
