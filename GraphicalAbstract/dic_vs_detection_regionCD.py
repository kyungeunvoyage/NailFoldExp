"""Region CD: DIC deformation vs On-touch accuracy — figure candidates."""
from pathlib import Path
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from PIL import Image
from scipy import stats
from scipy.optimize import curve_fit

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DIC_CSV = REPO / "(New)Analysis" / "DIC" / "hold_phase_data_Final2.csv"
DATA_DIR = REPO / "Data" / "(ATD)CurData"
OUT_DIR = HERE

DIC_METRIC = "Deformation radius (mm)"
DIC_METRICS = [
    ("Deformation radius (mm)", "Deformation radius (mm)", "deformation"),
    ("Penetration depth (mm)", "Max normal surface displacement (mm)", "displacement"),
    ("Max compressive strain (%)", "Max compressive strain (%)", "strain"),
]
CONDITION = "On-touch (Mid)"
CD_AREAS = ("C", "D")
CRITERION = 0.80
Y_LABEL = "Detection accuracy (%)"
Y_LIM = (-12.0, 108.0)
Y_TICKS = [0, 50, 100]
SCATTER_COLOR = "#B0B0B0"
MEAN_COLOR = mcolors.to_hex(plt.cm.Blues(0.85))
LINE_GRAY = "#888888"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11,
    "axes.linewidth": 1,
    "axes.edgecolor": "#000000",
    "xtick.color": "#000000",
    "ytick.color": "#000000",
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.width": 1,
    "ytick.major.width": 1,
    "axes.grid": False,
})


def calc_score(row):
    if row["Target"] == 0:
        return 100.0 if row["Response"] == 0 else 0.0
    return max(0.0, (1 - abs(row["Target"] - row["Response"]) / row["Target"]) * 100.0)


def _fmt_force(f):
    return f"{f:.2f}".rstrip("0").rstrip(".") if f != int(f) else f"{f:.1f}"


def _sem(vals):
    vals = np.asarray(vals, dtype=float)
    vals = vals[~np.isnan(vals)]
    n = len(vals)
    if n < 2:
        return 0.0
    return float(np.std(vals, ddof=1) / np.sqrt(n))


def logistic(x, x0, k):
    return 1.0 / (1.0 + np.exp(-(x - x0) / k))


def threshold_x(x0, k, criterion=CRITERION):
    return x0 + k * np.log(criterion / (1.0 - criterion))


def _despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)


def _axes_labels(ax, xlabel="Deformation radius (mm)"):
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=15, labelpad=8)
    ax.set_ylabel(Y_LABEL, fontsize=18, labelpad=8)


def _acc(y):
    return np.asarray(y, dtype=float) * 100.0


def _apply_y_axis(ax):
    ax.set_autoscaley_on(False)
    ax.set_ylim(*Y_LIM)
    ax.set_yticks(Y_TICKS)
    ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(Y_TICKS))
    ax.spines["left"].set_bounds(Y_LIM[0], 100)
    ax.tick_params(axis="both", labelsize=14)


def _set_y_acc(ax):
    _apply_y_axis(ax)


def _x_ticks_and_lim(x, pad=0.16):
    """Nice ticks that cover the data; xlim snaps to the first and last tick."""
    x = np.asarray(x, dtype=float)
    lo, hi = _data_xlim(x, pad)
    locator = matplotlib.ticker.MaxNLocator(nbins=6, min_n_ticks=4, prune=None)
    ticks = np.asarray(locator.tick_values(lo, hi), dtype=float)
    data_lo, data_hi = float(np.min(x)), float(np.max(x))
    if ticks.size == 0:
        return lo, hi, ticks
    first = ticks[ticks <= data_lo + 1e-8]
    last = ticks[ticks >= data_hi - 1e-8]
    x0 = float(first[-1]) if first.size else float(ticks[0])
    x1 = float(last[0]) if last.size else float(ticks[-1])
    if x1 < data_hi:
        step = float(np.min(np.diff(ticks))) if ticks.size > 1 else (hi - lo)
        x1 = x1 + step
        ticks = np.append(ticks, x1)
    vis = ticks[(ticks >= x0 - 1e-8) & (ticks <= x1 + 1e-8)]
    return x0, x1, vis


def _apply_x_axis(ax, x, pad=0.16):
    x0, x1, vis = _x_ticks_and_lim(x, pad=pad)
    ax.set_autoscalex_on(False)
    ax.set_xlim(x0, x1)
    ax.set_xticks(vis)
    ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator(vis))
    ax.spines["bottom"].set_bounds(x0, x1)


def _trim_x_axis(ax):
    x = getattr(ax, "_nf_xdata", None)
    if x is None:
        return
    _apply_x_axis(ax, x)


def _hline_80(ax):
    ax.axhline(80.0, color="#8A8A8A", lw=1.15, ls=(0, (5.5, 3.5)), zorder=1)


def _force_legend(ax, handles):
    # matplotlib fills columns first; reorder so rows read 0.16  0.4 / 0.6  1.0
    ncol = 2
    n = len(handles)
    nrow = int(np.ceil(n / ncol))
    padded = list(handles) + [None] * (nrow * ncol - n)
    grid = np.array(padded, dtype=object).reshape(nrow, ncol)
    col_handles = [h for h in grid.T.flatten() if h is not None]
    leg = ax.legend(
        handles=col_handles, loc="lower right",
        bbox_to_anchor=(0.98, 0.065),
        fontsize=15, title="Stimuli Force (g)", title_fontsize=15,
        frameon=False, ncol=ncol,
        borderpad=0.15, labelspacing=0.22, handletextpad=0.25,
        handlelength=0.8, handleheight=0.65, borderaxespad=0.0,
        columnspacing=0.85, alignment="center",
    )
    if hasattr(leg, "_legend_box"):
        leg._legend_box.sep = 1
    title = leg.get_title()
    title.set_ha("center")
    leg.set_zorder(11)
    return leg


def _fit_legend_panel(ax, leg=None):
    """Draw a rounded panel that hugs the legend text bbox."""
    from matplotlib.patches import FancyBboxPatch
    from matplotlib.transforms import Bbox

    if leg is None:
        leg = ax.get_legend()
    if leg is None:
        return
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bb = leg.get_window_extent(renderer)
    pad = 3.0
    bb = Bbox.from_extents(bb.x0 - pad, bb.y0 - pad, bb.x1 + pad, bb.y1 + pad)
    bb_ax = bb.transformed(ax.transAxes.inverted())
    panel = FancyBboxPatch(
        (bb_ax.x0, bb_ax.y0), bb_ax.width, bb_ax.height,
        boxstyle="round,pad=0.0,rounding_size=0.012",
        facecolor=(1, 1, 1, 0.85),
        edgecolor="#D0D0D0",
        linewidth=0.7,
        transform=ax.transAxes,
        zorder=leg.get_zorder() - 0.1,
        clip_on=False,
    )
    ax.add_patch(panel)


def _force_colors(forces):
    forces = list(forces)
    n = max(len(forces) - 1, 1)
    return {
        f: mcolors.to_hex(plt.cm.Blues(0.40 + 0.50 * i / n))
        for i, f in enumerate(forces)
    }


def _save(fig, path):
    for ax in fig.get_axes():
        ax.set_autoscale_on(False)
        _apply_y_axis(ax)
        xdata = getattr(ax, "_nf_xdata", None)
        if xdata is not None:
            _apply_x_axis(ax, xdata)
    fig.savefig(path, dpi=200, facecolor="white", edgecolor="none",
                bbox_inches="tight", pad_inches=0.20)
    plt.close(fig)
    print(f"saved → {path}")


def load_dic_cd_by_force(metric=DIC_METRIC):
    df = pd.read_csv(DIC_CSV)
    sub = df[df["Region"] == "Region CD"].copy()
    if sub.empty:
        raise ValueError("No Region CD rows in DIC CSV")
    if metric not in sub.columns:
        raise KeyError(f"No DIC column {metric!r}")
    trial = sub.groupby(["Force", "Trial"], as_index=False)[metric].mean()
    rows = []
    for force, g in trial.groupby("Force"):
        vals = g[metric].to_numpy(dtype=float)
        rows.append({
            "force_g": float(force),
            "dic_mean": float(np.mean(vals)),
            "dic_sem": _sem(vals),
            "n_trials": int(g["Trial"].nunique()),
        })
    return pd.DataFrame(rows).sort_values("force_g").reset_index(drop=True)


def load_atd_cd_subject_points():
    files = sorted(DATA_DIR.glob("P*_AbsoluteThresholdDetection.csv"))
    if not files:
        raise FileNotFoundError(f"No ATD CSVs in {DATA_DIR}")
    df = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    df["Condition"] = df["Condition"].astype(str).str.strip()
    df["Force_Val"] = df["Force"].astype(str).str.extract(r"(\d+\.?\d*)").astype(float)
    df["Area"] = df["Area"].astype(str).str.strip().str.upper()
    df = df[df["Area"].isin(CD_AREAS)].copy()
    df["Score"] = df.apply(calc_score, axis=1)
    sub = df[df["Condition"] == CONDITION].copy()
    if sub.empty:
        raise ValueError(f"No rows for {CONDITION} in Areas C/D")
    by_sa = sub.groupby(["SubjectID", "Area", "Force_Val"], as_index=False)["Score"].mean()
    by_s = by_sa.groupby(["SubjectID", "Force_Val"], as_index=False)["Score"].mean()
    by_s["ratio_correct"] = by_s["Score"] / 100.0
    return by_s.rename(columns={"SubjectID": "subject", "Force_Val": "force_g"})[
        ["subject", "force_g", "ratio_correct"]
    ]


def join_subject_points(dic, atd):
    rows = []
    for _, d in dic.iterrows():
        match = atd[np.isclose(atd["force_g"].to_numpy(dtype=float), d["force_g"])]
        if match.empty:
            continue
        for _, a in match.iterrows():
            rows.append({
                "force_g": float(d["force_g"]),
                "subject": a["subject"],
                "dic_x": float(d["dic_mean"]),
                "dic_sem": float(d["dic_sem"]),
                "ratio_correct": float(a["ratio_correct"]),
            })
    if not rows:
        raise ValueError("No overlapping forces between DIC Region CD and ATD C+D")
    return pd.DataFrame(rows).sort_values(["force_g", "subject"]).reset_index(drop=True)


def summarize_means(points):
    rows = []
    for force, g in points.groupby("force_g"):
        rows.append({
            "force_g": float(force),
            "dic_x": float(g["dic_x"].iloc[0]),
            "dic_sem": float(g["dic_sem"].iloc[0]),
            "acc_mean": float(g["ratio_correct"].mean()),
            "acc_sem": _sem(g["ratio_correct"]),
            "n": int(g["subject"].nunique()),
        })
    return pd.DataFrame(rows).sort_values("force_g").reset_index(drop=True)


def fit_logistic(x, y):
    x = np.asarray(x, dtype=float)
    y = np.clip(np.asarray(y, dtype=float), 1e-4, 1.0 - 1e-4)
    x0_0 = float(np.median(x))
    k0 = max(0.05, float(np.ptp(x)) / 4.0)
    popt, _ = curve_fit(
        logistic, x, y, p0=(x0_0, k0),
        bounds=([x.min() - 1.0, 1e-4], [x.max() + 1.0, 5.0]),
        maxfev=8000,
    )
    return float(popt[0]), float(popt[1])


def _data_xlim(x, pad=0.10):
    x = np.asarray(x, dtype=float)
    span = max(float(np.ptp(x)), 0.1)
    return x.min() - pad * span, x.max() + pad * span


# ---------------------------------------------------------------------------
# A  Mean ± SEM trajectory  — cleanest one-sentence story
# ---------------------------------------------------------------------------
def plot_a_means(means, path):
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    x = means["dic_x"].to_numpy()
    y = _acc(means["acc_mean"])
    _hline_80(ax)
    ax.plot(x, y, color=MEAN_COLOR, lw=1.8, zorder=2)
    ax.errorbar(
        x, y, xerr=means["dic_sem"], yerr=_acc(means["acc_sem"]),
        fmt="none", ecolor=MEAN_COLOR, elinewidth=1.2, capsize=3.5, zorder=3,
    )
    ax.scatter(x, y, s=90, facecolor=MEAN_COLOR, edgecolor="#000000",
               linewidth=0.8, zorder=4)
    for _, row in means.iterrows():
        ax.annotate(
            f"{_fmt_force(row['force_g'])} g",
            (row["dic_x"], float(_acc(row["acc_mean"]))),
            textcoords="offset points", xytext=(8, 6), fontsize=10,
        )
    _axes_labels(ax)
    ax.set_xlim(*_data_xlim(x, 0.18))
    _set_y_acc(ax)
    ax.set_title("A   Mean ± SEM trajectory", fontsize=12, pad=10, loc="left")
    _despine(ax)
    _save(fig, path)


# ---------------------------------------------------------------------------
# B  Subject cloud + means  — variation + the rise
# ---------------------------------------------------------------------------
def plot_b_cloud_means(points, means, path):
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    rng = np.random.default_rng(42)
    x = points["dic_x"].to_numpy()
    y = _acc(points["ratio_correct"])
    span = float(np.ptp(np.unique(x)))
    jitter = rng.uniform(-0.025 * span, 0.025 * span, size=len(x))
    _hline_80(ax)
    ax.scatter(x + jitter, y, s=16, facecolor=SCATTER_COLOR, edgecolor="none",
               alpha=0.35, zorder=2)
    xm, ym = means["dic_x"].to_numpy(), _acc(means["acc_mean"])
    ax.plot(xm, ym, color=MEAN_COLOR, lw=1.8, zorder=3)
    ax.errorbar(
        xm, ym, yerr=_acc(means["acc_sem"]), fmt="none",
        ecolor=MEAN_COLOR, elinewidth=1.3, capsize=3, zorder=4,
    )
    ax.scatter(xm, ym, s=70, marker="D", facecolor=MEAN_COLOR,
               edgecolor="#000000", linewidth=0.7, zorder=5)
    for _, row in means.iterrows():
        ax.annotate(
            f"{_fmt_force(row['force_g'])} g",
            (row["dic_x"], float(_acc(row["acc_mean"]))),
            textcoords="offset points", xytext=(8, 7), fontsize=10,
        )
    _axes_labels(ax)
    ax.set_xlim(*_data_xlim(x, 0.16))
    _set_y_acc(ax)
    ax.set_title("B   Subjects + mean trajectory", fontsize=12, pad=10, loc="left")
    _despine(ax)
    _save(fig, path)


# ---------------------------------------------------------------------------
# C  Psychometric sigmoid  — 80% threshold on deformation
# ---------------------------------------------------------------------------
def plot_c_psychometric(points, path, title="C   Psychometric sigmoid"):
    x = points["dic_x"].to_numpy(dtype=float)
    y = points["ratio_correct"].to_numpy(dtype=float)
    x0, k = fit_logistic(x, y)
    x_thr = threshold_x(x0, k)
    r_p, p_p = stats.pearsonr(x, y)
    rng = np.random.default_rng(42)
    span = float(np.ptp(np.unique(x)))
    jitter = rng.uniform(-0.018 * span, 0.018 * span, size=len(x))
    x_min, x_max = float(x.min()), float(x.max())
    x_lo, x_hi = 0.0, x_max + 0.15 * (x_max - x_min)
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    _hline_80(ax)
    ax.plot(np.linspace(x_lo, x_min, 80),
            _acc(logistic(np.linspace(x_lo, x_min, 80), x0, k)),
            color="#000000", lw=1.4, ls="--", zorder=2)
    ax.plot(np.linspace(x_min, x_hi, 160),
            _acc(logistic(np.linspace(x_min, x_hi, 160), x0, k)),
            color="#000000", lw=1.4, ls="-", zorder=2)
    ax.axvline(x_thr, color="#C8C8C8", lw=0.9, ls=":", zorder=1)
    ax.scatter(x + jitter, _acc(y), s=18, facecolor=SCATTER_COLOR, edgecolor="none",
               alpha=0.35, zorder=3)
    _axes_labels(ax)
    ax.set_xlim(x_lo, x_hi)
    _set_y_acc(ax)
    if title:
        ax.set_title(title, fontsize=12, pad=10, loc="left")
    _despine(ax)
    _save(fig, path)
    print(f"  80% threshold = {x_thr:.3f} mm   r = {r_p:.3f}  p = {p_p:.4f}")
    return x0, k, x_thr


# ---------------------------------------------------------------------------
# D  Color by force  — which cloud is which force
# ---------------------------------------------------------------------------
def plot_d_force_color(points, means, path, xlabel=None, title=None):
    if xlabel is None:
        xlabel = "Deformation radius (mm)"
    forces = list(means["force_g"])
    colors = _force_colors(forces)
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    rng = np.random.default_rng(42)
    span = float(np.ptp(means["dic_x"].to_numpy()))
    _hline_80(ax)
    ax.plot(means["dic_x"], _acc(means["acc_mean"]), color=LINE_GRAY,
            lw=1.2, zorder=2)
    for force, g in points.groupby("force_g"):
        jitter = rng.uniform(-0.022 * span, 0.022 * span, size=len(g))
        ax.scatter(
            g["dic_x"].to_numpy() + jitter, _acc(g["ratio_correct"]),
            s=22, facecolor=colors[force], edgecolor="none", alpha=0.55, zorder=3,
            clip_on=True,
        )
    ax.scatter(
        means["dic_x"], _acc(means["acc_mean"]), s=80,
        facecolor=[colors[f] for f in means["force_g"]],
        edgecolor="#000000", linewidth=0.8, zorder=4,
    )
    handles = [
        Line2D([0], [0], marker="o", color="none",
               markerfacecolor=colors[f], markeredgecolor="none",
               markersize=12, label=_fmt_force(f))
        for f in forces
    ]
    _force_legend(ax, handles)
    _axes_labels(ax, xlabel=xlabel)
    ax._nf_xdata = points["dic_x"].to_numpy()
    _apply_x_axis(ax, ax._nf_xdata)
    _apply_y_axis(ax)
    if title:
        ax.set_title(title, fontsize=12, pad=10, loc="left")
    _despine(ax)
    _fit_legend_panel(ax)
    _save(fig, path)


def plot_d_three_metrics(metric_data, path):
    """1x3 Figure D: same accuracy scatter, three DIC x-axes."""
    n = len(metric_data)
    fig, axes = plt.subplots(1, n, figsize=(5.0 * n, 4.4), sharey=True)
    if n == 1:
        axes = [axes]
    forces = list(metric_data[0][2]["force_g"])
    colors = _force_colors(forces)
    rng = np.random.default_rng(42)
    for ax, (xlabel, points, means) in zip(axes, metric_data):
        span = float(np.ptp(means["dic_x"].to_numpy()))
        _hline_80(ax)
        ax.plot(means["dic_x"], _acc(means["acc_mean"]), color=LINE_GRAY, lw=1.2, zorder=2)
        for force, g in points.groupby("force_g"):
            jitter = rng.uniform(-0.022 * span, 0.022 * span, size=len(g))
            ax.scatter(
                g["dic_x"].to_numpy() + jitter, _acc(g["ratio_correct"]),
                s=18, facecolor=colors[force], edgecolor="none", alpha=0.55, zorder=3,
                clip_on=True,
            )
        ax.scatter(
            means["dic_x"], _acc(means["acc_mean"]), s=70,
            facecolor=[colors[f] for f in means["force_g"]],
            edgecolor="#000000", linewidth=0.8, zorder=4,
        )
        ax.set_xlabel(xlabel, fontsize=15, labelpad=8)
        ax._nf_xdata = points["dic_x"].to_numpy()
        _apply_x_axis(ax, ax._nf_xdata)
        _apply_y_axis(ax)
        _despine(ax)
    axes[0].set_ylabel(Y_LABEL, fontsize=18, labelpad=8)
    handles = [
        Line2D([0], [0], marker="o", color="none",
               markerfacecolor=colors[f], markeredgecolor="none",
               markersize=12, label=_fmt_force(f))
        for f in forces
    ]
    _force_legend(axes[-1], handles)
    fig.tight_layout()
    _fit_legend_panel(axes[-1])
    for ax in axes:
        _trim_x_axis(ax)
    _save(fig, path)


# ---------------------------------------------------------------------------
# E  Within-subject spaghetti  — the rise happens inside people
# ---------------------------------------------------------------------------
def plot_e_spaghetti(points, means, path):
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    _hline_80(ax)
    n_lines = 0
    for _, g in points.groupby("subject"):
        g = g.sort_values("dic_x")
        if len(g) < 2:
            continue
        ax.plot(
            g["dic_x"], _acc(g["ratio_correct"]),
            color="#B8C9DC", lw=0.8, alpha=0.45, zorder=2,
        )
        n_lines += 1
    ax.plot(means["dic_x"], _acc(means["acc_mean"]), color=MEAN_COLOR, lw=2.2, zorder=3)
    ax.scatter(
        means["dic_x"], _acc(means["acc_mean"]), s=80,
        facecolor=MEAN_COLOR, edgecolor="#000000", linewidth=0.8, zorder=4,
    )
    for _, row in means.iterrows():
        ax.annotate(
            f"{_fmt_force(row['force_g'])} g",
            (row["dic_x"], float(_acc(row["acc_mean"]))),
            textcoords="offset points", xytext=(8, 6), fontsize=10,
        )
    _axes_labels(ax)
    ax.set_xlim(*_data_xlim(points["dic_x"], 0.16))
    _set_y_acc(ax)
    ax.set_title(f"E   Within-subject trajectories  (n={n_lines})",
                 fontsize=12, pad=10, loc="left")
    _despine(ax)
    _save(fig, path)


# ---------------------------------------------------------------------------
# F  Two-panel: both rise with force, then map onto each other
# ---------------------------------------------------------------------------
def plot_f_two_panel(means, path):
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.2))
    fvals = means["force_g"].to_numpy()
    xticklabels = [_fmt_force(f) for f in fvals]

    ax = axes[0]
    ax.errorbar(
        fvals, means["dic_x"], yerr=means["dic_sem"],
        fmt="o-", color=MEAN_COLOR, lw=1.6, markersize=8,
        markeredgecolor="#000000", markeredgewidth=0.6,
        ecolor=MEAN_COLOR, capsize=3, zorder=3,
    )
    ax.set_xticks(fvals)
    ax.set_xticklabels(xticklabels)
    ax.set_xlabel("Stimulus force (g)", fontsize=11, labelpad=8)
    ax.set_ylabel("Deformation radius (mm)", fontsize=11, labelpad=8)
    ax.set_title("Force to tissue", fontsize=11, pad=8)
    _despine(ax)

    ax = axes[1]
    _hline_80(ax)
    ax.errorbar(
        fvals, _acc(means["acc_mean"]), yerr=_acc(means["acc_sem"]),
        fmt="o-", color=MEAN_COLOR, lw=1.6, markersize=8,
        markeredgecolor="#000000", markeredgewidth=0.6,
        ecolor=MEAN_COLOR, capsize=3, zorder=3,
    )
    ax.set_xticks(fvals)
    ax.set_xticklabels(xticklabels)
    ax.set_xlabel("Stimulus force (g)", fontsize=11, labelpad=8)
    ax.set_ylabel(Y_LABEL, fontsize=18, labelpad=8)
    _set_y_acc(ax)
    ax.set_title("Force to detection", fontsize=11, pad=8)
    _despine(ax)

    fig.suptitle("F   Same force, both increase", fontsize=12, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, path)


def make_contact_sheet(pngs, out_path, ncol=3):
    imgs = [Image.open(p).convert("RGB") for p in pngs]
    h = min(im.height for im in imgs)
    scaled = []
    for im in imgs:
        w = round(im.width * h / im.height)
        scaled.append(im.resize((w, h), Image.Resampling.LANCZOS))
    col_w = max(im.width for im in scaled)
    nrow = int(np.ceil(len(scaled) / ncol))
    pad = 18
    canvas = Image.new(
        "RGB",
        (ncol * col_w + (ncol + 1) * pad, nrow * h + (nrow + 1) * pad),
        "white",
    )
    for i, im in enumerate(scaled):
        r, c = divmod(i, ncol)
        x = pad + c * col_w + (col_w - im.width) // 2
        y = pad + r * h
        canvas.paste(im, (x, y))
    canvas.save(out_path)
    print(f"contact sheet → {out_path}")


if __name__ == "__main__":
    atd = load_atd_cd_subject_points()

    metric_d_data = []
    d_paths = []
    for col, label, stem in DIC_METRICS:
        dic = load_dic_cd_by_force(col)
        points = join_subject_points(dic, atd)
        means = summarize_means(points)
        xlabel = label
        out = OUT_DIR / f"dic_vs_detection_regionCD_D_{stem}.png"
        plot_d_force_color(points, means, out, xlabel=xlabel, title=None)
        csv_path = OUT_DIR / f"dic_vs_detection_regionCD_D_{stem}.csv"
        points.to_csv(csv_path, index=False, float_format="%.6f")
        print(f"  {stem} table → {csv_path}")
        print(means.assign(metric=col).to_string(index=False))
        metric_d_data.append((xlabel, points, means))
        d_paths.append(out)

    # keep previous D filename as the deformation panel
    plot_d_force_color(
        metric_d_data[0][1], metric_d_data[0][2],
        OUT_DIR / "dic_vs_detection_regionCD_D_force_color.png",
        xlabel=metric_d_data[0][0], title="D   Colored by force",
    )
    plot_d_three_metrics(
        metric_d_data,
        OUT_DIR / "dic_vs_detection_regionCD_D_three_metrics.png",
    )

    dic = load_dic_cd_by_force(DIC_METRIC)
    points = join_subject_points(dic, atd)
    means = summarize_means(points)
    csv_path = OUT_DIR / "dic_vs_detection_regionCD.csv"
    points.to_csv(csv_path, index=False, float_format="%.6f")

    paths = [
        OUT_DIR / "dic_vs_detection_regionCD_A_means.png",
        OUT_DIR / "dic_vs_detection_regionCD_B_cloud.png",
        OUT_DIR / "dic_vs_detection_regionCD_C_psychometric.png",
        OUT_DIR / "dic_vs_detection_regionCD_D_force_color.png",
        OUT_DIR / "dic_vs_detection_regionCD_E_spaghetti.png",
        OUT_DIR / "dic_vs_detection_regionCD_F_two_panel.png",
    ]
    plot_a_means(means, paths[0])
    plot_b_cloud_means(points, means, paths[1])
    plot_c_psychometric(points, paths[2])
    plot_e_spaghetti(points, means, paths[4])
    plot_f_two_panel(means, paths[5])
    plot_c_psychometric(points, OUT_DIR / "dic_vs_detection_regionCD.png", title="")

    sheet = OUT_DIR / "dic_vs_detection_regionCD_candidates.png"
    make_contact_sheet(paths, sheet, ncol=3)
    three = OUT_DIR / "dic_vs_detection_regionCD_D_three_metrics.png"
    if sys.platform == "darwin":
        subprocess.run(["open", str(three)], check=False)
        for p in d_paths:
            subprocess.run(["open", str(p)], check=False)
