"""Overlay On-touch ATD detection accuracy on baseImg a–f sites."""
from pathlib import Path
import subprocess
import sys

# Non-interactive backend — Cursor/VS Code Run 버튼에서 GUI hang 방지
import matplotlib
matplotlib.use("Agg")

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable

from periungual_heatmap import COORDS, HEAT_CMAP, load_base_img, rbf_belt_rgba

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DATA_DIR = REPO / "Data" / "(ATD)CurData"
OUT_DIR = HERE

FORCE_G = 0.6
CONDITION = "On-touch (Mid)"
regions = ["a", "b", "c", "d", "e", "f"]
coords = COORDS


def calc_score(row):
    if row["Target"] == 0:
        return 100.0 if row["Response"] == 0 else 0.0
    return max(0.0, (1 - abs(row["Target"] - row["Response"]) / row["Target"]) * 100.0)


def load_ontouch_accuracy_by_area(force_g=FORCE_G):
    """Subject-mean Score per Area, then grand mean. force_g=None → all forces."""
    files = sorted(DATA_DIR.glob("P*_AbsoluteThresholdDetection.csv"))
    if not files:
        raise FileNotFoundError(f"No ATD CSVs in {DATA_DIR}")
    df = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    df["Condition"] = df["Condition"].astype(str).str.strip()
    df["Force_Val"] = df["Force"].astype(str).str.extract(r"(\d+\.?\d*)").astype(float)
    df["Area"] = df["Area"].astype(str).str.strip().str.upper()
    df = df[df["Area"].isin(list("ABCDEF"))].copy()
    df["Score"] = df.apply(calc_score, axis=1)

    sub = df[df["Condition"] == CONDITION].copy()
    if force_g is not None:
        sub = sub[sub["Force_Val"] == force_g]
    if sub.empty:
        raise ValueError(f"No rows for {CONDITION}" + (f" @ {force_g} g" if force_g else ""))

    by_sa = sub.groupby(["SubjectID", "Area"], as_index=False)["Score"].mean()
    by_area = by_sa.groupby("Area")["Score"].mean()
    forces = sorted(sub["Force_Val"].unique())
    return (
        {k.lower(): float(by_area[k]) for k in "ABCDEF"},
        by_sa["SubjectID"].nunique(),
        forces,
    )


def generate_overlay_heatmap(metric_name, data_values, output_filename,
                             *, n_subjects=None, forces=None, open_preview=False):
    img = load_base_img()
    img_w, img_h = img.size
    site_xy = [coords[r] for r in regions]
    site_z = [data_values[r] for r in regions]

    rgba, norm, vmin, vmax = rbf_belt_rgba(img, site_xy, site_z, alpha=0.62)

    fig, ax = plt.subplots(figsize=(img_w / 100, img_h / 100), dpi=100)
    ax.imshow(img)
    ax.imshow(rgba, extent=(0, img_w, img_h, 0), interpolation="bilinear")
    # No numeric callout panels — baseImg already has a–f region labels
    ax.axis("off")

    sm = ScalarMappable(norm=norm, cmap=HEAT_CMAP)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.65, pad=0.02)
    cbar.set_label(metric_name, fontsize=11)

    out_path = OUT_DIR / output_filename
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05, facecolor="white")
    plt.close(fig)
    print(f"[{metric_name}] saved → {out_path}")

    # Exact site values mapped into this figure
    force_str = (
        "all"
        if forces is None
        else ";".join(f"{float(f):g}" for f in forces)
    )
    rows = []
    for r, (x, y), z in zip(regions, site_xy, site_z):
        rows.append({
            "figure": output_filename,
            "metric": metric_name,
            "condition": CONDITION,
            "forces_g": force_str,
            "n_subjects": n_subjects,
            "site": r,
            "area": r.upper(),
            "x_px": x,
            "y_px": y,
            "mapped_value": z,
            "colorbar_vmin": vmin,
            "colorbar_vmax": vmax,
        })
    csv_path = out_path.with_suffix(".csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"  mapped values → {csv_path}")

    if open_preview and sys.platform == "darwin":
        subprocess.run(["open", str(out_path)], check=False)


if __name__ == "__main__":
    acc, n_subj, forces = load_ontouch_accuracy_by_area(FORCE_G)
    print(f"On-touch @ {FORCE_G:g} g — n={n_subj}")
    for r in regions:
        print(f"  {r.upper()}: {acc[r]:5.1f}%")
    generate_overlay_heatmap(
        f"Detection accuracy (%)  |  On-touch  {FORCE_G:g} g",
        acc, "Heatmap_Detection_OnTouch_0p6g.png",
        n_subjects=n_subj, forces=forces,
    )

    acc_all, n_all, forces_all = load_ontouch_accuracy_by_area(force_g=None)
    print(f"On-touch ALL — n={n_all} forces={[float(f) for f in forces_all]}")
    for r in regions:
        print(f"  {r.upper()}: {acc_all[r]:5.1f}%")
    generate_overlay_heatmap(
        "Detection accuracy (%)  |  On-touch  (all forces)",
        acc_all, "Heatmap_Detection_OnTouch_allForces.png",
        n_subjects=n_all, forces=forces_all,
        open_preview=True,
    )
