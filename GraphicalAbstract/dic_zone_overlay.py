"""DIC periungual-belt overlay: Region A (lateral) + Region CD (proximal) @ 1.0 g."""
from pathlib import Path
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable

from periungual_heatmap import COORDS, HEAT_CMAP, load_base_img, rbf_belt_rgba

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DIC_CSV = REPO / "(New)Analysis" / "DIC" / "hold_phase_data_Final2.csv"
OUT_DIR = HERE

DIC_FORCE_G = 1.0
DIC_METRIC = "Deformation radius (mm)"
DIC_METRIC_LABEL = "Deformation radius (mm)"
EXTRA_METRICS = [
    ("Penetration depth (mm)", "Max normal surface displacement (mm)"),
    ("Max compressive strain (%)", "Max compressive strain (%)"),
]


def load_dic_region_means(force_g=DIC_FORCE_G, metric=DIC_METRIC):
    df = pd.read_csv(DIC_CSV)
    sub = df[(df["Force"] == force_g) & (df["Region"].isin(["Region A", "Region CD"]))].copy()
    if sub.empty:
        raise ValueError(f"No DIC rows for Region A/CD @ {force_g} g")
    trial = sub.groupby(["Region", "Trial"], as_index=False)[metric].mean()
    means = trial.groupby("Region")[metric].mean()
    return {
        "Region A": float(means["Region A"]),
        "Region CD": float(means["Region CD"]),
    }


def dic_sites_from_regions(val_a, val_cd):
    mid = 0.5 * (val_a + val_cd)
    order = ["a", "b", "c", "d", "e", "f"]
    source = {
        "a": "Region A", "f": "Region A",
        "c": "Region CD", "d": "Region CD",
        "b": "mean(A,CD)", "e": "mean(A,CD)",
    }
    vals = {
        "a": val_a, "f": val_a,
        "c": val_cd, "d": val_cd,
        "b": mid, "e": mid,
    }
    site_xy = [COORDS[k] for k in order]
    site_z = [vals[k] for k in order]
    return order, site_xy, site_z, source, vals


def generate_dic_zone_overlay(metric_col, metric_label, output_filename,
                              force_g=DIC_FORCE_G, open_preview=False):
    img = load_base_img()
    img_w, img_h = img.size
    means = load_dic_region_means(force_g, metric_col)
    val_a, val_cd = means["Region A"], means["Region CD"]
    order, site_xy, site_z, source, _ = dic_sites_from_regions(val_a, val_cd)

    rgba, norm, vmin, vmax = rbf_belt_rgba(img, site_xy, site_z, alpha=0.62)

    fig, ax = plt.subplots(figsize=(img_w / 100, img_h / 100), dpi=100)
    ax.imshow(img)
    ax.imshow(rgba, extent=(0, img_w, img_h, 0), interpolation="bilinear")
    # No numeric callout panels — baseImg already has a–f region labels
    ax.axis("off")

    sm = ScalarMappable(norm=norm, cmap=HEAT_CMAP)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.65, pad=0.02)
    cbar.set_label(f"{metric_label}  |  DIC  {force_g:g} g", fontsize=11)

    out_path = OUT_DIR / output_filename
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05, facecolor="white")
    plt.close(fig)
    print(f"[DIC] {metric_col}  A={val_a:.4f}  CD={val_cd:.4f}")
    print(f"  saved → {out_path}")

    rows = []
    for site, (x, y), z in zip(order, site_xy, site_z):
        rows.append({
            "figure": output_filename,
            "metric": metric_col,
            "metric_label": metric_label,
            "force_g": force_g,
            "site": site,
            "source_region": source[site],
            "region_A_value": val_a,
            "region_CD_value": val_cd,
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
    return out_path


if __name__ == "__main__":
    generate_dic_zone_overlay(
        DIC_METRIC, DIC_METRIC_LABEL,
        "Heatmap_DIC_zones_1p0g_deformation.png",
    )
    for col, label in EXTRA_METRICS:
        stem = "displacement" if "Penetration" in col else "strain"
        generate_dic_zone_overlay(col, label, f"Heatmap_DIC_zones_1p0g_{stem}.png")
