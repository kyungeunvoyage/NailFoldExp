"""3D indentation profile from hold_phase_data_Final2.csv (Glabrous, 1.0 g)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CSV_PATH = HERE / "hold_phase_data_Final2.csv"
OUT_NO_GRID = HERE / "skin_indentation_3d_downward.png"
OUT_GRID = HERE / "skin_indentation_3d_downward_grid.png"

df = pd.read_csv(CSV_PATH)
summary = (
    df.groupby(["Region", "Force"])[
        ["Deformation radius (mm)", "Penetration depth (mm)"]
    ]
    .mean()
    .reset_index()
)
target_row = summary[
    (summary["Region"] == "Glabrous") & (np.isclose(summary["Force"], 1.0))
].iloc[0]
depth = float(target_row["Penetration depth (mm)"])
radius = float(target_row["Deformation radius (mm)"])
print(f"Glabrous 1.0 g  mean radius={radius:.4f} mm  depth={depth:.4f} mm")

x = np.linspace(-5, 5, 160)
y = np.linspace(-5, 5, 160)
X, Y = np.meshgrid(x, y)
R = np.sqrt(X**2 + Y**2)


def gaussian_indent(depth, radius):
    return -depth * np.exp(-(R**2) / (2 * (radius / 2) ** 2))


Z_down = gaussian_indent(depth, radius)


def save_plot(path, show_grid, show_title=True, show_axis_labels=True):
    fig = plt.figure(figsize=(10, 8), dpi=300, facecolor="none")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor((0, 0, 0, 0))
    fig.patch.set_alpha(0)

    surf = ax.plot_surface(
        X, Y, Z_down, cmap="viridis_r", edgecolor="none",
        rstride=1, cstride=1, antialiased=True,
    )

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor((1, 1, 1, 0))
        axis.pane.set_alpha(0)
        if not show_grid:
            axis._axinfo["grid"]["linewidth"] = 0
    ax.grid(show_grid)

    if show_axis_labels:
        ax.set_xlabel("X (mm)", fontsize=11, labelpad=10)
        ax.set_ylabel("Y (mm)", fontsize=11, labelpad=10)
        ax.set_zlabel("Displacement (mm)", fontsize=11, labelpad=10)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.invert_yaxis()

    cbar = fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label("Penetration Depth (mm)", fontsize=11)
    cbar.ax.set_facecolor("none")
    cbar.outline.set_alpha(1)

    ax.view_init(elev=25, azim=-45)
    if show_title:
        ax.set_title(
            "3D Skin Indentation Profile (Downward Deformation)",
            fontsize=13, pad=15, fontweight="bold",
        )

    plt.tight_layout()
    fig.savefig(
        path, dpi=300, bbox_inches="tight",
        transparent=True, facecolor="none", edgecolor="none",
    )
    plt.close(fig)
    print(f"saved → {path}")


save_plot(OUT_NO_GRID, show_grid=False)
save_plot(OUT_GRID, show_grid=True)
save_plot(HERE / "skin_indentation_3d_downward_notitle.png", show_grid=False, show_title=False)
save_plot(HERE / "skin_indentation_3d_downward_grid_notitle.png", show_grid=True, show_title=False)
save_plot(
    HERE / "skin_indentation_3d_downward_grid_noaxislabels.png",
    show_grid=True, show_axis_labels=False,
)


def _fmt_force(f):
    return f"{f:.2f}".rstrip("0").rstrip(".") if f != int(f) else f"{f:.1f}"


def _style_3d_ax(ax, show_grid, show_axis_labels):
    ax.set_facecolor((0, 0, 0, 0))
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor((1, 1, 1, 0))
        axis.pane.set_alpha(0)
        if not show_grid:
            axis._axinfo["grid"]["linewidth"] = 0
    ax.grid(show_grid)
    if show_axis_labels:
        ax.set_xlabel("X (mm)", fontsize=10, labelpad=8)
        ax.set_ylabel("Y (mm)", fontsize=10, labelpad=8)
        ax.set_zlabel("Displacement (mm)", fontsize=10, labelpad=8)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.invert_yaxis()
    ax.view_init(elev=25, azim=-45)


def save_by_force(path, region="Region CD", show_grid=True, show_axis_labels=True):
    """Same XYZ/color scale across forces so larger force → visibly larger dent."""
    sub = summary[summary["Region"] == region].sort_values("Force")
    if sub.empty:
        raise ValueError(f"No rows for {region}")
    depths = sub["Penetration depth (mm)"].to_numpy(dtype=float)
    radii = sub["Deformation radius (mm)"].to_numpy(dtype=float)
    forces = sub["Force"].to_numpy(dtype=float)
    zmin = -1.15 * float(np.max(depths))
    vmin, vmax = zmin, 0.0

    n = len(sub)
    fig, axes = plt.subplots(
        1, n, figsize=(4.2 * n, 4.6), dpi=300,
        subplot_kw={"projection": "3d"}, facecolor="none",
    )
    fig.patch.set_alpha(0)
    if n == 1:
        axes = [axes]

    last_surf = None
    for ax, force, depth_i, radius_i in zip(axes, forces, depths, radii):
        Z = gaussian_indent(depth_i, radius_i)
        last_surf = ax.plot_surface(
            X, Y, Z, cmap="viridis_r", edgecolor="none",
            rstride=1, cstride=1, antialiased=True,
            vmin=vmin, vmax=vmax,
        )
        ax.set_zlim(zmin, 0.02)
        _style_3d_ax(ax, show_grid, show_axis_labels)
        ax.set_title(f"{_fmt_force(force)} g", fontsize=13, pad=8)

    cbar = fig.colorbar(last_surf, ax=axes, shrink=0.55, pad=0.04, fraction=0.025)
    cbar.set_label("Displacement (mm)", fontsize=11)
    cbar.ax.set_facecolor("none")
    fig.savefig(
        path, dpi=300, bbox_inches="tight",
        transparent=True, facecolor="none", edgecolor="none",
    )
    plt.close(fig)
    print(f"saved → {path}")
    print(sub[["Force", "Deformation radius (mm)", "Penetration depth (mm)"]].to_string(index=False))


save_by_force(HERE / "skin_indentation_3d_by_force.png")
save_by_force(
    HERE / "skin_indentation_3d_by_force_noaxislabels.png",
    show_axis_labels=False,
)


def _force_colors(forces):
    forces = list(forces)
    n = max(len(forces) - 1, 1)
    return {
        f: mcolors.to_hex(plt.cm.Blues(0.40 + 0.50 * i / n))
        for i, f in enumerate(forces)
    }


def _region_force_rows(region="Region CD"):
    sub = summary[summary["Region"] == region].sort_values("Force")
    if sub.empty:
        raise ValueError(f"No rows for {region}")
    return sub


def save_overlay(path, region="Region CD", show_grid=True, show_axis_labels=True):
    """All forces in one 3D axes: nested bowls, darker = larger force."""
    sub = _region_force_rows(region)
    forces = sub["Force"].to_numpy(dtype=float)
    depths = sub["Penetration depth (mm)"].to_numpy(dtype=float)
    radii = sub["Deformation radius (mm)"].to_numpy(dtype=float)
    colors = _force_colors(forces)
    zmin = -1.20 * float(np.max(depths))
    theta = np.linspace(0, 2 * np.pi, 120)

    fig = plt.figure(figsize=(8.2, 6.6), dpi=300, facecolor="none")
    ax = fig.add_subplot(111, projection="3d")
    fig.patch.set_alpha(0)
    _style_3d_ax(ax, show_grid, show_axis_labels)
    ax.set_zlim(zmin, 0.02)

    # Largest first so smaller dents sit inside.
    for force, depth_i, radius_i in zip(forces[::-1], depths[::-1], radii[::-1]):
        Z = gaussian_indent(depth_i, radius_i)
        rgba = mcolors.to_rgba(colors[force], alpha=0.38)
        ax.plot_surface(
            X, Y, Z, color=rgba, edgecolor="none",
            rstride=2, cstride=2, shade=True, linewidth=0, antialiased=True,
        )
        ax.plot(
            radius_i * np.cos(theta), radius_i * np.sin(theta),
            np.zeros_like(theta), color=colors[force], lw=1.8, zorder=10,
        )

    handles = [
        Line2D([0], [0], color=colors[f], lw=2.4, label=f"{_fmt_force(f)} g")
        for f in forces
    ]
    ax.legend(
        handles=handles, loc="upper left", frameon=False,
        fontsize=11, title="Stimuli Force", title_fontsize=11,
    )
    fig.savefig(
        path, dpi=300, bbox_inches="tight",
        transparent=True, facecolor="none", edgecolor="none",
    )
    plt.close(fig)
    print(f"saved → {path}")


def save_profile(path, region="Region CD"):
    """2D center cut: one glance at wider + deeper with force."""
    sub = _region_force_rows(region)
    forces = sub["Force"].to_numpy(dtype=float)
    depths = sub["Penetration depth (mm)"].to_numpy(dtype=float)
    radii = sub["Deformation radius (mm)"].to_numpy(dtype=float)
    colors = _force_colors(forces)
    xline = np.linspace(-5, 5, 400)
    zmin = -1.20 * float(np.max(depths))

    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=300, facecolor="none")
    fig.patch.set_alpha(0)
    ax.set_facecolor((0, 0, 0, 0))
    for force, depth_i, radius_i in zip(forces, depths, radii):
        z = -depth_i * np.exp(-(xline**2) / (2 * (radius_i / 2) ** 2))
        ax.fill_between(xline, z, 0, color=colors[force], alpha=0.18, linewidth=0)
        ax.plot(xline, z, color=colors[force], lw=2.2, label=f"{_fmt_force(force)} g")
    ax.axhline(0, color="#B0B0B0", lw=0.8, zorder=0)
    ax.set_xlim(-5, 5)
    ax.set_ylim(zmin, 0.008)
    ax.set_xlabel("X (mm)", fontsize=13)
    ax.set_ylabel("Displacement (mm)", fontsize=13)
    ax.tick_params(labelsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        frameon=False, fontsize=11, title="Stimuli Force", title_fontsize=11,
        loc="lower left",
    )
    fig.savefig(
        path, dpi=300, bbox_inches="tight",
        transparent=True, facecolor="none", edgecolor="none",
    )
    plt.close(fig)
    print(f"saved → {path}")


def save_overlay_and_profile(path, region="Region CD"):
    """One figure: nested 3D + 2D cut."""
    sub = _region_force_rows(region)
    forces = sub["Force"].to_numpy(dtype=float)
    depths = sub["Penetration depth (mm)"].to_numpy(dtype=float)
    radii = sub["Deformation radius (mm)"].to_numpy(dtype=float)
    colors = _force_colors(forces)
    zmin = -1.20 * float(np.max(depths))
    theta = np.linspace(0, 2 * np.pi, 120)
    xline = np.linspace(-5, 5, 400)

    fig = plt.figure(figsize=(12.2, 5.4), dpi=300, facecolor="none")
    fig.patch.set_alpha(0)
    ax3 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2)
    _style_3d_ax(ax3, show_grid=True, show_axis_labels=True)
    ax3.set_zlim(zmin, 0.02)
    for force, depth_i, radius_i in zip(forces[::-1], depths[::-1], radii[::-1]):
        Z = gaussian_indent(depth_i, radius_i)
        rgba = mcolors.to_rgba(colors[force], alpha=0.38)
        ax3.plot_surface(
            X, Y, Z, color=rgba, edgecolor="none",
            rstride=2, cstride=2, shade=True, linewidth=0, antialiased=True,
        )
        ax3.plot(
            radius_i * np.cos(theta), radius_i * np.sin(theta),
            np.zeros_like(theta), color=colors[force], lw=1.8,
        )

    ax2.set_facecolor((0, 0, 0, 0))
    for force, depth_i, radius_i in zip(forces, depths, radii):
        z = -depth_i * np.exp(-(xline**2) / (2 * (radius_i / 2) ** 2))
        ax2.fill_between(xline, z, 0, color=colors[force], alpha=0.18, linewidth=0)
        ax2.plot(xline, z, color=colors[force], lw=2.2, label=f"{_fmt_force(force)} g")
    ax2.axhline(0, color="#B0B0B0", lw=0.8, zorder=0)
    ax2.set_xlim(-5, 5)
    ax2.set_ylim(zmin, 0.008)
    ax2.set_xlabel("X (mm)", fontsize=13)
    ax2.set_ylabel("Displacement (mm)", fontsize=13)
    ax2.tick_params(labelsize=11)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.legend(
        frameon=False, fontsize=11, title="Stimuli Force", title_fontsize=11,
        loc="lower left",
    )
    fig.tight_layout()
    fig.savefig(
        path, dpi=300, bbox_inches="tight",
        transparent=True, facecolor="none", edgecolor="none",
    )
    plt.close(fig)
    print(f"saved → {path}")


save_overlay(HERE / "skin_indentation_3d_by_force_overlay.png")
save_profile(HERE / "skin_indentation_2d_by_force_profile.png")
save_overlay_and_profile(HERE / "skin_indentation_by_force_glance.png")
