"""Shared figure export settings and palette for Force Discrimination analysis plots."""

from matplotlib.colors import LinearSegmentedColormap

FIG_SIZE = (14.0, 6.4)
SAVE_DPI = 300
EXPORT_PX = (int(round(FIG_SIZE[0] * SAVE_DPI)), int(round(FIG_SIZE[1] * SAVE_DPI)))

# PNAS-style palette (matches Stats(GEE).py / ATD figures)
SLATE_BLUE = "#56708A"
OLIVE      = "#686F12"
WINE       = "#7F212B"
WINE_LIGHT = "#F5E0E3"  # pale wine for heatmap low values
CREAM      = "#EDE2D0"
BLACK      = "#1A1A1A"

PALETTE_4 = [SLATE_BLUE, OLIVE, WINE, CREAM]
REGION_ORDER = ["A", "B", "C", "D", "E", "F"]
REGION_PALETTE = {
    region: PALETTE_4[i % len(PALETTE_4)]
    for i, region in enumerate(REGION_ORDER)
}
HEATMAP_CMAP = LinearSegmentedColormap.from_list("fd_wine", [WINE_LIGHT, WINE])


def save_figure_png(fig, path):
    """Save PNG at fixed publication size (14.0 × 6.4 in @ 300 dpi → 4200 × 1920 px)."""
    fig.savefig(
        path,
        dpi=SAVE_DPI,
        format="png",
        facecolor="white",
        edgecolor="none",
        pil_kwargs={"compress_level": 1},
    )
