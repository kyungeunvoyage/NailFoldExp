"""Shared periungual-belt heatmap helpers for ATD + DIC overlays."""
from pathlib import Path

import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from scipy import ndimage
from scipy.interpolate import Rbf
from PIL import Image

HERE = Path(__file__).resolve().parent
BASE_IMAGE_PATH = HERE / "baseImg.png"

HEAT_CMAP = LinearSegmentedColormap.from_list(
    "nailfold_hot",
    ["#FFF7D6", "#FFE14A", "#FF9A00", "#E85A20", "#C31213"],
    N=256,
)

COORDS = {
    "a": (118, 247),
    "b": (132, 405),
    "c": (239, 486),
    "d": (384, 485),
    "e": (485, 401),
    "f": (522, 248),
}


def periungual_belt_mask(img_rgba, soft_sigma=1.2, close_r=52, thicken=3):
    """Periungual belt only — fills double white U, excludes nail plate."""
    arr = np.asarray(img_rgba)
    rgb = arr[:, :, :3].astype(np.float32)
    alpha = arr[:, :, 3]
    gray = rgb.mean(axis=2)
    finger = (alpha > 128) & (gray > 18)
    white = (rgb > 220).all(axis=2) & (alpha > 200)

    yy, xx = np.ogrid[-close_r:close_r + 1, -close_r:close_r + 1]
    disk = xx * xx + yy * yy <= close_r * close_r
    ring = ndimage.binary_closing(white, structure=disk)
    ring = ndimage.binary_dilation(ring, iterations=thicken)

    envelope = ndimage.binary_dilation(white, iterations=6)
    inside = ndimage.binary_fill_holes(envelope)
    dist = ndimage.distance_transform_edt(~white)
    nail_plate = inside & (dist > (close_r * 0.55))

    hard = ring & finger & ~nail_plate
    soft = ndimage.gaussian_filter(hard.astype(np.float32), sigma=soft_sigma)
    soft = np.clip(soft / max(float(soft.max()), 1e-6), 0.0, 1.0)
    soft[~finger] = 0.0
    soft[nail_plate] = 0.0
    return soft


def data_vmin_vmax(values, frac=0.35, floor=1e-3):
    lo, hi = float(min(values)), float(max(values))
    pad = max((hi - lo) * frac, abs(hi) * 0.05, floor)
    return lo - pad, hi + pad


def rbf_belt_rgba(img_rgba, site_xy, site_z, vmin=None, vmax=None, alpha=0.62):
    """Interpolate site values over the continuous periungual belt → RGBA."""
    img_w, img_h = img_rgba.size
    xs = [p[0] for p in site_xy]
    ys = [p[1] for p in site_xy]
    zs = list(site_z)
    if vmin is None or vmax is None:
        vmin, vmax = data_vmin_vmax(zs)

    rbf = Rbf(xs, ys, zs, function="multiquadric", smooth=0.5, epsilon=40.0)
    xi = np.linspace(0, img_w - 1, img_w)
    yi = np.linspace(0, img_h - 1, img_h)
    XI, YI = np.meshgrid(xi, yi)
    ZI = np.clip(rbf(XI, YI), vmin, vmax)

    soft = periungual_belt_mask(img_rgba)
    norm = Normalize(vmin=vmin, vmax=vmax)
    rgba = HEAT_CMAP(norm(ZI))
    rgba[..., 3] = soft * alpha
    return rgba, norm, vmin, vmax


def load_base_img():
    return Image.open(BASE_IMAGE_PATH).convert("RGBA")
