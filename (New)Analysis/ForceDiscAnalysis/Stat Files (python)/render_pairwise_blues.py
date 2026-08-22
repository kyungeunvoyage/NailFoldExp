"""Render the two FD pairwise Final figures in Blues (same hue, different sat).

Outputs → (New)Analysis/bone_figures/ForceDisc/
  (Final)gee_pairwise_plot_horizontal_2col.png       — higher saturation
  (Final)gee_pairwise_samediff_horizontal_2col.png   — lower saturation

Does not leave FD_PAIRWISE_BLUE set; Final/ is restored afterward.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import colorsys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[2]
BONE_FD = REPO / "(New)Analysis" / "bone_figures" / "ForceDisc"
OUT_STATS = REPO / "(New)Analysis" / "ForceDiscAnalysis" / "Output" / "Stats(GEE)"
OUT_SD = REPO / "(New)Analysis" / "ForceDiscAnalysis" / "Output" / "SameDiff_GEE"
FINAL = REPO / "(New)Analysis" / "ForceDiscAnalysis" / "Final"
PY = sys.executable


def blues_hex(*, saturation: float, value: float = 0.58, cmap_t: float = 0.72) -> str:
    """Same Blues hue; only saturation (and fixed V) differ."""
    r, g, b, _ = plt.colormaps["Blues"](cmap_t)
    h, _s, _v = colorsys.rgb_to_hsv(r, g, b)
    return mcolors.to_hex(colorsys.hsv_to_rgb(h, float(saturation), float(value)))


# Higher sat → Stats GEE pairwise (circles)
# Lower sat  → SameDiff pairwise (triangles)
# Wide sat gap: pale_box_face() compresses chroma, so start farther apart.
BLUE_HIGH_SAT = blues_hex(saturation=0.92, value=0.52)
BLUE_LOW_SAT = blues_hex(saturation=0.22, value=0.62)


def _run(script: str, blue: str) -> None:
    env = os.environ.copy()
    env["FD_PAIRWISE_BLUE"] = blue
    env["MPLCONFIGDIR"] = env.get("MPLCONFIGDIR", "/tmp/mplconfig-nailfold")
    print(f"\n=== {script}  FD_PAIRWISE_BLUE={blue} ===")
    subprocess.run([PY, str(SCRIPT_DIR / script)], cwd=str(SCRIPT_DIR), env=env, check=True)


def main() -> None:
    BONE_FD.mkdir(parents=True, exist_ok=True)
    print(f"Blues high-sat (Stats pairwise):    {BLUE_HIGH_SAT}")
    print(f"Blues low-sat  (SameDiff pairwise): {BLUE_LOW_SAT}")

    # Backup Final samediff (script publishes into Final/)
    sd_final = FINAL / "(Final)gee_pairwise_samediff_horizontal_2col.png"
    sd_bak = FINAL / "(Final)gee_pairwise_samediff_horizontal_2col.png.__bak_pre_blues"
    if sd_final.is_file():
        shutil.copy2(sd_final, sd_bak)

    _run("Stats(GEE).py", BLUE_HIGH_SAT)
    src_stats = OUT_STATS / "gee_pairwise_plot_horizontal_2col.png"
    dst_stats = BONE_FD / "(Final)gee_pairwise_plot_horizontal_2col.png"
    shutil.copy2(src_stats, dst_stats)
    print(f"→ {dst_stats}")

    _run("gee_pairwise_samediff.py", BLUE_LOW_SAT)
    src_sd = OUT_SD / "gee_pairwise_samediff_horizontal_2col.png"
    dst_sd = BONE_FD / "(Final)gee_pairwise_samediff_horizontal_2col.png"
    shutil.copy2(src_sd, dst_sd)
    print(f"→ {dst_sd}")

    # Restore Final samediff to pre-blues
    if sd_bak.is_file():
        shutil.move(str(sd_bak), str(sd_final))
        print(f"Restored Final → {sd_final.name}")

    # Re-render Stats Output with default #10559A so Output/ is not left on Blues
    _run("Stats(GEE).py", "#10559A")

    print("\nDone. Blues variants only in bone_figures/ForceDisc/")


if __name__ == "__main__":
    main()
