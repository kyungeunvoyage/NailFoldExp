"""
Export Fig3 (Kao Paint vs Periungual On-touch) at 2-column width.

Usage:
    python ATD_C1_Fig_export_10559A_2col.py
"""

import importlib.util
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ANIKA_PATH = os.path.join(SCRIPT_DIR, "ATD_C1_Fig(Anika).py")

spec = importlib.util.spec_from_file_location("atd_anika", ANIKA_PATH)
anika = importlib.util.module_from_spec(spec)
spec.loader.exec_module(anika)

EXPORT_2COL = (("2col", 2102),)
STEM = "Fig3_fingerpad_paint_vs_periungual_ontouch_10559A"

if __name__ == "__main__":
    anika.sns.set_theme(style="white")
    pale_kw = dict(
        peri_box_brightness=anika.COND_BOX_BRIGHTNESS,
        peri_box_alpha_hex=anika.COND_BOX_ALPHA_HEX,
        peri_box_saturation_scale=anika.COND_BOX_SATURATION_SCALE,
    )
    anika.export_fig3_10559A_2col(**pale_kw)
    print("\nDone.")
