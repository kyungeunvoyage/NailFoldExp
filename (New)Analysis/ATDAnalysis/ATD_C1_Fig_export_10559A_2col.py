"""
Export Fig3 (Kao Paint vs Periungual On-touch) at 2-column width
with Periungual On-touch color #10559A instead of #295E11.

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
# Teal is high-saturation — pale box fill only (scatter keeps full saturation).
PERI_BOX_BRIGHTNESS = 0.88
PERI_BOX_SATURATION_SCALE = 0.40
PERI_BOX_ALPHA_HEX = "40"  # ~25% opacity

if __name__ == "__main__":
    anika.sns.set_theme(style="white")
    anika.plot_kao_vs_periungual(
        anika.df_raw[anika.df_raw["Condition"] == "On-touch (Mid)"],
        anika.ONTouch_LABEL,
        anika.ON_TOUCH_TEAL,
        STEM,
        export_widths=EXPORT_2COL,
        peri_box_brightness=PERI_BOX_BRIGHTNESS,
        peri_box_alpha_hex=PERI_BOX_ALPHA_HEX,
        peri_box_saturation_scale=PERI_BOX_SATURATION_SCALE,
    )
    print("\nDone.")
