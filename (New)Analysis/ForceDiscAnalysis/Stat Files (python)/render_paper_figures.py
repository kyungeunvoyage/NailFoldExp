"""
render_paper_figures.py
=======================
세 논문 피겨를 동일한 스타일로 렌더링합니다.

  Fig A  fd_onnail_vs_offnail_meanCI      (FD_OnnailVsOffnail.py)
  Fig B  dissociation_overlay_panel       (Force_ATD.py)
  Fig C  gee_pairwise_plot_horizontal     (On-off(GEE).py)

공유 스타일
  FONT_TICK  = 16
  FONT_LABEL = 17
  FIG_SIZE   = (8.0, 4.5) inches  →  모든 axes 높이 동일 → y축 간격 동일
  SAVE_DPI   = 600
  TARGET_W   = 2102 px (2col)

출력: paper_output/
"""

import os, io, importlib.util
from pathlib import Path

os.environ["PAPER_RENDER"] = "1"   # suppress auto-generation in each module

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"]     = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]
import matplotlib.pyplot as plt
from PIL import Image

# ─── Shared style ─────────────────────────────────────────────────────────────
FONT_TICK  = 16
FONT_LABEL = 17
FIG_SIZE   = (8.0, 4.5)
SAVE_DPI   = 600
TARGET_W   = 2102

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR    = SCRIPT_DIR / "final_paper_output"
OUT_DIR.mkdir(exist_ok=True)


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def save_paper(fig, filename):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=SAVE_DPI, bbox_inches="tight",
                pad_inches=0.05, facecolor="white")
    buf.seek(0)
    master = Image.open(buf).convert("RGB")
    h = round(TARGET_W * master.height / master.width)
    out = OUT_DIR / filename
    master.resize((TARGET_W, h), Image.Resampling.LANCZOS).save(out)
    print(f"  Saved: {out.name}  ({TARGET_W}×{h} px)")


# =============================================================================
# Fig A  –  FD Mean ± 95% CI  (On-nail vs Off-nail)
# =============================================================================
print("\n[Fig A] FD Mean ± CI  (On-nail vs Off-nail)")

fd = _load(SCRIPT_DIR / "FD_OnnailVsOffnail.py", "fd_mod")
fd.FONT_TICK    = FONT_TICK
fd.FONT_LABEL   = FONT_LABEL
fd.YLIM_TOP_CAP = 100   # cap y-axis at 100%
# figsize is already (8.0, 4.5) inside draw_fd_meanCI_figure

fig_a = fd.draw_fd_meanCI_figure()   # returns fig when PAPER_RENDER=1
save_paper(fig_a, "paper_fd_onnail_vs_offnail_meanCI.png")
plt.close(fig_a)


# =============================================================================
# Fig B  –  Dissociation overlay panel  (ATD + FD, dual y-axis)
# =============================================================================
print("\n[Fig B] Dissociation overlay panel")

ov = _load(SCRIPT_DIR / "Force_ATD.py", "ov_mod")
# overlay function hardcodes FONT_X + 4, so set base = target - 4
ov.FONT_TICK  = FONT_TICK  - 4    # 12  → +4 = 16
ov.FONT_LABEL = FONT_LABEL - 4    # 10  → +4 = 14
# figsize: function uses FIG_SIZE[0] * 1.05, patch so result = 8.0
ov.FIG_SIZE   = (FIG_SIZE[0] / 1.05, FIG_SIZE[1])   # (7.619…, 4.5)

fig_b = ov.make_overlay_panel()
save_paper(fig_b, "paper_dissociation_overlay_panel.png")
plt.close(fig_b)


# =============================================================================
# Fig C  –  GEE pairwise horizontal boxplot
# =============================================================================
print("\n[Fig C] GEE pairwise horizontal")

gee = _load(SCRIPT_DIR / "On-off(GEE).py", "gee_mod")
gee.FONT_TICK  = FONT_TICK
gee.FONT_LABEL = FONT_LABEL
gee.FIG_SIZE   = (FIG_SIZE[0], FIG_SIZE[1] + 1.0)   # taller for wider y-tick spacing

fig_c = gee.make_pairwise_figure("horizontal")
save_paper(fig_c, "paper_gee_pairwise_horizontal.png")
plt.close(fig_c)


print(f"\n✓ 세 피겨 모두 저장 완료: {OUT_DIR}")
