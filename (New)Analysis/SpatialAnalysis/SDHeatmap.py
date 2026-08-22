import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.optimize import curve_fit
from scipy import stats

# ==========================================
# 1. Path Configuration
# ==========================================
DATA_DIR = "/Users/kyungeunjung/NailFoldExp/Data/(SD)CurData"
SAVE_DIR = "/Users/kyungeunjung/NailFoldExp/(New)Analysis/SpatialAnalysis"
os.makedirs(SAVE_DIR, exist_ok=True)

GRID_SPACING_MM = 1.5
THRESHOLD = 0.75  # 75% criterion for JND


# ==========================================
# 2. Load all Spatial Discrimination data
# ==========================================
def _parse_grid(s):
    m = re.match(r"g(-?\d+)", str(s).strip())
    return float(m.group(1)) if m else np.nan


def _parse_force(s):
    m = re.match(r"([\d.]+)", str(s).strip())
    return float(m.group(1)) if m else np.nan


def _psychometric(x, x50, beta, lapse=0.02):
    return 0.5 + (0.48 - lapse) / (1.0 + np.exp(-beta * (x - x50)))


def _fit_jnd(xs, ys, target=THRESHOLD):
    popt, _ = curve_fit(
        _psychometric, xs, ys,
        p0=(3.0, 1.0),
        bounds=([0.1, 0.05], [15.0, 10.0]),
        maxfev=8000,
    )
    x_arr = np.linspace(0, 20, 20000)
    y_arr = _psychometric(x_arr, *popt)
    jnd = float(x_arr[np.argmin(np.abs(y_arr - target))])
    return popt, jnd


csv_files = sorted(glob.glob(os.path.join(DATA_DIR, "P*_SpatialDiscrimination.csv")))
if not csv_files:
    raise FileNotFoundError(f"No SD CSVs found in {DATA_DIR}")

df = pd.concat(
    [pd.read_csv(f, encoding="utf-8-sig") for f in csv_files],
    ignore_index=True,
)
df["pos_1st_mm"] = df["Stim_1st"].apply(_parse_grid) * GRID_SPACING_MM
df["pos_2nd_mm"] = df["Stim_2nd"].apply(_parse_grid) * GRID_SPACING_MM
df["abs_offset_mm"] = (df["pos_2nd_mm"] - df["pos_1st_mm"]).abs()
df["force_g"] = df["Force"].apply(_parse_force)
df["IsCorrect"] = pd.to_numeric(df["IsCorrect"], errors="coerce")
df = df.dropna(subset=["IsCorrect", "abs_offset_mm", "force_g"])

print(f"[Load] {len(csv_files)} files, {df['Subject'].nunique()} subjects, {len(df)} trials")

# Subject → group psychometric per force
subj_acc = (
    df.groupby(["Subject", "force_g", "abs_offset_mm"], as_index=False)
      .agg(accuracy=("IsCorrect", "mean"))
)
grp_acc = (
    subj_acc.groupby(["force_g", "abs_offset_mm"], as_index=False)
            .agg(mean_acc=("accuracy", "mean"))
)

force_fits = {}
for force in sorted(grp_acc["force_g"].unique()):
    sub = grp_acc[grp_acc["force_g"] == force].sort_values("abs_offset_mm")
    popt, jnd = _fit_jnd(sub["abs_offset_mm"].values, sub["mean_acc"].values)
    force_fits[force] = {"popt": popt, "jnd": jnd, "data": sub}
    print(f"  {force:g} g → group JND = {jnd:.2f} mm")

# Pooled across force (for inset label / geometry)
subj_pool = (
    df.groupby(["Subject", "abs_offset_mm"], as_index=False)
      .agg(accuracy=("IsCorrect", "mean"))
)
grp_pool = (
    subj_pool.groupby("abs_offset_mm", as_index=False)
             .agg(mean_acc=("accuracy", "mean"))
             .sort_values("abs_offset_mm")
)
popt_pool, jnd_pool = _fit_jnd(
    grp_pool["abs_offset_mm"].values, grp_pool["mean_acc"].values
)
print(f"  pooled → group JND = {jnd_pool:.2f} mm")

# Per-subject JNDs → force invariance check
jnd_rows = []
for (subj, force), g in subj_acc.groupby(["Subject", "force_g"]):
    g = g.sort_values("abs_offset_mm")
    if len(g) < 3:
        continue
    try:
        _, jnd = _fit_jnd(g["abs_offset_mm"].values, g["accuracy"].values)
        jnd_rows.append({"Subject": subj, "force_g": force, "jnd_mm": jnd})
    except Exception:
        continue

jnd_df = pd.DataFrame(jnd_rows)
wide = jnd_df.pivot(index="Subject", columns="force_g", values="jnd_mm").dropna()
force_invariant = False
p_wilcox = np.nan
if {1.0, 26.0}.issubset(set(wide.columns)) and len(wide) >= 3:
    _, p_wilcox = stats.wilcoxon(wide[1.0], wide[26.0])
    force_invariant = p_wilcox >= 0.05
    print(
        f"  Wilcoxon JND(1g vs 26g): p={p_wilcox:.3f}, "
        f"force-invariant={force_invariant} (n={len(wide)})"
    )


# ==========================================
# 3. Data-driven graphical abstract heatmap
# ==========================================
# Plot units = mm. Two stimulus points separated by the pooled JND.
# Place on the x-axis so the 2.9 mm gap is readable against tick marks.
jnd_mm = float(jnd_pool)
p1 = np.array([-jnd_mm / 2.0, 0.0])
p2 = np.array([+jnd_mm / 2.0, 0.0])

# Spatial field: discriminability contrast from fitted psychometric.
pad = 1.5
lim = jnd_mm / 2.0 + pad
x = np.linspace(-lim, lim, 300)
y = np.linspace(-lim, lim, 300)
X, Y = np.meshgrid(x, y)

d1 = np.sqrt((X - p1[0]) ** 2 + (Y - p1[1]) ** 2)
d2 = np.sqrt((X - p2[0]) ** 2 + (Y - p2[1]) ** 2)
acc1 = _psychometric(d1, *popt_pool)
acc2 = _psychometric(d2, *popt_pool)
Z = acc1 - acc2  # signed discriminability field from real psychometric

fig = plt.figure(figsize=(7.0, 6.2), dpi=300)
fig.patch.set_alpha(0.0)
ax = fig.add_axes([0.12, 0.12, 0.72, 0.80])
ax.patch.set_alpha(0.0)

cf = ax.contourf(X, Y, Z, levels=50, cmap="viridis_r")
ax.scatter(
    [p1[0], p2[0]], [p1[1], p2[1]],
    c="#909090", edgecolors="#202020", s=120, linewidth=1.5, zorder=10,
    label="Stimulus sites",
)

# Explicit JND distance annotation (readable on the x-axis)
bracket_y = 0.55
ax.annotate(
    "",
    xy=(p2[0], bracket_y), xytext=(p1[0], bracket_y),
    arrowprops=dict(arrowstyle="<->", color="black", lw=1.4),
    zorder=12,
)
ax.text(
    0.0, bracket_y + 0.18, f"{jnd_mm:.1f} mm",
    ha="center", va="bottom", fontsize=12, fontweight="bold", color="black",
    zorder=13,
)

# ax.text(
#     0.92, 0.92, "Fig 6", transform=ax.transAxes,
#     fontsize=14, fontweight="normal", ha="right", va="top", color="black",
# )

# Ticks at 1 mm; include ±JND/2-friendly marks
tick_max = int(np.floor(lim))
ticks = np.arange(-tick_max, tick_max + 1, 1)
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xlabel("x (mm)", fontsize=13)
ax.set_ylabel("y (mm)", fontsize=13)
ax.tick_params(axis="both", which="both", direction="in",
               length=5, width=1.0, labelsize=12)
ax.set_aspect("equal")
for spine in ax.spines.values():
    spine.set_edgecolor("#101010")
    spine.set_linewidth(1.5)

# Colorbar for discriminability field
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="4%", pad=0.10)
cax.patch.set_alpha(0.0)
cbar = fig.colorbar(cf, cax=cax)
cbar.set_label(r"$\Delta P$ (site 1 $-$ site 2)", fontsize=12)
cbar.ax.tick_params(labelsize=11, direction="in", length=3.5, width=0.8)

# Marker legend
stim_handle = Line2D(
    [0], [0], marker="o", color="none",
    markerfacecolor="#909090", markeredgecolor="#202020",
    markeredgewidth=1.2, markersize=10, label="Stimulus sites",
)
leg = ax.legend(
    handles=[stim_handle],
    loc="lower left",
    frameon=True,
    fancybox=False,
    edgecolor="#202020",
    facecolor="none",
    framealpha=0.0,
    fontsize=11,
    handletextpad=0.4,
    borderpad=0.35,
)

# ==========================================
# 4. Save (transparent background)
# ==========================================
output_path = os.path.join(SAVE_DIR, "JND_Abstract_Inset.png")
w_in, h_in = fig.get_size_inches()
plt.savefig(
    output_path, dpi=300, bbox_inches="tight",
    transparent=True, facecolor="none", edgecolor="none",
)
plt.close()

print(f"\nSaved → {output_path}")
print(f"  points separated by {jnd_mm:.2f} mm (pooled group JND)")
print(f"  field = psychometric(d→p1) − psychometric(d→p2)")
print(f"  figsize = {w_in:.1f}×{h_in:.1f} in @300 dpi (transparent PNG)")
