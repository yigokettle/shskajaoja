# -*- coding: utf-8 -*-
"""图4b（范文风格）：单张 3D DEM 曲面 + jet 彩虹配色 + 服务点贴地标注。

复刻用户补充范文「采集点整体分布图」的呈现：
  - 单一 3D 曲面，jet 彩虹色阶（蓝=低海拔 → 红=高海拔）
  - 服务区用红点贴合地形表面，旁标编号
  - 独立竖直色条标海拔
数据口径与 q1_paper.py 完全一致（同一 DEM、同一节点坐标）。
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from mpl_toolkits.mplot3d import Axes3D  # noqa
from PIL import Image

try:
    from scipy.ndimage import gaussian_filter
except Exception:
    gaussian_filter = None

Image.MAX_IMAGE_PIXELS = None
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "_review" / "D" / "code" / "data"
DEM_TIF = (ROOT / "数据" / "镇龙乡地理空间数据" / "镇龙乡及周边地理数据"
           / "数字高程模型数据（DEM）" / "镇龙乡及周边30米DEM.tif")
FIG = ROOT / "论文" / "图"
FIG.mkdir(parents=True, exist_ok=True)

# ---- 中文字体 ----
for fp in ["/System/Library/Fonts/Hiragino Sans GB.ttc",
           "/System/Library/Fonts/STHeiti Medium.ttc"]:
    if Path(fp).exists():
        fm.fontManager.addfont(fp)
mpl.rcParams["font.sans-serif"] = ["Hiragino Sans GB", "STHeiti", "Arial Unicode MS"]
mpl.rcParams["axes.unicode_minus"] = False

# ---- 节点 ----
nodes = pd.read_csv(DATA / "服务区数据.csv").rename(
    columns={"V": "id", "x": "lon", "y": "lat", "h": "elev"}).set_index("id")
SIDS = [f"S{i:03d}" for i in range(1, 16)]

# ---- DEM ----
im = Image.open(DEM_TIF)
DEM = np.array(im, dtype=np.float64)
NY, NX = DEM.shape
PS = 1.0 / 3600.0
LON0 = 109.03277777777778
LAT0 = 23.224722222222223


def dem_elev(lon, lat):
    col = (np.asarray(lon) - LON0) / PS
    row = (LAT0 - np.asarray(lat)) / PS
    r = np.clip(np.round(row).astype(int), 0, NY - 1)
    c = np.clip(np.round(col).astype(int), 0, NX - 1)
    return DEM[r, c]


# 裁剪到节点包络（外扩 0.012°）
lons, lats = nodes.lon.values, nodes.lat.values
lon_min, lon_max = lons.min() - 0.012, lons.max() + 0.012
lat_min, lat_max = lats.min() - 0.012, lats.max() + 0.012
c0 = int((lon_min - LON0) / PS); c1 = int((lon_max - LON0) / PS)
r0 = int((LAT0 - lat_max) / PS); r1 = int((LAT0 - lat_min) / PS)
step = 2
sub = DEM[r0:r1, c0:c1].astype(np.float64)[::step, ::step]
if gaussian_filter is not None:
    sub = gaussian_filter(sub, sigma=2.2)  # 更平滑，接近范文大尺度地形观感
gy, gx = sub.shape
LON = LON0 + (c0 + np.arange(gx) * step) * PS
LAT = LAT0 - (r0 + np.arange(gy) * step) * PS
LONg, LATg = np.meshgrid(LON, LAT)

zmin, zmax = float(np.nanmin(sub)), float(np.nanmax(sub))
norm = plt.Normalize(zmin, zmax)

# ---- 绘图（范文风格 jet 彩虹）：单张三维图，点位强制置顶不被山体遮挡 ----
from matplotlib import patheffects as pe
fig = plt.figure(figsize=(11.5, 9))
ax = fig.add_subplot(111, projection="3d")
ax.set_box_aspect((1, 1, 0.55))
# 关键：关闭自动深度排序，使 zorder 被严格遵守 → 点/杆/编号永远画在曲面之上
ax.computed_zorder = False

surf = ax.plot_surface(LONg, LATg, sub, cmap="jet", norm=norm,
                       rstride=1, cstride=1, linewidth=0,
                       antialiased=True, shade=True, alpha=0.9, zorder=0)

# 服务点：从地表竖杆上抬 + 红点 + 编号，全部 zorder 高于曲面
zspan = zmax - zmin
for sid in SIDS:
    s = nodes.loc[sid]
    zs = float(dem_elev(s.lon, s.lat))
    z_top = zs + 0.12 * zspan
    ax.plot([s.lon, s.lon], [s.lat, s.lat], [zs, z_top],
            color="black", lw=0.8, alpha=0.75, zorder=5)
    ax.scatter(s.lon, s.lat, z_top, color="red", s=60,
               edgecolor="white", linewidth=1.1, depthshade=False, zorder=6)
    ax.text(s.lon, s.lat, z_top + 0.04 * zspan, sid[1:], fontsize=11,
            color="black", ha="center", va="bottom", fontweight="bold", zorder=7,
            path_effects=[pe.withStroke(linewidth=2.4, foreground="white")])

# O01 调度中心（金边黑星）
o = nodes.loc["O01"]
zo = float(dem_elev(o.lon, o.lat))
zo_top = zo + 0.12 * zspan
ax.plot([o.lon, o.lon], [o.lat, o.lat], [zo, zo_top],
        color="black", lw=1.0, alpha=0.8, zorder=5)
ax.scatter(o.lon, o.lat, zo_top, color="black", s=240, marker="*",
           edgecolor="gold", linewidth=1.5, depthshade=False, zorder=6)
ax.text(o.lon, o.lat, zo_top + 0.04 * zspan, "O01", fontsize=11.5,
        color="black", ha="center", va="bottom", fontweight="bold", zorder=7,
        path_effects=[pe.withStroke(linewidth=2.4, foreground="white")])

# 竖直色条
cb = fig.colorbar(surf, ax=ax, fraction=0.026, pad=0.12, shrink=0.62)
cb.set_label("海拔高度 / m", fontsize=12)
cb.outline.set_linewidth(0.6)

ax.set_xlabel("经度 (°)", fontsize=13, labelpad=12)
ax.set_ylabel("纬度 (°)", fontsize=13, labelpad=12)
ax.set_zlabel("海拔高度 / m", fontsize=13, labelpad=10)
ax.tick_params(labelsize=10)
ax.set_zlim(zmin, zmax + 0.22 * zspan)
ax.view_init(elev=32, azim=-62)
for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
    pane.pane.set_facecolor((1, 1, 1, 0.0))
    pane.pane.set_edgecolor((0.75, 0.75, 0.8, 0.6))
ax.set_title("图4b  镇龙乡 30m DEM 高程可视化与 15 个服务区空间分布", fontsize=13, pad=8)

fig.savefig(FIG / "fig4b_dem_jet_points.pdf")
fig.savefig(FIG / "fig4b_dem_jet_points.png", dpi=210)
plt.close(fig)
print("已输出 fig4b_dem_jet_points.png/.pdf 至", FIG)
print(f"高程范围 {zmin:.0f}~{zmax:.0f} m，网格 {gy}×{gx}")
