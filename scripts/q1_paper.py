# -*- coding: utf-8 -*-
"""Q1 论文用统一计算 + 可视化脚本（论文手交付）。

口径严格对齐赛题附录 2 与已交叉核对的 cszd 基准解：
  * 水平巡航能耗  E_hor = E_use * d / L_g(q)
  * 爬升附加能耗  E_up  = (M_g0 + q) * g * h_up / (eta_up * 3.6e6)
  * 等效航程      L_g(q) = L0 - (L0 - LF) * (q/Q)^{3/2}
  * 巡航海拔      = 航段沿途 DEM 最高像元 + 50 m（航段级取高）
  * 作业高度      O01 = 地面海拔；服务区 = 地面海拔 + 30 m
  * 往返口径      去程带载 q、回程空载 q=0
  * 安全上限      E_trip <= (1-rho) * E_use
输出：论文/图 下的全部 Q1 图（矢量 PDF + PNG）与 q1_recompute.csv。
"""
from __future__ import annotations
import math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from mpl_toolkits.mplot3d import Axes3D  # noqa
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "_review" / "D" / "code" / "data"
DEM_TIF = (ROOT / "数据" / "镇龙乡地理空间数据" / "镇龙乡及周边地理数据"
           / "数字高程模型数据（DEM）" / "镇龙乡及周边30米DEM.tif")
FIG = ROOT / "论文" / "图"
FIG.mkdir(parents=True, exist_ok=True)

# ---------------- 中文字体 ----------------
for fp in ["/System/Library/Fonts/Hiragino Sans GB.ttc",
           "/System/Library/Fonts/STHeiti Medium.ttc",
           "/System/Library/Fonts/Supplemental/Songti.ttc"]:
    if Path(fp).exists():
        fm.fontManager.addfont(fp)
mpl.rcParams["font.sans-serif"] = ["Hiragino Sans GB", "STHeiti", "Arial Unicode MS"]
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["figure.dpi"] = 120
mpl.rcParams["savefig.bbox"] = "tight"
mpl.rcParams["font.size"] = 9.5
# ---- 全局美学：浅灰背景 + 细网格 + 无上/右框线 ----
mpl.rcParams["axes.facecolor"] = "#FBFBFD"
mpl.rcParams["figure.facecolor"] = "white"
mpl.rcParams["axes.edgecolor"] = "#B0B0B8"
mpl.rcParams["axes.linewidth"] = 0.9
mpl.rcParams["axes.grid"] = True
mpl.rcParams["grid.color"] = "#D8D8E0"
mpl.rcParams["grid.linewidth"] = 0.6
mpl.rcParams["grid.alpha"] = 0.7
mpl.rcParams["axes.axisbelow"] = True
mpl.rcParams["axes.titleweight"] = "bold"
mpl.rcParams["axes.titlesize"] = 10
mpl.rcParams["legend.frameon"] = True
mpl.rcParams["legend.framealpha"] = 0.9
mpl.rcParams["legend.edgecolor"] = "#CCCCCC"
mpl.rcParams["xtick.color"] = "#444444"
mpl.rcParams["ytick.color"] = "#444444"
mpl.rcParams["xtick.direction"] = "out"
mpl.rcParams["ytick.direction"] = "out"


def _despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

G = 9.8
KWH_J = 3.6e6
CRUISE_CLEARANCE = 50.0
OPS_O01 = 0.0
OPS_S = 30.0
DEG_LAT_M = 110574.0
DEG_LON_M_EQ = 111320.0

# ---------------- 数据 ----------------
nodes = pd.read_csv(DATA / "服务区数据.csv").rename(
    columns={"V": "id", "x": "lon", "y": "lat", "h": "elev"}).set_index("id")
utypes = pd.read_csv(DATA / "运输无人机_机型参数.csv")
GT = {}
for _, r in utypes.iterrows():
    GT[r["type"]] = dict(
        M0=float(r["M_g0"]), Q=float(r["Q_g"]), V=float(r["V_g"]),
        L0=float(r["L_0"]), LF=float(r["L_F"]), Euse=float(r["E_use"]),
        rho=float(r["ρ_g"]) / 100.0, eta=float(r["η_up"]),
        v_up=float(r["v_k^up"]), vc=float(r["v_k^cr"]), v_dn=float(r["v_k^down"]),
        t_prep=float(r["T_setup"]), t_load=float(r["T_load"]),
        t_hand=float(r["T_handover"]), t_hand_box=float(r["T_handover_p"]))
SIDS = [f"S{i:03d}" for i in range(1, 16)]
GNAMES = {"A": "A(轻载25kg)", "B": "B(中载30kg)", "C": "C(重载80kg)"}
GCOLOR = {"A": "#4C72B0", "B": "#55A868", "C": "#C44E52"}

demand = pd.read_csv(DATA / "物资需求.csv")

# ---------------- DEM ----------------
im = Image.open(DEM_TIF)
DEM = np.array(im, dtype=np.float64)
NY, NX = DEM.shape
PS = 1.0 / 3600.0
LON0 = 109.03277777777778   # 左上角像元中心经度（tiepoint）
LAT0 = 23.224722222222223   # 左上角像元中心纬度


def dem_elev(lon, lat):
    col = (np.asarray(lon) - LON0) / PS
    row = (LAT0 - np.asarray(lat)) / PS
    r = np.clip(np.round(row).astype(int), 0, NY - 1)
    c = np.clip(np.round(col).astype(int), 0, NX - 1)
    return DEM[r, c]


def planar_m(lon0, lat0, lon1, lat1):
    mlat = (lat0 + lat1) / 2.0
    dx = (lon1 - lon0) * DEG_LON_M_EQ * math.cos(math.radians(mlat))
    dy = (lat1 - lat0) * DEG_LAT_M
    return math.hypot(dx, dy)


def max_elev_path(lon0, lat0, lon1, lat1, step_m=15.0):
    d = planar_m(lon0, lat0, lon1, lat1)
    n = max(int(math.ceil(d / step_m)) + 1, 2)
    t = np.linspace(0, 1, n)
    e = dem_elev(lon0 + (lon1 - lon0) * t, lat0 + (lat1 - lat0) * t)
    return float(np.nanmax(e))


def leg_geom(i, j):
    ri, rj = nodes.loc[i], nodes.loc[j]
    d = planar_m(ri.lon, ri.lat, rj.lon, rj.lat)
    zmax = max_elev_path(ri.lon, ri.lat, rj.lon, rj.lat)
    zc = zmax + CRUISE_CLEARANCE
    oi = ri.elev + (OPS_O01 if i == "O01" else OPS_S)
    oj = rj.elev + (OPS_O01 if j == "O01" else OPS_S)
    return dict(d=d, zc=zc, h_up=max(0.0, zc - oi), h_dn=max(0.0, zc - oj))


def equiv_range(gt, q):
    q = min(max(q, 0.0), gt["Q"])
    return gt["L0"] - (gt["L0"] - gt["LF"]) / gt["Q"] ** 1.5 * q ** 1.5


def leg_energy(g, i, j, q):
    gt = GT[g]
    lg = leg_geom(i, j)
    e_hor = lg["d"] / equiv_range(gt, q) * gt["Euse"]
    e_up = (gt["M0"] + q) * G * lg["h_up"] / gt["eta"] / KWH_J
    return e_hor + e_up


def roundtrip_energy(g, sid, q):
    return leg_energy(g, "O01", sid, q) + leg_energy(g, sid, "O01", 0.0)


def energy_components(g, sid, q):
    gt = GT[g]
    lo, lb = leg_geom("O01", sid), leg_geom(sid, "O01")
    e_hor_out = lo["d"] / equiv_range(gt, q) * gt["Euse"]
    e_up_out = (gt["M0"] + q) * G * lo["h_up"] / gt["eta"] / KWH_J
    e_hor_back = lb["d"] / equiv_range(gt, 0.0) * gt["Euse"]
    e_up_back = gt["M0"] * G * lb["h_up"] / gt["eta"] / KWH_J
    return e_hor_out, e_up_out, e_hor_back, e_up_back


def max_safe_payload(g, sid, rho=None):
    gt = GT[g]
    rho = gt["rho"] if rho is None else rho
    limit = (1 - rho) * gt["Euse"]
    if roundtrip_energy(g, sid, gt["Q"]) <= limit:
        return gt["Q"], "RATED", limit
    if roundtrip_energy(g, sid, 0.0) > limit:
        return 0.0, "INFEASIBLE", limit
    lo, hi = 0.0, gt["Q"]
    for _ in range(100):
        mid = (lo + hi) / 2
        if roundtrip_energy(g, sid, mid) <= limit:
            lo = mid
        else:
            hi = mid
        if hi - lo <= 1e-7:
            break
    return lo, "ENERGY_LIMITED", limit


# ================= 计算全表 =================
rows = []
for sid in SIDS:
    lg = leg_geom("O01", sid)
    for g in ("A", "B", "C"):
        q, st, lim = max_safe_payload(g, sid)
        rows.append(dict(sid=sid, g=g, d=lg["d"], zc=lg["zc"],
                         h_up=lg["h_up"], qmax=q, status=st,
                         e_empty=roundtrip_energy(g, sid, 0.0),
                         e_full=roundtrip_energy(g, sid, GT[g]["Q"]),
                         limit=lim))
df = pd.DataFrame(rows)
df.to_csv(ROOT / "论文" / "q1_recompute.csv", index=False, encoding="utf-8-sig")
n_lim = (df.status == "ENERGY_LIMITED").sum()
print(f"[校验] 组合={len(df)} 额定={(df.status=='RATED').sum()} 能量受限={n_lim}")
print(df[df.status == "ENERGY_LIMITED"][["g", "sid", "qmax"]].to_string(index=False))

QMAX = {(r.g, r.sid): r.qmax for r in df.itertuples()}

# ================= 图 1：最大安全载荷热力图 =================
mat = np.array([[QMAX[(g, s)] for s in SIDS] for g in ("A", "B", "C")])
ratio = mat / np.array([[GT[g]["Q"]] for g in ("A", "B", "C")])
fig, ax = plt.subplots(figsize=(9.4, 3.0))
im1 = ax.imshow(ratio, cmap="RdYlGn", vmin=0.7, vmax=1.0, aspect="auto")
ax.set_xticks(range(15)); ax.set_xticklabels(SIDS, rotation=45, ha="right")
ax.set_yticks(range(3)); ax.set_yticklabels([GNAMES[g] for g in ("A", "B", "C")])
# 白色单元格分隔线
ax.set_xticks(np.arange(-.5, 15, 1), minor=True)
ax.set_yticks(np.arange(-.5, 3, 1), minor=True)
ax.grid(which="minor", color="white", linewidth=1.6)
ax.tick_params(which="minor", length=0)
ax.tick_params(which="major", length=0)
for i in range(3):
    for j in range(15):
        v = mat[i, j]
        lim = ratio[i, j] < 0.999
        txt = f"{v:.1f}" if lim else f"{v:.0f}"
        ax.text(j, i, txt, ha="center", va="center", fontsize=7.5,
                color="#7A0000" if lim else "#1A1A1A",
                fontweight="bold" if lim else "normal")
cb = fig.colorbar(im1, ax=ax, fraction=0.025, pad=0.01)
cb.set_label("最大安全载荷 / 额定载重")
cb.outline.set_linewidth(0.5)
ax.set_title("图1  最大安全载荷矩阵：仅 C 型 5 区 + B 型 S008 受返航能量约束（标红加粗），其余 39 组合均达额定载重", fontsize=9.5)
fig.savefig(FIG / "fig1_maxpayload_heatmap.pdf")
fig.savefig(FIG / "fig1_maxpayload_heatmap.png", dpi=200)
plt.close(fig)

# ================= 图 2：能耗分解堆叠柱（近/中/远各1区）=================
sample = ["S011", "S002", "S008"]  # 近(2.8km) 中(7.4km) 远(8.1km)
fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), sharey=True)
labels = ["去程水平", "去程爬升", "返程水平", "返程爬升"]
cols = ["#4C72B0", "#8FB0DA", "#C44E52", "#E8A2A2"]
for ax, sid in zip(axes, sample):
    gs = ("A", "B", "C")
    bottoms = np.zeros(3)
    comps = np.array([energy_components(g, sid, QMAX[(g, sid)]) for g in gs])
    for k in range(4):
        ax.bar(range(3), comps[:, k], bottom=bottoms, color=cols[k],
               label=labels[k] if sid == sample[0] else None, width=0.6)
        bottoms += comps[:, k]
    for x, g in enumerate(gs):
        ax.plot([x - 0.35, x + 0.35], [(1 - GT[g]["rho"]) * GT[g]["Euse"]] * 2,
                "k--", lw=1)
        ax.text(x, bottoms[x] + 0.1, f"{bottoms[x]:.1f}", ha="center", fontsize=7.5)
    ax.set_xticks(range(3)); ax.set_xticklabels(gs)
    d = leg_geom("O01", sid)["d"] / 1000
    ax.set_title(f"{sid}（{d:.1f} km）", fontsize=9)
    ax.set_xlabel("机型")
    ax.grid(axis="y", alpha=0.35); ax.set_axisbelow(True)
    _despine(ax)
axes[0].set_ylabel("往返能耗 / kWh")
fig.legend(loc="upper center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, 1.02))
fig.suptitle("图2  单架次往返能耗分解：水平巡航能耗占主导，虚线为各机型可用能量安全上限 (1-ρ)E_use", y=1.09, fontsize=9.5)
fig.savefig(FIG / "fig2_energy_breakdown.pdf")
fig.savefig(FIG / "fig2_energy_breakdown.png", dpi=200)
plt.close(fig)

# ================= 图 3：等效航程与往返能耗曲线 =================
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
qq = np.linspace(0, 1, 100)
for g in ("A", "B", "C"):
    gt = GT[g]
    qs = qq * gt["Q"]
    axes[0].plot(qs, [equiv_range(gt, q) / 1000 for q in qs], color=GCOLOR[g], label=GNAMES[g])
axes[0].set_xlabel("载荷 q / kg"); axes[0].set_ylabel("等效航程 L_g(q) / km")
axes[0].set_title("(a) 载荷–等效航程曲线（凹减，满载最短）", fontsize=9)
axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3); _despine(axes[0])
sid = "S008"
for g in ("A", "B", "C"):
    gt = GT[g]
    qs = np.linspace(0, gt["Q"], 100)
    axes[1].plot(qs, [roundtrip_energy(g, sid, q) for q in qs], color=GCOLOR[g], label=GNAMES[g])
    axes[1].axhline((1 - gt["rho"]) * gt["Euse"], color=GCOLOR[g], ls=":", lw=1)
    qm = QMAX[(g, sid)]
    if qm < gt["Q"] - 1e-3:
        axes[1].plot(qm, roundtrip_energy(g, sid, qm), "o", color=GCOLOR[g], ms=6)
        axes[1].annotate(f"q_max={qm:.1f}", (qm, roundtrip_energy(g, sid, qm)),
                         textcoords="offset points", xytext=(-10, 8), fontsize=7.5, color=GCOLOR[g])
axes[1].set_xlabel("载荷 q / kg"); axes[1].set_ylabel("往返能耗 / kWh")
axes[1].set_title(f"(b) {sid} 往返能耗随载荷单调增（点线为安全上限）", fontsize=9)
axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3); _despine(axes[1])
fig.suptitle("图3  能量安全边界的几何：载重越大等效航程越短、往返能耗越高，交点即最大安全载荷", y=1.02, fontsize=9.5)
fig.savefig(FIG / "fig3_range_energy_curve.pdf")
fig.savefig(FIG / "fig3_range_energy_curve.png", dpi=200)
plt.close(fig)

# ================= 图 4：三维 DEM 地形（山体阴影光照）+ 地面投影等高地毯 + 航线 =================
from matplotlib.colors import LightSource, LinearSegmentedColormap
try:
    from scipy.ndimage import gaussian_filter
except Exception:
    gaussian_filter = None
# 裁剪到节点包络
lons = nodes.lon.values; lats = nodes.lat.values
lon_min, lon_max = lons.min() - 0.012, lons.max() + 0.012
lat_min, lat_max = lats.min() - 0.012, lats.max() + 0.012
c0 = int((lon_min - LON0) / PS); c1 = int((lon_max - LON0) / PS)
r0 = int((LAT0 - lat_max) / PS); r1 = int((LAT0 - lat_min) / PS)
sub = DEM[r0:r1, c0:c1].astype(np.float64)
step = 2
sub = sub[::step, ::step]
# 轻度平滑，抑制 30m DEM 逐像元噪声，山脊更连贯（不改变量级）
if gaussian_filter is not None:
    sub = gaussian_filter(sub, sigma=1.1)
gy, gx = sub.shape
LON = LON0 + (c0 + np.arange(gx) * step) * PS
LAT = LAT0 - (r0 + np.arange(gy) * step) * PS
LONg, LATg = np.meshgrid(LON, LAT)

# 高级配色：低地绿→黄褐→岩灰白（大地色带），叠加山体阴影
terr = LinearSegmentedColormap.from_list("kterrain", [
    "#2E6E43", "#5B9A55", "#A7C267", "#E4D08A",
    "#C9A26B", "#9C7A55", "#B9AFA6", "#F2F0EE"])
zmin, zmax = float(np.nanmin(sub)), float(np.nanmax(sub))
norm = plt.Normalize(zmin, zmax)
ls = LightSource(azdeg=315, altdeg=45)
rgb = ls.shade(sub, cmap=terr, norm=norm, blend_mode="soft", vert_exag=2.0)

o = nodes.loc["O01"]
fig = plt.figure(figsize=(14, 6.4))

# ---- (a) 三维山体阴影地形（纯观赏，展示地形起伏与光照）----
from matplotlib import patheffects as pe
ax = fig.add_subplot(1, 2, 1, projection="3d")
ax.set_box_aspect((1, 1, 0.55))
ax.computed_zorder = False  # 关闭自动深度排序，使点位/编号严格置顶不被山体遮挡
ax.plot_surface(LONg, LATg, sub, facecolors=rgb, rstride=1, cstride=1,
                linewidth=0, antialiased=False, shade=False, zorder=0)
# 服务点：从地表竖杆上抬 + 点 + 编号，全部 zorder 高于曲面
zspan3d = zmax - zmin
for sid in SIDS:
    s = nodes.loc[sid]
    zs = float(dem_elev(s.lon, s.lat))
    z_top = zs + 0.12 * zspan3d
    lim3d = df[(df.g == "C") & (df.sid == sid)].status.iloc[0] == "ENERGY_LIMITED"
    ax.plot([s.lon, s.lon], [s.lat, s.lat], [zs, z_top],
            color="black", lw=0.7, alpha=0.7, zorder=5)
    ax.scatter(s.lon, s.lat, z_top, color="#D62728" if lim3d else "#F5F5F5",
               s=48, edgecolor="#7A0000" if lim3d else "#333", linewidth=1.0,
               depthshade=False, zorder=6)
    ax.text(s.lon, s.lat, z_top + 0.035 * zspan3d, sid[1:], fontsize=8,
            color="#111", ha="center", va="bottom", fontweight="bold", zorder=7,
            path_effects=[pe.withStroke(linewidth=2.0, foreground="white")])
# O01 调度中心（金边黑星）
zo3d = float(dem_elev(o.lon, o.lat))
zo3d_top = zo3d + 0.12 * zspan3d
ax.plot([o.lon, o.lon], [o.lat, o.lat], [zo3d, zo3d_top],
        color="black", lw=0.9, alpha=0.75, zorder=5)
ax.scatter(o.lon, o.lat, zo3d_top, color="#111", s=180, marker="*",
           edgecolor="gold", linewidth=1.3, depthshade=False, zorder=6)
ax.text(o.lon, o.lat, zo3d_top + 0.035 * zspan3d, "O01", fontsize=8.5,
        color="#111", ha="center", va="bottom", fontweight="bold", zorder=7,
        path_effects=[pe.withStroke(linewidth=2.0, foreground="white")])
ax.set_xlabel("经度 / °", labelpad=6, fontsize=8.5)
ax.set_ylabel("纬度 / °", labelpad=6, fontsize=8.5)
ax.set_zlabel("海拔 / m", labelpad=4, fontsize=8.5)
ax.tick_params(labelsize=7.5)
ax.set_zlim(zmin, zmax + 0.22 * zspan3d)
ax.view_init(elev=34, azim=-58)
for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
    pane.pane.set_facecolor((1, 1, 1, 0.0))
    pane.pane.set_edgecolor((0.82, 0.82, 0.86, 0.5))
ax.grid(True, color="#DDDDE3", linewidth=0.4)
ax.set_title("(a) 三维山体阴影地形 + 服务点位（红=C型受限，白=达额定）", fontsize=9.5, pad=0)

# ---- (b) 二维山体阴影平面图 + 服务点/航线/标签（深度排序完美，全部可读）----
ax2 = fig.add_subplot(1, 2, 2)
extent = [LON.min(), LON.max(), LAT.min(), LAT.max()]
ax2.imshow(rgb, extent=extent, origin="upper", aspect="auto", zorder=1)
cs = ax2.contour(LONg, LATg, sub, levels=12, colors="white",
                 linewidths=0.4, alpha=0.45, zorder=2)
# 航线
for sid in SIDS:
    s = nodes.loc[sid]
    ax2.plot([o.lon, s.lon], [o.lat, s.lat], color="#12345B",
             lw=1.1, alpha=0.85, zorder=3)
# 服务点 + 标签
for sid in SIDS:
    s = nodes.loc[sid]
    lim = df[(df.g == "C") & (df.sid == sid)].status.iloc[0] == "ENERGY_LIMITED"
    ax2.scatter(s.lon, s.lat, color="#D62728" if lim else "#F5F5F5",
                s=70, edgecolor="#7A0000" if lim else "#333", linewidth=1.0,
                zorder=5)
    ax2.annotate(sid[1:], (s.lon, s.lat), textcoords="offset points",
                 xytext=(6, 5), fontsize=7.5, fontweight="bold",
                 color="#111", zorder=6,
                 path_effects=None)
# O01
ax2.scatter(o.lon, o.lat, color="#111", s=260, marker="*",
            edgecolor="gold", linewidth=1.4, zorder=7)
ax2.annotate("O01 调度中心", (o.lon, o.lat), textcoords="offset points",
             xytext=(8, -14), fontsize=9, fontweight="bold", color="#111", zorder=8)
ax2.set_xlabel("经度 / °"); ax2.set_ylabel("纬度 / °")
ax2.set_title("(b) 二维地形俯视：红点=C型受返航能量约束区，白点=达额定区", fontsize=9.5)
ax2.grid(False)
# 颜色条
mappable = plt.cm.ScalarMappable(norm=norm, cmap=terr); mappable.set_array([])
cb = fig.colorbar(mappable, ax=ax2, fraction=0.046, pad=0.02)
cb.set_label("地面海拔 / m", fontsize=9.5); cb.outline.set_linewidth(0.5)

fig.suptitle("图4  镇龙乡 30m DEM 山体阴影地形与 O01→Si 单点往返航线（左：三维观赏；右：二维俯视含节点标注）",
             fontsize=10.5, y=0.99)
fig.savefig(FIG / "fig4_dem3d_routes.pdf")
fig.savefig(FIG / "fig4_dem3d_routes.png", dpi=200)
plt.close(fig)

# ================= 图 5：ρ 敏感性曲线 =================
rhos = np.linspace(0.10, 0.40, 31)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
crit_sids = ["S008", "S004", "S002", "S012", "S003"]
for sid in crit_sids:
    ys = [max_safe_payload("C", sid, rho=r)[0] for r in rhos]
    axes[0].plot(rhos * 100, ys, lw=1.8, label=sid)
axes[0].axvline(20, color="gray", ls="--", lw=1); axes[0].text(20.3, 40, "基准 ρ=20%", fontsize=7.5)
axes[0].axhline(80, color="k", ls=":", lw=0.8); axes[0].text(10.3, 81, "C 型额定 80kg", fontsize=7.5)
axes[0].set_xlabel("返航安全余量 ρ / %"); axes[0].set_ylabel("C 型最大安全载荷 / kg")
axes[0].set_title("(a) C 型受限服务区的 q_max 随 ρ 递减", fontsize=9)
axes[0].legend(fontsize=7.5, ncol=2); axes[0].grid(alpha=0.3); _despine(axes[0])
# 全局最少架次随 rho 变化（各区独立最少架次的下界）
def min_sorties_all(rho):
    total = 0
    for sid in SIDS:
        mass = demand[demand.service == sid]
        w = (mass.total_boxes * mass.mass_per_box).sum()
        qc = max_safe_payload("C", sid, rho=rho)[0]
        total += max(1, math.ceil(w / qc)) if qc > 0 else 99
    return total
ys = [min_sorties_all(r) for r in rhos]
axes[1].step(rhos * 100, ys, where="mid", color="#C44E52", lw=1.8)
axes[1].axvline(20, color="gray", ls="--", lw=1)
axes[1].text(20.3, 19, "基准 18 架次", fontsize=7.5)
axes[1].set_xlabel("返航安全余量 ρ / %"); axes[1].set_ylabel("单点组批最少架次数下界")
axes[1].set_title("(b) 架次数下界对 ρ 的阶梯响应", fontsize=9)
axes[1].set_ylim(16, 60)  # 聚焦有效区间，ρ>37% 后急升越界不再展示
axes[1].grid(alpha=0.3); _despine(axes[1])
fig.suptitle("图5  返航安全余量敏感性：q_max 连续下降、组批架次阶梯式跳变", y=1.02, fontsize=9.5)
fig.savefig(FIG / "fig5_rho_sensitivity.pdf")
fig.savefig(FIG / "fig5_rho_sensitivity.png", dpi=200)
plt.close(fig)

# ================= 图 6：组批方案架次结构 + 载荷/体积占用率 =================
plan = pd.read_csv(DATA / "Q1_mixed_N_opt_plan.csv")
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
# (a) 每区架次数与机型
by_area = plan.groupby("service")
xs = np.arange(15)
cW = {"B": 0, "C": 0}
nb = plan.groupby(["service", "type"]).size().unstack(fill_value=0).reindex(SIDS).fillna(0)
b1 = axes[0].bar(xs, nb.get("C", pd.Series(0, index=SIDS)), color=GCOLOR["C"], label="C 型架次")
b2 = axes[0].bar(xs, nb.get("B", pd.Series(0, index=SIDS)),
                 bottom=nb.get("C", pd.Series(0, index=SIDS)), color=GCOLOR["B"], label="B 型架次")
axes[0].set_xticks(xs); axes[0].set_xticklabels(SIDS, rotation=45, ha="right")
axes[0].set_ylabel("架次数"); axes[0].legend(fontsize=8)
axes[0].set_title("(a) 最少架次(9C+9B=18)组批结构", fontsize=9)
axes[0].grid(axis="y", alpha=0.35); axes[0].set_axisbelow(True); _despine(axes[0])
# (b) 每架次载荷占用率与体积占用率
plan2 = plan.copy()
plan2["load_ratio"] = plan2.mass_kg / plan2["safe_payload_kg"]
plan2["vol_ratio"] = plan2.volume_m3 / plan2.type.map({g: GT[g]["V"] for g in GT})
axes[1].scatter(plan2.load_ratio * 100, plan2.vol_ratio * 100,
                c=[GCOLOR[t] for t in plan2.type], s=70, edgecolor="white", lw=0.8,
                alpha=0.9, zorder=3)
axes[1].axvline(100, color="gray", ls="--", lw=0.8)
axes[1].axhline(100, color="gray", ls="--", lw=0.8)
axes[1].set_xlabel("载荷占用率 = 装载质量 / 最大安全载荷 /%")
axes[1].set_ylabel("体积占用率 = 装载体积 / 舱容 /%")
axes[1].set_title("(b) 各架次受质量约束主导，体积普遍富余", fontsize=9)
axes[1].set_xlim(20, 110); axes[1].set_ylim(20, 110); axes[1].grid(alpha=0.3)
_despine(axes[1])
from matplotlib.lines import Line2D
axes[1].legend(handles=[Line2D([0], [0], marker="o", color="w", markerfacecolor=GCOLOR[g],
               markersize=8, label=f"{g} 型") for g in ("B", "C")], fontsize=8)
fig.suptitle("图6  货箱组批方案：18 架次(9C+9B)可行，能力瓶颈是安全载荷而非舱容", y=1.02, fontsize=9.5)
fig.savefig(FIG / "fig6_batching.pdf")
fig.savefig(FIG / "fig6_batching.png", dpi=200)
plt.close(fig)

print("图已输出至", FIG)
