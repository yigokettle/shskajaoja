# -*- coding: utf-8 -*-
"""导出「图4」所用的全部数据。

图4 = 左三维山体阴影地形 + 右二维俯视(节点/航线/受限标注)。
其数据可分为三块，全部落盘到 论文/图4数据/ ：
  1) fig4_nodes.csv      —— O01 与 15 个服务区坐标、地面海拔、C型最大安全载荷与受限状态
  2) fig4_routes.csv     —— 15 条 O01→Si 航线端点（右图放射线）
  3) fig4_dem_meta.txt   —— DEM 裁剪窗口、栅格分辨率、山体阴影/配色渲染参数
并另存裁剪后的高程栅格 fig4_dem_grid.npz（含经纬网格），便于原样复现。
"""
from __future__ import annotations
import math
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "_review" / "D" / "code" / "data"
DEM_TIF = (ROOT / "数据" / "镇龙乡地理空间数据" / "镇龙乡及周边地理数据"
           / "数字高程模型数据（DEM）" / "镇龙乡及周边30米DEM.tif")
OUT = ROOT / "论文" / "图4数据"
OUT.mkdir(parents=True, exist_ok=True)

G = 9.8
KWH_J = 3.6e6
CRUISE_CLEARANCE = 50.0
OPS_O01, OPS_S = 0.0, 30.0
DEG_LAT_M, DEG_LON_M_EQ = 110574.0, 111320.0

# ---- 节点与机型 ----
nodes = pd.read_csv(DATA / "服务区数据.csv").rename(
    columns={"V": "id", "x": "lon", "y": "lat", "h": "elev"}).set_index("id")
utypes = pd.read_csv(DATA / "运输无人机_机型参数.csv")
GT = {}
for _, r in utypes.iterrows():
    GT[r["type"]] = dict(M0=float(r["M_g0"]), Q=float(r["Q_g"]), V=float(r["V_g"]),
                         L0=float(r["L_0"]), LF=float(r["L_F"]), Euse=float(r["E_use"]),
                         rho=float(r["ρ_g"]) / 100.0, eta=float(r["η_up"]))
SIDS = [f"S{i:03d}" for i in range(1, 16)]

# ---- DEM ----
im = Image.open(DEM_TIF)
DEM = np.array(im, dtype=np.float64)
NY, NX = DEM.shape
PS = 1.0 / 3600.0
LON0, LAT0 = 109.03277777777778, 23.224722222222223


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


def equiv_range(gt, q):
    q = min(max(q, 0.0), gt["Q"])
    return gt["L0"] - (gt["L0"] - gt["LF"]) / gt["Q"] ** 1.5 * q ** 1.5


def roundtrip_energy(g, sid, q):
    gt = GT[g]
    ri, rj = nodes.loc["O01"], nodes.loc[sid]
    d = planar_m(ri.lon, ri.lat, rj.lon, rj.lat)
    zc = max_elev_path(ri.lon, ri.lat, rj.lon, rj.lat) + CRUISE_CLEARANCE
    oi = ri.elev + OPS_O01
    oj = rj.elev + OPS_S
    h_out = max(0.0, zc - oi)
    h_back = max(0.0, zc - oj)
    e = d / equiv_range(gt, q) * gt["Euse"] + (gt["M0"] + q) * G * h_out / gt["eta"] / KWH_J
    e += d / equiv_range(gt, 0.0) * gt["Euse"] + gt["M0"] * G * h_back / gt["eta"] / KWH_J
    return e, d, zc


def max_safe_payload_C(sid):
    gt = GT["C"]
    limit = (1 - gt["rho"]) * gt["Euse"]
    if roundtrip_energy("C", sid, gt["Q"])[0] <= limit:
        return gt["Q"], "RATED"
    lo, hi = 0.0, gt["Q"]
    for _ in range(100):
        mid = (lo + hi) / 2
        if roundtrip_energy("C", sid, mid)[0] <= limit:
            lo = mid
        else:
            hi = mid
    return lo, "ENERGY_LIMITED"


# ---- 1) 节点表 ----
rows = []
o = nodes.loc["O01"]
rows.append(dict(id="O01", 角色="调度中心", lon=o.lon, lat=o.lat, 地面海拔_m=o.elev,
                 距O01水平距离_km=0.0, 航段巡航海拔_m="", C型最大安全载荷_kg="", C型状态="", 图中标记="★金星"))
for sid in SIDS:
    s = nodes.loc[sid]
    _, d, zc = roundtrip_energy("C", sid, GT["C"]["Q"])
    qc, st = max_safe_payload_C(sid)
    rows.append(dict(id=sid, 角色="服务区", lon=s.lon, lat=s.lat, 地面海拔_m=s.elev,
                     距O01水平距离_km=round(d / 1000, 3), 航段巡航海拔_m=round(zc, 1),
                     C型最大安全载荷_kg=round(qc, 2), C型状态=st,
                     图中标记="●红点(受限)" if st == "ENERGY_LIMITED" else "○白点(达额定)"))
nodes_out = pd.DataFrame(rows)
nodes_out.to_csv(OUT / "fig4_nodes.csv", index=False, encoding="utf-8-sig")

# ---- 2) 航线表 ----
route_rows = []
for sid in SIDS:
    s = nodes.loc[sid]
    _, d, zc = roundtrip_energy("C", sid, GT["C"]["Q"])
    route_rows.append(dict(航线=f"O01->{sid}", 起点lon=o.lon, 起点lat=o.lat,
                           终点lon=s.lon, 终点lat=s.lat, 水平距离_km=round(d / 1000, 3),
                           巡航海拔_m=round(zc, 1)))
pd.DataFrame(route_rows).to_csv(OUT / "fig4_routes.csv", index=False, encoding="utf-8-sig")

# ---- 3) DEM 裁剪 + 渲染参数 ----
lons, lats = nodes.lon.values, nodes.lat.values
lon_min, lon_max = lons.min() - 0.012, lons.max() + 0.012
lat_min, lat_max = lats.min() - 0.012, lats.max() + 0.012
c0 = int((lon_min - LON0) / PS); c1 = int((lon_max - LON0) / PS)
r0 = int((LAT0 - lat_max) / PS); r1 = int((LAT0 - lat_min) / PS)
step = 2
sub = DEM[r0:r1, c0:c1].astype(np.float64)[::step, ::step]
gy, gx = sub.shape
LON = LON0 + (c0 + np.arange(gx) * step) * PS
LAT = LAT0 - (r0 + np.arange(gy) * step) * PS
# 保存裁剪栅格(平滑前的原始高程)与经纬轴，供原样复现
np.savez_compressed(OUT / "fig4_dem_grid.npz", elev=sub, lon=LON, lat=LAT)

meta = f"""图4 数据说明
============================================================
来源文件
  - 节点坐标 : _review/D/code/data/服务区数据.csv (O01 + S001..S015)
  - 机型参数 : _review/D/code/data/运输无人机_机型参数.csv (A/B/C)
  - 高程栅格 : 数据/.../镇龙乡及周边30米DEM.tif

DEM 原始栅格
  - 尺寸(行×列) : {NY} × {NX}
  - 像元分辨率   : {PS:.8f}° (=1/3600°, 约30 m)
  - 左上角像元中心(tiepoint) : 经度 {LON0}, 纬度 {LAT0}
  - 像元中心经度 = LON0 + col*PS ; 纬度 = LAT0 - row*PS

图4 裁剪窗口(节点包络外扩 0.012°)
  - 经度范围 : [{LON.min():.6f}, {LON.max():.6f}]
  - 纬度范围 : [{LAT.min():.6f}, {LAT.max():.6f}]
  - 行列裁剪 : row[{r0}:{r1}], col[{c0}:{c1}]
  - 降采样步长 step = {step} (每2像元取1)
  - 裁剪后网格 : {gy} 行 × {gx} 列 (见 fig4_dem_grid.npz: elev/lon/lat)

渲染参数(左三维 + 右二维山体阴影)
  - 平滑        : scipy.ndimage.gaussian_filter(sigma=1.1) 仅用于显示，不改高程量级
  - 配色 kterrain: #2E6E43 → #5B9A55 → #A7C267 → #E4D08A → #C9A26B → #9C7A55 → #B9AFA6 → #F2F0EE
  - 山体阴影    : matplotlib LightSource(azdeg=315, altdeg=45), blend_mode='soft', vert_exag=2.0
  - 高程归一化  : Normalize(vmin={float(np.nanmin(sub)):.1f}, vmax={float(np.nanmax(sub)):.1f}) m
  - 三维视角    : elev=34, azim=-58, box_aspect=(1,1,0.55)

派生量口径(fig4_nodes.csv / fig4_routes.csv)
  - 水平距离 d  : 等经纬平面近似, 纬向110574 m/°, 经向111320*cos(mlat) m/°
  - 巡航海拔 zc : 航段直线沿途 DEM 最高像元 + 50 m 净空
  - 爬升高度    : 去程 zc-(O01地面) ; 返程 zc-(服务区地面+30 m)
  - C型最大安全载荷: 二分求 E_trip(q) = (1-ρ)·E_use, ρ=0.20, E_use(C)=8.0 kWh, 上限6.4 kWh
  - 受限判定    : q_max<80kg → ENERGY_LIMITED(红点) ; =80kg → RATED(白点)
============================================================
"""
(OUT / "fig4_dem_meta.txt").write_text(meta, encoding="utf-8")

print("已导出到", OUT)
print(nodes_out.to_string(index=False))
"""受限服务区(红点):"""
print("\n红点(C型受限):",
      list(nodes_out[nodes_out.C型状态 == "ENERGY_LIMITED"].id))
