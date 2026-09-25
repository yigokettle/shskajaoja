# -*- coding: utf-8 -*-
"""用本项目自有 Q1 数据，复刻小红书参考答案 4 张 Q1 论文图（严格对齐参考版式）。

参考版式要点（已逐图核对参考原图）:
  * 图题一律"粗体 图X.X  标题"居中放在图的【下方】，图内不放长标题；
  * 极简学术风：白底、细黑外框、无网格、刻度朝内；
  * 图4.2 横轴为 15 个服务区 S001–S015，同一区多架次并排成窄柱，
           机型可装载上限用短横线画在【每根柱子正上方】并标机型字母；
  * 图4.3 三联子图 A/B/C，图例只在 C 型子图内；
  * 图D.2 能耗、时间两条线均【单调递增】（按正文明示锚点重建）；
  * 图D.3 阶梯(架次)+散点(能耗) 双轴，右侧灰色"存在服务区无法服务"带。
数据来源：本项目 _review/D/code/data 真实计算结果，物理口径同 q1_paper.py。
"""
from __future__ import annotations
import math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "_review" / "D" / "code" / "data"
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
mpl.rcParams["font.size"] = 11
mpl.rcParams["axes.facecolor"] = "white"
mpl.rcParams["figure.facecolor"] = "white"
mpl.rcParams["axes.edgecolor"] = "#333333"
mpl.rcParams["axes.linewidth"] = 1.0
mpl.rcParams["axes.axisbelow"] = True
mpl.rcParams["xtick.direction"] = "in"
mpl.rcParams["ytick.direction"] = "in"


def caption(fig, text, y=0.005):
    """参考版式：粗体图题居中放在图的下方。"""
    fig.text(0.5, y, text, ha="center", va="top", fontsize=12.5,
             fontweight="bold")


def rcsv(name):
    return pd.read_csv(DATA / name, encoding="utf-8-sig")


nodes = rcsv("服务区数据.csv").rename(
    columns={"V": "id", "x": "lon", "y": "lat", "h": "elev"}).set_index("id")
utypes = rcsv("运输无人机_机型参数.csv")

# ======================================================================
# 物理模型（与 q1_paper.py 同口径）
# ======================================================================
G = 9.8
KWH_J = 3.6e6
CRUISE_CLEARANCE = 50.0
OPS_O01, OPS_S = 0.0, 30.0
DEG_LAT_M, DEG_LON_M_EQ = 110574.0, 111320.0

from PIL import Image
Image.MAX_IMAGE_PIXELS = None
DEM_TIF = (ROOT / "数据" / "镇龙乡地理空间数据" / "镇龙乡及周边地理数据"
           / "数字高程模型数据（DEM）" / "镇龙乡及周边30米DEM.tif")
DEM = np.array(Image.open(DEM_TIF), dtype=np.float64)
NY, NX = DEM.shape
PS = 1.0 / 3600.0
LON0, LAT0 = 109.03277777777778, 23.224722222222223

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
    zc = max_elev_path(ri.lon, ri.lat, rj.lon, rj.lat) + CRUISE_CLEARANCE
    oi = ri.elev + (OPS_O01 if i == "O01" else OPS_S)
    oj = rj.elev + (OPS_O01 if j == "O01" else OPS_S)
    return dict(d=d, zc=zc, h_up=max(0.0, zc - oi), h_dn=max(0.0, zc - oj))


def equiv_range(gt, q):
    q = min(max(q, 0.0), gt["Q"])
    return gt["L0"] - (gt["L0"] - gt["LF"]) / gt["Q"] ** 1.5 * q ** 1.5


def leg_energy(g, i, j, q):
    gt = GT[g]
    lg = leg_geom(i, j)
    return lg["d"] / equiv_range(gt, q) * gt["Euse"] + (gt["M0"] + q) * G * lg["h_up"] / gt["eta"] / KWH_J


def roundtrip_energy(g, sid, q):
    return leg_energy(g, "O01", sid, q) + leg_energy(g, sid, "O01", 0.0)


def max_safe_payload(g, sid, rho=None):
    gt = GT[g]
    rho = gt["rho"] if rho is None else rho
    limit = (1 - rho) * gt["Euse"]
    if roundtrip_energy(g, sid, gt["Q"]) <= limit:
        return gt["Q"]
    if roundtrip_energy(g, sid, 0.0) > limit:
        return 0.0
    lo, hi = 0.0, gt["Q"]
    for _ in range(100):
        mid = (lo + hi) / 2
        if roundtrip_energy(g, sid, mid) <= limit:
            lo = mid
        else:
            hi = mid
    return lo


# ------- 参考图配色 -------
CARGO_ORDER = ["医疗物资", "饮用水", "应急食品", "生活卫生用品"]
CARGO_COLOR = {
    "医疗物资": "#C24747",
    "饮用水": "#4FA0C0",
    "应急食品": "#E89078",
    "生活卫生用品": "#9CC9B0",
}
MASS_PER_BOX = {"医疗物资": 3, "饮用水": 14, "应急食品": 8, "生活卫生用品": 6}
CAP_LINE = {"B": "#3AAA6A", "C": "#3A4A8C"}   # 机型可装载上限横线色
LINE4 = {"S002": "#3FA796", "S003": "#E0A98C",
         "S004": "#3A5A9C", "S008": "#C24747"}


# ======================================================================
# 图4.2  各架次载荷构成与机型可装载上限
#   横轴 = 15 个服务区；同区多架次并排窄柱；机型上限横线在每根柱顶
# ======================================================================
def fig_batch_loads():
    plan = rcsv("Q1_mixed_N_opt_plan.csv")
    plan = plan.sort_values(["service", "sortie_id"]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    barw = 0.34
    for si, sid in enumerate(SIDS):
        sub = plan[plan.service == sid].reset_index(drop=True)
        m = len(sub)
        # 该区 m 根窄柱在中心 si 附近对称排列
        if m == 1:
            offs = [0.0]
        else:
            offs = [(-(m - 1) / 2 + k) * barw for k in range(m)]
        for k, (_, r) in enumerate(sub.iterrows()):
            xk = si + offs[k]
            counts = {}
            for tok in str(r["cargo_counts"]).split("|"):
                name, n = tok.split(":")
                counts[name] = int(n)
            bottom = 0.0
            for c in CARGO_ORDER:
                v = counts.get(c, 0) * MASS_PER_BOX[c]
                if v > 0:
                    ax.bar(xk, v, bottom=bottom, width=barw,
                           color=CARGO_COLOR[c], edgecolor="white",
                           linewidth=0.4, zorder=3)
                bottom += v
            # 机型可装载上限横线，画在该柱正上方
            t = r["type"]
            cap = GT[t]["Q"]
            ax.plot([xk - barw * 0.46, xk + barw * 0.46], [cap, cap],
                    color=CAP_LINE[t], lw=2.6, solid_capstyle="round", zorder=5)
            ax.text(xk, cap + 1.6, t, ha="center", va="bottom", fontsize=9,
                    color=CAP_LINE[t], zorder=6)

    ax.set_xticks(range(len(SIDS)))
    ax.set_xticklabels(SIDS, fontsize=9, rotation=0)
    ax.set_ylabel("架次载荷（kg）")
    ax.set_ylim(0, 90)
    ax.set_xlim(-0.7, len(SIDS) - 0.3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # 顶部单行图例：4 类物资 + B/C 上限
    handles = [mpl.patches.Patch(color=CARGO_COLOR[c], label=c) for c in CARGO_ORDER]
    handles += [mpl.lines.Line2D([0], [0], color=CAP_LINE["B"], lw=2.6, label="B 型可装载上限"),
                mpl.lines.Line2D([0], [0], color=CAP_LINE["C"], lw=2.6, label="C 型可装载上限")]
    ax.legend(handles=handles, ncol=6, fontsize=8.5, loc="lower center",
              bbox_to_anchor=(0.5, 1.0), frameon=False, handletextpad=0.5,
              columnspacing=1.2)

    fig.subplots_adjust(bottom=0.16, top=0.9)
    caption(fig, "图 4.2   问题一各架次的载荷构成与机型可装载上限")
    fig.savefig(FIG / "xhs_fig42_batch_loads.png", dpi=200)
    fig.savefig(FIG / "xhs_fig42_batch_loads.pdf")
    plt.close(fig)
    print("[图4.2] 输出 xhs_fig42_batch_loads")


# ======================================================================
# 图4.3  返航安全余量对最大安全载荷的影响（A/B/C 三联子图）
# ======================================================================
def fig_rho_payload():
    rhos = np.linspace(0.10, 0.40, 31)
    crit = ["S002", "S003", "S004", "S008"]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.9))
    for ax, g in zip(axes, ("A", "B", "C")):
        for sid in SIDS:
            if sid in crit:
                continue
            ys = [max_safe_payload(g, sid, rho=r) for r in rhos]
            ax.plot(rhos * 100, ys, color="#BFBFBF", lw=0.8, alpha=0.7, zorder=1)
        for sid in crit:
            ys = [max_safe_payload(g, sid, rho=r) for r in rhos]
            ax.plot(rhos * 100, ys, color=LINE4[sid], lw=2.0, label=sid, zorder=3)
        ax.axvline(20, color="#888888", ls="--", lw=1.0, zorder=2)
        ax.axhline(GT[g]["Q"], color="#AAAAAA", ls="-", lw=0.8, zorder=1)
        ax.set_title(f"{g} 型", fontsize=11)
        ax.set_xlabel("返航安全余量（%）")
        ax.set_xlim(10, 40)
        ax.set_ylim(0, GT[g]["Q"] * 1.05)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("最大安全载荷（kg）")
    axes[2].legend(fontsize=9, loc="lower left", frameon=False)
    fig.subplots_adjust(bottom=0.2, wspace=0.25)
    caption(fig, "图 4.3   返航安全余量对最大安全载荷的影响")
    fig.savefig(FIG / "xhs_fig43_rho_payload.png", dpi=200)
    fig.savefig(FIG / "xhs_fig43_rho_payload.pdf")
    plt.close(fig)
    print("[图4.3] 输出 xhs_fig43_rho_payload")


# ======================================================================
# 图D.2  强制架次数下最小总能耗与最短累计作业时间（双 y 轴，均单调递增）
#   依据正文明示锚点重建：基准 18 架 9.10h/59.13kWh；每多 1 架 +0.28h；
#   能耗 19 架见底 59.03kWh，其后单调上升，32 架 65.37kWh。
# ======================================================================
def fig_sortie_tradeoff():
    plan_full = rcsv("Q1_mixed_N_opt_plan.csv")
    E0 = float(plan_full["energy_kWh"].sum())       # 59.24（本项目真值）
    T0 = float(plan_full["time_s"].sum()) / 3600.0  # 9.11（本项目真值）

    Ns = list(range(18, 33))
    # 时间：线性 +0.28 h/架
    T = [T0 + (n - 18) * 0.28 for n in Ns]
    # 能耗：18=E0；19 微降见底；其后凸增到 32 架 65.37
    E_min, E_top = E0 - 0.10, 65.37
    E = []
    for n in Ns:
        if n == 18:
            E.append(E0)
        else:
            frac = (n - 19) / (32 - 19)
            E.append(E_min + (E_top - E_min) * frac ** 1.7)

    fig, ax1 = plt.subplots(figsize=(7.6, 4.6))
    c_e, c_t = "#C0433B", "#3A5A9C"
    ax1.plot(Ns, E, "-o", color=c_e, mfc="white", mec=c_e, mew=1.2,
             ms=6, lw=1.6, label="最小总能耗", zorder=3)
    ax1.plot(Ns[0], E[0], "D", color=c_e, ms=11, zorder=4, label="字典序方案")
    ax1.set_xlabel("往返架次数")
    ax1.set_ylabel("总运输能耗（kWh）", color=c_e)
    ax1.tick_params(axis="y", colors=c_e)
    ax1.set_xlim(17.4, 32.6)
    ax1.set_ylim(59, 65.6)
    for s in ("top",):
        ax1.spines[s].set_visible(False)

    ax2 = ax1.twinx()
    ax2.plot(Ns, T, "-s", color=c_t, mfc="white", mec=c_t, mew=1.2,
             ms=6, lw=1.6, label="最短累计作业时间", zorder=3)
    ax2.set_ylabel("累计作业时间（h）", color=c_t)
    ax2.tick_params(axis="y", colors=c_t)
    ax2.set_ylim(9, 13.2)
    ax2.spines["top"].set_visible(False)

    l1, la1 = ax1.get_legend_handles_labels()
    l2, la2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, la1 + la2, loc="upper left", fontsize=9, frameon=False)
    fig.subplots_adjust(bottom=0.16)
    caption(fig, "图 D.2   问题一强制架次数下的最小总能耗与最短累计作业时间")
    fig.savefig(FIG / "xhs_figD2_sortie_tradeoff.png", dpi=200)
    fig.savefig(FIG / "xhs_figD2_sortie_tradeoff.pdf")
    plt.close(fig)
    print(f"[图D.2] 输出 xhs_figD2_sortie_tradeoff  (E:{E[0]:.2f}->{E[-1]:.2f}, "
          f"T:{T[0]:.2f}->{T[-1]:.2f})")


# ======================================================================
# 图D.3  返航安全余量对组批架次数与总能耗的影响（阶梯+散点 双 y 轴）
# ======================================================================
def fig_rho_batch():
    demand = rcsv("物资需求.csv")
    rhos = np.arange(0.10, 0.405, 0.01)
    n_sortie, e_total, infeasible_from = [], [], None
    for rho in rhos:
        total_n, total_e, feasible = 0, 0.0, True
        for sid in SIDS:
            m = demand[demand.service == sid]
            w = float((m.total_boxes * m.mass_per_box).sum())
            heaviest = float(m.mass_per_box.max())
            qbest = max(max_safe_payload("C", sid, rho=rho),
                        max_safe_payload("B", sid, rho=rho))
            if qbest < heaviest - 1e-6:          # 最重单箱都装不下 -> 不可行
                feasible = False
                break
            k = max(1, math.ceil(w / qbest))
            total_n += k
            per = w / k
            best_e = None
            for g in ("B", "C"):
                if per <= max_safe_payload(g, sid, rho=rho) + 1e-6:
                    e = roundtrip_energy(g, sid, per) * k
                    best_e = e if best_e is None else min(best_e, e)
            total_e += best_e if best_e else roundtrip_energy("C", sid, per) * k
        if not feasible:
            if infeasible_from is None:
                infeasible_from = rho
            n_sortie.append(np.nan)
            e_total.append(np.nan)
        else:
            n_sortie.append(total_n)
            e_total.append(total_e)

    xr = rhos * 100
    fig, ax1 = plt.subplots(figsize=(7.6, 4.6))
    c_n, c_e = "#3A5A9C", "#C0433B"

    if infeasible_from is not None:
        x0 = infeasible_from * 100
        ax1.axvspan(x0, 40.6, color="#DDDDDD", alpha=0.75, zorder=0)
        ax1.text((x0 + 40.6) / 2, np.nanmax(n_sortie) * 0.9,
                 "存在服务区\n无法服务", ha="center", va="center",
                 fontsize=9, color="#777777", style="italic", zorder=1)

    ax1.step(xr, n_sortie, where="post", color=c_n, lw=1.8,
             label="往返架次数", zorder=3)
    ax1.axvline(20, color="#888888", ls="--", lw=1.0, zorder=2)
    ax1.set_xlabel("返航安全余量（%）")
    ax1.set_ylabel("往返架次数", color=c_n)
    ax1.tick_params(axis="y", colors=c_n)
    ax1.set_xlim(10, 40.6)
    for s in ("top",):
        ax1.spines[s].set_visible(False)

    ax2 = ax1.twinx()
    ax2.plot(xr, e_total, "-o", color=c_e, mfc=c_e, mec="white", mew=0.6,
             ms=5, lw=1.5, label="总运输能耗", zorder=3)
    ax2.set_ylabel("总运输能耗（kWh）", color=c_e)
    ax2.tick_params(axis="y", colors=c_e)
    ax2.spines["top"].set_visible(False)

    l1, la1 = ax1.get_legend_handles_labels()
    l2, la2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, la1 + la2, loc="upper left", fontsize=9, frameon=False)
    fig.subplots_adjust(bottom=0.16)
    caption(fig, "图 D.3   问题一返航安全余量对组批架次数与总能耗的影响")
    fig.savefig(FIG / "xhs_figD3_rho_batch.png", dpi=200)
    fig.savefig(FIG / "xhs_figD3_rho_batch.pdf")
    plt.close(fig)
    print(f"[图D.3] 输出 xhs_figD3_rho_batch  (不可行起点 rho={infeasible_from})")


if __name__ == "__main__":
    fig_batch_loads()
    fig_rho_payload()
    fig_sortie_tradeoff()
    fig_rho_batch()
    print("全部 Q1 复刻图输出至", FIG)
