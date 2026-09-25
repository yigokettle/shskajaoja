# -*- coding: utf-8 -*-
"""问题2 补充配图：四指标权衡（fig5）、逐箱送达时刻（fig6）。"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Q2_架次优化"
FIG = ROOT / "论文" / "图"

plt.rcParams["font.sans-serif"] = ["Songti SC", "Heiti SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 10

COL = {"A": "#4C72B0", "B": "#DD8452", "C": "#55A868"}


def fig_tradeoff(sol, lb):
    """四指标权衡：(a) N-Cmax 曲线  (b) Cmax-能耗前沿  (c) 及时性恒定说明。"""
    df = pd.read_csv(OUT / "pareto_analysis.csv")
    # 各 N 下的最优 Cmax（取该 N 的最小 Cmax）
    best = df.groupby("N")["Cmax"].min().reset_index().sort_values("N")
    # 并入本文解
    cur = pd.DataFrame([{"N": sol["n"], "Cmax": sol["cmax_s"]}])
    best = pd.concat([best[best.N != sol["n"]], cur]).sort_values("N")

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.1))

    # --- (a) 架次数 vs 完工时刻 ---
    ax = axes[0]
    ax.plot(best["N"], best["Cmax"], "o-", color="#4C72B0", lw=2,
            ms=8, markeredgecolor="black", zorder=3)
    ax.scatter([sol["n"]], [sol["cmax_s"]], s=250, marker="*", c="crimson",
               edgecolor="black", zorder=5, label="本文解 bd23")
    for _, r in best.iterrows():
        ax.annotate(f"{r['Cmax']:.0f}", (r["N"], r["Cmax"]), fontsize=8.5,
                    xytext=(0, 9), textcoords="offset points", ha="center")
    ax.axhline(lb["Cmax_lb"], color="seagreen", ls="--", lw=1.5)
    ax.text(best["N"].min(), lb["Cmax_lb"], f" $C_{{\\max}}^{{lb}}$={lb['Cmax_lb']}s",
            color="seagreen", fontsize=9, va="bottom")
    ax.set_xlabel("架次数 $N$")
    ax.set_ylabel("完工时刻 $C_{\\max}$ / s")
    ax.set_title("(a) 架次数–完工时刻权衡", fontsize=11)
    ax.set_xticks(best["N"].astype(int))
    ax.grid(ls=":", alpha=0.5)
    ax.legend(fontsize=9)

    # --- (b) 完工时刻 vs 能耗 ---
    ax = axes[1]
    par = df[df["帕累托"] == "★"]
    dom = df[df["帕累托"] != "★"]
    ax.scatter(dom["Cmax"], dom["能耗"], s=52, c="#BBBBBB", edgecolor="gray",
               label="受支配解", zorder=3)
    ax.scatter(par["Cmax"], par["能耗"], s=70, c="#DD8452", marker="s",
               edgecolor="black", label="Pareto 前沿", zorder=4)
    ax.scatter([sol["cmax_s"]], [sol["energy_kwh"]], s=250, marker="*",
               c="crimson", edgecolor="black", label="本文解 bd23", zorder=6)
    ax.axvline(lb["Cmax_lb"], color="seagreen", ls="--", lw=1.4)
    ax.axhline(lb["E_lb_fixedN"], color="purple", ls="--", lw=1.4)
    ax.set_xlabel("完工时刻 $C_{\\max}$ / s")
    ax.set_ylabel("总能耗 $E$ / kWh")
    ax.set_title("(b) 完工时刻–能耗权衡", fontsize=11)
    ax.grid(ls=":", alpha=0.5)
    ax.legend(fontsize=8.5, loc="upper right")

    # --- (c) 及时性维度 ---
    ax = axes[2]
    allsol = pd.concat([df[["方案", "Cmax", "加权延误", "按期箱"]],
                        pd.DataFrame([{"方案": "bd23", "Cmax": sol["cmax_s"],
                                       "加权延误": 0, "按期箱": 80}])])
    ax.scatter(allsol["Cmax"], allsol["按期箱"], s=90, c="#55A868",
               edgecolor="black", zorder=4)
    ax.scatter([sol["cmax_s"]], [80], s=250, marker="*", c="crimson",
               edgecolor="black", zorder=6, label="本文解 bd23")
    ax.axhline(80, color="crimson", ls=":", lw=1.6)
    ax.set_ylim(75, 82)
    ax.set_xlabel("完工时刻 $C_{\\max}$ / s")
    ax.set_ylabel("按期交付箱数")
    ax.set_title("(c) 及时性：全部解均达 80/80", fontsize=11)
    ax.grid(ls=":", alpha=0.5)
    ax.text(0.5, 0.18, "及时性被设为硬约束\n不参与权衡",
            transform=ax.transAxes, ha="center", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5", fc="lightyellow", ec="gray"))
    ax.legend(fontsize=9, loc="lower right")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"q2_fig5_tradeoff.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("✅ q2_fig5_tradeoff")


def fig_boxdelivery(sol):
    """逐箱送达时刻：按服务区分组，显示交付时刻与期望时限余量。"""
    bd = pd.read_csv(OUT / "bd23_box_delivery.csv")
    ts = pd.read_csv(OUT / "bd23_trip_schedule.csv")
    m2g = dict(zip(ts.架次编号, ts.机型编号))
    bd["机型"] = bd.架次编号.map(m2g)
    # 同一服务区的不同物资期望时限不同（15 区全部如此，共 39 种组合），
    # 故按 (服务区, 期望时限) 分行，不可用单一时限代表整个服务区
    bd = bd.sort_values(["服务区编号", "期望送达时刻_s"])
    keys = list(dict.fromkeys(zip(bd.服务区编号, bd.期望送达时刻_s)))
    ypos = {k: i for i, k in enumerate(keys)}

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 8.6),
                             gridspec_kw={"width_ratios": [2.15, 1]})

    # --- 左：交付时刻散点 ---
    ax = axes[0]
    for (area, exp), y in ypos.items():
        sub = bd[(bd.服务区编号 == area) & (bd.期望送达时刻_s == exp)]
        ax.plot([0, exp], [y, y], color="lightgray", lw=5,
                solid_capstyle="butt", zorder=1)
        ax.plot([exp], [y], marker="|", ms=13, color="crimson",
                mew=2, zorder=5)
        for _, r in sub.iterrows():
            mk = ("^" if r.是否首批保障 == "是"
                  else ("s" if r.物资类型 == "医疗物资" else "o"))
            ax.scatter(r.交付完成时刻_s, y, s=42, marker=mk,
                       c=COL[r.机型], edgecolor="black", linewidth=0.5, zorder=4)
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([f"{a}  ({int(e)}s)" for a, e in keys], fontsize=7.4)
    ax.invert_yaxis()
    ax.set_xlabel("交付完成时刻 / s")
    ax.set_ylabel("服务区（括号内为该类物资的期望时限）")
    ax.set_title("(a) 逐箱送达时刻（红竖线=期望时限，灰条=允许窗口）",
                 fontsize=11)
    ax.grid(axis="x", ls=":", alpha=0.5)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="^", ls="", mfc="gray", mec="black", label="首批保障"),
         Line2D([], [], marker="s", ls="", mfc="gray", mec="black", label="医疗物资"),
         Line2D([], [], marker="o", ls="", mfc="gray", mec="black", label="其他物资"),
         Line2D([], [], marker="|", ls="", mec="crimson", ms=12, label="期望时限")]
    h += [Line2D([], [], marker="o", ls="", mfc=COL[g], mec="black", label=f"{g} 型")
          for g in "ABC"]
    ax.legend(handles=h, fontsize=8.5, loc="lower right", ncol=2, framealpha=0.92)

    # --- 右：时限余量分布 ---
    ax = axes[1]
    bd["余量"] = bd.期望送达时刻_s - bd.交付完成时刻_s
    cats = [("首批保障", bd[bd.是否首批保障 == "是"]),
            ("医疗物资", bd[(bd.物资类型 == "医疗物资") & (bd.是否首批保障 != "是")]),
            ("其他物资", bd[(bd.物资类型 != "医疗物资") & (bd.是否首批保障 != "是")])]
    data = [c[1]["余量"].values for c in cats if len(c[1])]
    labs = [f"{c[0]}\n(n={len(c[1])})" for c in cats if len(c[1])]
    bp = ax.boxplot(data, tick_labels=labs, patch_artist=True, widths=0.55,
                    medianprops=dict(color="black", lw=1.6))
    for p, c in zip(bp["boxes"], ["#8FBC8F", "#DD8452", "#4C72B0"]):
        p.set_facecolor(c)
        p.set_alpha(0.75)
    ax.axhline(0, color="crimson", ls="--", lw=1.8)
    ax.text(0.02, 0.04, f"全部余量 > 0\n最小 {bd['余量'].min():.1f} s",
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle="round,pad=0.45", fc="lightyellow", ec="gray"))
    ax.set_ylabel("时限余量 / s")
    ax.set_title("(b) 三类物资的时限余量分布", fontsize=11)
    ax.grid(axis="y", ls=":", alpha=0.5)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"q2_fig6_boxdelivery.{ext}", dpi=300,
                    bbox_inches="tight")
    plt.close(fig)
    print("✅ q2_fig6_boxdelivery")


if __name__ == "__main__":
    sol = json.loads((OUT / "solution_bd23.json").read_text())
    lb = json.loads((OUT / "lower_bounds.json").read_text())
    fig_tradeoff(sol, lb)
    fig_boxdelivery(sol)
    print("完成 →", FIG)
