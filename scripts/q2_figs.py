# -*- coding: utf-8 -*-
"""问题2 论文配图：甘特图、理论下界对比、Pareto 前沿。"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Q2_架次优化"
FIG = ROOT / "论文" / "图"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Songti SC", "Heiti SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 10

DRONES = {"A": 4, "B": 2, "C": 2}
BATTERIES = {"A": 6, "B": 4, "C": 4}
COL = {"A": "#4C72B0", "B": "#DD8452", "C": "#55A868"}


def color_assign(sub, cap, names, key_end):
    """区间图贪心着色：为架次分配实体资源编号。"""
    free, busy, out = list(names), [], {}
    for t in sorted(sub, key=lambda z: z["start_s"]):
        busy.sort()
        while busy and busy[0][0] <= t["start_s"] + 1e-6:
            free.append(busy.pop(0)[1])
        free.sort()
        nm = free.pop(0)
        busy.append((key_end(t), nm))
        out[t["idx"]] = nm
    return out


def fig_gantt(sol):
    trips = sol["trips"]
    uav, bat = {}, {}
    for g in "ABC":
        sub = [t for t in trips if t["model"] == g]
        if not sub:
            continue
        uav |= color_assign(sub, DRONES[g],
                            [f"{g}{k+1}" for k in range(DRONES[g])],
                            lambda t: t["start_s"] + t["duration_s"])
        bat |= color_assign(sub, BATTERIES[g],
                            [f"{g}-B{k+1}" for k in range(BATTERIES[g])],
                            lambda t: t["start_s"] + t["duration_s"] + t["charge_s"])

    fig, axes = plt.subplots(2, 1, figsize=(11, 7.2),
                             gridspec_kw={"height_ratios": [8, 14]})
    # --- 上：无人机甘特 ---
    ax = axes[0]
    order = [f"{g}{k+1}" for g in "ABC" for k in range(DRONES[g])]
    ypos = {nm: i for i, nm in enumerate(order)}
    for t in trips:
        nm = uav[t["idx"]]
        ax.barh(ypos[nm], t["duration_s"], left=t["start_s"], height=0.62,
                color=COL[t["model"]], edgecolor="white", linewidth=0.6)
        ax.text(t["start_s"] + t["duration_s"] / 2, ypos[nm],
                str(len(t["boxes"])), ha="center", va="center",
                fontsize=7.5, color="white", fontweight="bold")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.invert_yaxis()
    ax.set_xlim(0, sol["cmax_s"] * 1.02)
    ax.axvline(sol["cmax_s"], color="crimson", ls="--", lw=1.4)
    ax.text(sol["cmax_s"], -0.9, f"$C_{{\\max}}$={sol['cmax_s']:.0f}s",
            color="crimson", ha="right", va="bottom", fontsize=9.5)
    ax.set_ylabel("无人机")
    ax.set_title("(a) 整机占用甘特图（条内数字为载箱数）", fontsize=11, pad=8)
    ax.grid(axis="x", ls=":", alpha=0.45)
    ax.legend(handles=[Patch(color=COL[g], label=f"{g} 型") for g in "ABC"],
              loc="lower center", bbox_to_anchor=(0.5, 1.06), ncol=3,
              fontsize=9, frameon=False)

    # --- 下：电池甘特（含充电段）---
    ax = axes[1]
    order = [f"{g}-B{k+1}" for g in "ABC" for k in range(BATTERIES[g])]
    ypos = {nm: i for i, nm in enumerate(order)}
    for t in trips:
        nm = bat[t["idx"]]
        ax.barh(ypos[nm], t["duration_s"], left=t["start_s"], height=0.62,
                color=COL[t["model"]], edgecolor="white", linewidth=0.6)
        ax.barh(ypos[nm], t["charge_s"], left=t["start_s"] + t["duration_s"],
                height=0.62, color=COL[t["model"]], alpha=0.3,
                hatch="///", edgecolor="white", linewidth=0.6)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.invert_yaxis()
    ax.set_xlim(0, sol["cmax_s"] * 1.02)
    ax.axvline(sol["cmax_s"], color="crimson", ls="--", lw=1.4)
    ax.set_xlabel("时刻 $t$ / s")
    ax.set_ylabel("电池组")
    ax.set_title("(b) 电池占用甘特图（实色=飞行，斜纹=充电）", fontsize=11, pad=8)
    ax.grid(axis="x", ls=":", alpha=0.45)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"q2_fig1_gantt.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("✅ q2_fig1_gantt")


def fig_bounds(sol, lb):
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.9))
    items = [
        ("架次数 $N$", lb["N_lb"], sol["n"], "架", 0),
        ("完工时刻 $C_{\\max}$", lb["Cmax_lb"], sol["cmax_s"], "s", 0),
        ("总能耗 $E$", lb["E_lb_fixedN"], sol["energy_kwh"], "kWh", 2),
    ]
    for ax, (name, low, cur, unit, nd) in zip(axes, items):
        bars = ax.bar(["理论下界", "本文解"], [low, cur],
                      color=["#8FBC8F", "#4C72B0"], width=0.55,
                      edgecolor="black", linewidth=0.7)
        for b, v in zip(bars, [low, cur]):
            ax.text(b.get_x() + b.get_width() / 2, v,
                    f"{v:.{nd}f}", ha="center", va="bottom", fontsize=10,
                    fontweight="bold")
        gap = (cur - low) / low * 100 if low else 0
        ax.set_title(f"{name}  (缺口 {gap:.1f}%)", fontsize=11)
        ax.set_ylabel(unit)
        ax.set_ylim(0, cur * 1.22)
        ax.grid(axis="y", ls=":", alpha=0.45)
        ax.annotate("", xy=(1, cur), xytext=(1, low),
                    arrowprops=dict(arrowstyle="<->", color="crimson", lw=1.3))
        ax.text(1.12, (low + cur) / 2, "缺口", color="crimson",
                fontsize=9, rotation=90, va="center")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"q2_fig2_bounds.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("✅ q2_fig2_bounds")


def fig_pareto(sol, lb):
    df = pd.read_csv(OUT / "pareto_analysis.csv")
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    par = df[df["帕累托"] == "★"]
    dom = df[df["帕累托"] != "★"]
    ax.scatter(dom["Cmax"], dom["能耗"], s=58, c="#BBBBBB",
               marker="o", label="受支配解", zorder=3, edgecolor="gray")
    ax.scatter(par["Cmax"], par["能耗"], s=78, c="#DD8452",
               marker="s", label="Pareto 前沿", zorder=4, edgecolor="black")
    ax.scatter([sol["cmax_s"]], [sol["energy_kwh"]], s=230, c="crimson",
               marker="*", label="本文解 bd23", zorder=6, edgecolor="black")
    for _, r in df.iterrows():
        ax.annotate(r["方案"], (r["Cmax"], r["能耗"]), fontsize=7.5,
                    xytext=(4, 4), textcoords="offset points", color="dimgray")
    ax.axvline(lb["Cmax_lb"], color="seagreen", ls="--", lw=1.4)
    ax.text(lb["Cmax_lb"], ax.get_ylim()[1], f" $C_{{\\max}}^{{lb}}$={lb['Cmax_lb']}s",
            color="seagreen", fontsize=9, va="top")
    ax.axhline(lb["E_lb_fixedN"], color="purple", ls="--", lw=1.4)
    ax.text(ax.get_xlim()[1], lb["E_lb_fixedN"],
            f"$E^{{lb}}$={lb['E_lb_fixedN']:.2f} ", color="purple",
            fontsize=9, ha="right", va="bottom")
    ax.set_xlabel("完工时刻 $C_{\\max}$ / s")
    ax.set_ylabel("总能耗 $E$ / kWh")
    ax.set_title("问题2 解集的 $C_{\\max}$–能耗权衡与理论下界", fontsize=11.5)
    ax.grid(ls=":", alpha=0.5)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"q2_fig3_pareto.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("✅ q2_fig3_pareto")


def fig_load(sol):
    """各机型整机/电池负载均衡对比。"""
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    gs = list("ABC")
    fl, bl = [], []
    for g in gs:
        t = [x for x in sol["trips"] if x["model"] == g]
        fl.append(sum(x["duration_s"] for x in t) / DRONES[g])
        bl.append(sum(x["duration_s"] + x["charge_s"] for x in t) / BATTERIES[g])
    w, xs = 0.34, np.arange(3)
    b1 = ax.bar(xs - w / 2, fl, w, label="整机负载 $\\Sigma d_j/K_g$",
                color="#4C72B0", edgecolor="black", linewidth=0.7)
    b2 = ax.bar(xs + w / 2, bl, w, label="电池负载 $\\Sigma (d_j+c_j)/B_g$",
                color="#DD8452", edgecolor="black", linewidth=0.7)
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                    f"{b.get_height():.0f}", ha="center", va="bottom", fontsize=9)
    ax.axhline(sol["cmax_s"], color="crimson", ls="--", lw=1.4,
               label=f"$C_{{\\max}}$={sol['cmax_s']:.0f}s")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{g} 型\n({sol['models'].get(g,0)} 架次)" for g in gs])
    ax.set_ylabel("负载 / s")
    ax.set_title("各机型整机与电池负载均衡性", fontsize=11.5)
    ax.grid(axis="y", ls=":", alpha=0.45)
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"q2_fig4_load.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("✅ q2_fig4_load")


if __name__ == "__main__":
    sol = json.loads((OUT / "solution_bd23.json").read_text())
    lb = json.loads((OUT / "lower_bounds.json").read_text())
    fig_gantt(sol)
    fig_bounds(sol, lb)
    fig_pareto(sol, lb)
    fig_load(sol)
    print("全部完成 →", FIG)
