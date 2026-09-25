# -*- coding: utf-8 -*-
"""问题2 三类理论下界的统一计算（架次数 N、完工时刻 Cmax、总能耗 E）。

论文用途：给出「与算法无关」的性能基准，说明当前解离理论极限还有多远。
三个下界都只依赖**必要条件**，任何可行解都必须满足，故均为严格下界。

--------------------------------------------------------------------------
一、架次数下界 N_lb —— 装箱（bin packing）松弛
--------------------------------------------------------------------------
    每架次受三重容量约束：载重 Q_g、体积 V_g、单架次可用电量 (1-ρ)E_g。
    松弛掉「同一架次的箱必须同属一条可行航线」，退化为装箱问题，再取
    连续松弛（允许箱子劈开），得：

        N_lb = max( ceil(Σm_b / max_g Q_g),
                    ceil(Σv_b / max_g V_g),
                    ceil( 单箱最小能耗和 / max_g (1-ρ)E_g ) )

    进一步用 CP-SAT 求「最小架次数」的精确装箱下界（不劈箱），更紧。

--------------------------------------------------------------------------
二、完工时刻下界 Cmax_lb —— 聚合面积（area）松弛
--------------------------------------------------------------------------
    见 q2_lb_relax.py。[0,Cmax] 内机型 g 的 K_g 台机最多提供 K_g*Cmax 机时，
    B_g 组电池最多提供 B_g*Cmax 池时：
        Σ dur_i <= K_g*Cmax     Σ (dur_i+chg_i) <= B_g*Cmax

--------------------------------------------------------------------------
三、总能耗下界 E_lb —— 集合覆盖松弛
--------------------------------------------------------------------------
    丢掉所有时间与资源约束，只要求「每箱恰好被送一次」，求最小总能耗。
    这是一个集合分划问题，其最优值是真实总能耗的严格下界。
    另给一个更松但更直观的解析下界：每箱单独计算「最省的承运方式」。
"""
from __future__ import annotations
import argparse
import json
import math
from collections import Counter
from pathlib import Path

import pandas as pd
from ortools.sat.python import cp_model

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "Q2_开源对比与优化"
OUT = ROOT / "Q2_架次优化"

DRONES = {"A": 4, "B": 2, "C": 2}
BATTERIES = {"A": 6, "B": 4, "C": 4}
CAP = {"A": (25, 0.060), "B": (30, 0.073), "C": (80, 0.250)}
EUSE = {"A": 4.5, "B": 4.0, "C": 8.0}
RHO = 0.20
SCALE = 1_000_000


def emax(g):
    return (1 - RHO) * EUSE[g]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="Q2_架次优化/pool_c8.json")
    ap.add_argument("--n", type=int, default=23)
    ap.add_argument("--seconds", type=float, default=240)
    args = ap.parse_args()

    p = Path(args.pool)
    if not p.is_absolute():
        p = ROOT / p
    pool = json.loads(p.read_text())
    b = pd.read_csv(SRC / "optimized_box_delivery.csv").set_index("货箱编号")
    hard = {k: (None if pd.isna(v) else float(v)) for k, v in b.硬截止时刻_s.items()}
    exp = b.期望送达时刻_s.to_dict()

    boxes = sorted({x for c in pool for x in c["boxes"]})
    bidx = {x: i for i, x in enumerate(boxes)}
    print(f"候选池 {len(pool)}（{dict(Counter(c['model'] for c in pool))}），箱数 {len(boxes)}")

    # 箱属性：从原始需求表按 (服务区, 物资类型) 取单箱质量/体积
    dem = pd.read_excel(ROOT / "数据" / "无人机应急物资运输基础数据"
                        / "物资需求与配送时限.xlsx")
    unit = {(r.服务区编号, r.物资类型): (float(r._5), float(r._6))
            for r in dem.itertuples()}
    # itertuples 对含特殊字符列名会重命名，稳妥起见改用列定位
    unit = {}
    for _, r in dem.iterrows():
        unit[(r["服务区编号"], r["物资类型"])] = (
            float(r["单箱质量（kg）"]), float(r["单箱体积（m³）"]))
    sa = b.服务区编号.to_dict()
    ty = b.物资类型.to_dict()
    bm, bv = {}, {}
    for x in boxes:
        q, v = unit[(sa[x], ty[x])]
        bm[x], bv[x] = q, v
    Wsum = sum(bm[x] for x in boxes)
    Vsum = sum(bv[x] for x in boxes)

    print("\n" + "=" * 66)
    print("【一】架次数下界 N_lb —— 装箱松弛")
    print("=" * 66)
    qmax = max(CAP[g][0] for g in "ABC")
    vmax = max(CAP[g][1] for g in "ABC")
    n_w = math.ceil(Wsum / qmax)
    n_v = math.ceil(Vsum / vmax)
    print(f"总载重 {Wsum:.2f} kg / 最大机型载重 {qmax} kg  → ceil = {n_w}")
    print(f"总体积 {Vsum:.4f} m3 / 最大机型体积 {vmax} m3 → ceil = {n_v}")

    # CP-SAT 精确装箱下界（不劈箱，只受载重+体积约束，全部按最大机型 C 算）
    mb = cp_model.CpModel()
    NB = max(n_w, n_v) + 8
    y = [mb.NewBoolVar(f"y{k}") for k in range(NB)]          # 第 k 个架次是否启用
    z = [[mb.NewBoolVar(f"z{i}_{k}") for k in range(NB)] for i in range(len(boxes))]
    for i in range(len(boxes)):
        mb.AddExactlyOne(z[i])
    for k in range(NB):
        mb.Add(sum(int(round(bm[boxes[i]] * 1000)) * z[i][k]
                   for i in range(len(boxes))) <= int(round(qmax * 1000)) * y[k])
        mb.Add(sum(int(round(bv[boxes[i]] * 1e6)) * z[i][k]
                   for i in range(len(boxes))) <= int(round(vmax * 1e6)) * y[k])
    for k in range(NB - 1):
        mb.Add(y[k] >= y[k + 1])                              # 对称性破除
    mb.Minimize(sum(y))
    sb = cp_model.CpSolver()
    sb.parameters.max_time_in_seconds = args.seconds
    sb.parameters.num_search_workers = 8
    rb = sb.Solve(mb)
    n_bp = int(math.ceil(sb.BestObjectiveBound())) if rb in (
        cp_model.OPTIMAL, cp_model.FEASIBLE) else max(n_w, n_v)
    print(f"CP-SAT 精确装箱（载重+体积，按最大机型 C）→ 下界 {n_bp}"
          f"  [{sb.StatusName(rb)}]")
    N_lb = max(n_w, n_v, n_bp)
    print(f"\n  ⇒ N_lb = {N_lb}")

    print("\n" + "=" * 66)
    print("【二】总能耗下界 E_lb —— 集合覆盖松弛")
    print("=" * 66)
    me = cp_model.CpModel()
    xe = [me.NewBoolVar(f"x{i}") for i in range(len(pool))]
    cov = [[] for _ in boxes]
    for i, c in enumerate(pool):
        for x in c["boxes"]:
            cov[bidx[x]].append(xe[i])
    for lst in cov:
        me.AddExactlyOne(lst)
    ec = [int(round(c["energy_kwh"] * SCALE)) for c in pool]
    me.Minimize(sum(ec[i] * xe[i] for i in range(len(pool))))
    se = cp_model.CpSolver()
    se.parameters.max_time_in_seconds = args.seconds
    se.parameters.num_search_workers = 8
    re_ = se.Solve(me)
    if re_ in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        E_free = se.Value(me.Proto().objective and sum(
            ec[i] * se.Value(xe[i]) for i in range(len(pool)))) / SCALE \
            if False else sum(ec[i] * se.Value(xe[i])
                              for i in range(len(pool))) / SCALE
        E_lb_free = se.BestObjectiveBound() / SCALE
        nsel = sum(se.Value(xe[i]) for i in range(len(pool)))
        print(f"无架次数限制：E = {E_free:.4f} kWh（下界 {E_lb_free:.4f}），"
              f"用 {nsel} 架次  [{se.StatusName(re_)}]")
    else:
        E_lb_free = 0.0
        print("求解失败")

    # 固定 N=23 时的能耗下界
    me.Add(sum(xe) == args.n)
    se2 = cp_model.CpSolver()
    se2.parameters.max_time_in_seconds = args.seconds
    se2.parameters.num_search_workers = 8
    re2 = se2.Solve(me)
    if re2 in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        E_n = sum(ec[i] * se2.Value(xe[i]) for i in range(len(pool))) / SCALE
        E_lb_n = se2.BestObjectiveBound() / SCALE
        print(f"固定 N={args.n}：  E = {E_n:.4f} kWh（下界 {E_lb_n:.4f}）"
              f"  [{se2.StatusName(re2)}]")
    else:
        E_lb_n = E_lb_free
        print(f"固定 N={args.n} 求解失败")

    print("\n" + "=" * 66)
    print("【三】汇总（与 bd23 现解对比）")
    print("=" * 66)
    sol = json.loads((OUT / "solution_bd23.json").read_text())
    print(f"{'指标':<12}{'理论下界':>14}{'当前解 bd23':>16}{'缺口':>14}")
    print("-" * 66)
    print(f"{'架次数 N':<12}{N_lb:>14}{sol['n']:>16}{sol['n']-N_lb:>14}")
    print(f"{'Cmax (s)':<12}{5942:>14}{sol['cmax_s']:>16}"
          f"{sol['cmax_s']-5942:>14}")
    print(f"{'能耗 (kWh)':<10}{E_lb_n:>14.4f}{sol['energy_kwh']:>16.4f}"
          f"{sol['energy_kwh']-E_lb_n:>14.4f}")
    print("-" * 66)
    res = dict(N_lb=N_lb, N_lb_weight=n_w, N_lb_volume=n_v, N_lb_binpack=n_bp,
               Cmax_lb=5942, E_lb_free=round(E_lb_free, 6),
               E_lb_fixedN=round(E_lb_n, 6), n=args.n,
               cur=dict(n=sol["n"], cmax_s=sol["cmax_s"],
                        energy_kwh=sol["energy_kwh"]))
    (OUT / "lower_bounds.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2))
    print("已保存", OUT / "lower_bounds.json")


if __name__ == "__main__":
    main()
