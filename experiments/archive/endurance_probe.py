"""续航 / 可达性探针（14.9 Q&A）—— **用引擎实测**，不用算术。

回答两个问题（fish 2026-09-25 问）
---------------------------------
Q1 **从吃饱到饥饿要多久？**
Q2 **满能量（且不饿）能移动多远？**

做法（为什么必须实测而不是算）
-----------------------------
代谢链路里有三样东西互相耦合，手算必然漏：
  * `digest_rate = base_metabolism × (0.5 + g1×1.5) × eff_activity`（**基因 g1 直接放大消耗**）
  * `cost_meta  = base_metabolism × (0.5 + g1×1.5) × age_mult`（未成年/老年各有一个倍率）
  * 移动扣 `move_cost × cold_penalty × (0.5 + g6)`，**只有真正移动那一 tick 才扣**
再加上==温度 `activity_factor` 逐 tick 变动 == ⇒ 只有跑引擎才知道真实续航。

受控处理（保证测的是"续航"不是"觅食"）
---------------------------------------
* **每 tick 后把资源清零** ⇒ 全程吃不到东西 ⇒ 纯粹量"油箱能撑多久"；
* **`maturity_fraction = 1.01`** ⇒ 永未成年 ⇒ **不繁殖**（繁殖会把父代能量抽走，污染续航）；
* ARS 关、迁徙关、捕食默认 ⇒ 不引入额外 Behavior 项。

三条被测的线
-----------
* **饥饿线** `energy < starve_frac × max_energy` = 180（到了这条还不吃 ⇒ 判饿死）
* **力竭线** `energy < exhaust_frac × max_energy` = 102（**不管胃里有没有东西** ⇒ 真·死亡线之一）
* **死亡** 个体从 `alive` 集合消失
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

from simulation.config import SimConfig  # noqa: E402
from simulation.sphere_engine import SphereEngine  # noqa: E402
from simulation.genes import Gene  # noqa: E402

ROWS, COLS = 60, 120
_DEG_PER_CELL = 3.0            # 每行/每列 3°（球面 60×120）
_CELLS_PER_RAD = ROWS / math.pi   # ≈19.1 格/弧度（60 行 = 180° = π rad）


def great_circle_cells(f1: int, f2: int) -> float:
    """两个平铺索引之间的球面距离，**单位 = 格**。"""
    r1, c1 = divmod(int(f1), COLS)
    r2, c2 = divmod(int(f2), COLS)
    lat1 = math.radians(90.0 - (r1 + 0.5) * _DEG_PER_CELL)
    lat2 = math.radians(90.0 - (r2 + 0.5) * _DEG_PER_CELL)
    dlon = math.radians((c2 - c1) * _DEG_PER_CELL)
    cos_a = (math.sin(lat1) * math.sin(lat2)
             + math.cos(lat1) * math.cos(lat2) * math.cos(dlon))
    cos_a = min(1.0, max(-1.0, cos_a))
    return math.acos(cos_a) * _CELLS_PER_RAD


def make_cfg(seed: int, initial_energy: float, speed_max: float) -> SimConfig:
    """13.5 B 臂底盘（patchy + bgzero）——与正式批同一来源。"""
    c = SimConfig(seed=seed)
    c.simulation.use_sim_core = False
    c.simulation.l2_dash = False
    c.population.max_count = 600
    c.predation.forage_tradeoff_k = 0.0
    c.predation.enabled = False     # 🔴 捕食会让个体**还没饿就被吃掉** ⇒ 污染续航测量
    c.resources.distribution = "patchy"
    c.resources.bg_production_zero = True
    c.resources.patch_count = 30
    c.resources.patch_radius = 2
    c.resources.patch_regrowth_mult = 1.195
    # 🔴 第一次跑踩的坑：只把 `_grid` 清零是不够的——**下一 tick 会先再生、再被吃掉**
    #    （regrowth_rate=0.5 ⇒ 每格每 tick 长出 0.5 质量 ⇒ 实测个体能量不降反升）。
    #    ⇒ 必须把**再生率本身**归零，才是真正的"无食物世界"。
    c.resources.regrowth_rate = 0.0
    o = c.organisms
    o.max_energy = 600.0
    o.initial_energy = float(initial_energy)
    o.starve_frac = 0.30
    o.exhaust_frac = 0.17
    o.eat_efficiency = 7.5
    o.assim_herb = 0.4
    o.assim_carn = 0.8
    o.stomach_cap_mass = 25.0
    o.eat_threshold_frac = 0.6
    o.photo_max = 0.0
    o.maturity_fraction = 1.01          # ⇒ 永未成年 ⇒ **不繁殖**（繁殖会抽走父代能量）
    c.ars.enabled = False               # 本探针量的是"油箱"，不开任何新机制
    if speed_max > 0:
        c.subpos.enabled = True
        c.subpos.speed_max = float(speed_max)
    return c


def run_one(seed: int, initial_energy: float, speed_max: float,
            full_stomach: bool, t_max: int, debug: bool = False) -> list[dict]:
    """跑一个 run，返回每个创始代个体的一行续航记录。"""
    eng = SphereEngine(make_cfg(seed, initial_energy, speed_max))
    grid = eng.resources._grid

    ids = eng._id[: len(eng._flat)].copy()
    p0 = len(ids)
    start_flat = eng._flat[:p0].copy()
    g1 = eng._genes[:p0, Gene.METABOLIC].copy()      # 代谢基因 ⇒ 续航的主因
    g0 = eng._genes[:p0, Gene.MOVE_PROB].copy()

    if full_stomach:
        eng._stomach[:p0] = float(eng.config.organisms.stomach_cap_mass)
    start_stomach = eng._stomach[:p0].copy()

    rec = {int(i): {
        "seed": seed, "id": int(i), "g1": float(g1[k]), "g0": float(g0[k]),
        "start_flat": int(start_flat[k]), "start_energy": float(eng._energy[k]),
        "start_stomach": float(start_stomach[k]),
        "t_stomach_empty": None, "t_h50": None, "t_hungry": None, "t_exhaust": None, "t_death": None,
        "steps": 0, "steps_till_hungry": None,
        "steps_till_empty": None, "dist_till_empty": None,
        "dist_till_hungry": None, "dist_total": None,
    } for k, i in enumerate(ids)}

    for t in range(1, t_max + 1):
        grid[:] = 0.0                                   # ⇒ **全程无食物**（进 tick 前清零）
        eng.step()
        grid[:] = 0.0                                   # （出 tick 后再清一次，双保险）
        cur_flat = eng._flat[: len(eng._flat)]
        cur_ids = eng._id[: len(eng._flat)]
        cur_en = eng._energy[: len(eng._flat)]
        cur_st = eng._stomach[: len(eng._flat)]
        alive = set(int(x) for x in cur_ids)
        for k, i in enumerate(cur_ids):
            r = rec.get(int(i))
            if r is None:
                continue
            f = int(cur_flat[k])
            if f != r.setdefault("_prev_flat", f):
                r["steps"] += 1
            r["_prev_flat"] = f
            e = float(cur_en[k])
            if r["t_stomach_empty"] is None and float(cur_st[k]) <= 1e-9:
                r["t_stomach_empty"] = t
                r["steps_till_empty"] = r["steps"]
                r["dist_till_empty"] = great_circle_cells(r["start_flat"], f)
            if r["t_h50"] is None and e < 300.0:         # 50% 线（"还有一半油"）
                r["t_h50"] = t
            if r["t_hungry"] is None and e < 180.0:      # starve_frac × max_energy
                r["t_hungry"] = t
                r["steps_till_hungry"] = r["steps"]
                r["dist_till_hungry"] = great_circle_cells(r["start_flat"], f)
            if r["t_exhaust"] is None and e < 102.0:     # exhaust_frac × max_energy
                r["t_exhaust"] = t
        for i in rec:
            if rec[i]["t_death"] is None and i not in alive and rec[i].get("_prev_flat") is not None:
                rec[i]["t_death"] = t
                # 🔴 修正"死亡早于饥饿"的假象：个体在 step 内部死亡时我们读不到它最后一刻的
                #    能量 ⇒ `t_hungry` 会被系统性记晚。饿死者的死因必然是跌破饥饿线
                #    ⇒ 用死亡 tick 兜底（对还有 None 的等级同理）。
                for k_dst, thr in (("t_h50", 300.0), ("t_hungry", 180.0), ("t_exhaust", 102.0)):
                    if rec[i][k_dst] is None:
                        rec[i][k_dst] = t
                if rec[i]["steps_till_hungry"] is None:
                    rec[i]["steps_till_hungry"] = rec[i]["steps"]
                if rec[i]["dist_till_hungry"] is None:
                    rec[i]["dist_till_hungry"] = great_circle_cells(
                        rec[i]["start_flat"], rec[i].get("_prev_flat", rec[i]["start_flat"]))
                if rec[i]["dist_total"] is None:
                    rec[i]["dist_total"] = great_circle_cells(
                        rec[i]["start_flat"], rec[i].get("_prev_flat", rec[i]["start_flat"]))
        if len(eng._flat) == 0:
            break
        if debug and t % 100 == 0:
            sample = [r for r in rec.values() if r["t_death"] is None][:3]
            print(f"       [dbg t={t}] " + " | ".join(
                f"id{s['id']}: steps={s['steps']} E={float(cur_en[list(cur_ids).index(s['id'])]) if s['id'] in set(int(x) for x in cur_ids) else -1:.1f}"
                for s in sample))

    out = []
    for r in rec.values():
        r.pop("_prev_flat", None)
        out.append(r)
    print(f"       死因分布：{dict(getattr(eng, '_run_deaths', {}))}")
    return out


FIELDS = ["arm", "seed", "id", "g1", "g0", "start_energy", "start_stomach",
          "t_stomach_empty", "t_h50", "t_hungry", "t_exhaust", "t_death",
          "steps", "steps_till_hungry", "dist_till_hungry", "dist_total",
          "steps_till_empty", "dist_till_empty"]


def _q(vals, p):
    v = [x for x in vals if x is not None]
    return round(float(np.percentile(v, p)), 1) if v else None


def summarize(tag: str, rows: list[dict]) -> dict:
    def col(k):
        return [r[k] for r in rows]
    return {
        "arm": tag,
        "n": len(rows),
        "胃清空 tick 中位": _q(col("t_stomach_empty"), 50),
        "到半油(300) 中位": _q(col("t_h50"), 50),
        "到饥饿(180) 中位": _q(col("t_hungry"), 50),
        "到饥饿 P10/P90": f"{_q(col('t_hungry'), 10)} / {_q(col('t_hungry'), 90)}",
        "到力竭(102) 中位": _q(col("t_exhaust"), 50),
        "死亡 tick 中位": _q(col("t_death"), 50),
        "走的格数(到饥饿) 中位": _q(col("steps_till_hungry"), 50),
        "净位移(到饥饿) 中位": _q(col("dist_till_hungry"), 50),
        "净位移(到饥饿) P90": _q(col("dist_till_hungry"), 90),
        "走到胃空时的路程": _q(col("steps_till_empty"), 50),
        "走到胃空时的净位移": _q(col("dist_till_empty"), 50),
        "g1低(<0.33)到饿": _q([r["t_hungry"] for r in rows if r["g1"] < 0.33], 50),
        "g1中(0.33-0.66)到饿": _q([r["t_hungry"] for r in rows if 0.33 <= r["g1"] < 0.66], 50),
        "g1高(≥0.66)到饿": _q([r["t_hungry"] for r in rows if r["g1"] >= 0.66], 50),
    }


ARMS = [
    ("① 满能量600 + 空胃 / 1格每tick", dict(initial_energy=600.0, full_stomach=False, speed_max=0.0)),
    ("② 满能量600 + 饱胃25 / 1格每tick", dict(initial_energy=600.0, full_stomach=True, speed_max=0.0)),
    ("③ 满能量600 + 饱胃25 / 0.25格每tick", dict(initial_energy=600.0, full_stomach=True, speed_max=0.25)),
    ("④ 13.5出生态 300 + 空胃 / 1格每tick", dict(initial_energy=300.0, full_stomach=False, speed_max=0.0)),
]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="续航/可达性探针")
    ap.add_argument("--seeds", default="42,7,11")
    ap.add_argument("--t-max", type=int, default=1500)
    ap.add_argument("--arms", default="1,2,3,4", help="挑哪几臂，逗号分隔（1-4）")
    ap.add_argument("--out", default="results/endurance_probe.csv")
    ap.add_argument("--debug", action="store_true")
    a = ap.parse_args(argv)

    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    picks = [int(s) for s in a.arms.split(",") if s.strip()]
    print(f"== 续航探针：{len(picks)} 臂 × {len(seeds)} seed，t_max={a.t_max} ==")
    print("    （每 tick 后资源清零 ⇒ 全程无食物；maturity=1.01 ⇒ 不繁殖）\n")

    all_rows, summary = [], []
    t0 = time.time()
    for idx in picks:
        tag, kw = ARMS[idx - 1]
        rows = []
        for seed in seeds:
            rows += run_one(seed, kw["initial_energy"], kw["speed_max"],
                            kw["full_stomach"], a.t_max, a.debug)
        for r in rows:
            r["arm"] = tag
        all_rows += rows
        s = summarize(tag, rows)
        summary.append(s)
        print(f"[{tag}] n={s['n']}  "
              f"胃清空={s['胃清空 tick 中位']}｜半油(300)={s['到半油(300) 中位']}"
              f"｜到饥饿(180)={s['到饥饿(180) 中位']}"
              f"（P10/P90 {s['到饥饿 P10/P90']}）｜到力竭(102)={s['到力竭(102) 中位']}"
              f"｜死亡={s['死亡 tick 中位']}")
        print(f"      走到胃空为止：累计跨格 {s['走到胃空时的路程']} 格"
              f"｜净位移 {s['走到胃空时的净位移']} 格"
              f" ｜按代谢基因 g1 分层到饿：低 {s['g1低(<0.33)到饿']} / "
              f"中 {s['g1中(0.33-0.66)到饿']} / 高 {s['g1高(≥0.66)到饿']} tick")
        print(f"      走到饿为止：累计跨格 {s['走的格数(到饥饿) 中位']} 格"
              f"｜净位移 中位 {s['净位移(到饥饿) 中位']} 格 / P90 {s['净位移(到饥饿) P90']} 格"
              f"  （{time.time()-t0:.0f}s）")

    if a.out:
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
            w.writeheader()
            w.writerows(all_rows)
        print(f"\n✅ 明细 CSV：{out}（{len(all_rows)} 行，含每一个体）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
