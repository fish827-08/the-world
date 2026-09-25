"""直线续航探针（14.9 Q&A 追加）—— **满能量满油，一路直着走能走多少格才饿死**。

与 `endurance_probe.py` 的区别
-----------------------------
`endurance_probe` 量的是**真实处境**（引擎默认的随机走 = 布朗式）；本脚本量的是
**上界**：把"会不会找路"这件事人为锁成"永远沿一个方向走"，看油箱到底值多少格。

怎么造出"直线"（**不改引擎代码**）
---------------------------------
复用 14.9 已经写好的 ARS 赶路腿，把它的旋钮推到极限：
  * `gain=200`（惯性项 ±200，其他 score 项都在 1 以下 ⇒ **方向完全由惯性主导**）
  * `giveup=1e9`（**永不"失望"** ⇒ 永不右转 90°，方向一旦定下就不变）
  * `MOVE_PROB=1` + `ROOTING=0`（**每 tick 必走**，不掷"走不走"的骰子）
  * `heading=4`（Moore 槽位"右"= 向东）⇒ 沿**纬线**走，**不会撞极点**
    （⚠️ 若朝北/南走，60 格就撞极点，极区 ARS 豁免 ⇒ 方向会乱 ⇒ 无法测"直线"）

为什么必须"无食物 + 关捕食 + 禁繁殖"：见 `endurance_probe.py` 的三条受控处理。

🔴 球面几何的必然结果（这不是 bug）
---------------------------------
世界 120 列 ⇒ **向东走满 120 格就回到原点**。所以"能走多少格"要分两个口径：
  * **路径长度**（走过多少格）—— 与几何无关，就是答案
  * **净位移**（离起点多远）—— 被球面绕圈钳住，**上限 = 半个世界（约 60 格 / 180°）**
"""
from __future__ import annotations

import argparse
import csv
import math
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
_DEG_PER_CELL = 3.0
_CELLS_PER_RAD = ROWS / math.pi


def great_circle_cells(f1: int, f2: int) -> float:
    r1, c1 = divmod(int(f1), COLS)
    r2, c2 = divmod(int(f2), COLS)
    lat1 = math.radians(90.0 - (r1 + 0.5) * _DEG_PER_CELL)
    lat2 = math.radians(90.0 - (r2 + 0.5) * _DEG_PER_CELL)
    dlon = math.radians((c2 - c1) * _DEG_PER_CELL)
    cos_a = (math.sin(lat1) * math.sin(lat2)
             + math.cos(lat1) * math.cos(lat2) * math.cos(dlon))
    cos_a = min(1.0, max(-1.0, cos_a))
    return math.acos(cos_a) * _CELLS_PER_RAD


def make_cfg(seed: int, straight: bool) -> SimConfig:
    c = SimConfig(seed=seed)
    c.simulation.use_sim_core = False
    c.simulation.l2_dash = False
    c.population.max_count = 600
    c.predation.enabled = False
    c.resources.distribution = "patchy"
    c.resources.bg_production_zero = True
    c.resources.patch_count = 30
    c.resources.patch_radius = 2
    c.resources.patch_regrowth_mult = 1.195
    c.resources.regrowth_rate = 0.0          # 🔴 再生率归零（只清 _grid 不够，见 §13.1）
    o = c.organisms
    o.max_energy = 600.0
    o.initial_energy = 600.0                 # 满油箱出发
    o.starve_frac = 0.30                     # 饿死线 180
    o.exhaust_frac = 0.17
    o.eat_efficiency = 7.5
    o.assim_herb = 0.4
    o.assim_carn = 0.8
    o.stomach_cap_mass = 25.0
    o.eat_threshold_frac = 0.6
    o.photo_max = 0.0
    o.maturity_fraction = 1.01               # 禁繁殖
    c.ars.enabled = bool(straight)
    c.ars.gain = 200.0                       # 惯性项压倒一切 ⇒ 方向锁死
    c.ars.giveup = 10 ** 9                   # 永不失失望 ⇒ 永不右转
    c.ars.theta = 0.5
    c.ars.giveup = 10 ** 9
    return c


def run(seed: int, straight: bool, t_max: int) -> tuple[list[dict], dict]:
    e = SphereEngine(make_cfg(seed, straight))
    p0 = len(e._flat)
    ids = e._id[:p0].copy()
    start = e._flat[:p0].copy()
    e._energy[:p0] = 600.0
    e._stomach[:p0] = 25.0                   # 饱胃
    e._genes[:p0, Gene.MOVE_PROB] = 1.0      # 每 tick 必走
    e._genes[:p0, Gene.ROOTING] = 0.0
    e._heading[:p0] = 4                      # 向东（沿纬线，不撞极点）
    g1 = e._genes[:p0, Gene.METABOLIC].copy()
    g6 = e._genes[:p0, Gene.MOVE_COST].copy()

    rec = {int(i): {"seed": seed, "id": int(i), "g1": float(g1[k]), "g6": float(g6[k]),
                    "start_flat": int(start[k]), "steps": 0, "east": 0,
                    "same_dir": 0, "_last": None, "_dirs": {},
                    "max_dist": 0.0, "t_max_dist": None, "max_dist_till_hungry": None,
                    "t_hungry": None, "t_death": None, "steps_till_hungry": None,
                    "dist_till_hungry": None, "dist_total": None}
           for k, i in enumerate(ids)}

    grid = e.resources._grid
    for t in range(1, t_max + 1):
        grid[:] = 0.0
        e.step()
        grid[:] = 0.0
        cur_flat = e._flat[: len(e._flat)]
        cur_ids = e._id[: len(e._flat)]
        cur_en = e._energy[: len(e._flat)]
        alive = set(int(x) for x in cur_ids)
        for k, i in enumerate(cur_ids):
            r = rec.get(int(i))
            if r is None:
                continue
            f = int(cur_flat[k])
            prev = r.get("_prev")
            if prev is None:
                r["_prev"] = f
            elif f != prev:
                r["steps"] += 1
                r0, c0 = divmod(int(prev), COLS)
                r1, c1 = divmod(f, COLS)
                _dr = r1 - r0
                _dc = (c1 - c0 + COLS) % COLS
                if _dc > COLS // 2:
                    _dc -= COLS                          # 取最短环绕
                _delta = (_dr, _dc)
                if r["_last"] == _delta:
                    r["same_dir"] += 1                   # 🔴 不依赖方位命名：连续两步同向
                r["_last"] = _delta
                r["_dirs"][_delta] = r["_dirs"].get(_delta, 0) + 1
                if r1 == r0 and (c1 - c0) % COLS == 1:
                    r["east"] += 1
                d = great_circle_cells(r["start_flat"], f)
                if d > r["max_dist"]:
                    r["max_dist"] = d
                    r["t_max_dist"] = t
                r["_prev"] = f
            if r["t_hungry"] is None and float(cur_en[k]) < 180.0:
                r["t_hungry"] = t
                r["steps_till_hungry"] = r["steps"]
                r["dist_till_hungry"] = great_circle_cells(r["start_flat"], f)
                r["max_dist_till_hungry"] = r["max_dist"]
        for i in rec:
            if rec[i]["t_death"] is None and i not in alive and rec[i].get("_prev") is not None:
                rec[i]["t_death"] = t
                if rec[i]["t_hungry"] is None:
                    rec[i]["t_hungry"] = t
                    rec[i]["steps_till_hungry"] = rec[i]["steps"]
                    rec[i]["dist_till_hungry"] = great_circle_cells(
                        rec[i]["start_flat"], rec[i]["_prev"])
                rec[i]["dist_total"] = great_circle_cells(
                    rec[i]["start_flat"], rec[i]["_prev"])
        if len(e._flat) == 0:
            break

    for r in rec.values():
        r.pop("_prev", None)
        r["_dirs"] = r["_dirs"] or {}
        _top = max(r["_dirs"].items(), key=lambda kv: kv[1]) if r["_dirs"] else ((0, 0), 0)
        r["top_dir"] = f"{_top[0][0]},{_top[0][1]}"
        r["top_dir_frac"] = (round(_top[1] / r["steps"], 3) if r["steps"] else None)
        r["same_dir_frac"] = (round(r["same_dir"] / r["steps"], 3) if r["steps"] else None)
        r.pop("_dirs", None)
        r.pop("_last", None)
    info = {"final": dict(getattr(e, "_run_deaths", {})), "n_left": len(e._flat)}
    return list(rec.values()), info


def band_of(start_flat) -> str:
    """按出生纬度分带（**必须分层**：近极纬圈很短，向东走几格就绕回来）。"""
    lat = abs(90.0 - (int(start_flat) // COLS + 0.5) * _DEG_PER_CELL)
    if lat < 30.0:
        return "赤道带(|lat|<30°)"
    if lat < 60.0:
        return "中纬(30-60°)"
    return "近极(>60°)"


def stats(rows: list[dict]) -> dict:
    def q(k, p):
        v = [r[k] for r in rows if r[k] is not None]
        return round(float(np.percentile(v, p)), 1) if v else None
    return {
        "n": len(rows),
        "到饿(180) tick": q("t_hungry", 50),
        "路径长度 中位": q("steps_till_hungry", 50),
        "路径长度 P10/P90": f"{q('steps_till_hungry', 10)} / {q('steps_till_hungry', 90)}",
        "最远能到 中位": q("max_dist", 50),
        "最远能到 最大": q("max_dist", 100),
        "何时到最远 tick": q("t_max_dist", 50),
        "同向步占比 中位": q("same_dir_frac", 50),
        "最常见方向占比 中位": q("top_dir_frac", 50),
        "分带": {b: stats_band([r for r in rows if r["band"] == b]) for b in
                 ("赤道带(|lat|<30°)", "中纬(30-60°)", "近极(>60°)")},
    }


def stats_band(rows: list[dict]) -> dict:
    if not rows:
        return {}
    def q(k, p):
        v = [r[k] for r in rows if r[k] is not None]
        return round(float(np.percentile(v, p)), 1) if v else None
    return {"n": len(rows), "步数中位": q("steps_till_hungry", 50),
            "最远能到": q("max_dist", 50),
            "最远(P90)": q("max_dist", 90),
            "主方向占比": q("top_dir_frac", 50)}


ARMS = [("A 直线走（ARS 惯性锁向）", True), ("B 随机走（同条件每tick必动）", False)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="直线续航探针")
    ap.add_argument("--seeds", default="42,7,11,101,202")
    ap.add_argument("--t-max", type=int, default=1200)
    ap.add_argument("--out", default="results/straightline_probe.csv")
    a = ap.parse_args(argv)
    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]

    print("== 直线续航探针：满能量600 + 饱胃25，每 tick 必走 1 格 ==")
    print("   （无食物 / 关捕食 / 禁繁殖；方向：向东沿纬线）\n")
    all_rows, table = [], {}
    t0 = time.time()
    for tag, straight in ARMS:
        rows = []
        for seed in seeds:
            rr, info = run(seed, straight, a.t_max)
            rows += rr
        for r in rows:
            r["arm"] = tag
            r["band"] = band_of(r["start_flat"])
        all_rows += rows
        s = stats(rows)
        table[tag] = s
        print(f"[{tag}] n={s['n']}")
        print(f"    到饿(180)={s['到饿(180) tick']} tick｜路径长度 {s['路径长度 中位']} 格"
              f"（P10/P90 {s['路径长度 P10/P90']}）")
        print(f"    最远能到 {s['最远能到 中位']} 格（最远个体 {s['最远能到 最大']} 格）"
              f"，峰值出现在第 {s['何时到最远 tick']} tick")
        print(f"    同向步占比 {s['同向步占比 中位']}（随机走应 ≈0.25）"
              f"｜最常见方向占比 {s['最常见方向占比 中位']}"
              f"  （{time.time()-t0:.0f}s）")
        for b, bs in s["分带"].items():
            if bs:
                print(f"      · {b:<18} n={bs['n']:<5} 步数中位={bs['步数中位']:<6} "
                      f"最远能到={bs['最远能到']:<7} (P90 {bs['最远(P90)']:<6}) "
                      f"主方向占比={bs['主方向占比']}")

    if a.out:
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=["arm", "seed", "id", "g1", "g6", "start_flat",
                                               "steps", "east", "same_dir_frac", "top_dir",
                                               "top_dir_frac", "max_dist", "t_max_dist",
                                               "t_hungry", "t_death", "steps_till_hungry",
                                               "dist_till_hungry", "max_dist_till_hungry",
                                               "dist_total"], extrasaction="ignore")
            w.writeheader()
            w.writerows(all_rows)
        print(f"\n✅ 明细 CSV：{out}（{len(all_rows)} 行）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
