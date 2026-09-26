"""稳态 K 探针（14.10 P0.2）—— 实测「世界放大后到底能养活多少个体」。

为什么必须实测
--------------
我们现在的 `K ≈ 3.67 × 产能格` 是在 **60×120** 上标定的线性外推。
评审（2026-09-26）指出这很可能是高估：世界放大后，斑块间距变大，
"找不到食物"本身会压低**实际** K —— 食物长满了没人吃。
⇒ 在决定要不要投稀疏化那套大工程之前，必须先把这个数测出来。

本探针回答三件事
----------------
1. **稳态个体数 K**：在每个斑块密度下跑到种群平台期，报平台期中位数
2. **资源饱和度**：斑块存量/容量（≈1 ⇒ 食物堆着没人吃 ⇒ 瓶颈是"可达性"不是"产量"）
3. **性能耦合**：不同 N 下的 ms/tick，用来估"食物变多 ⇒ 生物变多 ⇒ 慢多少"

口径（B4）
----------
* **K** = 最后 20% tick 的种群中位数（先断言尾部斜率已平）
* **产能格** = `capacity > 0` 的格数（`bg_production_zero=True` ⇒ 只有斑块有产能）
* **资源饱和度** = `Σ存量 / Σ容量`（全部产能格）
* 🔴 **1 − 此值 ≠ 被吃掉的比例**（后者见 `food_util_frac`，R190/E-035，实测 ~2%）
  —— 别把"食物堆着"读成"食物被充分利用"（R211 命名裁定）
* **外推 K** = 3.67 × 产能格（旧标定，用于对照）

用法
----
    .venv\\Scripts\\python.exe experiments/steady_k_probe.py --patches 30,60,120 --ticks 6000
    .venv\\Scripts\\python.exe experiments/steady_k_probe.py --patches 30 --subpos on --ticks 4000
    .venv\\Scripts\\python.exe experiments/steady_k_probe.py --patches 480 --max-count 30000 --k 2.5 --subpos on
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simulation.config import SimConfig                      # noqa: E402
from simulation.sphere_engine import SphereEngine            # noqa: E402
from experiments.scaling_rescale import (                    # noqa: E402
    apply_post_build, rescale_config,
)

# --- R98 纪律：Windows GBK 控制台兜底（非 ASCII print 会让脚本 rc=1 假失败；F-R15 族）---
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass  # 非 TTY / 旧解释器：不因诊断能力缺失而阻断运行


K_PER_CELL = 3.6667      # 旧标定：3.67 体/产能格（60×120 上测的）


def _bar(cur: int, tot: int, extra: str = "", width: int = 24) -> str:
    frac = cur / max(tot, 1)
    done = int(frac * width)
    return f"[{'#' * done}{'.' * (width - done)}] {frac * 100:5.1f}% {extra}"


def make_cfg(seed: int, rows: int, cols: int, pop: int, patches: int,
             subpos: bool, speed_max: float, gain: float, subdiv: int,
             k: float = 1.0, max_count: int = 0) -> tuple[SimConfig, dict]:
    c = SimConfig(seed=seed)
    c.world.rows, c.world.cols = rows, cols
    c.resources.distribution = "patchy"
    c.resources.bg_production_zero = True
    c.resources.patch_regrowth_mult = 1.195
    c.resources.patch_count = patches
    if pop > 0:
        c.population.initial_count = pop
    if max_count > 0:
        c.population.max_count = int(max_count)   # 撞顶则实测=配置读数（T4 口径 30000）
    if subpos:
        c.simulation.use_sim_core = False        # subpos 与 Rust 路径互斥（H3 硬报错）
        c.subpos.enabled = True
        c.subpos.speed_max = float(speed_max)
        c.subpos.speed_gain = float(gain)
        c.subpos.subdiv = int(subdiv)
    # 时间压缩（R205/R206）：`k=1` 逐位不变；k≠1 走项目统一重标（四类量纲表）
    notes = rescale_config(c, k) if abs(k - 1.0) > 1e-12 else {"signals_duration": int(c.signals.duration_ticks)}
    return c, notes


def run_one(seed: int, rows: int, cols: int, pop: int, patches: int, ticks: int,
            sample: int, subpos: bool, speed_max: float, gain: float, subdiv: int,
            max_minutes: float, stop_stable: int = 4, k: float = 1.0,
            max_count: int = 0) -> tuple[list[dict], dict]:
    cfg, notes = make_cfg(seed, rows, cols, pop, patches, subpos, speed_max, gain,
                          subdiv, k, max_count)
    t0 = time.time()
    eng = SphereEngine(cfg)
    apply_post_build(eng, notes)                 # 构造后项（信号寿命 ÷k）
    prod = eng.resources._capacity > 0
    n_prod = int(prod.sum())
    cap_sum = float(eng.resources._capacity[prod].sum())

    rows_out: list[dict] = []
    t_wall0 = time.time()
    t_slice = time.perf_counter()
    last = 0
    stop_reason = "tick 用尽"
    for t in range(1, ticks + 1):
        eng.step()
        if eng.world.n_cells and len(eng._flat) == 0:
            stop_reason = "灭绝"
        if t % sample == 0 or t == ticks or stop_reason == "灭绝":
            dt_ms = (time.perf_counter() - t_slice) / max(t - last, 1) * 1e3
            last = t
            t_slice = time.perf_counter()
            N = int(len(eng._flat))
            stock = float(eng.resources._grid[prod].sum())
            rows_out.append({
                "seed": seed, "patches": patches, "k": k, "tick": t, "pop": N,
                "saturation": (stock / cap_sum if cap_sum > 0 else float("nan")),
                "ms_per_tick": dt_ms,
                "mean_energy": float(eng._energy[:N].mean()) if N else float("nan"),
                "mean_gen": float(eng._generation[:N].max()) if N else float("nan"),
            })
            # ETA 只用**实测样本**推（R189：禁瞬时速率外推；此处用刚测完这一段的 ms/tick）
            eta_min = (ticks - t) * dt_ms / 1e3 / 60.0
            print("  " + _bar(t, ticks, f"t={t:<6} N={N:<7} 饱和度={stock / max(cap_sum, 1e-9):.3f}"
                                      f" {dt_ms:6.2f} ms/tick  ETA {eta_min:5.1f} min"), flush=True)
            # 早停：连续 stop_stable 个采样点相对变化 < 3% ⇒ 已到平台
            if stop_stable > 0 and len(rows_out) >= stop_stable + 1:
                _tail = [r["pop"] for r in rows_out[-(stop_stable + 1):]]
                _ok = all(abs(_tail[k + 1] - _tail[k]) <= 0.03 * max(_tail[k], 1)
                          for k in range(len(_tail) - 1))
                if _ok:
                    stop_reason = f"平台期（连续 {stop_stable} 段 <3%）"
                    break
            if stop_reason == "灭绝":
                break
        if (time.time() - t_wall0) / 60.0 > max_minutes:
            stop_reason = f"超时 {max_minutes} 分钟"
            break

    pop_tail = [r["pop"] for r in rows_out[len(rows_out) * 4 // 5:]] or [0]
    ms_tail = [r["ms_per_tick"] for r in rows_out[len(rows_out) * 4 // 5:]] or [0.0]
    sat_tail = [r["saturation"] for r in rows_out[len(rows_out) * 4 // 5:]]
    # 平台判据（T4 口径）：**末 3 个采样点**两两相对变化 ≤ 3%
    tail3 = [int(r["pop"]) for r in rows_out[-3:]]
    platform_ok = (len(tail3) == 3 and all(
        abs(tail3[i + 1] - tail3[i]) <= 0.03 * max(tail3[i], 1) for i in range(2)))
    summary = {
        "patches": patches, "seed": seed, "cells": rows * cols, "k": k,
        "productive_cells": n_prod,
        "K_extrapolated": round(K_PER_CELL * n_prod, 1),
        "K_measured": int(np.median(pop_tail)),
        "pop_max": max((r["pop"] for r in rows_out), default=0),
        "pop_final": pop_tail[-1],
        "tail3_pops": tail3,
        "platform_reached": bool(platform_ok),
        "saturation_tail": round(float(np.median(sat_tail)), 4) if sat_tail else None,
        "ms_per_tick_tail": round(float(np.median(ms_tail)), 2),
        "max_gen": max((r["mean_gen"] for r in rows_out if r["mean_gen"] == r["mean_gen"]),
                       default=0),
        "ticks_done": rows_out[-1]["tick"] if rows_out else 0,
        "wall_s": round(time.time() - t0, 1),
        "stop": stop_reason,
    }
    return rows_out, summary


def main() -> None:
    ap = argparse.ArgumentParser(description="稳态 K 实测（世界放大后能养活多少个体）")
    ap.add_argument("--rows", type=int, default=480)
    ap.add_argument("--cols", type=int, default=960)
    ap.add_argument("--patches", default="30,60,120")
    ap.add_argument("--pop", type=int, default=2000)
    ap.add_argument("--ticks", type=int, default=6000)
    ap.add_argument("--sample", type=int, default=250)
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--subpos", choices=("off", "on"), default="off")
    ap.add_argument("--speed-max", type=float, default=0.25)
    ap.add_argument("--gain", type=float, default=0.25)
    ap.add_argument("--subdiv", type=int, default=20)
    ap.add_argument("--max-minutes", type=float, default=25.0)
    ap.add_argument("--k", type=float, default=1.0,
                    help="时间压缩倍率（R205 定档 k=2.5 ⇒ 昼夜 960）；k=1 逐位不变")
    ap.add_argument("--max-count", type=int, default=0,
                    help="population.max_count 覆盖（0=用配置默认 5000）；T4 口径 30000——"
                         "撞顶则实测变成配置读数")
    ap.add_argument("--stop-stable", type=int, default=4,
                    help="连续 N 个采样点相对变化 < 3 个百分点即判平台并早停（0=关）")
    ap.add_argument("--out", default="results/steady_k_probe.csv")
    a = ap.parse_args()

    patch_list = [int(x) for x in a.patches.split(",") if x.strip()]
    seeds = [int(x) for x in a.seeds.split(",") if x.strip()]

    print(f"== 稳态 K 探针：{a.rows}x{a.cols}（{a.rows * a.cols:,} 格）"
          f"，斑块 {patch_list}，初始 {a.pop}，{a.ticks} tick，"
          f"subpos={a.subpos}，k={a.k:g}，max_count={a.max_count or '默认'}，seed {seeds} ==")
    print(f"   旧标定外推公式：K ≈ {K_PER_CELL:.2f} × 产能格\n")

    all_rows: list[dict] = []
    summaries: list[dict] = []
    for p in patch_list:
        for sd in seeds:
            print(f"--- 斑块 {p} / seed {sd} ---", flush=True)
            rows_out, s = run_one(sd, a.rows, a.cols, a.pop, p, a.ticks, a.sample,
                                  a.subpos == "on", a.speed_max, a.gain, a.subdiv,
                                  a.max_minutes, a.stop_stable, a.k, a.max_count)
            all_rows.extend(rows_out)
            summaries.append(s)
            print(f"    ⇒ 产能格 {s['productive_cells']}｜外推 K {s['K_extrapolated']}"
                  f"｜**实测 K {s['K_measured']}**（峰值 {s['pop_max']}）"
                  f"｜饱和度 {s['saturation_tail']}｜{s['ms_per_tick_tail']} ms/tick"
                  f"｜{s['wall_s']}s｜{s['stop']}\n", flush=True)

    print("=" * 100)
    print(f"{'斑块数':>7}{'产能格':>9}{'外推K':>9}{'实测K':>9}{'比值':>8}"
          f"{'峰值':>8}{'饱和度':>9}{'ms/tick':>9}{'墙钟s':>8}{'代数':>6}{'平台':>5}  终止")
    print("-" * 100)
    for s in summaries:
        ratio = (s["K_measured"] / s["K_extrapolated"]) if s["K_extrapolated"] else float("nan")
        print(f"{s['patches']:>7}{s['productive_cells']:>9}{s['K_extrapolated']:>9.0f}"
              f"{s['K_measured']:>9}{ratio:>8.2f}{s['pop_max']:>8}"
              f"{s['saturation_tail']:>9}{s['ms_per_tick_tail']:>9.2f}{s['wall_s']:>8.0f}"
              f"{s['max_gen']:>6.0f}{'是' if s['platform_reached'] else '否':>4}  {s['stop']}")

    out = Path(a.out)
    if a.out and all_rows:
        out.parent.mkdir(parents=True, exist_ok=True)
        with io.open(out, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)
        print(f"\n明细已写入 {out}（{len(all_rows)} 行）")
        # 伴生 summary.json：供 `experiments/batch_runner.py` 判 done（`result.final_N` + 跑满
        # `switches.ticks_target`）；单 run 调用时顶层即该 run，多 run 时附全量于 `runs`
        last = summaries[-1]
        doc = {
            "switches": {"ticks_target": int(a.ticks), "rows": a.rows, "cols": a.cols,
                         "patches": last["patches"], "k": last["k"], "seeds": seeds,
                         "subpos": a.subpos, "speed_max": a.speed_max, "gain": a.gain,
                         "subdiv": a.subdiv, "max_count": a.max_count or None},
            "result": {"final_N": int(last["pop_final"]), "final_tick": int(last["ticks_done"]),
                       "platform_reached": bool(last["platform_reached"]),
                       "K_measured": last["K_measured"],
                       "K_extrapolated": last["K_extrapolated"],
                       "tail3_pops": last["tail3_pops"], "max_gen": last["max_gen"],
                       "saturation_tail": last["saturation_tail"],
                       "ms_per_tick_tail": last["ms_per_tick_tail"], "stop": last["stop"]},
            "runs": summaries,
        }
        sp = out.with_suffix(".summary.json")
        with io.open(sp, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, ensure_ascii=False, indent=2)
        print(f"summary 已写入 {sp}")

    print("\n读法：")
    print("  · **实测 K / 外推 K < 1** ⇒ 评审说得对，线性外推高估了（世界放大后食物找不到）")
    print("  · **资源饱和度接近 1** ⇒ 瓶颈是「可达性」不是「产量」⇒ 加食物不涨种群"
          "（注意：饱和度 ≠ 被吃掉的比例）")
    print("  · ms/tick 随 N 增长（次线性）⇒ 这一列是「生物变多会不会影响性能」的答案")


if __name__ == "__main__":
    main()
