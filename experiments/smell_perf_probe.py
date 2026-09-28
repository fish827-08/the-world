"""气味场性能实测（R240 T8 / 设计稿 §8.3-1）—— **配对口径**（R219）：相对值 + N + 并发 + 分解。

为什么必须实测
--------------
设计稿 §8.2 的红线：气味场是"每 tick 全场更新"的东西，**正是稀疏化要消灭的地板税类型**
（稀疏化 B 已把 N=0 地板从 13.63 压到 0.13 ms/tick）。所以本探针只回答一个问题：

    **开气味场（1–4 通道）到底多花多少 ms/tick？**（在 N=0 地板档与 N≈2000 档分别测）

口径（写死，防"数字没出处"）
----------------------------
* **配对**：同一世界/种子/tick 数，跑 **关档** 与 **开档** 两条臂 ⇒ 报 `Δ = on − off` 与相对比；
* **并发**：本探针单进程、串行两臂（无并发干扰）；报 `concurrency=1`；
* **分解**：`SmellField.probe()['ms_per_update']` 的 4 类（衰减 / 注入 / 粗网格拉普拉斯 / 上采样叠加）
  ÷ `update_every` ⇒ 折成 **ms/tick**；
* **N=0 地板档**：`--pop 0`（无个体）⇒ 直接对照"稀疏化成果有没有被吃回去"。

用法
----
    python3.12 experiments/smell_perf_probe.py                       # 默认三档
    python3.12 experiments/smell_perf_probe.py --quick                # 少 tick（冒烟）
输出：`results/smell_v1/smell_perf.json` + `smell_perf.csv`（**产物入数据仓**，见 R240 §〇-4）
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

# --- R98 纪律：Windows GBK 控制台兜底（非 ASCII print 会让脚本 rc=1 假失败；F-R15 族）---
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass  # 非 TTY / 旧解释器：不因诊断能力缺失而阻断运行


def _mk(rows: int, cols: int, pop: int, patches: int, channels, ticks: int,
        seed: int = 42, sparse: bool = False) -> SimConfig:
    c = SimConfig(seed=seed)
    c.world.rows, c.world.cols = rows, cols
    c.resources.distribution = "patchy"
    c.resources.bg_production_zero = True
    c.resources.patch_regrowth_mult = 1.195
    c.resources.patch_count = patches
    if pop > 0:
        c.population.initial_count = pop
    else:
        c.population.initial_count = 0
    c.smell.channels = tuple(channels)
    c.simulation.sparse_fields = bool(sparse)   # 稀疏化 B 开关（正交：本探针用它给"地板"对照）
    return c


def run_arm(rows: int, cols: int, pop: int, patches: int, channels, ticks: int,
            warm: int = 20, sparse: bool = False) -> dict:
    """跑一条臂（warm 预热不计时）⇒ ms/tick + 气味场分解。"""
    cfg = _mk(rows, cols, pop, patches, channels, ticks, sparse=sparse)
    eng = SphereEngine(cfg)
    for _ in range(warm):
        eng.step()
    t0 = time.perf_counter()
    for _ in range(ticks):
        eng.step()
    ms = (time.perf_counter() - t0) / max(ticks, 1) * 1e3
    N = int(len(eng._flat))
    out = {"ms_per_tick": round(ms, 4), "N_end": N, "channels": list(channels),
           "concurrency": 1, "sparse_fields": bool(sparse),
           "mem_field_MB": round(len(channels) * rows * cols * 8 / 1e6, 3)}
    if eng.smell is not None:
        p = eng.smell_probe()
        k = max(int(p["update_every"]), 1)
        out["downsample_eff"] = p["downsample_eff"]
        out["updates"] = p["updates"]
        out["per_tick_ms"] = {kk: round(vv / k, 4) for kk, vv in p["ms_per_update"].items()}
        out["inj_cells_last"] = p["inj_cells_last"]
        # 读路径（**只给"被读格"做插值**，不进每 tick 路径）与物化整场（诊断用）的单次成本
        n_read = min(int(len(eng._flat)), 1000)
        if n_read > 0:
            cells = eng._flat[:n_read].astype(np.int64)
        else:                       # N=0（地板档）⇒ 用等距格样本，保证读路径也有数
            n_read = 1000
            cells = np.linspace(0, eng.world.n_cells - 1, n_read).astype(np.int64)
        t2 = time.perf_counter()
        eng.smell.at(cells, channels[0])
        out["read_ms_per_call"] = round((time.perf_counter() - t2) * 1e3, 4)
        out["read_cells"] = n_read
        t3 = time.perf_counter()
        eng.smell.to_full()
        out["to_full_ms_per_call"] = round((time.perf_counter() - t3) * 1e3, 3)
    return out


def use_delta(rows: int, cols: int, pop: int, patches: int, channels, ticks: int) -> dict:
    """R244 §二：**消费端**配对 —— 同一世界/种子/tick 跑三臂：
    场关 ｜ 场开+消费关 ｜ 场开+消费开 ⇒ 报"消费端净增"与"合计"。
    """
    a = run_arm(rows, cols, pop, patches, (), ticks)
    b = run_arm(rows, cols, pop, patches, channels, ticks)
    c = run_use_arm(rows, cols, pop, patches, channels, ticks)
    return {
        "field_off": a, "field_on_use_off": b, "field_on_use_on": c,
        "delta_field_ms": round(b["ms_per_tick"] - a["ms_per_tick"], 4),
        "delta_use_ms": round(c["ms_per_tick"] - b["ms_per_tick"], 4),
        "delta_total_ms": round(c["ms_per_tick"] - a["ms_per_tick"], 4),
        "rel_use_pct": round((c["ms_per_tick"] - b["ms_per_tick"])
                             / max(b["ms_per_tick"], 1e-9) * 100.0, 2),
    }


def _mk_use(rows, cols, pop, patches, channels, use, sparse=False):
    c = _mk(rows, cols, pop, patches, channels, 0, sparse=sparse)
    c.smell.use_in_move = bool(use)
    return c


def run_use_arm(rows: int, cols: int, pop: int, patches: int, channels, ticks: int) -> dict:
    """场开 + 消费开 臂（与 `run_arm` 同口径，只多开消费端）。"""
    cfg = _mk_use(rows, cols, pop, patches, channels, True)
    eng = SphereEngine(cfg)
    for _ in range(20):
        eng.step()
    t0 = time.perf_counter()
    for _ in range(ticks):
        eng.step()
    ms = (time.perf_counter() - t0) / max(ticks, 1) * 1e3
    p = eng.smell_probe()
    return {"ms_per_tick": round(ms, 4), "N_end": int(len(eng._flat)),
            "read_cells": p["read_path_cells"], "hat_clip_n": p["hat_clip_n"],
            "concurrency": 1, "use_in_move": True}


def paired(rows: int, cols: int, pop: int, patches: int, channels, ticks: int,
           sparse: bool = False) -> dict:
    """同一配置跑 关档 / 开档 两臂 ⇒ Δ 与相对比（配对口径）。"""
    off = run_arm(rows, cols, pop, patches, (), ticks, sparse=sparse)
    on = run_arm(rows, cols, pop, patches, channels, ticks, sparse=sparse)
    delta = on["ms_per_tick"] - off["ms_per_tick"]
    return {
        "world": [rows, cols], "pop_init": pop, "patches": patches,
        "channels": list(channels), "n_chan": len(channels), "ticks": ticks,
        "sparse_fields": bool(sparse),
        "off": off, "on": on,
        "delta_ms_per_tick": round(delta, 4),
        "rel": round(delta / off["ms_per_tick"], 4) if off["ms_per_tick"] > 0 else None,
        "share_of_off_pct": round(delta / off["ms_per_tick"] * 100.0, 2)
        if off["ms_per_tick"] > 0 else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="气味场性能实测（配对口径）")
    ap.add_argument("--quick", action="store_true", help="少 tick（冒烟用）")
    ap.add_argument("--out", default="results/smell_v1")
    a = ap.parse_args()
    ticks_big = 120 if a.quick else 300
    ticks_n0 = 400 if a.quick else 1200

    cases = [
        # (说明, rows, cols, pop, patches, channels, ticks, sparse_fields)
        ("A N=0 地板（480×960，一通道）", 480, 960, 0, 480, ("food",), ticks_n0, False),
        ("B N=0 地板（480×960，四通道）", 480, 960, 0, 480,
         ("food", "prey", "risk", "kin"), ticks_n0, False),
        ("C N≈2000（480×960，一通道）", 480, 960, 2000, 480, ("food",), ticks_big, False),
        ("D N≈2000（480×960，四通道）", 480, 960, 2000, 480,
         ("food", "prey", "risk", "kin"), ticks_big, False),
        ("E 小世界冒烟（60×120，四通道）", 60, 120, 200, 30,
         ("food", "prey", "risk", "kin"), 400 if a.quick else 1200, False),
        # 🔴 红线对照：**稀疏化 B 开档**（地板已被压到 ~0.13）时，气味场的 Δ 是否仍小
        ("F N=0 + 稀疏化开（480×960，四通道）", 480, 960, 0, 480,
         ("food", "prey", "risk", "kin"), ticks_n0, True),
    ]
    print("== 气味场性能实测（配对口径：同世界/种子/tick，关档 vs 开档）==")
    print(f"   并发 = 1（单进程串行两臂）｜ tick 数：小档 {ticks_n0} / 大档 {ticks_big}\n")
    hdr = f"{'档':<34}{'关档':>9}{'开档':>9}{'Δ':>9}{'Δ/关档':>9}{'N末':>7}{'场MB':>7}"
    print(hdr)
    print("-" * len(hdr.encode("gbk", errors="replace")))
    rows_out = []
    for name, r, c, pop, patch, chans, ticks, sparse in cases:
        res = paired(r, c, pop, patch, chans, ticks, sparse=sparse)
        rows_out.append({"case": name, **res})
        print(f"{name:<34}{res['off']['ms_per_tick']:>9.3f}{res['on']['ms_per_tick']:>9.3f}"
              f"{res['delta_ms_per_tick']:>9.3f}{res['share_of_off_pct']:>8.2f}%"
              f"{res['on']['N_end']:>7}{res['on']['mem_field_MB']:>7.2f}")
        if "per_tick_ms" in res["on"]:
            pt = res["on"]["per_tick_ms"]
            print(f"{'':<34}    分解（ms/tick）：衰减 {pt.get('decay', 0):.4f}｜注入 {pt.get('inject', 0):.4f}"
                  f"｜粗网格扩散 {pt.get('coarse_lap', 0):.4f}｜s_eff {res['on'].get('downsample_eff')}"
                  f"｜注入格 {res['on'].get('inj_cells_last')}")
            print(f"{'':<34}    读路径（非每 tick）：插值@N={res['on'].get('read_cells')} "
                  f"{res['on'].get('read_ms_per_call')} ms/次｜物化整场 {res['on'].get('to_full_ms_per_call')} ms/次")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "smell_perf.json").write_text(
        json.dumps({"[R240·T8·云端开发·云启]": "气味场 v1 性能实测（配对口径）",
                    "cases": rows_out}, ensure_ascii=False, indent=2), encoding="utf-8")
    with io.open(out / "smell_perf.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "case", "world", "pop_init", "patches", "n_chan", "ticks",
            "off_ms", "on_ms", "delta_ms", "share_of_off_pct", "N_end", "mem_field_MB",
            "decay_ms", "inject_ms", "coarse_lap_ms", "read_ms_per_call",
            "read_cells", "to_full_ms_per_call"])
        w.writeheader()
        for r in rows_out:
            pt = r["on"].get("per_tick_ms", {})
            w.writerow({"case": r["case"], "world": f"{r['world'][0]}x{r['world'][1]}",
                        "pop_init": r["pop_init"], "patches": r["patches"],
                        "n_chan": r["n_chan"], "ticks": r["ticks"],
                        "off_ms": r["off"]["ms_per_tick"], "on_ms": r["on"]["ms_per_tick"],
                        "delta_ms": r["delta_ms_per_tick"], "share_of_off_pct": r["share_of_off_pct"],
                        "N_end": r["on"]["N_end"], "mem_field_MB": r["on"]["mem_field_MB"],
                        "decay_ms": pt.get("decay"), "inject_ms": pt.get("inject"),
                        "coarse_lap_ms": pt.get("coarse_lap"),
                        "read_ms_per_call": r["on"].get("read_ms_per_call"),
                        "read_cells": r["on"].get("read_cells"),
                        "to_full_ms_per_call": r["on"].get("to_full_ms_per_call")})
    print(f"\n产物：[云端开发·云启] 已写 {out}/smell_perf.json + .csv（**入数据仓**，R240 §〇-4）")

    # ── R244 §二：消费端配对（三臂 × 两个 N 档 —— 读数随 N×候选数线性，须显式报 N）──
    print("\n== R244 §二 气味场**消费端**配对（480×960，四通道）==")
    chans4 = ("food", "prey", "risk", "kin")
    for _pop in (2000, 200):
        ud = use_delta(480, 960, _pop, 480, chans4, ticks_big)
        print(f"   N≈{_pop}：场关 {ud['field_off']['ms_per_tick']:.3f} ｜ 场开·消费关 "
              f"{ud['field_on_use_off']['ms_per_tick']:.3f} ｜ 场开·消费开 "
              f"{ud['field_on_use_on']['ms_per_tick']:.3f} ms/tick")
        print(f"          ⇒ 场本体 Δ {ud['delta_field_ms']:+.3f}｜**消费端净增 "
              f"{ud['delta_use_ms']:+.3f}**（{ud['rel_use_pct']:+.2f}%）｜合计 "
              f"{ud['delta_total_ms']:+.3f} ms/tick｜N末 {ud['field_on_use_on']['N_end']}"
              f"｜截断计数 {ud['field_on_use_on']['hat_clip_n']}")
        (out / f"smell_use_perf_N{_pop}.json").write_text(
            json.dumps({"[R244·§二·云端开发·云启]": f"气味场消费端配对实测（三臂，pop_init={_pop}）",
                        **ud}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"   ⇒ 另存 {out}/smell_use_perf_N*.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())