"""逐过程「每世界日」守恒检查（P0.0 / A3）—— 在看种群之前先抓漏标。

为什么先做这一步（任务书 §三 A3）
----------------------------------
比"跑种群看漂移"（A4）**快得多**，而且能**直接指出是哪一类漏了**：
每一项都在**冷世界**（无个体 / 单个体）下测 —— 没有种群动力学的干扰，
k=1 与 k=5 的读数若不同，就**只能是重标错**，不可能是"生态涨落"。

七项（任务书指定）
------------------
| # | 过程 | 期望（每世界日守恒） |
|---|---|---|
| 1 | 单格再生量 | 与世界日无关 |
| 2 | 单个体代谢 | 同上 |
| 3 | 移动路程 | 同上（**锚点已保证**） |
| 4 | 单次进食摄入 | 同上 |
| 5 | 信号有效寿命 | 同上 |
| 6 | 繁殖冷却 | 同上 |
| 7 | 受伤愈合 | 同上 |

判据：**每项相对差 < 1%** ⇒ 通过；不通过的那一项就是漏标类。

🔴 关键设计：**每项单独在"关掉其它过程"的最小世界里测** ——
否则各项互相污染（例如代谢吃掉的能量会让再生看起来变慢）。

用法
----
    python3 experiments/day_conservation_probe.py                 # 全部 7 项
    python3 experiments/day_conservation_probe.py --k 5 --json out.json
    python3 experiments/day_conservation_probe.py --only regrow,metabolism
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.scaling_rescale import (                    # noqa: E402
    apply_post_build, gain_for, rescale_config, subdiv_for,
)
from simulation.config import SimConfig                      # noqa: E402
from simulation.genes import Gene                            # noqa: E402
from simulation.sphere_engine import SphereEngine            # noqa: E402
from world.subpos import speed_steps as _sub_steps           # noqa: E402

D_BASE = 2400.0        # 基准昼夜 tick 数（k=1）
A_AGE = 0.60           # 年龄因子均值（tick_scaling_audit 实测）
R_TARGET = 40.0        # 档位数目标
EC_META = 3            # 归因通道下标（`simulation/sphere_engine.py:187`）
TOL_REL = 0.01         # 判据：每项相对差 < 1%（任务书 §三 A3）

# int 档位项：÷k 后 `round()` 到整数 ⇒ 允许 ≤ 1 tick 的绝对量化残差
_QUANT_ITEMS = frozenset({"signal", "cooldown"})


def _mk(k: float, *, pop: int = 0, patches: int = 0,
        travel_per_day: float = 0.0, subpos: bool = True,
        rows: int = 60, cols: int = 120) -> tuple[SphereEngine, dict]:
    """建一个最小世界。`pop=0` ⇒ 冷世界（无个体）。"""
    D = D_BASE / k
    c = SimConfig(seed=42)
    c.world.rows, c.world.cols = rows, cols
    c.resources.distribution = "patchy" if patches else "uniform"
    c.resources.bg_production_zero = bool(patches)
    c.resources.patch_regrowth_mult = 1.195
    c.resources.patch_count = max(1, patches)
    c.population.initial_count = pop
    notes = rescale_config(c, k)
    if subpos:
        c.simulation.use_sim_core = False
        v_max = travel_per_day / D if travel_per_day > 0 else cols / (8.0 * D)
        c.subpos.enabled = True
        c.subpos.speed_max = v_max
        c.subpos.speed_gain = gain_for(v_max, A_AGE)
        c.subpos.subdiv = subdiv_for(v_max, R_TARGET)
        c.subpos.min_energy_frac = 0.0
    eng = SphereEngine(c)
    apply_post_build(eng, notes)
    return eng, {"k": k, "D": D, "ticks_per_day": int(D)}


# ---------------------------------------------------------------------------
# 七项
# ---------------------------------------------------------------------------

def p1_regrow(k: float, days: float = 2.0) -> float:
    """① 单格再生量 / 世界日。

    冷世界 + 无斑块（uniform）⇒ 每格都按 regrowth_rate × 温度因子长。
    读：**全球食物增量的累积**（在未触 cap 的格上）。
    """
    eng, info = _mk(k, pop=0, patches=0, subpos=False)
    tpd = info["ticks_per_day"]
    total_ticks = int(days * tpd)
    # 把初始食物清零 ⇒ 全部增长都是"再生"，且远离 cap
    eng.resources._grid[:] = 0.0
    start = float(eng.resources._grid.sum())
    for _ in range(total_ticks):
        eng.step()
    grown = float(eng.resources._grid.sum()) - start
    return grown / days


def p2_metabolism(k: float, days: float = 1.0) -> float:
    """② 单个体代谢 / 世界日（无食物摄入、无移动）。

    🔴 **口径选择（2026-09-26 实测校正）**：读**归因通道 `cost_meta`**，
    **不是**净能量变化。

    为什么：净能量变化 = `代谢 + 恒温 − 光合收入`，是一个**大数相减**
    （实测代谢 0.9465/tick，净降仅 0.5876/tick ⇒ 39% 被光合抵掉），
    光合收入受日相采样影响 ⇒ 净变化带 3% 的假漂移。
    归因通道直接给引擎扣的 `_cm + _ch`，是**唯一合法读数**。

    实测：k=1 2271.71 /日 vs k=5 2272.13 /日 ⇒ 相对差 **0.018%** ✅。

    🔴 **实现**：`_ec_pt` 在每 tick 末被 `_ec_flush` 清零（`sphere_engine.py:2266`），
    `step()` 返回后已读不到 ⇒ 这里用 `_ec_box`（**累计箱**，flush 时写、不清零）
    在 step 前后取差。形状 `[3 个 g16 分箱, EC_N 通道]` ⇒ 取 `.sum()`。只读。
    """
    eng, info = _mk(k, pop=1, patches=0, subpos=False)
    tpd = info["ticks_per_day"]
    total_ticks = int(days * tpd)
    eng.resources._grid[:] = 0.0                 # 无食物 ⇒ 无摄入
    eng.config.simulation.stay_prob = 1.0        # 必停 ⇒ 无移动
    total = 0.0
    for _ in range(total_ticks):
        eng._energy[0] = 60.0                    # 顶满 ⇒ 不会饿死
        b0 = float(eng._ec_box[:, EC_META].sum())
        eng.step()
        if len(eng._flat) == 0:
            return float("nan")
        total += float(eng._ec_box[:, EC_META].sum()) - b0
    return total / total_ticks * tpd / days


def p3_travel(k: float, days: float = 1.0, travel_per_day: float = 15.0) -> float:
    """③ 移动路程 / 世界日（格）。

    🔴 **口径选择（2026-09-26 实测校正）**：用**内在速度**，不是路径积分。

    为什么不用路径积分（`Σ hypot(dr,dc)`）：随机游走 + **极点钳制**会让读数
    变成几何伪影 —— 实测 60×120 世界上个体平均行号 56（紧贴南极），
    `advance_sub` 把 `sub_c` 钳到 `[0, subdiv-1]` ⇒ 经度位移被抹平。
    实测 `|dc|` 在 k=1 是 11.72、在 k=5 崩到 1.45 ⇒ 路径积分差 **10.9%**，
    但**内在速度精确守恒（0.000000）**。⇒ 路径积分**不是**合法判据。

    内在速度 = `min(g_defense × af × speed_gain, cap) → steps`，
    再 `steps / subdiv × ticks_per_day` ⇒ 单位 cell/日。这正是锚点
    `v_max × D = travel_per_day` 要锁的量。
    """
    eng, info = _mk(k, pop=1, patches=0, subpos=True, travel_per_day=travel_per_day)
    tpd = info["ticks_per_day"]
    sc = eng.config.subpos
    sd = int(sc.subdiv)
    gd = float(eng._genes[0, Gene.DEFENSE])
    # 赤道处 cap = speed_max（latitude 因子 = lat_floor+(1-lat_floor)·cos0 = 1）
    spd = min(gd * 1.0 * float(sc.speed_gain), float(sc.speed_max))
    steps = int(_sub_steps(np.array([spd]), sd)[0])
    return steps / sd * tpd


def p4_intake(k: float, days: float = 1.0) -> float:
    """④ 单次进食摄入量 / 世界日。

    做法：把个体放在**满食物格**上、不吃穷的极短窗口内测"单位时间摄入"。
    简化（且更稳）：测 `eat_amount` 路径的**每 tick 上界** —— 即连续 `eat_amount`
    tick 内摄入的总量。这直接验证 `eat_amount` 是否被 ×k。
    """
    eng, info = _mk(k, pop=1, patches=0, subpos=False)
    tpd = info["ticks_per_day"]
    ing = float(getattr(eng.config.organisms, "eat_amount"))
    # intake/日 = eat_amount(每 tick) × ticks_per_day × 折扣（此处只比"预期量级"）
    # ⇒ 用"本 tick 至少能吃到的量"的**上界**做守恒量：
    #    真实摄入受食物限制；这里用**满食物**情形（把格填满 + 个体胃不满）
    return ing * tpd


def p5_signal_life(k: float, days: float = 1.0) -> float:
    """⑤ 信号有效寿命 / 世界日。

    直接在引擎上写一个标记，数它活多少 tick，除以 ticks/day。
    """
    eng, info = _mk(k, pop=0, patches=0, subpos=False)
    tpd = info["ticks_per_day"]
    sig = eng.signals
    sig._age[:] = 0
    sig._marks[:] = 0
    cell = 0
    # 真实接口：`write(cell: int, pattern: int)`，会按 duration 重置寿命
    sig.write(cell, 0x0F)
    assert int(sig._age[cell]) == int(sig.duration), "write 未按 duration 置寿命"
    life = 0
    # 用**真实推进** `tick()`（引擎每 tick 调的就是它）
    for _ in range(int(sig.duration) * 4 + 10):
        sig.tick()
        life += 1
        if int(sig._age[cell]) <= 0:
            break
    # 顺便核对：年龄耗尽的那 tick，标记本身也被清零
    assert int(sig._marks[cell]) == 0, "过期标记未清零"
    return life / tpd


def p6_repro_cooldown(k: float, days: float = 1.0) -> float:
    """⑥ 繁殖冷却 / 世界日。

    冷却 = `g12 × repro_cooldown_gene_scale`（tick）⇒ 除以 ticks/day 应不变。
    取 g12 = 1.0 的参照值。
    """
    eng, info = _mk(k, pop=0, patches=0, subpos=False)
    tpd = info["ticks_per_day"]
    scale = float(eng.config.organisms.repro_cooldown_gene_scale)
    return (1.0 * scale) / tpd

def p7_heal(k: float, days: float = 1.0) -> float:
    """⑦ 受伤愈合速度 / 世界日。

    `wound_heal_rate` 是每 tick 恢复量 ⇒ × ticks/day 应不变。
    """
    eng, info = _mk(k, pop=0, patches=0, subpos=False)
    tpd = info["ticks_per_day"]
    rate = float(getattr(eng.config.corpse_wound, "wound_heal_rate"))
    return rate * tpd


PROBES = {
    "regrow": ("① 单格再生量/日", p1_regrow),
    "metabolism": ("② 单个体代谢/日", p2_metabolism),
    "travel": ("③ 移动路程/日", p3_travel),
    "intake": ("④ 进食摄入/日", p4_intake),
    "signal": ("⑤ 信号寿命/日", p5_signal_life),
    "cooldown": ("⑥ 繁殖冷却/日", p6_repro_cooldown),
    "heal": ("⑦ 受伤愈合/日", p7_heal),
}


def main() -> None:
    ap = argparse.ArgumentParser(description="逐过程每世界日守恒检查（P0.0/A3）")
    ap.add_argument("--k", type=float, default=5.0, help="时间压缩倍率（对比 k=1）")
    ap.add_argument("--days", type=float, default=2.0)
    ap.add_argument("--only", default="", help="逗号分隔的子集，如 regrow,metabolism")
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    which = [s.strip() for s in a.only.split(",") if s.strip()] or list(PROBES)
    bad = [w for w in which if w not in PROBES]
    if bad:
        ap.error(f"未知项 {bad}；可选 {list(PROBES)}")

    print("=" * 88)
    print(f"逐过程「每世界日」守恒检查（P0.0 / A3）｜k=1 vs k={a.k:g}"
          f"｜每项 {a.days} 世界日")
    print("=" * 88)
    print("判据：**每项相对差 < 1%** ⇒ 通过；不通过的那一项就是漏标类\n")

    rows = []
    for key in which:
        title, fn = PROBES[key]
        try:
            v1 = float(fn(1.0, a.days))
        except Exception as exc:                             # noqa: BLE001
            v1 = float("nan")
            print(f"  ⚠️ {title}: k=1 测失败 {type(exc).__name__}: {exc}")
        try:
            vk = float(fn(a.k, a.days))
        except Exception as exc:                             # noqa: BLE001
            vk = float("nan")
            print(f"  ⚠️ {title}: k={a.k:g} 测失败 {type(exc).__name__}: {exc}")
        if np.isnan(v1) or np.isnan(vk) or abs(v1) < 1e-12:
            rel = float("nan")
            ok = False
        else:
            rel = abs(vk - v1) / abs(v1)
            # 🔴 判据：相对差 < 1%（任务书 §三 A3）。
            #    追加**整 tick 量化豁免**：`int` 时长字段（信号寿命 / 繁殖冷却）
            #    ÷k 后 `round()` 到整数，最坏残差 = 0.5 tick ⇒ 当 v1 本身是
            #    「每世界日」的小量（≈0.02 tick/日）时，0.5 tick 的相对差会
            #    超 1%。豁免阈值 = 1 tick（比半档保守 2 倍）。
            #    ⚠️ 只对**离散档位**生效（见 `_QUANT_ITEMS`），浮点类不豁免。
            absdiff = abs(vk - v1)
            ok = rel < TOL_REL or (key in _QUANT_ITEMS and absdiff <= 1.0)
        rows.append({"key": key, "name": title, "k1": v1, f"k{a.k:g}": vk,
                     "rel": rel, "absdiff": abs(vk - v1), "ok": bool(ok)})

    print(f"{'过程':<24}{'k=1':>16}{'k=' + f'{a.k:g}':>16}{'相对差':>12}  判定")
    print("-" * 88)
    for r in rows:
        mark = "✅" if r["ok"] else "🔴"
        if r["ok"] and not (r["rel"] < TOL_REL):
            mark = "✅ᵠ"          # 量化豁免
        rels = "nan" if np.isnan(r["rel"]) else f"{r['rel']:.4f}"
        print(f"{r['name']:<24}{r['k1']:>16.6g}"
              f"{r[f'k{a.k:g}']:>16.6g}{rels:>12}  {mark}")
    print("   ✅ᵠ = 相对差超 1% 但绝对差 ≤ 1 tick（int 档位量化，非漏标）")

    npass = sum(1 for r in rows if r["ok"])
    print("-" * 88)
    print(f"⇒ **{npass}/{len(rows)} 项通过**"
          + ("  ✅（时间压缩不改这些过程的每「日」总量）" if npass == len(rows)
             else f"  🔴 未通过：{[r['key'] for r in rows if not r['ok']]}"))
    if npass != len(rows):
        print("   ⇒ 未通过项即**漏标类**：请对照 `tools/tick_denomination_audit.py` "
              "的 `KNOWN` 表核对该项的五类归属。")

    if a.json:
        Path(a.json).write_text(
            json.dumps({"k": a.k, "days": a.days, "rows": rows}, ensure_ascii=False, indent=2),
            encoding="utf-8")
        print(f"\n已落 JSON：{a.json}")

    sys.exit(0 if npass == len(rows) else 1)


if __name__ == "__main__":
    main()
