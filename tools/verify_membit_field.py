#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P1 门字段静默坑 —— 复现与守卫（`[实验员·砚]`，2026-10-03，v1.2 复核 §二）

🔴 **为什么需要这个脚本**
------------------------
`memory_v2` **不在 `SimConfig` 顶层**，而在 `cfg.info_structure.memory_v2`
（`simulation/config.py:635`，属 `InfoStructureConfig`）。

而顶层 `SimConfig` **没有** `memory_v2` 属性（`hasattr(c, "memory_v2") == False`），
但**给它赋值不会报错** —— Python 允许给 dataclass 实例新建属性：

    cfg = SimConfig(); cfg.memory_v2 = True     # 不报错，但引擎完全读不到
    cfg.info_structure.memory_v2               # ← 引擎读的是这个，仍是 False

⇒🔴 **静默坑**：执行者以为自己开了 `memory_v2`，实际没开 ⇒ `_work_memory` 全程保持 `-1`
⇒ `mem_bit_frac` **恒 0** ⇒ P1-b 会 FAIL
⇒ 而执行者会误判为"**私有信息路线被证伪**"，实则只是"开关没开"。
⇒ **这是 R326「改装置只改预设档、实际未生效」的同族坑，只是这次发生在字段归属上。**

（A/B 开关对比的通用教训：**跑完A/B 两臂必须先断言"读回值确实不同"**，
否则 A/B 设计本身失效 —— 两臂没分开却比较结果。本脚本末尾即做此断言。）

**用法**
-----

    .venv/Scripts/python.exe tools/verify_membit_field.py

退出码：0 = 字段行为符合预期（可安全用于 P1-a）；1 = 行为与预期不符。
**本脚本只读，不改任何生产代码。**
"""
from __future__ import annotations

import sys
from dataclasses import fields
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, ok, detail))
    print(f"  {'✅' if ok else '❌'} {name}" + (f" —— {detail}" if detail else ""))


def mem_bit_frac(mv2: bool, ticks: int = 60, seed: int = 12345):
    """跑一读，读回 mem_bit_frac（口径同 `sphere_engine.py:1522`）。"""
    from simulation.config import SimConfig
    from simulation.sphere_engine import SphereEngine

    cfg = SimConfig()
    cfg.seed = seed
    cfg.signal_alphabet = "8"
    # 🔴 关键：字段在 info_structure 下，不是顶层
    cfg.info_structure.memory_v2 = mv2
    if mv2:
        # v2 开启要求 memory_gradient == "orientation"（config.py:668 的 C5 断言）
        cfg.info_structure.memory_gradient = "orientation"
    eng = SphereEngine(cfg)
    wm_initial = int((eng._work_memory >= 0).sum())
    for _ in range(ticks):
        eng.step()
    frac = (eng._mem_bit_on / eng._mem_bit_n) if eng._mem_bit_n else None
    wm_final = int((eng._work_memory >= 0).sum())
    return frac, eng._mem_bit_on, eng._mem_bit_n, wm_initial, wm_final


def main() -> int:
    from simulation.config import SimConfig
    from simulation.sphere_engine import SphereEngine  # noqa: F401  (确认可导入)

    print("=" * 72)
    print("P1 门字段静默坑 —— 复现与守卫")
    print("=" * 72)

    # ---------- 1. 字段归属 ----------
    print("\n[1] 字段归属（`memory_v2` 在哪一层）")
    top_has = hasattr(SimConfig(), "memory_v2")
    check("顶层 SimConfig 无 memory_v2 属性", not top_has,
          f"hasattr={top_has}")

    flds = {f.name for f in fields(SimConfig())}
    check("`memory_v2` 不在 SimConfig 的 dataclass 字段里", "memory_v2" not in flds,
          "⇒ 顶层赋值会成为『无用新属性』")

    cfg = SimConfig()
    cfg.memory_v2 = True                      # 🔴 故意写错一层（不报错！）
    check("顶层赋值被静默接受（这就是坑）", hasattr(cfg, "memory_v2"),
          "Python 允许给实例新建属性")
    check("但引擎读的那一层仍是 False（赋值无效）",
          cfg.info_structure.memory_v2 is False,
          "⇒ 引擎读 `info_structure.memory_v2`")

    # ---------- 2. A/B 两臂必须真的不同 ----------
    print("\n[2] A/B 复现（正确字段路径，两臂必须真的不同）")
    frac_off, on_off, n_off, wm0, wm1 = mem_bit_frac(False)
    frac_on, on_on, n_on, _wm0_on, wm1_on = mem_bit_frac(True)
    print(f"    memory_v2=False ⇒ mem_bit={on_off}/{n_off}frac={frac_off}")
    print(f"    memory_v2=True  ⇒ mem_bit={on_on}/{n_on}frac={frac_on}")

    check("A/B 两臂读回值确实不同（A/B 设计有效）",
          (frac_off or 0) > 0 and (frac_on or 0) == 0,
          f"off={frac_off} / on={frac_on}")

    # ---------- 3. 与 v1.2 记录的方向一致 ----------
    print("\n[3] 方向核验（v1.2 §5.1 记录的实测）")
    check("memory_v2=True ⇒ mem_bit 恒 0（v1.2 结论正确）", (frac_on or 0) == 0.0,
          f"实测 frac={frac_on}")
    check("memory_v2=False ⇒ mem_bit > 0（默认档天然可过 P1-b）", (frac_off or 0) > 0,
          f"实测 frac≈{frac_off}（v1.2 记≈28%；量级 0.3–0.5，非精确值）")
    check("memory_v2=False ⇒ _work_memory 有非空格", wm1 > 0, f"末态非空 {wm1} 格")
    check("memory_v2=True  ⇒ _work_memory 全 -1（v2 写新数组）", wm1_on == 0,
          f"末态非空 {wm1_on} 格")

    # ---------- 汇总 ----------
    n_ok = sum(1 for _, ok, _ in CHECKS if ok)
    n_all = len(CHECKS)
    print("\n" + "=" * 72)
    print(f"汇总：{n_ok}/{n_all} 通过")
    if n_ok != n_all:
        print("⇒ 🔴 有断言不符；若为 P1 语境，**先查字段与开关，不要下机制结论**")
    else:
        print("⇒ 🟢 字段行为符合预期；P1-a 断言可引用本脚本")
    print("=" * 72)
    print("\n📌 给 P1-a 自检脚本（tools/check_p1a_switches.py）的建议：")
    print("   断言必须读 `cfg.info_structure.memory_v2`，并显式打印所读对象；")
    print("   🔴 绝不可写 `cfg.memory_v2`（静默无效）。")
    print("   R339 §四 纪律：签名前先 `git log` + 板面核归属，禁代签。")

    return 0 if n_ok == n_all else 1


if __name__ == "__main__":
    raise SystemExit(main())