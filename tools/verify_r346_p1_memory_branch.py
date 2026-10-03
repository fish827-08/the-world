# -*- coding: utf-8 -*-
"""🔴 v1.2 复核的实跑证据：`memory_v2` 两分支下`mem_bit` 实测。

背景：天平 `tools/check_p1a_switches.py` 输出里写
    「memory_v2=False ⇒ 未开 ⇒ _work_memory 全 -1 ⇒ mem_bit 恒 0 ⇒ P1-b 必 FAIL」
    「🔧 怎么修：开 memory_v2」
🔴 **本脚本证明该提示方向反了**（`sphere_engine.py:3584-3593`）：

    if _mv2_food_on:      # memory_v2=True  → 走 _mem_v2_write（**不写 _work_memory**）
    elif food_rich.any(): # memory_v2=False → **正常写 _work_memory** ← 默认走这条

⇒ `False` 才是"有记忆"的分支，`True` 反而让 mem_bit 归零。
⇒ **照天平脚本的"怎么修"去开 memory_v2，会把 P1-b 打成 0。**

纪律：本脚本**只读**、不写任何项目文件；跑在**缩小档**（省机时），
      🔴 **不是 device s2 档** ⇒ 数值不可直接引用到正式批，只用于**方向判定**。

用法（项目根目录）：
    .venv/Scripts/python.exe tools/verify_r346_p1_memory_branch.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from simulation.config import SimConfig          # noqa: E402
from simulation.sphere_engine import SphereEngine  # noqa: E402

# 缩小档（🔴 非device s2，仅方向判定用）
ROWS, COLS, PATCHES, POP, TICKS, SEED = 120, 240, 60, 300, 2000, 201

_flag = 0
_hits = 0


def check(label: str, ok: bool, note: str = "") -> None:
    global _flag, _hits
    _hits += 1
    if not ok:
        _flag += 1
    print(f"[{'FLAG' if not ok else 'OK  '}] {label}")
    if ok:
        print(f"         命中 True{'  ' + note if note else ''}")


def run(memory_v2: bool) -> tuple[float | None, int, int, int]:
    cfg = SimConfig()
    cfg.rows, cfg.cols, cfg.patches, cfg.pop = ROWS, COLS, PATCHES, POP
    cfg.seed = SEED
    cfg.info_structure.enabled = True
    cfg.info_structure.memory_v2 = bool(memory_v2)
    if memory_v2:
        # 硬 assert：memory_v2 要求 memory_gradient == "orientation"
        cfg.info_structure.memory_gradient = "orientation"
    cfg.signal_alphabet = "8"
    eng = SphereEngine(cfg)
    for _ in range(TICKS):
        eng.step()
    on, n = int(eng._mem_bit_on), int(eng._mem_bit_n)
    frac = round(on / n, 6) if n else None
    wm = eng._work_memory[: len(eng._id)]
    nonempty = int((wm >= 0).sum())
    print(f"       memory_v2={str(memory_v2):5s} ⇒ _mem_bit_on={on:6d} "
          f"_mem_bit_n={n:6d} frac={frac} 记忆非空槽={nonempty}")
    return frac, nonempty, on, n


print("=" * 74)
print("【实跑】memory_v2 两分支（120x240 / pop 300 / 2000 tick / alphabet=8）")
print("=" * 74)

f_off, n_off, on_off, n_n_off = run(False)
print()
f_on, n_on, on_on, n_n_on = run(True)

print()
print("=" * 74)
print("【判定】")
print("=" * 74)

check("断言1：memory_v2=False ⇒ _work_memory **有内容**（默认分支是 `elif food_rich`）",
      n_off > 0, f"非空槽={n_off}")
check("断言2：memory_v2=False ⇒ mem_bit_frac **> 0**（P1-b 天然可通过）",
      bool(f_off) and f_off > 0, f"frac={f_off}")
check("断言3：memory_v2=True ⇒ _work_memory **全空**（v2 取代 v1，不写 _work_memory）",
      n_on == 0, f"非空槽={n_on}")
check("断言4：memory_v2=True ⇒ mem_bit_frac == 0（**开它会让 P1-b 归零**）",
      f_on == 0.0, f"frac={f_on}")
check("断言5：🔴 天平脚本的『怎么修=开 memory_v2』方向**反了**"
      "（False 才是有记忆的分支）",
      (n_off > 0 and f_off > 0) and (n_on == 0 and f_on == 0.0),
      "⇒ 保持 False；开 True 会把 mem_bit 打成 0")

print()
print("=" * 74)
print(f"合计 {_hits} 项；FLAG 数 = {_flag}")
print("含义：FLAG = 我的 v1.2 复核意见与实测不一致。")
print("⚠️ 档位是缩小档（120x240），**不是 device s2** ⇒ 数值不可直接引用到正式批。")
print("   方向结论（False 才有记忆 / True 归零）与档位无关，由代码分支决定。")
