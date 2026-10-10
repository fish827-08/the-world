#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GAP-A-EMIT 只读核对：`sig_run_len_hist_1..8` 合计是否 ≡ 发射密度？

结论（本脚本实测给出）：**不等价，且与发射量不成比例** ⇒ M1 判据"发射率不降"
须另立专列（需求单 `_archive/2026-10-10-退役团队-归档/docs-旧档/tasks/需求单-GAP-A-EMIT-发射率专列-20261004.md`）。

纪律：🔴 **零机时**——不跑引擎、不消费 RNG、不改任何文件、不动已锁口径（R225）。
做法：直接驱动 s3 探针的**真实仪表代码** `_M0Instruments._track_runs`，
喂入四个手工构造的发射场景（真发射人·次已知），对照 G3 直方图合计。
写入一律经 stub 的 `signals.write_many`（= 引擎唯一写点，B1）⇒ 与 `task/G6-EMITRATE`
的类级挂钩同装同跑，两枝合并后本核对件仍可复跑（不是只在本枝自证）。

关键结构事实（读码所得，脚本量化）：`hist_run` 仅在**同一 vid 隔 tick 后再发一次**
时把上一条串入桶（`s3_memory_probe.py` `_track_runs` 的 else 分支）⇒
"停过之后再没发"的个体，其末串**永不入账**；连发不断的个体在窗口内**一条都不入**。

用法：
    python tools/audit_gap_a_emit.py            # 打表 + 结论
    python tools/audit_gap_a_emit.py --selftest # 断言"比值不稳定"（回归自锁）
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from experiments.s3_memory_probe import _M0Instruments   # noqa: E402

DUR, ROWS, COLS = 20, 4, 10
N_CELLS = ROWS * COLS
IDS = [100, 101, 102, 103, 104, 105]
CELL = {i: (i - 100) * 6 + 3 for i in IDS}      # 各占不同格，隔离 ambig 变量


class _World:
    cols, rows, _pole_top, _pole_bottom = COLS, ROWS, 0, ROWS - 1


class _Signals:
    def __init__(self) -> None:
        self._marks = np.zeros(N_CELLS, dtype=np.uint8)
        self._age = np.zeros(N_CELLS, dtype=np.int32)
        self.duration = DUR

    def write_many(self, cells, patterns):
        """🔴 B1（PI 21:4x 班 · 镜阻塞项）：照 `tests/test_r358_m0_instruments.py` 的 stub 做法。

        引擎唯一写入点 = `world/signal_field.py:104 write_many(cells, patterns)`
        （调用点 `sphere_engine.py:3664`）⇒ stub 也**只经它写**。`task/G6-EMITRATE` 的
        `_M0Instruments.attach` 会自检 `signals.write_many` 存在性并在 MRO 里找定义类装
        类级挂钩 ⇒ 本 stub 缺该法则两枝同合后本核对件**当场炸**（`RuntimeError: 🔴 M0 G6
        装配失败`），GAP-A 的证据就此不可复跑。写语义与真实件一致（`pat>0 ⇒ age=duration`）。
        """
        cells = np.asarray(cells, dtype=np.int64)
        pats = np.asarray(patterns, dtype=np.uint8)
        self._marks[cells] = pats
        self._age[cells] = np.where(pats > 0, self.duration, 0).astype(np.int32)


class _FakeEng:
    """只提供 `_M0Instruments` 只读访问的字段（attach 自检口径不变）。"""

    def __init__(self) -> None:
        self.world, self.signals = _World(), _Signals()
        self._id = np.array(IDS, dtype=np.int64)
        self._flat = np.array([CELL[i] for i in IDS], dtype=np.int64)
        self._nb_table = np.arange(N_CELLS, dtype=np.int64)[:, None].repeat(8, 1)
        self._pole_nb = np.zeros((2, 8), dtype=np.int64)
        self._mem_bit_on = self._mem_bit_n = 0
        self._genes = np.zeros((len(IDS), 24), dtype=np.float64)
        self._alpha = "8"


def measure(sched: dict[int, list[int]], ticks: int) -> dict:
    """逐 tick 驱动真实 after_step，返回 G3 读数与已知真值。"""
    eng = _FakeEng()
    inst = _M0Instruments()
    inst.attach(eng)
    for t in range(1, ticks + 1):
        sig = eng.signals
        hit = sig._marks > 0
        sig._age[hit] = np.maximum(sig._age[hit] - 1, 0)     # 本步未重写 ⇒ age 递减
        for vid, ts in sched.items():
            if t in ts:
                sig.write_many([CELL[vid]], [5])       # 唯一写入点 ⇒ G6 挂钩数得到（B1）
        inst.after_step(eng, t, sampled=False)
    h = inst.hist_run
    return {
        "truth": sum(len(v) for v in sched.values()),          # 真发射人·次
        "sum_hist": int(h[1:9].sum()),                         # “合计”口径
        "weighted": int(sum(k * h[k] for k in range(1, 8)) + h[8] * 8),
        "open_streaks": len(inst.streak),                      # 永不入账的串数
        "ambig": int(inst.ambig_n),
        "fresh_cells": int(inst.fresh_n),
    }

R12 = list(range(1, 13))
SCENARIOS = [
    ("S1 全体连发 t=1..12（最高密度态）", {i: R12 for i in IDS}, 12),
    ("S2 各发一次后不再发", {i: [i - 99] for i in IDS}, 12),
    ("S3 各发 1,2 → 停 → 9,10", {i: [1, 2, 9, 10] for i in IDS}, 12),
    ("S4 各发 1 → 停 → 9（单点串）", {i: [1, 9] for i in IDS}, 12),
]


def main() -> int:
    ap = argparse.ArgumentParser(description="GAP-A-EMIT 零机时核对")
    ap.add_argument("--selftest", action="store_true", help="断言比值不稳定（回归自锁）")
    args = ap.parse_args()

    rows = []
    for name, sched, ticks in SCENARIOS:
        m = measure(sched, ticks)
        ratio = m["sum_hist"] / m["truth"] if m["truth"] else float("nan")
        rows.append((name, m, ratio))

    print("GAP-A-EMIT：`sig_run_len_hist_1..8` 合计 vs 真发射人·次（真实仪表代码驱动）\n")
    print(f"{'场景':34} {'真发射':>6} {'Σhist':>6} {'Σk·hist':>8} {'Σhist/真':>9} {'未入账串':>8}")
    print("-" * 78)
    for name, m, ratio in rows:
        print(f"{name:34} {m['truth']:>6} {m['sum_hist']:>6} {m['weighted']:>8} "
              f"{ratio:>8.1%} {m['open_streaks']:>8}")

    ratios = [r for _, _, r in rows]
    print("\n结论：**不等价**。① 单位错（串数 ≠ 人·次，缺长度加权）；"
          "② 长度 >8 落溢出桶不可逆；③ 串只在'同体隔 tick 后再发'时入桶 ⇒ 连发/末发者归零；"
          "④ 同格多写者只记一 vid（`ambig_n`）；⑤ 累计直方图无 tick/种群分母 ⇒ 出不了'率'。")
    print(f"实测比值跨度 {min(ratios):.0%} → {max(ratios):.0%}（同装置族、同 ticks）"
          "⇒ 连'固定系数换算'都不成立。")
    print("⇒ 需求单：`_archive/2026-10-10-退役团队-归档/docs-旧档/tasks/需求单-GAP-A-EMIT-发射率专列-20261004.md`"
          "（真值抓手 = `signals.write_many` 的类级挂钩旁路计数，探针侧零引擎改动可出 G6 "
          "发射率列；🔴 不是引擎 `_emit_count`——需求单 §二 实测它在 M1 装置下恒为 0）。")

    if args.selftest:
        assert min(ratios) == 0.0, f"S1/S2 应归零，实得 {ratios}"
        assert max(ratios) > min(ratios), "比值应随场景漂移"
        assert rows[0][1]["weighted"] < rows[0][1]["truth"], "S1 长度加权仍应低估（溢出+未收尾）"
        print("\n[selftest] PASS —— 归零与漂移两性质均锁住")
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
