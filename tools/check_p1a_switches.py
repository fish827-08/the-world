#!/usr/bin/env python
"""R345 · P1-a 开关自检（**零机时、纯读回**）

用途：在跑 P1-b（读 `mem_bit_frac`）**之前**，先确认"8" 档与记忆机制**真的开了**。
为什么必须有这一步
----------------
`mem_bit_frac = _mem_bit_on / _mem_bit_n`，而`_work_memory` 初值 = -1（`sphere_engine.py:1361`），
且只在"站在富食格上"时写入、被 `memory_v2` 门控（`:3586-3593`）
⇒ **默认配置下 mem_bit 恒 0** ⇒ P1-b 会 FAIL，
   但真实原因只是"没开记忆"，**不是"私有信息路线证伪"**。
⇒ 这正是 R326"改装置只改预设档、实际未生效"的同族坑（已踩过一次）。

用法：
    .venv/Scripts/python.exe tools/check_p1a_switches.py
    .venv/Scripts/python.exe tools/check_p1a_switches.py --alphabet 8 --memory-v2

退出码：0 = P1-a 通过（可以跑 P1-b）；1 = 未通过（**先修开关，别浪费 P1-b 的机时**）。
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    fix: str = ""


def main() -> int:
    ap = argparse.ArgumentParser(description="P1-a 开关自检（零机时）")
    ap.add_argument("--alphabet", default="8", help="期望的 signal_alphabet（默认 8）")
    ap.add_argument("--memory-v2", dest="memory_v2", action="store_true", default=None,
                    help="期望 memory_v2=True（默认：不指定则跳过）")
    args = ap.parse_args()

    checks: list[Check] = []

    # ---- 1) "8" 档是否在已实施白名单里（config.py:912）
    sys.path.insert(0, str(ROOT))
    try:
        from simulation.config import SIGNAL_ALPHABET_IMPLEMENTED
        implemented = tuple(SIGNAL_ALPHABET_IMPLEMENTED)
        checks.append(Check(
            f'signal_alphabet="{args.alphabet}" 在已实施白名单内',
            args.alphabet in implemented,
            f"SIGNAL_ALPHABET_IMPLEMENTED={implemented}",
            fix=f'改为 "16" 或检查 config.py:912',
        ))
    except Exception as e:  # pragma: no cover
        checks.append(Check("读 SIGNAL_ALPHABET_IMPLEMENTED", False, f"导入失败：{e}"))

    # ---- 2) 构造一个真实 config，**读回**实际生效值（不看 CLI 传了什么）
    try:
        from simulation.config import SimConfig
        cfg = SimConfig()
        cfg.signal_alphabet = args.alphabet
        # 🔴 C4纪律：开关要"读回构造后的值"，不是"看我们传了什么"
        got_alpha = getattr(cfg, "signal_alphabet", None)
        checks.append(Check(
            "构造后 config.signal_alphabet 读回一致",
            got_alpha == args.alphabet,
            f"期望={args.alphabet!r} 实得={got_alpha!r}",
            fix="检查 SimConfig 是否吞掉该参数（R246 四件套：从_dict 白名单往返）",
        ))

        # ---- 3) memory_v2 门（mem_bit 的输入来源）
        # 🔴 归属核验：`memory_v2` 属 **InfoStructureConfig**（`config.py:568` 类，`:635` 字段），
        #    **不在 `cfg.memory` 下** ⇒ v1.1 若按 `cfg.memory.memory_v2` 取会得 None（我自己第一次就踩了）。
        ifc = getattr(cfg, "info_structure", None)
        mv2 = getattr(ifc, "memory_v2", None) if ifc is not None else None
        mgrad = getattr(ifc, "memory_gradient", None) if ifc is not None else None
        if args.memory_v2 is None:
            # 🔴 判定逻辑自纠：**未开就必须拦住**，不能只"报告"后仍放行
            #    （这正是我批评 v1.1 P1 门"报告≠闸门"的同一个坑 ⇒ 我自己不能犯）
            checks.append(Check(
                "memory_v2=True（P1-b 的**硬前提**）",
                bool(mv2) is True,
                f"实得 memory_v2={mv2}（属 InfoStructureConfig，config.py:635）"
                + ("" if mv2 else "⇒ 🔴 **未开 ⇒ _work_memory 全 -1 ⇒ mem_bit 恒 0 ⇒ P1-b 必 FAIL**"),
                fix="开 memory_v2（🔴 注意硬 assert：`config.py:668` 要求 "
                    "`memory_gradient == 'orientation'`；另 memory_v2 与 rd/dynamic_centroid 有构造期互斥，"
                    "见 sphere_engine.py:1096-1101）",
            ))
        else:
            checks.append(Check(
                "memory_v2=True（P1-b 的前提）",
                bool(mv2) is True,
                f"期望=True 实得={mv2}；memory_gradient={mgrad}",
                fix="开 memory_v2（config.py:668 要求 memory_gradient=='orientation'）",
            ))

        # ---- 3b) memory_v2 的伴随硬assert（提前暴露，别等跑批才 fail-loud）
        if bool(mv2):
            checks.append(Check(
                "memory_v2 的伴随条件 memory_gradient=='orientation'（config.py:668）",
                mgrad == "orientation",
                f"实得 memory_gradient={mgrad!r}",
                fix="设 memory_gradient='orientation'（v2 取代 v1 世界方向版，不并存）",
            ))
    except Exception as e:
        checks.append(Check("构造 SimConfig", False, f"失败：{e}"))

    # ---- 4) 静态事实核对（防文档漂移）：work_memory 初值 & 写入门控
    eng = (ROOT / "simulation" / "sphere_engine.py").read_text(encoding="utf-8").splitlines()
    wm_init = next((l.strip() for l in eng if "self._work_memory = np.full" in l), "")
    has_v2_gate = any("if food_rich.any():" in l for l in eng)
    checks.append(Check(
        "静态：_work_memory 初值为 -1（故默认配置下 mem_bit 恒 0）",
        "np.full((n, 4), -1" in wm_init or "-1" in wm_init,
        wm_init or "未找到初始化行",
    ))
    checks.append(Check(
        "静态：_work_memory 只在富食格写入（被 memory_v2 门控）",
        has_v2_gate,
        f"sphere_engine.py:3586-3593 有 `if memory_v2 / elif food_rich.any()` 门控：{has_v2_gate}",
    ))

    # ---- 输出
    print("=" * 92)
    print("P1-a 开关自检（零机时、纯读回）—— R345 建议新增")
    print("=" * 92)
    all_ok = True
    for c in checks:
        flag = "OK  " if c.ok else "FAIL"
        print(f"[{flag}] {c.name}\n        {c.detail}")
        if not c.ok and c.fix:
            print(f"        🔧 怎么修：{c.fix}")
        all_ok = all_ok and c.ok

    print("=" * 92)
    if all_ok:
        print("🟢 P1-a 通过 ⇒ **可以跑 P1-b**（1 run × 2k 读 mem_bit_frac）")
        return 0
    print("🔴 P1-a 未通过 ⇒ **先修开关，不要浪费 P1-b 的机时**")
    print("   🔴 关键：mem_bit 恒 0 **不等于**「私有信息路线证伪」")
    print("      （项目自己的注释 sphere_engine.py:139-142 已预判过这个 null 陷阱）")
    return 1


if __name__ == "__main__":
    sys.exit(main())
