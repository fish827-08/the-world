#!/usr/bin/env python
"""R344 反馈的**机器可核证据** —— 逐条核《设计总档案 v1.0》的关键事实断言。

用途：让天平反馈里的每条"[实测]"都可复跑复核（避免"我说是就是"）。
用法：
    .venv/Scripts/python.exe tools/verify_r344_claims.py

退出码：0 = 全部断言与总档案一致；1 = 有不一致（需更正档案）。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "simulation" / "sphere_engine.py"
CONFIG = ROOT / "simulation" / "config.py"
SIGNAL = ROOT / "world" / "signal_field.py"

results: list[tuple[str, bool, str]] = []


def read(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def check(name: str, ok: bool, detail: str) -> None:
    results.append((name, ok, detail))


def main() -> int:
    eng = read(ENGINE)
    cfg = read(CONFIG)
    sig = read(SIGNAL)

    # ---- 错 1：donation 属OracleConfig（C5 oracle），非信号系统账目
    oracle_cls = next((i for i, l in enumerate(cfg) if l.startswith("class OracleConfig")), None)
    donation_line = next((i for i, l in enumerate(cfg) if re.search(r"^\s*donation:\s*float", l)), None)
    check(
        "错1a: donation 字段归属 OracleConfig",
        donation_line is not None and oracle_cls is not None and donation_line > oracle_cls,
        f"OracleConfig@{oracle_cls+1}, donation@{donation_line+1}",
    )
    # assert donation >= SIGNAL_COST 是否在 `if self.enabled:` 之内
    assert_line = next((i for i, l in enumerate(cfg) if "assert self.donation >= SIGNAL_COST" in l), None)
    gate = None
    if assert_line is not None:
        for j in range(assert_line, max(0, assert_line - 30), -1):
            if "if self.enabled:" in cfg[j]:
                gate = j
                break
    check(
        "错1b: 该 assert 受 OracleConfig.enabled 门控（非信号运行时）",
        assert_line is not None and gate is not None,
        f"assert@{assert_line+1},门控行@{gate+1 if gate else None}",
    )

    # ---- 错 2：_interpret 内容项已进 score（未被 rep_w 门控）
    has_interp_in_score = any(
        "score + (0.4 * perc)[:, None] * interp" in l or "0.4 * perc" in l and "interp" in l
        for l in eng
    )
    reads_marks = any("self.signals._marks[nb]" in l for l in eng)
    check("错2a: 决策确实读取 marks 内容", reads_marks, f"命中 {reads_marks}")
    check("错2b: _interpret 加成已进 score（B4'未通电'表述过强）", has_interp_in_score,
          f"命中 {has_interp_in_score}")

    # ---- 错 3：signal_alphabet "8" 档是否已实施
    has_8_branch = any('if self._alpha == "8"' in l for l in eng)
    doc_mentions_8 = any('"8"' in l and "mem_bit" in l for l in eng[:200])
    cfg_says_unimplemented = any('"8"(B③,未实施)' in l for l in cfg)
    check("错3a: 代码存在 \"8\" 档实际分支", has_8_branch, f"命中 {has_8_branch}")
    check("错3b: encode docstring 描述了 \"8\" 档", doc_mentions_8, f"命中 {doc_mentions_8}")
    check("错3c: config.py 注释仍写'未实施'（陈旧注释，需同步）", cfg_says_unimplemented,
          f"命中 {cfg_says_unimplemented}")

    # ---- 错 4：_interpret 当前索引维度（16项 = 单个 4-bit 模式）
    m = re.search(r"self\._interpret\s*=\s*self\.rng\.normal\([^)]*size=\(\s*\w+\s*,\s*(\d+)\s*\)", "\n".join(eng))
    idx = m.group(1) if m else "?"
    check("错4: _interpret 索引维度 = 16（单模式）⇒ M4 前需扩表", idx == "16", f"实际索引={idx}")

    # ---- 补 1（我一度怀疑、现已自查核实的点）：发射**确实**扣 SIGNAL_COST
    # 🔴 我最初只看了 write_many(:3664) 附近就断言"不扣费" ⇒ **我自己核错了**。
    #    实际扣费在 :3629-3633（含"付不起就不发"的守卫）⇒ §1.2 逻辑链第一环成立。
    can_afford = any("can_afford = energy[emitters] >= SIGNAL_COST" in l for l in eng)
    deduct_line = next(
        (i for i, l in enumerate(eng) if "energy[emitters] -= SIGNAL_COST" in l), None
    )
    check(
        "补1: 发射**确实扣** SIGNAL_COST（§1.2'发射纯付出'成立；我曾误判为不扣）",
        can_afford and deduct_line is not None,
        f"支付能力守卫={can_afford}；扣费行={deduct_line+1 if deduct_line else None}",
    )

    # ---- 异 3：R8.2 庇护所 vs"禁止新增生态机制"是否字面冲突（人工判断，此处仅提示）
    check("异3: R8.2(庇护所) 与 §5 禁止新增生态机制 字面冲突（需写成唯一例外）", True,
          "文档级冲突，非代码可核 ⇒ 由天平在反馈中提出")

    # ---- 输出
    print("=" * 92)
    print("R344 反馈证据核验 —— 《设计总档案 v1.0》关键事实断言")
    print("=" * 92)
    ok_all = True
    for name, ok, detail in results:
        flag = "OK  " if ok else "FLAG"
        print(f"[{flag}] {name}\n        {detail}")
        ok_all = ok_all and ok
    print("=" * 92)
    print(f"合计 {len(results)} 项；需更正档案的 FLAG 数 = {sum(1 for _, ok, _ in results if not ok)}")
    print("含义：FLAG 表示**总档案的表述与代码不一致**（不表示代码有 bug）。")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
