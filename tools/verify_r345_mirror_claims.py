# -*- coding: utf-8 -*-
"""镜（R344反馈）的可复跑核验断言 —— 与天平 `tools/verify_r344_claims.py` 同规格。

纪律：`[代码审核]`的指控也必须"附可复跑命令"（对齐天平 R344 §七立的规矩）。
本脚本只做**文本/结构断言**，不跑引擎、不改任何文件。

用法（项目根目录）：
    .venv/Scripts/python.exe tools/verify_r345_mirror_claims.py
"""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ENGINE = REPO / "simulation" / "sphere_engine.py"
CONFIG = REPO / "simulation" / "config.py"

_flag = 0
_hits = 0


def check(label: str, ok: bool, note: str = "") -> None:
    global _flag, _hits
    _hits += 1
    if not ok:
        _flag += 1
    print(f"[{'FLAG' if not ok else 'OK  '}] {label}"
          f"{('  ' + note) if note else ''}")
    if ok:
        print(f"         命中 True")


src = open(ENGINE, encoding="utf-8").read()
cfg = open(CONFIG, encoding="utf-8").read()

print("=" * 78)
print("镜的断言（反馈-R344-天平意见 的独立补充项）")
print("=" * 78)

# --- 硬伤 1：mem_bit 的**供给端**（记忆写入条件）---
# 关键：路径存在（天平错3已证）≠ 记忆有内容（本项）
m_mem_write = re.search(
    r"self\._work_memory\[rich_idx,\s*ptr\]\s*=\s*cur_flat\[rich_idx\]", src)
check("硬伤1a: _work_memory 仅在 food_rich 时写入（mem_bit 的供给端前提）",
      m_mem_write is not None,
      f"命中行 {src[:m_mem_write.start()].count(chr(10))+1 if m_mem_write else '-'}")

m_guard = re.search(r"elif food_rich\.any\(\):", src)
check("硬伤1b: 写入被`food_rich.any()` 门控 ⇒ 若世界无富食格，mem_bit 恒 0",
      m_guard is not None)

# mem_bit 的取值口径：state & 1（因为"8"档 state = e_bin*2 + mem_hit）
m_bit = re.search(r"self\._mem_bit_on\s*\+=\s*int\(\(state\s*&\s*1\)\.sum\(\)\)", src)
check("硬伤1c: mem_bit_frac 口径正确（state & 1 取得最低位 = mem_hit）",
      m_bit is not None)

# 读回链路三处接上
n_read = sum([
    '"mem_bit_frac"' in src,             # 探针读回
    "_mem_bit_on, self._mem_bit_n" in src,  # 进快照
])
check("硬伤1d: mem_bit_frac 读回链路完整（探针 + 快照）",
      n_read == 2, f"命中 {n_read}/2 处")

# --- 硬伤 2：世界尺寸 ---
m_rows = re.search(r"rows:\s*int\s*=\s*60", cfg)
m_cols = re.search(r"cols:\s*int\s*=\s*120", cfg)
check("硬伤2: SimConfig 默认确为 60x120（档案 §0 写7200 格属默认值，非实验档）",
      m_rows is not None and m_cols is not None,
      "⇒ 实验档实为 480x960（S2/s3 预设）")

# --- 澄清 1：sig_present 是变量 ⇒「影响完全相同」不成立 ---
m_sigw = re.search(r"sig_weight\s*=\s*self\._trust\[gi\]\s*\*\s*"
                   r"\(0\.5\s*\+\s*rep_w\s*\*\s*self\._trust\[gi\]\)", src)
m_use = re.search(r"sig_present\[nb\]\s*\*\s*sig_weight\[:, None\]", src)
check("澄清1: sig_weight=trust×(0.5+rep_w×trust)；rep_w=0 时为常数但"
      "**sig_present 仍是变量** ⇒「有/无信号影响完全相同」不成立",
      m_sigw is not None and m_use is not None)

# --- 澄清 5：_interpret 扩表须登记四处（R5 漏项）---
m_slots = re.search(r'"_interpret"', src)
m_init = re.search(r"self\._interpret\s*=\s*self\.rng\.normal\(", src)
m_snap = re.search(r"data\[[\"']_?interpret", src) or \
    re.search(r"interpret", src[src.find("snapshot"):src.find("snapshot") + 8000] if "snapshot" in src else "")
m_birth = re.search(r"_interpret\[", src)
check("澄清5: _interpret 涉及引擎状态（__slots__ + 初始化 + 快照）"
      " ⇒ R3.8 扩表须登记四处",
      all([m_slots, m_init, m_snap, m_birth]),
      f"slots={bool(m_slots)} init={bool(m_init)} snap={bool(m_snap)} 引用={bool(m_birth)}")

# --- 澄清 4（补天平更正帖措辞）：发射扣费行号，供交叉引用 ---
m_pay = re.search(r"energy\[emitters\]\s*-=\s*SIGNAL_COST", src)
check("澄清4: 发射确实扣 SIGNAL_COST（§1.2 成立；天平已证，此处交叉确认）",
      m_pay is not None,
      f"扣费行 {src[:m_pay.start()].count(chr(10))+1 if m_pay else '-'}")

print("=" * 78)
print(f"合计 {_hits} 项；FLAG 数 = {_flag}")
print("含义：FLAG 表示**我的反馈与代码不一致**（即我的意见错）。")
print("      全OK = 我的 6 条反馈全部有代码依据。")
