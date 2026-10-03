#!/usr/bin/env python
"""文档一致性体检 —— 查《设计总档案》里"同一组门/同一概念在两处写法不一致"。

为什么需要（**实测已发生两次**）
--------------------------
R346：v1.2 停止令写"4 道门"但列了 5 个；小节标题仍写"v1.1 新增"。
R347：§5.1 用 `P1-a/P1-b/P1-c`，而 §9.1 用 `P1`/`P1b` ⇒ **同名不同义**
      （§5.1 的 `P1-b` = 私有信息门；§9.1 的 `P1b` = 距离门）。

🔴 **根因是"同一组事实写在两处"** ⇒ 与其每次人肉核对，不如让脚本查。

用法：
    .venv/Scripts/python.exe tools/check_doc_consistency.py
    .venv/Scripts/python.exe tools/check_doc_consistency.py --file docs/设计总档案/AI-设计总档案-v1.2.md

退出码：0 = 一致；1 = 发现不一致（需修档案）。
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "docs" / "设计总档案" / "AI-设计总档案-v1.2.md"

# 期望的门名标准写法（canonical）
CANON = ("P1-a", "P1-b", "P1-c", "P2", "P3")
# 已知的历史旧名 → 应改成的标准名
LEGACY = {
    "P1b": "P1-c",     # 旧：距离门 ⇒ 现：P1-c
    "P1-b 距离": "P1-c",
    "**P1 ": "**P1-b ",  # 旧：P1 私有信息门 ⇒ 现：P1-b
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=str(DEFAULT))
    args = ap.parse_args()
    fp = Path(args.file)
    if not fp.exists():
        print(f"[FAIL] 文件不存在：{fp}")
        return 1
    lines = fp.read_text(encoding="utf-8").splitlines()

    issues: list[tuple[str, str]] = []

    # ---- 检查 1：门名出现位置 + 是否用了旧名
    print("=" * 92)
    print("文档一致性体检 —— 门名 / 旧名 / 计数声明")
    print("=" * 92)
    print(f"文件：{fp.name}（{len(lines)} 行）\n")

    # 🔴 变更记录区里的旧名是**正确的历史陈述**（记录当时用的旧名）⇒ 跳过
    hist_start = next((i for i, l in enumerate(lines, 1) if l.startswith("## 变更记录")), len(lines) + 1)

    legacy_hits: dict[str, list[int]] = defaultdict(list)
    canon_hits: dict[str, list[int]] = defaultdict(list)
    hist_hits: list[int] = []
    for i, l in enumerate(lines, 1):
        for c in CANON:
            if re.search(rf"(?<![A-Za-z0-9-]){re.escape(c)}(?![A-Za-z0-9-])", l):
                canon_hits[c].append(i)
        if re.search(r"(?<![A-Za-z0-9-])P1b(?![A-Za-z0-9-])", l):
            if i >= hist_start:
                hist_hits.append(i)
            else:
                legacy_hits["P1b"].append(i)

    print("【检查 1】门名出现位置")
    for c in CANON:
        print(f"  {c:6s} 出现 {len(canon_hits[c]):3d} 处  行号 {canon_hits[c][:8]}")
    for k, v in legacy_hits.items():
        print(f"  🔴旧名 {k:6s} 出现 {len(v):3d} 处  行号 {v[:8]} ⇒ 应改为 {LEGACY.get(k, '?')}")
    if legacy_hits:
        issues.append(("门名旧写法（正文）", f"发现旧名 {list(legacy_hits)}"))
    if hist_hits:
        print(f"  ℹ️旧名在变更记录区 {len(hist_hits)} 处 ⇒ **正确历史陈述，不算问题**")

    # ---- 检查 2：§5.1 与 §9.1 两处表格的门名是否一致
    print("\n【检查 2】§5.1（详细定义）与 §9.1（汇总）门名一致性")
    # 找两张表：含 "P1-a" 的行 vs 含 "P1" 开头的行
    detail_line = next((i for i, l in enumerate(lines, 1) if l.startswith("| **P1-a")), None)
    summary_lines = [i for i, l in enumerate(lines, 1)
                     if re.match(r"^\|\s*\*{0,2}P1\b", l) and i != detail_line
                     and i < hist_start]
    if detail_line and summary_lines:
        print(f"  §5.1 详细定义首行：{detail_line}")
        print(f"  §9.1 汇总表候选行：{summary_lines}")
        if legacy_hits.get("P1b") or any(
            "**P1 " in lines[i - 1] or l.startswith("| **P1 私有信息门") for i in summary_lines
        ):
            print("  🔴 汇总表仍用旧名 ⇒ 同一组门两处写法不一致（同名不同义风险）")
            issues.append(("两处门名不一致", f"§9.1 行 {summary_lines}"))
    else:
        print(f"  （未定位到两张表：detail={detail_line}, summary={summary_lines}）")

    # ---- 检查 3：停止令"X 道门"与实际列出的门数是否一致
    print("\n【检查 3】停止令门数声明 vs 实际列出")
    for i, l in enumerate(lines, 1):
        m = re.search(r"\*\*M1 之前有\s*(\d+)\s*道门\*\*.*?（见[^）]*）", l)
        if m:
            claimed = int(m.group(1))
            listed = re.findall(r"P1-[abc]|P2|P3", l)
            uniq = sorted(set(listed))
            print(f"  行 {i}：声称 {claimed} 道｜实际列出 {len(uniq)} 个 → {uniq}")
            if claimed != len(uniq):
                print("  🔴 门数声明与列出不一致")
                issues.append(("门数声明", f"声称 {claimed} 实列 {len(uniq)}"))
            else:
                print("  ✅ 一致")

    # ---- 检查 4：已知的"1 格"表述是否还留在风险表里
    print("\n【检查 4】已作废前提（记忆格离发射者仅 1 格）是否仍出现在文档里")
    NEG = ("不是事实", "待测量", "已正名", "是错误", "被否定", "而非", "不成立", "P1-c 距离门待测")
    bad = []
    for i, l in enumerate(lines, 1):
        if not re.search(r"记忆格.{0,16}(仅|只)\s*1\s*格|绝大多数在 1 格内", l):
            continue
        # 🔴 否定语境排除：若同句已声明该前提不成立/待测量 ⇒ 不算残留（R347 误报过一次）
        if any(n in l for n in NEG):
            print(f"     （行 {i} 命中关键词但处于**否定语境**，已排除）")
            continue
        bad.append((i, l.strip()[:90]))
    if bad:
        for i, t in bad:
            print(f"  🔴 行 {i}：{t}")
        issues.append(("已作废前提残留", f"{len(bad)} 处"))
    else:
        print("  ✅ 无残留")

    # ---- 汇总
    print("\n" + "=" * 92)
    if issues:
        print(f"发现 {len(issues)} 类不一致：")
        for what, detail in issues:
            print(f"  🔴 {what}：{detail}")
        print("\n⇒ 这些是**文档层**问题（不改代码），需档案作者修订。")
        return 1
    print("✅ 未发现不一致。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
