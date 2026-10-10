#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把「批次索引（R 编号 → 帖）」写入 `_share/路线共识.md`（幂等：已有则替换）。

用法
----
    python.exe tools/write_rindex.py [--archive 讨论板-20260926.md]

`--archive` 省略时**自动取 `_share/archive/` 下最新的归档件**（按文件名排序），
避免每轮归档都要手改脚本。

🔴 为什么要入库到 `tools/`：第八轮（2026-09-25）生成索引的脚本躺在 `_trash_local/`
（未跟踪、不共享），**且把归档路径写死** ⇒ 第九轮归档时差点漏更新索引
（教训库 ⑯「手工清单不是验收物」的同类：路径/清单写死 + 不在版本控制里 ⇒ 必然腐坏）。
"""
import re
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
CONSENSUS = ROOT / "_share" / "路线共识.md"
BOARDF = ROOT / "_share" / "讨论板.md"
ARCHDIR = ROOT / "_share" / "archive"
MARK = "## 批次索引"


def latest_archive() -> Path:
    cands = sorted(ARCHDIR.glob("讨论板-*.md"))
    if not cands:
        raise SystemExit(f"❌ {ARCHDIR} 下无归档件 ⇒ abort")
    return cands[-1]


def collect(srcs: list[tuple[str, Path]]) -> list[str]:
    rows = []
    for _tag, path in srcs:
        t = path.read_text(encoding="utf-8")
        ms = list(re.finditer(r"^### \[([^\]]+)\] · ([^\n（]+)", t, re.M))
        for i, m in enumerate(ms):
            end = ms[i + 1].start() if i + 1 < len(ms) else len(t)
            seg = t[m.start():end]
            # 🔴 R 号已过 200 ⇒ 不能用 R1[5-9][0-9]（会把 R200+ 全判成 "—"）
            rn = re.search(r"\b(R[12]\d\d)\b", seg[:700])
            r = rn.group(1) if rn else "—"
            title = ""
            for line in seg.splitlines()[1:8]:
                s = line.strip().lstrip("#").lstrip("*").strip()
                if not s or s.startswith("**内容**") or s == "内容：":
                    continue
                if s.startswith("主题"):
                    s = s.split("：", 1)[-1]
                if len(s) > 25:
                    title = s.replace("|", "／")[:92].strip("*").strip()
                    break
            role = m.group(1).replace("·天平", "").replace("·", "").strip()
            dt = m.group(2).strip()[:16]
            rows.append(f"| **{r}** | {dt} | {role} | {title} |")
    return rows


def main() -> int:
    arch = latest_archive()
    if len(sys.argv) > 2 and sys.argv[1] == "--archive":
        arch = ARCHDIR / sys.argv[2]
        if not arch.is_file():
            print(f"❌ 指定归档件不存在：{arch}", file=sys.stderr)
            return 2
    srcs = [("archive", arch), ("board", BOARDF)]
    print(f"来源：{arch.name} + {BOARDF.name}")

    header = f"""{MARK}（讨论板归档 R 编号索引；维护：`[所有者]`）

> **用途**：归档后仍能**按 R 编号定位**（`AGENT.md §3.3.6` 归档前检查单 ②的兜底）。
> **来源**：`_share/archive/{arch.name}`（**最近一轮归档之前**）＋ `_share/讨论板.md`（**之后**）。
> **检索**：`python collab-toolkit/tools/board_check.py find --id R190 --include-archive`
> **生成**：`python tools/write_rindex.py`（幂等替换；**不要手改此表**）
> ⚠️ 本表由脚本从两处板文件**机械抽取**：摘要 = 该帖首行；`R` 为 `—` 表示该帖**未标 R 号**；
> 表中**不含** 09-22 之前已归档的 R1–R151（见 `_share/archive/` 早前批次与本文件 §一 主表）。

| R | 日期 | 角色 | 摘要 |
|---|---|---|---|
"""
    rows = collect(srcs)
    block = header + "\n".join(rows) + "\n\n---\n"

    txt = CONSENSUS.read_text(encoding="utf-8")
    if MARK in txt:
        i = txt.index(MARK)
        j = txt.find("\n---\n", i)
        if j < 0:
            print("❌ 找不到块尾，abort", file=sys.stderr)
            return 2
        txt = txt[:i] + block + txt[j + len("\n---\n"):]
        print("[替换] 已更新批次索引")
    else:
        txt = txt.rstrip() + "\n\n---\n\n" + block
        print("[追加] 已写入批次索引")

    CONSENSUS.write_text(txt, encoding="utf-8")
    print(f"✅ {CONSENSUS.relative_to(ROOT)}（{len(txt.encode('utf-8')):,} B）｜索引 {len(rows)} 行")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
