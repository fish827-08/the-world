#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""讨论板归档（通用化；第九轮起入库 `_trash_local` → `tools/`）。

用法
----
    python.exe tools/archive_board.py --cut "### [所有者·天平] · 2026-09-26 01:58" \
        --arch 讨论板-20260926.md --round 第九轮 \
        --pin-file <置顶文本> --head-file <快照+未落定项+归档说明>

做什么
------
1. 把板面**切点之前**的全部内容（含旧 PIN/旧快照/旧归档说明）**append-only** 移入
   `_share/archive/<arch>`（只搬不删，git 历史完整）；
2. 用 `--pin-file` + `--head-file` 重建板首（PIN 区 / 状态快照 / 未落定项 / 归档说明）；
3. 切点之后的帖**原样保留**在新板；
4. 自检：新板必须含 PIN-BEGIN / 状态快照 / 未落定项，且**署名帖数量 = 切点后的帖数**。

守卫（fail-loud，**不做静默跳过**）
----------------------------------
* 板面 < `--min-bytes`（默认 200 000）⇒ 拒绝（没到归档线，别折腾）；
* 切点锚点**必须唯一命中**；不命中或多次命中 ⇒ 拒绝；
* 归档目标文件**已存在** ⇒ 拒绝（防重复归档覆盖历史）。

依据：`AGENT.md` §3.3（板归档）+ §3.3.6（归档前检查单）。
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
BOARD = ROOT / "_share" / "讨论板.md"
ARCHDIR = ROOT / "_share" / "archive"


def main() -> int:
    ap = argparse.ArgumentParser(description="讨论板归档（append-only 移入 _share/archive/）")
    ap.add_argument("--cut", required=True, help="切点锚点（该帖的行首字符串，必须唯一命中）")
    ap.add_argument("--arch", required=True, help="归档文件名（置于 _share/archive/）")
    ap.add_argument("--round", default="本轮", help="轮次名（写进归档件头部说明）")
    ap.add_argument("--pin-file", required=True, help="新置顶区文本（含 PIN-BEGIN/END 注释标记）")
    ap.add_argument("--head-file", required=True, help="新板首（快照+未落定项+归档说明）")
    ap.add_argument("--min-bytes", type=int, default=200_000, help="板面低于此值拒绝归档")
    ap.add_argument("--archist", default="[归档人未署名]",
                    help="归档人署名（写进归档件头部；R371 不许顶名）")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    if not BOARD.is_file():
        print("❌ 找不到讨论板", file=sys.stderr)
        return 2
    raw = BOARD.read_bytes()
    # Windows 换行防御（R13 归档实录）：板面为 CRLF 时，读入归一为 LF、写回还原
    # CRLF，避免 text 模式二次转换产生 \r\r\n 损坏；git 存储恒为 LF，不受影响。
    crlf = b"\r\n" in raw
    txt = raw.decode("utf-8").replace("\r\n", "\n")
    if len(raw) < a.min_bytes:
        print(f"❌ 守卫拒绝：板面 {len(raw):,} B < 阈值 {a.min_bytes:,} B ⇒ 未到归档线，abort",
              file=sys.stderr)
        return 2

    hits = [m.start() for m in re.finditer(re.escape(a.cut), txt)]
    if len(hits) != 1:
        print(f"❌ 守卫拒绝：切点锚点命中 {len(hits)} 次（必须 == 1）⇒ abort", file=sys.stderr)
        return 2
    i = txt.rfind("\n", 0, hits[0]) + 1
    before, after = txt[:i], txt[i:]

    arch_path = ARCHDIR / a.arch
    if arch_path.exists():
        print(f"❌ 守卫拒绝：归档件 {arch_path} 已存在（防覆盖历史）⇒ abort", file=sys.stderr)
        return 2

    pin = Path(a.pin_file).read_text(encoding="utf-8")
    head = Path(a.head_file).read_text(encoding="utf-8")
    if "<!-- PIN-BEGIN -->" not in pin or "<!-- PIN-END -->" not in pin:
        print("❌ 守卫拒绝：--pin-file 缺 PIN-BEGIN/END 标记 ⇒ abort", file=sys.stderr)
        return 2

    n_posts_after = len(re.findall(r"^### \[[^\]]+\]", after, re.M))
    print(f"板面 {len(raw):,} B｜切点 @{i:,}｜归档 {len(before):,} 字符｜"
          f"新板保留 {len(after):,} 字符（{n_posts_after} 帖）")

    arch_header = (f"# 讨论板归档（{a.round}）—— {a.arch}\n\n"
                   f"> 归档人：`{a.archist}` ｜ 触发：板面 **{len(raw)/1024:.1f} KB** > 阈值 200 KB\n"
                   f"> 归档范围：板首 ~ **切点 `{a.cut}` 之前**（**append-only，不删内容**）\n"
                   f"> 检索：`python collab-toolkit/tools/board_check.py find --kw \"…\" --include-archive`\n"
                   f"> 工具：`tools/archive_board.py`（幂等守卫：切点必须唯一命中 + 归档件不得已存在）\n\n"
                   f"---\n\n")
    if a.dry_run:
        print(f"[dry-run] 将写 {arch_path}（{(arch_header + before).__len__():,} 字符）")
        print(f"[dry-run] 新板 {len((pin + head + after).encode('utf-8')):,} B")
        return 0

    crlf = b"\r\n" in raw
    def dump(path: Path, s: str) -> None:
        data = (s.replace("\r\n", "\n").replace("\n", "\r\n") if crlf
                else s).encode("utf-8")
        path.write_bytes(data)

    dump(arch_path, arch_header + before)
    dump(BOARD, pin + head + after)

    nb = BOARD.read_text(encoding="utf-8")
    posts = len(re.findall(r"^### \[[^\]]+\]", nb, re.M))
    ok = ("<!-- PIN-BEGIN -->" in nb and "## 一、状态快照" in nb
          and "## 二、未落定项" in nb and posts == n_posts_after)
    print(f"✅ 归档件 {arch_path.relative_to(ROOT)}（{arch_path.stat().st_size:,} B）")
    print(f"✅ 新板 {BOARD.relative_to(ROOT)}（{BOARD.stat().st_size:,} B）")
    print(f"自检：署名帖 {posts}/{n_posts_after}｜PIN {'有' if '<!-- PIN-BEGIN -->' in nb else '无'}｜"
          f"快照 {'有' if '## 一、状态快照' in nb else '无'}｜"
          f"未落定项 {'有' if '## 二、未落定项' in nb else '无'} ⇒ {'✅ 通过' if ok else '❌ 不通过'}")
    return 0 if ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
