#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ref_check.py — 仓库内路径引用扫描与迁移对账（FILE-REORG 零断链核验，提案 §五）

三种用法：
  snapshot  --out before.json            迁移前基线：记录每个 src 文件引用的路径及当时可解析性
  diff      --base before.json [--from C] 迁移后对账：旧基线里"可解析"的引用现在是否仍成立
  check     [--strict]                    现状体检：当前不可解析的引用清单（活跃区/历史区分列）

判定口径（diff）：
  基线中 resolved=True 的每条引用 (src, ref)：
    A. 现在仍可解析              ⇒ OK（原地成立）
    B. 目标被 git 识别为重命名（old→new）：
       - src 属历史区（_share/、_archive/ 下）⇒ 历史留痕（有意不回头改，不计断链）
       - src 现内容已含 new 路径              ⇒ 已更新（活跃文档随迁移刷新）
       - 否则                                 ⇒ 待更新（🔴 计入断链）
    C. 目标消失且无重命名映射                 ⇒ 🔴 断链（计入断链）
  断链=0 ⇒ rc 0；否则 rc 1。--from 指定 rename 探测基提交（默认 HEAD）。

历史区约定出处：FILE-REORG 提案 §四"只改活文档与索引，板上历史引用不回头改"。
零删除纪律（F-R10）：本工具只读，唯一写入是 --out/--json-out 报告文件。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ROOT = os.path.dirname(HERE)

PREFIXES = (
    "docs", "_advice", "_eval_reports", "_roadmap", "_share", "_archive",
    "tools", "experiments", "simulation", "tests", "collab-toolkit",
    "sim_core", "scripts", "results", "observatory",
)
EXTS = ("md", "py", "json", "csv", "txt", "png", "so", "yaml", "yml", "toml", "ipynb")
REF_RE = re.compile(
    r"(?<![\w/])(" + "|".join(re.escape(p) for p in PREFIXES) + r")/"
    r"[\w\-./À-鿿（）·，、()\[\] ]*?\.(?:" + "|".join(EXTS) + r")"
    r"(?![\w])"
)
SCAN_EXTS = {".md", ".py"}
SKIP_DIRS = {".git", ".worktrees", ".venv", "__pycache__", "_output",
             "_trash_local", "node_modules", "_pt", ".locks", ".cargo",
             "target", "build", "dist"}
HIST_PREFIXES = ("_share/", "_archive/")
# 明显是示例/通配/占位的引用串，不算真引用
JUNK_HINTS = ("YYYY", "MM-DD", "<", "…", "*", "?", "example", "xxx")


def norm(p: str) -> str:
    return p.replace("\\", "/").lstrip("./")


def is_probable_ref(ref: str) -> bool:
    if any(h in ref for h in JUNK_HINTS):
        return False
    # 尾部标点收尾的截断（中文正文里 "docs/x.md，" 已被正则排除逗号，双保险）
    return ref[-1] not in ("，", "。", "、", "；") and "  " not in ref


def iter_scan_files(root: str):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in sorted(dirnames)
                       if d not in SKIP_DIRS and not d.startswith(".qoder")]
        for fn in filenames:
            ext = os.path.splitext(fn)[1]
            if ext in SCAN_EXTS:
                fpath = os.path.join(dirpath, fn)
                rp = os.path.relpath(fpath, root).replace("\\", "/")
                if os.path.getsize(fpath) > 3_000_000:
                    continue
                yield rp, fpath


def extract_refs(fpath: str):
    """返回 [(lineno, ref)]；读失败静默（二进制/编码怪件不是引用源）。"""
    try:
        with open(fpath, encoding="utf-8", errors="strict") as f:
            lines = f.read().splitlines()
    except (OSError, UnicodeDecodeError):
        try:
            with open(fpath, encoding="utf-8", errors="replace") as f:
                lines = f.read().splitlines()
        except OSError:
            return []
    out = []
    for i, line in enumerate(lines, 1):
        for m in REF_RE.finditer(line):
            ref = norm(m.group(0))
            if is_probable_ref(ref):
                out.append((i, ref))
    return out


def resolve(root: str, ref: str) -> bool:
    p = os.path.join(root, ref.replace("/", os.sep))
    return os.path.exists(p)


def in_hist(src: str) -> bool:
    return src.startswith(HIST_PREFIXES)


def scan(root: str) -> list:
    entries = []
    for rp, fpath in iter_scan_files(root):
        for lineno, ref in extract_refs(fpath):
            entries.append({
                "src": rp, "ref": ref, "line": lineno,
                "resolved": resolve(root, ref),
            })
    return entries


def git_rename_map(root: str, base: str) -> dict:
    """base..工作树 的重命名映射 {old_path: new_path}（-M 相似度默认）。"""
    r = subprocess.run(["git", "-C", root, "-c", "core.quotepath=off",
                        "diff", "--name-status", "-M", base, "--"],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    if r.returncode != 0:
        raise SystemExit(f"[错误] git diff {base} 失败: {r.stderr.strip()[:200]}")
    m = {}
    for line in r.stdout.splitlines():
        parts = line.split("\t")
        if parts and parts[0].startswith("R") and len(parts) >= 3:
            m[norm(parts[1].strip('"'))] = norm(parts[2].strip('"'))
    return m


def cmd_snapshot(a) -> int:
    root = os.path.abspath(a.root)
    entries = scan(root)
    data = {"root": os.path.basename(root), "count": len(entries),
            "resolved": sum(1 for e in entries if e["resolved"]),
            "entries": entries}
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    print(f"[snapshot] 引用 {len(entries)} 条（可解析 {data['resolved']}）→ {a.out}")
    return 0


def cmd_diff(a) -> int:
    root = os.path.abspath(a.root)
    with open(a.base, encoding="utf-8") as f:
        base = json.load(f)
    rmap = git_rename_map(root, a.from_commit or "HEAD")
    src_cache = {}

    def src_has_new(src, new_path):
        if src not in src_cache:
            try:
                with open(os.path.join(root, src.replace("/", os.sep)),
                          encoding="utf-8", errors="replace") as f:
                    src_cache[src] = f.read()
            except OSError:
                src_cache[src] = ""
        return new_path in src_cache[src]

    broken, stale_pending, hist_kept, updated, ok, gone_src = [], [], [], [], 0, []
    for e in base["entries"]:
        if not e["resolved"]:
            continue  # 基线里本来就不可解析的（他仓数据件等）不追责
        ref = e["ref"]
        # 源文件自身可能随迁移换位（重整的常态）：先按 rename 映射换算再判定
        src = rmap.get(e["src"], e["src"])
        if not os.path.exists(os.path.join(root, src.replace("/", os.sep))):
            gone_src.append((e["src"], ref))  # 源失踪且无 rename ⇒ 异常
            continue
        if resolve(root, ref):
            ok += 1
            continue
        new = rmap.get(ref)
        if new and resolve(root, new):
            if in_hist(src):
                hist_kept.append((src, ref, new))
            elif src_has_new(src, new):
                updated.append((src, ref, new))
            else:
                stale_pending.append((src, ref, new))
        elif new:
            stale_pending.append((src, ref, new + "（新位也不存在？）"))
        else:
            broken.append((src, ref))
    total_bad = len(broken) + len(stale_pending) + len(gone_src)
    print(f"[diff] 基线可解析引用：原地OK {ok}｜随迁移更新 {len(updated)}｜"
          f"历史区留痕 {len(hist_kept)}｜🔴 断链 {len(broken)}｜🔴 待更新 {len(stale_pending)}｜"
          f"🔴 源失踪 {len(gone_src)}")
    for lst, tag in ((broken, "断链"), (stale_pending, "待更新"), (gone_src, "源失踪")):
        for item in lst[:40]:
            print(f"  [{tag}] {item}")
        if len(lst) > 40:
            print(f"  [{tag}] …另 {len(lst) - 40} 条")
    if a.json_out:
        with open(a.json_out, "w", encoding="utf-8", newline="\n") as f:
            json.dump({"ok": ok, "updated": updated, "hist_kept": hist_kept,
                       "broken": broken, "stale_pending": stale_pending,
                       "src_gone": gone_src}, f, ensure_ascii=False, indent=1)
    print(f"[结论] 断链合计 {total_bad} ⇒ " + ("零断链 ✅" if total_bad == 0 else "禁止 push ❌"))
    return 0 if total_bad == 0 else 1


def cmd_check(a) -> int:
    root = os.path.abspath(a.root)
    entries = scan(root)
    bad_active = [e for e in entries if not e["resolved"] and not in_hist(e["src"])]
    bad_hist = [e for e in entries if not e["resolved"] and in_hist(e["src"])]
    print(f"[check] 引用 {len(entries)} 条｜活跃区不可解析 {len(bad_active)}｜"
          f"历史区不可解析 {len(bad_hist)}（历史留痕，不计断链）")
    seen = set()
    for e in bad_active:
        key = (e["src"], e["ref"])
        if key in seen:
            continue
        seen.add(key)
        print(f"  [未解析] {e['src']}:{e['line']} → {e['ref']}")
    if a.strict and bad_active:
        return 1
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="ref_check.py",
                                 description="路径引用快照/迁移对账（FILE-REORG §五）")
    ap.add_argument("--root", default=DEFAULT_ROOT)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s1 = sub.add_parser("snapshot"); s1.add_argument("--out", required=True)
    s2 = sub.add_parser("diff")
    s2.add_argument("--base", required=True)
    s2.add_argument("--from", dest="from_commit", default=None,
                    help="rename 探测基提交（默认 HEAD）")
    s2.add_argument("--json-out", default=None)
    s3 = sub.add_parser("check"); s3.add_argument("--strict", action="store_true")
    a = ap.parse_args(argv)
    return {"snapshot": cmd_snapshot, "diff": cmd_diff, "check": cmd_check}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
