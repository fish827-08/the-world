#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ref_check.py —— 仓内引用断链扫描与批前后对表（FILE-REORG §五，卡验收硬项）

一句话：**搬文件之前拍一张"引用全景照"，搬完之后对表——新增断链必须为空才准 push**。

三类动作：
  snapshot  --out before.json            扫描全部追踪 md/py/sh 中的仓内相对路径 token，记存在性
  check                                  同上但只打印断链（退出码：有断链=4）
  diff      --before before.json         重扫当前树，与快照对表 ⇒ 只报**新增**断链
            （历史引用不回头改：--exclude 命中的文件不进新增断链，见 FILE-REORG 提案 §四）

口径：
- token = 形如 `a/b.ext` 的相对路径串（含中文段）；解析先后两个基准：仓根、引用文件所在目录；
- 外部/他仓路径不计（http、盘符、`the-world-data/` 前缀、`~`）——它们断不断与本仓搬迁无关；
- 快照记录的是**全量存在性**；diff 语义 = after 断链集合 − before 断链集合（按 token 计，
  不按引用文件计——引用文件自己也可能被搬动）。

退出码：0 无新增断链 / 4 存在新增断链 / 1 用法或环境错误。
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

SCAN_SUFFIXES = (".md", ".py", ".sh")
EXTS = r"md|py|sh|csv|json|txt|rs|toml|yaml|yml|html|png|jpg|so|ipynb|ps1|bat"
SEG = r"[\w.\-一-鿿]"
# 中文段允许，段间以 / 分隔；至少一个 /（纯本目录文件名不算"仓内路径 token"，噪音太大）。
# 可选前缀：scheme:// 、~/ 、盘符:/ —— 带前缀者整体匹配后由 is_external 丢弃（防 URL 内子串误报）。
PREF = r"(?:[A-Za-z][A-Za-z0-9+.-]*://|~/|[A-Za-z]:[/\\])"
TOKEN_RE = re.compile(rf"{PREF}?{SEG}+(?:/{SEG}+)+\.(?:{EXTS})")
PREF_RE = re.compile(PREF)

DEFAULT_EXCLUDE = ("_share/讨论板.md", "_share/archive/", "_archive/")


def norm(p: str) -> str:
    return p.replace("\\", "/").rstrip("/")


def is_external(tok: str) -> bool:
    t = tok.lower()
    return bool(PREF_RE.match(t)) or t.startswith("the-world-data/") or t.startswith("/")


def extract_tokens(text: str) -> set:
    out = set()
    for m in TOKEN_RE.finditer(text):
        tok = m.group(0)
        if tok.count("/") < 1 or is_external(tok):
            continue
        # 去掉行首引用样式 `path:12` 的冒号部分由正则天然截断（冒号不在 SEG 里）
        out.add(tok)
    return out


def resolve(tok: str, root: str, src_dir: str) -> bool:
    """token 是否指向存在的路径（仓根或引用文件目录为基准，任一命中即可）。"""
    for base in (root, src_dir):
        if os.path.exists(os.path.normpath(os.path.join(base, tok))):
            return True
    # 允许 `docs/xxx.md:31` 之类由 TOKEN_RE 天然止步于冒号；允许锚点尾缀被截断的情况少
    return False


def git_tracked(root: str) -> list:
    p = subprocess.run(["git", "-c", "core.quotepath=off", "ls-files", "-z"],
                       cwd=root, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if p.returncode != 0:
        raise RuntimeError(f"git ls-files 失败 rc={p.returncode}: {p.stderr[:200]}")
    return [f for f in (p.stdout or "").split("\0") if f]


def scan(root: str, files: list | None = None) -> dict:
    """全量扫描：{"tokens": {token: [引用文件...]}, "broken": {token: [...]}}（存在=假即断链）。"""
    files = files if files is not None else git_tracked(root)
    tokens: dict = {}
    for f in files:
        if not f.endswith(SCAN_SUFFIXES):
            continue
        p = os.path.join(root, f)
        if not os.path.isfile(p):
            continue
        try:
            with open(p, encoding="utf-8", errors="replace") as fh:
                text = fh.read()
        except OSError:
            continue
        src_dir = os.path.dirname(os.path.join(root, f))
        for tok in extract_tokens(text):
            rec = tokens.setdefault(tok, {"ok": False, "refs": []})
            rec["refs"].append(f)
            if not rec["ok"] and resolve(tok, root, src_dir):
                rec["ok"] = True
    for v in tokens.values():
        v["refs"] = sorted(v["refs"])
    broken = {t: v["refs"] for t, v in tokens.items() if not v["ok"]}
    return {"total_tokens": len(tokens), "tokens": tokens, "broken": broken}


def excluded(src: str, patterns) -> bool:
    src = norm(src)
    for pat in patterns:
        if pat.endswith("/") and src.startswith(pat):
            return True
        if src == pat:
            return True
    return False


def new_broken(before_tokens: dict, after: dict, exclude) -> list:
    """after 中断链、但在 before 快照中**不是**断链（token 级）的条目 ⇒ 本次搬迁引入的断链。"""
    broken_before = set(before_tokens["broken"])
    out = []
    for tok, refs in sorted(after["broken"].items()):
        if tok in broken_before:
            continue
        live_refs = [r for r in refs if not excluded(r, exclude)]
        if live_refs:
            out.append({"token": tok, "refs": live_refs})
    return out


def cmd_snapshot(args) -> int:
    root = os.path.abspath(args.root)
    s = scan(root)
    payload = {"root": norm(root), "generated": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
               "broken_count": len(s["broken"]), "tokens": s["tokens"], "broken": s["broken"]}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    print(f"[ref_check] 快照 {args.out}：token {s['total_tokens']} 个，"
          f"存量断链 {len(s['broken'])} 个（基线，不算新账）")
    return 0


def cmd_check(args) -> int:
    root = os.path.abspath(args.root)
    s = scan(root)
    print(f"[ref_check] token {s['total_tokens']}，断链 {len(s['broken'])}")
    for tok, refs in sorted(s["broken"].items()):
        print(f"  🔴 {tok}  ← {', '.join(refs[:3])}{' ...' if len(refs) > 3 else ''}")
    return 4 if s["broken"] else 0


def cmd_diff(args) -> int:
    root = os.path.abspath(args.root)
    with open(args.before, encoding="utf-8") as f:
        before = json.load(f)
    exclude = list(DEFAULT_EXCLUDE) + list(args.exclude or [])
    after = scan(root)
    nb = new_broken(before, after, exclude)
    print(f"[ref_check] 对表 {args.before} → 当前树：新增断链 {len(nb)} 个"
          f"（排除历史引用面：{', '.join(exclude)}）")
    for e in nb:
        print(f"  🔴 {e['token']}  ← {', '.join(e['refs'][:3])}")
    if nb:
        print("[ref_check] ⇒ 断链未归零，不得 push 迁移批（FILE-REORG §五）")
        return 4
    print("[ref_check] ⇒ 零新增断链 ✅")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="ref_check.py", description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=".", help="仓库根（默认 cwd）")
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("snapshot"); p.add_argument("--out", required=True); p.set_defaults(fn=cmd_snapshot)
    p = sub.add_parser("check");    p.set_defaults(fn=cmd_check)
    p = sub.add_parser("diff")
    p.add_argument("--before", required=True)
    p.add_argument("--exclude", action="append",
                   help="引用文件命中即不计新增断链（目录以 / 结尾），可多次；默认 _share/讨论板.md,_share/archive/,_archive/")
    p.set_defaults(fn=cmd_diff)
    args = ap.parse_args(argv)
    if not getattr(args, "fn", None):
        ap.print_help(); return 1
    try:
        return args.fn(args)
    except Exception as e:
        sys.stderr.write(f"[ref_check·异常] {type(e).__name__}: {e}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
