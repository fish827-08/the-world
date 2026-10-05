#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""install_git_hook.py —— 署名闸门钩子安装器（R393②，卡 SIGN-HOOK）

一句话：**装一次钩子，署错名就提交不出去**（不再靠人记得手敲 `-c`）。

装的是 `pre-commit`（`git rev-parse --git-path hooks` 的落点 = **全仓共享**，
一个仓库的所有工作树都受管）。钩子本体只做一件事：调 `sign_hook.py check`，
rc=2 ⇒ 拒绝提交并打印修法。

三个命令：
  install [--root R] [--force]   写钩子（幂等；已装则只报"无需改动"）
  status  [--root R]             查装没装、谁装的、指向哪个 sign_hook.py
  remove  [--root R]             卸载（🔴 只删自己带 marker 的文件；他人的先备份再让路）

🔴 纪律：
- 钩子文件里**没有** marker ⇒ 视为他人资产 ⇒ 拒绝覆盖（`--force` 才改名备份后重装，不删）；
- `install/remove` 改的是 `.git/hooks/`（**不进版本库**，各机各仓自装）⇒
  交付物是"工具 + 板上说明"，不是"替别人改了他的仓库"。

退出码：0 成功 / 1 用法或环境错误 / 2 已存在他人钩子（需 --force）。
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

TZ8 = timezone(timedelta(hours=8))
_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ROOT = os.path.dirname(os.path.dirname(_HERE))

HOOK_NAME = "pre-commit"
MARKER = "the-world-sign-gate"
MARKER_VER = 2   # v2 = 校验器/解释器三处回落 + 找不到时警告放行（v1 只记绝对路径）
CHECKER = os.path.join(_HERE, "sign_hook.py")


class HookError(Exception):
    def __init__(self, msg: str, code: int = 1):
        super().__init__(msg)
        self.code = code


def _git(root: str, args: list) -> tuple:
    try:
        p = subprocess.run(["git"] + list(args), cwd=root, capture_output=True,
                           text=True, timeout=60, encoding="utf-8", errors="replace")
        return p.returncode, (p.stdout or "").strip(), (p.stderr or "").strip()
    except Exception as e:
        return 127, "", f"{type(e).__name__}: {e}"


def hooks_dir(root: str) -> str:
    rc, out, err = _git(root, ["rev-parse", "--git-path", "hooks"])
    if rc != 0 or not out:
        raise HookError(f"读不到 hooks 目录（git rev-parse --git-path hooks）：rc={rc} {err[:160]}")
    if not os.path.isabs(out):
        out = os.path.join(root, out)
    return os.path.normpath(out)


def hook_path(root: str) -> str:
    return os.path.join(hooks_dir(root), HOOK_NAME)


def to_posix(p: str) -> str:
    return p.replace("\\", "/")


def render_hook(py: str, checker: str, installer: str) -> str:
    """生成 sh 钩子。

    两处必须能容错，否则"闸门"反而变成全队锁死：
    1. **解释器**：装时记的绝对路径优先，探不到再回落 PATH 里的 python/python3/py；
    2. **校验器**：本树 → 装时记录的路径 → 主工作树（分支删掉/工作树收编后仍找得到）。
       🔴 三处都找不到 ⇒ **警告放行**（rc=0）：工具失踪不该拦住提交，但把"闸门不在场"
       打进 stderr 让人看得见（fail-loud ≠ fail-block，二者区别写在机制文档里）。
    """
    return f"""#!/bin/sh
# {MARKER} v{MARKER_VER} —— 提交署名闸门（R393②，卡 SIGN-HOOK）
# 装于 {datetime.now(TZ8).strftime('%Y-%m-%d %H:%M')} 由 {os.path.basename(installer)}；卸载：python {to_posix(installer)} remove
# 规则：作者位 ∈ 花名册 且（在 .worktrees/<卡号> 里时）= 本卡负责人；主树/无卡 ⇒ 只警告不阻断
GATE=""
for C in "$(pwd -P)/collab-toolkit/tools/sign_hook.py" \\
         "{to_posix(checker)}" \\
         "$(git worktree list --porcelain 2>/dev/null | sed -n 's/^worktree //p; q')/collab-toolkit/tools/sign_hook.py"; do
    if [ -f "$C" ]; then GATE="$C"; break; fi
done
if [ -z "$GATE" ]; then
    echo "[署名闸门] 🔴 找不到 sign_hook.py（三处回落均失败）⇒ 本次警告放行；请跑 python collab-toolkit/tools/install_git_hook.py install 重装" >&2
    exit 0
fi
PY=""
for P in "{to_posix(py)}" python python3 py; do
    if command -v "$P" >/dev/null 2>&1; then PY="$P"; break; fi
    if [ -x "$P" ]; then PY="$P"; break; fi
done
if [ -z "$PY" ]; then
    echo "[署名闸门] 🔴 PATH 里没有 python ⇒ 本次警告放行（请重装时带可用解释器）" >&2
    exit 0
fi
# git 钩子的 cwd = 工作树根；用 pwd 而不是 git 输出路径（Windows 中文路径经 git 会变 GBK）
"$PY" "$GATE" check --root "$(pwd -P)"
rc=$?
if [ $rc -ne 0 ]; then
    echo "[署名闸门] rc=$rc —— 提交被拦（详见上方 🔴 行与修法）" >&2
fi
exit $rc
"""


def classify_existing(path: str) -> dict:
    """已存在的钩子是谁的：ours / foreign / absent。"""
    if not os.path.isfile(path):
        return {"kind": "absent", "text": ""}
    with open(path, encoding="utf-8", errors="replace") as f:
        text = f.read()
    mine = MARKER in text
    ver = None
    m = re.search(re.escape(MARKER) + r" v(\d+)", text)
    if m:
        ver = int(m.group(1))
    return {"kind": "ours" if mine else "foreign", "text": text, "ver": ver,
            "up_to_date": mine and ver == MARKER_VER}


def cmd_install(args) -> int:
    root = os.path.abspath(args.root)
    if not os.path.isfile(CHECKER):
        raise HookError(f"找不到校验器 {CHECKER}（本工具与 sign_hook.py 必须同目录）")
    path = hook_path(root)
    st = classify_existing(path)
    print(f"钩子落点: {path}（.git/hooks = 全仓所有工作树共享）")
    if st["kind"] == "absent":
        pass
    elif st["kind"] == "ours":
        if st["up_to_date"] and not args.force:
            print("[已安装] 版本与指向均一致 ⇒ 无需改动（幂等）")
            return 0
        print(f"[重装] 现 v{st['ver']} → v{MARKER_VER}（marker 自查为本工具产物）")
    else:
        print(f"[占用] 该位置是**别人的**钩子（无 {MARKER} marker），前 3 行：")
        for line in st["text"].splitlines()[:3]:
            print("   | " + line)
        if not args.force:
            sys.stderr.write("[拒绝] 不覆盖他人钩子 —— 加 --force 则先备份为 "
                             f"{HOOK_NAME}.foreign-<时间戳> 再装（🔴 不删）\n")
            return 2
        bak = path + ".foreign-" + datetime.now(TZ8).strftime("%Y%m%d-%H%M%S")
        os.replace(path, bak)
        print(f"  已备份原件 → {bak}")
    body = render_hook(sys.executable, CHECKER, os.path.join(_HERE, "install_git_hook.py"))
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(body)
    try:
        os.chmod(path, 0o755)
    except OSError:
        pass
    print(f"[已安装] {MARKER} v{MARKER_VER}")
    print("  钩子内校验器三处回落：本树 → 装时记录 → 主工作树；均无 ⇒ 警告放行")
    print(f"  装时记录: {CHECKER}")
    print(f"  记录的解释器: {sys.executable}")
    print("  立即自测: python " + to_posix(os.path.join(_HERE, "sign_hook.py")) + " check")
    return 0


def cmd_status(args) -> int:
    root = os.path.abspath(args.root)
    path = hook_path(root)
    st = classify_existing(path)
    print(f"钩子落点: {path}")
    if st["kind"] == "absent":
        print("[未安装] 署名闸门不在场 ⇒ 提交仍可署错名。装：python install_git_hook.py install")
        return 1
    print(f"[{('本工具装的' if st['kind'] == 'ours' else '他人装的')}]"
          f" marker={MARKER if st['kind'] == 'ours' else '无'} ver={st['ver']}")
    if st["kind"] == "ours":
        print("  装时记录的校验器: " + CHECKER)
        print("  一致: " + ("是" if st["up_to_date"] else
                          f"否（本工具版本 v{MARKER_VER}，跑 install 升级）"))
    return 0


def cmd_remove(args) -> int:
    root = os.path.abspath(args.root)
    path = hook_path(root)
    st = classify_existing(path)
    if st["kind"] == "absent":
        print(f"[跳过] 没有钩子: {path}")
        return 0
    if st["kind"] != "ours":
        sys.stderr.write(f"[拒绝] 该钩子不带 {MARKER} marker，不是本工具装的 ⇒ 不动\n")
        return 2
    bak = path + ".removed-" + datetime.now(TZ8).strftime("%Y%m%d-%H%M%S")
    os.replace(path, bak)
    print(f"[已卸载] 原文件保留为 {bak}（F-R10 只移不删）")
    return 0


def main(argv=None) -> int:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--root", default=DEFAULT_ROOT,
                        help="仓库根（默认 = collab-toolkit 的上一级）")
    ap = argparse.ArgumentParser(prog="install_git_hook.py",
                                 description="署名闸门 pre-commit 钩子安装器（R393②）")
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("install", parents=[common], help="装 pre-commit 钩子（幂等）")
    p.add_argument("--force", action="store_true", help="他人钩子：备份后让路")
    p.set_defaults(fn=cmd_install)
    p = sub.add_parser("status", parents=[common], help="查装没装 / 谁的 / 版本")
    p.set_defaults(fn=cmd_status)
    p = sub.add_parser("remove", parents=[common], help="卸载（只删自己装的，先备份）")
    p.set_defaults(fn=cmd_remove)
    args = ap.parse_args(argv)
    if not getattr(args, "fn", None):
        ap.print_help()
        return 1
    try:
        return args.fn(args)
    except HookError as e:
        sys.stderr.write(f"[错误] {e}\n")
        return e.code


if __name__ == "__main__":
    sys.exit(main())
