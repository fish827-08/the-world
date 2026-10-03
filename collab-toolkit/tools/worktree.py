#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""worktree.py —— 任务卡隔离工作树（R359 B1，蓝本 rag-kb/orchestra/worktree.py）

一句话：**把"两个人别碰同一份工作树"从纪律变成物理隔离**。

三个命令：
  setup <卡号>   建隔离目录 `.worktrees/<卡号>` + 检出 `task/<卡号>` 分支
  enter <卡号>   只打印该卡的隔离目录路径（给别的会话/脚本用）
  clean <卡号>   清理工作树（**提交保留在分支上**，分支不删）

🔴 the-world 四点适配（照搬会出事，见派工单 R359 §二）：

| # | 适配点 | 为什么 |
|---|--------|--------|
| 1 | 隔离目录 = **`.worktrees/`** | R342 已把 5 个 `_wt_*` 旧冻结树收编进 `_archive/` ⇒ 避免命名撞车 |
| 2 | 🔴 显式打印 **主仓 `.venv` 解释器和 `--python` 覆盖口** | worktree 内**没有** `.venv`、也没有 `sim_core.so` ⇒ worker 必须知道去哪跑 python（统一 `.venv/Scripts/python.exe`）|
| 3 | 🔴 **不实现 `prune`** | `AGENT.md:493` 全员禁用 `git worktree prune`（2026-09-12 事故）⇒ 只做 setup/enter/clean |
| 4 | `.worktrees/` 入 `.gitignore`（见主仓 `.gitignore` 协作线段）| 否则污染 `git status`（= R343 教训：游离文件淹没真改动）|

隔离证明：`setup` 之后，主工作区 `git checkout task/<卡号>` 会被 git 拒绝
（"already exists"），因为该分支已被 worktree 占用。

退出码：0 成功 / 1 使用错误 / 2 冲突或已存在（setup 幂等复用时仍为 0）。
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ROOT = os.path.dirname(os.path.dirname(_HERE))  # collab-toolkit/tools -> the-world/

# 卡号白名单：防命令注入（git 参数直接拼命令）
CARD_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]{0,63})$")

# 🔴 适配点 1：隔离目录固定 `.worktrees/`（不叫 `_wt_*`）
WORKDIR_NAME = ".worktrees"
BRANCH_PREFIX = "task/"

MSG_NO_PRUNE = (
    "本工具**不提供** prune —— `AGENT.md:493` 全员禁用 `git worktree prune`"
    "（2026-09-12 事故：误删他人冻结树）。隔离目录靠 `.gitignore` 兜底，"
    "长期不用的 clean 由人工在确认无分支引用后 `git branch -D`。"
)


class WorktreeError(Exception):
    def __init__(self, msg: str, code: int = 1):
        super().__init__(msg)
        self.code = code


# ---------------- 纯函数（可单测，不碰 git） ----------------
def plan_paths(root: str, card: str) -> dict:
    """卡号 → 目录 / 分支名的映射。"""
    return {"dir": os.path.join(root, WORKDIR_NAME, card),
            "branch": BRANCH_PREFIX + card}


def check_card(card: str) -> str:
    """卡号格式校验（白名单），非法直接抛错。"""
    if not CARD_RE.match(card or ""):
        raise WorktreeError(
            f"卡号不合法: {card!r}（只允许字母数字, 点, 下划线, 连字符；"
            f"必须以字母数字开头）—— 防命令注入")
    return card


def resolve_python(root: str, override: str | None = None) -> tuple:
    """解析 worktree 内应该用的解释器。

    返回 (exe, exists, why)。🔴 适配点 2：**一律指向主仓 .venv**，
    因为 worktree 里没有 `.venv` / `sim_core.so`。裸机场景（R249）不存在
    `.venv` ⇒ exists=False，仍如实返回并给出提示（**fail-loud，不静默回落到
    sys.executable** —— R249 的教训："宣称'检测不到自动走 X'前先验证'检测'成立"）。
    """
    if override:
        return override, os.path.isfile(override), "--python 指定"
    if os.name == "nt":
        rel = os.path.join(".venv", "Scripts", "python.exe")
    else:
        rel = os.path.join(".venv", "bin", "python")
    exe = os.path.join(root, rel)
    return exe, os.path.isfile(exe), "主仓 .venv（worktree 内无 .venv，见 R249）"


def plan_setup(root: str, card: str) -> dict:
    """setup 的**计划**（纯函数）：将要执行的 git 命令 + 各处存在性。

    `cmd=None` 表示无需建分支（复用已有 branch 或已有 worktree）。
    """
    paths = plan_paths(root, card)
    wt_exists = os.path.isdir(os.path.join(paths["dir"], ".git")) or \
        os.path.isfile(os.path.join(paths["dir"], ".git"))
    wt_dir_exists = os.path.isdir(paths["dir"])
    r = run_git_capture(root, ["rev-parse", "--verify", "refs/heads/" + paths["branch"]])
    branch_exists = r[0] == 0
    cmd = None
    if not wt_exists:
        if branch_exists:
            cmd = ["worktree", "add", paths["dir"], paths["branch"]]
        else:
            cmd = ["worktree", "add", "-b", paths["branch"], paths["dir"]]
    return {"card": card, "paths": paths, "wt_exists": wt_exists,
            "wt_dir_exists": wt_dir_exists, "branch_exists": branch_exists,
            "cmd": cmd}


def status_line(root: str, card: str) -> dict:
    """enter 用的轻量探测：目录在不在、分支在不在、干净不干净。"""
    paths = plan_paths(root, card)
    wt = os.path.isdir(os.path.join(paths["dir"], ".git")) or \
        os.path.isfile(os.path.join(paths["dir"], ".git"))
    r = run_git_capture(root, ["rev-parse", "--verify", "refs/heads/" + paths["branch"]])
    return {"card": card, "dir": paths["dir"], "branch": paths["branch"],
            "worktree": wt, "branch": r[0] == 0}


# ---------------- git 调用（测试可 monkeypatch） ----------------
def run_git_capture(root: str, args: list) -> tuple:
    try:
        p = subprocess.run(["git"] + list(args), cwd=root,
                           capture_output=True, text=True, timeout=60,
                           encoding="utf-8", errors="replace")
        return p.returncode, (p.stdout or "").strip(), (p.stderr or "").strip()
    except Exception as e:  # 无 git / 环境异常
        return 127, "", f"{type(e).__name__}: {e}"


def run_git_print(root: str, args: list) -> int:
    rc, out, err = run_git_capture(root, args)
    if rc == 0:
        if out:
            print("   " + out)
        return 0
    sys.stderr.write("   git 失败(rc=%d): %s\n" % (rc, (err or out or "")[:300]))
    return rc


# ---------------- 子命令 ----------------
def cmd_setup(args) -> int:
    root = os.path.abspath(args.root)
    card = check_card(args.card)
    plan = plan_setup(root, card)
    exe, exists, why = resolve_python(root, args.python)

    print(f"隔离工作树 setup —— 卡号={card}")
    print(f"  隔离目录: {os.path.relpath(plan['paths']['dir'], root) or plan['paths']['dir']}")
    print(f"  分支    : {plan['paths']['branch']}")
    print(f"  解释器  : {exe}（{why}；存在={exists}）")
    if not exists:
        print("  [警告] 该路径不存在 —— 裸机/无 venv 环境（R249）；"
              "跑批请用 `--python` 显式指定，别回落到默认值")
    if plan["wt_exists"]:
        print("  [复用] 该卡的工作树已存在 —— 幂等返回，不重建（不删数据）")
    elif plan["cmd"] is None:
        print("  [异常] 目录既不是 worktree 又无需建分支 —— 放弃（不猜）")
        return 1
    else:
        print("  → 执行: git " + " ".join(plan["cmd"]))
        rc = run_git_print(root, plan["cmd"])
        if rc != 0:
            print("  [错误] worktree 建立失败（多半是分支已被主工作区占用）")
            return 1
        print("  [隔离已生效] 主工作区 `git checkout " + plan["paths"]["branch"] +
              "` 会被 git 拒绝（分支已被本 worktree 占用）")
    print("-" * 60)
    print(MSG_NO_PRUNE)
    print("常用: python.exe " + os.path.join(".venv", "Scripts", "python.exe")
          if os.name == "nt" else "常用: " + exe)
    return 0


def cmd_enter(args) -> int:
    root = os.path.abspath(args.root)
    card = check_card(args.card)
    st = status_line(root, card)
    if not st["worktree"]:
        sys.stderr.write(f"[错误] 该卡没有隔离工作树: {st['dir']}\n"
                         f"       先跑 `worktree.py setup {card}`\n")
        return 1
    print(st["dir"])
    exe, exists, _ = resolve_python(root, args.python)
    print(f"（分支 {st['branch']}；解释器 {exe} exists={exists}）", file=sys.stderr)
    return 0


def cmd_clean(args) -> int:
    """清理工作树；**提交保留在分支上**（不删分支、不删数据，F-R10/R-R10）。"""
    root = os.path.abspath(args.root)
    card = check_card(args.card)
    paths = plan_paths(root, card)
    if not os.path.isdir(paths["dir"]):
        print(f"[复用/跳过] 目录不存在: {os.path.relpath(paths['dir'], root)}")
        return 0
    dirty = run_git_capture(root, ["-C", paths["dir"], "status", "--porcelain"])[1]
    if dirty and not args.force:
        print(f"[拒绝] 该工作树有未提交改动（{len(dirty.splitlines())} 处）⇒ "
              f"先提交，或加 `--force`（改动会随目录一起消失）")
        return 2
    rc = run_git_capture(root, ["worktree", "remove", paths["dir"], "--force"])
    if rc[0] != 0:
        sys.stderr.write("  git worktree remove 失败(rc=%d): %s\n" % (rc[0], rc[2][:300]))
        return 1
    print(f"已清理: {os.path.relpath(paths['dir'], root)} —— "
          f"分支 {paths['branch']} 与其中的提交**保留**（clean 不删数据）")
    if not args.force:
        print("（提示：本工具无 prune；分支清理请人工确认引用后 `git branch -D`）")
    return 0


def main(argv=None) -> int:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--root", default=DEFAULT_ROOT,
                        help="仓库根（默认自动探测 = collab-toolkit 的上一级）")
    ap = argparse.ArgumentParser(
        prog="worktree.py",
        description="任务卡隔离工作树 setup/enter/clean（R359 B1；🔴 无 prune）")
    sub = ap.add_subparsers(dest="cmd")

    p = sub.add_parser("setup", parents=[common], help="建隔离目录 + 检出 task/<卡号>")
    p.add_argument("card", help="卡号，如 R359-B1")
    p.add_argument("--python", default=None,
                   help="显式指定解释器（默认指向主仓 .venv；裸机必给）")
    p.set_defaults(fn=cmd_setup)

    p = sub.add_parser("enter", parents=[common], help="打印隔离目录路径")
    p.add_argument("card", help="卡号")
    p.add_argument("--python", default=None)
    p.set_defaults(fn=cmd_enter)

    p = sub.add_parser("clean", parents=[common], help="清理工作树（提交保留在分支上）")
    p.add_argument("card", help="卡号")
    p.add_argument("--force", action="store_true", help="允许丢弃未提交改动")
    p.set_defaults(fn=cmd_clean)

    args = ap.parse_args(argv)
    if not getattr(args, "fn", None):
        ap.print_help()
        return 1
    try:
        return args.fn(args)
    except WorktreeError as e:
        sys.stderr.write("[错误] %s\n" % e)
        return e.code


if __name__ == "__main__":
    sys.exit(main())
