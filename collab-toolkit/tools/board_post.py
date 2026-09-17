#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""board_post.py — 讨论板发言全流程封装（零号规则"带锁 7 步"一键执行，协作线 T-1）

把 AGENT.md §3.1.1 + §3.1.2 的手工流程固化为脚本：
    fetch -> pull --ff-only -> 取锁 -> 重读板尾 -> 校验并追加
    -> 路径限定 commit -> push -> ls-remote 对账 -> 放锁

纪律内建：
  - 只追加：新内容 = 旧内容 + 分隔符 + 新帖（构造上保证 append-only）
  - UTF-8 无 BOM 校验（F-R15 系列教训）
  - 路径限定提交 + `git show --stat` 核文件数（应为 1）
  - push 失败不自动 rebase（教训 9：rebase 中断会损 .git）——保留本地提交，人工处理
  - 对账用 ls-remote（F-R24：refs/remotes 不可信）
  - 锁异常退出也会放锁（finally）

退出码：0 成功 / 1 环境或用法错误 / 2 他人活跃锁 / 3 push 失败（提交已留本地）/ 4 对账失败
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
SIG_START_RE = re.compile(r"^###\s*\[[^\]]+\]\s*·\s*\d{4}-\d{2}-\d{2}[ T]+\d{1,2}:\d{2}")


class PostError(Exception):
    def __init__(self, msg: str, code: int = 1):
        super().__init__(msg)
        self.code = code


def run_git(root: str, args: list, check: bool = True,
            credential_helper: str | None = None) -> subprocess.CompletedProcess:
    cmd = ["git"]
    if credential_helper:
        cmd += ["-c", f"credential.helper={credential_helper}"]
    cmd += args
    r = subprocess.run(cmd, cwd=root, capture_output=True,
                       text=True, encoding="utf-8", errors="replace")
    if check and r.returncode != 0:
        raise PostError(
            f"git {' '.join(args)} 失败(rc={r.returncode}): "
            f"{(r.stderr or r.stdout).strip()[:300]}")
    return r


# ---------------- 消息处理（纯函数，可测） ----------------
def load_message(path: str) -> str:
    """读消息文件：必须 UTF-8 无 BOM。"""
    with open(path, "rb") as f:
        raw = f.read()
    if raw.startswith(b"\xef\xbb\xbf"):
        raise PostError(f"消息文件带 UTF-8 BOM（违反无 BOM 纪律）: {path}")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as e:
        raise PostError(f"消息文件不是合法 UTF-8: {path} ({e})")
    if not text.strip():
        raise PostError("消息为空")
    return text.strip("\n")


def ensure_signature(message: str, role: str | None) -> str:
    """消息首行须为带时分的署名行；缺省时若给了 --role 则自动补。"""
    if SIG_START_RE.match(message):
        return message
    if not role:
        raise PostError(
            "消息首行不是署名行（### [角色] · YYYY-MM-DD HH:MM …）；"
            "请补署名头，或用 --role 让脚本自动生成")
    now = datetime.now(TZ8)
    head = (f"### {role} · {now.strftime('%Y-%m-%d %H:%M')}"
            f"（Asia/Shanghai）")
    return head + "\n\n" + message


def build_appended(old: str, message: str) -> str:
    """构造追加结果：old + 分隔 + 新帖。保证 old 是新内容的严格前缀。"""
    body = old.rstrip("\n") + "\n\n---\n\n" + message.rstrip("\n") + "\n"
    assert body.startswith(old.rstrip("\n")), "append-only 构造失败"
    return body


def board_tail_summary(board_path: str, n: int = 1) -> str:
    """板尾最近 n 条署名帖的一句话摘要（append 前重读板尾用）。"""
    sig_re = re.compile(
        r"^###\s*\[[^\]]+\]\s*·\s*\d{4}-\d{2}-\d{2}(?:[ T]+\d{1,2}:\d{2})?")
    hits = []
    with open(board_path, encoding="utf-8") as f:
        for line in f:
            if sig_re.match(line):
                hits.append(line.strip())
    if not hits:
        return "（板内无署名帖）"
    return "\n".join("  板尾: " + h for h in hits[-n:])


# ---------------- 锁 ----------------
def lock_cmd(args, *lock_args) -> subprocess.CompletedProcess:
    cmd = [sys.executable, args.lock_tool] + list(lock_args) + [
        "--lock-dir", os.path.join(args.root, "_share", ".locks")]
    return subprocess.run(cmd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")


def acquire_lock(args) -> None:
    r = lock_cmd(args, "acquire", "--slot", args.slot,
                 "--task", args.task, "--ttl", str(args.ttl))
    sys.stdout.write(r.stdout or "")
    if r.returncode == 2:
        raise PostError("他人持活跃锁 —— 按 §3.1.2 等待，不得强行写入", code=2)
    if r.returncode == 4:
        # 同槽位锁已存在：可能是你上次进程被中途终止（环境已知问题）留下的
        raise PostError(
            f"槽位 {args.slot} 已有锁（可能是上次进程被中断的残留）。"
            f"先确认: `share_lock.py status`；确认为残留后 "
            f"`share_lock.py release --slot {args.slot}`，再重跑本脚本（幂等，不会重复上帖）",
            code=2)
    if r.returncode != 0:
        raise PostError(f"取锁失败(rc={r.returncode}): {(r.stderr or '').strip()[:200]}")


def release_lock(args) -> None:
    r = lock_cmd(args, "release", "--slot", args.slot)
    sys.stdout.write(r.stdout or "")
    if r.returncode != 0:
        print(f"[警告] 放锁失败(rc={r.returncode})——TTL {args.ttl}s 后自动失效，"
              f"或人工 `share_lock.py release --slot {args.slot}`")


# ---------------- 置顶块兼容（T-6） ----------------
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from board_pin import split_board as _split_board  # type: ignore
    PIN_SUPPORT = True
except Exception:  # 退化：整块当正文（置顶区不会被误改，但也不会被特殊对待）
    PIN_SUPPORT = False

    def _split_board(text):
        return None, text


# ---------------- 主流程 ----------------
def cmd_post(args) -> int:
    root = os.path.abspath(args.root)
    args.root = root
    board = os.path.join(root, args.board)
    locked = False
    try:
        # ⓪ 环境检查
        if not os.path.isdir(os.path.join(root, ".git")):
            raise PostError(f"不是 git 仓库: {root}")
        if not os.path.isfile(board):
            raise PostError(f"讨论板不存在: {board}")
        if not os.path.isfile(args.lock_tool):
            raise PostError(f"锁工具不存在: {args.lock_tool}")
        message = ensure_signature(load_message(args.message_file), args.role)

        if args.dry_run:
            print("[dry-run] 仅校验，不写板、不取锁、不提交")
            print(f"[dry-run] 仓库: {root}")
            print(f"[dry-run] 目标板: {args.board}（{os.path.getsize(board)} 字节）")
            print(f"[dry-run] 消息 {len(message)} 字符，署名行: {message.splitlines()[0]}")
            print("[dry-run] 板尾核对: " + board_tail_summary(board))
            print("[dry-run] 通过——去掉 --dry-run 即按 7 步执行")
            return 0

        # ① fetch + pull --ff-only（F-R24 下 fetch 的 refs 告警无害，pull 走 FETCH_HEAD）
        print("[1/7] git fetch + pull --ff-only ...")
        run_git(root, ["fetch", args.remote], check=False)
        r = run_git(root, ["pull", "--ff-only", args.remote, args.branch], check=False)
        if r.returncode != 0:
            raise PostError(
                "pull --ff-only 失败（本地可能与远端分叉）——"
                "请人工处理后再发帖，脚本不自动 rebase:\n"
                + (r.stderr or r.stdout).strip()[:300])

        # ⓪ 取锁（pull 之后取，缩小持锁窗口）
        print("[2/7] 取写入锁 ...")
        acquire_lock(args)
        locked = True

        # ② 重读板尾 + 追加（只动置顶块之后的正文；幂等防重复帖）
        print("[3/7] 重读板尾（确认无人插队）:")
        print(board_tail_summary(board))
        with open(board, encoding="utf-8") as f:
            old = f.read()
        pin, body = _split_board(old)  # T-6：置顶块原样保留，正文按只追加处理
        if pin:
            print("      （检测到置顶区：本次只追加正文，置顶块原样保留）")
        if message.strip() in body:
            print("[4/7] 相同内容已在板上——跳过追加（幂等保护，防重复帖）")
        else:
            new = build_appended(body, message)
            if pin is not None:
                new = pin + new
            with open(board, "w", encoding="utf-8", newline="\n") as f:
                f.write(new)
            print(f"[4/7] 已追加 {len(new) - len(old)} 字节"
                  f"（板面 {len(old)} -> {len(new)}）")

        # ③ 路径限定提交 + show --stat 核文件数（无变更则跳过）
        dirty = run_git(root, ["status", "--porcelain", "--", args.board]).stdout.strip()
        if dirty:
            print("[5/7] 提交（仅讨论板路径）...")
            run_git(root, ["add", "--", args.board])
            who = (args.role or f"[{args.slot}]").strip()
            run_git(root, ["commit", "-m", f"share: {who} {args.task}"])
            stat = run_git(root, ["show", "--stat", "--format=", "HEAD"]).stdout
            files = [ln for ln in stat.splitlines() if "|" in ln]
            if len(files) != 1:
                raise PostError(
                    f"提交文件数异常（期望 1，实际 {len(files)}）——已提交，"
                    f"请人工 `git show --stat` 核对:\n{stat.strip()[:300]}")
        else:
            print("[5/7] 板面无未提交变更——直接补推")

        # ④ push + 对账
        print("[6/7] push ...")
        r = run_git(root, ["push", args.remote, f"HEAD:{args.branch}"],
                    check=False, credential_helper=args.credential_helper or None)
        if r.returncode != 0:
            raise PostError(
                "push 被拒绝——远端可能有新提交。提交已保留在本地；"
                "处理办法：`git pull --ff-only` 后**直接重跑本脚本**"
                "（幂等设计：同内容不会重复上板）；勿用 --rebase（教训 9）:\n"
                + (r.stderr or r.stdout).strip()[:300], code=3)
        head = run_git(root, ["rev-parse", "HEAD"]).stdout.strip()
        rr = run_git(root, ["ls-remote", args.remote, f"refs/heads/{args.branch}"],
                     check=False)
        remote_sha = rr.stdout.split()[0] if rr.stdout.strip() else ""
        if remote_sha != head:
            raise PostError(
                f"对账失败: 本地 HEAD={head[:8]} 远端={remote_sha[:8] or '?'} —— "
                f"push 疑似未生效，请人工 ls-remote 复核", code=4)
        print(f"[7/7] 对账 OK: HEAD == {args.remote}/{args.branch} ({head[:8]})")
        print("[OK] 发言完成（未 push 不算发言 —— 已 push 并对账）")
        return 0
    except PostError as e:
        print(f"[中止] {e}")
        return e.code
    finally:
        if locked:
            release_lock(args)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="board_post.py",
        description="讨论板发言全流程封装（零号规则带锁 7 步）")
    ap.add_argument("--root", default=DEFAULT_ROOT, help="仓库根（默认自动探测）")
    ap.add_argument("--board", default=os.path.join("_share", "讨论板.md"),
                    help="目标板（相对仓库根，默认 _share/讨论板.md）")
    ap.add_argument("--remote", default="gitee")
    ap.add_argument("--branch", default="main")
    ap.add_argument("--slot", required=True, help="锁槽位（owner/dev/eval/web/cloud/collab…）")
    ap.add_argument("--task", required=True, help="本次写入事项（一句话，进锁与提交信息）")
    ap.add_argument("--message-file", required=True, help="发言内容文件（UTF-8 无 BOM）")
    ap.add_argument("--role", default=None,
                    help="板面角色（如 '[协作]'）；消息缺署名头时自动生成")
    ap.add_argument("--ttl", type=int, default=900, help="锁 TTL 秒（默认 900）")
    ap.add_argument("--lock-tool",
                    default=os.path.join(DEFAULT_ROOT, "tools", "share_lock.py"),
                    help="share_lock.py 路径")
    ap.add_argument("--credential-helper", default="manager",
                    help="push 凭据 helper（默认 manager；空串禁用）")
    ap.add_argument("--dry-run", action="store_true", help="只校验不执行")
    args = ap.parse_args(argv)
    return cmd_post(args)


if __name__ == "__main__":
    sys.exit(main())
