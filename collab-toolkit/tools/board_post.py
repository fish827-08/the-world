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
  - 🔴 **F-R10 家族适配（2026-09-18）**：本机**仓库内 Python 删除会 fail-closed，反复尝试
    会触发 SIGTERM**（"进程守卫"）⇒ 本工具**全程不做任何删除**；收尾只写标记/挪移；
    另加**信号兜底**（被杀时打印现场 + 写日志）与**阶段日志**（幂等恢复依据）

退出码：0 成功 / 1 环境或用法错误 / 2 锁占用 / 3 push 失败（提交已留本地）/ 4 对账失败
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
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


# ---------------- 阶段日志（幂等恢复依据；只写不删） ----------------
STATE = {"stage": "start", "journal": None, "args": None}


def journal_path(root: str, slot: str) -> str:
    return os.path.join(root, "_share", ".locks", f"{slot}.post-journal.json")


def journal_write(stage: str, extra: dict | None = None) -> None:
    """把当前阶段写入日志（覆盖写，不删除）。失败静默——日志是辅助，不影响主流程。"""
    STATE["stage"] = stage
    p = STATE.get("journal")
    if not p:
        return
    rec = {
        "slot": getattr(STATE["args"], "slot", "?"),
        "task": getattr(STATE["args"], "task", ""),
        "board": getattr(STATE["args"], "board", "?"),
        "stage": stage,
        "updated_at": datetime.now(TZ8).isoformat(timespec="seconds"),
        "pid": os.getpid(),
    }
    rec.update(extra or {})
    try:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8", newline="\n") as f:
            json.dump(rec, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def journal_read(root: str, slot: str) -> dict:
    try:
        with open(journal_path(root, slot), encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def install_signal_guard() -> None:
    """被杀兜底：SIGTERM/SIGINT 时打印现场 + 写日志，再退出（不做删除动作）。

    F-R10 家族实测：本机可能对含"删除类动作"的进程直接 SIGTERM。
    这里只打印/写文件（不删除、不重放 git 操作），避免二次触发。
    """
    def handler(signum, frame):
        rec = journal_read(getattr(STATE["args"], "root", "."),
                           getattr(STATE["args"], "slot", "?"))
        print()
        print(f"[中断] 收到信号 {signum} —— 本工具被环境终止（F-R10 家族已知现象）")
        print(f"       最后阶段: {STATE['stage']}（阶段日志: {STATE.get('journal')}）")
        if rec:
            print(f"       日志记录: stage={rec.get('stage')} task={rec.get('task')}")
        print("       恢复: 直接**重跑本脚本**（幂等：已追加的内容不会重复上板）；"
              "若锁残留: `python tools/share_lock.py release --slot <slot>`"
              "（新版 release 走挪移，不删除）")
        raise SystemExit(130)

    for sig in (getattr(signal, "SIGTERM", None), getattr(signal, "SIGINT", None)):
        if sig is not None:
            try:
                signal.signal(sig, handler)
            except Exception:
                pass


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


TRANSIENT_HINTS = ("index.lock", "unable to create", "could not lock",
                   "resource temporarily unavailable", "connection reset",
                   "operation timed out")


def run_git(root: str, args: list, check: bool = True,
            credential_helper: str | None = None,
            retries: int = 1) -> subprocess.CompletedProcess:
    """跑 git；对**瞬时故障**自动重试一次（Windows 索引锁/连接抖动等）。

    flaky 定位（2026-09-18）：同类失败在不同轮次随机出现，且报错多为
    `index.lock`/空 stderr —— 属环境抖动而非逻辑错误 ⇒ 单次重试 + 记录。
    非瞬时失败（如 push 被拒）不重试，交主流程判断。
    """
    cmd = ["git"]
    if credential_helper:
        cmd += ["-c", f"credential.helper={credential_helper}"]
    cmd += args
    last = None
    for attempt in range(retries + 1):
        r = subprocess.run(cmd, cwd=root, capture_output=True,
                           text=True, encoding="utf-8", errors="replace")
        last = r
        if r.returncode == 0:
            return r
        msg = ((r.stderr or "") + (r.stdout or "")).lower()
        transient = (not msg.strip()) or any(h in msg for h in TRANSIENT_HINTS)
        if not transient or attempt >= retries:
            break
        print(f"      [重试] git {' '.join(args)} 瞬时失败(rc={r.returncode})，"
              f"{(r.stderr or r.stdout).strip()[:120]} —— 0.5s 后重试")
        time.sleep(0.5)
    if check and last is not None and last.returncode != 0:
        raise PostError(
            f"git {' '.join(args)} 失败(rc={last.returncode}): "
            f"{(last.stderr or last.stdout).strip()[:300]}")
    return last


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
    if r.returncode == 4 and getattr(args, "take_own_lock", False):
        print("      （检测到本槽位残留锁 + 已指定 --take-own-lock："
              "走 --refresh 挪移旧锁后重取）")
        r = lock_cmd(args, "acquire", "--slot", args.slot, "--task", args.task,
                     "--ttl", str(args.ttl), "--refresh")
    sys.stdout.write(r.stdout or "")
    if r.returncode == 2:
        raise PostError("他人持活跃锁 —— 按 §3.1.2 等待，不得强行写入", code=2)
    if r.returncode == 4:
        # 同槽位锁已存在：可能是你上次进程被中途终止（环境已知问题）留下的
        raise PostError(
            f"槽位 {args.slot} 已有锁（可能是上次进程被中断的残留）。"
            f"先确认: `share_lock.py status`；确认为残留后 "
            f"`share_lock.py release --slot {args.slot}`（挪移，不删除），再重跑本脚本"
            f"——或加 `--take-own-lock` 让本工具自动挪移重取（幂等，不会重复上帖）",
            code=2)
    if r.returncode != 0:
        raise PostError(f"取锁失败(rc={r.returncode}): {(r.stderr or '').strip()[:200]}")


def release_lock(args) -> None:
    r = lock_cmd(args, "release", "--slot", args.slot)
    sys.stdout.write(r.stdout or "")
    if r.returncode != 0:
        print(f"[警告] 放锁失败(rc={r.returncode})——TTL {args.ttl}s 后自动失效，"
              f"或人工 `share_lock.py release --slot {args.slot}`")


# ---------------- 写侧编码自检（F-R31：emoji 被写成 CESU-8 代理对） ----------------
try:
    from board_check import utf8_problems  # type: ignore
except Exception:
    def utf8_problems(data):                # 退化：仅校验可解码
        raw = data.encode("utf-8") if isinstance(data, str) else data
        try:
            raw.decode("utf-8")
            return []
        except UnicodeDecodeError as e:
            return [f"utf-8 解码失败: {e}"]


def guard_utf8(text: str, what: str) -> None:
    """写入前自检：发现 CESU-8/解码问题即中止（不把污染写进文件）。"""
    problems = utf8_problems(text)
    if problems:
        raise PostError(f"{what} 编码自检未过（F-R31 家族）: {'; '.join(problems)}"
                        f"\n        修法：用文本/工具写入真实 UTF-8 字符，"
                        f"勿用 `\\uD83D\\uDD34` 这类代理对转义写文件")


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
    STATE["args"] = args
    STATE["journal"] = journal_path(root, args.slot)
    install_signal_guard()
    try:
        # ⓪ 环境检查
        if not os.path.isdir(os.path.join(root, ".git")):
            raise PostError(f"不是 git 仓库: {root}")
        if not os.path.isfile(board):
            raise PostError(f"讨论板不存在: {board}")
        if not os.path.isfile(args.lock_tool):
            raise PostError(f"锁工具不存在: {args.lock_tool}")
        message = ensure_signature(load_message(args.message_file), args.role)
        guard_utf8(message, "发言内容")   # F-R31：写前自检
        prev = journal_read(root, args.slot)   # 先读上次记录，再写本次
        journal_write("start", {"message_sha16": sha256_text(message),
                                "message_chars": len(message)})
        if prev and prev.get("stage") not in (None, "done") \
                and prev.get("message_sha16") == sha256_text(message):
            print(f"[提示] 阶段日志显示上次同期运行中断在 '{prev.get('stage')}'，"
                  f"本次将按幂等路径续做（阶段日志: {STATE['journal']}）")

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
        journal_write("locked")

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
            with open(board, "rb") as f:           # 写后回读自检（F-R31）
                after = utf8_problems(f.read())
            if after:
                with open(board, "w", encoding="utf-8", newline="\n") as f:
                    f.write(old)                   # 回滚到写前内容（不删除）
                raise PostError(f"写入后编码自检未过，已回滚: {'; '.join(after)}")
            print(f"[4/7] 已追加 {len(new) - len(old)} 字节"
                  f"（板面 {len(old)} -> {len(new)}）")
            journal_write("appended", {"board_bytes": len(new)})

        # ③ 路径限定提交 + show --stat 核文件数（无变更则跳过；先看暂存区再提交，
        #    避免"nothing to commit"在 Windows/CRLF 环境下变成假失败）
        dirty = run_git(root, ["status", "--porcelain", "--", args.board]).stdout.strip()
        if dirty:
            print("[5/7] 提交（仅讨论板路径）...")
            run_git(root, ["add", "--", args.board])
            staged = run_git(root, ["diff", "--cached", "--quiet", "--", args.board],
                             check=False, retries=0).returncode
            if staged == 0:
                print("      暂存区无实际变更（可能是行尾归一化）——跳过提交")
            else:
                who = (args.role or f"[{args.slot}]").strip()
                # 🔴 R236 修复：**必须带 pathspec**（`-- <board>`）。
                # 不带 pathspec 的 `git commit` 会提交"索引里的一切" —— 而多会话**共用同一工作树/索引**
                # （教训㉒：git 工作区是单例资源），别线此刻 staged 的改动会被本线**一并提交**，
                # 等于替别人提交未完成的活（2026-09-27 实测事故：轻舟 T-F 的 4 个文件被 R236 上板一起提交）。
                run_git(root, ["commit", "-m", f"share: {who} {args.task}", "--", args.board])
            stat = run_git(root, ["show", "--stat", "--format=", "HEAD"]).stdout
            files = [ln for ln in stat.splitlines() if "|" in ln]
            if len(files) != 1:
                raise PostError(
                    f"提交文件数异常（期望 1，实际 {len(files)}）——已提交，"
                    f"请人工 `git show --stat` 核对:\n{stat.strip()[:300]}")
        else:
            print("[5/7] 板面无未提交变更——直接补推")
        journal_write("committed", {"head": run_git(root, ["rev-parse", "--short", "HEAD"]).stdout.strip()})

        # ④ push + 对账
        print("[6/7] push ...")
        r = run_git(root, ["push", args.remote, f"HEAD:{args.branch}"],
                    check=False, credential_helper=args.credential_helper or None)
        if r.returncode != 0:
            journal_write("push_failed")
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
            journal_write("verify_failed", {"head": head, "remote": remote_sha})
            raise PostError(
                f"对账失败: 本地 HEAD={head[:8]} 远端={remote_sha[:8] or '?'} —— "
                f"push 疑似未生效，请人工 ls-remote 复核", code=4)
        journal_write("done", {"head": head})
        print(f"[7/7] 对账 OK: HEAD == {args.remote}/{args.branch} ({head[:8]})")
        print("[OK] 发言完成（未 push 不算发言 —— 已 push 并对账）")
        return 0
    except PostError as e:
        journal_write(f"abort: {e.code}")
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
    ap.add_argument("--take-own-lock", action="store_true",
                    help="本槽位有残留锁时自动挪移重取（仅用于确认是自己上次被中断的锁）")
    ap.add_argument("--dry-run", action="store_true", help="只校验不执行")
    args = ap.parse_args(argv)
    return cmd_post(args)


if __name__ == "__main__":
    sys.exit(main())
