#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""task_card.py —— 任务卡状态机（R359 B2）

一句话：**把"这件事做到哪了"从聊天记忆变成机器可读的一行卡**。

命令：
  add <卡号>        建卡（目标/输入/约束/验收，均**有字符上限**）
  claim <卡号> <人> 认领：pending -> claimed
  done <卡号>       提交：claimed -> done（结果 ≤1000 / 执行摘要 ≤200）
  verify <卡号>     验收：done|failed -> verified（终态）
  status            一行一卡，看清全部卡"现在到哪了"
  stale [--apply]   claimed 卡滞留超时 => 打回 pending（默认 dry-run）

状态机（照 orchestra §2）：
  pending -> claimed -> done -> verified（终态）
  failed  -> verified
  done / failed 均可被**打回 pending**（验收不通过：verify --reject）

存储：`_share/任务卡.md` —— **人可读 markdown 表格 + 机器可解析**（一行一卡，
正好是 `status` 的输出格式）。写 _share/ 前取锁（R111），全程零删除（F-R10）。

🔴 关键设计：每张卡**有字符上限**（结果 ≤1000、执行摘要 ≤200）。没有上限，
一行一卡就做不成；这也是"做到哪了"能被一眼扫完的前提。
"""
from __future__ import annotations

import argparse
import os
import re
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

DEFAULT_FILE = os.path.join("_share", "任务卡.md")
CARD_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]{0,63})$")
ID_RE = re.compile(r"^\d{2}-\d{2} \d{2}:\d{2}$")  # 卡内时间：10-03 20:05

# 状态机（ADJ = 允许从该状态出发的转移）
STATUSES = ("pending", "claimed", "done", "failed", "verified")
TERMINAL = ("verified",)
TRANSITIONS = {
    "add": ("pending",),              # 建卡只产生 pending
    "claim": ("pending",),            # pending -> claimed
    "done": ("claimed",),             # claimed -> done
    "fail": ("claimed",),             # claimed -> failed
    "verify": ("done", "failed"),     # -> verified
    "reject": ("done", "failed"),     # -> pending
    "stale": ("claimed",),            # claimed -> pending
}

# 🔴 字符上限（派工单 R359 §二：有上限才谈得上"一行一卡"）
LIMITS = {"goal": 200, "inputs": 300, "constraints": 200, "accept": 200,
          "result": 1000, "summary": 200, "owner": 40}
DEFAULT_LIMIT = 200

# stale 默认阈值（分钟）—— board_check 的异常检测同参
STALE_CLAIM_MIN = 30

HEADERS = ["卡号", "状态", "负责", "目标", "输入", "约束", "验收", "建卡", "更新", "结果"]
COL_OF = {h: i for i, h in enumerate(HEADERS)}
# 卡内时间列
COL_TS = (COL_OF["建卡"], COL_OF["更新"])


class CardError(Exception):
    def __init__(self, msg: str, code: int = 1):
        super().__init__(msg)
        self.code = code


def now_str() -> str:
    return datetime.now(TZ8).strftime("%Y-%m-%d %H:%M")


def card_ts(s: str):
    """解析卡内时间戳（10-03 20:05 或 2026-10-03 20:05）→ aware datetime 或 None。"""
    if not s:
        return None
    s = s.strip()
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d", "%m-%d %H:%M", "%m-%d"):
        try:
            d = datetime.strptime(s, fmt)
        except ValueError:
            continue
        return d.replace(tzinfo=TZ8) if d.tzinfo is None else d
    return None


def _now() -> datetime:
    return datetime.now(TZ8)


def minutes_since(s: str) -> float:
    t = card_ts(s)
    return 0.0 if t is None else (_now() - t).total_seconds() / 60.0


def esc(cell: str) -> str:
    return (cell or "").replace("|", "\\|").replace("\n", " ").strip()


def unesc(cell: str) -> str:
    return re.sub(r"\\\|", "|", cell or "").strip()


def check_card(card: str) -> str:
    if not CARD_RE.match(card or ""):
        raise CardError(f"卡号不合法: {card!r}（字母数字/点/下划线/连字符，"
                        f"字母数字开头；防命令注入）")
    return card


def check_limit(field: str, value: str) -> str:
    """字符上限校验（超限直接报错，不静默截断 —— 截断会丢证据）。"""
    lim = LIMITS.get(field, DEFAULT_LIMIT)
    if value is None:
        return ""
    if len(value) > lim:
        raise CardError(f"{field} 超长: {len(value)} > {lim} 字符 —— 拆到别处写，"
                        f"或精简后再提交（上限是「一行一卡」的前提）")
    return value


# ---------------- 纯解析（board_check 复用） ----------------
def parse_card_row(line: str) -> dict | None:
    """解析一行卡 → dict；非卡行返回 None。"""
    if not line.startswith("|"):
        return None
    cells = line.strip().strip("|").split("|")
    if len(cells) < len(HEADERS):
        return None
    d = {}
    for h in HEADERS:
        d[h] = unesc(cells[COL_OF[h]])
    if not d["卡号"] or d["状态"] not in STATUSES:
        return None
    return d


def parse_cards(text: str) -> list:
    out = []
    for line in text.splitlines():
        d = parse_card_row(line)
        if d:
            out.append(d)
    return out


def find_card(cards: list, card: str) -> dict | None:
    for d in cards:
        if d["卡号"] == card:
            return d
    return None


# ---------------- 文件 I/O ----------------
def card_path(root: str, path: str | None) -> str:
    p = path or DEFAULT_FILE
    return p if os.path.isabs(p) else os.path.join(root, p)


def read_cards(root: str, path: str | None = None) -> tuple:
    """返回 (text, cards)；文件不存在时 text=''。"""
    p = card_path(root, path)
    if not os.path.isfile(p):
        return "", []
    with open(p, encoding="utf-8") as f:
        text = f.read()
    return text, parse_cards(text)


def _render_header() -> str:
    return ("# 任务卡（R359 B2 —— 一行一卡：机器可解析 + 人可读）\n\n"
            "| " + " | ".join(HEADERS) + " |\n"
            "|" + "|".join(["---"] * len(HEADERS)) + "|\n")


def write_cards(root: str, cards: list, path: str | None = None,
                lock_tool: str | None = None, slot: str = "collab",
                task: str = "task_card", take_own: bool = False) -> None:
    """整表重写（文件小、表内即权威）；UTF-8 无 BOM + 写后回读校验（F-R31）。"""
    p = card_path(root, path)
    lines = [_render_header()]
    for d in cards:
        lines.append("| " + " | ".join(esc(d.get(h, "")) for h in HEADERS) + " |\n")
    body = "".join(lines)
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(body)
    # 回读校验 + 回滚
    with open(p, encoding="utf-8") as f:
        back = f.read()
    if back != body:
        with open(p, "w", encoding="utf-8", newline="\n") as f:
            f.write(body)
        raise CardError(f"写后回读不一致（{p}）⇒ 已回滚上次内容；检查是否被并发写入占用")


def acquire_lock(lock_tool: str, lock_dir: str, slot: str, task: str,
                 ttl: int, take_own: bool = False) -> None:
    try:
        r = subprocess.run([sys.executable, lock_tool, "acquire", "--slot", slot,
                            "--task", task, "--ttl", str(ttl),
                            "--lock-dir", lock_dir],
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=60)
    except Exception as e:
        raise CardError(f"取锁异常: {type(e).__name__}: {e}")
    if r.returncode == 4 and take_own:
        subprocess.run([sys.executable, lock_tool, "acquire", "--slot", slot,
                        "--task", task, "--ttl", str(ttl), "--refresh",
                        "--lock-dir", lock_dir],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
        return
    if r.returncode != 0:
        raise CardError(f"取锁失败(rc={r.returncode}): "
                        f"{(r.stderr or r.stdout or '').strip()[:200]}"
                        f"；或人工 `share_lock.py release --slot {slot}`")


def release_lock(lock_tool: str, lock_dir: str, slot: str) -> None:
    try:
        subprocess.run([sys.executable, lock_tool, "release", "--slot", slot,
                        "--lock-dir", lock_dir],
                       capture_output=True, text=True, timeout=60,
                       encoding="utf-8", errors="replace")
    except Exception:
        pass


# ---------------- 子命令 ----------------
def _need_card(cards: list, card: str) -> dict:
    d = find_card(cards, card)
    if not d:
        raise CardError(f"卡号不存在: {card}（先 add）", code=2)
    return d


def _need_state(d: dict, card: str, froms: tuple) -> None:
    if d["状态"] not in froms:
        raise CardError(f"{card} 当前状态 {d['状态']}，该操作只认 {froms}；"
                        f"状态机见文件头注释", code=2)


def cmd_add(args) -> int:
    root = os.path.abspath(args.root)
    card = check_card(args.card)
    ts = now_str()
    payload = {"卡号": card, "状态": "pending", "负责": args.owner or "",
               "目标": check_limit("goal", args.goal or ""),
               "输入": check_limit("inputs", args.inputs or ""),
               "约束": check_limit("constraints", args.constraints or ""),
               "验收": check_limit("accept", args.accept or ""),
               "建卡": ts, "更新": ts, "结果": ""}

    def _mutate(cards):
        old = find_card(cards, card)
        if old and not args.overwrite:
            raise CardError(f"卡号已存在: {card}（--overwrite 才更新）", code=2)
        if old:
            old.update(payload)
            return cards, f"{card} 已存在 ⇒ 覆盖更新（pending）"
        cards.append({k: payload[k] for k in HEADERS})
        return cards, f"{card} 建卡（pending）"

    _write_with_lock(root, card, args, _mutate)
    return 0


def cmd_claim(args) -> int:
    root = os.path.abspath(args.root)
    card = check_card(args.card)
    owner = args.owner or getattr(args, "owner_opt", "") or ""
    if not owner:
        raise CardError("claim 必须给认领人（位置参数或 --owner）")

    def _mutate(cards):
        d = _need_card(cards, card)
        _need_state(d, card, ("pending",))
        d["状态"], d["负责"], d["更新"] = "claimed", owner, now_str()
        return cards, f"{card} pending -> claimed（{owner}）"

    _write_with_lock(root, card, args, _mutate)
    return 0


def cmd_done(args) -> int:
    root = os.path.abspath(args.root)
    card = check_card(args.card)
    result = check_limit("result", args.result or "")
    summary = check_limit("summary", args.summary or "")
    body = (result + " ｜ " + summary) if (result and summary) else (result or summary)

    def _mutate(cards):
        d = _need_card(cards, card)
        _need_state(d, card, ("claimed",))
        d["状态"], d["结果"], d["更新"] = "done", body, now_str()
        return cards, f"{card} claimed -> done"

    _write_with_lock(root, card, args, _mutate)
    return 0


def cmd_verify(args) -> int:
    root = os.path.abspath(args.root)
    card = check_card(args.card)

    def _mutate(cards):
        d = _need_card(cards, card)
        _need_state(d, card, ("done", "failed"))
        if args.reject:
            d["状态"], d["更新"] = "pending", now_str()
            note = f"{card} done/failed -> pending（**验收打回**，待重做）"
        else:
            d["状态"], d["更新"] = "verified", now_str()
            note = f"{card} -> verified（终态）"
        return cards, note

    _write_with_lock(root, card, args, _mutate)
    return 0


def cmd_status(args) -> int:
    root = os.path.abspath(args.root)
    text, cards = read_cards(root, args.file)
    if args.json:
        print("[]" if not cards else
              __import__("json").dumps(cards, ensure_ascii=False, indent=2))
        return 0
    if not cards:
        print("（任务卡表为空 —— 尚无卡）")
        return 0
    width = args.width
    print(f"任务卡 {len(cards)} 张（状态：pending/claimed/done/failed/verified；"
          f"verified 为终态）")
    print("-" * 72)
    for d in cards:
        age = minutes_since(d["更新"])
        tail = ""
        if d["状态"] == "claimed":
            tail = f"（滞留 {age:.0f}min）"
        elif d["状态"] in ("done", "failed"):
            tail = f"（待核 {age:.0f}min）"
        goal = d["目标"].replace("\n", " ")
        ac = d["验收"].replace("\n", " ")
        print(f"{d['卡号']:<14} [{d['状态']:<8}] {d['负责'] or '—':<10} "
              f"{goal[:max(8, width - 46)]}{tail}")
        print(f"  验收: {ac[:max(8, width - 6)]}")
    print("-" * 72)
    n_claim = sum(1 for d in cards if d["状态"] == "claimed")
    n_done = sum(1 for d in cards if d["状态"] == "done")
    n_pend = sum(1 for d in cards if d["状态"] == "pending")
    n_verify = sum(1 for d in cards if d["状态"] == "verified")
    print(f"汇总: pending {n_pend} / claimed {n_claim} / done {n_done}（待核）/"
          f"verified {n_verify}")
    return 0


def cmd_stale(args) -> int:
    root = os.path.abspath(args.root)
    minutes = args.minutes
    text, cards = read_cards(root, args.file)
    hits = [d for d in cards if d["状态"] == "claimed"
            and minutes_since(d["更新"]) > minutes]
    if args.json:
        import json
        print(json.dumps([{"卡号": d["卡号"], "负责": d["负责"],
                           "状态": d["状态"], "更新": d["更新"],
                           "滞留分钟": round(minutes_since(d["更新"]), 1)}
                          for d in hits], ensure_ascii=False, indent=2))
        return 0
    if not hits:
        print(f"stale：无 claimed 卡滞留 > {minutes} 分钟")
        return 0
    print(f"stale：{len(hits)} 张 claimed 卡滞留 > {minutes} 分钟：")
    for d in hits:
        print(f"  {d['卡号']:<14} [{d['状态']}] {d['负责'] or '—':<10} "
              f"更新 {d['更新']}（{minutes_since(d['更新']):.0f}min）")
    if not args.apply:
        print("（dry-run；加 `--apply` 打回 pending）")
        return 0

    def _mutate(cards):
        hit_ids = {d["卡号"] for d in hits}
        done = []
        for d in cards:
            if d["状态"] == "claimed" and d["卡号"] in hit_ids:
                d["状态"], d["更新"] = "pending", now_str()
                done.append(d["卡号"])
        return cards, f"stale 打回 {len(done)} 张 → pending: {', '.join(done)}"

    _write_with_lock(root, "stale", args, _mutate)
    return 0


def _write_with_lock(root, card, args, mutate):
    """取锁 → 重读 → 变更（mutate）→ 写盘 → 放锁（R111）。

    `mutate(cards) -> (cards, note)`：同时支持单卡与整表变更（stale 改多张）。
    """
    lock_tool = args.lock_tool or os.path.join(root, "tools", "share_lock.py")
    lock_dir = os.path.join(root, "_share", ".locks")
    slot = args.lock_slot
    need_lock = getattr(args, "no_lock", False) is False
    if need_lock and not os.path.isfile(lock_tool):
        raise CardError(f"锁工具不存在: {lock_tool}（写 _share/ 前取锁；"
                        f"测试环境用 --no-lock）")
    if need_lock:
        acquire_lock(lock_tool, lock_dir, slot, f"task_card {card}",
                     args.ttl, take_own=getattr(args, "take_own_lock", False))
    try:
        text, cards = read_cards(root, args.file)
        cards, note = mutate(cards)
        write_cards(root, cards, args.file)
    finally:
        if need_lock:
            release_lock(lock_tool, lock_dir, slot)
    print(note)


def main(argv=None) -> int:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--root", default=DEFAULT_ROOT)
    common.add_argument("--file", default=None, help=f"卡表路径（默认 {DEFAULT_FILE}）")
    common.add_argument("--lock-tool", default=None, help="share_lock.py 路径")
    common.add_argument("--lock-slot", default="collab", help="锁槽位（默认 collab）")
    common.add_argument("--ttl", type=int, default=900, help="锁 TTL 秒")
    common.add_argument("--no-lock", action="store_true", help="跳过取锁（测试/离线）")
    common.add_argument("--take-own-lock", action="store_true",
                        help="检测到残留锁时自动 refresh 重取")

    ap = argparse.ArgumentParser(
        prog="task_card.py",
        description="任务卡状态机（R359 B2）：pending→claimed→done→verified")
    sub = ap.add_subparsers(dest="cmd")

    p = sub.add_parser("add", parents=[common], help="建卡")
    p.add_argument("card")
    p.add_argument("--goal", help=f"目标（≤{LIMITS['goal']}）")
    p.add_argument("--inputs", help=f"输入（≤{LIMITS['inputs']}）")
    p.add_argument("--constraints", help=f"约束（≤{LIMITS['constraints']}）")
    p.add_argument("--accept", help=f"验收（≤{LIMITS['accept']}）")
    p.add_argument("--owner", default="")
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(fn=cmd_add, _overwrite=False)

    p = sub.add_parser("claim", parents=[common], help="认领（pending → claimed）")
    p.add_argument("card")
    p.add_argument("owner", nargs="?", default=None)
    p.add_argument("--owner", dest="owner_opt", default=None)
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(fn=cmd_claim)

    p = sub.add_parser("done", parents=[common], help="提交（claimed → done）")
    p.add_argument("card")
    p.add_argument("--result", help=f"结果（≤{LIMITS['result']}）")
    p.add_argument("--summary", help=f"执行摘要（≤{LIMITS['summary']}）")
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(fn=cmd_done)

    p = sub.add_parser("verify", parents=[common], help="验收（done/failed → verified）")
    p.add_argument("card")
    p.add_argument("--reject", action="store_true", help="打回 pending（不通过）")
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(fn=cmd_verify)

    p = sub.add_parser("status", parents=[common], help="一行一卡")
    p.add_argument("--json", action="store_true")
    p.add_argument("--width", type=int, default=70)
    p.set_defaults(fn=cmd_status)

    p = sub.add_parser("stale", parents=[common], help="超时 claimed 打回 pending")
    p.add_argument("--minutes", type=int, default=STALE_CLAIM_MIN)
    p.add_argument("--apply", action="store_true")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_stale)

    args = ap.parse_args(argv)
    if not getattr(args, "fn", None):
        ap.print_help()
        return 1
    try:
        return args.fn(args)
    except CardError as e:
        sys.stderr.write("[错误] %s\n" % e)
        return e.code


if __name__ == "__main__":
    sys.exit(main())
