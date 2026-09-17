#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""board_pin.py — 讨论板"置顶区"管理（协作线 T-6，fish 2026-09-17 指示）

置顶区 = 讨论板**顶部**的一个受控区块，用于放"当前要事"（待决事项/公告/必读指引），
让新会话与各线一眼看到重点（治 P5：角色上下文重载成本）。

## 纪律（与"只追加"并存的关键设计）

1. 置顶块由**显式标记**包住：`<!-- PIN-BEGIN -->` … `<!-- PIN-END -->`
2. **块外内容仍严格只追加**：所有写板工具先 `split_board()` 把板切成
   `(pin, rest)`，插入/追加只动各自部分，`rest` 保持字节不变
3. 仅 `[所有者]`/fish 维护置顶内容（本工具会提示；不强制身份验证）
4. 上限：**5 条 / 2KB**；超过即报错（防置顶区自己变成新的膨胀源）
5. 新鲜度：头部含"更新于 YYYY-MM-DD"，超过 7 天未更新 → 体检报"过期"
6. 写操作走写入锁（§3.1.2），与发言同规矩

用法：
    python board_pin.py show                       # 打印置顶区
    python board_pin.py set --file pin.md          # 用文件内容替换置顶区（自动包标记）
    python board_pin.py add --text "…"             # 追加一条
    python board_pin.py remove --index 2           # 删除第 2 条
    python board_pin.py check                      # 校验（标记/大小/条数/新鲜度）

退出码：0 成功 / 1 环境或用法错误 / 2 他人活跃锁 / 3 校验不通过
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

PIN_BEGIN = "<!-- PIN-BEGIN -->"
PIN_END = "<!-- PIN-END -->"
PIN_MAX_ITEMS = 5
PIN_MAX_BYTES = 2048
PIN_MAX_AGE_DAYS = 7
PIN_HEADER_RE = re.compile(r"更新于\s*(\d{4}-\d{2}-\d{2})")
PIN_ITEM_RE = re.compile(r"^>\s*\d+\.\s")


class PinError(Exception):
    def __init__(self, msg: str, code: int = 1):
        super().__init__(msg)
        self.code = code


# ---------------- 切分 / 拼接（纯函数，可测） ----------------
def split_board(text: str) -> tuple:
    """→ (pin_block 或 None, rest)。rest 含置顶块之后的一切（字节不变）。"""
    i = text.find(PIN_BEGIN)
    if i < 0:
        return None, text
    j = text.find(PIN_END, i)
    if j < 0:
        raise PinError("置顶块有 BEGIN 无 END（板面损坏，请人工修复）")
    j += len(PIN_END)
    return text[i:j], text[j:]


def join_board(pin: str | None, rest: str) -> str:
    return rest if pin is None else pin + rest


def parse_items(pin: str) -> list:
    return [ln for ln in pin.splitlines() if PIN_ITEM_RE.match(ln)]


def pin_age_days(pin: str, today=None) -> float | None:
    m = PIN_HEADER_RE.search(pin or "")
    if not m:
        return None
    d = datetime.strptime(m.group(1), "%Y-%m-%d").date()
    today = today or datetime.now(TZ8).date()
    return (today - d).days


def build_pin(items: list, note: str = "", now=None) -> str:
    now = now or datetime.now(TZ8)
    head = (f"> 📌 **置顶区**（维护：`[所有者]`｜更新于 "
            f"{now.strftime('%Y-%m-%d %H:%M')}｜上限 {PIN_MAX_ITEMS} 条 / "
            f"{PIN_MAX_BYTES}B）")
    lines = [PIN_BEGIN, head]
    if note:
        lines.append(f"> {note}")
    for n, it in enumerate(items, 1):
        lines.append(f"> {n}. {it.strip()}")
    lines += [">", "> 说明：本区块由 `[所有者]` 维护；改动请走"
                  " `collab-toolkit/tools/board_pin.py`；块外内容仍只追加。",
              PIN_END]
    return "\n".join(lines) + "\n"


def validate_pin(pin: str, today=None) -> list:
    out = []
    if pin is None:
        return [{"level": "warn", "item": "pin", "msg": "无置顶区（未启用）"}]
    items = parse_items(pin)
    size = len(pin.encode("utf-8"))
    if size > PIN_MAX_BYTES:
        out.append({"level": "error", "item": "pin",
                    "msg": f"置顶区 {size}B 超过上限 {PIN_MAX_BYTES}B"})
    if len(items) > PIN_MAX_ITEMS:
        out.append({"level": "error", "item": "pin",
                    "msg": f"置顶条目 {len(items)} 条超过上限 {PIN_MAX_ITEMS} 条"})
    age = pin_age_days(pin, today)
    if age is None:
        out.append({"level": "warn", "item": "pin", "msg": "置顶区缺'更新于 YYYY-MM-DD'字段"})
    elif age > PIN_MAX_AGE_DAYS:
        out.append({"level": "warn", "item": "pin",
                    "msg": f"置顶区已 {age} 天未更新（阈值 {PIN_MAX_AGE_DAYS} 天）"})
    if not out:
        out.append({"level": "ok", "item": "pin",
                    "msg": f"置顶区正常：{len(items)} 条 / {size}B / "
                           f"{age} 天前更新"})
    return out


# ---------------- 落盘（带锁） ----------------
def _run(cmd, cwd) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")


def lock(args, *lock_args) -> subprocess.CompletedProcess:
    return _run([sys.executable, args.lock_tool] + list(lock_args)
                + ["--lock-dir", os.path.join(args.root, "_share", ".locks")],
                args.root)


def write_board(args, new_text: str) -> None:
    """写板：取锁 → 写 → 放锁（写失败也放锁）。"""
    acq = lock(args, "acquire", "--slot", args.slot, "--task", args.task,
               "--ttl", str(args.ttl))
    sys.stdout.write(acq.stdout or "")
    if acq.returncode == 2:
        raise PinError("他人持活跃锁 —— 等待（§3.1.2）", code=2)
    if acq.returncode != 0:
        raise PinError(f"取锁失败(rc={acq.returncode})")
    try:
        with open(args.board, "w", encoding="utf-8", newline="\n") as f:
            f.write(new_text)
    finally:
        rel = lock(args, "release", "--slot", args.slot)
        sys.stdout.write(rel.stdout or "")


def read_board(args) -> str:
    if not os.path.isfile(args.board):
        raise PinError(f"讨论板不存在: {args.board}")
    with open(args.board, encoding="utf-8") as f:
        return f.read()


# ---------------- 命令 ----------------
def cmd_show(args) -> int:
    text = read_board(args)
    pin, _ = split_board(text)
    if pin is None:
        print("[提示] 未启用置顶区（板上无 PIN-BEGIN 标记）")
        return 0
    print(pin)
    return 0


def cmd_check(args) -> int:
    text = read_board(args)
    pin, _ = split_board(text)
    findings = validate_pin(pin)
    for f in findings:
        icon = {"ok": "[OK]  ", "warn": "[警告]", "error": "[错误]"}.get(f["level"], "[?]")
        print(f"{icon} {f['msg']}")
    return 3 if any(f["level"] == "error" for f in findings) else 0


def cmd_set(args) -> int:
    with open(args.file, encoding="utf-8") as f:
        raw = f.read()
    # 允许文件里已含标记，或只有纯条目
    if PIN_BEGIN in raw:
        new_pin = raw[raw.find(PIN_BEGIN):].strip() + "\n"
        body_hint = "（文件已含标记，直接采用）"
    else:
        items = [ln.lstrip("-* ").strip() for ln in raw.splitlines()
                 if ln.strip() and not ln.strip().startswith(">")]
        if not items:
            raise PinError("置顶内容为空")
        new_pin = build_pin(items)
        body_hint = f"（由 {len(items)} 条纯条目构建）"
    findings = validate_pin(new_pin)
    bad = [f for f in findings if f["level"] == "error"]
    if bad:
        for f in bad:
            print(f"[错误] {f['msg']}")
        return 3
    text = read_board(args)
    pin_old, rest = split_board(text)
    if pin_old is None:
        # 首次启用：置顶块插到板首，其后保持原文（只加不减）
        new_text = new_pin + "\n---\n\n" + text.lstrip("\n")
    else:
        # 已有置顶块：只替换块本身，rest 字节不变（块外仍只追加）
        new_text = join_board(new_pin, rest)
    if args.dry_run:
        print(f"[dry-run] 置顶区将更新 {body_hint}；板面 {len(text)} -> {len(new_text)} 字节")
        print(new_pin)
        return 0
    write_board(args, new_text)
    print(f"[OK] 置顶区已更新 {body_hint}；板面 {len(text)} -> {len(new_text)} 字节")
    return 0


def _load_items(args) -> tuple:
    text = read_board(args)
    pin, rest = split_board(text)
    if pin is None:
        raise PinError("板上无置顶区 —— 先用 set 创建")
    return pin, rest, parse_items(pin)


def cmd_add(args) -> int:
    pin, rest, items = _load_items(args)
    items.append(args.text)
    new_pin = build_pin(items)
    findings = validate_pin(new_pin)
    if any(f["level"] == "error" for f in findings):
        for f in findings:
            if f["level"] == "error":
                print(f"[错误] {f['msg']}")
        return 3
    if args.dry_run:
        print("[dry-run]" + new_pin)
        return 0
    write_board(args, join_board(new_pin, rest))
    print(f"[OK] 已追加置顶条目（现 {len(items)} 条）")
    return 0


def cmd_remove(args) -> int:
    pin, rest, items = _load_items(args)
    idx = args.index
    if not (1 <= idx <= len(items)):
        raise PinError(f"序号越界：{idx}（现 {len(items)} 条）")
    removed = items.pop(idx - 1)
    if not items:
        raise PinError("删除后置顶区为空 —— 请用 set 重写或保留至少 1 条")
    if args.dry_run:
        print(f"[dry-run] 将删除第 {idx} 条: {removed}")
        return 0
    write_board(args, join_board(build_pin(items), rest))
    print(f"[OK] 已删除第 {idx} 条: {removed}")
    return 0


def main(argv=None) -> int:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--root", default=DEFAULT_ROOT)
    common.add_argument("--board", default=None)
    common.add_argument("--slot", default="owner", help="锁槽位（默认 owner）")
    common.add_argument("--task", default="维护讨论板置顶区")
    common.add_argument("--ttl", type=int, default=900)
    common.add_argument("--lock-tool",
                        default=os.path.join(DEFAULT_ROOT, "tools", "share_lock.py"))
    common.add_argument("--dry-run", action="store_true")

    ap = argparse.ArgumentParser(prog="board_pin.py",
                                 description="讨论板置顶区管理（T-6）")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn, help_ in (("show", cmd_show, "打印置顶区"),
                            ("check", cmd_check, "校验置顶区")):
        sub.add_parser(name, parents=[common], help=help_).set_defaults(fn=fn)
    p_set = sub.add_parser("set", parents=[common], help="替换置顶区")
    p_set.add_argument("--file", required=True)
    p_set.set_defaults(fn=cmd_set)
    p_add = sub.add_parser("add", parents=[common], help="追加一条")
    p_add.add_argument("--text", required=True)
    p_add.set_defaults(fn=cmd_add)
    p_rm = sub.add_parser("remove", parents=[common], help="删除一条")
    p_rm.add_argument("--index", type=int, required=True)
    p_rm.set_defaults(fn=cmd_remove)

    args = ap.parse_args(argv)
    args.root = os.path.abspath(args.root)
    if args.board is None:
        args.board = os.path.join(args.root, "_share", "讨论板.md")
    args.board = os.path.abspath(args.board)
    try:
        return args.fn(args)
    except PinError as e:
        print(f"[中止] {e}")
        return e.code


if __name__ == "__main__":
    sys.exit(main())
