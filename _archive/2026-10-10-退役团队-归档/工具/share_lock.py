#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""share_lock.py — `_share/` 写入锁（防并行写入 / 覆盖 / 双执行）

设计（2026-09-17，Owner 指令）:
  - 锁目录: `_share/.locks/`，每角色一个锁文件 `<slot>.lock`（内容 = 单行 JSON）
  - 获取 = 原子创建（O_CREAT|O_EXCL）；他人持活跃锁时拒绝
  - 过期 = TTL（默认 900s），过期锁视为无效（可被自动清理），防"崩死残留永久堵门"
  - 同机多会话: 文件系统实时互斥（硬保护）
  - 跨机（云端）: 锁文件随发言 commit 一并推送（best-effort 软保护）；
    最终防线仍是 git push 的 fast-forward 检查 + §3.1.1 "append 前重读板尾"

用法（写板 7 步流程中的 ⓪ 与 ⑦）:
    python tools/share_lock.py acquire --slot dev --task "追加 R111 帖"
    ...（拉取、重读板尾、追加、commit、push）...
    python tools/share_lock.py release --slot dev

其他:
    python tools/share_lock.py status [--json]     # 看谁在写 / 有无过期锁
    python tools/share_lock.py stale-clean [--yes] # 清过期锁（默认 dry-run）

兼容: 输出仅用中文与 ASCII（F-R15 教训：特殊符号在 GBK 控制台会炸）；
      退出码: 0 成功 / 1 用法或环境错误 / 2 他人活跃锁 / 3 竞态 / 4 重复获取
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from datetime import datetime, timedelta, timezone

# ---------- 兼容 GBK 控制台（F-R15）----------
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# ---------- 常量 ----------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
DEFAULT_LOCK_DIR = os.path.join(_ROOT, "_share", ".locks")
DEFAULT_TTL = 900  # 15 分钟
TZ8 = timezone(timedelta(hours=8))

SLOTS = {
    "owner": "[所有者]/天平",
    "dev": "[本地开发]/老工",
    "eval": "[内评]",
    "web": "[联网]/网评",
    "cloud": "[云端]",
    "collab": "[协作]",
}


def _now() -> datetime:
    return datetime.now(TZ8)


def _slot_label(slot: str) -> str:
    return SLOTS.get(slot, slot)


def _lock_path(lock_dir: str, slot: str) -> str:
    return os.path.join(lock_dir, f"{slot}.lock")


def _read_lock(path: str):
    """读锁文件内容（JSON）；坏文件返回 None（按存在但不可解析处理）。"""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _age_seconds(info, path: str) -> float:
    """锁年龄（秒）。优先 JSON 的 ts（跨机可信），后备文件 mtime。"""
    if info and isinstance(info.get("ts"), str):
        try:
            t = datetime.fromisoformat(info["ts"])
            if t.tzinfo is None:
                t = t.replace(tzinfo=TZ8)
            return max(0.0, (_now() - t).total_seconds())
        except Exception:
            pass
    try:
        return max(0.0, time.time() - os.path.getmtime(path))
    except OSError:
        return 0.0


def _iter_locks(lock_dir: str):
    """产出 (path, slot, info)。info 可能为 None（坏文件）。"""
    if not os.path.isdir(lock_dir):
        return
    for name in sorted(os.listdir(lock_dir)):
        if name.endswith(".lock"):
            p = os.path.join(lock_dir, name)
            yield p, name[: -len(".lock")], _read_lock(p)


def _fmt_entry(slot: str, info, age: float) -> str:
    label = (info or {}).get("label") or _slot_label(slot)
    task = (info or {}).get("task") or "—"
    host = (info or {}).get("host") or "?"
    return f"{slot} ({label}) age={age / 60:.1f}min task={task} host={host}"


def _lock_ttl(info) -> int:
    """锁自身的 TTL（写入时记录）；缺失/损坏时用默认值。"""
    try:
        ttl = int((info or {}).get("ttl"))
        return ttl if ttl > 0 else DEFAULT_TTL
    except Exception:
        return DEFAULT_TTL


def _retire(path: str, lock_dir: str) -> bool:
    """把锁文件挪出活动区（**不删除**）。

    F-R10 家族（2026-09-18 稳定复现）：本机 shim 的删除（回收站）异常
    （`SHFileOperationW 0x2`）⇒ 仓库内 Python `unlink/remove` fail-closed，
    且反复尝试会触发进程级强制终止。⇒ 本工具**一律不删除**，改用
    `rename` 挪移到 `<lock_dir>/_retired/`（已验证 rename 可用）。
    成功返回 True；失败返回 False（不抛异常）。
    """
    try:
        import time as _t
        retire_dir = os.path.join(lock_dir, "_retired")
        os.makedirs(retire_dir, exist_ok=True)
        dst = os.path.join(
            retire_dir,
            f"{os.path.basename(path)}.{_t.strftime('%Y%m%d-%H%M%S')}-{_t.time_ns() % 1000000:06d}")
        os.rename(path, dst)
        return True
    except OSError:
        return False


# ---------- 命令 ----------
def cmd_acquire(args) -> int:
    lock_dir = args.lock_dir
    os.makedirs(lock_dir, exist_ok=True)
    slot = args.slot

    # ① 扫描现有锁（过期判定用【锁自身的 TTL】，而非本次命令的 TTL）
    active, stale = [], []
    for p, s, info in _iter_locks(lock_dir):
        age = _age_seconds(info, p)
        (stale if age >= _lock_ttl(info) else active).append((p, s, info, age))

    # ② 自己已持有？
    mine = [e for e in active if e[1] == slot]
    if mine:
        p, s, info, age = mine[0]
        print(f"[拒绝] 你（{slot}）已持有锁: {p}")
        print(f"        {_fmt_entry(s, info, age)}")
        if getattr(args, "refresh", False):
            if _retire(p, lock_dir):
                print("        --refresh: 旧锁已挪移至 _retired/（不删除，F-R10 家族），继续重新获取")
            else:
                print("        --refresh 失败：无法挪移旧锁（可能被占用）")
                return 1
        else:
            print("        如需刷新 TTL: 先 release 再加 --refresh；或直接继续用（TTL 内有效）")
            return 4

    # ③ 他人活跃锁？拒绝
    others = [e for e in active if e[1] != slot]
    if others:
        for p, s, info, age in others:
            print(f"[拒绝] {_fmt_entry(s, info, age)}")
        print("        -> 对方写入未结束。等待其 release；")
        print("           （仅当确认对方崩死/走失时）用 stale-clean --force-slot 清理后重试")
        return 2

    # ④ 清理过期锁（含自己的与无主的；F-R10 家族：挪移而非删除）
    for p, s, info, age in stale:
        if _retire(p, lock_dir):
            print(f"[-] 已移出过期锁 {os.path.basename(p)} (slot={s}, age={age / 60:.1f}min)")

    # ⑤ 原子创建
    path = _lock_path(lock_dir, slot)
    info = {
        "slot": slot,
        "label": _slot_label(slot),
        "task": args.task or "",
        "ts": _now().isoformat(timespec="seconds"),
        "ttl": args.ttl,
        "pid": os.getpid(),
        "host": (os.environ.get("COMPUTERNAME") or socket.gethostname() or "?")[:24],
    }
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False)
            f.write("\n")
    except FileExistsError:
        print(f"[竞态] {path} 已存在（另一进程刚抢到）。请 status 查看后重试。")
        return 3
    except OSError as e:
        print(f"[错误] 创建锁失败: {e}")
        return 1

    expire_at = (_now() + timedelta(seconds=args.ttl)).strftime("%H:%M:%S")
    print(f"[OK] 已获取锁: {slot} ({_slot_label(slot)})")
    print(f"     task: {info['task'] or '—'}")
    print(f"     TTL : {args.ttl}s（至 {expire_at} 自动失效）")
    print(f"     锁文件: {path}")
    return 0


def cmd_release(args) -> int:
    path = _lock_path(args.lock_dir, args.slot)
    if not os.path.exists(path):
        print(f"[注意] 无锁可放（{path} 不存在）")
        return 1
    info = _read_lock(path)
    who = (info or {}).get("label") or _slot_label(args.slot)
    if _retire(path, args.lock_dir):
        print(f"[OK] 已释放锁: {args.slot} ({who})（挪移至 _retired/——本机不删除，F-R10 家族）")
    else:
        print("[注意] 释放未能物理移走锁文件（环境 rename 异常）")
        print("       ——不致命：锁将按 TTL 自动失效（acquire 侧自动接管陈旧锁）")
    return 0


def cmd_status(args) -> int:
    rows = []
    for p, s, info in _iter_locks(args.lock_dir):
        ttl = (info or {}).get("ttl") or DEFAULT_TTL
        try:
            ttl = int(ttl)
        except Exception:
            ttl = DEFAULT_TTL
        age = _age_seconds(info, p)
        rows.append({
            "slot": s,
            "label": (info or {}).get("label") or _slot_label(s),
            "task": (info or {}).get("task") or "",
            "age_s": round(age, 1),
            "ttl_s": ttl,
            "state": "active" if age < ttl else "stale",
            "ts": (info or {}).get("ts") or "?",
            "host": (info or {}).get("host") or "?",
        })
    if args.json:
        print(json.dumps({"lock_dir": args.lock_dir, "locks": rows},
                         ensure_ascii=False, indent=2))
        return 0
    if not rows:
        print(f"[空闲] 无锁（{args.lock_dir}）—— 可以安全写入")
        return 0
    print(f"锁目录: {args.lock_dir}")
    for r in rows:
        flag = "[活跃]" if r["state"] == "active" else "[过期]"
        print(f"  {flag} {r['slot']:<8s} {r['label']:<16s} "
              f"age={r['age_s'] / 60:5.1f}min / ttl={r['ttl_s']}s  "
              f"task={r['task'] or '—'}  host={r['host']}")
    n_active = sum(1 for r in rows if r["state"] == "active")
    n_stale = len(rows) - n_active
    print(f"共 {len(rows)} 把锁: 活跃 {n_active} / 过期 {n_stale}"
          + ("（可 stale-clean）" if n_stale else ""))
    return 0


def cmd_stale_clean(args) -> int:
    targets, kept = [], []
    for p, s, info in _iter_locks(args.lock_dir):
        ttl = (info or {}).get("ttl") or DEFAULT_TTL
        try:
            ttl = int(ttl)
        except Exception:
            ttl = DEFAULT_TTL
        age = _age_seconds(info, p)
        if age >= ttl:
            targets.append((p, s, age))
        else:
            kept.append((s, age, ttl))
    # --force-slot: 强制清指定 slot 的锁（含活跃；用于确认崩死的场景）
    if args.force_slot:
        fp = _lock_path(args.lock_dir, args.force_slot)
        if os.path.exists(fp):
            targets.append((fp, args.force_slot, _age_seconds(_read_lock(fp), fp)))
    if not targets:
        print("[OK] 无过期锁可清理")
        for s, age, ttl in kept:
            print(f"  [保留] {s}: age={age / 60:.1f}min < ttl={ttl}s（活跃）")
        return 0
    for p, s, age in targets:
        action = "已移出" if args.yes else "将移出"
        print(f"  [{action}] {os.path.basename(p)} (slot={s}, age={age / 60:.1f}min)")
        if args.yes:
            if not _retire(p, args.lock_dir):
                print("          移出失败（环境 rename 异常）——锁将按 TTL 自动失效")
    if not args.yes:
        print("（dry-run；确认无误后加 --yes 执行）")
    return 0


def main(argv=None) -> int:
    # `--lock-dir` 挂在每个子命令（parents 模式），允许写在子命令之后。
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--lock-dir", default=DEFAULT_LOCK_DIR,
                        help=f"锁目录（默认 {DEFAULT_LOCK_DIR}）")

    ap = argparse.ArgumentParser(
        prog="share_lock.py",
        description="the-world `_share/` 写入锁（防并行写入）")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_acq = sub.add_parser("acquire", parents=[common], help="获取写入锁")
    p_acq.add_argument("--slot", required=True,
                       help="角色槽位（owner/dev/eval/web/cloud/collab 或自定义）")
    p_acq.add_argument("--task", default="", help="本次写入事项（一句话）")
    p_acq.add_argument("--ttl", type=int, default=DEFAULT_TTL, help="锁有效期秒数（默认 900）")
    p_acq.add_argument("--refresh", action="store_true", help="若自己已持锁则先移除再获取")
    p_acq.set_defaults(fn=cmd_acquire)

    p_rel = sub.add_parser("release", parents=[common], help="释放写入锁")
    p_rel.add_argument("--slot", required=True)
    p_rel.set_defaults(fn=cmd_release)

    p_st = sub.add_parser("status", parents=[common], help="查看锁状态")
    p_st.add_argument("--json", action="store_true", help="机器可读输出")
    p_st.set_defaults(fn=cmd_status)

    p_sc = sub.add_parser("stale-clean", parents=[common], help="清理过期锁（默认 dry-run）")
    p_sc.add_argument("--yes", action="store_true", help="执行删除（默认只预览）")
    p_sc.add_argument("--force-slot", default=None,
                      help="强制清理指定 slot 的锁（含活跃；确认对方崩死时才用）")
    p_sc.set_defaults(fn=cmd_stale_clean)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
