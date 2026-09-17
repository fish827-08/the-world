#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""board_check.py — `_share/` 板面体检 + 协作成本度量（协作线 T-2 / T-4）

两个子命令：
  check    一键体检：大小 / 署名格式 / 时钟 / 锁状态 / 引用断链
  metrics  协作成本度量：裁定密度 / 板增长速率 / 事故率 / 发帖节奏

设计原则：
  - 只读：本工具不修改任何文件（体检 = 只读，纪律同"扫描=只读"）
  - 自包含：仅依赖标准库；仓库根由 --root 指定（默认 = 本文件上三级，
    即 collab-toolkit/tools/board_check.py -> the-world/）
  - GBK 控制台安全：输出仅中文 + ASCII（F-R15 教训）

退出码（check）：0 全部正常 / 1 有警告 / 2 有错误（metrics 恒 0）
"""
from __future__ import annotations

import argparse
import json
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
PROJECT_START = "2026-09-12"  # _share/ 建立日（路线共识 R1 时代）

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ROOT = os.path.dirname(os.path.dirname(_HERE))

# _share/ 交流文件清单（体检对象）
SHARE_FILES = ["讨论板.md", "路线共识.md", "待办与交接.md", "路线图.md",
               "README.md", "花名册.md"]
SIZE_WARN_KB = 130
SIZE_ERR_KB = 150  # AGENT.md §3.3 硬上限

# 署名行：### [角色] · YYYY-MM-DD HH:MM（旧帖可能无 HH:MM，只警告不报错）
SIG_RE = re.compile(
    r"^###\s*\[([^\]]+)\]\s*·\s*(\d{4}-\d{2}-\d{2})(?:[ T]+(\d{1,2}:\d{2}))?")
SIG_FULL_RE = re.compile(
    r"^###\s*\[[^\]]+\]\s*·\s*\d{4}-\d{2}-\d{2}[ T]+\d{1,2}:\d{2}")
# 帖内小标题（如 "### 七、我的请求"）不是署名，不参与格式检查
SIG_LIKE_RE = re.compile(r"^###\s*\[[^\]]+\]")

# 反引号内路径抽取（引用断链检查）
BACKTICK_RE = re.compile(r"`([^`\n]+)`")
PATH_RE = re.compile(r"[A-Za-z0-9_./一-鿿（）()\-]+\.(?:md|py|csv|json)")
# R111 移动对照（_share/README.md §一.2）
MOVED_PREFIXES = {
    "_share/评估-": "docs/评估-EVAL/",
    "_share/评审-": "docs/评估-EVAL/",
    "_share/设计-": "docs/设计文档/",
    "_share/规格-": "docs/设计文档/",
    "_share/预注册-": "docs/预注册/",
    "_share/派工-": "docs/tasks/",
    "_share/战略定位-": "docs/决策与评审/",
    "_share/实况.md": "docs/决策与评审/",
    "_share/分支治理-": "docs/决策与评审/",
}


def _now() -> datetime:
    return datetime.now(TZ8)


def find_share_dir(root: str) -> str:
    return os.path.join(root, "_share")


# ---------------- 单项检查（返回 finding dict 列表） ----------------
def check_sizes(share_dir: str) -> list:
    out = []
    for name in SHARE_FILES:
        p = os.path.join(share_dir, name)
        if not os.path.isfile(p):
            out.append({"level": "warn", "item": "size",
                        "msg": f"{name} 不存在"})
            continue
        kb = os.path.getsize(p) / 1024.0
        if kb >= SIZE_ERR_KB:
            lvl, why = "error", f">= {SIZE_ERR_KB}KB 归档硬上限（§3.3）"
        elif kb >= SIZE_WARN_KB:
            lvl, why = "warn", f">= {SIZE_WARN_KB}KB 预警线"
        else:
            lvl, why = "ok", ""
        msg = f"{name}: {kb:.1f}KB" + (f" —— {why}" if why else "")
        out.append({"level": lvl, "item": "size", "msg": msg,
                    "file": name, "kb": round(kb, 1)})
    return out


def iter_posts(board_path: str):
    """产出板内署名帖：(行号, 角色, 日期, 时分或None, 完整签名是否带时分)。"""
    with open(board_path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            m = SIG_RE.match(line)
            if m:
                yield {"line": i, "role": m.group(1), "date": m.group(2),
                       "time": m.group(3), "raw": line.rstrip("\n")}


def check_signatures(board_path: str) -> list:
    out, posts = [], list(iter_posts(board_path))
    no_time = [p for p in posts if p["time"] is None]
    out.append({"level": "ok", "item": "sig",
                "msg": f"署名帖共 {len(posts)} 条，格式均可解析"})
    if no_time:
        sample = ", ".join(f"L{p['line']}" for p in no_time[:5])
        out.append({"level": "warn", "item": "sig",
                    "msg": f"{len(no_time)} 条署名缺 HH:MM（历史帖不补改，仅提示）: {sample}"})
    # 可疑：以 ### [x] 开头但完全不符合署名格式
    bad = []
    with open(board_path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if SIG_LIKE_RE.match(line) and not SIG_RE.match(line):
                bad.append(i)
    if bad:
        out.append({"level": "error", "item": "sig",
                    "msg": f"{len(bad)} 行疑似署名但格式不合规: "
                           + ", ".join(f"L{x}" for x in bad[:5])})
    return out


def check_clock(board_path: str) -> list:
    posts = list(iter_posts(board_path))
    timed = [p for p in posts if p["time"]]
    if not timed:
        return [{"level": "warn", "item": "clock", "msg": "板内无带时分的署名，无法估时钟"}]
    last = timed[-1]
    try:
        hh, mm = last["time"].split(":")
        dt = datetime.strptime(last["date"], "%Y-%m-%d").replace(
            hour=int(hh), minute=int(mm), tzinfo=TZ8)
    except ValueError:
        return [{"level": "warn", "item": "clock",
                 "msg": f"末帖时间不可解析: {last['raw'][:60]}"}]
    lag = (_now() - dt).total_seconds() / 60.0
    lvl = "ok"
    if lag < -5:
        lvl = "warn"  # 末帖时间在未来 —— 发帖机时钟偏快
    msg = (f"末帖: [{last['role']}] {last['date']} {last['time']}，"
           f"距本机 {lag:.0f} 分钟" + ("（未来时间——对端时钟偏快？）" if lag < -5 else ""))
    return [{"level": lvl, "item": "clock", "msg": msg,
             "lag_min": round(lag, 1)}]


def check_locks(share_dir: str) -> list:
    lock_dir = os.path.join(share_dir, ".locks")
    if not os.path.isdir(lock_dir):
        return [{"level": "ok", "item": "lock", "msg": "无锁目录（空闲）"}]
    out, n = [], 0
    for name in sorted(os.listdir(lock_dir)):
        if not name.endswith(".lock"):
            continue
        n += 1
        p = os.path.join(lock_dir, name)
        try:
            with open(p, encoding="utf-8") as f:
                info = json.load(f)
        except Exception:
            info = {}
        ttl = int(info.get("ttl") or 900)
        age = time.time() - os.path.getmtime(p)
        ts = info.get("ts")
        if ts:
            try:
                t = datetime.fromisoformat(ts)
                if t.tzinfo is None:
                    t = t.replace(tzinfo=TZ8)
                age = max(0.0, (_now() - t).total_seconds())
            except Exception:
                pass
        state = "active" if age < ttl else "stale"
        lvl = "warn" if state == "stale" else "ok"
        out.append({"level": lvl, "item": "lock",
                    "msg": f"{name}: {state} age={age / 60:.1f}min "
                           f"slot={info.get('slot', '?')} task={info.get('task', '—')}"})
    if n == 0:
        out.append({"level": "ok", "item": "lock", "msg": "锁目录空闲（可写入）"})
    return out


def check_refs(root: str, board_path: str, limit: int = 10) -> list:
    """引用断链检查：反引号内的仓库相对路径是否存在。

    路径解析顺序：仓库根 -> _share/（板帖常写板内相对路径）；两处都不存在才算断链。
    """
    share_dir = os.path.dirname(board_path)
    missing, checked = [], 0
    with open(board_path, encoding="utf-8") as f:
        text = f.read()
    for span in BACKTICK_RE.findall(text):
        if "://" in span:
            continue
        for m in PATH_RE.findall(span):
            path = m.split(":")[0].lstrip(".").rstrip(".,;)")
            if not path or path.startswith(("http", "www.")):
                continue
            checked += 1
            if (os.path.exists(os.path.join(root, path))
                    or os.path.exists(os.path.join(share_dir, path))):
                continue
            hint = ""
            for old, new in MOVED_PREFIXES.items():
                if path.startswith(old):
                    hint = f" -> 已移至 {new}（R111）"
                    break
            item = path + hint
            if item not in missing:
                missing.append(item)
    out = [{"level": "ok", "item": "ref", "msg": f"引用路径共查 {checked} 处"}]
    if missing:
        sample = "\n    ".join(missing[:limit])
        more = f"（另 {len(missing) - limit} 处略）" if len(missing) > limit else ""
        out.append({"level": "warn", "item": "ref",
                    "msg": f"{len(missing)} 处引用路径当前不存在"
                           f"（历史帖旧路径/数据文件属已知，R111 对照见 _share/README.md）:"
                           f"\n    {sample}{more}",
                    "missing": missing})
    return out


def check_pin(board_path: str, max_age_days: int | None = None) -> list:
    """T-6 置顶区体检（委托 board_pin.validate_pin；不可用时降级为标记检查）。"""
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from board_pin import split_board, validate_pin
    except Exception:
        with open(board_path, encoding="utf-8") as f:
            text = f.read()
        has_b = "<!-- PIN-BEGIN -->" in text
        has_e = "<!-- PIN-END -->" in text
        lvl = "ok" if (has_b and has_e) else "warn"
        return [{"level": lvl, "item": "pin",
                 "msg": f"置顶标记 BEGIN={has_b} END={has_e}（board_pin 不可用，仅粗检）"}]
    with open(board_path, encoding="utf-8") as f:
        text = f.read()
    try:
        pin, _ = split_board(text)
    except Exception as e:
        return [{"level": "error", "item": "pin", "msg": f"置顶块解析失败: {e}"}]
    return validate_pin(pin)


def check_remote_refs(root: str, remote: str, branch: str, net: bool) -> list:
    """F-R24 体检：remote-tracking refs 健康度。

    F-R24（2026-09-17 登记）：本机环境 `refs/remotes/` 写入被静默拒绝，
    `gitee/main` 等引用可能缺失或失真；落后检查须改用 `git ls-remote`。
    """
    ref = f"refs/remotes/{remote}/{branch}"
    r = subprocess.run(["git", "rev-parse", "--verify", ref],
                       cwd=root, capture_output=True, text=True)
    local = r.stdout.strip() if r.returncode == 0 else None
    out = []
    if local is None:
        out.append({"level": "warn", "item": "f-r24",
                    "msg": f"{ref} 不存在 —— F-R24 疑似生效中；"
                           f"落后检查一律用 `git ls-remote {remote} refs/heads/{branch}`，"
                           f"勿引用 {remote}/{branch} 形式"})
    else:
        out.append({"level": "ok", "item": "f-r24",
                    "msg": f"{ref} = {local[:8]}（存在；F-R24 下仍建议 ls-remote 复核）"})
    if net:
        r2 = subprocess.run(["git", "ls-remote", remote, f"refs/heads/{branch}"],
                            cwd=root, capture_output=True, text=True, timeout=60)
        remote_sha = r2.stdout.split()[0] if r2.returncode == 0 and r2.stdout.strip() else None
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                              capture_output=True, text=True).stdout.strip()
        if remote_sha is None:
            out.append({"level": "warn", "item": "f-r24",
                        "msg": f"ls-remote {remote} 失败（网络/凭据？）"})
        elif remote_sha == head:
            out.append({"level": "ok", "item": "f-r24",
                        "msg": f"ls-remote 对账: HEAD == {remote}/{branch} ({remote_sha[:8]})"})
        else:
            out.append({"level": "warn", "item": "f-r24",
                        "msg": f"HEAD({head[:8]}) != {remote}/{branch}({remote_sha[:8]})"
                               f" —— 有未推送或未拉取提交"})
        if local and remote_sha and local != remote_sha:
            out.append({"level": "warn", "item": "f-r24",
                        "msg": f"本地 {ref}({local[:8]}) 与远端实际({remote_sha[:8]}) 不符"
                               f" —— remote-tracking 失真，勿用于落后判断"})
    return out


# ---------------- metrics（T-4 数据采集） ----------------
def _git_lines(root: str, args: list) -> list:
    try:
        r = subprocess.run(["git"] + args, cwd=root, capture_output=True,
                           text=True, timeout=60, encoding="utf-8",
                           errors="replace")
        return r.stdout.splitlines() if r.returncode == 0 else []
    except Exception:
        return []


def collect_metrics(root: str) -> dict:
    share_dir = find_share_dir(root)
    board = os.path.join(share_dir, "讨论板.md")
    consensus = os.path.join(share_dir, "路线共识.md")

    posts = list(iter_posts(board)) if os.path.isfile(board) else []
    by_date: dict = {}
    for p in posts:
        by_date[p["date"]] = by_date.get(p["date"], 0) + 1

    days = max(1, (_now().date() - datetime.strptime(
        PROJECT_START, "%Y-%m-%d").date()).days + 1)

    r_numbers, f_numbers = set(), set()
    for path in (consensus, board):
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8") as f:
            text = f.read()
        r_numbers.update("R" + n for n in
                         re.findall(r"(?<![A-Za-z0-9-])R(\d{1,3})\b", text))
        f_numbers.update(re.findall(r"\bF-[A-Z]+\d+\b", text))

    board_kb = os.path.getsize(board) / 1024.0 if os.path.isfile(board) else 0.0
    commit_days: dict = {}
    for line in _git_lines(root, ["log", "--format=%ad", "--date=short", "--", "_share/"]):
        if line.strip():
            commit_days[line.strip()] = commit_days.get(line.strip(), 0) + 1

    return {
        "root": root,
        "as_of": _now().isoformat(timespec="seconds"),
        "days_since_start": days,
        "board_kb": round(board_kb, 1),
        "board_posts": len(posts),
        "board_posts_by_date": dict(sorted(by_date.items())),
        "ruling_count": len(r_numbers),
        "ruling_density_per_day": round(len(r_numbers) / days, 2),
        "defect_count": len(f_numbers),
        "defect_rate_per_day": round(len(f_numbers) / days, 2),
        "board_growth_kb_per_day": round(board_kb / days, 1),
        "share_commits_by_date": dict(sorted(commit_days.items())),
        "notes": {
            "ruling_density": "路线共识+讨论板去重 R 编号数 / 项目天数（裁定密度）",
            "defect_rate": "F-* 缺陷编号去重数 / 项目天数（事故率代理）",
            "board_growth": "当前讨论板 KB / 项目天数（含归档后重置，偏低估）",
            "latency": "单事项处理时延与各线等待时间暂无自动化采集，"
                       "见 docs/团队协作机制-v1.md §度量（人工抽样口径）",
        },
    }


# ---------------- 输出 ----------------
def print_findings(findings: list) -> tuple:
    icons = {"ok": "[OK]  ", "warn": "[警告]", "error": "[错误]"}
    n_warn = n_err = 0
    for f in findings:
        lvl = f["level"]
        if lvl == "warn":
            n_warn += 1
        elif lvl == "error":
            n_err += 1
        print(f"{icons.get(lvl, '[?]')} ({f['item']}) {f['msg']}")
    return n_warn, n_err


def cmd_check(args) -> int:
    root = os.path.abspath(args.root)
    share_dir = find_share_dir(root)
    board = os.path.join(share_dir, "讨论板.md")
    if not os.path.isfile(board):
        print(f"[错误] 讨论板不存在: {board}")
        return 2

    findings = []
    findings += check_sizes(share_dir)
    findings += check_signatures(board)
    findings += check_clock(board)
    findings += check_pin(board)
    findings += check_locks(share_dir)
    findings += check_remote_refs(root, args.remote, args.branch, args.net)
    if not args.no_refs:
        findings += check_refs(root, board)

    if args.json:
        print(json.dumps(findings, ensure_ascii=False, indent=2))
    else:
        print(f"板面体检 —— root={root}")
        print(f"时间: {_now().strftime('%Y-%m-%d %H:%M:%S')} (Asia/Shanghai)")
        print("-" * 60)
        n_warn, n_err = print_findings(findings)
        print("-" * 60)
        print(f"结论: {n_err} 错误 / {n_warn} 警告"
              + (" —— 建议处理后再发帖" if n_err else ""))
        if n_err:
            return 2
        return 1 if n_warn else 0
    n_err = sum(1 for f in findings if f["level"] == "error")
    n_warn = sum(1 for f in findings if f["level"] == "warn")
    return 2 if n_err else (1 if n_warn else 0)


def cmd_metrics(args) -> int:
    root = os.path.abspath(args.root)
    m = collect_metrics(root)
    if args.json:
        print(json.dumps(m, ensure_ascii=False, indent=2))
        return 0
    print(f"协作成本度量 —— as of {m['as_of']}")
    print("-" * 60)
    print(f"项目天数（自 {PROJECT_START}）: {m['days_since_start']} 天")
    print(f"当前讨论板: {m['board_kb']}KB / {m['board_posts']} 帖")
    print(f"裁定密度: {m['ruling_count']} 个 R 编号 -> "
          f"{m['ruling_density_per_day']} 个/天")
    print(f"事故率(代理): {m['defect_count']} 个 F-* 编号 -> "
          f"{m['defect_rate_per_day']} 个/天")
    print(f"板增长速率(粗估): {m['board_growth_kb_per_day']} KB/天")
    if m["board_posts_by_date"]:
        print("板帖按日: " + ", ".join(
            f"{d}:{n}" for d, n in m["board_posts_by_date"].items()))
    if m["share_commits_by_date"]:
        print("_share 提交按日: " + ", ".join(
            f"{d}:{n}" for d, n in m["share_commits_by_date"].items()))
    print("-" * 60)
    for k, v in m["notes"].items():
        print(f"  注[{k}]: {v}")
    return 0


def main(argv=None) -> int:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--root", default=DEFAULT_ROOT,
                        help="仓库根（默认自动探测 = collab-toolkit 的上一级）")
    ap = argparse.ArgumentParser(
        prog="board_check.py",
        description="the-world `_share/` 板面体检 + 协作成本度量（只读）")
    sub = ap.add_subparsers(dest="cmd")

    p_ck = sub.add_parser("check", parents=[common], help="板面体检（默认）")
    p_ck.add_argument("--json", action="store_true")
    p_ck.add_argument("--no-refs", action="store_true", help="跳过引用断链检查")
    p_ck.add_argument("--net", action="store_true",
                      help="联网对账（git ls-remote；F-R24 下推荐）")
    p_ck.add_argument("--remote", default="gitee")
    p_ck.add_argument("--branch", default="main")
    p_ck.set_defaults(fn=cmd_check)

    p_mt = sub.add_parser("metrics", parents=[common], help="协作成本度量（T-4）")
    p_mt.add_argument("--json", action="store_true")
    p_mt.set_defaults(fn=cmd_metrics)

    args = ap.parse_args(argv)
    if not getattr(args, "fn", None):
        # 无子命令 = check
        args = ap.parse_args(["check"] + (argv or []))
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
