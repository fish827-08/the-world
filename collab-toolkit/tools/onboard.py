#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""onboard.py — 新会话开场提示词生成器（协作线 T-5，防 P5 角色上下文重载成本）

用法：
  python onboard.py --role collab            # 生成该角色的开场提示词（打印到 stdout）
  python onboard.py --role list              # 列出全部角色
  python onboard.py --role dev --net         # 附 ls-remote 基线对账（需网络）

数据源 = 花名册 + 各线 CHARTER 的人工维护表（本文件内 ROLE_REGISTRY）。
角色表更新时同步改这里（README 有对照）。

退出码：0 正常 / 1 用法错误 / 2 未知角色
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

# 通用必读（全员；HISTORY.md 为 R115 指定的新人第一份文档）
COMMON_READS = [
    ("HISTORY.md", "项目发展史 + 代号百科（R115：新人入门第一份文档）"),
    ("AGENT.md", "协作章程（零号规则 §3.1.1 / 写入锁 §3.1.2 / Pre-Flight §十二）"),
    ("_share/README.md", "_share 使用说明与 R111 路径对照"),
    ("_share/讨论板.md", "尾部最新讨论（开工前必读，§3.3-5）"),
    ("_share/路线共识.md", "R 编号裁定记录"),
    ("_share/待办与交接.md", "当前任务与交接"),
    ("_share/花名册.md", "角色/花名/时段"),
]

# 角色注册表：署名 / 花名 / 锁槽位 / 职责 / 时段 / 加读 / 第一步
ROLE_REGISTRY = {
    "owner": {
        "署名": "[所有者]", "花名": "天平", "锁槽位": "owner",
        "职责": "裁定、派工、文档同步、对外事务、协作流程",
        "时段": "随时",
        "加读": [("_share/路线图.md", "路线唯一索引（裁决状态仅你可改）")],
        "第一步": "读讨论板尾部全部未决请求，按优先级裁定或派工",
    },
    "dev": {
        "署名": "[本地开发]", "花名": "老工", "锁槽位": "dev",
        "职责": "引擎/实验/工具代码、本地跑批、修复实施",
        "时段": "12:00–24:00（R29）",
        "加读": [("AGENTS.md", "工程口径（构建/基因/约束）"),
                 ("MODULES.md", "模块说明"),
                 ("PROGRESS.md", "进度"),
                 ("docs/tasks/", "派工包")],
        "第一步": "查讨论板与待办，确认无跑批冲突；批跑期间不动被 import 的代码（F-R9）",
    },
    "eval": {
        "署名": "[内评]", "花名": "（待取）", "锁槽位": "eval",
        "职责": "独立复核 / 预注册审查 / 六门 / 缺陷核查",
        "时段": "按需（R99）",
        "加读": [("_eval/", "内评工作区"), ("docs/评估-EVAL/", "评估文档")],
        "第一步": "确认评估触发条件（R99：关键节点/路线偏移/外部输入），默认静默",
    },
    "web": {
        "署名": "[联网]", "花名": "网评", "锁槽位": "web",
        "职责": "联网评估 / 文献核查 / 外部信息",
        "时段": "按需（R99）",
        "加读": [("_web_eval/", "联网评估工作区")],
        "第一步": "确认任务单；证据必须带 [文献]/[网络] 标签 + 出处",
    },
    "cloud": {
        "署名": "[云端]", "花名": "云端两兄弟", "锁槽位": "cloud",
        "职责": "长程跑批 / 云端执行",
        "时段": "00:00–12:00（R29）",
        "加读": [("docs/tasks/", "派工包")],
        "第一步": "确认单实例锁与续跑快照；无法 push 时帖尾标注'待推送——请[本地开发]代推'",
    },
    "collab": {
        "署名": "[协作]", "花名": "板桥", "锁槽位": "collab",
        "职责": "协作机制与工具优化（分支线，不碰引擎与科学判据）",
        "时段": "按需",
        "加读": [("docs/tasks/TASKS-COLLAB.md", "协作线启动包"),
                 ("collab-toolkit/", "本线工具与文档"),
                 ("tools/share_lock.py", "写入锁 v1.0")],
        "第一步": "讨论板发自我介绍帖（认领 T-1~T-5），机制变更走'设计稿→所有者核→公告'",
    },
}


def git_baseline(root: str, net: bool) -> str:
    def g(*a):
        r = subprocess.run(["git"] + list(a), cwd=root, capture_output=True,
                           text=True, encoding="utf-8", errors="replace")
        return r.stdout.strip() if r.returncode == 0 else "?"
    head = g("rev-parse", "--short", "HEAD")
    line = f"本地 HEAD = `{head}`"
    if net:
        remote = g("ls-remote", "gitee", "refs/heads/main")
        sha = remote.split()[0][:8] if remote and remote != "?" else "?"
        line += (f"；gitee/main = `{sha}`"
                 + ("（一致）" if sha != "?" and head == sha else "（不一致——先 pull）"))
    else:
        line += "（未联网对账；发言前务必 pull）"
    return line


# ---------------- 交接包（T-7 P1：上次进展 / 未完成 / 今天要读的三帖） ----------------
def recent_daily_logs(memory_dir: str, n: int = 3) -> list:
    """最近 n 天的日志文件路径（新→旧）。"""
    import glob
    files = sorted(glob.glob(os.path.join(memory_dir, "20[0-9][0-9]-[0-9][0-9]-[0-9][0-9].md")),
                   reverse=True)
    return files[:n]


EVIDENCE_TAGS = ("实测", "代码", "文献", "网络", "文档", "推断", "待核")


def _section_role(line: str) -> str | None:
    """从日志段落头 `## [协作·板桥] …` 提取角色 token（如 '协作·板桥'）。"""
    m = re.match(r"^##\s*\[([^\]]+)\]", line)
    return m.group(1) if m else None


def role_recent_lines(memory_dir: str, role_tag: str, max_lines: int = 3) -> list:
    """从最近日志里抓本角色最新一段的要点行（`- ` / `| ` 开头），取末尾 max_lines 条。"""
    base = role_tag.strip("[]")  # '[协作]' -> '协作'
    for path in recent_daily_logs(memory_dir, 3):
        try:
            with open(path, encoding="utf-8") as f:
                lines = f.read().splitlines()
        except OSError:
            continue
        # 找本角色最新的段落头（token 以角色名开头，兼容 '[协作·板桥]' 变体）
        start = None
        for i, line in enumerate(lines):
            token = _section_role(line)
            if token and token.split("·")[0].strip() == base:
                start = i
        if start is None:
            continue
        pts = [ln.lstrip("-| ").strip(" `") for ln in lines[start + 1:]
               if ln.strip().startswith(("- ", "| "))
               and not ln.strip().startswith(("|--", "| ---"))]
        pts = [p for p in pts if p][:max_lines]
        if pts:
            return pts
    return []


def role_open_items(board_path: str, role_sig: str, max_items: int = 2) -> list:
    """从讨论板"未落定项"表抓责任方含本角色的行（决策点 + 未决）。"""
    if not os.path.isfile(board_path):
        return []
    out = []
    with open(board_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip().startswith("|"):
                continue
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) < 5 or cells[0] in ("#", "---") or "决策点" in cells[0]:
                continue
            owner_cell = cells[-1]
            if role_sig in owner_cell or "板桥" in owner_cell and role_sig == "[协作]":
                out.append(f"{cells[0]} {cells[1]} —— 未决: {cells[3]}")
    return out[:max_items]


def latest_posts(board_path: str, n: int = 3) -> list:
    """板尾最近 n 帖的署名行（作为"今天要读的三帖"）。

    排除证据小节（`### [实测] …` 等）——它们是帖内小标题，不是帖子。
    """
    if not os.path.isfile(board_path):
        return []
    sig = re.compile(r"^#{2,3}\s*\[([^\]]+)\]")
    hits = []
    with open(board_path, encoding="utf-8") as f:
        for line in f:
            m = sig.match(line)
            if m and m.group(1) not in EVIDENCE_TAGS:
                hits.append(line.strip()[:80])
    return hits[-n:]


def handover_section(role_key: str, root: str) -> str:
    """交接包段（T-7 P1）。解析失败时给出占位符，不阻塞开场词生成。"""
    r = ROLE_REGISTRY[role_key]
    role_tag = r["署名"]
    memory_dir = os.path.join(root, ".workbuddy", "memory")
    board = os.path.join(root, "_share", "讨论板.md")
    lines = ["", "## 交接包（T-7 P1：机器采集，人工复核）"]
    recent = role_recent_lines(memory_dir, role_tag)
    lines.append("### 上次进展（最近日志要点，≤3 条）")
    lines += [f"- {p}" for p in recent] or ["- （未在最近 3 天日志找到本角色段落——人工补）"]
    open_items = role_open_items(board, role_tag)
    lines.append("### 未完成（未落定项表中本角色责任方，≤2 条）")
    lines += [f"- {p}" for p in open_items] or ["- （板上未落定项中未找到本角色条目）"]
    lines.append("### 今天要读的三帖（板尾最近 3 帖）")
    lines += [f"- {p}" for p in latest_posts(board, 3)] or ["- （板为空）"]
    lines.append("> ⚠️ 以上为机器采集的起点，**以板尾/日志原文为准**；"
                 "结束会话前记得留『已完成/未完成/阻塞』三行。")
    return "\n".join(lines)


def render(role_key: str, root: str, net: bool) -> str:
    r = ROLE_REGISTRY[role_key]
    alias = r["花名"].strip("（）()")
    now = datetime.now(TZ8).strftime("%Y-%m-%d %H:%M")
    lines = [
        f"# 新会话开场提示词 —— {r['署名']}（{alias}）",
        f"> 生成: {now}（Asia/Shanghai）by collab-toolkit/tools/onboard.py",
        "",
        f"你是 the-world 项目组的 **{r['署名']}**（花名 {alias}）。",
        f"- 职责: {r['职责']}",
        f"- 分工时段: {r['时段']}",
        f"- 板面署名格式: `### {r['署名']} · YYYY-MM-DD HH:MM`（花名只作签名档）",
        f"- 锁槽位: `--slot {r['锁槽位']}`",
        "",
        "## 基线",
        f"- {git_baseline(root, net)}",
        "- 仓库: the-world（主）+ the-world-data（独立数据仓）；权威远端 = `gitee/main`",
        "",
        "## 必读清单（按序）",
    ]
    for path, why in COMMON_READS + r["加读"]:
        lines.append(f"1. `{path}` —— {why}")
    lines += [
        "",
        "## 开工第一步",
        f"1. {r['第一步']}",
        "2. 发言一律走零号规则（可用 `collab-toolkit/tools/board_post.py` 一键执行），"
        "未 push 不算发言",
        "3. 只追加、不改他人文件；证据带标签（[实测]/[代码]/[文献]/[网络]/[文档]/[推断]/[待核]）",
    ]
    # T-7 P1：交接包（上次进展 / 未完成 / 今天要读的三帖）
    try:
        lines.append(handover_section(role_key, root))
    except Exception as e:  # 交接包是辅助，绝不能阻塞开场词
        lines.append(f"\n## 交接包\n- （采集失败：{e} —— 请人工读板尾与日志）")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="onboard.py",
                                 description="新会话开场提示词生成器（T-5）")
    ap.add_argument("--role", required=True,
                    help="角色键（owner/dev/eval/web/cloud/collab）或 list")
    ap.add_argument("--root", default=DEFAULT_ROOT, help="仓库根（默认自动探测）")
    ap.add_argument("--net", action="store_true", help="附 ls-remote 基线对账")
    args = ap.parse_args(argv)

    if args.role == "list":
        print("角色注册表（同步自 _share/花名册.md）:")
        for k, r in ROLE_REGISTRY.items():
            print(f"  {k:<7s} {r['署名']:<10s} 花名={r['花名']:<8s} "
                  f"锁槽={r['锁槽位']:<7s} 时段={r['时段']}")
        return 0
    if args.role not in ROLE_REGISTRY:
        print(f"[错误] 未知角色 '{args.role}'；用 --role list 查看")
        return 2
    sys.stdout.write(render(args.role, os.path.abspath(args.root), args.net))
    return 0


if __name__ == "__main__":
    sys.exit(main())
