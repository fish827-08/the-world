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
