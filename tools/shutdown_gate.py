#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""关机闸门（R222）：**确认"本机可以关机了吗"**，默认只检查、不关机。

背景
----
fish 2026-09-27 02:0x 指示：本地任务全部完成后关机，**关机前必须确保所有工作已上传远程**。
本项目有两个仓库（主仓 + **嵌套数据仓**，各自独立 remote）⇒ 人肉核对必漏 ⇒ 做成闸门。

检查项（全部 ✅ 才允许关机）
----------------------------
1. **主仓**：工作区干净（无未提交/未跟踪的非忽略改动）且 `main` 本地 == `gitee/main`
2. **数据仓**：同上（实验结果**不进主仓**，`.gitignore` 已排除 ⇒ 只在数据仓）
3. **无重负载 python 进程**（CPU > `--busy-cpu` 秒）⇒ 有批跑/测试在跑就不许关
4. **可选** `--require-summary`：指定的 run 产物必须已落盘（防"跑到一半就没了"）

用法
----
    python.exe tools/shutdown_gate.py                      # 只检查（默认）
    python.exe tools/shutdown_gate.py --require-summary results/s1_diag
    python.exe tools/shutdown_gate.py --shutdown           # 全绿才真的关机（带 90s 宽限）
    shutdown /a                                            # 后悔药：取消关机

⚠️ `--shutdown` 会真的关机：给出 **90 秒宽限**，期间 `shutdown /a` 可取消。
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "the-world-data"
REMOTE = "gitee"
BRANCH = "main"


def run(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(args, cwd=str(cwd), capture_output=True, text=True,
                          encoding="utf-8", errors="replace")


def check_repo(name: str, path: Path, verbose: bool) -> tuple[bool, list[str]]:
    msgs: list[str] = []
    ok = True
    if not (path / ".git").exists():
        return True, [f"{name}: 无 .git（跳过）"]

    st = run(["git", "status", "--short"], path)
    # `_trash_local/` 是本地草稿区（不入库、不共享）⇒ 不该阻塞关机
    dirty = [l for l in (st.stdout or "").splitlines()
             if l.strip() and "_trash_local" not in l]
    # 只统计"非忽略"的改动：git status --short 已不含被忽略文件
    if dirty:
        ok = False
        msgs.append(f"❌ {name}: 工作区有 {len(dirty)} 项未提交/未跟踪")
        if verbose:
            for l in dirty[:10]:
                msgs.append(f"     {l}")
    else:
        msgs.append(f"✅ {name}: 工作区干净")

    loc = run(["git", "rev-parse", "--short", BRANCH], path).stdout.strip()
    # remote 名按仓库实测取：主仓 = `gitee`，数据仓只有 `origin`
    remotes = [x for x in (run(["git", "remote"], path).stdout or "").split() if x.strip()]
    rname = REMOTE if REMOTE in remotes else (remotes[0] if remotes else "")
    if not rname:
        msgs.append(f"⚠️ {name}: 无 remote ⇒ 不能确认已上传")
        return False, msgs
    out = run(["git", "ls-remote", rname, BRANCH], path).stdout.strip().split()
    rem = out[0][:len(loc)] if out else ""
    if not rem:
        msgs.append(f"⚠️ {name}: 查不到远端（网络？）⇒ 不能确认已上传")
        ok = False
    elif loc == rem:
        msgs.append(f"✅ {name}: 已上传（local={loc} == {REMOTE}/{BRANCH}）")
    else:
        ok = False
        msgs.append(f"❌ {name}: 未上传（local={loc} vs {REMOTE}={rem}）")
    return ok, msgs


def busy_python(min_cpu: float) -> list[str]:
    """返回重负载 python 进程摘要（Windows 用 tasklist/wmic 不可用 ⇒ 用 PowerShell）。"""
    ps = ["powershell", "-NoProfile", "-Command",
          "Get-Process python -ErrorAction SilentlyContinue | "
          "Where-Object { $_.CPU -gt " + str(min_cpu) + " } | "
          "ForEach-Object { $_.Id.ToString() + ':' + [math]::Round($_.CPU,0) }"]
    r = subprocess.run(ps, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return [x for x in (r.stdout or "").split() if x.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description="关机闸门：确认本机可否安全关机")
    ap.add_argument("--busy-cpu", type=float, default=120.0,
                    help="python 进程 CPU 秒数超过此值视为'在跑'（默认 120）")
    ap.add_argument("--require-summary", default=None,
                    help="要求该目录下每个 *.csv 都有对应 *.summary.json（防半截产物）")
    ap.add_argument("--shutdown", action="store_true", help="全绿才真的关机（90s 宽限）")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()

    print("=== 关机闸门（R222） ===")
    all_ok = True

    ok1, m1 = check_repo("主仓", ROOT, a.verbose)
    all_ok &= ok1
    print("\n".join(m1))

    ok2, m2 = check_repo("数据仓", DATA, a.verbose)
    all_ok &= ok2
    print("\n".join(m2))

    busy = busy_python(a.busy_cpu)
    if busy:
        all_ok = False
        print(f"❌ 进程：{len(busy)} 个重负载 python（{', '.join(busy[:6])}…）⇒ 有任务在跑")
    else:
        print("✅ 进程：无重负载 python")

    if a.require_summary:
        d = ROOT / a.require_summary
        csvs = sorted(d.glob("*.csv")) if d.is_dir() else []
        miss = [c.name for c in csvs if not (d / (c.stem + ".summary.json")).exists()]
        if miss:
            all_ok = False
            print(f"❌ 产物：{len(miss)} 个 run 缺 summary（例：{miss[:4]}）⇒ 未跑完")
        else:
            print(f"✅ 产物：{len(csvs)} 个 run 均有 summary")

    print()
    if not all_ok:
        print("🛑 **不允许关机** —— 上面有 ❌ 项未通过。修完再跑本脚本。")
        return 1

    print("🟢 **全部通过，可以关机。**")
    if not a.shutdown:
        print("   （只检查、未关机。要关机请加 `--shutdown`；后悔药用 `shutdown /a`）")
        return 0

    print("   ⇒ 90 秒后关机；取消请执行 `shutdown /a`")
    r = subprocess.run(["shutdown", "/s", "/t", "90", "/c", "the-world 收工关机（R222）"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    print("   shutdown rc =", r.returncode, (r.stdout or r.stderr or "").strip()[:200])
    return 0 if r.returncode == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
