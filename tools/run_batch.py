#!/usr/bin/env python3
"""跑批 + **同终端进度条**（外挂启动器，不改 `batch_runner.py` —— 规避 F-R9）。

用法（**项目根目录**）
----------------------
    python.exe tools/run_batch.py --preset cstep3gate --skip-existing

  · `--preset` 之后的所有参数**原样转发**给 `experiments/batch_runner.py`
  · 子进程输出 → `_rerun_logs/<preset>/_runner.log`（不再刷屏）
  · 本进程在**同一个终端**原地重绘：整体进度条 + 各 run 行 + runner 日志尾若干行
  · 子进程结束时打印：完成数 / 退出码 / 墙钟 / 日志路径

为什么是"外挂"而不是直接改 runner
--------------------------------
把进度条并进 `batch_runner.py` 是正解，但**批跑期间不得修改被 import 的代码**（F-R9）。
本脚本**只启动子进程 + 只读产物**，不 import 任何被批跑使用的模块 ⇒ **批跑期亦可用**。
（正式并入 runner 属 `[本地开发]` 待办；在其落地前，本脚本即为"实验运行时有进度条"的实现。）

与 `tools/batch_progress.py` 的关系
-----------------------------------
后者是**只读旁观**工具（另开终端看进度，或后台抓日志）；
本脚本让它**出现在跑批的那个终端里**（复用其渲染函数，单一渲染源）。
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from batch_progress import render, snapshot, target_ticks  # noqa: E402

CLEAR = "\033[2J\033[H"


def read_tail(path: str, n: int = 4) -> list[str]:
    """读日志尾部 n 行；文件不存在或读失败 ⇒ 返回空列表（不抛）。"""
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            lines = fh.read().splitlines()
        return [ln for ln in lines[-n:] if ln.strip()]
    except Exception:
        return []


def main() -> int:
    ap = argparse.ArgumentParser(
        add_help=False,
        description="跑批 + 同终端进度条（外挂，不改 batch_runner.py）",
    )
    ap.add_argument("--preset", required=True, help="预设名，如 cstep3gate")
    ap.add_argument("--interval", type=float, default=3.0, help="刷新间隔秒（默认 3）")
    ap.add_argument("--width", type=int, default=20, help="进度条宽度（默认 20 格）")
    ap.add_argument("--ticks", type=int, default=12000, help="回退目标 tick（能解析 preset 时以 preset 为准）")
    ap.add_argument("--tail", type=int, default=4, help="同时显示的 runner 日志尾行数（默认 4；0=不显示）")
    ap.add_argument("--plain", action="store_true", help="不清屏（滚动输出，便于重定向抓取）")
    ap.add_argument("-h", "--help", action="help")
    args, passthru = ap.parse_known_args()

    logdir = os.path.join(ROOT, "_rerun_logs", args.preset)
    os.makedirs(logdir, exist_ok=True)
    logpath = os.path.join(logdir, "_runner.log")

    cmd = [sys.executable, "-u", os.path.join("experiments", "batch_runner.py"),
           "--preset", args.preset, *passthru]
    print("[run_batch] 启动子进程: " + " ".join(cmd[2:]))
    print("[run_batch] runner 日志 → " + os.path.relpath(logpath, ROOT))
    print()

    t0 = time.time()
    ticks, logi = target_ticks(args.preset, args.ticks)
    clear = (not args.plain) and sys.stdout.isatty()
    if not clear:
        print("[run_batch] 非交互输出（或 --plain）⇒ 滚动模式；终端下运行时为原地刷新进度条\n")

    with open(logpath, "w", encoding="utf-8", errors="replace") as logf:
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=logf, stderr=subprocess.STDOUT)
        try:
            while True:
                rc = proc.poll()
                runs, meta = snapshot(args.preset, ticks, logi)
                if runs:
                    body = (f"[{time.strftime('%H:%M:%S')}] 已用 {(time.time() - t0) / 60:.1f} min"
                            f" · 目标 {ticks} tick/run\n"
                            + render(args.preset, runs, meta, ticks, args.width))
                else:
                    body = f"[{time.strftime('%H:%M:%S')}] 尚无 CSV（待批跑写出）"
                if args.tail:
                    tail = read_tail(logpath, args.tail)
                    if tail:
                        body += "\n── runner 日志尾 ──\n" + "\n".join(tail)
                if clear:
                    sys.stdout.write(CLEAR + body + "\n")
                elif rc is None:
                    print(body, flush=True)
                sys.stdout.flush()
                if rc is not None:
                    break
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n[run_batch] 收到 Ctrl+C —— 等待子进程退出（最多 10s）…")
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    print("[run_batch] 子进程未退出，已放弃等待（请手动检查）")

    runs, meta = snapshot(args.preset, ticks, logi)
    print()
    if runs:
        done = sum(1 for r in runs if r["done"])
        print(f"[run_batch] 结束：{done}/{len(runs)} 完成 · 退出码 {proc.returncode}"
              f" · 墙钟 {(time.time() - t0) / 60:.1f} min")
        print(render(args.preset, runs, meta, ticks, args.width))
    else:
        print(f"[run_batch] 结束：无产物 · 退出码 {proc.returncode}")
    print("[run_batch] 完整 runner 日志：" + os.path.relpath(logpath, ROOT))
    return proc.returncode or 0


if __name__ == "__main__":
    raise SystemExit(main())
