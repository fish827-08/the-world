#!/usr/bin/env python
"""批跑进度条 —— R118 需求登记项的交付（只读，**批跑进行中也安全**）

背景
----
R118（fish 2026-09-18 指示）："下次实验可以做一个简单的进度条更好"。

设计约束（为什么是独立工具而不是直接改 runner）
----------------------------------------------
🔴 **F-R9：批跑期间不得改被 import 的代码**。α 批（`cstep3alpha`）正在跑，而它 import
`experiments/batch_runner.py` ⇒ 现在改 runner 会污染在跑的批。故本工具**独立**实现：
  · **只读**：不 import 引擎、不写 `_rerun_logs/`、**不删除任何文件**（F-R10 家族）；
  · 待批跑结束后，把同一份计算并入 `batch_runner` 的汇总行（同一口径、避免两套实现漂移）。

进度从哪来
----------
`_rerun_logs/<preset>/*.csv` 的**行数**。runner 每 `--log-interval`（默认 1000）tick 写一行
⇒ **这是不插桩前提下唯一可得的分辨率**（`*.progress.json` 只在 5000 的整数倍写，粒度更粗）。
⇒ 所以进度条的最小步长 = 1000 tick；不要误读成"卡住了"。

用法
----
    python tools/batch_progress.py --preset cstep3alpha            # 报一次
    python tools/batch_progress.py --preset cstep3alpha --watch    # 循环报（默认 60s）
    python tools/batch_progress.py --preset cstep3alpha --ticks 12000
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LOG_ROOT = REPO / "_rerun_logs"
BAR_FULL, BAR_EMPTY = "█", "░"


def read_lock() -> dict | None:
    """读批跑锁（只读；不存在则返回 None）。"""
    p = LOG_ROOT / ".batch_runner.lock"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _csv_rows(p: Path) -> int:
    """CSV 数据行数（不含表头）。小文件，直接数行。"""
    n = 0
    with p.open(encoding="utf-8", errors="replace") as fh:
        for _ in fh:
            n += 1
    return max(0, n - 1)


def target_ticks(preset: str, fallback: int) -> tuple[int, int]:
    """从 preset 定义解析出 (ticks_target, log_interval)。

    用 `batch_runner.preset_runs`（**CLI 与测试共用的同一函数**，纯展开、无副作用）。
    解析失败则回退 `fallback` / 1000 —— **显式回退并打印**，不静默。
    """
    ticks, logi = fallback, 1000
    try:
        sys.path.insert(0, str(REPO))
        from experiments.batch_runner import preset_runs  # noqa: PLC0415

        runs = preset_runs(preset, REPO)
        for r in runs:
            cmd = list(getattr(r, "cmd", []) or [])
            for i, tok in enumerate(cmd):
                if tok == "--ticks" and i + 1 < len(cmd):
                    ticks = int(cmd[i + 1])
                if tok == "--log-interval" and i + 1 < len(cmd):
                    logi = int(cmd[i + 1])
            break
    except Exception as exc:  # 显式回退（教训 2：不静默）
        print(f"  [note] 无法从 preset 解析 ticks，回退 {fallback}"
              f"（{type(exc).__name__}）", flush=True)
    return ticks, logi


def bar(frac: float, width: int = 20) -> str:
    frac = 0.0 if frac < 0 else (1.0 if frac > 1 else frac)
    n = int(round(frac * width))
    return BAR_FULL * n + BAR_EMPTY * (width - n)


def snapshot(preset: str, ticks: int, logi: int) -> tuple[list[dict], dict]:
    """采集一次进度快照（只读）。"""
    d = LOG_ROOT / preset
    lock = read_lock()
    started = None
    if lock and lock.get("started"):
        try:
            started = time.mktime(time.strptime(lock["started"], "%Y-%m-%d %H:%M:%S"))
        except Exception:
            started = None
    elapsed = (time.time() - started) / 60.0 if started else None

    runs = []
    for csv_p in sorted(d.glob("*.csv")):
        # 🔴 2026-09-19 修复（死循环根因）：批跑器结束时会在**同一目录**写 `_summary.csv`
        # （它自己的汇总表），而它**永远没有配套 `.summary.json`** ⇒ 若不过滤，`all(done)`
        # 永不成立 ⇒ `--watch` 死循环。批跑器的自有文件一律以 `_` 开头 ⇒ 过滤掉。
        if csv_p.name.startswith("_"):
            continue
        name = csv_p.stem
        try:
            tick = _csv_rows(csv_p) * logi
        except Exception:
            tick = 0
        done = (d / f"{name}.summary.json").exists()
        rate = (tick / elapsed) if elapsed and elapsed > 0.05 else None
        eta = ((ticks - tick) / rate) if (rate and rate > 0 and tick < ticks) else None
        runs.append(dict(name=name, tick=min(tick, ticks), target=ticks,
                         done=done, rate=rate, eta=eta))
    return runs, dict(lock=lock, elapsed=elapsed)


def render(preset: str, runs: list[dict], meta: dict, ticks: int, width: int) -> str:
    total = len(runs)
    done = sum(1 for r in runs if r["done"])
    tick_sum = sum(r["tick"] for r in runs)
    frac = (tick_sum / (total * ticks)) if total and ticks else 0.0
    # 整体 ETA = 未完成 run 的最大 ETA（批跑墙钟 = 最慢的那个）
    etas = [r["eta"] for r in runs if (not r["done"]) and r["eta"] is not None]
    eta_txt = f"{max(etas):.1f} min" if etas else "—"
    el = meta.get("elapsed")
    el_txt = f"{el:.1f} min" if el else "—"
    lock = meta.get("lock") or {}
    live = "运行中" if lock.get("preset") == preset else "（无活动锁）"

    out = []
    out.append(f"{preset}  [{bar(frac, width)}] {frac * 100:5.1f}%   "
               f"完成 {done}/{total}   {live}   已用 {el_txt}   ETA ≈ {eta_txt}")
    for r in runs:
        mark = "✅" if r["done"] else "  "
        f = r["tick"] / r["target"] if r["target"] else 0.0
        rt = f"{r['rate']:6.0f} t/min" if r["rate"] else "     —    "
        et = f"{r['eta']:5.1f} min" if (r["eta"] is not None and not r["done"]) else "    —    "
        out.append(f" {mark} {r['name']:10s} [{bar(f, width)}] {f * 100:4.0f}%  "
                   f"{r['tick']:>6d}/{r['target']}  {rt}  ETA {et}")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description="批跑进度条（只读，批跑中可用）")
    ap.add_argument("--preset", required=True, help="预设名，如 cstep3alpha")
    ap.add_argument("--ticks", type=int, default=12000,
                    help="回退用的目标 tick（默认 12000）；能解析 preset 时以 preset 为准")
    ap.add_argument("--width", type=int, default=20, help="进度条宽度（默认 20 格）")
    ap.add_argument("--watch", action="store_true", help="循环刷新")
    ap.add_argument("--interval", type=float, default=60.0, help="刷新间隔秒（--watch）")
    ap.add_argument("--max-minutes", type=float, default=240.0,
                    help="--watch 的兜底上限（分钟，默认 240；0=不限）—— 防「永不满足」型死循环")
    args = ap.parse_args()
    _t0 = time.time()

    ticks, logi = target_ticks(args.preset, args.ticks)
    while True:
        runs, meta = snapshot(args.preset, ticks, logi)
        if not runs:
            print(f"[{time.strftime('%H:%M:%S')}] {args.preset}: 尚无 CSV"
                  f"（批跑未开始或目录不对）", flush=True)
            return 0
        # 全部带 flush=True：管道/重定向下 stdout 有缓冲，不 flush 会"看起来没有任何输出"
        print(f"[{time.strftime('%H:%M:%S')}]  目标 {ticks} tick/run"
              f" · 分辨率 {logi} tick/行", flush=True)
        print(render(args.preset, runs, meta, ticks, args.width), flush=True)
        if not args.watch:
            return 0
        if all(r["done"] for r in runs):
            print("ALL_DONE", flush=True)
            return 0
        if args.max_minutes and (time.time() - _t0) > args.max_minutes * 60:
            print(f"TIMEOUT({args.max_minutes}min) —— 未全部完成，退出（可用 --max-minutes 调整）",
                  flush=True)
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
