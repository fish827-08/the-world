#!/usr/bin/env python3
"""viz v0 导出器：快照 → 数据包（帧序列 + 实体缓冲 + meta.json）。

接口契约：viz/CONTRACT.md（🔒 v0）。只读引擎公开状态；不修改 simulation/。
帧 k ↔ tick = start + k*stride（含 t0 首帧）；复跑确定性由快照续跑保证。
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from simulation.sphere_engine import SphereEngine  # noqa: E402

CHANNEL_KEYS = ("resource", "energy", "fruit")
CHANNEL_LABELS = {"resource": "资源场", "energy": "种群能量密度", "fruit": "果实场"}
ENTITY_COLUMNS = ("flat", "sub_r", "sub_c", "energy", "age", "generation", "mode")
Q_LO, Q_HI = 0.005, 0.995
_SUBSAMPLE_TARGET = 200_000


class DatapackError(RuntimeError):
    """数据包无法按契约 v0 生成（快照损坏 / 源不可用）。"""


def _subsample(a) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64).ravel()
    if a.size <= _SUBSAMPLE_TARGET:
        return a
    return a[:: int(np.ceil(a.size / _SUBSAMPLE_TARGET))]


def _check_flat(flat: np.ndarray, n_cells: int) -> None:
    if flat.size and (int(flat.min()) < 0 or int(flat.max()) >= n_cells):
        raise DatapackError(
            f"实体 flat 越界 [{int(flat.min())}, {int(flat.max())}]，合法 [0, {n_cells}) ⇒ 快照/引擎状态已损坏"
        )


def _check_sub(sr: np.ndarray, sc: np.ndarray, rows: int, cols: int, subdiv: int) -> None:
    """契约 v0 修订1：sub_r/sub_c 为亚格绝对坐标，值域 [0, rows*subdiv) × [0, cols*subdiv)。"""
    if sr.size and (int(sr.min()) < 0 or int(sr.max()) >= rows * subdiv):
        raise DatapackError(
            f"实体 sub_r 越界 [{int(sr.min())}, {int(sr.max())}]，合法 [0, {rows * subdiv})（subdiv={subdiv}）"
        )
    if sc.size and (int(sc.min()) < 0 or int(sc.max()) >= cols * subdiv):
        raise DatapackError(
            f"实体 sub_c 越界 [{int(sc.min())}, {int(sc.max())}]，合法 [0, {cols * subdiv})（subdiv={subdiv}）"
        )


def _source_grid(eng: SphereEngine, key: str, n: int):
    """取通道源栅格（拷贝，防与引擎活网格别名）；源不可用（缺键/形状不符）返回 None。"""
    if key == "energy":
        P = len(eng._id)
        flat = eng._flat[:P].astype(np.int64)
        _check_flat(flat, n)
        return np.bincount(flat, weights=eng._energy[:P], minlength=n).astype(np.float64)
    if key == "resource":
        a = getattr(getattr(eng, "resources", None), "_grid", None)
    else:  # fruit
        a = getattr(eng, "_fruit_grid", None)
    if a is None or np.asarray(a).shape != (n,):
        return None
    return np.asarray(a).astype(np.float64)  # astype 默认 copy ⇒ 帧间独立，勿改 asarray


def capture_grids(eng: SphereEngine, channels) -> dict:
    n = eng.config.world.rows * eng.config.world.cols
    out = {}
    for key in channels:
        a = _source_grid(eng, key, n)
        if a is not None:
            out[key] = a
    return out


def capture_entities(eng: SphereEngine) -> np.ndarray:
    P = len(eng._id)
    rows, n_cols = eng.config.world.rows, eng.config.world.cols
    _check_flat(eng._flat[:P], rows * n_cols)
    _check_sub(eng._sub_r[:P], eng._sub_c[:P], rows, n_cols, int(eng.config.subpos.subdiv))
    arrs = [
        eng._flat[:P],
        eng._sub_r[:P],
        eng._sub_c[:P],
        eng._energy[:P],
        eng._age[:P],
        eng._generation[:P],
        eng._mode[:P],
    ]
    return np.stack([np.asarray(c, dtype=np.float32) for c in arrs], axis=1)  # (P, 7)


def _scale_from_samples(samples: np.ndarray) -> dict:
    s = samples[np.isfinite(samples)]
    if s.size == 0:
        lo, hi = 0.0, 1.0
    else:
        lo, hi = float(np.quantile(s, Q_LO)), float(np.quantile(s, Q_HI))
        if hi <= lo:
            hi = lo + 1.0
    return {
        "mode": "quantile",
        "p_lo": lo,
        "p_hi": hi,
        "min": float(s.min()) if s.size else 0.0,
        "max": float(s.max()) if s.size else 0.0,
        "q_lo": Q_LO,
        "q_hi": Q_HI,
    }


def quantize(frame: np.ndarray, scale: dict) -> np.ndarray:
    v = (frame - scale["p_lo"]) / (scale["p_hi"] - scale["p_lo"])
    return np.rint(np.clip(v, 0.0, 1.0) * 255.0).astype(np.uint8)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ticks", type=int, default=0, help="续跑 tick 数（0=仅快照当前帧）")
    ap.add_argument("--stride", type=int, default=1, help="采样间隔 tick/帧")
    ap.add_argument("--channels", default="resource,energy")
    ap.add_argument("--series", default=None, help="可选：随包复制的 series.csv 源路径")
    ap.add_argument("--force-python", action="store_true", help="强制 Python 引擎路径（默认随快照）")
    ap.add_argument("--mem-guard-mb", type=int, default=512)
    args = ap.parse_args(argv)

    channels = [c.strip() for c in args.channels.split(",") if c.strip()]
    bad = [c for c in channels if c not in CHANNEL_KEYS]
    if bad or not channels:
        print(f"[viz-export] 未知通道 {bad}；可选 {list(CHANNEL_KEYS)}", file=sys.stderr)
        return 2
    if args.ticks < 0 or args.stride < 1:
        print("[viz-export] 要求 ticks>=0 且 stride>=1", file=sys.stderr)
        return 2

    eng = SphereEngine.load_snapshot(args.snapshot)
    if args.force_python and eng._use_sim_core:
        eng._use_sim_core = False
        print("[viz-export] 已强制 Python 引擎路径（显示用途）")

    rows, cols = eng.config.world.rows, eng.config.world.cols
    n_cells = rows * cols
    t0 = int(eng._tick)
    n_frames = args.ticks // args.stride + 1
    est = n_frames * n_cells * 8 * len(channels)
    if est > args.mem_guard_mb * 1024 * 1024:
        print(
            f"[viz-export] 帧缓冲预估 {est / 1e6:.0f} MB 超限（{args.mem_guard_mb} MB）"
            " ⇒ 调大 --stride 或 --mem-guard-mb",
            file=sys.stderr,
        )
        return 2

    try:
        first_grids = capture_grids(eng, channels)
        eff_channels = [c for c in channels if c in first_grids]
        missing = [c for c in channels if c not in first_grids]
        if missing:
            print(
                f"[viz-export] 通道不可用已跳过 {missing}（源缺键/形状不符）；实际导出 {eff_channels}",
                file=sys.stderr,
            )
        if not eff_channels:
            print("[viz-export] 所有请求通道均不可用，放弃导出", file=sys.stderr)
            return 2
        if args.ticks % args.stride:
            print(
                f"[viz-export] 注：ticks={args.ticks} 非 stride={args.stride} 整数倍 ⇒ "
                f"末 {args.ticks % args.stride} tick 无采样帧（帧轴止于 +{(n_frames - 1) * args.stride}）",
                file=sys.stderr,
            )
        grid_frames = [first_grids]
        ent_frames = [capture_entities(eng)]
        for i in range(1, args.ticks + 1):
            eng.step()
            if i % args.stride == 0:
                grid_frames.append(capture_grids(eng, eff_channels))
                ent_frames.append(capture_entities(eng))
    except DatapackError as e:
        print(f"[viz-export] 错误：{e}", file=sys.stderr)
        return 2

    out = Path(args.out)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    (out / "entities").mkdir(parents=True, exist_ok=True)

    ch_meta = []
    for key in eff_channels:
        scale = _scale_from_samples(
            np.concatenate([_subsample(f[key]) for f in grid_frames])
        )
        d = out / "frames" / f"ch_{key}"
        d.mkdir(parents=True, exist_ok=True)
        for k, f in enumerate(grid_frames):
            quantize(f[key], scale).tofile(d / f"{k:05d}.bin")
        ch_meta.append(
            {"key": key, "label": CHANNEL_LABELS[key], "kind": "grid", "scale": scale, "cmap": "viridis"}
        )

    ent_norm = {}
    for ci, name in ((3, "energy"), (4, "age")):
        if ent_frames and any(f.shape[0] for f in ent_frames):
            ent_norm[name] = _scale_from_samples(
                np.concatenate([_subsample(f[:, ci]) for f in ent_frames if f.shape[0]])
            )
    for k, f in enumerate(ent_frames):
        f.astype("<f4").tofile(out / "entities" / f"{k:05d}.f32")

    if args.series:
        shutil.copyfile(args.series, out / "series.csv")

    meta = {
        "contract_version": "v0",
        "world": {"rows": rows, "cols": cols, "n_cells": n_cells, "projection": "equirect"},
        "tick": {
            "start": t0,
            "end": t0 + (n_frames - 1) * args.stride,
            "stride": args.stride,
            "n_frames": n_frames,
        },
        "channels": ch_meta,
        "entities": {
            "columns": list(ENTITY_COLUMNS),
            "dtype": "float32",
            "subdiv": int(eng.config.subpos.subdiv),  # 契约 v0 修订1：sub_r/sub_c 为亚格绝对坐标
            "norm": {k: {kk: v[kk] for kk in ("p_lo", "p_hi")} for k, v in ent_norm.items()},
        },
        "source": {
            "snapshot": str(args.snapshot),
            "snapshot_tick": t0,
            "config_fingerprint": eng.config.fingerprint(),
            "engine": "rust" if eng._use_sim_core else "python",
        },
        "generator": {
            "tool": "viz/exporter",
            "version": "v0",
            "created_at": datetime.now().isoformat(timespec="seconds"),
        },
    }
    with open(out / "meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=1)

    print(
        f"[viz-export] OK: {out}｜世界 {rows}x{cols}｜tick {meta['tick']['start']}→{meta['tick']['end']}"
        f" stride={args.stride}｜帧 {n_frames}｜通道 {eff_channels}｜引擎 {meta['source']['engine']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
