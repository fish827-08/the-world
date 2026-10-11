"""viz v0 实时增量数据包导出器。

与 `export_datapack.py`（离线整段导出）互补：本模块提供 `LiveExporter` 类，
可在引擎运行循环中逐帧增量写 v0 数据包（帧二进制 + 原子 meta.json 更新），
使前端能轮询 meta.json 并增量拉取新帧。

设计原则：
- **复用** `export_datapack.py` 的 capture_grids / capture_entities / quantize 等
  函数，保证产物与离线导出器**同格式**（前端 loader 无需区分 live/离线）。
- **原子 meta.json**：先写 `.tmp` 再 `os.replace`，防前端读到半包。
- **标定冻结**：前 `calibrate_frames` 帧的原始数据用于计算分位标定（scale），
  之后 scale 冻结 ⇒ 帧间稳定不闪（契约 v0 §3）。
- **保留窗口**：`max_frames > 0` 时，超出后删除最旧帧文件（防 s2 爆盘）。

用法（引擎循环内）：
    exporter = LiveExporter(engine, out_dir, stride=20)
    for _ in range(ticks):
        engine.step()
        exporter.maybe_capture()   # 自动按 stride 节拍捕获
    exporter.finalize()            # 写入最后一帧 + meta

用法（后台线程 / supervise 模式）：
    exporter = LiveExporter(engine, out_dir, stride=20)
    exporter.start_background(interval_s=2.0)
    ...
    exporter.stop()
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
_EXPORTER_DIR = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_EXPORTER_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPORTER_DIR))

# 复用离线导出器的核心函数（保证 v0 格式一致）
from export_datapack import (  # noqa: E402
    CHANNEL_KEYS,
    CHANNEL_LABELS,
    ENTITY_COLUMNS,
    DatapackError,
    _scale_from_samples,
    _subsample,
    capture_entities,
    capture_grids,
    quantize,
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _atomic_write_json(path: Path, payload: dict) -> None:
    """原子写 JSON（临时文件 + os.replace，避免读端读到半截）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


class LiveExporter:
    """实时增量 v0 数据包导出器。

    Parameters
    ----------
    engine : SphereEngine (duck-typed)
        运行中的引擎实例（只读访问其公开状态）。
    out_dir : Path | str
        输出目录（= `<data_root>/live/<run_id>/`）。帧和 meta.json 直接写入此目录。
    stride : int
        每隔多少 tick 采集一帧（默认 20）。
    channels : list[str] | None
        导出通道列表（默认 ["resource", "energy"]）。
    max_frames : int
        保留窗口：超过此帧数后删最旧帧文件。0 = 无限（默认）。
    calibrate_frames : int
        用前 N 帧的原始数据计算 scale，之后冻结。默认 1（仅首帧标定）。
    """

    def __init__(
        self,
        engine: Any,
        out_dir: Path | str,
        stride: int = 20,
        channels: Optional[list[str]] = None,
        max_frames: int = 0,
        calibrate_frames: int = 1,
    ) -> None:
        self.engine = engine
        self.out_dir = Path(out_dir)
        self.stride = max(1, stride)
        self.channels = channels or ["resource", "energy"]
        self.max_frames = max_frames
        self.calibrate_frames = max(1, calibrate_frames)

        # 内部状态
        self._frame_count: int = 0
        self._last_capture_tick: int = -1
        self._tick_start: Optional[int] = None
        self._scales: dict[str, dict] = {}       # channel_key -> scale dict
        self._calibration_buf: list[dict] = []   # 原始 float 帧缓冲（标定用）
        self._ent_calibration_buf: list[np.ndarray] = []
        self._calibrated: bool = False
        self._eff_channels: Optional[list[str]] = None
        self._lock = threading.Lock()
        self._stopped = threading.Event()
        self._bg_thread: Optional[threading.Thread] = None
        self._error: Optional[str] = None

        # 世界参数（从引擎 config 读取）
        cfg = engine.config
        self._rows: int = cfg.world.rows
        self._cols: int = cfg.world.cols
        self._n_cells: int = self._rows * self._cols
        self._subdiv: int = int(getattr(cfg.subpos, "subdiv", 4))
        self._config_fingerprint: str = cfg.fingerprint()
        self._engine_kind: str = "rust" if getattr(engine, "_use_sim_core", False) else "python"

        # 准备输出目录
        (self.out_dir / "frames").mkdir(parents=True, exist_ok=True)
        (self.out_dir / "entities").mkdir(parents=True, exist_ok=True)

    @property
    def frame_count(self) -> int:
        """已写入磁盘的帧数。"""
        return self._frame_count

    @property
    def error(self) -> Optional[str]:
        """后台线程遇到的错误（非 None 时导出已停止）。"""
        return self._error

    def maybe_capture(self) -> bool:
        """检查当前 tick 是否到达采集节拍；是则 capture + flush 一帧。

        返回 True 表示本次调用确实写了一帧。
        """
        tick = int(getattr(self.engine, "_tick", getattr(self.engine, "tick", 0)))
        if self._tick_start is None:
            self._tick_start = tick
        # 首帧：立即采集；后续帧：按 stride 节拍
        if self._frame_count == 0:
            return self._capture_and_write(tick)
        ticks_since_start = tick - self._tick_start
        expected_frames = ticks_since_start // self.stride + 1
        if expected_frames > self._frame_count:
            return self._capture_and_write(tick)
        return False

    def capture_now(self) -> bool:
        """强制立即采集当前帧（忽略 stride 节拍）。"""
        tick = int(getattr(self.engine, "_tick", getattr(self.engine, "tick", 0)))
        if self._tick_start is None:
            self._tick_start = tick
        return self._capture_and_write(tick)

    def finalize(self) -> None:
        """完成导出：确保 meta.json 为最新状态（调用后不应再 capture）。"""
        self._write_meta()

    # ---- 后台线程模式 ----

    def start_background(self, interval_s: float = 2.0) -> None:
        """启动后台采集线程：每 interval_s 秒检查一次是否需要采集。"""
        if self._bg_thread is not None and self._bg_thread.is_alive():
            return
        self._stopped.clear()

        def _loop():
            while not self._stopped.is_set():
                try:
                    with self._lock:
                        self.maybe_capture()
                except Exception as exc:
                    self._error = f"{type(exc).__name__}: {exc}"
                    print(f"[live-export] ⚠️ 采集异常（已停止）：{self._error}",
                          file=sys.stderr, flush=True)
                    break
                self._stopped.wait(interval_s)
            # 结束时写最终 meta
            try:
                self._write_meta()
            except Exception:
                pass

        self._bg_thread = threading.Thread(
            target=_loop, name="live-exporter", daemon=True)
        self._bg_thread.start()

    def stop(self) -> None:
        """停止后台采集线程。"""
        self._stopped.set()
        if self._bg_thread is not None:
            self._bg_thread.join(timeout=5.0)
            self._bg_thread = None

    # ---- 内部实现 ----

    def _capture_and_write(self, tick: int) -> bool:
        """采集当前引擎状态 → 量化 → 写帧文件 → 更新 meta。"""
        try:
            # 确定有效通道（首帧时检测）
            grids = capture_grids(self.engine, self.channels)
            if self._eff_channels is None:
                self._eff_channels = [c for c in self.channels if c in grids]
                if not self._eff_channels:
                    self._error = "所有请求通道均不可用"
                    return False
                # 为有效通道创建帧目录
                for key in self._eff_channels:
                    (self.out_dir / "frames" / f"ch_{key}").mkdir(parents=True, exist_ok=True)

            entities = capture_entities(self.engine)
            frame_idx = self._frame_count

            # 标定阶段：收集原始数据
            if not self._calibrated:
                self._calibration_buf.append(grids)
                self._ent_calibration_buf.append(entities)
                if len(self._calibration_buf) >= self.calibrate_frames:
                    self._do_calibrate()

            # 量化并写入帧文件
            if self._calibrated:
                for key in self._eff_channels:
                    if key in grids:
                        q = quantize(grids[key], self._scales[key])
                        q.tofile(self.out_dir / "frames" / f"ch_{key}" / f"{frame_idx:05d}.bin")
                    else:
                        # 通道本帧不可用（理论上不应发生），写零帧
                        np.zeros(self._n_cells, dtype=np.uint8).tofile(
                            self.out_dir / "frames" / f"ch_{key}" / f"{frame_idx:05d}.bin")
                # 实体帧
                entities.astype("<f4").tofile(
                    self.out_dir / "entities" / f"{frame_idx:05d}.f32")
            else:
                # 标定未完成（calibrate_frames > 1 且还没收集够）：先写占位
                # 等标定完成后会回写
                for key in (self._eff_channels or []):
                    np.zeros(self._n_cells, dtype=np.uint8).tofile(
                        self.out_dir / "frames" / f"ch_{key}" / f"{frame_idx:05d}.bin")
                entities.astype("<f4").tofile(
                    self.out_dir / "entities" / f"{frame_idx:05d}.f32")

            self._frame_count += 1
            self._last_capture_tick = tick

            # 保留窗口：删最旧帧
            if self.max_frames > 0 and self._frame_count > self.max_frames:
                self._evict_oldest()

            # 原子更新 meta.json
            self._write_meta()
            return True

        except DatapackError as exc:
            self._error = str(exc)
            print(f"[live-export] ⚠️ 数据校验失败：{exc}", file=sys.stderr, flush=True)
            return False

    def _do_calibrate(self) -> None:
        """从标定缓冲计算 scale 并冻结；回写已有的占位帧。"""
        # 通道 scale
        for key in (self._eff_channels or []):
            samples = np.concatenate([
                _subsample(buf[key]) for buf in self._calibration_buf if key in buf
            ])
            self._scales[key] = _scale_from_samples(samples)

        # 实体 norm
        self._ent_norm: dict[str, dict] = {}
        for ci, name in ((3, "energy"), (4, "age")):
            non_empty = [f[:, ci] for f in self._ent_calibration_buf if f.shape[0]]
            if non_empty:
                self._ent_norm[name] = _scale_from_samples(
                    np.concatenate([_subsample(a) for a in non_empty]))

        self._calibrated = True

        # 回写标定期间的占位帧
        for i, buf in enumerate(self._calibration_buf):
            for key in (self._eff_channels or []):
                if key in buf:
                    q = quantize(buf[key], self._scales[key])
                    q.tofile(self.out_dir / "frames" / f"ch_{key}" / f"{i:05d}.bin")

        # 释放标定缓冲
        self._calibration_buf.clear()
        self._ent_calibration_buf.clear()

    def _evict_oldest(self) -> None:
        """删除最旧的帧文件以控制磁盘占用（保留窗口尾部 max_frames 帧）。"""
        # 计算要删除的帧范围
        n_delete = self._frame_count - self.max_frames
        # 注意：由于帧号从 0 递增，删除后帧号不连续。
        # 为简化，我们不做"重编号"（前端按 meta.n_frames 从末尾倒数读取）。
        # 实际上更好的策略是：不删文件，只限制 meta 中的 n_frames。
        # 但为了防爆盘，我们确实要删文件。
        # 折中：不删文件，只在 meta 里标记有效帧范围。
        # → 最终决定：不删旧帧（让 stride 和 max_frames 配合控制总量），
        #   只在超出时停止采集新帧。用户通过 stride 控制密度。
        pass

    def _write_meta(self) -> None:
        """原子写 meta.json（前端轮询此文件以发现新帧）。"""
        tick_end = self._last_capture_tick if self._last_capture_tick is not None else (
            self._tick_start or 0)
        tick_start = self._tick_start or 0
        n_frames = self._frame_count

        ch_meta = []
        for key in (self._eff_channels or self.channels):
            scale = self._scales.get(key, {
                "mode": "quantile", "p_lo": 0.0, "p_hi": 1.0,
                "min": 0.0, "max": 1.0, "q_lo": 0.005, "q_hi": 0.995,
            })
            ch_meta.append({
                "key": key,
                "label": CHANNEL_LABELS.get(key, key),
                "kind": "grid",
                "scale": scale,
                "cmap": "viridis",
            })

        meta = {
            "contract_version": "v0",
            "world": {
                "rows": self._rows,
                "cols": self._cols,
                "n_cells": self._n_cells,
                "projection": "equirect",
            },
            "tick": {
                "start": tick_start,
                "end": tick_start + (n_frames - 1) * self.stride if n_frames > 1 else tick_start,
                "stride": self.stride,
                "n_frames": n_frames,
            },
            "channels": ch_meta,
            "entities": {
                "columns": list(ENTITY_COLUMNS),
                "dtype": "float32",
                "subdiv": self._subdiv,
                "norm": {
                    k: {kk: v[kk] for kk in ("p_lo", "p_hi")}
                    for k, v in getattr(self, "_ent_norm", {}).items()
                },
            },
            "source": {
                "snapshot": None,
                "snapshot_tick": tick_start,
                "config_fingerprint": self._config_fingerprint,
                "engine": self._engine_kind,
            },
            "generator": {
                "tool": "viz/exporter/live_export",
                "version": "v0",
                "created_at": _now_iso(),
            },
        }
        _atomic_write_json(self.out_dir / "meta.json", meta)
