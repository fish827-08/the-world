"""biosphere.checkpoint：跑批可恢复协议（>1000 tick 跑批必备；补充指令 5）。

协议
----
- 所有 >1000 tick 的跑批（experiments 各子命令）**每 N tick 写一个
  checkpoint**（pickle 整个模拟器：种群/rng 态/资源/信号层/观测日志），
  命名 ``ckpt_<tag>_t<tick>.pkl``；
- 同目录 ``<tag>.run.log`` **追加**一行进度（tick/存活/出生/时间戳）；
- 恢复：同一子命令加 ``--resume <pkl>`` ⇒ 从 checkpoint 精确续跑
  （rng 态入 pickle ⇒ 续跑与直跑逐位一致，有测试钉住）；
- 汇报里写"恢复步骤"一行。

为什么 pickle：模拟器状态含只读 numpy 数组 + PCG64 位生成器态 +
嵌套 dataclass，pickle 是唯一无损且零依赖的序列化；Genome 的只读
视图在 ``__setstate__`` 里重新落闸。
"""
from __future__ import annotations

import gzip
import os
import pickle
import time
from pathlib import Path
from typing import Dict, Optional

#: 压缩开关（默认开：观测日志的 numpy 数组重复率高，gzip 省 3-5× 磁盘）
GZIP_ENABLED: bool = True


def save_checkpoint(sim, path, meta: Optional[Dict] = None) -> Path:
    """把模拟器整体写盘（含 meta；默认 gzip + 原子落盘）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"sim": sim, "meta": dict(meta or {}),
               "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    tmp = path.with_suffix(path.suffix + ".tmp")
    opener = (gzip.open if GZIP_ENABLED else open)
    with opener(tmp, "wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
        if not GZIP_ENABLED:
            fh.flush()
    os.replace(tmp, path)      # 原子落盘（防半截文件被当 checkpoint）
    return path


def load_checkpoint(path) -> "object":  # noqa: F821 - 返回 BiosphereSimulator
    """读回模拟器（精确续跑）。"""
    path = Path(path)
    opener = gzip.open if GZIP_ENABLED else open
    with opener(path, "rb") as fh:
        payload = pickle.load(fh)
    return payload["sim"]


def append_run_log(ckpt_dir, tag: str, line: str) -> None:
    """跑批进度日志（追加式）。"""
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    with open(ckpt_dir / f"{tag}.run.log", "a", encoding="utf-8") as fh:
        fh.write(line.rstrip() + "\n")


__all__ = ["save_checkpoint", "load_checkpoint", "append_run_log"]
