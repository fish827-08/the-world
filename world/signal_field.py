"""田字格信号场：每格 4 子格 0/1 = 16 种标记模式。

语言涌现的信号载体。生物可在当前格写入标记（耗能，由引擎扣），标记持续
duration tick 后自动消失。其他生物经过时可读取标记，获取前者留下的信息。

16 种模式作为字母表绰绰有余（人类音素最少的 Rotokas 语仅 11 个），
田字格的 4 个位置提供词内结构（槽位语法），高阶语言需文化积累涌现组合规则。

存储：uint8 低 4 位（bit0~bit3 对应 4 个子格），0 = 无标记。
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from world.sphere_world import SphereWorld


class SignalField:
    __slots__ = (
        "world",
        "duration",
        "_marks",
        "_age",
        # ---- R217 §五 #1 稀疏化 B：活跃集推进（默认关；见 `enable_sparse`）----
        "_sparse",
        "_active_mask",
        "_active_idx",
    )

    def __init__(self, world: SphereWorld, duration: int = 50) -> None:
        """构造信号场。

        参数
        ----
        world : SphereWorld
            球面世界（决定格子数）。
        duration : int
            标记持续多少 tick 后自动消失。默认 50。
        """
        self.world = world
        self.duration = int(duration)
        self._marks = np.zeros(world.n_cells, dtype=np.uint8)   # sparse:init
        self._age = np.zeros(world.n_cells, dtype=np.int32)     # sparse:init
        # ---- R217 §五 #1 稀疏化 B：活跃集（默认关 = 全场推进逐位不变）----
        self._sparse = False
        self._active_mask = None
        self._active_idx = None

    # ---- 稀疏推进（R217 §五 #1 稀疏化 B；默认关）-----------------------------

    def enable_sparse(self) -> None:
        """启用稀疏推进（只推进"有标记（`age>0`）"的格）。

        逐位等价是**构造性**的：全场版 `tick` 的 `active = age>0` 本就是挑选操作，
        稀疏版把这套整数运算（`-=1` / 清零）原样施加在活跃集上，**同一批格、同一顺序**。
        """
        self._sparse = True
        self.rebuild_sparse()

    def rebuild_sparse(self) -> None:
        """（重）建活跃集 = `age > 0`。

        **启用时与快照恢复后必须调用**：活跃集是派生量、不进快照；`_marks`/`_age` 被整体
        替换后（快照恢复）必须重建，否则活跃集陈旧 ⇒ 会漏推进（破等价）。
        口径与全场版一致（以 `age > 0` 为准，而非 `marks != 0` —— 快照里两者可能不同步，
        而全场版只认 `age`）。
        """
        if not self._sparse:
            return
        self._active_mask = self._age > 0
        self._active_idx = np.flatnonzero(self._active_mask)

    def _activate(self, cells: NDArray[np.int64], patterns: NDArray) -> None:
        """把新写入（pattern≠0）的格并入活跃集（**数组入参**，供 `write_many`）。

        🔴 集内**不许有重复格**：`tick` 按格递减年龄，重复项会**双重递减**
        ⇒ 寿命减半、破等价。故对新增部分做 `np.unique`（重复只可能来自同批写入，
        规模 = 本 tick 写入者数，非格数）。
        """
        new = cells[(np.asarray(patterns) & 0x0F) > 0]
        if new.size == 0:
            return
        fresh = new[~self._active_mask[new]]
        if fresh.size == 0:
            return
        self._active_mask[new] = True
        self._active_idx = np.concatenate([self._active_idx, np.unique(fresh)])

    # ---- 写入 ----------------------------------------------------------------

    def write(self, cell: int, pattern: int) -> None:
        """在指定格写入标记模式（0~15），覆盖已有标记，重置寿命。

        pattern=0 等同于清除标记。
        """
        self._marks[cell] = np.uint8(pattern & 0x0F)   # sparse:active（下行同函数内维护活跃集）
        self._age[cell] = self.duration if (pattern & 0x0F) else 0   # sparse:active
        if self._sparse and (pattern & 0x0F) and not self._active_mask[cell]:
            self._active_mask[cell] = True
            self._active_idx = np.concatenate(
                [self._active_idx, np.asarray([cell], dtype=np.int64)])

    def write_many(
        self, cells: NDArray[np.int64], patterns: NDArray[np.uint8]
    ) -> None:
        """批量写入（向量化）。cells 和 patterns 等长。"""
        p = patterns & 0x0F
        self._marks[cells] = p   # sparse:active（下行 `_activate` 并入活跃集）
        self._age[cells] = np.where(p > 0, self.duration, 0).astype(np.int32)   # sparse:active
        if self._sparse:
            self._activate(cells, p)

    # ---- 读取 ----------------------------------------------------------------

    def read(self, cell: int) -> int:
        """读取指定格的当前标记模式（0=无标记）。"""
        return int(self._marks[cell])

    def read_many(self, cells: NDArray[np.int64]) -> NDArray[np.uint8]:
        """批量读取，返回副本。"""
        return self._marks[cells].copy()

    # ---- 时间推进 ------------------------------------------------------------

    def tick(self) -> None:
        """推进一个 tick：所有标记年龄-1，过期清零。"""
        if self._sparse:
            idx = self._active_idx
            if idx.size == 0:
                return
            self._age[idx] -= 1   # sparse:active（同函数内剪枝+维护掩码）
            keep = self._age[idx] > 0
            if not keep.all():
                drop = idx[~keep]
                self._marks[drop] = 0   # sparse:active（同函数内从活跃集剔除）
                self._age[drop] = 0     # sparse:active
                self._active_mask[drop] = False
                self._active_idx = idx[keep]
            return
        active = self._age > 0
        self._age[active] -= 1   # sparse:full（默认分支，逐位不变）
        expired = active & (self._age <= 0)
        self._marks[expired] = 0   # sparse:full
        self._age[expired] = 0     # sparse:full

    # ---- 工具 ----------------------------------------------------------------

    def clear(self, cell: int) -> None:
        """清除指定格的标记。

        ⚠️ 稀疏档下**不**即时从活跃集移除（数组搜索成本 > 收益）：该格成为"陈旧项"
        （`age == 0, marks == 0`），下一次 `tick` 顺带剪枝 —— 与全场版语义一致
        （全场版对该格同样什么都不做）。
        """
        self._marks[cell] = 0   # sparse:active（陈旧项由下次 tick 剪枝，见 docstring）
        self._age[cell] = 0     # sparse:active

    def snapshot(self) -> NDArray[np.uint8]:
        """返回二维快照 (rows, cols)，用于可视化。"""
        return self._marks.reshape(self.world.rows, self.world.cols).copy()

    def active_count(self) -> int:
        """当前有标记的格子数。"""
        return int(np.count_nonzero(self._marks))

    def total_marks(self) -> int:
        """所有标记的子格点亮总数（用于统计信号密度）。"""
        return int(np.unpackbits(self._marks).sum())
