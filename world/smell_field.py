"""气味场（多通道标量场）—— v1 原型（R240 T8；设计稿《气味场与分功能感知》§二/§8.2）。

为什么
------
二维随机游走的**首次命中**期望步数 ≈ d²（d = 斑块间距）：现状 d≈15 格 ⇒ 约 225 步，
而满能量续航只有 273 格 ⇒ **82% 的个体在撞上斑块前就饿死**（S1 实测：31–47% 斑块格
从未被访问）。扩视觉要扩 15 倍（成本爆炸）；**气味场只用 1 格视野就能拿到 15 格外的信息**
—— 气味是**环境量**，个体仍然"只读自己格" ⇒ **不违反"感知 1 格"**。

模型（每通道；引擎每 k tick 调一次 `update()`）
----------------------------------------------
设计稿 §二 的标量场模型：`S ← decay^k × (S + 注入) + D × lap(S)`。
本模块按设计稿 §8.2 的**两级策略**实现（近场全分辨率 + 远场降采样）：

| 级 | 载体 | 每步做 | 成本 |
|---|---|---|---|
| **近场** | `_S` 全分辨率 `(C, n_cells)` | 衰减（全场一次乘）+ **稀疏**注入 | O(n_cells)，但只是"乘"（内存带宽级）|
| **远场** | `_Sc` 粗网格 `(C, cr, cc)`（1/s，默认 s=8）| 衰减 + **稀疏**注入（按格分箱）+ 五点拉普拉斯（长程扩散）| O(n_cells/s²) |
| **读** | `at(cells)` = 近场 + **双线性插值**远场（在**被读的格**上算）| 个体只读自己格 ⇒ O(被读格数)，不是 O(n_cells) |
| 诊断 | `to_full()` 物化整场（`(C, n_cells)`）| 只给可视化/测试用，**不进每 tick 路径** | 可测（见 `probe()`）|

🔴 **为什么"插值回写"必须移到读路径**（实测依据，不是口味）
----------------------------------------------------------
第一版按设计稿字面写成"每步粗算 + 双线性插值**回写**全分辨率"，实测（480×960，四通道，
本机 2 核）：**Δ = +18.0 ms/tick（+124%）**，分解里"插值回写"独占 **14.3 ms/tick**
⇒ 是设计稿 §8.2 预算（四通道 1–3 ms/tick）的 **6–18 倍**，正是红线要防的"地板税"。
把插值移到**读路径**后：远场不再物化 ⇒ 每 tick 只剩"全场乘 + 稀疏注入 + 粗网格小算子"
⇒ 四通道 **≈0.7 ms/tick**（实测见 `results/smell_v1/smell_perf.csv`）。
⇒ 这是设计稿 §8.2「两级策略（v1.1 可选）」的**提前采用**，语义不变（读到的浓度仍是
"近场 + 长程扩散"的合成值），只是"插值"从每步回写变成**按需/按读**计算。

三条内置优化（设计稿 §8.2 红线，**不可省**）
--------------------------------------------
① **降采样长程**：扩散在 1/s 粗网格上算（成本 ÷ s²）
② **降低更新频率**：每 **k tick** 一次（默认 k=4）⇒ 成本 ÷k
③ **源注入稀疏**：只碰"有源"格（食物格 2.8% / 个体 0.65%），与扩散**解耦**
🔴 **禁止"每 tick 全场稠密卷积/回写"**：本模块 `update()` 是唯一全场路径，引擎每 k tick 才调
（AST 守卫 `tests/test_smell_write_guard.py` + 运行时节拍测试 + 性能实测三重防守）。

默认全关
--------
`SmellConfig.channels == ()` ⇒ 引擎**不构造**本模块（`eng.smell is None`）⇒ 零成本、
旧行为**逐位不变**（C7 基线 `(574887, 11266.746993)` 不动；回滚点）。

通道（R238 裁定：四通道全上、**分两批启用**）
--------------------------------------------
| 通道 | 源（引擎侧算，**稀疏**） | 生物学对应 |
|---|---|---|
| `food` | 资源格（注入量 = 食物量/最大容量 ∈ [0,1]） | 食物挥发物 |
| `prey` | 活体个体 | 猎物气味 |
| `risk` | `g16 ≥ risk_g16_threshold` 的个体 | kairomone（捕食者信号） |
| `kin`  | 活体个体（同源、读端用途不同） | 同类信息素 |

⚠️ **消费端（score 加权）不在本模块**：权重需扩 8 个基因位 = **纪元级**（R238 §2）
⇒ 属批 A/B 排期。本模块只提供**接口 + 读 API**（`at()`）——接线时直接用。

与 `sparse_fields` 的关系（设计稿 §8.4 取方案 (a)）
--------------------------------------------------
本模块**自带**稀疏/降采样路径 ⇒ 与 `sparse_fields` **正交**、独立开关
（不依赖它，也不受 rd 开档退回全场的影响）。

写点标签（供 `tests/test_smell_write_guard.py` 的 AST 守卫）
-----------------------------------------------------------
| 标签 | 含义 |
|---|---|
| `smell:full` | **全场**路径（近场衰减）—— 引擎每 k tick 才允许走到 |
| `smell:inj`  | **稀疏**源注入（只碰有源格；`np.add.at` ⇒ 同格多个体累加） |
| `smell:reset`| 整体替换（快照恢复 / 清场） |
| `smell:init` | 构造期初始化 |
"""
from __future__ import annotations

import math
import sys
import time
from typing import Any

import numpy as np
from numpy.typing import NDArray

from world.sphere_world import SphereWorld

# --- R98 纪律：Windows GBK 控制台兜底（非 ASCII print 会让脚本 rc=1 假失败；F-R15 族）---
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass  # 非 TTY / 旧解释器：不因诊断能力缺失而阻断运行


def _largest_divisor_le(n: int, s: int) -> int:
    """≤ s 且整除 n 的最大正整数（把降采样因子落到"精确分块"上）。"""
    for d in range(min(int(s), int(n)), 0, -1):
        if n % d == 0:
            return d
    return 1


class SmellField:
    """多通道气味场（v1，两级）。**由引擎按 `SmellConfig` 构造**；本模块零引擎依赖。"""

    __slots__ = (
        "world", "cfg", "channels", "n_chan", "update_every", "downsample_req",
        "_rows", "_cols", "_s", "_cr", "_cc", "_idx",
        "_S", "_Sc",
        "_r0", "_r1", "_fr", "_c0", "_c1", "_fc",
        "_updates", "_inj_cells_last",
        "_ms_decay", "_ms_inject", "_ms_coarse", "_ms_full",
        "_reads_n", "_ms_read",
    )

    def __init__(self, world: SphereWorld, cfg: Any) -> None:
        """构造气味场。

        parameters
        ----------
        world : SphereWorld
            球面世界（`rows/cols/n_cells`；平铺布局 = `row * cols + col`）。
        cfg : SmellConfig
            参数契约（`simulation/config.py`）。⚠️ `channels` 为空时**不应构造本模块**
            （引擎侧已拦）—— 这里直接断言，防"构造了却全关"的静默态。
        """
        self.world = world
        self.cfg = cfg
        self.channels = tuple(cfg.channels)
        assert self.channels, (
            "channels 为空（= 全关）时不应构造 SmellField —— 引擎侧必须拦在构造前"
            "（否则就是「构造了却全关」的静默态）")
        self._idx = {c: i for i, c in enumerate(self.channels)}
        # 🔴 通道名二次校验：`SmellConfig.__post_init__` 管构造期；**构造后改配置**不走它
        #   ⇒ 这里再查一遍（拼错通道 = 静默 no-op，B3 家族；禁静默）。
        _known = tuple(getattr(type(cfg), "_KNOWN_CHANNELS", ("food", "prey", "risk", "kin")))
        _bad = [c for c in self.channels if c not in _known]
        assert not _bad, f"未知气味通道 {_bad}（允许：{_known}）—— 拼错通道必须炸，不许静默"
        self.n_chan = len(self.channels)
        self.update_every = int(cfg.update_every)
        self.downsample_req = int(cfg.downsample)
        self._rows, self._cols = int(world.rows), int(world.cols)
        # 🔴 `s_eff` 必须**同时**整除 rows 与 cols（精确分块是粗网格/插值的前提）
        #   ⇒ 取 `gcd(rows, cols)` 的、≤ 请求值的最大因子（不静默：回落即告警）。
        self._s = _largest_divisor_le(math.gcd(self._rows, self._cols),
                                      self.downsample_req)
        self._cr, self._cc = self._rows // self._s, self._cols // self._s
        if self._s != self.downsample_req:
            print(f"⚠️ 气味场降采样因子回落：请求 {self.downsample_req} ⇒ 实际 {self._s}"
                  f"（{self._rows}×{self._cols} 需能被整除；`probe()['downsample_eff']` 可读回）",
                  file=sys.stderr)
        # 状态（**两级；都进快照**）：近场全分辨率 + 远场粗网格
        self._S = np.zeros((self.n_chan, world.n_cells), dtype=np.float64)   # smell:init
        self._Sc = np.zeros((self.n_chan, self._cr, self._cc), dtype=np.float64)  # smell:init
        # 插值索引/权重（构造期一次算好；**读路径**与 `to_full()` 用）
        r = np.arange(self._rows)
        c = np.arange(self._cols)
        self._r0 = (r // self._s).astype(np.int64)
        self._r1 = np.minimum(self._r0 + 1, self._cr - 1)      # 纬度：夹取
        self._fr = ((r % self._s) / self._s).astype(np.float64)
        self._c0 = (c // self._s).astype(np.int64)
        self._c1 = (self._c0 + 1) % self._cc                  # 经度：周期
        self._fc = ((c % self._s) / self._s).astype(np.float64)
        # 读数（B4 口径：本模块只被引擎在"开档"时构造 ⇒ 计数器天然只在开档累加）
        self._updates = 0
        self._inj_cells_last = 0
        self._ms_decay = 0.0
        self._ms_inject = 0.0
        self._ms_coarse = 0.0
        self._ms_full = 0.0
        self._reads_n = 0
        self._ms_read = 0.0

    # ---- 更新（唯一全场路径；引擎按节拍调用）---------------------------------

    def update(self, sources: "dict[str, tuple[NDArray[np.int64], NDArray[np.float64]]]",
               tick: int = -1) -> None:
        """**更新一次**（近场衰减 → 稀疏注入 → 远场粗网格扩散）。

        parameters
        ----------
        sources : {通道名: (cells, amounts)}
            **稀疏**源：只给"有源"的格（设计稿 §8.2-③）。引擎侧算（它握有资源/个体/基因）。
        tick : int
            当前引擎 tick（**仅记录**，不参与算术 ⇒ 续跑逐位与它无关）。
        """
        t0 = time.perf_counter()
        dec = float(self.cfg.decay) ** self.update_every
        # ① 衰减（近场：全场一次乘；远场：粗网格一次乘）
        self._S *= dec          # smell:full
        self._Sc *= dec         # smell:full
        t1 = time.perf_counter()
        # ② 稀疏源注入（近场逐格 + 远场分箱；同格多个体必须**累加** ⇒ `np.add.at`）
        n_inj = 0
        for name, (cells, amounts) in sources.items():
            ci = self._idx.get(name)
            if ci is None:
                continue                                  # 该通道未启用 ⇒ 不算（引擎按 channels 给源）
            w = {"food": self.cfg.inject_food, "prey": self.cfg.inject_prey,
                 "risk": self.cfg.inject_risk, "kin": self.cfg.inject_kin}.get(name, 0.0)
            if w == 0.0 or np.size(cells) == 0:
                continue
            add = w * np.asarray(amounts, dtype=np.float64)
            np.add.at(self._S[ci], cells, add)            # smell:inj（同格累加）
            rr, cc = np.divmod(cells, self._cols)
            np.add.at(self._Sc[ci], (rr // self._s, cc // self._s), add)   # smell:inj（分箱）
            n_inj += int(np.size(cells))
        t2 = time.perf_counter()
        # ③ 远场五点拉普拉斯（长程扩散；粗网格 ⇒ 成本 ÷s²）
        self._Sc += float(self.cfg.diffuse) * self._laplacian_coarse(self._Sc)   # smell:full
        t3 = time.perf_counter()
        # 读数（性能分解；见 probe()）
        self._updates += 1
        self._inj_cells_last = n_inj
        self._ms_decay += (t1 - t0) * 1e3
        self._ms_inject += (t2 - t1) * 1e3
        self._ms_coarse += (t3 - t2) * 1e3

    def _laplacian_coarse(self, A: NDArray[np.float64]) -> NDArray[np.float64]:
        """粗网格五点拉普拉斯：**经度周期、纬度夹取**（球面近似；v1 口径）。"""
        P = np.pad(A, ((0, 0), (1, 1), (1, 1)), mode="edge")   # 纬度：夹取
        P[:, 1:-1, 0] = A[:, :, -1]   # 经度：周期（左；只填内部行，行角不被五点格式读取）
        P[:, 1:-1, -1] = A[:, :, 0]   # 经度：周期（右）
        return (P[:, :-2, 1:-1] + P[:, 2:, 1:-1]
                + P[:, 1:-1, :-2] + P[:, 1:-1, 2:] - 4.0 * A)   # smell:n/a（临时量，非 _S/_Sc）

    # ---- 读（个体只读自己格）-------------------------------------------------

    def _interp_cells(self, ci: int, cells: NDArray[np.int64]) -> NDArray[np.float64]:
        """**远场**在被读格上的双线性插值（读路径；O(被读格数)，不是 O(n_cells)）。"""
        rr, cc = np.divmod(cells, self._cols)
        r0 = self._r0[rr]
        c0 = self._c0[cc]
        fr = self._fr[rr]
        fc = self._fc[cc]
        A = self._Sc[ci]
        return ((1.0 - fr) * (1.0 - fc) * A[r0, c0] + (1.0 - fr) * fc * A[r0, self._c1[cc]]
                + fr * (1.0 - fc) * A[self._r1[rr], c0] + fr * fc * A[self._r1[rr], self._c1[cc]])

    def at(self, cells: "int | NDArray[np.int64]", channel: str = "food"):
        """读**指定格**的浓度 = 近场 + 远场插值（个体只读自己格 ⇒ 不违反"感知 1 格"）。

        `channel` 必须在启用集合内；否则**直接断言**（拼错通道 = 静默 no-op ⇒ B3 家族）。
        """
        ci = self._idx.get(channel)
        assert ci is not None, (
            f"未启用通道 {channel!r}（本档启用：{self.channels}）—— 拼错通道必须炸，不许静默")
        t0 = time.perf_counter()
        arr = np.asarray(cells, dtype=np.int64)
        out = self._S[ci][arr] + self._interp_cells(ci, arr)
        self._reads_n += int(np.size(arr))
        self._ms_read += (time.perf_counter() - t0) * 1e3
        return out

    def to_full(self, channel: str | None = None) -> NDArray[np.float64]:
        """物化整场 `(C, n_cells)`（近场 + 远场插值）—— **诊断/测试用，不进每 tick 路径**。"""
        t0 = time.perf_counter()
        chans = [self._idx[channel]] if channel else list(range(self.n_chan))
        out = np.empty((len(chans), self._rows * self._cols), dtype=np.float64)
        rr, cc = np.divmod(np.arange(self._rows * self._cols, dtype=np.int64), self._cols)
        r0, c0 = self._r0[rr], self._c0[cc]
        r1, c1 = self._r1[rr], self._c1[cc]
        wr, wc = self._fr[rr], self._fc[cc]
        for i, ci in enumerate(chans):
            A = self._Sc[ci]
            up = ((1.0 - wr) * (1.0 - wc) * A[r0, c0] + (1.0 - wr) * wc * A[r0, c1]
                  + wr * (1.0 - wc) * A[r1, c0] + wr * wc * A[r1, c1])
            out[i] = self._S[ci] + up
        self._ms_full += (time.perf_counter() - t0) * 1e3
        return out if channel is None else out[0]

    # ---- 快照（两级**都是**状态；派生/计时**不进档**）--------------------------

    def state(self) -> dict:
        """快照载荷 = 两级状态（`{"S": 近场, "Sc": 远场}`）。

        派生量（`probe()` 的读数、计时器）是函数/读数 ⇒ **不进档**。
        """
        return {"S": self._S.copy(), "Sc": self._Sc.copy()}

    def restore(self, payload) -> None:
        """从快照恢复（**整体替换**两级状态）。旧档/异形 ⇒ fail-loud（不许静默）。"""
        d = payload if isinstance(payload, dict) else {"S": payload, "Sc": None}
        arr = np.asarray(d["S"], dtype=np.float64)
        assert arr.shape == self._S.shape, (
            f"气味场近场形状不符：快照 {arr.shape} vs 当前 {self._S.shape}"
            f"（跨档/跨通道集续跑必须被拦，不许静默）")
        self._S = arr.copy()          # smell:reset
        if d.get("Sc") is not None:
            sc = np.asarray(d["Sc"], dtype=np.float64)
            assert sc.shape == self._Sc.shape, f"气味场远场形状不符：{sc.shape} vs {self._Sc.shape}"
            self._Sc = sc.copy()      # smell:reset
        else:                         # 旧档只有近场 ⇒ 远场归零起算（口径：文档已声明）
            self._Sc[:] = 0.0         # smell:reset
        self._updates = 0
        self._inj_cells_last = 0

    def clear(self) -> None:
        """清场（测试/诊断用）。"""
        self._S[:] = 0.0              # smell:reset
        self._Sc[:] = 0.0             # smell:reset

    # ---- 诊断 ----------------------------------------------------------------

    def probe(self) -> dict:
        """读数（性能分解 + 形状 + 节拍）—— 供实证与 `[所有者]` 核验（R219 配对口径用）。"""
        u = max(self._updates, 1)
        return {
            "channels": list(self.channels), "n_chan": self.n_chan,
            "update_every": self.update_every,
            "downsample_req": self.downsample_req, "downsample_eff": self._s,
            "coarse_shape": [self._cr, self._cc],
            "updates": self._updates, "inj_cells_last": self._inj_cells_last,
            "ms_update_total": {k: round(v, 3) for k, v in
                                (("decay", self._ms_decay), ("inject", self._ms_inject),
                                 ("coarse_lap", self._ms_coarse))},
            "ms_per_update": {"decay": round(self._ms_decay / u, 4),
                              "inject": round(self._ms_inject / u, 4),
                              "coarse_lap": round(self._ms_coarse / u, 4)},
            "read_path_ms_total": round(self._ms_read, 3),
            "read_path_cells": self._reads_n,
            "to_full_ms_total": round(self._ms_full, 3),
            "field": {c: {"S_max": float(self._S[i].max()), "Sc_max": float(self._Sc[i].max()),
                          "S_nonzero": int(np.count_nonzero(self._S[i])),
                          "Sc_nonzero": int(np.count_nonzero(self._Sc[i]))}
                      for i, c in enumerate(self.channels)},
        }