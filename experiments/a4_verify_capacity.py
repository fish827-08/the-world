"""A4/R19 生态门验证（本地开发）— 修复后代码重跑，4 机制为主 + 3 机制对照。

裁定依据：
- **R19**：R4 批运行于修复提交前 ⇒ 整体降级；生态门须用**修复后**代码重跑
  **5 seed × 60k、`arbitrary_codebook=true`**。
- **V12 口径**：4 机制为主 + 3 机制对照臂；两档同 seed 集、同 tick、同 log 间隔。
- **R22 (F-D15)**：manifest 必须记录 git commit + sim_core 指纹 + 配置指纹 + 全部开关真实状态。

用法：
  python experiments/a4_verify_capacity.py --mode on --seed 42 --ticks 60000 --codebook 1 \
      --out _rerun_logs/a4_fix/asym_on_cb1_s42.csv --snapshot-every 5000

断点续跑（快照）—— ⚠️ **F-R7：快照默认被 `.gitignore` 排除**
  R19 首批的快照因数据仓 `.gitignore` 含 `*.npz|*.pkl` 而**从未入库**，
  直接后果是"本地接不上、只能从 0 重跑"，白丢约 9 小时。
  ⇒ **D-23a 处置**：快照统一写到 `--snapshot-dir`（默认 `_rerun_logs/snap/`）下的
  **固定文件名 `<tag>.snapshot.npz`（原地覆盖）**，该目录已在 `.gitignore` 中放开，
  **暂停/中断时提交一次即可入库**；每 run 只 1 个文件（≈4–6 MB），10 run ≈ 40–60 MB。
  ❌ 切勿放开 `*.npz` 通配：120 个/run × 4 MB ≈ 480 MB/run，会把仓库再撑爆。
  中断后原命令重跑即自动续跑（读取已存在的快照，从 e._tick+1 继续）
  - --fresh 强制从 tick 0 重来
  ⚠️ D2 的感知噪声走【全局 np.random】（已知缺陷 F-D2），引擎快照不含它；
     故本脚本额外存取 np.random 状态，保证续跑与"不中断连续跑"**逐位一致**。
生态门判定：跑满目标 tick 且 tick>=10000 起 N 全程 >0（= N 稳定 >0 持续 50k）。
进度可见：CSV 每采样点 flush；同目录 <name>.progress.json 每 5000 tick 更新。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from simulation.config import (  # noqa: E402
    CALIBRATION_M_RANGE,
    SIGNAL_ALPHABET_IMPLEMENTED,
    SIGNAL_ALPHABET_STATES,
    ActionSelectionConfig,
    CorpseWoundConfig,
    HungerModConfig,
    InfoStructureConfig,
    PredationConfig,
    ResourceDynamicsConfig,
    SimConfig,
    SubposConfig,
)
from simulation.sphere_engine import SphereEngine  # noqa: E402
# 🔴 13.8：P7 要求产物自证 `migrate_gene_slot == 23` ⇒ 必须从基因表取**常量**，
#    不可抄字面量 23（抄了就永远"通过"，位号真错了也测不出来）。
from simulation.genes import Gene  # noqa: E402
# 🔴 F2 落裁（镜 19:46 审 / PI R383 697e171）：装置预设档单一真源（R278 §三）
#    —— a4 此前**无任何装置口径入口**（rows/cols 吃 config 默认 60×120）⇒ C1 批
#    下达不了 s2 装置。接入后：不传 --device ⇒ 逐位等于旧版（T1 基石）。
from tools.device_presets import add_device_arg, resolve_device  # noqa: E402


# D-19：provenance 统一走 simulation.provenance（硬校验，不再本地静默 None/"unknown"）
from simulation.provenance import collect as prov_collect, validate as prov_validate
from observatory.statistics import (  # D-16：单一口径实现
    codebook_convergence, max_generation_current, predation_fraction,
)
from observatory.statistics import selection_gradient  # D-17：⑤ 单一口径


# R139/R140：g16 直方图的箱数（值域 [0,1] 均分）。**改动它等于改判据口径** ⇒ 单一真源。
G16_BINS = 10


def _hist10(x) -> list[int]:
    """g16 的 10-bin 直方图（值域 [0,1]，右开区间；越界值夹到两端）。"""
    a = np.asarray(x, dtype=np.float64)
    if a.size == 0:
        return [0] * G16_BINS
    idx = np.clip((a * G16_BINS).astype(np.int64), 0, G16_BINS - 1)
    return np.bincount(idx, minlength=G16_BINS)[:G16_BINS].tolist()


# R141 P0 派工单 §1.3 锁定的列名（🔴 勿改名——`tools/calib_solve.py` 按此消费）
EC_CSV_COLS = ("g_lo_n", "g_lo_net_mean", "g_lo_net_p50", "g_lo_net_var",
               "g_mid_n", "g_mid_net_mean", "g_mid_net_p50", "g_mid_net_var",
               "g_hi_n", "g_hi_net_mean", "g_hi_net_p50", "g_hi_net_var",
               "forage_in_mean", "pred_in_mean", "prey_energy_mean")


def _ec_csv_row(e) -> dict:
    """把 `_ec_flush` 刚写的那一行净收入统计摊平成 CSV 列（None ⇒ 空串，R120 口径）。"""
    ts = e.ec_timeseries()
    row = ts[-1] if ts else {}
    out = {}
    for k in EC_CSV_COLS:
        v = row.get(k)
        out[k] = "" if v is None else (int(v) if k.endswith("_n") else round(float(v), 6))
    return out


def _probe_csv(probe: dict | None, key: str):
    """L1/L2 探针读数进 CSV 列（R150 B4）。

    🔴 探针为 `None`（开关关 ⇒ **未适用**）或该键为 `None`（分母为 0）⇒ **空串**，
    **不是 0**（R120 / §五.12 口径铁律："没测" ≠ "测出零"）。
    """
    if not probe:
        return ""
    v = probe.get(key)
    return "" if v is None else v


# ══════════════════════════════════════════════════════════════════════════════
# 13.6 S2 四读数（R193 §四 / R194 §S2；`[云端·开发]`，2026-09-24）
# ══════════════════════════════════════════════════════════════════════════════
# 🔴 **纯观测**（D-18 探针同族）：零 RNG、零行为改变 ⇒ 不传任何新参数时的轨迹
#    与 S1 逐位一致（C7 digest `(574887, 11266.746993)` 不动）。
# 🔴 **口径必须与 R190/R191 的工具可复算对齐**（否则"新读数"与"旧判读"不可比）：
#    · `food_util_frac`   =（取食入能量 ÷ 净吸收）÷（Σ名义再生 × tick 数）
#        —— R190 §三「**实际被吃掉的质量 ÷ 名义再生总量**」
#           源：`tools/p135_verdict.py` 第二部分 ①（分母 = `g_all × T`）
#        ⚠️ 净吸收 = `eat_efficiency × assim_herb`；引擎账本 `intake_forage` 已是
#          **净吸收能量**（`_dg = 质量 × eat_efficiency × _assim`，引擎 2021 行）⇒ 除法正确。
#    · `patch_visit_frac` = 曾被"有生物占据"过的斑块格数 ÷ 斑块格总数
#        —— R191「已访问斑块%」（源：`tools/spatial_diag.py`：`visited[e._flat[:P]] = True`）
#    · `on_patch_frac`    = 生物落在斑块格的比例 —— R191「生物在斑块%」（同源）
#    · `gud_var`          = 生物**当前所占格**的 GUD 方差
#        —— R192 §C′ 建议的 MVT 读数（**本项目首次定义**，口径见下注）
#    · 附赠 `patch_stock_frac` = 斑块总存量 ÷ 斑块总容量（R191「斑块存量/容量」；复算用）
#
# 🔴 GUD（giving-up density）本项目的定义（**首次落码 ⇒ 必须自解释**）：
#    GUD_格 = 生物当前所在格的**相对存量** `stock/capacity`（0=吃空，1=满）。
#    `gud_var` = 这些格上 GUD 的**样本方差**（ddof=1）。
#    MVT 预期：最优觅食 ⇒ 各格被放弃时的存量趋同 ⇒ **方差低**；
#    方差高 ⇒ 位置间质量差异大/觅食未达平衡（正是"森林 vs 草原"要区分的量）。
#    ⚠️ 所占格 < 2（含灭绝）⇒ **None**（禁写 0，R120）；容量 0 的格（bgzero 背景格）
#    从 GUD 样本中**剔除**（分母为 0 ⇒ 该格 GUD 无定义，不是 0）。
SPATIAL_NOTE = (
    "S2 四读数（纯观测，零 RNG）：food_util_frac=取食质量÷名义再生总量（R190 口径）；"
    "patch_visit_frac=曾被占据过的斑块格比例 / on_patch_frac=生物在斑块格比例（R191 口径）；"
    "gud_var=所占格 stock/capacity 的样本方差（MVT 的 GUD；本线首次定义，见代码注释）；"
    "None=未适用（灭绝/样本<2），不是 0。口径源：tools/p135_verdict.py + tools/spatial_diag.py"
)


class SpatialReadings:
    """13.6 S2 四读数的**累计器**（每 tick 观测 + 采样点快照；纯观测）。

    为什么需要累计器：`patch_visit_frac` 是"**曾经**被占据过"的**历史量**，
    必须逐 tick 累积（不是某一刻的瞬时量）。

    断点续跑（F-R12 家族）：`visited` 掩码与"取食能量结转"存**侧车 npz**
    （`<out>.spatial.npz`，与快照同目录同节拍）——因为引擎快照**不含能量账本**
    （`energy_ledger` 只读 `_ec_global`，而它不入快照）⇒ 不结转会把利用率的
    分子（本段取食）与分母（全程名义再生×tick）**错配**（静默错读）。
    侧车缺失时 ⇒ 结转记为 False 并由产物自曝（`spatial_carry_ok`），**不假装**。
    """

    def __init__(self, e, sidecar: "Path | None" = None, *, carried_e: float = 0.0,
                 visited: "np.ndarray | None" = None, carry_ok: bool = True) -> None:
        self.n = int(e.world.n_cells)
        # 🔴 `uniform` 世界里 `_patch_mask is None`（ResourceField 只在 patchy 分支建掩码）
        #   ⇒ 必须归一成"无斑块"（**不是崩、也不是全 False 记 0**）：
        #   斑块类读数在 uniform 下 = **None（未适用）**（R120/DEL-7），
        #   而 `food_util_frac` 仍适用（分母 = 全世界名义再生，与 R190 的 A 臂口径 3153.1 同源）。
        raw = getattr(e.resources, "_patch_mask", None)
        if raw is None:
            self.patch = np.zeros(self.n, dtype=bool)
        else:
            self.patch = np.asarray(raw, dtype=bool)
            if self.patch.shape != (self.n,):       # 形状异常 ⇒ 同"无斑块"处理（可自证）
                self.patch = np.zeros(self.n, dtype=bool)
        self.has_patches = bool(self.patch.any())
        self.cap = np.asarray(e.resources._capacity, dtype=np.float64).copy()
        self.area_patch = int(self.patch.sum())
        # R190 口径：Σ名义再生取 **t=0 的名义值**（与静态 K 公式同一个量 ⇒ 可对账）
        self.sigma_regen = float(np.asarray(
            e.resources._regrowth_amount(0), dtype=np.float64).sum())
        self.visited = (np.zeros(self.n, dtype=bool) if visited is None else visited.copy())
        self.carried_e = float(carried_e)      # 前几段的取食入能量结转（净吸收口径）
        self.carry_ok = bool(carry_ok)
        self.sidecar = sidecar

    # ---- 每 tick ----
    def observe(self, e) -> None:
        """标记"本 tick 被生物占据过的格"（含非斑块格 —— 与 R191 工具一致）。

        ⚠️ 人口数用 `len(e._id)`（= a4 全脚本的约定；引擎不变式 `len(_flat) == len(_id)`
        也成立，但显式用 `_id` ⇒ 与"人为截断 _id 模拟灭绝"的测试语义一致）。
        """
        P = len(e._id)
        if P:
            self.visited[e._flat[:P]] = True

    # ---- 读数（采样点可调，纯读）----
    def _intake_e_total(self, e) -> float:
        led = (e.energy_ledger() or {}).get("global") or {}
        return self.carried_e + float(led.get("intake_forage_sum") or 0.0)

    def food_util_frac(self, e, tick: int) -> "float | None":
        """R190 口径：取食质量 ÷（Σ名义再生 × tick）。tick=0 ⇒ None（分母 0）。"""
        if tick <= 0 or self.sigma_regen <= 0.0:
            return None
        eff = float(getattr(e.config.organisms, "eat_efficiency", 3.0) or 3.0)
        ah = float(getattr(e.config.organisms, "assim_herb", 1.0) or 1.0)
        net_abs = eff * ah
        if net_abs <= 0.0:
            return None
        mass = self._intake_e_total(e) / net_abs
        return mass / (self.sigma_regen * tick)

    def sample(self, e, tick: int) -> dict:
        """采样点的五读数（GUD/在斑块/访问率 = 瞬时；利用率 = 累计）。"""
        P = len(e._id)
        stock = np.asarray(e.resources._grid, dtype=np.float64)
        cap_patch = float(self.cap[self.patch].sum())
        out = {
            "food_util_frac": self.food_util_frac(e, tick),
            # 🔴 无斑块世界（uniform / 掩码缺失）⇒ 斑块类读数 = None（**未适用**，
            #   不是 0）——否则 uniform 批会被读成"从不访问斑块"的假阴性（E 类缺陷）
            "patch_stock_frac": (float(stock[self.patch].sum()) / cap_patch
                                 if (self.has_patches and cap_patch > 0.0) else None),
            "patch_visit_frac": (float(self.visited[self.patch].sum()) / self.area_patch
                                 if self.has_patches and self.area_patch else None),
            "on_patch_frac": (float(np.mean(self.patch[e._flat[:P]]))
                              if (P and self.has_patches) else None),
            "gud_var": None,
            "gud_mean": None,
        }
        if P:
            cells = e._flat[:P]
            c = self.cap[cells]
            ok = c > 0.0                       # 容量 0 的格 GUD 无定义 ⇒ 剔除
            if ok.any():
                vals = stock[cells][ok] / c[ok]
                out["gud_mean"] = float(np.mean(vals))
                if vals.size >= 2:
                    out["gud_var"] = float(np.var(vals, ddof=1))
        return out

    # ---- 侧车（续跑结转）----
    def save_sidecar(self, e) -> "Path | None":
        if self.sidecar is None:
            return None
        self.sidecar.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            self.sidecar,
            visited=self.visited,
            carried_e=np.array(self.carried_e + float(
                ((e.energy_ledger() or {}).get("global") or {}).get(
                    "intake_forage_sum") or 0.0)),
        )
        return self.sidecar

    @classmethod
    def load_sidecar(cls, path: "Path", e) -> "SpatialReadings":
        d = np.load(path)
        vis = np.asarray(d["visited"], dtype=bool)
        if vis.shape != (e.world.n_cells,):
            # 世界几何变了（不该发生：续跑同快照）⇒ 不静默用错形状
            raise SystemExit(f"侧车 visited 形状 {vis.shape} ≠ 世界格数 {e.world.n_cells}")
        return cls(e, sidecar=path, carried_e=float(d["carried_e"]), visited=vis)

    def summary(self, e, tick: int) -> dict:
        s = self.sample(e, tick)
        p = len(e._id)
        return {
            **{k: (None if v is None else round(float(v), 6)) for k, v in s.items()},
            "sigma_regen_nominal": round(self.sigma_regen, 3),
            "patch_cells": self.area_patch,
            "has_patches": bool(self.has_patches),
            "visited_patch_cells": int(self.visited[self.patch].sum()),
            "n_individuals": p,
            "tick": int(tick),
            "food_util_numerator_mass": round(
                self._intake_e_total(e) / max(1e-9, float(
                    getattr(e.config.organisms, "eat_efficiency", 3.0) or 3.0)
                    * float(getattr(e.config.organisms, "assim_herb", 1.0) or 1.0)), 3),
            "spatial_carry_ok": bool(self.carry_ok),
            "note": SPATIAL_NOTE,
            "caliber_src": "R190（利用率）/ R191（访问率·在斑块）/ R192（GUD）/ R193 §四（S2）",
        }


# 🔴 F6（镜 19:46 审 / PI 697e171）：钩子丢失 = fail-loud 文案（窗末 drain 与收尾共用）。
#   丢失场景 = 快照续跑经 `cls(config)` 重建引擎 ⇒ `_rd_pred_kill_log` 静默回 None
#   ⇒ 其后每窗零记录、侧车前半有数后半空 —— 旧的"or []"写法把它伪装成"本窗无杀"。
_RD_HOOK_LOST_MSG = (
    "🔴 C1 D-1 钩子丢失（_rd_pred_kill_log 为 None）：快照续跑会经 cls(config) "
    "重建引擎 ⇒ 钩子静默回 None（F6）。rd 批与快照续跑互斥——请加 --fresh 重跑。"
)


class RdInstruments:
    """C1 捕食信息价值批 · 分块死亡明细侧车（D-2 骨架；设计稿 §四）。

    职责：
    - 分块映射：flat cell → block_id ∈ [0, block_rows×block_cols)
      ⚠️ 必须用 world.rows/cols 现算（**禁**复用主表读数段的 //120 硬编码，T13 锁）
    - 每窗末处理 D-1 钩子日志 → 侧车一（*_rd_windows.csv）逐行
      （钩子元组 = (id, 死亡格, tick)；F5 后含稳定身份，见 sphere_engine 两处 append）
    - 收尾聚合 → 侧车二（*_rd_run.csv）一行/run
    - risk 通道读数采样（臂 on 时；SmellField.at() 锚点）

    默认不实例化（--rd-instruments 关 ⇒ 零开销，T1/T12 锁）。
    """

    def __init__(self, e, *, channel: str = "risk",
                 block_rows: int = 4, block_cols: int = 4,
                 sample_every: int = 250,
                 windows_path: "Path | None" = None,
                 run_path: "Path | None" = None) -> None:
        self.channel = str(channel)
        self.block_rows = int(block_rows)
        self.block_cols = int(block_cols)
        self.n_blocks = self.block_rows * self.block_cols
        self.sample_every = int(sample_every)
        self.windows_path = windows_path
        self.run_path = run_path
        # 世界几何（T13：必须从 world 取，禁 //120）
        self._rows = int(e.world.rows)
        self._cols = int(e.world.cols)
        self._n_cells = int(e.world.n_cells)
        # 预计算 flat → block_id 映射表（一次性 O(n_cells)）
        self._block_map = self._build_block_map()
        # 侧车一：逐窗行缓冲
        self._win_fh = None
        self._win_writer = None
        self._win_idx = 0
        # 侧车二：逐 run 聚合缓冲
        self._run_rows = []  # list[dict]，收尾一次写
        # 总 kill 计数（manifest 对账用）
        self.total_kills = 0
        # 窗边界采样用（occ 窗均估计）
        self._prev_pop = None
        self._prev_block_pop = None  # 块级 occ（per-block 窗均估计）

    def _build_block_map(self) -> np.ndarray:
        """flat cell index → block_id ∈ [0, n_blocks)。T4/T13 锚。"""
        flat_idx = np.arange(self._n_cells, dtype=np.int64)
        row = flat_idx // self._cols
        col = flat_idx % self._cols
        lat_band = (row * self.block_rows) // self._rows
        lon_band = (col * self.block_cols) // self._cols
        return (lat_band * self.block_cols + lon_band).astype(np.int32)

    def install_hook(self, e) -> None:
        """安装 D-1 钩子（载体侧）。T11a/T11b/T14 锁。"""
        # fail-loud：Rust 路径不得装（D-1 只在 Python 捕食段）
        if getattr(e, "_sim_core", None) is not None:
            raise SystemExit(
                "C1 D-1 钩子只在 Python 路径生效；"
                "--rd-instruments + use_sim_core=True 不兼容（§八 D2 纪律）"
            )
        e._rd_pred_kill_log = []

    def open_sidecars(self) -> None:
        """打开侧车一 CSV 写器。"""
        if self.windows_path is None:
            return
        self.windows_path.parent.mkdir(parents=True, exist_ok=True)
        self._win_fh = self.windows_path.open("w", encoding="utf-8", newline="")
        cols = (["seed", "arm", "win_idx", "win_start_tick", "win_end_tick"]
                + [f"kills_{b}" for b in range(self.n_blocks)]
                + [f"occ_{b}" for b in range(self.n_blocks)]
                + [f"dens_{b}" for b in range(self.n_blocks)]
                # 🔴 F7（镜 19:46 审 / PI 697e171）：`kills_cum_total` = 含本窗累计。
                #   主表 d_pred 是**累计口径**（log-interval=1000t 粒度）⇒ 对账粒度
                #   = 每 4 窗（250t×4）一行，**不可**逐窗对（见 §五 T5b 改稿）。
                + ["win_kills_total", "kills_cum_total",
                   "win_pop_start", "win_pop_end", "risk_read_n"])
        self._win_writer = csv.DictWriter(self._win_fh, fieldnames=cols)
        self._win_writer.writeheader()

    def process_window(self, e, tick: int, *, seed: int, arm: str) -> None:
        """窗末处理：排空 D-1 日志 → 写侧车一行。"""
        log = getattr(e, "_rd_pred_kill_log", None)
        # 🔴 F6（镜 19:46 审 / PI 697e171）：None = 钩子丢失（快照续跑重建引擎）
        #   ⇒ fail-loud。**不得**再写 `or []`——那会把"后半程零记录"伪装成
        #   "本窗无杀"（静默污染 J-A/J-B 的窗序列）。
        if log is None:
            raise SystemExit(_RD_HOOK_LOST_MSG)
        # 分块 kills（F5 后元组 = (id, 死亡格, tick)；本窗只消费死亡格）
        kills = np.zeros(self.n_blocks, dtype=np.int64)
        if log:
            flat_arr = np.array([cell for _, cell, _t in log], dtype=np.int64)
            blocks = self._block_map[flat_arr]
            np.add.at(kills, blocks, 1)
        # 当前种群 + 块级 occupancy
        P = len(e._id)
        cur_pop = P
        cur_block_pop = np.zeros(self.n_blocks, dtype=np.float64)
        if P:
            block_of_each = self._block_map[e._flat[:P]]
            np.add.at(cur_block_pop, block_of_each, 1.0)
        prev_pop = self._prev_pop if self._prev_pop is not None else cur_pop
        prev_block_pop = (self._prev_block_pop
                          if self._prev_block_pop is not None
                          else cur_block_pop)
        # 块级 occ 窗均估计 = (窗起 + 窗止)/2（§四-4-2 [口径]）
        occ_block = (prev_block_pop + cur_block_pop) / 2.0
        # 密度 = kills / max(occ × sample_every, 1)
        dens = np.where(
            occ_block > 0,
            kills / (occ_block * self.sample_every),
            np.nan,
        )
        # risk 读数采样数（臂 on 时 = P；off 时 = 0）
        risk_read_n = P if self._smell_channel_active(e) else 0
        # 写行
        win_start = max(0, tick - self.sample_every + 1)
        row = {
            "seed": seed, "arm": arm,
            "win_idx": self._win_idx,
            "win_start_tick": win_start,
            "win_end_tick": tick,
            **{f"kills_{b}": int(kills[b]) for b in range(self.n_blocks)},
            **{f"occ_{b}": round(float(occ_block[b]), 2)
               for b in range(self.n_blocks)},
            **{f"dens_{b}": (round(float(dens[b]), 6)
                             if not np.isnan(dens[b]) else "")
               for b in range(self.n_blocks)},
            "win_kills_total": int(kills.sum()),
            # 🔴 F7：含本窗的**累计**（主表 d_pred 差分对账锚；每 4 窗=1000t 一行）
            "kills_cum_total": int(self.total_kills) + int(kills.sum()),
            "win_pop_start": int(prev_pop),
            "win_pop_end": int(cur_pop),
            "risk_read_n": int(risk_read_n),
        }
        if self._win_writer is not None:
            self._win_writer.writerow(row)
            self._win_fh.flush()
        self._run_rows.append({
            "win_idx": self._win_idx, "kills": kills.copy(),
            "occ_block": occ_block.copy(), "dens": dens.copy(),
        })
        self.total_kills += int(kills.sum())
        self._win_idx += 1
        self._prev_pop = cur_pop
        self._prev_block_pop = cur_block_pop.copy()
        # 排空日志（O(窗内死亡数) 内存）
        log.clear()

    def _smell_channel_active(self, e) -> bool:
        """检查 rd-channel 是否在引擎 smell channels 中。"""
        channels = tuple(getattr(e.config.smell, "channels", ()) or ())
        return self.channel in channels

    def close(self) -> None:
        """关闭侧车一写器。"""
        if self._win_fh is not None:
            self._win_fh.close()
            self._win_fh = None

    def write_run_summary(self, *, seed: int, arm: str) -> None:
        """写侧车二（逐 run 聚合，一行）。§四-4-3。"""
        if self.run_path is None or not self._run_rows:
            return
        self.run_path.parent.mkdir(parents=True, exist_ok=True)
        # AR1 per block（跨窗 dens 序列 lag-1 ρ）
        ar1_vals = {}
        for b in range(self.n_blocks):
            dens_seq = np.array([r["dens"][b] for r in self._run_rows],
                                dtype=np.float64)
            valid = dens_seq[~np.isnan(dens_seq)]
            if len(valid) >= 3:
                ar1_vals[b] = float(np.corrcoef(valid[:-1], valid[1:])[0, 1])
            else:
                ar1_vals[b] = float("nan")
        # 块排序稳定性（跨窗 Spearman）
        stability = float("nan")
        if len(self._run_rows) >= 3:
            from scipy.stats import spearmanr as _spearman  # 惰性导入
            rho_sum = 0.0
            rho_n = 0
            for i in range(len(self._run_rows) - 1):
                d1 = self._run_rows[i]["dens"]
                d2 = self._run_rows[i + 1]["dens"]
                mask = ~(np.isnan(d1) | np.isnan(d2))
                if mask.sum() >= 3:
                    rho, _ = _spearman(d1[mask], d2[mask])
                    if not np.isnan(rho):
                        rho_sum += rho
                        rho_n += 1
            if rho_n > 0:
                stability = rho_sum / rho_n
        # CV per window（16 块 dens 的 CV，跨窗均值）
        cv_vals = []
        for r in self._run_rows:
            d = r["dens"]
            valid = d[~np.isnan(d)]
            if len(valid) >= 2 and np.mean(valid) > 0:
                cv_vals.append(float(np.std(valid, ddof=1) / max(np.mean(valid), 1e-9)))
        cv_mean = float(np.mean(cv_vals)) if cv_vals else float("nan")
        row = {
            "seed": seed, "arm": arm,
            "rd_n_windows": len(self._run_rows),
            "rd_block_rows": self.block_rows,
            "rd_block_cols": self.block_cols,
            "rd_channel": self.channel,
            "rd_total_kills": self.total_kills,
            "rd_block_cv": round(cv_mean, 6) if not np.isnan(cv_mean) else "",
            "rd_block_stability": (round(stability, 6)
                                   if not np.isnan(stability) else ""),
            **{f"rd_block_ar1_{b}": (round(ar1_vals[b], 6)
                                     if not np.isnan(ar1_vals[b]) else "")
               for b in range(self.n_blocks)},
        }
        cols = list(row.keys())
        with self.run_path.open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerow(row)


def _moments(x) -> tuple[float, float, float]:
    """样本标准差 / 偏度 / **超额**峰度（矩法）—— BC 双峰系数的输入（R135 第 -1 步①）。

    BC = (skew² + 1) / (kurt + 3(n-1)²/((n-2)(n-3)))，BC > 5/9 提示双峰。
    ⚠️ 调用方须先保证 **n ≥ 3**：峰度是四阶矩，样本过小时方差极大（可有可无的天文数字）。
    """
    a = np.asarray(x, dtype=np.float64)
    n = int(a.size)
    if n < 3:
        return 0.0, 0.0, 0.0
    d = a - a.mean()
    m2 = float((d ** 2).mean())
    m3 = float((d ** 3).mean())
    m4 = float((d ** 4).mean())
    if m2 <= 1e-300:          # 退化分布（所有值相同）
        return 0.0, 0.0, 0.0
    std = float(np.sqrt(m2 * n / (n - 1)))          # ddof=1
    skew = m3 / (m2 ** 1.5)
    kurt = m4 / (m2 ** 2) - 3.0                     # 超额峰度（正态 = 0）
    return round(std, 4), round(skew, 4), round(kurt, 4)


def build(mode: str, codebook: bool, seed: int, ticks: int, *,
          max_count: int = 5000, neutral: bool = False,
          sig_disabled: bool = False, oracle: bool = False,
          measure: bool = False, oracle_donation: float | None = None,
          oracle_persistence: int | None = None,
          gain_multiplier: float | None = None,
          calibration_arm: bool = False,
          signal_mode: str | None = None,
          signal_alphabet: str | None = None,
          gate_mode: str | None = None, gate_delta: str | None = None,
          distribution: str | None = None,
          # A′ 记忆朝向梯度（2026-09-19）：`orientation` 才启用；默认 `none` = 原式（逐位等价）
          memory_gradient: str = "none",
          memory_gradient_gain: float = 0.3,
          # PC-1（R134，2026-09-20）：S2 关捕食 / S1 软顶 / 零模型臂的瓶颈开关
          predation_enabled: bool = True,
          soft_cap_target: float = 0.0,
          learning_bottleneck: bool = True,
          reputation_weight: float = 0.0,
          # R135 第3步 A-连续（2026-09-20）：凸 trade-off 取食倍率 (1−g16)^k
          forage_tradeoff_k: float = 0.0,
          # R152/P0（2026-09-22）：捕食生态位结构 6 参（**默认值 = 旧行为，逐位一致**）
          attack_cost: float = 0.1, transfer_ratio: float = 0.4,
          attack_gene_gate: float = 0.3, attack_prob_coef: float = 0.2,
          success_floor: float = 0.1, success_ceil: float = 0.9,
          # R141 P0-2（派工单 §二）：g16 初始投放（"" = 旧行为；"0.05,0.5,0.9" = 校准批）
          init_g16_clusters: str = "",
          # R144/R145：能量封顶开关（默认 False = 与 E-017~E-031/calib1 可比）
          energy_cap: bool = False,
          # 13.7 季节（迁移 S4/S5）：tilt_deg（度）+ season_period（tick）
          #   None/None（默认）⇒ 不覆盖 config ⇒ 关档逐位等价
          tilt_deg: float | None = None,
          season_period: int | None = None,
          # 🔴 R196 光驱动再生（迁移驱动源）：light_sensitivity>0 ⇒ 再生乘光照因子
          #   None（默认）⇒ 不覆盖 config（= 0.0）⇒ 逐位等价
          light_sensitivity: float | None = None,
          light_normalize: bool | None = None,
          # R145 §七.1：光合产能覆盖（`photo_max`；None = 不覆盖）——供配对臂用
          photo_max: float | None = None,
          # 🔴 13.5 ③（2026-09-24，fish 批准）：能量标定 7 参
          #   **默认值一律 = 现状行为**（⇒ 关档仍逐位等价，C7 基线不动）
          stomach_cap_mass: float = 0.0,     # 独立胃容量（0 = 沿用旧公式 max_energy/3*0.5）
          eat_threshold_frac: float = 0.0,   # 胃 ≥ 该比例×容量 就不吃（0 = 旧：没满就吃）
          starve_frac: float = 0.0,          # 饿死阈值（能量<比例 且 胃空；0 = 旧：仅 energy<=0）
          exhaust_frac: float = 0.0,         # 力竭阈值（能量<比例，无论胃里有没有食；0 = 关）
          assim_herb: float = 1.0,           # 素食吸收率（1.0 = 无损失）
          assim_carn: float = 1.0,           # 肉食（尸体腿）吸收率
          assim_return_frac: float = 1.0,    # 未吸收部分回流本格的比例（assim=1 时无作用）
          # 🔴 13.5 ①②（2026-09-24）：食物绑定 —— 背景产能归零 + 斑块再生倍率
          bg_production_zero: bool = False,  # 背景容量/再生/存量三者归零（只留斑块生产）
          patch_regrowth_mult: float | None = None,   # None = 用 config 默认
          # 🔴 13.6 S1（2026-09-24，R193 派工）：三种地形的几何参数（**默认 = config 现状**
          #   ⇒ 不传时逐位等价）。地形 = count×radius×mult 三参数组合（设计稿 §一）：
          #     森林 12/3/1.62（少而大 ⇒ 聚集）｜草原 60/1/1.06（多而小 ⇒ 分散）｜
          #     荒漠 10/1/2.47（少而小 ⇒ 难找）。`patch_capacity_mult` 默认 None = 不覆盖
          #   （config 3.0）—— 只有需要改**容量**倍率时才传（地形表只用再生倍率定 K）。
          patch_count: int = 30,             # 斑块中心数（config 默认 30）
          patch_radius: int = 2,             # 斑块半径（config 默认 2）
          patch_capacity_mult: float | None = None,   # None = 用 config 默认（3.0）
          # 🔴 R187（2026-09-24）：`eat_efficiency` 语义澄清为**完全燃烧值**。
          #   13.5 必须与吸收率**同批**传：7.5 × 0.4 = 3.0 = 现状净吸收（⇒ 不灭绝）；
          #   单传 assim=0.4 会让净吸收腰斩 ⇒ 低代谢个体赤字 ⇒ 连锁灭绝（R187 实测）。
          eat_efficiency: float | None = None,        # None = 用 config 默认（3.0）
          # 🔴 13.5 参数联动（R188 冒烟）：改死亡阈值必须同步改初始能量，否则开局集体饿死
          max_energy: float | None = None,            # 体能上限（None = config 默认 300）
          initial_energy: float | None = None,        # 初始能量（None = config 默认 60）
          # R146/R149 L1/L2（R150 B1/B2；**默认全关 = 旧行为**，逐位等价）
          l1_seek: bool = False, l1_fear: bool = False, l2_dash: bool = False,
          w_seek_max: float = 0.5, w_fear: float = 0.5,
          l1_prey_mode: str = "lowagg",
          # S1 骨架（设计稿 §三/§5.2）：尸体—食腐 + 血条—受伤（**默认全关 = 旧行为**，
          # 逐位等价；机制未接线，开关仅建字段 + 读回）
          corpse_enabled: bool = False, corpse_energy_frac: float = 0.9,
          corpse_decay_ticks: int = 600, corpse_to_plant_frac: float = 0.5,
          corpse_patch_boost: float = 0.5, corpse_cap_per_cell: int = 200,
          scav_gate: float = 0.5, scav_s: float = 2.0,
          wound_enabled: bool = False, wound_base: float = 0.35,
          wound_heal_rate: float = 0.001, wound_heal_energy_cost: float = 0.05,
          contest_enabled: bool = False, holder_adv: float = 1.2,
          escalation_gap: float = 0.25, contest_cost_energy: float = 0.5,
          w_fear_health: float = 0.5, need_aggression_k: float = 0.5,
          wound_fear_threshold: float = 0.3,
          # 13.4 波 1（设计稿 §二/§三）：亚格连续坐标（**默认全关 = 旧行为**，逐位等价）
          subpos_enabled: bool = False, subdiv: int = 4,
          speed_gain: float = 4.0, speed_max: float = 2.0,
          min_energy_frac: float = 0.05, lat_floor: float = 0.3,
          stay_base: float = 0.0, stay_food_k: float = 0.0,
          stay_signal_k: float = 0.0, stay_fear_k: float = 0.0,
          stay_max: float = 0.8,
          # 13.4 波 2A（T2，R178）：资源动态（斑块休耕—死亡—轮作；**默认关 = 旧行为**）
          resource_dynamics_enabled: bool = False,
          rest_ticks: int = 60,
          rest_threshold: float = 0.3,
          death_threshold: float = 0.8,
          damage_recovery: float = 0.5,
          kill_frac: float = 10.0,
          kill_denom: str = "regrowth", dead_regen_ticks: int = 2000,
          dead_cell_max_frac: float = 0.5,
          kill_patch_only: bool = True, rotate_same_row_only: bool = True,
          # 13.4 波 2B（T3）：视野 / 单格上限 / 社交归一化（默认 = 旧行为）
          perception_span: int = 1, cell_occupancy_cap: int = 3,
          cell_occupancy_cap_enabled: bool = False,
          social_norm: str = "auto",
          # 🔴 13.8 日历—罗盘式定向迁徙（g23；**默认关 = 旧行为逐位等价**）
          #   `migration_enabled=False`（默认）⇒ 移动段整段不执行 ⇒ C7 基线不动。
          migration_enabled: bool = False,
          migration_gain: float = 50.0,
          migration_min_abs_anomaly: float = 0.0,
          # 🔴 R247 饥饿调制（HM；**默认关 = 旧行为逐位等价**）
          #   ① 走停腿 `p_eff = clip(p × (1 + α·h_norm), 0, 1)`（三处同式含 Rust）
          #   ② 感知腿 `perc_eff = perc × (1 + β·h_norm)`（三处同式含 Rust）
          hunger_mod_enabled: bool = False,
          hunger_alpha: float = 0.5, hunger_beta: float = 0.5,
          hunger_h_mid: float = 0.5,
          hunger_stay_gain: float = 0.0,
          # 🔴 R239/R258 ASM 模式仲裁（**默认 fusion = 现状逐位等价**）
          #   `arbitration` 需 `smell.channels ⊇ {food,risk,kin}`（引擎构造期 fail-loud）；
          #   与 L1/L2/HM ②/v2/迁徙/ARS/softmax/噪声/占位上限等**互斥**（同由引擎拦）。
          asm_mode: str = "fusion",
          asm_base_explore: float = 0.2, asm_hyst: float = 0.15,
          asm_hold_ticks: int = 20,
          asm_w_feed: float = 1.0, asm_w_hunger: float = 0.5,
          asm_w_flee: float = 1.0, asm_w_join: float = 0.5,
          # 🔴 R240 T8 气味场通道（空 = 关 ⇒ 旧行为逐位等价）
          smell_channels: str = "",
          # 🔴 F2 装置面（镜 19:46 审 / PI 697e171；**默认 None ⇒ 不覆盖 ⇒ 逐位等价**）
          #   --device 决议值的落点；口径 = steady_k_probe.make_cfg 同源。
          rows: int | None = None, cols: int | None = None,
          pop: int | None = None,
          bg_low_prod_frac: float | None = None,
          bg_low_cap_mult: float | None = None) -> SphereEngine:
    c = SimConfig(seed=seed)
    c.simulation.ticks = ticks
    c.simulation.use_sim_core = False          # D2 须走 Python 路径（AGENTS.md）
    c.simulation.history_limit = 100           # 环形缓冲，限内存（不改变语义）
    c.population.initial_count = 200           # R4 manifest 真实口径
    c.population.max_count = max_count         # R41：标杆批口径 3240（⑤ 不饱和前提）
    # 🔴 F2：世界尺度 + 初始投放（0 参 ⇒ 不触 ⇒ 逐位等价）
    #   rows/cols 走构造后赋值（同 steady_k_probe.make_cfg:262 口径）；赋值绕过
    #   __post_init__ ⇒ 显式复刻两条断言（F1 同型教训："赋值不校验"）。
    if rows is not None:
        c.world.rows = int(rows)
    if cols is not None:
        c.world.cols = int(cols)
    if rows is not None or cols is not None:
        assert c.world.rows >= 3, "至少要 3 行（上下极 + 至少一行赤道带）"
        assert c.world.cols >= 4, "经度至少 4 列"
    if pop is not None:
        assert int(pop) <= int(max_count), (
            f"initial_count {pop} > max_count {max_count}：引擎按 initial_count "
            "**实际投放**（max_count 只卡繁殖）⇒ 开局即超上限，饱和前提被静默改变")
        c.population.initial_count = int(pop)
    # PC-1（R134）：S1 软顶（目标窗形式，冒烟后修订）/ S2 关捕食（默认 = 旧行为）
    c.population.soft_cap_target = float(soft_cap_target)
    # ⚠️ 这里是**整体替换** PredationConfig ⇒ **必须**把 k 一并传进去，
    #    否则 A-连续的凸度会被静默重置为 0（F1 同型：传了开关却没生效）。
    c.predation = PredationConfig(
        enabled=bool(predation_enabled),
        forage_tradeoff_k=float(forage_tradeoff_k),
        # R152/P0：捕食生态位结构 6 参（**整体替换 ⇒ 必须一次传全**，漏一个就被静默重置为默认）
        attack_cost=float(attack_cost),
        transfer_ratio=float(transfer_ratio),
        attack_gene_gate=float(attack_gene_gate),
        attack_prob_coef=float(attack_prob_coef),
        success_floor=float(success_floor),
        success_ceil=float(success_ceil),
    )
    c.genome.init_g16_clusters = str(init_g16_clusters or "")   # R141 P0-2
    c.organisms.energy_cap_enabled = bool(energy_cap)           # R144：能量封顶（新纪元开关）
    if photo_max is not None:
        c.organisms.photo_max = float(photo_max)                # R145：光合配对臂
    # 🔴 13.5 ③（2026-09-24）：能量标定 —— 默认值 = 现状 ⇒ 关档逐位等价
    _ocfg = c.organisms
    _ocfg.stomach_cap_mass = float(stomach_cap_mass)            # 独立胃容量（>0 才生效）
    _ocfg.eat_threshold_frac = float(eat_threshold_frac)        # 胃≥比例×容量 就不吃
    _ocfg.starve_frac = float(starve_frac)                      # 饿死阈值（且胃空）
    _ocfg.exhaust_frac = float(exhaust_frac)                    # 力竭阈值（无论胃）
    _ocfg.assim_herb = float(assim_herb)                        # 素食吸收率
    _ocfg.assim_carn = float(assim_carn)                        # 肉食吸收率
    _ocfg.assim_return_frac = float(assim_return_frac)          # 未吸收回流比例
    if eat_efficiency is not None:
        _ocfg.eat_efficiency = float(eat_efficiency)            # R187：完全燃烧值
    # 🔴 13.5 参数联动（R188）：阈值改了 ⇒ 初始能量必须跟着改（否则开局集体饿死）
    if max_energy is not None:
        _ocfg.max_energy = float(max_energy)
    if initial_energy is not None:
        _ocfg.initial_energy = float(initial_energy)
    # 🔴 13.7 季节（派工：迁移 S4/S5）——默认 0/0 = 关 ⇒ 关档逐位等价
    if tilt_deg is not None:
        c.light.tilt_rad = float(np.deg2rad(float(tilt_deg)))
    if season_period is not None:
        c.light.season_period = int(season_period)
    # 🔴 R196 光驱动再生 —— 默认 None = 不覆盖（保持 0.0）⇒ 逐位等价
    if light_sensitivity is not None:
        c.resources.light_sensitivity = float(light_sensitivity)
    if light_normalize is not None:
        c.resources.light_normalize = bool(light_normalize)
    # 🔴 13.8 日历—罗盘式定向迁徙（g23）—— 默认关 ⇒ 移动段整段不执行 ⇒ 逐位等价（C7）。
    #   M1（enabled ∧ use_sim_core）与 M2（enabled ∧ 无季节）都由**引擎构造期**硬报错，
    #   这里不重复判（少一处逻辑 = 少一个不一致的机会）。
    c.migration.enabled = bool(migration_enabled)
    c.migration.gain = float(migration_gain)
    c.migration.min_abs_anomaly = float(migration_min_abs_anomaly)
    # 🔴 R247 饥饿调制（HM）—— 默认关 ⇒ 旧行为逐位等价（C7）。
    #   ① 走/停腿与 ② 感知腿**三处同式**（Python 参考 / Rust / 向量化快路径）；
    #   与 `use_sim_core` **无互斥**（Rust 侧已同式实现并重编，声明见交付贴）。
    #   ⚠️ `stay_gain` 为**预留字段**（subpos 驻留调制，当前未接线）⇒ 仅进指纹/读回。
    c.hunger_mod = HungerModConfig(
        enabled=bool(hunger_mod_enabled),
        alpha=float(hunger_alpha),
        beta=float(hunger_beta),
        h_mid=float(hunger_h_mid),
        stay_gain=float(hunger_stay_gain),
    )
    # 🔴 R239/R258 ASM 模式仲裁 —— 默认 `fusion` ⇒ 移动段走原融合式 ⇒ 逐位等价（C7）。
    #   `arbitration` 的**前置条件**（气味通道 food/risk/kin）与**互斥表**由引擎构造期
    #   fail-loud 拦（本处不重复判：少一处逻辑 = 少一个不一致的机会，同 13.8 口径）。
    #   ⚠️ 时间压缩（k）下 `hold_ticks` 是 DURATION（÷k），其余权重无量纲（tick 面额清点见
    #   `tools/tick_denomination_audit.py`）。
    c.action_selection = ActionSelectionConfig(
        mode=str(asm_mode),
        base_explore=float(asm_base_explore),
        hyst=float(asm_hyst),
        hold_ticks=int(asm_hold_ticks),
        w_feed=float(asm_w_feed),
        w_hunger=float(asm_w_hunger),
        w_flee=float(asm_w_flee),
        w_join=float(asm_w_join),
    )
    # 🔴 R240 T8 气味场（默认空元组 = 全关 ⇒ 不建场、不消费 ⇒ 逐位等价）。
    #   仲裁档的 salience 输入 = Ŝ_food/Ŝ_risk/Ŝ_kin ⇒ 臂配置里必须显式给三通道。
    if str(smell_channels).strip():
        c.smell.channels = tuple(
            s.strip() for s in str(smell_channels).split(",") if s.strip())
    # R146/R149 L1/L2（R150 B1/B2）：**默认全关 ⇒ 旧行为**（H1 逐位等价，C7 已钉死）
    c.simulation.l1_seek = bool(l1_seek)
    c.simulation.l1_fear = bool(l1_fear)
    c.simulation.l2_dash = bool(l2_dash)
    c.simulation.w_seek_max = float(w_seek_max)
    c.simulation.w_fear = float(w_fear)
    c.simulation.l1_prey_mode = str(l1_prey_mode)
    # S1 骨架（设计稿 §三）：尸体—食腐 + 血条—受伤，**整体替换** CorpseWoundConfig
    # （默认全关 = 旧行为，逐位等价；机制未接线 ⇒ 开关只进指纹/读回）
    c.corpse_wound = CorpseWoundConfig(
        corpse_enabled=bool(corpse_enabled),
        corpse_energy_frac=float(corpse_energy_frac),
        corpse_decay_ticks=int(corpse_decay_ticks),
        corpse_to_plant_frac=float(corpse_to_plant_frac),
        corpse_patch_boost=float(corpse_patch_boost),
        corpse_cap_per_cell=int(corpse_cap_per_cell),
        scav_gate=float(scav_gate),
        scav_s=float(scav_s),
        wound_enabled=bool(wound_enabled),
        wound_base=float(wound_base),
        wound_heal_rate=float(wound_heal_rate),
        wound_heal_energy_cost=float(wound_heal_energy_cost),
        contest_enabled=bool(contest_enabled),
        holder_adv=float(holder_adv),
        escalation_gap=float(escalation_gap),
        contest_cost_energy=float(contest_cost_energy),
        w_fear_health=float(w_fear_health),
        need_aggression_k=float(need_aggression_k),
        wound_fear_threshold=float(wound_fear_threshold),
    )
    # 13.4 波 1：亚格连续坐标（**整体替换** SubposConfig ⇒ 须一次传全，防漏传静默重置）
    c.subpos = SubposConfig(
        enabled=bool(subpos_enabled),
        subdiv=int(subdiv),
        speed_gain=float(speed_gain),
        speed_max=float(speed_max),
        min_energy_frac=float(min_energy_frac),
        lat_floor=float(lat_floor),
        stay_base=float(stay_base),
        stay_food_k=float(stay_food_k),
        stay_signal_k=float(stay_signal_k),
        stay_fear_k=float(stay_fear_k),
        stay_max=float(stay_max),
    )
    # 13.4 波 2A（T2，R178；波2 修 v2 三态机阈值）：资源动态（**整体替换** ResourceDynamicsConfig ⇒ 一次传全）
    c.resource_dynamics = ResourceDynamicsConfig(
        enabled=bool(resource_dynamics_enabled),
        rest_ticks=int(rest_ticks),
        rest_threshold=float(rest_threshold),
        death_threshold=float(death_threshold),
        damage_recovery=float(damage_recovery),
        kill_frac=float(kill_frac),
        kill_denom=str(kill_denom),
        dead_regen_ticks=int(dead_regen_ticks),
        dead_cell_max_frac=float(dead_cell_max_frac),
        kill_patch_only=bool(kill_patch_only),
        rotate_same_row_only=bool(rotate_same_row_only),
    )
    # 13.4 波 2B（T3）：视野 / 单格上限 / 社交归一化（默认 = 旧行为）
    c.simulation.perception_span = int(perception_span)
    c.simulation.cell_occupancy_cap = int(cell_occupancy_cap)
    c.simulation.cell_occupancy_cap_enabled = bool(cell_occupancy_cap_enabled)
    c.simulation.social_norm = str(social_norm)
    # 2026-09-19（R127/C8）：**原为硬编码 "uniform"**（注释"R4 manifest 真实口径"）——
    # 该硬编码使 C1a / C1b / C2 / α / gate / α8 **全部跑在 uniform 世界**
    # （容量只随纬度变 ⇒ 食物位置**可由位置预测** ⇒ 信息本不值钱），**且无任何告警**：
    # C4 对账"开关 vs 设计"时两边都是 uniform ⇒ 判通过。⇒ 见 R127 的 **C8 前提对账**。
    # 现改为**可配**：默认仍 "uniform"（与历史批可比）；"patchy" 须**显式指定**且**先过前提冒烟**。
    c.resources.distribution = str(distribution) if distribution else "uniform"
    # 🔴 13.5 ①②（2026-09-24）：食物绑定两旋钮 —— 必须在 `distribution` 之后设置
    #   （两者只在 patchy 分支有意义；uniform 下 bg_production_zero 无作用）
    c.resources.bg_production_zero = bool(bg_production_zero)
    if patch_regrowth_mult is not None:
        c.resources.patch_regrowth_mult = float(patch_regrowth_mult)
    # 🔴 F2：背景低产能带两参（R320/R326 装置口径；None ⇒ 不覆盖 ⇒ 逐位等价）
    #   ⚠️ 字段名不对称：`bg_low_cap_mult` → `resources.bg_cap_mult`
    #   （同 steady_k_probe.make_cfg:271 口径，勿照名直写）。
    if bg_low_prod_frac is not None:
        c.resources.bg_low_prod_frac = float(bg_low_prod_frac)
    if bg_low_cap_mult is not None:
        c.resources.bg_cap_mult = float(bg_low_cap_mult)
    # 🔴 13.6 S1（2026-09-24）：地形几何三参数（**默认 = config 现状 ⇒ 不传逐位等价**）
    #   ⚠️ 这里用构造后赋值：`ResourceConfig.__post_init__` 的校验（count≥1 / radius≥1 /
    #      capacity_mult>1）只在构造期跑 ⇒ 赋值不触发校验。⇒ 本段**显式复刻**那三条断言，
    #      让非法地形参数在 a4 层就 fail-loud（F1 同型教训："赋值绕过 __post_init__"）。
    c.resources.patch_count = int(patch_count)
    c.resources.patch_radius = int(patch_radius)
    if patch_capacity_mult is not None:
        c.resources.patch_capacity_mult = float(patch_capacity_mult)
    assert c.resources.patch_count >= 1, "patch_count 至少 1"
    assert c.resources.patch_radius >= 1, "patch_radius 至少 1"
    assert c.resources.patch_capacity_mult > 1.0, "patch_capacity_mult 必须 > 1（否则无富集）"
    # A′ 走向构造参数（而非构造后赋值）：`__post_init__` 只在构造时跑 ⇒ 赋值不会校验（F1 同型教训）
    d2 = InfoStructureConfig(
        enabled=True,
        learning_bottleneck=bool(learning_bottleneck),
        memory_gradient=str(memory_gradient),
        memory_gradient_gain=float(memory_gradient_gain),
        reputation_weight=float(reputation_weight),   # PC-1：C3 人为拉满（R134 裁定 1.0）
    )
    d2.learning_rate = 0.05
    d2.arbitrary_codebook = codebook           # R19/V12：4 机制=True，3 机制对照=False
    d2.steels_alignment = True
    d2.measure_signal_response = measure       # D-18 ⑥ 探针（纯观测）
    if mode == "off":                          # 对称对照：全感知/无噪声/argmax
        d2.perception_radius = 8
        d2.perception_noise = 0.0
        d2.softmax_tau = 0.0
    c.info_structure = d2
    c.neutral_genes = neutral                  # 漂变零模型（冻结 g14/g15）
    c.signal_disabled = sig_disabled           # 信号禁用臂
    c.oracle.enabled = oracle                  # D-8 oracle 正向对照臂
    # D-27④-A：剂量扫描（None ⇒ 沿用配置默认，行为与旧版逐位一致）
    if oracle_donation is not None:
        c.oracle.donation = float(oracle_donation)
    if oracle_persistence is not None:
        c.oracle.persistence = int(oracle_persistence)
    # R97 增益校准档（2026-09-16）：**先登记旗、再设 m**，并在此显式做 C5 v2 自洽检查
    # （dataclass 构造后赋值不触发 __post_init__ ⇒ 不能只依赖它；F1 教训同型）。
    if calibration_arm:
        c.oracle.is_calibration_arm = True
    if gain_multiplier is not None:
        c.oracle.gain_multiplier = float(gain_multiplier)
    if c.oracle.gain_multiplier != 1.0 and not c.oracle.is_calibration_arm:
        raise SystemExit(
            f"C5 v2 硬失败：gain_multiplier={c.oracle.gain_multiplier} ≠ 1.0 "
            f"但未登记校准臂（须 --calibration-arm）"
        )
    if c.oracle.is_calibration_arm:
        lo, hi = CALIBRATION_M_RANGE
        _mm = float(c.oracle.gain_multiplier)
        if not (_mm == 1.0 or lo <= _mm <= hi):
            raise SystemExit(
                f"C5 v2 硬失败：校准臂 gain_multiplier={_mm} 只允许 1.0（配对基线）"
                f"或落在 [{lo}, {hi}]（处理臂）"
            )
    # R100 条件 7（随机信号自检）：`signal_mode="random"` ⇒ 信号与个体状态无关（**无信息**）
    # ⇒ 若 ratio/ρ 同样上升，即实证「增益不依赖信号内容」（V-1 C-4 的可执行检验）。
    if signal_mode is not None:
        c.signal_mode = str(signal_mode)
    # R113/R121 信号字母表（默认 "16" = 现状；"4" = 仅能量 2 位）
    # R123/B② 门控臂（付款闸；仪器 ⇒ 必须登记校准臂）
    if gate_mode is not None:
        c.oracle.gate_mode = str(gate_mode)
    if gate_delta is not None:
        c.oracle.gate_delta = str(gate_delta)
    if c.oracle.gate_mode == "delta_positive" and not c.oracle.is_calibration_arm:
        raise SystemExit(
            "门控臂是仪器（改付款规则）⇒ 必须同时给 --calibration-arm（R100 条件 5）"
        )
    if c.oracle.gate_mode == "delta_positive" and not c.info_structure.measure_signal_response:
        raise SystemExit(
            "门控臂需要 measure_signal_response=True（否则 Δ≡0 ⇒ 付款全消失 = 假结论源）"
        )
    if signal_alphabet is not None:
        _sa = str(signal_alphabet)
        if _sa not in SIGNAL_ALPHABET_IMPLEMENTED:
            raise SystemExit(
                f"signal_alphabet={_sa!r} 尚未实施（已实现：{SIGNAL_ALPHABET_IMPLEMENTED}）"
                "—— 不静默降级（教训 2）"
            )
        c.signal_alphabet = _sa
    return SphereEngine(c)


def _mg_frac(e, key: str) -> str:
    """A′ 朝向梯度计数占比；未适用/不可用 ⇒ **空串**（不是 0，同 R120 n/a 口径）。"""
    st = e.memory_gradient_stats()
    if not st or not st.get("counters_available"):
        return ""
    v = st.get(key)
    return "" if v is None else str(v)


def main() -> None:
    # ⚠️ 2026-09-15（内评代修，对应已上板的同类缺陷）：本文件续跑路径 print("\u21bb ...")，
    #    而 Windows 默认 GBK 控制台**无法编码 U+21BB (↻)** ⇒ UnicodeEncodeError ⇒ **rc=1 假失败**。
    #    症状：`test_f_r10_f_r12` 的 3 例失败（含 F-R13 回归），且**只在续跑路径**触发
    #    —— 而 D-27-④ 的剂量测量（4 剂量 × 3 seed，须多轮续批）**正要走这条路**。
    #    与 `preflight_check.py` 的同类修复同源（同一错型：“仪器坏了却看不见”的孪生面）。
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass  # 非 TTY / 旧解释器：降级不重配，不因诊断能力缺失而阻断运行
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("on", "off"), required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--ticks", type=int, default=60000)
    ap.add_argument("--log-interval", type=int, default=1000)
    ap.add_argument("--codebook", type=int, default=1,
                    help="1=四机制(arbitrary_codebook=True, R19 主) / 0=三机制对照")
    ap.add_argument("--out", required=True)
    ap.add_argument("--snapshot-every", type=int, default=5000,
                    help="每 N tick 写一次快照（0=不写）")
    ap.add_argument("--snapshot-dir", default="_rerun_logs/snap",
                    help="D-23a：单一最新快照目录（固定文件名原地覆盖，已放开 gitignore）")
    ap.add_argument("--fresh", action="store_true",
                    help="忽略已有快照，从 tick 0 重跑")
    # ---- D-24 小规格验证批（R47/R53）新臂型 ----
    ap.add_argument("--arm", choices=("main", "control", "zero", "sigoff", "oracle"),
                    default=None,
                    help="main=4机制主臂 / control=3机制对照 / zero=漂变零模型(冻结g14g15) / "
                         "sigoff=信号禁用 / oracle=正向对照（D-8）。指定即开 ⑥探针+⑤观测")
    ap.add_argument("--max-count", type=int, default=5000,
                    help="种群上限（R41：⑤ 不饱和前提=3240；R19 口径=5000）")
    # ---- 🔴 F2（镜 19:46 审 / PI R383 697e171）：装置预设档（R278 §三 防呆）----
    #   `--device s2` 一次把装置字段摁到该档口径（480×960/pop 10000/rgm 1.195/纯零背景），
    #   并打印"实际生效装置"回显（第三人眼校验点）。显式传的 --rows 等**优先于预设**。
    #   默认 None ⇒ 不传时本模块完全不介入 ⇒ 与旧版逐位等价（T1 基石）。
    #   ⚠️ 显式覆盖旗标与 resolve_device 的"显式优先"检测配套：旗标必须存在，否则
    #     `--rows 300 --device s2` 会在 argparse 处直接报 unknown argument。
    add_device_arg(ap)
    ap.add_argument("--rows", type=int, default=None,
                    help="世界行数（默认 None = config 默认 60；--device 预设可覆盖）")
    ap.add_argument("--cols", type=int, default=None,
                    help="世界列数（默认 None = config 默认 120；--device 预设可覆盖）")
    ap.add_argument("--patches", type=int, default=None,
                    help="斑块中心数（装置档口径 = resources.patch_count；"
                         "默认 None = 沿用 --patch-count 的现有语义）")
    ap.add_argument("--pop", type=int, default=None,
                    help="初始投放（initial_count；默认 None = 既有 200 口径。"
                         "⚠️ 若 > --max-count 会 fail-loud——引擎按 initial_count 实际投放，"
                         "超上限会静默改变饱和前提）")
    ap.add_argument("--rgm", type=float, default=None,
                    help="斑块再生倍率（= --patch-mult 的装置档名；两者同义，"
                         "显式 --rgm 优先）")
    ap.add_argument("--bg-low-prod-frac", dest="bg_low_prod_frac", type=float,
                    default=None,
                    help="背景低产能格占比（R320/R326 口径；默认 None = 不覆盖）")
    ap.add_argument("--bg-low-cap-mult", dest="bg_low_cap_mult", type=float,
                    default=None,
                    help="背景低产能格容量倍率（映射 resources.bg_cap_mult；"
                         "默认 None = 不覆盖）")
    ap.add_argument("--measure", action="store_true",
                    help="显式开 ⑤观测+⑥探针（--arm 已隐含）")
    ap.add_argument("--no-measure", action="store_true",
                    help="显式关闭探针（覆盖 --arm 隐含；纯观测，关闭不改变模拟数值，"
                         "供不使用 response Δ 的批量 run 提速）")
    # D-27④-A（R86 修订）：oracle 剂量扫描参数。默认 None ⇒ 沿用配置默认值（不改行为）。
    ap.add_argument("--oracle-donation", "--donation", dest="oracle_donation",
                    type=float, default=None,
                    help="覆盖 oracle.donation（剂量扫描用）；**仅 --arm oracle 生效**。"
                         "`--donation` 是等性别名（供 batch_runner `--<key>` 拼参用）")
    ap.add_argument("--oracle-persistence", type=int, default=None,
                    help="覆盖 oracle.persistence（归因窗口 tick）；仅 --arm oracle 生效")
    # R97 增益校准档（R105 放行 ④ 实施）：`m` 必须与校准臂登记成对出现（C5 v2）。
    ap.add_argument("--gain-multiplier", "--m", dest="gain_multiplier",
                    type=float, default=None,
                    help="覆盖 oracle.gain_multiplier（增益档）；**仅 --arm oracle 生效**，"
                         "且 m≠1 必须同时给 --calibration-arm")
    ap.add_argument("--signal-mode", dest="signal_mode", default=None,
                    choices=["state", "random", "evolved"],
                    help="覆盖 signal_mode；`random` = 信号与个体状态无关（**无信息**）"
                         "⇒ 供 R100 条件 7「随机信号自检」用")
    ap.add_argument("--gate-mode", dest="gate_mode", default=None,
                    choices=["none", "delta_positive"],
                    help="R123/B② 门控臂：付款闸 = 只对 Δ_i>0 的到达付款（**仪器**，"
                         "须与 --calibration-arm 同用）")
    ap.add_argument("--gate-delta", dest="gate_delta", default=None,
                    choices=["content", "full"],
                    help="Δ 口径：content=只去**内容项**（保留存在性项，**付款闸推荐**）；"
                         "full=去两项（现探针口径）")
    # A′ 记忆朝向梯度：默认 none（原式 ⇒ 与历史批逐位可比）
    ap.add_argument("--memory-gradient", dest="memory_gradient", default="none",
                    choices=("none", "orientation"),
                    help="A′：none=原式（记忆格==邻居格才加分）；orientation=朝向梯度（cos 允许为负）")
    ap.add_argument("--memory-gradient-gain", dest="memory_gradient_gain",
                    type=float, default=0.3,
                    help="A′：朝向梯度增益（默认 0.3，与原式同量级）")
    # PC-1（R134）：S2 关捕食 / S1 软顶 / 零模型臂瓶颈开关（值式 flag ⇒ preset 可用 key=value）
    ap.add_argument("--predation-enabled", dest="predation_enabled", default="true",
                    choices=("true", "false"),
                    help="PC-1 S2：false=关捕食（单营养级）；默认 true=原式")
    ap.add_argument("--soft-cap-target", dest="soft_cap_target", type=float, default=0.0,
                    help="PC-1 S1（目标窗形式）：N* = target×max_count，出生率在 N* 处线性归零；"
                         "0=关（默认=原式）。R134 PC-1 拟 0.6（窗 [0.2K,0.95K] 的中位）")
    ap.add_argument("--energy-cap", dest="energy_cap", default="false",
                    choices=("true", "false"),
                    help="R144 能量封顶：true=每 tick 末钳制 energy ≤ max_energy（**新纪元**）；"
                         "默认 false=旧行为（与 E-017~E-031/calib1 可比）")
    ap.add_argument("--photo-max", dest="photo_max", type=float, default=None,
                    help="R145：光合产能覆盖（默认 None=不覆盖）；配对臂用 --photo-max 0 关光合")
    # ---- R146/R149 L1 感知追击 + L2 机动性（R150 B1；**全部默认关 = 旧行为**）----
    # 🔴 臂间开关差必须从产物自证 ⇒ 全部进 `switches`（C4）。
    # ⚠️ `store_true`：preset 变体里写裸名（`"l1-seek"`）即可 —— `expand()` 对无值键
    #    只拼开关名（见 `cli_flag`/`expand` 的 `if val:` 分支）。
    ap.add_argument("--l1-seek", dest="l1_seek", action="store_true",
                    help="L1a 追猎项（默认关 = 旧行为，逐位等价）")
    ap.add_argument("--l1-fear", dest="l1_fear", action="store_true",
                    help="L1b 恐惧项（默认关 = 旧行为，逐位等价）")
    ap.add_argument("--l2-dash", dest="l2_dash", action="store_true",
                    help="L2 机动性（两段式冲刺；默认关 = 旧行为，逐位等价）")
    ap.add_argument("--w-seek-max", dest="w_seek_max", type=float, default=0.5,
                    help="L1a 权重上限（预注册两档：0.5(B)/0.25(C)，两档都过才算）")
    ap.add_argument("--w-fear", dest="w_fear", type=float, default=0.5,
                    help="L1b 固定权重（D 臂 = 0，作 fear 是否存在的操作检查）")
    ap.add_argument("--l1-prey-mode", dest="l1_prey_mode", default="lowagg",
                    choices=("lowagg", "any"),
                    help="猎物代理场：lowagg=低 g16 个体 / any=任意占格者（E′ 归因臂）")
    # ---- S1 骨架（设计稿 §三/§5.2）：尸体—食腐 + 血条—受伤 -------------------
    # 🔴 全部**默认关/默认值 = 旧行为**（逐位等价，C7 digest 钉死）。机制未接线 ⇒
    #    这些开关只进指纹与 `switches` 读回（C4），S2/S3 才接线。
    # ⚠️ 开 + `use_sim_core=True` ⇒ 引擎构造期硬报错（H3 fail-loud，同 L1/L2 先例）。
    ap.add_argument("--corpse-enabled", dest="corpse_enabled", action="store_true",
                    help="尸体—食腐通道（默认关 = 旧行为；S1 只建字段，机制不接线）")
    ap.add_argument("--corpse-energy-frac", dest="corpse_energy_frac", type=float, default=0.9,
                    help="死亡时剩余能量 × 该比例 转入尸体格（留 10%% 分解即失）")
    ap.add_argument("--corpse-decay-ticks", dest="corpse_decay_ticks", type=int, default=600,
                    help="尸体存续 tick（≈ 世代时间 ⇒ 脉冲可累积，Noy-Meir）")
    ap.add_argument("--corpse-to-plant-frac", dest="corpse_to_plant_frac", type=float, default=0.5,
                    help="腐烂归还植物池比例（其余为分解损失）")
    ap.add_argument("--corpse-patch-boost", dest="corpse_patch_boost", type=float, default=0.5,
                    help="腐烂处资源 +50%%（持续 2000 tick）")
    ap.add_argument("--corpse-cap-per-cell", dest="corpse_cap_per_cell", type=int, default=200,
                    help="单格尸体**能量**上限（R161 裁定：非尸具数；200 ≈ 一具满能量尸体）")
    ap.add_argument("--scav-gate", dest="scav_gate", type=float, default=0.5,
                    help="食腐 Hill 半效点（**不是硬门槛**；R156 陷阱修正 1）")
    ap.add_argument("--scav-s", dest="scav_s", type=float, default=2.0,
                    help="食腐 Hill 陡度（scav_mult = g16^s/(g16^s+gate^s)）")
    ap.add_argument("--wound-enabled", dest="wound_enabled", action="store_true",
                    help="血条—受伤（消耗战；默认关 = 旧行为；S1 只建字段，机制不接线）")
    ap.add_argument("--wound-base", dest="wound_base", type=float, default=0.35,
                    help="每次成功攻击扣血条 Δ（×(0.5+0.5×攻击性)；一次击杀需 3 次成功）")
    ap.add_argument("--wound-heal-rate", dest="wound_heal_rate", type=float, default=0.001,
                    help="每 tick 恢复（上限 1.0；完全愈合 1000 tick ≈ 1/10 寿命）")
    ap.add_argument("--wound-heal-energy-cost", dest="wound_heal_energy_cost",
                    type=float, default=0.05,
                    help="愈合耗能 / tick（不免费）")
    ap.add_argument("--contest-enabled", dest="contest_enabled", action="store_true",
                    help="争夺食物战（RHP；默认关 = 旧行为；S3 接线）")
    ap.add_argument("--holder-adv", dest="holder_adv", type=float, default=1.2,
                    help="持有者优势（Parker 1974；R161 裁定默认 1.2 —— 0.3 被 RHP 的 (0.3+g16) 淹没）")
    ap.add_argument("--escalation-gap", dest="escalation_gap", type=float, default=0.25,
                    help="不升级的 RHP 差阈值（只有接近才升级）")
    ap.add_argument("--contest-cost-energy", dest="contest_cost_energy", type=float, default=0.5,
                    help="驱逐战的代价（防'免费赶人'）")
    ap.add_argument("--w-fear-health", dest="w_fear_health", type=float, default=0.5,
                    help="血条恐惧项权重（低血条 ⇒ 更恐惧；能力导向）")
    ap.add_argument("--wound-fear-threshold", dest="wound_fear_threshold", type=float,
                    default=0.3,
                    help="血条恐惧触发门槛（1−health ≥ 此值才生效；13.4 波 3）")
    ap.add_argument("--need-aggression-k", dest="need_aggression_k", type=float, default=0.5,
                    help="饥饿激进项强度（固定 0.5；D 臂设 0 = 关'饥饿更激进'）")
    # ---- 13.4 波 1：亚格连续坐标（R169/R175；**默认全关 = 旧行为**）----
    # 臂身份必须能从产物自证（C4）⇒ 全部进 `switches` 读回。
    ap.add_argument("--subpos-enabled", dest="subpos_enabled", action="store_true",
                    help="亚格连续坐标（默认关 = 旧行为，逐位等价；与 l2_dash 互斥 H3）")
    ap.add_argument("--subpos-subdiv", dest="subdiv", type=int, default=4,
                    help="每格 4×4=16 亚位置 ⇒ 最小步长 0.25 格")
    ap.add_argument("--subpos-speed-gain", dest="speed_gain", type=float, default=4.0,
                    help="speed = clamp(mob_eff×gain, 0, speed_max)；4.0 ≈ 历史 1+dash_frac")
    ap.add_argument("--subpos-speed-max", dest="speed_max", type=float, default=2.0,
                    help="速度上限（格/tick）；=2 使 subpos 包含 L2 冲刺语义")
    ap.add_argument("--subpos-min-energy-frac", dest="min_energy_frac", type=float,
                    default=0.05,
                    help="移动能量门槛（纯能量阈值；0.05×max_energy=15，拦'真要饿死'）")
    ap.add_argument("--subpos-lat-floor", dest="lat_floor", type=float, default=0.3,
                    help="极区移速折减下限（speed_cap = speed_max×(lat_floor+(1−lat_floor)cos)）")
    ap.add_argument("--subpos-stay-base", dest="stay_base", type=float, default=0.0,
                    help="停留概率基座（默认 0 = 默认走；语义反转）")
    ap.add_argument("--subpos-stay-food-k", dest="stay_food_k", type=float, default=0.0,
                    help="本格还有余粮 ⇒ 停着吃（权重）")
    ap.add_argument("--subpos-stay-signal-k", dest="stay_signal_k", type=float, default=0.0,
                    help="收到信号 ⇒ 停（权重）")
    ap.add_argument("--subpos-stay-fear-k", dest="stay_fear_k", type=float, default=0.0,
                    help="邻域有威胁 ⇒ 停（权重；本波不接线，只读回）")
    ap.add_argument("--subpos-stay-max", dest="stay_max", type=float, default=0.8,
                    help="停留概率上限（反退化闸：任何个体至少 20%% 概率移动）")
    # ---- 13.4 波 2A：资源动态（T2；**默认关 = 旧行为**）----
    ap.add_argument("--resource-dynamics-enabled", dest="resource_dynamics_enabled",
                    action="store_true",
                    help="斑块休耕—死亡—轮作（默认关 = 旧行为，逐位等价；H3 拦 Rust）")
    ap.add_argument("--rest-ticks", dest="rest_ticks", type=int, default=60,
                    help="休耕时长（该格 N tick 内再生=0；波2 修 v2：300→60，文献轮牧 30–60）")
    ap.add_argument("--rest-threshold", dest="rest_threshold", type=float, default=0.3,
                    help="累计损伤 ≥ 此值（相对容量）⇒ 进入休耕（波2 修 v2 三态机）")
    ap.add_argument("--death-threshold", dest="death_threshold", type=float, default=0.8,
                    help="累计损伤 ≥ 此值 ⇒ 死亡（USDA 摘叶 70–90%% 重伤近死口径）")
    ap.add_argument("--damage-recovery", dest="damage_recovery", type=float, default=0.5,
                    help="休耕到期损伤衰减系数（∈(0,1]；恢复期后损伤部分恢复）")
    ap.add_argument("--kill-frac", dest="kill_frac", type=float, default=10.0,
                    help="被吃强度 > 此值（= kill_mult，**当期再生倍数**；R178 裁定 10）⇒ 斑块死亡")
    ap.add_argument("--kill-denom", dest="kill_denom", default="regrowth",
                    choices=("regrowth", "capacity"),
                    help="kill 分母：regrowth=当期再生（R178 裁定）/ capacity=设计稿字面")
    ap.add_argument("--dead-regen-ticks", dest="dead_regen_ticks", type=int, default=2000,
                    help="死格重入候选池等待（0 = 硬拒绝：不可逆荒漠化）")
    ap.add_argument("--dead-cell-max-frac", dest="dead_cell_max_frac", type=float,
                    default=0.5,
                    help="反荒漠化闸：死格占比超它 ⇒ 强制加速重生")
    ap.add_argument("--kill-patch-only", dest="kill_patch_only", default="true",
                    choices=("true", "false"),
                    help="只有斑块格会死（防背景格大范围被打散）")
    ap.add_argument("--rotate-same-row-only", dest="rotate_same_row_only", default="true",
                    choices=("true", "false"),
                    help="只在同行交换斑块加成（跨行破坏 Σcapacity 守恒）")
    # ---- 13.4 波 2B（T3）：视野 2 格 / 单格上限 / 社交归一化 ----
    ap.add_argument("--perception-span", dest="perception_span", type=int, default=1,
                    choices=(1, 2),
                    help="感知半径（跳数）：1=默认（旧行为）/ 2=两圈（构造级）")
    ap.add_argument("--cell-occupancy-cap", dest="cell_occupancy_cap", type=int, default=3,
                    help="单格个体上限（score 层剔除满格；落本格不受限）")
    ap.add_argument("--cell-occupancy-cap-enabled", dest="cell_occupancy_cap_enabled",
                    action="store_true",
                    help="单格上限开关（默认关 = 旧行为，逐位等价）")
    ap.add_argument("--social-norm", dest="social_norm", default="auto",
                    help="社交项归一化除数：auto=每格实际邻居数（F1 修复）/ 数字=冻结常量")
    ap.add_argument("--init-g16-clusters", dest="init_g16_clusters", default="",
                    help="R141 P0-2：g16 初始投放（逗号分隔，按簇等分人口）；"
                         "空=旧行为。校准批用 \"0.05,0.5,0.9\"")
    ap.add_argument("--forage-tradeoff-k", dest="forage_tradeoff_k", type=float, default=0.0,
                    help="R135 第3步 A-连续：取食倍率 (1−g16)^k（凸 trade-off）；"
                         "0=关（默认，与旧版逐位一致）；本批取 2.0（凸/加速下降 ⇒ 中间态杂食者吃亏）")
    # ---- 🔴 13.5（2026-09-24，fish 批准）：食物绑定 + 能量标定 ---------------------
    # 全部**默认 = 现状行为** ⇒ 关档逐位等价（C7 基线不动），可安全用于配对臂。
    # 术语速记（先人话后术语）：胃容量 = "饭盒大小"；吸收率 = "吃进去多少真变成能量"；
    #   饿死 = "找不到吃的"（能量低**且**胃空）；力竭 = "累垮了"（能量极低，无论胃）。
    ap.add_argument("--bgzero", dest="bg_production_zero", action="store_true",
                    help="13.5 ①：背景产能归零（背景格容量/再生/存量三者归零，只留斑块生产）；"
                         "默认关（= 现状，背景贡献 67.7%% 容量 / 78%% 再生）")
    ap.add_argument("--patch-mult", dest="patch_regrowth_mult", type=float, default=None,
                    help="13.5 ②：斑块再生倍率；None = 用 config 默认（不改动）")
    # ---- 🔴 13.6 S1（2026-09-24，R193 派工）：三种地形的几何参数 -------------------
    # 默认 = config 现状（30 / 2 / 不覆盖）⇒ 关档逐位等价；地形 preset 显式传值。
    # ⚠️ 与 `--patch-mult`（斑块**再生**倍率）不同：`--patch-capacity-mult` 是**容量**倍率。
    ap.add_argument("--patch-count", dest="patch_count", type=int, default=30,
                    help="13.6 地形：斑块中心数（默认 30 = config 现状）；"
                         "森林 12 / 草原 60 / 荒漠 10（设计稿 §一）")
    ap.add_argument("--patch-radius", dest="patch_radius", type=int, default=2,
                    help="13.6 地形：斑块半径（默认 2 = config 现状）；"
                         "森林 3（块大）/ 草原 1 / 荒漠 1（块小）")
    ap.add_argument("--patch-capacity-mult", dest="patch_capacity_mult", type=float,
                    default=None,
                    help="13.6 地形：斑块格**容量**倍率；None = 用 config 默认（3.0，不改动）")
    ap.add_argument("--eat-efficiency", dest="eat_efficiency", type=float, default=None,
                    help="🔴 R187：吃进去的质量→能量的倍率（语义 = 完全燃烧值）；"
                         "None = 用 config 默认。13.5 须与 --assim-herb 同批：7.5 × 0.4 = 3.0")
    ap.add_argument("--stomach-cap-mass", dest="stomach_cap_mass", type=float, default=0.0,
                    help="13.5 ③ 独立胃容量（质量单位）；0 = 沿用旧公式（默认，逐位一致）")
    ap.add_argument("--eat-threshold-frac", dest="eat_threshold_frac", type=float, default=0.0,
                    help="13.5 ③ 胃 ≥ 该比例×容量 就不吃（不饿不吃）；0 = 旧行为（没满就吃）")
    ap.add_argument("--starve-frac", dest="starve_frac", type=float, default=0.0,
                    help="13.5 ③ 饿死阈值：能量 < 该比例×体能 **且胃空** 才死；0 = 旧判据（energy<=0）")
    ap.add_argument("--exhaust-frac", dest="exhaust_frac", type=float, default=0.0,
                    help="13.5 ③ 力竭阈值：能量 < 该比例×体能 即死（**无论胃里有没有食**）；0 = 关")
    ap.add_argument("--assim-herb", dest="assim_herb", type=float, default=1.0,
                    help="13.5 ③ 素食吸收率（吃进去的质量里多少变成能量）；1.0 = 无损失（默认）")
    ap.add_argument("--assim-carn", dest="assim_carn", type=float, default=1.0,
                    help="13.5 ③ 肉食（尸体腿）吸收率；1.0 = 与素食同（默认）")
    ap.add_argument("--assim-return-frac", dest="assim_return_frac", type=float, default=1.0,
                    help="13.5 ③ 未吸收部分回流本格（植物池）的比例；assim=1 时无作用")
    # 🔴 13.5 参数联动的必要件（R188 冒烟发现）：**改死亡阈值必须同步改初始能量**。
    #   否则 `initial_energy(60) < starve_frac(0.30)×max_energy(300)=90` ⇒ **开局集体饿死**
    #   （冒烟实测：2000 tick 后 N=10、饿死 197）。这属于"参数联动"，不是机制问题。
    ap.add_argument("--max-energy", dest="max_energy", type=float, default=None,
                    help="体能上限（能量封顶与饥饿度/成功率的共同分母）；None = 用 config 默认（300）")
    ap.add_argument("--initial-energy", dest="initial_energy", type=float, default=None,
                    help="初始能量；None = 用 config 默认（60）。须 > starve_frac×max_energy")
    # ---- R152/P0（2026-09-22）：**捕食生态位结构** 6 个参数上 CLI -----------------
    # 🔴 目的：P0 要测"给捕食者真实代价 + 让中间态最差"，而这些值此前**只能改源码**
    #    ⇒ 无法做臂间对照（F1 家族："传了开关却没生效"的可预防形态）。
    # ⚠️ 全部**默认 = 旧行为**（逐位一致）：0.1 / 0.4 / 0.3 / 0.2 / 0.1 / 0.9。
    ap.add_argument("--attack-cost", dest="attack_cost", type=float, default=0.1,
                    help="每次**真出手**的能耗（现 0.1 ≈ 代谢的 0.1%%；P0 拟 1.0 = 高风险）")
    ap.add_argument("--transfer-ratio", dest="transfer_ratio", type=float, default=0.4,
                    help="捕食成功抢走猎物能量比例（现 0.4；P0 拟 0.8）")
    ap.add_argument("--attack-gate", dest="attack_gene_gate", type=float, default=0.3,
                    help="攻击性低于该值不发动攻击（现 0.3；P0 拟 0.15）")
    ap.add_argument("--attack-prob-coef", dest="attack_prob_coef", type=float, default=0.2,
                    help="出手概率 ≈ 攻击性 × 该系数 × 饥饿度（现 0.2；P0 拟 0.5）")
    ap.add_argument("--success-floor", dest="success_floor", type=float, default=0.1,
                    help="成功率下限（现 0.1；P0 拟 0.35）")
    ap.add_argument("--success-ceil", dest="success_ceil", type=float, default=0.9,
                    help="成功率上限（现 0.9；P0 拟 0.95）")
    ap.add_argument("--learning-bottleneck", dest="learning_bottleneck", default="true",
                    choices=("true", "false"),
                    help="PC-1 零模型臂：false=关学习瓶颈（配码本关=零模型）；默认 true")
    # PC-1：C3 人为拉满（R134 裁定 reputation_weight=1.0）；signal-disabled 值式 flag
    # （--arm sigoff 的语义照旧，两者取或——禁用臂用哪个都行，preset 统一走值式）
    ap.add_argument("--reputation-weight", dest="reputation_weight",
                    type=float, default=0.0,
                    help="R2 声誉权重（C3 人为拉满用；R134 PC-1 裁定 1.0）")
    ap.add_argument("--signal-disabled", dest="signal_disabled_flag", default="false",
                    choices=("true", "false"),
                    help="true=信号常关（PC-1 禁用臂）；与 --arm sigoff 取或")
    ap.add_argument("--signal-alphabet", dest="signal_alphabet", default=None,
                    choices=list(SIGNAL_ALPHABET_IMPLEMENTED),
                    help='信号字母表档位（R113/R121）："16"=现状 4 位（默认）；'
                         '"4"=仅能量 2 位（code=e_bin+1∈{1..4}，删 f_bit/n_bit）。'
                         '⚠️ 与 "16" 批次（C1a/C1b/C2）的 ratio/codebook_conv 不可直接比较')
    ap.add_argument("--distribution", dest="distribution", default=None,
                    choices=("uniform", "patchy"),
                    help="食物分布（R127/C8 前提开关）：uniform=默认（与历史批可比）；"
                         "patchy=空间斑块（**须先过前提冒烟**：容量空间异质性）")
    ap.add_argument("--calibration-arm", action="store_true",
                    help="登记本臂为**校准臂**（`is_calibration_arm=True`）；"
                         "R100 条件 5：未登记而 m≠1 ⇒ 硬失败")
    # ── 13.7 季节（迁移 S4/S5）——默认 None = 不覆盖 config ⇒ 关档逐位等价 ──
    ap.add_argument("--tilt-deg", dest="tilt_deg", type=float, default=None,
                    help="黄赤交角（**度**）；None=无季节（默认，逐位等价）。"
                         "如 23.44 = 地球真实值。须与 --season-period 同时给")
    ap.add_argument("--season-period", dest="season_period", type=int, default=None,
                    help="一个季节循环的 tick 数（'一年'）；None=无季节（默认）。"
                         "R149 前置门：观测长度须 ≥ 3×该值")
    # ── 🔴 R196 光驱动再生 —— 默认 None = 不覆盖 config（= 0.0）⇒ 逐位等价 ──
    ap.add_argument("--light-sensitivity", dest="light_sensitivity", type=float,
                    default=None,
                    help="再生量对光照的敏感度（clip(光照,0,1)^该值 的指数）。"
                         "None/0 = 不看光（默认，逐位等价）；1.0 = 标准光合响应。"
                         "与 --tilt-deg/--season-period 合用 ⇒ 食物带随季节移动")
    ap.add_argument("--light-normalize", dest="light_normalize",
                    action="store_true", default=None,
                    help="光照因子按全球均值归一化（保全球平均再生量，只改空间分布）")
    # ── 🔴 13.8 日历—罗盘式定向迁徙（g23）—— 默认 False = 旧行为逐位等价 ──
    ap.add_argument("--migration", dest="migration_enabled", action="store_true",
                    default=False,
                    help="开启日历—罗盘式定向迁徙（g23）：候选分数加 "
                         "gain·g23·A(t)·Δ|φ|。🔴 **必须同时给 --tilt-deg 与 "
                         "--season-period**（无季节 ⇒ A≡0 ⇒ 项恒 0 ⇒ 假阴性），"
                         "且**不可与 --use-sim-core 同用**（Rust 未实现）——"
                         "两种情形都在构造期硬报错（设计稿 §3.5 M1/M2）")
    ap.add_argument("--migration-gain", dest="migration_gain", type=float, default=50.0,
                    help="迁移项全局增益（默认 50；S2 扫档建议 20 —— "
                         "项量级约等于觅食项的十分之一，见设计稿 §4 P6）")
    ap.add_argument("--migration-min-abs-anomaly", dest="migration_min_abs_anomaly",
                    type=float, default=0.0,
                    help="|A| 门槛：本地日长异常绝对值 ≤ 该值则跳过迁移项"
                         "（0 = 不设门槛，默认；A∈[−0.5,0.5]）")
    # ── 🔴 R247 饥饿调制（HM）—— 默认关 = 旧行为逐位等价 ──
    ap.add_argument("--hunger-mod-enabled", dest="hunger_mod_enabled",
                    action="store_true", default=False,
                    help="开启饥饿调制（HM）：① 走停 p_eff = clip(p×(1+α·h_norm),0,1)；"
                         "② 感知 perc_eff = perc×(1+β·h_norm)。三处同式"
                         "（Python/Rust/向量化快路径），可与 --use-sim-core 同用")
    ap.add_argument("--hunger-alpha", dest="hunger_alpha", type=float, default=0.5,
                    help="① 走/停调制强度（默认 0.5；0 = 该腿等价于关）")
    ap.add_argument("--hunger-beta", dest="hunger_beta", type=float, default=0.5,
                    help="② 感知权重调制强度（默认 0.5；0 = 该腿等价于关）")
    ap.add_argument("--hunger-h-mid", dest="hunger_h_mid", type=float, default=0.5,
                    help="中性饥饿度（默认 0.5；此点上下产生差异）")
    ap.add_argument("--hunger-stay-gain", dest="hunger_stay_gain", type=float,
                    default=0.0,
                    help="（预留：当前未接线）subpos 驻留调制 —— 仅进指纹/读回")
    # ── 🔴 R239/R258 ASM 模式仲裁 —— 默认 fusion = 旧行为逐位等价 ──
    #   仲裁档需 smell.channels ⊇ {food,risk,kin}（引擎构造期 fail-loud）；
    #   与 L1/L2/HM②/v2/迁徙/ARS/softmax/噪声/占位上限/视野2/smell.use_in_move 互斥。
    ap.add_argument("--asm-mode", dest="asm_mode", default="fusion",
                    choices=("fusion", "arbitration"),
                    help="模式仲裁档（规格 §三）：fusion=融合 score（默认，旧行为）；"
                         "arbitration=WTA 仲裁（显著度 Ŝ_food/Ŝ_risk/Ŝ_kin + 滞回 + 最小锁定）")
    ap.add_argument("--asm-base-explore", dest="asm_base_explore", type=float, default=0.2,
                    help="explore 模式基础显著度（规格 §三；仅 arbitration 档生效）")
    ap.add_argument("--asm-hyst", dest="asm_hyst", type=float, default=0.15,
                    help="切换滞回阈值：best 需超当前模式显著度该值才换（严格 >）")
    ap.add_argument("--asm-hold-ticks", dest="asm_hold_ticks", type=int, default=20,
                    help="最小锁定 tick 数（时间量纲 ⇒ 时间压缩下 ÷k）")
    ap.add_argument("--asm-w-feed", dest="asm_w_feed", type=float, default=1.0,
                    help="feed 显著度权重（Ŝ_food × 该值 + w_hunger×hunger）")
    ap.add_argument("--asm-w-hunger", dest="asm_w_hunger", type=float, default=0.5,
                    help="饥饿项权重（HM ② 在仲裁档由本权重接管；两档不叠加）")
    ap.add_argument("--asm-w-flee", dest="asm_w_flee", type=float, default=1.0,
                    help="flee 显著度权重（Ŝ_risk × 该值）")
    ap.add_argument("--asm-w-join", dest="asm_w_join", type=float, default=0.5,
                    help="join 显著度权重（(Ŝ_kin + density) × 该值）")
    # ── 🔴 R240 T8 气味场通道（空 = 关 ⇒ 旧行为逐位等价）──
    ap.add_argument("--smell-channels", dest="smell_channels", default=None,
                    # F9(a) 勘误（镜 19:46 审）：旧 help 写 "signal" —— 非法通道
                    # （SmellConfig.__post_init__ assert）；合法集 = _KNOWN_CHANNELS
                    # = food,prey,risk,kin。
                    help="逗号分隔的气味通道（food,prey,risk,kin 的子集；空=关）。"
                         "ASM 仲裁档需含 food,risk,kin 三通道。"
                         "⚠️ 默认 None（未传）≠ 空串：续跑时只有**显式传了**才与快照对账"
                         "（distribution 同款；未传 = 沿用快照自带）")
    # ---- C1 捕食信息价值批（D-2 侧车骨架；本线 = [本地开发·性能线] 轻舟）----------
    # 默认全关 ⇒ 主表/manifest 逐字节等价（T1/T12 锁）。
    ap.add_argument("--rd-instruments", dest="rd_instruments", action="store_true",
                    default=False,
                    help="C1：开分块死亡明细侧车（两表 + manifest rd 节）。"
                         "默认关 ⇒ 零行为改动（T1/T11a）")
    ap.add_argument("--rd-channel", dest="rd_channel", default="risk",
                    help="C1 读数通道名（默认 risk；未来批 pheromone 同代码路径，T3 锁）")
    ap.add_argument("--rd-block-rows", dest="rd_block_rows", type=int, default=4,
                    help="C1 分块行数（默认 4 ⇒ 4×4=16 块；§八-4 锁定）")
    ap.add_argument("--rd-block-cols", dest="rd_block_cols", type=int, default=4,
                    help="C1 分块列数（默认 4 ⇒ 4×4=16 块；§八-4 锁定）")
    ap.add_argument("--rd-sample-every", dest="rd_sample_every", type=int, default=250,
                    help="C1 窗宽（默认 250 tick；§八 锁 250t）")
    args = ap.parse_args()
    # ---- 🔴 F2：装置预设档决议（R278 §三；紧接 parse_args —— 任何下游读取 args 前）----
    #   不传 --device ⇒ resolve_device 删掉 device 键并返回 {} ⇒ 零介入（逐位等价）。
    #   传了 ⇒ 未显式给的装置字段被摁到预设值 + stderr 打印"实际生效装置"整行。
    _device_applied = resolve_device(
        args, sys.argv[1:],
        logger=lambda m: print(m, file=sys.stderr))
    # ---- 防呆（F2 面）：initial_count > max_count ⇒ 引擎会**实际投放** initial_count
    #   个个体（sphere_engine 构造期按 initial_count 建种群），max_count 只卡繁殖 ⇒
    #   pop 超上限会静默改变"⑤ 不饱和前提"⇒ 早拒。
    if args.pop is not None and int(args.pop) > int(args.max_count):
        ap.error(
            f"--pop {args.pop} > --max-count {args.max_count}："
            "引擎按 initial_count 实际投放（max_count 只卡繁殖）⇒ 开局即超上限，"
            "饱和前提被静默改变。请显式抬高 --max-count。"
        )
    # ---- C1 fail-loud：--rd-instruments + Rust 路径 ⇒ 硬拒（D-1 钩子只在 Python）----
    # T11c 锁：Rust 侧不接线 ⇒ 开了也拿不到数据 ⇒ 不如早报错。
    if args.rd_instruments and args.mode == "on":
        # --mode on 不一定 = Rust（取决于 build() 里 use_sim_core），但 C1 本批
        # 全 Python 路径（§八 D2 纪律 use_sim_core=False）⇒ mode=on 不会触 Rust。
        # 此处仅做防御性提示；真正拦截在引擎构造后（e._sim_core 检查）。
        pass
    # ---- C1 fail-loud：rd 开 ⇒ 必须有 smell 通道含 risk（或对应 rd-channel）----
    if args.rd_instruments:
        _rd_ch = str(args.rd_channel).strip()
        _smell_ch = [s.strip() for s in str(args.smell_channels or "").split(",")
                     if s.strip()]
        if _rd_ch and _rd_ch not in _smell_ch:
            ap.error(
                f"--rd-instruments + --rd-channel {_rd_ch} 需要 "
                f"--smell-channels 含 {_rd_ch}（SmellField.at() 读数前置）"
            )
    # ---- C1 fail-loud：分块几何合理性 ----
    if args.rd_instruments:
        if args.rd_block_rows < 1 or args.rd_block_cols < 1:
            ap.error("--rd-block-rows/--rd-block-cols 必须 ≥ 1")
        if args.rd_sample_every < 1:
            ap.error("--rd-sample-every 必须 ≥ 1")
    # ── 🔴 13.8 工具侧 fail-loud（设计稿 §3.5 的 M2 前置版）────────────────────
    # 引擎侧 M2 已拦"enabled ∧ 无季节"，但**工具侧也要拦**：否则命令行给
    # `--migration` 忘了 `--tilt-deg`，报错信息指向"引擎构造失败"，运维会误以为
    # 引擎坏了 —— 这里直接说清"是命令行缺参数"，把诊断成本降到零。
    if args.migration_enabled and (args.tilt_deg is None or args.season_period is None):
        ap.error(
            "--migration 必须同时给 --tilt-deg 与 --season-period："
            "无季节 ⇒ 赤纬 δ(t)≡0 ⇒ 逐格日长 P≡0.5（与纬度无关）⇒ "
            "日历轴异常 A≡0 ⇒ 迁移项**逐候选恒 0**（argmax 逐位不变）"
            "⇒ 实验只会读出「迁徙无效」的**假阴性**（错在实验设计，不在机制）。"
        )

    arm = args.arm
    neutral = arm == "zero"
    sig_disabled = arm == "sigoff" or (args.signal_disabled_flag == "true")
    oracle_on = arm == "oracle"
    # R59/F-R9：control = 3 机制对照 ⇒ 关码本（arm 语义优先于 --codebook 默认值 1）。
    # 若无此行，batch grid 只传 arm 时 control 会与 main 同配置同轨迹（2026-09-13 D-24 实测复现）。
    if arm == "control":
        args.codebook = 0
    # D-27④-A：剂量参数只在 oracle 臂有意义——**非 oracle 臂传了就直接报错**，
    # 不静默忽略（教训 2：静默 no-op 最危险）。
    if arm != "oracle" and (args.oracle_donation is not None
                            or args.oracle_persistence is not None):
        raise SystemExit(
            "--oracle-donation/--oracle-persistence 仅在 --arm oracle 下生效"
            f"（当前 arm={arm!r}）——请勿静默传参"
        )
    measure = (bool(args.measure) or arm is not None) and not args.no_measure

    started = time.strftime("%Y-%m-%d %H:%M:%S")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    prog = out.with_suffix(".progress.json")
    # D-23a：单一最新快照（固定文件名、原地覆盖）落在已放开 gitignore 的目录；
    # 兼容旧批次：新路径不存在时回退到 <out>.snapshot.npz（旧命名）。
    snap = Path(args.snapshot_dir) / f"{out.stem}.snapshot.npz"
    rngp = snap.with_suffix(".rngstate.pkl")
    if not snap.exists():
        legacy = out.with_suffix(".snapshot.npz")
        if legacy.exists():
            snap, rngp = legacy, out.with_suffix(".rngstate.pkl")
    # ---- F-R13：快照目录可能不存在 ----
    # `--snapshot-dir` 是调用方给的路径，本脚本此前只 mkdir 了 `out.parent`
    # ⇒ `save_snapshot` 在 `np.savez_compressed` 处抛 FileNotFoundError（实测：
    # test_f_r10_f_r12 两轮续批集成 2 例失败）。旧命名回退时 snap.parent 即
    # out.parent（已建），此处幂等，无副作用。
    snap.parent.mkdir(parents=True, exist_ok=True)

    # ---- 断点续跑：快照存在且非 --fresh 则从快照恢复 ----
    resumed = False
    if not args.fresh and snap.exists():
        e = SphereEngine.load_snapshot(str(snap))       # 配置由快照自带，指纹天然一致
        if rngp.exists():                               # F-D2：补回全局 np.random 状态
            with open(rngp, "rb") as fh:
                np.random.set_state(pickle.load(fh))
        start_tick = int(e._tick)
        resumed = start_tick > 0
    else:
        # ── 🔴 R239/R258 ASM 工具侧 fail-loud（新建档；续跑由快照自带配置 + 上方对账表拦）──
        #   引擎构造期本来就硬报错（互斥表/前置通道），但报错文本是引擎口径 ⇒ 运维不知道
        #   该改哪个命令行开关。这里先拦，直接给出可执行的修法（同 13.8 M2 前置版口径）。
        if str(args.asm_mode) == "arbitration":
            _miss_ch = [
                _c for _c in ("food", "risk", "kin")
                if _c not in [s.strip() for s in
                              str(args.smell_channels or "").split(",") if s.strip()]
            ]
            if _miss_ch:
                ap.error(
                    f"--asm-mode arbitration 需要 --smell-channels 含 {_miss_ch}："
                    "salience 输入 = Ŝ_food/Ŝ_risk/Ŝ_kin（缺则引擎构造期硬报错）"
                )
            if str(args.mode) != "off":
                ap.error(
                    "--asm-mode arbitration 需 --mode off：仲裁的模式目标整段替换融合 "
                    "score，而 `--mode on` 下 D2 的 softmax_tau/感知噪声（config 默认 >0）"
                    "会与它同开互斥（引擎构造期硬报错；`--mode off` = 全感知/无噪声/argmax 档）"
                )
        e = build(args.mode, bool(args.codebook), args.seed, args.ticks,
                  max_count=args.max_count, neutral=neutral,
                  sig_disabled=sig_disabled, oracle=oracle_on, measure=measure,
                  oracle_donation=args.oracle_donation,
                  oracle_persistence=args.oracle_persistence,
                  gain_multiplier=args.gain_multiplier,
                  calibration_arm=bool(args.calibration_arm),
                  signal_mode=args.signal_mode,
                  signal_alphabet=args.signal_alphabet,
                  gate_mode=args.gate_mode, gate_delta=args.gate_delta,
                  distribution=args.distribution,
                  memory_gradient=args.memory_gradient,
                  memory_gradient_gain=args.memory_gradient_gain,
                  predation_enabled=(args.predation_enabled == "true"),
                  soft_cap_target=args.soft_cap_target,
                  learning_bottleneck=(args.learning_bottleneck == "true"),
                  reputation_weight=args.reputation_weight,
                  forage_tradeoff_k=args.forage_tradeoff_k,
                  init_g16_clusters=args.init_g16_clusters,
                  energy_cap=(args.energy_cap == "true"),
                  photo_max=args.photo_max,
                  # 🔴 13.5（2026-09-24）：食物绑定 + 能量标定（默认 = 现状 ⇒ 关档逐位等价）
                  bg_production_zero=bool(args.bg_production_zero),
                  # 🔴 F2：--rgm 为装置档名（= --patch-mult 同义）；显式 --rgm 优先
                  patch_regrowth_mult=(args.rgm if args.rgm is not None
                                       else args.patch_regrowth_mult),
                  # 🔴 F2 装置面（--device 决议值；None ⇒ 不覆盖 ⇒ 逐位等价）
                  rows=args.rows, cols=args.cols, pop=args.pop,
                  bg_low_prod_frac=args.bg_low_prod_frac,
                  bg_low_cap_mult=args.bg_low_cap_mult,
                  # 13.6 S1 地形几何（默认 = config 现状）
                  # 🔴 F2：--patches（装置档名，= resources.patch_count）优先于
                  #   旧 --patch-count——两者同 dest 底，装置批用 --patches。
                  patch_count=(args.patches if args.patches is not None
                               else args.patch_count),
                  patch_radius=args.patch_radius,
                  patch_capacity_mult=args.patch_capacity_mult,
                  # 13.7 季节（默认 None/None ⇒ 不覆盖 ⇒ 逐位等价）
                  tilt_deg=args.tilt_deg,
                  season_period=args.season_period,
                  # 🔴 R196 光驱动再生（默认 None ⇒ 不覆盖 ⇒ 逐位等价）
                  light_sensitivity=args.light_sensitivity,
                  light_normalize=args.light_normalize,
                  # 🔴 13.8 日历—罗盘式定向迁徙（默认关 ⇒ 逐位等价；M1/M2 由引擎拦）
                  migration_enabled=bool(args.migration_enabled),
                  migration_gain=float(args.migration_gain),
                  migration_min_abs_anomaly=float(args.migration_min_abs_anomaly),
                  # 🔴 R247 饥饿调制（默认关 ⇒ 逐位等价；无 Rust 互斥）
                  hunger_mod_enabled=bool(args.hunger_mod_enabled),
                  hunger_alpha=float(args.hunger_alpha),
                  hunger_beta=float(args.hunger_beta),
                  hunger_h_mid=float(args.hunger_h_mid),
                  hunger_stay_gain=float(args.hunger_stay_gain),
                  # 🔴 R239/R258 ASM 模式仲裁（默认 fusion ⇒ 逐位等价；前置/互斥由引擎拦）
                  asm_mode=str(args.asm_mode),
                  asm_base_explore=float(args.asm_base_explore),
                  asm_hyst=float(args.asm_hyst),
                  asm_hold_ticks=int(args.asm_hold_ticks),
                  asm_w_feed=float(args.asm_w_feed),
                  asm_w_hunger=float(args.asm_w_hunger),
                  asm_w_flee=float(args.asm_w_flee),
                  asm_w_join=float(args.asm_w_join),
                  # 🔴 R240 T8 气味通道（空 = 关；仲裁档需含 food,risk,kin）
                  smell_channels=str(args.smell_channels or ""),
                  stomach_cap_mass=args.stomach_cap_mass,
                  eat_threshold_frac=args.eat_threshold_frac,
                  starve_frac=args.starve_frac,
                  exhaust_frac=args.exhaust_frac,
                  assim_herb=args.assim_herb,
                  assim_carn=args.assim_carn,
                  assim_return_frac=args.assim_return_frac,
                  eat_efficiency=args.eat_efficiency,
                  max_energy=args.max_energy, initial_energy=args.initial_energy,
                  # R146/R149（R150 B1）：L1/L2 臂身份（默认全关 = 旧行为）
                  l1_seek=bool(args.l1_seek), l1_fear=bool(args.l1_fear),
                  l2_dash=bool(args.l2_dash), w_seek_max=args.w_seek_max,
                  w_fear=args.w_fear, l1_prey_mode=args.l1_prey_mode,
                  # S1 骨架：尸体—食腐 + 血条—受伤（默认全关 = 旧行为）
                  corpse_enabled=bool(args.corpse_enabled),
                  corpse_energy_frac=args.corpse_energy_frac,
                  corpse_decay_ticks=args.corpse_decay_ticks,
                  corpse_to_plant_frac=args.corpse_to_plant_frac,
                  corpse_patch_boost=args.corpse_patch_boost,
                  corpse_cap_per_cell=args.corpse_cap_per_cell,
                  scav_gate=args.scav_gate, scav_s=args.scav_s,
                  wound_enabled=bool(args.wound_enabled),
                  wound_base=args.wound_base,
                  wound_heal_rate=args.wound_heal_rate,
                  wound_heal_energy_cost=args.wound_heal_energy_cost,
                  contest_enabled=bool(args.contest_enabled),
                  holder_adv=args.holder_adv, escalation_gap=args.escalation_gap,
                  contest_cost_energy=args.contest_cost_energy,
                  w_fear_health=args.w_fear_health,
                  need_aggression_k=args.need_aggression_k,
                  wound_fear_threshold=args.wound_fear_threshold,
                  # 13.4 波 1：亚格连续坐标（默认全关 = 旧行为）
                  subpos_enabled=bool(args.subpos_enabled),
                  subdiv=args.subdiv, speed_gain=args.speed_gain,
                  speed_max=args.speed_max,
                  min_energy_frac=args.min_energy_frac,
                  lat_floor=args.lat_floor,
                  stay_base=args.stay_base, stay_food_k=args.stay_food_k,
                  stay_signal_k=args.stay_signal_k, stay_fear_k=args.stay_fear_k,
                  stay_max=args.stay_max,
                  # 13.4 波 2A：资源动态（默认关 = 旧行为）
                  resource_dynamics_enabled=bool(args.resource_dynamics_enabled),
                  rest_ticks=args.rest_ticks,
                  rest_threshold=args.rest_threshold,
                  death_threshold=args.death_threshold,
                  damage_recovery=args.damage_recovery,
                  kill_frac=args.kill_frac,
                  kill_denom=args.kill_denom,
                  dead_regen_ticks=args.dead_regen_ticks,
                  dead_cell_max_frac=args.dead_cell_max_frac,
                  kill_patch_only=(args.kill_patch_only == "true"),
                  rotate_same_row_only=(args.rotate_same_row_only == "true"),
                  # 13.4 波 2B（T3）：视野 / 单格上限 / 社交归一化
                  perception_span=args.perception_span,
                  cell_occupancy_cap=args.cell_occupancy_cap,
                  cell_occupancy_cap_enabled=bool(args.cell_occupancy_cap_enabled),
                  social_norm=args.social_norm,
                  # R152/P0：捕食生态位结构 6 参
                  attack_cost=args.attack_cost, transfer_ratio=args.transfer_ratio,
                  attack_gene_gate=args.attack_gene_gate,
                  attack_prob_coef=args.attack_prob_coef,
                  success_floor=args.success_floor, success_ceil=args.success_ceil)
        start_tick = 0
    # 🔴 内评未闭合项 #6（2026-09-20 修）：**provenance 必须在跑之前采集**。
    #   收尾时采集记的是"跑完之后的代码树"——长批期间若有人推提交（E-027 就发生过），
    #   manifest 会把**后人的代码**记成这批的产物 ⇒ 既不可复现，也会把排障带向错误方向。
    #   `rng_draws` 例外：它只有收尾才准 ⇒ 在 summary 组装处单独回填。
    prov_start = prov_collect(e.config, python_info=True)
    if resumed:
        print(f"  ↻ 从快照续跑：tick {start_tick} → {args.ticks}")
        # R121 §3.1 + C4 读回：续跑时配置**由快照自带** ⇒ 命令行若另给档位而快照不同，
        # 必须**硬失败**而非静默忽略（F-R21/C5 同族：传了开关却没生效）
        if (args.gate_mode is not None or args.gate_delta is not None) and arm != "oracle":
            raise SystemExit(
                f"--gate-mode/--gate-delta 仅在 --arm oracle 下生效（当前 arm={arm!r}）"
                "——请勿静默传参"
            )
        if args.signal_alphabet is not None and (
            str(args.signal_alphabet) != str(e.config.signal_alphabet)
        ):
            raise SystemExit(
                f"signal_alphabet 冲突：命令行 {args.signal_alphabet!r} vs "
                f"快照 {e.config.signal_alphabet!r} —— 跨档不得续跑（R121 §3.1，特性非缺陷）"
            )
        # R127/C8：distribution 是**前提开关** ⇒ 续跑时必须与快照一致（否则前提被静默换掉）
        if args.distribution is not None and (
            str(args.distribution) != str(e.config.resources.distribution)
        ):
            raise SystemExit(
                f"distribution 冲突：命令行 {args.distribution!r} vs "
                f"快照 {e.config.resources.distribution!r} —— 前提开关不得跨批混用（R127 C8）"
            )
        # R146/R149：L1/L2 开关是**臂身份**（A 臂 vs B 臂的唯一差别）⇒ 续跑时命令行若与
        # 快照不符，必须**硬失败**而非静默沿用快照（同 F-R21/C5 家族："传了开关没生效"）。
        # 段二正是"从段一快照续跑"⇒ 这条检查就是段二不错臂的机器保证。
        for _k, _cli in (("l1_seek", bool(args.l1_seek)), ("l1_fear", bool(args.l1_fear)),
                         ("l2_dash", bool(args.l2_dash))):
            _snap_v = bool(getattr(e.config.simulation, _k))
            if _cli != _snap_v:
                raise SystemExit(
                    f"{_k} 冲突：命令行 {_cli} vs 快照 {_snap_v} —— 臂身份不得静默混用"
                    "（段二续跑必须与段一同臂）"
                )
        if str(args.l1_prey_mode) != str(e.config.simulation.l1_prey_mode):
            raise SystemExit(
                f"l1_prey_mode 冲突：命令行 {args.l1_prey_mode!r} vs "
                f"快照 {e.config.simulation.l1_prey_mode!r}（E′ 臂身份）"
            )
        # 13.4（T1–T4）：波 1/2/3 开关同为**臂身份**（A–E 臂的唯一差别）⇒ 续跑时
        # 命令行若与快照不符，必须**硬失败**（同 F-R21/C5 家族："传了开关没生效"）。
        # 🔴 漏传后果实测路径：`load_snapshot(config=None)` 配置由快照自带 ⇒ 命令行
        #   13.4 开关被**静默忽略**（不报错）⇒ 段二会不知不觉跑成别的臂。本检查补上。
        #   `perception_span` 是档位（1/2）非布尔 ⇒ 单独按 int 比较。
        for _k, _cli, _snap in (
            ("subpos_enabled", bool(args.subpos_enabled),
             bool(e.config.subpos.enabled)),
            ("resource_dynamics_enabled", bool(args.resource_dynamics_enabled),
             bool(e.config.resource_dynamics.enabled)),
            ("cell_occupancy_cap_enabled", bool(args.cell_occupancy_cap_enabled),
             bool(e.config.simulation.cell_occupancy_cap_enabled)),
            ("corpse_enabled", bool(args.corpse_enabled),
             bool(e.config.corpse_wound.corpse_enabled)),
            ("wound_enabled", bool(args.wound_enabled),
             bool(e.config.corpse_wound.wound_enabled)),
            # 13.8：迁徙开关同为**臂身份**（mig_base/mig_g/mig_2g/mig_noseason 的唯一差别）
            ("migration_enabled", bool(args.migration_enabled),
             bool(e.config.migration.enabled)),
            # R247：饥饿调制开关同为**臂身份**（hm_on / hm_off 的唯一差别）
            ("hunger_mod_enabled", bool(args.hunger_mod_enabled),
             bool(e.config.hunger_mod.enabled)),
            # R239/R258：ASM 模式档同为**臂身份**（asm_arb / asm_fusion 的唯一差别）。
            #   ⚠️ 默认 "fusion" ⇒ 续跑 arbitration 档时**必须重传 --asm-mode arbitration**，
            #   否则此处硬失败（防段二不知情降回 fusion；F-R21/C5 家族）。
            ("action_selection.mode", str(args.asm_mode),
             str(e.config.action_selection.mode)),
        ):
            if _cli != _snap:
                raise SystemExit(
                    f"{_k} 冲突：命令行 {_cli} vs 快照 {_snap} —— 臂身份不得静默混用"
                    "（段二续跑必须与段一同臂）"
                )
        # 🔴 R240 T8：气味通道是 arbitration 档的**前置**（salience 输入 = Ŝ_food/Ŝ_risk/Ŝ_kin）
        #   ⇒ 显式传了才与快照对账（未传 = 沿用快照自带；distribution 同款口径）。
        #   显式传而不同 ⇒ 硬失败：否则"想换通道"的意图会被 load_snapshot(config=None) 静默吞掉。
        if args.smell_channels is not None:
            _cli_ch = tuple(
                s.strip() for s in str(args.smell_channels).split(",") if s.strip())
            _snap_ch = tuple(e.config.smell.channels)
            if _cli_ch != _snap_ch:
                raise SystemExit(
                    f"smell_channels 冲突：命令行 {_cli_ch} vs 快照 {_snap_ch} —— "
                    "气味通道是 arbitration 档的 salience 输入（前置），不得静默混用"
                )
        if int(args.perception_span) != int(e.config.simulation.perception_span):
            raise SystemExit(
                f"perception_span 冲突：命令行 {args.perception_span} vs "
                f"快照 {e.config.simulation.perception_span}（C/D/E 臂身份）"
            )
        # 🔴 13.6（S2，2026-09-24）：**地形三参数同为臂身份**（三地形 preset 的唯一差别）
        #   —— 与上表同族：`load_snapshot(config=None)` 让命令行被静默忽略（C5/F-R21），
        #   段二若漏传 `--patch-count/--patch-radius` ⇒ 会不知情地跑成"现状 30/2"的地形。
        for _k, _cli, _snap_v in (
            ("patch_count", int(args.patch_count), int(e.config.resources.patch_count)),
            ("patch_radius", int(args.patch_radius), int(e.config.resources.patch_radius)),
        ):
            if _cli != _snap_v:
                raise SystemExit(
                    f"{_k} 冲突：命令行 {_cli} vs 快照 {_snap_v} —— 地形身份不得静默混用"
                    "（段二续跑必须与段一同臂）"
                )
    # R121 §3.4：**指标口径必须随档位走**（"16"⇒16、"4"⇒4）。
    # 漏传的后果：数组宽度恒 16，未用槽恒"一致" ⇒ 收敛度**系统性虚高**（静默错误）。
    _n_alpha = SIGNAL_ALPHABET_STATES[str(e.config.signal_alphabet)]

    # ---- 13.6 S2 四读数（纯观测；见 `SpatialReadings` 类注释与 `SPATIAL_NOTE`）----
    # 侧车 = 续跑时把"曾访问掩码 + 已结转取食能量"带过来（引擎快照不含能量账本）。
    _spatial_path = snap.parent / f"{out.stem}.spatial.npz"
    if resumed and _spatial_path.exists():
        sr = SpatialReadings.load_sidecar(_spatial_path, e)
    else:
        # 非续跑（或侧车缺失）⇒ 从零起；缺失时**自曝** carry_ok=False（不假装全覆盖）
        sr = SpatialReadings(e, sidecar=_spatial_path,
                             carry_ok=not resumed)

    # ---- C1 D-2 侧车骨架（--rd-instruments 关 ⇒ rd 恒 None ⇒ 零开销，T1/T12）----
    rd = None
    if args.rd_instruments:
        # 🔴 F6（镜 19:46 审 / PI 697e171）：rd 批**禁续跑**——续跑三重失配：
        #   ① 快照重建引擎 ⇒ 钩子静默回 None（后程零记录）；② win_idx 从 0 重编
        #   与 tick 对不上；③侧车一以 "w" 覆写 ⇒ 续跑前窗口全丢。与其静默错，
        #   不如早拒（run 级 fail-loud）。
        if resumed:
            raise SystemExit(
                "C1 --rd-instruments 与快照续跑互斥（F6）：钩子丢失/窗口编号/侧车"
                "覆写三重失配。本批禁续跑 ⇒ 请加 --fresh 或换新 --out 前缀重跑。"
            )
        # fail-loud：Rust 路径硬拒（D-1 钩子只在 Python 捕食段）
        if getattr(e, "_sim_core", None) is not None:
            raise SystemExit(
                "C1 --rd-instruments 不兼容 Rust 路径（D-1 钩子只在 Python）"
            )
        _rd_win_path = out.parent / f"{out.stem}_rd_windows.csv"
        _rd_run_path = out.parent / f"{out.stem}_rd_run.csv"
        rd = RdInstruments(
            e, channel=args.rd_channel,
            block_rows=args.rd_block_rows, block_cols=args.rd_block_cols,
            sample_every=args.rd_sample_every,
            windows_path=_rd_win_path, run_path=_rd_run_path,
        )
        rd.install_hook(e)
        rd.open_sidecars()
        # 首窗 occ 估计基线（全局 + 块级）
        _init_P = len(e._id)
        rd._prev_pop = _init_P
        _init_bp = np.zeros(rd.n_blocks, dtype=np.float64)
        if _init_P:
            np.add.at(_init_bp, rd._block_map[e._flat[:_init_P]], 1.0)
        rd._prev_block_pop = _init_bp

    fields = ["tick", "N", "g14", "g15", "g16", "trust",
              # R135 第 -1 步①（2026-09-20）：g16 **分布矩**——
              # 此前只有均值 ⇒ **判不了双峰**（双峰与单峰可同均值）。
              # BC 双峰系数 = (skew²+1)/(kurt+3(n-1)²/((n-2)(n-3)))，故输出 std/skew/kurt/n。
              # ⚠️ `g4` 是**搭车诊断列**：g16 与 g4（进食量倍率，强选择）的初始 LD 会把
              #    g16 捎带上漂（实测 s42 Δg16=+0.095 全由此而来）⇒ 没有 g4 就无法区分
              #    "g16 被选择" 与 "g16 被搭车"。详见 `sphere_engine._genome_summary`。
              "g16_std", "g16_skew", "g16_kurt", "g4",
              # R139/R140 派工 **P0 观测列**（2026-09-21）：
              # ① `g16_h0..h9` = g16 的 **10-bin 直方图**（值域 [0,1] 均分）——
              #    ⚠️ 没有它，**置换零分布无从下手** ⇒ 双峰永远不可判（E-030 的教训：
              #    只存矩 ⇒ shuffle/重算全落空；且 BC 单用已证明是误判机器）。
              # ② `mean_energy` —— E_prey 推算的最大不确定源（R140 §四），优先级最高。
              # ③ 死因**时间序列**（累计值，差分可得区间）：看死因结构随相位怎么变。
              # ④ `g3` 寿命基因均值 —— 验证"长寿命是否被选择"（maturity ∝ lifespan ⇒ 应有晚熟代价）。
              *[f"g16_h{i}" for i in range(G16_BINS)],
              "mean_energy", "d_starv", "d_pred", "d_old", "g3",
              # R141 P0 派工单 §1.3（🔴 **列名锁定**——`tools/calib_solve.py` 按此消费，勿改名）
              # 三腿 = g16 三分箱 lo/mid/hi；net = forage + pred − meta − move − attack（不含光合）
              "g_lo_n", "g_lo_net_mean", "g_lo_net_p50", "g_lo_net_var",
              "g_mid_n", "g_mid_net_mean", "g_mid_net_p50", "g_mid_net_var",
              "g_hi_n", "g_hi_net_mean", "g_hi_net_p50", "g_hi_net_var",
              "forage_in_mean", "pred_in_mean", "prey_energy_mean",
              # R145 补丁②：囤积**现象**占比（R147 §二 发现 2 改名为 over_cap_frac，
              # 以区别于**仪器**口径 cap_residual_frac —— 两者正交，勿混读）
              "over_cap_frac",
              "max_gen", "max_gen_cur",       # R77：高水位 / 当刻最深（两个口径分列）
              "mean_row", "polar_frac",
              "codebook_conv", "pred_frac",   # D-16：R31③/R38③ 判据列
              "resp_a", "resp_b", "oracle_ratio",   # D-18⑥/D-8（累计口径）
              "mem_bit_frac",   # R128 §五 步骤 0：mem_bit 取值分布的**时间序列**（累计口径）
              # A′（2026-09-19）：**先证"测到了"**再判读 ⇒ 两个可观测性占比（累计口径）
              "mem_grad_slots_frac", "mem_grad_trig_frac",
              # R146/R149 L1/L2（R150 B4）：闸门读数的**时间序列**（累计口径；关档 ⇒ 空串
              # = **未适用**，不是 0）。`seek_zero_frac` 高 = **自熄**（邻域无猎物代理）——
              # 预注册允许结局「猎物池枯竭」，**不得**被读成"没接线"。
              "seek_term_mean", "seek_zero_frac", "fear_term_mean",
              "dash_frac", "mob_eff_mean",
              # S1 骨架（设计稿 §5.2 项 10）：尸体—食腐 + 血条 CSV 列
              # （关档 = 空壳读数：corpse_total/corpse_eaten/wound_n/contest_n 恒 0，
              #   health_mean 恒 1.0 —— S2/S3 接线后才非平凡）
              "corpse_total", "corpse_eaten",
              "health_mean", "health_low_frac", "wound_n", "contest_n",
              # S3 交互（设计稿 §5.4 项 6/7）：争夺战持有者胜率 + 血条恐惧项反退化
              "contest_win_by_holder_frac", "fear_health_flat_frac",
              # 13.4 波 1（R169/R175）：亚格坐标读数（关档 ⇒ 空串 = 未适用，R120 口径）
              "subpos_flat_moves", "subpos_slow_frac",
              # 13.4 波 2A（T2，R176 §12.5 四条）：资源动态读数（关档 ⇒ 空串 = 未适用）
              "dead_cell_frac", "resting_cell_frac",
              "patch_kill_n", "patch_reborn_n", "mean_capacity_effective",
              # 13.4 波 2B（T3）：视野/单格上限读数（关档 ⇒ 空串 = 未适用）
              "span_downgrade_frac", "cap_blocked_n", "cap_stay_n",
              # 13.6 S2（R193 §四）：四读数 + 复算用存量比（未适用 ⇒ 空串 = None，禁写 0）
              "food_util_frac", "patch_visit_frac", "on_patch_frac",
              "gud_var", "gud_mean", "patch_stock_frac"]
    # ---- F-R12：续跑必须**按 tick 幂等**写 CSV ----
    # 原因（2026-09-15 D-24 实测）：续跑直接 `open("a")` 追加 ⇒ 多轮续批会把
    # [start_tick 之前] 的 tick 重复写入（云端 20+ 轮续批：main_s42 16 个重复、
    # main_s43 30 个重复 ⇒ 时序非单调，且**首轮完全看不出来**）。
    # 改法：先截断到 `start_tick`（保留 tick ≤ start_tick 的行），再继续追加。
    if resumed and out.exists():
        _keep = [
            r for r in csv.DictReader(out.open(encoding="utf-8"))
            if str(r.get("tick", "")).isdigit() and int(r["tick"]) <= start_tick
        ]
        with out.open("w", encoding="utf-8", newline="") as _fh:
            _w = csv.DictWriter(_fh, fieldnames=fields)
            _w.writeheader()
            _w.writerows(_keep)
    fh = out.open("a" if resumed else "w", encoding="utf-8", newline="")
    w = csv.DictWriter(fh, fieldnames=fields)
    if not resumed:
        w.writeheader()
    fh.flush()

    last = start_tick
    for t in range(start_tick + 1, args.ticks + 1):
        e.step()
        sr.observe(e)          # 13.6 S2：逐 tick 标记"被占据过的格"（纯观测）
        # ---- C1 D-2：rd 窗末处理（独立于 log_interval；T1/T12 门控）----
        if rd is not None and t % rd.sample_every == 0:
            rd.process_window(e, t, seed=args.seed, arm=arm)
        if t % args.log_interval == 0 or e.extinct:
            P = len(e._id)
            # 🔴 F2（镜 19:46 审 / PI 697e171）：原为 `// 120` 硬编码 mini 列数 ⇒
            #   480×960 档下 mean_row/polar_frac **静默错值**。改 world.cols 现算。
            r = (e._flat[:P] // e.world.cols) if P else np.zeros(0)
            # 极区切点 = 行数 10%（60 行 ⇒ 6 ⇒ 原 r<=5/r>=54 逐位同口径）
            _row_cut = max(1, e.world.rows // 10)
            # R140 P0：死因**时间序列**（累计口径，差分可得区间值）。
            # 键名按 `DeathCause` 成员名归一化（枚举 str() 形如 "DeathCause.STARVATION"）。
            _dct = {str(k).split(".")[-1].upper(): int(v)
                    for k, v in e.death_cause_totals().items()}
            # R146/R149 L1/L2 读数（R150 B4）：**每次只取一次探针**（关档 ⇒ None ⇒ 空串）
            _l1p, _l2p = e.l1_probe(), e.l2_probe()
            w.writerow({
                "tick": t, "N": P,
                "g14": round(float(e._genes[:P, 14].mean()), 4) if P else "",
                "g15": round(float(e._genes[:P, 15].mean()), 4) if P else "",
                # g16 AGGRESSION 均值（所有者 09-20 01:02 派工：测"攻击基因固化"假说，
                # 捕食态内战振荡的基因层证据——此前 g16 从未入 CSV）
                "g16": round(float(e._genes[:P, 16].mean()), 4) if P else "",
                **({} if P < 3 else dict(zip(
                    ("g16_std", "g16_skew", "g16_kurt"), _moments(e._genes[:P, 16])))),
                "g4": round(float(e._genes[:P, 4].mean()), 4) if P else "",
                **({} if P == 0 else dict(zip(
                    (f"g16_h{i}" for i in range(G16_BINS)), _hist10(e._genes[:P, 16])))),
                "mean_energy": round(float(e._energy[:P].mean()), 4) if P else "",
                "d_starv": _dct.get("STARVATION", 0),
                "d_pred": _dct.get("PREDATION", 0),
                "d_old": _dct.get("OLD_AGE", 0),
                "g3": round(float(e._genes[:P, 3].mean()), 4) if P else "",
                # 囤积**现象**占比（单一出处 = 引擎 `over_cap_frac()`；P=0 ⇒ None ⇒ 写空）
                "over_cap_frac": ("" if e.over_cap_frac() is None else e.over_cap_frac()),
                # R141 P0：逐 tick 净收入统计（`_ec_flush` 每 tick 追加一行；n=0 ⇒ None ⇒ 写空）
                **_ec_csv_row(e),
                "trust": round(float(e._trust[:P].mean()), 4) if P else "",
                "max_gen": int(e._max_generation),          # R77：历史高水位（不回落）
                "max_gen_cur": max_generation_current(e),   # R77：当刻最深（只看存活）
                "mean_row": round(float(r.mean()), 3) if P else "",
                # 🔴 F2：极区带 = 上下各 rows//10 行（60 行档 = 与原 5/54 切点逐位同值）
                "polar_frac": round(float(((r < _row_cut)
                                           | (r >= e.world.rows - _row_cut)).mean()), 4)
                if P else "",
                # D-16：
                # R121 §3.4：n_states 随 signal_alphabet（CSV 列口径）
                "codebook_conv": (
                    round(codebook_convergence(e._codebook[:P], n_states=_n_alpha), 4)
                    if P else ""
                ),
                "pred_frac": round(predation_fraction(e.death_cause_totals()), 4),
                # D-18⑥/D-8（累计口径；未开探针时恒 0）
                "resp_a": e.signal_response_stats()["resp_a_exposure"],
                "resp_b": e.signal_response_stats()["resp_b_delta"],
                "oracle_ratio": e.oracle_stats()["oracle_return_ratio"],
                # R128 §五 步骤 0：`mem_bit` 取值分布（累计占比 = mem_bit=1 的发射 / 状态编码发射）
                # 非 "8" 档 ⇒ 空串（**未适用**，不是 0；同 R120 的 ratio n/a 口径）
                # A′：朝向梯度的**可观测性**（非 orientation 模式 ⇒ 空串 = 未适用，不是 0）
                "mem_grad_slots_frac": _mg_frac(e, "slots_frac"),
                "mem_grad_trig_frac": _mg_frac(e, "trigger_frac"),
                "mem_bit_frac": (
                    e.alphabet_stats()["mem_bit_frac"]
                    if e.alphabet_stats()["mem_bit_frac"] is not None else ""
                ),
                # R146/R149 L1/L2（R150 B4）：闸门读数的时间序列（关档 ⇒ 空串 = 未适用）
                "seek_term_mean": _probe_csv(_l1p, "seek_term_mean"),
                "seek_zero_frac": _probe_csv(_l1p, "seek_zero_frac"),
                "fear_term_mean": _probe_csv(_l1p, "fear_term_mean"),
                "dash_frac": _probe_csv(_l2p, "dash_frac"),
                "mob_eff_mean": _probe_csv(_l2p, "mob_eff_mean"),
                # S1 骨架（设计稿 §5.2 项 10）：尸体—食腐 + 血条读数（空壳口径，
                # 同 result.corpse/result.wound —— S1 允许值可为 0）
                "corpse_total": round(float(e._corpse_energy.sum()), 6),
                "corpse_eaten": int(e._corpse_eaten_n),
                "health_mean": (round(float(e._health[:P].mean()), 4) if P else ""),
                "health_low_frac": (
                    round(float((e._health[:P] < 0.5).mean()), 4) if P else ""
                ),
                "wound_n": int(e._wound_n),
                "contest_n": int(e._contest_n),
                # S3 交互（设计稿 §5.4 项 6/7）：争夺战持有者胜率 + 血条恐惧项反退化
                # （None ⇒ 空串 = 未适用，R120 口径）
                "contest_win_by_holder_frac": _probe_csv(
                    e.wound_probe(), "contest_win_by_holder_frac"),
                "fear_health_flat_frac": _probe_csv(
                    e.wound_probe(), "fear_health_flat_frac"),
                # 13.4 波 1：亚格坐标读数（关档 ⇒ None ⇒ 空串 = 未适用）
                "subpos_flat_moves": _probe_csv(e.subpos_probe(), "mean_flat_moves"),
                "subpos_slow_frac": _probe_csv(e.subpos_probe(), "slow_frac"),
                # 13.4 波 2A：资源动态四条（关档 ⇒ None ⇒ 空串 = 未适用）
                "dead_cell_frac": _probe_csv(
                    e.resource_dynamics_probe(), "dead_cell_frac"),
                "resting_cell_frac": _probe_csv(
                    e.resource_dynamics_probe(), "resting_cell_frac"),
                "patch_kill_n": _probe_csv(
                    e.resource_dynamics_probe(), "patch_kill_n"),
                "patch_reborn_n": _probe_csv(
                    e.resource_dynamics_probe(), "patch_reborn_n"),
                "mean_capacity_effective": _probe_csv(
                    e.resource_dynamics_probe(), "mean_capacity_effective"),
                # 13.4 波 2B（T3）：视野/单格上限（关档 ⇒ None ⇒ 空串）
                "span_downgrade_frac": _probe_csv(
                    e.wave2b_probe(), "span_downgrade_frac"),
                "cap_blocked_n": _probe_csv(e.wave2b_probe(), "cap_blocked_n"),
                "cap_stay_n": _probe_csv(e.wave2b_probe(), "cap_stay_n"),
                # 13.6 S2：四读数（None ⇒ 空串 = 未适用，R120 口径）
                **{k: ("" if v is None else round(float(v), 6))
                   for k, v in sr.sample(e, t).items()},
            })
            fh.flush()
            last = t
            if e.extinct:
                break
            if t % 5000 == 0:
                prog.write_text(json.dumps({"seed": args.seed, "mode": args.mode,
                                            "codebook": args.codebook, "tick": t,
                                            "N": P}, ensure_ascii=False), encoding="utf-8")
        # 快照独立节拍（不依赖 log_interval），保证任意时刻中断最多丢 snapshot-every 个 tick
        if args.snapshot_every and t % args.snapshot_every == 0:
            e.save_snapshot(str(snap))
            with open(rngp, "wb") as fh2:
                pickle.dump(np.random.get_state(), fh2)
            sr.save_sidecar(e)     # 13.6 S2：侧车与快照**同节拍**（续跑口径一致）
    fh.close()
    # ---- C1 D-2：rd 侧车收尾（关档 ⇒ rd is None ⇒ 零开销）----
    if rd is not None:
        # 最后一个不完整窗（若 tick 不是 sample_every 的倍数）也排空
        _log = getattr(e, "_rd_pred_kill_log", None)
        # 🔴 F6：收尾排空处同样 fail-loud（不得静默跳过 = 前半有数后半空的来源）
        if _log is None:
            raise SystemExit(_RD_HOOK_LOST_MSG)
        if len(_log) > 0:
            rd.process_window(e, args.ticks, seed=args.seed, arm=arm)
        rd.close()
        rd.write_run_summary(seed=args.seed, arm=arm)

    rows = list(csv.DictReader(out.open(encoding="utf-8")))
    tail = [r for r in rows if int(r["tick"]) >= 10000]
    reached = last >= args.ticks
    stable50k = bool(tail) and all(int(r["N"]) > 0 for r in tail) and last >= 10000
    dc = {str(k): int(v) for k, v in e.death_cause_totals().items()}
    t0 = e.genome_t0_stats()          # R135 第 -1 步③：t=0 基线（搭车诊断）
    # provenance（**启动采集值**为准；仅 rng_draws 用收尾值）
    prov = dict(prov_start)
    prov["rng_draws"] = int(e.rng_draws)
    prov["collected_at"] = "run_start"
    # 漂移自检：跑的过程中若有人推提交 ⇒ 记录双方，供事后归因（不失败，因为产物仍有效）
    prov_end = prov_collect(e.config, python_info=False)
    if prov_end.get("code_tree_sha256") != prov.get("code_tree_sha256"):
        prov["code_tree_drift"] = True
        prov["code_tree_at_finish"] = prov_end.get("code_tree_sha256")
        prov["code_subtrees_at_finish"] = prov_end.get("code_subtrees")
    prov_validate(prov, require_sim_core=bool(e.config.simulation.use_sim_core))
    summary = {
        "manifest": {
            **prov,
            "started": started, "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
            "script": "experiments/a4_verify_capacity.py",
        },
        "switches": {                              # R22：全部开关真实状态
            "mode": args.mode, "seed": args.seed, "ticks_target": args.ticks,
            "resumed_from_snapshot": bool(resumed), "start_tick": start_tick,
            "arbitrary_codebook": bool(args.codebook),
            "arm": arm,
            "measure_signal_response": bool(measure),
            "oracle_enabled": bool(oracle_on),
            "oracle_donation": float(e.config.oracle.donation) if oracle_on else None,
            "oracle_persistence": int(e.config.oracle.persistence) if oracle_on else None,
            "perception_radius": int(e.config.info_structure.perception_radius),
            "perception_noise": float(e.config.info_structure.perception_noise),
            "softmax_tau": float(e.config.info_structure.softmax_tau),
            "learning_bottleneck": bool(e.config.info_structure.learning_bottleneck),
            "steels_alignment": bool(e.config.info_structure.steels_alignment),
            "reputation_weight": float(e.config.info_structure.reputation_weight),
            # A′（2026-09-19）：开关**必须可读回**（C4）—— 冒烟时实测它曾缺席 ⇒
            # 无法回答"跑的到底是哪个档"，与 R127 的 C8 前提对账同型缺陷。
            "memory_gradient": str(e.config.info_structure.memory_gradient),
            "memory_gradient_gain": float(e.config.info_structure.memory_gradient_gain),
            # 13.11：记忆权重基因位（g22）—— 开关 + **位号自证**（P7 家族：位号错位
            # 是"接了却没接对"的隐形来源）；S3.5 批的臂身份 = 本键。
            "memory_weight_gene": bool(e.config.info_structure.memory_weight_gene),
            "memory_weight_gene_slot": int(Gene.MEMORY_WEIGHT),
            # R135 第 -1 步③：**t=0 基线**（跨批可比 + 搭车诊断）
            # 没有它，同一个 g16 读数在 A 批是"选择"、在 B 批是"搭车"，无法分辨。
            "g16_t0_mean": round(t0["g16_mean"], 4),
            "g16_t0_std": round(t0["g16_std"], 4),
            "g4_t0_mean": round(t0["g4_mean"], 4),
            "corr_g16_g4_t0": (round(t0["corr_g16_g4"], 4)
                               if t0.get("corr_g16_g4") is not None else None),
            # PC-1（R134）：S1/S2 开关必须可读回（C4——E-027 的 16 码事故同型预防）
            "predation_enabled": bool(e.config.predation.enabled),
            "soft_cap_target": float(e.config.population.soft_cap_target),
            # R135 第3步 A-连续：凸度必须可读回（C4）
            "forage_tradeoff_k": float(e.config.predation.forage_tradeoff_k),
            # R152/P0：捕食生态位结构 6 参必须可读回（C4）—— 臂间差就在这几个键上，
            # 缺席 ⇒ 外复核只能靠 preset 名（R136 §一 增量 2 的同型缺口）
            "attack_cost": float(e.config.predation.attack_cost),
            "transfer_ratio": float(e.config.predation.transfer_ratio),
            "attack_gene_gate": float(e.config.predation.attack_gene_gate),
            "attack_prob_coef": float(e.config.predation.attack_prob_coef),
            "success_floor": float(e.config.predation.success_floor),
            "success_ceil": float(e.config.predation.success_ceil),
            # R141 P0-2：初始投放必须可读回（C4）
            "init_g16_clusters": str(e.config.genome.init_g16_clusters),
            # R144/R145：能量封顶与光合必须可读回（C4；且封顶开关进指纹 ⇒ 纪元可判）
            "energy_cap_enabled": bool(e.config.organisms.energy_cap_enabled),
            "photo_max": float(e.config.organisms.photo_max),
            # 🔴 R178（13.4 波 2A，T2）：`eat_amount 0.5→0.9` 是**构造级变更**
            #    （C7 基线 digest 变 ⇒ 禁跨纪元比较捕食口径）⇒ 必须可从产物自证（C4）。
            "eat_amount": float(e.config.organisms.eat_amount),
            # 🔴 13.5（2026-09-24）①②③：食物绑定 + 能量标定 ⇒ 全部必须可读回（C4），
            #    否则"传了参数却没生效"会静默（F1 家族；老工 00:20 实测过 bg_production_zero 不生效）。
            "bg_production_zero": bool(e.config.resources.bg_production_zero),
            "patch_regrowth_mult": float(e.config.resources.patch_regrowth_mult),
            # 🔴 13.6 S1（2026-09-24）：**地形身份必须可从产物自证**（C4）——
            #    三地形 preset 的唯一差别就在这 3 个键上（count/radius/capacity_mult）；
            #    缺席 ⇒ 外复核只能靠 preset 名（R136 §一 增量 2 的同型缺口）。
            "patch_count": int(e.config.resources.patch_count),
            "patch_radius": int(e.config.resources.patch_radius),
            "patch_capacity_mult": float(e.config.resources.patch_capacity_mult),
            "stomach_cap_mass": float(e.config.organisms.stomach_cap_mass),
            "eat_threshold_frac": float(e.config.organisms.eat_threshold_frac),
            "starve_frac": float(e.config.organisms.starve_frac),
            "exhaust_frac": float(e.config.organisms.exhaust_frac),
            "assim_herb": float(e.config.organisms.assim_herb),
            "assim_carn": float(e.config.organisms.assim_carn),
            "assim_return_frac": float(e.config.organisms.assim_return_frac),
            "eat_efficiency": float(e.config.organisms.eat_efficiency),
            "initial_energy": float(e.config.organisms.initial_energy),
            "max_energy": float(e.config.organisms.max_energy),
            # 🔴 13.7（2026-09-24）：季节身份必须可从产物自证（C4）——
            #    迁移判据（S5）唯一的前提开关；缺席 ⇒ 无法判"是否有季节"。
            "tilt_rad": float(e.config.light.tilt_rad),
            "tilt_deg": float(np.rad2deg(e.config.light.tilt_rad)),
            "season_period": int(e.config.light.season_period),
            # 🔴 R196（2026-09-24）：光驱动再生身份必须可从产物自证（C4）——
            #    迁徙驱动源的唯一开关；缺席 ⇒ 无法判"食物带是否随季节移动"。
            "light_sensitivity": float(
                getattr(e.config.resources, "light_sensitivity", 0.0)
            ),
            "light_normalize": bool(
                getattr(e.config.resources, "light_normalize", False)
            ),
            # 🔴 13.8（2026-09-24）：迁徙身份必须可从产物自证（C4）——
            #    主判据 ρ(Δ|φ|, g23) 的唯一前提开关；缺席 ⇒ 无法判"迁徙是否在跑"。
            "migration_enabled": bool(e.config.migration.enabled),
            "migration_gain": float(e.config.migration.gain),
            "migration_min_abs_anomaly": float(e.config.migration.min_abs_anomaly),
            # ---- 🔴 R247 饥饿调制（HM；C4 读回；臂身份 = `hunger_mod_enabled`）----
            # 关档 = 旧行为 ⇒ 这些键仍是"默认值读回"，不叫"未适用"（开关可读回是硬要求）。
            "hunger_mod_enabled": bool(e.config.hunger_mod.enabled),
            "hunger_alpha": float(e.config.hunger_mod.alpha),
            "hunger_beta": float(e.config.hunger_mod.beta),
            "hunger_h_mid": float(e.config.hunger_mod.h_mid),
            "hunger_stay_gain": float(e.config.hunger_mod.stay_gain),
            # ---- 🔴 R239/R258 ASM 模式仲裁（C4 读回；臂身份 = `action_selection.mode`）----
            # 关档（fusion）= 旧行为 ⇒ 这些键仍是"默认值读回"，不叫"未适用"（同 HM 口径）。
            "asm_mode": str(e.config.action_selection.mode),
            "asm_base_explore": float(e.config.action_selection.base_explore),
            "asm_hyst": float(e.config.action_selection.hyst),
            "asm_hold_ticks": int(e.config.action_selection.hold_ticks),
            "asm_w_feed": float(e.config.action_selection.w_feed),
            "asm_w_hunger": float(e.config.action_selection.w_hunger),
            "asm_w_flee": float(e.config.action_selection.w_flee),
            "asm_w_join": float(e.config.action_selection.w_join),
            # 🔴 R240 T8（2026-09-28 补接）：气味通道 = arbitration 档 salience 的**前置**
            #    ⇒ 必须可从产物自证（C4）；缺席 ⇒ 外复核无法判"Ŝ 三通道是否真在跑"。
            "smell_channels": [str(s) for s in e.config.smell.channels],
            # ---- C1 捕食信息价值批（D-2 manifest rd 节；§四-4-4）----
            # 🔴 F8（镜 19:46 审 / PI 697e171）：**整节条件写入** —— rd 关 ⇒ 本节
            #   一个键都不落 ⇒ manifest 与"无此代码"逐位等价（T1）。原先"全
            #   False/0/空"的无条件写法会在 rd 关态多出 7 个键 ⇒ 破坏 T1（自相矛盾）。
            **({
                "rd_instruments": True,
                "rd_channel": str(args.rd_channel),
                "rd_block_rows": int(args.rd_block_rows),
                "rd_block_cols": int(args.rd_block_cols),
                "rd_sample_every": int(args.rd_sample_every),
                "rd_hook_installed": getattr(e, "_rd_pred_kill_log", None) is not None,
                "rd_n_kill_log_total": rd.total_kills if rd is not None else 0,
            } if args.rd_instruments else {}),
            # ---- 🔴 F2 装置档回读（仅 --device 实际介入时写；不传 ⇒ 零键 ⇒ 与旧版
            #   逐字节同）—— R321/R326 教训："探针默认 ≠ 预设档"必须从产物自证。
            **({
                "device": str(args.device),
                "device_applied": dict(_device_applied),
                "world_rows": int(e.config.world.rows),
                "world_cols": int(e.config.world.cols),
            } if _device_applied else {}),
            # 🔴 R247：`HungerModConfig` 经 asdict 进 fingerprint ⇒ 快照/续跑配置指纹
            #    已含 HM；续跑混臂由上方 switches 冲突表拦（同 13.4 家族）。
            # 🔴 P7：基因位号必须自证（= 23）—— 位号错位是"接了却没接对"的隐形来源
            "migrate_gene_slot": int(Gene.MIGRATE_BIAS),
            # 🔴 R136 §一 增量 2（C4 自证缺口）：PC-1 三臂的 `switches.arm` **全为 main**，
            #    码本/瓶颈两个开关读不到 ⇒ **臂间开关差无法从产物自证**（外复核只能靠 preset 名）。
            "arbitrary_codebook": bool(e.config.info_structure.arbitrary_codebook),
            "learning_bottleneck": bool(e.config.info_structure.learning_bottleneck),
            "use_sim_core": bool(e.config.simulation.use_sim_core),
            # ---- R146/R149 L1/L2（R150 B1/B2）：**臂身份必须能从产物自证**（C4）----
            # A 臂 vs B/C/D/E′ 的唯一差别就在这几个键上 ⇒ 缺席 ⇒ 外复核只能靠 preset 名
            # （R136 §一 增量 2 的同型缺口）。
            "l1_seek": bool(e.config.simulation.l1_seek),
            "l1_fear": bool(e.config.simulation.l1_fear),
            "l2_dash": bool(e.config.simulation.l2_dash),
            "w_seek_max": float(e.config.simulation.w_seek_max),
            "w_fear": float(e.config.simulation.w_fear),
            "l1_prey_mode": str(e.config.simulation.l1_prey_mode),
            # ---- S1 骨架（设计稿 §三/§5.2 项 9）：尸体—食腐 + 血条（C4 读回）----
            # 臂间差（B/C/D/E 臂）就落在这几个键上 ⇒ 缺席 ⇒ 外复核只能靠 preset 名
            # （R136 §一 增量 2 的同型缺口）。
            "corpse_enabled": bool(e.config.corpse_wound.corpse_enabled),
            "corpse_energy_frac": float(e.config.corpse_wound.corpse_energy_frac),
            "corpse_decay_ticks": int(e.config.corpse_wound.corpse_decay_ticks),
            "corpse_to_plant_frac": float(e.config.corpse_wound.corpse_to_plant_frac),
            "corpse_patch_boost": float(e.config.corpse_wound.corpse_patch_boost),
            "corpse_cap_per_cell": int(e.config.corpse_wound.corpse_cap_per_cell),
            # 🔴 R165 0-1（2026-09-23，老工）：**单位自证** —— 尸体池以**能量**记，
            #    食腐入胃处按 ÷`eat_efficiency` 折算成质量。13.3 批（p1corpse）
            #    无此标记且用了"池当质量"的旧口径（×3 放大）⇒ 能量层读数不可跨纪比较。
            "corpse_pool_unit": str(e.config.corpse_wound.corpse_pool_unit),
            "scav_to_energy_divisor": float(e.config.organisms.eat_efficiency),
            "scav_gate": float(e.config.corpse_wound.scav_gate),
            "scav_s": float(e.config.corpse_wound.scav_s),
            "wound_enabled": bool(e.config.corpse_wound.wound_enabled),
            "wound_base": float(e.config.corpse_wound.wound_base),
            "wound_heal_rate": float(e.config.corpse_wound.wound_heal_rate),
            "wound_heal_energy_cost": float(e.config.corpse_wound.wound_heal_energy_cost),
            "contest_enabled": bool(e.config.corpse_wound.contest_enabled),
            "holder_adv": float(e.config.corpse_wound.holder_adv),
            "escalation_gap": float(e.config.corpse_wound.escalation_gap),
            "contest_cost_energy": float(e.config.corpse_wound.contest_cost_energy),
            "w_fear_health": float(e.config.corpse_wound.w_fear_health),
            "need_aggression_k": float(e.config.corpse_wound.need_aggression_k),
            "wound_fear_threshold": float(e.config.corpse_wound.wound_fear_threshold),
            # ---- 13.4 波 1：亚格连续坐标（C4 读回；臂身份 = `subpos_enabled`）----
            # 关档 = 旧行为 ⇒ 这些键仍是"默认值读回"，不叫"未适用"（开关可读回是硬要求）。
            "subpos_enabled": bool(e.config.subpos.enabled),
            "subpos_subdiv": int(e.config.subpos.subdiv),
            "subpos_speed_gain": float(e.config.subpos.speed_gain),
            "subpos_speed_max": float(e.config.subpos.speed_max),
            "subpos_min_energy_frac": float(e.config.subpos.min_energy_frac),
            "subpos_lat_floor": float(e.config.subpos.lat_floor),
            "subpos_stay_base": float(e.config.subpos.stay_base),
            "subpos_stay_food_k": float(e.config.subpos.stay_food_k),
            "subpos_stay_signal_k": float(e.config.subpos.stay_signal_k),
            "subpos_stay_fear_k": float(e.config.subpos.stay_fear_k),
            "subpos_stay_max": float(e.config.subpos.stay_max),
            # ---- 13.4 波 2A：资源动态（C4 读回；臂身份 = `resource_dynamics_enabled`）----
            "resource_dynamics_enabled": bool(e.config.resource_dynamics.enabled),
            "rest_ticks": int(e.config.resource_dynamics.rest_ticks),
            # 波2 修 v2（方案 A）：累计损伤三态机阈值（C4 自证）
            "rest_threshold": float(e.config.resource_dynamics.rest_threshold),
            "death_threshold": float(e.config.resource_dynamics.death_threshold),
            "damage_recovery": float(e.config.resource_dynamics.damage_recovery),
            "kill_frac": float(e.config.resource_dynamics.kill_frac),
            "kill_denom": str(e.config.resource_dynamics.kill_denom),
            "dead_regen_ticks": int(e.config.resource_dynamics.dead_regen_ticks),
            "dead_cell_max_frac": float(e.config.resource_dynamics.dead_cell_max_frac),
            "kill_patch_only": bool(e.config.resource_dynamics.kill_patch_only),
            "rotate_same_row_only": bool(
                e.config.resource_dynamics.rotate_same_row_only),
            # ---- 13.4 波 2B（T3）：视野 / 单格上限 / 社交归一化（C4 读回）----
            "perception_span": int(e.config.simulation.perception_span),
            "perception_cap": int(getattr(e.config.simulation, "perception_cap", 32)),
            "cell_occupancy_cap": int(e.config.simulation.cell_occupancy_cap),
            "cell_occupancy_cap_enabled": bool(
                e.config.simulation.cell_occupancy_cap_enabled),
            "social_norm": str(e.config.simulation.social_norm),
            # L2 几何/成本参数（「参数制造分化」的可核查性）
            "dash_min_energy_frac": float(e.config.organisms.dash_min_energy_frac),
            "dash_cost_kappa": float(e.config.organisms.dash_cost_kappa),
            "dash_cost_exp": float(e.config.organisms.dash_cost_exp),
            "young_mob_mult": float(e.config.organisms.young_mob_mult),
            "old_mob_mult": float(e.config.organisms.old_mob_mult),
            "far_cap": int(e.config.organisms.far_cap),
            "distribution": e.config.resources.distribution,
            "initial_count": int(e.config.population.initial_count),
            "max_count": int(e.config.population.max_count),
            "neutral_genes": bool(e.config.neutral_genes),
            "signal_disabled": bool(e.config.signal_disabled),
            # R97 增益校准档（条件 5 机器强制拒收依据 + C4 读回核对）
            "oracle_gain_multiplier": float(e.config.oracle.gain_multiplier),
            "is_calibration_arm": bool(e.config.oracle.is_calibration_arm),
            "signal_mode": str(e.config.signal_mode),
            # R113/R121 C4 读回：字母表档位（必须与命令行/预注册一致；跨档不可比）
            "signal_alphabet": str(e.config.signal_alphabet),
            # R123/B② C4 读回：门控臂档位（仪器参数必须可核对，防"传了没生效"）
            "oracle_gate_mode": str(e.config.oracle.gate_mode),
            "oracle_gate_delta": str(e.config.oracle.gate_delta),
        },
        "result": {
            # ⚠️ 必须用引擎真实 tick，不能用 last（=最后一次【采样】的 tick）：
            # log_interval 默认 1000，跑 1500 tick 时 last 会停在 1000 ⇒ final_tick 记错
            # （60k 恰好是 1000 的倍数才一直没暴露）。R42/R38 判据依赖该字段。
            "final_tick": int(e._tick), "final_N": len(e._id), "extinct": bool(e.extinct),
            "reached_60k": reached, "stable_N_gt0_last50k": stable50k,
            "eco_gate_pass": bool(reached and stable50k),
            # ---- F-R17：`eco_gate_pass` 的**适用域**必须随数据走 ----
            # 该门判据含 `last >= 10000` 硬编码（为 60k 批定义）⇒ **短程批恒 False**。
            # 缺适用域说明时，短程批的 False 会被误读成"生态崩溃"（本人 2026-09-15 实测踩到）。
            "eco_gate_scope": f"{args.ticks} tick 批",
            "eco_gate_applicable": bool(args.ticks >= 10000),
            "born_total": int(e.total_born), "died_total": int(e.total_died),
            "deaths_by_cause": dc,
            # D-16：终局判据值（区制分层用：饱和封顶 vs 捕食主导）
            "final_codebook_conv": (
                round(
                    codebook_convergence(
                        e._codebook[: len(e._id)], n_states=_n_alpha
                    ),
                    4,
                )
                if len(e._id) else 0.0
            ),
            "final_pred_frac": round(predation_fraction(dc), 4),
            # R77：max_gen 两个口径**分名记录**（此前同一列名 `max_gen` 在不同脚本里
            # 分别指"当刻最深"与"历史高水位"⇒ 跨脚本比较必错）
            "final_max_gen_highwater": int(e._max_generation),
            "final_max_gen_current": max_generation_current(e),
            # D-17 ⑤（R42）：非饱和窗主口径 + 饱和窗诊断（两窗不得合并）
            "selection_gradient": selection_gradient(e),
            # D-18 ⑥（R43）：三联报
            "signal_response": e.signal_response_stats(),
            # D-8 oracle（O-4/O-7）：manifest 必录 return_ratio
            # （内含 R121 §4.1 的 food_band_true_sig 零机时 counter）
            "oracle": e.oracle_stats(),
            # R121 §3：字母表档位 + §3.5 码值域守卫（bad_code_n 在 "4" 下必须恒 0）
            "alphabet": e.alphabet_stats(),
            # A′（2026-09-19）：记忆朝向梯度的可观测性计数（非 orientation ⇒ **None 未适用**）
            "memory_gradient": e.memory_gradient_stats(),
            # R135 第 -1 步④：互捕结构量化（攻击者/猎物 g16 直方图 + Δ + 真决斗三级拆分）
            "cannibalism": e.cannibalism_stats(),
            # R144/R145：能量封顶探针 + 状态量边界自检
            "energy_cap": e.energy_cap_probe(),
            # R146/R149（R150 B1/B2）：L1 两项 + L2 机动性读数（**关档 ⇒ None = 未适用**）
            "l1": e.l1_probe(),
            "l2": e.l2_probe(),
            # 🔴 13.8：日历—罗盘式定向迁徙读数（**关档 ⇒ None = 未适用**，R120 口径）。
            #    含反退化占比 mig_flat_frac（R148-1）与量级 mig_term_abs_mean（P6）。
            "migration": e.migration_probe(),
            # 🔴 R239/R258：模式仲裁读数（**关档 ⇒ None = 未适用**，R120 口径）。
            #    含 sw_n/hold_block_n（dithering 红线主读数）与四模式占比。
            "asm": e.asm_probe(),
            "bounds": e.state_bounds_check(),
            # S1 骨架（设计稿 §5.2 项 9）：尸体—食腐 + 血条读数块（**空壳**，值可为 0）
            "corpse": e.corpse_probe(),
            "wound": e.wound_probe(),
            # 13.4 波 1：亚格坐标读数（关档 ⇒ None = 未适用，R120 口径）
            "subpos": e.subpos_probe(),
            # 13.4 波 2A：资源动态读数（关档 ⇒ None = 未适用）+ 守恒自检（恒应过）
            "resource_dynamics": e.resource_dynamics_probe(),
            "resource_dynamics_conservation": e.resource_dynamics_conservation(),
            # 13.4 波 2B（T3）：视野/单格上限读数
            "wave2b": e.wave2b_probe(),
            # 13.6 S2（R193 §四）：四读数块（食物利用率/斑块访问率/在斑块占比/GUD 方差
            #   + R191 复算用存量比）；口径与 self-证 字段见 `SPATIAL_NOTE`
            "spatial": sr.summary(e, int(e._tick)),
            # R141 P0（派工单 §1.3，🔴 段名与结构锁定 —— `calib_solve.py` 按此消费）
            "energy_ledger": e.energy_ledger(),
            # R135 第 -1 步③：t=0 基因组基线（搭车诊断）
            "genome_t0": {k: v for k, v in t0.items() if k != "gene_means"},
            # R121 §4.2：记忆继承卫生（"生而知之"量化；纯观测）
            "inheritance": e.inheritance_stats(),
        },
    }
    out.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    # ---- F-R10：收尾**不再删除** progress.json ----
    # 原因（2026-09-15 D-24 实测）：删除会被环境 safe-delete 守卫 fail-closed 拒绝
    # （作用域内累计删除超限 ⇒ 拒删并终止进程）⇒ 子进程 rc=1 **假失败**，
    # 污染批次状态与退出码（数据无损：上一行 summary 已先写）。
    # 改法：写"完成标记"覆盖 progress.json（不删文件 ⇒ 不触发守卫）。
    # 顺序保证：summary 先写、标记后写 ⇒ **任何时刻**中断都不丢数据。
    prog.write_text(
        json.dumps(
            {
                "status": "complete",
                "final_tick": int(e._tick),
                "final_N": len(e._id),
                "finished": summary["manifest"]["finished"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    s = summary["result"]
    print(f"[{args.mode} cb={args.codebook} s{args.seed}] tick={s['final_tick']} "
          f"N={s['final_N']} eco_gate={s['eco_gate_pass']} "
          f"born={s['born_total']} died={s['died_total']}")


if __name__ == "__main__":
    main()
