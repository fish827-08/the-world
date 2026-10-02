"""斑块"休耕—死亡—轮作"（13.4 波 2）—— 资源—消费者**负反馈**的纯逻辑模块。

出处：fish 2026-09-23 01:20 构想 + `[所有者]` R176 §12（文献范式：Noy-Meir 1975 /
Westoby 1989 状态转换理论；Klausmeier / von Hardenberg *Grazing Away the Resilience
of Patterned Ecosystems*；Christensen 2003 内蒙古草原放牧阈值 0.49；Urad 荒漠草原 2025）。

为什么需要它
------------
现状（`ResourceField._regrowth_amount`）的再生**只由温度驱动**：

    growth = regrowth_rate × clip((温度+20)/20, 0, 1)^temp_sensitivity × (斑块 2.0 / 背景 mult)

⇒ **一格今天被吃光，下一 tick 照样按温度满速恢复** ⇒ 取食压力在资源侧**没有任何反馈**。
P0/P0.6 的诊断结论是「缺的不是参数，而是**资源可持续供给的结构**」——
本模块提供那条缺失的负反馈：

    取食压力 ↑ ⇒ 斑块死亡 ↑ ⇒ 总产能 ↓ ⇒ 种群 ↓ ⇒ 取食压力 ↓ ⇒ 死格重生 ⇒ 产能恢复

本质三件事
----------
1. **休耕（rest）**：某格被吃过 ⇒ 之后 `rest_ticks` 内**再生为 0**（存量还不还，看引擎）。
2. **死亡（kill）**：某格被吃**强度超过阈值** ⇒ 该格**不再生长**（斑块格同时**失去斑块加成**）。
3. **轮作（rotation）**：死格 `dead_regen_ticks` 后**重入候选池**，同时把"斑块加成"
   **搬到一个随机背景格**（fish 原话："等作为随机斑点生长位"）。
   ⇒ 与文献里的"荒漠化（永久裸地 ⇒ 不可逆）"**根本不同**：这是 **shifting mosaic**，
   空间上不断迁移的斑块镶嵌 ⇒ **可逆**。

🔴 动手前实测的三条（`tools/wave2_patch_probe.py`，零 RNG，一条命令复现）
--------------------------------------------------------------------
**(A) `kill_denom="capacity"` 是"纬度抽奖"** —— `capacity = capacity_per_area × 格面积`，
而格面积跨纬度差 **151 倍**（赤道 patch 格 120.0 vs 极点 patch 格 3.14）。
每格每 tick 最大取食量 = `eat_amount(0.5) × eat_mult_max(1.5) × occ_cap(3)` = **2.25** ⇒
阈 0.7 时**只有 1/817 个斑块格**会被杀死（且全在极区行 0/1/59），**赤道行比值仅 0.074**
⇒ 本机制最想做的事（惩罚"停着把富格吃死"）在生命区**恰好永不触发**。

**(B) 因此分母改用「当期再生量」**（= 设计稿 §2.5「格内吃>长」的同一个量）：
斑块格 单人 0.75 / 3 人 **2.25**，背景格 单人 1.71 / 3 人 5.12 ⇒ **与纬度无关**
⇒ 阈值有唯一物理含义：**「同格几个人的取食量超过这格当期再生」**。
（`kill_denom="capacity"` 仍保留，供字面口径复核。）

**(C) 轮作必须"同行交换"** —— 面积只依赖纬度 ⇒ 同一行内所有格面积相同。
实测：同行交换 843 次 ⇒ `ΔΣcapacity = 0.000000e+00`（逐位）；
**跨行交换 ⇒ `ΔΣcapacity = +1.886e+03`（相对 1%）** ⇒ 破坏 `ResourceField` 构造期的
「Σcapacity 守恒」不变量 ⇒ 本模块**只在同一行内**交换（`rotate_same_row_only`）。

契约（与 L2/Subpos 同族）
------------------------
* **默认关**（`enabled=False`）⇒ 所有方法**退化为 no-op**（`growth_multiplier` 返回全 1、
  两个写方法直接返回）⇒ 引擎即使无条件调用也**不进任何新代码路径**（C7 逐位等价）。
* **不 import 引擎**、**不持有 RNG** —— 随机性一律由调用方通过 `rand_u` 传入
  （⇒ 模块内无隐藏抽取；同 `rand_u` ⇒ 逐位可复现）。
* 状态量边界（R145 C9 家族）：`_mask`/`_dead`/`_rest_until` 全部可自检，见
  `conservation_check()`（Σcapacity 与"面积加权再生倍率"两条不变量）。
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

__all__ = ["ResourceDynamics", "KILL_DENOMS"]

#: `kill_denom` 可选口径（见模块 docstring (A)/(B)）
KILL_DENOMS = ("regrowth", "capacity")


class ResourceDynamics:
    """斑块休耕/死亡/轮作的状态机（按格；**纯逻辑**，不依赖引擎）。

    典型接线（引擎侧，三段）：

    ```python
    rd = ResourceDynamics.from_field(cfg.resource_dynamics, self.resources, self.world)
    # ① 再生闸：growth = rf._regrowth_amount(tick) * rd.growth_multiplier()
    # ② 取食后：rd.note_tick(cell_intake, growth, tick)        # 记休耕 + 判死
    # ③ 每 tick 末：rd.rotate(tick, rand_u=self.rng.random(k))  # 重生 + 搬加成
    #    （加成已搬走 ⇒ 还需把 rf._capacity 重算：rf._capacity[:] = rd.capacity_from_base(base)）
    ```

    参数
    ----
    n_cells : int
        格子总数。
    cfg : Any
        配置对象（`ResourceDynamicsConfig`）；缺字段时按默认值回退（向前兼容）。
    patch_mask : NDArray[bool] 或 None
        构造期的斑块掩码（`ResourceField._patch_mask`）；None ⇒ 全背景。
    cols : int
        经度格数（用于把 `flat` 换算成"行"⇒ 同行交换）。
    base_capacity : NDArray[float64]
        **不含斑块倍率**的基准容量（= `capacity_per_area × 格面积`）。
        ⚠️ 与 `ResourceField._capacity` 不同：后者**已含**倍率。
    patch_capacity_mult / bg_capacity_mult : float
        构造期两个容量倍率（`patch_capacity_mult` / 背景倍率）。
    patch_regrowth_mult : float
        构造期斑块再生倍率（默认 2.0）。
    """

    def __init__(
        self,
        n_cells: int,
        cfg: Any,
        *,
        patch_mask: Optional[np.ndarray],
        cols: int,
        base_capacity: np.ndarray,
        patch_capacity_mult: float,
        bg_capacity_mult: float,
        patch_regrowth_mult: float = 2.0,
    ) -> None:
        g = lambda k, d: getattr(cfg, k, d)  # noqa: E731 — 缺字段回退（向前兼容）
        self.enabled = bool(g("enabled", False))
        self.rest_ticks = int(g("rest_ticks", 60))   # R208 §三：兜底必须 == config 默认（config.py:1044）
        # 波2 修 v2（fish 批准方案 A）：累计损伤三态机阈值
        self.rest_threshold = float(g("rest_threshold", 0.3))
        self.death_threshold = float(g("death_threshold", 0.8))
        self.damage_recovery = float(g("damage_recovery", 0.5))
        self.kill_frac = float(g("kill_frac", 10.0))   # R208 §三：兜底 == config 默认（再生倍数口径，非比例）
        self.kill_denom = str(g("kill_denom", "regrowth"))
        self.dead_regen_ticks = int(g("dead_regen_ticks", 2000))
        self.dead_cell_max_frac = float(g("dead_cell_max_frac", 0.5))
        self.kill_patch_only = bool(g("kill_patch_only", True))
        self.rotate_same_row_only = bool(g("rotate_same_row_only", True))
        # 🔴 R226 修复②：rd **自有独立 RNG 流**（不再消费引擎 `self.rng`）。
        #   动机：`rotate` 原用 `self.rng.random(n_patch)` ⇒ 与种群动力学共享 RNG 流，
        #   启用 rd 即改变种群随机序列（可对拍性差、归因不干净）。独立流后：
        #     · 引擎 RNG 序列与 rd 开关**完全解耦**（关/开档同序）
        #     · rd 随机性来源单一、可独立复现（seed 由 cfg 或默认恒定）
        _rd_seed = int(g("rng_seed", 20260927))
        self._rng = np.random.default_rng(_rd_seed)
        # 🔴 `dead_regen_ticks = 0` ⇒ "永不重生" = 文献里的**不可逆荒漠化** ⇒ 硬拒绝
        assert self.dead_regen_ticks > 0, (
            "dead_regen_ticks=0 会让斑块永不再生（不可逆荒漠化）—— 这不是本设计要的东西；"
            "若确实要测不可逆，请另立开关并在板上裁定"
        )
        assert self.kill_denom in KILL_DENOMS, f"kill_denom 必须是 {KILL_DENOMS}"
        assert 0.0 <= self.kill_frac, "kill_frac 必须 ≥ 0"
        assert 0.0 < self.dead_cell_max_frac <= 1.0, "dead_cell_max_frac ∈ (0,1]"

        self.n_cells = int(n_cells)
        self.cols = int(cols)
        self._row = np.arange(self.n_cells, dtype=np.int64) // self.cols
        self._base_capacity = np.asarray(base_capacity, dtype=np.float64)
        assert self._base_capacity.shape == (self.n_cells,), "base_capacity 形状不符"
        self.patch_capacity_mult = float(patch_capacity_mult)
        self.bg_capacity_mult = float(bg_capacity_mult)
        self.patch_regrowth_mult = float(patch_regrowth_mult)

        # 掩码：初始 = 构造期斑块掩码；轮作会就地搬移（同行交换 ⇒ Σcapacity 逐位守恒）
        if patch_mask is None:
            self._mask = np.zeros(self.n_cells, dtype=bool)
        else:
            self._mask = np.array(patch_mask, dtype=bool, copy=True)
        assert self._mask.shape == (self.n_cells,), "patch_mask 形状不符"

        # 🔴 R226 修复③（核心）：`bg_production_zero=True`（背景格容量=0、永久荒漠）时
        #   **禁止轮作搬移** —— 搬移的前提是"背景格可承载"，而该设定下背景格零产能。
        #   搬移会把唯一粮仓的位置**迁移**，导致个体站位错配（实测 ON 臂仅 50–72%
        #   个体站在有食物的格上 vs OFF 臂 81–97%）⇒ 吃不到 ⇒ 饿死 ⇒ 灭绝。
        #   语义：斑块"死而就地重生"（原地不动），不搬家。判据 = 背景容量倍率为 0。
        self.bg_production_zero = (float(bg_capacity_mult) == 0.0)
        # 轮作搬移开关（False ⇒ 加成不搬、死格原地保留斑块身份）
        self.rotate_moves_patch = bool(g("rotate_moves_patch", True))
        # 🔴 R226 修复③：停产乘子（0.0 = 旧行为"停产"）。`bg_production_zero=True`
        #   ⇒ 缺省改 0.5（"减产不停产"，避免唯一粮仓被永久砍掉 ⇒ 灭绝）。
        _def_mult = 0.5 if self.bg_production_zero else 0.0
        self.dead_regen_mult = float(g("dead_regen_mult", _def_mult))
        self.rest_regen_mult = float(g("rest_regen_mult", _def_mult))

        # 逐格计时器 / 状态
        self._rest_until = np.full(self.n_cells, -1, dtype=np.int64)
        self._dead = np.zeros(self.n_cells, dtype=bool)
        self._dead_since = np.full(self.n_cells, -1, dtype=np.int64)
        # 波2 修 v2：**累计损伤**（= Σ(被吃量/容量)，随快照走；休耕到期按 recovery 衰减）
        self._damage = np.zeros(self.n_cells, dtype=np.float64)
        # 本 tick 刚死亡的斑块格（等待 `rotate` 把加成搬走；**必须初始化**，
        # 否则 `note_tick` 与 `rotate` 分属两段时会出现"加成悬空"的中间态）
        self._demoted = np.zeros(self.n_cells, dtype=bool)

        # 🔴 A2（轻舟 2026-09-30）：容量缓存 + 掩码版本号（缓存失效依据）。
        #   `_cap_cache` 非状态量 ⇒ 不进快照；load 后由引擎调 `invalidate_capacity()` 复位。
        #   `_mask_ver` 每次掩码写 +1（唯一失效通道：掩码变 ⇒ 容量变）。
        self._cap_cache: Optional[np.ndarray] = None
        self._mask_ver = 0
        # 🔴 A2-④（轻舟）：`rotate` 子集扫描的派生索引（非状态量 ⇒ 不进快照）。
        #   不变量：`_dead_idx == flatnonzero(_dead)`、`_rest_idx == flatnonzero(_rest_until >= 0)`，
        #   两者**升序唯一**（force 的 stable argsort 依赖升序；重复项不可能出现）。
        #   外部直改 `_dead`/`_rest_until`（旧档/调试/测试）后必须调 `rebuild_indexes()`。
        self._dead_idx = np.zeros(0, dtype=np.int64)
        self._rest_idx = np.zeros(0, dtype=np.int64)

        # 计数器（只增）
        self.rest_set_n = 0        # 累计"设过休耕"的 (格·次)
        self.patch_kill_n = 0      # 累计死亡事件
        self.patch_reborn_n = 0    # 累计"死格到期重生"事件
        self.forced_reborn_n = 0   # 累计"反荒漠化闸"强制重生事件
        self.promote_n = 0         # 累计"斑块加成搬迁"事件

        # 构造期基线（`conservation_check` / `validate_writeback` 用）
        self._n_mask_0 = int(self._mask.sum()) if self._mask is not None else 0
        self._cap_total_0 = float(self.capacity_from_base().sum())

    # ---- 构造（便捷入口） ---------------------------------------------------

    @classmethod
    def from_field(cls, cfg: Any, rfield: Any, world: Any) -> "ResourceDynamics":
        """从 `ResourceField` + `SphereWorld` 读构造期常量（**只读**，不 import 引擎）。

        ⚠️ 需要 `ResourceField` 的三个内部量：`_patch_mask` / `_capacity` / `distribution`。
        它们本身就是构造期只读量（`_capacity` 只在轮作时由本模块**经引擎**重算）。

        🔴 R328 甲（轻舟 2026-10-02）：**背景产能是否为零改读显式真值**
        `rfield.bg_production_zero`，不再用 `cap[~mask][0]`（背景第 0 格）反推 ——
        带低产能带（`bg_low_prod_frac > 0`）时第 0 格偶发落在带内（容量 > 0）
        ⇒ 全局倍率被误判为 0.05 ⇒ 整个背景（含荒漠）被判成有产能 = **"抽奖"**
        （R300 缺陷：per-seed 随机分成 A 支"纯斑块"/B 支"全图可产"两支）。
        显式 True ⇒ 确定性 `b_mult = 0.0`（纯斑块语义）；显式 False（均匀背景档）
        ⇒ 保留原单格推断（背景均匀时首格即精确值，逐位兼容）。
        """
        n = int(world.n_cells)
        mask = getattr(rfield, "_patch_mask", None)
        cap = np.asarray(rfield._capacity, dtype=np.float64)
        area = np.asarray(world.cell_area(np.arange(n)), dtype=np.float64)
        base = float(rfield.capacity_per_area) * area
        if mask is None or not np.asarray(mask).any():
            p_mult, b_mult = 1.0, 1.0
        else:
            m = np.asarray(mask, dtype=bool)
            p_mult = float(cap[m][0] / base[m][0])
            if bool(getattr(rfield, "bg_production_zero", False)):
                b_mult = 0.0
            else:
                b_mult = float(cap[~m][0] / base[~m][0])
        return cls(
            n, cfg, patch_mask=mask, cols=int(world.cols),
            base_capacity=base, patch_capacity_mult=p_mult, bg_capacity_mult=b_mult,
            patch_regrowth_mult=float(getattr(rfield, "_patch_regrowth_mult", 2.0)),
        )

    # ---- ① 再生闸 -----------------------------------------------------------

    def growth_multiplier(self, idx: Optional[np.ndarray] = None) -> np.ndarray:
        """本 tick 每格的**再生乘子**（0 = 不生长，1 = 照常）。

        🔴 R226 修复③（核心，本次灭绝根因）：原实现"休耕 或 已死 ⇒ 乘子 0"。
        在 `bg_production_zero=True`（背景格容量=0、**斑块格是唯一粮仓**）的世界里，
        "停产"等价于**永久砍产能**——实测仅 2.6% 斑块格停产就让 K 从 ~700 崩到 ~80
        （正常世界背景格贡献 67.7% 容量、兜得住，S1 世界没有任何兜底）。修法：

          · **已死格**：仍按 `dead_regen_mult` 生产（默认 0.0 = 旧行为；在
            `bg_production_zero=True` 下自动改 0.5 —— "死而富集"：受损但不停产）。
          · **休耕格**：按 `rest_regen_mult`（默认 0.0 = 旧行为）生产；休耕语义
            由"停产"改为"减产"（轮牧的休养期不是绝收期）。
          · `enabled=False` ⇒ 全 1（调用方可无条件相乘，C7 逐位等价不变）。

        idx : NDArray[int64] 或 None
            🔴 A2（轻舟）：只算这些格（返回形状 = idx 形状）。None ⇒ 全场，
            与旧行为逐位不变。子集版逐元素同式（dead 段 → rest 段 min 次序一致）。
        """
        if idx is None:
            if not self.enabled:
                return np.ones(self.n_cells, dtype=np.float64)
            mult = np.ones(self.n_cells, dtype=np.float64)
            if self._dead.any():
                mult[self._dead] = self.dead_regen_mult
            resting = self._rest_until >= 0
            if resting.any():
                mult[resting] = np.minimum(mult[resting], self.rest_regen_mult)
            return mult
        if not self.enabled:
            return np.ones(idx.size, dtype=np.float64)
        mult = np.ones(idx.size, dtype=np.float64)
        d = self._dead[idx]
        if d.any():
            mult[d] = self.dead_regen_mult
        rest = self._rest_until[idx] >= 0
        if rest.any():
            mult[rest] = np.minimum(mult[rest], self.rest_regen_mult)
        return mult

    def capacity_multiplier(self) -> np.ndarray:
        """每格的**容量倍率**（斑块倍率 / 背景倍率）。轮作搬移会改变它。"""
        return np.where(self._mask, self.patch_capacity_mult, self.bg_capacity_mult)

    def capacity_from_base(self) -> np.ndarray:
        """按当前掩码重算逐格容量（= `base_capacity × 倍率`）。

        ⚠️ 引擎侧需要把结果写回 `ResourceField._capacity`（轮作后才生效）。
        🔴 A2（轻舟）：结果**带缓存**（掩码版本不变 ⇒ 直接复用；掩码写处调
        `invalidate_capacity()` 失效）。返回的是缓存本体 ⇒ **调用方不得就地修改**。
        """
        if self._cap_cache is None:
            self._cap_cache = self._base_capacity * self.capacity_multiplier()
        return self._cap_cache

    def invalidate_capacity(self) -> None:
        """掩码变更 ⇒ 容量缓存失效 + 版本号自增（引擎用它判"回写是否需重算"）。"""
        self._cap_cache = None
        self._mask_ver += 1

    # ---- ② 取食后：休耕 + 判死 ----------------------------------------------

    def note_tick(self, intake: np.ndarray, growth: np.ndarray, tick: int,
                  eaten_idx: Optional[np.ndarray] = None) -> None:
        """结算"这一 tick 被吃了多少"：累计损伤 + 判休耕 + 判死亡。

        参数
        ----
        intake : NDArray[float64]，形状 (n_cells,)
            每格本 tick **被吃掉的量**（质量单位，`np.bincount(flats, weights=taken)`）。
        growth : NDArray[float64]，形状 (n_cells,)
            每格本 tick 的**名义再生量**（未乘 `growth_multiplier`）。
            仅作 `kill_frac` 极端密度通道的分母；主通道分母 = 容量。
        tick : int
            当前时间步（休耕截止与死亡时刻按它记）。
        eaten_idx : NDArray[int64] 或 None
            🔴 A2（轻舟）：本 tick 被吃格索引（**契约 = `np.flatnonzero(intake)`**）。
            引擎稀疏路径传入 ⇒ 省一次全场扫描；None ⇒ 内部自行 flatnonzero（等价）。

        说明（波2 修 v2，fish 批准方案 A；替代旧"intake>0 即休耕"）
        ----
        * **累计损伤** `_damage += intake / capacity`（容量 = 当期含倍率容量）：
          "吃掉相当于几格满存量的量"。轻取食（一次 0.9 / 容量 ~50 ≈ 0.02）增速慢，
          反复啃食（文献：反复摘叶 → 根衰减）才累积到阈值。
        * **休耕**：`_damage ≥ rest_threshold`（默认 0.3）⇒ 进入休耕（固定
          `rest_ticks` 时长；⚠️ **只对未休耕格设置 ⇒ 休耕期被吃不刷新** ——
          方案 A 的核心，也是轮牧"休养期不再被啃食累积"的语义）。
        * **判死**：主通道 = `_damage ≥ death_threshold`（默认 0.8，全格生效；
          对应 USDA 摘叶梯度"70–90% 重伤近死"）；补充通道 = 单 tick
          `intake/growth > kill_frac`（保留极端密度瞬间死亡；仍受 kill_patch_only）。
        * **休耕期可被吃**（方案 A）：休耕格存量若还在，个体照常取食 ⇒ intake>0
          ⇒ `_damage` 继续累计 ⇒ 可能推进到死亡（"被啃食的休耕地退化"）。
        * `kill_patch_only`：仅约束**补充通道**；主通道（累计损伤）全格生效 ——
          背景格重度退化同样可死，由反荒漠化闸（dead_cell_max_frac）兜底强制重生。

        🔴 A2（轻舟 2026-09-30）：全链**子集化**（只结算被吃格）——逐位等价：
        逐元素同式/同 dtype/同运算次序；死亡与休耕写点的索引升序与全场版
        `flatnonzero` 一致（搬移循环的 RNG 消费次序因此不变）。
        """
        if not self.enabled:
            return
        intake = np.asarray(intake, dtype=np.float64)
        growth = np.asarray(growth, dtype=np.float64)

        e = (np.flatnonzero(intake) if eaten_idx is None
             else np.asarray(eaten_idx, dtype=np.int64))
        if e.size == 0:
            return

        # --- 累计损伤（主通道分母 = 当期容量；容量 > 0 恒成立）---
        cap_e = self.capacity_from_base()[e]      # A2：缓存 gather（版本未变 ⇒ 零重算）
        denom_e = np.where(cap_e > 0.0, cap_e, 1.0)
        self._damage[e] += intake[e] / denom_e

        # --- 判死：主通道（累计损伤超阈，全格）+ 补充通道（单 tick 极端密度）---
        dead_e = self._dead[e]                    # 读于任何 _dead 写之前（与全场版同序）
        newly_sel = (~dead_e) & (self._damage[e] >= self.death_threshold)
        if self.kill_denom == "regrowth":
            dk_e = growth[e]
        else:
            dk_e = self._base_capacity[e]
        valid_e = (dk_e > 0.0) & (~dead_e)        # 全场版另有 `intake > 0`：子集上恒真
        if self.kill_patch_only:
            valid_e &= self._mask[e]
        if valid_e.any():
            ratio_e = np.zeros(e.size, dtype=np.float64)
            ratio_e[valid_e] = intake[e][valid_e] / dk_e[valid_e]
            newly_sel |= valid_e & (ratio_e > self.kill_frac)
        newly_idx = e[newly_sel]                  # 升序（e 升序 ⇒ 与全场 flatnonzero 同序）
        n_new = int(newly_idx.size)
        if n_new:
            self._dead[newly_idx] = True
            self._dead_since[newly_idx] = tick
            self._rest_until[newly_idx] = -1
            self._damage[newly_idx] = 0.0          # 死亡清零（重生后从 0 累计）
            self.patch_kill_n += n_new
            # 🔴 A2-④：派生索引维护（死亡 ⇒ 退出休耕索引 + 加入死亡索引；升序保持）
            self._rest_idx = self._idx_drop(self._rest_idx, newly_idx)
            self._dead_idx = self._idx_merge(self._dead_idx, newly_idx)
            # 🔴 R226 修复①（核心）：斑块格死亡 ⇒ **立刻、本 tick 内**把斑块加成搬到
            #   同行背景格（原子搬移）。旧实现把它推到**下一 tick** 的 `rotate()` 里做
            #   ⇒ 在 `bg_production_zero=True`（背景格容量=0、永久荒漠）的世界里，
            #   死格降级与加成落位**跨 tick 分离** ⇒ 延迟窗口内"生产格数"净减少
            #   （实测 tick500 mask 12805→12720、每 tick 取食 −16% ⇒ 物种灭绝）。
            #   现在降级与落位同一 tick 完成 ⇒ **每 tick 内生产格守恒**。
            _demoted = newly_idx[self._mask[newly_idx]]
            if _demoted.size:
                if self.rotate_moves_patch and not self.bg_production_zero:
                    # 正常世界（背景格有产能）：降级 + 同 tick 原子搬移加成到新格
                    self._mask[_demoted] = False
                    self.invalidate_capacity()   # A2：掩码变 ⇒ 容量缓存失效
                    for i in _demoted.tolist():
                        if not self._promote_same_row(int(i)):
                            # 无候选格 ⇒ 撤回降级（保格优先）
                            self._mask[i] = True
                            self.invalidate_capacity()
                # 🔴 bg_production_zero=True（斑块=唯一粮仓）⇒ **不搬移**：
                #   死斑块格**保留斑块身份**（死而富集），原地"重生"时不丢粮仓位置。
                #   理由见 `__init__` 的 `bg_production_zero` 注释：搬移会让粮仓
                #   位置迁移、个体站位错配 ⇒ 灭绝。轮作只体现在"再生闸"（死亡期停长）。
                # 搬移已即时完成/无需搬移 ⇒ 清零 `_demoted`
                self._demoted[:] = False

        # --- 休耕：损伤超阈且未休耕未死 ⇒ 进入休耕（固定时长，**不刷新**）---
        # 🔴 读 `_dead`/`_rest_until`/`_damage` 的**更新后**值（本段在死亡写之后，与全场版同序）
        enter_rest = (
            (~self._dead[e]) & (self._rest_until[e] < 0)
            & (self._damage[e] >= self.rest_threshold)
        )
        if enter_rest.any():
            er_idx = e[enter_rest]
            self._rest_until[er_idx] = tick + self.rest_ticks
            self.rest_set_n += int(er_idx.size)
            self._rest_idx = self._idx_merge(self._rest_idx, er_idx)   # A2-④

    # ---- ③a 派生索引维护（A2-④：rotate 子集扫描的前置）------------------------

    def rebuild_indexes(self) -> None:
        """由状态数组重建派生索引（**外部直改 `_dead`/`_rest_until` 或 load 后必调**）。

        🔴 A2-④（轻舟）：`rotate` 只扫 `_rest_idx`（休耕格）/`_dead_idx`（死格），不再
        全场扫 460800 格。两索引是**派生量**（不进快照），不变量：
          · `_dead_idx == flatnonzero(_dead)`（升序）
          · `_rest_idx == flatnonzero(_rest_until >= 0)`（升序）
        绕过 `note_tick`/`rotate` 的直接数组写（快照恢复、调试、测试桩）都必须随后
        调用本函数，否则 rotate 会漏结算（不炸但结果不同 = 静默）。
        """
        self._dead_idx = np.flatnonzero(self._dead)
        self._rest_idx = np.flatnonzero(self._rest_until >= 0)

    def indexes_consistent(self) -> bool:
        """A2-④ 不变量自检（O(n)，供测试/诊断用；热路径不调）。"""
        return (np.array_equal(self._dead_idx, np.flatnonzero(self._dead))
                and np.array_equal(
                    self._rest_idx, np.flatnonzero(self._rest_until >= 0)))

    @staticmethod
    def _idx_merge(base: np.ndarray, add: np.ndarray) -> np.ndarray:
        """升序归并（唯一化）；`add` 为空 ⇒ 原样返回（不复制）。"""
        if add.size == 0:
            return base
        return np.union1d(base, add)

    @staticmethod
    def _idx_drop(base: np.ndarray, drop: np.ndarray) -> np.ndarray:
        """从升序 `base` 中剔除 `drop`（保持升序）；任一为空 ⇒ 原样返回。"""
        if base.size == 0 or drop.size == 0:
            return base
        return base[~np.isin(base, drop)]

    # ---- ③ 轮作：重生 + 搬加成 + 反荒漠化闸 ---------------------------------

    def rotate(self, tick: int) -> None:
        """死格到期重入候选池 + 到期休耕恢复 + 反荒漠化闸。

        🔴 R226 修复①：**斑块加成搬移已移到 `note_tick` 内**（死亡当 tick 原子完成）
        ⇒ 本函数不再做搬移、不再消费 RNG（`rand_u` 参数已移除）。
        """
        if not self.enabled:
            return

        # --- ① 防御性补搬（正常情况下 `note_tick` 已即时搬完 ⇒ 此处恒空）---
        if self._demoted.any():
            for i in np.flatnonzero(self._demoted):
                self._promote_same_row(int(i))
            self._demoted[:] = False

        # --- ② 到期休耕恢复（2026-09-23 实验：B–E 臂灭绝根因 + 波2 修 v2）---
        # 休耕到期 ⇒ `_rest_until` 重置 -1（恢复生长）+ **累计损伤按 `damage_recovery`
        # 衰减**（文献：恢复期后损伤部分恢复）。此前只有"死亡重生 / 反荒漠化闸"
        # 重置它 ⇒ 一次被吃 = 永久休耕 ⇒ resting_cell_frac 单调冲到 ~95%。
        # 🔴 A2-④（轻舟）：改**子集扫描** `_rest_idx`（不变量见 `rebuild_indexes` 注释）——
        #   逐位等价：全场版三项合成 `(rest_until≥0)∧(¬dead)∧(tick≥rest_until)` 在子集上
        #   逐元素同式（`rest_until≥0` 由不变量恒真；陈旧项自愈剔除，防外部直改漏维护）。
        er = self._rest_idx
        if er.size:
            rt = self._rest_until[er]
            _stale = rt < 0
            if _stale.any():                    # 自愈：陈旧项（外部直改数组/旧档遗留）
                er = er[~_stale]
                rt = rt[~_stale]
                self._rest_idx = er
            if er.size:
                expired = (~self._dead[er]) & (tick >= rt)
                if expired.any():
                    ex_idx = er[expired]
                    self._damage[ex_idx] *= self.damage_recovery
                    self._rest_until[ex_idx] = -1
                    self._rest_idx = er[~expired]

        # --- ③ 到期重生：死格 → 背景格（"重入候选池"）---
        # 🔴 A2-④：子集扫描 `_dead_idx`（升序 ⇒ 与全场版 `flatnonzero(_dead)` 同集同序）。
        di = self._dead_idx
        if di.size:
            _live = self._dead[di]
            if not _live.all():                 # 自愈：陈旧项（外部直改数组/旧档遗留）
                di = di[_live]
                self._dead_idx = di
            if di.size:
                due = (tick - self._dead_since[di]) >= self.dead_regen_ticks
                if due.any():
                    due_idx = di[due]
                    self._dead[due_idx] = False
                    self._rest_until[due_idx] = -1
                    self._dead_since[due_idx] = -1
                    self.patch_reborn_n += int(due_idx.size)
                    self._dead_idx = di[~due]

        # --- ③ 反荒漠化闸：死格占比超阈 ⇒ 强制重生（从最老的开始）---
        max_dead = int(self.dead_cell_max_frac * self.n_cells)
        n_dead = int(self._dead_idx.size)
        if n_dead > max_dead:
            need = n_dead - max_dead
            cand = self._dead_idx
            order = cand[np.argsort(self._dead_since[cand], kind="stable")]
            force = order[:need]
            self._dead[force] = False
            self._dead_since[force] = -1
            self._rest_until[force] = -1
            self.forced_reborn_n += int(force.size)
            self._dead_idx = self._idx_drop(self._dead_idx, force)

    # ---- ④ 修复②：回写前 diff 校验（R226 裁定②）----------------------------

    def validate_writeback(self, cap_after: np.ndarray, mask_after: np.ndarray) -> dict:
        """回写前校验：搬移只该**换位置**，不该改「生产格数」或「总容量」。

        🔴 R226 修复②（裁定②"回写前 diff 校验"）：本函数作**兜底防线**（非唯一防线）。
        返回读数；发现不变量破裂即 **fail-loud**（raise），不给"静默荒漠化"留缝。

        不变量：
          · `mask.sum()`（生产格数）**恒等于初始值** —— 搬移只搬加成、不增不减格数
            （本次灭绝 bug 的漏检点：tick500 生产格 12805→12720）。
          · `Σcapacity` 与初始值**相对差 < 1e-9**（`bg_production_zero=False` 时守恒；
            `=True` 时"斑块部分守恒"——生产格数不变 ⇒ 斑块总容量不变）。
        """
        if not self.enabled:
            return {"checked": False}
        n_mask = int(np.asarray(mask_after).sum())
        cap_sum = float(np.asarray(cap_after).sum())
        n_mask0 = self._n_mask_0
        cap_sum0 = self._cap_total_0
        mask_ok = (n_mask == n_mask0)
        cap_rel = abs(cap_sum - cap_sum0) / max(abs(cap_sum0), 1e-12)
        cap_ok = (cap_rel < 1e-9)
        out = {
            "checked": True,
            "mask_n": n_mask, "mask_n0": n_mask0, "mask_ok": mask_ok,
            "cap_sum": cap_sum, "cap_sum_0": cap_sum0,
            "cap_rel_diff": cap_rel, "cap_ok": cap_ok,
        }
        if not mask_ok:
            raise RuntimeError(
                f"🔴 rd 不变量破裂：生产格数 {n_mask0} → {n_mask}（搬移只该换位置、不该丢格）。"
                "这正是 bg_production_zero=True 下导致灭绝的漏洞 —— 请检查搬移原子性。"
            )
        if not cap_ok:
            raise RuntimeError(
                f"🔴 rd 不变量破裂：Σcapacity 相对漂移 {cap_rel:.3e} > 1e-9 "
                f"（{cap_sum0} → {cap_sum}）—— 搬移破坏了容量守恒。"
            )
        return out

    def _promote_same_row(self, dead_cell: int) -> bool:
        """把斑块加成搬到一个**与 `dead_cell` 同行**的随机背景格（逐位守恒）。

        🔴 R226 修复①②：改用**自有 RNG**（`self._rng`）⇒ 不再消费引擎随机流；
        调用时机由 `rotate()` 改到 `note_tick()` 死亡判定内（同 tick 原子搬移）。

        Returns
        -------
        bool : 是否成功搬移。**False ⇒ 未找到候选格**（调用方必须回退：
               `bg_production_zero=True` 下"降级却不落位"= 永久丢一个生产格 ⇒ 灭绝）。
        """
        row = self._row
        if self.rotate_same_row_only:
            cand = np.flatnonzero((~self._mask) & (row == row[dead_cell]) & (~self._dead))
        else:
            cand = np.flatnonzero((~self._mask) & (~self._dead))
        if cand.size == 0:
            # 🔴 无候选格 ⇒ 返回 False，由调用方**撤回降级**（保格优先）。
            #   ⚠️ 绝不跨行搬：跨行交换会破坏 Σcapacity 守恒（docstring (C) 实测 +1.886e+03）。
            return False
        j = int(cand[self._rng.integers(0, cand.size)])
        self._mask[j] = True
        self.invalidate_capacity()   # A2：掩码写点（含 rotate 防御补搬路径）⇒ 缓存失效
        self.promote_n += 1
        return True

    # ---- 读数与自检 ---------------------------------------------------------

    def probe(self) -> dict:
        """四条必读读数（R176 §12.5）+ 有效产能。**关档 ⇒ None（未观测，非 0）**。"""
        if not self.enabled:
            return {
                "enabled": False,
                "dead_cell_frac": None, "resting_cell_frac": None,
                "patch_kill_n": None, "patch_reborn_n": None,
                "forced_reborn_n": None, "mean_capacity_effective": None,
                "note": "未启用 ⇒ 未观测（None，非 0）；开档后才有意义",
            }
        cap_total = float(self.capacity_from_base().sum())
        return {
            "enabled": True,
            "dead_cell_frac": round(float(self._dead.mean()), 6),
            "resting_cell_frac": round(float((self._rest_until >= 0).mean()), 6),
            "rest_threshold": float(self.rest_threshold),
            "death_threshold": float(self.death_threshold),
            "damage_recovery": float(self.damage_recovery),
            "mean_damage": round(float(self._damage.mean()), 6),
            "damage_over_rest_frac": round(
                float((self._damage >= self.rest_threshold).mean()), 6),
            "patch_kill_n": int(self.patch_kill_n),
            "patch_reborn_n": int(self.patch_reborn_n),
            "forced_reborn_n": int(self.forced_reborn_n),
            "promote_n": int(self.promote_n),
            "rest_set_n": int(self.rest_set_n),
            "dead_n": int(self._dead.sum()),
            "patch_cells": int(self._mask.sum()),
            "mean_capacity_effective": round(cap_total / max(1, self.n_cells), 6),
            "capacity_total_rel": round(cap_total / max(1e-12, self._cap_total_0), 9),
            "note": ("波2 修 v2 三态机：damage=Σ(被吃/容量)｜休耕阈值 "
                     + f"{self.rest_threshold}｜死亡阈值 {self.death_threshold}"
                     + f"｜kill_frac={self.kill_frac}（极端密度补充通道）"
                     + f"｜kill_patch_only={self.kill_patch_only}"
                     + "｜⚠️ 只看人口会漏掉'世界正在荒漠化但还没崩'的中间态 ⇒ 四条读数缺一不可"),
        }

    def conservation_check(self) -> dict:
        """两条构造期不变量的自检（R145 C9 家族 / 波 2 §五）。

        * `capacity_total_rel`：`Σcapacity` 相对构造期的比值 ⇒ 轮作不该改变它（应恒 = 1）
        * `regrow_area_weighted`：`Σ(再生倍率 × 面积) / Σ面积` ⇒ 应恒 = 1.0
          （斑块面积不变时背景倍率也不必重算；**同行交换**保证这一点）
        """
        if not self.enabled:
            return {"enabled": False, "capacity_total_rel": None,
                    "regrow_area_weighted": None, "note": "未启用（未观测）"}
        area = self._base_capacity / max(1e-12, float(self._base_capacity.sum()))
        pa = float(area[self._mask].sum())
        ba = float(area[~self._mask].sum())
        if ba > 0:
            bg_rg = (1.0 - self.patch_regrowth_mult * pa) / ba
        else:
            bg_rg = 1.0
        rg = float((np.where(self._mask, self.patch_regrowth_mult, bg_rg) * area).sum())
        cap_total = float(self.capacity_from_base().sum())
        return {
            "enabled": True,
            "capacity_total_rel": round(cap_total / max(1e-12, self._cap_total_0), 9),
            "cap_total_now": round(cap_total, 6),
            "cap_total_0": round(self._cap_total_0, 6),
            "regrow_area_weighted": round(rg, 9),
            "note": ("两条都应恒为 1；偏离 ⇒ 轮作破坏了构造期守恒 "
                     "（先查是否跨行交换 / 面积权重口径）"),
        }
