"""ResourceField：球面世界的资源场（模块一 · 文件 3）。

模块职责（这一份是"世界的食物分布"）
-----------------------------------
前两份文件把"土地"（SphereWorld）和"天气"（LightAndTemperature）
准备好了，这一份在土地上铺设"食物"：
- 每块格子能存多少食物（容量）由它的【面积】决定：
  赤道格子大 → 能存得多；极点格子小 → 只能存一点点，
  所以极地天然"养不活多少生物"（竞争少）；
- 食物会随时间【缓慢恢复】（再生），恢复速度受【温度】影响：
  太冷的地方（极地、深夜）恢复慢，热带正午恢复快；
- 生物【吃掉】食物时，格子里的食物减少（见 consume）。

为什么这样设计（和你的设想对齐）
--------------------------------
你提到过"极地温度更低、生物竞争偏少"，这里的容量缩放在空间上
实现了"极地养不活多少"，再生受温度抑制在时间上实现了"冷的地方
食物长得慢"。两件事都不需要写死规则，是由环境自然涌现的。

时间约定：regrow(tick) 接收当时的 tick 计算温度因子，
所以昼夜会带来"白天恢复快、晚上恢复慢"的节律。
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from world.sphere_world import SphereWorld
from world.light_and_temperature import LightAndTemperature


class ResourceField:
    """网格上的可再生的食物场（布局按 [row][col] 平铺一维数组）。

    实例属性（__slots__ 声明的全部字段）说明
    ---------------------------------------
    world : SphereWorld
        所属网格（提供面积权重 cell_area 与索引转换）。
    lt : LightAndTemperature
        光照温度场（提供温度，用于再生速度的时间调节）。
    capacity_per_area : float
        单位面积的食物容量。格子容量 = 该值 × 格子面积权重。
        （面积权重近赤道≈1，极点≈0.026，因此极点容量很小）
    regrowth_rate : float
        基准再生速度（每 tick 每格恢复的食物量，温度因子为 1 时）。
    temp_sensitivity : float
        再生受温度影响的强弱（0=完全不看温度，越大越看温度）。
    light_sensitivity : float
        再生受**光照强度**影响的强弱（0=完全不看光，与旧机制完全一致）。
        >0 时，再生量额外乘 clip(光照,0,1)^light_sensitivity（"光合作用"）。
        与季节机制（LightConfig.tilt_rad/season_period）合用时：
        夏季日照长 ⇒ 该半球再生快 ⇒ 食物丰度带随季节南北移动（"绿浪"）。
    light_normalize : bool
        是否把光照因子按全球均值归一化（保持全球平均再生量不变）。
        🔴 它是**保总量**的开关：开启后季节只改变再生的**空间分布**（南北此消彼长），
        不改变全球总量 ⇒ 与光照的零均值特性一致，回归季节关闭时必须无影响。
        关闭时全球总量在极昼/极夜间整体起伏（信号更强、但格局会随半周期整体涨落）。
    _grid : NDArray[float64], 形状 (n_cells,) 平铺
        每个格子当前的食物存量（扁平数组，通过 flat 索引访问）。
    _capacity : NDArray[float64], 形状 (n_cells,) 平铺
        每个格子的食物容量上限（构造时按面积算好，之后只读）。
    """

    __slots__ = (
        "world",
        "lt",
        "capacity_per_area",
        "regrowth_rate",
        "temp_sensitivity",
        "light_sensitivity",
        "light_normalize",
        "distribution",
        "_grid",
        "_capacity",
        "_patch_mask",
        "_patch_regrowth_mult",
        "_bg_regrowth_mult",
        "bg_production_zero",
        # ---- 🔴 R242 背景低产能带（默认全 False/0 = 现行为）----
        "_bg_low_mask",
        "_bg_low_cap_mult",
        "_bg_low_regrowth_mult",
        # ---- R217 §五 #1 稀疏化 B：惰性再生（默认关；见 `enable_lazy`）----
        "_lazy",
        "_dirty_mask",
    )

    def __init__(
        self,
        world: SphereWorld,
        lt: LightAndTemperature,
        capacity_per_area: float = 40.0,
        regrowth_rate: float = 0.5,
        temp_sensitivity: float = 1.0,
        light_sensitivity: float = 0.0,
        light_normalize: bool = False,
        distribution: str = "uniform",
        patch_count: int = 30,
        patch_radius: int = 2,
        patch_capacity_mult: float = 3.0,
        # ⚠️ 与 `ResourceConfig.patch_regrowth_mult` **同一默认值**（13.5 ② = 3.0）：
        #   两处默认值若漂移，直接构造 ResourceField 的测试/工具会与引擎得到不同世界
        #   （R163 §P1-1「三处默认值不一致」同族）
        patch_regrowth_mult: float = 3.0,
        background_fill: float = 0.1,
        initial_fill: float = 0.5,
        patch_seed: int = 42,
        bg_production_zero: bool = False,
        # ---- 🔴 R242 背景低产能带（默认 0 = 现行为逐位等价）----
        bg_low_prod_frac: float = 0.0,
        bg_low_cap_mult: float = 0.0,
    ) -> None:
        """铺好初始食物。

        uniform（默认）：每格均匀填充到容量的 initial_fill 倍，行为与旧版完全一致。
        patchy：食物聚簇到斑块，背景压低；两条守恒保证总食物量不变：
          ① 容量守恒：Σ_capacity 与 uniform 版相等（种群承载上限不变）
          ② 再生守恒：周期平均总再生量与 uniform 版相等（时间上供给不变）
        patch 中心由独立 rng（patch_seed）选择，不消费调用方的 rng，
        保证不影响引擎的 RNG 消费顺序（同 seed 同结果）。

        参数
        ----
        world, lt, capacity_per_area, regrowth_rate, temp_sensitivity :
            同旧版。
        light_sensitivity : 🔴 **R196 光驱动再生**（2026-09-24）——
            再生量额外乘 `clip(光照,0,1)^light_sensitivity`。光照本身由
            `LightConfig.tilt_rad`/`season_period` 驱动正弦季节摆动 ⇒
            开季节后南北半球日照此消彼长 ⇒ 食物丰度带随季节南北移动。
            物理原型：光合作用有效辐射（PAR）随日照时长/太阳高度角变化
            —— 即"日照长短决定初级生产力"（Sverdrup 临界深度、绿浪假说）。
            默认 0.0 = **完全不看光** ⇒ 与旧版逐位一致（回归安全）。
        light_normalize : 见类文档。默认 False。
        distribution : "uniform" | "patchy"，默认 uniform。
        patch_count : 斑块中心数。
        patch_radius : 斑块半径（邻居扩散层数）。
        patch_capacity_mult : 斑块格容量倍率（必须 > 1）。
        patch_regrowth_mult : 斑块格再生倍率。
        background_fill : 背景格初始食物占比（0~1，建议压低）。
        initial_fill : uniform 模式的初始填充比例（0~1）。
        patch_seed : patchy 模式选中心的独立随机种子。
        bg_production_zero : 🔴 **13.5 ①**（2026-09-24）背景产能归零 ——
            背景格**容量与再生都置 0**（初始存量也随之为 0）⇒ 只留斑块生产。
            目的：让"食物"第一次成为**绑定约束**（现状背景贡献 67.7% 容量 / 78% 再生 ⇒
            解析 K ≈ 12 400 ≫ 软顶 1 944 ⇒ 食物从不绑定 ⇒ 位置无信息价值）。
            ⚠️ 开启后**有意打破**两条构造期守恒（Σcapacity → 斑块部分 32.3%；再生守恒式 → 不成立）
            —— 这是本机制的目的，不是缺陷。默认 False = 现状（既有 patchy 批仍可复现）。
        """
        self.world = world
        self.lt = lt
        self.capacity_per_area = float(capacity_per_area)
        self.regrowth_rate = float(regrowth_rate)
        self.temp_sensitivity = float(temp_sensitivity)
        self.light_sensitivity = float(light_sensitivity)
        self.light_normalize = bool(light_normalize)
        if self.light_sensitivity < 0.0:
            raise ValueError(
                f"light_sensitivity 必须 ≥ 0，收到 {light_sensitivity}"
            )
        self.distribution = distribution

        areas = world.cell_area(np.arange(world.n_cells))
        base_cap = areas * capacity_per_area

        if distribution == "patchy":
            # 1) 选斑块中心（独立 rng，不碰调用方 rng → 不影响引擎 RNG 消费顺序）
            rng = np.random.default_rng(patch_seed)
            centers = rng.choice(
                world.n_cells,
                size=min(patch_count, world.n_cells),
                replace=False,
            )
            # 2) 邻居扩散 radius 层 → patch_mask（构造时一次性，可接受）
            patch_mask = np.zeros(world.n_cells, dtype=bool)
            patch_mask[centers] = True
            for _ in range(patch_radius):
                cells = np.flatnonzero(patch_mask)
                all_nb = np.concatenate([world.neighbors(int(c)) for c in cells])
                patch_mask[all_nb] = True
            self._patch_mask = patch_mask
            # 3) 容量守恒：patch 格 × mult，背景格 × bg_mult，Σ 与 uniform 版相等
            #   🔴 13.5 ①（2026-09-24）：`bg_production_zero=True` ⇒ **背景容量直接归零**
            #     （只留斑块生产）。此时**有意打破**容量守恒（Σcapacity 降到斑块部分 32.3%），
            #     所以不能再用 `bg_cap_mult = (total − patch·mult)/bg_area` 那条守恒式，
            #     也必须跳过下面 `bg_cap_mult <= 0` 的报错（归零是**目的**，不是参数过大的症状）。
            patch_area = float(areas[patch_mask].sum())
            bg_area = float(areas[~patch_mask].sum())
            total_area = float(areas.sum())
            if bg_production_zero:
                bg_cap_mult = 0.0
            else:
                bg_cap_mult = (total_area - patch_area * patch_capacity_mult) / bg_area
                if bg_cap_mult <= 0:
                    raise ValueError(
                        f"斑块容量倍率过大：patch_capacity_mult={patch_capacity_mult}, "
                        f"patch_count={patch_count} 导致背景容量为负。请调小倍率或斑块数。"
                    )
            # ---- 🔴 R242 背景低产能带：子采样部分背景格给极低容量 ----
            #   依据：R241 根因（背景全零 ⇒ 沙漠不可穿越 ⇒ 门B 结构性失败）。
            #   做法：在**零产能背景**之上，按 bg_low_prod_frac 抽出**一部分背景格**，
            #     给它们 `base_cap × bg_cap_mult`（极低）⇒ 稀疏"绿洲链"续命带。
            #   🔴 用**独立 rng**（从 patch_seed 派生另一条流），**不消费引擎 self.rng**
            #     ⇒ 不影响引擎 RNG 消费顺序（同 seed 同结果；C7 逐位等价）。
            #   ✅ bg_low_prod_frac = 0（默认）⇒ 整块跳过 ⇒ 与原路径**逐位相同**。
            bg_low_mask = np.zeros(world.n_cells, dtype=bool)
            if (not bg_production_zero) and bg_low_prod_frac > 0.0:
                # 注意：`bg_production_zero=False` 时背景本就有"守恒摊均"产能，
                # 低产能带会被摊均值覆盖 ⇒ 语义冲突。故本机制**只在背景归零时有意义**。
                raise ValueError(
                    "bg_low_prod_frac 只在 bg_production_zero=True 时有意义"
                    "（否则背景已有守恒摊均产能，低产能带被覆盖）。"
                    "请设 bg_production_zero=True + bg_cap_mult 显式低倍率。"
                )
            if bg_production_zero and bg_low_prod_frac > 0.0:
                bg_idx = np.flatnonzero(~patch_mask)
                n_low = int(round(bg_low_prod_frac * bg_idx.size))
                if n_low > 0:
                    rng_bg = np.random.default_rng(patch_seed + 1_000_003)
                    picked = rng_bg.choice(bg_idx, size=n_low, replace=False)
                    bg_low_mask[picked] = True
            self._bg_low_mask = bg_low_mask
            self._bg_low_cap_mult = float(bg_low_cap_mult)
            self._capacity = np.where(
                patch_mask,
                base_cap * patch_capacity_mult,
                np.where(
                    bg_low_mask,
                    base_cap * self._bg_low_cap_mult,
                    base_cap * bg_cap_mult,
                ),
            )
            # 4) 再生守恒：patch_mult × patch_frac + bg_mult × bg_frac = 1
            #   ⚠️ 同上：背景归零时该守恒式**不再成立**（这是 13.5 的目的）⇒ 直接取 0.0。
            patch_frac = patch_area / total_area
            bg_frac = bg_area / total_area
            self._patch_regrowth_mult = float(patch_regrowth_mult)
            self._bg_regrowth_mult = (
                0.0 if bg_production_zero
                else float((1.0 - patch_regrowth_mult * patch_frac) / bg_frac)
            )
            self._bg_low_regrowth_mult = float(bg_low_cap_mult)
            # 5) 初始填充：patch 格填满，背景格填 background_fill
            #   ⚠️ 背景归零 ⇒ `_capacity` 为 0 ⇒ `_grid` 自动为 0（连初始存量也不给）
            #   🔴 R242：低产能背景格容量极低 ⇒ 初始存量按 background_fill 给一点点
            #     （"绿洲链"要有初始food才能立刻续命；不给等于开局饿死，机制失去意义）
            self._grid = np.where(       # sparse:init（构造期；派生集尚不存在）
                patch_mask,
                self._capacity * 1.0,
                self._capacity * background_fill,
            )
            self.bg_production_zero = bool(bg_production_zero)
        else:
            self._capacity = base_cap
            self._grid = self._capacity * initial_fill   # sparse:init（构造期）
            self._patch_mask = None
            self._patch_regrowth_mult = 1.0
            self._bg_regrowth_mult = 1.0
            self._bg_low_mask = None
            self._bg_low_cap_mult = 0.0
            self._bg_low_regrowth_mult = 0.0
            self.bg_production_zero = False
        # ---- R217 §五 #1 稀疏化 B：惰性再生（默认关 = 全场路径逐位不变）----
        self._lazy = False
        self._dirty_mask = None

    # ---- 查询（只看不吃） ---------------------------------------------------

    def capacity_at(self, flat) -> np.ndarray:
        """问：这个格子的食物上限是多少？

        通俗理解：格子的"粮仓容量"。赤道格子大粮仓大（≈40），
        极点格子小粮仓小（≈1）。容量主要看面积，不看时间。

        参数
        ----
        flat : int 或 NDArray[int64]
            平铺索引（可传一个或一组）。

        返回
        ----
        float 或 NDArray[float64] : 该格（这些格）的食物容量上限。
        """
        out = self._capacity[np.asarray(flat, dtype=np.int64)]
        return out.item(0) if out.ndim == 0 else out

    def amount_at(self, flat) -> np.ndarray:
        """问：这个格子现在还有多少食物？

        通俗理解：看粮仓里还剩多少。刚构造时是容量的一半，
        吃了会少，再生会慢慢补回来。

        参数
        ----
        flat : int 或 NDArray[int64]
            平铺索引（可传一个或一组）。

        返回
        ----
        float 或 NDArray[float64] : 当前食物存量（不会超过容量）。
        """
        out = self._grid[np.asarray(flat, dtype=np.int64)]
        return out.item(0) if out.ndim == 0 else out

    def total(self) -> float:
        """问：整个球面一共还剩多少食物？

        参数
        ----
        无。

        返回
        ----
        float : 所有格子食物存量之和（统计/日志用）。
        """
        return float(self._grid.sum())

    # ---- 消费（吃） ---------------------------------------------------------

    def consume(self, flat, amount: float) -> float:
        """吃掉一个格子的食物（最多吃到存量，不欠账）。

        通俗理解：一只生物在某格咬了一口食物。这格食物够就全吃，
        不够就只吃到剩的；存量不可能变成负数。

        参数
        ----
        flat : int
            要吃的格子（单个）。
        amount : float
            想吃的量。

        返回
        ----
        float : 实际吃到的量（不会大于存量，也不会大于想要的量）。
        """
        i = int(np.asarray(flat, dtype=np.int64))
        available = float(self._grid[i])
        taken = min(amount, available)
        self._grid[i] = available - taken            # sparse:mark（减写入 ⇒ 下面打脏）
        if self._lazy:
            # R217 稀疏化 B：能**减** `_grid` 的写入 ⇒ 必须打脏（本类 consume/consume_many
            # 之外，py 非 rd 路径上没有别的减写入）。🔴 漏标 = 破等价的唯一途径；
            # 多标无害（净格再结算一次是逐位 no-op）。只增写入（deposit / 尸体归还 /
            # patch_boost）造不出 `grid < cap` ⇒ 不必打脏。
            self._dirty_mask[i] = True
        return taken

    def consume_many(self, flats, amount: float) -> np.ndarray:
        """一批生物同时吃（各自在自己格子里吃）。

        通俗理解：把一整串生物送去同时进食。每只只在自己那格吃，
        各吃各的，一起结算。是 consume 的批量版（引擎一帧内
        全部生物一次算完，比一只一只快几十倍）。

        说明（同格多只时）：同一格的几只【均分】该格的存量——
        每格总消耗 = min(存量, 想吃的量 × 该格只数)，然后按只平分，
        所以绝不可能把格子吃到负数（欠账）。

        参数
        ----
        flats : NDArray[int64]
            一组格子平铺索引（每只生物各占一个）。
        amount : float
            每只生物想吃的量（吃相同的量）。

        返回
        ----
        NDArray[float64] : 每只生物实际吃到的量（与入参一一对应）。
        """
        flats = np.asarray(flats, dtype=np.int64)
        if self._lazy and flats.size:
            # R217 稀疏化 B：批量打脏（重复索引对 bool 赋值幂等 ⇒ 无需去重）
            self._dirty_mask[flats] = True
        # 统计每个格子同时被多少只吃（np.add.at 对重复索引累加）
        cnt = np.zeros(self._grid.size, dtype=np.int64)
        np.add.at(cnt, flats, 1)
        # 每格总消耗 = min(存量, 单只想要 × 该格只数)；按只均分
        avail = self._grid[flats]
        per_cell = np.minimum(avail, amount * cnt[flats])
        share = per_cell / np.maximum(1, cnt[flats])
        np.subtract.at(self._grid, flats, share)     # sparse:mark（减写入 ⇒ 上面打脏）
        return share

    # ---- 再生（食物慢慢长回来） -----------------------------------------------

    def deposit(self, flat, amount) -> np.ndarray:
        """把一批食物**放回**格子（按容量封顶；`consume_many` 的逆操作）。

        通俗理解：某只生物吃进去的东西没全吸收，剩下的（碎屑）掉回地上那格。
        放不进去的部分（超过粮仓容量）就丢了。

        用途
        ----
        🔴 **13.5 ③**：`OrganismConfig.assim_return_frac`（未吸收部分回流植物池，闭环）。
        供引擎在消化段调用：`actual = rf.deposit(cells, (1−assim)×digest_mass×return_frac)`。

        参数
        ----
        flat : 标量或 NDArray[int64]
            目标格（可传一个或一组）。
        amount : float 或 NDArray[float64]
            想放回的量（质量）；与 `flat` **一一对应**。

        返回
        ----
        NDArray[float64] : 逐个体**实际**放回的量（溢出被截掉；与入参一一对应）。
        """
        flat = np.asarray(flat, dtype=np.int64)
        add = np.asarray(amount, dtype=np.float64)
        if flat.size == 0:
            return np.zeros(0, dtype=np.float64)
        add = np.broadcast_to(add, flat.shape).astype(np.float64, copy=False)
        # ⚠️ 不用 `np.clip(..., out=...)`：标量入参时 `room` 是 numpy 标量（无 out 支持）
        room = np.maximum(self._capacity[flat] - self._grid[flat], 0.0)
        actual = np.minimum(np.maximum(add, 0.0), room)
        np.add.at(self._grid, flat, actual)          # sparse:inc（只增 ⇒ 无需打脏）
        return actual

    def regrow(self, tick: int) -> None:
        """让全世界的食物都长一点（一次性整场更新）。

        通俗理解：过了一个 tick，每格食物都加上一点"恢复量"，
        但上限是粮仓容量，长满了就不再长。而且——
        冷的地方（极点、深夜）恢复量会打折，温度越低长得越慢，
        热带正午长得最快。这是"随 tick 推进"的时间节律（昼夜）。

        参数
        ----
        tick : int
            当前时间步。决定此时各格温度 → 决定恢复量折扣。

        返回
        ----
        None。直接修改内部存量数组。
        """
        if self._lazy:
            # R217 §五 #1 稀疏化 B：只结算脏格（逐位等价论证见 `_regrow_lazy`）
            self._regrow_lazy(tick)
            return
        growth = self._regrowth_amount(tick)
        np.minimum(self._capacity, self._grid + growth,   # sparse:full（默认分支，逐位不变）
                   out=self._grid)

    # ---- 惰性再生（R217 §五 #1 稀疏化 B；默认关 ⇒ 全场路径逐位不变）------------

    def enable_lazy(self) -> bool:
        """启用惰性再生（只结算脏格）。**返回是否启用**；前提不满足 ⇒ False（退回全场）。

        🔴 逐位等价前提（全满足才可开；少一条都可能破等价 —— 逐条理由）：
          ① `temp_sensitivity == 1.0` ⇒ 跳过 `np.power`：pow 的**子集执行**与全场执行
             不保证同位（SIMD 通道/尾块处理差异），值域无法先验排除 ⇒ 不赌；
          ② `light_sensitivity == 0.0` ⇒ 无光照因子（其归一化分支是**全场归约**
             `lf.mean()`，子集算不出）；
          ③ `regrowth_rate / _patch_regrowth_mult / _bg_regrowth_mult ≥ 0`
             ⇒ 增长非负 ⇒ 夹取引理 `min(cap, cap+g) == cap` 成立。⚠️ **patchy 非归零档
             `_bg_regrowth_mult` 可为负**（守恒式 `(1−pm·patch_frac)/bg_frac`）——负增长会把
             净格拉低 ⇒ 惰性必须退场（本档实测默认参数下可为负，不是假想情形）；
          ④ `distribution == "patchy"` 且掩码在：本机制的目标档（uniform 档全体初始半满
             ⇒ 脏格≈全场，收益≈0，且首段反而多一次 `flatnonzero`）。
        """
        if self.distribution != "patchy" or self._patch_mask is None:
            return False
        if self.temp_sensitivity != 1.0 or self.light_sensitivity != 0.0:
            return False
        if (self.regrowth_rate < 0.0 or self._patch_regrowth_mult < 0.0
                or self._bg_regrowth_mult < 0.0):
            return False
        self._lazy = True
        self.rebuild_lazy()
        return True

    def rebuild_lazy(self) -> None:
        """（重）建脏格集 = `_grid < _capacity`。

        **启用时与快照恢复后必须调用**（脏格集是派生量，不进快照；`_grid` 被整体替换
        的路径 —— 快照恢复 —— 会把既有掩码变成陈旧的"欠标" ⇒ 会破等价）。
        """
        if not self._lazy:
            return
        self._dirty_mask = self._grid < self._capacity

    def _regrow_lazy(self, tick: int) -> None:
        """惰性结算：只对脏格推进**一 tick** 的再生。

        逐位等价论证（三条）：
          ① **窗口恒 1 tick**：每 tick 都结算全部脏格 ⇒ 无多 tick 合并、无欠账
             ⇒ 与全场路径**逐步同式**（不是"数学上收敛到同一值"）；
          ② **净格不动**：`grid == cap` 且 `g ≥ 0`（见 `enable_lazy` 前提③）
             ⇒ `min(cap, cap+g) == cap`（IEEE 下 `cap+g ≥ cap` 恒成立）⇒ 跳过 = 逐位 no-op；
          ③ **子集执行 == 全场执行**：`min(cap_i, grid_i + g_i)` 全是逐元素 IEEE 定值算子，
             与数组长度无关（已排除 pow / 光照归约 / 负增长三类风险）。
        """
        idx = np.flatnonzero(self._dirty_mask)      # 0.28 ms @460800（掩码扫描，O(n_cells)）
        if idx.size == 0:
            return
        growth = self._regrowth_amount(tick, idx)   # 子集链，O(|脏格|)
        cap = self._capacity[idx]
        new = self._grid[idx] + growth
        np.minimum(cap, new, out=new)
        self._grid[idx] = new                        # sparse:lazy（子集结算；脏标记同函数内维护）
        # 到顶格转净（未到顶的留脏，下 tick 继续；`>=` 而非 `==`：min 后不可能 > cap）
        self._dirty_mask[idx[new >= cap]] = False

    def _regrowth_amount(self, tick: int, idx=None) -> NDArray[np.float64]:
        """计算每格本 tick 应恢复的食物量（内部函数）。

        规则：恢复量 = 基准恢复率 × **温度因子** × **光照因子**（后者可选）。
        温度因子 = clip((温度+20°C)/20, 0, 1)^temp_sensitivity：
          - 温度 ≥ 0°：因子 1，恢复满速；
          - 温度 0°~-20°：逐渐变小，越冷恢复越慢；
          - 温度 ≤ -20°：因子≈0，几乎不恢复（极地冰封）。
        光照因子 = clip(光照, 0, 1)^light_sensitivity：
          - light_sensitivity = 0（默认）⇒ 因子恒为 1 ⇒ **与旧版逐位一致**；
          - >0 ⇒ "日照越长、太阳越高 ⇒ 初级生产力越强"（光合作用有效辐射）。
            配合季节机制 ⇒ 夏季半球再生快、冬季半球再生慢，
            食物丰度带随季节南北移动 ⇒ 迁徙的**驱动源**。

        参数
        ----
        tick : int
            当前时间步（用于查温度、光照）。
        idx : NDArray[int64] | None, 默认 None（R217 §五 #1 稀疏化 B）
            `None` ⇒ 全场（**旧行为，逐位不变**）；给子集 ⇒ 只算这些格（惰性再生用）。
            子集版与全场版**逐元素同式**（同算子、同顺序、无归约）⇒ 前提成立时逐位一致。

        返回
        ----
        NDArray[float64] : 形状与请求的格集一致（`idx=None` ⇒ (n_cells,)）。
        """
        if idx is None:
            cells = np.arange(self.world.n_cells)
            pm = self._patch_mask        # 原样引用（不拷贝）⇒ 全场路径与旧版逐位/逐字相同
        else:
            cells = np.asarray(idx, dtype=np.int64)
            pm = self._patch_mask[cells] if self._patch_mask is not None else None
        temps = self.lt.temperature(cells, tick)
        # 因子 = (温度+20)/20，clip 到 [0,1]：
        #   ≥0° → 1（满速）；-20° → 0（停摆）；中间线性过渡
        factor = np.clip((temps + 20.0) / 20.0, 0.0, 1.0)
        factor = np.power(factor, self.temp_sensitivity)
        growth = self.regrowth_rate * factor
        # ---- 🔴 R196 光驱动再生（light_sensitivity=0 时下面整块恒等跳过）----
        if self.light_sensitivity > 0.0:
            ill = self.lt.illumination(cells, tick)
            lf = np.power(np.clip(ill, 0.0, 1.0), self.light_sensitivity)
            if self.light_normalize:
                # 按全球均值归一化 ⇒ 只改空间分布，不改全球总量
                m = float(lf.mean())
                if m > 1e-9:
                    lf = lf / m
            growth = growth * lf
        # patchy 守恒：斑块格 × patch_mult，背景格 × bg_mult，周期总再生量不变
        if self.distribution == "patchy" and pm is not None:
            # ---- 🔴 R242 背景低产能带：三类格各取自己的再生倍率 ----
            #   原版是两路 `np.where(pm, patch_mult, bg_mult)`；本档把背景细分为
            #   「低产能背景格（`_bg_low_mask`）」与「零产能背景格」两路。
            #   ✅ 默认档 `_bg_low_mask` 全 False ⇒ 三路退化为两路，且
            #      `_bg_low_regrowth_mult` 不参与 ⇒ 与原路径**逐位相同**（C7 不动）。
            if self._bg_low_mask is not None and self._bg_low_mask.any():
                low = (self._bg_low_mask[cells] if idx is not None
                       else self._bg_low_mask)
                growth = np.where(
                    pm,
                    growth * self._patch_regrowth_mult,
                    np.where(
                        low,
                        growth * self._bg_low_regrowth_mult,
                        growth * self._bg_regrowth_mult,
                    ),
                )
            else:
                growth = np.where(
                    pm,
                    growth * self._patch_regrowth_mult,
                    growth * self._bg_regrowth_mult,
                )
        return growth
        return growth

    # ---- 快照 / 调试 ---------------------------------------------------------

    def snapshot(self) -> NDArray[np.float64]:
        """拷贝一份当前食物存量的二维网格图（行×列）。

        通俗理解：给现在的食物分布"拍张照"传给可视化，
        拍到的是一张 60×120 的图，一格一个值。

        参数
        ----
        无。

        返回
        ----
        NDArray[float64] : 形状 (rows, cols) 的拷贝，改它不影响场内数据。
        """
        return self._grid.reshape(self.world.rows, self.world.cols).copy()

    def __repr__(self) -> str:
        """打印这场的简要信息。

        返回
        ----
        str : 例如 "ResourceField(cells=7200, total=xx/xx)"。
        """
        cap = float(self._capacity.sum())
        return f"ResourceField(cells={self.world.n_cells}, total={self.total():.1f}/{cap:.1f})"