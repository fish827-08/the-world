"""biosphere.world_adapter：世界字段只读适配 + MVP 合成小世界（W0-B）。

通俗理解
--------
生物圈**不直接碰** ``world/`` 的任何东西：所有世界字段（海陆、海拔、水深、
斑块、季节相位）都经本模块的 :class:`WorldView` 读入。这就是 B 线对
"跨线接口"的唯一出口——将来接真实世界时改这里（且须先报变更单），
simulator/resolve/metrics 一概不动。

两条路径：

1. **合成小世界**（MVP 现用）：:func:`synthetic_world` 造一块小型 2D 网格
   （陆海混合 + 若干食物斑块 + 季节相位），资源场由 :class:`SyntheticResource`
   承载（世界侧资源的本地替身）。
2. **真实世界**（W1+ 预留）：:func:`adapt_real_world` 以鸭子类型只读提取
   ``EnvField.surface_type / h_m / sea_level_m``、``ResourceField`` 斑块掩码、
   ``LightAndTemperature`` 季节参数；缺字段 ⇒ **NotImplementedError**（fail-loud，
   不静默退化）。

B18 起另有两个**只读/单向**接口：:func:`sample_climate` 读 A3
``AtmoClimateSource`` 的 t2m/precip 出口；:func:`sync_real_capacity` 把世界
``ResourceField._capacity`` 同步进生物资源（**只同步容量，不同步存量**）。

读取纪律：本模块对传入的世界对象**只读**（除合成资源场由 biosphere 自理）。
容量同步只写生物侧 ``SyntheticResource.capacity``，不写回世界。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

# ── surface_type 编码（与 world/env_field.py 的 C0b 编码一致；只读引用） ──

SURFACE_LAND = 0
SURFACE_OCEAN = 1
SURFACE_SHELF = 2
SURFACE_RIVER = 3
SURFACE_LAKE_FRESH = 4
SURFACE_LAKE_SALT = 5

#: 水生类别集合（陆 = 其余）
WATER_SURFACES = frozenset({
    SURFACE_OCEAN, SURFACE_SHELF, SURFACE_RIVER,
    SURFACE_LAKE_FRESH, SURFACE_LAKE_SALT,
})


@dataclass(frozen=True)
class WorldView:
    """世界结构只读视图（biosphere 侧唯一的世界接口）。

    Attributes:
        rows / cols / n_cells: 网格尺寸（等距圆柱投影的扁担）。
        surface_type: uint8 (n,)，编码同上（合成世界只用 LAND/OCEAN）。
        elevation_m: float64 (n,)，海拔（陆地 ≥ 0，海洋 < 0）。
        depth_m: float64 (n,)，水深（海洋 > 0，陆地 = 0）。
        soil_water: float64 (n,)，土壤水（陆生约束用；水生格无意义置 0）。
        soil_water_min: 陆生细胞的最低土壤水阈值（世界侧参数——
            B11 补充指令：栖息阈值从 world_adapter 读，不写死在生物圈）。
            合成世界默认 2.0；真实世界由 A 线降水/土壤水耦合定（接口见
            ``adapt_real_world`` 的 ``soil_water_min`` 参数）。
        patch_id: int32 (n,)，食物斑块编号（-1 = 斑块外）。
        neighbors: tuple of arrays，每格的 von Neumann 邻格（列向环绕、
            行向截断；极点截断是 MVP 合成世界的简化，接真实世界时换成
            ``SphereWorld.neighbors``）。
        tilt_rad / season_period: 季节参数（0 = 无季节）。
    """

    rows: int
    cols: int
    n_cells: int
    surface_type: np.ndarray
    elevation_m: np.ndarray
    depth_m: np.ndarray
    patch_id: np.ndarray
    neighbors: Tuple[np.ndarray, ...]
    soil_water: np.ndarray = None  # type: ignore[assignment]
    soil_water_min: float = 2.0
    tilt_rad: float = 0.0
    season_period: int = 0

    def __post_init__(self) -> None:
        if self.soil_water is None:
            object.__setattr__(
                self, "soil_water",
                np.zeros(self.n_cells, dtype=np.float64))

    def is_water(self, flat: np.ndarray) -> np.ndarray:
        """布尔掩码：水生格（逐格子集查询）。"""
        st = self.surface_type
        return np.isin(st, np.array(sorted(WATER_SURFACES), dtype=st.dtype))

    def water_mask(self) -> np.ndarray:
        """全场水生布尔掩码（B7 冒烟/统计用）。"""
        return self.is_water(np.arange(self.n_cells, dtype=np.int64))

    def season_phase(self, tick: int) -> float:
        """季节相位 ∈ [0,1)（无季节恒 0）。"""
        if self.season_period <= 1 or abs(self.tilt_rad) <= 1e-12:
            return 0.0
        return (tick % self.season_period) / float(self.season_period)

    def rc(self, flat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """flat → (row, col)。"""
        flat = np.asarray(flat, dtype=np.int64)
        return flat // self.cols, flat % self.cols

    def grid_distance(self, a: int, b: int) -> float:
        """网格曼哈顿距（列向环绕、行向不绕；B6 逃逸项用）。"""
        ra, ca = int(a) // self.cols, int(a) % self.cols
        rb, cb = int(b) // self.cols, int(b) % self.cols
        dcol = min(abs(ca - cb), self.cols - abs(ca - cb))
        return float(abs(ra - rb) + dcol)

    def flat_of(self, row: np.ndarray, col: np.ndarray) -> np.ndarray:
        """(row, col) → flat（col 环绕、row 截断）。"""
        row = np.clip(np.asarray(row, dtype=np.int64), 0, self.rows - 1)
        col = np.asarray(col, dtype=np.int64) % self.cols
        return row * self.cols + col


@dataclass(frozen=True)
class SyntheticResource:
    """MVP 合成资源场（世界侧 ResourceField 的本地替身）。

    接真实世界后，资源读写改走 ``ResourceField.consume/deposit``（跨线，
    须变更单）；MVP 期内由 biosphere 自理，保持 simulator 与世界解耦。
    """

    amount: np.ndarray      # float64 (n,)，当前资源量
    capacity: np.ndarray    # float64 (n,)，容量上限
    regrow_rate: float      # 每 tick 回复比例（向 capacity 指数恢复）


# ── 合成小世界 ────────────────────────────────────────────────────────────


def _von_neumann_table(rows: int, cols: int) -> Tuple[np.ndarray, ...]:
    """每格 4 邻域（列环绕、行截断）；顺序固定 (上, 下, 左, 右)。"""
    table: List[np.ndarray] = []
    for r in range(rows):
        for c in range(cols):
            up = (max(r - 1, 0)) * cols + c
            down = (min(r + 1, rows - 1)) * cols + c
            left = r * cols + (c - 1) % cols
            right = r * cols + (c + 1) % cols
            table.append(np.array([up, down, left, right], dtype=np.int64))
    return tuple(table)


def _synthetic_soil_water(n: int, patch_id: np.ndarray, land_mask: np.ndarray,
                          rng: np.random.Generator) -> np.ndarray:
    """合成土壤水：斑块格 4–10、背景陆格 0–3.5、水格 0（阈值默认 2.0 ⇒
    背景陆格偏干、斑块格可居——陆生约束有空间结构）。"""
    sw = np.zeros(n, dtype=np.float64)
    land = np.flatnonzero(land_mask)
    on_patch = land[patch_id[land] >= 0]
    off_patch = land[patch_id[land] == -1]
    sw[on_patch] = 4.0 + 6.0 * rng.random(on_patch.size)
    sw[off_patch] = 3.5 * rng.random(off_patch.size)
    return sw


def synthetic_world(
    rows: int = 32,
    cols: int = 64,
    seed: int = 0,
    land_frac: float = 0.45,
    patch_count: int = 10,
    patch_radius: int = 2,
    patch_capacity: float = 12.0,
    bg_capacity: float = 1.0,
    regrow_rate: float = 0.08,
    tilt_rad: float = 0.20,
    season_period: int = 400,
    soil_water_min: float = 2.0,
) -> Tuple[WorldView, SyntheticResource]:
    """造一块 MVP 合成小世界：陆海混合 + 圆形食物斑块 + 季节。

    结构（surface_type/elevation/depth/patch_id）用独立 rng 生成，
    **不消费模拟器主 rng**（与 sphere_engine 的 patch_seed 同思路）。
    """
    if rows < 4 or cols < 4:
        raise ValueError("合成世界至少 4×4")
    rng = np.random.default_rng(seed)
    n = rows * cols

    # 陆/海：值噪声阈值化（粗粒度，保证连片）
    coarse = max(2, rows // 8)
    lattice = rng.random((rows // coarse + 2, cols // coarse + 2))
    field = np.repeat(np.repeat(lattice, coarse, axis=0), coarse, axis=1)
    field = field[:rows, :cols]
    ocean = field > (1.0 - land_frac)

    surface_type = np.where(ocean, SURFACE_OCEAN, SURFACE_LAND).astype(np.uint8)
    surface_flat = surface_type.reshape(-1)
    rand2d = rng.random((rows, cols))
    elev = np.where(ocean, -1.0 - 3.0 * rand2d, 0.5 + 4.0 * rand2d)
    elev_flat = elev.reshape(-1).astype(np.float64)
    depth_flat = np.where(ocean, -elev, 0.0).reshape(-1).astype(np.float64)

    # 斑块：只落在陆地（MVP 食物在陆地；水生个体吃背景资源）
    patch_id = np.full(n, -1, dtype=np.int32)
    capacity = np.full(n, bg_capacity, dtype=np.float64)
    land_cells = np.flatnonzero(~ocean.reshape(-1))
    placed = 0
    for pid in range(patch_count):
        if land_cells.size == 0:
            break
        center = int(land_cells[int(rng.integers(0, land_cells.size))])
        cr, cc = center // cols, center % cols
        cells = [
            (cr + dr) * cols + (cc + dc) % cols
            for dr in range(-patch_radius, patch_radius + 1)
            for dc in range(-patch_radius, patch_radius + 1)
            if 0 <= cr + dr < rows
        ]
        for f in cells:
            if patch_id[f] == -1 and surface_flat[f] == SURFACE_LAND:
                patch_id[f] = pid
                capacity[f] = patch_capacity
                placed += 1
    if placed == 0:
        raise RuntimeError("斑块放置失败（陆地不足？）")

    view = WorldView(
        rows=rows, cols=cols, n_cells=n,
        surface_type=surface_flat,
        elevation_m=elev_flat,
        depth_m=depth_flat,
        patch_id=patch_id,
        neighbors=_von_neumann_table(rows, cols),
        soil_water=_synthetic_soil_water(n, patch_id, land_mask=~ocean.reshape(-1),
                                         rng=np.random.default_rng(seed + 77)),
        soil_water_min=soil_water_min,
        tilt_rad=tilt_rad, season_period=season_period,
    )
    resource = SyntheticResource(
        amount=capacity.copy(),
        capacity=capacity,
        regrow_rate=regrow_rate,
    )
    return view, resource


# ── 真实世界只读适配（W1+ 预留；当前不接） ────────────────────────────────


def adapt_real_world(
    world,
    env_field=None,
    resource_field=None,
    light=None,
    soil_water_min: float = 2.0,
) -> WorldView:
    """从真实世界对象只读提取 :class:`WorldView`（鸭子类型，fail-loud）。

    Args:
        world: ``SphereWorld``（提供 rows/cols/n_cells/neighbors）。
        env_field: ``EnvField``（surface_type / h_m / sea_level_m / w）。
        resource_field: ``ResourceField``（斑块掩码；可选）。
        light: ``LightAndTemperature``（tilt_rad / season_period；可选）。
        soil_water_min: 陆生约束的土壤水阈值（**A 线耦合点**：真实降水/
            土壤水量纲由世界侧定，biosphere 只读不写死）。

    Raises:
        NotImplementedError: 任一必需字段缺失（不静默退化）。
    """
    rows = int(getattr(world, "rows"))
    cols = int(getattr(world, "cols"))
    n = int(getattr(world, "n_cells"))

    if env_field is None:
        raise NotImplementedError(
            "adapt_real_world 需要 EnvField（surface_type/h_m/sea_level_m）；"
            "MVP 期请用 synthetic_world。")
    surface_type = getattr(env_field, "surface_type", None)
    h_m = getattr(env_field, "h_m", None)
    sea_level = getattr(env_field, "sea_level_m", None)
    if surface_type is None or h_m is None or sea_level is None:
        raise NotImplementedError(
            "EnvField 缺 surface_type/h_m/sea_level_m（C0b 未开？）")
    surface_type = np.asarray(surface_type, dtype=np.uint8)
    h_m = np.asarray(h_m, dtype=np.float64)
    sea_level = float(sea_level)
    depth_m = np.maximum(sea_level - h_m, 0.0)

    # 斑块编号：ResourceField 的 _patch_mask + BFS（只读计算）
    patch_id = np.full(n, -1, dtype=np.int32)
    if resource_field is not None:
        mask = getattr(resource_field, "_patch_mask", None)
        if mask is not None:
            mask = np.asarray(mask, dtype=bool)
            neighbors = tuple(np.asarray(nb) for nb in
                             (world.neighbors(int(f)) for f in range(n)))
            seen = np.zeros(n, dtype=bool)
            pid = 0
            for f in np.flatnonzero(mask):
                if seen[f]:
                    continue
                stack = [int(f)]
                seen[f] = True
                while stack:
                    cur = stack.pop()
                    patch_id[cur] = pid
                    for nb in neighbors[cur]:
                        if mask[nb] and not seen[nb]:
                            seen[nb] = True
                            stack.append(int(nb))
                pid += 1

    tilt = 0.0
    period = 0
    if light is not None:
        tilt = float(getattr(light, "tilt_rad", 0.0))
        period = int(getattr(light, "season_period", 0))

    # 土壤水（陆生约束用）：读 EnvField.w（C0b/env 开档才有）；
    # 缺失 ⇒ fail-loud（不静默造 0——那会让陆生约束恒假）。
    w = getattr(env_field, "w", None)
    if w is None:
        raise NotImplementedError(
            "adapt_real_world 需要 EnvField.w（土壤水）供陆生约束；"
            "env 未开 ⇒ 拿不到。阈值由调用方经 soil_water_min 传入"
            "（A 线降水/土壤水耦合点）。")
    soil_water = np.asarray(w, dtype=np.float64)

    neighbors = tuple(np.asarray(world.neighbors(int(f)), dtype=np.int64)
                      for f in range(n))
    return WorldView(
        rows=rows, cols=cols, n_cells=n,
        surface_type=surface_type,
        elevation_m=h_m,
        depth_m=depth_m,
        patch_id=patch_id,
        neighbors=neighbors,
        soil_water=soil_water,
        soil_water_min=float(soil_water_min),
        tilt_rad=tilt, season_period=period,
    )


def adapt_real_resource(resource_field) -> SyntheticResource:
    """从真实 ``ResourceField`` 只读快照资源场（B17.3 集成冒烟）。

    **快照语义**：复制 ``_grid/_capacity`` ⇒ biosphere 的消耗/回复不会写回
    世界侧资源；资源单位也尚未与生物能量标定（A3 耦合点）。因此该路径只
    证明"生物能在真世界字段上连通运行"，不用于生态强弱判读。
    """
    grid = getattr(resource_field, "_grid", None)
    capacity = getattr(resource_field, "_capacity", None)
    rate = getattr(resource_field, "regrowth_rate", None)
    if grid is None or capacity is None or rate is None:
        raise NotImplementedError(
            "adapt_real_resource 需要 ResourceField 的 _grid/_capacity/"
            "regrowth_rate；缺字段 ⇒ 不静默退化。")
    grid = np.asarray(grid, dtype=np.float64)
    capacity = np.asarray(capacity, dtype=np.float64)
    if grid.shape != capacity.shape:
        raise ValueError(
            f"资源场形状不一致: amount{grid.shape} vs capacity{capacity.shape}")
    return SyntheticResource(
        amount=grid.copy(), capacity=capacity.copy(),
        regrow_rate=float(rate))


def sync_real_capacity(resource: SyntheticResource, resource_field) -> int:
    """把世界 ``ResourceField._capacity`` 同步进生物侧容量（B18）。

    **只同步容量，不同步存量**（与世界侧 ``set_env_factors`` 同语义：
    突然改存量等于凭空增删食物）。返回同步的格数（= 世界总格数）。
    """
    cap = getattr(resource_field, "_capacity", None)
    if cap is None:
        raise NotImplementedError(
            "sync_real_capacity 需要 ResourceField._capacity；缺字段 ⇒ 不静默退化。")
    cap = np.asarray(cap, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(cap)) or float(cap.min()) < 0.0:
        raise ValueError("世界资源容量含非有限/负值，拒绝同步")
    target = resource.capacity
    if not isinstance(target, np.ndarray):
        raise TypeError("SyntheticResource.capacity 必须是 ndarray（原地同步）")
    if target.shape != cap.shape:
        raise ValueError(
            f"容量形状不一致: biosphere{target.shape} vs world{cap.shape}")
    target[...] = cap
    return int(cap.size)


@dataclass(frozen=True)
class ClimateSample:
    """一个交换相位的 A3 出口读数（**陆格**统计；天气源为 synthetic）。"""

    tick: int
    n_land: int
    t2m_c_min: float
    t2m_c_mean: float
    t2m_c_max: float
    precip_mm_yr_min: float
    precip_mm_yr_mean: float
    precip_mm_yr_max: float


def sample_climate(source, land_mask, tick: int) -> ClimateSample:
    """读 ``AtmoClimateSource`` 的 t2m/precip 出口并做陆格统计（B18）。

    只读：不推进天气、不改源状态。``land_mask`` 为世界格序布尔掩码
    （通常取 ``EnvField.land_mask``，与 A18/A20 容量标定同一口径）。
    """
    land = np.asarray(land_mask, dtype=bool).reshape(-1)
    if land.size == 0 or not bool(land.any()):
        raise ValueError("sample_climate 需要非空陆地掩码")
    flat = np.flatnonzero(land)
    t2m = np.asarray(
        source.temperature_c(flat, int(tick)), dtype=np.float64).reshape(-1)
    if t2m.size != flat.size:
        raise ValueError(
            f"temperature_c 返回长度 {t2m.size} ≠ 陆格数 {flat.size}")
    precip = source.precip_mm_yr()
    if precip is None:
        raise RuntimeError(
            "source.precip_mm_yr() 返回 None —— 天气源未 bind / 无降水场")
    precip = np.asarray(precip, dtype=np.float64).reshape(-1)
    if precip.size != land.size:
        raise ValueError(
            f"precip_mm_yr 长度 {precip.size} ≠ 世界格数 {land.size}")
    pr_land = precip[land]
    if not np.all(np.isfinite(t2m)):
        raise ValueError("A3 t2m 含 NaN/Inf，拒绝入读数")
    if not np.all(np.isfinite(pr_land)):
        raise ValueError("A3 precip 含 NaN/Inf，拒绝入读数")
    return ClimateSample(
        tick=int(tick), n_land=int(flat.size),
        t2m_c_min=float(t2m.min()), t2m_c_mean=float(t2m.mean()),
        t2m_c_max=float(t2m.max()),
        precip_mm_yr_min=float(pr_land.min()),
        precip_mm_yr_mean=float(pr_land.mean()),
        precip_mm_yr_max=float(pr_land.max()),
    )


__all__ = [
    "WorldView", "SyntheticResource", "synthetic_world", "adapt_real_world",
    "adapt_real_resource", "sync_real_capacity", "ClimateSample",
    "sample_climate",
    "SURFACE_LAND", "SURFACE_OCEAN", "SURFACE_SHELF", "SURFACE_RIVER",
    "SURFACE_LAKE_FRESH", "SURFACE_LAKE_SALT", "WATER_SURFACES",
]
