"""ENV 场：地形 → 气候态水分 → 格子产能因子（「纯斑块世界」底层规则线）。

口径来源
--------
* `docs/设计文档/设计-纯斑块世界-底层规则自组织-20261011.md`（§十五 四臂协议）；
* `docs/设计文档/设计-ENV-FIELD-物理场与格子产能-20261008.md`（场定义 H/W/P、归一化
  条款、守卫与算路口径）。

流水线（全部**构造期**算一次 ⇒ 每 tick 零成本；不进快照 —— 从 config 确定性重建）
----------------------------------------------------------------------------------
    H  ← 3D 值噪声 fBm（球面采样；独立 rng 流，**不消费引擎 RNG**）
        └ 🔴 `cfg.c0b_enabled=True`（默认关 ⇒ 整块跳过 = 逐位等价）：H 米制化
          `h_m = h^coast_gamma × peak_altitude_m`（§5.2：海拔单位化是 C0b 前置）
          并按**目标陆比取面积权分位**定海平面 `sea_level_m` ⇒ 海洋格集合 = 天然出水口
    T  = base_temperature(φ) − lapse_c×H            （气候态温度，°C）
        └ 🔴 C0b 开档 ⇒ 换成真实递减率 `T = base − 6.5 ℃/km × h_m`（Γ 见 §2.1-⑥）
        └ `cfg.uniform_base_t=True`（去混杂选项，默认关）⇒ base_temperature(φ) 换成其
          **面积权全球均值**常数（只关纬度梯度、不动全球均值；治 Magnus 纬度压印混杂）
        └ `cfg.insolation_profile`（自然剖面；**默认开 = 正式世界默认档**，裁定一 2026-10-14）
          ⇒ base_temperature(φ) 换成 **EBM 年平气温剖面**（辐照数值积分；常数全为地球来源
          ⇒ 纬度影响与自然界一致）；显式 False = classic 档（C7 回滚点，逐位旧公式）
          （两选项互斥，fail-loud）
    q  = 6.11×exp(17.27T/(237.3+T))                 （Magnus 饱和水汽压，hPa）
    Pr = q×(1 + orographic×H)                       （抬升增雨）
        └ 🔴 `cfg.c1_enabled=True`（默认关 ⇒ 整块跳过 = 逐位等价，§5.3）：
          `Pr ×= band(φ)`，`band = 1 + A_wet·G((φ−φ_itcz)/σ_wet)
          − A_dry·[G((φ−φ_dry)/σ_dry) + G((φ+φ_dry)/σ_dry)]`（G(x)=exp(−x²)），
          并按**面积权均值 = 1** 归一（同 f_g/f_c 纪律）⇒ 只修降水的纬度形状，不碰 Et
    Et = et_frac×q                                  （蒸散）
    A  = D8 汇流累积 max(Pr−Et, 0)                    （priority-flood 路由树；
        默认档极点行 = 出水口；🔴 C0b 开档 ⇒ 种子 = 海洋格 ∪ 极点行，见下方「C0b 近似」）
    W  = max(Pr−Et, 0) × (1 + runoff_gain×A/(A+A_ref))   （汇流带更湿；A_ref = 99 分位）
    f_g = u^ws / 面积权均值            （再生因子；u = W/W_ref，W_ref = W 的面积权均值）
    f_c = clip(u,0,k_max)^cs / 面积权均值（容量因子）
    🔴 C0b 开档追加产出（纯读数层，不改上面任何一条公式）：
        ocean/land/shelf/lake(淡水/盐湖)/river ⇒ `surface_type`（uint8 编码）+ 三条守恒不变量

🔴 C0b 的结构性近似（实现层声明，不许扩读；§5.2 + §2.3-②）
--------------------------------------------------------------
* **极点行必须与海洋格一起作 priority-flood 种子**：`_nb_table` 里极点行除规范槽
  （col 0）外全是 `-1` 占位，且相邻纬度带的邻居经 `rc_to_flat` 坍缩**只指回规范槽**
  ⇒ 非规范极槽在邻居图上**孤立**，不作种子就永远打不开 ⇒ 触发既有 fail-loud
  （"仅覆盖 M/N 格"）。与旧口径的区别 = 极点不再决定全球填洼面（旧档 50.8% 被抬成
  "极海"），只在**极地点自身高程**处充当局部出边 ⇒ 不达海的洼地由此成为内流盆地。
* **湖的淡/咸二分在本模型下退化**：填洼面上每格的接收者链严格下降 ⇒ 链终点只可能是
  "海洋格"或"非海洋极点根"；湖必有溢流口（priority-flood 保证），故湖自身永不是终点。
  照 §5.2 字面规则实现：溢流链达海 ⇒ 过水淡水湖，不达海（终止于内流盆地）⇒ 盐湖。
* **嵌套洼地按"单层溢流面"近似**（§2.3-②：depression hierarchy 留待需要时）：
  同一溢流口（链上第一个非洼地格）的洼地格合并记为一个湖。

🔴 两条写死的口径（防"数字没出处"）
------------------------------------
* **水只进 W，W 只进因子**：全链路**没有**任何 `if 河道: 产能×k` 分支 ——
  "河 = 肥沃"不是硬编码，是 f 的结果；`river_mask` 只是**读数命名层**
  （不进产能输入、不作判据前置；砚 69d6e0c §三-3 五道防退化闸的引擎侧落点）。
* **归一化条款**（ENV-FIELD §二-5 硬要求）：两个因子都按**面积权均值 = 1** 归一
  ⇒ Σ容量**逐位守恒**（f_c 乘在容量上、面积权均值恰为 1）、Σ名义再生零阶（面积权）
  守恒 ⇒ K 尺度不被连续场整体抬高/压低（以免抹掉 C4 的"稀缺"前提）。

已明写的近似（实现层声明，不许扩读）
------------------------------------
* 拓扑用 `world._nb_table`（真 8 邻居，极点行 `-1` 占位）—— 比 ENV-FIELD §三-2 的
  "(ii) 粗网格五点"更严格；**极点行**在平滑里跳过、在汇流里当**出水口**（水离场）；
* 填洼 = **priority-flood**（Barnes 2014 的堆式单遍变体，`filled[j] = max(H[j], filled[开堆者]+eps)`，
  eps = 1e-6）。旧"抬升至 min 邻居+eps"的向量化近似需要 O(盆地深度) 轮迭代
  （60×120 冒烟实测 200 轮不收敛 ⇒ 换掉，不许静默留余洼）。
  不可达格（邻居图不连通）⇒ 直接炸（fail-loud），不静默把余洼当汇。
* 路由（接收者）= **`filled` 面的 D8 最陡下降**（Barnes 2014 标准用法）：非洼格
  `filled == H` ⇒ 即地形最陡下降；洼内沿 eps 梯度 ⇒ 流向溢流口。
  无环：每格的开堆者邻元 `filled` 严格更小 ⇒ 最陡下降链严格下降。
* **填洼的实测语义**（`[实测]` s2/seed 42，2026-10-11 诊断）：闭环球面 + 极点=唯一出水口
  ⇒ 低于**极环高度**（两规范出水槽均为 0.4577）的低地连成整体"极海"，`filled` 近似常值面
  （H<0.4577 内拟合抬升面 `filled−H ≈ 0.4535 − 0.98×H`，R²=0.985；被抬升格 50.8%，
  连通块 98.7% 单块 ⇒ 海平面型而非局部碗型）。A 的语义 = **流向两极的汇流树**。
  这是设计（极点=出水口）x 未侵蚀噪声地形的**结构性后果**，不是数值缺陷；
  若将来要"少海/无海"世界，动地形或极环高度即可，**不动路由链**。
"""
from __future__ import annotations

import heapq

import numpy as np
from numpy.typing import NDArray

from world.sphere_world import SphereWorld
from world.light_and_temperature import LightAndTemperature

#: Magnus 公式常数（Tetens 系数；T 为 °C，输出 hPa）
_MAGNUS_A = 17.27
_MAGNUS_B = 237.3
_MAGNUS_C = 6.11
#: 温度下钳（防 |T| 极端时除零；本项目 t_equator/t_pole 默认远在安全域内）
_T_FLOOR = -60.0
#: fBm 独立流常数（与 salt 组合出流名；防同 seed 不同机制撞流）
_STREAM_NOISE = 0xE17F
_STREAM_SHUFFLE = 0x5A17
#: priority-flood 抬升步长（保 filled 沿接收者链严格下降 ⇒ 路由树无环）
_FLOOD_EPS = 1e-6
#: C0b 真实环境直减率 Γ = 6.5 ℃/km（§2.1-⑥：干 9.8 / 湿 6.0 / 环境 6.5 ℃/km；
#:   `c0b_enabled` 开档时替代现象学常数 `lapse_c`（H∈[0,1] 全程温差 20 ℃）——
#:   出处 = 报告 §5.2「把 lapse_c=20 的现象学常数换成真实海拔尺度与真实递减率」）
_LAPSE_C_PER_KM = 6.5

#: C0b `surface_type` 编码（🔴 载体层，永不进判据前置 = ENV-FIELD §四-4 红线 2；
#:   也是 §2.3-④ 热物性三件套（α/c/k）未来逐格赋值的挂载点）
SURFACE_LAND = 0
SURFACE_OCEAN = 1
SURFACE_SHELF = 2
SURFACE_RIVER = 3
SURFACE_LAKE_FRESH = 4
SURFACE_LAKE_SALT = 5
#: 溢流链终点类别（§5.2「河终点全为海/湖/内流盆地」）
OUTLET_SEA = 0
OUTLET_INLAND = 1

#: 一层 EBM 自然剖面常数（全为地球观测值；`insolation_profile` 选项用，零自由参数）
_SOLAR_CONSTANT_WM2 = 1361.0        # 太阳常数 S0（W/m²）
_OBLIQUITY_RAD = np.radians(23.44)  # 黄赤交角 ε
_PLANETARY_ALBEDO = 0.29            # 行星反照率 α
_SIGMA_SB = 5.670374419e-8          # 斯特藩–玻尔兹曼常数 σ（W·m⁻²·K⁻⁴）
_GREENHOUSE_K = 33.0                # 温室增量 ΔT_gh（K，现代地球值）
_DAYS_PER_YEAR = 365                # 年采样点数（等距覆盖整年）


def _insolation_base_temperature(lat_rad: NDArray) -> NDArray:
    """EBM 年平气温剖面（°C）：年平辐照数值积分 → 有效辐射温度 + 温室增量。

    公式（圆轨道忽略偏心率；标准日均辐照）：
        δ(d) = ε·sin(2πd/T)                    太阳赤纬（年内采样）
        h0   = arccos(clip(−tanφ·tanδ, −1, 1)) 日落时角（含极昼/极夜）
        H(d) = (S0/π)[h0·sinφ·sinδ + cosφ·cosδ·sin h0]   日均辐照（W/m²）
        S̄    = mean_d H(d)                     年平辐照
        T_s  = ((1−α)·S̄/σ)^(1/4) + ΔT_gh − 273.15        °C

    期望锚点（真实地球）：S̄(赤道)≈415 W/m²、S̄(极点)≈177 W/m²；
    T_s ≈ +28.6°C / −23.0°C。输入宜为**每行纬度**（≤ rows；本档调用即如此）；
    逐日循环累积 ⇒ 对任意长度输入无 (n×365) 中间量。
    """
    phi = np.clip(np.asarray(lat_rad, dtype=np.float64),
                  -(np.pi / 2.0 - 1e-6), np.pi / 2.0 - 1e-6)
    tan_phi = np.tan(phi)
    s_phi = np.sin(phi)
    c_phi = np.cos(phi)
    days = np.linspace(0.0, 2.0 * np.pi, _DAYS_PER_YEAR, endpoint=False)
    s_mean = np.zeros(phi.shape, dtype=np.float64)
    for t in days:
        delta = _OBLIQUITY_RAD * np.sin(t)
        tan_delta = np.tan(delta)
        cos_h0 = np.clip(-tan_phi * tan_delta, -1.0, 1.0)
        h0 = np.arccos(cos_h0)
        s_mean += h0 * s_phi * np.sin(delta) + c_phi * np.cos(delta) * np.sin(h0)
    s_mean *= _SOLAR_CONSTANT_WM2 / np.pi / _DAYS_PER_YEAR
    t_eff = ((1.0 - _PLANETARY_ALBEDO) * s_mean / _SIGMA_SB) ** 0.25
    return t_eff + _GREENHOUSE_K - 273.15



def _value_noise_3d(rng, p: NDArray, lattice: int, freq: float) -> NDArray:
    """单八度 3D 值噪声：p (3,n) ∈ [-1,1]³ → (n,)，格点值 standard normal。

    三线性插值 + smoothstep 权重；格点索引按 `lattice` 取模（周期格点 ⇒ 无接缝）。
    """
    grid = rng.standard_normal((lattice, lattice, lattice))
    x = (p[0] + 1.0) * 0.5 * (lattice * freq)
    y = (p[1] + 1.0) * 0.5 * (lattice * freq)
    z = (p[2] + 1.0) * 0.5 * (lattice * freq)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    z0 = np.floor(z).astype(np.int64)
    tx = x - x0
    ty = y - y0
    tz = z - z0
    tx = tx * tx * (3.0 - 2.0 * tx)
    ty = ty * ty * (3.0 - 2.0 * ty)
    tz = tz * tz * (3.0 - 2.0 * tz)
    x1 = (x0 + 1) % lattice
    y1 = (y0 + 1) % lattice
    z1 = (z0 + 1) % lattice
    x0 %= lattice
    y0 %= lattice
    z0 %= lattice
    c000 = grid[x0, y0, z0]
    c100 = grid[x1, y0, z0]
    c010 = grid[x0, y1, z0]
    c110 = grid[x1, y1, z0]
    c001 = grid[x0, y0, z1]
    c101 = grid[x1, y0, z1]
    c011 = grid[x0, y1, z1]
    c111 = grid[x1, y1, z1]
    x00 = c000 + (c100 - c000) * tx
    x10 = c010 + (c110 - c010) * tx
    x01 = c001 + (c101 - c001) * tx
    x11 = c011 + (c111 - c011) * tx
    y0v = x00 + (x10 - x00) * ty
    y1v = x01 + (x11 - x01) * ty
    return y0v + (y1v - y0v) * tz


def _area_threshold_below(values: NDArray[np.float64], areas: NDArray[np.float64],
                          frac_below: float) -> float:
    """面积权分位阈值：使「面积权 ≤ 阈值」的面积占比尽量贴近 `frac_below`（§2.3-①）。

    为何用面积权而不是格数分位：等距圆柱投影下格面积 ∝ cosφ，极区格小 ⇒
    按格数取分位会得到**错误的全球面积比**（§5.2 守恒①要的是面积比）。

    离散网格上阈必然落在某格上 ⇒ 面积比误差 ≤ 该格面积占比（实测 ≈1e-6 量级），
    故“恒等到 1e-9”不可严格成立——实现拆成两条可严格校核的不变量：
      ①a 海比 + 陆比 == 1（分划互补，1e-9）；
      ①b |实现陆比 − 目标陆比| ≤ 单格面积（实测值进 probe，验收按 ±0.5%）。
    确定性：`argsort(kind="stable")` + `argmin`（平手取首个）⇒ **零 RNG 消费**。
    """
    order = np.argsort(values, kind="stable")
    asc = values[order]
    cum = np.cumsum(areas[order])
    k = int(np.argmin(np.abs(cum - frac_below * float(cum[-1]))))
    return float(asc[k])


def _pointer_jump(link: NDArray[np.int64]) -> NDArray[np.int64]:
    """指针跳步（路径压缩）：返回每格沿 `link` 前进到的**吸收点**（link[i]==i 的格）。

    用途（C0b，两者都是 O(n log 深度) 的**向量化**实现，零 Python 标量循环）：
      * `link = 接收者（根指向自身）` ⇒ 每格的**汇流终点**（root）→ 流域闭合校核；
      * `link = 洼地格→接收者、非洼地格→自身` ⇒ 湖格的**第一个下游非洼地格 = 溢流口**。
    正确性：`link` 沿填洼面严格下降 ⇒ 无环 ⇒ 不动点存在；每轮 `link = link[link]`
    把到不动点的距离**折半** ⇒ ≤ ⌈log2 n⌉+1 轮收敛（超上限 = 图有环 ⇒ fail-loud）。
    """
    s = np.asarray(link, dtype=np.int64).copy()
    for _ in range(64):                        # log2(460800) ≈ 19 ⇒ 64 轮是宽裕上界
        nxt = s[s]
        if np.array_equal(nxt, s):
            return s
        s = nxt
    raise FloatingPointError(
        "C0b 指针跳步 64 轮未收敛（沿接收者链应严格下降 ⇒ 图无环）——"
        "路由树异常，中止（不静默给错标签）。"
    )


class EnvField:
    """一次构造、全静态的地形—水分—产能因子场（引擎构造期建好，之后只读）。

    实例属性（`__slots__` 全部字段）
    ------------------------------
    world, lt : 网格与光温场（只读引用）。
    cfg : `EnvFieldConfig`（参数契约）。
    h : NDArray[float64] (n_cells,) —— 地形 H ∈ [0,1]（已含 shuffle 置换，若开）。
    h_routed : NDArray[float64] (n_cells,) —— 路由面（平滑 + 填洼后的 H；只用于汇流）。
    temp : NDArray[float64] (n_cells,) —— 气候态温度 T = base − lapse×H（°C）。
    w : NDArray[float64] (n_cells,) —— 水分场 W ≥ 0。
    acc : NDArray[float64] (n_cells,) —— 汇流累积 A（自产 yield 已含）。
    river_mask : NDArray[bool] (n_cells,) —— 汇流带（A ≥ 98 分位；**读数命名层**）。
    filled : NDArray[float64] (n_cells,) —— 填洼面（priority-flood 输出；只进路由，不进物理量）。
    receivers : NDArray[int64] (n_cells,) —— 路由树父节点（`filled` 面 D8 最陡下降；-1 = 出水口）。
    pit_first, flood_filled : int —— 平滑面局部极小值数（洼地首数）/ 被淹没抬升格数（诊断）。

    🔴 以下均为 **C0b 开档专属**（`c0b_enabled=False` ⇒ 一律 `None`，且不参与任何计算）
    h_m : NDArray[float64] (n,) —— 真实海拔（米）= h^coast_gamma × peak_altitude_m。
    sea_level_m : float —— 海平面（米），由 `land_fraction` 的**面积权分位**定义。
    ocean_mask / land_mask / shelf_mask : NDArray[bool] (n,) —— 海 / 陆 / 大陆架（浅海分段）。
    lake_mask : NDArray[bool] (n,) —— 湖格（= 洼地格 ∧ 非海；与 `flood_filled` 同语义）。
    fresh_lake_mask / salt_lake_mask : NDArray[bool] (n,) —— 溢流链达海 / 不达海（§5.2）。
    river_mask_legacy : NDArray[bool] (n,) —— 旧「acc ≥ P98」读数层（开档下仍保留，供兼容对照）。
    surface_type : NDArray[uint8] (n,) —— 海/陆/湖/河表面类型层（🔴 载体层，不进判据前置）。
    lake_count / max_lake_cells / max_lake_basin_cells : int —— 湖数（按溢流口合并）/
        最大湖格数 / 最大湖所在流域的格数（流域闭合校核用）。
    conservation : dict —— §5.2 三条守恒不变量的实测值与残差。
    """

    __slots__ = (
        "world", "lt", "cfg",
        "h", "h_routed", "temp", "w", "acc", "river_mask", "receivers",
        "w_ref", "pit_first", "flood_filled", "filled",
        "_gf", "_cf",
        # ---- C0b（默认关 ⇒ 全 None ⇒ 不进任何计算 = 逐位等价）----
        "h_m", "sea_level_m", "ocean_mask", "land_mask", "shelf_mask",
        "lake_mask", "fresh_lake_mask", "salt_lake_mask", "river_mask_legacy",
        "surface_type", "lake_count", "max_lake_cells", "max_lake_root",
        "max_lake_basin_cells", "conservation",
        # ---- C1（默认关 ⇒ None ⇒ Pr 行逐位不变 = 逐位等价）----
        "band",
    )

    def __init__(self, world: SphereWorld, lt: LightAndTemperature,
                 cfg, base_seed: int = 42) -> None:
        self.world = world
        self.lt = lt
        self.cfg = cfg
        if bool(cfg.uniform_base_t) and bool(cfg.insolation_profile):
            raise ValueError(
                "uniform_base_t 与 insolation_profile 互斥：同开时温度分支顺序执行会静默"
                "以 insolation 覆盖 uniform ⇒ 宁炸不静默（config 层 __post_init__ 拦截不了"
                " post-init 赋值，故引擎侧兜底）。"
            )
        n = world.n_cells
        cells = np.arange(n, dtype=np.int64)

        # ---- 1) 地形 H：3D 值噪声 fBm（独立流；球面坐标采样 ⇒ 无接缝/极点奇点）----
        rows, cols = world.rows, world.cols
        lat_cell = np.repeat(np.asarray(world._lat, dtype=np.float64), cols)
        lon = 2.0 * np.pi * np.arange(cols, dtype=np.float64) / cols
        lon_cell = np.tile(lon, rows)
        cos_lat = np.cos(lat_cell)
        p = np.stack([cos_lat * np.cos(lon_cell),
                      cos_lat * np.sin(lon_cell),
                      np.sin(lat_cell)])                 # (3, n) 单位球面点
        rng = np.random.default_rng([int(base_seed), int(cfg.salt), _STREAM_NOISE])
        h = np.zeros(n, dtype=np.float64)
        amp, freq, norm = 1.0, 1.0, 0.0
        for _ in range(int(cfg.terrain_octaves)):
            h += amp * _value_noise_3d(rng, p, int(cfg.terrain_lattice), freq)
            norm += amp
            amp *= 0.5
            freq *= 2.0
        h /= norm                                            # 八度归一：H 幅值 ~N(0, σ)
        lo, hi = float(h.min()), float(h.max())
        if hi - lo < 1e-12:
            raise FloatingPointError("ENV 地形退化为常值（幅值 ~0）—— 参数异常，中止")
        h = (h - lo) / (hi - lo)                             # → [0,1]
        if cfg.shuffle:
            # 🔴 置换零模型（对照臂）：保 H 直方图、毁空间结构；独立流（不扰动噪声流）
            rng_sh = np.random.default_rng([int(base_seed), int(cfg.salt), _STREAM_SHUFFLE])
            h = h[rng_sh.permutation(n)]
        self.h = h

        # ---- 1b) C0b：真实海拔（H 米制化）+ 面积权分位定海平面（默认关 ⇒ 整块跳过）----
        #   🔴 `self.h` 保持"原始归一地形 [0,1]"不动 ⇒ 开/关档 `h` 层逐位可比，
        #     米制高程另存 `h_m`（归因干净：变化只可能来自 h_m/海平面，不来自噪声层）。
        areas = world.cell_area(cells)                   # 面积权（海平面分位 + W_ref 归一用）
        c0b = bool(cfg.c0b_enabled)
        self.h_m = None
        self.sea_level_m = None
        self.ocean_mask = None
        self.land_mask = None
        self.shelf_mask = None
        self.lake_mask = None
        self.fresh_lake_mask = None
        self.salt_lake_mask = None
        self.river_mask_legacy = None
        self.surface_type = None
        self.lake_count = None
        self.max_lake_cells = None
        self.max_lake_root = None
        self.max_lake_basin_cells = None
        self.conservation = None
        h_eff = h                                        # 关档 = h 本体 ⇒ 后续逐位不变
        if c0b:
            gamma = float(cfg.coast_gamma)
            if gamma != 1.0:
                # e^γ 海岸复杂度重映射（γ<1 ⇒ 低地被抬高 ⇒ 海岸更碎、陆比升高；§2.3-①）
                h_eff = np.power(h, gamma)
            h_m = h_eff * float(cfg.peak_altitude_m)     # 真实海拔（米）
            sea_level_m = _area_threshold_below(
                h_m, areas, 1.0 - float(cfg.land_fraction))
            ocean = h_m <= sea_level_m
            self.h_m = h_m
            self.sea_level_m = float(sea_level_m)
            self.ocean_mask = ocean
            self.land_mask = ~ocean
            # 大陆架 = 海洋 ∧ 海平面以下 0～shelf_depth_m（§2.3-①：坡折外缘均深 ≈130 m）
            self.shelf_mask = ocean & (h_m > sea_level_m - float(cfg.shelf_depth_m))

        # ---- 2) 气候态温度与水量平衡（Magnus）----
        base_t = lt.base_temperature(cells)                  # 纬度基温（年气候态，°C）
        if bool(cfg.uniform_base_t):
            # 🔴 去混杂选项（`uniform_base_t`；默认关 = 不走本支 ⇒ 逐位不变）：
            #   基温置**面积权全球均值**常数 —— 只杀纬度梯度、不抬降全球均值
            #   （面积权均温逐位保持；动机 = ADR 教训 2「Magnus 纬度压印」：
            #   T=base_t(φ)−lapse×H 过 Magnus 后 W 仍带纬度暗通道 ⇒ 置换零模型不纯）。
            #   本支零 RNG 消费（在噪声/置换流之后）⇒ 开档不改引擎 RNG 序列。
            _areas_t = world.cell_area(cells)
            _t_awm = float(np.sum(base_t * _areas_t) / np.sum(_areas_t))
            base_t = np.full(n, _t_awm, dtype=np.float64)
        if bool(cfg.insolation_profile):
            # 🔴 自然剖面（`insolation_profile`；**默认开 = 正式世界默认档**，裁定一 2026-10-14；
            #   显式 False = classic 档逐位回滚点）：
            #   基温置 **EBM 年平气温剖面**（辐照数值积分；常数全为地球来源——太阳常数/
            #   黄赤交角/反照率/玻尔兹曼/温室增量）⇒ 纬度影响与自然界一致，替代手定
            #   t_pole/t_equator 线性 cosφ。剖面只依赖 |φ| ⇒ 按行算再广播（零 RNG 消费）。
            _lat_row = np.asarray(world._lat, dtype=np.float64)
            base_t = np.repeat(_insolation_base_temperature(_lat_row), cols)
        temp = (base_t - (_LAPSE_C_PER_KM / 1000.0) * h_m) if c0b \
            else (base_t - float(cfg.lapse_c) * h_eff)
        #   ↑ C0b 开档 ⇒ 真实递减率 Γ=6.5 ℃/km × 真实海拔（替代 `lapse_c=20` 的现象学常数，
        #     §5.2）；关档 ⇒ `h_eff is h` ⇒ 与旧式 `base_t − lapse_c×h` 逐位相同。
        np.clip(temp, _T_FLOOR, None, out=temp)              # 🔴 防极端配置除零
        self.temp = temp
        q = _MAGNUS_C * np.exp(_MAGNUS_A * temp / (_MAGNUS_B + temp))
        # ---- 2b) C1：静态副热带干槽（显式纬度调制；默认关 ⇒ 整块跳过 = 逐位等价）----
        #   🔴 机理注（§5.3）：净产水 = max(Pr − Et, 0) ⇒ 只要副热带 band 压到
        #     < et_frac/(1+orographic×H) ≤ et_frac，C1 **不依赖蒸发约束**即可造出零产水带；
        #     “桶不再永远满”（蒸发受地面水约束）留给 C4a-2 做动态版。本机制只修降水侧，不碰 Et。
        #   零 RNG 消费（band 只依赖纬度 + 面积权归一），且发生在噪声/置换流之后。
        c1 = bool(cfg.c1_enabled)
        self.band = None
        if c1:
            phi = np.degrees(lat_cell)
            g_wet = np.exp(-((phi - float(cfg.itcz_deg)) / float(cfg.itcz_width_deg)) ** 2)
            g_dry = (np.exp(-((phi - float(cfg.dry_deg)) / float(cfg.dry_width_deg)) ** 2)
                     + np.exp(-((phi + float(cfg.dry_deg)) / float(cfg.dry_width_deg)) ** 2))
            raw_band = (1.0 + float(cfg.wet_amp) * g_wet - float(cfg.dry_amp) * g_dry)
            # 降水乘子非负：谷底压到 0（= 副热带无净产水）正是机制目的；负值无物理意义。
            raw_band = np.maximum(raw_band, 0.0)
            _m = float(np.sum(raw_band * areas) / np.sum(areas))
            if _m <= 1e-12:
                raise FloatingPointError(
                    f"C1 band 的面积权均值 = {_m!r}（应 >0）—— 参数组合把全球压成单侧"
                    "干槽？中止（不静默除零）。")
            band = raw_band / _m                              # 面积权均值 = 1（对照干净）
            self.band = band
            net = q * (band * (1.0 + float(cfg.orographic) * h_eff) - float(cfg.et_frac))
        else:
            net = q * (1.0 + float(cfg.orographic) * h_eff - float(cfg.et_frac))
        yield_ = np.maximum(net, 0.0)                        # 径流产额（土壤赤字 ⇒ 无径流）

        # ---- 3) 路由面：平滑（真 8 邻居）+ priority-flood 填洼（Barnes 2014）----
        nb = world._nb_table                                 # (n, 8)，极点行 = -1
        valid = nb >= 0
        safe_nb = np.where(valid, nb, 0)
        cnt = valid.sum(axis=1).astype(np.float64)
        pole_rows = cnt == 0.0                               # 极点行（真拓扑邻居 = 整行）

        h_r = h_eff.copy()
        if int(cfg.terrain_smooth) > 0:
            alpha = 0.5
            for _ in range(int(cfg.terrain_smooth)):
                mean_nb = h_r[safe_nb].sum(axis=1) / np.maximum(cnt, 1.0)
                new = (1.0 - alpha) * h_r + alpha * mean_nb
                new[pole_rows] = h_r[pole_rows]              # 极点行跳过（邻居 = 整行，见模块头）
                h_r = new

        # priority-flood：堆式单遍（极点行 = 出水口）。不变量（每次开堆）：
        #   filled[j] = max(H[j], filled[开堆者] + eps) ≥ filled[开堆者] + eps
        # ⇒ 沿"接收者 = 开堆者"的链，filled **严格下降** ⇒ 树无环、无需迭代收敛
        #   （旧"抬升至 min 邻居+eps"需 O(盆地深度) 轮，60×120 实测 200 轮不收敛 ⇒ 弃）。
        h_list = h_r.tolist()
        nb_l = nb.tolist()
        # 极点行在 `_nb_table` 是 -1 占位（真邻居 = 相邻纬度带整行，见 `sphere_world`）；
        # 但带内格的 `_nb_table` 只指回**规范槽 col 0**（rc_to_flat 极点坍缩）。
        # ⇒ 遍历表**只给规范槽**（0 与 (rows-1)*cols）补出边 = 与接收者邻接表严格同图；
        #   若给全部极槽补出边，带内格可能被"非规范槽"打开，而该槽不在它的 nb 邻接里
        #   ⇒ 最陡下降无严格下界 ⇒ 实测出环（60×120 cycle at 120）。
        #   其余极槽照旧当出水口（recv=-1、无出边、空转会弹出）。
        nb_l[0] = world._pole_nb[0].tolist()
        nb_l[(world.rows - 1) * world.cols] = world._pole_nb[1].tolist()
        done = bytearray(n)
        filled_l = h_list[:]
        heap: list = []
        # 🔴 种子集：关档 = 极点行（旧口径，逐位不变）；C0b 开档 = 海洋格 ∪ 极点行
        #   （§5.2"取消极点唯一出水口"；极点行仍须一并作种——非规范极槽在邻居图上
        #   孤立，不作种子就永远打不开 ⇒ 触发下方 fail-loud；详见模块头「C0b 的结构性近似」）
        if not c0b:
            seed_mask = pole_rows
            sea_interior = None
        else:
            # 海洋格全部标 done（filled = 自身高程），但**只有海岸格入堆**：
            #   洋面内部格的邻居全部已 done ⇒ 弹它会空转（却把堆深从“前线级”抬到
            #   “全球级”⇒ 每次 push/pop 变慢）。只入海岸格与全入堆**逐位等价**：
            #   陆格的首次开启者 = 它的海洋邻居中 filled 最小者（堆按 filled 升序弹出），
            #   而该海洋邻居必然是海岸格（它已有陆邻居 = 本格）⇒ 结果不变。
            sea_cells = np.flatnonzero(ocean)
            nb_sea = nb[sea_cells]
            has_land_nb = np.any((nb_sea >= 0) & ~ocean[np.where(nb_sea >= 0, nb_sea, 0)],
                                 axis=1)
            seed_mask = pole_rows.copy()
            seed_mask[sea_cells[has_land_nb]] = True
            sea_interior = sea_cells[~has_land_nb]
        for i in np.flatnonzero(seed_mask).tolist():
            done[i] = 1
            heap.append((h_list[i], i))
        if sea_interior is not None:
            # 洋面内部格：标 done、不入堆（上面已论证与全入堆逐位等价）
            for i in sea_interior.tolist():
                done[i] = 1
        heapq.heapify(heap)
        while heap:
            fv, i = heapq.heappop(heap)
            for j in nb_l[i]:
                if j >= 0 and not done[j]:
                    done[j] = 1
                    hv = h_list[j]
                    nv = fv + _FLOOD_EPS
                    if hv > nv:
                        nv = hv
                    filled_l[j] = nv
                    heapq.heappush(heap, (nv, j))
        n_done = int(sum(done))
        if n_done < n:
            raise FloatingPointError(
                f"ENV 路由：priority-flood 仅覆盖 {n_done}/{n} 格（邻居图不连通？）——"
                f"中止（不静默把余洼当汇）。"
            )
        filled = np.asarray(filled_l, dtype=np.float64)
        del nb_l, h_list, filled_l, done

        # 路由 = **filled 面的 D8 最陡下降**（Barnes 2014 的标准出口用法）：
        #   非洼格 filled == H ⇒ 就是地形最陡下降；洼内沿 eps 梯度 ⇒ 流向溢流口（湖面出流）。
        #   无环证明：每格 j 的开堆者 o 满足 filled[o] < filled[j]（严格，见上）⇒
        #   argmin 邻居的 filled < filled[j] ⇒ 沿接收者链严格下降 ⇒ 无环。
        filled_nb = filled[safe_nb]
        filled_nb = np.where(valid, filled_nb, np.inf)
        jmin = np.argmin(filled_nb, axis=1)                  # 平手取首列 ⇒ 确定性
        row_i = np.arange(n)
        receivers = np.where(valid[row_i, jmin], nb[row_i, jmin], -1).astype(np.int64)
        if c0b:
            # 🔴 海洋格 = 天然出水口：水量在海岸**离场**（不再在洋面上向下游搬运）
            #   ⇒ 下游累加/终点分类都以海岸为界（§5.2「海面即天然出口集合」）。
            receivers[ocean] = -1

        # 诊断：平滑面局部极小值数（"洼地首数"，填洼前的计数）
        v_nb = h_r[safe_nb]
        v_nb = np.where(valid, v_nb, np.inf)
        pit_first = int(((h_r < v_nb.min(axis=1)) & (~pole_rows)).sum())
        self.pit_first = pit_first
        self.flood_filled = int(np.count_nonzero(filled > h_r + 1e-12))
        self.h_routed = h_r
        self.filled = filled
        self.receivers = receivers

        # ---- 4) D8 汇流累积 A：按路由面 filled 降序（接收者恒更低 ⇒ 无环、无需求拓扑序）----
        #   纯 Python 标量循环（~0.3 s @460k，构造期一次性）；顺序 = filled 降序，
        #   处理到 i 时其全部上游已加完 ⇒ a[i] 定格后再摊给它唯一的接收者。
        order = np.argsort(-filled, kind="stable").tolist()
        recv_l = receivers.tolist()
        a_l = yield_.tolist()
        for i in order:
            r = recv_l[i]
            if r >= 0:
                a_l[r] += a_l[i]
        acc = np.asarray(a_l, dtype=np.float64)
        self.acc = acc
        a_ref = float(np.percentile(acc, 99.0))
        if a_ref <= 0.0:
            a_ref = 1.0                                      # 全零场兜底（配置异常，防 0/0）
        amp_r = 1.0 + float(cfg.runoff_gain) * acc / (acc + a_ref)
        self.w = yield_ * amp_r
        acc_p98 = float(np.percentile(acc, 98.0))
        self.river_mask = acc >= acc_p98          # 关档 = 旧「acc ≥ P98」读数层，逐位不变

        # ---- 4b) C0b：海/陆/湖/河表面类型层 + 三条守恒不变量（§5.2；默认关 ⇒ 整块跳过）----
        #   🔴 全部算法向量化（指针跳步 + unique + bincount）⇒ **零新增 RNG 消费**、
        #   零 Python 标量循环；复杂度 O(n log n)，在§5.2 的「+<0.5 s @s2」预算内。
        if c0b:
            land = self.land_mask
            self.river_mask_legacy = acc >= acc_p98      # 旧 P98 读数层保留兼容
            # 洼地格 = 被填洼抬升的格（与 `flood_filled` 同一条判据 ⇒ 复用语义）；湖 = 洼地 ∧ 非海
            depression = filled > h_r + 1e-12
            lake = depression & land
            self.lake_mask = lake
            # 每格的汇流终点 root（沿接收者链走到出水口）与湖格的溢流口（链上第一个非湖格）
            root = _pointer_jump(np.where(receivers >= 0, receivers, cells))
            reaches_sea = ocean[root]
            spill = _pointer_jump(np.where(lake & (receivers >= 0), receivers, cells))
            fresh = lake & reaches_sea
            salt = lake & ~reaches_sea
            self.fresh_lake_mask = fresh
            self.salt_lake_mask = salt
            # 河 = 陆地汇流格 ∧ acc ≥ P98（河道不含水面）；终点必为海/湖/内流盆地三类之一
            self.river_mask = (acc >= acc_p98) & land & ~lake
            # 🔴 表面类型层（载体层，永不进判据前置 = ENV-FIELD §四-4 红线 2）
            st = np.full(n, SURFACE_LAND, dtype=np.uint8)
            st[ocean] = SURFACE_OCEAN
            st[self.shelf_mask] = SURFACE_SHELF
            st[self.river_mask] = SURFACE_RIVER
            st[fresh] = SURFACE_LAKE_FRESH
            st[salt] = SURFACE_LAKE_SALT
            self.surface_type = st
            # 湖数/最大湖（按溢流口分组；unique/argmax 平手取首 ⇒ 确定性）
            if lake.any():
                uniq, cnt_lk = np.unique(spill[lake], return_counts=True)
                self.lake_count = int(uniq.size)
                self.max_lake_cells = int(cnt_lk.max())
                big = int(uniq[int(np.argmax(cnt_lk))])
                r_star = int(root[int(np.flatnonzero(lake & (spill == big))[0])])
            else:
                self.lake_count = 0
                self.max_lake_cells = 0
                r_star = -1
            self.max_lake_root = r_star
            root_counts = np.bincount(root, minlength=n)
            self.max_lake_basin_cells = (int(root_counts[r_star]) if r_star >= 0 else 0)

            # ---- §5.2 三条守恒不变量（构造期一次核账；残差入 probe + 单测）----
            area_sum = float(areas.sum())
            land_area_frac = float(areas[land].sum()) / area_sum
            ocean_area_frac = float(areas[ocean].sum()) / area_sum
            root_cells = receivers < 0
            basin_yield = np.bincount(root, weights=yield_, minlength=n)
            sea_roots = root_cells & ocean
            inland_roots = root_cells & land
            tot_yield = float(yield_.sum())
            tot_root_acc = float(acc[root_cells].sum())
            land_yield = float(yield_[land].sum())
            runoff_to_sea = float((acc[sea_roots] - yield_[sea_roots]).sum())
            inland_hold = float(acc[inland_roots].sum())
            self.conservation = {
                "land_area_frac": land_area_frac,
                "ocean_area_frac": ocean_area_frac,
                # ①a 海陆面积比互补恒等（分划，可严格到 1e-9）
                "area_complement_resid": abs(land_area_frac + ocean_area_frac - 1.0),
                # ①b 目标陆比偏差（离散阈必然落在某格 ⇒ 不可严格 1e-9；实测值归档）
                "land_frac_target": float(cfg.land_fraction),
                "land_frac_resid": abs(land_area_frac - float(cfg.land_fraction)),
                # ② 流域闭合：Σ全局产额 = Σ各终点 acc；且 陆地产水 = 入海径流 + 内流终点蓄
                "yield_sum": tot_yield,
                "root_acc_sum": tot_root_acc,
                "closure_resid": abs(tot_yield - tot_root_acc),
                "land_yield": land_yield,
                "runoff_to_sea": runoff_to_sea,
                "inland_hold": inland_hold,
                "budget_resid": abs(land_yield - runoff_to_sea - inland_hold),
                # ③ surface_type 计数与掩码逐位互证（下面当场校验，不靠事后读）
                "surface_counts": {str(int(k)): int(v) for k, v in
                                   enumerate(np.bincount(st, minlength=6))},
                "root_cells": int(root_cells.sum()),
                "ocean_roots": int(sea_roots.sum()),
                "inland_roots": int(inland_roots.sum()),
                "max_lake_root": r_star,
                "max_lake_basin_cells": self.max_lake_basin_cells,
                "max_lake_basin_yield": (float(basin_yield[r_star])
                                         if r_star >= 0 else 0.0),
                "max_lake_basin_acc": (float(acc[r_star]) if r_star >= 0 else 0.0),
            }
            # 🔴 结构性自洽校验（③ 的硬面）：失败 = 标签层写错，宁炸不静默
            scale = max(1.0, abs(tot_yield))
            if self.conservation["area_complement_resid"] > 1e-9:
                raise FloatingPointError(
                    f"C0b 守恒①a 破坏：海比+陆比 = {land_area_frac + ocean_area_frac!r}（应 1）")
            if self.conservation["closure_resid"] > 1e-6 * scale:
                raise FloatingPointError(
                    f"C0b 守恒② 破坏：Σyield={tot_yield!r} vs Σacc[终点]={tot_root_acc!r}"
                    "（汇流累加与链结构不一致）——中止。")
            if int((self.river_mask & (lake | ocean)).sum()) or \
                    int((salt & fresh).sum()) or int((lake & ocean).sum()):
                raise FloatingPointError(
                    "C0b 守恒③ 破坏：河/湖/海 掩码不互斥（surface_type 必与掩码逐位一致）。")
            if int((self.shelf_mask & ~ocean).sum()):
                raise FloatingPointError("C0b 大陆架掩码越出海洋集合——定义错，中止。")

        # ---- 5) 因子（按面积权均值 = 1 归一；sensitivity=0 ⇒ None = 整块跳过）----
        self.w_ref = float(np.sum(self.w * areas) / np.sum(areas))
        if not np.isfinite(self.w_ref) or self.w_ref <= 0.0:
            raise FloatingPointError(
                f"ENV 水分场 W 的面积权均值 = {self.w_ref!r}（应 >0）—— "
                f"et_frac/orographic/lapse_c 组合使竞量衡全负？参数异常，中止（不静默）。"
            )
        self._gf = None
        self._cf = None

    # ---- 因子出口（None = 此项关闭 ⇒ 引擎/资源场整块跳过 = 逐位等价）----------

    def _factor(self, sens: float, clip_hi) -> NDArray[np.float64] | None:
        if sens <= 0.0:
            return None
        u = self.w / self.w_ref
        if clip_hi is not None:
            u = np.clip(u, 0.0, clip_hi)
        f = np.power(u, sens)
        areas = self.world.cell_area(np.arange(self.world.n_cells, dtype=np.int64))
        m = float(np.sum(f * areas) / np.sum(areas))
        if m > 1e-12:                                        # 面积权均值归一（防 0/0）
            f = f / m
        return f

    def growth_factor(self) -> NDArray[np.float64] | None:
        """再生因子 f_g = u^water_sensitivity / 面积权均值；`water_sensitivity=0` ⇒ None。"""
        if self.cfg.water_sensitivity <= 0.0:
            return None
        if self._gf is None:
            self._gf = self._factor(float(self.cfg.water_sensitivity), None)
        return self._gf

    def capacity_factor(self) -> NDArray[np.float64] | None:
        """容量因子 f_c = clip(u,0,k_max)^cap_sensitivity / 面积权均值；`cap_sensitivity=0` ⇒ None。"""
        if self.cfg.cap_sensitivity <= 0.0:
            return None
        if self._cf is None:
            self._cf = self._factor(float(self.cfg.cap_sensitivity), float(self.cfg.k_max))
        return self._cf

    # ---- 诊断（读侧：probe/判读图用；**不进任何判据前置**）----------------------

    def probe(self) -> dict:
        """场统计快照（probe/manifest 用；纯读数，不影响动力学）。"""
        n = self.world.n_cells
        areas = self.world.cell_area(np.arange(n, dtype=np.int64))
        a_sum = float(np.sum(areas))
        u = self.w / self.w_ref
        out = {
            "salt": int(self.cfg.salt), "shuffle": bool(self.cfg.shuffle),
            "h_mean": float(np.sum(self.h * areas) / a_sum),
            "h_std": float(np.sqrt(np.sum((self.h - np.sum(self.h * areas) / a_sum) ** 2
                                          * areas) / a_sum)),
            "temp_min": float(self.temp.min()), "temp_max": float(self.temp.max()),
            "w_mean": self.w_ref, "w_p50": float(np.percentile(self.w, 50.0)),
            "w_p99": float(np.percentile(self.w, 99.0)), "w_max": float(self.w.max()),
            "u_p50": float(np.percentile(u, 50.0)), "u_p99": float(np.percentile(u, 99.0)),
            "u_max": float(u.max()),
            "acc_p50": float(np.percentile(self.acc, 50.0)),
            "acc_p99": float(np.percentile(self.acc, 99.0)),
            "river_cells": int(self.river_mask.sum()),
            "river_frac": float(self.river_mask.sum()) / n,
            "pit_first": self.pit_first, "flood_filled": self.flood_filled,
        }
        gf, cf = self.growth_factor(), self.capacity_factor()
        for tag, f in (("gf", gf), ("cf", cf)):
            if f is None:
                out[f"{tag}_on"] = False
            else:
                out[f"{tag}_on"] = True
                out[f"{tag}_min"] = float(f.min())
                out[f"{tag}_p99"] = float(np.percentile(f, 99.0))
                out[f"{tag}_max"] = float(f.max())
        # ---- C0b 开档读数（🔴 关档时不新增任何键 ⇒ 旧 probe/manifest 逐位不变）----
        if bool(self.cfg.c0b_enabled):
            out["c0b_on"] = True
            out["sea_level_m"] = self.sea_level_m
            out["land_area_frac"] = float(self.conservation["land_area_frac"])
            out["ocean_area_frac"] = float(self.conservation["ocean_area_frac"])
            out["shelf_cells"] = int(self.shelf_mask.sum())
            out["lake_cells"] = int(self.lake_mask.sum())
            out["lake_count"] = int(self.lake_count)
            out["fresh_lake_cells"] = int(self.fresh_lake_mask.sum())
            out["salt_lake_cells"] = int(self.salt_lake_mask.sum())
            out["max_lake_cells"] = int(self.max_lake_cells)
            out["max_lake_basin_cells"] = int(self.max_lake_basin_cells)
            out["river_cells_c0b"] = int(self.river_mask.sum())
            out["river_cells_legacy_p98"] = int(self.river_mask_legacy.sum())
            out["ocean_roots"] = int(self.conservation["ocean_roots"])
            out["inland_roots"] = int(self.conservation["inland_roots"])
            out["runoff_to_sea"] = float(self.conservation["runoff_to_sea"])
            out["inland_hold"] = float(self.conservation["inland_hold"])
            out["closure_resid"] = float(self.conservation["closure_resid"])
            out["surface_counts"] = dict(self.conservation["surface_counts"])
            out["flood_filled_frac"] = float(self.flood_filled) / n   # 对旧 50.8% 失真读数
        # ---- C1 开档读数（🔴 关档不新增键；band 归一与零产水带占比）----
        if bool(self.cfg.c1_enabled):
            a_sum = float(np.sum(areas))
            out["c1_on"] = True
            out["band_min"] = float(self.band.min())
            out["band_max"] = float(self.band.max())
            out["band_awm"] = float(np.sum(self.band * areas) / a_sum)
            out["w_zero_frac"] = float(np.count_nonzero(self.w <= 0.0)) / n
        return out

    def __repr__(self) -> str:  # pragma: no cover - 诊断用
        return (f"EnvField(cells={self.world.n_cells}, salt={self.cfg.salt}, "
                f"shuffle={self.cfg.shuffle}, pits={self.pit_first}/filled{self.flood_filled}, "
                f"rivers={float(self.river_mask.mean()):.3f})")
