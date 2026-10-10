"""SimConfig：整个模拟的配置文件（模块二 · 文件 1）。

模块职责
--------
集中管理所有数值参数（网格大小、光照、温度、资源、生物、基因、种群、
运行时长……），让"修改规则 = 改一个配置文件"，不用改代码。
与模块一的关系：SphereWorld / LightAndTemperature / ResourceField 的
构造参数都来自这里；它们自己不写死数值。

为什么用 dataclass？
--------------------
- 旧项目用 pydantic 校验，但本项目从零开始，为减少依赖（后面要接 Rust)
  改用 Python 内置 dataclass + 简单的 __post_init__ 断言；
- 配置即复现：same config + same seed → 完全相同的模拟。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields


@dataclass
class WorldConfig:
    """球面网格尺寸。"""

    rows: int = 60          # 纬度行数（row 0 = 上极 → row rows-1 = 下极）
    cols: int = 120         # 经度列数（经度环绕，col 0 == col cols-1 相邻）

    def __post_init__(self) -> None:
        assert self.rows >= 3, "至少要 3 行（上下极 + 至少一行赤道带）"
        assert self.cols >= 4, "经度至少 4 列"


@dataclass
class LightConfig:
    """光照温度参数（对应 LightAndTemperature 构造参数）。"""

    rotation_period: int = 2400   # 世界自转一圈的 tick 数（决定昼夜节律）
    t_equator: float = 30.0       # 赤道基温（抽象温度单位）
    t_pole: float = -20.0         # 极点基温（恒冷）
    day_boost: float = 6.0        # 昼夜温差幅度（正午比深夜高这么多）
    lat_base_ref: float = 1.0     # 纬度光照敏感度（越大极地越暗越冷）
    # ---- 13.7 季节（默认关 = 旧行为逐位等价）----
    #   tilt_rad ≠ 0 且 season_period > 1 ⇒ 启用季节：
    #   太阳赤纬 δ(t) = tilt_rad · sin(2πt / season_period) ⇒ 光照富集纬度带
    #   南北周期性移动 ⇒ 温度带移动 ⇒ 资源再生带移动 ⇒ 迁徙的物理驱动。
    #   ⚠️ tilt_rad=0.0（默认）⇒ 完全走旧路径（逐位等价，C7 digest 不动）。
    tilt_rad: float = 0.0         # 黄赤交角（弧度）；0.0 = 无季节（默认）
    season_period: int = 0        # 一个季节循环的 tick 数；0 = 无季节（默认）

    def __post_init__(self) -> None:
        assert self.rotation_period >= 10, "自转周期至少 10 tick"
        assert self.t_equator > self.t_pole, "赤道必须比极地暖"
        assert self.tilt_rad >= 0.0, "倾角非负（弧度）"
        assert self.season_period >= 0, "季节周期非负"
        if self.tilt_rad > 0.0 and self.season_period > 0:
            assert self.season_period >= 2, "启用季节时 season_period 至少 2"
            # 🔴 R149 前置门（软提示）：观测长度须 ≥ 3 个季节周期才看得到完整周期；
            #    此处不断言（观测长度在别处），由实验设计侧保证。
        elif self.tilt_rad > 0.0 or self.season_period > 0:
            raise ValueError(
                "季节须 tilt_rad>0 且 season_period>0 同时给出（否则语义不明）："
                f"tilt_rad={self.tilt_rad}, season_period={self.season_period}"
            )


@dataclass
class ResourceConfig:
    """资源（食物）参数（对应 ResourceField 构造参数）。"""

    capacity_per_area: float = 40.0   # 每单位面积的食物上限（容量）
    regrowth_rate: float = 0.5        # 基准再生：温度合适时每 tick 每格长多少
    # 🔴 13.4 波 2A（T2）：**本字段不改**（R178 只裁定 `eat_amount 0.5→0.9`）。
    #   配 0.9 后格内"吃>长" = 0.9/0.5 = **1.8 倍**（比设计稿 §11.1 建议的 1.5 更紧），
    #   全球账：总再生 `Σcell_area(4586)×0.5 = 2293` vs 总需求 `N(1944)×0.9 = 1750` ⇒ **+31%**
    #   （R178 核算：需求 1750 vs 总再生 3153 为**含斑块倍率**口径 ⇒ +80%；此处按基准再生口径
    #    更保守，仍安全）。⚠️ 改 `eat_amount` 已**改构造** ⇒ C7 基线 digest 变，
    #   须同步 7 处写死它的测试（任务书 T2 已列）。
    temp_sensitivity: float = 1.0     # 再生对温度的依赖（0=不 care，越大越敏感）
    initial_fill: float = 0.5         # 初始填充比例（每格开始有多少食物，0~1）

    # ---- 🔴 R196 光驱动再生（2026-09-24）------------------------------
    # 再生量额外乘 clip(光照,0,1)^light_sensitivity（"光合作用"）。
    # 与 `LightConfig.tilt_rad`/`season_period` 合用时：夏季半球日照长 ⇒ 再生快，
    # 冬季半球日照短 ⇒ 再生慢 ⇒ 食物丰度带随季节南北移动（"绿浪"）。
    # ⚠️ light_sensitivity=0（默认）时**整块跳过** ⇒ 与旧版逐位一致（回归安全）。
    # 物理原型：光合作用有效辐射（PAR）随日照时长/太阳高度角变化
    #   —— Sverdrup 临界深度（光限春季藻华）/ 绿浪假说（有蹄类跟随返青带）。
    light_sensitivity: float = 0.0    # 再生对光照的依赖（0=完全不看光，与旧版一致）
    light_normalize: bool = False     # 光照因子按全球均值归一化（保持全球平均再生量）

    # ---- 斑块化（L1，守恒版）------------------------------------------
    # distribution="uniform" 时以下字段全部不生效，行为与旧版完全一致。
    # "patchy" 时：食物聚簇到斑块，背景压低；两条守恒保证总食物量不变：
    #   ① 容量守恒：Σ_capacity 与 uniform 版相等（种群承载上限不变）
    #   ② 再生守恒：周期平均总再生量与 uniform 版相等（时间上供给不变）
    distribution: str = "uniform"          # "uniform" | "patchy"
    patch_count: int = 30                  # 斑块中心数（默认保守，避免覆盖过大）
    patch_radius: int = 2                  # 斑块半径（格，邻居扩散层数）
    patch_capacity_mult: float = 3.0       # 斑块格容量倍率（>1；建议 1.5~4，过大会背景容量为负）
    patch_regrowth_mult: float = 3.0       # 斑块格再生倍率
    # 🔴 13.5 ②（2026-09-24 派工；**2.0 → 3.0**）：**斑块再生校准**。
    #   依据 = K 公式反推（`K = Σ名义再生 ÷ 人均需求`，R185 实测命中，误差 1–3%）：
    #     背景归零后 Σ_斑块 名义再生 = 680（质量/tick，@倍率 2.0）
    #     加 `assim_herb=0.4` ⇒ 人均需求 = 0.762 ÷ (3.0×0.4) = 0.635 ⇒ K ≈ 1 071（偏低）
    #     ⇒ 斑块再生 ×1.5（倍率 2.0→3.0）⇒ Σ = 1 020 ⇒ **K ≈ 1 606** ∈ [1200,1900] ✓
    #   ⚠️ 为什么选**倍率**而不是 `regrowth_rate`（派工单"二选一"）：后者是**全局**基准，
    #     会一起改动所有非 patchy 预设与 13.4 复现；本字段**只作用于 patchy 分支**。
    #   ✅ 对 C7 无影响：9 处 digest 测试全部用 `distribution="uniform"`（本字段不参与）。
    #   ⚠️ `patch_mult × patch_frac + bg_mult × bg_frac = 1`（再生守恒式）会**自动重算** bg 倍率
    #     ⇒ 仍守恒；但若同时开 `bg_production_zero` ⇒ 该守恒式**被有意打破**（见该字段注释）。
    background_fill: float = 0.1            # 背景格初始食物占比（压低，否则协作无收益）
    bg_production_zero: bool = False
    # 🔴 13.5 ①（2026-09-24 派工）：**背景产能归零**（背景格容量 0 + 再生 0）⇒ **只留斑块生产**。
    #   依据（R183/R185）：背景格贡献了 **67.7% 的容量、78% 的再生**（Σ再生 3 153 中背景占 2 473）
    #     ⇒ 食物丰裕到"解析 K ≈ 12 400 vs 软顶 1 944"（可撑 6.4 倍人口）⇒ **食物从不绑定**
    #     ⇒ "位置（斑块）"因此在信息上一钱不值（E-019/E-023/E-027 的解释前提）。
    #   默认 False = **现状**（背景照常生产）⇒ 既有 patchy 批（13.4 各批）仍可复现；
    #   13.5 的预设显式置 True。
    #   ⚠️ 开启后两条构造期不变量**被有意打破**（这是本机制的目的，不是缺陷）：
    #     ① `Σcapacity` 从 183 430 降到 **32.3%**（斑块部分）；② 再生守恒式（=1）不再成立。
    #   相关自检须按"期望值"判，不能仍按 1.0 判（见 `ResourceField.__init__` 注释）。

    # ---- 🔴 R242 几何放宽（2026-09-28）：背景低产能带 ----------------------
    #   依据：R241 根因（`bg_production_zero=True` 下生产格仅占 2.78%，斑块是零食物沙漠
    #     中的孤岛 ⇒ 个体 97–100% 时间被钉在斑块格 ⇒ **跨斑块探索几何上不可能** ⇒ 门B 结构性失败）。
    #   本组字段给**部分背景格**极低产能，形成"绿洲链"续命带：穿行时偶尔踩到 ⇒ 胃不空 ⇒
    #     饿死豁免 ⇒ 能走更远；但净摄入 < 代谢 ⇒ **不养活**（不把沙漠变成宜居区）。
    #
    #   🔴 为什么不直接关掉 `bg_production_zero`（青梧执行令点名）：`resource_field` 的
    #     容量守恒式 `bg_cap_mult = (total − patch·mult)/bg_area` 会**自动摊均**背景容量
    #     ⇒ 背景拿回"正常产能" ⇒ 沙漠变宜居区 ⇒ **斑块的信息价值直接消失**（13.5① 的立论前提被推翻）。
    #     必须用**显式低倍率**接管背景，才能既开口子又保持"背景远差于斑块"的格局。
    #
    #   ✅ 两个字段**默认 0** ⇒ 走原路径（背景全 0 产能）⇒ **C7 digest 不动**（逐位等价）。
    bg_low_prod_frac: float = 0.0   # 背景格中获得低产能的比例（0=全零产能=现行为）
    bg_cap_mult: float = 0.0        # 背景低产能格的容量/再生倍率（0=零产能=现行为）

    def __post_init__(self) -> None:
        assert self.capacity_per_area > 0, "容量为正"
        assert self.regrowth_rate >= 0, "再生率非负"
        assert 0.0 <= self.initial_fill <= 1.0, "初始填充比例在 0~1"
        assert self.distribution in ("uniform", "patchy"), "distribution 只能是 uniform 或 patchy"
        # ---- 🔴 R196 光驱动再生 ----
        assert self.light_sensitivity >= 0.0, "light_sensitivity 必须非负（0=不看光）"
        assert isinstance(self.light_normalize, bool), "light_normalize 必须是布尔值"
        if self.distribution == "patchy":
            assert self.patch_count >= 1, "斑块数至少 1"
            assert self.patch_radius >= 1, "斑块半径至少 1"
            assert self.patch_capacity_mult > 1.0, "斑块容量倍率必须 > 1（否则无富集）"
            assert self.patch_regrowth_mult > 0, "斑块再生倍率为正"
            # ---- 🔴 R242 背景低产能带 ----
            assert 0.0 <= self.bg_low_prod_frac <= 1.0, "bg_low_prod_frac 应在 0~1"
            assert self.bg_cap_mult >= 0.0, "bg_cap_mult 非负"
            # 语义守卫：给了比例却不给倍率（或反之）⇒ 静默无变化（C9 家族）⇒ 直接报错
            if self.bg_low_prod_frac > 0.0:
                assert self.bg_cap_mult > 0.0, (
                    "bg_low_prod_frac>0 但 bg_cap_mult=0 ⇒ 低产能格仍是 0 产能，"
                    "配置静默无效（C9）。请同时给出 bg_cap_mult>0。"
                )
            assert 0.0 <= self.background_fill <= 1.0, "背景填充比例在 0~1"


@dataclass
class OrganismConfig:
    """个体能量收支与生命周期参数。"""

    initial_energy: float = 60.0      # 新生个体起始能量
    # 🔴 R144/R145 事故修正（2026-09-21）：本字段原注释写「多余溢出丢弃」，
    #   但**代码里从来没有过钳制** —— 文档承诺了不存在的不变量，十天无人核对。
    #   现改为：钳制由 `energy_cap_enabled` 显式控制（**默认关 = 与 E-017~E-031 可比**）。
    max_energy: float = 300.0         # 能量上限（**仅在 energy_cap_enabled=True 时钳制**；
    #                                    False ⇒ 旧行为：能量可无限累积，实测可达 33× 上限）
    energy_cap_enabled: bool = False  # 🔴 能量封顶开关（R144 四件之①）；
    #                                    关 = 旧纪元（可跨批比较 E-017~E-031/calib1）；
    #                                    开 = **新纪元**（禁跨比）。钳制点在 `_advance_one_tick` 末。
    base_metabolism: float = 0.6      # 每 tick 基础维持消耗（体温/活动）
    move_cost: float = 0.4            # 移动一格的基础能量消耗
    eat_amount: float = 0.9           # 每 tick 每格进食量上限
    # 🔴 0.5 → **0.9**（13.4 波 2A，T2；R178 裁定：0.6 在斑块格不成立——斑块再生实测
    #    0.832 > 0.6 ⇒ 斑块格永不枯竭）。0.9 时：斑块格净耗 +0.068（~750 tick 吃空）、
    #    背景格 +0.513（~27 tick 吃空）；全球账 1944×0.9=1750 vs 总再生 3153 ⇒ 裕度 +80%。
    #    ⚠️ 改构造 ⇒ C7 基线 digest 变，须同步 7 处写死它的测试（见任务书 T2）。
    eat_efficiency: float = 3.0       # 每单位食物转化为能量的倍率
    # 🔴 R148 §五.2（内评 §一.2 选项 b，所有者裁定「暂不统一但必须补声明」）：
    #   **`stomach` 有两条独立的容量上限，取决于写入路径** —— 这是**既有行为**，不是缺陷：
    #     ① **进食路径**（`sphere_engine.py` 取食段）：`max_energy/eat_efficiency × 0.5 × cap_mult`，
    #        `cap_mult = 0.5 + g5×1.5` ⇒ 均值 ≈62.5（上限 100，受 `g5 STOMACH_CAP` 调节）
    #     ② **捕食路径**（Python 捕食段 ／ `sim_core/src/predation.rs`）：`max_energy/eat_efficiency` = 100
    #        （满容量、无 ×0.5、不受 g5 缩放）
    #   ⇒ 两路径**各自只与自己的容量比较**；Python 与 Rust **互相一致** ⇒ 不是双路径漂移。
    #   ⚠️ 后果（须写进判读）：① 任何用 `stomach` 的指标**必须声明路径口径**；
    #     ② **`g5` 的选择效应是路径依赖的** ⇒ 若将来把 g5 纳入判据，须先统一两容量（= 第三纪元）。
    #   统一方案（内评推荐 (a)，未采纳）：由 `g5` 决定**唯一**上限、所有写入路径共用。
    photo_max: float = 0.1            # 光合最大产能：光照=1（赤道正午）时每 tick 产这么多
    # ============ 13.5 ③（2026-09-24 派工）：波 2-γ 能量标定补齐 ============
    #   🔴 **全部默认值 = 现状行为**（关 ⇒ 逐位等价；13.5 的预设显式改值）。
    #   ⚠️ 这批字段由 `[所有者]` 在 `sphere_engine.py` 接线（能量/取食/消化/死亡段），
    #     `[本地开发]` 只负责让字段在此**存在**且默认不改行为（R186 §三/§四）。
    stomach_cap_mass: float = 0.0
    #   胃容量（"饭盒大小"，**质量**单位）。**默认 0 ⇒ 用旧派生式**：
    #     `max_energy/eat_efficiency × 0.5 × (0.5 + g5×1.5)`（均值 ≈62.5）。
    #   >0 ⇒ **独立胃容量**（13.5 拟 25 质量 = 75 能量 = 体能 12.5%）。
    #   🔴 为什么必须独立（R166 §三 坑②）：否则它会跟着 `max_energy` 一起变大
    #     （300→600 ⇒ 胃也 ×2）⇒ 把"体能↑ 胃↓"这个设计意图**静默抵消**。
    #   ⚠️ 现状有**两条**胃容量口径（进食路径 ×0.5×cap_mult vs 捕食路径 =max/eff，
    #     见本类 `eat_efficiency` 处声明）⇒ 本字段接管后须**明确两条路径共用哪一个**。
    eat_threshold_frac: float = 0.0
    #   进食阈值（"不饿不吃"）：胃 < 容量 × 此值 才进食。默认 0.0 ⇒ **永远吃**（现状）；
    #   13.5 拟 0.6。⚠️ 与 `eat_amount` 联动：阈值会**降低**实际取食量 ⇒ 影响 K 的分子侧。
    starve_frac: float = 0.0
    #   **饿死**阈（占 `max_energy` 比例）：能量 < 此比例 **且** 胃≈空 ⇒ 死。
    #   默认 0.0 ⇒ 等价现状（能量 ≤0 才死）。13.5 拟 0.30（= 180 能量 @max 600）。
    exhaust_frac: float = 0.0
    #   **力竭**阈：能量 < 此比例 ⇒ 死（**无论胃里有没有食**）。
    #   默认 0.0 ⇒ 等价现状。13.5 拟 0.17（= 100 能量 @max 600）。
    #   🔴 语义顺序（必须写进单测）：先判 exhaust（力竭，无视胃），再判 starve（饿死，需胃空）
    #     ⇒ 防"抱着食物饿不死的僵尸态"。
    assim_herb: float = 1.0
    #   食草吸收率（"吃进去有多少变成能量"）。默认 1.0 ⇒ **无吸收损失**（现状：质量 ×eat_efficiency直通）；
    #   13.5 拟 0.4（文献：食草 0.36–0.78）。
    assim_carn: float = 1.0
    #   食肉吸收率。默认 1.0 = 现状；13.5 拟 0.8（文献：食肉 0.47–0.92）。
    #   ⚠️ 与 `eat_efficiency`(=3.0) 的分工：`eat_efficiency` 是**质量→能量**的固定换算（**不动**），
    #     本对是**吸收比例**（1−assim 的部分不变成能量）。
    assim_return_frac: float = 1.0
    #   未吸收部分**回流到本格植物池**的比例（碎屑回流，闭环；默认 1.0 = 全回流）。
    #   实现接口 = `ResourceField.deposit(flat, amount)`（13.5 新增，按容量封顶）。
    #   ⚠️ `assim_*=1.0` 时没有"未吸收部分" ⇒ 本字段**不起作用**（现状等价）✓
    homeo_upkeep: float = 0.15        # 恒温个体每 tick 的额外维持费（换取低温不减速）
    maturity_fraction: float = 0.15   # 成熟年龄 = 寿命的几成 → 达到才能繁衍（防止一出生就生）
    senile_fraction: float = 0.75     # 老年年龄 = 寿命的几成 → 进入衰老期，维持费上升
    growth_mult: float = 1.6          # 未成年（幼体）每 tick 维持费倍率（在长身体，吃得多耗得多）
    senile_mult: float = 1.4          # 老年每 tick 维持费倍率（器官退化，维持费上升）
    lifespan_mult: float = 1.0        # 寿命基准缩放（战役参数：不动昼夜，直接压寿命→世代缩短；1.0=旧行为）
    # 活性温度门（隐式选择压，审计标注 [隐含] → A2 收编）：
    # 冷血个体有效活动 = 环境活动 × (niche_floor + niche_gain × 温度适配度)。
    # 0.4 保底=即使完全不适配温度仍有 40% 活动 → 弱化 g11 温度偏好的选择梯度。
    niche_floor: float = 0.4
    niche_gain: float = 0.6

    # ===== L2 机动性（R146/R149；**由 [本地开发] 线添加**，仅这些字段）=====
    # 🔴 全部默认 = **旧行为**（L2 关闭时不读取 ⇒ H1 逐位等价）。
    #    `g18 DEFENSE` → **MOBILITY**（语义变更，见 `AGENT.md` §十三 13.2；旧批 g18 均值不可与新批比）
    dash_min_energy_frac: float = 0.2   # 冲刺（走 2 格）的**纯能量阈值** = frac × max_energy
    #                                     口径纪律（派工单 §3.2.1）：**仅此一条**，禁引入饱食/饥饿语义
    dash_cost_kappa: float = 1.0        # 冲刺耗能系数：cost × (1 + κ·(2^p − 1))
    dash_cost_exp: float = 2.0          # 代价指数 p（外鉴：凸性有支持、指数 2 是插值约定；p∈{1,2,3} 先注册）
    young_mob_mult: float = 0.55        # 幼体机动折减（fish 指定；**设计选择**，勿挂"老=慢"文献）
    old_mob_mult: float = 0.55          # 老年机动折减（同上）
    far_cap: int = 32                   # strict 2 圈规模 > 该值的格**不可冲刺**（拓扑退化）
    #                                     `[实测]` 不变区间 [16,119] ⇒ 非可调旋钮

    # ============ P0.0 阶段 A1（2026-09-26）：硬编码 tick 常数搬进配置 ============
    #   🔴 原位置 `sphere_engine.py`：`_repro_cooldown[ri] = genes[.., REPRO_COOLDOWN] * 60.0`
    #   量纲：**基因 × 每单位基因的 tick 数** = tick 时长 ⇒ 时间压缩时 **÷ k**。
    #   为什么它是"tick 常数"：`g12` 是**无量纲基因**，`60.0` 才是把基因翻译成
    #     世界时间的换算系数 ⇒ 昼夜一变，同一条 `g12` 的物理含义就变了（F4）。
    repro_cooldown_gene_scale: float = 60.0

    def __post_init__(self) -> None:
        assert self.initial_energy < self.max_energy, "初始能量要小于上限"
        assert self.eat_amount > 0, "进食量上限为正"
        assert self.photo_max >= 0, "光合产能非负"
        assert self.homeo_upkeep >= 0, "恒温维持费非负"
        assert 0.0 < self.maturity_fraction < self.senile_fraction < 1.0, (
            "成熟年龄要在老年年龄之前，且都要在寿限内"
        )
        assert self.growth_mult >= 1.0, "幼体代谢倍率至少 1"
        assert self.senile_mult >= 1.0, "老年代谢倍率至少 1"
        assert 0.05 < self.lifespan_mult <= 8.0, "lifespan_mult 在 (0.05, 8.0]"
        # 13.5 ③ 能量标定（默认为 0/1.0 = 现状 ⇒ 以下断言对默认值恒成立）
        assert self.stomach_cap_mass >= 0.0, "胃容量（质量）非负；0 = 用旧派生式"
        assert 0.0 <= self.eat_threshold_frac <= 1.0, "进食阈值是比例（0~1）"
        assert 0.0 <= self.starve_frac <= 1.0, "饿死阈是比例（0~1）"
        assert 0.0 <= self.exhaust_frac <= 1.0, "力竭阈是比例（0~1）"
        assert 0.0 < self.assim_herb <= 1.0, "食草吸收率 ∈ (0,1]（1.0 = 现状无损失）"
        assert 0.0 < self.assim_carn <= 1.0, "食肉吸收率 ∈ (0,1]（1.0 = 现状无损失）"
        assert 0.0 <= self.assim_return_frac <= 1.0, "回流比例 ∈ [0,1]"
        # P0.0 A1：繁殖冷却换算系数（基因 ⇒ tick）
        assert self.repro_cooldown_gene_scale > 0.0, "繁殖冷却换算系数为正"
        # ⚠️ 双阈值**不**强制 exhaust < starve：两者语义独立
        #   （`exhaust_frac` = 力竭，无视胃；`starve_frac` = 饿死，需胃空）。
        #   但若同时开启则应满足 `exhaust_frac ≤ starve_frac`（力竭先触发）——
        #   违反时**只警告不改值**（13.5 拟 0.17 / 0.30 ✓）。
        if self.exhaust_frac > 0.0 and self.starve_frac > 0.0:
            assert self.exhaust_frac <= self.starve_frac, (
                "双阈值同时开启时应 exhaust_frac ≤ starve_frac（力竭先于饿死触发）；"
                f"现值 {self.exhaust_frac} / {self.starve_frac}"
            )


@dataclass
class SignalsConfig:
    """信号场参数（P0.0 阶段 A1：把引擎里的**硬编码 tick 常数**搬进配置）。

    🔴 为什么必须搬进来：它是**以 tick 计价**的量 ⇒ 时间压缩（D 2400→480，k=5）时
    必须 ÷k 才有物理意义。硬编码在引擎里 ⇒ 机械清点（`tools/tick_denomination_audit.py`）
    扫不到 ⇒ **静默改科学**（F4）。
    """

    # 信号标记的有效寿命（tick）：写入时置为该值，每 tick −1，到 0 清除。
    #   原位置：`simulation/sphere_engine.py` `SignalField(self.world, duration=50)`
    #   量纲：**tick 时长** ⇒ 时间压缩时 **÷ k**。
    #   ⚠️ 语义耦合：`oracle.attribution_ok` 用 `age >= duration - persistence`
    #      判「最近 persistence tick 内写入」⇒ 两者必须**同量纲**地重标（都 ÷k）。
    duration_ticks: int = 50

    def __post_init__(self) -> None:
        assert self.duration_ticks >= 1, "信号寿命至少 1 tick"


@dataclass
class GenomeConfig:
    """基因底物参数：定长连续基因链 + 变异。"""

    gene_count: int = 24            # 基因数量：g0~g13 核心行为；g14 感知/g15 信号/g16 攻击/g19 扎根 已接线；
                                    # g17 食性/g18 防御/g20 享乐/g21 处理位/g22 信任阈值/g23 为声明未接线（预留位，变异无行为效果）
    gene_min: float = 0.0           # 基因取值下限
    gene_max: float = 1.0           # 基因取值上限
    mutation_rate: float = 0.05     # 每个基因发生变异的概率
    mutation_sigma: float = 0.05    # 变异震荡幅度（相对基因区间宽度）

    # R141（2026-09-21）：**g16 初始投放**（能量校准预实验用）。
    #   空串（默认）⇒ 旧行为（uniform 抽样）。给 `"0.05,0.5,0.9"` ⇒ 初始个体**均分**到
    #   这些 g16 值上（交错分配保证各组 n 尽量相等）⇒ **保证三组的样本量都够**。
    # ⚠️ 这**不是**自然分布 ⇒ 相应批次必须标为**仪器性质**（不进科学判读）。
    init_g16_clusters: str = ""

    def __post_init__(self) -> None:
        assert self.gene_max > self.gene_min, "上限要大于下限"
        assert 0.0 <= self.mutation_rate <= 1.0, "变异率在 0~1"


@dataclass
class PopulationConfig:
    """种群规模参数。"""

    initial_count: int = 200        # 初始个体数量
    max_count: int = 5000           # 种群硬上限（防失控）
    # 🔴 14.9 T1b（青梧找回入库，原在 stash）：创始代混合投放——
    #   >0 ⇒ 按比例把创始代撒进食物格（斑块内），其余仍均匀随机；
    #   0.0 = 纯均匀撒点（旧行为，逐位等价；RNG 调用序列不变）。
    #   用途：斑块出生子集测"富斑耗尽→失望离开"（卡点1）、荒漠出生子集测"找路"（卡点2），
    #   两子集读数分列。**实验投放参数，不改世界规则**（斑块布局/再生全不动）。
    init_patch_frac: float = 0.0

    # PC-1（R134，2026-09-20）：**S1 软顶**（δ/E-026 诊断"硬顶⇒无亚顶平衡态⇒ρ 无窗"的最小修改）。
    # 形态（冒烟后修订，见板帖）：**目标窗形式**——繁殖候选通过率
    #     p_soft = clamp((N* − N)/N*, 0, 1)，N* = soft_cap_target × max_count。
    # N*>0 ⇒ N 稳态钉在 N* 附近（出生≈死亡的选择窗）；N* = 0（默认）⇒ 与旧版**逐位一致**。
    # 硬顶保留为兜底。⚠️ True 消费额外 RNG（每 tick P 个 uniform）⇒ 新配置。
    # 🔴 修订原因（首版线性 (1−N/K) 实测失败）：无捕食世界死亡≈0 ⇒ 任何 p_soft>0 的尾部
    #    都把 N 推到硬顶（12k 冒烟 N_eq=3240=K，稳态窗门不过）⇒ 补偿必须**在 N* 处归零**。
    soft_cap_target: float = 0.0

    def __post_init__(self) -> None:
        assert self.initial_count >= 1, "至少一个个体"
        assert self.max_count >= self.initial_count, "上限不小于初始"
        assert 0.0 <= self.soft_cap_target < 1.0, "软顶目标须 ∈ [0,1)（0=关闭；1 等于没顶）"


@dataclass
class SimulationConfig:
    """引擎运行参数。"""

    ticks: int = 1000               # 计划运行的最大 tick 数
    stop_on_extinction: bool = True    # 种群归零时提前停
    history_limit: int = 0          # 统计历史保留上限（0=无限，长程实验用环形尾部）
    use_sim_core: bool = False      # True=种群数值管线走 Rust（sim_core.step_vectors）
    # ===== L2 机动性总开关（R146/R149；**由 [本地开发] 线添加**）=====
    # 🔴 默认 False = **旧行为**（不消费任何 L2 字段 ⇒ H1 逐位等价）。
    # ⚠️ H3：置 True 且 `use_sim_core=True` ⇒ **构造期直接抛 NotImplementedError**（fail-loud，
    #    不并进 `_d2_asym` —— 那是"静默换路径"，正是 dash PR 的翻车形态）。
    l2_dash: bool = False
    # ===== L1 感知追击（R146/R149；**由 [所有者·天平] 线添加** = B1）=====
    # 🔴 默认全关 = **旧行为**（关档不消费任何 L1 字段 ⇒ H1 逐位等价）。
    # ⚠️ H3：置 True 且 `use_sim_core=True` ⇒ **构造期直接抛**（同 `l2_dash`，不静默换路径）。
    # ⚠️ 两项必须**逐候选格**求值（R148-1 认账：逐个体标量 ⇒ 全候选取同值 ⇒ argmax 逐位
    #    不变、softmax 数学无效应 = "接了却一行行为没改"的静默 no-op）。
    l1_seek: bool = False           # L1a 追猎：`w_seek_max × g17`（g17 = DIET，语义变更见 §十三 13.2）
    l1_fear: bool = False           # L1b 恐惧：固定权重 `w_fear`（**无自由度**；方向由物理给出）
    w_seek_max: float = 0.5         # L1a 权重**上限**（预注册：0.5(B)/0.25(C) 两档都过才算）
    w_fear: float = 0.5             # L1b 固定权重（D 臂 = 0，作"fear 是否存在"的操作检查）
    # 猎物代理场口径（**不是"可食集"** —— 引擎里邻格**任何人**皆可被吃，故须自解释命名）：
    #   "lowagg" = `Σ 1[g16 ≤ attack_gene_gate]`（低攻击性个体代理）｜"any" = 任意占格者（E′ 归因臂）
    l1_prey_mode: str = "lowagg"
    social_move_weight: float = 1.0 # 移动决策群居项权重（D0 修复：densities 按邻居上限归一化后与感知项同量级，此项可扫描 0~2）
    stay_prob: float = 0.0          # 停驻率（战役参数：move_prob × (1-stay_prob)，0=旧行为每tick必移判定；配合D2感知半径4=等效扩世界）
    # ===== 13.4 波 2B：感知范围 / 单格上限 / 社交归一化（T3；**默认 = 旧行为**）=====
    # 🔴 `perception_span` 是**跳数**（1 = 现状；2 = 两圈），与 `perception_radius`（邻居
    #   个数 {4,8}）是**两个正交维度**（设计稿 §一 F2）—— 前者扩距离、后者换形状。
    # ⚠️ span=2 需重建 `_nb_table` ⇒ **构造级**（digest 变，须同步测试）；span=1 逐位等价。
    perception_span: int = 1         # 感知半径（跳数）：1（默认=旧行为）/ 2
    perception_cap: int = 32         # F3 闸（设计稿 §一）：ring1+2 规模 > cap ⇒ 该格降级为 1 圈
    # 🔴 F1 修复（任务书 T3）：`densities` 归一化除数由 stride(120) 改**实际邻居数** ⇒
    #    社交项增强 ~15 倍（C9 型缺陷修复）＝ 构造级变更（digest 变）。保留显式常量
    #    语义：`social_norm` 仅在 span=1 时 = 实际邻居数；改它 = 改社交项量级 = 构造级。
    # ⚠️ 与波 2 设计稿（[本地开发] 建议冻结 120）冲突 —— 任务书（[所有者]）裁定改实际
    #    邻居数，本字段 = 裁定落点（默认 "auto" ⇒ 每格实际邻居数）。
    social_norm: str = "auto"        # "auto"=每格实际邻居数（F1 修复）| 数字=冻结常量
    cell_occupancy_cap: int = 3      # 单格个体上限（默认 3；score 层剔除满格，落本格不受限）
    #   🔴 默认 3 但**不进新路径**（cap 只在 `_cap_on` 时生效 ⇒ 默认关 = 旧行为逐位等价）
    cell_occupancy_cap_enabled: bool = False   # 🔴 独立开关：默认关（旧行为），T3 接线
    # 🔴 "看见才出手"（T3）：`perception_span` 统一供给资源/信号/猎物/威胁感知；
    #    `attack_range` 独立恒 1 格（看得见 ≠ 够得着）。该开关随 `perception_span=2` 生效。
    attack_range: int = 1            # 攻击射程（恒 1，独立于感知范围）
    # ===== R217 §五 #1 稀疏化 B（惰性/稀疏结算；**由 [本地开发·性能线] 添加**）=====
    # 目标：消"大世界每 tick 固定烧全场"的地板税（480×960 T4 档实测：N=0 时
    #   regrow 全场链 ≈40 ms + LT ≈5 ms + 信号场 ≈6 ms 墙钟/tick，占稳态 130 ms 的 ~1/3）。
    # 开 ⇒ **Python 路径**下：① 资源再生只结算"脏格"（`_grid < _capacity`）；
    #   ② 信号场只推进"有标记（`age>0`）"的格。两者都是**逐位等价**改写
    #   （论证见 `world/resource_field.py::_regrow_lazy` 与 `world/signal_field.py::tick`）：
    #   窗口恒 1 tick、子集执行与全场逐元素同式、净格在非负增长下是逐位 no-op。
    # ⚠️ **范围锁**（引擎侧）：
    #   `use_sim_core=True` ⇒ 两侧都不启用（Rust 直写 `_grid`/`_marks`，打脏点覆盖不到）；
    #   R231 T-E：**信号侧与资源侧解耦** —— 信号活跃集（`age>0`）自包含、与 rd 正交
    #   （rd 不写 `_marks`）⇒ **rd 开档信号侧照常稀疏**。
    #   R244 v1：**资源侧放开 rd+bgzero 档**（子集再生 + 分母补算，逐位对拍通过）；
    #   `rd ∧ ¬bgzero` ⇒ **构造期 fail-loud**（轮作搬斑块 ⇒ 掩码/容量逐 tick 变化 ⇒
    #   净格恒净前提破裂；宁炸不静默）—— 见 `SphereEngine.__init__`。
    #   资源侧前提校验（`ResourceField.enable_lazy`：patchy / `temp_sensitivity==1.0` /
    #   非负倍率等）不满足 ⇒ 资源侧静默退回全场（信号侧不受影响）。
    # 🔴 默认 False = **旧行为逐位不变**（全场分支代码原样保留；本开关 = 唯一回滚点）。
    # ⚠️ 挂进 `SimConfig` ⇒ 经 `asdict` **自动进 `fingerprint()`**（同 `subpos` 家族）：
    #   开关不同的两份配置指纹不同（跨档续跑会被拦——它虽不改数值，但改算路）。
    sparse_fields: bool = False

    def __post_init__(self) -> None:
        assert self.ticks >= 1, "至少跑一个 tick"
        assert 0.0 <= self.stay_prob < 0.95, "stay_prob 在 [0, 0.95)"
        assert self.perception_span in (1, 2), "perception_span 只许 1/2（跳数，与 radius 正交）"
        assert self.cell_occupancy_cap >= 1, "cell_occupancy_cap ≥ 1"
        if self.social_norm not in ("auto",):
            try:
                float(self.social_norm)
            except ValueError:
                raise AssertionError("social_norm 须是 'auto' 或数字字符串")
        assert self.attack_range >= 1, "attack_range ≥ 1"


@dataclass
class PleasureConfig:
    """愉悦度系统参数（L2）：预测误差驱动的内在动机系统。

    核心公式：愉悦度 = 实际获得 − 预期获得（RPE，对应多巴胺系统）。
    不是"做好事给分"，而是"比预期好就愉悦"。
    """

    enabled: bool = True               # 总开关（False 时愉悦度数组仍存在但不更新）
    expectation_size: int = 120        # 情境数（能量5×食物4×邻居3×信号2=120）
    alpha: float = 0.05                # EWMA 预期学习率（越小越慢、越稳定）
    valence_decay: float = 0.95        # valence 每 tick 衰减（回到中性 0）
    arousal_decay: float = 0.97        # arousal 衰减（意外事件→高唤醒）
    baseline_rate: float = 0.001       # baseline 慢漂移率（习惯化）
    optimism: float = 0.8               # 初始乐观系数（expectation = optimism × max_reward）
    max_reward: float = 2.0             # 单 tick 最大可能收益（用于乐观初始化归一化）
    w_energy: float = 0.5               # 事件收益：Δ能量权重
    w_info: float = 0.3                 # 事件收益：信息增益权重
    w_social: float = 0.2               # 事件收益：社会增益权重
    inheritance_noise: float = 0.02     # 繁殖时 expectation 继承噪声（文化传递载体）
    # 社会事件基础效价（审计标注 [隐含] → A2 收编）：
    # 有同伴 ≥1 → +social_rpe / 孤独 → alone_rpe（注意非对称：0.2 vs -0.1）。
    social_rpe: float = 0.2
    alone_rpe: float = -0.1

    def __post_init__(self) -> None:
        assert 0 < self.alpha <= 1, "alpha 应在 (0,1]"
        assert 0 < self.valence_decay <= 1, "valence_decay 应在 (0,1]"
        assert self.expectation_size == 120, "当前情境编码固定为 5×4×3×2=120"


@dataclass
class PredationConfig:
    """捕食参数（L4，影响 g16 攻击性的 fitness）。

    审计标注：能量转移率 0.4 是[隐含]最强"战斗红利"、成功率乘 g16 是[刻意]强选择、
    攻击概率系数/门槛为[隐含]（→ A2 收编，默认值保持旧行为逐位一致）。
    """

    # PC-1（R134，2026-09-20）：**总开关**。True（默认）⇒ 与旧版**逐位一致**（C7 digest 钉死）；
    # False ⇒ **跳过整个捕食相**（同类相食 G16 不发动；pred_frac 恒 0）——PC-1 单营养级构造件。
    # ⚠️ False 是**新配置**（RNG 消费随之改变），不是"旧行为的变体"；两开关都经 to_dict 进指纹。
    enabled: bool = True

    attack_cost: float = 0.1           # 每次攻击的能耗（无论成败）
    attack_prob_coef: float = 0.2      # 攻击概率 ≈ g16 × 系数 × 饥饿度
    attack_gene_gate: float = 0.3      # g16 低于该值不发动攻击
    success_gene_gain: float = 0.5     # 成功率 = 能量比 × (0.5 + g16×gain)，clamp
    success_floor: float = 0.1         # 成功率下限
    success_ceil: float = 0.9          # 成功率上限
    transfer_ratio: float = 0.4        # 捕食成功：猎物能量转移比例
    stomach_transfer: float = 0.4      # 猎物胃粮转移比例

    # ------------------------------------------------------------------
    # R135 第 3 步 **A-连续**（2026-09-20）：营养级专化的**凸 trade-off**。
    #
    # 取食倍率 `forage_mult(g16) = (1 − g16) ** k`：
    #   k = 0（默认）⇒ 恒 1 ⇒ **与旧版逐位一致**（C7 digest 钉死）
    #   k = 1     ⇒ 线性权衡（g16=0.5 仍能吃 50%）
    #   k > 1     ⇒ **凸（加速下降）**：g16=0.5 只吃 (0.5)^k ⇒ k=2 时仅 25%
    #      ⇒ 中间态"杂食者"两边都不精 ⇒ 这是文献里唯一经检验能产生**进化分支**
    #        （g16 分布双峰）的路径：Geritz et al. 1998, *Evol. Ecol.* 12:35（凸权衡 + 频率依赖）。
    #   与云端开发者 §三"陡峭表"的对照（k=1.74 最接近该表；本批取 **k=2.0** 以保证凸度足够）：
    #      g16:    0.0    0.2    0.5    0.8    1.0
    #      表:    1.00   0.80   0.30   0.05   0.00
    #      k=2:   1.00   0.64   0.25   0.04   0.00
    # ⚠️ 只动**取食侧**。捕猎成功率侧保持原式 `energy_ratio × (0.5 + g16×0.5)`
    #    （其在 Rust `predation.rs:99`，改它要重编；且 R135 裁定"一次只动曲率"）。
    forage_tradeoff_k: float = 0.0

    def __post_init__(self) -> None:
        assert self.attack_cost > 0
        assert 0 <= self.success_floor < self.success_ceil <= 1.0
        assert 0.0 <= self.transfer_ratio <= 1.0
        assert 0.0 <= self.stomach_transfer <= 1.0
        assert 0.0 <= self.attack_gene_gate <= 1.0
        assert self.forage_tradeoff_k >= 0.0, "凸度非负（0 = 关闭 = 旧行为）"


# ---------------------------------------------------------------- "丰盛"阈值（R121 §4.1：**命名 + 度量**）
# 🔴 2026-09-18 立（R121 核阅 §4.1 批准；内评 09-17 §七.2 派工）：
#    此前"食物 >= 0.5x容量"这个判定在**多处各自硬编码**（信号 `f_bit`、工作记忆写入），
#    而付款/信任/学习另用 `CultureConfig.food_threshold = 0.3` ⇒ 同源概念多值声明，
#    且两处**语义并不相同**。本次**只命名、不改数值**——改数值属**行为变更**，
#    与在产批次（C1a/C1b/C2）不可比，且须预注册（R121 明文）。
#    两者务必分清：
#      · `FOOD_RICH_LEVEL`            = "此地食物多到**值得记住**"（工作记忆写入 `:730`；
#                                        信号 `f_bit` 已随 R113 的 "4" 档删除）
#      · `CultureConfig.food_threshold` = "此地有**足够食物可食 / 值得付款**"（付款/信任/学习）
#    边界带 `(food_threshold, FOOD_RICH_LEVEL] = (0.3, 0.5]` 上的错位（系统判该付款、
#    信号却宣告"无食物"）由 `oracle_stats()["food_band_true_sig"]` 的**零机时 counter** 量化。
#    ⇒ **裁定点**：该占比 >5% 时再决定"对齐到 0.3"还是"对齐到 0.5"（R121 §4.1）。
FOOD_RICH_LEVEL: float = 0.5


@dataclass
class CultureConfig:
    """文化学习参数（L5，信任系统）。

    审计标注：信任更新非对称（+trust_true / −trust_false）是[隐含]反合作偏置，
    → A2 收编。默认值保持旧行为（真 +0.05 / 假 −0.1）。
    """

    food_threshold: float = 0.3        # "邻格有粮"判定阈值（**可食/值得付款**；与模块级 `FOOD_RICH_LEVEL`=0.5"值得记住"**语义不同**，见其注释）
    trust_true: float = 0.05           # 信号验证为真 → 信任上升幅度
    trust_false: float = 0.1           # 信号验证为假 → 信任下降幅度（注意取负前传）

    def __post_init__(self) -> None:
        assert 0.0 <= self.food_threshold <= 1.0
        assert self.trust_true >= 0
        assert self.trust_false >= 0


# ---------------------------------------------------------------- 信号发射成本（**单一真源**）
# 🔴 C5（2026-09-15 立）：此值此前在 `simulation/sphere_engine.py` 的 **两个分支里各硬编码一次**
#    （Rust 路径与 Python 路径），而 `experiments/preflight_check.py` 又抄了一份 ⇒ **同源值三处声明**，
#    是"改一处漏一处"的典型温床（本项目已付过 6 次同型学费）。
#    ⇒ 现统一为**模块级唯一常量**；引擎与 preflight 都必须引用它，不得再写字面量。
#    改动它 ⇒ 必须同时复核 `OracleConfig` 的保本断言与所有历史批的可比性。
SIGNAL_COST: float = 0.1

# R97 增益校准档的 `m` 合法区间（R100 条件 5 / v1.1 §五 C5 v2）。
# 同 SIGNAL_COST：**模块级唯一常量**，引擎/preflight/判读脚本一律引用，不得写字面量。
CALIBRATION_M_RANGE: tuple[float, float] = (1.2, 1.5)


@dataclass
class InfoStructureConfig:
    """D2 信息结构重构参数（语言涌现最小充分条件）。

    四大机制（评估报告 M2 gate，元宝/六模型共识）：
      1. 学习瓶颈：解读表不遗传，幼体从有限(信号,后果)观察样本归纳
      2. 任意性：信号pattern=个体可遗传码本[状态]，映射可漂移可协商
      3. 信息不对称：感知半径8→4 + 感知噪声 + softmax(tau)替代argmax
      4. Steels对齐：同格相遇概率性解读表对齐（带噪声）

    默认 enabled=False（完全不改变现有行为，向后兼容）；
    开启后唯一验收判据：g15 在无手调补贴下从 0.5 上升。
    """

    enabled: bool = False                # D2 总开关（False 时全部机制不执行，行为与旧版完全一致）

    # ---- 机制1：学习瓶颈 ----
    learning_bottleneck: bool = True     # 解读表不遗传，幼体随机初始化后通过观察学习
    learning_rate: float = 0.15          # 幼体观察学习率（每次观察更新解读表的步长）
    learning_samples_max: int = 60       # 学习瓶颈：单个体最多观察学习次数（达到后停止学习=瓶颈）
    learning_maturity_ticks: int = 1200  # 幼体学习窗口长度（tick），超过后不再学习

    # ---- 机制2：任意性码本 ----
    arbitrary_codebook: bool = True       # 信号pattern=码本[状态]，而非硬编码状态函数
    codebook_mutation_rate: float = 0.02 # 码本每位突变概率（繁殖时）
    codebook_mutation_sigma: float = 0.3 # 码本突变时新映射的随机强度

    # ---- 机制3：信息不对称 ----
    perception_radius: int = 4            # 感知半径：4=Von Neumann(上下左右), 8=Moore(全邻)
    perception_noise: float = 0.05        # 感知噪声σ（食物/信号感知时加高斯噪声）
    softmax_tau: float = 0.15             # 移动决策softmax温度（0=argmax旧行为，越大越随机探索）

    # ---- 机制3.5：R2 声誉权重（B1，C3 发送者获益通道的最小验证） ----
    # 移动得分中"信号项"随**自身 trust** 放大的系数：sig_weight = 0.5 + reputation_weight × trust。
    # 0.0 = 关闭（默认；行为与旧版逐位一致）。
    # 设计依据：docs/决策与评审/评估-EVAL-可行性分析-g15高位payoff-20260911.md §三（候选 R2）。
    # ⚠️ 纪律（R6）：生态门通过前【只开发、不判读】——不得据此跑判读实验或下结论。
    reputation_weight: float = 0.0

    # ---- 机制4：Steels 对齐 ----
    steels_alignment: bool = True         # 同格相遇时概率性解读表对齐
    alignment_rate: float = 0.1           # 相遇时对齐概率（每tick每对）
    alignment_step: float = 0.15          # 对齐步长（解读表向对方收敛的比例）
    alignment_noise: float = 0.02         # 对齐时附加噪声σ

    # ---- 机制3.6：A′ 记忆**朝向梯度**（设计稿 `docs/设计文档/设计-A档记忆朝向梯度-20260919.md`）----
    # 🔴 **不**受 `enabled` 门控（与 ⑥ 探针同规格）：A′ 必须能**单独**开关，否则测试会被
    #    学习瓶颈/任意性码本/softmax 等一堆 D2 机制污染 ⇒ 不再是单变量。
    # 现状问题：原记忆加分只在「记忆格 **恰好等于** 某个邻居格」时生效，而那格**本来就能直读**
    #    （`food_ratio[nbc]` 在同一次决策里已被读到）⇒ **按构造就是冗余奖励**。
    # "orientation"：改为朝向梯度 —— 记忆格（可远在感知之外）按其**方向**给对应邻居加分：
    #    `gain(c) = memory_gradient_gain · perc · max_m cos(方向(cur→c), 方向(cur→m))`
    #    🔴 `cos` **允许为负** ⇒ 背向邻居被减分 ⇒ 这才是"梯度"（不是单纯吸引）。
    # "none" = 原式（**默认** ⇒ 与旧版逐位一致，可对拍/回退/当同批对照臂）。
    memory_gradient: str = "none"
    memory_gradient_gain: float = 0.3

    # ---- D-18 ⑥ 探针（R43：信号响应率三联报）----
    # 纯观测（零 RNG、零行为改变——有测试断言逐位一致）；只增每 tick 一点算术开销。
    # False = 关闭（默认；完全无开销）。⑥a 暴露率 / ⑥b Δ_i / ⑥=⑥a×⑥b + argmax 翻转率辅助。
    measure_signal_response: bool = False

    # ---- S3 前置：记忆改造 v2（egocentric 版；实施规格 §一）----
    # 🔴 **不受 `enabled` 门控**（与 memory_gradient 同规格）：必须能**单独**开关（单变量）。
    # v2 取代 v1 orientation 的"世界方向"语义（R239 §一 红线：世界方向 = 变相罗盘）：
    #   记忆存**自我参照系方位 q**（Moore 槽位 0–7），打分用 `cos(候选相对朝向 − q)`，
    #   全程只用 当前格 + 邻居 + `_heading`，**不存/不用任何世界坐标**。
    # 默认 False = 现有行为逐位不变（回滚点；digest `(574887, 11266.746993)` 不漂移）。
    memory_v2: bool = False            # v2 总开关。False = 现有行为逐位不变（回滚点）
    memory_dist_scale: float = 15.0    # 精记忆距离衰减尺度 d0（= 斑块间距量级，R234 §一 实测）
    memory_degrade_thr: float = 20.0   # 精→粗降级距离阈值（格）。超过即**永久降级**（不可逆，R236 §四-A）
    memory_coarse_gain: float = 0.15   # 粗记忆增益 = 0.5 × memory_gradient_gain(0.3)（R236 §四-A）
    memory_ttl: int = 1000             # 记忆时效（tick）。age > TTL ⇒ 槽视为空（R236 §四-B-3：5000→1000）
    memory_noise: bool = False         # 朝向噪声档。False = 精确路径整合；True = 记忆方位旋转叠加噪声（±1 档）
    memory_noise_p: float = 0.1        # 噪声档：每 tick 以该概率扰动记忆方位 ±1 档
    memory_v2_dynamic_centroid: bool = True  # R264 质心动态化：rd 轮作搬移后同步重算受影响斑块质心。
    # True = 动态模式（rd+背景产能下允许 v2，搬移后质心表自动更新）；
    # False = 静态模式（rd+背景产能下 fail-loud 硬报错，保留旧行为）。

    # ---- 13.11 记忆权重基因位（g22；S3.5 演化级前置；R258 §二）----
    # 关（默认）= 固定权重（精 0.3/粗 0.15，= S3 批原语义，逐位不变）；
    # 开 = 乘子 `2×g22` 生效（g22∈[0,1] 均匀 ⇒ 乘子∈[0,2]、总体均值 1.0）。
    # 🔴 只在 memory_v2 打分路径被消费 ⇒ 要求 memory_v2=True（否则静默无消费点）。
    memory_weight_gene: bool = False

    def __post_init__(self) -> None:
        assert self.perception_radius in (4, 8), "感知半径只支持4(Von Neumann)或8(Moore)"
        assert 0.0 <= self.learning_rate <= 1.0
        assert self.learning_samples_max >= 1
        assert self.learning_maturity_ticks >= 1
        assert 0.0 <= self.codebook_mutation_rate <= 1.0
        assert self.perception_noise >= 0
        assert self.softmax_tau >= 0
        assert self.reputation_weight >= 0, "声誉权重非负（0=关闭）"
        assert self.memory_gradient in ("none", "orientation"), \
            "memory_gradient 只支持 none（原式）或 orientation（朝向梯度）"
        assert self.memory_gradient_gain >= 0, "记忆朝向梯度增益非负"
        assert 0.0 <= self.alignment_rate <= 1.0
        assert 0.0 <= self.alignment_step <= 1.0
        assert self.alignment_noise >= 0
        # S3 记忆 v2 预注册断言（实施规格 §一）：
        assert (not self.memory_v2) or self.memory_gradient == "orientation", \
            "memory_v2=True 要求 memory_gradient == 'orientation'（v2 取代 v1 世界方向版，不并存）"
        assert self.memory_dist_scale > 0, "memory_dist_scale 必须 > 0"
        assert self.memory_degrade_thr > 0, "memory_degrade_thr 必须 > 0"
        assert self.memory_coarse_gain >= 0, "memory_coarse_gain 必须 ≥ 0"
        assert self.memory_ttl > 0, "memory_ttl 必须 > 0"
        assert 0.0 <= self.memory_noise_p <= 1.0, "memory_noise_p ∈ [0,1]"
        # 13.11 记忆权重基因位：乘子只在 v2 打分路径消费 ⇒ 不与 v2 同开 = 静默 no-op（B3 家族）
        assert (not self.memory_weight_gene) or self.memory_v2, \
            "memory_weight_gene=True 要求 memory_v2=True（乘子只在 v2 打分路径被消费）"


@dataclass
class OracleConfig:
    """V-1 oracle 正向对照参数（R39 / D-8，利益对齐型 / Lewis 共利）。

    默认 enabled=False（关闭时行为与旧版**逐位一致**，C-6 先例同
    `reputation_weight`）。见 `simulation/oracle.py` 模块 docstring 与
    `_share/规格-V1-oracle引擎级-20260913.md`（v1.2）。
    """

    enabled: bool = False              # 默认关；关闭时行为与旧版逐位一致
    donation: float = 0.05             # 单次成功通信的能量转移量
    persistence: int = 10              # 归因窗口（tick）；0 = 仅本 tick
    require_food: bool = True          # True=利益对齐（落点须有食物）；False=仅"有人听"
    # C5 逃生阀：显式声明"本批有意研究补偿不足区制"（**仅供 Pre-Flight 放行**）。
    # ⚠️ 语义（2026-09-15）：它**不压制下面的硬断言**——因为 `from_dict` 必须能**忠实回放**旧快照，
    #    若断言可被字段压制，就只能靠"在 from_dict 里偷偷改写值"来兼容 ⇒ 破坏"载入=原样"契约。
    #    ⇒ 分工：**硬断言**保证"跑不起来不自洽的配置"；**pre-flight** 读回该字段决定是否**放行**。
    allow_non_breakeven: bool = False
    # ---- R97 增益校准档（2026-09-16 实施；R100 条件 1–7 / R102 更正 / R103 命名 / R105 判据）----
    # 唯一变更点：把**每发送者的终身额度上限**由 `1×` 抬到 `m×`：
    #     budget = m × EMISSION_COST × emit_count − oracle_gain
    # 默认 1.0 ⇒ **与现状逐位兼容**（旧快照缺该键 ⇒ 回退默认，同 reputation_weight 先例）。
    # ⚠️ **C-2 守恒恒成立**（R102 §三 二阶更正 + 内评 §二 会签）：抬 `m` 只放宽"发送者**能收多少**"
    #    的上限，每笔转移仍是「扣减 == 入账」⇒ 纯再分配、**不生成能量** ⇒ 无需任何例外声明。
    #    发送者的"净收益"由**接收者支付**承担 ⇒ 接收侧代价必须随 `ratio` 一并报告（v1.1 §4.3）。
    gain_multiplier: float = 1.0
    # 校准臂登记（R100 条件 5「机器强制拒收」+ R103 §二.2）。
    # 未登记而 `m ≠ 1` ⇒ **硬失败**；标了旗而 `m == 1` 亦报错（m=1 属科学臂，防登记口径漂移）。
    is_calibration_arm: bool = False
    # ---- R123/B② 门控臂（2026-09-18 实施；R122 移交清单 #3）----
    # 动机（内评交叉核验 §一 判定 `[云端开发]` 质疑**成立**）：现付款条件 `sel = true_sig`
    # **不引用**"接收者的选择是否被信号改变" ⇒ 阳性可能是"食物占位"（谁站在猎物旁谁收款）。
    # 本档把 D-18 ⑥ 探针**已算**的反事实 Δ_i 接成付款闸：`ok = ... & (Δ_i > 0)`。
    #   · `gate_delta="content"`（**默认、付款闸口径**）= 只去**内容项** `0.4*perc*interp`、
    #     **保留存在性项** `sp*sig_weight` ⇒ 闸门问的是"**内容**是否有用"。若用 `full`，
    #     闸门会被"信标"穿透（存在性单独就够引路）⇒ 重演本要修的问题（本板 23:37 帖 §三）。
    #   · `gate_delta="full"` = 去掉存在性 + 内容（现探针口径）⇒ 并列报告用。
    # ⚠️ 门控臂是**仪器**（改付款规则、不改机制语义）⇒ 必须 `is_calibration_arm=True`（R100 条件 5），
    #    且必须同时开 `measure_signal_response` + 信息不对称路径（否则 Δ≡0 ⇒ **付款全消失**，
    #    看起来像"信号无用"的**假结论**——比没有数据更坏）。三处齐发硬失败（config/引擎/a4）。
    gate_mode: str = "none"            # "none"（默认=现状）/ "delta_positive"（门控臂）
    gate_delta: str = "content"        # "content"（付款闸推荐）/ "full"

    def __post_init__(self) -> None:
        assert self.donation >= 0, "donation 非负"
        assert self.persistence >= 0, "persistence 非负"
        assert self.gain_multiplier >= 1.0, (
            f"gain_multiplier({self.gain_multiplier}) 不得 < 1.0：增益档只**放宽上限**，不收紧"
        )
        # ---- R123/B② 门控臂：字段自洽（跨字段检查在 SimConfig 侧 + 引擎入口 + a4）----
        assert self.gate_mode in ("none", "delta_positive"), (
            f"gate_mode 非法：{self.gate_mode!r}（只支持 none / delta_positive）"
        )
        assert self.gate_delta in ("content", "full"), (
            f"gate_delta 非法：{self.gate_delta!r}（只支持 content / full）"
        )
        if self.gate_mode == "delta_positive":
            assert self.is_calibration_arm, (
                "门控臂是仪器（改付款规则、不改机制语义）⇒ 必须登记 "
                "is_calibration_arm=True（R100 条件 5：校准臂不进科学判定）"
            )
        # ---- C5 v2 规格自洽（增益档版；2026-09-16）----
        # 三态：① 科学臂（m=1）保持原语义；② 校准臂（m>1）必须登记且 m 落在区间内；
        #      ③ 未登记而 m≠1 / 标旗而 m=1 ⇒ 一律硬失败（防"校准档静默混入科学判读"）。
        if self.is_calibration_arm:
            lo, hi = CALIBRATION_M_RANGE
            # 校准臂的两类：**处理臂** m ∈ [1.2, 1.5]（R100 条件 5 区间）；
            # **配对基线** m == 1.0（增益档关闭的对照，仍须登记 ⇒ 同样被条件 5 拒收于科学判读）。
            # 只有这两类合法；其余（如 m=1.1 或 m>1.5）一律失败。
            assert self.gain_multiplier == 1.0 or lo <= self.gain_multiplier <= hi, (
                f"C5 v2 失败：校准臂 gain_multiplier({self.gain_multiplier}) 只允许 "
                f"1.0（配对基线）或落在 [{lo}, {hi}]（处理臂）⇒ 其余一律失败"
            )
        else:
            assert self.gain_multiplier == 1.0, (
                f"C5 v2 失败：gain_multiplier({self.gain_multiplier}) ≠ 1.0 但**未登记为校准臂** "
                f"⇒ 硬失败（R100 条件 5；若确为校准档，须显式设 is_calibration_arm=True）"
            )
        # ---- C5 规格自洽（2026-09-15 立；事故原型 = D-24 的 G-A 不过） ----
        # V-1 曾把 donation 定为 0.05 而 SIGNAL_COST 是 0.1 ⇒ 结构性上限 = 0.05/0.1 = 0.5 < 1.0
        # ⇒ "保本"语义**在数学上不可达**，oracle 必然检出不了阳性（G-A 必不过）。
        # C4 只问"开关有没有生效"，本断言问的是"生效的值彼此自不自洽" ⇒ 这是 C5 的第一条落地。
        if self.enabled:
            assert self.donation >= SIGNAL_COST, (
                f"C5 规格自洽失败：oracle 已启用，但 donation({self.donation}) < "
                f"SIGNAL_COST({SIGNAL_COST}) ⇒ return_ratio 结构性上限 = "
                f"{self.donation / SIGNAL_COST:.4f} < 1.0 ⇒ \"保本\"语义不可达"
                f"（D-24 G-A 不过的根因）。请反解 donation ≥ SIGNAL_COST/转化率；"
                f"若确要研究补偿不足区制，请显式设 allow_non_breakeven=True。"
            )


@dataclass
class CorpseWoundConfig:
    """尸体—食腐 + 血条—受伤（S1 骨架；设计稿 §三参数表）。

    🔴 **S1 阶段只建字段 + 读回，机制一律不接线**（开关存在但不动行为，
    默认全关 = 旧行为逐位一致，C7 digest 钉死）。S2/S3 才接线。
    🔴 全部须可 CLI 调 + `switches` 读回（C4）—— 参数表见
    `docs/设计文档/设计-尸体食腐与血条恐惧-云端实施-20260922.md` §三。
    """

    # ---- 尸体—食腐通道（R156 定稿；fish 01:2x 构想）----
    corpse_enabled: bool = False          # 总开关（默认关 = 旧行为；S2 接线）
    corpse_energy_frac: float = 0.9       # 死亡时剩余能量 ×0.9 转入尸体格（留 10% 分解即失）
    corpse_decay_ticks: int = 600         # 尸体存续 tick（≈ 世代时间 ⇒ 脉冲可累积，Noy-Meir）
    corpse_to_plant_frac: float = 0.5     # 腐烂归还植物池比例（其余为分解损失）
    corpse_patch_boost: float = 0.5       # 腐烂处资源 +50%（持续 2000 tick）
    # 🔴 cap 由 3 → 30 → **200**（R161 裁定：语义 = **单格尸体能量上限**，不是"尸具数"）。
    #   猎物尸体能量 = 剩余能量 × 0.9 ≈ **126–198** ⇒ cap 必须 ≥ 一具完整尸体，
    #   否则一进格就被钳掉（cap=3 时 corpse_total 2000 tick 仅 ~100 ⇒ 尸体通道价值被压没）。
    #   取 200 ≈ 一具满能量尸体（既容纳完整尸体、又保留"防极点/聚集处无界堆叠"的意图）。
    #   ⚠️ 三处默认值必须一致：本字段 / a4 CLI / `build()` 签名（R161 P1-1 的教训）。
    corpse_cap_per_cell: int = 200        # 单格尸体能量上限
    # 🔴 R165 0-1（2026-09-23 裁定）：**池的单位标记**。设计稿与注释口径都是
    #    「能量」（`corpse_energy_frac`=剩余能量×0.9、cap=单格**能量**上限 200）。
    #    单位进 `config_fingerprint` ⇒ **纪元可自证**（13.3 批无此字段且用了
    #    「池当质量」的旧口径 ⇒ 能量层读数不可跨纪比较）。
    corpse_pool_unit: str = "energy"      # 只允许 "energy"（换算点在食腐入胃处）
    scav_gate: float = 0.5                # 食腐 Hill 半效点（**不是硬门槛**）
    scav_s: float = 2.0                   # 食腐 Hill 陡度（scav_mult = g16^s/(g16^s+gate^s)）

    # ---- 血条—受伤（H1–H5；fish 01:48 构想）----
    wound_enabled: bool = False           # 总开关（默认关 = 旧行为；S2 接线）
    wound_base: float = 0.35              # 每次成功攻击扣血条 Δ（×(0.5+0.5×攻击性)）
    wound_heal_rate: float = 0.001        # 每 tick 恢复（上限 1.0；完全愈合 1000 tick）
    wound_heal_energy_cost: float = 0.05  # 愈合耗能 / tick（不免费）

    # ---- 争夺食物战（H3；S3 接线）----
    contest_enabled: bool = False         # 总开关（默认关 = 旧行为；S3 接线）
    # 🔴 holder_adv 由 0.3 → 1.2（2026-09-22 自行优化，fish 授权"23 自行优化"）：
    #   原值 0.3 被 RHP 中 `(0.3+g16)` 的 g16 线性放大淹没（g16 差 0.3 ⇒ RHP ~2×，
    #   0.3 只给 1.3× ⇒ 持有者胜率实测 ~0.41，远低于判据④ >55%）；且原实现
    #   "撤退分支不计胜负"导致读数系统性低估（已修复，loser==j 记 holder 胜）。
    #   1.2 时**正式配置**（默认 max_count=5000）多 seed 实测持有者胜率 0.66–0.85
    #   （判据④ >55% 达标，且保留挑战者 ~30% 胜率 ⇒ 争夺战不失去意义）。
    holder_adv: float = 1.2               # 持有者优势（Parker 1974）
    escalation_gap: float = 0.25          # 不升级的 RHP 差阈值（只有接近才升级）
    contest_cost_energy: float = 0.5      # 驱逐战的代价（防"免费赶人"）

    # ---- 恐惧/激进项（S3；方向相反，各自开关）----
    w_fear_health: float = 0.5            # 血条恐惧项权重（低血条 ⇒ 更恐惧；能力导向）
    # 🔴 13.4 波 3（T4）：血条恐惧**带门槛连续**（fish 01:20 批准）—— `1−health < 0.3`
    #    不触发（受轻伤不恐惧，重伤才怕）。
    wound_fear_threshold: float = 0.3     # 血条恐惧触发门槛（1−health ≥ 此值才生效）
    need_aggression_k: float = 0.5        # 饥饿激进项强度（固定 0.5；D 臂设 0 = 关"饥饿更激进"）

    def __post_init__(self) -> None:
        assert 0.0 <= self.corpse_energy_frac <= 1.0, "corpse_energy_frac ∈ [0,1]"
        assert self.corpse_decay_ticks >= 1, "corpse_decay_ticks ≥ 1"
        assert 0.0 <= self.corpse_to_plant_frac <= 1.0, "corpse_to_plant_frac ∈ [0,1]"
        assert self.corpse_patch_boost >= 0, "corpse_patch_boost 非负"
        assert self.corpse_cap_per_cell >= 1, "corpse_cap_per_cell ≥ 1"
        assert 0.0 <= self.scav_gate <= 1.0, "scav_gate ∈ [0,1]（半效点，非门槛）"
        assert self.scav_s > 0, "scav_s > 0"
        assert 0.0 <= self.wound_base <= 1.0, "wound_base ∈ [0,1]"
        assert 0.0 <= self.wound_heal_rate <= 1.0, "wound_heal_rate ∈ [0,1]"
        assert self.wound_heal_energy_cost >= 0, "wound_heal_energy_cost 非负"
        assert self.holder_adv >= 0, "holder_adv 非负（持有者优势）"
        assert self.escalation_gap >= 0, "escalation_gap 非负"
        assert self.contest_cost_energy >= 0, "contest_cost_energy 非负"
        assert self.w_fear_health >= 0, "w_fear_health 非负"
        assert self.need_aggression_k >= 0, "need_aggression_k 非负"


# 旧存档回退（corpse_wound）：未知键忽略、缺失键用默认值（同 oracle/_INFO_FIELDS 先例）。
_CORPSE_WOUND_FIELDS: frozenset = frozenset(f.name for f in fields(CorpseWoundConfig))


# 旧存档回退（oracle）：未知键忽略、缺失键用默认值。
_ORACLE_FIELDS: frozenset = frozenset(f.name for f in fields(OracleConfig))


# 旧存档回退（B1）：只接受当前 dataclass 已知字段——未知键忽略、缺失键用默认值。
# 新增 reputation_weight 后，旧快照的 info_structure 缺该键仍可正常加载。
_INFO_FIELDS: frozenset = frozenset(f.name for f in fields(InfoStructureConfig))


@dataclass
class FruitConfig:
    """果实-种子传播参数（L10a，植物-动物协同进化）。

    核心链路：植物蓄力→结果→动物吃果实→摄入种子→排泄→萌发新植物。
    默认 enabled=False（不改变现有行为），开启后新增果实场/蓄力/种子携带状态。
    """

    enabled: bool = False              # 总开关（False 时完全不执行 L10 步骤）
    plant_threshold: float = 0.5       # g19 >= 此值判定为植物（会结果）
    charge_rate: float = 0.1           # 每 tick 蓄力速率（× g8 光合产能）
    fruit_threshold: float = 10.0      # 蓄力达此值→结果释放
    fruit_ratio: float = 0.8           # 释放到果实场的能量比例（其余损耗）
    eat_rate: float = 0.05             # 动物每 tick 吃果实比例（× 果实能量）
    digest_ratio: float = 0.7          # 吃果实的能量消化率
    seed_intake_prob: float = 0.3      # 吃果实时摄入种子的概率
    excretion_prob: float = 0.1         # 携带种子每 tick 排泄概率
    germination_prob: float = 0.5       # 排泄后种子萌发概率
    seed_energy: float = 5.0            # 萌发新植物的初始能量
    max_seed_carried: int = 5           # 单个体最大携带种子数

    def __post_init__(self) -> None:
        assert 0.0 <= self.plant_threshold <= 1.0
        assert self.charge_rate >= 0
        assert self.fruit_threshold > 0
        assert 0.0 <= self.fruit_ratio <= 1.0
        assert 0.0 <= self.eat_rate <= 1.0
        assert 0.0 <= self.digest_ratio <= 1.0
        assert 0.0 <= self.seed_intake_prob <= 1.0
        assert 0.0 <= self.excretion_prob <= 1.0
        assert 0.0 <= self.germination_prob <= 1.0
        assert self.seed_energy > 0
        assert self.max_seed_carried >= 0


# ---------------------------------------------------------------- 信号字母表（R113 / R121）
# 🔴 2026-09-18 立（R113 所有者 09-17 22:30 裁定；R121 09-18 核阅批准实现形态）：
#    档位（**共用同一可逆开关**）：
#      · `"16"` = 现状 4 位（能量2 + 食物1 + 邻居1），state 0–15、码域 **1–15**（0 保留=无信号）
#      · `"4"`  = R113 基线 2 位（**仅能量**）：`state = e_bin`、`code = e_bin + 1` ⇒ 码域 **1–4**
#                顺手消灭 `state=0` 的"隐形发射 + 清除他人标记"后果（设计稿 §2.4）
#      · `"8"`  = B③ 载体 3 位（能量2 + 记忆1）：`code = e_bin*2 + mem_bit + 1` ⇒ 码域 **1–8**
#                **R121 §五.2 已批准，但按设计稿 §六 排在下一步：本版**未实施**（fail-loud）
#    ⚠️ `signal_alphabet` 进 `to_dict()`/`config_fingerprint()` ⇒ **跨档续跑硬报错**。
#       这是**特性不是缺陷**：防"16 码快照被 4 码批静默续跑"这类混口径事故（设计稿 §3.1）。
#    ⚠️ **纪元纪律**（内评 09-17 §三.3）：`"4"` 批次与 `"16"` 批次（C1a/C1b/C2）的
#       `ratio` / `codebook_conv` **不可直接比较**；报告必须标注 `signal_alphabet`。
#    ⚠️ RNG 契约（设计稿 §3.3）：切档**不改变任何 RNG 抽取值与形状** ⇒ 同 seed 下轨迹差异
#       只能来自码语义，不来自随机流错位（便于 C7 逐位对拍）。
SIGNAL_ALPHABET_STATES: dict[str, int] = {"16": 16, "4": 4, "8": 8}
SIGNAL_ALPHABET_CODE_MAX: dict[str, int] = {"16": 15, "4": 4, "8": 8}
# 已落地实现的档位；其余在 config 构造/引擎初始化时**硬失败**（不静默降级，教训 2）
# 已落地实现的档位；其余在 config 构造/引擎初始化时**硬失败**（不静默降级，教训 2）
#   `"8"`（B③ 记忆位，R123 2026-09-18 实施）：`code = e_bin*2 + mem_bit + 1` ⇒ 码域 **1–8**
SIGNAL_ALPHABET_IMPLEMENTED: tuple[str, ...] = ("16", "4", "8")


@dataclass
class SubposConfig:
    """亚格连续坐标（13.4 波 1）—— 把"每 tick 必走满 1 格"细化为 0.25 格粒度。

    为什么
    ------
    旧行为：移动决策从 8 邻格选 1，**位移恒 = 1 格**，且候选不含自身 ⇒ **没有"原地不动"**
    ⇒ 世界体感小（个体每 tick 都在跳整格，既不能慢行、也不能停下来进食或观察）。

    本配置把位移细化为 `0 / 0.25 / … / speed_max` 格（`subdiv=4`），并把语义从
    "以概率**移动**"改为"**默认走、以概率停**"（`stay_prob_eff` 决定停，见设计稿 §十.4）。

    🔴 三条不变式（违反即返工）
    --------------------------
    * **I1** `enabled=False` ⇒ **不进任何新代码路径** ⇒ C7 逐位等价（基线 `(573985, 8171.692943)`）
    * **I2** `_flat` 恒与 `(sub_r, sub_c)` 一致；派生**只走 `world.rc_to_flat`**
      （极点坍缩/经度环绕**只有一处定义**，防两处规则漂移）
    * **I3** **不新增每 tick 随机抽取** —— 速度由已算好的 `mob_eff` **确定性**推导。
      这是 L2 的核心契约：`rand_choice` 被 `%100` / `//100` **位域拆分**为
      "冲刺闸 + 盲选方向"两用，动了随机流形状就作废 L2 的全部对拍价值。

    🔴 H3 互斥（fail-loud）
    -----------------------
    `enabled=True` 且 `SimulationConfig.l2_dash=True` ⇒ **构造期直接抛**
    （同 `l2_dash` / `corpse_enabled` 的先例，禁止静默换路径）：
    两者都是"**走多远**"的乘子（L2 只有 1/2 两档，subpos 是它的一般化），同开会
    语义冲突，且 `dash_frac` 与 `steps` 两个读数互相污染 ⇒ 归因不干净。

    ⚠️ 另一个必须记住的构造后果（设计稿 §二）
    -----------------------------------------
    老逻辑移动者**位移恒 ≥ 1 格 ⇒ 跨格 100%**；subpos 下位移可以 < 1 格 ⇒ **不换格**。
    而取食/信号/尸体/资源**全部按整数格**（I2）⇒ 位移不足 1 格**在格层面等于没动**。
    因此判据必须用 **`mean_flat_moves_per_tick`（真正跨格的比例）**，**不是 `mean_steps`**
    —— 后者会把"原地挪小步"误读成"在移动"。见 §2.2 的校准（`speed_gain` 初值 4.0）。
    """

    enabled: bool = False            # 总开关（默认关 = 旧行为**逐位一致**；S1 骨架不接线机制）
    subdiv: int = 4                  # 每格 4×4 = 16 个亚位置 ⇒ 最小步长 0.25 格
    speed_gain: float = 4.0          # speed = clamp(mob_eff × gain, 0, speed_max)
    #                                  初值 4.0：`mob_eff` 实测均值 ≈0.27 ⇒ 0.27×4 ≈ 1.08 格，
    #                                  贴近历史 `1 + dash_frac`（1.12–1.24）⇒ 跨格频率不塌（§2.2）
    speed_max: float = 2.0           # 速度上限（格/tick）；=2 使 subpos **包含** L2 的冲刺语义
    min_energy_frac: float = 0.05    # 移动能量门槛（**纯能量阈值**；0.05 × max_energy = 15）
    #  🔴 初值是 0.2（沿用 L2 的 `dash_min_energy_frac`），**实测证明用错了口径**：
    #     L2 的门槛管的是"**冲刺**"（额外行为，门槛高合理）；而 subpos 管的是"**移动**"本身
    #     （基本行为）。0.2 × 300 = **60**，恰好等于 `initial_energy`(60)，而维持消耗 0.6/tick
    #     ⇒ 几十 tick 后**全体低于门槛** ⇒ 实测 **97.87% 的移动者走 0 步**（自检 `_run_slow_n`）。
    #  ⇒ 降到 0.05（=15 能量）：只拦"真的要饿死"的个体，不拦正常觅食移动。
    lat_floor: float = 0.3           # 极区移速折减下限：
    #                                  speed_cap = speed_max × (lat_floor + (1−lat_floor)·cos 纬度)
    #                                  ⇒ 赤道 100%、极区 30%（fish 00:30"极区移速更慢"）
    # ---- 停留概率（语义反转：**默认走、以概率停**；由状态/信息驱动）----
    stay_base: float = 0.0           # 基座停留概率
    stay_food_k: float = 0.0         # 本格还有余粮 ⇒ 停着吃
    stay_signal_k: float = 0.0       # 收到信号 ⇒ 停（去看/听）
    stay_fear_k: float = 0.0         # 邻域有威胁 ⇒ 停（隐蔽）
    stay_max: float = 0.8            # 🔴 停留概率**上限** = 硬性**反退化闸**
    #                                  （任何个体至少 20% 概率移动；防"一群不动的生物"，R171 §十）

    def __post_init__(self) -> None:
        assert self.subdiv >= 1, "subdiv 至少 1"
        assert self.speed_gain >= 0.0, "speed_gain 非负"
        assert self.speed_max > 0.0, "speed_max 为正"
        assert 0.0 <= self.min_energy_frac <= 1.0, "min_energy_frac 是比例（0~1）"
        assert 0.0 <= self.lat_floor <= 1.0, "lat_floor 在 0~1"
        for _nm in ("stay_base", "stay_food_k", "stay_signal_k", "stay_fear_k"):
            assert 0.0 <= getattr(self, _nm) <= 1.0, f"{_nm} 在 0~1"
        assert 0.0 <= self.stay_max < 1.0, "stay_max 必须 < 1（反退化闸：不许全停）"


@dataclass
class HungerModConfig:
    """饥饿调制（HM，R247；实施规格 `docs/设计文档/实施规格-移动决策三件-20260928.md` §一）。

    机制（三处同式；默认关 ⇒ 与本机制加入前**逐位等价**）
    ----------------------------------------------------
    现状：移动打分公式里**没有能量项** ⇒ 饿与饱的个体行为完全相同。本配置把
    "自己有多饿"接进移动决策的两条通道（③ 离家倾向归 S3，**本批不做**）：

        hunger = clip(1 − energy / max_energy, 0, 1)          # 0=饱 / 1=快饿死
        h_norm = (hunger − h_mid) / max(1e-9, 1 − h_mid)      # 中性点 h_mid 归一

        ① 走/停：move_prob_eff = clip(move_prob × (1 + alpha × h_norm), 0, 1)
                  （subpos 路径作用于"移动概率" p_move = 1 − stay_eff）
        ② 感知：  perc_eff = perc × (1 + beta × h_norm)
                  （只调制 `perc` 赋值行 ⇒ 食物/信号/记忆/解读项同乘、
                   `soc × 密度` 项不受影响 —— "饿 ⇒ 更专注找吃的/信号，
                   社交项相对被压"）

    🔴 三处同式（漏一处 = 双路径分岔）：Python 参考循环 / Rust `movement.rs` /
      向量化快路径 `_move_decide_batch`。R247 已三处全接线 ⇒ 快路径**不**进
      `_batch_ok` 排除表（HM 在快路径内逐行调制 `perc`，见调用点注释）。

    🔴 参数三级标注（AGENT.md §12.10）：`alpha` / `beta` / `h_mid` / `stay_gain`
       = **KNOB**（可独立调）；`enabled` = 总开关（默认关 = 回滚点）。
    ⚠️ 不新增状态数组（只用已有 `self._energy`）⇒ 不触 `__slots__`/快照那套纪律。
    ⚠️ 实施纪律（规格 §三）：**不得在饱和世界单独测**（没有值得去的地方 ⇒ "更愿意走"
       只会变成"更快的随机游走"）；须 S2 过门后同批或紧随。
    """

    enabled: bool = False      # 🔴 默认关 ⇒ 整块跳过 ⇒ 基线 digest 不动（逐位等价）
    alpha: float = 0.5         # KNOB｜① 走/停调制强度
    beta: float = 0.5          # KNOB｜② 感知权重调制强度
    h_mid: float = 0.5         # KNOB｜中性饥饿度（此点上下才产生差异）
    stay_gain: float = 0.0     # KNOB｜（预留）subpos 路径驻留调制 —— 本批**不消费**

    def __post_init__(self) -> None:
        assert isinstance(self.enabled, bool), "hunger_mod.enabled 必须是布尔值"
        assert 0.0 <= self.alpha <= 2.0, "hunger_mod.alpha ∈ [0, 2]"
        assert 0.0 <= self.beta <= 2.0, "hunger_mod.beta ∈ [0, 2]"
        assert 0.0 <= self.h_mid <= 1.0, "hunger_mod.h_mid ∈ [0, 1]"
        assert 0.0 <= self.stay_gain <= 1.0, "hunger_mod.stay_gain ∈ [0, 1]（预留字段）"


@dataclass
class ActionSelectionConfig:
    """模式仲裁（ASM，R238/R239；实施规格 `docs/设计文档/实施规格-移动决策三件-20260928.md` §三）。

    机制（三处同式；默认 `"fusion"` ⇒ 与本机制加入前**逐位等价**）
    ------------------------------------------------------------
    现状 = **加权融合**：`score = perc·(0.5·food_ratio + sig·w) + soc·density + …`，
    所有动机**同时**进入一个标量、彼此线性抵消（"食物近、风险更近"时可能既不敢去也不逃）。
    仲裁档改为**先选模式、再在模式内行动**：

        sal_feed    = w_feed·Ŝ_food + w_hunger·hunger     # Ŝ = 气味通道归一值（[0,1]）
        sal_flee    = w_flee·Ŝ_risk                        # ⚠️ 设计稿的 (1+g_fear) 乘子留批 B（无空闲基因位）
        sal_join    = w_join·Ŝ_kin·(2·g13 − 1)             # g13 = SOCIABILITY（既有基因）
        sal_explore = base_explore
        ② 若 tick − _mode_tick < hold_ticks            ⇒ 保持 _mode（最小锁定，防抖动）
           否则若 max(sal) > sal[_mode] + hyst        ⇒ _mode = argmax(sal); _mode_tick = tick
        ③ 模式内选邻居（三处同式）：
           feed → argmax Ŝ_food(nb)｜flee → argmin Ŝ_risk(nb)
           join → argmax (Ŝ_kin(nb) + 密度项)｜explore → 随机（平局分支 ⇒ rand_choice mod n）

    🔴 首批红线（规格 §3.3/§3.5）
    ----------------------------
    * **信号不接模式**（否则"一喊全跑"抹平信息不对称）—— 留 S3/C3 批；
    * **HM 的 ②（感知权重调制）在仲裁档下关闭**（由 `w_hunger` 承担，避免双重调制）；
      HM ①（走停调制）照常生效（modulation 发生在仲裁**之前**的走停闸）；
    * **不新增基因位**（6 个预留位已耗尽 ⇒ 扩位 = 更大纪元；设计稿"批 B 走基因位"）：
      `w_*` / `hyst` / `hold_ticks` 本批全是**固定参数**；
    * **不做"优先级靠人定"的排序**：模式间**只比显著度**，不写死模式优先级。

    🔴 前置条件（引擎构造期 fail-loud，`sphere_engine.__init__` 的 ASM 守卫块）
    ------------------------------------------------------------------------
    * 仲裁需要 `Ŝ_food / Ŝ_risk / Ŝ_kin` ⇒ **`smell.channels ⊇ {food, risk, kin}`**
      （缺则炸，不许静默降级成融合）；
    * 仲裁档下与下列机制**互斥**（它们都改"融合 score"或选格语义，同开会静默吞掉其一）：
      softmax / 感知噪声 / 记忆梯度 orientation / 记忆 v2 / L1 寻食-避害 / L2 dash /
      span2 / 占位上限 / 迁徙 / ARS / 声誉权重 / 伤血恐惧 / `smell.use_in_move`。

    ⚠️ 新增两状态数组（`_mode` / `_mode_tick`）⇒ 四处登记 + 快照（规格 §3.2）。
    ⚠️ 实施纪律（规格 §〇）：HM / 气味消费端 / ASM **一次只开一件**（都改移动决策）。

    🔴 参数三级标注（AGENT.md §12.10）：`mode` = 总开关（默认 fusion = 回滚点）；
    `base_explore / hyst / hold_ticks / w_feed / w_hunger / w_flee / w_join` = **KNOB**（可独立调）。
    """

    mode: str = "fusion"        # 🔴 默认 fusion = 现状（回滚点）；"arbitration" = 仲裁档
    base_explore: float = 0.2   # KNOB｜探索模式保底显著度（= 探索的时间占比下限倾向）
    hyst: float = 0.15          # KNOB｜迟滞（进入阈值 − 退出阈值；越大越迟钝）
    hold_ticks: int = 20        # KNOB｜最小锁定（防 dithering，规格 §3.4-4 头号风险）
    w_feed: float = 1.0         # KNOB｜sal_feed 的气味项权重
    w_hunger: float = 0.5       # KNOB｜sal_feed 的饥饿项权重（HM ② 的替代通道）
    w_flee: float = 1.0         # KNOB｜sal_flee 权重
    w_join: float = 0.5         # KNOB｜sal_join 权重

    _KNOWN_MODES = ("fusion", "arbitration")

    def __post_init__(self) -> None:
        assert isinstance(self.mode, str) and self.mode in self._KNOWN_MODES, (
            f"action_selection.mode={self.mode!r} 未实现（只实现 {self._KNOWN_MODES}）"
            f" —— 拼错的模式**必须炸**，不许静默退回 fusion")
        assert 0.0 <= self.base_explore <= 10.0, "base_explore ∈ [0, 10]"
        assert 0.0 <= self.hyst <= 10.0, "hyst ∈ [0, 10]（0 = 无迟滞，仅供变异测试）"
        assert isinstance(self.hold_ticks, int) and self.hold_ticks >= 0, (
            "hold_ticks 必须是 ≥0 的整数（0 = 无锁定，仅供变异测试）")
        for nm in ("w_feed", "w_hunger", "w_flee", "w_join"):
            v = float(getattr(self, nm))
            # `v == v` 挡 NaN（config.py 不依赖 numpy ⇒ 不用 isnan）
            assert v == v and 0.0 <= v <= 10.0, (
                f"{nm}={v} 非法（需有限且 ∈ [0,10]；符号语义由模式内 argmax/argmin 决定，"
                f"不许负权重制造二义）")


@dataclass
class MigrationConfig:
    """日历—罗盘式定向迁徙（13.8；fish 2026-09-25 批准 / 设计稿 `docs/设计文档/设计-日历罗盘式定向迁徙-20260925.md`）。

    为什么
    ------
    13.7 已把季节接上（`band_res` 摆幅 1.4–3° → 21–37°），但 **个体仍不会迁徙**
    （R197 实测：掉头率 0.525 ≈ 0.5；且 R196 臂个体位移 6.16° **反而小于**
    无季节基线 7.49°）。根因已定位到信噪比：

        食物带移动 0.0015°/tick  vs  个体随机运动 0.025–0.030°/tick  ⇒ **慢 17–20 倍**

    ⇒ 「让个体感知食物梯度」这条路**在信噪比上就被堵死**。本机制改走真实鸟类
    用的两件东西：**一个日历（本地日照时长）+ 一个罗盘（自己的绝对纬度）**，
    再加一个自由基因位 `g23`，让"要不要用、用多强"由选择压自己决定。

    机制（移动打分加一项，逐候选）
    ------------------------------
        A_i(t)   = photoperiod(φ_self, t) − 0.5      # 本地日照异常见量，>0 = 本地夏季
        Δ|φ|(n)  = |φ_n| − |φ_self|                   # 该候选"离极地更近/更远"
        score(n) += enabled · gain · g23_i · A_i(t) · Δ|φ|(n)

    * **本地夏季（A>0）⇒ 奖励往极地走；本地冬季（A<0）⇒ 奖励往赤道走。**
    * 🔴 用 `|φ|` 而非带符号 `φ` ⇒ **南北半球自动都对，无需半球分支**。

    为什么这不是"把答案写进模型"
    ----------------------------
    只加**可观测维度**（日照时长 / 绝对纬度）+ **一条可被使用的通路**（`g23` 缩放）；
    `g23` **完全自由**（可为 0／任意正负），**不给任何额外能量/繁殖奖励**。
    ⇒ 若迁徙无益，选择压会把 `g23` 压向 0 ⇒ **机制自己会证伪自己**。

    🔴 三条硬约束（与 `SubposConfig` / `L2` 同一套纪律）
    ---------------------------------------------------
    * **I1** `enabled=False`（默认）⇒ **整块跳过** ⇒ C7 逐位等价
      （基线 `(574887, 11266.746993)`）
    * **I2** 新机制**只在 Python 路径**实现 ⇒ 与 Rust 互斥（M1 fail-loud，§14.7）
    * **I3** **无季节 ⇒ A ≡ 0 ⇒ 迁移项恒 0 ⇒ 开关形同虚设** ⇒ M2 fail-loud
      （否则会得到"开了迁徙但没反应"的**假阴性**）

    🔴 H3 互斥（fail-loud，两条缺一不可；均**不许 warning**）
    --------------------------------------------------------
    * **M1** `enabled=True` ∧ `SimulationConfig.use_sim_core=True`
      ⇒ 构造期 `NotImplementedError`
    * **M2** `enabled=True` ∧ **无季节**（`tilt_rad=0` 或 `season_period<=1`）
      ⇒ 构造期 `ValueError`
    """

    enabled: bool = False      # 默认关 ⇒ 整块跳过 ⇒ 与本机制加入前**逐位等价**
    gain: float = 50.0         # 全局权重 w_mig（**实验旋钮，不是基因**；S2 的 P6 量级门定它）
    min_abs_anomaly: float = 0.0   # |A| 低于此值不计（抑制春秋分附近的噪声翻转）；
                                   #   0.0 = 不设门槛（默认，最小实现）

    def __post_init__(self) -> None:
        """字段断言（与全仓配置类同规格：构造即校验，fail-loud）。"""
        assert isinstance(self.enabled, bool), "migration.enabled 必须是布尔值"
        assert self.gain >= 0.0, "migration.gain 必须非负"
        assert 0.0 <= self.min_abs_anomaly <= 0.5, (
            "migration.min_abs_anomaly 是 |A| 门槛（A∈[−0.5,0.5]）⇒ 须在 0~0.5"
        )


@dataclass
class ArsConfig:
    """ARS 双模式觅食（14.9；fish 2026-09-25 批准；设计稿 §9–§11 v4）。

    机制（两腿，默认关 ⇒ 逐位等价）
    -------------------------------
    * **赶路腿（extensive）**：个体按 `Gene.PERSISTENCE`(g20) 沿上一步方向继续走
      （`score += gain · pers · cos(heading→候选)`；cos 对反向天然 = −1 ⇒ **不回头软惩罚**）；
      连走 `giveup` tick 没咬到 ⇒ **随机重选方向**（排除由 cos 隐式完成）。
    * **期待腿**：`_out_taken`（本 tick 实际咬到量）更新**快/慢两个 EMA**；
      **快 < θ·慢 ⇒ 切赶路**（= 个体自己的"最近吃得不如之前好"）；
      **咬到 ⇒ 切回驻留**。这就是 MVT 的"跟自己最近的经验比"，不依赖全局平均。
    * **油箱对称耦合（D3，fish 09-25）**：`stomach_cap ×= (1+κ·pers)` **且**
      `base_metabolism ×= (1+κ·pers)` ⇒ 油箱大的维持也贵 ⇒ **不是白来的补贴**，
      净效果由环境决定（食物稀 ⇒ 油箱大的赢）。

    🔴 三条硬约束
    -------------
    * **I1** `enabled=False` ⇒ 不进任何新代码路径 ⇒ C7 逐位等价
    * **I2** 只在 Python 路径 ⇒ `enabled ∧ use_sim_core` 构造期硬报错（H3-A1）
    * **I3** 极区豁免：`|lat| > lat_exempt_deg` 的个体**不加惯性项**（极点邻居 = 整行 120 个，
      0–7 方向编码不成立；实测保持东西向 120 步绕回原点 ⇒ 持续长度必须有限）
    """

    enabled: bool = False          # 默认关 ⇒ 整块跳过 ⇒ C7 逐位等价
    gain: float = 1.0              # 惯性权重 w_pers（实验旋钮）
    theta: float = 0.5             # 快 < θ×慢 ⇒ 切赶路（θ 越高越容易"失望"）
    kappa: float = 0.0             # 油箱对称耦合系数（0=关；≤0.5 上限）
    giveup: int = 20               # 赶路模式下连走多少 tick 没咬到就重选方向（≈1–3×自由程）
    fast_tau: float = 50.0         # 快平均时间常数（tick）
    slow_tau: float = 500.0        # 慢平均时间常数（tick）；必须 > fast_tau
    lat_exempt_deg: float = 85.0   # 极区豁免阈值（度）

    def __post_init__(self) -> None:
        assert isinstance(self.enabled, bool), "ars.enabled 必须是布尔值"
        assert self.gain >= 0.0, "ars.gain 非负"
        assert 0.0 < self.theta <= 2.0, "ars.theta 须在 (0, 2]"
        assert 0.0 <= self.kappa <= 0.5, "ars.kappa ∈ [0, 0.5]（fish D3：总格数不超太多）"
        assert int(self.giveup) >= 1, "ars.giveup ≥ 1"
        assert self.fast_tau > 1.0 and self.slow_tau > self.fast_tau, (
            "ars.slow_tau 必须大于 fast_tau（否则'快/慢'无意义）")
        assert 0.0 <= self.lat_exempt_deg <= 90.0, "ars.lat_exempt_deg ∈ [0, 90]"


@dataclass
class ResourceDynamicsConfig:
    """斑块"休耕—死亡—轮作"（13.4 波 2；fish 01:20 构想 / R176 §12）。

    为什么
    ------
    现状再生**只由温度驱动** ⇒ 一格今天被吃光，下一 tick 照样满速恢复 ⇒
    **取食压力在资源侧没有任何反馈**。本机制给出那条缺的负反馈：

        取食压力 ↑ ⇒ 斑块死亡 ↑ ⇒ 总产能 ↓ ⇒ 种群 ↓ ⇒ 取食压力 ↓ ⇒ 死格重生 ⇒ 产能恢复

    ▶ 实现落在**新文件** `world/resource_dynamics.py`（纯逻辑、零引擎依赖）⇒ 本配置只是它的参数契约。
    ▶ `enabled=False`（默认）⇒ 模块全部方法退化为 no-op ⇒ 引擎无条件调用也**不进新代码路径**
      （与 `SubposConfig` / `L2` 同一套"C7 逐位等价"纪律）。

    🔴 两个字段偏离设计稿 §12.5 字面（依据 = `tools/wave2_patch_probe.py` 的零 RNG 实测）
    ---------------------------------------------------------------------------------
    *(A) `kill_denom`：字面口径（`被吃量 / capacity`）是"纬度抽奖"* —— `capacity = capacity_per_area
    × 格面积`，格面积跨纬度差 **151 倍**（赤道斑块格 120.0 vs 极点斑块格 3.14），而每格每 tick
    最大取食量只有 `0.5 × 1.5 × 3 = 2.25` ⇒ 阈 0.7 时**只有 1/817 个斑块格**会被杀死，且全在
    极区行（0/1/59），**赤道行比值仅 0.074** ⇒ 该机制在生命区**永不触发**（≈ no-op）。
    ⇒ 默认改 `"regrowth"`（分母 = **当期再生量** = §2.5「格内吃>长」的同一个量）：
      斑块格 单人 0.75 / 3 人 **2.25**、背景格 单人 1.71 / 3 人 5.12 ⇒ **与纬度无关**，
      阈值读作「**同格几个人的取食量超过这格当期再生**」。字面口径仍可用 `"capacity"` 复核。

    *(B) `kill_patch_only`：只有斑块格会死（默认 True）* —— `kill_denom="regrowth"` 下背景格
    "单人取食/再生"就有 1.71 ⇒ 若允许背景格死，世界会被瞬间打散（大范围荒漠）；
    而设计意图本就是"**斑块**因过牧而死"。`rotate_same_row_only` 见模块 docstring (C)：
    **跨行交换会破坏 `Σcapacity` 守恒**（实测 +1.886e+03 = 相对 1%）⇒ 只在同行内搬加成。
    """

    enabled: bool = False            # 默认关 = 旧行为**逐位等价**（派生量全 no-op）
    rest_ticks: int = 60             # 休耕时长：该格 N tick 内再生 = 0（13.4 波2 修 v2：
                                     #   300→60，文献轮牧 30–60 天口径；配合 rest_threshold
                                     #   只在"吃够 30% 容量"才休耕 ⇒ 60 足够恢复）
    # 🔴 13.4 波2 修 v2（2026-09-23，fish 批准方案 A）：**累计损伤三态状态机**。
    #   旧版"intake>0 即休耕"被实测击穿（B–E 臂 resting 82–92%，一次被吃 = 停摆 300）。
    #   新口径 = **累计被吃量 / 容量**（damage ∈ [0,∞)）：
    #     damage ≥ rest_threshold   ⇒ 进入休耕（固定 rest_ticks，**被吃不刷新**）
    #     damage ≥ death_threshold  ⇒ 死亡（斑块加成搬走；与 kill_frac 极端密度通道并存）
    #   文献锚（轮牧 Take-Half-Leave-Half / USDA 摘叶梯度：50% 轻伤 / 70% 重伤 / 90% 近死）：
    #   rest 0.3 / death 0.8 对应"轻伤可恢复 / 重伤退化"，比行业 40–60% 保守。
    rest_threshold: float = 0.3     # 损伤 ≥ 30% 容量 ⇒ 休耕
    death_threshold: float = 0.8    # 损伤 ≥ 80% 容量 ⇒ 死亡
    damage_recovery: float = 0.5    # 休耕到期损伤乘此系数（部分恢复；文献：恢复期后损伤减半）
    # 🔴 0.7 → **10.0**（13.4 波 2A，T2；R178 裁定：`kill_mult` 初值 10 =
    #    "一 tick 吃掉 **10 倍当期再生** ⇒ 死"，**禁用裸 `regrowth_rate`**——否则纬度抽奖回归）。
    #   ⚠️ 收编件字段名保留 `kill_frac`（模块读它），语义 = 再生倍数阈值（不是 <1 的比例）。
    #   ⚠️ 波2 修 v2：判死主通道改 `death_threshold`（累计损伤）；`kill_frac` 保留为
    #   **极端密度瞬间死亡**的补充通道（单 tick intake/growth > 10 仍死），两通道共用死格处理。
    kill_frac: float = 10.0          # 被吃强度 > 此值（= kill_mult，当期再生倍数）⇒ 斑块死亡
    kill_denom: str = "regrowth"     # 🆕 "regrowth"（默认，纬度无关）| "capacity"（设计稿字面）
    dead_regen_ticks: int = 2000     # 死格**重入候选池**的等待（🔴 0 = 永不 ⇒ 硬拒绝：那是文献里的不可逆荒漠化）
    dead_cell_max_frac: float = 0.5  # 🔴 **反荒漠化闸**：死格占比超过它 ⇒ 强制加速重生
    kill_patch_only: bool = True     # 🆕 只有斑块格会死（防背景格大范围被打散）
    rotate_same_row_only: bool = True  # 🆕 只在**同行**（同面积）交换斑块加成 ⇒ Σcapacity 逐位守恒

    def __post_init__(self) -> None:
        assert self.rest_ticks >= 0, "rest_ticks 非负"
        assert self.kill_frac >= 0.0, "kill_frac 非负"
        assert self.kill_denom in ("regrowth", "capacity"), (
            "kill_denom 只允许 'regrowth' / 'capacity'（见类 docstring (A)）"
        )
        assert self.dead_regen_ticks > 0, (
            "dead_regen_ticks=0 会让斑块**永不再生**（不可逆荒漠化）—— 本设计要的是可逆的"
            " shifting mosaic；若确实要测不可逆，请另立开关并先在板上裁定"
        )
        assert 0.0 < self.dead_cell_max_frac <= 1.0, "dead_cell_max_frac ∈ (0,1]"
        # 波2 修 v2 断言：阈值有序 + 恢复系数 ∈ (0,1]
        assert 0.0 < self.rest_threshold < self.death_threshold, (
            "rest_threshold < death_threshold（先休耕后死亡）")
        assert 0.0 < self.damage_recovery <= 1.0, "damage_recovery ∈ (0,1]"


@dataclass
class SmellConfig:
    """气味场（多通道标量场）—— R240 T8 / 设计稿《气味场与分功能感知》§二、§8.2。

    为什么
    ------
    2D 随机游走的**首次命中**期望步数 ≈ d²（d = 斑块间距）：现状 d≈15 格 ⇒ ~225 步，
    而满能量续航只有 273 格 ⇒ **82% 的个体在撞上斑块前就饿死**（S1 实测 31–47% 斑块格
    从未被访问）。扩视觉要扩 15 倍（成本爆炸）；**气味场只用 1 格视野就能拿到 15 格外的信息**
    （气味是**环境量**，个体仍"只读自己格"⇒ 不违反"感知 1 格"）。

    模型（每个通道一个标量场，**每 k tick 更新一次**）
    -------------------------------------------------
        S ← decay^k × (S + 注入) + D × lap_粗↑(S)

    🔴 三条内置优化（设计稿 §8.2，**红线，不可省**）
    -----------------------------------------------
    ① **降采样长程**：扩散（五点拉普拉斯）在 **1/s 粗网格**上算，再**双线性插值**回全分辨率
       ⇒ 成本 ÷ s²（默认 s=8 ⇒ ÷64）
    ② **降低更新频率**：每 **k tick** 一次（默认 k=4）⇒ 成本 ÷k
    ③ **源注入稀疏**：只碰"有源"格（食物格 2.8% / 个体 0.65%），与扩散**解耦**
    🔴 **禁止"每 tick 全场稠密卷积"** —— 那会把稀疏化刚拿到的 N=0 地板（13.63→0.13 ms/tick）
       一口吃回去。`world/smell_field.py::update()` 是唯一全场路径，引擎每 k tick 只调一次。

    ▶ 实现落在**新文件** `world/smell_field.py`（纯逻辑、零引擎依赖）⇒ 本配置只是它的参数契约。
    ▶ **默认全关**：`channels == ()` ⇒ 引擎**不构造**该模块（`eng.smell is None`）⇒
      零成本 + 旧行为**逐位不变**（C7 回滚点）。
    ▶ 与 `sparse_fields` 的关系（设计稿 §8.4 取方案 (a)）：本模块**自带**稀疏/降采样路径
      ⇒ **正交、独立开关**（不依赖 `sparse_fields`，也不受 rd 开档退回全场的影响）。
    ▶ 通道（R238 裁定：四通道全上、分两批启用）：`food` / `prey` / `risk` / `kin`。
      ⚠️ **消费端（score 加权）不在本配置**：权重需扩 8 个基因位 = **纪元级**（R238 §2）
      ⇒ 属批 A/B 排期；本模块只提供**接口 + 读 API**（`SmellField.at()`）。

    🔴 与设计稿的两处口径说明（防"数字没出处"）
    -------------------------------------------
    * 设计稿写 `smell_channels`；本仓配置按**分组惯例**落为 `SmellConfig.channels`（同 `signals.*`）。
    * `downsample` 只在 `rows/cols` 都能整除时**精确**；否则构造期自动降到"≤s 的最大公约数"
      （`SmellField.probe()["downsample_eff"]` 可读回，且与请求值不同时会告警 —— 不静默）。
    """

    channels: tuple[str, ...] = ()   # 启用通道（**空 = 全关**；子集 of ("food","prey","risk","kin")）
    update_every: int = 4            # 每 k tick 更新一次（§8.2-②；线索延迟 k tick）
    downsample: int = 8              # 粗网格因子 s（§8.2-①；实际取 min(s, gcd(rows,cols))）
    decay: float = 0.88              # **每 tick** 衰减；每次更新实际乘 `decay**update_every`
    diffuse: float = 0.12            # 每次更新的扩散权重（粗网格五点拉普拉斯 × 本系数）
    inject_food: float = 1.0         # ① 食物通道注入权重（注入量 = 食物量/最大容量 ∈ [0,1]）
    inject_prey: float = 1.0         # ② 猎物通道（源 = 活体个体）
    inject_risk: float = 1.0         # ③ 风险通道（源 = g16 ≥ risk_g16_threshold 的个体）
    inject_kin: float = 1.0          # ④ 同类通道（源 = 活体个体；与 prey 同源、读端不同用途）
    risk_g16_threshold: float = 0.6  # ③ 的源定义：`g16`（AGGRESSION）≥ 此值 ⇒ 视为捕食者

    # ---- R244 §二：气味场**消费端**（把通道值接进移动打分）----------------------
    #  ✅ 已完成；🟢 R244 立项（实施规格-移动决策三件-20260928.md §二）
    #
    #  score += perc × Σ_ch  w_ch · Ŝ_ch(c)        （c = **候选邻居格**；仍是"感知 1 格"）
    #  Ŝ_ch(c) = clip( S_ch(c) / S_max_ch , 0, 1 )，S_max_ch = inject_max_ch / (1 − decay^k)
    #            （**解析上界**：本模块递推 `S ← decay^k·S + 注入` 的不动点 ⇒ 构造期可算）
    #
    #  🔴 两处口径（写死，防"数字没出处"）
    #  * **权重自带符号**：设计稿 §2.2 写 `− w_risk·Ŝ_risk`，§2.1 又给 `w_risk=-0.5`（"负权重=回避"）
    #    ⇒ 二者**互相矛盾**（双重取负会变成吸引）。本实现取"**权重自带符号**"（`Σ w_ch·Ŝ_ch`），
    #    与 §2.1 的默认值自洽；若口径定为 `−w_risk` 形式 ⇒ 把默认改回 `+0.5` 即可（一行）。
    #  * **固定权重**（本批=机制存在性验证）：扩"每通道 1 位权重 + 1 位阈值"= 8 位 = **纪元级**
    #    （R238 §2）⇒ 留批 B；本组字段即为届时基因位的默认值来源。
    #
    #  ▶ 与 `channels`（场本体开关）**分开**：本开关只管"读不读/怎么读" ⇒ 场关时本项 fail-loud。
    #  ▶ 性能口径（R244 §2.4）：每 tick **一次**候选格并集取数（O(N×k)，不是 O(n_cells)）⇒ 读路径。
    use_in_move: bool = False        # 🔴 消费端总开关（默认关 = 回滚点，逐位不变）
    w_food: float = 0.5              # ① 食物通道权重（正 = 趋近）
    w_prey: float = 0.0
    w_risk: float = -0.5             # ③ 风险通道权重（负 = 回避；见上"权重自带符号"）
    w_kin: float = 0.0
    norm_mode: str = "analytic"      # 归一化：`analytic`（解析上界，本批唯一实现）/ `window`（未实现）

    _KNOWN_CHANNELS = ("food", "prey", "risk", "kin")
    _KNOWN_NORM_MODES = ("analytic",)

    def __post_init__(self) -> None:
        # 元组化（`to_dict` ⇒ tuple、快照 JSON ⇒ list；统一成 tuple ⇒ 指纹/比较稳定）
        self.channels = tuple(str(c) for c in (self.channels or ()))
        bad = [c for c in self.channels if c not in self._KNOWN_CHANNELS]
        assert not bad, (
            f"未知气味通道 {bad} ⇒ 会把**拼错的通道**静默跑成 no-op（B3 家族）"
            f"；允许：{self._KNOWN_CHANNELS}")
        assert len(set(self.channels)) == len(self.channels), "通道不得重复"
        assert self.update_every >= 1, "update_every 至少 1（=1 即每 tick；红线见类 docstring）"
        assert 1 <= self.downsample <= 64, "downsample ∈ [1,64]"
        assert 0.0 < self.decay <= 1.0, "decay ∈ (0,1]（=1 表示不衰减）"
        assert 0.0 <= self.diffuse <= 0.25, (
            "diffuse ∈ [0,0.25]：五点拉普拉斯显式格式的稳定域（>0.25 会振荡/发散）")
        for nm in ("inject_food", "inject_prey", "inject_risk", "inject_kin"):
            assert getattr(self, nm) >= 0.0, f"{nm} 非负"
        assert 0.0 <= self.risk_g16_threshold <= 1.0, "risk_g16_threshold ∈ [0,1]"
        # ---- R244 §二：消费端 ----
        for nm in ("w_food", "w_prey", "w_risk", "w_kin"):
            v = float(getattr(self, nm))
            # `v == v` 挡 NaN（config.py 不依赖 numpy ⇒ 不用 isnan）
            assert v == v and -10.0 <= v <= 10.0, f"{nm}={v} 非法（需有限且 |v| ≤ 10，防手滑量级）"
        assert self.norm_mode in self._KNOWN_NORM_MODES, (
            f"norm_mode={self.norm_mode!r} 未实现（本批只实现 {self._KNOWN_NORM_MODES}）"
            f" —— 未实现的模式**必须炸**，不许静默退回解析上界")
        if self.use_in_move:
            assert self.channels, (
                "use_in_move=True 但 channels 为空（场本体全关）⇒ 消费端会静默 no-op；"
                "请先启用至少一个通道（B3 家族：禁静默空跑）")


#: `from_dict` 的字段白名单（旧存档缺键 ⇒ 回退默认；多出的键 ⇒ 忽略而非报错）
_RESOURCE_DYNAMICS_FIELDS: frozenset = frozenset(
    f.name for f in fields(ResourceDynamicsConfig)
)

#: 13.8 迁徙配置白名单（同 `_RESOURCE_DYNAMICS_FIELDS` 规格：旧存档缺键回退默认）。
_MIGRATION_FIELDS: frozenset = frozenset(f.name for f in fields(MigrationConfig))

#: 14.9 ARS 配置白名单（同规格）。
_ARS_FIELDS: frozenset = frozenset(f.name for f in fields(ArsConfig))

#: P0.0 A1 信号场配置白名单（同规格；A1 之前的存档无 `signals` 键 ⇒ 回退默认）。
_SIGNALS_FIELDS: frozenset = frozenset(f.name for f in fields(SignalsConfig))

#: 13.4 波 1 亚格连续坐标白名单（同规格）。
#   🔴 R233 T-F 修复（2026-09-27）：`from_dict` 此前**整段漏传 `subpos`** ⇒ 走
#   `load_snapshot(config=None)` 的快照续跑会把 subpos **静默退回默认**（`enabled=False`，
#   `subdiv=4`/`speed_max=2.0`/`gain=4.0`）——运动模型被换掉 ⇒ "续跑 ≠ 连续跑"。
#   此前未被发现：云端的 save/resume 验证跑在 subpos **关**（默认）档；T-F 才用 rd+subpos 档暴露。
_SUBPOS_FIELDS: frozenset = frozenset(f.name for f in fields(SubposConfig))

#: R247 饥饿调制白名单（同规格；R247 之前的存档无 `hunger_mod` 键 ⇒ 回退默认关）。
_HUNGER_MOD_FIELDS: frozenset = frozenset(f.name for f in fields(HungerModConfig))
#: R240 T8 气味场配置白名单（同规格；T8 之前的存档无 `smell` 键 ⇒ 回退默认 = 全关）。
_SMELL_FIELDS: frozenset = frozenset(f.name for f in fields(SmellConfig))
#: R238/R239 模式仲裁白名单（同规格；ASM 之前的存档无 `action_selection` 键 ⇒ 回退 fusion）。
_ACTION_SELECTION_FIELDS: frozenset = frozenset(
    f.name for f in fields(ActionSelectionConfig)
)


@dataclass
class EnergyProbeConfig:
    """ENERGY-CLOSE **步A 只读累加器**（A2 独立实测侧的散逸/逃逸分账埋点）。

    口径：`_archive/2026-10-10-退役团队-归档/docs-旧档/设计文档/记录-ENERGY-CLOSE步A-通道枚举与A2三账口径-20261007.md` §三/§四
    （7 通道 = 4 散逸 `dis_meta/dis_move/dis_attack/dis_signal`
    + 3 逃逸 `esc_death_e/esc_pred_e/esc_pred_s`）。
    🔴 **列面纪律（轻舟约束 1 + 砚⑤）**：本机制**不动** `EC_*` 枚举、**不动**
    `energy_ledger` 列名（`tools/calib_solve.py` 按现列面消费）⇒ 全部读数走
    **sidecar 新键**（`energy_probe_rows()` / `energy_probe_totals()`），
    且 sidecar 行带 `tick` + 群体规模锚，可与主 CSV 逐 tick 行回 join。
    🔴 **默认关 = 旧行为逐位不变**（关档时引擎侧整段不执行 ⇒ 零开销；
    C7 钉死基线 `(574887, 11266.746993)` 不许动 = 唯一回滚点）。
    ⚠️ 挂进 `SimConfig` ⇒ 经 `asdict` **自动进 `fingerprint()`**（跨档续跑被拦，
    同 `smell` / `action_selection`）⇒ commit/manifest 须写明"**指纹变、行为不变**"。
    ⚠️ **只在 Python 路径可用**：`use_sim_core=True` 时维持/代谢扣费发生在 Rust 内部
    ⇒ 构造期**硬失败**（不放行 = 一整批空 sidecar 白跑）。
    """

    enabled: bool = False        # 🔴 总开关：默认关（关档逐位等价 = 唯一回滚点）
    row_every: int = 1           # sidecar 行频（每 N tick 落一行；1 = 逐 tick）
    max_rows: int = 0            # 0 = 不限；>0 ⇒ 环形保留最近 N 行（长批防内存）

    def __post_init__(self) -> None:
        assert isinstance(self.enabled, bool), "enabled 必须是布尔值"
        assert self.row_every >= 1, (
            "row_every ≥ 1（0/负数 = 永不落行，属配置错误而非关档 ⇒ 不许静默）"
        )
        assert self.max_rows >= 0, "max_rows ≥ 0（0 = 不限）"


#: 步A 能量探针白名单（同规格；步A 之前的存档无 `energy_probe` 键 ⇒ 回退默认 = 全关）。
_ENERGY_PROBE_FIELDS: frozenset = frozenset(f.name for f in fields(EnergyProbeConfig))


@dataclass
class EnvFieldConfig:
    """ENV 场：地形 → 气候态水分 → 格子产能因子（「纯斑块世界」底层规则）。

    口径来源
    --------
    * `docs/设计文档/设计-纯斑块世界-底层规则自组织-20261011.md`（§十五 拍板：全规则一次到位、
      ≤4 run 半对照、单 run ≤2000 tick）；
    * `docs/设计文档/设计-ENV-FIELD-物理场与格子产能-20261008.md`（场定义 H/W/P、归一化条款、
      四条算路与守卫口径）。

    是什么（与 ENV-FIELD 设计稿的差别，写死防混）
    --------------------------------------------
    本机制**在既有斑块世界之上再加一层连续产能因子**（H→W→f_g/f_c 乘在斑块倍率之上），
    **不撤** patch_mask / 斑块倍率（那是 ENV-FIELD 稿的 (a) 形态替换 = 后续纪元）。
    全部新场是**构造期静态派生量** ⇒ **不进快照**（确定性可从 config 重建 ⇒ 存档零改动）。

    🔴 默认关与回滚点
    -----------------
    * `enabled=False`（默认）⇒ 引擎**不构造** EnvField（零成本 + 旧行为逐位等价 = C7 回滚点）；
    * 开档但 `water_sensitivity=0 ∧ cap_sensitivity=0` ⇒ 两个因子整块跳过（恒 1）
      ⇒ **同样逐位等价**（ENV-FIELD §二-5 的同型"回归安全"设计）。

    🔴 fail-loud 组合（构造期，见 `sphere_engine` 接线；禁静默）
    ----------------------------------------------------------
    * `enabled ∧ simulation.use_sim_core` ⇒ 报错（Rust regrow 不吃新因子 ⇒ 静默 no-op，
      同 H3 先例 `sphere_engine.py:1026-1036`）；
    * `enabled ∧ resource_dynamics.enabled` ⇒ 报错（**未验证组合**：rd 的容量/掩码重建
      与空间变化容量因子未对拍，宁炸不静默）；
    * `enabled ∧ simulation.sparse_fields` ⇒ **资源侧惰性显式退回全场** + `eng.env_note`
      一行（ENV-FIELD §五-4 口径 (ii)：不静默；信号侧与 ENV 正交不受影响）。

    🔴 RNG 纪律（新工具四律之一）
    ----------------------------
    噪声取**独立流** `default_rng([engine_seed, salt, 常数])`，**不消费引擎 `self.rng`**
    （同 `patch_seed` 先例 `resource_field.py:167`）⇒ 同 seed 下开/关档引擎 RNG 序列逐位不变。

    ⚠️ 挂进 `SimConfig` ⇒ 经 `asdict` **自动进 `fingerprint()`**（跨档续跑被拦，同 `smell` /
    `energy_probe`）⇒ commit/manifest 须写明"**指纹变、行为不变**"（默认关档）。
    """

    enabled: bool = False            # 总开关；False ⇒ 引擎不构造 EnvField（零成本 + 逐位等价）
    salt: int = 0                    # 独立流盐：换盐 = 换地形格局（不动 SimConfig.seed）
    terrain_lattice: int = 64        # 3D 值噪声立方格点分辨率（64³ ⇒ 特征尺度 ≈ 5.6°）
    terrain_octaves: int = 6         # fBm 八度数（振幅 ×0.5、频率 ×2 每级）
    terrain_smooth: int = 2          # 路由前平滑遍数（真 8 邻居均值；抑制噪声级微洼）
    shuffle: bool = False            # 🔴 置换零模型（对照臂）：空间置换 H（保直方图、毁结构）
    lapse_c: float = 20.0            # H∈[0,1] 全程温差（°C）：T = base_temperature − lapse×H
    orographic: float = 1.0          # 抬升增雨系数：Pr = q × (1 + orographic×H)
    et_frac: float = 0.5             # 蒸散比：Et = et_frac × q（q = Magnus 饱和水汽压，hPa）
    runoff_gain: float = 2.0         # 汇流增益：amp = 1 + gain×A/(A+A_ref)（0 ⇒ 无河汇流项）
    water_sensitivity: float = 0.0   # 再生因子指数：f_g = u^ws / 面积权均值（0 ⇒ 整块跳过）
    cap_sensitivity: float = 0.0     # 容量因子指数：f_c = clip(u,0,k_max)^cs / 均值（0 ⇒ 跳过）
    k_max: float = 4.0               # 容量因子 clip 上界（河带封顶：cap×k_max^cs）

    def __post_init__(self) -> None:
        assert isinstance(self.enabled, bool), "enabled 必须是布尔值"
        assert isinstance(self.shuffle, bool), "shuffle 必须是布尔值"
        assert self.terrain_lattice >= 4, "terrain_lattice ≥ 4（太小 = 全场一个团块）"
        assert self.terrain_octaves >= 1, "terrain_octaves ≥ 1"
        assert self.terrain_smooth >= 0, "terrain_smooth ≥ 0"
        assert self.k_max >= 1.0, "k_max ≥ 1（< 1 会把富水格容量压低 = 反向机制）"
        # `v == v` 挡 NaN（config.py 不依赖 numpy ⇒ 不用 isnan；同 SmellConfig 先例）
        for nm in ("lapse_c", "orographic", "et_frac", "runoff_gain",
                   "water_sensitivity", "cap_sensitivity"):
            v = float(getattr(self, nm))
            assert v == v and abs(v) <= 100.0, f"{nm}={v} 非法（需有限且 |v| ≤ 100）"
        assert self.lapse_c >= 0.0, "lapse_c ≥ 0（递减率非负）"
        assert self.orographic >= 0.0, "orographic ≥ 0"
        assert self.et_frac >= 0.0, "et_frac ≥ 0"
        assert self.runoff_gain >= 0.0, "runoff_gain ≥ 0"
        assert self.water_sensitivity >= 0.0, "water_sensitivity ≥ 0"
        assert self.cap_sensitivity >= 0.0, "cap_sensitivity ≥ 0"


#: 纯斑块世界 ENV 场白名单（同规格；本批之前的存档无 `env_field` 键 ⇒ 回退默认 = 全关）。
_ENV_FIELD_FIELDS: frozenset = frozenset(f.name for f in fields(EnvFieldConfig))


@dataclass
class SimConfig:
    """顶层配置：唯一事实来源，决定一次完整模拟。"""

    seed: int = 42                  # 随机数种子（同配置+同种子 → 同结果）
    world: WorldConfig = field(default_factory=WorldConfig)
    light: LightConfig = field(default_factory=LightConfig)
    resources: ResourceConfig = field(default_factory=ResourceConfig)
    organisms: OrganismConfig = field(default_factory=OrganismConfig)
    genome: GenomeConfig = field(default_factory=GenomeConfig)
    population: PopulationConfig = field(default_factory=PopulationConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    pleasure: PleasureConfig = field(default_factory=PleasureConfig)
    predation: PredationConfig = field(default_factory=PredationConfig)
    culture: CultureConfig = field(default_factory=CultureConfig)
    info_structure: InfoStructureConfig = field(default_factory=InfoStructureConfig)
    fruit: FruitConfig = field(default_factory=FruitConfig)
    # ---- 尸体—食腐 + 血条（S1 骨架；默认全关 = 旧行为逐位一致）----
    corpse_wound: CorpseWoundConfig = field(default_factory=CorpseWoundConfig)
    # ---- 13.4 波 1：亚格连续坐标（默认关 = 旧行为**逐位一致**）----
    #   ⚠️ 挂进 SimConfig ⇒ 由 `to_dict()`/`config_fingerprint()`（= `asdict`）**自动进指纹**
    #      ⇒ "跨档续跑"会被拦（同 `signal_alphabet` 的机制）。
    subpos: SubposConfig = field(default_factory=SubposConfig)
    # ---- 13.4 波 2：斑块"休耕—死亡—轮作"（默认关 = 旧行为逐位等价）----
    #   实现落在新文件 `world/resource_dynamics.py`；本配置是它的参数契约。
    resource_dynamics: ResourceDynamicsConfig = field(default_factory=ResourceDynamicsConfig)
    # ---- 14.9 ARS 双模式觅食（默认关 = 旧行为**逐位一致**）----
    ars: ArsConfig = field(default_factory=ArsConfig)
    # ---- 13.8 日历—罗盘式定向迁徙（默认关 = 旧行为**逐位等价**）----
    #   消费 13.7 的 δ(t)（**不改** illumination / light_sensitivity），只**新增**
    #   `LightAndTemperature.photoperiod`。挂进 SimConfig ⇒ 经 `asdict` **自动进指纹**
    #   （跨档续跑被拦，同 `subpos` / `resource_dynamics`）。
    migration: MigrationConfig = field(default_factory=MigrationConfig)
    # ---- P0.0 阶段 A1（2026-09-26）：信号场 tick 常数（默认 50 = 旧行为逐位等价）----
    #   挂进 SimConfig ⇒ 经 `asdict` **自动进指纹**（跨档续跑被拦，同 `subpos`）。
    signals: SignalsConfig = field(default_factory=SignalsConfig)
    # ---- R247 饥饿调制（HM；默认关 = 旧行为**逐位等价**）----
    #   实施规格：`docs/设计文档/实施规格-移动决策三件-20260928.md` §一（已冻结）。
    #   挂进 SimConfig ⇒ 经 `asdict` **自动进指纹**（跨档续跑被拦，同 `subpos`）。
    hunger_mod: HungerModConfig = field(default_factory=HungerModConfig)
    # ---- R238/R239 模式仲裁（ASM；默认 `mode="fusion"` = 现状**逐位等价**）----
    #   实施规格：同文件 §三（已冻结）。依赖 `smell.channels` 含 food/risk/kin
    #   （缺 ⇒ 引擎构造期 fail-loud，不许静默降级）。
    action_selection: ActionSelectionConfig = field(default_factory=ActionSelectionConfig)
    # ---- ENERGY-CLOSE 步A：只读能量探针（**默认关 = 旧行为逐位等价**）----
    #   读数一律走 sidecar 新键，**不动** `EC_*` 枚举与 `energy_ledger` 列面（轻舟约束 1）。
    #   挂进 SimConfig ⇒ 经 `asdict` 自动进指纹（跨档续跑被拦，同 `smell` / `action_selection`）。
    energy_probe: EnergyProbeConfig = field(default_factory=EnergyProbeConfig)
    # ---- 纯斑块世界（2026-10-11）：ENV 场（地形→气候态水分→产能因子；**默认关 = 逐位等价**）----
    #   实现落在新文件 `world/env_field.py`；构造期静态派生量 ⇒ **不进快照**（可从 config 重建）。
    #   挂进 SimConfig ⇒ 经 `asdict` 自动进指纹（跨档续跑被拦，同 `smell` / `energy_probe`）。
    env_field: EnvFieldConfig = field(default_factory=EnvFieldConfig)

    # ---- D1 零模型三开关（进 fingerprint，用于对照实验） ----
    neutral_genes: bool = False          # 零模型：只冻结 g14/g15（感知/信号），其余照常演化（C3 修正）
    signal_disabled: bool = False        # 不发信号：发射概率恒0，接收/解读照常
    signal_mode: str = "state"           # 信号编码：state(现状)/random(独立rng随机)/evolved(D2码本暂未接线)
    signal_alphabet: str = "16"          # 信号字母表（R113/R121）："16"(现状,4位)/"4"(2位,仅能量)/"8"(B③,未实施)——见模块顶部常量族

    # ---- V-1 oracle 正向对照（R39 / D-8）；旧存档缺失回退默认关闭 ----
    oracle: OracleConfig = field(default_factory=OracleConfig)

    # ---- R240 T8：气味场（多通道标量场；**默认全关 = 零成本 + 旧行为逐位不变**）----
    #   实现落在新文件 `world/smell_field.py`；挂进 SimConfig ⇒ 经 `asdict` 自动进指纹
    #   （跨档续跑被拦，同 `subpos` / `resource_dynamics` / `signals`）。
    smell: SmellConfig = field(default_factory=SmellConfig)

    # ---- 可复现性辅助：配置 ⇄ dict ------------------------------

    def to_dict(self) -> dict:
        """把整个配置导出成普通 dict（存档/对比用）。"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "SimConfig":
        """从 dict 重建配置（保证存档可得回同样的配置）。

        旧存档缺失 predation/culture 键时回退默认值（保持向前兼容）。
        """
        return cls(
            seed=data["seed"],
            world=WorldConfig(**data["world"]),
            light=LightConfig(**data["light"]),
            resources=ResourceConfig(**data["resources"]),
            organisms=OrganismConfig(**data["organisms"]),
            genome=GenomeConfig(**data["genome"]),
            population=PopulationConfig(**data["population"]),
            simulation=SimulationConfig(**data["simulation"]),
            # L2 新增的愉悦度配置此前遗漏；旧存档缺少该键时回退默认值。
            pleasure=(
                PleasureConfig(**data["pleasure"])
                if "pleasure" in data
                else PleasureConfig()
            ),
            # A2 新增捕食/文化配置（隐式选择压收编）；旧存档回退默认值。
            predation=(
                PredationConfig(**data["predation"])
                if "predation" in data
                else PredationConfig()
            ),
            culture=(
                CultureConfig(**data["culture"])
                if "culture" in data
                else CultureConfig()
            ),
            # D2 信息结构重构配置；旧存档回退默认值（enabled=False，不改变旧行为）。
            # B1：用 _INFO_FIELDS 过滤未知键 → 旧快照缺 reputation_weight 时回退默认 0.0。
            info_structure=InfoStructureConfig(
                **{
                    k: v
                    for k, v in (data.get("info_structure") or {}).items()
                    if k in _INFO_FIELDS
                }
            ),
            # L10a 新增果实-种子传播配置；旧存档回退默认值（enabled=False）。
            fruit=(
                FruitConfig(**data["fruit"])
                if "fruit" in data
                else FruitConfig()
            ),
            # 尸体—食腐 + 血条配置（S1 骨架）；旧存档回退默认值（全关 = 旧行为）。
            corpse_wound=CorpseWoundConfig(
                **{
                    k: v
                    for k, v in (data.get("corpse_wound") or {}).items()
                    if k in _CORPSE_WOUND_FIELDS
                }
            ),
            # 13.4 波 2：斑块动态配置；旧存档缺失 ⇒ 回退默认（enabled=False = 旧行为）。
            #   B1 同款：用字段白名单过滤未知键（防旧存档带多余键时 TypeError）。
            resource_dynamics=ResourceDynamicsConfig(
                **{
                    k: v
                    for k, v in (data.get("resource_dynamics") or {}).items()
                    if k in _RESOURCE_DYNAMICS_FIELDS
                }
            ),
            # 13.8：日历—罗盘式定向迁徙配置；旧存档缺失 ⇒ 回退默认（enabled=False = 旧行为）。
            #   同款字段白名单过滤（向后兼容：13.8 之前的存档无 `migration` 键）。
            migration=MigrationConfig(
                **{
                    k: v
                    for k, v in (data.get("migration") or {}).items()
                    if k in _MIGRATION_FIELDS
                }
            ),
            # 14.9：ARS 双模式觅食；旧存档缺失 ⇒ 回退默认（enabled=False = 旧行为）。
            ars=ArsConfig(
                **{
                    k: v
                    for k, v in (data.get("ars") or {}).items()
                    if k in _ARS_FIELDS
                }
            ),
            # D-8：oracle 配置；旧存档缺失时回退默认关闭（C-6/C-7 先例同 reputation_weight）。
            oracle=OracleConfig(
                **{
                    k: v
                    for k, v in (data.get("oracle") or {}).items()
                    if k in _ORACLE_FIELDS
                }
            ),
            # D1 零模型三开关；旧存档回退默认值。
            neutral_genes=data.get("neutral_genes", False),
            signal_disabled=data.get("signal_disabled", False),
            signal_mode=data.get("signal_mode", "state"),
            # R113/R121 信号字母表；旧存档回退 "16"（= 旧行为，逐位兼容）
            signal_alphabet=data.get("signal_alphabet", "16"),
            # P0.0 A1：信号场 tick 常数；旧存档缺失 ⇒ 回退默认 50（= 旧行为逐位等价）。
            #   同款字段白名单过滤（向后兼容：A1 之前的存档无 `signals` 键）。
            signals=SignalsConfig(
                **{
                    k: v
                    for k, v in (data.get("signals") or {}).items()
                    if k in _SIGNALS_FIELDS
                }
            ),
            # 13.4 波 1：亚格连续坐标；旧存档缺失 ⇒ 回退默认（enabled=False = 旧行为）。
            #   🔴 R233 T-F：此前**漏传**（快照续跑静默丢 subpos）——补上并进往返测试防守。
            subpos=SubposConfig(
                **{
                    k: v
                    for k, v in (data.get("subpos") or {}).items()
                    if k in _SUBPOS_FIELDS
                }
            ),
            # R247 饥饿调制；旧存档缺失 ⇒ 回退默认（enabled=False = 旧行为逐位等价）。
            #   同款字段白名单过滤（向后兼容：R247 之前的存档无 `hunger_mod` 键）。
            hunger_mod=HungerModConfig(
                **{
                    k: v
                    for k, v in (data.get("hunger_mod") or {}).items()
                    if k in _HUNGER_MOD_FIELDS
                }
            ),
            # R240 T8：气味场；旧存档缺失 ⇒ 回退默认（channels=() = 全关 = 旧行为逐位不变）。
            #   同款字段白名单过滤（T8 之前的存档无 `smell` 键）。
            #   ⚠️ 与 subpos 同族教训：**新组必须在此显式接线**，否则 `load_snapshot(config=None)`
            #   的续跑会把该组静默退回默认（R233 T-F）。
            smell=SmellConfig(
                **{
                    k: v
                    for k, v in (data.get("smell") or {}).items()
                    if k in _SMELL_FIELDS
                }
            ),
            # R238/R239 模式仲裁；旧存档缺失 ⇒ 回退默认（mode="fusion" = 现状逐位等价）。
            #   同款字段白名单过滤（向后兼容：ASM 之前的存档无 `action_selection` 键）。
            action_selection=ActionSelectionConfig(
                **{
                    k: v
                    for k, v in (data.get("action_selection") or {}).items()
                    if k in _ACTION_SELECTION_FIELDS
                }
            ),
            # ENERGY-CLOSE 步A 能量探针；旧存档缺失 ⇒ 回退默认（enabled=False = 逐位等价）。
            #   同款字段白名单过滤（步A 之前的存档无 `energy_probe` 键）。
            #   🔴 同 R233 T-F 教训：**新组必须在此显式接线**，否则 `load_snapshot(config=None)`
            #   的续跑会把该组静默退回默认（探针档悄悄变关档 = 空 sidecar 白跑）。
            energy_probe=EnergyProbeConfig(
                **{
                    k: v
                    for k, v in (data.get("energy_probe") or {}).items()
                    if k in _ENERGY_PROBE_FIELDS
                }
            ),
            # 纯斑块世界 ENV 场；旧存档缺失 ⇒ 回退默认（enabled=False = 旧行为逐位等价）。
            #   🔴 同 R233 T-F 教训：**新组必须在此显式接线**，否则 `load_snapshot(config=None)`
            #   的续跑会把该组静默退回默认（开档批"续跑 ≠ 连续跑"）。
            env_field=EnvFieldConfig(
                **{
                    k: v
                    for k, v in (data.get("env_field") or {}).items()
                    if k in _ENV_FIELD_FIELDS
                }
            ),
        )

    def fingerprint(self) -> str:
        """配置的规范字符串；两份配置是否完全一致（排查复现用）。"""
        import json

        return json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False)