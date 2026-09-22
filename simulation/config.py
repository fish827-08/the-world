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

    def __post_init__(self) -> None:
        assert self.rotation_period >= 10, "自转周期至少 10 tick"
        assert self.t_equator > self.t_pole, "赤道必须比极地暖"


@dataclass
class ResourceConfig:
    """资源（食物）参数（对应 ResourceField 构造参数）。"""

    capacity_per_area: float = 40.0   # 每单位面积的食物上限（容量）
    regrowth_rate: float = 0.5        # 基准再生：温度合适时每 tick 每格长多少
    # 🔴 13.4 波 2（fish 00:38 裁定，**本波暂不改**）：**格内**取食速度必须 > 斑块生长速度。
    #   本字段将由 0.5 → **0.4**（配 `OrganismConfig.eat_amount` 0.6）。
    #   ⚠️ 为什么**不**在波 1 改（已裁定，见板帖 R174）：
    #     (1) 改它 = **改构造** ⇒ 会改变人口承载，并迫使更新 **7 处写死的 C7 基线 digest**
    #         （`(573985, 8171.692943)`）⇒ 与"坐标基础设施"混在一笔，会让波 1 的验收点
    #         「关档逐位等价」**失去可验证性**（同时改两件事 ⇒ 归因不干净）；
    #     (2) §14.6「阶段边界必须可发布 / 一次只改一件事」；
    #     (3) 它与 `cell_occupancy_cap`（单格个体上限，见 §十一.2）**必须同批** ——
    #         两者共同决定"**站着不动到底能不能活**"，分开做会得到互相矛盾的读数。
    #   📐 依据（R171 §十.2 的构造巧合）：原 0.5 与 `eat_amount`(0.5) **数值恰好相等** ⇒
    #     "一个静止个体独占一格"时"取 0.5 / 补 0.5"恰好动态平衡 ⇒ 静止成为**可持续策略**。
    #   📐 波 2 改后：格内比值 **1.5** ⇒ 静止者 `40/(0.6−0.4)` = **200 tick 吃空一格**；
    #     全球裕度：总再生 `Σcell_area(4586)×0.4 = 1834` vs 总需求 `N(1944)×0.6 = 1166` ⇒ **+57%**。
    #     （两个层面不矛盾：格内"吃>长"⇒ 必须移动；全球"需求<再生"⇒ 整体不枯竭。）
    temp_sensitivity: float = 1.0     # 再生对温度的依赖（0=不 care，越大越敏感）
    initial_fill: float = 0.5         # 初始填充比例（每格开始有多少食物，0~1）

    # ---- 斑块化（L1，守恒版）------------------------------------------
    # distribution="uniform" 时以下字段全部不生效，行为与旧版完全一致。
    # "patchy" 时：食物聚簇到斑块，背景压低；两条守恒保证总食物量不变：
    #   ① 容量守恒：Σ_capacity 与 uniform 版相等（种群承载上限不变）
    #   ② 再生守恒：周期平均总再生量与 uniform 版相等（时间上供给不变）
    distribution: str = "uniform"          # "uniform" | "patchy"
    patch_count: int = 30                  # 斑块中心数（默认保守，避免覆盖过大）
    patch_radius: int = 2                  # 斑块半径（格，邻居扩散层数）
    patch_capacity_mult: float = 3.0       # 斑块格容量倍率（>1；建议 1.5~4，过大会背景容量为负）
    patch_regrowth_mult: float = 2.0       # 斑块格再生倍率
    background_fill: float = 0.1            # 背景格初始食物占比（压低，否则协作无收益）

    def __post_init__(self) -> None:
        assert self.capacity_per_area > 0, "容量为正"
        assert self.regrowth_rate >= 0, "再生率非负"
        assert 0.0 <= self.initial_fill <= 1.0, "初始填充比例在 0~1"
        assert self.distribution in ("uniform", "patchy"), "distribution 只能是 uniform 或 patchy"
        if self.distribution == "patchy":
            assert self.patch_count >= 1, "斑块数至少 1"
            assert self.patch_radius >= 1, "斑块半径至少 1"
            assert self.patch_capacity_mult > 1.0, "斑块容量倍率必须 > 1（否则无富集）"
            assert self.patch_regrowth_mult > 0, "斑块再生倍率为正"
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
    eat_amount: float = 0.5           # 每 tick 每格进食量上限
    # 🔴 13.4 波 2（fish 00:38 裁定，**本波暂不改**）：将由 0.5 → **0.6**，
    #   与 `ResourceConfig.regrowth_rate`(0.4) 配对实现"**格内吃 > 长**"（比值 1.5）。
    #   ⚠️ 本波不改的三条理由、全球裕度核算、以及"必须与 `cell_occupancy_cap`（单格个体上限）
    #   同批"这条依赖 —— 全部见 `regrowth_rate` 处注释。
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

    def __post_init__(self) -> None:
        assert self.ticks >= 1, "至少跑一个 tick"
        assert 0.0 <= self.stay_prob < 0.95, "stay_prob 在 [0, 0.95)"


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
    rest_ticks: int = 300            # 被吃后**休耕**：该格 N tick 内再生 = 0
    kill_frac: float = 0.7           # 被吃强度 > 此值 ⇒ 斑块死亡（分母见 `kill_denom`）
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


#: `from_dict` 的字段白名单（旧存档缺键 ⇒ 回退默认；多出的键 ⇒ 忽略而非报错）
_RESOURCE_DYNAMICS_FIELDS: frozenset = frozenset(
    f.name for f in fields(ResourceDynamicsConfig)
)


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

    # ---- D1 零模型三开关（进 fingerprint，用于对照实验） ----
    neutral_genes: bool = False          # 零模型：只冻结 g14/g15（感知/信号），其余照常演化（C3 修正）
    signal_disabled: bool = False        # 不发信号：发射概率恒0，接收/解读照常
    signal_mode: str = "state"           # 信号编码：state(现状)/random(独立rng随机)/evolved(D2码本暂未接线)
    signal_alphabet: str = "16"          # 信号字母表（R113/R121）："16"(现状,4位)/"4"(2位,仅能量)/"8"(B③,未实施)——见模块顶部常量族

    # ---- V-1 oracle 正向对照（R39 / D-8）；旧存档缺失回退默认关闭 ----
    oracle: OracleConfig = field(default_factory=OracleConfig)

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
        )

    def fingerprint(self) -> str:
        """配置的规范字符串；两份配置是否完全一致（排查复现用）。"""
        import json

        return json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False)