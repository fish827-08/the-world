"""simulation.genes：基因位语义注册表（引擎/Rust/观测台共享的唯一事实源）。

通俗理解
--------
基因链是形状 (N, gene_count) 的数组：第 0 列是"移动概率"、第 3 列是"寿命"、
第 16 列是"攻击性"……这些索引过去散落在 Python 引擎、Rust 内核、观测台三处，
靠注释和记忆约定。一旦新增基因位，很容易写错索引（比如把感知 g14 和攻击 g16
记混），而且一处改了别处忘记同步。

本注册表把【索引 ↔ 语义】集中登记，成为唯一事实源：
- Python 引擎用 ``Gene.LIFE`` 而不是裸数字 ``3``；
- Rust 内核用 ``G_LIFE``（与 Python 常量同值，lib.rs 加载时带断言校验）；
- 观测台 trait 表从这个注册表读取名字，不再手写 24 个字符串。

对外接口
--------
- ``class Gene(IntEnum)``   基因位常量（成员名 = 语义名，值 = 列索引）
- ``GENE_COUNT``            基因位数（= len(Gene)）
- ``GENE_TRAIT_NAMES``      trait 表语义名元组（24 项，与列索引一一对应）
- ``GENE_SEMANTICS``        基因功能说明元组（每列一条）
- ``GENE_WIRED``            已接线的基因位集合（未接线 = 预留位，观察台仍跟踪漂移）
- ``gene_index(name)``      按语义名取列索引（跨层统一入口）
- ``gene_name(idx)``        按列索引取语义名
"""
from __future__ import annotations

from enum import IntEnum


class Gene(IntEnum):
    """24 位基因。成员值即 _genes 列索引；成员名即语义名（跨层唯一标识）。

    命名规则：短横下划线式语义名，与引擎消费点一一对应。
    新增基因位 = 在末尾追加一个成员，同步更新 GENE_SEMANTICS/GENE_TRAIT_NAMES；
    已接线的基因在引擎/Rust 消费，未接线的仅被观察台跟踪漂移（预留位）。
    """

    MOVE_PROB = 0            # 移动概率（被 g19 植物化缩放）
    METABOLIC = 1            # 代谢倍率（消化快慢）
    REPRO_THRESHOLD = 2      # 繁殖能量门槛
    LIFE_GENE = 3            # 寿命基因（影响 lifespan 派生）
    EAT_AMOUNT = 4           # 进食量
    STOMACH_CAP = 5          # 胃容量
    MOVE_COST = 6            # 移动能耗
    PARENTAL_INVEST = 7      # 传代投入（分给子代的能量比例）
    PHOTOSYNTHESIS = 8       # 光合产能
    HOMEOTHERM = 9           # 恒温性
    FORAGE_NEIGHBOR = 10     # 邻格觅食倾向
    TEMP_PREF = 11           # 温度偏好
    REPRO_COOLDOWN = 12      # 繁殖冷却长度
    SOCIABILITY = 13         # 群居性（移动决策 soc 项，Rust movement 消费）
    PERCEPTION = 14          # 感知半径（移动决策 perc 项，Rust movement 消费）
    SIGNAL_STRENGTH = 15     # 信号发射概率（L3）
    AGGRESSION = 16          # 攻击性（捕食概率，L4，Rust predation 消费）
    DIET = 17                # 食性（未接线，预留）
    DEFENSE = 18             # 防御（未接线，预留）
    ROOTING = 19             # 植物化扎根（移动概率 ×(1-g19) + 额外光合，L4）
    HEDONISM = 20            # 享乐敏感（未接线，预留）
    PROCESSING = 21          # 处理位（未接线，预留）
    TRUST_GENE = 22          # 信任阈值（未接线，预留）
    MIGRATE_BIAS = 23        # 迁徙倾向（日历—罗盘式定向迁徙，13.8 接线；原 RESERVED）


GENE_COUNT: int = len(Gene)

# name -> Gene 反向索引（注册表一致性由 IntEnum 内建保证，无需重复校验）
_INDEX_BY_NAME = {g.name: int(g) for g in Gene}


def _to_snake(name: str) -> str:
    """成员名（如 MOVE_PROB）转 trait 表语义名（move_prob）。"""
    return name.lower()


def gene_index(name: str) -> int:
    """按语义名取列索引（大小写不敏感）。

    Args:
        name: 语义名，如 "MOVE_PROB" / "move_prob" / "ROOTING"。
    Returns:
        列索引（0~GENE_COUNT-1）。
    Raises:
        KeyError: 未知语义名。
    """
    key = name.upper()
    if key not in _INDEX_BY_NAME:
        raise KeyError(f"未知基因语义名 {name!r}；可用：{sorted(_INDEX_BY_NAME)}")
    return _INDEX_BY_NAME[key]


def gene_name(idx: int) -> str:
    """按列索引取语义名（trait 表小写风格）。

    Args:
        idx: 列索引。
    Returns:
        语义名小写形式（如 "move_prob"）。
    Raises:
        IndexError: 索引越界。
    """
    if not 0 <= idx < GENE_COUNT:
        raise IndexError(f"基因索引 {idx} 越界（0~{GENE_COUNT-1}）")
    return _to_snake(Gene(idx).name)


# trait 表语义名（小写，与 GENE_TRAIT_NAMES 完全一致；index = 列索引）
GENE_TRAIT_NAMES: tuple[str, ...] = tuple(gene_name(i) for i in range(GENE_COUNT))

# 基因功能说明（每列一条；写在这里便于跨层引用，避免散落注释）
GENE_SEMANTICS: tuple[str, ...] = (
    "移动概率",            # g0
    "代谢倍率",            # g1
    "繁殖能量门槛",        # g2
    "寿命基因",            # g3
    "进食量",              # g4
    "胃容量",              # g5
    "移动能耗",            # g6
    "传代投入",            # g7
    "光合产能",            # g8
    "恒温性",              # g9
    "邻格觅食倾向",        # g10
    "温度偏好",            # g11
    "繁殖冷却长度",        # g12
    "群居性",              # g13
    "感知半径",            # g14
    "信号发射概率",        # g15
    "攻击性",              # g16
    "食性（未接线）",      # g17
    "防御（未接线）",      # g18
    "植物化扎根",          # g19
    "享乐敏感（未接线）",  # g20
    "处理位（未接线）",    # g21
    "信任阈值（未接线）",  # g22
    "迁徙倾向（日历—罗盘）",  # g23（13.8 接线；原「预留」）
)

# 已接线的基因位（引擎/Rust 消费）；未接线的仅是预留位（观察台跟踪漂移）
_GENE_WIRED: set[int] = {
    int(Gene.MOVE_PROB), int(Gene.METABOLIC), int(Gene.REPRO_THRESHOLD),
    int(Gene.LIFE_GENE), int(Gene.EAT_AMOUNT), int(Gene.STOMACH_CAP),
    int(Gene.MOVE_COST), int(Gene.PARENTAL_INVEST), int(Gene.PHOTOSYNTHESIS),
    int(Gene.HOMEOTHERM), int(Gene.FORAGE_NEIGHBOR), int(Gene.TEMP_PREF),
    int(Gene.REPRO_COOLDOWN), int(Gene.SOCIABILITY), int(Gene.PERCEPTION),
    int(Gene.SIGNAL_STRENGTH), int(Gene.AGGRESSION), int(Gene.ROOTING),
    int(Gene.MIGRATE_BIAS),
}
GENE_WIRED: frozenset[int] = frozenset(_GENE_WIRED)

# 基因元数据（G3，C3 基因扩展性）：每基因的进化参数，观察台/分析工具直接消费。
# 与 GENE_SEMANTICS 平行，index = 列索引；纯扩展，不改变引擎行为。
#
# 字段说明：
# - mutation_scale: 突变标准差（高斯突变的 σ）。核心代谢/寿命基因保守（小），
#   探索性基因（信号/感知/群居）宽松（大）；预留位用默认 0.1。
# - selection_direction: 选择方向。+1=高值正向选择（越高越有利），
#   -1=低值负向选择（越低越有利），0=中性/依赖环境。仅作观察台标注用，
#   引擎不强制（选择压来自生态动力学，不来自此元数据）。
from dataclasses import dataclass


@dataclass(frozen=True)
class GeneMeta:
    """单基因元数据。"""
    mutation_scale: float
    selection_direction: int  # +1 / 0 / -1


GENE_META: tuple[GeneMeta, ...] = (
    GeneMeta(0.05, 0),    # g0  移动概率：中性（依赖环境）
    GeneMeta(0.03, 0),    # g1  代谢倍率：保守，中性
    GeneMeta(0.05, -1),   # g2  繁殖阈值：低阈值易繁殖（负向选择，但受资源约束）
    GeneMeta(0.02, 0),    # g3  寿命基因：高度保守
    GeneMeta(0.05, 1),    # g4  进食量：高值有利（正向选择）
    GeneMeta(0.05, 1),    # g5  胃容量：高值有利
    GeneMeta(0.03, -1),   # g6  移动能耗：低值有利（负向选择）
    GeneMeta(0.05, 0),    # g7  传代投入：中性（r/K 策略权衡）
    GeneMeta(0.05, 1),    # g8  光合产能：高值有利（植物化路径）
    GeneMeta(0.05, 0),    # g9  恒温性：中性（温度依赖）
    GeneMeta(0.05, 1),    # g10 邻格觅食倾向：高值有利（资源稀缺时）
    GeneMeta(0.05, 0),    # g11 温度偏好：中性（环境依赖）
    GeneMeta(0.05, -1),   # g12 繁殖冷却：低值有利（繁殖快），但受资源约束
    GeneMeta(0.08, 0),    # g13 群居性：探索性，中性（密度依赖）
    GeneMeta(0.08, 1),    # g14 感知半径：探索性，高值有利（信息优势）
    GeneMeta(0.10, 0),    # g15 信号发射概率：高探索性，中性（信号成本/收益权衡）
    GeneMeta(0.08, 0),    # g16 攻击性：中性（捕食/防御权衡，密度依赖）
    GeneMeta(0.10, 0),    # g17 食性（预留）：默认
    GeneMeta(0.10, 0),    # g18 防御（预留）：默认
    GeneMeta(0.08, 1),    # g19 植物化扎根：高值有利（静态生态位）
    GeneMeta(0.10, 0),    # g20 享乐敏感（预留）：默认
    GeneMeta(0.10, 0),    # g21 处理位（预留）：默认
    GeneMeta(0.10, 0),    # g22 信任阈值（预留）：默认
    GeneMeta(0.10, 0),    # g23 迁徙倾向（13.8 接线；探索性位，宽松突变）
)


def gene_meta(idx: int) -> GeneMeta:
    """按列索引取基因元数据。

    Args:
        idx: 列索引（0~GENE_COUNT-1）。
    Returns:
        GeneMeta（mutation_scale, selection_direction）。
    Raises:
        IndexError: 索引越界。
    """
    if not 0 <= idx < GENE_COUNT:
        raise IndexError(f"基因索引 {idx} 越界（0~{GENE_COUNT-1}）")
    return GENE_META[idx]


def validate_gene_wiring() -> list[tuple[str, int, int]]:
    """Python ↔ Rust 基因索引双写校验（G2，C3 基因扩展性）。

    把 Python 侧 Gene 枚举的全部 (name, value) 传给 Rust 侧 validate_gene_wiring，
    逐位对照。返回不一致列表 [(name, rust_value, py_value), ...]；空列表表示一致。

    用途：
    - 引擎初始化时调用（use_sim_core=True），漂移则直接报错。
    - tests/test_genes_registry.py 调用，故意改一侧索引→返回非空→测试通过。

    Returns:
        不一致列表；空列表表示全部一致。
    """
    try:
        import sim_core
    except ImportError:
        # sim_core 未安装时跳过校验（纯 Python 路径不需要 Rust 侧常量）
        return []
    py_names = [g.name for g in Gene]
    py_values = [int(g) for g in Gene]
    return sim_core.validate_gene_wiring(py_names, py_values)