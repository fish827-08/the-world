"""biosphere.genome：可变长基因组（B 线 MVP，W0-B）。

通俗理解
--------
每个生物的"基因"就是一串浮点数。这串数分三段：

1. **参数段**（前 30 个槽位，``PARAM_SLOT_COUNT``）：每个槽位有固定语义
   （比如 0 号 = 觅食感知增益、13 号 = 水生适应度），按名字寻址，
   由 ``PARAM_SLOTS`` 登记表唯一事实源管理——和 ``simulation/genes.py``
    的注册表思路一致，禁裸写索引。
2. **GRN 段**（第 30~71 槽，42 个基因）：喂给 ``develop.py`` 的基因调控
   网络，发育成控制器参数的"非线性调制层"——**映射本身可演化**。
3. **冗余段**（第 64 槽起）：插入突变多出来的基因，当前发育不读，
   但参与遗传与变异（为将来扩展预留）。

长度可变：名义 64 槽（``GENOME_NOMINAL_SLOTS``），插删突变使实际长度在
``[GENOME_MIN_SLOTS, GENOME_MAX_SLOTS]`` 间浮动。插入/删除发生在参数段或
GRN 段内部时会顺移下游基因（类似 frameshift），这是突变效应的一部分。

确定性纪律
----------
``mutate`` 消费 ``numpy.random.Generator`` 的顺序**写死**：
先抽突变类型，再抽位置，最后抽值。同 seed 同基因组 ⇒ 突变结果逐位一致。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

# ── 布局常量（冻结；改布局 = 存档格式升级，须先报变更单） ──────────────────

#: 名义槽位数（MVP 目标长度；30 参数 + 42 GRN 基因）
GENOME_NOMINAL_SLOTS: int = 72
#: 长度下界（删除突变到此不再删）
GENOME_MIN_SLOTS: int = 32
#: 长度上界（插入突变到此不再插）
GENOME_MAX_SLOTS: int = 128
#: 参数段槽位数（固定语义段）
PARAM_SLOT_COUNT: int = 30
#: GRN 基因段起始槽位
GRN_GENE_OFFSET: int = PARAM_SLOT_COUNT
#: GRN 基因段长度（6 节点：36 权重 + 6 阈值；与 develop.GRN_NODES 联动）
GRN_GENE_COUNT: int = 42

# ── 参数段注册表（唯一事实源；禁裸写索引，用 PARAM_SLOTS[name]） ──────────

#: 参数段语义名 → 槽位索引。**追加只能加在末尾**（改索引 = 旧基因组作废）。
PARAM_SLOTS: Dict[str, int] = {
    "sense_food": 0,          # 觅食感知增益（食物场强度权重）
    "sense_threat": 1,        # 威胁感知增益（附近攻击者/受击记忆）
    "sense_kin": 2,           # 同类感知增益（结群/粘附的感知基础）
    "sense_signal": 3,        # 信号感知增益（读信号层的权重）
    "move_bias": 4,           # 基础移动倾向（不动与动的先验比）
    "eat_greed": 5,           # 进食倾向（本地有食物时 EAT 的得分权重）
    "attack_drive": 6,        # 攻击倾向
    "share_drive": 7,         # 分享倾向（给低能同类输能）
    "signal_drive": 8,        # 发信号倾向
    "repro_drive": 9,         # 繁殖倾向
    "metabolism_base": 10,    # 基础代谢率倍率
    "move_cost_mult": 11,     # 移动能耗乘子
    "habitat_aquatic": 12,    # 水生适应（<阈值进不了水）
    "habitat_terrestrial": 13,  # 陆生适应（<阈值上不了岸）
    "adhesion": 14,           # 粘附（结群吸引力强度）
    "defense": 15,            # 防御（被攻击减伤）
    "eat_amount": 16,         # 单次进食量
    "signal_payload": 17,     # 单次信号强度
    "repro_thresh": 18,       # 繁殖能量门槛
    "lifespan_gene": 19,      # 寿命基因（年龄上限）
    "give_up": 20,            # 失望阈值（连续没吃到就换方向）
    "persistence": 21,        # 赶路惯性（沿上一步方向继续走的强度）
    # ── C3 v2（B14.3；owner 已批：徽章含义为空，规则只写物理） ──
    "badge": 22,              # 徽章基因（离散化 K 类；含义不写死）
    "emit_rate": 23,          # 信号发射率（发射耗能的速率）
    "strength": 24,           # 强度（打斗威力加成；与 defense 对立）
    "response_0": 25,         # 对徽章 0 的反应权重（ signed：+攻/-避 ）
    "response_1": 26,
    "response_2": 27,
    "response_3": 28,
    "learning_rate": 29,      # 记忆学习率（0 = 纯演化；首轮默认关）
}

assert len(PARAM_SLOTS) == PARAM_SLOT_COUNT, "PARAM_SLOTS 数量须等于 PARAM_SLOT_COUNT"
assert set(PARAM_SLOTS.values()) == set(range(PARAM_SLOT_COUNT)), "参数槽位须连续无洞"

# ── 突变类型 ──────────────────────────────────────────────────────────────


class MutationKind:
    """突变类型（字符串常量，免 enum 依赖）。"""

    POINT = "point"      # 点突变：某槽位加减噪声
    INSERT = "insert"    # 插入：某位置复制/新插一个基因（长度 +1）
    DELETE = "delete"    # 删除：某位置删一个基因（长度 -1）


#: 突变类型概率（顺序固定：POINT/INSERT/DELETE）
MUTATION_PROBS: Tuple[float, float, float] = (0.70, 0.15, 0.15)

#: 点突变噪声标准差（高斯）
POINT_MUTATION_SCALE: float = 0.25


@dataclass
class EvolutionConfig:
    """演化尺度参数（M4 去写死：进世界族参数，默认 = 历史常量）。

    ``mutation_scale``：点突变高斯尺度；``grn_w_scale`` /
    ``grn_input_scale``：GRN 权重/输入缩放（develop 读）。
    """

    mutation_scale: float = POINT_MUTATION_SCALE
    grn_w_scale: float = 2.0
    grn_input_scale: float = 0.25


@dataclass(frozen=True)
class MutationRecord:
    """一次突变的记录（进谱系度量用）。"""

    kind: str
    position: int
    old_value: float
    new_value: float


@dataclass(frozen=True)
class Genome:
    """一个个体的一串浮点基因。

    Attributes:
        genes: float64 一维数组，长度 ``[GENOME_MIN_SLOTS, GENOME_MAX_SLOTS]``。
            设为只读视图，防止意外原地改（基因组是不可变值对象）。
        uid: 基因组实例编号（由模拟器的谱系计数器分配；仅用于缓存键，
            不参与任何数值计算）。
    """

    genes: np.ndarray
    uid: int = 0

    def __post_init__(self) -> None:
        genes = np.asarray(self.genes, dtype=np.float64)
        if genes.ndim != 1:
            raise ValueError(f"genome 须为一维数组，得到 shape={genes.shape}")
        object.__setattr__(self, "genes", genes)
        genes.setflags(write=False)

    def __len__(self) -> int:
        return int(self.genes.shape[0])

    def with_gene(self, slot: int, value: float) -> "Genome":
        """改一个槽位后的新基因组（只读原基因组；C3 v2 徽章重排用）。

        ``uid`` 必须保留：``ControllerCache`` 以 ``genome.uid`` 为键，
        丢失 uid 会让多个基因组命中同一个控制器（B17.1 实测）。
        """
        genes = np.array(self.genes, dtype=np.float64)
        genes[int(slot)] = float(value)
        return Genome(genes=genes, uid=self.uid)

    def __setstate__(self, state) -> None:
        """pickle 续跑后重新落只读闸（checkpoint 协议依赖）。"""
        self.__dict__.update(state)
        self.genes.setflags(write=False)

    def param(self, name: str) -> float:
        """按语义名读参数段原始基因值（未 squash）。"""
        return float(self.genes[PARAM_SLOTS[name]])

    def grn_genes(self) -> np.ndarray:
        """GRN 段基因（长度不足时右侧补零，恒为 42）。"""
        seg = self.genes[GRN_GENE_OFFSET:GRN_GENE_OFFSET + GRN_GENE_COUNT]
        if seg.shape[0] < GRN_GENE_COUNT:
            seg = np.concatenate(
                [seg, np.zeros(GRN_GENE_COUNT - seg.shape[0], dtype=np.float64)]
            )
        return np.array(seg, dtype=np.float64)

    def fingerprint(self) -> Tuple[float, ...]:
        """内容指纹（用于测试/日志；不用于热路径）。"""
        return tuple(float(x) for x in self.genes)


def random_genome(rng: np.random.Generator, uid: int = 0,
                  length: int = GENOME_NOMINAL_SLOTS) -> Genome:
    """随机初始基因组：标准正态，截断到 [-3, 3]。"""
    if not (GENOME_MIN_SLOTS <= length <= GENOME_MAX_SLOTS):
        raise ValueError(f"length 越界: {length}")
    genes = np.clip(rng.standard_normal(length), -3.0, 3.0)
    return Genome(genes=genes, uid=uid)


def mutate(genome: Genome, rng: np.random.Generator,
           uid: int = 0,
           evo: "EvolutionConfig" = None) -> Tuple[Genome, MutationRecord]:
    """对基因组做一次突变，返回新基因组与突变记录。

    RNG 消费顺序（写死，保确定性）：
    ① 抽突变类型（按 MUTATION_PROBS）→ ② 抽位置 → ③ 抽值（点突变为高斯
    噪声，插入为一次新的标准正态）。

    长度贴边时插入/删除自动退化为点突变（RNG 照常消费，结果仍确定）。
    """
    genes = genome.genes
    n = genes.shape[0]

    # ① 类型
    kind_roll = float(rng.random())
    if kind_roll < MUTATION_PROBS[0]:
        kind = MutationKind.POINT
    elif kind_roll < MUTATION_PROBS[0] + MUTATION_PROBS[1]:
        kind = MutationKind.INSERT
    else:
        kind = MutationKind.DELETE

    # ② 位置
    pos = int(rng.integers(0, n))

    # 长度边界 ⇒ 退化为点突变
    if kind == MutationKind.INSERT and n >= GENOME_MAX_SLOTS:
        kind = MutationKind.POINT
    elif kind == MutationKind.DELETE and n <= GENOME_MIN_SLOTS:
        kind = MutationKind.POINT

    # ③ 值 + 组装
    if kind == MutationKind.POINT:
        scale = POINT_MUTATION_SCALE if evo is None else evo.mutation_scale
        delta = float(rng.standard_normal()) * scale
        old = float(genes[pos])
        new = math.fsum([old, delta])
        new_genes = genes.copy()
        new_genes[pos] = new
        rec = MutationRecord(kind, pos, old, new)
    elif kind == MutationKind.INSERT:
        value = float(np.clip(rng.standard_normal(), -3.0, 3.0))
        new_genes = np.concatenate(
            [genes[:pos], np.array([value], dtype=np.float64), genes[pos:]]
        )
        rec = MutationRecord(kind, pos, float("nan"), value)
    else:  # DELETE
        old = float(genes[pos])
        new_genes = np.concatenate([genes[:pos], genes[pos + 1:]])
        rec = MutationRecord(kind, pos, old, float("nan"))

    return Genome(genes=new_genes, uid=uid), rec


def validate_genome(genome: Genome) -> List[str]:
    """基因组校验；返回错误列表（空 = 通过）。

    不变量：
    1. 长度在 ``[GENOME_MIN_SLOTS, GENOME_MAX_SLOTS]``；
    2. 全部有限（无 NaN/Inf）；
    3. 基因值在 [-10, 10]（ squash 前的粗护栏，防爆缸）。
    """
    errors: List[str] = []
    n = len(genome)
    if not (GENOME_MIN_SLOTS <= n <= GENOME_MAX_SLOTS):
        errors.append(f"length={n} 越界 [{GENOME_MIN_SLOTS}, {GENOME_MAX_SLOTS}]")
    if not bool(np.all(np.isfinite(genome.genes))):
        errors.append("存在非有限基因值（NaN/Inf）")
    if n and float(np.max(np.abs(genome.genes))) > 10.0:
        errors.append("存在 |gene| > 10 的基因值")
    return errors


# ── 发育期 squash 工具（develop.py 也用） ─────────────────────────────────

def sigmoid(x: float) -> float:
    """数值稳定的 sigmoid（标量）。"""
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def squash01(x: float) -> float:
    """原始基因 → [0,1]（发育读出的第一道变换）。"""
    return sigmoid(x)


# re-export 供测试与外部引用
__all__ = [
    "GENOME_NOMINAL_SLOTS", "GENOME_MIN_SLOTS", "GENOME_MAX_SLOTS",
    "PARAM_SLOT_COUNT", "GRN_GENE_OFFSET", "GRN_GENE_COUNT",
    "PARAM_SLOTS", "MutationKind", "MUTATION_PROBS", "POINT_MUTATION_SCALE",
    "MutationRecord", "Genome", "random_genome", "mutate", "validate_genome",
    "sigmoid", "squash01",
]
