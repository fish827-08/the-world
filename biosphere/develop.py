"""biosphere.develop：GRN 发育——基因组 → 控制器参数（B 线 MVP，W0-B）。

通俗理解
--------
基因组是一串数，生物体真正"用"的是一组控制器参数（感知增益、动作阈值、
生理常数……）。``develop`` 就是那个"从基因到表型"的函数：

1. **参数段直读**：前 22 个槽位按语义名读出，sigmoid 压到 [0,1]；
2. **GRN 调制**：第 22~63 槽 42 个基因组成一个 6 节点的基因调控网络
   （6×6 权重矩阵 + 6 个阈值）。网络从一个固定初态出发，被"母体因子"
   （参数段的粗读值）驱动迭代 16 步，终态节点活动再把上面的阈值/增益
   做非线性缩放——**这条映射本身由基因编码，可演化**。

确定性纪律
----------
``develop`` 是**纯函数**（无 RNG、无全局状态、无时间依赖）：同基因组
⇒ 控制器逐位一致（B4 有测试钉住）。运算全部走固定顺序的 NumPy 标量/小矩阵
操作，不依赖 dict/set 迭代序。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from biosphere.c3 import C3_BADGE_K
from biosphere.genome import (
    GRN_GENE_COUNT, Genome, PARAM_SLOTS, sigmoid,
)

# ── GRN 结构常量（冻结） ──────────────────────────────────────────────────

#: 调控节点数（权重矩阵 N×N + N 阈值 = N²+N 基因；N=6 ⇒ 42）
GRN_NODES: int = 6
#: 发育迭代步数（固定）
GRN_STEPS: int = 16
#: 权重矩阵缩放（W = GRN_W_SCALE * tanh(genes)）
GRN_W_SCALE: float = 2.0
#: 母体因子注入强度
GRN_INPUT_SCALE: float = 0.25

assert GRN_NODES * GRN_NODES + GRN_NODES == GRN_GENE_COUNT, "GRN 基因数须与节点数联动"

# ── 控制器参数（表型） ────────────────────────────────────────────────────


@dataclass(frozen=True)
class ControllerParams:
    """发育产物：一个个体本生命周期内使用的全部控制器参数。

    分四组：
    - 感知增益（0~2）：读世界各通道的权重；
    - 动作倾向（0~1）：各动作的基础得分权重；
    - 生理常数：代谢/移动成本/寿命/繁殖门槛；
    - 栖息与粘附：水生/陆生适应、粘附、防御。
    """

    # 感知增益
    sense_food: float
    sense_threat: float
    sense_kin: float
    sense_signal: float
    # 动作倾向
    move_bias: float
    eat_greed: float
    attack_drive: float
    share_drive: float
    signal_drive: float
    repro_drive: float
    # 生理
    metabolism: float        # 每 tick 基础能耗倍率（0.5~2.0）
    move_cost_mult: float    # 移动能耗乘子（0.5~2.0）
    lifespan: float          # 年龄上限（200~2000 tick）
    repro_thresh: float      # 繁殖能量门槛（40~120）
    # 栖息 / 社交
    habitat_aquatic: float   # 水生适应（0~1）
    habitat_terrestrial: float  # 陆生适应（0~1）
    adhesion: float          # 粘附（0~1）
    defense: float           # 防御（0~1）
    # 量
    eat_amount: float        # 单次进食量（0.2~1.0）
    signal_payload: float    # 单次信号强度（0~1）
    give_up: float           # 失望阈值（3~20）
    persistence: float       # 赶路惯性（0~1）
    # ── C3 v2（B14.3；徽章含义为空；带默认值 ⇒ 旧构造不炸） ──
    badge: int = 0                # 徽章类（0..K-1；连续基因离散化）
    emit_rate: float = 0.0        # 信号发射率（0~1）
    strength: float = 0.0         # 强度（0~1；打斗威力加成）
    response: Tuple[float, ...] = (0.0, 0.0, 0.0, 0.0)  # 各族徽章反应（signed）
    learning_rate: float = 0.0    # 记忆学习率（0 = 纯演化）

    def as_dict(self) -> Dict[str, float]:
        """定长 dict（键序固定；JSON 序列化用）。

        C3 v2 的 ``response`` 元组拆成 ``response_0..K-1``（JSON 友好）。
        """
        out: Dict[str, float] = {}
        for name in _FIELD_ORDER:
            val = getattr(self, name)
            if name == "response":
                for k, v in enumerate(val):
                    out[f"response_{k}"] = float(v)
            else:
                out[name] = float(val)
        return out


_FIELD_ORDER: Tuple[str, ...] = tuple(
    f for f in ControllerParams.__dataclass_fields__  # type: ignore[union-attr]
)


# ── GRN 解码 ──────────────────────────────────────────────────────────────

def _decode_grn(genome: Genome) -> Tuple[np.ndarray, np.ndarray]:
    """42 个 GRN 基因 → (W (6,6), thresholds (6,))。

    W = GRN_W_SCALE * tanh(前 36 个)，thresholds = tanh(后 6 个)。
    """
    g = genome.grn_genes()
    w = GRN_W_SCALE * np.tanh(g[: GRN_NODES * GRN_NODES]).reshape(GRN_NODES, GRN_NODES)
    t = np.tanh(g[GRN_NODES * GRN_NODES:])
    return w, t


def _run_grn(w: np.ndarray, t: np.ndarray, maternal: np.ndarray) -> np.ndarray:
    """GRN 发育动力学：固定初态 + 母体因子驱动，迭代 GRN_STEPS 步。

    s ← clip(W·s + t + GRN_INPUT_SCALE·maternal, 0, 1)
    初态 s0 = 0.5·1（固定常数，不占基因——"受精卵"环境全员相同）。
    """
    s = np.full(GRN_NODES, 0.5, dtype=np.float64)
    for _ in range(GRN_STEPS):
        s = np.clip(w @ s + t + GRN_INPUT_SCALE * maternal, 0.0, 1.0)
    return s


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else (hi if x > hi else x)


def develop(genome: Genome) -> ControllerParams:
    """基因组 → 控制器参数（纯函数；B4 钉住确定性）。"""
    g = genome.genes

    # ① 参数段直读（sigmoid 到 [0,1]）
    raw = {name: sigmoid(float(g[idx])) for name, idx in PARAM_SLOTS.items()}

    # ② GRN 调制：母体因子 = 前 4 个感知基因的粗读值（固定取法）
    maternal = np.array(
        [raw["sense_food"], raw["sense_threat"], raw["sense_kin"], raw["sense_signal"]]
        + [0.0, 0.0],
        dtype=np.float64,
    )
    w, t = _decode_grn(genome)
    s = _run_grn(w, t, maternal)

    # ③ 读出：GRN 节点活动对阈值/增益做 [0.5, 1.5] 的非线性缩放
    mod = 0.5 + s  # ∈ [0.5, 1.5]

    sense_food = _clamp(2.0 * raw["sense_food"] * mod[0], 0.0, 2.0)
    sense_threat = _clamp(2.0 * raw["sense_threat"] * mod[1], 0.0, 2.0)
    sense_kin = _clamp(2.0 * raw["sense_kin"] * mod[2], 0.0, 2.0)
    sense_signal = _clamp(2.0 * raw["sense_signal"] * mod[3], 0.0, 2.0)

    move_bias = _clamp(raw["move_bias"] * mod[4], 0.0, 1.0)
    eat_greed = _clamp(raw["eat_greed"] * mod[5], 0.0, 1.0)
    attack_drive = _clamp(raw["attack_drive"] * (2.0 - mod[0]), 0.0, 1.0)
    share_drive = _clamp(raw["share_drive"] * mod[2], 0.0, 1.0)
    signal_drive = _clamp(raw["signal_drive"] * mod[3], 0.0, 1.0)
    repro_drive = _clamp(raw["repro_drive"] * mod[4], 0.0, 1.0)

    metabolism = _clamp(0.5 + 1.5 * raw["metabolism_base"], 0.5, 2.0)
    move_cost_mult = _clamp(0.5 + 1.5 * raw["move_cost_mult"], 0.5, 2.0)
    lifespan = _clamp(200.0 + 1800.0 * raw["lifespan_gene"], 200.0, 2000.0)
    repro_thresh = _clamp(40.0 + 80.0 * raw["repro_thresh"], 40.0, 120.0)

    # ④ C3 v2 读出（B14.3）：徽章 = sigmoid(gene)×K 离散化；反应 =  signed
    badge_k = C3_BADGE_K
    badge = min(badge_k - 1, int(sigmoid(float(g[PARAM_SLOTS["badge"]])) * badge_k))
    emit_rate = _clamp(raw["emit_rate"], 0.0, 1.0)
    strength = _clamp(raw["strength"] * mod[4], 0.0, 1.0)
    response = tuple(
        _clamp(2.0 * raw[f"response_{k}"] - 1.0, -1.0, 1.0)
        for k in range(badge_k)
    )
    learning_rate = _clamp(raw["learning_rate"], 0.0, 1.0)

    return ControllerParams(
        sense_food=sense_food,
        sense_threat=sense_threat,
        sense_kin=sense_kin,
        sense_signal=sense_signal,
        move_bias=move_bias,
        eat_greed=eat_greed,
        attack_drive=attack_drive,
        share_drive=share_drive,
        signal_drive=signal_drive,
        repro_drive=repro_drive,
        metabolism=metabolism,
        move_cost_mult=move_cost_mult,
        lifespan=lifespan,
        repro_thresh=repro_thresh,
        habitat_aquatic=_clamp(raw["habitat_aquatic"], 0.0, 1.0),
        habitat_terrestrial=_clamp(raw["habitat_terrestrial"], 0.0, 1.0),
        adhesion=_clamp(raw["adhesion"] * mod[2], 0.0, 1.0),
        defense=_clamp(raw["defense"] * (2.0 - mod[1]), 0.0, 1.0),
        eat_amount=_clamp(0.2 + 0.8 * raw["eat_amount"], 0.2, 1.0),
        signal_payload=_clamp(raw["signal_payload"], 0.0, 1.0),
        give_up=_clamp(3.0 + 17.0 * raw["give_up"], 3.0, 20.0),
        persistence=_clamp(raw["persistence"] * mod[5], 0.0, 1.0),
        badge=badge,
        emit_rate=emit_rate,
        strength=strength,
        response=response,
        learning_rate=learning_rate,
    )


class ControllerCache:
    """uid → ControllerParams 缓存（个体基因组创建后不变，只算一次）。"""

    def __init__(self) -> None:
        self._cache: Dict[int, ControllerParams] = {}

    def get(self, genome: Genome) -> ControllerParams:
        hit = self._cache.get(genome.uid)
        if hit is None:
            hit = develop(genome)
            self._cache[genome.uid] = hit
        return hit

    def __len__(self) -> int:
        return len(self._cache)


__all__ = [
    "GRN_NODES", "GRN_STEPS", "ControllerParams", "develop", "ControllerCache",
]
