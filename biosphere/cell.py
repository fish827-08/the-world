"""biosphere.cell：细胞尺度模块（W4；与个体尺度模块分开建）。

补充指令（2026-10-18）冻结的定义
--------------------------------
**细胞 = agent**：能粘（adhesion）、能双稳态分化（bistable
differentiation）、能动（每 tick ≤ 1 格）。**不是分子细胞**。
多细胞 = N 个 agent 粘成团，按团繁殖/死亡。本模块只管**细胞尺度**
（类型/栖息约束/分化开关）；个体尺度（基因组/发育/动作/裁定）在
genome/develop/action/resolve，团尺度在 groups.py。

栖息 = 物理约束（不写标签）
--------------------------
- 水生细胞：所在格 ``depth_m > 0``；
- 陆生细胞：所在格 ``soil_water ≥ world.soil_water_min``（世界侧阈值，
  从 world_adapter 读；真实世界耦合 A 线降水/土壤水）；
- 两栖细胞：两者皆可，但**各有代价**（代谢罚减半而非全免）；
- 未定准细胞（两个栖息基因都低于 0.5）：哪都能待，全程付费。

细胞类型由两个栖息基因决定（可演化 ⇒ 登陆/下水是演化事件）。

双稳态分化（迟滞开关）
----------------------
``diff_state ∈ {0, 1}``：邻体内个体数 ≥ ``DIFF_HI`` ⇒ 翻 1；
≤ ``DIFF_LO`` ⇒ 翻 0；中间保持（迟滞 = 双稳态）。state=1（体细胞态）
表达全额粘附，state=0（生殖/游动态）只表达 ``DIFF_ADHESION_SCALE``
比例的粘附 ⇒ 分化有机械后果（分工：体细胞粘团、生殖细胞游动）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from biosphere.develop import ControllerParams

# ── 细胞类型 ──────────────────────────────────────────────────────────────

#: 类型判定门限（两个栖息基因 ≥ 此值 = 该生境可达）
TYPE_GATE: float = 0.50

#: 占用不适配的代谢罚基数（× HABITAT_PENALTY 由 simulator 侧乘）
PENALTY_MISMATCH: float = 1.0     # 定准细胞待错生境
PENALTY_AMPHIBIOUS: float = 0.5   # 两栖"各有代价"
PENALTY_UNADAPTED: float = 1.0    # 未定准全程付费


@dataclass
class HabitatConfig:
    """栖息约束参数（M7 去写死：门限与罚值进世界族参数）。"""

    type_gate: float = TYPE_GATE
    penalty_mismatch: float = PENALTY_MISMATCH
    penalty_amphibious: float = PENALTY_AMPHIBIOUS
    penalty_unadapted: float = PENALTY_UNADAPTED


TYPE_AQUATIC = "aquatic"
TYPE_TERRESTRIAL = "terrestrial"
TYPE_AMPHIBIOUS = "amphibious"
TYPE_UNADAPTED = "unadapted"

# ── 双稳态分化（迟滞开关） ────────────────────────────────────────────────

#: 翻 1 / 翻 0 的邻体数门限（迟滞区间 = 双稳态）
DIFF_HI: int = 3
DIFF_LO: int = 1
#: state=0 时的粘附表达比例（生殖/游动态）
DIFF_ADHESION_SCALE: float = 0.5


def cell_type(c: ControllerParams,
              gate: float = TYPE_GATE) -> str:
    """两个栖息基因 ⇒ 细胞类型（可演化）。"""
    aq = c.habitat_aquatic >= gate
    terr = c.habitat_terrestrial >= gate
    if aq and terr:
        return TYPE_AMPHIBIOUS
    if aq:
        return TYPE_AQUATIC
    if terr:
        return TYPE_TERRESTRIAL
    return TYPE_UNADAPTED


def can_occupy(ctype: str, world, cell: int) -> bool:
    """栖息硬约束（MOVE 目标格准入）：按世界字段判定，不写标签。"""
    if ctype == TYPE_AQUATIC:
        return float(world.depth_m[cell]) > 0.0
    if ctype == TYPE_TERRESTRIAL:
        return float(world.soil_water[cell]) >= world.soil_water_min
    return True    # 两栖/未定准：物理上都能待（代价在代谢侧）


def occupancy_penalty(ctype: str, world, cell: int,
                      hab: "HabitatConfig" = None) -> float:
    """待在不适配生境的代谢罚基数（0 = 完全适配）。"""
    if hab is None:
        hab = HabitatConfig()
    if ctype == TYPE_AQUATIC:
        return 0.0 if float(world.depth_m[cell]) > 0.0 \
            else hab.penalty_mismatch
    if ctype == TYPE_TERRESTRIAL:
        if float(world.soil_water[cell]) >= world.soil_water_min:
            return 0.0
        return hab.penalty_mismatch
    if ctype == TYPE_AMPHIBIOUS:
        return hab.penalty_amphibious   # 两者皆可但各有代价
    return hab.penalty_unadapted        # 未定准：全程付费


def adhesion_expression(adhesion_gene_param: float, diff_state: int) -> float:
    """分化对粘附的表达调制（state=0 只表达一半 ⇒ 分工有机械后果）。"""
    scale = 1.0 if diff_state == 1 else DIFF_ADHESION_SCALE
    return adhesion_gene_param * scale


def update_differentiation(state: int, kin_count: int) -> int:
    """迟滞更新：邻体数 ≥ DIFF_HI ⇒ 翻 1；≤ DIFF_LO ⇒ 翻 0；中间保持。"""
    if state == 0 and kin_count >= DIFF_HI:
        return 1
    if state == 1 and kin_count <= DIFF_LO:
        return 0
    return state


@dataclass
class Differentiation:
    """双稳态分化开关（迟滞；输入 = 邻体数）。"""

    state: int = 0

    def update(self, kin_count: int) -> bool:
        """按邻体数更新状态；返回是否翻转。"""
        new = update_differentiation(self.state, kin_count)
        flipped = new != self.state
        self.state = new
        return flipped


def type_histogram(controllers: Dict[int, ControllerParams]) -> Dict[str, int]:
    """一批控制器的细胞类型直方（度量用）。"""
    out: Dict[str, int] = {TYPE_AQUATIC: 0, TYPE_TERRESTRIAL: 0,
                           TYPE_AMPHIBIOUS: 0, TYPE_UNADAPTED: 0}
    for c in controllers.values():
        out[cell_type(c)] += 1
    return out


__all__ = [
    "TYPE_GATE", "TYPE_AQUATIC", "TYPE_TERRESTRIAL", "TYPE_AMPHIBIOUS",
    "TYPE_UNADAPTED", "PENALTY_MISMATCH", "PENALTY_AMPHIBIOUS",
    "PENALTY_UNADAPTED", "DIFF_HI", "DIFF_LO", "DIFF_ADHESION_SCALE",
    "HabitatConfig",
    "cell_type", "can_occupy", "occupancy_penalty", "adhesion_expression",
    "update_differentiation", "Differentiation", "type_histogram",
]
