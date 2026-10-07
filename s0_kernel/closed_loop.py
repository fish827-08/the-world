"""合成闭环档 + 变异清单：把「硬零门」和「它真的能抓坏」两件事一次说清。

三段式处置（镜 §四，PI 20:4x 照准）：

1. **合成闭环世界** ⇒ 硬零门（`close()` 恰好 `0.0`）+ 变异必须**变红**；
2. **旧装置** ⇒ 只出量级报告，不设阈值（`escape_probe` 已跑，见前置实测记录）；
3. **A2 硬门** ⇒ 只在步B 开启批生效（那时才有热账/营养池可闭）。

本模块只负责第 1 段。验收形按**砚 ⑦ 裁定（M4/M5 入验收栏）**：

> 注入已知量 ⇒ 恒等式差 = 注入量（**方向检验**），"不崩"不算过。

所以每条变异都带**逐量域的期望残差**，测试断言的是"差值等于我注入的那个数、且符号对"，
不是"抛了个异常"。`MUTATIONS` 里每条对应现役引擎一个真实写点 ⇒ 门禁不是摆设的证据。

🔴 期望表还顺带把**量域可见性**写成了可执行事实：往 G 上捅一刀，`ALL` 量域红、`{E}` 量域**瞎**。
这正是砚① 把 A2 量域裁成"全载体域 ΔΣ(E+S+G+C+F)"的理由——选 `{E}` 会漏掉资源池侧的漏账。
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from .books import Books
from .channels import ChannelViolation
from .world import World, WorldOptions, default_registry, run  # noqa: F401  (run 供测试直接取用)

#: 硬零档允许的最大二进制小数位（超过 ⇒ 加减要舍入，"恰好 0.0" 变假红）
_MAX_FRAC_BITS = 20


def _hard_zero_offenders(opt: WorldOptions) -> list[str]:
    """列出会让硬零档"假红"的参数：分母非 2 的幂、或小数位超过 `_MAX_FRAC_BITS`。

    浮点常量只要落在 2 的稀疏格子上（如 `0.1` 的分母其实是 2^55），与 `8.0` 相加就要舍入，
    `measured`（数组重求和）与 `predicted`（账本累加）就差了位宽那一档。那不是漏账，是位宽
    ——所以硬零档先把参数面锁死，别让人误读成"守恒被破坏了"。
    """
    bad = []
    for name, val in vars(opt).items():
        if not isinstance(val, float) or val == 0.0:
            continue
        den = Fraction(val).denominator
        bits = den.bit_length() - 1          # 值 = 分子 × 2^-bits
        if den & (den - 1) or bits > _MAX_FRAC_BITS:
            bad.append(f"{name}={val!r}（分母 2^{bits}）")
    return bad


def closed_preset(opt: WorldOptions | None = None) -> tuple[World, Books]:
    """硬零档：折算与比例全取 2 的幂 ⇒ 往返逐位精确；非硬零参数当场拒绝（附原因）。"""
    o = opt or WorldOptions()
    bad = _hard_zero_offenders(o)
    if bad:
        raise ChannelViolation(
            "硬零档参数不合位宽前提：" + ", ".join(bad)
            + f"（小数位需 ≤2^{_MAX_FRAC_BITS} 且分母为 2 的幂；真实档位请走 `tolerance_preset`）"
        )
    world = World(o)
    return world, world.books(default_registry())


def tolerance_preset(opt: WorldOptions) -> tuple[World, Books]:
    """真实档位（`eff=0.8` 那类非 2 幂系数）：**只能配容差判读**，不许当硬零门用。"""
    world = World(opt)
    return world, world.books(default_registry())


# ── 变异：每条模拟一类真实漏账，期望残差 = 注入量（砚⑦ 方向检验）────
def _m1_inject_unledgered(world: World, books: Books) -> None:
    """I3@:2790 资源再生**无物质预算**：植物池凭空多，账本不知道。"""
    world._g[0] += 1.0


def _m2_read_not_deduct(world: World, books: Books) -> None:
    """T7@:1797 / :4663 捕食"读不扣"的等价形态：真实扣了源，却只登记了流向。

    现役引擎是 `energy[prey]` 从不减、那份能量在 `keep=~dead` 压缩时蒸发 ⇒ 实测掉了、账本没记。
    这里用"未登记的真实扣除 + 已登记的等量流入"复现同一**漏记侧**。
    """
    with world.holder(G=0, S=0):
        books.channel("graze", "G", "S", 0.5)   # 账本：G→S 0.5（两侧都动，本身恒平）
    world._g[0] -= 0.5                           # 真实：G 又少 0.5（无通道）⇒ 净逃逸


def _m3_heat_unledgered(world: World, books: Books) -> None:
    """:3309 未吸收回流**无账户**：散逸没记 DIS，能量直接消失。"""
    world._e[0] -= 0.25


def _m4_escape_as_transfer(world: World, books: Books) -> None:
    """把逃逸登记成转移（串账）⇒ 恒等式会自动成立 = 假绿入口，必须当场炸。"""
    with world.holder(G=0, E=0):
        books.channel("graze", "G", "E", 0.5)    # 声明是 G→S


def _m5_unit_confusion(world: World, books: Books) -> None:
    """R163 单位串账：质量端按能量落账（少除一次 `eff`）。

    两端**一致地**用错系数，恒等式看不出来（错得自洽）⇒ 这里复现真实形态：一端折算、一端没折。
    """
    with world.holder(G=0, S=0):
        books.channel("graze", "G", "S", 0.5)
    world._s[0] += 0.5 / world.opt.eff - 0.5


def _m6_bypass_apply(world: World, books: Books) -> None:
    """绕过 `channel()` 直接调用 `apply()` ⇒ 守卫令牌必须拒绝。"""
    with world.holder(C=0):
        world.apply("C", 1.0)


@dataclass(frozen=True)
class Mutation:
    fn: object
    #: "residual" = 期望恒等式差成文；"raise" = 期望当场 `ChannelViolation`
    mode: str
    #: 砚⑦：注入已知量 ⇒ 逐量域期望差（能量当量，含符号）。`mode="raise"` 时为 None
    expect: dict[str, float] | None = None


#: `eff=0.5` 档的期望表：mass 账户（S/G）的扰动折能量域要 ×0.5，能量账户（E/C/F）不变。
MUTATIONS: dict[str, Mutation] = {
    "M1_inject_unledgered": Mutation(_m1_inject_unledgered, "residual",
                                     {"ALL": 0.5, "E": 0.0, "ENERGY_ONLY": 0.0}),
    "M2_read_not_deduct": Mutation(_m2_read_not_deduct, "residual",
                                   {"ALL": -0.25, "E": 0.0, "ENERGY_ONLY": 0.0}),
    "M3_heat_unledgered": Mutation(_m3_heat_unledgered, "residual",
                                   {"ALL": -0.25, "E": -0.25, "ENERGY_ONLY": -0.25}),
    "M4_escape_as_transfer": Mutation(_m4_escape_as_transfer, "raise"),
    "M5_unit_confusion": Mutation(_m5_unit_confusion, "residual",
                                  {"ALL": 0.25, "E": 0.0, "ENERGY_ONLY": 0.0}),
    "M6_bypass_apply": Mutation(_m6_bypass_apply, "raise"),
}

__all__ = ["ChannelViolation", "MUTATIONS", "Mutation", "World", "WorldOptions",
           "closed_preset", "default_registry", "run", "tolerance_preset"]
