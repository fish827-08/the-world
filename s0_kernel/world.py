"""S0-kernel-a：通道驱动的迷你世界（**零引擎依赖**）。

目的只有一个——把「全局守恒断言」的**硬零门**放在一个"所有状态变化必须经通道登记"的
合成世界上，因为：

1. 现役引擎的埋点要碰 `simulation/sphere_engine.py` / `config.py`（轻舟文件面，ack 未到）；
2. 砚的 A2 量域口径未裁；
3. 硬零门本身与这两件事无关——它检验的是"**每一笔能量变动都能被追回一次登记**"。

因此本世界**不 import 任何 `simulation.*`**（`tests/test_s0_kernel.py` 有守卫测试）。
将来引擎埋点落地，是"把同一本账挂到引擎的 measure/apply 上"，不是改这里。

## 状态真值在数组里，不在账本里

`measure()` 把 `_e/_s/_g/_c/_f` 五组数组**重新求和**（`math.fsum`），这就是 A2 的实测侧；
`Books` 那份 `_ledger` 完全不知道数组长什么样。所以往数组上直接捅一刀（不登记）
⇒ 实测动了、账本没动 ⇒ `close()` 非零 ⇒ 门禁变红。这正是 `_audit_*` 抓不到的那类破坏。

## 精确 0.0 的前提

折算系数取 **2 的幂**（`eff=0.5`、`assim_frac=0.5`）⇒ 质量↔能量往返在二进制浮点下**逐位精确**，
`close()` 才敢要求"恰好 0.0"而不是"小于容差"。真实档位（`eat_efficiency=0.8` 那类）
落在容差档里，由 `tests` 单独标出，不混进硬零门。

## 不吃 RNG

事件用 `tick`/下标的整余数决定，**不调用任何随机源** ⇒ 同参数跑两次逐位相同，
也满足"新工具不消费 RNG"四律。
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass, field

from .accounts import ACCOUNTS, to_energy
from .books import Books
from .channels import Channel, ChannelRegistry, ChannelViolation


class ConservationViolation(RuntimeError):
    """守恒残差非零 ⇒ 有通道漏记／串账／旁路写。步A 门禁的失败形态。"""


# ── 通道定义（迷你世界用到的全部；(kind, src, dst) 由注册表锁死）──────
CHANNEL_DEFS: tuple[Channel, ...] = (
    Channel("photo", "INJ", None, "G", site="I1@:3283", note="光合注入（无物质预算）"),
    Channel("regrow", "INJ", None, "G", site="I3@:2790", note="资源再生（无预算）"),
    Channel("graze", "TRF", "G", "S", site="T1@:3449", note="啃食：植物池→胃"),
    Channel("assimilate", "TRF", "S", "E", site="T2@:3321", note="吸收：胃→活体"),
    Channel("digest_loss", "DIS", "S", None, site="D9@:3323", note="消化散逸（未吸收部分）"),
    Channel("metabolism", "DIS", "E", None, site="D1@:3368", note="基础代谢热"),
    Channel("locomotion", "DIS", "E", None, site="D2@:3373", note="移动耗散"),
    Channel("signal_cost", "DIS", "E", None, site="D4@:3633", note="信号发射费"),
    Channel("corpse_deposit", "TRF", "E", "C", site="T5@:1760", note="死亡投尸（归还部分）"),
    Channel("corpse_leak", "ESC", "E", None, site="E7@:1800", note="尸体未归还（逃逸）"),
    Channel("fruit_charge", "TRF", "E", "F", site="I4@:5957", note="果实蓄力（亲代出资）"),
    Channel("fruit_fall", "TRF", "F", "G", site="T8@:4669", note="果实落地入植物池"),
    Channel("seed_escape", "ESC", "F", None, site="E2@:4663", note="种子被吞走（逃逸）"),
)


def default_registry() -> ChannelRegistry:
    return ChannelRegistry(CHANNEL_DEFS)


@dataclass
class WorldOptions:
    n: int = 6                  # 个体数
    cells: int = 4              # 格数
    eff: float = 0.5            # 质量↔能量折算点（2 的幂 ⇒ 硬零档逐位精确）
    e0: float = 8.0             # 个体初始能量
    s0: float = 0.0             # 个体初始胃量（原生质量）
    g0: float = 20.0            # 每格初始植物量（原生质量）
    photo_energy: float = 0.25  # 每格每 tick 光合注入（能量域）
    regrow_every: int = 5       # 每 N tick 追加一次再生注入（0 = 关）
    regrow_energy: float = 0.5
    graze_energy: float = 1.0   # 单次啃食上限（能量域）
    assim_frac: float = 0.5     # 吸收比例（2 的幂）
    meta_energy: float = 0.125
    move_energy: float = 0.0625
    signal_energy: float = 0.125   # 2 的幂：非 dyadic 常量（如 0.1）会让"恰好 0.0"变成奢望
    corpse_give: float = 0.5    # 死亡时进尸骸池的比例（其余为逃逸）
    starve_energy: float = 0.5  # 低于此能量即死亡并投尸（2 的幂）
    #: 跨域桥系数来源注明（砚①：凡涉 S↔E 必须注明系数来源，不许隐性混加）
    eff_source: str = "s0_kernel 合成档；现役真源=organisms.eat_efficiency，桥在 sphere_engine:3321"
    fruit_every: int = 3        # 每 N tick 有一次果实蓄力→落地
    fruit_energy: float = 0.25
    seed_leak_frac: float = 0.25  # 果实落地前逃逸的比例（能量域，2 的幂）


class World:
    """五账户迷你世界。账户真值 = 下面五组数组之和；改写只能通过 `apply`（受令牌守卫）。"""

    def __init__(self, opt: WorldOptions | None = None) -> None:
        self.opt = opt or WorldOptions()
        o = self.opt
        if o.n < 1 or o.cells < 1:
            raise ValueError("n/cells 必须 >= 1")
        self._e = [float(o.e0) for _ in range(o.n)]                    # E energy
        self._s = [float(o.s0) for _ in range(o.n)]                    # S mass
        self._g = [float(o.g0) for _ in range(o.cells)]                # G mass
        self._c = [0.0 for _ in range(o.cells)]                        # C energy
        self._f = [0.0 for _ in range(o.cells)]                        # F energy
        self._alive = [True] * o.n
        self._holders: dict[str, int] = {}
        self._token = False
        self.tick = 0
        self.events: list[str] = []
        # A 路（PI 22:0x 落裁③）用：**本 tick 被写过的槽位**及其写前原值。
        #   记的是"位置 + 状态真值的原像"，**不是通道登记量** ⇒ 仍属实测侧；
        #   与 `Books._ledger` 没有数据通路（源码级守卫见 tests/test_s0_kernel.py）。
        self._touch: dict[tuple[str, int], float] = {}

    # ── A 路：事件集合实测 ─────────────────────────────────────────
    def begin_tick(self) -> None:
        """tick 起点：清空事件集合槽位表（全量核对侧 `Books.rollover()` 另调）。"""
        self._touch.clear()

    def eventset_delta(self, accs: tuple[str, ...]) -> float:
        """**只对本 tick 登记过写点的槽位**求实测 Δ（能量域）。

        = PI 落裁③"A 路：事件集合求和 + 每 N tick 全量核对"里的那个事件集合侧。
        🔴 它**看不见旁路写**（不经 `apply` 的 `world._g[0] += x` 不在槽位表里）
        ⇒ 全量重求和必须**周期性**跑（每 N tick）；变异档 M1 正是这条前提的实证。
        """
        tot = 0.0
        for (acc, slot), before in self._touch.items():
            if acc not in accs:
                continue
            store = {"E": self._e, "S": self._s, "G": self._g, "C": self._c,
                     "F": self._f}[acc]
            tot += to_energy(store[slot] - before, acc, self.opt.eff)
        return tot

    # ── 注入给 Books 的三个闭包 ────────────────────────────────────
    def books(self, reg: ChannelRegistry | None = None) -> Books:
        return Books(reg or default_registry(), measure=self.measure, apply=self.apply,
                     grant=self._grant, revoke=self._revoke, eff=self.opt.eff,
                     eff_source=self.opt.eff_source,
                     eventset=self.eventset_delta, tick_start=self.begin_tick)

    def measure(self) -> dict[str, float]:
        """A2 实测侧：对**真实数组**求和（与账本无任何数据通路）。"""
        return {
            "E": math.fsum(self._e), "S": math.fsum(self._s), "G": math.fsum(self._g),
            "C": math.fsum(self._c), "F": math.fsum(self._f),
        }

    def apply(self, acc: str, native_delta: float) -> None:
        if not self._token:
            raise ChannelViolation(
                f"账户 {acc!r} 被旁路改写（Δ={native_delta}）：所有能量/质量变动必须经 "
                f"Books.channel() 登记通道"
            )
        slot = self._holders.get(acc)
        if slot is None:
            raise ChannelViolation(
                f"{acc} 账户当前无持有者：调用世界前先用 holder() 声明这笔量落在哪个个体/格")
        store = {"E": self._e, "S": self._s, "G": self._g, "C": self._c, "F": self._f}[acc]
        # 写点登记：只记"哪个槽位、写前是多少"（原像），不记通道声明的量 ⇒ 实测侧
        self._touch.setdefault((acc, slot), store[slot])
        store[slot] += native_delta

    def _grant(self) -> None:
        self._token = True

    def _revoke(self) -> None:
        self._token = False

    @contextmanager
    def holder(self, **kw: int):
        """声明本笔通道动作的各账户落点，例如 `world.holder(G=cell, S=i)`。"""
        unknown = sorted(set(kw) - set(ACCOUNTS))
        if unknown:
            raise ChannelViolation(f"holder() 含未知账户 {unknown}")
        prev = dict(self._holders)
        self._holders.update(kw)
        try:
            yield self
        finally:
            self._holders = prev

    # ── 只读视图（测试/报告用；不供账本消费）───────────────────────
    @property
    def energy(self) -> list[float]:
        return list(self._e)

    @property
    def stomach(self) -> list[float]:
        return list(self._s)

    @property
    def grid(self) -> list[float]:
        return self._g

    @property
    def corpses(self) -> list[float]:
        return self._c

    @property
    def fruit(self) -> list[float]:
        return self._f

    @property
    def alive(self) -> list[bool]:
        return list(self._alive)

    # ── 一步：全部写能量动作都必须走 books.channel ──────────────────
    def step(self, books: Books) -> None:
        o = self.opt
        t = self.tick
        for cell in range(o.cells):
            with self.holder(G=cell):
                books.channel("photo", None, "G", o.photo_energy)
            if o.regrow_every and t and t % o.regrow_every == 0 and self._g[cell] > 0.0:
                with self.holder(G=cell):
                    books.channel("regrow", None, "G", o.regrow_energy)
        for i in range(o.n):
            if not self._alive[i]:
                continue
            cell = i % o.cells
            bite = min(o.graze_energy, self._g[cell] * o.eff)
            if bite > 0.0:
                with self.holder(G=cell, S=i):
                    books.channel("graze", "G", "S", bite)
            avail = self._s[i] * o.eff
            if avail > 0.0:
                assim = avail * o.assim_frac
                with self.holder(S=i, E=i):
                    books.channel("assimilate", "S", "E", assim)
                    books.channel("digest_loss", "S", None, avail - assim)
            meta = min(o.meta_energy, self._e[i])
            if meta > 0.0:
                with self.holder(E=i):
                    books.channel("metabolism", "E", None, meta)
            mv = o.move_energy if self._e[i] >= o.move_energy else 0.0
            if mv > 0.0:
                with self.holder(E=i):
                    books.channel("locomotion", "E", None, mv)
            if t % 3 == i % 3 and self._e[i] >= o.signal_energy:
                with self.holder(E=i):
                    books.channel("signal_cost", "E", None, o.signal_energy)
            if o.fruit_every and t % o.fruit_every == 0 and self._e[i] >= o.fruit_energy:
                with self.holder(E=i, F=cell):
                    books.channel("fruit_charge", "E", "F", o.fruit_energy)
                # F 是能量账户 ⇒ 余额本身就是能量域，不再乘 eff（乘了就是单位串账）
                leak = self._f[cell] * o.seed_leak_frac
                rest = self._f[cell] - leak
                with self.holder(F=cell):
                    if leak > 0.0:
                        books.channel("seed_escape", "F", None, leak)
                if rest > 0.0:
                    with self.holder(F=cell, G=cell):
                        books.channel("fruit_fall", "F", "G", rest)
            # 饿死：把残存能量按 `corpse_give` 投尸、其余记为逃逸 ⇒ 死细胞量**必须**两笔凑满
            if self._e[i] < o.starve_energy:
                self._alive[i] = False
                rem = self._e[i]
                if rem > 0.0:
                    with self.holder(E=i, C=cell):
                        books.channel("corpse_deposit", "E", "C", rem * o.corpse_give)
                        books.channel("corpse_leak", "E", None, rem * (1.0 - o.corpse_give))
        self.events.append(f"t{t}")
        self.tick += 1


# ── 门禁：逐 tick 判残差 ──────────────────────────────────────────
def check_tick(books: Books, domains: tuple[str, ...] = ("ALL", "E", "ENERGY_ONLY"),
               exact: bool = True, carriers: bool = True) -> dict[str, float]:
    """返回各量域残差；`exact` 档要求**恰好 0.0**，非零即 `ConservationViolation`。

    `carriers=True` 时连**分载体五列**一起判（砚①：合计过、分账不过 = 两笔反向漏账互抵，
    只看合计抓不到）。
    """
    resid = {d: books.close(d) for d in domains}
    bad = {d: v for d, v in resid.items() if (v != 0.0 if exact else abs(v) > 1e-9)}
    if carriers:
        for acc, row in books.carrier_report().items():
            v = row["residual"]
            if v != 0.0 if exact else abs(v) > 1e-9:
                bad[f"carrier_{acc}"] = v
    if bad:
        raise ConservationViolation(
            f"tick 残差非零 {bad}——有能量变动没追回一次通道登记"
        )
    return resid


def run(books: Books, world: World, ticks: int, exact: bool = True,
        domains: tuple[str, ...] = ("ALL", "E", "ENERGY_ONLY")) -> list[dict[str, float]]:
    """逐 tick 跑并判门；`books.begin_tick()` 在每 tick 头重置基线。"""
    out = []
    for _ in range(ticks):
        books.begin_tick()
        world.step(books)
        out.append(check_tick(books, domains, exact=exact))
    return out
