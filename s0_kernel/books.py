"""通道账（A2 的**记账侧**）：只登记、**绝不产生实测值**。

## 为什么不能像 `_audit_*` 那样自己维护余额

现役引擎的 `_audit_*` 三账（`:521`/`:5817`）由 `apply_oracle` 逐笔"扣减==入账"结构保证
首尾相等，`:5815-5819` 自述"抓不到守恒被破坏"。镜 §四 判它 **A1 构造恒等、不是守恒检验**。
如果本模块也像它一样在 `channel()` 里维护一份余额、再用同一份余额算"实测 ΔΣ"，
残差就**由构造保证为 0**——正是鱼令要堵的 :3283 洞会溜过去的假绿。

所以这里把两侧彻底分开：

    measured_delta_sigma(domain)  ← 只调 `measure()`（世界自己把真实数组求和）
    predicted_delta(domain)       ← 只读 `_ledger` + 注册表声明
    close(domain) = measured − predicted        # 唯一会合点

`measure` 是**注入的闭包**，本模块不知道它读什么；`apply` 也是注入的，且世界端有守卫令牌，
绕过 `channel()` 直接改余额 ⇒ `ChannelViolation`。

## 量域（步A 待定 1，归砚裁）——做成可算式

对任意账户子集 D，精确恒等式是

    ΔΣ_D = Σ_{dst∈D} in(ch) − Σ_{src∈D} out(ch)      （mass 账户折到能量域）

`D = 全集` ⇒ TRF 自动抵消，退化成 `INJ − DIS − ESC`；
`D = {E}` ⇒ TRF **不抵消**（进 E 的消化项与出 E 的投尸项都留着）⇒ 两量域差的是一堆实项。
⇒ 砚选哪个都能出数，且能把"为什么不同"指到具体通道上（`domain_gap_report()`）。

🔴 **所有登记量一律用能量域**，落账时按账户原生单位折算 ⇒ 单位串账（现役 R163
"1 单位尸体吐 3 倍能量"同族）在这里进不来。
"""

from __future__ import annotations

from collections.abc import Callable

from .accounts import ACCOUNTS, ENERGY_ACCOUNTS, MASS_ACCOUNTS, to_energy
from .channels import DIS, ESC, INJ, TRF, ChannelRegistry, ChannelViolation

#: 量域 ⇒ 成员集合。**砚 10-07 20:56 ① 已裁 `ALL` = A2 正式**（全载体域 ΔΣ(E+S+G+C+F)）；
#: 另两档保留为对照——`{E}` 对资源池侧漏账是**瞎的**（见 M1 的方向检验），留着当"为什么不能选它"的证据。
DOMAIN_ACCOUNTS: dict[str, tuple[str, ...]] = {
    "ALL": ACCOUNTS,                 # ← 砚① 已裁：A2 正式（分载体五列另见 carrier_report）
    "E": ("E",),                     # 只闭合活体能量（镜 §四 A2 原话的字面版，已被①否为唯一口径）
    "ENERGY_ONLY": ENERGY_ACCOUNTS,  # 只闭合能量域账户（跳过胃/资源池的质量停留）
}


def resolve_domain(domain: str) -> tuple[str, ...]:
    try:
        return DOMAIN_ACCOUNTS[domain]
    except KeyError:
        raise ChannelViolation(
            f"未知量域 {domain!r}（可选 {sorted(DOMAIN_ACCOUNTS)}）") from None


class Books:
    """通道账。`measure`/`apply`/`grant`/`revoke` 由世界注入，本类不持有状态真值。

    - `measure()` → `{账户: 原生单位余额}`（世界的真实数组求和，A2 的实测侧）
    - `apply(acc, native_delta)` → 唯一被允许改写余额的函数（受守卫令牌保护）
    - `grant()` / `revoke()` → 开/关写窗口（只在 `channel()` 内部成对使用）
    - `eff` = 质量↔能量唯一折算点（现役 `organisms.eat_efficiency`），必须为正
    """

    def __init__(self, reg: ChannelRegistry, *, measure: Callable[[], dict[str, float]],
                 apply: Callable[[str, float], None],
                 grant: Callable[[], None], revoke: Callable[[], None],
                 eff: float, eff_source: str = "",
                 eventset: Callable[[tuple[str, ...]], float] | None = None,
                 tick_start: Callable[[], None] | None = None) -> None:
        if not (eff > 0.0):
            raise ChannelViolation(f"eff 必须为正（质量↔能量折算点），收到 {eff}")
        if MASS_ACCOUNTS and not eff_source:
            raise ChannelViolation(
                "跨域桥系数必须注明来源（砚①：账本凡涉 S↔E 必须注明系数来源，不许隐性混加）")
        self._reg = reg
        self._measure = measure
        self._apply = apply
        self._grant = grant
        self._revoke = revoke
        self._eff = float(eff)
        self._eff_source = eff_source
        self._eventset = eventset
        self._tick_start = tick_start
        self._kind: dict[str, float] = {k: 0.0 for k in (INJ, TRF, DIS, ESC)}
        self._ledger: dict[str, float] = {}
        self._sigma0 = self._as_energy(self._snap())

    # ── 单位折算（只在实测侧使用；登记量恒为能量域）────────────────
    def _native(self, acc: str, energy_amt: float) -> float:
        return energy_amt / self._eff if acc in MASS_ACCOUNTS else energy_amt

    def _snap(self) -> dict[str, float]:
        snap = self._measure()
        unknown = sorted(set(snap) - set(ACCOUNTS))
        if unknown:
            raise ChannelViolation(f"measure() 返回未知账户 {unknown}")
        return {a: float(snap.get(a, 0.0)) for a in ACCOUNTS}

    def _as_energy(self, snap: dict[str, float]) -> dict[str, float]:
        return {a: to_energy(snap[a], a, self._eff) for a in snap}

    # ── 唯一写入面 ─────────────────────────────────────────────────
    def channel(self, cid: str, src: str | None, dst: str | None, amount: float) -> None:
        """登记一次通道动作并落到真实余额（`amount` 一律**能量域**）。

        调用点声明的 (src, dst) 必须与注册表一致 ⇒ "把逃逸说成转移"这类串账当场炸。
        """
        ch = self._reg[cid]
        if (src, dst) != (ch.src, ch.dst):
            raise ChannelViolation(
                f"{cid}: 调用点 src={src} dst={dst} 与注册声明 src={ch.src} dst={ch.dst} 不符"
                f"——串账是假绿的入口，不容忍"
            )
        if not (amount >= 0.0):
            raise ChannelViolation(f"{cid}: 量必须非负（方向由 kind 决定），收到 {amount}")
        self._kind[ch.kind] += amount
        self._ledger[cid] = self._ledger.get(cid, 0.0) + amount
        self._grant()
        try:
            if src is not None:
                self._apply(src, -self._native(src, amount))
            if dst is not None:
                self._apply(dst, self._native(dst, amount))
        finally:
            self._revoke()

    # ── 实测账（只读世界真值，不看 `_ledger`）──────────────────────
    def measured_delta_sigma(self, domain: str = "ALL") -> float:
        accs = resolve_domain(domain)
        cur = self._as_energy(self._snap())
        return sum(cur[a] for a in accs) - sum(self._sigma0[a] for a in accs)

    def measured_delta_per_account(self) -> dict[str, float]:
        """分载体实测账（能量当量）：mass 账户一律过 `eff_source` 注明的桥折算。"""
        cur = self._as_energy(self._snap())
        return {a: cur[a] - self._sigma0[a] for a in ACCOUNTS}

    # ── A 路实测（事件集合侧；PI 22:0x 落裁③）───────────────────────
    def measured_delta_sigma_eventset(self, domain: str = "ALL") -> float:
        """只对本 tick 的**写点集合**求实测 Δ（原像来自世界数组，不来自 `_ledger`）。

        用途：460,800 格正式档不可能每 tick 全量重求和（`e2a2193` 同族的实测口径），
        故逐 tick 判 `close_eventset()`，每 N tick 判一次 `eventset_agreement()`。
        🔴 未注入 `eventset` 闭包 ⇒ 当场报错，**不静默退化成全量**（那会让 A 路假绿）。
        """
        if self._eventset is None:
            raise ChannelViolation(
                "本 Books 未注入事件集合侧（A 路需要 eventset 闭包；不注入即不可用，"
                "禁止悄悄退回全量重求和）"
            )
        return self._eventset(resolve_domain(domain))

    def eventset_agreement(self, domain: str = "ALL") -> float:
        """**PI 落裁③ 硬门**：全量实测 − 事件集合实测，差必须 = 0（每 N tick 判一次）。

        非零 ⇒ 有写点没进事件集合（旁路写／apply 之外的直改），A 路当场不可信。
        """
        return self.measured_delta_sigma(domain) - self.measured_delta_sigma_eventset(domain)

    def close_eventset(self, domain: str = "ALL") -> float:
        """A 路的逐 tick 便宜判据：事件集合实测 − 通道账预测。"""
        return self.measured_delta_sigma_eventset(domain) - self.predicted_delta(domain)

    # ── 通道账（只读 `_ledger`，不看世界真值）─────────────────────
    def predicted_delta(self, domain: str = "ALL") -> float:
        accs = set(resolve_domain(domain))
        tot = 0.0
        for cid, amt in self._ledger.items():
            ch = self._reg[cid]
            if ch.dst in accs:
                tot += amt
            if ch.src in accs:
                tot -= amt
        return tot

    def from_channels(self, kind: str) -> float:
        if kind not in self._kind:
            raise ChannelViolation(f"未知 kind {kind!r}（可选 {sorted(self._kind)}）")
        return self._kind[kind]

    def heat(self) -> float:
        """热账 = Σ_DIS（砚⑥：散逸/逃逸**分列**，逃逸**不入热账**——吞掉逃逸 A2 就丢了最敏感的分账）。"""
        return self.from_channels(DIS)

    def escaped(self) -> float:
        """逃逸账 = Σ_ESC，与热账平行成文；步B 是否归并属设计决策，本包不归并。"""
        return self.from_channels(ESC)

    def per_channel(self) -> dict[str, float]:
        return dict(self._ledger)

    def predicted_delta_per_account(self) -> dict[str, float]:
        """分载体通道账（能量当量）：`dst=acc` 加、`src=acc` 减——与合计同表，防"合计对、分账错"。"""
        out = {a: 0.0 for a in ACCOUNTS}
        for cid, amt in self._ledger.items():
            ch = self._reg[cid]
            if ch.dst is not None:
                out[ch.dst] += amt
            if ch.src is not None:
                out[ch.src] -= amt
        return out

    # ── 唯一会合点 ─────────────────────────────────────────────────
    def close(self, domain: str = "ALL") -> float:
        """残差；合成闭环档必须**恰好 0.0**。非零 ⇒ 有通道漏记／串账／旁路写。"""
        return self.measured_delta_sigma(domain) - self.predicted_delta(domain)

    def carrier_report(self) -> dict[str, dict[str, float]]:
        """砚①·分载体五列 + 合计同表：每账户 `measured/predicted/residual`（能量当量）。

        合计过、分账不过 = 两笔反向漏账互相抵消——只看 `close("ALL")` 抓不到，
        所以每列都得单独为 0。
        """
        cur = self._as_energy(self._snap())
        pred = self.predicted_delta_per_account()
        out = {}
        for a in ACCOUNTS:
            m = cur[a] - self._sigma0[a]
            out[a] = {"measured": m, "predicted": pred[a], "residual": m - pred[a]}
        total_m = sum(v["measured"] for v in out.values())
        total_p = sum(v["predicted"] for v in out.values())
        out["TOTAL"] = {"measured": total_m, "predicted": total_p,
                        "residual": total_m - total_p}
        return out

    def meta(self) -> dict[str, object]:
        """manifest 用：跨域桥系数与其来源（砚①要求注明）、量域候选、通道面。"""
        return {"eat_efficiency": self._eff, "eff_source": self._eff_source,
                "domains": tuple(DOMAIN_ACCOUNTS), "channels": self._reg.cids()}

    def domain_gap_report(self) -> dict[str, float]:
        """各量域残差 + "为什么不同"的通道级差异项，供砚裁量域时直接指到通道。"""
        gap = {f"residual_{d}": self.close(d) for d in DOMAIN_ACCOUNTS}
        allc = set(resolve_domain("ALL"))
        eonly = set(resolve_domain("E"))
        for cid, amt in sorted(self._ledger.items()):
            ch = self._reg[cid]
            a = (ch.dst in allc) - (ch.src in allc)
            b = (ch.dst in eonly) - (ch.src in eonly)
            if a != b:
                gap[f"term__{cid}"] = amt * (b - a)
        return gap

    # ── tick 边界 ─────────────────────────────────────────────────
    def begin_tick(self) -> None:
        """tick 初：世界真值成为新基线，通道账清零，事件集合（写点表）清空。"""
        self._sigma0 = self._as_energy(self._snap())
        if self._tick_start is not None:
            self._tick_start()
        for k in self._kind:
            self._kind[k] = 0.0
        self._ledger.clear()

    @property
    def measured_sigma(self) -> dict[str, float]:
        return self._as_energy(self._snap())

    @property
    def registry(self) -> ChannelRegistry:
        return self._reg

    @property
    def eff(self) -> float:
        return self._eff
