"""通道注册表——每条通道**声明**自己的类型与进出账户，写能量时必须按声明登记。

四类（步A 通道枚举清单的分类，候选词表候砚裁）：

- `INJ` 注入：外部源进入某账户（光合、再生、果实蓄力）⇒ `src is None`
- `TRF` 转移：账户之间换主体 ⇒ `src`/`dst` 都是账户，能量域净额为 0
- `DIS` 散逸：转为系统外的热/不可用 ⇒ `dst is None`
- `ESC` 逃逸：离开可审计系统**且代码没记它为散逸/注入** ⇒ `dst is None`（A2 要闭合的就是它）

🔴 `TRF` 与 `ESC` 的区别是本次全部赌注所在：把逃逸登记成转移 ⇒ 恒等式自动成立 ⇒ 假绿
（镜 §〇："注入被吞进状态桶、热当尾差 ⇒ 恒等自动成立 ⇒ 恰好放过鱼令要堵的 :3283 洞"）。
因此 `channel()` 校验**调用点声明的 kind 必须与注册表一致**，且 `DIS`/`ESC` 必须给 `dst=None`；
不一致当场 `ChannelViolation`。这条校验使"串账"从"读代码才能发现"变成"跑就炸"。
"""

from __future__ import annotations

from dataclasses import dataclass

from .accounts import ACCOUNTS

INJ, TRF, DIS, ESC = "INJ", "TRF", "DIS", "ESC"
KINDS: tuple[str, ...] = (INJ, TRF, DIS, ESC)

#: 净额不进 `Σ` 的 kind（离开或进入系统）
EXTERNAL_DST: frozenset[str] = frozenset({DIS, ESC})


class ChannelViolation(RuntimeError):
    """通道未登记／kind 与注册表不符／进出账户与声明不符 ⇒ 当场炸，不容忍。"""


@dataclass(frozen=True)
class Channel:
    cid: str                      # "photo" / "digest" / "decay" / ...
    kind: str                     # INJ | TRF | DIS | ESC
    src: str | None               # 出账户；None = 系统外
    dst: str | None               # 入账户；None = 系统外
    site: str = ""                # 现役引擎行号锚（"E2@:4663"），便于结论回灌
    note: str = ""

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ChannelViolation(f"{self.cid}: kind={self.kind!r} 不在 {KINDS}")
        if self.kind == INJ and (self.src is not None or self.dst not in ACCOUNTS):
            raise ChannelViolation(f"{self.cid}: INJ 必须 src=None 且 dst 是已知账户")
        if self.kind == TRF and (self.src not in ACCOUNTS or self.dst not in ACCOUNTS):
            raise ChannelViolation(f"{self.cid}: TRF 必须两端都是已知账户")
        if self.kind in EXTERNAL_DST and (self.src not in ACCOUNTS or self.dst is not None):
            raise ChannelViolation(f"{self.cid}: {self.kind} 必须 src 是账户且 dst=None")


class ChannelRegistry:
    """通道定义的唯一真源。未登记的 `cid` 一律拒绝。"""

    def __init__(self, channels: tuple[Channel, ...] = ()) -> None:
        self._by_id: dict[str, Channel] = {}
        for ch in channels:
            self.add(ch)

    def add(self, ch: Channel) -> None:
        if ch.cid in self._by_id:
            raise ChannelViolation(f"重复登记通道 {ch.cid!r}")
        self._by_id[ch.cid] = ch

    def __getitem__(self, cid: str) -> Channel:
        try:
            return self._by_id[cid]
        except KeyError:
            raise ChannelViolation(
                f"未登记通道 {cid!r}（已登记：{sorted(self._by_id)}）"
            ) from None

    def cids(self) -> tuple[str, ...]:
        return tuple(sorted(self._by_id))

    def of_kind(self, kind: str) -> tuple[Channel, ...]:
        return tuple(c for c in self._by_id.values() if c.kind == kind)
