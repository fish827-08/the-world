"""`AtmoCoreLike` 协议（同 INTERFACE §8 签名）+ 真实扩展的薄适配器。

A3 只认这个协议：合成核（`synthetic.SyntheticCore`）与未来的 `atmo_core`
（经 `ModuleAtmoCore` 包装）可以零改动互换。
"""
from __future__ import annotations

from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class AtmoCoreLike(Protocol):
    """A1 动力核在 A3 眼里的样子。

    🔴 `synthetic` 必须如实声明：True = 合成场（只证链路，不作天气结论）；
    False = 真实动力核。桩（`AtmoCoreStub`）的**所有**方法都抛 NotImplementedError。
    """

    synthetic: bool

    def grid_spec(self) -> Mapping[str, Any]: ...

    def make_state(self) -> dict[str, np.ndarray]: ...

    def step(self, state: Mapping[str, np.ndarray],
             forcing: Mapping[str, Any], dt: float) -> Mapping[str, Any]: ...

    def export(self, state: Mapping[str, np.ndarray]) -> Mapping[str, Any]: ...

    def digest(self, state: Mapping[str, np.ndarray]) -> str: ...


_SUPPORTED_INTERFACE_VERSIONS = ("0.1.0-draft", "1.0.0", "1.0.1", "1.0.2")


class ModuleAtmoCore:
    """把 `import atmo_core as ac` 的**模块级函数**包成 `AtmoCoreLike`。

    只做签名适配与 fail-loud：模块缺哪个函数就报哪个，不伪造。
    A1 交付后 `Core=ModuleAtmoCore(atmo_core, cfg)` 即可接入；A3 测试**不依赖**它。
    """

    synthetic = False

    def __init__(self, module: Any, cfg: Any) -> None:
        for fn in ("grid_spec", "make_state", "step", "digest"):
            if not callable(getattr(module, fn, None)):
                raise NotImplementedError(
                    f"atmo_core 缺 §8 函数 {fn}() —— A1 尚未交付，拒绝用假数据顶替")
        self._m = module
        self._cfg = cfg
        self._grid: Mapping[str, Any] | None = None
        iface = getattr(module, "INTERFACE_VERSION", None)
        if iface is not None and str(iface) not in _SUPPORTED_INTERFACE_VERSIONS:
            raise ValueError(f"不认识的 atmo_core.INTERFACE_VERSION={iface!r}")
        self.interface_version = iface

    def grid_spec(self) -> Mapping[str, Any]:
        c = self._cfg
        if self._grid is None:
            self._grid = self._m.grid_spec(
                nlat=c.nlat, nlon=c.nlon, nlev=c.nlev, kind=c.kind)
        return self._grid

    def make_state(self) -> dict[str, np.ndarray]:
        c = self._cfg
        return dict(self._m.make_state(
            self.grid_spec(), T0=float(c.initial_t_k), ps0=float(c.initial_ps_pa)))

    def step(self, state, forcing, dt: float):
        # OPEN-5：原地更新，返回 diag（不返回 state）
        return self._m.step(state, forcing, dt=float(dt))

    def export(self, state):
        fn = getattr(self._m, "export", None)
        if not callable(fn):
            raise NotImplementedError(
                "atmo_core 缺 §8.7 的 export(st, g)（INTERFACE v1.0.2）——"
                "不自行按 T[7] 猜 t2m，宁可显式失败")
        return fn(state, self.grid_spec())

    def digest(self, state) -> str:
        return str(self._m.digest(state))
