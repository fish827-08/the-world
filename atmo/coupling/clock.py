"""M2′ 时钟（INTERFACE §7.3 / 总审 OPEN-1）：tick 整数倍交换，零漂移。

```
k       = floor(dt_atmo / tick_seconds)
n_sub   = 1
dt_step = k * tick_seconds          # ≤ dt_atmo
time_s  = (t // k) * dt_step        # 整数倍 ⇒ 零漂移（不做浮点累加）
```

交换相位：世界 tick `t` 满足 `t % k == 0`（t=0 先装边界；t=k, 2k, … 交换）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class CouplingClock:
    tick_seconds: float
    dt_atmo: float
    step_mode: str = "tick_multiple"
    _k: int = field(init=False, repr=False)
    _dt_step: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        ts = float(self.tick_seconds)
        dt = float(self.dt_atmo)
        if not math.isfinite(ts) or ts <= 0.0:
            raise ValueError(f"tick_seconds 必须有限且 >0，实得 {self.tick_seconds!r}")
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError(f"dt_atmo 必须有限且 >0，实得 {self.dt_atmo!r}")
        if str(self.step_mode) == "tick_split":
            raise NotImplementedError(
                "step_mode='tick_split'（M1，每 tick 一步）v0 未实现："
                "INTERFACE OPEN-1 明确 M2′ 为默认、M1 只留兼容开关。")
        if str(self.step_mode) != "tick_multiple":
            raise ValueError(f"step_mode 非法：{self.step_mode!r}")
        k = int(math.floor(dt / ts))
        if k < 1:
            raise ValueError(
                f"dt_atmo={dt} < tick_seconds={ts}：M2′ 要求至少一个整 tick")
        object.__setattr__(self, "_k", k)
        object.__setattr__(self, "_dt_step", float(k) * ts)
        object.__setattr__(self, "tick_seconds", ts)
        object.__setattr__(self, "dt_atmo", dt)

    @property
    def k(self) -> int:
        """每 k 个世界 tick 交换一次。"""
        return self._k

    @property
    def dt_step(self) -> float:
        """实际推进步长（秒）＝ k·tick_seconds ≤ dt_atmo。"""
        return self._dt_step

    @property
    def n_sub(self) -> int:
        return 1

    @property
    def dt_remainder(self) -> float:
        """`dt_atmo − dt_step`（M2′ 舍掉的余量；0 表示精确整数倍）。"""
        return self.dt_atmo - self._dt_step

    def is_exchange(self, tick: int) -> bool:
        t = int(tick)
        if t < 0:
            raise ValueError(f"世界 tick 必须 ≥0，实得 {tick}")
        return t % self._k == 0

    def step_index(self, tick: int) -> int:
        t = int(tick)
        if t < 0:
            raise ValueError(f"世界 tick 必须 ≥0，实得 {tick}")
        return t // self._k

    def time_s(self, tick: int) -> float:
        """atmo 时钟（秒）：整数商乘 dt_step ⇒ 零漂移。"""
        return float(self.step_index(tick)) * self._dt_step

    def exchanges_per_year(self, year_s: float) -> int:
        y = float(year_s)
        if not math.isfinite(y) or y <= 0.0:
            raise ValueError(f"year_s 必须有限且 >0，实得 {year_s!r}")
        n = y / self._dt_step
        r = int(round(n))
        if abs(n - r) > 1e-9 * max(1.0, abs(n)):
            raise ValueError(
                f"year_s={y!r} 不是 dt_step={self._dt_step!r} 的整数倍"
                f"（{n!r} 步/年）—— 年长与耦合时钟口径不一致，拒绝四舍五入")
        return r

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "step_mode": str(self.step_mode),
            "tick_seconds": self.tick_seconds,
            "dt_atmo_s": self.dt_atmo,
            "k_ticks": self._k,
            "n_sub": self.n_sub,
            "dt_step_s": self._dt_step,
            "dt_remainder_s": self.dt_remainder,
        }
