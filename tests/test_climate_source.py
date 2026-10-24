"""B3 气候源接口：逐位等价 + 反退化（假源）+ 契约等价。

三条各自钉一件事：
1. `EnvFieldSource` 是纯包装 ⇒ 与直调 `light`/`env_field` **逐位相同**；
   未接通道（降水/风）**显式抛 NotImplementedError**，不许 0 冒充。
2. **假源必须真的被用**：把 `EnvFieldSource` 换成 `_ConstantSource` 后，
   资源层的 `_capacity`/`_env_growth` 与温度都必须变——否则接口是摆设。
3. 契约形状：`isinstance(..., ClimateSource)`、`probe()` 报源名、`state/restore` 可用。
"""
from __future__ import annotations

import numpy as np
import pytest

from simulation.config import EnvFieldConfig, SimConfig
from simulation.sphere_engine import SphereEngine
from world.climate import ClimateSource, EnvFieldSource


class _ConstantSource:
    """测试假源：温度恒 20 ℃、生长因子恒 2、容量因子恒 3（与真源必然不同）。"""

    name = "constant"
    T_C = 20.0
    GF = 2.0
    CF = 3.0

    def __init__(self, env_field=None, light=None, *a, **k):  # noqa: ARG002
        self._n = light.world.n_cells if light is not None else 1

    def bootstrap(self, world):  # noqa: ARG002
        pass

    def refresh(self, tick):  # noqa: ARG002
        pass

    def growth_factor(self):
        return np.full(self._n, self.GF)

    def capacity_factor(self):
        return np.full(self._n, self.CF)

    def temperature_c(self, flat, tick):  # noqa: ARG002
        return np.full(len(flat), self.T_C)

    def precip_mm_yr(self):
        raise NotImplementedError("未接：假源不提供降水")

    def wind(self):
        raise NotImplementedError("未接：假源不提供风")

    def probe(self) -> dict:
        return {"climate_source_name": self.name}

    def state(self) -> dict:
        return {}

    def restore(self, payload):  # noqa: ARG002
        pass


def _cfg(seed: int = 5) -> SimConfig:
    cfg = SimConfig(seed=seed)
    cfg.world.rows, cfg.world.cols = 60, 120
    cfg.population.initial_count = 50
    cfg.env_field = EnvFieldConfig(enabled=True, water_sensitivity=1.0,
                                   cap_sensitivity=1.0)
    return cfg


def test_b3_envfield_source_is_bitwise_wrapper():
    """① 逐位等价 + 未接通道显式报错。"""
    eng = SphereEngine(_cfg())
    src = eng.climate
    assert src.name == "env_field"
    flat = eng._flat
    for tick in (0, 7, 2400):
        assert np.array_equal(src.temperature_c(flat, tick),
                              eng.light.temperature(flat, tick)), \
            f"tick={tick}：接口温度与直调 light.temperature 不逐位"
    assert np.array_equal(src.growth_factor(), eng.env_field.growth_factor())
    assert np.array_equal(src.capacity_factor(), eng.env_field.capacity_factor())
    with pytest.raises(NotImplementedError, match="未接"):
        src.precip_mm_yr()
    with pytest.raises(NotImplementedError, match="未接"):
        src.wind()


def test_b3_fake_source_is_actually_used(monkeypatch):
    """② 反退化（最重要）：假源的值必须真的进资源层与温度，而不是悄悄回落。"""
    import simulation.sphere_engine as se
    real = SphereEngine(_cfg())
    monkeypatch.setattr(se, "EnvFieldSource", _ConstantSource)
    fake = SphereEngine(_cfg())

    assert fake.climate.name == "constant"
    # 温度：假源恒 20 ℃；真源有纬度梯度
    t_fake = fake.climate.temperature_c(fake._flat, 0)
    t_real = real.climate.temperature_c(real._flat, 0)
    assert np.all(t_fake == _ConstantSource.T_C), "假源温度没被调用"
    assert float(t_real.std()) > 0.0, "真源温度竟然无梯度（装置问题）"
    # 资源层：GF=2 / CF=3 必须让容量与再生因子与真源不同
    assert not np.allclose(fake.resources._capacity, real.resources._capacity), \
        "假源 capacity_factor 没进资源层 ⇒ 插口是摆设"
    assert not np.allclose(fake.resources._env_growth, real.resources._env_growth), \
        "假源 growth_factor 没进资源层 ⇒ 插口是摆设"


def test_b3_contract_shape_and_probe():
    """③ 契约形状：Protocol 可 isinstance、probe 报源名、state/restore 可用。"""
    eng = SphereEngine(_cfg())
    assert isinstance(eng.climate, ClimateSource)
    assert isinstance(eng.climate, EnvFieldSource)
    assert eng.climate.probe()["climate_source_name"] == "env_field"
    assert eng.climate_probe()["climate_source_name"] == "env_field"
    assert eng.climate.state() == {}
    eng.climate.restore({})          # 静态源 no-op，不抛
