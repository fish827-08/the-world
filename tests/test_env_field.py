"""纯斑块世界 ENV 场 —— 回归单测（设计稿 §十一 土地雷与守卫 / ENV-FIELD §二-5）。

覆盖（每条都断言**可观测行为**，不是"字段存在"）：
    E1 关档逐位等价：env off vs env on + 双 sens=0 ⇒ 30 tick 后 grid/capacity 逐位一致
    E2 面积权均值归一：ws=cs=1 ⇒ f_g/f_c 面积权均值 = 1（±1e-12）—— Σ容量守恒前提
    E3 确定性：同 seed 两次构造 h/w/acc 逐位一致；换 salt ⇒ h 变（独立流真的在动）
    E4 置换零模型：shuffle ⇒ H 多重集不变（排序后逐位相等）+ 空间粗糙度翻倍
    E5 守卫：env ∧ sim_core ⇒ NotImplementedError；env ∧ rd ⇒ ValueError（fail-loud）
    E6 路由树无环不变量：filled[receivers] < filled 严格成立；receivers=-1 恰为两极点行
    E7 sparse 显式退场：env on + sparse_fields ⇒ 实场 _lazy=False + env_note 留痕
    E8 因子出口：sens=0 ⇒ None（整块跳过）；runoff_gain↑ ⇒ w 逐点不减、峰值上升

设计原则（教训库）：默认关 = 旧行为逐位一致（E1 钉死）；每条只动一个参数、
断言"变化方向与量级"。
"""

from __future__ import annotations

import numpy as np
import pytest

from simulation.config import EnvFieldConfig, SimConfig
from simulation.sphere_engine import SphereEngine
from world.env_field import EnvField
from world.light_and_temperature import LightAndTemperature
from world.sphere_world import SphereWorld

_ROWS, _COLS = 60, 120          # 代码默认微型冒烟档（env 构造 ~60ms）


def _field(seed: int = 42, **env_kw) -> EnvField:
    world = SphereWorld(_ROWS, _COLS)
    lt = LightAndTemperature(world)
    cfg = EnvFieldConfig(enabled=True, **env_kw)
    return EnvField(world, lt, cfg, base_seed=seed)


def _engine(env_on: bool, sens: float = 0.0, ticks: int = 30) -> SphereEngine:
    cfg = SimConfig(seed=42)
    cfg.simulation.use_sim_core = False          # ENV 场走 Python 路径（构造期守卫同口径）
    if env_on:
        cfg.env_field.enabled = True
        cfg.env_field.water_sensitivity = sens
        cfg.env_field.cap_sensitivity = sens
    e = SphereEngine(cfg)
    for _ in range(ticks):
        e.step()
    return e


# ---------------- E1 关档逐位等价 ----------------

def test_e1_env_sens0_bit_equivalent() -> None:
    e_off = _engine(env_on=False)
    e_on = _engine(env_on=True, sens=0.0)
    assert np.array_equal(e_off.resources._grid, e_on.resources._grid)
    assert np.array_equal(e_off.resources._capacity, e_on.resources._capacity)


# ---------------- E2 面积权均值归一 ----------------

def test_e2_factor_awm_is_one() -> None:
    ef = _field(water_sensitivity=1.0, cap_sensitivity=1.0)
    areas = ef.world.cell_area(np.arange(ef.world.n_cells, dtype=np.int64))
    for f in (ef.growth_factor(), ef.capacity_factor()):
        assert f is not None
        awm = float(np.sum(f * areas) / np.sum(areas))
        assert abs(awm - 1.0) <= 1e-12, f"面积权均值 = {awm!r}（应为 1）"


# ---------------- E3 确定性 ----------------

def test_e3_determinism_and_salt() -> None:
    a = _field(seed=42)
    b = _field(seed=42)
    for attr in ("h", "w", "acc"):
        assert np.array_equal(getattr(a, attr), getattr(b, attr)), attr
    c = _field(seed=42, salt=1)
    assert not np.array_equal(a.h, c.h)


# ---------------- E4 置换零模型 ----------------

def test_e4_shuffle_preserves_histogram_breaks_structure() -> None:
    # ⚠️ 60×120 下 fBm 特征尺度 ≈2 格 ⇒ 粗糙度基准已接近白噪声（base/shuffled ≈ 1.03）
    #   ⇒ 结构判别放在 240×480 上做（ratio 实测 ≈1.65；阈值留 1.4）。
    world = SphereWorld(240, 480)
    lt = LightAndTemperature(world)
    base = EnvField(world, lt, EnvFieldConfig(enabled=True), base_seed=42)
    sh = EnvField(world, lt, EnvFieldConfig(enabled=True, shuffle=True), base_seed=42)
    assert np.array_equal(np.sort(base.h), np.sort(sh.h))   # 多重集（直方图）逐位不变

    def roughness(h: np.ndarray) -> float:
        nb = world._nb_table
        valid = nb >= 0
        d = np.abs(h[np.where(valid, nb, 0)] - h[:, None])
        return float(d[valid].mean())

    r_base, r_sh = roughness(base.h), roughness(sh.h)
    assert r_sh > 1.4 * r_base, f"粗糙度 {r_base:.4f} -> {r_sh:.4f}（置换应显著变粗）"


# ---------------- E5 守卫 ----------------

def test_e5_guards_fail_loud() -> None:
    cfg = SimConfig(seed=42)
    cfg.simulation.use_sim_core = True
    cfg.env_field.enabled = True
    with pytest.raises(NotImplementedError):
        SphereEngine(cfg)

    cfg2 = SimConfig(seed=42)
    cfg2.simulation.use_sim_core = False
    cfg2.env_field.enabled = True
    cfg2.resource_dynamics.enabled = True
    with pytest.raises(ValueError):
        SphereEngine(cfg2)


# ---------------- E6 路由树不变量 ----------------

def test_e6_receivers_strict_descent_and_outlets() -> None:
    ef = _field()
    recv, fl = ef.receivers, ef.filled
    m = recv >= 0
    assert (fl[recv[m]] < fl[m]).all(), "接收者链必须严格下降（无环的充分条件）"
    # 出水口 = 两极点行（其 `_nb_table` 全 -1 ⇒ 无有效邻元 ⇒ receiver=-1）
    assert int((~m).sum()) == 2 * _COLS


# ---------------- E7 sparse 显式退场 ----------------

def test_e7_sparse_explicit_retreat_note() -> None:
    cfg = SimConfig(seed=42)
    cfg.simulation.use_sim_core = False
    cfg.simulation.sparse_fields = True
    cfg.env_field.enabled = True
    e = SphereEngine(cfg)
    assert e.resources._lazy is False
    assert e.env_note and "resource_lazy=off" in e.env_note


# ---------------- E8 因子出口 ----------------

def test_e8_factor_gating_and_runoff_monotone() -> None:
    ef0 = _field()                                   # 双 sens=0
    assert ef0.growth_factor() is None
    assert ef0.capacity_factor() is None

    g0 = _field(water_sensitivity=1.0, runoff_gain=0.0)
    g2 = _field(water_sensitivity=1.0, runoff_gain=2.0)
    assert float(g2.w.max()) > float(g0.w.max())
    assert (g2.w >= g0.w - 1e-15).all()              # 汇流项只加分、不减分
