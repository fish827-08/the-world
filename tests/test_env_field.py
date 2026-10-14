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
    E9 去混杂选项默认关：uniform_base_t=False ⇒ temp 逐位等于旧公式（回滚点钉死）
    E10 去混杂选项生效：uniform_base_t=True ⇒ |corr(u,|lat|)| 显著消退 + 噪声流不动 + 面积权均温保持
    E11 引擎接线：uniform_base_t 经 SimConfig → SphereEngine → EnvField 送达（防静默退回默认）
    E12 自然剖面默认开（裁定一）：显式 False（classic）⇒ temp 逐位旧公式；双开 ⇒ fail-loud 双层
    E13 EBM 剖面物理锚点：赤道 ≈ +28.6°C / 极点 ≈ −23.0°C（真实地球带）＋对称＋单调＋有限
    E14 自然剖面生效 + 接线：temp 逐位 = EBM 剖面 − lapse×H；噪声流不动；引擎送达冒烟

设计原则（教训库）：默认关 = 旧行为逐位一致（E1 钉死）；每条只动一个参数、
断言"变化方向与量级"。
"""

from __future__ import annotations

import numpy as np
import pytest

from simulation.config import EnvFieldConfig, SimConfig
from simulation.sphere_engine import SphereEngine
from world.env_field import EnvField, _insolation_base_temperature
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


# ---------------- E9 去混杂选项 uniform_base_t（默认关 = 旧公式逐位）----------------

def test_e9_uniform_base_t_default_off_bit_equivalent() -> None:
    """uniform 默认关 + classic 档（显式）⇒ `temp` 逐位等于旧公式 `clip(base_t(φ) − lapse×H, −60)`。

    本测试同时钉两件事：uniform 选项默认关（不许静默启用）＋ classic 档逐位旧公式
    （C7 回滚点保真，数值逐位比较非 allclose）。2026-10-14 裁定一后默认档=insolation，
    故此处显式钉 classic 才走到该公式。
    """
    ef = _field(water_sensitivity=1.0, cap_sensitivity=1.0, insolation_profile=False)
    assert ef.cfg.uniform_base_t is False, "去混杂选项必须默认关"
    cells = np.arange(ef.world.n_cells, dtype=np.int64)
    ref = ef.lt.base_temperature(cells) - float(ef.cfg.lapse_c) * ef.h
    np.clip(ref, -60.0, None, out=ref)
    assert np.array_equal(ef.temp, ref), "默认档 temp 偏离旧公式（逐位）"
    # 类型守卫（fail-loud）：非 bool 值构造期报错，不许静默当 truthy
    with pytest.raises(AssertionError):
        EnvFieldConfig(uniform_base_t=1)


def test_e10_uniform_base_t_removes_latitude_imprint() -> None:
    """开档 ⇒ W 的纬度压印显著消退（治 ADR 教训 2 的 Magnus 混杂）。

    断言"变化方向与量级"（同 E4 风格）：classic 对照档 |corr(u,|lat|)| 强（≈0.9 级），
    开档后应塌到噪声水平（地形偶然残余）。同种子下噪声流逐位不动（h 同 ⇒ RNG 顺序零变化）。
    两档均钉 classic：uniform 只去梯度不动均值，前提是两档同一 base_t 出发。
    """
    off = _field(water_sensitivity=1.0, cap_sensitivity=1.0, insolation_profile=False)
    on = _field(water_sensitivity=1.0, cap_sensitivity=1.0,
                uniform_base_t=True, insolation_profile=False)
    world = off.world
    alat = np.abs(np.repeat(np.asarray(world._lat, dtype=np.float64), world.cols))

    def _corr(a, b) -> float:
        return float(np.corrcoef(a, b)[0, 1])

    c_off = _corr(off.w / off.w_ref, alat)
    c_on = _corr(on.w / on.w_ref, alat)
    assert abs(c_off) > 0.5, f"默认档应被纬度强压印（实测 {c_off:.4f}）"
    assert abs(c_on) < 0.25, f"去混杂档压印应显著消退（实测 {c_on:.4f}）"
    # RNG 纪律：噪声独立流消费顺序两档逐位一致（只动分子式，不动随机数）
    assert np.array_equal(off.h, on.h), "uniform_base_t 动到了噪声流（RNG 顺序契约）"
    # 只去梯度、不抬降全局：面积权均温逐位保持
    areas = world.cell_area(np.arange(world.n_cells, dtype=np.int64))

    def _aw(x) -> float:
        return float(np.sum(x * areas) / np.sum(areas))

    assert abs(_aw(off.temp) - _aw(on.temp)) <= 1e-9


def test_e11_engine_wiring_accepts_uniform_base_t() -> None:
    """引擎侧接线冒烟：`SimConfig.env_field.uniform_base_t=True` 能构造 EnvField 并跑 tick。

    开档不触碰默认禁用组合（`use_sim_core=False` 前提已由 E5 守卫覆盖），
    此处仅证"配置 → 引擎 → EnvField"这条链真的把开关送达（防接线漏传静默退回默认）。
    """
    cfg = SimConfig(seed=42)
    cfg.simulation.use_sim_core = False
    cfg.env_field.enabled = True
    cfg.env_field.insolation_profile = False   # 默认档已翻 insolation（裁定一）⇒ 显式关（互斥）
    cfg.env_field.uniform_base_t = True
    cfg.env_field.water_sensitivity = 1.0
    cfg.env_field.cap_sensitivity = 1.0
    e = SphereEngine(cfg)
    assert e.env_field is not None
    assert bool(e.env_field.cfg.uniform_base_t) is True, "开关未送达 EnvField"
    for _ in range(5):
        e.step()


# -------- E12–E14 自然剖面选项 insolation_profile（裁定一 2026-10-14：默认开 = 正式世界默认档）--------

def test_e12_insolation_profile_default_on_and_mutex() -> None:
    """默认档 = insolation（裁定一）；显式 False = classic 档逐位旧公式；互斥双层 fail-loud。"""
    ef = _field(water_sensitivity=1.0, cap_sensitivity=1.0)
    assert ef.cfg.insolation_profile is True, "自然剖面应为正式世界默认档（裁定一）"
    ef_old = _field(water_sensitivity=1.0, cap_sensitivity=1.0, insolation_profile=False)
    cells = np.arange(ef_old.world.n_cells, dtype=np.int64)
    ref = ef_old.lt.base_temperature(cells) - float(ef_old.cfg.lapse_c) * ef_old.h
    np.clip(ref, -60.0, None, out=ref)
    assert np.array_equal(ef_old.temp, ref), "classic 档 temp 偏离旧公式（逐位）"
    with pytest.raises(AssertionError):
        EnvFieldConfig(insolation_profile=1)                          # 类型守卫
    with pytest.raises(AssertionError):
        EnvFieldConfig(uniform_base_t=True, insolation_profile=True)  # 互斥（config 层）
    # 互斥第二层：post-init 赋值绕过 __post_init__ ⇒ 引擎侧兜底必须 fail-loud（防静默覆盖）
    cfg_bad = EnvFieldConfig(enabled=True)
    cfg_bad.uniform_base_t = True
    with pytest.raises(ValueError):
        EnvField(ef_old.world, ef_old.lt, cfg_bad, base_seed=42)


def test_e13_ebm_profile_physical_anchors() -> None:
    """EBM 年平气温剖面物理锚点（独立于 ENV 场构造）：真实地球带＋对称＋单调＋有限。

    锚点为**真实地球年平值带**（非本函数自洽值）：赤道 ≈ +28.6°C、极点 ≈ −23.0°C；
    容差取带（±2~4°C）——只钉"物理量级对"，不钉与实现同源的精确数（防循环验证）。
    """
    phi = np.radians(np.array([0.0, 30.0, 60.0, 90.0]))
    ts = _insolation_base_temperature(phi)
    assert np.all(np.isfinite(ts)), "剖面出现非有限值（极区 tan 奇点未钳住）"
    assert 26.0 < float(ts[0]) < 31.0, f"赤道年平气温应 ≈ +28.6°C（实测 {ts[0]:.2f}）"
    assert -27.0 < float(ts[3]) < -19.0, f"极点年平气温应 ≈ −23.0°C（实测 {ts[3]:.2f}）"
    # 南半球对称（年平只依赖 |φ|）
    ts_s = _insolation_base_temperature(-phi)
    assert np.allclose(ts, ts_s, atol=1e-9), "年平剖面在南北半球必须对称"
    # 单调下降（|φ| ∈ [0°, 80°] 网格逐点）
    grid = np.radians(np.linspace(0.0, 80.0, 81))
    vals = _insolation_base_temperature(grid)
    assert np.all(np.diff(vals) < 0.0), "年平气温应随 |φ| 单调下降"


def test_e14_insolation_profile_effect_and_wiring() -> None:
    """生效：`temp` 逐位 = EBM 剖面 − lapse×H；噪声流不动；引擎接线送达冒烟。"""
    off = _field(water_sensitivity=1.0, cap_sensitivity=1.0, insolation_profile=False)
    on = _field(water_sensitivity=1.0, cap_sensitivity=1.0, insolation_profile=True)
    world = off.world
    assert np.array_equal(off.h, on.h), "insolation_profile 动到了噪声流（RNG 顺序契约）"
    lat_row = np.asarray(world._lat, dtype=np.float64)
    ref = np.repeat(_insolation_base_temperature(lat_row), world.cols) \
        - float(on.cfg.lapse_c) * on.h
    np.clip(ref, -60.0, None, out=ref)
    assert np.array_equal(on.temp, ref), "开档 temp 未等于 EBM 剖面 − lapse×H（逐位）"
    assert not np.array_equal(off.temp, on.temp), "自然剖面应改变 temp（与线性 cosφ 不同）"
    # 引擎接线：字段送达 + 5 tick 冒烟（防接线漏传静默退回默认）
    cfg = SimConfig(seed=42)
    cfg.simulation.use_sim_core = False
    cfg.env_field.enabled = True
    cfg.env_field.insolation_profile = True
    cfg.env_field.water_sensitivity = 1.0
    cfg.env_field.cap_sensitivity = 1.0
    e = SphereEngine(cfg)
    assert e.env_field is not None
    assert bool(e.env_field.cfg.insolation_profile) is True, "开关未送达 EnvField"
    for _ in range(5):
        e.step()

