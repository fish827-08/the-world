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
    E15 C0b 关档逐位钉死：既有层组合 digest 不变 + 新层全 None + RNG 消费数不变
    E16 C0b 开档物理锚：面积权陆比落目标（§5.2 判据）+ "极海"失真消失
    E17 C0b 三条守恒不变量：①海陆面积比 ②流域闭合（含最大湖单流域）③计数与掩码互证
    E18 C0b 表面类型层齐全：海/陆/大陆架/河/湖均有格 + 互斥 + 与掩码逐位一致
    E19 C0b 零 RNG：开档 `h` 逐位不变 + 同 seed 可复现 + 换陆比不动噪声流
    E20 C0b 参数域断言 + 类型守卫（fail-loud，不许静默降级）
    E21 C0b 组合闸：c0b ∧ sim_core ⇒ NotImplementedError；c0b ∧ rd ⇒ ValueError
    E22 C0b 旧档回退：无 c0b_* 键 ⇒ 默认关（R233 T-F 家族）；带键 ⇒ 逐字段还原

设计原则（教训库）：默认关 = 旧行为逐位一致（E1 钉死）；每条只动一个参数、
断言"变化方向与量级"。
"""

from __future__ import annotations

import numpy as np
import pytest

from simulation.config import EnvFieldConfig, SimConfig
from simulation.sphere_engine import SphereEngine
from tools.env_field_digest import SCALARS, digest_layers, layer_dict
from world.env_field import (
    EnvField, SURFACE_LAND, SURFACE_LAKE_FRESH, SURFACE_LAKE_SALT, SURFACE_OCEAN,
    SURFACE_RIVER, SURFACE_SHELF, _insolation_base_temperature,
)
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


# ---------------- E15–E23 C0b（真实海拔 + 海平面 ⇒ 海/湖/河/陆表面层）----------------
#   实施依据：`docs/技术文档/技术-动态天气路线评估与选型-20261014.md` §5.2。
#   验收层级：场级、秒级（构造期静态 ⇒ 不需机时报批）。

#: 关档（`c0b_enabled=False`）60×120/seed42 既有层组合 digest（C7 回滚点；
#:   口径 = `tools/env_field_digest.py` 模块头：图层固定序 + 标量 "%.17g"）
_C0B_OFF_DIGEST = {
    ("classic", 0.0): "ae6ce6c912762aa9f06419300c17783dc4c3f499920d8d764825fb59ad5b83a1",
    ("classic", 1.0): "c669b0c231051336dbb583438224710a86050e05b64219822e3d66cf55cb2a2d",
    ("default", 0.0): "3cfd06bdb516565a8b5d58848479af7036a613f1330d44a4f9d592847ea9794e",
    ("default", 1.0): "b3576603217c6c0851c1d6b0249be816ec74973038abb6eb027b21d7cd3a0983",
}

#: C0b 开档专属属性（关档必须全 None ⇒ "不进任何计算"的可观测证据）
_C0B_ATTRS = ("h_m", "sea_level_m", "ocean_mask", "land_mask", "shelf_mask",
              "lake_mask", "fresh_lake_mask", "salt_lake_mask", "river_mask_legacy",
              "surface_type", "lake_count", "max_lake_cells", "max_lake_root",
              "max_lake_basin_cells", "conservation")


def _digest_of(ef: EnvField) -> str:
    return digest_layers(layer_dict(ef), {n: getattr(ef, n) for n in SCALARS})


def test_e15_c0b_default_off_bit_equivalent() -> None:
    """C0b 默认关 ⇒ 既有全部层组合 digest 逐位钉死 + 新层全 None + 引擎 RNG 消费数不变。"""
    assert EnvFieldConfig().c0b_enabled is False, "C0b 必须默认关"
    for mode, insolation in (("classic", False), ("default", True)):
        for sens in (0.0, 1.0):
            ef = _field(seed=42, insolation_profile=insolation,
                        water_sensitivity=sens, cap_sensitivity=sens)
            assert _digest_of(ef) == _C0B_OFF_DIGEST[(mode, sens)], (mode, sens)
            for attr in _C0B_ATTRS:
                assert getattr(ef, attr) is None, f"关档 {attr} 应 None"
            assert "c0b_on" not in ef.probe(), "关档 probe 不得新增键（旧 manifest 逐位不变）"
    a = SimConfig(seed=42)
    a.simulation.use_sim_core = False
    b = SimConfig(seed=42)
    b.simulation.use_sim_core = False
    b.env_field.enabled = True
    b.env_field.c0b_enabled = True
    assert SphereEngine(a).rng.draws == SphereEngine(b).rng.draws, \
        "C0b 新增 RNG 消费（地形走独立流 ⇒ 引擎序列必须不动）"


def test_e16_c0b_land_ratio_and_polar_sea_gone() -> None:
    """开档物理锚：面积权陆比落目标（±0.5%）+ "极海"失真消失 + 真实递减率。"""
    off = _field(seed=42)                                   # 关档：极点 = 唯一出水口
    on = _field(seed=42, c0b_enabled=True)
    cons = on.conservation
    assert abs(cons["land_area_frac"] - 0.30) <= 0.005, cons["land_area_frac"]
    assert abs(cons["ocean_area_frac"] - 0.70) <= 0.005, cons["ocean_area_frac"]
    assert bool(on.ocean_mask.any()) and bool(on.land_mask.any())
    # 🔴 极海失真消失：关档 seed42 被抬成"极海"的格 60×120 = 40.0%（s2 = 50.8%）
    assert off.flood_filled > 1000 and on.flood_filled * 20 < off.flood_filled, \
        f"flood_filled {off.flood_filled} -> {on.flood_filled}（应掉两量级以上）"
    # 开档被抬升的格全部在陆上（海格自身不填洼）⇒ 湖格数 == flood_filled
    assert int(on.lake_mask.sum()) == on.flood_filled
    # 真实递减率 Γ=6.5 ℃/km × 真实海拔（替代 lapse_c=20 的现象学常数）
    lat_row = np.asarray(on.world._lat, dtype=np.float64)
    ref = np.repeat(_insolation_base_temperature(lat_row), on.world.cols) \
        - (6.5 / 1000.0) * on.h_m
    np.clip(ref, -60.0, None, out=ref)
    assert np.array_equal(on.temp, ref), "开档 temp 未等于 EBM 剖面 − 6.5℃/km × h_m"
    assert abs(on.h_m.max() - 4500.0) <= 1e-9, "h_m 峰值应落在 peak_altitude_m"


def test_e17_c0b_conservation_three_invariants() -> None:
    """§5.2 三条守恒不变量（构造期一次核账；含最大湖单流域闭合）。"""
    ef = _field(seed=42, c0b_enabled=True)
    cons = ef.conservation
    # ①a 海陆面积比互补恒等（分划；①b 目标偏差已由 E16 ±0.5% 把住）
    assert cons["area_complement_resid"] <= 1e-12
    # ② 流域闭合：Σyield == Σacc[终点]（bincount 独立重算 vs 顺序累加）
    assert cons["closure_resid"] <= 1e-9 * max(1.0, cons["yield_sum"])
    # ②b 记账：陆地产水 == 入海径流 + 内流终点蓄
    assert cons["budget_resid"] <= 1e-9 * max(1.0, cons["land_yield"])
    # ③ surface_type 计数与掩码逐位互证（独立重算）
    counts = np.bincount(ef.surface_type, minlength=6)
    assert int(counts[SURFACE_OCEAN] + counts[SURFACE_SHELF]) == int(ef.ocean_mask.sum())
    assert int(counts[SURFACE_SHELF]) == int(ef.shelf_mask.sum())
    assert int(counts[SURFACE_LAKE_FRESH]) == int(ef.fresh_lake_mask.sum())
    assert int(counts[SURFACE_LAKE_SALT]) == int(ef.salt_lake_mask.sum())
    assert int(counts[SURFACE_RIVER]) == int(ef.river_mask.sum())
    assert int(counts.sum()) == ef.surface_type.size
    # ③b 大陆架 ⊆ 海；海格自身即出水口（receivers == -1）
    assert not bool((ef.shelf_mask & ~ef.ocean_mask).any())
    assert bool((ef.receivers[ef.ocean_mask] == -1).all())

    world = SphereWorld(240, 480)
    lt = LightAndTemperature(world)
    big = EnvField(world, lt, EnvFieldConfig(enabled=True, c0b_enabled=True),
                   base_seed=42)
    if big.max_lake_cells > 0:                               # 最大湖单流域闭合
        c2 = big.conservation
        assert abs(c2["max_lake_basin_yield"] - c2["max_lake_basin_acc"]) \
            <= 1e-9 * max(1.0, c2["max_lake_basin_acc"])
        assert c2["max_lake_basin_cells"] >= big.max_lake_cells


def test_e18_c0b_surface_labels_present_and_exclusive() -> None:
    """表面类型层齐全 + 互斥 + 新河层是旧 P98 读数层的陆地子集。"""
    world = SphereWorld(240, 480)
    lt = LightAndTemperature(world)
    ef = EnvField(world, lt, EnvFieldConfig(enabled=True, c0b_enabled=True),
                  base_seed=42)
    st = ef.surface_type
    for code, name in ((SURFACE_OCEAN, "海"), (SURFACE_SHELF, "大陆架"),
                       (SURFACE_LAND, "陆"), (SURFACE_RIVER, "河"),
                       (SURFACE_LAKE_FRESH, "淡水湖")):
        assert int((st == code).sum()) > 0, f"{name} 标签为空"
    assert ef.lake_count > 0 and ef.max_lake_cells > 0
    # 互斥（surface_type 每格只属一类的前提）
    assert not bool((ef.river_mask & ef.lake_mask).any())
    assert not bool((ef.river_mask & ef.ocean_mask).any())
    assert not bool((ef.lake_mask & ef.ocean_mask).any())
    assert not bool((ef.salt_lake_mask & ef.fresh_lake_mask).any())
    # 🔴 盐湖与"不达海终点"严格同现。结构注：填洼面上每条链只能终止于
    #   「海洋格」或「非海洋极点根」⇒ inland_roots == 0 时盐湖必为 0（本模型
    #   下的结构退化，不是实现缺陷；已列入实施汇报的交裁项）。
    if ef.conservation["inland_roots"] == 0:
        assert int(ef.salt_lake_mask.sum()) == 0
    # 旧 P98 读数层保留兼容：新河层 ⊆ 旧层（只额外去掉水面）
    assert int((ef.river_mask & ~ef.river_mask_legacy).sum()) == 0


def test_e19_c0b_adds_no_rng_consumption() -> None:
    """零新增 RNG：开档 `h` 逐位不变；同 seed 可复现；换陆比只改海陆划界。"""
    off = _field(seed=42)
    a = _field(seed=42, c0b_enabled=True)
    b = _field(seed=42, c0b_enabled=True)
    assert np.array_equal(off.h, a.h), "C0b 动到了噪声流（RNG 顺序契约）"
    for attr in ("h", "temp", "w", "acc", "filled", "receivers", "surface_type"):
        assert np.array_equal(getattr(a, attr), getattr(b, attr)), attr
    c = _field(seed=42, c0b_enabled=True, land_fraction=0.5)
    assert np.array_equal(c.h, off.h), "换陆比动到了噪声流"
    assert np.array_equal(c.temp, a.temp), "换陆比不得改温度（只改海陆划界）"
    assert c.conservation["land_area_frac"] > a.conservation["land_area_frac"], \
        "陆比扫参方向错（0.5 档应更多陆）"


def test_e20_c0b_config_domain_assertions() -> None:
    """参数域 + 类型守卫（fail-loud；不许静默当 truthy / 不许无定义阈值）。"""
    with pytest.raises(AssertionError):
        EnvFieldConfig(c0b_enabled=1)                        # 非布尔
    with pytest.raises(AssertionError):
        EnvFieldConfig(land_fraction=0.0)
    with pytest.raises(AssertionError):
        EnvFieldConfig(land_fraction=1.0)
    with pytest.raises(AssertionError):
        EnvFieldConfig(peak_altitude_m=0.0)
    with pytest.raises(AssertionError):
        EnvFieldConfig(shelf_depth_m=-1.0)
    with pytest.raises(AssertionError):
        EnvFieldConfig(shelf_depth_m=5000.0)                 # ≥ 默认 peak 4500
    with pytest.raises(AssertionError):
        EnvFieldConfig(coast_gamma=0.0)
    with pytest.raises(AssertionError):
        EnvFieldConfig(coast_gamma=float("nan"))
    ok = EnvFieldConfig(c0b_enabled=True, land_fraction=0.292,
                        peak_altitude_m=8848.0, shelf_depth_m=200.0, coast_gamma=0.7)
    assert ok.coast_gamma == 0.7


def test_e21_c0b_combo_gates_fail_loud() -> None:
    """两条既有组合闸对 C0b 同样生效（宁炸不静默 no-op）。"""
    cfg = SimConfig(seed=42)
    cfg.simulation.use_sim_core = True
    cfg.env_field.enabled = True
    cfg.env_field.c0b_enabled = True
    with pytest.raises(NotImplementedError):
        SphereEngine(cfg)

    cfg2 = SimConfig(seed=42)
    cfg2.simulation.use_sim_core = False
    cfg2.env_field.enabled = True
    cfg2.env_field.c0b_enabled = True
    cfg2.resource_dynamics.enabled = True
    with pytest.raises(ValueError):
        SphereEngine(cfg2)


def test_e22_c0b_old_archive_falls_back_to_off() -> None:
    """旧存档（无 c0b_* 键）⇒ 全部回退默认关；带键 ⇒ 逐字段还原（R233 T-F 家族防线）。"""
    data = SimConfig(seed=42).to_dict()
    for key in ("c0b_enabled", "land_fraction", "peak_altitude_m",
                "shelf_depth_m", "coast_gamma"):
        data["env_field"].pop(key, None)
    back = SimConfig.from_dict(data).env_field
    default = EnvFieldConfig()
    assert back.c0b_enabled is False
    assert back.land_fraction == default.land_fraction
    assert back.peak_altitude_m == default.peak_altitude_m
    assert back.shelf_depth_m == default.shelf_depth_m
    assert back.coast_gamma == default.coast_gamma

    cfg = SimConfig(seed=42)
    cfg.env_field.c0b_enabled = True
    cfg.env_field.land_fraction = 0.42
    cfg.env_field.coast_gamma = 0.8
    got = SimConfig.from_dict(cfg.to_dict()).env_field
    assert got.c0b_enabled is True and got.land_fraction == 0.42 and got.coast_gamma == 0.8


# 🔴 E23（C0b-②：海洋掩码乘性入口）在 ② 那一笔提交里随 `resource_field`/引擎接线同时入库。

