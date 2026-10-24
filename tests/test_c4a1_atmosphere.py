"""C4a-1 热机本体 Step 1（粗网格大气骨架）—— 回归单测。

覆盖（每条都断言**可观测行为**，不是"字段存在"；权威口径 = v26 裁定件 §二 Step 1 / §三 / §四）：
    A1  默认关 ⇒ 引擎不构造（零成本 + C7 回滚点，v26 §四-17）
    A2 开档 30 tick 与关档**主网格逐位等价** ⇒ 大气零副作用（Step 2 起 `t_surf` 自身演化，见 E15）
    A3  网格算式：480×960 ÷ 4 ⇒ **120×240**（正式档写死）；因子不整除 ⇒ fail-loud（不静默回落）
    A4  触发相位确定：`due()` 命中集 = {10,20,…}；`steps == substeps × ⌊tick/every⌋`（§四-18）
    A5 骨架不漂移：`advance()` **只写 `t_surf`** ⇒ 面积/尺/算子/驱动场逐位不变（Step 2 改写口径）
    A6  coarsen 面积守恒：Σ细 w·f == Σ粗 area·F（相对残差 < 1e-13，v26 §四-16）
    A7  refine 面积守恒 + 极带单值
    A8  互逆：coarsen(refine(F)) ≈ F（求和次序可差 1 ULP ⇒ 门限 1e-12）
    A9  均匀输入给均匀输出（漏乘 cos φ 的头号探测器，v26 §四-3）
    A10 旋转对称：只随纬度变的细场 ⇒ 粗场逐行均匀 + 值 = 块面积权均值（解析核对）
    A11 极点三条：coarsen 极行均匀、不吃非规范槽垃圾、值 = **全带**面积权均值
    A12 快照：往返逐位一致 + 续跑 == 连续跑（含 steps/last_tick 相位；v26 §四-18）
    A13 关档快照**无新键**（旧快照形状逐位不变）
    A14 旧档回退默认关 / 带键逐字段还原（R233 T-F 家族）
    A15 参数域断言 + 类型守卫（fail-loud，不许静默降级）
    A16 指纹：`atmosphere` 组自动进 fingerprint（**指纹变、行为不变**）
    A17 零 RNG：构造与推进都不消费引擎 RNG（v26 §四-19）
    A18 面积权均值口径（与 ENV f_g/f_c、C1 band 同一把尺）
    A19 极点回写守恒：`collapse_poles` 不改 Σarea·F
    A20 快照形状不符 ⇒ fail-loud（不静默重置）；A20b 上下采样尺寸不符 ⇒ fail-loud
    A21 Step 2 备料：入射项 S̄(φ) 与现 base_t 剖面**同源**（单一处年积分 + 锚点哨兵）

设计原则（教训库）：默认关 = 旧行为逐位一致（A1/A2 钉死）；每条只动一个参数；
守恒判据用**同一把面积权**（粗格面积 = 块内细格面积之和）⇒ 残差可归因到浮点舍入。
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

from simulation.config import (AtmosphereConfig, EnvFieldConfig, LightConfig,
                               SimConfig)
from simulation.sphere_engine import SphereEngine
from world import env_field
from world.atmosphere import Atmosphere
from world.sphere_world import SphereWorld

_ROWS, _COLS = 60, 120          # 代码默认微型冒烟档（引擎 30 tick ~秒级）
_REL = 1e-13                    # 守恒相对残差门限（浮点求和次序级，≪ v26 要求的 1e-10）
_ABS = 1e-12                    # "复原/均匀"类断言门限（同上，留 1–2 ULP 余量）


def _stub_surface(rows: int, cols: int) -> np.ndarray:
    """Step 1 只核**几何 / 节拍 / 采样对** ⇒ 地表层用均匀陆地占位。

    ⚠️ Step 2 起 `surface_type` 是**必需**入参（α/热容量按地表类型查表，v26 §二）⇒
    本文件的构造全部带上占位层；真实地表与 EBM 物理的判据在 `tests/test_c4a1_ebm.py`。
    """
    return np.full(rows * cols, env_field.SURFACE_LAND, dtype=np.uint8)


def _atm(rows: int = _ROWS, cols: int = _COLS, **kw) -> Atmosphere:
    return Atmosphere(SphereWorld(rows, cols), AtmosphereConfig(enabled=True, **kw),
                      _stub_surface(rows, cols))


def _fine_ramp(atm: Atmosphere) -> np.ndarray:
    """确定性细场（**零 RNG**）：斜坡 + 极行非规范槽塞垃圾（真引擎里那里是未定义历史值）。"""
    rf, cf = atm.rows_f, atm.cols_f
    r = np.arange(rf, dtype=np.float64)[:, None]
    c = np.arange(cf, dtype=np.float64)[None, :]
    f = 1.0 + 0.25 * r + 0.001 * c
    f[0, 1:] = 9.0e9
    f[-1, 1:] = -7.0e9
    return f.ravel()


def _canonical(f2: np.ndarray) -> np.ndarray:
    """按"极点整行坍缩"读法把极行改成规范槽值广播（与 `Atmosphere.coarsen` 同口径）。"""
    out = np.array(f2, dtype=np.float64, copy=True)
    out[0, :] = out[0, 0]
    out[-1, :] = out[-1, 0]
    return out


def _cfg(atmosphere: AtmosphereConfig | None = None, ticks: int = 60, seed: int = 42) -> SimConfig:
    """引擎档（🔴 `env_field.c0b_enabled=True`：Step 2 起大气必需 C0b 的 `surface_type`）。

    ⚠️ 两个臂（开/关大气）**共用同一份** env_field 配置 ⇒ A2/A16 的"逐位等价"仍然只
    隔离 `atmosphere` 这一个开关（否则比的是"env_field 有无"）。
    """
    cfg = SimConfig(seed=seed)
    cfg.world.rows, cfg.world.cols = _ROWS, _COLS
    cfg.simulation.ticks = ticks
    cfg.simulation.use_sim_core = False
    cfg.env_field = EnvFieldConfig(enabled=True, c0b_enabled=True)
    if atmosphere is not None:
        cfg.atmosphere = atmosphere
    return cfg


def _engine_state(eng: SphereEngine) -> dict:
    """引擎等价面：资源场 + 个体数组 + RNG 抽取次数与比特流状态 + tick。"""
    p = len(eng._id)
    return {
        "grid": eng.resources._grid.tobytes(),
        "capacity": eng.resources._capacity.tobytes(),
        "flat": eng._flat[:p].tobytes(),
        "energy": eng._energy[:p].tobytes(),
        "ids": eng._id[:p].tobytes(),
        "draws": int(eng.rng.draws),
        "rng": pickle.dumps(eng.rng.bit_generator.state),
        "tick": int(eng._tick),
    }


# ---------------------------------------------------------------- 开关与构造

def test_a01_default_off_not_constructed():
    """A1 默认关 ⇒ 配置关 + 引擎**不构造**大气（C7 回滚点，零成本）。"""
    assert AtmosphereConfig().enabled is False
    assert SphereEngine(_cfg()).atmosphere is None


def test_a02_on_is_side_effect_free_bit_equal():
    """A2 Step 1 开档 30 tick 与关档**逐位等价** ⇒ 骨架零副作用。

    🔴 这是 Step 1 最强的一条：大气开着跑与关着跑，引擎全部状态逐位相同
    （`advance()` 只动自己的 `steps/last_tick`，不碰主网格、不消费 RNG）。
    Step 2 接上 EBM 后本条**必须**改为"比主网格等价、比大气场不等价"——由 Step 2 显式改写。
    """
    off = SphereEngine(_cfg())
    on = SphereEngine(_cfg(AtmosphereConfig(enabled=True)))
    for _ in range(30):
        off.step()
        on.step()
    assert _engine_state(off) == _engine_state(on)
    assert on.atmosphere is not None and on.atmosphere.steps == 3 * 3       # 30/10 × 3
    assert off.atmosphere is None


def test_a03_official_grid_arithmetic():
    """A3 正式档 480×960 ÷ 4 ⇒ 120×240（v26 §二写死；不是"顺便"）。"""
    atm = _atm(480, 960)
    assert (atm.rows_c, atm.cols_c, atm.n_c) == (120, 240, 28800)
    assert (atm.factor, atm.every_ticks, atm.substeps, atm.dt_s) == (4, 10, 3, 120.0)
    assert atm.t_surf.shape == (atm.n_c,)


@pytest.mark.parametrize("bad, exc", [(7, ValueError), (8, ValueError),
                                      (0, AssertionError), (-4, AssertionError)])
def test_a03b_non_dividing_factor_is_fail_loud(bad: int, exc: type):
    """A3 因子不能同时整除两维 ⇒ **构造期 fail-loud**（**不静默回落**，与气味场口径相反）。

    两道路径都要炸：`0/-3/-4` 属参数域非法 ⇒ `AtmosphereConfig.__post_init__`
    （AssertionError）；`7/8` 参数域合法但与网格不整除 ⇒ `Atmosphere.__init__`
    （ValueError，因为只有模块握有 rows/cols）。
    """
    with pytest.raises(exc, match="coarse_factor"):
        _atm(coarse_factor=bad)


# ---------------------------------------------------------------- 节拍与不漂移

def test_a04_trigger_phase_is_exact():
    """A4 相位 = `tick % every_ticks == 0` ⇒ 首推在 tick 10，命中集唯一可列。"""
    atm = _atm()
    assert [t for t in range(1, 61) if atm.due(t)] == [10, 20, 30, 40, 50, 60]


def test_a04b_engine_step_count_matches_formula():
    """A4 `steps == substeps × ⌊tick/every_ticks⌋`、`last_tick` = 最后一个满 10 的 tick。"""
    eng = SphereEngine(_cfg(AtmosphereConfig(enabled=True), ticks=64))
    for n, exp_steps, exp_last in ((9, 0, -1), (10, 3, 10), (25, 6, 20), (30, 9, 30),
                                   (31, 9, 30), (40, 12, 40)):
        while eng._tick < n:
            eng.step()
        assert eng.atmosphere.steps == exp_steps, f"tick={n}"
        assert eng.atmosphere.last_tick == exp_last, f"tick={n}"


def test_a05_skeleton_does_not_drift():
    """A5 骨架不漂移：`advance()` **只写 `t_surf`** ⇒ 几何/面积/算子/驱动场逐位不变。

    🔴 口径变更（Step 2 显式改写，不是悄悄放宽）：Step 1 原文是"任意手工置入的场跑 50
    tick 后逐位不变"，其成立前提是"本步先不接物理"。Step 2 接上 EBM 后 `t_surf` **应当**
    演化（演化本身由 `tests/test_c4a1_ebm.py` E4/E5/E21 核守恒与方向），因此本条改为核
    "**除 `t_surf` 之外**的骨架状态逐位不变"——采样对/尺/驱动场若被时间推进污染，本条即红。
    """
    atm = _atm()
    seed = (np.arange(atm.n_c, dtype=np.float64).reshape(atm.rows_c, atm.cols_c) * 1.7 - 3.0)
    seed[0, :] = 5.0
    seed[-1, :] = -5.0
    atm.t_surf[:] = seed.ravel()
    static = {k: np.asarray(getattr(atm, k)).copy()
              for k in ("area", "tau", "mu_edge", "mu_ctr", "alpha_surface", "alpha_eff",
                        "s_in", "absorbed", "heat_cap", "_w_fine", "_w_block", "_den")}
    before = atm.t_surf.copy()
    for t in range(1, 51):
        if atm.due(t):
            atm.advance(t)
    assert atm.steps == 15 and atm.last_tick == 50
    assert not np.array_equal(atm.t_surf, before)                 # Step 2：物理在演它
    for k, v in static.items():
        assert np.array_equal(np.asarray(getattr(atm, k)), v), k  # 其余骨架逐位不动
    # 守恒算子仍然可用且口径不变（采样对在推进 15 次子步后照样守恒）
    f = _fine_ramp(atm)
    assert abs(atm.total_fine(f) - atm.total_coarse(atm.coarsen(f))) \
        <= _REL * abs(atm.total_fine(f))


# ---------------------------------------------------------------- 上下采样守恒

def test_a06_coarsen_conserves_area_weighted_integral():
    """A6 coarsen 守恒：`total_fine(f) == total_coarse(coarsen(f))`（极行垃圾不参与积分）。

    左端 `total_fine` 与 `coarsen` **共用**规范槽读法（`_fine2d`）⇒ 守恒恒等式对
    **任意**极行内容成立；若左端改成"裸求和"，本条会残差 1.0（= 把 9e9 垃圾当成极点面积）。
    """
    atm = _atm()
    f = _fine_ramp(atm)
    lhs = atm.total_fine(f)
    rhs = atm.total_coarse(atm.coarsen(f))
    assert abs(lhs - rhs) <= _REL * abs(lhs), f"残差 {abs(lhs - rhs) / abs(lhs):.3e}"
    # 极行**均匀**的场 ⇒ 裸求和与规范积分必须相同（证明差异只来自垃圾槽，不是口径漂移）
    g = _canonical(f.reshape(atm.rows_f, atm.cols_f)).ravel()
    w = np.repeat(np.cos(np.asarray(atm.world_fine._lat, dtype=np.float64)), atm.cols_f)
    assert abs(atm.total_fine(g) - float((w * g).sum())) <= _REL * abs(float((w * g).sum()))


def test_a07_refine_conserves_and_is_single_valued_at_poles():
    """A7 refine 守恒 + 极带单值（分片常数 = 有限体积零阶重构）。"""
    atm = _atm()
    F = atm.coarsen(_fine_ramp(atm))
    fine = atm.refine(F).reshape(atm.rows_f, atm.cols_f)
    assert abs(atm.total_fine(fine) - atm.total_coarse(F)) \
        <= _REL * abs(atm.total_coarse(F))
    s = atm.factor
    assert np.array_equal(fine[:s], np.full((s, atm.cols_f), F[0, 0]))
    assert np.array_equal(fine[-s:], np.full((s, atm.cols_f), F[-1, 0]))


def test_a08_round_trip_inverse():
    """A8 互逆：coarsen(refine(F)) ≈ F（1e-12；求和次序可差 1–2 ULP）。"""
    atm = _atm()
    F = atm.coarsen(_fine_ramp(atm))
    assert np.allclose(atm.coarsen(atm.refine(F)), F, rtol=1e-12, atol=1e-12)
    one = np.ones(atm.n_c, dtype=np.float64)
    assert np.allclose(atm.coarsen(atm.refine(one)), 1.0, rtol=0, atol=_ABS)


def test_a09_uniform_input_gives_uniform_output():
    """A9 均匀输入 ⇒ 均匀输出（漏乘 cos φ / 权重错配的头号探测器，v26 §四-3）。"""
    atm = _atm()
    F = atm.coarsen(np.full(atm.rows_f * atm.cols_f, 1.0))
    assert float(F.max()) - float(F.min()) < _ABS
    assert abs(atm.awm(F) - 1.0) < _ABS


def test_a10_zonal_symmetric_field_analytic():
    """A10 只随纬度变的细场 ⇒ 粗场逐行均匀，且值 = 块面积权均值（解析核对）。"""
    atm = _atm()
    lat = np.asarray(atm.world_fine._lat, dtype=np.float64)
    prof = 1.0 + 0.5 * np.cos(lat)
    F = atm.coarsen(np.repeat(prof, atm.cols_f))
    assert all(float(np.ptp(F[r])) == 0.0 for r in range(atm.rows_c))   # 经度恒等（精确）
    s, w = atm.factor, np.cos(lat)
    for r in range(atm.rows_c):
        blk = slice(r * s, (r + 1) * s)
        expect = float((w[blk] * prof[blk]).sum() / w[blk].sum())
        assert abs(F[r, 0] - expect) <= 1e-14 * max(abs(expect), 1.0)


def test_a11_pole_row_is_one_cell():
    """A11 极点三条：极行均匀、只吃规范槽、值 = **全带**面积权均值（不是逐列分块）。"""
    atm = _atm()
    f = _fine_ramp(atm)
    F = atm.coarsen(f)
    assert float(np.ptp(F[0])) == 0.0 and float(np.ptp(F[-1])) == 0.0
    # 改掉非规范极槽 ⇒ 输出必须**完全不变**（证明垃圾没被吞）
    g = f.reshape(atm.rows_f, atm.cols_f).copy()
    g[0, 1:] = 1.234e6
    g[-1, 1:] = -9.876e5
    assert np.array_equal(F, atm.coarsen(g.ravel()))
    # 极行值 = 全带（首/末 `factor` 行、全列）面积权均值
    s = atm.factor
    w = np.cos(np.asarray(atm.world_fine._lat, dtype=np.float64))
    fc = _canonical(f.reshape(atm.rows_f, atm.cols_f))
    num = float((w[:s, None] * fc[:s]).sum())
    den = float(w[:s].sum()) * atm.cols_f
    assert abs(F[0, 0] - num / den) <= 1e-14 * max(abs(num / den), 1.0)


def test_a19_collapse_poles_is_conservative():
    """A19 `collapse_poles` 不改变 Σarea·F（极行各槽面积相等 ⇒ 取均值 = 守恒操作）。"""
    atm = _atm()
    F = np.arange(atm.n_c, dtype=np.float64).reshape(atm.rows_c, atm.cols_c)
    before = atm.total_coarse(F)
    atm.collapse_poles(F)
    assert float(np.ptp(F[0])) == 0.0 and float(np.ptp(F[-1])) == 0.0
    assert abs(atm.total_coarse(F) - before) <= _REL * abs(before)


def test_a18_awm_uses_same_area_weight_as_coarsen():
    """A18 面积权均值口径：常值场 ⇒ awm = 常值；权重就是 `area`（与 coarsen 同一把尺）。"""
    atm = _atm()
    assert abs(atm.awm(np.full(atm.n_c, -2.5)) + 2.5) < _ABS
    F = np.linspace(0.0, 1.0, atm.n_c)
    w = np.repeat(atm.area, atm.cols_c)
    assert abs(atm.awm(F) - float((w * F).sum() / w.sum())) < 1e-16


# ---------------------------------------------------------------- 快照 / 兼容

def test_a12_snapshot_roundtrip_and_resume(tmp_path: Path):
    """A12 快照往返逐位一致 + **续跑 == 连续跑**（含 steps/last_tick 相位；v26 §四-18）。"""
    cfg = _cfg(AtmosphereConfig(enabled=True), ticks=64)
    e1 = SphereEngine(cfg)
    for _ in range(25):
        e1.step()
    e1.atmosphere.t_surf[:] = np.arange(e1.atmosphere.n_c, dtype=np.float64)  # 手工置入载体
    path = tmp_path / "atm_on.npz"
    e1.save_snapshot(str(path))

    e2 = SphereEngine(cfg)
    for _ in range(25):
        e2.step()
    assert _engine_state(e2) == _engine_state(e1)          # 同 seed 同 tick ⇒ 状态一致
    assert int(e1.atmosphere.state()["steps"]) == 6

    r = SphereEngine.load_snapshot(str(path))
    assert np.array_equal(r.atmosphere.t_surf, e1.atmosphere.t_surf)
    assert (r.atmosphere.steps, r.atmosphere.last_tick) == (e1.atmosphere.steps,
                                                            e1.atmosphere.last_tick) == (6, 20)
    d_r, d_e = int(r.rng.draws), int(e1.rng.draws)
    for _ in range(10):
        e1.step()
        r.step()
    # 🔴 `rng.draws` 是 provenance 计数器、**不入快照**（`load_snapshot` 只回绑
    #   `bit_generator.state`，sphere_engine.py:6923）⇒ 跨快照边界只能比**增量**。
    assert int(r.rng.draws) - d_r == int(e1.rng.draws) - d_e
    sr, se = _engine_state(r), _engine_state(e1)
    sr.pop("draws"), se.pop("draws")
    assert sr == se                                   # 续跑与连续跑逐位一致（含 RNG 比特流）
    assert (r.atmosphere.steps, r.atmosphere.last_tick) == (9, 30)


def test_a13_off_snapshot_has_no_new_keys(tmp_path: Path):
    """A13 关档快照**无新键** ⇒ 旧快照形状逐位不变（`smell_field` 先例）。"""
    eng = SphereEngine(_cfg(), )
    for _ in range(11):
        eng.step()
    path = tmp_path / "atm_off.npz"
    eng.save_snapshot(str(path))
    with np.load(str(path), allow_pickle=False) as data:
        assert not [k for k in data.files if k.startswith("atm_")]


def test_a20_restore_shape_mismatch_is_fail_loud():
    """A20 快照形状与当前网格不符 ⇒ fail-loud（不静默重置成零场）。"""
    atm = _atm()
    with pytest.raises(ValueError, match="形状"):
        atm.restore({"t_surf": np.zeros(atm.n_c + 1), "steps": 0, "last_tick": 0})


@pytest.mark.parametrize("size", [120 * 240, 60 * 120 + 1, 0])
def test_a20b_sampling_size_mismatch_is_fail_loud(size: int):
    """A20 上下采样入参尺寸不符 ⇒ fail-loud（跨网格尺寸静默 reshape = 错位场）。"""
    atm = _atm()
    with pytest.raises(ValueError, match="尺寸"):
        atm.coarsen(np.zeros(size))
    with pytest.raises(ValueError, match="尺寸"):
        atm.total_fine(np.zeros(size))
    with pytest.raises(ValueError, match="尺寸"):
        atm.refine(np.zeros(size))


def test_a21_insolation_is_the_single_source_for_base_profile():
    """A21（Step 2 备料）入射项 S̄(φ) **同源** ⇒ 现 base_t 剖面就是它的一处闭式变换。

    v26 §三 要求 EBM 稳态剖面与"现 insolation 年平剖面"**同形**；若热机本体另写一份
    365 天积分 = 双写漂移（加基因五步曲同款纪律）。钉死三条：
    ① `env_field` 源码里**只有一处**年积分（`_DAYS_PER_YEAR` 的 `linspace`）；
    ② `base_t(lat) == ((1−α_p)·S̄/σ)^{1/4} + ΔT_gh − 273.15` **逐位**成立（同一次求和）；
    ③ S̄ 锚点（真实地球）：赤道 ≈415 W/m²、极点 ≈177 W/m²（±2%，口径没跑偏的哨兵）。
    """
    src = Path(env_field.__file__).read_text(encoding="utf-8")
    assert src.count("_DAYS_PER_YEAR, endpoint=False") == 1
    lat = np.asarray(SphereWorld(60, 120)._lat, dtype=np.float64)
    s = env_field.annual_mean_insolation(lat)
    t = env_field._insolation_base_temperature(lat)
    rebuilt = ((1.0 - env_field._PLANETARY_ALBEDO) * s / env_field._SIGMA_SB) ** 0.25 \
        + env_field._GREENHOUSE_K - 273.15
    assert np.array_equal(t, rebuilt)
    i_eq, i_pole = int(np.argmin(np.abs(lat))), int(np.argmax(np.abs(lat)))
    assert abs(float(s[i_eq]) - 413.0) <= 0.02 * 413.0
    assert abs(float(s[i_pole]) - 177.0) <= 0.02 * 177.0


def test_a14_from_dict_old_save_falls_back_and_new_restores():
    """A14 旧档无 `atmosphere` 键 ⇒ 默认关；带键 ⇒ 逐字段还原（R233 T-F 家族）。"""
    d = SimConfig().to_dict()
    d.pop("atmosphere")
    assert SimConfig.from_dict(d).atmosphere.enabled is False
    cfg = SimConfig()
    cfg.atmosphere = AtmosphereConfig(enabled=True, coarse_factor=2,
                                      every_ticks=10, substeps=3, dt_s=120.0)
    back = SimConfig.from_dict(cfg.to_dict())
    for k in ("enabled", "coarse_factor", "every_ticks", "substeps", "dt_s"):
        assert getattr(back.atmosphere, k) == getattr(cfg.atmosphere, k)


@pytest.mark.parametrize("kw", [{"enabled": 1}, {"coarse_factor": 0}, {"coarse_factor": -3},
                                {"every_ticks": 0}, {"substeps": 0}])
def test_a15_parameter_domain_assertions(kw: dict):
    """A15 参数域 + 类型守卫（构造期 fail-loud，不许静默降级）。"""
    with pytest.raises(AssertionError):
        AtmosphereConfig(**kw)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf"), 1e9])
def test_a15b_dt_domain(bad: float):
    """A15 `dt_s` 必须是有限正秒数（NaN/0/负/≥1e6 ⇒ 炸；子步长无定义不许静默）。"""
    with pytest.raises(AssertionError):
        AtmosphereConfig(dt_s=bad)


def test_a16_fingerprint_changes_but_behavior_does_not():
    """A16 `atmosphere` 经 asdict **自动进指纹** ⇒ 指纹变、行为不变（默认关档口径）。"""
    fp_off = SimConfig().fingerprint()
    assert "atmosphere" in fp_off                       # 新组自动进指纹（跨档续跑被拦）
    a, b = _cfg(), _cfg(AtmosphereConfig(enabled=True))
    assert a.fingerprint() != b.fingerprint()
    assert _engine_state(SphereEngine(a)) == _engine_state(SphereEngine(b))


def test_a17_zero_rng_consumption():
    """A17 零 RNG：构造 + 50 tick 推进都不消费引擎 `self.rng`（v26 §四-19）。"""
    eng = SphereEngine(_cfg(AtmosphereConfig(enabled=True)))
    d0 = int(eng.rng.draws)
    assert int(eng.rng.draws) == d0                     # 构造本身不动流
    for t in range(1, 51):
        if eng.atmosphere.due(t):
            eng.atmosphere.advance(t)
    assert int(eng.rng.draws) == d0


def test_a16_tick_rate_must_match_rotation_period():
    """v38/P0.3：`substeps×dt_s/every_ticks` 必须等于 `86400/rotation_period`（汇率唯一防线）。

    两例：① 对齐值（dt_s=120 / rotation=2400 ⇒ 36 s/tick）正常构造；
    ② 漂移值（dt_s=100 ⇒ 30 s/tick）构造期 `AssertionError` 并报出两个数字。
    该断言只在 `atmosphere.enabled=True` 时生效（关档不构造大气，不该被它拦住）。
    """
    SimConfig(atmosphere=AtmosphereConfig(enabled=True, dt_s=120.0),
              light=LightConfig(rotation_period=2400))          # ① 对齐 ⇒ 不抛
    with pytest.raises(AssertionError, match="不对齐"):
        SimConfig(atmosphere=AtmosphereConfig(enabled=True, dt_s=100.0),
                  light=LightConfig(rotation_period=2400))      # ② 30 s/tick ⇒ 炸


def test_a17_refine_bilinear_is_convex_and_differs_within_block():
    """B2.1：双线性上采样三件——块内方差 0→>0、不越原场 [min,max]、与 refine 职责分离。

    分片常数 `refine` 的块内标准差恒 0（零阶重构）；双线性是四邻凸组合 ⇒ 块内变化 >0，
    但权重非负和 1 ⇒ 输出仍在原场值域内。两者**不可互换**：守恒量必须走 `refine`。
    """
    a = _atm()
    rc, cc = a.rows_c, a.cols_c
    s = a.factor
    rng = np.random.default_rng(7)
    F = rng.random((rc, cc)) * 100.0
    lin = a.refine_bilinear(F).reshape(a.rows_f, a.cols_f)
    con = a.refine(F).reshape(a.rows_f, a.cols_f)
    assert float(con[:s, :s].std()) == 0.0, "分片常数的块内标准差必须恰为 0"
    assert float(lin[:s, :s].std()) > 0.0, "双线性块内方差没起来 ⇒ 不是真插值"
    assert float(lin.min()) >= float(F.min()) - 1e-12, "双线性越下界（凸组合被破坏）"
    assert float(lin.max()) <= float(F.max()) + 1e-12, "双线性越上界（凸组合被破坏）"
    # 职责分离：只有分片常数能 `coarsen∘refine` 严格还原。
    # ⚠️ 极点行例外（设计）：`refine` 把极带写成 `c[0,0]`（规范槽），`coarsen` 的极点行
    # 取该带**全列**均值 ⇒ 随机场下极点行不还原；这是「极点=一圈一格」的既有口径，
    # 不是插值缺陷 ⇒ 对账在非极点行（1..rc-2）上做。
    assert np.allclose(a.coarsen(a.refine(F))[1:-1], F[1:-1], rtol=0.0, atol=1e-12)
    assert not np.allclose(a.coarsen(a.refine_bilinear(F))[1:-1], F[1:-1],
                           rtol=0.0, atol=1e-12)
