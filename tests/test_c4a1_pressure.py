"""C4a-1 热机本体 Step 3a（单层大气质量 proxy → 静力平衡 → 地面气压）—— 回归单测。

权威口径 = `内部汇报/话事人传达-20261016-C4a-1热机本体开工裁定.md`（v26）
§二 因果链「单层大气 → 质量 proxy + 静力平衡 → 气压 P」+ §四-9（🔴 禁 `P∝T` 硬写）
+ §四-4（℃/K 唯一换算点）+ §四-11/15（质量、非负）+ v30 ⑤（两时钟分离，本批接受准静态）。

覆盖（每条断言可观测行为 / 守恒恒等式，不是"字段存在"）：
    P1  非负/有限三条：`P>0`、柱质量 `m>0`、全有限（§四-15）+ probe 读数全有限
    P2  🔴 单位纪律：气压只吃 `t_kelvin()`（唯一换算点）—— 手工压高公式用 K **复现**、
        用 ℃ 代入**不复现**（§四-4：本批第一次真正消费 K）
    P3  🔴 §四-9 符号判据：地面气压与柱温**反号**相关（暖区地面低压）+ 解析单调性
        均匀增温扰动下全球地面气压必须**下降**（质量不能随温度凭空增加）
    P3b 单暖斑装置：斑心地面是**低压**（与 Step 3b 的 #7 风向测试同一背景场）
    P4  🌍 全球质量锚：`AWM(P_s) ≡ 101325 Pa` 到机器精度且**推进后仍成立** ⇒ 温度只能在
        柱间重新分配质量、不能造/丢质量（§四-11 的质量侧对应物）；两种记账自洽
    P5  静力平衡恒等式：`P_s ≡ P_顶 + m·g` 逐格成立（"气压 = 空气柱重量"的直接核对）
    P6  诊断量**不进档**（键集合恒为 5：场 + 两风分量 + 两节拍）+ `restore()` 后与连续跑
        **逐位一致**（§四-18 的"确定性重建"分支）；P6b 引擎级续跑 == 连续跑（气压也纳入比对）
    P7  异常 fail-loud：`layer_depth_m` 越出理想化有效域 / T(K)≤0 / NaN / Inf ⇒ 抛错不 clip
    P8  关档逐位等价（C7）：开档 60 tick 主网格 + 生物 + RNG 面与关档**逐位全等** +
        关档快照无 `atm_*` 键 + 开档**零 RNG 消费**（v26 §四-17/19）
    P9  结构读数：赤道地面气压 < 两极（Step 3b 风向的**气压前提**）
    P10 参数域守卫 + 旧档回退：`layer_depth_m` 非法 ⇒ AssertionError；旧存档缺字段 ⇒
        回退默认、新字段可往返（`from_dict` 白名单自动含新字段）
    P11 极行不重复计数（§四-13）：推进后极行气压经度恒单值
    P12 读数只读：`probe()` 不改任何状态
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

from simulation.config import AtmosphereConfig, EnvFieldConfig, SimConfig
from simulation.sphere_engine import SphereEngine
from world import env_field as ef
from world.atmosphere import (Atmosphere, _G_ACCEL, _HYPSO_RATIO_MAX, _P_REF_PAS,
                              _R_DRY_AIR)
from world.sphere_world import SphereWorld

_ROWS, _COLS = 60, 120            # 微型冒烟档（4 因子 ⇒ 粗网格 15×30）
_REL = 1e-13                      # 守恒相对残差门限（浮点求和次序级，≪ v26 的 1e-10）
_ABS = 1e-12


# ------------------------------------------------------------------ 装置

def _surface(rows: int = _ROWS, cols: int = _COLS, kind: str = "mixed") -> np.ndarray:
    """确定性地表类型层（**零 RNG**）；kind="ocean" ⇒ α/c 不随经纬度变（干净装置）。"""
    idx = np.arange(rows * cols, dtype=np.int64).reshape(rows, cols)
    if kind == "ocean":
        return np.full(idx.size, ef.SURFACE_OCEAN, dtype=np.uint8)
    st = np.where(idx < rows * cols // 2, ef.SURFACE_OCEAN, ef.SURFACE_LAND)
    st[:, ::3] = ef.SURFACE_RIVER
    st[rows // 4, 1] = ef.SURFACE_LAKE_SALT
    st[rows // 4, 2] = ef.SURFACE_LAKE_FRESH
    st[rows // 3, :4] = ef.SURFACE_SHELF
    return st.ravel().astype(np.uint8)


def _surface_symmetric(rows: int = _ROWS, cols: int = _COLS) -> np.ndarray:
    """南北**镜像对称**地表（对称输入必须给对称输出，同 Step 2 E8 装置）。"""
    idx = np.arange(rows * cols, dtype=np.int64).reshape(rows, cols)
    st = np.where(idx < rows * cols // 2, ef.SURFACE_OCEAN, ef.SURFACE_LAND)
    st[:, ::3] = ef.SURFACE_RIVER
    st[rows // 2:, :] = st[:rows // 2, :][::-1]
    return st.ravel().astype(np.uint8)


def _atm(rows: int = _ROWS, cols: int = _COLS, **akw) -> Atmosphere:
    surf = akw.pop("surface", None)
    cfg = AtmosphereConfig(enabled=True, **akw)
    return Atmosphere(SphereWorld(rows, cols), cfg,
                      _surface(rows, cols) if surf is None else surf)


def _cfg(atmosphere: AtmosphereConfig | None = None, ticks: int = 60,
         seed: int = 42) -> SimConfig:
    cfg = SimConfig(seed=seed)
    cfg.world.rows, cfg.world.cols = _ROWS, _COLS
    cfg.simulation.ticks = ticks
    cfg.simulation.use_sim_core = False
    cfg.env_field = EnvFieldConfig(enabled=True, c0b_enabled=True)
    cfg.atmosphere = atmosphere if atmosphere is not None else AtmosphereConfig(enabled=True)
    return cfg


def _engine_state(eng: SphereEngine) -> dict:
    """引擎等价面（与 Step 1/2 同口径：主网格 + 生物 + RNG 状态 + tick，**不含大气**）。"""
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


_ENV_LAYERS = (
    "h", "h_routed", "temp", "w", "acc", "filled", "w_ref", "h_m", "band",
    "ocean_mask", "land_mask", "shelf_mask", "lake_mask", "fresh_lake_mask",
    "salt_lake_mask", "river_mask", "river_mask_legacy", "surface_type", "receivers",
    "pr",
    "sea_level_m", "lake_count", "max_lake_cells", "max_lake_root",
    "max_lake_basin_cells",
)


def _env_surfaces(eng: SphereEngine) -> dict:
    """`env_field` 各层的逐位指纹（开档必须**一个字节都不动**，§四-17 C7 侧翼）。

    ⚠️ `EnvField` 用 `__slots__` ⇒ 不能 `vars()`，按声明名单取（新增层须同步本名单）。
    """
    out = {}
    for name in _ENV_LAYERS:
        val = getattr(eng.env_field, name, None)
        if isinstance(val, np.ndarray):
            out[name] = val.tobytes()
        elif isinstance(val, (int, float)) and not isinstance(val, bool):
            out[name] = float(val)
    return out


def _hypsometric(a: Atmosphere, t_celsius: bool = False) -> np.ndarray:
    """独立复算压高公式（层顶气压由面积权归一反解）；`t_celsius=True` ⇒ **故意**用 ℃ 代入。"""
    tk = a.t_kelvin() if not t_celsius else a.t_surf.copy()
    raw = np.exp(a._hypso_k / tk)
    w = np.repeat(a.area, a.cols_c)
    return _P_REF_PAS * raw / float((w * raw).sum() / w.sum())


# ---------------------------------------------------------------- 正误与守恒

def test_p1_pressure_is_positive_finite_and_probe_reads_it():
    a = _atm()
    assert a.p_surf.shape == (a.n_c,) and a.mass_col.shape == (a.n_c,)
    assert bool(np.all(np.isfinite(a.p_surf))) and bool(np.all(np.isfinite(a.mass_col)))
    assert float(a.p_surf.min()) > 0.0, "§四-15：P 必须为正"
    assert float(a.mass_col.min()) > 0.0, "§四-15：活动层柱质量必须为正"
    pr = a.probe()
    for k in ("p_awm_pa", "p_min_pa", "p_max_pa", "mass_awm_kg_m2", "p_t_corr",
              "p_awm_rel_residual", "hypso_k", "p_col_ratio_minus1", "p_top_pa"):
        assert np.isfinite(pr[k]), f"probe[{k}] 非有限"
    assert pr["p_col_ratio_minus1"] <= _HYPSO_RATIO_MAX, "理想化有效域读数应在界内"
    assert pr["p_t_corr"] < 0.0, "probe 的 §四-9 符号读数必须为负"


def test_p2_kelvin_is_the_only_conversion_point():
    """§四-4：气压链唯一消费 K 的地方就是状态方程 —— 用 ℃ 代入必须**不复现**。"""
    a = _atm()
    assert np.allclose(a.p_surf, _hypsometric(a), rtol=0.0, atol=_ABS * _P_REF_PAS), \
        "K 口径的手工复算必须复现 p_surf"
    bad = _hypsometric(a, t_celsius=True)
    assert not np.allclose(a.p_surf, bad, rtol=1e-6, atol=1e-6), \
        "p_surf 竟然等于用 ℃ 代状态方程的结果 ⇒ 唯一换算点被绕过（§四-4 破口）"
    # 单位口径的量级自检：℃ 代入 ⇒ 指数分母小 273 倍 ⇒ 柱压比荒谬放大
    assert float(bad.max() / bad.min() - 1.0) > 5.0 * float(
        a.p_surf.max() / a.p_surf.min() - 1.0)


def test_p3_warm_column_gives_surface_low_not_high():
    """🔴 §四-9：暖区地面**低压**（`P=P₀·T/T₀` 的错写会给正相关 ⇒ 单胞直接反号）。"""
    a = _atm(surface=_surface(kind="ocean"))
    assert a.aw_corr(a.p_surf, a.t_kelvin()) < 0.0, \
        "地面气压与柱温必须反号相关（暖区地面低压）"
    # 全局单调性：整层增温 ⇒ 同一质量锚下地面气压**下降**（质量不能随温度凭空增加）
    t0 = a.t_surf.copy()
    a.t_surf[:] = t0 + 1.0
    a._diagnose_pressure()
    p_up = a.awm(a.p_surf)
    a.t_surf[:] = t0
    a._diagnose_pressure()
    assert p_up == pytest.approx(_P_REF_PAS, rel=_REL), \
        "均匀增温扰动下质量锚也必须成立（AWM(P_s) 恒 = P₀）"
    p_z_before = a._p_z
    a.t_surf[:] = t0 + 1.0
    a._diagnose_pressure()
    assert a._p_z > p_z_before, \
        "整层增温 ⇒ 柱膨胀、质量外泄到层顶以上 ⇒ 层顶气压必须**升高**（地面 AWM 不动）"


def test_p3b_single_warm_spot_is_a_surface_low():
    """单暖斑装置（与 Step 3b 的 #7 风向测试同一背景场）：斑心必须是**低压**。"""
    a = _atm(surface=_surface(kind="ocean"))
    rc, cc = a.rows_c, a.cols_c
    t2 = np.full((rc, cc), 14.0)                      # 均匀背景（℃）⇒ 气压本应处处相等
    i0, j0 = rc // 2, cc // 4                          # 赤道带暖斑
    di = (np.arange(rc)[:, None] - i0) ** 2
    dj = np.minimum(np.abs(np.arange(cc) - j0),
                    cc - np.abs(np.arange(cc) - j0)) ** 2   # 纬向周期距离 ⇒ 接缝连续（§四-12）
    t2 += 8.0 * np.exp(-di / 8.0 - dj / 8.0)
    a.t_surf[:] = t2.ravel()
    a._diagnose_pressure()
    p2 = a.p_surf.reshape(rc, cc)
    assert p2[i0, j0] < p2[i0, (j0 + 3) % cc], "暖斑中心气压必须低于其东侧 3 格"
    assert p2[i0, j0] < p2[i0, (j0 + cc // 2) % cc], "暖斑中心必须低于同纬远端"
    assert p2[i0, j0] < p2[0, 0], "暖斑中心必须低于极区"
    assert a.aw_corr(a.p_surf, a.t_kelvin()) < 0.0, "暖斑构造下相关系数应为负"


def test_p4_global_mass_anchor_survives_advance():
    """`AWM(P_s) ≡ P₀` 到机器精度，且推进（温度演化）后仍成立 ⇒ 质量只重新分配。"""
    a = _atm()
    assert abs(a.awm(a.p_surf) - _P_REF_PAS) / _P_REF_PAS < _REL
    for k in range(1, 5):
        a.advance(a.every_ticks * k)
        r = abs(a.awm(a.p_surf) - _P_REF_PAS) / _P_REF_PAS
        assert r < _REL, f"推进 {a.steps} 子步后质量锚残差 {r:.3e} 越界 = 凭空造/丢质量"
        assert float(a.p_surf.min()) > 0.0 and float(a.mass_col.min()) > 0.0
    expect = (_P_REF_PAS - a._p_z) / _G_ACCEL          # 同一常数的两种记账必须自洽
    assert abs(a.awm(a.mass_col) - expect) / abs(expect) < _REL


def test_p5_hydrostatic_identity_holds_cell_by_cell():
    """§四-9 的字面要求：气压 = 空气柱重量 ⇒ `P_s ≡ P_顶 + m·g` 逐格恒等。"""
    a = _atm()
    a.advance(a.every_ticks)
    re_p = a._p_z + a.mass_col * _G_ACCEL
    rel = float(np.max(np.abs(re_p - a.p_surf) / a.p_surf))
    assert rel < _REL, f"静力平衡恒等式相对残差 {rel:.3e} 越界"


# ---------------------------------------------------------------- 快照与重建

def test_p6_pressure_is_rebuilt_not_stored_and_matches_bit_for_bit():
    a = _atm()
    for k in range(1, 4):
        a.advance(10 * k)
    st = a.state()
    assert set(st) == {"t_surf", "u_east", "u_north", "steps", "last_tick"}, \
        "诊断量进档 = 快照被派生量放大（§四-18 走的是'确定性重建'分支，不该多键；" \
        "`u_*` 是积分状态量 ⇒ 必须在键里）"
    b = _atm()
    b.restore(st)
    assert b.p_surf.tobytes() == a.p_surf.tobytes(), "restore 后气压必须与连续跑逐位一致"
    assert b.mass_col.tobytes() == a.mass_col.tobytes()
    assert (b.steps, b.last_tick) == (a.steps, a.last_tick)


def test_p6b_engine_resume_equals_continuous_run(tmp_path: Path):
    """引擎级：开档快照续跑 == 连续跑，**气压与质量 proxy 也逐位一致**（§四-18）。"""
    cfg = _cfg()
    e1 = SphereEngine(cfg)
    for _ in range(30):
        e1.step()
    path = tmp_path / "atm_p.npz"
    e1.save_snapshot(str(path))
    r = SphereEngine.load_snapshot(str(path))
    assert np.array_equal(r.atmosphere.p_surf, e1.atmosphere.p_surf)
    assert np.array_equal(r.atmosphere.mass_col, e1.atmosphere.mass_col)
    for _ in range(20):
        e1.step()
        r.step()
    assert e1.atmosphere.p_surf.tobytes() == r.atmosphere.p_surf.tobytes()
    with np.load(str(path), allow_pickle=False) as data:
        assert sorted(k for k in data.files if k.startswith("atm_")) == \
            ["atm_last_tick", "atm_steps", "atm_t_surf", "atm_u_east", "atm_u_north"]


def test_p7_out_of_range_and_nonfinite_are_fail_loud():
    with pytest.raises(FloatingPointError):
        _atm(layer_depth_m=40000.0)          # 层顶取到平流层顶 ⇒ 柱压比出理想化有效域
    a = _atm()
    a.t_surf[:] = -300.0                      # ⇒ 绝对温度 ≤ 0 K
    with pytest.raises(FloatingPointError):
        a._diagnose_pressure()
    a.t_surf[:] = np.nan
    with pytest.raises(FloatingPointError):
        a._diagnose_pressure()
    a.t_surf[:] = np.inf
    with pytest.raises(FloatingPointError):
        a._diagnose_pressure()


# ---------------------------------------------------------------- 关档与 RNG

def test_p8_off_mode_is_bit_identical_and_has_no_keys(tmp_path: Path):
    on_cfg, off_cfg = _cfg(), _cfg()
    off_cfg.atmosphere.enabled = False
    on, off = SphereEngine(on_cfg), SphereEngine(off_cfg)
    for _ in range(60):
        on.step()
        off.step()
        assert on.rng.draws == off.rng.draws, "Step 3a 引入了新 RNG 消费（§四-19）"
    assert _engine_state(on) == _engine_state(off), "开档改了主 tick 面 ⇒ C7 回滚点破"
    assert on.atmosphere is not None and on.atmosphere.steps == 18      # 6 推进 × 3 子步
    assert float(np.max(np.abs(on.atmosphere.p_surf))) > 0.0, "开档必须真的算了气压"
    path = tmp_path / "off.npz"
    off.save_snapshot(path)
    with np.load(str(path), allow_pickle=False) as data:
        assert not [k for k in data.files if k.startswith("atm_")]
    assert SphereEngine.load_snapshot(str(path)).atmosphere is None


def test_p8b_on_mode_leaves_every_env_layer_bit_identical():
    """开档（有气压诊断）60 tick ⇒ `env_field` 的**全部**浮点层逐位不变（大气只读）。"""
    on_cfg, off_cfg = _cfg(), _cfg()
    off_cfg.atmosphere.enabled = False
    on, off = SphereEngine(on_cfg), SphereEngine(off_cfg)
    for _ in range(60):
        on.step()
        off.step()
    assert _env_surfaces(on) == _env_surfaces(off), "大气写了环境场 ⇒ C7 侧翼判据破"


# ---------------------------------------------------------------- 结构与配置

def test_p9_equator_is_the_surface_pressure_trough():
    a = _atm(surface=_surface_symmetric())
    p2 = a.p_surf.reshape(a.rows_c, a.cols_c)
    i_eq = int(np.argmin(np.abs(a.mu_ctr)))
    assert p2[i_eq, 0] < p2[0, 0], "赤道地面气压必须低于南极（否则 Step 3b 风向反号）"
    assert p2[i_eq, 0] < p2[-1, 0], "赤道地面气压必须低于北极"
    assert a.probe()["p_t_eq_minus_pole_pa"] < 0.0


def test_p9b_symmetric_forcing_gives_symmetric_pressure():
    """§四-3 的球面自检：对称地表 + 对称驱动 ⇒ 南北半球气压剖面镜像（不偏爱某半球）。"""
    a = _atm(surface=_surface_symmetric())
    a.advance(a.every_ticks)
    p2 = a.p_surf.reshape(a.rows_c, a.cols_c)
    half = a.rows_c // 2
    north = p2[half + 1:, :][::-1]
    south = p2[:half, :]
    scale = float(np.max(np.abs(p2)))
    assert float(np.max(np.abs(north - south))) / scale < 1e-9, \
        "剖面南北不对称 ⇒ 经向离散有偏（cosφ 权重或极帽质心问题）"


@pytest.mark.parametrize("kw", [
    {"layer_depth_m": 0.0}, {"layer_depth_m": -1500.0}, {"layer_depth_m": float("nan")},
    {"layer_depth_m": 6.0e4},
])
def test_p10_layer_depth_domain_is_fail_loud(kw: dict):
    with pytest.raises(AssertionError):
        AtmosphereConfig(enabled=True, **kw)


def test_p10b_from_dict_falls_back_and_round_trips_new_field():
    """旧档（缺整组 `atmosphere` 或缺 `layer_depth_m`）⇒ 回退默认；带键 ⇒ 逐字段还原。

    🔴 两档都必须落回"默认关"：`enabled` 一旦从旧档里凭空变 True，关档逐位等价就废了
    （R233 T-F 家族；同 `test_c4a1_atmosphere.py::test_a14` 口径）。
    """
    default = AtmosphereConfig()
    d = SimConfig().to_dict()
    d.pop("atmosphere")
    assert SimConfig.from_dict(d).atmosphere.layer_depth_m == default.layer_depth_m
    d = SimConfig().to_dict()
    d["atmosphere"].pop("layer_depth_m")
    assert SimConfig.from_dict(d).atmosphere.layer_depth_m == default.layer_depth_m
    assert SimConfig.from_dict(d).atmosphere.enabled is default.enabled

    cfg = SimConfig()
    cfg.atmosphere = AtmosphereConfig(enabled=True, layer_depth_m=2200.0)
    back = SimConfig.from_dict(cfg.to_dict())
    assert back.atmosphere.layer_depth_m == 2200.0
    assert bool(back.atmosphere.enabled) is True


def test_p11_pole_row_pressure_is_single_valued():
    a = _atm()
    for k in range(1, 4):
        a.advance(10 * k)
        p2 = a.p_surf.reshape(a.rows_c, a.cols_c)
        assert float(p2[0].max() - p2[0].min()) == 0.0, "极行被赋多个气压值 = §四-13 重复计数"
        assert float(p2[-1].max() - p2[-1].min()) == 0.0


def test_p12_probe_is_read_only():
    a = _atm()
    before = (a.t_surf.copy(), a.p_surf.copy(), a.mass_col.copy(), a.steps, a.last_tick)
    pr = a.probe()
    assert a.t_surf.tobytes() == before[0].tobytes()
    assert a.p_surf.tobytes() == before[1].tobytes()
    assert a.mass_col.tobytes() == before[2].tobytes()
    assert (a.steps, a.last_tick) == (before[3], before[4])
    assert pr["hypso_k"] == pytest.approx(1500.0 * _G_ACCEL / _R_DRY_AIR, rel=1e-15)
