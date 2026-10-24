"""C4a-1 热机本体 Step 2（EBM 地表温度场）—— 回归单测。

权威口径 = `内部汇报/话事人传达-20261016-C4a-1热机本体开工裁定.md`（v26）
§二 Step 2 + §三 验收判据 + §四 自查清单。方程 `C·∂T/∂t = S(1−α) − (A+B·T) + D∇²T`。

覆盖（每条断言**可观测行为 / 守恒恒等式**，不是"字段存在"）：
    E1  前置闸：拿不到 C0b `surface_type` ⇒ 构造期/引擎期 fail-loud（不静默退化地表）
    E2  未知地表编码 ⇒ fail-loud（不查表回落）
    E3  初值 = 现 insolation 年平剖面（面积权粗化 + 极行闭合）⇒ 与 v26 §二 同源
    E4  🔴 收支闭合恒等式：Σarea·C·ΔT/Δt == Σarea·[S(1−α_eff)−(A+B·T)]（扩散零贡献，机器精度）
    E5  🔴 自伴两恒等式：Σarea·∇²T = 0、Σarea·T·∇²T ≤ 0（v26 §四-1/2/3）
    E6  算子 vs 连续解析：l=2 勒让德模本征值 −6D（离散一致性，s2 档实测 1.7e-4）
    E7  🔴 能量锚：稳态全球均值 ≈ +14 ℃（`planetary_offset`）；字面 α ⇒ ≈ +49 ℃（冲突①量级钉死）
    E8  稳态剖面形貌：赤道单峰、经向极差；🔴 零模型②（关扩散）极差 2.5×、两极更极端
    E9  零模型①：均匀入射 + 单一地表 ⇒ 稳态完全均匀（无经向梯度）且推进不破坏
    E10 一把尺：`area` 与 Δμ 严格成正比 ⇒ τ 的 μ→area 只换尺不换系数；极帽质心解析核对
    E11 🔴 单位纪律：`t_surf` 是 ℃、`t_kelvin()` 是 K，只有一个换算点；K 下界 > 0（§四-4/15）
    E12 非负/有限 fail-loud：NaN / T(K)≤0 / OLR 下界越界 ⇒ 抛错，不静默截断
    E13 显式稳定性硬校核：把 dt 改进不稳区 ⇒ 构造期 ValueError（§四-6/14"先查步长"）
    E14 参数域守卫：Step 2 三个新开关非法值 ⇒ 构造期 AssertionError
    E15 开档（**有物理**）60 tick 主网格逐位等价 + `t_surf` 确实演化（C7 回滚点，比 Step 1 更强）
    E16 快照：开档续跑 == 连续跑（演化后的场 + 相位）；静态驱动场不进档 = 确定性重建
    E16b 关档快照**无 `atm_*` 键**且加载后 `atmosphere is None`（旧档形状逐位不变）
    E17 零 RNG：开档跑 100 tick 与关档同流同位（v26 §四-19，纯确定性物理）
    E18 反照率闭合：α_eff ∈ (0,1) 且闭合到 α_p；`surface_raw` 档 α_eff ≡ α_surface 逐位
    E19 极帽不重复计数：极行只有一面通量（解析核对）+ 推进后极行经度恒均匀（§四-13）
    E20 读数：probe() 全有限、含 §三 判据所需量、且**只读不改状态**
    E21 松驰方向：sign(AWM dT/dt) == sign(稳态 − 当前) + `C·AWM(dT/dt)=∫吸收−A−B·T̄` 精确恒等式

稳态判据装置（E7/E8/E9）用**线性解**：EBM 热惯性 C/B ≈ 242 天，而正式档每个主 tick 只推进
45 物理秒 ⇒ 收敛到稳态需 ~10⁶ tick ⇒ 生态时窗内 T 是**准静态**的（这是 v26 参数组的固有
时间尺度差，不是实现缺陷；已写进汇报，稳态锚因此用稳态解装置而非长循环）。
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

from simulation.config import AtmosphereConfig, EnvFieldConfig, SimConfig
from simulation.sphere_engine import SphereEngine
from world import env_field as ef
from world.atmosphere import (Atmosphere, _ALBEDO_ANCHOR, _EBM_A_WM2, _EBM_B_WM2K,
                              _EBM_D_WM2K, _OLR_T_FLOOR_C)
from world.sphere_world import SphereWorld

_ROWS, _COLS = 60, 120            # 微型冒烟档（4 因子 ⇒ 粗网格 15×30）
_REL = 1e-13                      # 守恒相对残差门限（浮点求和次序级，≪ v26 的 1e-10）
_ABS = 1e-12                      # "复原/均匀"类断言门限（留 1–2 ULP 余量）
_KELVIN0 = 273.15


# ------------------------------------------------------------------ 装置

def _surface(rows: int = _ROWS, cols: int = _COLS, kind: str = "mixed") -> np.ndarray:
    """确定性地表类型层（**零 RNG**）：南北半球异质地表 + 经度纹理（让 α 有经度起伏）。

    kind="ocean" ⇒ 全海洋（零模型①装置：反照率与热容量都不随经纬度变）；
    kind="river" ⇒ 全"纯水"格（c=4182 ⇒ 热容量恰为 C_ref = 10 m 水当量的字面兑现）。
    """
    idx = np.arange(rows * cols, dtype=np.int64).reshape(rows, cols)
    if kind == "ocean":
        return np.full(idx.size, ef.SURFACE_OCEAN, dtype=np.uint8)
    if kind == "river":
        return np.full(idx.size, ef.SURFACE_RIVER, dtype=np.uint8)
    st = np.where(idx < rows * cols // 2, ef.SURFACE_OCEAN, ef.SURFACE_LAND)
    st[:, ::3] = ef.SURFACE_RIVER
    st[rows // 4, 1] = ef.SURFACE_LAKE_SALT
    st[rows // 4, 2] = ef.SURFACE_LAKE_FRESH
    st[rows // 3, :4] = ef.SURFACE_SHELF
    return st.ravel().astype(np.uint8)


def _surface_symmetric(rows: int = _ROWS, cols: int = _COLS) -> np.ndarray:
    """南北**镜像对称**的地表层（零模型③装置：对称输入必须给对称输出）。

    混合地表（`_surface`）的 α 半球不对称 ⇒ 剖面峰值自然偏离赤道一行；判"离散是否偏爱
    某个半球"必须用对称地表把物理不对称剔掉 ⇒ 这里只留经度纹理、纬度上下镜像。
    """
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


def _cfg(atmosphere: AtmosphereConfig | None = None, c0b: bool = True,
         ticks: int = 60, seed: int = 42) -> SimConfig:
    cfg = SimConfig(seed=seed)
    cfg.world.rows, cfg.world.cols = _ROWS, _COLS
    cfg.simulation.ticks = ticks
    cfg.simulation.use_sim_core = False
    cfg.env_field = EnvFieldConfig(enabled=True, c0b_enabled=c0b)
    cfg.atmosphere = atmosphere if atmosphere is not None else AtmosphereConfig(enabled=True)
    return cfg


def _engine_state(eng: SphereEngine) -> dict:
    """引擎等价面（与 Step 1 同口径：主网格 + 生物 + RNG 状态 + tick，**不含大气**）。"""
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


def _steady(a: Atmosphere) -> np.ndarray:
    """稳态线性解装置：解 `B·T − D∇²T = S(1−α_eff) − A`（逐列；rc×rc 三对角 ⇒ 稠密解）。

    与时间推进**共用同一套** `tau`/`area` ⇒ 判据核的是同一个离散算子（不是另写一份物理）。
    ⚠️ 必须尊重 `cfg.diffusion_on`（零模型②档下矩阵退化为 B·I ⇒ 极差直接由 S 的经向梯度决定）。
    """
    rc, cc = a.rows_c, a.cols_c
    rhs = a.absorbed.reshape(rc, cc) - _EBM_A_WM2
    if not bool(a.cfg.diffusion_on):
        return rhs / _EBM_B_WM2K
    m = np.zeros((rc, rc))
    for i in range(rc):
        m[i, i] = _EBM_B_WM2K
        if i > 0:
            m[i, i - 1] -= a.tau[i - 1] / a.area[i]
            m[i, i] += a.tau[i - 1] / a.area[i]
        if i < rc - 1:
            m[i, i + 1] -= a.tau[i] / a.area[i]
            m[i, i] += a.tau[i] / a.area[i]
    return np.linalg.solve(m, rhs)


# ------------------------------------------------------------------ 前置闸与初值

def test_e01_missing_surface_type_is_fail_loud():
    """E1 拿不到 C0b `surface_type` ⇒ 引擎/模块都构造期炸（不静默退化成"全球一种地表"）。"""
    with pytest.raises(ValueError, match="surface_type"):
        SphereEngine(_cfg(c0b=False))
    with pytest.raises(ValueError, match="surface_type"):
        Atmosphere(SphereWorld(_ROWS, _COLS), AtmosphereConfig(enabled=True), None)


def test_e02_unknown_surface_code_is_fail_loud():
    """E2 未知地表编码 ⇒ 炸（新增类型必须先补 C0b 热物性表，不许查表回落）。"""
    st = _surface().astype(np.uint8)
    st[7] = 9
    with pytest.raises(ValueError, match="未知编码"):
        Atmosphere(SphereWorld(_ROWS, _COLS), AtmosphereConfig(enabled=True), st)


def test_e03_initial_condition_is_the_existing_annual_profile():
    """E3 初值 = 现 insolation 年平剖面（面积权粗化 + 极行闭合），且极行经度恒均匀。

    这条把"Step 2 初值与 v26 §二 指定的真源同源"钉住：改辐照真源 ⇒ 本条立刻变红
    （而不是悄悄换一把尺）。
    """
    a = _atm()
    lat_f = np.asarray(a.world_fine._lat, dtype=np.float64)
    base = np.repeat(ef.annual_mean_base_temperature(lat_f), a.cols_f)
    expect = a.coarsen(base)
    a.collapse_poles(expect)
    assert np.array_equal(a.t_surf, expect.ravel())
    t2 = a.t_surf.reshape(a.rows_c, a.cols_c)
    assert float(np.ptp(t2[0])) == 0.0 and float(np.ptp(t2[-1])) == 0.0
    # 初值必须落在长波线性化的有效区内（否则第一次子步就炸 = 初条件本身非法）
    assert float(a.t_surf.min()) > _OLR_T_FLOOR_C


# ------------------------------------------------------------------ 守恒恒等式

def test_e04_energy_closure_is_machine_exact():
    """E4 收支闭合：`Σarea·C·ΔT/Δt == Σarea·[吸收 − (A+B·T)]`（扩散项全球积分为零）。

    🔴 非平凡：左右两边**分别**独立求和（右端不含任何来自左端的量）⇒ 只有当
    `Σarea·∇²T = 0` 真正成立时等式才成立 ⇒ 漏乘 cos φ / 不对称离散都会在这里露出来。
    """
    a = _atm()
    w = np.repeat(a.area, a.cols_c)
    before = a.t_surf.copy()
    a._ebm_substep()
    after = a.t_surf
    lhs = float((w * a.heat_cap * (after - before) / a.dt_s).sum())
    rhs = float((w * (a.absorbed - (_EBM_A_WM2 + _EBM_B_WM2K * before))).sum())
    scale = max(abs(float((w * a.absorbed).sum())), 1e-300)
    assert abs(lhs - rhs) <= _REL * scale, f"残差 {abs(lhs - rhs) / scale:.3e}"


def test_e05_laplacian_is_self_adjoint_and_dissipative():
    """E5 自伴两恒等式：`Σarea·∇²T = 0`（机器精度）、`Σarea·T·∇²T ≤ 0`（v26 §四-1/2）。"""
    a = _atm()
    w = np.repeat(a.area, a.cols_c)
    rc, cc = a.rows_c, a.cols_c
    for prof in (np.sin(3.0 * a.mu_ctr), a.mu_ctr ** 2 - 0.3,
                 np.where(a.mu_ctr > 0.0, 1.0, -1.0)):
        t2 = np.broadcast_to(np.asarray(prof, dtype=np.float64)[:, None], (rc, cc)).copy()
        t2 += 0.01 * np.arange(cc, dtype=np.float64)[None, :]     # 加经度起伏（非零纹理）
        lt = a._laplacian(t2)
        num = float((w * lt.ravel()).sum())
        den = float((w * np.abs(lt.ravel())).sum())
        assert abs(num) <= _REL * max(den, 1e-300), f"∫∇²T 残差 {num / den:.3e}"
        # 耗散性：Σ T·LT = −Σ_面 τ·(ΔT)² ≤ 0（造热 = 符号写反，这里必红）
        assert float((w * t2.ravel() * lt.ravel()).sum()) <= 0.0
        # 恒等式右边**独立**核对一次（望远镜相消不是只靠左边=0 蒙过去）
        dT = t2[1:] - t2[:-1]
        expect = -float((a.tau[:, None] * dT ** 2).sum())
        got = float((w * t2.ravel() * lt.ravel()).sum())
        assert abs(got - expect) <= _REL * max(abs(expect), 1e-300)


def test_e06_operator_matches_analytic_legendre_mode():
    """E6 离散算子 vs 连续 `D ∂_μ[(1−μ²)∂_μ]`：l=2 模本征值 = −6D（s2 档实测 1.7e-4）。"""
    a = _atm(480, 960, coarse_factor=4)
    p2 = 0.5 * (3.0 * a.mu_ctr ** 2 - 1.0)                        # μ=sin φ 上的 l=2 勒让德
    t2 = np.broadcast_to(p2[:, None], (a.rows_c, a.cols_c)).copy()
    lt = a._laplacian(t2)
    cont = -6.0 * _EBM_D_WM2K * p2
    rel = float(np.max(np.abs(lt[:, 0] - cont)) / np.max(np.abs(cont)))
    assert rel < 2e-3, f"l=2 模偏差 {rel:.3e}"


def test_e10_area_is_exactly_proportional_to_mu_thickness():
    """E10 一把尺：`area_i/(Δμ_i·factor/Δφ)` 是常数 ⇒ τ 只换尺不换系数（§四-16 的尺）。"""
    a = _atm()
    dmu = np.diff(a.mu_edge)
    dphi = float(a.world_fine._lat[1] - a.world_fine._lat[0])
    ratio = a.area / (dmu * (float(a.factor) / dphi))
    assert float(np.max(ratio) / np.min(ratio) - 1.0) < 1e-5
    # 极帽质心 = (μ₁−1)/2（不是行心）⇒ 极点格不被当成"一行普通格"重复计数
    # （门限 1e-14：两种写法在浮点上本就差 ~1 ULP，这里核的是**口径**不是最后一位）
    assert abs(float(a.mu_ctr[0] - 0.5 * (a.mu_edge[1] - 1.0))) < 1e-14
    assert abs(float(a.mu_ctr[-1] - 0.5 * (1.0 + a.mu_edge[-2]))) < 1e-14
    # 质心严格单调 + 关于赤道反对称（对称网格 ⇒ 算子不该偏爱某个半球）
    assert bool(np.all(np.diff(a.mu_ctr) > 0.0))
    assert float(np.max(np.abs(a.mu_ctr + a.mu_ctr[::-1]))) < 1e-14


# ------------------------------------------------------------------ 判据锚与零模型

def test_e07_global_mean_anchor_and_the_albedo_conflict_magnitude():
    """E7 🔴 能量锚：闭合档稳态全球均值 ≈ +14 ℃；字面 α 档 ≈ +49 ℃（冲突①量级钉死）。

    v26 §二 同时写了「α 按 surface_type」（Step 2）与解析锚 (340×0.7−210)/2=+14 ℃，
    两者**互斥**（锚隐含 α_p=0.30，本世界面积权 α̅≈0.093）。默认档 `planetary_offset`
    满足锚；`surface_raw` 保留字面口径 ⇒ 两条读数都在这里被钉住，改动会立刻变红。
    🔴 不是"调参凑出来的"：A/B/D/C/α_p 全取 v26 明文值，闭合偏移只由 α_p 与面积权 α̅ 决定。
    微型档读数 +14.08 / +48.7；s2 档 = +14.10 / +49.4（差在世界 α̅ 的海陆比）。
    """
    a = _atm()
    t_bar = a.awm(_steady(a))
    assert abs(t_bar - 14.0) < 0.5, f"闭合档全球均值 {t_bar:+.3f} ℃（锚 +14）"
    raw = _atm(albedo_closure="surface_raw")
    t_raw = raw.awm(_steady(raw))
    assert 45.0 < t_raw < 55.0, f"字面 α 档全球均值 {t_raw:+.3f} ℃（应 ≈ +49）"
    assert t_raw - t_bar > 30.0                                  # 冲突量级 = 35 ℃ 级
    # 全球均值是**通量恒等式**（∫∇²T=0 ⇒ B·T̄ = ∫吸收 − A），与网格无关 ⇒ 只依赖吸收
    assert abs(t_bar - (a.awm(a.absorbed) - _EBM_A_WM2) / _EBM_B_WM2K) < _ABS


def test_e08_steady_profile_shape_and_no_diffusion_null_model():
    """E8 剖面形貌 + 🔴 零模型②（关扩散 ⇒ 无经向热输送、极值更极端，v26 §三）。

    三条独立判据：① **对称地表 ⇒ 对称剖面**（离散不偏爱半球，峰值恰在赤道）；
    ② 混合地表的剖面**单峰**（从赤道到两极单调变冷 ⇒ 热机方向正确）；
    ③ 关扩散后极差放大、两极更冷、赤道更热，而**全球均值不变**（∫∇²T=0 的正面证据）。
    """
    sym = _atm(surface=_surface_symmetric())
    zs = _steady(sym).mean(axis=1)
    i_eq = int(np.argmin(np.abs(sym.mu_ctr)))
    assert int(np.argmax(zs)) == i_eq                             # 对称 ⇒ 峰值恰在赤道行
    assert float(np.max(np.abs(zs - zs[::-1]))) < 1e-9            # 逐位对称（只余 LU 舍入）

    a = _atm()
    t_full = _steady(a)
    zm = t_full.mean(axis=1)
    ip = int(np.argmax(zm))
    assert abs(ip - i_eq) <= 1                                    # 混合地表：峰在赤道一带
    assert bool(np.all(np.diff(zm[:ip + 1]) > 0.0))               # 峰侧单调升
    assert bool(np.all(np.diff(zm[ip:]) < 0.0))                   # 峰侧单调降 ⇒ 全球单峰
    assert zm[0] < zm[1] and zm[-1] < zm[-2]                      # 两极是各自的极值

    spread_on = float(zm.max() - zm.min())
    b = _atm(diffusion_on=False)
    t_off = _steady(b)
    zm_off = t_off.mean(axis=1)
    spread_off = float(zm_off.max() - zm_off.min())
    assert spread_off > 2.0 * spread_on, f"关扩散极差 {spread_off} vs 开扩散 {spread_on}"
    assert float(zm_off.min()) < float(zm.min())                  # 极地更冷
    assert float(zm_off.max()) > float(zm.max())                  # 赤道更热
    # 全球均值**不**受扩散影响（∫∇²T=0）⇒ 两档 AWM 相同（到机器精度）
    assert abs(a.awm(t_full) - b.awm(t_off)) < 1e-9
    # 扩散是把热从热的地方搬到冷的地方 ⇒ 每个极值都被拉回全球均值（不产生新极值）
    t_bar = a.awm(t_full)
    assert float(zm.max()) > t_bar > float(zm.min())
    assert abs(float(zm_off.max()) - float(zm.max())) > 1.0       # 不是"关掉等于没关"


def test_e09_uniform_insolation_null_model_is_zonally_uniform():
    """E9 零模型①：均匀入射 + 单一地表 ⇒ 稳态**完全均匀**（无经向梯度，v26 §三）。

    这是"漏乘 cos φ / 极带重复计数"的第二号探测器：几何错 ⇒ 均匀驱动也会长出梯度。
    """
    a = Atmosphere(SphereWorld(_ROWS, _COLS),
                   AtmosphereConfig(enabled=True, insolation_mode="uniform"),
                   _surface(kind="ocean"))
    t_star = _steady(a)
    assert float(np.ptp(t_star)) < _ABS
    expect = (a.awm(a.absorbed) - _EBM_A_WM2) / _EBM_B_WM2K
    assert abs(float(t_star[0, 0]) - expect) < 1e-9
    # 均匀态是 EBM 的不动点 ⇒ 推进不产生梯度（也不产生"数值呼吸"）
    a.t_surf[:] = expect
    for tick in (10, 20, 30):
        a.advance(tick)
    assert float(np.ptp(a.t_surf)) < _ABS
    assert a.probe()["zonal_spread_max"] < _ABS
    assert abs(a.awm(a.t_surf) - expect) < 1e-9


# ------------------------------------------------------------------ 单位与 fail-loud

def test_e11_temperature_unit_boundary():
    """E11 🔴 单位纪律（v26 §四-4）：`t_surf`=℃、`t_kelvin()`=K，且 K 下界 > 0。

    ℃ 用于 `A+B·T`（v26 自己的锚只在 ℃ 成立），K 用于状态方程（Step 3）⇒ 两个口径之间
    必须**只有一个换算点**，这里把它钉成可断言的 API。
    """
    a = _atm()
    assert np.array_equal(a.t_kelvin(), a.t_surf + _KELVIN0)
    assert float(a.t_kelvin().min()) > 0.0
    assert a.probe()["t_kelvin_min"] > 0.0
    # 混用 K 会被锚反证：把 ℃ 当 K 代入 ⇒ OLR≈A+B·288≈786 W/m² ⇒ T̄→荒谬
    assert _EBM_A_WM2 + _EBM_B_WM2K * 288.0 > 700.0
    # ℃ 与 K 的分界就在 OLR 下界：−105 ℃ 以下线性化 OLR 变负
    assert _EBM_A_WM2 + _EBM_B_WM2K * (_OLR_T_FLOOR_C + 1.0) > 0.0
    assert _EBM_A_WM2 + _EBM_B_WM2K * (_OLR_T_FLOOR_C - 1.0) < 0.0
    assert abs(_OLR_T_FLOOR_C + _EBM_A_WM2 / _EBM_B_WM2K) < 1e-15


@pytest.mark.parametrize("fill", [
    float("nan"),        # NaN 必须炸（不静默替换）
    -300.0,              # T(K) < 0
    -106.0,              # OLR = A+B·T < 0（长波线性化的外推区）
    float("inf"),
])
def test_e12_nonfinite_and_out_of_range_are_fail_loud(fill: float):
    """E12 非负/有限 fail-loud（v26 §四-15）：任意子步出现异常 ⇒ 报错，禁止截断。"""
    a = _atm()
    a.t_surf[:] = fill
    with pytest.raises(FloatingPointError):
        a._ebm_substep()
    # 未被就地写坏 ⇒ 炸在第 1 步，不留"半更新"状态
    assert not np.all(np.isfinite(a.t_surf)) or float(a.t_surf.min()) <= _OLR_T_FLOOR_C


def test_e13_explicit_stability_is_checked_at_construction():
    """E13 显式稳定性硬校核（§四-6/14）：把 dt 改进不稳区 ⇒ 构造期 ValueError。

    正式档（s2）实测 μ≈1.6e-2、微型档 1.8e-4，都 ≪ 门限 0.5 ⇒ 校核平时不干扰；
    但 `dt_s=5e5`（微型档 μ≈0.59）必炸 ⇒ 谁把节拍改进不稳区，构造期就拦住，
    而不是跑一万 tick 后收到 NaN（"不稳定先查步长，别加非物理平滑"）。

    ⚠️ 本测试**只核扩散门** ⇒ 一律 `damping_on=False`：Step 3b 另有一个同族校核
    `dt/τ ≤ 0.5`，它先于 μ 达到不稳区（τ=86400 s 时 dt≈43 200 s，而 μ 门限要到 4×10⁵ s）
    ⇒ 不关掉就会两个门互相遮蔽。阻尼门自身由 `tests/test_c4a1_wind.py` 的 W 系列专测。
    """
    _atm(damping_on=False)                                  # 默认档：构造通过
    with pytest.raises(ValueError, match="稳定数"):
        _atm(dt_s=5.0e5, damping_on=False)
    # 校核随 dt 单调（不是随机阈值）：临界点 μ=0.5 ⇒ dt ≈ 0.5/μ(150)·150
    critical = 0.5 / _atm(damping_on=False)._mu_diff * 120.0   # v38 汇率：dt_s=120
    assert 1.0e5 < critical < 5.0e5
    _atm(dt_s=critical * 0.5, damping_on=False)             # 门内
    with pytest.raises(ValueError, match="稳定数"):
        _atm(dt_s=critical * 2.0, damping_on=False)


@pytest.mark.parametrize("kw", [
    {"albedo_closure": "planetary"},                        # 拼写错误不许静默走默认
    {"albedo_closure": 1},
    {"insolation_mode": "yearly"},
    {"insolation_mode": None},
    {"diffusion_on": "yes"},
    {"diffusion_on": 1},
])
def test_e14_new_switch_domain_is_fail_loud(kw: dict):
    """E14 参数域守卫：三个 Step 2 开关的非法值 ⇒ 构造期 AssertionError。"""
    with pytest.raises(AssertionError):
        AtmosphereConfig(enabled=True, **kw)


# ------------------------------------------------------------------ 引擎侧与快照

def test_e15_on_mode_leaves_main_grid_bit_identical():
    """E15 开档（**有物理**）60 tick ⇒ 主网格/生物/RNG 与关档逐位等价，而 `t_surf` 演化。

    本批大气只**读**主网格、不写任何主网格场 ⇒ C7 回滚点比 Step 1 更强：
    物理开着跑，生态侧仍是同一条轨迹（Step 4 起若开始反哺，本条要显式改写）。
    """
    on_cfg, off_cfg = _cfg(), _cfg()
    off_cfg.atmosphere.enabled = False
    on, off = SphereEngine(on_cfg), SphereEngine(off_cfg)
    for _ in range(60):
        on.step()
        off.step()
    assert _engine_state(on) == _engine_state(off)
    fresh = SphereEngine(on_cfg)
    assert not np.array_equal(on.atmosphere.t_surf, fresh.atmosphere.t_surf)
    assert on.atmosphere.steps == 18 and on.atmosphere.last_tick == 60
    # 演化方向合法：全球均值向稳态 +14 ℃ 靠（初值 AWM ≈ +14.07 ⇒ 不该跑到别处）
    assert abs(on.atmosphere.probe()["t_awm"] - 14.0) < 2.0


def test_e16_snapshot_resume_equals_continuous_run(tmp_path: Path):
    """E16 开档快照：续跑 == 连续跑（演化后的温度场 + 子步相位，v26 §四-18）。"""
    cfg = _cfg()
    e1 = SphereEngine(cfg)
    for _ in range(30):
        e1.step()
    path = tmp_path / "atm_ebm.npz"
    e1.save_snapshot(str(path))
    r = SphereEngine.load_snapshot(str(path))
    assert np.array_equal(r.atmosphere.t_surf, e1.atmosphere.t_surf)
    assert (r.atmosphere.steps, r.atmosphere.last_tick) == (
        e1.atmosphere.steps, e1.atmosphere.last_tick)
    for _ in range(20):
        e1.step()
        r.step()
    assert np.array_equal(r.atmosphere.t_surf, e1.atmosphere.t_surf)
    # 静态驱动场（α/C/S/τ）**不进快照** ⇒ 由 config + surface_type 确定性重建（零 RNG）
    # Step 3a 的诊断量 `p_surf`/`mass_col` 同样不进档（`t_surf` 的纯函数 ⇒ restore 重算）；
    # Step 3b 的 `atm_u_east`/`atm_u_north` **必须进档**（时间积分的状态量，重建不出来）。
    with np.load(str(path), allow_pickle=False) as data:
        assert sorted(k for k in data.files if k.startswith("atm_")) == \
            ["atm_last_tick", "atm_steps", "atm_t_surf", "atm_u_east", "atm_u_north"]
    assert r.atmosphere.cfg.albedo_closure == "planetary_offset"


def test_e16b_off_mode_snapshot_has_no_atm_keys(tmp_path: Path):
    """E16b 关档快照**无 `atm_*` 键**（旧快照形状逐位不变）且加载后 `atmosphere is None`。"""
    cfg = _cfg()
    cfg.atmosphere.enabled = False
    eng = SphereEngine(cfg)
    for _ in range(11):
        eng.step()
    path = tmp_path / "atm_off.npz"
    eng.save_snapshot(str(path))
    with np.load(str(path), allow_pickle=False) as data:
        assert not [k for k in data.files if k.startswith("atm_")]
    back = SphereEngine.load_snapshot(str(path))
    assert back.atmosphere is None
    assert _engine_state(back)["tick"] == 11


def test_e17_zero_rng_consumption_with_physics():
    """E17 零 RNG：开档跑 100 tick 与同 seed 关档**同流同位**（v26 §四-19）。

    物理全为确定性算式 ⇒ 不许碰引擎 RNG（否则 A/B 档不可比、跨 seed 复现失效）。
    `rng.draws` 不入快照 ⇒ 这里只在"同构造"的两个引擎之间比。
    """
    on_cfg, off_cfg = _cfg(), _cfg()
    off_cfg.atmosphere.enabled = False
    on, off = SphereEngine(on_cfg), SphereEngine(off_cfg)
    assert int(on.rng.draws) == int(off.rng.draws)          # 构造本身不动流
    for _ in range(100):
        on.step()
        off.step()
    assert int(on.rng.draws) == int(off.rng.draws)
    assert on.rng.bit_generator.state == off.rng.bit_generator.state
    assert on.atmosphere.steps == 30


def test_e18_albedo_closure_bounds_and_raw_identity():
    """E18 α_eff ∈ (0,1)（越界构造期已炸）；`surface_raw` 档 α_eff ≡ α_surface 逐位。"""
    a = _atm()
    assert float(a.alpha_eff.min()) > 0.0 and float(a.alpha_eff.max()) < 1.0
    assert abs(a.awm(a.alpha_eff) - _ALBEDO_ANCHOR) < 0.02      # 闭合到行星反照率
    assert abs(float(a.albedo_offset) - (_ALBEDO_ANCHOR - a.awm(a.alpha_surface))) < 1e-15
    raw = _atm(albedo_closure="surface_raw")
    assert np.array_equal(raw.alpha_eff, raw.alpha_surface)     # 字面档 = 不闭合
    assert float(raw.albedo_offset) == 0.0
    # 面积权粗化不产生新极值（α 落在 C0b 查表值域内 ⇒ 没有"采样造出的反照率"）
    tab = np.array([ef.SURFACE_THERMOPHYSICS[k][:2]
                    for k in sorted(ef.SURFACE_THERMOPHYSICS)], dtype=np.float64)
    assert float(a.alpha_surface.min()) >= float(tab[:, 0].min()) - 1e-15
    assert float(a.alpha_surface.max()) <= float(tab[:, 0].max()) + 1e-15
    # 热容量 C = C_ref·c/c_w（10 m 水当量）⇒ 严格正；面积权粗化只取块内均值 ⇒
    # 值域必须**夹在表的 c 值之间**（采样不造出新极值 = 守恒采样的物理意义）
    assert float(a.heat_cap.min()) > 0.0
    from world.atmosphere import _C_REF, _C_WATER
    c_lo = _C_REF * float(tab[:, 1].min()) / _C_WATER
    c_hi = _C_REF * float(tab[:, 1].max()) / _C_WATER
    assert float(a.heat_cap.min()) >= c_lo - 1e-9
    assert float(a.heat_cap.max()) <= c_hi + 1e-9
    # 纯水（c=4182）格 ⇒ C 恰为 C_ref ⇒ 常数 4.182e7 是"10 m 水当量"的字面兑现（v26 ≈4.2e7）
    pure = Atmosphere(SphereWorld(_ROWS, _COLS), AtmosphereConfig(enabled=True),
                      _surface(kind="river"))
    assert abs(float(pure.heat_cap.max()) - _C_REF) < 1e-9 * _C_REF


def test_e19_pole_row_has_single_face_flux():
    """E19 极帽：只有一面通量（μ=±1 处 1−μ²=0 ⇒ 边界通量恒零），且推进后极行经度恒均匀。

    v26 §四-13 的"重复计数"变体：极点 = 一圈一格，若把它当普通行处理，同一个极点格
    会被赋予 cols_c 个矛盾值 ⇒ 全球积分随 cols_c 漂移。极行只有一面通量 = 这里解析核对。
    """
    a = _atm()
    rc = a.rows_c
    t2 = np.broadcast_to(np.arange(rc, dtype=np.float64)[:, None], (rc, a.cols_c)).copy()
    lt = a._laplacian(t2)
    assert abs(float(lt[0, 0]) - float(a.tau[0] * (t2[1, 0] - t2[0, 0]) / a.area[0])) < 1e-15
    assert abs(float(lt[-1, 0]) + float(a.tau[-1] * (t2[-1, 0] - t2[-2, 0]) / a.area[-1])) < 1e-15
    # 极行的驱动项单值 ⇒ 推进后仍单值（一圈一格不被拆成 cols_c 个矛盾值）
    for tick in (10, 20):
        a.advance(tick)
    t_new = a.t_surf.reshape(rc, a.cols_c)
    assert float(np.ptp(t_new[0])) == 0.0 and float(np.ptp(t_new[-1])) == 0.0
    # 极行热通量符号合法：南极只被"往北输热"拖走 ⇒ LT_0 与 (T_1−T_0) 同号（耗散，不造热）
    assert float(np.sign(lt[0, 0]) * np.sign(t2[1, 0] - t2[0, 0])) >= 0.0


def test_e20_probe_is_finite_and_read_only():
    """E20 读数：probe() 全有限、含 §三 判据所需量，且**只读不改状态**。"""
    a = _atm()
    before = a.t_surf.copy()
    p = a.probe()
    for k in ("t_awm", "absorbed_awm", "s_awm", "olr_awm", "imbalance_rel", "mu_diff_max",
              "albedo_offset", "alpha_surface_awm", "t_eq", "t_pole_n", "t_pole_s",
              "zonal_spread_max", "area_rel_residual", "t_kelvin_min",
              "ebm_A_WM2", "ebm_B_WM2K", "ebm_D_WM2K"):
        assert k in p, k
        assert np.isfinite(p[k]), (k, p[k])
    assert p["mu_diff_max"] < 0.5
    assert p["area_rel_residual"] < _REL
    assert np.array_equal(a.t_surf, before)                       # 只读


def test_e21_relaxation_direction_is_thermodynamically_legal():
    """E21 松驰方向：全球均值趋势的符号 = sign(稳态 − 当前)（🔴 不许"越松越偏"）。

    用**单一地表**（热容量逐格相同）⇒ `C·AWM(dT/dt) = AWM(吸收) − A − B·T̄` 是精确恒等式
    （∫∇²T=0），符号判据不依赖任何近似。⚠️ 生态时窗内每主 tick 只推进 45 物理秒而
    C/B≈242 天 ⇒ 到位需要 ~10⁶ tick（准静态，见模块头）；本条因此只核**方向**，
    不核"跑了 50 tick 就接近稳态"——那种断言是在掩盖时间口径问题。
    """
    a = _atm(surface=_surface(kind="ocean"))
    t_bar = a.awm(a.t_surf)
    t_star = a.awm(_steady(a))
    td = a.awm(a.tendency())
    assert float(np.ptp(a.heat_cap)) / float(a.heat_cap.max()) < 1e-15   # 单一地表 ⇒ C 逐格相同
    assert np.sign(td) == np.sign(t_star - t_bar)
    # 精确恒等式：C·AWM(dT/dt) == AWM(吸收) − (A + B·T̄)（扩散项全球积分为零）
    lhs = float(a.heat_cap[0]) * td
    rhs = a.awm(a.absorbed) - (_EBM_A_WM2 + _EBM_B_WM2K * t_bar)
    assert abs(lhs - rhs) <= 1e-9 * max(abs(rhs), 1.0)
    # 趋势幅度与时间尺度自洽：松弛率 = B/C ⇒ |dT̄/dt| = |T̄*−T̄|/(C/B)（不"越松越猛"）
    assert abs(td) < abs(t_star - t_bar) / (a.heat_cap[0] / _EBM_B_WM2K) * 1.01
    # 离散极值原理（"扩散不过冲"）：全场最热格的 ∇²T ≤ 0、最冷格 ≥ 0 ⇒ 扩散不造新极值
    t2 = a.t_surf.reshape(a.rows_c, a.cols_c)
    lt = a._laplacian(t2).ravel()
    assert float(lt[int(np.argmax(t2))]) <= 0.0
    assert float(lt[int(np.argmin(t2))]) >= 0.0
