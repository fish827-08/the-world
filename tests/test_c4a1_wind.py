"""C4a-1 热机本体 Step 3b（压力梯度 + Rayleigh 阻尼 → 风）—— 回归单测。

权威口径 = `内部汇报/话事人传达-20261016-C4a1热机本体开工裁定.md`（v26）§二 Step 3
「压力梯度 −∇P/ρ + Rayleigh 阻尼（**无科氏力**）→ 风 u」+ §三 验收判据（单暖斑 ⇒ 近地面
风向暖斑辐合）+ §四-7/11/12/13/14/15/17/18/19 + v30 ⑤.4（Rayleigh τ 必答：风能否在
10-tick 节拍内跟上压力梯度）。

覆盖（每条都是可观测行为 / 解析恒等式，不是"字段存在"）：
    W1  🔴 §四-7 头号判据：单暖斑 ⇒ 四周的风**吹向斑心**（低层辐合的**风侧**证据）
    W2  🔴 W1 的**鉴别力**自证：把气压换成 §四-9 点名的错写 `P=P₀·T/T₀` ⇒ 风向必然
        **背离**暖斑 ⇒ 说明 W1 不是恒真断言（本批不写死结果，也不写"永不为假"的测试）
    W3  因果方向不颠倒：`_integrate_wind()` / `wind_forces()` **只写 u**（T/P/mass/节拍
        逐位不动）⇒ v26 §二「风不反过来加热/平流热量」在函数级被钉住
    W4  🌍 v30 ⑤.4 的解析律：恒定力下 `u_k = τ·F·(1−(1−dt/τ)^k)` **逐格成立**（离散精确解）
        ⇒ τ 的物理含义 = 达到平衡风 `u_eq=τ·F` 的时间常数，不是装饰
    W5  零模型③：`damping_on=False` ⇒ `u = k·dt·F` **线性无界**；同场开阻尼 ⇒ 有界 ≤τ·|F|
        ⇒ 反证 Rayleigh 摩擦是**必需项**（v26 §三"全部要跑"）
    W6  §四-14 极地滤波：纬向力的放大倍率被 `max(c,0.1)` **截到 10 倍封顶**，而 c≥0.1
        的行**不受裁**（不裁 = 极圈 54 倍；乱裁 = 改中纬物理 ⇒ 两头都不许）
    W7  经向力 ∝ `cos φ`（"北极点没有北"）：极点归零、中纬不受裁 ⇒ 与 W6 的**相反**行为
        同时成立才说明两个方向的几何都写对了
    W8  §四-12 经度接缝：把气压场整体平移一列 ⇒ 风场也整体平移一列（内圈**逐位**）
    W9  §四-18 快照：u **必须进档**（键集合 = 5）+ 往返逐位 + 旧档缺键 ⇒ 静止起算 +
        形状不符 ⇒ fail-loud；W9b 引擎级续跑 == 连续跑（u/P/T 三路都比）
    W10 🔴 关档逐位等价（C7 回滚点）+ 开档**零 RNG 消费** + 关档快照无 `atm_*` 键
    W11 fail-loud：P/T 含 NaN/Inf ⇒ `FloatingPointError`；`dt/τ > 0.5` ⇒ 构造期 `ValueError`
        （而 `damping_on=False` 时同一 dt_s **不该**炸 ⇒ 校核与物理项一一对应）
    W12 参数域守卫：`rayleigh_tau_s` 非法 / `damping_on` 非布尔 ⇒ AssertionError；
        旧存档缺两个新字段 ⇒ 回退默认、带字段 ⇒ 往返
    W13 读数只读：`probe()` 前后所有场与计数逐位不变；新增风读数全有限
    W14 结构判据：低层风**两支都朝赤道**（赤道辐合的气压驱动侧）+ 南北镜像（`u_n` 反对称、
        `u_e` 对称，对称地表装置）
    W15 力平衡而非地转：`AWM(u·F) > 0`（风**穿过等压线**流向低压）——本批无科氏力 ⇒
        不可能是 90° 转定的地转风（那会让 `u·F ≡ 0`）
    W16 🔴 向量化力 == **朴素三重循环参考实现**（真演化场当输入；抓 `np.roll` 接缝错位、
        极行折叠、方向因子写反——同"双路径逐位对拍"的口径，只是这里两条路都在 Python）
    W17 🔴 §四-11 质量连续：`Σ area·∇·u = 0` 到**机器精度**（真实风 / 纯经向 / 纯纬向三档，
        两个方向的相消机制分开钉）
    W18 §四-7 散度侧：单暖斑 ⇒ 斑心**辐合**（`div<0`）+ 守恒推论"必须有辐散区且不在斑心"
    W19 `div_u()` 只读 + 对 u **一次齐次**（无极外面漏进的常数项）+ 静止风 ⇒ 散度恰零
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest

from simulation.config import AtmosphereConfig, EnvFieldConfig, SimConfig
from simulation.sphere_engine import SphereEngine
from world import env_field as ef
from world.atmosphere import (Atmosphere, _DAMP_STABILITY_MAX, _EARTH_RADIUS_M,
                              _P_REF_PAS, _POLAR_C_FLOOR, _R_DRY_AIR)
from world.sphere_world import SphereWorld

_ROWS, _COLS = 60, 120            # 微型冒烟档（4 因子 ⇒ 粗网格 15×30）
_REL = 1e-12
_ABS = 1e-12


# ------------------------------------------------------------------ 装置

def _surface(rows: int = _ROWS, cols: int = _COLS, kind: str = "mixed") -> np.ndarray:
    """确定性地表类型层（**零 RNG**）；kind="ocean" ⇒ 驱动场只随纬度变（干净装置）。"""
    idx = np.arange(rows * cols, dtype=np.int64).reshape(rows, cols)
    if kind == "ocean":
        return np.full(idx.size, ef.SURFACE_OCEAN, dtype=np.uint8)
    st = np.where(idx < rows * cols // 2, ef.SURFACE_OCEAN, ef.SURFACE_LAND)
    st[:, ::3] = ef.SURFACE_RIVER
    return st.ravel().astype(np.uint8)


def _surface_symmetric(rows: int = _ROWS, cols: int = _COLS) -> np.ndarray:
    """南北**镜像对称**地表（对称输入必须给对称输出，同 Step 2 E8 / Step 3a P9b 装置）。"""
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
    """引擎等价面（与 Step 1/2/3a 同口径：主网格 + 生物 + RNG 状态 + tick，**不含大气**）。"""
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


def _warm_spot(a: Atmosphere, amp: float = 8.0, bg: float = 14.0):
    """均匀背景 + **赤道带高斯暖斑**（纬向用周期距离 ⇒ 接缝连续，§四-12）。

    背景取均匀 ⇒ 气压的**唯一**水平结构来自暖斑 ⇒ 风向判据不被背景环流污染
    （背景环流另有 W14 专测）。返回斑心 (i0, j0)。
    """
    rc, cc = a.rows_c, a.cols_c
    i0, j0 = rc // 2, cc // 4
    di = (np.arange(rc)[:, None] - i0) ** 2
    dj = np.minimum(np.abs(np.arange(cc) - j0),
                    cc - np.abs(np.arange(cc) - j0)) ** 2
    t2 = np.full((rc, cc), bg) + amp * np.exp(-di / 8.0 - dj / 8.0)
    a.t_surf[:] = t2.ravel()
    a._diagnose_pressure()
    return i0, j0


def _ramp_pressure(a: Atmosphere, slope: float = 0.02) -> None:
    """把 `p_surf` 置成**经向线性斜坡**（`P₀(1+slope·μ)`、纬向恒等）⇒ 力在时间上恒定。

    🔴 只用于 W4/W5 的"解析律"装置：`_integrate_wind()` 不写 P（W3 已证）⇒ 力恒定 ⇒
    `u_k` 有**离散精确解**可比。这不是"造一个物理场景"，是把 τ 的定义拿出来单独称。
    """
    p = _P_REF_PAS * (1.0 + slope * a.mu_ctr)
    a.p_surf[:] = np.repeat(p, a.cols_c)


def _hand_forces(a: Atmosphere, p2: np.ndarray, t_k: np.ndarray):
    """**朴素逐格**复算 `(F_east, F_north)`（三重 Python 循环，独立于向量化实现）。

    与 `Atmosphere.wind_forces()` 的差别是**写法**（手算下标、不用 `np.roll`、逐格标量
    除法）⇒ 能抓到"接缝/极行/方向因子"级别的错位，而不只是把同一份代码再跑一遍。
    离散口径必须**同式**（面梯度 → 节点取相邻两面平均；极行折环带均值），否则对拍的是
    两个不同格式，差异属物理而非笔误。
    """
    rc, cc = a.rows_c, a.cols_c
    dmu = a._dmu_ctr
    dl = float(a._dlam_c)

    def grad_north(i: int, j: int) -> float:
        if i == 0:                                     # μ=−1 处无"外面" ⇒ 只取内面
            return (float(p2[1][j]) - float(p2[0][j])) / float(dmu[0])
        if i == rc - 1:
            return (float(p2[rc - 1][j]) - float(p2[rc - 2][j])) / float(dmu[rc - 2])
        lo = (float(p2[i][j]) - float(p2[i - 1][j])) / float(dmu[i - 1])
        hi = (float(p2[i + 1][j]) - float(p2[i][j])) / float(dmu[i])
        return 0.5 * (lo + hi)

    def grad_east(i: int, j: int) -> float:            # 经度**周期**（接缝与普通列同式）
        hi = (float(p2[i][(j + 1) % cc]) - float(p2[i][j])) / dl
        lo = (float(p2[i][j]) - float(p2[i][(j - 1) % cc])) / dl
        return 0.5 * (hi + lo)

    fe = np.zeros((rc, cc))
    fn = np.zeros((rc, cc))
    for i in range(rc):
        c = float(a._c_ctr[i])
        ce = max(c, _POLAR_C_FLOOR)                    # 🔴 只有**纬向**分母被滤波
        for j in range(cc):
            irho = _R_DRY_AIR * float(t_k[i][j]) / float(p2[i][j])
            fn[i][j] = -(c * irho / _EARTH_RADIUS_M) * grad_north(i, j)
            fe[i][j] = -(irho / (_EARTH_RADIUS_M * ce)) * grad_east(i, j)
        if i == 0 or i == rc - 1:                      # 一圈一格 ⇒ 极行的力折成环带均值
            fn[i] = float(np.mean(fn[i]))
            fe[i] = float(np.mean(fe[i]))
    return fe, fn


def _polar_rows(a: Atmosphere) -> tuple:
    """首末行（极点"一圈一格"的两行）索引，便于把极行判据写成一行。"""
    return (0, a.rows_c - 1)


# ---------------------------------------------------------------- 风向与因果方向

def test_w1_near_surface_wind_converges_into_the_warm_spot():
    """🔴 §四-7 / v26 §三：单暖斑 ⇒ 四周的风**吹向斑心**（低层辐合）。"""
    a = _atm(surface=_surface(kind="ocean"))
    rc, cc = a.rows_c, a.cols_c
    i0, j0 = _warm_spot(a)
    for _ in range(3):
        a._integrate_wind()
    ue = a.u_east.reshape(rc, cc)
    un = a.u_north.reshape(rc, cc)
    assert ue[i0, (j0 + 1) % cc] < -1e-12, "斑东侧的风必须向西（朝斑）"
    assert ue[i0, (j0 - 1) % cc] > 1e-12, "斑西侧的风必须向东（朝斑）"
    assert un[i0 + 1, j0] < -1e-12, "斑北侧的风必须向南（朝斑）"
    assert un[i0 - 1, j0] > 1e-12, "斑南侧的风必须向北（朝斑）"
    # 斑心是气压极小 ⇒ 自身不受力（对称性），四周净流入 = 辐合
    assert abs(ue[i0, j0]) < 1e-12 and abs(un[i0, j0]) < 1e-12, \
        "暖斑中心的梯度由对称性相消 ⇒ 风必须 ≈0（不为 0 说明梯度算子错位）"


def test_w2_wrong_pressure_law_would_flip_the_wind():
    """W1 必须**有鉴别力**：错写 `P=P₀·T/T₀`（同向）⇒ 风必然背离暖斑（= 单胞反号）。

    这条不是"再跑一遍 W1"：它把 `p_surf` 换成 §四-9 点名的错写，**同一个**梯度+风算子
    必须给出**相反符号**的四邻 ⇒ 说明 W1 的四条断言真能证伪（错实现会让它恒真）。
    """
    a = _atm(surface=_surface(kind="ocean"))
    rc, cc = a.rows_c, a.cols_c
    i0, j0 = _warm_spot(a)
    p_right = a.p_surf.reshape(rc, cc).copy()
    tk = a.t_kelvin()
    a.p_surf[:] = (_P_REF_PAS * tk / a.awm(tk)).ravel()      # 🔴 §四-9 点名的错写
    assert p_right[i0, j0] < p_right[i0, (j0 + 3) % cc], \
        "真公式下暖斑是**低压**（否则 W1 的装置本身已与 §四-9 反号口径脱钩）"
    p_wrong = a.p_surf.reshape(rc, cc)
    assert p_wrong[i0, j0] > p_wrong[i0, (j0 + 3) % cc], \
        "错写没把暖斑变成**高压中心** ⇒ 反例装置立不起来，下面的符号翻转就无意义"
    for _ in range(3):
        a._integrate_wind()
    ue = a.u_east.reshape(rc, cc)
    un = a.u_north.reshape(rc, cc)
    assert ue[i0, (j0 + 1) % cc] > 1e-12, "错写下斑东侧的风应改吹**离开**斑心"
    assert ue[i0, (j0 - 1) % cc] < -1e-12
    assert un[i0 + 1, j0] > 1e-12
    assert un[i0 - 1, j0] < -1e-12


def test_w3_wind_integration_touches_only_the_wind():
    """`_integrate_wind()` 只写 `u_*` ⇒ "风不反过来加热/平流"在函数级被钉住（§二 因果链）。"""
    a = _atm()
    for k in range(1, 4):
        a.advance(10 * k)                       # 让 T/P 先长出非平凡结构
    before = (a.t_surf.copy(), a.p_surf.copy(), a.mass_col.copy(), a.steps, a.last_tick)
    fe, fn = a.wind_forces()
    assert (a.t_surf.tobytes() == before[0].tobytes()
            and a.p_surf.tobytes() == before[1].tobytes()
            and a.mass_col.tobytes() == before[2].tobytes()), "wind_forces 改写了状态"
    assert (a.steps, a.last_tick) == (before[3], before[4])
    a._integrate_wind()
    assert a.t_surf.tobytes() == before[0].tobytes(), "风写回温度场 = 因果颠倒（§二）"
    assert a.p_surf.tobytes() == before[1].tobytes(), "风写回气压 = 质量被风凭空搬动"
    assert (a.steps, a.last_tick) == (before[3], before[4])
    assert np.abs(a.u_north).max() > 0.0 or np.abs(a.u_east).max() > 0.0, "风必须真的动了"
    assert np.all(np.isfinite(fe)) and np.all(np.isfinite(fn))


# ---------------------------------------------------------------- τ 的解析律与零模型③

def test_w4_wind_relaxes_to_tau_force_on_the_exact_discrete_law():
    """🔴 v30 ⑤.4 必答的落地：恒定力下 `u_k = τ·F·(1−(1−dt/τ)^k)` **逐格**成立。

    这条把 τ 从"一个数"钉成"平衡风 `u_eq=τ·F` 的时间常数"：显式 Euler 的递推
    `u_{k+1} = (1−r)u_k + dt·F`（`r=dt/τ`）解析解就是上式 ⇒ 实测必须到机器精度。
    连续近似 `1−e^{−k·dt/τ}` 与离散解的差 ≈ `O(k·r²)` ⇒ 单独校一个 1% 门（口径：
    本批用离散解判**实现**，用连续解判**量级**，不混用）。
    """
    a = _atm(surface=_surface(kind="ocean"))
    rc, cc = a.rows_c, a.cols_c
    _ramp_pressure(a)
    _, fn = a.wind_forces()
    tau, dt = float(a.cfg.rayleigh_tau_s), a.dt_s
    r = dt / tau
    assert r < _DAMP_STABILITY_MAX
    for k in range(1, 21):
        a._integrate_wind()
        got = a.u_north.reshape(rc, cc)
        expect = tau * fn * (1.0 - (1.0 - r) ** k)
        assert np.allclose(got, expect, rtol=1e-12, atol=_ABS), \
            f"第 {k} 子步的 u 偏离 Rayleigh 递推解析解 ⇒ τ 不是那个意思"
    cont = tau * fn * (1.0 - np.exp(-20.0 * r))
    assert np.allclose(a.u_north.reshape(rc, cc), cont, rtol=1e-2, atol=1e-3), \
        "离散解与连续松弛解量级不符 ⇒ dt/τ 已大到不能当天时间常数用"
    assert a.u_east.max() == 0.0, "纯经向气压斜坡不该产生纬向风（漏了方向耦合）"


def test_w5_zero_model_three_no_damping_diverges_linearly():
    """零模型③（v26 §三"全部要跑"）：关摩擦 ⇒ `u = k·dt·F` **无界线性**；开摩擦 ⇒ 渐近 ≤τ·|F|。

    ⚠️ 时程必须跨过 τ：`dt/τ = 1.7e-3` ⇒ 10 子步内两档只差 0.8%（用 10 子步比"谁大"= 什么
    都没比）。判据分两段——短段钉**格式**（严格按 `k·dt·F`，逐位），长段钉**量级**（关阻尼
    跑到 5.2 倍平衡风还在涨，开阻尼贴住 `τ·|F|` 不动）。
    """
    b = _atm(surface=_surface(kind="ocean"), damping_on=False)
    d = _atm(surface=_surface(kind="ocean"))
    rc, cc = b.rows_c, b.cols_c
    _ramp_pressure(b)
    _ramp_pressure(d)
    _, fb = b.wind_forces()
    _, fd = d.wind_forces()
    assert np.array_equal(fb, fd), "两档的力必须逐位相同（只差阻尼项本身）"
    assert float(np.abs(fb).max()) > 0.0, "力全零 ⇒ 发散判据是空的"
    tau, dt = float(b.cfg.rayleigh_tau_s), b.dt_s
    for k in range(1, 11):
        b._integrate_wind()
        assert np.allclose(b.u_north.reshape(rc, cc), k * dt * fb, rtol=1e-12, atol=_ABS), \
            "关阻尼时 u 必须严格按 k·dt·F 线性增长（多出任何衰减 = 阻尼没真关）"
        d._integrate_wind()
        assert np.all(np.abs(d.u_north.reshape(rc, cc))
                      <= tau * np.abs(fd) + _ABS), "开阻尼却越过平衡风 ⇒ τ 的界是假的"
    short = float(np.abs(b.u_north).max())          # 10 子步：两档几乎一样（格式段专用）
    n_more = 3750                                   # 合计 3760 子步 × 120 s = 451200 s ≈ 5.22 τ（v38 汇率）
    for _ in range(n_more):
        b._integrate_wind()
        d._integrate_wind()
    k_all = 10 + n_more
    assert np.allclose(b.u_north.reshape(rc, cc), k_all * dt * fb,
                       rtol=1e-9, atol=_ABS), \
        "跑了 3010 子步就不再是 k·dt·F ⇒ 关阻尼档里混进了衰减项"
    ub = float(np.abs(b.u_north).max())
    ud = float(np.abs(d.u_north).max())
    assert ub > 5.0 * tau * float(np.abs(fb).max()), "关阻尼 5.2τ 后仍没越过平衡风 ⇒ 无界性没证成"
    assert ud <= 1.000001 * tau * float(np.abs(fd).max()), \
        "开阻尼却越过 `τ·|F|` 上界 ⇒ Rayleigh 摩擦不是渐近有界的"
    assert ud > 0.99 * tau * float(np.abs(fd).max()), \
        "开阻尼 5.2τ 后还离平衡风很远 ⇒ 阻尼比公式弱（τ 的含义与实现不符）"
    assert ub / ud > 5.0, "长时程两档没拉开 ⇒ '阻尼是必需项'这条反证不成立"
    assert ub > 100.0 * short, "关阻尼档 3010 子步相对 10 子步没线性放大 ⇒ 中途被什么压住了"


# ---------------------------------------------------------------- 极地口径（§四-13/14）

def test_w6_zonal_force_is_capped_by_the_polar_filter_not_by_accident():
    """§四-14：纬向力分母必须是 `max(c,0.1)`——极行被**截到 10 倍封顶**，c≥0.1 的行**不受裁**。

    判据写成"从力反解出实际用掉的分母"⇒ 两头都被钉住：不裁 ⇒ 极圈放大 10–54 倍；
    乱裁（全局降 c 或改 dt）⇒ 中纬物理被改（v26 §四-14 明示"先极地滤波、不降全局步"）。
    ⚠️ 装置必须用 **s2 几何**（120×240 ⇒ 极带质心 c ≈ 0.018 < 0.1）：微型 15×30 的粗网格
    行距 12°，最北质心 c ≈ 0.149 根本够不到滤波线 ⇒ 拿它测极地滤波 = 测了个空集。
    """
    a = _atm(120, 240, coarse_factor=1)
    rc, cc = a.rows_c, a.cols_c
    c_raw = a._c_ctr
    assert float(c_raw.min()) < _POLAR_C_FLOOR, \
        "装置若够不到滤波线，本测试就失去对象 ⇒ 换网格（不许改成恒真或调低门限凑过）"
    lam = 2.0 * np.pi * np.arange(cc) / cc
    p2 = _P_REF_PAS * (1.0 + 1e-3 * np.cos(lam)[None, :]) * np.ones((rc, 1))
    t_k = np.full((rc, cc), 14.0 + 273.15)
    a.p_surf[:] = p2.ravel()
    a.t_surf[:] = (t_k - 273.15).ravel()
    fe, _ = a.wind_forces()
    fe = fe.reshape(rc, cc)
    g_e = np.stack([(p2[:, (j + 1) % cc] - p2[:, (j - 1) % cc]) / (2.0 * a._dlam_c)
                    for j in range(cc)], axis=1)
    inv_rho = _R_DRY_AIR * t_k / p2
    # |F_e| = inv_rho·|∂_λP|/(a·c_eff) ⇒ 逐列反解 c_eff。只看**内圈**：极行的力已折成环带
    # 均值（§四-13），逐列反解对它无意义；∂_λP=0 的列（波峰/波谷）反解也无定义 ⇒ 一并剔除。
    with np.errstate(divide="ignore", invalid="ignore"):
        c_used = np.abs(inv_rho * g_e / (_EARTH_RADIUS_M * fe))
    hit = np.zeros((rc, cc), dtype=bool)
    hit[1:-1] = np.abs(g_e[1:-1]) > 0.0
    eff = np.maximum(c_raw, _POLAR_C_FLOOR)
    expect = np.broadcast_to(eff[:, None], (rc, cc))
    assert int(hit.sum()) > 0, "装置退化成'内圈一列梯度都没有' ⇒ 本测试失去对象"
    assert np.allclose(c_used[hit], expect[hit], rtol=1e-12, atol=0.0), \
        "纬向力的分母不是 max(c,0.1)（= 极地滤波没按 §四-14 的口径写）"
    capped = c_raw < _POLAR_C_FLOOR
    assert int(np.sum(capped[1:-1])) >= 2, "滤波线在**内圈**一行都没生效 ⇒ 反例装置是空的"
    assert np.all(eff[capped] == _POLAR_C_FLOOR), "该被截的行没截到底"
    assert float(np.max(1.0 / c_raw[capped])) > 1.0 / _POLAR_C_FLOOR + 1e-12, \
        "滤波线以下没有 >10 倍的原始放大 ⇒ 本测试对象是空的"
    assert float(np.max(1.0 / eff)) <= 1.0 / _POLAR_C_FLOOR + 1e-15, \
        "滤波后仍存在超过 10 倍的纬向放大"
    mid = int(np.argmax(c_raw))
    assert not capped[mid] and np.isclose(eff[mid], c_raw[mid], rtol=1e-15), \
        "c≥0.1 的中纬行被误裁 = 改了中纬物理"


def test_w7_meridional_force_carries_cos_latitude_and_vanishes_at_the_pole():
    """经向力 ∝ `cos φ`（与 W6 的 `1/cos φ` **相反**）⇒ 极点归零、"北极点没有北"不炸。"""
    a = _atm(surface=_surface(kind="ocean"))
    rc, cc = a.rows_c, a.cols_c
    p2 = _P_REF_PAS * (1.0 + 0.02 * a.mu_ctr)[:, None] * np.ones((1, cc))
    t_k = np.full((rc, cc), 14.0 + 273.15)
    a.p_surf[:] = p2.ravel()
    a.t_surf[:] = (t_k - 273.15).ravel()
    _, fn = a.wind_forces()
    g_n = np.zeros((rc, cc))
    for i in range(rc):
        ip, im = min(i + 1, rc - 1), max(i - 1, 0)
        g_n[i] = (p2[ip] - p2[im]) / (a.mu_ctr[ip] - a.mu_ctr[im])
    inv_rho = _R_DRY_AIR * t_k / p2
    expect = -(a._c_ctr[:, None] * inv_rho / _EARTH_RADIUS_M) * g_n
    got = fn.reshape(rc, cc)
    assert np.allclose(got, expect, rtol=1e-12, atol=_ABS), \
        "经向力不是 −(c/(ρa))∂_μP（漏乘 c = 极点会给出有限风 = §四-13 重复计数的入口）"
    i0, i1 = _polar_rows(a)
    assert abs(float(got[i1, 0])) < 0.2 * float(np.abs(got).max()), \
        "极点行的经向力没有显著衰减 ⇒ c 因子没起作用"
    assert float(np.abs(got).max()) > 0.0, "全零 = 力根本没算（判据不能靠 0 通过）"


def test_w8_longitude_seam_is_not_special():
    """§四-12：把气压场整体**平移一列** ⇒ 力和风都整体平移一列（内圈**逐位**）。

    极行不参与平移判定（它的力是环带均值 = 一圈一格的**有意**特殊处理，§四-13）；
    风从**静止**起积分子步 ⇒ `u` 是力的纯函数，逐位平移才说明接缝没被偷偷补过。
    """
    a = _atm()
    rc, cc = a.rows_c, a.cols_c
    for k in range(1, 5):
        a.advance(10 * k)                       # 用真实演化场（含 α/C 纹理）当输入
    p0 = a.p_surf.reshape(rc, cc).copy()
    t0 = a.t_surf.reshape(rc, cc).copy()
    assert float(np.abs(p0 - p0.mean()).max()) > 1.0, "气压场无水平结构 ⇒ 平移判定是空的"

    def response(dp, dt_):
        a.p_surf[:] = dp.ravel()
        a.t_surf[:] = dt_.ravel()               # 🔴 平移要**两场一起**：只挪 P 不挪 T 改的是
        a.u_east[:] = 0.0
        a.u_north[:] = 0.0                      # 物理输入，不是坐标系（1/ρ 会真的变）
        fe, fn = a.wind_forces()
        for _ in range(3):
            a._integrate_wind()
        return (fe, fn, a.u_east.reshape(rc, cc).copy(), a.u_north.reshape(rc, cc).copy())

    fe0, fn0, ue0, un0 = response(p0, t0)
    fe1, fn1, ue1, un1 = response(np.roll(p0, 1, axis=1), np.roll(t0, 1, axis=1))
    for got, base, name in ((fe1, fe0, "F_east"), (fn1, fn0, "F_north"),
                            (ue1, ue0, "u_east"), (un1, un0, "u_north")):
        assert np.array_equal(got[1:-1], np.roll(base[1:-1], 1, axis=1)), \
            f"内圈 `{name}` 不随经度平移 = 接缝被特殊处理（§四-12）"
    # 极行 = 环带均值：同一批数换求和顺序 ⇒ 只按 ULP 比（rtol 1e-13），不比逐位
    assert np.allclose(fe1[[0, -1]], np.roll(fe0[[0, -1]], 1, axis=1), rtol=1e-13, atol=0), \
        "极行的环带均值随平移变了 = 折叠口径依赖经度起点（一圈一格被破）"
    assert np.allclose(un1[[0, -1]], np.roll(un0[[0, -1]], 1, axis=1), rtol=1e-13, atol=0)


def test_w8b_polar_row_is_single_valued_after_advance():
    """§四-13 推进版：极行两条分量在多次推进后仍**经度恒单值**（一圈一格）。"""
    a = _atm()
    rc, cc = a.rows_c, a.cols_c
    for k in range(1, 8):
        a.advance(10 * k)
    ue = a.u_east.reshape(rc, cc)
    un = a.u_north.reshape(rc, cc)
    for arr, name in ((ue, "u_east"), (un, "u_north")):
        for i in _polar_rows(a):
            assert float(arr[i].max() - arr[i].min()) == 0.0, \
                f"极点行被赋予 {cc} 个不同的 {name} = 同一个极点格重复计数"
    assert a.probe()["u_polar_spread"] == 0.0


# ---------------------------------------------------------------- 快照 / 关档 / 异常

def test_w9_wind_is_state_not_diagnosis(tmp_path: Path):
    """🔴 u 是**时间积分的状态量** ⇒ 必须进档；往返逐位；旧档缺键 ⇒ 静止；形状不符 ⇒ 炸。"""
    a = _atm()
    for k in range(1, 6):
        a.advance(10 * k)
    st = a.state()
    assert set(st) == {"t_surf", "u_east", "u_north", "steps", "last_tick"}
    assert np.abs(st["u_north"]).max() > 0.0, "推进 5 次后风仍为 0 ⇒ 风没进状态"
    b = _atm()
    b.restore(st)
    assert b.u_east.tobytes() == a.u_east.tobytes()
    assert b.u_north.tobytes() == a.u_north.tobytes()
    assert b.p_surf.tobytes() == a.p_surf.tobytes()
    # 不进档的诊断量必须被 restore 重算（Step 3a 口径，这里再钉一次）
    assert b.mass_col.tobytes() == a.mass_col.tobytes()
    old = {k: v for k, v in st.items() if k not in ("u_east", "u_north")}
    c = _atm()
    c.restore(old)                                  # Step 3a 及更早的旧档
    assert float(np.abs(c.u_east).max()) == 0.0 and float(np.abs(c.u_north).max()) == 0.0, \
        "旧档缺 `atm_u_*` 必须从**静止**起算（不许静默猜测历史风场）"
    assert c.t_surf.tobytes() == a.t_surf.tobytes()
    bad = dict(st)
    bad["u_north"] = np.zeros(a.n_c + 1, dtype=np.float64)
    with pytest.raises(ValueError, match="u_north"):
        _atm().restore(bad)


def test_w9b_engine_resume_equals_continuous_run(tmp_path: Path):
    """引擎级：开档快照续跑 == 连续跑，**风、气压、温度三路都逐位一致**（§四-18）。"""
    cfg = _cfg()
    e1 = SphereEngine(cfg)
    for _ in range(30):
        e1.step()
    path = tmp_path / "atm_w.npz"
    e1.save_snapshot(str(path))
    r = SphereEngine.load_snapshot(str(path))
    for key in ("u_east", "u_north", "t_surf", "p_surf"):
        assert getattr(r.atmosphere, key).tobytes() == getattr(e1.atmosphere, key).tobytes(), key
    for _ in range(20):
        e1.step()
        r.step()
    for key in ("u_east", "u_north", "t_surf", "p_surf", "mass_col"):
        assert getattr(r.atmosphere, key).tobytes() == getattr(e1.atmosphere, key).tobytes(), key
    assert (r.atmosphere.steps, r.atmosphere.last_tick) == \
        (e1.atmosphere.steps, e1.atmosphere.last_tick)
    with np.load(str(path), allow_pickle=False) as data:
        assert sorted(k for k in data.files if k.startswith("atm_")) == \
            ["atm_last_tick", "atm_steps", "atm_t_surf", "atm_u_east", "atm_u_north"]


def test_w10_off_mode_is_bit_identical_and_has_no_keys(tmp_path: Path):
    """🔴 C7 回滚点：开档（有风）60 tick 与关档**逐位等价** + 开档零 RNG 消费。"""
    on_cfg, off_cfg = _cfg(), _cfg()
    off_cfg.atmosphere.enabled = False
    on, off = SphereEngine(on_cfg), SphereEngine(off_cfg)
    for _ in range(60):
        on.step()
        off.step()
        assert on.rng.draws == off.rng.draws, "Step 3b 引入了新 RNG 消费（§四-19）"
    assert _engine_state(on) == _engine_state(off), "开档改了主 tick 面 ⇒ C7 回滚点破"
    assert on.atmosphere is not None and on.atmosphere.steps == 18       # 6 推进 × 3 子步
    wind = float(np.abs(on.atmosphere.u_east).max() + np.abs(on.atmosphere.u_north).max())
    assert wind > 0.0, "开档必须真的算了风（wind=0 的'等价'是没跑起来的假通过）"
    path = tmp_path / "off_w.npz"
    on.save_snapshot(path)
    with np.load(str(path), allow_pickle=False) as data:
        assert all(k in data.files for k in ("atm_u_east", "atm_u_north"))
    off.save_snapshot(tmp_path / "off2.npz")
    with np.load(str(tmp_path / "off2.npz"), allow_pickle=False) as data:
        assert not [k for k in data.files if k.startswith("atm_")]
    assert SphereEngine.load_snapshot(str(tmp_path / "off2.npz")).atmosphere is None


def test_w10b_atmosphere_never_writes_main_grid_fields():
    """开档 60 tick ⇒ `env_field` 与资源面**一个字节都不动**（大气只在自己的粗网格演化）。"""
    on_cfg, off_cfg = _cfg(), _cfg()
    off_cfg.atmosphere.enabled = False
    on, off = SphereEngine(on_cfg), SphereEngine(off_cfg)
    layers = ("temp", "w", "h", "surface_type", "ocean_mask", "land_mask", "river_mask")
    for _ in range(60):
        on.step()
        off.step()
    for name in layers:
        va, vb = getattr(on.env_field, name), getattr(off.env_field, name)
        assert np.asarray(va).tobytes() == np.asarray(vb).tobytes(), name


def test_w11_nonfinite_and_damping_instability_are_fail_loud():
    a = _atm(surface=_surface(kind="ocean"))
    _ramp_pressure(a)
    a.u_north[:] = np.nan
    with pytest.raises(FloatingPointError, match="风场"):
        a._integrate_wind()
    a.u_north[:] = 0.0
    a.u_east[:] = np.inf
    with pytest.raises(FloatingPointError, match="风场"):
        a._integrate_wind()
    a.u_east[:] = 0.0
    p = a.p_surf.copy()
    p[3] = np.nan
    a.p_surf[:] = p
    with pytest.raises(FloatingPointError, match="气压梯度力"):
        a.wind_forces()
    a.p_surf[:] = p.copy()
    a.p_surf[3] = np.inf
    with pytest.raises(FloatingPointError):
        a.wind_forces()
    # dt/τ 进门限外 ⇒ 构造期炸（μ 扩散门此时仍在门内）
    tau = float(AtmosphereConfig().rayleigh_tau_s)
    dt_bad = tau * (_DAMP_STABILITY_MAX + 0.08)
    with pytest.raises(ValueError, match="阻尼"):
        _atm(dt_s=dt_bad)
    _atm(dt_s=dt_bad, damping_on=False)          # 关阻尼 ⇒ 没有衰减项 ⇒ 不受该门约束


def test_w12_config_domain_and_old_save_fallback():
    """参数域守卫 + 旧档回退：两个新字段非法 ⇒ AssertionError；缺字段 ⇒ 回退默认。"""
    default = AtmosphereConfig()
    for kw in ({"rayleigh_tau_s": 0.0}, {"rayleigh_tau_s": -1.0},
               {"rayleigh_tau_s": float("nan")}, {"rayleigh_tau_s": float("inf")},
               {"damping_on": 1}, {"damping_on": None}):
        with pytest.raises(AssertionError):
            AtmosphereConfig(enabled=True, **kw)
    d = SimConfig().to_dict()
    d.pop("atmosphere")
    back = SimConfig.from_dict(d)
    assert back.atmosphere.rayleigh_tau_s == default.rayleigh_tau_s
    assert back.atmosphere.damping_on is default.damping_on
    d = SimConfig().to_dict()
    d["atmosphere"].pop("rayleigh_tau_s")
    d["atmosphere"].pop("damping_on")
    back = SimConfig.from_dict(d)
    assert back.atmosphere.rayleigh_tau_s == default.rayleigh_tau_s
    cfg = SimConfig()
    cfg.atmosphere = AtmosphereConfig(enabled=True, rayleigh_tau_s=43200.0, damping_on=False)
    rt = SimConfig.from_dict(cfg.to_dict())
    assert rt.atmosphere.rayleigh_tau_s == 43200.0
    assert rt.atmosphere.damping_on is False


def test_w13_probe_is_read_only_and_wind_keys_are_finite():
    a = _atm()
    for k in range(1, 4):
        a.advance(10 * k)
    before = (a.t_surf.copy(), a.p_surf.copy(), a.mass_col.copy(),
              a.u_east.copy(), a.u_north.copy(), a.steps, a.last_tick)
    pr = a.probe()
    assert a.t_surf.tobytes() == before[0].tobytes()
    assert a.u_east.tobytes() == before[3].tobytes()
    assert a.u_north.tobytes() == before[4].tobytes()
    assert (a.steps, a.last_tick) == (before[5], before[6])
    for key in ("rayleigh_tau_s", "dt_over_tau", "advance_s", "advance_over_tau",
                "force_max_ms2", "force_awm_ms2", "u_eq_max_ms", "u_rms_awm_ms",
                "ke_awm_m2_s2", "u_max_ms", "u_polar_spread", "polar_rows_filtered"):
        assert key in pr, f"probe 缺 Step 3b 读数 `{key}`"
        assert np.isfinite(pr[key]), f"`{key}` 非有限：{pr[key]}"
    assert pr["dt_over_tau"] == pytest.approx(a.dt_s / 86400.0, rel=1e-15)
    assert pr["advance_over_tau"] == pytest.approx(a.dt_s * a.substeps / 86400.0, rel=1e-15)
    # 力恒零的装置 = **均匀 T**（零模型①的不动点：`insolation_mode="uniform"` 只把入射摊平，
    # 初值仍是纬度剖面 ⇒ 刚构造完力并不为零）。此时没有平衡风可比 ⇒ 读数必须是 `nan`，
    # 不是 0/0 崩、也不是"0 除 0 = 达标"的假通过。
    u = _atm(surface=_surface(kind="ocean"), insolation_mode="uniform")
    u.t_surf[:] = 14.0
    u._diagnose_pressure()
    pr0 = u.probe()
    assert pr0["force_max_ms2"] == 0.0 and pr0["force_awm_ms2"] == 0.0, \
        "均匀 T 档的力不恒零 ⇒ 梯度算子或极行折叠里混进了结构"
    assert pr0["u_max_ms"] == 0.0 and pr0["ke_awm_m2_s2"] == 0.0
    assert np.isnan(pr0["u_eq_ratio_awm"]), \
        "无平衡风可比时给数 = 把'除零'伪装成'判据通过'"


# ---------------------------------------------------------------- 结构与能量符号

def test_w14_low_level_wind_blows_equatorward_in_both_hemispheres():
    """年平剖面 ⇒ 赤道低压、极区高压 ⇒ 低层风**两支都朝赤道**（单胞的辐合侧）。"""
    a = _atm(surface=_surface_symmetric())
    rc, cc = a.rows_c, a.cols_c
    for k in range(1, 61):
        a.advance(10 * k)
    un = a.u_north.reshape(rc, cc).mean(axis=1)
    i_eq = int(np.argmin(np.abs(a.mu_ctr)))
    south = un[:i_eq]
    north = un[i_eq + 1:]
    far_s = np.abs(a.mu_ctr[:i_eq]) > 0.05
    far_n = np.abs(a.mu_ctr[i_eq + 1:]) > 0.05
    assert np.all(south[far_s] > 0.0), "南半球的低层风必须向北（= 朝赤道）"
    assert np.all(north[far_n] < 0.0), "北半球的低层风必须向南（= 朝赤道）"
    assert abs(float(un[i_eq])) < max(float(np.abs(south).max()), 1e-30), \
        "赤道行的经向风应是整条剖面的极小（辐合中心）"
    # 南北镜像：`u_n` 反对称、`u_e` 对称（对称地表 ⇒ 对称输入必给对称输出）
    ue = a.u_east.reshape(rc, cc)
    scale = max(float(np.abs(un).max()), float(np.abs(ue).max()))
    assert np.allclose(un, -un[::-1], rtol=1e-9, atol=1e-9 * scale), \
        "经向风南北不对称 = 极区/赤道处理有一侧写歪（v26 §四-13 家族）"
    assert np.allclose(ue, ue[::-1], rtol=1e-9, atol=1e-9 * scale), \
        "纬向风南北镜像应同向（反射不翻转东向）"


def test_w15_wind_crosses_isobars_because_there_is_no_coriolis():
    """🔴 力平衡 ≠ 地转：本批**无科氏力** ⇒ `AWM(u·F) > 0`（风穿过等压线流向低压）。

    加了科氏力（地转）后 `u ⊥ F` ⇒ 该读数会趋于 0；今天它是明显正数 = "不是地转风"的证据。
    同时 `u` 的动能不能无限涨：开阻尼时 `KE ≤ ½·(τ·|F|)²`（W5 已核，这里核全球均值口径）。
    """
    a = _atm(surface=_surface_symmetric())
    for k in range(1, 31):
        a.advance(10 * k)
    fe, fn = a.wind_forces()
    pw = a.awm(a.u_east * fe.ravel() + a.u_north * fn.ravel())
    assert pw > 0.0, f"u·F = {pw}（地转风会 ≈0；本批无科氏力 ⇒ 必须为正）"
    pr = a.probe()
    tau = float(a.cfg.rayleigh_tau_s)
    # 逐格 `|u| ≤ τ|F|` ⇒ 动能 `≤ ½(τF)²`；均值口径同理（W5 已在逐格核过上界）
    assert pr["ke_awm_m2_s2"] <= 0.5 * (tau * pr["force_max_ms2"]) ** 2 + 1e-9, \
        "动能越过平衡风上界 ⇒ 阻尼项没生效"
    assert pr["u_eq_ratio_awm"] < 1.0, "开阻尼时 u/u_eq 必须 <1（渐近逼近 τ·F）"


# ---------------------------------------------------------------- 质量连续（§四-11）

def test_w17_area_weighted_divergence_vanishes_to_machine_precision():
    """🔴 §四-11：`Σ area·∇·u = 0` 必须是**望远镜相消**（机器精度），不是"≈0 的物理近似"。

    三种场都测：① 真实演化风；② 人工"只有经向环流"的风（纬向恒零）；③ 只有纬向的风
    （经向恒零，且必须**逐行**恒零）。③ 把两个方向的相消机制分开 ⇒ 少乘一个 `c`、
    面平均写成节点差、极外面没归零，都会在其中一支露出来。
    """
    a = _atm()
    rc, cc = a.rows_c, a.cols_c
    for k in range(1, 8):
        a.advance(10 * k)
    pr = a.probe()
    assert pr["div_area_flux_scale"] > 0.0, \
        "风散度全零 ⇒ 守恒判据是空的（先把风跑起来再谈 =0）"
    assert pr["div_area_sum_rel_residual"] < 1e-13, \
        f"probe 口径的真实风残差 {pr['div_area_sum_rel_residual']:.3e} 超机器精度档"

    def residual(u_e, u_n):
        a.u_east[:] = u_e.ravel()
        a.u_north[:] = u_n.ravel()
        dv = a.div_u()
        own = float(np.sum(a.area[:, None] * np.abs(dv)))       # 每支场**自己**的通量尺
        assert own > 0.0, "该装置的散度全零 ⇒ 守恒判据是空的"
        return abs(float(np.sum(a.area[:, None] * dv))) / own, dv

    rel, _ = residual(a.u_east.reshape(rc, cc), a.u_north.reshape(rc, cc))
    assert rel < 1e-13, f"真实演化风的 `Σarea·div` 相对残差 {rel:.3e} 超机器精度档"
    mer = np.zeros((rc, cc))
    mer[:] = np.sin(2.0 * np.pi * np.arange(rc)[:, None] / (3.0 * rc))    # 纯经向、随纬度变
    rel, dv = residual(mer * 0.0, mer)
    assert rel < 1e-13, f"纯经向风的经向项没望远镜相消（残差 {rel:.3e}）"
    zon = np.zeros((rc, cc))
    zon[:] = np.cos(3.0 * 2.0 * np.pi * np.arange(cc)[None, :] / cc)      # 纯纬向、3 波
    rel, dv = residual(zon, zon * 0.0)
    assert rel < 1e-13, f"纯纬向风的纬向项没逐行周期相消（残差 {rel:.3e}）"
    pr = _atm()
    for k in range(1, 8):
        pr.advance(10 * k)
    assert pr.probe()["div_area_sum_rel_residual"] < 1e-13
    assert pr.probe()["div_area_flux_scale"] > 0.0


def test_w18_divergence_sign_converges_into_the_warm_spot():
    """§四-7 的**散度侧**：暖斑地面低压 ⇒ 风辐合 ⇒ 斑心 `∇·u < 0`（3c 的 `w>0` 前置）。

    与 W1 同一装置（均匀背景 + 单暖斑），但从"四周吹向斑心"升级到"净面积通量为负"⇒
    控制体口径被真的数值钉住，而不只靠四个符号。
    """
    a = _atm(surface=_surface(kind="ocean"))
    rc, cc = a.rows_c, a.cols_c
    i0, j0 = _warm_spot(a)
    for _ in range(5):
        a._integrate_wind()
    dv = a.div_u()
    assert float(dv[i0, j0]) < -1e-12, f"暖斑中心必须是**辐合**（实得 {dv[i0, j0]:.3e}）"
    ring = [dv[i0, (j0 + 2) % cc], dv[i0, (j0 - 2) % cc], dv[i0 + 1, j0], dv[i0 - 1, j0]]
    assert all(float(v) < 0.0 for v in ring), "斑心四邻(半径2)也应辐合"
    # 守恒的推论：有辐合就必须有辐散，且辐散区**不在**斑心附近（否则符号场只有一个符号）
    dv2 = dv.reshape(rc, cc)
    i_far, j_far = np.unravel_index(int(np.argmax(dv2)), dv2.shape)
    assert float(dv2[i_far, j_far]) > 0.0, "整场没有辐散区 ⇒ 质量被凭空吞掉（§四-11 反证）"
    dlon = min(abs(j_far - j0), cc - abs(j_far - j0))
    assert abs(i_far - i0) + dlon >= 2, f"最大辐散落在斑心本身（{i_far},{j_far}）⇒ 符号结构反了"
    assert abs(float(np.sum(a.area[:, None] * dv))) <= 1e-13 * float(
        np.sum(a.area[:, None] * np.abs(dv))), "有符号的辐合/辐散结构下守恒仍然必须成立"


def test_w19_div_u_is_read_only_and_polar_outer_faces_carry_no_flux():
    """`div_u()` 只读（同 probe 口径）+ 极外面零通量（§四-13：球面没有"外面"）。"""
    a = _atm()
    for k in range(1, 5):
        a.advance(10 * k)
    before = (a.u_east.copy(), a.u_north.copy(), a.p_surf.copy(), a.t_surf.copy(),
              a.mass_col.copy(), a.steps, a.last_tick)
    dv = a.div_u()
    assert dv.shape == (a.rows_c, a.cols_c)
    for got, base in ((a.u_east, before[0]), (a.u_north, before[1]),
                      (a.p_surf, before[2]), (a.t_surf, before[3]),
                      (a.mass_col, before[4])):
        assert got.tobytes() == base.tobytes(), "div_u 写了状态 ⇒ 诊断量不该有副作用"
    assert (a.steps, a.last_tick) == (before[5], before[6])
    # 极外面零通量的直接证据：把**整个** u 场乘 2 ⇒ 每一行（含极行）的 div 必须恰好 2 倍
    a.u_east[:] = 2.0 * a.u_east
    a.u_north[:] = 2.0 * a.u_north
    dv2 = a.div_u()
    assert np.allclose(dv2, 2.0 * dv, rtol=1e-12, atol=0.0), \
        "散度对 u 不是一次齐次 ⇒ 极外面漏进了非通量项（或常数源被写死）"
    rc, cc = a.rows_c, a.cols_c
    assert float(np.abs(dv2[0]).max()) > 0.0, "极行散度全零 = 极帽被整行抹掉（§四-13 的另一半）"
    a.u_east[:] = 0.0
    a.u_north[:] = 0.0
    assert float(np.abs(a.div_u()).max()) == 0.0, "静止风必须给出**恰好**零散度（不许有常数源）"


# ---------------------------------------------------------------- 参考实现对拍

def test_w16_forces_match_a_naive_scalar_reference():
    """🔴 向量化的力 == 朴素三重循环参考实现（**独立写法**的对拍，同 Rust 双路径口径）。

    `np.roll` 的接缝、极行折叠、方向因子（乘 c vs 除 c）都是"向量化只错一列"的坑；
    拿手算下标的标量实现对拍 ⇒ 错了必然露。真输入用**演化后的场**（含 α/C 纹理），
    不用人工正弦 ⇒ 对拍覆盖的是批跑真正吃的那个数。
    """
    a = _atm()
    rc, cc = a.rows_c, a.cols_c
    for k in range(1, 5):
        a.advance(10 * k)
    p2 = a.p_surf.reshape(rc, cc).copy()
    t_k = a.t_kelvin().reshape(rc, cc).copy()
    assert float(np.abs(p2 - float(p2.mean())).max()) > 1.0, "输入无结构 ⇒ 对拍是空的"
    he, hn = _hand_forces(a, p2, t_k)
    fe, fn = a.wind_forces()
    assert np.allclose(fe, he, rtol=1e-12, atol=0.0), \
        "纬向力与朴素参考不符 ⇒ 向量化实现有列/行下标错位或漏因子"
    assert np.allclose(fn, hn, rtol=1e-12, atol=0.0), \
        "经向力与朴素参考不符 ⇒ 面→节点写法或极行折叠不一致"
    scale = float(np.abs(he).max()) + float(np.abs(hn).max())
    assert scale > 0.0, "参考实现自己也全零 = 两边一起没算"
    rel = float(np.abs(fe - he).max() + np.abs(fn - hn).max()) / scale
    assert rel < 1e-12, f"两写法相对偏差 {rel:.3e} 超机器精度档 ⇒ 不是舍入而是格式差"
