"""C4a-1 热机本体 Step 3c（辐合/辐散 → 垂直运动代理 + 零模型套件）—— 回归单测。

权威口径 = `内部汇报/话事人传达-20261016-C4a-1热机本体开工裁定.md`（v26）§二 因果链末环
「风的水平散度 ∇·u = 垂直运动代理（暖区辐合上升 / 冷区辐散下沉）」+ §三 验收判据
「环流结构：赤道近地面辐合带、副热带辐散带（单胞）稳定且关于赤道对称；暖区垂直速度 >0」
+ §四-8/11/12/13/15/16 + 派工项 3「零模型全套 ①②③④」。

覆盖（每条都是可观测行为 / 解析恒等式，不是"字段存在"）：
    W20 `w = −z_b·∇·u` 恒等式 + **唯一长度尺度**（`layer_depth_m` 只出现一次 ⇒ 翻倍即翻倍）
        + 纯诊断（不进快照 / restore 重算 / 只读）
    W21 🔴 §四-8 头号判据：单暖斑 ⇒ **暖区 w>0（上升）** 且 `w` 与 `T` 异常面积权**正相关**；
        W21b 鉴别力：§四-9 点名的错写 `P∝T` ⇒ 同一装置下 `w` 必然**下沉**、相关翻负
    W22 单胞结构（EBM 稳态 + 风平衡 τF 装置）：赤道带辐合、高纬辐散、**辐散支起始纬度**
        落在 [30°,46°] 带宽内，并与"由 T 剖面预测的通量峰"同位（35.3° 常密度玩具值只作**对照**）
    W23 「稳定」的可核验版本 = 形状自相似：力恒定（冻结 P）下 `div_k` 的**归一化形状**与
        子步数无关 ⇒ 环流位形不随时间漂移；南北**镜像对称**（`div(−μ)=div(μ)`、`u_n` 反对称）
    W24 §四-11 环流侧：辐合区与辐散区各自**非零**、符号相反，两者面积权积分相消到机器精度
        （只给"全球和=0"会被"处处为 0"骗过 ⇒ 两边都必须有内容）
    W25 §四-12/13 延伸：极外面零通量 ⇒ 极行 `div` 只来自内面；接缝不是特例（整场平移一列
        ⇒ `div`/`w` 同样平移）
    W26 🔴 零模型①：均匀入射 + 单一地表 + 均匀初温 ⇒ `T/P/u/div/w` **逐位恒零梯度**（风≈0
        的最强形式）；W26b 反面口径：**只**关入射梯度而保留剖面初值 ⇒ 风仍在（批长远短于
        辐射弛豫时间 ⇒ ① 的判据必须写清初值条件，否则是一条恒假断言）
    W27 零模型②：`D=0` 稳态 ⇒ 极差更大（E8 已钉 T 侧）⇒ **本批钉环流侧**：风更强、
        上升支更强、符号结构不变，且辐散支起始纬度**向极地移动**（剖面变尖的直接后果）
    W28 零模型③：关阻尼 ⇒ `|w|` 随子步线性无界，但 `Σ area·∇·u ≡ 0` **不依赖阻尼**而成立
        （守恒是几何恒等式，不是耗散的结果 ⇒ 这条在发散档下仍绿才是真守恒）
    W29 零模型④ + 键集合：关档不构造 ⇒ 无 `w/div` 读数；开档 3c 七个键**逐名**在场（新增
        读数不改物理：把带界常数改掉 ⇒ `t/p/u/div` 逐位不动，只有带读数动）
    W30 `awm_band` 与 `awm` **同一把尺**：全带恒等式（到舍入）+ 分带可加性 + 空带 fail-loud
    W31 §四-15：`u` 含 NaN/Inf ⇒ `div_u()` / `vertical_velocity()` 直接抛 `FloatingPointError`

⚠️ 时间口径（v30 ⑤.4 的延续，写判据时必须知道）：批长 2000 tick = 90 000 s = 1.04 τ ⇒ 风只到
平衡值 ≈65%，而 EBM 辐射弛豫时间 `C/B ≈ 2.1×10⁷ s` ⇒ 批内温度仍停在**初值剖面**（动了 0.4%）。
故本文件把「环流位形」判据放在 **EBM 稳态 + 风平衡** 装置上（`_steady_t`/`_relax_wind`，
同 Step 2 的 E8/E9 先例），批内瞬态读数另由 s2 验收批给出并在汇报里对照（两者辐散支起始
纬度不同：稳态 ≈38°、批内 ≈49°，差值由剖面形状解释，见汇报 §五）。
"""
from __future__ import annotations

import numpy as np
import pytest

from simulation.config import AtmosphereConfig
from world import env_field as ef
from world import atmosphere as atmo
from world.atmosphere import (Atmosphere, _BAND_EQ_MAX_DEG, _BAND_SUB_MAX_DEG,
                              _BAND_SUB_MIN_DEG, _EBM_A_WM2, _EBM_B_WM2K, _EARTH_RADIUS_M,
                              _P_REF_PAS)
from world.sphere_world import SphereWorld

_REL = 1e-12
_ABS = 1e-12
# 3c 新增读数（W29 逐名校核；改键必须同步这里，防"悄悄加口径"）
_NEW_KEYS = ("w_top_absmax_ms", "w_top_awm_ms", "w_eq_band_ms", "w_sub_band_ms",
             "div_eq_band_per_m", "div_sub_band_per_m", "w_t_anom_corr")


# ------------------------------------------------------------------ 装置

def _surface(rows: int, cols: int, kind: str = "ocean") -> np.ndarray:
    """确定性地表层（**零 RNG**）；`kind="ocean"` ⇒ 驱动场只随纬度变（干净装置）。"""
    if kind == "ocean":
        return np.full(rows * cols, ef.SURFACE_OCEAN, dtype=np.uint8)
    idx = np.arange(rows * cols, dtype=np.int64).reshape(rows, cols)
    st = np.where(idx < rows * cols // 2, ef.SURFACE_OCEAN, ef.SURFACE_LAND)
    st[:, ::3] = ef.SURFACE_RIVER
    return st.ravel().astype(np.uint8)


def _surface_symmetric(rows: int, cols: int) -> np.ndarray:
    """南北**镜像对称**地表（对称输入必须给对称输出，同 Step 2 E8 / 3b W14 装置）。"""
    idx = np.arange(rows * cols, dtype=np.int64).reshape(rows, cols)
    st = np.where(idx < rows * cols // 2, ef.SURFACE_OCEAN, ef.SURFACE_LAND)
    st[:, ::3] = ef.SURFACE_RIVER
    st[rows // 2:, :] = st[:rows // 2, :][::-1]
    return st.ravel().astype(np.uint8)


def _atm(rows: int = 120, cols: int = 240, factor: int = 2, surface: str = "mixed",
         **akw) -> Atmosphere:
    surf = akw.pop("surface_arr", None)
    if surf is None:
        surf = (_surface_symmetric(rows, cols) if surface == "sym"
                else _surface(rows, cols, surface))
    return Atmosphere(SphereWorld(rows, cols), AtmosphereConfig(enabled=True,
                                                                coarse_factor=factor, **akw),
                      surf)


def _steady_t(a: Atmosphere) -> np.ndarray:
    """EBM **稳态解**（℃，(rows_c, cols_c)）：解 `B·T − D∇²T = S(1−α_eff) − A`。

    与 `_ebm_substep` 共用同一套 `tau/area` ⇒ 核的是同一个离散算子（Step 2 E8/E9 先例）；
    `diffusion_on=False` 档下矩阵退化为 `B·I` ⇒ 极差直接由入射梯度决定（零模型②装置）。
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


def _relax_wind(a: Atmosphere, k: int = 400) -> Atmosphere:
    """把风推到**平衡值** `u_eq = τ·F`（T/P 冻结 ⇒ 力恒定，W4 已证的离散精确解）。

    ⚠️ 这是"读环流位形"的装置，不是批内状态：批内只到 ≈65% 平衡值（v30 ⑤.4）。位形的**符号
    结构**与幅度无关（所有格同步松弛 ⇒ 见 W23），所以稳态位形与批内位形可比、不可互替。
    """
    for _ in range(int(k)):
        a._integrate_wind()
    return a


def _steady_cell(rows: int = 480, cols: int = 960, factor: int = 4, **akw) -> Atmosphere:
    """稳态单胞装置：地表南北对称 + T 取 EBM 不动点 + 风推到 τF 平衡。"""
    a = _atm(rows, cols, factor, surface="sym", **akw)
    a.t_surf[:] = _steady_t(a).ravel()
    a._diagnose_pressure()
    return _relax_wind(a)


def _div_crossing_lat(a: Atmosphere) -> float:
    """纬向平均 `∇·u` 由**负（辐合）变正（辐散）**的首个过零纬度（°，取半球内）。"""
    rc, cc = a.rows_c, a.cols_c
    z = a.zonal_mean(a.div_u())
    lat = np.degrees(np.arcsin(a.mu_ctr))
    half = rc // 2
    for i in range(half, rc - 1):
        if z[i] < 0.0 <= z[i + 1]:
            return float(abs(lat[i + 1]))
    return float("nan")


def _flux_peak_from_t(a: Atmosphere) -> float:
    """由 **T 剖面**独立预测经向通量峰纬度（= `∇·u` 过零处）：`c·u_n ∝ c²·∂_μT/T`。

    只用连续律（`u_n = τF_n` 的解析形式 + 理想气体），不读 `u` ⇒ 与实现独立。
    搜索限 `|μ| ≤ 0.95`：极点行是单侧差分，`∂_μT` 在那里被"一圈一格"的几何主导，
    拿来定峰位是装置错（实测会把峰推到 88°）。
    """
    mu = a.mu_ctr
    tk = np.asarray(a.t_kelvin(), dtype=np.float64).reshape(a.rows_c, a.cols_c).mean(axis=1)
    g = np.gradient(tk, mu)
    h = np.abs((1.0 - mu**2) * g / tk)
    keep = np.abs(mu) <= 0.95
    idx = np.arange(a.rows_c)[keep][int(np.argmax(h[keep]))]
    return float(abs(np.degrees(np.arcsin(mu[idx]))))


def _toy_peak_constant_density(a: Atmosphere) -> float:
    """常密度玩具极限（`1/T` 权重取掉）的峰位，用来把"实现值 vs 35.3°"的差**归因**。

    P₂ 拟合 `T(μ)` 后取 `h₀=(1−μ²)|3T₂μ|` 的峰 ⇒ 纯几何极限；与实测的差 = 剖面形状 +
    状态方程权重的贡献（不是"实现偏了"的证据）。
    """
    mu = a.mu_ctr
    tk = np.asarray(a.t_kelvin(), dtype=np.float64).reshape(a.rows_c, a.cols_c).mean(axis=1)
    basis = np.stack([np.ones_like(mu), (3.0 * mu**2 - 1.0) / 2.0], axis=1)
    coef = np.linalg.lstsq(basis, tk, rcond=None)[0]
    t2 = float(coef[1])
    mm = np.linspace(0.01, 0.99, 4000)
    h0 = (1.0 - mm**2) * np.abs(3.0 * t2 * mm)
    return float(np.degrees(np.arcsin(mm[int(np.argmax(h0))])))


def _warm_spot(a: Atmosphere, amp: float = 8.0, bg: float = 14.0):
    """均匀背景 + 赤道带高斯暖斑（同 3b W1/W2 装置；纬向用周期距离 ⇒ 接缝连续）。"""
    rc, cc = a.rows_c, a.cols_c
    i0, j0 = rc // 2, cc // 4
    di = (np.arange(rc)[:, None] - i0) ** 2
    dj = np.minimum(np.abs(np.arange(cc) - j0), cc - np.abs(np.arange(cc) - j0)) ** 2
    a.t_surf[:] = (np.full((rc, cc), bg) + amp * np.exp(-di / 8.0 - dj / 8.0)).ravel()
    a._diagnose_pressure()
    return i0, j0


# ------------------------------------------------- w 的定义、唯一尺度与只读性

def test_w20_vertical_proxy_is_minus_layer_depth_times_divergence():
    """`w ≡ −z_b·∇·u` **逐位**成立，且 `layer_depth_m` 是**唯一**长度尺度（翻倍即翻倍）。"""
    a = _atm(240, 480, 4)
    for k in range(1, 6):
        a.advance(10 * k)
    dv = a.div_u()
    wx = a.vertical_velocity()
    assert np.array_equal(wx, -float(a.cfg.layer_depth_m) * dv), "w 不是 −z_b·∇·u"
    # 唯一尺度：同 T/P/u，只把 z_b 翻倍 ⇒ w 逐位翻倍（不存在第二个隐藏的长度/缩放因子）
    b = _atm(240, 480, 4, layer_depth_m=2.0 * float(a.cfg.layer_depth_m))
    b.t_surf[:] = a.t_surf
    b._diagnose_pressure()
    b.u_east[:] = a.u_east
    b.u_north[:] = a.u_north
    assert np.array_equal(b.vertical_velocity(), 2.0 * wx), "w 对 z_b 不是线性 ⇒ 混进了第二个尺度"
    # 🔴 纯诊断：不进快照（键集合仍 5），restore 后由 t_surf+u 重算 ⇒ 逐位一致
    assert set(a.state()) == {"t_surf", "u_east", "u_north", "steps", "last_tick"}
    st = a.state()
    c = _atm(240, 480, 4)
    c.restore(st)
    assert np.array_equal(c.vertical_velocity(), wx), "w 是诊断量却重建不出来 = 有隐藏历史依赖"


def test_w20b_vertical_velocity_is_read_only():
    """`div_u()`/`vertical_velocity()` 前后所有状态逐位不动（诊断算子不许写回）。"""
    a = _atm(120, 240, 2)
    for k in range(1, 5):
        a.advance(10 * k)
    snap = (a.t_surf.tobytes(), a.p_surf.tobytes(), a.mass_col.tobytes(),
            a.u_east.tobytes(), a.u_north.tobytes(), a.steps, a.last_tick)
    a.vertical_velocity()
    a.div_u()
    a.probe()
    assert (a.t_surf.tobytes(), a.p_surf.tobytes(), a.mass_col.tobytes(),
            a.u_east.tobytes(), a.u_north.tobytes(), a.steps, a.last_tick) == snap


# ------------------------------------------------------ §四-8 暖升冷沉（头号判据）

def test_w21_warm_spot_gives_rising_motion_and_positive_correlation():
    """🔴 v26 §三/§四-8：单暖斑 ⇒ 暖区 `w>0`（上升）且 `w` 与 `T` 异常面积权**正相关**。"""
    a = _atm(240, 480, 4)
    rc, cc = a.rows_c, a.cols_c
    i0, j0 = _warm_spot(a)
    _relax_wind(a, 60)
    wx = a.vertical_velocity().reshape(rc, cc)
    assert wx[i0, j0] > 0.0, "斑心（气压极小 ⇒ 四周净流入）必须上升"
    ring = np.array([wx[i0, (j0 + d) % cc] for d in (-2, -1, 1, 2)]
                    + [wx[i0 + d, j0] for d in (-2, -1, 1, 2)])
    assert float(ring.min()) > 0.0, f"斑的四周必须全部上升（实测最小 {ring.min():.3e}）"
    anom = a.t_surf - a.awm(a.t_surf)
    corr = a.aw_corr(a.vertical_velocity(), anom)
    assert corr > 0.3, f"w 与 T 异常正相关判据失败（corr={corr:.3f}）—— 算子或符号写反"
    # 全球质量连续（§四-11）与上升支同时成立：上升的总量 = 下沉的总量
    assert abs(a.probe()["div_area_sum_rel_residual"]) < 1e-12


def test_w21b_wrong_pressure_law_flips_vertical_motion():
    """W21 的**鉴别力**：错写 `P=P₀·T/T̄`（§四-9 点名禁止）⇒ 同一装置下暖区必然**下沉**。

    这条不是"再跑一遍"：它把 W21 从恒真断言里救出来 —— 若 `div`/`w` 的符号或 `∇·u` 的
    极地口径写错，正相关可能碰巧成立；换成同向气压律后必须翻负，否则判据没有内容。
    """
    a = _atm(240, 480, 4)
    rc, cc = a.rows_c, a.cols_c
    i0, j0 = _warm_spot(a)
    tk = a.t_kelvin()
    a.p_surf[:] = (_P_REF_PAS * tk / a.awm(tk)).ravel()     # 🔴 暖区改成高压
    _relax_wind(a, 60)
    wx = a.vertical_velocity().reshape(rc, cc)
    assert wx[i0, j0] < 0.0, "错写气压律下斑心仍上升 ⇒ W21 是恒真断言（判据失效）"
    anom = a.t_surf - a.awm(a.t_surf)
    assert a.aw_corr(a.vertical_velocity(), anom) < -0.3


# ---------------------------------------------------------- 单胞结构与位形

def test_w22_single_cell_structure_at_ebm_steady_state():
    """🔴 §三 环流结构（稳态装置）：赤道带辐合 + 高纬辐散，**辐散支起始纬度**在带宽内。

    带宽取 [30°,46°]：下界 = 常密度玩具的 35.3°（`_toy_peak_constant_density` 实测复现），
    上界留状态方程 `1/T` 权重与剖面非 P₂ 的贡献。🔴 这是**带宽判定**，不是"把数硬凑到 30°"
    ——真值若落在带外，本条变红并交话事人，不许改带界常数迁就读数。
    """
    a = _steady_cell()
    rc, cc = a.rows_c, a.cols_c
    assert (rc, cc) == (120, 240), "本条刻意用 s2 天气网格几何（1.5° 行）定位形"
    assert a.awm_band(a.div_u(), 0.0, _BAND_EQ_MAX_DEG) < 0.0, "赤道带必须辐合"
    cross = _div_crossing_lat(a)
    assert 30.0 <= cross <= 46.0, f"辐散支起始纬度 {cross:.2f}° 出带 ⇒ 结构或判据要重新呈报"
    pred = _flux_peak_from_t(a)
    assert abs(pred - cross) <= 5.0, f"T 剖面独立预测 {pred:.2f}° 与实测 {cross:.2f}° 脱节"
    toy = _toy_peak_constant_density(a)
    assert abs(toy - 35.26) < 2.0, f"常密度玩具极限应为 35.3°，实测 {toy:.2f}°（几何核对失效）"
    # 辐散支**存在**且非退化（只有"处处同号"才算不上单胞）
    dv = a.div_u()
    assert float(np.abs(dv[dv > 0]).max()) > 0.0 and float(np.abs(dv[dv < 0]).max()) > 0.0
    # 🔴 单胞的**拓扑**（本批无科氏 ⇒ 辐散区是"辐合带以外的全部高纬"，不是副热带窄带）：
    # 行平均 `div` 的符号序列必须恰为 `+…+ −…− +…+`（**两次**换号、含赤道的一段连续辐合带、
    # 两极侧为辐散）⇒ 若渗进第二/第三个胞（换号 ≥4 次）或辐合带不连赤道，本条变红。
    row = np.sign(dv.reshape(rc, cc).mean(axis=1))
    seq = [int(s) for s in row if s != 0]
    assert len([i for i in range(1, len(seq)) if seq[i] != seq[i - 1]]) == 2, \
        f"符号换号次数 != 2 ⇒ 不是单胞：{seq}"
    assert seq[0] > 0 and seq[-1] > 0, "极侧必须辐散（暖赤道低压的入流来自高纬）"
    assert seq[len(seq) // 2] < 0, "赤道侧必须辐合（换号两次 ⇒ 中段就是赤道带）"
    # 暖升冷沉在稳态下更强（对照 §四-8 的相关判据）
    assert a.aw_corr(a.vertical_velocity(), a.t_surf - a.awm(a.t_surf)) > 0.9


def test_w23_pattern_is_self_similar_in_time_and_north_south_symmetric():
    """🔴 §三「稳定且关于赤道对称」的可核验版本 = **形状不随时间漂移** + 镜像对称。

    「稳定」不能靠"跑长一点看它没炸"：力恒定（冻结 P，W3 已证 `_integrate_wind` 不写 P）时
    每个格按同一个因子 `1−(1−dt/τ)^k` 松弛 ⇒ `div` 的**归一化形状**必须逐位不变、过零纬度
    不漂移。任何"极点特殊处理/极带滤波渗进散度"的实现都会在这里露出来。
    """
    a = _atm(240, 480, 4, surface="sym")
    a.t_surf[:] = _steady_t(a).ravel()
    a._diagnose_pressure()
    shapes = []
    crossings = []
    for k in range(1, 13):
        a._integrate_wind()
        dv = a.div_u()
        peak = float(np.abs(dv).max())
        assert peak > 0.0
        shapes.append(dv / peak)
        crossings.append(_div_crossing_lat(a))
    assert len(shapes) == 12
    ref = shapes[7]                                  # 中段一次作参考
    for k, s in enumerate(shapes, start=1):
        assert np.max(np.abs(s - ref)) < 1e-12, f"第 {k} 子步的环流形状漂了 ⇒ 位形随时间变"
    assert len(set(crossings)) == 1, f"辐散支起始纬度随时间漂移：{sorted(set(crossings))}"
    # 南北镜像：div 对称、u_n 反对称（对称地表 ⇒ 对称输出，Step 2 E8 同口径）。
    # 🔴 阈值不是"随便放宽"：残差地板来自 `_steady_t` 的 LU 三对角解（**不是**风/散度链——
    # 那两步对镜像是精确的：力只依赖 P，P 只依赖 T，积分是逐格同因子缩放）。
    # 本机实测相对残差 div 2.2e-13 / u_n 5.8e-14 ⇒ 取 1e-11（≈50× 余量，跨平台 LU 序差异吃不下）。
    rc, cc = a.rows_c, a.cols_c
    d2 = a.div_u().reshape(rc, cc)
    un = a.u_north.reshape(rc, cc)
    assert float(np.max(np.abs(d2 - d2[::-1]))) < 1e-11 * float(np.abs(d2).max())
    assert float(np.max(np.abs(un + un[::-1]))) < 1e-11 * float(np.abs(un).max())


def test_w24_convergence_and_divergence_branches_both_have_content():
    """§四-11 的**环流侧**：两支各自非零、符号相反，面积权积分相消到机器精度。

    只断言"全球和 = 0"会被"处处为 0"（风没跑起来）骗过 ⇒ 必须先证两支都存在、
    再用同一个 `area` 尺把它们加起来核残差。
    """
    a = _steady_cell()
    dv = a.div_u().reshape(a.rows_c, a.cols_c)
    w = np.repeat(np.asarray(a.area, dtype=np.float64), a.cols_c).reshape(a.rows_c, a.cols_c)
    conv = float((w * np.minimum(dv, 0.0)).sum())
    diver = float((w * np.maximum(dv, 0.0)).sum())
    assert conv < 0.0 and diver > 0.0, f"辐合支 {conv:.3e} / 辐散支 {diver:.3e} 必须异号且非零"
    scale = max(abs(conv), abs(diver))
    assert abs(conv + diver) / scale < 1e-12, "两支总量不相等 ⇒ 质量在单层大气里凭空生灭"
    p = a.probe()
    assert abs(p["div_area_sum_rel_residual"]) < 1e-12
    assert p["div_min_per_m"] < 0.0 < p["div_max_per_m"]
    assert abs(p["w_top_awm_ms"]) / max(p["w_top_absmax_ms"], 1e-300) < 1e-12, \
        "全球均值的 w 不为零 = 单层大气在凭空吞吐质量"


# ------------------------------------------------------ §四-12/13 的 3c 延伸

def test_w25_polar_outer_face_carries_no_flux_and_seam_is_not_special():
    """极外面 `c=0` ⇒ 极行 `div` 只来自内面；把 `u` 整体平移一列 ⇒ `div`/`w` 同样平移。"""
    a = _steady_cell(240, 480, 4)
    rc, cc = a.rows_c, a.cols_c
    un = a.u_north.reshape(rc, cc)
    d2 = a.div_u().reshape(rc, cc)
    # 极行的经向项 = +flux_n[0]/Δμ̃₀（南侧外面零）⇒ 与内面通量同号等值
    c_face0 = float(np.sqrt(max(0.0, 1.0 - a.mu_edge[1] ** 2)))
    expect = c_face0 * 0.5 * (un[0] + un[1]) / (a.area[0] / a._scale_mu2area) / _EARTH_RADIUS_M
    assert np.allclose(d2[0], expect, rtol=1e-15, atol=0.0), \
        "极行多算了一个外面通量（= 极点重复计数）"
    # 接缝：u 平移一列 ⇒ 散度场平移一列（逐位；极点行也一起走，因为它是环带均值）
    d0 = a.div_u().copy()
    w0 = a.vertical_velocity().copy()
    a.u_east[:] = np.roll(a.u_east.reshape(rc, cc), 1, axis=1).ravel()
    a.u_north[:] = np.roll(a.u_north.reshape(rc, cc), 1, axis=1).ravel()
    d1 = a.div_u().reshape(rc, cc)
    w1 = a.vertical_velocity().reshape(rc, cc)
    assert np.array_equal(d1, np.roll(d0.reshape(rc, cc), 1, axis=1)), "§四-12 接缝被特殊处理"
    assert np.array_equal(w1, np.roll(w0.reshape(rc, cc), 1, axis=1))
    assert a.probe()["u_polar_spread"] == 0.0


# ------------------------------------------------------------- 零模型①②③④

def test_w26_null_model_one_uniform_drive_gives_exactly_no_wind():
    """🔴 零模型①（最强形式）：均匀入射 + 单一地表 + 均匀初温 ⇒ 逐位无梯度、无风、无垂直运动。

    这条同时是"漏乘 cos φ / 极带重复计数 / 气压里渗进经纬度纹理"的第二号探测器：驱动场
    完全均匀时，任何**几何**错误都会凭空长出梯度 ⇒ 断言一律取"恰为 0"而不是"接近 0"。
    """
    a = Atmosphere(SphereWorld(120, 240),
                   AtmosphereConfig(enabled=True, coarse_factor=2, insolation_mode="uniform"),
                   _surface(120, 240, "ocean"))
    t_star = (a.awm(a.absorbed) - _EBM_A_WM2) / _EBM_B_WM2K       # 均匀驱动的唯一不动点
    a.t_surf[:] = t_star
    a._diagnose_pressure()
    assert float(np.ptp(a.p_surf)) == 0.0, "气压在均匀温度场里长出了水平结构"
    for k in range(1, 13):
        a.advance(10 * k)
    assert float(np.ptp(a.t_surf)) == 0.0
    assert float(np.abs(a.u_east).max()) == 0.0 and float(np.abs(a.u_north).max()) == 0.0
    assert float(np.abs(a.div_u()).max()) == 0.0
    assert float(np.abs(a.vertical_velocity()).max()) == 0.0
    p = a.probe()
    assert p["force_max_ms2"] == 0.0 and p["w_top_absmax_ms"] == 0.0
    # 常数 T ⇒ T 异常恒零 ⇒ `aw_corr` 按口径返回 nan（分母 0 不给"0 相关"的假象）
    assert np.isnan(p["w_t_anom_corr"])


def test_w26b_null_model_one_needs_uniform_initial_state_to_be_meaningful():
    """①的反面口径（**必须留在测试里**）：只关入射梯度、保留剖面初值 ⇒ 风照旧存在。

    批长 9e4 s vs 辐射弛豫 `C/B≈2.1e7 s`（= 0.4%）⇒ 初值剖面在批内几乎不动。所以"均匀入射
    ⇒ 风≈0"若不同时把初温设为均匀，就是一条**跑不成立的判据**（历史上这类"字面判据"最容易
    被静默放宽成容差）。本条把这条差异钉成事实，而不是留给汇报里的一句话。
    """
    a = Atmosphere(SphereWorld(120, 240),
                   AtmosphereConfig(enabled=True, coarse_factor=2, insolation_mode="uniform"),
                   _surface(120, 240, "ocean"))
    assert float(np.ptp(a.t_surf)) > 40.0, "初值就该是现 insolation 年平剖面（E3 口径），否则装置失效"
    for k in range(1, 13):
        a.advance(10 * k)
    assert float(np.abs(a.u_north).max()) > 0.1, "IC 剖面下竟无风 ⇒ 力或气压链路断了"
    assert float(np.ptp(a.t_surf)) > 0.9 * 40.0, "批内梯度塌了 = 弛豫时间不再是 2.1e7 s（口径要重述）"


def test_w27_null_model_two_no_diffusion_strengthens_the_cell():
    """零模型②的**环流侧**：`D=0` 稳态 ⇒ 剖面更尖 ⇒ 风更强、上升支更强、结构不变，
    且辐散支起始纬度**向极地移动**（通量峰随高纬梯度增强而移位，不是随机变差）。
    """
    on = _steady_cell(240, 480, 4)
    off = _steady_cell(240, 480, 4, diffusion_on=False)
    t_on, t_off = _steady_t(on), _steady_t(off)
    assert float(np.ptp(t_off)) > 2.0 * float(np.ptp(t_on)), "D=0 的稳态极差没有更极端（E8 的 T 侧口径）"
    assert float(np.abs(off.u_north).max()) > float(np.abs(on.u_north).max()), \
        "温差更大却风更弱 ⇒ 气压链路不吃温度梯度"
    w_on, w_off = on.probe(), off.probe()
    assert w_off["w_eq_band_ms"] > w_on["w_eq_band_ms"] > 0.0, "赤道上升支没被加强"
    assert w_off["w_top_absmax_ms"] > w_on["w_top_absmax_ms"]
    assert on.awm_band(on.div_u(), 0.0, _BAND_EQ_MAX_DEG) < 0.0
    assert off.awm_band(off.div_u(), 0.0, _BAND_EQ_MAX_DEG) < 0.0, "关扩散后赤道仍辐合 ⇒ 单胞方向不变"
    c_on, c_off = _div_crossing_lat(on), _div_crossing_lat(off)
    assert c_off > c_on, f"剖面变尖应把辐散支推向极地（实测 {c_on:.2f}° → {c_off:.2f}°）"
    # 两支的守恒都必须成立（零模型不豁免 §四-11）
    for p in (w_on, w_off):
        assert abs(p["div_area_sum_rel_residual"]) < 1e-12


def test_w28_null_model_three_diverges_but_mass_continuity_survives():
    """零模型③：关阻尼 ⇒ `|w|` 随子步线性无界，但 `Σ area·∇·u ≡ 0` **照样逐位成立**。

    这条的要点是"守恒不依赖耗散"：散度恒等式来自几何（望远镜相消），所以风炸了它仍然绿；
    反过来若实现里漏了 `1/c` 或极面通量，发散档会更早把残差放大到可见 ⇒ 有鉴别力。
    """
    b = _atm(240, 480, 4, surface="sym", damping_on=False)
    b.t_surf[:] = _steady_t(b).ravel()
    b._diagnose_pressure()
    dt = b.dt_s
    _, f_n = b.wind_forces()
    peak = []
    for k in range(1, 4001):
        b._integrate_wind()
        if k % 1000 == 0:
            dv = b.div_u()
            expect = (k * dt) * f_n                   # u_k = k·dt·F（无阻尼精确解）
            assert np.allclose(b.u_north.reshape(b.rows_c, b.cols_c), expect,
                               rtol=1e-9, atol=0.0), f"第 {k} 子步不再是线性发散（阻尼没真关）"
            assert abs(b.probe()["div_area_sum_rel_residual"]) < 1e-12, "发散档把守恒炸了"
            peak.append(float(np.abs(dv).max()))
    assert peak[0] < peak[1] < peak[2] and peak[2] > 2.5 * peak[0]
    d = _steady_cell(240, 480, 4)
    assert peak[2] > 3.0 * float(np.abs(d.div_u()).max()), \
        "关阻尼跑了 3000 子步仍不超过开阻尼的平衡散度 ⇒ 零模型③没有反证任何东西"


# ------------------------------------------------------ 键集合 / 一把尺 / 异常

def test_w29_off_mode_has_no_circulation_readouts_and_on_mode_key_set():
    """零模型④ + 键集合：关档不构造（无 `w/div` 读数）；开档七个新键**逐名**在场；
    🔴 带界是**读数窗口**（改它 ⇒ 物理场逐位不动，只有带读数动）。"""
    from simulation.config import EnvFieldConfig, SimConfig
    from simulation.sphere_engine import SphereEngine

    def _cfg(enabled: bool) -> SimConfig:
        cfg = SimConfig(seed=7)
        cfg.world.rows, cfg.world.cols = 120, 240
        cfg.simulation.ticks = 60
        cfg.simulation.use_sim_core = False
        cfg.env_field = EnvFieldConfig(enabled=True, c0b_enabled=True)
        cfg.atmosphere = AtmosphereConfig(enabled=enabled)
        return cfg

    off = SphereEngine(_cfg(False))
    for _ in range(30):
        off.step()
    assert off.atmosphere is None, "关档仍构造大气 ⇒ C7 回滚点风险"
    on = SphereEngine(_cfg(True))
    for _ in range(30):
        on.step()
    p = on.atmosphere.probe()
    for key in _NEW_KEYS:
        assert key in p, f"3c 读数键 `{key}` 缺失"
    assert set(_NEW_KEYS) <= set(p)

    # 读数窗口不是旋钮：把带界改成 [0,5]/[80,85] 再算一次，物理面必须逐位不动
    a = _atm(120, 240, 2)
    for k in range(1, 5):
        a.advance(10 * k)
    before = (a.t_surf.tobytes(), a.p_surf.tobytes(), a.u_east.tobytes(),
              a.u_north.tobytes(), a.div_u().tobytes())
    p0 = a.probe()
    old = (atmo._BAND_EQ_MAX_DEG, atmo._BAND_SUB_MIN_DEG, atmo._BAND_SUB_MAX_DEG)
    try:
        atmo._BAND_EQ_MAX_DEG, atmo._BAND_SUB_MIN_DEG, atmo._BAND_SUB_MAX_DEG = 5.0, 80.0, 85.0
        p1 = a.probe()
    finally:
        atmo._BAND_EQ_MAX_DEG, atmo._BAND_SUB_MIN_DEG, atmo._BAND_SUB_MAX_DEG = old
    after = (a.t_surf.tobytes(), a.p_surf.tobytes(), a.u_east.tobytes(),
             a.u_north.tobytes(), a.div_u().tobytes())
    assert before == after, "带界渗进了计算路径 ⇒ 它就不再是「读数窗口」而是旋钮"
    assert p0["w_eq_band_ms"] != p1["w_eq_band_ms"] and p0["w_sub_band_ms"] != p1["w_sub_band_ms"]
    assert p0["div_awm_per_m"] == p1["div_awm_per_m"], "全球量被带界改动 = 口径不独立"


def test_w30_awm_band_shares_one_ruler_with_awm_and_fails_loud_when_empty():
    """`awm_band` 与 `awm` **同一把 `area` 尺**：全带恒等式 + 分带可加性 + 空带 `ValueError`。

    分子按格求和、分母只按行求和会差整整一个 `cols_c`（=120 倍）而**符号看不出来**（实施本笔
    时真踩过），故恒等式与可加性两条都要钉；全带对 `div` 这种相消场只到舍入 ⇒ 用非相消场核。
    """
    a = _atm(240, 480, 4)
    for k in range(1, 5):
        a.advance(10 * k)
    for arr in (a.t_surf, a.p_surf, a.u_north):
        assert abs(a.awm_band(arr, 0.0, 90.0) - a.awm(arr)) <= 1e-13 * abs(a.awm(arr)) + _ABS
    # 分带可加性：面积权按带拆开再合起来 = 全球均值
    dv = a.div_u()
    edges = [(0.0, 20.0), (20.0, 40.0), (40.0, 60.0), (60.0, 90.0)]
    lat = np.degrees(np.arcsin(a.mu_ctr))
    w_all = np.repeat(np.asarray(a.area, dtype=np.float64), a.cols_c).reshape(a.rows_c, a.cols_c)
    num = sum(a.awm_band(dv, lo, hi) * float(w_all[(lat >= lo) & (lat <= hi)].sum())
              for lo, hi in edges)
    assert abs(num / float(w_all.sum()) - a.awm(dv)) < 1e-12 * max(abs(a.awm(dv)), 1.0) + 1e-30
    with pytest.raises(ValueError, match="带内没有粗网格行"):
        a.awm_band(dv, 91.0, 95.0)


def test_w31_nonfinite_wind_is_fail_loud_in_divergence_and_vertical_proxy():
    """§四-15：`u` 含 NaN/Inf ⇒ `div_u()`/`vertical_velocity()` 抛 `FloatingPointError`（不 clip）。"""
    for fill in (float("nan"), float("inf")):
        a = _atm(120, 240, 2)
        for k in range(1, 4):
            a.advance(10 * k)
        a.u_north[13] = fill
        with pytest.raises(FloatingPointError, match="非有限"):
            a.div_u()
        with pytest.raises(FloatingPointError, match="非有限"):
            a.vertical_velocity()
