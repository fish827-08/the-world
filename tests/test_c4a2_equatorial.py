"""C4a-2 **P-eq**（owner v35 §二-2 开工清单首件）—— 赤道线剖面 + 镜像对称 + 赤道无风带。

权威口径 = `内部汇报/话事人传达-20261017-C4a1收口裁定与C4a2放行-v35.md` §二-2：
「P-eq：**赤道线 u/v 剖面读数** + 对称性判据（镜像残差 1e-12 ⇒ `v(0)=0` 数学必然，
补实测读数）；「**赤道无风带**」写进零模型判据字面」＋ §二-2 规则「每个新场进 probe 时
同步进 digest 报告」（本件**只加读数、不加场** ⇒ digest 图层口径不动，见汇报 §P-eq）。

覆盖（每条都是可核验恒等式 / 可观测行为，不是"字段存在"）：
    eq01 `equator_rows` 的**奇偶两档定义**唯一且夹住赤道；几何不对称 ⇒ `FloatingPointError`
    eq02 `equator_profile` = 与 `div_u` **同一套面索引**的算术平均（逐位）+ 接缝不是特例
         + 三个新读数方法**纯只读**（场与计数逐位不变，W13 同口径）
    eq03 🔴 对称装置下 `v(0)=0`：剖面经向风 `|v(0)| ≤ 1e-12·u_max`（实测 4.7e-18 相对）
         + 五场镜像相对残差 `< 1e-12` + 对称装置**无经度纹理** ⇒ 赤道线纬向风恰为 0
    eq04 「赤道无风带」判据有内容：无风带（|φ|≤5°）与信风侧（5–20°）**都有行**且
         速率比 < 1（实测 0.229）；带内无行 ⇒ `nan`（"问不了"，不是 0）；`awm_band_nan`
         在有行档与 `awm_band` **逐位同尺**
    eq05 🔴 零模型判据字面：均匀入射 + 单一地表 + 均匀初温 ⇒ 无风 ⇒
         **赤道无风带两支同零**、剖面恒零、速率比 `nan`（判据在风关掉时**没有内容**，
         这正是它反证"读数来自风场、不是隐藏常数"的方式）
    eq06 `mirror_rel` 的口径自检（对称/反对称/恒零/形状四档，全用合成场，不借物理装置）
    eq07 🔴 带界（5°/20°）是**读数窗口不是旋钮**：改它 ⇒ 物理场与全球量逐位不动，
         只有两条带读数动（W29 先例）

⚠️ **1e-12 门限的实测地板必须写实**（不放宽、不改带界 —— 迁就读数 = 禁区）：
* 本文件的**中档装置**（rows_f 240×480、factor 4 ⇒ rows_c=60，`_steady_t` 稳态 + 400 子步
  平衡风）`[实测]`：`mirror_rel_u_north` 5.34e-14、`mirror_rel_div` 2.06e-13、
  `mirror_rel_w` 2.06e-13、`mirror_rel_t_surf` 2.33e-15、`mirror_rel_p_surf` 1.42e-16
  ⇒ **五条全过 1e-12**。
* **s2 正式档**（480×960、rows_c=120，C4a-1 验收批同一装置）`[实测]`：`un_mirror_rel`
  **3.23e-13 过**、`div_mirror_rel` **2.10e-12 不过** ⇒ owner 的 1e-12 字面在 s2 的 `div`
  上**未达标**，如实呈报（残差地板来自 `_steady_t` 的 LU 三对角解，不在风/散度链，
  见 `tests/test_c4a1_circulation.py` W23 的同条注记与 `_rerun_logs/c4a1-step3/` 读数件）。
* 本批**无科氏力** ⇒ "赤道无风带"的物理内容是"经向风过赤道换向 ⇒ 赤道附近速率小"，
  不是信风辐聚的真实动力学；无风带随科氏参数"长胖"属 **P-coriolis / C4a-4**。
"""
from __future__ import annotations

import numpy as np
import pytest

from simulation.config import AtmosphereConfig
from world import atmosphere as atmo
from world.atmosphere import (Atmosphere, _BAND_CALM_MAX_DEG, _BAND_TRADE_MAX_DEG,
                              _EBM_A_WM2, _EBM_B_WM2K)
from world.sphere_world import SphereWorld

from tests.test_c4a1_circulation import (_relax_wind, _surface, _steady_t,
                                         _surface_symmetric)

_SYM_ROWS, _SYM_COLS, _SYM_FACTOR = 240, 480, 4      # rows_c=60（偶档 ⇒ 赤道是面）
_MIRROR_MAX = 1e-12                                   # owner v35 §二-2 的字面门限


# ------------------------------------------------------------------ 装置

def _sym_steady_cell(rows: int = _SYM_ROWS, cols: int = _SYM_COLS,
                     factor: int = _SYM_FACTOR) -> Atmosphere:
    """**南北镜像对称**地表 + EBM 稳态温度 + 平衡风（读位形的装置，不是批内状态）。

    与 C4a-1 的 W23 / s2 验收批**同一装置族** ⇒ 残差可直接与历史读数对照（模块 docstring
    给了两档 [实测] 数）。⚠️ 本装置的地表只有纬度差异 + `::3` 河列，而 RIVER/OCEAN
    反照率同值（0.06）⇒ 稳态 `T/P` 沿经度**恒均匀** ⇒ `u_east` 恰为 0（eq03 拿来当
    "无经度纹理"的鉴别力，而不是当"赤道静"的证据）。
    """
    a = Atmosphere(SphereWorld(rows, cols),
                   AtmosphereConfig(enabled=True, coarse_factor=factor),
                   _surface_symmetric(rows, cols))
    a.t_surf[:] = _steady_t(a).ravel()
    a._diagnose_pressure()
    _relax_wind(a, k=400)
    return a


def _band_lat(a: Atmosphere) -> np.ndarray:
    return np.abs(np.degrees(np.arcsin(a.mu_ctr)))


# ------------------------------------------------------------- eq01 奇偶定义

@pytest.mark.parametrize("rows,cols,factor", [(240, 480, 4), (120, 240, 2), (60, 120, 4),
                                              (30, 60, 2), (12, 24, 2)])
def test_eq01_equator_rows_is_defined_by_row_parity_and_straddles_the_equator(
        rows, cols, factor):
    """`equator_rows` 两档都成立、且**唯一夹住赤道**（`argmin` 在偶档会偏向固定一侧）。"""
    a = Atmosphere(SphereWorld(rows, cols), AtmosphereConfig(enabled=True, coarse_factor=factor),
                   _surface(rows, cols))
    rc = a.rows_c
    i_s, i_n = a.equator_rows()
    mu = a.mu_ctr
    assert mu[i_s] < 0.0 < mu[i_n] if rc % 2 == 0 else mu[i_s - 1] < 0.0 < mu[i_s + 1]
    assert (i_s == i_n) is bool(rc % 2)
    assert (i_n - i_s) == (0 if rc % 2 else 1)
    # 唯一性 = **序**性质：赤道对以南全是南半球、以北全是北半球（奇偶两档同一条说法；
    # ⚠️ 不用"相邻对严格夹零"来表述 —— 奇档的中心行 μ 常是 ±0.0，严格不等号会假失败）
    assert bool(np.all(mu[:i_s] < 0.0)) and bool(np.all(mu[i_n + 1:] > 0.0))
    p = a.probe()
    assert (p["equator_i_south"], p["equator_i_north"], p["equator_is_face"]) == (
        i_s, i_n, rc % 2 == 0)


def test_eq01b_equator_rows_is_fail_loud_when_the_grid_stops_being_symmetric():
    """μ 质心不再跨过 0（= 几何被改写/极帽被重排）⇒ 宁炸，不"就近取一行"。"""
    a = Atmosphere(SphereWorld(120, 240), AtmosphereConfig(enabled=True, coarse_factor=2),
                   _surface(120, 240))
    a.mu_ctr = np.abs(a.mu_ctr) + 1e-3                     # 全正 ⇒ 没有赤道
    with pytest.raises(FloatingPointError, match="没有夹住赤道"):
        a.equator_rows()
    b = Atmosphere(SphereWorld(60, 120), AtmosphereConfig(enabled=True, coarse_factor=4),
                   _surface(60, 120))                          # rows_c=15（奇档）
    b.mu_ctr = np.abs(b.mu_ctr) + 1e-3
    with pytest.raises(FloatingPointError, match="中心行不是赤道行"):
        b.equator_rows()


# ------------------------------------------------- eq02 面口径 / 接缝 / 只读

def test_eq02_profile_uses_the_same_face_average_as_div_u_and_only_reads():
    """剖面 = 邻两行的**算术平均**（逐位）；经度平移一列 ⇒ 剖面同样平移；三个方法零写入。"""
    a = _sym_steady_cell()
    rc, cc = a.rows_c, a.cols_c
    i_s, i_n = a.equator_rows()
    ue = a.u_east.reshape(rc, cc)
    un = a.u_north.reshape(rc, cc)
    eq_u, eq_v = a.equator_profile()
    assert eq_u.shape == (cc,) and eq_v.shape == (cc,)
    assert np.array_equal(eq_u, 0.5 * (ue[i_s] + ue[i_n]))          # 与 div_u 的面速度同尺
    assert np.array_equal(eq_v, 0.5 * (un[i_s] + un[i_n]))

    # 接缝不是特例（§四-12 口径，W25 同族）：整场平移一列 ⇒ 剖面同样平移一列
    a.u_east[:] = np.roll(ue, 1, axis=1).ravel()
    a.u_north[:] = np.roll(un, 1, axis=1).ravel()
    pu, pv = a.equator_profile()
    assert np.array_equal(pu, np.roll(eq_u, 1)) and np.array_equal(pv, np.roll(eq_v, 1))

    # 只读自证：`equator_rows`/`equator_profile`/`mirror_rel`/`probe` 前后各场与计数逐位不变
    b = _sym_steady_cell()
    snap = (b.t_surf.tobytes(), b.p_surf.tobytes(), b.mass_col.tobytes(),
            b.u_east.tobytes(), b.u_north.tobytes(), b.div_u().tobytes(),
            b.vertical_velocity().tobytes(), b.steps, b.last_tick)
    for _ in range(3):
        b.equator_rows()
        b.equator_profile()
        b.mirror_rel(b.div_u())
        b.probe()
    after = (b.t_surf.tobytes(), b.p_surf.tobytes(), b.mass_col.tobytes(),
             b.u_east.tobytes(), b.u_north.tobytes(), b.div_u().tobytes(),
             b.vertical_velocity().tobytes(), b.steps, b.last_tick)
    assert snap == after, "P-eq 读数渗进了计算路径 ⇒ 它就不再是读数"
    # 剖面必须喂 2-D 粗网格场（拿主网格/展平场来核 = 口径错，宁炸）
    with pytest.raises(ValueError, match="镜像残差要求粗网格 2-D 场"):
        b.mirror_rel(b.t_surf)


# ---------------------------------------- eq03 对称装置：v(0)=0 + 五场镜像残差

def test_eq03_meridional_wind_on_the_equator_is_zero_by_symmetry():
    """🔴 v35 的字面：镜像残差到 1e-12 ⇒ `v(0)=0` 是**数学必然**；本条把"必然"落成实测数。

    偶档（rows_c=60）赤道是面 ⇒ `v(0) = ½(u_n[29] + u_n[30])`，而 `u_n` 关于赤道**反对称**
    ⇒ 相加只剩求和次序的舍入（实测 −5.55e-17 m/s，对 `u_max=11.69` 的相对值 4.7e-18）。
    同一装置下 `u_east` 沿赤道线**恰为 0.0**（本装置无经度纹理 ⇒ 纬向力恒零）：这条不是
    "赤道静"的证据，而是"剖面没从别处插出噪声"的鉴别力。
    """
    p = _sym_steady_cell().probe()
    assert p["u_max_ms"] > 1.0, "风没跑起来 ⇒ 装置失效，后面的零断言全是恒真"
    assert abs(p["equator_v_mean_ms"]) <= _MIRROR_MAX * p["u_max_ms"]
    assert p["equator_v_absmax_ms"] <= _MIRROR_MAX * p["u_max_ms"]
    assert p["equator_spd_mean_ms"] <= _MIRROR_MAX * p["u_max_ms"]
    for key in ("equator_u_mean_ms", "equator_u_min_ms", "equator_u_max_ms"):
        assert p[key] == 0.0, f"{key} 长出非零值 ⇒ 对称装置里渗进了经度纹理"
    # 两条口径必须**互相印证**（只断言"剖面≈0"会被"整个风场≈0"骗过去）
    assert p["equator_spd_absmax_ms"] <= _MIRROR_MAX * p["u_max_ms"]
    assert p["u_max_ms"] > 1e3 * p["equator_v_absmax_ms"]


def test_eq03b_mirror_residuals_of_the_steady_symmetric_cell_meet_one_e12():
    """五场镜像相对残差 `< 1e-12`（中档 [实测] 全过；s2 的 `div` 2.10e-12 **不过** ⇒ 见模块头）。

    🔴 反"处处为零"的护栏：反对称分量必须**真的**在动（`mirror_rel_u_north` 非零但小于门限、
    `u_max` 与经向风峰值同量级），否则"残差达标"只是"风没跑"的另一种写法。
    """
    a = _sym_steady_cell()
    p = a.probe()
    for key in ("mirror_rel_t_surf", "mirror_rel_p_surf", "mirror_rel_u_east",
                "mirror_rel_u_north", "mirror_rel_div", "mirror_rel_w"):
        assert p[key] < _MIRROR_MAX, f"{key}={p[key]:.3e} ≥ 1e-12（如实失败，不放宽门限）"
    assert p["mirror_rel_u_north"] > 0.0, "u_n 逐位对称 = 风没跑起来"
    un_peak = float(np.abs(a.u_north).max())
    assert 0.1 * p["u_max_ms"] < un_peak <= p["u_max_ms"], "经向风与合成风速量级脱节 ⇒ 读数不同尺"


# ------------------------------------------------ eq04 赤道无风带的读法

def test_eq04_calm_band_is_calmer_than_the_trade_side_and_both_bands_have_rows():
    """「赤道无风带」= |φ|≤5° 的面积权**速率**比信风侧（5–20°）小（实测比 0.229）。

    🔴 带界先钉死再读数（`_BAND_CALM_MAX_DEG`/`_BAND_TRADE_MAX_DEG` 是模块常数，
    不参与任何物理项）；判据用**比值**而不是"绝对小于某 m/s"⇒ 不吃风速标定（红线：禁自调
    τ / 风速标定）。⚠️ 本批无科氏力 ⇒ 这里的"静"来自经向风过赤道换向（eq03），
    宽度随科氏参数变化属 C4a-4 的 P-coriolis 读法。
    """
    a = _sym_steady_cell()
    lat = _band_lat(a)
    n_calm = int(((lat >= 0.0) & (lat <= _BAND_CALM_MAX_DEG)).sum())
    n_trade = int(((_BAND_CALM_MAX_DEG < lat) & (lat <= _BAND_TRADE_MAX_DEG)).sum())
    assert n_calm >= 1 and n_trade >= 1, f"装置分辨率取不到两条带（{n_calm}/{n_trade} 行）"
    p = a.probe()
    assert p["spd_calm_band_ms"] > 0.0 and p["spd_trade_band_ms"] > 0.0
    assert p["spd_calm_band_ms"] < p["spd_trade_band_ms"], "赤道不比信风侧静 ⇒ 判据不成立"
    assert p["calm_trade_ratio"] == p["spd_calm_band_ms"] / p["spd_trade_band_ms"]
    assert 0.0 < p["calm_trade_ratio"] < 1.0
    # "有内容"的另一半：无风带确实远弱于全场峰值（实测 1.01 vs 11.69 m/s ⇒ 0.086 倍）
    assert p["spd_calm_band_ms"] < 0.5 * p["u_max_ms"]


def test_eq04b_empty_band_gives_nan_not_zero_and_the_nan_variant_is_the_same_ruler():
    """微型档（rows_c=6 的 |φ|≤5° 带**没有行**）⇒ `nan` = "这条问不了"，不是 0、不是炸；
    带内有行时 `awm_band_nan` 与 `awm_band` **逐位同尺**（不留第二把尺）。"""
    a = Atmosphere(SphereWorld(12, 24), AtmosphereConfig(enabled=True, coarse_factor=2),
                   _surface_symmetric(12, 24))
    a.t_surf[:] = _steady_t(a).ravel()
    a._diagnose_pressure()
    _relax_wind(a, k=200)
    lat = _band_lat(a)
    assert int((lat <= _BAND_CALM_MAX_DEG).sum()) == 0, "微型档装置前提变了（带里有行了）"
    spd = np.sqrt(a.u_east.reshape(a.rows_c, a.cols_c) ** 2
                  + a.u_north.reshape(a.rows_c, a.cols_c) ** 2)
    with pytest.raises(ValueError, match="带内没有粗网格行"):
        a.awm_band(spd, 0.0, _BAND_CALM_MAX_DEG)
    assert np.isnan(a.awm_band_nan(spd, 0.0, _BAND_CALM_MAX_DEG))
    trade = a.awm_band(spd, _BAND_CALM_MAX_DEG, _BAND_TRADE_MAX_DEG)
    assert a.awm_band_nan(spd, _BAND_CALM_MAX_DEG, _BAND_TRADE_MAX_DEG) == trade
    p = a.probe()
    assert np.isnan(p["spd_calm_band_ms"]) and p["spd_trade_band_ms"] > 0.0
    assert np.isnan(p["calm_trade_ratio"]), "带取不到时比值必须「问不了」，不能给 0/1 的假达标"
    # rows_c=3（奇档，只有 |φ|=0/48.6°）：剖面方法本身仍可跑，但 `probe()` 仍被**既有**
    # `_BAND_SUB`（25–45°）的硬失败拦着 ⇒ P-eq 没有新增分辨率下限，也没解除旧的下限
    tiny = Atmosphere(SphereWorld(6, 12), AtmosphereConfig(enabled=True, coarse_factor=2),
                      _surface(6, 12, "ocean"))
    assert tiny.rows_c == 3 and tiny.equator_rows() == (1, 1)
    tu, tv = tiny.equator_profile()
    assert tu.shape == (tiny.cols_c,) and np.array_equal(tu, tiny.u_east.reshape(
        tiny.rows_c, tiny.cols_c)[1])
    with pytest.raises(ValueError, match="带内没有粗网格行"):
        tiny.probe()


# --------------------------- eq05 🔴 零模型判据字面（含「赤道无风带」）

def test_eq05_null_model_one_zeroes_the_equatorial_calm_band_on_both_sides():
    """🔴 **零模型①（关风）判据字面**：均匀入射 + 单一地表 + 均匀初温 ⇒ 无梯度、无风 ⇒
    **赤道无风带**与信风侧两支的面积权速率**同为 0**、赤道线 u/v 剖面**恒 0**、
    无风带/信风侧**比值 `nan`**（风关掉时这条判据没有内容 —— 反证读数来自风场本身，
    不是模块里藏着的常数）。

    判据必须连同**初值条件**一起写（W26b 的先例）：只关入射梯度而保留纬度初温剖面 ⇒
    风照旧存在 ⇒ "无风带两支同零"不成立（本条的第二半句就是钉这件事）。
    """
    a = Atmosphere(SphereWorld(120, 240),
                   AtmosphereConfig(enabled=True, coarse_factor=2, insolation_mode="uniform"),
                   _surface(120, 240, "ocean"))
    a.t_surf[:] = (a.awm(a.absorbed) - _EBM_A_WM2) / _EBM_B_WM2K     # 均匀驱动的唯一不动点
    for k in range(1, 13):
        a.advance(10 * k)
    p = a.probe()
    assert float(np.abs(a.u_east).max()) == 0.0 and float(np.abs(a.u_north).max()) == 0.0
    for key in ("equator_u_mean_ms", "equator_u_min_ms", "equator_u_max_ms",
                "equator_v_mean_ms", "equator_v_absmax_ms", "equator_spd_mean_ms",
                "equator_spd_absmax_ms", "spd_calm_band_ms", "spd_trade_band_ms"):
        assert p[key] == 0.0, f"关风档 `{key}` 仍非零 ⇒ 该读数不来自风场"
    assert np.isnan(p["calm_trade_ratio"])
    for key in ("mirror_rel_t_surf", "mirror_rel_p_surf", "mirror_rel_u_east",
                "mirror_rel_u_north", "mirror_rel_div", "mirror_rel_w"):
        assert p[key] == 0.0
    # 反面口径（W26b 同族）：只关入射梯度、**保留**纬度初温 ⇒ 判据立刻有内容、且仍"静"
    b = Atmosphere(SphereWorld(120, 240),
                   AtmosphereConfig(enabled=True, coarse_factor=2, insolation_mode="uniform"),
                   _surface_symmetric(120, 240))
    assert float(np.ptp(b.t_surf)) > 40.0, "初值就该是现 insolation 年平剖面（E3 口径），否则装置失效"
    for k in range(1, 13):
        b.advance(10 * k)
    q = b.probe()
    assert q["u_max_ms"] > 0.1, "IC 剖面下竟无风 ⇒ 力或气压链路断了"
    assert q["spd_trade_band_ms"] > 0.0 and q["spd_calm_band_ms"] > 0.0
    assert q["spd_calm_band_ms"] < q["spd_trade_band_ms"], "有风档里赤道不再更静 ⇒ 判据方向变了"
    assert 0.0 < q["calm_trade_ratio"] < 1.0
    # ⚠️ 本档**不**拿 `mirror_rel_*` 过 1e-12：对称地表在全球同反照率下只剩 `heat_cap` 的
    # 经度差（河列 c=4182 vs 洋 3993）⇒ 瞬态会真长出经向/纬向纹理（实测 `u_e` 赤道 ±3.9e-5
    # m/s、`mirror_rel_u_east` 8.9e-11）= **物理**不是数值残差，与 eq03b 的稳态装置不同类。


# ------------------------------------------------ eq06 口径自检（合成场）

def test_eq06_mirror_residual_definition_checked_on_synthetic_fields():
    """对称/反对称/恒零/相对尺四档全用**合成场**核 —— 判据的尺不能只在物理装置里自证。"""
    a = Atmosphere(SphereWorld(12, 24), AtmosphereConfig(enabled=True, coarse_factor=2),
                   _surface(12, 24))
    rc, cc = a.rows_c, a.cols_c
    rows = np.arange(rc, dtype=np.float64)[:, None]
    sym = np.tile((rows - (rc - 1) / 2.0) ** 2, (1, cc))          # 关于赤道对称
    antisym = np.tile(rows - (rc - 1) / 2.0, (1, cc))             # 反对称
    assert a.mirror_rel(sym) == 0.0
    assert a.mirror_rel(sym, odd=True) == pytest.approx(2.0)      # 反对称场拿偶口径 ⇒ 满残差
    assert a.mirror_rel(antisym, odd=True) == 0.0
    assert a.mirror_rel(antisym) == pytest.approx(2.0)
    assert a.mirror_rel(np.zeros((rc, cc))) == 0.0                 # 恒零 ⇒ 0（不是 nan）
    noise = sym.copy()
    noise[0, 0] += 1e-9
    # 相对尺 = 残差/max|arr|；扰动本身只有 1e-9，减法带来 ~ulp(6.25)=9e-16 的绝对舍入
    # ⇒ 该档只能核到 1e-7 量级（再严就是在测浮点减法，不是测口径）
    assert a.mirror_rel(noise) == pytest.approx(1e-9 / float(sym.max()), rel=1e-5)
    with pytest.raises(ValueError, match="镜像残差要求粗网格 2-D 场"):
        a.mirror_rel(np.zeros(rc))


# --------------------------------------------- eq07 带界是窗口不是旋钮

def test_eq07_calm_and_trade_band_windows_are_readout_only_not_knobs():
    """改 `_BAND_CALM/_BAND_TRADE` ⇒ 物理场与**全球量**逐位不动，只有两条带读数动。

    这是 W29 对 `_BAND_EQ/_BAND_SUB` 的同一条自证，挪到 P-eq 的新窗口上：判据字面里
    出现"赤道无风带"之后，带界宽度就成了最容易被事后调的东西 ⇒ 先用测试锁死它
    （"带界是问法不是旋钮"，owner v35 §一-1 对冲突①的原话）。
    """
    a = Atmosphere(SphereWorld(12, 24), AtmosphereConfig(enabled=True, coarse_factor=2),
                   _surface_symmetric(12, 24))
    a.t_surf[:] = _steady_t(a).ravel()
    a._diagnose_pressure()
    _relax_wind(a, k=200)
    fields = (a.t_surf.tobytes(), a.p_surf.tobytes(), a.u_east.tobytes(), a.u_north.tobytes(),
              a.div_u().tobytes(), a.vertical_velocity().tobytes())
    p0 = a.probe()
    old = (atmo._BAND_CALM_MAX_DEG, atmo._BAND_TRADE_MAX_DEG)
    try:
        atmo._BAND_CALM_MAX_DEG, atmo._BAND_TRADE_MAX_DEG = 48.0, 60.0
        p1 = a.probe()
    finally:
        atmo._BAND_CALM_MAX_DEG, atmo._BAND_TRADE_MAX_DEG = old
    fields_after = (a.t_surf.tobytes(), a.p_surf.tobytes(), a.u_east.tobytes(),
                    a.u_north.tobytes(), a.div_u().tobytes(), a.vertical_velocity().tobytes())
    assert fields == fields_after, "带界渗进了计算路径 ⇒ 它是旋钮，判据作废"
    assert np.isnan(p0["spd_calm_band_ms"]) and not np.isnan(p1["spd_calm_band_ms"])
    assert p0["spd_trade_band_ms"] != p1["spd_trade_band_ms"], "窗口换了而带读数没动 = 读数没接线"
    for key in ("u_max_ms", "u_rms_awm_ms", "div_awm_per_m", "w_top_awm_ms",
                "equator_v_mean_ms", "mirror_rel_div", "t_awm", "p_awm_pa"):
        assert p0[key] == p1[key], f"全球量 `{key}` 随带界变 ⇒ 口径不独立"
