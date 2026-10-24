"""C4a-2 Step A（水汽平流 + 蒸发约束 + 水量账本）—— 回归单测。

权威口径 = owner 派工 C4a-2 范围 + `内部汇报/话事人传达-20261017-C4a1收口裁定与C4a2放行-v35.md`
§三/§四（T1–T5 全走 A 支）+ `内部汇报/汇报-20261017-C4a2动手前设计三则.md` +
**owner 2026-10-17 单位修订令**（`qv`=层均混合比 kg/kg、`bucket`=每单位陆地面积 kg/m²、
`wb`=mm）。单位真源 = `world/atmosphere.py` 模块头「C4a-2 Step A 交付」段。

三把尺（本文件每条断言都标尺，不看文档不会读错）：
* `_kg_kg` = 状态量比湿（无量纲）；`col = qv × mass_col` 才是**被输运 / 被收支**的量；
* `_mm`    = 柱水与储水（kg m⁻² 水当量 ≡ mm；`_MM_PER_KG_M2` 恰为 1.0）；
* `_mm_yr` = 收支年率 = 累积 mm ÷ (`steps × dt_s` 物理秒) × `_SECONDS_PER_YEAR`。

覆盖（每条断言钉可观测行为 / 守恒恒等式，不是"字段存在"）：
    m01 常数口径：Magnus 三常数与 `env_field` **同值** + 五个水汽常数 🔴 **不入配置**
    m02 构造期 fail-loud：无 `pr` / 全球无水 / `pr` 负 / `pr` 面积权 0 ⇒ 炸
    m03 默认关不分配：`moisture_on=False` ⇒ 无 `qv`/`bucket`/`wb` 属性 + probe 无水汽键
    m04 🔴 三把尺落地：`column_water() ≡ qv×mass_col`（唯一换算点）+ `qv=1 ⇒ col=mass_col`
        + `AWM(p_rate) ≡ P̄=990 mm/yr`（T4 锚由构造成立）
    m05 平流守恒：`Σarea·∇·(柱水·v) ≡ 0` ⇒ 单步平流不改 `AWM(柱水)` + 算子只读
    m06 迎风 = 保正凸组合 + 极值不增（尖峰装置 `min ≥ 0`、`max ≤ 前 max`）
    m07 🔴 判据②账本闭合：整批手工复算恒等式与 `wb[4]` 一致 + `wb_imbalance_rel` 在容差内
    m08 🔴 判据①主装置（极冠大陆 + 常数北风）：`sea2land > 0` + 内陆递减（分带 + 纬向剖面）
    m09 判据①反证臂：静止风 ⇒ 通量与蒸发恒 0（`E_pot ∝ |v|`）
    m10 🔴 零模型④（关平流）：海→陆通量塌到 0 而源汇仍开 ⇒ 平流是通量唯一来源
    m11 🔴 零模型⑤（关水量约束）：同装置桶透支为负而账本仍闭合；开约束档恒 ≥0
    m12 纯海格的桶恒 0（保护 `bucket_min/max` 读数不被"海上的陆面水"污染）
    m13 🔴 容量在陆面尺上（不乘占比）：手算溢出账 + 子格混合列（wf=0.5）同样用 `W_max` 本身
    m14 蒸发 deficit 用混合比（式子里没有 `mass_col`）+ 饱和/过饱和 ⇒ 恰 0（不"负蒸发"）
    m15 干格不凭空下雨：`qv=0` 且无风 ⇒ 实际降水**恰为 0**，潜在降水 >0
    m16 换算尺手算：`mm_yr` 与 `residence_days` 复算逐位 + 量级门（滞留 ≪ 1 年 ⇒ 批是瞬态）
    m17 快照：键集合 8/5 + 往返逐位 + 缺键干起算 + 形状不符/负值/NaN fail-loud
    m18 🔴 被动性（本 Step 的可证伪判据）：开/关两档热机链 5 个场跑 400 子步**逐位相同**
    m19 probe 只读：水汽开档 probe 前后所有状态与只读缓存逐位不变
    m20 配置：`from_dict` 旧档回默认（水汽必落回关）+ `moisture_on` 无总闸 ⇒ 构造期炸
    m21 平流守恒的反面：`AWM(qv)`（比湿尺）**不**守恒 ⇒ 证明被输运量取柱水不是审美选择
    m22 判据①口径对照：海岸对齐档三种陆域权重**逐位相同**；混合列档份额口径 ≠ 掩码口径
    m23 🔴 CFL 校核按**高纬最小格距**：纬向项分母含 `c_eff` ⇒ 极地行 ν > 赤道行
    m24 CFL fail-loud + 一次性预警：ν>1 ⇒ 炸且状态一字未动；ν∈(0.5,1] ⇒ 只 warn 一次
    m25 引擎级红线：开/关水汽两档主网格等价面 + `t/p/u` 逐位相同 + `rng.draws` 相同
    m26 引擎级快照键随闸：开档写 `atm_qv/bucket/wb`、关档**形状逐位不变**；开档读缺键档 ⇒
        干起算不炸；开档往返后续跑 == 连续跑（水汽链也逐位）
    m27 restore 补校验（复核 §六-1）：`bucket`/`wb` 与 `qv` 同语义 fail-loud；两条例外写实
        （零模型⑤的负桶放行、账本 `Ocean`/`imbalance` 可负只校有限）
"""
from __future__ import annotations

import pickle
import warnings
from pathlib import Path

import numpy as np
import pytest

from simulation.config import AtmosphereConfig, EnvFieldConfig, SimConfig
from simulation.sphere_engine import SphereEngine
from tests.test_c4a1_circulation import _steady_t
from world import env_field as ef
from world.atmosphere import (Atmosphere, _C_E, _CFL_STABILITY_MAX, _CFL_WARN_MAX,
                              _EARTH_RADIUS_M, _EPS_WS, _HPA_TO_PA, _MAGNUS_A, _MAGNUS_B,
                              _MAGNUS_C_HPA, _MM_PER_KG_M2, _P_ANCHOR_KG_M2_S,
                              _POLAR_C_FLOOR, _R_DRY_AIR, _SECONDS_PER_YEAR, _W_MAX_KG_M2,
                              _WB_TOL_REL)
from world.env_field import SURFACE_LAND, SURFACE_OCEAN
from world.sphere_world import SphereWorld

_ROWS, _COLS, _FACTOR = 240, 480, 4      # 判据①装置 ⇒ 粗网格 60×120（陆 30 行）
_MICRO = (120, 240)                      # 其余装置 ⇒ 30×60（秒级）
_REL = 1e-12                             # 守恒/闭合相对门限（= 模块 `_WB_TOL_REL` 同尺）
_N_CRIT = 1200                           # 判据①主装置子步数（≈3.6 日，瞬态；见 m08 声明）


# ------------------------------------------------------------------ 装置

def _cap_land(rows: int, cols: int) -> np.ndarray:
    """北半球整块陆、南半球整块海（🔴 与粗格对齐 ⇒ `wf ∈ {0,1}`、**一条海岸线**）。

    选型理由（相对早先的"赤道纬度带陆块"装置）：判据①的"内陆递减"要的是**单调的离水距**，
    赤道带装置的热源在带心（暖 ⇒ 饱和比湿大 ⇒ 柱水天然高），BFS 环又同时混两条海岸 ⇒
    分带读数会**随离岸距离上升**，被测条件被装置自己抹掉。极冠大陆只有一条海岸 ⇒ 离水距
    与纬度单调同向，雨出（rain-out）剖面可直接读。
    """
    idx = np.arange(rows * cols, dtype=np.int64).reshape(rows, cols)
    return np.where(idx < rows * cols // 2, SURFACE_OCEAN,
                    SURFACE_LAND).ravel().astype(np.uint8)


def _mixed_land(rows: int, cols: int) -> np.ndarray:
    """海岸**不对齐**粗格（边界落在降采样块**中间的整行**）⇒ 出现子格混合列 `wf=0.5`。"""
    idx = np.arange(rows * cols, dtype=np.int64).reshape(rows, cols)
    return np.where(idx < (rows // 2 + 2) * cols, SURFACE_OCEAN,
                    SURFACE_LAND).ravel().astype(np.uint8)



def _all_land(rows: int, cols: int) -> np.ndarray:
    return np.full(rows * cols, SURFACE_LAND, dtype=np.uint8)


def _pr_sea(rows: int, cols: int) -> np.ndarray:
    """降水型只给海洋（陆上 0）⇒ 陆格 `p_pot=0`：桶**只出不进**。

    零模型⑤的必需装置：早先用均匀 `pr` 时，`P = min(柱水, p_rate·Δt)` 把整柱水当场落回桶里
    ⇒ "透支为负"的反证被**同子步的降水**填平（arm D 的 `bucket_min_mm` 恒 0 就是这么来的）。
    ⚠️ 这条不是 bug，是该零模型的口径要求：反证必须让"可用水"真的断源。
    """
    idx = np.arange(rows * cols, dtype=np.int64).reshape(rows, cols)
    return np.where(idx < rows * cols // 2, 1.0, 0.0).ravel()


def _atm(rows: int = _ROWS, cols: int = _COLS, *, factor: int = _FACTOR,
         surface=None, pr=None, no_pr: bool = False, wind=None,
         steady: bool = True, moisture_on: bool = True, **akw) -> Atmosphere:
    """水汽装置：`enabled=True` + 可选冻结到 EBM **稳态解**（判据不测瞬态弛豫）。"""
    st = _cap_land(rows, cols) if surface is None else surface
    prf = None if no_pr else (np.ones(rows * cols, dtype=np.float64) if pr is None else pr)
    a = Atmosphere(SphereWorld(rows, cols),
                   AtmosphereConfig(enabled=True, coarse_factor=factor,
                                    moisture_on=moisture_on, **akw),
                   st, pr_fine=prf)
    if steady:
        a.t_surf[:] = _steady_t(a).ravel()
        a._diagnose_pressure()          # mass_col 必须与新 T 一致（比湿↔柱水的尺）
    if wind is not None:
        a.u_east[:] = float(wind[0])
        a.u_north[:] = float(wind[1])
    return a


def _run(a: Atmosphere, n: int) -> Atmosphere:
    """直接跑 n 个**水分子步**（T/P/u 冻结 = 强迫被动示踪）。

    ⚠️ `steps` 只在 `advance()` 里涨 ⇒ 手工推进必须自计数，否则 `mm/yr` 换算的分母是 0。
    """
    for k in range(n):
        a._moisture_substep()
        a.steps = k + 1
    return a


def _flux_with_weight(a: Atmosphere, w2d: np.ndarray) -> float:
    """按给定陆域权重手工复算海→陆通量（mm/yr）—— 与 `_sea2land_flux_mm_yr` 同式。"""
    dq, _cfl = a.moisture_advection()
    w = a.area[:, None] * w2d
    return -float((w * dq).sum()) / float(w.sum()) * _SECONDS_PER_YEAR


# ------------------------------------------------------- m01 常数与配置边界

def test_m01_constants_match_env_field_and_are_not_config_fields():
    """Magnus 与 `env_field` 同值（蒸发/降水的湿度尺必须一条）；🔴 水汽常数不入配置。"""
    assert (_MAGNUS_A, _MAGNUS_B, _MAGNUS_C_HPA) == (ef._MAGNUS_A, ef._MAGNUS_B, ef._MAGNUS_C)
    assert _EPS_WS == 0.622 and _C_E == 1.32e-3
    assert _W_MAX_KG_M2 == 150.0
    assert _P_ANCHOR_KG_M2_S * _SECONDS_PER_YEAR == pytest.approx(990.0, rel=1e-12)
    # 单位修订令的尺：1 kg/m² 水当量 = 1 mm ⇒ 换算因子恰为 1.0（写进式子里而不是"显然"）
    assert _MM_PER_KG_M2 == 1.0
    names = {f for f in AtmosphereConfig.__dataclass_fields__}
    for forbidden in ("w_max_kg_m2", "c_e", "p_anchor_mm_yr", "eps_ws", "magnus_a",
                      "magnus_b", "magnus_c"):
        assert forbidden not in names, f"{forbidden} 不许成为可调配置项（禁'调参硬凑'）"
    # Step A 只许加**三个 bool 闸**（其余全是模块常数）
    assert {"moisture_on", "moisture_advect_on", "water_limit_on"} <= names
    assert (AtmosphereConfig().moisture_on is False        # 🔴 总闸默认关 = 逐位等价前提
            and AtmosphereConfig().moisture_advect_on is True
            and AtmosphereConfig().water_limit_on is True)


# ------------------------------------------------------ m02 构造期 fail-loud

@pytest.mark.parametrize("kw, exc", [
    ({"no_pr": True}, ValueError),                          # 拿不到 `env_field.pr`
    ({"pr": np.zeros(_MICRO[0] * _MICRO[1])}, FloatingPointError),   # 降水型全 0 ⇒ 归一化无定义
    ({"pr": -np.ones(_MICRO[0] * _MICRO[1])}, FloatingPointError),   # 降水型为负
])
def test_m02_missing_or_illegal_pr_is_fail_loud_at_construction(kw: dict, exc):
    with pytest.raises(exc):
        _atm(*_MICRO, **kw)


def test_m02b_no_ocean_anywhere_is_fail_loud_not_silent_zero_moisture():
    """全球无水面 ⇒ 没有蒸发源：必须炸，不许"构造了水汽场但恒零"（B3 家族）。"""
    with pytest.raises(ValueError, match="没有水面"):
        _atm(*_MICRO, surface=_all_land(*_MICRO))


def test_m02c_moisture_on_without_master_switch_is_config_error():
    """`moisture_on=True` 而 `enabled=False` ⇒ 无人读取 = 静默失效（C9 家族）⇒ 构造期炸。"""
    with pytest.raises(AssertionError, match="moisture_on"):
        AtmosphereConfig(enabled=False, moisture_on=True)


# ------------------------------------------------- m03 默认关：不分配、不读数

def test_m03_off_arm_allocates_nothing_and_probe_has_no_moisture_keys():
    a_off = _atm(*_MICRO, moisture_on=False)
    for name in ("qv", "bucket", "wb", "water_frac", "p_rate", "land_dist"):
        assert not hasattr(a_off, name), f"关档不该分配 `{name}`（内存/tick 面必须不变）"
    assert a_off.moisture_on is False
    pr = a_off.probe()
    assert not [k for k in pr if k.startswith(("qv_", "qcol_", "bucket_", "rh_"))], \
        "关档 probe 混进水汽键 = 旧读数形状被改（红线 1 侧翼）"
    assert "moisture_on" not in pr and "sea2land_flux_mm_yr" not in pr

    a_on = _atm(*_MICRO)
    assert a_on.moisture_on is True
    for name in ("qv", "bucket", "wb", "water_frac", "p_rate", "land_dist"):
        assert hasattr(a_on, name)
    p_on = a_on.probe()
    assert p_on["moisture_on"] is True and p_on["moisture_advect_on"] is True \
        and p_on["water_limit_on"] is True
    # 干起算 / 空桶 / 零账本（不是"随机初值"）
    assert float(a_on.qv.max()) == 0.0 and float(a_on.bucket.max()) == 0.0
    assert a_on.wb.shape == (6,) and float(np.max(np.abs(a_on.wb))) == 0.0


# ------------------------------------------- m04 三把尺与唯一换算点（修订令）

def test_m04_column_water_is_the_single_conversion_and_precip_anchor_holds():
    a = _atm(*_MICRO)
    col = a.column_water()
    assert col.shape == a.qv.shape
    assert np.array_equal(col, a.qv * a.mass_col), "柱水必须 = qv×mass_col（唯一换算点）"
    # qv 是无量纲混合比 ⇒ 注入 1 kg/kg 时柱水 = 该格空气柱质量（尺的自检）
    a.qv[:] = 1.0
    assert np.array_equal(a.column_water(), a.mass_col)
    a.qv[:] = 0.0
    # T4 走 A 支：降水**总量**钉在 990 mm/yr，`pr` 只给空间型 ⇒ AWM(p_rate) ≡ P̄ 由构造成立
    assert float(np.max(np.abs(a.p_rate))) > 0.0
    assert a.awm(a.p_rate) * _SECONDS_PER_YEAR == pytest.approx(990.0, rel=1e-9)
    assert a.probe()["p_anchor_mm_yr"] == pytest.approx(990.0, rel=1e-12)
    # 桶的容量读数是**陆面尺**（不乘占比）
    assert a.probe()["w_max_kg_m2_land"] == _W_MAX_KG_M2


# ------------------------------------------- m05/m06/m21 平流核的守恒与单调性

def test_m05_advection_conserves_area_weighted_column_water_and_is_read_only():
    a = _atm(*_MICRO, wind=(0.0, 5.0))
    rc, cc = a.rows_c, a.cols_c
    rr = np.arange(rc, dtype=np.float64)[:, None]
    jj = np.arange(cc, dtype=np.float64)[None, :]
    a.qv[:] = (0.004 * (1.0 + 0.6 * np.sin(3.0 * rr / rc))
               * (1.0 + 0.3 * np.cos(5.0 * jj / cc))).ravel()
    a._diagnose_pressure()
    before_qv = a.qv.copy()
    before_u = (a.u_east.copy(), a.u_north.copy())
    col0 = a.column_water().reshape(rc, cc)
    awm0 = a.awm(col0)
    div, cfl = a.moisture_advection()
    assert np.array_equal(a.qv, before_qv) and np.array_equal(a.u_east, before_u[0]) \
        and np.array_equal(a.u_north, before_u[1]), "平流算子写了状态 = 被动性破"
    assert 0.0 < cfl < _CFL_WARN_MAX, "本档 ν 应在安全区（判据读数不该靠边界跑）"
    new = col0 - a.dt_s * div
    assert abs(a.awm(new) - awm0) <= _REL * awm0, "Σarea·∇·(柱水·v) 应当望远镜相消"
    assert a.probe()["adv_area_sum_rel_residual"] <= _REL


def test_m06_upwind_is_positive_convex_combination_and_extremum_diminishing():
    """单格尖峰 + 常风：更新是邻格的**凸组合** ⇒ 不出负、极值不增（一阶迎风的单调性）。"""
    a = _atm(*_MICRO, wind=(2.0, 3.0))
    rc, cc = a.rows_c, a.cols_c
    col0 = np.zeros((rc, cc))
    col0[rc // 3, cc // 4] = 10.0
    a.qv[:] = (col0 / a.mass_col.reshape(rc, cc)).ravel()   # 让柱水尺上是尖峰
    before = float(col0.max())
    div, cfl = a.moisture_advection()
    assert cfl <= 1.0
    new = col0 - a.dt_s * div
    assert float(new.min()) >= 0.0, "保正凸组合下出负 = 格式或校核被绕过（红线 4）"
    assert float(new.max()) <= before * (1.0 + 1e-15), "尖峰被放大 = 迎风不是单调格式"
    assert float(new.sum()) <= before + 1e-12


def test_m21_mixing_ratio_is_not_the_conservative_quantity_by_design():
    """反证"被输运量取柱水不是审美"：同一算子下 `AWM(柱水)` 守恒、`AWM(qv)` **不**守恒。

    `mass_col` 是随温度变化的诊断量（不被平流）⇒ 若把比湿当守恒量输运，恒等式虽然成立，
    但它不再是水量守恒，账本里会凭空长出一个"随柱质量走"的源（红线 4 的隐蔽版本）。
    """
    a = _atm(*_MICRO, wind=(0.0, 5.0))
    rc, cc = a.rows_c, a.cols_c
    a.qv[:] = (0.002 + 0.001 * np.linspace(0.0, 1.0, rc))[:, None].repeat(cc, axis=1).ravel()
    col0 = a.column_water().reshape(rc, cc)
    div, _cfl = a.moisture_advection()
    new_col = col0 - a.dt_s * div
    new_qv = new_col / a.mass_col.reshape(rc, cc)
    dc = abs(a.awm(new_col) - a.awm(col0)) / a.awm(col0)
    dq = abs(a.awm(new_qv) - a.awm(a.qv.reshape(rc, cc))) / a.awm(a.qv.reshape(rc, cc))
    assert dc <= _REL, "柱水尺必须守恒"
    assert dq > 1e-6, "比湿尺在这里竟也守恒 ⇒ 装置没拉开 mass_col 的纬度差（判据失去对象）"


# -------------------------------------------- m07 判据②：水量账本闭合（整批）

def test_m07_water_ledger_closes_over_a_batch():
    a = _run(_atm(*_MICRO, wind=(0.0, 4.0)), 300)
    col = a.column_water()
    land = 1.0 - a.water_frac
    # 手工复算：初值干/空 ⇒ 库存 + 净入海 = 累积 imbalance（= wb[4]，只剩求和次序噪声）
    lhs = a.awm(col) + a.awm(a.bucket * land) + a.wb[3] / _MM_PER_KG_M2
    scale = abs(a.awm(col)) + abs(a.wb[3]) + abs(a.wb[0]) + abs(a.wb[1])
    assert abs(lhs - a.wb[4]) <= _REL * scale, "账本恒等式与最终状态不一致（漏项）"
    p = a.probe()
    assert p["wb_imbalance_rel"] <= _WB_TOL_REL
    assert abs(p["wb_imbalance_mm"]) <= _WB_TOL_REL * scale
    assert p["evap_mm_yr"] > 0.0 and p["precip_mm_yr"] > 0.0
    assert a.wb[5] >= a.wb[1], "潜在降水累积必须 ≥ 实际（取走量以柱水为上限）"
    assert 0.0 < p["precip_realized_frac"] <= 1.0
    assert float(col.min()) >= 0.0 and float(a.bucket.min()) >= 0.0
    assert float(a.bucket.max()) <= _W_MAX_KG_M2 * (1.0 + _WB_TOL_REL)


# --------------------------------------- m08/m09 判据①：海→陆输运 + 内陆递减

def test_m08_criterion1_sea_to_land_flux_and_inland_decrease():
    """判据①主装置 = 极冠大陆 + 常数北风 6 m/s，1200 个水分子步。

    ⚠️ 时序声明（owner v35 §二-2：批内数字只作**瞬态**）：1200 子步 × 150 s = 50 h 物理时间
    ≈ 0.7 倍水滞留时间 ⇒ 这是**建立中**的雨出剖面，不是稳态通量。本条钉的是**符号与单调形**
    （判据①问的是"有没有海→陆输运、内陆是不是递减"），量的稳态口径属 s2 长批 + C4a-3。
    """
    a = _run(_atm(wind=(0.0, 6.0)), _N_CRIT)
    p = a.probe()
    rc, cc = a.rows_c, a.cols_c
    assert p["sea2land_flux_mm_yr"] > 0.0, "判据①：陆域水汽辐合必须为正（海→陆）"
    assert p["evap_mm_yr"] > 0.0 and p["precip_mm_yr"] > 0.0
    # 分带问法（离水距 BFS，只服务读数）：d1 > d2 > d3 ≥ d4p，且远内陆显著更干
    d = [p[f"qcol_dist{k}_mm"] for k in (1, 2, 3)] + [p["qcol_dist4p_mm"]]
    assert d[0] > d[1] > d[2] >= d[3], f"内陆递减破：{d}"
    assert d[3] < 0.5 * d[0], f"远内陆没有明显更干：{d}"
    assert p["qcol_dist0_mm"] < d[0], "海面格不该比近海岸陆格更湿（源区被倒置）"
    # 纬向平均剖面（连续版同一条判据）：陆域行 31..50 上柱水沿极地方向不增
    col = a.column_water().reshape(rc, cc)
    zm = col.mean(axis=1)[rc // 2 + 1: rc // 2 + 21]
    assert bool(np.all(np.diff(zm) <= 1e-12)), "纬向平均柱水向极地回升 = 雨出剖面不单调"
    assert zm[0] > 2.0 * zm[-1], f"海岸与内陆温差不足：{zm[0]:.3f} vs {zm[-1]:.3f}"
    assert p["wb_imbalance_rel"] <= _WB_TOL_REL
    assert p["qv_awm_kg_kg"] > 0.0 and 0.0 < p["rh_awm"] < 1.5


def test_m09_criterion1_antiproof_still_air_gives_exactly_zero_flux():
    """反证臂（判据①的"关风"零模型）：`|v|=0` ⇒ 蒸发势与平流同时恒零 ⇒ 全场恰 0。"""
    a = _run(_atm(wind=(0.0, 0.0)), 200)
    p = a.probe()
    assert p["sea2land_flux_mm_yr"] == 0.0
    assert a.wb[0] == 0.0, "无风仍有蒸发 ⇒ E_pot 的 |v| 因子被绕过（红线 2 之外的物理错）"
    assert float(a.qv.max()) == 0.0 and float(a.bucket.max()) == 0.0
    assert p["qcol_awm_mm"] == 0.0 and p["wb_imbalance_mm"] == 0.0


# ------------------------------------------------ m10/m11/m12 零模型④⑤ + 桶

def test_m10_zero_model_4_advection_off_kills_flux_but_keeps_sources():
    """零模型④：关 `−∇·(柱水·v)` ⇒ 海→陆通量**恰为 0**，但源汇仍开（局地基线）。"""
    a = _run(_atm(*_MICRO, wind=(0.0, 5.0), moisture_advect_on=False), 300)
    p = a.probe()
    assert p["moisture_advect_on"] is False
    assert p["sea2land_flux_mm_yr"] == 0.0, "关平流后仍有'海→陆通量' = 判据①被测条件被抹掉"
    assert p["adv_area_sum_rel_residual"] == 0.0, "关平流档不该重算散度"
    assert p["evap_mm_yr"] > 0.0 and p["precip_mm_yr"] > 0.0, "零模型只关一项，源汇必须仍在"
    assert p["wb_imbalance_rel"] <= _WB_TOL_REL
    # 局地球水：柱水不得越过"本地蒸发 + 本地降水上限"的干态 ⇒ 全球量仍钉在源汇平衡上
    assert p["qcol_awm_mm"] > 0.0


def test_m11_zero_model_5_water_limit_off_lets_bucket_go_negative():
    """零模型⑤：断掉陆上降水（`_pr_sea`）⇒ 桶只出不进。开约束 = 恒 ≥0；关约束 = 透支为负。

    🔴 账本两档都逐位闭合 ⇒ "永远湿"改的是**可用水**，不是守恒（C1 静态 band 的老问题在
    这里以反证形式给出，ENV-FIELD §5.3 留给 C4a-2 的动态版）。
    """
    kw = dict(surface=_cap_land(*_MICRO), pr=_pr_sea(*_MICRO), wind=(0.0, 5.0))
    on = _run(_atm(*_MICRO, **kw), 300)
    off = _run(_atm(*_MICRO, water_limit_on=False, **kw), 300)
    land = (1.0 - on.water_frac) > 0.0
    assert float(land.astype(np.int64).sum()) > 0
    assert float(on.bucket.min()) >= 0.0 and float(on.wb.max()) <= _W_MAX_KG_M2 + 1e-9
    assert float(off.bucket.min()) < 0.0, "关约束档桶没有透支 = 反证没接到被测条件"
    assert float(off.bucket[land].min()) < 0.0, "透支必须落在**陆面格**（每单位陆地面积尺）"
    for x in (on, off):
        assert x.probe()["wb_imbalance_rel"] <= _WB_TOL_REL
    assert off.probe()["evap_mm_yr"] > on.probe()["evap_mm_yr"], "关约束 = 陆面蒸发更多"
    # 关约束档不许"负蒸发"：水面项与陆面项各自进账，柱水仍非负
    assert float(off.column_water().min()) >= 0.0


def test_m12_pure_ocean_cells_never_accumulate_a_land_bucket():
    """纯海格（`wf==1`）的桶从不被写入 ⇒ `bucket_min/max` 读数不被"海上的陆面水"污染。"""
    a = _run(_atm(*_MICRO, wind=(0.0, 5.0)), 400)
    ocean = (1.0 - a.water_frac) == 0.0
    assert float(ocean.astype(np.int64).sum()) > 0
    assert float(np.max(np.abs(a.bucket[ocean]))) == 0.0
    assert float(a.bucket[ocean].min()) == 0.0 and float(a.bucket[ocean].max()) == 0.0
    p = a.probe()
    # 两套尺的问句并存且互不覆盖：陆面尺湿度 vs 整格尺储量
    assert p["bucket_awm_land_mm"] >= 0.0
    assert p["bucket_cell_awm_mm"] == pytest.approx(
        a.awm(a.bucket * (1.0 - a.water_frac)), rel=1e-15)
    assert p["bucket_awm_land_mm"] != p["bucket_cell_awm_mm"] or p["bucket_awm_land_mm"] == 0.0


# --------------------------------------------- m13/m14/m15 陆面尺、deficit、上限

def test_m13_capacity_is_per_unit_land_area_not_cell_area():
    """手算溢出账：`R = land·max(0, bucket − W_max)`，容量**不乘**占比（owner 修订令字面）。

    两档同一断言 ⇒ 反证"容量按整格算"的旧口径已退役：子格混合列（`wf=0.5`）的容量仍是
    `W_max` 本身，溢出只按陆面占比进整格账。
    """
    rows, cols = _MICRO
    b0 = _W_MAX_KG_M2 + 50.0
    for surf in (_cap_land, _mixed_land):
        a = _atm(rows, cols, surface=surf(rows, cols))
        a.bucket[:] = b0
        a._moisture_substep()
        a.steps = 1
        land = 1.0 - a.water_frac
        p = a.probe()
        assert float(a.bucket.max()) == _W_MAX_KG_M2, "溢出后桶应恰回容量（陆面尺的 W_max）"
        assert p["runoff_mm_yr"] == pytest.approx(
            50.0 * a.awm(land) * _SECONDS_PER_YEAR / a.dt_s, rel=1e-12), \
            "径流账必须 = 陆面尺溢出 × 陆面占比（整格面积权），不是陆面尺本身"
        assert p["precip_mm_yr"] == 0.0 and p["evap_mm_yr"] == 0.0, "无风 ⇒ 只该有溢出项"
        assert p["wb_imbalance_rel"] <= _WB_TOL_REL
        # 海格的桶可以被人手注入（真实动力学里永不可达：`np.where(has_land,…)` 保证）
        assert float(a.bucket[land == 0.0].max()) == _W_MAX_KG_M2



def test_m14_evaporation_deficit_is_mixing_ratio_not_column_water():
    a = _atm(*_MICRO, wind=(0.0, 3.0))
    t = a.t_surf
    es = _HPA_TO_PA * _MAGNUS_C_HPA * np.exp(_MAGNUS_A * t / (_MAGNUS_B + t))
    q_sat = _EPS_WS * es / (a.p_surf - es)
    assert np.allclose(a.saturation_mixing_ratio(), q_sat, rtol=0.0, atol=0.0), \
        "饱和混合比公式复算不逐位（Magnus/P_s 接线漂了）"
    rho_a = a.p_surf / (_R_DRY_AIR * a.t_kelvin())
    a.qv[:] = 0.0
    e0 = a.evaporation_potential()
    assert np.array_equal(e0, _C_E * rho_a * 3.0 * q_sat), "E_pot 手算复算不等"
    # 饱和 ⇒ 恰 0（deficit 用比湿 ⇒ 式子里没有 mass_col）
    a.qv[:] = q_sat
    assert float(a.evaporation_potential().max()) == 0.0
    # 过饱和 ⇒ 仍 0（不"负蒸发"：凝结一律走降水汇，T3 A 支没有潜热出口）
    a.qv[:] = q_sat * 1.5
    assert float(a.evaporation_potential().min()) == 0.0
    assert float(a.evaporation_potential().max()) == 0.0
    # 风翻倍 ⇒ 通量翻倍（|v| 是唯一动力学因子；同尺检验 m09 的恒零臂）
    a.qv[:] = 0.0
    a.u_north[:] = 6.0
    assert np.array_equal(a.evaporation_potential(), 2.0 * e0)


def test_m15_dry_air_does_not_rain_out_of_nothing():
    """`qv=0` + 无风（无平流输入、无蒸发）⇒ 实际降水**恰为 0**，潜在降水 >0。"""
    a = _atm(*_MICRO, wind=(0.0, 0.0))
    a._moisture_substep()
    a.steps = 1
    p = a.probe()
    assert a.wb[1] == 0.0, "`P=min(柱水,·)` 的截断失效 = 干球凭空下雨"
    assert a.wb[5] > 0.0, "潜在降水累积必须仍进 wb[5]（realized_frac 的分母）"
    assert p["precip_realized_frac"] == 0.0
    assert float(a.qv.max()) == 0.0 and float(a.bucket.max()) == 0.0
    assert p["wb_imbalance_mm"] == 0.0


# ------------------------------------------------ m16 换算尺与量级（mm/yr、天）

def test_m16_year_and_residence_conversions_are_hand_reproducible():
    a = _run(_atm(*_MICRO, wind=(0.0, 5.0)), 250)
    sec = float(a.steps) * a.dt_s
    p = a.probe()
    for key, slot in (("evap_mm_yr", 0), ("precip_mm_yr", 1),
                      ("runoff_mm_yr", 2), ("ocean_mm_yr", 3)):
        assert p[key] == pytest.approx(a.wb[slot] / sec * _SECONDS_PER_YEAR, rel=1e-12), key
    col_awm = a.awm(a.column_water())
    assert p["residence_days_global"] == pytest.approx(
        col_awm / (a.wb[1] / sec * 86400.0), rel=1e-12)
    # 量级门（`residence_days` 曾因多除一次 86400 给出 1e10 day 的荒谬读数）
    assert 0.1 < p["residence_days_global"] < 60.0
    # 批的物理时长 < 水滞留时间 ⇒ 判据读数**只能是瞬态**（v35 §二-2 的时序声明依据）
    assert _N_CRIT * a.dt_s < 10.0 * p["residence_days_global"] * 86400.0
    assert 0.0 < p["evap_mm_yr"] < 1.0e5


# ------------------------------------------------------------- m17 快照与回退

def test_m17_snapshot_keys_track_gate_and_round_trip_bit_exact():
    a = _run(_atm(*_MICRO, wind=(0.0, 5.0)), 60)
    st = a.state()
    assert set(st) == {"t_surf", "u_east", "u_north", "steps", "last_tick",
                       "qv", "bucket", "wb"}, "开档键集合应为 5+3（诊断量不进档）"
    b = _atm(*_MICRO, wind=(0.0, 5.0))
    b.restore(st)
    for key in ("qv", "bucket", "wb", "t_surf", "u_east", "u_north"):
        assert np.array_equal(getattr(b, key), st[key]), f"`{key}` 往返不逐位"
    assert b.steps == a.steps and b.last_tick == a.last_tick
    assert set(_atm(*_MICRO, moisture_on=False).state()) == {
        "t_surf", "u_east", "u_north", "steps", "last_tick"}, "关档不许写水汽键（红线 1）"

    # 缺键 = Step A 之前的旧档 ⇒ 干起算 / 空桶 / 零账本（不炸；但**不承诺**与从头开档可比）
    c = _atm(*_MICRO, wind=(0.0, 5.0))
    c.restore({"t_surf": st["t_surf"], "steps": st["steps"], "last_tick": st["last_tick"]})
    assert float(c.qv.max()) == 0.0 and float(c.bucket.max()) == 0.0
    assert float(np.max(np.abs(c.wb))) == 0.0

    with pytest.raises(ValueError, match="形状"):
        c.restore({**st, "qv": np.zeros(a.n_c + 1)})
    with pytest.raises(FloatingPointError, match="比湿为负"):
        bad = st.copy()
        bad["qv"] = np.full(a.n_c, -1e-6)
        c.restore(bad)
    with pytest.raises(FloatingPointError):
        bad = st.copy()
        bad["qv"] = np.full(a.n_c, np.nan)
        c.restore(bad)


# --------------------------------------------- m18/m19 被动性与 probe 只读（红线）

def test_m18_thermal_chain_is_bit_identical_with_moisture_on_and_off():
    """🔴 本 Step 的**可证伪判据**：水汽是纯被动示踪 ⇒ 热机链 5 个场逐位相同（400 子步）。

    派工原文"Step A（纯被动，T/P/u 与 C4a-1 逐位同）"的字面落实 ⇒ 只要有一条水汽量反写
    回 `T/P/u/mass_col`（例如提前接潜热 = T3 明确不接），本条立刻红。
    """
    kw = dict(wind=(0.0, 6.0))
    on = _atm(*_MICRO, moisture_on=True, **kw)
    off = _atm(*_MICRO, moisture_on=False, **kw)
    assert float(np.max(np.abs(on.t_surf - off.t_surf))) == 0.0
    assert float(np.max(np.abs(on.u_north))) > 0.0 and \
        np.array_equal(on.u_north, off.u_north), "装置必须真的有非零风（且两档同源）"
    for k in range(400):
        for x in (on, off):
            x._ebm_substep()
            x._diagnose_pressure()
            x._integrate_wind()
        on._moisture_substep()
        on.steps = off.steps = k + 1
    for name in ("t_surf", "p_surf", "mass_col", "u_east", "u_north"):
        assert np.array_equal(getattr(on, name), getattr(off, name)), \
            f"开水汽后 `{name}` 变了 ⇒ 被动性破（红线 4 的因果方向）"
    assert float(on.qv.max()) > 0.0, "开档必须真的算了水汽（否则逐位相同是空转）"


def test_m19_probe_is_read_only_with_moisture_on():
    a = _run(_atm(*_MICRO, wind=(0.0, 5.0)), 120)
    names = ("t_surf", "p_surf", "mass_col", "u_east", "u_north", "qv", "bucket", "wb",
             "water_frac", "p_rate", "land_dist", "steps", "last_tick")
    before = {n: (a.__getattribute__(n).copy() if isinstance(
        a.__getattribute__(n), np.ndarray) else a.__getattribute__(n)) for n in names}
    cfl, adv, warned = a._cfl_max, a._adv_res_rel, a._cfl_warned
    p1 = a.probe()
    p2 = a.probe()
    for n in names:
        cur = a.__getattribute__(n)
        assert np.array_equal(cur, before[n]) if isinstance(cur, np.ndarray) \
            else cur == before[n], f"probe() 写了 `{n}`"
    assert (a._cfl_max, a._adv_res_rel, a._cfl_warned) == (cfl, adv, warned)
    assert set(p1) == set(p2), "probe 两次调用键集合不同 = 内部有状态"
    for key, val in p1.items():
        other = p2[key]
        both_nan = (isinstance(val, float) and isinstance(other, float)
                    and np.isnan(val) and np.isnan(other))
        assert val == other or both_nan, f"probe 不自幂：`{key}` {val} vs {other}"


# --------------------------------------------------------------- m20 配置面

def test_m20_from_dict_falls_back_to_moisture_defaults():
    """旧档（整组缺键 / 缺三个 bool）⇒ 回退默认；🔴 水汽必须落回**关**（逐位等价前提）。"""
    d = SimConfig().to_dict()
    d.pop("atmosphere")
    assert SimConfig.from_dict(d).atmosphere.moisture_on is False
    for key in ("moisture_on", "moisture_advect_on", "water_limit_on"):
        d = SimConfig().to_dict()
        d["atmosphere"].pop(key)
        got = SimConfig.from_dict(d).atmosphere
        assert getattr(got, key) == getattr(AtmosphereConfig(), key)
    cfg = SimConfig()
    cfg.atmosphere = AtmosphereConfig(enabled=True, moisture_on=True,
                                      moisture_advect_on=False, water_limit_on=False)
    back = SimConfig.from_dict(cfg.to_dict())
    assert (back.atmosphere.moisture_on, back.atmosphere.moisture_advect_on,
            back.atmosphere.water_limit_on) == (True, False, False)


# ------------------------------------------- m22/m23/m24 判据①口径与红线 5（CFL）

def test_m22_sea_to_land_weightings_agree_only_on_coarse_aligned_coast():
    """海岸对齐 ⇒ 三种陆域权重**逐位相同**；混合列档 ⇒ 份额口径 ≠ 掩码口径（选型依据）。"""
    a = _run(_atm(*_MICRO, wind=(0.0, 5.0)), 150)
    wf = a.water_frac.reshape(a.rows_c, a.cols_c)
    assert bool(np.all((wf == 0.0) | (wf == 1.0))), "对齐档必须没有子格混合列"
    share = _flux_with_weight(a, 1.0 - wf)
    mask = _flux_with_weight(a, (wf < 1.0).astype(np.float64))
    pure = _flux_with_weight(a, (wf == 0.0).astype(np.float64))
    assert a._sea2land_flux_mm_yr() == share == mask == pure, \
        "海岸对齐粗格的三种权重必须重合（否则口径选择无法自证）"

    m = _run(_atm(*_MICRO, surface=_mixed_land(*_MICRO), wind=(0.0, 5.0)), 150)
    wfm = m.water_frac.reshape(m.rows_c, m.cols_c)
    assert bool(np.any((wfm > 0.0) & (wfm < 1.0))), "混合列装置没造出 wf∈(0,1)"
    assert _flux_with_weight(m, 1.0 - wfm) != _flux_with_weight(
        m, (wfm < 1.0).astype(np.float64)), "混合列档两口径应给出不同问句"
    assert m.probe()["wb_imbalance_rel"] <= _WB_TOL_REL


def test_m23_cfl_is_checked_on_the_fine_high_latitude_cells(monkeypatch):
    """红线 5：ν 的分母含 `c_eff = max(c, _POLAR_C_FLOOR)` ⇒ 高纬小格距**算进**校核，
    不是算完再补。三条各自钉一个可观测事实（纯纬向常风 ⇒ ν 只由经度项给）。"""
    a = _atm(*_MICRO, wind=(1.0, 0.0))
    a._moisture_substep()
    a.steps = 1
    c_eff = np.maximum(a._c_ctr, _POLAR_C_FLOOR)
    nu = 2.0 / (_EARTH_RADIUS_M * c_eff * a._dlam_c)          # 1 m/s 常风的双面出流
    assert a._cfl_max == pytest.approx(a.dt_s * float(nu.max()), rel=1e-12), \
        "ν_max 与几何手算不符 ⇒ 校核取的不是高纬最小格距"
    assert int(np.argmax(nu)) in (0, a.rows_c - 1), "最大 ν 不在极带行 ⇒ 极带放大漏算"
    assert float(nu.max()) > 2.0 * float(nu[a.rows_c // 2]), "极带未被 1/c 放大（判据空转）"

    # 地板**是** ν 的一部分：抬高地板 ⇒ ν 整体抬到 `1/0.5` 的口径（极带被 cap，赤道不受影响）
    with monkeypatch.context() as mp:
        mp.setattr("world.atmosphere._POLAR_C_FLOOR", 0.5)
        b = _atm(*_MICRO, wind=(1.0, 0.0))
        b._moisture_substep()
        assert b._cfl_max == pytest.approx(
            b.dt_s * 2.0 / (_EARTH_RADIUS_M * 0.5 * b._dlam_c), rel=1e-12)
        assert b._cfl_max < a._cfl_max, "抬高地板后 ν 没有下降 ⇒ 校核吃的是原始 1/c，不是 c_eff"



def test_m24_cfl_above_one_is_fail_loud_and_half_range_warns_once():
    """ν>1 ⇒ 炸（🔴 不 clip `qv`、不全局限小步）；ν∈(0.5,1] ⇒ 一次性 RuntimeWarning。"""
    base = _atm(*_MICRO, wind=(0.0, 1.0))
    _div, nu_per_ms = base.moisture_advection()      # ν 与风线性（单向常风 ⇒ 严格成正比）
    assert nu_per_ms > 0.0

    hot = _atm(*_MICRO, wind=(0.0, _CFL_STABILITY_MAX * 1.5 / nu_per_ms))
    snap = (hot.qv.copy(), hot.bucket.copy(), hot.wb.copy())
    with pytest.raises(FloatingPointError, match="CFL"):
        hot._moisture_substep()
    assert np.array_equal(hot.qv, snap[0]) and np.array_equal(hot.bucket, snap[1]) \
        and np.array_equal(hot.wb, snap[2]), "越界后仍改了状态 = 校核没拦在写之前"

    warm = _atm(*_MICRO, wind=(0.0, _CFL_WARN_MAX * 1.5 / nu_per_ms))
    assert _CFL_WARN_MAX < warm.moisture_advection()[1] < _CFL_STABILITY_MAX
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        warm._moisture_substep()
        warm._moisture_substep()
    hits = [w for w in rec if issubclass(w.category, RuntimeWarning) and "CFL" in str(w.message)]
    assert len(hits) == 1, "预警应一次性（刷日志 = 长跑批里把 stderr 淹掉）"
    assert warm.probe()["cfl_warned"] is True and warm.probe()["cfl_max"] > _CFL_WARN_MAX


# ------------------------------------------------- m25/m26 引擎级红线（关档等价）

def _cfg(moisture_on: bool = True, ticks: int = 24, seed: int = 42) -> SimConfig:
    cfg = SimConfig(seed=seed)
    cfg.world.rows, cfg.world.cols = _MICRO
    cfg.simulation.ticks = ticks
    cfg.simulation.use_sim_core = False
    cfg.env_field = EnvFieldConfig(enabled=True, c0b_enabled=True)
    cfg.atmosphere = AtmosphereConfig(enabled=True, coarse_factor=_FACTOR,
                                      moisture_on=moisture_on)
    return cfg


def _engine_state(eng: SphereEngine) -> dict:
    """引擎等价面（与 C4a-1 同口径：主网格 + 生物 + RNG 状态 + tick，**不含大气**）。"""
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


def test_m25_engine_moisture_on_leaves_main_tick_and_thermal_chain_bit_identical():
    """红线 1+2 的引擎级闸门：开/关水汽跑同一 120 tick ⇒ 主网格、热机链、RNG 消费逐位相同。

    ⚠️ `[实测]` 附带发现（进首次汇报的挂账，不是 bug）：本装置 `|v|` 弛豫到 ≈1.4 m/s 之前
    `qv ≡ 0`（24 tick 时 `wb[0] > 0` 而 `qcol_awm_mm = 0`）。原因是 `P = min(柱水, p_rate·Δt)`
    的**每子步降水容量**（P̄=990 mm/yr ⇒ 4.7e-3 mm/子步）大于低风下的蒸发供给
    （`E_pot ∝ |v|`）⇒ 蒸出去的水当场全部落回，大气库恒零 ⇒ 平流与海→陆通量也无从发生
    （`sea2land = −0.0`）。120 tick 后 `col_awm=3.0e-3 mm`、400 tick `0.165 mm` 且通量转正。
    ⇒ 判据①的**可观测门槛**是风（属 C4a-1 的 u_max 读数域），而"库能存多少水"要等 C4a-3 的
    饱和限制/诊断云给降水一个弛豫时间才谈得上稳态口径（owner v35 §二-2 的稳态装置要求）。
    """
    on, off = SphereEngine(_cfg(True, ticks=120)), SphereEngine(_cfg(False, ticks=120))
    assert on.atmosphere.moisture_on is True and off.atmosphere.moisture_on is False
    for _ in range(120):
        on.step()
        off.step()
        assert on.rng.draws == off.rng.draws, "水汽链引入了新 RNG 消费（红线 2）"
    assert _engine_state(on) == _engine_state(off), "开档改了主 tick 面 ⇒ C7 回滚点破"
    for name in ("t_surf", "p_surf", "mass_col", "u_east", "u_north"):
        assert np.array_equal(getattr(on.atmosphere, name), getattr(off.atmosphere, name)), \
            f"引擎级 `{name}` 被水汽改写 ⇒ 被动性破"
    a = on.atmosphere
    assert float(a.qv.max()) > 0.0, "本 tick 数下水汽链必须已经存住水（否则逐位相同是空转）"
    assert a.wb[0] > 0.0 and a.wb[1] > 0.0 and float(a.bucket.max()) > 0.0
    assert a.probe()["moisture_on"] is True and "sea2land_flux_mm_yr" in a.probe()
    assert a.probe()["wb_imbalance_rel"] <= _WB_TOL_REL


def test_m25b_engine_at_zero_wind_spinup_holds_no_moisture_at_all():
    """docstring 里那条门槛的可执行版本：早期（`|v|`<门槛）⇒ `柱水` **恰为 0**、通量 0。

    钉住"没有幻影水汽"：账本记了蒸发与降水（`wb[0]=wb[1]>0`），但大气库存与海陆输量为零
    ⇒ Step A 的收支不会在低风区凭空造出一个"看得见的水汽场"。
    """
    eng = SphereEngine(_cfg(True, ticks=24))
    for _ in range(24):
        eng.step()
    a = eng.atmosphere
    assert float(np.abs(a.u_north).max()) < 1.0
    assert float(a.qv.max()) == 0.0 and float(a.column_water().max()) == 0.0
    assert a.probe()["sea2land_flux_mm_yr"] == 0.0
    assert a.wb[0] > 0.0 and a.wb[1] == a.wb[0]        # 蒸多少降多少（同子步全雨出）
    assert a.probe()["precip_realized_frac"] < 1.0


def test_m26_engine_snapshot_writes_keys_only_when_gated(tmp_path: Path):
    """T5 走 A 支：`SNAPSHOT_VERSION` **不升**；开档追加 3 键、关档形状逐位不变；
    开档往返后续跑 == 连续跑（水汽是累积状态 ⇒ 不进档就重建不出来）。"""
    assert SphereEngine.SNAPSHOT_VERSION == 3, "T5 走 A 支 = 不升版本号（升了要交裁）"
    on = SphereEngine(_cfg(True))
    for _ in range(24):
        on.step()
    path_on = tmp_path / "on.npz"
    on.save_snapshot(path_on)
    with np.load(str(path_on), allow_pickle=False) as data:
        assert {"atm_qv", "atm_bucket", "atm_wb"} <= set(data.files)
        assert data["atm_qv"].shape == (on.atmosphere.n_c,)

    off = SphereEngine(_cfg(False))
    for _ in range(24):
        off.step()
    path_off = tmp_path / "off.npz"
    off.save_snapshot(path_off)
    with np.load(str(path_off), allow_pickle=False) as data:
        assert sorted(k for k in data.files if k.startswith("atm_")) == \
            ["atm_last_tick", "atm_steps", "atm_t_surf", "atm_u_east", "atm_u_north"], \
            "关档快照多写键 = 旧档形状被改（红线 1）"

    resumed = SphereEngine.load_snapshot(str(path_on))
    for name in ("qv", "bucket", "wb", "t_surf", "u_east", "u_north"):
        assert np.array_equal(getattr(resumed.atmosphere, name),
                              getattr(on.atmosphere, name)), f"`{name}` 未随快照走"
    assert resumed.atmosphere.steps == on.atmosphere.steps
    for _ in range(12):
        on.step()
        resumed.step()
    for name in ("qv", "bucket", "wb", "t_surf", "p_surf", "u_east", "u_north"):
        assert np.array_equal(getattr(on.atmosphere, name),
                              getattr(resumed.atmosphere, name)), f"续跑 ≠ 连续跑（`{name}`）"
    # `rng_draws` 是 provenance 计数器、按设计不进快照（C4a-1 P6b 同口径）⇒ 比对时排除
    assert {k: v for k, v in _engine_state(on).items() if k != "draws"} == \
        {k: v for k, v in _engine_state(resumed).items() if k != "draws"}


# ------------------------------------------ m27 restore 对 bucket/wb 的同语义校验

def test_m27_restore_validates_bucket_and_ledger_with_qv_semantics():
    """复核件 §六-1（owner 2026-10-17 转达令）：`bucket`/`wb` 与 `qv` 同语义 fail-loud。

    口径 = "档里的值必须落在**写入侧不变式**的范围内"，两条例外必须写实（不是放松）：
    ① 桶的非负只在 `water_limit_on=True` 下由构造保证 ⇒ 零模型⑤的档**按设计**带负桶，
       该档只校有限性（否则反证臂的快照无法续跑）；
    ② 账本 `Ocean`（海洋库净收支）与 `imbalance`（带符号残差）**可为负** ⇒ 只校有限性，
       对六槽做"非负"会误杀合法账本。
    """
    src = _run(_atm(*_MICRO, wind=(0.0, 5.0)), 60)
    st = src.state()

    # 好档自证：校验不许误杀合法快照（往返逐位，含 bucket/wb）
    ok = _atm(*_MICRO, wind=(0.0, 5.0))
    ok.restore(st)
    assert np.array_equal(ok.bucket, st["bucket"]) and np.array_equal(ok.wb, st["wb"])

    for key, bad_val, msg in (("bucket", -1e-6, "水桶为负"),
                              ("bucket", np.nan, "水桶含 NaN"),
                              ("wb", np.nan, "账本含 NaN"),
                              ("wb", np.inf, "账本含 NaN")):
        bad = st.copy()
        arr = np.array(st[key], dtype=np.float64, copy=True)
        arr.flat[3] = bad_val
        bad[key] = arr
        with pytest.raises(FloatingPointError, match=msg):
            ok.restore(bad)

    # wb 的**每一槽**都在校验范围内（含按定义可负的 Ocean/imbalance ⇒ 有限性无死角）
    for slot in range(6):
        bad = st.copy()
        arr = np.array(st["wb"], dtype=np.float64, copy=True)
        arr[slot] = np.inf
        bad["wb"] = arr
        with pytest.raises(FloatingPointError, match="账本含 NaN"):
            ok.restore(bad)

    # 合法负值不误杀：账本 Ocean/imbalance 槽为负 ⇒ 放行（非负校验只属于 qv/bucket）
    signed = st.copy()
    arr = np.array(st["wb"], dtype=np.float64, copy=True)
    arr[3] = -7.5
    arr[4] = -1e-15
    signed["wb"] = arr
    ok.restore(signed)
    assert float(ok.wb[3]) == -7.5 and float(ok.wb[4]) == -1e-15

    # 零模型⑤口径：`water_limit_on=False` 的负桶是**设计行为**（m11）⇒ 只校有限、放行续跑
    dry = _atm(*_MICRO, wind=(0.0, 5.0), water_limit_on=False)
    neg = st.copy()
    neg["bucket"] = np.full(src.n_c, -1.0)
    dry.restore(neg)
    assert float(dry.bucket.min()) == -1.0
    with pytest.raises(FloatingPointError, match="水桶含 NaN"):
        bad = st.copy()
        bad["bucket"] = np.full(src.n_c, np.nan)
        dry.restore(bad)

    # 形状口径不变（既有校验没被新校验抢掉）：`bucket` 形状不符仍是 ValueError「形状」
    with pytest.raises(ValueError, match="形状"):
        ok.restore({**st, "bucket": np.zeros(src.n_c + 1)})

