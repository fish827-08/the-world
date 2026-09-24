"""13.7 季节光照 + 光驱动再生 —— 回归单测（`[所有者·天平]` 补写，R197 V2）。

背景（R197 V2 缺口）：13.7 交付时**零单测** —— `tests/` 里
`light_sensitivity` / `light_normalize` / `tilt_rad` / `season_period` **0 命中**
⇒ DEL-8（变异测试：改坏接线 ⇒ 测试必须变红）**结构性无法执行**。
本文件补上，并把"关档等价 / 真生效 / H3 / 变异敏感性"四条都变成断言。

覆盖：
    S1 关档逐位等价：全默认 vs 显式传默认 ⇒ digest 完全相同
    S2 C7 基线不动：默认档 vs 钉死基线 `(574887, 11266.746993)`
    S3 `tilt_rad`/`season_period` **真生效**：日累积光照的**南北不对称随季节翻转**
       （关档：南北对称且不随时间变；开档：t=P/4 北半球亮、t=3P/4 南半球亮）
    S4 `light_sensitivity` **真生效**：再生量场异于关档（零机时，直接调内部函数）
    S5 `light_normalize` 的**保总量语义**：开启后全球平均再生量显著更接近关档
    S6 **H3 fail-loud ×2**：开季节 / 开光驱动再生 + `use_sim_core=True` ⇒ 构造期硬报错
    S7 配置自洽：`tilt_rad` 与 `season_period` 必须**同时**给（否则语义不明 ⇒ ValueError）
    S8 🔴 **DEL-8 变异敏感性（机械化）**：开档 vs 关档 digest **必须不同** ——
       若有人删掉 `SphereEngine.__init__` 里对 `light_sensitivity`/`tilt_rad` 的**透传**
       （即 13.5 `bg_production_zero` 家族那种"少一行接线 ⇒ 死字段"），本测试**立刻变红**。
       这是把"变异测试"从**人工流程**变成**常驻断言**的做法。

设计原则（教训库）：**默认关 = 旧行为逐位一致**（S1/S2 钉死）；每条只改一个参数、
断言**可观测的行为差异**（不是"字段存在"）——避免"有测试却不验不变量"（A#8）。
"""

from __future__ import annotations

import numpy as np
import pytest

from simulation.config import InfoStructureConfig, SimConfig
from simulation.sphere_engine import SphereEngine

ROT = 2400            # 自转周期（默认）：一天
SEASON = 4 * ROT      # 一个季节循环 = 4 天（便于 t=P/4 与 t=0 日相位对齐）


def _engine(*, tilt_rad: float = 0.0, season_period: int = 0,
            light_sensitivity: float = 0.0, light_normalize: bool = False,
            use_sim_core: bool = False, max_count: int = 600) -> SphereEngine:
    """构造引擎；季节/光照走**配置字段**（与 a4 的 CLI 同一条路径）。"""
    cfg = SimConfig(seed=42)
    cfg.simulation.use_sim_core = use_sim_core
    cfg.simulation.l2_dash = False
    cfg.population.max_count = max_count
    cfg.predation.forage_tradeoff_k = 0.0
    cfg.info_structure = InfoStructureConfig(
        enabled=True, learning_rate=0.05, memory_gradient="none"
    )
    cfg.organisms.young_mob_mult = 0.55
    cfg.organisms.old_mob_mult = 0.55
    cfg.organisms.dash_min_energy_frac = 0.2
    cfg.organisms.far_cap = 32
    cfg.light.tilt_rad = float(tilt_rad)
    cfg.light.season_period = int(season_period)
    cfg.resources.light_sensitivity = float(light_sensitivity)
    cfg.resources.light_normalize = bool(light_normalize)
    return SphereEngine(cfg)


def _run(e: SphereEngine, ticks: int = 50) -> SphereEngine:
    for _ in range(ticks):
        if e.extinct:
            break
        e.step()
    return e


def _digest(e: SphereEngine) -> tuple[int, float]:
    return (int(e._flat.sum()), round(float(e._energy.sum()), 6))


def _daily_mean_illum(e: SphereEngine, t0: int, samples: int = 24) -> np.ndarray:
    """一个自转周期内的**逐格平均光照**（= 日累积光积分的代理量）。

    `LightAndTemperature.illumination(flat, tick)` 是**瞬时**光照，含昼夜项
    ⇒ 必须在整日上平均，才能看到"季节"这一纬度带效应（这是 R196 文档
    `photoperiod_check.py` 的 DLI 口径在测试里的最小化复用）。
    """
    flat = np.arange(e.world.n_cells)
    acc = np.zeros(e.world.n_cells, dtype=float)
    for k in range(samples):
        acc += e.light.illumination(flat, t0 + k * ROT // samples)
    return acc / samples


def _hemi_means(e: SphereEngine, t0: int) -> tuple[float, float]:
    """返回 (北半球平均, 南半球平均) 的日累积光照。

    ⚠️ `world.latitude_of()` 吃的是**纬度行号**（0..rows-1），不是平铺索引
    ⇒ 必须先用 `flat_to_rc()` 转换（这里踩过一次）。
    """
    flat = np.arange(e.world.n_cells)
    rows_idx, _cols = e.world.flat_to_rc(flat)
    lat = np.asarray(e.world.latitude_of(rows_idx), dtype=float)
    ill = _daily_mean_illum(e, t0)
    north = lat > 0.0
    south = ~north
    return float(ill[north].mean()), float(ill[south].mean())


# --------------------------------------------------------------------------- S1
def test_s1_defaults_are_bitwise_identical() -> None:
    """全默认 vs **显式传默认值** ⇒ 逐位等价（关档不改变旧行为）。"""
    a = _run(_engine(), 50)
    b = _run(_engine(tilt_rad=0.0, season_period=0, light_sensitivity=0.0,
                     light_normalize=False), 50)
    assert _digest(a) == _digest(b), (
        f"显式默认值改变了行为：{_digest(a)} vs {_digest(b)} ⇒ 接线在默认档**不等价**")


def test_s1b_matches_c7_baseline() -> None:
    """默认档 = 仓库钉死的 C7 基线（防"孤立等价"）。"""
    assert _digest(_run(_engine(), 50)) == (574887, 11266.746993), (
        f"C7 基线漂移：{_digest(_run(_engine(), 50))} ≠ (574887, 11266.746993)"
        " ⇒ 13.7 接线动了默认档")


# --------------------------------------------------------------------------- S3
def test_s3_season_flips_hemispheric_asymmetry() -> None:
    """`tilt_rad`/`season_period` **真生效**：日累积光照的南北不对称随季节翻转。

    判据（强于"数组不同"）：
      · 关档：南北半球**近似对称**（|北−南| 小）**且不随时间变**（两时刻相同）；
      · 开档：t=P/4（北半球夏）北 > 南；t=3P/4（南半球夏）南 > 北 ⇒ **符号翻转**。
    """
    off = _engine()
    n0, s0 = _hemi_means(off, 0)
    n1, s1 = _hemi_means(off, SEASON // 4)
    assert abs(n0 - s0) < 0.02, f"关档时南北应近似对称，实得 北{n0:.4f} 南{s0:.4f}"
    assert abs(n0 - n1) < 1e-12 and abs(s0 - s1) < 1e-12, (
        "关档时日照不应随 tick 变（季节未开）")

    on = _engine(tilt_rad=np.deg2rad(23.44), season_period=SEASON)
    nA, sA = _hemi_means(on, SEASON // 4)            # 北半球夏
    nB, sB = _hemi_means(on, 3 * SEASON // 4)        # 南半球夏
    assert nA > sA, f"t=P/4 应北亮：北{nA:.4f} 南{sA:.4f}"
    assert sB > nB, f"t=3P/4 应南亮：北{nB:.4f} 南{sB:.4f}"
    asym_A, asym_B = nA - sA, nB - sB
    assert asym_A > 0.02 and asym_B < -0.02, (
        f"季节不对称幅度过小：t=P/4 Δ{asym_A:+.4f}｜t=3P/4 Δ{asym_B:+.4f}"
        "（若此处失败 ⇒ tilt_rad 未真正影响光照）")


def test_s3b_season_period_controls_the_cycle() -> None:
    """`season_period` **真生效**：周期不同 ⇒ 同一 tick 的光照不同。

    取 t = 半个**长周期**：长周期 P1=8·ROT 在 t=4·ROT 处刚到 1/4；短周期
    P2=4·ROT 在 t=4·ROT 处已回到起点（整周期）⇒ 两者日照必然不同。
    """
    a = _engine(tilt_rad=np.deg2rad(23.44), season_period=8 * ROT)
    b = _engine(tilt_rad=np.deg2rad(23.44), season_period=4 * ROT)
    na, sa = _hemi_means(a, 4 * ROT)
    nb, sb = _hemi_means(b, 4 * ROT)
    assert abs((na - sa) - (nb - sb)) > 1e-6, (
        f"season_period 未生效：ΔA={na - sa:+.6f} 与 ΔB={nb - sb:+.6f} 相同")


# --------------------------------------------------------------------------- S4
def test_s4_light_sensitivity_changes_regrowth_field() -> None:
    """`light_sensitivity` **真生效**：再生量场异于关档（零机时，直调内部函数）。"""
    flat = np.arange(7200)
    off = _engine().resources._regrowth_amount(0)
    on = _engine(tilt_rad=np.deg2rad(23.44), season_period=SEASON,
                 light_sensitivity=1.0).resources._regrowth_amount(0)
    assert off.shape == on.shape == (7200,)
    assert not np.allclose(off, on), (
        "light_sensitivity=1.0 未改变再生量场 ⇒ 该字段是**死字段**（no-op）")
    assert float(on.mean()) < float(off.mean()), (
        "光照因子 ≤1 ⇒ 未归一化时全球平均再生量应**下降**："
        f"开档 {on.mean():.6g} vs 关档 {off.mean():.6g}")


# --------------------------------------------------------------------------- S5
def test_s5_light_normalize_preserves_global_total() -> None:
    """`light_normalize` 的**保总量语义**：开启后全球平均再生量显著更接近关档。"""
    tick = SEASON // 4
    off = _engine().resources._regrowth_amount(tick)
    raw = _engine(tilt_rad=np.deg2rad(23.44), season_period=SEASON,
                  light_sensitivity=1.0,
                  light_normalize=False).resources._regrowth_amount(tick)
    nor = _engine(tilt_rad=np.deg2rad(23.44), season_period=SEASON,
                  light_sensitivity=1.0,
                  light_normalize=True).resources._regrowth_amount(tick)
    d_raw = abs(float(raw.mean()) - float(off.mean()))
    d_nor = abs(float(nor.mean()) - float(off.mean()))
    assert d_nor < d_raw, (
        f"normalize 未收紧全球总量：|归一−关档|={d_nor:.6g} ≥ |未归一−关档|={d_raw:.6g}")
    assert d_nor / max(float(off.mean()), 1e-12) < 0.15, (
        f"normalize 后全球平均再生量仍偏离关档 >15%："
        f"{float(nor.mean()):.6g} vs {float(off.mean()):.6g}")
    # 且它**只**改空间分布：逐格仍与关档不同
    assert not np.allclose(nor, off), "normalize 把再生量场变回了关档 ⇒ 分布未改变"


# --------------------------------------------------------------------------- S6
def test_s6_h3_fail_loud_on_season() -> None:
    """H3：开季节 + `use_sim_core=True` ⇒ **构造期硬报错**（禁静默走无季节光照）。"""
    with pytest.raises(NotImplementedError) as ei:
        _engine(tilt_rad=np.deg2rad(23.44), season_period=SEASON, use_sim_core=True)
    assert "season" in str(ei.value).lower(), f"报错信息未指明 season：{ei.value}"


def test_s6b_h3_fail_loud_on_light_sensitivity() -> None:
    """H3：开光驱动再生 + `use_sim_core=True` ⇒ **构造期硬报错**。"""
    with pytest.raises(NotImplementedError) as ei:
        _engine(light_sensitivity=1.0, use_sim_core=True)
    assert "light_sensitivity" in str(ei.value), f"报错信息未指明字段：{ei.value}"


# --------------------------------------------------------------------------- S7
def test_s7_config_rejects_half_given_season() -> None:
    """配置自洽：`tilt_rad` 与 `season_period` 必须**同时**给（否则语义不明）。

    ⚠️ 该断言在 **`LightConfig.__post_init__`**（配置层），不在
    `LightAndTemperature`（运行层）—— 初版写错了层，已修。
    """
    from simulation.config import LightConfig
    with pytest.raises(ValueError):
        LightConfig(tilt_rad=np.deg2rad(23.44), season_period=0)
    with pytest.raises(ValueError):
        LightConfig(tilt_rad=0.0, season_period=SEASON)
    # 同时给 ⇒ 合法
    ok = LightConfig(tilt_rad=np.deg2rad(23.44), season_period=SEASON)
    assert ok.season_period == SEASON


# --------------------------------------------------------------------------- S8
def test_s8_mutation_sensitivity_open_vs_off_digest_differs() -> None:
    """🔴 **DEL-8 变异测试（机械化）**：每个接线**各自**都必须可被检测。

    🔴 **为什么拆成三条（本轮实测教训）**：初版只测"两个机制同时开"。
    我实际注入变异（把 `light_sensitivity` 的透传改成写死 `0.0`）后发现
    **它不变红** —— 因为 `tilt_rad` 仍透传 ⇒ 季节仍在生效 ⇒ digest 仍不同
    ⇒ **只掩盖在"另一个机制还活着"里**。⇒ 必须**逐机制**各测一条。

    三条断言的语义：
      S8a **只开季节**（λ=0）≠ 关档 ⇒ 守护 `tilt_rad`/`season_period` 的透传；
      S8b **只开光驱动再生**（无季节）≠ 关档 ⇒ 守护 `light_sensitivity`/`light_normalize` 的透传；
      S8c 两者同开 ≠ 关档 ⇒ 守护组合路径。

    ⇒ 任一条透传被删（= 13.5 `bg_production_zero` 家族那种"少一行接线 ⇒ 死字段"），
      对应那条立刻变红。这是把"改坏接线 ⇒ 测试变红"从人工流程变成**常驻断言**。
    """
    off = _digest(_run(_engine(), 50))

    season_only = _digest(_run(_engine(tilt_rad=np.deg2rad(23.44),
                                       season_period=SEASON), 50))
    assert season_only != off, (
        f"**只开季节**时 digest 与关档相同（{off}）⇒ `tilt_rad`/`season_period` "
        "**未接进引擎**（no-op）：检查 `SphereEngine.__init__` 对 `LightAndTemperature` 的透传")

    light_only = _digest(_run(_engine(light_sensitivity=1.0,
                                      light_normalize=False), 50))
    assert light_only != off, (
        f"**只开光驱动再生**时 digest 与关档相同（{off}）⇒ `light_sensitivity` "
        "**未接进引擎**（no-op）：检查 `SphereEngine.__init__` 对 `ResourceField` 的透传")

    both = _digest(_run(_engine(tilt_rad=np.deg2rad(23.44), season_period=SEASON,
                                light_sensitivity=1.0), 50))
    assert both != off, (
        f"**两者同开**时 digest 与关档相同（{off}）⇒ 季节／光驱动再生均未接进引擎")
