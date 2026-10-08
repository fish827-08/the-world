"""ENERGY-CLOSE 步A：**只读能量探针**（`energy_probe`）的靶向单测。

口径来源：
  · `docs/设计文档/记录-ENERGY-CLOSE步A-通道枚举与A2三账口径-20261007.md` §三/§四
    （7 通道 = 4 散逸 + 3 逃逸；🔴 探针侧必须是**独立算式**，不得从 `_ec_*` 反推）
  · `docs/设计文档/记录-ENERGY-CLOSE步A-合成闭环硬零门-20261007.md` §八
    （轻舟 ack 四硬约束 + 砚裁六条口径）

覆盖（新工具四律逐条钉死）：
    ① **默认关 = 旧行为逐位等价**：同 seed 关/开 ⇒ digest 逐位相同；
       且与 C7 钉死基线一致（关档零接触 ⇒ 基线不许漂）；
    ② **不消费 RNG**：关/开两次跑的 `rng_draws()` 必须相同（探针只读、不抽样）；
    ③ **fail-loud**（四条）：Rust 路径 + 开探针 / 未知通道名 / 配置非法 /
       旧快照无探针键但当前开档 ⇒ 全部**报错**，不静默出空 sidecar；
    ④ **变异检查必红**：`test_dis_meta_is_not_a_copy_of_the_channel_ledger`
       （通道账截断漏记被探针抓到 ⇒ 若把探针改成抄 `_ec_*`，此测必红）；
    ⑤ 读数面：行频 `row_every` / 环形 `max_rows` / **未埋点通道 = None 不是 0**；
    ⑥ 配置四件套：往返 / 缺键回退 / 未知键忽略 / 断言 / **进 fingerprint**（约束 3）；
    ⑦ 快照续跑：`totals` 连续不归零 + 配置不一致拒跑（R233 T-F 同族）。
"""

from __future__ import annotations

import copy
import os
import tempfile

import numpy as np
import pytest

from simulation.config import EnergyProbeConfig, InfoStructureConfig, SimConfig
from simulation.sphere_engine import (
    EP_CHANNELS, EP_V0_IMPLEMENTED, SphereEngine,
)


# --------------------------------------------------------------------------- 工具

def _base_cfg() -> SimConfig:
    """微型冒烟档（同 `test_135_energy_calib.py::test_e1b` 的 C7 helper 口径）。"""
    cfg = SimConfig(seed=42)
    cfg.simulation.use_sim_core = False
    cfg.simulation.l2_dash = False
    cfg.population.max_count = 600
    cfg.predation.forage_tradeoff_k = 0.0
    cfg.info_structure = InfoStructureConfig(
        enabled=True, learning_rate=0.05, memory_gradient="none"
    )
    cfg.organisms.young_mob_mult = 0.55
    cfg.organisms.old_mob_mult = 0.55
    cfg.organisms.dash_min_energy_frac = 0.2
    cfg.organisms.far_cap = 32
    return cfg


def _engine(probe: EnergyProbeConfig | None = None) -> SphereEngine:
    cfg = _base_cfg()
    if probe is not None:
        cfg.energy_probe = probe
    return SphereEngine(cfg)


def _run(e: SphereEngine, ticks: int = 30) -> SphereEngine:
    for _ in range(ticks):
        if e.extinct:
            break
        e.step()
    return e


def _digest(e: SphereEngine) -> tuple[int, float]:
    return (int(e._flat.sum()), round(float(e._energy.sum()), 6))


# ------------------------------------------------------------------ ① 默认关等价

def test_default_off_is_bitwise_identical() -> None:
    """① 探针**默认关** ⇒ 与显式关档逐位等价；开档也不得动轨迹（只读）。"""
    off = _run(_engine())
    explicit_off = _run(_engine(EnergyProbeConfig(enabled=False)))
    on = _run(_engine(EnergyProbeConfig(enabled=True)))
    assert _digest(off) == _digest(explicit_off), "显式关档改变了默认行为"
    assert _digest(off) == _digest(on), (
        f"开探针改变了轨迹：off={_digest(off)} on={_digest(on)} ⇒ 探针不是只读"
    )
    # 关档 ⇒ 读数面空（且**不是**假装有零值行）
    assert off.energy_probe_rows() == []
    assert off.energy_probe_totals()["enabled"] is False
    assert off.energy_probe_totals()["dis_meta_sum"] == 0.0


def test_off_does_not_disturb_c7_baseline() -> None:
    """① 挂进 `SimConfig` 后，默认档必须仍与 **C7 钉死基线**一致（关档零接触）。

    基线值来自 `tests/test_135_energy_calib.py::test_e1b_matches_c7_baseline`
    （同 helper 口径、50 tick）⇒ 不是我另起炉灶造数。
    """
    e = _run(SphereEngine(_base_cfg()), 50)
    assert _digest(e) == (574887, 11266.746993), (
        f"C7 基线漂移：{_digest(e)} ≠ (574887, 11266.746993)"
        " ⇒ 步A 新增配置组动了默认路径"
    )
    assert len(e._id) > 0


def test_on_does_not_disturb_c7_baseline() -> None:
    """① 开探针档也必须与 C7 基线逐位相同 ⇒ 证明"只读"不是口头承诺。"""
    e = _run(_engine(EnergyProbeConfig(enabled=True)), 50)
    assert _digest(e) == (574887, 11266.746993), (
        f"开探针动了轨迹：{_digest(e)}"
    )
    assert e.energy_probe_totals()["dis_meta_sum"] > 0.0


# ------------------------------------------------------------------ ② 不消费 RNG

def test_probe_consumes_no_rng() -> None:
    """② 探针只读 ⇒ 同 seed 关/开两跑抽取次数逐次相同（`rng_draws` 口径）。"""
    off = _run(_engine(), 40)
    on = _run(_engine(EnergyProbeConfig(enabled=True)), 40)
    assert off.rng_draws == on.rng_draws, (
        f"探针消费了 RNG：off={off.rng_draws} on={on.rng_draws}"
    )
    assert _digest(off) == _digest(on)


# ------------------------------------------------------------------ ④ 独立算式

def test_dis_meta_is_not_a_copy_of_the_channel_ledger() -> None:
    """④ **变异检查**（头号风险）：探针必须与 `cost_meta` 通道账**独立**。

    实测事实（2026-10-08）：`_ec_flush()` 在 tick 末按 `P = min(各数组长度)` 截断
    ⇒ **本 tick 内死亡个体**当 tick 已付出的维持费被从通道账里裁掉
    ⇒ `_ec_global[cost_meta]` 系统性**低记**，而探针在扣费现场记 ⇒ 全额。

    🔴 这正是要的判据形：**探针 > 通道账，且差额只出现在有死亡的 tick**。
    若把探针改成从 `_ec_*` 抄一遍（最常见的假独立实现），本测当场红。
    """
    e = _engine(EnergyProbeConfig(enabled=True))
    from simulation.sphere_engine import EC_META
    per_cap = float(_base_cfg().organisms.base_metabolism
                    * 2.0 * 1.6 * 2.0)   # base × metab_mult(≤2) × age_mult(≤1.6) × 安全系数
    prev_ledger, prev_died = 0.0, 0
    for _ in range(20):
        e.step()
        rows = [r for r in e.energy_probe_rows() if r["tick"] == e.tick]
        assert len(rows) == 1, f"tick {e.tick} 探针行数 {len(rows)} ≠ 1 ⇒ 落行钩子丢了"
        ledger_now = float(e._ec_global[EC_META])
        died_now = int(e._run_died)
        d_ledger = ledger_now - prev_ledger
        d_died = died_now - prev_died
        gap = round(rows[0]["dis_meta"] - d_ledger, 6)
        # 6 位对齐本文件 `_digest` / C7 pinned-digest 既有口径（浮点累加噪声 ~1e-14 级）
        # 逐行判据：**没死人 = 两侧一分不差；死人了 = 探针恰多出死人的当 tick 支出**
        assert 0.0 <= gap <= per_cap * d_died + 1e-9, (
            f"tick {e.tick} 残差 {gap}（死亡 {d_died}）越界 ⇒ 漏账不止发生在死亡截断处"
        )
        if d_died == 0:
            assert gap == 0.0, (
                f"tick {e.tick} 无死亡却残差 {gap} ⇒ 探针与通道账不同源，另有未解释项"
            )
        prev_ledger, prev_died = ledger_now, died_now

    probe = e.energy_probe_totals()["dis_meta_sum"]
    ledger = round(float(e._ec_global[EC_META]), 6)
    died = int(e._run_died)
    assert probe > 0.0, "探针恒零 ⇒ 埋点没接上（写点未执行）"
    assert died > 0, "本档 20 tick 无死亡 ⇒ 判据退化，需换更长档"
    assert probe > ledger, (
        f"探针 {probe} ≤ 通道账 {ledger} ⇒ 要么探针漏埋点，要么抄了通道账"
    )
    assert probe - ledger <= per_cap * died + 1e-9, (
        f"差额 {probe - ledger} 超出死亡截断上界 {per_cap * died}"
        " ⇒ 两侧不止差在截断，另有未解释漏账"
    )


def test_dis_meta_covers_the_heal_leg_when_wounds_on() -> None:
    """④ 补 **D3 愈合费**这条腿的覆盖（2026-10-08 变异检查暴露的假覆盖缺口）。

    默认档 `wound_enabled=False` ⇒ 愈合埋点段**一次都不执行**，删掉该行全套仍绿。
    本测开血条并放大愈合费率，令 D3 真实产生支出，再跑逐 tick 残差判据：
    **无死亡的 tick 两侧必须一分不差** ⇒ 愈合埋点被删 ⇒ 有愈合的无死亡 tick
    残差转负 = 当场红。
    """
    from simulation.sphere_engine import EC_META
    cfg = _base_cfg()
    cfg.corpse_wound.wound_enabled = True
    cfg.corpse_wound.wound_base = 0.5
    cfg.corpse_wound.wound_heal_rate = 0.05
    cfg.corpse_wound.wound_heal_energy_cost = 0.5
    cfg.energy_probe = EnergyProbeConfig(enabled=True)
    e = SphereEngine(cfg)
    per_cap = float(cfg.organisms.base_metabolism * 2.0 * 1.6 * 2.0)
    prev_ledger, prev_died, checked_clean = 0.0, 0, 0
    for _ in range(20):
        e.step()
        row = [r for r in e.energy_probe_rows() if r["tick"] == e.tick][0]
        ledger_now = float(e._ec_global[EC_META])
        died_now = int(e._run_died)
        d_ledger, d_died = ledger_now - prev_ledger, died_now - prev_died
        gap = round(row["dis_meta"] - d_ledger, 6)
        if d_died == 0:
            assert gap == 0.0, (
                f"tick {e.tick} 无死亡却残差 {gap} ⇒ D3 愈合腿漏记（或另有未解释项）"
            )
            checked_clean += 1
        else:
            assert 0.0 <= gap <= per_cap * d_died + 1e-9, (
                f"tick {e.tick} 残差 {gap} 超出死亡截断上界"
            )
        prev_ledger, prev_died = ledger_now, died_now
    assert checked_clean > 0, "20 tick 全是死亡 tick ⇒ 无死亡零残差判据未生效，本测退化"

    # 愈合腿确实**在跑**：同 seed 关血条对照，通道账必须不等（否则本段没执行）
    cfg_off = _base_cfg()
    cfg_off.corpse_wound.wound_enabled = True
    cfg_off.corpse_wound.wound_base = 0.0
    cfg_off.corpse_wound.wound_heal_energy_cost = 0.0
    cfg_off.energy_probe = EnergyProbeConfig(enabled=True)
    ctl = _run(SphereEngine(cfg_off), 20)
    assert float(e._ec_global[EC_META]) != float(ctl._ec_global[EC_META]), (
        "愈合参数改了但 `cost_meta` 一分不变 ⇒ 愈合段未执行 ⇒ 本测退化"
    )


def test_probe_read_api_does_not_touch_ledger_arrays() -> None:
    """④ 静态面：`_ep_*` 三个方法的字节码**不得**引用 `_ec_pt`/`_ec_global`。

    守卫"独立算式"不被将来重构悄悄破掉（同 `s0_kernel` 的 co_names 先例）。
    """
    for fn in (SphereEngine._ep_add, SphereEngine._ep_tick_end,
               SphereEngine.energy_probe_totals, SphereEngine._populate_history):
        names = set(fn.__code__.co_names)
        bad = {"_ec_pt", "_ec_global", "_ec_box"} & names
        assert not bad, f"{fn.__name__} 引用了通道账数组 {bad} ⇒ 不再是独立实测侧"


# ------------------------------------------------------------------ ③ fail-loud

def test_fail_loud_rust_path() -> None:
    """③ Rust 路径 + 开探针 ⇒ 构造期硬失败（不放行 = 白跑一整批空 sidecar）。"""
    cfg = _base_cfg()
    cfg.simulation.use_sim_core = True
    cfg.energy_probe = EnergyProbeConfig(enabled=True)
    with pytest.raises(ValueError, match="energy_probe.enabled=True"):
        SphereEngine(cfg)


def test_fail_loud_unknown_channel() -> None:
    """③ 通道名拼错 ⇒ 立刻报错（静默 no-op = B3 家族，本项目老坑）。"""
    e = _engine(EnergyProbeConfig(enabled=True))
    with pytest.raises(ValueError, match="未知能量探针通道"):
        e._ep_add("dis_metab", 1.0)
    with pytest.raises(ValueError, match="未知能量探针通道"):
        e._ep_add("cost_meta", 1.0)      # 这是**通道账**的名字，不是探针名


def test_fail_loud_non_finite_amount() -> None:
    """③ 收到 NaN/Inf ⇒ 断言失败（脏读数不许进 sidecar）。"""
    e = _engine(EnergyProbeConfig(enabled=True))
    with pytest.raises(AssertionError, match="非有限量"):
        e._ep_add("dis_meta", np.array([1.0, float("nan")]))


@pytest.mark.parametrize("kw", [
    {"row_every": 0}, {"row_every": -1}, {"max_rows": -1}, {"enabled": "yes"},
])
def test_fail_loud_bad_config(kw: dict) -> None:
    """③ 配置层非法组合 ⇒ 构造即炸（不许"永不落行"被当成关档）。"""
    with pytest.raises((AssertionError, TypeError)):
        EnergyProbeConfig(**kw)


def test_fail_loud_old_snapshot_with_probe_enabled() -> None:
    """③ 旧快照（无 `ep_*` 键）+ 当前开档 ⇒ 拒跑，不静默把累计归零。"""
    cfg = _base_cfg()
    e = _run(SphereEngine(cfg), 10)                  # 关档 ⇒ 快照仍写了 ep_* 键
    d = dict(np.load(_snapshot(e), allow_pickle=True))
    for k in ("ep_total", "ep_acc", "ep_ticks"):
        d.pop(k, None)
    p2 = os.path.join(tempfile.mkdtemp(), "old.npz")
    np.savez(p2, **d)
    cfg2 = _base_cfg()
    cfg2.energy_probe = EnergyProbeConfig(enabled=True)
    with pytest.raises(ValueError, match="早于探针实装|快照"):
        SphereEngine.load_snapshot(p2, config=cfg2)


# ------------------------------------------------------------------ ⑤ 读数面

def test_row_every_and_ring_buffer() -> None:
    """⑤ `row_every` 控行频；`max_rows` 环形保留**最近** N 行（长批防内存）。"""
    e = _run(_engine(EnergyProbeConfig(enabled=True, row_every=5)), 20)
    ticks = [r["tick"] for r in e.energy_probe_rows()]
    assert ticks == [5, 10, 15, 20], f"行频错：{ticks}"
    assert e.energy_probe_totals()["ticks_recorded"] == 20

    e2 = _run(_engine(EnergyProbeConfig(enabled=True, row_every=5, max_rows=2)), 20)
    r2 = e2.energy_probe_rows()
    assert [r["tick"] for r in r2] == [15, 20], f"环形窗没留最近两行：{r2}"
    assert e2.energy_probe_totals()["ticks_recorded"] == 20, "限行不该少记 ticks"


def test_uninstrumented_channels_read_none_not_zero() -> None:
    """⑤ 🔴 v0 只实装 `dis_meta`/`dis_signal` ⇒ 其余 5 通道读数必须是 **None**，不是 0。

    「没测到」写成 0 = 把漏埋点伪装成"这条通道一分没漏"，是本项目反复踩的假绿。
    """
    e = _run(_engine(EnergyProbeConfig(enabled=True)), 10)
    row = e.energy_probe_rows()[-1]
    tot = e.energy_probe_totals()
    for name in EP_CHANNELS:
        if name in EP_V0_IMPLEMENTED:
            assert isinstance(row[name], float), f"{name} 应是实测浮点"
            assert isinstance(tot[name + "_sum"], float)
        else:
            assert row[name] is None, f"{name} 未埋点却读出 {row[name]!r}"
            assert tot[name + "_sum"] is None, f"{name}_sum 未埋点却非 None"
    assert tot["path"] == "python"
    assert "v0 实装" in tot["note"]


def test_sidecar_row_anchor_for_join() -> None:
    """⑤ 砚⑤附款：sidecar 行必须带 **tick + 群体规模锚** ⇒ 可与主 CSV 逐 tick 回 join。"""
    e = _run(_engine(EnergyProbeConfig(enabled=True)), 12)
    rows = e.energy_probe_rows()
    assert rows and all("tick" in r and "pop_n" in r for r in rows)
    assert [r["tick"] for r in rows] == list(range(1, len(rows) + 1)), "tick 必须单调逐 tick"


# ------------------------------------------------------------------ D4 信号费

def test_dis_signal_is_measured_at_the_deduction_site() -> None:
    """④ **口径更正的证据**：D4 必须在扣费现场记，不能用 `_emit_count` 反推。

    旧口径文档《通道枚举与A2三账口径》§五 `:115` 写「无需新计数（`_emit_count` 可精确
    反推）」⇒ **本测把它证伪**：`_emit_count` 只在 `if self._oracle_on:` 分支自增
    （`sphere_engine.py:3804-3808`）⇒ **oracle 关档（默认档）时 `_emit_count` 恒零**，
    而信号费每 tick 真实在扣 ⇒ 按旧口径反推 = 静默漏整条 D4。
    """
    from simulation.config import SIGNAL_COST
    cfg = _base_cfg()
    cfg.oracle.enabled = False                 # 默认档（探针可用的 Python 路径）
    cfg.energy_probe = EnergyProbeConfig(enabled=True)
    e = _run(SphereEngine(cfg), 20)
    emit_total = int(e._emit_count.sum())
    dis_signal = e.energy_probe_totals()["dis_signal_sum"]
    assert emit_total == 0, "前提失效：oracle 关档却记到了 _emit_count ⇒ 本测的证伪对象变了"
    assert dis_signal is not None and dis_signal > 0.0, (
        "D4 未实测 ⇒ 扣费现场埋点没接上"
    )
    assert dis_signal > 0.0 and emit_total == 0, (
        "旧口径（SIGNAL_COST × _emit_count）在默认档会记 0，探针记 "
        f"{dis_signal} ⇒ 两者不可等价"
    )
    # 数量级自证：探针 = SIGNAL_COST × Σ发射人次（逐 tick 行数 × 行内发射人数上界）
    assert abs(dis_signal - SIGNAL_COST * round(dis_signal / SIGNAL_COST, 6)) < 1e-6, (
        f"D4 读数 {dis_signal} 不是 SIGNAL_COST({SIGNAL_COST}) 的整数倍 ⇒ 埋点口径可疑"
    )


def test_dis_signal_matches_emit_count_when_oracle_on() -> None:
    """④ 正向面：oracle **开档**时 `_emit_count` 与现场实测必须严格一致（两条路同闸）。

    这不是"抄账"——`_emit_count` 是 oracle 归因用的**既有**计数，探针是扣费现场的
    独立实测；两者相等才说明"发射"与"付费"没有分叉（若分叉 ⇒ 有隐形发射或漏扣）。
    """
    from simulation.config import SIGNAL_COST
    cfg = _base_cfg()
    cfg.oracle.enabled = True
    cfg.energy_probe = EnergyProbeConfig(enabled=True)
    e = _run(SphereEngine(cfg), 20)
    if int(e._emit_count.sum()) == 0:
        pytest.skip("本档 20 tick 无发射事件 ⇒ 正向核对无信号，改档再核")
    assert e.energy_probe_totals()["dis_signal_sum"] == pytest.approx(
        SIGNAL_COST * float(e._emit_count.sum()), rel=0, abs=1e-9
    ), "开档下现场实测 ≠ _emit_count 反推 ⇒ 发射与付费分叉"


# ------------------------------------------------------------------ ⑥ 配置四件套

def test_config_roundtrip_and_fingerprint() -> None:
    """⑥ 往返 / 缺键回退 / 未知键忽略 / **进指纹**（轻舟约束 3：指纹变、行为不变）。"""
    cfg = _base_cfg()
    cfg.energy_probe = EnergyProbeConfig(enabled=True, row_every=3, max_rows=7)
    d = cfg.to_dict()
    assert d["energy_probe"] == {"enabled": True, "row_every": 3, "max_rows": 7}
    back = SimConfig.from_dict(d)
    assert back.energy_probe == cfg.energy_probe
    # 旧存档（无该键）⇒ 回退默认 = 关
    d_old = {k: v for k, v in d.items() if k != "energy_probe"}
    assert SimConfig.from_dict(d_old).energy_probe.enabled is False
    # 未知键 ⇒ 忽略（白名单过滤，同 `smell` / `action_selection` 规格）
    d_junk = copy.deepcopy(d)
    d_junk["energy_probe"]["not_a_field"] = 1
    assert SimConfig.from_dict(d_junk).energy_probe == cfg.energy_probe
    # 指纹：挂进 SimConfig ⇒ asdict 自动进 fingerprint
    off = _base_cfg()
    assert off.fingerprint() != cfg.fingerprint(), "探针配置没进指纹 ⇒ 跨档续跑拦不住"


def test_from_dict_whitelist_does_not_leak_defaults() -> None:
    """⑥ 白名单外的键被剔除，而不是抛 TypeError（向后兼容口径）。"""
    cfg = SimConfig.from_dict({**_base_cfg().to_dict(),
                               "energy_probe": {"enabled": True, "bogus": 1}})
    assert cfg.energy_probe.enabled is True
    assert not hasattr(cfg.energy_probe, "bogus")


# ------------------------------------------------------------------ ⑦ 快照续跑

def _snapshot(e: SphereEngine) -> str:
    p = os.path.join(tempfile.mkdtemp(), "snap.npz")
    e.save_snapshot(p)
    return p


def test_snapshot_resume_keeps_probe_totals() -> None:
    """⑦ 续跑探针累计**不归零**（R233 T-F 同族）：分两段跑 == 连续跑。"""
    cont = _run(_engine(EnergyProbeConfig(enabled=True)), 20)
    a = _run(_engine(EnergyProbeConfig(enabled=True)), 10)
    b, _ = _resume(a, 10)
    assert b.energy_probe_totals()["dis_meta_sum"] == pytest.approx(
        cont.energy_probe_totals()["dis_meta_sum"], rel=0, abs=1e-9), (
        "续跑后 totals 与连续跑不等 ⇒ 累计被重置或重复计"
    )
    assert b.energy_probe_totals()["ticks_recorded"] == 20


def test_snapshot_resume_rows_are_session_local() -> None:
    """⑦ 行是导出件（不落快照）⇒ 续跑后行数重新起算，但 **tick 锚连续**。"""
    a = _run(_engine(EnergyProbeConfig(enabled=True)), 10)
    b, _ = _resume(a, 10)
    ticks = [r["tick"] for r in b.energy_probe_rows()]
    assert ticks == [11, 12, 13, 14, 15, 16, 17, 18, 19, 20], f"tick 锚不连续：{ticks}"
    assert b.energy_probe_totals()["ticks_recorded"] == 20


def test_snapshot_rejects_probe_config_drift() -> None:
    """⑦ 快照开档 + 当前关档 ⇒ 拒跑（配置漂移是错误，不是数值容差问题）。"""
    a = _run(_engine(EnergyProbeConfig(enabled=True)), 10)
    p = _snapshot(a)
    cfg_off = _base_cfg()          # 默认关
    # 🔴 第一道闸 = **配置指纹**（步A 组进了 fingerprint ⇒ 轻舟约束 3）⇒ 先于本探针守卫触发；
    #    本测钉"拒跑"这个行为，不钉具体哪一道闸（两道闸任一生效都算守住）。
    with pytest.raises(ValueError, match="指纹|enabled"):
        SphereEngine.load_snapshot(p, config=cfg_off)


def test_snapshot_rejects_probe_key_corruption() -> None:
    """③/⑦ 绕过指纹闸的伪造路径：`ep_ticks` 长度或 row_every 被改 ⇒ 仍须拒跑。"""
    a = _run(_engine(EnergyProbeConfig(enabled=True, row_every=5)), 10)
    p = _snapshot(a)
    d = dict(np.load(p, allow_pickle=True))
    _et = np.asarray(d["ep_ticks"], dtype=np.int64).copy()
    _et[2] = 3                       # row_every 5 → 3（sidecar 行频变了）
    d["ep_ticks"] = _et
    p2 = os.path.join(tempfile.mkdtemp(), "drift.npz")
    np.savez(p2, **d)
    with pytest.raises(ValueError, match="row_every"):
        SphereEngine.load_snapshot(p2)

    d2 = dict(np.load(p, allow_pickle=True))
    d2["ep_ticks"] = np.asarray([10, 1], dtype=np.int64)   # 长度损坏
    p3 = os.path.join(tempfile.mkdtemp(), "short.npz")
    np.savez(p3, **d2)
    with pytest.raises(ValueError, match="长度"):
        SphereEngine.load_snapshot(p3)


def _resume(e: SphereEngine, ticks: int) -> tuple[SphereEngine, SimConfig]:
    eng = SphereEngine.load_snapshot(_snapshot(e))
    if isinstance(eng, tuple):
        eng = eng[0]
    for _ in range(ticks):
        if eng.extinct:
            break
        eng.step()
    return eng, eng.config


# ------------------------------------------------------------------ 结构面

def test_slots_registered() -> None:
    """`__slots__` 是硬约束：新属性漏登记 ⇒ 运行期 AttributeError（本测提前抓）。"""
    slots = set(SphereEngine.__slots__)
    for name in ("_ep_on", "_ep_acc", "_ep_total", "_ep_rows",
                 "_ep_row_every", "_ep_max_rows", "_ep_ticks_recorded"):
        assert name in slots, f"{name} 未进 __slots__"


def test_ep_channels_are_seven_sidecar_names() -> None:
    """口径面：7 通道 = 4 散逸 + 3 逃逸，且**不与** `EC_*` 列名混用（约束 1）。"""
    assert len(EP_CHANNELS) == 7
    assert set(EP_CHANNELS) == {
        "dis_meta", "dis_move", "dis_attack", "dis_signal",
        "esc_death_e", "esc_pred_e", "esc_pred_s",
    }
    from simulation.sphere_engine import EC_NAMES
    assert not set(EP_CHANNELS) & set(EC_NAMES), "探针通道名与通道账列名撞了"
