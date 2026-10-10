"""ENERGY-CLOSE 步A：**只读能量探针**（`energy_probe`）的靶向单测。

口径来源：
  · `_archive/2026-10-10-退役团队-归档/docs-旧档/设计文档/记录-ENERGY-CLOSE步A-通道枚举与A2三账口径-20261007.md` §三/§四
    （7 通道 = 4 散逸 + 3 逃逸；🔴 探针侧必须是**独立算式**，不得从 `_ec_*` 反推）
  · `_archive/2026-10-10-退役团队-归档/docs-旧档/设计文档/记录-ENERGY-CLOSE步A-合成闭环硬零门-20261007.md` §八
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
    EP_CHANNELS, EP_IMPLEMENTED, SphereEngine,
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


def _resid_probe(e: SphereEngine, channel: str, ec_index: int,
                 ticks: int = 20, require_truncation: bool = True) -> list[tuple]:
    """逐 tick 判据（散逸腿通用）：**没死人 = 两侧一分不差**。

    🔴 硬判据两条，各抓一类假实现：
    ① `d_died == 0 ⇒ gap == 0`：**删埋点**（探针少记）⇒ 残差转负 ⇒ 当场红；
    ② `0 ≤ gap ≤ 该 tick 探针全额`：被截断的死人支出必然是当 tick 总支出的**子集**
       ⇒ 抄通道账（gap = −截断量 < 0）与离谱多记（gap > 全额）都被同一式夹住。
    「残差 ∝ 死亡数」作为**观测量**返回（每笔人均差），不当硬门——硬门要的只是
    "两侧差只可能出现在死亡 tick 且不超过该 tick 全额"，额度口径归砚，我不自裁。

    返回 [(tick, gap, d_died, gap_per_died)]；无干净 tick 或无死亡 tick ⇒ 报退化。
    """
    out, prev_ledger, prev_died = [], 0.0, 0
    clean, hit = 0, 0
    for _ in range(ticks):
        if e.extinct:
            break
        e.step()
        rows = [r for r in e.energy_probe_rows() if r["tick"] == e.tick]
        assert len(rows) == 1, f"tick {e.tick} 探针行数 {len(rows)} ≠ 1 ⇒ 落行钩子丢了"
        probe_tick = rows[0][channel]
        ledger_now = float(e._ec_global[ec_index])
        died_now = int(e._run_died)
        d_ledger, d_died = ledger_now - prev_ledger, died_now - prev_died
        gap = round(probe_tick - d_ledger, 6)
        if d_died == 0:
            assert gap == 0.0, (
                f"{channel}：tick {e.tick} 无死亡却残差 {gap} ⇒ 埋点漏记"
                "（探针少记）或两侧不同源（另有未解释项）"
            )
            clean += 1
        else:
            assert 0.0 <= gap <= round(probe_tick, 6) + 1e-9, (
                f"{channel}：tick {e.tick} 残差 {gap} 不在 [0, 该 tick 探针全额 "
                f"{probe_tick}] ⇒ 负 = 抄通道账/截断方向反了；超全额 = 多记"
            )
            if gap > 0.0:
                hit += 1
        out.append((e.tick, gap, d_died, round(gap / d_died, 6) if d_died else 0.0))
        prev_ledger, prev_died = ledger_now, died_now
    assert clean > 0, f"{channel}：{ticks} tick 全为死亡 tick ⇒ 零残差判据未生效，本测退化"
    if require_truncation:
        assert hit > 0, (
            f"{channel}：{ticks} tick 无一处死亡截断 ⇒ 判据退化（换更长档，"
            "或本腿支出按现行压缩次序根本不会被裁——那要走板帖报备，不许静默关判据）"
        )
    return out


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


def test_dis_move_default_branch_leg() -> None:
    """④ D5 位移费·**默认档**那条腿（`subpos.enabled=False ∧ l2_dash=False`）。"""
    from simulation.sphere_engine import EC_MOVE
    e = _engine(EnergyProbeConfig(enabled=True))
    assert not e.config.subpos.enabled and not e.config.simulation.l2_dash
    _resid_probe(e, "dis_move", EC_MOVE, 20)
    assert e.energy_probe_totals()["dis_move_sum"] > 0.0


def test_dis_move_l2_branch_leg() -> None:
    """④ D5 位移费·**L2 档**那条腿（`:4620`）——默认档不执行 ⇒ 必须显式开档才有覆盖。

    ⚠️ 顺带发现（**只报不定性、不改**）：`_run_flat_move_n`（主判据 `mean_flat_moves` 的分子）
    连同 `_run_mover_sub_n` / `_run_slow_n` 的 `+=` 只在 **subpos 分支**里（`:4602-4604`）
    ⇒ `subpos.enabled=False`（**含本仓默认档**）下分母 `_run_mover_sub_n = 0`
    ⇒ `:2332` 的 `mean_flat_moves` 恒 `None`。所以本测不能拿它当"这腿真走过"的守卫，
    改用「默认档对照 + 通道账增量」两条替代。判读归砚/PI。
    """
    from simulation.sphere_engine import EC_MOVE
    cfg = _base_cfg()
    cfg.simulation.l2_dash = True
    cfg.subpos.enabled = False
    cfg.energy_probe = EnergyProbeConfig(enabled=True)
    e = SphereEngine(cfg)
    assert e.config.simulation.l2_dash, "L2 档没开上 ⇒ 本测测的是默认腿，退化"
    _resid_probe(e, "dis_move", EC_MOVE, 20)
    ctl = _run(_engine(EnergyProbeConfig(enabled=True)), 20)   # 默认档（非 L2）
    a = e.energy_probe_totals()["dis_move_sum"]
    b = ctl.energy_probe_totals()["dis_move_sum"]
    assert a > 0.0 and b > 0.0, f"dis_move 两侧有零 ⇒ 档位没跑出位移支出（a={a} b={b}）"
    assert float(e._ec_global[EC_MOVE]) > 0.0, "通道账 EC_MOVE 恒零 ⇒ 位移段未执行，本测退化"
    assert a != b, "L2 档与默认档 dis_move 一分不差 ⇒ 没走到 L2 计费式，本测测的是默认腿"


def test_dis_move_subpos_branch_leg() -> None:
    """④ D5 位移费·**subdiv 档**那条腿（`:4610`，按实际步数比例计费）。"""
    from simulation.sphere_engine import EC_MOVE
    cfg = _base_cfg()
    cfg.subpos.enabled = True
    cfg.simulation.l2_dash = False
    cfg.energy_probe = EnergyProbeConfig(enabled=True)
    e = SphereEngine(cfg)
    assert e.config.subpos.enabled, "subpos 档没开上 ⇒ 本测测的是默认腿，退化"
    _resid_probe(e, "dis_move", EC_MOVE, 20)
    assert e.energy_probe_totals()["dis_move_sum"] > 0.0
    assert sum(e._steps_hist) > 0, "subdiv 档无人走步 ⇒ 该腿未执行，本测退化"


def test_dis_attack_predation_leg() -> None:
    """④ D6 捕食出手费（`:4813` 现场直记 vs `:4902` 批量账）。"""
    from simulation.sphere_engine import EC_ATTACK
    e = _engine(EnergyProbeConfig(enabled=True))
    # ⚠️ 实测（C7 档 40 tick）：D6 侧**一次死亡截断都没出现** ⇒ 本腿只跑零残差判据
    #    （删埋点仍当场红：残差转负）。截断捕获能力由 `dis_meta` 腿与 D7 争夺腿证明。
    #    🔴 不静默放宽：这条"未见截断"是**观测**，写进记录 §7.5 供砚判读量级。
    _resid_probe(e, "dis_attack", EC_ATTACK, 40, require_truncation=False)
    assert e._duel["real_attempts"] > 0, "出手数为零 ⇒ D6 腿未执行，本测退化"
    assert e.energy_probe_totals()["dis_attack_sum"] > 0.0


def test_dis_attack_contest_leg_when_contest_on() -> None:
    """④ D7 争夺出手费（`contest_enabled` **默认关** ⇒ 不开档就是假覆盖，照 §7.4 教训）。

    D7 的通道账写入是**循环外批量一次**（`:2016`），探针在扣费现场逐次记 ⇒
    把批量那行删掉也必须被本测抓到（残差转正 = 通道账少记）。
    """
    from simulation.sphere_engine import EC_ATTACK
    cfg = _base_cfg()
    cfg.corpse_wound.contest_enabled = True
    cfg.energy_probe = EnergyProbeConfig(enabled=True)
    e = SphereEngine(cfg)
    assert e.config.corpse_wound.contest_enabled
    _resid_probe(e, "dis_attack", EC_ATTACK, 40)
    assert e._contest_n > 0, "20 tick 无一次争夺 ⇒ D7 腿未执行，本测退化（另需查档）"


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
    """⑤ 🔴 已实装腿（4/4 散逸 + E1）读数 = 实测浮点；**未埋点的逃逸腿读数必须是 None**，不是 0。

    「没测到」写成 0 = 把漏埋点伪装成"这条通道一分没漏"，是本项目反复踩的假绿。
    """
    e = _run(_engine(EnergyProbeConfig(enabled=True)), 10)
    row = e.energy_probe_rows()[-1]
    tot = e.energy_probe_totals()
    for name in EP_CHANNELS:
        if name in EP_IMPLEMENTED:
            assert isinstance(row[name], float), f"{name} 应是实测浮点"
            assert isinstance(tot[name + "_sum"], float)
        else:
            assert row[name] is None, f"{name} 未埋点却读出 {row[name]!r}"
            assert tot[name + "_sum"] is None, f"{name}_sum 未埋点却非 None"
    assert tot["path"] == "python"
    assert "未观测，不是 0" in tot["note"], tot["note"]
    # 散逸 4/4 + 逃逸 E1 = 步A 已实装面；留白必须是**显式**的（esc_pred_* 未埋点），不是默认值
    assert set(EP_IMPLEMENTED) == {
        "dis_meta", "dis_move", "dis_attack", "dis_signal", "esc_death_e"
    }
    assert all(n.startswith("esc_") for n in set(EP_CHANNELS) - set(EP_IMPLEMENTED))


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


# ------------------------------------------------------------------ 逃逸腿 E1（死亡逃逸）

def test_esc_death_e_tracks_deaths_when_corpse_off() -> None:
    """E1 · 默认档（`corpse_enabled=False`）：死者体能在压缩点**全额消失** ⇒ 逃逸 = 全额。

    两条硬判据各抓一类假实现：
    ① 无死亡 tick ⇒ 该行必须**恰好 0**（抓"无条件加"/把上 tick 余量串记）；
    ② 死亡 tick ⇒ 人均逃逸 ≤ `max_energy`（抓多记）。
    上界口径不是我新造的容差 —— 引擎自己两条扣费封顶（`sphere_engine.py:2237-2238`）：
    进食 `max_energy/eat_eff × 0.5 × cap_mult` ≤100、捕食 `max_energy/eat_eff` = 100，
    均 < `max_energy`(300) ⇒ 个体能量上界取 `max_energy` 是现行口径的**推论**。
    """
    cfg = _base_cfg()
    assert cfg.corpse_wound.corpse_enabled is False, "前提失效：默认档 corpse 不是关的"
    cfg.energy_probe = EnergyProbeConfig(enabled=True)
    e = SphereEngine(cfg)
    ceiling = float(cfg.organisms.max_energy)
    prev_died, died_seen, paid_ticks = 0, 0, 0
    for _ in range(40):
        if e.extinct:
            break
        e.step()
        rows = [r for r in e.energy_probe_rows() if r["tick"] == e.tick]
        assert len(rows) == 1, f"tick {e.tick} 探针行数 {len(rows)} ≠ 1"
        d_died = int(e._run_died) - prev_died
        prev_died += d_died
        died_seen += d_died
        esc = rows[0]["esc_death_e"]
        assert isinstance(esc, float), "E1 已实装却读出 None ⇒ 通道登记没跟上"
        if d_died == 0:
            assert esc == 0.0, f"tick {e.tick} 无死亡却逃逸 {esc} ⇒ 埋点无条件加"
        else:
            assert 0.0 <= esc <= d_died * ceiling + 1e-9, (
                f"tick {e.tick} 死亡 {d_died} 逃逸 {esc} > {d_died}×max_energy ⇒ 多记"
            )
            if esc > 0.0:
                paid_ticks += 1
    assert died_seen > 0, "40 tick 无死亡 ⇒ 本测退化（E1 从未被执行）"
    assert paid_ticks > 0, "40 tick 有死亡但逃逸全 0 ⇒ 删埋点（漏账）"
    assert e.energy_probe_totals()["esc_death_e_sum"] > 0.0


def test_esc_death_e_is_independent_recalc_not_a_copy_of_deposit() -> None:
    """E1 · 开 corpse 档：探针（独立重算）与投放侧既有计数器必须满足**恒等式**。

    `deposit = frac × ΣE_dead`（`_deposit_corpse:1821-1829`，钳制前"意图"口径）
    `esc     = (1 − frac) × ΣE_dead`（同一时刻、同一 `dead` 掩码，公式**独立写第二遍**）
    ⇒ `esc / deposit = (1 − frac)/frac`，与轨迹无关的**常数**。
    抓四类假实现：删埋点(0)、把逃逸记成全额(`1/frac` 倍)、双计(2×)、
    直接抄 `_corpse_deposited_e`(1) —— 各落在此式一处。🔴 探针不读该计数器 ⇒ 抄不了。
    """
    cfg = _base_cfg()
    cfg.corpse_wound.corpse_enabled = True
    cfg.energy_probe = EnergyProbeConfig(enabled=True)
    e = _run(SphereEngine(cfg), 40)
    frac = float(cfg.corpse_wound.corpse_energy_frac)
    esc = e.energy_probe_totals()["esc_death_e_sum"]
    dep = float(e._corpse_deposited_e)      # 投放侧**既有**读数，仅作对照
    assert dep > 0.0, "40 tick 无投放 ⇒ 本测退化"
    assert esc > 0.0, "开档有投放但探针记 0 ⇒ E1 埋点没接上"
    assert round(esc / dep, 6) == round((1.0 - frac) / frac, 6), (
        f"esc/deposit = {esc / dep}，恒等式要求 (1−frac)/frac = {(1.0 - frac) / frac}"
        " ⇒ 两侧不同源（探针不是同一时刻的独立重算）"
    )


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
