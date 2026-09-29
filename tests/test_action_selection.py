"""R239/R258 ASM 模式仲裁（`action_selection.mode="arbitration"`）—— 回归单测。

规格：`docs/设计文档/实施规格-移动决策三件-20260928.md` §三（3.1 配置 / 3.2 状态 /
3.3 算法 / 3.4 测试清单 / 3.5 预注册判据）+ §五 交付检查单。

覆盖（规格 §3.4 七例 → 本文档编号）：
    ① 默认 `fusion`：C7 钉死 digest `(574887, 11266.746993)` 不动（关档零接触）；
    ② 迟滞：微超（< hyst）不切、越阈（> hyst）切、**边界严格 `>`**；
    ③ 锁定：`hold_ticks` 内必不切换（函数级 + 引擎 `hold_block_n` 可观测）；
    ④ **dithering 反例（头号风险）**：交替最优输入 ⇒ 切换率 ≤ 1/hold_ticks
       （函数级**精确上界** + 引擎级 `sw_n ≤ 个体-tick/hold + N₀`）；
    ⑤ 排他性：flee 档**不被食物吸引**（"食物近、风险更近"构造：同场景 feed 去食、
       flee 去最低风险格 ⇒ 目标不同）；
    ⑥ **变异检查（机械化）**：`hold_ticks=0` ⇒ 例 ④ 的上界必被打破（红）；
    ⑦ 接线：四模式 salience 各自可被引擎观测（feed/flee/join 切换到；explore 兜底
       + 平局随机分支）；
    ⑧ 双路径：Python vs Rust 逐位一致；快路径 `_move_decide_batch` 仍启用且 lockstep；
    ⑨ 快照 roundtrip：`_mode`/`_mode_tick` 逐位一致 + 繁殖/死亡后长度锁步；
    ⑩ fail-loud：缺通道 / 互斥机制 / 拼错模式必须炸；
    ⑪ 配置四件套：往返 / 缺键回退 / 未知键忽略 / 断言 / 进 fingerprint；
    ⑫ `SmellField.channel_norm` 单通道归一读（与 `combined()` 同式、截断计数不静默）。

🔴 规格 §3.3 红线（本文件也守）：**首批不把信号接到模式**；**HM ② 在仲裁档下关闭**。
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from simulation.config import (
    ActionSelectionConfig, InfoStructureConfig, SimConfig,
)
from simulation.genes import Gene
from simulation.sphere_engine import SphereEngine, _asm_wta_select
from world.smell_field import SmellField

_COLS = 120


# --------------------------------------------------------------------------- 工具

def _engine(*, seed: int = 42, n: int = 200, asm: bool = True,
            channels=("food", "risk", "kin"), use_sim_core: bool = False,
            subpos: bool = False, max_count: int | None = None) -> SphereEngine:
    cfg = SimConfig(seed=seed)
    cfg.simulation.use_sim_core = use_sim_core
    cfg.population.initial_count = n
    if max_count is not None:
        cfg.population.max_count = max_count
    if subpos:
        cfg.subpos.enabled = True
    cfg.smell = replace(cfg.smell, channels=tuple(channels))
    cfg.action_selection = ActionSelectionConfig(
        mode="arbitration" if asm else "fusion")
    return SphereEngine(cfg)


def _run(e: SphereEngine, ticks: int = 50) -> SphereEngine:
    for _ in range(ticks):
        if e._extinct:
            break
        e.step()
    return e


def _digest(e: SphereEngine) -> tuple[int, float]:
    return (int(e._flat.sum()), round(float(e._energy.sum()), 6))


@pytest.fixture
def no_field_update(monkeypatch):
    """冻结气味场更新（构造题场景专用；被删掉的只是**更新**，读/消费路径照走）。"""
    monkeypatch.setattr(SmellField, "update", lambda self, *a, **k: None)


def _set_field(e: SphereEngine, *, food=None, risk=None, kin=None) -> None:
    """直接写 `_S`（`{格: 值}`）并**令按代缓存失效**（否则读到上一代的 0）。"""
    e.smell._S[:] = 0.0
    e.smell._Sc[:] = 0.0
    for ch, vals in (("food", food), ("risk", risk), ("kin", kin)):
        if not vals:
            continue
        ci = e.smell._idx[ch]
        for cell, v in vals.items():
            e.smell._S[ci, int(cell)] = float(v)
    e._smell_stamp = np.full(e.world.n_cells, -1, dtype=np.int32)
    e._smell_gen = 5


def _single(*, own_food: float = 0.0, own_risk: float = 0.0, own_kin: float = 0.0,
            g13: float = 0.5, energy_frac: float = 1.0, mode: int = 3,
            tick_lock: bool = False, seed: int = 5) -> SphereEngine:
    """单个体引擎：钉死"必走"，模式/tick 预置（`tick_lock` ⇒ 本 tick 必不切换）。"""
    e = _engine(seed=seed, n=1, max_count=40)
    cell = 30 * _COLS + 60
    e._flat[0] = cell
    e._energy[0] = float(e.config.organisms.max_energy) * energy_frac
    e._genes[0, Gene.MOVE_PROB] = 1.0
    e._genes[0, Gene.ROOTING] = 0.0
    e._genes[0, Gene.AGGRESSION] = 0.0     # 自身不注入 risk（构题纯化）
    e._genes[0, Gene.SOCIABILITY] = g13
    _set_field(e, food={cell: own_food}, risk={cell: own_risk}, kin={cell: own_kin})
    e._mode[0] = mode
    e._mode_tick[0] = e._tick if tick_lock else -10**9
    return e


def _neighbors(e: SphereEngine, cell: int) -> np.ndarray:
    nb = np.asarray(e.world.neighbors(cell), dtype=np.int64)
    assert (nb >= 0).all(), "构造题要求内部格（8 邻全有效）"
    return nb


# --------------------------------------------------------------------------- ①
def test_default_fusion_c7_baseline_unchanged() -> None:
    """① 默认 `fusion` ⇒ 与钉死的 C7 基线逐位一致（关档零接触，回滚点）。"""
    cfg = SimConfig(seed=42)
    cfg.simulation.use_sim_core = False
    cfg.simulation.l2_dash = False
    cfg.population.max_count = 600
    cfg.predation.forage_tradeoff_k = 0.0
    cfg.info_structure = InfoStructureConfig(
        enabled=True, learning_rate=0.05, memory_gradient="none")
    cfg.organisms.young_mob_mult = 0.55
    cfg.organisms.old_mob_mult = 0.55
    cfg.organisms.dash_min_energy_frac = 0.2
    cfg.organisms.far_cap = 32
    assert cfg.action_selection.mode == "fusion", "默认必须是 fusion（回滚点）"
    e = _run(SphereEngine(cfg), 50)
    assert _digest(e) == (574887, 11266.746993), (
        f"C7 基线漂移：{_digest(e)} ⇒ ASM 关档动了默认路径")
    assert e.asm_probe() is None, "fusion 档读数必须是 None（未适用），不是 0"
    assert e._asm_on is False


def test_default_off_then_on_changes_trajectory() -> None:
    """⑥ 的一环：同一环境开档 ⇒ digest 变化（机制真接线，不是字段存在）。"""
    off = _digest(_run(_engine(seed=11, n=120, asm=False), 40))
    on = _digest(_run(_engine(seed=11, n=120, asm=True), 40))
    assert off != on, "仲裁档与融合档 digest 相同 ⇒ 机制未接线（静默 no-op）"


# --------------------------------------------------------------------------- ②
def _sal(*pairs) -> np.ndarray:
    s = np.zeros((4, 1), dtype=np.float64)
    for mode, v in pairs:
        s[mode, 0] = v
    return s


def test_hysteresis_marginal_no_switch() -> None:
    """② 微超（0.125 < hyst 0.25）⇒ **不切**（迟滞挡住）。数值取二进制精确值，避开浮点尾差。"""
    mode = np.array([0], dtype=np.int8)
    mt = np.array([-10**9], dtype=np.int32)
    sal = _sal((0, 0.25), (1, 0.375), (3, 0.2))
    sw, hb = _asm_wta_select(sal, mode, mt, 0, 0.25, 20)
    assert not sw[0] and not hb[0], "微超阈值却切换了 ⇒ 迟滞未生效"
    assert int(mode[0]) == 0


def test_hysteresis_over_threshold_switches() -> None:
    """② 越阈（0.5 > hyst 0.25）⇒ 切到 argmax（且 `_mode_tick` 刷新）。"""
    mode = np.array([0], dtype=np.int8)
    mt = np.array([-10**9], dtype=np.int32)
    sal = _sal((0, 0.25), (1, 0.75), (3, 0.2))
    sw, _ = _asm_wta_select(sal, mode, mt, 7, 0.25, 20)
    assert sw[0] and int(mode[0]) == 1
    assert int(mt[0]) == 7, "切换必须刷新 `_mode_tick`（否则锁定计时错乱）"


def test_hysteresis_boundary_is_strict() -> None:
    """② 边界：`best − cur == hyst` **不是**"越阈"（判据是严格 `>`）⇒ 不切。"""
    mode = np.array([0], dtype=np.int8)
    mt = np.array([-10**9], dtype=np.int32)
    sal = _sal((0, 0.25), (1, 0.50), (3, 0.2))
    sw, _ = _asm_wta_select(sal, mode, mt, 0, 0.25, 20)
    assert not sw[0], "`== hyst` 被判成越阈 ⇒ 判据符号被改（须严格 >）"


def test_argmax_tie_takes_first_mode() -> None:
    """确定性：两模式同显著度 ⇒ `argmax` 取**首个**（不许引入随机）。"""
    mode = np.array([3], dtype=np.int8)
    mt = np.array([-10**9], dtype=np.int32)
    sal = _sal((1, 0.80), (2, 0.80), (3, 0.2))
    sw, _ = _asm_wta_select(sal, mode, mt, 0, 0.15, 20)
    assert sw[0] and int(mode[0]) == 1


# --------------------------------------------------------------------------- ③
def test_lock_blocks_switch_even_with_large_gap() -> None:
    """③ 锁定：`tick − mode_tick < hold` ⇒ 必不切（显著度差再大也一样）。"""
    mode = np.array([0], dtype=np.int8)
    mt = np.array([100], dtype=np.int32)
    sal = _sal((0, 0.0), (1, 1.0), (3, 0.2))
    sw, hb = _asm_wta_select(sal, mode, mt, 105, 0.15, 20)
    assert not sw[0] and int(mode[0]) == 0, "锁定期内切换了 ⇒ 最小锁定失守"
    assert hb[0], "被锁定挡下的'本应切换'必须计数（`hold_block_n` 读数口径）"


def test_lock_expires_exactly_at_hold() -> None:
    """③ 边界：`tick − mode_tick == hold` ⇒ **允许**切换（判据是 ≥）。"""
    mode = np.array([0], dtype=np.int8)
    mt = np.array([100], dtype=np.int32)
    sal = _sal((0, 0.0), (1, 1.0), (3, 0.2))
    sw, hb = _asm_wta_select(sal, mode, mt, 120, 0.15, 20)
    assert sw[0] and int(mode[0]) == 1 and not hb[0]
    assert int(mt[0]) == 120


def test_empty_population_noop() -> None:
    """P=0（灭绝后）⇒ 安全空转，不抛错、不改状态。"""
    mode = np.zeros(0, dtype=np.int8)
    mt = np.zeros(0, dtype=np.int32)
    sal = np.zeros((4, 0), dtype=np.float64)
    sw, hb = _asm_wta_select(sal, mode, mt, 3, 0.15, 20)
    assert sw.size == 0 and hb.size == 0


# --------------------------------------------------------------------------- ④
def _dither_switches(hyst: float, hold: int, ticks: int = 400) -> int:
    """两模式**交替最优**（幅度 1.0 ≫ hyst）⇒ 返回累计切换数（单个体）。"""
    mode = np.array([3], dtype=np.int8)
    mt = np.array([-10**9], dtype=np.int32)
    n_sw = 0
    for t in range(ticks):
        sal = _sal((0 if t % 2 == 0 else 1, 1.0), (3, 0.1))
        sw, _ = _asm_wta_select(sal, mode, mt, t, hyst, hold)
        n_sw += int(sw[0])
    return n_sw


def test_dithering_switch_rate_bounded_by_hold() -> None:
    """④ **头号风险**：交替最优输入下，切换次数 ≤ `ticks/hold + 1`（每两次切换必隔 ≥hold）。"""
    ticks, hold = 400, 20
    n_sw = _dither_switches(0.15, hold, ticks)
    assert n_sw <= ticks // hold + 1, (
        f"抖动率 {n_sw}/{ticks} 超过 1/hold={1/hold} 上界 ⇒ 最小锁定未生效")
    assert n_sw > 0, "一次都没切 ⇒ 构造题失效（本测试空过）"


def test_mutation_hyst_and_hold_zero_makes_dithering_red() -> None:
    """⑥ 变异检查（机械化）：`hyst=0 ∧ hold_ticks=0` ⇒ ④ 的上界**必被打破**（红）。

    ⇒ 例 ④ 不是"恒真"的断言：它**依赖** `hyst`/`hold` 两条接线都真在起作用。
    """
    ticks, hold = 400, 20
    base = _dither_switches(0.15, hold, ticks)
    mut = _dither_switches(0.0, 0, ticks)
    bound = ticks // hold + 1
    assert mut > bound, (
        f"变异档（hyst=0/hold=0）切换 {mut} 次仍在界 {bound} 内 ⇒ ④ 空过（无判别力）")
    assert mut > 10 * base, (
        f"变异档仅 {mut} 次 vs 正常 {base} 次 ⇒ 锁定/迟滞未真正参与")


def test_engine_level_dithering_rate_bounded() -> None:
    """④ 引擎级：`sw_n ≤ 个体-tick总数/hold + N₀`（无出生 ⇒ 每个体至多多切一次）。

    构造：`max_count = initial_count` ⇒ 繁殖恒被硬顶挡住 ⇒ 个体总数恒 = N₀。
    """
    e = _engine(seed=3, n=120, max_count=120)
    _run(e, 300)
    p = e.asm_probe()
    assert p is not None
    total = int(sum(p["mode_n"]))
    n0 = 120
    bound = total // p["hold_ticks"] + n0
    assert p["sw_n"] <= bound, (
        f"引擎级抖动：sw_n={p['sw_n']} > 上界 {bound}（= 个体-tick/hold + N₀）"
        f" ⇒ 锁定未生效或 `mode_tick` 未随切换刷新")
    assert p["sw_n"] > 0, "从未切换 ⇒ 本测试空过"


# --------------------------------------------------------------------------- ⑤
def test_flee_mode_not_attracted_to_food(no_field_update) -> None:
    """⑤ 排他性：同场景下 feed 档去食物、flee 档去**最低风险**格（≠ 食物格）。

    构造（唯一最大食物格 = `nbs[3]`，其风险**最高** 0.9；八邻风险两两不同 ⇒ 唯一 argmin）：
      * feed  ⇒ argmax Ŝ_food = `nbs[3]`（食物格）；
      * flee  ⇒ argmin Ŝ_risk = `nbs[4]`（**不是**食物格）。
    ⇒ 若仲裁档下 flee 仍被食物吸引（= 融合语义残留），本断言必红。
    """
    risks = {0: 0.9, 1: 0.7, 2: 0.5, 3: 0.3, 4: 0.1, 5: 0.2, 6: 0.4, 7: 0.6}
    targets = {}
    for mode in (0, 1):
        e = _single(mode=mode)
        cell = int(e._flat[0])
        nbs = _neighbors(e, cell)
        risk_map = {int(nb): risks[k] for k, nb in enumerate(nbs)}
        food_cell = int(nbs[3])
        _set_field(e, food={food_cell: 2.0}, risk=risk_map)
        e._mode[0] = mode
        e._mode_tick[0] = e._tick          # 锁定本 tick（模式绝对不切）
        e.step()
        assert int(e._mode[0]) == mode, "锁定期内模式被切 ⇒ 构题失效"
        targets[mode] = int(e._flat[0])
        if mode == 0:
            assert targets[0] == food_cell, "feed 档未去食物格 ⇒ 模式内目标没接线"
        else:
            assert targets[1] == min(risk_map, key=risk_map.get), (
                f"flee 档目标 {targets[1]} 不是最低风险格 ⇒ argmin Ŝ_risk 未生效")
            assert targets[1] != food_cell, (
                "flee 档仍被食物吸走（食物格风险最高却去了那里）⇒ 融合语义残留")
    assert targets[0] != targets[1], "两模式目标相同 ⇒ 模式内目标未接线"


# --------------------------------------------------------------------------- ⑦
@pytest.mark.parametrize(
    "kw, want",
    [
        (dict(own_food=2.0), 0),                                   # feed 胜出
        (dict(own_risk=2.0), 1),                                   # flee 胜出
        (dict(own_kin=2.0, g13=1.0), 2),                           # join 胜出（g13>0.5）
        (dict(own_kin=2.0, g13=0.0), 3),                           # g13<0.5 ⇒ sal_join<0 ⇒ explore
        (dict(), 3),                                               # 无源 ⇒ explore 兜底
    ],
)
def test_salience_wiring_engine_observed(no_field_update, kw, want) -> None:
    """⑦ 四模式 salience 各自经**引擎接线**可观测（`w_*`/`base_explore`/`g13` 真的进了公式）。"""
    e = _single(**kw)
    e.step()
    assert int(e._mode[0]) == want, (
        f"salience 选模结果 {int(e._mode[0])} ≠ 期望 {want}（kw={kw}）⇒ 对应接线失守")


def test_hunger_channel_enters_feed_salience(no_field_update) -> None:
    """⑦ `w_hunger` 通道：无气味但极饿 ⇒ 仍切 feed（HM ② 让位后的饥饿通道真接线）。

    能量取 `max×0.05`（hunger=0.95 ⇒ `sal_feed ≈ 0.475 > 0.2 + hyst`）——**不能取 0**
    （能量 0 ⇒ 首 tick 饿死压缩 ⇒ `_mode` 长度归零，读不到读数）。
    """
    e = _single(energy_frac=0.05)
    e.step()
    assert int(e._mode[0]) == 0, "饥饿项未进 `sal_feed` ⇒ `w_hunger` 是死参数"


def test_explore_mode_uses_tie_random_branch(no_field_update) -> None:
    """⑦ explore 档目标函数全 0 ⇒ **平局分支**（随机），且仍落在合法邻格内。"""
    e = _single()
    cell = int(e._flat[0])
    nbs = set(int(x) for x in _neighbors(e, cell))
    e.step()
    assert int(e._mode[0]) == 3
    assert int(e._flat[0]) in nbs, "explore 档未走平局随机分支（或跳到非法格）"


def test_hm2_closed_under_arbitration() -> None:
    """🔴 规格 §3.3：仲裁档下 **HM ② 无行为效应**（perc 不被消费 ⇒ 由 `w_hunger` 承担）。

    观测：`alpha=0`（HM ① = 恒等，走停不动）+ `beta=0.8`（HM ② 开）⇒ digest 与
    HM 全关**逐位相同**。若哪天 HM ② 被接回仲裁打分（双重调制），本断言变红。
    """
    base = _digest(_run(_engine(seed=9, n=120), 40))
    e = _engine(seed=9, n=120)
    e.config.hunger_mod.enabled = True
    e.config.hunger_mod.alpha = 0.0
    e.config.hunger_mod.beta = 0.8
    assert e._asm_on is True
    on = _digest(_run(e, 40))
    assert on == base, (
        f"仲裁档下 HM ② 仍在起作用（{on} ≠ {base}）⇒ 违反规格 §3.3（双重调制）")

    # 反向控制：**融合档**下同一 beta ⇒ digest 必须变（证明本测试有判别力）
    f_base = _digest(_run(_engine(seed=9, n=120, asm=False), 40))
    f = _engine(seed=9, n=120, asm=False)
    f.config.hunger_mod.enabled = True
    f.config.hunger_mod.alpha = 0.0
    f.config.hunger_mod.beta = 0.8
    assert _digest(_run(f, 40)) != f_base, "融合档下 HM ② 竟无效应 ⇒ 对照失效"


# --------------------------------------------------------------------------- ⑧
def test_dual_path_bitwise_identical_arbitration() -> None:
    """⑧ Python(use_sim_core=False) vs Rust(True)：仲裁开档 50 tick 逐位一致。"""
    py = _run(_engine(seed=7, n=150, use_sim_core=False), 50)
    rs = _run(_engine(seed=7, n=150, use_sim_core=True), 50)
    assert py._asm_on and rs._asm_on
    assert _digest(py) == _digest(rs), (
        f"仲裁档双路径分岔：py={_digest(py)} rust={_digest(rs)}")
    assert np.array_equal(py._flat, rs._flat), "仲裁档双路径 `_flat` 分岔"
    assert np.array_equal(py._mode, rs._mode), "仲裁档双路径 `_mode` 分岔"


@pytest.mark.parametrize("subpos", [False, True])
def test_batch_fast_path_lockstep_with_asm(subpos: bool) -> None:
    """⑧ 仲裁开档**不失**快路径，且与参考循环逐 tick 逐位一致（vanilla + subpos）。"""
    kw = dict(seed=13, n=150, subpos=subpos)
    e_on = _engine(**kw)
    e_off = _engine(**kw)
    e_off._batch_move_on = False
    calls = [0]
    orig = SphereEngine._move_decide_batch

    def counting(self, *a, **k):
        calls[0] += 1
        return orig(self, *a, **k)

    SphereEngine._move_decide_batch = counting
    try:
        for t in range(60):
            e_on.step()
            e_off.step()
            assert np.array_equal(e_on._flat, e_off._flat), f"t={t} _flat 分岔"
            assert np.array_equal(e_on._energy, e_off._energy), f"t={t} _energy 分岔"
            assert np.array_equal(e_on._genes, e_off._genes), f"t={t} _genes 分岔"
            assert np.array_equal(e_on._id, e_off._id), f"t={t} _id 分岔"
            assert np.array_equal(e_on._mode, e_off._mode), f"t={t} _mode 分岔"
    finally:
        SphereEngine._move_decide_batch = orig
    assert calls[0] > 0, "仲裁开档把快路径挤掉了 ⇒ 本对拍空过（门表被误加）"


def test_batch_objective_matches_reference_for_all_modes() -> None:
    """⑧ 补：**四种模式**逐一下，快路径与参考循环同结果（防 `np.where` 分支漏档）。"""
    for mode in (0, 1, 2, 3):
        kw = dict(seed=17, n=120)
        e_on = _engine(**kw)
        e_off = _engine(**kw)
        e_off._batch_move_on = False
        for e in (e_on, e_off):
            e._mode[:] = np.int8(mode)
            # 锁死本 tick 模式（`hold > |mode_tick| + ticks` ⇒ `allow ≡ False`）：
            #   初值 `mode_tick = -10^9` ⇒ 取 `10^12` 才真锁住（取 10^9 会 Δ=10^9+t ≥ 10^9 ⇒ 放行）
            e._asm_hold = 10**12
        for t in range(30):
            e_on.step()
            e_off.step()
            assert np.array_equal(e_on._flat, e_off._flat), f"mode={mode} t={t} 分岔"
        assert int(e_on._mode[0]) == mode and int(e_off._mode[0]) == mode


# --------------------------------------------------------------------------- ⑨
def test_snapshot_roundtrip_mode_state(tmp_path) -> None:
    """⑨ 快照：`_mode`/`_mode_tick` 逐位往返 + 数组长度与 `_flat` 锁步。"""
    e = _run(_engine(seed=4, n=150, max_count=400), 120)
    assert e._asm_mode_n.sum() > 0
    p = tmp_path / "asm_snap.npz"
    e.save_snapshot(str(p))
    e2 = SphereEngine.load_snapshot(str(p))
    assert np.array_equal(e._mode, e2._mode), "`_mode` 未随快照往返（续跑丢模式状态）"
    assert np.array_equal(e._mode_tick, e2._mode_tick), "`_mode_tick` 未随快照往返"
    P = len(e._flat)
    assert e._mode.shape[0] == P and e._mode_tick.shape[0] == P
    # 续跑 20 tick：双引擎（原 + 恢复）逐位一致 ⇒ 快照语义完整
    e3 = _run(e, 20)
    e4 = _run(e2, 20)
    assert _digest(e3) == _digest(e4), "恢复后轨迹分岔 ⇒ 快照缺状态"
    assert np.array_equal(e3._mode, e4._mode)


def test_birth_growth_registration_lockstep() -> None:
    """⑨ 繁殖增长分支：子代追加后 `_mode`/`_mode_tick` 同步增长，子代初值 = explore(3)。"""
    e = _engine(seed=2, n=100, max_count=160)
    ocfg = e.config.organisms
    e._energy[:] = float(ocfg.max_energy)
    e._repro_cooldown[:] = 0.0
    e._genes[:, Gene.REPRO_THRESHOLD] = 0.0
    e._age[:] = (ocfg.maturity_fraction * e._lifespan(
        e._genes[:, Gene.LIFE_GENE])).astype(np.int64)   # 恰好成年（未老死）
    n0 = len(e._flat)
    e.step()
    n1 = len(e._flat)
    assert n1 > n0, "构造未触发繁殖 ⇒ 增长分支未被覆盖（本测试空过）"
    assert e._mode.shape[0] == n1 and e._mode_tick.shape[0] == n1, (
        "繁殖后 `_mode`/`_mode_tick` 未同步增长 ⇒ 下次压缩即错位")
    assert int(e._mode[n0:].min()) == 3, "子代模式初值必须是 explore(3)"


def test_death_compaction_registration_lockstep() -> None:
    """⑨ 死亡压缩分支：同一 `keep` 掩码 ⇒ 存活个体模式**不错位**（奇偶染色法）。"""
    e = _engine(seed=2, n=100, max_count=160)
    n0 = len(e._flat)
    e._mode[:] = 2                      # 全员染成 join
    e._mode_tick[:] = 12345
    kill = np.zeros(n0, dtype=bool)
    kill[::2] = True                    # 杀偶数号
    e._energy[kill] = 0.0
    e._stomach[kill] = 0.0
    e.step()
    assert len(e._flat) < n0, "构造未触发死亡 ⇒ 压缩分支未被覆盖（本测试空过）"
    assert e._mode.shape[0] == len(e._flat) == e._mode_tick.shape[0]
    assert np.all(e._mode == 2), "压缩后模式与个体错位（`keep` 掩码漏了 ASM 数组）"
    assert np.all(e._mode_tick == 12345)


# --------------------------------------------------------------------------- ⑩
def test_fail_loud_missing_smell_channels() -> None:
    """⑩ 缺 `Ŝ_food/Ŝ_risk/Ŝ_kin` ⇒ 构造期炸（不许静默降级成融合）。"""
    cfg = SimConfig(seed=1)
    cfg.smell = replace(cfg.smell, channels=("food",))
    cfg.action_selection = ActionSelectionConfig(mode="arbitration")
    with pytest.raises(NotImplementedError, match="risk"):
        SphereEngine(cfg)


def test_fail_loud_mutex_mechanisms() -> None:
    """⑩ 互斥机制（规格 §3.3 前置条件）⇒ 构造期炸：同开会**静默吞掉其一**。"""
    def _cfg(mut) -> SimConfig:
        c = SimConfig(seed=1)
        c.smell = replace(c.smell, channels=("food", "risk", "kin"))
        c.action_selection = ActionSelectionConfig(mode="arbitration")
        mut(c)
        return c

    cases = {
        "l1_seek": lambda c: setattr(c.simulation, "l1_seek", True),
        "l2_dash": lambda c: setattr(c.simulation, "l2_dash", True),
        "softmax": lambda c: setattr(
            c, "info_structure",
            InfoStructureConfig(enabled=True, softmax_tau=0.5)),
        "noise": lambda c: setattr(
            c, "info_structure",
            InfoStructureConfig(enabled=True, perception_noise=0.1)),
        "mem_grad": lambda c: setattr(
            c, "info_structure",
            InfoStructureConfig(enabled=True, memory_gradient="orientation")),
        "rep_w": lambda c: setattr(
            c, "info_structure",
            InfoStructureConfig(enabled=True, reputation_weight=0.3)),
        "use_in_move": lambda c: setattr(c.smell, "use_in_move", True),
        "span2": lambda c: setattr(c.simulation, "perception_span", 2),
        # migration 自带"无季节 ⇒ 恒 no-op ⇒ 硬报错"的前置守卫（先于 ASM 守卫）
        #   ⇒ 构题必须先把季节打开，才轮到 ASM 守卫发火（守的是**同一个静默吞并**风险）。
        "migration": lambda c: (
            setattr(c.light, "tilt_rad", 0.41),
            setattr(c.light, "season_period", 960),
            setattr(c.migration, "enabled", True)),
        "ars": lambda c: setattr(c.ars, "enabled", True),
    }
    for name, mut in cases.items():
        try:
            SphereEngine(_cfg(mut))
        except NotImplementedError as exc:
            assert "互斥" in str(exc), f"{name} 的报错口径不对：{exc}"
        else:
            pytest.fail(f"互斥机制 {name} 与仲裁同开却未炸 ⇒ 静默吞并风险")


def test_fail_loud_unknown_mode() -> None:
    """⑩ 拼错的模式**必须炸**（不许静默退回 fusion）。"""
    assert ActionSelectionConfig(mode="fusion").mode == "fusion"
    assert ActionSelectionConfig(mode="arbitration").mode == "arbitration"
    with pytest.raises(AssertionError):
        ActionSelectionConfig(mode="arbitraton")


# --------------------------------------------------------------------------- ⑪
def test_config_four_piece() -> None:
    """⑪ 配置四件套：往返 / 缺键回退 / 未知键忽略 / 断言 / 进 fingerprint。"""
    a = ActionSelectionConfig(
        mode="arbitration", base_explore=0.3, hyst=0.25, hold_ticks=7,
        w_feed=0.9, w_hunger=0.4, w_flee=0.8, w_join=0.6)
    cfg = SimConfig(seed=1)
    cfg.action_selection = a
    d = cfg.to_dict()
    assert d["action_selection"] == {
        "mode": "arbitration", "base_explore": 0.3, "hyst": 0.25, "hold_ticks": 7,
        "w_feed": 0.9, "w_hunger": 0.4, "w_flee": 0.8, "w_join": 0.6}

    back = SimConfig.from_dict(d)
    assert back.action_selection == a

    d2 = dict(d)
    d2.pop("action_selection")
    assert SimConfig.from_dict(d2).action_selection == ActionSelectionConfig(), (
        "旧存档缺 `action_selection` 键必须回退 fusion（ASM 之前的档）")

    d3 = dict(d)
    d3["action_selection"] = {**d["action_selection"], "bogus_key": 123}
    assert SimConfig.from_dict(d3).action_selection == a, "未知键必须忽略而非报错"

    c2 = SimConfig(seed=1)
    c2.action_selection = ActionSelectionConfig(mode="arbitration")
    assert cfg.fingerprint() != c2.fingerprint(), "action_selection 必须进 fingerprint"

    with pytest.raises(AssertionError):
        ActionSelectionConfig(base_explore=-0.1)
    with pytest.raises(AssertionError):
        ActionSelectionConfig(hyst=10.1)
    with pytest.raises(AssertionError):
        ActionSelectionConfig(hold_ticks=-1)
    with pytest.raises(AssertionError):
        ActionSelectionConfig(hold_ticks=1.5)
    with pytest.raises(AssertionError):
        ActionSelectionConfig(w_feed=-0.1)
    with pytest.raises(AssertionError):
        ActionSelectionConfig(w_flee=10.1)


# --------------------------------------------------------------------------- ⑫
def test_channel_norm_single_channel_read(no_field_update) -> None:
    """⑫ `channel_norm`：`clip(S/S_max)` 同式、只算被读格、截断计数进 `_clip_n`。"""
    e = _single()
    cell = int(e._flat[0])
    nbs = _neighbors(e, cell)
    s_max = float(e.smell.s_max("food"))
    _set_field(e, food={int(nbs[0]): 2.0 * s_max, int(nbs[1]): 0.5 * s_max})
    got = e.smell.channel_norm(np.asarray([int(nbs[0]), int(nbs[1])]), "food")
    assert got[0] == pytest.approx(1.0), "超上界未截断到 1.0"
    assert got[1] == pytest.approx(0.5)
    assert e.smell.probe()["hat_clip_n"] >= 1, "截断未计数（静默 ⇒ B3 家族）"


def test_channel_norm_unknown_channel_raises(no_field_update) -> None:
    """⑫ 拼错通道必须炸（不许静默返回 0）。"""
    e = _single()
    with pytest.raises(AssertionError):
        e.smell.channel_norm(np.asarray([0], dtype=np.int64), "foood")
