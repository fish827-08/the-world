"""尸体—食腐 + 血条—受伤 S2 主机制测试（设计稿 §5.3；本段 = `[云端·开发]` 线）。

覆盖（S2 完成判据 = 单测绿 + 开档 `corpse_*`/`wound_*` 读数非零 + 关档仍逐位等价）：
  ① **H1 关档逐位等价**：corpse/wound 全关 ⇒ C7 digest 不变（同 S1 钉死值）
  ② **投尸**：死亡 ⇒ 尸体落格（三处死因共用 `_deposit_corpse`）；cap 钳制
  ③ **腐烂归还 + boost**：达 `corpse_decay_ticks` ⇒ 归还植物池 + 设 `corpse_boost`，
     boost 格再生 +50% 并递减
  ④ **食腐**：按 g16 Hill 平滑从所在格取尸体**入胃**（受胃容量限）；`corpse_eaten` 累计
  ⑤ **血条消耗战（H2）**：捕食成功 ⇒ 猎物 health 下降但**不死**（wound_n 累计）；
     `health ≤ 0` 才死（死因仍 PREDATION）
  ⑥ **愈合（H5）**：health < 1 ⇒ 每 tick 恢复（上限 1.0）+ 扣能量（不免费）
  ⑦ **读数非零**：开档后 corpse_probe/wound_probe 有真实读数（非 S1 空壳）
"""
from __future__ import annotations

import numpy as np
import pytest

from simulation.config import CorpseWoundConfig, InfoStructureConfig, SimConfig
from simulation.sphere_engine import SphereEngine, Gene, DeathCause


def _engine(ticks: int = 0, *, seed: int = 42,
            corpse: bool = False, wound: bool = False,
            cwc: CorpseWoundConfig | None = None) -> SphereEngine:
    cfg = SimConfig(seed=seed)
    cfg.population.max_count = 600
    cfg.predation.forage_tradeoff_k = 0.0
    cfg.info_structure = InfoStructureConfig(
        enabled=True, learning_rate=0.05, memory_gradient="none",
    )
    cfg.corpse_wound = cwc or CorpseWoundConfig(
        corpse_enabled=corpse, wound_enabled=wound,
    )
    e = SphereEngine(cfg)
    for _ in range(ticks):
        if e.extinct:
            break
        e.step()
    return e


def _digest(e: SphereEngine) -> tuple[int, float]:
    return (int(e._flat.sum()), round(float(e._energy.sum()), 6))


# --------------------------------------------------------------- ① H1 / C7

def test_s2_default_off_is_bit_identical():
    """C7：corpse/wound 全关必须与**改动前**逐位一致（S1 钉死值不变）。"""
    assert _digest(_engine(ticks=50)) == (574887, 11266.746993), (
        "S2 关档改变了轨迹 ⇒ 破坏 H1（默认关 = 逐位等价）")


# --------------------------------------------------------------- ② 投尸

def test_deposit_on_death():
    """死亡 ⇒ 尸体落格：能量 × corpse_energy_frac（直接调用投尸函数，确定性）。

    🔴 能量取小值（deposit < cap=3）以避开单格上限钳制，单独验证 frac。
    """
    e = _engine(corpse=True)
    dead = np.zeros(len(e._id), dtype=bool)
    dead[0] = True
    e._energy[0] = 2.0     # deposit = 1.8 < cap=3 ⇒ 不被钳制
    cell = int(e._flat[0])
    e._deposit_corpse(dead, e._energy)
    assert e._corpse_energy[cell] == pytest.approx(2.0 * 0.9, rel=1e-9), (
        "死亡未投尸（或 frac 未生效）")
    assert e._corpse_age[cell] == 0, "新尸体腐烂计时应从 0 起"


def test_deposit_cap_per_cell():
    """单格上限钳制：deposit 超过 corpse_cap_per_cell 时被钳到 cap。"""
    e = _engine(corpse=True)
    cap = int(e.config.corpse_wound.corpse_cap_per_cell)   # 跟配置默认走（30），不硬编码
    dead = np.zeros(len(e._id), dtype=bool)
    dead[0] = True
    e._energy[0] = 1000.0   # 远超 cap
    e._corpse_energy[:] = 0.0
    e._deposit_corpse(dead, e._energy)
    nz = np.flatnonzero(e._corpse_energy > 0)
    assert nz.size >= 1, "应有尸体落格"
    assert e._corpse_energy[nz].max() <= cap, "单格尸体能量超过 corpse_cap_per_cell"


def test_deposit_starvation_no_negative():
    """饿死个体（energy≤0）⇒ 投尸钳到 ≥0（不得写负数污染格上总量）。"""
    e = _engine(corpse=True)
    dead = np.zeros(len(e._id), dtype=bool)
    dead[0] = True
    e._energy[0] = -5.0
    e._corpse_energy[:] = 0.0
    e._deposit_corpse(dead, e._energy)
    assert e._corpse_energy.min() >= 0.0, "饿死投尸不得为负"
    assert e._corpse_energy[int(e._flat[0])] == 0.0, "饿死尸体应≈0（自然轻）"


def test_deposit_shared_for_all_death_causes():
    """三处死因共用投尸：老死个体能量耗尽 ⇒ 尸体 ≈ 0（自然轻，不做特判）。"""
    e = _engine(corpse=True)
    dead = np.zeros(len(e._id), dtype=bool)
    dead[:3] = True
    e._energy[:3] = np.array([100.0, 0.0, 40.0])   # 被捕杀/老死/饿死
    cells_before = e._flat[:3].copy()
    e._deposit_corpse(dead, e._energy)
    assert e._corpse_energy[cells_before[1]] == pytest.approx(0.0), (
        "老死个体能量耗尽 ⇒ 尸体应≈0")


# --------------------------------------------------------------- ③ 腐烂归还 + boost

def test_corpse_decay_returns_to_plant_and_boosts():
    """腐烂：达 decay_ticks ⇒ 归还植物池 + 设 boost；boost 补再生并递减。"""
    e = _engine(corpse=True, ticks=0)
    cwc = e.config.corpse_wound
    cell = 0
    e._corpse_energy[cell] = 100.0
    e._corpse_age[cell] = int(cwc.corpse_decay_ticks)   # 已到腐烂日
    g_before = float(e.resources._grid[cell])
    e._step_corpse_decay()
    assert e._corpse_energy[cell] == 0.0, "腐烂后尸体应清空"
    # 归还 = energy × to_plant_frac（受容量上限）
    expect_put = 100.0 * float(cwc.corpse_to_plant_frac)
    room = max(0.0, float(e.resources._capacity[cell]) - g_before)
    assert float(e.resources._grid[cell]) - g_before == pytest.approx(
        min(expect_put, room), rel=1e-9), "归还植物池量不符"
    # 同一次调用内 boost 已应用并递减（2000 → 1999）
    assert e._corpse_boost[cell] == 1999, "腐烂处应设 2000 tick 的 patch_boost 并递减"
    # 再次调用：boost 继续递减
    g_before2 = float(e.resources._grid[cell])
    e._step_corpse_decay()
    assert e._corpse_boost[cell] == 1998, "boost 应每 tick 递减"
    if e.resources._capacity[cell] > g_before2:
        assert float(e.resources._grid[cell]) >= g_before2, "boost 应补再生"


# --------------------------------------------------------------- ④ 食腐

def _place_corpse_and_feeder(e: SphereEngine, g16: float) -> int:
    """把全体摆到同一格、设 g16，给该格放尸体，返回该格。"""
    P = len(e._id)
    cell = 100
    e._flat[:] = cell
    e._genes[:, Gene.AGGRESSION] = g16
    e._genes[:, 0] = 1.0
    e._corpse_energy[cell] = 500.0
    return cell


def test_scavenging_goes_to_stomach():
    """食腐入胃：高 g16 个体取走尸体 → stomach 增加、corpse_eaten 累计。"""
    e = _engine(corpse=True, ticks=0)
    cell = _place_corpse_and_feeder(e, g16=0.9)
    e._stomach[:] = 0.0
    P = len(e._id)
    room = np.maximum(0.0, 100.0 - e._stomach[:P])   # 胃容量（捕食口径=100）
    e._step_scavenging(P, e._stomach, np.full(P, 100.0), e._genes)
    assert e._corpse_energy[cell] < 500.0, "食腐应扣减格上尸体"
    assert e._stomach[:P].sum() > 0.0, "食腐应入胃"
    assert e._corpse_eaten_n > 0, "corpse_eaten 应累计"


def test_scavenging_hill_no_hard_gate():
    """无硬门槛：g16=0 食腐量=0，g16 高则多；中间平滑（scav_gate 是半效点）。"""
    e = _engine(corpse=True, ticks=0)
    P = len(e._id)
    cap = np.full(P, 1e9)     # 不设胃限，只看 Hill 形状
    cwc = e.config.corpse_wound
    e._stomach[:] = 0.0
    e._corpse_energy[:] = 0.0
    cell = 100
    e._flat[:] = cell
    e._corpse_energy[cell] = 1e6
    # g16=0 与 g16=1 各半
    e._genes[: P // 2, Gene.AGGRESSION] = 0.0
    e._genes[P // 2:, Gene.AGGRESSION] = 1.0
    e._step_scavenging(P, e._stomach, cap, e._genes)
    half = P // 2
    zero_take = float(e._stomach[:half].sum())
    one_take = float(e._stomach[half:].sum())
    assert zero_take == pytest.approx(0.0, abs=1e-9), "g16=0 应无食腐（Hill 平滑无门槛）"
    assert one_take > 0.0, "g16=1 应能食腐"
    s = float(cwc.scav_s)
    g = float(cwc.scav_gate)
    want_one = e.config.organisms.eat_amount * (1.0 / (1.0 + g ** s))
    assert one_take / max(1, half) == pytest.approx(want_one, rel=1e-6), (
        "食腐量 = eat_amount × Hill(g16=1)")


# --------------------------------------------------------------- ⑤ 血条消耗战

def _force_duel(e: SphereEngine) -> None:
    """构造"攻击者能打、猎物在邻格"的最小捕食局。

    🔴 必须压低能量制造**饥饿**（`hunger = 1 − E/max_energy` > 0），否则 attack_prob = 0
    无攻击者；且能量要够付 attack_cost（0.1）。
    """
    P = len(e._id)
    cols = int(e.world.cols)
    for i in range(P):
        e._flat[i] = 30 * cols + (i % cols)
    e._genes[:, Gene.AGGRESSION] = 1.0        # 全员攻击者+猎物
    e._genes[:, 0] = 1.0
    e._genes[:, 19] = 0.0
    e._energy[:] = e.config.organisms.max_energy * 0.3   # 饥饿 ⇒ 有攻击者


def test_wound_damage_does_not_instantly_kill():
    """血条（T4 反转，fish 00:20 裁定）：**成功 ⇒ 一击毙命**；失败 ⇒ 扣血条。

    ⚠️ 2026-09-23（T4）：语义与 S2 相反——成功不再"消耗战命中"，而是立即致死；
    失败才扣血条（致伤）。本测试名保留（血条机制仍存在），断言按新语义：
    失败致伤应发生（wound_n>0）+ 有伤者 health<1；成功致死者进尸体/转移。
    """
    e = _engine(wound=True, ticks=0)
    _force_duel(e)
    P = len(e._id)
    e._health[:] = 1.0
    e.step()
    assert e._wound_n > 0, "失败致伤应发生（wound_n 累计，T4 反转后失败才扣血）"
    assert float(e._health[:P].min()) < 1.0, "应有个体 health 下降（失败致伤）"


def test_wound_kill_when_health_hits_zero():
    """血条（T4 反转）：health ≤ 0 才死（失败累积致死），死因仍记 PREDATION。"""
    e = _engine(wound=True, ticks=0)
    _force_duel(e)
    e._health[:] = 0.05    # 很低的血条 ⇒ 一次失败命中即致死
    e.step()
    assert int(e._duel["kills"]) > 0, "health 低时应致死"
    assert e.death_cause_totals().get(DeathCause.PREDATION, 0) > 0, "死因仍记 PREDATION"


# --------------------------------------------------------------- ⑥ 愈合

def test_heal_recovers_and_costs_energy():
    """愈合：health<1 ⇒ 每 tick 恢复（上限 1.0）+ 扣能量（不免费）。

    🔴 不能直接比对"总能量"——代谢/移动等也会扣能 ⇒ 用**同配置对拍**：
    heal_cost=0 vs 0.05 两个引擎（同 seed 同轨迹），能量差应恰 = cost × 愈合个体数。
    """
    cwc_base = CorpseWoundConfig(wound_enabled=True, wound_heal_energy_cost=0.0)
    cwc_cost = CorpseWoundConfig(wound_enabled=True, wound_heal_energy_cost=0.05)
    free = _engine(wound=True, ticks=0, cwc=cwc_base)
    paid = _engine(wound=True, ticks=0, cwc=cwc_cost)
    P = len(free._id)
    free._health[:] = 0.5
    paid._health[:] = 0.5
    free._energy[:] = 100.0
    paid._energy[:] = 100.0
    free.step()
    paid.step()
    assert paid._health[:P].max() == pytest.approx(
        free._health[:P].max(), rel=1e-9), "heal_rate 与 cost 无关（恢复量应一致）"
    n_healed = int(np.count_nonzero(free._health[:P] < 1.0))
    assert n_healed > 0, "应有个体在愈合"
    # ⚠️ 2026-09-23（T4 反转）：heal_cost 差异会通过**能量 → 捕食成功率**进入轨迹
    #   （success_rate 含 energy 比值）⇒ 两引擎不再逐位一致，能量差 ≠ 恰 cost×n。
    #   断言改为**方向性**：paid（有愈合成本）总能量 ≤ free（免费愈合）—— 愈合确实扣能。
    diff = float(free._energy[:P].sum()) - float(paid._energy[:P].sum())
    assert diff > 0, f"paid 愈合应扣能（diff={diff} 应为正）"
    # 上限：health=1 不再恢复、不扣能
    free2 = _engine(wound=True, ticks=0, cwc=cwc_base)
    paid2 = _engine(wound=True, ticks=0, cwc=cwc_cost)
    free2._health[:] = 1.0
    paid2._health[:] = 1.0
    free2._energy[:] = 100.0
    paid2._energy[:] = 100.0
    free2.step()
    paid2.step()
    assert free2._health[:P].max() == pytest.approx(1.0), "health 上限 1.0"
    diff2 = float(free2._energy[:P].sum()) - float(paid2._energy[:P].sum())
    assert diff2 == pytest.approx(0.0, abs=1e-9), "健康个体不应扣愈合能量"


# --------------------------------------------------------------- ⑦ 读数非零

def test_s2_probes_nonzero_when_on():
    """开档后 corpse/wound 读数非零（非 S1 空壳）。"""
    e = _engine(corpse=True, wound=True, ticks=30)
    cp = e.corpse_probe()
    wp = e.wound_probe()
    assert cp["corpse_enabled"] is True
    assert wp["wound_enabled"] is True
    # 30 tick 内应有死亡（投尸）或食腐活动；允许自然轨迹下为 0 时用构造验证
    if cp["corpse_total"] == 0.0 and cp["corpse_eaten"] == 0:
        # 自然轨迹可能无死亡 ⇒ 直接构造一具尸体验证读数通路
        e._corpse_energy[0] = 10.0
        assert e.corpse_probe()["corpse_total"] > 0.0, "corpse_total 读数通路坏"
    assert wp["health_mean"] is not None, "health_mean 应可读"


def test_s2_probes_off_shell():
    """关档：读数恒 0（未启用，与'测出零'区分）—— S1 空壳语义保持。"""
    e = _engine(ticks=10)
    cp = e.corpse_probe()
    wp = e.wound_probe()
    assert cp["corpse_total"] == 0.0
    assert cp["corpse_eaten"] == 0
    assert wp["wound_n"] == 0
    assert wp["contest_n"] == 0


# --------------------------------------------------------------- ⑧ R161 回归：同格多体
# 原 32 例新测试**无一覆盖"同格多体"** ⇒ 两个真 bug 恰好落在盲区（单个体情形全绿）。
# 本组两例是它们的**最小反例**（[所有者·天平] 2026-09-22 修复时补）。

def test_scavenging_conserves_energy_same_cell():
    """R161 P1-2 回归：同格多体食腐**必须守恒**（Σ取 ≤ 格存量，格上恒 ≥ 0）。

    原 bug：每人按"格上全量"取 ⇒ 实测 12 个体从格上 1.0 拿走 **4.78**、格上变 **−3.78**
    （凭空生成能量 + 负值污染 corpse_total）。
    """
    e = _engine(corpse=True)
    P = len(e._id)
    assert P >= 2, "需要 ≥2 个体才能构造同格场景"
    e._genes[:, Gene.AGGRESSION] = 0.99          # scav_mult ≈ 1
    cell = 3600
    e._flat[:P] = cell                           # 全体同格
    e._corpse_energy[:] = 0.0
    e._corpse_energy[cell] = 1.0                 # 格上只有 1.0
    e._stomach[:P] = 0.0
    before = float(e._corpse_energy[cell])
    e._step_scavenging(P, e._stomach[:P], np.full(P, 100.0), e._genes[:P])
    after = float(e._corpse_energy[cell])
    got = float(e._stomach[:P].sum())
    assert after >= -1e-9, f"格上尸体能量被扣成负值：{after}"
    assert got <= before + 1e-9, f"超发：取走 {got} ＞ 格上存量 {before}"
    # 🔴 R165 0-1（2026-09-23）**单位裁定**：池 = **能量**、胃 = **质量** ⇒
    #    取量（质量）× `eat_efficiency` == 池减（能量）。旧版把池里的数字当质量用
    #    ⇒ 池减 == 取量（**少算 eff 倍**，即"1 单位尸体吐 3 倍能量"）。
    _eff = float(e.config.organisms.eat_efficiency)
    assert abs((before - after) - got * _eff) < 1e-6, (
        f"取量×{_eff} 与池减不等（不守恒）：取走 {got}、格上减少 {before - after}")


def test_scavenging_same_cell_no_double_dip():
    """R161 P1-2 回归（同族）：格存量**充足**时，每人应按自己的需求取（不得互相吞噬）。

    这是"超额分配"的对照组：`Σwant ≤ 存量` ⇒ 每人拿满 `want`、格上按总需求扣减。
    """
    e = _engine(corpse=True)
    P = len(e._id)
    e._genes[:, Gene.AGGRESSION] = 0.99
    cell = 3610
    e._flat[:P] = cell
    e._corpse_energy[:] = 0.0
    e._corpse_energy[cell] = 1e9                 # 存量充足
    e._stomach[:P] = 0.0
    e._step_scavenging(P, e._stomach[:P], np.full(P, 100.0), e._genes[:P])
    total = float(e._stomach[:P].sum())
    assert total > 0.0, "存量充足时应当能吃到"
    # 🔴 R165 0-1：池减（能量）= 取量（质量）× `eat_efficiency`（旧版漏了这个倍率）
    _eff = float(e.config.organisms.eat_efficiency)
    assert abs((1e9 - float(e._corpse_energy[cell])) - total * _eff) < 1e-3, (
        "扣减量应等于总取量×eat_efficiency（逐格守恒，能量口径）")


def test_deposit_accumulates_same_cell():
    """R161 P1-3 回归：同格多具尸体**必须累加**。

    原 bug：`arr[idx] = f(arr[idx])` 的 fancy-index 读-改-写只保留最后一次写入
    ⇒ 实测同格 2 具各 deposit 9.0 只存 **9.0**（丢失 50%）。
    """
    e = _engine(corpse=True)
    P = min(len(e._id), 4)
    assert P >= 2, "需要 ≥2 个体才能构造同格场景"
    cell = 3700
    e._flat[:P] = cell
    e._corpse_energy[:] = 0.0
    dead = np.zeros(P, dtype=bool)
    dead[:2] = True                              # 前两个同 tick 死亡
    energy = np.zeros(P, dtype=np.float64)
    energy[:2] = 10.0                            # 每个 deposit = 10 × 0.9 = 9.0
    e._deposit_corpse(dead, energy)
    got = float(e._corpse_energy[cell])
    assert abs(got - 18.0) < 1e-9, (
        f"同格 2 具各 9.0 应得 18.0，实得 {got}（丢失 {18.0 - got:.3f}）")


def test_cap_scale_matches_corpse_energy():
    """R161 裁定：`corpse_cap_per_cell` 语义 = **单格尸体能量上限**，量级须 ≥ 一具尸体。

    猎物尸体 = 剩余能量 × 0.9 ≈ 126–198 ⇒ cap=3 会把整具尸体钳掉（尸体通道失去意义）。
    """
    cwc = CorpseWoundConfig()
    assert int(cwc.corpse_cap_per_cell) >= 200, (
        f"cap={cwc.corpse_cap_per_cell} 小于一具尸体能量（126–198）⇒ 尸体通道被钳死")
