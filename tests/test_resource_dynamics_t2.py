"""13.4 波 2A 资源动态接线测试（任务书 T2；分支 dev/13.4-wave1）。

覆盖（T2 完成判据 + 波2 修 v2 三态机，fish 批准方案 A）：
  ① **关档逐位等价**：资源动态全关 + eat_amount=0.9（构造变更）⇒ 新 C7 基线
     digest == (574887, 11266.746993)（已同步 9 个测试文件）
  ② **休耕生效**：`_damage ≥ rest_threshold`（默认 0.3，相对容量）⇒ 休耕 N tick 内
     再生 = 0（growth_multiplier = 0）；轻取食（damage < 0.3）**不休耕**
  ③ **死亡触发**：主通道 = `_damage ≥ death_threshold`（0.8，全格）；补充通道 =
     单 tick intake/growth > kill_mult(10)（kill_patch_only 限定斑块）
  ④ **轮作重生**：死格 dead_regen_ticks 后重入候选池 + 斑块加成同行搬移
  ⑤ **conservation_check 恒过**：capacity_total_rel / regrow_area_weighted ≈ 1
  ⑥ **dead_cell_max_frac 闸生效**：死格占比超阈 ⇒ 强制重生
  ⑦ **H3**：resource_dynamics ∧ use_sim_core ⇒ 构造期 NotImplementedError
  ⑧ **接线读数**：probe 关档 None（未适用）/ 开档 dict 齐全
  ⑨ **休耕不刷新**（方案 A 核心）：休耕期被吃 ⇒ rest_until 不变
  ⑩ **到期恢复 + 损伤衰减**：休耕到期 ⇒ growth 恢复 + `_damage × damage_recovery`
"""
from __future__ import annotations

import numpy as np
import pytest

from simulation.config import (
    InfoStructureConfig, ResourceDynamicsConfig, SimConfig,
)
from simulation.sphere_engine import SphereEngine


def _engine(ticks: int = 0, *, seed: int = 42,
            rd: bool = False, use_sim_core: bool = False,
            patchy: bool = False) -> SphereEngine:
    cfg = SimConfig(seed=seed)
    cfg.population.max_count = 600
    cfg.predation.forage_tradeoff_k = 0.0
    cfg.info_structure = InfoStructureConfig(
        enabled=True, learning_rate=0.05, memory_gradient="none",
    )
    cfg.resources.distribution = "patchy" if patchy else "uniform"
    cfg.simulation.use_sim_core = bool(use_sim_core)
    cfg.resource_dynamics = ResourceDynamicsConfig(enabled=bool(rd))
    e = SphereEngine(cfg)
    for _ in range(ticks):
        if e.extinct:
            break
        e.step()
    return e


def _digest(e: SphereEngine) -> tuple[int, float]:
    return (int(e._flat.sum()), round(float(e._energy.sum()), 6))


# --------------------------------------------------------------- ① 关档等价

def test_rd_off_bit_identical():
    """C7：eat_amount=0.9 后新基线（构造变更已同步测试）⇒ 资源动态关档 = 逐位一致。"""
    assert _digest(_engine(ticks=50)) == (574887, 11266.746993), (
        "资源动态关档改变了轨迹 ⇒ 破坏①（默认关 = 逐位等价）")

# --------------------------------------------------------------- ② 休耕生效（波2 修 v2 三态机）

def test_rest_stops_regrowth():
    """休耕：`_damage ≥ rest_threshold`（相对容量）⇒ 再生乘子 0；轻取食不休耕。"""
    e = _engine(rd=True)
    rd = e._rd
    rt = int(rd.rest_ticks)
    intake = np.zeros(e.world.n_cells)
    growth = np.ones(e.world.n_cells)
    # 直接构造：格 0 累计损伤超休耕阈（= 被吃 ≥ 30% 容量）⇒ 下一 note_tick 进入休耕
    cap0 = float(rd.capacity_from_base()[0])
    intake[0] = rd.rest_threshold * cap0 + 1e-9
    rd.note_tick(intake, growth, 100)
    gm = rd.growth_multiplier()
    assert gm[0] == 0.0, "损伤超阈格应休耕（再生乘子 0）"
    assert gm[5] == 1.0, "未吃格应照常生长"
    assert rd.rest_set_n >= 1, "应记录休耕事件"
    assert int(rd._rest_until[0]) == 100 + rt, "休耕截止 = tick + rest_ticks"
    # 轻取食（远小于阈值）⇒ 不休耕
    rd2 = e._rd
    rd2._damage[:] = 0.0
    rd2._rest_until[:] = -1
    intake2 = np.zeros(e.world.n_cells)
    intake2[0] = 0.01 * cap0           # 0.01 × 容量 ≪ 0.3
    rd2.note_tick(intake2, growth, 150)
    assert rd2.growth_multiplier()[0] == 1.0, "轻取食（< rest_threshold）不应休耕"


def test_damage_accumulates_across_ticks():
    """累计损伤：多次轻取食逐步逼近休耕阈 ⇒ 第 N 次触发（文献"反复摘叶 → 退化"）。"""
    e = _engine(rd=True)
    rd = e._rd
    cap0 = float(rd.capacity_from_base()[0])
    growth = np.ones(e.world.n_cells)
    # 每次吃 0.1 × 容量（< 0.3）⇒ 3 次累计 0.3 触发休耕
    n_times = 0
    for t in range(10):
        intake = np.zeros(e.world.n_cells)
        intake[0] = 0.1 * cap0
        rd.note_tick(intake, growth, 100 + t)
        n_times += 1
        if rd._rest_until[0] >= 0:
            break
    assert n_times == 3, f"0.1×3=0.3 应恰好第 3 次触发休耕（实测第 {n_times} 次）"
    assert rd.rest_set_n >= 1


def test_rest_does_not_refresh():
    """⑨ 方案 A 核心：休耕期被吃 ⇒ `rest_until` 不变（不刷新休耕时间）。"""
    e = _engine(rd=True)
    rd = e._rd
    rt = int(rd.rest_ticks)
    cap0 = float(rd.capacity_from_base()[0])
    growth = np.ones(e.world.n_cells)
    t0 = 100
    intake = np.zeros(e.world.n_cells)
    intake[0] = rd.rest_threshold * cap0 + 1e-9    # 触发休耕
    rd.note_tick(intake, growth, t0)
    assert int(rd._rest_until[0]) == t0 + rt
    # 休耕期继续被吃（多次）⇒ rest_until 保持 t0+rt（不被刷新）
    for t in (t0 + 1, t0 + 5, t0 + 30):
        intake = np.zeros(e.world.n_cells)
        intake[0] = 0.1 * cap0
        rd.note_tick(intake, growth, t)
        assert int(rd._rest_until[0]) == t0 + rt, (
            f"休耕期被吃不应刷新（t={t} 时 rest_until={rd._rest_until[0]}）")
    # 休耕期被吃累计 damage ⇒ 推进到死亡（方案 A：被啃食的休耕地退化）
    assert rd._damage[0] >= rd.rest_threshold


def test_rest_expires_and_recovers():
    """⑩ 休耕**到期恢复** + 损伤衰减（2026-09-23 实验 B–E 臂灭绝根因的回归测试）。

    缺陷原型：`note_tick` 设 `_rest_until = tick + rest_ticks` 后，**到期从不重置
    -1** ⇒ 一次被吃 = **永久休耕** ⇒ resting_cell_frac 单调冲到 ~95% ⇒ 灭绝。
    """
    e = _engine(rd=True)
    rd = e._rd
    rt = int(rd.rest_ticks)
    cap0 = float(rd.capacity_from_base()[0])
    growth = np.ones(e.world.n_cells)
    t0 = 100
    intake = np.zeros(e.world.n_cells)
    intake[0] = rd.rest_threshold * cap0 + 1e-9
    rd.note_tick(intake, growth, t0)
    d0 = float(rd._damage[0])
    assert rd.growth_multiplier()[0] == 0.0, "休耕中应 0"
    # 到期前 rotate ⇒ 仍休耕（不提前恢复）
    rd.rotate(t0 + rt - 1, np.array([0.5]))
    assert rd.growth_multiplier()[0] == 0.0, "到期前不应恢复"
    # 到期后 rotate ⇒ 恢复生长 + damage 按 recovery 衰减
    rd.rotate(t0 + rt, np.array([0.5]))
    assert rd.growth_multiplier()[0] == 1.0, "到期后应恢复生长（rest_until 重置）"
    assert float(rd._damage[0]) == pytest.approx(d0 * rd.damage_recovery, rel=1e-9), (
        "休耕到期损伤应按 damage_recovery 衰减")


# --------------------------------------------------------------- ③ 死亡触发

def test_kill_triggers_death():
    """死亡（补充通道）：被吃强度 > kill_mult(10) ⇒ 斑块格死亡 + 斑块加成搬走。"""
    e = _engine(rd=True, patchy=True)
    rd = e._rd
    n_patch0 = int(rd._mask.sum())
    assert n_patch0 > 0, "patchy 下应有斑块格"
    # 构造：某斑块格被吃 20× 当期再生（> 10）⇒ 死亡
    patch_cell = int(np.flatnonzero(rd._mask)[0])
    intake = np.zeros(e.world.n_cells)
    intake[patch_cell] = 20.0        # 远大于 kill_mult=10
    growth = np.ones(e.world.n_cells)
    rd.note_tick(intake, growth, 200)
    assert rd._dead[patch_cell], "超阈被吃应触发斑块死亡"
    assert rd.patch_kill_n == 1, "应记 1 次死亡事件"
    assert not rd._mask[patch_cell], "死亡斑块格应失去斑块加成"


def test_damage_death_channel():
    """死亡（主通道，方案 A）：`_damage ≥ death_threshold`（0.8）⇒ 全格可死（含背景）。"""
    e = _engine(rd=True, patchy=True)
    rd = e._rd
    growth = np.ones(e.world.n_cells)
    # 背景格（非斑块）累计被吃 ≥ 80% 容量 ⇒ 死亡（主通道不受 kill_patch_only 限制）
    bg = int(np.flatnonzero(~rd._mask)[0])
    cap_bg = float(rd.capacity_from_base()[bg])
    intake = np.zeros(e.world.n_cells)
    intake[bg] = rd.death_threshold * cap_bg + 1e-9
    rd.note_tick(intake, growth, 300)
    assert rd._dead[bg], "累计损伤 ≥ 80% 容量应死亡（主通道，含背景格）"
    assert rd.patch_kill_n >= 1
    assert rd._damage[bg] == 0.0, "死亡格损伤应清零（重生后从 0 累计）"
    # 损伤未达死亡阈 ⇒ 不死
    e2 = _engine(rd=True, patchy=True)
    rd2 = e2._rd
    bg2 = int(np.flatnonzero(~rd2._mask)[0])
    cap2 = float(rd2.capacity_from_base()[bg2])
    intake2 = np.zeros(e2.world.n_cells)
    intake2[bg2] = 0.5 * cap2        # 0.5 < 0.8
    rd2.note_tick(intake2, growth, 300)
    assert not rd2._dead[bg2], "0.5 容量损伤不应死亡（只到休耕档）"
    assert rd2._rest_until[bg2] >= 0, "0.5 ≥ 0.3 应已休耕"


# --------------------------------------------------------------- ④ 轮作重生

def test_rotation_reborn_and_promote():
    """轮作：死格到期重生 + 斑块加成**同行**搬到随机背景格（Σcapacity 守恒）。"""
    e = _engine(rd=True, patchy=True)
    rd = e._rd
    n_patch0 = int(rd._mask.sum())
    assert n_patch0 > 0, "patchy 下应有斑块格"
    # 构造一个死格：置死 + 置 `_demoted`（= note_tick 判死后留下的"加成待搬"标记）
    patch_cell = int(np.flatnonzero(rd._mask)[0])
    rd._dead[patch_cell] = True
    rd._dead_since[patch_cell] = 0
    rd._rest_until[patch_cell] = -1
    rd._mask[patch_cell] = False
    rd._demoted[patch_cell] = True
    # rotate：先搬走 `_demoted` 的加成（promote_n ≥ 1），再处理到期重生（tick 3000 ≥ 2000）
    rd.rotate(3000, np.linspace(0.0, 1.0, max(n_patch0, 1)))
    assert rd.promote_n >= 1, "死格斑块加成应被搬移（promote）"
    assert not rd._dead[patch_cell], "到期死格应重生"
    assert rd.patch_reborn_n >= 1, "应记重生事件"
    # 守恒：Σcapacity 不变（同行交换 ⇒ 逐位）
    cc = rd.conservation_check()
    assert abs(cc["capacity_total_rel"] - 1.0) < 1e-6, (
        f"轮作后 Σcapacity 应守恒（{cc['capacity_total_rel']}）")


# --------------------------------------------------------------- ⑤ conservation

def test_conservation_check_passes():
    """conservation_check 恒过：构造后两条守恒 ≈ 1（关档 None）。"""
    e = _engine(rd=True)
    cc = e.resource_dynamics_conservation()
    assert cc is not None
    assert abs(cc["capacity_total_rel"] - 1.0) < 1e-6
    assert abs(cc["regrow_area_weighted"] - 1.0) < 1e-6
    # 关档 ⇒ None（未适用）
    e2 = _engine(rd=False)
    assert e2.resource_dynamics_conservation() is None


# --------------------------------------------------------------- ⑥ 反荒漠化闸

def test_dead_cell_max_frac_gate():
    """反荒漠化闸：死格占比超 dead_cell_max_frac ⇒ 强制重生（forced_reborn_n>0）。"""
    e = _engine(rd=True, patchy=True)
    rd = e._rd
    n = e.world.n_cells
    # 人为制造 60% 死格（> 0.5 闸）
    rd._dead[:] = False
    kill_frac_cells = int(0.6 * n)
    rd._dead[:kill_frac_cells] = True
    rd._dead_since[:kill_frac_cells] = 0
    n_patch = int(rd._mask.sum())
    rd.rotate(100, np.linspace(0.0, 1.0, max(n_patch, 1)))
    assert rd.forced_reborn_n > 0, "超阈应触发强制重生（反荒漠化闸）"
    assert int(rd._dead.sum()) <= int(0.5 * n), "闸后死格应回落到 ≤ 阈值"


# --------------------------------------------------------------- ⑦ H3

def test_rd_use_sim_core_h3():
    """H3 fail-loud：resource_dynamics ∧ use_sim_core ⇒ 构造期 NotImplementedError。"""
    with pytest.raises(NotImplementedError):
        _engine(rd=True, use_sim_core=True)


# --------------------------------------------------------------- ⑧ 接线读数

def test_rd_probe_readback():
    """接线读数：probe 关档 None（未适用）/ 开档 dict 含四条 + mean_capacity_effective。"""
    e = _engine(ticks=2000, rd=True)
    pr = e.resource_dynamics_probe()
    assert pr is not None, "开档 probe 不应为 None"
    for k in ("dead_cell_frac", "resting_cell_frac",
              "patch_kill_n", "patch_reborn_n", "mean_capacity_effective"):
        assert k in pr, f"probe 缺键 {k}"
    # ⚠️ 波2 修 v2（方案 A，2026-09-23）：死格占比可顶到反荒漠化闸（0.5）循环——
    #    死格由闸强制重生，属**受控的强平衡**，不是无限荒漠化；断言从 <0.1 放宽为
    #    "不越过闸 + resting 受控 + 种群存活"（旧 <0.1 是 kill_frac=10 极端通道时代的
    #    世界观：死亡几乎不触发）。
    assert pr["dead_cell_frac"] <= 0.55, (
        f"冒烟判据：dead_cell_frac 应被反荒漠化闸压在 0.5 附近（实测 {pr['dead_cell_frac']}）")
    assert pr["resting_cell_frac"] is not None and pr["resting_cell_frac"] < 0.5, (
        f"resting 应受控（三态机不复现 82–92% 过冲；实测 {pr['resting_cell_frac']}）")
    assert len(e._id) > 0, "2000 tick 冒烟种群应存活"
    e_off = _engine(ticks=50, rd=False)
    assert e_off.resource_dynamics_probe() is None, "关档应 None（未适用）"
