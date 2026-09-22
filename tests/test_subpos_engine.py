"""13.4 波 1「亚格连续坐标」引擎级回归测试（`[所有者·天平]` 线）。

覆盖（对应设计稿 §六 的 E1–E9）：
  E1 **关档逐位等价**（C7 基线 `(573985, 8171.692943)`）
  E3 **H3 fail-loud** 两条：subpos × use_sim_core、subpos × l2_dash
  E4 **I2**：开档下 `_flat` 与亚格坐标恒一致（防两条路径漂移）
  E5 **扩容/压缩同步**：出生与死亡后三数组长度恒等
  E6 **快照往返**（含**缺键回退**：旧快照 ⇒ 由 `_flat` 回推格中心）
  E8 **反退化**：`steps` 分布不是单一档（否则速度映射塌成常数）
  E9 **极点无幽灵位移**：极点行 `sub_c` 被钳制、`flat` 自洽
  E10 **开关语义**：开档必须真的改行为（防"接了却没生效"）
"""
from __future__ import annotations

import numpy as np
import pytest

from simulation.config import InfoStructureConfig, SimConfig
from simulation.sphere_engine import SphereEngine
from world.subpos import flat_of_sub

C7_BASE = (573985, 8171.692943)


def _engine(ticks: int = 50, *, subpos: bool = False, seed: int = 42,
            max_count: int = 600, l2: bool = False, use_core: bool = False,
            **sub_kw) -> SphereEngine:
    cfg = SimConfig(seed=seed)
    cfg.simulation.use_sim_core = use_core
    cfg.simulation.l2_dash = l2
    cfg.population.max_count = max_count
    cfg.predation.forage_tradeoff_k = 0.0
    cfg.info_structure = InfoStructureConfig(
        enabled=True, learning_rate=0.05, memory_gradient="none")
    cfg.organisms.young_mob_mult = 0.55
    cfg.organisms.old_mob_mult = 0.55
    cfg.organisms.dash_min_energy_frac = 0.2
    cfg.organisms.far_cap = 32
    cfg.subpos.enabled = subpos
    for k, v in sub_kw.items():
        setattr(cfg.subpos, k, v)
    e = SphereEngine(cfg)
    for _ in range(ticks):
        if e.extinct:
            break
        e.step()
    return e


def _digest(e: SphereEngine) -> tuple[int, float]:
    return (int(e._flat.sum()), round(float(e._energy.sum()), 6))


# ------------------------------------------------------------------ E1

def test_E1_subpos_off_is_bit_identical():
    """C7：`subpos.enabled=False` ⇒ 与**接坐标之前**逐位一致。"""
    assert _digest(_engine(50, subpos=False)) == C7_BASE, (
        "关档改变了轨迹 ⇒ 破坏 I1（默认关 = 逐位等价）")


# ------------------------------------------------------------------ E10

def test_E10_subpos_on_actually_changes_behaviour():
    """开档必须**真的**改行为（防"接了却没生效" —— dash PR 的同族风险）。"""
    assert _digest(_engine(50, subpos=True)) != _digest(_engine(50, subpos=False))


# ------------------------------------------------------------------ E3

@pytest.mark.parametrize("kw,what", [
    (dict(subpos=True, use_core=True), "subpos × use_sim_core"),
    (dict(subpos=True, l2=True), "subpos × l2_dash（语义互斥）"),
])
def test_E3_h3_fail_loud(kw, what):
    """H3：两种非法组合都必须**构造期**抛出（不静默换路径）。"""
    with pytest.raises(NotImplementedError):
        _engine(2, **kw)


def test_E3_h3_not_triggered_when_off():
    """关档时 H3 不得误伤：`l2_dash=True` + subpos 关是**合法**的。"""
    _engine(2, subpos=False, l2=True)   # 不抛即通过


# ------------------------------------------------------------------ E4

@pytest.mark.parametrize("ticks", [1, 50, 200])
def test_E4_flat_consistent_with_subpos(ticks: int):
    """I2：开档下 `_flat` 必须恒等于由亚格坐标派生的格（逐位）。"""
    e = _engine(ticks, subpos=True)
    n = len(e._id)
    back = flat_of_sub(e._sub_r[:n], e._sub_c[:n],
                       int(e.config.subpos.subdiv), e.world)
    assert np.array_equal(back, e._flat[:n]), (
        f"tick={ticks}：亚格坐标与 `_flat` 脱钩（I2 被破坏）")


# ------------------------------------------------------------------ E5

def test_E5_array_lengths_stay_synced():
    """扩容（出生）与压缩（死亡）后，亚格数组必须与 `_flat`/`_energy` 同长。"""
    e = _engine(400, subpos=True, max_count=300)
    n = len(e._id)
    assert len(e._sub_r) == len(e._sub_c) == len(e._flat) == len(e._energy) == n


# ------------------------------------------------------------------ E6

def test_E6_snapshot_roundtrip(tmp_path):
    """快照往返：`_sub_r/_sub_c` 逐位一致。"""
    e = _engine(30, subpos=True)
    p = str(tmp_path / "snap.npz")
    e.save_snapshot(p)
    # ⚠️ `load_snapshot` 是 **classmethod 且返回新 engine** ——
    #    写成 `e2.load_snapshot(p)` 会**丢掉返回值**（这是本测试第一版踩的坑）。
    e2 = SphereEngine.load_snapshot(p)
    assert np.array_equal(e2._sub_r, e._sub_r)
    assert np.array_equal(e2._sub_c, e._sub_c)


def test_E6_snapshot_missing_keys_fallback():
    """**缺键回退**：旧快照（无 sub_r/sub_c）⇒ 由 `_flat` 回推**格中心**，不报错。"""
    e = _engine(20, subpos=True)
    n = len(e._id)
    # 模拟旧快照读入后的状态：先用 `_flat` 回推，与 `_init_from_flat` 的语义一致
    from world.subpos import init_from_flat
    r, c = init_from_flat(e._flat[:n], int(e.config.subpos.subdiv), e.world)
    back = flat_of_sub(r, c, int(e.config.subpos.subdiv), e.world)
    assert np.array_equal(back, e._flat[:n]), "缺键回退路径不满足 I2"


# ------------------------------------------------------------------ E8

def test_E8_anti_degeneracy_steps_not_constant():
    """反退化：开档后 `steps` 必须**不是单一档**（否则速度映射塌成常数 ⇒ 机制等于没做）。"""
    e = _engine(60, subpos=True)
    hist = e._steps_hist
    assert e._run_mover_sub_n > 0, "没有移动者 ⇒ 测试前提不成立"
    nonzero = int((hist > 0).sum())
    assert nonzero >= 2, (
        f"steps 只落在 {nonzero} 个档位 ⇒ 速度映射塌成常数，机制名存实亡。hist={hist.tolist()}")


def test_E8_off_has_no_readouts():
    """关档时读数必须保持 0（口径：**未适用**，不用假数据充数）。"""
    e = _engine(20, subpos=False)
    assert (e._run_mover_sub_n, e._run_flat_move_n, e._run_slow_n) == (0, 0, 0)


# ------------------------------------------------------------------ E9

def test_E9_pole_no_ghost_displacement():
    """极点：`sub_c` 必须被钳制在 `[0, subdiv)`（极点物理上是一个点 ⇒ 无"沿纬度带滑动"）。"""
    from world.subpos import advance_sub, init_from_flat
    e = _engine(1, subpos=False)
    s = int(e.config.subpos.subdiv)
    flat = np.array([1 * e.world.cols + 60], dtype=np.int64)   # row 1 的经度中部
    r0, c0 = init_from_flat(flat, s, e.world)
    r2, c2 = advance_sub(r0, c0, np.array([-1]), np.array([0]), np.array([s]), s, e.world)
    assert int(r2[0] // s) == 0, "应落入上极行"
    assert 0 <= int(c2[0]) < s, f"极点行 sub_c 未钳制：{int(c2[0])}"


# ------------------------------------------------------------------ stay

def test_stay_max_is_anti_degeneracy_gate():
    """`stay_max=0.8` ⇒ 仍有移动者（硬性反退化闸生效：不许全停）。"""
    e = _engine(30, subpos=True, stay_base=0.8)
    assert e._run_mover_sub_n > 0, "stay_base=0.8 时无人移动 ⇒ 反退化闸失效"
