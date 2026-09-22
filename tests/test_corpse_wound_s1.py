"""尸体—食腐 + 血条—受伤 S1 骨架测试（设计稿 §5.2；本段 = `[云端·开发]` 线）。

覆盖：
  ① **H1**：corpse/wound/contest 全关 ⇒ **逐位等价**（C7 对拍，基线同
     `test_a_continuous`/`test_l1_terms`/`test_l2_dash` 钉死值）
  ② **H3 fail-loud**：`use_sim_core=True` + 任一开关开 ⇒ 构造期必抛（**不得**静默忽略）
  ③ **开关读回**：配置字段可从产物读回（C4）+ 进**配置指纹**（纪元可自证）
  ④ **数据结构存在且对齐**：格数组 corpse_energy/corpse_age 形状 = (n_cells,)；
     个体数组 health 初值 1.0、与种群同步扩容/压缩
  ⑤ **空壳读数块**：corpse_probe()/wound_probe() 返回字典（值可为 0 —— 设计稿 §5.2 项 9）
  ⑥ **快照往返**：新数组随快照保存/恢复；旧快照缺键回退零值/初值
"""
from __future__ import annotations

import numpy as np
import pytest

from simulation.config import CorpseWoundConfig, InfoStructureConfig, SimConfig
from simulation.sphere_engine import SphereEngine


def _engine(ticks: int = 0, *, use_core: bool = False, seed: int = 42,
            corpse: bool = False, wound: bool = False, contest: bool = False,
            cwc: CorpseWoundConfig | None = None) -> SphereEngine:
    cfg = SimConfig(seed=seed)
    cfg.simulation.use_sim_core = use_core
    cfg.population.max_count = 600
    cfg.predation.forage_tradeoff_k = 0.0
    cfg.info_structure = InfoStructureConfig(
        enabled=True, learning_rate=0.05, memory_gradient="none",
    )
    cfg.corpse_wound = cwc or CorpseWoundConfig(
        corpse_enabled=corpse, wound_enabled=wound, contest_enabled=contest,
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

def test_corpse_wound_default_off_is_bit_identical():
    """C7：三开关全关必须与**改动前**逐位一致（基线同 `test_l2_dash`/`test_a_continuous`）。"""
    assert _digest(_engine(ticks=50)) == (574887, 11266.746993), (
        "S1 关档改变了轨迹 ⇒ 破坏 H1（默认关 = 逐位等价）")


# --------------------------------------------------------------- ② H3

@pytest.mark.parametrize("on", ["corpse", "wound", "contest"])
def test_s1_with_sim_core_raises(on: str):
    """H3：任一开关开 + `use_sim_core=True` ⇒ **构造期直接抛**（不静默走旧路径）。"""
    cfg = SimConfig(seed=1)
    cfg.simulation.use_sim_core = True
    kw = {"corpse_enabled": on == "corpse",
          "wound_enabled": on == "wound",
          "contest_enabled": on == "contest"}
    cfg.corpse_wound = CorpseWoundConfig(**kw)
    with pytest.raises(NotImplementedError, match="尸体—食腐"):
        SphereEngine(cfg)


def test_s1_all_off_with_sim_core_ok():
    """全关 + use_sim_core=True 必须**能构造**（H3 只拦"开 + Rust"，不拦关档）。"""
    cfg = SimConfig(seed=1)
    cfg.simulation.use_sim_core = True
    e = SphereEngine(cfg)
    assert e._corpse_energy.sum() == 0.0


# --------------------------------------------------------------- ③ 开关读回

def test_corpse_wound_switches_visible_for_epoch():
    """开关与参数必须可从产物读回（C4）；且须进**配置指纹** ⇒ 纪元可自证。"""
    e = _engine(corpse=True, wound=True, contest=True)
    cw = e.config.corpse_wound
    assert cw.corpse_enabled is True
    assert cw.wound_enabled is True
    assert cw.contest_enabled is True
    assert float(cw.corpse_energy_frac) == 0.9
    assert int(cw.corpse_decay_ticks) == 600
    # 指纹：任一开关/参数变化 ⇒ fingerprint 不同（纪元可判）
    base = SimConfig()
    for field in ("corpse_enabled", "wound_enabled", "contest_enabled",
                  "corpse_energy_frac", "corpse_decay_ticks", "wound_base",
                  "holder_adv", "w_fear_health", "need_aggression_k"):
        b = SimConfig()
        setattr(b.corpse_wound, field, {"corpse_enabled": True, "wound_enabled": True,
                                        "contest_enabled": True, "corpse_energy_frac": 0.5,
                                        "corpse_decay_ticks": 300, "wound_base": 0.2,
                                        "holder_adv": 0.5, "w_fear_health": 0.1,
                                        "need_aggression_k": 0.0}[field])
        assert base.fingerprint() != b.fingerprint(), f"{field} 未进指纹 ⇒ 纪元不可判"


# --------------------------------------------------------------- ④ 数据结构

def test_corpse_grid_arrays_shape_and_zeros():
    """格数组形状 = (n_cells,) 且初值 0（机制未接线 ⇒ 恒零）。"""
    e = _engine()
    assert e._corpse_energy.shape == (e.world.n_cells,)
    assert e._corpse_age.shape == (e.world.n_cells,)
    assert e._corpse_energy.dtype == np.float64
    assert e._corpse_age.dtype == np.int64
    assert e._corpse_energy.sum() == 0.0
    assert e._corpse_age.sum() == 0


def test_health_array_initialized_to_one_and_aligned():
    """health 初值 1.0；跑若干 tick 后仍与种群数组同长（随生死同步）。"""
    e = _engine(ticks=30)
    assert np.all(e._health == 1.0)
    assert len(e._health) == len(e._id)


# --------------------------------------------------------------- ⑤ 空壳读数块

def test_corpse_probe_shell():
    """corpse_probe() 返回字典：开关读回 + 空壳值（corpse_total/eaten 恒 0）。"""
    e = _engine(ticks=10)
    pr = e.corpse_probe()
    assert pr["corpse_enabled"] is False
    assert pr["corpse_total"] == 0.0
    assert pr["corpse_eaten"] == 0
    assert pr["corpse_age_max"] == 0


def test_wound_probe_shell():
    """wound_probe() 返回字典：开关读回 + 空壳值（health_mean 恒 1.0，wound/contest 恒 0）。"""
    e = _engine(ticks=10)
    pr = e.wound_probe()
    assert pr["wound_enabled"] is False
    assert pr["contest_enabled"] is False
    assert pr["health_mean"] == pytest.approx(1.0)
    assert pr["health_low_frac"] == 0.0
    assert pr["wound_n"] == 0
    assert pr["contest_n"] == 0


# --------------------------------------------------------------- ⑥ 快照往返

def test_snapshot_roundtrip_corpse_wound_arrays(tmp_path):
    """快照往返保持新数组；缺键（旧快照）回退零值/初值。"""
    e = _engine(ticks=10)
    snap = tmp_path / "s1.npz"
    e.save_snapshot(str(snap))
    e2 = SphereEngine.load_snapshot(str(snap))
    np.testing.assert_array_equal(e2._corpse_energy, e._corpse_energy)
    np.testing.assert_array_equal(e2._corpse_age, e._corpse_age)
    np.testing.assert_array_equal(e2._health, e._health)
    assert e2._corpse_eaten_n == e._corpse_eaten_n
    assert e2._wound_n == e._wound_n
    assert e2._contest_n == e._contest_n


def test_load_snapshot_missing_keys_falls_back(tmp_path):
    """旧快照（无 corpse_energy/health 键）加载 ⇒ 回退零值/初值（不报错）。"""
    e = _engine(ticks=5)
    snap = tmp_path / "old.npz"
    e.save_snapshot(str(snap))
    import numpy as _np
    data = dict(_np.load(snap, allow_pickle=True))
    for k in ("corpse_energy", "corpse_age", "health",
              "corpse_eaten_n", "wound_n", "contest_n"):
        data.pop(k, None)
    old = tmp_path / "stripped.npz"
    _np.savez_compressed(old, **data)
    e2 = SphereEngine.load_snapshot(str(old))
    assert e2._corpse_energy.sum() == 0.0
    assert e2._corpse_age.sum() == 0
    assert np.all(e2._health == 1.0)
    assert e2._wound_n == 0


# --------------------------------------------------------------- ⑦ from_dict 兼容

def test_from_dict_missing_corpse_wound_falls_back():
    """旧配置 dict（无 corpse_wound 键）⇒ from_dict 回退默认（全关 = 旧行为）。"""
    c = SimConfig(seed=7)
    d = c.to_dict()
    d.pop("corpse_wound", None)
    c2 = SimConfig.from_dict(d)
    assert c2.corpse_wound.corpse_enabled is False
    assert c2.fingerprint() == c.fingerprint(), "缺键回退后指纹应与默认一致"
