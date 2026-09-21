"""PC-1 构造开关（R134，2026-09-20）回归：S2 `predation.enabled` / S1 `population.soft_cap_target`。

设计稿：`docs/设计文档/设计-PC1极简正对照-实现侧-20260919.md`（S1 冒烟后修订为**目标窗形式**）；
所有者裁定：字母表 "16"、`reputation_weight=1.0`、稳态窗 [0.2K, 0.95K]。

四条约定（设计稿 §一）：
  · **默认 = 旧行为** ⇒ 由 `tests/test_memory_gradient.py` 的钉死 digest 全绿作 C7 证据
    （本文件不重复钉 digest，只测新取值的构造性断言与"开关确实生效"）；
  · S2 off ⇒ **捕食死亡恒 0**（构造性断言）且与 on 的轨迹**不同**（防静默 no-op，教训 11）；
  · S1 on（target=0.6）⇒ 出生被密度抑制（方向断言，固定 seed 下确定）；
  · 两开关都进 `config_fingerprint`（to_dict = asdict 自动覆盖，防续跑跨配置）。
"""
from __future__ import annotations

from simulation.config import InfoStructureConfig, SimConfig
from simulation.sphere_engine import SphereEngine


def _engine(*, predation: bool = True, soft_cap_target: float = 0.0, seed: int = 42,
            max_count: int = 600, initial_count: int = 200,
            full_food: bool = False) -> SphereEngine:
    cfg = SimConfig(seed=seed)
    cfg.simulation.use_sim_core = False
    cfg.population.max_count = max_count
    cfg.population.initial_count = initial_count
    cfg.population.soft_cap_target = soft_cap_target
    cfg.predation.enabled = predation
    if full_food:
        cfg.resources.background_fill = 1.0
        cfg.resources.initial_fill = 1.0
    cfg.info_structure = InfoStructureConfig(enabled=True, learning_rate=0.05)
    return SphereEngine(cfg)


def _run(e: SphereEngine, ticks: int = 120):
    pred_deaths = 0
    total_deaths = 0
    for _ in range(ticks):
        if e.extinct:
            break
        st = e.step()
        for k, v in st.deaths_by_cause.items():
            total_deaths += int(v)
            if getattr(k, "name", str(k)) == "PREDATION":
                pred_deaths += int(v)
    return pred_deaths, total_deaths


def _digest(e: SphereEngine, ticks: int = 80):
    for _ in range(ticks):
        if e.extinct:
            break
        e.step()
    return (int(e._flat.sum()), round(float(e._energy.sum()), 6))


# ------------------------------------------------------------ S2 关捕食
def test_predation_off_has_zero_predation_deaths():
    """🔴 构造性断言：S2 off ⇒ **捕食死亡恒 0**（且总死亡 > 0，保证断言非空洞）。"""
    pred_deaths, total_deaths = _run(_engine(predation=False))
    assert pred_deaths == 0, "捕食已关却仍有 PREDATION 死亡 ⇒ 开关没接上（静默 no-op）"
    assert total_deaths > 0, "120 tick 内零死亡 ⇒ 断言空洞（换更长的 run）"


def test_predation_switch_is_behaviorally_active():
    """on 与 off 的轨迹必须不同（防静默 no-op，教训 11）。"""
    a = _digest(_engine(predation=True))
    b = _digest(_engine(predation=False))
    assert a != b, "S2 两档逐位相同 ⇒ 关捕食没接进引擎"


# ------------------------------------------------------------ S1 软顶（目标窗形式）
def test_soft_cap_switch_is_behaviorally_active():
    """on（target=0.6）与 off 的轨迹/出生数必须不同（防静默 no-op，教训 11）。

    场景设计（三轮实测的结论）：本世界出生**本身稀有**（成熟期长 + 能量门槛），
    且均衡 N ≪ K ⇒ 软顶只在出生瞬间绑定 ⇒ 用 **K=60/ini=50 + 食物拉满 + 1200 tick +
    固定 seed**：引擎确定性 ⇒ `born_on < born_off`（实测 17 < 21）是稳定回归断言；
    真正的抑制力度由 PC-1 冒烟（稳态窗门 N_eq∈[0.2K,0.95K]）在 12k 尺度验收。
    """
    def run(soft: float):
        e = _engine(soft_cap_target=soft, max_count=60, initial_count=50, full_food=True)
        born = 0
        for _ in range(1200):
            if e.extinct:
                break
            born += e.step().born
        return born, _digest(e, ticks=0)

    born_off, dig_off = run(0.0)
    born_on, dig_on = run(0.6)
    assert dig_on != dig_off, "S1 两档逐位相同 ⇒ 软顶没接进繁殖门（静默 no-op）"
    assert born_on < born_off, f"软顶未抑制出生（on={born_on} vs off={born_off}）"


# ------------------------------------------------------------ 指纹与缺省
def test_switches_enter_config_fingerprint():
    """两开关必须进 `config_fingerprint`（to_dict=asdict 自动覆盖 ⇒ 防续跑跨配置）。"""
    base = SimConfig().fingerprint()
    c1 = SimConfig(); c1.predation.enabled = False
    c2 = SimConfig(); c2.population.soft_cap_target = 0.6
    assert c1.fingerprint() != base != c2.fingerprint()
    d = SimConfig().to_dict()
    assert d["predation"]["enabled"] is True
    assert d["population"]["soft_cap_target"] == 0.0


def test_defaults_are_legacy_behavior():
    """🔴 默认值 = 旧行为（这是两个开关的全部兼容性承诺）。"""
    c = SimConfig()
    assert c.predation.enabled is True
    assert c.population.soft_cap_target == 0.0


def test_soft_cap_target_validation():
    """N* = 1.0 等于没顶（软顶形同虚设）⇒ 配置期硬失败。"""
    try:
        SimConfig().population.soft_cap_target = 1.0  # 构造后赋值不触发校验（F1 教训）
    except AttributeError:
        pass  # __slots__/frozen 等实现差异下允许
    import pytest
    from simulation.config import PopulationConfig
    with pytest.raises(AssertionError):
        PopulationConfig(soft_cap_target=1.0)
    with pytest.raises(AssertionError):
        PopulationConfig(soft_cap_target=-0.1)
