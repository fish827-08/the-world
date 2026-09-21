"""R144 能量封顶四件（开关 / 测试 / 冒烟 / 纪元声明）之**测试**。

背景（R144/R145 事故）：`max_energy = 300.0  # 能量上限（多余溢出丢弃）` —— 那句
"溢出丢弃"**代码里从来没有过**，M 天无人核对（C4 只问"开关=脚本意图吗"，**没人问
"文档声明=代码实际吗"**）。实测 `mean_energy` 涨到 9896 = **33× 上限**，
选择压力被稀释。修复（`7516ba9`）本身正确，但**无开关/无测试/无冒烟/无纪元声明**。

本模块钉死四件事：
  ① **默认关 = 逐位等价**（回到 `7516ba9` 之前的行为 ⇒ 与 E-017~E-031/calib1 可比）
  ② 开时 `energy ≤ max_energy` **恒成立**（并留下"活体探针"读数）
  ③ 开关进配置指纹（纪元可判）
  ④ `photo_max` 覆盖可用（R145 §七.1 的配对臂需要）
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from simulation.config import InfoStructureConfig, SimConfig  # noqa: E402
from simulation.sphere_engine import SphereEngine  # noqa: E402


def _engine(*, cap: bool = False, seed: int = 42, max_count: int = 600,
            ticks: int = 0, photo_max: float | None = None) -> SphereEngine:
    cfg = SimConfig(seed=seed)
    cfg.simulation.use_sim_core = False
    cfg.population.max_count = max_count
    cfg.organisms.energy_cap_enabled = cap
    if photo_max is not None:
        cfg.organisms.photo_max = float(photo_max)
    cfg.info_structure = InfoStructureConfig(
        enabled=True, learning_rate=0.05, memory_gradient="none",
    )
    e = SphereEngine(cfg)
    for _ in range(ticks):
        if e.extinct:
            break
        e.step()
    return e


def _digest(e: SphereEngine, ticks: int = 50) -> tuple:
    for _ in range(ticks):
        if e.extinct:
            break
        e.step()
    return int(e._flat.sum()), round(float(e._energy.sum()), 6)


# ---------------------------------------------------------------- ① 默认关 = 逐位等价

def test_default_is_off():
    """默认必须**关**（否则新纪元会悄悄污染所有既有批的可比性）。"""
    assert SimConfig().organisms.energy_cap_enabled is False


def test_cap_off_is_bitwise_identical_to_pre_fix_baseline():
    """🔴 C7：关时的 digest 必须等于**加钳制之前**的基线。

    基线 `(573985, 8171.692943)` 取自 `7516ba9` 之前的代码
    （`git stash` 实测，配置：50 tick / seed 42 / max_count 600 / D2 enabled）。
    这条一旦变 ⇒ **既有 12 批的可比性被破坏**。
    """
    d = _digest(_engine(cap=False))
    assert d == (573985, 8171.692943), (
        f"关时 digest 变了（{d}）⇒ 默认不再是旧行为，跨批可比性被破坏"
    )


def test_cap_on_starts_identical_then_enforces_on_injection():
    """🔴 确定性验证钳制**真的接线**（不依赖"自然跑会不会超限"）。

    ⚠️ 两条被测试自己抓出来的教训（首版都踩了）：
      ① 用 50 tick 断言"两档不同" ⇒ **假失败**：初始能量 60、上限 300，短跑到不了上限；
      ② 改 300 tick 后仍**逐位相同** ⇒ 该环境（max_count=600 / uniform / D2）**人均能量
         压根到不了 300** ⇒ "自然跑出差异"这个判据**本身不成立**（不是 no-op）。
    ⇒ 改用**注入法**：手工把能量顶到上限之上，跑 1 tick，看是否被钳回。
    """
    a = _engine(cap=False, ticks=0)
    b = _engine(cap=True, ticks=0)
    assert (int(a._flat.sum()), round(float(a._energy.sum()), 6)) == \
           (int(b._flat.sum()), round(float(b._energy.sum()), 6)), "t=0 两档不应有差异"

    cap = b.config.organisms.max_energy
    seen = {}
    for flag in (False, True):
        eng = _engine(cap=flag, ticks=0)
        P = len(eng._id)
        eng._energy[:P] = cap * 5.0          # 注入：5× 上限
        eng.step()
        P2 = len(eng._id)
        seen[flag] = float(eng._energy[:P2].max()) if P2 else 0.0
    assert seen[False] > cap, "关档注入后竟被钳制 ⇒ 默认关不是旧行为"
    assert seen[True] <= cap, f"开档注入后仍达 {seen[True]:.2f} > 上限 {cap} ⇒ 钳制没接线"


# ---------------------------------------------------------------- ② 开时不变式

def test_cap_on_enforces_energy_bound():
    """开时 `energy ≤ max_energy` 必须**恒成立**（跑 300 tick 后仍成立）。"""
    e = _engine(cap=True, ticks=300)
    P = len(e._id)
    assert P > 0
    over = int(np.count_nonzero(e._energy[:P] > e.config.organisms.max_energy))
    assert over == 0, f"{over} 个个体越限 ⇒ 钳制失效"


def test_cap_on_probe_is_zero():
    """R145 补丁②的**活体探针**：钳制后统计的越限占比应恒 0。"""
    e = _engine(cap=True, ticks=300)
    pr = e.energy_cap_probe()
    assert pr["enabled"] is True
    assert pr["frac_max"] == 0.0 and pr["over_cap_seen"] == 0


def test_cap_off_probe_reports_disabled():
    """关时探针要如实说"没启用"，**不能**假报 0（未观测 ≠ 观测到 0）。"""
    pr = _engine(cap=False, ticks=100).energy_cap_probe()
    assert pr["enabled"] is False


def test_state_bounds_check_clean():
    """边界自检（R145 补丁③）：正常运行时三类违例都应为 0。"""
    b = _engine(cap=True, ticks=300).state_bounds_check()
    assert b["n"] > 0
    assert b["energy_over_cap"] == 0
    assert b["age_negative"] == 0
    # ⚠️ `stomach_over_cap` **不保证 0**：进食与捕食两条路径的胃容量口径不同
    #    （后者是前者的约 1.6 倍）⇒ 捕食后可能超过"进食口径"的上限。这是**既有行为**，
    #    自检如实报两个口径；此处只锚定"捕食口径必须无越限"。
    assert b["stomach_over_cap_pred"] == 0, "捕食口径的胃容量都被突破了 ⇒ 真异常"


# ---------------------------------------------------------------- ③ 指纹/纪元

def test_cap_switch_is_in_config_fingerprint():
    """开关必须进指纹 ⇒ 否则新纪元与旧纪元产物**无法从产物区分**。"""
    a, b = SimConfig(), SimConfig()
    b.organisms.energy_cap_enabled = True
    assert a.fingerprint() != b.fingerprint(), "封顶开关未进指纹 ⇒ 纪元不可判"


# ---------------------------------------------------------------- ④ photo_max

def test_photo_max_zero_kills_photosynthesis():
    """`--photo-max 0` ⇒ 光合收入为 0（R145 §六 的"不删基因、改配置"做法）。"""
    e = _engine(photo_max=0.0, ticks=200)
    led = e.energy_ledger()
    if led.get("path") == "python":
        assert led["global"]["intake_photo_sum"] == 0.0, "photo_max=0 仍有光合收入"


def test_photo_max_changes_trajectory():
    on = _digest(_engine(photo_max=0.1))
    off = _digest(_engine(photo_max=0.0))
    assert on != off, "photo_max 覆盖未生效（静默 no-op）"
