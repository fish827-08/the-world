"""L1 感知追击（R146/R149）回归测试 —— 本段 = `[所有者·天平]` 线（B1）。

覆盖：
  ① **H1**：`l1_seek`/`l1_fear` 全关 ⇒ **逐位等价**（C7 对拍，基线同 `test_a_continuous` 钉死值）
  ② **开关真的生效**（开 ≠ 关，两项各自独立）
  ③ **H3 fail-loud**：`use_sim_core=True` + L1 开 ⇒ 必抛（**不得**静默换路径）
  ④ 🔴 **逐候选反退化**（R148-1 的正面预防）：seek 项**按候选格**求值 ——
     "跨候选取同值 ⇒ 该项对该个体逐位 no-op"，故 `seek_flat_frac` 是**判据**不是装饰
  ⑤ **自熄**（预注册允许结局「猎物池枯竭」）：全体高 g16 ⇒ 项恒 0 ⇒ `seek_zero_frac == 1`
  ⑥ **恐惧项**：`danger` 非空才施加（其它情况**不施加**，不是施加 0 —— 口径区分）
  ⑦ **H2 形状**：源码级断言 —— 我加的 L1 段内**不得**出现任何 `self.rng.` 调用
  ⑧ 读数口径：全关 ⇒ `l1_probe()` 返回 **None（未适用）**，不是 0（R120 铁律）
  ⑨ 开关进**配置指纹**（纪元可自证）
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from simulation.config import InfoStructureConfig, SimConfig
from simulation.sphere_engine import SphereEngine


def _engine(seek: bool = False, fear: bool = False, ticks: int = 0, *,
            use_core: bool = False, seed: int = 42, w_seek: float = 0.5,
            w_fear: float = 0.5, prey: str = "lowagg") -> SphereEngine:
    cfg = SimConfig(seed=seed)
    cfg.simulation.use_sim_core = use_core
    cfg.simulation.l1_seek = seek
    cfg.simulation.l1_fear = fear
    cfg.simulation.w_seek_max = w_seek
    cfg.simulation.w_fear = w_fear
    cfg.simulation.l1_prey_mode = prey
    cfg.population.max_count = 600
    cfg.predation.forage_tradeoff_k = 0.0
    cfg.info_structure = InfoStructureConfig(
        enabled=True, learning_rate=0.05, memory_gradient="none",
    )
    e = SphereEngine(cfg)
    for _ in range(ticks):
        if e.extinct:
            break
        e.step()
    return e


def _digest(e: SphereEngine) -> tuple[int, float]:
    return (int(e._flat.sum()), round(float(e._energy.sum()), 6))


def _row_layout(e: SphereEngine, g16_of) -> None:
    """把全体摆到**同一纬度行**（row 30 起）的连续格上，人人是移动者、能量充足。

    为什么用这个布局：任一个体的**同格邻居**被占、"上一行（row 29）"全空 ⇒
    `低攻击性场` 在候选之间**必然取不同值**（有 1 也有 0）⇒ 逐候选求值可判（见 ④）。
    `g16_of(i)` 由调用方给定（控制"谁算追猎者 / 谁算猎物 / 谁算威胁"）。

    🔴 注意项的**系数含个体自己的 g16**（`w_seek_max × g17 × perc × g16_i × lowagg_occ`）
    ⇒ **追猎者必须自己 g16 > 0**，否则它对任何猎物都不感兴趣（项恒 0）。乱设会得到
    "看起来没接线"的假象（本测试初版正是这样被自己抓到的）。
    """
    P = len(e._id)
    cols = int(e.world.cols)
    base = 30 * cols
    for i in range(P):
        e._flat[i] = base + (i % cols)
    e._genes[:, 0] = 1.0                      # MOVE_PROB = 1 ⇒ 全体都是移动者
    e._genes[:, 19] = 0.0                     # 不扎根
    e._energy[:P] = e.config.organisms.max_energy
    for i in range(P):
        e._genes[i, 16] = float(g16_of(i))    # AGGRESSION


# --------------------------------------------------------------- ① H1 / C7

def test_l1_default_off_is_bit_identical():
    """C7：L1 全关必须与**改动前**逐位一致（基线同 `test_l2_dash`/`test_a_continuous`）。"""
    assert _digest(_engine(ticks=50)) == (542646, 11197.859208), (
        "L1 关档改变了轨迹 ⇒ 破坏 H1（默认关 = 逐位等价）")


def test_l1_on_actually_changes_behaviour():
    """开档必须**真的改行为**（防"接了却没生效"—— R148-1 的同族风险）。"""
    off = _digest(_engine(ticks=50))
    assert _digest(_engine(seek=True, ticks=50)) != off, "seek 项接了但没改行为"
    assert _digest(_engine(fear=True, ticks=50)) != off, "fear 项接了但没改行为"


# --------------------------------------------------------------- ③ H3

def test_l1_with_sim_core_raises():
    """H3：L1 开 + `use_sim_core=True` ⇒ **构造期直接抛**（不静默走旧路径）。"""
    cfg = SimConfig(seed=1)
    cfg.simulation.use_sim_core = True
    cfg.simulation.l1_seek = True
    with pytest.raises(NotImplementedError, match="L1/L2"):
        SphereEngine(cfg)


# --------------------------------------------------------------- ④ 逐候选反退化

def test_seek_term_is_per_candidate_not_a_constant():
    """🔴 R148-1 的**正面预防**：seek 项若对某个体所有候选取同值 ⇒ 逐位 no-op。

    布局：同一行上「追猎者（g16=0.9）↔ 猎物（g16=0.0）」交替 ⇒ 追猎者的候选里
    **既有猎物（同格邻居）又有空格（row 29）** ⇒ 项在候选间**必然不同**。
    预期：`seek_flat_frac ≈ 0.5` —— 恰好是**猎物那半**（自身 g16=0 ⇒ 系数 0 ⇒ 项恒 0）。
    🔴 这条同时是"**项必须乘自己的 g16**"的凭证：若写成 `score + 常数`（逐个体标量），
    追猎者那半也会变 flat ⇒ `seek_flat_frac → 1.0`，测试立刻变红。
    """
    e = _engine(seek=True)
    _row_layout(e, lambda i: 0.9 if i % 2 == 0 else 0.0)
    e.step()
    pr = e.l1_probe()
    assert pr is not None and pr["dec_n"] > 0, "没有个体进入求值路径 ⇒ 测试前提不成立"
    assert pr["seek_term_mean"] > 0.0, "追猎者存在的布局下项均值必须为正（加分项）"
    assert pr["seek_flat_frac"] == pytest.approx(0.5, abs=0.06), (
        f"seek_flat_frac={pr['seek_flat_frac']} —— 只有「自身 g16=0」的那半应当 flat；"
        "若接近 1.0 则说明该行被写成了逐个体标量（R148-1 的 no-op 形态）")
    assert pr["seek_zero_frac"] == pytest.approx(0.5, abs=0.06), (
        "自熄占比应同样恰为「非追猎者」那半（项恒 0 的两种来源须分得开）")


def test_seek_self_extinction_is_counted():
    """⑤ **自熄**（预注册允许结局「猎物池枯竭」，设计稿 §3.4）：全体高 g16 ⇒ 无猎物代理。

    ⇒ 项**恒 0** ⇒ `seek_zero_frac == 1`（**逐位 no-op**）。它高**不等于**"没接线"，
    所以必须与 `seek_flat_frac` 分开报 —— 这条测试就是那个区分的凭证。
    """
    e = _engine(seek=True)
    _row_layout(e, lambda i: 0.9)
    e.step()
    pr = e.l1_probe()
    assert pr is not None and pr["dec_n"] > 0
    assert pr["seek_zero_frac"] == 1.0, "全体高 g16 却没有被判为自熄 ⇒ 计数口径错"
    assert pr["seek_term_mean"] == 0.0


def test_fear_applied_only_when_danger_present():
    """⑥ 恐惧项**只在 `danger` 非空时施加**（相邻威胁存在）——不是"施加 0"。

    布局：同一行上 g16 交替（0.9 / 0.0）⇒ **只有低 g16 那一半**看得到威胁
    （高 g16 者的同格邻居全是低 g16 ⇒ `agg_field == 0`）⇒ `fear_applied_ind_frac ≈ 0.5`；
    且其候选里既有威胁方向又有空位 ⇒ `cos` 不恒定 ⇒ `fear_flat_frac == 0`。
    """
    e = _engine(fear=True, seek=False)
    _row_layout(e, lambda i: 0.9 if i % 2 == 0 else 0.0)
    e.step()
    pr = e.l1_probe()
    assert pr is not None and pr["dec_n"] > 0
    assert pr["fear_applied_ind_frac"] == pytest.approx(0.5, abs=0.06), (
        f"fear_applied_ind_frac={pr['fear_applied_ind_frac']} —— 交替布局下**恰有一半**"
        "（低 g16 者）看得到威胁；高 g16 者的邻居全低 ⇒ 无威胁（不该被计入）")
    assert pr["fear_flat_frac"] == 0.0, "fear 项跨候选取同值 ⇒ 对它们逐位 no-op"
    assert pr["fear_term_mean"] < 0.0, "恐惧项是对威胁方向的**扣分**（均值应为负）"


def test_fear_degeneracy_detector_catches_all_dangerous_neighbourhood():
    """⑥b 反退化探测器的**有效性**：所有候选格都有威胁 ⇒ `cos ≡ 1` ⇒ 恒值（真 no-op）。

    做法：把全体堆在 5×5 小块里（块心个体 8 个邻居全被占）且**全是高 g16**
    ⇒ 块心个体的 `danger == nb` ⇒ 项恒等（`flat`）。块边个体仍有空邻居 ⇒ 恒值占比 ∈ (0,1)。
    这条测试证明"`fear_flat_frac` 高"确实对应 R149-4 描述的那个退化形态。
    """
    e = _engine(fear=True, seek=False)
    P = len(e._id)
    cols = int(e.world.cols)
    block = [(28 + dr) * cols + (30 + dc) for dr in range(5) for dc in range(5)]
    for i in range(P):
        e._flat[i] = block[i % len(block)]
    e._genes[:, 0] = 1.0
    e._genes[:, 19] = 0.0
    e._genes[:, 16] = 0.9
    e._energy[:P] = e.config.organisms.max_energy
    e.step()
    pr = e.l1_probe()
    assert pr is not None
    assert pr["fear_flat_frac"] is not None and pr["fear_flat_frac"] > 0.0, (
        "块心个体 danger==nb（cos 恒 1）却没有被判为恒值 ⇒ 探测器失效")


# --------------------------------------------------------------- ⑦ H2 形状

def test_l1_block_adds_no_rng_draws():
    """H2：**我加的** L1 代码段内不得出现 `self.rng.`（不新增、也不跳过抽取）。

    只扫本线新增的段（"R146/R149 L1 两项" → 到 `if d2_softmax:` 之前），避免把既有
    `u = self.rng.random()`（软 max 的那次抽取，正是要**保留**的形状）误判为违规。
    """
    src = Path("simulation/sphere_engine.py").read_text(encoding="utf-8")
    head = "# ── R146/R149 L1 两项（**逐候选**"
    i0 = src.index(head)
    block = src[i0:src.index("# D2-3 softmax", i0)]
    code = "\n".join(l for l in block.splitlines() if not l.strip().startswith("#"))
    assert "self.rng." not in code, "L1 段内出现 self.rng. 调用 ⇒ 违反 H2"
    # 形状保持的依据：softmax 的那次抽取仍在（跨臂随机流形状不变）
    assert "u = self.rng.random()" in src


# --------------------------------------------------------------- ⑧⑨ 口径与纪元

def test_probe_is_none_when_off():
    """全关 ⇒ `l1_probe()` 返回 **None（未适用）**，不是 0（R120/§五.12 口径铁律）。"""
    assert _engine(ticks=10).l1_probe() is None
    # 只开一项也要有读数（L1 层"部分开启"是合法的）
    assert _engine(seek=True, ticks=10).l1_probe() is not None


def test_switches_visible_for_epoch():
    """L1 开关与权重必须可从产物读回（C4）；且须进**配置指纹** ⇒ 纪元可自证。"""
    e = _engine(seek=True, fear=True, w_seek=0.25, w_fear=0.0)
    assert bool(e.config.simulation.l1_seek) is True
    assert float(e.config.simulation.w_seek_max) == 0.25
    for field in ("l1_seek", "l1_fear", "w_seek_max", "w_fear", "l1_prey_mode"):
        a, b = SimConfig(), SimConfig()
        setattr(b.simulation, field, {"l1_seek": True, "l1_fear": True,
                                      "w_seek_max": 0.25, "w_fear": 0.0,
                                      "l1_prey_mode": "any"}[field])
        assert a.fingerprint() != b.fingerprint(), f"{field} 未进指纹 ⇒ 纪元不可判"
