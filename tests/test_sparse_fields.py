"""R217 §五 #1 稀疏化 B（惰性再生 / 稀疏信号）的**逐位等价**回归。

被验对象（默认关；开 = 性能路径，声称数值逐位不变）：
* `ResourceField.enable_lazy` / `_regrow_lazy`（只结算脏格 = `_grid < _capacity`）
* `SignalField.enable_sparse` / `tick`（只推进活跃集 = `age > 0`）

覆盖：
- 单元：子集 `_regrowth_amount(tick, idx)` 与全场版**逐位一致**（核心等价声明）
- 单元：惰性 `regrow` 与全场版逐 tick 逐位一致（同随机消费/回流序列）
- 单元：稀疏 `tick` 与全场版逐 tick 逐位一致（同随机写入/清除序列）
- 前提守卫：不安全配置（ts≠1 / 光敏≠0 / 负背景倍率 / uniform）**自动拒绝启用**
- 引擎：同 seed 开关两跑 digest 逐位一致（**且断言机制真的开了**，防静默 no-op）
- 引擎：范围锁（use_sim_core ⇒ 两侧都不启用；resource_dynamics 开 ⇒ 仅**资源侧**不启用；
  R231 T-E 起信号侧与资源侧解耦 ⇒ rd 开档信号照常稀疏，对拍见 ④b）
- 引擎：rd 开档信号稀疏对拍（逐 tick digest + 快照续跑）—— R231 T-E 新增
- 快照：稀疏档 save→load 后继续跑，与不中断的同批逐位一致（派生集重建正确）
- 变异（R219 §二-2）：**漏标脏**的变异体必须让对拍变红（consume / consume_many 各一）；
  **多标脏**的变异体必须仍然全绿（"多标无害"才真成立）
"""
from __future__ import annotations

import hashlib
from contextlib import contextmanager

import numpy as np
import pytest

from simulation.config import InfoStructureConfig, SimConfig
from simulation.sphere_engine import SphereEngine
from world.light_and_temperature import LightAndTemperature
from world.resource_field import ResourceField
from world.signal_field import SignalField
from world.sphere_world import SphereWorld


# ---------------------------------------------------------------- 构造helpers

def _world_lt(rows: int = 24, cols: int = 48, rotation: int = 120):
    world = SphereWorld(rows=rows, cols=cols)
    lt = LightAndTemperature(
        world, rotation_period=rotation, t_equator=30.0, t_pole=-20.0,
        day_boost=6.0, lat_base_ref=1.0,
    )
    return world, lt


def _rf_pair(rows: int = 24, cols: int = 48, patches: int = 8, **over):
    """同一世界/LT 上构造 全场 / 惰性 两个 ResourceField（T4 同款开关）。"""
    world, lt = _world_lt(rows, cols)
    kw = dict(
        capacity_per_area=40.0, regrowth_rate=0.5, distribution="patchy",
        patch_count=patches, patch_seed=42, bg_production_zero=True,
        patch_regrowth_mult=1.195,
    )
    kw.update(over)
    full = ResourceField(world, lt, **kw)
    lazy = ResourceField(world, lt, **kw)
    return world, lt, full, lazy


def _sha(arr: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(arr).tobytes()).hexdigest()


# ---------------------------------------------------------------- ① 子集再生等价

def test_regrowth_amount_subset_bitwise_matches_full():
    """🔴 核心声明：`_regrowth_amount(t, idx)` 与全场结果在 idx 上**逐位相同**。

    这是惰性再生的全部数值风险所在（子集执行 vs 全场执行）。跨整个昼夜周期、
    多组随机子集、含空/全量子集都验一遍。
    """
    world, lt, full, _ = _rf_pair()
    n = world.n_cells
    rng = np.random.default_rng(3)
    for t in range(0, 240, 7):                       # 覆盖 2 个昼夜周期（rotation=120）
        g_full = full._regrowth_amount(t)
        for size in (0, 1, 5, 137, n):
            idx = rng.choice(n, size=size, replace=False) if size else np.zeros(0, np.int64)
            g_sub = full._regrowth_amount(t, idx)
            assert _sha(g_sub) == _sha(g_full[idx]), (
                f"子集再生与全场逐位不一致：tick={t}, size={size}"
            )


# ---------------------------------------------------------------- ② 惰性 regrow 等价

def test_lazy_regrow_bitwise_matches_full():
    """同随机"吃/回流"序列下，惰性 regrow 与全场 regrow 逐 tick 逐位一致。"""
    world, _, full, lazy = _rf_pair()
    assert lazy.enable_lazy() is True, "前提应满足（T4 同款开关）却拒绝了启用"
    patch = np.flatnonzero(full._patch_mask)
    rng = np.random.default_rng(5)
    saw_dirty = False

    for t in range(240):
        # 随机消费（含同格多只 = 重复索引）+ 随机回流（只增写入，不打脏）
        k = int(rng.integers(1, 40))
        cells = rng.choice(patch, size=k, replace=True)
        amt = float(rng.random() * 3.0)
        for rf in (full, lazy):
            rf.consume_many(cells, amt)
        d_cells = rng.choice(patch, size=5, replace=True)
        d_amt = rng.random(5) * 2.0
        for rf in (full, lazy):
            rf.deposit(d_cells, d_amt)
        if t % 9 == 0:                               # 单格消费路径也覆盖
            c = int(rng.integers(0, world.n_cells))
            for rf in (full, lazy):
                rf.consume(c, 1.0)

        full.regrow(t)
        lazy.regrow(t)
        assert _sha(full._grid) == _sha(lazy._grid), f"惰性再生与全场不一致：tick={t}"
        if lazy._dirty_mask.any():
            saw_dirty = True

    # 防静默 no-op：脏格集必须真的非空过（否则惰性分支等于没跑）
    assert saw_dirty, "整批脏格集恒空 ⇒ 惰性路径没被走到（测试空转）"


def test_lazy_enable_refuses_unsafe_configs():
    """前提不满足 ⇒ `enable_lazy()` 返回 False（保持全场路径）。"""
    # ① temp_sensitivity ≠ 1.0（pow 子集执行无同位保证）
    _, _, _, a = _rf_pair(temp_sensitivity=2.0)
    assert a.enable_lazy() is False
    # ② light_sensitivity ≠ 0（含全场归约的归一化分支）
    _, _, _, b = _rf_pair(light_sensitivity=1.0)
    assert b.enable_lazy() is False
    # ③ uniform 档（收益≈0，不是目标档）
    world, lt = _world_lt()
    c = ResourceField(world, lt, capacity_per_area=40.0, regrowth_rate=0.5)
    assert c.enable_lazy() is False
    # ④ 负背景再生倍率（负增长 ⇒ 夹取引理不成立）——patchy 非归零档可为负
    _, _, _, d = _rf_pair(patches=24, patch_regrowth_mult=6.0,
                          patch_capacity_mult=1.5, bg_production_zero=False)
    assert d._bg_regrowth_mult < 0.0, "本用例需要 bg_mult<0 才有效（构造前提）"
    assert d.enable_lazy() is False


# ---------------------------------------------------------------- ③ 稀疏信号等价

def test_sparse_signals_bitwise_matches_full():
    """同随机写入/清除序列下，稀疏 tick 与全场 tick 的 `_marks`/`_age` 逐 tick 逐位一致。"""
    world = SphereWorld(rows=24, cols=48)
    n = world.n_cells
    full = SignalField(world, duration=5)
    sp = SignalField(world, duration=5)
    sp.enable_sparse()
    rng = np.random.default_rng(11)

    for t in range(200):
        k = int(rng.integers(1, 30))
        cells = rng.integers(0, n, size=k)
        pats = rng.integers(0, 16, size=k).astype(np.uint8)
        for f in (full, sp):
            f.write_many(cells, pats)
        # 单格写路径（write）：两侧同源 ⇒ 先把随机数抽好，再各自施加
        c1, p1 = int(rng.integers(0, n)), int(rng.integers(0, 16))
        for f in (full, sp):
            f.write(c1, p1)
        if rng.random() < 0.2:                       # 显式 clear（稀疏档留陈旧项，tick 剪枝）
            c2 = int(rng.integers(0, n))
            for f in (full, sp):
                f.clear(c2)
        c3, p3 = int(rng.integers(0, n)), int(rng.integers(0, 16))
        for f in (full, sp):
            f.write(c3, p3)

        for f in (full, sp):
            f.tick()
        assert _sha(full._marks) == _sha(sp._marks), f"marks 不一致：tick={t}"
        assert _sha(full._age) == _sha(sp._age), f"age 不一致：tick={t}"

    assert sp._active_idx is not None and sp._active_idx.size == int((sp._age > 0).sum())


# ---------------------------------------------------------------- ④ 引擎级对拍

def _cfg(sparse: bool, rows: int = 60, cols: int = 120, patches: int = 30,
         pop: int = 300, ticks: int = 150) -> SimConfig:
    c = SimConfig(seed=42)
    c.world.rows, c.world.cols = rows, cols
    c.resources.distribution = "patchy"
    c.resources.bg_production_zero = True
    c.resources.patch_regrowth_mult = 1.195
    c.resources.patch_count = patches
    c.population.initial_count = pop
    c.simulation.ticks = ticks
    c.simulation.use_sim_core = False
    c.simulation.sparse_fields = sparse
    return c


def _run(cfg: SimConfig, ticks: int) -> tuple[SphereEngine, bool, bool]:
    """跑到 ticks；返回 (引擎, 是否见过非空脏格集, 是否见过活跃信号集)。"""
    e = SphereEngine(cfg)
    saw_dirty = False
    saw_active = False
    for _ in range(ticks):
        if e.extinct:
            break
        e.step()
        if e.resources._lazy and e.resources._dirty_mask.any():
            saw_dirty = True
        if e.signals._sparse and e.signals._active_idx.size > 0:
            saw_active = True
    return e, saw_dirty, saw_active


def _digest(e: SphereEngine) -> tuple:
    return (
        int(e._flat.sum()), round(float(e._energy.sum()), 6), int(e.tick),
        _sha(e.resources._grid), _sha(e.signals._marks), _sha(e.signals._age),
    )


def test_engine_sparse_matches_default_bitwise():
    """同 seed / 同配置，`sparse_fields` 开 vs 关 ⇒ 全状态逐位一致。"""
    ea, _, _ = _run(_cfg(sparse=False), 150)
    eb, saw_dirty, _ = _run(_cfg(sparse=True), 150)
    # 🔴 防静默 no-op（教训 11）：机制必须真的开了、且真的被用到
    assert eb.resources._lazy is True, "惰性再生没开 ⇒ 测试空转"
    assert eb.signals._sparse is True, "稀疏信号没开 ⇒ 测试空转"
    assert saw_dirty, "整批没有出现过脏格 ⇒ 惰性再生路径没被走到"
    assert ea.tick == eb.tick and eb.tick > 10, "两跑未同步推进（或过早灭绝）"
    assert _digest(ea) == _digest(eb), "开关两跑逐位不一致 ⇒ 稀疏化破了等价"


def test_engine_sparse_scope_guards():
    """范围锁：Rust 路径 / 资源动态开 ⇒ 不启用（机制静默退回全场）。"""
    c1 = _cfg(sparse=True)
    c1.simulation.use_sim_core = True
    e1 = SphereEngine(c1)
    assert e1._sparse_fields is False
    assert e1.resources._lazy is False and e1.signals._sparse is False

    c2 = _cfg(sparse=True)
    c2.resource_dynamics.enabled = True
    e2 = SphereEngine(c2)
    # R231 T-E：rd 开 ⇒ **仅资源侧**退场；信号侧已解耦（与 rd 正交）⇒ 照常稀疏
    assert e2._sparse_fields is False and e2.resources._lazy is False
    assert e2.signals._sparse is True

    # 资源侧前提不满足 ⇒ 只有资源退回（信号照常稀疏）
    c3 = _cfg(sparse=True)
    c3.resources.temp_sensitivity = 2.0
    e3 = SphereEngine(c3)
    assert e3._sparse_fields is True
    assert e3.resources._lazy is False
    assert e3.signals._sparse is True


def test_engine_sparse_snapshot_roundtrip(tmp_path):
    """稀疏档 save→load：派生集重建正确 ⇒ 续跑与不中断同批逐位一致。"""
    e, _, _ = _run(_cfg(sparse=True), 60)
    path = tmp_path / "sparse_snap.npz"
    e.save_snapshot(str(path))
    e2 = SphereEngine.load_snapshot(str(path))
    assert e2.resources._lazy is True, "快照恢复后惰性再生丢失 ⇒ 会静默退回（或更糟）"
    assert e2.signals._sparse is True
    assert _digest(e) == _digest(e2), "快照往返本身不一致（本测试前提）"

    for _ in range(40):
        if e.extinct:
            break
        e.step()
        e2.step()
    assert _digest(e) == _digest(e2), "恢复后续跑与不中断同批不一致 ⇒ 派生集重建有误"


# ------------------------------------------------- ④b rd 开档信号解耦（R231 T-E）


def _rd_cfg(sparse: bool, ticks: int = 150) -> SimConfig:
    """`_cfg` + `resource_dynamics` 开（T-E 目标档：rd ∧ 信号稀疏）。"""
    c = _cfg(sparse=sparse, ticks=ticks)
    c.resource_dynamics.enabled = True
    return c


def test_engine_rd_on_signal_sparse_matches_full_bitwise():
    """R231 T-E：rd 开档，信号稀疏开 vs 关 ⇒ **逐 tick** 全状态逐位一致。

    资源侧两跑均为全场（资源侧闸未放开 ⇒ `_lazy is False`）⇒ 本测试只验
    "信号稀疏在 rd 开档下不破等价"（rd 与信号活跃集正交）。
    """
    ea, eb = SphereEngine(_rd_cfg(sparse=False)), SphereEngine(_rd_cfg(sparse=True))
    # 🔴 防静默 no-op：rd 真开、信号侧真稀疏（关侧真全场）、资源侧两跑都保持全场
    assert ea._rd.enabled is True and eb._rd.enabled is True
    assert eb.signals._sparse is True and ea.signals._sparse is False
    assert ea.resources._lazy is False and eb.resources._lazy is False

    saw_active = False
    for t in range(150):
        if ea.extinct or eb.extinct:
            break
        ea.step()
        eb.step()
        saw_active |= eb.signals._active_idx.size > 0
        assert _digest(ea) == _digest(eb), f"rd 开档信号稀疏在 tick {t + 1} 破了等价"
    assert ea.tick > 10, "两跑过早结束（灭绝）⇒ 证据不足"
    assert saw_active, "信号活跃集全程为空 ⇒ 等价是平凡真（本测试空转）"


def test_engine_rd_on_signal_sparse_snapshot_roundtrip(tmp_path):
    """rd 开档稀疏信号 save→load：活跃集（派生量）重建 ⇒ 续跑与不中断同批逐位一致。"""
    e = SphereEngine(_rd_cfg(sparse=True))
    for _ in range(60):
        if e.extinct:
            break
        e.step()
    assert e.tick > 10, "过早灭绝 ⇒ 快照证据不足"
    path = tmp_path / "rd_signal_snap.npz"
    e.save_snapshot(str(path))
    e2 = SphereEngine.load_snapshot(str(path))
    assert e2.signals._sparse is True, "快照恢复后信号稀疏丢失（闸未随配置还原）"
    assert e2.resources._lazy is False, "rd 档资源侧应保持全场（现状语义）"
    assert _digest(e) == _digest(e2), "快照往返本身不一致（本测试前提）"

    for _ in range(40):
        if e.extinct:
            break
        e.step()
        e2.step()
    assert _digest(e) == _digest(e2), "恢复后续跑与不中断同批不一致 ⇒ 活跃集重建有误"


def test_c7_digest_identical_with_sparse_switch():
    """C7 钉死 digest `(574887, 11266.746993)` 与 `sparse_fields` 开关**无关**。

    复刻 `tests/test_a_continuous.py::test_k0_is_bitwise_equivalent_to_baseline`
    的配置（seed 42 / max_count 600 / D2 enabled / memory_gradient=none / k=0）。
    ⚠️ 该配置 resources 是 **uniform** ⇒ `enable_lazy()` 按前提④拒绝（全场路径）；
    本测试证明的是**信号稀疏**在 live 跑中不破 C7 基线（另一条测试补 patchy 侧）。
    """
    def _run(sparse: bool):
        cfg = SimConfig(seed=42)
        cfg.simulation.use_sim_core = False
        cfg.simulation.sparse_fields = sparse
        cfg.population.max_count = 600
        cfg.predation.forage_tradeoff_k = 0.0
        cfg.info_structure = InfoStructureConfig(
            enabled=True, learning_rate=0.05, memory_gradient="none",
        )
        e = SphereEngine(cfg)
        saw_active = 0
        for _ in range(50):
            if e.extinct:
                break
            e.step()
            if e.signals._sparse:
                saw_active += int(e.signals._active_idx.size > 0)
        return e, (int(e._flat.sum()), round(float(e._energy.sum()), 6)), saw_active

    _, d0, _ = _run(False)
    e1, d1, saw1 = _run(True)
    assert d0 == (574887, 11266.746993), f"C7 默认档基线漂了（本测试前提）：{d0}"
    assert e1.signals._sparse is True and e1.resources._lazy is False
    assert saw1 > 0, "信号稀疏在本次跑中全程空转 ⇒ 等价是平凡真，构不成证据"
    assert d1 == d0, f"稀疏档 digest 与默认档不同：{d1} vs {d0}"


# ---------------------------------------------------------------- ⑤ 变异测试（R219 §二-2）
#
# "漏标脏"是本机制**唯一**的正确性风险 ⇒ 必须证明打脏点**承重**（删掉 ⇒ 对拍红），
# 且"多标脏无害"真的成立（多标 ⇒ 仍绿）。手工清单不是验收物（教训 ⑯）。

class _MaskSink:
    """哑掩码：`__setitem__` 吞掉写入 ⇒ 精确模拟"漏标脏"（其余行为不变）。"""

    def __setitem__(self, key, value):   # noqa: ANN001
        pass


@contextmanager
def _mutant(kind: str):
    """变异体上下文：'drop_consume' / 'drop_consume_many'（漏标）/ 'over_mark'（多标）。

    产出计数器 dict（`n` = 变异体被走到的次数）—— 供"测试非空转"断言。
    """
    cnt = {"n": 0}
    orig_r = ResourceField.regrow
    origs = {name: getattr(ResourceField, name) for name in ("consume", "consume_many")}

    if kind.startswith("drop_"):
        name = kind[len("drop_"):]
        orig = origs[name]

        def wrapper(self, *a, **kw):
            cnt["n"] += 1
            real, self._dirty_mask = self._dirty_mask, _MaskSink()
            try:
                return orig(self, *a, **kw)
            finally:
                self._dirty_mask = real

        setattr(ResourceField, name, wrapper)
    elif kind == "over_mark":
        def wrapper(self, tick, *a, **kw):
            cnt["n"] += 1
            if self._lazy:
                self._dirty_mask[::3] = True        # 多标：一大批净格也被标脏
            return orig_r(self, tick, *a, **kw)

        ResourceField.regrow = wrapper
    else:
        raise ValueError(f"未知变异体：{kind}")
    try:
        yield cnt
    finally:
        ResourceField.regrow = orig_r
        for name, orig in origs.items():
            setattr(ResourceField, name, orig)


def _pair_run(ticks: int = 60) -> tuple[str, str]:
    """同随机"吃/回流"序列跑全场 vs 惰性两个场；返回末态 `_grid` sha 对（**不做相等断言**）。"""
    world, _, full, lazy = _rf_pair()
    assert lazy.enable_lazy() is True
    patch = np.flatnonzero(full._patch_mask)
    rng = np.random.default_rng(5)
    for t in range(ticks):
        k = int(rng.integers(1, 40))
        cells = rng.choice(patch, size=k, replace=True)
        amt = float(rng.random() * 3.0)
        for rf in (full, lazy):
            rf.consume_many(cells, amt)
        d_cells = rng.choice(patch, size=5, replace=True)
        d_amt = rng.random(5) * 2.0
        for rf in (full, lazy):
            rf.deposit(d_cells, d_amt)
        if t % 9 == 0:
            c = int(rng.integers(0, world.n_cells))
            for rf in (full, lazy):
                rf.consume(c, 1.0)
        full.regrow(t)
        lazy.regrow(t)
    return _sha(full._grid), _sha(lazy._grid)


def test_mutation_dropped_consume_many_marking_breaks_equivalence():
    """变异体①：吞掉 `consume_many` 的打脏写 ⇒ 逐位对拍**必须红**。"""
    with _mutant("drop_consume_many") as cnt:
        sha_full, sha_lazy = _pair_run()
    assert cnt["n"] > 0, "变异体没被走到 ⇒ 本测试无效（场景没覆盖该路径）"
    assert sha_full != sha_lazy, (
        "吞掉 consume_many 的打脏后仍逐位一致 ⇒ 该打脏点不承重（或惰性路径没生效）"
    )


def test_mutation_dropped_consume_marking_breaks_equivalence():
    """变异体②：吞掉 `consume`（单格路径）的打脏写 ⇒ 对拍**必须红**。"""
    with _mutant("drop_consume") as cnt:
        sha_full, sha_lazy = _pair_run()
    assert cnt["n"] > 0, "变异体没被走到 ⇒ 本测试无效"
    assert sha_full != sha_lazy, "吞掉 consume 的打脏后仍逐位一致 ⇒ 该打脏点不承重"


def test_mutation_over_marking_stays_equivalent():
    """变异体③：**多标脏**（每 tick 把 1/3 净格也标脏）⇒ 数值**仍须逐位一致**。"""
    with _mutant("over_mark") as cnt:
        sha_full, sha_lazy = _pair_run()
    assert cnt["n"] > 0, "变异体没被走到 ⇒ 本测试无效"
    assert sha_full == sha_lazy, (
        "多标脏改变了数值 ⇒ 夹取引理（净格再结算一次是逐位 no-op）不成立"
    )
