"""rd-A2 子集化靶向单测（R294 交付；白盒 + 逐位对拍）。

被验对象 = A2 六项里承重的子集化路径（其余两项 ①子集清零/⑥引擎侧回写跳过
由引擎级 A/B 用例覆盖）：
  ① `growth_multiplier(idx)` 子集版 ≡ 全场版（dead 段 → rest 段 `min`，次序同上）
  ② `capacity_from_base` 缓存 + `invalidate_capacity`（掩码版本号；掩码写点不漏失效）
  ③ `note_tick(eaten_idx=...)` 子集全链 ≡ `eaten_idx=None` 全链（含 `_rng` 状态逐位）
  ④ `rotate` 子集扫描 ≡ **参考全场实现**（inline 复刻 main 版，逐行同式）
     + 派生索引（`_dead_idx`/`_rest_idx`）不变量与陈旧项自愈
  ⑤ 引擎真跑：`indexes_consistent()` 全程成立 + 快照恢复后重建正确
  ⑥ 引擎级 A/B：rd 开档 `sparse_fields` on/off **逐 tick digest 逐位一致**
     （≥2000 tick；含死亡/休耕/搬移/回写跳过的真实路径）

纪律：每个用例都断言"目标路径真的被走到"（防静默空转），且尽量与
"参考实现/全场路径"对拍而不是只对常量。
"""
from __future__ import annotations

import hashlib

import numpy as np
import pytest

from simulation.config import SimConfig
from simulation.sphere_engine import SphereEngine
from world.light_and_temperature import LightAndTemperature
from world.resource_dynamics import ResourceDynamics
from world.resource_field import ResourceField
from world.sphere_world import SphereWorld


# ---------------------------------------------------------------- 构造 helpers

def _world_lt(rows: int = 24, cols: int = 48, rotation: int = 120):
    world = SphereWorld(rows=rows, cols=cols)
    lt = LightAndTemperature(
        world, rotation_period=rotation, t_equator=30.0, t_pole=-20.0,
        day_boost=6.0, lat_base_ref=1.0,
    )
    return world, lt


def _mk_rf_and_rd(rows: int = 24, cols: int = 48, patches: int = 8, **over):
    """一个小世界 + patchy 资源场 + 由其派生的 rd（构造确定性：无 RNG 消费）。"""
    world, lt = _world_lt(rows, cols)
    kw = dict(
        capacity_per_area=40.0, regrowth_rate=0.5, distribution="patchy",
        patch_count=patches, patch_seed=42, bg_production_zero=True,
        patch_regrowth_mult=1.195,
    )
    kw.update(over)
    rf = ResourceField(world, lt, **kw)
    cfg = SimConfig(seed=42).resource_dynamics
    cfg.enabled = True
    rd = ResourceDynamics.from_field(cfg, rf, world)
    return world, rf, rd


def _sha(arr: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(arr).tobytes()).hexdigest()


def _state(rd: ResourceDynamics) -> dict:
    return {
        "damage": rd._damage, "dead": rd._dead, "dead_since": rd._dead_since,
        "rest_until": rd._rest_until, "mask": rd._mask,
        "dead_idx": rd._dead_idx, "rest_idx": rd._rest_idx,
        "counters": (rd.patch_kill_n, rd.patch_reborn_n, rd.forced_reborn_n,
                     rd.promote_n, rd.rest_set_n),
        "rng": rd._rng.bit_generator.state,
    }


def _assert_state_equal(a: ResourceDynamics, b: ResourceDynamics, tag: str,
                        idx: bool = True) -> None:
    """逐位对拍两 rd；`idx=False` 时跳过派生索引（参考实现不维护它们，见 rotate 用例）。"""
    sa, sb = _state(a), _state(b)
    for key in sa:
        if not idx and key in ("dead_idx", "rest_idx"):
            continue
        va, vb = sa[key], sb[key]
        if isinstance(va, np.ndarray):
            assert va.shape == vb.shape and np.array_equal(va, vb), (
                f"{tag}: `{key}` 不等（{int((va != vb).sum()) if va.shape == vb.shape else 'N/A'} 处差异）")
        else:
            assert va == vb, f"{tag}: `{key}` {va!r} != {vb!r}"


def _mk_pools(rd: ResourceDynamics, seed: int = 23, n_rest: int = 40, n_kill: int = 20):
    """固定两池斑块格：休耕池（0.04×容量/次 ⇒ t≈8 累计到 0.3 进休耕，比率<kill_frac 不死）
    与死亡池（1.2×容量 ⇒ 主通道瞬死）。确定性（固定 seed）。"""
    patch = np.flatnonzero(rd._mask)
    assert patch.size >= n_rest + n_kill, "斑块格不够，用例前提不成立"
    pick = np.random.default_rng(seed).choice(patch, size=n_rest + n_kill, replace=False)
    return np.sort(pick[:n_rest]), np.sort(pick[n_rest:])


def _rotate_reference(rd: ResourceDynamics, tick: int) -> None:
    """A2 之前的 **rotate 全场扫描实现**（`git show main:world/resource_dynamics.py`
    的 ②③ 段逐行复刻）——供子集版对拍；不做派生索引维护。"""
    if not rd.enabled:
        return
    if rd._demoted.any():
        for i in np.flatnonzero(rd._demoted):
            rd._promote_same_row(int(i))
        rd._demoted[:] = False
    expired = (rd._rest_until >= 0) & (~rd._dead) & (tick >= rd._rest_until)
    if expired.any():
        rd._damage[expired] *= rd.damage_recovery
        rd._rest_until[expired] = -1
    due = rd._dead & ((tick - rd._dead_since) >= rd.dead_regen_ticks)
    if due.any():
        n_due = int(due.sum())
        rd._dead[due] = False
        rd._rest_until[due] = -1
        rd._dead_since[due] = -1
        rd.patch_reborn_n += n_due
    max_dead = int(rd.dead_cell_max_frac * rd.n_cells)
    n_dead = int(rd._dead.sum())
    if n_dead > max_dead:
        need = n_dead - max_dead
        cand = np.flatnonzero(rd._dead)
        order = cand[np.argsort(rd._dead_since[cand], kind="stable")]
        force = order[:need]
        rd._dead[force] = False
        rd._dead_since[force] = -1
        rd._rest_until[force] = -1
        rd.forced_reborn_n += int(force.size)


def _draw_intake(rf: ResourceField, n: int, rest_pool: np.ndarray, kill_pool: np.ndarray):
    """确定性取食：休耕池 0.04×容量（累计到 0.3 ⇒ 休耕、比率 < kill_frac 不死）；
    死亡池 1.2×容量（主通道瞬死）；其余格零。"""
    intake = np.zeros(n, dtype=np.float64)
    intake[rest_pool] = rf._capacity[rest_pool] * 0.04
    intake[kill_pool] = rf._capacity[kill_pool] * 1.2
    return intake


# ---------------------------------------------------- ① growth_multiplier 子集

def test_growth_multiplier_subset_bitwise_matches_full():
    """`growth_multiplier(idx)` 与全场版在 idx 上逐位相同（含"既死又休耕"的 min 段）。"""
    world, _, rd = _mk_rf_and_rd()
    assert rd.enabled and rd.bg_production_zero, "用例前提：bgzero 档（dead/rest 乘子 0.5）"
    n = world.n_cells
    rng = np.random.default_rng(7)
    idx_dead = rng.choice(n, size=37, replace=False)
    idx_rest = rng.choice(n, size=23, replace=False)
    rd._dead[idx_dead] = True
    rd._rest_until[idx_rest] = 100
    both = np.intersect1d(idx_dead, idx_rest)
    assert both.size > 0, "本用例需要'既死又休耕'格（min 段承重）"
    rd.rebuild_indexes()

    full = rd.growth_multiplier()
    assert (full[idx_dead] == rd.dead_regen_mult).all(), "死格乘子没生效"
    assert (full[both] == min(rd.dead_regen_mult, rd.rest_regen_mult)), \
        "min(dead, rest) 段没生效"
    assert (full != 1.0).any(), "本用例必须有非 1 乘子（防 no-op）"

    for size in (0, 1, 5, 137, n):
        idx = (np.sort(rng.choice(n, size=size, replace=False)) if size
               else np.zeros(0, dtype=np.int64))
        sub = rd.growth_multiplier(idx)
        assert sub.shape == idx.shape
        assert _sha(sub) == _sha(full[idx]), f"子集乘子与全场不一致：size={size}"

    rd.enabled = False
    assert (rd.growth_multiplier(np.arange(n)) == 1.0).all(), "关档应恒 1"


# ------------------------------------------------- ② capacity 缓存 + 失效写点

def test_capacity_cache_identity_and_invalidation():
    """缓存本体复用（零重算）；`invalidate_capacity` 版本号 +1 且反映新掩码。"""
    world, _, rd = _mk_rf_and_rd()
    c1 = rd.capacity_from_base()
    assert c1 is rd.capacity_from_base(), "第二次调用应命中缓存（本体复用）"
    v0 = rd._mask_ver

    i = int(np.flatnonzero(rd._mask)[0])
    rd._mask[i] = False           # 绕过写点直改（契约要求调用方自行失效）
    assert rd.capacity_from_base() is c1, "直改未失效时缓存应保持（合同语义）"

    rd.invalidate_capacity()
    c2 = rd.capacity_from_base()
    assert c2 is not c1 and rd._mask_ver == v0 + 1, "失效未生效"
    assert c2[i] == rd.bg_capacity_mult * rd._base_capacity[i], "失效后未反映新掩码"


def test_capacity_invalidated_by_note_tick_promote():
    """`note_tick` 死亡搬移路径（bgzero=False ⇒ 允许搬）不得漏失效容量缓存。"""
    world, rf, rd = _mk_rf_and_rd(bg_production_zero=False)
    assert not rd.bg_production_zero and rd.rotate_moves_patch
    n = world.n_cells
    patch = np.flatnonzero(rd._mask)
    cap_before = rd.capacity_from_base().copy()
    v0 = rd._mask_ver

    intake = np.zeros(n, dtype=np.float64)
    intake[patch[:3]] = cap_before[patch[:3]] * 1.5    # 主通道（damage≥0.8）致死
    rd.note_tick(intake, np.ones(n), tick=1)
    assert rd.patch_kill_n >= 3 and rd.promote_n > 0, "搬移路径没被走到（用例空转）"
    assert rd._mask_ver > v0, "掩码变了但版本号没动 ⇒ 缓存会静默过期"
    cap_after = rd.capacity_from_base()
    assert not np.array_equal(cap_before, cap_after), "搬移后容量应改变"
    assert int(cap_after.sum() > 0)


def test_capacity_invalidated_by_promote_same_row_direct():
    """`_promote_same_row` 自身也是掩码写点（含 rotate 防御补搬路径）⇒ 必须失效。"""
    world, _, rd = _mk_rf_and_rd(bg_production_zero=False)
    rd.capacity_from_base()          # 预热缓存
    v0 = rd._mask_ver
    i = int(np.flatnonzero(rd._mask)[0])
    assert rd._promote_same_row(i) is True, "应能找到同行候选格"
    assert rd._mask_ver == v0 + 1 and rd.promote_n == 1


# --------------------------------------------------- ③ note_tick 子集 ≡ 全场

@pytest.mark.parametrize("bgzero", [True, False])
def test_note_tick_subset_bitwise_matches_full(bgzero: bool):
    """同输入下 `eaten_idx=flatnonzero(intake)` ≡ `None`（含搬移 ⇒ `_rng` 状态逐位）。"""
    world, rf, rd_full = _mk_rf_and_rd(bg_production_zero=bgzero)
    _, _, rd_sub = _mk_rf_and_rd(bg_production_zero=bgzero)
    n = world.n_cells
    rest_pool, kill_pool = _mk_pools(rd_full)
    saw_dead = saw_rest = saw_move = False

    for t in range(1, 121):
        intake = _draw_intake(rf, n, rest_pool, kill_pool)
        growth = rf._regrowth_amount(t)
        rd_full.note_tick(intake, growth, t)
        rd_sub.note_tick(intake, growth, t, eaten_idx=np.flatnonzero(intake))
        _assert_state_equal(rd_full, rd_sub, f"bgzero={bgzero} t={t}")
        saw_dead |= rd_full.patch_kill_n > 0
        saw_rest |= rd_full.rest_set_n > 0
        saw_move |= rd_full.promote_n > 0

    assert saw_dead, "整批无死亡 ⇒ 判死路径没被走到（用例空转）"
    assert saw_rest, "整批无休耕 ⇒ 休耕路径没被走到（用例空转）"
    assert saw_move == (not bgzero), (
        f"bgzero={bgzero} 时搬移发生与否不符（True 档不应搬 / False 档应搬）")


def test_note_tick_empty_subset_is_noop():
    """空子集（intake 全 0）⇒ 直接返回，不触碰任何状态（含 `_rng`）。"""
    _, _, rd = _mk_rf_and_rd()
    n = rd.n_cells
    before = _state(rd)
    rd.note_tick(np.zeros(n), np.zeros(n), tick=1, eaten_idx=np.zeros(0, np.int64))
    after = _state(rd)
    for key in before:
        va, vb = before[key], after[key]
        if isinstance(va, np.ndarray):
            assert np.array_equal(va, vb), f"空子集动了 `{key}`"
        else:
            assert va == vb, f"空子集动了 `{key}`"


# ----------------------------------------------- ④ rotate 子集 ≡ 参考全场实现

@pytest.mark.parametrize("scenario,regen,frac,n_kill", [
    ("due", 5, 0.9, 20),            # 到期重生为主（死格 5 tick 后重入）
    ("force", 10 ** 9, 0.02, 48),   # 反荒漠化闸为主（闸阈 = 23 格 ⇒ 死 48 格逼闸）
])
def test_rotate_subset_matches_reference_full_scan(scenario: str, regen: int, frac: float,
                                                   n_kill: int):
    """子集 rotate ≡ main 版全场扫描（含到期恢复/重生/强制重生 + 索引不变量）。"""
    world, rf, rd_sub = _mk_rf_and_rd()
    _, _, rd_ref = _mk_rf_and_rd()
    for rd in (rd_sub, rd_ref):
        rd.dead_regen_ticks = regen
        rd.dead_cell_max_frac = frac
        rd.rest_ticks = 3
    n = world.n_cells
    rest_pool, kill_pool = _mk_pools(rd_sub, n_kill=n_kill)
    saw_expired = saw_due = saw_force = False

    for t in range(1, 201):
        intake = _draw_intake(rf, n, rest_pool, kill_pool)
        growth = rf._regrowth_amount(t)
        eaten = np.flatnonzero(intake)
        rd_sub.note_tick(intake, growth, t, eaten_idx=eaten)
        rd_ref.note_tick(intake, growth, t, eaten_idx=eaten)
        rest_before = int((rd_sub._rest_until >= 0).sum())
        rd_sub.rotate(t)
        _rotate_reference(rd_ref, t)
        _assert_state_equal(rd_sub, rd_ref, f"{scenario} t={t}", idx=False)
        assert rd_sub.indexes_consistent(), f"{scenario} t={t}: 派生索引破不变量"
        saw_expired |= rest_before > 0 and int((rd_sub._rest_until >= 0).sum()) < rest_before
        saw_due |= rd_sub.patch_reborn_n > 0
        saw_force |= rd_sub.forced_reborn_n > 0

    assert saw_expired, f"{scenario}: 无休耕到期事件（用例空转）"
    if scenario == "due":
        assert saw_due, "due 场景没走到到期重生"
    else:
        assert saw_force, "force 场景没走到反荒漠化闸"


def test_rotate_index_self_heal_drops_stale_entries():
    """陈旧索引项（外部直改/旧档遗留）⇒ rotate 自愈剔除，且结果仍 ≡ 参考实现。"""
    world, rf, rd_sub = _mk_rf_and_rd()
    _, _, rd_ref = _mk_rf_and_rd()
    for rd in (rd_sub, rd_ref):
        rd.dead_regen_ticks = 5
        rd.rest_ticks = 3
    n = world.n_cells
    rest_pool, kill_pool = _mk_pools(rd_sub)
    t = 0
    while t < 40:
        t += 1
        intake = _draw_intake(rf, n, rest_pool, kill_pool)
        growth = rf._regrowth_amount(t)
        eaten = np.flatnonzero(intake)
        rd_sub.note_tick(intake, growth, t, eaten_idx=eaten)
        rd_ref.note_tick(intake, growth, t, eaten_idx=eaten)
        rd_sub.rotate(t)
        _rotate_reference(rd_ref, t)
    assert rd_sub.patch_kill_n > 0, "前置：需要已有死亡记录"

    # 注入两类陈旧项：休耕索引里的非休耕格 / 死亡索引里的非死格
    live = np.flatnonzero((rd_sub._rest_until < 0) & (~rd_sub._dead))
    assert live.size >= 2
    stale_rest = int(live[-1])
    stale_dead = int(live[-2])
    rd_sub._rest_idx = np.append(rd_sub._rest_idx, stale_rest)
    rd_sub._dead_idx = np.append(rd_sub._dead_idx, stale_dead)
    assert not rd_sub.indexes_consistent(), "注入后应暂时破不变量（前提）"

    for _ in range(20):
        t += 1
        intake = _draw_intake(rf, n, rest_pool, kill_pool)
        growth = rf._regrowth_amount(t)
        eaten = np.flatnonzero(intake)
        rd_sub.note_tick(intake, growth, t, eaten_idx=eaten)
        rd_ref.note_tick(intake, growth, t, eaten_idx=eaten)
        rd_sub.rotate(t)
        _rotate_reference(rd_ref, t)
        _assert_state_equal(rd_sub, rd_ref, f"自愈 t={t}", idx=False)
        assert rd_sub.indexes_consistent(), f"自愈 t={t}: 陈旧项没被剔除"


# ------------------------------------- ⑤ 引擎真跑：不变量 + 快照重建

def _engine_cfg(sparse: bool, rows: int = 24, cols: int = 48, patches: int = 8,
                pop: int = 150) -> SimConfig:
    c = SimConfig(seed=42)
    c.world.rows, c.world.cols = rows, cols
    c.resources.distribution = "patchy"
    c.resources.bg_production_zero = True
    c.resources.patch_regrowth_mult = 1.195
    c.resources.patch_count = patches
    c.population.initial_count = pop
    c.simulation.use_sim_core = False
    c.simulation.sparse_fields = sparse
    c.resource_dynamics.enabled = True
    return c


def test_engine_indexes_consistent_and_snapshot_rebuild(tmp_path):
    """真引擎跑（rd 开）：索引不变量全程成立；快照恢复后重建正确且状态逐位。"""
    e = SphereEngine(_engine_cfg(sparse=False))
    rd = e._rd
    for _ in range(200):
        if e.extinct:
            break
        e.step()
        assert rd.indexes_consistent(), f"tick={e.tick}: 引擎路径漏维护派生索引"
    assert e.tick > 20, "过早灭绝 ⇒ 证据不足"
    assert rd.patch_kill_n > 0 or rd.rest_set_n > 0, (
        "整跑无死亡无休耕 ⇒ 索引路径没被走到（用例空转）")

    p = tmp_path / "rd_a2_idx.npz"
    e.save_snapshot(str(p))
    r = SphereEngine.load_snapshot(str(p))
    assert r._rd.indexes_consistent(), "load 后未重建派生索引"
    _assert_state_equal(e._rd, r._rd, "快照往返")

    for _ in range(30):
        if e.extinct:
            break
        e.step()
        r.step()
        assert rd.indexes_consistent() and r._rd.indexes_consistent()
        _assert_state_equal(e._rd, r._rd, "续跑")


# ------------------------------------- ⑥ 引擎级 A/B：sparse on/off 逐位一致

def _digest(e: SphereEngine) -> tuple:
    rd = e._rd
    return (e.tick, len(e._id), int(e._flat.sum()), round(float(e._energy.sum()), 6),
            _sha(e.resources._grid), _sha(e.resources._capacity),
            _sha(e.resources._patch_mask), _sha(rd._mask), _sha(rd._dead),
            _sha(rd._rest_until), _sha(rd._damage),
            (rd.patch_kill_n, rd.patch_reborn_n, rd.forced_reborn_n,
             rd.promote_n, rd.rest_set_n))


def test_engine_rd_sparse_ab_2000_ticks_bitwise():
    """rd 开档：`sparse_fields` on/off 2000 tick 逐 tick digest 逐位一致。

    稀疏侧同时吃到 A2 的全部子集路径（子集清零 / 缓存 gather / note_tick 子集 /
    rotate 索引扫描 / growth_multiplier(idx) / 回写跳过）；关侧 = 全场基线。
    末段断言机制非空转（脏格/死亡/休耕/回写版本变化都真发生过）。
    """
    ea, eb = SphereEngine(_engine_cfg(sparse=False)), SphereEngine(_engine_cfg(sparse=True))
    assert ea._rd.enabled and eb._rd.enabled
    assert ea.resources._lazy is False and eb.resources._lazy is True, (
        "开侧资源惰性没开 ⇒ A2 路径没被走到（测试空转）")
    saw_dirty = saw_dead = saw_rest = False
    for t in range(2000):
        if ea.extinct or eb.extinct:
            break
        ea.step()
        eb.step()
        saw_dirty |= bool(eb.resources._dirty_mask.any())
        saw_dead |= eb._rd.patch_kill_n > 0
        saw_rest |= eb._rd.rest_set_n > 0
        assert _digest(ea) == _digest(eb), f"rd 开档 A/B 在 tick {t + 1} 破了逐位等价"
    assert ea.tick >= 2000, f"两跑提前结束（tick={ea.tick}，灭绝={ea.extinct}）⇒ 证据不足"
    assert saw_dirty and saw_dead and saw_rest, (
        f"机制空转（dirty={saw_dirty} dead={saw_dead} rest={saw_rest}）⇒ 等价是平凡真")
