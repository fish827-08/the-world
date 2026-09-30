"""T1b：记忆 v2 移动段的 cos + 内联零散段**批量化** —— 回归单测（R294 裁实施）。

交付对拍矩阵（与评估稿/分支交付口径一致，常驻化）：
    C1 引擎级逐位等价：同种子双引擎（T1b 开 vs `_batch_move_on=False` 参考循环）
       ⇒ 全状态数组逐位相等（`_flat`/`_heading`/`_mem_az/_dist/_tick/_degraded`/
       能量/基因/age/id）——整条轨迹（含死亡/繁殖）不漂移。
    C2 cos 批量 == 标量（逐行逐位）：手造记忆态（空槽/新鲜/过期/降级 × heading ∈
       {-1, 0..7} × 记忆权重基因 关/开）⇒ `_memory_egocentric_cos_batch` 每行与
       `_memory_egocentric_cos` 逐位相等；全空槽行 == 0（对齐标量提前返回）。
    C3 邻居表等价：标准行（2..rows-3）**全格** `_nb_table[cell]` ≡ `neighbors(cell)`
       —— 批量 gather 的前提（原型里是 460,800 格穷举；此处常驻小世界版）。
    C4 闸门：任一异质机制开启（示例 = HM）⇒ T1b 不执行（计数恒 0）。
    C5 非空转：T1b 计数 > 0 且 `_mem_az` 有写入（防"测了个寂寞"，DEL-8 教训）。
"""

from __future__ import annotations

import numpy as np

from simulation.config import SimConfig
from simulation.sphere_engine import SphereEngine


def _mk_engine(*, seed: int = 42, max_count: int = 600, hm: bool = False) -> SphereEngine:
    """S3 装置（patchy + v2 记忆；**复刻 R288/60k 装置口径**）。

    ⚠️ 与 `test_memory_v2_egocentric._engine` 的差别：此处 `info_structure.enabled`
    保持 **False**（只翻 `memory_v2`）——这正是 60k 装置与 T1b 原型探针的口径
    （`enabled=True` 会带出 radius=4/noise=0.05/tau=0.15 的 D2 默认 ⇒ T1b 闸门关）。
    """
    cfg = SimConfig(seed=seed)
    cfg.simulation.use_sim_core = False        # v2 与 Rust 互斥（H3 硬报错）
    cfg.simulation.l2_dash = False
    cfg.population.max_count = max_count
    cfg.hunger_mod.enabled = hm
    cfg.info_structure.memory_v2 = True        # 不受 enabled 门控（S3 单变量规格）
    cfg.info_structure.memory_gradient = "orientation"
    c_ = cfg.resources
    c_.distribution = "patchy"
    c_.bg_production_zero = True
    c_.patch_count = 30
    c_.patch_radius = 2
    c_.patch_regrowth_mult = 1.195
    o_ = cfg.organisms
    o_.max_energy = 600.0
    o_.initial_energy = 300.0
    o_.starve_frac = 0.30
    o_.exhaust_frac = 0.17
    o_.eat_efficiency = 7.5
    o_.assim_herb = 0.4
    o_.assim_carn = 0.8
    o_.stomach_cap_mass = 25.0
    o_.eat_threshold_frac = 0.6
    o_.photo_max = 0.0
    o_.far_cap = 32
    return SphereEngine(cfg)


# ---------------------------------------------------------------- C1 引擎级逐位等价

def test_c1_engine_ab_bitwise():
    """同种子双引擎 250 tick：T1b 开 ≡ 参考循环（全状态逐位）。"""
    a = _mk_engine(seed=42)
    b = _mk_engine(seed=42)
    b._batch_move_on = False                   # 纯参考循环（对拍基线）
    for _ in range(250):
        if a.extinct or b.extinct:
            break
        a.step()
        b.step()
    # C5 非空转（防线先行）：T1b 真被执行、记忆真被写入
    assert a._t1b_batched_n > 0, "T1b 未被触发（闸门语义或装置不对，测试无效）"
    assert b._t1b_batched_n == 0, "基线引擎不应走 T1b"
    assert (a._mem_az >= 0).any(), "记忆槽未被写入（cos 段可能空转，测试无效）"
    assert bool(a.extinct) == bool(b.extinct)
    for name in ("_flat", "_heading", "_mem_az", "_mem_dist", "_mem_tick",
                 "_mem_degraded", "_energy", "_genes", "_age", "_id",
                 "_trust", "_giveup_ct"):
        x, y = getattr(a, name), getattr(b, name)
        assert np.array_equal(x, y), f"{name} 不逐位一致（T1b 与参考循环分岔）"
    assert int(a.tick) == int(b.tick)


# ---------------------------------------------------------------- C2 cos 批量 == 标量

def test_c2_cos_batch_bitwise_vs_scalar():
    """手造记忆态（空/鲜/过期/降级 × heading × 权重基因）逐位对拍。"""
    e = _mk_engine(seed=7)
    rng = np.random.default_rng(11)
    n = e._flat.size
    sel = rng.choice(n, size=min(400, n), replace=False)
    cells = e._flat[sel]
    rows = cells // e.world.cols
    keep = (rows >= 2) & (rows <= e.world.rows - 3)      # 批量契约 = 标准行
    sel, cells = sel[keep], cells[keep]
    assert sel.size > 100, "标准行样本过少（测试装置异常）"
    e._mem_az[sel] = rng.integers(-1, 8, size=(sel.size, 4)).astype(np.int8)
    e._mem_tick[sel] = e._tick - rng.integers(
        -50, e._mem_ttl + 200, size=(sel.size, 4))
    e._mem_dist[sel] = rng.uniform(0.0, 60.0, size=(sel.size, 4)).astype(np.float32)
    e._mem_degraded[sel] = rng.random((sel.size, 4)) < 0.3
    hd = rng.integers(-1, 8, size=sel.size)
    e._heading[sel] = hd
    nb = e._nb_table[cells]

    def _check():
        G = e._memory_egocentric_cos_batch(sel, nb, hd)
        bad = 0
        for i in range(sel.size):
            g0 = e._memory_egocentric_cos(
                int(sel[i]), int(cells[i]), nb[i], int(hd[i]))
            if not np.array_equal(g0, G[i]):
                bad += 1
        assert bad == 0, f"批量 cos 与标量不逐位一致：{bad}/{sel.size} 行"

    _check()                                   # 权重基因关（默认）
    e._mem_weight_gene = True                  # 13.11 乘子分支（g22 已随机初始化）
    _check()
    # 全空槽行（含过期）== 0（对齐标量提前返回语义）
    e._mem_weight_gene = False
    e._mem_az[sel[0]] = -1
    e._mem_tick[sel[0]] = -1
    G = e._memory_egocentric_cos_batch(sel[:1], nb[:1], hd[:1])
    assert float(np.abs(G).max()) == 0.0
    g0 = e._memory_egocentric_cos(
        int(sel[0]), int(cells[0]), nb[0], int(hd[0]))
    assert float(np.abs(g0).max()) == 0.0
    # 全过期：TTL 之外的槽不得贡献（批量与标量同）
    e._mem_az[sel[1]] = 3
    e._mem_tick[sel[1]] = e._tick - e._mem_ttl - 5
    G = e._memory_egocentric_cos_batch(sel[1:2], nb[1:2], hd[1:2])
    g0 = e._memory_egocentric_cos(
        int(sel[1]), int(cells[1]), nb[1], int(hd[1]))
    assert np.array_equal(g0, G[0]) and float(np.abs(G).max()) == 0.0


# ---------------------------------------------------------------- C3 邻居表等价

def test_c3_nb_table_equiv_std_rows():
    """标准行全格：`_nb_table` 与 `neighbors()` 逐位一致（gather 前提常驻化）。"""
    e = _mk_engine(seed=1)
    w = e.world
    for r in range(2, w.rows - 2):
        base = r * w.cols
        for c in range(w.cols):
            f = base + c
            nb0 = np.asarray(w.neighbors(f))
            assert nb0.shape == (8,), f"标准行 {f} 邻居数 != 8"
            assert np.array_equal(nb0, w._nb_table[f]), f"表与 neighbors 不一致：{f}"


# ---------------------------------------------------------------- C4 闸门

def test_c4_gate_closed_when_hm_on():
    """HM（或任一异质机制）开启 ⇒ T1b 整段不执行（回退参考循环）。"""
    e = _mk_engine(seed=42, hm=True)
    for _ in range(30):
        if e.extinct:
            break
        e.step()
    assert e._t1b_batched_n == 0, "HM 开启时 T1b 不应执行"
