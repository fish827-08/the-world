"""R240 T8 气味场 v1 单测（`[云端开发·云启]`，2026-09-27）—— 对设计稿 §8.3 预登记验收。

覆盖：
  A **默认关 ⇒ 零成本 + 不进新路径**：`channels=()` ⇒ `eng.smell is None`、`smell_probe() is None`
  B **开档零污染**：气味场**只读环境量、写自己的场**（不消费 RNG、不改引擎状态）
     ⇒ 开档（未消费）的种群轨迹与关档**逐位一致**（这是"可回滚 + 不扰动"的硬证据）
  C **场力学**：注入可达 / 衰减 = `decay**k` / 粗网格扩散把值铺到非源格 / 粗网格经度周期闭合
  D **节拍（红线②）**：开档 T tick ⇒ `updates == T // k`（**不是每 tick**）；两次更新之间读数恒定
  E **稀疏注入（红线③）**：`inj_cells_last == 源格数`（只碰有源格）
  F **快照往返 + 续跑逐位**：场随档进出；半程存档→恢复→续跑 ≡ 不中断连续跑（场 + 种群逐位）
  G **配置往返（R233 T-F 同族）**：`from_dict(to_dict(c)).smell == c.smell`（新组必须接线）
  H **读接口（①食物通道可用）**：个体读**自己格**；未启用通道读 ⇒ fail-loud；斑块世界里食物场聚集在斑块
  I **降采样回落**：请求因子不整除 `rows/cols` ⇒ 取最大公约数（`downsample_eff` 可读回）
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from simulation.config import SimConfig  # noqa: E402
from simulation.sphere_engine import SphereEngine  # noqa: E402


def _cfg(**smell_kw) -> SimConfig:
    """小世界 + 斑块（`bgzero`）—— 让"食物源"天然稀疏（对应设计稿 §8.2-③ 的 2.8%）。"""
    c = SimConfig(seed=7)
    c.world.rows, c.world.cols = 8, 12
    c.population.initial_count = 60
    c.resources.distribution = "patchy"
    c.resources.bg_production_zero = True
    c.resources.patch_regrowth_mult = 1.195
    for k, v in smell_kw.items():
        setattr(c.smell, k, v)
    return c


def _state(eng: SphereEngine) -> tuple:
    P = len(eng._id)
    return (tuple(int(x) for x in eng._flat[:P]),
            tuple(float(x) for x in eng._energy[:P]),
            tuple(float(x) for x in eng._stomach[:P]))


# ─────────────────────── A：默认关 ───────────────────────

def test_default_off_gate_and_probe_none():
    """默认 `channels=()` ⇒ 不构造（`smell is None`）+ 读数 `None`（B4：未适用 ≠ 0）。"""
    eng = SphereEngine(_cfg())
    assert eng.config.smell.channels == ()
    assert eng.smell is None
    assert eng.smell_probe() is None
    for _ in range(10):
        eng.step()                     # 关档 ⇒ 只多一次 None 判断
    assert eng.smell is None


# ─────────────────────── B：开档零污染 ───────────────────────

def test_smell_on_does_not_perturb_trajectory():
    """气味场只读环境量、只写自己的场 ⇒ 开档（未消费）轨迹与关档**逐位一致**。

    这条同时覆盖"不消费 RNG"（H2/H1 形状纪律）：气味场若偷用 `self.rng`，
    随机流会错位 ⇒ 本测试立刻红。
    """
    off, on = SphereEngine(_cfg()), SphereEngine(_cfg(channels=("food", "prey")))
    for _ in range(40):
        off.step()
        on.step()
    assert off.smell is None and on.smell is not None
    assert _state(off) == _state(on), "开档（不消费）的轨迹与关档不逐位 ⇒ 气味场污染了引擎状态/RNG"
    # 场确实在算（不是空转）
    assert on.smell_probe()["updates"] >= 40 // on.config.smell.update_every


# ─────────────────────── C：场力学 ───────────────────────

def test_mechanics_inject_decay_diffuse():
    """注入可达；衰减 = `decay**k`/次；粗网格扩散铺到非源格。"""
    c = _cfg(channels=("food",), update_every=4, decay=0.9, diffuse=0.2)
    eng = SphereEngine(c)
    sf = eng.smell
    src = np.array([5], dtype=np.int64)
    sf.update({"food": (src, np.array([1.0]))}, tick=4)          # smell:inj 路径
    v0 = float(sf.at(src, "food")[0])
    assert v0 > 0.5, f"注入后源格应有值，实得 {v0}"
    # 非源格被扩散铺到（粗网格拉普拉斯 → 双线性插值）
    others = np.array([6, 17, 4], dtype=np.int64)
    assert float(np.abs(sf.at(others, "food")).sum()) > 0.0, "扩散没铺到非源格"
    # 只衰减（无注入、diffuse=0）⇒ 一次更新乘 decay**k
    c2 = _cfg(channels=("food",), update_every=4, decay=0.9, diffuse=0.0)
    e2 = SphereEngine(c2)
    e2.smell.update({"food": (src, np.array([1.0]))}, tick=4)
    a = float(e2.smell.at(src, "food")[0])
    e2.smell.update({}, tick=8)
    b = float(e2.smell.at(src, "food")[0])
    assert b == pytest.approx(a * 0.9 ** 4, rel=1e-12), f"衰减应为 decay**k：{a} → {b}"


def test_coarse_grid_longitude_is_periodic():
    """粗网格经度**周期**：跨越 `col=cols-1 → 0` 的扩散必须发生（球面无边界）。"""
    c = _cfg(channels=("food",), update_every=1, decay=0.999, diffuse=0.25)
    eng = SphereEngine(c)
    cols = int(eng.world.cols)
    last_col = np.arange(cols - 1, 8 * cols, cols, dtype=np.int64)   # 最后一列
    eng.smell.update({"food": (last_col, np.ones(last_col.size))}, tick=1)
    first_col = np.arange(0, 8 * cols, cols, dtype=np.int64)
    v = float(np.abs(eng.smell.at(first_col, "food")).sum())
    assert v > 0.0, "经度不闭合（第 0 列没收到来自最后一列的扩散）"


# ─────────────────────── D/E：节拍与稀疏 ───────────────────────

def test_cadence_and_sparse_injection():
    """开档 T tick ⇒ `updates == T // k`（红线②）；`inj_cells_last == 源格数`（红线③）。"""
    k = 4
    engines = _cfg(channels=("food",), update_every=k)
    eng = SphereEngine(engines)
    n_food = int(np.count_nonzero(eng.resources._grid > 0.0))
    assert 0 < n_food < eng.world.n_cells, "前提：食物源应当**稀疏**（否则本测试退化）"
    for t in range(1, 21):
        eng.step()
        if t % k == 0:
            assert eng.smell_probe()["updates"] == t // k
    p = eng.smell_probe()
    assert p["updates"] == 20 // k, f"节拍不对：{p['updates']} != {20 // k}（红线②：每 tick 全场？）"
    assert p["inj_cells_last"] == n_food, (
        f"注入格数 {p['inj_cells_last']} != 食物源格数 {n_food}（红线③：注入未稀疏？）")


def test_read_is_stable_between_updates():
    """两次更新之间（k=4）读数恒定（线索延迟 k tick —— 设计稿 §8.2-② 的代价，明示不隐）。"""
    eng = SphereEngine(_cfg(channels=("food",), update_every=4))
    eng.step()
    eng.step()          # t=2
    v1 = float(eng.smell.at(np.array([0]), "food")[0])
    eng.step()          # t=3
    v2 = float(eng.smell.at(np.array([0]), "food")[0])
    assert v1 == v2, "t=2 与 t=3 之间场不应变化（更新节拍 k=4）"


# ─────────────────────── F：快照往返 + 续跑逐位 ───────────────────────

def test_snapshot_roundtrip_and_resume_bitwise(tmp_path: Path):
    """存档 → 恢复：场逐位；半程存档→续跑 ≡ 不中断连续跑（**场 + 种群**都逐位）。"""
    cfg = _cfg(channels=("food", "prey"), update_every=4)
    e_cont, e_snap = SphereEngine(cfg), SphereEngine(cfg)
    for _ in range(30):
        e_cont.step()
        e_snap.step()
    path = tmp_path / "smell.snapshot.npz"
    e_snap.save_snapshot(str(path))
    load = SphereEngine.load_snapshot(str(path))          # 配置随档（含 smell 组）
    assert load.config.smell.channels == ("food", "prey"), "快照没带回 smell 配置（from_dict 漏接线？）"
    np.testing.assert_array_equal(load.smell._S, e_snap.smell._S, err_msg="近场未逐位恢复")
    np.testing.assert_array_equal(load.smell._Sc, e_snap.smell._Sc, err_msg="远场未逐位恢复")
    for _ in range(30):
        e_cont.step()
        load.step()
    np.testing.assert_array_equal(load.smell._S, e_cont.smell._S, err_msg="续跑后近场不逐位")
    np.testing.assert_array_equal(load.smell._Sc, e_cont.smell._Sc, err_msg="续跑后远场不逐位")
    assert _state(load) == _state(e_cont), "续跑后种群不逐位"


def test_injection_accumulates_same_cell():
    """同格多个体必须**累加**（`np.add.at`；fancy-index `+=` 会丢重复 ⇒ 静默少算）。"""
    c = _cfg(channels=("prey",), decay=1.0, diffuse=0.0, update_every=1)
    eng = SphereEngine(c)
    eng.smell.update({"prey": (np.array([5, 5, 7]), np.ones(3))}, tick=1)
    assert float(eng.smell._S[0, 5]) == 2.0, f"同格两体应累加到 2.0，实得 {eng.smell._S[0, 5]}"
    assert float(eng.smell._S[0, 7]) == 1.0
    rr, cc = np.divmod(np.array([5]), eng.world.cols)
    # 5 与 7 落在**同一粗格**（s_eff=4 ⇒ 粗列 1）⇒ 远场**块均值** = (2+1)/s² = 3/16
    #   🔴 R244 自查修正：v1 首版此处漏 ÷s²（粗网格=分箱和）⇒ 远场被放大 s² 倍；已修。
    _s2 = float(eng.smell._s * eng.smell._s)
    assert float(eng.smell._Sc[0, int(rr[0]) // eng.smell._s, int(cc[0]) // eng.smell._s]) == 3.0 / _s2


def test_snapshot_has_smell_key_only_when_on(tmp_path: Path):
    """全关 ⇒ 快照**无** `smell_S`/`smell_Sc` 键（不虚增档体积）；开档 ⇒ 有且形状正确。"""
    p1, p2 = tmp_path / "off.npz", tmp_path / "on.npz"
    SphereEngine(_cfg()).save_snapshot(str(p1))
    d1 = np.load(p1, allow_pickle=True)
    assert "smell_S" not in d1 and "smell_Sc" not in d1, "全关档不应带 smell_* 键"
    eng = SphereEngine(_cfg(channels=("food",)))
    eng.step()
    eng.save_snapshot(str(p2))
    d = np.load(p2, allow_pickle=True)
    assert d["smell_S"].shape == (1, eng.world.n_cells)
    assert d["smell_Sc"].shape == (1, eng.smell._cr, eng.smell._cc)


# ─────────────────────── G：配置往返 ───────────────────────

def test_config_roundtrip_includes_smell():
    """R233 T-F 同族：新配置组必须**显式接线**到 `from_dict`，否则续跑静默退回默认。"""
    c = SimConfig(seed=3)
    c.smell.channels = ("food", "risk")
    c.smell.update_every = 7
    c.smell.downsample = 4
    back = SimConfig.from_dict(c.to_dict())
    assert back.smell == c.smell, f"往返丢字段：{back.smell} != {c.smell}"
    assert back.fingerprint() == c.fingerprint()
    # 旧存档（无 smell 键）⇒ 回退默认（全关）
    d = c.to_dict()
    d.pop("smell")
    assert SimConfig.from_dict(d).smell.channels == ()


# ─────────────────────── H：读接口 ───────────────────────

def test_read_own_cell_and_fail_loud_on_unknown_channel():
    """个体读**自己格**；未启用通道 ⇒ 断言（禁静默 no-op）。"""
    eng = SphereEngine(_cfg(channels=("food",)))
    for _ in range(8):
        eng.step()
    vals = eng.smell.at(eng._flat[:5], "food")
    assert vals.shape == (5,)
    assert np.all(vals >= 0.0)
    with pytest.raises(AssertionError):
        eng.smell.at(eng._flat[:1], "kin")      # 未启用通道
    with pytest.raises(AssertionError):
        SimConfig(seed=1).smell.__class__(channels=("foods",))   # 构造期校验
    with pytest.raises(AssertionError):
        bad = _cfg(channels=("foods",))         # 构造后改配置不走 __post_init__ ⇒ 模块侧自查
        SphereEngine(bad)


def test_food_channel_tracks_patches():
    """①食物通道可用：食物场浓度在**斑块格**上明显高于背景格（斑块 = 源）。"""
    eng = SphereEngine(_cfg(channels=("food",), update_every=1))
    for _ in range(4):
        eng.step()
    patch = np.flatnonzero(eng.resources._grid > 0.0)
    bg = np.flatnonzero((eng.resources._grid <= 0.0) & (eng.resources._capacity <= 0.0))
    assert patch.size and bg.size
    vp = float(np.mean(eng.smell.at(patch, "food")))
    vb = float(np.mean(eng.smell.at(bg, "food"))) if bg.size else 0.0
    assert vp > vb, f"斑块格浓度({vp:.4f}) 应高于纯背景格({vb:.4f})"


# ─────────────────────── I：降采样回落 ───────────────────────

def test_downsample_falls_back_to_divisor(capsys):
    """请求因子不整除 `rows/cols` ⇒ 落到 ≤ 请求值的最大公约数，并**告警**（不静默）。"""
    cfg = SimConfig(seed=1)
    cfg.world.rows, cfg.world.cols = 8, 12          # 12 % 8 != 0
    cfg.smell.channels = ("food",)
    cfg.smell.downsample = 8
    eng = SphereEngine(cfg)
    p = eng.smell_probe()
    assert p["downsample_req"] == 8
    assert p["downsample_eff"] == 4, f"8 与 12 的最大公约数应取 4，实得 {p['downsample_eff']}"
    assert (8 // p["downsample_eff"]) * p["downsample_eff"] == 8
    assert (12 // p["downsample_eff"]) * p["downsample_eff"] == 12
    assert "降采样因子回落" in capsys.readouterr().err, "回落必须**告警**（禁静默改口径）"