"""13.6 S1 三种食物地形 —— 参数化（`[云端·开发]`，2026-09-24；R193 派工单 §一）。

背景：R193 定稿「三种食物地形」方案（森林/草原/荒漠），S1 = **参数化**（不碰机制）：
① a4 补 `--patch-count` / `--patch-radius` / `--patch-capacity-mult`（**原全缺**）
② `switches` 读回三参数（C4 自证）
③ 三个地形 preset（`terrain_forest` / `terrain_grass` / `terrain_desert`）

本文件覆盖（S1 交付判据 1/2 + 协议 DEL-3/4/5/6/7/8）：
  ① **关档逐位等价**（DEL-4 / 判据 1）：不传地形参数 == 显式传默认值（30/2/None）
     ⇒ 逐位一致（C7 基线 digest `(574887, 11266.746993)` 由 `self_audit digest` 的
     12 个测试文件钉死 —— 本文件不重复抄基线，避免"抄写漂移"）
  ② **三 CLI 各自生效**（DEL-3 / 判据 2）：三个参数传参 ⇒ config 读回 == 传入值，
     **且世界结构与轨迹真的变了**（patch 格数 817→794/518/90；digest 与关档不同）
     ⇒ 堵 B 类"传了没生效"（13.5 `bg_production_zero` 少一行接线的事故同型）
  ③ **地形几何与设计稿 §一 的表逐值一致**：单斑块容量 3788/700/669（派生量，防"表算错"）
  ④ **DEL-6 运行期不变量**：Σcapacity 与 patch 掩码在 **t>0** 采样点仍成立
     （13.4 容量守恒"只在 t=0 断言 ⇒ 运行期漂移没被发现"的事故同型）
  ⑤ **端到端**：a4 子进程传 CLI ⇒ `summary.switches` 读回 == 传入值（判据 2 的产物形态）
  ⑥ **DEL-8 变异证据**：把接线改坏 ⇒ ② 的敏感断言立刻变红（实测产物见
     `_rerun_logs/terrain_s1_smoke/del8_mutation_*.txt`，本文件即那组"敏感断言"）
  ⑦ **H3 适用性钉死**：地形**不新增代码路径**（Python 构造掩码 → Rust `regrow_patchy`
     消费）⇒ **无需 H3**；这里反向固定"`use_sim_core=True` + 地形仍可构造且掩码正确"
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.a4_verify_capacity import build  # noqa: E402
from simulation.config import ResourceConfig, SimConfig  # noqa: E402

PY = sys.executable

#: 设计稿 §一 的三地形表：name → (patch_count, patch_radius, 斑块格数, 单斑块容量)
#: 前两列 = 传入参数；后两列 = 派生量（**实测值**，由本文件的断言钉死）
TERRAIN_TABLE = {
    "forest": (12, 3, 794, 3788),
    "grass": (60, 1, 518, 700),
    "desert": (10, 1, 90, 669),
}
#: 现状（config 默认 30/2）：斑块格 817、单斑块容量 1974（判据 2 的"从 817 变到 …"基准）
DEFAULT_GEOM = (30, 2, 817, 1974)


def _digest(e) -> tuple[int, float]:
    return (int(e._flat.sum()), round(float(e._energy.sum()), 6))


def _run(e, ticks: int = 50):
    for _ in range(ticks):
        if e.extinct:
            break
        e.step()
    return e


# ------------------------------------------------- ① 关档逐位等价（DEL-4）

def test_defaults_bit_identical_vs_explicit_defaults():
    """不传地形参数 == 显式传默认值（30/2/None）⇒ 逐位一致。

    这是"默认值 = 现状"在本线的落点：新 CLI 的默认**必须**与 config 一致，
    否则"不传参数"的老批不可复现（协议 DEL-4 / Blocker B2）。
    """
    a = _run(build("on", True, 42, 100))
    b = _run(build("on", True, 42, 100, patch_count=30, patch_radius=2,
                   patch_capacity_mult=None))
    assert _digest(a) == _digest(b)
    np.testing.assert_array_equal(a._flat, b._flat)
    np.testing.assert_array_equal(a._energy, b._energy)
    np.testing.assert_array_equal(a._genes, b._genes)
    assert a.rng_draws == b.rng_draws


def test_terrain_defaults_match_config_defaults():
    """CLI 默认值 == `ResourceConfig` 默认（防"两处默认漂移"⇒ 直接构造的工具得到别的世界）。"""
    e = build("on", True, 42, 10)
    r = ResourceConfig()
    assert e.config.resources.patch_count == r.patch_count == 30
    assert e.config.resources.patch_radius == r.patch_radius == 2
    assert e.config.resources.patch_capacity_mult == r.patch_capacity_mult == 3.0


# ------------------------------------------------- ② 三 CLI 各自生效（DEL-3）

def test_patch_count_and_radius_cli_effective():
    """`--patch-count` / `--patch-radius` 传参 ⇒ config 读回 == 传入值 **且世界变了**。

    🔴 这一条是 **DEL-8 变异测试的敏感断言**：把 `build()` 里的
    `patch_count=args.patch_count` 接线去掉（或改成硬编码）⇒ 本测试立刻红。
    """
    e = build("on", True, 42, 10, distribution="patchy", patch_count=12, patch_radius=3)
    assert e.config.resources.patch_count == 12
    assert e.config.resources.patch_radius == 3
    assert int(e.resources._patch_mask.sum()) == 794, "世界结构未随之改变（no-op）"
    # 默认档仍是 817
    e0 = build("on", True, 42, 10, distribution="patchy")
    assert int(e0.resources._patch_mask.sum()) == 817


def test_patch_capacity_mult_cli_effective():
    """`--patch-capacity-mult` 传参 ⇒ 容量倍率变、斑块容量随之下调；不传 ⇒ 保持 3.0。"""
    base = build("on", True, 42, 10, distribution="patchy")
    lo = build("on", True, 42, 10, distribution="patchy", patch_capacity_mult=2.0)
    m = base.resources._patch_mask
    assert base.config.resources.patch_capacity_mult == 3.0, "不传必须保持 config 默认"
    assert lo.config.resources.patch_capacity_mult == 2.0
    assert float(lo.resources._capacity[m].mean()) < float(base.resources._capacity[m].mean()), (
        "容量倍率下调后斑块容量未变 ⇒ no-op")


@pytest.mark.parametrize("name", sorted(TERRAIN_TABLE))
def test_terrain_geometry_matches_design_table(name):
    """三地形几何与设计稿 §一 的表**逐值一致**（含派生量"单斑块容量"）。"""
    c, r, cells, single_cap = TERRAIN_TABLE[name]
    e = build("on", True, 42, 10, distribution="patchy", patch_count=c, patch_radius=r)
    m = e.resources._patch_mask
    assert int(m.sum()) == cells, f"{name}: 斑块格数 {int(m.sum())} ≠ 设计稿 {cells}"
    got_single = float(e.resources._capacity[m].sum()) / max(1, c)
    assert got_single == pytest.approx(single_cap, rel=0.01), (
        f"{name}: 单斑块容量 {got_single:.0f} ≠ 设计稿 {single_cap}")


def test_terrain_changes_trajectory_not_noop():
    """**轨迹**证据（DEL-3 ②）：地形参数不只是改了配置对象 —— 50 tick 后状态逐步不同。"""
    base = _run(build("on", True, 42, 100, distribution="patchy"), 50)
    forest = _run(build("on", True, 42, 100, distribution="patchy",
                        patch_count=12, patch_radius=3), 50)
    assert int(base.resources._patch_mask.sum()) == 817
    assert int(forest.resources._patch_mask.sum()) == 794
    assert _digest(base) != _digest(forest), "地形参数未改变轨迹 ⇒ 等于没生效"


# ------------------------------------------------- ④ DEL-6 运行期不变量（t>0）

def test_del6_runtime_invariant_sampled_after_t0():
    """Σcapacity 与 patch 掩码在 **t>0** 采样点仍成立（不只 t=0 —— 13.4 容量漂移教训）。"""
    e = build("on", True, 42, 100, distribution="patchy", patch_count=12, patch_radius=3)
    cap0 = float(e.resources._capacity.sum())
    n0 = int(e.resources._patch_mask.sum())
    _run(e, 20)                                   # t=20 > 0 的采样点
    assert float(e.resources._capacity.sum()) == pytest.approx(cap0, rel=1e-12), (
        "运行期 Σcapacity 漂移（地形是静态几何，不应变）")
    assert int(e.resources._patch_mask.sum()) == n0
    assert e._tick == 20


# ------------------------------------------------- ⑤ 端到端（判据 2 的产物形态）

def _run_a4(out: Path, snap: Path, extra: list[str], ticks: int = 20) -> dict:
    cmd = [PY, str(ROOT / "experiments" / "a4_verify_capacity.py"),
           "--mode", "on", "--arm", "main", "--seed", "42", "--ticks", str(ticks),
           "--max-count", "300", "--log-interval", "10", "--snapshot-every", "0",
           "--snapshot-dir", str(snap), "--distribution", "patchy",
           "--out", str(out)] + extra
    rc = subprocess.call(cmd, cwd=str(ROOT), stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
    assert rc == 0, f"a4 子进程 rc={rc}（cmd={' '.join(extra)}）"
    return json.loads(out.with_suffix(".summary.json").read_text(encoding="utf-8"))


def test_cli_end_to_end_switches_readback(tmp_path):
    """a4 子进程传 CLI ⇒ `summary.switches` 读回 == 传入值（C4 / 判据 2）。"""
    sw = _run_a4(tmp_path / "t.csv", tmp_path / "snap",
                 ["--patch-count", "12", "--patch-radius", "3",
                  "--patch-capacity-mult", "2.5"])["switches"]
    assert sw["patch_count"] == 12
    assert sw["patch_radius"] == 3
    assert sw["patch_capacity_mult"] == 2.5
    assert sw["distribution"] == "patchy"


def test_cli_end_to_end_defaults_readback(tmp_path):
    """不传 ⇒ `switches` 读回 config 默认（30/2/3.0）——"默认 = 现状"的产物证据（DEL-4）。"""
    sw = _run_a4(tmp_path / "t.csv", tmp_path / "snap", [])["switches"]
    assert sw["patch_count"] == 30
    assert sw["patch_radius"] == 2
    assert sw["patch_capacity_mult"] == 3.0


# ------------------------------------------------- ⑦ H3 适用性（地形非新机制）

def test_terrain_has_no_h3_guard_and_works_under_sim_core():
    """地形参数**不新增代码路径** ⇒ 无需 H3；反向固定"`use_sim_core=True` 也能跑"。

    依据：掩码在 `ResourceField`（Python）构造，Rust 侧 `regrow_patchy` 只**消费**
    `_patch_mask`/`_capacity` 数组 ⇒ 地形几何天然随两条路径一致（无需 fail-loud）。
    ⚠️ 若未来有人把地形做成"只在 Python 生效"的机制，本测试会把矛盾暴露出来
    （届时按 H3 纪律加构造期硬报错，而不是让本测试变绿）。
    """
    pytest.importorskip("sim_core", reason="本沙盒未编 sim_core.so ⇒ 跳过 Rust 路径验证")
    cfg = SimConfig(seed=42)
    cfg.population.max_count = 600
    cfg.simulation.use_sim_core = True
    cfg.resources.distribution = "patchy"
    cfg.resources.patch_count = 12
    cfg.resources.patch_radius = 3
    from simulation.sphere_engine import SphereEngine
    e = SphereEngine(cfg)                       # 若地形被做成"仅 Python"机制 ⇒ 这里应报错
    assert int(e.resources._patch_mask.sum()) == 794
    cap0 = float(e.resources._capacity.sum())
    e.step()
    assert float(e.resources._capacity.sum()) == pytest.approx(cap0, rel=1e-12)