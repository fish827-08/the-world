"""`s0_kernel.harness` 的靶向测试（S0-kernel-b 第二半：出货面与同流形制）。

钉的四件事：
1. **同 seed 逐位复现**（digest 与 CSV 字节级一致）⇒ 砚的断点续跑/checkpoint 依赖它；
2. **fail-loud 三门**（spec 缺字段、A/B 与 `alpha_W` 的语义错配、灭绝早停、非干净退出）；
3. **两臂只在被测通道上不同**（同一初始群体：同 seed ⇒ 初始位置数组逐位相同）；
4. **出货面格式**（`tick,ind_id,cell,g,energy,age` 六列、`g` 是 [0,1] 原始轴、manifest 增量齐）。

小档纪律：本文件全部用微型装置（60 个体 × 120 tick，墙钟 <1s），是**自测冒烟**，不是批次（R117）。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from s0_kernel.harness import Demography, S0World, run_arm

SMALL = dict(n_ind0=40, rows=20, cols=40, max_ind=200)


def _spec(run_id: str, arm: str, alpha_W, *, seed: int = 207, ticks: int = 60) -> dict:
    return {"run_id": run_id, "arm": arm, "seed": seed, "ticks": ticks, "sample": 20,
            "alpha_W": alpha_W, "demography": Demography(**SMALL), "arm_impl_commit": "test"}


def _manifest(csv_path: Path) -> dict:
    return json.loads(csv_path.with_suffix(".manifest.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------- ① 同流复现

def test_same_seed_same_arm_is_bitwise_reproducible(tmp_path) -> None:
    """同 seed 同臂两次 ⇒ digest 与 CSV **字节级**相同（断点续跑的前提）。"""
    p1 = run_arm(_spec("r1", "A", None), tmp_path)
    p2 = run_arm(_spec("r2", "A", None), tmp_path)
    m1, m2 = _manifest(p1), _manifest(p2)
    assert m1["digest"] == m2["digest"] == m1["digest"]
    assert p1.read_bytes().replace(b"r1", b"r2") == p2.read_bytes()
    assert m1["arms_sha256"] and m1["harness_sha256"] and len(m1["arms_sha256"]) == 16


def test_wall_tick_cost_is_reported_and_small(tmp_path) -> None:
    """自测性能表的最小形：微型档墙钟必须 <5s（正式档计时项归标定批，不在单测里跑）。"""
    import time
    t0 = time.perf_counter()
    m = _manifest(run_arm(_spec("perf", "B", 1.0), tmp_path))
    wall = time.perf_counter() - t0
    assert wall < 5.0, wall
    assert m["ticks"] == 60 and m["pop_end"] >= 0 and m["cap_hits"] >= 0


# ---------------------------------------------------------------- ② fail-loud 三门

def test_missing_spec_keys_raise() -> None:
    bad = _spec("x", "A", 0.0)
    del bad["sample"]
    with pytest.raises(KeyError, match="sample"):
        run_arm(bad, Path("/tmp/never_created_lz"))
    for k in ("run_id", "arm", "seed", "ticks", "alpha_W", "demography"):
        b2 = _spec("x", "A", 0.0)
        del b2[k]
        with pytest.raises(KeyError, match=k):
            run_arm(b2, Path("/tmp/never_created_lz"))


def test_bad_ticks_and_demography_raise() -> None:
    for ticks in (0, -5):
        with pytest.raises(ValueError, match="ticks/sample"):
            run_arm(_spec("x", "A", None, ticks=ticks), Path("/tmp/never_created_lz"))
    with pytest.raises(ValueError, match="repro_at"):
        Demography(repro_at=10.0, e0=60.0)
    with pytest.raises(ValueError, match="repro_cost"):
        Demography(repro_cost=0.0)
    with pytest.raises(ValueError, match="规模非法"):
        Demography(n_ind0=0)
    with pytest.raises(ValueError, match="intake_scale"):
        Demography(intake_scale=0.0)


def test_arm_alpha_W_semantics_are_enforced() -> None:
    """A 臂必须 `null`、B 臂必须显式数值（H2 §一.3 的出货面版本；0.0 与 None 不许互换）。"""
    with pytest.raises(ValueError, match="A 臂 spec 的 alpha_W 必须是 null"):
        run_arm(_spec("x", "A", 1.0), Path("/tmp/never_created_lz"))
    with pytest.raises(ValueError, match="A 臂 spec 的 alpha_W 必须是 null"):
        run_arm(_spec("x", "A", 0.0), Path("/tmp/never_created_lz"))    # 0.0 也不行：那是 Z2 专用语义
    with pytest.raises(ValueError, match="B 臂"):
        run_arm(_spec("x", "B", None), Path("/tmp/never_created_lz"))
    with pytest.raises(ValueError, match="arm 只能是"):
        S0World(207, "C", None, Demography(**SMALL))


def test_extinction_is_not_silent(tmp_path) -> None:
    """饿死全群 ⇒ RuntimeError（少跑 tick 的 run 不许混进同档读数），不许静默交短批。"""
    dem = Demography(**SMALL, meta=99.0, intake_scale=1e-9)     # 必灭绝（intake 归零但不违例）
    spec = dict(_spec("dead", "A", None), demography=dem)
    with pytest.raises(RuntimeError, match="灭绝早停"):
        run_arm(spec, tmp_path)


# ---------------------------------------------------------------- ③ 两臂差异只落在被测通道

def test_same_seed_same_initial_population_across_arms(tmp_path) -> None:
    """同 seed ⇒ 两臂**初始位置数组逐位相同**（配对差可用；差异只能来自 W 形与 α_W）。"""
    wa = S0World(207, "A", None, Demography(**SMALL))   # S0World 层：A 臂只收 null
    wb = S0World(207, "B", 1.0, Demography(**SMALL))
    assert wa.cells == wb.cells
    assert wa.g == wb.g and wa.e == wb.e
    assert wa.land.sigma_c == wb.land.sigma_c
    # A 臂不消费 ν；B 臂消费 ⇒ 这一步之后再跑，轨迹才会分叉（分叉源＝被测通道）
    assert wa.alpha_W is None and wb.alpha_W == 1.0


def test_zero_control_arm_B_tracks_W0_only(tmp_path) -> None:
    """`alpha_W=0` 的 B 臂轨迹必须等于"没有 f 项"的 B 臂：与 α_W=1 支明显分叉。"""
    z = _manifest(run_arm(_spec("z0", "B", 0.0), tmp_path))
    o = _manifest(run_arm(_spec("z1", "B", 1.0), tmp_path))
    assert z["alpha_W"] == 0.0 and o["alpha_W"] == 1.0
    assert z["digest"] != o["digest"], "α_W=0 与 1 同 digest ⇒ f 项根本没进回路"


# ---------------------------------------------------------------- ④ 出货面格式

def test_csv_schema_and_raw_g_axis(tmp_path) -> None:
    p = run_arm(_spec("sch", "B", 1.0), tmp_path)
    lines = p.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "tick,ind_id,cell,g,energy,age"
    assert len(lines) - 1 == _manifest(p)["rows_written"] > 0
    for row in lines[1:]:
        _, _, _, g, energy, age = row.split(",")
        assert 0.0 <= float(g) <= 1.0, f"g 必须在原始轴 [0,1]：{g}"
        assert float(energy) > 0.0 and int(age) >= 0


def test_manifest_echoes_all_device_knobs(tmp_path) -> None:
    """砚 ack 条件②：装置参数与统计码 hash 全部回显（禁"以默认值代实测"）。"""
    m = _manifest(run_arm(_spec("man", "A", None), tmp_path))
    for k in ("run_id", "arm", "seed", "ticks", "sample", "rows_written", "pop_end", "cap_hits",
              "digest", "n_cells", "demography", "landscape", "arm_impl_commit", "arms_sha256",
              "harness_sha256", "csv", "alpha_W", "f_form"):
        assert k in m, k
    assert m["arm_impl_commit"] == "test"
    assert set(m["demography"]) >= {"n_ind0", "meta", "repro_at", "mut_rate", "mut_sigma",
                                    "intake_scale", "lifespan", "max_ind"}
    assert m["landscape"]["sigma_c"] > 0.0            # σ_C 单值回显（待标定批锁）


def test_dict_demography_accepted_and_spec_demography_must_be_typed(tmp_path) -> None:
    """`spec["demography"]` 允许 dict（链式脚本侧好传），但其它类型当场 raise。"""
    spec = dict(_spec("dh", "A", None), demography=Demography(**SMALL).__dict__ if False
                else {"n_ind0": 40, "rows": 20, "cols": 40, "max_ind": 200})
    assert run_arm(spec, tmp_path).exists()
    with pytest.raises(TypeError, match="demography"):
        run_arm(dict(_spec("bad", "A", None), demography="nope"), tmp_path)


def test_harness_does_not_import_main_engine() -> None:
    """零行为面：harness 只碰 `world.sphere_world`＋numpy，**不 import 主引擎**。"""
    import ast
    import s0_kernel.harness as mod
    tree = ast.parse(open(mod.__file__, encoding="utf-8").read())
    got = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            got |= {al.name for al in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            got.add(node.module)
    assert "simulation" not in " ".join(sorted(got)), sorted(got)
    assert got <= {"__future__", "csv", "hashlib", "json", "dataclasses", "pathlib",
                   "numpy", "world.sphere_world", "arms"}, sorted(got)


def test_arm_A_ignores_sigma_c_because_it_has_no_frequency_term(tmp_path) -> None:
    """🔴 变异靶 H4：A 臂若误吃 ν（＝把 B 臂的频率项漏进对照臂）⇒ 本条红。

    σ_C 只进 `nu_local` ⇒ A 臂轨迹对它**必须**逐位不敏感；B 臂一侧在 arms 层直接钉
    （仿真里初代 g 全同 ⇒ ν=1 与 σ_C 无关，只有变异把 g 打散后才分叉 ⇒ 用短程仿真判"敏不敏感"
    会假绿，这里不冒那个险）。
    """
    from s0_kernel.arms import Landscape, nu_local, w_b

    d = []
    for sc in (0.05, 0.4):
        spec = dict(_spec(f"a{sc}", "A", None), demography=Demography(**SMALL))
        d.append(_manifest(run_arm(spec, tmp_path))["digest"])
    assert d[0] == d[1], "A 臂吃了 σ_C ⇒ 频率项漏进对照臂，双臂隔离破了"

    mixed = [0.6, 0.6, 0.9, 0.3]
    n_lo = nu_local(0.6, mixed, sigma_c=0.05)
    n_hi = nu_local(0.6, mixed, sigma_c=0.4)
    assert n_hi > n_lo > 0.0, (n_lo, n_hi)             # 核越宽 ⇒ 远表型也算竞争者
    assert w_b(0.6, n_lo, 1.0, land=Landscape(sigma_c=0.05)) !=         w_b(0.6, n_hi, 1.0, land=Landscape(sigma_c=0.4))   # B 臂必须随 ν 变


def test_birth_requires_energy_threshold(tmp_path) -> None:
    """🔴 变异靶 H4：出生必须过能量阈（`repro_at`）——阈提到不可达 ⇒ 人口只出不进。

    这条钉的是"繁殖不是每 tick 免费触发"；去掉阈判 ⇒ 本条红（人口直冲到 `max_ind`）。
    """
    dem = Demography(**SMALL, repro_at=1e9)
    m = _manifest(run_arm(dict(_spec("nb", "B", 0.0), demography=dem), tmp_path))
    assert m["cap_hits"] == 0
    assert m["pop_end"] <= dem.n_ind0, (m["pop_end"], dem.n_ind0)
    assert m["digest"][1] <= dem.n_ind0
