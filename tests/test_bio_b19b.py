"""B19b：观测 K 收敛判定 + 轻量长跑路径 + 2,000 步等价性门。

预注册：``docs/预注册/预注册-B19b-K收敛-20261021.md``（含《修订 1》）。

覆盖 §7.1 要求的 oracle：

1. 收敛纯函数：平台 / 晚升 / 振荡 / 末段漂移（中途通过但后缀不足）/ 灭绝 /
   窗口边界 / 边缘确认（``T_K=5000``）/ 参数守卫；
2. 整批三档结论（5/5、4/5、≤3/5）+ 灭绝阻断"全局收敛"；
3. ``_check_equivalence`` 逐项正反例（探针/终局/生卒/同步/气候）；
4. light 路径与完整路径在小世界逐项一致（探针/生卒/资源哈希/气候/大气 digest）；
5. ``SimLog`` 采集开关：light 只留生卒与能量账本，默认路径行为不变；
6. light 接线守卫（baseline 只在 light 下、series_every=1、ticks ≥ 2000）与
   ``run_batch`` 的 schema/判读/成本中止线；
7. 2,000 tick 端到端等价性门（真跑，``slow``）：完整路径当基线 ⇒ light 过门；
   扰动基线 ⇒ :class:`EquivalenceError`。

TIER = "mainline"；除标 ``slow`` 的 2,000 tick 端到端门之外全部进档②。
"""
from __future__ import annotations

import copy
from pathlib import Path

import pytest

from biosphere import world_coupled as wc
from biosphere.convergence import (
    CONCLUSION_GLOBAL, CONCLUSION_MAJORITY, CONCLUSION_NOT_GLOBAL,
    K_EPS, K_N, K_T0, K_TMAX, K_W,
    STATUS_CONVERGED, STATUS_EXTINCT, STATUS_NOT_CONVERGED,
    batch_k_convergence_verdict, k_convergence_verdict,
)
from biosphere.metrics import SimLog
from biosphere.resolve import ResolveEvent, ResolveReport

TIER = "mainline"

#: 小世界档（沿用 tests/test_bio_b18.py 的 24×48；天气 8×16）
SMALL = dict(init_pop=8, rows=24, cols=48, nlat=8, nlon=16, max_pop=100000)

#: B19 基线读数在 gitignored 的 ``results/``（独立数据仓），干净克隆/主树可能没有；
#: 依赖它的校验只在基线在场时跑，长跑前须先放样（预注册 §5.2）。
_B19_BASELINE_PRESENT = Path(wc.B19_BASELINE_PATH).is_file()
_B19_BASELINE_SKIP = pytest.mark.skipif(
    not _B19_BASELINE_PRESENT,
    reason=("B19 基线读数 results/biosphere/b19_uncapped.json 只在跑批工作树"
            "（数据仓）存在，本机无该文件；长跑前按预注册 §5.2 放样后此例自恢复"))


# ── 构造序列的助手 ────────────────────────────────────────────────────────


def _series_from(fn, t_max: int = K_TMAX) -> list:
    """``P[0..t_max]``：``fn(s)`` 给 s≥1 的值，``P[0]`` = ``fn(0)``。"""
    return [int(fn(s)) for s in range(t_max + 1)]


def _plateau(value: int = 1000, t_max: int = K_TMAX) -> list:
    return _series_from(lambda s: value, t_max)


def _rising(start: int = 200, slope: float = 1.0, t_max: int = K_TMAX) -> list:
    return _series_from(lambda s: start + int(slope * s), t_max)


def _step_at(step: int, low: int = 1000, high: int = 1500,
             t_max: int = K_TMAX) -> list:
    """``s ≤ step`` 为 low，之后跳到 high（阶跃）。"""
    return _series_from(lambda s: low if s <= step else high, t_max)


def _fake_baseline(series, *, born: int, died: int,
                   at_step: int = wc.B19B_EQUIV_AT) -> dict:
    """按 B19 pass0 的形状造一个假基线（只填等价性门读的字段）。"""
    probes = {str(t): int(series[t + 1]) for t in (0, 500, 1000, 1500, 1999)}
    climate = [{"tick": 25 * i, "t2m_c_mean": 10.0 + 0.01 * i,
                "precip_mm_yr_mean": 500.0 + i}
               for i in range(at_step // 25 + 1)]
    return {
        "population": {
            "alive_probes": probes,
            "final_alive": int(series[at_step]),
            "total_born": int(born),
            "total_died": int(died),
        },
        "weather": {"samples": climate},
    }


def _equiv_args(series, *, born=100, died=80, at_step=wc.B19B_EQUIV_AT):
    baseline = _fake_baseline(series, born=born, died=died, at_step=at_step)
    return dict(
        total_born=born, total_died=died,
        sync_count=at_step // 25, sync_max_abs_diff=0.0,
        sync_hold_equal=True,
        climate=baseline["weather"]["samples"],
        baseline=baseline, at_step=at_step,
    )


# ── 一、收敛纯函数：状态与 T_K ────────────────────────────────────────────


def test_plateau_converges_from_first_eval_point():
    v = k_convergence_verdict(_plateau(1000))
    assert v["status"] == STATUS_CONVERGED
    assert v["T_K"] == 1000                 # 首个评估点即合格 ⇒ 后缀最长
    assert v["suffix_len"] == 11
    assert v["K"] == 1000.0
    assert v["std"] == 0.0 and v["rel_std"] == 0.0
    assert v["edge_confirmation"] is False  # 后缀 ≫ N，不是边缘确认
    assert v["eval_points"] == list(range(1000, 6001, 500))
    assert all(r == 0.0 for r in v["r_values"])
    assert v["params"] == {"W": 500, "eps": 0.10, "N": 3, "t0": 500,
                           "t_max": 6000}


def test_monotone_rise_is_not_converged_but_reports_final_window():
    """持续指数上升：每个评估点都超阈 ⇒ 后缀 0，不报 K（§4.1 未收敛行）。"""
    P = _series_from(lambda s: int(200 * 1.001 ** s))
    v = k_convergence_verdict(P)
    assert v["status"] == STATUS_NOT_CONVERGED
    assert v["T_K"] is None and v["K"] is None
    assert v["suffix_len"] == 0
    # 预注册 §4.1：未收敛仍报末窗均值，但必须标注"非 K"
    assert v["final_window_mean"] == pytest.approx(
        sum(int(200 * 1.001 ** s) for s in range(5501, 6001)) / 500)
    assert all(r >= K_EPS for r in v["r_values"])


def test_slow_linear_drift_can_pass_eps_late():
    """§9.4 已披露的风险：线性上升的相对变化按 1/s 衰减 ⇒ 末窗可能 <ε。

    本例末评估点 r≈0.092 < 0.10（后缀 1 < N=3 ⇒ 仍判未收敛），
    但若漂移更慢就会误判为平台 ⇒ 判读时必须同时看末窗相对标准差。
    """
    v = k_convergence_verdict(_rising(200, 1.0))
    assert v["status"] == STATUS_NOT_CONVERGED
    assert v["suffix_len"] == 1 < K_N
    assert v["r_values"][-1] < K_EPS < v["r_values"][-2]
    assert v["final_window_rel_std"] < 0.05


def test_mid_run_plateau_then_late_drift_is_not_converged():
    """预注册 §3.2 第 5 条：历史上通过 3 窗不算数，必须稳到 T_max。"""
    P = _series_from(lambda s: 1000 if s <= 4000 else 1000 + (s - 4000))
    v = k_convergence_verdict(P)
    # 1000..4000 的评估点 r=0（中途确实有 ≥3 窗通过）
    assert v["r_values"][:7] == [0.0] * 7
    assert sum(1 for r in v["r_values"] if r < K_EPS) == 7
    # 但末段漂移 ⇒ 从 T_max 向后的合格后缀 < N
    assert v["status"] == STATUS_NOT_CONVERGED
    assert v["suffix_len"] < K_N
    assert v["T_K"] is None


def test_short_period_oscillation_is_smoothed_by_window_mean():
    """周期 20 的 ±15% 振荡：窗口均值判据会平滑掉（预注册 §9.4 已披露）。"""
    P = _series_from(lambda s: 1000 + (150 if s % 2 == 0 else -150))
    v = k_convergence_verdict(P)
    assert v["status"] == STATUS_CONVERGED
    assert v["K"] == pytest.approx(1000.0, abs=1.0)
    assert all(r < K_EPS for r in v["r_values"])


def test_extinction_short_circuits_and_reports_last_window():
    P = _plateau(1000)
    for s in range(3000, K_TMAX + 1):
        P[s] = 0
    v = k_convergence_verdict(P)
    assert v["status"] == STATUS_EXTINCT
    assert v["extinct_at"] == 3000
    assert v["T_K"] is None and v["K"] is None
    assert v["r_values"] is None and v["suffix_len"] == 0
    assert v["pre_extinction_window_mean"] == pytest.approx(
        sum(P[2500:3000]) / 500)


def test_extinction_at_first_step_has_no_pre_window():
    P = _plateau(100)
    P[1] = 0
    v = k_convergence_verdict(P)
    assert v["status"] == STATUS_EXTINCT and v["extinct_at"] == 1
    assert v["pre_extinction_window_mean"] is None


def test_edge_confirmation_when_T_K_is_5000():
    """后缀恰好 = N ⇒ ``T_K=5000``，预注册 §3.2 第 6 条要求标"边缘确认"。"""
    P = _step_at(4000, low=1000, high=1500)
    v = k_convergence_verdict(P)
    assert v["status"] == STATUS_CONVERGED
    assert v["T_K"] == 5000 and v["suffix_len"] == K_N
    assert v["edge_confirmation"] is True
    assert v["K"] == 1500.0
    # 4500 那一窗跨了阶跃 ⇒ 不合格
    assert v["r_values"][v["eval_points"].index(4500)] >= K_EPS


def test_longer_suffix_is_not_edge_confirmation():
    P = _step_at(2000, low=1000, high=1500)
    v = k_convergence_verdict(P)
    assert v["status"] == STATUS_CONVERGED
    assert v["T_K"] == 3000 and v["suffix_len"] == 7
    assert v["edge_confirmation"] is False


def test_final_window_excludes_p0():
    """``P[0]`` 是初始值，不进任何窗口 ⇒ 不许污染 K。"""
    P = _plateau(500)
    P[0] = 10 ** 6
    v = k_convergence_verdict(P)
    assert v["status"] == STATUS_CONVERGED
    assert v["K"] == 500.0 and v["final_window_mean"] == 500.0


def test_spike_at_5500_lands_in_previous_window_only():
    """末窗 = ``P[5501..6000]``：``P[5500]`` 只进上一窗 ⇒ 末窗均值不受影响。"""
    P = _plateau(500)
    P[5500] = 10 ** 6
    v = k_convergence_verdict(P)
    assert v["final_window_mean"] == 500.0
    assert v["status"] == STATUS_NOT_CONVERGED   # r(5500)/r(6000) 都被尖峰抬起
    assert v["K"] is None
    assert v["r_values"][-1] >= K_EPS


def test_series_longer_than_tmax_is_accepted():
    P = _plateau(700, t_max=K_TMAX + 50)
    v = k_convergence_verdict(P)
    assert v["series_len"] == K_TMAX + 51
    assert v["K"] == 700.0


def test_reduced_horizon_still_aligns():
    """自检用短时限（t_max=1000）：只要 ``(t_max-t0) % W == 0`` 即可判。"""
    v = k_convergence_verdict([1000] * 1001, t_max=1000)
    assert v["eval_points"] == [1000]
    assert v["status"] == STATUS_NOT_CONVERGED   # 后缀 1 < N=3
    assert v["suffix_len"] == 1


@pytest.mark.parametrize("kwargs", [
    {"W": 0}, {"eps": 0.0}, {"eps": 1.0}, {"N": 0},
    {"t0": 400},                     # t0 < W
    {"t_max": 900},                  # t_max < t0+W
    {"t_max": 5750},                 # (t_max-t0) % W != 0
])
def test_param_guards(kwargs):
    with pytest.raises(ValueError):
        k_convergence_verdict(_plateau(1000), **kwargs)


def test_short_series_rejected():
    with pytest.raises(ValueError, match="长度"):
        k_convergence_verdict([1000] * 100, t_max=K_TMAX)


# ── 二、整批三档结论 ──────────────────────────────────────────────────────


def test_batch_global_convergence_5_of_5():
    series = {s: _plateau(1000 + 100 * i)
              for i, s in enumerate((42, 43, 44, 45, 46))}
    out = batch_k_convergence_verdict(series)
    assert out["conclusion"] == CONCLUSION_GLOBAL
    assert (out["n_total"], out["n_converged"], out["n_extinct"]) == (5, 5, 0)
    assert out["converged_seeds"] == [42, 43, 44, 45, 46]
    cross = out["K_cross_seed"]
    assert cross["n"] == 5
    assert cross["min"] == 1000.0 and cross["max"] == 1400.0
    assert cross["median"] == 1200.0


def test_batch_majority_4_of_5_is_not_global():
    series = {42: _plateau(1000), 43: _plateau(1100),
              44: _rising(200, 2.0), 45: _plateau(900), 46: _plateau(1200)}
    out = batch_k_convergence_verdict(series)
    assert out["conclusion"] == CONCLUSION_MAJORITY
    assert out["n_converged"] == 4 and out["not_converged_seeds"] == [44]
    assert out["K_cross_seed"]["n"] == 4        # 未收敛 seed 不并入 K 汇总
    assert 44 not in out["K_cross_seed"]["seeds"]


def test_batch_three_converged_is_not_global():
    series = {42: _plateau(1000), 43: _rising(200, 2.0),
              44: _rising(300, 3.0), 45: _plateau(900), 46: _plateau(1200)}
    out = batch_k_convergence_verdict(series)
    assert out["conclusion"] == CONCLUSION_NOT_GLOBAL
    assert out["n_converged"] == 3


def test_batch_extinction_blocks_global_conclusion():
    P = _plateau(1000)
    for s in range(2000, K_TMAX + 1):
        P[s] = 0
    series = {42: _plateau(1000), 43: _plateau(1100), 44: P,
              45: _plateau(900), 46: _plateau(1200)}
    out = batch_k_convergence_verdict(series)
    assert out["conclusion"] != CONCLUSION_GLOBAL
    assert out["conclusion"] == CONCLUSION_MAJORITY
    assert out["extinct_seeds"] == [44] and out["n_extinct"] == 1
    assert out["per_seed"][44]["status"] == STATUS_EXTINCT


def test_batch_no_convergence_gives_no_cross_seed_K():
    series = {42: _rising(200, 1.0), 43: _rising(200, 2.0)}
    out = batch_k_convergence_verdict(series)
    assert out["conclusion"] == CONCLUSION_NOT_GLOBAL
    assert out["K_cross_seed"] is None


# ── 三、2,000 步等价性检查器（逐项正反例） ────────────────────────────────


def test_equivalence_passes_on_exact_baseline():
    series = _plateau(3227, t_max=wc.B19B_EQUIV_AT)
    res = wc._check_equivalence(series, **_equiv_args(series))
    assert res["passed"] is True and res["mismatches"] == []
    assert res["at_step"] == 2000
    assert set(res["probes"]) == {"P[1]", "P[501]", "P[1001]", "P[1501]",
                                 "P[2000]"}
    assert all(p["ok"] for p in res["probes"].values())
    assert all(c["ok"] for c in res["checks"].values())
    assert res["climate"] == {"n_expected": 81, "n_actual": 81, "ok": True}


@pytest.mark.parametrize("patch,frag", [
    ({"series_at": (1, +1)}, "probe P[1]"),
    ({"series_at": (2000, +1)}, "final_alive"),
    ({"born": +1}, "total_born"),
    ({"died": +1}, "total_died"),
    ({"sync_count": 79}, "sync_count"),
    ({"sync_max_abs_diff": 1e-12}, "sync_max_abs_diff"),
    ({"sync_hold_equal": False}, "sync_hold_equal"),
    ({"climate_at": (3, 1.0)}, "climate"),
    ({"climate_short": True}, "climate"),
])
def test_equivalence_flags_each_mismatch(patch, frag):
    series = _plateau(3227, t_max=wc.B19B_EQUIV_AT)
    kwargs = _equiv_args(series)
    if "series_at" in patch:
        idx, delta = patch["series_at"]
        series = list(series)
        series[idx] += delta
        # 基线仍按未扰动序列造 ⇒ 只有 actual 侧变
    if "born" in patch:
        kwargs["total_born"] += patch["born"]
    if "died" in patch:
        kwargs["total_died"] += patch["died"]
    if "sync_count" in patch:
        kwargs["sync_count"] = patch["sync_count"]
    if "sync_max_abs_diff" in patch:
        kwargs["sync_max_abs_diff"] = patch["sync_max_abs_diff"]
    if "sync_hold_equal" in patch:
        kwargs["sync_hold_equal"] = patch["sync_hold_equal"]
    climate = [dict(c) for c in kwargs["climate"]]
    if "climate_at" in patch:
        i, d = patch["climate_at"]
        climate[i]["t2m_c_mean"] += d
    if patch.get("climate_short"):
        climate = climate[:80]
    kwargs["climate"] = climate
    res = wc._check_equivalence(series, **kwargs)
    assert res["passed"] is False
    assert any(frag in m for m in res["mismatches"]), res["mismatches"]


def test_equivalence_rejects_moving_checkpoint():
    series = _plateau(1000, t_max=wc.B19B_EQUIV_AT)
    kwargs = _equiv_args(series, at_step=wc.B19B_EQUIV_AT)
    kwargs["at_step"] = 1000
    with pytest.raises(ValueError, match="固定为 2000"):
        wc._check_equivalence(series, **kwargs)


def test_load_baseline_rejects_wrong_sha256(tmp_path):
    p = tmp_path / "b19.json"
    p.write_text('{"seeds": []}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="sha256 不符"):
        wc.load_b19_baseline(str(p), "0" * 64)
    by_seed, meta = wc.load_b19_baseline(str(p), "")
    assert by_seed == {} and meta["expected_sha256"] == ""


@_B19_BASELINE_SKIP
def test_b19_baseline_file_is_intact():
    """基线读数必须可读、指纹匹配、5 seed 齐（长跑前置门）。"""
    by_seed, meta = wc.load_b19_baseline()
    assert meta["sha256"] == wc.B19_BASELINE_SHA256
    assert sorted(by_seed) == [42, 43, 44, 45, 46]
    for row in by_seed.values():
        assert set(row["population"]["alive_probes"]) == {
            "0", "500", "1000", "1500", "1999"}
        assert len(row["weather"]["samples"]) == 81
        assert row["resource"]["sync_count"] == 80


# ── 四、SimLog 采集开关（修订 1） ─────────────────────────────────────────


def _report(tick: int = 0) -> ResolveReport:
    ev = ResolveEvent(tick=tick, actor_id=1, kind="eat", status="ok",
                      target_cell=0, target_id=-1, actor_energy_delta=0.4)
    return ResolveReport(tick=tick, events=[ev], births=[(1, 2, 0)],
                         deaths=[(3, "starved")], energy_paid=0.05,
                         energy_in=0.4, endowment_paid=0.0)


def test_simlog_default_keeps_frames_and_events():
    log = SimLog()
    assert log.capture_frames is True and log.capture_events is True
    log.record_frame(0, [], None)               # 默认路径仍要真帧数据
    assert len(log.frames) == 1                 # 空种群 ⇒ 记一帧空数组
    log.record_report(_report())
    assert len(log.events) == 1


def test_simlog_light_drops_frames_and_events_keeps_ledger():
    log = SimLog(capture_frames=False, capture_events=False)
    # world=None 也能过 ⇒ 证明 record_frame 在任何取值前就返回（不碰世界）
    log.record_frame(0, [], None)
    log.record_chain_frame(0, None, None, None, None)
    log.record_group_frame(0, None, None, None, None)
    assert log.frames == [] and log.chain_frames == [] and log.group_frames == []
    log.record_report(_report())
    assert log.events == []
    assert log.births == [(1, 2, 0)]
    assert log.deaths == [(3, "starved", 0)]
    assert log.energy_paid == pytest.approx(0.05)
    assert log.energy_in == pytest.approx(0.4)


def test_light_drops_graveyard_keeps_death_ledger():
    """修订 1·R1.6：墓地只供 diagnose 谱系用 ⇒ light 不留，死账仍逐笔记。"""
    full = wc.run_pass(42, ticks=300, **SMALL)
    light = wc.run_pass(42, ticks=300, light=True, **SMALL)
    assert full["population"]["total_died"] > 0
    assert light["population"]["total_died"] == full["population"]["total_died"]
    assert light["population"]["death_causes"] == full["population"]["death_causes"]
    # 轨迹不受影响（哈希与探针逐项相同）
    assert light["resource"]["amount_sha256"] == full["resource"]["amount_sha256"]
    assert light["population"]["alive_probes"] == full["population"]["alive_probes"]


def test_population_retain_dead_default_true():
    """默认（完整路径）必须照旧保留墓地 ⇒ diagnose 谱系可用。"""
    from biosphere.resolve import Population
    assert Population().retain_dead is True


# ── 五、light 路径接线 ────────────────────────────────────────────────────


def test_light_series_shape_and_repeatable_sha():
    a = wc.run_pass(42, ticks=120, light=True, **SMALL)
    b = wc.run_pass(42, ticks=120, light=True, **SMALL)
    assert len(a["series"]) == 121 and a["series"][0] == SMALL["init_pop"]
    assert a["series"] == b["series"]
    assert a["series_sha256"] == b["series_sha256"]
    assert a["params"]["light"] is True and a["params"]["series_every"] == 1
    # 轻量路径不产生事后全量统计（键在但值为 None，JSON 里可辨识）
    assert a["run_digest"] is None and a["behaviors"] is None
    assert a["lineage"] is None and a["action_counts"] is None
    assert a["convergence"] is None            # ticks < 1000 ⇒ 不判收敛


def test_light_matches_full_path_small_world():
    """修订 1 的核心论证：关掉观测累积不改变轨迹。"""
    full = wc.run_pass(42, ticks=120, **SMALL)
    light = wc.run_pass(42, ticks=120, light=True, **SMALL)
    assert light["population"]["alive_probes"] == full["population"]["alive_probes"]
    assert light["population"]["final_alive"] == full["population"]["final_alive"]
    assert light["population"]["total_born"] == full["population"]["total_born"]
    assert light["population"]["total_died"] == full["population"]["total_died"]
    assert light["population"]["death_causes"] == full["population"]["death_causes"]
    for key in ("amount_sha256", "capacity_sha256", "world_grid_sha256"):
        assert light["resource"][key] == full["resource"][key]
    assert light["resource"]["sync_count"] == full["resource"]["sync_count"]
    assert light["weather"]["samples"] == full["weather"]["samples"]
    assert light["atmo"]["digest"] == full["atmo"]["digest"]
    assert light["series"][-1] == full["population"]["final_alive"]


def test_light_series_every_is_thinned():
    p = wc.run_pass(42, ticks=100, light=True, series_every=25, **SMALL)
    assert p["series_steps"] == [0, 25, 50, 75, 100]
    assert len(p["series"]) == 5
    assert p["convergence"] is None


def test_light_guards():
    with pytest.raises(ValueError, match="series_every"):
        wc.run_pass(42, ticks=10, series_every=0, **SMALL)
    with pytest.raises(ValueError, match="只在 light"):
        wc.run_pass(42, ticks=10, baseline={"population": {}}, **SMALL)
    with pytest.raises(ValueError, match="series_every=1"):
        wc.run_pass(42, ticks=10, light=True, series_every=5,
                    baseline={"population": {}}, **SMALL)
    with pytest.raises(ValueError, match="equiv_at"):
        wc.run_pass(42, ticks=100, light=True,
                    baseline={"population": {}}, **SMALL)


def test_light_determinism_key_uses_series_sha():
    a = wc.run_pass(42, ticks=120, light=True, **SMALL)
    b = wc.run_pass(42, ticks=120, light=True, **SMALL)
    assert wc._det_key_light(a) == wc._det_key_light(b)
    c = wc.run_pass(43, ticks=120, light=True, **SMALL)
    assert wc._det_key_light(a) != wc._det_key_light(c)


def test_run_batch_light_summary_and_cost_abort(tmp_path):
    """light 批：schema/判读注记/流式墙钟；成本中止线可触发。"""
    out = tmp_path / "b19b.json"
    s = wc.run_batch([42], ticks=1000, repeat=2, light=True,
                     baseline_path=None, prereg=wc.B19B_PREREG_PATH,
                     out=out, **SMALL)
    assert s["schema"] == "biosphere-b19b-v1"
    assert s["prereg"] == wc.B19B_PREREG_PATH
    assert s["baseline"] is None
    assert s["equivalence"]["all_passed"] is None
    assert s["convergence"]["formal_judgement"] is False
    assert s["convergence"]["conclusion"] == "not_formal_tmax"
    assert s["seeds"][0]["determinism"]["equal"] is True
    log = out.with_suffix(".passes.jsonl")
    import json
    recs = [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines()]
    assert len(recs) == 2 and all(r["light"] for r in recs)
    assert all(r["series_sha256"] for r in recs)
    assert recs[0]["equivalence_passed"] is False   # 未过门（无基线）

    with pytest.raises(RuntimeError, match="COST-ABORT"):
        wc.run_batch([42], ticks=1000, repeat=2, light=True,
                     baseline_path=None, max_pass_abort_s=0.0, **SMALL)


def test_run_batch_light_guards():
    with pytest.raises(ValueError, match="每步采样"):
        wc.run_batch([42], ticks=100, light=True, series_every=5,
                     baseline_path=None, **SMALL)
    with pytest.raises(ValueError, match="必须给 B19 基线"):
        wc.run_batch([42], ticks=wc.B19B_TMAX, light=True,
                     baseline_path=None, **SMALL)


@_B19_BASELINE_SKIP
def test_run_batch_rejects_unknown_seed_against_baseline():
    """seed 不在基线里 ⇒ 开跑前拒绝（需基线在场才能走到 seed 校验）。"""
    with pytest.raises(ValueError, match="缺少 seed"):
        wc.run_batch([99], ticks=wc.B19B_TMAX, light=True, **SMALL)


def test_run_batch_short_series_skips_verdict(tmp_path):
    s = wc.run_batch([42], ticks=200, repeat=2, light=True,
                     baseline_path=None, **SMALL)
    conv = s["convergence"]
    assert conv["series_too_short"] is True
    assert conv["per_seed"] == {} and conv["K_cross_seed"] is None
    assert conv["conclusion"] == "not_formal_tmax"
    assert "序列不足" in conv["note"] and "ticks=200" in conv["note"]


# ── 六、2,000 tick 端到端等价性门（真跑；长跑不变量 ⇒ slow） ─────────────


@pytest.mark.slow
def test_equivalence_gate_end_to_end_at_2000_ticks():
    """完整路径当基线 ⇒ light 必须过门；扰动基线 ⇒ 抛 EquivalenceError。

    为什么慢是必要的：预注册 §5.2 的检查点固定在 2,000 步，缩短就检验不到
    "轻量路径没有改变 2,000 步之前的轨迹"这条合同（小世界 2,000 tick ≈5 s）。
    """
    full = wc.run_pass(42, ticks=wc.B19B_EQUIV_AT, **SMALL)
    baseline = {
        "population": {
            "alive_probes": full["population"]["alive_probes"],
            "final_alive": full["population"]["final_alive"],
            "total_born": full["population"]["total_born"],
            "total_died": full["population"]["total_died"],
        },
        "weather": {"samples": full["weather"]["samples"]},
    }
    light = wc.run_pass(42, ticks=wc.B19B_EQUIV_AT, light=True,
                        baseline=baseline, **SMALL)
    eq = light["equivalence"]
    assert eq["passed"] is True, eq["mismatches"]
    assert eq["at_step"] == 2000
    assert light["series"][-1] == full["population"]["final_alive"]
    assert light["population"]["total_born"] == full["population"]["total_born"]

    bad = copy.deepcopy(baseline)
    bad["population"]["total_born"] += 1
    with pytest.raises(wc.EquivalenceError, match="total_born"):
        wc.run_pass(42, ticks=wc.B19B_EQUIV_AT, light=True,
                    baseline=bad, **SMALL)

    bad2 = copy.deepcopy(baseline)
    bad2["weather"]["samples"][10]["t2m_c_mean"] += 0.5
    with pytest.raises(wc.EquivalenceError, match="climate"):
        wc.run_pass(42, ticks=wc.B19B_EQUIV_AT, light=True,
                    baseline=bad2, **SMALL)
