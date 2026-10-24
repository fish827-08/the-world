"""B18：生物进真世界联调的适配层 + 耦合运行测试。

覆盖预注册 J4 oracle：
1. ``sample_climate`` 的陆掩码/统计/出口调用；
2. ``sync_real_capacity`` 逐位复制容量、不动存量、形状守卫；
3. 小世界同 seed 两次耦合运行逐位一致；
4. s1 seed42 t=0 容量 sum 命中 A20 锚；
5. EAT 1 食物单位 = 1 能量单位（U4 账本契约）。

TIER = "mainline"；全部进档②。
另含 B19 上限接线两例（``max_pop`` 透传 / J4 cap-not-binding）
与收口两例（``prereg`` 参数化 / pass 级墙钟流式 + 超线告警）。
"""
from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from biosphere.action import ActionKind, Intent
from biosphere.develop import ControllerParams
from biosphere.genome import Genome
from biosphere.resolve import (
    Individual, Population, commit_intents, resolve_tick,
)
from biosphere.world_adapter import (
    SURFACE_LAND, SyntheticResource, WorldView, _von_neumann_table,
    sample_climate, sync_real_capacity,
)
from biosphere import world_coupled as wc

TIER = "mainline"


class _FakeSource:
    """只实现 B18 消费的两个出口。"""

    def __init__(self) -> None:
        self.calls: list[tuple[list[int], int]] = []
        self.pr = np.array([100.0, 0.0, 300.0, 500.0], dtype=np.float64)

    def temperature_c(self, flat, tick):
        f = np.asarray(flat, dtype=np.int64)
        self.calls.append((f.tolist(), int(tick)))
        return 20.0 + 0.1 * f.astype(np.float64)

    def precip_mm_yr(self):
        return self.pr.copy()


class _FakeResourceField:
    def __init__(self, capacity) -> None:
        self._capacity = np.asarray(capacity, dtype=np.float64)


def test_sample_climate_land_mask_and_stats():
    src = _FakeSource()
    land = np.array([True, False, True, True])
    s = sample_climate(src, land, tick=7)
    assert s.tick == 7
    assert s.n_land == 3
    assert s.t2m_c_min == pytest.approx(20.0)
    assert s.t2m_c_max == pytest.approx(20.3)
    assert s.t2m_c_mean == pytest.approx((20.0 + 20.2 + 20.3) / 3.0)
    assert s.precip_mm_yr_min == pytest.approx(100.0)
    assert s.precip_mm_yr_max == pytest.approx(500.0)
    assert s.precip_mm_yr_mean == pytest.approx((100.0 + 300.0 + 500.0) / 3.0)
    # 只请求陆格，且 tick 原样透传
    assert src.calls == [([0, 2, 3], 7)]

    with pytest.raises(ValueError):
        sample_climate(src, np.zeros(4, dtype=bool), tick=0)

    class _BadPr(_FakeSource):
        def __init__(self) -> None:
            super().__init__()
            self.pr = np.array([1.0, 2.0], dtype=np.float64)

    with pytest.raises(ValueError):
        sample_climate(_BadPr(), land, tick=0)


def test_sync_real_capacity_copies_exact_and_keeps_amount():
    resource = SyntheticResource(
        amount=np.array([9.0, 8.0, 7.0]),
        capacity=np.zeros(3, dtype=np.float64),
        regrow_rate=0.5,
    )
    before = resource.amount.copy()
    n = sync_real_capacity(resource, _FakeResourceField([1.0, 2.0, 3.0]))
    assert n == 3
    assert np.array_equal(resource.capacity, np.array([1.0, 2.0, 3.0]))
    # 只同步容量：存量不变（不凭空补/删食物）
    assert np.array_equal(resource.amount, before)

    with pytest.raises(ValueError):
        sync_real_capacity(resource, _FakeResourceField([1.0, 2.0]))

    class _Missing:
        pass

    with pytest.raises(NotImplementedError):
        sync_real_capacity(resource, _Missing())


def _ctrl(**overrides) -> ControllerParams:
    base = dict(
        sense_food=1.0, sense_threat=1.0, sense_kin=1.0, sense_signal=1.0,
        move_bias=0.5, eat_greed=0.5, attack_drive=0.0, share_drive=0.0,
        signal_drive=0.0, repro_drive=0.5, metabolism=1.0, move_cost_mult=1.0,
        lifespan=1000.0, repro_thresh=40.0, habitat_aquatic=1.0,
        habitat_terrestrial=1.0, adhesion=0.0, defense=0.0, eat_amount=0.5,
        signal_payload=0.5, give_up=10.0, persistence=0.0,
    )
    base.update(overrides)
    return ControllerParams(**base)


def test_eat_one_to_one():
    """U4：EAT 从资源扣 gain、向能量加同一 gain（1:1）。"""
    n = 16
    world = WorldView(
        rows=4, cols=4, n_cells=n,
        surface_type=np.full(n, SURFACE_LAND, dtype=np.uint8),
        elevation_m=np.zeros(n), depth_m=np.zeros(n),
        patch_id=np.full(n, -1, dtype=np.int32),
        neighbors=_von_neumann_table(4, 4),
    )
    pop = Population()
    ind = Individual(uid=1, genome=Genome(genes=np.zeros(64)),
                     cell=0, energy=50.0)
    pop.individuals.append(ind)
    pop.next_id = 2
    amounts = np.zeros(n, dtype=np.float64)
    amounts[0] = 1.0
    resource = SyntheticResource(
        amount=amounts, capacity=np.maximum(amounts, 1.0), regrow_rate=0.0)
    intent = Intent(actor_id=1, kind=ActionKind.EAT, target_cell=0,
                    energy_cost=0.05)
    report = resolve_tick(
        pop, commit_intents(pop, [intent]), world, resource,
        np.zeros(n), tick=0, rng=np.random.default_rng(0),
        controllers={1: _ctrl(eat_amount=0.4)})
    assert resource.amount[0] == pytest.approx(0.6)
    assert ind.energy == pytest.approx(50.0 - 0.05 + 0.4)
    assert report.energy_in == pytest.approx(0.4)


def test_coupled_run_is_deterministic():
    kw = dict(ticks=40, init_pop=8, rows=24, cols=48, nlat=8, nlon=16)
    a = wc.run_pass(42, **kw)
    b = wc.run_pass(42, **kw)
    assert a["run_digest"] == b["run_digest"]
    assert a["resource"]["amount_sha256"] == b["resource"]["amount_sha256"]
    assert a["resource"]["capacity_sha256"] == b["resource"]["capacity_sha256"]
    assert a["resource"]["world_grid_sha256"] == b["resource"]["world_grid_sha256"]
    assert a["population"]["final_alive"] == b["population"]["final_alive"]
    assert a["population"]["total_born"] == b["population"]["total_born"]
    assert a["population"]["total_died"] == b["population"]["total_died"]
    assert a["weather"]["samples"] == b["weather"]["samples"]
    assert a["atmo"]["digest"] == b["atmo"]["digest"]


def test_s1_seed42_capacity_matches_a20_anchor():
    eng, src = wc.build_weather_world(42)
    land = np.asarray(eng.env_field.land_mask, dtype=bool).reshape(-1)
    cap = np.asarray(eng.resources._capacity, dtype=np.float64).reshape(-1)
    total = float(cap[land].sum())
    rel = abs(total - wc.A20_S1_MULT_CAP_SUM) / wc.A20_S1_MULT_CAP_SUM
    assert rel <= wc.U1_REL_TOL
    s = sample_climate(src, land, tick=0)
    assert s.n_land == int(land.sum())
    assert np.isfinite(s.t2m_c_mean)
    assert np.isfinite(s.precip_mm_yr_mean)


def test_run_pass_forwards_max_pop():
    """B19：max_pop 必须透传到 ResolveConfig（cap=1 禁繁殖、大 cap 放开）。"""
    kw = dict(ticks=60, init_pop=8, rows=24, cols=48, nlat=8, nlon=16)
    capped = wc.run_pass(42, max_pop=1, **kw)
    open_ = wc.run_pass(42, max_pop=100000, **kw)
    assert capped["params"]["max_pop"] == 1
    assert open_["params"]["max_pop"] == 100000
    assert capped["population"]["total_born"] == 0
    assert capped["action_counts"].get("reproduce:repro_no_room", 0) > 0
    assert open_["population"]["total_born"] > 0
    assert wc.run_pass(42, **kw)["params"]["max_pop"] == 512


def test_judgements_cap_not_binding():
    """B19 J4：cap=1 必绑定、cap=100000 在 24×48 小世界不绑定。"""
    kw = dict(ticks=60, init_pop=8, repeat=2, rows=24, cols=48,
              nlat=8, nlon=16)
    capped = wc.run_batch([42], max_pop=1, **kw)
    open_ = wc.run_batch([42], max_pop=100000, **kw)
    assert capped["judgements"]["J4_cap_not_binding"]["pass"] is False
    assert open_["judgements"]["J4_cap_not_binding"]["pass"] is True
    assert open_["judgements"]["J4_cap_not_binding"]["peak_max"] < 100000


@pytest.mark.parametrize("ticks,expected", [(40, 1), (80, 3), (200, 8)])
def test_judgements_u3_expected_sync_follows_ticks(ticks, expected):
    """U3 期望同步次数 = ticks // 25（6,000t 的 240 不能再用 2,000t 的 80）。"""
    kw = dict(ticks=ticks, init_pop=8, repeat=2, rows=24, cols=48,
              nlat=8, nlon=16)
    s = wc.run_batch([42], max_pop=100000, **kw)
    u3 = s["judgements"]["J3_units"]["U3_capacity_sync"]
    assert u3["expected_sync_count"] == expected
    assert [r["sync_count"] for r in u3["rows"]] == [expected]
    assert u3["pass"] is True
    for row in u3["rows"]:
        assert row["ok"] is True
        assert row["sync_max_abs_diff"] == 0.0


def test_judgements_u3_rejects_stale_sync_count():
    """过期的 80（2,000t 门常量）套到 200t 批上必须判 FAIL，不得静默通过。"""
    kw = dict(ticks=200, init_pop=8, repeat=2, rows=24, cols=48,
              nlat=8, nlon=16)
    s = wc.run_batch([42], max_pop=100000, **kw)
    per_seed = copy.deepcopy(s["seeds"])
    assert per_seed[0]["passes"][0]["resource"]["sync_count"] == 8
    per_seed[0]["passes"][0]["resource"]["sync_count"] = 80
    j = wc._judgements(per_seed, 2, 100000, 200)
    assert j["J3_units"]["U3_capacity_sync"]["pass"] is False
    assert j["J3_units"]["U3_capacity_sync"]["rows"][0]["ok"] is False


def test_run_batch_prereg_metadata_is_parameterized():
    """B19 收口：prereg 路径可传；默认仍 B18（旧读数口径不变）。"""
    kw = dict(ticks=40, init_pop=8, repeat=2, rows=24, cols=48,
              nlat=8, nlon=16)
    default = wc.run_batch([42], **kw)
    custom = wc.run_batch([42], prereg=wc.B19_PREREG_PATH, **kw)
    assert default["prereg"] == wc.B18_PREREG_PATH
    assert custom["prereg"] == wc.B19_PREREG_PATH


def test_run_batch_streams_pass_walls_and_alerts(tmp_path):
    """B19 收口：每 pass 结束即落盘 JSONL；超线立即告警（不依赖末次汇总）。"""
    kw = dict(ticks=40, init_pop=8, repeat=2, rows=24, cols=48,
              nlat=8, nlon=16)
    out = tmp_path / "batch.json"
    s = wc.run_batch([42], out=out, max_pass_s=0.0, **kw)
    log = out.with_suffix(".passes.jsonl")
    recs = [json.loads(line)
            for line in log.read_text(encoding="utf-8").splitlines()]
    assert [(r["seed"], r["pass"]) for r in recs] == [(42, 0), (42, 1)]
    assert all(r["over_budget"] for r in recs)
    assert all(r["prereg"] == wc.B18_PREREG_PATH for r in recs)
    assert s["wall_log"] == str(log)
    assert len(s["wall_alerts"]) == 2
    assert out.exists()
