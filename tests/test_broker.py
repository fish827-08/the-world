"""快照桥（broker）的测试：节拍采集 / 环形缓冲 / JSON 序列化 / WebSocket 推流。"""
from __future__ import annotations

import asyncio
import json

from observatory.broker import SnapshotBroker, serve
from simulation.config import SimConfig, SimulationConfig, WorldConfig
from simulation.sphere_engine import SphereEngine

_KEYS = {"tick", "population", "max_generation", "total_energy", "total_resource",
         "mean_age", "mean_energy", "individuals"}


def _engine(ticks: int = 300, count: int = 50) -> SphereEngine:
    cfg = SimConfig(
        seed=3,
        world=WorldConfig(rows=12, cols=16),
        simulation=SimulationConfig(ticks=ticks),
        population=PopulationConfigShort(initial_count=count),
    )
    return SphereEngine(cfg)


# ---- 同步泵 -------------------------------------------------------------


def test_pump_collects_at_interval() -> None:
    eng = _engine(ticks=300, count=40)
    broker = SnapshotBroker(eng, interval=100)
    snaps = broker.pump()
    # tick 100 / 200 / 300 各采一次 → 3 份标量快照
    assert [s["tick"] for s in snaps] == [100, 200, 300]
    assert broker.snapshots == snaps
    assert eng.finished  # 泵出循环后已置完成标志


def test_pump_ring_bounded() -> None:
    eng = _engine(ticks=300, count=30)
    broker = SnapshotBroker(eng, interval=50, max_snapshots=2)
    broker.pump()
    assert len(broker.snapshots) == 2
    assert broker.snapshots[-1]["tick"] == 300  # 只留最新 2 份


def test_snapshot_json_safe_and_scalar() -> None:
    eng = _engine(ticks=10, count=25)
    broker = SnapshotBroker(eng, interval=5)
    broker.pump()
    snap = broker.snapshots[-1]
    assert set(snap.keys()) == _KEYS
    assert snap["individuals"] == []  # 环形缓存只存标量版
    text = json.dumps(snap, ensure_ascii=False)  # 必须可序列化
    assert "tick" in text


def test_broadcast_snapshot_carries_individuals() -> None:
    eng = _engine(ticks=10, count=25)
    broker = SnapshotBroker(eng, interval=5)
    full = broker.snapshot(include_individuals=True)
    assert len(full["individuals"]) == 25
    one = full["individuals"][0]
    assert {"id", "flat", "energy", "generation", "age"} == set(one.keys())
    assert isinstance(one["id"], int) and isinstance(one["flat"], int)


# ---- WebSocket 推流 ---------------------------------------------------------


def test_ws_streams_snapshots() -> None:
    """端到端：起服务 → 客户端连上 → 收到兜底快照 + 带个体的推流快照。"""

    async def scenario() -> list[dict]:
        eng = _engine(ticks=300, count=120)  # 稍大种群，让泵有时间推流
        broker = SnapshotBroker(eng, interval=100)
        from websockets.asyncio.client import connect

        received: list[dict] = []
        async with serve(broker, host="127.0.0.1", port=0, ticks=300) as server:
            uri = f"ws://127.0.0.1:{server.port}"
            async with connect(uri, open_timeout=5) as ws:
                ws_ctx = ws
                while len(received) < 3:
                    try:
                        raw = await asyncio.wait_for(ws_ctx.recv(), timeout=20)
                    except (TimeoutError, asyncio.TimeoutError):
                        break
                    received.append(json.loads(raw))
        return received

    received = asyncio.run(scenario())
    assert len(received) >= 2, f"只收到 {len(received)} 份"
    assert all("population" in m for m in received)
    ticks = [m["tick"] for m in received]
    assert len(set(ticks)) > 1  # 兜底 + 推流至少两份不同 tick
    assert any(m["individuals"] for m in received), "推流快照应携带个体明细"


def test_ws_runs_to_completion_without_clients() -> None:
    """无客户端时服务也要正常跑完（不因没有订阅者而卡住）。"""

    async def scenario() -> bool:
        eng = _engine(ticks=150, count=40)
        broker = SnapshotBroker(eng, interval=50)
        async with serve(broker, host="127.0.0.1", port=0, ticks=150) as server:
            while not eng.finished:
                await asyncio.sleep(0.02)
            return broker.snapshots[-1]["tick"] == 150

    assert asyncio.run(scenario())


from simulation.config import PopulationConfig as PopulationConfigShort  # noqa: E402