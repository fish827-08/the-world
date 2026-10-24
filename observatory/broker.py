"""observatory.broker：快照桥（模块四 · WebSocket 实时推送）。

通俗理解
--------
观察台统计是"离线落盘"（CSV/JSON 文件）；快照桥是"实时直播"：
模拟每走 interval 个 tick，就把当前世界状态打包成一份 JSON ——
全局尺度（活着几只/食物剩多少/平均能量）+ 个体明细（每个个体
的 id/坐标/能量/世代）——推送给所有连上的 WebSocket 客户端
（前端渲染消费用，见 PROJECT-DESCRIPTION 快照推送约定：每 100 tick）。

对外接口
--------
- SnapshotBroker(engine, interval, max_snapshots)
    · pump(ticks=None, observer=None)   同步泵：推进引擎，按节拍采集；
      可选挂一个 EvolutionObserver 同步采样（长程实验 + 直播两用）。
    · snapshots                        有界环形缓冲（标量版快照列表）
    · snapshot(include_individuals)    现时刻一份快照 dict（JSON 安全）
    · _queue                           广播队列（泵线程 → asyncio 广播）
- async serve(broker, host, port, ticks=None)
    启动 WebSocket 服务：泵在后台线程跑，快照经队列逐条广播。
- main(argv)   命令行入口：py -m observatory.broker --port 8765 --ticks 10000

线程模型（rng/ndarray 都留在泵线程，broker 不作跨线程共享）
- 泵线程：engine.step() + 采集 + 入队（只碰引擎）；
- asyncio 主循环：从队列取快照 → 序列化 → 广播给所有客户端。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import queue
import threading
from collections import deque
from contextlib import suppress
from typing import Optional

import numpy as np

from simulation.config import PopulationConfig, SimConfig, SimulationConfig
from simulation.sphere_engine import SphereEngine

_END_MARK = object()  # 泵线程结束哨兵（先进先出的停机信号）


class SnapshotBroker:
    """一台引擎的实时快照泵：每 interval tick 采一份 JSON 快照。"""

    def __init__(
        self,
        engine: SphereEngine,
        interval: int = 100,
        max_snapshots: int = 4_096,
    ) -> None:
        assert interval >= 1, "快照节拍必须 ≥ 1 tick"
        self.engine = engine
        self.interval = interval
        self._ring: deque[dict] = (
            deque(maxlen=max_snapshots) if max_snapshots > 0 else deque()
        )
        self._queue: "queue.Queue[dict]" = queue.Queue()
        self._next_tick = interval  # 首个快照出现在 tick == interval

    # ---- 只读 ----------------------------------------------------------

    @property
    def snapshots(self) -> list[dict]:
        """已采集的标量快照（有界环形，最新的存尾部）。"""
        return list(self._ring)

    # ---- 采集 ----------------------------------------------------------

    def snapshot(self, include_individuals: bool = False) -> dict:
        """当前时刻一份快照（JSON 安全；id/坐标转 Python int）。

        include_individuals=True 时携带个体明细（前端渲染用）；
        广播带明细、环形缓存只存标量版（避免 5000 个体 × 4096 份 OOM）。
        """
        e = self.engine
        individuals: list[dict] = []
        if include_individuals and len(e._id) > 0:
            individuals = [
                {
                    "id": int(i),
                    "flat": int(f),
                    "energy": round(float(en), 6),
                    "generation": int(g),
                    "age": int(a),
                }
                for i, f, en, g, a in zip(e._id, e._flat, e._energy, e._generation, e._age)
            ]
        n = len(e._id)
        return {
            "tick": int(e.tick),
            "population": n,
            "max_generation": int(e._max_generation),
            "total_energy": round(float(e._energy.sum()), 6),
            "total_resource": round(float(e.resources.total()), 6),
            "mean_age": round(float(np.asarray(e._age).mean()), 4) if n else 0.0,
            "mean_energy": round(float(np.asarray(e._energy).mean()), 6) if n else 0.0,
            "individuals": individuals,
        }

    def pump(
        self,
        ticks: Optional[int] = None,
        observer=None,
    ) -> list[dict]:
        """同步泵：推进引擎直到自然结束或 ticks；每 interval 采一次快照。

        observer 为可选的 EvolutionObserver，逐 tick 喂入统计
        （离线实验与实时直播共用同一台引擎时的采样口）。
        返回本次泵送新采集的标量快照列表。
        """
        limit = self.engine.config.simulation.ticks if ticks is None else ticks
        collected: list[dict] = []
        while not self.engine.finished and self.engine.tick < limit:
            stats = self.engine.step()
            if observer is not None:
                observer.observe(stats)
            if self.engine.tick >= self._next_tick:
                scalar = self.snapshot(include_individuals=False)
                self._ring.append(scalar)
                collected.append(scalar)
                # 广播版带个体明细（实时直播前端用）
                self._queue.put(self.snapshot(include_individuals=True))
                self._next_tick += self.interval
        # 与 engine.run() 对齐：出循环后按终止条件置位（CLI 据此退出）
        self.engine._finished = self.engine._end_condition_met()
        return collected


# ---- WebSocket 服务 -------------------------------------------------------


class SnapshotServer:
    """一台 websockets 服务：泵线程跑引擎，主循环广播快照。

    用法（异步上下文管理器）：
        async with serve(broker, host, port, ticks) as server:
            ...
    """

    def __init__(
        self,
        broker: SnapshotBroker,
        host: str = "127.0.0.1",
        port: int = 8765,
        ticks: Optional[int] = None,
    ) -> None:
        self.broker = broker
        self.host = host
        self.requested_port = port
        self.ticks = ticks
        self._clients: set = set()
        self.server = None  # websockets Server，__aenter__ 后才有

    # ---- 生命周期 ------------------------------------------------------

    async def __aenter__(self) -> "SnapshotServer":
        from websockets.asyncio.server import serve

        self.server = await serve(self._handler, self.host, self.requested_port)
        self._pump_task = asyncio.create_task(self._pump_broadcast())
        return self

    async def __aexit__(self, *exc) -> None:
        self._pump_task.cancel()
        # CancelledError 继承 BaseException（Python 3.8+），不能用 suppress(Exception)，
        # 否则 pump 任务恰好在 asyncio.to_thread 中间被取消时会漏出未抑制的 CancelledError。
        with suppress(BaseException):
            await self._pump_task
        if self.server is not None:
            self.server.close()
            with suppress(BaseException):
                await self.server.wait_closed()

    # ---- 属性 ----------------------------------------------------------

    @property
    def port(self) -> int:
        """实际监听端口（port=0 时由系统分配）。"""
        return self.server.sockets[0].getsockname()[1]

    @property
    def finished(self) -> bool:
        return self.broker.engine.finished

    # ---- 内部 ----------------------------------------------------------

    async def _handler(self, ws) -> None:
        self._clients.add(ws)
        try:
            # 新客户端先补发最新标量快照（落点晚也能立刻看到画面）
            latest = self.broker.snapshots[-1] if self.broker.snapshots else None
            if latest is not None:
                await ws.send(json.dumps(latest, ensure_ascii=False))
            async for _ in ws:  # 保持连接直到客户端断开
                pass
        finally:
            self._clients.discard(ws)

    def _pump_worker(self) -> None:
        self.broker.pump(ticks=self.ticks)
        self.broker._queue.put(_END_MARK)

    async def _pump_broadcast(self) -> None:
        thread = threading.Thread(
            target=self._pump_worker, name="snapshot-pump", daemon=True
        )
        thread.start()
        while True:
            snap = await asyncio.to_thread(self.broker._queue.get)
            if snap is _END_MARK:
                break
            msg = json.dumps(snap, ensure_ascii=False)
            for client in list(self._clients):
                try:
                    await client.send(msg)
                except Exception:
                    self._clients.discard(client)
        thread.join(timeout=1.0)


def serve(
    broker: SnapshotBroker,
    host: str = "127.0.0.1",
    port: int = 8765,
    ticks: Optional[int] = None,
) -> SnapshotServer:
    """快照服务入口：async with serve(broker, ...) as server: ..."""
    return SnapshotServer(broker, host=host, port=port, ticks=ticks)


# ---- 独立入口：python -m observatory.broker -------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Snapshot bridge（模块四 · WebSocket 快照推流）")
    p.add_argument("--rows", type=int, default=60)
    p.add_argument("--cols", type=int, default=120)
    p.add_argument("--count", type=int, default=200, help="初始个体数")
    p.add_argument("--ticks", type=int, default=10_000)
    p.add_argument("--interval", type=int, default=100, help="快照节拍（tick）")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sim-core", action="store_true")
    args = p.parse_args(argv)

    from simulation.config import WorldConfig

    cfg = SimConfig(
        seed=args.seed,
        world=WorldConfig(rows=args.rows, cols=args.cols),
        simulation=SimulationConfig(ticks=args.ticks, use_sim_core=args.sim_core),
        population=PopulationConfig(initial_count=args.count),
    )
    engine = SphereEngine(cfg)
    broker = SnapshotBroker(engine, interval=args.interval)

    async def app() -> None:
        async with serve(
            broker, host=args.host, port=args.port, ticks=args.ticks
        ) as server:
            print(
                f"快照桥已启动: ws://{args.host}:{server.port}  "
                f"（泵 {args.ticks} tick，节拍 {args.interval} tick）",
                flush=True,
            )
            while not engine.finished:
                await asyncio.sleep(0.5)
            print("模拟结束，已推流最后一批快照。", flush=True)

    asyncio.run(app())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())