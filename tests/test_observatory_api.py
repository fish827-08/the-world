"""observatory.api 的 smoke 测试：app 可导入 / 端点契约 / fail-loud / 轻量拉起。

🔴 **不拉任何重量级 s2 run**（装置档 s2 = 480×960 = 460,800 格）：
- 拉起类测试用 tmp_path 里的**假脚本**（认 --device/--seed/--ticks/--out，秒退）；
- 快照类测试用 8×12 的**微型引擎**（构造 ~0.3s，save_snapshot ~0.01s）；
- 数据仓根一律 monkeypatch 到 tmp_path（不碰真 `the-world-data/`）。

覆盖：路由存在性、/health、/devices 白名单、GET /experiments（空/404/过滤）、
GET /datapacks（空/索引 viz+live/kind 过滤/坏 kind）、POST /experiments 的
四类拒绝（未知 device / 裸 rows-cols / 缺字段 / 脚本不存在）、真拉起（非 live 与
live-supervise 两条路径）、POST /snapshots（inproc 真引擎 / live 子进程 / 404 / 409）、
POST /experiments/{id}/cancel。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import observatory.api as api
from tools.device_presets import preset_names

_EXPECTED_PATHS = {
    "/health", "/devices", "/experiments", "/experiments/{run_id}",
    "/experiments/{run_id}/cancel", "/snapshots", "/snapshots/export", "/datapacks",
}

#: 假脚本（非 live）：解析 API 注入的参数，写一份 marker，秒退
_FAKE_SCRIPT = '''"""smoke 用假实验脚本（秒退，不碰引擎）。"""
import argparse, json, sys
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--device")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--ticks", type=int)
    ap.add_argument("--arm", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sample", type=int, default=None)
    a = ap.parse_args(argv if argv is not None else sys.argv[1:])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "marker.json").write_text(json.dumps({
        "device": a.device, "seed": a.seed, "ticks": a.ticks, "arm": a.arm,
        "out": str(out.resolve()), "sample": a.sample,
        "argv": list(argv if argv is not None else sys.argv[1:]),
    }, ensure_ascii=False), encoding="utf-8")
    print("FAKE-RUN-OK device=%s seed=%s ticks=%s" % (a.device, a.seed, a.ticks))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''

#: 假脚本（live）：构造微型真引擎并存活一段时间，供 POST /snapshots 打快照
_FAKE_LIVE_SCRIPT = '''"""smoke 用假实验脚本（live）：构造 8x12 微型引擎，等快照请求。"""
import argparse, sys, time
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--device")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--ticks", type=int, default=5)
    ap.add_argument("--arm", type=int, default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv if argv is not None else sys.argv[1:])

    from simulation.config import (PopulationConfig, SimConfig, SimulationConfig,
                                   WorldConfig)
    from simulation.sphere_engine import SphereEngine

    cfg = SimConfig(
        seed=a.seed,
        world=WorldConfig(rows=8, cols=12),
        simulation=SimulationConfig(ticks=a.ticks),
        population=PopulationConfig(initial_count=5),
    )
    eng = SphereEngine(cfg)          # ← supervisor 的钩子在这里捕获引擎
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "READY").write_text("ok", encoding="utf-8")
    print("READY", flush=True)

    deadline = time.time() + 180
    while time.time() < deadline:
        try:
            if not eng.finished:
                eng.step()
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            print("step 停止：%r" % (exc,), flush=True)
        time.sleep(0.05)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def _wait_until(pred, timeout: float = 120.0, interval: float = 0.25, msg: str = ""):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return
        time.sleep(interval)
    raise AssertionError(f"超时（{timeout}s）：{msg}")


def _wait_ready(client, run_id: str, out_dir: Path, timeout: float = 120.0) -> None:
    """等 supervise 子进程把引擎建好（READY）；子进程若提前死掉，**立即**带日志报错。"""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if (out_dir / "READY").is_file():
            return
        st = client.get(f"/experiments/{run_id}").json()["status"]
        if st in (api.STATUS_FINISHED, api.STATUS_FAILED):
            d = client.get(f"/experiments/{run_id}", params={"tail_lines": 40}).json()
            raise AssertionError(
                f"子进程提前结束（status={d['status']} reason={d['ended_reason']} "
                f"rc={d['returncode']}）；日志尾巴：{d['log_tail']}")
        time.sleep(0.25)
    tail = client.get(f"/experiments/{run_id}", params={"tail_lines": 40}).json()["log_tail"]
    raise AssertionError(f"超时（{timeout}s）：未见 READY；日志尾巴：{tail}")


def _tiny_engine(ticks: int = 3):
    from simulation.config import (PopulationConfig, SimConfig, SimulationConfig,
                                   WorldConfig)
    from simulation.sphere_engine import SphereEngine

    cfg = SimConfig(
        seed=1,
        world=WorldConfig(rows=8, cols=12),
        simulation=SimulationConfig(ticks=ticks),
        population=PopulationConfig(initial_count=5),
    )
    eng = SphereEngine(cfg)
    for _ in range(2):
        eng.step()
    return eng


def _fake_run(run_id: str, data_root: Path, live: bool = False,
              status: str = api.STATUS_FINISHED) -> api.RunState:
    """直接造一条 run 台账（不打子进程），用于快照端点的分支测试。"""
    out_dir = data_root / "live" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    run = api.RunState(
        run_id=run_id, name=run_id, device=preset_names()[0], seed=1, ticks=5,
        arm=None, script="experiments/_fake_.py", args=[], command=[], live=live,
        out_dir=str(out_dir), log_path=str(out_dir / api._LOG_NAME),
        status=status, returncode=0 if status == api.STATUS_FINISHED else None,
        ended_reason="exit_ok" if status == api.STATUS_FINISHED else None,
    )
    api.REGISTRY.add(run)
    return run


@pytest.fixture()
def data_root(tmp_path, monkeypatch):
    root = tmp_path / "the-world-data"
    (root / "live").mkdir(parents=True)
    (root / "viz").mkdir(parents=True)
    monkeypatch.setenv("THE_WORLD_DATA_ROOT", str(root))
    monkeypatch.setenv("THE_WORLD_API_ALLOW_EXTERNAL_SCRIPT", "1")  # 允许跑 tmp 里的假脚本
    api.REGISTRY.reset()
    yield root
    api.REGISTRY.reset()


@pytest.fixture()
def client(data_root):
    with TestClient(api.app) as c:
        yield c


# ---- app 与元信息 -----------------------------------------------------------


def test_app_importable_and_routes_registered() -> None:
    assert api.app is not None
    paths = {getattr(r, "path", None) for r in api.app.routes}
    missing = _EXPECTED_PATHS - paths
    assert not missing, f"缺路由：{missing}"


def test_health(client, data_root) -> None:
    r = client.get("/health")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "ok"
    assert body["api_version"] == api.API_VERSION
    assert body["devices"] == preset_names()
    assert Path(body["data_root"]) == data_root.resolve()
    assert body["data_root_exists"] is True
    assert body["default_script"] == api.DEFAULT_SCRIPT


def test_devices_whitelist_matches_single_source(client) -> None:
    body = client.get("/devices").json()
    assert [d["name"] for d in body["devices"]] == preset_names()
    for d in body["devices"]:
        assert "rows" in d["fields"] and "cols" in d["fields"]


# ---- GET /experiments -------------------------------------------------------


def test_list_experiments_empty(client) -> None:
    r = client.get("/experiments")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["count"] == 0 and body["total"] == 0
    assert body["runs"] == [] and body["by_status"] == {}


def test_get_experiment_unknown_id_is_404(client) -> None:
    r = client.get("/experiments/no-such-run")
    assert r.status_code == 404
    assert r.json()["error_code"] == "run_not_found"


def test_list_experiments_bad_status_filter(client) -> None:
    r = client.get("/experiments", params={"status": "exploded"})
    assert r.status_code == 400
    assert r.json()["error_code"] == "invalid_request"


# ---- GET /datapacks ---------------------------------------------------------


def test_datapacks_empty(client) -> None:
    body = client.get("/datapacks").json()
    assert body["count"] == 0 and body["packs"] == []
    assert body["data_root_exists"] is True


def test_datapacks_indexes_viz_and_live(client, data_root) -> None:
    # viz 包：契约 v0（viz/CONTRACT.md）
    viz = data_root / "viz" / "v0_sample"
    (viz / "frames" / "ch_resource").mkdir(parents=True)
    (viz / "frames" / "ch_resource" / "00000.bin").write_bytes(b"\x00" * 96)
    (viz / "entities").mkdir(parents=True)
    (viz / "entities" / "00000.f32").write_bytes(b"\x00" * 28)
    (viz / "series.csv").write_text("tick,pop\n0,5\n", encoding="utf-8")
    (viz / "meta.json").write_text(json.dumps({
        "contract_version": "v0",
        "world": {"rows": 8, "cols": 12, "n_cells": 96, "projection": "equirect"},
        "tick": {"start": 0, "end": 20, "stride": 10, "n_frames": 3},
        "channels": [{"key": "resource"}],
    }, ensure_ascii=False), encoding="utf-8")

    # live 包：本服务写的 run.json 台账（无 meta.json ⇒ 回退 run.json）
    run = _fake_run("20261011T000000Z-s2-abcdef", data_root)

    body = client.get("/datapacks").json()
    assert body["count"] == 2
    by_id = {p["id"]: p for p in body["packs"]}
    assert set(by_id) == {"viz/v0_sample", f"live/{run.run_id}"}

    v = by_id["viz/v0_sample"]
    assert v["kind"] == "viz" and v["meta_source"] == "meta.json"
    assert v["contract_version"] == "v0"
    assert v["world"] == {"rows": 8, "cols": 12, "n_cells": 96}
    assert v["tick"] == {"start": 0, "end": 20, "stride": 10, "n_frames": 3}
    assert v["stats"]["n_frame_files"] == 1 and v["stats"]["n_entity_files"] == 1
    assert v["stats"]["has_series_csv"] is True and v["stats"]["n_files"] >= 4

    lv = by_id[f"live/{run.run_id}"]
    assert lv["kind"] == "live" and lv["meta_source"] == "run.json"
    assert lv["run_status"] == api.STATUS_FINISHED

    # kind 过滤 + include_meta=False
    only_viz = client.get("/datapacks", params={"kind": "viz"}).json()
    assert [p["id"] for p in only_viz["packs"]] == ["viz/v0_sample"]
    slim = client.get("/datapacks",
                      params={"include_meta": False, "stats": False}).json()
    assert all(p["meta"] is None and p["stats"] is None for p in slim["packs"])
    assert slim["packs"][0]["meta_source"] is not None  # 来源仍报，只是不内联全文


def test_datapacks_bad_kind(client) -> None:
    r = client.get("/datapacks", params={"kind": "everything"})
    assert r.status_code == 400 and r.json()["error_code"] == "invalid_request"


# ---- POST /experiments：fail-loud 四连 --------------------------------------


def test_post_experiments_rejects_unknown_device(client) -> None:
    r = client.post("/experiments",
                    json={"device": "s9", "seed": 1, "ticks": 10})
    assert r.status_code == 400, r.text
    body = r.json()
    assert body["error_code"] == "unknown_device"
    assert body["detail"]["allowed_devices"] == preset_names()


def test_post_experiments_rejects_raw_rows_cols(client) -> None:
    """纪律 R5.6：不许传裸 rows/cols 绕开装置白名单 ⇒ extra=forbid 直接 422。"""
    r = client.post("/experiments", json={
        "device": preset_names()[0], "seed": 1, "ticks": 10,
        "rows": 480, "cols": 960,
    })
    assert r.status_code == 422, r.text
    body = r.json()
    assert body["error_code"] == "invalid_request"
    fields = {f["field"] for f in body["detail"]["fields"]}
    assert {"rows", "cols"} <= fields


def test_post_experiments_rejects_missing_fields(client) -> None:
    r = client.post("/experiments", json={})
    assert r.status_code == 422
    fields = {f["field"] for f in r.json()["detail"]["fields"]}
    assert {"device", "seed", "ticks"} <= fields


def test_post_experiments_rejects_bad_values(client) -> None:
    dev = preset_names()[0]
    assert client.post("/experiments",
                       json={"device": dev, "seed": 1, "ticks": 0}).status_code == 422
    assert client.post("/experiments",
                       json={"device": dev, "seed": -1, "ticks": 10}).status_code == 422
    assert client.post("/experiments",
                       json={"device": dev, "seed": 1, "ticks": "abc"}).status_code == 422


def test_post_experiments_rejects_missing_script(client) -> None:
    r = client.post("/experiments", json={
        "device": preset_names()[0], "seed": 1, "ticks": 10,
        "script": "experiments/_definitely_not_here_.py",
    })
    assert r.status_code == 400, r.text
    assert r.json()["error_code"] == "script_not_found"


# ---- POST /experiments：真拉起（轻量） --------------------------------------


def test_post_experiments_spawns_and_finishes(client, data_root, tmp_path) -> None:
    script = tmp_path / "fake_run.py"
    script.write_text(_FAKE_SCRIPT, encoding="utf-8")

    r = client.post("/experiments", json={
        "device": preset_names()[0], "seed": 7, "ticks": 5, "arm": 1,
        "name": "smoke", "script": str(script),
        "extra_args": ["--sample", "20"], "live": False,
    })
    assert r.status_code == 201, r.text
    run = r.json()
    assert run["status"] in (api.STATUS_PENDING, api.STATUS_RUNNING)
    assert run["pid"] and run["device"] == preset_names()[0]
    assert run["live"] is False
    assert "--device" in run["args"] and "--out" in run["args"]
    assert run["out_dir"].startswith(str(data_root.resolve()))

    run_id = run["run_id"]
    _wait_until(
        lambda: client.get(f"/experiments/{run_id}").json()["status"]
        in (api.STATUS_FINISHED, api.STATUS_FAILED),
        timeout=120, msg=f"run {run_id} 未在限时内结束")

    detail = client.get(f"/experiments/{run_id}", params={"tail_lines": 40}).json()
    assert detail["status"] == api.STATUS_FINISHED, detail
    assert detail["returncode"] == 0 and detail["ended_reason"] == "exit_ok"
    assert detail["finished_at"] and detail["started_at"]
    assert any("FAKE-RUN-OK" in ln for ln in detail["log_tail"]), detail["log_tail"]

    # 产物目录 = <data_root>/live/<run_id>/，且参数确实注入到了脚本
    out_dir = Path(detail["out_dir"])
    assert out_dir == data_root / "live" / run_id
    marker = json.loads((out_dir / "marker.json").read_text(encoding="utf-8"))
    assert marker["device"] == preset_names()[0]
    assert marker["seed"] == 7 and marker["ticks"] == 5 and marker["arm"] == 1
    assert marker["sample"] == 20            # extra_args 透传
    assert Path(marker["out"]).resolve() == out_dir.resolve()
    assert (out_dir / api._RUN_LEDGER).is_file()
    assert (out_dir / api._LOG_NAME).is_file()

    # 列表里能看到它，且状态过滤生效（by_status 统计全量 run，不受过滤影响）
    listed = client.get("/experiments").json()
    assert listed["count"] == 1 and listed["by_status"] == {api.STATUS_FINISHED: 1}
    filtered = client.get("/experiments", params={"status": "running"}).json()
    assert filtered["count"] == 0 and filtered["total"] == 0
    assert filtered["by_status"] == {api.STATUS_FINISHED: 1}


def test_post_experiments_live_mode_uses_supervisor(client, tmp_path) -> None:
    script = tmp_path / "fake_live.py"
    script.write_text(_FAKE_LIVE_SCRIPT, encoding="utf-8")
    r = client.post("/experiments", json={
        "device": preset_names()[0], "seed": 3, "ticks": 5,
        "name": "smoke-live", "script": str(script), "live": True,
    })
    assert r.status_code == 201, r.text
    run = r.json()
    assert run["command"][1:4] == ["-m", "observatory.api", "supervise"]
    assert "--" in run["command"]

    run_id = run["run_id"]
    out_dir = Path(run["out_dir"])
    _wait_ready(client, run_id, out_dir, timeout=120)

    detail = client.get(f"/experiments/{run_id}").json()
    assert detail["status"] == api.STATUS_RUNNING, detail
    assert any("[supervise]" in ln for ln in detail["log_tail"]), detail["log_tail"]

    # 收尾：终止（CTRL_BREAK_EVENT / 退回 terminate），状态落 failed+cancelled
    cancel = client.post(f"/experiments/{run_id}/cancel", params={"grace_s": 15})
    assert cancel.status_code == 200, cancel.text
    assert cancel.json()["status"] == api.STATUS_FAILED
    assert cancel.json()["ended_reason"] == "cancelled"
    assert cancel.json()["returncode"] is not None
    # 已结束 ⇒ 再取消要 409
    assert client.post(f"/experiments/{run_id}/cancel").status_code == 409


# ---- POST /snapshots --------------------------------------------------------


def test_snapshot_unknown_run_is_404(client) -> None:
    r = client.post("/snapshots", json={"run_id": "nope"})
    assert r.status_code == 404 and r.json()["error_code"] == "run_not_found"


def test_snapshot_bad_body(client) -> None:
    assert client.post("/snapshots", json={}).status_code == 422
    assert client.post("/snapshots",
                       json={"run_id": "x", "tick": -1}).status_code == 422
    assert client.post("/snapshots",
                       json={"run_id": "x", "unknown": 1}).status_code == 422


def test_snapshot_inproc_with_real_engine(client, data_root) -> None:
    """进程内挂一台真引擎（8×12 微型）⇒ 直接调 SphereEngine.save_snapshot。"""
    run = _fake_run("inproc-smoke", data_root, live=True, status=api.STATUS_RUNNING)
    eng = _tiny_engine()
    api.attach_engine(run.run_id, eng)
    try:
        r = client.post("/snapshots", json={"run_id": run.run_id})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["mode"] == "inproc" and body["run_id"] == run.run_id
        assert body["tick"] == int(eng.tick)
        assert body["gene_count"] == eng.config.genome.gene_count
        assert body["config_fingerprint"] == eng.config.fingerprint()
        assert body["snapshot_version"] == 3
        assert body["size_bytes"] == Path(body["path"]).stat().st_size > 0
        assert body["rel_path"].startswith("live/inproc-smoke/snapshots/")
        # 只回元数据：响应是 JSON，且体量远小于 npz 本体（没把快照塞回来）
        assert r.headers["content-type"].startswith("application/json")
        assert not r.content.startswith(b"PK")          # 不是 zip/npz 字节流
        assert len(r.content) < body["size_bytes"]
        assert set(body) == {
            "path", "rel_path", "size_bytes", "created_at", "tick",
            "gene_count", "config_fingerprint", "snapshot_version",
            "run_id", "mode", "request_id",
        }

        # 指定 tick 标注 + 台账里记下这次快照
        r2 = client.post("/snapshots", json={"run_id": run.run_id, "tick": 999})
        assert r2.status_code == 200 and r2.json()["tick"] == 999
        detail = client.get(f"/experiments/{run.run_id}").json()
        assert len(detail["snapshots"]) == 2
        assert {s["tick"] for s in detail["snapshots"]} == {int(eng.tick), 999}
    finally:
        api.detach_engine(run.run_id)


def test_snapshot_unsupported_when_not_live(client, data_root) -> None:
    run = _fake_run("nolive-smoke", data_root, live=False, status=api.STATUS_RUNNING)
    r = client.post("/snapshots", json={"run_id": run.run_id})
    assert r.status_code == 409 and r.json()["error_code"] == "snapshot_unsupported"


def test_snapshot_unsupported_when_finished(client, data_root) -> None:
    run = _fake_run("done-smoke", data_root, live=True, status=api.STATUS_FINISHED)
    r = client.post("/snapshots", json={"run_id": run.run_id})
    assert r.status_code == 409 and r.json()["error_code"] == "snapshot_unsupported"


def test_snapshot_live_end_to_end(client, data_root, tmp_path) -> None:
    """全链路：POST /experiments(live) → 子进程建引擎 → POST /snapshots → ack 元数据。"""
    script = tmp_path / "fake_live2.py"
    script.write_text(_FAKE_LIVE_SCRIPT, encoding="utf-8")
    run = client.post("/experiments", json={
        "device": preset_names()[0], "seed": 11, "ticks": 5,
        "name": "smoke-snap", "script": str(script), "live": True,
    }).json()
    run_id = run["run_id"]
    out_dir = Path(run["out_dir"])
    try:
        _wait_ready(client, run_id, out_dir, timeout=120)

        r = client.post("/snapshots",
                        json={"run_id": run_id, "wait_s": 90})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["mode"] == "live" and body["run_id"] == run_id
        assert body["snapshot_version"] == 3 and body["gene_count"] == 24
        assert isinstance(body["config_fingerprint"], str) and body["config_fingerprint"]
        assert isinstance(body["tick"], int) and body["tick"] >= 0
        p = Path(body["path"])
        assert p.is_file() and p.parent == out_dir / "snapshots"
        assert body["size_bytes"] == p.stat().st_size > 0
        assert body["rel_path"].startswith(f"live/{run_id}/snapshots/")
        assert (out_dir / api._SNAPSHOT_ACK).is_file()

        # 快照文件确实是 npz（只读标量元数据键；**不开 allow_pickle**，
        # 避免反序列化 object 数组——只碰我们自己刚写的文件也一样走安全口径）
        import numpy as np
        with np.load(p) as z:
            assert int(z["snapshot_version"]) == 3
            assert int(z["gene_count"]) == 24
    finally:
        client.post(f"/experiments/{run_id}/cancel", params={"grace_s": 15})


def test_snapshot_timeout_is_loud(client, data_root) -> None:
    """live run 但没有 supervisor 应答 ⇒ 409 snapshot_timeout（不静默挂死）。"""
    run = _fake_run("timeout-smoke", data_root, live=True, status=api.STATUS_RUNNING)
    r = client.post("/snapshots", json={"run_id": run.run_id, "wait_s": 1.0})
    assert r.status_code == 409, r.text
    assert r.json()["error_code"] == "snapshot_timeout"


# ---- GET /snapshots + POST /snapshots/export（Task #13：前端自主加载快照）----


def _make_engine_snapshot(path: Path, ticks: int = 3) -> int:
    """用微型真引擎在 path 产一份快照，返回快照时刻的 tick。"""
    eng = _tiny_engine(ticks=ticks)
    path.parent.mkdir(parents=True, exist_ok=True)
    eng.save_snapshot(str(path))
    return int(eng.tick)


def _make_fake_experiment_npz(path: Path) -> None:
    """非快照 npz（C0 类实验数据：无 snapshot_version 键 ⇒ 应被扫描排除）。"""
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, capacity=np.zeros(96), stock_t0=np.zeros(96))


def test_list_snapshots_scans_and_annotates(client, data_root) -> None:
    """live 快照 + 散落快照都进列表，非快照 npz 排除并计入 skipped。"""
    run_id = "20261011T000000Z-list-abc123"
    tick = _make_engine_snapshot(
        data_root / "live" / run_id / "snapshots" / "snapshot_t5.npz")
    stray_tick = _make_engine_snapshot(data_root / "imports" / "stray.npz")
    _make_fake_experiment_npz(data_root / "c0_energy" / "c0_x.npz")
    _make_fake_experiment_npz(data_root / "viz" / "leak.npz")  # viz/ 不扫

    body = client.get("/snapshots").json()
    assert body["data_root"] == str(data_root.resolve())
    by_rel = {s["snapshot"]: s for s in body["snapshots"]}
    assert set(by_rel) == {
        f"live/{run_id}/snapshots/snapshot_t5.npz", "imports/stray.npz"}
    live_item = by_rel[f"live/{run_id}/snapshots/snapshot_t5.npz"]
    assert live_item["run_id"] == run_id
    assert live_item["tick"] == tick
    assert live_item["size_bytes"] > 0 and live_item["modified_at"]
    stray = by_rel["imports/stray.npz"]
    assert stray["run_id"] is None and stray["tick"] == stray_tick
    assert body["skipped"] == 1
    assert body["count"] == len(body["snapshots"]) == 2


def test_list_snapshots_falls_back_to_ledger_tick(client, data_root) -> None:
    """npz 内无 tick 标量（模拟旧版快照）时，退 run 台账的 tick 记录。"""
    import numpy as np

    run = _fake_run("ledger-smoke", data_root)
    snap_dir = data_root / "live" / run.run_id / "snapshots"
    src = snap_dir / "snapshot_t7.npz"
    _make_engine_snapshot(src)
    with np.load(src, allow_pickle=True) as z:   # 重写一份剔除 tick 键的 npz
        np.savez(src, **{k: z[k] for k in z.files if k != "tick"})
    run.snapshots.append({
        "path": str(snap_dir / "snapshot_t7.npz"),
        "rel_path": f"live/{run.run_id}/snapshots/snapshot_t7.npz",
        "tick": 424242,
    })
    api.REGISTRY.persist(run)

    body = client.get("/snapshots").json()
    item = next(s for s in body["snapshots"] if s["snapshot"].endswith("snapshot_t7.npz"))
    assert item["tick"] == 424242 and item["run_id"] == run.run_id


def test_export_snapshot_end_to_end(client, data_root) -> None:
    """全链路：微型引擎快照 → POST export → 产物结构 + check_datapack rc=0。"""
    import sys as _sys

    rel = "imports/e2e.npz"
    src_tick = _make_engine_snapshot(data_root / Path(rel))
    r = client.post("/snapshots/export", json={
        "snapshot": rel, "name": "E2E TestPack", "ticks": 0, "stride": 20,
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["pack_id"] == "viz/snap_E2E-TestPack"
    assert body["name"] == "snap_E2E-TestPack"
    assert body["n_frames"] == 1                      # ticks=0 ⇒ 仅当前帧
    assert body["rows"] == 8 and body["cols"] == 12
    assert body["size_bytes"] > 0

    pack_dir = data_root / "viz" / "snap_E2E-TestPack"
    assert Path(body["path"]).resolve() == pack_dir.resolve()
    meta = json.loads((pack_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["contract_version"] == "v0"
    assert meta["tick"]["start"] == src_tick and meta["tick"]["n_frames"] == 1
    assert [c["key"] for c in meta["channels"]] == ["resource", "energy"]
    assert (pack_dir / "frames" / "ch_resource" / "00000.bin").stat().st_size == 96
    assert (pack_dir / "entities" / "00000.f32").stat().st_size % 28 == 0
    # 导出日志落在 viz/.export_logs/（“.”开头 ⇒ 不被 GET /datapacks 索引）
    assert (data_root / "viz" / ".export_logs" / "snap_E2E-TestPack.log").is_file()
    assert all(not p["id"].startswith("viz/.export_logs")
               for p in client.get("/datapacks").json()["packs"])

    # check_datapack 独立复检 rc=0
    _sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "viz" / "exporter"))
    import check_datapack as cd

    assert cd.main([str(pack_dir)]) == 0


def test_export_snapshot_rejects_bad_paths(client, data_root) -> None:
    """防穿越：.. 逃逸 / data_root 外绝对路径 / 非 .npz / 不存在 ⇒ 400 invalid_snapshot。"""
    outside = str(Path(data_root).parent / "elsewhere.npz")
    cases = ["../outside.npz", "a/b/../../../secret.npz", outside,
             "x/y.npz.txt", "nope/missing.npz"]
    for snap in cases:
        r = client.post("/snapshots/export", json={"snapshot": snap})
        assert r.status_code == 400, (snap, r.text)
        assert r.json()["error_code"] == "invalid_snapshot", snap


def test_export_snapshot_bad_request_bodies(client, data_root) -> None:
    """422（pydantic）与 400（通道白名单）fail-loud。"""
    assert client.post("/snapshots/export", json={}).status_code == 422
    assert client.post("/snapshots/export",
                       json={"snapshot": "a.npz", "ticks": -1}).status_code == 422
    assert client.post("/snapshots/export",
                       json={"snapshot": "a.npz", "stride": 0}).status_code == 422
    assert client.post("/snapshots/export",
                       json={"snapshot": "a.npz", "unknown": 1}).status_code == 422
    _make_engine_snapshot(data_root / "imports" / "c.npz")
    r = client.post("/snapshots/export",
                    json={"snapshot": "imports/c.npz", "channels": ["bogus"]})
    assert r.status_code == 400 and r.json()["error_code"] == "invalid_request"


def test_export_snapshot_invalid_content_fails_loud(client, data_root) -> None:
    """伪快照（无 snapshot_version 的实验 npz）通过路径校验但导出器失败
    ⇒ 500 snapshot_export_failed（统一信封，不 500 裸奔）。"""
    _make_fake_experiment_npz(data_root / "c0_energy" / "c0_bad.npz")
    r = client.post("/snapshots/export", json={"snapshot": "c0_energy/c0_bad.npz"})
    assert r.status_code == 500, r.text
    body = r.json()
    assert body["error_code"] == "snapshot_export_failed"
    assert body["detail"]["rc"] not in (None, 0)   # 导出器非零退出（1=未捕获异常 / 2=契约拒绝）
    assert body["message"]                          # 信封里带人类可读原因
    assert any((data_root / "viz" / ".export_logs").iterdir())
