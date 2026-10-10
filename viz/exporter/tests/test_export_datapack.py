"""导出器冒烟测试：合成快照 → 数据包 → 校验器（契约 v0）。

跑法（仓根或本 worktree 根）：python -m pytest viz/exporter/tests -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT))

from simulation.config import SimConfig  # noqa: E402
from simulation.sphere_engine import SphereEngine  # noqa: E402

sys.path.insert(0, str(_ROOT / "viz" / "exporter"))
import check_datapack as cd  # noqa: E402
import export_datapack as ed  # noqa: E402


def _make_datapack(tmp_path: Path, name: str = "dp") -> Path:
    eng = SphereEngine(SimConfig())  # 默认 60×120；Python 路径
    snap = tmp_path / "s.npz"
    eng.save_snapshot(str(snap))
    out = tmp_path / name
    rc = ed.main(
        ["--snapshot", str(snap), "--out", str(out), "--ticks", "20", "--stride", "10"]
    )
    assert rc == 0
    return out


def test_meta_contract(tmp_path):
    dp = _make_datapack(tmp_path)
    meta = json.loads((dp / "meta.json").read_text(encoding="utf-8"))
    assert meta["contract_version"] == "v0"
    w, t = meta["world"], meta["tick"]
    assert w["rows"] * w["cols"] == w["n_cells"]
    assert t["n_frames"] == 3  # 20 // 10 + 1（含 t0）
    assert t["end"] == t["start"] + 20
    assert meta["source"]["engine"] == "python"
    assert meta["entities"]["subdiv"] == SimConfig().subpos.subdiv  # 契约 v0 修订1
    for ch in meta["channels"]:
        s = ch["scale"]
        assert s["p_hi"] > s["p_lo"]
        assert s["min"] <= s["p_lo"] and s["max"] >= s["p_hi"] - 1e-9


def test_frame_files(tmp_path):
    dp = _make_datapack(tmp_path)
    meta = json.loads((dp / "meta.json").read_text(encoding="utf-8"))
    n_cells, n_frames = meta["world"]["n_cells"], meta["tick"]["n_frames"]
    for ch in meta["channels"]:
        d = dp / "frames" / f"ch_{ch['key']}"
        bins = sorted(d.glob("*.bin"))
        assert len(bins) == n_frames
        for i, b in enumerate(bins):
            assert b.name == f"{i:05d}.bin"
            assert b.stat().st_size == n_cells
            arr = np.fromfile(b, dtype=np.uint8)
            assert arr.size == n_cells


def test_entity_files(tmp_path):
    dp = _make_datapack(tmp_path)
    fs = sorted((dp / "entities").glob("*.f32"))
    assert len(fs) == 3
    for p in fs:
        assert p.stat().st_size % 28 == 0
        if p.stat().st_size:
            a = np.fromfile(p, dtype="<f4")
            assert np.isfinite(a).all()
            flat = a[0::7]
            assert flat.min() >= 0 and flat.max() < 60 * 120


def test_checker_pass(tmp_path):
    dp = _make_datapack(tmp_path)
    assert cd.main([str(dp)]) == 0


def test_checker_catches_corrupt(tmp_path):
    dp = _make_datapack(tmp_path)
    bad = dp / "frames" / "ch_resource" / "00001.bin"
    bad.write_bytes(bad.read_bytes()[:-1])  # 截断一字节
    assert cd.main([str(dp)]) == 2


def test_determinism(tmp_path):
    a = _make_datapack(tmp_path, "a")
    b = _make_datapack(tmp_path, "b")
    fa = sorted((a / "frames" / "ch_energy").glob("*.bin"))
    fb = sorted((b / "frames" / "ch_energy").glob("*.bin"))
    assert len(fa) == len(fb)
    for pa, pb in zip(fa, fb):
        assert pa.read_bytes() == pb.read_bytes()


def test_empty_entities_ok(tmp_path):
    """P=0 空实体帧：实体文件 0 字节、norm 空、校验器照过（契约 §4）。"""
    eng = SphereEngine(SimConfig())
    eng._id = eng._id[:0]
    snap = tmp_path / "s.npz"
    eng.save_snapshot(str(snap))
    out = tmp_path / "dp"
    rc = ed.main(["--snapshot", str(snap), "--out", str(out), "--ticks", "10", "--stride", "10"])
    assert rc == 0
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert meta["entities"]["norm"] == {}
    ents = sorted((out / "entities").glob("*.f32"))
    assert len(ents) == meta["tick"]["n_frames"]
    assert all(p.stat().st_size == 0 for p in ents)
    assert cd.main([str(out)]) == 0


def test_flat_out_of_bounds_rejected(tmp_path, capsys):
    """flat 越界 ⇒ 导出期急停 rc=2（旧行为：bincount 静默扩帧，坏包直到校验才暴露）。"""
    eng = SphereEngine(SimConfig())
    n = eng.config.world.rows * eng.config.world.cols
    eng._flat[0] = n + 5
    snap = tmp_path / "s.npz"
    eng.save_snapshot(str(snap))
    out = tmp_path / "dp"
    rc = ed.main(["--snapshot", str(snap), "--out", str(out), "--ticks", "0"])
    assert rc == 2
    assert "flat 越界" in capsys.readouterr().err
    assert not (out / "meta.json").exists()
    rc2 = ed.main(["--snapshot", str(snap), "--out", str(out), "--ticks", "0", "--channels", "resource"])
    assert rc2 == 2  # 实体路径守卫（无 energy 通道时）


def test_sub_coord_out_of_bounds_rejected(tmp_path, capsys):
    """sub_r/sub_c 越界（契约 v0 修订1 值域）⇒ 导出期急停 rc=2。"""
    eng = SphereEngine(SimConfig())
    eng._sub_r[0] = eng.config.world.rows * eng.config.subpos.subdiv + 1
    snap = tmp_path / "s.npz"
    eng.save_snapshot(str(snap))
    out = tmp_path / "dp"
    rc = ed.main(["--snapshot", str(snap), "--out", str(out), "--ticks", "0", "--channels", "resource"])
    assert rc == 2
    assert "sub_r 越界" in capsys.readouterr().err


def test_unavailable_channel_skipped(tmp_path, monkeypatch, capsys):
    """通道源不可用 ⇒ 跳过 + 警示 + meta 只记可用通道（契约 §6 fruit 为可选）。"""
    eng = SphereEngine(SimConfig())
    snap = tmp_path / "s.npz"
    eng.save_snapshot(str(snap))
    del eng._fruit_grid
    monkeypatch.setattr(
        ed.SphereEngine, "load_snapshot", classmethod(lambda cls, path, config=None: eng)
    )
    out = tmp_path / "dp"
    rc = ed.main(
        ["--snapshot", str(snap), "--out", str(out), "--ticks", "10", "--stride", "10",
         "--channels", "resource,energy,fruit"]
    )
    assert rc == 0
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert [c["key"] for c in meta["channels"]] == ["resource", "energy"]
    assert "fruit" in capsys.readouterr().err
    assert cd.main([str(out)]) == 0


def test_all_channels_unavailable_rejected(tmp_path, monkeypatch, capsys):
    """所有请求通道均不可用 ⇒ 拒绝导出 rc=2（不出空包）。"""
    eng = SphereEngine(SimConfig())
    del eng._fruit_grid
    monkeypatch.setattr(
        ed.SphereEngine, "load_snapshot", classmethod(lambda cls, path, config=None: eng)
    )
    out = tmp_path / "dp"
    rc = ed.main(["--snapshot", str(tmp_path / "x.npz"), "--out", str(out), "--ticks", "0", "--channels", "fruit"])
    assert rc == 2
    assert not (out / "meta.json").exists()


def test_grid_frames_not_aliased(tmp_path):
    """帧缓冲须为拷贝：改帧不得回写引擎活网格（否则所有帧会变成末帧）。"""
    eng = SphereEngine(SimConfig())
    g = ed.capture_grids(eng, ["resource"])["resource"]
    g[:] = -999.0
    assert float(eng.resources._grid.min()) != -999.0


def test_checker_subdiv_and_subcoord_range(tmp_path):
    """校验器：subdiv 非法 ⇒ 2；缺失 ⇒ 仅 WARN；sub_r/sub_c 越界 ⇒ 2（契约 v0 修订1）。"""
    dp = _make_datapack(tmp_path, "dp_sd")
    meta_p = dp / "meta.json"
    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    meta["entities"]["subdiv"] = 0
    meta_p.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    assert cd.main([str(dp)]) == 2
    del meta["entities"]["subdiv"]
    meta_p.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    assert cd.main([str(dp)]) == 0  # 旧包缺省可回退

    dp2 = _make_datapack(tmp_path, "dp_sc")
    p = sorted((dp2 / "entities").glob("*.f32"))[0]
    a = np.fromfile(p, dtype="<f4").reshape(-1, 7)
    assert a.shape[0] > 0
    a[0, 1] = 10**6  # sub_r 越界
    a.astype("<f4").tofile(p)
    assert cd.main([str(dp2)]) == 2


def test_tail_ticks_note(tmp_path, capsys):
    """ticks 非 stride 整数倍：帧轴止于最后采样点，且 stderr 明示末段无帧。"""
    eng = SphereEngine(SimConfig())
    snap = tmp_path / "s.npz"
    eng.save_snapshot(str(snap))
    out = tmp_path / "dp"
    rc = ed.main(["--snapshot", str(snap), "--out", str(out), "--ticks", "25", "--stride", "10"])
    assert rc == 0
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert meta["tick"]["n_frames"] == 3
    assert meta["tick"]["end"] == meta["tick"]["start"] + 20
    assert "无采样帧" in capsys.readouterr().err
    assert cd.main([str(out)]) == 0
