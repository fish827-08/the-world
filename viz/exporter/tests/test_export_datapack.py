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
