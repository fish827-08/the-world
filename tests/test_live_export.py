"""实时增量导出器 (live_export) + --series auto 的测试。

覆盖：
- LiveExporter 增量写帧 + meta.json 帧数递增
- v0 格式通过 check_datapack 校验
- 后台线程模式（start_background / stop）
- --series auto 自动扫描并复制时序 CSV
- --series PATH 原有行为不变

跑法：python -m pytest tests/test_live_export.py -q
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "viz" / "exporter"))

from simulation.config import (  # noqa: E402
    PopulationConfig,
    SimConfig,
    SimulationConfig,
    WorldConfig,
)
from simulation.sphere_engine import SphereEngine  # noqa: E402

import check_datapack as cd  # noqa: E402
import export_datapack as ed  # noqa: E402
from live_export import LiveExporter  # noqa: E402


def _tiny_engine(ticks: int = 50, rows: int = 8, cols: int = 12) -> SphereEngine:
    """构造微型引擎（8×12，秒级）。"""
    cfg = SimConfig(
        seed=42,
        world=WorldConfig(rows=rows, cols=cols),
        simulation=SimulationConfig(ticks=ticks),
        population=PopulationConfig(initial_count=10),
    )
    return SphereEngine(cfg)


# ---- LiveExporter 增量写帧 ----


class TestLiveExporterIncremental:
    """验证 LiveExporter 逐帧增量写入 + meta.json 帧数递增。"""

    def test_basic_incremental_frames(self, tmp_path):
        """跑引擎，每 stride tick 采集一帧，验证帧文件数递增。"""
        eng = _tiny_engine(ticks=50)
        out = tmp_path / "live_dp"
        exporter = LiveExporter(eng, out, stride=10)

        # 首帧（tick=0）
        assert exporter.maybe_capture() is True
        assert exporter.frame_count == 1
        assert (out / "meta.json").is_file()
        meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
        assert meta["tick"]["n_frames"] == 1

        # 跑 10 tick → 第 2 帧
        for _ in range(10):
            eng.step()
        assert exporter.maybe_capture() is True
        assert exporter.frame_count == 2
        meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
        assert meta["tick"]["n_frames"] == 2

        # 跑 5 tick → 未到 stride，不采集
        for _ in range(5):
            eng.step()
        assert exporter.maybe_capture() is False
        assert exporter.frame_count == 2

        # 再跑 5 tick → 到 stride，采集
        for _ in range(5):
            eng.step()
        assert exporter.maybe_capture() is True
        assert exporter.frame_count == 3

        # 验证帧文件存在
        for ch_dir in (out / "frames").iterdir():
            bins = sorted(ch_dir.glob("*.bin"))
            assert len(bins) == 3
        ents = sorted((out / "entities").glob("*.f32"))
        assert len(ents) == 3

    def test_meta_contract_v0(self, tmp_path):
        """验证 meta.json 符合 v0 契约字段。"""
        eng = _tiny_engine(ticks=30)
        out = tmp_path / "live_meta"
        exporter = LiveExporter(eng, out, stride=10)

        for _ in range(30):
            exporter.maybe_capture()
            eng.step()
        exporter.finalize()

        meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
        assert meta["contract_version"] == "v0"
        assert meta["world"]["rows"] == 8
        assert meta["world"]["cols"] == 12
        assert meta["world"]["n_cells"] == 96
        assert meta["world"]["projection"] == "equirect"
        assert meta["tick"]["stride"] == 10
        assert meta["tick"]["n_frames"] >= 3
        assert meta["entities"]["subdiv"] == 4
        assert meta["entities"]["columns"] == [
            "flat", "sub_r", "sub_c", "energy", "age", "generation", "mode"]
        assert meta["generator"]["tool"] == "viz/exporter/live_export"
        # 通道有效
        for ch in meta["channels"]:
            assert ch["scale"]["p_hi"] > ch["scale"]["p_lo"]

    def test_check_datapack_passes(self, tmp_path):
        """v0 格式通过 check_datapack 校验器。"""
        eng = _tiny_engine(ticks=40)
        out = tmp_path / "live_check"
        exporter = LiveExporter(eng, out, stride=10)

        for _ in range(40):
            exporter.maybe_capture()
            eng.step()
        exporter.finalize()

        assert cd.main([str(out)]) == 0

    def test_atomic_meta_write(self, tmp_path):
        """meta.json 写入是原子的（无 .tmp 残留）。"""
        eng = _tiny_engine(ticks=20)
        out = tmp_path / "live_atomic"
        exporter = LiveExporter(eng, out, stride=5)

        for _ in range(20):
            exporter.maybe_capture()
            eng.step()
        exporter.finalize()

        # 不应有 .tmp 文件残留
        tmps = list(out.rglob("*.tmp"))
        assert tmps == []
        # meta.json 可正常解析
        meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
        assert meta["tick"]["n_frames"] == exporter.frame_count

    def test_frame_binary_sizes(self, tmp_path):
        """帧文件大小 = rows*cols（uint8）；实体文件大小 = P*7*4（float32）。"""
        eng = _tiny_engine(ticks=10)
        out = tmp_path / "live_sizes"
        exporter = LiveExporter(eng, out, stride=5)

        exporter.maybe_capture()
        for _ in range(5):
            eng.step()
        exporter.maybe_capture()
        exporter.finalize()

        n_cells = 8 * 12
        for ch_dir in (out / "frames").iterdir():
            for b in ch_dir.glob("*.bin"):
                assert b.stat().st_size == n_cells
        for f in (out / "entities").glob("*.f32"):
            assert f.stat().st_size % 28 == 0  # 7 columns × 4 bytes


class TestLiveExporterBackground:
    """验证后台线程模式。"""

    def test_background_captures_frames(self, tmp_path):
        """start_background → 引擎跑 → stop，验证有帧产出。"""
        eng = _tiny_engine(ticks=100)
        out = tmp_path / "live_bg"
        exporter = LiveExporter(eng, out, stride=5)
        exporter.start_background(interval_s=0.1)

        # 跑引擎
        for _ in range(100):
            eng.step()
            time.sleep(0.01)

        exporter.stop()
        assert exporter.frame_count >= 2
        assert (out / "meta.json").is_file()
        meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
        assert meta["tick"]["n_frames"] == exporter.frame_count

    def test_background_stop_idempotent(self, tmp_path):
        """多次 stop 不报错。"""
        eng = _tiny_engine(ticks=10)
        out = tmp_path / "live_bg2"
        exporter = LiveExporter(eng, out, stride=5)
        exporter.start_background(interval_s=0.1)
        time.sleep(0.3)
        exporter.stop()
        exporter.stop()  # 第二次不报错


# ---- --series auto ----


class TestSeriesAuto:
    """验证 export_datapack.py 的 --series auto 功能。"""

    def test_series_auto_finds_generations_csv(self, tmp_path):
        """快照同目录有 generations.csv → 自动复制为 series.csv。"""
        eng = _tiny_engine(ticks=5)
        snap = tmp_path / "snapshot.npz"
        eng.save_snapshot(str(snap))

        # 写一个 generations.csv 到快照同目录
        csv_content = "tick,pop,mean_energy\n0,10,5.0\n10,12,5.5\n"
        (tmp_path / "generations.csv").write_text(csv_content, encoding="utf-8")

        out = tmp_path / "dp_series"
        rc = ed.main([
            "--snapshot", str(snap), "--out", str(out),
            "--ticks", "10", "--stride", "10", "--series", "auto",
        ])
        assert rc == 0
        series = out / "series.csv"
        assert series.is_file()
        # 表头原样（不重排不改列）
        assert series.read_text(encoding="utf-8") == csv_content

    def test_series_auto_finds_arm_csv(self, tmp_path):
        """快照同目录有 arm1.csv → 按优先级匹配。"""
        eng = _tiny_engine(ticks=5)
        snap = tmp_path / "snapshot.npz"
        eng.save_snapshot(str(snap))

        csv_content = "tick,arm,pop\n0,1,5\n"
        (tmp_path / "arm1.csv").write_text(csv_content, encoding="utf-8")

        out = tmp_path / "dp_arm"
        rc = ed.main([
            "--snapshot", str(snap), "--out", str(out),
            "--ticks", "5", "--stride", "5", "--series", "auto",
        ])
        assert rc == 0
        assert (out / "series.csv").read_text(encoding="utf-8") == csv_content

    def test_series_auto_prefers_generations_over_arm(self, tmp_path):
        """generations.csv 优先级高于 arm*.csv。"""
        eng = _tiny_engine(ticks=5)
        snap = tmp_path / "snapshot.npz"
        eng.save_snapshot(str(snap))

        (tmp_path / "generations.csv").write_text("tick,gen\n0,1\n", encoding="utf-8")
        (tmp_path / "arm2.csv").write_text("tick,arm\n0,2\n", encoding="utf-8")

        out = tmp_path / "dp_pref"
        rc = ed.main([
            "--snapshot", str(snap), "--out", str(out),
            "--ticks", "5", "--stride", "5", "--series", "auto",
        ])
        assert rc == 0
        content = (out / "series.csv").read_text(encoding="utf-8")
        assert "gen" in content  # generations.csv 被选中

    def test_series_auto_no_csv_graceful(self, tmp_path):
        """无可用 CSV → 不报错，不写 series.csv。"""
        eng = _tiny_engine(ticks=5)
        snap = tmp_path / "snapshot.npz"
        eng.save_snapshot(str(snap))

        out = tmp_path / "dp_nocsv"
        rc = ed.main([
            "--snapshot", str(snap), "--out", str(out),
            "--ticks", "5", "--stride", "5", "--series", "auto",
        ])
        assert rc == 0
        assert not (out / "series.csv").exists()

    def test_series_explicit_path_still_works(self, tmp_path):
        """--series PATH（原有行为）不受影响。"""
        eng = _tiny_engine(ticks=5)
        snap = tmp_path / "snapshot.npz"
        eng.save_snapshot(str(snap))

        csv_src = tmp_path / "my_series.csv"
        csv_src.write_text("tick,val\n0,42\n", encoding="utf-8")

        out = tmp_path / "dp_explicit"
        rc = ed.main([
            "--snapshot", str(snap), "--out", str(out),
            "--ticks", "5", "--stride", "5", "--series", str(csv_src),
        ])
        assert rc == 0
        assert (out / "series.csv").read_text(encoding="utf-8") == "tick,val\n0,42\n"

    def test_series_auto_in_out_dir(self, tmp_path):
        """输出目录中已有 CSV → 也能被 auto 发现。"""
        eng = _tiny_engine(ticks=5)
        snap = tmp_path / "snapshot.npz"
        eng.save_snapshot(str(snap))

        out = tmp_path / "dp_out_csv"
        out.mkdir(parents=True)
        # 在输出目录放一个 run_segment_0.csv
        (out / "run_segment_0.csv").write_text("tick,seg\n0,0\n", encoding="utf-8")

        rc = ed.main([
            "--snapshot", str(snap), "--out", str(out),
            "--ticks", "5", "--stride", "5", "--series", "auto",
        ])
        assert rc == 0
        assert (out / "series.csv").read_text(encoding="utf-8") == "tick,seg\n0,0\n"
