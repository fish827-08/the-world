"""B4：demo 冒烟测试（可独立跑 + 同 seed 逐位一致 + 异 seed 有差）。

纪律（项目对拍口径）：
- subprocess 测试**双钉 UTF-8**（env PYTHONIOENCODING=utf-8 +
  run(encoding="utf-8")）——父侧不指定 encoding 时中文 Windows 会按 GBK
  解码子进程 UTF-8 输出（存量 52 红的成因，见 内部工程指南 测试状态节）；
- 同 seed 两次运行 ⇒ metrics JSON 字节级一致；
- 不同 seed ⇒ 输出不同（证明 seed 真的进入模拟，不是常量）；
- demo 不 import 旧 sphere_engine（独立可跑）。
"""
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_demo(out_path: Path, ticks: int = 100, seed: int = 42) -> str:
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"   # 双钉之一：子进程侧
    proc = subprocess.run(
        [sys.executable, "-m", "biosphere.demo",
         "--ticks", str(ticks), "--seed", str(seed),
         "--out", str(out_path)],
        cwd=str(REPO_ROOT), env=env,
        capture_output=True, encoding="utf-8",  # 双钉之二：父侧解码
        timeout=120,
    )
    assert proc.returncode == 0, f"demo 失败:\n{proc.stderr}"
    return proc.stdout


class TestDemoSmoke:
    def test_same_seed_byte_identical(self, tmp_path):
        out1 = tmp_path / "m1.json"
        out2 = tmp_path / "m2.json"
        _run_demo(out1, ticks=100, seed=42)
        _run_demo(out2, ticks=100, seed=42)
        assert out1.read_bytes() == out2.read_bytes(), "同 seed 两次不一致"

    def test_different_seed_differs(self, tmp_path):
        out1 = tmp_path / "s42.json"
        out2 = tmp_path / "s43.json"
        _run_demo(out1, ticks=100, seed=42)
        _run_demo(out2, ticks=100, seed=43)
        assert out1.read_bytes() != out2.read_bytes(), "异 seed 输出相同（seed 没进模拟？）"

    def test_summary_and_metrics_shape(self, tmp_path):
        out = tmp_path / "m.json"
        stdout = _run_demo(out, ticks=100, seed=42)
        assert "final_alive=" in stdout
        assert "max_generation=" in stdout
        m = json.loads(out.read_text(encoding="utf-8"))
        assert m["schema"] == "biosphere-metrics-v0.1"
        assert m["ticks"] == 100 and m["seed"] == 42
        pop = m["population"]
        # 记账恒等式：初代 + 生 - 死 = 终存（48 为 demo 默认 pop）
        assert pop["final_alive"] == 48 + pop["total_born"] - pop["total_died"]
        assert len(pop["alive_series"]) == 100
        # 谱系节点数 = 曾存活个体数
        assert m["lineage"]["n_individuals"] == 48 + pop["total_born"]

    def test_no_legacy_engine_dependency(self, tmp_path):
        """demo 进程不得 import 旧 sphere_engine。"""
        out = tmp_path / "m.json"
        env = dict(os.environ)
        env["PYTHONIOENCODING"] = "utf-8"
        proc = subprocess.run(
            [sys.executable, "-c",
             "import biosphere.demo, sys; "
             "assert 'simulation.sphere_engine' not in sys.modules, "
             "'demo  import 了旧引擎'; print('clean')"],
            cwd=str(REPO_ROOT), env=env,
            capture_output=True, encoding="utf-8", timeout=120)
        assert proc.returncode == 0, proc.stderr
        assert "clean" in proc.stdout
