"""R213 §三 守卫：`tools/callsite_hardcode_audit.py` —— Rust 边界实参不得裸数字字面量。

被守护的缺陷（A1 半接线）
------------------------
`self._sim_core.reproduce_batch(..., 60.0)`：Python 参考路径读
`config.organisms.repro_cooldown_gene_scale`，Rust 调用点写死 60.0。
默认档两值相等 ⇒ **C7 永远抓不到**；`k!=1` 时双路径第一次生育就分岔。
本文件把"边界必须参数化"机器化（口径与诚实边界见工具 docstring）。

  S1 变异测试（核心）：临时树注入 `..., 60.0` 必报；改成读 config 不报
  S2 白名单：在册字面量不报且计入命中；**失配条目**（已无对应调用点）必报为过期
  S3 真仓扫描：发现数必须为 0、白名单全命中、覆盖下限达标
  S4 覆盖 fail-loud：目录缺失 / 调用点数过少 -> CoverageError
  S5 CLI：真仓 rc=0；`--json` 真落盘且 findings/stale 均为空
  S6 R98：入口 UTF-8 兜底存在（工具 print 含非 GBK 字符）

注：本文件不用 print；断言串保持 GBK 可编码（R98 守卫扫 tests/）。
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from experiments.scan_nonascii_print import source_is_protected as _r98_protected

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "callsite_hardcode_audit.py"

GOOD_CALL = """\
class E:
    def step(self, cfg):
        self._sim_core.step_movement(a, b, cfg.organisms.repro_cooldown_gene_scale)
"""

BAD_CALL = """\
class E:
    def step(self, cfg):
        self._sim_core.step_movement(a, b, 60.0)
"""


def _load_tool():
    spec = importlib.util.spec_from_file_location("callsite_hardcode_audit", TOOL)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["callsite_hardcode_audit"] = mod
    spec.loader.exec_module(mod)
    return mod


csa = _load_tool()


def _make_tree(tmp_path: Path, body: str) -> Path:
    (tmp_path / "simulation").mkdir(exist_ok=True)
    (tmp_path / "simulation" / "eng.py").write_text(body, encoding="utf-8")
    (tmp_path / "world").mkdir(exist_ok=True)
    (tmp_path / "world" / "sanity.py").write_text("x = 1\n", encoding="utf-8")
    return tmp_path


def _scan_fixture(root: Path, whitelist: list[dict] | None = None) -> dict:
    return csa.scan(root=root, dirs=("simulation", "world"), min_files_total=1,
                    min_calls=1, whitelist=whitelist if whitelist is not None else [])


def test_s1_mutation_inject_then_detect(tmp_path):
    """核心：注入裸字面量必报（含参数标签与值），改成读 config 不报。"""
    root = _make_tree(tmp_path, BAD_CALL)
    rep = _scan_fixture(root)
    assert [f["fn"] for f in rep["findings"]] == ["step_movement"], rep["findings"]
    assert rep["findings"][0]["arg"] == "pos[2]"
    assert rep["findings"][0]["value"] == "60.0"

    (root / "simulation" / "eng.py").write_text(GOOD_CALL, encoding="utf-8")
    rep2 = _scan_fixture(root)
    assert rep2["findings"] == []
    assert rep2["coverage"]["n_calls"] == 1


def test_s2_whitelist_matches_and_stale_is_loud(tmp_path):
    """在册字面量放行并计入命中；失配条目（该条已无对应调用点）报为过期。"""
    root = _make_tree(tmp_path, BAD_CALL)
    live = [{"file": "simulation/eng.py", "fn": "step_movement", "arg": "pos[2]",
             "value": 60.0, "why": "fixture"}]
    rep = _scan_fixture(root, whitelist=live)
    assert rep["findings"] == []
    assert rep["coverage"]["n_whitelist_matched"] == 1
    assert rep["stale_whitelist"] == []

    stale = live + [{"file": "simulation/eng.py", "fn": "nonexistent_fn", "arg": "pos[0]",
                     "value": 1.0, "why": "fixture-stale"}]
    rep2 = _scan_fixture(root, whitelist=stale)
    assert [w["fn"] for w in rep2["stale_whitelist"]] == ["nonexistent_fn"]


def test_s3_real_repo_zero_findings_and_coverage():
    """真仓：边界已全参数化（发现 = 0）；白名单全命中、覆盖下限达标。"""
    rep = csa.scan()
    assert rep["findings"] == [], (
        "Rust 边界出现裸数字字面量（R213 纪律）：\n"
        + "\n".join(f"  {m['where']} {m['fn']} {m['arg']} = {m['value']}"
                    for m in rep["findings"]))
    assert rep["stale_whitelist"] == [], (
        "白名单失配（条目过期须删）：\n"
        + "\n".join(f"  {w['file']} {w['fn']} {w['arg']}" for w in rep["stale_whitelist"]))
    cov = rep["coverage"]
    assert cov["n_files"] >= 6
    assert cov["n_calls"] >= 10
    assert cov["n_whitelist"] >= 1
    assert cov["n_whitelist_matched"] == cov["n_whitelist"], "白名单必须全部命中（否则即过期）"


def test_s4_coverage_fail_loud(tmp_path):
    (tmp_path / "simulation").mkdir()
    with pytest.raises(csa.CoverageError):
        csa.scan(root=tmp_path, dirs=("simulation", "world"), min_files_total=1,
                 min_calls=1, whitelist=[])

    root = _make_tree(tmp_path, GOOD_CALL)
    with pytest.raises(csa.CoverageError):
        csa.scan(root=root, dirs=("simulation", "world"), min_files_total=1,
                 min_calls=5, whitelist=[])


def test_s5_cli_real_repo_rc0_and_json(tmp_path):
    out = tmp_path / "rep.json"
    cp = subprocess.run(
        [sys.executable, str(TOOL), "--json", str(out)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=str(ROOT), timeout=600)
    assert cp.returncode == 0, cp.stdout + cp.stderr
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["findings"] == []
    assert doc["stale_whitelist"] == []
    assert doc["coverage"]["n_calls"] >= 10


def test_s6_gbk_guard_present_in_tool():
    """R98：工具 print 含非 GBK 字符，入口必须有 UTF-8 兜底。

    判据**派生**自 R98 守卫本体（`scan_nonascii_print.source_is_protected`，AST 检调用点）
    —— 不在本文件复制"含标记串"式的弱判据（R216 §三）。
    """
    assert _r98_protected(TOOL.read_text(encoding="utf-8"))
