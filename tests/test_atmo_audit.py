# TIER = "fast"
"""AV 写死审计门单测：planted pattern 命中 / 干净文件不误报。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.atmo.audit import Finding, scan_file, scan_paths

_PLANT_TABLE = "[" + ", ".join(f"{0.1 * i:.2f}" for i in range(30)) + "]"
PLANTED = f'''
import numpy as np
pr_anchor = 990.0
TABLE = np.array({_PLANT_TABLE})
def apply_wet_dry(x):
    return x * TABLE[: x.size]
'''

CLEAN = '''
import numpy as np

def stefan_boltzmann(t, sigma=5.67e-8):
    return sigma * t ** 4

def relax(temp, target, rate):
    return temp + rate * (target - temp)
'''


def _scan_text(tmp_path, text, name="mod.py"):
    d = tmp_path / "scan_root"
    d.mkdir(exist_ok=True)
    p = d / name
    p.write_text(text, encoding="utf-8")
    return scan_paths([d])


def test_planted_patterns_detected_as_high(tmp_path):
    findings = _scan_text(tmp_path, PLANTED)
    highs = [f for f in findings if f.severity == "HIGH"]
    rules = {f.rule for f in highs}
    assert "forbidden_identifier" in rules      # pr_anchor
    assert "literal_table" in rules             # 30 元素字面量表
    assert any(f.detail.startswith("已知写死形态") for f in highs)


def test_clean_physics_code_no_high(tmp_path):
    findings = _scan_text(tmp_path, CLEAN)
    assert [f for f in findings if f.severity == "HIGH"] == []


def test_constant_climate_field_flagged_medium(tmp_path):
    src = "pr_surface = 2.3e-5\n\n\ndef f(x):\n    return x + pr_surface\n"
    findings = _scan_text(tmp_path, src)
    m = [f for f in findings if f.rule == "constant_climate_field"]
    assert len(m) == 1 and m[0].line == 1


def test_parse_error_reported_not_crash(tmp_path):
    findings = _scan_text(tmp_path, "def broken(:\n", name="bad.py")
    assert any(f.rule == "parse_error" for f in findings)


def test_scan_paths_missing_root_ignored(tmp_path):
    assert scan_paths([tmp_path / "no_such_dir"]) == []


def test_finding_dataclass_fields():
    f = Finding("HIGH", "rule", "path.py", 3, "detail")
    assert (f.severity, f.rule, f.path, f.line, f.detail) == ("HIGH", "rule", "path.py", 3, "detail")
