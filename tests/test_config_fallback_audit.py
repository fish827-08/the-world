"""R208 §三 守卫：`tools/config_fallback_audit.py` —— 兜底值必须 == config 默认值。

被守护的缺陷
------------
`getattr(cfg, "字段", 兜底)` 的兜底会在 config 默认值变更后**悄悄过期**：
实例 `world/resource_dynamics.py` 的 `rest_ticks` 兜底 300 vs config 60（5 倍），
同族还有 `kill_frac` 0.7 vs 10.0（比例口径 -> 再生倍数口径的遗留）。
兜底只在**旧存档缺字段**时触发 ⇒ 触发即静默改生态；
"向前兼容"的正确语义是旧存档拿**当前**默认值（R208 §三 裁定）。

本文件把纪律机器化：
  S1 变异测试（核心）：**别名形态**（`g = lambda k, d: getattr(cfg, k, d)`）注入不一致必报、对齐不报
  S2 直呼 `getattr(x, "字段", 字面量)` 形态同样检出；`None` 兜底（存在性检查）不算发现
  S3 真仓扫描：不一致数必须为 0（纪律已全仓对齐）；覆盖下限达标
  S4/S5 覆盖 fail-loud：目录缺失 / 文件数过少 -> CoverageError
  S6 CLI：真仓 rc=0；`--json` 真落盘
  S7 R98：入口 UTF-8 兜底存在（工具 print 含非 GBK 字符）
  S8 R213 §六：4 处"闸型"兜底固定为"兜底 == config 默认值"（不许回退成旧闸值）

注：本文件不用 print；断言串保持 GBK 可编码（R98 守卫扫 tests/）。
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "config_fallback_audit.py"

CFG_FIXTURE = """\
class Cfg:
    rest_ticks: int = 60
    kill_frac: float = 10.0
"""


def _load_tool():
    spec = importlib.util.spec_from_file_location("config_fallback_audit", TOOL)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["config_fallback_audit"] = mod
    spec.loader.exec_module(mod)
    return mod


cfa = _load_tool()


def _make_tree(tmp_path: Path) -> Path:
    (tmp_path / "simulation").mkdir()
    (tmp_path / "simulation" / "config.py").write_text(CFG_FIXTURE, encoding="utf-8")
    (tmp_path / "world").mkdir()
    return tmp_path


def _scan_fixture(root: Path) -> dict:
    return cfa.scan(root=root, dirs=("world",), min_files_per_dir=1,
                    min_files_total=1, min_hits=1)


def test_s1_mutation_alias_form_inject_then_detect(tmp_path):
    """核心：真实代码用的别名形态 —— 注入不一致必报，改回默认不报。"""
    root = _make_tree(tmp_path)
    mod = root / "world" / "mod.py"
    mod.write_text(
        "def build(cfg):\n"
        "    g = lambda k, d: getattr(cfg, k, d)\n"
        "    return int(g('rest_ticks', 300))\n",
        encoding="utf-8")
    rep = _scan_fixture(root)
    assert [m["field"] for m in rep["mismatches"]] == ["rest_ticks"], rep["mismatches"]
    assert rep["mismatches"][0]["fallback"] == "300"
    assert rep["mismatches"][0]["config_default"] == "60"

    mod.write_text(
        "def build(cfg):\n"
        "    g = lambda k, d: getattr(cfg, k, d)\n"
        "    return int(g('rest_ticks', 60))\n",
        encoding="utf-8")
    assert _scan_fixture(root)["mismatches"] == []


def test_s2_direct_getattr_and_none_existence_check(tmp_path):
    """直呼形态生效；`None` 兜底（存在性检查）单独计数、不算发现。"""
    root = _make_tree(tmp_path)
    (root / "world" / "mod.py").write_text(
        "def build(cfg, obj):\n"
        "    a = getattr(cfg, 'kill_frac', 0.7)\n"
        "    b = getattr(cfg, 'kill_frac', 10.0)\n"
        "    c = getattr(obj, 'ars', None)\n"
        "    return a, b, c\n",
        encoding="utf-8")
    rep = _scan_fixture(root)
    assert len(rep["mismatches"]) == 1
    assert rep["mismatches"][0]["field"] == "kill_frac"
    assert rep["coverage"]["n_none_fallback"] == 1


def test_s3_real_repo_has_zero_mismatch_and_coverage(tmp_path):
    """真仓：纪律已全仓对齐（不一致 = 0）；覆盖下限达标。"""
    rep = cfa.scan()
    assert rep["mismatches"] == [], (
        "兜底与 config 默认不一致（R208 纪律）：\n"
        + "\n".join(f"  {m['where']} {m['field']}: {m['fallback']} vs {m['config_default']}"
                    for m in rep["mismatches"]))
    cov = rep["coverage"]
    assert cov["n_files"] >= 8
    assert cov["n_hits"] >= 50
    assert cov["n_alias_files"] >= 1, "别名形态至少出现在 1 个文件（resource_dynamics 的 g 别名）"


def test_s4_missing_dir_is_fail_loud(tmp_path):
    (tmp_path / "simulation").mkdir()
    (tmp_path / "simulation" / "config.py").write_text(CFG_FIXTURE, encoding="utf-8")
    with pytest.raises(cfa.CoverageError):
        cfa.scan(root=tmp_path, dirs=("world",), min_files_per_dir=1,
                 min_files_total=1, min_hits=1)


def test_s5_file_floor_is_fail_loud(tmp_path):
    root = _make_tree(tmp_path)
    (root / "world" / "mod.py").write_text("x = 1\n", encoding="utf-8")
    with pytest.raises(cfa.CoverageError):
        cfa.scan(root=root, dirs=("world",), min_files_per_dir=3,
                 min_files_total=3, min_hits=1)


def test_s6_cli_real_repo_rc0_and_json(tmp_path):
    out = tmp_path / "rep.json"
    cp = subprocess.run(
        [sys.executable, str(TOOL), "--json", str(out)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        cwd=str(ROOT), timeout=600)
    assert cp.returncode == 0, cp.stdout + cp.stderr
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["mismatches"] == []
    assert doc["coverage"]["n_files"] >= 8


def test_s7_gbk_guard_present_in_tool():
    """R98：工具 print 含非 GBK 字符（字段歧义等），入口必须有 UTF-8 兜底。"""
    assert 'reconfigure(encoding="utf-8"' in TOOL.read_text(encoding="utf-8")


def test_s8_r213_gate_type_fallbacks_pin_config_defaults():
    """R213 §六：4 处"闸型"兜底维持"兜底 == config 默认值"（不回退成旧闸值）。

    背景：这 4 处的**旧兜底**语义上分别是"机制关/更严"（w_fear_health 0.0 关血条恐惧项、
    need_aggression_k 0.0 关饥饿激进、eat_threshold_frac 1.0 更严、assim_return_frac 0.0 不回灌）。
    按 R208 §三 字面（兜底 == 当前默认）已对齐；R213 §六 裁定维持对齐、不做 enabled 闸门。
    本守卫把裁定值钉死：任何回退或私自改值都会红（若 config 默认属裁定变更，请一并更新本表）。
    """
    rep = cfa.scan()
    gates = {
        "w_fear_health": {"0.5"},
        "need_aggression_k": {"0.5"},
        "eat_threshold_frac": {"0.0"},
        "assim_return_frac": {"1.0"},
    }
    hits_by_field: dict[str, list[dict]] = {}
    for h in rep["hits"]:
        hits_by_field.setdefault(h["field"], []).append(h)
    for field, expected in gates.items():
        rows = hits_by_field.get(field, [])
        assert rows, f"{field} 没有兜底点（覆盖可疑/字段被改名）——R213 §六 守卫失效"
        for r in rows:
            assert r["fallback"] == r["config_default"], (
                f"R213 §六：兜底须 == config 默认值；{r['where']} {field} "
                f"兜底 {r['fallback']} vs 默认 {r['config_default']}")
            assert r["fallback"] in expected, (
                f"R213 §六 已裁定 {field} 兜底为 {expected}，实际 {r['fallback']}"
                f"（{r['where']}）——回退须回板重裁")
    assert [m for m in rep["mismatches"]] == []
