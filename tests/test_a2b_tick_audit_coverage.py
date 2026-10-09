"""T6 / A2b 守卫：`tools/tick_denomination_audit.py` 的**扫描覆盖**。

被守护的缺陷（验收件 §3.2）
---------------------------
旧版扫描清单是**手写**的 4 个路径，其中 `world/resources.py` 根本不存在
（真名 `world/resource_field.py`），且 `if p.is_file():` **静默跳过** ⇒
`world/` 6 个业务文件里只有 1 个真被扫到。变异测试证实：往
`world/light_and_temperature.py` 注入假 tick 常数**完全扫不出**。

本文件把三条纪律机器化：
  S1 覆盖自检**恒跑**（与 `--no-literals` 解耦，踩过"解耦错致静默 rc=0"的坑）
  S2 glob 派生（禁手写清单）：曾漏扫的 world 文件必须纳入、幽灵路径必须消失
  S3 **变异测试（核心）**：注入前不报 / 注入后必报（不跑变异测试 = 不算完成）
  S4/S5/S6 数量下限 / 缺失目录 / `--no-literals` 下仍 fail-loud（退出码 2）
  S7 GBK 控制台（R98：本工具 print 含非 GBK 字符，必须入口兜底）
  S8 `--md` 真落 Markdown 表（旧版接受该参数但从不使用，是死参数）
  S9 扫描目标解析失败 = 覆盖失败（旧版 `except SyntaxError: return []` 静默吞）

注：本文件所有 print/断言字符串只用 GBK 可编码字符（R98 守卫会扫 tests/）。
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "tick_denomination_audit.py"
MUTANT_FILE = ROOT / "world" / "light_and_temperature.py"
MUTANT_LINE = b"\n_A2B_MUTANT_DURATION_TICKS = 137  # duration\n"


def _load_tool():
    spec = importlib.util.spec_from_file_location("tick_denomination_audit", TOOL)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["tick_denomination_audit"] = mod
    spec.loader.exec_module(mod)
    return mod


tk = _load_tool()


def _run_tool(*args: str, env: dict | None = None) -> subprocess.CompletedProcess:
    """跑工具（显式 utf-8 解码：本机 GBK 下 `text=True` 会崩，属项目已知环境基线）。"""
    return subprocess.run(
        [sys.executable, str(TOOL), *args],
        capture_output=True, text=True,
        encoding="utf-8", errors="replace",
        cwd=str(ROOT), env=env, timeout=600)


def _scan_files_rel() -> set[str]:
    return {tk._rel(p) for p in tk._derive_scan_files()}


# ---------------------------------------------------------------- S1

def test_s1_coverage_selfcheck_runs_even_with_no_literals(tmp_path):
    """S1：`--no-literals` 只跳过字面量；覆盖自检与 JSON 的 scan_files 仍然在。"""
    jp = tmp_path / "a2b.json"
    r = _run_tool("--no-literals", "--json", str(jp))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "扫描覆盖自检" in r.stdout
    data = json.loads(jp.read_text(encoding="utf-8"))
    assert len(data["scan_files"]) >= tk._SCAN_MIN_FILES_TOTAL
    assert data["coverage"]["n_files"] == len(data["scan_files"])
    assert data["literals"] == []          # 快档确实跳过了字面量


# ---------------------------------------------------------------- S2

def test_s2_glob_covers_formerly_missing_files_and_drops_ghost():
    """S2：曾漏扫的 world/ 文件全部纳入；幽灵路径 world/resources.py 必须消失。"""
    files = _scan_files_rel()
    for rel in ("world/light_and_temperature.py", "world/resource_field.py",
                "world/subpos.py", "world/sphere_world.py",
                "world/resource_dynamics.py", "world/signal_field.py",
                "simulation/sphere_engine.py"):
        assert rel in files, f"{rel} 必须被 glob 纳入（旧手写清单漏扫）"
    assert "world/resources.py" not in files      # 幽灵路径（真名 resource_field.py）
    assert "simulation/config.py" not in files    # 验收口径：排除 config.py（字段已由 §3.1 全量枚举）
    assert "simulation/__init__.py" not in files


# ---------------------------------------------------------------- S3（核心变异测试）

def test_s3_mutation_inject_then_detect(tmp_path):
    """S3（核心，变异测试 / DEL-8 手法）：往 world/light_and_temperature.py 注入
    tick 常数 -- 注入前工具**不该报**、注入后**必须报**（旧版此处完全扫不出）。"""
    if not MUTANT_FILE.is_file():
        pytest.skip("靶文件不存在")
    orig = MUTANT_FILE.read_bytes()

    def _hits(rows: list[dict], lineno: int) -> list[dict]:
        return [x for x in rows
                if x["file"] == "world/light_and_temperature.py"
                and x["value"] == 137 and x["line"] == lineno]

    def _scan(tag: str) -> list[dict]:
        jp = tmp_path / f"{tag}.json"
        r = _run_tool("--json", str(jp))
        assert r.returncode in (0, 1), f"工具硬失败：\n{r.stdout}\n{r.stderr}"
        return json.loads(jp.read_text(encoding="utf-8"))["literals"]

    mutant_text = (orig + MUTANT_LINE).decode("utf-8")
    lineno = next(i for i, ln in enumerate(mutant_text.splitlines(), 1)
                  if "_A2B_MUTANT_DURATION_TICKS" in ln)
    try:
        assert not _hits(_scan("before"), lineno), "靶点不干净：注入前就报了 137"
        MUTANT_FILE.write_bytes(orig + MUTANT_LINE)
        hit = _hits(_scan("after"), lineno)
        assert hit, "注入后**扫不出** = 覆盖缺口回归（旧版正是此症状）"
        assert hit[0]["cls"] == "DURATION" and not hit[0]["noise"], hit[0]
    finally:
        MUTANT_FILE.write_bytes(orig)
    assert MUTANT_FILE.read_bytes() == orig, "靶文件未还原！"
    assert not _hits(_scan("restored"), lineno), "还原后仍报 137"


# ---------------------------------------------------------------- S4 / S5 / S6

def test_s4_file_floor_fail_loud(tmp_path):
    """S4：目录在但文件数低于下限 => CoverageError（拒绝"扫了个寂寞"）。"""
    for d in tk._SCAN_DIRS:
        (tmp_path / d).mkdir()
        (tmp_path / d / "tiny.py").write_text("x = 1\n", encoding="utf-8")
    with pytest.raises(tk.CoverageError):
        tk._derive_scan_files(root=tmp_path)


def test_s5_missing_dir_fail_loud(tmp_path):
    """S5：目录缺失 => CoverageError（不许 is_file() 式静默跳过）。"""
    with pytest.raises(tk.CoverageError):
        tk._derive_scan_files(root=tmp_path)


def test_s6_missing_dir_exits_2_even_with_no_literals(tmp_path, monkeypatch, capsys):
    """S6：目录缺失时 `--no-literals` 也必须 rc!=0（踩过的坑：曾解耦错 => 静默 rc=0）。"""
    monkeypatch.setattr(tk, "ROOT", tmp_path)      # 空目录：simulation/world 都不存在
    monkeypatch.setattr(sys, "argv", ["tick_denomination_audit.py", "--no-literals"])
    with pytest.raises(SystemExit) as ei:
        tk.main()
    assert ei.value.code == 2
    assert "A2b fail-loud" in capsys.readouterr().out


# ---------------------------------------------------------------- S7 / S8 / S9

def test_s7_gbk_console_does_not_crash(tmp_path):
    """S7（R98 同族）：GBK 控制台下不许因 print 崩掉（本工具 print 含非 GBK 字符）。"""
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "gbk"
    r = _run_tool("--no-literals", env=env)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "Traceback" not in r.stderr


def test_s8_md_table_is_written(tmp_path):
    """S8：`--md` 真落 Markdown 表（旧版接受该参数但从不使用 - 死参数，已实现）。"""
    mp = tmp_path / "audit.md"
    r = _run_tool("--no-literals", "--md", str(mp))
    assert r.returncode == 0, r.stdout + r.stderr
    text = mp.read_text(encoding="utf-8")
    assert "| 字段 | 默认值 | 类 | 依据 |" in text
    assert "`organisms.base_metabolism`" in text


def test_s9_syntax_error_is_fail_loud(tmp_path):
    """S9：扫描目标解析失败 => CoverageError（旧版 `except SyntaxError: return []` 静默吞）。"""
    bad = tmp_path / "broken.py"
    bad.write_text("def f(:\n    pass\n", encoding="utf-8")
    with pytest.raises(tk.CoverageError):
        tk._scan_literals(bad)


# ---------------------------------------------------------------- S10（2026-10-09 板桥补）

def test_s10_s0_kernel_scanned_and_memory_fields_classified():
    """S10（R394③ 三红修复守卫）：
    a) `s0_kernel/arms.py` + `harness.py` 必须在 glob 派生的发现项里（人口学旋钮承载 tick 面额）；
    b) G6 记忆档 5 字段必须有人工判后的 KNOWN 归类（未分类 => rc=1，s1/s7/s8 连坐红即此因）。"""
    files = _scan_files_rel()
    for rel in ("s0_kernel/arms.py", "s0_kernel/harness.py"):
        assert rel in files, f"{rel} 必须被纳入扫描（S0 装置旋钮不许留在清单外）"
    for full, want in (("info_structure.memory_ttl", tk.DURATION),
                       ("info_structure.memory_noise_p", tk.PROB),
                       ("info_structure.memory_dist_scale", tk.INVARIANT),
                       ("info_structure.memory_degrade_thr", tk.INVARIANT),
                       ("info_structure.memory_coarse_gain", tk.INVARIANT)):
        grp, name = full.split(".", 1)
        cls, _why = tk._suggest(grp, name)
        assert cls == want, f"{full} 应归 {want}，实得 {cls}"
        assert cls != "UNCLASSIFIED"
