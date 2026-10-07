"""F-R21 回归：**preset 生成的 CLI 必须是目标脚本真正支持的开关**。

缺陷原型（R108 立号）
---------------------
`r97cal` preset 的网格键写作 `gain_multiplier`（下划线），而 `expand()` 直接拼 `--{key}`
⇒ 生成 `--gain_multiplier`，但 `argparse` 注册的是 `--gain-multiplier`
⇒ **参数不被识别**，该 preset 事实上不可直接执行（云端只能绕行 shell；无数据损失）。

本测试把「preset 的 CLI 与脚本开关一致」**机器化**：对**每一个** preset，
用目标脚本的 `--help` 取出真实开关集，再断言 preset 展开出的每个 `--flag` 都在其中
⇒ 同类缺陷（任意 preset、任意键名笔误）在 CI 即被拦，不必等云端跑挂。
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments import batch_runner as br  # noqa: E402

def _python_candidates() -> tuple[str, ...]:
    """解释器候选序（**只修取径，不动 preset 语义与期望值**——PI 22:0x 落裁①）。

    缺陷原型：原写作 `str(ROOT / ".venv" / "Scripts" / "python.exe")` 硬编码**主树** venv，
    而 `ROOT` 在 `.worktrees/<卡号>` 下指向 worktree（无 `.venv`）⇒ 该文件 10 例
    必 `FileNotFoundError`（主树同文件全绿；PI `806279a` 值班记逐例集合差互证）。
    ①当前解释器 = 跑 pytest 的那个 python（worktree 下天然回指主树 venv）；
    ②②Windows 主树 venv；③*nix/云机 venv。逐候选**验存在**后才用。
    """
    return (sys.executable,
            str(ROOT / ".venv" / "Scripts" / "python.exe"),
            str(ROOT / ".venv" / "bin" / "python"))


def _resolve_python(candidates: tuple[str, ...] | None = None) -> str:
    """返回第一个真实存在的解释器；全落空 ⇒ **当场报错**（fail-loud，禁 skip 伪装成绿）。"""
    cands = _python_candidates() if candidates is None else candidates
    for cand in cands:
        if cand:
            try:
                if Path(cand).is_file():
                    return str(cand)
            except OSError:
                continue
    raise RuntimeError(
        "找不到可用 Python 解释器，候选逐条验存在均落空："
        + " / ".join(str(c) for c in cands)
        + "（本测试须用仓库 venv 的解释器跑目标脚本 `--help`；不降级、不 skip）"
    )


PY = _resolve_python()
_FLAG_RE = re.compile(r"(--[A-Za-z0-9][-A-Za-z0-9]*)")
_HELP_CACHE: dict[str, set[str]] = {}


def _script_flags(script: str) -> set[str]:
    """目标脚本支持的开关全集（取自 `--help`）。"""
    if script not in _HELP_CACHE:
        out = subprocess.run([PY, str(ROOT / script), "--help"],
                             capture_output=True, text=True, cwd=str(ROOT))
        text = (out.stdout or "") + (out.stderr or "")
        _HELP_CACHE[script] = set(_FLAG_RE.findall(text))
        # 兜底：`--help` 本身
        _HELP_CACHE[script] |= {"--help", "-h"}
    return _HELP_CACHE[script]


# ------------------------------------------------------------ 单测：键名转换

def test_cli_flag_translates_underscore_to_hyphen():
    """F-R21 本体：键用下划线、CLI 用短横线。"""
    assert br.cli_flag("gain_multiplier") == "--gain-multiplier"
    assert br.cli_flag("max_count") == "--max-count"
    assert br.cli_flag("snapshot-every") == "--snapshot-every"   # 已是短横线 ⇒ 幂等
    assert br.cli_flag("arm") == "--arm"


def test_expand_keeps_template_placeholder_but_fixes_cli():
    """模板占位符仍用**原始键**（下划线），CLI 用短横线 —— 两者不得混淆。"""
    runs = br.expand(["gain_multiplier=1.0,1.3", "seed=42"],
                     ["mode=on", "max-count=3240"],
                     template="_tmp_test/m{gain_multiplier}_s{seed}.csv",
                     script="experiments/a4_verify_capacity.py",
                     workdir=Path("."))
    assert runs, "应展开出 run"
    argv = runs[0].cmd
    assert "--gain-multiplier" in argv and "--gain_multiplier" not in argv
    assert "--max-count" in argv
    # 模板用**原始键**（下划线）取值 ⇒ out 文件名里能看到 m 档与 seed
    assert runs[0].out.name == "m1.0_s42.csv"


def test_expand_grid_values_are_passed_positionally_after_flag():
    runs = br.expand(["seed=42,43"], [], "t_{seed}.csv",
                     "experiments/a4_verify_capacity.py", Path("."))
    assert [r.cmd[-1] for r in runs] == ["42", "43"]


# ------------------------------------------------------------ 全 preset 一致性（核心）

@pytest.mark.parametrize("name", sorted(br.PRESETS))
def test_preset_cli_flags_are_all_accepted_by_target_script(name):
    """🔴 **全 preset 排查**（机器化）：每个 `--flag` 都必须在目标脚本 `--help` 里出现。"""
    p = br.PRESETS[name]
    runs = br.preset_runs(name, Path("."))
    assert runs, f"{name}: 未展开出任何 run"
    valid = _script_flags(p["script"])
    bad: set[str] = set()
    for r in runs:
        for a in r.cmd:
            if a.startswith("--"):
                bad |= {a} - valid
    assert not bad, (
        f"{name}: 生成了目标脚本不支持的开关 {sorted(bad)}（F-R21 同型：键名与 "
        f"argparse 不一致）⇒ 该 preset 事实上不可直接执行"
    )


# 已执行完毕的**历史批**（其数据已在数据仓、且 run 名未与更早批次冲突）⇒ 不追溯本守卫。
# 本守卫自 2026-09-16（C 步）起对**新 preset 强制**。
LEGACY_PRESETS = {"r19", "d24", "d27dose", "d27recv", "r97cal", "r97cal_rand"}


@pytest.mark.parametrize("name", sorted(br.PRESETS))
def test_preset_shares_no_snapshot_dir_with_other_names(name):
    """🔴 **快照目录隔离守卫**：凡写快照的**新** preset 必须显式给 `--snapshot-dir`。

    理由（本项目真实事故型）：默认目录 `_rerun_logs/snap/` 下，run 名一旦与**历史批**重名
    （如 C 步的 `zero_s42` / `main_s42` 与 D-24 同名），引擎会**静默从旧快照续跑** ——
    `--ticks` 小于已跑 tick 时循环体为空 ⇒ 空产出、假失败（我在 preflight 首跑亲历过）。
    """
    if name in LEGACY_PRESETS:
        pytest.skip(f"{name}: 历史批（不追溯）")
    p = br.PRESETS[name]
    writes_snap = any(a.split("=")[0] == "snapshot-every" and a.split("=")[-1] != "0"
                      for a in p["fixed"])
    if not writes_snap:
        return
    dirs = [a.split("=", 1)[1] for a in p["fixed"] if a.startswith("snapshot-dir=")]
    assert dirs, (f"{name}: 写快照却未指定 `--snapshot-dir` ⇒ 会与历史批共用默认目录，"
                  f"存在静默续跑风险")
    assert dirs[0] != "_rerun_logs/snap", f"{name}: 不得使用默认快照目录（重名风险）"


def test_all_presets_have_required_keys():
    """结构自检：每个 preset 必备键（`template` 可用 `variants` 代替）。

    防手写 preset 漏字段 ⇒ 展开时 KeyError / 静默少跑。
    """
    for name, p in br.PRESETS.items():
        assert {"script", "grid", "fixed"} <= set(p), f"{name} 缺键"
        assert isinstance(p["grid"], list) and p["grid"], f"{name}: grid 须非空 list"
        assert p.get("template") or p.get("variants"), \
            f"{name}: 须有 template 或 variants（否则无处取 --out 模板）"
        if p.get("variants"):
            for v in p["variants"]:
                assert {"name", "args", "template"} <= set(v), f"{name}: variant 缺键"
                assert v["template"].count("{seed}") == 1, \
                    f"{name}/{v['name']}: 变体模板须含 {{seed}}"


def test_r97cal_preset_is_now_runnable_shape():
    """F-R21 的**定向回归**：`r97cal` 必须生成 `--gain-multiplier`（而非下划线版）。"""
    p = br.PRESETS["r97cal"]
    runs = br.expand(p["grid"], p["fixed"], p["template"], p["script"], Path("."))
    assert len(runs) == 18, "r97cal = 3 档 m × 6 seed = 18 run"
    assert all("--gain-multiplier" in r.cmd for r in runs)
    assert all("--calibration-arm" in r.cmd for r in runs), "校准臂旗必须在（条件 5 依据）"
    assert not any("--gain_multiplier" in r.cmd for r in runs)
    # 模板里的 m 值必须来自网格（1.0 / 1.3 / 1.5）
    ms = sorted({r.out.name.split("_")[0] for r in runs})
    assert ms == ["m1.0", "m1.3", "m1.5"]


def test_python_resolver_rejects_missing_candidates(tmp_path):
    """取径修复的**定向守卫**（变异必红）：候选逐条都不存在 ⇒ 必须当场 RuntimeError。

    退化原型 = 老写法"拼一个 `ROOT/.venv` 路径就交差"（worktree 下不存在 ⇒
    10 例 FileNotFoundError，红得看不出原因）。现在必须**验存在**且落空即炸，
    不许静默返回假路径、更不许 `pytest.skip` 把红洗成绿。
    """
    bad = (str(tmp_path / "nope.exe"), "", str(tmp_path / "nope2"))
    with pytest.raises(RuntimeError, match="找不到可用 Python 解释器"):
        _resolve_python(bad)


def test_python_resolver_prefers_first_existing_candidate(tmp_path):
    """存在性优先、顺序敏感：第一个存在的候选胜出（不回退硬编码主树路径）。"""
    ok = tmp_path / "python.exe"
    ok.write_text("")          # 只验 is_file，不执行
    assert _resolve_python((str(ok), "zzz")) == str(ok)
    # 目录不算存在（`.venv/` 半截目录不能当解释器）
    d = tmp_path / "dir"
    d.mkdir()
    assert _resolve_python((str(d), str(ok))) == str(ok)


def test_resolved_py_is_the_running_interpreter_in_worktrees():
    """双侧同绿的机读保证：`PY` 必须是**真实存在**的文件且能跑 `--help`。

    主树与 `.worktrees/<卡号>` 下都成立（PI 验收=双侧同绿）；`cwd=ROOT` 亦随
    `Path(__file__)` 落到当轮载体，不跨载体引用主树目录。
    """
    assert Path(PY).is_file(), f"PY 指向不存在的解释器：{PY}"
    out = subprocess.run([PY, "-c", "print(1)"], capture_output=True, text=True,
                         cwd=str(ROOT))
    assert out.returncode == 0 and out.stdout.strip() == "1"
