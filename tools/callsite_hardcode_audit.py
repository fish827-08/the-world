"""调用点硬编码审计（R213 §三 纪律）—— Rust 边界（`self._sim_core.*`）的实参不得是裸数字字面量。

为什么
------
A1 半接线（2026-09-26，云启发现 + 轻舟 P1 独立定位）：
`self._sim_core.reproduce_batch(..., 60.0)` 的冷却换算系数写死 `60.0`，
而 Python 参考路径读 `config.organisms.repro_cooldown_gene_scale`。
默认档两值相等（`C7` 只按默认值跑 ⇒ **永远抓不到**）；`k≠1` 时（该字段 = `60/k`）
双路径在**第一次生育**就分岔（云启实测 `born 0 != 1`）。
纪律（R213 裁定）：加速核调用点的参数必须**参数化**（读 config / 由调用方传入），
不得字面量直写 —— 本工具把这条纪律机器化。

判据
----
1. 扫 `simulation/`、`world/` 里全部 `self._sim_core.<fn>(...)` 调用；
2. 实参（位置或关键字）是**裸数字字面量**（含 `-0.5` 一元负号；`bool` 不算）⇒ **发现**；
3. 白名单（`WHITELIST`）逐条放行，**每条必须带理由**；白名单条目**失配即报**（防"白名单过期后继续放行"）；
4. 覆盖 fail-loud：目录缺失 / 文件数 / 调用点数 / 白名单命中数 下限 ⇒ `CoverageError` ⇒ rc=2。

口径备注（诚实边界）
--------------------
* 本扫描是**代理指标**：它抓"字面量直写"这一 A1 的**确切形态**；
  对"先存到变量再传"（`x = 60.0; f(..., x)`）不可见 —— 那类要靠双路径对拍（C7 + k≠1）。
* 白名单只放行**确无 config 字段**的设计常量（须在 `why` 里写明出处与参考路径同源证据）。

退出码：`0`=无发现且白名单全部命中｜`1`=有发现或白名单失配｜`2`=覆盖硬失败（fail-loud）

用法
----
    .venv\\Scripts\\python.exe tools/callsite_hardcode_audit.py
    .venv\\Scripts\\python.exe tools/callsite_hardcode_audit.py --json results/callsite_hardcode.json
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

# --- R98 纪律：Windows GBK 控制台兜底（非 ASCII print 会让脚本 rc=1 假失败；F-R15 族）---
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass  # 非 TTY / 旧解释器：不因诊断能力缺失而阻断运行

ROOT = Path(__file__).resolve().parents[1]
SCAN_DIRS = ("simulation", "world")
MIN_FILES_PER_DIR = 1
MIN_FILES_TOTAL = 6
MIN_CALLS = 10

# 白名单：确无 config 字段的设计常量。键 = (文件, 函数名, 参数标签, 值)。
# 🔴 新增条目**必须**在 `why` 里给出：出处（设计稿/裁定）+ "参考路径同源"证据。
WHITELIST: list[dict] = [
    {
        "file": "simulation/sphere_engine.py",
        "fn": "predation_and_culture",
        "arg": "pos[20]",
        "value": 0.1,
        "why": "culture_alpha：文化学习 EWMA 速率（`interpret += alpha*(mean-interpret)`）。"
               "config 无对应字段（`PleasureConfig.alpha` 是愉悦度预期学习率，不同物）；"
               "Python 参考路径同一常量 0.1 ⇒ 双路径同源，非半接线。",
    },
]


class CoverageError(RuntimeError):
    """扫描覆盖不达标（fail-loud：绝不静默返回空结果）。"""


def _num(node: ast.AST):
    """数字字面量取值；非数字返回 `_MISS`（哨兵）。"""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
            and not isinstance(node.value, bool):
        return node.value
    if (isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub)
            and isinstance(node.operand, ast.Constant)):
        v = node.operand.value
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return -v
    return _MISS


_MISS = object()


def _is_sim_core_call(node: ast.AST) -> bool:
    """`self._sim_core.<fn>(...)` 形态。"""
    if not isinstance(node, ast.Call):
        return False
    f = node.func
    return (isinstance(f, ast.Attribute) and isinstance(f.value, ast.Attribute)
            and isinstance(f.value.value, ast.Name) and f.value.value.id == "self"
            and f.value.attr == "_sim_core")


def _literals(call: ast.Call) -> list[tuple[str, object]]:
    """调用实参里的裸数字字面量：`[("pos[i]", 值) | ("kw[名]", 值), ...]`。"""
    out: list[tuple[str, object]] = []
    for i, a in enumerate(call.args):
        v = _num(a)
        if v is not _MISS:
            out.append((f"pos[{i}]", v))
    for kw in call.keywords:
        if kw.arg is None:                      # `**kwargs` 展开：不猜
            continue
        v = _num(kw.value)
        if v is not _MISS:
            out.append((f"kw[{kw.arg}]", v))
    return out


def scan(root: Path = ROOT, dirs: tuple[str, ...] = SCAN_DIRS,
         min_files_per_dir: int = MIN_FILES_PER_DIR, min_files_total: int = MIN_FILES_TOTAL,
         min_calls: int = MIN_CALLS, whitelist: list[dict] | None = None) -> dict:
    """扫描并分类；覆盖不达标抛 `CoverageError`。返回 dict（可直接 JSON 化）。"""
    wl = WHITELIST if whitelist is None else whitelist
    files: list[Path] = []
    for d in dirs:
        dd = root / d
        if not dd.is_dir():
            raise CoverageError(f"扫描目录不存在：{dd}")
        found = sorted(p for p in dd.rglob("*.py") if p.name not in ("config.py", "__init__.py"))
        if len(found) < min_files_per_dir:
            raise CoverageError(f"{d} 只找到 {len(found)} 个文件（下限 {min_files_per_dir}）—— 覆盖可疑")
        files.extend(found)
    if len(files) < min_files_total:
        raise CoverageError(f"总文件数 {len(files)} < 下限 {min_files_total}")

    n_calls = 0
    findings, matched = [], []
    for p in files:
        try:
            tree = ast.parse(p.read_text(encoding="utf-8"))
        except SyntaxError as e:
            raise CoverageError(f"解析失败（覆盖硬失败）：{p}: {e}") from e
        rel = p.relative_to(root).as_posix()
        for node in ast.walk(tree):
            if not _is_sim_core_call(node):
                continue
            n_calls += 1
            for label, value in _literals(node):
                hit = next((w for w in wl if w["file"] == rel and w["fn"] == node.func.attr
                            and w["arg"] == label and w["value"] == value), None)
                row = {"where": f"{rel}:{node.lineno}", "fn": node.func.attr,
                       "arg": label, "value": repr(value)}
                if hit is not None:
                    matched.append(row)
                else:
                    findings.append(row)

    if n_calls < min_calls:
        raise CoverageError(f"调用点数 {n_calls} < 下限 {min_calls} —— 解析或覆盖可疑")
    stale = [w for w in wl if not any(
        m["fn"] == w["fn"] and m["arg"] == w["arg"] and m["value"] == repr(w["value"])
        and m["where"].startswith(f"{w['file']}:") for m in matched)]
    return {
        "scan_files": [p.relative_to(root).as_posix() for p in files],
        "coverage": {"n_files": len(files), "n_calls": n_calls,
                     "n_whitelist": len(wl), "n_whitelist_matched": len(matched)},
        "findings": findings, "whitelist_matched": matched, "stale_whitelist": stale,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="调用点硬编码审计（Rust 边界实参不得裸数字字面量；R213 §三）")
    ap.add_argument("--json", default="", help="把清点结果写入该 JSON 路径")
    a = ap.parse_args()

    try:
        rep = scan()
    except CoverageError as e:
        print(f"[覆盖硬失败] {e}")
        sys.exit(2)

    cov = rep["coverage"]
    print(f"== 调用点硬编码审计：{cov['n_files']} 文件｜`_sim_core.*` 调用 {cov['n_calls']}"
          f"｜白名单 {cov['n_whitelist']}（命中 {cov['n_whitelist_matched']}） ==")
    print(f"❗ 裸数字字面量实参（白名单外）：{len(rep['findings'])}")
    for m in rep["findings"]:
        print(f"   {m['where']}  {m['fn']}  {m['arg']} = {m['value']}")
    if rep["stale_whitelist"]:
        print(f"❗ 白名单失配（该条已无对应调用点 ⇒ 过期，须删）：{len(rep['stale_whitelist'])}")
        for w in rep["stale_whitelist"]:
            print(f"   {w['file']}  {w['fn']}  {w['arg']} = {w['value']!r}")

    if a.json:
        out = Path(a.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            json.dump(rep, fh, ensure_ascii=False, indent=2)
        print(f"明细已写入 {out}")

    bad = bool(rep["findings"]) or bool(rep["stale_whitelist"])
    print("⇒ " + ("有不一致（须裁定/修）" if bad else "OK：边界实参全部参数化（或白名单在册）"))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
