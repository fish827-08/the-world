"""config 兜底值审计（R208 §三 纪律）—— `getattr(cfg, "字段", 字面量)` 的兜底必须 == config 默认值。

为什么
------
`world/resource_dynamics.py:113` 的 `g("rest_ticks", 300)` 与
`ResourceDynamicsConfig.rest_ticks = 60`（config.py:1044）不一致：兜底只在**旧存档缺字段**时触发，
一旦触发就把休耕从 60 变 300（5 倍）⇒ **静默改生态**。
纪律（R208 裁定）：**兜底值必须等于 config 默认值** —— "向前兼容"的正确语义是
"旧存档拿到**当前**默认值"，而不是拿一个历史值。

清点口径
--------
* 目标 = `simulation/`、`world/`（glob 派生，排除 `config.py`/`__init__.py`）里的兜底点：
  * `getattr(x, "字段", 字面量)` 直呼；
  * **别名形态**（历史 bug 就是这种）：`g = lambda k, d: getattr(cfg, k, d)` 之后的 `g("字段", 字面量)` ——
    ⇒ 只扫 `getattr` 直呼会**漏掉真正的那个坑**（`rest_ticks`），故必须跟别名；
* `None` 兜底 = "对象存在性检查"（如 `getattr(config, "ars", None)`）⇒ 单独计数，**不算发现**；
* 字段名在 `simulation/config.py` **唯一** ⇒ 与默认值比对，**不等 = 不一致**（退出码 1，须裁定）；
* 同名多默认（如 `enabled`/`gain`）⇒ **字段歧义**，仅报告（目标对象不一定是那一类 config）；
* 不在 config ⇒ 仅报告（多半是运行时状态对象的兜底）。

退出码：`0`=无不一致｜`1`=有不一致（纪律违反）｜`2`=覆盖硬失败（fail-loud）
覆盖自检**恒跑**：目录不存在／文件数过少／命中数过少 ⇒ `CoverageError` ⇒ 退出码 2。
（A2b 教训：清单/兜底写死 ⇒ 必然腐坏；覆盖必须 glob 派生 + 数量下限。）

用法
----
    .venv\\Scripts\\python.exe tools/config_fallback_audit.py
    .venv\\Scripts\\python.exe tools/config_fallback_audit.py --json results/cfg_fallback.json
"""
from __future__ import annotations

import argparse
import ast
import io
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
CONFIG_REL = "simulation/config.py"
MIN_FILES_PER_DIR = 4
MIN_FILES_TOTAL = 8
MIN_HITS = 50

_MISS = object()


class CoverageError(RuntimeError):
    """扫描覆盖不达标（fail-loud：绝不静默返回空结果）。"""


def _lit(node: ast.AST):
    """字面量取值；非字面量返回 `_MISS`（哨兵）。"""
    if isinstance(node, ast.Constant):
        return node.value
    if (isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub)
            and isinstance(node.operand, ast.Constant)):
        v = node.operand.value
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return -v
    return _MISS


def config_defaults(config_path: Path) -> dict[str, list[tuple[str, object]]]:
    """`字段名 -> [(类名, 默认值), ...]`（AST 解析，含全部 dataclass 的注解赋值）。"""
    tree = ast.parse(config_path.read_text(encoding="utf-8"))
    out: dict[str, list[tuple[str, object]]] = {}
    for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
        for stmt in cls.body:
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.value is not None:
                v = _lit(stmt.value)
                if v is not _MISS:
                    out.setdefault(stmt.target.id, []).append((cls.name, v))
    return out


def _is_fallback_getattr(call: ast.AST, params: list[str]) -> bool:
    """`getattr(<obj>, params[0], params[1])` 形态（别名 lambda 的本体）。"""
    return (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
            and call.func.id == "getattr" and len(call.args) == 3
            and isinstance(call.args[1], ast.Name) and call.args[1].id == params[0]
            and isinstance(call.args[2], ast.Name) and call.args[2].id == params[1])


def fallback_aliases(tree: ast.Module) -> set[str]:
    """找出绑定到 `lambda k, d: getattr(obj, k, d)`（或等价单行 `def`）的局部别名名。"""
    names: set[str] = set()
    for node in ast.walk(tree):
        pairs: list[tuple[str, ast.AST | None]] = []
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)):
            pairs.append((node.targets[0].id, node.value))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            pairs.append((node.target.id, node.value))
        elif (isinstance(node, ast.FunctionDef) and len(node.body) == 1
                and isinstance(node.body[0], ast.Return)):
            params = [a.arg for a in node.args.args]
            ret = node.body[0].value
            if len(params) == 2 and ret is not None and _is_fallback_getattr(ret, params):
                names.add(node.name)
        for name, val in pairs:
            if val is None:
                continue
            if isinstance(val, ast.Lambda):
                params = [a.arg for a in val.args.args]
                if len(params) == 2 and _is_fallback_getattr(val.body, params):
                    names.add(name)
    return names


def _iter_sites(tree: ast.Module, aliases: set[str]):
    """产出 (行号, 字段名, 兜底字面量) —— 直呼 getattr + 别名调用两种形态。"""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name_node = default_node = None
        if (isinstance(node.func, ast.Name) and node.func.id == "getattr"
                and len(node.args) == 3):
            name_node, default_node = node.args[1], node.args[2]
        elif isinstance(node.func, ast.Name) and node.func.id in aliases and len(node.args) == 2:
            name_node, default_node = node.args[0], node.args[1]
        if name_node is None:
            continue
        if not isinstance(name_node, ast.Constant) or not isinstance(name_node.value, str):
            continue
        default = _lit(default_node)
        if default is _MISS:
            continue
        yield node.lineno, name_node.value, default


def scan(root: Path = ROOT, dirs: tuple[str, ...] = SCAN_DIRS,
         config_rel: str = CONFIG_REL, min_files_per_dir: int = MIN_FILES_PER_DIR,
         min_files_total: int = MIN_FILES_TOTAL, min_hits: int = MIN_HITS) -> dict:
    """扫描并分类；覆盖不达标抛 `CoverageError`。返回 dict（可直接 JSON 化）。"""
    config_path = root / config_rel
    if not config_path.is_file():
        raise CoverageError(f"config 不存在：{config_path}")
    cfgdef = config_defaults(config_path)

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

    mism, ambig, unknown = [], [], []
    n_hits = n_none = n_alias_files = 0
    for p in files:
        try:
            tree = ast.parse(p.read_text(encoding="utf-8"))
        except SyntaxError as e:
            raise CoverageError(f"解析失败（覆盖硬失败）：{p}: {e}") from e
        aliases = fallback_aliases(tree)
        if aliases:
            n_alias_files += 1
        rel = p.relative_to(root).as_posix()
        for lineno, field, default in _iter_sites(tree, aliases):
            if default is None:
                n_none += 1
                continue
            n_hits += 1
            where = f"{rel}:{lineno}"
            found = cfgdef.get(field)
            if not found:
                unknown.append({"where": where, "field": field, "fallback": repr(default)})
            elif len({v for _, v in found}) > 1:
                ambig.append({"where": where, "field": field, "fallback": repr(default),
                              "candidates": [[c, repr(v)] for c, v in found]})
            elif found[0][1] != default:
                mism.append({"where": where, "field": field, "fallback": repr(default),
                             "config_default": repr(found[0][1]), "config_class": found[0][0]})

    if n_hits < min_hits:
        raise CoverageError(f"命中数 {n_hits} < 下限 {min_hits} —— 解析或覆盖可疑")
    return {
        "scan_files": [p.relative_to(root).as_posix() for p in files],
        "coverage": {"n_files": len(files), "n_hits": n_hits, "n_none_fallback": n_none,
                     "n_alias_files": n_alias_files},
        "mismatches": mism, "ambiguous": ambig, "unknown": unknown,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="config 兜底值审计（兜底必须 == config 默认值）")
    ap.add_argument("--json", default="", help="把清点结果写入该 JSON 路径")
    a = ap.parse_args()

    try:
        rep = scan()
    except CoverageError as e:
        print(f"[覆盖硬失败] {e}")
        sys.exit(2)

    cov = rep["coverage"]
    print(f"== config 兜底值审计：{cov['n_files']} 文件｜字面量兜底 {cov['n_hits']}"
          f"｜None 兜底（存在性检查）{cov['n_none_fallback']} ==")
    print(f"❗ 与 config 唯一默认不一致：{len(rep['mismatches'])}")
    for m in rep["mismatches"]:
        print(f"   {m['where']}  {m['field']}: 兜底 {m['fallback']}"
              f" vs {m['config_class']}.{m['field']} = {m['config_default']}")
    print(f"字段歧义（同名多默认，仅报告）：{len(rep['ambiguous'])}"
          f"｜不在 config（仅报告）：{len(rep['unknown'])}")
    for u in rep["unknown"]:
        print(f"   ? {u['where']}  {u['field']}={u['fallback']}")

    if a.json:
        out = Path(a.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with io.open(out, "w", encoding="utf-8") as fh:
            json.dump(rep, fh, ensure_ascii=False, indent=2)
        print(f"\n已写入 {out}")

    sys.exit(1 if rep["mismatches"] else 0)


if __name__ == "__main__":
    main()
