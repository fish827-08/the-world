"""AV 写死审计门：扫描 atmo 生产代码中的硬编码场/预设图案。

只读、只报、不改生产代码。扫描范围由调用方给定（正常 = atmo_core/ 与 atmo/）。
严重级：HIGH = 已知写死形态命中（pr_anchor 家族 / 大型字面量表）；MEDIUM = 可疑命名/常量气候场赋值。
退出码：有 HIGH 即 1。
"""
import argparse
import ast
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

# 已知写死形态（对齐写死账本 H4 与 B16-oracle 家族）：标识符级命中即 HIGH
FORBIDDEN_IDENTIFIERS = {
    "pr_anchor", "g_wet", "g_dry", "itcz_deg", "wet_amp", "dry_amp",
    "rain_band", "precip_band", "weather_table", "insolation_table",
}
# 可疑命名（需人工复核）：
SUSPICIOUS_NAME_RE = re.compile(
    r"(anchor|preset|hardcod|lookup_table|_band_|band_amplitude|天气表|日照表)", re.I
)
# 常量气候场赋值（模块级 target 名以气候场前缀开头且直接赋数值字面量）：
CLIMATE_FIELD_ASSIGN_RE = re.compile(r"^(pr|precip|rain|insol|toa|sw_|lw_|t2m|sst|cloud_)", re.I)
MIN_TABLE_ELEMENTS = 24


@dataclass
class Finding:
    severity: str
    rule: str
    path: str
    line: int
    detail: str


SEVERITY_RANK = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}


def _numeric_const(node) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
        and not isinstance(node.value, bool)


def _literal_tables(tree):
    out = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.List, ast.Tuple)):
            n = sum(1 for e in node.elts if _numeric_const(e))
            if n >= MIN_TABLE_ELEMENTS:
                out.append(node)
        elif isinstance(node, ast.Call):
            for arg in node.args:
                if isinstance(arg, ast.List):
                    n = sum(1 for e in arg.elts if _numeric_const(e))
                    if n >= MIN_TABLE_ELEMENTS:
                        out.append(node)
    return out


def _target_names(node):
    names = []
    if isinstance(node, ast.Assign):
        for t in node.targets:
            for sub in ast.walk(t):
                if isinstance(sub, ast.Name):
                    names.append(sub.id)
    return names


def scan_file(path: Path) -> list:
    src = path.read_text(encoding="utf-8")
    findings = []
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return [Finding("MEDIUM", "parse_error", str(path), e.lineno or 0, str(e))]
    lines = src.splitlines()
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Name):
            names = [node.id]
        elif isinstance(node, ast.Attribute):
            names = [node.attr]
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names = [node.name]
        for name in names:
            if name in FORBIDDEN_IDENTIFIERS:
                findings.append(Finding(
                    "HIGH", "forbidden_identifier", str(path), getattr(node, "lineno", 0),
                    f"已知写死形态标识符: {name}"))
            elif SUSPICIOUS_NAME_RE.search(name):
                findings.append(Finding(
                    "MEDIUM", "suspicious_name", str(path), getattr(node, "lineno", 0),
                    f"可疑命名: {name}"))
        if isinstance(node, ast.Assign) and getattr(node, "col_offset", 1) == 0:
            for tname in _target_names(node):
                if CLIMATE_FIELD_ASSIGN_RE.match(tname) and isinstance(node.value, ast.Constant) \
                        and _numeric_const(node.value):
                    findings.append(Finding(
                        "MEDIUM", "constant_climate_field", str(path), node.lineno,
                        f"模块级常量气候场赋值: {tname} = {node.value!r}"))
    for node in _literal_tables(tree):
        findings.append(Finding(
            "HIGH", "literal_table", str(path), node.lineno,
            f"字面量数值表 >= {MIN_TABLE_ELEMENTS} 个元素（疑似查表场）"))
    return findings


def scan_paths(roots) -> list:
    findings = []
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        for p in sorted(root.rglob("*.py")):
            findings.extend(scan_file(p))
    findings.sort(key=lambda f: (SEVERITY_RANK.get(f.severity, 9), f.path, f.line))
    return findings


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="atmo-audit")
    ap.add_argument("roots", nargs="*", default=["atmo_core", "atmo"],
                    help="扫描根目录（默认 atmo_core atmo）")
    a = ap.parse_args(argv)
    findings = scan_paths(a.roots)
    highs = [f for f in findings if f.severity == "HIGH"]
    print(json.dumps([asdict(f) for f in findings], ensure_ascii=False, indent=1))
    print(f"total={len(findings)} HIGH={len(highs)} "
          f"MEDIUM={sum(1 for f in findings if f.severity == 'MEDIUM')}")
    return 1 if highs else 0


if __name__ == "__main__":
    sys.exit(main())
