"""气味场写入点守卫（R240 T8 / 设计稿 §8.3-4）—— 比照 `tests/test_sparse_write_guard.py`。

气味场是**新的全场状态**（`SmellField._S`，见 `world/smell_field.py`）⇒ 每个写入点必须自带
机器可读标签；另把设计稿的**两条红线**钉成结构测试：

| # | 红线 | 本守卫怎么查 |
|---|---|---|
| ① | **禁止"每 tick 全场稠密卷积"** | 模块 AST 里不得出现 `convolve` / `fft` / `scipy`（全场路径只许是"衰减 + 稀疏注入 + 粗网格拉普拉斯"） |
| ② | 全场路径**唯一**且**受节拍闸**约束 | 引擎里对 `smell.update(...)` 的调用，其**所在函数内**必须出现 `update_every`（= 节拍闸）；运行时节拍另有 `tests/test_smell_v1.py` 计数断言 |

标签（`world/smell_field.py` 内）：
`smell:full` 全场路径（衰减 / 扩散项叠加）｜`smell:inj` 稀疏源注入｜
`smell:reset` 整体替换（快照恢复 / 清场）｜`smell:init` 构造期初始化｜`smell:n/a` 非 `_S` 的临时量。
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: 被扫的数组 → (允许的接收者属性名, 本类所在文件名)
ARRAY_OWNERS = {
    "_S": ("smell", "smell_field.py"),     # 近场（全分辨率）
    "_Sc": ("smell", "smell_field.py"),    # 远场（粗网格）
}
SCANNED = (
    "world/smell_field.py",
    "simulation/sphere_engine.py",
)
TAG_RE = re.compile(r"#\s*smell:(full|inj|reset|init|n/a)\b")
#: 红线①：全场稠密卷积家族的禁用词（AST 层面查 import 名与属性名）
BANNED_CALL = {"convolve", "fftconvolve", "convolve2d", "filter2d"}
BANNED_IMPORT = {"scipy", "scipy.signal", "scipy.ndimage", "numpy.fft"}


def _chain(node: ast.AST):
    """展开 `a.b.c` ⇒ ['a','b','c']；不是纯属性链 ⇒ None。"""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return list(reversed(parts))
    return None


def _is_field_array(node: ast.AST, fname: str) -> bool:
    # 下标解包：`np.add.at(self._S[ci], ...)` / `self._Sc[ci]` 等 ⇒ 看基对象
    while isinstance(node, ast.Subscript):
        node = node.value
    chain = _chain(node)
    if not chain:
        return False
    if chain[-1] not in ARRAY_OWNERS:
        return False
    recv_ok, own_file = ARRAY_OWNERS[chain[-1]]
    if len(chain) >= 2 and chain[-2] == recv_ok:      # `X.smell._S`
        return True
    return fname == own_file and len(chain) == 2 and chain[0] == "self"   # `self._S`


class _WriteSites(ast.NodeVisitor):
    """收集 `_S` 的写入点：赋值/自增目标、`out=` 实参、`np.*.at` 首参。"""

    def __init__(self, fname: str) -> None:
        self.fname = fname
        self.func: list[ast.AST] = []
        self.sites: list[tuple[int, str, ast.AST | None]] = []

    def _add(self, lineno: int) -> None:
        fn = self.func[-1] if self.func else None
        self.sites.append((lineno, getattr(fn, "name", "<module>"), fn))

    def _target(self, t: ast.AST) -> None:
        node = t.value if isinstance(t, ast.Subscript) else t
        if _is_field_array(node, self.fname):
            self._add(t.lineno)

    def visit_Assign(self, node: ast.Assign) -> None:
        for t in node.targets:
            self._target(t)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self._target(node.target)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.target is not None:
            self._target(node.target)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        for kw in node.keywords:
            if kw.arg == "out" and _is_field_array(kw.value, self.fname):
                self._add(node.lineno)
        args = list(node.args)
        if (isinstance(node.func, ast.Attribute) and node.func.attr == "at"
                and args and _is_field_array(args[0], self.fname)):
            self._add(node.lineno)
        self.generic_visit(node)

    def visit_FunctionDef(self, node) -> None:
        self.func.append(node)
        self.generic_visit(node)
        self.func.pop()

    visit_AsyncFunctionDef = visit_FunctionDef


def _scan() -> list[dict]:
    out: list[dict] = []
    for rel in SCANNED:
        src = (ROOT / rel).read_text(encoding="utf-8")
        lines = src.splitlines()
        w = _WriteSites(Path(rel).name)
        w.visit(ast.parse(src))
        for lineno, func, fn in w.sites:
            code = lines[lineno - 1]
            m = TAG_RE.search(code)
            out.append({"file": rel, "line": lineno, "func": func,
                        "tag": m.group(1) if m else None, "code": code.strip(),
                        "func_seg": ast.get_source_segment(src, fn) if fn is not None else None})
    return out


def test_scan_is_not_vacuous_and_all_write_sites_annotated():
    """① 扫描非空转；② `_S` 的每个写入点都带 `# smell:<tag>` 标签。"""
    sites = _scan()
    assert len(sites) >= 4, (
        f"扫描只找到 {len(sites)} 个写入点 ⇒ 探测器失效（文件改名/结构变化？）")
    bad = [s for s in sites if s["tag"] is None]
    assert not bad, (
        "以下气味场写入点没有 `# smell:<tag>` 标注 ⇒ 新写入点可能绕过红线（§8.2）：\n"
        + "\n".join(f"  {s['file']}:{s['line']} [{s['func']}] {s['code']}" for s in bad))


def test_redline_no_dense_convolution():
    """🔴 红线①：气味场模块**不得**出现全场稠密卷积（convolve / fft / scipy）。"""
    src = (ROOT / "world/smell_field.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                assert a.name not in BANNED_IMPORT, f"禁 import {a.name}（红线①：全场稠密卷积）"
        if isinstance(node, ast.ImportFrom):
            assert (node.module or "") not in BANNED_IMPORT, f"禁 from {node.module} import"
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr not in BANNED_CALL, (
                f"禁 `.{node.func.attr}(...)`（红线①：全场稠密卷积 —— 会把 N=0 地板吃回去）")


def test_redline_single_full_path_behind_cadence_gate():
    """🔴 红线②：引擎对 `smell.update(...)` 的调用，所在函数内必须有节拍闸 `update_every`。"""
    src = (ROOT / "simulation/sphere_engine.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    found = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        chain = _chain(node.func) if isinstance(node.func, ast.Attribute) else None
        if not chain or chain[-1] != "update" or len(chain) < 2 or chain[-2] != "smell":
            continue
        found += 1
        seg = ast.get_source_segment(src, node) or ""
        # 找所在函数（用行号反查函数名/段）
        fseg = ""
        for f in ast.walk(tree):
            if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef)) and f.body:
                if f.lineno <= node.lineno <= (f.end_lineno or f.lineno):
                    fseg = ast.get_source_segment(src, f) or ""
                    break
        assert "update_every" in fseg, (
            f"sphere_engine.py:{node.lineno} 调用了 `{seg}`，但所在函数内没有 `update_every`"
            " ⇒ 全场路径未受节拍闸约束（红线②：每 tick 全场 = 地板税）")
    assert found >= 1, "扫描未找到 `smell.update(...)` 调用点 ⇒ 探测器失效"


def test_tag_semantics_are_where_they_claim():
    """标签与现实对齐：`init` 只在构造期；`reset` 只在整体替换处；`full`/`inj` 只在 `update`。"""
    sites = _scan()
    by_tag: dict[str, set[str]] = {}
    for s in sites:
        by_tag.setdefault(s["tag"] or "", set()).add(s["func"])
    assert by_tag.get("init", set()) == {"__init__"}, f"init 位置异常：{by_tag.get('init')}"
    assert by_tag.get("full", set()) <= {"update"}, f"full 位置异常：{by_tag.get('full')}"
    assert by_tag.get("inj", set()) <= {"update"}, f"inj 位置异常：{by_tag.get('inj')}"
    assert by_tag.get("reset", set()) <= {"restore", "clear", "load_snapshot"}, (
        f"reset 位置异常：{by_tag.get('reset')}")
    assert by_tag.get("full") and by_tag.get("inj"), "update 里必须同时有全场（full）与稀疏（inj）写点"
    # 引擎侧（若开档）：快照恢复点必须是 reset
    for s in sites:
        if s["file"].endswith("sphere_engine.py"):
            assert s["tag"] == "reset", f"引擎侧对 `smell._S` 的写入只允许在恢复点：{s}"