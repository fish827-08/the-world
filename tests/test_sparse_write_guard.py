"""稀疏化 B 写入点守卫（R219 §二-2）：字段数组的写入点**由扫描派生**（AST），
新增未标注写入 ⇒ 红灯（fail-loud）。

背景：惰性再生/稀疏信号的**唯一正确性风险 = 漏标脏**（漏掉一个能减少 `_grid` 的
写入点 ⇒ 该格少长一 tick ⇒ 破逐位等价）。"人工清单不是验收物"（教训 ⑯）
⇒ 本守卫扫描生产代码（`world/` + `simulation/`），要求**每个**字段数组写入点
在本行带机器可读标签 `# sparse:<tag>`：

| tag | 含义 |
|---|---|
| `mark` | 该写入**能减少**数组 ⇒ 必须打脏 / 并入活跃集（资源侧 = `consume`/`consume_many`） |
| `active` | 信号类写入 ⇒ 必须在**同一函数内**维护活跃集（`write`/`write_many`/稀疏 `tick`/`clear`） |
| `inc` | 只增（不可能造出 `grid < cap`）⇒ 无需打脏 |
| `lazy` | 惰性路径自己的子集结算写（脏标记在 `_regrow_lazy` 内部处理） |
| `full` | 全场路径（默认分支；旧行为逐位不变的落点） |
| `init` | 构造期初始化（彼时派生集尚不存在） |
| `reset` | 整体替换（快照恢复）⇒ 必须在同函数内 `rebuild_lazy` / `rebuild_sparse` |
| `n/a` | 机制范围外（Rust 直写、rd 路径 —— 已被范围锁排除） |

新增写入点若未标注 ⇒ 本测试失败，逼迫作者声明它属于哪一类（而不是靠人记得加打脏）。
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# 被扫的数组 → (允许的接收者属性名, 本类所在文件名)
ARRAY_OWNERS = {
    "_grid": ("resources", "resource_field.py"),
    "_marks": ("signals", "signal_field.py"),
    "_age": ("signals", "signal_field.py"),
}
SCANNED = (
    "world/resource_field.py",
    "world/signal_field.py",
    "simulation/sphere_engine.py",
)
TAG_RE = re.compile(r"#\s*sparse:(mark|active|inc|lazy|full|init|reset|n/a)\b")


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
    """该表达式是否指向"我们负责的字段数组"。"""
    chain = _chain(node)
    if not chain:
        return False
    name = chain[-1]
    if name not in ARRAY_OWNERS:
        return False
    recv_ok, own_file = ARRAY_OWNERS[name]
    if len(chain) >= 2 and chain[-2] == recv_ok:      # `X.resources._grid` 等
        return True
    # 字段类自己的模块内：`self._grid` / `self._marks` / `self._age`
    return fname == own_file and len(chain) == 2 and chain[0] == "self"


class _WriteSites(ast.NodeVisitor):
    """收集写入点：赋值/自增目标、`out=` 实参、`np.*.at` 首参、Rust 调用实参。"""

    def __init__(self, fname: str) -> None:
        self.fname = fname
        self.func: list[ast.AST] = []
        self.sites: list[tuple[int, str, ast.AST | None]] = []   # (lineno, 函数名, 函数节点)

    def _add(self, lineno: int) -> None:
        fn = self.func[-1] if self.func else None
        self.sites.append((lineno, getattr(fn, "name", "<module>"), fn))

    # -- 目标形（赋值 / 自增 / 注解赋值）--
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
        # `out=<字段数组>`
        for kw in node.keywords:
            if kw.arg == "out" and _is_field_array(kw.value, self.fname):
                self._add(node.lineno)
        args = list(node.args)
        # `np.add.at(arr, ...)` / `np.subtract.at(arr, ...)`
        if (isinstance(node.func, ast.Attribute) and node.func.attr == "at"
                and args and _is_field_array(args[0], self.fname)):
            self._add(node.lineno)
        # Rust 直写：`self._sim_core.*(...)` 实参里带字段数组（本仓现有调用全是就地写/只读，
        # 一律标 `n/a` —— 一旦新增"会被范围锁漏掉"的用法，标签会把它逼到台面上）
        recv = _chain(node.func) if isinstance(node.func, ast.Attribute) else None
        if recv and len(recv) >= 2 and recv[-2] == "_sim_core":
            cand = args + [kw.value for kw in node.keywords]
            if any(_is_field_array(a, self.fname) for a in cand):
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
            out.append({
                "file": rel, "line": lineno, "func": func,
                "tag": m.group(1) if m else None, "code": code.strip(),
                "func_seg": ast.get_source_segment(src, fn) if fn is not None else None,
            })
    return out


def test_scan_is_not_vacuous_and_all_write_sites_annotated():
    """① 扫描非空转；② 每个字段数组写入点都带 `# sparse:<tag>` 标签。"""
    sites = _scan()
    assert len(sites) >= 10, (
        f"扫描只找到 {len(sites)} 个写入点 ⇒ 探测器失效（文件改名/结构变化？）"
    )
    bad = [s for s in sites if s["tag"] is None]
    assert not bad, (
        "以下字段数组写入点没有 `# sparse:<tag>` 标注 ⇒ 新增写入点可能漏打脏（R219 §二-2）：\n"
        + "\n".join(f"  {s['file']}:{s['line']} [{s['func']}] {s['code']}" for s in bad)
    )


def test_mark_sites_are_wired_to_dirty_mask():
    """`mark` 标签只许出现在会**减少** `_grid` 的两处，且同函数体内确有打脏写。"""
    sites = _scan()
    marks = [s for s in sites if s["tag"] == "mark"]
    assert {s["func"] for s in marks} == {"consume", "consume_many"}, (
        f"`mark` 标注出现在意外位置：{[(s['file'], s['line'], s['func']) for s in marks]}"
    )
    src = (ROOT / "world/resource_field.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in ("consume", "consume_many"):
            seg = ast.get_source_segment(src, node) or ""
            assert "_dirty_mask" in seg, (
                f"{node.name} 标了 `mark` 却没有 `_dirty_mask` 打脏写 ⇒ 标签与现实脱节"
            )


def test_reset_sites_rebuild_derived_sets():
    """`reset` 标签处（整体替换数组）**同函数内**必须调用 `rebuild_lazy` / `rebuild_sparse`。"""
    for s in _scan():
        if s["tag"] != "reset":
            continue
        seg = s["func_seg"] or ""
        assert ("rebuild_lazy" in seg) or ("rebuild_sparse" in seg), (
            f"{s['file']}:{s['line']} 整体替换了字段数组，但所在函数 [{s['func']}] 内"
            "未见 `rebuild_lazy`/`rebuild_sparse` ⇒ 陈旧派生集会漏结算"
        )
