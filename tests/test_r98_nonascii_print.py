"""R98 纪律：**非 ASCII print 全域守卫**（F-R10 / F-R14 / F-R15 同族）。

背景：中文 Windows 默认 GBK 控制台下，`print()` 里出现 **GBK 无法编码**的字符
（`↻` U+21BB、emoji ✅❌🔴、`⇒` U+21D2 …）会抛 `UnicodeEncodeError` ⇒ 脚本 **rc=1**
⇒ 批次状态/退出码被污染（**外围把真结果搅坏**，而数据其实无损）。
已发生实例：`a4_verify_capacity.py` 续跑路径 `print("  ↻ …")`（F-R15，内评代修）。

本测试把"扫描"变成**不可回退的纪律**：受控目录下不允许存在
「print 含 GBK 不可编码字符」**且**「未做 UTF-8 兜底」的脚本。

🔴 保护判据 = **AST 检调用点**（R216 §三② 裁定），不是"源码含标记串"：
原实现里守卫只因**定义了**标记常量就自判"已保护"⇒ 抓不到自己（自指盲区）。
回归测试见 `test_criterion_is_ast_call_not_substring` 与 `test_scanner_itself_is_protected`。
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from experiments.scan_nonascii_print import (      # noqa: E402  (conftest 已把 ROOT 入 sys.path)
    DEFAULT_DIRS as SC_DEFAULT_DIRS,
    scan_file as sc_scan_file,
    source_is_protected as sc_source_is_protected,
)


def test_no_unprotected_non_gbk_print_in_controlled_tree():
    offenders = []
    for d in SC_DEFAULT_DIRS:
        base = ROOT / d
        if not base.is_dir():
            continue
        for p in sorted(base.rglob("*.py")):
            if "__pycache__" in p.parts:
                continue
            src = p.read_text(encoding="utf-8")
            if sc_source_is_protected(src):     # 真调用过 reconfigure ⇒ 运行期安全
                continue
            bad = sc_scan_file(p)
            if bad:
                offenders.append((p.relative_to(ROOT).as_posix(),
                                  sorted({c for _, c, _ in bad})))
    assert not offenders, (
        "以下脚本在 GBK 控制台下会 rc=1 假失败（print 含 GBK 不可编码字符且无兜底）：\n"
        + "\n".join(f"  {f}: {chars}" for f, chars in offenders)
        + "\n修法：入口加 sys.stdout.reconfigure(encoding='utf-8', errors='replace')"
    )


def test_scanner_detects_gbk_unencodable_and_ignores_chinese(tmp_path):
    """扫描器自身的行为：中文（GBK 可编码）不得报，emoji/箭头必须报。"""
    good = tmp_path / "good.py"
    good.write_text('print("中文没问题 ⇒ 这个不行")\n', encoding="utf-8")
    assert any(c == "⇒" for _, c, _ in sc_scan_file(good))
    assert not any(c == "中" for _, c, _ in sc_scan_file(good))

    ok = tmp_path / "ok.py"
    ok.write_text('print("纯中文与 ASCII 都安全")\n', encoding="utf-8")
    assert sc_scan_file(ok) == []


def test_criterion_is_ast_call_not_substring(tmp_path):
    """R216 §三②：判据 = **真的调用** reconfigure —— 只"提到"标记串不算受保护。

    防的正是自指盲区：旧判据下，把标记串写进常量/注释/报错文案就足以自判"已保护"
    （守卫本体就是这么漏掉自己的）。
    """
    mere = tmp_path / "mere.py"
    mere.write_text(
        'MARK = \'reconfigure(encoding="utf-8"\'\n'
        '# 或写进文案："入口加 sys.stdout.reconfigure(encoding=\'utf-8\')"\n',
        encoding="utf-8")
    assert not sc_source_is_protected(mere.read_text(encoding="utf-8")), (
        "仅含标记串的源码不得判为受保护（这正是自指盲区的成因）")

    real = tmp_path / "real.py"
    real.write_text(
        'import sys\n'
        'sys.stdout.reconfigure(encoding="utf-8", errors="replace")\n',
        encoding="utf-8")
    assert sc_source_is_protected(real.read_text(encoding="utf-8"))

    # 无 encoding 关键字的 reconfigure（如 only errors=）不算编码兜底
    half = tmp_path / "half.py"
    half.write_text('import sys\nsys.stdout.reconfigure(errors="replace")\n',
                    encoding="utf-8")
    assert not sc_source_is_protected(half.read_text(encoding="utf-8"))


def test_scanner_itself_is_protected():
    """🔴 自指盲区回归：守卫本体必须**真的**调用 reconfigure（R216 §三①）。

    旧实现下本测试必红：`scan_nonascii_print.py` 只定义 `PROTECT_MARK` 常量、
    从不调用 ⇒ 却因"含标记串"被自己判为受保护。
    """
    src = (ROOT / "experiments" / "scan_nonascii_print.py").read_text(encoding="utf-8")
    assert sc_source_is_protected(src), "守卫本体未真的调用 reconfigure（自指盲区复发）"
