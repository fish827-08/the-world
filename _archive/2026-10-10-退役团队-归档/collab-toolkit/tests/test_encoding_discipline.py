# -*- coding: utf-8 -*-
"""编码纪律回归检查（ONBOARD-V2 收尾 F2，2026-10-04）。

Windows 控制台默认 GBK ⇒ `subprocess.run(..., text=True)` 不写 `encoding`
时用 GBK 解码子进程输出，中文一多就抛 `UnicodeDecodeError`。异常发生在
subprocess 的 `_readerthread` 里，**结果常常照对**，只在 pytest 里冒成一条
warning（真跑时可能被吞掉）⇒ 这类"看起来没事"的噪声最容易演变成吞错。

本测试不做行为断言，做**静态断言**：collab-toolkit 的 tools/ 与 tests/ 里
任何抓输出的 subprocess 调用都必须显式 `encoding=`。
"""
import ast
import glob
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
TOOLKIT = os.path.dirname(_HERE)
SUBPROCESS_FUNCS = {"run", "Popen", "check_output", "check_call", "call"}
CAPTURE_KWARGS = {"capture_output", "stdout", "stderr", "text",
                  "universal_newlines", "pipes"}


def _offending(path: str) -> list:
    tree = ast.parse(open(path, encoding="utf-8").read(), filename=path)
    bad = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else \
            (func.id if isinstance(func, ast.Name) else "")
        if name not in SUBPROCESS_FUNCS:
            continue
        kw = {k.arg for k in node.keywords if k.arg}
        if not kw & CAPTURE_KWARGS:
            continue
        if "encoding" not in kw:
            bad.append(f"{os.path.relpath(path, TOOLKIT)}:{node.lineno}")
    return bad


def test_no_subprocess_without_explicit_encoding():
    files = sorted(glob.glob(os.path.join(TOOLKIT, "tools", "*.py")) +
                   glob.glob(os.path.join(TOOLKIT, "tests", "*.py")))
    assert files, "没扫到任何文件 —— 路径探测本身坏了"
    offenders = [item for p in files for item in _offending(p)]
    assert not offenders, \
        "subprocess 抓输出必须显式 encoding='utf-8'（Windows GBK 解码噪声）: " + \
        ", ".join(offenders)
