"""pytest 根路径配置。

让 tests/ 下的用例能 import 项目的顶层包（core / simulation / world）。

[test-hygiene] 临时根守卫（B2.0 建立，B4.0 修正两种成因）
--------------------------------------------------------
`tmp_path` / `tmpdir` 默认落在 `tempfile.gettempdir()`。若该目录在**仓库内**，
`git rev-parse` 会向上找到仓库的 `.git` ⇒ `test_git_commit_hard_fails_outside_repo`
期望的 `RuntimeError` 永不抛（假红），且工作树会被 `pytest-of-*` / `tmp*` 产物污染。

**两种成因必须分开处理**（B4.0 裁定）：

1. **真·配置错误**：`TEMP`/`TMP`/`TMPDIR` 显式指向仓库内 ⇒ **fatal**，请改配置。
2. **受限环境回退**：`TEMP`/`TMP` 已设但**所有候选都不可写**（Windows ACL 坏 / 沙箱），
   `gettempdir()` 只能回退到 CWD ⇒ **warning + 继续**，并**自动**启用会话结束清理
   （不需要再手工设 `WB_ALLOW_TMP_IN_REPO`；该变量仍保留为显式逃生门）。

判据用**实际写入探测**而不是 `os.access`：Windows 的 ACL 下 `os.access(p, W_OK)`
并不可靠（会对不可写目录返回 True），只有真建一个临时文件才能定论。
"""
import os
import sys
import tempfile
import warnings
from pathlib import Path

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

_TMP_ENV_KEYS = ("TEMP", "TMP", "TMPDIR")

# [W0-T / 2026-10-18] 旧线用例已整体 `git mv` 到 `tests/archive/`（59 文件 / 624 例），
# 归档目录**不参与收集**。归档≠删除：`git show <commit>:tests/test_xxx.py` 可查历史。
# pytest.ini 里的 `legacy` marker **暂留**（回退或局部跑归档件时仍可用）。
collect_ignore_glob = ["archive/*"]


def _resolve(p: str) -> Path:
    try:
        return Path(p).resolve()
    except Exception:  # noqa: BLE001 - 解析失败按"不可用"处理
        return Path(p)


def _is_in_repo(p: Path, root: Path) -> bool:
    return p == root or root in p.parents


def _writable(p: str) -> bool:
    """实际写一个临时文件（`os.access` 在 Windows ACL 下不可靠）。"""
    try:
        os.makedirs(p, exist_ok=True)
        fd, name = tempfile.mkstemp(dir=p, prefix="_wb_probe_")
        os.close(fd)
        os.remove(name)
        return True
    except Exception:  # noqa: BLE001
        return False


def pytest_configure(config):
    root = Path(__file__).resolve().parents[1]
    tmp = _resolve(tempfile.gettempdir())
    if not _is_in_repo(tmp, root):
        config._wb_tmp_fallback = False
        return

    declared = [os.environ[k] for k in _TMP_ENV_KEYS if os.environ.get(k)]
    declared_in_repo = [p for p in declared if _is_in_repo(_resolve(p), root)]
    if declared_in_repo:
        raise SystemExit(
            "[test-hygiene] TEMP/TMP 显式指向仓库内 ⇒ tmp_path 会让 git 测试假红并污染"
            f"工作树：{[str(p) for p in declared_in_repo]}（仓库={root}）。"
            "这是配置错误，请把 TEMP/TMP 改到仓库外，或用 --basetemp 指到仓库外。")

    # 第二支：候选全不可写 / 根本没有 TEMP ⇒ tempfile 回退 CWD
    # （受限环境的必然结果，不是配置错误）
    if not declared or not any(_writable(p) for p in declared):
        warnings.warn(
            "[test-hygiene] TEMP/TMP 候选均不可写或未设置 ⇒ tempfile 回退到仓库根 "
            f"({tmp})；本会话将自动清理 basetemp。若换到可写环境，本条自动消失。",
            RuntimeWarning, stacklevel=1)
        config._wb_tmp_fallback = True
        return

    # 其它（无环境变量 / 有个别可写却没被选中）⇒ 保守 fatal
    raise SystemExit(
        f"[test-hygiene] 临时根在仓库内（{tmp}）且未命中'候选全不可写'分支；"
        "请设 TEMP/TMP 或用 --basetemp 指到仓库外。")


def pytest_sessionfinish(session, exitstatus):  # noqa: ARG001 - pytest hook 签名
    """受限环境回退时，会话结束删掉自己的 basetemp（防累积成 17 个目录）。"""
    cfg = session.config
    if not (getattr(cfg, "_wb_tmp_fallback", False)
            or os.environ.get("WB_ALLOW_TMP_IN_REPO") == "1"):
        return
    root = Path(__file__).resolve().parents[1]
    tmp = _resolve(tempfile.gettempdir())
    if not _is_in_repo(tmp, root):
        return
    # ⚠️ 不传 `--basetemp` 时 `option.basetemp` 是 None；实际路径在 tmp_path_factory 里
    base = None
    try:
        base = session.config._tmp_path_factory.getbasetemp()
    except Exception:  # noqa: BLE001 - 未创建则回退到 option
        base = getattr(cfg.option, "basetemp", None)
    if not base:
        return
    p = Path(base)
    try:
        if p.exists() and _is_in_repo(p.resolve(), root):
            import shutil
            shutil.rmtree(p, ignore_errors=True)
            # basetemp 的父目录 `pytest-of-<user>` 若已空也删（否则仓库根仍留一个壳）
            parent = p.parent
            if parent.name.startswith("pytest-of-") and parent.exists():
                try:
                    if not any(parent.iterdir()):
                        parent.rmdir()
                except Exception:  # noqa: BLE001 - 空壳删不掉不算错
                    pass
    except Exception as exc:  # noqa: BLE001 - 删不掉要打印，不许静默
        print(f"[test-hygiene] basetemp 清理失败（{p}）：{exc}", file=sys.stderr)
