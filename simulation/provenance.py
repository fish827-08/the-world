"""provenance（出处）硬校验 + RNG 抽取计数器。

任务：**D-19**（`_share/讨论板.md` 17:01 任务清单第 1 项）
依据：**R31④**（provenance 必须可解析，不得静默写 `unknown`）；**V-1 O-6**（`rng_draws` 计数器）

背景（为什么要有这个文件）
--------------------------
- R19 上批的 manifest 记了 `git_commit=815d900`，而该 commit **在仓库中不存在**（悬空）
  ⇒ 数据无法回溯到代码。教训：provenance 字段**静默失败**（写 `unknown`/`None`）等于没有。
- 因此本模块要求：**commit 不可解析 ⇒ 抛错**，而不是写 `unknown`。
- `rng_draws` 让"两条路径/两次重跑是否消费了同一条随机流"可被**机械校验**
  （不必靠肉眼看轨迹）。

用法
----
```python
from simulation.provenance import CountingRNG, collect, validate

engine.rng = CountingRNG(np.random.default_rng(seed))   # 随机流不变，只加计数
prov = collect(config)          # 收集 provenance
validate(prov)                  # 硬校验：不过就抛 RuntimeError
prov["rng_draws"] = engine.rng.draws
```
"""
from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class CountingRNG:
    """包住 `np.random.Generator`：统计抽取次数，**随机流本身逐位不变**。

    显式包装了引擎实际用到的方法；其余方法/属性经 `__getattr__` 直传给被包对象，
    因此不会改变任何既有行为（包括 `bit_generator`，快照/恢复依赖它）。
    """

    __slots__ = ("_gen", "draws")

    def __init__(self, gen) -> None:
        self._gen = gen
        self.draws = 0

    # 未显式包装的一律直传（bit_generator / 其他分布方法等）
    def __getattr__(self, name: str):
        return getattr(self._gen, name)

    def _bump(self, value):
        self.draws += 1
        return value

    def random(self, *a, **k):
        return self._bump(self._gen.random(*a, **k))

    def integers(self, *a, **k):
        return self._bump(self._gen.integers(*a, **k))

    def normal(self, *a, **k):
        return self._bump(self._gen.normal(*a, **k))

    def uniform(self, *a, **k):
        return self._bump(self._gen.uniform(*a, **k))

    def standard_normal(self, *a, **k):
        return self._bump(self._gen.standard_normal(*a, **k))


def git_commit(repo_root: str | os.PathLike | None = None) -> str:
    """返回 HEAD commit；**不可解析时抛 RuntimeError，不返回 'unknown'**。"""
    root = str(repo_root or ROOT)
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception as exc:  # 非 git 目录 / 无 commit / git 不可用
        raise RuntimeError(
            f"D-19 provenance 硬校验失败：无法解析 git commit（repo={root}）：{exc}"
        ) from exc
    sha = out.strip()
    if len(sha) != 40:
        raise RuntimeError(f"D-19 provenance 硬校验失败：commit 形态异常 {sha!r}")
    return sha


def git_dirty(repo_root: str | os.PathLike | None = None) -> bool:
    """工作区是否有未提交改动（无法判定时保守返回 True）。"""
    root = str(repo_root or ROOT)
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return True
    return bool(out.strip())


def sim_core_sha256() -> str | None:
    """Rust 扩展二进制指纹；未安装时返回 None（由 `validate` 决定是否视为失败）。"""
    try:
        import sim_core  # noqa: F401
    except Exception:
        return None
    path = getattr(__import__("sim_core"), "__file__", None)
    if not path or not os.path.exists(path):
        return None
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()[:32]


# 决定模拟语义的代码目录（改动其中任一 .py 都可能改变结果）
CODE_TREE_DIRS: tuple[str, ...] = ("simulation", "observatory", "world", "core", "experiments")


def _hash_py_files(root: Path, files: list[Path]) -> str:
    """对给定 `.py` 文件集合做**确定性哈希**（按相对路径排序；`路径\0内容\0` 累加）。

    单一实现：`code_tree_sha256`（合并全树）与 `code_subtree_sha256s`（逐子树）共用它，
    避免两份算法漂移（本项目已付过"两处同病"的学费，见 F-R23）。
    """
    h = hashlib.sha256()
    for p in sorted(files, key=lambda q: q.relative_to(root).as_posix()):
        h.update(p.relative_to(root).as_posix().encode("utf-8"))
        h.update(b"\0")
        h.update(p.read_bytes())
        h.update(b"\0")
    return h.hexdigest()[:32]


def _py_files_of(root: Path, d: str) -> list[Path]:
    base = root / d
    if not base.is_dir():
        return []
    return [q for q in base.rglob("*.py") if "__pycache__" not in q.parts]


def code_subtree_sha256s(
    repo_root: str | os.PathLike | None = None,
    dirs: tuple[str, ...] = CODE_TREE_DIRS,
) -> dict[str, str]:
    """**逐子树**代码指纹（R122 移交余项 #6「manifest 引擎子树哈希」）。

    为什么在 `code_tree_sha256` 之外还要它：
    合并哈希只能回答"**代码变了吗**"，回答不了"**哪一层变了**"——
    例如批跑期间若只有 `experiments/`（脚本层）变动、而 `simulation/observatory/world/core`
    （**决定模拟语义**的层）逐字节未变，则判决应是"**引擎未变、非混版本**"。
    逐子树哈希让这一判决**一眼可读**，无需再逐 commit 手 diff（D-24 做过一次，代价高）。

    ⚠️ 口径：各子树**独立**哈希（**不**跨子树合并）⇒ 便于比较；目录缺失 ⇒ 空串。
    """
    root = Path(repo_root or ROOT)
    out: dict[str, str] = {}
    for d in dirs:
        files = _py_files_of(root, d)
        # 目录缺失 / 无 .py ⇒ **空串**（显式"缺席"；不用"空集的哈希"——那看起来像真实指纹）
        out[d] = _hash_py_files(root, files) if files else ""
    return out


def changed_subtrees(prev: dict, cur: dict) -> list[str]:
    """比较两次 `code_subtree_sha256s`（或两批 manifest 的 `code_subtrees`），列出**变化的子树**。

    用法（批内自检）：`changed_subtrees(a["code_subtrees"], b["code_subtrees"])`
    ⇒ 空列表 = 两批**同一份引擎代码**（可直接宣称"非混版本"）。
    """
    a = (prev or {}).get("code_subtrees", prev) if isinstance(prev, dict) else {}
    b = (cur or {}).get("code_subtrees", cur) if isinstance(cur, dict) else {}
    keys = sorted(set(a) | set(b))
    return [k for k in keys if a.get(k) != b.get(k)]


def code_tree_sha256(
    repo_root: str | os.PathLike | None = None,
    dirs: tuple[str, ...] = CODE_TREE_DIRS,
) -> str:
    """（原 docstring 见 git 历史；**语义未改动**）合并全树指纹 —— 复用 `_hash_py_files`。"""
    root = Path(repo_root or ROOT)
    files: list[Path] = []
    for d in dirs:
        files.extend(_py_files_of(root, d))
    return _hash_py_files(root, files)


def collect(config, repo_root: str | os.PathLike | None = None,
            rng_draws: int | None = None, python_info: bool = True) -> dict:
    """收集 provenance 字典（不抛错；校验交给 `validate`）。"""
    import sys

    prov: dict = {
        "git_commit": git_commit(repo_root),
        "git_dirty": git_dirty(repo_root),
        "sim_core_sha256": sim_core_sha256(),
        # D-19+：代码树内容指纹（与浮动 HEAD 解耦；Python 路径下 sim_core_sha256
        # 恒 None，此前**没有任何代码指纹字段**）
        "code_tree_sha256": code_tree_sha256(repo_root),
        # R122 #6：逐子树指纹 ⇒ 可直接回答"哪一层代码变了"（合并哈希回答不了）
        "code_subtrees": code_subtree_sha256s(repo_root),
        "config_fingerprint": config.fingerprint(),
    }
    if rng_draws is not None:
        prov["rng_draws"] = int(rng_draws)
    if python_info:
        prov["python"] = sys.version.split()[0]
    return prov


def validate(prov: dict, require_sim_core: bool = False) -> None:
    """硬校验：任一必需字段缺失/为占位值 ⇒ 抛 RuntimeError。

    Args:
        prov: `collect()` 的产物
        require_sim_core: True 时要求 Rust 扩展指纹存在（走 Rust 路径的批次应开）
    """
    forbidden = {"unknown", "", None, "None"}
    for key in ("git_commit", "config_fingerprint", "code_tree_sha256"):
        if key not in prov or prov[key] in forbidden:
            raise RuntimeError(
                f"D-19 provenance 硬校验失败：{key} 缺失或为占位值（{prov.get(key)!r}）"
            )
    sha = str(prov["git_commit"])
    if len(sha) != 40:
        raise RuntimeError(f"D-19 provenance 硬校验失败：commit 形态异常 {sha!r}")
    if require_sim_core and not prov.get("sim_core_sha256"):
        raise RuntimeError(
            "D-19 provenance 硬校验失败：本批次走 Rust 路径但未记录 sim_core 指纹"
        )
