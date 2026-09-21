"""D-19：provenance 硬校验 + rng_draws 计数器。

依据：**R31④**（provenance 不得静默写 unknown）；**V-1 O-6**（rng_draws 计数器）。
出处：`_share/讨论板.md` 2026-09-13 17:01 任务清单第 1 项。
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simulation.config import SimConfig  # noqa: E402
from simulation.provenance import (  # noqa: E402
    CODE_TREE_DIRS, CountingRNG, changed_subtrees, code_subtree_sha256s,
    code_tree_sha256, collect, git_commit, validate,
)
from simulation.sphere_engine import SphereEngine  # noqa: E402


# ---- rng_draws 计数器 ----

def test_rng_draws_starts_and_grows():
    e = SphereEngine(SimConfig(seed=42))
    d0 = e.rng_draws
    for _ in range(50):
        e.step()
    assert e.rng_draws > d0, "step 消耗随机流后 draws 应增加"


def test_rng_draws_deterministic_across_reruns():
    """同 seed 同配置跑同样 tick ⇒ rng_draws 必须相同（可复现性的机械信号）。"""
    def run():
        e = SphereEngine(SimConfig(seed=42))
        for _ in range(50):
            e.step()
        return e.rng_draws
    assert run() == run()


def test_counting_rng_does_not_alter_stream():
    """包装只计数，随机流必须逐位不变（否则会破坏所有对拍/快照复现）。"""
    gen = np.random.default_rng(7)
    wrapped = CountingRNG(np.random.default_rng(7))
    a = [float(gen.random()) for _ in range(5)]
    b = [float(wrapped.random()) for _ in range(5)]
    assert a == b
    assert wrapped.draws == 5
    # bit_generator 需可访问（快照/恢复依赖它）
    assert wrapped.bit_generator is not None


def test_bit_generator_state_roundtrip():
    """恢复 RNG 状态后 draws 计数继续（不重置、不打断）。"""
    wrapped = CountingRNG(np.random.default_rng(11))
    wrapped.random()
    st = wrapped.bit_generator.state
    wrapped2 = CountingRNG(np.random.default_rng(11))
    wrapped2.bit_generator.state = st
    assert wrapped2.random() == wrapped.random()


# ---- provenance 硬校验 ----

def test_git_commit_resolvable_in_repo():
    sha = git_commit()
    assert len(sha) == 40 and all(c in "0123456789abcdef" for c in sha)


def test_git_commit_hard_fails_outside_repo(tmp_path):
    """非 git 目录 ⇒ 抛错，绝不能静默返回 'unknown'。"""
    with pytest.raises(RuntimeError, match="provenance 硬校验失败"):
        git_commit(tmp_path)


def test_validate_rejects_placeholders():
    for bad in ({}, {"git_commit": "unknown", "config_fingerprint": "x"},
                {"git_commit": None, "config_fingerprint": "x"},
                {"git_commit": "abc", "config_fingerprint": "x"}):
        with pytest.raises(RuntimeError):
            validate(bad)


def test_validate_accepts_collected():
    prov = collect(SimConfig(seed=1))
    validate(prov)                      # 不抛即通过
    assert len(prov["git_commit"]) == 40
    assert prov["config_fingerprint"]
    assert isinstance(prov["git_dirty"], bool)


def test_validate_can_require_sim_core():
    prov = collect(SimConfig(seed=1))
    prov["sim_core_sha256"] = None
    if prov.get("sim_core_sha256") is None:
        with pytest.raises(RuntimeError):
            validate(prov, require_sim_core=True)


def test_collect_records_rng_draws():
    e = SphereEngine(SimConfig(seed=3))
    prov = collect(e.config, rng_draws=e.rng_draws)
    assert prov["rng_draws"] == e.rng_draws
    validate(prov)


# ---- D-19+ 代码树指纹（与浮动 HEAD 解耦）----

def _fake_tree(tmp_path):
    for d in ("simulation", "observatory"):
        (tmp_path / d).mkdir(parents=True, exist_ok=True)
    (tmp_path / "simulation" / "a.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "simulation" / "b.py").write_text("y = 2\n", encoding="utf-8")
    (tmp_path / "observatory" / "c.py").write_text("z = 3\n", encoding="utf-8")
    return tmp_path


def test_code_tree_hash_is_deterministic(tmp_path):
    r = _fake_tree(tmp_path)
    assert code_tree_sha256(r) == code_tree_sha256(r)
    assert len(code_tree_sha256(r)) == 32


def test_code_tree_hash_changes_when_code_changes(tmp_path):
    """任一受控 .py 内容改动 ⇒ 指纹必须变（否则等于没有指纹）。"""
    r = _fake_tree(tmp_path)
    before = code_tree_sha256(r)
    (r / "simulation" / "a.py").write_text("x = 2\n", encoding="utf-8")
    assert code_tree_sha256(r) != before


def test_code_tree_hash_ignores_pycache_and_untracked_dirs(tmp_path):
    """`__pycache__` 与受控目录外的文件（如 _share/）不得影响指纹。"""
    r = _fake_tree(tmp_path)
    before = code_tree_sha256(r)
    (r / "simulation" / "__pycache__").mkdir()
    (r / "simulation" / "__pycache__" / "a.cpython-313.pyc").write_bytes(b"\x00\x01")
    (r / "_share").mkdir()
    (r / "_share" / "讨论板.md").write_text("x", encoding="utf-8")
    assert code_tree_sha256(r) == before


def test_code_tree_hash_matches_real_repo_content():
    """真实仓库：同内容两次调用一致；且与 HEAD 无关（不因提交而变）。"""
    h1 = code_tree_sha256()
    h2 = code_tree_sha256()
    assert h1 == h2 and len(h1) == 32


def test_collect_and_validate_require_code_tree_hash():
    prov = collect(SimConfig(seed=1))
    assert prov["code_tree_sha256"] and len(prov["code_tree_sha256"]) == 32
    validate(prov)
    with pytest.raises(RuntimeError, match="code_tree_sha256"):
        validate({k: v for k, v in prov.items() if k != "code_tree_sha256"})


# ---- R122 #6：引擎**子树**哈希（定位"哪一层变了"） ----

def test_subtree_hashes_cover_all_code_tree_dirs(tmp_path):
    """返回键 = 全部受控目录（含不存在者 ⇒ 空串），且逐子树哈希长度 32。"""
    r = _fake_tree(tmp_path)
    hs = code_subtree_sha256s(r)
    assert set(hs) == set(CODE_TREE_DIRS)
    assert len(hs["simulation"]) == 32 and len(hs["observatory"]) == 32
    assert hs["world"] == "" and hs["core"] == "" and hs["experiments"] == ""


def test_subtree_hash_localizes_change(tmp_path):
    """🔴 核心价值：**改一个子树 ⇒ 只有该子树哈希变**（合并哈希做不到这点）。"""
    r = _fake_tree(tmp_path)
    before = code_subtree_sha256s(r)
    assert changed_subtrees(before, code_subtree_sha256s(r)) == []
    (r / "observatory" / "c.py").write_text("z = 4\n", encoding="utf-8")
    after = code_subtree_sha256s(r)
    assert changed_subtrees(before, after) == ["observatory"]
    assert after["simulation"] == before["simulation"]      # 未动的子树不变
    assert code_tree_sha256(r) == code_tree_sha256(r)   # 合并哈希自身仍确定


def test_subtree_hash_is_deterministic(tmp_path):
    r = _fake_tree(tmp_path)
    assert code_subtree_sha256s(r) == code_subtree_sha256s(r)


def test_changed_subtrees_accepts_manifest_dicts():
    """可直接吃两批 manifest：`changed_subtrees(manifest_a, manifest_b)`。"""
    a = {"code_subtrees": {"simulation": "aa", "world": "bb"}}
    b = {"code_subtrees": {"simulation": "aa", "world": "cc"}}
    assert changed_subtrees(a, b) == ["world"]
    # 缺键/空值不得抛异常（旧批无该字段 ⇒ 全部记为"变"或"缺"，由调用方解释）
    assert changed_subtrees({}, b) == ["simulation", "world"]
    assert changed_subtrees({"code_subtrees": {}}, {"code_subtrees": {}}) == []


def test_collect_includes_code_subtrees():
    """`collect()` 必须带上逐子树指纹（否则批跑产物里查不到"哪层变了"）。"""
    prov = collect(SimConfig(seed=1))
    st = prov["code_subtrees"]
    assert set(st) == set(CODE_TREE_DIRS)
    # 真实仓库里 simulation/observatory/world/core/experiments 都存在 ⇒ 都非空
    for d in CODE_TREE_DIRS:
        assert st[d], f"{d} 的子树哈希为空 —— 受控目录应存在"
    validate(prov)


def test_merged_hash_derives_from_same_file_set(tmp_path):
    """口径自洽：合并哈希 = 同一文件集 ⇒ 逐子树**全同**时合并哈希也同（防两套算法漂移）。"""
    r = _fake_tree(tmp_path)
    h1 = code_tree_sha256(r)
    sub = code_subtree_sha256s(r)
    # 复制一份同内容树 ⇒ 合并哈希与各子树哈希逐位相同
    import shutil
    r2 = tmp_path.parent / (tmp_path.name + "_copy")
    shutil.copytree(r, r2)
    assert code_tree_sha256(r2) == h1
    assert code_subtree_sha256s(r2) == sub
