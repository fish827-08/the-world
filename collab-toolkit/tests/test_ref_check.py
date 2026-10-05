# -*- coding: utf-8 -*-
"""ref_check.py 单元测试（FILE-REORG 零断链核验工具，提案 §五）。

两层：纯函数（正则/判定）+ 临时 git 仓集成（snapshot/diff/check 真跑）。
纪律：只用 shutil.move 挪文件、零删除（F-R10）；绝不碰主工作树。
"""
import json
import os
import shutil
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import ref_check as rc  # noqa: E402


def _git(root, args):
    r = subprocess.run(["git"] + args, cwd=str(root), capture_output=True,
                       text=True, encoding="utf-8", errors="replace")
    assert r.returncode == 0, "git %s 失败: %s" % (" ".join(args), r.stderr)
    return r


def _write(root, rel, text):
    p = os.path.join(str(root), rel.replace("/", os.sep))
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


@pytest.fixture()
def repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, ["init", "-b", "main"])
    _write(root, "docs/a.md", "# A\n正文\n")
    _write(root, "docs/b.md", "见 [A](docs/a.md) 与 `tools/t1.py`\n")
    _write(root, "tools/t1.py", "# 引 docs/a.md\nX = 1\n")
    _write(root, "README.md", "入口：docs/a.md、docs/b.md\n")
    _write(root, "_share/实况.md", "历史帖引用 docs/b.md（不回头改）\n")
    _git(root, ["add", "-A"])
    _git(root, ["-c", "user.email=t@the-world.local", "-c", "user.name=test",
                "commit", "-m", "init"])
    return str(root)


# ---------------- 纯函数层 ----------------
def test_regex_extracts_known_prefixes():
    line = "见 docs/x/y.md 和 tools/z.py，另有 http://e.com/f.md 不算"
    refs = [m.group(0) for m in rc.REF_RE.finditer(line)]
    assert any(r == "docs/x/y.md" for r in refs)
    assert any(r == "tools/z.py" for r in refs)
    assert not any("http" in r for r in refs)


def test_junk_hints_filtered():
    assert not rc.is_probable_ref("docs/YYYY-MM.md")
    assert not rc.is_probable_ref("docs/….md")
    assert rc.is_probable_ref("docs/机制-署名闸门-v1-20261004.md")


def test_in_hist_scope():
    assert rc.in_hist("_share/讨论板.md")
    assert rc.in_hist("_archive/归档记录.md")
    assert not rc.in_hist("docs/README.md")


# ---------------- snapshot ----------------
def test_snapshot_counts(repo):
    out = os.path.join(repo, "_snap.json")
    assert rc.main(["--root", repo, "snapshot", "--out", out]) == 0
    data = json.load(open(out, encoding="utf-8"))
    assert data["resolved"] >= 6  # 4 个 a.md 引用 + b.md 里 tools/t1.py + README 引 b
    assert any(e["src"] == "docs/b.md" and e["ref"] == "docs/a.md"
               and e["resolved"] for e in data["entries"])


# ---------------- diff：迁移三分类 ----------------
def _move_and_commit(root, old, new):
    os.makedirs(os.path.dirname(os.path.join(root, new)), exist_ok=True)
    shutil.move(os.path.join(root, old.replace("/", os.sep)),
                os.path.join(root, new.replace("/", os.sep)))
    _git(root, ["add", "-A"])
    _git(root, ["-c", "user.email=t@the-world.local", "-c", "user.name=test",
                "commit", "-m", f"move {old} -> {new}"])


def test_diff_zero_broken_when_all_updated(repo):
    snap = os.path.join(repo, "_snap.json")
    rc.main(["--root", repo, "snapshot", "--out", snap])
    _move_and_commit(repo, "docs/a.md", "docs/子目录/a.md")
    # 活跃区随迁移刷新引用（README/docs/b.md/tools/t1.py），历史区 _share 不动
    _write(repo, "README.md", "入口：docs/子目录/a.md、docs/b.md\n")
    _write(repo, "docs/b.md", "见 [A](docs/子目录/a.md) 与 `tools/t1.py`\n")
    _write(repo, "tools/t1.py", "# 引 docs/子目录/a.md\nX = 1\n")
    # _share/实况.md 引 docs/b.md（未迁移）——原地仍解析，不构成断链
    assert rc.main(["--root", repo, "diff", "--base", snap, "--from", "HEAD~1"]) == 0


def test_diff_pending_update_flagged(repo):
    snap = os.path.join(repo, "_snap.json")
    rc.main(["--root", repo, "snapshot", "--out", snap])
    _move_and_commit(repo, "docs/a.md", "docs/子目录/a.md")
    _write(repo, "README.md", "入口：docs/a.md、docs/b.md\n")  # 活跃区忘改
    r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "ref_check.py"),
                        "--root", repo, "diff", "--base", snap, "--from", "HEAD~1"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert r.returncode == 1 and "待更新" in r.stdout


def test_diff_hist_scope_not_counted(repo):
    _write(repo, "_share/历史引用.md", "旧帖引 tools/t1.py\n")
    _git(repo, ["add", "-A"])
    _git(repo, ["-c", "user.email=t@the-world.local", "-c", "user.name=test",
                "commit", "-m", "add hist ref"])
    snap = os.path.join(repo, "_snap.json")
    rc.main(["--root", repo, "snapshot", "--out", snap])
    _move_and_commit(repo, "tools/t1.py", "tools/工具线/t1.py")
    # 活跃区唯一引用者 docs/b.md 随迁移刷新；历史区 _share/历史引用.md 有意不改
    _write(repo, "docs/b.md", "见 [A](docs/a.md) 与 `tools/工具线/t1.py`\n")
    r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "ref_check.py"),
                        "--root", repo, "diff", "--base", snap, "--from", "HEAD~1"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "历史区留痕 1" in r.stdout and "随迁移更新 1" in r.stdout


def test_diff_target_gone_no_rename_is_broken(repo):
    snap = os.path.join(repo, "_snap.json")
    rc.main(["--root", repo, "snapshot", "--out", snap])
    shutil.move(os.path.join(repo, "docs", "a.md"),
                os.path.join(os.path.dirname(repo), "a_outside.md"))  # 挪出仓 ⇒ 无 rename 映射
    _git(repo, ["add", "-A"])
    _git(repo, ["-c", "user.email=t@the-world.local", "-c", "user.name=test",
                "commit", "-m", "drop a.md"])
    r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "ref_check.py"),
                        "--root", repo, "diff", "--base", snap, "--from", "HEAD~1"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert r.returncode == 1 and "[断链]" in r.stdout


# ---------------- check ----------------
def test_check_separates_active_and_hist(repo):
    _write(repo, "docs/b.md", "坏引用 docs/不存在.md；历史区另说\n")
    _write(repo, "_share/旧帖.md", "坏引用 tools/没这件.py\n")
    r = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "ref_check.py"),
                        "--root", repo, "check"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert "活跃区不可解析 1" in r.stdout and "历史区不可解析 1" in r.stdout
    strict = subprocess.run([sys.executable, os.path.join(ROOT, "tools", "ref_check.py"),
                             "--root", repo, "check", "--strict"],
                            capture_output=True, text=True, encoding="utf-8")
    assert strict.returncode == 1
