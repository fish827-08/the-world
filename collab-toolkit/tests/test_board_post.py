# -*- coding: utf-8 -*-
"""board_post.py 测试 —— 纯函数单测 + 本地裸仓库端到端集成（不触网）。"""
import json
import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import board_post as bp  # noqa: E402

REAL_LOCK_TOOL = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "tools", "share_lock.py"))


def git(cwd, *args, check=True):
    r = subprocess.run(["git"] + list(args), cwd=cwd, capture_output=True,
                       text=True, encoding="utf-8", errors="replace")
    if check and r.returncode != 0:
        raise AssertionError(f"git {' '.join(args)} -> {r.returncode}: {r.stderr[:200]}")
    return r


# ---------------- 纯函数 ----------------
def test_load_message_rejects_bom(tmp_path):
    p = tmp_path / "m.md"
    p.write_bytes(b"\xef\xbb\xbf### [x] hi")
    with pytest.raises(bp.PostError):
        bp.load_message(str(p))


def test_load_message_rejects_non_utf8(tmp_path):
    p = tmp_path / "m.md"
    p.write_bytes("中文".encode("gbk"))
    with pytest.raises(bp.PostError):
        bp.load_message(str(p))


def test_ensure_signature_passthrough():
    msg = "### [协作] · 2026-09-17 22:40（Asia/Shanghai）\n\n**主题**：x"
    assert bp.ensure_signature(msg, None) == msg


def test_ensure_signature_autogen():
    out = bp.ensure_signature("**主题**：只有正文", "[协作]")
    assert out.startswith("### [协作] · ")
    assert "（Asia/Shanghai）" in out.splitlines()[0]
    assert "**主题**" in out


def test_ensure_signature_requires_role():
    with pytest.raises(bp.PostError):
        bp.ensure_signature("**主题**：无署名", None)


def test_build_appended_is_prefix_extension():
    old = "# 板\n\n### [a] · 2026-09-17 10:00\n旧帖\n"
    new = bp.build_appended(old, "### [b] · 2026-09-17 11:00\n新帖")
    assert new.startswith(old.rstrip("\n"))
    assert "旧帖" in new and "新帖" in new
    assert new.endswith("\n")


# ---------------- 端到端（本地裸仓库） ----------------
@pytest.fixture()
def repo_pair(tmp_path):
    """bare 远端 + 工作克隆，含 _share/讨论板.md 初始提交。

    夹具显式关闭平台噪声（autocrlf/gpgsign/fileMode）——2026-09-18 flaky 定位：
    全局 git 配置会干扰"是否有变更/能否提交"的判定，导致同一用例时好时坏。
    """
    bare = tmp_path / "remote.git"
    work = tmp_path / "work"
    git(tmp_path, "init", "--bare", str(bare))
    git(tmp_path, "clone", str(bare), str(work))
    for k, v in (("user.email", "t@t"), ("user.name", "t"),
                 ("core.autocrlf", "false"), ("core.fileMode", "false"),
                 ("commit.gpgsign", "false"), ("core.safecrlf", "false")):
        git(work, "config", k, v)
    share = work / "_share"
    (share / ".locks").mkdir(parents=True)
    (share / "讨论板.md").write_text(
        "# 讨论板\n\n### [所有者] · 2026-09-17 20:00\n初始帖\n", encoding="utf-8")
    git(work, "add", "_share/讨论板.md")
    git(work, "commit", "-m", "init")
    git(work, "push", "origin", "HEAD:main")
    git(str(bare), "symbolic-ref", "HEAD", "refs/heads/main")  # 让后续 clone 检出 main
    return str(work)


def _post_args(work, msg_file, **over):
    args = ["--root", work, "--slot", "collab", "--task", "测试发帖",
            "--message-file", str(msg_file), "--role", "[协作]",
            "--remote", "origin", "--branch", "main",
            "--credential-helper", "", "--lock-tool", REAL_LOCK_TOOL]
    for k, v in over.items():
        flag = f"--{k.replace('_', '-')}"
        if v is True:
            args.append(flag)          # store_true 旗标
        elif v not in (False, None):
            args += [flag, str(v)]
    return args


def test_end_to_end_post(repo_pair, tmp_path):
    msg = tmp_path / "post.md"
    msg.write_text("**主题**：端到端测试帖\n\n**内容**：你好，板。", encoding="utf-8")
    rc = bp.main(_post_args(repo_pair, msg))
    assert rc == 0
    board = (tmp_path / "work" / "_share" / "讨论板.md").read_text(encoding="utf-8")
    assert "初始帖" in board and "端到端测试帖" in board
    assert "### [协作] · " in board  # 自动署名
    # 本地最后一提交是本帖，且远端对账一致（ls-remote == HEAD）
    subject = git(repo_pair, "log", "-1", "--format=%s").stdout
    assert "测试发帖" in subject
    head = git(repo_pair, "rev-parse", "HEAD").stdout.strip()
    remote = git(repo_pair, "ls-remote", "origin", "refs/heads/main").stdout.split()[0]
    assert head == remote
    # 锁已释放
    locks = list((tmp_path / "work" / "_share" / ".locks").glob("*.lock"))
    assert locks == []
    # 阶段日志：done（幂等恢复依据）
    j = json.loads((tmp_path / "work" / "_share" / ".locks" /
                    "collab.post-journal.json").read_text(encoding="utf-8"))
    assert j["stage"] == "done" and j["head"] == head


def test_take_own_lock_recovers_residual(repo_pair, tmp_path):
    """本槽位残留锁（模拟上次被 SIGTERM）→ --take-own-lock 自动挪移重取并完成。"""
    locks = tmp_path / "work" / "_share" / ".locks"
    r = subprocess.run([sys.executable, REAL_LOCK_TOOL, "acquire", "--slot", "collab",
                        "--task", "上次被杀残留", "--lock-dir", str(locks)],
                       capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    assert r.returncode == 0
    msg = tmp_path / "post.md"
    msg.write_text("**主题**：残留锁恢复\n", encoding="utf-8")

    # 不带旗标：应当被拒（rc=2），板面不变
    before = (tmp_path / "work" / "_share" / "讨论板.md").read_bytes()
    assert bp.main(_post_args(repo_pair, msg)) == 2
    assert (tmp_path / "work" / "_share" / "讨论板.md").read_bytes() == before

    # 带旗标：自动挪移重取 → 成功
    assert bp.main(_post_args(repo_pair, msg, take_own_lock=True)) == 0
    board = (tmp_path / "work" / "_share" / "讨论板.md").read_text(encoding="utf-8")
    assert "残留锁恢复" in board
    assert list(locks.glob("*.lock")) == []
    assert list((locks / "_retired").glob("collab.lock*"))  # 旧锁被挪移而非删除


def test_push_conflict_keeps_commit_and_releases_lock(repo_pair, tmp_path,
                                                      monkeypatch):
    """push 被拒绝（模拟竞态：pull 后远端又前进）-> rc=3，提交留本地，锁已放。"""
    real_run_git = bp.run_git

    def fake_run_git(root, args, check=True, credential_helper=None, **kwargs):
        # **kwargs：测试替身必须容忍工具新增参数——2026-09-18 实测教训：
        # 工具加 `retries=` 之后，替身 TypeError ⇒ 2 例确定性失败（曾被误当 flaky）
        if args and args[0] == "push":
            return subprocess.CompletedProcess(
                ["git", "push"], 1, "",
                "! [rejected] main -> main (fetch first)")
        return real_run_git(root, args, check=check,
                            credential_helper=credential_helper, **kwargs)

    monkeypatch.setattr(bp, "run_git", fake_run_git)
    msg = tmp_path / "post.md"
    msg.write_text("**主题**：冲突测试\n", encoding="utf-8")
    rc = bp.main(_post_args(repo_pair, msg))
    assert rc == 3
    # 提交留在本地（含我们的帖），锁已释放
    board = (tmp_path / "work" / "_share" / "讨论板.md").read_text(encoding="utf-8")
    assert "冲突测试" in board
    assert list((tmp_path / "work" / "_share" / ".locks").glob("*.lock")) == []


def test_dry_run_writes_nothing(repo_pair, tmp_path):
    msg = tmp_path / "post.md"
    msg.write_text("**主题**：干跑\n", encoding="utf-8")
    before = (tmp_path / "work" / "_share" / "讨论板.md").read_bytes()
    rc = bp.main(_post_args(repo_pair, msg, dry_run=True))
    # argparse: dry_run 是 flag，放最后
    after = (tmp_path / "work" / "_share" / "讨论板.md").read_bytes()
    assert rc == 0
    assert before == after


def test_rerun_after_failed_push_is_idempotent(repo_pair, tmp_path, monkeypatch):
    """push 失败后重跑同一消息：板上内容不重复（幂等补推）。"""
    msg = tmp_path / "post.md"
    msg.write_text("**主题**：幂等测试帖\n", encoding="utf-8")

    real_run_git = bp.run_git
    state = {"fail_push": True}

    def fake_run_git(root, args, check=True, credential_helper=None, **kwargs):
        if state["fail_push"] and args and args[0] == "push":
            return subprocess.CompletedProcess(["git", "push"], 1, "", "rejected")
        return real_run_git(root, args, check=check,
                            credential_helper=credential_helper, **kwargs)

    monkeypatch.setattr(bp, "run_git", fake_run_git)
    assert bp.main(_post_args(repo_pair, msg)) == 3   # 第一次：push 失败
    state["fail_push"] = False
    assert bp.main(_post_args(repo_pair, msg)) == 0   # 重跑：补推成功
    board = (tmp_path / "work" / "_share" / "讨论板.md").read_text(encoding="utf-8")
    assert board.count("幂等测试帖") == 1             # 只出现一次


def test_lock_busy_aborts(repo_pair, tmp_path):
    """他人活跃锁 -> rc=2，板面不变。"""
    r = subprocess.run(
        [sys.executable, REAL_LOCK_TOOL, "acquire", "--slot", "dev",
         "--task", "占锁", "--lock-dir",
         str(tmp_path / "work" / "_share" / ".locks")],
        capture_output=True, text=True,
        encoding="utf-8", errors="replace")
    assert r.returncode == 0
    msg = tmp_path / "post.md"
    msg.write_text("**主题**：应被拒绝\n", encoding="utf-8")
    before = (tmp_path / "work" / "_share" / "讨论板.md").read_bytes()
    rc = bp.main(_post_args(repo_pair, msg))
    assert rc == 2
    assert (tmp_path / "work" / "_share" / "讨论板.md").read_bytes() == before
