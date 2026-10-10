# -*- coding: utf-8 -*-
"""worktree.py 单元测试 —— 隔离工作树三命令（R359 B1）。

测试分两层：
  1) **纯函数层**（check_card / plan_paths / resolve_python / plan_setup）—— 不碰 git；
  2) **集成层**（setup/enter/clean 三命令）—— 在临时 git 仓库里真跑。
🔴 纪律：绝不碰主工作树（不 `git worktree add` 到 the-world），只在 tmp_path 里造仓库。
"""
import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import worktree as wt  # noqa: E402


# ---------------- 夹具：临时 git 仓库 ----------------
def _git(root, args, check=True):
    r = subprocess.run(["git"] + args, cwd=str(root), capture_output=True,
                       text=True, encoding="utf-8", errors="replace")
    if check:
        assert r.returncode == 0, "git %s 失败: %s" % (" ".join(args), r.stderr)
    return r


@pytest.fixture()
def repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, ["init", "-b", "main"])
    with open(root / "README.md", "w", encoding="utf-8") as f:
        f.write("# 测试仓\n")
    _git(root, ["add", "README.md"])
    _git(root, ["-c", "user.email=t@the-world.local",
                "-c", "user.name=test", "commit", "-m", "init"])
    return str(root)


# ---------------- 纯函数层 ----------------
def test_check_card_rejects_injection():
    for bad in ("R1; rm -rf /", "../evil", "", "-x", "R1 "):
        with pytest.raises(wt.WorktreeError):
            wt.check_card(bad)


def test_plan_paths_mapping():
    p = wt.plan_paths("/root", "R359-B1")
    assert p["dir"] == os.path.join("/root", ".worktrees", "R359-B1")
    assert p["branch"] == "task/R359-B1"


def test_resolve_python_points_at_main_venv():
    """适配点 2：worktree 内无 .venv ⇒ 一律指向主仓 .venv（不猜其他路径）。"""
    exe, exists, why = wt.resolve_python(wt.DEFAULT_ROOT)
    assert exe.endswith(os.path.join(".venv", "Scripts", "python.exe"))
    assert why  # 提示文案必须给得出（fail-loud 的一部分）


def test_resolve_python_override_wins():
    exe, exists, _ = wt.resolve_python(wt.DEFAULT_ROOT, override="/usr/bin/python3")
    assert exe == "/usr/bin/python3" and exists is False  # 不存在要如实报，不静默回落


def test_plan_setup_plans_new_branch(repo):
    plan = wt.plan_setup(repo, "R359-T")
    assert plan["wt_exists"] is False and plan["branch_exists"] is False
    assert plan["cmd"] == ["worktree", "add", "-b", "task/R359-T",
                           os.path.join(repo, ".worktrees", "R359-T")]


def test_plan_setup_is_idempotent_after_setup(repo):
    wt.main(["setup", "R359-T", "--root", repo])
    plan = wt.plan_setup(repo, "R359-T")
    assert plan["wt_exists"] is True and plan["cmd"] is None  # 复用，不重建


# ---------------- 集成层：三命令 ----------------
def test_setup_creates_isolated_worktree(repo, capsys):
    assert wt.main(["setup", "R359-T", "--root", repo]) == 0
    wt_dir = os.path.join(repo, ".worktrees", "R359-T")
    # 🔴  Linked worktree 的 .git 在 Windows 是**文件**、Linux 是**目录**，
    # 两种都要认（worktree.py 的 plan_setup 里的判定同款）。
    assert os.path.isfile(os.path.join(wt_dir, ".git")) or \
        os.path.isdir(os.path.join(wt_dir, ".git"))
    out = capsys.readouterr().out
    assert "task/R359-T" in out
    # 提示里必须出现「不实现 prune」的纪律说明（AGENT.md:493 禁 worktree prune）
    assert "prune" in out


def test_setup_runs_twice_without_error(repo):
    assert wt.main(["setup", "R359-T", "--root", repo]) == 0
    assert wt.main(["setup", "R359-T", "--root", repo]) == 0  # 幂等


def test_enter_prints_path(repo, capsys):
    wt.main(["setup", "R359-T", "--root", repo])
    assert wt.main(["enter", "R359-T", "--root", repo]) == 0
    assert os.path.join(repo, ".worktrees", "R359-T") in capsys.readouterr().out


def test_enter_without_setup_fails(repo):
    assert wt.main(["enter", "R359-NOPE", "--root", repo]) != 0


def test_main_worktree_checkout_is_refused(repo):
    """🔴 隔离生效的判据：主工作区 checkout 该分支时 git 必须拒绝。"""
    wt.main(["setup", "R359-T", "--root", repo])
    r = _git(repo, ["checkout", "task/R359-T"], check=False)
    assert r.returncode != 0
    assert "already" in (r.stderr + r.stdout).lower()


def test_worktrees_dir_is_gitignored():
    """适配点 4：`.worktrees/` 必须在 .gitignore 里，否则污染 git status（R343）。"""
    gi = os.path.join(wt.DEFAULT_ROOT, ".gitignore")
    assert os.path.isfile(gi)
    with open(gi, encoding="utf-8") as f:
        assert any(l.strip().rstrip("/") == ".worktrees" for l in f), \
            ".gitignore 缺 `.worktrees/`"


def test_clean_keeps_branch_and_commits(repo, capsys):
    wt.main(["setup", "R359-T", "--root", repo])
    wt_dir = os.path.join(repo, ".worktrees", "R359-T")
    with open(os.path.join(wt_dir, "note.md"), "w", encoding="utf-8") as f:
        f.write("x\n")
    _git(wt_dir, ["add", "note.md"])
    _git(wt_dir, ["-c", "user.email=t@the-world.local",
                  "-c", "user.name=test", "commit", "-m", "隔离区提交"])
    assert wt.main(["clean", "R359-T", "--root", repo]) == 0
    assert not os.path.isdir(wt_dir)          # 目录没了
    r = _git(repo, ["log", "-1", "--format=%s", "task/R359-T"])
    assert "隔离区提交" in r.stdout           # 提交仍躺在分支上（F-R10 不删数据）


def test_clean_refuses_dirty_worktree(repo, capsys):
    wt.main(["setup", "R359-T", "--root", repo])
    with open(os.path.join(repo, ".worktrees", "R359-T", "dirty.txt"),
              "w", encoding="utf-8") as f:
        f.write("x\n")
    assert wt.main(["clean", "R359-T", "--root", repo]) == 2
    assert os.path.isdir(os.path.join(repo, ".worktrees", "R359-T"))


# ---------------- 署名纪律（R370 / R227，ONBOARD-V2 ③） ----------------
def test_plan_identity_pair_rules():
    assert wt.plan_identity("板桥", "banqiao@the-world.local")["action"] == "set"
    assert wt.plan_identity(None, None)["action"] == "ask"
    for half in (("板桥", None), (None, "banqiao@the-world.local")):
        assert wt.plan_identity(*half)["action"] == "bad"  # 只给一半 ⇒ 不猜邮箱


def test_setup_without_name_reminds_loudly(repo, capsys):
    """不给 --name ⇒ 必须打印 R370 提醒 + 当前署名（fail-loud，不静默）。"""
    assert wt.main(["setup", "R359-T", "--root", repo]) == 0
    out = capsys.readouterr().out
    assert "R370" in out and "user.name" in out and "当前署名" in out


def test_setup_with_name_writes_identity(repo, capsys):
    wt_dir = os.path.join(repo, ".worktrees", "R359-T")
    rc = wt.main(["setup", "R359-T", "--root", repo,
                  "--name", "板桥", "--email", "banqiao@the-world.local"])
    assert rc == 0
    assert "署名已生效" in capsys.readouterr().out
    name, mail = wt.current_identity(wt_dir)
    assert name == "板桥" and mail == "banqiao@the-world.local"


def test_setup_rejects_half_identity(repo, capsys):
    assert wt.main(["setup", "R359-T", "--root", repo, "--name", "板桥"]) == 1
    assert "成对" in capsys.readouterr().err


# ---------------- --role：署名文件注入（卡 WORKTREE-SIGN / R371①） ----------------
def test_plan_identity_role_wins():
    assert wt.plan_identity(None, None, "collab")["action"] == "role"
    assert wt.plan_identity("板桥", "b@the-world.local", "collab")["action"] == "role"
    assert wt.plan_identity("板桥", "b@the-world.local")["action"] == "set"
    assert wt.plan_identity("板桥", None, None)["action"] == "bad"
    # R393①：--owner 走花名册反查；与 --role 同时给 ⇒ 拒绝（不猜人要哪个）
    assert wt.plan_identity(None, None, None, "板桥")["action"] == "owner"
    assert wt.plan_identity(None, None, "collab", "板桥")["action"] == "bad"


def test_setup_with_role_writes_tree_config_not_shared_config(repo, capsys):
    """--role ⇒ 写本树署名文件 + **本树独占** config.worktree；主树身份一点不动。

    🔴 R393① 之后判据升级：不是"谁都不写"，而是"只写自己那棵树"——
    隔离证明 = 主工作树 `git config user.name` 前后一致（R371 竞写病灶不得复发）。
    """
    wt_dir = os.path.join(repo, ".worktrees", "R359-T")
    before = wt.current_identity(repo)          # 主树身份（可能来自全局 config）
    assert wt.main(["setup", "R359-T", "--root", repo, "--role", "collab"]) == 0
    out = capsys.readouterr().out
    assert "本树署名文件已生成" in out and "板桥" in out
    import git_id
    assert git_id.load_sign_file(wt_dir) == ("板桥", "banqiao@the-world.local")
    assert wt.current_identity(wt_dir) == ("板桥", "banqiao@the-world.local")
    assert wt.current_identity(repo) == before, "主树身份被覆写 ⇒ R371 竞写复发"


def test_setup_owner_by_hua_ming(repo, capsys):
    """R393①：`--owner <花名>` 按花名册字典反查邮箱并写本树署名（免逐命令 -c）。"""
    wt_dir = os.path.join(repo, ".worktrees", "R393")
    assert wt.main(["setup", "R393", "--root", repo, "--owner", "板桥"]) == 0
    out = capsys.readouterr().out
    assert "banqiao@the-world.local" in out and "免逐命令 -c" in out
    assert wt.current_identity(wt_dir) == ("板桥", "banqiao@the-world.local")
    import git_id
    assert git_id.load_sign_file(wt_dir) == ("板桥", "banqiao@the-world.local")


def test_setup_owner_accepts_board_signature_and_combos(repo, capsys):
    """负责列里的写法（`[协作]` / `PI·fish(1)` / 带 emoji 的 `轻舟 ⚡`）都认。"""
    for card, owner, want in (("K1", "[协作]", "板桥"),
                              ("K2", "PI·fish(1)", "fish(1)"),
                              ("K3", "轻舟 ⚡", "轻舟")):
        assert wt.main(["setup", card, "--root", repo, "--owner", owner,
                        "--no-tree-config"]) == 0
        capsys.readouterr()
        import git_id
        assert git_id.load_sign_file(os.path.join(repo, ".worktrees", card))[0] == want


def test_setup_owner_unknown_fails_loud(repo, capsys):
    assert wt.main(["setup", "R9", "--root", repo, "--owner", "查无此人"]) == 1
    assert "花名册查不到" in capsys.readouterr().err


def test_setup_owner_and_role_are_mutually_exclusive(repo, capsys):
    assert wt.main(["setup", "R9", "--root", repo, "--owner", "板桥",
                    "--role", "collab"]) == 1
    assert "二选一" in capsys.readouterr().err


def test_setup_no_tree_config_keeps_shared_config(repo, capsys):
    """--no-tree-config ⇒ 只写署名文件，不触发 config.worktree / 不碰 extensions 开关。"""
    before = wt.current_identity(repo)
    wt_dir = os.path.join(repo, ".worktrees", "R394")
    assert wt.main(["setup", "R394", "--root", repo, "--owner", "板桥",
                    "--no-tree-config"]) == 0
    assert "--no-tree-config" in capsys.readouterr().out
    assert wt.current_identity(wt_dir) == before        # 本树也没被写


def test_apply_tree_identity_reports_switch_flip(repo):
    """开关前后状态必须如实回报（实测：git 不会自动开 extensions.worktreeConfig）。"""
    assert wt.main(["setup", "R395", "--root", repo, "--owner", "板桥",
                    "--no-tree-config"]) == 0
    wt_dir = os.path.join(repo, ".worktrees", "R395")
    res = wt.apply_tree_identity(wt_dir, "板桥", "banqiao@the-world.local")
    assert res["ok"] is True, res
    assert res["scope_after"] == "true"
    assert wt.current_identity(wt_dir) == ("板桥", "banqiao@the-world.local")


def test_setup_with_undecided_role_fails_loud(repo, capsys):
    assert wt.main(["setup", "R359-T", "--root", repo, "--role", "web"]) == 1
    assert "花名未定" in capsys.readouterr().err


def test_setup_rejects_unknown_role(repo, capsys):
    assert wt.main(["setup", "R359-T", "--root", repo, "--role", "nope"]) == 1
