# -*- coding: utf-8 -*-
"""git_id.py 单元测试 —— 署名注入与提交前校验（卡 WORKTREE-SIGN / R371①）。

两层：
  1) 纯函数层（env_pairs / shell_lines / 署名文件往返 / 注册表映射）；
  2) **行为层**：在临时 git 仓库里真提交，验证 env 注入确实改变作者位
     （这是"免逐命令 -c"成立与否的唯一判据）。
🔴 全程只碰 tmp_path，绝不碰 the-world 主工作树。
"""
import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import git_id  # noqa: E402
import onboard  # noqa: E402


def _git(root, args, env=None):
    return subprocess.run(["git"] + list(args), cwd=str(root),
                          capture_output=True, text=True, encoding="utf-8",
                          errors="replace", env=env)


@pytest.fixture(autouse=True)
def _clean_git_config_env():
    """隔离 GIT_CONFIG_* 注入位。

    🔴 为什么必须有：`git_id.apply_to_process()` 会改**进程级** os.environ，而别的
    测试文件（test_board_post）调过它 ⇒ 全量连跑时这些键泄漏到本文件，"裸提交"也带上
    别人注入的身份，行为层测试随执行顺序变红（R116 要求连跑全绿 ⇒ 必须顺序无关）。
    """
    saved = {k: v for k, v in os.environ.items() if k.startswith("GIT_CONFIG_")}
    for k in saved:
        del os.environ[k]
    yield
    for k in [k for k in os.environ if k.startswith("GIT_CONFIG_")]:
        del os.environ[k]
    os.environ.update(saved)


@pytest.fixture()
def repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, ["init", "-b", "main"])
    (root / "README.md").write_text("# t\n", encoding="utf-8")
    _git(root, ["add", "README.md"])
    _git(root, ["commit", "-m", "init"],
         env={**os.environ, **git_id.env_pairs("初始人", "chushi@the-world.local")})
    return str(root)


# ---------------- 纯函数：env 注入 ----------------
def test_env_pairs_shape():
    e = git_id.env_pairs("板桥", "banqiao@the-world.local")
    assert e["GIT_CONFIG_COUNT"] == "2"
    assert e["GIT_CONFIG_KEY_0"] == "user.name" and e["GIT_CONFIG_VALUE_0"] == "板桥"
    assert e["GIT_CONFIG_KEY_1"] == "user.email" and e["GIT_CONFIG_VALUE_1"] == "banqiao@the-world.local"


def test_sign_file_roundtrip(tmp_path):
    root = str(tmp_path / "wt")
    os.makedirs(root)
    p = git_id.write_sign_file(root, "板桥", "banqiao@the-world.local")
    assert os.path.basename(p) == git_id.SIGN_FILENAME
    assert git_id.load_sign_file(root) == ("板桥", "banqiao@the-world.local")
    assert git_id.load_sign_file(str(tmp_path / "nothing")) == (None, None)


# ---------------- 行为层：注入确实改作者位 ----------------
def test_env_injection_changes_commit_author(repo):
    """🔴 本卡核心判据：不设 git config、不敲 -c，仅靠 env ⇒ 作者位 = 注入的花名。"""
    env = {**os.environ, **git_id.env_pairs("板桥", "banqiao@the-world.local")}
    with open(os.path.join(repo, "a.txt"), "w", encoding="utf-8") as f:
        f.write("x\n")
    assert _git(repo, ["add", "a.txt"], env=env).returncode == 0
    r = _git(repo, ["commit", "-m", "注入提交"], env=env)
    assert r.returncode == 0, r.stderr
    assert git_id.author_of(repo) == ("板桥", "banqiao@the-world.local")
    # 🔴 注入只活在进程里，**一个字节都不落配置文件**（这是"零共享写"的判据；
    #    对照实测：本机全局身份 = 小鱼 <…@user.noreply.gitee.com>，真人账号 ——
    #    忘了设身份就会提交到 fish 头上，故必须注入而非依赖 config）。
    local_name = _git(repo, ["config", "--local", "--get", "user.name"]).stdout.strip()
    assert local_name == ""
    assert git_id.config_identity(repo)[0] != "板桥"


def test_injection_overrides_local_config(repo):
    """R371 竞写场景的正面防御：即便本地 config 已被别的会话写成"砚"，
    带 env 的提交作者位仍是本人 —— 优先级 env > 命令行 -c > 本地 > 全局。"""
    _git(repo, ["config", "user.name", "砚"])
    _git(repo, ["config", "user.email", "yan@the-world.local"])
    env = {**os.environ, **git_id.env_pairs("镜", "jing@the-world.local")}
    with open(os.path.join(repo, "b.txt"), "w", encoding="utf-8") as f:
        f.write("y\n")
    _git(repo, ["add", "b.txt"], env=env)
    r = _git(repo, ["commit", "-m", "竞写场景提交"], env=env)
    assert r.returncode == 0, r.stderr
    assert git_id.author_of(repo) == ("镜", "jing@the-world.local")
    # 而"裸提交"（不带 env）就会挂到被覆写的身份上 ⇒ 这正是必须注入的理由
    with open(os.path.join(repo, "c.txt"), "w", encoding="utf-8") as f:
        f.write("z\n")
    _git(repo, ["add", "c.txt"])
    _git(repo, ["commit", "-m", "裸提交"])
    assert git_id.author_of(repo) == ("砚", "yan@the-world.local")


def test_verify_identity_ok_and_drift(repo):
    findings = git_id.verify_identity(repo, ("初始人", "chushi@the-world.local"))
    assert all(f["level"] == "ok" for f in findings) and len(findings) == 2
    bad = git_id.verify_identity(repo, ("板桥", "banqiao@the-world.local"))
    assert any(f["level"] == "error" for f in bad)  # 署名漂移必须 fail-loud


def test_config_identity_honours_expected_identity(repo):
    """给了应然署名 ⇒ 按 env 读"真生效值"，不被共享 config 误导。"""
    n, e = git_id.config_identity(repo, "镜", "jing@the-world.local")
    assert (n, e) == ("镜", "jing@the-world.local")


# ---------------- 应然署名来自注册表 ----------------
def test_role_identity_from_registry():
    assert git_id.role_identity("collab") == ("板桥", "banqiao@the-world.local")
    assert git_id.role_identity("dev")[0] == "轻舟"


def test_role_identity_fails_loud_on_undecided_name():
    """[联网] 花名"待取" ⇒ 不能凭空造邮箱（R249 精神：检测不成立就别宣称自动）。"""
    with pytest.raises(git_id.GitIdError) as ei:
        git_id.role_identity("web")
    assert "花名未定" in str(ei.value)


def test_role_identity_unknown_role():
    with pytest.raises(git_id.GitIdError):
        git_id.role_identity("nope")


def test_registry_emails_are_conformant():
    """ROLE_REGISTRY 邮箱字段：要么空（花名未定），要么 <拼音>@the-world.local。"""
    for key, reg in onboard.ROLE_REGISTRY.items():
        email = reg.get("邮箱")
        assert email is not None, f"{key} 缺 邮箱 字段"
        if reg["花名"] == "待取":
            assert email == ""
            continue
        assert git_id.EMAIL_RE.match(email), f"{key} 邮箱不合规: {email!r}"
        assert email.endswith("@the-world.local")


def test_pi_email_is_reserved_for_pi():
    """R370：pi@the-world.local 仅限 PI 本人会话 ⇒ 只有 pi 键能用它。"""
    users = [k for k, v in onboard.ROLE_REGISTRY.items()
             if v.get("邮箱") == "pi@the-world.local"]
    assert users == ["pi"]


# ---------------- CLI ----------------
def test_cli_verify_and_show(repo, capsys):
    assert git_id.main(["verify", "--root", repo,
                        "--name", "初始人", "--email", "chushi@the-world.local"]) == 0
    assert git_id.main(["verify", "--root", repo, "--name", "板桥",
                        "--email", "banqiao@the-world.local"]) == 2
    # 无从判断应然署名 ⇒ 必须拒绝放行（不静默通过）
    assert git_id.main(["verify", "--root", str(os.path.dirname(repo))]) == 1
    capsys.readouterr()
    assert git_id.main(["show", "--root", repo, "--name", "初始人",
                        "--email", "chushi@the-world.local"]) == 0
    out = capsys.readouterr().out
    # 断言用 ASCII 锚点：pytest 的 fd 捕获与工具内 stdout.reconfigure(utf-8) 叠在一起
    # 会让中文在捕获层乱码（harness 现象，非用户可见问题）
    assert "chushi@the-world.local" in out and "HEAD" in out and ".qoder-sign.env" in out


def test_cli_verify_passes_from_sign_file(repo, tmp_path):
    """署名文件存在 ⇒ verify 无需 --name/--email（工具线的"免逐命令 -c"闭环）。"""
    git_id.write_sign_file(repo, "初始人", "chushi@the-world.local")
    assert git_id.main(["verify", "--root", repo]) == 0


def test_cli_env_prints_sourceable_lines(capsys):
    assert git_id.main(["env", "--role", "collab"]) == 0
    out = capsys.readouterr().out
    assert 'export GIT_CONFIG_VALUE_0="板桥"' in out


# ---------------- owner：花名册字典反查（卡 SIGN-HOOK / R393①） ----------------
def test_owner_identity_accepts_all_spellings():
    assert git_id.owner_identity("板桥") == ("板桥", "banqiao@the-world.local")
    assert git_id.owner_identity("[协作]") == ("板桥", "banqiao@the-world.local")
    assert git_id.owner_identity("collab") == ("板桥", "banqiao@the-world.local")
    assert git_id.owner_identity("轻舟 ⚡") == ("轻舟", "qingzhou@the-world.local")
    assert git_id.owner_identity("PI·fish(1)") == ("fish(1)", "pi@the-world.local")


def test_owner_identity_fail_loud():
    for bad in ("查无此人", "", None):
        with pytest.raises(git_id.GitIdError) as e:
            git_id.owner_identity(bad)
        assert e.value.code == 1
    with pytest.raises(git_id.GitIdError):        # [联网] 花名"待取" ⇒ 不猜邮箱
        git_id.owner_identity("web")


def test_roster_identities_covers_registry():
    import onboard
    table = git_id.roster_identities()
    for key, v in onboard.ROLE_REGISTRY.items():
        assert table[key.lower()] == (key, v["花名"], v["邮箱"])
        if v["花名"]:
            assert table[git_id._strip_decor(v["花名"]).lower()] == (key, v["花名"], v["邮箱"])


def test_cli_owner_prints_pair(capsys):
    assert git_id.main(["owner", "砚"]) == 0
    assert capsys.readouterr().out.strip() == "砚\tyan@the-world.local"
    assert git_id.main(["owner", "幽灵"]) == 1


def test_owner_identity_strips_fullwidth_annotation():
    """澜舟 10-05 治理观察②：`砚（窗宽口径 R225 锁）`/`澜舟（实施）` 剥全角括注后查表。"""
    assert git_id.owner_identity("澜舟（实施）") == ("澜舟", "lanzhou@the-world.local")
    assert git_id.owner_identity("砚（窗宽口径 R225 锁）") == ("砚", "yan@the-world.local")
    assert git_id.owner_identity("板桥 （工具线）") == ("板桥", "banqiao@the-world.local")
    # 半角括号是花名真身（fish(1) 在册）⇒ 不剥：PI·fish(1) 仍解析为 fish(1)
    assert git_id.owner_identity("PI·fish(1)") == ("fish(1)", "pi@the-world.local")
    # 括注剥离先于分词：组合+括注仍取首个可查花名
    assert git_id.owner_identity("澜舟/砚（口径）") == ("澜舟", "lanzhou@the-world.local")


def test_owner_identity_annotation_only_fails_loud():
    """剥完括注什么都不剩 ⇒ fail-loud（不猜花名）。"""
    for bad in ("（实施）", "（只有备注）"):
        with pytest.raises(git_id.GitIdError):
            git_id.owner_identity(bad)
