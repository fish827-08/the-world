# -*- coding: utf-8 -*-
"""sign_hook.py / install_git_hook.py 单元测试（卡 SIGN-HOOK / R393②③）。

三层：
  1) **纯函数层**：card_of / owners_from_card_file / judge（拦不拦、降不降级，全在这）；
  2) **安装器层**：marker 幂等、他人钩子拒绝覆盖、remove 只动自己的；
  3) **行为层**：临时仓 + `.worktrees/<卡号>` + 真钩子真提交 ⇒
     错名**提交不出去**、本人能提交、主树只警告、PI 白名单放行。
🔴 全程只碰 tmp_path，绝不碰 the-world 主工作树，也绝不往真仓 `.git/hooks` 写东西。
"""
import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import git_id  # noqa: E402
import install_git_hook as ih  # noqa: E402
import sign_hook as sh  # noqa: E402

BANQIAO = ("板桥", "banqiao@the-world.local")
YAN = ("砚", "yan@the-world.local")
TOOLS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tools"))


def _git(root, args, env=None):
    return subprocess.run(["git"] + list(args), cwd=str(root), capture_output=True,
                          text=True, encoding="utf-8", errors="replace", env=env)


@pytest.fixture(autouse=True)
def _clean_git_config_env():
    """隔离 GIT_CONFIG_* 注入位（与 test_git_id 同因：全量连跑必须顺序无关，R116）。"""
    saved = {k: v for k, v in os.environ.items() if k.startswith("GIT_CONFIG_")}
    for k in saved:
        del os.environ[k]
    yield
    for k in [k for k in os.environ if k.startswith("GIT_CONFIG_")]:
        del os.environ[k]
    os.environ.update(saved)


CARD_TABLE = """# 任务卡

| 卡号 | 状态 | 负责 | 目标 | 输入 | 约束 | 验收 | 建卡 | 更新 | 结果 |
|---|---|---|---|---|---|---|---|---|---|
| C1 | claimed | 板桥 | 冒烟 |  |  |  |  |  |  |
| C2 | claimed | 轻舟/澜舟 | 双人卡 |  |  |  |  |  |  |
| C4 | claimed | 板桥+砚 | 加号双人卡 |  |  |  |  |  |  |
| C3 | pending | 不存在的人 | 坏负责列 |  |  |  |  |  |  |
"""


@pytest.fixture()
def repo(tmp_path):
    """临时仓 + 卡表 + 一条 `.worktrees/C1` 工作树 + 已装钩子。"""
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, ["init", "-b", "main"])
    (root / "_share").mkdir()
    (root / "_share" / "任务卡.md").write_text(CARD_TABLE, encoding="utf-8")
    (root / "README.md").write_text("# t\n", encoding="utf-8")
    _git(root, ["add", "README.md", "_share"])
    _git(root, ["commit", "-m", "init"],
         env={**os.environ, **git_id.env_pairs(*BANQIAO)})
    wt = str(tmp_path / "repo" / ".worktrees" / "C1")
    r = _git(root, ["worktree", "add", wt, "-b", "task/C1"])
    assert r.returncode == 0, r.stderr
    # 卡表在 C1 检出里也要有（新分支从 main 检出 ⇒ 已含 _share/任务卡.md）
    ins = subprocess.run([sys.executable, os.path.join(TOOLS, "install_git_hook.py"),
                          "install", "--root", str(root)],
                         capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert ins.returncode == 0, ins.stdout + ins.stderr
    return {"root": str(root), "wt": wt, "tmp": str(tmp_path)}


def _commit_in(path, ident, msg="演示提交", extra_env=None):
    """在 path 里做一次真提交（不写 config，全靠 -c ⇒ 模拟"忘设身份"的各种情形）。"""
    f = os.path.join(path, msg + ".md")
    with open(f, "w", encoding="utf-8") as fh:
        fh.write("x\n")
    _git(path, ["add", "-A"])
    env = dict(os.environ)
    env.update(git_id.env_pairs(*ident))
    if extra_env:
        env.update(extra_env)
    return _git(path, ["commit", "-m", msg], env=env)


# ---------------- 1) 纯函数：卡号识别 ----------------
def test_card_of_prefers_worktrees_path():
    assert sh.card_of("/x/the-world/.worktrees/SIGN-HOOK", "task/SIGN-HOOK") == "SIGN-HOOK"
    assert sh.card_of(r"C:\\repo\\.worktrees\\C1", "task/C1") == "C1"


def test_card_of_falls_back_to_branch():
    assert sh.card_of("/x/elsewhere", "task/R393") == "R393"
    assert sh.card_of("/x/elsewhere", "main") is None


def test_card_of_main_tree_is_never_hard():
    """🔴 R393③：主树即使检出 task/<卡> 分支也不给卡号 ⇒ 不硬拦（维护/合并归 PI）。"""
    assert sh.card_of("/x/the-world", "task/C1", main_tree=True) is None


def test_card_of_rejects_odd_card_names():
    assert sh.card_of("/x/.worktrees/", "task/") is None
    assert sh.card_of("/x/.worktrees/../evil", "main") is None


# ---------------- 1b) 纯函数：卡表「负责」列 ----------------
def test_owners_from_card_file(tmp_path):
    d = tmp_path / "_share"
    d.mkdir()
    (d / "任务卡.md").write_text(CARD_TABLE, encoding="utf-8")
    assert sh.owners_from_card_file(str(tmp_path), "C1") == ["板桥"]
    assert sh.owners_from_card_file(str(tmp_path), "C2") == ["轻舟", "澜舟"]
    assert sh.owners_from_card_file(str(tmp_path), "C4") == ["板桥", "砚"]  # 加号分隔（FILE-REORG 卡形态）
    assert sh.owners_from_card_file(str(tmp_path), "NOPE") == []
    assert sh.owners_from_card_file(str(tmp_path / "ghost"), "C1") == []


# ---------------- 1c) 纯函数：裁决表 ----------------
EXP1 = {"card_pairs": [BANQIAO], "file": None, "notes": [], "raw": "板桥"}
EXP_FILE_ONLY = {"card_pairs": [], "file": BANQIAO, "notes": [], "raw": ""}
ROSTER = set(sh.roster_names())


def _lvl(name, email, card, exp=EXP1):
    return sh.judge(name, email, card, exp, ROSTER)["level"]


def test_judge_blocks_wrong_owner_in_card_tree():
    assert _lvl(*YAN, "C1") == "block"
    v = sh.judge("砚", "yan@the-world.local", "C1", EXP1, ROSTER)
    joined = "\n".join(f["msg"] for f in v["findings"]) + "\n".join(v["fix"])
    assert "应然" in joined and "修法" in joined            # 打印修法（R393②）
    assert "task_card.py claim" in joined                   # 卡转人也有路


def test_judge_allows_card_owner():
    assert _lvl(*BANQIAO, "C1") == "ok"


def test_judge_blocks_non_roster_author():
    assert _lvl("小鱼", "14550830+little-fishy@user.noreply.gitee.com", "C1") == "block"


def test_judge_pi_whitelist_passes_even_in_card_tree():
    assert _lvl("fish(1)", "pi@the-world.local", "C1") == "ok"
    assert _lvl("PI", "pi@the-world.local", "C1") == "ok"


def test_judge_main_tree_downgrades_to_warn():
    """🔴 R393③：无卡 ⇒ 同样的错名只警告（不挡 fish/PI 维护操作）。"""
    assert _lvl(*YAN, None, EXP_FILE_ONLY) == "warn"
    assert _lvl("小鱼", "little-fishy@user.noreply.gitee.com", None,
                EXP_FILE_ONLY) == "warn"
    assert _lvl(*BANQIAO, None, EXP_FILE_ONLY) == "ok"


def test_judge_no_expected_value_is_warn_only():
    empty = {"card_pairs": [], "file": None, "notes": [], "raw": ""}
    assert _lvl(*YAN, "C9", empty) == "warn"       # 在册花名 + 无应然 ⇒ 放行但记账


def test_judge_dual_lock_drift_warns():
    """卡负责人 = 板桥、署名文件 = 砚 ⇒ 放行按卡，另发"双锁漂移"警告。"""
    exp = {"card_pairs": [BANQIAO], "file": YAN, "notes": [], "raw": "板桥"}
    v = sh.judge("板桥", "banqiao@the-world.local", "C1", exp, ROSTER)
    assert v["level"] == "warn"
    assert any("双锁漂移" in f["msg"] for f in v["findings"])


def test_judge_missing_author_blocks_in_card_tree():
    exp = {"card_pairs": [BANQIAO], "file": None, "notes": [], "raw": ""}
    assert sh.judge("", "", "C1", exp, ROSTER)["level"] == "block"


# ---------------- 1d) 发帖提交升格（SIGN-BOARD-ENFORCE：暂存区新增讨论板帖 ⇒ 无卡也硬拦） ----------------
QINGZHOU = ("轻舟", "qingzhou@the-world.local")
LANZHOU = ("澜舟", "lanzhou@the-world.local")
EMPTY_EXP = {"card_pairs": [], "file": None, "notes": [], "raw": ""}
POST_BQ = [("协作", BANQIAO)]
POST_QZ = [("轻舟", QINGZHOU)]
POST_JUNK = [("审核队列表", None)]


def _post_lvl(name, email, decls, exp=EMPTY_EXP, card=None):
    return sh.judge(name, email, card, exp, ROSTER, post_decls=decls)["level"]


def test_judge_post_impersonation_blocks_even_without_card():
    """连犯三证同型（a5981a0：板桥帖署=澜舟）：主树无卡也拦。"""
    assert _post_lvl(*BANQIAO, POST_QZ) == "block"
    assert _post_lvl(*BANQIAO, POST_QZ, EXP_FILE_ONLY) == "block"
    v = sh.judge("板桥", "banqiao@the-world.local", None, EXP_FILE_ONLY, ROSTER,
                 post_decls=POST_QZ)
    joined = "\n".join(f["msg"] for f in v["findings"])
    assert "冒充在册他人" in joined


def test_judge_post_author_matches_header_ok():
    assert _post_lvl(*BANQIAO, POST_BQ, EXP_FILE_ONLY) == "ok"


def test_judge_post_non_roster_author_blocks_without_card():
    """2d8c677 同型（真人 gitee 账号发帖）：无卡也拦——不静默放行。"""
    assert _post_lvl("小鱼", "14550830+little-fishy@user.noreply.gitee.com",
                     POST_BQ) == "block"


def test_judge_post_unknown_header_warns_only():
    """头花名册查不到 ⇒ 头文本放行（作者位才是权属依据），只警告。"""
    assert _post_lvl(*BANQIAO, POST_JUNK, EXP_FILE_ONLY) == "warn"


def test_judge_post_missing_author_blocks_without_card():
    assert _post_lvl("", "", POST_BQ) == "block"


def test_judge_pi_whitelist_still_passes_on_posts():
    assert _post_lvl("fish(1)", "pi@the-world.local", POST_QZ) == "ok"


def test_judge_non_post_main_tree_unchanged():
    """非发帖提交（post_decls=None）主树口径不变：R393③ 警告级，不挡 fish/PI 维护。"""
    assert _lvl(*YAN, None, EXP_FILE_ONLY) == "warn"
    assert sh.judge("小鱼", "14550830+little-fishy@user.noreply.gitee.com",
                    None, EMPTY_EXP, ROSTER)["level"] == "warn"


# ---------------- 2) 安装器 ----------------
def test_install_status_remove_cycle(tmp_path):
    root = tmp_path / "r"
    root.mkdir()
    _git(root, ["init", "-b", "main"])
    path = ih.hook_path(str(root))
    assert ih.classify_existing(path)["kind"] == "absent"
    assert ih.main(["install", "--root", str(root)]) == 0
    st = ih.classify_existing(path)
    assert st["kind"] == "ours" and st["ver"] == ih.MARKER_VER and st["up_to_date"]
    assert ih.main(["install", "--root", str(root)]) == 0          # 幂等：已装同版 ⇒ 直接返回 0
    assert ih.classify_existing(path)["up_to_date"]
    assert ih.main(["status", "--root", str(root)]) == 0
    assert ih.main(["remove", "--root", str(root)]) == 0
    assert not os.path.isfile(path)
    assert [f for f in os.listdir(os.path.dirname(path))
            if f.startswith(ih.HOOK_NAME + ".removed-")]        # F-R10 只移不删


def test_install_refuses_to_clobber_foreign_hook(tmp_path):
    root = tmp_path / "r"
    root.mkdir()
    _git(root, ["init", "-b", "main"])
    path = ih.hook_path(str(root))
    with open(path, "w", encoding="utf-8") as f:
        f.write("#!/bin/sh\necho 别人家的钩子\n")
    assert ih.main(["install", "--root", str(root)]) == 2         # 拒绝
    with open(path, encoding="utf-8") as f:
        assert "别人家" in f.read()
    assert ih.main(["install", "--root", str(root), "--force"]) == 0
    with open(path, encoding="utf-8") as f:
        assert ih.MARKER in f.read()
    assert any(x.startswith(ih.HOOK_NAME + ".foreign-")
               for x in os.listdir(os.path.dirname(path)))       # 原件备份，不删


def test_remove_refuses_foreign_hook(tmp_path):
    root = tmp_path / "r"
    root.mkdir()
    _git(root, ["init", "-b", "main"])
    path = ih.hook_path(str(root))
    with open(path, "w", encoding="utf-8") as f:
        f.write("#!/bin/sh\nexit 0\n")
    assert ih.main(["remove", "--root", str(root)]) == 2
    assert os.path.isfile(path)


def test_render_hook_has_three_gate_fallbacks():
    body = ih.render_hook("C:/py/python.exe", "C:/repo/collab-toolkit/tools/sign_hook.py",
                          "C:/repo/collab-toolkit/tools/install_git_hook.py")
    assert body.startswith("#!/bin/sh\n")
    assert "pwd -P" in body and "git worktree list --porcelain" in body
    assert "exit 0" in body                       # 找不到校验器 ⇒ 警告放行，不锁死全队
    assert ih.MARKER in body


# ---------------- 3) 行为层：真钩子真提交 ----------------
def test_real_hook_blocks_wrong_author(repo):
    r = _commit_in(repo["wt"], YAN, msg="故意错名")
    assert r.returncode != 0
    blob = r.stdout + r.stderr
    assert "banqiao@the-world.local" in blob            # 拦下：打印应然署名
    assert "the-world.local" in blob
    head = _git(repo["wt"], ["rev-parse", "HEAD"]).stdout.strip()
    assert head == _git(repo["root"], ["rev-parse", "main"]).stdout.strip()  # 没提交进去


def test_real_hook_allows_card_owner(repo):
    r = _commit_in(repo["wt"], BANQIAO, msg="本人提交")
    assert r.returncode == 0, r.stdout + r.stderr
    got = _git(repo["wt"], ["log", "-1", "--format=%an|%ae"]).stdout.strip()
    assert got == "板桥|banqiao@the-world.local"


def test_real_hook_pi_whitelist_in_card_tree(repo):
    r = _commit_in(repo["wt"], ("fish(1)", "pi@the-world.local"), msg="PI 合并")
    assert r.returncode == 0, r.stdout + r.stderr


def test_real_hook_main_tree_only_warns(repo):
    r = _commit_in(repo["root"], ("砚", "yan@the-world.local"), msg="主树维护")
    assert r.returncode == 0, r.stdout + r.stderr      # 警告级不阻断
    assert "警告级" in (r.stdout + r.stderr) or "⚠" in (r.stdout + r.stderr)


def test_real_hook_sign_file_alone_gates_when_no_card_row(repo):
    """卡表无此行（C9）⇒ 回落到本树署名文件判应然。"""
    wt9 = os.path.join(repo["tmp"], "repo", ".worktrees", "C9")
    r = _git(repo["root"], ["worktree", "add", wt9, "-b", "task/C9"])
    assert r.returncode == 0, r.stderr
    git_id.write_sign_file(wt9, *BANQIAO)
    assert _commit_in(wt9, YAN, msg="错名").returncode != 0
    assert _commit_in(wt9, BANQIAO, msg="对名").returncode == 0


def test_check_cli_json_reports_card(repo):
    import json
    r = subprocess.run([sys.executable, os.path.join(TOOLS, "sign_hook.py"),
                        "check", "--root", repo["wt"], "--json"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace",
                       env={**os.environ, **git_id.env_pairs(*BANQIAO)})
    assert r.returncode == 0, r.stderr
    v = json.loads(r.stdout)
    assert v["card"] == "C1" and v["level"] == "ok"
    assert v["expected"]["source"] == "任务卡负责人"


def test_evaluate_in_current_tree_is_read_only_and_shaped():
    """在跑测试的这棵树上真跑一遍 evaluate（只读）：结构完整、绝不抛。"""
    v = sh.evaluate(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    assert v["toplevel"] and v["branch"]
    assert v["level"] in ("ok", "warn", "block")
    assert set(v["expected"]) >= {"card_pairs", "file", "pairs", "source"}


# ---------------- 4) 发帖升格端到端（真钩子，主树无卡；tmp 仓内自足） ----------------
def test_staged_post_roles_and_real_hook_main_tree_post(repo):
    """主树无卡发帖：作者≠署名头拦（a5981a0 同型）/作者不在册拦（2d8c677 同型）/一致放行。"""
    root = repo["root"]
    board = os.path.join(root, "_share", "讨论板.md")
    with open(board, "w", encoding="utf-8") as f:
        f.write("# 讨论板\n\n### [所有者] · 2026-09-17 20:00\n初始帖\n")
    _git(root, ["add", "--", "_share/讨论板.md"])
    r = _git(root, ["commit", "-m", "board init"],
             env={**os.environ, **git_id.env_pairs("天平", "tianping@the-world.local")})
    assert r.returncode == 0, r.stdout + r.stderr     # 头[所有者]=天平 ⇒ 天平发帖放行
    assert sh.staged_post_roles(root) is None          # 无暂存 ⇒ 非发帖提交
    # ① 冒充在册他人：头 [轻舟]，作者 澜舟 ⇒ 真钩子拦（旧口径主树只警告）
    with open(board, "a", encoding="utf-8") as f:
        f.write("\n---\n\n### [轻舟] · 2026-10-05 21:00（Asia/Shanghai）\n\n冒名帖\n")
    _git(root, ["add", "--", "_share/讨论板.md"])
    assert sh.staged_post_roles(root) == ["轻舟"]
    r = _git(root, ["commit", "-m", "冒名帖"],
             env={**os.environ, **git_id.env_pairs(*LANZHOU)})
    assert r.returncode != 0
    assert "冒充在册他人" in (r.stdout + r.stderr)
    # ② 作者不在册（真人 gitee 账号）⇒ 拦
    r = _git(root, ["commit", "-m", "真人帖"],
             env={**os.environ, "GIT_CONFIG_COUNT": "2",
                  "GIT_CONFIG_KEY_0": "user.name",
                  "GIT_CONFIG_VALUE_0": "小鱼",
                  "GIT_CONFIG_KEY_1": "user.email",
                  "GIT_CONFIG_VALUE_1": "14550830+little-fishy@user.noreply.gitee.com"})
    assert r.returncode != 0
    assert "不在花名册" in (r.stdout + r.stderr)
    # ③ 头与作者一致 ⇒ 放行（验收例①：[协作]解析=板桥，作者=板桥）
    with open(board, "r+", encoding="utf-8") as f:
        txt = f.read().replace("### [轻舟] · 2026-10-05 21:00", "### [协作] · 2026-10-05 21:00")
        f.seek(0); f.write(txt); f.truncate()
    _git(root, ["add", "--", "_share/讨论板.md"])
    assert sh.staged_post_roles(root) == ["协作"]
    r = _git(root, ["commit", "-m", "板桥自帖"],
             env={**os.environ, **git_id.env_pairs(*BANQIAO)})
    assert r.returncode == 0, r.stdout + r.stderr
