# -*- coding: utf-8 -*-
"""onboard.py 测试。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import onboard as ob  # noqa: E402


def test_list_roles(capsys):
    assert ob.main(["--role", "list"]) == 0
    out = capsys.readouterr().out
    for key in ("owner", "dev", "eval", "web", "cloud", "collab"):
        assert key in out


def test_unknown_role(capsys):
    assert ob.main(["--role", "nobody"]) == 2


def test_render_collab_contains_essentials():
    text = ob.render("collab", ".", net=False)
    assert "### [协作] · YYYY-MM-DD HH:MM" in text
    assert "--slot collab" in text
    assert "TASKS-COLLAB.md" in text
    assert "AGENT.md" in text
    assert "（（" not in text.splitlines()[0]  # 花名不双括号


def test_render_all_roles_smoke():
    for key in ob.ROLE_REGISTRY:
        text = ob.render(key, ".", net=False)
        assert "必读清单" in text and "开工第一步" in text


# ---------------- 交接包（T-7 P1） ----------------
def test_role_recent_lines_prefers_latest_section(tmp_path):
    """同日日志有多个本角色段落 ⇒ 取最新的那个；'[协作·板桥]' 变体也能匹配。"""
    log = tmp_path / "2026-09-19.md"
    log.write_text(
        "## [协作] 早上的段落\n- 旧要点\n\n## [协作·板桥] 15:20 晚段\n- 新要点一\n- 新要点二\n",
        encoding="utf-8")
    pts = ob.role_recent_lines(str(tmp_path), "[协作]")
    assert pts == ["新要点一", "新要点二"]


def test_role_recent_lines_no_match(tmp_path):
    (tmp_path / "2026-09-19.md").write_text("## [所有者] 别人的段\n- x\n", encoding="utf-8")
    assert ob.role_recent_lines(str(tmp_path), "[协作]") == []


def test_role_open_items_filters_by_owner(tmp_path):
    board = tmp_path / "讨论板.md"
    board.write_text(
        "| # | 决策点 | 结论 | 未决 | 责任方 |\n|---|---|---|---|---|\n"
        "| 6 | 团队记忆 | 已派工 | 设计稿 | `[协作]` 板桥 |\n"
        "| 9 | 移交余项 | 认领 | #4 | `[本地开发]` |\n",
        encoding="utf-8")
    items = ob.role_open_items(str(board), "[协作]")
    assert len(items) == 1 and "团队记忆" in items[0]


def test_latest_posts_excludes_evidence_headers(tmp_path):
    board = tmp_path / "讨论板.md"
    board.write_text(
        "### [协作] · 2026-09-19 14:16\n帖\n### [实测] 验收\n证据小节\n"
        "### [本地开发] · 2026-09-19 16:02\n帖\n",
        encoding="utf-8")
    posts = ob.latest_posts(str(board), 3)
    assert len(posts) == 2 and all("实测" not in p.split("]")[0] for p in posts)


def test_render_contains_handover():
    text = ob.render("collab", ".", net=False)
    assert "交接包" in text and "上次进展" in text and "未完成" in text
