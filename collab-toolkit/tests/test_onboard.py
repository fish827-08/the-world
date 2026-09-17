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
