# -*- coding: utf-8 -*-
"""board_pin.py 测试 + 与 board_post 的协同（置顶块不被发帖破坏）。"""
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import board_pin as bpin  # noqa: E402
import board_post as bp  # noqa: E402

TZ8 = timezone(timedelta(hours=8))
REAL_LOCK_TOOL = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "tools", "share_lock.py"))


# ---------------- 纯函数 ----------------
def test_split_join_roundtrip():
    pin = bpin.build_pin(["第一条", "第二条"])
    text = pin + "\n---\n\n# 讨论板\n正文\n"
    p, rest = bpin.split_board(text)
    assert p == pin.rstrip("\n")
    assert bpin.join_board(p, rest) == text
    assert "正文" in rest and "第一条" not in rest


def test_split_without_pin():
    p, rest = bpin.split_board("# 讨论板\n无置顶\n")
    assert p is None
    assert rest == "# 讨论板\n无置顶\n"


def test_parse_items():
    pin = bpin.build_pin(["甲", "乙", "丙"])
    assert len(bpin.parse_items(pin)) == 3


def test_validate_ok():
    pin = bpin.build_pin(["甲"])
    findings = bpin.validate_pin(pin)
    assert findings[0]["level"] == "ok"


def test_validate_too_many_items():
    pin = bpin.build_pin([f"第{i}条" for i in range(6)])
    findings = bpin.validate_pin(pin)
    assert any(f["level"] == "error" and "超过上限" in f["msg"] for f in findings)


def test_validate_stale():
    now = datetime(2026, 9, 17, tzinfo=TZ8)
    pin = bpin.build_pin(["甲"], now=now - timedelta(days=30))
    findings = bpin.validate_pin(pin, today=now.date())
    assert any(f["level"] == "warn" and "未更新" in f["msg"] for f in findings)


def test_validate_missing_pin():
    findings = bpin.validate_pin(None)
    assert findings[0]["level"] == "warn"


def test_validate_unclosed_marker():
    with pytest.raises(bpin.PinError):
        bpin.split_board("<!-- PIN-BEGIN -->\n没有结尾\n")


# ---------------- 与 board_post 协同 ----------------
@pytest.fixture()
def repo_with_pin(tmp_path):
    bare = tmp_path / "remote.git"
    work = tmp_path / "work"
    subprocess.run(["git", "init", "--bare", str(bare)], check=True, capture_output=True,
                   text=True, encoding="utf-8", errors="replace")
    subprocess.run(["git", "clone", str(bare), str(work)], check=True, capture_output=True,
                   text=True, encoding="utf-8", errors="replace")
    for k, v in (("user.email", "t@t"), ("user.name", "t")):
        subprocess.run(["git", "config", k, v], cwd=work, check=True)
    share = work / "_share"
    (share / ".locks").mkdir(parents=True)
    pin = bpin.build_pin(["当前要事：C1b/C2 跑批中"])
    (share / "讨论板.md").write_text(
        pin + "\n---\n\n# 讨论板\n\n### [所有者] · 2026-09-17 20:00\n初始帖\n",
        encoding="utf-8")
    subprocess.run(["git", "add", "_share/讨论板.md"], cwd=work, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=work, check=True,
                   capture_output=True, text=True, encoding="utf-8", errors="replace")
    subprocess.run(["git", "push", "origin", "HEAD:main"], cwd=work, check=True,
                   capture_output=True, text=True, encoding="utf-8", errors="replace")
    return str(work)


def test_post_preserves_pin_block(repo_with_pin, tmp_path):
    msg = tmp_path / "post.md"
    msg.write_text("**主题**：置顶兼容测试\n", encoding="utf-8")
    rc = bp.main(["--root", repo_with_pin, "--slot", "collab", "--task", "测试",
                  "--message-file", str(msg), "--role", "[协作]",
                  "--remote", "origin", "--branch", "main",
                  "--credential-helper", "", "--lock-tool", REAL_LOCK_TOOL])
    assert rc == 0
    text = (tmp_path / "work" / "_share" / "讨论板.md").read_text(encoding="utf-8")
    pin, rest = bpin.split_board(text)
    assert pin is not None and "当前要事：C1b/C2 跑批中" in pin   # 置顶原样
    assert "置顶兼容测试" in rest                                # 新帖进正文
    assert text.index("PIN-BEGIN") < text.index("初始帖")        # 置顶仍在最前


def test_pin_set_and_add_remove(tmp_path, capsys):
    root = tmp_path / "r"
    share = root / "_share"
    (share / ".locks").mkdir(parents=True)
    board = share / "讨论板.md"
    board.write_text("# 讨论板\n\n### [所有者] · 2026-09-17 20:00\n初始帖\n",
                     encoding="utf-8")
    base = ["--root", str(root), "--lock-tool", REAL_LOCK_TOOL]

    pf = tmp_path / "pin.md"
    pf.write_text("跑批中：C1b/C2\n等裁定：仪器门重订\n", encoding="utf-8")
    assert bpin.main(["set", "--file", str(pf)] + base) == 0
    text = board.read_text(encoding="utf-8")
    assert "跑批中：C1b/C2" in text and "初始帖" in text

    assert bpin.main(["add", "--text", "09-19 前：C 专题征集"] + base) == 0
    pin, _ = bpin.split_board(board.read_text(encoding="utf-8"))
    assert len(bpin.parse_items(pin)) == 3

    assert bpin.main(["remove", "--index", "2"] + base) == 0
    pin, _ = bpin.split_board(board.read_text(encoding="utf-8"))
    items = bpin.parse_items(pin)
    assert len(items) == 2
    assert all("等裁定" not in it for it in items)

    assert bpin.main(["check"] + base) == 0


def test_pin_add_over_limit_rejected(tmp_path):
    root = tmp_path / "r"
    (root / "_share" / ".locks").mkdir(parents=True)
    board = root / "_share" / "讨论板.md"
    board.write_text(bpin.build_pin(["1", "2", "3", "4", "5"]) + "\n---\n\n正文\n",
                     encoding="utf-8")
    rc = bpin.main(["add", "--text", "第6条", "--root", str(root),
                    "--lock-tool", REAL_LOCK_TOOL])
    assert rc == 3
    assert bpin.parse_items(bpin.split_board(board.read_text(encoding="utf-8"))[0]) \
        .__len__() == 5
