# -*- coding: utf-8 -*-
"""board_check.py 单元测试 —— 用临时目录构造假 _share/，不碰真板。"""
import json
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import board_check as bc  # noqa: E402


# ---------------- 夹具 ----------------
@pytest.fixture()
def fake_share(tmp_path):
    """构造 the-world 假仓库：root/_share/讨论板.md 等。"""
    root = tmp_path / "repo"
    share = root / "_share"
    (share / ".locks").mkdir(parents=True)
    board = share / "讨论板.md"
    board.write_text(
        "# 讨论板\n\n"
        "### [所有者] · 2026-09-17 22:00（Asia/Shanghai）\n"
        "**主题**：测试帖一\n\n"
        "### [本地开发] · 2026-09-17\n"  # 缺 HH:MM -> 警告
        "**主题**：旧格式帖\n\n"
        "### 七、帖内小标题（不是署名）\n"
        "正文引用 `不存在目录/某某文件.md` 和 `_share/README.md`。\n",
        encoding="utf-8")
    (share / "README.md").write_text("# README\n", encoding="utf-8")
    return str(root), str(share), str(board)


# ---------------- 署名检查 ----------------
def test_iter_posts_parses_signatures(fake_share):
    _, _, board = fake_share
    posts = list(bc.iter_posts(board))
    assert len(posts) == 2
    assert posts[0]["role"] == "所有者"
    assert posts[0]["time"] == "22:00"
    assert posts[1]["time"] is None  # 旧格式


def test_check_signatures_warns_on_missing_time(fake_share):
    _, _, board = fake_share
    findings = bc.check_signatures(board)
    levels = {f["item"]: f["level"] for f in findings}
    assert any(f["level"] == "warn" and "缺 HH:MM" in f["msg"] for f in findings)
    assert not any(f["level"] == "error" for f in findings)


def test_check_signatures_flags_bad_sig_like(tmp_path):
    board = tmp_path / "讨论板.md"
    board.write_text("### [某人] 2026-09-17 缺间隔号与格式\n", encoding="utf-8")
    findings = bc.check_signatures(str(board))
    assert any(f["level"] == "error" for f in findings)


# ---------------- 大小检查 ----------------
def test_check_sizes_thresholds(fake_share):
    root, share, board = fake_share
    big = os.path.join(share, "路线图.md")
    with open(big, "w", encoding="utf-8") as f:
        f.write("x" * (140 * 1024))  # 140KB -> 预警
    findings = bc.check_sizes(share)
    by_file = {f.get("file"): f for f in findings if f.get("file")}
    assert by_file["路线图.md"]["level"] == "warn"
    assert by_file["README.md"]["level"] == "ok"


def test_check_sizes_error_over_200k(fake_share):
    """§3.3 阈值 2026-09-19 由 150 → 200KB；本测试钉死**工具与章程一致**。

    ⚠️ 若改 `bc.SIZE_ERR_KB` 而不同步 `AGENT.md` §3.3 ⇒ 两处漂移（F-R23 家族）。
    """
    _, share, _ = fake_share
    with open(os.path.join(share, "讨论板.md"), "w", encoding="utf-8") as f:
        f.write("x" * (201 * 1024))
    findings = bc.check_sizes(share)
    assert any(f["level"] == "error" and f.get("file") == "讨论板.md"
               for f in findings)
    # 150KB 已**不再**报警（阈值已放宽）
    with open(os.path.join(share, "讨论板.md"), "w", encoding="utf-8") as f:
        f.write("x" * (151 * 1024))
    findings = bc.check_sizes(share)
    assert not any(f["level"] == "error" and f.get("file") == "讨论板.md"
                   for f in findings)


# ---------------- 锁检查 ----------------
def test_check_locks_empty(fake_share):
    _, share, _ = fake_share
    findings = bc.check_locks(share)
    assert findings[0]["level"] == "ok"
    assert "空闲" in findings[0]["msg"]


def test_check_locks_stale_warns(fake_share):
    _, share, _ = fake_share
    lock = os.path.join(share, ".locks", "dev.lock")
    old = time.time() - 2000
    with open(lock, "w", encoding="utf-8") as f:
        json.dump({"slot": "dev", "ttl": 900,
                   "ts": "2026-09-17T20:00:00+08:00"}, f)
    os.utime(lock, (old, old))
    findings = bc.check_locks(share)
    assert any(f["level"] == "warn" and "stale" in f["msg"] for f in findings)


# ---------------- 引用断链 ----------------
def test_check_refs_resolves_share_relative(fake_share):
    root, share, board = fake_share
    findings = bc.check_refs(root, board)
    warn = [f for f in findings if f["level"] == "warn"]
    # `_share/README.md` 存在 -> 不报；`不存在目录/某某文件.md` -> 报
    assert len(warn) == 1
    assert warn[0]["missing"] == ["不存在目录/某某文件.md"]


def test_check_refs_moved_prefix_hint(fake_share):
    root, share, board = fake_share
    with open(board, "a", encoding="utf-8") as f:
        f.write("引用旧路径 `_share/评估-某某.md`。\n")
    findings = bc.check_refs(root, board)
    warn = [f for f in findings if f["level"] == "warn"][0]
    assert "已移至 docs/评估-EVAL/" in warn["msg"]


# ---------------- metrics ----------------
def test_collect_metrics_counts(fake_share):
    root, share, _ = fake_share
    with open(os.path.join(share, "路线共识.md"), "w", encoding="utf-8") as f:
        f.write("R1 R2 R2 R111 F-R9 F-R18 F-D2\n")  # R2 重复应去重
    m = bc.collect_metrics(root)
    assert m["board_posts"] == 2
    assert m["ruling_count"] == 3       # R1 R2 R111 去重；F-R9 的 R9 不计
    assert m["defect_count"] == 3       # F-R9 F-R18 F-D2
    assert m["days_since_start"] >= 1


def test_metrics_ruling_regex_excludes_fr(fake_share):
    """F-R9 不应被计为裁定 R9（负向后行断言）。"""
    root, share, _ = fake_share
    with open(os.path.join(share, "路线共识.md"), "w", encoding="utf-8") as f:
        f.write("只有 F-R9 和 F-D2，没有独立裁定。\n")
    m = bc.collect_metrics(root)
    assert m["ruling_count"] == 0
    assert m["defect_count"] == 2


# ---------------- 署名层级 / 新鲜度 / 检索（2026-09-19 新增） ----------------
def test_sig_level_error_on_double_hash(tmp_path):
    """`## [角色] …` 会让自动索引漏帖 ⇒ 必须报错。"""
    board = tmp_path / "讨论板.md"
    board.write_text("# 板\n\n## [内评] 昨晚任务审核（09-19 14:0x）\n正文\n",
                     encoding="utf-8")
    findings = bc.check_signature_levels(str(board))
    assert any(f["level"] == "error" and "L3" in f["msg"] for f in findings)


def test_sig_level_ok(tmp_path):
    board = tmp_path / "讨论板.md"
    board.write_text("### [协作] · 2026-09-19 14:00（Asia/Shanghai）\n正文\n",
                     encoding="utf-8")
    assert bc.check_signature_levels(str(board))[0]["level"] == "ok"


def test_latest_post_date_falls_back_to_body(tmp_path):
    """标题不带 `· 日期` 时，从标题后 6 行内取日期（兼容漏索引写法）。"""
    board = tmp_path / "讨论板.md"
    board.write_text("## [内评] 审稿\n审核时刻 2026-09-19 14:0x；基线 x\n",
                     encoding="utf-8")
    assert bc.latest_post_date(str(board)) == "2026-09-19"


def test_summary_freshness_stale(tmp_path):
    board = tmp_path / "讨论板.md"
    board.write_text(
        "## 一、状态快照（2026-09-18）\n\n"
        "### [所有者] · 2026-09-19 01:42（Asia/Shanghai）\n新帖\n",
        encoding="utf-8")
    findings = bc.check_summary_freshness(str(board))
    assert findings[0]["level"] == "warn"
    assert "2026-09-18" in findings[0]["msg"]


def test_summary_freshness_ok(tmp_path):
    board = tmp_path / "讨论板.md"
    board.write_text("## 一、状态快照（2026-09-19）\n\n"
                     "### [所有者] · 2026-09-19 01:42\n新帖\n", encoding="utf-8")
    assert bc.check_summary_freshness(str(board))[0]["level"] == "ok"


def test_todo_sync_lag_warns(tmp_path):
    root = tmp_path / "r"
    (root / "_share").mkdir(parents=True)
    board = root / "_share" / "讨论板.md"
    board.write_text("### [所有者] · 2026-09-19 01:42\n新帖\n", encoding="utf-8")
    (root / "_share" / "待办与交接.md").write_text(
        "| 2026-09-18 01:10 | 旧条目 |\n", encoding="utf-8")
    findings = bc.check_todo_sync(str(root / "_share"), str(board))
    assert findings[0]["level"] == "warn"


def test_find_cli_by_id(fake_share, capsys):
    root, share, _ = fake_share
    with open(os.path.join(share, "路线共识.md"), "w", encoding="utf-8") as f:
        f.write("| **R200** | 测试条目 |\n")
    rc = bc.main(["find", "--root", root, "--id", "R200"])
    out = capsys.readouterr().out
    assert rc == 0 and "R200" in out and "路线共识.md:1" in out


# ---------------- CLI ----------------
def test_cli_check_exit_code(fake_share, capsys):
    root, _, _ = fake_share
    rc = bc.main(["check", "--root", root])
    out = capsys.readouterr().out
    assert rc == 1  # 有警告（缺时分 + 断链），无错误
    assert "板面体检" in out


def test_cli_metrics_json(fake_share, capsys):
    root, _, _ = fake_share
    rc = bc.main(["metrics", "--root", root, "--json"])
    data = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert data["board_posts"] == 2


# ---------------- 编码污染（F-R31 家族） ----------------
def test_utf8_problems_clean():
    assert bc.utf8_problems("中文与 emoji 🔴 正常文本\n") == []


def test_utf8_problems_detects_cesu8():
    """emoji 被按 UTF-16 代理对逐个编码 ⇒ CESU-8（F-R31 事故原型）。"""
    bad = "中文 ".encode("utf-8") + b"\xed\xa0\xbd\xed\xb4\xb4" + " 尾部\n".encode("utf-8")
    problems = bc.utf8_problems(bad)
    assert problems and any("CESU-8" in p for p in problems)


def test_check_encoding_reports_file(tmp_path):
    good = tmp_path / "a.md"
    good.write_text("干净文件 🔴\n", encoding="utf-8")
    bad = tmp_path / "b.md"
    bad.write_bytes("污染 ".encode("utf-8") + b"\xed\xa0\xbd\xed\xb4\xb4" + b"\n")
    findings = bc.check_encoding([str(good), str(bad)])
    assert findings[0]["level"] == "error"
    assert "b.md" in findings[0]["msg"]


def test_check_encoding_all_clean(tmp_path):
    p = tmp_path / "a.md"
    p.write_text("干净 🔴\n", encoding="utf-8")
    assert bc.check_encoding([str(p)])[0]["level"] == "ok"
