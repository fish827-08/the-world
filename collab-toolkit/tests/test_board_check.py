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
        f.write("x" * (bc.SIZE_WARN_KB + 10) * 1024)  # 预警线 +10KB -> 警告
    findings = bc.check_sizes(share)
    by_file = {f.get("file"): f for f in findings if f.get("file")}
    assert by_file["路线图.md"]["level"] == "warn"
    assert by_file["README.md"]["level"] == "ok"


def test_check_sizes_error_over_limit(fake_share):
    """🔴 本测试钉死**工具阈值 == AGENT.md §3.3 硬上限**（F-R23 家族：两处漂移）。

    ⚠️ 阈值史：预警 130→200（2026-09-19）｜硬上限 200→**300**（2026-10-01 fish 定，
    见 board_check.py 注释）。原测试还钉着 200/130 的老值 ⇒ **套件一直红**，
    2026-10-03 由 R359 B3 交付时一并修（红套件不可交付）。
    🔴 改阈值时**必须同步**：`board_check.py` 常量 + `AGENT.md §3.3` + 本测试。
    """
    _, share, _ = fake_share
    with open(os.path.join(share, "讨论板.md"), "w", encoding="utf-8") as f:
        f.write("x" * ((bc.SIZE_ERR_KB + 1) * 1024))
    findings = bc.check_sizes(share)
    assert any(f["level"] == "error" and f.get("file") == "讨论板.md"
               for f in findings)
    # 越过预警线、但没到硬上限 ⇒ 只警告、不报错
    with open(os.path.join(share, "讨论板.md"), "w", encoding="utf-8") as f:
        f.write("x" * ((bc.SIZE_WARN_KB + 1) * 1024))
    findings = bc.check_sizes(share)
    assert any(f["level"] == "warn" and f.get("file") == "讨论板.md"
               for f in findings)
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


# ---------------- B3：任务卡池异常检测（四条规则各触发一次） ----------------
OLD_TS = "2026-10-03 07:00"   # 明显早于任何阈值；用旧日期 ⇒ 测试不依赖"此刻"


def _write_cards(path, rows):
    """rows = [(卡号, 状态, 负责, 建卡, 更新)]；写成 task_card 的同款表格。"""
    head = "| 卡号 | 状态 | 负责 | 目标 | 输入 | 约束 | 验收 | 建卡 | 更新 | 结果 |\n"
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("# 任务卡\n\n" + head + "|" + "---" * 10 + "|\n")
        for cid, st, who, t0, t1 in rows:
            f.write("| %s | %s | %s | 目标%s | - | - | - | %s | %s |  |\n"
                    % (cid, st, who, cid, t0, t1))


def test_card_pool_rule1_pending_too_few(tmp_path):
    """规则1 卡池告急：pending < 2。"""
    p = str(tmp_path / "cards.md")
    _write_cards(p, [("R1", "pending", "", OLD_TS, OLD_TS),
                     ("R2", "claimed", "甲", OLD_TS, OLD_TS)])
    out = bc.check_card_pool(str(tmp_path), path=p)
    hits = [f for f in out if f["level"] == "error" and "卡池告急" in f["msg"]]
    assert len(hits) == 1 and "pending 1 < 2" in hits[0]["msg"]


def test_card_pool_rule1_ok_when_enough_pending(tmp_path):
    p = str(tmp_path / "cards.md")
    _write_cards(p, [("R1", "pending", "", OLD_TS, OLD_TS),
                     ("R2", "pending", "", OLD_TS, OLD_TS)])
    assert not [f for f in bc.check_card_pool(str(tmp_path), path=p)
                if f["level"] == "error"]


def test_card_pool_rule2_claimed_stale(tmp_path):
    """规则2 claimed 滞留 > 30min。"""
    p = str(tmp_path / "cards.md")
    _write_cards(p, [("R1", "claimed", "甲", OLD_TS, OLD_TS)])
    out = bc.check_card_pool(str(tmp_path), path=p, stale_claim_min=1)
    hits = [f for f in out if f["level"] == "warn" and "claimed 滞留" in f["msg"]]
    assert len(hits) == 1 and "R1" in hits[0]["msg"] and "甲" in hits[0]["msg"]


def test_card_pool_rule3_done_backlog(tmp_path):
    """规则3 done 积压 > 5min 未核验（协调者可能挂了）。"""
    p = str(tmp_path / "cards.md")
    _write_cards(p, [("R1", "done", "甲", OLD_TS, OLD_TS)])
    out = bc.check_card_pool(str(tmp_path), path=p, done_min=1)
    hits = [f for f in out if f["level"] == "warn" and "done 积压" in f["msg"]]
    assert len(hits) == 1 and "R1" in hits[0]["msg"]


def test_card_pool_rule4_feedback_open(tmp_path):
    """规则4 反馈悬置：failed 卡（负反馈等裁决）超 10min 无人处理。"""
    p = str(tmp_path / "cards.md")
    _write_cards(p, [("R1", "failed", "甲", OLD_TS, OLD_TS)])
    out = bc.check_card_pool(str(tmp_path), path=p, feedback_min=1)
    hits = [f for f in out if f["level"] == "warn" and "反馈悬置" in f["msg"]]
    assert len(hits) == 1 and "R1" in hits[0]["msg"]


def test_card_pool_done_is_not_counted_as_feedback(tmp_path):
    """🔴 规则3 与规则4 不打架：done 只走 5min 那条（正反馈），失败才走 10min。"""
    p = str(tmp_path / "cards.md")
    _write_cards(p, [("R1", "done", "甲", OLD_TS, OLD_TS)])
    out = bc.check_card_pool(str(tmp_path), path=p, feedback_min=1)
    assert not [f for f in out if "反馈悬置" in f["msg"]]


def test_card_pool_silent_when_file_missing(tmp_path):
    """🔴 卡表还没建 ⇒ 静默跳过（否则机制刚落地就天天误报）。"""
    out = bc.check_card_pool(str(tmp_path), path=str(tmp_path / "不存在.md"))
    assert out == []


def test_card_pool_ok_when_all_fresh(tmp_path):
    p = str(tmp_path / "cards.md")
    now = bc._now().strftime("%Y-%m-%d %H:%M")
    _write_cards(p, [("R1", "pending", "", now, now),
                     ("R2", "pending", "", now, now)])
    out = bc.check_card_pool(str(tmp_path), path=p)
    assert any(f["level"] == "ok" and f["item"] == "cardpool" for f in out)
    assert not [f for f in out if f["level"] != "ok"]


def test_check_cli_accepts_card_file(tmp_path, fake_share):
    """--card-file 能透传（否则规则形同虚设）。"""
    root, share, _ = fake_share
    p = os.path.join(tmp_path, "cards.md")
    _write_cards(p, [("R1", "pending", "", OLD_TS, OLD_TS)])
    # 🔴 --root 属子解析器 ⇒ 必须放在子命令之后（老写法 `main(["--root", r,"check"])` 会撞）
    assert bc.main(["check", "--root", root, "--no-refs", "--no-encoding",
                    "--card-file", p]) == 2  # 卡池告急 = error ⇒ 2
