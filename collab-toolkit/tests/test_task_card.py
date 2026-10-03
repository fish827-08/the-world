# -*- coding: utf-8 -*-
"""task_card.py 单元测试 —— 任务卡状态机（R359 B2）。

🔴 纪律：全部走 `--no-lock` + `--file <临时路径>`，**不碰 `_share/` 真卡表**
（多会话共用工作树，测试污染真数据 = 事故）。
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import task_card as tc  # noqa: E402

HEADER = tc._render_header()


# ---------------- 夹具 ----------------
@pytest.fixture()
def table(tmp_path):
    """干净的临时卡表路径（尚未建文件）。"""
    p = tmp_path / "cards.md"
    return str(p)


def add_cli(path, card, **kw):
    argv = ["add", card, "--no-lock", "--file", path]
    for k, v in kw.items():
        argv.append("--" + k.replace("_", "-"))
        argv.append(v)
    return tc.main(argv)


# ---------------- 解析（纯函数，board_check 复用） ----------------
def test_parse_card_row_ok():
    d = tc.parse_card_row("| R1 | claimed | 板桥 | 目标 | 输入 | 约束 | 验收 | 2026-10-03 20:00 | 2026-10-03 20:05 | 结果 |")
    assert d and d["卡号"] == "R1" and d["状态"] == "claimed"


def test_parse_card_row_rejects_non_card():
    assert tc.parse_card_row("普通正文行") is None
    assert tc.parse_card_row("| 1 | 不是状态 | x | y |") is None  # 状态非法


def test_parse_cards_roundtrip(tmp_path):
    p = str(tmp_path / "cards.md")
    add_cli(p, "R1", goal="目标A", inputs="输入B", accept="验收C")
    text = open(p, encoding="utf-8").read()
    cards = tc.parse_cards(text)
    assert len(cards) == 1
    assert cards[0]["目标"] == "目标A" and cards[0]["状态"] == "pending"


def test_escape_pipes_preserves_cell():
    assert tc.esc("a|b") == "a\\|b" and tc.unesc(tc.esc("a|b")) == "a|b"


# ---------------- 卡号与字符上限（防注入 / 一行一卡前提） ----------------
def test_card_id_rejects_injection():
    for bad in ("R1; rm -rf /", "../evil", "", "-x", "R1 "):
        with pytest.raises(tc.CardError):
            tc.check_card(bad)


def test_char_limit_blocks_oversize(table, capsys):
    """🔴 断言**退出码 + stderr**：main() 把 CardError 折成非零码（CLI 契约）。"""
    assert add_cli(table, "R1", goal="x" * (tc.LIMITS["goal"] + 1)) != 0
    assert "超长" in capsys.readouterr().err
    assert not os.path.isfile(table)  # 超长不落盘（截断会丢证据，故直接拒）


def test_char_limit_allows_exact_boundary(table):
    add_cli(table, "R1", goal="x" * tc.LIMITS["goal"])
    assert os.path.isfile(table)


# ---------------- 状态机主干 ----------------
def test_add_then_status_pending(table, capsys):
    assert add_cli(table, "R1", goal="测试目标", accept="一行一卡") == 0
    assert tc.main(["status", "--no-lock", "--file", table]) == 0
    out = capsys.readouterr().out
    assert "R1" in out and "pending" in out and "汇总: pending 1" in out


def test_full_chain_pending_claimed_done_verified(table):
    f = ["--no-lock", "--file", table]
    assert add_cli(table, "R1", goal="目标") == 0
    assert tc.main(["claim", "R1", "板桥"] + f) == 0
    assert tc.parse_cards(open(table, encoding="utf-8").read())[0]["状态"] == "claimed"
    assert tc.main(["done", "R1"] + f + ["--result", "结果X", "--summary", "摘要Y"]) == 0
    assert tc.main(["verify", "R1"] + f) == 0
    d = tc.parse_cards(open(table, encoding="utf-8").read())[0]
    assert d["状态"] == "verified"
    assert "结果X" in d["结果"] and "摘要Y" in d["结果"]


def test_illegal_transition_is_rejected(table, capsys):
    """claimed 不能直接 verify（必须 done）—— 状态机守门，不是装饰。"""
    f = ["--no-lock", "--file", table]
    add_cli(table, "R1", goal="目标")
    tc.main(["claim", "R1", "板桥"] + f)
    assert tc.main(["verify", "R1"] + f) != 0
    assert "只认" in capsys.readouterr().err


def test_double_claim_is_rejected(table, capsys):
    f = ["--no-lock", "--file", table]
    add_cli(table, "R1", goal="目标")
    tc.main(["claim", "R1", "甲"] + f)
    assert tc.main(["claim", "R1", "乙"] + f) != 0
    assert "当前状态 claimed" in capsys.readouterr().err


def test_verify_reject_goes_back_to_pending(table):
    f = ["--no-lock", "--file", table]
    add_cli(table, "R1", goal="目标")
    tc.main(["claim", "R1", "板桥"] + f)
    tc.main(["done", "R1"] + f + ["--result", "半成品"])
    assert tc.main(["verify", "R1", "--reject"] + f) == 0
    d = tc.parse_cards(open(table, encoding="utf-8").read())[0]
    assert d["状态"] == "pending"  # 打回后可重新认领


def test_claim_needs_owner(table, capsys):
    assert tc.main(["claim", "R1"] + ["--no-lock", "--file", table]) != 0
    assert "认领人" in capsys.readouterr().err


# ---------------- 重复卡号 ----------------
def test_duplicate_card_rejected(table, capsys):
    add_cli(table, "R1", goal="目标")
    assert add_cli(table, "R1", goal="目标") != 0
    assert "已存在" in capsys.readouterr().err


def test_overwrite_updates_existing(table):
    f = ["--no-lock", "--file", table]
    add_cli(table, "R1", goal="旧目标")
    tc.main(["claim", "R1", "板桥"] + f)
    tc.main(["add", "R1", "--overwrite"] + f + ["--goal", "新目标"])
    cards = tc.parse_cards(open(table, encoding="utf-8").read())
    assert len(cards) == 1 and cards[0]["目标"] == "新目标"


# ---------------- stale（超时打回） ----------------
def _write_stale_table(path, ts="08:00"):
    """直接造一张「早上认领、到现在没动」的卡（绕开时间等待）。"""
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(HEADER)
        f.write("| R9 | claimed | 板桥 | 旧卡 | - | - | - | 2026-10-03 %s | "
                "2026-10-03 %s |  |\n" % (ts, ts))


def test_stale_dry_run_reports_without_changing(table, capsys):
    _write_stale_table(table)
    assert tc.main(["stale", "--no-lock", "--file", table]) == 0
    out = capsys.readouterr().out
    assert "R9" in out and "dry-run" in out
    assert tc.parse_cards(open(table, encoding="utf-8").read())[0]["状态"] == "claimed"


def test_stale_apply_rewinds_to_pending(table):
    _write_stale_table(table)
    assert tc.main(["stale", "--apply", "--no-lock", "--file", table]) == 0
    d = tc.parse_cards(open(table, encoding="utf-8").read())[0]
    assert d["状态"] == "pending"


def test_stale_no_hits_when_fresh(table, capsys):
    f = ["--no-lock", "--file", table]
    add_cli(table, "R1", goal="目标")
    tc.main(["claim", "R1", "板桥"] + f)
    assert tc.main(["stale"] + f) == 0
    assert "无 claimed 卡滞留" in capsys.readouterr().out


# ---------------- status 输出形态（验收：一行一卡） ----------------
def test_status_prints_one_line_per_card(table, capsys):
    f = ["--no-lock", "--file", table]
    add_cli(table, "R1", goal="目标一", accept="验收一")
    add_cli(table, "R2", goal="目标二", accept="验收二")
    tc.main(["status"] + f)
    out = capsys.readouterr().out
    assert "任务卡 2 张" in out
    assert "R1" in out and "R2" in out
    # 汇聚行必须列出两卡
    assert "pending 2" in out


def test_status_empty_table(table, capsys):
    assert tc.main(["status", "--no-lock", "--file", table]) == 0
    assert "为空" in capsys.readouterr().out
