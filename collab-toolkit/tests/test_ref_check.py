# -*- coding: utf-8 -*-
"""ref_check.py 单测（FILE-REORG §五 零断链机制）"""
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "tools"))
import ref_check as rc  # noqa: E402


def test_extract_tokens_basic():
    text = "见 docs/判读稿-N1-20261004.md 与 tools/share_lock.py；另有 AGENTS.md 单段不算。"
    toks = rc.extract_tokens(text)
    assert "docs/判读稿-N1-20261004.md" in toks
    assert "tools/share_lock.py" in toks
    assert all("AGENTS.md" not in t for t in toks)  # 无 / 不算仓内路径 token


def test_extract_tokens_external_skipped():
    text = ("https://gitee.com/a/b.py 和 C:/x/y.md 和 the-world-data/p1c/a.csv "
            "和 ~/d/e.md 都不算；docs/README.md 算")
    toks = rc.extract_tokens(text)
    assert toks == {"docs/README.md"}


def test_extract_tokens_ignores_colon_lineno():
    toks = rc.extract_tokens("真源：tools/device_presets.py:22")
    assert toks == {"tools/device_presets.py"}


def test_resolve_root_and_srcdir(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "a.md").write_text("x", encoding="utf-8")
    assert rc.resolve("docs/a.md", str(tmp_path), str(tmp_path))
    # 相对引用文件目录解析：docs/sub 引用 ../a.md 形式不算（不支持 ..），
    # 但 docs 内部文件里写 "a.md/…" 不适用；测第二基准：src_dir 下同名相对路径
    (tmp_path / "docs" / "sub").mkdir()
    (tmp_path / "docs" / "sub" / "b.md").write_text("x", encoding="utf-8")
    assert rc.resolve("sub/b.md", str(tmp_path), str(tmp_path / "docs"))
    assert not rc.resolve("nope/x.md", str(tmp_path), str(tmp_path))


def _mk_tree(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "live.md").write_text("# 活件\n", encoding="utf-8")
    (tmp_path / "a.md").write_text("引用 docs/live.md 与 docs/gone.md", encoding="utf-8")
    (tmp_path / "_share").mkdir()
    (tmp_path / "_share" / "讨论板.md").write_text("历史引用 docs/gone.md", encoding="utf-8")
    return ["a.md", "_share/讨论板.md", "docs/live.md"]


def test_scan_marks_broken(tmp_path):
    files = _mk_tree(tmp_path)
    s = rc.scan(str(tmp_path), files=files)
    assert "docs/live.md" in s["tokens"] and s["tokens"]["docs/live.md"]["ok"]
    assert "docs/gone.md" in s["broken"]
    assert set(s["broken"]["docs/gone.md"]) == {"a.md", "_share/讨论板.md"}


def test_new_broken_semantics(tmp_path):
    files = _mk_tree(tmp_path)
    before = rc.scan(str(tmp_path), files=files)
    # 搬走 live.md ⇒ 引用变断链（新增断链），gone.md 早已断（存量，不重复记账）
    (tmp_path / "docs" / "live.md").unlink()
    after = rc.scan(str(tmp_path), files=files)
    nb = rc.new_broken(before, after, exclude=[])
    toks = {e["token"] for e in nb}
    assert toks == {"docs/live.md"}
    # 唯一引用方落入默认排除面（板上历史引用不回头改）⇒ 不报新增
    nb2 = rc.new_broken(before, {"broken": {"docs/新断.md": ["a.md"]}}, exclude=[])
    assert {e["token"] for e in nb2} == {"docs/新断.md"}
    nb3 = rc.new_broken(before, {"broken": {"docs/新断.md": ["_share/讨论板.md"]}},
                        exclude=rc.DEFAULT_EXCLUDE)
    assert nb3 == []


def test_excluded_patterns():
    assert rc.excluded("_share/讨论板.md", rc.DEFAULT_EXCLUDE)
    assert rc.excluded("_archive/2026-09-17/x.md", rc.DEFAULT_EXCLUDE)
    assert rc.excluded("_share/archive/y.md", rc.DEFAULT_EXCLUDE)
    assert not rc.excluded("docs/README.md", rc.DEFAULT_EXCLUDE)


def test_cli_snapshot_and_diff(tmp_path, capsys, monkeypatch):
    files = _mk_tree(tmp_path)
    monkeypatch.setattr(rc, "git_tracked", lambda root: files)
    out = tmp_path / "snap.json"
    assert rc.main(["--root", str(tmp_path), "snapshot", "--out", str(out)]) == 0
    snap = json.loads(out.read_text(encoding="utf-8"))
    assert snap["broken_count"] == 1
    (tmp_path / "docs" / "live.md").unlink()
    assert rc.main(["--root", str(tmp_path), "diff", "--before", str(out)]) == 4
    assert "新增断链 1 个" in capsys.readouterr().out
    # 修引用后再对表 ⇒ 归零
    (tmp_path / "a.md").write_text("引用 docs/新位置.md", encoding="utf-8")
    (tmp_path / "docs" / "新位置.md").write_text("# x\n", encoding="utf-8")
    # _share/讨论板.md 的历史引用 live.md 仍在，但落在默认排除面 ⇒ 不算新增
    monkeypatch.setattr(rc, "git_tracked", lambda root: files + ["docs/新位置.md"])
    assert rc.main(["--root", str(tmp_path), "diff", "--before", str(out)]) == 0
