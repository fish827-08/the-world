#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""sign_hook.py —— 提交署名闸门（R393②③，卡 SIGN-HOOK）

一句话：**把"作者位写错"从"纪律问题"变成"提交不出去"**。

判定链（在待提交的工作树里跑）：

| 步 | 取什么 | 来源 |
|---|---|---|
| 1 | 卡号 | 工作树路径匹配 `<仓库根>/.worktrees/<卡号>`（或分支名 `task/<卡号>`） |
| 2 | **应然署名** | 有卡 ⇒ `_share/任务卡.md` 该卡「负责」列经花名册字典反查（**唯一放行依据**）；无卡行 ⇒ 回落本树 `.qoder-sign.env`；署名文件另有用途 = 报"双锁漂移" |
| 3 | **实然署名** | `git var GIT_AUTHOR_IDENT`（= git 本次真要用的作者位） |
| 4 | 裁决 | 见下表 |

裁决口径（🔴 R393③：只在**有卡**时硬拦，主树/无卡一律降级为警告，绝不挡 fish/PI 维护）：

- 有卡 + 作者位 ≠ 应然 ⇒ **拒绝**（rc=2）
- 有卡 + 作者名不在花名册 ⇒ **拒绝**（rc=2）
- 有卡 + 作者位 = `pi@the-world.local`（PI/fish 白名单）⇒ 放行 + 提示（合并/裁定动作本属 PI）
- 无卡 / 主树 ⇒ 只警告（rc=0），打印应然与实然
- 卡负责人 与 本树署名文件 不一致 ⇒ 警告（双锁漂移：卡转人了或树建错了）

退出码：0 放行（含警告）/ 1 用法或环境错误 / 2 拦截。

单独用：`python sign_hook.py check [--root <工作树>]`；装成钩子见 `install_git_hook.py`。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

WORKDIR_NAME = ".worktrees"          # 与 worktree.py 同源
BRANCH_PREFIX = "task/"
CARD_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]{0,63})$")
TASK_CARD_REL = os.path.join("_share", "任务卡.md")
# PI/fish 白名单（R393③）：邮箱是 pi@ 且花名属于 PI/所有者/真人这一串 ⇒ 不硬拦
PI_EMAIL = "pi@the-world.local"
PI_NAMES = {"fish(1)", "fish", "pi", "[pi]", "天平", "[所有者]", "所有者"}

RC_BLOCK = 2


# ---------------- git 探测 ----------------
def _git(root: str, args: list) -> tuple:
    try:
        p = subprocess.run(["git"] + list(args), cwd=root, capture_output=True,
                           text=True, timeout=60, encoding="utf-8", errors="replace")
        return p.returncode, (p.stdout or "").strip(), (p.stderr or "").strip()
    except Exception as e:
        return 127, "", f"{type(e).__name__}: {e}"


def toplevel(root: str) -> str:
    """工作树根。

    🔴 Windows 实测坑：git 的**文件路径**输出走本机 ANSI（GBK），含中文的仓库路径
    用 utf-8 解码会得到乱码 ⇒ 路径不可信。故：解不出来 / 目录不存在 ⇒ 返回空串，
    由调用方回落到 `cwd`（钩子本就跑在工作树根，见 install_git_hook.py）。
    """
    rc, out, _ = _git(root, ["rev-parse", "--show-toplevel"])
    if rc != 0 or not out:
        return ""
    p = os.path.normpath(out)
    return p if os.path.isdir(p) else ""


def is_main_worktree(root: str) -> bool:
    """是否主工作树：`--git-dir` == `--git-common-dir`。

    两条输出同源于 git（同样可能被 ANSI 化），**比较字符串本身**即可，不依赖解码正确。
    """
    d = _git(root, ["rev-parse", "--git-dir"])[1]
    c = _git(root, ["rev-parse", "--git-common-dir"])[1]
    if not d or not c:
        return False
    def norm(x):
        return os.path.normcase(os.path.normpath(x if os.path.isabs(x)
                                             else os.path.join(root, x)))
    return norm(d) == norm(c)


def current_branch(root: str) -> str:
    return _git(root, ["rev-parse", "--abbrev-ref", "HEAD"])[1]


def card_of(tl: str, branch: str = "", main_tree: bool = False) -> str | None:
    """工作树 → 卡号：优先路径 `<...>/.worktrees/<卡号>`，回落分支名 `task/<卡号>`。

    主工作树即使分支叫 task/X 也返回 None（🔴 R393③：主树一律警告级，不挡维护操作）。
    """
    segs = (tl or "").replace("\\", "/").rstrip("/").split("/")
    for i in range(len(segs) - 1, 0, -1):
        if segs[i - 1] == WORKDIR_NAME and CARD_RE.match(segs[i]):
            return segs[i]
    if main_tree:
        return None
    b = (branch or "").strip()
    if b.startswith(BRANCH_PREFIX):
        c = b[len(BRANCH_PREFIX):]
        return c if CARD_RE.match(c) else None
    return None



def author_ident(root: str) -> tuple:
    """git 本次真要用的作者位 (name, email)。`GIT_AUTHOR_IDENT` = Name <email> ts tz。"""
    rc, out, _ = _git(root, ["var", "GIT_AUTHOR_IDENT"])
    if rc != 0:
        return "", ""
    m = re.match(r"^(.*?)\s*<([^>]*)>", out)
    return (m.group(1).strip(), m.group(2).strip()) if m else (out, "")


# ---------------- 应然署名 ----------------
def expected_from_sign_file(tl: str) -> tuple:
    import git_id
    return git_id.load_sign_file(tl)


def owners_from_card_file(repo_root: str, card: str) -> list:
    """`_share/任务卡.md` 该卡的「负责」原文（可能是 `轻舟/澜舟`、`板桥+砚`、`PI·fish(1)`）。"""
    p = os.path.join(repo_root, TASK_CARD_REL)
    if not os.path.isfile(p):
        return []
    with open(p, encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip().startswith("|"):
                continue
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) >= 3 and cells[0] == card:
                return [c.strip() for c in re.split(r"[/／+＋]", cells[2]) if c.strip()]
    return []


def expected_identities(tl: str, card: str | None, main_tree: bool = False) -> dict:
    """应然署名：**卡负责人优先**（R393② 的判据就是"=该卡负责人"）。

    返回 {card_pairs, file, notes, raw}；`file`（本树署名文件）不参与放行判定，
    只用来报"双锁漂移"（卡转人了 / 树建错了）。查不到卡行时 file 才顶上。
    """
    import git_id
    out = {"card_pairs": [], "file": None, "notes": [], "raw": ""}
    if card:
        raw_list = owners_from_card_file(tl, card)
        if not raw_list and not main_tree:
            out["notes"].append(
                f"本树读不到卡 {card} 的「负责」列（{TASK_CARD_REL} 无此行）"
                "⇒ 应然回落到署名文件")
        for raw in raw_list:
            try:
                out["card_pairs"].append(git_id.owner_identity(raw, tl))
            except git_id.GitIdError as e:
                out["notes"].append(f"卡「负责」{raw!r} 花名册查不到：{e}")
        out["raw"] = "/".join(raw_list)
    n, e = expected_from_sign_file(tl)
    if n:
        out["file"] = (n, e or "")
    return out



# ---------------- 裁决（纯函数，可单测） ----------------
def _match(got: tuple, pairs: list) -> bool:
    for n, e in pairs:
        if (got[0] or "").strip() == n and ((not e) or (got[1] or "").strip() == e):
            return True
    return False


def judge(name: str, email: str, card: str | None, expected: dict,
          roster_names: set) -> dict:
    """→ {"level": ok|warn|block, "findings": [...], "fix": [...]}。

    findings 每条 {level, msg}；只有 `hard`（**有卡**）时才产 block 条目。
    """
    hard = card is not None
    findings = []

    def add(level, msg):
        findings.append({"level": level, "msg": msg})

    downgraded = "（无卡/主树 ⇒ 警告级不阻断，R393③）" if not hard else ""
    lvl_block = "block" if hard else "warn"
    card_pairs = [tuple(p) for p in expected.get("card_pairs", [])]
    file_pair = expected.get("file")
    pairs = card_pairs or ([tuple(file_pair)] if file_pair else [])
    source = "任务卡负责人" if card_pairs else ("署名文件" if file_pair else None)

    if not name and not email:
        add("warn", "读不到作者位（git var GIT_AUTHOR_IDENT 失败）" + downgraded)
        lvl = "block" if hard else "warn"
        return {"level": lvl, "findings": findings,
                "fix": FIX_HINTS if lvl == "block" else []}

    got = ((name or "").strip(), (email or "").strip())
    if email == PI_EMAIL or got[0].lower() in PI_NAMES:
        add("ok", f"PI/fish 白名单放行：{name} <{email}>"
                  + (f"（本卡应然={_fmt_pairs(card_pairs)}，裁定/合并动作归 PI）"
                     if card_pairs else ""))
        return {"level": "ok", "findings": findings, "fix": []}

    if got[0] not in roster_names:
        add(lvl_block, f"作者名 {name!r} 不在花名册在册花名里"
                       "（R227：禁匿名/代名提交）" + downgraded)
    if pairs:
        if not _match(got, pairs):
            add(lvl_block, f"{_where(card)}作者位 {name} <{email}> ≠ 应然 "
                           f"{_fmt_pairs(pairs)}（来源：{source}）" + downgraded)
        else:
            add("ok", f"作者位 = 应然（{name} <{email}>，来源 {source}）")
    elif hard:
        add("warn", f"卡 {card} 既无「负责」行也无本树署名文件 ⇒ 无应然值可对照（只查了花名册）")
    else:
        add("warn", "无卡且无署名文件 ⇒ 无应然值可对照" + downgraded)
    for note in expected.get("notes", []):
        add("warn", note)
    if card_pairs and file_pair and not _match(tuple(file_pair), card_pairs):
        add("warn", f"双锁漂移：卡 {card} 负责人 = {_fmt_pairs(card_pairs)}，"
                    f"本树署名文件 = {file_pair[0]} <{file_pair[1]}>"
                    "⇒ 卡转人了或本树建错了，请核对（放行只认卡负责人）")

    blocked = [f for f in findings if f["level"] == "block"]
    return {"level": "block" if blocked else (
                "warn" if any(f["level"] == "warn" for f in findings) else "ok"),
            "findings": findings,
            "fix": FIX_HINTS if blocked else []}



def _where(card):
    return f"[卡 {card}] "


def _fmt_pairs(pairs) -> str:
    return " 或 ".join(f"{n} <{e}>" for n, e in pairs) or "(无)"


FIX_HINTS = [
    "修法三选一：",
    "  a) 本树重设署名（推荐，一次生效）：python.exe collab-toolkit/tools/worktree.py setup <卡号> --owner <本人花名>",
    "  b) 只给本会话注入（零共享写）：eval \"$(python.exe collab-toolkit/tools/git_id.py env --role <角色键>)\"",
    "  c) 逐命令显式署名（R371 老口径）：git -c user.name=<花名> -c user.email=<花名拼音>@the-world.local commit …",
    "  d) 卡已转人：先改 _share/任务卡.md 的「负责」列（task_card.py claim <卡> --owner <新负责人>）再重提",
]


def roster_names(root: str = DEFAULT_ROOT) -> set:
    import git_id
    table = git_id.roster_identities(root)
    return {v[1] for v in table.values() if v[1]} | {v[0] for v in table.values()}


def evaluate(root: str | None = None) -> dict:
    """root 不可用（未给 / 不是目录 / sh 传进来的 POSIX 路径 Windows 解析不了）⇒ 回落 cwd。

    钩子场景 git 的 cwd = 工作树根，python 从 OS 拿到的就是合法路径，最稳。
    """
    cand = os.path.abspath(root) if root else ""
    base = cand if cand and os.path.isdir(cand) else os.getcwd()
    tl = toplevel(base) or base
    main = is_main_worktree(base)
    branch = current_branch(base)
    card = card_of(tl, branch, main)
    exp = expected_identities(tl, card, main)
    name, email = author_ident(base)
    v = judge(name, email, card, exp, roster_names(tl))
    v.update({"toplevel": tl, "branch": branch, "card": card,
              "author": {"name": name, "email": email},
              "expected": {**exp, "pairs": (exp["card_pairs"] or
                                            ([exp["file"]] if exp["file"] else [])),
                          "source": ("任务卡负责人" if exp["card_pairs"]
                                     else ("署名文件" if exp["file"] else None))},
              "main_tree": main})
    return v



# ---------------- 输出 ----------------
_ICON = {"ok": "  ✅", "warn": "  ⚠️", "block": "🔴 ⛔"}


def print_verdict(v: dict, stream=None) -> None:
    out = stream or sys.stdout
    where = v["card"] or ("主树" if v["main_tree"] else "无卡")
    out.write(f"[署名闸门] 位置={where} 分支={v['branch'] or '?'} "
              f"作者={v['author']['name'] or '?'} <{v['author']['email'] or '?'}>\n")
    for f in v["findings"]:
        out.write(f"{_ICON.get(f['level'], '  ·')} {f['msg']}\n")
    if v["level"] == "block":
        out.write("\n".join("[拒绝] " + h if i == 0 else h for i, h in enumerate(v["fix"])) + "\n")
    out.flush()


def cmd_check(args) -> int:
    v = evaluate(args.root)
    if args.json:
        print(json.dumps(v, ensure_ascii=False, indent=2))
    else:
        print_verdict(v, sys.stderr if v["level"] == "block" else sys.stdout)
    if v["level"] == "block":
        return RC_BLOCK
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="sign_hook.py",
                                 description="提交署名闸门：作者位∈花名册 且 =本卡负责人（R393）")
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("check", parents=[], help="校验当前工作树的作者位（钩子调的就是它）")
    p.add_argument("--root", default=DEFAULT_ROOT)
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_check)
    args = ap.parse_args(argv)
    if not getattr(args, "fn", None):
        ap.print_help()
        return 1
    try:
        return args.fn(args)
    except Exception as e:  # 钩子里异常 ⇒ 不静默放行（fail-loud），但也不误伤：给 rc=1 让人看见
        sys.stderr.write(f"[署名闸门·异常] {type(e).__name__}: {e}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
