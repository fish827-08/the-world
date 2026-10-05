#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""worktree.py —— 任务卡隔离工作树（R359 B1，蓝本 rag-kb/orchestra/worktree.py）

一句话：**把"两个人别碰同一份工作树"从纪律变成物理隔离**。

三个命令：
  setup <卡号>   建隔离目录 `.worktrees/<卡号>` + 检出 `task/<卡号>` 分支
                 （🔴 R370：建完必打印当前署名并提醒先设本人花名；
                   R393①：带 `--owner <花名>`（或 `--role <键>`）时自动写
                   本树署名文件 + 本树独占 config.worktree ⇒ 之后免逐命令 `-c`；
                   带 `--name <花名> --email <邮箱>` 时写 git config）
  enter <卡号>   只打印该卡的隔离目录路径（给别的会话/脚本用）
  clean <卡号>   清理工作树（**提交保留在分支上**，分支不删）

🔴 the-world 四点适配（照搬会出事，见派工单 R359 §二）：

| # | 适配点 | 为什么 |
|---|--------|--------|
| 1 | 隔离目录 = **`.worktrees/`** | R342 已把 5 个 `_wt_*` 旧冻结树收编进 `_archive/` ⇒ 避免命名撞车 |
| 2 | 🔴 显式打印 **主仓 `.venv` 解释器和 `--python` 覆盖口** | worktree 内**没有** `.venv`、也没有 `sim_core.so` ⇒ worker 必须知道去哪跑 python（统一 `.venv/Scripts/python.exe`）|
| 3 | 🔴 **不实现 `prune`** | `AGENT.md:493` 全员禁用 `git worktree prune`（2026-09-12 事故）⇒ 只做 setup/enter/clean |
| 4 | `.worktrees/` 入 `.gitignore`（见主仓 `.gitignore` 协作线段）| 否则污染 `git status`（= R343 教训：游离文件淹没真改动）|

隔离证明：`setup` 之后，主工作区 `git checkout task/<卡号>` 会被 git 拒绝
（"already exists"），因为该分支已被 worktree 占用。

退出码：0 成功 / 1 使用错误 / 2 冲突或已存在（setup 幂等复用时仍为 0）。
"""
from __future__ import annotations

import argparse
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
DEFAULT_ROOT = os.path.dirname(os.path.dirname(_HERE))  # collab-toolkit/tools -> the-world/

# 卡号白名单：防命令注入（git 参数直接拼命令）
CARD_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]{0,63})$")

# 🔴 适配点 1：隔离目录固定 `.worktrees/`（不叫 `_wt_*`）
WORKDIR_NAME = ".worktrees"
BRANCH_PREFIX = "task/"

MSG_NO_PRUNE = (
    "本工具**不提供** prune —— `AGENT.md:493` 全员禁用 `git worktree prune`"
    "（2026-09-12 事故：误删他人冻结树）。隔离目录靠 `.gitignore` 兜底，"
    "长期不用的 clean 由人工在确认无分支引用后 `git branch -D`。"
)


class WorktreeError(Exception):
    def __init__(self, msg: str, code: int = 1):
        super().__init__(msg)
        self.code = code


# ---------------- 纯函数（可单测，不碰 git） ----------------
def plan_paths(root: str, card: str) -> dict:
    """卡号 → 目录 / 分支名的映射。"""
    return {"dir": os.path.join(root, WORKDIR_NAME, card),
            "branch": BRANCH_PREFIX + card}


def check_card(card: str) -> str:
    """卡号格式校验（白名单），非法直接抛错。"""
    if not CARD_RE.match(card or ""):
        raise WorktreeError(
            f"卡号不合法: {card!r}（只允许字母数字, 点, 下划线, 连字符；"
            f"必须以字母数字开头）—— 防命令注入")
    return card


def resolve_python(root: str, override: str | None = None) -> tuple:
    """解析 worktree 内应该用的解释器。

    返回 (exe, exists, why)。🔴 适配点 2：**一律指向主仓 .venv**，
    因为 worktree 里没有 `.venv` / `sim_core.so`。裸机场景（R249）不存在
    `.venv` ⇒ exists=False，仍如实返回并给出提示（**fail-loud，不静默回落到
    sys.executable** —— R249 的教训："宣称'检测不到自动走 X'前先验证'检测'成立"）。
    """
    if override:
        return override, os.path.isfile(override), "--python 指定"
    if os.name == "nt":
        rel = os.path.join(".venv", "Scripts", "python.exe")
    else:
        rel = os.path.join(".venv", "bin", "python")
    exe = os.path.join(root, rel)
    return exe, os.path.isfile(exe), "主仓 .venv（worktree 内无 .venv，见 R249）"


def plan_identity(name: str | None, email: str | None, role: str | None = None,
                  owner: str | None = None) -> dict:
    """纯函数：setup 后怎么定署名（R370 身份纪律 / R227 / R371① / R393①）。

    - `--owner <花名>` → `owner`：**花名册字典反查**邮箱，写两处 = 本树署名文件
      `.qoder-sign.env` + 本树独占 `config.worktree`（之后本树所有 git 命令自动带署名）
    - `--role <键>` → `role`：同上，但按角色键查（onboard.ROLE_REGISTRY）
    - `--name` + `--email` 都给 → `set`：写 `git config`（共享落点，只在明确要设
      树/机默认身份时用）
    - 只给一半 → `bad`（不猜邮箱：中文花名拼不出邮箱，宁可停下问）
    - 都没给 → `ask`（打印提醒，不静默回落）
    """
    if owner and role:
        return {"action": "bad",
                "msg": f"--owner 与 --role 二选一（收到 owner={owner!r} role={role!r}）"}
    if owner:
        return {"action": "owner", "role": owner, "name": name, "email": email}
    if role:
        return {"action": "role", "role": role, "name": name, "email": email}
    if name and email:
        return {"action": "set", "name": name, "email": email}
    if name or email:
        return {"action": "bad",
                "msg": f"--name/--email 必须成对给（收到 name={name!r} email={email!r}）"}
    return {"action": "ask"}


def resolve_identity(root: str, plan: dict) -> tuple:
    """`role`/`owner` 计划 → (花名, 邮箱)。--name/--email 可覆盖注册表值。"""
    sys.path.insert(0, _HERE)
    import git_id
    if plan["action"] == "owner":
        name, email = git_id.owner_identity(plan["role"], root)
    else:
        name, email = git_id.role_identity(plan["role"], root)
    return plan.get("name") or name, plan.get("email") or email


def plan_setup(root: str, card: str) -> dict:
    """setup 的**计划**（纯函数）：将要执行的 git 命令 + 各处存在性。

    `cmd=None` 表示无需建分支（复用已有 branch 或已有 worktree）。
    """
    paths = plan_paths(root, card)
    wt_exists = os.path.isdir(os.path.join(paths["dir"], ".git")) or \
        os.path.isfile(os.path.join(paths["dir"], ".git"))
    wt_dir_exists = os.path.isdir(paths["dir"])
    r = run_git_capture(root, ["rev-parse", "--verify", "refs/heads/" + paths["branch"]])
    branch_exists = r[0] == 0
    cmd = None
    if not wt_exists:
        if branch_exists:
            cmd = ["worktree", "add", paths["dir"], paths["branch"]]
        else:
            cmd = ["worktree", "add", "-b", paths["branch"], paths["dir"]]
    return {"card": card, "paths": paths, "wt_exists": wt_exists,
            "wt_dir_exists": wt_dir_exists, "branch_exists": branch_exists,
            "cmd": cmd}


def status_line(root: str, card: str) -> dict:
    """enter 用的轻量探测：目录在不在、分支在不在、干净不干净。"""
    paths = plan_paths(root, card)
    wt = os.path.isdir(os.path.join(paths["dir"], ".git")) or \
        os.path.isfile(os.path.join(paths["dir"], ".git"))
    r = run_git_capture(root, ["rev-parse", "--verify", "refs/heads/" + paths["branch"]])
    return {"card": card, "dir": paths["dir"], "branch": paths["branch"],
            "worktree": wt, "branch": r[0] == 0}


# ---------------- git 调用（测试可 monkeypatch） ----------------
def run_git_capture(root: str, args: list) -> tuple:
    try:
        p = subprocess.run(["git"] + list(args), cwd=root,
                           capture_output=True, text=True, timeout=60,
                           encoding="utf-8", errors="replace")
        return p.returncode, (p.stdout or "").strip(), (p.stderr or "").strip()
    except Exception as e:  # 无 git / 环境异常
        return 127, "", f"{type(e).__name__}: {e}"


def run_git_print(root: str, args: list) -> int:
    rc, out, err = run_git_capture(root, args)
    if rc == 0:
        if out:
            print("   " + out)
        return 0
    sys.stderr.write("   git 失败(rc=%d): %s\n" % (rc, (err or out or "")[:300]))
    return rc


def apply_identity(wt_dir: str, name: str, email: str) -> int:
    """写入 git 署名（R370 身份纪律）。

    落点 = 仓库 local config（`config.worktree` 需 `extensions.worktreeConfig`，
    本仓未开 ⇒ 同机多工作树**共享**这份署名）。因此切身份时每次 setup 都要重设，
    别让上一个人的署名替你背锅。
    """
    for key, val in (("user.name", name), ("user.email", email)):
        rc, _, err = run_git_capture(wt_dir, ["config", key, val])
        if rc != 0:
            sys.stderr.write(f"  [署名写入失败] git config {key} rc={rc}: {err[:200]}\n")
            return rc
    got_name = run_git_capture(wt_dir, ["config", "user.name"])[1]
    got_mail = run_git_capture(wt_dir, ["config", "user.email"])[1]
    print(f"  [署名已生效] user.name={got_name or name} user.email={got_mail or email}")
    print("           （落点 = 仓库 local config，同机多工作树共享 ⇒ 换人开工必须重设）")
    return 0


def current_identity(wt_dir: str) -> tuple:
    """读该工作树当前生效的 git 署名（读不到返回空串，不抛）。"""
    return (run_git_capture(wt_dir, ["config", "user.name"])[1],
            run_git_capture(wt_dir, ["config", "user.email"])[1])


def apply_tree_identity(wt_dir: str, name: str, email: str) -> dict:
    """把署名写进**该工作树独占**的 `config.worktree`（R393①：自动带、免手敲 -c）。

    🔴 为什么不用裸 `git config`：裸写落到**共享 common config**（同机多树互相覆写
    = R371 病灶）。`--worktree` 是 git 官方的每树独立落点，但 git **不会自动开开关**
    （实测 rc=128 "–worktree cannot be used … unless the config extension worktreeConfig
    is enabled"）⇒ 需先 `git config extensions.worktreeConfig true`，这是一次
    **共享开关写**（只增强隔离、不改他人已有的值）⇒ 本函数如实回报开关前后状态，
    并可用 setup `--no-tree-config` 关掉。

    返回 dict(ok/rc/why/wrote/scope_before/scope_after/effective/msg)。
    """
    keys = (("user.name", name), ("user.email", email))
    scope_before = run_git_capture(wt_dir, ["config", "--get", "extensions.worktreeConfig"])[1]
    wrote, why, rc = {}, "", 0
    for key, val in keys:
        r = run_git_capture(wt_dir, ["config", "--worktree", key, val])
        if r[0] != 0 and "worktreeConfig" in (r[1] + r[2]):
            e = run_git_capture(wt_dir, ["config", "extensions.worktreeConfig", "true"])
            if e[0] != 0:
                return {"ok": False, "rc": e[0], "wrote": wrote, "why": (e[2] or e[1])[:200],
                        "scope_before": scope_before or "false", "scope_after": "false",
                        "effective": ("", ""), "msg": "开关开启失败（common config 不可写？）"}
            r = run_git_capture(wt_dir, ["config", "--worktree", key, val])
        if r[0] != 0:
            rc, why = r[0], (r[2] or r[1])[:200]
            break
        wrote[key] = val
    scope_after = run_git_capture(wt_dir, ["config", "--get", "extensions.worktreeConfig"])[1]
    got = current_identity(wt_dir)
    ok = rc == 0 and len(wrote) == len(keys) and got == (name, email)
    return {"ok": ok, "rc": rc, "why": why, "wrote": wrote,
            "scope_before": scope_before or "false", "scope_after": scope_after or "false",
            "effective": got,
            "msg": "" if ok else (why or "写后读回不符：%r/%r 应 %r/%r"
                                 % (got[0], got[1], name, email))}



# ---------------- 子命令 ----------------
def cmd_setup(args) -> int:
    root = os.path.abspath(args.root)
    card = check_card(args.card)
    plan = plan_setup(root, card)
    exe, exists, why = resolve_python(root, args.python)

    print(f"隔离工作树 setup —— 卡号={card}")
    print(f"  隔离目录: {os.path.relpath(plan['paths']['dir'], root) or plan['paths']['dir']}")
    print(f"  分支    : {plan['paths']['branch']}")
    print(f"  解释器  : {exe}（{why}；存在={exists}）")
    if not exists:
        print("  [警告] 该路径不存在 —— 裸机/无 venv 环境（R249）；"
              "跑批请用 `--python` 显式指定，别回落到默认值")
    if plan["wt_exists"]:
        print("  [复用] 该卡的工作树已存在 —— 幂等返回，不重建（不删数据）")
    elif plan["cmd"] is None:
        print("  [异常] 目录既不是 worktree 又无需建分支 —— 放弃（不猜）")
        return 1
    else:
        print("  → 执行: git " + " ".join(plan["cmd"]))
        rc = run_git_print(root, plan["cmd"])
        if rc != 0:
            print("  [错误] worktree 建立失败（多半是分支已被主工作区占用）")
            return 1
        print("  [隔离已生效] 主工作区 `git checkout " + plan["paths"]["branch"] +
              "` 会被 git 拒绝（分支已被本 worktree 占用）")
    print("-" * 60)
    ident = plan_identity(args.name, args.email, getattr(args, "role", None),
                          getattr(args, "owner", None))
    if ident["action"] == "bad":
        sys.stderr.write(f"  [拒绝] {ident['msg']} —— 中文花名拼不出邮箱，不猜\n")
        return 1
    if ident["action"] in ("role", "owner"):
        sys.path.insert(0, _HERE)
        import git_id
        flag = ("--owner " if ident["action"] == "owner" else "--role ") + ident["role"]
        try:
            name, email = resolve_identity(root, ident)
        except git_id.GitIdError as e:
            sys.stderr.write(f"  [拒绝] {e}\n")
            return e.code
        p = git_id.write_sign_file(plan["paths"]["dir"], name, email)
        print(f"  [本树署名文件已生成] {p}")
        print(f"     应然署名 = {name} <{email}>（花名册字典，来源 {flag}）")
        if getattr(args, "no_tree_config", False):
            print("     [跳过] 未写本树 git config（--no-tree-config）⇒ 靠 env 注入："
                  "eval \"$(python.exe "
                  f"{os.path.join(_HERE, 'git_id.py')} env --role {ident['role']})\"")
        else:
            res = apply_tree_identity(plan["paths"]["dir"], name, email)
            if res["ok"]:
                print(f"     [本树 git config 已写入 config.worktree] "
                      f"user.name={name} user.email={email}")
                print("     ⇒ 本树内**所有** git 命令作者位自动 = 本人花名，免逐命令 -c（R393①）")
                if res["scope_before"] != "true" and res["scope_after"] == "true":
                    print("     ⚠️ 为此打开了**共享**开关 extensions.worktreeConfig=true"
                          "（只增强隔离、不改他人已有值）")
                    print("        回退：git -C "
                          f"{root} config --unset extensions.worktreeConfig")
                print("     校验：python "
                      f"{os.path.join(_HERE, 'git_id.py')} verify --root {plan['paths']['dir']}")
            else:
                print(f"     🔴 [未生效] 本树 config.worktree 写入失败 rc={res['rc']}："
                      f"{res['why'] or res['msg']}")
                print("        ⇒ 回落到 env 注入（零共享写，效果同逐命令 -c）："
                      "eval \"$(python.exe "
                      f"{os.path.join(_HERE, 'git_id.py')} env --role {ident['role']})\"")
        print("     🔴 署名闸门（R393②，强烈建议同机装一次）：python.exe "
              f"{os.path.join(_HERE, 'install_git_hook.py')} install")
    elif ident["action"] == "set":

        rc = apply_identity(plan["paths"]["dir"], args.name, args.email)
        if rc != 0:
            return 1
    else:
        cur_name, cur_mail = current_identity(plan["paths"]["dir"])
        print(f"  当前署名: user.name={cur_name or '(未设置)'} "
              f"user.email={cur_mail or '(未设置)'}")
        print("  🔴 R370 身份纪律：worktree 内开工第一步先设本人署名（别让他人替你背锅）——")
        print(f"     git -C {plan['paths']['dir']} config user.name <本人花名> && "
              f"git -C {plan['paths']['dir']} config user.email <花名拼音>@the-world.local")
        print("     或本次 setup 直接带 --role <角色键>（推荐：写本树署名文件，不碰共享 config）")
        print("     或带 --name <花名> --email <邮箱>（脚本代写 git config）")
    print(MSG_NO_PRUNE)
    print(f"常用解释器: {exe}")
    return 0


def cmd_enter(args) -> int:
    root = os.path.abspath(args.root)
    card = check_card(args.card)
    st = status_line(root, card)
    if not st["worktree"]:
        sys.stderr.write(f"[错误] 该卡没有隔离工作树: {st['dir']}\n"
                         f"       先跑 `worktree.py setup {card}`\n")
        return 1
    print(st["dir"])
    exe, exists, _ = resolve_python(root, args.python)
    print(f"（分支 {st['branch']}；解释器 {exe} exists={exists}）", file=sys.stderr)
    return 0


def cmd_clean(args) -> int:
    """清理工作树；**提交保留在分支上**（不删分支、不删数据，F-R10/R-R10）。"""
    root = os.path.abspath(args.root)
    card = check_card(args.card)
    paths = plan_paths(root, card)
    if not os.path.isdir(paths["dir"]):
        print(f"[复用/跳过] 目录不存在: {os.path.relpath(paths['dir'], root)}")
        return 0
    dirty = run_git_capture(root, ["-C", paths["dir"], "status", "--porcelain"])[1]
    if dirty and not args.force:
        print(f"[拒绝] 该工作树有未提交改动（{len(dirty.splitlines())} 处）⇒ "
              f"先提交，或加 `--force`（改动会随目录一起消失）")
        return 2
    rc = run_git_capture(root, ["worktree", "remove", paths["dir"], "--force"])
    if rc[0] != 0:
        sys.stderr.write("  git worktree remove 失败(rc=%d): %s\n" % (rc[0], rc[2][:300]))
        return 1
    print(f"已清理: {os.path.relpath(paths['dir'], root)} —— "
          f"分支 {paths['branch']} 与其中的提交**保留**（clean 不删数据）")
    if not args.force:
        print("（提示：本工具无 prune；分支清理请人工确认引用后 `git branch -D`）")
    return 0


def main(argv=None) -> int:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--root", default=DEFAULT_ROOT,
                        help="仓库根（默认自动探测 = collab-toolkit 的上一级）")
    ap = argparse.ArgumentParser(
        prog="worktree.py",
        description="任务卡隔离工作树 setup/enter/clean（R359 B1；🔴 无 prune）")
    sub = ap.add_subparsers(dest="cmd")

    p = sub.add_parser("setup", parents=[common], help="建隔离目录 + 检出 task/<卡号>")
    p.add_argument("card", help="卡号，如 R359-B1")
    p.add_argument("--python", default=None,
                   help="显式指定解释器（默认指向主仓 .venv；裸机必给）")
    p.add_argument("--name", default=None,
                   help="本人花名（R370：给了就代写 git config user.name；须与 --email 成对）")
    p.add_argument("--email", default=None,
                   help="署名邮箱，如 banqiao@the-world.local（须与 --name 成对）")
    p.add_argument("--role", default=None,
                   help="角色键（如 collab/dev/eval），花名与邮箱取自 onboard.ROLE_REGISTRY；"
                        "写本树署名文件 .qoder-sign.env + 本树 config.worktree（R393①）")
    p.add_argument("--owner", default=None,
                   help="负责人**花名**（如 板桥/轻舟/砚），按花名册字典反查邮箱 —— 与 --role 二选一")
    p.add_argument("--no-tree-config", action="store_true",
                   help="只写署名文件，不碰 config.worktree（不触发 extensions.worktreeConfig 开关）")
    p.set_defaults(fn=cmd_setup)

    p = sub.add_parser("enter", parents=[common], help="打印隔离目录路径")
    p.add_argument("card", help="卡号")
    p.add_argument("--python", default=None)
    p.set_defaults(fn=cmd_enter)

    p = sub.add_parser("clean", parents=[common], help="清理工作树（提交保留在分支上）")
    p.add_argument("card", help="卡号")
    p.add_argument("--force", action="store_true", help="允许丢弃未提交改动")
    p.set_defaults(fn=cmd_clean)

    args = ap.parse_args(argv)
    if not getattr(args, "fn", None):
        ap.print_help()
        return 1
    try:
        return args.fn(args)
    except WorktreeError as e:
        sys.stderr.write("[错误] %s\n" % e)
        return e.code


if __name__ == "__main__":
    sys.exit(main())
