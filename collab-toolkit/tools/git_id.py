#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""git_id.py —— 署名注入与提交前校验（R371① / 卡 WORKTREE-SIGN）

一句话：把"作者位 = 本人花名"从**每条命令手敲 `-c`** 变成**开工设一次、之后自动带**。

🔴 为什么不用 `git config`（这是本工具存在的理由）：
本仓未开 `extensions.worktreeConfig` ⇒ `git config user.name` 写的是**共享的 common
config**，同机多工作树互相覆写 —— A 会话设板桥、B 会话设砚，谁最后写谁赢。
R371 立法的事故正文就是这个（"作者位错挂他人"）。

⇒ 改走 git 官方的**环境变量配置注入**（`GIT_CONFIG_COUNT` / `GIT_CONFIG_KEY_n` /
`GIT_CONFIG_VALUE_n`）：**进程级、零共享写、优先级高于一切配置文件**，
对 `git commit` / `git log` 及一切子进程命令一律生效。

三个命令：
  show    [--root R] [--role KEY]   三方对照：应然署名 / 当前生效署名 / 工作树署名文件
  env     --role KEY                打印可 source 的 export 行（POSIX sh）
  verify  [--root R] [--role KEY] [ref]
                                    校验 ref 的作者位 = 应然花名；不符 ⇒ rc=2 fail-loud

退出码：0 通过 / 1 用法或环境错误 / 2 校验不符（署名漂移）。
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
DEFAULT_ROOT = os.path.dirname(os.path.dirname(_HERE))
SIGN_FILENAME = ".qoder-sign.env"
EMAIL_RE = re.compile(r"^[A-Za-z0-9._+-]+@the-world\.local$")


class GitIdError(Exception):
    def __init__(self, msg: str, code: int = 1):
        super().__init__(msg)
        self.code = code


# ---------------- 应然署名（来自 onboard.ROLE_REGISTRY） ----------------
def role_identity(role_key: str, root: str = DEFAULT_ROOT) -> tuple:
    """角色键 → (花名, 邮箱)。花名未定（如 [联网]"待取"）⇒ 报错，不猜邮箱。"""
    sys.path.insert(0, _HERE)
    import onboard  # 同目录工具，延迟导入避免循环
    reg = onboard.ROLE_REGISTRY
    if role_key not in reg:
        raise GitIdError(f"未知角色键 {role_key!r}；可选项：{', '.join(sorted(reg))}")
    name = reg[role_key].get("花名") or ""
    email = reg[role_key].get("邮箱") or ""
    if not name or name == "待取":
        raise GitIdError(f"角色 {role_key}（{reg[role_key].get('署名')}）花名未定 ⇒ "
                         f"无法生成署名；请先在花名册定名，或用 --name/--email 显式给")
    if not EMAIL_RE.match(email):
        raise GitIdError(f"角色 {role_key} 的邮箱字段不合规：{email!r}"
                         f"（应为 <花名拼音>@the-world.local；R370 身份纪律）")
    return name, email


def all_role_emails(root: str = DEFAULT_ROOT) -> dict:
    sys.path.insert(0, _HERE)
    import onboard
    return {k: (v.get("花名"), v.get("邮箱")) for k, v in onboard.ROLE_REGISTRY.items()}


# ---------------- 注入（env）与落盘（署名文件） ----------------
def env_pairs(name: str, email: str) -> dict:
    """git 官方 env 配置注入：等价于所有命令自动带 `-c user.name=… -c user.email=…`。"""
    return {
        "GIT_CONFIG_COUNT": "2",
        "GIT_CONFIG_KEY_0": "user.name",
        "GIT_CONFIG_VALUE_0": name,
        "GIT_CONFIG_KEY_1": "user.email",
        "GIT_CONFIG_VALUE_1": email,
    }


def shell_lines(name: str, email: str) -> str:
    lines = [f"# 本工作树署名 = {name} <{email}>（R371①；开工第一步 source 本文件）"]
    for k, v in env_pairs(name, email).items():
        lines.append(f'export {k}="{v}"')
    lines.append("# PowerShell 请用:  Get-Content .qoder-sign.env | ForEach-Object { … }")
    lines.append("#   或直接: git -c user.name=" + name + " -c user.email=" + email + " commit …")
    return "\n".join(lines) + "\n"


def sign_file_path(root: str) -> str:
    return os.path.join(root, SIGN_FILENAME)


def write_sign_file(root: str, name: str, email: str) -> str:
    p = sign_file_path(root)
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(shell_lines(name, email))
    return p


def load_sign_file(root: str) -> tuple:
    """读工作树署名文件 → (name, email)；不存在返回 (None, None)。"""
    p = sign_file_path(root)
    if not os.path.isfile(p):
        return None, None
    got = {}
    with open(p, encoding="utf-8") as f:
        for line in f:
            m = re.match(r'\s*export\s+(\w+)="([^"]*)"', line)
            if m:
                got[m.group(1)] = m.group(2)
    name = got.get("GIT_CONFIG_VALUE_0") if got.get("GIT_CONFIG_KEY_0") == "user.name" else None
    email = got.get("GIT_CONFIG_VALUE_1") if got.get("GIT_CONFIG_KEY_1") == "user.email" else None
    return name, email


def apply_to_process(name: str, email: str) -> None:
    """把署名注入**当前进程**环境（工具自身跑 git 时就带上，免逐命令 -c）。"""
    os.environ.update(env_pairs(name, email))


# ---------------- 实际署名（读 git） ----------------
def _git(root: str, args: list, env: dict | None = None) -> tuple:
    try:
        r = subprocess.run(["git"] + list(args), cwd=root, capture_output=True,
                           text=True, timeout=60, encoding="utf-8",
                           errors="replace", env=env)
        return r.returncode, (r.stdout or "").strip(), (r.stderr or "").strip()
    except Exception as e:
        return 127, "", f"{type(e).__name__}: {e}"


def config_identity(root: str, name: str | None = None,
                    email: str | None = None) -> tuple:
    """读"此刻真生效"的 user.name/user.email：给了应然署名就按其 env 读（env 优先级最高），
    否则读配置文件（可能被同机其他工作树覆写 —— 这正是 R371 的病灶）。"""
    env = None
    if name and email:
        env = dict(os.environ)
        env.update(env_pairs(name, email))
    return (_git(root, ["config", "user.name"], env)[1],
            _git(root, ["config", "user.email"], env)[1])


def author_of(root: str, ref: str = "HEAD") -> tuple:
    rc, out, _ = _git(root, ["log", "-1", "--format=%an%x1f%ae", ref])
    if rc != 0 or not out:
        return None, None
    parts = out.split("\x1f")
    return parts[0], parts[1] if len(parts) > 1 else ""


def worktree_config_scope(root: str) -> str:
    """署名会落到哪：worktreeConfig 开 = 各树独立；关 = 共享 common config。"""
    rc, out, _ = _git(root, ["config", "--get", "extensions.worktreeConfig"])
    return "worktree（各树独立）" if rc == 0 and out == "true" else "common（同机多树共享，会互相覆写）"


# ---------------- 校验 ----------------
def verify_identity(root: str, expect: tuple, ref: str = "HEAD") -> list:
    """返回 findings 列表（level/name/expect/actual）。expect=(name,email)，None 位不查。"""
    name, email = author_of(root, ref)
    out = []
    for field, want, got in (("作者名", expect[0], name), ("作者邮箱", expect[1], email)):
        if want is None or want == "":
            continue
        if got != want:
            out.append({"level": "error", "item": "sign", "field": field,
                        "expect": want, "actual": got or "(读不到)",
                        "msg": f"{ref} {field}={got or '(读不到)'} 应={want}"})
        else:
            out.append({"level": "ok", "item": "sign", "field": field,
                        "expect": want, "actual": got,
                        "msg": f"{ref} {field}={got} ✅"})
    return out


# ---------------- 子命令 ----------------
def _expect_from_args(args) -> tuple:
    if getattr(args, "role", None):
        return role_identity(args.role, args.root)
    n, e = load_sign_file(args.root)
    if n or e:
        return n, e
    if getattr(args, "name", None) or getattr(args, "email", None):
        return args.name, args.email
    return None, None


def cmd_show(args) -> int:
    root = os.path.abspath(args.root)
    expect = _expect_from_args(args)
    cn, ce = config_identity(root, *expect)
    fn, fe = load_sign_file(root)
    an, ae = author_of(root)
    print(f"仓库根: {root}")
    print(f"署名落点: {worktree_config_scope(root)}"
          f"（⇒ 生效值按应然 env 读，不共享覆写）")
    print(f"  应然（--role/署名文件）: {expect[0] or '(未知)'} <{expect[1] or '(未知)'}>")
    print(f"  git config 生效值      : {cn or '(未设置)'} <{ce or '(未设置)'}>")
    print(f"  本树署名文件           : " +
          (f"{fn} <{fe}>" if fn else f"（无 {SIGN_FILENAME}）"))
    print(f"  HEAD 作者位            : {an or '(无提交)'} <{ae or ''}>")
    if expect[0] and (an, ae) != (expect[0], expect[1]) and an is not None:
        print("  ⚠️ HEAD 作者位与应然不符 —— 见 `verify` 明细")
        return 2
    return 0


def cmd_env(args) -> int:
    name, email = role_identity(args.role, os.path.abspath(args.root))
    print(shell_lines(name, email), end="")
    return 0


def cmd_verify(args) -> int:
    root = os.path.abspath(args.root)
    expect = _expect_from_args(args)
    if not expect[0]:
        sys.stderr.write("[错误] 无从判断应然署名：给 --role <键> / 本树署名文件 / "
                         "--name --email 三者之一（fail-loud，不静默放行）\n")
        return 1
    findings = verify_identity(root, expect, args.ref)
    for f in findings:
        print(("  " if f["level"] == "ok" else "🔴 ") + f["msg"])
    bad = [f for f in findings if f["level"] == "error"]
    if bad:
        sys.stderr.write(f"[拒绝] 署名漂移 {len(bad)} 处 —— 修法："
                         f"`source {SIGN_FILENAME}`（或逐命令 "
                         f"`git -c user.name=… -c user.email=… commit`）后重跑\n")
        return 2
    return 0


def main(argv=None) -> int:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--root", default=DEFAULT_ROOT)

    ap = argparse.ArgumentParser(prog="git_id.py",
                                 description="worktree 署名注入与提交前校验（R371①）")
    sub = ap.add_subparsers(dest="cmd")

    p = sub.add_parser("show", parents=[common], help="应然/生效/作者位 三方对照")
    p.add_argument("--role", default=None)
    p.add_argument("--name", default=None)
    p.add_argument("--email", default=None)
    p.set_defaults(fn=cmd_show)

    p = sub.add_parser("env", parents=[common], help="打印 export 行（可 source）")
    p.add_argument("--role", required=True)
    p.set_defaults(fn=cmd_env)

    p = sub.add_parser("verify", parents=[common], help="校验 ref 作者位 = 应然署名")
    p.add_argument("--role", default=None)
    p.add_argument("--name", default=None)
    p.add_argument("--email", default=None)
    p.add_argument("ref", nargs="?", default="HEAD")
    p.set_defaults(fn=cmd_verify)

    args = ap.parse_args(argv)
    if not getattr(args, "fn", None):
        ap.print_help()
        return 1
    try:
        return args.fn(args)
    except GitIdError as e:
        sys.stderr.write(f"[错误] {e}\n")
        return e.code


if __name__ == "__main__":
    sys.exit(main())
