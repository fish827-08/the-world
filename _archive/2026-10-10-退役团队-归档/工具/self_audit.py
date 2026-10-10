#!/usr/bin/env python
"""自审工具（**云端开发用**）—— 把"过去由外复核发现的问题"机械化。

依据：`docs/tasks/协议-云端自审与自优化-20260924.md`（R194）。
用途：在"本地不再逐阶段复核"的条件下，让云端**自己**把七类缺陷查出来。

用法（在 the-world 仓库根目录）：
    python tools/self_audit.py all                 # c9 + cli + git（默认，全部只读）
    python tools/self_audit.py c9                  # C9：文档声明 vs 代码实际
    python tools/self_audit.py cli                 # preset 参数 vs 目标脚本 argparse
    python tools/self_audit.py git                 # 交付卫生（未提交/未推送/分支/main 禁推）
    python tools/self_audit.py digest              # C7 关档逐位等价（跑基线测试文件）
    python tools/self_audit.py pack --phase S1 --include _rerun_logs/smoke
    python tools/self_audit.py pack --phase S1 --include "experiments/*.txt" --note "S1 冒烟"

设计原则
--------
1. **只读**（`pack` 除外，它只写 `_audit/`）—— 自审工具本身不得改变被测状态。
2. **机械可判**：每条检查给出「命中列表 + 计数」，不写"请仔细检查"。
3. **fail-loud**：任一检查命中 ⇒ 退出码 1（便于 CI/脚本串联）；`pack` 恒 0。
4. **误报优先于漏报**：拿不准的标 `⚠️ 待人工确认`，不假装是绿的。
"""

from __future__ import annotations

import argparse
import glob as _glob
import hashlib
import importlib.util
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
C7_BASELINE = "(574687, 11266.746993)"          # 仅用于"找不到基线"的提示串（真实值从测试里抓）
CODE_DIRS = ("simulation", "world", "experiments", "tests")
SKIP_DIR_PARTS = ("__pycache__", ".git", ".venv", "_backup", "_archive", "_gitee_review",
                  "_reclone", "_pt_h", "_rerun_logs", "_audit", "_trash_local")


# --------------------------------------------------------------------------- 基础

def rel(p: Path) -> str:
    try:
        return str(p.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(p)


def iter_code_files():
    """遍历被审计的代码文件（`.py`），跳过备份/产物目录。"""
    for d in CODE_DIRS:
        base = ROOT / d
        if not base.is_dir():
            continue
        for p in base.rglob("*.py"):
            if any(part in SKIP_DIR_PARTS for part in p.parts):
                continue
            yield p


_WORD_CACHE: dict[str, list[str]] = {}


def find_identifier(name: str) -> list[str]:
    """全代码树里搜标识符（词边界），返回命中的相对路径（去重、排序）。"""
    if name in _WORD_CACHE:
        return _WORD_CACHE[name]
    pat = re.compile(r"(?<![A-Za-z0-9_])" + re.escape(name) + r"(?![A-Za-z0-9_])")
    hits: list[str] = []
    for p in iter_code_files():
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if pat.search(txt):
            hits.append(rel(p))
    hits = sorted(set(hits))
    _WORD_CACHE[name] = hits
    return hits


def _table(rows: list[list[str]], header: list[str]) -> str:
    widths = [len(h) for h in header]
    for r in rows:
        for i, c in enumerate(r):
            widths[i] = max(widths[i], len(str(c)))
    out = ["| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(header)) + " |",
           "|" + "|".join("-" * (w + 2) for w in widths) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c).ljust(widths[i]) for i, c in enumerate(r)) + " |")
    return "\n".join(out)


# --------------------------------------------------------------------------- C9

def _declared_field_names(text: str) -> list[str]:
    """从「可自证字段」行里抽取反引号内标识符（原 `check_c9` 内联逻辑，抽成函数）。"""
    out: list[str] = []
    stop = {"result", "switches", "null", "true", "false", "none", "n/a", "csv", "json"}
    for line in text.splitlines():
        if "可自证字段" not in line:
            continue
        for span in re.findall(r"`([^`]+)`", line):
            for tok in re.split(r"[、,，/ ]+", span):
                tok = tok.strip()
                if (re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{2,}", tok)
                        and tok.lower() not in stop and tok not in out):
                    out.append(tok)
    return out


def check_c9() -> int:
    """C9：`AGENT.md §十三` 声明的 `switches`/`result` 字段，代码里真的存在吗？

    典型事故：文档写了一个字段名，代码里根本没有（或拼错）⇒ 外复核按文档去找读数 ⇒ 找不到。
    """
    print("\n=== C9：文档声明 vs 代码实际（AGENT.md §十三 可自证字段） ===")
    agent = ROOT / "AGENT.md"
    if not agent.is_file():
        print("⚠️ 找不到 AGENT.md —— 跳过")
        return 0
    txt = agent.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"^##\s*十三、.*?$(.*?)^##\s*十四、", txt, re.S | re.M)
    if not m:
        print("⚠️ AGENT.md 里找不到 §十三（纪元登记）—— 跳过")
        return 0
    sect = m.group(1)

    # 🔴 「已声明但尚未实现」的纪元：设计稿/派工单已发、代码还没写。这类纪元的字段
    #    **必然找不到**，若照常报 🔴 ⇒ 每立一个新纪元都撞一次假报警（假报警会淹没真
    #    报警，同 `git` 检查那次）。处理规则（三条缺一不可）：
    #      ① **必须**在该小节标题里显式写「待实现」才跳过（不许靠字段名猜）；
    #      ② 跳过项**显著报出**（打印小节名 + 全部字段），**禁静默**；
    #      ③ 实现完成后**必须同 commit 删掉标题里的标记**，否则字段永远不被校验。
    pending: list[tuple[str, list[str]]] = []
    active_blobs: list[str] = []
    for blk in re.split(r"^### ", sect, flags=re.M):
        head = blk.splitlines()[0].strip() if blk.strip() else ""
        if head and re.search(r"待实现|尚未实现", head):
            names = _declared_field_names(blk)
            if names:
                pending.append((head[:46], names))
            continue
        active_blobs.append(blk)
    declared = _declared_field_names("\n".join(active_blobs))

    if not declared and not pending:
        print("⚠️ §十三 未解析出任何字段名（格式变了？）—— 跳过")
        return 0

    missing: list[str] = []
    rows: list[list[str]] = []
    for name in declared:
        hits = find_identifier(name)
        if hits:
            rows.append([f"`{name}`", "✅", str(len(hits)), hits[0]])
        else:
            missing.append(name)
            rows.append([f"`{name}`", "🔴 代码中未找到", "0", "—"])

    print(_table(rows, ["字段名", "结论", "命中文件数", "首个命中"]))
    print(f"\n声明字段 {len(declared)} 个；代码中未找到 **{len(missing)}** 个。")
    if missing:
        print("🔴 未找到：" + "、".join(f"`{x}`" for x in missing))
        print("⇒ 逐项确认：(a) 拼写/命名不符（**C9 缺陷，必修**）；"
              "(b) 读数块名/CSV 列名（可接受，但须在文档里注明其归属）。"
              "**不许用「它是结果块名」一句话了事——必须给出它出现在哪一行**。")
        return 1
    print("✅ 全部名字都能在代码里按词边界找到"
          "（注意：这只证明「名字在」，不证明「接线对」）。")
    if pending:
        total = sum(len(n) for _, n in pending)
        print(f"\n⚠️ **跳过 {len(pending)} 个「待实现」纪元小节**（共 {total} 个字段**未校验**）：")
        for head, names in pending:
            print(f"  · {head} ⇒ " + "、".join(f"`{x}`" for x in names))
            print("    🔴 归属：设计稿/派工单已发，代码尚未实现（字段不存在是**预期**）；"
                  "**实现完成后必须同 commit 删掉小节标题里的「待实现」标记**，"
                  "否则这些字段永远不被 C9 校验（= 用标记开后门）。")
    return 0


# --------------------------------------------------------------------------- CLI

def _load_module(path: Path, name: str):
    """以独立模块加载仓库内脚本。

    🔴 **Py3.14 陷阱（2026-09-24 实测，本工具开发时当场踩到）**：`dataclasses` 在处理类时
    会做 `sys.modules.get(cls.__module__).__dict__` ⇒ 若模块**未注册进 `sys.modules`**，
    就抛 `AttributeError: 'NoneType' object has no attribute '__dict__'`。
    这与 13.4 那次「a4 在 Python 3.14 崩」的 **P0 同族**（本地 .venv 即 3.14.7）。
    ⇒ 必须在 `exec_module` **之前**注册；并保证仓库根在 `sys.path`（脚本自己的 import 才能解析）。
    """
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)      # type: ignore[union-attr]
    except Exception:
        sys.modules.pop(name, None)
        raise
    return mod


def _argparse_flags(src: str) -> set[str]:
    flags: set[str] = set()
    for call in re.findall(r"add_argument\((.*?)\)", src, re.S):
        flags.update(re.findall(r'"(?:-{1,2})([A-Za-z0-9_-]+)"', call))
    return flags


def check_cli() -> int:
    """preset 展开出的 `--flag`，目标脚本的 argparse 真的注册了吗？

    典型事故（F-R21 家族）：preset 写了个脚本不认识的开关 ⇒
    脚本若用 `parse_known_args` 就**静默忽略** ⇒ 跑出来的批次与预检广告的不是同一个配置。
    """
    print("\n=== CLI：preset 参数 vs 目标脚本 argparse ===")
    br = ROOT / "experiments" / "batch_runner.py"
    if not br.is_file():
        print("⚠️ 找不到 experiments/batch_runner.py —— 跳过")
        return 0
    try:
        mod = _load_module(br, "_sa_batch_runner")
    except Exception as exc:                                   # noqa: BLE001
        print(f"⚠️ batch_runner 导入失败（{type(exc).__name__}: {exc}）—— 跳过")
        return 1

    presets = getattr(mod, "PRESETS", {})
    bad: list[list[str]] = []
    known_flags: dict[str, set[str]] = {}
    known_relaxed: set[str] = set()
    n_run = 0

    for pname in sorted(presets):
        try:
            runs = mod.preset_runs(pname, ROOT)
        except Exception as exc:                               # noqa: BLE001
            bad.append([pname, "<展开失败>", f"{type(exc).__name__}: {exc}"])
            continue
        n_run += len(runs)
        for run in runs:
            cmd = list(run.cmd)
            script = rel((ROOT / cmd[0]) if not Path(cmd[0]).is_absolute() else Path(cmd[0]))
            sp = ROOT / cmd[0] if not Path(cmd[0]).is_absolute() else Path(cmd[0])
            if script not in known_flags:
                if sp.is_file():
                    src = sp.read_text(encoding="utf-8", errors="ignore")
                    known_flags[script] = _argparse_flags(src)
                    if "parse_known_args" in src:
                        known_relaxed.add(script)
                else:
                    known_flags[script] = set()
            for tok in cmd[1:]:
                if not tok.startswith("--"):
                    continue
                if tok[2:] not in known_flags[script]:
                    row = [pname, script, tok]
                    if row not in bad:
                        bad.append(row)

    print(f"扫了 **{len(presets)}** 个 preset / **{n_run}** 个 run。")
    if known_relaxed:
        print("🔴 以下脚本使用 `parse_known_args` ⇒ **未知开关会被静默忽略**，本条检查对它们尤其重要：")
        for s in sorted(known_relaxed):
            print(f"   - {s}")
    if bad:
        print("\n🔴 未注册的开关（**逐个确认：是笔误，还是脚本漏注册**）：")
        print(_table(bad, ["preset", "目标脚本", "未注册开关"]))
        print("\n⚠️ 注意：`--flag value` 形式下，本工具把所有 `--xxx` 词元当开关；"
              "若某个**取值**本身以 `--` 开头，会出现误报，人工确认即可。")
        return 1
    print("✅ 所有预设开关都能在目标脚本的 argparse 里找到。")
    return 0


# --------------------------------------------------------------------------- git

def _git(*args: str) -> tuple[int, str]:
    try:
        p = subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                           text=True, encoding="utf-8", errors="ignore")
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except OSError as exc:
        return 1, f"git 调用失败：{exc}"


def check_git() -> int:
    """交付卫生：有没有没提交的、有没有没推送的、有没有在 main 上乱推。

    🔴 零号规则：**未 push 不算发言**。
    """
    print("\n=== git：交付卫生 ===")
    rc, out = _git("status", "--porcelain")
    if rc != 0:
        print(f"⚠️ git status 失败：{out.strip()}")
        return 1
    dirty = [l for l in out.splitlines() if l.strip()]
    tracked_dirty = [l for l in dirty if not l.startswith("??")]
    untracked = [l for l in dirty if l.startswith("??")]

    _, br = _git("rev-parse", "--abbrev-ref", "HEAD")
    branch = br.strip() or "<未知>"
    _, head = _git("log", "-1", "--format=%h %s")
    _, nstat = _git("show", "--stat", "--oneline", "-1", "--format=")
    head_files = len([l for l in nstat.splitlines() if "|" in l])

    print(f"分支：`{branch}`｜HEAD：`{head.strip()}`｜HEAD 提交文件数：**{head_files}**")

    problems = 0
    if branch == "main":
        print("🔴 当前在 `main` 分支上 ⇒ 对 **`[云端·开发]`/`[本地开发]`** 而言违反分支纪律"
              "（新功能必须独立分支）；**`[所有者]` 在主工作树操作 `main` 属正常** —— 自行对号入座。")
        problems += 1
    if tracked_dirty:
        print(f"\n🔴 有 **{len(tracked_dirty)}** 个已跟踪文件未提交：")
        for l in tracked_dirty[:20]:
            print(f"   {l}")
        problems += 1
    if untracked:
        print(f"\n⚠️ 未跟踪文件 {len(untracked)} 个（多数是日志/产物，确认没有该入库的）：")
        for l in untracked[:15]:
            print(f"   {l}")

    # 未推送？—— 🔴 必须**按本分支**的远端比，不能硬编码 `gitee/main`
    # （2026-09-24 `[云端·开发]` 实测报告：在 `dev/terrain-s2` 上跑本工具恒报"未推送"，
    #   因为特性分支的提交本来就不在 `gitee/main` 里 ⇒ **假报警**会让真报警被忽略）。
    ref = None
    rc2, up = _git("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}")
    if rc2 == 0 and up.strip() and "@{u}" not in up:
        ref = up.strip()
    if ref is None and branch not in ("<未知>", "HEAD"):
        for cand in (f"gitee/{branch}", f"origin/{branch}"):
            rc3, _ = _git("rev-parse", "--verify", cand)
            if rc3 == 0:
                ref = cand
                break
    if ref is None:
        ref = "gitee/main"
    rc4, _ = _git("rev-parse", "--verify", ref)
    if rc4 == 0:
        _, cnt = _git("rev-list", "--count", f"{ref}..HEAD")
        ahead = cnt.strip()
        tag = "" if ref == "gitee/main" else f"（本分支 `{branch}` 的远端）"
        if ahead.isdigit() and int(ahead) > 0:
            print(f"\n🔴 有 **{ahead}** 个提交**未推送**到 `{ref}`{tag} ⇒ 按零号规则**不算交付**。")
            print(f"   ⇒ 对账命令：`git ls-remote gitee {branch}`（勿只看 refs/remotes，F-R24）")
            problems += 1
        else:
            print(f"\n✅ 与 `{ref}` 同步（无未推送提交）{tag}。")
    else:
        print(f"\n⚠️ 读不到远端跟踪引用 `{ref}` ⇒ 请先 `git fetch gitee`（本条无法判）。")
        print(f"   ⇒ 特性分支请手动对账：`git ls-remote gitee {branch}`")

    # 禁推物
    _, changed = _git("show", "--name-only", "--format=", "HEAD")
    forbidden = [f for f in changed.splitlines()
                 if f.endswith(("sim_core.so", ".npz", ".pkl")) or f.startswith("_rerun_logs/")]
    if forbidden:
        print(f"\n🔴 HEAD 提交里含**入禁文件**：{forbidden}")
        problems += 1
    return 1 if problems else 0


# --------------------------------------------------------------------------- digest

def check_digest(python: str | None = None) -> int:
    """C7：跑**所有含基线 digest 断言的测试文件** ⇒ 关档是否仍逐位等价。"""
    print("\n=== C7：关档逐位等价（跑含 digest 基线的测试文件） ===")
    tests = ROOT / "tests"
    files: list[str] = []
    if tests.is_dir():
        for p in sorted(tests.glob("*.py")):
            t = p.read_text(encoding="utf-8", errors="ignore")
            if re.search(r"\(\s*57\d{4},\s*\d+\.\d+\s*\)", t):
                files.append(rel(p))
    if not files:
        print("⚠️ tests/ 下没找到含 digest 基线的文件 ⇒ 无法机械化判 C7（**这本身是个问题**）。")
        return 1
    print(f"命中 {len(files)} 个文件：" + "、".join(f"`{f}`" for f in files))

    # 🔴 R199（云归 R198 ⑥-A 报的**工具缺陷**，非他那轮的引入）：
    #   原实现 `exe = python or .venv/Scripts/python.exe`，再 `if not exists: exe =
    #   sys.executable` ⇒ **显式给了 `--python` 也会被静默覆盖**（Linux 上
    #   `.venv/Scripts/python.exe` 不存在 ⇒ 恒定回落 `sys.executable`）。
    #   后果实例：`sim_core.so` 为 **3.12** 编译、驱动解释器是 **3.11** ⇒
    #   `import sim_core` **段错误（exit 139 / SIGSEGV）** ⇒ digest 门报"未全绿"
    #   却**看不到任何 pytest 失败** ⇒ 假报警且无从诊断（假报警会淹没真报警）。
    #   修法：**显式指定就必须生效**（不存在 ⇒ fail-loud）；未指定才按 .venv → 当前解释器。
    if python:
        exe = python
        if not Path(exe).exists():
            print(f"🔴 `--python` 指定的解释器不存在：{exe}（未回落，fail-loud）")
            return 1
    else:
        exe = str(ROOT / ".venv" / "Scripts" / "python.exe")
        if not Path(exe).exists():
            exe = sys.executable
    base = tempfile.mkdtemp(prefix="sa_basetemp_")       # 🔴 仓库外 + 每次全新
    cmd = [exe, "-m", "pytest", *files, "-q", "--basetemp", base]
    print("运行：" + " ".join(cmd))
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                       encoding="utf-8", errors="ignore")
    tail = [l for l in (p.stdout or "").splitlines() if l.strip()][-6:]
    for l in tail:
        print("   " + l)
    if p.returncode < 0:
        # 负返回码 = 被信号杀死（Linux 常见 −11 SIGSEGV）；段错误**不是** pytest 失败，
        #   pytest 报告里不会有失败项 ⇒ 必须显式提示，否则只能看到"未全绿"四个字。
        import signal as _sig
        try:
            name = _sig.Signals(-p.returncode).name
        except Exception:
            name = f"signal {-p.returncode}"
        print(f"\n🔴 pytest 进程被信号杀死（{name}，returncode={p.returncode}）"
              " ⇒ **不是测试失败**，而是解释器进程崩溃。"
              "\n   最常见原因：`sim_core` 扩展与驱动解释器的 Python 次版本 ABI 不匹配"
              "（如 `.so`/`.pyd` 为 3.12 编译、却用 3.11 跑 ⇒ `import sim_core` 段错误）。"
              "\n   ⇒ 用 `--python <与 sim_core 同版本的解释器>` 重跑；"
              "或在本机重建 sim_core 后再跑。")
        return 1
    if p.returncode != 0:
        print("\n🔴 digest 测试未全绿 ⇒ 关档**不再逐位等价**。"
              "要么回退，要么按 B1 登记纪元并说明（**不许悄悄改基线**）。")
        return 1
    print("\n✅ 关档逐位等价（基线未漂移）。")
    return 0


# --------------------------------------------------------------------------- pack

def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _latest_era() -> str:
    agent = ROOT / "AGENT.md"
    if not agent.is_file():
        return "<未知>"
    txt = agent.read_text(encoding="utf-8", errors="ignore")
    eras = re.findall(r"^###\s*(1[0-9]\.[0-9]+[^\n|]*)$", txt, re.M)
    return eras[-1].strip() if eras else "<未知>"


def cmd_pack(args) -> int:
    """打包**证据清单**（只写哈希与元数据，不复制文件 ⇒ 不撑仓库）。"""
    print("\n=== 证据包（evidence pack） ===")
    phase = args.phase
    outdir = ROOT / "_audit" / phase
    outdir.mkdir(parents=True, exist_ok=True)

    files: list[Path] = []
    for pat in args.include or []:
        hits = [Path(h) for h in sorted(_glob.glob(str(ROOT / pat), recursive=True))]
        if not hits:
            hits = [Path(h) for h in sorted(_glob.glob(pat, recursive=True))]
        if not hits:
            print(f"⚠️ `--include {pat}` 没匹配到任何文件")
        files.extend(h for h in hits if h.is_file())
    if not files:
        print("🔴 没收集到任何证据文件。证据包 = 回板的**必要条件**（协议 §三 B7）。")
        return 1

    _, head = _git("log", "-1", "--format=%h")
    _, br = _git("rev-parse", "--abbrev-ref", "HEAD")
    _, dirty = _git("status", "--porcelain")
    _, remote = _git("rev-parse", "--short", "gitee/main")

    rows, lines = [], []
    for f in sorted(set(files)):
        h = _sha256(f)
        st = f.stat()
        rows.append([f"`{rel(f)}`", f"{st.st_size:,}", time.strftime('%m-%d %H:%M', time.localtime(st.st_mtime)), h[:16]])
        lines.append(f"{h}  {st.st_size:>12,}  {rel(f)}")

    (outdir / "SHA256.txt").write_text(
        f"# {phase} 证据清单 —— sha256sum 格式（生成于 {time.strftime('%Y-%m-%d %H:%M:%S')}）\n"
        + "\n".join(lines) + "\n", encoding="utf-8")

    man = [
        f"# 证据包 · {phase}",
        "",
        f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 分支 / HEAD：`{br.strip()}` / `{head.strip()}`",
        f"- `gitee/main`：`{remote.strip() or '<未 fetch>'}`",
        f"- 未提交文件数：**{len([l for l in dirty.splitlines() if l.strip()])}**",
        f"- 当前纪元（AGENT.md §十三 末条）：**{_latest_era()}**",
        f"- 文件数：**{len(rows)}**",
        "",
    ]
    if getattr(args, "note", None):
        man += ["## 说明", "", args.note, ""]
    man += ["## 文件清单（完整 sha256 见同目录 `SHA256.txt`）", "", _table(rows, ["文件", "字节", "修改时间", "sha256(前16)"]), ""]
    manifest = outdir / "MANIFEST.md"
    manifest.write_text("\n".join(man), encoding="utf-8")

    print(_table(rows, ["文件", "字节", "修改时间", "sha256(前16)"]))
    print(f"\n✅ 已写 `{rel(manifest)}` + `{rel(outdir / 'SHA256.txt')}`")
    print("⇒ 回板时把这两个路径贴上去（**没有它，回板视为未完成**）。")
    return 0


# --------------------------------------------------------------------------- main

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="云端自审工具（协议见 docs/tasks/协议-云端自审与自优化-20260924.md）")
    ap.add_argument("check", choices=("all", "c9", "cli", "git", "digest", "pack"))
    ap.add_argument("--phase", default="misc", help="pack：阶段名（如 S1）")
    ap.add_argument("--include", action="append", default=[],
                    help="pack：证据路径/通配（可重复）")
    ap.add_argument("--note", default=None, help="pack：写进 MANIFEST 的说明")
    ap.add_argument("--python", default=None, help="digest：跑 pytest 用的解释器")
    args = ap.parse_args(argv)

    if args.check == "pack":
        return cmd_pack(args)

    t0 = time.time()
    rc = 0
    if args.check in ("all", "c9"):
        rc |= check_c9()
    if args.check in ("all", "cli"):
        rc |= check_cli()
    if args.check in ("all", "git"):
        rc |= check_git()
    if args.check == "digest":
        rc = check_digest(args.python)

    print(f"\n---- 自审结束：{'🔴 有命中项（见上）' if rc else '🟢 全部通过'} "
          f"｜耗时 {time.time() - t0:.1f}s ----")
    print("判定与处置见协议 §四/§五；**命中 Blocker 必须停下回板，不许自决**。")
    return 1 if rc else 0


if __name__ == "__main__":
    sys.exit(main())
