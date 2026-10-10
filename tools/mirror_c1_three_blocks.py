#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""mirror_c1_three_blocks.py —— 镜·C1 三阻塞「由红转守」复核件（R394④：审核人自写）

审头 `_archive/2026-10-10-退役团队-归档/docs-旧档/审核-C1-IMPLEMENT-钩子侧车与装置面-20261005.md` 留三条 🔴（当时实测 0 红）：
  C-A `write_run_summary` 唯一 scipy 入口在收尾 ⇒ 正式档跑完之后才崩（侧车二/manifest 全丢）
  C-B 第二处（致伤致死 site-2）钩子删除 ⇒ 33 例全绿
  C-C `test_t13_no_hardcoded_120` 同式自证 ⇒ `% 120` 硬编码 0 红

轻舟 @582c1f0 声称三处已修。本件不复用她的取证件，只做两件事：

  `mut` —— 把**我原来判 0 红的那三条变异**原样重放，外加两条我更严的变体：
      m-site1  site-1（`sphere_engine.py:4675`）append guard ⇒ `if False:`  ⇒ 期望 **红**
      m-site2  site-2（`:4708`，C-B 本体）摘除                ⇒ 期望 **红**
      m-cc120  列带映射 `col = flat_idx % self._cols` ⇒ `% 120`（C-C 本体）  ⇒ 期望 **红**
      m-ccrow  行带分母 `// self._rows` ⇒ `// 120`（我加严，非她列出的档）    ⇒ 期望 **红**
      m-ca     C-A fail-fast 条件改恒假（=修复前行为）        ⇒ 期望 **红**
  `ca` —— C-A 的 A/B（同一条 300t / 50 窗步长命令，缩微替身，非正式档）：
      A：带 fail-fast（交付原样）   ⇒ 期望 rc=2 且 主表/侧车一/侧车二/summary **一个都不落盘**
      B：fail-fast 恒假（=修复前）  ⇒ 期望 rc=1、主表+侧车一已落盘、侧车二+summary 全丢
      ⇒ A≠B 即这条守的**检出面**：证明它是"起跑前拦"，不是"把崩溃换个报错文案"。

退出码：0 = 全部符合期望 ｜ 1 = 有不符 ｜ 2 = 锚点未命中/基线不绿/环境缺件（绝不"没查也绿"）
"""
from __future__ import annotations

import argparse
import io
import os
import subprocess
import sys

ENGINE = os.path.join("simulation", "sphere_engine.py")
A4 = os.path.join("experiments", "a4_verify_capacity.py")
TEST = os.path.join("tests", "test_c1_rd_instruments.py")

G1_OLD = "                        if self._rd_pred_kill_log is not None:"
G2_OLD = "                            if self._rd_pred_kill_log is not None:"
FF_OLD = "        if _plan_windows >= 3:"
CC_OLD = "        col = flat_idx % self._cols"
CC_NEW = "        col = flat_idx % 120"
CR_OLD = "        lat_band = (row * self.block_rows) // self._rows"
CR_NEW = "        lat_band = (row * self.block_rows) // 120"

# (名称, 文件, 锚点, 替身, 期望)  —— 期望 "red"=必红（守住了）/ "green"=仍绿（守力盲区探针）
MUTS = [
    ("m-site1 site-1 append guard 摘除", ENGINE, G1_OLD, G1_OLD.split("if ")[0] + "if False:", "red"),
    ("m-site2 site-2 append guard 摘除（C-B 本体）", ENGINE, G2_OLD, G2_OLD.split("if ")[0] + "if False:", "red"),
    ("m-cc120 列带映射写死 %120（C-C 本体）", A4, CC_OLD, CC_NEW, "red"),
    ("m-ccrow 行带分母写死 //120（我加严）", A4, CR_OLD, CR_NEW, "red"),
    ("m-ca C-A fail-fast 条件恒假（缺 scipy 机上应红）", A4, FF_OLD, "        if _plan_windows >= 999999:", "red"),
]

# 300 tick / 50 窗步长 ⇒ 计划窗数 6 ≥3：正式档口径的缩微替身（R117：不跑正式档）
CA_ARGS = ["--mode", "off", "--seed", "7", "--ticks", "300", "--log-interval", "50",
           "--smell-channels", "risk", "--rd-instruments", "--rd-sample-every", "50"]


def _sub(tree: str, *args: str, env_extra=None) -> subprocess.CompletedProcess:
    env = dict(os.environ, PYTHONPATH=tree, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    env.update(env_extra or {})
    return subprocess.run([sys.executable, *args], cwd=tree, capture_output=True,
                          text=True, encoding="utf-8", errors="replace", env=env)


def _tail(out: str) -> str:
    lines = [l for l in out.splitlines()
             if any(k in l for k in ("passed", "failed", "error", "Error"))]
    return lines[-1].strip() if lines else out.strip()[-160:].replace("\n", " / ")


def _restore(tree: str, *files: str) -> None:
    subprocess.run(["git", "checkout", "--", *files], cwd=tree)


def _edit(tree: str, rel: str, old: str, new: str) -> bool:
    p = os.path.join(tree, rel)
    src = io.open(p, encoding="utf-8").read()
    if old not in src:
        return False
    io.open(p, "w", encoding="utf-8", newline="").write(src.replace(old, new, 1))
    return True


def _orig(tree: str, rel: str) -> str:
    return io.open(os.path.join(tree, rel), encoding="utf-8").read()


def cmd_mut(tree: str, sel: str = "") -> int:
    if not all(os.path.isfile(os.path.join(tree, f)) for f in (ENGINE, A4, TEST)):
        print("🔴 缺件（ENGINE/A4/TEST）", file=sys.stderr)
        return 2
    bad, e_orig, a_orig = 0, _orig(tree, ENGINE), _orig(tree, A4)
    cmd = ["-m", "pytest", TEST, "-q", "--tb=line"] + (["-k", sel] if sel else [])
    p = _sub(tree, *cmd)
    tail = _tail((p.stdout or "") + (p.stderr or ""))
    print(f"【基线 未变异】 {tail}")
    if "failed" in tail or "error" in tail.lower():
        print("   ⇒ 🔴 基线不绿 ⇒ 后续变异判读无效")
        return 2
    try:
        for name, rel, old, new, expect in MUTS:
            io.open(os.path.join(tree, ENGINE), "w", encoding="utf-8", newline="").write(e_orig)
            io.open(os.path.join(tree, A4), "w", encoding="utf-8", newline="").write(a_orig)
            if not _edit(tree, rel, old, new):
                print(f"【{name}】 ⇒ ⚠️ 锚点未命中 ⇒ 不判绿")
                bad += 1
                continue
            q = _sub(tree, *cmd)
            t = _tail((q.stdout or "") + (q.stderr or ""))
            got = "red" if ("failed" in t or "error" in t.lower()) else "green"
            hit = got == expect
            print(f"【{name}】 ⇒ {t}   期望 {expect} ⇒ {'✅' if hit else '❌ 与期望不符'}")
            if not hit:
                bad += 1
        io.open(os.path.join(tree, ENGINE), "w", encoding="utf-8", newline="").write(e_orig)
        io.open(os.path.join(tree, A4), "w", encoding="utf-8", newline="").write(a_orig)
    finally:
        _restore(tree, ENGINE, A4)
    r = _sub(tree, *cmd)
    t2 = _tail((r.stdout or "") + (r.stderr or ""))
    print(f"【还原复跑】 {t2}   {'✅ 已复原' if 'failed' not in t2 else '❌ 还原后不绿'}")
    if "failed" in t2:
        bad += 1
    print(f"\n# mut 汇总：{'全部符合期望（三阻塞已由红转守）' if bad == 0 else f'{bad} 项与期望不符'}")
    return 0 if bad == 0 else 1


def _ca_run(tree: str, out_csv: str) -> dict:
    o = os.path.join(out_csv, "ca.csv")
    p = _sub(tree, A4, *(CA_ARGS + ["--out", o]))
    side = os.path.join(out_csv, "ca_rd_windows.csv")
    return {"rc": p.returncode, "out": p.stdout + p.stderr,
            "main": os.path.isfile(o), "side1": os.path.isfile(side),
            "side2": os.path.isfile(os.path.join(out_csv, "ca_rd_run.csv")),
            "summary": os.path.isfile(os.path.join(out_csv, "ca.summary.json"))}


def cmd_ca(tree: str, workdir: str) -> int:
    os.makedirs(workdir, exist_ok=True)
    a4p = os.path.join(tree, A4)
    orig = _orig(tree, A4)
    try:
        if not _edit(tree, A4, FF_OLD, "        if _plan_windows >= 999999:"):
            print("🔴 C-A fail-fast 锚点未命中 ⇒ 无法判定", file=sys.stderr)
            return 2
        b = _ca_run(tree, os.path.join(workdir, "B_noguard"))
    finally:
        io.open(a4p, "w", encoding="utf-8", newline="").write(orig)
        _restore(tree, A4)
    a = _ca_run(tree, os.path.join(workdir, "A_guard"))

    print("A｜带 fail-fast（交付原样）："
          f"rc={a['rc']} 主表={a['main']} 侧车一={a['side1']} 侧车二={a['side2']} summary={a['summary']}")
    print("B｜fail-fast 恒假（=修复前行为）："
          f"rc={b['rc']} 主表={b['main']} 侧车一={b['side1']} 侧车二={b['side2']} summary={b['summary']}")
    sci = "No module named 'scipy'" in b["out"]
    print(f"   B 的失败原因含 \"No module named 'scipy'\" ⇒ {sci}")
    bad = 0
    for name, ok in [
        ("A rc=2（argparse 期拒跑）", a["rc"] == 2),
        ("A 什么都没落盘（启动期拦，不是跑完才崩）", not (a["main"] or a["side1"] or a["side2"] or a["summary"])),
        ("A stderr 点名 scipy", "scipy" in a["out"]),
        ("B 确实起跑（主表已写）", b["main"]),
        ("B 侧车一已 flush 而侧车二/summary 全丢（=我审头描述的最坏位置）", b["side1"] and not b["side2"] and not b["summary"]),
        ("B 崩因是缺 scipy（非别的错）", sci),
        ("A≠B（守有检出面）", a["rc"] != b["rc"]),
    ]:
        print(f"{'✅' if ok else '❌'} {name}")
        bad += 0 if ok else 1
    print(f"\n# ca 汇总：{'C-A 销项成立（启动期 fail-fast 有真实检出面）' if bad == 0 else f'{bad} 项不符'}")
    return 0 if bad == 0 else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="C1 三阻塞由红转守复核（镜·独立件）")
    ap.add_argument("cmd", choices=["mut", "ca"])
    ap.add_argument("--tree", required=True, help="被审树根（@582c1f0）")
    ap.add_argument("--sel", default="", help='pytest -k 选择器（默认跑整个靶向文件）')
    ap.add_argument("--workdir", default="", help="ca 子命令的产物目录（默认 TEMP/jing_c1_ca）")
    a = ap.parse_args(argv)
    tree = os.path.abspath(a.tree)
    if a.cmd == "mut":
        return cmd_mut(tree, a.sel)
    wd = a.workdir or os.path.join(os.environ.get("TEMP", "/tmp"), "jing_c1_ca")
    return cmd_ca(tree, wd)


if __name__ == "__main__":
    sys.exit(main())
