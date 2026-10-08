#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""mirror_s1_readback.py —— 镜·S1 读回守销项独立复核件（R394④：审核人自写，不复用被审链脚本）

背景：镜队列① 审头 S1 指出「CLI→`run_one` 最后一公里无读回 ⇒ 删接线仍全绿（变异 M4）」；
轻舟 @9bac9a8 以「生效值读回守 + T7 真跑断言」修复，PI @75689ce 合 main。本件复跑并**加严**：

  subcommand `mut`  —— 变异必红复跑（四条，含两条轻舟/PI 未测的新链段）
      r1 删 CLI→run_one 接线（= 轻舟 M4 原案）        ⇒ 期望 T7 红
      r2 连 eff_out 一起删（_eff 空 ⇒ 键为 None）      ⇒ 期望 T7 红
      r3 **run_one→引擎断链**（构造时强摁默认档）        ⇒ 期望 T7 红  ← 新链段
      r4 读回改写入参回声（单端）                      ⇒ 期望 T7 绿（守力来源探针，见 c3）
      c3 = r4 双端 + r3 ⇒ 期望 **T7 与 T3 全绿 = 假绿**  ← 据此提加固建议

  subcommand `q1`   —— Q1 默认档逐字节等价（base 树 vs head 树）
      a 表头相同 / b 数据行逐位相同（剔计时列）/ c meta 相同（剔 out）
      d-f runs：条数同、新增键集 == {rd_enabled_effective, signal_alphabet_effective}、
         无键丢失、共有键逐值相同（剔计时键）

用法：
    py tools/mirror_s1_readback.py mut --tree <含 tests/test_s3_rdswitch.py 的树>
    py tools/mirror_s1_readback.py q1 --base <pre-S1 树> --head <post-S1 树>

退出码：0 = 全部符合期望 ｜ 1 = 有不符 ｜ 2 = 锚点未命中/环境缺件（不静默判绿）
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import subprocess
import sys

PROBE = os.path.join("experiments", "s3_memory_probe.py")
TESTS = os.path.join("tests", "test_s3_rdswitch.py")
TIME_COLS = {"ms_per_tick", "wall_s"}
TIME_KEYS = {"wall_s"}
NEW_KEYS = {"rd_enabled_effective", "signal_alphabet_effective"}

# 变异锚点（被审件行宽变动即 fail-loud ⇒ 绝不"锚点没命中"却报绿）
WIRE_OLD = ('                    rd_on=(a.rd_mode == "on"), alphabet=a.alphabet,\n'
            '                    eff_out=_eff)')
WIRE_R1 = '                    eff_out=_eff)'
WIRE_R2 = '                    rd_on=(a.rd_mode == "on"), alphabet=a.alphabet)'
ENG_OLD = '            rd_on=rd_on, alphabet=alphabet)\n    # 🔴 S1 读回守'
ENG_R3 = '            rd_on=True, alphabet="16")\n    # 🔴 S1 读回守'
EFF_OLD_RD = '        eff_out["rd_enabled"] = bool(eng.config.resource_dynamics.enabled)'
EFF_NEW_RD = '        eff_out["rd_enabled"] = bool(rd_on)'
EFF_OLD_AB = '        eff_out["signal_alphabet"] = str(eng.config.signal_alphabet)'
EFF_NEW_AB = '        eff_out["signal_alphabet"] = str(alphabet)'

MUTS = [
    ("r1 删 CLI→run_one 接线（轻舟 M4 原案）", [(WIRE_OLD, WIRE_R1)], "red"),
    ("r2 连 eff_out 一起删", [(WIRE_OLD, WIRE_R2)], "red"),
    ("r3 run_one→引擎断链（强摁默认档）", [(ENG_OLD, ENG_R3)], "red"),
    ("r4 读回降级为入参回声（rd 端）", [(EFF_OLD_RD, EFF_NEW_RD)], "green"),
    ("c3 双端回声 + 引擎断链（组合 ⇒ 期望假绿）",
     [(EFF_OLD_RD, EFF_NEW_RD), (EFF_OLD_AB, EFF_NEW_AB), (ENG_OLD, ENG_R3)], "green"),
]


def _py(tree: str, *args: str) -> subprocess.CompletedProcess:
    env = dict(os.environ, PYTHONPATH=tree, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    return subprocess.run([sys.executable, *args], cwd=tree, capture_output=True,
                          text=True, encoding="utf-8", errors="replace", env=env)


def _pytest_tail(out: str) -> str:
    lines = [l for l in out.splitlines()
             if any(k in l for k in ("passed", "failed", "error"))]
    return lines[-1].strip() if lines else out.strip()[-160:].replace("\n", " / ")


def _run_sel(tree: str, sel: str):
    """返回 (是否红, 汇总行)；None = 采集/collect 失败 ⇒ 一律不按绿处理。"""
    p = _py(tree, "-m", "pytest", TESTS, "-q", "--tb=line", "-k", sel)
    tail = _pytest_tail((p.stdout or "") + (p.stderr or ""))
    return (("failed" in tail) if "error" not in tail.lower() else None), tail


def cmd_mut(tree: str, sel: str = "T7") -> int:
    f = os.path.join(tree, PROBE)
    if not (os.path.isfile(f) and os.path.isfile(os.path.join(tree, TESTS))):
        print(f"🔴 缺件：{f} 或 {TESTS}", file=sys.stderr)
        return 2
    orig = io.open(f, encoding="utf-8").read()
    bad = 0
    ok, tail = _run_sel(tree, sel)
    print(f"【基线 -k {sel} 未变异】 {tail}")
    if ok is not False:
        print("   ⇒ 🔴 基线不绿，后续变异判读无效（先修环境/被审件）")
        return 2
    try:
        for name, edits, expect in MUTS:
            src = orig
            for old, new in edits:
                if old not in src:
                    print(f"【{name}】 ⇒ ⚠️ 锚点未命中 ⇒ 无法判定（不计绿）")
                    bad += 1
                    src = None
                    break
                src = src.replace(old, new, 1)
            if src is None:
                continue
            io.open(f, "w", encoding="utf-8", newline="").write(src)
            try:
                red, tail = _run_sel(tree, sel)
                got = "red" if red else ("green" if red is False else "error")
                hit = (got == expect)
                print(f"【{name}】 ⇒ {tail}   期望 {expect} ⇒ {'✅' if hit else '❌ 与期望不符'}")
                if not hit:
                    bad += 1
            finally:
                subprocess.run(["git", "checkout", "--", PROBE], cwd=tree)
    finally:
        subprocess.run(["git", "checkout", "--", PROBE], cwd=tree)
    ok2, tail2 = _run_sel(tree, sel)
    print(f"【还原复跑】 {tail2}   {'✅ 已复原' if ok2 is False else '❌ 还原后仍不绿'}")
    if ok2 is not False:
        bad += 1
    print(f"\n# mut 汇总：{'全部符合期望' if bad == 0 else f'{bad} 项与期望不符'}")
    return 0 if bad == 0 else 1


def _strip_csv(path: str):
    rows = list(csv.reader(io.open(path, encoding="utf-8", errors="replace")))
    if not rows:
        raise ValueError(f"空 CSV：{path}")
    hdr, keep = rows[0], [i for i, c in enumerate(rows[0]) if c not in TIME_COLS]
    return hdr, [tuple(r[i] for i in keep) for r in rows[1:] if r]


def cmd_q1(base: str, head: str) -> int:
    d = os.path.join(os.environ.get("TEMP", "/tmp"), "jing_s1_q1")
    os.makedirs(d, exist_ok=True)
    outs, rc = {}, 0
    for tag, tree in (("base", base), ("head", head)):
        o = os.path.join(d, f"{tag}.csv")
        p = _py(tree, PROBE, "--rows", "8", "--cols", "16", "--patches", "6", "--pop", "40",
                "--seeds", "902,903", "--ticks", "60", "--sample", "10", "--arms", "both",
                "--out", o)
        if p.returncode != 0 or not os.path.isfile(o) \
                or not os.path.isfile(os.path.splitext(o)[0] + ".summary.json"):
            print(f"🔴 {tag} 默认档微缩跑失败 rc={p.returncode}\n{_pytest_tail(p.stderr[-400:])}")
            rc = 2
        outs[tag] = o
    if rc:
        return rc
    ah, ar = _strip_csv(outs["base"])
    bh, br = _strip_csv(outs["head"])
    # summary 命名口径 = 被审件 :1773 `os.path.splitext(out)[0] + ".summary.json"`
    side = {t: os.path.splitext(p)[0] + ".summary.json" for t, p in outs.items()}
    A = json.load(io.open(side["base"], encoding="utf-8"))
    B = json.load(io.open(side["head"], encoding="utf-8"))
    ma, mb = dict(A["meta"]), dict(B["meta"])
    ma.pop("out", None), mb.pop("out", None)
    checks = [("a 表头逐字相同", ah == bh, f"{len(ah)} 列"),
              ("b 数据行逐位相同（剔计时列）", ar == br, f"{len(ar)} 行")]
    checks.append(("c meta 相同（剔 out）", ma == mb, ""))
    checks.append(("d runs 条数相同", len(A["runs"]) == len(B["runs"]), f"{len(A['runs'])} 条"))
    extra = [set(r) - set(s) for s, r in zip(A["runs"], B["runs"])]
    miss = [set(s) - set(r) for s, r in zip(A["runs"], B["runs"])]
    checks.append(("e runs 新增键集 == 声明的 2 键", all(e == NEW_KEYS for e in extra), ""))
    checks.append(("f 无键丢失", not any(miss), ""))
    diffs = [(i, k) for i, (s, r) in enumerate(zip(A["runs"], B["runs"]))
             for k in set(s) & set(r) if k not in TIME_KEYS and s[k] != r[k]]
    checks.append(("g 共有键逐值相同（剔计时键）", not diffs, f"差异 {len(diffs)} 处"))
    for name, ok, note in checks:
        print(f"Q1-{name} : {'✅' if ok else '❌'} {note}")
        if not ok:
            rc = 1
    if rc == 0:
        print(f"\n# Q1 结论：默认档 CSV/meta 逐字节等价；runs 仅多 2 个生效值键（交付须向判读线广播）")
    return rc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("mut", help="变异必红复跑")
    m.add_argument("--tree", required=True, help="被审树根（含 tests/test_s3_rdswitch.py）")
    m.add_argument("--sel", default="T7",
                   help='pytest -k 选择器；传 "T7 or T3" 可复现 c3 双测同谋假绿')
    q = sub.add_parser("q1", help="默认档逐字节等价")
    q.add_argument("--base", required=True, help="pre-S1 树根")
    q.add_argument("--head", required=True, help="post-S1 树根")
    a = ap.parse_args(argv)
    if a.cmd == "mut":
        return cmd_mut(os.path.abspath(a.tree), a.sel)
    return cmd_q1(os.path.abspath(a.base), os.path.abspath(a.head))


if __name__ == "__main__":
    sys.exit(main())
