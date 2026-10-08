#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""镜 · S3-RDSWITCH 独立复核件（R394④：不复用被审链路的留痕脚本）。

子命令（全部**只读**，不写被审源文件、不跑跑批之外的东西）：
  equiv     Q1 默认档逐字节等价 —— 同一微缩命令在两棵树（base / head）各跑一次，逐格比 CSV + 比 meta 键值
  switches  Q2 改档真落地 + Q3 非法档 fail-loud
  g4        Q4 `--alphabet 8` 下 G4 记忆位两臂读数（0.0 vs n/a 之辨，审头 S2 的证据）

用法示例（在任一工作树内）：
  py tools/mirror_s3_recalc.py equiv    --a ../jing_s3_base --b .
  py tools/mirror_s3_recalc.py switches --tree .
  py tools/mirror_s3_recalc.py g4       --tree .
判据与阈值不在此件里裁定，件只出「相同/不同、rc、逐臂读数」三类事实。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

PROBE = "experiments/s3_memory_probe.py"
# 微缩档：只为等价/落地取证，不用于任何生态读数
MICRO = ["--rows", "8", "--cols", "16", "--patches", "6", "--pop", "40",
         "--seeds", "902,903", "--ticks", "60", "--sample", "10", "--arms", "both"]
DROP_COLS = {"ms_per_tick", "wall_s", "out"}          # 计时/路径类必然抖动
G4_FRAC, G4_ON, G4_N = ("g4_mem_bit_frac", "g4_mem_bit_on", "g4_mem_bit_n")


def run_probe(tree: Path, tag: str, extra=(), m0: bool = True, out_dir: Path = None):
    """在一棵树里跑探针；PYTHONPATH 指向该树以便 import 被测包。"""
    out_dir = out_dir or Path(tempfile.mkdtemp(prefix="jing_s3_"))
    out = out_dir / f"{tag}.csv"
    cmd = [sys.executable, PROBE, *MICRO, *(["--m0-instruments"] if m0 else []),
           *extra, "--out", str(out)]
    env = dict(os.environ, PYTHONPATH=str(tree.resolve()),
               PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    r = subprocess.run(cmd, cwd=str(tree), capture_output=True, text=True,
                       encoding="utf-8", errors="replace", env=env)
    if r.returncode != 0:
        print(f"  [{tag}] rc={r.returncode}\n  stdout…{r.stdout[-500:]}\n"
              f"  stderr…{r.stderr[-700:]}")
    return r.returncode, out, out.with_suffix(".summary.json")


def load_csv(path: Path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = [{k: v for k, v in r.items() if k not in DROP_COLS}
                for r in csv.DictReader(f)]
    return [list(r.values()) for r in rows], rows


def cmp_grid(a, b, label, expect_same=True):
    same = a == b
    hit = same == expect_same
    if same:
        shape = f"{len(a[0])} 列 × {len(a)} 行" if a else "空"
        print(f"  {'✅' if hit else '🔴'} {label}：逐位相同（{shape}）")
    else:
        ncol = max(len(a[0]), len(b[0])) if a and b else 0
        diff = [(i, j) for i in range(min(len(a), len(b))) for j in range(ncol)
                if a[i][j] != b[i][j]]
        print(f"  {'✅' if hit else '🔴'} {label}：{len(diff)} 处不同，前 3 = "
              f"{[(i, j, a[i][j], b[i][j]) for i, j in diff[:3]]}")
    return same


def meta_of(path: Path):
    with open(path, encoding="utf-8") as f:
        m = dict(json.load(f)["meta"])
    m.pop("out", None)
    return m


def cmd_equiv(a: Path, b: Path):
    print("Q1 —— 默认档逐字节等价（两棵树同命令、均不带两开关；计时列剔除）")
    rca, ca, sa = run_probe(a, "equiv_a")
    rcb, cb, sb = run_probe(b, "equiv_b")
    if min(rca, rcb) != 0:
        return 2
    ga, _ = load_csv(ca)
    gb, _ = load_csv(cb)
    ok = cmp_grid(ga, gb, "CSV 全表", expect_same=True)
    ma, mb = meta_of(sa), meta_of(sb)
    keys = sorted(set(ma) - set(mb)), sorted(set(mb) - set(ma))
    kv = sorted(k for k in set(ma) & set(mb) if ma[k] != mb[k])
    print(f"  meta 键差 a-only={keys[0]} b-only={keys[1]}｜值差={kv or '无'}")
    dropped = [k for k in ("rd_mode", "alphabet") if k in mb]
    print(f"  {'✅' if not dropped else '🔴'} b 侧默认档 meta 摘除后不含两开关键"
          f"（实测含 = {dropped}）")
    print("Q1 =>", "PASS" if (ok and not keys[0] and not keys[1] and not kv
                             and not dropped) else "FAIL")
    return 0 if (ok and not kv and not dropped) else 1


def cmd_switches(tree: Path):
    print("Q2 —— 改档真落地：--rd-mode off --alphabet 8 的数据必须**不同于**默认档")
    rc0, c0, s0 = run_probe(tree, "sw_default")
    rc1, c1, s1 = run_probe(tree, "sw_off8",
                            ["--rd-mode", "off", "--alphabet", "8"])
    if min(rc0, rc1) != 0:
        return 2
    g0, _ = load_csv(c0)
    g1, _ = load_csv(c1)
    same = cmp_grid(g0, g1, "改档 vs 默认", expect_same=False)
    m1 = meta_of(s1)
    print(f"  改档 meta 键值 = {{'rd_mode': {m1.get('rd_mode')!r}, "
          f"'alphabet': {m1.get('alphabet')!r}}}")
    print("  ⇒ 相同 = 开关被静默忽略（🔴）；不同 = 落地生效（✅）")

    print("\nQ3 —— 非法档 fail-loud")
    rc3, _, _ = run_probe(tree, "sw_bad", ["--alphabet", "12"], m0=False)
    print(f"  {'✅' if rc3 != 0 else '🔴'} rc={rc3}（非 0 即 fail-loud）")
    return 0 if (not same and rc3 != 0) else 1


def cmd_g4(tree: Path):
    print("Q4 —— `--alphabet 8` 下 G4 两臂读数（审头 S2：0.0 与 n/a 之辨）")
    rc, csv_path, _ = run_probe(tree, "g4_off8", ["--rd-mode", "off", "--alphabet", "8"])
    if rc != 0:
        return 2
    _, rows = load_csv(csv_path)
    for arm in ("mem_on", "mem_off"):
        sel = [r for r in rows if r.get("arm") == arm]
        print(f"  arm={arm}｜{G4_FRAC}[:6]={[r[G4_FRAC] for r in sel[:6]]}"
              f"｜{G4_ON}[:6]={[r[G4_ON] for r in sel[:6]]}"
              f"｜{G4_N} 首末={sel[0][G4_N] if sel else '-'}→{sel[-1][G4_N] if sel else '-'}")
    print("  口径：mem_on 臂 = memory_v2=True ⇒ 引擎不再写/读 `_work_memory`"
          "（sphere_engine.py:3585）⇒ mem_bit≡0 而分母非零 ⇒ 列值是 0.0 而不是 n/a；"
          "作者在 :1461-1468 有一次性 stderr 告警，但列值本身仍不区分「不可测」与「实测 0%」。")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    pe = sp.add_parser("equiv")
    pe.add_argument("--a", required=True, help="基线树（如 d659141）")
    pe.add_argument("--b", required=True, help="被审树（如 24d1fd8）")
    ps = sp.add_parser("switches")
    ps.add_argument("--tree", default=".")
    pg = sp.add_parser("g4")
    pg.add_argument("--tree", default=".")
    a = ap.parse_args()
    if a.cmd == "equiv":
        sys.exit(cmd_equiv(Path(a.a), Path(a.b)))
    if a.cmd == "switches":
        sys.exit(cmd_switches(Path(a.tree)))
    sys.exit(cmd_g4(Path(a.tree)))


if __name__ == "__main__":
    main()
