#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""镜 · G6-EMITRATE 独立复核件（只读审核，不改被审分支）

审头：`docs/审核-G6发射率列与GAP-A-20261005.md`
对象：`task/G6-EMITRATE@7fbe4ae`（+ 出处 `task/GAP-A-EMIT@e8d70f7`）

四个子命令（全部零跑批、微缩装置秒级；R117 不起跑正式批）：

    # 前置：把两枝检出成两个 detached worktree（用完即弃）
    git worktree add --detach .worktrees/mg_base 6d165e9      # merge-base(main, 7fbe4ae)
    git worktree add --detach .worktrees/mg_head 7fbe4ae

    python tools/mirror_g6_recalc.py equiv      --base .worktrees/mg_base --head .worktrees/mg_head
    python tools/mirror_g6_recalc.py window     --head .worktrees/mg_head [--ticks 300 --sample 10]
    python tools/mirror_g6_recalc.py mutations  --head .worktrees/mg_head
    python tools/mirror_g6_recalc.py liveness   --head .worktrees/mg_head
    python tools/mirror_g6_recalc.py gapa-against-head --audit <GAP-A worktree>/tools/audit_gap_a_emit.py --head .worktrees/mg_head

退出码：0 = 全部通过；1 = 出现"应当成立却不成立"的比对项（各子命令自带判读）。
纪律：本件**只调用**被审代码的公开入口，不 import 被审测试文件、不复用开发侧读数脚本。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

G6_COLS = ["g6_emit_events_cum", "g6_emit_person_ticks_cum",
           "g6_emit_events_window", "g6_emit_rate_window", "g6_emit_rate_cum"]
TIMING = {"ms_per_tick"}
MICRO = ["--rows", "8", "--cols", "16", "--patches", "6", "--pop", "40",
         "--seeds", "7,8", "--arms", "both"]


def _run_probe(wt: Path, out: Path, ticks: int, sample: int, extra=()) -> Path:
    cmd = [sys.executable, "experiments/s3_memory_probe.py", *MICRO,
           "--ticks", str(ticks), "--sample", str(sample),
           "--out", str(out), *extra]
    env = dict(os.environ, PYTHONPATH=str(wt), PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    r = subprocess.run(cmd, cwd=str(wt), capture_output=True, text=True,
                       encoding="utf-8", errors="replace", env=env)
    if r.returncode != 0:
        raise RuntimeError(f"探针退出 {r.returncode}\n{r.stderr[-2000:]}")
    return out.with_suffix(".summary.json")


def _load(path: Path, drop) -> tuple[list, list]:
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    cols = [c for c in rows[0] if c not in drop]
    return cols, [[r[c] for c in cols] for r in rows]


def _cmp(name: str, a, b) -> bool:
    ca, ra = a
    cb, rb = b
    ok = ca == cb and ra == rb
    if ca != cb:
        print(f"  {name}: 🔴 表头差 仅A={sorted(set(ca) - set(cb))} 仅B={sorted(set(cb) - set(ca))}")
    if ra != rb:
        bad = sum(1 for x, y in zip(ra, rb) for p, q in zip(x, y) if p != q)
        print(f"  {name}: 🔴 单元格差 {bad} 处")
    if ok:
        print(f"  {name}: ✅ 逐位相同（{len(ca)} 列 × {len(ra)} 行）")
    return ok


# ---------------------------------------------------------------- equiv
def cmd_equiv(args):
    base, head, work = Path(args.base), Path(args.head), Path(args.workdir or tempfile.mkdtemp())
    work.mkdir(parents=True, exist_ok=True)
    print("=== 新工具四律：默认关逐字节等价 / 仪器中立 / meta 摘项 ===")
    b = work / "base.csv"
    bsp = _run_probe(base, b, args.ticks, args.sample)
    o = work / "head_off.csv"
    osp = _run_probe(head, o, args.ticks, args.sample)
    n = work / "head_on.csv"
    nsp = _run_probe(head, n, args.ticks, args.sample, ["--m0-instruments"])
    bo = work / "base_on.csv"
    bosp = _run_probe(base, bo, args.ticks, args.sample, ["--m0-instruments"])

    e1 = _cmp("E1 默认关 CSV（head-off vs base，除 ms_per_tick）",
              _load(b, TIMING), _load(o, TIMING))
    e2 = _cmp("E2 仪器中立（head-on 剥 g6_* 除计时 vs base-on）",
              _load(bo, TIMING), _load(n, TIMING | set(G6_COLS)))

    def meta(p):
        d = json.load(open(p, encoding="utf-8"))
        return d["meta"], d["runs"]

    mb, _ = meta(bsp)
    mo, ro = meta(osp)
    mn, rn = meta(nsp)
    kb, km, kn = set(mb), set(mo), set(mn)
    diff = {"仅base": sorted(kb - km), "仅head关档": sorted(km - kb)}
    vals = {k: (mb[k], mo[k]) for k in sorted(kb & km) if k != "out" and mb[k] != mo[k]}
    e3 = not any(diff.values()) and not vals
    print(f"  E3 meta 摘项: base={len(kb)} 键 / head关档={len(km)} 键 键差={diff} 值差={vals} "
          f"{'✅' if e3 else '🔴'}")
    print(f"     head开档多出的 meta 键: {sorted(kn - kb)}（应且仅应 m0_emit_window_frac）")
    newrun = sorted({k for r in rn for k in r} - {k for r in ro for k in r})
    print(f"     head开档多出的 run 级键: {newrun}")
    print(f"小结：E1={'✅' if e1 else '🔴'} E2={'✅' if e2 else '🔴'} E3={'✅' if e3 else '🔴'}")
    return 0 if (e1 and e2 and e3) else 1


# ---------------------------------------------------------------- window
def cmd_window(args):
    head = Path(args.head)
    work = Path(args.workdir or tempfile.mkdtemp())
    out = work / "win.csv"
    sp = _run_probe(head, out, args.ticks, args.sample, ["--m0-instruments"])
    rows = list(csv.DictReader(open(out, newline="", encoding="utf-8")))
    summ = {(str(r["seed"]), r["arm"]): r for r in json.load(open(sp, encoding="utf-8"))["runs"]}
    groups = {}
    for r in rows:
        groups.setdefault((r["seed"], r["arm"]), []).append(r)
    n = len(next(iter(groups.values())))
    ki = int(args.frac * n)                       # 已裁式（PI 20:5x 班裁② index 式 k=int(0.25n)）
    kr = max(1, int(round(n * args.frac)))        # 实装式（emit_pooled:846）
    print(f"=== 两套 25% 窗切点：n={n}  int()={ki}  max(1,round())={kr} "
          f"{'⇒ 本档两者一致' if ki == kr else '⇒ 🔴 分叉'} ===")

    def seg(g, i0, i1, col):
        base = int(g[i0 - 1][col]) if i0 else 0
        return int(g[i1][col]) - base

    def seg_rate(g, i0, i1):
        ev = seg(g, i0, i1, "g6_emit_events_cum")
        pt = seg(g, i0, i1, "g6_emit_person_ticks_cum")
        return ev / pt if pt else float("nan")

    print(f"{'seed/arm':<18}{'int窗':>10}{'round窗':>10}{'summary':>10}  零码真值区间")
    worst = 0
    for key, g in sorted(groups.items()):
        ri = seg_rate(g, n - ki, n - 1) / seg_rate(g, 0, ki - 1)
        rr = seg_rate(g, n - kr, n - 1) / seg_rate(g, 0, kr - 1)
        s = summ.get(key)
        z = s["emit_zero_pattern_n"]
        tot = s["emit_events_total"]
        fpt = sum(int(g[i]["g6_emit_person_ticks_cum"])
                  - (int(g[i - 1]["g6_emit_person_ticks_cum"]) if i else 0) for i in range(ki))
        lpt = sum(int(g[i]["g6_emit_person_ticks_cum"])
                  - (int(g[i - 1]["g6_emit_person_ticks_cum"]) if i else 0)
                  for i in range(n - ki, n))
        fe = seg(g, 0, ki - 1, "g6_emit_events_cum")
        le = seg(g, n - ki, n - 1, "g6_emit_events_cum")
        if fe - z > 0:
            hi = (le / lpt) / ((fe - z) / fpt)
            lo = (max(le - z, 0) / lpt) / (fe / fpt)
            span = f"[{min(lo, hi):.4f},{max(lo, hi):.4f}]"
            cross = " 🔴跨1.0⇒方向不可判" if min(lo, hi) < 1 <= max(lo, hi) else ""
        else:
            span, cross = "首段可能被减穿", " 🔴"
        flip = "  🔴 跨1.0" if (ri < 1 <= rr) or (rr < 1 <= ri) else ""
        print(f"{key[0]+'/'+key[1]:<18}{ri:>10.4f}{rr:>10.4f}{s['emit_rate_ratio']:>10.4f}"
              f"  {z}/{tot}={z / tot:.1%} {span}{cross}{flip}")
        worst += abs(ri - rr) > 1e-9
    print(f"\n小结：两式给出不同 ratio 的 run 数 = {worst}/{len(groups)}"
          f"（`emit_pooled` 的 summary 值恒等于 round 式 ⇒ 与 int 式并存即「两套窗」）")
    print("注：切点式 PI 20:5x 班裁② 已落纸（index 式 k=int(0.25n)）⇒ 本件报的是「实装未对齐已裁文本」，"
          "改哪一行由开发做，「两窗是否同一件事」请 PI 复审（回避见审头 §一）。")
    return 0


# ---------------------------------------------------------------- mutations
MUTS = [
    ("M1", "转调前漏计数（交付自述）", "        self.g6_events += n", "        self.g6_events += 0"),
    ("M2", "分母误用采样点数（交付自述）",
     '            return (float(d_ev) / float(d_pt)) if d_pt else float("nan")',
     "            return (float(d_ev) / float(i1 - i0 + 1))"),
    ("M3", "detach 不还原类方法",
     "        cls.write_many = ent[0]\n        _G6_TAPS.pop(cls, None)",
     "        _G6_TAPS.pop(cls, None)"),
    ("M4", "去掉「有写入格却零调用」fail-loud",
     "        if self.g6_calls == 0 and int(fresh.size) > 0 and not self.g6_nan:",
     "        if False:"),
    ("M5", "去掉 _summary_meta 关档摘项",
     '    if not getattr(a, "m0_instruments", False):\n        m.pop("m0_emit_window_frac", None)',
     "    pass"),
    ("M6", "去掉 Rust 路径 NaN 兜底",
     '        if self.g6_nan:\n            return {k: float("nan") for k in _G6_NUM_COLS}',
     "        if False:\n            return {k: float('nan') for k in _G6_NUM_COLS}"),
    ("M7", "去掉「窗差分为负 ⇒ 停跑」",
     "        if d_ev < 0 or d_pt < 0:\n            raise RuntimeError(",
     "        if False:\n            raise RuntimeError("),
    ("M8", "去掉「tick 未前进 ⇒ 停跑」",
     "        if int(t) <= self.g6_t_last:\n            raise RuntimeError(",
     "        if False:\n            raise RuntimeError("),
    ("M9", "去掉「write_many 非普通函数」装配期 fail-loud",
     "        if type(raw) is not types.FunctionType:", "        if False:"),
    ("M10", "去掉「signals.write_many 不存在」fail-loud",
     '        if not callable(getattr(eng.signals, "write_many", None)):', "        if False:"),
    ("M11", "去掉「MRO 找不到定义类」fail-loud", "        if cls is None:", "        if False:"),
    ("M12", "分子改为「只数非零码」（无声改口径）",
     "        n = int(np.size(patterns))", "        n = int(np.count_nonzero(patterns))"),
]
TESTS = ["tests/test_g6_emit_columns.py", "tests/test_r358_m0_instruments.py"]


def cmd_mutations(args):
    """变异注入**只在自建的临时 detached worktree 里做** ⇒ 被审分支的工作树零改动。"""
    head = Path(args.head)
    sha = subprocess.run(["git", "-C", str(head), "rev-parse", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    root = Path(args.workdir or tempfile.mkdtemp())
    root.mkdir(parents=True, exist_ok=True)
    mut = root / "mg_mut_wt"
    if mut.exists():
        shutil.rmtree(str(mut), ignore_errors=True)
    add = subprocess.run(["git", "-C", str(head), "worktree", "add", "--detach",
                          str(mut), sha], capture_output=True, text=True)
    if not mut.exists():
        print(f"🔴 临时 worktree 建不起来：{add.stderr[-500:]}")
        return 2
    probe = mut / "experiments" / "s3_memory_probe.py"
    env = dict(os.environ, PYTHONPATH=str(mut), PYTHONIOENCODING="utf-8", PYTHONUTF8="1")

    def run_tests():
        r = subprocess.run([sys.executable, "-m", "pytest", *TESTS, "-q", "--no-header",
                            "-p", "no:cacheprovider"], cwd=str(mut), capture_output=True,
                           text=True, encoding="utf-8", errors="replace", env=env)
        out = r.stdout + r.stderr
        red = [ln.split("::")[1].split(" ")[0] for ln in out.splitlines()
               if ln.startswith("FAILED")]
        tail = [ln for ln in out.splitlines() if "passed" in ln or "failed" in ln]
        return red, (tail[-1] if tail else "?")

    def restore():
        subprocess.run(["git", "checkout", "--", "experiments/s3_memory_probe.py"],
                       cwd=str(mut), capture_output=True)

    try:
        red, summ = run_tests()
        print(f"基线（临时树 @{sha[:7]}）：{summ}")
        print("（🔴 无测试变红 = 该性质无常驻断言，只有交付时的一次实跑证据）\n")
        uncovered = []
        for tag, claim, old, new in MUTS:
            restore()                        # 先还原，再在干净文本上叠一次变异
            src = probe.read_text(encoding="utf-8")
            if old not in src:
                print(f"{tag:<5}{claim:<34} 🔴 靶文本未命中")
                uncovered.append(tag)
                continue
            probe.write_text(src.replace(old, new, 1), encoding="utf-8")
            red, summ = run_tests()
            mark = f"✅ {len(red)} 红 {red[:3]}" if red else "🔴 无测试变红"
            if not red:
                uncovered.append(tag)
            print(f"{tag:<5}{claim:<34} {summ:<26} {mark}")
        restore()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=str(mut),
                                capture_output=True, text=True).stdout.strip()
        print(f"\n小结：无守变异 = {uncovered or '无'}；临时树还原后"
              f"{'干净' if not dirty else f'🔴 脏 {dirty}'}")
    finally:
        subprocess.run(["git", "-C", str(head), "worktree", "remove", "--force", str(mut)],
                       capture_output=True)
        ver = subprocess.run(["git", "status", "--porcelain"], cwd=str(head),
                             capture_output=True, text=True).stdout.strip()
        print(f"被审工作树状态：{'未改动 ✅' if not ver else f'🔴 {ver}'}")
    return 0


# ---------------------------------------------------------------- liveness
def cmd_liveness(args):
    head = Path(args.head)
    sys.path.insert(0, str(head))
    import numpy as np
    import experiments.s3_memory_probe as P
    from experiments.s3_memory_probe import _M0Instruments

    class Sig:
        def __init__(self, n=32, duration=5):
            self._marks = np.zeros(n, np.uint8)
            self._age = np.zeros(n, np.int32)
            self.duration = duration

        def write_many(self, cells, patterns):
            c = np.asarray(cells, np.int64)
            p = np.asarray(patterns, np.uint8)
            self._marks[c] = p
            self._age[c] = np.where(p > 0, self.duration, 0).astype(np.int32)

    class Eng:
        def __init__(self, n_ind=6):
            self.signals = Sig()
            self.world = type("W", (), {"rows": 4, "cols": 8, "_pole_top": 0,
                                        "_pole_bottom": 3})()
            self._nb_table = np.tile(np.arange(32, dtype=np.int64)[:, None], (1, 8))
            self._pole_nb = np.zeros((2, 8), dtype=np.int64)
            self._flat = np.arange(1, n_ind + 1, dtype=np.int64)
            self._id = 10 + np.arange(n_ind, dtype=np.int64)
            self._genes = np.zeros((n_ind, 32))
            self._genes[:, 15] = 0.5
            self._mem_bit_on = self._mem_bit_n = 0
            self._alpha = "16"

    eng, a = Eng(), _M0Instruments()
    a.attach(eng)
    t = 0
    for _ in range(5):
        t += 1
        eng.signals.write_many([1, 2], [3, 4])
        a.after_step(eng, t, sampled=False)
    b = _M0Instruments()
    b.attach(eng)
    for _ in range(5):
        t += 1
        eng.signals.write_many([3, 4], [5, 6])
        a.after_step(eng, t, sampled=False)
    b.detach()
    a._g6_cols()
    for _ in range(5):
        t += 1
        eng.signals.write_many([5, 6], [7, 8])
        a.after_step(eng, t, sampled=True)
    d = a._g6_cols()
    a.detach()
    print("=== 「中途失效」活性检验（GAP-A §二 防的那类形态，反向）===")
    print(f"  真实发射：阶段① 10 人·次 + 阶段③ 10 人·次（b 接管期另计 10）")
    print(f"  a.g6_calls 冻结在 {a.g6_calls}（阶段③ 未增长）⇒ fail-loud 条件 `==0` 不成立")
    print(f"  阶段③ 落列：Δevents={d['g6_emit_events_window']} "
          f"rate_window={d['g6_emit_rate_window']} ⇒ 明明有发射，读数却是「归零」")
    print(f"  是否抛错：否 ⇒ 🔴 「中途失效」不在兜底范围内（「从未生效」才兜）")
    print("  现实触发路径：run_one 每 run 新建仪表 + finally 卸 ⇒ 串行不触发；"
          "并枝/并发或漏 detach 的第二仪表可触发 ⇒ 建议加"
          "「本窗有 fresh 写入却 Δevents=0 ⇒ 停跑」的活性判据。")
    return 0


# ---------------------------------------------------------------- gapa
def cmd_gapa(args):
    """把 GAP-A 的核对件放到 G6 之上跑 —— 验「两枝同合后证据是否仍复跑得动」。

    副本落在**临时目录**（`<tmp>/tools/`），不往被审工作树写一个文件。
    """
    audit = Path(args.audit)
    head = Path(args.head)
    tmp = Path(args.workdir or tempfile.mkdtemp()) / "gapa" / "tools"
    tmp.mkdir(parents=True, exist_ok=True)
    copy = tmp / "audit_gap_a_emit.py"
    shutil.copyfile(audit, copy)
    env = dict(os.environ, PYTHONPATH=str(head), PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    r = subprocess.run([sys.executable, str(copy), "--selftest"], cwd=str(tmp.parent),
                       capture_output=True, text=True, encoding="utf-8", errors="replace",
                       env=env)
    print("=== GAP-A 核对件（e8d70f7）× G6 探针（7fbe4ae）===")
    print(f"退出码 {r.returncode}")
    print("\n".join((r.stdout + r.stderr).strip().splitlines()[-8:]))
    return 0 if r.returncode == 0 else 1


def main():
    ap = argparse.ArgumentParser(description="镜 · G6 独立复核件")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, need_base=True):
        if need_base:
            p.add_argument("--base", required=True, help="merge-base worktree 路径")
        p.add_argument("--head", required=True, help="task/G6-EMITRATE worktree 路径")
        p.add_argument("--workdir", default="")
        p.add_argument("--ticks", type=int, default=120)
        p.add_argument("--sample", type=int, default=10)

    pe = sub.add_parser("equiv", help="四律逐字节实证")
    common(pe)
    pe.set_defaults(fn=cmd_equiv)
    w = sub.add_parser("window", help="两套窗切点 + 零码归属")
    common(w, need_base=False)
    w.add_argument("--frac", type=float, default=0.25)
    w.set_defaults(fn=cmd_window)
    m = sub.add_parser("mutations", help="变异必红独立复核（自建临时树，不动被审树）")
    m.add_argument("--head", required=True)
    m.add_argument("--workdir", default="")
    m.set_defaults(fn=cmd_mutations)
    lv = sub.add_parser("liveness", help="挂钩「中途失效」活性检验")
    lv.add_argument("--head", required=True)
    lv.set_defaults(fn=cmd_liveness)
    ga = sub.add_parser("gapa-against-head", help="GAP-A 核对件 × G6")
    ga.add_argument("--audit", required=True)
    ga.add_argument("--head", required=True)
    ga.add_argument("--workdir", default="")
    ga.set_defaults(fn=cmd_gapa)
    a = ap.parse_args()
    sys.exit(a.fn(a))


if __name__ == "__main__":
    main()
