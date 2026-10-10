"""480×960 世界 / 满载 2 万+ 个体的**机时与内存预算**（纯算术，不跑世界）。

为什么是脚本而不是一段话
------------------------
fish 要的是"先数学上算一下"。把公式与数字**写死在文档里**必然腐坏（教训库 ⑲）；
写成脚本 ⇒ 改一个世界尺寸就能重算，且**口径写死在注释里**（避免再起争议）。

两个性能模型（**都用**，给区间而不是单一数字）
--------------------------------------------
* **A（评审拟合，旧）**：`ms/tick ≈ 0.055·cells/1000 + 0.006·N + 0.4`
  来源：早期探针（`_archive/2026-10-10-退役团队-归档/docs-旧档/tasks/派工-尺度重标定与稀疏化-20260926.md` §性能）。
* **B（480×960 实测，新）**：`ms/tick ≈ 20.7 + 0.0138·N`
  来源：稳态 K 实测（N 从 64 → 4 997 拟合；`_archive/2026-10-10-退役团队-归档/docs-旧档/tasks/答复-第二轮评审-20260926.md` §188-205）。

⚠️ **两模型的分歧点 = 个体边际成本**：A 说 6 μs/体，B 说 **13.8 μs/体**（差 2.4×，原因未定位，
已登记为"疑似还有一处 O(cells) 项"）。⇒ **外推到 N=22 000（超出 B 的拟合域 4.4×）时
必须给区间、并标注"未验证"**，不得只报一个数。

用法
----
python.exe experiments/perf_budget_calc.py
python.exe experiments/perf_budget_calc.py --rows 480 --cols 960 --n 22000 --ticks 20000
"""

from __future__ import annotations

import argparse
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── 模型常数（带出处，改这里必须同步改文档） ────────────────────────────────
A_CELLS_PER_1K = 0.055     # ms / 1000 格（模型 A）
A_PER_IND = 0.006          # ms / 个体（模型 A）
A_FIXED = 0.4              # ms（模型 A）
B_FIXED = 20.7             # ms（模型 B，480×960 实测截距）
B_PER_IND = 0.0138         # ms / 个体（模型 B，实测斜率）

NB_STRIDE_MIN = 8          # 邻居表列数下限（sphere_engine.py:575）
COMPACT_STRIDE = 8         # P0.1 紧凑化后的列数
BYTES_I64 = 8

# 单机规格（**实测**，来源：讨论板 2026-09-26 云归/云启硬件档案）
#   cores = cgroup CPU 配额（硬约束，不是可见核数）
#   par_eff = 实测聚合加速比 / 路数（云启 2 路实测聚合 1.33× ⇒ 0.67；云归 3.93/4.0 ⇒ 0.98）
MACHINES = {
    "云归": dict(ram_gib=8.0, cores=4.0, par_eff=0.98,
                 note="cgroup 4.0 核 / 8 GiB / rustc 1.93 / 无 swap"),
    "云启": dict(ram_gib=4.0, cores=2.0, par_eff=0.67,
                 note="cgroup 2.0 核 / 4 GiB / rustc 1.92"),
}
RUN_OVERHEAD_MB = 300.0    # 除邻居表外的常驻（格域场 + 个体数组 + Python 解释器）估


def ms_tick(cells: int, n: int) -> tuple[float, float]:
    a = A_CELLS_PER_1K * cells / 1000.0 + A_PER_IND * n + A_FIXED
    b = B_FIXED + B_PER_IND * n
    return a, b


def nb_bytes(cells: int, cols: int, stride: int) -> float:
    return cells * max(NB_STRIDE_MIN, cols if stride is None else stride) * BYTES_I64


def main() -> int:
    ap = argparse.ArgumentParser(description="机时/内存预算（纯算术）")
    ap.add_argument("--rows", type=int, default=480)
    ap.add_argument("--cols", type=int, default=960)
    ap.add_argument("--n", default="2000,5000,22000", help="个体数（逗号分隔）")
    ap.add_argument("--ticks", type=int, default=20000, help="单 run tick 数")
    ap.add_argument("--runs", type=int, default=6, help="批的 run 数")
    a = ap.parse_args()

    cells = a.rows * a.cols
    ns = [int(x) for x in a.n.split(",")]
    nb_now = nb_bytes(cells, a.cols, None)
    nb_cpt = nb_bytes(cells, a.cols, COMPACT_STRIDE)

    print(f"世界 {a.rows}×{a.cols} = {cells:,} 格")
    print(f"邻居表（现状 stride=max(8,{a.cols})）：{nb_now/1024**3:.2f} GiB "
          f"({nb_now/1024**2:,.0f} MB)")
    print(f"邻居表（P0.1 紧凑 stride=8）：{nb_cpt/1024**2:,.1f} MB "
          f"⇒ 省 {nb_now/nb_cpt:.0f}×")
    print()

    print(f"{'N':>7} | {'ms/tick A':>9} {'ms/tick B':>9} | "
          f"{'单run机时 A':>11} {'单run机时 B':>11} | {'批(6run) A':>10} {'批(6run) B':>10}")
    print("-" * 78)
    for n in ns:
        ma, mb = ms_tick(cells, n)
        ha = ma * a.ticks / 1000.0 / 60.0
        hb = mb * a.ticks / 1000.0 / 60.0
        print(f"{n:>7,} | {ma:>9.1f} {mb:>9.1f} | "
              f"{ha:>10.1f}m {hb:>10.1f}m | {ha*a.runs/60:>9.1f}h {hb*a.runs/60:>9.1f}h")
    print(f"（口径：{a.ticks:,} tick/run × {a.runs} run；机时 = ms/tick×tick 数）")
    print()

    print("=== 并行度 = min(内存路数, CPU 配额路数)；聚合吞吐 = 路数 × 实测并行效率 ===")
    tot = {"现状": 0.0, "P0.1后": 0.0}
    for name, m in MACHINES.items():
        ram_mb = m["ram_gib"] * 1024
        for tag, nb in (("现状", nb_now), ("P0.1后", nb_cpt)):
            per_run = nb / 1024**2 + RUN_OVERHEAD_MB
            p_mem = int(ram_mb // per_run) if per_run else 0
            p = min(max(p_mem, 0), int(m["cores"]))
            thr = p * m["par_eff"]
            tot[tag] += thr
            ok = "✅" if p >= 1 else "❌ 跑不了"
            print(f"  {name}（{m['note']}）{tag}: 单 run ≈ {per_run:,.0f} MB ⇒ "
                  f"内存可 {p_mem} 路 / CPU 配额 {int(m['cores'])} 路 ⇒ **并行 {p} 路** "
                  f"（聚合 {thr:.2f}×）{ok}")
    print(f"  ⇒ 两台合计聚合吞吐：现状 {tot['现状']:.2f}× ／ P0.1 后 {tot['P0.1后']:.2f}× "
          f"（**提升 {tot['P0.1后']/max(tot['现状'],1e-9):.2f}×**）")
    print()
    n_ref = ns[-1]
    mb = ms_tick(cells, n_ref)[1]
    h_serial = mb * a.ticks / 1000.0 / 60.0 * a.runs / 60.0
    for tag in ("现状", "P0.1后"):
        print(f"  N={n_ref:,}、{a.runs} run × {a.ticks:,} tick 的批："
              f"串行 {h_serial:.1f} h ⇒ {tag} 并行后 {h_serial/max(tot[tag],1e-9):.1f} h")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
