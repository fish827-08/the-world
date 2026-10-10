"""D-26a：oracle 四环节诊断（内评 `_archive/2026-10-10-退役团队-归档/_eval/D24判读预析-六门与后续-20260914.md` §4.1）。

用途
----
D-24 六门里 **G-A（oracle 阳性对照）不过**：人为接上"信号成功→有回报"通道，
仪器**本该**检出阳性，实测却没有。判据侧只给了聚合的 `oracle_return_ratio`
（0.027–0.095），**看不出卡在哪一环**。本脚本把链路拆成四环分开计数：

    ① emissions   发射事件数
      ⇒ had_signal / true_sig   （接收者落点有信号 / 其中有食物）
    ② attrib_found 归因命中（落点登记过发送者 且 发送者仍存活）
      ⇒ in_window / not_self / budget_ok
    ④ applied     实际成交（能量转移发生）

并给出**结构性上限**检查：C-9 保本封顶使
`return_ratio ≤ 1`，而单次转移 `donation` 与单次发射成本 `SIGNAL_COST` 的
比值 `donation/EMISSION_COST` 决定了"即使转化率 100%"时能到多少——
若该比值 < 1，则**无论生态怎么演化都到不了保本**（仪器参数自相矛盾）。

用法
----
    .venv\\Scripts\\python.exe experiments/d26_oracle_funnel.py --seed 42 --ticks 8000
    .venv\\Scripts\\python.exe experiments/d26_oracle_funnel.py --seeds 42,43,44 --ticks 8000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simulation.config import InfoStructureConfig, SimConfig  # noqa: E402
from simulation.oracle import EMISSION_COST  # noqa: E402
from simulation.sphere_engine import SphereEngine  # noqa: E402


# --- R98 纪律：Windows GBK 控制台兜底（非 ASCII print 会让脚本 rc=1 假失败；F-R15 族）---
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass  # 非 TTY / 旧解释器：不因诊断能力缺失而阻断运行

# 漏斗各级（顺序即因果顺序；后级 ⊆ 前级）
STAGES = (
    ("emissions", "①发射事件"),
    ("had_signal", "落点有信号"),
    ("true_sig", "落点有信号+食物"),
    ("selected", "进入 oracle 选择集"),
    ("attrib_found", "②归因命中(存活)"),
    ("in_window", "归因窗口内"),
    ("not_self", "非自反馈"),
    ("budget_ok", "C-9 额度>0"),
    ("applied", "④实际成交"),
)


def build(seed: int, ticks: int, max_count: int, donation: float | None) -> SphereEngine:
    c = SimConfig(seed=seed)
    c.simulation.ticks = ticks
    c.simulation.use_sim_core = False
    c.population.initial_count = 200
    c.population.max_count = max_count
    c.resources.distribution = "uniform"
    d2 = InfoStructureConfig(enabled=True)
    d2.learning_bottleneck = True
    d2.learning_rate = 0.05
    d2.arbitrary_codebook = True
    d2.steels_alignment = True
    c.info_structure = d2
    c.oracle.enabled = True
    if donation is not None:
        c.oracle.donation = float(donation)
    return SphereEngine(c)


def run_one(seed: int, ticks: int, max_count: int, donation: float | None) -> dict:
    e = build(seed, ticks, max_count, donation)
    for _ in range(ticks):
        if e.extinct:
            break
        e.step()
    f = e.oracle_funnel()
    return {
        "seed": seed,
        "N": len(e._id),
        "tick": int(e._tick),
        "donation": float(e.config.oracle.donation),
        "emission_cost": EMISSION_COST,
        "ratio": float(e.oracle_stats()["oracle_return_ratio"]),
        "funnel": f,
    }


def report(res: dict) -> None:
    f = res["funnel"]
    print(f"\n=== seed {res['seed']} · {res['tick']} tick · N={res['N']} "
          f"· donation={res['donation']} · SIGNAL_COST={res['emission_cost']} ===")
    prev = None
    for key, label in STAGES:
        v = f[key]
        if prev is None or prev == 0:
            keep = "—"
        else:
            keep = f"{100.0 * v / prev:6.2f}%"
        print(f"  {label:22s} {v:>12,}   留存/上级 {keep}")
        prev = v
    cap = res["donation"] / res["emission_cost"] if res["emission_cost"] else 0.0
    conv = f["applied"] / f["emissions"] if f["emissions"] else 0.0
    print(f"  {'-' * 58}")
    print(f"  转化率 applied/emissions = {conv:.4f}")
    print(f"  实测 return_ratio        = {res['ratio']:.4f}   （保本线 = 1.0）")
    print(f"  ⚠️ 结构性上限 donation/SIGNAL_COST = {cap:.3f}")
    if cap < 1.0:
        print(f"     ⇒ donation({res['donation']}) < SIGNAL_COST({res['emission_cost']})："
              f"**即使每 1 次发射都产生 1 次转移，ratio 也只能到 {cap:.3f} < 1**")
        print("     ⇒ 仪器参数自相矛盾（保本不可达）——这是机制/参数问题，不是生态问题")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42", help="逗号分隔")
    ap.add_argument("--ticks", type=int, default=8000)
    ap.add_argument("--max-count", type=int, default=3240)
    ap.add_argument("--donation", type=float, default=None,
                    help="覆盖 oracle.donation（默认沿用配置）")
    args = ap.parse_args()
    results = []
    for s in (int(x) for x in args.seeds.split(",") if x.strip()):
        r = run_one(s, args.ticks, args.max_count, args.donation)
        report(r)
        results.append(r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
