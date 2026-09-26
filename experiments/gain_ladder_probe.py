"""speed_gain 定档探针（解析版）—— 替代「用实测 ā 反解 gain」。

为什么要有这个探针
------------------
`experiments/abar_age_probe.py`（2026-09-26）实测出：**ā 不是一个世界常数**
—— 20 000 tick 仍在漂移，且三 seed 在 10 000 tick 的 ā = 0.331 / 0.193 / 0.420
（**极差 2.18×**）。分量拆解显示：年龄因子 `af` 很快稳在 0.63–0.68，
漂移**几乎全部来自 g18（机动基因）的演化分化**（seed 7 跌到 0.35、seed 101 涨到 0.72）。

⇒ 结论：`gain = v_max / ā` **不可用** —— 分母是会漂、且跨 seed 差 2 倍的量，
   标定完下一批就漂走，且不同 seed 的档位占用完全不同。

替代路径（本探针）
------------------
直接拿**经验联合分布** `(g18, af)`（真实跑到 T tick 后取样），对
`(v_max, subdiv, gain)` 网格**解析**算档位占用：

    speed = clip(g18 × af × gain, 0, cap)      cap = v_max（赤道）
    steps = clip(round(speed × subdiv), 0, subdiv)

报：占用档数、顶格%（steps == subdiv）、零步%（steps == 0）、中位档。
定档判据（C3 原判据 + 云归 §3.4 建议）：
    **占用档数 ≥ 5** 且 **顶格% < 5%** 且 **零步% 不作为扣分项**（低速摊薄是正确行为）。

用法
----
python.exe experiments/gain_ladder_probe.py --ticks 10000 --seeds 42,7,101
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# R98：中文 Windows 默认 GBK 控制台，print 里含 ⇒/✅ 等会 UnicodeEncodeError ⇒ rc=1
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simulation.config import SimConfig                      # noqa: E402
from simulation.genes import Gene                            # noqa: E402
from simulation.sphere_engine import SphereEngine            # noqa: E402


def _cfg(seed: int) -> SimConfig:
    """13.5 纪元配置（斑块食物 + 背景零产出）。"""
    c = SimConfig(seed=seed)
    c.resources.distribution = "patchy"
    c.resources.bg_production_zero = True
    c.resources.patch_regrowth_mult = 1.195
    return c


def collect(seed: int, ticks: int) -> tuple[np.ndarray, np.ndarray]:
    """跑到 ticks，返回存活个体的 (g18, af)。"""
    eng = SphereEngine(_cfg(seed))
    t0 = time.time()
    for t in range(1, ticks + 1):
        eng.step()
        if len(eng._id) == 0:
            print(f"[seed {seed}] 灭绝于 {t}", file=sys.stderr)
            return np.array([]), np.array([])
        if t % 2000 == 0:
            el = time.time() - t0
            print(f"[seed {seed}] t={t} N={len(eng._id)} | {el:.0f}s ETA "
                  f"{el/t*(ticks-t):.0f}s", file=sys.stderr)
    P = len(eng._id)
    genes = eng._genes[:P]
    age = eng._age[:P].astype(np.float64)
    g18 = genes[:, Gene.DEFENSE].astype(np.float64)
    ls = eng._lifespan(genes[:, Gene.LIFE_GENE].astype(np.float64))
    o = eng.config.organisms
    af = np.ones(P, dtype=np.float64)
    af[age < o.maturity_fraction * ls] = o.young_mob_mult
    af[age >= o.senile_fraction * ls] = o.old_mob_mult
    return g18, af


def ladder(g18: np.ndarray, af: np.ndarray, v_max: float, subdiv: int,
           gain: float) -> dict:
    """解析算档位占用（赤道 cap = v_max）。"""
    if g18.size == 0:
        return {}
    spd = np.clip(g18 * af * gain, 0.0, v_max)
    steps = np.clip(np.rint(spd * subdiv), 0, subdiv).astype(np.int64)
    uniq = np.unique(steps)
    # 🔴 顶格的正确定义 = **撞到速度上限 v_max**（被 clip 掉），
    #    不是 `steps == subdiv`（后者要求 speed ≥ 1 格/tick；v_max=0.25 时**永远不可能**）。
    #    第一版就写错成后者 ⇒ 全表 顶格%=0.0，判据形同虚设。
    return {
        "v_max": v_max, "subdiv": subdiv, "gain": round(gain, 4),
        "gain_over_vmax": round(gain / v_max, 3),
        "n_levels": int(len(uniq)),
        "sat_frac": float(np.mean(spd >= v_max - 1e-12)),     # 撞上限 = 真·顶格
        "at_max_step": float(np.mean(steps >= int(round(v_max * subdiv)))),
        "zero_frac": float(np.mean(steps == 0)),
        "median_step": float(np.median(steps)),
        "mean_speed": float(spd.mean()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="speed_gain 定档（解析档位占用）")
    ap.add_argument("--seeds", default="42,7,101")
    ap.add_argument("--ticks", type=int, default=10000)
    ap.add_argument("--vmax", default="0.25")
    ap.add_argument("--subdiv", default="20")
    ap.add_argument("--gain-mult", default="0.8,1.2,1.6,2.0,2.5,3.0,4.0,6.0,8.0",
                    help="gain = mult × v_max 的取值表")
    ap.add_argument("--json", default=None)
    ap.add_argument("--dump", default=None,
                    help="落盘经验联合分布 (g18, af) 到 npz（后续扫网格不必重跑世界）")
    a = ap.parse_args()

    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    vmaxs = [float(x) for x in a.vmax.split(",")]
    subdivs = [int(x) for x in a.subdiv.split(",")]
    mults = [float(x) for x in a.gain_mult.split(",")]

    out = {}
    for sd in seeds:
        g18, af = collect(sd, a.ticks)
        if g18.size == 0:
            continue
        rows = []
        for vm in vmaxs:
            for sb in subdivs:
                for m in mults:
                    rows.append(ladder(g18, af, vm, sb, m * vm))
        if a.dump:
            np.savez(a.dump.replace(".npz", "") + f"_s{sd}.npz", g18=g18, af=af)
            print(f"[dump] {a.dump.replace('.npz','')}_s{sd}.npz")
        out[str(sd)] = {
            "n": int(g18.size),
            "g18_p10": float(np.percentile(g18, 10)),
            "g18_p50": float(np.percentile(g18, 50)),
            "g18_p90": float(np.percentile(g18, 90)),
            "af_mean": float(af.mean()),
            "abar": float((g18 * af).mean()),
            "rows": rows,
        }

    print("\n=== 档位占用（判据：档数 ≥5 且 顶格% <5%；零步% 不扣分；中位档宜落在梯度中段） ===")
    print(f"{'seed':>5} {'v_max':>6} {'subdiv':>6} {'gain/vmax':>9} {'档数':>4} "
          f"{'顶格%':>6} {'零步%':>6} {'中位档':>6}")
    for sd, d in out.items():
        for r in d["rows"]:
            ok = "✅" if (r["n_levels"] >= 5 and r["sat_frac"] < 0.05) else "  "
            print(f"{sd:>5} {r['v_max']:>6} {r['subdiv']:>6} {r['gain_over_vmax']:>9} "
                  f"{r['n_levels']:>4} {r['sat_frac']*100:>6.1f} {r['zero_frac']*100:>6.1f} "
                  f"{r['median_step']:>6.1f} {ok}")
    print("\n=== 各 seed 分布摘要 ===")
    for sd, d in out.items():
        print(f"seed {sd}: N={d['n']} ā={d['abar']:.4f} af={d['af_mean']:.4f} "
              f"g18 p10/p50/p90 = {d['g18_p10']:.3f}/{d['g18_p50']:.3f}/{d['g18_p90']:.3f}")

    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        print(f"已写 {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
