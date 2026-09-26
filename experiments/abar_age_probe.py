"""ā（平均机动效率）年龄结构收敛探针 —— 回答「测 ā 的标准 tick 数」。

背景（14.10 / P0.0 阶段 C 争议）
--------------------------------
云归在 300 tick 测到 ā=0.2993、幼体占比 100%；天平此前记的是 ā=0.5998。
两者差的根因假设：300 tick 时个体还没活过成熟年龄（maturity_age = maturity_fraction × lifespan），
故 age_factor 恒 = young_mob_mult(0.55) ⇒ ā 被系统性压低。

本探针把"假设"变成实测：按时间采样 **瞬时** ā（不是 engine 里的累计运行均值），
同时报年龄结构（幼体/成体/老年占比）与种群 N，并给出**收敛判定**。

口径（写死在此，避免再起争议）
------------------------------
* 总体 = **全体存活个体**（`_id` 长度 P 之前的 indices；植物不计）。
  不是"仅移动者" —— speed_gain 的用途是让**平均个体**的速度落在设计点。
* mob_eff = g18(DEFENSE→MOBILITY) × age_factor，age_factor 三档：
  age < maturity_fraction×ls → young_mob_mult；
  age ≥ senile_fraction×ls   → old_mob_mult；否则 1.0。
  （与 sphere_engine.py:3283-3295 逐字一致）
* **不含** 纬度 cap 与能量门槛（那两样是"施加后"的折减，属于另一层；
  本探针要的是"基因×年龄"这一层的 ā，正是 speed_gain 公式里的 ā）。
* 收敛判据（预注册，先算后看）：把时间轴切成 1000 tick 的窗口，
  取窗口内所有采样点的 ā 均值，**首次**满足「相邻两窗口相对差 < 1%」的窗口起点
  ⇒ 记为 T_star；报告取 **2×T_star** 作为"标准 tick 数"建议（留一倍余量）。

用法
----
python.exe experiments/abar_age_probe.py --ticks 12000 --sample 250 --seeds 42,7
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
from simulation.genes import Gene
from simulation.sphere_engine import SphereEngine


def _cfg(seed: int, patchy: bool) -> SimConfig:
    """13.5 纪元配置（斑块食物 + 背景零产出）。"""
    c = SimConfig(seed=seed)
    if patchy:
        c.resources.distribution = "patchy"
        c.resources.bg_production_zero = True
        c.resources.patch_regrowth_mult = 1.195
    return c


def sample(eng: SphereEngine) -> dict:
    """瞬时 ā 与年龄结构（全体存活个体）。"""
    P = len(eng._id)
    if P == 0:
        return {"P": 0, "abar": None, "young_frac": None, "old_frac": None,
                "g18_mean": None, "age_mean": None, "ls_p50": None}
    genes = eng._genes[:P]
    age = eng._age[:P].astype(np.float64)
    g18 = genes[:, Gene.DEFENSE].astype(np.float64)
    ls = eng._lifespan(genes[:, Gene.LIFE_GENE].astype(np.float64))
    o = eng.config.organisms
    af = np.ones(P, dtype=np.float64)
    af[age < o.maturity_fraction * ls] = o.young_mob_mult
    af[age >= o.senile_fraction * ls] = o.old_mob_mult
    return {
        "P": int(P),
        "abar": float((g18 * af).mean()),
        "young_frac": float(np.mean(age < o.maturity_fraction * ls)),
        "old_frac": float(np.mean(age >= o.senile_fraction * ls)),
        "g18_mean": float(g18.mean()),
        "age_mean": float(age.mean()),
        "ls_p50": float(np.percentile(ls, 50)),
        "af_mean": float(af.mean()),
    }


def _convergence(samples: list[dict], win: int) -> dict:
    """按 win-tick 窗口求 ā 均值，找首次相邻窗口相对差 <1% 的窗口起点。"""
    pts = [(s["t"], s["abar"]) for s in samples if s.get("abar") is not None]
    if len(pts) < 2:
        return {"T_star": None, "win": win, "windows": []}
    tmax = pts[-1][0]
    nwin = int(tmax // win)
    wins = []
    for i in range(nwin):
        lo, hi = i * win, (i + 1) * win
        vals = [a for t, a in pts if lo < t <= hi]
        if vals:
            wins.append({"lo": lo, "hi": hi, "abar": float(np.mean(vals)), "n": len(vals)})
    T_star = None
    for i in range(1, len(wins)):
        prev, cur = wins[i - 1]["abar"], wins[i]["abar"]
        if prev and abs(cur - prev) / prev < 0.01:
            T_star = wins[i - 1]["lo"]
            break
    return {"T_star": T_star, "win": win, "windows": wins}


def main() -> int:
    ap = argparse.ArgumentParser(description="ā 年龄结构收敛探针")
    ap.add_argument("--seeds", default="42,7")
    ap.add_argument("--ticks", type=int, default=12000)
    ap.add_argument("--sample", type=int, default=250, help="采样间隔 tick")
    ap.add_argument("--win", type=int, default=1000, help="收敛窗口宽度 tick")
    ap.add_argument("--no-patchy", action="store_true", help="用默认 uniform 资源")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    out = []
    for sd in seeds:
        eng = SphereEngine(_cfg(sd, not a.no_patchy))
        samples = []
        t0 = time.time()
        for t in range(1, a.ticks + 1):
            eng.step()
            if len(eng._id) == 0:
                print(f"[seed {sd}] 种群灭绝于 tick {t}", file=sys.stderr)
                break
            if t % a.sample == 0:
                s = sample(eng)
                s["t"] = t
                samples.append(s)
                el = time.time() - t0
                eta = el / t * (a.ticks - t)
                print(f"[seed {sd}] t={t:6d} N={s['P']:6d} "
                      f"ā={s['abar']:.4f} 幼={s['young_frac']*100:5.1f}% "
                      f"老={s['old_frac']*100:4.1f}% | {el:.0f}s ETA {eta:.0f}s",
                      file=sys.stderr)
        conv = _convergence(samples, a.win)
        out.append({"seed": sd, "samples": samples, "convergence": conv})
        print(f"[seed {sd}] T* = {conv['T_star']} ⇒ 建议标准 tick = "
              f"{2*conv['T_star'] if conv['T_star'] else None}", file=sys.stderr)

    print("\n=== ā 时间线（窗口均值） ===")
    for r in out:
        w = r["convergence"]["windows"]
        print(f"seed {r['seed']}: " + " ".join(
            f"[{x['hi']}]{x['abar']:.4f}" for x in w))
    res = {"ticks": a.ticks, "sample": a.sample, "win": a.win,
           "patchy": not a.no_patchy, "runs": out}
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
        print(f"已写 {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
