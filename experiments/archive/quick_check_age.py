"""quick_check_age：生命周期年龄机制的快速长程验证（模块二收尾）。

目的
----
验证引擎第 8、7 条规则（成熟年龄门槛 + 年龄能量需求）是否带来设计效果：
  1. 未成熟（< 寿命×15%）绝不能繁衍 → 新生命不会"一出生就生"；
  2. 群落能持续演化若干代（不快速灭绝），繁衍确实在推进；
  3. 寿命基因 g3 的多样性被保留（std 不塌缩到单点）→ 同时存在
     "速生速死"（寿命短、成熟早、周转快）与"晚熟长寿"（寿命长、
     成熟晚、慢慢攒后劲）两类生存策略。

用法
----
    py -m experiments.quick_check_age [-t 6000] [-s 20260906]

依赖
----
- simulation.sphere_engine.SphereEngine
- simulation.config.SimConfig
直接读引擎内部数组（_genes / _age / _generation）做统计，不新增接口。
"""
from __future__ import annotations

import argparse

import numpy as np

from simulation.config import SimConfig, PopulationConfig
from simulation.sphere_engine import SphereEngine

MATURE_FRACTION = 0.15   # 成熟年龄 = 寿命×15%（与 config 默认对齐）
SENILE_FRACTION = 0.75


def lifespan(day: float, g3: np.ndarray) -> np.ndarray:
    """寿命（tick）= 一昼夜 × (1 + g3×7)：最短 1 昼夜、最长 8 昼夜。"""
    return day * (1.0 + g3 * 7.0)


def snapshot(engine: SphereEngine) -> dict:
    genes = engine._genes
    age = engine._age.astype(np.float64)
    life = lifespan(engine.config.light.rotation_period, genes[:, 3])
    ratio = age / life  # 已活到寿命的几成
    q = np.percentile(life, [25, 50, 75]).astype(int)
    return {
        "tick": engine.tick,
        "n": len(engine._id),
        "gen": engine._max_generation,
        "adult": float((ratio >= MATURE_FRACTION).sum()) / max(1, len(engine._id)),
        "elder": float((ratio >= SENILE_FRACTION).sum()) / max(1, len(engine._id)),
        "p": q,
        "life": life,
    }


def report(snap: dict) -> None:
    print(
        f"tick={snap['tick']:5d} 种群={snap['n']:4d} "
        f"世代={snap['gen']:3d} "
        f"成年={snap['adult']:0.2f} 老年={snap['elder']:0.2f} "
        f"g3寿命P25/P50/P75={snap['p']}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--ticks", type=int, default=4000)
    ap.add_argument("-s", "--seed", type=int, default=20260906)
    args = ap.parse_args()

    cfg = SimConfig(
        seed=args.seed,
        population=PopulationConfig(initial_count=200, max_count=1500),
        simulation=SimConfig().simulation.__class__(
            ticks=args.ticks, history_limit=200
        ),
    )
    engine = SphereEngine(cfg)
    engine.run(args.ticks)

    if engine.extinct:
        print(f"\n[终止] 第 {engine.tick} tick 灭绝")
        return

    for hs in engine.history:
        if hs.tick % 500 == 0:
            print(
                f"tick={hs.tick:5d} 种群={hs.population:4d} "
                f"born={hs.born:4d} died={hs.died:4d}"
            )

    print("\n===== 结束统计 =====")
    snap = snapshot(engine)
    report(snap)
    life = snap["life"]
    born = engine._run_born
    died = engine._run_died
    print(
        f"存活 {len(engine._id)} 只，累计出生 {born}，累计死亡 {died}"
    )
    print(
        f"寿命g3多样std = {life.std():.1f} tick "
        f"(分布 {life.min():.0f} ~ {life.max():.0f})"
    )
    if len(life) >= 20:
        fast = life[life <= np.percentile(life, 25)]
        slow = life[life >= np.percentile(life, 75)]
        print(
            f"速生(g3低) n={len(fast)} 平均寿命={fast.mean():.0f}；"
            f"晚熟(g3高) n={len(slow)} 平均寿命={slow.mean():.0f}"
        )

    ok = born > 0 and not engine.extinct and life.std() >= 200.0
    print(
        "\n结论：", "策略分化保留且持续繁衍" if ok else "未验证（需人工排查）"
    )


if __name__ == "__main__":
    main()