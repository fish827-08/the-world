"""速度量化探针（14.10 尺度重标定用）—— 回答"速度基因有没有分辨力"。

背景
----
`subpos` 把位移量化为 ``steps = floor(speed × subdiv + 0.5)``，其中

    speed = clip(g18(MOBILITY) × age_factor × speed_gain, 0, speed_cap)
    speed_cap = speed_max × (lat_floor + (1−lat_floor)·cos φ)

⇒ 速度的实际分辨率 = ``speed_max × subdiv`` 档（赤道处）。
  若 ``speed_max=0.25`` 而 ``subdiv=4``（现状默认），可达档只有 ``{0, 0.25}`` 两级
  ⇒ **速度基因退化成"走/不走"的二元开关**，选择压作用面消失。

本探针直接读引擎的 ``subpos_probe()['steps_hist']``（真实运行、真实年龄结构），
对若干 (speed_max, speed_gain, subdiv) 组合报：

* **可达档数**（理论）与 **实际占用档数**（观测）
* 各档占比（是否存在"全员挤一档"的坍缩）
* **速度基因 g18 与年龄因子的实际取值范围**（回答"0.25 是基因拉满时的上限"）

用法
----
    .venv\\Scripts\\python.exe experiments/speed_quant_probe.py
    .venv\\Scripts\\python.exe experiments/speed_quant_probe.py --ticks 800 --seeds 42,7
"""
from __future__ import annotations

import argparse
import io
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simulation.config import SimConfig                      # noqa: E402
from simulation.genes import Gene                            # noqa: E402
from simulation.sphere_engine import SphereEngine            # noqa: E402


_GENE_MIN, _GENE_MAX = 0.0, 1.0


def _cfg(seed: int, speed_max: float, speed_gain: float, subdiv: int) -> SimConfig:
    """13.5 B 臂（斑块食物）+ subpos 开（subpos 与 Rust 路径互斥 ⇒ 必须 Python）。"""
    c = SimConfig(seed=seed)
    c.simulation.use_sim_core = False
    c.resources.distribution = "patchy"
    c.resources.bg_production_zero = True
    c.resources.patch_regrowth_mult = 1.195
    c.subpos.enabled = True
    c.subpos.speed_max = float(speed_max)
    c.subpos.speed_gain = float(speed_gain)
    c.subpos.subdiv = int(subdiv)
    return c


def run_one(seed: int, speed_max: float, speed_gain: float, subdiv: int,
            ticks: int) -> dict:
    eng = SphereEngine(_cfg(seed, speed_max, speed_gain, subdiv))
    for _ in range(ticks):
        eng.step()
        if len(eng._flat) == 0:
            break

    p = eng.subpos_probe()
    _gmin = float(eng.config.genome.gene_min)
    _gmax = float(eng.config.genome.gene_max)
    genes = eng._genes
    g18 = genes[:, Gene.DEFENSE].astype(np.float64)
    alive = np.ones(len(g18), dtype=bool)
    return {
        "seed": seed,
        "n_alive": int(len(g18)),
        "steps_hist": (np.asarray(p["steps_frac"], dtype=np.float64) if p else None),
        "g18_mean": float(g18.mean()) if len(g18) else None,
        "g18_std": float(g18.std()) if len(g18) else None,
        "g18_min": float(g18.min()) if len(g18) else None,
        "g18_max": float(g18.max()) if len(g18) else None,
        "speed_max": float(speed_max),
        "speed_gain": float(speed_gain),
        "subdiv": int(subdiv),
        "run_slow_frac": (float(p["slow_frac"]) if p and p.get("slow_frac") is not None else None),
    }


def speed_table(speed_max: float, speed_gain: float, subdiv: int,
                g18_vals: np.ndarray) -> dict:
    """给定基因值，复算赤道处 (speed, steps) 的解析分布（用于交叉核对）。"""
    cap = float(speed_max)                       # 赤道：cos φ = 1 ⇒ cap = speed_max
    spd = np.clip(g18_vals * float(speed_gain), 0.0, cap)
    steps = np.clip(np.floor(spd * subdiv + 0.5), 0, subdiv).astype(np.int64)
    uniq = np.unique(steps)
    return {
        "speed_mean": float(spd.mean()),
        "speed_p10": float(np.percentile(spd, 10)),
        "speed_p90": float(np.percentile(spd, 90)),
        "at_cap_frac": float(np.mean(spd >= cap - 1e-12)),
        "n_levels_used": int(len(uniq)),
        "levels": uniq.tolist(),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="速度量化 / 档位坍缩探针")
    ap.add_argument("--ticks", type=int, default=600)
    ap.add_argument("--seeds", default="42,7")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]

    # (speed_max, speed_gain, subdiv, 说明)
    combos = [
        (2.0, 4.0, 4, "现状默认：speed_max=2.0, gain=4.0, subdiv=4"),
        (0.25, 4.0, 4, "只把 speed_max 压到 0.25（gain/subdiv 不动）"),
        (0.25, 0.25, 4, "gain 同步压到 speed_max（让基因铺满量程）"),
        (0.25, 0.25, 20, "再提 subdiv 到 20（量子 0.05）"),
        (0.5, 0.5, 20, "速度 0.5 / subdiv 20（世界×0.5 的对照）"),
        (1.0, 1.0, 20, "速度 1.0 / subdiv 20"),
    ]

    print(f"== 速度量化探针：{a.ticks} tick × {len(seeds)} seed ==")
    print("（steps = floor(speed×subdiv+0.5)；speed = clip(g18×age_factor×gain, 0, speed_max×cos 相关)）\n")
    hdr = (f"{'组合':<44}{'可达档':>7}{'占用档':>7}{'量子':>7}"
           f"{'速度均值':>9}{'顶格占比':>9}{'g18均值':>9}{'零步占比':>9}")
    print(hdr)
    print("-" * len(hdr.encode("gbk", errors="replace")))

    for (smax, gain, subdiv, note) in combos:
        rows = []
        for sd in seeds:
            t0 = time.time()
            rows.append(run_one(sd, smax, gain, subdiv, a.ticks))
        hist = np.mean([r["steps_hist"] for r in rows if r["steps_hist"] is not None],
                       axis=0)
        used = int(np.count_nonzero(hist > 0.005))        # 占比 >0.5% 才算"占用"
        reach = int(round(smax * subdiv)) + 1             # 可达档数（含 0）
        g18 = np.concatenate([np.array([r["g18_mean"]]) for r in rows])
        g18m = float(np.mean([r["g18_mean"] for r in rows if r["g18_mean"] is not None]))
        # 解析复算（用实测 g18 的均值/标准差构造均匀近似不必要；直接报实测 g18 范围）
        tbl = speed_table(smax, gain, subdiv,
                          np.linspace(_GENE_MIN, _GENE_MAX, 2001))   # 全量程解析
        zero_frac = float(hist[0]) if len(hist) else float("nan")
        print(f"{note:<44}{reach:>7}{used:>7}{1.0/subdiv:>7.3f}"
              f"{tbl['speed_mean']:>9.3f}{tbl['at_cap_frac']:>9.3f}"
              f"{g18m:>9.3f}{zero_frac:>9.3f}")
        print(f"    档位占比：" + "  ".join(
            f"{i}:{v:.3f}" for i, v in enumerate(hist) if v > 0.001))
        if rows[0]["g18_min"] is not None:
            print(f"    实测 g18 范围（存活个体）："
                  f"{rows[0]['g18_min']:.3f}–{rows[0]['g18_max']:.3f}"
                  f"（均值 {rows[0]['g18_mean']:.3f}，σ {rows[0]['g18_std']:.3f}）"
                  f"｜存活 {rows[0]['n_alive']}")
        print()


if __name__ == "__main__":
    main()
