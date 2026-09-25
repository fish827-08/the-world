"""ARS 扫参 / 对照批（14.9）——把卡点 1/卡点 6 的旋钮做成 CLI，输出结构化 CSV。

用途
----
给"云/本地并行派工"用的**统一入口**：一个命令跑完一整张参数网格，
同一份脚本 ⇒ 同一份前提（C8 前提对账的最大风险就是各写一遍脚本）。

三个被扫的旋钮（对应卡点 1/卡点 6）
-----------------------------------
* ``--regrowth``  = ``resources.patch_regrowth_mult``（**斑块耗尽标定** —— 卡点1）
* ``--subpos``    = ``subpos.speed_max``（0 = 不启用；= 减速 ⇒ 世界体感变大）
* ``--theta/--fast-tau/--slow-tau/--gain/--kappa/--giveup`` = ARS 自身旋钮

🔴 纪律（照 §14.6 / R127 / R189）
-------------------------------
* **不改世界底层规则**：基础配置固定为 13.5 B 臂（patchy + bgzero），只动上面的旋钮；
* **必须先在构造期设置 ARS 参数**（曾经因为构造后才改而"扫了没生效"）⇒ 本脚本全部在 ``make_cfg`` 内设置；
* **进度**：单行覆盖式进度条（非 TTY 自动静默；``ARS_SWEEP_PROGRESS=0`` 可关）；
* **ETA 只用实测样本推**（不做瞬时速率外推）。

输出
----
屏幕打印「按 arm 汇总」表（每个 arm 在各 seed 上的创始代荒漠出生存活率 + 赶路占比）；
若给了 ``--out`` 则额外落盘一份 CSV（每行 = 一个 (组合, seed, 读数时刻)）。

用法示例
--------
::

    # 卡点1：斑块耗尽标定（3 再生档 × 2 速度档 × ARS 关/开）
    python.exe experiments/ars_sweep.py --regrowth 1.195,0.6,0.3 --subpos 0,0.25 ^
        --gain 0,3 --out results/ars_calib_depletion.csv

    # 卡点6：期待时间常数（τ 必须与能量预算同量级）
    python.exe experiments/ars_sweep.py --slow-tau 100,300,500 --fast-tau 10,50 ^
        --theta 0.5,0.8 --gain 3 --out results/ars_calib_tau.csv

    # 冒烟 §12.2.1：先跑 1 run 验证前提（强制 ≤2k tick）
    python.exe experiments/ars_sweep.py --max-runs 1 --t-read 500 --t-end 1000
"""
from __future__ import annotations

import argparse
import csv
import itertools
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

from simulation.config import SimConfig  # noqa: E402
from simulation.sphere_engine import SphereEngine  # noqa: E402

# ---------------------------------------------------------------- 进度条

_TTY = sys.stdout.isatty() and os.environ.get("ARS_SWEEP_PROGRESS", "1") != "0"


def _progress(done: int, total: int, t0: float, label: str) -> None:
    """单行覆盖式进度（ETA 只用已耗时的线性外推，禁用瞬时速率）。"""
    if not _TTY or total <= 0:
        return
    frac = done / total
    bar_len = 28
    filled = int(bar_len * frac)
    bar = "#" * filled + "." * (bar_len - filled)
    elapsed = time.time() - t0
    eta = (elapsed / done * (total - done)) if done else float("nan")
    msg = f"\r  [{bar}] {done}/{total} {frac*100:5.1f}% | {label:<34.34} | 已用 {elapsed/60:.1f}m ETA {eta/60:.1f}m"
    sys.stdout.write(msg)
    sys.stdout.flush()
    if done >= total:
        sys.stdout.write("\n")


# ---------------------------------------------------------------- 配置

def make_cfg(
    seed: int,
    ars: bool,
    gain: float,
    theta: float,
    kappa: float,
    giveup: int,
    fast_tau: float,
    slow_tau: float,
    regrowth: float,
    speed_max: float,
) -> SimConfig:
    """13.5 B 臂（patchy + bgzero）为底盘；**所有 ARS 参数在构造前设置**。"""
    c = SimConfig(seed=seed)
    c.simulation.use_sim_core = False          # ARS 只在 Python 路径（H3-A1）
    c.simulation.l2_dash = False               # 与 subpos 语义冲突 ⇒ 构造期硬报错
    c.population.max_count = 600
    c.predation.forage_tradeoff_k = 0.0
    c.resources.distribution = "patchy"
    c.resources.bg_production_zero = True
    c.resources.patch_count = 30
    c.resources.patch_radius = 2
    c.resources.patch_regrowth_mult = float(regrowth)
    o = c.organisms
    o.max_energy = 600.0
    o.initial_energy = 300.0
    o.starve_frac = 0.30
    o.exhaust_frac = 0.17
    o.eat_efficiency = 7.5
    o.assim_herb = 0.4
    o.assim_carn = 0.8
    o.stomach_cap_mass = 25.0
    o.eat_threshold_frac = 0.6
    o.photo_max = 0.0
    # ---- ARS（务必在构造前！）----
    c.ars.enabled = bool(ars)
    c.ars.gain = float(gain)
    c.ars.theta = float(theta)
    c.ars.kappa = float(kappa)
    c.ars.giveup = int(giveup)
    c.ars.fast_tau = float(fast_tau)
    c.ars.slow_tau = float(slow_tau)
    # ---- subpos 减速（speed_max=0 ⇒ 不启用，回到旧"每 tick 1 格"）----
    if speed_max > 0:
        c.subpos.enabled = True
        c.subpos.speed_max = float(speed_max)
    return c


# ---------------------------------------------------------------- 主流程

FIELDS = [
    "arm", "seed", "read_tick",
    "ars_on", "gain", "theta", "kappa", "giveup", "fast_tau", "slow_tau",
    "regrowth", "speed_max",
    "founder_desert_n", "surv_n", "surv_frac", "N", "gen",
    "ars_dec_n", "ars_extensive_frac", "ars_sw_ie_n", "ars_sw_ei_n", "ars_rerand_n",
]


def _f(x, nd: int = 6):
    return round(float(x), nd)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="ARS 扫参/对照批（14.9）")
    ap.add_argument("--seeds", default="42,7,11", help="世界种子（推断单位=世界种子，禁个体级）")
    ap.add_argument("--t-read", default="1000", help="读数时刻（可逗号分隔多个）")
    ap.add_argument("--t-end", type=int, default=3000)
    ap.add_argument("--gain", default="0,3", help="0 ⇒ ARS 关；>0 ⇒ ARS 开且用该 gain")
    ap.add_argument("--theta", default="0.5")
    ap.add_argument("--kappa", default="0")
    ap.add_argument("--giveup", default="20")
    ap.add_argument("--fast-tau", default="50")
    ap.add_argument("--slow-tau", default="500")
    ap.add_argument("--regrowth", default="1.195", help="斑块再生倍率（<1.195 ⇒ 更易耗尽）")
    ap.add_argument("--subpos", default="0", help="subpos.speed_max；写 0 ⇒ 不启用 subpos")
    ap.add_argument("--out", default="", help="CSV 输出路径（默认不落盘，只打屏）")
    ap.add_argument("--max-runs", type=int, default=0, help="0=不限；调试用")
    a = ap.parse_args(argv)

    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    read_ts = [int(t) for t in str(a.t_read).split(",") if t.strip()]
    grid = dict(
        gain=[float(v) for v in a.gain.split(",") if v.strip()],
        theta=[float(v) for v in a.theta.split(",") if v.strip()],
        kappa=[float(v) for v in a.kappa.split(",") if v.strip()],
        giveup=[int(v) for v in a.giveup.split(",") if v.strip()],
        fast_tau=[float(v) for v in a.fast_tau.split(",") if v.strip()],
        slow_tau=[float(v) for v in a.slow_tau.split(",") if v.strip()],
        regrowth=[float(v) for v in a.regrowth.split(",") if v.strip()],
        speed_max=[float(v) for v in a.subpos.split(",") if v.strip()],
    )
    keys = list(grid)
    combos = [dict(zip(keys, vals)) for vals in itertools.product(*(grid[k] for k in keys))]
    runs = [(combo, seed) for combo in combos for seed in seeds]
    if a.max_runs:
        runs = runs[: a.max_runs]

    print(f"== ARS 扫参批：{len(runs)} run（{len(combos)} 组合 × {len(seeds)} seed）"
          f"｜读数 t={read_ts}｜t_end={a.t_end} ==")
    print(f"   网格：{ {k: v for k, v in grid.items() if len(v) > 1} }")
    rows = []
    t0 = time.time()
    for i, (combo, seed) in enumerate(runs, 1):
        ars = combo["gain"] > 0
        arm = (f"g{combo['gain']:.3g}_th{combo['theta']:.2g}_k{combo['kappa']:.2g}"
               f"_rg{combo['regrowth']:g}_sp{combo['speed_max']:g}")
        _progress(i - 1, len(runs), t0, f"{arm} seed={seed}")
        cfg = make_cfg(seed, ars, combo["gain"], combo["theta"], combo["kappa"],
                       combo["giveup"], combo["fast_tau"], combo["slow_tau"],
                       combo["regrowth"], combo["speed_max"])
        try:
            e = SphereEngine(cfg)
        except AssertionError as exc:                      # fail-loud：参数不合法就停
            print(f"\n🔴 构造失败 {arm} seed={seed}: {exc}")
            return 2
        grid0 = np.asarray(e.resources._grid)
        n0 = len(e._flat)
        desert0 = e._id[:n0][grid0[e._flat[:n0]] <= 0.0]   # 创始代荒漠出生子集（主判据口径）
        d0 = int(len(desert0))
        snapshots = {}
        for t in range(1, a.t_end + 1):
            e.step()
            if t in read_ts:
                alive = np.isin(e._id[: len(e._flat)], desert0)
                snapshots[t] = (
                    _f(alive.mean() if d0 else float("nan"), 4), int(alive.sum()),
                    int(len(e._flat)), int(e._max_generation),
                )
        p = e.ars_probe() or {}
        for t, (frac, k, n, gen) in snapshots.items():
            rows.append({
                "arm": arm, "seed": seed, "read_tick": t,
                "ars_on": int(ars), "gain": combo["gain"], "theta": combo["theta"],
                "kappa": combo["kappa"], "giveup": combo["giveup"],
                "fast_tau": combo["fast_tau"], "slow_tau": combo["slow_tau"],
                "regrowth": combo["regrowth"], "speed_max": combo["speed_max"],
                "founder_desert_n": d0, "surv_n": k, "surv_frac": frac,
                "N": n, "gen": gen,
                "ars_dec_n": p.get("ars_dec_n"),
                "ars_extensive_frac": p.get("ars_extensive_frac"),
                "ars_sw_ie_n": p.get("ars_sw_ie_n"),
                "ars_sw_ei_n": p.get("ars_sw_ei_n"),
                "ars_rerand_n": p.get("ars_rerand_n"),
            })
    _progress(len(runs), len(runs), t0, "done")

    # ---- 屏幕摘要：按 arm 汇总首个读数时刻 ----
    print("\n=== 汇总（首个读数时刻，按 arm 分列；均值跨 seed）===")
    t_ref = read_ts[0]
    arms = sorted({r["arm"] for r in rows})
    print(f"{'arm':<26}{'存活率(各seed)':<28}{'均值':>8}  赶路占比")
    for arm in arms:
        rs = [r for r in rows if r["arm"] == arm and r["read_tick"] == t_ref]
        vals = " / ".join(f"{v*100:.1f}%" for v in (r["surv_frac"] for r in rs))
        mean = _f(np.mean([r["surv_frac"] for r in rs]) * 100, 1)
        ext = _f(np.nanmean([r["ars_extensive_frac"] for r in rs
                             if r["ars_extensive_frac"] is not None]), 3) \
            if any(r["ars_extensive_frac"] is not None for r in rs) else None
        print(f"{arm:<26}{vals:<28}{mean:>7.1f}%  {ext}")

    if a.out:
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
        print(f"\n✅ CSV 已写入：{out}（{len(rows)} 行）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
