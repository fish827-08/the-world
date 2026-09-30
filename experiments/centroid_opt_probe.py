"""R273 质心优化探针：**同 seed 增量 vs 全量逐位对拍 + 计时**（S3 装置口径）。

用途（R273 §三 验收 ①③ 的机械化证据）：
  · 对拍：每 `--check-every` tick 把引擎 `_patch_centroid`（增量维护）与
    `_build_patch_centroid(当前掩码)`（全量重算）逐位比较 ⇒ 断言相等；
  · 计时：每段（`--check-every`）输出 ms/tick + 掩码变化 tick 数；
  · 计数：monkeypatch 统计 全量/增量 调用次数（关档臂应恒 0/0；开档臂全量应恒 1=构造期）；
  · 关档臂（mem_off）断言质心表恒 -1（L1 门控：关档零质心工作）。

用法（本机 venv）：
    python -m experiments.centroid_opt_probe --seed 104 --ticks 250 --arm both
    python -m experiments.centroid_opt_probe --seed 101 --ticks 250 --arm on   # 留守型对照

装置 = `experiments/s3_memory_probe.run_one` 的同源构造（make_cfg + apply_post_build）。
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from simulation.sphere_engine import SphereEngine
from experiments.steady_k_probe import make_cfg, apply_post_build

DEVICE = dict(
    rows=480, cols=960, patches=1700, pop=10000,
    speed_max=0.125, gain=0.125, subdiv=80, k=2.5, max_count=30000,
    rgm=1.195, bg_low_prod_frac=0.4, bg_low_cap_mult=0.05,
)


def _instrument() -> dict:
    """在**模块全局**上挂计数器（引擎按名字运行期解析 ⇒ 生效；探针结束不还原）。"""
    import simulation.sphere_engine as se
    cnt = {"full": 0, "inc": 0}
    orig_f, orig_i = se._build_patch_centroid, se._update_patch_centroid_incremental

    def _cf(*a, **k):
        cnt["full"] += 1
        return orig_f(*a, **k)

    def _ci(*a, **k):
        cnt["inc"] += 1
        return orig_i(*a, **k)

    se._build_patch_centroid = _cf
    se._update_patch_centroid_incremental = _ci
    cnt["_orig_full"] = orig_f
    return cnt


def run_arm(seed: int, ticks: int, mem_on: bool, check_every: int,
            cnt: dict) -> tuple[int, float, int]:
    """跑单臂；返回 (掩码变化 tick 数, 全程 ms/tick, 对拍次数)。"""
    c, notes = make_cfg(
        seed, DEVICE["rows"], DEVICE["cols"], DEVICE["pop"], DEVICE["patches"],
        True, DEVICE["speed_max"], DEVICE["gain"], DEVICE["subdiv"],
        DEVICE["k"], DEVICE["max_count"], DEVICE["rgm"],
        bg_low_prod_frac=DEVICE["bg_low_prod_frac"],
        bg_low_cap_mult=DEVICE["bg_low_cap_mult"],
    )
    c.simulation.use_sim_core = False
    c.resource_dynamics.enabled = True
    if mem_on:
        c.info_structure.memory_v2 = True
        c.info_structure.memory_gradient = "orientation"
    else:
        c.info_structure.memory_v2 = False
        c.info_structure.memory_gradient = "none"
    eng = SphereEngine(c)
    apply_post_build(eng, notes)
    assert bool(eng.config.resources.bg_production_zero) is True

    n_chg = 0
    n_check = 0
    prev = (eng.resources._patch_mask.copy()
            if eng.resources._patch_mask is not None else None)
    t0 = time.perf_counter()
    t_slice, last = t0, 0
    for t in range(1, ticks + 1):
        eng.step()
        cur = eng.resources._patch_mask
        if prev is not None and cur is not None and not np.array_equal(prev, cur):
            n_chg += 1
        prev = cur.copy() if cur is not None else None
        if t % check_every == 0 or t == ticks:
            dt_ms = (time.perf_counter() - t_slice) / max(t - last, 1) * 1e3
            t_slice, last = time.perf_counter(), t
            if cur is not None:
                if mem_on:
                    exp = cnt["_orig_full"](eng.world, cur)
                    ok = np.array_equal(eng._patch_centroid, exp)
                    n_check += 1
                    assert ok, f"🔴 tick {t}: 增量表 ≠ 全量重算（逐位对拍失败）"
                else:
                    assert bool((eng._patch_centroid == -1).all()), (
                        f"🔴 tick {t}: 关档臂质心表非初值（L1 门控失效）")
            print(f"  seed={seed} arm={'mem_on ' if mem_on else 'mem_off'} "
                  f"tick={t:4d} ms/tick={dt_ms:8.1f} 掩码变化累计={n_chg}")
    wall_ms = (time.perf_counter() - t0) / max(ticks, 1) * 1e3
    return n_chg, wall_ms, n_check


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=104)
    ap.add_argument("--ticks", type=int, default=250)
    ap.add_argument("--arm", choices=("both", "on", "off"), default="both")
    ap.add_argument("--check-every", type=int, default=50)
    args = ap.parse_args()

    cnt = _instrument()
    arms = {"both": (True, False), "on": (True,), "off": (False,)}[args.arm]
    for mem_on in arms:
        cnt["full"] = cnt["inc"] = 0
        n_chg, wall_ms, n_check = run_arm(
            args.seed, args.ticks, mem_on, args.check_every, cnt)
        n_fb = max(cnt["full"] - 1, 0) if mem_on else 0
        print(f"  小结 seed={args.seed} arm={'mem_on' if mem_on else 'mem_off'}："
              f"全程 {wall_ms:.1f} ms/tick｜掩码变化 {n_chg} tick｜对拍 {n_check} 次全过｜"
              f"调用计数 full={cnt['full']}（含极区回退 {n_fb}）inc={cnt['inc']}")
    print("✅ 探针完成（逐位对拍全过；计数口径：开档 full = 1〔构造期〕+ 极区回退次数 "
          "〔增量触及 {0,1,rows-2,rows-1} 行 ⇒ 返回 False ⇒ 引擎回退全量〕、"
          "关档 full=inc=0）")


if __name__ == "__main__":
    main()
