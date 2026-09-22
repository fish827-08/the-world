"""R163 独立复核：尸体—食腐通道的裁决性探针（老工线，2026-09-22）。

**用途**：把 R161/R163 的四项复核变成可复现证据（纯 ASCII 输出、零 RNG、
直调引擎内部函数、不改任何仓库文件）。

四项复核：
  1. 两个缺陷（P1-2 食腐超发 / P1-3 投尸同格丢失）在**修复前版本**上复现
  2. 修复后两场景**逐格守恒** + 随机重复
  3. 全量测试（由调用方在外层跑，见下）
  4. `alloc = min(demand, supply)` 分配逻辑的边界（demand=0 / supply=0 / 浮点）

**外加两项单位审计**（本探针的独立发现，见 §4）：
  - 尸体池（**能量**单位）写进 `stomach`（**植物质量**池，消化 ×`eat_efficiency`）
  - 腐烂归还植物池时被 `capacity − grid` 截断的比例

复现「修复前」两步（不动工作树，全部在仓库外）：
    TMP=/c/Users/圣羽/Desktop/temp
    git archive 800d258^ | tar -x -C $TMP/_r163_old
    cd $TMP/_r163_old && <venv-python> $TMP/_r163_probe.py     # 应重现 4.78 / −3.78 与 9.0
    cd <repo>            && <venv-python> tools/r163_corpse_audit.py   # 应恒守恒

全量测试（`--basetemp` 必须是 **Windows 路径**，bash 风格 `/c/...` 会被解析成 `C:\\c\\...`）：
    .venv/Scripts/python.exe -m pytest tests/ -q --ignore=tests/test_broker.py \
        -p no:cacheprovider --basetemp "C:/Users/<you>/Desktop/temp/_bt"
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.getcwd())
import numpy as np  # noqa: E402

from simulation.config import SimConfig  # noqa: E402
from simulation.sphere_engine import SphereEngine  # noqa: E402
import simulation.sphere_engine as se  # noqa: E402


def mk(cap=200, max_count=600, seed=42):
    c = SimConfig(seed=seed)
    c.simulation.use_sim_core = False
    c.population.max_count = max_count
    cw = c.corpse_wound
    cw.corpse_enabled = True
    cw.wound_enabled = True
    cw.contest_enabled = True
    cw.corpse_cap_per_cell = int(cap)
    return SphereEngine(c)


def sec1_scavenge(e, n_ind, g16, supply, cell=100):
    P = len(e._id)
    idx = np.arange(n_ind)
    e._flat[:P] = 777
    e._flat[idx] = cell
    e._genes[:P, 16] = g16
    e._stomach[:P] = 0.0
    e._corpse_energy[:] = 0.0
    e._corpse_energy[cell] = supply
    b_corpse = float(e._corpse_energy.sum())
    b_stom = float(e._stomach[:P].sum())
    e._step_scavenging(P, e._stomach, np.full(P, 1e9), e._genes)
    take = float(e._stomach[:P].sum()) - b_stom
    a_corpse = float(e._corpse_energy.sum())
    return take, float(e._corpse_energy[cell]), (b_corpse - a_corpse), take > supply + 1e-9


def sec2_deposit(e, n_dead, each, cell=200):
    P = len(e._id)
    e._corpse_energy[:] = 0.0
    e._corpse_age[:] = 0
    dead = np.zeros(P, dtype=bool)
    dead[:n_dead] = True
    e._flat[:P] = 777
    e._flat[:n_dead] = cell
    energy = np.zeros(P, dtype=np.float64)
    energy[:n_dead] = each / float(e.config.corpse_wound.corpse_energy_frac)
    e._deposit_corpse(dead, energy)
    return float(e._corpse_energy[cell])


def sec3_random(e, trials, seed=7):
    rng = np.random.default_rng(seed)
    P = len(e._id)
    bad, cov = [], {"supply0": 0, "demand0": 0, "supply<demand": 0}
    for t in range(trials):
        n = int(rng.integers(1, min(P, 40) + 1))
        cell = int(rng.integers(0, e.world.n_cells))
        supply = float(rng.choice([0.0, 1e-12, 1e-3, 0.5, 3.0, 50.0, 1e6]))
        g = float(rng.choice([0.0, 1e-9, 0.3, 0.5, 0.7, 1.0]))
        room = float(rng.choice([1e-9, 0.01, 0.5, 1e9]))
        e._flat[:P] = 777
        e._flat[:n] = cell
        e._genes[:P, 16] = g
        e._stomach[:P] = 0.0
        e._corpse_energy[:] = 0.0
        e._corpse_energy[cell] = supply
        b0, s0 = float(e._corpse_energy.sum()), float(e._stomach[:P].sum())
        e._step_scavenging(P, e._stomach, np.full(P, max(room, 0.0) if room > 0 else 0.0), e._genes)
        take = float(e._stomach[:P].sum()) - s0
        ded = b0 - float(e._corpse_energy.sum())
        cov["supply0"] += supply <= 0.0
        cov["demand0"] += g <= 1e-9
        cov["supply<demand"] += (0.0 < supply < 1e6) and room > 100.0
        errs = []
        if not np.isfinite(take):
            errs.append("NaN/inf")
        if take > supply + 1e-9:
            errs.append(f"take {take:.6g} > supply {supply:.6g}")
        if abs(take - ded) > 1e-9 * max(1.0, abs(take) + abs(ded)):
            errs.append(f"take {take:.9g} != ded {ded:.9g}")
        if float(e._corpse_energy.min()) < -1e-12:
            errs.append("negative")
        if errs:
            bad.append((t, n, supply, g, room, errs))
    return bad, cov


def sec4_units(e):
    P = len(e._id)
    cell = 100
    e._flat[:P] = 777
    e._flat[0] = cell
    e._corpse_energy[:] = 0.0
    e._corpse_energy[cell] = 100.0
    e._stomach[:P] = 0.0
    e._genes[0, 16] = 0.99
    before = float(e._corpse_energy[cell])
    e._step_scavenging(1, e._stomach, np.full(P, 1e9), e._genes)
    mass = float(e._stomach[0])
    eff = float(e.config.organisms.eat_efficiency)
    # 腐烂归还
    e2 = mk()
    e2._corpse_energy[:] = 0.0
    e2._corpse_energy[cell] = 200.0
    e2._corpse_age[:] = 0
    e2._corpse_age[cell] = int(e2.config.corpse_wound.corpse_decay_ticks)
    g0 = float(e2.resources._grid[cell])
    cap = float(e2.resources._capacity[cell])
    e2._step_corpse_decay()
    g1 = float(e2.resources._grid[cell])
    frac = float(e2.config.corpse_wound.corpse_to_plant_frac)
    return {"pool_before": before, "mass_to_stomach": mass,
            "energy_yield": mass * eff, "multiplier": eff,
            "plant_capacity": cap, "expect_ret": 200.0 * frac,
            "actual_ret": g1 - g0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="随机重复次数")
    ap.add_argument("--cap", type=int, default=200)
    args = ap.parse_args()

    print("engine:", se.__file__)
    print("python:", sys.version.split()[0],
          "| eat_efficiency:", SimConfig().organisms.eat_efficiency,
          "| cap default:", SimConfig().corpse_wound.corpse_cap_per_cell)
    print()
    print("=== 1/2. scavenging: 1 cell, N conspecifics, measured against supply ===")
    print(f"  {'n':>3} {'g16':>5} {'supply':>8} {'take':>10} {'cell_after':>12} {'overdraw':>9}")
    for n_ind, g16, sup in ((12, 0.99, 1.0), (12, 0.50, 1.0), (1, 0.99, 1.0), (12, 0.99, 5.0)):
        take, after, ded, over = sec1_scavenge(mk(cap=args.cap), n_ind, g16, sup)
        print(f"  {n_ind:>3} {g16:>5.2f} {sup:>8.3f} {take:>10.4f} {after:>12.4f} "
              f"{str(over):>9}  (take-ded={take - ded:+.2e})")
    print("  overdraw=True ⇒ 取量超过格上存量（旧版 bug 的签名）")

    print()
    print("=== 2/2. deposit: N dead in the SAME cell, 9.0 each ===")
    for nd in (1, 2, 4):
        got = sec2_deposit(mk(cap=args.cap), nd, 9.0)
        print(f"  n_dead={nd} expect={nd * 9.0:>6.1f} got={got:>7.1f} "
              f"lost={(1 - got / (nd * 9.0)) * 100:>5.1f}%")

    print()
    print(f"=== 3. randomized repeat (n={args.n}) ===")
    bad, cov = sec3_random(mk(cap=args.cap), args.n)
    print("  violations =", len(bad), "| coverage:", cov)
    for b in bad[:5]:
        print("   BAD:", b)

    print()
    print("=== 4. unit audit (corpse pool is documented as ENERGY units) ===")
    u = sec4_units(mk(cap=args.cap))
    print(f"  scavenge: pool {u['pool_before']:.4f} -> mass into stomach "
          f"{u['mass_to_stomach']:.4f} -> energy {u['energy_yield']:.4f} "
          f"(x{u['multiplier']:.1f})")
    print(f"  decay->plant: expect {u['expect_ret']:.1f}, actual {u['actual_ret']:.4f} "
          f"(capacity {u['plant_capacity']:.4f}) => "
          f"clamped {(1 - u['actual_ret'] / u['expect_ret']) * 100:.1f}%")


if __name__ == "__main__":
    main()
