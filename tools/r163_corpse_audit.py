"""R163 独立复核：尸体—食腐通道的裁决性探针（老工线，2026-09-22）。

**用途**：把 R161/R163 的四项复核变成可复现证据（纯 ASCII 输出、零 RNG、
直调引擎内部函数、不改任何仓库文件）。

四项复核：
  1. 两个缺陷（P1-2 食腐超发 / P1-3 投尸同格丢失）在**修复前版本**上复现
  2. 修复后两场景**逐格守恒** + 随机重复
  3. 全量测试（由调用方在外层跑，见下）
  4. `alloc = min(demand, supply)` 分配逻辑的边界（demand=0 / supply=0 / 浮点）

**单位审计（§4/§5）与版本无关的判决口径**：
  - 守恒恒等式 = **入胃质量 × `eat_efficiency` ≡ 池减（能量）**
      13.4 起（R165 0-1）：成立；13.3 及更早：**不成立**（池被当质量用 ⇒ 少算 eff 倍）
  - 池收支闭账：`total ≡ 投放 − 溢出 − 食腐 − 腐烂`（残差应恒 0；13.4 起可读）
  - 腐烂归还仍被 `capacity − grid` 截掉约 99.5%（**13.4 波 2 才处理**，此处只登记）
同一把尺子可同时判**三个修订**（13.3- / 13.3 / 13.4），故本工具兼作跨版本判决件。

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


def _eff(e) -> float:
    return float(e.config.organisms.eat_efficiency)


def _zero_scav(e, P: int) -> None:
    """版本容错：旧修订无 `_stomach_scav`（R165 0-2 才加）。"""
    ss = getattr(e, "_stomach_scav", None)
    if ss is not None:
        ss[:P] = 0.0


def sec1_scavenge(e, n_ind, g16, supply, cell=100):
    """单位无关的判决性检查（同一把尺子判三个修订）：

      守恒恒等式（**能量口径**）：`入胃质量 × eat_efficiency ≡ 池减`
        - 13.4 起：成立（入胃处按 ÷eff 折算）
        - 13.3 及更早：**不成立**（池里的数字被当质量用 ⇒ 池减 == 入胃质量，少算 eff 倍）
    """
    P = len(e._id)
    idx = np.arange(n_ind)
    e._flat[:P] = 777
    e._flat[idx] = cell
    e._genes[:P, 16] = g16
    e._stomach[:P] = 0.0
    _zero_scav(e, P)
    e._corpse_energy[:] = 0.0
    e._corpse_energy[cell] = supply
    b_corpse = float(e._corpse_energy.sum())
    b_stom = float(e._stomach[:P].sum())
    e._step_scavenging(P, e._stomach, np.full(P, 1e9), e._genes)
    take = float(e._stomach[:P].sum()) - b_stom
    drawn = b_corpse - float(e._corpse_energy.sum())
    eff = _eff(e)
    return {
        "take": round(take, 6),
        "drawn": round(drawn, 6),
        "ratio": round(take / drawn, 6) if drawn > 0 else None,
        "cell_after": round(float(e._corpse_energy[cell]), 6),
        "overdraw": drawn > supply + 1e-9,           # 池被取超（旧版签名）
        "negative": float(e._corpse_energy.min()) < -1e-12,
        "energy_ok": abs(take * eff - drawn) < 1e-9,  # 13.4 恒等式
    }


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
    eff = _eff(e)
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
        _zero_scav(e, P)
        e._corpse_energy[:] = 0.0
        e._corpse_energy[cell] = supply
        b0, s0 = float(e._corpse_energy.sum()), float(e._stomach[:P].sum())
        e._step_scavenging(P, e._stomach, np.full(P, max(room, 0.0)), e._genes)
        take = float(e._stomach[:P].sum()) - s0
        drawn = b0 - float(e._corpse_energy.sum())
        cov["supply0"] += supply <= 0.0
        cov["demand0"] += g <= 1e-9
        cov["supply<demand"] += (0.0 < supply < 1e6) and room > 100.0
        errs = []
        if not np.isfinite(take):
            errs.append("NaN/inf")
        if drawn > supply + 1e-9:
            errs.append(f"drawn {drawn:.6g} > supply {supply:.6g}")
        if abs(take * eff - drawn) > 1e-9 * max(1.0, abs(drawn)):
            errs.append(f"take*eff {take * eff:.9g} != drawn {drawn:.9g}")
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
    _zero_scav(e, P)
    e._genes[0, 16] = 0.99
    before = float(e._corpse_energy[cell])
    e._step_scavenging(1, e._stomach[:P], np.full(P, 1e9), e._genes[:P])
    mass = float(e._stomach[0])
    drawn = before - float(e._corpse_energy[cell])
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
            "pool_drawn": drawn, "energy_yield": mass * eff, "multiplier": eff,
            "ratio_ok": abs(mass * eff - drawn) < 1e-9,
            "plant_capacity": cap, "expect_ret": 200.0 * frac,
            "actual_ret": g1 - g0}


def sec5_pool_balance(e, ticks=300):
    """R165 0-1：尸体池收支**闭账**（残差应恒 0）。"""
    for _ in range(ticks):
        if e.extinct:
            break
        e.step()
    cp = e.corpse_probe()
    # 版本容错：13.3 及更早无这四个流计数器（.get ⇒ None，不报 KeyError）
    return {k: cp.get(k) for k in (
        "corpse_pool_unit", "corpse_deposited_e", "corpse_overflow_e",
        "corpse_scav_e", "corpse_decayed_e", "corpse_total",
        "corpse_balance_residual")}


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
    print("  invariants: drawn<=supply | cell>=0 | take*eat_eff==drawn (energy identity)")
    print(f"  {'n':>3} {'g16':>5} {'supply':>8} {'take(mass)':>11} {'drawn(E)':>10} "
          f"{'take/drawn':>11} {'cell_after':>11} {'overdrw':>7} {'neg':>5} {'E-ident':>7}")
    for n_ind, g16, sup in ((12, 0.99, 1.0), (12, 0.50, 1.0), (1, 0.99, 1.0), (12, 0.99, 5.0)):
        r = sec1_scavenge(mk(cap=args.cap), n_ind, g16, sup)
        print(f"  {n_ind:>3} {g16:>5.2f} {sup:>8.3f} {r['take']:>11.4f} {r['drawn']:>10.4f} "
              f"{str(r['ratio']):>11} {r['cell_after']:>11.4f} {str(r['overdraw']):>7} "
              f"{str(r['negative']):>5} {str(r['energy_ok']):>7}")
    print("  => 13.4+: overdrw/neg False, E-ident True, take/drawn == 1/eat_efficiency")
    print("  => 13.3-: overdrw/neg True (BUG), E-ident False, take/drawn == 1.0")

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
    print("=== 4. unit audit (pool = ENERGY units; single conversion at ingestion) ===")
    u = sec4_units(mk(cap=args.cap))
    print(f"  scavenge: pool {u['pool_before']:.4f} -> mass into stomach "
          f"{u['mass_to_stomach']:.4f}")
    print(f"            pool drawn = {u['pool_drawn']:.6f} | mass x eff = "
          f"{u['energy_yield']:.6f} | equal: {u['ratio_ok']}")
    print(f"            => 1 unit of corpse yields 1 unit of energy "
          f"(divide by eat_efficiency={u['multiplier']:.1f} at ingestion)")
    print("            [pre-13.4 behaviour: x3 inflation -- pool drawn == mass]")
    print(f"  decay->plant: expect {u['expect_ret']:.1f}, actual {u['actual_ret']:.4f} "
          f"(capacity {u['plant_capacity']:.4f}) => "
          f"clamped {(1 - u['actual_ret'] / u['expect_ret']) * 100:.1f}%")
    print("            [KNOWN leftover, wave-2 scope: energy->mass still mismatched]")

    print()
    print("=== 5. pool balance closure (R165 0-1; residual must be 0) ===")
    b = sec5_pool_balance(mk(), ticks=300)
    for k, v in b.items():
        print(f"  {k:<26} = {v}")
    print("  => residual 0 means the pool has exactly 4 flows "
          "(deposit/overflow/scav/decay) and all are booked")


if __name__ == "__main__":
    main()
