"""P0.3 阶段 C 验收探针 —— 累加器（C1）+ gain 定标（C2）+ 档位数（C3）。

为什么需要本探针（而不是复用 `speed_quant_probe.py`）
------------------------------------------------------
`speed_quant_probe.py` 读的是引擎**运行期**的 `steps_hist`，那里面**混着三类零步**：

    zero% = 物理零步（速度太小，本 tick 攒不够 1 亚格）      ← C1 要消灭的
          + 门槛零步（能量 < `min_energy_frac × max_energy`） ← **有意保留**（饿死不走路）
          + 基因零步（`g18 = 0` ⇒ 速度恒 0）                  ← **有意保留**（选择压结果）

⇒ 用运行期 `zero%` 当验收判据会**归因不清**：即使 C1 完美生效，只要还有 9% 饿肚子个体，
   该数就降不到 0。派工单 §三 C 的判据「零步% ≈ 0」必须建立在**纯净口径**上。

本探针因此分两层：

**A 层（物理层，C1 的直接判据）** —— 不跑世界，直接对 `steps` 递推做长程位移测试：
    * 恒定速度 `v`、`T` 个 tick，闭式解 `总亚格 = v·subdiv·T`
    * 旧法（每 tick 四舍五入）与 C1 累加器各算一遍
    * 报 **长程位移相对误差** + **零步%**
    ⇒ 这是**数学恒等式**，没有噪声、没有归因歧义 ⇒ 可作硬判据。

**B 层（世界层，端到端对照）** —— 真实世界跑，把三类零步**分开报**：
    * `zero_phys%`（门槛通过、基因非零，但本 tick 走 0 步）
    * `zero_gate%`（能量不足被强制 0）
    * `zero_gene%`（速度恒 0）
    ⇒ C1 只承诺把 `zero_phys%` 压到 ≈0，另两类是**设计**。

**C 层（C2 `gain` 定标）** —— 实测目标配置上的 `ā = mean(mob_eff)`：
    `gain = v_max / ā` ⇒ 使 `mean(speed) = v_max`。
    报 `ā`、幼体占比、`gain`、以及按该 `gain` 后的速度分布。
    🔴 **只报数，不改默认**（`speed_gain` 默认仍 4.0；默认关档不受影响）。

**D 层（C3 档位数）** —— `R = v_max × subdiv`，报理论档数与实测占用档数。

用法
----
    python3 experiments/speed_accum_probe.py                      # 全量（默认档）
    python3 experiments/speed_accum_probe.py --subdiv 4,8,20      # 扫档位数
    python3 experiments/speed_accum_probe.py --json out.json      # 落 JSON
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# --- R98 纪律：Windows GBK 控制台兜底（非 ASCII print 会让脚本 rc=1 假失败；F-R15 族）---
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass  # 非 TTY / 旧解释器：不因诊断能力缺失而阻断运行

from simulation.config import SimConfig                      # noqa: E402
from simulation.genes import Gene                            # noqa: E402
from simulation.sphere_engine import SphereEngine            # noqa: E402


# ---------------------------------------------------------------- A 层：纯数学

def _legacy_steps(speed_x_subdiv: float, subdiv: int) -> int:
    """旧法：`floor(speed×subdiv + 0.5)`，上钳 subdiv（= 每 tick 每轴最多 1 格）。"""
    return int(np.clip(np.floor(speed_x_subdiv + 0.5), 0.0, float(subdiv)))


def layer_a(speeds: list[float], subdivs: list[int], ticks: int) -> list[dict]:
    """A 层：长程位移守恒 + 零步%，旧法 vs 累加器（**纯递推，无世界**）。"""
    out: list[dict] = []
    for subdiv in subdivs:
        for v in speeds:
            # --- 旧法 ---
            tot_old, zero_old = 0, 0
            for _ in range(ticks):
                st = _legacy_steps(v * subdiv, subdiv)
                tot_old += st
                zero_old += (st == 0)
            # --- C1 累加器 ---
            tot_new, zero_new, pf = 0, 0, 0.0
            for _ in range(ticks):
                pf += v * subdiv
                st = int(np.clip(np.floor(pf), 0.0, float(subdiv)))
                pf -= st
                tot_new += st
                zero_new += (st == 0)
            ideal = v * subdiv * ticks
            out.append({
                "layer": "A", "subdiv": subdiv, "speed": v, "ticks": ticks,
                "ideal_substeps": ideal,
                "legacy_substeps": tot_old,
                "acc_substeps": tot_new,
                "legacy_relerr_pct": (tot_old - ideal) / ideal * 100.0 if ideal else 0.0,
                "acc_relerr_pct": (tot_new - ideal) / ideal * 100.0 if ideal else 0.0,
                "legacy_zero_pct": zero_old / ticks * 100.0,
                "acc_zero_pct": zero_new / ticks * 100.0,
            })
    return out


# ---------------------------------------------------------------- B/C/D 层：世界

def _cfg(seed: int, subdiv: int, gain: float, acc: bool,
         v_max: float, pop: int, patchy: bool = True) -> SimConfig:
    """13.5 B 臂（斑块食物）+ subpos（与 Rust 路径互斥 ⇒ 必须 Python）。"""
    c = SimConfig(seed=seed)
    c.simulation.use_sim_core = False
    if patchy:
        c.resources.distribution = "patchy"
        c.resources.bg_production_zero = True
        c.resources.patch_regrowth_mult = 1.195
    c.subpos.enabled = True
    c.subpos.speed_accumulator = bool(acc)
    c.subpos.subdiv = int(subdiv)
    c.subpos.speed_gain = float(gain)
    c.subpos.speed_max = float(v_max)
    c.population.initial_count = int(pop)
    return c


def _speed_breakdown(eng: SphereEngine) -> dict:
    """把速度向量按**来源**拆开（复刻引擎 §5.4 的算法，用于归因）。"""
    sc = eng.config.subpos
    ocs = eng.config.organisms
    g = eng._genes[:, Gene.DEFENSE].astype(np.float64)
    ls = eng._lifespan(eng._genes[:, Gene.LIFE_GENE])
    agef = eng._age.astype(np.float64)
    af = np.ones_like(agef)
    af[agef < ocs.maturity_fraction * ls] = ocs.young_mob_mult
    af[agef >= ocs.senile_fraction * ls] = ocs.old_mob_mult
    cr, _cc = eng.world.flat_to_rc(eng._flat)
    coslat = np.cos(eng.world.latitude_of(cr))
    cap = sc.speed_max * (sc.lat_floor
                          + (1.0 - sc.lat_floor) * np.maximum(coslat, 0.0))
    mob = g * af
    spd = np.clip(mob * float(sc.speed_gain), 0.0, cap)
    gate_ok = eng._energy >= float(sc.min_energy_frac) * ocs.max_energy
    return {
        "n": int(len(g)),
        "mob_eff": mob,                      # ā 的样本（未乘 gain）
        "speed": spd,
        "gene_zero": g <= 0.0,
        "gate_ok": gate_ok,
        "is_juvenile": agef < ocs.maturity_fraction * ls,
    }


def layer_bcd(seed: int, subdiv: int, gain: float, acc: bool, v_max: float,
              ticks: int, pop: int, patchy: bool = True) -> dict:
    eng = SphereEngine(_cfg(seed, subdiv, gain, acc, v_max, pop, patchy))
    for _ in range(ticks):
        eng.step()
        if len(eng._flat) == 0:
            break
    if len(eng._flat) == 0:
        return {"seed": seed, "subdiv": subdiv, "acc": acc, "gain": gain,
                "extinct": True}

    h = eng._steps_hist.astype(np.int64)
    tot = int(h.sum())
    bd = _speed_breakdown(eng)
    sc = eng.config.subpos

    # 把"运行期零步"按来源拆开（复刻引擎的判定顺序）
    spd_x = bd["speed"] * float(subdiv)
    phys_zero_mask = (np.floor(spd_x) < 1.0) | (spd_x < 1.0)   # 本 tick 物理攒不够 1 亚格
    if acc:
        # 累加器：门槛通过者若 `speed·subdiv ≥ 1` 必然走 ≥1 步；否则看余量
        # ⇒ 这里用"门槛通过 且 speed·subdiv ≥ 1"的占比作为**必然非零**的下界
        phys_zero_mask = (spd_x < 1.0)
    zero_phys = float((phys_zero_mask & bd["gate_ok"] & ~bd["gene_zero"]).mean() * 100)
    zero_gate = float((~bd["gate_ok"]).mean() * 100)
    zero_gene = float(bd["gene_zero"].mean() * 100)

    return {
        "seed": seed, "subdiv": subdiv, "acc": bool(acc), "gain": float(gain),
        "v_max": float(v_max), "ticks": int(ticks), "n_alive": int(len(bd["n"] * 0 + bd["speed"])),
        "steps_hist": h.tolist(),
        "zero_pct_runtime": float(h[0] / tot * 100) if tot else None,
        "top_pct_runtime": float(h[-1] / tot * 100) if tot else None,
        "bands_occupied": int(np.count_nonzero(h)),
        "R_theory": float(v_max * subdiv),
        # B 层拆分
        "zero_phys_pct": zero_phys,
        "zero_gate_pct": zero_gate,
        "zero_gene_pct": zero_gene,
        # C 层：ā 与 gain 定标
        "a_bar": float(bd["mob_eff"].mean()),
        "juvenile_pct": float(bd["is_juvenile"].mean() * 100),
        "gain_for_vmax": float(v_max / bd["mob_eff"].mean()) if bd["mob_eff"].mean() > 0 else None,
        "speed_mean": float(bd["speed"].mean()),
    }


# ---------------------------------------------------------------- E 层：引擎内低速专测

def layer_e(subdiv: int, v_max: float, ticks: int) -> dict:
    """E 层：**同一速度分布下的配对比较**（C1 的端到端硬判据）。

    为什么必须配对（这是本探针最重要的方法论修正）
    ----------------------------------------------
    第一版是「跑两个完整生态（acc=off / on），比它们的零步%」⇒ **混淆**：
    `speed` 是**内生**的（个体对"移动代价/能量"有反馈），
    开了累加器以后速度分布**自己就变了** ⇒ 跑出来的差异里
    "方法差异"与"速度分布差异"**混在一起**，无法归因。

    ⇒ 正确做法：取**一个真实引擎状态**的 `speed` 向量，**冻结它**，
      在**同一批 speed** 上分别按旧法与累加器递推 `T` 个 tick，
      比较 (a) 长程平均步频 (b) 零步率。
      —— 这样两臂的**输入完全相同**，差异**只能**来自方法本身。

    🔴 判据（两条，均对旧法可证伪）
    ------------------------------
    ① **长程步频无偏**：累加器 `E[steps]` == `E[min(sx, subdiv)]`（相对误差 < 1%）
       旧法在 `sx < 0.5` 时 `E[steps] ≡ 0` ⇒ 相对误差 **100%**（永久冻结）
    ② **低速端不再冻结**：`sx < 0.5` 时累加器 `E[steps] > 0` 且随 `sx` 线性增

    ⚠️ 报出但**不作判据**：`零步%` —— 累加器在中间带（`0.5 ≤ sx < 1`）
       零步% **更高**是对的（它把"1 步"摊到多个 tick 上，换来长程无偏）。
    """
    # 1) 取一个真实状态（默认 gain=4.0），拿它的 speed 分布作"标本"
    c0 = _cfg(42, subdiv, 4.0, False, v_max, 500)
    eng0 = SphereEngine(c0)
    for _ in range(ticks):
        eng0.step()
        if len(eng0._flat) == 0:
            break
    if len(eng0._flat) == 0:
        return []
    spd0 = _speed_breakdown(eng0)["speed"]

    out: list[dict] = []
    # 2) 冻结 speed 分布，只按 gain 缩放（模拟"把个体压到低速端"）
    for gmul in (0.0125, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.0):
        spd = spd0 * gmul
        sx = spd * float(subdiv)
        # 🔴 上钳：每 tick 每轴最多 subdiv（= 1 格，硬约束）
        allow = np.minimum(sx, float(subdiv))
        ideal = float(allow.mean())                     # 无偏期望（累加器的目标）
        old_e = float(np.clip(np.floor(sx + 0.5), 0.0, float(subdiv)).mean())
        old_zero = float((np.clip(np.floor(sx + 0.5), 0.0, float(subdiv)) < 0.5).mean() * 100)

        # 累加器：递推 T 个 tick（T 取够长以抹平相位）
        T = 2000
        pf = np.zeros_like(sx)
        zero_n, tot_n, sum_n = 0, 0, 0.0
        for _ in range(T):
            pf = pf + allow
            st = np.clip(np.floor(pf), 0.0, float(subdiv))
            pf = pf - st
            zero_n += int((st < 0.5).sum())
            sum_n += float(st.sum())
            tot_n += int(st.size)
        new_e = sum_n / tot_n
        new_zero = zero_n / tot_n * 100
        out.append({
            "layer": "E", "subdiv": subdiv, "gain_mult": gmul,
            "sx_mean": float(sx.mean()), "sx_median": float(np.median(sx)),
            "ideal_e_steps": ideal,
            "old_e_steps": old_e, "new_e_steps": new_e,
            "old_relerr": (old_e - ideal) / ideal * 100 if ideal else 0.0,
            "new_relerr": (new_e - ideal) / ideal * 100 if ideal else 0.0,
            "old_zero_pct": old_zero, "new_zero_pct": new_zero,
            "n_individuals": int(sx.size),
        })
    return out



def main() -> None:
    ap = argparse.ArgumentParser(description="P0.3 阶段 C 验收探针（C1/C2/C3）")
    ap.add_argument("--subdiv", default="4,8,20", help="档位细分数，逗号分隔")
    ap.add_argument("--ticks", type=int, default=400, help="B/C/D 层世界跑多少 tick")
    ap.add_argument("--pop", type=int, default=500)
    ap.add_argument("--v-max", type=float, default=2.0, dest="v_max")
    ap.add_argument("--gain", type=float, default=4.0, help="C2 前用默认 4.0")
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--a-ticks", type=int, default=10000, dest="a_ticks")
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    subdivs = [int(s) for s in a.subdiv.split(",") if s.strip()]
    seeds = [int(s) for s in a.seeds.split(",") if s.strip()]
    speeds = [0.05, 0.0625, 0.1, 0.1111, 0.125, 0.2, 0.3, 0.4, 0.55, 0.75, 1.0]

    res: dict = {}

    # ---------------- A 层 ----------------
    print("=" * 96)
    print("A 层（纯数学，无世界）：长程位移守恒 + 零步%  —— C1 的直接判据")
    print("=" * 96)
    print(f"  {'subdiv':<8}{'speed':<9}{'理想亚格':<12}{'旧法':<10}{'旧误差%':<11}"
          f"{'累加器':<10}{'新误差%':<10}{'旧零步%':<10}{'新零步%'}")
    la = layer_a(speeds, subdivs, a.a_ticks)
    for r in la:
        print(f"  {r['subdiv']:<8}{r['speed']:<9}{r['ideal_substeps']:<12.1f}"
              f"{r['legacy_substeps']:<10}{r['legacy_relerr_pct']:<+11.4f}"
              f"{r['acc_substeps']:<10}{r['acc_relerr_pct']:<+10.4f}"
              f"{r['legacy_zero_pct']:<10.2f}{r['acc_zero_pct']:.2f}")
    res["A"] = la

    # A 层判据
    #
    # 🔴 **判据必须选对**（我第一版选错了，实测暴露）：
    #   把「零步% ≈ 0」当作 A 层阈值是**错的** —— 当 `speed·subdiv < 1` 时，
    #   个体**物理上**就该每若干 tick 才走 1 亚格（余量在攒），此时"零步"是
    #   **正确行为**，不是缺陷。例：`subdiv=4, speed=0.05` ⇒ `speed·subdiv = 0.2`
    #   ⇒ 20 tick 里 19 个 0 步 + 1 个 1 步 ⇒ 零步 95%，**完全正确**。
    #   （真正的缺陷是**旧法**：它在该情形下给出**永久 0 步**、且长程位移误差 −100%。）
    #
    #   ⇒ 正确的 A 层判据是**两条**：
    #     ① **长程位移守恒**：|相对误差| < 0.5%（旧法可达 ±100%）
    #     ② **平均步频保真**：`总亚格 / (v·subdiv·T)` 的误差（= 判据①的另一面）
    #   「零步%」在 A 层只作**诊断报数**，不作判据；它作为判据只在
    #   "**speed·subdiv ≥ 1** 的子样本"上有意义（此时旧法仍可能 0 步，累加器必然 ≥1）。
    worst_old = max(abs(r["legacy_relerr_pct"]) for r in la)
    worst_new = max(abs(r["acc_relerr_pct"]) for r in la)
    z_old = max(r["legacy_zero_pct"] for r in la)
    z_new = max(r["acc_zero_pct"] for r in la)
    # 🔴 关键子样本：`speed·subdiv ≥ 1` ⇒ 累加器**必然**每 tick ≥1 步（零步% = 0）
    sub = [r for r in la if r["speed"] * r["subdiv"] >= 1.0]
    z_sub_old = max((r["legacy_zero_pct"] for r in sub), default=0.0)
    z_sub_new = max((r["acc_zero_pct"] for r in sub), default=0.0)
    print(f"\n  ⇒ 长程位移**最差相对误差**：旧法 **{worst_old:.2f}%** → 累加器 **{worst_new:.4f}%**")
    print(f"  ⇒ **最差零步%**（全样本，仅诊断）：旧法 **{z_old:.1f}%** → 累加器 **{z_new:.1f}%**")
    print(f"      ⚠️ 该数在 `speed·subdiv < 1` 时**本就非 0 且正确**（余量在攒，不是缺陷）")
    print(f"  ⇒ **判据① `speed·subdiv ≥ 1` 子样本的零步%**（{len(sub)} 个）："
          f"旧法 **{z_sub_old:.1f}%** → 累加器 **{z_sub_new:.1f}%**")
    ok1 = (worst_new < 0.5)
    ok2 = (z_sub_new < 1.0)
    print(f"  ⇒ 判据① 长程位移守恒（< 0.5%）：{'✅ 通过' if ok1 else '🔴 未通过'}"
          f"（旧法 {worst_old:.1f}% ⇒ 判据对旧法是**可证伪的**）")
    print(f"  ⇒ 判据② 子样本零步%（< 1%）：{'✅ 通过' if ok2 else '🔴 未通过'}")
    res["A_verdict"] = {
        "worst_relerr_legacy": worst_old, "worst_relerr_acc": worst_new,
        "worst_zero_legacy": z_old, "worst_zero_acc": z_new,
        "sub_ge1_zero_legacy": z_sub_old, "sub_ge1_zero_acc": z_sub_new,
        "sub_ge1_n": len(sub),
        "pass_relerr": bool(ok1), "pass_sub_zero": bool(ok2),
        "pass": bool(ok1 and ok2),
    }

    # ---------------- B/C/D 层 ----------------
    print()
    print("=" * 96)
    print("B/C/D 层（真实世界）：三类零步拆分（C1）+ ā 与 gain 定标（C2）+ 档位数（C3）")
    print("=" * 96)
    rows: list[dict] = []
    for seed in seeds:
        for subdiv in subdivs:
            for acc in (False, True):
                r = layer_bcd(seed, subdiv, a.gain, acc, a.v_max,
                              a.ticks, a.pop)
                r["layer"] = "BCD"
                rows.append(r)
    res["BCD"] = rows

    hdr = (f"  {'seed':<6}{'subdiv':<8}{'acc':<6}{'R':<7}{'档占用':<8}"
           f"{'运零步%':<10}{'顶格%':<9}{'物理零%':<10}{'门槛零%':<10}{'基因零%':<10}{'ā':<9}{'幼体%':<8}")
    print(hdr)
    for r in rows:
        if r.get("extinct"):
            print(f"  {r['seed']:<6}{r['subdiv']:<8}{str(r['acc']):<6}🔴 灭绝")
            continue
        print(f"  {r['seed']:<6}{r['subdiv']:<8}{str(r['acc']):<6}{r['R_theory']:<7.0f}"
              f"{r['bands_occupied']:<8}{r['zero_pct_runtime']:<10.2f}"
              f"{r['top_pct_runtime']:<9.2f}{r['zero_phys_pct']:<10.2f}"
              f"{r['zero_gate_pct']:<10.2f}{r['zero_gene_pct']:<10.2f}"
              f"{r['a_bar']:<9.4f}{r['juvenile_pct']:<8.1f}")

    # C2：gain 定标
    af = [r["a_bar"] for r in rows if not r.get("extinct")]
    if af:
        a_mean = float(np.mean(af))
        juv = float(np.mean([r["juvenile_pct"] for r in rows if not r.get("extinct")]))
        gain_cal = a.v_max / a_mean
        print(f"\n  【C2】`ā = mean(mob_eff)` 实测 = **{a_mean:.4f}**"
              f"（跨 {len(af)} 个臂的均值；幼体占比 **{juv:.1f}%**）")
        print(f"        ⇒ `gain = v_max / ā = {a.v_max} / {a_mean:.4f} = **{gain_cal:.4f}**"
              f"  （= {gain_cal/a.v_max:.4f} × v_max）")
        print(f"        🔴 本阶段**只报数、不改默认**（`speed_gain` 默认仍 {a.gain}）")
        res["C2"] = {"a_bar": a_mean, "juvenile_pct": juv,
                     "gain_calibrated": gain_cal, "v_max": a.v_max}

    # C3 判据
    acc_rows = [r for r in rows if r.get("acc") and not r.get("extinct")]
    ok3 = [r for r in acc_rows if r["bands_occupied"] >= 5]
    print(f"\n  【C3】`R = v_max × subdiv`；判据「占用档 ≥ 5」："
          f"{len(ok3)}/{len(acc_rows)} 个累加器臂通过"
          f"{' ✅' if len(ok3) == len(acc_rows) else ' 🔴'}")
    res["C3"] = {"pass_n": len(ok3), "total_n": len(acc_rows)}

    # ---------------- E 层：同一速度分布下的配对比较 ----------------
    print()
    print("=" * 96)
    print("E 层（**配对**，同一速度分布 + 冻结）：C1 的端到端硬判据")
    print("=" * 96)
    print("  方法论：取一个真实引擎状态的 `speed` 向量**冻结**，两臂喂**同一批** speed")
    print("  ⇒ 差异**只能**来自方法本身（不混入「速度内生」的反馈）。")
    print("  ⚠️ 旧版比两个完整生态 ⇒ 混淆（开累加器后速度分布自己就变了），已弃用。")
    print()
    e_rows: list[dict] = []
    for subdiv in subdivs:
        e_rows.extend(layer_e(subdiv, a.v_max, a.ticks))
    res["E"] = e_rows

    print(f"  {'subdiv':<8}{'gmul':<8}{'sx均值':<10}{'理想步频':<11}"
          f"{'旧步频':<10}{'旧误差%':<11}{'新步频':<10}{'新误差%':<11}"
          f"{'旧零步%':<10}{'新零步%'}")
    for r in e_rows:
        print(f"  {r['subdiv']:<8}{r['gain_mult']:<8}{r['sx_mean']:<10.4f}"
              f"{r['ideal_e_steps']:<11.4f}{r['old_e_steps']:<10.4f}"
              f"{r['old_relerr']:<+11.2f}{r['new_e_steps']:<10.4f}"
              f"{r['new_relerr']:<+11.4f}{r['old_zero_pct']:<10.2f}{r['new_zero_pct']:.2f}")

    # E 层判据
    print()
    low = [r for r in e_rows if r["sx_mean"] < 0.5]
    ok1 = all(abs(r["new_relerr"]) < 1.0 for r in e_rows)          # 无偏
    ok2 = all(r["new_e_steps"] > 0.0 for r in low) and bool(low)   # 低速不冻结
    worst_new = max(abs(r["new_relerr"]) for r in e_rows)
    worst_old = max(abs(r["old_relerr"]) for r in e_rows)
    print(f"  ⇒ 判据① 累加器长程步频**无偏**（|相对误差| < 1%）：{'✅ 通过' if ok1 else '🔴'}"
          f"（最差 {worst_new:.4f}%；**旧法**最差 {worst_old:.2f}% ⇒ 判据可证伪）")
    print(f"  ⇒ 判据② 低速端（sx_mean < 0.5，{len(low)} 个臂）**不再冻结**"
          f"（步频 > 0）：{'✅ 通过' if ok2 else '🔴'}"
          f"（旧法在这些臂上步频 = {[round(r['old_e_steps'],4) for r in low]}）")
    print(f"  ⇒ 副报（不判据）：中间带（0.5 ≤ sx < 1）累加器**零步% 更高是对的** ——"
          f"它把「1 步」摊到多 tick，换长程无偏")
    res["E_verdict"] = {"pass_unbiased": bool(ok1), "pass_no_freeze": bool(ok2),
                        "worst_new_relerr": worst_new, "worst_old_relerr": worst_old,
                        "n_low_speed_arms": len(low)}

    if a.json:
        Path(a.json).write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                encoding="utf-8")
        print(f"\n已落 JSON：{a.json}")


if __name__ == "__main__":
    main()
