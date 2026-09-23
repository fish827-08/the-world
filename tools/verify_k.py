"""tools/verify_k.py —— 素食者 K 验收工具（13.5 唯一硬判据）

用途
----
在没有捕食、没有光合的"纯素食世界"里测**承载力 K**，并检验它是否等于解析预测：

    K = Σ_斑块 名义再生(质量/tick) ÷ 人均需求(质量/tick)
    人均需求 = 人均支出(能量/tick) ÷ ( eat_efficiency × 吸收率 )

为什么需要它（人话）
------------------
"食物够不够"不该靠感觉判断：只要 **K ≥ 软顶**，个体永远吃不饿 ⇒ **位置（斑块）不值钱** ⇒
一切依赖"找食物/传位置信息"的机制实验都失去前提（R183/R184/R185）。
所以本工具是 13.5 的门：**K 必须落进 [1200, 1900]（低于软顶 1944）**。

设计要点
--------
* **只改一个变量**：食物供给（`--regrowth`）；其余全部固定（关捕食、关光合、关软顶、抬高硬顶）。
* **自适应纪元**：吸收率/胃容量等新字段**用 `getattr` 读**，缺失时按"现状值"处理 ⇒
  本工具在 13.4（assim=1、背景有产能）与 13.5（assim=0.4、背景归零）**都能跑**，便于前后对比。
* **两个 K**：`K_prior` 用固定人均支出（0.762，R185 实测）做**事前预测**；
  `K_meas` 用本次跑出来的实际人均支出算 ⇒ 二者差距大说明"支出假设"失真（食物紧时移动更多）。
* 无副作用：不写快照、不动配置默认值。

用法
----
    python.exe tools/verify_k.py                      # 默认 regrowth=0.5, 4000 tick
    python.exe tools/verify_k.py --regrowth 0.125 --ticks 4000
    python.exe tools/verify_k.py --gate               # 按 13.5 判据返回退出码（0=过）
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulation.config import InfoStructureConfig, SimConfig  # noqa: E402
from simulation.sphere_engine import SphereEngine  # noqa: E402

PRIOR_NEED_E = 0.762     # 人均支出（能量/tick）—— R185 在"食物丰裕"世界实测
GATE_LO, GATE_HI = 1200.0, 1900.0


def _get(obj, name, default):
    return getattr(obj, name, default)


def build(regrowth: float, max_count: int, seed: int) -> SimConfig:
    c = SimConfig(seed=seed)
    c.simulation.use_sim_core = False           # 新机制强制 Python（§14.7）
    c.population.max_count = max_count          # 抬高硬顶 ⇒ 让食物（而非硬顶）决定 K
    c.population.soft_cap_target = 0.0          # 关软顶 ⇒ 关掉"出生率节流"这个人为限制
    c.resources.distribution = "patchy"
    c.resources.regrowth_rate = regrowth
    c.predation.enabled = False                 # 只有素食者
    c.info_structure = InfoStructureConfig(enabled=True, learning_rate=0.05,
                                           memory_gradient="none")
    for holder in (c.organisms, c.population):
        if hasattr(holder, "energy_cap_enabled"):
            holder.energy_cap_enabled = True
    if hasattr(c.organisms, "photo_max"):
        c.organisms.photo_max = 0.0             # 关光合（食物无关的能量来源）
    for f in ("corpse_enabled", "wound_enabled", "contest_enabled"):
        if hasattr(c.corpse_wound, f):
            setattr(c.corpse_wound, f, False)
    if hasattr(c, "resource_dynamics"):
        c.resource_dynamics.enabled = False     # 本工具只测"基础食物供给 vs 需求"
    return c


def main() -> int:
    ap = argparse.ArgumentParser(description="素食者 K 验收工具（13.5 硬判据）")
    ap.add_argument("--regrowth", type=float, default=0.5, help="基准再生率（唯一被改的变量）")
    ap.add_argument("--ticks", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-count", dest="max_count", type=int, default=12000)
    ap.add_argument("--report-every", dest="report_every", type=int, default=500)
    ap.add_argument("--gate", action="store_true", help="按 13.5 判据返回退出码")
    ap.add_argument("--json-out", dest="json_out", default="")
    a = ap.parse_args()

    e = SphereEngine(build(a.regrowth, a.max_count, a.seed))

    # ---- 解析预测（事前）----
    eff = float(_get(e.config.organisms, "eat_efficiency", 3.0))
    assim = float(_get(e.config.organisms, "assim_herb", 1.0))       # 13.5 前不存在 ⇒ 1.0
    need_e_prior = PRIOR_NEED_E
    need_m_prior = need_e_prior / max(1e-9, eff * assim)
    g_all = float(e.resources._regrowth_amount(0).sum())
    k_prior = g_all / need_m_prior

    print("=" * 78)
    print("素食者 K 验收 ｜ seed=%d regrowth=%.3f ticks=%d max_count=%d" % (a.seed, a.regrowth, a.ticks, a.max_count))
    print("  eat_efficiency=%.2f ｜ assim_herb=%.2f（缺失按 1.0 ⇒ 13.4 行为）" % (eff, assim))
    print("  Σ名义再生 = %.1f 质量/tick ｜ 人均需求(事前) = %.4f 质量/tick" % (g_all, need_m_prior))
    print("  ⇒ K_prior = %.0f" % k_prior)
    print("-" * 78)

    t0 = time.time()
    prev_n, prev_d = len(e._flat), 0
    traj = []
    led_prev = {"cost_meta_sum": 0.0, "cost_move_sum": 0.0, "cost_attack_sum": 0.0}

    def _exp_sum(led):
        return sum(float(led.get(k) or 0.0) for k in
                   ("cost_meta_sum", "cost_move_sum", "cost_attack_sum"))

    exp_prev = _exp_sum((e.energy_ledger() or {}).get("global") or {})
    need_m_win = float("nan")
    for k in range(a.ticks // a.report_every):
        for _ in range(a.report_every):
            if e.extinct:
                break
            e.step()
        n = len(e._flat)
        dc = {str(x).split(".")[-1]: y for x, y in e.death_cause_totals().items()}
        dtot = sum(dc.values())
        births = (n - prev_n) + (dtot - prev_d)
        prev_n, prev_d = n, dtot
        me = float(e._energy[:n].mean()) if n else 0.0
        # 末段窗口的人均支出（比全程平均更接近平衡态估计）
        exp_now = _exp_sum((e.energy_ledger() or {}).get("global") or {})
        need_m_win = ((exp_now - exp_prev) / a.report_every / max(1, n)) / max(1e-9, eff * assim)
        exp_prev = exp_now
        traj.append({"tick": a.report_every * (k + 1), "N": n, "births": births,
                     "starved_cum": dc.get("STARVATION", 0), "old_age_cum": dc.get("OLD_AGE", 0),
                     "mean_energy": round(me, 2), "need_mass_window": round(need_m_win, 4)})
        print("  t=%-5d N=%-6d Δbirth=%-6d 饿死累计=%-7d 老死=%-6d ｜ mean_energy=%-6.1f 末段人均需求=%.3f"
              % (a.report_every * (k + 1), n, births, dc.get("STARVATION", 0),
                 dc.get("OLD_AGE", 0), me, need_m_win))
        if e.extinct:
            print("  ⚠️ 灭绝")
            break
    wall = time.time() - t0

    n_end = len(e._flat)
    need_m_meas = need_m_win
    k_meas = g_all / need_m_meas if need_m_meas == need_m_meas and need_m_meas > 0 else float("nan")

    dc = {str(x).split(".")[-1]: y for x, y in e.death_cause_totals().items()}
    starve = dc.get("STARVATION", 0)
    old = dc.get("OLD_AGE", 0)
    result = {
        "seed": a.seed, "regrowth": a.regrowth, "ticks": a.ticks,
        "sigma_regrowth": g_all, "assim_herb": assim,
        "need_mass_prior": need_m_prior, "K_prior": round(k_prior, 1),
        "need_mass_measured": round(need_m_meas, 4) if need_m_meas == need_m_meas else None,
        "K_measured": round(k_meas, 1) if k_meas == k_meas else None,
        "N_end": n_end, "N_over_Kprior": round(n_end / k_prior, 3) if k_prior else None,
        "starved_cum": starve, "old_age_cum": old,
        "starve_share": round(starve / max(1, starve + old), 3),
        "extinct": bool(e.extinct), "wall_sec": round(wall, 1),
        "traj": traj,
    }
    print("-" * 78)
    print("  末态 N=%-6d ｜ K_prior=%.0f ｜ N/K_prior=%.2f ｜ 实测人均需求=%.4f ⇒ K_meas=%.0f"
          % (n_end, k_prior, n_end / k_prior if k_prior else 0, need_m_meas, k_meas))
    print("  死因：饿死 %d / 老死 %d（饿死占比 %.0f%%）｜ 墙钟 %.1fs"
          % (starve, old, 100 * starve / max(1, starve + old), wall))
    ok = (GATE_LO <= n_end <= GATE_HI) and not e.extinct
    print("  13.5 判据 [%.0f, %.0f]：%s" % (GATE_LO, GATE_HI, "✅ 过" if ok else "❌ 不过"))
    if a.json_out:
        with open(a.json_out, "w", encoding="utf-8") as fh:
            json.dump(result, fh, ensure_ascii=False, indent=2)
        print("  已写 %s" % a.json_out)
    return 0 if (ok or not a.gate) else 1


if __name__ == "__main__":
    raise SystemExit(main())
