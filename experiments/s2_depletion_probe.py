"""S2 斑块可耗竭 · 代价量化探针（R225 §三 / §四-云归）。

设计依据（R225 §三 预注册判据，先写死）：
  ① 代价   ：K 相对基线（rgm=1.195，K≈915）的下降幅度
  ② 信息价值（核心）：on 臂「斑块级饱和度空间方差」是否显著 > off 臂（控制臂基线方差）
  ③ 消费端信号      ：饿死占比是否上升（S1 基线 0.22–0.30 往上）

⚠️ 判据②口径修正（R225 进度帖已请裁）：R225 §三原写"现状 ≡ 0（全域一个数）"，
   但实测 off 臂（rd 关）斑块方差已 > 0 ⇒ 控制臂天然非 0。故判据②应读作
   "on 臂方差 显著 > off 臂"，本探针在 summary 的 criterion_2 字段直接给 on vs off 比较。

两臂：
  off = resource_dynamics 关（必须逐字段复现 S1 的 K≈915）
  on  = resource_dynamics 开（局部休耕—死亡—轮作 ⇒ 斑块可耗竭）

🔴 配置对齐纪律：控制臂必须逐字段对齐 S1 的 1.195 档（tools/s1_rgm_scan.py 的 T4 字典
   + build_cmd），否则 K / 饿死占比 不可与 S1 比（同 T4 P0 的 provenance 坑）。
   S1 基线：rows=480 cols=960 patches=480 pop=2000 ticks=40000
            subpos=on speed_max=0.125 gain=0.125 subdiv=80 k=2.5 max_count=30000
            patch_regrowth_mult=1.195  stop_stable=0（无早停）
   两臂统一走 Python 路径（use_sim_core=False）。

判据② 的新指标「斑块级饱和度」实现：把 `_patch_mask` 按 `world.neighbors`
做连通分量标注（尊重球面拓扑 + 经度缠绕）。

🔴 指标修复（R225 复盘）：原版 `sat = Σgrid / Σ(实时)capacity` 是**相对比值**，
   斑块被 rd 杀死时「容量↓ + 存量↓」同步发生 ⇒ 比值仍≈1.0 ⇒ 绝对崩塌被完全遮住；
   且 `cap_sum` 只在初始化算一次，rd 运行中降容量后分母失效。
   现改为三路口径：
     · patch_sat_*     ：Σgrid / Σ(实时)capacity  —— 旧相对口径（保留，作对照）
     · patch_sat_init_*：Σgrid / Σ(初始)capacity  —— 被杀/耗竭斑块显著↓，
                         能直接看见「哪些斑块塌了」⇒ 空间结构信号
     · patch_abs_food_*：每块绝对食物量（不归一）的方差 —— 纯空间异质性
   全局口径同步改为实时容量：global_sat = 实时stock / 实时cap；并补 abs_food /
   abs_cap / cap_lost_frac（相对初始总容量的损失，rd 杀死斑块会直接抬升）。
"""

import argparse
import json
import os
import time
from collections import deque

import numpy as np

from simulation.sphere_engine import SphereEngine
from experiments.steady_k_probe import make_cfg, apply_post_build


# S1 1.195 档基线配置（tools/s1_rgm_scan.py 的 T4 字典 + build_cmd）。
# 控制臂必须逐字段对齐，否则 K≈915 / 饿死占比 0.22–0.30 不可复现。
S1_BASE = dict(
    rows=480, cols=960, patches=480, pop=2000,
    speed_max=0.125, gain=0.125, subdiv=80, k=2.5, max_count=30000,
)


# ---------- R241 四层指标（L0 轮休 / L1 探索 / L2 分化 / L3 人均摄入） ----------
#
# 依据：R241 复议方案（讨论板 2026-09-28 02:1x，青梧提出，云归实现）。
# 背景：旧判据②（全局斑块方差）天花板敏感、且只测"食物不均匀"，
#       没测"生物会不会主动离开"。语义链 = ①斑块可耗尽 → ②不均匀 →
#       ③**生物主动离开找新食物** → ④信息有价值 → ⑤C3。②③④ 缺直接判据。
#
#   L0 机制验证：rd 到底有没有在轮休（读 rd 内部数组，零额外成本）
#   L1 探索度  ：生物会不会离开去找新食物（逐 tick visit 集合）
#   L2 分化度  ：访问过的斑块间剩余食物不均（排除死区）+ 基尼 + 方差分解
#   L3 人均摄入：每个生物的真实待遇（人口归一）
#
# 🔴 只读引擎公开/内部已有量，不碰引擎（R241 §五 边界）。

def l0_rd_state(eng) -> dict:
    """L0：读 rd 内部数组，看轮休/耗竭事件规模。

    恒 0 ⇒ rd 没在轮休（regen 太快/没吃空）⇒ 先调参，别判效应。
    """
    rd = getattr(eng, "_rd", None)
    if rd is None:
        return {"l0_dead_n": 0, "l0_rest_n": 0, "l0_demoted_n": 0,
                "l0_mask_n": 0, "l0_rd_available": 0}
    dead = getattr(rd, "_dead", None)
    rest = getattr(rd, "_rest_until", None)
    mask = getattr(rd, "_mask", None)
    demoted = getattr(rd, "_demoted", None)
    return {
        "l0_rd_available": 1,
        "l0_dead_n": int(dead.sum()) if dead is not None else 0,
        "l0_rest_n": int((rest >= 0).sum()) if rest is not None else 0,
        "l0_demoted_n": int(demoted.sum()) if demoted is not None else 0,
        "l0_mask_n": int(mask.sum()) if mask is not None else 0,
    }


def gini(x: np.ndarray) -> float:
    """基尼系数（0=完全均等，1=极度集中）。空/单元素 ⇒ 0。"""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size <= 1:
        return 0.0
    if x.min() < 0:
        x = x - x.min()
    s = x.sum()
    if s <= 0:
        return 0.0
    xs = np.sort(x)
    n = xs.size
    idx = np.arange(1, n + 1, dtype=float)
    return float((2.0 * (idx * xs).sum()) / (n * s) - (n + 1.0) / n)


def l2_variance_decomposition(grid: np.ndarray, labels: np.ndarray,
                              visited: np.ndarray) -> dict:
    """L2：访问掩码内的方差/基尼 + 方差分解。

    σ²_total = p·σ²_visited + p(1−p)(μ_v − μ_u)²
      p       = 访问斑块占比
      σ²_visited = 访问组内方差
      (μ_v−μ_u)² = 组间均值差（访问 vs 未访问）
    """
    labs = labels[labels >= 0]
    g = grid[labels >= 0]
    if labs.size == 0:
        return {"l2_visited_gini": 0.0, "l2_visited_var": 0.0,
                "l2_total_var": 0.0, "l2_between_share": 0.0,
                "l2_visited_frac": 0.0, "l2_visited_mean": 0.0,
                "l2_unvisited_mean": 0.0, "l2_total_gini": 0.0}
    uniq, inv = np.unique(labs, return_inverse=True)
    patch_food = np.zeros(uniq.size, dtype=float)
    np.add.at(patch_food, inv, g)
    vis_mask = np.zeros(uniq.size, dtype=bool)
    for lab in np.unique(labels[visited]):
        if lab >= 0:
            vis_mask[np.searchsorted(uniq, lab)] = True
    n_vis = int(vis_mask.sum())
    p = n_vis / uniq.size if uniq.size else 0.0
    total_var = float(patch_food.var())
    if n_vis == 0 or n_vis == uniq.size:
        var_v = float(patch_food[vis_mask].var()) if n_vis else 0.0
        return {"l2_visited_gini": gini(patch_food[vis_mask]) if n_vis else 0.0,
                "l2_visited_var": var_v,
                "l2_total_var": total_var,
                "l2_between_share": 0.0,
                "l2_visited_frac": float(p),
                "l2_visited_mean": float(patch_food[vis_mask].mean()) if n_vis else 0.0,
                "l2_unvisited_mean": float(patch_food[~vis_mask].mean()) if n_vis < uniq.size else 0.0,
                "l2_total_gini": gini(patch_food)}
    v = patch_food[vis_mask]
    u = patch_food[~vis_mask]
    var_v = float(v.var())
    between = p * (1.0 - p) * float((v.mean() - u.mean()) ** 2)
    return {
        "l2_visited_gini": gini(v),
        "l2_visited_var": var_v,
        "l2_total_var": total_var,
        "l2_between_share": (between / total_var) if total_var > 0 else 0.0,
        "l2_visited_frac": float(p),
        "l2_visited_mean": float(v.mean()),
        "l2_unvisited_mean": float(u.mean()),
        "l2_total_gini": gini(patch_food),
    }


# ---------- 斑块级饱和度（判据② 新指标） ----------

def _label_patches(world, patch_mask: np.ndarray) -> np.ndarray:
    """用 world.neighbors 对 patch_mask 做连通分量标注（尊重拓扑 + 经度缠绕）。

    返回 shape=(n_cells,) 的 int 标签数组，非斑块格为 -1。
    """
    n = world.n_cells
    labels = np.full(n, -1, dtype=np.int64)
    patch_cells = np.flatnonzero(patch_mask)
    nxt = 0
    for seed in patch_cells:
        if labels[seed] != -1:
            continue
        lab = nxt
        nxt += 1
        q = deque([int(seed)])
        labels[seed] = lab
        while q:
            c = q.popleft()
            for nb in world.neighbors(c):
                nb = int(nb)
                if patch_mask[nb] and labels[nb] == -1:
                    labels[nb] = lab
                    q.append(nb)
    return labels


def patch_saturation(world, eng, labels: np.ndarray,
                     init_patch_cap: dict) -> dict:
    """返回斑块级饱和度的统计（三路口径，见文件头注释）。

    init_patch_cap: {label: 该斑块初始容量和} —— 用初始化时的 capacity 算，
                    被杀/耗竭斑块的当前 grid 相对它显著↓ ⇒ 空间结构可见。
    """
    grid = eng.resources._grid
    cap = eng.resources._capacity          # 实时容量（rd 会改变）
    patch_cells = labels >= 0
    if not patch_cells.any():
        return {"n_patches": 0, "patch_sat_mean": float("nan"),
                "patch_sat_var": 0.0, "patch_sat_range": 0.0,
                "patch_sat_init_var": 0.0, "patch_abs_food_var": 0.0}
    labs = labels[patch_cells]
    g = grid[patch_cells]
    c = cap[patch_cells]
    uniq = np.unique(labs)
    sats, sats_init, abs_food = [], [], []
    for lab in uniq:
        m = labs == lab
        denom_live = c[m].sum()
        sats.append(g[m].sum() / denom_live if denom_live > 0 else 0.0)
        ic = init_patch_cap.get(int(lab), 0.0)
        sats_init.append(g[m].sum() / ic if ic > 0 else 0.0)
        abs_food.append(g[m].sum())
    sats = np.array(sats, dtype=float)
    sats_init = np.array(sats_init, dtype=float)
    abs_food = np.array(abs_food, dtype=float)
    return {
        "n_patches": int(uniq.size),
        "patch_sat_mean": float(sats.mean()),
        "patch_sat_var": float(sats.var()),
        "patch_sat_range": float(sats.max() - sats.min()),
        "patch_sat_init_var": float(sats_init.var()),
        "patch_abs_food_var": float(abs_food.var()),
    }


# ---------- 单 run ----------

def run_one(seed, rows, cols, pop, patches, ticks, sample, rgm, rd_on):
    # 🔴 对齐 S1：subpos=on + S1_BASE 的 k/gain/subdiv/speed_max/max_count
    c, notes = make_cfg(
        seed, rows, cols, pop, patches, True,
        S1_BASE["speed_max"], S1_BASE["gain"], S1_BASE["subdiv"],
        S1_BASE["k"], S1_BASE["max_count"], rgm,
    )
    c.simulation.use_sim_core = False          # 两臂统一 Python 路径（= S1 基线）
    c.resource_dynamics.enabled = bool(rd_on)  # 处理臂开局部可耗竭
    eng = SphereEngine(c)
    apply_post_build(eng, notes)

    # 🔴 用初始化时的 capacity 快照算「每块初始容量和」+ 初始总容量
    #    （rd 运行中会杀斑块、降容量，必须用初始值做分母才看得到损失）
    init_cap_full = eng.resources._capacity.copy()
    labels = _label_patches(eng.world, eng.resources._patch_mask)
    uniq0 = np.unique(labels[labels >= 0])
    init_patch_cap = {int(lab): float(init_cap_full[labels == lab].sum())
                     for lab in uniq0}
    init_cap_sum = float(eng.resources._capacity.sum())

    rows_out = []
    t_slice = time.perf_counter()
    last = 0
    stop = "tick 用尽"
    # R241 L1：逐 tick 累积「被访问过的斑块」；first_visit_tick 记首访
    visited = np.zeros(eng.world.n_cells, dtype=bool)
    first_visit = {}
    n_patch_total = int(np.unique(labels[labels >= 0]).size)
    # R241 L0：轮休事件累计（逐 tick 采样，取每个采样点的瞬时值 + 全程累计峰值）
    l0_peak = {"dead": 0, "rest": 0, "demoted": 0}
    l0_any = {"dead": 0, "rest": 0, "demoted": 0}
    for t in range(1, ticks + 1):
        eng.step()
        # L1：本 tick 的生物占格 → 折算到斑块标签（向量化，仅 bgzero 档 labels 稳定）
        flat = eng._flat
        if flat.size and n_patch_total:
            cells = np.unique(flat)
            visited[cells] = True
            lab_here = labels[cells]
            for lab in np.unique(lab_here):
                if lab >= 0 and int(lab) not in first_visit:
                    first_visit[int(lab)] = t
        # L0：瞬时轮休状态（每 tick 读，成本可忽略：两个布尔/整数数组 sum）
        l0 = l0_rd_state(eng)
        if l0["l0_rd_available"]:
            l0_peak["dead"] = max(l0_peak["dead"], l0["l0_dead_n"])
            l0_peak["rest"] = max(l0_peak["rest"], l0["l0_rest_n"])
            l0_peak["demoted"] = max(l0_peak["demoted"], l0["l0_demoted_n"])
            l0_any["dead"] += 1 if l0["l0_dead_n"] > 0 else 0
            l0_any["rest"] += 1 if l0["l0_rest_n"] > 0 else 0
            l0_any["demoted"] += 1 if l0["l0_demoted_n"] > 0 else 0
        if eng.world.n_cells and len(eng._flat) == 0:
            stop = "灭绝"
        if t % sample == 0 or t == ticks or stop == "灭绝":
            dt = (time.perf_counter() - t_slice) / max(t - last, 1) * 1e3
            last = t
            t_slice = time.perf_counter()
            N = int(len(eng._flat))
            cap_live = eng.resources._capacity          # 实时容量
            prod = cap_live > 0
            cap_sum_live = float(cap_live[prod].sum())
            stock = float(eng.resources._grid[prod].sum())
            sat = stock / cap_sum_live if cap_sum_live > 0 else float("nan")
            cap_lost_frac = (1.0 - cap_sum_live / init_cap_sum
                             if init_cap_sum > 0 else float("nan"))
            dc = eng.death_cause_totals()
            # 键为 DeathCause 枚举：str(k)="DeathCause.STARVATION" → 取末段
            dv = {str(k).split(".")[-1]: int(v) for k, v in dc.items()}
            ps = patch_saturation(eng.world, eng, labels, init_patch_cap)
            # ---- L1 探索度（累积到本采样点） ----
            n_vis_patch = len(first_visit)
            fv = np.array(sorted(first_visit.values()), dtype=float)
            l1 = {
                "l1_visited_patch_n": n_vis_patch,
                "l1_visited_patch_frac": (n_vis_patch / n_patch_total
                                          if n_patch_total else 0.0),
                "l1_first_visit_median": float(np.median(fv)) if fv.size else float("nan"),
                "l1_first_visit_p25": float(np.percentile(fv, 25)) if fv.size else float("nan"),
                "l1_first_visit_p75": float(np.percentile(fv, 75)) if fv.size else float("nan"),
            }
            # ---- L2 分化度（访问掩码内 + 方差分解） ----
            l2 = l2_variance_decomposition(eng.resources._grid, labels, visited)
            # ---- L3 人均摄入（累计算，人口归一） ----
            l3 = {"l3_forage_sum": float("nan"), "l3_intake_per_capita": float("nan"),
                  "l3_net_sum": float("nan"), "l3_net_per_capita": float("nan"),
                  "l3_pop_mean": float("nan")}
            if t == ticks or stop == "灭绝":
                try:
                    led = eng.energy_ledger()
                    gsum = led.get("global", {})
                    fs = float(gsum.get("intake_forage_sum", 0.0) or 0.0)
                    nsum = (float(gsum.get("intake_forage_sum", 0.0) or 0.0)
                            + float(gsum.get("intake_pred_sum", 0.0) or 0.0)
                            + float(gsum.get("intake_scav_sum", 0.0) or 0.0)
                            - float(gsum.get("cost_meta_sum", 0.0) or 0.0)
                            - float(gsum.get("cost_move_sum", 0.0) or 0.0)
                            - float(gsum.get("cost_attack_sum", 0.0) or 0.0))
                    # 人口均值：以采样点人口（含本点 N）估算 N̄
                    pops = [r["pop"] for r in rows_out] + [N]
                    pop_mean = sum(pops) / len(pops)
                    denom = max(pop_mean * t, 1.0)
                    l3 = {"l3_forage_sum": fs,
                          "l3_intake_per_capita": fs / denom,
                          "l3_net_sum": nsum,
                          "l3_net_per_capita": nsum / denom,
                          "l3_pop_mean": pop_mean}
                except Exception as e:
                    l3["l3_note"] = f"energy_ledger 不可用: {e}"
            rows_out.append({
                "seed": seed, "arm": "on" if rd_on else "off", "tick": t,
                "pop": N, "global_sat": sat,
                "abs_food": stock, "abs_cap": cap_sum_live,
                "cap_lost_frac": cap_lost_frac,
                "d_starv": dv.get("STARVATION", 0),
                "d_old": dv.get("OLD_AGE", 0),
                "d_pred": dv.get("PREDATION", 0),
                "ms_per_tick": dt, **ps, **l1, **l2, **l3,
                "l0_peak_dead": l0_peak["dead"], "l0_peak_rest": l0_peak["rest"],
                "l0_peak_demoted": l0_peak["demoted"],
                "l0_any_dead": l0_any["dead"], "l0_any_rest": l0_any["rest"],
                "l0_any_demoted": l0_any["demoted"],
            })
            if stop == "灭绝":
                break
    return rows_out, stop


def _tail_mean(rows_out, key, n=3):
    tail = rows_out[-n:]
    if not tail:
        return 0.0
    return sum(r[key] for r in tail) / len(tail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=S1_BASE["rows"])
    ap.add_argument("--cols", type=int, default=S1_BASE["cols"])
    ap.add_argument("--patches", type=int, default=S1_BASE["patches"])
    ap.add_argument("--pop", type=int, default=S1_BASE["pop"])
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--ticks", type=int, default=40000)
    ap.add_argument("--sample", type=int, default=250)
    ap.add_argument("--rgm", type=float, default=1.195)
    ap.add_argument("--arms", choices=("both", "on", "off"), default="both")
    ap.add_argument("--out", default="results/s2_depletion.csv")
    a = ap.parse_args()

    seeds = [int(x) for x in a.seeds.split(",") if x.strip()]
    arms = (["off", "on"] if a.arms == "both" else [a.arms])
    header = ["seed", "arm", "tick", "pop", "global_sat",
              "abs_food", "abs_cap", "cap_lost_frac",
              "d_starv", "d_old", "d_pred", "ms_per_tick",
              "n_patches", "patch_sat_mean", "patch_sat_var", "patch_sat_range",
              "patch_sat_init_var", "patch_abs_food_var",
              # ---- R241 L1 探索度 ----
              "l1_visited_patch_n", "l1_visited_patch_frac",
              "l1_first_visit_median", "l1_first_visit_p25", "l1_first_visit_p75",
              # ---- R241 L2 分化度 ----
              "l2_visited_gini", "l2_visited_var", "l2_total_var",
              "l2_between_share", "l2_visited_frac", "l2_visited_mean",
              "l2_unvisited_mean", "l2_total_gini",
              # ---- R241 L3 人均摄入 ----
              "l3_forage_sum", "l3_intake_per_capita",
              "l3_net_sum", "l3_net_per_capita", "l3_pop_mean",
              # ---- R241 L0 轮休（累计峰值 / 出现 tick 数） ----
              "l0_peak_dead", "l0_peak_rest", "l0_peak_demoted",
              "l0_any_dead", "l0_any_rest", "l0_any_demoted"]

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        f.write(",".join(header) + "\n")
        print(f"# S2 探针：rows={a.rows} cols={a.cols} patches={a.patches} "
              f"pop={a.pop} rgm={a.rgm} ticks={a.ticks} seeds={seeds} arms={arms}")
        print(",".join(header))
        summary = []
        for sd in seeds:
            for arm in arms:
                t0 = time.time()
                rows_out, stop = run_one(sd, a.rows, a.cols, a.pop, a.patches,
                                         a.ticks, a.sample, a.rgm, arm == "on")
                for r in rows_out:
                    line = ",".join(str(r[h]) for h in header)
                    f.write(line + "\n")
                    print(line)
                last = rows_out[-1]
                wall = time.time() - t0
                var_tail = _tail_mean(rows_out, "patch_sat_init_var")
                var_live_tail = _tail_mean(rows_out, "patch_sat_var")
                summary.append({
                    "seed": sd, "arm": arm, "stop": stop,
                    "K_final": last["pop"], "global_sat": last["global_sat"],
                    "cap_lost_frac": last["cap_lost_frac"],
                    "patch_sat_init_var": last["patch_sat_init_var"],
                    "patch_sat_init_var_tail": round(var_tail, 6),
                    "patch_sat_var": last["patch_sat_var"],
                    "patch_sat_var_tail": round(var_live_tail, 6),
                    "patch_abs_food_var": last["patch_abs_food_var"],
                    "patch_sat_range": last["patch_sat_range"],
                    "n_patches": last["n_patches"],
                    "d_starv": last["d_starv"], "d_pred": last["d_pred"],
                    "wall_s": round(wall, 1),
                    # ---- R241 四层 ----
                    "l0_peak_dead": last["l0_peak_dead"],
                    "l0_peak_rest": last["l0_peak_rest"],
                    "l0_peak_demoted": last["l0_peak_demoted"],
                    "l0_any_dead": last["l0_any_dead"],
                    "l0_any_rest": last["l0_any_rest"],
                    "l0_any_demoted": last["l0_any_demoted"],
                    "l1_visited_patch_n": last["l1_visited_patch_n"],
                    "l1_visited_patch_frac": last["l1_visited_patch_frac"],
                    "l1_first_visit_median": last["l1_first_visit_median"],
                    "l2_visited_gini": last["l2_visited_gini"],
                    "l2_visited_var": last["l2_visited_var"],
                    "l2_total_var": last["l2_total_var"],
                    "l2_between_share": last["l2_between_share"],
                    "l2_visited_frac": last["l2_visited_frac"],
                    "l3_forage_sum": last["l3_forage_sum"],
                    "l3_intake_per_capita": last["l3_intake_per_capita"],
                    "l3_net_sum": last["l3_net_sum"],
                    "l3_net_per_capita": last["l3_net_per_capita"],
                    "l3_pop_mean": last["l3_pop_mean"],
                })
                print(f"# → seed {sd} arm {arm}: K={last['pop']} "
                      f"sat={last['global_sat']:.3f} "
                      f"capLost={last['cap_lost_frac']:.3f} "
                      f"patchVarInit={last['patch_sat_init_var']:.5f} "
                      f"(tail {var_tail:.5f}) "
                      f"patchRange={last['patch_sat_range']:.3f} | "
                      f"L0 dead={last['l0_peak_dead']}/rest={last['l0_peak_rest']} "
                      f"anyD={last['l0_any_dead']} anyR={last['l0_any_rest']} | "
                      f"L1 vis={last['l1_visited_patch_n']}/{last['n_patches']} "
                      f"({last['l1_visited_patch_frac']:.3f}) "
                      f"med={last['l1_first_visit_median']} | "
                      f"L2 gini={last['l2_visited_gini']:.4f} "
                      f"vvar={last['l2_visited_var']:.1f} "
                      f"betw={last['l2_between_share']:.3f} | "
                      f"L3 in/cap={last['l3_intake_per_capita']:.1f} "
                      f"net/cap={last['l3_net_per_capita']:.1f} | "
                      f"stop={stop} wall={wall:.0f}s", flush=True)

    # ---------- 判据②（修正口径）：on 臂斑块「相对初始容量」方差 是否 > off 臂 ----------
    #   用 patch_sat_init_var（被杀/耗竭斑块显著↓ ⇒ 空间结构可见）；
    #   同时保留旧相对口径 patch_sat_var 作对照。
    by_seed = {}
    for s in summary:
        by_seed.setdefault(s["seed"], {})[s["arm"]] = s
    c2 = {}
    for sd, arms_d in by_seed.items():
        o = arms_d.get("off")
        n = arms_d.get("on")
        if o is not None and n is not None and o["patch_sat_init_var_tail"] > 0:
            c2[sd] = {
                "off_var_init_tail": o["patch_sat_init_var_tail"],
                "on_var_init_tail": n["patch_sat_init_var_tail"],
                "on_gt_off_init": bool(n["patch_sat_init_var_tail"]
                                       > o["patch_sat_init_var_tail"]),
                "ratio_on_off_init": round(n["patch_sat_init_var_tail"]
                                           / o["patch_sat_init_var_tail"], 3),
                "off_var_live_tail": o["patch_sat_var_tail"],
                "on_var_live_tail": n["patch_sat_var_tail"],
            }
    off_vals = [v["off_var_init_tail"] for v in c2.values()]
    on_vals = [v["on_var_init_tail"] for v in c2.values()]
    c2_overall = {
        "off_var_init_mean": round(sum(off_vals) / len(off_vals), 6) if off_vals else None,
        "on_var_init_mean": round(sum(on_vals) / len(on_vals), 6) if on_vals else None,
        "verdict_on_gt_off_init": (all(v["on_gt_off_init"] for v in c2.values())
                                   if c2 else None),
    }

    out = {
        "params": vars(a),
        "summary": summary,
        "criterion_2": {"per_seed": c2, "overall": c2_overall},
    }
    with open(a.out.replace(".csv", ".summary.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\n=== S2 summary ===")
    for s in summary:
        print(s)
    print("\n=== 判据②（on > off 斑块方差）===")
    print(json.dumps(out["criterion_2"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
