# -*- coding: utf-8 -*-
"""T1b 原型探针：cos 段 + 内联零散段 **批量化** 的实测增益与逐位校验。

方法（仅探针，不改引擎）：
1. 建同款装置（seed101, rd on, memv2 on, 60k 档）→ 预热 800；
2. 用 wrapper 捕获**一个真实 tick** 的全部 `_memory_egocentric_cos` 调用
   （idx / 格号 / nb 数组 / heading）——即逐个体循环的真实输入面；
3. 在**同一快照**上跑两侧：
   - 标量侧 = 引擎循环体子段逐字复制（cos 调**真实方法**，含调用开销）；
   - 批量侧 = 原型批量化（表 gather + (M,8) 广播 + 批量 cos + 批量 tie/argmax/heading）；
4. 逐位对比（cos 向量 / targets / heading 更新 / 收集三元组）；
5. 交替计时取中位 ⇒ 段省 ms/t @N。

用法：PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe _trash_local/_t1b_batch_proto.py [seed] [warm]
"""
import sys, os, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from experiments.steady_k_probe import make_cfg, apply_post_build
from simulation.sphere_engine import SphereEngine
from simulation.genes import Gene

B = dict(rows=480, cols=960, patches=1700, pop=10000,
         speed_max=0.125, gain=0.125, subdiv=80, k=2.5, max_count=30000, rgm=1.195,
         bg_low_prod_frac=0.4, bg_low_cap_mult=0.05)


def build(seed, mem_on=True):
    c, notes = make_cfg(seed, B["rows"], B["cols"], B["pop"], B["patches"], True,
                        B["speed_max"], B["gain"], B["subdiv"], B["k"],
                        B["max_count"], B["rgm"],
                        bg_low_prod_frac=B["bg_low_prod_frac"],
                        bg_low_cap_mult=B["bg_low_cap_mult"])
    c.simulation.use_sim_core = False
    c.resource_dynamics.enabled = True
    c.info_structure.memory_v2 = bool(mem_on)
    c.info_structure.memory_gradient = "orientation" if mem_on else "none"
    eng = SphereEngine(c)
    apply_post_build(eng, notes)
    return eng


seed = int(sys.argv[1]) if len(sys.argv) > 1 else 101
warm = int(sys.argv[2]) if len(sys.argv) > 2 else 800
eng = build(seed, True)
print(f"预热到 t={warm} …", flush=True)
for _ in range(warm):
    eng.step()
N = len(eng._flat)
print(f"N = {N}（t={warm} 后）", flush=True)

# ---------------- ① 捕获一个真实 tick 的逐个体输入 ----------------
orig_cos = SphereEngine._memory_egocentric_cos
cap = []


def wrap(self, idx, cur, nb, heading_now):
    cap.append((int(idx), int(cur), np.asarray(nb, dtype=np.int64).copy(),
                int(heading_now)))
    return orig_cos(self, idx, cur, nb, heading_now)


SphereEngine._memory_egocentric_cos = wrap
eng.step()
SphereEngine._memory_egocentric_cos = orig_cos

M_all = len(cap)
lens = np.array([len(c[2]) for c in cap])
ok8 = lens == 8
rows_n, cols_n = int(eng.world.rows), int(eng.world.cols)
cells_all = np.array([c[1] for c in cap], dtype=np.int64)
row_of = cells_all // cols_n
std = ok8 & (row_of >= 2) & (row_of <= rows_n - 3)
print(f"捕获：cos 调用 {M_all}（= 移动个体）；len(nb)==8 {int(ok8.sum())}；"
      f"标准行(2..rows-3) {int(std.sum())}；非标准 {int((~std).sum())}"
      f"（len≠8 的 {int((~ok8).sum())} = 极区 960 邻）", flush=True)

pos = np.flatnonzero(std)
Ms = pos.size
miv = np.array([cap[p][0] for p in pos], dtype=np.int64)      # 个体下标 idx
cb = np.array([cap[p][1] for p in pos], dtype=np.int64)       # 格号
NB_cap = np.stack([cap[p][2] for p in pos])                   # (Ms,8) 捕获邻表
HD_cap = np.array([cap[p][3] for p in pos], dtype=np.int64)   # (Ms,)

# 引擎状态（同一快照；两实现输入完全一致）
P = len(eng._flat)
genes = eng._genes
food_ratio = eng.resources._grid / eng._cap_floor
sig_present = (eng.signals._marks > 0).astype(np.float64)
occ = np.bincount(eng._flat[:P], minlength=eng.world.n_cells)
densities = np.divide(occ, eng._nb_norm, dtype=np.float64) \
    * eng.config.simulation.social_move_weight
marks = eng.signals._marks
interp2d = eng._interpret
trust = eng._trust
rep_w = (eng.config.info_structure.reputation_weight
         if eng.config.info_structure.enabled else 0.0)
rand_choice = np.random.default_rng(11).integers(0, 1_000_000, size=Ms, dtype=np.int64)
nb_table = eng.world._nb_table

# 批量路径的前置：表 gather 必须与逐格 neighbors() 同值（标准行）
NB_g = nb_table[cb]
assert (NB_g == NB_cap).all(), "标准行表 gather 与捕获 nb 不一致！"
print("表 gather 校验：标准行 (M,8) 与真实 nb 逐位一致 ✓", flush=True)

# 命中率画像（真实分布的写照）
az_all = eng._mem_az[miv]
tick_all = eng._mem_tick[miv]
valid_all = (az_all >= 0) & ((eng._tick - tick_all) <= eng._mem_ttl)
MK_all = marks[NB_cap]
print(f"画像：有有效记忆槽的行 {int(valid_all.any(axis=1).sum())}/{Ms}；"
      f"全空 {int((~valid_all.any(axis=1)).sum())}；"
      f"有信号候选的行 {int((MK_all > 0).any(axis=1).sum())}", flush=True)

# ---------------- ② 批量实现（原型） ----------------
def batch_cos(miv, cb, nb, hd_now):
    az = eng._mem_az[miv]                                  # (Ms,4) int8
    ticks = eng._mem_tick[miv]                             # (Ms,4)
    valid = (az >= 0) & ((eng._tick - ticks) <= eng._mem_ttl)
    dirs = eng._dirs8_const                                # (8,)
    ang = np.where(
        hd_now[:, None] >= 0,
        (dirs[None, :].astype(np.int64) - hd_now[:, None]) % 8,
        dirs[None, :].astype(np.int64))                    # (Ms,8)
    qs = np.where(valid, az.astype(np.int64), 0)           # 空槽占位 0（随后掩掉）
    cosm = eng._ars_cos[ang[:, :, None], qs[:, None, :]]   # (Ms,8,4)
    deg = eng._mem_degraded[miv]
    dvals = eng._mem_dist[miv].astype(np.float64)
    gains = np.where(deg, np.float64(eng._mem_coarse_gain),
                     np.float64(eng._mem_gain))
    if eng._mem_weight_gene:
        gains = gains * (2.0 * genes[miv, Gene.MEMORY_WEIGHT])[:, None]
    expd = np.where(deg, 1.0,
                    np.exp(-np.clip(dvals, 0.0, 1e6) / eng._mem_dist_scale))
    sc = (gains[:, None, :] * cosm) * expd[:, None, :]     # (Ms,8,4)
    sc = np.where(valid[:, None, :], sc, -np.inf)
    G = sc.max(axis=2)                                     # (Ms,8)
    G[~valid.any(axis=1)] = 0.0                            # 标量：无有效槽 ⇒ 全 0
    return G


def batch_segment(hd_arr):
    """返回 (targets, up_pos, hd_old, hd_new, score)。hd_arr 会被原地更新（std 行）。"""
    nb = NB_g
    hd_now = hd_arr[miv]
    perc = genes[miv, Gene.PERCEPTION]
    soc = (genes[miv, Gene.SOCIABILITY] - 0.5) * 2.0
    fr = food_ratio[nb]
    sp = sig_present[nb]
    sw = trust[miv] * (0.5 + rep_w * trust[miv])
    score = perc[:, None] * (fr * 0.5 + sp * sw[:, None]) + soc[:, None] * densities[nb]
    G = batch_cos(miv, cb, nb, hd_now)
    gmax = np.abs(G).max(axis=1)
    add = gmax > 1e-9
    score[add] += perc[add][:, None] * G[add]
    MK = marks[nb]
    has = (MK > 0).any(axis=1)
    if has.any():
        IN = np.zeros_like(score)
        nz = MK > 0
        ridx = np.broadcast_to(miv[:, None], MK.shape)
        IN[nz] = interp2d[ridx[nz], MK[nz]]
        score[has] += 0.4 * perc[has][:, None] * IN[has]
    smax = score.max(axis=1)
    smin = score.min(axis=1)
    tie = (smax - smin) < 1e-9
    tgt = nb[np.arange(Ms), np.argmax(score, axis=1)]
    if tie.any():
        tp = np.flatnonzero(tie)
        tgt[tp] = nb[tp, rand_choice[tp] % 8]
    EQ = nb == tgt[:, None]
    cnt = EQ.sum(axis=1)
    uniq = cnt == 1
    pos8 = np.argmax(EQ, axis=1)
    up = np.flatnonzero(uniq)
    hd_old = hd_arr[miv[up]].copy()
    hd_arr[miv[up]] = pos8[up]
    return tgt, up, hd_old, pos8[up], score


# ---------------- ③ 标量参考（引擎循环体子段逐字复制） ----------------
def scalar_segment(hd_arr):
    targets = np.empty(Ms, dtype=np.int64)
    up = []
    hd_old_l, hd_new_l = [], []
    scores = []
    for i in range(Ms):
        idx = int(miv[i])
        nb = NB_cap[i]
        hd_now = int(hd_arr[idx])
        perc = genes[idx, Gene.PERCEPTION]
        soc = (genes[idx, Gene.SOCIABILITY] - 0.5) * 2.0
        fr = food_ratio[nb].copy()
        sp = sig_present[nb].copy()
        sig_weight = trust[idx] * (0.5 + rep_w * trust[idx])
        score = perc * (fr * 0.5 + sp * sig_weight) + soc * densities[nb]
        g = orig_cos(eng, idx, int(cb[i]), nb, hd_now)
        if float(np.abs(g).max()) > 1e-9:
            score = score + perc * g
        nb_sigs = marks[nb]
        if (nb_sigs > 0).any():
            itp = np.array(
                [interp2d[idx, int(s)] if s > 0 else 0.0 for s in nb_sigs],
                dtype=np.float64)
            score = score + 0.4 * perc * itp
        if score.max() - score.min() < 1e-9:
            targets[i] = nb[int(rand_choice[i] % len(nb))]
        else:
            targets[i] = nb[int(np.argmax(score))]
        scores.append(score)
        _eq = (nb == targets[i])
        if int(_eq.sum()) == 1:
            _pos = int(np.argmax(_eq))
            _hd_new = _pos if len(nb) == 8 else int(eng.world._VON_NEUMANN_IDX[_pos])
            _hd_old = int(hd_arr[idx])
            hd_arr[idx] = _hd_new
            up.append(i)
            hd_old_l.append(_hd_old)
            hd_new_l.append(_hd_new)
    return (targets, np.array(up, dtype=np.int64),
            np.array(hd_old_l, dtype=np.int64),
            np.array(hd_new_l, dtype=np.int64), scores)


# ---------------- ④ 逐位对拍（同一快照、各自独立 heading 副本） ----------------
hd1 = eng._heading.copy()
hd2 = eng._heading.copy()
t_s, up_s, hdo_s, hdn_s, sc_s = scalar_segment(hd1)
t_b, up_b, hdo_b, hdn_b, sc_b = batch_segment(hd2)

print("—" * 72)
print("【对拍】标量复制 vs 批量原型（同输入）")
print(f"  targets 逐位相等：{bool((t_s == t_b).all())}"
      f"（不等 {int((t_s != t_b).sum())} 行）")
print(f"  heading 更新行集：标量 {up_s.size}｜批量 {up_b.size}；"
      f"行集相等 {bool(np.array_equal(up_s, up_b))}")
print(f"  hd_old/hd_new 逐位相等：{bool(np.array_equal(hdo_s, hdo_b))}"
      f" / {bool(np.array_equal(hdn_s, hdn_b))}")
print(f"  heading 全数组相等：{bool((hd1 == hd2).all())}"
      f"（差异 {int((hd1 != hd2).sum())}）")
neq = int((t_s != t_b).sum())
if neq == 0:
    # score 逐行逐位（更强证据）
    bad = 0
    worst = 0.0
    for i in range(Ms):
        if sc_s[i].shape != sc_b[i].shape:
            bad += 1
            continue
        if not np.array_equal(sc_s[i], sc_b[i]):
            bad += 1
            worst = max(worst, float(np.abs(sc_s[i] - sc_b[i]).max()))
    print(f"  score (Ms,8) 逐位相等行数：{Ms - bad}/{Ms}（不等 {bad}，最大 |Δ| {worst:.3e}）")
# cos 向量逐位（批量 G vs 真实方法逐行）
G_b = batch_cos(miv, cb, NB_g, HD_cap)
cn = 0
for i in range(Ms):
    g_ref = orig_cos(eng, int(miv[i]), int(cb[i]), NB_cap[i], int(HD_cap[i]))
    if not np.array_equal(g_ref, G_b[i]):
        cn += 1
print(f"  cos 向量 G (Ms,8) 逐位相等行数：{Ms - cn}/{Ms}（不等 {cn}）")

# ---------------- ⑤ 交替计时 ----------------
def time_it(fn, reps):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)) * 1e3   # ms


hd1 = eng._heading.copy()
hd2 = eng._heading.copy()
# 预热
scalar_segment(hd1)
batch_segment(hd2)
t_sc, t_ba = [], []
for k in range(7):
    t_sc.append(time_it(lambda: scalar_segment(hd1), 1))
    t_ba.append(time_it(lambda: batch_segment(hd2), 1))
t_sc_m = float(np.median(t_sc))
t_ba_m = float(np.median(t_ba))

# cos 单段对照（只 cos：真实方法逐行 vs 批量）
t_cos_s, t_cos_b = [], []
for k in range(7):
    t_cos_s.append(time_it(lambda: [orig_cos(eng, int(miv[i]), int(cb[i]),
                                             NB_cap[i], int(HD_cap[i]))
                                    for i in range(Ms)], 1))
    t_cos_b.append(time_it(lambda: batch_cos(miv, cb, NB_g, HD_cap), 1))
t_cos_s_m = float(np.median(t_cos_s))
t_cos_b_m = float(np.median(t_cos_b))

print("—" * 72)
print(f"【计时】同一快照、交替中位（Ms={Ms}，tick 段口径 ms/t）")
print(f"  段② 内联+cos 全段： 标量 {t_sc_m:.2f} ms ｜ 批量 {t_ba_m:.2f} ms "
      f"⇒ 省 {t_sc_m - t_ba_m:.2f} ms/t（{t_sc_m / max(t_ba_m, 1e-9):.1f}×）")
print(f"  段① cos 单段：      标量 {t_cos_s_m:.2f} ms ｜ 批量 {t_cos_b_m:.2f} ms "
      f"⇒ 省 {t_cos_s_m - t_cos_b_m:.2f} ms/t（{t_cos_s_m / max(t_cos_b_m, 1e-9):.1f}×）")
print(f"  段② 内联去掉 cos：  标量 {t_sc_m - t_cos_s_m:.2f} ｜ 批量 {t_ba_m - t_cos_b_m:.2f} "
      f"⇒ 省 {(t_sc_m - t_cos_s_m) - (t_ba_m - t_cos_b_m):.2f} ms/t")
print("  （对照：天平计数法 @N=2581：cos 32.8 ｜ 内联 35.6 ⇒ 合计 68.4 ms/t）")

# ---------------- ⑥ 冲刷兼容性：list vs 数组 ----------------
save = (eng._mem_az.copy(), eng._mem_dist.copy(),
        eng._mem_tick.copy(), eng._mem_degraded.copy())
try:
    idx_l = [int(miv[i]) for i in up_s]
    hdo_l = [int(x) for x in hdo_s]
    hdn_l = [int(x) for x in hdn_s]
    eng._mem_v2_step_batch(idx_l, hdo_l, hdn_l, 1.0)
    az_l = eng._mem_az.copy()
    eng._mem_az, eng._mem_dist, eng._mem_tick, eng._mem_degraded = (
        save[0].copy(), save[1].copy(), save[2].copy(), save[3].copy())
    eng._mem_v2_step_batch(miv[up_b], hdo_b, hdn_b, 1.0)
    az_a = eng._mem_az.copy()
    print("【冲刷】_mem_v2_step_batch(list) vs (数组)：逐位相等 "
          f"{bool(np.array_equal(az_l, az_a))}", flush=True)
finally:
    eng._mem_az, eng._mem_dist, eng._mem_tick, eng._mem_degraded = save
print("完成。", flush=True)
