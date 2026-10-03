# -*- coding: utf-8 -*-
"""R291 §四 派工 ①③（补跑）：算法路线子操作微基准（合成数组，按 v1 实测稀疏度）。

不构建引擎（省 ~5 min 预热 + 内存）：n=460800，稀疏度照 `_rd_audit.log` 实测
（dead=4589 / rest=1061 / damage≥thr=5123 / eaten≈0.43%）。绝对 ms 受本机 60k 批跑
放大 ⇒ 只看**配对差**（同窗口）+ 与 v1 段计时交叉验证。

三个候选：
  P1 capacity 合并：全量 `capacity_from_base()` vs 缓存 + `cap[eaten]` gather。
  P2 note_tick 子集化：逐条全量表达式计时（给出可省上界）+ 子集实测。
  P3 growth_multiplier 子集 / rotate 扫描子集：全量 vs 子集逐位 + 配对计时。
"""
import time
import numpy as np

rng = np.random.default_rng(20260930)
n = 460_800
n_dead, n_rest, n_eaten, n_patch = 4589, 1061, 1980, 12805

# ---- 合成状态（形状/稀疏度照实测；值域合理即可）----
mask = np.zeros(n, dtype=bool)
mask[rng.choice(n, n_patch, replace=False)] = True
base = rng.uniform(30.0, 60.0, n)
pmult, bmult = 1.195, 0.0            # 装置口径（bgzero ⇒ 背景容量倍率 0）
dead = np.zeros(n, dtype=bool)
dead[rng.choice(n, n_dead, replace=False)] = True
rest_until = np.full(n, -1, dtype=np.int64)
rest_until[rng.choice(n, n_rest, replace=False)] = 1500
damage = rng.uniform(0.0, 0.2, n)
damage[rng.choice(n, 5123, replace=False)] = rng.uniform(0.3, 0.9, 5123)
intake = np.zeros(n, dtype=np.float64)
eat_idx = rng.choice(n, n_eaten, replace=False)
intake[eat_idx] = rng.uniform(0.05, 13.0, n_eaten)
growth = rng.uniform(0.0, 0.03, n)
dirty_idx = rng.choice(n, 3895, replace=False)

def bench(name, fn, rep=30):
    fn()
    t0 = time.perf_counter()
    for _ in range(rep):
        fn()
    dt = (time.perf_counter() - t0) / rep * 1e3
    print(f"  {name:<52} {dt:8.3f} ms/次", flush=True)
    return dt

print(f"n={n} dead={n_dead} rest={n_rest} eaten={n_eaten} patch={n_patch}")

print("— P1 capacity 合并 —", flush=True)
t_full = bench("capacity_from_base 全量（现状，本次/每 tick 两处）",
               lambda: base * np.where(mask, pmult, bmult))
cap_cache = base * np.where(mask, pmult, bmult)
t_gather = bench("cap[eaten] gather（合并后 note_tick 用）",
                 lambda: cap_cache[eat_idx])
t_copy = bench("_capacity[:] = cap_new 回写拷贝（可随合并跳过）",
               lambda: cap_cache.copy())
eq1 = np.array_equal((base * np.where(mask, pmult, bmult))[eat_idx], cap_cache[eat_idx])
print(f"  等价：gather == 全量[eaten]：{eq1}", flush=True)

print("— P2 note_tick 子集化 —", flush=True)
def nt_full_ops():
    eaten = intake > 0.0
    if eaten.any():
        denom = np.where(cap_cache > 0.0, cap_cache, 1.0)
        d = damage.copy()
        d[eaten] += intake[eaten] / denom[eaten]
    newly = np.zeros(n, dtype=bool)
    if eaten.any():
        newly |= eaten & (~dead) & (damage >= 0.3)
    valid = (growth > 0.0) & (~dead) & (intake > 0.0)
    if valid.any():
        ratio = np.zeros(n, dtype=np.float64)
        ratio[valid] = intake[valid] / growth[valid]
        newly |= valid & (ratio > 0.5)
    _ = int(newly.sum())
    enter = eaten & (~dead) & (rest_until < 0) & (damage >= 0.3)
    _ = int(enter.sum())
    return d, newly

def nt_sub_ops():
    eaten = intake > 0.0
    if eaten.any():
        eidx = np.flatnonzero(eaten)
        dmg_e = damage[eidx]
        dmg_e += intake[eidx] / np.where(cap_cache[eidx] > 0.0, cap_cache[eidx], 1.0)
        dead_e = dead[eidx]
        newly_e = (~dead_e) & (dmg_e >= 0.3)
        nk = growth[eidx] > 0.0
        ratio = np.zeros(eidx.size)
        ratio[nk] = intake[eidx][nk] / growth[eidx][nk]
        newly_e |= nk & (ratio > 0.5)
        _ = int(newly_e.sum())
        rest_e = (~dead_e) & (rest_until[eidx] < 0) & (dmg_e >= 0.3)
        _ = int(rest_e.sum())
        d = damage.copy(); d[eidx] = dmg_e
    return d, None

bench("note_tick 全量表达式组（现状近似）", nt_full_ops)
bench("note_tick 子集表达式组（eaten 子集）", nt_sub_ops)
d1, _ = nt_full_ops(); d2, _ = nt_sub_ops()
print(f"  等价：damage 变更逐位一致：{np.array_equal(d1, d2)}", flush=True)

print("— P3 growth_multiplier / rotate 扫描 —", flush=True)
def gm_full():
    mult = np.ones(n)
    if dead.any():
        mult[dead] = 0.5
    resting = rest_until >= 0
    if resting.any():
        mult[resting] = np.minimum(mult[resting], 0.0)
    return mult

gm_full()
m_full = gm_full()
def gm_sub():
    dd = dead[dirty_idx]
    m = np.where(dd, 0.5, 1.0)
    rr = rest_until[dirty_idx] >= 0
    m = np.minimum(m, np.where(rr, 0.0, 1.0))
    return m

m_sub = gm_sub()
t_gm_full = bench("growth_multiplier 全量（现状）", gm_full)
t_gm_sub = bench("growth_multiplier 子集（dirty 3895）", gm_sub)
print(f"  等价：全量[dirty] == 子集：{np.array_equal(m_full[dirty_idx], m_sub)}", flush=True)

dead_since = np.full(n, 1000, dtype=np.int64)   # 预分配（真实代码也是既有数组，无逐次分配）

def rot_full():
    expired = (rest_until >= 0) & (~dead) & (1500 >= rest_until)
    _ = int(expired.sum())
    due = dead & ((1500 - dead_since) >= 500)
    _ = int(due.sum())
    _ = int(dead.sum())

def rot_sub():
    ri = np.flatnonzero(rest_until >= 0)
    expired = (~dead[ri]) & (1500 >= rest_until[ri])
    _ = int(expired.sum())
    di = np.flatnonzero(dead)
    due = ((1500 - dead_since[di]) >= 500)
    _ = int(due.sum())

t_rot_full = bench("rotate 扫描组 全量（expired+due+sum）", rot_full)
t_rot_sub = bench("rotate 扫描组 子集（rest/dead 索引 + 子集判定）", rot_sub)

print("— 汇总 —", flush=True)
print(f"  P1 合并：全量 {t_full:.3f} ×2 处 + 回写拷贝 {t_copy:.3f} − gather {t_gather:.3f}"
      f" ⇒ 总省 ≈ {2 * t_full + t_copy - t_gather:.3f} ms/（本窗口）", flush=True)
print(f"  P3 gm 子集省 ≈ {t_gm_full - t_gm_sub:.3f} ms｜rotate 扫描省 ≈ "
      f"{t_rot_full - t_rot_sub:.3f} ms", flush=True)
