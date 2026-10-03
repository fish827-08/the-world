# -*- coding: utf-8 -*-
"""R315 分析：验证 fish 的「行为二分 / 基因不分」假设。

核心问题：
  Q1 出走型（离斑块远的个体）与留守型，在基因 g22（记忆权重）上是否分化？
     —— 若 g22 是决定因素，出走群体应系统性偏低（或偏高）。
  Q2 off 臂的"出走者"是否处境更差（能量/健康）——支持「出走=死路」？
  Q3 各 seed 的斑块几何是否不同——支持「初始投放决定出走型存活」？

数据：short_hr_20k 终态快照（seed 189-192 × on/off，sample=2000，--keep-ckpt 保留）。
      快照含 resource_patch_mask / flat / genes / energy / health / age / generation。

口径：**瞬时**（当前位置到最近斑块的栅格距离）。注意探针层的「累计最大位移」
      没有进快照，所以这是**横截面代理**，不是累计活动范围。
      栅格距离用多源 BFS：列环绕、极点坍缩（与引擎 rc_to_flat 一致）。
"""
import numpy as np
import os

ROWS, COLS = 480, 960
TOTAL = ROWS * COLS
G22 = 22  # Gene.MEMORY_WEIGHT

SNAP = "_rerun_logs/short_hr_20k/results/snap/s3_s%d_mem_%s.snapshot.npz"

BINS = [0, 5, 10, 20, 50, 10 ** 9]
LABELS = ["0-5", "5-10", "10-20", "20-50", "50+"]


def nearest_patch_dist(pm, maxd=120):
    """多源 BFS：每格到最近斑块格的栅格距离。未达 = -1。"""
    d = np.full(TOTAL, -1, dtype=np.int32)
    d[pm] = 0
    frontier = np.where(pm)[0]
    step = 0
    while frontier.size and step < maxd:
        step += 1
        r = frontier // COLS
        c = frontier % COLS
        nbs = [
            np.where(r > 0, frontier - COLS, -1),           # 上一行（极点坍缩）
            np.where(r < ROWS - 1, frontier + COLS, -1),    # 下一行
            r * COLS + (c - 1) % COLS,                      # 左（环绕）
            r * COLS + (c + 1) % COLS,                      # 右（环绕）
        ]
        cand = np.concatenate(nbs)
        cand = cand[cand >= 0]
        cand = np.unique(cand)
        cand = cand[d[cand] < 0]
        if cand.size == 0:
            break
        d[cand] = step
        frontier = cand
    return d


def load(s, arm):
    z = np.load(SNAP % (s, arm), allow_pickle=True)
    return dict(pm=z["resource_patch_mask"], flat=z["flat"],
                g=z["genes"][:, G22], en=z["energy"], hp=z["health"],
                age=z["age"], gen=z["generation"])


def main():
    cache = {}
    for s in (189, 190, 191, 192):
        for arm in ("off", "on"):
            if os.path.exists(SNAP % (s, arm)):
                a = load(s, arm)
                a["d"] = nearest_patch_dist(a["pm"])
                a["dist"] = a["d"][a["flat"]]
                a["inside"] = a["pm"][a["flat"]]
                cache[(s, arm)] = a

    print("=" * 100)
    print("A. 瞬时「斑块内 vs 斑块外」的 g22 / 能量 / 健康  （4 世界 × 2 臂）")
    print("=" * 100)
    print("%-5s %-4s %7s %8s %8s %10s %10s %9s %9s %9s %9s" % (
        "seed", "arm", "N", "内%", "外%", "g22内", "g22外",
        "能内", "能外", "健内", "健外"))
    for s in (189, 190, 191, 192):
        for arm in ("off", "on"):
            a = cache.get((s, arm))
            if a is None:
                continue
            ins = a["inside"]
            print("%-5d %-4s %7d %7.2f%% %7.2f%% %10.4f %10.4f %9.1f %9.1f %9.3f %9.3f" % (
                s, arm, len(ins), ins.mean() * 100, (1 - ins.mean()) * 100,
                a["g"][ins].mean(), a["g"][~ins].mean(),
                a["en"][ins].mean(), a["en"][~ins].mean(),
                np.median(a["hp"][ins]), np.median(a["hp"][~ins])))

    print()
    print("=" * 100)
    print("B. 按「离最近斑块的栅格距离」分箱：g22 / 能量 / 健康 （判「出走=死路」）")
    print("=" * 100)
    for s in (189, 191, 192):
        for arm in ("off", "on"):
            a = cache.get((s, arm))
            if a is None:
                continue
            d = a["dist"]
            print("--- seed %d · %s 臂 （N=%d）---" % (s, arm, len(d)))
            print("  %-7s %8s %8s %11s %11s %11s" % (
                "距(格)", "个体数", "占比", "g22均值", "能量中位", "健康中位"))
            for i in range(len(LABELS)):
                m = (d >= BINS[i]) & (d < BINS[i + 1])
                if m.sum() == 0:
                    print("  %-7s %8d %7.2f%% %11s %11s %11s" % (LABELS[i], 0, 0.0, "-", "-", "-"))
                    continue
                print("  %-7s %8d %7.2f%% %11.4f %11.1f %11.3f" % (
                    LABELS[i], int(m.sum()), m.mean() * 100,
                    a["g"][m].mean(), np.median(a["en"][m]), np.median(a["hp"][m])))
            # g22 直方图（内 vs 外）10 桶
            e_in = np.histogram(a["g"][a["inside"]], bins=10, range=(0, 1))[0]
            e_out = np.histogram(a["g"][~a["inside"]], bins=10, range=(0, 1))[0]
            e_in = e_in / max(e_in.sum(), 1)
            e_out = e_out / max(e_out.sum(), 1)
            print("  g22 直方图 内: %s" % np.round(e_in, 3).tolist())
            print("  g22 直方图 外: %s" % np.round(e_out, 3).tolist())
            print()

    print("=" * 100)
    print("C. 斑块几何 × 世界（检验「初始投放决定出走型存活」）")
    print("=" * 100)
    disp = {189: 9.88, 190: 10.00, 191: 6.68, 192: 9.91}
    print("%-5s %9s %9s %11s %11s %11s %10s" % (
        "seed", "斑块格数", "斑块块数", "质心NN中位", "质心NN_p10", "质心NN_p90", "出走型%"))
    for s in (189, 190, 191, 192):
        a = cache[(s, "on")]
        pm = a["pm"]
        # 连通块计数（BFS on patch cells）
        seen = np.zeros(TOTAL, dtype=bool)
        nblob = 0
        cen = []
        idx = np.where(pm)[0]
        for p0 in idx.tolist():
            if seen[p0]:
                continue
            nblob += 1
            stack = [p0]
            seen[p0] = True
            acc_r = 0.0
            acc_c = 0.0
            n = 0
            while stack:
                p = stack.pop()
                r, c = divmod(p, COLS)
                acc_r += r
                acc_c += c
                n += 1
                for q in (p - COLS, p + COLS,
                          r * COLS + (c - 1) % COLS,
                          r * COLS + (c + 1) % COLS):
                    if 0 <= q < TOTAL and pm[q] and not seen[q]:
                        seen[q] = True
                        stack.append(q)
            cen.append((acc_r / n, acc_c / n))
        cen = np.array(cen)
        # 质心两两最近邻距离（大圆，格）
        lat = (cen[:, 0] + 0.5) * np.pi / ROWS - np.pi / 2
        lon = (cen[:, 1] + 0.5) * 2 * np.pi / COLS
        xyz = np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], 1)
        nn = np.full(len(xyz), np.nan)
        for i in range(len(xyz)):
            dd = np.linalg.norm(xyz - xyz[i], axis=1)
            dd[i] = np.inf
            j = int(np.argmin(dd))
            nn[i] = 2 * np.arcsin(min(dd[j] / 2, 1.0)) * ROWS / np.pi
        print("%-5d %9d %9d %11.2f %11.2f %11.2f %9.2f%%" % (
            s, int(pm.sum()), nblob, np.median(nn),
            np.percentile(nn, 10), np.percentile(nn, 90), disp[s]))


if __name__ == "__main__":
    main()
