"""L1/L2 设计前置几何探针（确定性，零 RNG、零机时；2026-09-21）。

**目的**：在写实现之前，先回答两个"构造性前提"问题（C8 精神）：

  Q1（逃脱） 猎物先动（step 5）⇒ 捕食者后动（step 5.5，射程 1 格）。
             把猎物候选集从 r=1 扩到 r≤2，**真能让猎物脱离捕食者邻域吗？**
  Q2（追击） 猎物先跑到 q 之后，捕食者的候选集里**是否存在一格与 q 相邻**
             （相邻才吃得到）？r=1 vs r≤2 的差别多大？

**口径**（务必与引擎一致，见 `simulation/sphere_engine.py`）：
  - 移动**允许同格重叠**（`self._flat[mi] = targets` 不查占用）
  - 捕食**同格不可互吃** ⇒ 与捕食者距离 **0 反而安全**，距离 **1 = 有风险**
  - 逃逸/追击距离用**球面图上最短步数**（= 跳数），不是欧氏距离

用法：
    python.exe tools/l1l2_geom_probe.py            # 默认 60×120
    python.exe tools/l1l2_geom_probe.py --step 37  # 采样步长
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict

sys.path.insert(0, ".")

from world.sphere_world import SphereWorld  # noqa: E402


def _adj_sets(world: SphereWorld) -> list[set[int]]:
    """每个格的 1 圈邻居集合（极点格 = 相邻纬度带整行 ⇒ 很大）。"""
    return [set(int(x) for x in world.neighbors(c)) for c in range(world.n_cells)]


def _ring2(adj: list[set[int]], c: int) -> set[int]:
    """严格 2 圈（不含 c 自身、不含 1 圈）。"""
    out: set[int] = set()
    for d in adj[c]:
        out |= adj[d]
    out.discard(c)
    out -= adj[c]
    return out


def _lat_band(world: SphereWorld, c: int) -> str:
    r = int(world.flat_to_rc([c])[0][0])
    mid = world.rows // 2
    d = abs(r - mid)
    if d >= world.rows // 2 - 1:
        return "polar"
    if d >= world.rows // 4:
        return "mid"
    return "equator"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=60)
    ap.add_argument("--cols", type=int, default=120)
    ap.add_argument("--step", type=int, default=37, help="扫描捕食者位置的步长")
    args = ap.parse_args()

    world = SphereWorld(args.rows, args.cols)
    adj = _adj_sets(world)
    ring2 = {}
    poles = [0, world.n_cells - 1]

    # —— 候选集规模统计 ——
    sizes: dict[str, list[int]] = defaultdict(list)
    sample = sorted(set(list(range(0, world.n_cells, args.step)) + poles))
    for s in sample:
        sizes[_lat_band(world, s)].append(len(adj[s]) + len(_ring2(adj, s)))
    print("=== 候选集规模（r<=2 时的候选格数；r=1 时 = 1 圈邻居数）===")
    for band in ("equator", "mid", "polar"):
        v = sizes.get(band) or []
        if not v:
            continue
        r1 = [len(adj[s]) for s in sample if _lat_band(world, s) == band]
        print(f"  {band:<8} n={len(v):>3}  r=1 中位 {sorted(r1)[len(r1)//2]:>4} "
              f"| r<=2 中位 {sorted(v)[len(v)//2]:>4}  范围 {min(v)}–{max(v)}")

    # —— Q1 逃脱 / Q2 追击 ——
    q1 = defaultdict(lambda: [0, 0, 0, 0])   # band -> [r1_risk_n, r1_n, r2_risk_n, r2_n]
    # band -> [r1_reach_n, r1_n, r2_reach_n, r2_n, r1_cnt_sum, r2_cnt_sum]
    q2 = defaultdict(lambda: [0, 0, 0, 0, 0, 0])
    same_cell_gain = defaultdict(lambda: [0, 0])
    pair_n = defaultdict(lambda: [0, 0])     # band -> [(s,p) 对数, 候选对总数(含 r<=2)]

    for s in sample:
        band = _lat_band(world, s)
        adj_s = adj[s]
        cand1_s = adj_s
        cand2_s = adj_s | _ring2(adj, s)
        for p in adj_s:                       # 猎物在捕食者邻格（有风险）
            c1 = adj[p]
            c2 = c1 | _ring2(adj, p)
            pair_n[band][0] += 1
            pair_n[band][1] += len(c2)
            for cand, tag in ((c1, 0), (c2, 2)):
                n = len(cand)
                risk = sum(1 for c in cand if c in adj_s)          # d==1 ⇒ 仍会被吃
                same = sum(1 for c in cand if c == s)              # d==0 ⇒ 同格，安全
                if tag == 0:
                    q1[band][0] += risk
                    q1[band][1] += n
                    same_cell_gain[band][0] += same
                    same_cell_gain[band][1] += n
                else:
                    q1[band][2] += risk
                    q1[band][3] += n
            # Q2：猎物跑到 q 后，捕食者能否落到 q 的邻格（相邻才吃得到）
            for q in c2:
                n1 = len(cand1_s & adj[q])
                n2 = len(cand2_s & adj[q])
                q2[band][0] += 1 if n1 else 0
                q2[band][1] += 1
                q2[band][2] += 1 if n2 else 0
                q2[band][3] += 1
                q2[band][4] += n1
                q2[band][5] += n2

    print()
    print("=== Q1 逃脱：猎物移动后『仍处捕食者邻格（可被吃）』的比例 ===")
    print(f"  {'纬度带':<8}{'r=1':>10}{'r<=2':>10}{'绝对降幅':>10}{'相对降幅':>10}")
    tot = [0, 0, 0, 0]
    for band in ("equator", "mid", "polar"):
        v = q1.get(band)
        if not v:
            continue
        for i in range(4):
            tot[i] += v[i]
        f1, f2 = v[0] / max(1, v[1]), v[2] / max(1, v[3])
        print(f"  {band:<8}{f1*100:>9.2f}%{f2*100:>9.2f}%"
              f"{(f1-f2)*100:>9.2f}pp{(f1-f2)/max(1e-9, f1)*100:>9.1f}%")
    f1, f2 = tot[0] / max(1, tot[1]), tot[2] / max(1, tot[3])
    print(f"  {'合计':<8}{f1*100:>9.2f}%{f2*100:>9.2f}%"
          f"{(f1-f2)*100:>9.2f}pp{(f1-f2)/max(1e-9, f1)*100:>9.1f}%")

    print()
    print("=== Q1b 其中『跳进捕食者同格』这条既有的免费逃生路 ===")
    for band in ("equator", "mid", "polar"):
        v = same_cell_gain.get(band)
        if not v or not v[1]:
            continue
        print(f"  {band:<8} r=1 候选里同格占比 {v[0]/v[1]*100:>6.2f}%"
              "   ← 这条**旧构造已有**（同格不可互吃）")

    print()
    print("=== Q2 追击：捕食者候选集中『存在格与猎物新位置相邻』的比例（可吃到）===")
    print(f"  {'纬度带':<8}{'r=1':>10}{'r<=2':>10}{'绝对提升':>10}{'相对提升':>10}")
    tot = [0, 0, 0, 0]
    for band in ("equator", "mid", "polar"):
        v = q2.get(band)
        if not v:
            continue
        for i in range(4):
            tot[i] += v[i]
        f1, f2 = v[0] / max(1, v[1]), v[2] / max(1, v[3])
        print(f"  {band:<8}{f1*100:>9.2f}%{f2*100:>9.2f}%"
              f"{(f2-f1)*100:>9.2f}pp{(f2-f1)/max(1e-9, f1)*100:>9.1f}%")
    f1, f2 = tot[0] / max(1, tot[1]), tot[2] / max(1, tot[3])
    print(f"  {'合计':<8}{f1*100:>9.2f}%{f2*100:>9.2f}%"
          f"{(f2-f1)*100:>9.2f}pp{(f2-f1)/max(1e-9, f1)*100:>9.1f}%")

    print()
    print("=== Q1c 新增的『2 圈候选』自己有多少是有风险的（d=2 那 16 格）===")
    print("     —— 这是判定『盲扩候选集能否帮逃脱』的直接证据")
    print(f"  {'纬度带':<8}{'新增格数':>10}{'其中 d==1（仍被吃）':>22}{'占比':>10}")
    for band in ("equator", "mid", "polar"):
        v = q1.get(band)
        if not v:
            continue
        n_new = v[3] - v[1]
        risk_new = v[2] - v[0]
        if n_new <= 0:
            continue
        print(f"  {band:<8}{n_new:>10}{risk_new:>22}{risk_new/n_new*100:>9.2f}%")

    print()
    print("=== Q1d 恐避型猎物真正需要的东西：候选集里『安全格』的**个数**（E[#]）===")
    print(f"  {'纬度带':<8}{'r=1 安全格':>12}{'r<=2 安全格':>12}{'净增':>8}")
    for band in ("equator", "mid", "polar"):
        v = q1.get(band)
        pn = pair_n.get(band)
        if not v or not pn or not pn[0]:
            continue
        safe1 = (v[1] - v[0]) / pn[0]
        safe2 = (v[3] - v[2]) / pn[0]
        print(f"  {band:<8}{safe1:>12.2f}{safe2:>12.2f}{safe2-safe1:>+8.2f}")

    print()
    print("=== Q2b 追击：候选集中『与猎物新位置相邻』的**格数**（均值）===")
    print(f"  {'纬度带':<8}{'r=1 可用格':>12}{'r<=2 可用格':>12}{'净增':>8}")
    for band in ("equator", "mid", "polar"):
        v = q2.get(band)
        if not v or not v[1]:
            continue
        m1 = v[4] / v[1]
        m2 = v[5] / v[3]
        print(f"  {band:<8}{m1:>12.2f}{m2:>12.2f}{m2-m1:>+8.2f}")

    print()
    print("=== 结论速览 ===")
    print("  · Q1 降幅 = r<=2 的**净收益上界**（猎物并不总选安全格 ⇒ 实测更小）")
    print("  · Q2 提升 = 捕食者**能追到**的概率上界（还要过 seek 排序与能量闸）")
    print("  · 两者都 ≫0 才说明『机动性是有效自由度』；若 Q1 已≈0 或 Q2 已≈100%，")
    print("    则这一层的边际作用有限 ⇒ 设计重心应转移（记入设计稿取舍）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
