"""H2 位域独立性检验（**权威复现件**；R149-6⑤ 收入 `tools/`）。

**来源与沿革**（保留作者痕迹，勿抹）：
  - 原为 `[内评]` 的**独立第三方实现**（`_archive/2026-10-10-退役团队-归档/_eval/l1l2_bitfield_check.py`，2026-09-21 23:02），
    起因：老工 §4.3 的检验（χ²=9775.6 / df 9801）在其 commit 里**没有对应实现** ⇒ 该 `[实测]`
    **无复现路径**（`tools/l1l2_geom_probe.py` 当时只 +了 FAR_CAP 全扫）。
  - **R149-6⑤ 裁定收编到 `tools/`**（本文件即权威件）；`tools/l1l2_geom_probe.py --bitfield`
    调用本文件 ⇒ 任一大改须**两边同时**跑一遍（互为对照）。

**约定（与设计稿 §4.2/§4.3 对齐）**：
  - `rand_choice = rng.integers(0, 1_000_000, size=Nm)`（`sphere_engine.py`：`rand_choice = self.rng.integers`）
  - `lo = rc % 100`（冲刺闸）｜`hi = rc // 100`（选格）⇒ `10^6 = 100 × 10^4` 双射
  - 列联表 = `lo × (hi // 100)` ⇒ 100×100，df = 99×99 = 9801
  - 禁止写法对照 = 同一 `u` 比两个阈值（`u<0.5` vs `u<0.3`）
  - 🔴 **可达 `far_len` 只有 {12, 16}**（strict 2 圈规模；其余 >32 被 `FAR_CAP` 排除）⇒
    模偏差只有这两档：`16 ⇒ 0`、`12 ⇒ +0.08%`（**不是**设计稿 v2 误写的 21–24 那组）

用法（ASCII 输出，避免 GBK 控制台问题）：
    python.exe -X utf8 tools/l1l2_bitfield_check.py
    python.exe -X utf8 tools/l1l2_bitfield_check.py --geom   # 附几何边界复核（含 FAR_CAP 不变区间）
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

SEED = 20260921          # 写死：换 seed 会改 χ²（这是抽样的性质检验，不是构造性证明）
N = 10 ** 6


def bitfield() -> int:
    rng = np.random.default_rng(SEED)
    rc = rng.integers(0, 1_000_000, size=N)
    lo = rc % 100
    hi = rc // 100
    hib = hi // 100

    tab = np.zeros((100, 100), dtype=np.int64)
    np.add.at(tab, (lo, hib), 1)
    exp = N / 10000.0
    chi2 = float(((tab - exp) ** 2 / exp).sum())
    print("[bitfield] n=%d seed=%d" % (N, SEED))
    print("  bijection  : 10^6 = 100 x 10^4 -> lo in [0,100), hi in [0,10000)  OK")
    print("  chi2       : %.1f   (df=9801, expected 9801; cell min=%d max=%d)"
          % (chi2, tab.min(), tab.max()))
    print("  corr(lo,hi): %+.5f   corr(lo,hi//100): %+.5f"
          % (np.corrcoef(lo, hi)[0, 1], np.corrcoef(lo, hib)[0, 1]))
    print("  verdict    : %s" % ("no evidence of dependence" if abs(chi2 - 9801) < 4 * np.sqrt(2 * 9801) else "CHECK"))

    print("  dash rate  : (lo < int(100*mob))")
    for mob in (0.25, 0.50, 0.90):
        print("    mob=%.2f -> %.4f  (theory int(100*mob)/100 = %.4f)"
              % (mob, float((lo < int(100 * mob)).mean()), int(100 * mob) / 100.0))

    # 可达 far_len：strict 2 圈规模（几何全扫）= {12, 16}（其余 >32 被 FAR_CAP 排除）
    print("  modulo bias: hi %% far_len  (reachable far_len = {12, 16})")
    for far in (12, 16):
        c = np.bincount(hi % far, minlength=far)
        # 确定性上界：10000 = far*q + r ⇒ 前 r 个残差各多 1 ⇒ 相对均值的偏差
        q, r = divmod(10000, far)
        det = ((q + 1) / (10000 / far) - 1) * 100 if r else 0.0
        print("    far=%2d counts %d..%d  observed dev %+.3f%%  |  deterministic bias %+.3f%%"
              "  (10000 %% %d = %d)"
              % (far, c.min(), c.max(), (c.max() / c.mean() - 1) * 100, det, far, r))

    u = rng.random(N)
    corr = float(np.corrcoef((u < 0.5).astype(float), (u < 0.3).astype(float))[0, 1])
    print("  FORBIDDEN  : same u vs two thresholds corr = %+.4f  (must be ~1 => not independent)" % corr)
    return 0


def geom() -> int:
    sys.path.insert(0, ".")
    from world.sphere_world import SphereWorld

    w = SphereWorld(60, 120)
    adj = [set(int(x) for x in w.neighbors(c)) for c in range(w.n_cells)]

    def ring2(c: int) -> set[int]:
        out: set[int] = set()
        for d in adj[c]:
            out |= adj[d]
        out.discard(c)
        out -= adj[c]
        return out

    r2 = np.array([len(ring2(c)) for c in range(w.n_cells)], dtype=np.int64)
    sizes = {int(v): int((r2 == v).sum()) for v in np.unique(r2)}
    print("[geom] strict-2-ring size histogram: %s" % sorted(sizes.items()))
    base = set(np.nonzero(r2 > 16)[0].tolist())
    same = [c for c in range(16, 125) if set(np.nonzero(r2 > c)[0].tolist()) == base]
    print("  caps giving the SAME exclusion set: %d..%d  (size %d = %.2f%%)"
          % (min(same), max(same), len(base), len(base) / w.n_cells * 100))
    print("  rows affected: %s" % sorted(set((np.nonzero(r2 > 16)[0] // w.cols).tolist())))
    print("  reachable far_len (not excluded): %s"
          % sorted(int(v) for v in np.unique(r2) if v <= 32))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--geom", action="store_true", help="附：strict 2 圈与 FAR_CAP 边界复核")
    a = ap.parse_args()
    rc = bitfield()
    if a.geom:
        rc |= geom()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
