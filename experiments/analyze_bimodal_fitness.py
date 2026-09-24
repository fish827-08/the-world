#!/usr/bin/env python3.12
"""
适应度地形分析 v2 —— 解析判定 g16 进化分支（双峰）的必要条件。

W(g) = (1-g)^k * F                          [素食：凸权衡]
     + 1[g>gate] * [ p(g)*sigma(g)*E_prey*r - p(g)*c ]
  p(g)     = g * coef * hunger
  sigma(g) = clip(E/(E+Ep) * (0.5 + g*gain), floor, ceil)

判定：适应度地形中是否存在"内部极小"（谷）——这是进化分支
（disruptive selection → 双峰）的必要条件。

用解析导数 + 高精度网格 + 忽略数值噪声（阈值法）来判定。
"""
import numpy as np
import sys


# 🔴 R98 纪律（F-R14/F-R15 同族）：中文 Windows 默认 **GBK** 控制台下，print 里的
#    emoji / 箭头（`⇒` `✅` `❌` 等）会抛 UnicodeEncodeError ⇒ 脚本 **rc=1 假失败**，
#    把批次退出码搅坏（数据其实无损）。入口强制 UTF-8，`errors="replace"` 兜底，
#    **绝不因编码丢结果**。守卫测试：`tests/test_r98_nonascii_print.py`。
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

COEF = 0.2      # attack_prob_coef
GATE = 0.3      # attack_gene_gate
GAIN = 0.5      # success_gene_gain
FLOOR, CEIL = 0.1, 0.9
R = 0.4         # transfer_ratio
C = 0.1         # attack_cost


def sigma(g, e_self, e_prey):
    ratio = e_self / max(e_self + e_prey, 1e-9)
    return np.clip(ratio * (0.5 + g * GAIN), FLOOR, CEIL)


def W(g, k, F, e_self, e_prey, hunger, cost_scale=1.0):
    g = np.asarray(g, dtype=float)
    forage = (1.0 - g) ** k * F
    p = g * COEF * hunger
    s = sigma(g, e_self, e_prey)
    gain_side = p * s * e_prey * R
    cost_side = p * C * cost_scale
    mask = (g > GATE).astype(float)
    return forage + mask * (gain_side - cost_side)


def find_valley(k, F=1.0, e_self=150.0, e_prey=150.0, hunger=0.5,
                cost_scale=1.0, n=200001, tol=1e-6):
    """在 g∈(gate,1) 上找内部极小；用平滑序列 + 显著性阈值滤噪声。"""
    g = np.linspace(0.0, 1.0, n)
    w = W(g, k, F, e_self, e_prey, hunger, cost_scale)
    dw = np.gradient(w, g)

    # 只在 g>GATE 段找驻点（gate 以下无捕食项）
    idx = np.where(g > GATE)[0]
    gw, dww = g[idx], dw[idx]

    # 驻点：导数变号
    sign = np.sign(dww)
    turns = np.where(np.diff(sign) != 0)[0]
    stations = []
    for i in turns:
        if i <= 0 or i >= len(gw) - 1:
            continue
        gi = gw[i]
        wi = w[idx[i]]
        # 用两侧斜率判断极值类型（更稳）
        slope_l = (w[idx[i]] - w[idx[i - 20 if i >= 20 else 0]])
        slope_r = (w[idx[min(i + 20, len(idx) - 1)]] - w[idx[i]])
        if slope_l < -tol and slope_r > tol:
            stations.append((float(gi), float(wi), "极小(谷)"))
        elif slope_l > tol and slope_r < -tol:
            stations.append((float(gi), float(wi), "极大(峰)"))
    return g, w, stations


def report(k, **kw):
    label = kw.pop("label", "")
    g, w, st = find_valley(k, **kw)
    print(f"\n{'─'*70}")
    print(f"k={k}  " + "  ".join(f"{a}={b}" for a, b in kw.items()) + f"  {label}")
    print(f"{'─'*70}")
    # 采样 W 值
    print(f"  W(g):  g=0 → {w[0]:.4f} | g=.3 → {w[int(.3*len(g))]:.4f} | "
          f"g=.5 → {w[int(.5*len(g))]:.4f} | g=.7 → {w[int(.7*len(g))]:.4f} | "
          f"g=1 → {w[-1]:.4f}")
    valleys = [s for s in st if s[2] == "极小(谷)" and 0.05 < s[0] < 0.95]
    peaks = [s for s in st if s[2] == "极大(峰)" and 0.05 < s[0] < 0.95]
    if valleys:
        for gv, wv, _ in valleys:
            print(f"  🔻 内部极小: g={gv:.4f}  W={wv:.4f}")
    if peaks:
        for gp, wp, _ in peaks:
            print(f"  🔺 内部极大: g={gp:.4f}  W={wp:.4f}")
    if valleys:
        print(f"  ⇒ ✅ **存在内部谷** ⇒ 分支(双峰)必要条件满足")
    else:
        print(f"  ⇒ ❌ 无内部谷 ⇒ 单向收敛（边界最优）")
    return bool(valleys)


if __name__ == "__main__":
    print("=" * 70)
    print("A) 基线：只调 k（我昨晚做过的）")
    print("=" * 70)
    for k in (0.0, 1.0, 2.0, 3.0, 5.0):
        report(k, label="基线")

    print("\n\n" + "=" * 70)
    print("B) 恢复剧增效应：能量不对称 e_self >> e_prey（Step1 假设）")
    print("=" * 70)
    for es, ep in ((150, 150), (300, 150), (900, 150)):
        report(2.0, e_self=es, e_prey=ep, label=f"不对称{es}/{ep}")

    print("\n\n" + "=" * 70)
    print("C) 提高攻击成本（Step2 假设：造中间态劣势）")
    print("=" * 70)
    for cs in (1, 5, 20, 50, 100, 200):
        report(2.0, cost_scale=cs, label=f"cost×{cs}")

    print("\n\n" + "=" * 70)
    print("D) 组合：不对称 + 高成本")
    print("=" * 70)
    for es, cs in ((300, 20), (300, 50), (900, 50), (900, 100)):
        report(2.0, e_self=es, e_prey=150, cost_scale=cs,
               label=f"{es}/150 cost×{cs}")
