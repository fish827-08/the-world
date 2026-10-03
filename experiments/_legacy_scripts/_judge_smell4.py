"""R335 气味场四通道批（云机器）判读 —— 预注册 A/B 双主门。

判据（云启预注册稿 §三，跑前锁定）：
  A（主门·food 引导）：`l1_visited_patch_frac` @t=2000 的**逐 seed 配对差（on−off）> 0** 的 seed ≥ 14/24
  B（主门·risk 回避）：捕食死亡率 `d_pred/pop` @t=2000 的配对差 < 0 的 seed ≥ 14/24
  A/B 各自独立判阴。
"""
import csv
import glob
import os

OFF = "_rerun_logs/smell4/off"
ON = "_rerun_logs/smell4/on"


def load(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def last(path):
    r = load(path)
    return r[-1] if r else None


def f(x, k, d=float("nan")):
    try:
        return float(x[k])
    except Exception:
        return d


seeds = sorted({os.path.basename(p).split("_")[1]
                for p in glob.glob(os.path.join(OFF, "*.csv"))}, key=int)
print("配对数（seed 数）= %d" % len(seeds))

rows = []
for s in seeds:
    fo = os.path.join(OFF, "s2g2_%s_on.csv" % s)
    fn = os.path.join(ON, "s2g2_%s_on.csv" % s)
    if not (os.path.exists(fo) and os.path.exists(fn)):
        print("  ⚠ seed %s 缺文件" % s)
        continue
    o, n = last(fo), last(fn)
    if o is None or n is None:
        continue
    # A: l1_visited_patch_frac
    a_o, a_n = f(o, "l1_visited_patch_frac"), f(n, "l1_visited_patch_frac")
    # B: 捕食死亡率 = d_pred / pop
    po, pn = f(o, "pop"), f(n, "pop")
    b_o = f(o, "d_pred") / po if po else float("nan")
    b_n = f(n, "d_pred") / pn if pn else float("nan")
    rows.append(dict(seed=s, tick=o["tick"],
                     a_o=a_o, a_n=a_n, dA=a_n - a_o,
                     b_o=b_o, b_n=b_n, dB=b_n - b_o,
                     K_o=po, K_n=pn))

print()
print("=== 逐 seed 配对表（t=%s）===" % rows[0]["tick"])
print("%-6s %9s %9s %10s %6s | %10s %10s %12s %6s" %
      ("seed", "l1_off", "l1_on", "ΔA", "A+", "predR_off", "predR_on", "ΔB", "B-"))
for r in rows:
    print("%-6s %9.4f %9.4f %+10.5f %6s | %10.6f %10.6f %+12.7f %6s" %
          (r["seed"], r["a_o"], r["a_n"], r["dA"], "✓" if r["dA"] > 0 else "×",
           r["b_o"], r["b_n"], r["dB"], "✓" if r["dB"] < 0 else "×"))

nA = sum(1 for r in rows if r["dA"] > 0)
nB = sum(1 for r in rows if r["dB"] < 0)
N = len(rows)
print()
print("=== 主门判定（门槛 ≥14/24）===")
print("  A（food 引导：l1 配对差 >0）: %d/%d  ⇒ %s" %
      (nA, N, "**通过**" if nA >= 14 else "**未通过**"))
print("  B（risk 回避：捕食率配对差 <0）: %d/%d  ⇒ %s" %
      (nB, N, "**通过**" if nB >= 14 else "**未通过**"))

import statistics as st
print()
print("=== 效应量（中位）===")
print("  ΔA 中位 = %+.5f（l1 已访斑块占比，正=气味让个体访问更多斑块）" %
      st.median([r["dA"] for r in rows]))
print("  ΔB 中位 = %+.7f（捕食死亡率，负=气味降低被捕食）" %
      st.median([r["dB"] for r in rows]))
print("  K 比值中位 = %.3f（on/off）" %
      st.median([r["K_n"] / r["K_o"] for r in rows if r["K_o"]]))
