"""S2 判据② 信噪比探针 —— 用**历史 40k 数据**判断"能否降低 tick"。

数据源：`the-world-data/results/s2_depletion/s2_depletion.csv`（旧档 bg_low 0.4/0.05，
seed 42/43/44 × arm off/on × 40 000 tick × sample 250 ⇒ 160 行/run）。

要回答的三个问题：
  1. 判据 `patch_sat_init_var`（末段 3 行均值，代码 `_tail_mean(n=3)`）的 **on>off 方向**
     在不同**截止 tick** 下是否稳定？（稳 ⇒ 可以降 tick；不稳 ⇒ 降 tick = 抽签）
  2. 判据量的**信噪比**：on/off 均值差 ÷ 组内标准差。>1 才有判别力。
  3. 末段窗口长度（3 / 10 / 20 / 40 行）对结论的影响。

跑法（项目根目录）：
    .venv/Scripts/python.exe -u _trash_local/_judge2_snr_probe.py
"""
import csv
import collections
import statistics

P = "the-world-data/results/s2_depletion/s2_depletion.csv"

rows = list(csv.DictReader(open(P, encoding="utf-8")))
by = collections.defaultdict(dict)  # (seed, arm) -> {tick: value}
for x in rows:
    by[(x["seed"], x["arm"])][int(x["tick"])] = float(x["patch_sat_init_var"])
seeds = sorted({x["seed"] for x in rows})


def tail_mean(seed, arm, upto, n=3):
    """末段 n 行均值（口径同代码 `_tail_mean`），只取 tick<=upto 的行。"""
    d = by[(seed, arm)]
    ks = sorted(k for k in d if k <= upto)
    if not ks:
        return None
    return sum(d[k] for k in ks[-n:]) / len(ks[-n:])


print("=" * 78)
print("① 末段 3 行口径下，on>off 方向随【截止 tick】的变化")
print("=" * 78)
CUTS = (5000, 10000, 15000, 20000, 30000, 40000)
print("%-6s %s" % ("seed", " ".join("%11s" % ("t<=%dk" % (t // 1000)) for t in CUTS)))
for s in seeds:
    cells = []
    for t in CUTS:
        o, n = tail_mean(s, "off", t), tail_mean(s, "on", t)
        cells.append("%11s" % ("%s%.4f" % ("+" if n > o else "-", abs(n - o))))
    print("%-6s %s" % (s, " ".join(cells)))

print()
print("=" * 78)
print("② 信噪比：均值差 ÷ 组内标准差（>1 才有判别力）")
print("=" * 78)
for s in seeds:
    o = [v for _, v in sorted(by[(s, "off")].items())]
    n = [v for _, v in sorted(by[(s, "on")].items())]
    sd = max(statistics.pstdev(o), 1e-9)
    print("  seed %s: off %.5f±%.5f | on %.5f±%.5f | 差 %+.5f = %.2f×组内SD"
          % (s, statistics.mean(o), statistics.pstdev(o),
             statistics.mean(n), statistics.pstdev(n),
             statistics.mean(n) - statistics.mean(o),
             abs(statistics.mean(n) - statistics.mean(o)) / sd))

print()
print("=" * 78)
print("③ 同一 t=40k，末段窗口长度对结论的影响")
print("=" * 78)
for n_ in (3, 10, 20, 40):
    res = []
    for s in seeds:
        o, n = tail_mean(s, "off", 40000, n_), tail_mean(s, "on", 40000, n_)
        res.append(n > o)
    print("  末段 %-3d 行（= 最后 %5d tick）：on>off = %d/3  %s"
          % (n_, n_ * 250, sum(res), ["+" if x else "-" for x in res]))

print()
print("=" * 78)
print("④ 各截止 tick 下的『on>off 票数』汇总（决定三态判定）")
print("=" * 78)
for t in CUTS:
    votes = sum(1 for s in seeds if tail_mean(s, "on", t) > tail_mean(s, "off", t))
    print("  截止 t=%-6d ⇒ on>off %d/3" % (t, votes))
