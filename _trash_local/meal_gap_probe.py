"""只读算术：估算 13.5 参数下，个体**站在富斑块上**相邻两次"咬到"的间隔 tick 数。

用途：检验 ARS 设计稿的退出条件 `ticks_since_feed > window` 在富斑上是否可能触发。
（不修改任何文件，纯计算。）
"""
import sys

sys.stdout.reconfigure(encoding="utf-8")

# —— 13.5 preset 实测参数 ——
STOMACH_CAP_MASS = 25.0      # --stomach-cap-mass=25
EAT_THRESHOLD_FRAC = 0.6     # --eat-threshold-frac=0.6
EAT_AMOUNT = 0.9             # 13.4 起 eat 0.5 -> 0.9
BASE_METABOLISM = 0.6        # config.py:152


def meal_gap(g1: float, g4: float, g5: float, activity: float,
             food_always_rich: bool = True, ticks: int = 400) -> list[int]:
    """返回相邻两次咬到的间隔序列（tick）。"""
    cap = STOMACH_CAP_MASS * (0.5 + g5 * 1.5)
    gate = cap * EAT_THRESHOLD_FRAC
    eat_mult = 0.5 + g4 * 1.0
    metab_mult = 0.5 + g1 * 1.5

    stomach = 0.0
    since = 0
    gaps: list[int] = []
    for _ in range(ticks):
        # 步骤 2：胃 -> 能量（消化）
        digest_rate = BASE_METABOLISM * metab_mult * activity
        stomach -= min(stomach, digest_rate)
        # 步骤 4：进食（只有 stomach < gate 才吃；富斑 => 想吃多少有多少）
        ate = False
        if stomach < gate:
            want = min(EAT_AMOUNT * eat_mult, cap - stomach)
            got = want if food_always_rich else 0.0   # 富斑：want 全额满足
            stomach += got
            if got > 1e-12:
                ate = True
        if ate:
            gaps.append(since)
            since = 0
        else:
            since += 1
    return gaps[1:]   # 丢掉第一次（含起步爬坡）


print("=== 富斑块上，相邻两次「咬到」的间隔（tick）===")
print(f"{'g1':>5}{'g4':>5}{'g5':>5}{'活动度':>8}{'间隔中位数':>12}{'间隔均值':>10}{'最大':>8}")
rows = []
for g1 in (0.0, 0.5, 1.0):
    for g4 in (0.25, 0.5, 0.75):
        for activity in (0.5, 1.0):
            g = meal_gap(g1, g4, 0.5, activity)
            if not g:
                continue
            g_sorted = sorted(g)
            med = g_sorted[len(g_sorted) // 2]
            rows.append((med, g1, g4, activity, sum(g) / len(g), max(g)))
for med, g1, g4, activity, mean, mx in sorted(rows):
    print(f"{g1:>5.2f}{g4:>5.2f}{0.5:>5.2f}{activity:>8.1f}{med:>12d}{mean:>10.1f}{mx:>8d}")

print()
print("=== 结论判读 ===")
allgaps: list[int] = []
for g1 in (0.0, 0.5, 1.0):
    for g4 in (0.25, 0.5, 0.75):
        for activity in (0.5, 1.0):
            allgaps += meal_gap(g1, g4, 0.5, activity)
if allgaps:
    s = sorted(allgaps)
    print(f"  全部情形的间隔范围：{min(s)} ~ {max(s)} tick；中位 {s[len(s)//2]} tick")
    print("  ⇒ 若 ARS 的 window 大于这个范围，ticks_since_feed **永远超不过 window**")
    print("    ⇒ 退出条件在富斑上**根本不会触发** = 静默 no-op（B2 缺陷形态）")
