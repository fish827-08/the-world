"""只读算术：13.5 参数下，个体在**不进食**的情况下能走多远（可达半径）。

用途：回答「把移动降到 0.25 格/tick，满能力最多能走多少格」。
不修改任何文件，纯计算。
"""
import math
import sys

sys.stdout.reconfigure(encoding="utf-8")

# —— 13.5 B 臂实测参数 ——
MAX_ENERGY = 600.0
INITIAL_ENERGY = 300.0
STARVE_FRAC = 0.30          # 能量 < 0.30×max ⇒ 饿死
BASE_METABOLISM = 0.6
MOVE_COST = 0.4             # config 默认

STARVE_AT = STARVE_FRAC * MAX_ENERGY
BUFFER = INITIAL_ENERGY - STARVE_AT
print(f"出生能量 {INITIAL_ENERGY}；饿死线 {STARVE_AT} ⇒ **可动用预算 {BUFFER} 能量**\n")


def reach(g1: float, g6: float, activity: float, speed: float, move_prob: float = 1.0):
    """返回 (能活多少 tick, 直线可达格数, 随机游走有效半径格数)"""
    metab = BASE_METABOLISM * (0.5 + g1 * 1.5) * activity
    per_move = MOVE_COST * (0.5 + g6)
    per_tick = metab + per_move * move_prob
    ticks = BUFFER / per_tick
    straight = ticks * speed                 # 直线：位移 ~ 步数×步长
    random_r = math.sqrt(ticks) * speed      # 布朗：净位移 ~ √步数×步长
    return ticks, straight, random_r


print("=== 可达半径 vs 移动速度（g1=0.5 中代谢、g6=0.5 中移动能耗、活动度=1.0）===")
print(f"{'速度(格/tick)':>14}{'能活tick':>10}{'直线可达':>10}{'随机走有效半径':>16}{'最近斑块 7.7 格够不够':>22}")
for sp in (1.0, 0.5, 0.25, 0.15, 0.125, 0.0625):
    t, s, r = reach(0.5, 0.5, 1.0, sp)
    ok_s = "够（直线）" if s >= 7.7 else "**不够**"
    ok_r = "够（随机）" if r >= 7.7 else "**不够**"
    print(f"{sp:>14.4f}{t:>10.0f}{s:>10.1f}{r:>16.1f}   直线:{ok_s} / 随机:{ok_r}")

print("\n=== 基因两端（速度 0.25 时）===")
print(f"{'g1':>5}{'g6':>5}{'能活tick':>10}{'直线可达':>10}{'随机走有效半径':>16}")
for g1 in (0.0, 0.5, 1.0):
    for g6 in (0.0, 0.5, 1.0):
        t, s, r = reach(g1, g6, 1.0, 0.25)
        print(f"{g1:>5.1f}{g6:>5.1f}{t:>10.0f}{s:>10.1f}{r:>16.1f}")

print("\n=== 关键对照：速度 1.0（现状）下 ===")
for g1, g6 in ((0.0, 0.0), (0.5, 0.5), (1.0, 1.0)):
    t, s, r = reach(g1, g6, 1.0, 1.0)
    print(f"  g1={g1:.1f} g6={g6:.1f} ⇒ 活 {t:.0f} tick｜直线 {s:.0f} 格（**够横穿世界 60 格**）"
          f"｜随机走 {r:.1f} 格（刚好在最近斑块 7.7 格附近 ⇒ 实测存活 ~14% 吻合）")

print("\n=== 判读 ===")
print("  • 现状（1 格/tick）：随机走的净位移 ~10 格，最近斑块 7.7 格 ⇒ **卡在临界点上**")
print("    ⇒ 这正解释了实测「约 14% 存活」：刚好够得着，纯靠运气。")
print("  • 降到 0.25 格/tick：随机走的有效半径掉到 ~2.5 格 ⇒ **远小于 7.7 ⇒ 几乎全灭**")
print("    ⇒ 🔴 **减速必须与直线惯性同时上**，否则种群直接崩。")
print("  • 加了直线惯性后（0.25 速度）：直线可达 26 格 ≫ 7.7 ⇒ 够用，且**选择压会强烈偏好高 pers**")
print("    ⇒ 这正好是我们要的：让「会不会找路」变成一个**真有用的性状**。")
