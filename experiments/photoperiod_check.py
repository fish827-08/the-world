"""可复核：engine 的 illumination 是否真的再现"夏季日照更久"。
用与引擎完全相同的天球公式，算 annual DLI（日累积光积分）与昼长。"""
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

TILT = np.deg2rad(23.44)
P = 6000            # season_period (ticks per year)
ROT = 250           # rotation_period (ticks per day) —— 引擎默认
DAYS = P / ROT      # 一年多少"天"
print(f"一年 = {P} tick, 一天 = {ROT} tick ⇒ 一年 {DAYS:.0f} 天")

def dli(lat_deg, tick, n_col=120):
    """整圈经度的 DLI(该 tick 的全球面平均光照) & 该纬度昼长占比"""
    phi = np.deg2rad(lat_deg)
    decl = TILT * np.sin(2*np.pi*(tick % P)/P)
    lon = np.linspace(0, 2*np.pi, n_col, endpoint=False)
    # 子午线随自转扫描，等效对 h 积分
    z = np.sin(phi)*np.sin(decl) + np.cos(phi)*np.cos(decl)*np.cos(lon)
    I = np.maximum(0.0, z)
    day_frac = float(np.mean(I > 0))
    return float(I.mean()), day_frac

print("\n纬度   夏至DLI   冬至DLI   夏/冬比   夏至昼长  冬至昼长")
for lat in [0, 23.44, 45, 60, 66.5, 75]:
    s_dli, s_day = dli(lat, P/4)      # 夏至
    w_dli, w_day = dli(lat, 3*P/4)    # 冬至
    ratio = s_dli/w_dli if w_dli > 1e-6 else float('inf')
    print(f"{lat:5.1f}  {s_dli:8.4f} {w_dli:8.4f}  {ratio:7.2f}   "
          f"{s_day*24:6.1f}h  {w_day*24:6.1f}h")
