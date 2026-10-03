"""快照恢复不变量检查：_id 键控账本长度是否随 _next_id 恢复。

背景：P2 验证时用快照做“旧/新实现同状态对拍”，恢复后 10 tick 内崩在
`_rs_children[self._id[ri]] += 1`（index 5469 / size 3003）。本脚本最小化复现。
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
from simulation.config import SimConfig
from simulation.sphere_engine import SphereEngine

LEDGERS = ("_rs_children", "_rs_observed", "_rs_g15", "_rs_age", "_rs_energy",
           "_rs_cc0", "_rs_cohort", "_emit_count", "_oracle_gain")
P = "_trash_local/_snap_invariant.npz"

cfg = SimConfig(seed=1)
cfg.population.initial_count = 5
e = SphereEngine(cfg)
print(f"构造后：_next_id={e._next_id}  _measure_resp={e._measure_resp}  _oracle_on={e._oracle_on}")
print(f"  账本长度：{ {a: len(getattr(e, a)) for a in LEDGERS} }")
e._id[:] = np.array([30, 31, 32, 33, 34], dtype=e._id.dtype)   # 现存个体 id 均 > initial_count
e._next_id = 40                                                # 模拟已跑出一批出生
before = {a: len(getattr(e, a)) for a in LEDGERS}
e.save_snapshot(P)
e2 = SphereEngine.load_snapshot(P)
after = {a: len(getattr(e2, a)) for a in LEDGERS}
print(f"恢复后：_next_id={e2._next_id}")
print(f"  _id[:]={e2._id[:5].tolist()}")
print(f"  账本长度：{after}")
bad = [a for a in LEDGERS if after[a] != e2._next_id]
print(f"⇒ 不变量 `len(账本) == _next_id`：{'违反 ' + str(bad) if bad else '成立 ✅'}")
# 功能面：恢复后首个出生即崩？
try:
    for _ in range(30):
        e2.step()
    print("⇒ 恢复后 30 tick 未崩（本构造未触发出生）")
except IndexError as ex:
    print(f"⇒ 恢复后 step() 崩：IndexError: {ex}")
