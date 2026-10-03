# -*- coding: utf-8 -*-
"""R290 v2：seed101 on/off 双臂，t800-900 窗口，全量计数器（含模块级质心函数）。"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import simulation.sphere_engine as se
from experiments.steady_k_probe import make_cfg, apply_post_build
from simulation.sphere_engine import SphereEngine
from world.resource_dynamics import ResourceDynamics
from world.resource_field import ResourceField
from world.signal_field import SignalField

stats = {}
def wrap_any(obj, name, tag):
    if not hasattr(obj, name):
        print(f"  [skip] {tag}"); return
    orig = getattr(obj, name)
    def f(*a, **k):
        t0 = time.perf_counter()
        r = orig(*a, **k)
        dt = time.perf_counter() - t0
        s = stats.setdefault(tag, [0, 0.0]); s[0] += 1; s[1] += dt
        return r
    setattr(obj, name, f)

wrap_any(se, "_build_patch_centroid", "MOD:_build_patch_centroid")
wrap_any(se, "_update_patch_centroid_incremental", "MOD:_update_patch_centroid_incremental")
for name in ["_advance_one_tick", "_step_population", "_memory_egocentric_cos",
             "_memory_orientation_cos", "_mem_v2_write", "_mem_v2_step_batch",
             "_mem_v2_step_one", "_update_pleasure"]:
    wrap_any(SphereEngine, name, f"ENG:{name}")
for name in ["note_tick", "rotate", "capacity_from_base", "growth_multiplier"]:
    wrap_any(ResourceDynamics, name, f"RD:{name}")
wrap_any(ResourceField, "_regrowth_amount", "RES:_regrowth_amount")
wrap_any(SignalField, "tick", "SIG:tick")

def build(seed, mem_on):
    B = dict(rows=480, cols=960, patches=1700, pop=10000,
             speed_max=0.125, gain=0.125, subdiv=80, k=2.5, max_count=30000, rgm=1.195,
             bg_low_prod_frac=0.4, bg_low_cap_mult=0.05)
    c, notes = make_cfg(seed, B["rows"], B["cols"], B["pop"], B["patches"], True,
                        B["speed_max"], B["gain"], B["subdiv"], B["k"], B["max_count"], B["rgm"],
                        bg_low_prod_frac=B["bg_low_prod_frac"], bg_low_cap_mult=B["bg_low_cap_mult"])
    c.simulation.use_sim_core = False
    c.resource_dynamics.enabled = True
    c.info_structure.memory_v2 = bool(mem_on)
    c.info_structure.memory_gradient = "orientation" if mem_on else "none"
    eng = SphereEngine(c)
    apply_post_build(eng, notes)
    return eng

for arm, mem_on in [("on", True), ("off", False)]:
    for s in stats.values():
        s[0] = 0; s[1] = 0.0
    eng = build(101, mem_on)
    print(f"[{arm}] 预热到 t=800 …", flush=True)
    for _ in range(800):
        eng.step()
    # 取 N（试多种属性名）
    N = "?"
    for attr in ("_n_alive", "n_alive", "_alive_count", "count"):
        v = getattr(eng, attr, None)
        if v is not None:
            N = v() if callable(v) else v; break
    if N == "?":
        N = int(len([x for x in getattr(eng, "_id", []) if x is not None])
                if hasattr(eng, "_id") else 0) or "?"
    for s in stats.values():
        s[0] = 0; s[1] = 0.0
    print(f"[{arm}] 测量 t800→900，N≈{N}", flush=True)
    T = 100
    t0 = time.perf_counter()
    for _ in range(T):
        eng.step()
    dt = time.perf_counter() - t0
    print(f"\n===== [{arm}] 窗口: {dt:.1f}s / {T}t = {dt/T*1e3:.1f} ms/t (N≈{N}) =====")
    print(f"{'方法':>46} {'次/t':>9} {'ms/t':>9} {'占整tick':>8} {'µs/次':>9}")
    items = sorted(stats.items(), key=lambda kv: -kv[1][1])
    for tag, (n, tot) in items:
        if n == 0: continue
        print(f"{tag:>46} {n/T:>9.1f} {tot/T*1e3:>9.2f} {tot/dt*100:>7.1f}% {tot/n*1e6:>9.1f}")
    print()

print("[done]")
