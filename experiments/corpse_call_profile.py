"""T-G profile：`_step_corpse_decay` 内全场 `_regrowth_amount` 的调用量/收益（v1 前置）。

问题（R233 §七 T-G）：`sphere_engine.py:1372-1382` —— 只要有 1 格 `corpse_boost>0`，
就调**全场** `resources._regrowth_amount(tick)`（460 800 格），却只取 `bo` 子集。
`_regrowth_amount(tick, idx=...)` **已支持子集**（R217 稀疏化 B 加的接口）⇒ 改动量极小。

本脚本给出三件事（引擎级 A/B 配对，同进程、同种子、同 N）：
  1. `bo>0` 的 tick 占比 + `|bo|` 分布（S2 同款档 + corpse 开）
  2. 配对 ms/tick：原版（全场） vs 子集版 —— 并断言两版**逐位一致**（纯优化）
  3. 外推：60k 批下预计省多少（按实测 delta × tick 数）

用法：.venv\\Scripts\\python.exe experiments/corpse_call_profile.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from experiments.scaling_rescale import apply_post_build            # noqa: E402
from experiments.steady_k_probe import make_cfg                     # noqa: E402
from simulation.sphere_engine import SphereEngine                   # noqa: E402

TICKS = 1500
CORPSE_T0 = 300        # 前 300 tick 喂饱（有死才有尸）


def build() -> SphereEngine:
    c, notes = make_cfg(seed=42, rows=480, cols=960, pop=2000, patches=480,
                        subpos=True, speed_max=0.125, gain=0.125, subdiv=80,
                        k=2.5, max_count=30000, rgm=1.195)
    c.simulation.use_sim_core = False
    c.resource_dynamics.enabled = True
    c.corpse_wound.corpse_enabled = True           # T-G 被测臂：腐烂 boost 生效
    e = SphereEngine(c)
    apply_post_build(e, notes)
    return e


def step_corpse_decay_subset(self) -> None:
    """子集版（与 `sphere_engine._step_corpse_decay` 逐位等价，只把全场换成 `idx`）。"""
    _cwc = self.config.corpse_wound
    has = self._corpse_energy > 0.0
    if has.any():
        self._corpse_age[has] += 1
        ripe = has & (self._corpse_age >= int(_cwc.corpse_decay_ticks))
        if ripe.any():
            cells = np.flatnonzero(ripe)
            self._corpse_decayed_e += float(self._corpse_energy[cells].sum())
            ret = self._corpse_energy[cells] * float(_cwc.corpse_to_plant_frac)
            room = np.maximum(
                0.0, self.resources._capacity[cells] - self.resources._grid[cells])
            put = np.minimum(ret, room)
            self.resources._grid[cells] += put
            self._corpse_boost[cells] = 2000
            self._corpse_energy[cells] = 0.0
            self._corpse_age[cells] = 0
    bo = self._corpse_boost > 0
    if bo.any():
        idx = np.flatnonzero(bo)
        boost_amt = (
            self.resources._regrowth_amount(self._tick, idx)   # ← 唯一改动：子集
            * float(_cwc.corpse_patch_boost)
        )
        room = np.maximum(0.0, self.resources._capacity[idx] - self.resources._grid[idx])
        self.resources._grid[idx] += np.minimum(boost_amt, room)
        self._corpse_boost[idx] -= 1


def digest(e: SphereEngine) -> tuple:
    import hashlib

    def _h(a):
        return hashlib.sha1(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]
    return (e.tick, len(e._id), int(e._flat.sum()), round(float(e._energy.sum()), 6),
            _h(e.resources._grid), _h(e._corpse_boost), _h(e._corpse_energy),
            _h(e._corpse_age), int(e._corpse_decayed_e))


def run_arm(subset: bool, n_bo: list, ms: list, dig: list) -> SphereEngine:
    orig = SphereEngine._step_corpse_decay
    if subset:
        SphereEngine._step_corpse_decay = step_corpse_decay_subset
    try:
        e = build()
        t_slice = time.perf_counter()
        for t in range(1, TICKS + 1):
            e.step()
            n_bo.append(int((e._corpse_boost > 0).sum()))
            dig.append(digest(e))
            if t % 50 == 0:
                now = time.perf_counter()
                ms.append((now - t_slice) / 50 * 1e3)
                t_slice = now
        return e
    finally:
        SphereEngine._step_corpse_decay = orig


def main() -> int:
    print("== T-G：`_step_corpse_decay` 全场 `_regrowth_amount` profile（S2 同款 + corpse 开）==")
    nbo_f: list[int] = []
    ms_f: list[float] = []
    dig_f: list[tuple] = []
    t0 = time.time()
    e_f = run_arm(False, nbo_f, ms_f, dig_f)
    t_full = time.time() - t0

    nbo_s: list[int] = []
    ms_s: list[float] = []
    dig_s: list[tuple] = []
    t0 = time.time()
    e_s = run_arm(True, nbo_s, ms_s, dig_s)
    t_sub = time.time() - t0

    nbo = np.array(nbo_f)
    n_ticks = len(nbo)
    frac_bo = float((nbo > 0).mean())
    print(f"[调用量] tick 数 {n_ticks}；`bo>0` 占比 **{frac_bo * 100:.1f}%**"
          f"（{int((nbo > 0).sum())}/{n_ticks}）")
    if (nbo > 0).any():
        pos = nbo[nbo > 0]
        print(f"         |bo| 分布（仅 bo>0 的 tick）：中位 {int(np.median(pos))}"
              f"｜均值 {pos.mean():.1f}｜p95 {int(np.percentile(pos, 95))}｜max {int(pos.max())}")
        print(f"         全场格数 460800 ⇒ 子集/全场 ≈ {pos.mean() / 460800 * 100:.4f}%")

    ms_f_a, ms_s_a = np.array(ms_f), np.array(ms_s)
    ok_bit = dig_f == dig_s
    print(f"[逐位] 两版 digest 全等：{'是 ✓（纯优化）' if ok_bit else '否 ✗ —— 子集版不等价，废弃!'}")
    if not ok_bit:
        bad = [i for i, (a, b) in enumerate(zip(dig_f, dig_s)) if a != b][:5]
        print(f"       首错采样点（每 50 tick 一个）：{bad}")

    # 配对：只看"bo>0"的采样窗口（每 50 tick 一个样本 ⇒ 窗口含 bo 的判据=窗口末值>0）
    pos_mask = np.array([nbo_f[(i + 1) * 50 - 1] > 0 for i in range(len(ms_f))])
    d_all = ms_f_a - ms_s_a
    print(f"[配对 ms/tick（每 50 tick 均值）] 全窗口：原版 {ms_f_a.mean():.2f}｜子集 "
          f"{ms_s_a.mean():.2f}｜Δ {d_all.mean():.2f}")
    if pos_mask.any():
        print(f"    其中 `bo>0` 窗口（{int(pos_mask.sum())}/{len(ms_f)}）：原版 "
              f"{ms_f_a[pos_mask].mean():.2f}｜子集 {ms_s_a[pos_mask].mean():.2f}｜"
              f"**Δ {d_all[pos_mask].mean():.2f} ms/tick**")
    if (~pos_mask).any():
        print(f"    `bo==0` 窗口（{int((~pos_mask).sum())}）：Δ {d_all[~pos_mask].mean():.2f}"
              f"（应为 ~0：无调用即无差异）")
    print(f"[墙钟] 原版臂 {t_full:.0f}s｜子集臂 {t_sub:.0f}s｜N 终值 {len(e_f._id)}/{len(e_s._id)}"
          f"｜Σgrid {e_f.resources._grid.sum():.0f}/{e_s.resources._grid.sum():.0f}")
    print(f"[外推] 若 `bo>0` 占比同此（{frac_bo * 100:.0f}%）⇒ 60k tick 单 run 预计省 "
          f"**{d_all[pos_mask].mean() * frac_bo * 60000 / 1e3 / 60:.1f} 分钟**（[外推]，按本机实测 Δ）")
    return 0 if ok_bit else 1


if __name__ == "__main__":
    raise SystemExit(main())
