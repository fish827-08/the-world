"""R233 T-F 冒烟：rd 开（S2 同款档）+ 快照续跑端到端 —— 云端通宵旗舰的前置。

问题（R233 §七 T-F）：云启交付的 `--save-every` / `--resume-from` 只在 rd 关（S1 档）验过；
rd 开档（S2/旗舰 主档）**从未验**。要答：快照能否忠实捕获并恢复
  ① rd 状态（掩码/容量/内部数组/计数器）② 信号稀疏状态（T-E 解耦后的目标组合）
⇒ "续跑 == 连续跑"逐位一致？

口径（= `experiments/s2_depletion_probe.py` 的 S1_BASE，"S2 同款"）：
  480×960 / patches=480 / pop=2000 / subpos on / speed_max=gain=0.125 / subdiv=80
  / k=2.5 / max_count=30000 / rgm=1.195 ＋ `resource_dynamics.enabled=True`
  ＋ `sparse_fields=True`（rd 开 + 资源/信号两侧稀疏；R244 v1 起资源侧也子集化）。
流程模拟 `--save-every 500`：t=500、1000 经 `save_ckpt` 存三件套；"中断"发生在 t=1000
存档后；从 t=1000 快照 `load_ckpt` 续跑到 1500，与连续跑**逐 tick digest** 对拍。

🔴 历史（首跑即抓到真缺陷，2026-09-27）：`SimConfig.from_dict` 曾整段漏传 `subpos`
  ⇒ `load_snapshot(config=None)` 静默把 subpos 退回默认（换运动模型）⇒ 续跑**不可**。
  修复见 `simulation/config.py`；本脚本此后应恒为"可"（回归即炸）。
🔴 v1 更新（R244，2026-09-28）：rd ∧ bgzero（本档）已**放行**资源侧惰性 ⇒ `_lazy` 应为
  True（此前锁 rd 时为 False）。本冒烟同时覆盖"脏格集（派生量）跨快照重建"。

用法：.venv\\Scripts\\python.exe experiments/rd_resume_smoke.py
边界：**同机同树**才保证逐位（跨机续跑不保证，R218 §二）。
"""
from __future__ import annotations

import hashlib
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

from experiments.scaling_rescale import apply_post_build  # noqa: E402
from experiments.steady_k_probe import load_ckpt, make_cfg, save_ckpt  # noqa: E402
from simulation.sphere_engine import SphereEngine  # noqa: E402

SNAP_DIR = ROOT / "_rerun_logs" / "snap" / "smoke_rd"
TAG = "smoke_rd_s2like_s42"
T_SNAP = 1000          # "中断"点 = 最后一组快照
T_END = 1500
SAVE_EVERY = 500


def build() -> SphereEngine:
    """S2 同款 + rd 开 + 稀疏开（R244 v1 起资源侧（rd ∧ bgzero）也放行 ⇒ `_lazy` 应为 True）。"""
    c, notes = make_cfg(seed=42, rows=480, cols=960, pop=2000, patches=480,
                        subpos=True, speed_max=0.125, gain=0.125, subdiv=80,
                        k=2.5, max_count=30000, rgm=1.195)
    c.simulation.use_sim_core = False
    c.resource_dynamics.enabled = True
    c.simulation.sparse_fields = True
    e = SphereEngine(c)
    apply_post_build(e, notes)
    return e


def sha(a: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def digest(e: SphereEngine) -> tuple:
    return (e.tick, len(e._id), int(e._flat.sum()), round(float(e._energy.sum()), 6),
            sha(e.resources._grid), sha(e.signals._marks), sha(e.signals._age))


def rd_state(e: SphereEngine) -> dict:
    rd = e._rd
    return {
        "mask": rd._mask.copy(), "dead": rd._dead.copy(),
        "rest_until": rd._rest_until.copy(), "dead_since": rd._dead_since.copy(),
        "demoted": rd._demoted.copy(), "damage": rd._damage.copy(),
        "kill_n": int(rd.patch_kill_n), "reborn_n": int(rd.patch_reborn_n),
        "forced_n": int(rd.forced_reborn_n), "promote_n": int(rd.promote_n),
        "rest_set_n": int(rd.rest_set_n),
        "n_mask_0": int(rd._n_mask_0), "cap_total_0": round(float(rd._cap_total_0), 9),
        "cap_now_sha": sha(rd.capacity_from_base()),
    }


def rd_diff(a: dict, b: dict) -> list[str]:
    out = []
    for k, va in a.items():
        vb = b[k]
        if isinstance(va, np.ndarray):
            if not (va.shape == vb.shape and np.array_equal(va, vb)):
                out.append(f"{k}: 数组不等（{np.count_nonzero(va != vb)} 格差异）")
        elif va != vb:
            out.append(f"{k}: {va!r} != {vb!r}")
    return out


def main() -> int:
    fails: list[str] = []
    t0 = time.time()
    meta = {"tag": TAG, "rows": 480, "cols": 960, "patches": 480, "seed": 42,
            "k": 2.5, "rgm": 1.195, "sample": 250, "save_every": SAVE_EVERY}

    # ---------- A：连续跑（参考臂） ----------
    A = build()
    assert A._rd.enabled is True and A.signals._sparse is True, "机制没开 ⇒ 冒烟空转"
    assert A.resources._lazy is True, "rd ∧ bgzero 档资源侧惰性应放行（R244 v1）"
    dA: dict[int, tuple] = {}
    rdA: dict[int, dict] = {}
    for t in range(1, T_END + 1):
        A.step()
        if t == T_SNAP:
            rdA[T_SNAP] = rd_state(A)
        elif t > T_SNAP:
            dA[t] = digest(A)
    rdA[T_END] = rd_state(A)
    assert not A.extinct and A.tick == T_END, f"参考臂早灭（tick={A.tick}）⇒ 证据不足"
    print(f"[A 连续跑] 到 t={T_END} 完成（{time.time() - t0:.0f}s，N={len(A._id)}，"
          f"Σgrid={A.resources._grid.sum():.0f}）")

    # ---------- B：跑到"中断"点，按 --save-every 存三件套 ----------
    B = build()
    for t in range(1, T_SNAP + 1):
        B.step()
        if t % SAVE_EVERY == 0:
            save_ckpt(B, SNAP_DIR, TAG, meta)
    snap = SNAP_DIR / f"{TAG}.snapshot.npz"
    rngp, metap = snap.with_suffix(".rngstate.pkl"), snap.with_suffix(".meta.json")
    assert snap.exists() and rngp.exists() and metap.exists(), \
        f"三件套不齐：{list(SNAP_DIR.glob(TAG + '.*'))}"
    d_diff = rd_diff(rdA[T_SNAP], rd_state(B))
    if d_diff:
        fails.append("同 t=1000 两独立跑 rd 状态不一致（构造不确定性？）：" + "; ".join(d_diff))
    print(f"[B 跑到 t={T_SNAP} 并存档] 三件套 = {snap.name} / {rngp.name} / {metap.name}")

    # ---------- B2：从快照"续跑"到 T_END，逐 tick 对拍 ----------
    B2, meta2, start_tick = load_ckpt(snap)
    print(f"[B2 续跑] load_ckpt 返回 start_tick={start_tick}（应 {T_SNAP}）；"
          f"sparse={B2.signals._sparse} lazy={B2.resources._lazy} rd={B2._rd.enabled} "
          f"duration={B2.signals.duration}")
    if start_tick != T_SNAP:
        fails.append(f"start_tick={start_tick} != {T_SNAP}")
    if not (B2.signals._sparse is True and B2.resources._lazy is True and B2._rd.enabled):
        fails.append(f"恢复后机制真值不对：sparse={B2.signals._sparse} lazy={B2.resources._lazy}")
    # v1：脏格集是派生量（不进快照）⇒ 恢复后应与 `_grid < _capacity` 严格一致
    dirty_ok = np.array_equal(
        B2.resources._dirty_mask, B2.resources._grid < B2.resources._capacity)
    if not dirty_ok:
        fails.append("恢复后脏格集与 grid<capacity 不一致（rebuild_lazy 未生效？）")

    # 恢复瞬间：rd 内部数组 / 计数器 与参考臂逐位
    d_diff = rd_diff(rdA[T_SNAP], rd_state(B2))
    if d_diff:
        fails.append("恢复瞬间 rd 状态与参考臂不一致：" + "; ".join(d_diff))
    # 信号活跃集重建：应与 age>0 的派生一致
    act = np.flatnonzero(B2.signals._age > 0)
    act_ok = np.array_equal(np.sort(B2.signals._active_idx), act)
    if not act_ok:
        fails.append(f"信号活跃集重建不符：|idx|={B2.signals._active_idx.size} vs |age>0|={act.size}")
    print(f"   恢复瞬间：rd 状态逐位一致{'✓' if not d_diff else '✗'}"
          f"｜活跃集 |idx|={B2.signals._active_idx.size}（与 age>0 全等{'✓' if act_ok else '✗'}）")

    mism = []
    for t in range(T_SNAP + 1, T_END + 1):
        B2.step()
        if digest(B2) != dA[t]:
            mism.append(t)
            if len(mism) >= 5:
                break
    if mism:
        fails.append(f"续跑段逐 tick digest 不一致：首错 tick={mism[0]}（共检 {len(dA)} tick）")
    else:
        print(f"   逐 tick digest 对拍：{len(dA)}/{len(dA)} 全等 ✓（t={T_SNAP + 1}..{T_END}）")

    d_diff = rd_diff(rdA[T_END], rd_state(B2))
    if d_diff:
        fails.append("终态 rd 状态不一致：" + "; ".join(d_diff))
    da, db = A.death_cause_totals(), B2.death_cause_totals()
    if {str(k): int(v) for k, v in da.items()} != {str(k): int(v) for k, v in db.items()}:
        fails.append(f"死因账本不一致：{da} vs {db}")
    print(f"[终态 t={T_END}] N A/B2 = {len(A._id)}/{len(B2._id)}｜Σgrid = "
          f"{A.resources._grid.sum():.0f}/{B2.resources._grid.sum():.0f}｜死因 = {da}")

    print("=" * 78)
    if fails:
        print("== 结论：**不可** ==")
        for f in fails:
            print("  ✗ " + f)
        return 1
    print(f"== 结论：**可**（rd 开 + S2 同款档，save-every→中断→resume 逐位一致；"
          f"墙钟 {time.time() - t0:.0f}s）==")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
