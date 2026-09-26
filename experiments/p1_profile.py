"""P1 profile（R210 §三）—— 定位「60–74 μs/体」到底花在哪。

背景（派工包 §三，`[实测]`）：Rust 臂 1.5 μs/体 vs Python+subpos 60–74 μs/体（差 40 倍）。
**但这是推断，须用 profile 证实或推翻** ⇒ 本脚本。

为什么要跑「轨迹 + 定点窗口」而不是跑几 tick 就 profile
------------------------------------------------------
`[实测]` 新鲜起手的 480×960 + pop 3000 **前 10 tick 就要崩**（N 2949→2924→…→867@t25）
——早期态里尸体/伤口/信号/胃容都还空着 ⇒ **子系统没被激活，成本被系统性低估**
（实测早段 27.7 μs/体 vs T4 成熟态 48–55 μs/体）。
⇒ 本脚本沿 T4 同一条轨迹跑到成熟态（默认 t=10 000，N≈2 400），
在 `--checkpoints` 指定 tick 处开 **cProfile 窗口**（只影响计时、不影响状态/RNG）。

三臂
----
| 臂 | use_sim_core | subpos | 用途 |
|---|---|---|---|
| `py+subpos` | False | True | T4 实测臂（60–74 μs/体的当事人） |
| `py` | False | False | 与 rust 臂**同轨迹**（subpos 关）⇒ 干净的边际成本对照 |
| `rust` | True | False | 目标（1.5 μs/体） |

产出（`results/`，gitignored）
* `<arm>.csv`：逐采样 `t, N, ms_per_tick, μs/体`（R189：ETA 只用实测样本推）
* `<arm>_t<tick>_profile.txt`：该 checkpoint 的 tottime / cumtime 前 `--top` 表
* 运行末：**ms/tick = a（固定）+ b（每体）× N** 的最小二乘拟合 ⇒ b 即"边际 μs/体"

🔴 纪律
* 不改任何引擎代码（纯观测）；配置与 T4 逐字段一致（`results/t4_steady_k_k25` 的 switches）。
* ms/tick **只取非 profile 窗口**；profile 窗口只用于**相对占比**（cProfile 会放大绝对时间）。

用法
----
    .venv\\Scripts\\python.exe experiments/p1_profile.py --arms py+subpos
    .venv\\Scripts\\python.exe experiments/p1_profile.py --arms py --run-to 4000
"""
from __future__ import annotations

import argparse
import cProfile
import csv
import io as _io
import pstats
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simulation.config import SimConfig                      # noqa: E402
from simulation.sphere_engine import SphereEngine            # noqa: E402
from experiments.scaling_rescale import (                    # noqa: E402
    apply_post_build, rescale_config,
)

# --- R98 纪律：Windows GBK 控制台兜底（非 ASCII print 会让脚本 rc=1 假失败；F-R15 族）---
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass  # 非 TTY / 旧解释器：不因诊断能力缺失而阻断运行


ARMS: dict[str, tuple[bool, bool]] = {      # 名字 -> (use_sim_core, subpos)
    "py+subpos": (False, True),
    "py": (False, False),
    "rust": (True, False),
}


def safe(name: str) -> str:
    return name.replace("+", "_")


def make_cfg(seed: int, rows: int, cols: int, pop: int, patches: int,
             use_sim_core: bool, subpos: bool, speed_max: float, gain: float,
             subdiv: int, k: float, max_count: int) -> tuple[SimConfig, dict]:
    """T4 同款配置（`results/t4_steady_k_k25/s42_full.summary.json` 的 switches）。"""
    c = SimConfig(seed=seed)
    c.world.rows, c.world.cols = rows, cols
    c.resources.distribution = "patchy"
    c.resources.bg_production_zero = True
    c.resources.patch_regrowth_mult = 1.195
    c.resources.patch_count = patches
    if pop > 0:
        c.population.initial_count = pop
    if max_count > 0:
        c.population.max_count = int(max_count)
    c.simulation.use_sim_core = bool(use_sim_core)
    if subpos:
        c.subpos.enabled = True
        c.subpos.speed_max = float(speed_max)
        c.subpos.speed_gain = float(gain)
        c.subpos.subdiv = int(subdiv)
    notes = rescale_config(c, k) if abs(k - 1.0) > 1e-12 else {
        "signals_duration": int(c.signals.duration_ticks)}
    return c, notes


def dump_profile(pr: cProfile.Profile, path: Path, top: int) -> str:
    st = pstats.Stats(pr)
    parts = []
    for sort_key, title in (("tottime", "自时间 tottime"), ("cumtime", "累计 cumtime")):
        st.sort_stats(sort_key)
        buf = _io.StringIO()
        st.stream = buf
        st.print_stats(top)
        st.stream = sys.stdout
        parts.append(f"---- {title} 前 {top} ----\n" + buf.getvalue())
    text = "\n".join(parts)
    path.write_text(text, encoding="utf-8")
    return text


def dump_line_profile(lp, path: Path, top: int) -> str:
    buf = _io.StringIO()
    try:
        lp.print_stats(stream=buf)
    except TypeError:                     # 旧版签名：只写 stdout
        _old, sys.stdout = sys.stdout, buf
        try:
            lp.print_stats()
        finally:
            sys.stdout = _old
    text = buf.getvalue()
    path.write_text(text, encoding="utf-8")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return "\n".join(lines[:200])


def run_arm(name: str, use_sim_core: bool, subpos: bool, a, out_dir: Path) -> None:
    cfg, notes = make_cfg(a.seed, a.rows, a.cols, a.pop, a.patches, use_sim_core,
                          subpos, a.speed_max, a.gain, a.subdiv, a.k, a.max_count)
    t0 = time.time()
    eng = SphereEngine(cfg)
    apply_post_build(eng, notes)
    print(f"\n=== 臂 {name}（use_sim_core={use_sim_core}, subpos={subpos}）"
          f"｜构造 {time.time() - t0:.1f}s ⇒ 跑到 t={a.run_to} ===", flush=True)

    ckpts = sorted({int(x) for x in a.checkpoints.split(",") if x.strip()})
    samples: list[dict] = []
    last_t, t_slice = 0, time.perf_counter()
    t = 0
    while t < a.run_to:
        eng.step()
        t += 1
        if t in ckpts:
            if a.line_profile:
                from line_profiler import LineProfiler      # 可选依赖（本机已装 5.0.2）
                from world.resource_field import ResourceField
                lp = LineProfiler()
                for fn in (SphereEngine._advance_one_tick, SphereEngine._step_population,
                           ResourceField.regrow, ResourceField._regrowth_amount):
                    lp.add_function(fn)
                lp.enable()
                for _ in range(a.profile_window):
                    eng.step()
                lp.disable()
                t += a.profile_window
                txt = dump_line_profile(
                    lp, out_dir / f"{safe(name)}_t{t - a.profile_window}_lineprofile.txt", a.top)
                print(f"    ── **行级** profile 窗口 t={t - a.profile_window}..{t}"
                      f"（N={len(eng._flat)}）──", flush=True)
                for line in txt.splitlines():
                    print("    " + line, flush=True)
            else:
                pr = cProfile.Profile()
                pr.enable()
                for _ in range(a.profile_window):
                    eng.step()
                pr.disable()
                t += a.profile_window
                txt = dump_profile(pr, out_dir / f"{safe(name)}_t{t - a.profile_window}_profile.txt", a.top)
                print(f"    ── profile 窗口 t={t - a.profile_window}..{t}（N={len(eng._flat)}）──", flush=True)
                for line in txt.splitlines():
                    if line.strip() and ("ncalls" in line or ".py:" in line or "built-in" in line):
                        print("    " + line, flush=True)
            last_t, t_slice = t, time.perf_counter()
            continue
        if t % a.sample == 0:
            dt_ms = (time.perf_counter() - t_slice) / max(t - last_t, 1) * 1e3
            N = int(len(eng._flat))
            samples.append({"arm": name, "t": t, "N": N, "ms_per_tick": round(dt_ms, 3),
                            "us_per_body": round(dt_ms * 1e3 / max(N, 1), 2)})
            last_t, t_slice = t, time.perf_counter()
            eta_min = (a.run_to - t) * dt_ms / 1e3 / 60.0
            print(f"    t={t:<6} N={N:<6} {dt_ms:7.2f} ms/tick（每体 {dt_ms * 1e3 / max(N, 1):5.1f} μs）"
                  f"  ETA {eta_min:5.1f} min", flush=True)

    out_csv = out_dir / f"{safe(name)}.csv"
    with _io.open(out_csv, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["arm", "t", "N", "ms_per_tick", "us_per_body"])
        w.writeheader()
        w.writerows(samples)

    if len(samples) >= 3:
        Ns = np.array([s["N"] for s in samples], dtype=float)
        ms = np.array([s["ms_per_tick"] for s in samples], dtype=float)
        b, aa = np.polyfit(Ns, ms, 1)          # ms/tick = aa + b × N
        print(f"    ⇒ 拟合（{len(samples)} 采样）：**ms/tick = {aa:.1f} + {b * 1e3:.2f} × N（μs/体）**"
              f"，拟合区间 N {int(Ns.min())}–{int(Ns.max())}", flush=True)
    print(f"    ⇒ 明细已写入 {out_csv}（墙钟 {time.time() - t0:.0f}s）", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="P1 profile：Python(+subpos) 路径热函数定位")
    ap.add_argument("--rows", type=int, default=480)
    ap.add_argument("--cols", type=int, default=960)
    ap.add_argument("--patches", type=int, default=480)
    ap.add_argument("--pop", type=int, default=3000,
                    help="初始个体数（T4 平台期 ≈3 000；用同量级起步）")
    ap.add_argument("--k", type=float, default=2.5, help="时间压缩（T4 定档 k=2.5）")
    ap.add_argument("--speed-max", type=float, default=0.125)
    ap.add_argument("--gain", type=float, default=0.125)
    ap.add_argument("--subdiv", type=int, default=80)
    ap.add_argument("--max-count", type=int, default=30000, help="同 T4（防撞顶干扰）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--run-to", type=int, default=10000)
    ap.add_argument("--sample", type=int, default=250)
    ap.add_argument("--checkpoints", default="2500,5000,10000",
                    help="在这些 tick 处开 profile 窗口（逗号分隔）")
    ap.add_argument("--profile-window", type=int, default=10)
    ap.add_argument("--line-profile", action="store_true",
                    help="checkpoint 窗口改用 line_profiler **行级**归因"
                         "（`_step_population` 是巨型函数，函数级表看不进去）")
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--arms", default="py+subpos", help="逗号分隔，可选 " + "/".join(ARMS))
    ap.add_argument("--out-dir", default="results/p1_profile")
    a = ap.parse_args()

    names = [x.strip() for x in a.arms.split(",") if x.strip()]
    bad = [n for n in names if n not in ARMS]
    if bad:
        raise SystemExit(f"未知臂 {bad}；可选 {list(ARMS)}")
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"== P1 profile：{a.rows}x{a.cols}（{a.rows * a.cols:,} 格）"
          f"，斑块 {a.patches}，pop {a.pop}，k={a.k:g}，subdiv={a.subdiv}"
          f"，跑到 t={a.run_to}，checkpoints {a.checkpoints}，seed {a.seed} ==")

    for n in names:
        uc, sp = ARMS[n]
        run_arm(n, uc, sp, a, out_dir)

    print("\n读法：")
    print("  · tottime 前几 = **真正吃 CPU 的函数**；ncalls ≈ N 或 N 的倍数 ⇒ **逐个体调用**")
    print("  · 拟合斜率 b = **边际 μs/体**（与截距 a 的固定成本分开看）⇒ 与 Rust 臂 1.5 μs/体 对照")


if __name__ == "__main__":
    main()
