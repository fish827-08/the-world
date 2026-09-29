"""稳态 K 探针（14.10 P0.2）—— 实测「世界放大后到底能养活多少个体」。

为什么必须实测
--------------
我们现在的 `K ≈ 3.67 × 产能格` 是在 **60×120** 上标定的线性外推。
评审（2026-09-26）指出这很可能是高估：世界放大后，斑块间距变大，
"找不到食物"本身会压低**实际** K —— 食物长满了没人吃。
⇒ 在决定要不要投稀疏化那套大工程之前，必须先把这个数测出来。

本探针回答三件事
----------------
1. **稳态个体数 K**：在每个斑块密度下跑到种群平台期，报平台期中位数
2. **资源饱和度**：斑块存量/容量（≈1 ⇒ 食物堆着没人吃 ⇒ 瓶颈是"可达性"不是"产量"）
3. **性能耦合**：不同 N 下的 ms/tick，用来估"食物变多 ⇒ 生物变多 ⇒ 慢多少"

口径（B4）
----------
* **K** = 最后 20% tick 的种群中位数（先断言尾部斜率已平）
* **产能格** = `capacity > 0` 的格数（`bg_production_zero=True` ⇒ 只有斑块有产能）
* **资源饱和度** = `Σ存量 / Σ容量`（全部产能格）
* 🔴 **1 − 此值 ≠ 被吃掉的比例**（后者见 `food_util_frac`，R190/E-035，实测 ~2%）
  —— 别把"食物堆着"读成"食物被充分利用"（R211 命名裁定）
* **外推 K** = 3.67 × 产能格（旧标定，用于对照）

用法
----
    .venv\\Scripts\\python.exe experiments/steady_k_probe.py --patches 30,60,120 --ticks 6000
    .venv\\Scripts\\python.exe experiments/steady_k_probe.py --patches 30 --subpos on --ticks 4000
    .venv\\Scripts\\python.exe experiments/steady_k_probe.py --patches 480 --max-count 30000 --k 2.5 --subpos on

快照 / 续跑 / 分片（R221 §四，`[云端开发·云启]` 2026-09-27）
--------------------------------------------------------
* `--save-every N` —— 每 N tick 存一次**中快照**（默认 0 = 关；长批建议 500）
* `--resume-from <snap|dir>` —— 从快照续跑（可给**单个快照**或**目录**⇒ 逐个续）；
  世界/配置/标签**一律以快照为准**；`--out` 里该 run 的旧行**自动去重**（tick > 快照点丢弃）
* `--shard K/N` 或 `--shard p:sd[,rgm:p:sd…]` —— 只跑 `(斑块, seed)` 子集 ⇒ **分片到多机**用
* `--rd {on,off}` —— 开 `resource_dynamics`（S2 同款臂：bgzero + rgm 1.195 + patches 480）。
  ⚠️ 续跑（`--resume-from`）时**以快照配置为准**，本开关仅对"新跑"生效（不一致会告警）
* 产物（`--snapshot-dir`，默认 `_rerun_logs/snap/`，已在 `.gitignore` 白名单）：
  `<tag>.snapshot.npz`（引擎态）+ `<tag>.rngstate.pkl`（F-D2 全局 np.random）+ `<tag>.meta.json`（run 身份与 k）
* 🔴 **逐位一致的边界**：**同机同树**续跑 ≡ "不中断连续跑"（含 RNG 序列与死因账本）；
  **跨机**续跑**不保证**（numpy/CPU 差异，R218 §二）⇒ 长批优先"按 run 分片"，而不是"同 run 跨机续"
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simulation.config import SimConfig                      # noqa: E402
from simulation.sphere_engine import SphereEngine            # noqa: E402
from experiments.scaling_rescale import (                    # noqa: E402
    SUBDIV_STD, apply_post_build, rescale_config,
)

# --- R98 纪律：Windows GBK 控制台兜底（非 ASCII print 会让脚本 rc=1 假失败；F-R15 族）---
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass  # 非 TTY / 旧解释器：不因诊断能力缺失而阻断运行


K_PER_CELL = 3.6667      # 旧标定：3.67 体/产能格（60×120 上测的）


def _bar(cur: int, tot: int, extra: str = "", width: int = 24) -> str:
    frac = cur / max(tot, 1)
    done = int(frac * width)
    return f"[{'#' * done}{'.' * (width - done)}] {frac * 100:5.1f}% {extra}"


# ══════════════════════════════════════════════════════════════════════════
# 快照 / 续跑 / 分片（R221 §四）
# ══════════════════════════════════════════════════════════════════════════
def run_tag(rows: int, cols: int, patches: int, seed: int, k: float, rgm: float,
            rd: bool = False) -> str:
    """run 的唯一标识（快照文件名 / 分片声明用；确定性 + 可读）。

    `rd=True` ⇒ 追加 `_rd` 后缀（rd 改变轨迹 ⇒ 开关两臂的快照/行**不许同名互顶**）。
    """
    return f"w{rows}x{cols}_p{patches}_s{seed}_k{k:g}_rgm{rgm:g}" + ("_rd" if rd else "")


def _ckpt_paths(snapshot_dir: "str | Path", tag: str) -> tuple[Path, Path, Path]:
    """(快照, rng 侧车, meta) 三个**固定路径**（原地覆盖；D-23a：每 run 只 1 组文件）。

    🔴 命名 **`<tag>.snapshot.npz`**（不是 `<tag>.npz`）——`_collect_snapshots` 按此 glob；
    ⚠️ 别用 `Path.with_suffix` 从 `<tag>.snapshot` 拼（它会把 `.snapshot` 当成后缀**吃掉**）。
    """
    snap = Path(snapshot_dir) / f"{tag}.snapshot.npz"
    return (snap, snap.with_suffix(".rngstate.pkl"), snap.with_suffix(".meta.json"))


def save_ckpt(eng: SphereEngine, snapshot_dir: "str | Path", tag: str, meta: dict) -> None:
    """D-23a 范式：`save_snapshot` + rng 侧车 + run meta（续跑三件套，同节拍写出）。

    ⚠️ rng 侧车是为 F-D2（D2 感知噪声走**全局 np.random**，引擎快照不含它）——
    与 `experiments/a4_verify_capacity.py` 同一处理。
    """
    snap, rngp, metap = _ckpt_paths(snapshot_dir, tag)
    snap.parent.mkdir(parents=True, exist_ok=True)
    eng.save_snapshot(str(snap))
    with open(rngp, "wb") as fh:
        pickle.dump(np.random.get_state(), fh)
    doc = dict(meta)
    doc["tick_saved"] = int(eng._tick)
    with open(metap, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, ensure_ascii=False, indent=1)


def load_ckpt(snap_path: "str | Path") -> tuple[SphereEngine, dict, int]:
    """读快照 + rng 侧车 + meta ⇒ `(eng, meta, start_tick)`。

    · 引擎侧：`load_snapshot(config=None)` ⇒ **配置/指纹随快照走**（R218 §二 键表）；
      缺键回退由 `load_snapshot` 自身保证（R216：ARS 六数组等可选键）
    · **k≠1 的信号寿命**不在快照键表里，但由**快照配置**在构造期恢复（`__init__` 读
      `config.signals.duration_ticks`）⇒ 这里断言自证，防将来回归（B4）
    · rng 侧车缺失 ⇒ 告警（续跑轨迹可能与连续跑不逐位）；meta 缺失 ⇒ 从快照配置**反推**标签
    """
    snap = Path(snap_path)
    if not snap.exists():
        raise FileNotFoundError(f"快照不存在：{snap}")
    eng = SphereEngine.load_snapshot(str(snap))
    assert int(eng.signals.duration) == int(eng.config.signals.duration_ticks), (
        "信号寿命未随快照配置恢复（k≠1 续跑会静默错位）")
    rngp = snap.with_suffix(".rngstate.pkl")
    if rngp.exists():
        with open(rngp, "rb") as fh:
            np.random.set_state(pickle.load(fh))
    else:
        print(f"⚠️ 缺 rng 侧车 `{rngp.name}` ⇒ 续跑轨迹可能与连续跑不一致（F-D2 全局 np.random）",
              file=sys.stderr)
    metap = snap.with_suffix(".meta.json")
    if metap.exists():
        with open(metap, "r", encoding="utf-8") as fh:
            meta = json.load(fh)
    else:
        cfg = eng.config
        meta = {"tag": snap.name.replace(".snapshot.npz", ""),
                "rows": int(cfg.world.rows), "cols": int(cfg.world.cols),
                "patches": int(cfg.resources.patch_count), "seed": int(cfg.seed),
                "k": round(2400.0 / max(float(cfg.light.rotation_period), 1e-9), 4),
                "rgm": float(cfg.resources.patch_regrowth_mult),
                "subpos": bool(cfg.subpos.enabled), "_derived_from_config": True}
        print(f"⚠️ 缺 meta `{metap.name}` ⇒ 标签改由快照配置反推（k={meta['k']:g}）；"
              f"本工具自己写的快照都会有 meta", file=sys.stderr)
    return eng, meta, int(eng._tick)


def _collect_snapshots(spec: "str | Path") -> list[Path]:
    """`--resume-from`：单个快照文件，或一个目录（⇒ 取其下全部 `*.snapshot.npz`，按名排序）。"""
    p = Path(spec)
    if p.is_dir():
        snaps = sorted(p.glob("*.snapshot.npz"))
        if not snaps:
            raise FileNotFoundError(f"--resume-from 目录里没有 *.snapshot.npz：{p}")
        return snaps
    if not p.exists():
        raise FileNotFoundError(f"--resume-from 不存在：{p}")
    return [p]


def select_runs(grid: list[tuple[float, int, int]],
                spec: "str | None") -> list[tuple[float, int, int]]:
    """`--shard` 选择：`K/N`（确定性分片：共 N 份取第 K 份）或显式 `p:sd` / `rgm:p:sd`（`,` 分隔）。

    返回保持 `grid` 原始顺序的子集；**匹配 0 个 ⇒ 抛 ValueError**（B3：禁静默空跑）。
    """
    if not spec:
        return list(grid)
    s = str(spec).strip()
    if "/" in s:
        kk, nn = (int(x) for x in s.split("/", 1))
        if not (1 <= kk <= nn):
            raise ValueError(f"--shard {spec!r}：K/N 需满足 1 ≤ K ≤ N")
        out = [g for i, g in enumerate(grid) if i % nn == kk - 1]
    else:
        exact: set[tuple[float, int, int]] = set()
        pairs: set[tuple[int, int]] = set()
        for item in s.split(","):
            parts = [x.strip() for x in item.split(":") if x.strip()]
            if len(parts) == 2:
                pairs.add((int(parts[0]), int(parts[1])))
            elif len(parts) == 3:
                exact.add((round(float(parts[0]), 9), int(parts[1]), int(parts[2])))
            else:
                raise ValueError(f"--shard 项无法解析：{item!r}（形如 `p:sd` 或 `rgm:p:sd`）")
        out = [g for g in grid
               if (g[1], g[2]) in pairs or (round(g[0], 9), g[1], g[2]) in exact]
    if not out:
        raise ValueError(f"--shard {spec!r} 匹配 0 个 run（网格共 {len(grid)} 个）——禁静默空跑")
    return out


_INT_COLS = ("seed", "patches", "tick", "pop", "d_starv", "d_old", "d_pred", "rd")
_FLOAT_COLS = ("k", "rgm", "saturation", "ms_per_tick", "mean_energy", "mean_gen")


def read_rows(path: "str | Path | None") -> list[dict]:
    """读既有 CSV（**数值列转型** ⇒ 可与新行混算/合并）。文件不存在 ⇒ 空表。"""
    p = Path(path) if path else None
    if p is None or not p.exists():
        return []
    with io.open(p, "r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for c in _INT_COLS:
            if r.get(c) not in (None, ""):
                r[c] = int(float(r[c]))
        for c in _FLOAT_COLS:
            if r.get(c) not in (None, ""):
                r[c] = float(r[c])
    return rows


def _row_key(r: dict) -> tuple[float, int, int, int]:
    # rd 进键（T-F）：开关两臂轨迹不同 ⇒ 行不许互相合并；**旧 CSV 无该列 ⇒ 0**（向后兼容）
    return (round(float(r["rgm"]), 9), int(r["patches"]), int(r["seed"]),
            int(r.get("rd", 0) or 0))


def merge_rows(old: list[dict], new: list[dict],
               resumed: dict[tuple[float, int, int, int], int]) -> list[dict]:
    """续跑合并：非 resumed run 的旧行**全留**；resumed run 只留 `tick ≤ start_tick`；并入新行。

    · `new` 里**本就包含** `prior_rows`（来自旧文件，供 summary 用全史）⇒ 同键以 **new 为准**（覆盖）
    · 但 **`new` 内部**出现同键 ⇒ **fail-loud**（教训同族：云端 20+ 轮续批曾出现
      main_s42 的 16 个**重复行**；重复键必须炸出来，不许静默去重）
    """
    out: dict[tuple[float, int, int, int, int], dict] = {}
    for r in old:
        st = resumed.get(_row_key(r))
        if st is None or int(r["tick"]) <= st:
            out[_row_key(r) + (int(r["tick"]),)] = r
    seen_new: set[tuple[float, int, int, int, int]] = set()
    for r in new:
        kk = _row_key(r) + (int(r["tick"]),)
        if kk in seen_new:
            raise ValueError(f"CSV 合并出现重复行（本次产出内部重复）：{kk}")
        seen_new.add(kk)
        out[kk] = r
    return [out[kk] for kk in sorted(out)]


def make_cfg(seed: int, rows: int, cols: int, pop: int, patches: int,
             subpos: bool, speed_max: float, gain: float, subdiv: int,
             k: float = 1.0, max_count: int = 0,
             rgm: float = 1.195,
             bg_low_prod_frac: float = 0.0,
             bg_low_cap_mult: float = 0.0) -> tuple[SimConfig, dict]:
    c = SimConfig(seed=seed)
    c.world.rows, c.world.cols = rows, cols
    c.resources.distribution = "patchy"
    c.resources.bg_production_zero = True
    # S1 诊断（R217 §三②）：斑块再生倍率是**本探针的唯一扫描变量**
    #   {1.195 = T4 现状, 0.8, 0.6, 0.4} ⇒ 看 K 与饱和度怎么响应
    c.resources.patch_regrowth_mult = float(rgm)
    c.resources.patch_count = patches
    # 🔴 R242 背景低产能带（默认 0/0 ⇒ 走原路径，逐位等价）
    c.resources.bg_low_prod_frac = float(bg_low_prod_frac)
    c.resources.bg_cap_mult = float(bg_low_cap_mult)
    if pop > 0:
        c.population.initial_count = pop
    if max_count > 0:
        c.population.max_count = int(max_count)   # 撞顶则实测=配置读数（T4 口径 30000）
    if subpos:
        c.simulation.use_sim_core = False        # subpos 与 Rust 路径互斥（H3 硬报错）
        c.subpos.enabled = True
        c.subpos.speed_max = float(speed_max)
        c.subpos.speed_gain = float(gain)
        c.subpos.subdiv = int(subdiv)
    # 时间压缩（R205/R206）：`k=1` 逐位不变；k≠1 走项目统一重标（四类量纲表）
    notes = rescale_config(c, k) if abs(k - 1.0) > 1e-12 else {"signals_duration": int(c.signals.duration_ticks)}
    return c, notes


def run_one(seed: int, rows: int, cols: int, pop: int, patches: int, ticks: int,
            sample: int, subpos: bool, speed_max: float, gain: float, subdiv: int,
            max_minutes: float, stop_stable: int = 4, k: float = 1.0,
            max_count: int = 0, rgm: float = 1.195, rd: "bool | None" = None,
            save_every: int = 0, snapshot_dir: "str | Path | None" = None,
            resume_from: "str | Path | None" = None,
            prior_rows: "list[dict] | None" = None) -> tuple[list[dict], dict]:
    """跑一个 run（或**从其快照续跑**）。

    `resume_from` 非空 ⇒ **世界/配置/标签一律以快照为准**（`seed/patches/k/rgm/rd` 就地改写）；
    `ticks` 仍是**绝对目标 tick**；`prior_rows` = 该 run 在快照点之前的旧 CSV 行（用于合并）。
    `save_every > 0` 时每 N tick 写一组检查点（`save_ckpt`），中断最多丢 N 个 tick。
    `rd`（T-F，2026-09-27）：`True/False` = 开/关 `resource_dynamics`（S2 同款臂：bgzero +
    rgm 1.195 + patches 480）；`None` = 未指定（**续跑**时以快照为准，不一致只告警不炸）。
    """
    t0 = time.time()
    if resume_from is not None:
        eng, meta, start_tick = load_ckpt(resume_from)
        seed = int(meta.get("seed", seed))
        patches = int(meta.get("patches", patches))
        k = float(meta.get("k", k))
        rgm = float(meta.get("rgm", rgm))
        rows = int(meta.get("rows", rows))       # 世界尺寸也以快照为准（summary 的 `cells` 用它）
        cols = int(meta.get("cols", cols))
        tag = str(meta.get("tag") or run_tag(rows, cols, patches, seed, k, rgm))
        # 🔴 采样节拍决定 CSV 行网格 ⇒ 续跑必须与出快照那次**同 `--sample`**（否则行网格静默错位）
        if meta.get("sample") is not None and int(meta["sample"]) != int(sample):
            raise ValueError(
                f"续跑采样节拍不一致：快照 meta 记 sample={int(meta['sample'])}，"
                f"命令行 --sample={int(sample)} ⇒ 传同一个 --sample 再来（禁静默错位）")
        # T-F：rd / subpos 一律以**快照配置**为准（引擎态自带）；CLI 开关仅对"新跑"生效
        rd_on = bool(getattr(eng.config.resource_dynamics, "enabled", False))
        subpos = bool(getattr(eng.config.subpos, "enabled", False))
        speed_max = float(getattr(eng.config.subpos, "speed_max", speed_max))
        gain = float(getattr(eng.config.subpos, "speed_gain", gain))
        subdiv = int(getattr(eng.config.subpos, "subdiv", subdiv))
        if rd is not None and bool(rd) != rd_on:
            print(f"  ⚠️ --rd {'on' if rd else 'off'} 与快照配置（rd {'开' if rd_on else '关'}）"
                  f"不一致 ⇒ **以快照为准**（本开关对续跑不生效）", file=sys.stderr)
        rows_out: list[dict] = list(prior_rows or [])
        print(f"  ↻ 续跑：{tag} @ tick {start_tick} → {ticks}"
              f"（世界/斑块/k/rgm/rd **以快照为准**；本段墙钟预算 {max_minutes:g} min）", flush=True)
    else:
        cfg, notes = make_cfg(seed, rows, cols, pop, patches, subpos, speed_max, gain,
                              subdiv, k, max_count, rgm)
        rd_on = bool(rd)
        if rd_on:
            # T-F：rd 与 subpos 同档（S2 主臂）——`bgzero` 死斑块保身份（R226 修复①）
            cfg.resource_dynamics.enabled = True
        eng = SphereEngine(cfg)
        apply_post_build(eng, notes)             # 构造后项（信号寿命 ÷k）
        start_tick = 0
        rows_out = []
        tag = run_tag(rows, cols, patches, seed, k, rgm, rd_on)
    if save_every and snapshot_dir is None:
        raise ValueError("--save-every 需要 --snapshot-dir（快照写哪儿）")
    prod = eng.resources._capacity > 0
    n_prod = int(prod.sum())
    cap_sum = float(eng.resources._capacity[prod].sum())
    meta_doc = {"tag": tag, "rows": rows, "cols": cols, "patches": patches, "seed": seed,
                "k": k, "rgm": rgm, "subpos": bool(subpos), "speed_max": speed_max,
                "gain": gain, "subdiv": subdiv, "max_count": max_count or None,
                "rd": bool(rd_on),
                "sample": int(sample),
                "ticks_target": int(ticks), "save_every": int(save_every),
                "config_fingerprint": eng.config.fingerprint(),
                "written_at": time.strftime("%Y-%m-%d %H:%M:%S")}

    t_wall0 = time.time()
    t_slice = time.perf_counter()
    last = start_tick
    stop_reason = "tick 用尽"
    for t in range(start_tick + 1, ticks + 1):
        eng.step()
        if eng.world.n_cells and len(eng._flat) == 0:
            stop_reason = "灭绝"
        if t % sample == 0 or t == ticks or stop_reason == "灭绝":
            dt_ms = (time.perf_counter() - t_slice) / max(t - last, 1) * 1e3
            last = t
            t_slice = time.perf_counter()
            N = int(len(eng._flat))
            stock = float(eng.resources._grid[prod].sum())
            # 🔴 R217 §三①：引擎**已有**死因账本（`death_cause_totals()`，与 history_limit 无关），
            #   但本探针此前**没读它** ⇒ 平台期"K 被什么顶上"无从判断。
            #   这里记**累计**值 ⇒ 分析侧用相邻采样差得到"窗口内死因构成"（平台期口径）。
            _dc = eng.death_cause_totals()
            _dv = {getattr(k, "name", str(k)): int(v) for k, v in _dc.items()}
            rows_out.append({
                "seed": seed, "patches": patches, "k": k, "rgm": rgm, "rd": int(rd_on),
                "tick": t, "pop": N,
                "d_starv": _dv.get("STARVATION", 0),
                "d_old": _dv.get("OLD_AGE", 0),
                "d_pred": _dv.get("PREDATION", 0),
                "saturation": (stock / cap_sum if cap_sum > 0 else float("nan")),
                "ms_per_tick": dt_ms,
                "mean_energy": float(eng._energy[:N].mean()) if N else float("nan"),
                "mean_gen": float(eng._generation[:N].max()) if N else float("nan"),
            })
            # ETA 只用**实测样本**推（R189：禁瞬时速率外推；此处用刚测完这一段的 ms/tick）
            eta_min = (ticks - t) * dt_ms / 1e3 / 60.0
            print("  " + _bar(t, ticks, f"t={t:<6} N={N:<7} 饱和度={stock / max(cap_sum, 1e-9):.3f}"
                                      f" {dt_ms:6.2f} ms/tick  ETA {eta_min:5.1f} min"), flush=True)
            # 早停：连续 stop_stable 个采样点相对变化 < 3% ⇒ 已到平台
            if stop_stable > 0 and len(rows_out) >= stop_stable + 1:
                _tail = [r["pop"] for r in rows_out[-(stop_stable + 1):]]
                _ok = all(abs(_tail[k + 1] - _tail[k]) <= 0.03 * max(_tail[k], 1)
                          for k in range(len(_tail) - 1))
                if _ok:
                    stop_reason = f"平台期（连续 {stop_stable} 段 <3%）"
                    break
            if stop_reason == "灭绝":
                break
        # 快照节拍（**独立于采样节拍**，R221：任意时刻中断最多丢 `save_every` 个 tick）
        if save_every and t % save_every == 0:
            save_ckpt(eng, snapshot_dir, tag, meta_doc)
        if (time.time() - t_wall0) / 60.0 > max_minutes:
            stop_reason = f"超时 {max_minutes} 分钟"
            break

    pop_tail = [r["pop"] for r in rows_out[len(rows_out) * 4 // 5:]] or [0]
    ms_tail = [r["ms_per_tick"] for r in rows_out[len(rows_out) * 4 // 5:]] or [0.0]
    sat_tail = [r["saturation"] for r in rows_out[len(rows_out) * 4 // 5:]]
    # 平台判据（T4 口径）：**末 3 个采样点**两两相对变化 ≤ 3%
    # 🔴 R217 §三①：**平台期**死因构成 = 末 20% 采样窗口内的增量（累计量之差）
    _tw = rows_out[len(rows_out) * 4 // 5:]
    if len(_tw) < 2:                       # 采样点太少（短跑冒烟）⇒ 退化为全程差
        _tw = rows_out[:]
    if len(_tw) >= 2:
        _d0, _d1 = _tw[0], _tw[-1]
        _win = {k: int(_d1[k]) - int(_d0[k]) for k in ("d_starv", "d_old", "d_pred")}
    else:
        _win = {"d_starv": 0, "d_old": 0, "d_pred": 0}
    _win_n = sum(_win.values())
    death_tail = {
        "n": _win_n,
        "starvation": _win["d_starv"],
        "old_age": _win["d_old"],
        "predation": _win["d_pred"],
        "share_starvation": round(_win["d_starv"] / _win_n, 4) if _win_n else None,
        "share_old_age": round(_win["d_old"] / _win_n, 4) if _win_n else None,
        "share_predation": round(_win["d_pred"] / _win_n, 4) if _win_n else None,
    }
    _cum_last = rows_out[-1] if rows_out else {"d_starv": 0, "d_old": 0, "d_pred": 0}
    tail3 = [int(r["pop"]) for r in rows_out[-3:]]
    platform_ok = (len(tail3) == 3 and all(
        abs(tail3[i + 1] - tail3[i]) <= 0.03 * max(tail3[i], 1) for i in range(2)))
    summary = {
        "patches": patches, "seed": seed, "cells": rows * cols, "k": k, "rgm": rgm,
        "rd": bool(rd_on),
        "death_cause_tail": death_tail,
        "deaths_cum_starvation": int(_cum_last["d_starv"]),
        "deaths_cum_old_age": int(_cum_last["d_old"]),
        "deaths_cum_predation": int(_cum_last["d_pred"]),
        "productive_cells": n_prod,
        "K_extrapolated": round(K_PER_CELL * n_prod, 1),
        "K_measured": int(np.median(pop_tail)),
        "pop_max": max((r["pop"] for r in rows_out), default=0),
        "pop_final": pop_tail[-1],
        "tail3_pops": tail3,
        "platform_reached": bool(platform_ok),
        "saturation_tail": round(float(np.median(sat_tail)), 4) if sat_tail else None,
        "ms_per_tick_tail": round(float(np.median(ms_tail)), 2),
        "max_gen": max((r["mean_gen"] for r in rows_out if r["mean_gen"] == r["mean_gen"]),
                       default=0),
        "ticks_done": rows_out[-1]["tick"] if rows_out else 0,
        "wall_s": round(time.time() - t0, 1),
        "stop": stop_reason,
    }
    return rows_out, summary


def _print_run_summary(s: dict) -> None:
    """单 run 小结行（原样保留既有格式；续跑/正常两路共用）。"""
    print(f"    ⇒ 产能格 {s['productive_cells']}｜外推 K {s['K_extrapolated']}"
          f"｜**实测 K {s['K_measured']}**（峰值 {s['pop_max']}）"
          f"｜饱和度 {s['saturation_tail']}｜{s['ms_per_tick_tail']} ms/tick"
          f"｜{s['wall_s']}s｜{s['stop']}\n", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="稳态 K 实测（世界放大后能养活多少个体）")
    ap.add_argument("--rows", type=int, default=480)
    ap.add_argument("--cols", type=int, default=960)
    ap.add_argument("--patches", default="30,60,120")
    ap.add_argument("--pop", type=int, default=2000)
    ap.add_argument("--ticks", type=int, default=6000)
    ap.add_argument("--sample", type=int, default=250)
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--subpos", choices=("off", "on"), default="off")
    ap.add_argument("--speed-max", type=float, default=0.25)
    ap.add_argument("--gain", type=float, default=None,
                    help="速度增益；**默认 None ⇒ = speed_max**（R204 §二 / R205 定档 `gain = v_max`，"
                         "构造保证顶格%% ≡ 0）")
    ap.add_argument("--subdiv", type=int, default=SUBDIV_STD,
                    help=f"速度档位细分（定档 {SUBDIV_STD}，R213 §四；改动须回板）")
    ap.add_argument("--max-minutes", type=float, default=25.0)
    ap.add_argument("--k", type=float, default=1.0,
                    help="时间压缩倍率（R205 定档 k=2.5 ⇒ 昼夜 960）；k=1 逐位不变")
    ap.add_argument("--max-count", type=int, default=0,
                    help="population.max_count 覆盖（0=用配置默认 5000）；T4 口径 30000——"
                         "撞顶则实测变成配置读数")
    ap.add_argument("--stop-stable", type=int, default=4,
                    help="连续 N 个采样点相对变化 < 3 个百分点即判平台并早停（0=关）")
    ap.add_argument("--patch-regrowth-mult", dest="rgm", default="1.195",
                    help="S1 诊断：斑块再生倍率扫描（逗号分隔）。"
                         "默认 1.195 = T4 现状；建议 1.195,0.8,0.6,0.4（R217 §三②）")
    ap.add_argument("--out", default="results/steady_k_probe.csv")
    # ── R221 §四：快照 / 续跑 / 分片 ─────────────────────────────────────────
    ap.add_argument("--save-every", type=int, default=0,
                    help="每 N tick 存一次检查点（**0=关**；长批建议 500）")
    ap.add_argument("--snapshot-dir", default="_rerun_logs/snap",
                    help="检查点目录（默认 `_rerun_logs/snap`，已在 .gitignore 白名单）")
    ap.add_argument("--resume-from", default=None,
                    help="从快照续跑：单个 `*.snapshot.npz` 或一个目录（逐个续）；"
                         "**世界/斑块/k/rgm 以快照为准**，`--ticks` 仍是绝对目标")
    ap.add_argument("--shard", default=None,
                    help="分片：`K/N`（共 N 份取第 K 份）或显式 `p:sd[,rgm:p:sd…]` ⇒ 分片到多机")
    ap.add_argument("--rd", choices=("on", "off"), default=None,
                    help="resource_dynamics（S2 同款臂：bgzero + rgm 1.195 + patches 480）；"
                         "**默认 None ⇒ 不碰配置**（续跑时一律以快照配置为准，不一致仅告警）")
    a = ap.parse_args()

    patch_list = [int(x) for x in a.patches.split(",") if x.strip()]
    seeds = [int(x) for x in a.seeds.split(",") if x.strip()]
    rgm_list = [float(x) for x in str(a.rgm).split(",") if x.strip()]
    gain = a.speed_max if a.gain is None else float(a.gain)     # 定档：gain = v_max
    r_ladder = a.speed_max * a.subdiv

    print(f"== 稳态 K 探针：{a.rows}x{a.cols}（{a.rows * a.cols:,} 格）"
          f"，斑块 {patch_list}，初始 {a.pop}，{a.ticks} tick，"
          f"subpos={a.subpos}，k={a.k:g}，max_count={a.max_count or '默认'}，"
          f"rd={a.rd or '默认(关)'}，seed {seeds} ==")
    print(f"   速度：v_max={a.speed_max} gain={gain} subdiv={a.subdiv}"
          f"（R = v_max×subdiv = {r_ladder:g}）")
    if a.subpos == "on" and r_ladder < 5:
        print(f"   ⚠️ R = {r_ladder:g} < 5 ⇒ 档数不足（量化悬崖，设计稿 §12.1）", file=sys.stderr)
    if a.save_every:
        print(f"   快照：每 {a.save_every} tick ⇒ {a.snapshot_dir}/<tag>.snapshot.npz"
              f"（+ `.rngstate.pkl` + `.meta.json`）")
    if a.resume_from:
        print(f"   续跑模式：--resume-from {a.resume_from}（网格/分片参数**忽略**，以快照为准）")
    print(f"   旧标定外推公式：K ≈ {K_PER_CELL:.2f} × 产能格\n")

    grid_all = [(rgm, p, sd) for rgm in rgm_list for p in patch_list for sd in seeds]
    out_path = Path(a.out) if a.out else None
    old_rows = read_rows(out_path)               # 既有 CSV（续跑合并用；无则空表）
    all_rows: list[dict] = []
    summaries: list[dict] = []
    resumed_keys: dict[tuple[float, int, int], int] = {}   # (rgm, p, sd) → start_tick

    if a.resume_from:
        snaps = _collect_snapshots(a.resume_from)
        print(f"=== 续跑 {len(snaps)} 组快照 ⇒ 目标 tick {a.ticks} ===")
        for snap in snaps:
            _e0, meta0, start_tick = load_ckpt(snap)          # 只读标签（run_one 会再读一次）
            sd, p = int(meta0["seed"]), int(meta0["patches"])
            kk = float(meta0.get("k", a.k))
            rr = float(meta0.get("rgm", 1.195))
            _wr, _wc = int(meta0.get("rows", a.rows)), int(meta0.get("cols", a.cols))
            if (p, sd) not in [(g[1], g[2]) for g in grid_all] or abs(kk - a.k) > 1e-9 \
                    or abs(rr - rgm_list[0]) > 1e-9 or (_wr, _wc) != (a.rows, a.cols):
                print(f"   ⚠️ 快照 `{snap.name}` 身份（{_wr}x{_wc}/p={p}/s={sd}/k={kk:g}/rgm={rr:g}）"
                      f"与命令行（{a.rows}x{a.cols}/p={patch_list}/s={seeds}/k={a.k:g}/rgm={rgm_list}）"
                      f"不一致 ⇒ **以快照为准**", file=sys.stderr)
            _rd0 = 1 if bool(getattr(_e0.config.resource_dynamics, "enabled", False)) else 0
            prior = [r for r in old_rows
                     if _row_key(r) == (round(rr, 9), p, sd, _rd0)
                     and int(r["tick"]) <= start_tick]
            print(f"--- 续跑 rgm {rr:g} / 斑块 {p} / seed {sd}"
                  f"{'（rd 开）' if _rd0 else ''} ---", flush=True)
            rows_out, s = run_one(sd, a.rows, a.cols, a.pop, p, a.ticks, a.sample,
                                  a.subpos == "on", a.speed_max, gain, a.subdiv,
                                  a.max_minutes, a.stop_stable, kk, a.max_count, rr,
                                  rd=(a.rd == "on" if a.rd else None),
                                  save_every=a.save_every, snapshot_dir=a.snapshot_dir,
                                  resume_from=snap, prior_rows=prior)
            all_rows.extend(rows_out)
            summaries.append(s)
            resumed_keys[(round(rr, 9), p, sd, _rd0)] = start_tick
            _print_run_summary(s)
    else:
        grid = select_runs(grid_all, a.shard)
        if a.shard:
            print(f"=== 分片 --shard {a.shard}：本机负责 **{len(grid)}/{len(grid_all)}** 个 run"
                  f"（tag = 快照文件名，可直接抄给 [所有者] 做分机声明）===")
            for _rgm, _p, _sd in grid:
                print(f"   · rgm {_rgm:g} / 斑块 {_p} / seed {_sd}"
                      f"  ⇒  {run_tag(a.rows, a.cols, _p, _sd, a.k, _rgm)}")
            print()
        for rgm, p, sd in grid:
            print(f"--- rgm {rgm:g} / 斑块 {p} / seed {sd} ---", flush=True)
            rows_out, s = run_one(sd, a.rows, a.cols, a.pop, p, a.ticks, a.sample,
                                  a.subpos == "on", a.speed_max, gain, a.subdiv,
                                  a.max_minutes, a.stop_stable, a.k, a.max_count, rgm,
                                  rd=(a.rd == "on" if a.rd else None),
                                  save_every=a.save_every, snapshot_dir=a.snapshot_dir)
            all_rows.extend(rows_out)
            summaries.append(s)
            _print_run_summary(s)

    print("=" * 100)
    print(f"{'rgm':>6}{'斑块数':>7}{'产能格':>9}{'外推K':>9}{'实测K':>9}{'比值':>8}"
          f"{'峰值':>8}{'饱和度':>9}{'ms/tick':>9}{'墙钟s':>8}{'代数':>6}{'平台':>5}"
          f"{'饿/老/捕':>14}  终止")
    print("-" * 100)
    for s in summaries:
        ratio = (s["K_measured"] / s["K_extrapolated"]) if s["K_extrapolated"] else float("nan")
        _dt = s["death_cause_tail"]
        _sh = (f"{_dt['share_starvation']:.2f}/{_dt['share_old_age']:.2f}/"
               f"{_dt['share_predation']:.2f}" if _dt["n"] else "n/a")
        print(f"{s['rgm']:>6g}{s['patches']:>7}{s['productive_cells']:>9}{s['K_extrapolated']:>9.0f}"
              f"{s['K_measured']:>9}{ratio:>8.2f}{s['pop_max']:>8}"
              f"{s['saturation_tail']:>9}{s['ms_per_tick_tail']:>9.2f}{s['wall_s']:>8.0f}"
              f"{s['max_gen']:>6.0f}{'是' if s['platform_reached'] else '否':>4}"
              f"{_sh:>14}  {s['stop']}")

    out = out_path
    if out and all_rows:
        out.parent.mkdir(parents=True, exist_ok=True)
        # 续跑 ⇒ 与既有 CSV **合并**（该 run 的旧行只留 tick ≤ 快照点）；否则沿用**覆盖**语义
        merged = merge_rows(old_rows, all_rows, resumed_keys) if resumed_keys else all_rows
        with io.open(out, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(merged)
        _extra = f"；合并后 {len(merged)} 行（保留其他 run 的旧行）" if resumed_keys else ""
        print(f"\n明细已写入 {out}（本次 {len(all_rows)} 行{_extra}）")
        # 伴生 summary.json：供 `experiments/batch_runner.py` 判 done（`result.final_N` + 跑满
        # `switches.ticks_target`）；单 run 调用时顶层即该 run，多 run 时附全量于 `runs`
        last = summaries[-1]
        doc = {
            "switches": {"ticks_target": int(a.ticks), "rows": a.rows, "cols": a.cols,
                         "patches": last["patches"], "k": last["k"], "seeds": seeds,
                         "patch_regrowth_mult": rgm_list,
                         "subpos": a.subpos, "speed_max": a.speed_max, "gain": a.gain,
                         "subdiv": a.subdiv, "max_count": a.max_count or None,
                         "rd": bool(last.get("rd")),      # T-F：rd 开关（以快照/本跑真值为准）
                         # R221 §四：快照/续跑/分片（自证——回板要能看出这批是不是分片的）
                         "save_every": a.save_every, "shard": a.shard,
                         "resume_from": str(a.resume_from) if a.resume_from else None},
            "result": {"final_N": int(last["pop_final"]), "final_tick": int(last["ticks_done"]),
                       "platform_reached": bool(last["platform_reached"]),
                       "K_measured": last["K_measured"],
                       "K_extrapolated": last["K_extrapolated"],
                       "tail3_pops": last["tail3_pops"], "max_gen": last["max_gen"],
                       "saturation_tail": last["saturation_tail"],
                       "death_cause_tail": last["death_cause_tail"],
                       "ms_per_tick_tail": last["ms_per_tick_tail"], "stop": last["stop"]},
            "runs": summaries,
        }
        sp = out.with_suffix(".summary.json")
        if resumed_keys and sp.exists():     # 续跑 ⇒ 其他 run 的 summary **保留**（只换被续的那几个）
            try:
                old_doc = json.loads(sp.read_text(encoding="utf-8"))
                _by = {(round(float(r["rgm"]), 9), int(r["patches"]), int(r["seed"])): r
                       for r in old_doc.get("runs", [])}
                for r in summaries:
                    _by[(round(float(r["rgm"]), 9), int(r["patches"]), int(r["seed"]))] = r
                doc["runs"] = sorted(_by.values(),
                                     key=lambda r: (round(float(r["rgm"]), 9),
                                                    int(r["patches"]), int(r["seed"])))
            except (ValueError, KeyError, TypeError) as exc:
                print(f"⚠️ 旧 summary 解析失败（{exc!r}）⇒ 本次只写本次的 runs（不静默）",
                      file=sys.stderr)
        with io.open(sp, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, ensure_ascii=False, indent=2)
        print(f"summary 已写入 {sp}")

    print("\n读法：")
    print("  · **实测 K / 外推 K < 1** ⇒ 评审说得对，线性外推高估了（世界放大后食物找不到）")
    print("  · **资源饱和度接近 1** ⇒ 瓶颈是「可达性」不是「产量」⇒ 加食物不涨种群"
          "（注意：饱和度 ≠ 被吃掉的比例）")
    print("  · ms/tick 随 N 增长（次线性）⇒ 这一列是「生物变多会不会影响性能」的答案")


if __name__ == "__main__":
    main()
