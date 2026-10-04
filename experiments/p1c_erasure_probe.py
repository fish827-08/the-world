#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P1-c 擦除集探针（R358-T7）：反事实"被擦除集"距离/可直读率（纯观测）。

判据来源：`docs/设计文档/设计-R358-T1-根因判别-H1H2-20261003.md` §2.2/§3（砚 D4 复核、天平锁定后生效）。
判别量 = 反事实**被擦除集**：探针侧每体独立 FIFO 踩点日志（写入规则与引擎 `:3577-3580`
逐字同式），每采样取"最近 16/64 条中、**除最近 4 条之外**"的条目
（= 若记忆容量 4→16/64，本会保留而现被擦除的格），测它们到"该体当前格"的球面距离
与可直读率（口径 A/B 与 P1-c 同）。

- 纯观测：只读引擎数组（`_id/_flat/resources._grid/_capacity/_mem_ptr`）＋探针侧日志；
  不调引擎写方法、不消费主 RNG ⇒ 轨迹不受扰动。
- 相位披露（设计 §1.3-②）：引擎写入在步 4.4（移动**前**位置），探针在 `step()` 返回后
  观测（移动**后**）⇒ ≤1 tick 相位差；诊断列 `eng_wr_ticks`（引擎 `_mem_ptr` 前进次数）
  vs `probe_rich_ticks`（探针检出 ≥1 事件 tick 数），偏差超容差 ⇒ fail-loud 停跑。
- fail-loud 前置读回（R6.6）：alphabet=="8" / memory_v2 False / rd 关 / use_sim_core False /
  patches 读回==请求 / `_id` 严格升序；非判别装置（≠480×960/1700）须显式 `--smoke`。
- `--dm-guard <stored csv>`：跑完后共享列逐采样点与 P1-c 产物核对（round 4；
  排除 `ms_per_tick_window`），任一不等 ⇒ SystemExit（seed 207 必开，T1 设计 §2.2-5）。

用法（项目根目录）：
    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m experiments.p1c_erasure_probe \
        --device s2 --seed 207 --ticks 3000 --sample 100 \
        --dm-guard ../the-world/the-world-data/p1c/p1c_s207_t3000_v2.csv \
        --out _trash_local/p1c_erase/p1c_erase_s207_t3000.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

# R98 纪律：非 ASCII 输出在 GBK 控制台会 rc=1 假失败，入口统一兜底。
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from simulation.sphere_engine import SphereEngine               # noqa: E402
from simulation.config import FOOD_RICH_LEVEL                   # noqa: E402
from experiments.steady_k_probe import make_cfg, apply_post_build  # noqa: E402
from experiments.p1c_distance_probe import measure as measure_p1c  # noqa: E402
from world.subpos import sphere_dist_rows                       # noqa: E402
from tools.device_presets import add_device_arg, resolve_device  # noqa: E402

# subpos-family 常量（与 P1-c 探针逐位一致；装置字段走 CLI）
SPEED_MAX, GAIN, SUBDIV, K_TC, MAX_COUNT = 0.125, 0.125, 80, 2.5, 30000

_KEEP = 4            # 现容量（最近 4 条 = 引擎真保留 ⇒ 不计入擦除集）
_K16, _K64 = 16, 64  # 反事实容量窗
_S2 = (480, 960, 1700)   # 判别装置指纹（s2）
_DM_EXCLUDE = ("ms_per_tick_window",)

EPS = 1e-9


def _pct(x: np.ndarray, q: float):
    return float(np.percentile(x, q)) if x.size else None


def _frac(mask: np.ndarray):
    return float(mask.mean()) if mask.size else None


def _assert_ids_ascending(ids: np.ndarray, where: str = "p1c_erase") -> None:
    """引擎不变量：`_id` 自 `np.arange` + 追增 ⇒ 严格升序（T1 设计 §2.2-3）。

    打破 ⇒ 按 `_id` 重映射环行不可信 ⇒ fail-loud（不回退、回报）。
    """
    if ids.size > 1 and not bool((np.diff(ids) > 0).all()):
        raise RuntimeError(
            f"🔴 {where}：_id 非严格升序 ⇒ 环日志按 id 归属不可信（引擎不变量被打破）。")


def _read_moore(world, cur: np.ndarray, mem: np.ndarray) -> np.ndarray:
    """口径 B：记忆格 ∈ 当前格 Moore 8 邻（极点格=相邻纬度带整行）。

    与 `p1c_distance_probe.measure()` 逐字同式（含极点特判），保证两探针口径一致。
    """
    cur = np.asarray(cur, np.int64)
    mem = np.asarray(mem, np.int64)
    cur_row = cur // world.cols
    is_pole = (cur_row == 0) | (cur_row == world.rows - 1)
    readable = (mem == cur)
    normal = ~is_pole
    if normal.any():
        nb = world._nb_table[cur[normal]]
        readable[normal] = readable[normal] | (nb == mem[normal, None]).any(axis=1)
    for i in np.flatnonzero(is_pole):
        band = world._pole_nb[0 if cur_row[i] == 0 else 1]
        readable[i] = readable[i] or bool((band == mem[i]).any())
    return readable


class _RingLog:
    """按 `_id` 键控的 FIFO 踩点日志（`deque(maxlen=depth)`，逐事件追加）。

    窗口语义：`window(aid, k)` = 环内"最近 k 条中、除最近 `_KEEP` 条之外"的条目
    （k=16 ⇒ ≤12 条/体；k=64 ⇒ ≤60 条/体）；不足 5 条或不存在 ⇒ 空。
    ⚠️ 反事实口径披露（T1 §1.3-①）：每体独立 FIFO，非"引擎全局指针环的 16 变体"。
    """

    __slots__ = ("depth", "_d")

    def __init__(self, depth: int = _K64):
        if int(depth) <= _KEEP:
            raise ValueError(f"ring depth={depth} 必须 > {_KEEP}")
        self.depth = int(depth)
        self._d: dict[int, deque] = {}

    def add(self, aid: int, cell: int) -> None:
        r = self._d.get(aid)
        if r is None:
            r = deque(maxlen=self.depth)
            self._d[aid] = r
        r.append(int(cell))

    def window(self, aid: int, k: int) -> list[int]:
        r = self._d.get(aid)
        if not r:
            return []
        return list(r)[-k:-_KEEP]

    def n_events(self, aid: int) -> int:
        r = self._d.get(aid)
        return len(r) if r else 0


def _dm_guard_mismatches(new_rows: list[dict], stored_path) -> list[str]:
    """dm-guard：新产物与 P1-c 存档**共享列**逐采样点核对（空列表 = 通过）。

    口径（T1 §2.2-5）：排除 `ms_per_tick_window`；双侧 `round(,4)`；None/空串等价；
    行数不一致、tick 缺失、任一列不等 ⇒ 逐条报告。
    """
    stored_path = Path(stored_path)
    if not stored_path.exists():
        return [f"🔴 dm-guard：存档不存在 {stored_path}"]
    with open(stored_path, newline="", encoding="utf-8") as fh:
        rd = csv.DictReader(fh)
        stored = list(rd)
        scols = [c for c in (rd.fieldnames or []) if c not in _DM_EXCLUDE]
    if not scols:
        return [f"🔴 dm-guard：{stored_path} 无共享列可核对"]
    out: list[str] = []
    if len(new_rows) != len(stored):
        out.append(f"🔴 dm-guard：采样行数 {len(new_rows)} != 存档 {len(stored)}")
    by_tick = {str(r.get("tick", "")).strip(): r for r in new_rows}

    def _norm(v):
        if v is None:
            return None
        s = str(v).strip()
        if s == "":
            return None
        try:
            return round(float(s), 4)
        except ValueError:
            return s

    for row in stored:
        tk = str(row.get("tick", "")).strip()
        nr = by_tick.get(tk)
        if nr is None:
            out.append(f"🔴 dm-guard：存档 tick={tk} 在新产物中缺失")
            continue
        for c in scols:
            a, b = _norm(row.get(c)), _norm(nr.get(c))
            if a != b:
                out.append(f"🔴 dm-guard tick={tk} {c}: 存档={a!r} 新={b!r}")
    return out

class _ErasureTracker:
    """每 tick（非采样 tick 也更新）：事件检测 + 环追加 + 引擎写 tick 计数。

    事件检测与引擎 `:3577-3580` 逐字同式：`grid[cur] > FOOD_RICH_LEVEL * capacity[cur]`
    （比较符 `>`）；命中者把**当前格** append 进该体 FIFO 环。
    """

    __slots__ = ("rings", "_prev_ptr", "wr_ticks", "rich_ticks")

    def __init__(self, eng, depth: int = _K64):
        self.rings = _RingLog(depth)
        self._prev_ptr = int(eng._mem_ptr)
        self.wr_ticks = 0     # 窗内 `_mem_ptr` 前进次数 = 引擎写 tick 数
        self.rich_ticks = 0   # 窗内探针检出 ≥1 事件的 tick 数

    def observe(self, eng) -> None:
        ids = np.asarray(eng._id)
        _assert_ids_ascending(ids)
        cur = np.asarray(eng._flat)[:ids.size]
        g, cap = eng.resources._grid, eng.resources._capacity
        rich = g[cur] > FOOD_RICH_LEVEL * cap[cur]
        idx = np.flatnonzero(rich)
        if idx.size:
            self.rich_ticks += 1
            for i in idx:
                self.rings.add(int(ids[i]), int(cur[i]))
        ptr = int(eng._mem_ptr)
        if ptr != self._prev_ptr:
            self.wr_ticks += 1
        self._prev_ptr = ptr

    def take_window(self) -> tuple[int, int]:
        wr, rich = self.wr_ticks, self.rich_ticks
        self.wr_ticks = 0
        self.rich_ticks = 0
        return wr, rich


def _collect_window(eng, rings: _RingLog, k: int):
    """存活个体 × 环窗(k) → 拼接 (cells, cur) 数组（空 ⇒ 空数组）。"""
    ids = np.asarray(eng._id)
    flat = np.asarray(eng._flat)[:ids.size]
    cells: list[int] = []
    curs: list[int] = []
    for i in range(ids.size):
        w = rings.window(int(ids[i]), k)
        if w:
            cells.extend(w)
            curs.extend([int(flat[i])] * len(w))
    if not cells:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    return np.asarray(cells, np.int64), np.asarray(curs, np.int64)


def _dist_and_readable(eng, cells: np.ndarray, curs: np.ndarray):
    """口径 A 球面距离 + 口径 B 可直读（与 p1c measure() 同式）。"""
    if cells.size == 0:
        return np.empty(0, np.float64), np.empty(0, bool)
    world = eng.world
    c_r, c_c = world.flat_to_rc(curs)
    m_r, m_c = world.flat_to_rc(cells)
    d = np.asarray(sphere_dist_rows(
        c_r + 0.5, c_c + 0.5, m_r + 0.5, m_c + 0.5, world), dtype=np.float64)
    return d, _read_moore(world, curs, cells)


def measure_erase(eng, tracker: _ErasureTracker) -> dict:
    """采样 tick：K16/K64 环窗统计（K16 主判、K64 预注册次判；T1 §2.2-2）。"""
    c16, u16 = _collect_window(eng, tracker.rings, _K16)
    c64, u64 = _collect_window(eng, tracker.rings, _K64)
    d, rd = _dist_and_readable(
        eng, np.concatenate([c16, c64]), np.concatenate([u16, u64]))
    q = len(c16)
    d16_all, rd16_all = d[:q], rd[:q]
    d16_pos, rd16_pos = d16_all[d16_all > EPS], rd16_all[d16_all > EPS]
    d64, rd64 = d[q:], rd[q:]
    d64_pos, rd64_pos = d64[d64 > EPS], rd64[d64 > EPS]
    return {
        # pea_*：K16 窗全部条目（含 d==0）
        "pea_n": int(d16_all.size),
        "pea_d_med": _pct(d16_all, 50), "pea_d_p90": _pct(d16_all, 90),
        "pea_frac_d0": _frac(d16_all <= EPS),
        "pea_frac_dle1": _frac(d16_all <= 1.0 + EPS),
        "pea_frac_read_moore": _frac(rd16_all),
        # pep_*：K16 窗 d>0 子集（🔴 主判据，与 P1-c pos_* 同口径）
        "pep_n": int(d16_pos.size),
        "pep_d_med": _pct(d16_pos, 50), "pep_d_p90": _pct(d16_pos, 90),
        "pep_frac_dle1": _frac(d16_pos <= 1.0 + EPS),
        "pep_frac_read_moore": _frac(rd16_pos),
        # pep64_*：K64 窗 d>0 子集（预注册次判）
        "pep64_n": int(d64_pos.size),
        "pep64_d_med": _pct(d64_pos, 50), "pep64_d_p90": _pct(d64_pos, 90),
        "pep64_frac_dle1": _frac(d64_pos <= 1.0 + EPS),
        "pep64_frac_read_moore": _frac(rd64_pos),
        # 池化原料（采样器 pop）
        "_pea_pool": d16_all, "_pea_rd": rd16_all,
        "_pep_pool": d16_pos, "_pep_rd": rd16_pos,
        "_pep64_pool": d64_pos, "_pep64_rd": rd64_pos,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="P1-c 擦除集探针（反事实被擦除集；纯观测）")
    ap.add_argument("--seed", type=int, default=207)
    ap.add_argument("--ticks", type=int, default=3000)
    ap.add_argument("--sample", type=int, default=100)
    ap.add_argument("--out", required=True, help="逐采样 CSV 路径（必填）")
    ap.add_argument("--rows", type=int, default=480)
    ap.add_argument("--cols", type=int, default=960)
    ap.add_argument("--patches", type=int, default=1700)
    ap.add_argument("--pop", type=int, default=10000)
    ap.add_argument("--rgm", type=float, default=1.195)
    ap.add_argument("--bg-low-prod-frac", type=float, default=0.0)
    ap.add_argument("--bg-low-cap-mult", type=float, default=0.0)
    ap.add_argument("--sparse-fields", action="store_true",
                    help="与 smell_rd_01 A0 完全同旗标复跑用（本测量与稀疏正交）")
    ap.add_argument("--ring-depth", type=int, default=_K64,
                    help=f"FIFO 环深（默认 {_K64}；K64 窗需要 ≥64）")
    ap.add_argument("--dm-guard", default=None,
                    help="P1-c 存档 CSV 路径：共享列逐采样点核对（不等即停）")
    ap.add_argument("--smoke", action="store_true",
                    help="非判别装置（≠480×960/1700）时显式声明（仪表冒烟）")
    add_device_arg(ap)                      # R5.6：显式 --device（预设档防呆）
    a = ap.parse_args()
    resolve_device(a, sys.argv[1:])

    if a.ring_depth <= _KEEP:
        print(f"🔴 --ring-depth={a.ring_depth} 必须 > {_KEEP}。中止。", file=sys.stderr)
        return 2

    c, notes = make_cfg(
        a.seed, a.rows, a.cols, a.pop, a.patches, True,
        SPEED_MAX, GAIN, SUBDIV, K_TC, MAX_COUNT, a.rgm,
        bg_low_prod_frac=a.bg_low_prod_frac,
        bg_low_cap_mult=a.bg_low_cap_mult,
    )
    c.simulation.use_sim_core = False       # 纯 Python（本机无 .so；subpos 亦强制）
    c.simulation.sparse_fields = bool(a.sparse_fields)
    c.resource_dynamics.enabled = False     # 🔴 rd 关（M1 用 rd 关装置）
    c.signal_alphabet = "8"                 # 🔴 P1-a 前提（mem_bit 通路）
    # memory_v2 / info_structure.enabled 保持默认（False / False = A0 口径）

    e = SphereEngine(c)
    apply_post_build(e, notes)

    # ---- fail-loud 读回（R6.6 仪表先冒烟 + T1 §2.2-4 扩充）
    rb = {
        "signal_alphabet": getattr(e, "_alpha", None),
        "memory_v2": bool(getattr(c.info_structure, "memory_v2", None)),
        "info_structure_enabled": bool(getattr(c.info_structure, "enabled", None)),
        "rd_enabled": bool(getattr(getattr(e, "_rd", None), "enabled", False)),
        "use_sim_core": bool(getattr(c.simulation, "use_sim_core", True)),
        "subpos_enabled": bool(getattr(c.subpos, "enabled", False)),
        "sparse_cli": bool(a.sparse_fields),
        "rows": int(e.world.rows), "cols": int(e.world.cols),
        "patches": int(getattr(c.resources, "patch_count", -1)),
        "max_count": int(getattr(c.population, "max_count", -1)),
        "rgm": float(getattr(c.resources, "patch_regrowth_mult", -1.0)),
        "bg_low_prod_frac": float(getattr(c.resources, "bg_low_prod_frac", -1.0)),
        "bg_cap_mult": float(getattr(c.resources, "bg_cap_mult", -1.0)),
        "initial_pop": int(len(e._id)),
        "ring_depth": int(a.ring_depth),
        "smoke": bool(a.smoke),
        "notes": str(notes),
    }
    bad = []
    if rb["signal_alphabet"] != "8":
        bad.append(f'signal_alphabet 读回 {rb["signal_alphabet"]!r} != "8"')
    if rb["memory_v2"] is not False:
        bad.append("memory_v2 读回 != False（P1-b 前提；R6.7 字段归属）")
    if rb["rd_enabled"]:
        bad.append("rd 应关未关")
    if rb["use_sim_core"]:
        bad.append("use_sim_core 应为 False（纯 Python）")
    if rb["patches"] != int(a.patches):
        bad.append(f'patches 读回 {rb["patches"]} != 请求 {a.patches}（装置预设疑似未生效）')
    if (rb["rows"], rb["cols"], rb["patches"]) != _S2 and not a.smoke:
        bad.append(f'非判别装置 {rb["rows"]}x{rb["cols"]}p{rb["patches"]}'
                   "（≠480×960/1700）—— 仪表冒烟请显式 --smoke")
    try:
        _assert_ids_ascending(np.asarray(e._id))
    except RuntimeError as ex:
        bad.append(str(ex))
    if bad:
        raise SystemExit("🔴 P1-c 擦除探针前置读回失败：\n  - " + "\n  - ".join(bad))
    if a.dm_guard and not Path(a.dm_guard).exists():
        raise SystemExit(f"🔴 dm-guard 前置检查失败：存档不存在 {a.dm_guard} "
                         f"⇒ 拒跑（避免跑完 {a.ticks}t 才发现无法对锚）")

    print(f"# [P1-c-erase] seed={a.seed} ticks={a.ticks} sample={a.sample} "
          f"device={rb['rows']}x{rb['cols']} patches={rb['patches']} "
          f"pop0={rb['initial_pop']} ring={rb['ring_depth']} smoke={rb['smoke']}")
    print(f"# [P1-c-erase] 读回：alphabet={rb['signal_alphabet']} rd={rb['rd_enabled']} "
          f"memory_v2={rb['memory_v2']} info_enabled={rb['info_structure_enabled']} "
          f"sparse={a.sparse_fields} bg_low={rb['bg_low_prod_frac']}/{rb['bg_cap_mult']}")
    if a.dm_guard:
        print(f"# [P1-c-erase] dm-guard 将对锚：{a.dm_guard}")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["tick", "n_agents", "n_slots", "frac_empty", "d_med", "d_p90",
            "d_p99", "d_mean", "d_max", "frac_d0", "frac_dle1", "frac_dle15",
            "frac_read_moore", "ms_per_tick_window",
            "pos_n_slots", "pos_d_med", "pos_d_p90", "pos_frac_dle1",
            "pos_frac_read_moore",
            "pea_n", "pea_d_med", "pea_d_p90", "pea_frac_d0", "pea_frac_dle1",
            "pea_frac_read_moore",
            "pep_n", "pep_d_med", "pep_d_p90", "pep_frac_dle1",
            "pep_frac_read_moore",
            "pep64_n", "pep64_d_med", "pep64_d_p90", "pep64_frac_dle1",
            "pep64_frac_read_moore",
            "eng_wr_ticks", "probe_rich_ticks"]
    recs: list[dict] = []
    pool: list[np.ndarray] = []
    pos_pool: list[np.ndarray] = []
    pea_pool: list[np.ndarray] = []
    pea_rd: list[np.ndarray] = []
    pep_pool: list[np.ndarray] = []
    pep_rd: list[np.ndarray] = []
    pep64_pool: list[np.ndarray] = []
    pep64_rd: list[np.ndarray] = []
    wr_total = 0
    rich_total = 0
    tracker = _ErasureTracker(e, depth=a.ring_depth)
    t_prev = time.perf_counter()
    extinct_at = None
    diag_tol = max(2, a.sample // 20)
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        f.flush()
        os.fsync(f.fileno())
        for t in range(1, a.ticks + 1):
            e.step()
            if e.extinct:
                extinct_at = t
                break
            tracker.observe(e)
            if t % a.sample == 0:
                now = time.perf_counter()
                ms_t = (now - t_prev) / a.sample * 1000.0
                t_prev = now
                rec = measure_p1c(e)
                pool.append(rec.pop("_d_pool"))
                pos_pool.append(rec.pop("_pos_pool"))
                er = measure_erase(e, tracker)
                pea_pool.append(er.pop("_pea_pool"))
                pea_rd.append(er.pop("_pea_rd"))
                pep_pool.append(er.pop("_pep_pool"))
                pep_rd.append(er.pop("_pep_rd"))
                pep64_pool.append(er.pop("_pep64_pool"))
                pep64_rd.append(er.pop("_pep64_rd"))
                wr, rich = tracker.take_window()
                wr_total += wr
                rich_total += rich
                if abs(wr - rich) > diag_tol:
                    raise SystemExit(
                        f"🔴 擦除探针诊断失配（t={t} 窗 {a.sample}t）："
                        f"eng_wr_ticks={wr} vs probe_rich_ticks={rich} "
                        f"> 容差 {diag_tol} ⇒ 事件检测口径疑似有误，停跑"
                        "（T1 设计 §2.2-2）。")
                rec.update(er)
                rec["tick"] = t
                rec["eng_wr_ticks"] = wr
                rec["probe_rich_ticks"] = rich
                rec["ms_per_tick_window"] = round(ms_t, 2)
                recs.append(rec)
                w.writerow({k: (v if v is None else
                                (round(v, 4) if isinstance(v, float) else v))
                            for k, v in rec.items()})
                f.flush()
                os.fsync(f.fileno())       # R263：行级落盘，崩溃最多丢 1 个采样点
                print(f"  t={t:>6d} N={rec['n_agents']:>6d} slots={rec['n_slots']:>6d} "
                      f"可直读(Moore)={_fmt(rec['frac_read_moore'], '.4f')} "
                      f"d>0={_fmt(rec['pos_frac_read_moore'], '.4f')} | "
                      f"擦除K16 n={rec['pea_n']:>6d} 中位={_fmt(rec['pea_d_med'], '.3f')} "
                      f"可直读={_fmt(rec['pep_frac_read_moore'], '.4f')}(d>0 {rec['pep_n']}) | "
                      f"K64 d>0 可直读={_fmt(rec['pep64_frac_read_moore'], '.4f')}"
                      f"({rec['pep64_n']}) | diag {wr}/{rich} ms/t={rec['ms_per_tick_window']}",
                      flush=True)

    # ---- 池化（口径与 P1-c 汇总逐式一致）
    dd = np.concatenate(pool) if pool else np.empty(0)
    pd = np.concatenate(pos_pool) if pos_pool else np.empty(0)
    eps = 1e-9
    pooled = {
        "n_samples": len(recs), "n_dist_pairs": int(dd.size),
        "d_med": _pct(dd, 50), "d_p90": _pct(dd, 90), "d_p99": _pct(dd, 99),
        "d_mean": float(dd.mean()) if dd.size else None,
        "d_max": float(dd.max()) if dd.size else None,
        "frac_d0": _frac(dd <= eps), "frac_dle1": _frac(dd <= 1.0 + eps),
        "frac_dle15": _frac(dd <= 1.5 + eps),
    }
    if dd.size:
        wsum = sum(r["n_slots"] for r in recs)
        pooled["frac_read_moore"] = float(
            sum(r["frac_read_moore"] * r["n_slots"] for r in recs) / wsum)
    else:
        pooled["frac_read_moore"] = None
    if pd.size:
        wsum_pos = sum(r["pos_n_slots"] for r in recs)
        pooled["pos_n_dist_pairs"] = int(pd.size)
        pooled["pos_d_med"] = _pct(pd, 50)
        pooled["pos_d_p90"] = _pct(pd, 90)
        pooled["pos_frac_dle1"] = _frac(pd <= 1.0 + eps)
        pooled["pos_frac_read_moore"] = (
            float(sum(r["pos_frac_read_moore"] * r["pos_n_slots"]
                      for r in recs) / wsum_pos) if wsum_pos else None)
    else:
        pooled["pos_n_dist_pairs"] = 0
        pooled["pos_d_med"] = None
        pooled["pos_d_p90"] = None
        pooled["pos_frac_dle1"] = None
        pooled["pos_frac_read_moore"] = None

    def _pool_set(dp: list[np.ndarray], rp: list[np.ndarray]) -> dict:
        da = np.concatenate(dp) if dp else np.empty(0)
        ra = np.concatenate(rp) if rp else np.empty(0, bool)
        pos = da > eps
        return {
            "n": int(da.size),
            "d_med": _pct(da, 50), "d_p90": _pct(da, 90),
            "frac_d0": _frac(da <= eps), "frac_dle1": _frac(da <= 1.0 + eps),
            "frac_read_moore": _frac(ra),
            "n_pos": int(pos.sum()),
            "d_med_pos": _pct(da[pos], 50), "d_p90_pos": _pct(da[pos], 90),
            "frac_dle1_pos": _frac(da[pos] <= 1.0 + eps),
            "frac_read_moore_pos": _frac(ra[pos]),
        }

    pooled["pea"] = _pool_set(pea_pool, pea_rd)
    pooled["pep"] = _pool_set(pep_pool, pep_rd)
    pooled["pep64"] = _pool_set(pep64_pool, pep64_rd)
    pooled["diag"] = {"eng_wr_ticks_total": int(wr_total),
                      "probe_rich_ticks_total": int(rich_total)}

    # ---- dm-guard（硬门；seed 207 必开）
    dm = {"path": a.dm_guard, "n_mismatch": 0, "passed": None}
    if a.dm_guard:
        mism = _dm_guard_mismatches(recs, a.dm_guard)
        dm["n_mismatch"] = len(mism)
        dm["passed"] = not mism
        if mism:
            for line in mism[:20]:
                print(line, file=sys.stderr)
            if len(mism) > 20:
                print(f"  ...（共 {len(mism)} 条）", file=sys.stderr)
            raise SystemExit(
                f"🔴 dm-guard 失败：{len(mism)} 处不等（见上）⇒ 探针观测与 P1-c "
                "存档不同源/有漂移，停跑回报（T1 设计 §2.2-5）。")

    summary = {
        "probe": "p1c_erasure_probe", "argv": sys.argv[1:],
        "seed": a.seed, "ticks": a.ticks, "sample": a.sample,
        "extinct_at": extinct_at, "readback": rb, "per_sample": recs,
        "pooled": pooled, "dm_guard": dm,
        "notes": [
            "口径A = 球面大圆距离（单位=1 纬度格宽）；口径B = 引擎 Moore 直读（同 P1-c）",
            "pea_* = K16 窗全部条目；pep_* = K16 窗 d>0 子集（🔴 主判据）；"
            "pep64_* = K64 窗 d>0 子集（预注册次判）",
            "被擦除集 = 每体独立 FIFO 环中除最近 4 条之外的条目（反事实 K16/K64）",
            "相位披露：引擎写点在步 4.4（移动前），探针步末观测（移动后）⇒ ≤1 tick 差；"
            "诊断列 eng_wr_ticks vs probe_rich_ticks 兜底",
            "纯观测：不写引擎、不消费主 RNG；判读权归天平（本探针只出数）",
        ],
    }
    sp = out.with_name(out.stem + ".summary.json")
    with open(sp, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
        f.flush()
        os.fsync(f.fileno())

    pp = pooled["pea"]
    qq = pooled["pep"]
    r64 = pooled["pep64"]
    print("\n===== P1-c 擦除探针池化（全部采样点）=====")
    print(f"  采样点 n={pooled['n_samples']}；保留集（P1-c）：d>0 中位="
          f"{_fmt(pooled['pos_d_med'], '.3f')} 可直读={_fmt(pooled['pos_frac_read_moore'], '.4f')}")
    print(f"  擦除集 K16 全部：n={pp['n']} 中位={_fmt(pp['d_med'], '.3f')} "
          f"d≤1={_fmt(pp['frac_dle1'], '.4f')} 可直读={_fmt(pp['frac_read_moore'], '.4f')}")
    print(f"  擦除集 K16 d>0：n={qq['n_pos']} 中位={_fmt(qq['d_med_pos'], '.3f')} "
          f"p90={_fmt(qq['d_p90_pos'], '.3f')} d≤1={_fmt(qq['frac_dle1_pos'], '.4f')} "
          f"可直读={_fmt(qq['frac_read_moore_pos'], '.4f')}  ← 🔴 主判据")
    print(f"  擦除集 K64 d>0：n={r64['n_pos']} 中位={_fmt(r64['d_med_pos'], '.3f')} "
          f"可直读={_fmt(r64['frac_read_moore_pos'], '.4f')}（预注册次判）")
    print(f"  诊断：eng_wr_ticks={wr_total} probe_rich_ticks={rich_total}（容差/窗 "
          f"{diag_tol}）")
    if a.dm_guard:
        print(f"  ✅ dm-guard 通过：共享列与 {a.dm_guard} 逐行一致")
    print(f"  逐采样 CSV：{out}")
    print(f"  摘要 JSON：{sp}")
    if extinct_at is not None:
        print(f"  ⚠️ 灭绝于 tick={extinct_at}")
    return 0


def _fmt(v, spec: str) -> str:
    return "n/a" if v is None else format(v, spec)


if __name__ == "__main__":
    raise SystemExit(main())


