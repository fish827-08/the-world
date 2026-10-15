"""ENV 场级 digest 装置（C0b/C1 关档逐位等价对拍 + 开档物理锚读数）。

为什么有这个脚本
----------------
`docs/技术文档/技术-动态天气路线评估与选型-20261014.md` §5 验收要求：
C0b/C1 全部是**构造期静态**变更 ⇒ 验收不需要机时报批，只需要"关档逐位等价 +
开档物理锚"两组**场级、秒级**读数。裁定二流程当时用的是临时脚本（未入库，
口径不可复算），本装置把它固化为可复跑的仓库内工具。

digest 口径（自证，逐字写死）
----------------------------
1. **图层集合** `LAYERS`（顺序即口径，改动只许**追加到末尾**，换序 = 作废历史 digest）：
   `h, h_routed, temp, w, acc, filled, receivers, river_mask`；
   因子层 `gf, cf` 仅当 `water_sensitivity>0 / cap_sensitivity>0` 时纳入
   （None ⇒ 不入，与引擎侧"整块跳过"同语义）。
2. **逐场 sha256** = `sha256(np.ascontiguousarray(arr).tobytes())`（原始小端字节，
   不含任何文本格式化 ⇒ 对 ULP 变化敏感；先例 `tests/test_rng_isolation.py:33`）。
3. **组合 digest** = 对每个纳入层依次 `update(name_utf8 + b"|" + dtype + b"|" +
   shape_csv + b"#" + raw_bytes)`，再追加标量层 `w_ref/pit_first/flood_filled`
   的 `"%.17g"` 文本 ⇒ 单串 sha256。
4. **备选口径**（`variants`）：无名字纯拼接 / 逐场 hexdigest 再拼接 /
   `probe()` JSON —— 用于与历史 digest（裁定二 D0/D1）交叉核对；**对拍以正式
   口径（3）为准**，备选仅供归因。

用法
----
  # 关档逐位等价（正式档必须显式 --device s2）
  python tools/env_field_digest.py --device s2 --seed 42 --mode classic --json <path>
  python tools/env_field_digest.py --device s2 --seed 42 --mode default --json <path>
  # 开档验收（C0b / C0b+C1）
  python tools/env_field_digest.py --device s2 --seed 42 --mode default --c0b --json <path>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from simulation.config import EnvFieldConfig, SimConfig        # noqa: E402
from simulation.sphere_engine import SphereEngine              # noqa: E402
from tools.device_presets import add_device_arg, resolve_device  # noqa: E402
from world.env_field import EnvField                           # noqa: E402
from world.light_and_temperature import LightAndTemperature     # noqa: E402
from world.sphere_world import SphereWorld                     # noqa: E402

#: 图层顺序 = digest 口径（见模块 docstring 第 1 条；只许末尾追加）
LAYERS: tuple[str, ...] = (
    "h", "h_routed", "temp", "w", "acc", "filled", "receivers", "river_mask",
)
FACTORS: tuple[str, ...] = ("gf", "cf")
SCALARS: tuple[str, ...] = ("w_ref", "pit_first", "flood_filled")

#: base_t 三档（与 `EnvFieldConfig` 的两个布尔项一一对应；裁定一口径）
BASE_T_MODES: dict[str, dict] = {
    "classic": {"insolation_profile": False, "uniform_base_t": False},
    "default": {"insolation_profile": True, "uniform_base_t": False},
    "uniform": {"insolation_profile": False, "uniform_base_t": True},
}


def _sha(arr) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def _state_bytes(obj) -> bytes:
    """bit_generator.state 的确定性序列化（dict/int/ndarray 混合，键排序）。"""
    if isinstance(obj, np.ndarray):
        return obj.tobytes()
    if isinstance(obj, (bool, int, np.integer)):
        return str(int(obj)).encode("utf-8")
    if isinstance(obj, str):
        return obj.encode("utf-8")
    if isinstance(obj, dict):
        return b"|".join(_state_bytes(obj[k]) for k in sorted(obj))
    if isinstance(obj, (tuple, list)):
        return b"|".join(_state_bytes(v) for v in obj)
    return repr(obj).encode("utf-8")


def _git_head() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                           capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=10)
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:
        return ""


def layer_dict(ef: EnvField) -> dict[str, np.ndarray]:
    """EnvField 的全部静态场层（含 None 语义的因子层）。"""
    d = {name: getattr(ef, name) for name in LAYERS}
    for name, f in zip(FACTORS, (ef.growth_factor(), ef.capacity_factor())):
        if f is not None:
            d[name] = f
    return d


def digest_layers(layers: dict[str, np.ndarray], scalars: dict) -> str:
    """正式口径（模块 docstring 第 3 条）。"""
    h = hashlib.sha256()
    for name in LAYERS + FACTORS:
        if name not in layers:
            continue
        arr = np.ascontiguousarray(layers[name])
        shape = ",".join(str(s) for s in arr.shape)
        h.update(name.encode("utf-8") + b"|" + str(arr.dtype).encode("utf-8")
                 + b"|" + shape.encode("utf-8") + b"#")
        h.update(arr.tobytes())
    for name in SCALARS:
        h.update(("%.17g" % float(scalars[name])).encode("utf-8") + b";")
    return h.hexdigest()


def variant_digests(ef: EnvField, layers: dict[str, np.ndarray]) -> dict[str, str]:
    """备选口径（仅供与历史 digest 交叉核对，不作对拍依据）。"""
    order = [n for n in LAYERS + FACTORS if n in layers]
    cat = hashlib.sha256()
    named = hashlib.sha256()
    for n in order:
        cat.update(np.ascontiguousarray(layers[n]).tobytes())
        named.update(n.encode("utf-8"))
        named.update(np.ascontiguousarray(layers[n]).tobytes())
    per = hashlib.sha256("".join(_sha(layers[n]) for n in order).encode("utf-8"))
    sub = {
        "cat_h_temp_w_acc": ("h", "temp", "w", "acc"),
        "cat_h_w_acc": ("h", "w", "acc"),
        "cat_h_w_acc_temp_receivers": ("h", "w", "acc", "temp", "receivers"),
    }
    out = {
        "cat_plain": cat.hexdigest(),
        "cat_named": named.hexdigest(),
        "cat_sorted_plain": hashlib.sha256(b"".join(
            np.ascontiguousarray(layers[n]).tobytes() for n in sorted(order)
        )).hexdigest(),
        "per_concat": per.hexdigest(),
        "layers_json": hashlib.sha256(json.dumps(
            {n: _sha(layers[n]) for n in order}, sort_keys=True
        ).encode("utf-8")).hexdigest(),
        "probe_json": hashlib.sha256(
            json.dumps(ef.probe(), sort_keys=True).encode("utf-8")).hexdigest(),
    }
    for label, names in sub.items():
        hh = hashlib.sha256()
        for n in names:
            if n in layers:
                hh.update(np.ascontiguousarray(layers[n]).tobytes())
        out[label] = hh.hexdigest()
    return out


def lat_profile(ef: EnvField, bin_deg: float = 10.0) -> list:
    """|φ| 纬带剖面（C1 三层结构 + C0b 海陆对比的主读数；w==0 ⇔ yield==0）。"""
    world = ef.world
    lat_deg = np.abs(np.degrees(np.repeat(
        np.asarray(world._lat, dtype=np.float64), world.cols)))
    areas = world.cell_area(np.arange(world.n_cells, dtype=np.int64))
    edges = np.arange(0.0, 90.0 + bin_deg, bin_deg)
    idx = np.clip(np.searchsorted(edges, lat_deg, side="right") - 1,
                  0, edges.size - 2)
    rows = []
    for k in range(edges.size - 1):
        m = idx == k
        if not bool(m.any()):
            continue
        rows.append({
            "lat_lo": float(edges[k]), "lat_hi": float(edges[k + 1]),
            "cells": int(m.sum()),
            "w_awm": float(np.sum(ef.w[m] * areas[m]) / np.sum(areas[m])),
            "w_zero_frac": float(np.count_nonzero(ef.w[m] <= 0.0) / m.sum()),
            "band_mean": (float(ef.band[m].mean()) if ef.band is not None else None),
            "ocean_frac": (float(ef.ocean_mask[m].mean())
                           if getattr(ef, "ocean_mask", None) is not None else None),
            "h_mean": float(ef.h[m].mean()),
        })
    return rows


def build_env_field(rows: int, cols: int, seed: int, mode: str,
                   env_kw: dict) -> EnvField:
    if mode not in BASE_T_MODES:
        raise SystemExit(f"🔴 未知 --mode {mode!r}；可用：{sorted(BASE_T_MODES)}")
    world = SphereWorld(rows, cols)
    lt = LightAndTemperature(world)
    cfg = EnvFieldConfig(enabled=True, **BASE_T_MODES[mode], **env_kw)
    return EnvField(world, lt, cfg, base_seed=seed)


def env_report(ef: EnvField) -> dict:
    """digest + 逐场 sha + 关键物理读数（含 C0b/C1 开档锚，字段不存在则跳过）。"""
    layers = layer_dict(ef)
    scalars = {n: getattr(ef, n) for n in SCALARS}
    rep = {
        "layers": {n: _sha(a) for n, a in layers.items()},
        "digest": digest_layers(layers, scalars),
        "variants": variant_digests(ef, layers),
        "scalars": {n: float(v) for n, v in scalars.items()},
        "probe": ef.probe(),
    }
    # ---- C0b/C1 开档锚读数（新层关档时 = None ⇒ 不进报告，保持旧 JSON 形状）----
    for attr in ("sea_level_m", "ocean_mask", "land_mask", "shelf_mask",
                 "lake_mask", "fresh_lake_mask", "salt_lake_mask", "surface_type",
                 "river_mask_legacy", "lake_count", "max_lake_cells",
                 "max_lake_root", "max_lake_basin_cells", "band"):
        v = getattr(ef, attr, None)
        if v is None:
            continue
        rep.setdefault("c0b", {})
        rep["c0b"][attr] = (_sha(v) if isinstance(v, np.ndarray)
                            else (v if isinstance(v, (int, float, bool))
                                  else str(v)))
    cons = getattr(ef, "conservation", None)
    if cons:
        rep["conservation"] = dict(cons)
    return rep


def engine_report(rows: int, cols: int, seed: int, mode: str, ticks: int,
                  env_kw: dict) -> dict:
    """引擎侧等价面：env 场 + 资源初始层 + RNG 流（证"零新增 RNG 消费"）。"""
    cfg = SimConfig(seed=seed)
    cfg.world.rows, cfg.world.cols = rows, cols
    cfg.simulation.use_sim_core = False        # env 走 Python 路径（构造期守卫同口径）
    e = cfg.env_field
    e.enabled = True
    for k, v in BASE_T_MODES[mode].items():
        setattr(e, k, v)
    for k, v in env_kw.items():
        setattr(e, k, v)
    t0 = time.perf_counter()
    eng = SphereEngine(cfg)
    build_s = time.perf_counter() - t0
    rep = {
        "build_s": build_s,
        "capacity_sha": _sha(eng.resources._capacity),
        "grid_sha": _sha(eng.resources._grid),
        "rng_draws": int(eng.rng.draws),
        "rng_state_sha": hashlib.sha256(
            _state_bytes(eng.rng.bit_generator.state)).hexdigest(),
        "env_note": eng.env_note,
        "patch_mask_sha": _sha(eng.resources._patch_mask),
    }
    if ticks > 0:
        t0 = time.perf_counter()
        for _ in range(ticks):
            eng.step()
        rep["run_s"] = time.perf_counter() - t0
        rep[f"grid_sha_t{ticks}"] = _sha(eng.resources._grid)
        rep[f"rng_draws_t{ticks}"] = int(eng.rng.draws)
    return rep


def _parse_env_kw(a) -> dict:
    kw: dict = {}
    if a.sens > 0:
        kw["water_sensitivity"] = a.sens
        kw["cap_sensitivity"] = a.sens
    if a.c0b:
        kw["c0b_enabled"] = True
    if a.c1:
        kw["c1_enabled"] = True
    for nm in ("land_fraction", "dry_deg", "itcz_deg"):
        v = getattr(a, nm, None)
        if v is not None:
            kw[nm] = float(v)
    return kw


def main(argv=None) -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="ENV 场级 digest（关档逐位等价 + 开档锚）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mode", default="default", choices=sorted(BASE_T_MODES),
                    help="base_t 档：classic 历史对照 / default 现役默认(insolation) / uniform 诊断")
    ap.add_argument("--sens", type=float, default=0.0,
                    help="water=cap sensitivity（>0 ⇒ gf/cf 纳入 digest）")
    ap.add_argument("--c0b", action="store_true", help="开 C0b（真实海拔+海平面）")
    ap.add_argument("--c1", action="store_true", help="开 C1（静态副热带带）")
    ap.add_argument("--land-fraction", dest="land_fraction", type=float, default=None,
                    help="覆盖目标陆比（陆比扇扫 0.2/0.3/0.5 用）")
    ap.add_argument("--dry-deg", dest="dry_deg", type=float, default=None,
                    help="覆盖干槽中心纬度")
    ap.add_argument("--itcz-deg", dest="itcz_deg", type=float, default=None,
                    help="覆盖湿带中心纬度")
    ap.add_argument("--engine", action="store_true",
                    help="附引擎侧等价面（资源初始层 + RNG 流）")
    ap.add_argument("--ticks", type=int, default=0)
    ap.add_argument("--rows", type=int, default=480)
    ap.add_argument("--cols", type=int, default=960)
    ap.add_argument("--json", default="", help="结果落盘路径（诊断数据请指 the-world-data/）")
    ap.add_argument("--npz", default="",
                    help="静态场层落盘路径（海/陆/湖/河层给复审与 viz 用；只写数据仓）")
    add_device_arg(ap)
    a = ap.parse_args(argv if argv is not None else sys.argv[1:])
    device_applied = resolve_device(a, sys.argv[1:],
                                    logger=lambda m: print(m, file=sys.stderr))
    if not device_applied:
        raise SystemExit("🔴 场级诊断必须显式 --device s2（R5.6：吃默认微型档 ⇒ 读数不可比）")
    if (a.rows, a.cols) != (480, 960):
        print(f"[提示] 非 s2 正式档 rows×cols={a.rows}×{a.cols}（s2=480×960）"
              "——仅限小装置自检。", file=sys.stderr)
    env_kw = _parse_env_kw(a)
    t0 = time.perf_counter()
    ef = build_env_field(a.rows, a.cols, a.seed, a.mode, env_kw)
    construct_s = time.perf_counter() - t0
    out = {
        "probe": "env_field_digest",
        "date": datetime.now().isoformat(timespec="seconds"),
        "host": platform.node(),
        "git_head": _git_head(),
        "device_applied": device_applied,
        "mode": a.mode, "seed": a.seed, "rows": a.rows, "cols": a.cols,
        "env_kwargs": env_kw, "construct_s": construct_s,
        "env": env_report(ef),
        "lat_profile": lat_profile(ef),
    }
    if a.engine:
        out["engine"] = engine_report(a.rows, a.cols, a.seed, a.mode, a.ticks, env_kw)

    ev = out["env"]
    print(f"[digest:{a.mode}] seed={a.seed} {a.rows}x{a.cols} c0b={a.c0b} c1={a.c1} "
          f"construct={construct_s:.2f}s")
    print(f"  DIGEST = {ev['digest']}")
    for n, hx in ev["layers"].items():
        print(f"    layer {n:<11} = {hx[:16]}…{hx[-4:]}")
    pr = ev["probe"]
    print(f"  probe: w_mean={pr['w_mean']:.6g} flood_filled={pr['flood_filled']} "
          f"pit_first={pr['pit_first']} river_frac={pr['river_frac']:.4f}")
    if "c0b" in ev:
        print("  c0b: " + json.dumps(ev["c0b"], ensure_ascii=False, default=str))
    print("  lat_profile (|φ| 分箱: w_awm / w_zero_frac / band_mean):")
    for r in out["lat_profile"]:
        bm = "-" if r["band_mean"] is None else f"{r['band_mean']:.3f}"
        print(f"    [{r['lat_lo']:>2.0f},{r['lat_hi']:>2.0f}]  w={r['w_awm']:>9.4f}"
              f"  zero={r['w_zero_frac']:.3f}  band={bm}")
    if a.engine:
        en = out["engine"]
        print(f"  engine: build={en['build_s']:.2f}s draws={en['rng_draws']} "
              f"cap={en['capacity_sha'][:16]}… grid={en['grid_sha'][:16]}… "
              f"state={en['rng_state_sha'][:16]}…")
    if a.json:
        p = Path(a.json)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2, default=str)
            f.flush()
            os.fsync(f.fileno())
        print(f"  -> {p}")
    if a.npz:
        p = Path(a.npz)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {k: v for k, v in layer_dict(ef).items()}
        for attr in ("h_m", "ocean_mask", "land_mask", "shelf_mask", "lake_mask",
                     "fresh_lake_mask", "salt_lake_mask", "surface_type", "band"):
            v = getattr(ef, attr, None)
            if isinstance(v, np.ndarray):
                payload[attr] = v
        np.savez_compressed(p, **payload)
        print(f"  -> {p}")


if __name__ == "__main__":
    main()
