"""无生物纯场装置（run_bare_world.py）——「纯斑块世界」M0 探针。

口径
----
* 设计稿 `docs/设计文档/设计-纯斑块世界-底层规则自组织-20261011.md` §四/§十五；
  四臂编排 = 项目所有者 直令（≤4 run、一半对照、单 run ≤2000 tick）。
* 臂定义（锁在 `ARMS`，预注册引用本表；同 seed 同参数，**只动规则开关**）：
    1 现状（对照）       env 全关（= 现役规则绝对基线）
    2 置换零模型（对照） 全规则 + H 空间置换（空间归因 null）
    3 地形（处理）       H→W→P；无河（runoff_gain=0）
    4 全规则（处理）     地形 + 河流汇流（runoff_gain=2.0）
  【扩展臂 5–8：C 裁下去混杂/自然剖面候选（2026-10-13；跑批须 amend 预注册批准）】
    5 置换+去纬度        5 = 2 的 base_t 换面积权全球均值（uniform_base_t；杀纬度梯度）
    6 置换+自然剖面      6 = 2 的 base_t 换 EBM 年平气温剖面（insolation_profile）
    7 全规则+自然剖面    7 = 4 的 base_t 换 EBM 剖面
    8 全规则+去纬度      8 = 4 的 base_t 换面积权全球均值
  （5–8 相对 2/4 只动 `base_t` 一项；1–4 逐键冻结不动——旧预注册行仍以本表为真源。）
  `water_sensitivity = cap_sensitivity = 1.0`（线性耦合 = 最小假设；AWM=1 归一由
  ENV 场自身保证、k_max=4.0 默认上钳）、`terrain_*`/`lapse_c`/`orographic`/`et_frac`
  全走 EnvFieldConfig 默认。
* 无生物：`initial_count` **构造后置改 0**（先例 `experiments/smell_perf_probe.py:60`；
  构造期断言 initial_count≥1 只在新构造时生效）。N=0 时 `_step_population` 早退、
  资源再生在同 tick 前序步骤照跑 ⇒ 场照常演化（设计稿 §2.2 已核，代码锚 3344/2895）。
  ⚠️ 不走 `eng.run()`（其默认 `stop_on_extinction=True` 会因 N=0 立即停）⇒ `eng.step()` 循环。
* 落盘（**不进主仓**，进数据仓 `the-world-data/bare_world/`）：一条 run =
  一个 npz（静态场 + 期初/期中/期末存量快照）+ 一个 CSV（样点读数）+ 一个
  manifest.json（臂定义/生效装置/env probe/计时/git 号）。run 结束 flush+fsync。
* 引擎侧零标签：本脚本不定义"斑块类型"，只出连续场（判型在观察侧，设计稿 §八）。

用法
----
  python experiments/run_bare_world.py --arm 4 --device s2 --seed 42 --ticks 2000
"""
from __future__ import annotations

import argparse
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

from simulation.config import AtmosphereConfig, SimConfig  # noqa: E402
from simulation.sphere_engine import SphereEngine  # noqa: E402
from tools.device_presets import add_device_arg, resolve_device  # noqa: E402

#: 四臂定义（预注册引用本表；键名与 EnvFieldConfig 字段对齐）
#: 1–4 = 原四臂（`2b02269` 起逐键冻结）；5–8 = C 裁下去混杂候选（只比 2/4 多 `base_t` 键）
ARMS: dict[int, dict] = {
    1: {"name": "现状(对照)", "env": False, "shuffle": False,
        "runoff_gain": 0.0, "ws": 0.0, "cs": 0.0},
    2: {"name": "置换零模型(对照)", "env": True, "shuffle": True,
        "runoff_gain": 2.0, "ws": 1.0, "cs": 1.0},
    3: {"name": "地形(处理)", "env": True, "shuffle": False,
        "runoff_gain": 0.0, "ws": 1.0, "cs": 1.0},
    4: {"name": "全规则(处理)", "env": True, "shuffle": False,
        "runoff_gain": 2.0, "ws": 1.0, "cs": 1.0},
    5: {"name": "置换+去纬度", "env": True, "shuffle": True,
        "runoff_gain": 2.0, "ws": 1.0, "cs": 1.0, "base_t": "uniform"},
    6: {"name": "置换+自然剖面", "env": True, "shuffle": True,
        "runoff_gain": 2.0, "ws": 1.0, "cs": 1.0, "base_t": "insolation"},
    7: {"name": "全规则+自然剖面", "env": True, "shuffle": False,
        "runoff_gain": 2.0, "ws": 1.0, "cs": 1.0, "base_t": "insolation"},
    8: {"name": "全规则+去纬度", "env": True, "shuffle": False,
        "runoff_gain": 2.0, "ws": 1.0, "cs": 1.0, "base_t": "uniform"},
}


def build_cfg(arm: int, seed: int, rows: int, cols: int, patches: int,
              rgm: float, atm: dict | None = None) -> SimConfig:
    cfg = SimConfig(seed=seed)
    # ---- P0.4 大气开关（默认全 off ⇒ 与旧批逐位一致）----
    a = atm or {}
    kw = {"enabled": bool(a.get("atmosphere", False)),
          "moisture_on": bool(a.get("moisture", False)),
          "coriolis_on": bool(a.get("coriolis", False)),
          "coriolis_scale": float(a.get("coriolis_scale", 1.0))}
    if a.get("heat_advection"):
        # 🔴 D-1（v38）：main 上已无温度平流实现（Step B 留存档分支）。
        # 开关保留只为接口稳定；真要用必须等 P5 两层大气。
        if not hasattr(AtmosphereConfig(), "heat_advection_on"):
            raise SystemExit("🔴 main 上无 heat_advection 实现（D-1 已移除，P5 再启）；"
                             "复现崩溃请用 tag archive/c4a2-stepb-blocked-e6e0233")
        kw["heat_advection_on"] = True
    cfg.atmosphere = AtmosphereConfig(**kw)
    # 🔴 装置口径硬写为 s2 家族（R5.6：正式批必须显式 --device s2；这里的默认值
    #   与 DEVICE_PRESETS['s2'] 同步，忘传 --device 也不静默换装置）。
    cfg.world.rows, cfg.world.cols = int(rows), int(cols)
    cfg.simulation.use_sim_core = False       # env 需 Python 路径；四臂统一（可比性）
    cfg.resources.distribution = "patchy"
    cfg.resources.patch_count = int(patches)
    cfg.resources.bg_production_zero = True   # s2 口径：背景纯零产能
    cfg.resources.patch_regrowth_mult = float(rgm)
    cfg.population.initial_count = 0          # 🔴 构造后置改 0（先例 smell_perf_probe.py:60）
    spec = ARMS[arm]
    if spec["env"]:
        e = cfg.env_field
        e.enabled = True
        e.salt = 0
        e.shuffle = bool(spec["shuffle"])
        e.runoff_gain = float(spec["runoff_gain"])
        e.water_sensitivity = float(spec["ws"])
        e.cap_sensitivity = float(spec["cs"])
        # 1–4 臂逐键冻结在 classic：2026-10-14 起 insolation 为正式世界默认 ⇒ 此处显式钉回
        # （5/8 臂 uniform 同时关 insolation；6/7 臂显式开；防默认漂移破坏旧批可比性）
        base_t = spec.get("base_t", "classic")
        e.insolation_profile = (base_t == "insolation")
        if base_t == "uniform":
            e.uniform_base_t = True
        elif base_t not in ("classic", "insolation"):
            raise ValueError(f"未知 base_t 档：{base_t!r}（classic/uniform/insolation）")
    if bool(a.get("atmosphere", False)) and not cfg.env_field.enabled:
        # 🔴 大气的 EBM 必须按地表类型查 α/c 表（引擎有 fail-loud 守卫）⇒ 开大气即
        # 强制 C0b。⚠️ 这会改变该臂的资源场（C0b 的 f_g/f_c 注入）⇒ manifest 记
        # `env_forced_by_atmosphere`，读数不与纯 arm1 对照混用。
        cfg.env_field.enabled = True
        cfg.env_field.c0b_enabled = True
    return cfg


_ATM_MAP = {
    "tmin": "t_min", "tmax": "t_max", "t_awm": "t_awm", "cfl_max": "cfl_max",
    "u_rms": "u_rms_awm_ms", "u_max": "u_max_ms",
    "w_eq_band": "w_eq_band_ms", "div_sub_band": "div_sub_band_per_m",
    "precip_mm_yr": "precip_mm_yr", "evap_mm_yr": "evap_mm_yr",
    "sea2land_mm_yr": "sea2land_flux_mm_yr",
    "residence_days": "residence_days_global",
    "bucket_awm_land": "bucket_awm_land_mm",
}


def _atm_row(eng, tick: int) -> dict:
    """大气读数行（P0.4）。关档 ⇒ 空 dict（CSV 列面与旧批逐位一致）。

    tick 0 不调 probe（`steps=0` 时年率读数的分母为 0）⇒ 全 NaN 占位。
    """
    atm = getattr(eng, "atmosphere", None)
    if atm is None:
        return {}
    if tick <= 0:
        return {k: float("nan") for k in _ATM_MAP}
    pr = atm.probe()
    return {k: float(pr.get(src, float("nan"))) for k, src in _ATM_MAP.items()}


def _sample_row(tick: int, grid, cap_pos, pos) -> dict:
    s = grid[pos]
    sat = np.clip(s / cap_pos, 0.0, 1.0)
    return {
        "tick": tick,
        "stock_sum": float(grid.sum()),
        "stock_pos_mean": float(s.mean()),
        "stock_pos_std": float(s.std()),
        "stock_p50": float(np.percentile(s, 50.0)),
        "stock_p90": float(np.percentile(s, 90.0)),
        "sat_mean": float(sat.mean()),
        "sat_gt50_frac": float((sat > 0.5).mean()),
    }


def _git_head() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                           capture_output=True, text=True, timeout=10)
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:
        return ""


def run_arm(arm: int, seed: int, ticks: int, sample: int, out_dir: Path,
            device_applied: dict, rows: int, cols: int, patches: int,
            rgm: float, atm: dict | None = None,
            climate_source=None, season: dict | None = None,
            tick_seconds: int = 36) -> dict:
    spec = ARMS[arm]
    cfg = build_cfg(arm, seed, rows, cols, patches, rgm, atm)
    # A20.3：时间尺度（默认 36 s/tick；360 会在 SimConfig.__post_init__ 里同步 rotation/dt）
    from simulation.config import TimeScaleConfig
    cfg.time_scale = TimeScaleConfig(tick_seconds=int(tick_seconds))
    if int(tick_seconds) != 36:
        # 同步已由 __post_init__ 做过一次；这里重建一次 SimConfig 不便，直接按规则改字段
        cfg.light.rotation_period = int(cfg.time_scale.rotation_period)
        if bool(cfg.atmosphere.enabled):
            cfg.atmosphere.dt_s = float(cfg.time_scale.atmosphere_dt_s(
                cfg.atmosphere.substeps, cfg.atmosphere.every_ticks))
    # A18.4：季节是**世界族参数**（默认 0/0 = 无季节，逐位等价）
    if season:
        cfg.light.tilt_rad = float(season.get("tilt", 0.0))
        cfg.light.season_period = float(season.get("period", 0.0))
    t0 = time.perf_counter()
    # A18.2：有 GCM 表时注进去（降水空间型随之换源；水文在构造期就用表，见 A18.3）
    eng = SphereEngine(cfg, climate_source=climate_source)
    t_build = time.perf_counter() - t0

    res = eng.resources
    cap = res._capacity
    pos = cap > 0.0
    cap_pos = cap[pos]

    static: dict[str, np.ndarray] = {}
    env_probe = None
    if spec["env"]:
        ef = eng.env_field
        static = {"h": ef.h, "w": ef.w, "temp": ef.temp, "acc": ef.acc,
                  "river_mask": ef.river_mask}
        env_probe = ef.probe()

    snap_ticks = sorted({ticks // 2, ticks})
    snaps: dict[int, np.ndarray] = {0: res._grid.copy()}
    samples = [{**_sample_row(0, res._grid, cap_pos, pos), **_atm_row(eng, 0)}]
    t0 = time.perf_counter()
    for t in range(1, ticks + 1):
        eng.step()
        if t in snap_ticks:
            snaps[t] = res._grid.copy()
        if t % sample == 0 or t == ticks:
            samples.append({**_sample_row(t, res._grid, cap_pos, pos), **_atm_row(eng, t)})
    run_s = time.perf_counter() - t0

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"arm{arm}_s{seed}"
    npz_path = out_dir / f"{stem}.npz"
    csv_path = out_dir / f"{stem}.csv"
    man_path = out_dir / f"{stem}.manifest.json"

    payload: dict[str, np.ndarray] = {
        "capacity": cap, "patch_mask": res._patch_mask}
    payload.update(static)
    for t, arr in snaps.items():
        payload[f"stock_t{t}"] = arr
    np.savez_compressed(npz_path, **payload)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        f.write("arm,seed," + ",".join(samples[0].keys()) + "\n")
        for r in samples:
            f.write("%d,%d," % (arm, seed)
                    + ",".join(f"{v:.9g}" for v in r.values()) + "\n")
        f.flush()
        os.fsync(f.fileno())

    probe_json = None
    atm_obj = getattr(eng, "atmosphere", None)
    if atm_obj is not None:
        probe_json = atm_obj.probe()
        probe_path = out_dir / f"{stem}.probe.json"
        with open(probe_path, "w", encoding="utf-8") as f:
            json.dump(probe_json, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())

    manifest = {
        "probe": "run_bare_world",
        "date": datetime.now().isoformat(timespec="seconds"),
        "host": platform.node(),
        "git_head": _git_head(),
        "arm": arm, "arm_name": spec["name"], "arm_spec": spec,
        "seed": seed, "ticks": ticks, "sample_every": sample,
        "world": {"rows": rows, "cols": cols,
                  "n_cells": rows * cols},
        "device_applied": device_applied,
        "use_sim_core": False,
        "sparse_fields": bool(cfg.simulation.sparse_fields),
        "pop_initial": 0, "patches": patches, "rgm": rgm,
        "n_patch_cells": int(pos.sum()),
        # ---- A18.2 / A18.4：世界族参数与写死项取值必须进 manifest（审计 §五-3）----
        "pr_source": (env_probe or {}).get("pr_source", "proxy"),
        "pr_normalize": (env_probe or {}).get("pr_policy", "keep_total"),
        # C1 手画降水带只在**无表档**才有意义（有表时降水空间型来自表）
        "cl1_band_used": bool(getattr(cfg.env_field, "c1_enabled", False))
                          and (env_probe or {}).get("pr_source") is None,
        "npp": {"enabled": bool(cfg.env_field.npp.enabled),
                "form": str(cfg.env_field.npp.form),
                "c0": float(cfg.env_field.npp.c0), "r0": float(cfg.env_field.npp.r0),
                "patch_mode": str(cfg.env_field.npp.patch_mode),
                "local_window": int(cfg.env_field.npp.patch_local_window),
                "local_delta": float(cfg.env_field.npp.patch_local_delta)},
        "resources": {"distribution": str(cfg.resources.distribution),
                      "capacity_per_area": float(cfg.resources.capacity_per_area),
                      "regrowth_rate": float(cfg.resources.regrowth_rate),
                      "temp_sensitivity": float(cfg.resources.temp_sensitivity),
                      "patch_count": int(cfg.resources.patch_count),
                      "patch_capacity_mult": float(cfg.resources.patch_capacity_mult),
                      "patch_regrowth_mult": float(cfg.resources.patch_regrowth_mult)},
        "season": {"tilt_rad": float(cfg.light.tilt_rad),
                   "season_period": float(cfg.light.season_period)},
        "time_scale": {"tick_seconds": int(cfg.time_scale.tick_seconds),
                       "atmosphere_enabled": bool(cfg.atmosphere.enabled),
                       "rotation_period": int(cfg.light.rotation_period),
                       "atmosphere_dt_s": float(cfg.atmosphere.dt_s),
                       "year_ticks": int(cfg.time_scale.year_ticks)},
        "sea_level": {"mode": str(cfg.env_field.sea_level.mode),
                      "total_water_m": float(cfg.env_field.sea_level.total_water_m)},
        "env_probe": env_probe,
        "env_note": getattr(eng, "env_note", None),
        "timing": {"build_s": t_build, "run_s": run_s,
                   "ms_per_tick": run_s / ticks * 1000.0},
        "switches": {"atmosphere": bool(cfg.atmosphere.enabled),
                     "env_forced_by_atmosphere": bool(cfg.atmosphere.enabled and not spec["env"]),
                     "moisture": bool(cfg.atmosphere.moisture_on),
                     "coriolis": bool(getattr(cfg.atmosphere, "coriolis_on", False)),
                     "heat_advection": bool(getattr(cfg.atmosphere, "heat_advection_on", False)),
                     "atmosphere_config": cfg.atmosphere.__dict__},
        "outputs": {"npz": npz_path.name, "csv": csv_path.name,
                    "probe": (probe_path.name if probe_json is not None else None)},
        "snap_ticks": [0] + snap_ticks,
    }
    with open(man_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())

    print(f"[arm{arm} {spec['name']}] ticks={ticks} build={t_build:.2f}s "
          f"run={run_s:.1f}s ({run_s / ticks * 1000.0:.1f} ms/tick) "
          f"patch_cells={int(pos.sum())}")
    if env_probe is not None:
        print(f"  env: w_mean={env_probe['w_mean']:.4g} u_p99={env_probe['u_p99']:.3g} "
              f"u_max={env_probe['u_max']:.3g} river_frac={env_probe['river_frac']:.4f} "
              f"flood_filled={env_probe['flood_filled']} "
              f"gf_max={env_probe.get('gf_max')} cf_max={env_probe.get('cf_max')}")
    print(f"  -> {npz_path}")
    return manifest


def main(argv=None) -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="无生物纯场装置（四臂协议 §十五）")
    ap.add_argument("--arm", type=int, choices=sorted(ARMS), required=True,
                    help="1 现状 | 2 置换零模型 | 3 地形 | 4 全规则 | 5 置换+去纬度 | "
                         "6 置换+自然剖面 | 7 全规则+自然剖面 | 8 全规则+去纬度")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ticks", type=int, default=2000)
    ap.add_argument("--sample", type=int, default=50, help="CSV 样点间隔（tick）")
    ap.add_argument("--rows", type=int, default=480)
    ap.add_argument("--cols", type=int, default=960)
    ap.add_argument("--patches", type=int, default=1700)
    ap.add_argument("--rgm", type=float, default=1.195)
    ap.add_argument("--out", default=str(ROOT / "the-world-data" / "bare_world"))
    ap.add_argument("--atmosphere", choices=["off", "on"], default="off",
                    help="C4a-1 热机（默认 off；on 需配合 --ticks 与 --sample）")
    ap.add_argument("--moisture", choices=["off", "on"], default="off",
                    help="C4a-2 Step A 水汽（默认 off；需 --atmosphere on）")
    ap.add_argument("--coriolis", choices=["off", "on"], default="off",
                    help="B4.1 科氏力（默认 off；需 --atmosphere on）")
    ap.add_argument("--coriolis-scale", type=float, default=1.0,
                    help="G-P2 敏感性：0.5 = 半自转（默认 1.0 = 物理值）")
    ap.add_argument("--tick-seconds", type=int, default=36, choices=(36, 360),
                    help="世界时间尺度（360 会自动同步 rotation_period/dt_s）")
    ap.add_argument("--season-tilt", type=float, default=0.0,
                    help="黄赤交角（弧度）。0=无季节（当前默认）；0.409≈23.5°，需配 --season-period")
    ap.add_argument("--season-period", type=float, default=0.0,
                    help="世界一年多少 tick（0=无季节）")
    ap.add_argument("--table", default=None,
                    help="GCM 气候表 npz（给了就注 GCMTableSource：温度+降水都走表）")
    ap.add_argument("--year-ticks", type=int, default=87600,
                    help="世界一年多少 tick（表是 12 月层时才必须）")
    ap.add_argument("--precip-policy", choices=["keep_total", "table_total"],
                    default="keep_total", help="降水换源的总量口径（默认保持原总量）")
    ap.add_argument("--heat-advection", choices=["off", "on"], default="off",
                    help="⚠️ 仅用于复现 P5 阻断（s2 上必崩），不得进任何验收")
    add_device_arg(ap)
    a = ap.parse_args(argv if argv is not None else sys.argv[1:])
    device_applied = resolve_device(a, sys.argv[1:],
                                    logger=lambda m: print(m, file=sys.stderr))
    if a.season_period > 0.0 or a.season_tilt != 0.0:
        if a.season_period <= 0.0:
            raise SystemExit("🔴 --season-tilt 非 0 时必须给 --season-period（否则季节不推进）")
    if a.ticks < 1 or a.sample < 1:
        raise SystemExit("🔴 --ticks/--sample 必须 ≥1")
    # R278 防呆只覆盖 --device 路径；显式小 rows/cols 忘改回无回显 ⇒ 此处兜底提示
    if (a.rows, a.cols) != (480, 960):
        print(f"[提示] 非 s2 正式档：rows×cols={a.rows}×{a.cols}（s2=480×960）"
              "——仅限小装置自检，读数不与正式批可比（R5.6）。", file=sys.stderr)
    atm = {"atmosphere": a.atmosphere == "on", "moisture": a.moisture == "on",
           "coriolis": a.coriolis == "on",
           "coriolis_scale": a.coriolis_scale,
           "heat_advection": a.heat_advection == "on"}
    if atm["moisture"] and not atm["atmosphere"]:
        raise SystemExit("🔴 --moisture on 需同时 --atmosphere on（否则开关无人读取）")
    if atm["heat_advection"]:
        print("[警告] --heat-advection on 仅用于 P5 阻断复现（s2 会崩），不得进验收",
              file=sys.stderr)
    climate_source = None
    if a.table:
        from world.climate import GCMTableSource
        climate_source = GCMTableSource(Path(a.table), year_ticks=int(a.year_ticks))
    run_arm(a.arm, a.seed, a.ticks, a.sample, Path(a.out), device_applied,
            a.rows, a.cols, a.patches, a.rgm, atm, climate_source=climate_source,
            season={"tilt": a.season_tilt, "period": a.season_period},
            tick_seconds=a.tick_seconds)


if __name__ == "__main__":
    main()
