"""J0：SpeedyWeather.jl 最小干动力跑 + numpy 导出。

**主桥 = 进程内 juliacall**（2026-10-21 复核更正：原报告“juliacall 必然失败”**作废**——
在 `JULIA_DEPOT_PATH` 指向已 bootstrap 的 depot 后，`import juliacall` 实测可用，
且 1/5 天输出与 subprocess 桥**逐位相同**）。
**备桥 = subprocess**（`--bridge subprocess`，仅在 juliacall 不可用时使用）。

用法：
    PYTHONIOENCODING=utf-8 ../../.venv/Scripts/python.exe atmo_julia/run_minimal.py --days 5 --sweep
    ... --bridge subprocess --days 5        # 备桥（与 J0 首版同路）

环境（本机实测必需，脚本会自动补默认值并记录实际所用值）：
    JULIA_DEPOT_PATH = C:\\jd            （短 ASCII 路径；见 J0 报告「路径坑」）
    TEMP/TMP/TMPDIR  = C:\\tw\\...\\tmp  （Julia 的 Downloads 不认非 ASCII 临时路径）

产物（`_rerun_logs/20261021-A1-julia/`）：
    juliacall_day1..5/  每档 1–5 天的 .bin 场
    j0_fields.npz       主产物（--days 指定档，默认 5 天）
    j0_report.json      版本/环境/读数/可重复性/跨桥逐位对比

许可：SpeedyWeather.jl = **EUPL-1.2**（外部依赖，不 vendor、不改源码）。
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
# venv 在**主工作树**（worktree 在 <repo><worktree> 下），可用 A1_VENV 覆盖
VENV = Path(os.environ.get("A1_VENV", str(REPO.parents[1] / ".venv")))
OUT = REPO / "_rerun_logs" / "20261021-A1-julia"
JULIA = VENV / "julia_env" / "pyjuliapkg" / "install" / "bin" / "julia.exe"
PROJECT = REPO / "atmo_julia" / "julia_env"
CORE = REPO / "atmo_julia" / "j0_core.jl"


def _ascii_root() -> Path:
    """返回仓库根的 ASCII 别名路径（C:\\tw junction）；没有则回退真实路径。"""
    main = REPO.parents[1]
    alias = Path(r"C:\tw")
    try:
        if alias.resolve() == main.resolve():
            return alias / REPO.relative_to(main)
    except OSError:
        pass
    return REPO


def setup_env() -> dict[str, str]:
    """在 **import juliacall 之前**设好 depot/temp（Julia 启动时读取）。"""
    ascii_root = _ascii_root()
    depot = os.environ.get("JULIA_DEPOT_PATH") or os.environ.get("A1_JULIA_DEPOT")
    if not depot:
        jd = Path(r"C:\jd")
        depot = str(jd if jd.exists() else ascii_root / "_rerun_logs" / "20261021-A1-julia" / "julia_depot")
    os.environ["JULIA_DEPOT_PATH"] = depot

    tmp = os.environ.get("A1_JULIA_TMP") or str(ascii_root / "_rerun_logs" / "20261021-A1-julia" / "tmp")
    Path(tmp).mkdir(parents=True, exist_ok=True)
    for k in ("TEMP", "TMP", "TMPDIR"):
        os.environ[k] = tmp

    if any(ord(c) > 127 for c in depot):
        print(f"⚠️ depot 含非 ASCII 字符：{depot}（Julia Downloads 会崩）", file=sys.stderr)
    return {"depot": depot, "temp": tmp}


def run_julia_subprocess(days: int, trunc: int, nlayers: int, tag: str) -> dict:
    outdir = OUT / tag
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = [str(JULIA), f"--project={PROJECT}", "--startup-file=no",
           str(REPO / "atmo_julia" / "dump_fields.jl"), str(outdir), str(trunc), str(nlayers), str(days)]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=1800)
    wall = time.time() - t0
    if r.returncode != 0:
        raise RuntimeError(f"Julia 失败 rc={r.returncode}\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    line = [ln for ln in r.stdout.splitlines() if ln.startswith("J0_DONE")][-1]
    return {"outdir": outdir, "wall_s": wall, "summary": line}


def init_juliacall():
    """进程内桥：import juliacall → 激活 SpeedyWeather 工程 → include 核心。"""
    from juliacall import Main as jl  # noqa: PLC0415 —— 必须在 setup_env() 之后

    jl.seval(f'using Pkg; Pkg.activate(raw"{PROJECT}")')
    jl.seval("using SpeedyWeather")
    jl.include(str(CORE))
    return jl


def run_julia_inproc(jl, days: int, trunc: int, nlayers: int, tag: str) -> dict:
    outdir = OUT / tag
    outdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    summary = str(jl.j0_run(str(outdir), days, trunc, nlayers))
    wall = time.time() - t0
    return {"outdir": outdir, "wall_s": wall, "summary": summary}


def load(outdir: Path, npoints: int, nlayers: int) -> dict[str, np.ndarray]:
    pick = lambda n, shape: np.fromfile(outdir / n, dtype=np.float32).reshape(shape)
    return {
        "u": pick("u.bin", (npoints, nlayers)),
        "v": pick("v.bin", (npoints, nlayers)),
        "T": pick("T.bin", (npoints, nlayers)),
        "ps": np.fromfile(outdir / "ps.bin", dtype=np.float32),
        "lat": np.fromfile(outdir / "lat.bin", dtype=np.float32),
        "lon": np.fromfile(outdir / "lon.bin", dtype=np.float32),
    }


def parse_summary(line: str) -> dict:
    kv = {}
    for tok in line.split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            kv[k] = v
    return kv


def fields_equal(a: dict, b: dict, keys=("u", "v", "T", "ps")) -> bool:
    return all(np.array_equal(a[k], b[k]) for k in keys)


def load_tag(tag: str, npoints: int, nlayers: int) -> dict | None:
    d = OUT / tag
    if not (d / "T.bin").exists():
        return None
    return load(d, npoints, nlayers)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=5)
    ap.add_argument("--truncation", type=int, default=32)
    ap.add_argument("--nlayers", type=int, default=8)
    ap.add_argument("--bridge", choices=["juliacall", "subprocess"], default="juliacall")
    ap.add_argument("--sweep", action="store_true", help="额外跑 1–5 天各一档（juliacall 桥）")
    ap.add_argument("--no-repeat", action="store_true", help="跳过两次 1 天重复性对照")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    env_used = setup_env()

    if args.bridge == "juliacall":
        jl = init_juliacall()
        julia_version = str(jl.VERSION)
        runner = lambda days, tag: run_julia_inproc(jl, days, args.truncation, args.nlayers, tag)  # noqa: E731
        tag_main = f"juliacall_day{args.days}"
    else:
        if not JULIA.exists():
            print(f"找不到 Julia：{JULIA}", file=sys.stderr)
            return 2
        julia_version = "?"
        runner = lambda days, tag: run_julia_subprocess(days, args.truncation, args.nlayers, tag)  # noqa: E731
        tag_main = f"subprocess_day{args.days}"

    # ① 主跑
    main_run = runner(args.days, tag_main)
    kv = parse_summary(main_run["summary"])
    npoints, nlayers = int(kv["npoints"]), int(kv["nlayers"])
    f = load(main_run["outdir"], npoints, nlayers)
    julia_version = kv.get("julia", julia_version)

    # ② 可重复性：同配置**两次独立** 1 天运行，逐位比对（不再做 f2 与自身的死比较）
    repeatable = None
    if not args.no_repeat:
        rep1 = runner(1, f"{args.bridge}_repeat1")
        rep2 = runner(1, f"{args.bridge}_repeat2")
        f2 = load(rep1["outdir"], npoints, nlayers)
        f3 = load(rep2["outdir"], npoints, nlayers)
        repeatable = fields_equal(f2, f3)

    # ③ 1–5 天扫档（可选；验证长跨度稳定，不参与 npz；主跑那一档直接复用，不重复算）
    sweep = {}
    if args.sweep and args.bridge == "juliacall":
        for d in range(1, 6):
            if d == args.days:
                sweep[f"day{d}"] = {"wall_s": round(main_run["wall_s"], 2), "finite": kv.get("finite"),
                                    "T": kv.get("T"), "ps_Pa": kv.get("ps_Pa")}
                continue
            r = runner(d, f"juliacall_day{d}")
            kvd = parse_summary(r["summary"])
            sweep[f"day{d}"] = {"wall_s": round(r["wall_s"], 2), "finite": kvd.get("finite"),
                                "T": kvd.get("T"), "ps_Pa": kvd.get("ps_Pa")}

    # ④ 跨桥逐位对比（J0 首版 subprocess 产物：run_repeat = 1 天，run_main = 5 天）
    cross = {}
    for tag, days in (("run_repeat", 1), ("run_main", 5)):
        ref = load_tag(tag, npoints, nlayers)
        if ref is None:
            continue
        mine = f if days == args.days else load_tag(f"juliacall_day{days}", npoints, nlayers)
        if mine is None:
            continue
        cross[f"day{days}_vs_subprocess_{tag}"] = fields_equal(mine, ref)

    np.savez_compressed(
        OUT / "j0_fields.npz",
        u=f["u"], v=f["v"], T=f["T"], ps=f["ps"],
        lat=f["lat"], lon=f["lon"],
        sigma=np.array([float(x) for x in kv["sigma"].split(",")], dtype=np.float64),
        days=np.int64(args.days), truncation=np.int64(args.truncation),
    )

    meta = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "python": platform.python_version(),
        "bridge": ("juliacall（进程内）" if args.bridge == "juliacall"
                   else "subprocess（备桥）"),
        "juliacall": (importlib.metadata.version("juliacall") if args.bridge == "juliacall"
                      else "未使用（备桥）"),
        "julia": julia_version,
        "speedyweather": kv.get("speedy", "?"),
        "license": "SpeedyWeather.jl = EUPL-1.2（外部依赖，不 vendor、不改源码）",
        "env": env_used,
        "grid": {"npoints": npoints, "nlayers": nlayers, "truncation": args.truncation,
                 "note": "octahedral Gaussian（truncation=32 ≈ 旧 T21 档，48×96）"},
        "setup": ("PrimitiveDryModel(dynamics_only=true, orography=NoOrography, "
                  "land_sea_mask=AquaPlanetMask, ocean=AquaPlanet, land=nothing)"),
        "run": {"days": args.days, "wall_model_s": kv.get("wall"),
                "wall_call_s": round(main_run["wall_s"], 2)},
        "fields": {"u": kv.get("u"), "T": kv.get("T"), "ps_Pa": kv.get("ps_Pa"),
                   "finite": kv.get("finite")},
        "repeat_1day_bitwise": repeatable,
        "sweep_1to5d": sweep,
        "cross_bridge_bitwise": cross,
        "blockers": [],
        "notes": [
            "juliacall 进程内桥可用（2026-10-21 复核）；原报告“必然失败”结论已作废并更正。",
            "cross_bridge_bitwise 的参照物 = J0 首版 subprocess 产物（run_main 5 天 / run_repeat 1 天）。",
        ],
    }
    (OUT / "j0_report.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0 if kv.get("finite") == "true" else 1


if __name__ == "__main__":
    sys.exit(main())
