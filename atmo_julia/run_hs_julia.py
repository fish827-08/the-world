"""J1 第二轮补件：SpeedyWeather.jl **原生 T32** Held–Suarez → AV 冻结输入契约。

协议来源：`benchmarks/atmo/gate/J1_mapping_freeze.md`（AV 冻结，2026-10-21）
  * A1 只交**原生场 + manifest**（(npoints, nlayers)、ps=ln(Pa)、lat/lon 原生环序）；
  * 初值 `iso_w1` 由 AV 冻结件 `benchmarks.atmo.jia_remap --ic iso_w1` 材料化，
    A1 施加并把 sha256 回填 manifest（本脚本以**只读**方式调用该冻结工具；
    报告中明确：AV 需独立重跑同一命令核对 sha256）；
  * 映射 / 判据归 AV；本脚本的 `convert` / `judge` 只做 A1 **自检**，不作签门依据。

阶段（可单独跑，`--stage all` 串起来）：
    geometry  → native_geom/（只含几何的首快照 + manifest）
    ic        → native_geom/ic_iso_w1.npz + sha256（回填 manifest）
    run       → native_run1/ native_run2/（float32，120 天）+ native_f64/（Float64 对照）
    convert   → av_run1/ av_run2/ av_f64/（AV 冻结映射件产出，供 AV 判据）
    judge     → av_report_t32.json（**A1 自检**；判据未改、签门归 AV）

用法（worktree 根）：
    PYTHONIOENCODING=utf-8 ../../.venv/Scripts/python.exe atmo_julia/run_hs_julia.py --stage all
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:          # 直接以脚本跑时 sys.path[0]=atmo_julia/，需补仓库根
    sys.path.insert(0, str(REPO))
VENV = Path(os.environ.get("A1_VENV", str(REPO.parents[1] / ".venv")))
PROJECT = REPO / "atmo_julia" / "julia_env"
CORE = REPO / "atmo_julia" / "j1_core.jl"
PY = Path(sys.executable)

SEED = 42
DAYS = 120
SNAP_START = 30.0
SNAP_EVERY = 2.0
TRUNCATION = 32
NLAYERS = 8
DT_CLOCK = 144
SIGMA_HALF = [0.0, 0.05, 0.14, 0.26, 0.40, 0.55, 0.70, 0.85, 1.00]
SIGMA_FULL = [0.5 * (a + b) for a, b in zip(SIGMA_HALF[:-1], SIGMA_HALF[1:])]
OUT_DEFAULT = REPO / "_rerun_logs" / "20261021-A1-J1"


def _ascii_root() -> Path:
    main = REPO.parents[1]
    alias = Path(r"C:\tw")
    try:
        if alias.resolve() == main.resolve():
            return alias / REPO.relative_to(main)
    except OSError:
        pass
    return REPO


def setup_env(out_root: Path) -> dict[str, str]:
    depot = os.environ.get("JULIA_DEPOT_PATH") or os.environ.get("A1_JULIA_DEPOT")
    if not depot:
        jd = Path(r"C:\jd")
        depot = str(jd if jd.exists() else _ascii_root() / "_rerun_logs" / "20261021-A1-julia" / "julia_depot")
    os.environ["JULIA_DEPOT_PATH"] = depot
    # TEMP 必须 ASCII：绝对 out_root 走仓库 ASCII 别名（C:\tw junction），否则 Julia Downloads 会崩
    ascii_root = _ascii_root()
    base = out_root if out_root.is_absolute() else REPO / out_root
    try:
        rel = base.resolve().relative_to(REPO.resolve())
        tmp_default = str(ascii_root / rel / "tmp")
    except ValueError:
        tmp_default = str(base / "tmp")
    tmp = os.environ.get("A1_JULIA_TMP") or tmp_default
    Path(tmp).mkdir(parents=True, exist_ok=True)
    for k in ("TEMP", "TMP", "TMPDIR"):
        os.environ[k] = tmp
    return {"depot": depot, "temp": tmp}


def parse_summary(line: str) -> dict[str, str]:
    kv: dict[str, str] = {}
    for tok in line.split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            kv[k] = v
    return kv


def _forcing(kv: dict) -> dict:
    return {
        "omega": float(kv["omega"]), "g": float(kv["g"]), "R": float(kv["R"]),
        "cp": float(kv["cp"]), "radius": float(kv["radius"]), "p0": float(kv["p0"]),
        "delta_Ty": float(kv["delta_Ty"]), "delta_theta_z": float(kv["delta_theta_z"]),
        "sigma_b": float(kv["sigma_b"]), "ka": float(kv["ka"]), "ks": float(kv["ks"]),
        "kf": float(kv["kf"]),
        # AV 协议三键（冻结值；见 gate/J1_mapping_freeze.md §6.1）
        "tmax_eq_K": 315.0, "tmin_eq_K": 200.0, "ps0_Pa": 100000.0,
    }


def _config_hash(kv: dict, snap_days: list, days: float, dtype: str, ic_sha: str, forcing: dict) -> str:
    cfg = {
        "grid": {"kind": "octahedral_gaussian", "truncation": int(kv["truncation"]),
                 "npoints": int(kv["npoints"]), "nlayers": int(kv["nlayers"]),
                 "sigma": SIGMA_FULL},
        "dt_atmo_s": DT_CLOCK, "days": days, "snap_days": snap_days, "seed": SEED,
        "dtype": dtype, "ic": {"name": "iso_w1", "sha256": ic_sha},
        "forcing": forcing,
    }
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]


def _declared_deviations() -> dict:
    return {
        "forcing_impl": "SpeedyWeather 0.23 HeldSuarez（源码注释标注 HS96，参数/公式与 HS94 一致）",
        "drag": "LinearDrag（σb=0.7, τ=1 d）；PrimitiveDryModel 默认 SpeedLimitDrag，已显式替换",
        "time_integration": "Leapfrog 半隐式；**状态步长 Δt = clk.Δt = 144 s**（72,001 步 = 120 天）；"
                            "2Δt=288 s 只是中心差分跨度，不是状态步长",
        "vertical_grid": "σ_half 逐位照抄 atmo_core::SIGMA_HALF（非均匀）；层中 0.2 恰 200 hPa ⇒ "
                         "解决 AV 冻结件 R-1（HS-T-TROP-UP 空窗）；属 A1 模型配置自由，判据阈值未动",
        "mass_diagnostic": "ps 为 SpeedyWeather 原生预报量 ln(Pa)，**未做 exp**；AV 映射件升 float64 后 exp",
        "teq_lnp": "HeldSuarez Teq 用 ln(p/p0)（p=σ·ps），非 ln σ；ps 漂 1% 时差 ~0.1 K（AV §6.3）",
        "hyperdiffusion": "SpeedyWeather 默认 HyperDiffusion（power=4, τ=4 h, div τ=1 h, resolution_scaling=1）",
        "ic_materialization": "IC 由 AV 冻结工具 benchmarks.atmo.jia_remap --ic iso_w1 材料化"
                              "（A1 只读调用、未改一行）；sha256 须由 AV 独立重跑同一命令核对",
        "determinism": "BLAS / FFTW 各 1 线程（j1_core.jl 顶部显式设置）；两次独立 j1_run 同进程逐位",
    }


def _env_block(kv: dict) -> dict:
    return {
        "julia": kv.get("julia", "?"),
        "speedyweather": kv.get("speedy", "?"),
        "nthreads": int(kv.get("nthreads", 1)),
        "blas_threads": 1,
        "fftw_threads": 1,
        "python": platform.python_version(),
        "juliacall": importlib.metadata.version("juliacall"),
        "os": platform.platform(),
    }


def write_manifest(path: Path, kv: dict, snap_days: list, days: float, dtype: str,
                   ic_sha: str) -> dict:
    forcing = _forcing(kv)
    manifest = {
        "benchmark": "held_suarez",
        "days": float(days),
        "dt_atmo_s": float(DT_CLOCK),          # 状态步长 Δt（不是 2Δt）
        "snap_days": [float(d) for d in snap_days],
        "obliquity_deg": 0.0,
        "seed": SEED,
        "config_hash": _config_hash(kv, snap_days, days, dtype, ic_sha, forcing),
        "grid": {
            "npoints": int(kv["npoints"]),
            "nlayers": int(kv["nlayers"]),
            "sigma": SIGMA_FULL,               # 满层中心（σ），判据 plev = sigma*1000 hPa
            "truncation": int(kv["truncation"]),
            "kind": "octahedral_gaussian",
            "nlat": int(kv["nlat"]),
            "nring": int(kv["nring"]) if "nring" in kv else None,
            "sigma_half": SIGMA_HALF,
        },
        "export": {
            "dtype": dtype,
            "ps_units": "log(Pa)",
            "field_shape": "npoints,nlayers",
            "byte_order": "little",
        },
        "ic": {"name": "iso_w1", "sha256": ic_sha},
        "env": _env_block(kv),
        "forcing": forcing,
        "declared_deviations": _declared_deviations(),
        "initial_state": f"jia:iso_w1@{ic_sha[:12]}（AV 冻结件材料化；u=v=0, T=288+1K·sin(lon), ps=ln(1e5 Pa)）",
        "cfl_reported": [],
    }
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    return manifest


# ---------------------------------------------------------------- Julia 侧

def julia_start():
    from juliacall import Main as jl  # noqa: PLC0415 —— 必须在 setup_env() 之后

    jl.seval(f'using Pkg; Pkg.activate(raw"{PROJECT}")')
    jl.seval("using SpeedyWeather")
    jl.include(str(CORE))
    return jl


def run_geometry(out_root: Path, jl) -> dict:
    bindir = out_root / "_bin_geom"
    native = out_root / "native_geom"
    for d in (bindir, native):
        if d.exists():
            shutil.rmtree(d)
    bindir.mkdir(parents=True)
    native.mkdir(parents=True)
    t0 = time.time()
    summary = str(jl.j1_geometry(str(bindir), nlayers=NLAYERS))
    wall = time.time() - t0
    kv = parse_summary(summary)

    npoints, nlev = int(kv["npoints"]), int(kv["nlayers"])
    lat = np.fromfile(bindir / "lat.bin", dtype=np.float64)
    lon = np.fromfile(bindir / "lon.bin", dtype=np.float64)
    sigma_full = np.fromfile(bindir / "sigma_full.bin", dtype=np.float64)
    sigma_half = np.fromfile(bindir / "sigma_half.bin", dtype=np.float64)
    if lat.size != npoints or lon.size != npoints or sigma_full.size != nlev or sigma_half.size != nlev + 1:
        raise RuntimeError(f"几何尺寸不符: npoints={npoints} lat={lat.size} lon={lon.size} "
                           f"sigma={sigma_full.size} edges={sigma_half.size}")
    if not np.allclose(sigma_full, SIGMA_FULL, rtol=0, atol=1e-15) or \
       not np.allclose(sigma_half, SIGMA_HALF, rtol=0, atol=1e-15):
        raise RuntimeError(f"σ 阶梯不符: {sigma_full.tolist()} / {sigma_half.tolist()}")

    # 契约自检：AV 冻结件的 ring_of 能还原原生环结构（48 环、20→112、无极点）
    from benchmarks.atmo.jia_remap import ring_of  # noqa: PLC0415 — 只读调用
    rings = ring_of(lat, lon)
    if rings["nring"] != int(kv["nlat"]) or rings["max_ring"] != 112:
        raise RuntimeError(f"环结构与契约不符: nring={rings['nring']} max_ring={rings['max_ring']}"
                           "（T32 原生应为 48 环 / 最大环 112 点）")

    zeros = np.zeros((npoints, nlev), dtype=np.float32)
    np.savez_compressed(
        native / "snap_00001.npz",
        u=zeros, v=zeros, T=zeros,
        ps=np.full(npoints, np.log(100000.0), dtype=np.float32),
        lat=lat, lon=lon,
    )
    manifest = write_manifest(native / "manifest.json", kv, [0.0], 0.0, "float32", "PENDING")
    rep = {
        "stage": "geometry", "bindir": str(bindir), "native_dir": str(native),
        "summary": summary, "wall_s": round(wall, 2),
        "npoints": npoints, "nlayers": nlev, "nring": rings["nring"],
        "ring_counts_head": rings["counts"][:6].tolist(),
        "ring_counts_tail": rings["counts"][-6:].tolist(),
        "lat_reach_deg": float(np.max(np.abs(rings["lat"]))),
        "sigma_full": SIGMA_FULL, "sigma_half": SIGMA_HALF,
        "manifest_config_hash": manifest["config_hash"],
    }
    print(json.dumps(rep, ensure_ascii=False, indent=1))
    return rep


def materialize_ic(out_root: Path) -> dict:
    from benchmarks.atmo import jia_remap  # noqa: PLC0415 — 只读调用 AV 冻结件

    native = out_root / "native_geom"
    rep = jia_remap.materialize_ic(native, out_path=native / "ic_iso_w1.npz", name="iso_w1")
    # Julia 侧读 npz 会踩 PythonCall 取值语义 ⇒ 旁挂小端 float32 原始字节；sha 以 bin 字节复算
    ic = np.load(native / "ic_iso_w1.npz")
    bins = {}
    for k in ("u", "v", "T", "ps"):
        a = np.ascontiguousarray(ic[k], dtype="<f4")
        p = native / f"ic_iso_w1_{k}.bin"
        p.write_bytes(a.tobytes(order="C"))
        bins[k] = {"path": str(p), "shape": list(a.shape), "dtype": "float32_le"}
    sha_bin = hashlib.sha256(b"".join((native / f"ic_iso_w1_{k}.bin").read_bytes()
                                     for k in ("u", "v", "T", "ps"))).hexdigest()
    if sha_bin != rep["sha256"]:
        raise RuntimeError(f"IC bin 字节 sha 与 AV 材料化报告不符: {sha_bin} != {rep['sha256']}")
    mf_path = native / "manifest.json"
    mf = json.loads(mf_path.read_text(encoding="utf-8"))
    mf["ic"]["sha256"] = rep["sha256"]
    mf_path.write_text(json.dumps(mf, ensure_ascii=False, indent=1), encoding="utf-8")
    rep["manifest_updated"] = str(mf_path)
    rep["ic_bins"] = bins
    rep["ic_sha256_recomputed_from_bins"] = sha_bin
    rep["note"] = ("A1 以只读方式调用 AV 冻结工具材料化 IC；AV 需独立重跑 "
                   "`python -m benchmarks.atmo.jia_remap --native native_geom --ic iso_w1` "
                   "核对 sha256（应逐位一致）后才签门。")
    (out_root / "ic_materialization.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=1))
    return rep


def _read_snap(bindir: Path, tag: str, npoints: int, nlev: int, dt: np.dtype) -> dict:
    def rd(name: str, shape):
        a = np.fromfile(bindir / f"snap_{tag}_{name}.bin", dtype=dt)
        if a.size != int(np.prod(shape)):
            raise RuntimeError(f"{name} 大小 {a.size} != {shape}")
        return np.ascontiguousarray(a.reshape(shape, order="F"))
    return {"u": rd("u", (npoints, nlev)), "v": rd("v", (npoints, nlev)),
            "T": rd("T", (npoints, nlev)), "ps": rd("ps", (npoints,))}


def assemble_native(bindir: Path, native: Path, kv: dict, dtype: str, ic_sha: str,
                    snap_days: list, days: float) -> dict:
    native.mkdir(parents=True, exist_ok=True)
    npoints, nlev = int(kv["npoints"]), int(kv["nlayers"])
    dt = np.dtype("<f4") if dtype == "float32" else np.dtype("<f8")
    lat = np.fromfile(bindir / "lat.bin", dtype=np.float64)
    lon = np.fromfile(bindir / "lon.bin", dtype=np.float64)
    sigma_full = np.fromfile(bindir / "sigma_full.bin", dtype=np.float64)
    sigma_half = np.fromfile(bindir / "sigma_half.bin", dtype=np.float64)
    if lat.size != npoints or lon.size != npoints or sigma_full.size != nlev or sigma_half.size != nlev + 1:
        raise RuntimeError("坐标尺寸不符")
    tags = sorted(p.name[len("snap_"):-len("_u.bin")] for p in bindir.glob("snap_*_u.bin"))
    if [t for t in tags] != [f"{i:05d}" for i in range(1, len(tags) + 1)]:
        raise RuntimeError(f"快照编号不连续: {tags[:3]}…")
    shas = []
    for i, tag in enumerate(tags, start=1):
        snap = _read_snap(bindir, tag, npoints, nlev, dt)
        np.savez_compressed(native / f"snap_{i:05d}.npz", lat=lat, lon=lon, **snap)
        h = hashlib.sha256(b"".join(snap[k].tobytes() for k in ("u", "v", "T", "ps"))).hexdigest()
        shas.append(h)
    write_manifest(native / "manifest.json", kv, snap_days, days, dtype, ic_sha)
    return {"n_snaps": len(tags), "native_sha256": shas}


def run_one(out_root: Path, jl, tag: str, ic_path: Path, dtype: str,
            days: int = DAYS) -> dict:
    bindir = out_root / f"_bin_{tag}"
    native = out_root / f"native_{tag}"
    for d in (bindir, native):
        if d.exists():
            shutil.rmtree(d)
    bindir.mkdir(parents=True)
    ic_sha = json.loads((out_root / "native_geom" / "manifest.json").read_text(encoding="utf-8"))["ic"]["sha256"]
    t0 = time.time()
    summary = str(jl.j1_run(str(bindir), days=days, dt_clock_s=DT_CLOCK, nlayers=NLAYERS,
                            ic_path=str(ic_path), ic_sha256=ic_sha, dtype=dtype,
                            snap_start_day=SNAP_START, snap_every_day=SNAP_EVERY))
    wall = time.time() - t0
    kv = parse_summary(summary)
    nsnap = len(list(bindir.glob("snap_*_u.bin")))
    snap_days = [SNAP_START + i * SNAP_EVERY for i in range(nsnap)]
    info = assemble_native(bindir, native, kv, dtype, ic_sha, snap_days, float(days))
    if float(kv["dt_step_s"]) != float(DT_CLOCK):
        raise RuntimeError(f"dt_step_s={kv['dt_step_s']} != {DT_CLOCK}（元数据自相矛盾）")
    rep = {"tag": tag, "dir": str(native), "dtype": dtype, "wall_s": round(wall, 2),
           "summary": summary, "n_snaps": nsnap, "snap_days": snap_days,
           "native_sha256_head": info["native_sha256"][:3],
           "native_sha_n": info["native_sha256"],
           "config_hash": json.loads((native / "manifest.json").read_text(encoding="utf-8"))["config_hash"]}
    (out_root / f"run_{tag}.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1),
                                              encoding="utf-8")
    print(json.dumps({k: rep[k] for k in ("tag", "wall_s", "n_snaps", "summary")},
                     ensure_ascii=False, indent=1))
    return rep


# ---------------------------------------------------------------- AV 冻结件（只读调用）

def convert(out_root: Path, native_tag: str, av_tag: str) -> dict:
    native = out_root / f"native_{native_tag}"
    av = out_root / f"av_{av_tag}"
    if av.exists():
        shutil.rmtree(av)
    rep_path = out_root / f"remap_{av_tag}.json"
    cmd = [str(PY), "-m", "benchmarks.atmo.jia_remap", "--native", str(native),
           "--out", str(av), "--rep", str(rep_path)]
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=1800)
    (out_root / f"remap_{av_tag}_stdout.txt").write_text(r.stdout + "\n" + r.stderr, encoding="utf-8")
    if not rep_path.exists():
        raise RuntimeError(f"映射报告缺失（{av_tag}, rc={r.returncode}）：\n{r.stdout}\n{r.stderr}")
    rep = json.loads(rep_path.read_text(encoding="utf-8"))
    rep["_av_dir"] = str(av)
    rep["_cli_rc"] = r.returncode
    # AV 冻结件自检：单调性判据是**绝对** 1e-9（ps 量级 1e5 Pa ⇒ 相对 1e-14，低于 float64 舍入）。
    # 仅当全部告警都是"映射越界"且面积均值自证 0 条时才继续；其余任何告警一律致命。
    bad = rep.get("self_check", [])
    only_bounds = bool(bad) and all(str(x.get("why", "")).startswith("映射越界") for x in bad)
    rep["_self_check_class"] = ("bounds-roundoff" if only_bounds else
                                ("clean" if not bad else "fatal"))
    if r.returncode != 0 and not only_bounds:
        raise RuntimeError(f"映射失败（{av_tag}, rc={r.returncode}，"
                           f"self_check_bad={rep.get('self_check_bad')}）：\n{r.stdout}\n{r.stderr}")
    return rep


def judge(out_root: Path, av_a: str, av_b: str, out_name: str) -> dict:
    cmd = [str(PY), "-m", "benchmarks.atmo.run", "--suite", "held_suarez",
           "--run-dir", str(out_root / f"av_{av_a}"), "--run-dir-b", str(out_root / f"av_{av_b}"),
           "--out", str(out_root / out_name)]
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=1800)
    (out_root / (out_name.replace(".json", "_stdout.txt"))).write_text(r.stdout + "\n" + r.stderr,
                                                                      encoding="utf-8")
    rep = {}
    if (out_root / out_name).exists():
        rep = json.loads((out_root / out_name).read_text(encoding="utf-8"))
    return {"rc": r.returncode, "report": str(out_root / out_name),
            "verdict": rep.get("verdict"), "criteria": rep.get("criteria")}


def mass_diag(out_root: Path, av_tag: str) -> dict:
    cmd = [str(PY), str(REPO / "atmo_julia" / "check_mass_diag.py"), str(out_root / f"av_{av_tag}")]
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=600)
    rep = {}
    try:
        rep = json.loads(r.stdout)
    except json.JSONDecodeError:
        rep = {"raw_tail": r.stdout.strip()[-800:]}
    return {"rc": r.returncode, "diag": rep}


# ---------------------------------------------------------------- 主流程

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["geometry", "ic", "run", "convert", "judge", "finalize", "all"])
    ap.add_argument("--days", type=int, default=DAYS)
    ap.add_argument("--skip-f64", action="store_true", help="跳过 Float64 对照跑")
    ap.add_argument("--out-root", default=str(OUT_DEFAULT))
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    env_used = setup_env(out_root)
    report = {"runner": "atmo_julia/run_hs_julia.py (native T32, round 2)",
              "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "python": platform.python_version(),
              "juliacall": importlib.metadata.version("juliacall"),
              "license": "SpeedyWeather.jl = EUPL-1.2（未修改外部依赖，不 vendor）",
              "env": env_used,
              "protocol": "benchmarks/atmo/gate/J1_mapping_freeze.md",
              "gate": "A1 自检仅供参考；J1 门由 AV 独立材料化 IC / 重映射 / 裁判并签"}

    jl = None
    if args.stage in ("geometry", "run", "all"):
        jl = julia_start()

    if args.stage in ("geometry", "all"):
        report["geometry"] = run_geometry(out_root, jl)

    if args.stage in ("ic", "all"):
        report["ic"] = materialize_ic(out_root)

    if args.stage in ("run", "all"):
        ic = out_root / "native_geom" / "ic_iso_w1.npz"
        if not ic.exists():
            raise SystemExit(f"缺 IC: {ic}（先跑 --stage ic）")
        report["run1"] = run_one(out_root, jl, "run1", ic, "float32", args.days)
        report["run2"] = run_one(out_root, jl, "run2", ic, "float32", args.days)
        if not args.skip_f64:
            report["run_f64"] = run_one(out_root, jl, "f64", ic, "float64", args.days)

    if args.stage == "finalize":          # 复用已完成的跑批：只做映射/自检/汇总
        for t in ("run1", "run2", "f64"):
            p = out_root / f"run_{t}.json"
            if p.exists():
                report[t] = json.loads(p.read_text(encoding="utf-8"))

    if args.stage in ("convert", "finalize", "all"):
        tags = [("run1", "run1"), ("run2", "run2")]
        if (out_root / "native_f64").exists() and not args.skip_f64:
            tags.append(("f64", "f64"))
        report["remap"] = {}
        for native_tag, av_tag in tags:
            report["remap"][av_tag] = convert(out_root, native_tag, av_tag)

    if args.stage in ("judge", "finalize", "all"):
        report["judge_t32"] = judge(out_root, "run1", "run2", "av_report_t32.json")
        report["mass_diag_f32"] = mass_diag(out_root, "run1")
        if (out_root / "av_f64").exists() and not args.skip_f64:
            report["mass_diag_f64"] = mass_diag(out_root, "f64")

    # 原生两跑逐位（CONS-DET 的裁判侧交叉核对）
    if "run1" in report and "run2" in report:
        a = report["run1"]["native_sha_n"]
        b = report["run2"]["native_sha_n"]
        report["native_determinism"] = {
            "n": len(a), "mismatches": sum(1 for x, y in zip(a, b) if x != y),
            "manifest_equal": (out_root / "native_run1" / "manifest.json").read_bytes()
                              == (out_root / "native_run2" / "manifest.json").read_bytes(),
        }

    (out_root / "j1_round2_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items()
                      if k in ("run1", "run2", "run_f64", "native_determinism",
                               "judge_t32", "mass_diag_f32", "mass_diag_f64")},
                     ensure_ascii=False, indent=1)[:4000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
