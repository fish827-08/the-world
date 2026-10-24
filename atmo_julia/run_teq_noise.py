"""J1′ 诊断臂：`teq_noise` 30 天（**非门**，不参与 P1′）。

协议依据：`内部汇报/裁定执行-20261021-C路线-甲P1prime与乙冻结.md` §三「诊断」行
（`teq_noise` 门后诊断，≥12 天；本轮取 30 天，快照 day 2–30 每 2 天 = 15 个）。

* 初值由 AV 冻结件 `benchmarks.atmo.jia_remap --ic teq_noise`（seed=0）材料化；
* 复用 `j1_core.jl` 与 `run_hs_julia.py` 的模型构造/环境/装配函数，**不改模型行为**；
* 输出 `native_teq/`（15 快照 npz + manifest.json）与 `run_teq.json`；
* 本目录仅供观察控制臂稳定性，**不产出 15 条判据值**。

用法（worktree 根，需先跑完 `run_hs_julia.py --stage geometry|ic`）：
    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe atmo_julia/run_teq_noise.py \
        --out-root _rerun_logs/20261021-A1-J1prime
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_hs_julia as runner  # noqa: E402  —— 复用 J1 运行器（字节导入，不改行为）

DAYS = 30
SNAP_START = 2.0
SNAP_EVERY = 2.0
SEED = 0
IC_NAME = "teq_noise"
FIELDS = ("u", "v", "T", "ps")


def _config_hash_teq(kv: dict, snap_days: list, days: float, dtype: str, ic_sha: str) -> str:
    """与 runner._config_hash 同式，仅把 ic.name 写成 teq_noise（诊断件专用）。"""
    cfg = {
        "grid": {"kind": "octahedral_gaussian", "truncation": int(kv["truncation"]),
                 "npoints": int(kv["npoints"]), "nlayers": int(kv["nlayers"]),
                 "sigma": runner.SIGMA_FULL},
        "dt_atmo_s": runner.DT_CLOCK, "days": days, "snap_days": snap_days,
        "seed": runner.SEED, "dtype": dtype,
        "ic": {"name": IC_NAME, "sha256": ic_sha},
        "forcing": runner._forcing(kv),
    }
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]


def materialize(out_root: Path) -> dict:
    from benchmarks.atmo import jia_remap  # noqa: PLC0415 —— 只读调用 AV 冻结件

    geom = out_root / "native_geom"
    teq_geom = out_root / "native_geom_teq"
    if teq_geom.exists():
        shutil.rmtree(teq_geom)
    shutil.copytree(geom, teq_geom)
    mf_path = teq_geom / "manifest.json"
    mf = json.loads(mf_path.read_text(encoding="utf-8"))
    mf["ic"] = {"name": IC_NAME, "sha256": "PENDING"}
    mf_path.write_text(json.dumps(mf, ensure_ascii=False, indent=1), encoding="utf-8")

    ic_path = teq_geom / f"ic_{IC_NAME}.npz"
    rep = jia_remap.materialize_ic(teq_geom, out_path=ic_path, name=IC_NAME, seed=SEED)
    ic = np.load(ic_path)
    for k in FIELDS:
        a = np.ascontiguousarray(ic[k], dtype="<f4")
        (teq_geom / f"ic_{IC_NAME}_{k}.bin").write_bytes(a.tobytes(order="C"))
    sha_bin = hashlib.sha256(
        b"".join((teq_geom / f"ic_{IC_NAME}_{k}.bin").read_bytes() for k in FIELDS)
    ).hexdigest()
    if sha_bin != rep["sha256"]:
        raise RuntimeError(f"teq_noise IC bin sha 不符: {sha_bin} != {rep['sha256']}")
    mf["ic"]["sha256"] = rep["sha256"]
    mf_path.write_text(json.dumps(mf, ensure_ascii=False, indent=1), encoding="utf-8")
    rep["ic_bins_sha256_recomputed"] = sha_bin
    (out_root / "ic_materialization_teq.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    return rep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True)
    args = ap.parse_args()
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    env_used = runner.setup_env(out_root)

    ic_rep = materialize(out_root)
    ic_sha = ic_rep["sha256"]
    ic_path = out_root / "native_geom_teq" / f"ic_{IC_NAME}.npz"

    jl = runner.julia_start()
    bindir = out_root / "_bin_teq"
    native = out_root / "native_teq"
    for d in (bindir, native):
        if d.exists():
            shutil.rmtree(d)
    bindir.mkdir(parents=True)
    native.mkdir(parents=True)

    t0 = time.time()
    summary = str(jl.j1_run(
        str(bindir), days=DAYS, dt_clock_s=runner.DT_CLOCK, nlayers=runner.NLAYERS,
        ic_path=str(ic_path), ic_sha256=ic_sha, dtype="float32",
        snap_start_day=SNAP_START, snap_every_day=SNAP_EVERY))
    wall = time.time() - t0
    kv = runner.parse_summary(summary)

    nsnap = len(list(bindir.glob("snap_*_u.bin")))
    snap_days = [SNAP_START + i * SNAP_EVERY for i in range(nsnap)]
    info = runner.assemble_native(bindir, native, kv, "float32", ic_sha, snap_days, float(DAYS))

    # 修 manifest：诊断臂的 ic 名与 config_hash（装配函数硬编码 iso_w1）
    mf_path = native / "manifest.json"
    mf = json.loads(mf_path.read_text(encoding="utf-8"))
    mf["ic"] = {"name": IC_NAME, "sha256": ic_sha}
    mf["config_hash"] = _config_hash_teq(kv, snap_days, float(DAYS), "float32", ic_sha)
    mf["initial_state"] = (f"jia:{IC_NAME}@{ic_sha[:12]}（AV 冻结件材料化，seed={SEED}；"
                           "T=HS Teq + 1 K·N(0,1)，u=v=0，ps=ln(1e5 Pa)）")
    mf["diagnostic"] = {"role": "teq_noise control arm (non-gate)", "seed": SEED,
                        "days": DAYS, "snap_days": snap_days,
                        "protocol": "C 路线执行件 §三「诊断」行（≥12 天）"}
    mf_path.write_text(json.dumps(mf, ensure_ascii=False, indent=1), encoding="utf-8")

    rep = {
        "runner": "atmo_julia/run_teq_noise.py", "role": "teq_noise diagnostic (non-gate)",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "env": env_used, "seed": SEED, "days": DAYS, "dt_atmo_s": runner.DT_CLOCK,
        "ic": ic_rep, "wall_s": round(wall, 2), "summary": summary,
        "n_snaps": nsnap, "snap_days": snap_days,
        "native_sha256": info["native_sha256"],
        "native_dir": str(native), "config_hash": mf["config_hash"],
        "note": "诊断臂，不参与 P1′；A1 不产出判据值。",
    }
    (out_root / "run_teq.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: rep[k] for k in ("wall_s", "n_snaps", "summary", "config_hash")},
                     ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
