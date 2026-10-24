"""J1 质量漂移口径诊断（只读）：复核 CONS-MASS 实测值的构成，不改 AV 判据。

用法（仓库根）：
    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe atmo_julia/check_mass_diag.py \
        _rerun_logs/20261021-A1-J1/hs_julia_run1

输出（JSON，stdout）：
    judge_*       —— 直接调用 AV 冻结实现 conservation.mass_drift / ps_global_mean_series 的读数与返回形状
    corrected_*   —— 对经度取平均（真正的全球面积加权平均）后的漂移
    decomposition —— mean(ln ps) 漂移与 var(ln ps) 增长（ps=exp(ln ps) 的方差项）

纪律：只读 run_dir；不写 benchmarks/atmo/；判定仍归 AV。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from benchmarks.atmo import conservation  # noqa: E402 —— AV 冻结实现，只读调用
from benchmarks.atmo.loader import load_run  # noqa: E402


def diagnose(run_dir: Path) -> dict:
    r = load_run(run_dir)
    ps = np.stack([s["ps"] for s in r["snaps"]]).astype(np.float64)
    lat = np.asarray(r["coords"]["lat"], dtype=np.float64)
    days = np.asarray(r["meta"]["snap_days"], dtype=np.float64)

    # ① AV 冻结实现的原始读数（返回 (nt, nlon)，未对经度归约）
    series = conservation.ps_global_mean_series(ps, lat)
    rel = np.abs((series - series[0]) / series[0])
    i, j = np.unravel_index(int(np.argmax(rel)), rel.shape)

    # ② 补经度归约后的全球平均
    glob = series.mean(axis=-1)
    rel_glob = np.abs((glob - glob[0]) / glob[0])
    k = int(np.argmax(rel_glob))

    # ③ ln(ps) 均值/方差分解
    w = np.cos(np.deg2rad(lat))
    wb = np.broadcast_to(w[:, None], ps.shape[1:])
    norm = wb.sum()
    ln = np.log(ps)
    mean_ln = (ln * wb).sum(axis=(1, 2)) / norm
    var_ln = ((ln - mean_ln[:, None, None]) ** 2 * wb).sum(axis=(1, 2)) / norm

    return {
        "run_dir": str(run_dir),
        "n_snaps": int(ps.shape[0]),
        "judge_mass_drift": float(conservation.mass_drift(ps, lat)),
        "judge_series_shape": list(series.shape),
        "judge_worst": {
            "snap_index": int(i), "day": float(days[i]),
            "lon_index": int(j),
            "lon_deg": float(np.asarray(r["coords"]["lon"])[j]),
            "zonal_mean_ps_pa": float(series[i, j]),
            "zonal_mean_ps_t0_pa": float(series[0, j]),
        },
        "corrected_lon_mean_max_drift": float(rel_glob.max()),
        "corrected_lon_mean_worst_day": float(days[k]),
        "mean_ln_ps_drift_day_last": float(mean_ln[-1] - mean_ln[0]),
        "var_ln_ps_day_first": float(var_ln[0]),
        "var_ln_ps_max": float(var_ln.max()),
        "var_ln_ps_day_last": float(var_ln[-1]),
        "exp_var_half_rel_first_to_last": float(np.exp((var_ln[-1] - var_ln[0]) / 2) - 1),
        "note": "judge_* = AV 冻结实现读数（未归约经度）；corrected_* = 补经度平均；判定归 AV",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    args = ap.parse_args()
    print(json.dumps(diagnose(args.run_dir), ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
