"""AV 输入契约：atmo 输出快照目录（只读加载与校验）。

契约（依派工 20261020 §二；INTERFACE.md 冻结后若有出入，走接口变更单并在本文件升级）：
  <run_dir>/
    meta.json        必含：benchmark, dt_atmo_s, days, snap_days[], obliquity_deg,
                     seed, config_hash, cfl_reported[]（可选）；
                     湿套件另需：slab_heat_capacity, radiation(字符串描述), forcing{}
    snap_000000.npz…  每快照：u,v,T,ps 必备 (nlev,nlat,nlon)/ps (nlat,nlon)；
                     湿套件另需 q,pr,evap,sw_toa,lw_out,sst（SI；pr/evap kg m-2 s-1）
    坐标文件 coords.npz：lat (nlat,), lon (nlon,)；
                     plev_hpa (nlev,) 或 sigma (nlev,)（湿套件用 sigma 时需 sigma_edges (nlev+1,)）
"""
import json
from pathlib import Path

import numpy as np

DRY_SNAP_FIELDS = ("u", "v", "T", "ps")
WET_SNAP_FIELDS = DRY_SNAP_FIELDS + ("q", "pr", "evap", "sw_toa", "lw_out", "sst")
DRY_META_KEYS = ("benchmark", "dt_atmo_s", "days", "snap_days", "obliquity_deg", "seed", "config_hash")
WET_META_KEYS = DRY_META_KEYS + ("slab_heat_capacity", "radiation")


def load_run(run_dir):
    rd = Path(run_dir)
    if not rd.exists():
        raise ValueError(f"run_dir 不存在: {rd}")
    meta_path = rd / "meta.json"
    if not meta_path.exists():
        raise ValueError(f"缺 meta.json: {rd}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    wet = meta.get("benchmark") == "aquaplanet"
    missing_meta = [k for k in (WET_META_KEYS if wet else DRY_META_KEYS) if k not in meta]
    if missing_meta:
        raise ValueError(f"meta.json 缺键: {missing_meta}")

    coords_path = rd / "coords.npz"
    if not coords_path.exists():
        raise ValueError(f"缺 coords.npz: {rd}")
    with np.load(coords_path, allow_pickle=False) as d:
        coords = {k: d[k].astype(np.float64) for k in d.files}
    for k in ("lat", "lon"):
        if k not in coords:
            raise ValueError(f"coords.npz 缺 {k}")
    if "plev_hpa" not in coords and "sigma" not in coords:
        raise ValueError("coords.npz 需 plev_hpa 或 sigma 之一")
    if "sigma" in coords and "sigma_edges" not in coords:
        raise ValueError("sigma 坐标需 sigma_edges (nlev+1,)")

    snaps = []
    for p in sorted(rd.glob("snap_*.npz")):
        with np.load(p, allow_pickle=False) as d:
            snaps.append({k: d[k].astype(np.float64) for k in d.files})
    if not snaps:
        raise ValueError(f"无 snap_*.npz: {rd}")
    need = WET_SNAP_FIELDS if wet else DRY_SNAP_FIELDS
    missing_fields = sorted(set(need) - set(snaps[0]))
    if missing_fields:
        raise ValueError(f"快照缺场: {missing_fields}（契约见本文件 docstring）")

    nlat, nlon = len(coords["lat"]), len(coords["lon"])
    for i, s in enumerate(snaps):
        for k in need:
            if k == "ps" or k in ("sw_toa", "lw_out", "sst", "pr", "evap"):
                exp = (nlat, nlon)
            else:
                exp = None
            if exp is not None and s[k].shape != exp:
                raise ValueError(f"snap[{i}].{k} 形状 {s[k].shape} != {exp}")
            if exp is None and s[k].ndim != 3:
                raise ValueError(f"snap[{i}].{k} 需 (nlev,nlat,nlon)，实为 {s[k].shape}")
            if not s[k].flags["C_CONTIGUOUS"]:
                s[k] = np.ascontiguousarray(s[k])
    return {"meta": meta, "coords": coords, "snaps": snaps, "wet": wet}
