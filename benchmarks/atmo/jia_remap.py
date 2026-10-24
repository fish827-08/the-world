"""甲方案（SpeedyWeather.jl）原生输出 → AV loader 契约 的映射件（J1 裁判侧，2026-10-21 冻结）。

为什么映射归 AV：甲为 **Float32 + octahedral 约化高斯**（每环点数随纬度变化、无极点），而
`benchmarks/atmo/loader.py` 契约要求规则 `(nlev,nlat,nlon)`。映射方法是判定链条上的
自由度——若由被测方（A1）选映射并自跑自判，等于把判据交给被测方。故拆分：
**A1 只交原生场 + manifest，本件冻结映射与协议并独立产出 run 目录，判据/阈值一律不动**。

冻结的映射方法（逐条理由见 `gate/J1_mapping_freeze.md`）
--------------------------------------------------
1. **一阶保守面积重叠（finite-volume band-overlap），μ 与 λ 两向重叠面积相乘**：
   目标单元值 = ∑(重叠面积 × 源值) / 目标单元面积。选它而非双线性/谱重采样，因为
   ① 全球面积积分在 float64 舍入内精确守恒 ⇒ CONS-MASS 只测被测方、不掺映射假信号；
   ② 单调（目标值恒为源值凸组合 ⇒ 映射不会自己造出急流/EKE 极值）；③ 无插值相位漂移。
2. **纬度用 μ=sinφ**：原生环带宽取相邻节点 μ 中点，两端钉在 ±μ_max；目标为等 Δμ 单元。
   二者 μ 覆盖区间严格相同 ⇒ 面积账本闭合，不重不漏（`mu_coverage_defect` 断言）。
3. **极点处理**：约化高斯网格**没有极点**。μ 覆盖钉死在原生最大 ∣μ∣，**不做外推、不填极**；
   极帽外面积不进任何统计，覆盖率登记进 `meta.mapping`。
4. **升 Float64 时机**：读入即升 float64，映射全程 float64；磁盘 dtype 只在 manifest 声明。
   ps 若为 `log(Pa)`，在**升精度之后**才 exp（避免 float32 下 exp 丢尾数）。
5. **归约顺序固定**：环升序 → 目标行升序 → 列升序 ⇒ 映射自身逐位可复现。
6. **目标分辨率规则（由数据决定，不许人为选）**：`nlat = 原生环数`（纬度不降采样）、
   `nlon = 2 × max(环点数)`（经度只升不降，绝不丢原生信息）。

冻结的协议自由度（评审源码核查 2026-10-21 移交 AV 裁定，见冻结文档 §6）
--------------------------------------------------------------
- `Teq` 上限常数 **tmax_eq_K = 315 K**（HS94 原文 = SpeedyWeather 默认；乙的 288 不继承，理由见 §6.1）
  `tmin_eq_K = 200 K`、参考面压 `ps0_Pa = 100000 Pa`——`HS-FORCING` 的 12 键不含它们，
  故在**输入契约层**钉住（不符 ⇒ INPUT_CONTRACT_ERROR），`criteria.py`/`suites.py` 一字不动。
- 初值协议 `iso_w1`（= 288 K 等温 + 1 K·sin(lon) 波数 1，u=v=0，ps=ps0）为**主判臂**，
  `teq_noise` 为对照臂；IC 由本件 `materialize_ic` 材料化到甲的原生网格并给 sha256，A1 施加。

原生 run 目录契约（A1 必须满足；不满足 ⇒ INPUT_CONTRACT_ERROR，AV 不迁就适配）
------------------------------------------------------------------------
    <native_dir>/
      manifest.json      必含 MANIFEST_KEYS；grid 含 GRID_KEYS；export 含 EXPORT_KEYS；
                         forcing 含 HS94 十二键（SI；由 40 d/4 d/1 d 时间尺度换算而来）+
                         PROTOCOL_FORCING_KEYS；ic = {name, sha256}；env = {julia,
                         speedyweather, nthreads}；HS 套件另需声明已用 LinearDrag（kf>0）
      snap_%05d/         每快照：u.bin v.bin T.bin ps.bin lat.bin lon.bin
      或 snap_%05d.npz   同键名（J0 的 j0_fields.npz 风格）
      场布局：u/v/T = (npoints, nlayers)，ps/lat/lon = (npoints,)，小端 float32/float64
      第二跑（CONS-DET 用）：另一个同结构 native 目录，经本件转换后交 `run.py --run-dir-b`

J2 湿基准扩展（2026-10-22 提案，待评审；R-8 同模式：默认路径逐位不变）
------------------------------------------------------------------
- `benchmark=aquaplanet` 时字段集扩为 u,v,T,q / ps,pr,evap,sw_toa,lw_out,sst（顺序即导出序）；
  干燥臂（held_suarez）字段集、目标网格与产物**逐位不变**。
- `coords.sigma_edges` 直接取 `manifest.grid.sigma_half`（层界真源），**不由 sigma 中点反推**；
  `nlayers` 一律由 manifest 读（支持 16–24 层，不写死 8）。
- 目标网格仍为等 Δμ（nlat=原生环数、nlon=2×max_ring）⇒ v1 中点面积权精确，不导出 `mu_edges`，
  `suites.CONS-MASS` 路径不动。
- 湿臂附加输入契约见 `_validate_wet_manifest`；`moist_scheme` 只登记不校验语义，
  `pr/evap` 单位必须声明 `kg m-2 s-1`，通量符号原样透传（不取绝对值、不改号）。
- 提案与证据：`gate/J2_wet_remap_extension_20261022.md`。

J2 Track A reader（2026-10-22，评审派工；接口单 `gate/J2_trackA_reader_20261022.md`）
----------------------------------------------------------------------------
- 湿 manifest 的五个累积通量登记键（`FLUX_META_KEYS`）**白名单透传**到目标 `meta.json`：
  `flux_semantics/flux_interval_s/flux_accumulation/flux_valid_from_snap/flux_first_interval_s`。
  manifest 没有这些键（瞬时率模式 / 干臂）⇒ 目标 meta 不出现这些键，产物逐位不变。
- 判定侧的切片在 `suites.py`（`flux_semantics=="interval_mean"` ⇒ pr/evap 类判据从第 2 个快照起，
  CONS-WATER 的 W/days 用全窗）；本件只搬登记字段，不解释语义、不改判据。
- 五个键必须在 manifest **顶层**；只出现在 `manifest.export` 而顶层缺失 ⇒ `ValueError`（防错位）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from benchmarks.atmo import metrics, run_hs, suites

SNAP_FIELDS_3D = ("u", "v", "T")
SNAP_FIELDS_2D = ("ps",)
# J2 湿基准（aquaplanet）：字段集由契约固定；顺序即导出/自检顺序（干燥臂逐位不变）
WET_SNAP_FIELDS_3D = SNAP_FIELDS_3D + ("q",)
WET_SNAP_FIELDS_2D = SNAP_FIELDS_2D + ("pr", "evap", "sw_toa", "lw_out", "sst")
NATIVE_COORDS = ("lat", "lon")
SNAP_FIELDS = SNAP_FIELDS_3D + SNAP_FIELDS_2D + NATIVE_COORDS
WET_SNAP_FIELDS = WET_SNAP_FIELDS_3D + WET_SNAP_FIELDS_2D + NATIVE_COORDS
BENCHMARKS = ("held_suarez", "aquaplanet")
# Track A：A2 区间累积通量的登记键（单源在 suites；缺失 ⇒ 不写入目标 meta）
FLUX_META_KEYS = suites.FLUX_META_KEYS

MANIFEST_KEYS = ("benchmark", "days", "dt_atmo_s", "snap_days", "obliquity_deg",
                 "seed", "config_hash", "grid", "forcing", "export", "ic", "env")
GRID_KEYS = ("npoints", "nlayers", "sigma", "truncation")
EXPORT_KEYS = ("dtype", "ps_units", "field_shape", "byte_order")
IC_KEYS = ("name", "sha256")
ENV_KEYS = ("julia", "speedyweather", "nthreads")
HS94_FORCING_KEYS = tuple(suites.HS94_PARAMS)          # 12 键，真源在 suites
PROTOCOL_FORCING_KEYS = ("tmax_eq_K", "tmin_eq_K", "ps0_Pa")

# 冻结协议值（改动须走接口变更单 + 本文档，不改判据阈值）
PROTOCOL = {"tmax_eq_K": 315.0, "tmin_eq_K": 200.0, "ps0_Pa": 100000.0}
PROTOCOL_TOL = 0.005
IC_PROTOCOL = {
    "iso_w1": {"T0_K": 288.0, "amp_K": 1.0, "wave": 1},     # 主判臂
    "teq_noise": {"amp_K": 1.0},                            # 对照臂（乙 P1 下 ~105 步即炸）
}


# ---------------------------------------------------------------- 原生读取与校验

def _fields_for(mf: dict) -> tuple:
    """字段集由 benchmark 决定；干燥臂即 J1 冻结集（不读 sigma_half，产物逐位不变）。"""
    b = mf.get("benchmark")
    if b == "held_suarez":
        return SNAP_FIELDS_3D, SNAP_FIELDS_2D
    if b == "aquaplanet":
        return WET_SNAP_FIELDS_3D, WET_SNAP_FIELDS_2D
    raise ValueError(f"benchmark 需属 {BENCHMARKS}，实为 {b!r}")


SIGMA_EDGE_TOL = 1e-12


def _validate_wet_manifest(mf: dict) -> None:
    """J2 湿臂附加契约（J2_prereg §3）。只做边界值校验：moist_scheme 登记内容不经本件判语义。"""
    wmiss = [k for k in ("slab_heat_capacity", "radiation", "moist_scheme") if k not in mf]
    if wmiss:
        raise ValueError(f"aquaplanet manifest 缺键: {wmiss}")
    grid, export = mf["grid"], mf["export"]
    misplaced = [k for k in FLUX_META_KEYS if k in export and k not in mf]
    if misplaced:
        raise ValueError(
            f"flux 登记键只出现在 manifest.export 而顶层缺失: {misplaced}"
            "（Track A 契约要求顶层；见 gate/J2_trackA_reader_20261022.md）")
    nlev = int(grid["nlayers"])
    if nlev < 1:
        raise ValueError(f"grid.nlayers 需 ≥1，实为 {nlev}")
    if "sigma_half" not in grid:
        raise ValueError("aquaplanet manifest.grid 缺 sigma_half（层界 σ 真源；不得由 sigma 中点反推）")
    half = np.asarray(grid["sigma_half"], dtype=np.float64)
    if half.ndim != 1 or half.size != nlev + 1:
        raise ValueError(f"grid.sigma_half 形状 {half.shape} != ({nlev + 1},)")
    if not np.all(np.isfinite(half)):
        raise ValueError("grid.sigma_half 含非有限值")
    if not np.all(np.diff(half) > 0.0):
        raise ValueError("grid.sigma_half 需严格递增")
    if abs(half[0]) > SIGMA_EDGE_TOL or abs(half[-1] - 1.0) > SIGMA_EDGE_TOL:
        raise ValueError(f"grid.sigma_half 首尾需 0/1，实为 {half[0]!r}..{half[-1]!r}")
    sigma = np.asarray(grid["sigma"], dtype=np.float64)
    if sigma.size != nlev:
        raise ValueError(f"grid.sigma 长度 {sigma.size} != nlayers {nlev}")
    if not np.allclose(0.5 * (half[:-1] + half[1:]), sigma, atol=1e-9, rtol=0):
        raise ValueError("grid.sigma 与 sigma_half 的中点不一致（层厚/层心不匹配）")
    if not isinstance(mf["forcing"], dict):
        raise ValueError("manifest.forcing 需为对象（湿臂允许空 {}）")
    try:
        cap = float(mf["slab_heat_capacity"])
    except (TypeError, ValueError):
        raise ValueError(f"slab_heat_capacity 需数值，实为 {mf['slab_heat_capacity']!r}") from None
    if not math.isfinite(cap) or cap <= 0.0:
        raise ValueError(f"slab_heat_capacity 需有限正数，实为 {mf['slab_heat_capacity']!r}")
    if not isinstance(mf["radiation"], str) or not mf["radiation"].strip():
        raise ValueError("radiation 需非空字符串（AQ-FORCING 只查存在性）")
    if not isinstance(mf["moist_scheme"], dict) or not mf["moist_scheme"]:
        raise ValueError("moist_scheme 需非空对象（登记用；本件不校验方案语义）")
    for k in ("pr_units", "evap_units"):
        if export.get(k) != "kg m-2 s-1":
            raise ValueError(f"export.{k} 需 'kg m-2 s-1'，实为 {export.get(k)!r}")
    for k, want in (("snap_fields_3d", WET_SNAP_FIELDS_3D), ("snap_fields_2d", WET_SNAP_FIELDS_2D)):
        if k in export and set(export[k]) != set(want):
            raise ValueError(f"export.{k}={export[k]!r} 与契约字段集 {sorted(want)!r} 不符")


def _validate_manifest(mf: dict) -> None:
    missing = [k for k in MANIFEST_KEYS if k not in mf]
    if missing:
        raise ValueError(f"manifest 缺键: {missing}")
    for block, keys in (("grid", GRID_KEYS), ("export", EXPORT_KEYS),
                        ("ic", IC_KEYS), ("env", ENV_KEYS)):
        sub = mf[block]
        if not isinstance(sub, dict):
            raise ValueError(f"manifest.{block} 需为对象，实为 {type(sub).__name__}")
        gmiss = [k for k in keys if k not in sub]
        if gmiss:
            raise ValueError(f"manifest.{block} 缺键: {gmiss}")
    if mf["export"]["ps_units"] not in ("Pa", "log(Pa)"):
        raise ValueError(f"export.ps_units 只许 Pa / log(Pa)，实为 {mf['export']['ps_units']!r}")
    if str(mf["export"]["dtype"]) not in ("float32", "float64"):
        raise ValueError(f"export.dtype 需 float32/float64，实为 {mf['export']['dtype']!r}")
    if str(mf["export"]["byte_order"]) not in ("little", "<"):
        raise ValueError(f"export.byte_order 需 little，实为 {mf['export']['byte_order']!r}")
    if str(mf["export"]["field_shape"]) != "npoints,nlayers":
        raise ValueError(f"export.field_shape 需 'npoints,nlayers'，实为 {mf['export']['field_shape']!r}")
    b = mf["benchmark"]
    if b not in BENCHMARKS:
        raise ValueError(f"benchmark 需属 {BENCHMARKS}，实为 {b!r}")
    if b == "aquaplanet":
        if not isinstance(mf["ic"]["name"], str) or not mf["ic"]["name"].strip():
            raise ValueError(f"aquaplanet ic.name 需非空字符串，实为 {mf['ic']['name']!r}")
        if not isinstance(mf["ic"].get("sha256"), str) or not mf["ic"]["sha256"].strip():
            raise ValueError("aquaplanet ic.sha256 需非空（跑前登记，见 J2_prereg §11.5）")
        _validate_wet_manifest(mf)
        return
    if mf["ic"]["name"] not in IC_PROTOCOL:
        raise ValueError(f"ic.name 需属 {sorted(IC_PROTOCOL)}，实为 {mf['ic']['name']!r}")
    fmiss = [k for k in HS94_FORCING_KEYS + PROTOCOL_FORCING_KEYS if k not in mf["forcing"]]
    if fmiss:
        raise ValueError(f"manifest.forcing 缺键: {fmiss}（HS94 十二键 SI + 协议三键）")
    for k, ref in PROTOCOL.items():
        got = float(mf["forcing"][k])
        if abs(got - ref) > PROTOCOL_TOL * abs(ref):
            raise ValueError(
                f"forcing.{k}={got:g} 与冻结协议值 {ref:g} 不符（容差 {PROTOCOL_TOL:.1%}）；"
                f"改协议须先提交接口变更单，见 gate/J1_mapping_freeze.md §6")


def _dtype(mf) -> np.dtype:
    return np.dtype("<f4") if str(mf["export"]["dtype"]) == "float32" else np.dtype("<f8")


def _read_native_snap(d, mf: dict) -> dict:
    """一个快照（目录或 npz）→ float64 原生数组；升精度在此完成（冻结项 4）。
    读哪些场由 benchmark 决定：干燥臂 = J1 冻结集；湿臂 = J2 契约集。"""
    dt = _dtype(mf)
    f3, f2 = _fields_for(mf)
    fields = f3 + f2 + NATIVE_COORDS
    nlayers = int(mf["grid"]["nlayers"])
    npoints = int(mf["grid"]["npoints"])
    out = {}
    if Path(d).is_dir():
        for k in fields:
            p = Path(d) / f"{k}.bin"
            if not p.exists():
                raise ValueError(f"原生快照缺 {p.name}: {d}")
            out[k] = np.fromfile(p, dtype=dt).astype(np.float64)
    else:
        with np.load(d, allow_pickle=False) as z:
            miss = sorted(set(fields) - set(z.files))
            if miss:
                raise ValueError(f"原生快照 npz 缺场: {miss}（{d}）")
            for k in fields:
                out[k] = np.asarray(z[k], dtype=np.float64).ravel()
    for k in f3:
        if out[k].size != npoints * nlayers:
            raise ValueError(f"{k} 元素数 {out[k].size} != npoints*nlayers = {npoints * nlayers}")
        out[k] = out[k].reshape(npoints, nlayers)
    for k in f2 + NATIVE_COORDS:
        if out[k].size != npoints:
            raise ValueError(f"{k} 长度 {out[k].size} != npoints {npoints}")
    if mf["export"]["ps_units"] == "log(Pa)":
        out["ps"] = np.exp(out["ps"])
    return out


def native_sha256(d, fields=SNAP_FIELDS) -> str:
    """原生快照原始字节摘要（CONS-DET 的裁判侧交叉核对：映射前是否已不同）。
    目录形态按 `fields` 的固定序读；npz 形态直接哈希文件字节（与读哪些场无关）。"""
    h = hashlib.sha256()
    if Path(d).is_dir():
        for k in fields:
            h.update(Path(d).joinpath(f"{k}.bin").read_bytes())
    else:
        h.update(Path(d).read_bytes())
    return h.hexdigest()


def ring_of(lat, lon) -> dict:
    """由 lat/lon 还原环结构，并校验契约：纬度升序唯一、每环经度均匀覆盖 [0,360)。"""
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    order = np.argsort(lat, kind="stable")
    lat_s, lon_s = lat[order], lon[order]
    uniq, inv = np.unique(np.round(lat_s, 9), return_inverse=True)
    if uniq.size < 3:
        raise ValueError(f"纬度数 {uniq.size} 太少，环结构不可判定")
    counts = np.bincount(inv).astype(np.int64)
    idx = [order[np.where(inv == j)[0]] for j in range(uniq.size)]
    step_ref = 360.0 / counts
    for j in range(uniq.size):
        lo = np.sort(lon_s[idx[j]])
        if not np.allclose(lo, np.arange(counts[j]) * step_ref[j], atol=1e-4, rtol=0):
            raise ValueError(f"环 {j}（lat={uniq[j]:.4f}）经度非均匀 360/{counts[j]}、起点 0")
    return {"lat": uniq.astype(np.float64), "mu": np.sin(np.deg2rad(uniq)),
            "idx": idx, "counts": counts, "nring": int(uniq.size),
            "max_ring": int(counts.max())}


# ---------------------------------------------------------------- 保守重映射核心

def mu_bands(mu_src: np.ndarray, mu_max: float) -> tuple:
    """原生环的 μ 单元：内部 = 相邻节点中点；两端 = ±mu_max ⇒ 与目标覆盖严格同域，不重不漏。"""
    mid = 0.5 * (mu_src[:-1] + mu_src[1:])
    lo = np.concatenate(([-mu_max], mid))
    hi = np.concatenate((mid, [mu_max]))
    return lo, hi


def target_grid(rings: dict, nlat: int | None = None, nlon: int | None = None) -> dict:
    """目标规则网格：等 Δμ 的 nlat 行 × 均匀 nlon 列，μ 覆盖钉在原生最大 ∣μ∣（不外推）。"""
    nlat = int(rings["nring"]) if nlat is None else int(nlat)
    nlon = 2 * int(rings["max_ring"]) if nlon is None else int(nlon)
    if nlat != rings["nring"]:
        raise ValueError(f"目标 nlat 必须等于原生环数 {rings['nring']}（纬度不许降采样）")
    if nlon < rings["max_ring"]:
        raise ValueError(f"目标 nlon 不得小于最大环点数 {rings['max_ring']}（经度不许降采样）")
    mu_max = float(np.max(np.abs(rings["mu"])))
    mu_edges = np.linspace(-mu_max, mu_max, nlat + 1)
    mu_ctr = 0.5 * (mu_edges[:-1] + mu_edges[1:])
    return {"nlat": nlat, "nlon": nlon, "mu_max": mu_max, "mu_edges": mu_edges,
            "dmu": float(mu_edges[1] - mu_edges[0]), "dlam": 360.0 / nlon,
            "lat": np.rad2deg(np.arcsin(mu_ctr)), "lon": np.arange(nlon) * (360.0 / nlon)}


def mu_coverage_defect(rings: dict, tgt: dict) -> float:
    """目标每行分到的原生 μ 长度之和与 Δμ 的最大相对偏差 ⇒ '极区无空洞/不重不漏'的硬断言。"""
    lo, hi = mu_bands(rings["mu"], tgt["mu_max"])
    mu_e = tgt["mu_edges"]
    cov = np.zeros(tgt["nlat"])
    for j in range(rings["nring"]):
        cov += np.clip(np.minimum(hi[j], mu_e[1:]) - np.maximum(lo[j], mu_e[:-1]), 0.0, None)
    return float(np.max(np.abs(cov - tgt["dmu"])) / tgt["dmu"])


_LAM_CACHE: dict = {}


def _lam_overlap(nlon_src: int, nlon_dst: int) -> np.ndarray:
    """(nlon_dst, nlon_src)：经度重叠宽度（度）。行和 = dlam_dst、列和 = 360/nlon_src ⇒ λ 向守恒。"""
    key = (int(nlon_src), int(nlon_dst))
    w = _LAM_CACHE.get(key)
    if w is not None:
        return w
    step_s, step_d = 360.0 / nlon_src, 360.0 / nlon_dst
    edges_s = np.arange(nlon_src + 1) * step_s
    lo_d = np.arange(nlon_dst) * step_d
    hi_d = lo_d + step_d
    first = np.floor(lo_d / step_s).astype(np.int64)
    last = np.floor((hi_d - 1e-12) / step_s).astype(np.int64)
    w = np.zeros((nlon_dst, nlon_src), dtype=np.float64)
    for k in range(nlon_dst):
        for s in range(int(first[k]), min(int(last[k]), nlon_src - 1) + 1):
            ov = min(hi_d[k], edges_s[s + 1]) - max(lo_d[k], edges_s[s])
            if ov > 0.0:
                w[k, s] = ov
    _LAM_CACHE[key] = w
    return w


def remap_field(f_rings, rings: dict, tgt: dict) -> np.ndarray:
    """约化网格 (npoints,nlev) / (npoints,) → 规则 (nlev,nlat,nlon) / (nlat,nlon)。
    一阶保守面积重叠，固定归约序：环升序 → 目标行升序（冻结项 5）。"""
    f = np.asarray(f_rings, dtype=np.float64)
    squeeze = f.ndim == 1
    f = f[:, None] if squeeze else f
    nlev = f.shape[1]
    nlat, nlon = tgt["nlat"], tgt["nlon"]
    lo, hi = mu_bands(rings["mu"], tgt["mu_max"])
    mu_e = tgt["mu_edges"]
    acc = np.zeros((nlat, nlon, nlev), dtype=np.float64)
    for j in range(rings["nring"]):
        tmp = _lam_overlap(int(rings["counts"][j]), nlon) @ f[rings["idx"][j], :]
        ov = np.clip(np.minimum(hi[j], mu_e[1:]) - np.maximum(lo[j], mu_e[:-1]), 0.0, None)
        rows = np.nonzero(ov > 0.0)[0]
        if rows.size:
            acc[rows] += tmp * ov[rows][:, None, None]
    acc /= tgt["dmu"] * tgt["dlam"]
    out = np.moveaxis(acc, 2, 0)
    return np.ascontiguousarray(out[0] if squeeze else out)


def bounds_excess(a: np.ndarray, b: np.ndarray) -> float:
    """保守平均越出原生极值的绝对幅度（正 = 越界）。容差按场量级取相对（见 _bound_tol）：
    ps≈1e5 Pa 上 float64 的累计舍入约 1e-14 相对 ⇒ 固定 1e-9 绝对容差会误报。"""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(max(np.nanmin(a) - np.nanmin(b), np.nanmax(b) - np.nanmax(a)))


def _bound_tol(a: np.ndarray) -> float:
    return 1e-12 * max(1.0, float(np.nanmax(np.abs(np.asarray(a, dtype=np.float64)))))


def area_mean_target(b: np.ndarray, tgt: dict) -> np.ndarray:
    """目标网格上用**与原生相同的 μ 面积**求全球平均（逐层），用于守恒自证。
    判据路径 `metrics.area_weights` 已改为 Δ(sinφ) 带精确权重（评审 2026-10-21 F12），
    与本自证口径一致到 4.7e-15；此处保留独立实现作为第二真源的对账。"""
    w = np.diff(tgt["mu_edges"])[:, None] * np.full((1, tgt["nlon"]), tgt["dlam"] / 360.0)
    if b.ndim == 3:
        return np.array([np.sum(b[k] * w) / np.sum(w) for k in range(b.shape[0])])
    return np.array([float(np.sum(b * w) / np.sum(w))])


# ---------------------------------------------------------------- 原生参考量（映射无法美化也无法掩盖）

def native_cfl_max(nat: dict, rings: dict, tgt: dict, dt_s: float) -> float:
    """原生网格逐点平流 CFL：dx=R cosφ Δλ_ring（约化网格 Δλ 随纬度变）、dy=R Δφ_band。
    不借目标网格的 Δλ ⇒ 用的是被测系统自己的分辨率。仅作登记量（甲为半隐式谱核，
    CONS-CFL 作失效门不良定义 ⇒ 冻结文档 §5 待裁 #1）。"""
    R = metrics.R_EARTH
    lo, hi = mu_bands(rings["mu"], tgt["mu_max"])
    dphi = np.arcsin(np.clip(hi, -1.0, 1.0)) - np.arcsin(np.clip(lo, -1.0, 1.0))
    dy = R * dphi
    dlam = np.deg2rad(360.0 / rings["counts"])
    dx = R * np.maximum(np.cos(np.deg2rad(rings["lat"])), 1e-6) * dlam
    worst = 0.0
    for j in range(rings["nring"]):                     # 固定序：环升序取 max
        idx = rings["idx"][j]
        cu = np.abs(nat["u"][idx]).max() / dx[j]
        cv = np.abs(nat["v"][idx]).max() / dy[j]
        worst = max(worst, dt_s * (cu + cv))
    return float(worst)


def native_area_mean(f_rings, rings: dict, tgt: dict) -> np.ndarray:
    """原生 μ 面积加权全球平均（∫f dμdλ / 覆盖面积），逐层 (nlev,) 或 (1,)。"""
    lo, hi = mu_bands(rings["mu"], tgt["mu_max"])
    band = hi - lo
    f = np.asarray(f_rings, dtype=np.float64)
    if f.ndim == 1:
        f = f[:, None]
    zm = np.stack([f[rings["idx"][j]].mean(axis=0) for j in range(rings["nring"])])
    return (band[:, None] * zm).sum(axis=0) / band.sum()


# ---------------------------------------------------------------- run 目录产出

def _snap_paths(nd: Path) -> list:
    ps = [p for p in sorted(nd.glob("snap_*")) if p.is_dir() or p.suffix == ".npz"]
    if not ps:
        raise ValueError(f"原生目录无 snap_*: {nd}")
    return ps


def convert_native_run(native_dir, out_dir, nlat=None, nlon=None) -> dict:
    """甲原生 run → loader 契约 run（供**未改动**的 suites.py 判定）+ 映射自证报告。
    干燥臂产物与 J1 冻结版逐位一致；湿臂多出 q/pr/evap/sw_toa/lw_out/sst，
    且 sigma_edges 直接取 manifest.grid.sigma_half（层界真源，不用 sigma 中点反推）。"""
    nd = Path(native_dir)
    mf = json.loads((nd / "manifest.json").read_text(encoding="utf-8"))
    _validate_manifest(mf)
    f3, f2 = _fields_for(mf)
    wet = mf["benchmark"] == "aquaplanet"
    snap_paths = _snap_paths(nd)
    probe = _read_native_snap(snap_paths[0], mf)
    rings = ring_of(probe["lat"], probe["lon"])
    nlayers = int(mf["grid"]["nlayers"])
    tgt = target_grid(rings, nlat, nlon)
    defect = mu_coverage_defect(rings, tgt)
    if defect > 1e-10:
        raise ValueError(f"μ 面积账本不闭合（最大相对缺口 {defect:.3g}）⇒ 原生环结构异常")

    od = Path(out_dir)
    od.mkdir(parents=True, exist_ok=True)
    sigma = np.asarray(mf["grid"]["sigma"], dtype=np.float64)
    if sigma.size != nlayers:
        raise ValueError(f"manifest.grid.sigma 长度 {sigma.size} != nlayers {nlayers}")
    if wet:
        edges = np.asarray(mf["grid"]["sigma_half"], dtype=np.float64)
    else:
        edges = np.concatenate(([0.0], 0.5 * (sigma[:-1] + sigma[1:]), [1.0]))
    np.savez_compressed(od / "coords.npz",
                        lat=np.ascontiguousarray(tgt["lat"]),
                        lon=np.ascontiguousarray(tgt["lon"]),
                        sigma=np.ascontiguousarray(sigma),
                        sigma_edges=np.ascontiguousarray(edges))

    native_fields = f3 + f2 + NATIVE_COORDS
    checks, cfl_native, sha_native = [], [], []
    for i, sp in enumerate(snap_paths):
        nat = _read_native_snap(sp, mf)
        sha_native.append(native_sha256(sp, fields=native_fields))
        cfl_native.append(native_cfl_max(nat, rings, tgt, float(mf["dt_atmo_s"])))
        snap = {k: remap_field(nat[k], rings, tgt) for k in f3 + f2}
        payload = {k: np.ascontiguousarray(snap[k]) for k in f3}
        for k in f2:
            payload[k] = np.ascontiguousarray(snap[k])
        np.savez_compressed(od / f"snap_{i + 1:05d}.npz", **payload)
        for k in f3 + f2:
            a, b = nat[k], snap[k]
            exc, tol = bounds_excess(a, b), _bound_tol(a)
            if exc > tol:
                checks.append({"snap": i + 1, "field": k, "why": "映射越界（单调性破坏）",
                               "excess": exc, "tol": tol})
            nm = native_area_mean(a, rings, tgt)
            tm = area_mean_target(b, tgt)
            tol = 1e-11 * max(1.0, float(np.nanmax(np.abs(nm))))
            d = float(np.max(np.abs(tm - nm)))
            if d > tol:
                checks.append({"snap": i + 1, "field": k,
                               "why": f"面积均值漂移 {d:.3g} native={nm[0]:.9g} target={tm[0]:.9g}"})
            if not np.all(np.isfinite(b)):
                checks.append({"snap": i + 1, "field": k, "why": "映射后非有限"})

    meta = {
        "benchmark": mf["benchmark"], "dt_atmo_s": float(mf["dt_atmo_s"]),
        "days": float(mf["days"]), "snap_days": list(mf["snap_days"]),
        "obliquity_deg": float(mf["obliquity_deg"]), "seed": mf["seed"],
        "config_hash": mf["config_hash"],
        "forcing": {k: float(v) for k, v in mf["forcing"].items()},
        "declared_deviations": mf.get("declared_deviations", {}),
        "initial_state": f"jia:{mf['ic']['name']}@{mf['ic']['sha256'][:12]}",
        "cfl_reported": list(mf.get("cfl_reported", [])),
        "slab_heat_capacity": mf.get("slab_heat_capacity"),
        "radiation": mf.get("radiation"),
        "mapping": {
            "method": "conservative mu/lambda band-overlap (1st-order FV)",
            "frozen_doc": "benchmarks/atmo/gate/J1_mapping_freeze.md",
            "protocol": dict(PROTOCOL), "ic": dict(mf["ic"]), "env": dict(mf["env"]),
            "native": {"npoints": int(mf["grid"]["npoints"]), "nlayers": nlayers,
                       "nring": rings["nring"], "max_ring": rings["max_ring"],
                       "lat_reach_deg": float(np.max(np.abs(rings["lat"]))),
                       "truncation": mf["grid"]["truncation"],
                       "dtype": str(mf["export"]["dtype"]), "ps_units": mf["export"]["ps_units"]},
            "target": {"nlat": tgt["nlat"], "nlon": tgt["nlon"], "mu_cover": tgt["mu_max"],
                       "mu_cover_deg": float(np.rad2deg(math.asin(tgt["mu_max"]))),
                       "excluded_cap_area_frac": 1.0 - tgt["mu_max"]},
            "upcast": "float64 at read; ps exp after upcast when log(Pa)",
            "mu_coverage_defect": defect,
            "cfl_native_max": float(max(cfl_native)),
            "native_sha256_head": sha_native[:3], "native_sha_n": len(sha_native),
            "self_check_bad": len(checks),
        },
    }
    if wet:
        meta["moist_scheme"] = mf["moist_scheme"]
        meta["mapping"]["sigma_edges_source"] = "manifest.grid.sigma_half"
        for k in FLUX_META_KEYS:
            if k in mf:
                meta[k] = mf[k]
    (od / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"out_dir": str(od), "benchmark": mf["benchmark"],
            "fields_3d": list(f3), "fields_2d": list(f2),
            "rings": {"nring": rings["nring"], "counts_head": rings["counts"][:6].tolist(),
                      "counts_mid": rings["counts"][rings["nring"] // 2 - 1:rings["nring"] // 2 + 1].tolist(),
                      "counts_tail": rings["counts"][-6:].tolist(),
                      "max_ring": rings["max_ring"],
                      "lat_reach_deg": float(np.max(np.abs(rings["lat"])))},
            "target": {"nlat": tgt["nlat"], "nlon": tgt["nlon"],
                       "mu_cover_deg": meta["mapping"]["target"]["mu_cover_deg"],
                       "sigma_plev_hpa": (sigma * 1000.0).tolist()},
            "n_snaps": len(snap_paths), "cfl_native_max": float(max(cfl_native)),
            "mu_coverage_defect": defect,
            "self_check": checks[:20], "self_check_bad": len(checks)}


# ---------------------------------------------------------------- IC 材料化（协议在裁判侧）

def make_native_ic(lat, lon, sigma, name="iso_w1", seed=0):
    """AV 冻结初值 → 甲原生网格 (npoints,nlev)；ps 以 **ln(Pa)** 返回（= 甲磁盘约定）。
    Teq 剖面真源 = `run_hs.hs_teq`（与乙 prim.rs 同式）；甲的 Teq 用 ln(p/p0)（p=σ·ps）、
    乙用 ln σ，ps≈ps0 时一致，差登记为已声明偏差（冻结文档 §6.3）。"""
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    nlayers = sigma.size
    hp = suites.HS94_PARAMS
    if name == "teq_noise":
        teq = run_hs.hs_teq(np.sin(np.deg2rad(lat)), sigma, PROTOCOL["tmax_eq_K"],
                            hp["delta_Ty"], hp["delta_theta_z"], hp["R"] / hp["cp"])[:, :, 0]
        T = np.ascontiguousarray(teq.T)                       # (nlev,npts) → (npts,nlev)
        T = T + IC_PROTOCOL["teq_noise"]["amp_K"] * \
            np.random.default_rng(seed).standard_normal(T.shape)
    else:
        spec = IC_PROTOCOL["iso_w1"]
        T = np.full((lat.size, nlayers), spec["T0_K"], dtype=np.float64)
        T += spec["amp_K"] * np.sin(np.deg2rad(spec["wave"] * lon))[:, None]
    return {"u": np.zeros((lat.size, nlayers)),
            "v": np.zeros((lat.size, nlayers)),
            "T": np.ascontiguousarray(T),
            "ps": np.full(lat.size, math.log(PROTOCOL["ps0_Pa"]))}


def materialize_ic(native_dir, out_path=None, name="iso_w1", seed=0) -> dict:
    """读 A1 的网格几何（首个原生快照的 lat/lon + manifest 的 sigma）→ 机器可读 IC。
    A1 施加该 IC（float32，模型可精确表示）并把 sha256 回填 manifest.ic。"""
    nd = Path(native_dir)
    mf = json.loads((nd / "manifest.json").read_text(encoding="utf-8"))
    _validate_manifest(mf)
    if mf["ic"]["name"] != name:
        raise ValueError(f"manifest.ic.name={mf['ic']['name']!r} 与所求 IC {name!r} 不符")
    probe = _read_native_snap(_snap_paths(nd)[0], mf)
    ic = make_native_ic(probe["lat"], probe["lon"], np.asarray(mf["grid"]["sigma"]),
                        name=name, seed=seed)
    ic32 = {k: np.ascontiguousarray(v, dtype=np.float32) for k, v in ic.items()}
    op = Path(out_path) if out_path else nd / f"ic_{name}.npz"
    np.savez(op, **ic32)
    sha = hashlib.sha256(b"".join(ic32[k].tobytes() for k in ("u", "v", "T", "ps"))).hexdigest()
    return {"ic_path": str(op), "name": name, "seed": seed, "sha256": sha,
            "dtype": "float32", "ps_units": "log(Pa)",
            "shape": {"u": list(ic32["u"].shape), "ps": list(ic32["ps"].shape)},
            "T_range_K": [float(ic32["T"].min()), float(ic32["T"].max())]}


# ---------------------------------------------------------------- CLI

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="av-jia-remap")
    ap.add_argument("--native", required=True, help="A1 甲原生 run 目录（含 manifest.json）")
    ap.add_argument("--out", default=None, help="loader 契约 run 目录（判据输入）")
    ap.add_argument("--nlat", type=int, default=None)
    ap.add_argument("--nlon", type=int, default=None)
    ap.add_argument("--rep", default=None, help="映射自证报告 JSON 落点")
    ap.add_argument("--ic", default=None, choices=sorted(IC_PROTOCOL),
                    help="只做 IC 材料化（不转换结果场）")
    ap.add_argument("--ic-out", default=None)
    ap.add_argument("--ic-seed", type=int, default=0)
    a = ap.parse_args(argv)
    try:
        if a.ic:
            rep = materialize_ic(a.native, a.ic_out, a.ic, a.ic_seed)
        else:
            if not a.out:
                raise ValueError("转换结果场需 --out")
            rep = convert_native_run(a.native, a.out, a.nlat, a.nlon)
    except (ValueError, KeyError, FileNotFoundError, json.JSONDecodeError) as e:
        print(f"INPUT_CONTRACT_ERROR: {e}")
        return 1
    print(json.dumps(rep, ensure_ascii=False, indent=1))
    if a.rep:
        Path(a.rep).parent.mkdir(parents=True, exist_ok=True)
        Path(a.rep).write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    bad = rep.get("self_check_bad", 0)
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
