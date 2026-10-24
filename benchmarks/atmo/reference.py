"""AV 对照裁判：只读加载参考气候态（ExoPlaSim/观测，npz）并做统计对比。

红线：参考数据绝不作为模型输入（runner 不把本模块的加载结果喂给任何模型调用）。
参考文件约定（不进主仓，二进制放本地 benchmarks/atmo/out/reference/）：
  npz 含 lat (nlat,), plev (nlev,) [hPa，降序或升序均可], u_zm (nlev,nlat),
  t_zm (nlev,nlat)，可选 pr_zm (nlat,) [mm/d]。
"""
import numpy as np

REQUIRED_KEYS = ("lat", "plev", "u_zm", "t_zm")


def load_reference(path):
    with np.load(path, allow_pickle=False) as d:
        missing = [k for k in REQUIRED_KEYS if k not in d]
        if missing:
            raise ValueError(f"参考气候态缺字段: {missing}")
        return {k: d[k].astype(np.float64) for k in d.files}


def _interp_lat(a, src_lat, dst_lat):
    out = np.empty((a.shape[0], dst_lat.size))
    for i in range(a.shape[0]):
        order = np.argsort(src_lat)
        out[i] = np.interp(dst_lat, src_lat[order], a[i][order])
    return out


def interp_to_ref(model_zm, model_lat, model_plev, ref_lat, ref_plev):
    """把模型纬向平均场 (nlev, nlat) 插值到参考网格：先纬度线性插值，再 log(p) 垂直插值。"""
    model_lat = np.asarray(model_lat, dtype=np.float64)
    model_plev = np.asarray(model_plev, dtype=np.float64)
    ref_plev = np.asarray(ref_plev, dtype=np.float64)
    mo = np.argsort(np.log(model_plev))
    lp_m = np.log(model_plev)[mo]
    lp_r = np.sort(np.log(ref_plev))
    a = np.asarray(model_zm)[mo]
    lat_order = np.argsort(model_lat)
    lat_s = model_lat[lat_order]
    a_lat = np.empty((a.shape[0], len(ref_lat)))
    for i in range(a.shape[0]):
        a_lat[i] = np.interp(ref_lat, lat_s, a[i][lat_order])
    out = np.empty((lp_r.size, len(ref_lat)))
    for j in range(lp_r.size):
        out[j] = np.interp(lp_r[j], lp_m, a_lat[:, j])
    return out


def zonal_compare(model_zm, ref_zm):
    d = np.asarray(model_zm, dtype=np.float64) - np.asarray(ref_zm, dtype=np.float64)
    if d.size == 0:
        raise ValueError("对比集合为空")
    a = np.asarray(model_zm, dtype=np.float64).ravel()
    b = np.asarray(ref_zm, dtype=np.float64).ravel()
    am, bm = a.mean(), b.mean()
    denom = np.sqrt(((a - am) ** 2).sum() * ((b - bm) ** 2).sum())
    corr = float(((a - am) * (b - bm)).sum() / denom) if denom > 0 else float("nan")
    return {"bias": float(d.mean()), "rmse": float(np.sqrt((d ** 2).mean())),
            "pattern_corr": corr}
