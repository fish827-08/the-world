"""J1 映射冻结件自证 + J2 湿臂扩展：保守性、单调性、极区无空洞、逐位可复现、契约拒绝。

TIER = "mainline"（档②）。用解析构造的合成 octahedral 约化网格（每环点数随纬度变、无极点）
⇒ 不依赖 A1/甲的任何读数，也不依赖生产代码（判据/阈值一律不动，只测映射与契约层）。
J2 扩展的约束：干燥臂产物逐位不变（sigma_half 一律不读）；湿臂字段集/σ 层界/可变 nlev 按契约。
"""
import json
import math

import numpy as np
import pytest

from benchmarks.atmo import jia_remap, loader, metrics, suites

# 冻结文档 §5 的垂直窗口清单（真源在 suites.evaluate 内联）。改窗口必须同时改本表与冻结文档。
CRITERIA_P_WINDOWS = [
    ("HS-JET-U/LAT (jet_metrics 默认 200-400)", 200.0, 400.0),
    ("HS-SFC-U 与 HS-T-*-SFC (sigma>=0.9)", 900.0, 1e9),
    ("HS-T-TROP-UP", 200.0, 250.0),
    ("HS-EKE-ML/RATIO", 300.0, 700.0),
]
EMPTY_WINDOWS = {"HS-T-TROP-UP"}                       # 甲 8 层阶梯上的空窗 ⇒ 待裁项
JIA_SIGMA = [0.0625, 0.1875, 0.3125, 0.4375, 0.5625, 0.6875, 0.8125, 0.9375]  # 甲 nlayers=8 默认
LAT_R = [-70.0, -45.0, -20.0, -5.0, 5.0, 20.0, 45.0, 70.0]
COUNTS = [4, 8, 12, 16, 16, 12, 8, 4]

# 40 d / 4 d / 1 d 时间尺度 → 1/s（甲侧 HeldSuarez/LinearDrag 的登记换算，A1 交实测值）
HS_FORCING_DEMO = {
    "omega": 7.292e-5, "g": 9.81, "R": 287.0, "cp": 1004.0, "radius": 6.371e6,
    "p0": 100000.0, "delta_Ty": 60.0, "delta_theta_z": 10.0, "sigma_b": 0.7,
    "ka": 1.0 / (40.0 * 86400.0), "ks": 1.0 / (4.0 * 86400.0), "kf": 1.0 / 86400.0,
    "tmax_eq_K": 315.0, "tmin_eq_K": 200.0, "ps0_Pa": 100000.0,
}


def octa_grid():
    """8 环 × 点数 COUNTS（和 80）；纬度升序、无极点、每环经度均匀 360/n。"""
    lat = np.repeat(np.array(LAT_R), COUNTS).astype(np.float64)
    lon = np.concatenate([np.arange(n) * (360.0 / n) for n in COUNTS])
    return lat, lon


def manifest(nlevels=3, **over):
    lat, _ = octa_grid()
    mf = {
        "benchmark": "held_suarez", "days": 10.0, "dt_atmo_s": 900.0,
        "snap_days": [1.0, 2.0, 3.0], "obliquity_deg": 0.0, "seed": 7,
        "config_hash": "deadbeef", "cfl_reported": [0.4, 0.5, 0.6],
        "grid": {"npoints": int(lat.size), "nlayers": nlevels, "sigma": JIA_SIGMA[:nlevels],
                 "truncation": 5},
        "export": {"dtype": "float64", "ps_units": "log(Pa)",
                   "field_shape": "npoints,nlayers", "byte_order": "little"},
        "ic": {"name": "iso_w1", "sha256": "0" * 64},
        "env": {"julia": "1.13.2", "speedyweather": "0.23.0", "nthreads": 4},
        "forcing": dict(HS_FORCING_DEMO),
    }
    mf.update(over)
    return mf


def write_native_run(nd, seed=0, n_snap=3, nlevels=3):
    """合成甲原生 run 目录（npz 快照风格；u/v/T 任意有限值，ps 存 ln(Pa)）。"""
    nd.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    lat, lon = octa_grid()
    for i in range(n_snap):
        npts = lat.size
        np.savez(nd / f"snap_{i + 1:05d}.npz",
                 u=rng.standard_normal((npts, nlevels)),
                 v=rng.standard_normal((npts, nlevels)),
                 T=300.0 + 10.0 * rng.standard_normal((npts, nlevels)),
                 ps=np.log(20000.0 + 50.0 * rng.standard_normal(npts)),
                 lat=lat, lon=lon)
    mf = manifest(nlevels=nlevels)
    (nd / "manifest.json").write_text(json.dumps(mf), encoding="utf-8")
    return mf


# ---------------------------------------------------------------- 保守 / 单调 / 无空洞

def test_global_area_integral_exact_to_rounding():
    rng = np.random.default_rng(1)
    lat, lon = octa_grid()
    rings = jia_remap.ring_of(lat, lon)
    tgt = jia_remap.target_grid(rings)
    f = 300.0 + 10.0 * rng.standard_normal((lat.size, 5))
    b = jia_remap.remap_field(f, rings, tgt)
    assert b.shape == (5, rings["nring"], tgt["nlon"])
    np.testing.assert_allclose(jia_remap.area_mean_target(b, tgt),
                               jia_remap.native_area_mean(f, rings, tgt),
                               rtol=1e-11, atol=1e-9)


def test_constant_field_reproduced_exactly():
    """常数场映射后仍为常数 ⇒ 归一化（dmu·dlam）无误，是"不重不漏"的最强单例检验。"""
    lat, lon = octa_grid()
    rings = jia_remap.ring_of(lat, lon)
    tgt = jia_remap.target_grid(rings)
    for c in (-17.5, 101325.0):
        for f in (np.full(lat.size, c), np.full((lat.size, 4), c)):
            b = jia_remap.remap_field(f, rings, tgt)
            assert np.all(np.isfinite(b))
            assert np.max(np.abs(b - c)) < 1e-9


def test_no_new_extrema():
    """凸组合 ⇒ 映射不造极值（不能自己造出急流/EKE 极大）。"""
    rng = np.random.default_rng(2)
    lat, _ = octa_grid()
    rings = jia_remap.ring_of(lat, octa_grid()[1])
    tgt = jia_remap.target_grid(rings)
    f = 300.0 + 10.0 * rng.standard_normal((lat.size, 4))
    b = jia_remap.remap_field(f, rings, tgt)
    assert b.min() >= f.min() - 1e-12 and b.max() <= f.max() + 1e-12


def test_self_check_bound_tolerance_is_relative():
    """回归（2026-10-21 签门）：ps≈1e5 Pa 的舍入级越界（1.5e-9 绝对 = 1.5e-14 相对）
    曾因自检用固定 1e-9 绝对容差误报。容差须随场量级走，真越界（1e-3 相对）仍须被抓住。"""
    a = 1.0e5 * (1.0 + 1e-3 * np.linspace(-1.0, 1.0, 48))
    rounding = a * (1.0 + 2e-15)
    exc, tol = jia_remap.bounds_excess(a, rounding), jia_remap._bound_tol(a)
    assert 0.0 < exc < tol < 1e-6, (exc, tol)
    assert jia_remap.bounds_excess(a, a * 1.001) > jia_remap._bound_tol(a)


def test_ring_indicator_reaches_every_row():
    """环指标场（每环赋 j+1）：目标每行面积均值必须严格递增 ⇒ 任何空洞/漏带都会露形。"""
    lat, _ = octa_grid()
    rings = jia_remap.ring_of(lat, octa_grid()[1])
    tgt = jia_remap.target_grid(rings)
    assert jia_remap.mu_coverage_defect(rings, tgt) < 1e-12
    f = np.repeat(np.arange(1, rings["nring"] + 1, dtype=np.float64), COUNTS)
    b = jia_remap.remap_field(f, rings, tgt)                    # (nlat, nlon)
    row_mean = b.mean(axis=1)
    assert np.all(np.diff(row_mean) > 0), row_mean
    assert 1.0 <= row_mean[0] and row_mean[-1] <= rings["nring"]


def test_upcast_at_read_and_logps_exp():
    mf = manifest()
    assert jia_remap._dtype(mf) == np.dtype("<f8")
    lat, _ = octa_grid()
    nat = {"u": np.zeros((lat.size, 3)), "v": np.zeros((lat.size, 3)),
           "T": np.zeros((lat.size, 3)), "ps": np.full(lat.size, math.log(20000.0)),
           "lat": lat, "lon": octa_grid()[1]}
    assert abs(math.exp(nat["ps"][0]) - 20000.0) < 1e-6


def test_read_native_snap_upcasts_and_expires_ps(tmp_path):
    mf = write_native_run(tmp_path / "native", seed=3)
    sp = jia_remap._snap_paths(tmp_path / "native")[0]
    nat = jia_remap._read_native_snap(sp, mf)
    assert nat["T"].dtype == np.float64 and nat["T"].shape == (80, 3)
    assert 19000.0 < nat["ps"][0] < 21000.0                     # ln(Pa) 已 exp 成 Pa


# ---------------------------------------------------------------- 目标网格规则

def test_target_grid_rules():
    lat, lon = octa_grid()
    rings = jia_remap.ring_of(lat, lon)
    tgt = jia_remap.target_grid(rings)
    assert tgt["nlat"] == rings["nring"] == 8                   # 纬度不降采样
    assert tgt["nlon"] == 2 * max(COUNTS) == 32                 # 经度只升不降
    assert tgt["mu_max"] == pytest.approx(math.sin(math.radians(70.0)))
    assert np.all(np.diff(tgt["lat"]) > 0)
    with pytest.raises(ValueError, match="纬度不许降采样"):
        jia_remap.target_grid(rings, nlat=4)
    with pytest.raises(ValueError, match="经度不许降采样"):
        jia_remap.target_grid(rings, nlon=8)


def test_native_cfl_uses_this_rings_dx():
    """约化网格的 dx = R cosφ·(360/本环点数)：极环 4 点/纬 ⇒ 与中低纬完全不同的 dx。
    只在 ±70° 两环（4 点）放 u=40 ⇒ 最大 CFL 必等于该环的 dt·u/dx。"""
    lat, lon = octa_grid()
    rings = jia_remap.ring_of(lat, lon)
    tgt = jia_remap.target_grid(rings)
    u = np.zeros((lat.size, 2))
    for j, la in enumerate(rings["lat"]):
        if abs(la) > 60.0:
            u[rings["idx"][j]] = 40.0
    nat = {"u": u, "v": np.zeros((lat.size, 2))}
    dx_polar = metrics.R_EARTH * math.cos(math.radians(70.0)) * math.radians(360.0 / 4)
    assert jia_remap.native_cfl_max(nat, rings, tgt, 900.0) == pytest.approx(
        900.0 * 40.0 / dx_polar, rel=1e-12)


# ---------------------------------------------------------------- 契约拒绝（入口挡，不改判据）

def test_manifest_rejects_missing_env_key():
    mf = manifest()
    del mf["env"]["nthreads"]
    with pytest.raises(ValueError, match="manifest.env 缺键"):
        jia_remap._validate_manifest(mf)


def test_manifest_rejects_unfrozen_teq_constant():
    """乙的 t0=288 不得沿用：Teq 上限这个未冻结自由度已由冻结件 §6.1 钉为 315 K。"""
    mf = manifest()
    mf["forcing"]["tmax_eq_K"] = 288.0
    with pytest.raises(ValueError, match="tmax_eq_K"):
        jia_remap._validate_manifest(mf)


def test_manifest_rejects_missing_drag_rate():
    """kf（LinearDrag 地表拖曳）漏登记 ⇒ 入口即挡；换算错值另由 HS-FORCING 1% 门判。"""
    mf = manifest()
    del mf["forcing"]["kf"]
    with pytest.raises(ValueError, match="manifest.forcing 缺键"):
        jia_remap._validate_manifest(mf)


def test_manifest_rejects_wrong_ps_units():
    mf = manifest()
    mf["export"]["ps_units"] = "hPa"
    with pytest.raises(ValueError, match="ps_units"):
        jia_remap._validate_manifest(mf)


def test_ring_contract_rejects_nonuniform_lon():
    lat, lon = octa_grid()
    lon = lon.copy()
    lon[0] += 3.0
    with pytest.raises(ValueError, match="经度非均匀"):
        jia_remap.ring_of(lat, lon)


def test_convert_rejects_npoints_mismatch(tmp_path):
    nd = tmp_path / "native"
    mf = write_native_run(nd, seed=6)
    mf["grid"]["npoints"] = 40                                  # 声明半格却交满格
    (nd / "manifest.json").write_text(json.dumps(mf), encoding="utf-8")
    with pytest.raises(ValueError, match="npoints"):
        jia_remap.convert_native_run(nd, tmp_path / "out")


# ---------------------------------------------------------------- run 目录产出 + loader 契约

def test_convert_produces_loader_valid_run(tmp_path):
    nd = tmp_path / "native"
    write_native_run(nd, seed=4)
    rep = jia_remap.convert_native_run(nd, tmp_path / "avrun")
    assert rep["self_check_bad"] == 0 and rep["mu_coverage_defect"] < 1e-12
    run = loader.load_run(tmp_path / "avrun")
    assert run["wet"] is False
    s = run["snaps"][0]
    assert set(s) == {"u", "v", "T", "ps"}
    assert s["u"].shape == (3, 8, 32) and s["ps"].shape == (8, 32)
    assert s["u"].flags["C_CONTIGUOUS"]
    assert np.all(np.isfinite(s["T"]))
    meta = json.loads((tmp_path / "avrun" / "meta.json").read_text(encoding="utf-8"))
    assert meta["mapping"]["target"]["mu_cover_deg"] == pytest.approx(70.0)
    assert meta["mapping"]["protocol"] == jia_remap.PROTOCOL
    assert meta["dt_atmo_s"] == 900.0 and meta["seed"] == 7
    assert meta["initial_state"].startswith("jia:iso_w1@")


def test_mapping_is_bitwise_reproducible(tmp_path):
    nd = tmp_path / "native"
    write_native_run(nd, seed=5)
    for name in ("a", "b"):
        jia_remap.convert_native_run(nd, tmp_path / name)
    with np.load(tmp_path / "a/snap_00002.npz") as xa, np.load(tmp_path / "b/snap_00002.npz") as xb:
        for k in xa.files:
            assert xa[k].tobytes() == xb[k].tobytes()           # 映射不污染 CONS-DET
    assert (tmp_path / "a/meta.json").read_bytes() == (tmp_path / "b/meta.json").read_bytes()


def test_convert_self_check_flags_nonfinite(tmp_path):
    """原生场含 NaN ⇒ 映射自证必须报"映射后非有限"，而不是悄悄带病产出 run。"""
    nd = tmp_path / "native"
    mf = write_native_run(nd, seed=11)
    p = nd / "snap_00001.npz"
    with np.load(p) as z:
        bad = {k: z[k] for k in z.files}
    bad["T"][0, 0] = np.nan
    np.savez(p, **bad)
    rep = jia_remap.convert_native_run(nd, tmp_path / "out")
    assert rep["self_check_bad"] >= 1
    assert any("非有限" in c["why"] for c in rep["self_check"])


def test_cli_convert_ok(tmp_path, capsys):
    nd = tmp_path / "native"
    write_native_run(nd, seed=8)
    rc = jia_remap.main(["--native", str(nd), "--out", str(tmp_path / "out"),
                         "--rep", str(tmp_path / "rep.json")])
    assert rc == 0 and (tmp_path / "rep.json").exists()
    assert "self_check_bad" in capsys.readouterr().out


def test_cli_contract_error_exit(tmp_path, capsys):
    nd = tmp_path / "native"
    mf = write_native_run(nd, seed=7)
    mf["export"]["byte_order"] = "big"
    (nd / "manifest.json").write_text(json.dumps(mf), encoding="utf-8")
    assert jia_remap.main(["--native", str(nd), "--out", str(tmp_path / "out")]) == 1
    assert "INPUT_CONTRACT_ERROR" in capsys.readouterr().out


# ---------------------------------------------------------------- J2 湿臂扩展（2026-10-22 提案）

J2_SIGMA_HALF = [0.0, 0.05, 0.14, 0.26, 0.4, 0.55, 0.7, 0.85, 1.0]
J2_SIGMA = [0.5 * (a + b) for a, b in zip(J2_SIGMA_HALF[:-1], J2_SIGMA_HALF[1:])]


def wet_manifest(nlevels=8, sigma_half=None, **over):
    lat, _ = octa_grid()
    half = np.asarray(J2_SIGMA_HALF if sigma_half is None else sigma_half, dtype=np.float64)
    sigma = 0.5 * (half[:-1] + half[1:])
    mf = {
        "benchmark": "aquaplanet", "days": 120.0, "dt_atmo_s": 600.0,
        "snap_days": [30.0, 32.0], "obliquity_deg": 0.0, "seed": 11,
        "config_hash": "cafebabe", "cfl_reported": [0.1, 0.1],
        "grid": {"npoints": int(lat.size), "nlayers": nlevels,
                 "sigma": sigma.tolist(), "sigma_half": half.tolist(), "truncation": 5},
        "export": {"dtype": "float32", "ps_units": "log(Pa)",
                   "field_shape": "npoints,nlayers", "byte_order": "little",
                   "snap_fields_3d": ["u", "v", "T", "q"],
                   "snap_fields_2d": ["ps", "sst", "pr", "evap", "sw_toa", "lw_out"],
                   "pr_units": "kg m-2 s-1", "evap_units": "kg m-2 s-1"},
        "ic": {"name": "j2aqua_test", "sha256": "a" * 64},
        "env": {"julia": "1.13.2", "speedyweather": "0.23.0", "nthreads": 4},
        "slab_heat_capacity": 2.092e8,
        "radiation": "test radiation (one-band)",
        "moist_scheme": {"convection": "test"},
        "forcing": {},
    }
    mf.update(over)
    return mf


def write_wet_native_run(nd, seed=0, n_snap=2, nlevels=8, sigma_half=None, pr_value=None):
    """合成甲湿臂原生 run（npz 风格；字段/布局与 A2 manifest 一致，f32 落盘）。"""
    nd.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    lat, lon = octa_grid()
    npts = lat.size
    for i in range(n_snap):
        d = {
            "u": rng.standard_normal((npts, nlevels)),
            "v": rng.standard_normal((npts, nlevels)),
            "T": 300.0 + 10.0 * rng.standard_normal((npts, nlevels)),
            "q": 0.01 * rng.random((npts, nlevels)),
            "ps": np.log(1.0e5 + 100.0 * rng.standard_normal(npts)),
            "pr": 1e-5 * (1.0 + 0.2 * rng.random(npts)) if pr_value is None else pr_value,
            "evap": 0.9e-5 * np.ones(npts),
            "sw_toa": 240.0 + rng.standard_normal(npts),
            "lw_out": 238.0 + rng.standard_normal(npts),
            "sst": 300.0 + rng.standard_normal(npts),
            "lat": lat, "lon": lon,
        }
        np.savez(nd / f"snap_{i + 1:05d}.npz",
                 **{k: np.asarray(v, dtype=np.float32) for k, v in d.items()})
    mf = wet_manifest(nlevels=nlevels, sigma_half=sigma_half)
    (nd / "manifest.json").write_text(json.dumps(mf), encoding="utf-8")
    return mf


def test_fields_for_routes_by_benchmark():
    assert jia_remap._fields_for({"benchmark": "held_suarez"}) == \
        (jia_remap.SNAP_FIELDS_3D, jia_remap.SNAP_FIELDS_2D)
    assert jia_remap._fields_for({"benchmark": "aquaplanet"}) == \
        (jia_remap.WET_SNAP_FIELDS_3D, jia_remap.WET_SNAP_FIELDS_2D)
    with pytest.raises(ValueError, match="benchmark"):
        jia_remap._fields_for({"benchmark": "mars"})


def test_wet_convert_loader_valid_and_sigma_half_is_source(tmp_path):
    nd = tmp_path / "native"
    write_wet_native_run(nd, seed=21)
    rep = jia_remap.convert_native_run(nd, tmp_path / "avrun")
    assert rep["benchmark"] == "aquaplanet" and rep["self_check_bad"] == 0
    assert rep["fields_3d"] == ["u", "v", "T", "q"]
    run = loader.load_run(tmp_path / "avrun")
    assert run["wet"] is True
    assert set(run["snaps"][0]) == set(loader.WET_SNAP_FIELDS)
    assert run["snaps"][0]["q"].shape == (8, 8, 32)
    assert run["snaps"][0]["pr"].shape == (8, 32)
    np.testing.assert_array_equal(run["coords"]["sigma_edges"], np.asarray(J2_SIGMA_HALF))
    mid = np.concatenate(([0.0], 0.5 * (np.asarray(J2_SIGMA[:-1]) + np.asarray(J2_SIGMA[1:])), [1.0]))
    assert not np.allclose(run["coords"]["sigma_edges"], mid)   # 中点反推 ≠ 真层界
    assert run["meta"]["moist_scheme"] == {"convection": "test"}
    assert run["meta"]["mapping"]["sigma_edges_source"] == "manifest.grid.sigma_half"


def test_wet_variable_nlev(tmp_path):
    half = [0.0, 0.1, 0.3, 0.6, 0.85, 1.0]
    nd = tmp_path / "native"
    write_wet_native_run(nd, seed=22, nlevels=5, sigma_half=half)
    rep = jia_remap.convert_native_run(nd, tmp_path / "out")
    run = loader.load_run(tmp_path / "out")
    assert run["snaps"][0]["T"].shape == (5, 8, 32)
    assert run["coords"]["sigma"].size == 5
    np.testing.assert_array_equal(run["coords"]["sigma_edges"], np.asarray(half))
    assert rep["self_check_bad"] == 0


def test_wet_all_fields_conserved(tmp_path):
    nd = tmp_path / "native"
    mf = write_wet_native_run(nd, seed=23)
    jia_remap.convert_native_run(nd, tmp_path / "out")
    run = loader.load_run(tmp_path / "out")
    nat = jia_remap._read_native_snap(jia_remap._snap_paths(nd)[0], mf)
    rings = jia_remap.ring_of(nat["lat"], nat["lon"])
    tgt = jia_remap.target_grid(rings)
    for k in jia_remap.WET_SNAP_FIELDS_3D + jia_remap.WET_SNAP_FIELDS_2D:
        nm = jia_remap.native_area_mean(nat[k], rings, tgt)
        tm = jia_remap.area_mean_target(run["snaps"][0][k], tgt)
        np.testing.assert_allclose(tm, nm, rtol=1e-11,
                                   atol=1e-11 * max(1.0, float(np.max(np.abs(nm)))))


def test_wet_flux_sign_passthrough(tmp_path):
    """pr/evap 符号原样透传：不做绝对值、不擅自改号（符号语义留给判据）。"""
    nd = tmp_path / "native"
    write_wet_native_run(nd, seed=24, pr_value=-1.0e-5 * np.ones(80))
    jia_remap.convert_native_run(nd, tmp_path / "out")
    pr = loader.load_run(tmp_path / "out")["snaps"][0]["pr"]
    assert pr.max() <= 0.0
    np.testing.assert_allclose(pr, -1.0e-5, rtol=1e-5, atol=1e-12)


@pytest.mark.parametrize("mutate, match", [
    (lambda m: m["grid"].pop("sigma_half"), "sigma_half"),
    (lambda m: m.pop("moist_scheme"), "moist_scheme"),
    (lambda m: m.pop("radiation"), "radiation"),
    (lambda m: m.pop("slab_heat_capacity"), "slab_heat_capacity"),
    (lambda m: m["export"].__setitem__("pr_units", "mm/day"), "pr_units"),
    (lambda m: m["export"].__setitem__("evap_units", "W m-2"), "evap_units"),
    (lambda m: m["grid"]["sigma"].__setitem__(0, 0.5), "sigma_half"),
    (lambda m: m["ic"].__setitem__("sha256", ""), "ic.sha256"),
    (lambda m: m["grid"].__setitem__("nlayers", 0), "nlayers"),
    (lambda m: m.__setitem__("benchmark", "mars"), "benchmark"),
], ids=["sigma_half", "moist_scheme", "radiation", "slab", "pr_units", "evap_units",
        "sigma_mismatch", "ic_sha_empty", "zero_layers", "unknown_benchmark"])
def test_wet_manifest_rejections(mutate, match):
    mf = wet_manifest()
    mutate(mf)
    with pytest.raises(ValueError, match=match):
        jia_remap._validate_manifest(mf)


def test_wet_rejects_missing_field(tmp_path):
    nd = tmp_path / "native"
    write_wet_native_run(nd, seed=25)
    p = nd / "snap_00001.npz"
    with np.load(p) as z:
        d = {k: z[k] for k in z.files if k != "q"}
    np.savez(p, **d)
    with pytest.raises(ValueError, match="缺场"):
        jia_remap.convert_native_run(nd, tmp_path / "out")


def test_dry_path_ignores_sigma_half(tmp_path):
    """J1 冻结兼容：干燥臂即使 manifest 带了错误 sigma_half 也不得读取（产物逐位不变）。"""
    nd = tmp_path / "native"
    mf = write_native_run(nd, seed=26)
    mf["grid"]["sigma_half"] = [0.0, 0.9, 0.95, 1.0]
    (nd / "manifest.json").write_text(json.dumps(mf), encoding="utf-8")
    jia_remap.convert_native_run(nd, tmp_path / "out")
    with np.load(tmp_path / "out/coords.npz") as z:
        sigma = z["sigma"]
        want = np.concatenate(([0.0], 0.5 * (sigma[:-1] + sigma[1:]), [1.0]))
        np.testing.assert_array_equal(z["sigma_edges"], want)
    assert set(loader.load_run(tmp_path / "out")["snaps"][0]) == set(loader.DRY_SNAP_FIELDS)


def test_wet_flux_meta_passthrough(tmp_path):
    """Track A：湿 manifest 的五个白名单键逐值透传到目标 meta.json。"""
    nd = tmp_path / "native"
    mf = write_wet_native_run(nd, seed=27)
    mf.update({
        "flux_semantics": "interval_mean",
        "flux_interval_s": 172800.0,
        "flux_accumulation": "rate*dt per step; Float64 accumulator",
        "flux_valid_from_snap": 2,
        "flux_first_interval_s": 86400.0,
    })
    (nd / "manifest.json").write_text(json.dumps(mf), encoding="utf-8")
    rep = jia_remap.convert_native_run(nd, tmp_path / "out")
    assert rep["self_check_bad"] == 0
    meta = loader.load_run(tmp_path / "out")["meta"]
    for k in jia_remap.FLUX_META_KEYS:
        assert meta[k] == mf[k], k


def test_wet_manifest_rejects_flux_keys_nested_in_export(tmp_path):
    """评审 §36.4 缺陷 1：五键只在 manifest.export 而顶层缺失 ⇒ 报错，不得静默丢键。"""
    nd = tmp_path / "native"
    mf = write_wet_native_run(nd, seed=30)
    flux = {
        "flux_semantics": "interval_mean",
        "flux_interval_s": 172800.0,
        "flux_accumulation": "rate*dt per step; Float64 accumulator",
        "flux_valid_from_snap": 2,
        "flux_first_interval_s": 86400.0,
    }
    mf["export"].update(flux)
    (nd / "manifest.json").write_text(json.dumps(mf), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest.export"):
        jia_remap.convert_native_run(nd, tmp_path / "out")


def test_wet_flux_meta_absent_when_not_declared(tmp_path):
    """瞬时率模式（manifest 无键）⇒ 目标 meta 不得凭空出现 flux 登记键。"""
    nd = tmp_path / "native"
    write_wet_native_run(nd, seed=28)
    jia_remap.convert_native_run(nd, tmp_path / "out")
    meta = loader.load_run(tmp_path / "out")["meta"]
    assert not (set(jia_remap.FLUX_META_KEYS) & set(meta))


def test_dry_ignores_flux_meta_keys_bitwise(tmp_path):
    """干臂白名单边界：即使 manifest 出现 flux 登记键也不透传，产物与无键版逐位一致。"""
    nd_a = tmp_path / "native_a"
    write_native_run(nd_a, seed=29)
    jia_remap.convert_native_run(nd_a, tmp_path / "out_a")
    nd_b = tmp_path / "native_b"
    mf = write_native_run(nd_b, seed=29)
    mf.update({k: 1 for k in jia_remap.FLUX_META_KEYS})
    (nd_b / "manifest.json").write_text(json.dumps(mf), encoding="utf-8")
    jia_remap.convert_native_run(nd_b, tmp_path / "out_b")
    for name in ("meta.json", "coords.npz", "snap_00001.npz"):
        assert (tmp_path / "out_a" / name).read_bytes() == \
            (tmp_path / "out_b" / name).read_bytes(), name


# ---------------------------------------------------------------- IC 协议（AV 侧材料化）

def test_ic_iso_w1_is_frozen_protocol():
    lat, lon = octa_grid()
    ic = jia_remap.make_native_ic(lat, lon, np.array(JIA_SIGMA))
    spec = jia_remap.IC_PROTOCOL["iso_w1"]
    assert ic["T"].shape == (lat.size, 8)
    np.testing.assert_allclose(ic["u"], 0.0)
    assert ic["ps"][0] == pytest.approx(math.log(jia_remap.PROTOCOL["ps0_Pa"]))
    i = int(np.argmin(np.abs(lon)))                             # lon=0 ⇒ 扰动 0
    np.testing.assert_allclose(ic["T"][i], spec["T0_K"], atol=1e-12)
    j = int(np.argmin(np.abs(lon - 90.0)))                      # lon=90° ⇒ +amp
    np.testing.assert_allclose(ic["T"][j], spec["T0_K"] + spec["amp_K"], atol=1e-9)
    assert ic["T"].max() <= spec["T0_K"] + spec["amp_K"] + 1e-9
    assert ic["T"].min() >= spec["T0_K"] - spec["amp_K"] - 1e-9


def test_materialize_ic_writes_float32_and_stable_sha(tmp_path):
    nd = tmp_path / "native"
    write_native_run(nd, seed=9)
    rep = jia_remap.materialize_ic(nd, tmp_path / "ic.npz", "iso_w1")
    with np.load(tmp_path / "ic.npz") as z:
        assert z["T"].dtype == np.float32 and z["T"].shape == (80, 3)
        assert z["u"].shape == (80, 3) and z["ps"].shape == (80,)
    again = jia_remap.materialize_ic(nd, tmp_path / "ic2.npz", "iso_w1")
    assert rep["sha256"] == again["sha256"]                     # 同协议同几何 ⇒ 同一 IC
    assert rep["ps_units"] == "log(Pa)"


def test_materialize_ic_rejects_name_mismatch(tmp_path):
    nd = tmp_path / "native"
    write_native_run(nd, seed=10)
    with pytest.raises(ValueError, match="ic.name"):
        jia_remap.materialize_ic(nd, tmp_path / "ic.npz", "teq_noise")


def test_ic_teq_noise_uses_frozen_315_reference():
    """对照臂必须用冻结的 Teq 参考常数（315 K）⇒ 其热带低层平衡值明显高于 288 K 等温臂。"""
    lat, lon = octa_grid()
    ic = jia_remap.make_native_ic(lat, lon, np.array(JIA_SIGMA), name="teq_noise", seed=1)
    assert ic["T"].shape == (80, 8) and np.all(np.isfinite(ic["T"]))
    assert ic["T"].max() > 288.0 + 1.0


# ---------------------------------------------------------------- 可评估性表的证据锚

@pytest.mark.parametrize("label, lo, hi", CRITERIA_P_WINDOWS)
def test_pressure_window_populated_by_jia_ladder(label, lo, hi):
    """判据气压窗 ↔ 甲 8 层 σ 阶梯（p=σ·ps0）对照：空窗 ⇒ 该判据在甲上不良定义 ⇒ 报 owner 裁。"""
    plev = np.array(JIA_SIGMA) * 1000.0
    if label in EMPTY_WINDOWS:
        with pytest.raises(ValueError, match="空气压层窗口"):
            suites._lev_mask(plev, lo, hi)
    else:
        assert int(suites._lev_mask(plev, lo, hi).sum()) >= 1


def jia_T32_grid(nring=48, reach_deg=87.1591):
    """甲（SpeedyWeather T=32, nlayers=8）的环结构复刻：48 环、点数 20→112（步长 4）再对称、
    纬度触达 ±87.1591°、总点 3168。几何=接口事实（非结果值）；纬度用等 Δμ 近似，
    与真高斯节点的差 <1 环带，不影响"带是否有行"的判定。"""
    mu_max = math.sin(math.radians(reach_deg))
    mu = np.linspace(-mu_max, mu_max, nring)
    ramp = 20 + 4 * np.arange(nring // 2)
    counts = np.concatenate((ramp, ramp[::-1]))
    assert counts.sum() == 3168 and counts.max() == 112
    lat = np.repeat(np.rad2deg(np.arcsin(mu)), counts)
    lon = np.concatenate([np.arange(n) * (360.0 / n) for n in counts])
    return lat, lon


def test_lat_bands_populated_after_remap():
    """判据纬度带（|lat|≥20 急流、75–90° 极区、≤15° 热带）在目标网格上有行；
    甲纬度只触达 87.16° ⇒ 被排除的极帽面积 (1−μ_max) ≈ 0.13%：登记、不外推填极。"""
    lat, lon = jia_T32_grid()
    rings = jia_remap.ring_of(lat, lon)
    tgt = jia_remap.target_grid(rings)
    assert rings["nring"] == 48 and rings["max_ring"] == 112
    assert tgt["nlat"] == 48 and tgt["nlon"] == 224
    assert tgt["mu_max"] == pytest.approx(math.sin(math.radians(87.1591)), abs=1e-12)
    a = tgt["lat"]
    assert np.any(a >= 75.0), "75–90° 带在目标网格上无行"
    assert int(np.sum(np.abs(a) >= 75.0)) == 2            # 极区两行 = CONS-POLAR-* 的口径
    assert np.any(np.abs(a) <= 15.0) and np.any(np.abs(a) >= 20.0)
    assert 1.0 - tgt["mu_max"] < 2e-3
    assert jia_remap.mu_coverage_defect(rings, tgt) < 1e-12
