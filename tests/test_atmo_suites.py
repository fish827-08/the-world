# TIER = "fast"
"""AV 套件管线端到端自检：合成 run 目录（符合 loader 契约）→ evaluate 判定。

关键性质：合成场按解析式构造使 HS/AQ 判据**应当全 PASS**——证明判据可达、
判定逻辑非空转；再注入违规（obliquity、缺 meta 键、套件不符）验证判例路径。
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.atmo import conservation, loader, metrics, suites
from benchmarks.atmo.suites import HS94_PARAMS

NT, NLEV = 6, 3
PLEV = np.array([900.0, 500.0, 250.0])


def _grid(nlat=61, nlon=48):
    lat = np.linspace(-90.0, 90.0, nlat)
    lon = np.linspace(0.0, 360.0, nlon, endpoint=False)
    return lat, lon


def _hs_fields(lat):
    """构造满足全部 HS 判据的解析场：T 剖面、35° 高斯急流、中纬瞬变噪声（仅 500 hPa）。"""
    gauss35 = np.exp(-((np.abs(lat) - 35.0) / 12.0) ** 2)
    t_shape = np.array([1.0, 0.85, 0.71])
    T_zm = (310.0 - 60.0 * np.sin(np.deg2rad(lat)) ** 4)[None, :] * t_shape[:, None]
    A = np.array([0.0, 0.0, 30.0])
    u_zm = A[:, None] * gauss35[None, :] - 1.0
    amp = 3.0 * np.exp(-((np.abs(lat) - 50.0) / 15.0) ** 2) \
        + 0.5 * np.exp(-(lat / 20.0) ** 2)
    rng = np.random.default_rng(11)
    return u_zm, T_zm, amp, rng


def _write_hs_run(root, tag, with_noise=True):
    lat, lon = _grid()
    d = root / tag
    d.mkdir(parents=True)
    np.savez(d / "coords.npz", lat=lat, lon=lon, plev_hpa=PLEV)
    u_zm, T_zm, amp, rng = _hs_fields(lat)
    for t in range(NT):
        u = np.broadcast_to(u_zm[:, :, None], (NLEV, lat.size, lon.size)).copy()
        v = np.zeros_like(u)
        T = np.broadcast_to(T_zm[:, :, None], (NLEV, lat.size, lon.size)).copy()
        if with_noise:
            u[1] += amp[:, None] * rng.normal(0.0, 1.0, (1, lon.size))
            v[1] += amp[:, None] * rng.normal(0.0, 1.0, (1, lon.size))
        np.savez(d / f"snap_{t:06d}.npz", u=u, v=v, T=T,
                 ps=np.full((lat.size, lon.size), 100000.0))
    forcing = {k: float(x) for k, x in HS94_PARAMS.items()}
    (d / "meta.json").write_text(json.dumps({
        "benchmark": "held_suarez", "dt_atmo_s": 600.0, "days": 6,
        "snap_days": [float(i + 1) for i in range(NT)], "obliquity_deg": 0.0,
        "seed": 7, "config_hash": "hs-synth", "cfl_reported": [0.5],
        "forcing": forcing,
    }), encoding="utf-8")
    return d


def _write_aqua_run(root, tag, obliquity=0.0):
    nt = 10
    lat, lon = _grid()
    d = root / tag
    d.mkdir(parents=True)
    np.savez(d / "coords.npz", lat=lat, lon=lon,
             sigma=np.array([0.25, 0.6, 0.9]), sigma_edges=np.array([0.0, 0.5, 0.75, 1.0]))
    cosw = np.cos(np.deg2rad(lat))
    k = (cosw ** 2).mean() / cosw.mean()   # 使 evap−pr 全球面积平均恒为 0
    pr_zm = (10.0 * np.exp(-(lat / 8.0) ** 2)) / 86400.0          # SI
    evap_zm = pr_zm + (2.0 / 86400.0) * (cosw - k)
    pr = np.broadcast_to(pr_zm[:, None], (lat.size, lon.size)).copy()
    evap = np.broadcast_to(evap_zm[:, None], (lat.size, lon.size)).copy()
    for t in range(nt):
        q = np.full((NLEV, lat.size, lon.size), 0.01)
        np.savez(d / f"snap_{t:06d}.npz", u=np.full((NLEV, lat.size, lon.size), 5.0),
                 v=np.zeros((NLEV, lat.size, lon.size)),
                 T=np.full((NLEV, lat.size, lon.size), 300.0),
                 ps=np.full((lat.size, lon.size), 100000.0),
                 q=q, pr=pr, evap=evap,
                 sw_toa=np.full((lat.size, lon.size), 20.0),
                 lw_out=np.full((lat.size, lon.size), 20.0),
                 sst=np.full((lat.size, lon.size), 300.0))
    (d / "meta.json").write_text(json.dumps({
        "benchmark": "aquaplanet", "dt_atmo_s": 600.0, "days": 10,
        "snap_days": [float(i + 1) for i in range(nt)], "obliquity_deg": obliquity,
        "seed": 3, "config_hash": "aq-synth", "cfl_reported": [0.1],
        "radiation": "gray-body", "slab_heat_capacity": 1e7,
    }), encoding="utf-8")
    return d


def _write_gauss_mass_run(root, tag, *, with_mu_edges):
    """Gauss 32 节点 + P2 振幅渐增的 ps：v1 中点带边会读出伪漂移，精确带边≈0（R-8）。"""
    n, nlon = 32, 8
    nodes, gw = np.polynomial.legendre.leggauss(n)
    lat = np.degrees(np.arcsin(nodes))
    lon = np.linspace(0.0, 360.0, nlon, endpoint=False)
    edges = np.concatenate(([-1.0], -1.0 + 2.0 * np.cumsum(gw / 2.0)[:-1], [1.0]))
    p2 = (3.0 * nodes**2 - 1.0) / 2.0
    d = root / tag
    d.mkdir(parents=True)
    coords = {"lat": lat, "lon": lon, "plev_hpa": PLEV}
    if with_mu_edges:
        coords["mu_edges"] = edges
    np.savez(d / "coords.npz", **coords)
    ps_all = []
    for t, ci in enumerate(np.linspace(0.0, 2.34e-3, NT)):
        ps = 1e5 * (1.0 + ci * p2)[:, None] * np.ones((n, nlon))
        ps_all.append(ps)
        np.savez(d / f"snap_{t:06d}.npz",
                 u=np.zeros((NLEV, n, nlon)), v=np.zeros((NLEV, n, nlon)),
                 T=np.full((NLEV, n, nlon), 300.0), ps=ps)
    (d / "meta.json").write_text(json.dumps({
        "benchmark": "held_suarez", "dt_atmo_s": 600.0, "days": float(NT),
        "snap_days": [float(i + 1) for i in range(NT)], "obliquity_deg": 0.0,
        "seed": 9, "config_hash": "gauss-mass-synth",
    }), encoding="utf-8")
    return d, np.stack(ps_all), lat, edges


def _write_aqua_flux_run(root, tag, *, drop_first=False, flux_semantics=None,
                         flux_valid_from_snap=2, q_scales=None, nest_flux_keys=False):
    """Track A 切片证据：首快照 pr=0、evap 仅北半球（极端），其余为对称基准场。
    drop_first=True ⇒ 输出同一物理快照 2..N（snap_days 同步平移），作区间语义的对照。
    q_scales 逐快照缩放 q（默认恒 1）：非恒定 W 才能识别 CONS-WATER 的 W 窗口错切。
    nest_flux_keys=True ⇒ 五键只写进 meta['export']（复现评审 §36.4 缺陷 1）。"""
    nt = 10
    lat, lon = _grid()
    d = root / tag
    d.mkdir(parents=True)
    np.savez(d / "coords.npz", lat=lat, lon=lon,
             sigma=np.array([0.25, 0.6, 0.9]), sigma_edges=np.array([0.0, 0.5, 0.75, 1.0]))
    cosw = np.cos(np.deg2rad(lat))
    k = (cosw ** 2).mean() / cosw.mean()
    pr_base = (10.0 * np.exp(-(lat / 8.0) ** 2)) / 86400.0
    evap_base = pr_base + (2.0 / 86400.0) * (cosw - k)
    pr0 = np.zeros(lat.size)
    evap0 = (30.0 / 86400.0) * np.maximum(cosw, 0.0)
    scales = np.ones(nt) if q_scales is None else np.asarray(q_scales, dtype=np.float64)
    days = list(range(1, nt + 1)) if not drop_first else list(range(2, nt + 1))
    for j, t in enumerate((range(nt) if not drop_first else range(1, nt))):
        pr_zm = pr0 if t == 0 else pr_base
        evap_zm = evap0 if t == 0 else evap_base
        pr = np.broadcast_to(pr_zm[:, None], (lat.size, lon.size)).copy()
        evap = np.broadcast_to(evap_zm[:, None], (lat.size, lon.size)).copy()
        np.savez(d / f"snap_{j:06d}.npz", u=np.full((NLEV, lat.size, lon.size), 5.0),
                 v=np.zeros((NLEV, lat.size, lon.size)),
                 T=np.full((NLEV, lat.size, lon.size), 300.0),
                 ps=np.full((lat.size, lon.size), 100000.0),
                 q=np.full((NLEV, lat.size, lon.size), 0.01 * scales[t]),
                 pr=pr, evap=evap,
                 sw_toa=np.full((lat.size, lon.size), 20.0),
                 lw_out=np.full((lat.size, lon.size), 20.0),
                 sst=np.full((lat.size, lon.size), 300.0))
    meta = {
        "benchmark": "aquaplanet", "dt_atmo_s": 600.0, "days": float(days[-1]),
        "snap_days": [float(x) for x in days], "obliquity_deg": 0.0,
        "seed": 3, "config_hash": "aq-flux-synth", "cfl_reported": [0.1],
        "radiation": "gray-body", "slab_heat_capacity": 1e7,
    }
    if flux_semantics:
        flux = {
            "flux_semantics": flux_semantics,
            "flux_interval_s": 600.0,
            "flux_accumulation": "rate*dt per step; Float64 accumulator",
            "flux_first_interval_s": 600.0,
        }
        if flux_valid_from_snap is not None:
            flux["flux_valid_from_snap"] = flux_valid_from_snap
        if nest_flux_keys:
            meta["export"] = flux
        else:
            meta.update(flux)
    (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return d


def _wet_water_inputs(run_dir):
    run = loader.load_run(run_dir)
    lat = run["coords"]["lat"]
    snaps = run["snaps"]
    pr = np.stack([s["pr"] for s in snaps])
    evap = np.stack([s["evap"] for s in snaps])
    q = np.stack([s["q"] for s in snaps])
    ps = np.stack([s["ps"] for s in snaps])
    days = np.asarray(run["meta"]["snap_days"], dtype=np.float64)
    dp = np.diff(run["coords"]["sigma_edges"])[None, :, None, None] * ps[:, None, :, :]
    w_col = conservation.column_water(q, dp)
    w_ser = np.array([metrics.area_mean(w_col[i], lat) for i in range(q.shape[0])])
    return pr, evap, w_ser, days, lat


def test_aqua_interval_flux_uses_snapshots_from_second(tmp_path):
    """Track A：interval_mean ⇒ pr/evap 类判据用第 2..N 个快照（与丢首快照的对照逐位一致）；
    CONS-WATER 的 W/days 用**全窗**（与直接全窗重算逐位一致，且与错切到 [1:] 的窗不同）。
    瞬时口径受首快照极端值影响 ⇒ 切片非空转；能量判据不在本断言范围。"""
    q_scales = [1.0] + [2.0] * 9   # 首快照后 W 翻倍：错切 W 窗口会恰好掩盖这段变化
    a = _write_aqua_flux_run(tmp_path, "interval", flux_semantics="interval_mean",
                             q_scales=q_scales)
    b = _write_aqua_flux_run(tmp_path, "control", drop_first=True, q_scales=q_scales)
    ra = {r["id"]: r for r in suites.evaluate(suites.AQUAPLANET, a)["rows"]}
    rb = {r["id"]: r for r in suites.evaluate(suites.AQUAPLANET, b)["rows"]}
    for cid in ("AQ-ITCZ-LAT", "AQ-HEMI-DIFF", "AQ-ITCZ-P", "AQ-EP-SUBTROP"):
        assert ra[cid]["measured"] == rb[cid]["measured"], cid
    assert "flux=interval_mean" in ra["CONS-WATER"]["note"]

    pr, evap, w_ser, days, lat = _wet_water_inputs(a)
    full = conservation.water_closure(pr[1:], evap[1:], w_ser, days, lat)
    shifted = conservation.water_closure(pr[1:], evap[1:], w_ser[1:], days[1:], lat)
    assert ra["CONS-WATER"]["measured"] == full["relative"]
    assert shifted["dw_dt_si"] == 0.0          # 错切窗口把 day1→day2 的 W 跃变整段丢掉
    assert full["dw_dt_si"] > 0.0
    assert abs(full["relative"] - shifted["relative"]) > 0.1 * full["relative"]

    inst = _write_aqua_flux_run(tmp_path, "instant", q_scales=q_scales)
    ri = {r["id"]: r for r in suites.evaluate(suites.AQUAPLANET, inst)["rows"]}
    assert ri["AQ-ITCZ-P"]["measured"] != ra["AQ-ITCZ-P"]["measured"]
    assert ri["CONS-WATER"]["measured"] != ra["CONS-WATER"]["measured"]
    assert "flux=instant" in ri["CONS-WATER"]["note"]


@pytest.mark.parametrize("vfs", [None, 3])
def test_aqua_interval_requires_valid_from_snap_2(tmp_path, vfs):
    """interval_mean 的切片前提必须显式且为 2（1-based）；缺失或别的值 ⇒ fail-loud。"""
    d = _write_aqua_flux_run(tmp_path, "bad-vfs", flux_semantics="interval_mean",
                             flux_valid_from_snap=vfs)
    with pytest.raises(ValueError, match="flux_valid_from_snap"):
        suites.evaluate(suites.AQUAPLANET, d)


def test_aqua_flux_keys_nested_in_export_rejected(tmp_path):
    """评审 §36.4 缺陷 1：五键只出现在 meta['export'] 而顶层缺失 ⇒ 必须报错。"""
    d = _write_aqua_flux_run(tmp_path, "nested", flux_semantics="interval_mean",
                             nest_flux_keys=True)
    with pytest.raises(ValueError, match=r"meta\['export'\]"):
        suites.evaluate(suites.AQUAPLANET, d)


def test_aqua_unknown_flux_semantics_rejected(tmp_path):
    d = _write_aqua_flux_run(tmp_path, "bad", flux_semantics="cumulative")
    with pytest.raises(ValueError, match="flux_semantics"):
        suites.evaluate(suites.AQUAPLANET, d)


def test_cons_mass_threads_mu_edges_old_dir_falls_back(tmp_path):
    """R-8：coords 带 mu_edges ⇒ CONS-MASS 用精确权；缺键 ⇒ v1 中点带边逐位不变。"""
    d_old, ps, lat, edges = _write_gauss_mass_run(tmp_path, "old", with_mu_edges=False)
    d_new, _, _, _ = _write_gauss_mass_run(tmp_path, "new", with_mu_edges=True)
    row_old = {r["id"]: r for r in suites.evaluate(suites.HELD_SUAREZ, d_old)["rows"]}["CONS-MASS"]
    row_new = {r["id"]: r for r in suites.evaluate(suites.HELD_SUAREZ, d_new)["rows"]}["CONS-MASS"]
    assert row_old["note"] == "weights=v1(midpoint)"
    assert row_new["note"] == "weights=mu_edges"
    assert row_old["measured"] == conservation.mass_drift(ps, lat)
    assert row_new["measured"] == conservation.mass_drift(ps, lat, mu_edges=edges)
    assert 3.0e-6 < row_old["measured"] < 4.2e-6, row_old["measured"]
    assert row_new["measured"] < 1e-15, row_new["measured"]


def test_hs_synthetic_pass_all_physics(tmp_path):
    da = _write_hs_run(tmp_path, "a")
    db = _write_hs_run(tmp_path, "b")
    report = suites.evaluate(suites.HELD_SUAREZ, da, db)
    by_id = {r["id"]: r for r in report["rows"]}
    fails = [r["id"] for r in report["rows"] if r["status"] == "FAIL"]
    assert fails == [], f"合成场应全过，实际 FAIL: {fails}，读数={ {k: v['measured'] for k, v in by_id.items()} }"
    assert report["verdict"] == "PASS"
    assert by_id["HS-JET-U"]["measured"] > 25.0
    assert by_id["HS-EKE-ML"]["measured"] >= 1.0


def test_hs_without_run_b_is_incomplete(tmp_path):
    da = _write_hs_run(tmp_path, "a")
    report = suites.evaluate(suites.HELD_SUAREZ, da)
    assert report["verdict"] == "INCOMPLETE"
    det = [r for r in report["rows"] if r["id"] == "CONS-DET"][0]
    assert det["status"] == "NOT_RUN"


def test_aqua_synthetic_pass(tmp_path):
    da = _write_aqua_run(tmp_path, "a")
    db = _write_aqua_run(tmp_path, "b")
    report = suites.evaluate(suites.AQUAPLANET, da, db)
    fails = [r["id"] for r in report["rows"] if r["status"] == "FAIL"]
    assert fails == [], f"FAIL: {fails}，读数={ {r['id']: r['measured'] for r in report['rows']} }"
    assert report["verdict"] == "PASS"


def test_aqua_asymmetric_obliquity_marks_na(tmp_path):
    da = _write_aqua_run(tmp_path, "a", obliquity=23.5)
    db = _write_aqua_run(tmp_path, "b", obliquity=23.5)
    report = suites.evaluate(suites.AQUAPLANET, da, db)
    na = {r["id"] for r in report["rows"] if r["status"] == "N/A"}
    assert {"AQ-ITCZ-LAT", "AQ-HEMI-DIFF"} <= na
    assert report["verdict"] == "PASS"


def test_suite_mismatch_rejected(tmp_path):
    da = _write_hs_run(tmp_path, "a")
    with pytest.raises(ValueError, match="不符"):
        suites.evaluate(suites.AQUAPLANET, da)


def test_missing_meta_key_rejected(tmp_path):
    da = _write_hs_run(tmp_path, "a")
    meta = json.loads((da / "meta.json").read_text(encoding="utf-8"))
    del meta["seed"]
    (da / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError, match="seed"):
        suites.evaluate(suites.HELD_SUAREZ, da)
