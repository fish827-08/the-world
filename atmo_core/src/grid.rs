//! 球面网格：Gaussian（默认）/ 规则经纬，以及全部派生几何量。
//!
//! 口径（见 `atmo/INTERFACE.md` §0/§2，改动即作废旧读数）：
//! * `j = 0` 是**最南**行，向北递增；`i = 0` 是 0°E，向东递增，范围 `[0,360)` 周期。
//! * 面积权 `area[j] = dmu[j]/2`，`Σ area = 1`（全球平均 = `Σ_j area_j·mean_i(·)`）。
//! * 有限体积算子在 `(μ, λ)` 上做，权重直接用 `dmu` 与 `Δλ`；`dx_m/dy_m` 只供 CFL 报告。
//! * 所有超越函数走 `detmath`，保证跨平台逐位一致。

use crate::detmath;
use numpy::IntoPyArray;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

pub const RADIUS_M: f64 = 6_371_000.0;
/// 8 层 sigma 界面（模式顶 → 地表）
pub const SIGMA_HALF: [f64; 9] = [0.0, 0.05, 0.14, 0.26, 0.40, 0.55, 0.70, 0.85, 1.00];

#[derive(Clone, Debug)]
pub struct GridSpec {
    pub nlat: usize,
    pub nlon: usize,
    pub nlev: usize,
    pub kind: String,
    pub mu: Vec<f64>,
    pub dmu: Vec<f64>,
    pub area: Vec<f64>,
    pub lat_rad: Vec<f64>,
    pub lon_rad: Vec<f64>,
    /// 显示用：相邻行中心 μ 的中点边（INTERFACE §2.3：**仅供显示/画图**）
    pub lat_edges_rad: Vec<f64>,
    /// 真实求积区间边界（v1.0.1）：`-1 + 2·cumsum(area)`，两端钉 ±1；保守重映射必须用它
    pub mu_edges: Vec<f64>,
    /// 每行的真实纬向区间 `Δφ_j = asin(mu_edges[j+1]) − asin(mu_edges[j])`
    pub dphi: Vec<f64>,
    /// 行中心的 `cosφ`（**注意**：`mu` 是 `sinφ`，通量项必须用这个）
    pub cosphi: Vec<f64>,
    /// **面积一致**的 `cosφ`：`cosbar_j = dmu_j / dphi_j`（区间平均，见 `ops` 模块头）
    pub cosbar: Vec<f64>,
    pub coriolis: Vec<f64>,
    pub dx_m: Vec<f64>,
    pub dy_m: Vec<f64>,
    pub sigma_half: Vec<f64>,
    pub sigma_full: Vec<f64>,
    pub radius_m: f64,
    pub rotation_period_s: f64,
}

/// n 阶 Legendre 多项式在 z 处的值与导数。
fn legendre(n: usize, z: f64) -> (f64, f64) {
    let mut p1 = 1.0f64;
    let mut p2 = 0.0f64;
    for j in 1..=n {
        let p3 = p2;
        p2 = p1;
        p1 = ((2.0 * j as f64 - 1.0) * z * p2 - (j as f64 - 1.0) * p3) / (j as f64);
    }
    let pp = n as f64 * (z * p1 - p2) / (z * z - 1.0);
    (p1, pp)
}

/// 高斯节点（P_n 的根）与权重，**按 μ 升序返回**（南 → 北），`Σ w = 2`。
///
/// 算法：Newton 迭代（初值 `cos(π(i−0.25)/(n+0.5))`），收敛判据 `|Δz| < 1e-16`，
/// 上限 200 次（纯算术 ⇒ 逐位可复现）。
pub fn gauss_nodes(n: usize) -> (Vec<f64>, Vec<f64>) {
    assert!(n >= 2, "至少 2 个纬度点");
    let mut nodes: Vec<(f64, f64)> = Vec::with_capacity(n);
    for i in 1..=n {
        let mut z = detmath::cos(detmath::PI * (i as f64 - 0.25) / (n as f64 + 0.5));
        for _ in 0..200 {
            let (p, pp) = legendre(n, z);
            let z1 = z;
            z = z1 - p / pp;
            if (z - z1).abs() < 1e-16 {
                break;
            }
        }
        let (_, pp) = legendre(n, z);
        let w = 2.0 / ((1.0 - z * z) * pp * pp);
        nodes.push((z, w));
    }
    nodes.sort_by(|a, b| a.0.total_cmp(&b.0));
    (nodes.iter().map(|t| t.0).collect(), nodes.iter().map(|t| t.1).collect())
}

impl GridSpec {
    pub fn new(nlat: usize, nlon: usize, nlev: usize, kind: &str, rotation_period_s: f64) -> Self {
        assert!(nlat >= 4 && nlon >= 8, "网格太小");
        assert!(nlev >= 1, "nlev 至少 1");
        assert!(rotation_period_s > 0.0, "自转周期必须为正");

        let (mu, dmu) = match kind {
            "gaussian" => gauss_nodes(nlat),
            "regular" => {
                // 等距纬度（格中心）：φ_j = −90 + (j+0.5)·180/nlat
                let mut mu = Vec::with_capacity(nlat);
                let mut edges = Vec::with_capacity(nlat + 1);
                edges.push(-1.0f64);
                for j in 0..nlat {
                    let lat = -90.0 + (j as f64 + 0.5) * 180.0 / nlat as f64;
                    let m = detmath::sin(lat * detmath::PI / 180.0);
                    mu.push(m);
                    edges.push(detmath::sin((-90.0 + (j as f64 + 1.0) * 180.0 / nlat as f64)
                        * detmath::PI / 180.0));
                }
                *edges.last_mut().unwrap() = 1.0;
                let dmu: Vec<f64> = (0..nlat).map(|j| edges[j + 1] - edges[j]).collect();
                (mu, dmu)
            }
            other => panic!("未知网格类型 {other}（只支持 gaussian/regular）"),
        };

        // ① 显示用边界：μ 的中点，两端钉 ±1
        let mut lat_edges_mu = Vec::with_capacity(nlat + 1);
        lat_edges_mu.push(-1.0);
        for j in 0..nlat.saturating_sub(1) {
            lat_edges_mu.push(0.5 * (mu[j] + mu[j + 1]));
        }
        lat_edges_mu.push(1.0);

        // ② 真实求积边界（INTERFACE v1.0.1 / A3 保守重映射用）：-1 + 2·cumsum(area)
        let area: Vec<f64> = dmu.iter().map(|&d| d / 2.0).collect();
        let mut mu_edges = Vec::with_capacity(nlat + 1);
        {
            let mut acc = 0.0f64;
            mu_edges.push(-1.0);
            for j in 0..nlat.saturating_sub(1) {
                acc += area[j];
                mu_edges.push(-1.0 + 2.0 * acc);
            }
            mu_edges.push(1.0);
        }

        let lat_rad: Vec<f64> = mu.iter().map(|&m| detmath::asin(m)).collect();
        let lat_edges_rad: Vec<f64> = lat_edges_mu.iter().map(|&m| detmath::asin(m)).collect();
        let true_edges_rad: Vec<f64> = mu_edges.iter().map(|&m| detmath::asin(m)).collect();
        let lon_rad: Vec<f64> = (0..nlon)
            .map(|i| 2.0 * detmath::PI * i as f64 / nlon as f64)
            .collect();
        let dphi: Vec<f64> = (0..nlat)
            .map(|j| true_edges_rad[j + 1] - true_edges_rad[j])
            .collect();
        // 面积一致 cosφ：Δμ/Δφ（区间平均 —— 让 Σ area·散度 精确抵消到舍入）
        let cosbar: Vec<f64> = (0..nlat).map(|j| dmu[j] / dphi[j]).collect();
        // 行中心点值 cosφ
        let cosphi: Vec<f64> = mu.iter().map(|&m| (1.0 - m * m).sqrt()).collect();

        let omega = 2.0 * detmath::PI / rotation_period_s;
        let coriolis: Vec<f64> = mu.iter().map(|&m| 2.0 * omega * m).collect();

        let two_pi_r = 2.0 * detmath::PI * RADIUS_M;
        let dx_m: Vec<f64> = mu
            .iter()
            .map(|&m| two_pi_r * (1.0 - m * m).sqrt() / nlon as f64)
            .collect();
        let dy_m: Vec<f64> = (0..nlat)
            .map(|j| RADIUS_M * (lat_edges_rad[j + 1] - lat_edges_rad[j]))
            .collect();

        let sigma_half = SIGMA_HALF.to_vec();
        let sigma_full: Vec<f64> = (0..nlev)
            .map(|k| 0.5 * (sigma_half[k] + sigma_half[k + 1]))
            .collect();

        GridSpec {
            nlat,
            nlon,
            nlev,
            kind: kind.to_string(),
            mu,
            dmu,
            area,
            lat_rad,
            lon_rad,
            lat_edges_rad,
            mu_edges,
            dphi,
            cosbar,
            cosphi,
            coriolis,
            dx_m,
            dy_m,
            sigma_half,
            sigma_full,
            radius_m: RADIUS_M,
            rotation_period_s,
        }
    }

    /// 全球面积权平均（`Σ area_j · mean_i`）—— W1 起被守恒诊断使用
    #[allow(dead_code)]
    pub fn global_mean(&self, field: &[f64]) -> f64 {
        // field: (nlat*nlon)，C 序
        let mut acc = 0.0f64;
        for j in 0..self.nlat {
            let mut row = 0.0f64;
            for i in 0..self.nlon {
                row += field[j * self.nlon + i];
            }
            acc += self.area[j] * (row / self.nlon as f64);
        }
        acc
    }
}

/// 构造网格并返回全部派生量（Python 侧 `grid_spec`，见 `atmo/INTERFACE.md` §2.3）。
#[pyfunction]
#[pyo3(signature = (nlat=32, nlon=64, nlev=8, kind="gaussian", rotation_period_s=86400.0))]
pub fn grid_spec<'py>(
    py: Python<'py>,
    nlat: usize,
    nlon: usize,
    nlev: usize,
    kind: &str,
    rotation_period_s: f64,
) -> PyResult<Bound<'py, PyDict>> {
    if kind != "gaussian" && kind != "regular" {
        return Err(PyValueError::new_err(format!(
            "kind 只支持 gaussian/regular（收到 {kind}）"
        )));
    }
    if !(rotation_period_s.is_finite() && rotation_period_s > 0.0) {
        return Err(PyValueError::new_err("rotation_period_s 必须是有限正数"));
    }
    if nlat < 4 || nlon < 8 {
        return Err(PyValueError::new_err("网格太小（nlat≥4, nlon≥8）"));
    }
    let g = GridSpec::new(nlat, nlon, nlev, kind, rotation_period_s);
    let d = PyDict::new(py);
    d.set_item("nlat", g.nlat)?;
    d.set_item("nlon", g.nlon)?;
    d.set_item("nlev", g.nlev)?;
    d.set_item("kind", g.kind.clone())?;
    d.set_item("mu", g.mu.clone().into_pyarray(py))?;
    d.set_item("dmu", g.dmu.clone().into_pyarray(py))?;
    d.set_item("area", g.area.clone().into_pyarray(py))?;
    d.set_item(
        "lat_deg",
        g.lat_rad.iter().map(|&v| detmath::to_deg(v)).collect::<Vec<f64>>().into_pyarray(py),
    )?;
    d.set_item("lat_rad", g.lat_rad.clone().into_pyarray(py))?;
    d.set_item(
        "lon_deg",
        g.lon_rad.iter().map(|&v| detmath::to_deg(v)).collect::<Vec<f64>>().into_pyarray(py),
    )?;
    d.set_item("lon_rad", g.lon_rad.clone().into_pyarray(py))?;
    d.set_item(
        "lat_edges_deg",
        g.lat_edges_rad
            .iter()
            .map(|&v| detmath::to_deg(v))
            .collect::<Vec<f64>>()
            .into_pyarray(py),
    )?;
    d.set_item("coriolis", g.coriolis.clone().into_pyarray(py))?;
    d.set_item("mu_edges", g.mu_edges.clone().into_pyarray(py))?;
    d.set_item("dx_m", g.dx_m.clone().into_pyarray(py))?;
    d.set_item("dy_m", g.dy_m.clone().into_pyarray(py))?;
    d.set_item("sigma_half", g.sigma_half.clone().into_pyarray(py))?;
    d.set_item("sigma_full", g.sigma_full.clone().into_pyarray(py))?;
    d.set_item("radius_m", g.radius_m)?;
    d.set_item("rotation_period_s", g.rotation_period_s)?;
    Ok(d)
}

/// 只取高斯节点（度，升序）——AV 做网格自检用。
#[pyfunction]
pub fn gaussian_latitudes<'py>(py: Python<'py>, nlat: usize) -> PyResult<Bound<'py, numpy::PyArray1<f64>>> {
    let (mu, _w) = gauss_nodes(nlat);
    let lat: Vec<f64> = mu.iter().map(|&m| detmath::to_deg(detmath::asin(m))).collect();
    Ok(lat.into_pyarray(py))
}

/// 从 `grid_spec()` 返回的 dict 还原 `GridSpec`（算子/推进共用的入口）。
pub fn from_dict(d: &Bound<'_, PyDict>) -> PyResult<GridSpec> {
    let get = |k: &str| -> PyResult<Bound<'_, PyAny>> {
        d.get_item(k)?
            .ok_or_else(|| PyValueError::new_err(format!("网格字典缺键 `{k}`")))
    };
    let nlat = get("nlat")?.extract::<usize>()?;
    let nlon = get("nlon")?.extract::<usize>()?;
    let nlev = get("nlev")?.extract::<usize>()?;
    let kind = get("kind")?.extract::<String>()?;
    let rot = match d.get_item("rotation_period_s")? {
        Some(v) => v.extract::<f64>()?,
        None => 86400.0,
    };
    Ok(GridSpec::new(nlat, nlon, nlev, &kind, rot))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gaussian_nodes_t21_golden() {
        let (mu, w) = gauss_nodes(32);
        // 金标（numpy 复算，见汇报）：P32 的最南/次南/赤道两侧
        let golden = [
            (-9.972_638_618_494_815_7e-1, 0),
            (-9.856_115_115_452_683_8e-1, 1),
            (-4.830_766_568_773_831e-2, 15),
        ];
        for (expect, idx) in golden {
            assert!(
                (mu[idx] - expect).abs() < 1e-14,
                "mu[{idx}]={} 期望 {expect}",
                mu[idx]
            );
        }
        assert!((mu[31] + mu[0]).abs() < 1e-15, "应南北对称");
        let s: f64 = w.iter().sum();
        assert!((s - 2.0).abs() < 1e-14, "Σw={s} 应为 2");
    }

    #[test]
    fn area_sums_to_one() {
        for &(nlat, nlon) in &[(32usize, 64usize), (48, 96)] {
            let g = GridSpec::new(nlat, nlon, 8, "gaussian", 86400.0);
            let s: f64 = g.area.iter().sum();
            assert!((s - 1.0).abs() < 1e-14, "nlat={nlat} Σarea={s}");
            // 常数场的全球平均 = 1
            let f = vec![1.0f64; nlat * nlon];
            assert!((g.global_mean(&f) - 1.0).abs() < 1e-14);
        }
    }

    #[test]
    fn coriolis_and_geometry() {
        let g = GridSpec::new(32, 64, 8, "gaussian", 86400.0);
        let omega = 2.0 * detmath::PI / 86400.0;
        for j in 0..g.nlat {
            assert!((g.coriolis[j] - 2.0 * omega * g.mu[j]).abs() < 1e-18);
            // dx = 2πR cosφ/nlon
            let expect = 2.0 * detmath::PI * RADIUS_M
                * (1.0 - g.mu[j] * g.mu[j]).sqrt()
                / g.nlon as f64;
            assert!((g.dx_m[j] - expect).abs() < 1e-6);
        }
        // 赤道两侧行（±2.77°）的 f 很小，但**不是 0**：2Ω·sin(2.77°) ≈ 7.0e-6
        assert!(g.coriolis[15].abs() < 1e-5 && g.coriolis[16].abs() < 1e-5);
        assert!(g.coriolis[15] < 0.0 && g.coriolis[16] > 0.0, "f 应随纬度变号");
        // 极区纬向格距显著小于赤道（这是极区 CFL 的根源）
        assert!(g.dx_m[0] < 0.1 * g.dx_m[15]);
    }

    #[test]
    fn sigma_table() {
        let g = GridSpec::new(32, 64, 8, "gaussian", 86400.0);
        assert_eq!(g.sigma_half.len(), 9);
        assert_eq!(g.sigma_full.len(), 8);
        assert_eq!(g.sigma_half[0], 0.0);
        assert_eq!(g.sigma_half[8], 1.0);
        for k in 0..8 {
            assert!(g.sigma_full[k] > g.sigma_half[k] && g.sigma_full[k] < g.sigma_half[k + 1]);
        }
        assert!((g.sigma_full[0] - 0.025).abs() < 1e-15);
    }
}
