//! 球面算子（散度/涡度/梯度/拉普拉斯）—— **W1 交付**。
//!
//! ## 离散口径（与 `atmo/INTERFACE.md` §2.3 一致）
//!
//! 在 `(μ, λ)` 上做**有限体积**：`μ = sinφ`，`Δλ = 2π/nlon`。
//!
//! * 纬向：中心差分（周期），`d/dλ`。
//! * 经向：**界面通量差分**——把 `v·cosφ` 线性插值到行界面，再除以 `Δφ_j`
//!   （`Δφ_j` 来自真实求积边界 `mu_edges`，不是中点）。
//! * 度量因子用 **`cosbar_j = Δμ_j/Δφ_j`**（区间平均余弦），不是点值 `cosφ_j`。
//!   这样 `Σ_j area_j · D_j = 0` **精确成立**（内部通量抵消 + 极界通量为 0），
//!   AV 可直接用"全球平均散度 = 0（到舍入）"当判据。
//!
//! 边界：极点界面通量取 0（`v(±π/2)=0`）；经向梯度在首/末行用**三点单侧**二阶差分。

use numpy::ndarray::{Array2, Array3};
use numpy::{IntoPyArray, PyReadonlyArrayDyn};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};

use crate::detmath::PI;
use crate::grid::GridSpec;

// ---------------------------------------------------------------- 纯 Rust 核

/// 纬向中心差分 `∂f/∂λ`（周期）
#[inline]
fn dzonal(f: &[f64], j: usize, i: usize, nlon: usize, dlon: f64) -> f64 {
    let ip = if i + 1 == nlon { 0 } else { i + 1 };
    let im = if i == 0 { nlon - 1 } else { i - 1 };
    (f[j * nlon + ip] - f[j * nlon + im]) / (2.0 * dlon)
}

/// 经向中心差分 `∂f/∂φ`（首末行三点单侧）
fn dmerid(f: &[f64], j: usize, i: usize, g: &GridSpec) -> f64 {
    let (nlat, nlon) = (g.nlat, g.nlon);
    let phi = &g.lat_rad;
    if j > 0 && j + 1 < nlat {
        (f[(j + 1) * nlon + i] - f[(j - 1) * nlon + i]) / (phi[j + 1] - phi[j - 1])
    } else if j == 0 {
        // 三点单侧（非均匀网格）在 x0 的导数
        let (x0, x1, x2) = (phi[0], phi[1], phi[2]);
        let (f0, f1, f2) = (f[i], f[nlon + i], f[2 * nlon + i]);
        f0 * (2.0 * x0 - x1 - x2) / ((x0 - x1) * (x0 - x2))
            + f1 * (x0 - x2) / ((x1 - x0) * (x1 - x2))
            + f2 * (x0 - x1) / ((x2 - x0) * (x2 - x1))
    } else {
        let (x0, x1, x2) = (phi[nlat - 3], phi[nlat - 2], phi[nlat - 1]);
        let (f0, f1, f2) = (
            f[(nlat - 3) * nlon + i],
            f[(nlat - 2) * nlon + i],
            f[(nlat - 1) * nlon + i],
        );
        f0 * (x2 - x1) / ((x0 - x1) * (x0 - x2))
            + f1 * (x2 - x0) / ((x1 - x0) * (x1 - x2))
            + f2 * (2.0 * x2 - x0 - x1) / ((x2 - x0) * (x2 - x1))
    }
}

/// 水平散度 `∇·v`（一层）
pub fn divergence_level(g: &GridSpec, u: &[f64], v: &[f64], out: &mut [f64]) {
    let (nlat, nlon) = (g.nlat, g.nlon);
    let dlon = 2.0 * PI / nlon as f64;
    for j in 0..nlat {
        let inv = 1.0 / (g.radius_m * g.cosbar[j]);
        for i in 0..nlon {
            let du = dzonal(u, j, i, nlon, dlon);
            let fs = if j == 0 {
                0.0
            } else {
                0.5 * (v[(j - 1) * nlon + i] * g.cosphi[j - 1] + v[j * nlon + i] * g.cosphi[j])
            };
            let fnorth = if j + 1 == nlat {
                0.0
            } else {
                0.5 * (v[j * nlon + i] * g.cosphi[j] + v[(j + 1) * nlon + i] * g.cosphi[j + 1])
            };
            out[j * nlon + i] = inv * (du + (fnorth - fs) / g.dphi[j]);
        }
    }
}

/// 相对涡度 `ζ`
pub fn vorticity_level(g: &GridSpec, u: &[f64], v: &[f64], out: &mut [f64]) {
    let (nlat, nlon) = (g.nlat, g.nlon);
    let dlon = 2.0 * PI / nlon as f64;
    for j in 0..nlat {
        let inv = 1.0 / (g.radius_m * g.cosbar[j]);
        for i in 0..nlon {
            let dv = dzonal(v, j, i, nlon, dlon);
            // 界面上的 u·cosφ（边界用本行值外推）
            let gs = if j == 0 {
                u[j * nlon + i] * g.cosphi[j]
            } else {
                0.5 * (u[(j - 1) * nlon + i] * g.cosphi[j - 1] + u[j * nlon + i] * g.cosphi[j])
            };
            let gn = if j + 1 == nlat {
                u[j * nlon + i] * g.cosphi[j]
            } else {
                0.5 * (u[j * nlon + i] * g.cosphi[j] + u[(j + 1) * nlon + i] * g.cosphi[j + 1])
            };
            out[j * nlon + i] = inv * (dv - (gn - gs) / g.dphi[j]);
        }
    }
}

/// 水平平流 `u·∇f`（一层；`u` 的纬向分量乘 `1/(a·cosbar)`）
pub fn advect_level(g: &GridSpec, u: &[f64], v: &[f64], f: &[f64], out: &mut [f64]) {
    let (nlat, nlon) = (g.nlat, g.nlon);
    let dlon = 2.0 * PI / nlon as f64;
    for j in 0..nlat {
        let invx = 1.0 / (g.radius_m * g.cosbar[j]);
        for i in 0..nlon {
            let k = j * nlon + i;
            out[k] = u[k] * invx * dzonal(f, j, i, nlon, dlon) + v[k] * dmerid(f, j, i, g) / g.radius_m;
        }
    }
}

/// 水平梯度 `(∂f/∂x, ∂f/∂y)`
pub fn gradient_level(g: &GridSpec, f: &[f64], gx: &mut [f64], gy: &mut [f64]) {
    let (nlat, nlon) = (g.nlat, g.nlon);
    let dlon = 2.0 * PI / nlon as f64;
    for j in 0..nlat {
        let invx = 1.0 / (g.radius_m * g.cosbar[j]);
        for i in 0..nlon {
            gx[j * nlon + i] = invx * dzonal(f, j, i, nlon, dlon);
            gy[j * nlon + i] = dmerid(f, j, i, g) / g.radius_m;
        }
    }
}

/// 拉普拉斯 = `∇·(∇f)`（与上面两个算子同一套离散 ⇒ 自洽）
pub fn laplacian_level(g: &GridSpec, f: &[f64], out: &mut [f64]) {
    let n = g.nlat * g.nlon;
    let mut gx = vec![0.0f64; n];
    let mut gy = vec![0.0f64; n];
    gradient_level(g, f, &mut gx, &mut gy);
    divergence_level(g, &gx, &gy, out);
}

// ---------------------------------------------------------------- 形状适配

/// 读 float64 数组 → (shape, 逻辑序数据)
pub fn read_f64(obj: &Bound<'_, PyAny>) -> PyResult<(Vec<usize>, Vec<f64>)> {
    let ro = obj
        .extract::<PyReadonlyArrayDyn<f64>>()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("需要 float64 ndarray"))?;
    let arr = ro.as_array();
    let shape: Vec<usize> = arr.shape().to_vec();
    Ok((shape, arr.iter().copied().collect()))
}

/// 按输入形状写回
pub fn write_like(py: Python<'_>, shape: &[usize], data: Vec<f64>) -> PyResult<Py<PyAny>> {
    match shape.len() {
        2 => {
            let a = Array2::from_shape_vec((shape[0], shape[1]), data)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("形状不匹配: {e}")))?;
            Ok(a.into_pyarray(py).into_any().unbind())
        }
        3 => {
            let a = Array3::from_shape_vec((shape[0], shape[1], shape[2]), data)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("形状不匹配: {e}")))?;
            Ok(a.into_pyarray(py).into_any().unbind())
        }
        n => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "只支持 2D (nlat,nlon) 或 3D (nlev,nlat,nlon)，收到 {n} 维"
        ))),
    }
}

/// 逐层算子
fn map_levels(
    g: &GridSpec,
    a: &[f64],
    b: Option<&[f64]>,
    f: impl Fn(&GridSpec, &[f64], Option<&[f64]>, &mut [f64]),
) -> Vec<f64> {
    let n = g.nlat * g.nlon;
    let levels = a.len() / n;
    let mut out = vec![0.0f64; a.len()];
    for k in 0..levels {
        let sl = k * n..(k + 1) * n;
        let bs = b.map(|v| &v[sl.clone()]);
        f(g, &a[sl.clone()], bs, &mut out[sl]);
    }
    out
}

fn check_shape(g: &GridSpec, shape: &[usize], name: &str) -> PyResult<()> {
    let ok = (shape.len() == 2 && shape[0] == g.nlat && shape[1] == g.nlon)
        || (shape.len() == 3 && shape[1] == g.nlat && shape[2] == g.nlon);
    if !ok {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{name} 形状 {shape:?} 与网格 ({}, {}, {}) 不符",
            g.nlev, g.nlat, g.nlon
        )));
    }
    Ok(())
}

// ---------------------------------------------------------------- Python 面

/// 水平散度 `∇·v`
#[pyfunction]
#[pyo3(signature = (u, v, grid))]
pub fn divergence<'py>(
    py: Python<'py>,
    u: &Bound<'py, PyAny>,
    v: &Bound<'py, PyAny>,
    grid: &Bound<'py, PyDict>,
) -> PyResult<Py<PyAny>> {
    let g = crate::grid::from_dict(grid)?;
    let (su, du) = read_f64(u)?;
    let (sv, dv) = read_f64(v)?;
    check_shape(&g, &su, "u")?;
    check_shape(&g, &sv, "v")?;
    if su != sv {
        return Err(pyo3::exceptions::PyValueError::new_err("u/v 形状不一致"));
    }
    let out = map_levels(&g, &du, Some(&dv), |g, a, b, o| {
        divergence_level(g, a, b.unwrap(), o)
    });
    write_like(py, &su, out)
}

/// 相对涡度 `ζ = (1/(a cosφ))[∂v/∂λ − ∂(u cosφ)/∂φ]`
#[pyfunction]
#[pyo3(signature = (u, v, grid))]
pub fn vorticity<'py>(
    py: Python<'py>,
    u: &Bound<'py, PyAny>,
    v: &Bound<'py, PyAny>,
    grid: &Bound<'py, PyDict>,
) -> PyResult<Py<PyAny>> {
    let g = crate::grid::from_dict(grid)?;
    let (su, du) = read_f64(u)?;
    let (sv, dv) = read_f64(v)?;
    check_shape(&g, &su, "u")?;
    check_shape(&g, &sv, "v")?;
    let out = map_levels(&g, &du, Some(&dv), |g, a, b, o| {
        vorticity_level(g, a, b.unwrap(), o)
    });
    write_like(py, &su, out)
}

/// 水平梯度 `(∂f/∂x, ∂f/∂y)`
#[pyfunction]
#[pyo3(signature = (field, grid))]
pub fn gradient<'py>(
    py: Python<'py>,
    field: &Bound<'py, PyAny>,
    grid: &Bound<'py, PyDict>,
) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    let g = crate::grid::from_dict(grid)?;
    let (s, d) = read_f64(field)?;
    check_shape(&g, &s, "field")?;
    let n = g.nlat * g.nlon;
    let levels = d.len() / n;
    let mut gx = vec![0.0f64; d.len()];
    let mut gy = vec![0.0f64; d.len()];
    for k in 0..levels {
        let sl = k * n..(k + 1) * n;
        gradient_level(&g, &d[sl.clone()], &mut gx[sl.clone()], &mut gy[sl.clone()]);
    }
    Ok((write_like(py, &s, gx)?, write_like(py, &s, gy)?))
}

/// 水平拉普拉斯 `∇²f`
#[pyfunction]
#[pyo3(signature = (field, grid))]
pub fn laplacian<'py>(
    py: Python<'py>,
    field: &Bound<'py, PyAny>,
    grid: &Bound<'py, PyDict>,
) -> PyResult<Py<PyAny>> {
    let g = crate::grid::from_dict(grid)?;
    let (s, d) = read_f64(field)?;
    check_shape(&g, &s, "field")?;
    let out = map_levels(&g, &d, None, |g, a, _b, o| laplacian_level(g, a, o));
    write_like(py, &s, out)
}

/// 科氏参数 `f = 2Ω sinφ`（`(nlat,)`）
#[pyfunction]
#[pyo3(signature = (grid))]
pub fn coriolis<'py>(grid: &Bound<'py, PyDict>) -> PyResult<Py<PyAny>> {
    match grid.get_item("coriolis")? {
        Some(v) => Ok(v.unbind()),
        None => Err(pyo3::exceptions::PyValueError::new_err(
            "grid 字典缺 `coriolis`（应由 atmo_core.grid_spec 生成）",
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid(nlat: usize, nlon: usize) -> GridSpec {
        GridSpec::new(nlat, nlon, 8, "gaussian", 86400.0)
    }

    /// 零场必须给出零倾向；常数场必须给出零**纬向**梯度
    ///
    /// ⚠️ 球面上"常数 v"**不是**无散场（∇·(0,v,0) = −v tanφ/a），"常数 u"也不是无涡场
    /// （ζ = −u tanφ/a）——这两条是球面度量的真性质，测试不能假设为 0。
    #[test]
    fn zero_field_has_zero_derivatives() {
        let g = grid(32, 64);
        let n = g.nlat * g.nlon;
        let z = vec![0.0f64; n];
        let mut d = vec![0.0; n];
        divergence_level(&g, &z, &z, &mut d);
        assert!(d.iter().all(|x| *x == 0.0));
        vorticity_level(&g, &z, &z, &mut d);
        assert!(d.iter().all(|x| *x == 0.0));
        let (mut gx, mut gy) = (vec![0.0; n], vec![0.0; n]);
        gradient_level(&g, &z, &mut gx, &mut gy);
        assert!(gx.iter().all(|x| *x == 0.0) && gy.iter().all(|x| *x == 0.0));
        laplacian_level(&g, &z, &mut d);
        assert!(d.iter().all(|x| *x == 0.0));

        // 常数 u、v=0 ⇒ 散度 0；涡度 = −u tanφ/a（解析，容差 1%）
        let u = vec![3.0f64; n];
        divergence_level(&g, &u, &z, &mut d);
        assert!(d.iter().all(|x| x.abs() < 1e-18), "常数 u 的散度应为 0");
        vorticity_level(&g, &u, &z, &mut d);
        let mut worst = 0.0f64;
        for j in 2..g.nlat - 2 {
            let expect = 3.0 * g.mu[j] / (g.cosphi[j] * g.radius_m);
            for i in 0..g.nlon {
                worst = worst.max((d[j * g.nlon + i] - expect).abs() / expect.abs().max(1e-30));
            }
        }
        assert!(worst < 1e-2, "常数 u 涡度相对误差 {worst:e}");
    }

    /// 全球平均散度 = 0（精确抵消；AV 的守恒判据之一）
    #[test]
    fn global_mean_divergence_is_zero() {
        let g = grid(32, 64);
        let n = g.nlat * g.nlon;
        let mut u = vec![0.0f64; n];
        let mut v = vec![0.0f64; n];
        for j in 0..g.nlat {
            for i in 0..g.nlon {
                let lam = g.lon_rad[i];
                u[j * g.nlon + i] = 20.0 * crate::detmath::sin(2.0 * lam) * (1.0 - g.mu[j] * g.mu[j]).sqrt();
                v[j * g.nlon + i] = 7.0 * crate::detmath::cos(3.0 * lam) * g.mu[j];
            }
        }
        let mut d = vec![0.0f64; n];
        divergence_level(&g, &u, &v, &mut d);
        let m = g.global_mean(&d);
        assert!(m.abs() < 1e-16, "全球平均散度 {m:e} 不为零");
    }

    /// 固体旋转：u = Ω a cosφ ⇒ ζ = 2Ω sinφ，D = 0
    #[test]
    fn solid_body_rotation_gives_coriolis() {
        let g = grid(32, 64);
        let n = g.nlat * g.nlon;
        let omega0 = 1.0e-5;
        let mut u = vec![0.0f64; n];
        for j in 0..g.nlat {
            for i in 0..g.nlon {
                u[j * g.nlon + i] = omega0 * g.radius_m * g.cosphi[j];
            }
        }
        let v = vec![0.0f64; n];
        let mut d = vec![0.0f64; n];
        let mut z = vec![0.0f64; n];
        divergence_level(&g, &u, &v, &mut d);
        vorticity_level(&g, &u, &v, &mut z);
        assert!(d.iter().all(|x| x.abs() < 1e-14), "固体旋转散度应为 0");
        for j in 2..g.nlat - 2 {
            let expect = 2.0 * omega0 * g.mu[j];
            for i in 0..g.nlon {
                assert!(
                    (z[j * g.nlon + i] - expect).abs() < 1e-2 * expect.abs().max(1e-12),
                    "j={j} ζ={} 期望 {expect}",
                    z[j * g.nlon + i]
                );
            }
        }
    }

    /// 拉普拉斯与解析值：∇²(sinφ) = −2 sinφ / a²
    #[test]
    fn laplacian_of_first_harmonic() {
        let g = grid(32, 64);
        let n = g.nlat * g.nlon;
        let mut f = vec![0.0f64; n];
        for j in 0..g.nlat {
            for i in 0..g.nlon {
                f[j * g.nlon + i] = g.mu[j];
            }
        }
        let mut lap = vec![0.0f64; n];
        laplacian_level(&g, &f, &mut lap);
        let expect = -2.0 / (g.radius_m * g.radius_m);
        let mut err = 0.0f64;
        for j in 2..g.nlat - 2 {
            for i in 0..g.nlon {
                err = err.max((lap[j * g.nlon + i] - expect * g.mu[j]).abs());
            }
        }
        // 相对误差（二阶差分，Δφ≈0.1 rad ⇒ 误差 ~1e-3 量级）
        let scale = 2.0 / (g.radius_m * g.radius_m);
        assert!(err / scale < 1e-2, "∇²sinφ 相对误差 {:e}", err / scale);
    }

    /// 纬向波：∂/∂x 的解析值
    #[test]
    fn zonal_derivative_of_wave() {
        let g = grid(32, 64);
        let n = g.nlat * g.nlon;
        let m = 4.0;
        let mut f = vec![0.0f64; n];
        for j in 0..g.nlat {
            for i in 0..g.nlon {
                f[j * g.nlon + i] = crate::detmath::sin(m * g.lon_rad[i]);
            }
        }
        let mut gx = vec![0.0f64; n];
        let mut gy = vec![0.0f64; n];
        gradient_level(&g, &f, &mut gx, &mut gy);
        let j = 16;
        let mut err = 0.0f64;
        for i in 0..g.nlon {
            let dlam = 2.0 * PI / g.nlon as f64;
            let sinc = crate::detmath::sin(m * dlam) / (m * dlam);
            let expect =
                m * sinc * crate::detmath::cos(m * g.lon_rad[i]) / (g.radius_m * g.cosbar[j]);
            err = err.max((gx[j * g.nlon + i] - expect).abs() / expect.abs().max(1e-12));
        }
        assert!(err < 1e-8, "纬向波 ∂/∂x 相对误差 {err:e}");
        assert!(gy.iter().all(|x| x.abs() < 1e-15));
    }
}
