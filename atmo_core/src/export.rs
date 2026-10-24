//! 出口场导出（`atmo/INTERFACE.md` §8.7 / §10，**v1.0.2 新增**）。
//!
//! A1 只负责三项：`t2m / u10 / v10`；`precip / cloud / sw_sfc / lw_sfc` 由 A2 产出，
//! A3 组装。**禁止**在这里猜降水/云/辐射。
//!
//! ## v0 口径（写死；OPEN-3 裁定为"外推占位"）
//!
//! * 静力积分求层高度（从地面 0 m 起，v0 **不含地形**）：
//!   `z_half[k] = z_half[k+1] + (R_d·T_k/g)·ln(p_half[k+1]/p_half[k])`
//! * `t2m = T[nlev-1] + 0.0098·(z_c[nlev-1] − 2)`
//! * `u10 = u[nlev-1]·ln(10/z0)/ln(z_c/z0)`，`z0 = 0.1 m`
//!
//! 两者都是**近似**：返回 dict 带 `approx=true`，A3 不得当"地表能量平衡"用。

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::grid::GridSpec;

pub const R_D: f64 = 287.0;
pub const G: f64 = 9.81;
pub const Z0_M: f64 = 0.1;
pub const LAPSE_K_PER_M: f64 = 0.0098;

pub struct ExportOut {
    pub t2m: Vec<f64>,
    pub u10: Vec<f64>,
    pub v10: Vec<f64>,
}

/// 纯 Rust 核：`u/T` 为 `(nlev,nlat,nlon)`，`ps` 为 `(nlat,nlon)`（均 C 序）
pub fn export_fields(g: &GridSpec, u: &[f64], v: &[f64], t: &[f64], ps: &[f64]) -> ExportOut {
    let (nlat, nlon, nlev) = (g.nlat, g.nlon, g.nlev);
    let n = nlat * nlon;
    let mut z_half = vec![0.0f64; nlev + 1];
    let mut t2m = vec![0.0f64; n];
    let mut u10 = vec![0.0f64; n];
    let mut v10 = vec![0.0f64; n];

    for j in 0..nlat {
        for i in 0..nlon {
            let k2 = j * nlon + i;
            let psfc = ps[k2].max(1.0);
            // 静力积分：自下而上
            z_half[nlev] = 0.0;
            for k in (0..nlev).rev() {
                let tk = t[k * n + k2].max(1.0);
                let p_lo = psfc * g.sigma_half[k + 1];
                let p_hi = psfc * g.sigma_half[k];
                z_half[k] = z_half[k + 1] + (R_D * tk / G) * (p_lo / p_hi.max(1e-30)).ln();
            }
            // 最低层中心高度
            let k = nlev - 1;
            let tk = t[k * n + k2].max(1.0);
            let p_lo = psfc * g.sigma_half[k + 1];
            let p_c = psfc * g.sigma_full[k];
            let zc = z_half[k + 1] + (R_D * tk / G) * (p_lo / p_c.max(1e-30)).ln();
            t2m[k2] = t[k * n + k2] + LAPSE_K_PER_M * (zc - 2.0);
            let ratio = if zc > Z0_M {
                (10.0f64 / Z0_M).ln() / (zc / Z0_M).ln()
            } else {
                1.0
            };
            u10[k2] = u[k * n + k2] * ratio;
            v10[k2] = v[k * n + k2] * ratio;
        }
    }
    ExportOut { t2m, u10, v10 }
}

/// `fields = ac.export(state[, grid])`
///
/// `grid` 可省（A3 的 `ModuleAtmoCore.export(state)` 只传 state）：
/// 缺省时按 `state["u"].shape = (nlev,nlat,nlon)` 重建默认 T21 网格。
#[pyfunction]
#[pyo3(signature = (state, grid=None))]
pub fn export<'py>(
    py: Python<'py>,
    state: &Bound<'py, PyDict>,
    grid: Option<&Bound<'py, PyDict>>,
) -> PyResult<Bound<'py, PyDict>> {
    let need = |k: &str| -> PyResult<Bound<'py, pyo3::types::PyAny>> {
        state
            .get_item(k)?
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err(format!("状态缺 `{k}`")))
    };
    let (su, u) = crate::ops::read_f64(&need("u")?)?;
    let (_, v) = crate::ops::read_f64(&need("v")?)?;
    let (_, t) = crate::ops::read_f64(&need("T")?)?;
    let (_, ps) = crate::ops::read_f64(&need("ps")?)?;
    if su.len() != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "u 必须是 3D (nlev,nlat,nlon)",
        ));
    }
    let g = match grid {
        Some(d) => crate::grid::from_dict(d)?,
        None => GridSpec::new(su[1], su[2], su[0], "gaussian", 86400.0),
    };
    if g.nlat != su[1] || g.nlon != su[2] || g.nlev != su[0] {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "状态形状 {su:?} 与网格 ({}, {}, {}) 不符",
            g.nlev, g.nlat, g.nlon
        )));
    }
    let e = export_fields(&g, &u, &v, &t, &ps);
    let shape = [g.nlat, g.nlon];
    let d = PyDict::new(py);
    d.set_item("t2m", crate::ops::write_like(py, &shape, e.t2m)?)?;
    d.set_item("u10", crate::ops::write_like(py, &shape, e.u10)?)?;
    d.set_item("v10", crate::ops::write_like(py, &shape, e.v10)?)?;
    d.set_item("approx", true)?;
    Ok(d)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn isothermal_rest_profile() {
        let g = GridSpec::new(32, 64, 8, "gaussian", 86400.0);
        let n = g.nlat * g.nlon;
        let u = vec![10.0f64; 8 * n];
        let v = vec![-4.0f64; 8 * n];
        let t = vec![280.0f64; 8 * n];
        let ps = vec![101325.0f64; n];
        let e = export_fields(&g, &u, &v, &t, &ps);
        // 最低层中心高度约 640 m ⇒ t2m 比 T 暖约 6 K
        let t2m = e.t2m[0];
        assert!((t2m - 280.0).abs() > 4.0 && (t2m - 280.0).abs() < 8.0, "t2m={t2m}");
        // u10 约为 u 的一半
        assert!((e.u10[0] / 10.0 - 0.5).abs() < 0.05, "u10/u={}", e.u10[0] / 10.0);
        assert!(e.v10[0] < 0.0);
        assert!(e.t2m.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn zero_wind_stays_zero() {
        let g = GridSpec::new(32, 64, 8, "gaussian", 86400.0);
        let n = g.nlat * g.nlon;
        let e = export_fields(
            &g,
            &vec![0.0; 8 * n],
            &vec![0.0; 8 * n],
            &vec![288.0; 8 * n],
            &vec![101325.0; n],
        );
        assert!(e.u10.iter().all(|&x| x == 0.0));
        assert!(e.v10.iter().all(|&x| x == 0.0));
    }
}
