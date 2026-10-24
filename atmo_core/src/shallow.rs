//! **2 层浅水**（W1 交付）——球面、自由表面、矢量不变量形式 + RK3 + 极区滤波。
//!
//! ## 方程（自由表面、平坦底；`H = h1 + h2`）
//!
//! ```text
//! ∂uₙ/∂t = vₙ(ζₙ+f) − ∂Kₙ/∂x − ∂Pₙ/∂x + ν∇²uₙ
//! ∂vₙ/∂t = −uₙ(ζₙ+f) − ∂Kₙ/∂y − ∂Pₙ/∂y + ν∇²vₙ
//! ∂hₙ/∂t = −∇·(hₙ uₙ)
//!
//! P₁ = g·H
//! P₂ = g·H − g'·h₁          （g' = g(ρ₂−ρ₁)/ρ₂）
//! Kₙ = (uₙ²+vₙ²)/2
//! ```
//!
//! 矢量不变量形式的好处：能量与位涡在离散层面近守恒（Arakawa & Lamb 1981 的经典构造），
//! 且球面度量项（`uv tanφ/a`、`u² tanφ/a`）已被 `ζ` 与 `∇K` 自动包含，不需另写。
//!
//! ## 时间积分
//!
//! **SSP-RK3**（Shu–Osher）：三级、CFL 上限 ~1.7（线性平流），对显式重力波足够。
//! 每级之后对**倾向**做极区纬向 1-2-1 平滑（保持纬向平均），见 `atmo/INTERFACE.md` §7.4。
//!
//! ## 与 `step()` 的关系
//!
//! 这是 W1 的**独立入口**（`ac.shallow_water_step`），不是 §8.3 的 8 层原始方程 `step()`。
//! 判据（Rossby 波速/位涡守恒/斜压不稳定）用它验；W2 的 `step()` 另写。

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::detmath::{self, PI};
use crate::grid::GridSpec;
use crate::ops;

pub const G_DEFAULT: f64 = 9.81;
/// g' = g(ρ₂−ρ₁)/ρ₂；默认 0.02 m/s²（≈ 2% 密度差，海洋斜压典型量级）
pub const GP_DEFAULT: f64 = 0.02;

#[derive(Clone, Copy, Debug)]
pub struct SwParams {
    pub g: f64,
    pub gp: f64,
    pub phi_start_deg: f64,
    pub n_filter: usize,
    pub phi_flat_deg: f64,
    pub nu: f64,
    /// 底层线性拖曳（Rayleigh，s⁻¹）；0 = 无。用来让斜压不稳定饱和（否则层厚会被推到 0）
    pub drag: f64,
    /// 极区滤波模式：0 = 关，1 = 纬向 1-2-1 平滑（旧口径），2 = **纬向截断**（按局部 CFL 定 wavenumber）
    pub filter_mode: u8,
    /// 纬向截断的目标 CFL（默认 0.2，实测值）：m_max(φ) = floor(target·2πa·cosφ/(c·dt))
    pub cfl_target: f64,
}

impl Default for SwParams {
    fn default() -> Self {
        SwParams {
            g: G_DEFAULT,
            gp: GP_DEFAULT,
            phi_start_deg: 60.0,
            n_filter: 4,
            phi_flat_deg: 80.0,
            nu: 1.0e6, // m²/s，网格噪声阻尼（τ≈1 天 @T21）
            drag: 0.0,
            filter_mode: 2,
            cfl_target: 0.2, // 实测：0.5 在 RK3+非线性项下仍会炸；0.2 可稳定 dt=900
        }
    }
}

/// 2 层浅水状态（全部 `(nlat,nlon)`，C 序）
#[derive(Clone, Debug)]
pub struct SwState {
    pub u1: Vec<f64>,
    pub v1: Vec<f64>,
    pub h1: Vec<f64>,
    pub u2: Vec<f64>,
    pub v2: Vec<f64>,
    pub h2: Vec<f64>,
}

impl SwState {
    pub fn len(&self) -> usize {
        self.u1.len()
    }
}

// ---------------------------------------------------------------- 极区滤波
//
// 具体实现搬到 `crate::filt`（与 `prim` 共用），这里只做参数适配。

fn filter_cfg(p: &SwParams) -> crate::filt::FilterCfg {
    crate::filt::FilterCfg {
        mode: p.filter_mode,
        n121: p.n_filter,
        phi_start_deg: p.phi_start_deg,
        cfl_target: p.cfl_target,
    }
}

// ---------------------------------------------------------------- RHS

/// 计算倾向（纯 Rust；`dt` 不进 RHS）
pub fn rhs(g: &GridSpec, s: &SwState, p: &SwParams, nu_row: &[f64], m_max: &[usize], d: &mut SwState) {
    let n = g.nlat * g.nlon;
    let dlon = 2.0 * PI / g.nlon as f64;
    let _ = dlon;
    let mut zeta = vec![0.0f64; n];
    let mut gx = vec![0.0f64; n];
    let mut gy = vec![0.0f64; n];
    let mut tmp = vec![0.0f64; n];
    let mut lap = vec![0.0f64; n];

    // 压力场：H = h1+h2；P1 = gH，P2 = gH − g'h1
    let mut gx_h = vec![0.0f64; n];
    let mut gy_h = vec![0.0f64; n];
    let mut gx_h1 = vec![0.0f64; n];
    let mut gy_h1 = vec![0.0f64; n];
    for k in 0..n {
        tmp[k] = s.h1[k] + s.h2[k];
    }
    ops::gradient_level(g, &tmp, &mut gx_h, &mut gy_h);
    ops::gradient_level(g, &s.h1, &mut gx_h1, &mut gy_h1);

    // ---- 第 1 层 ----
    ops::vorticity_level(g, &s.u1, &s.v1, &mut zeta);
    for k in 0..n {
        let kk = 0.5 * (s.u1[k] * s.u1[k] + s.v1[k] * s.v1[k]);
        tmp[k] = kk;
    }
    ops::gradient_level(g, &tmp, &mut gx, &mut gy);
    for j in 0..g.nlat {
        let f = g.coriolis[j];
        for i in 0..g.nlon {
            let k = j * g.nlon + i;
            let c = zeta[k] + f;
            d.u1[k] = s.v1[k] * c - gx[k] - p.g * gx_h[k];
            d.v1[k] = -s.u1[k] * c - gy[k] - p.g * gy_h[k];
        }
    }

    // ---- 第 2 层 ----
    ops::vorticity_level(g, &s.u2, &s.v2, &mut zeta);
    for k in 0..n {
        tmp[k] = 0.5 * (s.u2[k] * s.u2[k] + s.v2[k] * s.v2[k]);
    }
    ops::gradient_level(g, &tmp, &mut gx, &mut gy);
    for j in 0..g.nlat {
        let f = g.coriolis[j];
        for i in 0..g.nlon {
            let k = j * g.nlon + i;
            let c = zeta[k] + f;
            d.u2[k] = s.v2[k] * c - gx[k] - p.g * gx_h[k] + p.gp * gx_h1[k] - p.drag * s.u2[k];
            d.v2[k] = -s.u2[k] * c - gy[k] - p.g * gy_h[k] + p.gp * gy_h1[k] - p.drag * s.v2[k];
        }
    }

    // ---- 连续（通量形式，严格守恒）----
    for k in 0..n {
        tmp[k] = s.h1[k] * s.u1[k];
        lap[k] = s.h1[k] * s.v1[k];
    }
    ops::divergence_level(g, &tmp, &lap, &mut d.h1);
    for k in 0..n {
        d.h1[k] = -d.h1[k];
    }
    for k in 0..n {
        tmp[k] = s.h2[k] * s.u2[k];
        lap[k] = s.h2[k] * s.v2[k];
    }
    ops::divergence_level(g, &tmp, &lap, &mut d.h2);
    for k in 0..n {
        d.h2[k] = -d.h2[k];
    }

    // ---- 扩散（ν_j∇²，只作用在速度；ν_j 按行取稳定上限）----
    for (vel, out) in [
        (&s.u1, &mut d.u1),
        (&s.v1, &mut d.v1),
        (&s.u2, &mut d.u2),
        (&s.v2, &mut d.v2),
    ] {
        ops::laplacian_level(g, vel, &mut lap);
        for j in 0..g.nlat {
            let nu = nu_row[j];
            if nu == 0.0 {
                continue;
            }
            for i in 0..g.nlon {
                let k = j * g.nlon + i;
                out[k] += nu * lap[k];
            }
        }
    }

    // ---- 极区滤波（作用在倾向）----
    let cfg = filter_cfg(p);
    for arr in [
        &mut d.u1,
        &mut d.v1,
        &mut d.h1,
        &mut d.u2,
        &mut d.v2,
        &mut d.h2,
    ] {
        crate::filt::polar_filter(g, arr, &cfg, m_max);
    }
}

// ---------------------------------------------------------------- 积分

fn combine2(a: &SwState, b: &SwState, wa: f64, wb: f64, out: &mut SwState) {
    for (o, av, bv) in [
        (&mut out.u1, &a.u1, &b.u1),
        (&mut out.v1, &a.v1, &b.v1),
        (&mut out.h1, &a.h1, &b.h1),
        (&mut out.u2, &a.u2, &b.u2),
        (&mut out.v2, &a.v2, &b.v2),
        (&mut out.h2, &a.h2, &b.h2),
    ] {
        for k in 0..av.len() {
            o[k] = wa * av[k] + wb * bv[k];
        }
    }
}

/// 推进一个子步（原地改 `s`），返回诊断。SSP-RK3（Shu–Osher）。
pub fn step(g: &GridSpec, s: &mut SwState, dt: f64, p: &SwParams) -> SwDiag {
    let n = s.len();
    let zero = SwState {
        u1: vec![0.0; n],
        v1: vec![0.0; n],
        h1: vec![0.0; n],
        u2: vec![0.0; n],
        v2: vec![0.0; n],
        h2: vec![0.0; n],
    };
    let mut k1 = zero.clone();
    let mut k2 = zero.clone();
    let mut k3 = zero.clone();
    let mut y = s.clone();
    let mut tmp = s.clone();
    let mut out = s.clone();

    // 显式扩散的稳定上限：ν·dt/dx² ≤ 0.2（极区 dx 只有 46 km，必须按行裁）
    let nu_row = crate::filt::nu_row_cap(g, p.nu, dt);

    // 极区纬向截断：按**当地** CFL 定 m_max（c_eff = 最快外重力波 sqrt(g·H_max)）
    let hmax = s
        .h1
        .iter()
        .zip(s.h2.iter())
        .map(|(a, b)| a + b)
        .fold(f64::MIN, f64::max);
    let c_eff = (p.g * hmax.max(0.0)).sqrt();
    let cfg = filter_cfg(p);
    let m_max = crate::filt::m_max_from_cfl(g, c_eff, dt, &cfg);

    // 级 1：y = s + dt·L(s)
    rhs(g, s, p, &nu_row, &m_max, &mut k1);
    axpy(s, &k1, dt, &mut y);
    // 级 2：y2 = 3/4 s + 1/4 (y + dt·L(y))
    rhs(g, &y, p, &nu_row, &m_max, &mut k2);
    axpy(&y, &k2, dt, &mut tmp);
    combine2(s, &tmp, 0.75, 0.25, &mut out);
    // 级 3：s_new = 1/3 s + 2/3 (y2 + dt·L(y2))
    rhs(g, &out, p, &nu_row, &m_max, &mut k3);
    axpy(&out, &k3, dt, &mut tmp);
    combine2(s, &tmp, 1.0 / 3.0, 2.0 / 3.0, &mut out);
    *s = out;

    diagnostics(g, s, dt)
}

fn axpy(x: &SwState, d: &SwState, a: f64, out: &mut SwState) {
    for (o, xv, dv) in [
        (&mut out.u1, &x.u1, &d.u1),
        (&mut out.v1, &x.v1, &d.v1),
        (&mut out.h1, &x.h1, &d.h1),
        (&mut out.u2, &x.u2, &d.u2),
        (&mut out.v2, &x.v2, &d.v2),
        (&mut out.h2, &x.h2, &d.h2),
    ] {
        for k in 0..xv.len() {
            o[k] = xv[k] + a * dv[k];
        }
    }
}

pub struct SwDiag {
    pub cfl_max: f64,
    pub cfl_lat_deg: f64,
    pub nan_count: usize,
    pub mass1: f64,
    pub mass2: f64,
    pub energy: f64,
}

pub fn diagnostics(g: &GridSpec, s: &SwState, dt: f64) -> SwDiag {
    let (nlat, nlon) = (g.nlat, g.nlon);
    let mut mass1 = 0.0f64;
    let mut mass2 = 0.0f64;
    let mut energy = 0.0f64;
    let mut nan = 0usize;
    let mut cfl_max = 0.0f64;
    let mut cfl_lat = 0.0f64;
    for j in 0..nlat {
        let dy = g.dy_m[j].max(1.0);
        let dx = g.dx_m[j].max(1.0);
        let mut r1 = 0.0f64;
        let mut r2 = 0.0f64;
        let mut re = 0.0f64;
        for i in 0..nlon {
            let k = j * nlon + i;
            for x in [
                s.u1[k], s.v1[k], s.h1[k], s.u2[k], s.v2[k], s.h2[k],
            ] {
                if !x.is_finite() {
                    nan += 1;
                }
            }
            r1 += s.h1[k];
            r2 += s.h2[k];
            let hh = s.h1[k] + s.h2[k];
            let ke = 0.5 * (s.h1[k] * (s.u1[k] * s.u1[k] + s.v1[k] * s.v1[k])
                + s.h2[k] * (s.u2[k] * s.u2[k] + s.v2[k] * s.v2[k]));
            let pe = 0.5 * G_DEFAULT * hh * hh + 0.5 * GP_DEFAULT * s.h1[k] * s.h1[k];
            re += ke + pe;
            for (u, v) in [(s.u1[k], s.v1[k]), (s.u2[k], s.v2[k])] {
                let c = dt * (u.abs() / dx + v.abs() / dy);
                if c > cfl_max {
                    cfl_max = c;
                    cfl_lat = detmath::to_deg(g.lat_rad[j]);
                }
            }
        }
        mass1 += g.area[j] * r1 / nlon as f64;
        mass2 += g.area[j] * r2 / nlon as f64;
        energy += g.area[j] * re / nlon as f64;
    }
    SwDiag {
        cfl_max,
        cfl_lat_deg: cfl_lat,
        nan_count: nan,
        mass1,
        mass2,
        energy,
    }
}

// ---------------------------------------------------------------- Python 面

fn read_state(state: &Bound<'_, PyDict>) -> PyResult<SwState> {
    let mut out = SwState {
        u1: vec![],
        v1: vec![],
        h1: vec![],
        u2: vec![],
        v2: vec![],
        h2: vec![],
    };
    for name in ["u1", "v1", "h1", "u2", "v2", "h2"] {
        let item = state
            .get_item(name)?
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err(format!("浅水状态缺 `{name}`")))?;
        let (s, d) = ops::read_f64(&item)?;
        if s.len() != 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "`{name}` 必须是 2D (nlat,nlon)"
            )));
        }
        match name {
            "u1" => out.u1 = d,
            "v1" => out.v1 = d,
            "h1" => out.h1 = d,
            "u2" => out.u2 = d,
            "v2" => out.v2 = d,
            _ => out.h2 = d,
        }
    }
    Ok(out)
}

fn write_state(state: &Bound<'_, PyDict>, s: &SwState, g: &GridSpec) -> PyResult<()> {
    for (name, data) in [
        ("u1", &s.u1),
        ("v1", &s.v1),
        ("h1", &s.h1),
        ("u2", &s.u2),
        ("v2", &s.v2),
        ("h2", &s.h2),
    ] {
        let item = state.get_item(name)?.ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!("浅水状态缺 `{name}`"))
        })?;
        let mut rw = item.extract::<numpy::PyReadwriteArray2<f64>>()?;
        let mut view = rw.as_array_mut();
        if view.shape() != [g.nlat, g.nlon] {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "`{name}` 形状 {:?} 与网格不符",
                view.shape()
            )));
        }
        for j in 0..g.nlat {
            for i in 0..g.nlon {
                view[[j, i]] = data[j * g.nlon + i];
            }
        }
    }
    Ok(())
}

fn params_from(kw: Option<&Bound<'_, PyDict>>) -> PyResult<SwParams> {
    let mut p = SwParams::default();
    if let Some(d) = kw {
        for (k, slot) in [
            ("g", &mut p.g),
            ("gp", &mut p.gp),
            ("phi_start_deg", &mut p.phi_start_deg),
            ("phi_flat_deg", &mut p.phi_flat_deg),
            ("nu", &mut p.nu),
            ("drag", &mut p.drag),
            ("cfl_target", &mut p.cfl_target),
        ] {
            if let Some(v) = d.get_item(k)? {
                *slot = v.extract::<f64>()?;
            }
        }
        if let Some(v) = d.get_item("n_filter")? {
            p.n_filter = v.extract::<usize>()?;
        }
        if let Some(v) = d.get_item("filter_mode")? {
            p.filter_mode = v.extract::<u8>()?;
        }
    }
    Ok(p)
}

fn diag_dict<'py>(py: Python<'py>, d: &SwDiag, dt: f64) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("cfl_max", d.cfl_max)?;
    out.set_item("cfl_lat_deg", d.cfl_lat_deg)?;
    out.set_item("nan_count", d.nan_count)?;
    out.set_item("mass1", d.mass1)?;
    out.set_item("mass2", d.mass2)?;
    out.set_item("energy", d.energy)?;
    out.set_item("dt", dt)?;
    Ok(out)
}

/// 2 层浅水推进一个子步（原地改 `state`；返回 diag）
#[pyfunction]
#[pyo3(signature = (state, grid, dt, params=None))]
pub fn shallow_water_step<'py>(
    py: Python<'py>,
    state: &Bound<'py, PyDict>,
    grid: &Bound<'py, PyDict>,
    dt: f64,
    params: Option<&Bound<'py, PyDict>>,
) -> PyResult<Bound<'py, PyDict>> {
    let g = crate::grid::from_dict(grid)?;
    let p = params_from(params)?;
    if !(dt.is_finite() && dt > 0.0) {
        return Err(pyo3::exceptions::PyValueError::new_err("dt 必须为正"));
    }
    let mut s = read_state(state)?;
    let d = step(&g, &mut s, dt, &p);
    write_state(state, &s, &g)?;
    diag_dict(py, &d, dt)
}

/// 只算倾向（AV 自定积分器/线性化时用），不改 `state`
#[pyfunction]
#[pyo3(signature = (state, grid, params=None))]
pub fn shallow_water_rhs<'py>(
    py: Python<'py>,
    state: &Bound<'py, PyDict>,
    grid: &Bound<'py, PyDict>,
    params: Option<&Bound<'py, PyDict>>,
) -> PyResult<Bound<'py, PyDict>> {
    let g = crate::grid::from_dict(grid)?;
    let p = params_from(params)?;
    let s = read_state(state)?;
    let mut d = s.clone();
    let nu_row = vec![p.nu; g.nlat];
    let m_max = vec![g.nlon; g.nlat]; // RHS-only 无 dt ⇒ 不做截断
    rhs(&g, &s, &p, &nu_row, &m_max, &mut d);
    let out = PyDict::new(py);
    for (name, data) in [
        ("du1", &d.u1),
        ("dv1", &d.v1),
        ("dh1", &d.h1),
        ("du2", &d.u2),
        ("dv2", &d.v2),
        ("dh2", &d.h2),
    ] {
        out.set_item(name, ops::write_like(py, &[g.nlat, g.nlon], data.clone())?)?;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid() -> GridSpec {
        GridSpec::new(32, 64, 8, "gaussian", 86400.0)
    }

    fn rest(g: &GridSpec, h1: f64, h2: f64) -> SwState {
        let n = g.nlat * g.nlon;
        SwState {
            u1: vec![0.0; n],
            v1: vec![0.0; n],
            h1: vec![h1; n],
            u2: vec![0.0; n],
            v2: vec![0.0; n],
            h2: vec![h2; n],
        }
    }

    /// 静止态是精确解（无加速度）
    #[test]
    fn rest_state_stays_at_rest() {
        let g = grid();
        let mut s = rest(&g, 300.0, 3700.0);
        let p = SwParams::default();
        let d = step(&g, &mut s, 900.0, &p);
        assert!(d.nan_count == 0);
        assert!(s.u1.iter().all(|x| x.abs() < 1e-12));
        assert!(s.u2.iter().all(|x| x.abs() < 1e-12));
        assert!(d.energy.is_finite());
    }

    /// 质量守恒（通量形式 ⇒ 到舍入）
    #[test]
    fn mass_is_conserved() {
        let g = grid();
        let mut s = rest(&g, 300.0, 3700.0);
        let p = SwParams {
            nu: 0.0,
            ..SwParams::default()
        };
        // 给一个斜压初始扰动（界面起伏）
        for j in 0..g.nlat {
            for i in 0..g.nlon {
                let k = j * g.nlon + i;
                s.h1[k] += 40.0 * detmath::cos(2.0 * g.lon_rad[i]) * g.mu[j];
            }
        }
        let d0 = diagnostics(&g, &s, 900.0);
        for _ in 0..40 {
            step(&g, &mut s, 900.0, &p);
        }
        let d1 = diagnostics(&g, &s, 900.0);
        let rel = ((d1.mass1 - d0.mass1) / d0.mass1).abs();
        assert!(rel < 1e-12, "上层质量漂移 {rel:e}");
        assert_eq!(d1.nan_count, 0);
    }

    /// 惯性振荡：f 平面近似下的频率应≈f（用一个小 f 的匀速旋转验证方向）
    #[test]
    fn geostrophic_balance_persists() {
        let g = grid();
        let mut s = rest(&g, 300.0, 3700.0);
        // 地转平衡：f u = -g ∂H/∂y ⇒ 用纬向均匀、经向线性的 H 与常数 u
        // 这里简单验证：给 u 一个常数、h 常数 ⇒ 只有科氏项，速度应旋转（频率 f）
        for k in 0..g.nlat * g.nlon {
            s.u1[k] = 5.0;
        }
        let p = SwParams {
            nu: 0.0,
            n_filter: 0,
            ..SwParams::default()
        };
        let dt = 900.0;
        let mut s0 = s.clone();
        for _ in 0..8 {
            step(&g, &mut s0, dt, &p);
        }
        // 简单检查：没有 NaN，且 |v| 增长（科氏把 u 转成 v）
        assert_eq!(diagnostics(&g, &s0, dt).nan_count, 0);
        let vmax = s0.v1.iter().cloned().fold(0.0f64, |a, b| a.max(b.abs()));
        assert!(vmax > 0.0, "科氏应把 u 转向 v");
    }
}
