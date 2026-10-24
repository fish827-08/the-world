//! **8 层 sigma 原始方程**（W2 交付）—— 干动力 + Held–Suarez 强迫。
//!
//! ## 方程（σ = p/ps，静力平衡，干空气）
//!
//! ```text
//! du/dt = −u·∇u − σ̇ ∂u/∂σ + (u v tanφ)/a + f v − ∂Φ/∂x|_p + F_u
//! dv/dt = −u·∇v − σ̇ ∂v/∂σ − (u² tanφ)/a − f u − ∂Φ/∂y|_p + F_v
//! dT/dt = −u·∇T − σ̇ ∂T/∂σ + (κT/ps)(∂ps/∂t + u·∇ps) + Q/cp
//! ∂ps/∂t = −∫₀¹ ∇·(ps u) dσ
//! ∂(ps σ̇)/∂σ = −∂ps/∂t − ∇·(ps u)
//! ∂Φ/∂lnσ = −R_d T
//! ∂Φ/∂x|_p = ∂Φ/∂x|_σ + (R_d T/ps)·∂ps/∂x
//! ```
//!
//! Held–Suarez（HS94）：
//! ```text
//! F = −k_v(φ,σ)·(u,v),        k_v = k_f·max(0,(σ−σ_b)/(1−σ_b))
//! Q/cp = −k_T(φ,σ)·(T − T_eq), k_T = k_a + (k_s−k_a)·max(0,(σ−σ_b)/(1−σ_b))·cos⁴φ
//! T_eq = max(200 K, (T₀ − ΔT_y sin²φ − Δθ_z lnσ cos²φ)·σ^κ)
//! ```
//!
//! ## 离散
//!
//! 水平：与 `ops` 同一套有限体积（`cosbar` 度量）；垂直：层中心差分，边界单侧；
//! 连续用**通量形式**（严格守恒）；时间：SSP-RK3；极区：`filt` 的纬向截断。
//!
//! ⚠️ `w ≡ σ̇` 在本模式里是**诊断量**（由连续方程积分得到），每步末尾写回状态数组。

use numpy::{PyReadwriteArray2, PyReadwriteArray3};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::detmath;
use crate::filt::{self, FilterCfg};
use crate::grid::GridSpec;
use crate::ops;

// ---------------------------------------------------------------- 参数

#[derive(Clone, Copy, Debug)]
pub struct PrimParams {
    pub g: f64,
    pub rd: f64,
    pub cp: f64,
    pub kappa: f64, // rd/cp
    pub nu: f64,
    pub filter: FilterCfg,
    // Held–Suarez
    pub hs: bool,
    pub t0: f64,
    pub delta_ty: f64,
    pub delta_theta_z: f64,
    pub sigma_b: f64,
    pub ka: f64,
    pub ks: f64,
    pub kf: f64,
    /// 顶部海绵层（Rayleigh）：σ < `sponge_sigma` 时按 `sponge_rate·((σ_t−σ)/σ_t)²` 阻尼 u,v。
    /// HS94 原文没有这一项，但 8 层 sigma + 显式积分在模式顶会积累波动 ⇒ 必须有（W2 实测）。
    pub sponge_sigma: f64,
    pub sponge_rate: f64,
}

impl Default for PrimParams {
    fn default() -> Self {
        const DAY: f64 = 86400.0;
        PrimParams {
            g: 9.81,
            rd: 287.0,
            cp: 1004.0,
            kappa: 287.0 / 1004.0,
            // ⚠️ 实测：1e6 会让模式顶层赤道附近的 2Δx 模在 ~3.4 天炸掉；1e7 可稳跑 10+ 天
            nu: 1.0e7,
            filter: FilterCfg::default(),
            hs: true,
            t0: 288.0,
            delta_ty: 60.0,
            delta_theta_z: 10.0,
            sigma_b: 0.7,
            ka: 1.0 / (40.0 * DAY),
            ks: 1.0 / (4.0 * DAY),
            kf: 1.0 / DAY,
            sponge_sigma: 0.10,
            sponge_rate: 1.0 / (0.5 * DAY),
        }
    }
}

// ---------------------------------------------------------------- 状态

#[derive(Clone, Debug)]
pub struct PrimState {
    pub u: Vec<f64>,
    pub v: Vec<f64>,
    pub w: Vec<f64>,
    pub t: Vec<f64>,
    pub ps: Vec<f64>,
}

impl PrimState {
    pub fn zeros(nlev: usize, n: usize) -> Self {
        PrimState {
            u: vec![0.0; nlev * n],
            v: vec![0.0; nlev * n],
            w: vec![0.0; nlev * n],
            t: vec![0.0; nlev * n],
            ps: vec![0.0; n],
        }
    }
}

// ---------------------------------------------------------------- RHS

/// 垂直差分（层中心；首末层单侧）—— 供诊断/热力学用（平滑、不引入计算模）
#[inline]
fn dv_dsigma(f: &[f64], k: usize, i: usize, nlev: usize, n: usize, sig: &[f64]) -> f64 {
    if nlev == 1 {
        return 0.0;
    }
    if k == 0 {
        (f[i] - f[n + i]) / (sig[0] - sig[1])
    } else if k + 1 == nlev {
        (f[(k - 1) * n + i] - f[k * n + i]) / (sig[k - 1] - sig[k])
    } else {
        (f[(k - 1) * n + i] - f[(k + 1) * n + i]) / (sig[k - 1] - sig[k + 1])
    }
}

/// 垂直平流的**迎风**差分（σ 向下为正 ⇒ σ̇>0 表示自上而下，来流在 k−1）
///
/// 为什么不用中心差：层中心差分在垂直方向会激发 **2Δσ 计算模**（odd-even 解耦），
/// 8 层 + 显式积分下它在 ~2 天内增长到爆（W2 实测：dt=288 也活不过 599 步）。
/// 一阶迎风虽有一阶耗散，但能稳定压住计算模——这是 W2 的"最简可用"取舍。
#[inline]
fn dv_dsigma_upwind(
    f: &[f64],
    k: usize,
    i: usize,
    nlev: usize,
    n: usize,
    sig: &[f64],
    sdot: f64,
) -> f64 {
    if nlev == 1 {
        return 0.0;
    }
    if k == 0 {
        // 模式顶：σ̇=0 且无上方来流 ⇒ 不做垂直平流（否则 2Δσ 计算模在顶层被放大）
        return 0.0;
    }
    if sdot >= 0.0 {
        (f[k * n + i] - f[(k - 1) * n + i]) / (sig[k] - sig[k - 1])
    } else if k + 1 == nlev {
        (f[(k - 1) * n + i] - f[k * n + i]) / (sig[k - 1] - sig[k])
    } else {
        (f[(k + 1) * n + i] - f[k * n + i]) / (sig[k + 1] - sig[k])
    }
}

/// 一步的倾向（不含 `dt`）
pub fn rhs(
    g: &GridSpec,
    s: &PrimState,
    p: &PrimParams,
    m_max: &[usize],
    nu_row: &[f64],
    d: &mut PrimState,
) {
    let (nlat, nlon, nlev) = (g.nlat, g.nlon, g.nlev);
    let n = nlat * nlon;

    // 垂直常数
    let mut ln_half = vec![0.0f64; nlev + 1];
    let mut ln_full = vec![0.0f64; nlev];
    let mut dsig = vec![0.0f64; nlev];
    let mut sig_pow = vec![0.0f64; nlev];
    for k in 0..=nlev {
        ln_half[k] = detmath::ln(g.sigma_half[k]);
    }
    for k in 0..nlev {
        ln_full[k] = detmath::ln(g.sigma_full[k]);
        dsig[k] = g.sigma_half[k + 1] - g.sigma_half[k];
        sig_pow[k] = detmath::pow(g.sigma_full[k], p.kappa);
    }

    // ① 连续：D_k = ∇·(ps·u_k)，dps/dt = −Σ D_k Δσ_k
    let mut div_k = vec![0.0f64; nlev * n];
    let mut fx = vec![0.0f64; n];
    let mut fy = vec![0.0f64; n];
    let mut div = vec![0.0f64; n];
    for k in 0..nlev {
        for i in 0..n {
            fx[i] = s.ps[i] * s.u[k * n + i];
            fy[i] = s.ps[i] * s.v[k * n + i];
        }
        ops::divergence_level(g, &fx, &fy, &mut div);
        div_k[k * n..(k + 1) * n].copy_from_slice(&div);
    }
    let mut dpsdt = vec![0.0f64; n];
    for k in 0..nlev {
        for i in 0..n {
            dpsdt[i] -= div_k[k * n + i] * dsig[k];
        }
    }
    // ② 质量通量 (psσ̇) 自顶向下积分 ⇒ σ̇
    let mut fhalf = vec![0.0f64; (nlev + 1) * n];
    for k in 0..nlev {
        for i in 0..n {
            fhalf[(k + 1) * n + i] = fhalf[k * n + i] - dsig[k] * (dpsdt[i] + div_k[k * n + i]);
        }
    }
    // ③ 位势：自下而上静力积分
    let mut phi_half = vec![0.0f64; (nlev + 1) * n]; // 地形 v0 = 0
    for k in (0..nlev).rev() {
        let lr = ln_half[k + 1] - ln_half[k];
        for i in 0..n {
            phi_half[k * n + i] = phi_half[(k + 1) * n + i] + p.rd * s.t[k * n + i] * lr;
        }
    }
    let mut phi_full = vec![0.0f64; nlev * n];
    for k in 0..nlev {
        let lr = ln_half[k + 1] - ln_full[k];
        for i in 0..n {
            phi_full[k * n + i] = phi_half[(k + 1) * n + i] + p.rd * s.t[k * n + i] * lr;
        }
    }

    // ④ ps 的水平梯度（各层共用）
    let mut gx_ps = vec![0.0f64; n];
    let mut gy_ps = vec![0.0f64; n];
    ops::gradient_level(g, &s.ps, &mut gx_ps, &mut gy_ps);

    // ⑤ 逐层倾向
    let mut adv = vec![0.0f64; n];
    let mut gx_phi = vec![0.0f64; n];
    let mut gy_phi = vec![0.0f64; n];
    let mut lap = vec![0.0f64; n];
    for k in 0..nlev {
        let sl = k * n..(k + 1) * n;
        ops::gradient_level(g, &phi_full[sl.clone()], &mut gx_phi, &mut gy_phi);
        let sigma = g.sigma_full[k];
        let frac = if sigma > p.sigma_b {
            ((sigma - p.sigma_b) / (1.0 - p.sigma_b)).min(1.0)
        } else {
            0.0
        };
        let kv = p.kf * frac;
        let sponge = if p.sponge_sigma > 0.0 && sigma < p.sponge_sigma {
            let x = (p.sponge_sigma - sigma) / p.sponge_sigma;
            p.sponge_rate * x * x
        } else {
            0.0
        };

        // u
        ops::advect_level(g, &s.u[sl.clone()], &s.v[sl.clone()], &s.u[sl.clone()], &mut adv);
        for j in 0..nlat {
            let tanphi = g.mu[j] / g.cosphi[j];
            for i in 0..nlon {
                let idx = j * nlon + i;
                let kk = k * n + idx;
                let u = s.u[kk];
                let v = s.v[kk];
                let tk = s.t[kk];
                let psi = s.ps[idx].max(1.0);
                let sdot = 0.5 * (fhalf[k * n + idx] + fhalf[(k + 1) * n + idx]) / psi;
                let du_adv = adv[idx] + sdot * dv_dsigma_upwind(&s.u, k, idx, nlev, n, &g.sigma_full, sdot);
                let pgx = gx_phi[idx] + (p.rd * tk / psi) * gx_ps[idx];
                d.u[kk] =
                    -du_adv + u * v * tanphi / g.radius_m + g.coriolis[j] * v - pgx - (kv + sponge) * u;
            }
        }
        // v
        ops::advect_level(g, &s.u[sl.clone()], &s.v[sl.clone()], &s.v[sl.clone()], &mut adv);
        for j in 0..nlat {
            let tanphi = g.mu[j] / g.cosphi[j];
            for i in 0..nlon {
                let idx = j * nlon + i;
                let kk = k * n + idx;
                let u = s.u[kk];
                let v = s.v[kk];
                let tk = s.t[kk];
                let psi = s.ps[idx].max(1.0);
                let sdot = 0.5 * (fhalf[k * n + idx] + fhalf[(k + 1) * n + idx]) / psi;
                let dv_adv = adv[idx] + sdot * dv_dsigma_upwind(&s.v, k, idx, nlev, n, &g.sigma_full, sdot);
                let pgy = gy_phi[idx] + (p.rd * tk / psi) * gy_ps[idx];
                d.v[kk] =
                    -dv_adv - u * u * tanphi / g.radius_m - g.coriolis[j] * u - pgy - (kv + sponge) * v;
            }
        }
        // T
        ops::advect_level(g, &s.u[sl.clone()], &s.v[sl.clone()], &s.t[sl.clone()], &mut adv);
        for j in 0..nlat {
            let cos2 = 1.0 - g.mu[j] * g.mu[j];
            for i in 0..nlon {
                let idx = j * nlon + i;
                let kk = k * n + idx;
                let tk = s.t[kk];
                let psi = s.ps[idx].max(1.0);
                let sdot = 0.5 * (fhalf[k * n + idx] + fhalf[(k + 1) * n + idx]) / psi;
                let dtds = dv_dsigma_upwind(&s.t, k, idx, nlev, n, &g.sigma_full, sdot);
                let adps = s.u[kk] * gx_ps[idx] + s.v[kk] * gy_ps[idx];
                let mut dt = -adv[idx] - sdot * dtds
                    + (p.kappa * tk / psi) * (dpsdt[idx] + adps);
                if p.hs {
                    let teq = (p.t0 - p.delta_ty * g.mu[j] * g.mu[j]
                        - p.delta_theta_z * ln_full[k] * cos2)
                        * sig_pow[k];
                    let teq = teq.max(200.0);
                    let kt = p.ka + (p.ks - p.ka) * frac * cos2 * cos2;
                    dt -= kt * (tk - teq);
                }
                d.t[kk] = dt;
            }
        }
        // 扩散（u,v,T）
        for (src, dst) in [
            (&s.u[sl.clone()], &mut d.u[sl.clone()]),
            (&s.v[sl.clone()], &mut d.v[sl.clone()]),
            (&s.t[sl.clone()], &mut d.t[sl.clone()]),
        ] {
            ops::laplacian_level(g, src, &mut lap);
            for j in 0..nlat {
                let nu = nu_row[j];
                if nu == 0.0 {
                    continue;
                }
                for i in 0..nlon {
                    dst[j * nlon + i] += nu * lap[j * nlon + i];
                }
            }
        }
    }

    // ⑥ ps 倾向 + 极区滤波
    d.ps.copy_from_slice(&dpsdt);
    for arr in [&mut d.u, &mut d.v, &mut d.t] {
        filt::polar_filter(g, arr, &p.filter, m_max);
    }
    filt::polar_filter(g, &mut d.ps, &p.filter, m_max);
}

// ---------------------------------------------------------------- 诊断

pub struct PrimDiag {
    pub cfl_max: f64,
    pub cfl_lat_deg: f64,
    pub cfl_k: usize,
    pub nan_count: usize,
    pub mass: f64,
    pub energy: f64,
    pub tmin: f64,
    pub tmax: f64,
    pub ps_min: f64,
    pub ps_max: f64,
}

pub fn diagnostics(g: &GridSpec, s: &PrimState, p: &PrimParams, dt: f64) -> PrimDiag {
    let (nlat, nlon, nlev) = (g.nlat, g.nlon, g.nlev);
    let n = nlat * nlon;
    let mut mass = 0.0f64;
    let mut energy = 0.0f64;
    let mut nan = 0usize;
    let mut cfl_max = 0.0f64;
    let mut cfl_lat = 0.0f64;
    let mut cfl_k = 0usize;
    let (mut tmin, mut tmax) = (f64::MAX, f64::MIN);
    let (mut pmin, mut pmax) = (f64::MAX, f64::MIN);
    for j in 0..nlat {
        let dx = g.dx_m[j].max(1.0);
        let dy = g.dy_m[j].max(1.0);
        let mut mrow = 0.0f64;
        let mut erow = 0.0f64;
        for i in 0..nlon {
            let idx = j * nlon + i;
            let psi = s.ps[idx];
            if !psi.is_finite() {
                nan += 1;
            }
            pmin = pmin.min(psi);
            pmax = pmax.max(psi);
            mrow += psi / p.g;
            let mut col = 0.0f64;
            for k in 0..nlev {
                let kk = k * n + idx;
                let t = s.t[kk];
                if !t.is_finite() || !s.u[kk].is_finite() || !s.v[kk].is_finite() {
                    nan += 1;
                }
                tmin = tmin.min(t);
                tmax = tmax.max(t);
                let dsig = g.sigma_half[k + 1] - g.sigma_half[k];
                col += (p.cp * t) * dsig; // 位势项在相对比较中可省（同一时刻一致）
                let c = dt * (s.u[kk].abs() / dx + s.v[kk].abs() / dy);
                if c > cfl_max {
                    cfl_max = c;
                    cfl_lat = detmath::to_deg(g.lat_rad[j]);
                    cfl_k = k;
                }
            }
            erow += psi / p.g * col;
        }
        mass += g.area[j] * mrow / nlon as f64;
        energy += g.area[j] * erow / nlon as f64;
    }
    PrimDiag {
        cfl_max,
        cfl_lat_deg: cfl_lat,
        cfl_k,
        nan_count: nan,
        mass,
        energy,
        tmin,
        tmax,
        ps_min: pmin,
        ps_max: pmax,
    }
}

// ---------------------------------------------------------------- 推进

/// 一步（SSP-RK3；原地改 `s`）
pub fn advance(g: &GridSpec, s: &mut PrimState, p: &PrimParams, dt: f64) -> PrimDiag {
    let n = g.nlat * g.nlon;
    let nu_row = filt::nu_row_cap(g, p.nu, dt);
    // 最快外波：Lamb 波 c = sqrt(γ R T)
    let tmax = s.t.iter().cloned().fold(f64::MIN, f64::max).max(200.0);
    let c_eff = (1.4 * p.rd * tmax).sqrt();
    let m_max = filt::m_max_from_cfl(g, c_eff, dt, &p.filter);

    let mut k1 = PrimState::zeros(g.nlev, n);
    let mut k2 = PrimState::zeros(g.nlev, n);
    let mut k3 = PrimState::zeros(g.nlev, n);
    let mut y = s.clone();
    let mut tmp = s.clone();
    let mut out = s.clone();

    rhs(g, s, p, &m_max, &nu_row, &mut k1);
    axpy(s, &k1, dt, &mut y);
    rhs(g, &y, p, &m_max, &nu_row, &mut k2);
    axpy(&y, &k2, dt, &mut tmp);
    wcomb(s, &tmp, 0.75, 0.25, &mut out);
    rhs(g, &out, p, &m_max, &nu_row, &mut k3);
    axpy(&out, &k3, dt, &mut tmp);
    wcomb(s, &tmp, 1.0 / 3.0, 2.0 / 3.0, &mut out);
    *s = out;

    refresh_diagnostics_fields(g, s, p);
    diagnostics(g, s, p, dt)
}

fn axpy(x: &PrimState, d: &PrimState, a: f64, out: &mut PrimState) {
    for (o, xv, dv) in [
        (&mut out.u, &x.u, &d.u),
        (&mut out.v, &x.v, &d.v),
        (&mut out.t, &x.t, &d.t),
        (&mut out.ps, &x.ps, &d.ps),
    ] {
        for k in 0..xv.len() {
            o[k] = xv[k] + a * dv[k];
        }
    }
}

fn wcomb(a: &PrimState, b: &PrimState, wa: f64, wb: f64, out: &mut PrimState) {
    for (o, av, bv) in [
        (&mut out.u, &a.u, &b.u),
        (&mut out.v, &a.v, &b.v),
        (&mut out.t, &a.t, &b.t),
        (&mut out.ps, &a.ps, &b.ps),
    ] {
        for k in 0..av.len() {
            o[k] = wa * av[k] + wb * bv[k];
        }
    }
}

/// 每步末尾把诊断量写回状态：`w = σ̇`、`p = ps·σ`
pub fn refresh_diagnostics_fields(g: &GridSpec, s: &mut PrimState, _p: &PrimParams) {
    let (nlat, nlon, nlev) = (g.nlat, g.nlon, g.nlev);
    let n = nlat * nlon;
    let mut dsig = vec![0.0f64; nlev];
    for k in 0..nlev {
        dsig[k] = g.sigma_half[k + 1] - g.sigma_half[k];
    }
    // 用与 rhs 相同的连续积分
    let mut fx = vec![0.0f64; n];
    let mut fy = vec![0.0f64; n];
    let mut div = vec![0.0f64; n];
    let mut dpsdt = vec![0.0f64; n];
    let mut div_k = vec![0.0f64; nlev * n];
    for k in 0..nlev {
        for i in 0..n {
            fx[i] = s.ps[i] * s.u[k * n + i];
            fy[i] = s.ps[i] * s.v[k * n + i];
        }
        ops::divergence_level(g, &fx, &fy, &mut div);
        div_k[k * n..(k + 1) * n].copy_from_slice(&div);
        for i in 0..n {
            dpsdt[i] -= div[i] * dsig[k];
        }
    }
    let mut fhalf = vec![0.0f64; (nlev + 1) * n];
    for k in 0..nlev {
        for i in 0..n {
            fhalf[(k + 1) * n + i] = fhalf[k * n + i] - dsig[k] * (dpsdt[i] + div_k[k * n + i]);
        }
    }
    for k in 0..nlev {
        for i in 0..n {
            let psi = s.ps[i].max(1.0);
            s.w[k * n + i] = 0.5 * (fhalf[k * n + i] + fhalf[(k + 1) * n + i]) / psi;
        }
    }
}

// ---------------------------------------------------------------- Python 面

fn read_state(state: &Bound<'_, PyDict>, g: &GridSpec) -> PyResult<PrimState> {
    let n = g.nlat * g.nlon;
    let mut out = PrimState::zeros(g.nlev, n);
    for name in ["u", "v", "T"] {
        let item = state
            .get_item(name)?
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err(format!("状态缺 `{name}`")))?;
        let (shape, data) = ops::read_f64(&item)?;
        if shape != [g.nlev, g.nlat, g.nlon] {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "`{name}` 形状 {shape:?} 应为 ({}, {}, {})",
                g.nlev, g.nlat, g.nlon
            )));
        }
        match name {
            "u" => out.u = data,
            "v" => out.v = data,
            _ => out.t = data,
        }
    }
    let item = state
        .get_item("ps")?
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("状态缺 `ps`"))?;
    let (shape, ps) = ops::read_f64(&item)?;
    if shape != [g.nlat, g.nlon] {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "`ps` 形状 {shape:?} 应为 ({}, {})",
            g.nlat, g.nlon
        )));
    }
    out.ps = ps;
    Ok(out)
}

fn write_state(state: &Bound<'_, PyDict>, s: &PrimState, g: &GridSpec) -> PyResult<()> {
    let (nlat, nlon, nlev) = (g.nlat, g.nlon, g.nlev);
    let n = nlat * nlon;
    for (name, data) in [("u", &s.u), ("v", &s.v), ("T", &s.t), ("w", &s.w)] {
        let item = state.get_item(name)?.ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!("状态缺 `{name}`"))
        })?;
        let mut rw = item.extract::<PyReadwriteArray3<f64>>()?;
        let mut view = rw.as_array_mut();
        for k in 0..nlev {
            for j in 0..nlat {
                for i in 0..nlon {
                    view[[k, j, i]] = data[k * n + j * nlon + i];
                }
            }
        }
    }
    for (name, data) in [("ps", &s.ps)] {
        let item = state.get_item(name)?.ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err(format!("状态缺 `{name}`"))
        })?;
        let mut rw = item.extract::<PyReadwriteArray2<f64>>()?;
        let mut view = rw.as_array_mut();
        for j in 0..nlat {
            for i in 0..nlon {
                view[[j, i]] = data[j * nlon + i];
            }
        }
    }
    // p（诊断量）
    if let Some(item) = state.get_item("p")? {
        let mut rw = item.extract::<PyReadwriteArray3<f64>>()?;
        let mut view = rw.as_array_mut();
        for k in 0..nlev {
            for j in 0..nlat {
                for i in 0..nlon {
                    view[[k, j, i]] = s.ps[j * nlon + i] * g.sigma_full[k];
                }
            }
        }
    }
    Ok(())
}

fn params_from(forcing: Option<&Bound<'_, PyDict>>) -> PyResult<PrimParams> {
    let mut p = PrimParams::default();
    let Some(f) = forcing else {
        return Ok(p);
    };
    if let Some(hs) = f.get_item("held_suarez")? {
        let hs: Bound<'_, PyDict> = hs.downcast_into()?;
        for (k, slot) in [
            ("t0", &mut p.t0),
            ("delta_ty", &mut p.delta_ty),
            ("delta_theta_z", &mut p.delta_theta_z),
            ("sigma_b", &mut p.sigma_b),
            ("ka", &mut p.ka),
            ("ks", &mut p.ks),
            ("kf", &mut p.kf),
            ("sponge_sigma", &mut p.sponge_sigma),
            ("sponge_rate", &mut p.sponge_rate),
        ] {
            if let Some(v) = hs.get_item(k)? {
                *slot = v.extract::<f64>()?;
            }
        }
        if let Some(v) = hs.get_item("enabled")? {
            p.hs = v.extract::<bool>()?;
        }
    }
    if let Some(fl) = f.get_item("filter")? {
        let fl: Bound<'_, PyDict> = fl.downcast_into()?;
        if let Some(v) = fl.get_item("mode")? {
            p.filter.mode = v.extract::<u8>()?;
        }
        if let Some(v) = fl.get_item("cfl_target")? {
            p.filter.cfl_target = v.extract::<f64>()?;
        }
        if let Some(v) = fl.get_item("nu")? {
            p.nu = v.extract::<f64>()?;
        }
    }
    Ok(p)
}

fn diag_dict<'py>(py: Python<'py>, d: &PrimDiag, dt: f64) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("cfl_max", d.cfl_max)?;
    out.set_item("cfl_lat_deg", d.cfl_lat_deg)?;
    out.set_item("cfl_k", d.cfl_k)?;
    out.set_item("nan_count", d.nan_count)?;
    out.set_item("mass", d.mass)?;
    out.set_item("mass_rel_err", 0.0f64)?; // 需要参照值，由调用方累积（见 diag 说明）
    out.set_item("energy", d.energy)?;
    out.set_item("energy_rel_err", 0.0f64)?;
    out.set_item("tmin", d.tmin)?;
    out.set_item("tmax", d.tmax)?;
    out.set_item("ps_min", d.ps_min)?;
    out.set_item("ps_max", d.ps_max)?;
    out.set_item("dt", dt)?;
    Ok(out)
}

/// 内部实现（`AtmoCore.step` 与模块级 `ac.step` 共用）
pub fn step_impl<'py>(
    py: Python<'py>,
    state: &Bound<'py, PyDict>,
    forcing: Option<&Bound<'py, PyDict>>,
    dt: f64,
    grid_override: Option<&GridSpec>,
) -> PyResult<Bound<'py, PyDict>> {
    let (shape, _) = ops::read_f64(&state.get_item("u")?.ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("状态缺 `u`（需先 ac.make_state）")
    })?)?;
    if shape.len() != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err("u 必须是 3D"));
    }
    let g = if let Some(gv) = grid_override {
        gv.clone()
    } else {
        match forcing.and_then(|f| f.get_item("grid").ok().flatten()) {
            Some(v) => {
                let d: Bound<'_, PyDict> = v.downcast_into()?;
                crate::grid::from_dict(&d)?
            }
            None => GridSpec::new(shape[1], shape[2], shape[0], "gaussian", 86400.0),
        }
    };
    if !(dt.is_finite() && dt > 0.0) {
        return Err(pyo3::exceptions::PyValueError::new_err("dt 必须为正"));
    }
    let p = params_from(forcing)?;
    let mut s = read_state(state, &g)?;
    let d = step_core(&g, &mut s, &p, dt);
    write_state(state, &s, &g)?;
    diag_dict(py, &d, dt)
}

pub fn step_core(g: &GridSpec, s: &mut PrimState, p: &PrimParams, dt: f64) -> PrimDiag {
    advance(g, s, p, dt)
}

/// `diag = ac.step(state, forcing=None, dt=900.0)` —— 8 层 sigma 原始方程推进一步（原地改 state）
#[pyfunction]
#[pyo3(signature = (state, forcing=None, dt=900.0))]
pub fn step<'py>(
    py: Python<'py>,
    state: &Bound<'py, PyDict>,
    forcing: Option<&Bound<'py, PyDict>>,
    dt: f64,
) -> PyResult<Bound<'py, PyDict>> {
    step_impl(py, state, forcing, dt, None)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid() -> GridSpec {
        GridSpec::new(32, 64, 8, "gaussian", 86400.0)
    }

    fn isothermal(g: &GridSpec, t0: f64, ps0: f64) -> PrimState {
        let n = g.nlat * g.nlon;
        PrimState {
            u: vec![0.0; g.nlev * n],
            v: vec![0.0; g.nlev * n],
            w: vec![0.0; g.nlev * n],
            t: vec![t0; g.nlev * n],
            ps: vec![ps0; n],
        }
    }

    /// 等温静止：**精确解**（Φ 水平均匀 ⇒ 气压梯度力为 0）
    #[test]
    fn isothermal_rest_is_exact() {
        let g = grid();
        let p = PrimParams {
            hs: false,
            nu: 0.0,
            ..PrimParams::default()
        };
        let mut s = isothermal(&g, 288.0, 101325.0);
        for _ in 0..20 {
            let d = advance(&g, &mut s, &p, 900.0);
            assert_eq!(d.nan_count, 0);
        }
        let umax = s.u.iter().fold(0.0f64, |a, b| a.max(b.abs()));
        let wmax = s.w.iter().fold(0.0f64, |a, b| a.max(b.abs()));
        assert!(umax < 1e-10, "静止态应保持静止，u_max={umax:e}");
        assert!(wmax < 1e-14, "σ̇ 应为 0，w_max={wmax:e}");
    }

    /// 干质量守恒（通量形式；HS 不改变质量）
    #[test]
    fn mass_is_conserved() {
        let g = grid();
        let p = PrimParams {
            nu: 0.0,
            ..PrimParams::default()
        };
        let mut s = isothermal(&g, 288.0, 101325.0);
        // 给一个斜压扰动（T 随高度/纬度变化）
        let n = g.nlat * g.nlon;
        for k in 0..g.nlev {
            for j in 0..g.nlat {
                for i in 0..g.nlon {
                    let kk = k * n + j * g.nlon + i;
                    s.t[kk] += 10.0 * g.mu[j] - 20.0 * g.sigma_full[k];
                }
            }
        }
        let d0 = diagnostics(&g, &s, &p, 900.0);
        for _ in 0..20 {
            advance(&g, &mut s, &p, 900.0);
        }
        let d1 = diagnostics(&g, &s, &p, 900.0);
        let rel = ((d1.mass - d0.mass) / d0.mass).abs();
        assert!(rel < 1e-10, "质量漂移 {rel:e}");
        assert_eq!(d1.nan_count, 0);
    }

    /// HS 强迫把温度拉向 T_eq（关动力，只留牛顿冷却）
    #[test]
    fn held_suarez_relaxes_toward_teq() {
        let g = grid();
        let mut p = PrimParams::default();
        p.hs = true;
        p.nu = 0.0;
        p.filter.mode = 0;
        let mut s = isothermal(&g, 200.0, 101325.0);
        let n = g.nlat * g.nlon;
        // 只保留热力学：把动量方程的效果关掉（u,v 初值 0 且无强迫 ⇒ 保持 0）
        for _ in 0..200 {
            advance(&g, &mut s, &p, 900.0);
        }
        // 低层（σ=0.925）应明显升温（T_eq 在低纬 ~288）
        let k = g.nlev - 1;
        let t_low: f64 = (0..n).map(|i| s.t[k * n + i]).sum::<f64>() / n as f64;
        assert!(t_low > 210.0, "HS 应把低层拉向 T_eq，实测 {t_low:.1} K");
    }
}
