//! 太阳几何（轨道/自转 → 日下点 → 大气顶短波）—— **W1 交付**。
//!
//! 契约（`atmo/INTERFACE.md` §1/§8.5）：
//! * 输入只有**轨道参数**与**时间**，输出 `cosZ` 与 `toa_sw`；
//! * **纯解析，禁止查表**（AV 有写死审计门）；
//! * 全部三角函数走 `crate::detmath`（跨平台逐位一致）。
//!
//! ## 公式（写死，改动即作废 AV 判据）
//!
//! ```text
//! L  = 2π·frac((t − t_eq)/year)        黄经（0 = 北半球春分）
//! sinδ = sin ε · sin L                  赤纬
//! ν  = L − ϖ                            真近点角（一阶近似）
//! (a/r)² = ((1 + e·cos ν)/(1 − e²))²
//! H  = 2π·frac(t/T_rot) + λ             时角（t=0 时 0°E 正午）
//! cosZ = max(0, sinφ sinδ + cosφ cosδ cos H)
//! toa  = S0 · (a/r)² · cosZ
//! ```
//!
//! 精度：忽略中心差（用 L 代替 ν）⇒ 日地距离误差 O(e²) ≈ 3e-4 ⇒ 通量误差 < 0.1%。

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::detmath::{self, PI};
use crate::grid::GridSpec;

/// 轨道/自转参数（默认值 = 地球；世界侧由 A3 从 `SimConfig` 透传）
#[derive(Clone, Debug)]
pub struct OrbitConfig {
    pub year_s: f64,
    pub rotation_period_s: f64,
    pub obliquity_rad: f64,
    pub eccentricity: f64,
    pub equinox_offset_s: f64,
    pub solar_constant_w_m2: f64,
    pub radius_m: f64,
    /// 近日点黄经（可选键；默认 = 地球 ~283°）。契约 §9.3 未列，属**追加可选键**。
    pub perihelion_lon_rad: f64,
}

impl Default for OrbitConfig {
    fn default() -> Self {
        OrbitConfig {
            year_s: 365.25 * 86400.0,
            rotation_period_s: 86400.0,
            obliquity_rad: 0.409_105_18,
            eccentricity: 0.0167,
            equinox_offset_s: 0.0,
            solar_constant_w_m2: 1361.0,
            radius_m: 6_371_000.0,
            perihelion_lon_rad: 4.938_4, // 283°（1 月 3 日近日点）
        }
    }
}

impl OrbitConfig {
    pub fn from_dict(d: &Bound<'_, PyDict>) -> PyResult<Self> {
        let mut o = OrbitConfig::default();
        let keys: [&str; 8] = [
            "year_s",
            "rotation_period_s",
            "obliquity_rad",
            "eccentricity",
            "equinox_offset_s",
            "solar_constant_w_m2",
            "radius_m",
            "perihelion_lon_rad",
        ];
        for k in keys {
            if let Some(v) = d.get_item(k)? {
                let x = v.extract::<f64>()?;
                match k {
                    "year_s" => o.year_s = x,
                    "rotation_period_s" => o.rotation_period_s = x,
                    "obliquity_rad" => o.obliquity_rad = x,
                    "eccentricity" => o.eccentricity = x,
                    "equinox_offset_s" => o.equinox_offset_s = x,
                    "solar_constant_w_m2" => o.solar_constant_w_m2 = x,
                    "radius_m" => o.radius_m = x,
                    _ => o.perihelion_lon_rad = x,
                }
            }
        }
        if !(o.year_s > 0.0 && o.rotation_period_s > 0.0 && o.solar_constant_w_m2 > 0.0) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "year_s / rotation_period_s / solar_constant_w_m2 必须为正",
            ));
        }
        Ok(o)
    }
}

/// `frac(x) = x − floor(x)`（确定性；只用于相位）
#[inline]
fn frac(x: f64) -> f64 {
    x - x.floor()
}

pub struct SolarOut {
    pub cosz: Vec<f64>,
    pub toa_sw: Vec<f64>,
    pub subsolar_lat_deg: f64,
    pub subsolar_lon_deg: f64,
    pub dist_au: f64,
}

/// 纯 Rust 核：`cosZ` 与大气顶短波（`(nlat,nlon)`，C 序）
pub fn solar_field(o: &OrbitConfig, t_seconds: f64, g: &GridSpec) -> SolarOut {
    // ① 黄经 / 赤纬
    let l = 2.0 * PI * frac((t_seconds - o.equinox_offset_s) / o.year_s);
    let sin_delta = (detmath::sin(o.obliquity_rad) * detmath::sin(l)).clamp(-1.0, 1.0);
    let delta = detmath::asin(sin_delta);
    let cos_delta = (1.0 - sin_delta * sin_delta).sqrt();

    // ② 日地距离因子
    let nu = l - o.perihelion_lon_rad;
    let r_ratio =
        (1.0 - o.eccentricity * o.eccentricity) / (1.0 + o.eccentricity * detmath::cos(nu));
    let flux_scale = o.solar_constant_w_m2 * r_ratio * r_ratio;

    // ③ 时角相位
    let rot_phase = 2.0 * PI * frac(t_seconds / o.rotation_period_s);

    let (nlat, nlon) = (g.nlat, g.nlon);
    let mut cosz = vec![0.0f64; nlat * nlon];
    let mut toa = vec![0.0f64; nlat * nlon];
    for j in 0..nlat {
        let sin_phi = g.mu[j];
        let cos_phi = (1.0 - sin_phi * sin_phi).sqrt();
        let a = sin_phi * sin_delta;
        let b = cos_phi * cos_delta;
        for i in 0..nlon {
            let h = rot_phase + g.lon_rad[i];
            let c = a + b * detmath::cos(h);
            let cz = if c > 0.0 { c } else { 0.0 };
            cosz[j * nlon + i] = cz;
            toa[j * nlon + i] = flux_scale * cz;
        }
    }

    let subsolar_lat_deg = detmath::to_deg(delta);
    let mut subsolar_lon_deg = detmath::to_deg(-rot_phase) % 360.0;
    if subsolar_lon_deg < 0.0 {
        subsolar_lon_deg += 360.0;
    }

    SolarOut {
        cosz,
        toa_sw: toa,
        subsolar_lat_deg,
        subsolar_lon_deg,
        dist_au: 1.0 / r_ratio,
    }
}

/// `s = ac.solar(orbit, t_seconds[, grid])`
///
/// `grid` 可省：优先用 `orbit["nlat"]/["nlon"]`，否则默认 T21(32×64)。
#[pyfunction]
#[pyo3(signature = (orbit, t_seconds, grid=None))]
pub fn solar<'py>(
    py: Python<'py>,
    orbit: &Bound<'py, PyDict>,
    t_seconds: f64,
    grid: Option<&Bound<'py, PyDict>>,
) -> PyResult<Bound<'py, PyDict>> {
    let o = OrbitConfig::from_dict(orbit)?;
    let g = match grid {
        Some(d) => crate::grid::from_dict(d)?,
        None => {
            let nlat = match orbit.get_item("nlat")? {
                Some(v) => v.extract::<usize>()?,
                None => 32,
            };
            let nlon = match orbit.get_item("nlon")? {
                Some(v) => v.extract::<usize>()?,
                None => 64,
            };
            GridSpec::new(nlat, nlon, 8, "gaussian", o.rotation_period_s)
        }
    };
    let s = solar_field(&o, t_seconds, &g);
    let d = PyDict::new(py);
    d.set_item("cosz", crate::ops::write_like(py, &[g.nlat, g.nlon], s.cosz)?)?;
    d.set_item(
        "toa_sw",
        crate::ops::write_like(py, &[g.nlat, g.nlon], s.toa_sw)?,
    )?;
    d.set_item("subsolar_lat_deg", s.subsolar_lat_deg)?;
    d.set_item("subsolar_lon_deg", s.subsolar_lon_deg)?;
    d.set_item("earth_sun_dist_au", s.dist_au)?;
    Ok(d)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid() -> GridSpec {
        GridSpec::new(32, 64, 8, "gaussian", 86400.0)
    }

    #[test]
    fn equinox_and_solstice_declination() {
        let o = OrbitConfig::default();
        let g = grid();
        let s0 = solar_field(&o, 0.0, &g);
        assert!(s0.subsolar_lat_deg.abs() < 1e-9, "春分赤纬应为 0");
        let quarter = o.year_s / 4.0;
        let s1 = solar_field(&o, quarter, &g);
        let eps_deg = detmath::to_deg(o.obliquity_rad);
        assert!(
            (s1.subsolar_lat_deg - eps_deg).abs() < 0.05,
            "夏至赤纬 {} 应≈{eps_deg}",
            s1.subsolar_lat_deg
        );
        let s3 = solar_field(&o, 3.0 * quarter, &g);
        assert!(
            (s3.subsolar_lat_deg + eps_deg).abs() < 0.05,
            "冬至赤纬应为 −ε"
        );
    }

    #[test]
    fn cosz_range_and_subsolar_point() {
        let o = OrbitConfig::default();
        let g = grid();
        let s = solar_field(&o, 12345.0, &g);
        assert!(s.cosz.iter().all(|&c| (-1e-12..=1.0 + 1e-12).contains(&c)));
        let cmax = s.cosz.iter().cloned().fold(0.0f64, f64::max);
        assert!(cmax > 0.995, "cosZ 最大值 {cmax}（网格分辨率决定，理论峰值 1）");
        assert!(s.toa_sw.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn annual_global_mean_insolation() {
        // 无偏心率时，全球年平均 TOA = S0/4
        let mut o = OrbitConfig::default();
        o.eccentricity = 0.0;
        let g = grid();
        let nsteps = 365;
        let mut acc = 0.0f64;
        for k in 0..nsteps {
            let t = o.year_s * (k as f64 + 0.5) / nsteps as f64;
            let s = solar_field(&o, t, &g);
            acc += g.global_mean(&s.toa_sw);
        }
        let mean = acc / nsteps as f64;
        let expect = o.solar_constant_w_m2 / 4.0;
        assert!(
            (mean - expect).abs() / expect < 0.01,
            "年平均 TOA {mean:.2} 应≈{expect:.2}"
        );
    }

    #[test]
    fn rotation_period_moves_subsolar_west() {
        let o = OrbitConfig::default();
        let g = grid();
        let a = solar_field(&o, 0.0, &g);
        let b = solar_field(&o, o.rotation_period_s / 4.0, &g);
        let d = (a.subsolar_lon_deg - b.subsolar_lon_deg + 360.0) % 360.0;
        assert!((d - 90.0).abs() < 1e-6, "1/4 天后日下点应西移 90°，实为 {d}");
    }
}
