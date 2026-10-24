//! 极区滤波与显式扩散的共用工具（`shallow` 与 `prim` 都用这一套）。
//!
//! * `filter_mode = 0`：关
//! * `filter_mode = 1`：纬向 1-2-1 平滑（旧口径）
//! * `filter_mode = 2`（默认）：**纬向截断** —— 只砍违反当地 CFL 的波数
//!
//! 截断阈值：`m_max(φ) = floor(cfl_target · 2πa·cosφ / (c_eff·dt))`，
//! 末端两个波数用余弦锥削过渡（避免 Gibbs 振铃）。实测 `cfl_target=0.2` 才能在 dt=900 稳定。

use crate::detmath::{self, PI};
use crate::grid::GridSpec;

#[derive(Clone, Copy, Debug)]
pub struct FilterCfg {
    pub mode: u8,
    pub n121: usize,
    pub phi_start_deg: f64,
    pub cfl_target: f64,
}

impl Default for FilterCfg {
    fn default() -> Self {
        FilterCfg {
            mode: 2,
            n121: 4,
            phi_start_deg: 60.0,
            cfl_target: 0.2,
        }
    }
}

/// 单行纬向截断（保留 `m <= m_max`）
pub fn truncate_row(nlon: usize, row: &mut [f64], m_max: usize, buf: &mut [f64]) {
    let nyq = nlon / 2;
    if m_max + 1 >= nyq {
        return;
    }
    let n = nlon as f64;
    let lim = m_max.max(1).min(63);
    let mut a = [0.0f64; 64];
    let mut b = [0.0f64; 64];
    for m in 0..=lim {
        let mut sa = 0.0f64;
        let mut sb = 0.0f64;
        for i in 0..nlon {
            let ang = 2.0 * PI * (m as f64) * (i as f64) / n;
            sa += row[i] * detmath::cos(ang);
            sb += row[i] * detmath::sin(ang);
        }
        a[m] = sa * 2.0 / n;
        b[m] = sb * 2.0 / n;
    }
    for i in 0..nlon {
        let mut acc = 0.5 * a[0];
        for m in 1..=lim {
            let ang = 2.0 * PI * (m as f64) * (i as f64) / n;
            let w = if m + 2 <= lim {
                1.0
            } else {
                let t = (lim as f64 - m as f64 + 1.0) / 3.0;
                let c = detmath::cos(0.5 * PI * (1.0 - t));
                c * c
            };
            acc += w * (a[m] * detmath::cos(ang) + b[m] * detmath::sin(ang));
        }
        buf[i] = acc;
    }
    row.copy_from_slice(&buf[..nlon]);
}

/// 极区滤波（作用在**倾向**上）
pub fn polar_filter(g: &GridSpec, f: &mut [f64], cfg: &FilterCfg, m_max: &[usize]) {
    if cfg.mode == 0 {
        return;
    }
    let nlon = g.nlon;
    let mut buf = vec![0.0f64; nlon];
    for j in 0..g.nlat {
        let sl = j * nlon..(j + 1) * nlon;
        if cfg.mode == 2 {
            truncate_row(nlon, &mut f[sl], m_max[j], &mut buf);
            continue;
        }
        if cfg.n121 == 0 {
            continue;
        }
        let lat_deg = detmath::to_deg(g.lat_rad[j]).abs();
        if lat_deg <= cfg.phi_start_deg {
            continue;
        }
        let t = ((lat_deg - cfg.phi_start_deg) / (90.0 - cfg.phi_start_deg)).min(1.0);
        let n = (t * cfg.n121 as f64).round() as usize;
        for _ in 0..n {
            for i in 0..nlon {
                let ip = if i + 1 == nlon { 0 } else { i + 1 };
                let im = if i == 0 { nlon - 1 } else { i - 1 };
                buf[i] = 0.25 * f[sl.start + im] + 0.5 * f[sl.start + i] + 0.25 * f[sl.start + ip];
            }
            f[sl.clone()].copy_from_slice(&buf);
        }
    }
}

/// 逐行 `m_max`（`c_eff` = 最快波速）
pub fn m_max_from_cfl(g: &GridSpec, c_eff: f64, dt: f64, cfg: &FilterCfg) -> Vec<usize> {
    (0..g.nlat)
        .map(|j| {
            if cfg.mode != 2 || c_eff * dt <= 0.0 {
                return g.nlon;
            }
            let num = cfg.cfl_target * 2.0 * PI * g.radius_m * g.cosphi[j];
            let m = (num / (c_eff * dt)).floor();
            if m < 0.0 {
                0
            } else {
                (m as usize).min(g.nlon)
            }
        })
        .collect()
}

/// 逐行扩散系数的稳定上限（`ν·dt/dx² ≤ 0.2`）
pub fn nu_row_cap(g: &GridSpec, nu: f64, dt: f64) -> Vec<f64> {
    (0..g.nlat)
        .map(|j| {
            let dx = g.dx_m[j].max(1.0);
            nu.min(0.2 * dx * dx / dt)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncate_preserves_mean_and_kills_high_m() {
        let nlon = 64;
        let mut row: Vec<f64> = (0..nlon)
            .map(|i| {
                let x = 2.0 * PI * i as f64 / nlon as f64;
                3.0 + detmath::cos(x) + 0.5 * detmath::cos(20.0 * x)
            })
            .collect();
        let mean0: f64 = row.iter().sum::<f64>() / nlon as f64;
        let mut buf = vec![0.0; nlon];
        truncate_row(nlon, &mut row, 6, &mut buf);
        let mean1: f64 = row.iter().sum::<f64>() / nlon as f64;
        assert!((mean1 - mean0).abs() < 1e-12, "截断必须保均值");
        // m=1 保留、m=20 去掉：与 3+cos 的差应很小
        let mut worst = 0.0f64;
        for i in 0..nlon {
            let x = 2.0 * PI * i as f64 / nlon as f64;
            worst = worst.max((row[i] - (3.0 + detmath::cos(x))).abs());
        }
        assert!(worst < 0.05, "低波数应基本保留，偏差 {worst:e}");
    }

    #[test]
    fn m_max_shrinks_toward_pole() {
        let g = GridSpec::new(32, 64, 8, "gaussian", 86400.0);
        let cfg = FilterCfg::default();
        let m = m_max_from_cfl(&g, 300.0, 900.0, &cfg);
        assert!(m[0] < m[16], "极区 m_max 必须小于赤道");
        assert!(m[0] <= 4, "极区应被截到 ≤4（实测 {}）", m[0]);
        assert!(m[16] >= 24, "赤道附近应基本不截断（实测 {}）", m[16]);
    }
}
