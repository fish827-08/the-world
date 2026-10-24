//! 光照温度计算（Rust+Rayon 并行版）。
//!
//! 每 tick 计算全格的光照和温度。这是大世界（80000格）的最大瓶颈，
//! 占总时间 37%。用 Rayon 并行处理，在多核环境下显著加速。
//!
//! 设计：
//! - 预计算 cos_lat、col_rad、base_temp（永久不变）
//! - 每 tick 计算 sun_lon，然后并行计算每格的 illumination 和 temperature
//! - 缓存当前 tick 的结果，避免重复计算
//! - 就地修改传入的 numpy 数组（零拷贝）

use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1};
use pyo3::prelude::*;
use rayon::prelude::*;

const PAR_THRESHOLD: usize = 2048;

/// 裸指针的 Sync/Send 包装（Rayon 闭包需要 Sync+Send）。
/// 必须通过 `.add(i)` 方法访问，不能直接 `.0.add(i)`——
/// Rust 闭包最小化捕获会只抓裸指针字段（非 Sync）而非整个 SyncPtr。
#[derive(Clone, Copy)]
struct SyncPtr<T>(*mut T);
unsafe impl<T> Sync for SyncPtr<T> {}
unsafe impl<T> Send for SyncPtr<T> {}
impl<T> SyncPtr<T> {
    #[inline(always)]
    unsafe fn add(self, i: usize) -> *mut T {
        self.0.add(i)
    }
}

#[pyclass]
pub struct LightTempRust {
    n_cells: usize,
    cos_lat: Vec<f64>,
    col_rad: Vec<f64>,
    base_temp: Vec<f64>,
    day_boost: f64,
    rotation_period: f64,
    cached_tick: i64,
}

#[pymethods]
impl LightTempRust {
    #[new]
    fn new(
        n_cells: usize,
        cos_lat: PyReadonlyArray1<'_, f64>,
        col_rad: PyReadonlyArray1<'_, f64>,
        t_equator: f64,
        t_pole: f64,
        day_boost: f64,
        rotation_period: f64,
    ) -> PyResult<Self> {
        let cos_lat = cos_lat.as_slice()?.to_vec();
        let col_rad = col_rad.as_slice()?.to_vec();

        // 预计算基温（永久不变）
        let base_temp: Vec<f64> = if n_cells >= PAR_THRESHOLD {
            cos_lat
                .par_iter()
                .map(|&c| t_pole + (t_equator - t_pole) * c)
                .collect()
        } else {
            cos_lat
                .iter()
                .map(|&c| t_pole + (t_equator - t_pole) * c)
                .collect()
        };

        Ok(LightTempRust {
            n_cells,
            cos_lat,
            col_rad,
            base_temp,
            day_boost,
            rotation_period,
            cached_tick: -1,
        })
    }

    /// 计算当前 tick 的全格光照和温度，就地写入输出数组。
    /// 如果 tick 与缓存相同，跳过计算（调用方应直接使用缓存）。
    fn compute(
        &mut self,
        tick: i64,
        illum_out: Bound<'_, PyArray1<f64>>,
        temp_out: Bound<'_, PyArray1<f64>>,
    ) -> PyResult<bool> {
        if self.cached_tick == tick {
            return Ok(false); // 缓存命中，调用方应使用已有数组
        }

        let pi = std::f64::consts::PI;
        let two_pi = 2.0 * pi;
        let sun_lon =
            (tick % self.rotation_period as i64) as f64 / self.rotation_period * two_pi;

        let n = self.n_cells;
        let cos_lat = &self.cos_lat;
        let col_rad = &self.col_rad;
        let base_temp = &self.base_temp;
        let day_boost = self.day_boost;

        let mut illum = unsafe { illum_out.as_slice_mut()? };
        let mut temp = unsafe { temp_out.as_slice_mut()? };

        if n >= PAR_THRESHOLD {
            // Rayon 并行：用 SyncPtr 包装裸指针（满足 Sync+Send）
            let illum_ptr = SyncPtr(illum.as_mut_ptr());
            let temp_ptr = SyncPtr(temp.as_mut_ptr());

            (0..n).into_par_iter().for_each(|i| {
                let mut lon_diff = col_rad[i] - sun_lon;
                lon_diff = (lon_diff + pi) % two_pi - pi;
                let cos_val = lon_diff.cos();
                let day_term = if cos_val > 0.0 { cos_val } else { 0.0 };
                let ill = cos_lat[i] * day_term;
                unsafe {
                    *illum_ptr.add(i) = ill;
                    *temp_ptr.add(i) = base_temp[i] + ill * day_boost;
                }
            });
        } else {
            // 单线程（避免 Rayon 调度开销）
            for i in 0..n {
                let mut lon_diff = col_rad[i] - sun_lon;
                lon_diff = (lon_diff + pi) % two_pi - pi;
                let cos_val = lon_diff.cos();
                let day_term = if cos_val > 0.0 { cos_val } else { 0.0 };
                let ill = cos_lat[i] * day_term;
                illum[i] = ill;
                temp[i] = base_temp[i] + ill * day_boost;
            }
        }

        self.cached_tick = tick;
        Ok(true) // 已更新
    }
}
