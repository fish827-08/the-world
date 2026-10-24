//! pleasure：愉悦度批量更新下沉（L7a C2）。
//!
//! 语义与 sphere_engine._update_pleasure 逐位等价：
//! 1. 情境编码（120 种：能量5×食物4×邻居3×信号2）
//! 2. 事件收益（Δ能量 + 社会增益 + 信息增益）
//! 3. RPE = 收益 − expectation[情境]
//! 4. 更新 valence（瞬态）/arousal（唤醒）/expectation（EWMA）/baseline（习惯化）
//!
//! 只更新前 P 个个体（本 tick 开始时存在的亲代），子代不参与本 tick。
//!
//! 并行化：纯逐元素运算，每个个体只读写自己的 valence/arousal/expectation[context]/baseline，
//! 无跨个体数据竞争。使用 Rayon + SyncPtr 包装裸指针，与单线程版本逐位一致。

use rayon::prelude::*;

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

const PAR_THRESHOLD: usize = 2048;

#[inline]
fn par_for<F>(n: usize, f: F)
where
    F: Fn(usize) + Sync + Send,
{
    if n < PAR_THRESHOLD {
        for i in 0..n {
            f(i);
        }
    } else {
        (0..n).into_par_iter().for_each(f);
    }
}

/// 愉悦度批量更新。
///
/// # 参数
/// - `flat`: 个体位置 (N,)
/// - `energy_now`: 当前能量 (N,)（步骤 8.5 时的能量）
/// - `energy_before`: tick 开始时能量 (N,)（步骤 1 前保存）
/// - `densities`: 每格个体数 (n_cells,)
/// - `resource_grid`: 资源存量 (n_cells,)
/// - `resource_capacity`: 资源容量 (n_cells,)
/// - `signal_marks`: 信号场标记 (n_cells,) uint8
/// - `valence`: 情绪效价 (N,)，就地修改
/// - `arousal`: 唤醒度 (N,)，就地修改
/// - `expectation`: 预期表 (N*120,) 展平，就地修改
/// - `baseline`: 基线 (N,)，就地修改
/// - `max_energy`: 最大能量（300.0）
/// - `alpha`: EWMA 学习率（0.05）
/// - `valence_decay`: valence 衰减率（0.95）
/// - `arousal_decay`: arousal 衰减率（0.97）
/// - `baseline_rate`: 基线漂移率（0.001）
/// - `max_reward`: 预期上限（2.0）
/// - `w_energy`: 能量收益权重（0.5）
/// - `w_info`: 信息收益权重（0.3）
/// - `w_social`: 社会收益权重（0.2）
#[allow(clippy::too_many_arguments)]
pub fn pleasure_update_batch(
    flat: &[i64],
    energy_now: &[f64],
    energy_before: &[f64],
    densities: &[f64],
    resource_grid: &[f64],
    resource_capacity: &[f64],
    signal_marks: &[u8],
    valence: &mut [f64],
    arousal: &mut [f64],
    expectation: &mut [f64],  // (N * 120) 展平
    baseline: &mut [f64],
    max_energy: f64,
    alpha: f64,
    valence_decay: f64,
    arousal_decay: f64,
    baseline_rate: f64,
    max_reward: f64,
    w_energy: f64,
    w_info: f64,
    w_social: f64,
) {
    let n = flat.len();
    debug_assert_eq!(energy_now.len(), n);
    debug_assert_eq!(energy_before.len(), n);
    debug_assert_eq!(valence.len(), n);
    debug_assert_eq!(arousal.len(), n);
    debug_assert_eq!(baseline.len(), n);
    debug_assert_eq!(expectation.len(), n * 120);

    let max_e = if max_energy > 0.0 { max_energy } else { 1.0 };

    let v_p = SyncPtr(valence.as_mut_ptr());
    let a_p = SyncPtr(arousal.as_mut_ptr());
    let exp_p = SyncPtr(expectation.as_mut_ptr());
    let bl_p = SyncPtr(baseline.as_mut_ptr());
    par_for(n, |i| {
        let cell = flat[i] as usize;
        unsafe {
            // 1) 情境编码
            let e_raw = energy_now[i] / max_e * 5.0;
            let e_bin = if e_raw < 0.0 { 0i64 } else if e_raw >= 5.0 { 4 } else { e_raw as i64 };
            let cap = if resource_capacity[cell] > 1e-9 { resource_capacity[cell] } else { 1e-9 };
            let food_ratio = (resource_grid[cell] / cap).clamp(0.0, 1.0);
            let f_raw = food_ratio * 4.0;
            let f_bin = if f_raw < 0.0 { 0i64 } else if f_raw >= 4.0 { 3 } else { f_raw as i64 };
            let n_count = densities[cell];
            let n_bin = if n_count <= 0.0 { 0i64 } else if n_count <= 2.0 { 1 } else { 2 };
            let context = (e_bin * 24 + f_bin * 6 + n_bin * 2) as usize;

            // 2) 事件收益
            let delta_e = ((energy_now[i] - energy_before[i]) / max_e).clamp(-1.0, 1.0);
            let social = if n_count > 0.0 { 0.2 } else { -0.1 };
            let info = if signal_marks[cell] > 0 { 0.5 } else { 0.0 };
            let reward = w_energy * delta_e + w_info * info + w_social * social;

            // 3) RPE
            let exp_idx = i * 120 + context;
            let exp = *exp_p.add(exp_idx);
            let rpe = reward - exp;

            // 4) 更新
            let mut v = (*v_p.add(i) + rpe * 0.3) * valence_decay;
            v = v.clamp(-1.0, 1.0);
            *v_p.add(i) = v;

            let mut a = (*a_p.add(i) + rpe.abs() * 0.2) * arousal_decay;
            a = a.clamp(0.0, 1.0);
            *a_p.add(i) = a;

            *exp_p.add(exp_idx) = (exp + alpha * rpe).clamp(0.0, max_reward);
            *bl_p.add(i) = *bl_p.add(i) * (1.0 - baseline_rate) + v * baseline_rate;
        }
    });
}
