//! step_vectors：种群数值管线（模块三 · 3.3）。
//!
//! 职责（对齐 内部项目概览 3.3 与 sphere_engine._step_population）
//! ------------------------------------------------------------------------
//! Python 引擎把每个 tick 的【确定性批量数值运算】下沉到这里，分两个阶段：
//!
//! - `step_vectors_stage1`（第 1~3 步）：光合收入、胃→能量转化（代谢）、
//!   基础维持+恒温维持扣费。跑在"进食"之前（进食只动胃`,由 Python 完成）。
//! - `step_vectors_stage2`（第 5~8 步）：移动扣费、年龄推进、死亡判定、
//!   繁殖冷却倒数、繁殖候选掩码。跑在"进食+RNG 决策"之后。
//!
//! 为什么分两段（而不是一个函数一次跑完）：
//! 进食（Python）会改写胃与资源格，且移动抽样/邻格觅食等 RNG 调用顺序
//! 必须与纯 Python 引擎完全一致（保证同种子同结果）；两段式把 RNG 缺口
//! 让给 Python，Rust 只碰确定性数值。
//!
//! 数值纪律（保证与 numpy 逐位一致）
//! ----------------------------------
//! 只用 f64 四则运算与 min/max/clip，不用 exp/pow/sqrt 等超越函数
//! （会引入 1-ULP 分歧）。逐元素处理顺序与 Python 引擎的向量化顺序一致，
//! 因此同一份输入下 Rust 与 numpy 输出逐位相等（regrow 已验证同模式）。
//!
//! 入参长度约定：除 genes 外所有数组长度均为 n（本 tick 存活者数）；
//! genes 为行主序展开的 (n × gene_count) 平坦切片。
//!
//! 并行化：stage1/stage2 均为纯逐元素运算（每个 i 完全独立、无跨元素读写），
//! 使用 Rayon 并行。SyncPtr 包装裸指针实现 Sync/Send（Rayon 保证索引 i 不重复，
//! 不存在数据竞争）。与单线程版本逐位一致（f64 运算顺序不变）。

use rayon::prelude::*;

/// 裸指针的 Sync/Send 包装：Rayon 并行迭代中每个索引只被一个线程访问，
/// 因此通过裸指针的读写不存在数据竞争。仅用于本模块内部的并行热路径。
/// 注意：必须通过方法访问指针，避免闭包最小化捕获只抓到裸指针字段。
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

/// 并行 for：n >= PAR_THRESHOLD 时用 Rayon，否则单线程。
/// 小规模下 Rayon 调度开销大于收益，阈值避免退化。
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

/// 第 1~3 步：光合收入 + 代谢转化 + 维持扣费（就地更新 energy/stomach）。
///
/// 语义与 sphere_engine._step_population 的第 1、2、3 步逐步一致：
/// ```text
/// 1) energy += illum · g8 · photo_max
/// 2) metab  = 0.5 + g1·1.5；rate = base_metabolism·metab·eff_activity
///    digest = min(stomach, rate)；energy += digest·eat_efficiency；stomach -= digest
/// 3) lifespan = day_length·(1 + g3·7)；age_mult 按年龄分幼/成/老；
///    energy -= base_metabolism·metab·age_mult；energy -= homeo_upkeep·g9
/// ```
/// age 只读（供给 age_mult 用）；+1 由 stage2 完成。
pub fn step_vectors_stage1(
    energy: &mut [f64],
    stomach: &mut [f64],
    genes: &[f64],
    gene_count: usize,
    eff_activity: &[f64],
    illumination: &[f64],
    age: &[i64],
    photo_max: f64,
    base_metabolism: f64,
    eat_efficiency: f64,
    growth_mult: f64,
    senile_mult: f64,
    maturity_fraction: f64,
    senile_fraction: f64,
    homeo_upkeep: f64,
    day_length: f64,
) {
    debug_assert_eq!(energy.len(), stomach.len());
    debug_assert_eq!(energy.len(), eff_activity.len());
    debug_assert_eq!(energy.len(), illumination.len());
    debug_assert_eq!(energy.len(), age.len());
    debug_assert_eq!(energy.len() * gene_count, genes.len());

    let n = energy.len();
    let e_p = SyncPtr(energy.as_mut_ptr());
    let s_p = SyncPtr(stomach.as_mut_ptr());
    par_for(n, |i| {
        let g = &genes[i * gene_count..(i + 1) * gene_count];
        unsafe {
            let e = &mut *e_p.add(i);
            let st = &mut *s_p.add(i);

            // 1) 光合收入
            *e += illumination[i] * g[8] * photo_max;

            // 2) 代谢转化
            let metab_mult = 0.5 + g[1] * 1.5;
            let digest_rate = base_metabolism * metab_mult * eff_activity[i];
            let digest = if *st < digest_rate { *st } else { digest_rate };
            *e += digest * eat_efficiency;
            *st -= digest;

            // 3) 维持扣费
            let life_span = day_length * (1.0 + g[3] * 7.0);
            let a = age[i] as f64;
            let age_mult = if a < maturity_fraction * life_span {
                growth_mult
            } else if a >= senile_fraction * life_span {
                senile_mult
            } else {
                1.0
            };
            *e -= base_metabolism * metab_mult * age_mult;
            *e -= homeo_upkeep * g[9];
        }
    });
}

/// 第 5~8 步：移动扣费 + 年龄推进 + 死亡判定 + 冷却倒数与繁殖候选。
///
/// 语义与 sphere_engine._step_population 的第 5(后半)、6、7、8 步一致：
/// ```text
/// 5) can_move = moved_raw && energy >= move_cost_ind；energy -= move_cost_ind（仅移动者）
/// 6) age += 1
/// 7) starved = energy <= 0；expired = !starved && (age_推进后 >= lifespan)；dead = starved|expired
/// 8) cooldown = max(0, cooldown-1)；repro = !dead && energy>=repro_thr
///             && cooldown<=0 && (推进前年龄 >= maturity_age)
/// ```
///
/// 说明：
/// - `moved_raw` 是 Python 用 RNG 抽好的"本次要不要走"（第 5 步前半）；
///   能量门槛判定放在本函数内（与 Python 同刻的状态一致），结果写 out_moved。
/// - 年龄用"推进前"判断成熟（与 Python 捕获 age_f 的时机一致），
///   用"推进后"判定老死（与 Python 第 7 步读 +1 后的数组一致）。
/// - 四个输出掩码 out_moved/out_starved/out_expired/out_repro 由调用方预分配，
///   长度 n；Python 据此做出生/死亡分流。
pub fn step_vectors_stage2(
    energy: &mut [f64],
    age: &mut [i64],
    cooldown: &mut [f64],
    genes: &[f64],
    gene_count: usize,
    moved_raw: &[bool],
    move_cost_ind: &[f64],
    out_moved: &mut [bool],
    out_starved: &mut [bool],
    out_expired: &mut [bool],
    out_repro: &mut [bool],
    day_length: f64,
    maturity_fraction: f64,
    max_energy: f64,
) {
    debug_assert_eq!(energy.len(), age.len());
    debug_assert_eq!(energy.len(), cooldown.len());
    debug_assert_eq!(energy.len(), moved_raw.len());
    debug_assert_eq!(energy.len(), move_cost_ind.len());
    debug_assert_eq!(energy.len(), out_moved.len());
    debug_assert_eq!(energy.len(), out_starved.len());
    debug_assert_eq!(energy.len(), out_expired.len());
    debug_assert_eq!(energy.len(), out_repro.len());
    debug_assert_eq!(energy.len() * gene_count, genes.len());

    let n = energy.len();
    let e_p = SyncPtr(energy.as_mut_ptr());
    let a_p = SyncPtr(age.as_mut_ptr());
    let c_p = SyncPtr(cooldown.as_mut_ptr());
    let om_p = SyncPtr(out_moved.as_mut_ptr());
    let os_p = SyncPtr(out_starved.as_mut_ptr());
    let oe_p = SyncPtr(out_expired.as_mut_ptr());
    let or_p = SyncPtr(out_repro.as_mut_ptr());
    par_for(n, |i| {
        let g = &genes[i * gene_count..(i + 1) * gene_count];
        let life_span = day_length * (1.0 + g[3] * 7.0);
        unsafe {
            let a = *a_p.add(i);
            let e = &mut *e_p.add(i);

            // 5) 移动扣费
            let can_move = moved_raw[i] && *e >= move_cost_ind[i];
            *om_p.add(i) = can_move;
            if can_move { *e -= move_cost_ind[i]; }

            // 6) 年龄推进
            *a_p.add(i) = a + 1;

            // 7) 死亡判定
            let starved = *e <= 0.0;
            let expired = !starved && (a as f64 + 1.0) >= life_span;
            *os_p.add(i) = starved;
            *oe_p.add(i) = expired;
            let dead = starved || expired;

            // 8) 冷却倒数 + 繁殖候选
            let cd_cell = &mut *c_p.add(i);
            let mut cd = *cd_cell - 1.0;
            if cd < 0.0 { cd = 0.0; }
            *cd_cell = cd;
            let repro_thr = (0.25 + g[2] * 0.65) * max_energy;
            let mature = (a as f64) >= maturity_fraction * life_span;
            *or_p.add(i) = !dead && *e >= repro_thr && cd <= 0.0 && mature;
        }
    });
}