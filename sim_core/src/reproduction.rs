//! 繁殖（L7b / T4）Rust 下沉：子代数值核心。
//!
//! 语义与 `sphere_engine._step_population` 步骤 8 逐位等价。
//!
//! 边界约定（与 L2/L4 下沉一致）：Rust 只做确定性元素级数值，
//! RNG 全部由 Python 预生成并按原顺序消费（mut 判定 → 突变噪声 →
//! expectation 继承噪声 → interpret 继承噪声），本模块只消费这些
//! 数组做纯算术，不产生新的随机数。ID 分配、np.concatenate 拼接、
//! 世代/亲代/年龄等零数组仍在 Python 侧。
//!
//! 为什么还能有收益（诚实标注）：繁殖段已是 NumPy 向量化，下沉本身
//! 无净收益；真正的热点在每 tick 的 15 次 `np.concatenate`，属 L7c
//!（slots 模式）范围。本模块的意义是保持"数值走 Rust"的架构一致性 +
//! 为 T5/L7c 提供稳定的子代数值出口。

use crate::genes::{G_PARENTAL_INVEST, G_REPRO_COOLDOWN};

/// 批量生成子代（就地修改父本 energy/stomach/cooldown）。
///
/// 全部数组按行主序展平；ri 为父本索引（K 个），输出行按 j = 0..K
/// 顺序对齐 ri（子代拼接顺序与 Python `flatnonzero(repro)[:K]` 一致）。
#[allow(clippy::too_many_arguments)]
pub fn reproduce_batch(
    ri: &[i64],
    genes: &[f64],          // (N * gene_count) 父本基因行
    energy: &mut [f64],     // (N) 父本能量（就地扣减 split）
    stomach: &mut [f64],    // (N) 父本胃粮（就地扣减 split）
    cooldown: &mut [f64],   // (N) 父本繁殖冷却（就地置 g12 × 60）
    exp: &[f64],            // (N * exp_size) 父本 expectation 行
    interpret: &[f64],      // (N * 16) 父本信号解读表行
    trust: &[f64],          // (N) 父本信任度
    baseline: &[f64],       // (N) 父本情绪基线
    mut_mask: &[u8],        // (K * gene_count) 1=该基因位突变
    gene_noise: &[f64],     // (K * gene_count) 突变噪声（未突变位取 0）
    exp_noise: &[f64],      // (K * exp_size) expectation 继承噪声
    interp_noise: &[f64],   // (K * 16) 解读表继承噪声
    child_genes: &mut [f64],   // (K * gene_count) 输出
    child_energy: &mut [f64],  // (K) 输出
    child_stomach: &mut [f64], // (K) 输出
    child_exp: &mut [f64],     // (K * exp_size) 输出
    child_interp: &mut [f64],  // (K * 16) 输出
    child_trust: &mut [f64],   // (K) 输出
    child_baseline: &mut [f64],// (K) 输出
    gene_count: usize,
    exp_size: usize,
    gene_min: f64,
    gene_max: f64,
    max_reward: f64,
    repro_cd_scale: f64,
) {
    let k = ri.len();
    for j in 0..k {
        let p = ri[j] as usize;
        if p >= genes.len() / gene_count {
            continue; // 防御：索引越界则跳过（不应发生）
        }
        let base = p * gene_count;
        let cbase = j * gene_count;

        // ── 基因遗传 + 突变（与 numpy clip(base + where(mut, noise, 0)) 一致）──
        for g in 0..gene_count {
            let noise = if cbase + g < mut_mask.len() && mut_mask[cbase + g] != 0 {
                gene_noise[cbase + g]
            } else {
                0.0
            };
            let v = genes[base + g] + noise;
            child_genes[cbase + g] = v.clamp(gene_min, gene_max);
        }

        // ── 传代投入切分（split = 0.3 + g7×0.4，读原值再扣减）──
        let split = 0.3 + genes[base + G_PARENTAL_INVEST] * 0.4;
        let ce = energy[p] * split;
        let cs = stomach[p] * split;
        energy[p] -= ce;
        stomach[p] -= cs;
        child_energy[j] = ce;
        child_stomach[j] = cs;

        // ── 冷却：生完进入 g12×60 tick 冷却（与 Python 同式）──
        cooldown[p] = genes[base + G_REPRO_COOLDOWN] * repro_cd_scale;

        // ── expectation 继承 + 噪声 + clip(0, max_reward)──
        let ebase = p * exp_size;
        let cexp = j * exp_size;
        for s in 0..exp_size {
            let v = exp[ebase + s] + exp_noise[cexp + s];
            child_exp[cexp + s] = v.clamp(0.0, max_reward);
        }

        // ── 解读表继承 + 噪声（不 clip）──
        let ibase = p * 16;
        let cibase = j * 16;
        for s in 0..16 {
            child_interp[cibase + s] = interpret[ibase + s] + interp_noise[cibase + s];
        }

        // ── trust / baseline 继承（与 Python 同式：trust×0.8+0.5×0.2）──
        child_trust[j] = trust[p] * 0.8 + 0.5 * 0.2;
        child_baseline[j] = baseline[p] * 0.5;
    }
}