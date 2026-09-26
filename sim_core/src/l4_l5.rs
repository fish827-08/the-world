//! L4+L5 合并 Rust 下沉：捕食 + 文化学习，共用一次 CSR 构建。
//!
//! 单独下沉捕食和文化学习时，每个函数各构建一次 cell→个体 CSR 列表，
//! N=5000 时两次构建的开销超过了 np.isin 的节省。合并后每 tick 只构建
//! 一次 CSR，捕食和文化学习共用，消除重复开销。
//!
//! 步骤顺序与引擎一致：5.5 捕食 → 6.5 文化学习（年龄已在 stage2 推进）。

use crate::culture;
use crate::predation;

/// 捕食 + 文化学习合并函数。
///
/// 所有参数语义与 predation::predation_attack 和 culture::culture_learn 一致。
#[allow(clippy::too_many_arguments)]
pub fn predation_and_culture(
    energy: &mut [f64],
    stomach: &mut [f64],
    predation_mask: &mut [bool],
    interpret: &mut [f64],
    flat: &[i64],
    genes: &[f64],
    age: &[i64],
    maturity_age: &[f64],
    attackers: &[i64],
    rand_prey: &[i64],
    rand_success: &[f64],
    neighbors: &[i64],
    pole_nb: &[i64],
    n_cells: usize,
    nb_stride: usize,
    gene_count: usize,
    n_cols: usize,
    pole_top: usize,
    pole_bottom: usize,
    max_energy: f64,
    eat_efficiency: f64,
    culture_alpha: f64,
    attack_cost: f64,
    success_gene_gain: f64,
    success_floor: f64,
    success_ceil: f64,
    transfer_ratio: f64,
    stomach_transfer: f64,
) {
    let n = flat.len();
    if n == 0 {
        return;
    }

    // ── 构建一次 CSR，捕食和文化学习共用 ──
    let mut cell_counts = vec![0usize; n_cells];
    for i in 0..n {
        let c = flat[i] as usize;
        if c < n_cells {
            cell_counts[c] += 1;
        }
    }
    let mut cell_indptr = vec![0usize; n_cells + 1];
    for c in 0..n_cells {
        cell_indptr[c + 1] = cell_indptr[c] + cell_counts[c];
    }
    let mut cell_indices = vec![0usize; n];
    let mut cell_pos = cell_indptr.clone();
    for i in 0..n {
        let c = flat[i] as usize;
        if c < n_cells {
            cell_indices[cell_pos[c]] = i;
            cell_pos[c] += 1;
        }
    }

    // ── 5.5 捕食（用共用 CSR）──
    if !attackers.is_empty() {
        predation::predation_attack_with_csr(
            energy, stomach, predation_mask,
            flat, genes, attackers, rand_prey, rand_success,
            neighbors, pole_nb, n_cells, nb_stride, gene_count,
            n_cols, pole_top, pole_bottom,
            max_energy, eat_efficiency,
            attack_cost, success_gene_gain, success_floor, success_ceil,
            transfer_ratio, stomach_transfer,
            &cell_indptr, &cell_indices,
        );
    }

    // ── 6.5 文化学习（用共用 CSR）──
    culture::culture_learn_with_csr(
        interpret, flat, age, maturity_age,
        neighbors, pole_nb, n_cells, nb_stride,
        n_cols, pole_top, pole_bottom, culture_alpha,
        &cell_indptr, &cell_indices,
    );
}
