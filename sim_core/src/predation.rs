//! 捕食（L4）Rust 下沉：高攻击基因个体攻击邻格猎物。
//!
//! 语义与 sphere_engine._step_population 步骤 5.5 逐位等价。
//! 提供两个入口：
//!   - `predation_attack`：自包含（自己构建 CSR），供独立调用/对拍
//!   - `predation_attack_with_csr`：接收预构建 CSR，供 l4_l5 合并调用

use crate::genes::G_AGGRESSION;

/// 捕食核心（接收预构建 CSR）。
///
/// 参数化（A2 收编）：attack_cost/成功率乘数/上下限/能量与胃粮转移比例
/// 均由配置传入；默认值 = 旧硬编码（0.1/0.5/0.1/0.9/0.4/0.4），双路径逐位等价。
#[allow(clippy::too_many_arguments)]
pub fn predation_attack_with_csr(
    energy: &mut [f64],
    stomach: &mut [f64],
    predation_mask: &mut [bool],
    flat: &[i64],
    genes: &[f64],
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
    attack_cost: f64,
    success_gene_gain: f64,
    success_floor: f64,
    success_ceil: f64,
    transfer_ratio: f64,
    stomach_transfer: f64,
    cell_indptr: &[usize],
    cell_indices: &[usize],
) {
    let n = flat.len();
    if n == 0 || attackers.is_empty() {
        return;
    }
    let stomach_cap = max_energy / eat_efficiency.max(1e-9);

    for (k, &idx_i64) in attackers.iter().enumerate() {
        let idx = idx_i64 as usize;
        if idx >= n {
            continue;
        }
        if energy[idx] <= attack_cost {
            continue;
        }
        let c = flat[idx] as usize;
        if c >= n_cells {
            continue;
        }

        let mut prey_list: Vec<usize> = Vec::with_capacity(16);
        // P0.1：普通格主表 8 项；极点格极点带整行（与旧 fat 表逐位同序）
        let nb = crate::neighbors::nb_slice(
            neighbors, pole_nb, c, n_cols, nb_stride, pole_top, pole_bottom,
        );
        for &nbc in nb {
            if nbc < 0 {
                continue;
            }
            let nbc = nbc as usize;
            if nbc >= n_cells {
                continue;
            }
            let start = cell_indptr[nbc];
            let end = cell_indptr[nbc + 1];
            for pos in start..end {
                let j = cell_indices[pos];
                if j != idx {
                    prey_list.push(j);
                }
            }
        }
        prey_list.sort_unstable();
        prey_list.dedup();

        if prey_list.is_empty() {
            continue;
        }

        let prey_pos = (rand_prey[k].rem_euclid(prey_list.len() as i64)) as usize;
        let prey = prey_list[prey_pos];

        if predation_mask[prey] {
            continue;
        }

        energy[idx] -= attack_cost;

        let total = energy[idx] + energy[prey];
        let energy_ratio = energy[idx] / total.max(1e-9);
        let g16 = genes[idx * gene_count + G_AGGRESSION];
        let success_rate =
            (energy_ratio * (0.5 + g16 * success_gene_gain)).clamp(success_floor, success_ceil);

        if rand_success[k] < success_rate {
            predation_mask[prey] = true;
            energy[idx] += energy[prey] * transfer_ratio;
            stomach[idx] = (stomach[idx] + stomach[prey] * stomach_transfer).min(stomach_cap);
            stomach[prey] = 0.0;
        }
    }
}

/// 捕食（自包含，自己构建 CSR）。
#[allow(clippy::too_many_arguments)]
pub fn predation_attack(
    energy: &mut [f64],
    stomach: &mut [f64],
    predation_mask: &mut [bool],
    flat: &[i64],
    genes: &[f64],
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
) {
    let n = flat.len();
    if n == 0 || attackers.is_empty() {
        return;
    }

    // 构建 CSR
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

    predation_attack_with_csr(
        energy, stomach, predation_mask, flat, genes,
        attackers, rand_prey, rand_success,
        neighbors, pole_nb, n_cells, nb_stride, gene_count,
        n_cols, pole_top, pole_bottom,
        max_energy, eat_efficiency,
        0.1, 0.5, 0.1, 0.9, 0.4, 0.4,  // A2 收编前的旧默认值（独立对拍接口不动）
        &cell_indptr, &cell_indices,
    );
}
