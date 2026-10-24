//! 文化学习（L5）Rust 下沉：幼体向邻格成体学习信号解读表。
//!
//! 语义与 sphere_engine._step_population 步骤 6.5 逐位等价。
//! 提供两个入口：
//!   - `culture_learn`：自包含（自己构建 CSR），供独立调用/对拍
//!   - `culture_learn_with_csr`：接收预构建 CSR，供 l4_l5 合并调用

/// 文化学习核心（接收预构建 CSR）。
pub fn culture_learn_with_csr(
    interpret: &mut [f64],
    flat: &[i64],
    age: &[i64],
    maturity_age: &[f64],
    neighbors: &[i64],
    pole_nb: &[i64],
    n_cells: usize,
    nb_stride: usize,
    n_cols: usize,
    pole_top: usize,
    pole_bottom: usize,
    alpha: f64,
    cell_indptr: &[usize],
    cell_indices: &[usize],
) {
    let n = flat.len();
    if n == 0 {
        return;
    }

    for idx in 0..n {
        if (age[idx] as f64) >= maturity_age[idx] {
            continue;
        }
        let c = flat[idx] as usize;
        if c >= n_cells {
            continue;
        }

        // 收集邻格所有个体（排序+去重，与 Python np.isin 集合操作一致）
        // P0.1：普通格主表 8 项；极点格极点带整行（与旧 fat 表逐位同序）
        let mut neighbor_inds: Vec<usize> = Vec::with_capacity(16);
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
                    neighbor_inds.push(j);
                }
            }
        }
        neighbor_inds.sort_unstable();
        neighbor_inds.dedup();

        // 累加邻格成体的解读表
        let mut adult_sum = [0.0f64; 16];
        let mut adult_count = 0usize;
        for &j in &neighbor_inds {
            if (age[j] as f64) >= maturity_age[j] {
                let base = j * 16;
                for k in 0..16 {
                    adult_sum[k] += interpret[base + k];
                }
                adult_count += 1;
            }
        }

        if adult_count > 0 {
            let base = idx * 16;
            let inv_count = 1.0 / adult_count as f64;
            for k in 0..16 {
                let mean = adult_sum[k] * inv_count;
                interpret[base + k] += alpha * (mean - interpret[base + k]);
            }
        }
    }
}

/// 文化学习（自包含，自己构建 CSR）。
pub fn culture_learn(
    interpret: &mut [f64],
    flat: &[i64],
    age: &[i64],
    maturity_age: &[f64],
    neighbors: &[i64],
    pole_nb: &[i64],
    n_cells: usize,
    nb_stride: usize,
    n_cols: usize,
    pole_top: usize,
    pole_bottom: usize,
    alpha: f64,
) {
    let n = flat.len();
    if n == 0 {
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

    culture_learn_with_csr(
        interpret, flat, age, maturity_age,
        neighbors, pole_nb, n_cells, nb_stride,
        n_cols, pole_top, pole_bottom, alpha,
        &cell_indptr, &cell_indices,
    );
}
