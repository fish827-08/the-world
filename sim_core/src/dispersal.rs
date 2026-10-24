//! L10a 果实-种子传播数值管线（先行版）。
//!
//! 与 simulation/sphere_engine.py 的 _step_fruit_charge / _step_eat_fruit 逐位一致。
//! 仅下沉确定性数值段（蓄力/释放/能量转移），随机部分（种子摄入/排泄/萌发）保留 Python 侧（L10b）。
//!
//! 注意：fruit_grid 是 (n_cells,) 数组，多植物同格用 np.add.at 累加；
//! Rust 侧用相同的累加语义（循环中顺序处理，同格多次 +=）。

/// 植物蓄力→结果（L10a 步骤 3.5）。
///
/// 逻辑：
/// 1. 植物判定 g19 >= plant_threshold
/// 2. 蓄力 += charge_rate × g8
/// 3. 蓄力 >= fruit_threshold → 释放：fruit_grid[cell] += charge × fruit_ratio，charge=0
pub fn fruit_charge_batch(
    flat: &[i64],
    genes: &[f64],
    gene_count: usize,
    fruit_charge: &mut [f64],
    fruit_grid: &mut [f64],
    g19_idx: usize,
    g8_idx: usize,
    plant_threshold: f64,
    charge_rate: f64,
    fruit_threshold: f64,
    fruit_ratio: f64,
) {
    let p = flat.len();
    for i in 0..p {
        let g19 = genes[i * gene_count + g19_idx];
        if g19 < plant_threshold {
            continue;
        }
        let g8 = genes[i * gene_count + g8_idx];
        fruit_charge[i] += charge_rate * g8;
        if fruit_charge[i] >= fruit_threshold {
            let cell = flat[i] as usize;
            let release = fruit_charge[i] * fruit_ratio;
            fruit_grid[cell] += release;
            fruit_charge[i] = 0.0;
        }
    }
}

/// 动物吃果实→能量转移（L10a 步骤 4.5）。
///
/// 逻辑：
/// 1. 动物判定 g19 < plant_threshold
/// 2. 当前格有果实 → eat_amount = fruit_grid[cell] × eat_rate
/// 3. energy += eat_amount × digest_ratio
/// 4. fruit_grid[cell] -= eat_amount（同格多动物累加扣减）
/// 5. fruit_grid 负值截断为 0
pub fn eat_fruit_batch(
    flat: &[i64],
    genes: &[f64],
    gene_count: usize,
    energy: &mut [f64],
    fruit_grid: &mut [f64],
    g19_idx: usize,
    plant_threshold: f64,
    eat_rate: f64,
    digest_ratio: f64,
) {
    let p = flat.len();
    // 第一遍：计算所有动物的 eat_amount（基于初始 fruit_grid，与 Python 向量化语义一致）
    // 收集 (index, cell, eat_amount)
    let mut eats: Vec<(usize, usize, f64)> = Vec::new();
    for i in 0..p {
        let g19 = genes[i * gene_count + g19_idx];
        if g19 >= plant_threshold {
            continue;
        }
        let cell = flat[i] as usize;
        let cell_fruit = fruit_grid[cell];
        if cell_fruit <= 0.0 {
            continue;
        }
        let eat_amount = cell_fruit * eat_rate;
        eats.push((i, cell, eat_amount));
    }
    // 第二遍：统一增加 energy、扣减 fruit_grid（同格多动物累加，与 np.add.at 一致）
    for (i, _cell, eat_amount) in &eats {
        energy[*i] += eat_amount * digest_ratio;
    }
    for (_i, cell, eat_amount) in &eats {
        fruit_grid[*cell] -= eat_amount;
    }
    // 负值截断（浮点误差防护）
    for v in fruit_grid.iter_mut() {
        if *v < 0.0 {
            *v = 0.0;
        }
    }
}
