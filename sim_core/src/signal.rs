//! signal：信号发射批量下沉（L7a C1）。
//!
//! 语义与 sphere_engine 步骤 4.5 逐位等价：
//! 1. 发射判定：rand_emit[i] < g15[i]（预生成，Python 侧消费 RNG 保序）
//! 2. 能量过滤：energy[i] >= emit_cost
//! 3. 耗能：energy[i] -= emit_cost（在模式编码之前！）
//! 4. 模式编码：e_bin(2位) + f_bit(1位) + n_bit(1位) = 4位 pattern (0~15)
//!    - e_bin = clip(energy / max_energy * 4, 0, 3)
//!    - f_bit = resource_grid[flat] > 0.5 * resource_capacity[flat]
//!    - n_bit = densities[flat] > 1（同格个体数 > 1，不是邻居数）
//! 5. 写入：signal_marks[flat] = pattern; signal_age[flat] = duration
//!
//! 多 emitter 同格：按循环顺序后写覆盖先写（与 Python write_many 行为一致）。

/// 批量信号发射。
///
/// # 参数
/// - `flat`: 个体位置 (N,)
/// - `energy`: 能量 (N,)，就地修改（扣发射耗能）
/// - `g15`: 信号发射基因 (N,)
/// - `rand_emit`: 预生成随机数 (N,)，用于发射概率判定
/// - `densities`: 每格个体数 (n_cells,)
/// - `resource_grid`: 资源存量 (n_cells,)
/// - `resource_capacity`: 资源容量 (n_cells,)
/// - `signal_marks`: 信号场标记 (n_cells,) uint8，就地修改
/// - `signal_age`: 信号场年龄 (n_cells,) int32，就地修改
/// - `emit_cost`: 发射耗能（0.1）
/// - `max_energy`: 最大能量（300.0）
/// - `duration`: 信号持续时长（50）
///
/// # 返回
/// 发射个体数
pub fn signal_emit_batch(
    flat: &[i64],
    energy: &mut [f64],
    g15: &[f64],
    rand_emit: &[f64],
    densities: &[f64],
    resource_grid: &[f64],
    resource_capacity: &[f64],
    signal_marks: &mut [u8],
    signal_age: &mut [i32],
    emit_cost: f64,
    max_energy: f64,
    duration: i32,
) -> usize {
    let n = flat.len();
    debug_assert_eq!(energy.len(), n);
    debug_assert_eq!(g15.len(), n);
    debug_assert_eq!(rand_emit.len(), n);

    let mut n_emit = 0;
    let max_e = if max_energy > 0.0 { max_energy } else { 1.0 };

    for i in 0..n {
        // 1. 发射判定
        if rand_emit[i] >= g15[i] {
            continue;
        }
        // 2. 能量过滤
        if energy[i] < emit_cost {
            continue;
        }
        // 3. 耗能（必须在模式编码之前，与 Python 顺序一致）
        energy[i] -= emit_cost;

        let cell = flat[i] as usize;

        // 4. 模式编码
        // e_bin = clip(energy / max_energy * 4, 0, 3)
        let e_raw = energy[i] / max_e * 4.0;
        let e_bin = if e_raw < 0.0 {
            0u8
        } else if e_raw >= 4.0 {
            3u8
        } else {
            e_raw as u8
        };
        // f_bit = resource_grid[cell] > 0.5 * resource_capacity[cell]
        let f_bit = if resource_grid[cell] > 0.5 * resource_capacity[cell] {
            1u8
        } else {
            0u8
        };
        // n_bit = densities[cell] > 1（同格个体数 > 1）
        let n_bit = if densities[cell] > 1.0 { 1u8 } else { 0u8 };

        let pattern = e_bin * 4 + f_bit * 2 + n_bit;

        // 5. 写入信号场
        signal_marks[cell] = pattern;
        signal_age[cell] = if pattern > 0 { duration } else { 0 };

        n_emit += 1;
    }

    n_emit
}
