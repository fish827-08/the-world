//! consume：资源场批量消耗（模块三 · 3.5）。
//!
//! 纯 Rust 数值核心（不依赖 pyo3 / numpy，可独立单测）。
//! 语义与 Python 侧 `world/resource_field.py::ResourceField.consume_many`
//! 逐位等价（引擎第 4 步进食的双路径之一）：
//!
//! ```text
//! cnt[cell]       = 该格本次被几只同时吃（bincount）
//! per_cell        = min(avail, 单只想吃量 × cnt[cell])      // 每格总消耗
//! share           = per_cell / max(1, cnt[cell])            // 每只实际分到
//! grid[cell]  -=  share（按 flats 数组顺序逐只扣，与 np.subtract.at 同序）
//! out_taken[i]    = share
//! ```
//!
//! 数值纪律（保证与 numpy 逐位一致）
//! ----------------------------------
//! - `min` 用 `a < b ? a : b`（对齐 `np.minimum` 的相等取 b 语义，-0.0/+0.0 一致）；
//! - 除法：`per_cell / max(1, cnt)` —— numpy 里 float64 / int64 先把 cnt 提升
//!   成 float64 再除，Rust 侧 `cnt as f64` 再做除法，IEEE754 结果相同；
//! - **先算完所有 share（读原始存量），再按 flats 顺序逐只累减**：与
//!   `np.subtract.at`（unbuffered，顺序累减）一致，同格多只时减法顺序逐位可复现；
//! - 只用 f64 四则与整数计数，不用超越函数。

/// 就地更新资源格存量（grid 被原地改写），把每只实吃量写进 out_taken。
///
/// 入参长度约定：`flats.len() == amounts.len() == out_taken.len()`；
/// 所有 `flats` 值必须落在 `0..grid.len()`（越界由绑定层校验，本函数假定合法）。
pub fn consume_many(
    grid: &mut [f64],
    flats: &[i64],
    amounts: &[f64],
    out_taken: &mut [f64],
) {
    debug_assert_eq!(flats.len(), amounts.len());
    debug_assert_eq!(flats.len(), out_taken.len());

    let n_cells = grid.len();
    let n = flats.len();

    // 1) 每格被几只同时吃（与 np.add.at(cnt, flats, 1) 等价，整数计数无舍入）
    let mut cnt = vec![0i64; n_cells];
    for &f in flats {
        cnt[f as usize] += 1;
    }

    // 2) 先读【原始存量】算每只实吃量（与 numpy `avail = grid[flats]`
    //    一次性快照一致——此时 grid 尚未被扣减）。
    for i in 0..n {
        let cell = flats[i] as usize;
        let avail = grid[cell];
        let per_cell = min_np(avail, amounts[i] * cnt[cell] as f64);
        let share = per_cell / (if cnt[cell] > 1 { cnt[cell] as f64 } else { 1.0 });
        out_taken[i] = share;
    }

    // 3) 按 flats 原始顺序逐只扣减（与 np.subtract.at 的 unbuffered 累减同序）。
    for i in 0..n {
        grid[flats[i] as usize] -= out_taken[i];
    }
}

/// 与 `np.minimum(a, b)` 等价：`a < b` 取 a，否则取 b（相等/±0.0 取 b）。
#[inline]
fn min_np(a: f64, b: f64) -> f64 {
    if a < b {
        a
    } else {
        b
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(grid: &mut [f64], flats: &[i64], amounts: &[f64]) -> Vec<f64> {
        let mut taken = vec![0.0; flats.len()];
        consume_many(grid, flats, amounts, &mut taken);
        taken
    }

    #[test]
    fn single_eater_full() {
        // 存量足够：吃多少给多少
        let mut grid = vec![10.0, 10.0];
        let taken = run(&mut grid, &[0], &[3.0]);
        assert_eq!(taken, vec![3.0]);
        assert_eq!(grid, vec![7.0, 10.0]);
    }

    #[test]
    fn single_eater_short() {
        // 存量不够：只能吃到空的，不欠账
        let mut grid = vec![2.0, 10.0];
        let taken = run(&mut grid, &[0], &[5.0]);
        assert_eq!(taken, vec![2.0]);
        assert_eq!(grid, vec![0.0, 10.0]);
    }

    #[test]
    fn shared_cell_equally_split() {
        // 两只在同一格，存量 12，各想吃 10 → 每格总消耗=min(12,10×2)=12，均分=6
        let mut grid = vec![12.0, 10.0];
        let taken = run(&mut grid, &[0, 0], &[10.0, 10.0]);
        assert_eq!(taken, vec![6.0, 6.0]);
        assert_eq!(grid, vec![0.0, 10.0]);
    }

    #[test]
    fn shared_cell_diff_amounts() {
        // 两只同格，存量 20，A 想吃 10、B 想吃 2：
        // per_cell_A = min(20, 10×2)=20 → 10；per_cell_B = min(20, 2×2)=4 → 2
        let mut grid = vec![20.0];
        let taken = run(&mut grid, &[0, 0], &[10.0, 2.0]);
        assert_eq!(taken, vec![10.0, 2.0]);
        assert_eq!(grid, vec![8.0]);
    }

    #[test]
    fn nonzero_take_on_floor() {
        // 存量 0：谁吃都是 0，格不变
        let mut grid = vec![0.0];
        let taken = run(&mut grid, &[0, 0], &[5.0, 5.0]);
        assert_eq!(taken, vec![0.0, 0.0]);
        assert_eq!(grid, vec![0.0]);
    }

    #[test]
    fn max_cap_scaling() {
        // 同格三只存量大到吃不完：每只都拿到自己的目标量
        let mut grid = vec![100.0];
        let taken = run(&mut grid, &[0, 0, 0], &[4.0, 4.0, 4.0]);
        assert_eq!(taken, vec![4.0, 4.0, 4.0]);
        assert_eq!(grid, vec![88.0]);
    }
}