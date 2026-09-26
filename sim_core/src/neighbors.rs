//! P0.1（T1）：紧凑邻居表寻址。
//!
//! 旧版把"普通格 8 邻"与"极点格整带（cols 邻，极点与相邻纬度带整行相邻）"
//! 塞进同一张定长表 `(n_cells, max(8, cols))` ⇒ 480×960 下 3 538.9 MB
//! （占单 run 内存 94.7%）。
//!
//! 拆成两张表：主表 `(n_cells, 8)` + 极点带 `(2, cols)`。**邻居集合与顺序逐位不变**
//! （主表行 = 旧表该行前 8 列；极点带行 = 旧表极点格那一整行，同序）
//! ⇒ 下游 `valid_nb`/`prey_list` 顺序不变 ⇒ 预生成随机数消费不变 ⇒ 纯结构重构。

/// 取格 `c` 的邻居切片。
///
/// - 普通格：`main[c*nb_stride .. +nb_stride]`（恰 8 项，无 `-1`）
/// - 极点格：`pole` 的第 0 行（`pole_top`）或第 1 行（`pole_bottom`），整带 `n_cols` 项
///
/// `pole` 长度须为 `2 * n_cols`（lib.rs 侧已校验）。
#[inline]
pub fn nb_slice<'a>(
    main: &'a [i64],
    pole: &'a [i64],
    c: usize,
    n_cols: usize,
    nb_stride: usize,
    pole_top: usize,
    pole_bottom: usize,
) -> &'a [i64] {
    let row = c / n_cols;
    if row == pole_top {
        &pole[..n_cols]
    } else if row == pole_bottom {
        &pole[n_cols..2 * n_cols]
    } else {
        let base = c * nb_stride;
        &main[base..base + nb_stride]
    }
}
