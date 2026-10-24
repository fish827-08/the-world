//! 状态数组的分配、布局校验与确定性摘要。
//!
//! 布局（`atmo/INTERFACE.md` §0/§4，**改动即作废旧 digest**）：
//! * `float64`、C 序、`(nlev, nlat, nlon)`；索引 `arr[k, j, i]`。
//! * prognostic：`u, v, w, T, ps, q`；`p` 是诊断量（每次 step 重算，**不进 digest**）。
//! * digest = FNV-1a 64，按 `u, v, w, T, ps, q` 的固定顺序、`k→j→i` 的逻辑序遍历每个
//!   `f64` 的 little-endian 8 字节。跨平台应逐位一致。

use numpy::ndarray::Array3;
use numpy::{IntoPyArray, PyReadonlyArrayDyn};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::grid::{GridSpec, SIGMA_HALF};

/// prognostic 字段（digest 的覆盖集合，**顺序固定**）
pub const PROGNOSTIC: [&str; 6] = ["u", "v", "w", "T", "ps", "q"];
/// 3D 的 prognostic 字段（`ps` 是 2D）
pub const PROGNOSTIC_3D: [&str; 5] = ["u", "v", "w", "T", "q"];

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// FNV-1a 64（纯算术 ⇒ 跨平台一致）
pub fn fnv1a_update(mut h: u64, bytes: &[u8]) -> u64 {
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

#[allow(dead_code)]
pub fn fnv1a(bytes: &[u8]) -> u64 {
    fnv1a_update(FNV_OFFSET, bytes)
}

fn get_usize(g: &Bound<'_, PyDict>, key: &str) -> PyResult<usize> {
    g.get_item(key)?
        .ok_or_else(|| PyValueError::new_err(format!("网格字典缺键 `{key}`")))?
        .extract::<usize>()
}

/// 按网格分配一套状态数组（`atmo/INTERFACE.md` §8.2）。
#[pyfunction]
#[allow(non_snake_case)]
#[pyo3(signature = (grid, T0=288.0, ps0=101325.0))]
pub fn make_state<'py>(
    py: Python<'py>,
    grid: &Bound<'py, PyDict>,
    T0: f64,
    ps0: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let nlat = get_usize(grid, "nlat")?;
    let nlon = get_usize(grid, "nlon")?;
    let nlev = get_usize(grid, "nlev")?;
    if nlev != SIGMA_HALF.len() - 1 {
        return Err(PyValueError::new_err(format!(
            "v0 只支持 nlev={}（当前 {nlev}；升层数见 INTERFACE §11 OPEN-6）",
            SIGMA_HALF.len() - 1
        )));
    }
    if !(T0.is_finite() && ps0.is_finite() && T0 > 0.0 && ps0 > 0.0) {
        return Err(PyValueError::new_err("T0/ps0 必须是有限正数"));
    }

    let out = PyDict::new(py);
    for name in PROGNOSTIC_3D {
        let a = Array3::<f64>::zeros((nlev, nlat, nlon));
        out.set_item(name, a.into_pyarray(py))?;
    }
    // T 填初值（均匀）
    {
        let t = Array3::<f64>::from_elem((nlev, nlat, nlon), T0);
        out.set_item("T", t.into_pyarray(py))?;
    }
    // ps（2D）
    {
        let ps = ndarray_zeros2(nlat, nlon, ps0);
        out.set_item("ps", ps.into_pyarray(py))?;
    }
    // p（3D，诊断量）：p[k] = ps0 · sigma_full[k]
    {
        let mut p = Array3::<f64>::zeros((nlev, nlat, nlon));
        for k in 0..nlev {
            let sk = 0.5 * (SIGMA_HALF[k] + SIGMA_HALF[k + 1]);
            for j in 0..nlat {
                for i in 0..nlon {
                    p[[k, j, i]] = ps0 * sk;
                }
            }
        }
        out.set_item("p", p.into_pyarray(py))?;
    }
    Ok(out)
}

fn ndarray_zeros2(nlat: usize, nlon: usize, fill: f64) -> numpy::ndarray::Array2<f64> {
    numpy::ndarray::Array2::<f64>::from_elem((nlat, nlon), fill)
}

/// 确定性摘要（16 位十六进制）。字段顺序固定，缺失字段报错。
#[pyfunction]
#[pyo3(signature = (state, fields=None))]
pub fn digest(state: &Bound<'_, PyDict>, fields: Option<Vec<String>>) -> PyResult<String> {
    let list: Vec<String> = match fields {
        Some(f) => f,
        None => PROGNOSTIC.iter().map(|s| s.to_string()).collect(),
    };
    let mut h = FNV_OFFSET;
    for name in &list {
        let item = state
            .get_item(name.as_str())?
            .ok_or_else(|| PyValueError::new_err(format!("状态缺字段 `{name}`")))?;
        let arr = item.extract::<PyReadonlyArrayDyn<f64>>().map_err(|_| {
            PyValueError::new_err(format!("字段 `{name}` 必须是 float64 ndarray（C 序建议）"))
        })?;
        // 字段名参与哈希 ⇒ 换顺序/换名字会改变结果（有意为之）
        h = fnv1a_update(h, name.as_bytes());
        for &v in arr.as_array().iter() {
            h = fnv1a_update(h, &v.to_le_bytes());
        }
    }
    Ok(format!("{h:016x}"))
}

/// 网格几何的摘要（用于"两边网格是否同一套"的快速断言）。
#[allow(dead_code)]
pub fn grid_digest(g: &GridSpec) -> String {
    let mut h = FNV_OFFSET;
    for &v in &g.mu {
        h = fnv1a_update(h, &v.to_le_bytes());
    }
    for &v in &g.dmu {
        h = fnv1a_update(h, &v.to_le_bytes());
    }
    h = fnv1a_update(h, &(g.nlon as u64).to_le_bytes());
    format!("{h:016x}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fnv_known_vectors() {
        // FNV-1a 64 的标准测试向量
        assert_eq!(fnv1a(b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(fnv1a(b"a"), 0xaf63_dc4c_8601_ec8c);
        assert_eq!(fnv1a(b"foobar"), 0x85944171f73967e8);
    }

    #[test]
    fn fnv_is_order_sensitive() {
        let a = fnv1a(&[1.0f64.to_le_bytes(), 2.0f64.to_le_bytes()].concat());
        let b = fnv1a(&[2.0f64.to_le_bytes(), 1.0f64.to_le_bytes()].concat());
        assert_ne!(a, b);
    }

    #[test]
    fn fnv_distinguishes_one_ulp() {
        let a = fnv1a(&1.0f64.to_le_bytes());
        let b = fnv1a(&f64::from_bits(1.0f64.to_bits() + 1).to_le_bytes());
        assert_ne!(a, b);
    }
}
