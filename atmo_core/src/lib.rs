//! # atmo_core —— 大气动力核心（乙方案）Rust crate / PyO3 扩展
//!
//! **契约**：`atmo/INTERFACE.md` **FROZEN v1.0.2**。
//! **分支** `feat/atmo-core` ｜ worktree `<worktree>`。
//!
//! ## 当前状态（W1，2026-10-20）
//!
//! | 模块 | 状态 |
//! |---|---|
//! | `detmath`（确定性 sin/cos/atan/asin） | ✅ |
//! | `grid`（Gaussian/规则、`mu_edges`、`cosbar`、科氏、sigma） | ✅ v1.0.1 的 `mu_edges` 已补 |
//! | `ops`（散度/涡度/梯度/拉普拉斯） | ✅ W1（有限体积，全球平均散度精确为零） |
//! | `solar`（轨道 → cosZ → 大气顶短波） | ✅ W1（纯解析，禁表） |
//! | `shallow`（2 层浅水 + RK3 + 极区滤波 + CFL） | ✅ W1（独立入口，见 §8 之外） |
//! | `export`（t2m/u10/v10） | ✅ v1.0.2 |
//! | `step`（8 层原始方程 + Held–Suarez） | 🟡 W2 |
//!
//! ## 硬约束（摘自 INTERFACE §0）
//!
//! * 纬度 `j=0` = 最南（升序）；经度 `i=0` = 0°E（东向、周期）。
//! * `float64` / C 序 / `(nlev, nlat, nlon)`；SI 单位。
//! * `w ≡ σ̇`（s⁻¹），不是 m/s。
//! * 默认关：本 crate 不 import 任何 `world/` 模块。
//! * 禁在核心路径写"日照分布表""气候态表""预设图案"。

mod detmath;
mod export;
mod filt;
mod grid;
mod ops;
mod prim;
mod shallow;
mod solar;
mod state;
mod step;

use pyo3::prelude::*;
use pyo3::types::PyDict;

/// 接口契约版本（与 `atmo/INTERFACE.md` 同号）
pub const INTERFACE_VERSION: &str = "1.0.2";

/// 契约自检：A2/A3/AV 在启动时断言用。
#[pyfunction]
fn contract<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("interface_version", INTERFACE_VERSION)?;
    d.set_item("array_dtype", "float64")?;
    d.set_item("array_order", "C")?;
    d.set_item("array_shape_3d", vec!["nlev", "nlat", "nlon"])?;
    d.set_item("index_order", "k,j,i")?;
    d.set_item("lat_order", "ascending_south_to_north")?;
    d.set_item("lon_range", "[0,360)")?;
    d.set_item("prognostic_fields", state::PROGNOSTIC.to_vec())?;
    d.set_item("digest_fields", state::PROGNOSTIC.to_vec())?;
    d.set_item("digest_algo", "fnv1a-64-le")?;
    d.set_item("w_is_sigma_dot", true)?;
    d.set_item("units", {
        let u = PyDict::new(py);
        u.set_item("u", "m/s")?;
        u.set_item("v", "m/s")?;
        u.set_item("w", "1/s (sigma_dot)")?;
        u.set_item("T", "K")?;
        u.set_item("p", "Pa")?;
        u.set_item("ps", "Pa")?;
        u.set_item("q", "kg/kg")?;
        u.set_item("sigma", "1")?;
        u
    })?;
    // 已实现 / 未实现（供 A3 在启动时判断能用到哪一步）
    d.set_item(
        "implemented",
        vec![
            "grid_spec",
            "gaussian_latitudes",
            "make_state",
            "digest",
            "divergence",
            "vorticity",
            "gradient",
            "laplacian",
            "coriolis",
            "solar",
            "export",
            "shallow_water_step",
            "shallow_water_rhs",
            "step",
        ],
    )?;
    d.set_item("not_implemented", Vec::<String>::new())?;
    Ok(d)
}

#[pymodule]
fn atmo_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("INTERFACE_VERSION", INTERFACE_VERSION)?;

    m.add_function(wrap_pyfunction!(contract, m)?)?;
    m.add_function(wrap_pyfunction!(grid::grid_spec, m)?)?;
    m.add_function(wrap_pyfunction!(grid::gaussian_latitudes, m)?)?;
    m.add_function(wrap_pyfunction!(state::make_state, m)?)?;
    m.add_function(wrap_pyfunction!(state::digest, m)?)?;
    m.add_function(wrap_pyfunction!(ops::divergence, m)?)?;
    m.add_function(wrap_pyfunction!(ops::vorticity, m)?)?;
    m.add_function(wrap_pyfunction!(ops::gradient, m)?)?;
    m.add_function(wrap_pyfunction!(ops::laplacian, m)?)?;
    m.add_function(wrap_pyfunction!(ops::coriolis, m)?)?;
    m.add_function(wrap_pyfunction!(solar::solar, m)?)?;
    m.add_function(wrap_pyfunction!(export::export, m)?)?;
    m.add_function(wrap_pyfunction!(shallow::shallow_water_step, m)?)?;
    m.add_function(wrap_pyfunction!(shallow::shallow_water_rhs, m)?)?;
    m.add_function(wrap_pyfunction!(prim::step, m)?)?;

    m.add_class::<step::AtmoCore>()?;
    Ok(())
}
