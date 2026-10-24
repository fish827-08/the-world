//! `AtmoCore` —— 模式实例（几何 + 配置），与 `step()` 推进入口。
//!
//! 设计要点（`atmo/INTERFACE.md` §7.5）：
//! * `step()` 是**纯函数式**的：同样的 `(state, forcing, dt)` ⇒ 逐位同样的输出；
//!   内部不回调 Python、不读可变全局状态。
//! * 只在**耦合步边界**与外部交换场；子步内不重读边界。
//! * 原地更新 `state`（OPEN-5 的建议口径），返回 `diag` 字典。
//!
//! 状态：W0 = 骨架（签名 + 错误信息），W1 = 2 层浅水，W2–W3 = 8 层原始方程。

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::grid::GridSpec;

#[pyclass(module = "atmo_core")]
pub struct AtmoCore {
    pub(crate) grid: GridSpec,
}

#[pymethods]
impl AtmoCore {
    /// `AtmoCore(nlat=32, nlon=64, nlev=8, rotation_period_s=86400.0)`
    #[new]
    #[pyo3(signature = (nlat=32, nlon=64, nlev=8, rotation_period_s=86400.0))]
    fn new(nlat: usize, nlon: usize, nlev: usize, rotation_period_s: f64) -> PyResult<Self> {
        Ok(AtmoCore {
            grid: GridSpec::new(nlat, nlon, nlev, "gaussian", rotation_period_s),
        })
    }

    #[getter]
    fn nlat(&self) -> usize {
        self.grid.nlat
    }
    #[getter]
    fn nlon(&self) -> usize {
        self.grid.nlon
    }
    #[getter]
    fn nlev(&self) -> usize {
        self.grid.nlev
    }

    /// `diag = step(state, forcing, dt)` —— 推进一个子步（原地改 `state`）
    #[pyo3(signature = (state, forcing=None, dt=900.0))]
    fn step<'py>(
        &self,
        py: Python<'py>,
        state: &Bound<'py, PyDict>,
        forcing: Option<&Bound<'py, PyDict>>,
        dt: f64,
    ) -> PyResult<Bound<'py, PyDict>> {
        crate::prim::step_impl(py, state, forcing, dt, Some(&self.grid))
    }

    /// 网格字典（与 `atmo_core.grid_spec` 同构）
    fn grid<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        crate::grid::grid_spec(
            py,
            self.grid.nlat,
            self.grid.nlon,
            self.grid.nlev,
            "gaussian",
            self.grid.rotation_period_s,
        )
    }

    fn __repr__(&self) -> String {
        format!(
            "AtmoCore(nlat={}, nlon={}, nlev={}, kind={}, rotation_period_s={})",
            self.grid.nlat,
            self.grid.nlon,
            self.grid.nlev,
            self.grid.kind,
            self.grid.rotation_period_s
        )
    }
}
