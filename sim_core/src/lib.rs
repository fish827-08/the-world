//! sim_core：digital-life-sphere 的 Rust 加速核（模块三）。
//!
//! 模块职责（对齐 PROJECT-DESCRIPTION.md 3.3）
//! -------------------------------------------
//! Python 引擎把"真正发热的批量数值运算"下沉到这里：
//!   - `regrow`：资源场再生（3.2 已完成，与 ResourceField 逐位等价）；
//!   - `step_vectors`：种群数值管线（3.3 移植中）。
//!
//! 边界约定
//! ---------
//! Rust 只算"数值"，不做"决策"：出生判定、世代/ID 维护、清理尸体都在
//! Python 侧。函数输入输出均为借用数组（与 numpy 零拷贝、就地更新）。
//!
//! 组织方式：`src/regrow.rs` / `src/step_vectors.rs` 是纯 Rust 数值核心
//! （不依赖 pyo3，可独立单测）；本文件只做 pyo3 绑定层（切片 ↔ numpy 数组）。
//! 温度/光照等环境量目前仍由 Python 侧计算后传入。
mod consume;
mod culture;
mod genes;
mod l4_l5;
mod movement;
mod pleasure;
mod predation;
mod regrow;
mod reproduction;
mod signal;
mod step_vectors;
mod dispersal;
mod light_temp;

/// 基因位索引常量（与 Python simulation.genes 注册表对应），供绑定层对外导出
// 说明：常量定义在 genes.rs（G_MOVE_PROB 等），这里仅 re-export 供 Python 侧
// 校验函数比对（validate_gene_wiring 直接引用 crate::genes::* 即可，无需 re-export）。
pub use genes::G_AGGRESSION;
pub use genes::G_PERCEPTION;
pub use genes::G_SOCIABILITY;

use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

/// 导出 Rust 侧基因索引常量，供 Python 侧与 simulation.genes 注册表逐位对照。
/// 返回 [(Rust 常量名, Rust 值, Python 语义名), ...]；值不一致时引擎初始化
/// 会直接报错，防止双写漂移静默变成错误行为。
#[pyfunction]
fn native_gene_indicators() -> Vec<(String, usize)> {
    use crate::genes::*;
    vec![
        (stringify!(G_SOCIABILITY).to_string(), G_SOCIABILITY),
        (stringify!(G_PERCEPTION).to_string(), G_PERCEPTION),
        (stringify!(G_AGGRESSION).to_string(), G_AGGRESSION),
    ]
}

/// 完整基因双写校验：Python 侧传入全部基因的 (name, value)，与 Rust 侧常量逐位对照。
///
/// 返回不一致列表 [(name, rust_value, py_value), ...]；空列表表示全部一致。
/// 用于 tests/test_genes_registry.py 的漂移检测：故意改一侧索引→此函数返回非空。
#[pyfunction]
fn validate_gene_wiring(
    py_names: Vec<String>,
    py_values: Vec<usize>,
) -> Vec<(String, usize, usize)> {
    use crate::genes::*;
    // Rust 侧全部常量（与 genes.rs 一一对应，新增基因时必须同步追加）
    let rust_all: Vec<(&str, usize)> = vec![
        ("G_MOVE_PROB", G_MOVE_PROB),
        ("G_METABOLIC", G_METABOLIC),
        ("G_REPRO_THRESHOLD", G_REPRO_THRESHOLD),
        ("G_LIFE_GENE", G_LIFE_GENE),
        ("G_EAT_AMOUNT", G_EAT_AMOUNT),
        ("G_STOMACH_CAP", G_STOMACH_CAP),
        ("G_MOVE_COST", G_MOVE_COST),
        ("G_PARENTAL_INVEST", G_PARENTAL_INVEST),
        ("G_PHOTOSYNTHESIS", G_PHOTOSYNTHESIS),
        ("G_HOMEOTHERM", G_HOMEOTHERM),
        ("G_FORAGE_NEIGHBOR", G_FORAGE_NEIGHBOR),
        ("G_TEMP_PREF", G_TEMP_PREF),
        ("G_REPRO_COOLDOWN", G_REPRO_COOLDOWN),
        ("G_SOCIABILITY", G_SOCIABILITY),
        ("G_PERCEPTION", G_PERCEPTION),
        ("G_SIGNAL_STRENGTH", G_SIGNAL_STRENGTH),
        ("G_AGGRESSION", G_AGGRESSION),
        ("G_DIET", G_DIET),
        ("G_DEFENSE", G_DEFENSE),
        ("G_ROOTING", G_ROOTING),
        ("G_PERSISTENCE", G_PERSISTENCE),
        ("G_GIVE_UP", G_GIVE_UP),
        ("G_TRUST_GENE", G_TRUST_GENE),
        ("G_MIGRATE_BIAS", G_MIGRATE_BIAS),
    ];

    let mut drift = Vec::new();
    if py_names.len() != rust_all.len() {
        // 数量不一致本身就是漂移（但不 panic，返回差异供调用方判断）
        drift.push((
            format!("COUNT_MISMATCH(rust={}, py={})", rust_all.len(), py_names.len()),
            rust_all.len(),
            py_names.len(),
        ));
    }
    let n = py_names.len().min(rust_all.len());
    for i in 0..n {
        let (r_name, r_val) = rust_all[i];
        let p_name = &py_names[i];
        let p_val = py_values[i];
        // 名字比对：Python 侧传 "MOVE_PROB"，Rust 侧是 "G_MOVE_PROB"，去掉 G_ 前缀
        let r_name_stripped = r_name.strip_prefix("G_").unwrap_or(r_name);
        if r_name_stripped != p_name.as_str() || r_val != p_val {
            // 🔴 必须区分「名字不一致」与「取值不一致」：
            //    实测事故（2026-09-26 云归）—— `sim_core` 编译产物早于 cb00da9
            //    （该提交把 20/21 号位从 G_HEDONISM/G_PROCESSING 改名为
            //    G_PERSISTENCE/G_GIVE_UP，**取值不变**）⇒ 陈旧 .so 报出
            //    ("PERSISTENCE", 20, 20) 这种"值相同却判漂移"的条目，
            //    光看元组**无法归因**，白花一轮排查。
            //    ⇒ 名字不一致时把 Rust 侧名字写进标签，一眼可辨（= 该重编 .so）。
            let kind = if r_name_stripped != p_name.as_str() {
                format!("NAME_MISMATCH(rust={})=>REBUILD_SIM_CORE", r_name_stripped)
            } else {
                "VALUE_MISMATCH".to_string()
            };
            drift.push((format!("{}|{}", p_name, kind), r_val, p_val));
        }
    }
    drift
}

/// fruit_charge：植物蓄力→结果（L10a 步骤 3.5），确定性数值管线下沉。
///
/// genes 为 (P, gene_count) 展平的一维数组；fruit_charge/fruit_grid 就地修改。
#[pyfunction]
fn fruit_charge(
    flat: PyReadonlyArray1<'_, i64>,
    genes: PyReadonlyArray1<'_, f64>,
    gene_count: usize,
    fruit_charge: Bound<'_, PyArray1<f64>>,
    fruit_grid: Bound<'_, PyArray1<f64>>,
    g19_idx: usize,
    g8_idx: usize,
    plant_threshold: f64,
    charge_rate: f64,
    fruit_threshold: f64,
    fruit_ratio: f64,
) -> PyResult<()> {
    let p = flat.as_array().len();
    require_len("genes", genes.as_array().len(), p * gene_count)?;
    require_len("fruit_charge", unsafe { fruit_charge.as_array().len() }, p)?;

    let mut fc = unsafe { fruit_charge.as_slice_mut()? };
    let mut fg = unsafe { fruit_grid.as_slice_mut()? };

    dispersal::fruit_charge_batch(
        flat.as_slice()?, genes.as_slice()?, gene_count,
        &mut fc, &mut fg,
        g19_idx, g8_idx, plant_threshold, charge_rate, fruit_threshold, fruit_ratio,
    );
    Ok(())
}

/// eat_fruit：动物吃果实→能量转移（L10a 步骤 4.5），确定性数值管线下沉。
///
/// genes 为 (P, gene_count) 展平；energy/fruit_grid 就地修改。
#[pyfunction]
fn eat_fruit(
    flat: PyReadonlyArray1<'_, i64>,
    genes: PyReadonlyArray1<'_, f64>,
    gene_count: usize,
    energy: Bound<'_, PyArray1<f64>>,
    fruit_grid: Bound<'_, PyArray1<f64>>,
    g19_idx: usize,
    plant_threshold: f64,
    eat_rate: f64,
    digest_ratio: f64,
) -> PyResult<()> {
    let p = flat.as_array().len();
    require_len("genes", genes.as_array().len(), p * gene_count)?;
    require_len("energy", unsafe { energy.as_array().len() }, p)?;

    let mut e = unsafe { energy.as_slice_mut()? };
    let mut fg = unsafe { fruit_grid.as_slice_mut()? };

    dispersal::eat_fruit_batch(
        flat.as_slice()?, genes.as_slice()?, gene_count,
        &mut e, &mut fg,
        g19_idx, plant_threshold, eat_rate, digest_ratio,
    );
    Ok(())
}

/// regrow：资源场再生（3.2）。
///
/// 语义与 `world/resource_field.py::ResourceField.regrow` 逐位等价：
/// 就地更新 `grid`（第一参数，numpy 数组），返回 None。
/// `temperature` 由 Python 侧光照温度场计算后传入（模块三暂不搬温度场）。
#[pyfunction]
#[pyo3(name = "regrow")]  // 导出名 regrow；内部函数名避免与 mod regrow 冲突
fn regrow_rs(
    grid: Bound<'_, PyArray1<f64>>,
    capacity: PyReadonlyArray1<'_, f64>,
    temperature: PyReadonlyArray1<'_, f64>,
    regrow_rate: f64,
    temp_sensitivity: f64,
) -> PyResult<()> {
    let mut g = unsafe { grid.as_slice_mut()? };
    let cap = capacity.as_slice()?;
    let temp = temperature.as_slice()?;
    regrow::regrow(&mut g, cap, temp, regrow_rate, temp_sensitivity);
    Ok(())
}

/// 缺失长度校验：把 `name` 数组的期望长度核对到 n，否则抛 ValueError。
fn require_len(name: &str, got: usize, want: usize) -> PyResult<()> {
    if got != want {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{name}: 长度 {got} 与个体数 {want} 不一致"
        )));
    }
    Ok(())
}

/// regrow_patchy：patchy 模式资源再生（L7e，3.2 的斑块化推广）。
///
/// 语义与 `ResourceField._regrowth_amount`（distribution="patchy"）逐位等价：
/// 每格恢复量 = 基准再生率 × 温度因子 × 空间倍率（patch/bg），min 到容量。
/// `patch_mask` 为 (n_cells,) uint8（1=斑块格），就地更新 `grid`。
#[pyfunction]
#[pyo3(name = "regrow_patchy")]  // 导出名 regrow_patchy；内部函数名避免与 mod regrow::regrow_patchy 混淆
fn regrow_patchy_rs(
    grid: Bound<'_, PyArray1<f64>>,
    capacity: PyReadonlyArray1<'_, f64>,
    temperature: PyReadonlyArray1<'_, f64>,
    patch_mask: PyReadonlyArray1<'_, u8>,
    patch_regrowth_mult: f64,
    bg_regrowth_mult: f64,
    regrow_rate: f64,
    temp_sensitivity: f64,
) -> PyResult<()> {
    let mut g = unsafe { grid.as_slice_mut()? };
    let cap = capacity.as_slice()?;
    let temp = temperature.as_slice()?;
    let mask = patch_mask.as_slice()?;
    let n = g.len();
    require_len("capacity", cap.len(), n)?;
    require_len("temperature", temp.len(), n)?;
    require_len("patch_mask", mask.len(), n)?;
    regrow::regrow_patchy(
        &mut g, cap, temp, mask, patch_regrowth_mult, bg_regrowth_mult,
        regrow_rate, temp_sensitivity,
    );
    Ok(())
}

/// consume_many：资源场批量消耗（3.5）。
///
/// 语义与 `ResourceField.consume_many` 逐位等价：按 `flats` 逐只把
/// `amounts[i]`（想吃的量）从 `grid[flats[i]]` 扣掉（同格均分、绝不欠账），
/// 实吃量写回 `out_taken`；`grid` 就地更新、返回 None。
#[pyfunction]
fn consume_many(
    grid: Bound<'_, PyArray1<f64>>,
    flats: PyReadonlyArray1<'_, i64>,
    amounts: PyReadonlyArray1<'_, f64>,
    out_taken: Bound<'_, PyArray1<f64>>,
) -> PyResult<()> {
    let n_cells = unsafe { grid.as_array().len() };
    let n = flats.as_array().len();
    require_len("amounts", amounts.as_array().len(), n)?;
    require_len("out_taken", unsafe { out_taken.as_array().len() }, n)?;

    let f = flats.as_slice()?;
    // 越界校验：任何 flat 不在 [0, n_cells) 内都直接报错，绝不进入
    // 索引运算（防 Rust slice 越界 panic / UB）。
    let idx_max = n_cells as i64 - 1;
    for &x in f {
        if x < 0 || x > idx_max {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "flats: 索引 {x} 超出资源格范围 [0, {n_cells})"
            )));
        }
    }

    let mut g = unsafe { grid.as_slice_mut()? };
    let mut out = unsafe { out_taken.as_slice_mut()? };
    consume::consume_many(&mut g, f, amounts.as_slice()?, &mut out);
    Ok(())
}

/// step_vectors_stage1：第 1~3 步（光合/代谢/维持），就地更新 energy、stomach。
///
/// 语义与 `sphere_engine._step_population` 第 1、2、3 步逐位一致（见
/// `src/step_vectors.rs` 说明）。所有环境量（eff_activity / illumination）
/// 由 Python 侧先算好传入；age 只读（i64，+1 由 stage2 完成）。
#[pyfunction]
fn step_vectors_stage1(
    energy: Bound<'_, PyArray1<f64>>,
    stomach: Bound<'_, PyArray1<f64>>,
    genes: PyReadonlyArray2<'_, f64>,
    eff_activity: PyReadonlyArray1<'_, f64>,
    illumination: PyReadonlyArray1<'_, f64>,
    age: PyReadonlyArray1<'_, i64>,
    photo_max: f64,
    base_metabolism: f64,
    eat_efficiency: f64,
    growth_mult: f64,
    senile_mult: f64,
    maturity_fraction: f64,
    senile_fraction: f64,
    homeo_upkeep: f64,
    day_length: f64,
) -> PyResult<()> {
    let n = unsafe { energy.as_array().len() };
    require_len("stomach", unsafe { stomach.as_array().len() }, n)?;
    require_len("eff_activity", eff_activity.as_array().len(), n)?;
    require_len("illumination", illumination.as_array().len(), n)?;
    require_len("age", age.as_array().len(), n)?;

    let g_arr = genes.as_array();
    let sh = g_arr.shape();
    let (n_rows, gene_count) = (sh[0], sh[1]);
    if n_rows != n || gene_count == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "genes: 形状 ({n_rows}, {gene_count}) 与个体数 {n} 不符"
        )));
    }
    let g_slice = g_arr.as_slice().ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("genes: 需要 C 连续（行主序）数组")
    })?;

    let mut e = unsafe { energy.as_slice_mut()? };
    let mut s = unsafe { stomach.as_slice_mut()? };
    let (ea, il, ag) = (eff_activity.as_slice()?, illumination.as_slice()?, age.as_slice()?);

    step_vectors::step_vectors_stage1(
        &mut e, &mut s, g_slice, gene_count, ea, il, ag,
        photo_max, base_metabolism, eat_efficiency,
        growth_mult, senile_mult, maturity_fraction, senile_fraction,
        homeo_upkeep, day_length,
    );
    Ok(())
}

/// step_vectors_stage2：第 5~8 步（移动扣费/年龄/死亡/冷却/繁殖候选）。
///
/// 在 Python 完成进食与 RNG 抽签后调用。`moved_raw` 是本次"要不要走"的
/// 抽样（Python 抽），能量门槛在本函数内置判定；四个 out_* 掩码由调用方
/// 预分配（长度 n），供 Python 做出生与死亡分流。
#[pyfunction]
fn step_vectors_stage2(
    energy: Bound<'_, PyArray1<f64>>,
    age: Bound<'_, PyArray1<i64>>,
    cooldown: Bound<'_, PyArray1<f64>>,
    genes: PyReadonlyArray2<'_, f64>,
    moved_raw: PyReadonlyArray1<'_, bool>,
    move_cost_ind: PyReadonlyArray1<'_, f64>,
    out_moved: Bound<'_, PyArray1<bool>>,
    out_starved: Bound<'_, PyArray1<bool>>,
    out_expired: Bound<'_, PyArray1<bool>>,
    out_repro: Bound<'_, PyArray1<bool>>,
    day_length: f64,
    maturity_fraction: f64,
    max_energy: f64,
) -> PyResult<()> {
    let n = unsafe { energy.as_array().len() };
    require_len("age", unsafe { age.as_array().len() }, n)?;
    require_len("cooldown", unsafe { cooldown.as_array().len() }, n)?;
    require_len("moved_raw", moved_raw.as_array().len(), n)?;
    require_len("move_cost_ind", move_cost_ind.as_array().len(), n)?;
    require_len("out_moved", unsafe { out_moved.as_array().len() }, n)?;
    require_len("out_starved", unsafe { out_starved.as_array().len() }, n)?;
    require_len("out_expired", unsafe { out_expired.as_array().len() }, n)?;
    require_len("out_repro", unsafe { out_repro.as_array().len() }, n)?;

    let g_arr = genes.as_array();
    let sh = g_arr.shape();
    let (n_rows, gene_count) = (sh[0], sh[1]);
    if n_rows != n || gene_count == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "genes: 形状 ({n_rows}, {gene_count}) 与个体数 {n} 不符"
        )));
    }
    let g_slice = g_arr.as_slice().ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("genes: 需要 C 连续（行主序）数组")
    })?;

    let mut e = unsafe { energy.as_slice_mut()? };
    let mut a = unsafe { age.as_slice_mut()? };
    let mut c = unsafe { cooldown.as_slice_mut()? };
    let (mr, mci) = (moved_raw.as_slice()?, move_cost_ind.as_slice()?);
    let mut om = unsafe { out_moved.as_slice_mut()? };
    let mut os = unsafe { out_starved.as_slice_mut()? };
    let mut oe = unsafe { out_expired.as_slice_mut()? };
    let mut orr = unsafe { out_repro.as_slice_mut()? };

    step_vectors::step_vectors_stage2(
        &mut e, &mut a, &mut c, g_slice, gene_count, mr, mci,
        &mut om, &mut os, &mut oe, &mut orr,
        day_length, maturity_fraction, max_energy,
    );
    Ok(())
}

/// culture_learn：文化学习（L5），幼体向邻格成体学习信号解读表。
///
/// 语义与 sphere_engine 步骤 6.5 逐位等价。interpret (N,16) 就地更新。
/// neighbors 是展平的 (n_cells*8,) 邻居表（普通格 8 邻，极点格可能含重复/负值）。
#[pyfunction]
fn culture_learn(
    interpret: Bound<'_, PyArray2<f64>>,
    flat: PyReadonlyArray1<'_, i64>,
    age: PyReadonlyArray1<'_, i64>,
    maturity_age: PyReadonlyArray1<'_, f64>,
    neighbors: PyReadonlyArray1<'_, i64>,
    n_cells: usize,
    nb_stride: usize,
    alpha: f64,
) -> PyResult<()> {
    let n = flat.as_array().len();
    require_len("age", age.as_array().len(), n)?;
    require_len("maturity_age", maturity_age.as_array().len(), n)?;

    let interp_arr = unsafe { interpret.as_array() };
    let sh = interp_arr.shape();
    if sh[0] != n || sh[1] != 16 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "interpret: 形状 ({}, {}) 应为 ({}, 16)",
            sh[0], sh[1], n
        )));
    }

    // neighbors 长度应为 n_cells * nb_stride
    let nb_len = neighbors.as_array().len();
    if nb_len != n_cells * nb_stride {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "neighbors: 长度 {} 应为 n_cells*nb_stride = {}",
            nb_len,
            n_cells * nb_stride
        )));
    }

    let mut interp = unsafe { interpret.as_slice_mut()? };
    culture::culture_learn(
        &mut interp,
        flat.as_slice()?,
        age.as_slice()?,
        maturity_age.as_slice()?,
        neighbors.as_slice()?,
        n_cells,
        nb_stride,
        alpha,
    );
    Ok(())
}

/// predation_attack：捕食（L4），饥饿驱动攻击邻格猎物。
///
/// 语义与 sphere_engine 步骤 5.5 逐位等价。energy/stomach/predation_mask 就地更新。
/// attack_mask 由 Python 侧按攻击概率+能量门槛筛选；rand_prey/rand_success 由
/// Python 侧预生成，保证 RNG 消费顺序与 Python 实现一致。
#[pyfunction]
fn predation_attack(
    energy: Bound<'_, PyArray1<f64>>,
    stomach: Bound<'_, PyArray1<f64>>,
    predation_mask: Bound<'_, PyArray1<bool>>,
    flat: PyReadonlyArray1<'_, i64>,
    genes: PyReadonlyArray2<'_, f64>,
    attackers: PyReadonlyArray1<'_, i64>,
    rand_prey: PyReadonlyArray1<'_, i64>,
    rand_success: PyReadonlyArray1<'_, f64>,
    neighbors: PyReadonlyArray1<'_, i64>,
    n_cells: usize,
    nb_stride: usize,
    max_energy: f64,
    eat_efficiency: f64,
) -> PyResult<()> {
    let n = flat.as_array().len();
    require_len("energy", unsafe { energy.as_array().len() }, n)?;
    require_len("stomach", unsafe { stomach.as_array().len() }, n)?;
    require_len("predation_mask", unsafe { predation_mask.as_array().len() }, n)?;
    let n_att = attackers.as_array().len();
    require_len("rand_prey", rand_prey.as_array().len(), n_att)?;
    require_len("rand_success", rand_success.as_array().len(), n_att)?;

    let g_arr = genes.as_array();
    let sh = g_arr.shape();
    let (n_rows, gene_count) = (sh[0], sh[1]);
    if n_rows != n || gene_count <= 16 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "genes: 形状 ({n_rows}, {gene_count}) 与个体数 {n} 不符（需含 g16）"
        )));
    }
    let g_slice = g_arr.as_slice().ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("genes: 需要 C 连续（行主序）数组")
    })?;

    let nb_len = neighbors.as_array().len();
    if nb_len != n_cells * nb_stride {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "neighbors: 长度 {nb_len} 应为 n_cells*nb_stride = {}",
            n_cells * nb_stride
        )));
    }

    let mut e = unsafe { energy.as_slice_mut()? };
    let mut s = unsafe { stomach.as_slice_mut()? };
    let mut pm = unsafe { predation_mask.as_slice_mut()? };

    predation::predation_attack(
        &mut e, &mut s, &mut pm,
        flat.as_slice()?, g_slice,
        attackers.as_slice()?,
        rand_prey.as_slice()?,
        rand_success.as_slice()?,
        neighbors.as_slice()?,
        n_cells, nb_stride, gene_count, max_energy, eat_efficiency,
    );
    Ok(())
}

/// predation_and_culture：捕食（L4）+ 文化学习（L5）合并，共用一次 CSR 构建。
///
/// 语义与引擎步骤 5.5 + 6.5 逐位等价。每 tick 只构建一次 cell→个体 CSR，
/// 消除单独下沉时两次构建的重复开销。energy/stomach/predation_mask/interpret 就地更新。
#[pyfunction]
fn predation_and_culture(
    energy: Bound<'_, PyArray1<f64>>,
    stomach: Bound<'_, PyArray1<f64>>,
    predation_mask: Bound<'_, PyArray1<bool>>,
    interpret: Bound<'_, PyArray2<f64>>,
    flat: PyReadonlyArray1<'_, i64>,
    genes: PyReadonlyArray2<'_, f64>,
    age: PyReadonlyArray1<'_, i64>,
    maturity_age: PyReadonlyArray1<'_, f64>,
    attackers: PyReadonlyArray1<'_, i64>,
    rand_prey: PyReadonlyArray1<'_, i64>,
    rand_success: PyReadonlyArray1<'_, f64>,
    neighbors: PyReadonlyArray1<'_, i64>,
    n_cells: usize,
    nb_stride: usize,
    max_energy: f64,
    eat_efficiency: f64,
    culture_alpha: f64,
    attack_cost: f64,
    success_gene_gain: f64,
    success_floor: f64,
    success_ceil: f64,
    transfer_ratio: f64,
    stomach_transfer: f64,
) -> PyResult<()> {
    let n = flat.as_array().len();
    require_len("energy", unsafe { energy.as_array().len() }, n)?;
    require_len("stomach", unsafe { stomach.as_array().len() }, n)?;
    require_len("predation_mask", unsafe { predation_mask.as_array().len() }, n)?;
    require_len("age", age.as_array().len(), n)?;
    require_len("maturity_age", maturity_age.as_array().len(), n)?;
    let n_att = attackers.as_array().len();
    require_len("rand_prey", rand_prey.as_array().len(), n_att)?;
    require_len("rand_success", rand_success.as_array().len(), n_att)?;

    let g_arr = genes.as_array();
    let sh = g_arr.shape();
    let (n_rows, gene_count) = (sh[0], sh[1]);
    if n_rows != n || gene_count <= 16 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "genes: 形状 ({n_rows}, {gene_count}) 与个体数 {n} 不符（需含 g16）"
        )));
    }
    let g_slice = g_arr.as_slice().ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("genes: 需要 C 连续（行主序）数组")
    })?;

    let interp_arr = unsafe { interpret.as_array() };
    let ish = interp_arr.shape();
    if ish[0] != n || ish[1] != 16 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "interpret: 形状 ({}, {}) 应为 ({}, 16)", ish[0], ish[1], n
        )));
    }

    let nb_len = neighbors.as_array().len();
    if nb_len != n_cells * nb_stride {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "neighbors: 长度 {nb_len} 应为 n_cells*nb_stride = {}",
            n_cells * nb_stride
        )));
    }

    let mut e = unsafe { energy.as_slice_mut()? };
    let mut s = unsafe { stomach.as_slice_mut()? };
    let mut pm = unsafe { predation_mask.as_slice_mut()? };
    let mut interp = unsafe { interpret.as_slice_mut()? };

    l4_l5::predation_and_culture(
        &mut e, &mut s, &mut pm, &mut interp,
        flat.as_slice()?, g_slice,
        age.as_slice()?, maturity_age.as_slice()?,
        attackers.as_slice()?,
        rand_prey.as_slice()?,
        rand_success.as_slice()?,
        neighbors.as_slice()?,
        n_cells, nb_stride, gene_count,
        max_energy, eat_efficiency, culture_alpha,
        attack_cost, success_gene_gain, success_floor, success_ceil,
        transfer_ratio, stomach_transfer,
    );
    Ok(())
}

/// step_movement：移动决策（L3 g14），逐个体计算邻格得分并选择目标。
///
/// 语义与引擎步骤 5 逐位等价。flat/energy 就地更新。
/// move_inds 是移动个体索引（Python 侧已筛选 move_mask & energy>cost）；
/// rand_choice 是预生成随机选择（得分无差异时用），保证 RNG 消费顺序一致。
#[pyfunction]
fn step_movement(
    flat: Bound<'_, PyArray1<i64>>,
    energy: Bound<'_, PyArray1<f64>>,
    genes: PyReadonlyArray2<'_, f64>,
    trust: PyReadonlyArray1<'_, f64>,
    work_memory: PyReadonlyArray1<'_, i64>,
    interpret: PyReadonlyArray2<'_, f64>,
    food_ratio: PyReadonlyArray1<'_, f64>,
    sig_present: PyReadonlyArray1<'_, f64>,
    densities: PyReadonlyArray1<'_, f64>,
    signal_marks: PyReadonlyArray1<'_, u8>,
    neighbors: PyReadonlyArray1<'_, i64>,
    move_inds: PyReadonlyArray1<'_, i64>,
    rand_choice: PyReadonlyArray1<'_, i64>,
    move_cost_ind: PyReadonlyArray1<'_, f64>,
    n_cells: usize,
    nb_stride: usize,
    // A′ 记忆朝向梯度（2026-09-19）：mode 0=none（原式）/ 1=orientation（朝向梯度）
    n_cols: usize,
    mem_grad_mode: u8,
    mem_grad_gain: f64,
) -> PyResult<()> {
    let n = unsafe { flat.as_array().len() };
    if n_cols == 0 || n_cells % n_cols != 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "n_cols={n_cols} 与 n_cells={n_cells} 不自洽（需整除且非零）"
        )));
    }
    if mem_grad_mode > 1 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "mem_grad_mode={mem_grad_mode} 非法（0=none / 1=orientation）"
        )));
    }
    require_len("energy", unsafe { energy.as_array().len() }, n)?;
    require_len("trust", trust.as_array().len(), n)?;
    require_len("work_memory", work_memory.as_array().len(), n * 4)?;
    require_len("move_cost_ind", move_cost_ind.as_array().len(), n)?;
    let n_move = move_inds.as_array().len();
    require_len("rand_choice", rand_choice.as_array().len(), n_move)?;
    require_len("food_ratio", food_ratio.as_array().len(), n_cells)?;
    require_len("sig_present", sig_present.as_array().len(), n_cells)?;
    require_len("densities", densities.as_array().len(), n_cells)?;
    require_len("signal_marks", signal_marks.as_array().len(), n_cells)?;

    let g_arr = genes.as_array();
    let sh = g_arr.shape();
    let (n_rows, gene_count) = (sh[0], sh[1]);
    if n_rows != n || gene_count <= 14 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "genes: 形状 ({n_rows}, {gene_count}) 与个体数 {n} 不符（需含 g14）"
        )));
    }
    let g_slice = g_arr.as_slice().ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("genes: 需要 C 连续（行主序）数组")
    })?;

    let interp_arr = interpret.as_array();
    let ish = interp_arr.shape();
    if ish[0] != n || ish[1] != 16 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "interpret: 形状 ({}, {}) 应为 ({}, 16)", ish[0], ish[1], n
        )));
    }
    let interp_slice = interp_arr.as_slice().ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("interpret: 需要 C 连续（行主序）数组")
    })?;

    let nb_len = neighbors.as_array().len();
    if nb_len != n_cells * nb_stride {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "neighbors: 长度 {nb_len} 应为 n_cells*nb_stride = {}",
            n_cells * nb_stride
        )));
    }

    let mut f = unsafe { flat.as_slice_mut()? };
    let mut e = unsafe { energy.as_slice_mut()? };

    movement::step_movement(
        &mut f, &mut e, g_slice,
        trust.as_slice()?, work_memory.as_slice()?, interp_slice,
        food_ratio.as_slice()?, sig_present.as_slice()?, densities.as_slice()?,
        signal_marks.as_slice()?,
        neighbors.as_slice()?,
        move_inds.as_slice()?, rand_choice.as_slice()?,
        move_cost_ind.as_slice()?,
        n_cells, nb_stride, gene_count,
        n_cols, mem_grad_mode, mem_grad_gain,
    );
    Ok(())
}

/// reproduce_batch：繁殖（L7b / T4）数值核心下沉。
///
/// 语义与引擎步骤 8 逐位等价；RNG 由 Python 按原顺序预生成传入，
/// 本函数只做确定性算术。父本 energy/stomach/cooldown 就地更新，
/// 子代各行写入预分配的 child_* 数组（Python 随后 np.concatenate）。
/// 形状约定：ri(K)、mut_mask/gene_noise(K×gc)、exp_noise(K×exp)等。
#[allow(clippy::too_many_arguments)]
#[pyfunction]
fn reproduce_batch(
    ri: PyReadonlyArray1<'_, i64>,
    genes: PyReadonlyArray2<'_, f64>,
    energy: Bound<'_, PyArray1<f64>>,
    stomach: Bound<'_, PyArray1<f64>>,
    cooldown: Bound<'_, PyArray1<f64>>,
    exp: PyReadonlyArray2<'_, f64>,
    interpret: PyReadonlyArray2<'_, f64>,
    trust: PyReadonlyArray1<'_, f64>,
    baseline: PyReadonlyArray1<'_, f64>,
    mut_mask: PyReadonlyArray2<'_, u8>,
    gene_noise: PyReadonlyArray2<'_, f64>,
    exp_noise: PyReadonlyArray2<'_, f64>,
    interp_noise: PyReadonlyArray2<'_, f64>,
    child_genes: Bound<'_, PyArray2<f64>>,
    child_energy: Bound<'_, PyArray1<f64>>,
    child_stomach: Bound<'_, PyArray1<f64>>,
    child_exp: Bound<'_, PyArray2<f64>>,
    child_interp: Bound<'_, PyArray2<f64>>,
    child_trust: Bound<'_, PyArray1<f64>>,
    child_baseline: Bound<'_, PyArray1<f64>>,
    gene_min: f64,
    gene_max: f64,
    max_reward: f64,
    repro_cd_scale: f64,
) -> PyResult<()> {
    let ri_s = ri.as_slice()?;
    let k = ri_s.len();
    let n = unsafe { energy.as_array().len() };
    require_len("stomach", unsafe { stomach.as_array().len() }, n)?;
    require_len("cooldown", unsafe { cooldown.as_array().len() }, n)?;
    require_len("trust", trust.as_array().len(), n)?;
    require_len("baseline", baseline.as_array().len(), n)?;
    require_len("child_energy", unsafe { child_energy.as_array().len() }, k)?;
    require_len("child_stomach", unsafe { child_stomach.as_array().len() }, k)?;
    require_len("child_trust", unsafe { child_trust.as_array().len() }, k)?;
    require_len("child_baseline", unsafe { child_baseline.as_array().len() }, k)?;

    let g = genes.as_array();
    let gsh = g.shape();
    let (n_rows, gene_count) = (gsh[0], gsh[1]);
    if n_rows != n {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "genes: 行数 {n_rows} 与个体数 {n} 不符"
        )));
    }
    let g_slice = g.as_slice().ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("genes: 需要 C 连续（行主序）数组")
    })?;

    let e_arr = exp.as_array();
    let esh = e_arr.shape();
    if esh[0] != n || esh[1] != 120 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "exp: 形状 ({}, {}) 应为 ({n}, 120)", esh[0], esh[1]
        )));
    }
    let e_slice = e_arr.as_slice().ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("exp: 需要 C 连续（行主序）数组")
    })?;

    let i_arr = interpret.as_array();
    let ish = i_arr.shape();
    if ish[0] != n || ish[1] != 16 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "interpret: 形状 ({}, {}) 应为 ({n}, 16)", ish[0], ish[1]
        )));
    }
    let i_slice = i_arr.as_slice().ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("interpret: 需要 C 连续（行主序）数组")
    })?;

    // K 行预算组：mut_mask/gene_noise (K×gc)、exp_noise (K×120)、interp_noise (K×16)，
    // 子代输出 (K×gc / K×120 / K×16)。
    let m_arr = mut_mask.as_array();
    let msh = m_arr.shape();
    let full_shape_check = |want: (usize, usize), got: (usize, usize)| -> bool {
        got == want
    };
    if !full_shape_check((k, gene_count), (msh[0], msh[1])) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "mut_mask: 形状 ({}, {}) 应为 ({k}, {gene_count})", msh[0], msh[1]
        )));
    }
    let m_slice = m_arr.as_slice().ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("mut_mask: 需要 C 连续（行主序）数组")
    })?;

    let gn_arr = gene_noise.as_array();
    let gnsh = gn_arr.shape();
    if (gnsh[0], gnsh[1]) != (k, gene_count) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "gene_noise: 形状 ({}, {}) 应为 ({k}, {gene_count})", gnsh[0], gnsh[1]
        )));
    }
    let gn_slice = gn_arr.as_slice().ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("gene_noise: 需要 C 连续（行主序）数组")
    })?;

    let en_arr = exp_noise.as_array();
    let ensh = en_arr.shape();
    if (ensh[0], ensh[1]) != (k, 120) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "exp_noise: 形状 ({}, {}) 应为 ({k}, 120)", ensh[0], ensh[1]
        )));
    }
    let en_slice = en_arr.as_slice().ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("exp_noise: 需要 C 连续（行主序）数组")
    })?;

    let in_arr = interp_noise.as_array();
    let insh = in_arr.shape();
    if (insh[0], insh[1]) != (k, 16) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "interp_noise: 形状 ({}, {}) 应为 ({k}, 16)", insh[0], insh[1]
        )));
    }
    let in_slice = in_arr.as_slice().ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("interp_noise: 需要 C 连续（行主序）数组")
    })?;

    let cg_arr = unsafe { child_genes.as_array() };
    if cg_arr.shape() != [k, gene_count] {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "child_genes: 形状 ({}, {}) 应为 ({k}, {gene_count})",
            cg_arr.shape()[0], cg_arr.shape()[1]
        )));
    }
    let mut cg = unsafe { child_genes.as_slice_mut()? };
    let ce_arr = unsafe { child_exp.as_array() };
    if ce_arr.shape() != [k, 120] {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "child_exp: 形状 ({}, {}) 应为 ({k}, 120)",
            ce_arr.shape()[0], ce_arr.shape()[1]
        )));
    }
    let mut ce = unsafe { child_exp.as_slice_mut()? };
    let ci_arr = unsafe { child_interp.as_array() };
    if ci_arr.shape() != [k, 16] {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "child_interp: 形状 ({}, {}) 应为 ({k}, 16)",
            ci_arr.shape()[0], ci_arr.shape()[1]
        )));
    }
    let mut ci = unsafe { child_interp.as_slice_mut()? };

    let mut e = unsafe { energy.as_slice_mut()? };
    let mut s = unsafe { stomach.as_slice_mut()? };
    let mut c = unsafe { cooldown.as_slice_mut()? };

    reproduction::reproduce_batch(
        ri_s, g_slice, &mut e, &mut s, &mut c,
        e_slice, i_slice, trust.as_slice()?, baseline.as_slice()?,
        m_slice, gn_slice, en_slice, in_slice,
        &mut cg, unsafe { child_energy.as_slice_mut()? },
        unsafe { child_stomach.as_slice_mut()? },
        &mut ce, &mut ci,
        unsafe { child_trust.as_slice_mut()? },
        unsafe { child_baseline.as_slice_mut()? },
        gene_count, 120, gene_min, gene_max, max_reward, repro_cd_scale,
    );
    Ok(())
}

/// signal_emit：信号发射（L3 g15），批量判定发射、耗能、编码模式、写入信号场。
///
/// 语义与 sphere_engine 步骤 4.5 逐位等价。energy/signal_marks/signal_age 就地更新。
/// rand_emit 由 Python 侧预生成（self.rng.random(P)），保证 RNG 消费顺序一致。
/// densities 由 Python 侧预计算（np.bincount），与移动下沉共用。
#[pyfunction]
fn signal_emit(
    flat: PyReadonlyArray1<'_, i64>,
    energy: Bound<'_, PyArray1<f64>>,
    g15: PyReadonlyArray1<'_, f64>,
    rand_emit: PyReadonlyArray1<'_, f64>,
    densities: PyReadonlyArray1<'_, f64>,
    resource_grid: PyReadonlyArray1<'_, f64>,
    resource_capacity: PyReadonlyArray1<'_, f64>,
    signal_marks: Bound<'_, PyArray1<u8>>,
    signal_age: Bound<'_, PyArray1<i32>>,
    emit_cost: f64,
    max_energy: f64,
    duration: i32,
) -> PyResult<usize> {
    let n = flat.as_array().len();
    require_len("energy", unsafe { energy.as_array().len() }, n)?;
    require_len("g15", g15.as_array().len(), n)?;
    require_len("rand_emit", rand_emit.as_array().len(), n)?;
    let n_cells = densities.as_array().len();
    require_len("resource_grid", resource_grid.as_array().len(), n_cells)?;
    require_len("resource_capacity", resource_capacity.as_array().len(), n_cells)?;
    require_len("signal_marks", unsafe { signal_marks.as_array().len() }, n_cells)?;
    require_len("signal_age", unsafe { signal_age.as_array().len() }, n_cells)?;

    let mut e = unsafe { energy.as_slice_mut()? };
    let mut sm = unsafe { signal_marks.as_slice_mut()? };
    let mut sa = unsafe { signal_age.as_slice_mut()? };

    let n_emit = signal::signal_emit_batch(
        flat.as_slice()?, &mut e, g15.as_slice()?, rand_emit.as_slice()?,
        densities.as_slice()?, resource_grid.as_slice()?, resource_capacity.as_slice()?,
        &mut sm, &mut sa,
        emit_cost, max_energy, duration,
    );
    Ok(n_emit)
}

/// pleasure_update：愉悦度批量更新（L2 RPE 预测误差驱动），L7a C2 下沉。
///
/// 语义与 sphere_engine._update_pleasure 逐位等价。
/// valence/arousal/expectation/baseline 就地更新。expectation 为 (N,120) 展平。
/// energy_before 由 Python 侧在 tick 开始时保存（步骤 1 前）。
#[pyfunction]
fn pleasure_update(
    flat: PyReadonlyArray1<'_, i64>,
    energy_now: PyReadonlyArray1<'_, f64>,
    energy_before: PyReadonlyArray1<'_, f64>,
    densities: PyReadonlyArray1<'_, f64>,
    resource_grid: PyReadonlyArray1<'_, f64>,
    resource_capacity: PyReadonlyArray1<'_, f64>,
    signal_marks: PyReadonlyArray1<'_, u8>,
    valence: Bound<'_, PyArray1<f64>>,
    arousal: Bound<'_, PyArray1<f64>>,
    expectation: Bound<'_, PyArray1<f64>>,
    baseline: Bound<'_, PyArray1<f64>>,
    max_energy: f64,
    alpha: f64,
    valence_decay: f64,
    arousal_decay: f64,
    baseline_rate: f64,
    max_reward: f64,
    w_energy: f64,
    w_info: f64,
    w_social: f64,
) -> PyResult<()> {
    let n = flat.as_array().len();
    require_len("energy_now", energy_now.as_array().len(), n)?;
    require_len("energy_before", energy_before.as_array().len(), n)?;
    require_len("valence", unsafe { valence.as_array().len() }, n)?;
    require_len("arousal", unsafe { arousal.as_array().len() }, n)?;
    require_len("baseline", unsafe { baseline.as_array().len() }, n)?;
    require_len("expectation", unsafe { expectation.as_array().len() }, n * 120)?;
    let n_cells = densities.as_array().len();
    require_len("resource_grid", resource_grid.as_array().len(), n_cells)?;
    require_len("resource_capacity", resource_capacity.as_array().len(), n_cells)?;
    require_len("signal_marks", signal_marks.as_array().len(), n_cells)?;

    let mut v = unsafe { valence.as_slice_mut()? };
    let mut a = unsafe { arousal.as_slice_mut()? };
    let mut exp = unsafe { expectation.as_slice_mut()? };
    let mut bl = unsafe { baseline.as_slice_mut()? };

    pleasure::pleasure_update_batch(
        flat.as_slice()?, energy_now.as_slice()?, energy_before.as_slice()?,
        densities.as_slice()?, resource_grid.as_slice()?, resource_capacity.as_slice()?,
        signal_marks.as_slice()?,
        &mut v, &mut a, &mut exp, &mut bl,
        max_energy, alpha, valence_decay, arousal_decay, baseline_rate, max_reward,
        w_energy, w_info, w_social,
    );
    Ok(())
}

#[pymodule]
fn sim_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(regrow_rs, m)?)?;
    m.add_function(wrap_pyfunction!(regrow_patchy_rs, m)?)?;
    m.add_function(wrap_pyfunction!(consume_many, m)?)?;
    m.add_function(wrap_pyfunction!(step_vectors_stage1, m)?)?;
    m.add_function(wrap_pyfunction!(step_vectors_stage2, m)?)?;
    m.add_function(wrap_pyfunction!(culture_learn, m)?)?;
    m.add_function(wrap_pyfunction!(predation_attack, m)?)?;
    m.add_function(wrap_pyfunction!(predation_and_culture, m)?)?;
    m.add_function(wrap_pyfunction!(step_movement, m)?)?;
    m.add_function(wrap_pyfunction!(reproduce_batch, m)?)?;
    m.add_function(wrap_pyfunction!(signal_emit, m)?)?;
    m.add_function(wrap_pyfunction!(pleasure_update, m)?)?;
    m.add_function(wrap_pyfunction!(fruit_charge, m)?)?;
    m.add_function(wrap_pyfunction!(eat_fruit, m)?)?;
    m.add_function(wrap_pyfunction!(native_gene_indicators, m)?)?;
    m.add_function(wrap_pyfunction!(validate_gene_wiring, m)?)?;
    m.add_class::<light_temp::LightTempRust>()?;
    Ok(())
}