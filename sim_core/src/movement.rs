//! 移动决策（L3 g14）Rust 下沉：逐个体计算 8 邻格得分并选择目标。
//!
//! 语义与 sphere_engine._step_population 步骤 5 逐位等价：
//!   - 得分 = 感知×(食物×0.5 + 信号×0.5×信任度) + 群居×密度
//!          + 工作记忆×0.3×感知 + 解读表×0.4×感知
//!   - 得分无差异时随机选邻格（rand_choice 预生成）
//!   - 否则选得分最高的邻格
//!   - 移动后扣移动耗能（move_cost_ind）
//!
//! 优化点：消除逐个体 Python 循环 + np.isin + np.max/min + np.argmax + np.array 等
//! 大量 numpy 小函数调用开销（N=5000 时占总耗时 50%+）。

use crate::genes::{G_PERCEPTION, G_SOCIABILITY};

/// 移动决策核心函数。
///
/// # 参数
/// - `flat`: (N,) 个体所在格子，就地更新（移动个体的新位置）
/// - `energy`: (N,) 能量，就地更新（扣移动耗能）
/// - `genes`: (N, gene_count) 基因（C 连续）
/// - `trust`: (N,) 信任度
/// - `work_memory`: (N, 4) 工作记忆（-1 表示空槽）
/// - `interpret`: (N, 16) 信号解读表
/// - `food_ratio`: (n_cells,) 每格食物比例
/// - `sig_present`: (n_cells,) 每格是否有信号（0/1）
/// - `densities`: (n_cells,) 每格个体密度
/// - `signal_marks`: (n_cells,) 每格信号标记（0~15，0=无信号）
/// - `neighbors`: (n_cells * nb_stride,) 展平邻居表（P0.1：普通格紧凑 8 列）
/// - `pole_nb`: (2 * n_cols,) 极点带（行 0 = 上极整带，行 1 = 下极整带）
/// - `move_inds`: (n_move,) 移动个体的索引（Python 侧已筛选 move_mask & energy>cost）
/// - `rand_choice`: (n_move,) 预生成随机选择（得分无差异时用，% n_valid_nb）
/// - `move_cost_ind`: (N,) 每个个体的移动耗能
/// - `n_cells`: 格子总数
/// - `nb_stride`: 主表每行邻居数（普通格）
/// - `gene_count`: 基因数
/// - `pole_top` / `pole_bottom`: 上下极点带的行号（`neighbors.rs::nb_slice`）
/// - `mem_grad_mode` / `mem_grad_gain`: A′ 记忆朝向梯度（0=none 原式 / 1=orientation）
/// - `h_norm`: (N,) 饥饿归一量（R247 HM ②）；**空 slice = 关档**（不进新代码路径）
/// - `hm_beta`: HM ② 强度（仅 `h_norm` 非空时消费）
/// - `smell_use`: R244 §二气味场消费开关（0=关 ⇒ 不读 `smell_score`）
/// - `smell_score`: (n_cells,) 气味综合得分（仅 `smell_use != 0` 时索引）
/// - `asm_on`: R239 ASM 模式仲裁开关（0=关 ⇒ 全段不执行，后四者不读）
/// - `asm_mode`: (N,) 本 tick 的个体模式（仅 `asm_on != 0` 时索引）
/// - `asm_sf` / `asm_sr` / `asm_sk`: (n_cells,) 食物/风险/亲缘通道归一值（同前）
#[allow(clippy::too_many_arguments)]
pub fn step_movement(
    flat: &mut [i64],
    energy: &mut [f64],
    genes: &[f64],
    trust: &[f64],
    work_memory: &[i64],
    interpret: &[f64],
    food_ratio: &[f64],
    sig_present: &[f64],
    densities: &[f64],
    signal_marks: &[u8],
    neighbors: &[i64],
    pole_nb: &[i64],
    move_inds: &[i64],
    rand_choice: &[i64],
    move_cost_ind: &[f64],
    n_cells: usize,
    nb_stride: usize,
    gene_count: usize,
    n_cols: usize,
    pole_top: usize,
    pole_bottom: usize,
    mem_grad_mode: u8,
    mem_grad_gain: f64,
    h_norm: &[f64],
    hm_beta: f64,
    // R244 §二：气味场消费端（`smell_use=0` ⇒ 不读 `smell_score`）
    smell_use: u8,
    smell_score: &[f64],
    // R239 ASM：模式仲裁（`asm_on=0` ⇒ 不读后四者；关档占位长度为 0/1 亦安全）
    asm_on: u8,
    asm_mode: &[i8],
    asm_sf: &[f64],
    asm_sr: &[f64],
    asm_sk: &[f64],
) {
    let n = flat.len();
    if n == 0 || move_inds.is_empty() {
        return;
    }
    let hm_on = !h_norm.is_empty();

    for (k, &idx_i64) in move_inds.iter().enumerate() {
        let idx = idx_i64 as usize;
        if idx >= n {
            continue;
        }
        let c = flat[idx] as usize;
        if c >= n_cells {
            continue;
        }

        // R247 HM ②：只调制 perc 赋值行（与 Python 参考循环/向量化快路径**逐字同式**：
        //   `perc * (1.0 + beta * h_norm[idx])`）。关档（h_norm 空）⇒ 括号内乘 1.0 都不做。
        let mut perc = genes[idx * gene_count + G_PERCEPTION];
        if hm_on {
            perc *= 1.0 + hm_beta * h_norm[idx];
        }
        let soc = (genes[idx * gene_count + G_SOCIABILITY] - 0.5) * 2.0;
        let trust_val = trust[idx];

        // 收集有效邻居（>=0），同时计算得分
        // P0.1：普通格取主表 8 项；极点格取极点带整行（cols 项，与旧 fat 表逐位同序）
        let nb = crate::neighbors::nb_slice(
            neighbors, pole_nb, c, n_cols, nb_stride, pole_top, pole_bottom,
        );
        let mut valid_nb: Vec<i64> = Vec::with_capacity(nb.len());
        let mut scores: Vec<f64> = Vec::with_capacity(nb.len());
        let mut n_valid = 0usize;

        for &nbc in nb {
            if nbc < 0 {
                continue;
            }
            let nbc = nbc as usize;
            if nbc >= n_cells {
                continue;
            }
            valid_nb.push(nbc as i64);

            // R239 ASM：仲裁档 ⇒ 基础得分**整段替换**为模式内目标函数（与 Python
            //   参考循环/向量化**同式**）：
            //     feed(0)    → Ŝ_food(nb)
            //     flee(1)    → −Ŝ_risk(nb)（即模式内 argmin Ŝ_risk）
            //     join(2)    → Ŝ_kin(nb) + 密度项（密度项与融合档**同一数组**）
            //     explore(3) → 0（全零 ⇒ 平局 ⇒ 随机分支，与 Python 一致）
            //   仲裁档下记忆/信号/气味/噪声等**全部不进 score**（构造期互斥守卫已挡）
            let mut s = if asm_on != 0 {
                match asm_mode[idx] {
                    0 => asm_sf[nbc],
                    1 => -asm_sr[nbc],
                    2 => asm_sk[nbc] + densities[nbc],
                    _ => 0.0,
                }
            } else {
                // 基础得分：感知×(食物×0.5 + 信号×0.5×信任) + 群居×密度
                perc * (food_ratio[nbc] * 0.5 + sig_present[nbc] * 0.5 * trust_val)
                    + soc * densities[nbc]
            };

            // 工作记忆（A′，2026-09-19）：
            //   mode = 0（none）⇒ **原式**：记忆格 == 该邻居格 ⇒ +gain·perc（行为不变）
            //   mode = 1（orientation）⇒ **朝向梯度**：
            //       s += gain · perc · max_m cos(方向(cur→nbc), 方向(cur→m))
            //       🔴 cos **允许为负** ⇒ 背向邻居减分 ⇒ 这才叫"梯度"（不是单纯吸引）
            //   ⚠️ 与 Python（`sphere_engine._memory_orientation_cos`）**逐字同式**：
            //       cos 用 `(u·v)/(|u||v|)` 的**同一展开**；经度环绕用 `rem_euclid`
            //       （≡ Python 对正除数取模）⇒ 保双路径逐位一致。
            let wm_base = idx * 4;
            if asm_on == 0 && mem_grad_mode == 0 {
                for w in 0..4 {
                    let mc = work_memory[wm_base + w];
                    if mc >= 0 && mc as usize == nbc {
                        s += mem_grad_gain * perc;
                        break;
                    }
                }
            } else if asm_on == 0 {
                let cur = flat[idx] as usize;
                let cr = (cur / n_cols) as f64;
                let cc = (cur % n_cols) as f64;
                let cols_f = n_cols as f64;
                let half = cols_f / 2.0;
                let nr = (nbc / n_cols) as f64;
                let nc0 = (nbc % n_cols) as f64;
                let dnr = nr - cr;
                let dnc = (nc0 - cc + half).rem_euclid(cols_f) - half;
                let un = (dnr * dnr + dnc * dnc).sqrt();
                let mut best: f64 = -2.0;
                if un > 0.0 {
                    for w in 0..4 {
                        let mc = work_memory[wm_base + w];
                        if mc < 0 {
                            continue;
                        }
                        let m = mc as usize;
                        if m == cur {
                            continue;
                        }
                        let mr = (m / n_cols) as f64;
                        let mc0 = (m % n_cols) as f64;
                        let dmr = mr - cr;
                        let dmc = (mc0 - cc + half).rem_euclid(cols_f) - half;
                        let mn = (dmr * dmr + dmc * dmc).sqrt();
                        if mn <= 0.0 {
                            continue;
                        }
                        let cosv = (dnr * dmr + dnc * dmc) / (un * mn);
                        if cosv > best {
                            best = cosv;
                        }
                    }
                }
                if best > -2.0 {
                    s += mem_grad_gain * perc * best;
                }
            }

            // 信号解读表：邻格有信号时，个体对该模式的解读影响得分
            //   R239 ASM：仲裁档**首批不接信号**（规格 §3.3 红线）⇒ 整段跳过
            let mark = signal_marks[nbc] as usize;
            if asm_on == 0 && mark > 0 {
                let interp_val = interpret[idx * 16 + mark];
                s += 0.4 * perc * interp_val;
            }

            // R244 §二：气味场消费端 —— 与 Python **同式同位置**（信号项之后）：
            //   `s += perc × smell_score[nbc]`，`smell_score[c] = Σ_ch w_ch·clip(S_ch(c)/S_max, 0, 1)`
            //   （由 Python 侧按"候选格并集"预算好；并集外恒 0）⇒ 双路径逐位一致。
            if smell_use != 0 {
                s += perc * smell_score[nbc];
            }

            scores.push(s);
            n_valid += 1;
        }

        if n_valid == 0 {
            continue;
        }

        // 只有一个有效邻居 → 直接选，不消费随机数（与 Python 一致）
        if n_valid == 1 {
            flat[idx] = valid_nb[0];
            energy[idx] -= move_cost_ind[idx];
            continue;
        }

        // 选择目标：得分无差异时随机选，否则选最高
        let mut max_score = scores[0];
        let mut min_score = scores[0];
        let mut max_idx = 0usize;
        for j in 1..n_valid {
            if scores[j] > max_score {
                max_score = scores[j];
                max_idx = j;
            }
            if scores[j] < min_score {
                min_score = scores[j];
            }
        }

        let target_idx = if (max_score - min_score) < 1e-9 {
            // 得分无差异 → 随机选有效邻居
            (rand_choice[k].rem_euclid(n_valid as i64)) as usize
        } else {
            max_idx
        };

        // 更新位置 + 扣移动耗能
        flat[idx] = valid_nb[target_idx];
        energy[idx] -= move_cost_ind[idx];
    }
}
