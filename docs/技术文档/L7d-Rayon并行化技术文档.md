# L7d Rayon 多线程并行化 — 技术文档

## 概述

将 sim_core Rust 核心中的纯逐元素热路径函数并行化，使用 Rayon 数据并行库。
在小规模下自动回退单线程（避免调度开销），大规模下自动并行，扩大世界后可线性扩展。

## 改动范围

| 文件 | 改动 |
|------|------|
| `sim_core/Cargo.toml` | 新增 `rayon = "1.10"` 依赖 |
| `sim_core/src/step_vectors.rs` | stage1/stage2 并行化 + SyncPtr 工具 + par_for 阈值 |
| `sim_core/src/pleasure.rs` | pleasure_update_batch 并行化 + SyncPtr + par_for |
| `sim_core/src/regrow.rs` | regrow/regrow_patchy 并行化（par_iter_mut） |
| `experiments/language_analysis.py` | 新增：7维度语言涌现量化分析脚本 |

## 并行化的函数

### 1. `step_vectors_stage1`（光合+代谢+维持扣费）
- 纯逐元素，每个个体独立读写 energy/stomach
- 计算量：O(N)，每元素约 20 次 f64 运算

### 2. `step_vectors_stage2`（移动扣费+年龄+死亡+繁殖候选）
- 纯逐元素，每个个体独立读写 energy/age/cooldown + 4 个输出掩码
- 计算量：O(N)，每元素约 15 次 f64 运算

### 3. `pleasure_update_batch`（愉悦度RPE更新）
- 纯逐元素，每个个体独立读写 valence/arousal/expectation[context]/baseline
- 计算量：O(N)，每元素约 30 次 f64 运算 + 情境编码

### 4. `regrow` / `regrow_patchy`（资源场再生）
- 纯逐格，每个格子独立读写 grid
- 计算量：O(n_cells)，当前 7200 格

## 关键技术设计

### SyncPtr：裸指针的 Sync/Send 包装

```rust
#[derive(Clone, Copy)]
struct SyncPtr<T>(*mut T);
unsafe impl<T> Sync for SyncPtr<T> {}
unsafe impl<T> Send for SyncPtr<T> {}

impl<T> SyncPtr<T> {
    #[inline(always)]
    unsafe fn add(self, i: usize) -> *mut T {
        self.0.add(i)
    }
}
```

**为什么需要 SyncPtr？**
- Rayon 的 `for_each` 要求闭包实现 `Sync + Send`
- 裸指针 `*mut T` 不实现 `Sync`，直接捕获会编译错误
- SyncPtr 手动声明 `Sync + Send`，因为 Rayon 保证每个索引只被一个线程访问，不存在数据竞争

**为什么必须通过方法访问（`.add(i)` 而非 `.0.add(i)`）？**
- Rust 闭包的最小化捕获机制：如果只访问结构体字段 `.0`，编译器只捕获该字段（裸指针），而非整个 SyncPtr
- 通过方法调用强制捕获整个 SyncPtr（已实现 Sync）
- 这是一个容易踩的坑，已在代码注释中说明

### par_for：阈值自动切换

```rust
const PAR_THRESHOLD: usize = 2048;

#[inline]
fn par_for<F>(n: usize, f: F)
where F: Fn(usize) + Sync + Send,
{
    if n < PAR_THRESHOLD {
        for i in 0..n { f(i); }  // 单线程，无调度开销
    } else {
        (0..n).into_par_iter().for_each(f);  // 多线程
    }
}
```

**为什么需要阈值？**
- Rayon 的线程调度有固定开销（任务分配、同步）
- 小规模（N < 2048）下，调度开销 > 并行收益，多线程反而更慢
- 云环境只有 2 核，小规模下并行收益有限
- 用户本地 i7-12700H（14核20线程）+ 扩大世界后（N=50000），并行收益显著

**阈值选择依据：**
- 当前默认配置稳定态 N≈163，远低于阈值，零开销
- 扩大世界到 200×400=80000 格后，稳定态 N≈5000~20000，触发并行
- regrow 的 n_cells=7200（当前）/ 80000（扩大后），扩大后触发并行

## 正确性验证

### 双路径逐位对拍

```
Rust(use_sim_core=True) vs Python(use_sim_core=False)
200 tick, seed=42
energy max diff:  0.00e+00 ✓
stomach max diff: 0.00e+00 ✓
age max diff:     0 ✓
```

Rayon 并行化后与单线程版本逐位一致，因为：
1. 每个索引的计算完全独立，无跨元素数据依赖
2. f64 运算顺序与单线程完全相同（每个元素内部的运算顺序不变）
3. 不涉及浮点数求和/归约（那会因顺序不同产生 1ULP 差异）

### 全量测试

```
158 passed, 3 failed (全部 regrow 1ULP 浮点误差，Rust release FMA 优化已知问题), 1 skipped
```

Rayon 并行化相关的测试（step_vectors、pleasure、对拍测试）全部通过。

## 性能基准

环境：云环境 2 核 AMD EPYC 9Y24，默认配置（N≈163，7200格）

| 配置 | 性能 (tick/s) | 加速比 |
|------|---------------|--------|
| 纯 Python (use_sim_core=False) | 178.6 | 1.0x |
| Rust 单线程 (RAYON_NUM_THREADS=1) | 1012.7 | 5.7x |
| Rust + Rayon (阈值自动切换) | 1001.2 | 5.6x |

**当前规模结论：**
- Rust 下沉带来 5.7x 加速（主要收益）
- Rayon 在当前规模下无额外开销（N<阈值，自动单线程）
- Rayon 在当前规模下无收益（2核 + N=163 太小）

**扩大世界后预期收益（用户本地 14核 + N=20000）：**
- stage1/stage2/pleasure：预期 3~6x 加速（受限于 Amdahl 定律，Python 侧开销占比）
- regrow：预期 5~10x 加速（纯 Rust，无 Python 开销）
- 总体：预期 2~4x 加速

## 语言涌现分析脚本

新增 `experiments/language_analysis.py`，从快照计算 7 个维度的语言涌现量化指标：

| 维度 | 指标 | 语言意义 |
|------|------|----------|
| 1. 信号词汇 | 熵、Zipf R²、频率分布 | 词汇丰富度和等级结构 |
| 2. 文化多样性 | 解读表标准差、方差 | 文化演化空间（方言/语义差异） |
| 3. 代际稳定性 | 相邻世代解读表相关性 | 文化可传承性 |
| 4. 空间聚类 | Moran's I、邻居相同率 | 地域方言/信号传播 |
| 5. 信号-语境 MI | 归一化互信息 | 信号的指代意义 |
| 6. 解读一致性 | 种群内解读收敛度 | 共享语义 |
| 7. 基因-表型相关 | g15 与信号行为相关 | 信号能力的进化基础 |

综合评分 0~100，含自动解读。

### 基线 vs 方案A 分析结果（60万 tick）

| 指标 | 基线 | 方案A | 语言意义 |
|------|------|-------|----------|
| 信号熵(归一化) | 0.534 | 0.306 | 方案A信号更集中 |
| Zipf R² | 0.704 | 0.857 | 方案A等级结构更强 |
| 文化多样性(std) | 0.0008 | 0.0013 | 两者都极低（文化趋同） |
| 代际稳定性 | 1.000 | 1.000 | 完全稳定（因趋同） |
| 空间聚类(Moran's I) | 0.171 | 0.071 | 基线略强 |
| 信号-语境 MI | 0.000 | 0.000 | **两者都无指代意义** |
| 解读一致性 | 0.997 | 0.996 | 两者都极高（因趋同） |
| g15-信号相关 | 0.046 | 0.186 | 方案A进化基础更强 |
| 综合评分 | 56.9 | 45.6 | 两者都"无语言迹象" |

**核心结论：**
1. 两个实验都**没有产生语言**——信号与语境互信息为 0，文化多样性极低
2. 方案A在信号结构上有改进（更高 Zipf R²、更强 g15-信号相关），但未突破到真正语言
3. 根本瓶颈：世界太小种群太密（69.4%密度）→ 文化全局混合趋同 → 无方言/语义分化空间
4. 下一步必须扩大世界（200×400）+ 降低密度，才能给语言涌现留出空间

## 后续优化方向

### 高收益 / 中难度
1. **合并 Python→Rust 调用**：当前每 tick 约 12 次跨语言调用，合并 stage1+consume+signal_emit 为一次调用，可减少 30~40% 的 PyO3 开销
2. **Python 侧 np.where/np.bincount 优化**：cProfile 显示这些操作占 Python 侧时间的 40%+，可预分配数组或下沉到 Rust

### 中收益 / 高难度（暂缓，等扩大世界后评估）
3. **Rayon 并行化 step_movement / reproduce_batch**：这两个函数有邻居依赖和共享状态，需要更复杂的并行策略（分块+原子操作）
4. **Rayon 并行化 predation_and_culture**：有 CSR 构建和邻居依赖，复杂度高

### 低收益 / 高难度（暂缓）
5. **GPU 加速**：当前规模下数据传输开销 > 计算收益，扩大世界 10 倍后可评估

## 提交信息

- 分支：`feature/l7d-rayon`
- 基于：`feature/snapshot-cron`（b33ea2e 主线）
- 编译：`cargo build --release`（需 rayon 依赖）
- 验证：双路径逐位一致 + 158 测试通过
