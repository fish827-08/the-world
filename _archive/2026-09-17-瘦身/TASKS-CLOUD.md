# 云端并行任务分配单（2026-09-07 dispatch）

> 分配方：本地主开发线（feature/dev-l7）
> 承接方：云端环境（feature/snapshot 快照机制已完工）
> 分工原则：与本地在跑的 A2（隐式选择压参数化）/ T4（繁殖下沉）改动区域隔离；
> 各自独立分支独立 commit，完成后统一由本地 review 合并进 feature/dev-l7。

---

## 一、分配的 3 项任务（预估 7~9h）

| 编号 | 任务 | 预估 | 规格来源 |
|------|------|------|---------|
| C1 | T2：L7a 信号发射 `signal_emit_batch` 下沉 Rust + 对拍 | 2h | L7-性能优化详细设计文档.md §三（L7a） |
| C2 | T3：L7a 愉悦度更新 `pleasure_update_batch` 下沉 Rust + 对拍 | 2~3h | L7-性能优化详细设计文档.md §三（L7a） |
| C3 | G1~G4：基因系统扩展性（接入流程/漂移测试/元数据/预留位规则） | 3~4h | TASKS-DEV.md §二 |

## 二、任务要求

### C1 = T2 信号发射下沉
- Rust 新增 `sim_core/src/signal.rs`，实现 `signal_emit_batch`。
- **先确认信号场实际存储格式**：当前是 `(n_cells,) uint8`（低 4 位记录田字格子格），
  不是 L7 文档 3.2.1 伪代码里的 `(n_cells, 4)`，以实际代码为准再实现（见 L7 §3.2.1 注意）。
- 发射判定/状态哈希（能量2位+食物1位+邻居1位=4位）/耗能逻辑必须与 Python 逐位一致。
- 验收：函数级对拍（同输入同预生成 rand_emit，signal_grid 与 energy 逐位相等）
  + 引擎级对拍（同 seed，use_sim_core=True vs False 逐 tick 信号场/能量一致）+ 全量 pytest 通过。

### C2 = T3 愉悦度更新下沉
- Rust 新增 `sim_core/src/pleasure.rs`，实现 `pleasure_update_batch`。
- 120 情境编码公式（能量5×食物4×邻居3×信号2）与 float binning 边界必须与 Python 逐位对齐，
  参数（EWMA α=0.05、valence 0.7/0.3、arousal 0.8/0.2、baseline 漂移）从配置/引擎参数传入。
- 验收：函数级对拍（120 情境逐位）+ 引擎级对拍 + 全量 pytest 通过。

### C3 = G1~G4 基因扩展性
- G1：新基因接入流程固化进 AGENT.md / MODULES.md（"加基因五步曲"）。
- G2：pytest 覆盖 `validate_gene_wiring`；故意改一侧索引→测试必须失败（tests/test_genes_registry.py）。
- G3：`GENE_SEMANTICS` 增加每基因 `mutation_scale` / `selection_direction` 元数据，观察台直接消费。
- G4：预留位接管规则写入文档（g17/g18/g20~g23 优先复用，耗尽前不扩 gene_count；扩位时触发存档格式升级提示）。
- 验收：全量 pytest 通过；文档补全；不改变现有行为（纯扩展）。

## 三、环境与分支约定

1. **分支**：从 `gitee/feature/dev-l7` 最新（pull/fetch 后）新建 `feature/cloud-tasks`；
   每任务独立 commit（提交信息带 C1/C2/C3 编号），完成后 push origin（gitee + github）。
2. **Rust 工具链**：cargo/rustc 1.81（msvc）在 `%USERPROFILE%\.cargo\bin`；
   maturin 装入开发用 venv（本仓库已有 `.venv-dev`，缺依赖 `pip install maturin numpy pytest websockets`）。
   **构建 sim_core 时 VIRTUAL_ENV 必须指向开发 venv，勿动实验用 `.venv`**。
3. **对拍/测试一律用开发 venv 的 python**：`.venv-dev\Scripts\python.exe`（勿用全局 python，会解析错同名 namespace 包）。
4. 与本地隔离事项：T4（繁殖下沉）、A2（捕食/文化参数化）由本地负责，
   **不要改动 `sim_core/src/l4_l5.rs`、`sim_core/src/predation.rs`、`simulation/config.py` 的 predation/culture 段**，
   也不要重复实现 T4。
5. 完成每项后更新 `PROGRESS.md`，并在 commit 信息里注明"经验收"。

## 四、返回方式

全部完成后 push `feature/cloud-tasks` 到 gitee/github，并更新 PROGRESS.md；
由本地统一 review → 合并进 feature/dev-l7（若有冲突，本地解）。

---

# 第二轮云端任务分配单（2026-09-07 dispatch #2）

> 承接方：云端环境（第一轮 C1/C2/C3 已全部完成并合入 feature/dev-l7，e531853 验 136 passed）
> 分工原则：与本地在跑的 T4（繁殖下沉，仅剩 Python 集成/对拍）改动区域隔离；
> 本轮三任务各自独立分支独立 commit，完成后 push `feature/cloud-tasks`，由本地 review 合并。

## 一、分配的 3 项任务（预估 8~11h）

| 编号 | 任务 | 预估 | 规格来源 |
|------|------|------|---------|
| C4 | A3：敏感性扫描脚本 + 后台低优先级跑核心 4 参数 | 3~5h（含后台跑批） | 隐式选择压审计清单.md §三 |
| C5 | L10：种子传播（蓄力/释放/传播落点/种子能量）下沉 Rust + 对拍 | 3~4h | L8-L10-扩展特性建议文档.md §L10 |
| C6 | T5（可选）：L7c slots 预分配模式替代每 tick `np.delete/concat` | 2~3h | TASKS-DEV.md §一 T5 |

## 二、任务要求

### C4 = A3 敏感性扫描（脚本 + 后台跑批）

- 目标：对高影响参数（§三发现的 4 个）做 ±20% / ±50% 扫描（500 代 × 3 seed），
  标出哪些参数显著改变终局基因分布。
- 做法：新建 `experiments/sensitivity_scan.py`（复用 `long_run_l4l5.py` 的引擎配置与增量落盘风格）：
  - 参数维度：`predation.transfer_ratio`（+/-20/50%）、`culture.trust_false`（对称化对照）、
    社会效价 `+0.2/-0.1`（对称化对照 `±0.15`）、活性保底 `0.4`（→0.0/0.2/0.6）；
  - 结果落到 `results/sensitivity/<param>_<value>_<seed>/`（termination.csv，字段含终局基因均值/分布）；
  - 汇总表 `results/sensitivity/summary.csv`：参数→终局基因分布差异。
- 注意事项：
  - 跑批用**实验用 `.venv`**（不是 `.venv-dev`），后台低优先级执行；写入必须是**增量落盘**（每代 flush），防中断丢数据；
  - seed 用 42/2024/777；500 代起步（如单次 >10min 可降至 300 代并在 summary 注明）。
- 验收：脚本可在两种 venv 下运行；summary.csv 齐 3 seed × 扫描档位；不改变引擎行为（只新增实验脚本）。

### C5 = L10 种子传播下沉（先行版：数值管线）

- 先读 `L8-L10-扩展特性建议文档.md` §L10 确认语义；**若云端侧尚未实现 L10 的 Python 参考实现，
  可在 Python 侧先实现 L10 参考实现（含蓄力/释放/传播/种子能量），再下沉 Rust，两步都走对拍**。
- Rust 新增 `sim_core/src/dispersal.rs`，实现种子传播批量管线（分值位运算→落点采样可保留 Python 侧，仅下沉确定性数值段）；
- 对拍：函数级（同输入同 rand_emit 类随机数组，落点/能量逐位一致）+ 引擎级（use_sim_core=True vs False）+ 全量 pytest。
- 注意与本地隔离：**不要动 `sim_core/src/l4_l5.rs`、`predation.rs`、`reproduction.rs`**。

### C6 = T5 slots 预分配预研（可选，做不出可回退）

- 目标：评估并（若顺序依赖风险可控）实现用"slot 池"替代每 tick `np.delete/np.concatenate` 的种群数组管理。
- 产出至少包含一个**评估报告**（`docs/slots-preview.md`）：存活性管理算法、死亡槽复用策略、
  与 RNG consume 顺序的交互风险点、性能预期（基准：T1~T4 后 N=5000 tick/s）；若结论可行再实现。
- 验收：评估报告 + （如实现）双路径 1000 tick 对拍 + 全量 pytest。

## 三、环境与分支约定（同第一轮）

1. 分支：`feature/cloud-tasks` 基础上新建 `feature/cloud-tasks-2`，每任务独立 commit（带 C4/C5/C6 编号）。
2. Rust 工具链/maturin/测试解释器约定同第一轮 §三（.venv-dev 构建，.venv 跑实验）。
3. 与本地隔离：T4（繁殖下沉）由本地负责，勿动 `reproduction.rs` 与 `sphere_engine.py` 繁殖步骤；
   A2 参数已定稿（`SimConfig.predation/culture`），勿改其默认值（扫描值由实验脚本传入覆盖）。
4. 完成每项后更新 PROGRESS.md，commit 信息注明"经验收"。

## 四、返回方式（同第一轮）

push `feature/cloud-tasks-2` 后由本地统一 review → 合并进 feature/dev-l7。