# 项目进度文档

## 项目概况
- 项目名称：digital-life-sphere
- 启动日期：2026-09-06
- 当前阶段：D2 信息结构重构已实现（四机制入 main，215 passed），**D2.1 参数网格云端夜间进行中（lr 4 档夜1 + mat 4 档夜2，推 feat/d2-grid）**；本地 R1 崩点定位（S1×2 + S2×1，60k tick）已完成——S1 阶段 g15 回落无上升（两 seed 终局逐项一致），**D2 gate（g15 无补贴从 0.5 上升）基准档未过，待 D2.1 互补档判读**；档A 三段循环向量化 + 可复现性修复（np.random.seed）已入 feat/d2-vec；后续按 **D0→D5 评估路线**推进（详见评估-EVAL-云端启动说明-20260909.md 与评估-EVAL-汇总意见-20260909.md）

## 总体进度
- [x] 拷贝 AI-CODE-DEVELOPMENT-RULES.md 到项目根目录
- [x] 规则文件加入 .gitignore
- [x] 项目方向评估（规则二）：输出评估报告
- [x] 确认项目总体方案
- [x] 设计整体架构，AI 提草案并与我逐项确认
- [x] 创建 PROJECT-DESCRIPTION.md（功能描述文档，含目录，v0.5 定稿）
- [x] 再创建 MODULES.md（模块实现说明：每个文件/类/方法的通俗解释）
- [x] 模块一开发（SphereWorld 世界模块）——3 个文件全部完成并验证
- [x] 模块二开发（球形引擎 Python 版）——config/tick/lifecycle/sphere_engine 完成；
      基因扩充 g0~g13 + 生命周期年龄（成熟/年龄能耗）；引擎 pytest 套件 7 例全通过；
      observatory 快速长程验证通过（experiments/quick_check_age.py）
- [ ] 模块三开发（Rust 加速核 Sim-core，✅ 全部完成：3.1~3.5 下沉 + 预计算邻居表优化，
      全量 pytest 34 例通过，代码已提交 3ed6fd0）
- [ ] 模块四开发（observatory 适配 + 快照桥，✅ 完成：traits/statistics/observer/experiment/
      __main__/persistence/io + broker 快照桥，统计口径=14 基因位+3 派生，快照推送=100 tick，
      全量 pytest 50 例通过，新增 16 例）
- [ ] 模块五开发（前端渲染层）

## 已完成记录
| 日期 | 完成内容 | 备注 |
|------|----------|------|
| 2026-09-06 | 初始化项目目录 | 名称 digital-life-sphere，位于 the-world 内部子目录 |
| 2026-09-06 | 复制 AI-CODE-DEVELOPMENT-RULES.md 并加入 .gitignore | 遵循记忆中的文档要求 |
| 2026-09-06 | 模块一：world/sphere_world.py | 土地拓扑：经纬网格 60×120、极点坍缩、面积梯度、8 邻 |
| 2026-09-06 | 模块一：world/light_and_temperature.py | 天气：固定太阳+自转（昼夜扫掠）、纬度基温+小温差、活性因子 |
| 2026-09-06 | 模块一：world/resource_field.py | 食物：容量随面积、再生随温度（同格多只均分不欠账） |
| 2026-09-06 | 创建 MODULES.md | 收录模块一全部类与方法的通俗解释、测试点 |
| 2026-09-06 | 模块二：simulation/config.py | 总参数本：8 个子配置 dataclass，配置即复现（to_dict/from_dict/fingerprint） |
| 2026-09-06 | 模块二：simulation/tick.py + core/lifecycle.py | TickStats 快照 + DeathCause 死因枚举（饿死/老死） |
| 2026-09-06 | 模块二：simulation/sphere_engine.py | 引擎核心：单 tick 流程 0~9、数组化推进、繁殖/死亡/清理 |
| 2026-09-06 | 基因扩充 g4~g13 | 进食量/饱食度/移动能耗/繁殖投入/光合/恒温/邻格觅食/温度偏好/繁殖冷却/群居性；机制单测 + 1000 tick 冒烟通过 |
| 2026-09-07 | 模块三 3.5：sim_core/src/consume.rs | 进食结算下沉 Rust：先数每格几只再算每只实吃量（同格均分、绝不欠账），按原序逐只扣减，与 numpy `consume_many` 逐位等价；lib.rs 绑定加长度+越界校验；引擎进食/邻格觅食双路径接入；函数级对拍 8 例 + 引擎级 4 例，全量 pytest 30 例通过 |
| 2026-09-07 | 预计算邻居表优化（world/sphere_world.py） | 构造时一次性算好全网格邻居表 `_nb_table`(7200,8) + `_pole_nb`(2,120)，`neighbors()` 改查表返回；引擎移动/觅食两处调用点零改动，行为零变化（新 tests/test_sphere_world.py 全网格 vs 旧算法逐位对拍）；基准 3.030/3.049s → 0.615/0.569s ≈ 5x，全量 pytest 34 例通过 |
| 2026-09-07 | 模块四：observatory 适配 + 快照桥（✅） | traits（14 基因位+3 派生，统计口径与引擎同公式）/ statistics（数组化聚合，直接读引擎 SoA 数组）/ observer（世代+节拍双触发器）/ experiment（嵌套 SimConfig + overrides 合并 + SphereEngine runner，run_single 每 tick 对齐终止条件置位）/ __main__ CLI（--rows/--cols/--sim-core）/ persistence/io（manifest+generations 落盘）；快照桥 broker：每 100 tick 采 JSON 快照（含个体明细广播），WebSocket 推流，环形只有标量防 OOM，新客户端连上补发最新快照；pytest 全量 50 例通过（新增 16 例）|
| 2026-09-07 | 提交并推送模块三、四到 GitHub（origin/main） | 提交：3ed6fd0（模块三）/ 8aefe06（模块四）；`git push origin main` 后本地与远端同步 | GitHub 远端 `fish827-08/digital-life-sphere`（SSH）为唯一备份，推送后 3 个提交（模块二收尾~模块四）全部上云 |
| 2026-09-07 | **L1 斑块资源（守恒版）** — feature/l1-patchy 分支 | ResourceConfig 加 7 个斑块参数 + 断言；ResourceField 守恒分布（容量+再生守恒）；引擎接入（patch_seed 独立 rng，patchy 时 Rust regrow 回退 Python）；修复 `set_initial_fill` 不存在的潜在 bug；tests/test_patchy_resource.py 18 例全过；全量 Python 路径 39 例通过；experiments/patchy_vs_uniform.py 对比实验：总食物守恒（差异<1%）、种群 0/3 灭绝、空间聚集未显现（无感知基因，预期内） | 语言涌现五要素第一步：资源空间异质化。守恒设计保证不破坏平衡。生态效应需等 L3 感知基因上线才能观测 |
| 2026-09-07 | **L2+L3 愉悦度+信号+感知** — feature/l1-patchy 分支 | L2 信号场 world/signal_field.py（田字格 4 子格=16 模式 uint8，duration=50tick，11 例测试）；L2 愉悦度（四数组 _valence/_arousal/_expectation(N,120)/_baseline，RPE 预测误差驱动，120 情境=能量5×食物4×邻居3×信号2，EWMA α=0.05，乐观初始化，繁殖继承，9 例测试）；L3 g14 感知（移动决策综合得分=感知×(食物×0.7+信号×0.3)+群居×密度）；L3 g15 信号（发射概率=g15，耗能 0.5，模式=状态哈希，信息增益接入愉悦度）；PleasureConfig 加入 SimConfig；tests/test_pleasure.py 9 例 + tests/test_perception_signal.py 10 例全过；全量 90 例通过；experiments/long_run_l2l3.py 长跑脚本；5000tick 实验：117tick/s@N=500，种群爆发到 5000，信号密度 6073 格，g14 上升(0.52→0.55)被保留，g15 下降(0.47→0.30)发射耗能>接收收益，愉悦度 RPE 动力学符合预期（乐观→失望-0.58→学习→回升-0.09） | 语言涌现硬件三件套：感知+信号+愉悦度。关键发现：g15 信号基因因发射耗能>接收收益被淘汰，说明信号系统必须配"接收收益"才能维持——后续需增大信号在移动决策中的权重或让信号直接指示食物。Rust 环境已搭建（rustup 1.98.1+maturin，sim_core 编译成功） |
| 2026-09-08 | **L7e patchy 再生下沉（feature/dev-l7 首任务）** | sim_core 新增 `regrow_patchy`（空间倍率守恒版 regrow：每格恢复 = 再生率×温度因子×patch/bg 倍率）；引擎 `_advance_one_tick` 三路分流（uniform→rust / patchy→rust_patchy / 旧 pyd 特性检测回退 Python）；函数级对拍 4 例（含长度校验）+ 引擎级 patchy 双路径 2 例；全量 pytest 113 例通过 + 1 跳过；基准（patchy，N0=5000）：Python 120 tick/s vs Rust 896 tick/s ≈ 7.5x | 实验脚本同批修复：`resources.patchy=True` 死代码→`distribution="patchy"`；增量落盘 CSV；ticks 参数化（`long_run_l4l5.py 300000`）|
| 2026-09-08 | **T4 繁殖 Step8 下沉 Rust（feature/dev-l7）** | sim_core 新增 `reproduce_batch`（RNG 数组预生成入参，Java 代基因/能量/文化继承在 Rust 内完成）；引擎双路径接入 + 函数级对拍 3 例 + 引擎级对拍（含死亡重排后的空槽位处理）；全量 pytest 139 例通过 | 繁殖判定与捕食/植物化的能量修正解耦（判定统一回 Python 计算），双路径 500tick 内逐位一致，39 行核心逻辑下沉，热路径进一步瘦身 |
| 2026-09-08 | **云端第一轮 C1/C2/C3 合入（e531853）** | C1 信号发射下沉 `signal_emit`（函数级对拍 6/6）；C2 愉悦度更新下沉 `pleasure_update`（对拍 5/5）；C3 基因系统扩展性（G1 加基因五步曲 / G2 双写漂移检测 18 例 / G3 元数据 / G4 预留位规则），gene_count 24 位（18 接线 + 6 预留） | sim_core/lib.rs 三函数拼接，`validate_gene_wiring` 自动比对 Python↔Rust 基因索引，漂移即 RuntimeError |
| 2026-09-08 | **云端第二轮 C4/C5/C6 合入（bf256ae）** | C4 敏感性扫描（10 参数组合 × 3 seed 调度器 + 子进程隔离修复段错误）；C5 L10a 果实-种子传播 Rust 下沉 `fruit_charge`/`eat_fruit`（函数级对拍 6 例，先行版）；C6 T5 slots 预分配评估（结论暂不实现：收益 <10%、RNG 顺序风险高） | 合并后全量 pytest **152 passed 1 skipped**；敏感性扫描实验本体由云端继续运行，产出 results/sensitivity/ |
| 2026-09-08 | **30 万 tick L4+L5 长实验 + 分支整理** | 长实验跑满 300k tick（153 tick/s / 1956s），终局 g14=0.834 g15=0.351 g16=0.466 g19=0.590，trust=0.920，cult_div=0.0009；结果 experiments/long_run_l4l5_result.csv（入库）；feature/dev-l7 并入 main，冗余分支清理，远程单一 main 主干 | 30 万 tick 全程数据（tick/种群/基因/信任/能量逐 2000 采样）+ 完整解读推送到 gitee 供云端继续实验/分析 |
| 2026-09-09 | **D0 止血（六模型评估第一道动作，主）** | ① sig_density bug（`e.signal_field`→`e.signals`，此前恒 0，大世界 g15=0.69"信号价值"观测支撑不存在）；② cult_div 口径统一 `.std(axis=0).mean()`（大世界重算 0.116→0.0117）；③ 移动群居项量纲归一化（densities 按邻居上限 8 归一 + `social_move_weight` 进 SimulationConfig，双路径逐位对拍 PASS，全量 164 passed）；④ pred_cum 实时取 `death_cause_totals()` + elapsed_h 跨段累计；⑤ 废弃语言 0-100 综合分（signal_context_mi 恒 0 + 样本不足无区分度），改 7 维子分+相对零模型超额量口径；`run_long_experiment` 新增 `--seed/--rotation-period` | 提交 81ce3ed / f30acbc / b7121c3 / 2d52bd6，D0 结束后空闲位数（g17/g18/g20~g23）与基因注册表不变；大世界 1M 快照按新口径重度量（signal_context_mi=0.0、cult_div=0.0117、熵 0.266、解读一致性 0.96） |
| 2026-09-09 | **六模型外部评估 + 云端交接** | 6 份评估报告入库 `_eval_reports/` + 汇总意见 v2（共识：**停新增先证伪**、cult_div 是热扩散方程解非文化演化、信号是指示符非符号（无任意性/信息不对称/组合性/联合收益）、学习瓶颈被整表遗传短路、世代时间锁死 25 代≈0、"百万代"撤出主线→主形态=千代×十seed×有对照）；DeepSeek 深度评审（快世代配置 + 厚核薄驱性能序）；云端任务包 + 启动说明定稿 | 路线：D0 止血→D1 零模型→D2 信息结构（g15 无补贴上升为唯一验收门）→D3 世代压缩→D4 性能（D2 后）→D5 复现性；L8/L9/L10 明确推迟；数据/快照存 gitee（私有，云端取数） |
| 2026-09-09 | **性能线 S0+S3（评估路线 D4 先行项，main 直做）** | **S3 系综脚本 experiments/run_ensemble.py**：多进程 seed 并发（绕 GIL），子进程循环续段到跑满，自动合并 per-seed 终局 + mean±std → results/ensemble/<tag>/summary.{csv,json} + commit/参数 manifest；**S0 bincount 复用**：信号段与移动段的占用计数合并（本 tick 移动前位置一次 bincount 复用，3→2 次/tick，语义逐位不变） | 双重验证：全量 pytest **164 passed 1 skipped**（同 D0 基线，Python/Rust 双路径对拍逐位一致 → 纯等价重构）；60×120 patchy 基准 255 tick/s。S1 厚核薄驱 / S2 rayon 按路线**待 D2 冻结后**再启动 |
| 2026-09-09 | **D2 信息结构重构完成（云端 feat/info-structure → 合入 main b1001eb 起）** | 四大机制：①学习瓶颈（解读表不遗传，幼体有限观察样本归纳 + learning_rate/samples_max/maturity_ticks 参数）；②任意性码本（pattern=个体可遗传码本[状态]，可漂移可协商）；③信息不对称（感知半径4/噪声/softmax 采样）；④Steels 对齐（同格相遇概率性解读表+码本对齐）。新增 InfoStructureConfig（17 参数）+ 快照 v3 + tests/test_d2_info_structure.py 17 例 | Observed：**D2 全机制默认参数 9000 tick 灭绝**（灭绝前 g15 0.49→0.63）；Rust 侧暂未实现，启用时走 Python 路径；唯一验收门 = **g15 无手调补贴从 0.5 上升** |
| 2026-09-09 | **battle 参数实现（lifespan_mult + stay_prob，合入 main 69e63bc）** | lifespan_mult（寿命基准缩放，(0.05,8]）+ stay_prob（停驻率，[0,0.95)）双战役参数进 config + 23 新测试（含双路径逐位对拍/RNG顺序/fingerprint/旧存档回退/行为断言/边界断言） | 与 D3 世代压缩互补：不动昼夜直接缩放寿命/停驻=等效扩世界；合入后全量 **215 passed 1 skipped** |
| 2026-09-09 | **云端 D2.1 参数网格批次定稿**（评估-EVAL-云端任务包-D21网格批次-20260909.md） | 分阶段渐进找"生态稳定（N>1000×50k tick）且 g15 无补贴上升"参数：**夜1 学习率 lr={0.10,0.30}×s{42,43}**、**夜2 学习窗口 mat（learning_maturity_ticks）={2400,3600}×s{42,43}**，共 8 实验 × 60k tick，产出推 **feat/d2-grid** | 本地跑基准档（lr=0.15/mat=1200）与云端互补（避免重复）；简报仅列数字不判读 |
| 2026-09-09 | **本地 R1 崩点定位（S1×2 + S2×1，60k tick，纯 Python 路径）** | S1=learning+steels（s42/s43）、S2=+arbitrary_codebook（s42）；修复 run_d2_experiment.py 为**即时落盘**（每 log_interval 追加+flush，进程中断不再丢数据）后重跑 | 结果：三实验均存活至 60k（N=5000）未灭绝（对比全机制默认 9k 灭绝）；**S1 两 seed 终局逐项完全一致（g14=0.8376、g15=0.2381、trust=0.6163）**；S2 g15=0.3395、codebook_conv=0.7543（码本持续分化未收敛）。**基准档 g15 无上升 → D2 gate 未过**，结论待 D2.1 互补档 |
| 2026-09-09 | **可复现性中高危修复（feat/d2-vec ac22600）** | 引擎 init 补 `np.random.seed(config.seed)`（D2 感知噪声/Steels 配对消费的全局 np.random 原不受控，同 seed 二次运行结果不同，"3 seed 一致"判据混入非受控噪声）+ 2 新测试（同 seed 逐位一致/异 seed 有差异）；Steels 两路径接线向量化 helper（原 helper 未接线为死代码）；综合报告补 g14/g15 修复前口径标注；性能路线归档 D2 下沉归属 S1 | 全量 **217 passed 1 skipped**；修复前 R1 旧数据作废（不可复现），以新代码重跑为准 |
| 2026-09-09 | **档A Python 路径向量化（feat/d2-vec d613c8b）** | learning 瓶颈观察学习（np.where 批量 EWMA）、softmax 移动决策（邻居 mask 批量，含球面不定长邻居修复）、Steels 对齐（helper）三段 per-individual 循环向量化 | 双路径逐位对拍 + 全量通过；**实测球面 stride=120 下移动段收益有限（<10%~负）**——D2 正式性能收益归档给 S1 step_tick 下沉一并 Rust 化（见 docs/性能优化路线） |

## 进行中
- 模块四已完成（8aefe06）；L1 斑块资源（守恒版）已完成；L2 愉悦度+信号场、L3 感知(g14)+信号(g15)已完成（全量 90 例通过，5000tick 实验完成，100 万 tick 后台长跑中）；下一步：L4 行为决策（捕食 g16/植物化 g19）或 L5 学习与继承（工作记忆/文化传递/信誉表），或修复 g15 信号基因接收收益不足问题
- **T4 繁殖下沉已完成**（sim_core.reproduce_batch + 引擎双路径接入 + 函数级对拍 3 例 + 引擎级对拍验证 139 passed）；**云端两轮任务已全部合入**：第一轮 C1/C2/C3（e531853）、第二轮 C4 敏感性扫描脚本/C5 L10a 种子传播下沉/C6 T5 slots 评估（bf256ae，合并后全量 **152 passed 1 skipped**）
- **分支结构已整理**（2026-09-08）：feature/dev-l7 并入 main（快进），冗余分支（feature/snapshot、feature/cloud-tasks、feature/cloud-tasks-2、feature/l1-patchy、mainline）合并后清理；远程以单一 main 为主干，云端用新 feature 分支开发后合入
- **30 万 tick L4+L5 长实验已完成**（153 tick/s，约 33 分钟，结果 experiments/long_run_l4l5_result.csv，结果文件已 gitignore）
  - 种群：264 → 5000（贴容量上限）、信号密度 6004 格；g14 感知 0.55→0.83（强正选）、g15 信号 0.29→0.35（微弱保留）、g16 攻击 0.34→0.47、g19 植物化 0.63→0.59（稳定）
  - 涌现观察：文化多样性 cult_div 0.251→0.001（解读表趋同，但 std 非零仍在漂移）、trust 0.91→0.92（高信任环境）、valence/arousal 归零（幸福适应）
  - 解读（Observed）：感知是唯一强正选基因；信号 g15 从 L3 时代的"发射成本>收益被淘汰"转为"微弱保留"（L4 捕食后信号有代价但仍存续，非零漂移的文化多样性暗示信息有一定价值）；未观察到语言/符号化涌现的直接证据
- **R1 路线文档已完成**（后续发展路线.md：性能→扩展→特性→实验阶段分级，**优先级更新为 L9→L10→L8**，L7d 触发条件；已接入 AGENTS.md"当前阶段"）
- **A4 中性化建议已完成**（隐式选择压中性化建议.md：7 项偏置逐条当前方向/是否刻意/中性化方案，默认不改变行为）
- **R2 评估材料打包已完成**（评估材料打包清单.md：12 项材料 9 项 ✅，缺口 = A3 敏感性表待云端 C4）
- **评估后新路线（D0~D5，2026-09-09）**：D0 止血已全部合入 main（164 passed）；D1 零模型三开关（云端 feat/zero-model）与 D3 世代压缩扫描（云端今晚 feat/gen-scan）待跑；D2 信息结构重构 = 科学杠杆最高（云端 feat/info-structure，g15 无补贴上升为唯一验收门）；D4 厚核薄驱 `step_tick` fusion 待 D2 落地后本地启动；D5 复现性/分支治理本地收尾中——详见评估-EVAL-云端启动说明-20260909.md
- **R1 崩点定位已完成（本地，2026-09-09/10）**：S1×2 + S2×1 全部 60k tick 存活未灭绝；基准档 **g15 无上升（S1=0.238 两 seed 一致 / S2=0.340），D2 gate 未过**——判读口径已更新为"等 D2.1 互补档"
- **D2.1 参数网格（云端夜间，进行中）**：夜1 lr×s42/43 + 夜2 mat×s42/43 共 8 实验跑 feat/d2-grid；本地基准档（lr=0.15/mat=1200）已由 R1 覆盖
- **battle 参数验证批次（待跑）**：rp=300 × lifespan_mult={1.0, 0.5, 0.25}，验证世代压缩后生态稳定性 + g15 趋势（配套 battle-params 已合入 main）

## 待办事项
1. 模块一 SphereWorld 开发（✅ 已完成）
2. 模块二 引擎核心（✅ config/tick/lifecycle/engine 完成，基因扩充 g0~g13 完成）
3. 模块二 收尾：pytest 用例固化（✅ tests 7 例通过）+ observatory 快速长程验证（✅）
4. 模块三 Rust 热核 Sim-core（✅ 3.1~3.5 + 邻居表优化，已提交 3ed6fd0）
5. 模块四 observatory 适配 + 快照桥（✅ 已提交 8aefe06 并推送远端）
6. 模块五 前端渲染层（未开始）
7. **语言涌现扩展 L1 斑块资源（守恒版）**（✅ 已完成，feature/l1-patchy 分支）
8. **语言涌现扩展 L2 数组级扩展**（✅ 已完成：愉悦度四数组/田字格信号场，commit b2df50c）
9. **语言涌现扩展 L3 基因解码扩展**（✅ 已完成：g14 感知/g15 信号，commit b2df50c；g15 修复在 L4 提交中）
10. **语言涌现扩展 L4 行为决策逻辑**（✅ 已完成：捕食 g16/植物化 g19，gene_count→24）
11. **语言涌现扩展 L5 学习与继承机制**（✅ 已完成：信任学习/工作记忆4槽/文化传递解读表）
12. 语言涌现扩展 L6 Rust 下沉（热点循环 PyO3，未开始；Rust 路径已可用，新增功能在 Python 侧，性能 117 tick/s）
13. **评估路线 D0~D5**（D0 止血 ✅ → D1 零模型 ✅ → D2 实现 ✅ → D2.1 参数网格进行中 → D3 世代压缩 ✅ → D4 性能 S1/S2 待 D2 冻结 → D5 复现性 ✅ 部分）
14. **D2.1 gate 判定**：等云端 8 实验落地，判读"生态稳定 + g15 无补贴上升"；本地基准档未过
15. **battle 参数验证批次**（rp=300 × lifespan_mult={1.0, 0.5, 0.25}）——**不涉 D2 机制，走 Rust 快路径（≈255 tick/s），无需等 S1，可与 D2.1 判读并行立即跑**
16. **neutral 对照设计修正**（只冻结目标基因 g14/g15，保留 g0~g13 生存基因）
17. **S1 step_tick 厚核薄驱 + D2 四机制 Rust 下沉**（**前置 = D2 参数冻结（gate 判定/锁参），而非战役完成**；DeepSeek V4 Pro 高难任务；与 battle 验证并行）
18. **D2 正式战役：千代 × 10 seed × 四臂对照**（前置 = D2 gate 通过 **+ S1 落地**——最贵批次约 33h，S1 后跑省 10x+，这是性能优化收益的最大应用点）
19. 模块五 前端渲染层（未开始）

---

## 项目总体发展规划与重大实验安排

### 总体目标
在球形有限世界上构建**语言涌现最小充分条件的测试平台**：四机制（学习瓶颈/任意性码本/信息不对称/Steels 对齐）为语言涌现铺路，主实验形态 = **千代 × 10 seed × 四臂对照**（"百万代"已撤出主线——统计功效靠重复不靠长度）。

### 阶段路线（D0→D5）
| 阶段 | 内容 | 状态 |
|------|------|------|
| D0 | 止血：sig_density/cult_div 口径/w_social 量纲/pred_cum/语言综合分 | ✅ 合入 main（164 passed） |
| D1 | 零模型三开关四臂（neutral 全冻结 / signal_disabled / signal_mode random） | ✅ 云端完成；neutral 臂设计缺陷（灭绝），改用 sigdis+random 两臂；结论全部按修复后口径重估 |
| D2 | 信息结构重构四机制 + gate（g15 无补贴从 0.5 上升） | 🔄 实现 ✅；**D2.1 参数网格进行中，基准档 gate 未过** |
| D3 | 世代压缩（rotation_period 2400→300~600，世代 18.9k→100~300 tick） | ✅ 实测 τ≈7.3×rp；battle 参数（lifespan_mult/stay_prob）已入 main |
| D4 | 性能：S0/S3 已完成；S1 step_tick（等 D2 冻结）+ S2 rayon；D2 四机制随 S1 一并 Rust 化 | 🔄 S0+S3 ✅；S1/S2 待 D2 冻结 |
| D5 | 复现性/分支治理/对外口径修正 | 🔄 可复现性修复 ✅（np.random.seed + 217 passed）；综合报告口径标注 ✅ |

### 重大实验安排（当前批次表）
| 批次 | 归属 | 参数 | 状态 |
|------|------|------|------|
| R1 崩点定位 | 本地 | S1（learning+steels）×{42,43} + S2（+codebook）×{42}，60k tick | ✅ 完成；均存活，g15 无上升，D2 gate 基准档未过 |
| D2.1 夜1 学习率 | 云端 | lr={0.10,0.30} × seed{42,43}（60k tick） | 🔄 夜间跑 → feat/d2-grid |
| D2.1 夜2 学习窗口 | 云端 | mat（learning_maturity_ticks）={2400,3600} × seed{42,43} | 🔄 夜间跑 → feat/d2-grid |
| battle 参数验证 | 云端 | rp=300 × lifespan_mult={1.0,0.5,0.25} × s{42,43}（200k tick，Rust 快路径） | ⬜ 夜 3 批次已入云端任务包（D2.1 完成后跑 → feat/d2-grid） |
| D2 正式战役 | 待定 | 千代 × 10 seed × 四臂（neutral 修正后） | ⬜ 依赖 D2 gate + **S1 落地后跑（省 10x+）** |
| 云端补跑 | 云端 | 世代压缩/零模型已完成 | ✅ 已入库（gen_scan / ensemble/zero_*） |

## 文档目录索引（2026-09-11 整理）

> 维护原则：核心文档常驻根目录并保持更新；决策/试行中文档集中 `docs/决策与评审/`（方向定案后再转移）；废弃历史入 `_archive/`。

| 分类 | 位置 | 文档 |
|------|------|------|
| **核心文档** | 根目录 | PROGRESS.md（进度/规划/索引）、PROJECT-DESCRIPTION.md（功能）、MODULES.md（模块）、AGENTS.md（规则） |
| **技术设计** | 根目录 + docs/ | L1-L6 技术汇总、L7 性能设计、L8-L10 扩展建议、快照机制设计、docs/性能优化路线、docs/L7d-Rayon |
| **工程文档** | 根目录 | 隐式选择压审计清单、隐式选择压中性化建议、TASKS-DEV/CLOUD/SNAPSHOT |
| **评估定案** | 根目录（评估-EVAL-*） | 综合报告、汇总意见、D1零模型结果、D2信息结构、D3世代压缩、大世界数据分析、Prompt 模板等 |
| **外部评估原文** | _eval_reports/ | 六模型评审报告 + 汇总（历史史料） |
| **决策与评审（试行）** | docs/决策与评审/ | 战略转向（待评审）、内部评估-战略转向、可行性分析-g15、任务包 S1/S2/lr细扫/D21网格（执行/裁决中） |
| **归档（废弃）** | _archive/ | 后续发展路线（旧路线）、slots-preview（T5 不实现）、分支状态清单（一次性） |

## 问题与决策记录
| 日期 | 问题 | 决策 | 原因 |
|------|------|------|------|
| 2026-09-06 | 项目语言与性能架构 | Python 主语言 + Rust 热点下沉（PyO3），方案 4，热核边界选 **B（数值+资源场）** | 保留旧版资产，性能瓶颈下沉 |
| 2026-09-06 | 世界拓扑 | 经纬网格（等距圆柱投影），极点经度坍缩为一格，8 邻 | 极区格子小于赤道，符合球面特性 |
| 2026-09-06 | 网格分辨率 | 纬度 60 × 经度 120（7,200 格） | 2° 格 |
| 2026-09-06 | 光照模型 | 固定太阳 + 世界自转 → 昼夜扫掠；温度由光照映射（极地冷、夜晚略冷） | 简化模型，生态涌现靠环境梯度 |
| 2026-09-06 | 资源初始分布 | 均匀分布 | 差异靠再生与个体行为涌现 |
| 2026-09-06 | 快照推送频率 | 每 100 tick | 后续按实际效果调整 |
| 2026-09-06 | 项目组织 | 新项目 digital-life-sphere，代码复用旧项目 | 一步步重构，全程可控可理解 |
| 2026-09-06 | 开发顺序 | 先世界模块 → 再放生物 | 世界先独立可运行/可视化 |
| 2026-09-06 | tick 语义 | **tick 是抽象时间单位**：1 tick 打个比方 = 现实世界 1 秒；"这一秒世界发生的变化"= 引擎跑一次 `step()` 的整套流程（资源再生→代谢→维持→进食→移动→衰老→死亡→繁殖）；不写成与现实的硬映射 | 让世界观对齐：时间不再"神秘"，就是一句"每秒世界推进一帧" |
| 2026-09-06 | 基因扩充（第 2 批） | 新增 g11 温度偏好 / g12 繁殖冷却 / g13 群居性，全部挂靠既有机制（温度响应/繁殖/移动） | 继续丰富多样性，但**不新增子系统**，热路径改动最小 |
| 2026-09-06 | 基因位评估（参照 The Bibites 28 基因） | **加入**：光合(冷血收益)、恒温（费能换低温不减速）、邻格觅食、温度偏好、繁殖冷却、群居性；**暂缓**：食性/捕食（要"尸体+伤害"机制）、脂肪储能（要存储池）、感知/交流（独立感知子系统，是"生物间能否交流"的评估结论——等引擎稳定后另立模块再评）、器官分配（器官系统） | 优先接入与现存机制正交的基因，避免为单个基因造一整块新系统 |
| 2026-09-06 | "一出生就能繁衍"问题 | 引入**生命周期年龄**：未到成熟年龄（=寿命×15%，寿命由 g3 定）一律不能繁衍；能量需求随年龄变：幼体×1.6（长身体）/成年×1/老年×1.4（器官退化） | 防止新生命前几代疯狂爆发；寿命长则成熟晚→自然涌现"速生速死"vs"晚熟长寿"两种策略 |
| 2026-09-06 | 生命周期年龄长程验证（模块二收尾） | 新增 experiments/quick_check_age.py 做 4000 tick 快速长程检查：结果未灭绝、世代达 12、寿命 g3 std=622 保留、速生(g3低 均2285)vs晚熟(g3高 均3762)两类策略并存 → 机制生效 | 验证成熟门槛确实抑制"出生即生"、能量需求随年龄分化，同时长寿基因多样性不被冲掉 |
| 2026-09-06 | **寿命数值平衡（寿命 vs 昼夜）** | 寿命公式 `200 + g3×3800`（[200,4000]，小于一昼夜 2400）改为 **一昼夜 × (1 + g3×7)**（[2400,19200]，最短整整一昼夜、最长八昼夜），并抽象为引擎 `_lifespan()` 统一出口（挂 `light.rotation_period`，改昼夜设置自动缩放） | 原寿命全部小于/接近一昼夜，生物来不及对昼夜做反应；新基准保证所有个体至少活满一昼夜，仍保留"速生(候1昼夜)vs晚熟(候8昼夜)"分化 |
| 2026-09-06 | 3.3 管线切分（为什么两段式） | `_step_population` 的数值运算拆成 **stage1（第 1~3 步：光合/代谢/维持）** 与 **stage2（第 5~8 步：移动扣费/年龄/死亡/冷却/繁殖候选）**，中间的"进食（步骤 4）+ 移动抽样/觅食目标选择（RNG）"留 Python | 步骤 4~5 依赖 RNG 且在 Python 侧要按原顺序消费随机数（同种子同结果）；Rust 只碰确定性数值，才能逐位对拍 |
| 2026-09-06 | 3.3 对拍保证策略 | ① 数值纪律：Rust 只用 f64 四则 + min/max，禁 exp/pow → 与 numpy 逐位一致；② 函数级：同一份随机种群分别走 py_stage1/2 参考实现与 Rust stage1/2，assert_array_equal 全部输出；③ 引擎级：同 seed 两个引擎（use_sim_core 开关），逐 tick TickStats + 最终内部数组逐位比较 | 三层递进：先证函数位级一致，再证接入引擎后 RNG 消费顺序不变、整体行为不变 |
| 2026-09-07 | 3.4 regrow 接入方式 | 不在 ResourceField 内部加开关，而是引擎 `_advance_one_tick` 里双路径：use_sim_core 时组好温度数组调 sim_core.regrow，否则调 ResourceField.regrow | 与 3.3 步进下沉同一模式，资源场类保持零改动；sens=1.0 严格逐位，≠1 允许 ≤1-ULP（函数级 1e-9 容差兜底） |
| 2026-09-07 | **性能基准结论（重要）** | 2000 个体 × 300 tick 基准：rust 3.043s vs python 3.078s = **1.01x**，行为位级一致；cProfile 定位瓶颈：`SphereWorld.neighbors()` 逐个体调用 190,722 次占 **~77%**（移动目标选择 + 邻格觅食两个 per-individual Python 循环，内部反复 flat_to_rc/clip/rc_to_flat），其次 numpy.clip 0.99s | 已下沉的向量化数值段本就便宜（numpy 已向量化），真正的热点是"逐个体循环里的邻居索引计算"；优化方向定为：**预计算全网格邻居表**（类成员缓存，一次算好 (n_cells, 8) int64 数组，循环内直接查行）——纯 Python 即可，不依赖 Rust |
| 2026-09-07 | 基准复跑确认（使用 venv 解释器） | 全量 pytest 22 例通过；基准复跑 python 3.030s vs rust 3.049s = **0.99x**（噪声内持平 = 1x），回归校验 PASS、行为位级一致 | 确认握手结论：Rust 下沉的 regrow/stage1/stage2 数值段无净收益，提速方向在预计算邻居表 |
| 2026-09-07 | 3.5 consume_many 下沉方式（进食/邻格觅食） | 引擎第 4 步进食 + 邻格补吃的两处批量消耗都走双路径：use_sim_core 时组好数组调 `sim_core.consume_many`（grid 就地改、实吃量写回 out_taken），否则走原来 `ResourceField.consume_many` | 与 regrow/stage1/stage2 同一接入模式；本轮不重跑基准（位置循环是瓶颈，不在进食段）——全量 pytest 30 例通过即为回归无破坏的证据 |
| 2026-09-07 | **预计算邻居表优化（已实施）** | 只在 `SphereWorld` 构造期动手：一次性算好 `_nb_table`(n_cells,8)（普通格 8 邻，含经度环绕/极点坍缩提前折算）与 `_pole_nb`(2,cols)（两极各一行相邻纬度带），`neighbors()` 从"每次 flat_to_rc+clip+rc_to_flat 现算"改为按行号查表返回（普通格直接取表行、极点格取 pole_nb 行）；引擎两处调用点零改动 | 构造成本一次性 ~毫秒级，却消掉引擎每 tick 数十万次的小 numpy 调用堆栈；查表与现算在全部 7200 格逐位一致（新 tests/test_sphere_world.py 用旧算法做参照对拍）；基准：python 3.030s→0.615s、rust 3.049s→0.569s ≈ **5x**，双引擎位级一致 REGRESSION PASS。当前 rust/python=1.08x，说明剩余瓶颈已在 Python 侧（RNG 消费/逐 tick 数组拼接），数值段与邻居段均已足够便宜，Rust 加速的收益到头了 |
| 2026-09-07 | 模块四统计口径（GenerationStats.trait_means 用什么） | 基因（14 位原值）+ 派生 trait（寿命/代谢倍率/成熟年龄，与引擎公式同源） | 直接观测表现型语义（人确认的推荐口径），派生列公式与引擎 `_lifespan`/代谢/成熟年龄同一来源 → 统计口径不会与行为脱节 |
| 2026-09-07 | 模块四范围（observatory 是否连带快照桥） | 连带快照桥一起（PROGRESS 待办口径） | 项目整体按"观察台 + 直播桥"一次交付，前端渲染再单独进模块五 |
| 2026-09-07 | 模块四 run_single 终止条件 | 手动逐 tick 循环里，每 tick 后执行 `engine._finished = engine._end_condition_met()`（与 `engine.run()` 语义对齐） | 适配时发现手动循环不置 `_finished` 会导致灭绝/跑满不停（浪费 CPU 且 ended_reason 误报 max_ticks）；对齐后灭绝走 stop_on_extinction 提前停 |
| 2026-09-07 | 球面版 resource_distribution 组实验裁剪 | 只保留 dist_uniform_sparse / dist_uniform_rich 两个均匀对照，删除旧版 patchy（斑块）实验 | 球面资源场（模块一）只支持均匀填充，斑块机制未实现；不预设机制，诚实标注（等资源场支持后补回） |
| 2026-09-07 | 快照桥消息设计 | tick/总能量/总资源/平均能量等标量 +（广播版）个体明细 {id, flat, energy, generation, age}；环形缓冲只存标量版 | 前端渲染直接消费明细；5000 个体 × 4096 份缓冲会 OOM，明细只在广播时带；新客户端连上先补发最新标量快照（落点晚也能看到画面） |
| 2026-09-07 | **L1 斑块资源守恒设计** | 两条守恒：①容量守恒（Σ_capacity 与 uniform 相等，bg_cap_mult=(total-patch_area×mult)/bg_area，必须>0）②再生守恒（面积加权 patch_mult×patch_frac+bg_mult×bg_frac=1）；初始食物总量**不守恒**（patch 格填满 capacity×1.0、背景格 capacity×background_fill，这是 patchy 的核心特征——斑块富集/背景贫瘠） | 容量/再生守恒保证种群承载上限和时间供给不变→不破坏平衡；初始食物不守恒是设计选择（两种世界的初始条件本就不同），在实验中如实标注。如需严格对比可调整 background_fill 使总量相等 |
| 2026-09-07 | **L1 patch 中心随机数隔离** | patch 中心选择用独立 rng（np.random.default_rng(patch_seed)），patch_seed=config.seed；不消费引擎 self.rng | 保证同 seed 下 uniform 与 patchy 引擎的个体初始基因/位置分布完全一致（RNG 消费顺序不变）→ 可复现、可对拍；测试 test_engine_patchy_does_not_consume_engine_rng 验证 |
| 2026-09-07 | **L1 patchy 模式 Rust regrow 回退** | patchy 时 `_advance_one_tick` 自动用 Python `ResourceField.regrow`（含空间倍率），uniform 保持双路径对拍 | Rust 侧 `sim_core.regrow` 暂未支持 patch_mask/空间倍率；patchy 模式回退 Python 保证行为正确，不影响默认 uniform 的 Rust 加速。后续 L6 可在 Rust 侧加 patch 支持 |
| 2026-09-07 | **L1 默认参数保守化** | patch_count=30, patch_radius=2, patch_capacity_mult=3.0, patch_regrowth_mult=2.0, background_fill=0.1 | 初版默认 60斑块/半径3/倍率6 导致背景容量为负（覆盖面积过大）；保守参数保证 bg_cap_mult>0、可直接跑通；用户可在 config 中调大 |
| 2026-09-07 | **L1 实验结果解读** | 2000tick 内空间聚集度 patchy(4.67) 低于 uniform(5.37)、基因多样性几乎相同——**不是失败，是预期** | L1 只加了资源斑块，没有感知基因（g14），生物无法感知斑块差异→不会主动聚集；空间 CV 高来自随机分布波动而非主动聚集。生态效应需等 L3 感知基因 + 更长时间（10000+tick）才能观测。L1 的价值是基础设施就绪+守恒验证通过 |
| 2026-09-07 | **L2 信号场+愉悦度系统** | 田字格 4 子格=16 模式 uint8 存储，duration=50tick 衰减，覆盖写入，批量读写；愉悦度四数组 valence/arousal/expectation(120情境)/baseline，RPE 预测误差驱动（愉悦=实际−预期），EWMA 学习 α=0.05，乐观初始化，繁殖继承+噪声 | 信号场是语言涌现的物理载体（16 模式=足够的符号空间）；愉悦度是内在动机系统（超越食物/繁衍的第三驱动力），RPE 机制保证"超预期才愉悦"，避免愉悦度退化为能量代理 |
| 2026-09-07 | **L3 g14感知/g15信号+5000tick实验** | g14 移动决策综合食物×0.7+信号×0.3+群居×密度；g15 发射概率=基因值，模式=状态哈希（能量2位+食物1位+邻居1位=4位），耗能0.5；5000tick N=500 patchy seed=42：种群爆发到5000，信号密度6073，**g14被保留(0.522→0.554)，g15被淘汰(0.473→0.301)** | g15 淘汰是演化诚实反馈：发射耗能0.5 > 接收收益（信号权重仅~0.16，信息增益不直接影响生存），沉默者适合度更高。与理论分析一致——没有接收者的信号系统必被沉默者摧毁。愉悦度 RPE 动力学完全符合设计（乐观初始化→竞争期低于预期→学习降低预期→回升） |
| 2026-09-07 | **g15修复方案** | 发射成本 0.5→0.1，移动决策信号权重 0.3→0.5（食物×0.5+信号×0.5），信息增益 0.3→0.5 | 降低发射成本让信号基因不被纯成本淘汰；提高信号权重让接收收益>发射成本；提高信息增益让愉悦度间接影响生存（通过移动决策的信任学习） |
| 2026-09-07 | **L4 捕食(g16)+植物化(g19)** | 捕食：攻击概率=g16×0.2×饥饿度（能量低才攻击），耗能0.1，成功率=能量比×攻击力加成，成功获猎物40%能量+胃粮，猎物PREDATION死亡；植物化：移动概率=g0×(1-g19)，光合收益×(1+g19)，统一在stage1后补保证双路径一致；gene_count 16→24 | 捕食引入种间竞争和选择压力（攻击/防御策略），植物化引入生态位分化（固定生产者vs移动消费者）；饥饿驱动攻击避免过度捕食（饱食时不攻击）；gene_count扩展到24位为后续基因预留空间 |
| 2026-09-07 | **L5 信任学习+工作记忆+文化传递** | 信任：_trust初始0.5，真信号（有信号+有食物）trust+0.05，假信号trust-0.1，移动决策信号权重×trust；工作记忆：4槽round-robin记录食物丰富格，移动决策记忆格+0.3×perc；文化传递：_interpret(N,16)解读表（正=移向/负=逃避），幼体向邻格成体EWMA收敛(α=0.1)，繁殖继承+噪声 | 信任学习解决"信号欺骗"问题（被骗过的个体不再信任信号）；工作记忆提供时空上下文（4槽容量=语言涌现的必要条件）；文化传递是Kirby迭代学习模型的简化版——代际传递+学习瓶颈→自发涌现共享信号系统，这是语言涌现的核心机制 |
| 2026-09-07 | **双路径对拍修复（死亡+繁殖判定）** | 步骤7死亡判定和步骤8繁殖判定统一用Python重新计算（不用Rust的out_starved/out_expired/out_repro），因为捕食(5.5)和植物化(3.5)在Rust stage2之后执行，会改变energy | Rust stage2内部的死亡/繁殖判定基于stage2时的energy，而捕食和植物化在stage2之后改变energy，导致双路径不一致；统一用Python计算后，双路径在500tick内逐位一致（test_engine_internal_state_bitwise_equal通过） |
| 2026-09-07 | **died字段bug修复** | _step_population返回的died只算了starved+expired，漏了predation；改成n_starved+n_expired+n_predation | 捕食死亡不计入died导致统计口径错误（deaths_by_cause有PREDATION但died=0），测试test_predation_kills发现此bug |