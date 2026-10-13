# experiments/archive/ — 旧线脚本归档（2026-10-13）

> **缘起**：`docs/技术文档/技术-代码体量体检与结构建议-20261013.md`（青梧）动作 1。
> **执行**：轻舟，分支 `feat/onboarding`；全部 `git mv` 纯移动，**未改一行行为**。
> **判据（活跃引用）**：被 `tests/`、`tools/`、`simulation/`、`world/`、`observatory/`、`core/`、`viz/`、
> `AGENTS.md`、或**现役设计文档**（纯斑块世界 20261011 / ENV-FIELD 20261008 / 设计总档案 v1.4）引用者
> **一律留驻原位**；其余（含仅被历史文档/台账**叙述性**提及者）入档。下表给出全部证据。
> **上一轮**：`../_legacy_scripts/`（2026-10-03 R342，28 个脚本从回收站收编）。本目录是新一轮
> （2026-10-13，94 个 `.py` → 67 入档 + 27 留驻）。两处皆为**只读留证区**，不是回收站。

## 一、留驻原位 —— 27 个脚本 + prerequisites.json（为何不能动）

| 构件 | 留驻证据（`文件:行` 均为实测 grep） |
|---|---|
| run_bare_world.py / read_bare_world.py | **当前线主探针 + 判读**：docs/项目状态.md、docs/预注册/预注册-纯斑块世界-四臂-20261011.md、docs/汇报/汇报-20261011-纯斑块世界四臂.md |
| preflight_check.py | tests/test_preflight_check.py、test_c8_prereq.py、test_r97_gain_calibration.py；`simulation/config.py:555-564` 注释锚定其为闸常量唯一出处 |
| batch_runner.py | 8 个测试文件 + tools/run_batch.py、tools/batch_progress.py |
| c8_prereq_check.py | `experiments/preflight_check.py:355` import + tests/test_c8_prereq.py |
| a4_verify_capacity.py | **16 个测试文件** + observatory/statistics.py、observatory/world_capacity.py + tools×3（容量核验最深载体） |
| steady_k_probe.py | 5 个测试文件 + tools/check_bg_lottery.py、tools/s1_rgm_scan.py（K 稳态探针 + 快照夹具） |
| scaling_rescale.py | tests/test_1410_k25_wiring.py + `tools/tick_denomination_audit.py:664`（重标四张表红线对齐） |
| s3_memory_probe.py | 7 个测试文件 + 6 个 tools 脚本（记忆线载体 + 快照夹具） |
| s2_depletion_probe.py | tests/test_r278_p3_device_flush.py、test_r336_criterion2_shard.py + ENV-FIELD 文档「6 现役」 |
| p1c_distance_probe.py / p1c_erasure_probe.py / p1c_density_sweep.py | 4 个测试文件 + tools/tier1_judge.py、tools/mirror_n4_recalc.py（可达性/擦除刻度） |
| t0_patch_geometry_probe.py / centroid_opt_probe.py | `docs/设计文档/设计-ENV-FIELD-物理场与格子产能-20261008.md:175`：**experiments/ 6 现役**之一 |
| smell_perf_probe.py / day_conservation_probe.py | `docs/设计文档/设计-纯斑块世界-底层规则自组织-20261011.md:47` 引为「**现役探针先例**」（N=0 构造技法） |
| scan_nonascii_print.py | R98 守卫本体：tests/test_r98_nonascii_print.py、test_callsite_hardcode_scan.py、test_config_fallback_audit.py |
| cstep_judge.py | tests/test_cstep_judge.py、tests/test_r110_criteria.py |
| d24_six_gates.py | tests/test_r97_gain_calibration.py（六门成组判据） |
| gate_pair_analysis.py | tests/test_gate_pair_analysis.py |
| r44_power_prerun.py | tests/test_r44_power_prerun.py |
| e057_repw_batch.py | tests/test_e057_batch.py（批编排靶向测试） |
| r97_calibration_judge.py | tests/test_r107_judge_v4.py（R97 判官靶向） |
| time_compress_check.py | `tests/test_1410_k25_wiring.py:164` 子进程实跑 |
| photoperiod_check.py | `tests/test_137_season_light.py:78` 引其 DLI 口径 |
| run_d2_experiment.py | AGENTS.md 明文工具（D2 调优专用） |
| prerequisites.json | `experiments/c8_prereq_check.py:46` 的 `PREREQ_FILE` 读取路径 |

> 说明：体检报告点名的 s2 耗竭线 / s3 记忆线 / p1c 擦除密度线 / a4 容量核验线「核心模块」——
> 按实测**均被测试/工具/现役文档耦合**（上表），故留驻；其**一次性兄弟脚本**已在下方归档。
> 留驻与否只看引用证据，不看线的新旧。

## 二、归档脚本（67 个）

🔴 **引用纪律（先读）**：

1. **K=2「留守/出走双稳态」= 已定性 bug 产物**：`resource_dynamics.py from_field` 用**第 0 格反推全局
   背景倍率**（R319/R326 抽奖缺陷），不是涌现。证据：`docs/设计文档/预注册-S2判据②-新装置档-20261002.md:16-19`。
   ⇒ 下表 `s2_*` 四件读出、`_judge_s2g2`，及以其世界模型为底的批（批A/批B 气味引导）**仅供复现，
   禁止引用其结论**。
2. 其余批次：引用任何归档脚本/数据的历史数字前，先查 `docs/实验台账.md` 与 `docs/实验记录.md` 的有效性列
   （2026-09~10 曾连续查出多批因抽奖缺陷 / 默认值回落 / 半硬编码而无效）。

| 脚本 | 用途（代码头注首行） |
|---|---|
| _compare_trees | 比较两棵树：文件清单、体量、快速哈希相等性 |
| _crosscheck_r336_vs_60k | 交叉验证：R336(新档) vs 旗舰60k(旧档) —— rd on/off 效应量是否一致 |
| _judge_s2g2 | 判读 S2 判据②（新装置档 12 run），严格按预注册稿 v4 口径 🔴K=2 |
| _verify_r336_table | 天平独立复核 R336 §之五「跨批交叉验证」表数字与支别分层 |
| a4_attrib_data | A4 溯源数据采集 —— 为《崩溃溯源报告》提供 [实测] 数据 |
| a4_backfill_pred_frac | D-16 回算：给已跑完的 R19 验收 summary 补 R38③ 分层判据 |
| a4_control_neighbors | A4 控制实验：是"方向偏置"还是"只看4个邻居"导致崩溃？ |
| a4_drift_isolate | A4 漂移源隔离探针（青梧 · 2026-09-26） |
| a4_e1_neighbor_order | A4-E1：邻居取法矩阵（uniform + N0=200） |
| a4_energy_probe | A4 能量/食物收支探针：ON vs OFF 的食物存量空间分布与种群位置 |
| a4_inflect | A4 细粒度拐点：ON vs OFF 前 240 tick，每 20 tick 采样 |
| a4_probe_latitude | A4 崩溃溯源探针 —— 验证"感知半径4 = nb[:4] 方向偏置"假设 |
| abar_age_probe | ā（平均机动效率）年龄结构收敛探针 —— 回答「测 ā 的标准 tick 数」 |
| active_set_probe | 活跃集探针（B 路线可行性基数）—— 不改引擎，只统计"真正需要每 tick 计算的格" |
| alpha_16v4_analysis | α 批配对分析：信号字母表 **16 vs 4**（同批、同参数） |
| analyze_bimodal_fitness | 适应度地形分析 v2 —— 判定 g16 进化分支（双峰）的必要条件 |
| analyze_migration | 迁徙判读汇总 —— 读多 seed summary.json，出三态判定 + 图 |
| analyze_migration_v3 | 迁徙判定 v3 —— 双检验（相位零分布 + 滞后对齐）多 seed 汇总 |
| ars_sweep | ARS 扫参/对照批（14.9）：卡点 1/卡点 6 旋钮 CLI，输出结构化 CSV |
| asm_perf_probe | R239/R258 ASM（模式仲裁）性能实测 —— 配对口径（R219） |
| bench_large_world | 大世界性能基准 |
| benchmark_sim_core | use_sim_core 性能基准（模块三 · 3.4 收尾） |
| c2_relay | C2 接力器 —— cstep1b 跑完后自动启动 cstep2rand |
| corpse_call_profile | T-G profile：`_step_corpse_decay` 内全场 `_regrowth_amount` 调用量/收益（v1 前置） |
| d26_oracle_funnel | D-26a：oracle 四环节诊断 |
| d27_dose_fit | D-27④-B：oracle 剂量-响应拟合与可达性检查（R86/R88/R91） |
| endurance_probe | 续航/可达性探针（14.9 Q&A）—— 用引擎实测 |
| floor_tax_probe | 地板税分解探针 —— 480×960 T4 档「N≈0 每 tick 固定开销」逐段归因 |
| gain_ladder_probe | speed_gain 定档探针（解析版）—— 替代「用实测 ā 反解 gain」 |
| language_analysis | 语言涌现分析脚本 |
| line_profile_probe | 逐行性能剖析 `_step_population`（480×960）—— "每格 0.17 μs"花在哪一行 |
| long_run_l2l3 | L2+L3 系统长跑实验（愉悦度+信号+感知） |
| long_run_l4l5 | L4+L5 长实验（60 万 tick）：观察语言涌现迹象 |
| nb_table_probe | 邻居表代价探针 —— `_nb_table` 补齐到 cols 列的内存/cache 代价 |
| p1_profile | P1 profile（R210 §三）—— 「60–74 μs/体」花在哪 |
| patchy_vs_uniform | L1 斑块资源（守恒版）的生态对比实验 |
| perf_budget_calc | 480×960 / 满载 2 万+ 个体的机时与内存预算（纯算术） |
| plot_r196 | R196 光驱动再生 —— 迁徙涌现图（多 seed） |
| power_diagnosis | 检验适配性诊断 —— 解释相位零分布 p 为何不稳定 |
| pred_rate_rule_test | 捕食「每 tick 概率」守恒口径对比（14.10 / P0.0 A2） |
| probe_band_movement | S4 判据：季节开 ⇒ 富集纬度带确实随时间周期移动？ |
| probe_individual_migration | 个体级迁徙验证 v3 —— 引擎原生稳定 ID 配对，分段落盘 |
| probe_individual_migration_control | 个体级迁徙决定性对照（v2：相邻点配对 + 三臂） |
| probe_migration | 迁徙探针 v2 —— 种群质心纬度是否跟随季节资源带移动 |
| quick_check_age | 生命周期年龄机制的快速长程验证（模块二收尾） |
| rd_resume_smoke | R233 T-F 冒烟：rd 开 + 快照续跑端到端 |
| recalc_all_lang_metrics | 批量重算所有归档快照的语言指标（F-D5 修复后） |
| run_ensemble | 种子系综调度器（多进程 seed 系综） |
| run_long_experiment | 长实验脚本：自定义世界大小、密度控制、总 tick 数 |
| run_r19_acceptance_batch | R19 验收批（R24 口径）：4 机制主臂 + 3 对照臂，5 seed × 60k |
| run_segment | 分段续跑脚本：快照恢复 → 跑 N tick → 保存 → 增量写统计 |
| run_sensitivity_all | A3 敏感性扫描调度：串行跑全部参数组合，生成 summary.csv ⚠️import 见 §四 |
| s2_readout | S2 三判据读判（R225 §三 预注册）🔴K=2 |
| s2_r242_readout | R242 判读：几何放宽 + 背景低产能带，主看门B 🔴K=2 |
| s2_r243_attractor_readout | R243-① 吸引子刻画扫描 —— S0/S1 判读 🔴K=2 |
| s2_r243_s1_readout | R243-① S1 普查判读（32 seed 池） 🔴K=2 |
| scaling_equilibrium | 尺度均衡求解器（14.10）—— (世界大小 × 时间压缩) 代价面 |
| sensitivity_scan | A3 敏感性扫描：4 个高影响参数 |
| sparse_lockstep_ab | 稀疏化 B 引擎级 A/B —— sparse on/off 逐位对拍 + 配对计时 |
| speed_quant_probe | 速度量化探针（14.10）—— "速度基因有没有分辨力" |
| straightline_probe | 直线续航探针（14.9 Q&A）—— 满油直走能走多少格 |
| t4_aggregate | T4 交付聚合（480×960 稳态 K @ k=2.5）—— 6 seed 全量轨迹分层表 |
| tick_scaling_audit | 尺度重标定勘察探针（14.10 / 评审 P0.0 & P0.2） |
| time_compress_ensemble | A4 判据适用性检验：多种子均值曲线下跨 k 差还剩多少 |
| time_compress_multiseed | A4 增强：多臂 + 多种子 —— 尾段相对差是噪声还是漂移 |
| time_compress_noise_floor | A4 噪声地板：同 k 不同 seed 的相对差 |
| world_scale_probe | 世界尺度性能探针 —— 放大世界到底贵不贵 |

**包装脚本（4 个，同为旧线）**：`run_d2x_parallel.ps1`（D2x 并行驱动）、`run_r19_resumable.sh`
（R19 断点续跑）、`w2w3_watchdog.sh`（波2/3 守护进程）、`local_rerun_d2x.bat`（本地重跑，**未入 git**）。

## 三、数据产物 `_artifacts/`（57 项）

| 类别 | 数量 | git 状态 |
|---|---|---|
| a4_attrib_out.txt / a4_e1_out.txt / a4_probe_out.txt | 3 | **tracked**（git mv 入档，历史保留） |
| long_*.csv（长跑轨迹） | 40 | 未跟踪（`.gitignore` 拦截，工作树产物） |
| manifest_*.json（d2x 批 manifest） | 9 | 未跟踪（同上） |
| baseline/small/state_lang_metrics*.json | 3 | 未跟踪（同上） |
| lang_metrics_v2/（语言指标快照 JSON ×23） | 1 目录 | 未跟踪（同上） |
| local_rerun_d2x.bat | 1 | 未跟踪（同上） |

> ⚠️ 未跟踪项 = **新克隆不含**（它们从未入 git）；本次仅在工作树内移位、纳入本 README 留证。
> 纪律上实验数据属 `the-world-data/` 独立仓——历史遗留在此的产物**不删除**，只归档。

## 四、复算须知（移动带来的已知失效点）

1. **import 破损（1 处）**：`run_sensitivity_all.py` 内 `from experiments.sensitivity_scan import …`
   → 复算时改 `from experiments.archive.sensitivity_scan import …`。
2. **相对路径失效**：`long_run_l4l5` / `run_segment` 引用的 `long_*.csv` 现在 `_artifacts/` 下；
   三个包装脚本（`.ps1`/`.sh`/`.bat`）内引用的 `experiments/xxx.py` 路径同样需改。
3. **运行方式**：按 `python experiments/archive/X.py` 直接运行（同目录裸 import 靠脚本目录入
   `sys.path` 仍可解析）；**不要**用 `-m experiments.archive.X`（同目录裸 import 会断）。
4. **留在原地的依赖仍有效**：`a4_drift_isolate`、`corpse_call_profile`、`floor_tax_probe`、
   `p1_profile`、`pred_rate_rule_test`、`rd_resume_smoke`、`sparse_lockstep_ab`、`speed_quant_probe`
   等引用的 `experiments.scaling_rescale` / `experiments.steady_k_probe`（均已留驻）不受影响；
   `time_compress_*` 三件引用的 `time_compress_check`（已留驻）同理。

## 五、验证

- `git status`：**92 个纯 rename**（experiments 73 + tools 19），无内容改动（`git diff --stat -M` 仅重命名）。
- 全量 pytest：移动后跑 `pytest tests/ -q`，失败集与移动前基线一致 ⇒ 测试仍绿（见提交信息 / 汇报）。
- 恢复原状：`git mv experiments/archive/<名>.py experiments/<名>.py` 即回滚（无内容变更可逆）。
