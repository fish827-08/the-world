# 云端任务包：S2 并行 + Rust 侧 RNG + 光温下沉（2026-09-10）

> **决策落定（2026-09-11，用户拍板 = 方案 A）**：采纳替代验收（≥1.7× 已达成 + **项④ 消分配/世界级物化收口**）；
> S3 纯 Rust 全引擎不立项（400 t/s 需放弃位级一致，路线级风险，缓议）。项④ 执行基准循
> `docs/S2-判读与替代目标-20260910.md` 第五节清单（浮点结合顺序保持 + 四步验证）。
> 判读已认可：400 t/s 在"位级一致纪律"下不可达（外推上限 ~2.1× → 80~95 t/s），科学吞吐
> 由 rp=300 世代压缩免费补足（10~30×）。

> 提供给 DeepSeek V4 Pro 的 S2 交接。S1（feat/s1-step-tick b8caf61）已完成：融合路径 step_tick、D2 四机制 4/4 下沉、双路径逐位对拍（≥1200 tick）、全量 221 passed。
> **Gate 原则（重要，承 S1 先例）：目标为 N=12000 ≥400 tick/s，但允许基于实测改判**——S1 已实证"数学不可达就诚实改判"，S2 同样：跑出测量基线后，若 400 不可达或收益/成本失控，如实报告并给最接近可行的替代目标，不要硬凑。

---

## 一、S1 遗留：为何仍慢（S1 诊断实测）

| 热点 | 数据 | 归属 |
|---|---|---|
| 参考路径 Python 编排内联 numpy | ~58% | 已被融合路径拿走（S1 后 ≈20~25%） |
| `predation_and_culture` | 13% | Rust 串行循环 → S2 rayon |
| `step_movement` | 8.7% | Rust 串行循环 → S2 rayon |
| 幼体文化学习（juvenile×邻×16 维均值） | 随年龄结构放大 | culture rayon |
| 邻格觅食逐元素 Python RNG 回调 | **3.5 ms/tick**（bounded_per_element 2.91ms） | → Rust 自带 RNG |
| 光照/温度每 tick 全场重算 | LightTempRust.compute | → 光温下沉 |
| 每 tick Vec 分配 + 世界级数组物化 | food_ratio/sig_present/densities/occ_pre | → 消除物化 |

S1 融合实测加速天花板 ~1.2~1.3×（Python 编排仅剩 ~20-25%）；**400 t/s 需约 8.3× → 必须靠本任务包的 S2 项**。

## 二、S2 目标与范围

**目标**：N=12000 基准 ≥400 tick/s（同 warmup 同状态对照；参考基线见 bench_s1.py，空闲时参考 48 t/s）。

**范围（优先级递减）**：
1. **rayon 并行**：movement / predation_and_culture / culture（当前串行 for，disjoint 写；写 new 数组后合并，避免竞态）
2. **Rust 侧自带 RNG**：消除逐元素 Python RNG 回调（bounded_per_element 2.91ms/tick 大头）——用 PCG64/达标 Lemire/zuricht 落 Rust，或把每 tick 的 RNG 请求批量等价化
3. **光照/温度下沉**：LightTempRust.compute 整体移入 step_tick
4. **消除分配**：每 tick Vec 分配 + food_ratio/sig_present/densities/occ_pre 世界级数组物化（复用 buffer / 见融合路径内部的字段登记）

**参考实现**：`sim_core/src/tick.rs`（~470 行）、`movement.rs/...`、`experiments/bench_s1.py / profile_s1.py / time_s1_phases.py / diag_s1_rng.py`、`docs/S1-实现状态与交接-20260910.md`

## 三、硬性约束（承 S1）

1. **对拍纪律**：并行 vs 串行逐位一致（S2 测试）；Rust 自带 RNG 与 Python numpy 同 seed **逐位一致**（至少 ≤1 ULP，按数值偏差政策登记）
2. **RNG 保序**：S1 的 RngSource 预言机契约延续；若引入 Rust 自带 RNG 替代回调，需同 seed 下输出与 Python 参考序列相同（否则对拍不成立、同 seed 可复现性破）
3. **双路径三安全**：Python 参考路径（预言机）保留不动；融合路径任何改动都跑 test_s1_parity
4. **D2 四机制语义不变**（参数入参化已就绪）

## 四、验收清单（可按实测调）

- [ ] rayon 并行三处后，parallel vs serial 逐位一致（测试）
- [ ] Rust RNG 替换后同 seed 全字段对拍（参考 vs 融合 ≥1200 tick）
- [ ] `bench_s1.py` 同 warmup 对照：N=12000 基准表（含 400 目标判读）
- [ ] 全量 pytest 221+ 通过
- [ ] 若 400 不可达：给出实测上限 + 最接近替代目标 + 依据（诚实改判，禁硬凑）

## 五、产出与报备

- 代码提交 `feat/s2-rayon-rng`（从 feat/s1-step-tick b8caf61 切），推 gitee
- 简报：基准表（N=12000 ref/fused 多档）+ 对拍结论 + 各 S2 项前后耗时（diag_s1_rng / time_s1_phases 口径）
- 不判读生态；数据只进 gitee；受阻停下报告