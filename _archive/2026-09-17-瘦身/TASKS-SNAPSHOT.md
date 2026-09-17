# TASKS-SNAPSHOT.md — 快照线任务清单

> 分支：`feature/snapshot`（供 gitee 长实验线使用，那边执行）
> 基线：8ffa68c（含 review 修复 + 基因语义注册表）
> 依据设计文档：`快照机制设计文档.md`（已在本分支）
> 目标：长实验分段续跑，突破云环境单次会话 ~15 分钟限制

---

## 任务总览（一个长任务：快照机制 + 分段续跑）

| # | 任务 | 验收标准 | 预估 |
|---|------|---------|------|
| 1 | `SphereEngine.save_snapshot(path)` | 保存全部状态到 npz <2s | 1h |
| 2 | `SphereEngine.load_snapshot(path, config=None)` | 恢复后与连续运行逐位一致 | 2h |
| 3 | RNG 状态保存/恢复（pickle bit_generator.state） | 恢复后随机序列与不保存连续运行一致 | 1h |
| 4 | 世界状态保存/恢复（resource/signal/light） | 三场状态一致 | 1h |
| 5 | 单元测试：保存→恢复→逐 tick 对拍 | B==C 逐位一致（文档 8.1 验证法） | 1h |
| 6 | `experiments/run_segment.py` 续跑脚本 | 从快照恢复→跑 N tick→存快照→追加统计 | 1h |
| 7 | pred_cum 等脚本级统计恢复（方案 B：读末行） | 统计文件跨段连续 | 0.5h |
| 8 | cron 集成验证（每 3 分钟/20k tick） | 连续 3 段无错 | 1h |

总计：~8.5h（1~1.5 天）

---

## 关键实现决策（来自设计文档，实现时必须遵守）

1. **npz 而非 pickle**：`np.savez_compressed(path, **data)`；RNG dict 唯一例外（pickle 后存 object 数组）
2. **保存有效切片 `[:count]`**：当前引擎是动态数组，保存 `_id[:P]` 而非完整数组
3. **配置指纹校验**：恢复时 config=None → 从快照 `config_dict` 恢复；config 提供 → 校验 fingerprint 一致
4. **RNG 必须保存**：`rng.bit_generator.state` + patchy 模式 `_patch_rng.bit_generator.state`
5. **字段核对**：引擎用 `len(self._id)` 表示有效个体数（无 `_count` 字段），保存时用 `P = len(self._id)`
6. **Rust 编译注意**：`sim_core` 改动后需 `$env:Path="$env:USERPROFILE\.cargo\bin;$env:Path"; ..\.venv\Scripts\python.exe -m maturin develop --release`

---

## 实施顺序

1. 实现 `save_snapshot`（引擎方法，第 129 行后新增）—— 数组 + 元数据 + 配置指纹
2. 实现 `load_snapshot`（类方法）—— 版本/配置/gene_count 校验 + 数组恢复 + RNG 恢复
3. 写对拍测试 `tests/test_snapshot.py`
4. 写 `experiments/run_segment.py`
5. 手动跑 3 段验证状态连续 + 统计追加正确
6. 任务完成后：PR 合入 mainline；快照线后续由 gitee 那边的 cron 使用

## 对拍测试关键用例（必须通过）

```
跑 1000 tick → save A
从 A load → 再跑 1000 tick → B
不保存连续跑 2000 tick → C
assert B == C（所有数组逐位相等：_id/_flat/_energy/_genes/_age/...）
```