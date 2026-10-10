# 证据包 · SMELL-USE

- 生成时间：2026-09-28 13:58:16
- 分支 / HEAD：`dev/yunqi-smell-v1` / `f806bca`
- `gitee/main`：`90aad0d`
- 未提交文件数：**0**
- 当前纪元（AGENT.md §十三 末条）：**14.8 🔴 云端自审与自优化（2026-09-24 立；fish 裁定「本地无法审核」）**
- 文件数：**17**

## 说明

R244 §二 气味场消费端：三处同式（Python/Rust/向量化）逐位一致 + 默认关逐位不变 + 配对性能（净增 0.24 ms/tick @N≈255）+ DEL-8 三变异红/绿

## 文件清单（完整 sha256 见同目录 `SHA256.txt`）

| 文件                                                          | 字节      | 修改时间        | sha256(前16)      |
|-------------------------------------------------------------|---------|-------------|------------------|
| `_rerun_logs/smell_v1_smoke/commit_msg_smell_use.txt`       | 3,282   | 09-28 13:56 | 90c82b7dd29a1581 |
| `_rerun_logs/smell_v1_smoke/del8use_M1_rust_term_green.txt` | 98      | 09-28 13:41 | 90d9370f450c12eb |
| `_rerun_logs/smell_v1_smoke/del8use_M1_rust_term_red.txt`   | 1,372   | 09-28 13:41 | bf1d7a3b75e04881 |
| `_rerun_logs/smell_v1_smoke/del8use_M2_py_term_green.txt`   | 98      | 09-28 13:41 | e294d58817276d53 |
| `_rerun_logs/smell_v1_smoke/del8use_M2_py_term_red.txt`     | 1,340   | 09-28 13:41 | da039912b668f387 |
| `_rerun_logs/smell_v1_smoke/del8use_M3_clip_green.txt`      | 98      | 09-28 13:41 | 90d9370f450c12eb |
| `_rerun_logs/smell_v1_smoke/del8use_M3_clip_red.txt`        | 1,324   | 09-28 13:41 | 1b0879588404e4ae |
| `_rerun_logs/smell_v1_smoke/pytest_full_use.txt`            | 1,163   | 09-28 13:56 | d9025d7b2e5933db |
| `_rerun_logs/smell_v1_smoke/self_audit_digest.txt`          | 1,695   | 09-28 13:50 | d854eee6cf66e041 |
| `_rerun_logs/smell_v1_smoke/smell_use_perf_full.txt`        | 3,398   | 09-28 13:48 | 0e95a81c96b077de |
| `results/smell_v1/smell_use_perf_N200.json`                 | 1,028   | 09-28 13:48 | 105e572120e2c440 |
| `results/smell_v1/smell_use_perf_N2000.json`                | 1,033   | 09-28 13:48 | 7c990ebe7b02054e |
| `sim_core/src/lib.rs`                                       | 42,531  | 09-28 13:56 | 93b2e10c8e23accd |
| `sim_core/src/movement.rs`                                  | 8,704   | 09-28 13:56 | 0950639467b6748b |
| `simulation/sphere_engine.py`                               | 344,580 | 09-28 13:56 | 5bd696998ffe732d |
| `tests/test_smell_use.py`                                   | 8,323   | 09-28 13:56 | 089bdf8101f41f5a |
| `world/smell_field.py`                                      | 21,228  | 09-28 13:56 | 87813de5d439e902 |
