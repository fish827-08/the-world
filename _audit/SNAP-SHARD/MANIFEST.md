# 证据包 · SNAP-SHARD

- 生成时间：2026-09-26 18:27:05
- 分支 / HEAD：`dev/yunqi-snap-shard` / `87e7072`
- `gitee/main`：`767e2d5`
- 未提交文件数：**0**
- 当前纪元（AGENT.md §十三 末条）：**14.8 🔴 云端自审与自优化（2026-09-24 立；fish 裁定「本地无法审核」）**
- 文件数：**14**

## 说明

R221 §四：快照/续跑/分片 —— 单测 6 例、验收端到端原文（续跑≡连续 + 分片声明）、DEL-8 三变异红/绿、digest 与全量留档

## 文件清单（完整 sha256 见同目录 `SHA256.txt`）

| 文件                                                            | 字节    | 修改时间        | sha256(前16)      |
|---------------------------------------------------------------|-------|-------------|------------------|
| `_rerun_logs/snap_shard_smoke/acc_resume_vs_continuous.txt`   | 6,683 | 09-26 18:17 | 84bd593082b30cca |
| `_rerun_logs/snap_shard_smoke/acc_shard.txt`                  | 1,485 | 09-26 18:17 | d76558b43719da76 |
| `_rerun_logs/snap_shard_smoke/commit_msg_snap_shard.txt`      | 2,682 | 09-26 18:26 | 4bd5459990623747 |
| `_rerun_logs/snap_shard_smoke/del8_M1_resume_start_green.txt` | 98    | 09-26 18:16 | 8fbdf7c8d1587f50 |
| `_rerun_logs/snap_shard_smoke/del8_M1_resume_start_red.txt`   | 1,758 | 09-26 18:16 | d1e03270a9783788 |
| `_rerun_logs/snap_shard_smoke/del8_M2_merge_rule_green.txt`   | 1,500 | 09-26 18:16 | cb5c856ae9ccf79f |
| `_rerun_logs/snap_shard_smoke/del8_M2_merge_rule_red.txt`     | 1,500 | 09-26 18:16 | 6943761e5babfecd |
| `_rerun_logs/snap_shard_smoke/del8_M2_sample_guard_green.txt` | 98    | 09-26 18:17 | 05bcb39ea3146a56 |
| `_rerun_logs/snap_shard_smoke/del8_M2_sample_guard_red.txt`   | 1,576 | 09-26 18:17 | 3f72c22b844fb768 |
| `_rerun_logs/snap_shard_smoke/del8_M3_shard_mod_green.txt`    | 98    | 09-26 18:17 | fe8935a2087aa615 |
| `_rerun_logs/snap_shard_smoke/del8_M3_shard_mod_red.txt`      | 1,808 | 09-26 18:17 | 449f13c4199cea61 |
| `_rerun_logs/snap_shard_smoke/pytest_full.txt`                | 1,163 | 09-26 18:26 | 573eabb962f7f828 |
| `_rerun_logs/snap_shard_smoke/self_audit_digest.txt`          | 1,695 | 09-26 18:22 | aea5b131f2fb9654 |
| `tests/test_221_snap_shard.py`                                | 8,999 | 09-26 18:16 | e3fcd24147e9defe |
