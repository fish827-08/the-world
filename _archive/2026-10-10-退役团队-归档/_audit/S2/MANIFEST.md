# 证据包 · S2

- 生成时间：2026-09-24 01:19:18
- 分支 / HEAD：`dev/terrain-s2` / `2726f88`
- `gitee/main`：`e0d415d`
- 未提交文件数：**0**
- 当前纪元（AGENT.md §十三 末条）：**14.8 🔴 云端自审与自优化（2026-09-24 立；fish 裁定「本地无法审核」）**
- 文件数：**14**

## 说明

13.6 S2（四读数）证据包。
① 阶段范围：只加读数（SpatialReadings，纯观测）+ CSV 六列 + result.spatial + 侧车续跑；零新机制、未改任何判据/阈值。
② 复算对账（seed42/6000t/C 臂，产物 carm_s42_6k.*）：访问率 44.68%(t1k)→53.37%(t6k) **逐位命中 R191 原文 44.7%→53.4%**；在斑块 100%（R191 93–100%）；存量比 0.9997（R191 0.94–1.00）；Σ再生 406.343（R190/188 406.3）；未访问 46.6%（R191 31–47%）；利用率 2.08%（R190 C_s42 2.2%，**tick 数不同 ⇒ 仅同量级**）。
③ 人读入口：S2_SMOKE_SUMMARY.md（含对账表、CSV 时序、DEL-8、红队三条）。
④ DEL-8 两层变异：del8_s2_M1_observe_{red,green}.txt（main 循环接线）/ del8_s2_M2_result_block_{red,green}.txt（产物接线）。
⑤ 修 a4 崩溃：uniform 世界 _patch_mask=None ⇒ 归一无斑块+斑块类读数=None（回归单测 test_uniform_world_patch_readings_are_none_not_crash）。
⑥ 快照与侧车（t.spatial.npz 形态）随包 ⇒ 可断点复现。

## 文件清单（完整 sha256 见同目录 `SHA256.txt`）

| 文件                                                                    | 字节     | 修改时间        | sha256(前16)      |
|-----------------------------------------------------------------------|--------|-------------|------------------|
| `_rerun_logs/terrain_s2_smoke/S2_SMOKE_SUMMARY.md`                    | 3,845  | 09-24 00:50 | 299f884b7b3a7e8d |
| `_rerun_logs/terrain_s2_smoke/carm_s42_6k.csv`                        | 3,108  | 09-24 00:43 | 993cb704a3148dc6 |
| `_rerun_logs/terrain_s2_smoke/carm_s42_6k.progress.json`              | 102    | 09-24 00:43 | 8c24d159b3a295df |
| `_rerun_logs/terrain_s2_smoke/carm_s42_6k.summary.json`               | 20,763 | 09-24 00:43 | ea595df399258b0b |
| `_rerun_logs/terrain_s2_smoke/commit_msg.txt`                         | 2,179  | 09-24 01:18 | e3cd8e97835b7753 |
| `_rerun_logs/terrain_s2_smoke/del8_s2_M1_observe_green.txt`           | 180    | 09-24 00:48 | 2ae95447c270ab60 |
| `_rerun_logs/terrain_s2_smoke/del8_s2_M1_observe_red.txt`             | 2,782  | 09-24 00:48 | eb1234b49f615f6b |
| `_rerun_logs/terrain_s2_smoke/del8_s2_M2_result_block_green.txt`      | 185    | 09-24 00:48 | e4264e9ee08328fa |
| `_rerun_logs/terrain_s2_smoke/del8_s2_M2_result_block_red.txt`        | 1,765  | 09-24 00:48 | e71155c1eba9e5e3 |
| `_rerun_logs/terrain_s2_smoke/del8_s2_mutation.py`                    | 3,392  | 09-24 00:48 | 21de82bf21d01e9c |
| `_rerun_logs/terrain_s2_smoke/log_carm_s42.txt`                       | 62     | 09-24 00:43 | 7151f3558cab9aa2 |
| `_rerun_logs/terrain_s2_smoke/snap/carm_s42_6k.snapshot.npz`          | 44,269 | 09-24 00:43 | 74a73077d170af54 |
| `_rerun_logs/terrain_s2_smoke/snap/carm_s42_6k.snapshot.rngstate.pkl` | 2,675  | 09-24 00:43 | ba21b711b91ca05a |
| `_rerun_logs/terrain_s2_smoke/snap/carm_s42_6k.spatial.npz`           | 1,209  | 09-24 00:43 | 4ba74339024da50a |
