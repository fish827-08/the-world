# 证据包 · S1

- 生成时间：2026-09-23 18:48:47
- 分支 / HEAD：`dev/terrain-s1` / `8323239`
- `gitee/main`：`663ca8c`
- 未提交文件数：**0**
- 当前纪元（AGENT.md §十三 末条）：**14.8 🔴 云端自审与自优化（2026-09-24 立；fish 裁定「本地无法审核」）**
- 文件数：**59**

## 说明

13.6 S1（地形参数化）证据包（终版 v2；含回板全文 board_post_S1.md 与全部自审留档）。
① 阶段范围：只做参数化（a4 补 --patch-count/--patch-radius/--patch-capacity-mult + switches 读回 + 三地形 preset）；不改机制、不动 Rust、未改任何判据/阈值/口径。
② 冒烟矩阵（7 run × 2k，seed42）：baseline=374｜ctrl135=95｜forest=10（6k 续跑仍 9）｜grass=65｜desert=5｜ctrl_carm(严格C臂)=15｜forest_carm(严格C臂)=12 ⇒ 低 N 属 bgzero 绑定 regime 固有，非地形接线缺陷。
③ 人读入口：S1_SMOKE_SUMMARY.md（读回证据/红队自审/C-B 臂歧义/哈希复核方法）+ board_post_S1.md（回板 7 问全文）。
④ 三份 --dry-run 全文：dryrun_terrain_{forest,grass,desert}.txt（各 4 run）。
⑤ DEL-8 两层变异：del8_L1_cli_to_build_{red,green}.txt + del8_L2_build_to_config_{red,green}.txt；del8_mutation_{red,green}.txt = 首次假绿教训。
⑥ 自审留档：self_audit_all.txt（c9/cli 绿；git 未推送=按 gitee/main 比对的分支假阳性）、self_audit_digest.txt（🟢 140 passed 基线未漂移）。
⑦ 哈希复核：sha256_verify.txt（59/59 OK；含 tools/self_audit.py SHA256.txt 3 列格式缺陷与绕过命令）。
⑧ 快照（snap*/*.npz + rngstate.pkl）随包 ⇒ 可断点复现。

## 文件清单（完整 sha256 见同目录 `SHA256.txt`）

| 文件                                                                                      | 字节      | 修改时间        | sha256(前16)      |
|-----------------------------------------------------------------------------------------|---------|-------------|------------------|
| `_rerun_logs/terrain_s1_smoke/S1_SMOKE_SUMMARY.md`                                      | 3,793   | 09-23 18:42 | 10e947ccb542eb8e |
| `_rerun_logs/terrain_s1_smoke/baseline_s42_smoke.csv`                                   | 1,515   | 09-23 18:23 | eaddd283ca901bbb |
| `_rerun_logs/terrain_s1_smoke/baseline_s42_smoke.progress.json`                         | 103     | 09-23 18:23 | 15973921bb794fcd |
| `_rerun_logs/terrain_s1_smoke/baseline_s42_smoke.summary.json`                          | 19,755  | 09-23 18:23 | f39738a6e882b0b7 |
| `_rerun_logs/terrain_s1_smoke/board_post_S1.md`                                         | 9,802   | 09-23 18:48 | 071c43f9a134b795 |
| `_rerun_logs/terrain_s1_smoke/commit_msg.txt`                                           | 1,944   | 09-23 18:41 | 4afe1a8b7f1d26f1 |
| `_rerun_logs/terrain_s1_smoke/commit_msg_audit.txt`                                     | 716     | 09-23 18:43 | 6525786ff8ff4e08 |
| `_rerun_logs/terrain_s1_smoke/ctrl135_s42_smoke.csv`                                    | 1,499   | 09-23 18:24 | 1e1055ba2fc78573 |
| `_rerun_logs/terrain_s1_smoke/ctrl135_s42_smoke.progress.json`                          | 102     | 09-23 18:24 | 0bd9a61c6fdff74d |
| `_rerun_logs/terrain_s1_smoke/ctrl135_s42_smoke.summary.json`                           | 19,723  | 09-23 18:24 | 211e188f97199e5d |
| `_rerun_logs/terrain_s1_smoke/ctrl_carm_s42_smoke.csv`                                  | 1,475   | 09-23 18:30 | 23988bfcd187417e |
| `_rerun_logs/terrain_s1_smoke/ctrl_carm_s42_smoke.progress.json`                        | 102     | 09-23 18:30 | 4eb42c6f2ebb9adf |
| `_rerun_logs/terrain_s1_smoke/ctrl_carm_s42_smoke.summary.json`                         | 19,726  | 09-23 18:30 | 7ed78e42a2680b52 |
| `_rerun_logs/terrain_s1_smoke/del8_L1_cli_to_build_green.txt`                           | 205     | 09-23 18:22 | 0b392c46c76b38b8 |
| `_rerun_logs/terrain_s1_smoke/del8_L1_cli_to_build_red.txt`                             | 1,223   | 09-23 18:22 | b122772fb582fadd |
| `_rerun_logs/terrain_s1_smoke/del8_L2_build_to_config_green.txt`                        | 208     | 09-23 18:22 | c406fc1499e50fdf |
| `_rerun_logs/terrain_s1_smoke/del8_L2_build_to_config_red.txt`                          | 11,246  | 09-23 18:22 | b91d896fc8a5a482 |
| `_rerun_logs/terrain_s1_smoke/del8_mutation.py`                                         | 4,501   | 09-23 18:22 | 1392e40dde6425f0 |
| `_rerun_logs/terrain_s1_smoke/del8_mutation_green.txt`                                  | 184     | 09-23 18:22 | 0a20974829f78960 |
| `_rerun_logs/terrain_s1_smoke/del8_mutation_red.txt`                                    | 246     | 09-23 18:22 | cdbcf3d156e8935d |
| `_rerun_logs/terrain_s1_smoke/desert_s42_smoke.csv`                                     | 1,425   | 09-23 18:25 | c4a52450d35ae579 |
| `_rerun_logs/terrain_s1_smoke/desert_s42_smoke.progress.json`                           | 101     | 09-23 18:25 | 54490a19afc7a8ea |
| `_rerun_logs/terrain_s1_smoke/desert_s42_smoke.summary.json`                            | 19,709  | 09-23 18:25 | fb809bfa4abd01ec |
| `_rerun_logs/terrain_s1_smoke/dryrun_terrain_desert.txt`                                | 2,777   | 09-23 18:18 | 444272bc60013fb3 |
| `_rerun_logs/terrain_s1_smoke/dryrun_terrain_forest.txt`                                | 2,776   | 09-23 18:18 | b31a201f1173208d |
| `_rerun_logs/terrain_s1_smoke/dryrun_terrain_grass.txt`                                 | 2,761   | 09-23 18:18 | 8efab057c2e08add |
| `_rerun_logs/terrain_s1_smoke/forest_carm_s42_smoke.csv`                                | 1,467   | 09-23 18:30 | dbea772b92c3c980 |
| `_rerun_logs/terrain_s1_smoke/forest_carm_s42_smoke.progress.json`                      | 102     | 09-23 18:30 | d78724dbcfe066ac |
| `_rerun_logs/terrain_s1_smoke/forest_carm_s42_smoke.summary.json`                       | 19,716  | 09-23 18:30 | 6bae18571ed0772d |
| `_rerun_logs/terrain_s1_smoke/forest_s42_smoke.csv`                                     | 2,729   | 09-23 18:26 | 2d62292b60b02af0 |
| `_rerun_logs/terrain_s1_smoke/forest_s42_smoke.progress.json`                           | 101     | 09-23 18:26 | ad6363f6de2fdd18 |
| `_rerun_logs/terrain_s1_smoke/forest_s42_smoke.summary.json`                            | 19,720  | 09-23 18:26 | a348cd2559f13879 |
| `_rerun_logs/terrain_s1_smoke/grass_s42_smoke.csv`                                      | 1,492   | 09-23 18:25 | 0f30705f85d0995c |
| `_rerun_logs/terrain_s1_smoke/grass_s42_smoke.progress.json`                            | 102     | 09-23 18:25 | 3bac5f414048a933 |
| `_rerun_logs/terrain_s1_smoke/grass_s42_smoke.summary.json`                             | 19,723  | 09-23 18:25 | 2a1c2ec01a3d8284 |
| `_rerun_logs/terrain_s1_smoke/self_audit_all.txt`                                       | 5,905   | 09-23 18:43 | 36936a3784892120 |
| `_rerun_logs/terrain_s1_smoke/self_audit_digest.txt`                                    | 1,378   | 09-23 18:46 | 094fcd869650fda5 |
| `_rerun_logs/terrain_s1_smoke/sha256_verify.txt`                                        | 1,145   | 09-23 18:47 | 566b05e98cd32dc7 |
| `_rerun_logs/terrain_s1_smoke/smoke_baseline_log.txt`                                   | 63      | 09-23 18:23 | cf9b4ee011453348 |
| `_rerun_logs/terrain_s1_smoke/smoke_ctrl135_log.txt`                                    | 61      | 09-23 18:24 | cfcab7c720d45011 |
| `_rerun_logs/terrain_s1_smoke/smoke_ctrl_carm_log.txt`                                  | 61      | 09-23 18:30 | 4b1c5faddb4d2e70 |
| `_rerun_logs/terrain_s1_smoke/smoke_desert_log.txt`                                     | 60      | 09-23 18:25 | 5041138df5af2adf |
| `_rerun_logs/terrain_s1_smoke/smoke_forest_carm_log.txt`                                | 61      | 09-23 18:30 | 62c506d87d4010a7 |
| `_rerun_logs/terrain_s1_smoke/smoke_forest_log.txt`                                     | 164     | 09-23 18:26 | f6ec1f3339890c0c |
| `_rerun_logs/terrain_s1_smoke/smoke_grass_log.txt`                                      | 61      | 09-23 18:25 | 782c48866bb12231 |
| `_rerun_logs/terrain_s1_smoke/snap/forest_s42_smoke.snapshot.npz`                       | 41,311  | 09-23 18:26 | 017d4503f741e18b |
| `_rerun_logs/terrain_s1_smoke/snap/forest_s42_smoke.snapshot.rngstate.pkl`              | 2,674   | 09-23 18:26 | 03c65e5280f6f250 |
| `_rerun_logs/terrain_s1_smoke/snap_base/baseline_s42_smoke.snapshot.npz`                | 428,602 | 09-23 18:23 | e287d4e80a6d3b41 |
| `_rerun_logs/terrain_s1_smoke/snap_base/baseline_s42_smoke.snapshot.rngstate.pkl`       | 2,674   | 09-23 18:23 | 085e1bfe032107e7 |
| `_rerun_logs/terrain_s1_smoke/snap_carm/forest_carm_s42_smoke.snapshot.npz`             | 40,303  | 09-23 18:30 | c9aedfed28c610f3 |
| `_rerun_logs/terrain_s1_smoke/snap_carm/forest_carm_s42_smoke.snapshot.rngstate.pkl`    | 2,674   | 09-23 18:30 | b5b114ff3ddeeee3 |
| `_rerun_logs/terrain_s1_smoke/snap_carm_ctrl/ctrl_carm_s42_smoke.snapshot.npz`          | 41,469  | 09-23 18:30 | 8ada654947eb9573 |
| `_rerun_logs/terrain_s1_smoke/snap_carm_ctrl/ctrl_carm_s42_smoke.snapshot.rngstate.pkl` | 2,675   | 09-23 18:30 | 85a5bdbbbd892650 |
| `_rerun_logs/terrain_s1_smoke/snap_ctrl/ctrl135_s42_smoke.snapshot.npz`                 | 122,497 | 09-23 18:24 | 033c6bddf2566c0d |
| `_rerun_logs/terrain_s1_smoke/snap_ctrl/ctrl135_s42_smoke.snapshot.rngstate.pkl`        | 2,675   | 09-23 18:24 | 582adfc62aecb604 |
| `_rerun_logs/terrain_s1_smoke/snap_d/desert_s42_smoke.snapshot.npz`                     | 33,606  | 09-23 18:25 | be364e8f23f728af |
| `_rerun_logs/terrain_s1_smoke/snap_d/desert_s42_smoke.snapshot.rngstate.pkl`            | 2,675   | 09-23 18:25 | 0a59debeb7db32ea |
| `_rerun_logs/terrain_s1_smoke/snap_g/grass_s42_smoke.snapshot.npz`                      | 95,107  | 09-23 18:25 | e2374643cb0ed07f |
| `_rerun_logs/terrain_s1_smoke/snap_g/grass_s42_smoke.snapshot.rngstate.pkl`             | 2,674   | 09-23 18:25 | d4b83562d5fb7e0c |
