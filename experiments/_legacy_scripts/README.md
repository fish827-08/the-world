# `_legacy_scripts/` —— 历史判读/编排脚本收编区

> **为什么有这个目录**（R342 文件存放规则，2026-10-03）：
> 这批脚本原先散在 `_trash_local/`——**该目录未进 `.gitignore` 也未入库**
> ⇒ 🔴 **它们等于"随时会丢"**：换机/换人拿不到 ⇒ **任何引用它们得出的结论都无法复现**。
> 10-02 已实测过这个代价：`run_smell_rd_01.sh` 若只在 `_trash_local/`，
> 云机上就是 `ls: No such file`（R341 记录）。
>
> **收编判据（客观、可复核）**：脚本文件名**被 `docs/` 或 `_share/` 引用**者收编
> ⇒ 即"某个已发布结论的复算入口"。**28/28 通过 `py_compile` 语法校验。**

---

## 一、🔴 溯源表（**每个脚本对应哪份已发布结论**）

| 脚本 | 出处（该脚本是这份结论的复算入口） |
|---|---|
| `_judge_s2g2.py` | `docs/设计文档/预注册-S2判据②-新装置档-20261002.md` |
| `_judge_smell4.py` | `docs/判读归档/R335-气味场四通道批判读-20261002.md` |
| `_judge_r324.py` | `docs/判读归档/R324-四条线正式判读与归档-20261002.md` |
| `_s3_judge.py` | `docs/实验记录.md` |
| `_s2_final_judge.py` | `docs/实验记录.md` |
| `_judge2_snr_probe.py` | `docs/设计文档/预注册-S2判据②-新装置档-20261002.md`（**信噪比 <1 的诊断脚本**） |
| `_branch_table.py` | `docs/判读归档/R324-四条线正式判读与归档-20261002.md`（**唯一正确的支别判定**：`bg_production_zero`） |
| `_analyze_dispersal.py` | `docs/判读归档/R315-留守型出走型-基因分化检验-20261001.md` |
| `_run_s2g2.py` | `docs/tasks/run_smell_rd_01.sh` 引用的编排器原型（**注意：新编排器已修正其 R218 bug**） |
| `_run_on169_172.py` | `_share/讨论板.md`（旗舰 60k on 臂编排） |
| `_t1b_batch_proto.py` | `_share/讨论板.md`（T1b 批原型） |
| `_bgfix_bitwise.py` | `_share/讨论板.md`（**R331 甲案验收：现档逐位零变化证据**） |
| `_rd_bgzero_1t.py` | `_share/讨论板.md`（rd bgzero 单 tick 校验） |
| `_rd_bmult_engine.py` | `_share/讨论板.md` |
| `_rd_opt_micro.py` | `_share/讨论板.md` |
| `_rd_a2_timing.py` | `_share/讨论板.md`（**rd-A 静默复测**） |
| `_rd_a2_timing_report.py` | `_share/讨论板.md`（同上·报告） |
| `_rd_a2_timing_q_report.py` | `_share/讨论板.md`（同上·16 跑 digest） |
| `_perf_bigN2.py` | `_share/讨论板.md` |
| `_perfline_hw_probe.py` | `_share/archive/讨论板-20260927.md` |
| `_snap_invariant.py` | `_share/archive/讨论板-20260927.md`（快照不变量） |
| `_t6_mutation_transcript.py` | `_share/archive/讨论板-20260927.md`（变异检查） |
| `_tc_default_digest.py` | `_share/archive/讨论板-20261001.md` |
| `_td_rd_floor.py` | `_share/archive/讨论板-20261001.md` |
| `_te_signal_bench.py` | `_share/archive/讨论板-20261001.md` |
| `_v1_bench.py` | `_share/archive/讨论板-20261001.md` |
| `_p2_ab.py` | `_share/archive/讨论板-20260927.md` |
| `_p2_bench.py` | `_share/archive/讨论板-20260927.md` |
| `_p2_lockstep.py` | `_share/archive/讨论板-20260927.md` |

---

## 二、🔴 使用注意（**别把它当成现行工具**）

1. **多数脚本是"一次性判读"用途**，参数写死在脚本内 ⇒ 复算时需按原批的参数改；
2. 🔴 **`_run_s2g2.py` 有已知 bug**（把同 seed 的两臂并行 ⇒ 违反 **R218** 同 seed 必须同机同序）
   ⇒ **现行编排器是 `docs/tasks/run_smell_rd_01.sh`**（已修 + 加 `selftest` 参数自检）；
3. 🔴 **`_judge2_snr_probe.py` 是诊断"判据信噪比 <1"的脚本**
   —— 它与 10-02 的结论"**判据② 尺子坏了**"直接相关，改判据形态前应先看它；
4. 脚本里的**路径写死**为原机器路径（如 `C:\Users\...\Desktop\temp\...`）⇒ 换机需改路径。

---

## 三、归档区其余内容（**已按R342 处理**）

| 目录 | 处置 |
|---|---|
| `_trash_local/_tree_8efb606/`（631 文件） | 代码树副本 ⇒ 移入 `_archive/frozen_trees/`（**审计价值**：复现"当时跑的是哪份代码"） |
| `_trash_local/_tree_t1b_base/`（636 文件） | 同上 |
| `_trash_local/` 其余 80 个未引用 `.py` + 126 个 `.txt` | 临时产物 ⇒ **进回收站**（R342 规则①：临时文件用完即删） |
| `_trash_local/_design_drafts_20261003/` | 旧设计稿 7 篇 ⇒ **移入 `_archive/`**（F-R10 不删） |

**证据**：`[实测]`（`grep` 引用统计 + `py_compile` 28/28 + 各目录 `find | wc -l`）｜`[文档]`（R342 `_share/规则-文件存放.md`）

— 天平 ⚖️ 2026-10-03
