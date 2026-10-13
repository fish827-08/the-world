# tools/archive/ — 旧线小工具归档（2026-10-13）

> 同 `../../experiments/archive/README.md` 判据：**只服务于已完成旧实验线、且现流程不再调用**的
> 清点/对账/复核小工具移入本目录（体检报告动作 1「tools/ 同理评估」）。`git mv` 纯移动，未改行为。
> tools/ 49 → **留驻 30 + 归档 19**。
>
> 留驻判断的两类反例（评估过、维持原位）：
> - 通用设备（非某条线专属）：`batch_progress.py`（批跑进度条）、`shutdown_gate.py`（机时闸门）；
> - 现役机制/文档的守卫：`check_doc_consistency.py`（设计总档案一致性）、`l1l2_*`（L1/L2 机制验证）、
>   `wave2_perception_probe.py`、`check_p1a_switches.py`、`verify_r34[456]*`（v1.4 定稿引用）等——
>   均有活跃文档或代码引用（详见各自矩阵，证据同 README 主判据）。

| 工具 | 用途（代码头注首行）| 备注 |
|---|---|---|
| audit_d1_mono_p1c | 定位"单调下降"不成立的具体位置（诚实性核查） | A4 读出对账对 |
| audit_d1_recheck_p1c | D1 独立复核：重算 A4 探针 CSV，与 summary 声称值对账 | A4 读出对账对 |
| mirror_c1_three_blocks | 镜·C1 三阻塞「由红转守」复核件（R394④） | 镜像复核一次性 |
| mirror_e057_recalc | E-057 rep_w 档扫批独立复算（R394④） | 镜像复核一次性 |
| mirror_g6_recalc | 镜·G6-EMITRATE 独立复核件 | 镜像复核一次性 |
| mirror_m1_smoke_recalc | M1-SMOKE 独立复算（2026-10-05，PI 核数门件） | 镜像复核一次性 |
| mirror_n4_recalc | N1 密度批·镜独立复算（R371） | 镜像复核一次性 |
| mirror_r369_recalc | R369 §三两种 pooling 独立复算 | 镜像复核一次性 |
| mirror_s1_readback | 镜·S1 读回守销项独立复核件（R394④） | 镜像复核一次性 |
| mirror_s3_recalc | 镜·S3-RDSWITCH 独立复核件（R394④） | 镜像复核一次性 |
| mirror_zeropat_audit | ZEROPAT-AUDIT 独立复算（R394④） | 镜像复核一次性 |
| pred_flux_recheck | R367-N3 评估稿判读脚本（零机时、只读既有产物） | N3 评估稿已交付 |
| r163_corpse_audit | R163 独立复核：尸体—食腐通道裁决性探针（老工线 2026-09-22） | 复核已完成 |
| r164_ledger_impact | 记账污染量化（R163 附带，2026-09-23 更正版） | 复核已完成 |
| terrain_diag_multiseed | S3 判据风险决策诊断②：多 seed 复核 | S3 诊断一次性 |
| terrain_diag_seed_basin | S3 判据风险决策诊断③：提高初始投放在非增长 seed 上是否有效 | S3 诊断一次性 |
| verify_membit_field | P1 门字段静默坑 —— 复现与守卫（砚，v1.2 复核 §二） | 复核已完成 |
| verify_r346_p1_memory_branch | v1.2 复核实跑证据：`memory_v2` 两分支下 `mem_bit` 实测 | 复核已完成 |
| yan_zeropat_data_audit | ZEROPAT-AUDIT · 砚数据面独立清算（2026-10-08，纯 stdlib、只读） | 清算已完成 |

**复算须知**：镜像/对账类部分脚本 import 留驻的 experiments 模块（如
`from experiments.s3_memory_probe import …`）——移动后 import 路径不变、仍有效；按
`python tools/archive/X.py` 直接运行即可。

**引用纪律**：与 `experiments/archive/README.md` §二 同——引用这里任何工具的旧结论数字前，
先查 `docs/实验台账.md` / `docs/实验记录.md` 的有效性列；K=2 相关（audit_d1_*、mirror_n4_recalc）
所服务的批次处于抽奖缺陷阴影内，仅供复现，禁止引用其结论。

## 二次核查（2026-10-13 晚，viz v1 合并后）

- **留驻 30 件 py 逐件重扫**（判据同 `../../experiments/archive/README.md` §一：tests/、现役代码、
  AGENTS/ONBOARDING/架构地图、现役设计文档/预注册）——**引用证据无变化、零新增入档**。
- **散置输出件** `_probe_b8.json` / `_probe_diff_smoke.json`（10-11 生成；根 `.gitignore` 的
  `*.json` 忽略 ⇒ 本就不入 git）：系 **C0 能量线**冒烟读数，生成器在未合并分支
  `feat/c0-energy-world:experiments/run_c0_energy.py`；该线待接续 ⇒ **留驻原位**（读数即证据）。
- 留驻件多数被**测试直接 import**（如 `a4_verify_capacity` 系 16 个测试文件的夹具载体）；
  进一步归档只能"连测试一起挪"（测试重构，非纯移动）⇒ 不属本目录范围，须单独立项评估。
