# J2 门跑数据受理与审计停手项裁定（2026-10-22）

> 裁定人：评审（总审）｜依据：owner 派工、rev2 `benchmarks/atmo/gate/J2_prereg_rev2_20261022.md` §七、台账 §37
> **状态：门跑数据 ACCEPTED（仅“输入受理”，不是 J2 PASS/FAIL；判读归 AV）**

---

## 一、裁定摘要

1. **A2 门跑数据受理**：rev2 协议的门跑已完成，产物完整、双跑逐位、manifest 合同命中；
   评审独立复验 **33/33 PASS**（§二）。
2. **审计停手项裁定（A2 报告 §6.1，选项 a）**：`atmo_core atmo` 的 **2 个 MEDIUM**
   （`atmo/physics/j2/make_ic.py:25 SST_EQ=300.0`、`:26 SST_DELTA=30.0`，规则 `constant_climate_field`）
   **按“预注册 IC 协议常量”登记接受**，不返工、不重跑、不改名规避。
   裁定理由与约束见 §三。
3. **放行 AV 判读**：AV 可按 main 冻结判读栈（`f926d10` / `9470422`）判 J2
   （12 能力 + 3 工程登记 + ExoPlaSim 只读对照）。评审不代判。
4. **本轮未做**：未改阈值/desc/ids/nlayers/argmax；未改冻结判据栈；未重跑门；未判 J2 PASS/FAIL。

---

## 二、评审独立核验（33/33 PASS）

脚本/结果：`_rerun_logs/20261022-review-gate-verify/verify_gate.py`、`verify.json`（gitignored）。
产物：`<worktree>/_rerun_logs/20261022-A2-J2-gate/`（约 113 MB，gitignored，未删改）。

### 2.1 Manifest 与双跑

| 检查 | 读数 |
|---|---|
| `manifest.json` 四份（run1/run2/native_run1/native_run2）字节 | **全同**，sha256 `81c5044f20328f355d91360a4f0e3deb23213dc5942fd4f7b474fbab9d46ffaf` |
| 五契约键位置 | **顶层**：`flux_semantics=interval_mean`、`flux_interval_s=172800.0`、`flux_accumulation=rate×dt per step; Float64 accumulator`、`flux_valid_from_snap=2`、`flux_first_interval_s=2592600.0` |
| `export` 内 `flux_*` | **无**（`[]`）；`interval_diagnostic_fields=["sw_toa_int","lw_out_int"]` |
| `config_hash` | `c30a54a6140e5c2b`（两跑同值） |
| 快照数 / 日期 | **46**；`snap_days = 30.0…120.0` 步长 2（逐项命中） |
| canonical sha 46 条 | **run1 == run2** |
| npz 文件 sha（46×2） | run1 == run2；且**逐文件实算命中记录**（mismatch=0） |
| run1/native_run1、run2/native_run2 npz | **逐文件字节相同**（0 mismatch） |

### 2.2 输入与代码锚

| 项 | 读数 |
|---|---|
| IC 六 bin 拼接 sha | **`ce6ea8c035c04cb347b15e496d8f9a30c19b34e402fd628ccfc6e0f7587dea76`**（= IC meta 规范场 sha = rev2 §七） |
| `run_j2.py` LF sha | `c74e77e9…00d8b`（命中 rev2） |
| `j2_core.jl` LF sha | `8b50bc75…dc395`（命中 rev2） |
| rev2 预注册件 LF sha | `3781327d…434755` |
| `make_ic.py` LF sha | `ad32105538b502c1cd8c53db26d1ef40b3feed06429ecd74ea7de315689cc41c` |

### 2.3 审计（评审独立复跑 main 冻结审计器）

| 扫描根 | 要求 | 复跑读数 |
|---|---|---|
| `atmo_julia` | HIGH=0 | **total=0 HIGH=0 MEDIUM=0** |
| `atmo_core atmo` | 0/0 | **total=2 HIGH=0 MEDIUM=2**（恰为 `make_ic.py:25/26`） |
| `benchmarks/atmo atmo_julia` | HIGH=0 | **total=1 HIGH=0 MEDIUM=1**（既有 `jia_remap.py:100 PROTOCOL_TOL`，J1′ 已登记） |

- 复跑命令：`python -m benchmarks.atmo.audit <worktree>/atmo_core <worktree>/atmo` ⇒
  与 A2 留档 `audit_atmo_core_atmo.json` 的 2 条 MEDIUM 逐项一致（路径/行号/规则/值）。
- 审计产物原样保留，未删改。

---

## 三、停手项裁定：接受 2 个 MEDIUM（选项 a）

### 3.1 事实

- 两常量位于 **IC 生成器** `atmo/physics/j2/make_ic.py`，仅用于 `build_ic()`
  （`:49 sst = SST_EQ − SST_DELTA·sin²φ`）与 IC 元数据字符串（`:82`）；
  **不进入模型步进、不参与调参、不是判据阈值**。
- 数值与 **跑前材料 §3.1**（`内部汇报/预注册-20261022-A2-J2湿基准跑前材料.md`，2026-10-22 跑前冻结）
  逐字一致：`sst = 300 − 30·sin²φ` K（赤道 300 K，极点 270 K）。
- 两常量自 `02d0d7b`（2026-10-21）起存在；此后 `make_ic.py` 唯一改动 `8a05560` 只改
  IC 名与初风（`u=8cosφ`），**未动两常量**（diff 已复核）。
- `main` 无 `atmo/physics/j2/` 目录 ⇒ J1′ 在 main 的 `atmo_core atmo 0/0/0` 基线**未包含此文件**；
  该 MEDIUM 对门跑扫描根是“新读数”，属于基线口径差，不是本轮改码引入。
- IC 规范场 sha 与冻结值一致 ⇒ 这两常量产出的就是被预注册的那个 IC。

### 3.2 裁定

**接受为已登记的 IC 协议常量（A2 选项 a）**，不返工、不重跑、不调整任何数值。

理由：审计规则 `constant_climate_field` 的目的是捕获**可能被用来调结果的硬编码气候场**；
本项是**先于数据冻结**的 IC 协议定义、且不参与模型物理，放开它不改变任何判据读数或阈值。
改代码把常量挪走/改名只会制造“规避审计”的坏先例，且无任何物理收益。

### 3.3 约束（写入最终签门件）

1. `SST_EQ`/`SST_DELTA` 与 `make_ic.py` 的 IC 语义在本门**冻结**；任何改动须**新预注册 + 重新审计**，
   且不得用来补救本门结果。
2. **禁止**改名/重构常量以规避 `constant_climate_field`；审计读数必须原样保留在案。
3. 本项以“已接受 MEDIUM（预注册 IC 协议常量）”登记；后续预注册应预先声明预期 MEDIUM，
   或把协议常量放入显式 protocol 元数据路径（工程改进项，不在本门阻断）。
4. 其余 MEDIUM（`jia_remap PROTOCOL_TOL`）维持既有“已知误报”登记，不入本门停手。

---

## 四、登记披露（不改数据，写入最终签门件）

1. **runner 内嵌 protocol 字符串为 rev1.1**：`j2_report.json.protocol = "J2_prereg_20261021.md (FROZEN rev1.1)"`。
   本次执行按 rev2（命令/配置/manifest 合同/哈希均命中 rev2 §七）；因 runner 哈希被冻结未改字符串。
   协议真源以 rev2 文件 + LF sha `3781327d…` 为准。
2. **首快照（day 30）为 Track A 占位区间**：4,321 步 = 2,592,600 s（时钟 `+1`）；
   AV reader 固定从快照 2 起用 pr/evap；`CONS-WATER` 的 W/days 全窗（rev2 §五.2）。
3. **IC 双 sha**：npz 原始文件字节 sha `b2633308…` ≠ 规范场 sha `ce6ea8c0…`；
   本门以规范场 sha 为准（zip 容器字节 vs 场内容）。
4. **代码面**：门跑在 A2 分支 `feat/atmo-physics` @ `428f02a`（merge main `9470422`）；
   判读冻结栈在 main `f926d10`/`9470422`；runner 功能文件哈希与 `2a49ce6` 一致。
5. **测试**：A2 分支档② **1586 passed / 0 failed / 10 skipped / 48 deselected，118.57 s**（失败集合 ∅）。

---

## 五、下一步

| # | 任务 | 负责 | 状态 |
|---|---|---|---|
| 1 | J2 判读（12 能力 + 3 工程登记 + ExoPlaSim 只读） | AV | **可开始**（owner 派工） |
| 2 | 核签 J2 签门件 | 评审 | 等 AV 报告 |
| 3 | J3 世界耦合 | A3 | 等 J2 PASS + owner 派工 |

> 本件只受理**门跑输入**；J2 的 PASS/FAIL 由 AV 按 rev2 §三/§四判读，评审在签门时复核。
