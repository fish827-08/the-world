# P1′ 判读预注册（AV 裁判侧）

**状态**：**PRE-REGISTERED**（2026-10-21；**先于任何 J1′ 数据**）
**性质**：AV 的**操作协议**，不是门定义。判定口径以 owner 的
`benchmarks/atmo/gate/P1prime_gate_definition_20261021.md`（rev1，ACTIVE）为准；
本件不改阈值/id/desc，不新增代码分支，不改变门角色。若本件与定义件冲突，**以定义件为准**并报 owner。
**适用**：甲（SpeedyWeather.jl）Held–Suarez 干基准 J1′ 确认跑的独立判读；乙冻结期间不适用。
**依据**：台账 §26（C 路线）、P1′ 定义件 §二/§四；J1 签门件（仅作历史对照）。

---

## 一、时间戳纪律与前置事实

- **先提交判读协议，后接收数据**：本件提交时，A1 工作树（`feat/atmo-planb2` @ `18a13c4`）
  的 `_rerun_logs/` 只有 `20261021-A1-J1`、`20261021-planb`、`20261022-planb2` 等，
  **无任何 J1′ 原生目录**；A1 的 J1′ 尚未派工/开跑。本件的 git 提交时间戳为证。
- 冻结门栈（LF 归一 sha256，复算于本件提交前；main `7f1c0c6`）：

| 文件 | sha256 |
|---|---|
| `criteria.py` | `f80dd8029843c73fc09b5dda65020e849c1e8527a89f44cfb5413ddb4a81ac52` |
| `loader.py` | `18c2f094fa85c7a05b8de6dad8b07ca658421b72ff9e878727234292947df7af` |
| `metrics.py` | `aaa0032b88d436fab6e0ba5d8f80ae59375fa9234946baf27c963b79524ca0c2` |
| `conservation.py` | `8ffbb0a522f52c15516150adb8811ba5cc9bec96ae8deae2d4b7c75f371d5eb8` |
| `run_hs.py` | `e1c069071a9d75deecdaf066f002157893e9a35d27d93820e1b6d4e3c6607fb1` |
| `suites.py` | `c204ca387f9030f2026a3ebe17e04789ca0893b4da52881f6fa36cfcee737c83` |

- 冻结映射件：`benchmarks/atmo/jia_remap.py` `bae272a86b24cec59deac8c97dd57ee89d0c9c2c2b7126c4c770c275129d2bf0`。
  **J1′ 不得修改映射件**；参数（`nlat`/`nlon`）只能由数据规则决定（见 §三）。
- IC 协议 `iso_w1`，期望 sha256（AV 本会话由 J1 网格几何独立材料化复算）：
  `0dfe78ae70b2c0ae926952ff0a74b6d91fa30e10d79222e0f10276afadd55d1a`。
- **J1 数据禁作 J1′ 读数**。J1 的 15 条读数只允许在工程债归因/趋势中作**带明确标签的历史对照**
  （"J1 历史，非 J1′"），绝不充当 J1′ 实测值、不参与 J1′ 判定。

---

## 二、被判输入（A1 原生交付的验收契约）

A1 只交**两个原生 run 目录**（`run1`、`run2`）+ manifest + 成本记录；结构须完全符合
`jia_remap.py` 顶部"原生 run 目录契约"。**校验失败 ⇒ `INPUT_CONTRACT_ERROR`，AV 不迁就、不签门、报 owner。**

### 2.1 协议字段（不符即停，不自行修数据）

| 项 | 要求 |
|---|---|
| `benchmark` | `held_suarez` |
| `days` | **120**（`dt_atmo_s=144`） |
| `snap_days` | **days 30–120、间隔 2 天，共 46 个**（30,32,…,120） |
| `seed` | **42**（两跑同 seed 同配置） |
| `grid` | `npoints=3168`、`nlayers=8`；lat/lon 还原出 **48 环**、每环经度均匀、无极点；`truncation` 声明 T32；`sigma` 8 层、含 200 hPa 层中 |
| `export` | `dtype=float32`、`ps_units=log(Pa)`、`field_shape=npoints,nlayers`、`byte_order=little` |
| `ic` | `name=iso_w1`、`sha256=0dfe78ae…55d1a`（与 §一 一致；AV 用 `materialize_ic` 从几何重算复核） |
| `forcing` | HS94 十二键（SI）+ `tmax_eq_K=315`、`tmin_eq_K=200`、`ps0_Pa=100000`（±0.5%）；HS 套件须声明已用 `LinearDrag(kf>0)` |
| `env` | `julia`/`speedyweather`/`nthreads` 记录（不判，只登） |
| `config_hash` | 记录（不判，只登） |

任何字段缺失/超差/两跑配置不一致 ⇒ **停手报 owner**，不得用另一跑顶替、不得改 manifest。

### 2.2 数据完整性

- 两跑快照数均为 46；`jia_remap.native_sha256` 逐快照记录**映射前**原生字节摘要（用于 CONS-DET 对账）。
- 成本：manifest/日志中的 **ms/step 与总墙钟**必须记录（R-5）。缺 ⇒ 签门件登记"未提供"，不阻断能力门。
- 只读 A1 交付目录；AV 不在 A1 目录内写文件（IC/映射产物落 AV 自己的 `_rerun_logs/`）。

---

## 三、判读流程（AV 独立执行，固定顺序）

1. **IC 复核**
   `python -m benchmarks.atmo.jia_remap --native run1 --ic iso_w1 --ic-out <av_ic.npz>`
   ⇒ sha256 必须 = `0dfe78ae…55d1a` = `manifest.ic.sha256`；否则停。
2. **重映射（两跑分别做）**
   `python -m benchmarks.atmo.jia_remap --native run1 --out av_run1 --rep map_run1.json`
   `python -m benchmarks.atmo.jia_remap --native run2 --out av_run2 --rep map_run2.json`
   - 目标网格由数据规则决定：`nlat = 原生环数（48）`、`nlon = 2 × max_ring`（不手选、不降采样）；
   - 必须 `mu_coverage_defect ≤ 1e-10`、`self_check_bad = 0`（映射越界/面积均值漂移/非有限 任一非零 ⇒ 停）；
   - 登记目标覆盖域（`mu_cover_deg`、`excluded_cap_area_frac`）与原生/映射 CFL（R-2 注记，不改判据）。
3. **判读**
   `python -m benchmarks.atmo.run --suite held_suarez --run-dir av_run1 --run-dir-b av_run2 --out reads.json`
   - 退出码只记录；**AV 逐条重读 15 个 row**（id/status/measured/limits/unit/note），不以退出码代替判定；
   - 判定实现 = 冻结 `suites.py`，**不另写判据**、不加跳过分支。
4. **映射守恒交叉核对**
   - 目标网格 `CONS-MASS` 读数 vs 原生 `jia_remap.native_area_mean(ps)` 相对首快照漂移，须**同量级**；
   - 目标为等 Δμ 行 ⇒ v1 中点带边在数学上即精确 Δμ（R-8 已证默认路径逐位不变），
     故此处 `weights=v1(midpoint)` 是**精确口径**，不是回退缺陷；签门件须写明。
5. **写死审计**
   - `python -m benchmarks.atmo.audit benchmarks/atmo atmo_julia` ⇒ **HIGH=0**；
   - `python -m benchmarks.atmo.audit atmo_core atmo` ⇒ **HIGH=0 / MEDIUM=0**；
   - 已知 MEDIUM=1：`jia_remap.py:72` 的 `PROTOCOL_TOL` 被 `constant_climate_field` 规则误报
     （常量协议值，非气候场）。**登记不改**；若出现新的 HIGH 或新的 MEDIUM ⇒ 停手报 owner。
6. **证据落盘**：`_rerun_logs/<YYYYMMDD>-AV-p1prime/`（gitignored）：`av_run1/`、`av_run2/`、
   `map_run1.json`、`map_run2.json`、`reads.json`、审计日志、IC npz。
   本机 Windows 用 `../../.venv/Scripts/python.exe`，前置 `PYTHONIOENCODING=utf-8`。

---

## 四、15 条判读（角色与阈值，一律按 v1 原判读）

读数窗口：days 30–120（46 快照），与 J1 一致。阈值/id/desc 一字不改。

| # | 判据 | 阈值（v1 原文） | P1′ 角色 |
|---|---|---|---|
| 1 | CONS-NAN | 0 个非有限 | **能力（强制）** |
| 2 | CONS-MASS | ≤1e-6 | **E-MASS 工程登记**（不计门） |
| 3 | CONS-CFL | ≤1 | **能力（强制）** |
| 4 | CONS-POLAR-AMP | ≤3 | **能力（强制）** |
| 5 | CONS-POLAR-TREND | ≤0.01 m/s/day | **E-POLAR 工程登记**（不计门） |
| 6 | HS-FORCING | 0 missing / 0 mismatch | **能力（强制）** |
| 7 | HS-JET-U | 15–40 m/s | **能力（强制）** |
| 8 | HS-JET-LAT | 20–60° | **能力（强制）** |
| 9 | HS-SFC-U | −8–2 m/s | **能力（强制）** |
| 10 | HS-T-TROP-SFC | 290–320 K | **能力（强制）** |
| 11 | HS-T-POLE-SFC | 225–265 K | **能力（强制）** |
| 12 | HS-T-TROP-UP | 185–230 K | **能力（强制）** |
| 13 | HS-EKE-ML | ≥1.0 | **能力（强制）** |
| 14 | HS-EKE-RATIO | ≥1.5 | **能力（强制）** |
| 15 | CONS-DET | 0 失配（46 快照 + meta 逐字节） | **能力（强制）** |

- 13 条能力：第 1、3、4、6–15 条；工程登记：第 2、5 条。
- **R-7(c) v2 不用于 J1′**（仍 PROPOSED、未冻结）：极区语义 = v1 现状，
  目标网格最外两行 ±77.95°/±69.45°，签门件登记。
- **CONS-DET 必须双跑**；单跑 / 缺 `run2` ⇒ `NOT_RUN` ⇒ **不签 PASS**。

---

## 五、P1′ 判定（只在签门件层执行）

### PASS（同时满足）

1. **13 条能力全部 PASS**（按上表 v1 阈值与冻结 `suites` 判，不另算）；
2. 15 条读数**全部产出**（任一 `NOT_RUN`/缺读数 ⇒ 不签 PASS）；
3. E-MASS / E-POLAR 均有**数值读数 + 书面归因**（按 §六），并按定义件 §三格式登录；
4. 写死审计 **0 HIGH**；
5. CONS-DET 为**双跑逐位**。

### FAIL

任一能力判据 FAIL，或任一读数缺失、审计有 HIGH、CONS-DET 未双跑 ⇒ **FAIL**。
**不写"接近通过"、不写"口径修正后通过"。**

### 工程登记条目

- 超过 v1 阈值**不构成门 FAIL**，但**必须**在签门件"工程债"节登记：
  读数、v1 阈值、v1 判读、归因、趋势、工程计划、责任人。
- **不得**写 N/A、删除、或并入其他条目。
- 若读数呈**发散趋势**（非平台/非结构性常数）⇒ **停签、标红报 owner**；
  不自动升级为门 FAIL，也不静默放行。
- 工程条目重新升为强制须 owner 新裁定 + 定义件 revision bump；AV 不自行改门。

### 停止规则（触发任一即停，不签、不续跑）

- 输入契约不符、`self_check_bad>0`、`mu_coverage_defect>1e-10`；
- 任一能力 FAIL（报 owner；**不自动回乙、不自动开新候选**）；
- 任一读数缺失 / CONS-DET 非双跑；
- 审计出现新 HIGH（或除已知误报外的新 MEDIUM）；
- 工程读数发散（停签报 owner，等裁定）。

---

## 六、工程债归因方法（预定义，数据到达后照此执行）

### E-MASS（CONS-MASS）

| 字段 | 预定义内容 |
|---|---|
| 读数 | 目标网格 `mass_drift`（46 快照，相对首快照=day30）；统计窗口 days 30–120；覆盖域 = 目标 μ 覆盖（极帽排除面积比照登） |
| 对照 | 原生 `native_area_mean(ps)` 同窗口相对漂移；两者须同量级 |
| 归因三元 | ① **映射效应**：目标 vs 原生漂移差（映射为精确面积守恒，应≈0）；② **表示效应**：`⟨ps⟩` 与 `exp(⟨ln ps⟩ + ½Var(ln ps))` 的闭合误差（原生与目标各算一次；J1 历史闭合误差 0.6%）；③ **精度效应**：A1 若交 f64 对照臂则并列，未交则登记"未提供"，**不得自行推断** |
| 趋势 | 首值/末值/峰谷、是否单调发散；与 J1 历史 `4.12793e-05`（f64 `4.43961e-05`）并列（标签"J1 历史，非 J1′"） |
| 工程计划/责任人 | **owner 指定**；AV 只登记，不代填计划 |

### E-POLAR（CONS-POLAR-TREND）

| 字段 | 预定义内容 |
|---|---|
| 读数 | 目标网格 v1 `polar_stability` 趋势（最外两行 ±77.95°/±69.45°；**不用 R-7(c) v2**），days 30–120 |
| 对照 | 原生最外 1–2 环的纬向平均 u 同段斜率（与 J1 同法），目标与原生**同号同量级** ⇒ 非映射/极帽伪影；若目标有而原生无 ⇒ 映射或选行效应，停签报 owner |
| 归因 | 结构/选行/映射/精度之一，附上述对照；不得只给结论 |
| 趋势 | 首末值/峰谷、是否单调发散；与 J1 历史 `+0.334148`（原生外环 −0.285/−0.239）并列（标签"J1 历史，非 J1′"） |
| 工程计划/责任人 | **owner 指定**；AV 只登记 |

---

## 七、禁止项（硬约束）

1. **不得用 J1 的场/读数**充当 J1′ 读数；J1 数值只可作带标签的历史对照。
2. **不得引用 A1 的自判结果**作为 AV 读数（只可事后对账；签门件以 AV 独立重算为准）。
3. **不得改** `criteria.py`/`loader.py`/`metrics.py`/`conservation.py`/`run_hs.py`/`suites.py`
   的阈值/id/desc/实现；不得在 `suites` 加"跳过工程条目"的分支。
4. **不得启用 R-7(c) v2**；不得自选极区选行口径。
5. **不得改冻结映射件** `jia_remap.py`；`nlat`/`nlon` 只按数据规则。
6. **不得调参数救结果**（映射方法/目标分辨率/统计窗口/容差一律按本件）。
7. **不得**在能力 FAIL 时自动回乙、自动开第四候选或自动降门。
8. **不得**写"接近通过"；工程条目不得 N/A/删除/合并。
9. 输入不符**不迁就**（不补数据、不改 manifest、不用 A1 的中间产物顶替原生场）。

---

## 八、交付物与落点

1. **签门件** `benchmarks/atmo/gate/P1prime_signoff_<YYYYMMDD>.md`，字段按定义件 §四：
   ① 冻门栈（main hash + 六文件 sha256）；② 被判件（两个原生目录、manifest、IC sha、dt/天数/
   快照密度/seed/版本、成本实测）；③ 15 条逐条表（**角色列** + 实测 + 阈值 + v1 判读）；
   ④ P1′ 复合判定 + **13/13 计数**；⑤ 工程债（E-MASS / E-POLAR，按定义件 §三）；
   ⑥ 审计门；⑦ 复现命令；⑧ 未做/风险。
2. **汇报** `内部汇报/汇报-<YYYYMMDD>-AV-P1prime独立判.md`（实测值 + 证据路径 + 停止/裁定项）。
3. 交付后更新 `内部团队文档/身份-AV.md` 与 `内部团队文档/任务队列.md` 自己的行，随交付 commit。
4. 分支 `feat/atmo-p1prime`；`git commit --only` + 明确 pathspec；**只推送主仓**。

---

## 九、变更控制

- 本件任何改动 = 新提交 + revision 标注；**数据到达前不改判读口径**。
- 若 owner 发布 P1′ 定义件 rev2，以 rev2 为准；受影响部分重跑并重签。
- 本件不追溯 J1；J1 = FAIL 定案不变。

---

**附：本件提交前已完成的只读核对**

- 六文件 LF 归一 sha256 与台账 §25.3 完全一致（`criteria.py` 未变）。
- `jia_remap.py` sha256 `bae272a8…d2bf0`（冻结件，未改）。
- IC 独立材料化 sha256 = `0dfe78ae…55d1a`（与 J1 manifest 一致；J1′ 同网格协议应再现同值）。
- 审计：`benchmarks/atmo atmo_julia` = HIGH 0 / MEDIUM 1（已知 `PROTOCOL_TOL` 误报）；
  `atmo_core atmo` = 0 / 0。
- A1 工作树无 J1′ 产物（`_rerun_logs/` 截至 `20261022-planb2`）。
