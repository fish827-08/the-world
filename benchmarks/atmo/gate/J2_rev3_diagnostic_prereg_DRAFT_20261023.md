# J2 rev3 诊断预注册（PROPOSED，未冻结）— 2026-10-23

> **状态：PROPOSED / NOT FROZEN。本件不是门定义、不判 J2、不判 rev3。**
> 冻结权在评审，放行权在 owner。冻结前不得被引用为任何判据。
>
> 依据：台账 §45（A2 Phase 1 部分交付复核）；owner 2026-10-23 批准「先两项诊断、再冻 rev3」。
>
> **不变项（红线）**：J2 rev2 = FAIL 不追溯；冻结判据栈七件哈希、阈值/id/desc、12 能力 + 3 工程登记、
> canonical IC 六 bin 拼接 sha `ce6ea8c035c04cb347b15e496d8f9a30c19b34e402fd628ccfc6e0f7587dea76`、
> nlayers、argmax、能量语义（瞬时 canonical SW/LW）、`make_ic.py` 常量 **全部冻结**；
> 不跑 120 天 J2 门；不调参救结果；不造新 IC/种子；不写「接近通过」「单 ITCZ」。

## 一、两项诊断（`diagnostic_only=true`，先于任何 rev3 门跑）

### D1 — 门口径在线水账（day30–120）

**目的**：补上 A2 §2.1 未覆盖的窗口——rev2 判据窗是 day30–120，而现有分解只做了 restart 后 day0–30。

- **起点/协议**：canonical IC（六 bin sha 见上）；T32/L8、dt=600 s、obliquity=0、seasons off、daily cycle on、
  `flux_mode=:interval`、seed 42、**单跑**、120 天 / 17,281 步（与 rev2 门协议逐项相同）。
- **仪器**：沿用 `_rerun_logs/20261022-A2-J2rev3-rework/j2_budget_core.jl` 的
  `TapVD/TapLSC/TapConv/TapHF/TapClip`，仅增加 day30–120 窗口累计输出；**不改模型、不改导出语义**。
- **记录量**：
  1. `clip_cum(t)` 在 day30/60/90/120 的累计裁负水量；
  2. `gate_resid` 在 day30–120 的分解：`clip_growth / P_tap−diagP / quadrature_drift / true_closure / E_tap−diagE`；
  3. `true_closure = dW_true − Σtaps` 及其相对 P。
- **预注册判读规则（先于数据，不得事后改）**：
  - **R1（口径为主）**：若 day30–120 的 `clip_growth` 占该窗 `gate_resid` 的 **≥50%**，且
    `true_closure` 相对 P **<1e-5** ⇒ 判定 rev2 `CONS-WATER` 超标以**测量口径**为主。
    此时 rev3 允许把被测水量改为**裁负前 / 谱一致水量**（Q-b），阈值 **1e-3 不变**；
    预注册与 AV 签门件均须明写「快照 `q` ≠ 模式内部 `q`」，AV 重判。
  - **R2（模型为主）**：若 clip 占比 **<50%** 或 `true_closure` 相对 P **≥1e-5** ⇒ 维持
    WP2 = 模型水账；rev3 只能 Q-c（不改导出、照实判，>1e-3 即 FAIL），或由 owner 走 C/D。
- **产物**：`_rerun_logs/20261023-A2-J2rev3-diag/budget_gate_window/`（gitignored）；报告只报数。

### D2 — 全窗同源逐位（`H_DUMP_ROOTS` 加 `:tendencies`）

**目的**：检验「dump 漏 `vars.tendencies` 是 120 天偏置主因」这一假设；10 天冒烟去首区间后
`exact`≈`single`（差 1.37e-3 W/m²），**不构成证据**（台账 §45.3）。

- **修复范围**：`j2_homology_core.jl` 的 `H_DUMP_ROOTS` 加 `:tendencies`；仅 harness 侧，
  不动外部包、不动判据栈、不动门协议。
- **D2a（20 天冒烟，必做）**：自 `j2aqua_v2`（sha `5eaca4a3f5abf7c0d0833b4bbbc4fd37ac2932950249eb717d1fb399e2970a6`）
  积分至 day20 落全状态检查点，续跑 20 天 vs 连续参考。
  **判据 = 逐位一致**：净 TOA 差序列恒 0，且全部导出场 bitwise 相同（不含首间隔 flux_off 占位）。
- **D2b（仅当 D2a 逐位才跑）**：自 `j2aqua_v2` 确定性重跑至 **day1825**，先核六场 canonical sha
  = `ce6ea8c0…7dea76`（证明重跑与构造 IC 同源）；再在 day1790 落检查点、续跑至 day1825 与连续参考对比。
  **判据 = 全窗逐位**。
- **预注册判读规则**：
  - **R3（exact 成立）**：D2a + D2b 全窗逐位 ⇒ rev3 允许 **R3-exact**（二进制全状态检查点装载）。
    rev3 预注册必须披露协议变更：**门 = 自 day1825 连续续跑 120 天，`AQ-TOA` 语义是「自旋末连续段」，
    不是「重启后恢复能力」**；canonical IC 六 bin 不变，新增伴生检查点须带长度/sha/网格 fail-loud；AV 重判。
  - **R4（exact 否决）**：D2a 不逐位 ⇒ exact 路线否决；rev3 只能用 **R2 预热**，且 **N 预注册为 60 天单一候选**
    （禁事后挑 N），验证窗 day60–120。R2 若仍不达标 ⇒ rev3 无可行同源修法，**停手报 owner（C/D）**。
- **产物**：`_rerun_logs/20261023-A2-J2rev3-diag/homology/`（gitignored）。

## 二、明确不做

- 不跑 120 天 J2 门；不判 J2 / rev3；不改判据/阈值/窗口/角色/能量语义。
- 不动 `benchmarks/atmo/` 七件冻结件与 `atmo_core/`；不动 SpeedyWeather.jl 外部包。
- 不改 `make_ic.py`；不造新 IC/种子；不写任何「通过/接近通过」。

## 三、冻结条件

1. A2 交 D1/D2 读数（仅数据 + 事实，不做路线裁定）。
2. 评审按本件 R1–R4 核验；触发 R1/R3 则起草 **rev3 预注册**（含协议变更披露与 AV 重判口径）→ 冻结。
3. 触发 R2/R4 则按对应分支：Q-c / R2-N=60 / 停手报 owner。
4. 冻结前，本件与 D1/D2 读数均不得作为判据引用。

## 四、成本（按 A2 实测外推，非承诺）

| 项 | 估时 | 依据 |
|---|---|---|
| D1（120 天单跑 + tap 分析） | 5–10 min | 30 天带 tap 实测 wall 59.8 s |
| D2a（20 天冒烟） | 1–2 min | 20 天 19.0 s 外推 |
| D2b（1860 天重跑 + 35 天续跑） | 30–35 min | 同上 |
| 合计 | **约 40–50 min** | Phase 1 盒内（≤1 天） |

## 五、签署

- 起草：评审（2026-10-23），依 owner「可以继续执行」。
- 冻结：待 D1/D2 读数后由评审冻；本件当前 **PROPOSED**。
