# P1′ 独立签门件（AV 裁判侧）—— J1′ 确认跑

**签门人**：AV（验证线，GLM-5.3-Flash，独立于 A1）
**签门日期**：2026-10-21（Asia/Shanghai）
**判定依据**：`gate/P1prime_gate_definition_20261021.md`（rev1，ACTIVE）+ `gate/P1prime_prereg_20261021.md`（AV 操作协议）
**被测件**：A1 交付 `aad8dd0`（分支 `feat/atmo-j1prime`），数据 `<worktree>/_rerun_logs/20261021-A1-J1prime/`
**判定流程位置**：A1 交付 → 机械审 M1–M8/G1–G6（报告 `内部汇报/机械审-20261021-J1prime-M1-M8-G1-G6.md`）→ **本件（AV 独立复算）**

---

## 0. 结论

# **P1′ = PASS**（13/13 能力判据 PASS；15/15 读数全部产出；审计 0 HIGH；CONS-DET 双跑逐位）

两条工程登记条目（E-MASS / E-POLAR）按 v1 阈值判为 FAIL，**不构成门 FAIL**，已按定义件 §三 格式登录工程债（§5），附独立归因。两条均**非单调发散**（峰后回落/有界振荡），不触发预注册停止规则；工程处置（计划/责任人）**待 owner 指定**，AV 不代填。

---

## 1. 冻门栈

判据运行所在分支 `feat/atmo-p1prime`（本签门工作树 `<worktree>`），血系 main `7f1c0c6`（含 R-8 合入 `1eac6dd`）；签门时 main HEAD = `adbf64e`（其后的承诺均为 docs-only，门栈文件未动——由本件独立复算 + 机械审 G1 blob 级双重证实）。

LF 归一 sha256（AV 于本工作树独立复算，七件全部命中预注册 §一 冻结值）：

| 文件 | sha256 | 状态 |
|---|---|---|
| `criteria.py` | `f80dd8029843c73fc09b5dda65020e849c1e8527a89f44cfb5413ddb4a81ac52` | ✅ 一字未动 |
| `loader.py` | `18c2f094fa85c7a05b8de6dad8b07ca658421b72ff9e878727234292947df7af` | ✅ |
| `metrics.py` | `aaa0032b88d436fab6e0ba5d8f80ae59375fa9234946baf27c963b79524ca0c2` | ✅（R-8 后值） |
| `conservation.py` | `8ffbb0a522f52c15516150adb8811ba5cc9bec96ae8deae2d4b7c75f371d5eb8` | ✅（R-8 后值） |
| `run_hs.py` | `e1c069071a9d75deecdaf066f002157893e9a35d27d93820e1b6d4e3c6607fb1` | ✅（R-8 后值） |
| `suites.py` | `c204ca387f9030f2026a3ebe17e04789ca0893b4da52881f6fa36cfcee737c83` | ✅（R-8 后值） |
| `jia_remap.py`（冻结映射件） | `bae272a86b24cec59deac8c97dd57ee89d0c9c2c2b7126c4c770c275129d2bf0` | ✅ 未改 |

## 2. 被判件（输入契约核验）

| 项 | 要求（预注册 §2.1） | 实测 | 判定 |
|---|---|---|---|
| 原生目录 | run1/run2 同结构 + manifest | `native_run1/`（46 快照 + manifest）、`native_run2/`（同）；另交 `native_f64`（归因臂）、`native_teq`（对照臂） | ✅ |
| manifest sha | 记录 | run1 = run2 = `6df44aa6…997ba7c3`（AV 独立复算字节同一）；f64 `990d20cc…`、teq `96fbb284…` | ✅ |
| benchmark / days / dt | held_suarez / 120 / 144 s | 逐字命中（`J1_DONE n_steps=72001 dt_clock_s=144.0`） | ✅ |
| snap_days | 30–120 步 2 共 46 | `[30+2i]` 程序判定 True，两跑各 46 | ✅ |
| seed | 42 | 42（两跑同） | ✅ |
| grid | 3168 点 / 8 层 / 48 环 / T32 | `npoints=3168 nlayers=8 nlat=48 truncation=32`；快照 lat 实测 48 环、每环 20–112 点、`|lat|max=87.159°` | ✅ |
| export | f32-LE / ps=log(Pa) / (npoints,nlayers) | 命中；σ 八层含 0.2（200 hPa，R-1 空窗已解除） | ✅ |
| ic | iso_w1 / `0dfe78ae…55d1a` | **AV 用冻结件独立材料化复算 = `0dfe78ae70b2c0ae926952ff0a74b6d91fa30e10d79222e0f10276afadd55d1a`**，与 manifest 一致；T∈[287,289]、u=v=0、ps=ln(1e5) | ✅ |
| forcing | HS94 十二键 + tmax 315/tmin 200/ps0 1e5 | 逐键命中（HS-FORCING 判据 0 missing / 0 mismatch）；`LinearDrag(kf>0)` 已声明 | ✅ |
| env | 记录不判 | julia 1.13.2 / speedyweather 0.23.0 / nthreads=1 / python 3.14.7 | 登记 |
| 成本（R-5） | 记录不判 | 积分循环 `J1_DONE wall_s`：run1 311.31 s（4.3237 ms/步）、**run2 291.01 s（4.0417 ms/步，JIT 热稳态）**、f64 377.81 s（5.2473）、teq 90.02 s（5.0008） | 登记 |
| **`grid.nring = null`** | —（机械审 §三-6 移交 AV 裁） | **不算输入契约违例**：`GRID_KEYS` 契约 = npoints/nlayers/sigma/truncation 四项，nring 是附加可选字段；环数由快照 lat 派生是冻结件的设计行为（`ring_of`）。建议 A1 后续补值或删键，仅文档性 | ✅ 不阻断 |

**目标网格（数据规则决定，非人选）**：`nlat=48`（原生环数）、`nlon=224`（2×max_ring=112）；等 Δμ 行，μ 覆盖 ±87.159°，极帽排除面积比 0.1229%。
**映射自证**：两跑 `mu_coverage_defect = 2.6678e-15 ≤ 1e-10`、`self_check_bad = 0`、各 46 快照。

## 3. 15 条逐条判读（v1 阈值一字不改；角色列按 P1′ 定义件 §一）

窗口 days 30–120（46 快照）。判定实现 = 冻结 `suites.py`，AV 逐条重读 row，不以退出码代替判定。

| # | 判据 | 角色 | 实测 | v1 阈值 | v1 判读 |
|---|---|---|---|---|---|
| 1 | CONS-NAN | 能力 | 0 个非有限（四场全有限） | 0 | ✅ PASS |
| 2 | CONS-MASS | **E-MASS 工程登记** | **4.127933031796103e-05**（@day 100 峰；末 1.9338e-05） | ≤1e-6 | ❌ FAIL（登记，见 §5.1） |
| 3 | CONS-CFL | 能力 | 0.09387707064984682（重算；原生 0.0491） | ≤1 | ✅ PASS |
| 4 | CONS-POLAR-AMP | 能力 | 0.925770955779815 | ≤3 | ✅ PASS |
| 5 | CONS-POLAR-TREND | **E-POLAR 工程登记** | **+0.33414784889156895** m/s/day（末 20%，days 102–120） | ≤0.01 | ❌ FAIL（登记，见 §5.2） |
| 6 | HS-FORCING | 能力 | missing=[] mismatch=[] | 0/0 | ✅ PASS |
| 7 | HS-JET-U | 能力 | 26.920473517278218 m/s | 15–40 | ✅ PASS |
| 8 | HS-JET-LAT | 能力 | 43.36590823949133° | 20–60 | ✅ PASS |
| 9 | HS-SFC-U | 能力 | −3.255401314690121 m/s | −8–2 | ✅ PASS |
| 10 | HS-T-TROP-SFC | 能力 | 304.4044676539849 K | 290–320 | ✅ PASS |
| 11 | HS-T-POLE-SFC | 能力 | 263.0878460959294 K | 225–265 | ✅ PASS |
| 12 | HS-T-TROP-UP | 能力 | 218.82554479654024 K | 185–230 | ✅ PASS |
| 13 | HS-EKE-ML | 能力 | 63.05753978963438 m²/s²（热带 4.053） | ≥1.0 | ✅ PASS |
| 14 | HS-EKE-RATIO | 能力 | 15.559012310836467 | ≥1.5 | ✅ PASS |
| 15 | CONS-DET | 能力 | 0 失配（46 快照 + meta 逐字节，双跑） | 0 | ✅ PASS |

**计数：能力 13/13 PASS；读数产出 15/15（无 NOT_RUN）。**
注记：CONS-MASS 权重口径 = `weights=v1(midpoint)`——目标网格等 Δμ 行上中点带边**数学上即精确 Δμ**（AV 数值证实：权重 ptp=2.2e-16、Σw=2μ_max=1.9975420145、max|w−Δμ_exact|=5.1e-15），非回退缺陷。R-2 注记：CONS-CFL 映射后 0.049→0.094（极区 dx 细化效应），口径照登、判据未动。

## 4. P1′ 复合判定

| 条件（定义件 §二） | 满足 |
|---|---|
| 13 条能力全部 PASS | ✅ |
| 15 条读数全部产出 | ✅ |
| E-MASS / E-POLAR 有数值读数 + 书面归因（§三 格式） | ✅（§5） |
| 写死审计 0 HIGH | ✅（§6） |
| CONS-DET 双跑逐位 | ✅ |

**⇒ P1′ = PASS。**

## 5. 工程债（按定义件 §三 格式；**不计门、不 N/A、不合并**）

### 5.1 E-MASS（CONS-MASS）

| 字段 | 内容 |
|---|---|
| 读数 | 目标网格（48×224 等 Δμ，μ 覆盖 ±87.159°，极帽排除 0.1229%）`mass_drift` = **4.127933031796103e-05**，@day 100 峰；期末 1.9338e-05；M₀=100002.022 Pa |
| v1 阈值 / v1 判读 | ≤1e-6；**FAIL**（不改写） |
| 归因（三元，预注册 §六 预定义方法） | ① **映射效应 ≈ 0**：目标漂移 4.127933e-05 vs 原生 μ 面积加权均值漂移 4.127933e-05（比值 1.0000）——保守一阶面积重叠映射不掺假信号；② **表示效应（主因）**：模式预报量 ln(ps)，⟨ps⟩=⟨exp(ln ps)⟩ 随均值/方差增长——闭合 Δ⟨ln ps⟩+½Δvar = 7.2657e-06+1.2181e-05 = **1.9447e-05** vs 末端漂移 1.9338e-05（**相对差 0.56%**）；③ **精度效应排除**：A1 已交 f64 对照臂（AV 独立重映射复算）**4.4396e-05**，不优于 f32（f32/f64=0.930）⇒ 非量化。另：极帽按最外两环均值补齐代理漂移 2.09e-05（本轮方法）≫1e-6 ⇒ 覆盖域外仍超限（J1 同法代理 1.2840e-5，口径未留存，仅历史对照） |
| 趋势 | 首值 0（day30 基准）→ 峰 4.1279e-05 @day100 → 末 1.9338e-05（day120）；**峰后回落、非单调发散**；与 J1 历史同值同形（J1′≡J1 逐位，见 §7） |
| 工程计划 / 责任人 | **待 owner 指定**。AV 观察（非计划）：若下游 J2 水/能量账本需要严格闭合，甲侧需评估质量修正器或改线性 ps 预报量；否则维持登记 |

### 5.2 E-POLAR（CONS-POLAR-TREND）

| 字段 | 内容 |
|---|---|
| 读数 | 目标网格 v1 `polar_stability` 趋势（最外两行 ±77.95°/±69.45°，**不用 R-7(c) v2**）= **+0.33414784889156895 m/s/day**（末 20%，days 102–120，max\|u\| 口径） |
| v1 阈值 / v1 判读 | ≤0.01 m/s/day；**FAIL**（不改写） |
| 归因（预注册 §六 预定义对照） | **非映射/选行伪影，是 69–78° 带的真实变率**：原生最外 1/2 环同段（days 102–120）纬向平均 \|u\| 斜率 **+0.0462 / +0.1285**、\|风\| 斜率 **+0.1808 / +0.0072**（f32，AV 独立重算）——目标（+0.334）与原生**同号同量级**；目标带内纬向平均斜率仅 +0.0382 ⇒ 判据读数由涡旋极值主导。⚠️ 精度敏感性：原生最外环 f64 臂斜率为 **−0.0799/−0.0531（zm）/−0.3610/−0.3780（\|风\|）**，与 f32 反号 ⇒ 最外环 max\|u\| 趋势精度敏感、噪声大，读数解释需谨慎 |
| 趋势 | 极区 max\|u\| 序列 6.265（day30）→ 峰 29.704（day94）→ 17.214（day120）；全窗斜率 +0.162/day（受 day94 峰主导）；**有界、非单调、非 runaway 发散**（day120 量值 17 m/s 与中纬急流 26.9 m/s 同量级，无放大迹象） |
| 工程计划 / 责任人 | **待 owner 指定**。AV 观察（非计划）：该 FAIL 与 J1 完全同值（同一跑），若要处置需先裁定"慢旋冲锚 vs 振荡增长"——建议 J2/J3 长跑中继续监视极区 max\|u\| 包络；R-7(c) v2（按纬度选行）仍 PROPOSED、未冻结、**未用于本门** |

## 6. 审计门

| 命令 | 结果 |
|---|---|
| `python -m benchmarks.atmo.audit benchmarks/atmo atmo_julia` | **HIGH=0 / MEDIUM=1**——唯一 MEDIUM = 已知误报（`jia_remap.py:72` `PROTOCOL_TOL` 被 `constant_climate_field` 规则误命中，常量协议值非气候场），与预注册登记一致，**无新 HIGH / 无新 MEDIUM** |
| `python -m benchmarks.atmo.audit atmo_core atmo` | **HIGH=0 / MEDIUM=0** |
| `python -m benchmarks.atmo.audit <A1prime>/atmo_julia`（生产代码，J1′ 实际产出码） | **HIGH=0 / MEDIUM=0** |

## 7. 复现（本机 Windows，AV 工作树 `<worktree>`）

```bash
cd <worktree>
D=<repo><worktree>/_rerun_logs/20261021-A1-J1prime
PY=../../.venv/Scripts/python.exe   # 前置 PYTHONIOENCODING=utf-8

# ① IC 复核（sha256 须 = 0dfe78ae…55d1a）
$PY -m benchmarks.atmo.jia_remap --native $D/native_run1 --ic iso_w1 --ic-out _rerun_logs/20261021-AV-p1prime/av_ic.npz
# ② 重映射（两跑分别）
$PY -m benchmarks.atmo.jia_remap --native $D/native_run1 --out _rerun_logs/20261021-AV-p1prime/av_run1 --rep _rerun_logs/20261021-AV-p1prime/map_run1.json
$PY -m benchmarks.atmo.jia_remap --native $D/native_run2 --out _rerun_logs/20261021-AV-p1prime/av_run2 --rep _rerun_logs/20261021-AV-p1prime/map_run2.json
# ③ 15 条判读（双跑）
$PY -m benchmarks.atmo.run --suite held_suarez --run-dir _rerun_logs/20261021-AV-p1prime/av_run1 --run-dir-b _rerun_logs/20261021-AV-p1prime/av_run2 --out _rerun_logs/20261021-AV-p1prime/reads.json
# ④ 工程债归因交叉核对（本件 §5 全部数字）
PYTHONPATH=. $PY _rerun_logs/20261021-AV-p1prime/xcheck_p1prime.py
# ⑤ 审计（见 §6）
```

证据落盘 `_rerun_logs/20261021-AV-p1prime/`：`av_ic.npz`、`av_run1/`、`av_run2/`、`av_run_f64/`、`map_run*.json`、`reads.json`、`xcheck_p1prime.py/.json`。

**一致性旁证（AV 自核）**：J1′ 的 46 个原生快照逐文件 sha256 与 J1 交付记录（`run_run1.json:native_sha_n`）**0/46 失配** ⇒ J1′ 与 J1 为逐位同一跑（同 IC/seed/协议，确定性预期内）。故 §5 中与 J1 历史值的逐位一致是**预期行为**，非"用 J1 数据充 J1′ 读数"——本件全部读数为 AV 对 J1′ 场的独立重算，J1 数值仅作带标签历史对照。

## 8. 未做 / 风险 / 移交

1. **未重跑积分**（f32×2+f64+teq ≈ 20 min CPU）：用 A1 交付场 + AV 独立重映射/重算；双跑逐位性另经原生 sha256（机械审 G3）与映射后 CONS-DET（本件）双证。
2. **teq_noise 对照臂未判**：A1 已交 30 天 teq 数据（IC `edd8cb86…`），按派工口径为门后诊断臂，P1′ 不判、不作门判据；若早爆仅标红。**留给门后/评审安排**。
3. **机械审 M7（PARTIAL FAIL）对 A1 报告的三处更正/补披露**（Julia `StructUtilsStaticArraysCoreExt` 扩展加载 Error 未披露、§四合计行 +326,000 B、§六键名/时间戳语义）：属 A1 报告文档项，AV 侧登记事实——报错后仍 `J1_DONE`、两跑 46/46 逐位、全场有限，现证据不支持"影响 J1′ 结果"；是否影响判读的最终口径归总审/owner。
4. **J1 原生极区对照方法未留存**：J1 签门件录"原生最外 1/2 环同段斜率 −0.285/−0.239"，其脚本未入证据目录（`r6_attribution.json` 仅含质量侧），AV ~35 个方法变体扫描未复现该确切值（含 f64 臂、v 风、风速、逐层、全窗）。本件改用**文档化独立方法**（§5.2，脚本已留证），结论同为"原生同号同量级 ⇒ 真实变率"。**记录缺口登记，建议后续判据附属脚本一律入证据目录。**
5. **R-7(c) v2 未用**（仍 PROPOSED）；极区语义 = v1 现状（±77.95°/±69.45° 两行）。
6. **J1 不追溯**：J1 在 v1 门下 13 PASS/2 FAIL 仍是 FAIL；本件 P1′ PASS **不是**"J1 通过"或"口径修正后通过"。
7. 工程债两条的**工程计划与责任人待 owner 指定**；若工程条目要升为强制门，须 owner 新裁定 + 定义件 revision bump。

---

**AV 签（独立于 A1；未用 A1 自判值；判据阈值/id/desc 一字未改；未改冻结映射件；未启用 R-7(c) v2；未因结果调口径）**
2026-10-21 · 验证线 AV

---

## 附：签门后补注（2026-10-21 22:20 · 台账 §30 同步）

**时间线（诚实登记）**：本签门件完成于 21:50、推送主仓 22:05；总审对机械审报告的复核裁定（台账 §30，main `74fed5e`）落于 22:13，晚于本签门。补注如下，**不改写已签判定**。

1. **§30 裁定与本件的关系**：总审复核机械审报告后裁定——M7 五项为 **A1 纯文档更正**（披露 Julia 扩展加载 Error、§四合计 59,812,995 B、根总量测量时刻、§六 `native_sha256` 键名、`generated_at` 进程启动时刻），**不重跑、不改代码/数据/判据**；`nring=null` 是否算输入契约项**交 AV 决定**；AV 签门后 A1 分支合入由评审决定。
2. **先于裁定完成是否影响有效性——不影响**：本件的重映射/15 条判读/审计**全部直接读原生场、manifest 与快照**（`jia_remap` 契约校验 + 逐快照计算），不引用 A1 报告任何文字；§30 明确的“产物、IC、manifest、快照哈希均不因文档更正而变”正是本判定的实物基础。机械审 M7 三项更正已在本件 §8-3 登记为 A1 文档项，与 §30 处置一致。
3. **`nring=null` 裁定（§30.2 正式交 AV，本件 §2 已裁）**：**不算输入契约违例**——`GRID_KEYS` 契约为 npoints/nlayers/sigma/truncation 四项，`nring` 是附加可选字段；环数由快照 lat 派生是冻结映射件的设计行为（`ring_of` 自校验：48 环、每环经度均匀）。A1 **不擅自改 manifest**（§30.2），是否补值属文档完善，由 A1 在文档更正 commit 中自行决定。
4. **报告引用（已回填，2026-10-21 22:45）**：A1 文档更正已落 = `018b294`（`feat/atmo-j1prime`，经 merge `3318978` 入 main）：披露 Julia 扩展加载 Error（附"之后仍 J1_DONE、双跑 46/46 逐位、全快照有限"旁证）、§四合计改为 59,812,995 B、根总量注明测量时刻、§六更正为 `run_teq.json:native_sha256`、`generated_at` 注明为进程启动时刻、登记 `nring=null` 不改 manifest。**本件判定所依赖的实物（IC sha256、manifest 字节、46×2 快照哈希、原生场）零变化**（AV 复核：更正 commit 对 `native_*` 与 `_bin_*` 零 diff，仅 docs/atmo_julia 报告侧文字）⇒ 15 条读数与 PASS 判定不变。引用链：交付 `aad8dd0` → 更正 `018b294` → main 合并 `3318978`；AV 侧合并 `f9dd228`（总审复核受理，台账 §31.2 独立复算 CONS-MASS 逐位一致、E-POLAR 6e-16 内命中）。
5. **移交状态**：P1′ 签门已完成并推送主仓（`feat/atmo-p1prime` @ `10fb677`）；A1 分支合入待评审按 §30 决定；工程债两条计划/责任人待 owner 指定。
