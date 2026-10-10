# 事实基线（BASELINE）

> 用途：**唯一事实来源**。所有后续评估以此为参照，不再依赖陈旧的 PROGRESS.md / AGENTS.md。
> 纪律：本文件只收录 `[实测]` 与 `[代码]` 两类。`[文档]` 内容不得进入本文件。
> 每次评估后更新，并记录更新时的 commit。

---

## B-0. 基线标识

| 项 | 值 |
|----|-----|
| 基线建立日期 | 2026-09-11（R1 评审后更新） |
| **当前主干目录** | `_gitee_review/` |
| 主干分支 / commit | `main` = **`55245c1`**（2026-09-11，原 f6fba18 已推进 4 提交） |
| 远端同步 | `main` 领先 `origin/main` **12 提交**；`gitee/main` 已同步至同一提交 |
| ⚠️ 未合入分支 | `feat/s2-rayon-rng`(+19)、`feat/s1-step-tick`(+12)、`feat/d2-vec`(+5) |
| Python | 3.14.7（项目 `.venv`） |
| Rust 扩展 | `.venv/Lib/site-packages/sim_core/*.cp314-win_amd64.pyd`，导出 **18 个函数** |
| ⚠️ 扩展缺口 | **仍无 `step_tick`** `[实测] 55245c1 复核 present=False` → F-V1 未闭 |
| 备用扩展 | `sim_core_dev/sim_core.cp314-win_amd64.pyd`（2026-09-11 00:20）**有 `step_tick`** |
| 工作区 | clean（`git status` = 0 项；npz 已归集 `data/snapshots/`） |
| 测试环境 | i7-12700H（20 逻辑核）, Windows |

### ⚠️ ref 陷阱（新增）

战略转向相关文档**只在 `gitee/*` ref 上**。本地 `main` 曾落后于 `gitee/main`。
**评估前必须 `git log --oneline -1 gitee/main` 并核对**，否则会评估过期状态。

### ⚠️ 远端同步纪律（2026-09-12 新增，硬性）

`[实测]` 本环境沙箱**阻断外部 HTTPS**（Windows 凭据存储不可访问 → schannel TLS 失败）。
`git fetch` / `git push` / `Invoke-WebRequest` / `curl` **全部失败**（TCP 可连，TLS 握手失败）。

**因此**：
1. 评估线**无法自主拉取** gitee —— 依赖所有者/开发线每次会话前 `git fetch gitee && git pull`
2. 每份评审单**必须写明所依据 commit**（已执行）
3. 若本地落后：**先报告落后再评估**，不得默默用旧状态
4. 若无法拉取：结论中**显式声明**"基于本地快照 `<sha>`，未含远端后续提交"
5. 无 push 权限时：写入本地 `_share/` 并标注"**待推送**"，由有权限成员代推

> 已发生过的真实事故：本地 `main` 曾落后 `gitee/main` 5+ 提交，导致评估陈旧状态。

### 数据资产分类（2026-09-12）

主仓 `main` **有意**同时承载引擎代码与云端实验数据 —— **这是云端实验数据，本地需要**，非误入库。

| 类别 | 路径 | 规模 | 评估线是否依赖 |
|------|------|------|--------------|
| 快照 | `data/snapshots/*.npz` | 22 个 ~83.6MB | ✅ MI 复算依赖 `snapshot_small.npz` |
| 实验产出 | `experiments/long_*.csv` | 10–70KB/个 | ✅ 原始证据 |
| 语言指标 | `experiments/*_lang_metrics*.json` | 数 KB | ✅（须补 seed/指纹/commit） |
| 系综汇总 | `results/ensemble/*/summary.*` | 数 KB | ✅ 跨 seed mean±std |

**`.gitignore` 与追踪共存**：`*.csv`/`*.json` 通配规则对**未追踪**文件生效；已追踪数据不受影响。
→ **以 `git ls-files` 实际结果为准**，不得因 gitignore 规则推断数据不存在。

### ⚠️ 目录真身对照表（防止再次评错对象）

| 目录 | 真身 | 是否可评估 |
|------|------|-----------|
| **`_gitee_review/`** | **二代最新主干** f6fba18 | ✅ **唯一评估对象** |
| `digital-life-sphere/` | 二代**陈旧** detached `b896cc1`（2026-09-08） | ❌ 已过时 3 天、落后 5+ 提交 |
| `digital_life/` | **一代**（平面 128×128，Stage 2 毕业），2 个 commit | ⚠️ 仅作教学/回归对照 |
| `_review/` | 二代早期快照（feature/l1-patchy，1 个 commit） | ❌ 历史副本 |

> 陷阱记录：`digital-life-sphere/` 用系统 Python 跑测试会得 **55 failed / 90 passed**，纯因其 `.venv` 内是 2026-09-07 的旧二进制（缺 `signal_emit`/`regrow_patchy`/`reproduce_batch` 等 7 个函数）。
> **这不是项目坏了。** 评估前必须先确认扩展与源码同版本。

---

## B-1. 测试状态 `[实测]`

```powershell
cd _gitee_review
& .\.venv\Scripts\python.exe -m pytest tests/ -q
```

| 结果 | 数量 | 判定 |
|------|------|------|
| passed | **216** | ✅ 无失败 |
| skipped | 1 | 见下 |
| errors | 3 | **全部为环境问题** |

**3 个 error 均为** `PermissionError: [WinError 5] ...pytest-of-Hide`（pytest `tmp_path` 在沙箱下无法建目录）：
`test_d2_info_structure.py::TestD2Snapshot::test_save_load_d2`、
`test_lifespan_mult.py::TestBackwardCompatibility::test_old_snapshot_no_mult_fallback_1`、
`test_observatory.py::test_persistence_roundtrip`。

**审计独立复跑** `-rs` 得 **214 passed / 2 skipped / 3 errors**（同一批 error）。差异来自两个 skip：
- `test_s1_step_tick.py:20` — `sim_core 无 step_tick（需重编译扩展）` ← **⚠️ 验证盲区，见 B-4**
- `test_l4_l5.py:205` — 300 tick 内无出生（依赖种群动态的条件 skip）

> 两次运行**均无失败**。计数差异是环境/状态差异，非项目不稳定。
> ⚠️ 任何文档中的"152/164/192/221 passed"均为**过期数字**。

---

## B-2. 性能 `[实测]`

```powershell
& .\.venv\Scripts\python.exe -m experiments.benchmark_sim_core -n 2000 -t 200 -s 42
& .\.venv\Scripts\python.exe -m experiments.benchmark_sim_core -n 4000 -t 150 -s 7
```

| N | Python | Rust | 提速 |
|---|--------|------|------|
| 2000 | 21.576 ms/tick（46.4 t/s） | 1.901 ms/tick（526 t/s） | **11.35×** |
| 4000 | 44.332 ms/tick（22.6 t/s） | 3.547 ms/tick（282 t/s） | **12.50×** |

**两次均 `REGRESSION CHECK: PASS`** —— population / total_energy / total_resource / extinct **逐位相同**。

**rayon 确定性** `[实测·审计]`：`RAYON_NUM_THREADS ∈ {1,2,4,8,16}`，N=3000（确保并行分支被走到），9 个数组 sha256 **全部相同**（`a4dc4992…`）。
> 结构性原因：全项目**无并行浮点归约**；所有 rayon 站点为逐索引互斥写。

**口径警告**：上述为**均匀资源、无 patchy**路径；tick/s 随 N 增长而下降（成本 ∝ N）。与文档"N=5000 / 116.7 t/s / 20.5×"**不是同一口径**，倍数不可直接比较。

---

## B-3. 基因接线 `[实测]`

```
total gene slots = 24
wired          = 18: MOVE_PROB, METABOLIC, REPRO_THRESHOLD, LIFE_GENE, EAT_AMOUNT,
                     STOMACH_CAP, MOVE_COST, PARENTAL_INVEST, PHOTOSYNTHESIS,
                     HOMEOTHERM, FORAGE_NEIGHBOR, TEMP_PREF, REPRO_COOLDOWN,
                     SOCIABILITY, PERCEPTION, SIGNAL_STRENGTH, AGGRESSION, ROOTING
reserved        =  6: DIET(g17), DEFENSE(g18), HEDONISM(g20),
                     PROCESSING(g21), TRUST_GENE(g22), RESERVED(g23)
```

`GENE_COUNT = 24 = len(Gene)` ✅

**⚠️ 运行时漂移检查只覆盖 3/24**：
```
sim_core.native_gene_indicators() → [('G_SOCIABILITY',13), ('G_PERCEPTION',14), ('G_AGGRESSION',16)]
```
引擎初始化（`sphere_engine.py:144-157`）只用这个 3 项列表。全量 `validate_gene_wiring()`（24 项）**仅被测试调用**。
`genes.py:217` 的 docstring 声称"引擎初始化时调用"—— **为假**。（详见 FINDINGS F-B1）

---

## B-4. 已知验证盲区 `[实测·审计]`

| 盲区 | 事实 |
|------|------|
| S1 融合路径 | 1062 行的 `tick.rs` 在默认套件中**模块级 skip**。换 `PYTHONPATH=sim_core_dev` 后 `test_s1_step_tick.py` → **10 passed** |
| 对拍窗口 | `test_s1_parity.py` 仅 1200 tick（D2 off）/ 200 tick（D2 on），相对 30 万 tick 结论**太短**，慢发散不会暴露 |
| 对拍自指 | `test_s1_step_tick.py` 的参考侧是同一个 `SphereEngine`；Python 自管阶段**两侧执行同一份代码**，无法发现该阶段错误（如码本突变在 `:686-691` 与 `:1416-1422` 逐字重复） |

---

## B-5. 零模型（D1）四臂终局 `[实测·审计复算]`

> 60×120 patchy，300k tick，seed 42/43/44，density cap 45%（`max_count=3240`）
> 数据：`experiments/long_zero_*.csv`、`results/ensemble/zero_*/`

| 臂 | n | g14 | g15 | trust | cult_div | max_gen | sig_density |
|----|---|-----|-----|-------|----------|---------|-------------|
| main | 3 | **0.470 ± 0.041** | 0.404 ± 0.130 | 0.843–0.853 | 0.0027–0.0047 | 17/17/18 | 2586–4297 |
| random | 3 | **0.475 ± 0.075** | 0.583 ± 0.137 | 0.784–0.957 | 0.0005–0.0050 | 18/17/17 | 3815–3959 |
| sigdis | 3 | **0.483 ± 0.260** | 0.417 ± 0.260 | **0.5000 精确** | 0.0005–0.0110 | 18/18/19 | **0 精确** |
| neutral | 3 | 0.5000（冻结） | 0.5 | 1.0000 | 0.295–0.510 | **175/177/177** | 4–10 |

**四条硬结论**
1. 四臂 g14 全部落在 **0.47–0.50**，与初值 0.5 无区别 → **旗舰"g14 强正选"不复现**
2. g15 方向跨臂跨 seed 不一致（sigdis 单臂 0.750/0.469/0.231，3.2× 跨度）→ 近中性漂变
3. `sigdis` 臂 trust **恒 0.5000** → trust 只在有信号时更新，是**食物富余度代理**
4. `cult_div` 在**不发信号**的 sigdis 臂同样趋同 → 趋同是**图热扩散机械必然**

**⚠️ `neutral` 臂失效**：N 崩到 2/5/2。全基因冻结不是"无选择基线"而是"无适应能力死局"。
修正设计（只冻结目标位 g14/g15，保留 g0~g13 演化）已写入 `PROGRESS.md` 待办 #16，**从未执行**。

**⚠️ 反直觉事实**：冻结臂达 **175–177 代**，演化臂仅 **17–19 代** → **`max_gen` 与适应度反相关，不能作演化指标。**

---

## B-6. 旗舰 30 万 tick 长跑 `[实测·审计复算]`

数据：`experiments/long_run_l4l5_result.csv`（150 行，每 2000 tick）

| 项 | 值 |
|----|-----|
| seed | **仅 42**（`long_run_l4l5.py:29` 硬编码） |
| `max_count` | **5000**（69% 密度）← **与 D1 的 3240/45% 不同，是未受控混淆变量** |
| 终局 g14/g15/g16/g19 | 0.8345 / 0.3510 / 0.4657 / 0.5900 |
| 终局 trust / cult_div | 0.9200 / 0.00093 |
| 吞吐 | 153.4 t/s，elapsed 0.5433 h |
| ⚠️ 自描述 | CSV **无 seed / config / commit 列**；`:36` 以追加模式打开固定路径，重跑会拼接 |

**判定**：该数据集为**单 seed 且不自描述**，不得作为方向性结论依据。

---

## B-7. 已确认的失效数据 `[实测·审计复算]`

| 数据 | 归档值 | 重算值 | 判定 |
|------|-------|--------|------|
| `small_lang_metrics.json` → `signal_context_mi` | `0.0`（附 `active_signals: 5904`） | **`0.1275`**（`normalized_mi 0.0869`，`active_signals: 3993`） | ❌ **失效** |
| 置换检验（300 shuffle） | 未做 | 零模型均值 0.00078，p95 0.00174 → **p ≈ 0.000** | — |
| `baseline_lang_metrics.json` → `total_score` | `56.94`（同文件却写"无语言迹象"） | 已废弃为 `null` | ❌ 自相矛盾（已修） |

> `active_signals` 不一致证明归档值来自**另一个（D0 修复前的饱和）状态**。
> **真相**：MI>0 恰恰因为 `f_bit = food > 0.5×capacity` 被**硬编码进模式定义** → 这是**定义性相关**，不是学习所得。
> 正确表述：信号是**指示符（index）**，不是符号（symbol）。

---

## B-8. 信号系统的实现事实 `[实测·审计]`

| 项 | 事实 |
|----|------|
| 模式位定义 | `pattern = e_bin(发送者自身能量档, 2bit) × 4 + f_bit(自身格食物>50%cap) + n_bit(自身格拥挤)` |
| **实际用到的模式数** | **4**（仅 12/13/14/15 非零：1849/305/1281/558；模式 1–11 = 0） |
| 原因 | 高 2 位是**发送者自己的能量档**，只有 `e_bin=3` 的个体才发射 |
| 数据结构 | 每格 1 个 `uint8`，多发射者**覆盖写** → **符号串结构不存在**，组合性/语法原理上不可达 |
| 文档声称 | "16 模式 = 足够的符号空间（类比音素/词位）" → ❌ **与实现不符，差 4 倍，且 2 位被自身状态占死** |

---

## B-9. D3 世代时间 `[文档+审计复算]`

世代间隔 ≈ **7.3 × rotation_period**（4 档 rp 全部吻合）

| rp | 世代间隔 | 200k tick 代数 |
|----|---------|--------------|
| 2400（默认） | 16667 | **12** |
| 1200 | 8696 | 23 |
| 600 | 4444 | 45 |
| 300 | 2247 | 89 |

**含义**：生态无死亡压力（94.9% 老死）时世代被**寿命上限**锁死。原"百万 tick"目标在默认参数下 ≈ 12 代。
> 六模型结论：**"百万代"撤出主线；主形态 = 千代 × 十 seed × 有对照。统计功效来自重复，不来自长度。**

---

## B-10. 项目自己已经走对的关键一步 `[文档]`

科学问题已从 **"语言会涌现吗"**（不可证伪）改为 **"语言涌现的最小充分条件是什么"**（可证伪、可定界）。
D2 四大机制（学习瓶颈 / 任意性码本 / 信息不对称 / Steels 对齐）逐条对应 Kirby 迭代学习、Saussure 任意性、Steels 语言博弈。

**但 D2 验收门未过** `[文档]`：S1 → g15 0.5→0.238；S2 → 0.34 且 `codebook_conv` **下降**（0.9945→0.7543，码本分化而非收敛）；全机制默认参数 9000 tick 灭绝。

---

## B-11. D2.1 参数网格 —— 数据无效 `[实测 2026-09-11 R1]`

> 这是**三份决策文档的经验基础**，也是本次审计最重要的发现。

| 检验项 | 结果 |
|--------|------|
| 7 组 × 2 seed 配对 | **3 组逐字节全等**（lr005/lr0075/lr0125），4 组**记录序列共享前缀** |
| lr010/lr030 公共前缀 | **4 行**：`2000:0.3674 4000:0.1482 6000:0.2062 8000:0.4124` |
| m2400/m3600 公共前缀 | 1 行 |
| **manifest 记录的 seed** | **全部 `config.seed = 42`**（含所有 `_s43` 文件） |
| manifest `ticks` | 1000（实际产出 **60000**）→ **manifest 与产出脱节** |

**判定**：D2.1 并非"2 seed 复现"，而是**一条轨迹**（因全局 RNG 流导致后期采样错位，造成"看似不同"的假象）。

**报告引用的数值本身准确**（lr010 s42=0.2442、s43=0.3459、峰值 0.8556/0.8695@14k 均复算一致）——
**数字没抄错，是数据不是它声称的东西。**

**⚠️ 反向解读实例**：战略转向报告把 lr030 的 `s42=0.288 / s43=0.449`（56% 差异）读作"**双稳态**"。
manifest 证明两者同为 `seed=42` → **正确解读是"复现失败"**。

---

## B-12. `signal_context_mi` 分歧 —— 已裁定 `[实测 2026-09-11 R1]`

用项目**自己的函数** + 项目**自己的快照**：

```
data/snapshots/snapshot_small.npz
  marks nonzero=3993  unique=[0 12 13 14 15]
  resource_grid range=(-0.0000, 120.0000)
>>> signal_context_mi(...) = {'mutual_information': 0.127534,
                              'normalized_mi': 0.086911,
                              'context_bins': 3, 'active_signals': 3993}
```

| 结论 | 内容 |
|------|------|
| 裁定 | **审计方正确：MI = 0.1275**；归档 `0.0` 为失效数据 |
| 本地得 0.0 之因 | 函数**确实接收** `resource_grid`；传全零/错误通道 → 单箱 → MI 构造性为 0 |
| 附加缺陷 | 分箱退化：`bin0=3569(89.4%) / bin1=125(3.1%) / bin2=298(7.5%)` → **须修** |
| 附加确认 | 模式发射 `{12:1849, 13:305, 14:1281, 15:558}`，模式 1–11 = 0 → **实际符号空间 = 4** |

---

## B-13. R4 批（ON/OFF）生态分档 —— 生态门不通过 `[实测 2026-09-12 R3/R4]`

> 数据：`_rerun_logs/r4_asym_batch/`（ON，15 run）、`_rerun_logs/ref_sym_off/`（OFF，15 run）；均已入库 `88c08ba`
> 口径：60×120，uniform，N0=200，max=5000，`use_sim_core=False`，log_interval=1000

| 项 | ON（info-asym ON, r=4） | OFF（r=8 对称） |
|----|----------------------|----------------|
| manifest 跑完（有 `end_time`） | **11/15** | **0/15**（全部只有 `start_time`） |
| 到 60000 且 N>0 | 6 个（N={144,146,2,2,47,30}） | 0（最大 tick 42000，多在 11–15k 中止） |
| 灭绝 run | 5（9812/9631/2371/29586/57476） | 0（中止非灭绝） |
| 两臂自变量差 | **仅** info-asym 四参（radius/noise/tau） | 同左 |
| `codebook_conv` | **15/15 恒 1.000**（因 `arbitrary_codebook=false`，**常数非测量**） | 同 |

**生态门判定**：按 `AGENTS.md` §生态稳定前置门（≥5 seed 多数存活、N 稳定 >0 持续 50k）→ **不通过**（每档均未过"多数"；lr100 虽 3/4 到 60k，但 2 个 N=2）。
**总判定**：R4 批 **D2 gate / C3 不可执行**（R3/R6）。

**⚠️ g15 分层口径更正（与 `_share/实况.md` §三③ 冲突）**：实况正文称"N≥30 的 5 run 终局 g15 全部 <0.5"，但其**自带的表**已含 0.563 / 0.551 / 0.887 三个 >0.5 ⇒ **正文措辞错误**（表正确，我的复算逐值一致）。正确表述：**5 个中 3 个终局 >0.5，其中 lr300_s44 升至 0.887@60k（非回落）**。见 F-D13。

## B-14. 生态崩溃归因 —— `nb[:4]` 缺陷，**已修** `[实测+代码 2026-09-12 R3/R4]`

| 事实 | 值 |
|------|-----|
| R4 批运行时间 | **10:12–10:52**（ON manifest `start_time`） |
| 修复引入时间 | **`7eeaa5f`：2026-09-12 12:21** ⇒ R4 批**跑在修复前** |
| main 现状 | `sphere_engine.py:759-766` 已用 `neighbors_von_neumann`（不再 `nb[:4]`） |
| E1 我亲手复跑（1500t×3seed） | `buggy_r4` 中位 **24** / `vn_r4` **120** / `off_r8` **105** / `diag_r4` 92 / `r6_noop` 80 |
| 与仓库 CSV | **逐值一致**（可复现） |
| 结论 | **信息不对称不致病**；崩溃 = `nb[:4]` 半平面偏置；已修 |
| ⚠️ 未验证 | 修复后 **5 seed × 60k** 的生态门（E1 仅 1500t） |

## B-15. 归档 manifest 自描述缺口 `[实测 2026-09-12 R4]`

R4 批两臂 manifest 含 `args/config/start_time/[end_time/final_tick/extinct/final_N]`，**不含** `git_commit` / `sim_core_sha256` / 配置指纹。
⇒ 当日代码在 12:21 变更（`7eeaa5f`）后，**归档无法自证所用代码版本**，只能以 `start_time` 推断为 pre-fix。见 F-D15。

## 更新记录

| 日期 | 更新内容 | 依据 commit |
|------|---------|------------|
| 2026-09-11 | 基线建立（B-0 ~ B-10） | `_gitee_review` main = f6fba18 |
| 2026-09-11 | **R1 更新**：主干推进至 `55245c1`；新增 ref 陷阱、B-11（D2.1 数据无效）、B-12（MI 裁定 0.1275）；`step_tick` 仍缺（F-V1 未闭） | `_gitee_review` main = 55245c1 |
| 2026-09-12 | **R3/R4 更新**：主干 = `313c307`（= `gitee/main`，**未 pull**）；新增 B-13（R4 批生态分档，**生态门不通过**）、B-14（`nb[:4]` 已修 + E1 复现）、B-15（manifest 缺 commit/指纹）。新增缺陷 F-D13~F-D16、F-R5 | `_gitee_review` main = 313c307 |
