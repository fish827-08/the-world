# 设计 · 生物圈 MVP v0.1（B 线 W0-B/W2 交付件）

> 起草：B 线（生物圈与演化）2026-10-18｜状态：**草案，待 owner/审核线批准后冻结**
> 上游：`docs/设计文档/设计总纲-世界孵化器-v0.1-20261018.md` §四（B 线 MVP 设计）、
> `内部汇报/派工-20261018-W0双线并行-结构文档与测试瘦身-提示词.md`（B 线 · W0-B/W2）
> 代码：`biosphere/`（genome / develop / action / resolve / metrics / world_adapter /
> simulator / demo / experiments），测试 `tests/test_bio_*.py`（88 例，2.3 s）
> 复现：`python -m biosphere.demo --ticks 1000 --seed 42`｜
> B6 对照：`python -m biosphere.experiments control --seeds 42,43,44 --ticks 1000`｜
> B8 双型：`python -m biosphere.experiments dualtype --seed 42 --ticks 2000`｜
> B7 冒烟：`python -m biosphere.world_smoke engine`（+ `npz --path <run_bare_world 产出>`）

---

## 〇、一句话

**点状 agent + 可变长基因组 + GRN 发育 + 六类带成本动作 + 两段式提交**，
跑在一块合成小世界（陆地/海洋/食物斑块/季节）上；轨迹自动分类为五类行为，
输出谱系与交互网络。**默认不接旧引擎、不改 `world/` 与 `simulation/`**——
biosphere 是纯新增包，"关档等价"由构造保证（无人 import 即零影响）。

---

## 一、范围与本版不做

| 做（MVP） | 不做（后续阶段） |
|---|---|
| 64 槽可变长基因组（插删突变） | GPU/Taichi（W3） |
| GRN 发育 → 控制器 | 接真实世界气候/斑块字段（W1+，须变更单） |
| 六动作意图 + 显式成本 | 多细胞/粘附成群/组选择（W4） |
| 两段式提交裁定（一格一写者） | 跨硬件逐位（改三级确定性） |
| 行为标签/谱系/交互网络 JSON | O 线化学起源（W5） |
| 合成小世界冒烟 | 学习瓶颈/码本（D2 家族，另线） |

---

## 二、世界界面（只读适配）

biosphere 经 `biosphere/world_adapter.py` 读世界字段，**禁直接 import `world/`**：

| 字段 | 来源（真实世界，W1+ 预留） | MVP 合成世界 |
|---|---|---|
| `surface_type` | `EnvField.surface_type`（uint8，C0b 编码：陆0/海1/架2/河3/淡湖4/咸湖5） | 陆/海二值 |
| `elevation_m` | `EnvField.h_m` | 陆 0.5–4.5 m，海 −1–−4 m |
| `depth_m` | `sea_level_m − h_m`（陆 = 0） | 同 |
| `patch_id` | `ResourceField._patch_mask` + BFS 连通标注 | 10 个半径 2 圆形斑块（只落陆地） |
| 季节相位 | `LightAndTemperature.tilt_rad / season_period` | tilt 0.20 rad / 周期 400 tick |

缺字段 ⇒ `NotImplementedError`（fail-loud，不静默退化）。资源场 MVP 期由
`SyntheticResource` 本地承载（世界侧 `ResourceField` 的替身）；接真实世界时
改走 `ResourceField.consume/deposit`，**须先报跨线变更单**。

---

## 三、基因组（64 槽，可变长）

布局（**冻结；改布局 = 存档格式升级，须变更单**）：

| 段 | 槽位 | 长度 | 说明 |
|---|---|---|---|
| 参数段 | 0–21 | 22 | 固定语义，按名寻址（`PARAM_SLOTS` 注册表） |
| GRN 段 | 22–63 | 42 | 6×6 权重（36）+ 6 阈值 |
| 冗余段 | 64+ | 0–64 | 插入突变产物，发育不读，参与遗传 |

长度 ∈ [32, 128]，名义 64。突变三型：**点突变**（70%，高斯噪声 σ=0.25）/
**插入**（15%，长度+1）/ **删除**（15%，长度−1）；贴边时插入/删除退化为点
突变（RNG 照常消费）。RNG 消费顺序写死：类型→位置→值。

参数段语义（槽位: 名字 —— 含义）：

| # | 名字 | 含义 | # | 名字 | 含义 |
|---|---|---|---|---|---|
| 0 | sense_food | 觅食感知增益 | 11 | move_cost_mult | 移动能耗乘子 |
| 1 | sense_threat | 威胁感知增益 | 12 | habitat_aquatic | 水生适应 |
| 2 | sense_kin | 同类感知增益 | 13 | habitat_terrestrial | 陆生适应 |
| 3 | sense_signal | 信号感知增益 | 14 | adhesion | 粘附（结群吸引） |
| 4 | move_bias | 基础移动倾向 | 15 | defense | 防御减伤 |
| 5 | eat_greed | 进食倾向 | 16 | eat_amount | 单次进食量 |
| 6 | attack_drive | 攻击倾向 | 17 | signal_payload | 信号强度 |
| 7 | share_drive | 分享倾向 | 18 | repro_thresh | 繁殖能量门槛 |
| 8 | signal_drive | 发信号倾向 | 19 | lifespan_gene | 寿命基因 |
| 9 | repro_drive | 繁殖倾向 | 20 | give_up | 失望阈值 |
| 10 | metabolism_base | 基础代谢倍率 | 21 | persistence | 赶路惯性 |

---

## 四、发育（GRN：基因组 → 控制器）

`develop(genome)` 是**纯函数**（无 RNG、同基因组 ⇒ 控制器逐位一致）：

1. 参数段 22 个基因 sigmoid 到 [0,1]；
2. GRN 段解码：`W = 2·tanh(前36)`、`t = tanh(后6)`；
3. 动力学：`s ← clip(W·s + t + 0.25·母体因子, 0, 1)`，迭代 16 步，
   初态 `s₀ = 0.5·1`（"受精卵"环境全员相同）；
4. 读出：感知增益 ×`(0.5+sᵢ)` 缩放，生理常量各自钳制。

**映射本身可演化**（W 和 t 都是基因）。已登记事实：参数钳制饱和
（canalization）会使**单次**点突变偶发表型静默（实测 ≈45%）——
测试 `test_bio_develop.py::test_single_mutation_often_silent_by_saturation`
钉住此现象；连续突变更必有表型变化（10 步内，20/20 种子）。

控制器参数域（`ControllerParams`，钳制值）：感知增益 [0,2]；动作倾向
[0,1]；metabolism [0.5,2]；move_cost_mult [0.5,2]；lifespan [200,2000]；
repro_thresh [40,120]；habitat_* [0,1]；adhesion/defense [0,1]；
eat_amount [0.2,1]；signal_payload [0,1]；give_up [3,20]；persistence [0,1]。

---

## 五、传感器与决策

每个 tick 每个体感知**自身格 + 4 邻域**（von Neumann，列向环绕）：

| 通道 | 量 | 用途 |
|---|---|---|
| 食物 | 本格/邻格资源量（归一化 /4） | EAT 得分、MOVE 食物梯度 |
| 同类 | 邻域个体数 | 结群（MOVE 吸引项） |
| 威胁（邻格） | 邻域个体攻击倾向之和 | MOVE 排斥项 |
| **威胁记忆**（B6 新增） | 最近攻击者所在格 + 距上次受击 tick 数（有效期 `THREAT_MEMORY_TICKS=12`） | MOVE **逃逸项**（见下） |
| 信号 | 本格信号层读数（上 tick 写入、衰减 0.90） | （MVP 观察项，W2 入决策） |

**B6 威胁记忆通道**（2026-10-18 增补）：个体被攻击**命中**（致死或致伤）时，
resolve 写入 `threat_cell`（攻击者格）与 `threat_age=0`；每 tick 老化 +1，
≥12 失活。决策侧 MOVE 邻格评分加逃逸项：

```
逃逸项 = sense_threat × THREAT_MEMORY_WEIGHT(3.0) × (1 − age/12)
         × (候选格 − 当前格) 的威胁格网格距差
```

即"比当前格更远离最近攻击者"的邻格得分更高——直接对应 fleeing 标签的
"受击后 3 tick 与攻击者距离增量 ≥ 0.5 格"。开关
`BioMVPConfig.threat_sensing`（默认 True）**一总关闭**三类威胁入口
（邻格排斥 + 记忆逃逸 + SIGNAL 威胁增强），供 B6 对照臂使用。

决策 = 六类动作得分 argmax + 确定性抖动（ε=0.15，每体每 tick 一次
`uniform(-ε,ε,6)` 抽取，RNG 按 uid 序消费）。得分设计全部 O(1) 量纲：

| 动作 | 得分 | 目标选择 |
|---|---|---|
| MOVE | move_bias × 最佳邻格分 + 0.05 | 食物梯度 + 同类吸引 − 威胁排斥 + 逃逸项最高格 |
| EAT | eat_greed × 2 × food01(本格) | 自身格 |
| ATTACK | attack_drive × (有邻目标?1:0) | 邻域最低能量个体 |
| SHARE | share_drive × (有饥邻?1:0) | 邻域能量 < 自身半数者 |
| SIGNAL | signal_drive × (0.5 + 威胁/食物增强) | 自身格 |
| REPRODUCE | repro_drive × (达标?2:0) | 无 |

---

## 六、动作与成本（显式记账）

| 动作 | 基准成本 | 效果 | 失败模式（成本照付） |
|---|---|---|---|
| MOVE | 0.30 × move_cost_mult | 位移一格 | stay_contested / stay_occupied / stay_habitat |
| EAT | 0.05 | 本格资源 −min(eat_amount, 余量)，能量 +同值 | eat_empty |
| ATTACK | 0.40 | 目标能量 −(4+8×drive)×(1−0.6×defense)；致死则掠食 +50%×死前能量 | attack_no_target / attack_survived |
| SHARE | 0.20 | 转移 payload（=10×share_drive，保留 1.0 底线） | share_no_target |
| SIGNAL | 0.15 | 信号层 +5×payload（邻居次 tick 可读） | 恒成功 |
| REPRODUCE | 0.50 | 子代落空邻格，父 −20（endowment），子 +20 | repro_no_energy / repro_no_room |

**记账口径**：尝试即付成本（成败都付）；`energy_in` 只计世界侧流入
（吃资源 + 掠食——掠食取自将死个体）；endowment 是系统内转移另计。
台账 = `ResolveReport.energy_paid / energy_in / endowment_paid`。

**栖息（可演化）**：进水需 `habitat_aquatic ≥ 0.30`，上岸需
`habitat_terrestrial ≥ 0.30`；两栖 = 两者都过线。错配栖息的个体每 tick
代谢罚 ×(1+1.0)。⇒ 登陆/下水是**可演化事件**（突变改变 habitat_*，选择
筛选）。

**繁殖**：能量 ≥ repro_thresh 且种群 < 512 且有空格 ⇒ 子代基因组 =
父基因组一次突变（RNG 消费在 resolve 内按 uid 序）。**死亡**：能量 ≤ 0
（starved）/ 年龄 ≥ lifespan（old_age）/ 被攻击致死（attacked）。

**B6 攻击口径补丁**：命中（伤害 > 0）即写受害者威胁记忆（致死也写，
谱系可追溯）；实际死否仍看能量（致伤幸存者获得逃逸窗口）。

---

## 七、两段式提交（确定性核心）

1. **第一段·并行写意图**：个体只写各自 `Intent` 缓冲，不碰共享状态；
2. **第二段·按 uid 升序裁定**：
   - 2a MOVE 声明（一格一写者：dest 先到先得；栖息不过线不声明）；
   - 2b 位移**链/环分解**（沿"目标格占用者是否也撤离"递归：空/环 ⇒ 通，
     撞不动者 ⇒ 断）——静态撤离表会漏判"让位者自己失败"造成双占（已修，
     `test_bio_resolve.py::test_chain_blocked_both_stay` 回归钉住）；
   - 2c 位移生效（uid 序）；2d 其它效果生效（uid 序：吃/攻/享/信/繁殖）。

⇒ 同 seed 两次**逐位一致**（`test_bio_demo.py::test_same_seed_byte_identical`
字节级钉住；metrics JSON 亦然）。唯一随机源 = 构造注入的 PCG64 Generator，
消费顺序 = uid 序（决策抖动）→ uid 序（繁殖突变）。

---

## 八、行为标签（预注册判据的读数器）

窗口制（默认全 run + 100-tick 分块序列）；距离 = 网格距（列向环绕）：

| 标签 | 规则 | 阈值 |
|---|---|---|
| foraging 觅食 | 窗口内进食净增益 ≥ 阈值 且 平均步长 ≥ 阈值 | 1.0 ｜ 0.20 格/tick |
| fleeing 逃避 | 受击（**命中即可：致死或致伤**）后 3 tick 内与攻击者距离增量 ≥ 阈值 | 0.5 格 |
| chasing 追击 | 攻击前 3 tick 内与目标距离减量 ≥ 阈值 | 0.5 格 |
| grouping 结群 | 窗口内最近邻距离均值 ≤ 阈值（有效样本 ≥ 30 帧） | 1.5 格 |
| signaling 信号 | 窗口内信号意图数 ≥ 阈值 | 1 |

### 预注册判据（MVP 批次，**待 owner/审核线批准**）

| # | 判据 | 通过条件 | 现状（seed 42 冒烟） |
|---|---|---|---|
| P1 生态门 | ≥3 seed 多数存活至 1000 tick 且无 500-tick 空窗 | 必过才谈行为 | 单 seed：48→91，峰值 94 ✅（多样本未跑） |
| P2 觅食/结群 | 多数 seed 两类标签 count > 0 | 觅食 27、结群 157 ✅ |
| P3 追击 | 观察项（不设通过线） | 22 |
| P4 逃避 | 观察项 | **B6 后 120 / 53 / 18（3 seed 全正）** ✅ 见 §8.1 |
| P5 谱系演进 | max_generation ≥ 3 | 5 ✅ |
| P6 消融可辨 | 四零模型 vs 完整档行为计数差（W2 定量） | 开关未接，见 §九 |

#### 8.1 B6 诊断与修复记录（2026-10-18）

**现象**：W0-B demo 攻击 2186 次、致死仅 131 次（7.0%），逃避标签 0。

**根因（两种都不是生态，是标签 bug）**：逃避标签原只统计
`status=="ok"`（= 致死攻击）——而**致死受害者 3 tick 后必然无帧**
（死者不进 `SimLog.frames`）⇒ "受击后 3 tick 距离增量"窗口**恒空**
⇒ fleeing 结构性 = 0。实测致死率 5.4–11.5%、幸存窗口 212–1485 个/run
全部被漏统计。修复 = 「命中即可（致死或致伤）」
（`metrics._ATTACK_HIT_STATUSES`；回归测试
`test_bio_threat.py::TestFleeingLabelFix`）。

**增强（威胁记忆通道）**：仅修标签后 fleeing 为漂移级（24/0/1，3 seed）；
加威胁记忆通道后见 §8.2 对照。攻击致死率低（≤12%）证明
`FLEE_DIST_GAIN_MIN=0.5` **结构性可达**（不是阈值问题）：命中窗口距离
增量分布 threat_on 臂 16–28% 达标、threat_off 臂 0.3–1.9%。

#### 8.2 B6 两臂对照（`python -m biosphere.experiments control`，1000 tick × 3 seed）

| 臂 | seed | fleeing | chasing | 致死率 | 窗口可用 | 窗口达标率 |
|---|---|---|---|---|---|---|
| threat_on | 42 | **120** | 54 | 7.6% | 1400 | 28.1% |
| threat_on | 43 | **53** | 14 | 8.4% | 675 | 21.6% |
| threat_on | 44 | **18** | 5 | 10.3% | 212 | 16.0% |
| threat_off | 42 | 4 | 7 | 11.0% | 1241 | 1.9% |
| threat_off | 43 | 0 | 3 | 9.9% | 290 | 0.3% |
| threat_off | 44 | 1 | 6 | 6.8% | 274 | 0.7% |

⇒ 威胁记忆通道使 fleeing 从"漂移级 ≈0"变为"三 seed 全正、量级与
chasing 同阶"；threat_off 臂同时致死率略升（猎物逃不掉）——生态上自洽。

#### 8.3 B8 捕食者-猎物双型对照（`python -m biosphere.experiments dualtype`，2000 tick）

人工初始化两类（**只差攻击基因一个槽位**：捕食者 +3.0 / 猎物 −3.0），
20 + 40 初代：

| seed | chasing(all) | fleeing(all) | 捕食者创始 chasing | 猎物创始 chasing | 致死率 |
|---|---|---|---|---|---|
| 42 | **14** | **136** | 5 | 1 | 9.8% |
| 43 | **9** | **32** | 5 | 0 | 6.7% |

⇒ **chasing 与 fleeing 同时出现**（"行为判据可执行"的证明）；创始捕食者的
chasing 是猎物的 5 倍（seed 43 猎物为 0），类型差异在行为读数上可见。
注意：950/579 次繁殖后突变会稀释初始类型 ⇒ 全 run 计数以后代为主，
创始分解见 metrics JSON `dualtype.founder_breakdown`。

#### 8.4 B10 chasing 偏低诊断（`chase_gain_distribution`；结论 = **行为性，判据不改**）

现象：B8 双型 2000 tick fleeing=136 但 chasing=14。按复盘件建议出对称分布
（已固化为 metrics 永久仪器，进每次 metrics JSON）：

| 场景 | hits | drop≥0.5 | **3 tick 前已贴贴** | 目标送上门 | 攻击者逼近 |
|---|---|---|---|---|---|
| 双型 seed42（2000t） | 2307 | 2.8% | **93.1%** | 4.4% | 2.6% |
| 双型 seed43（2000t） | 2692 | 1.2% | **97.1%** | 1.8% | 1.2% |
| demo seed42（1000t） | 1772 | 13.5% | **76.7%** | 9.9% | 13.3% |

**判定：行为问题，不是判据问题**（按派工令"行为问题如实记录、不改"）：
① 判据工作正常——chasing 标签恰落在"攻击者逼近"桶（双型 2.6%/1.2% ⇒ 14/9
个个体；demo 13.3% ⇒ 54 个），无漏计；
② 低计数源于行为：**攻击目标规则 = 最弱*相邻*个体**，而 77–97% 的命中在
3 tick 前目标就已贴贴（猎物觅食游走进攻击范围——"送上门"），攻击者
无需逼近；
③ 与 fleeing 的不对称部分是**机制性**的：fleeing 有 B6 威胁记忆通道
主动驱动逃逸，chasing **没有对应的猎食驱动**（MOVE 无寻猎项）。
**建议（记录待 W5 裁量，本轮不动）**：若要 chasing 成为一类主动行为，
加"寻猎 MOVE 项"（向邻域最低能量个体梯度移动）；标签阈值不动
（预注册判据，改须重走预注册）。

---

## 九、四个零模型（同跑对照；**MVP 期规格冻结、实现列 W2**）

| 零模型 | 语义 | MVP 实现状态 |
|---|---|---|
| Z1 关信号 | SIGNAL 动作禁用 + 信号层不衰减不写入 | 未接（W2：`ResolveConfig` 加开关） |
| Z2 关捕食 | ATTACK 禁用（个体间无能量转移） | 未接 |
| Z3 关粘附 | adhesion 恒 0（结群吸引消失，空间结构保留） | 可用初始基因组近似（adhesion 基因置 0） |
| Z4 全混合 | 无空间结构：每 tick 位置全局重排（食物照吃） | 未接（W2：resolve 后 shuffle） |

> 纪律：零模型与完整档**同 seed 同世界**对跑；差异只归因于被关机制。

---

## 十、参数总表（代码常量；改值 = 判据/口径变更，须重走预注册）

| 常量 | 值 | 出处 |
|---|---|---|
| 代谢 | 0.035 × metabolism(0.5–2)/tick | `simulator.METABOLISM_SCALE` |
| 栖息错配罚 | ×(1+1.0) | `simulator.HABITAT_PENALTY` |
| 信号衰减 | 0.90/tick | `simulator.SIGNAL_DECAY` |
| 季节调制 | 回复率 ×(1+0.30·sin 2π·phase) | `simulator.SEASON_AMP` |
| 决策抖动 ε | 0.15 | `simulator.JITTER` |
| 初代 | 48 体 / 60 能量 / 只落陆地 | `simulator.INIT_POP/INIT_ENERGY` |
| 世界 | 32×64、陆率 0.45、10 斑块 r2 cap 12、背景 cap 1.0、回复 0.08/tick | `BioMVPConfig` |
| 栖息下限 | 0.30 | `resolve.HABITAT_MIN` |
| **威胁记忆** | 有效期 12 tick；逃逸项权重 3.0 | `resolve.THREAT_MEMORY_TICKS` / `simulator.THREAT_MEMORY_WEIGHT` |
| **威胁传感开关** | 默认 True（False = 对照臂） | `BioMVPConfig.threat_sensing` |
| 攻击 | power 4+8×drive；mit 0.6×defense；loot 50% | `resolve.ATTACK_*` |
| 分享 | scale 10；reserve 1.0 | `resolve.SHARE_*` |
| 信号 | scale 5 | `resolve.SIGNAL_SCALE` |
| 繁殖 | endowment 20；max_pop 512 | `resolve.REPRO_ENDOWMENT/MAX_POP` |
| GRN | 6 节点 / 16 步 / W 缩放 2.0 / 注入 0.25 | `develop.GRN_*` |
| 基因组 | 名义 64、界 [32,128]、点/插/删 = 70/15/15% | `genome.*` |
| 标签阈值 | 见 §八 表 | `metrics.*_MIN/MAX` |

---

## 十一、测试与预算（B4 + B6）

`tests/test_bio_*.py` 共 **110 例 / 2.4 s**（本机 venv 3.14.7）：基因组
突变不变量（13）、GRN 确定性（9）、动作成本（8）、两段式裁定确定性 +
一格一写者 + 记账（27）、度量标签与形状（13）、**威胁通道 + fleeing 标签
修复（13，B6 新增）**、**真实字段适配（12，B7）**、**攻击窗口分布仪器
（10，B10）**、demo 冒烟 + 逐位一致（4）。
subprocess 测试双钉 UTF-8（env + `encoding="utf-8"`）。

---

## 十二、已知风险 / 未做

1. ~~**逃避未涌现**（P4 = 0）~~ **B6 已闭环**：根因 = 标签只算致死攻击
   （死者无帧 ⇒ 窗口恒空，非生态问题）；修复 + 威胁记忆通道后三 seed 全正
   （120/53/18），见 §8.1/§8.2。
2. **单 seed 冒烟**：生态门 P1 需 ≥3 seed 多样本（W2 批）。
3. **零模型开关未实现**（§九）：规格已冻结，实现列 W2。
4. **接真实世界未做**：`adapt_real_world` 已按鸭子类型写好但未跑通
   （须跨线变更单 + C0b 数据；B7 冒烟见汇报）。
5. **信号不入决策**：MVP 信号只进交互网络与信号层（读数存在，无获益通道）；
   C3（发送者获益）属后续。
6. 合成世界行向截断（极点简化）：接真实 `SphereWorld.neighbors` 后消失。

---

## 十三、W4 多细胞/合作（CPU；2026-10-18 补充指令冻结）

### 13.1 细胞定义（写死）

**细胞 = agent**：能粘（adhesion）、能双稳态分化（bistable
differentiation）、能动（每 tick ≤ 1 格）。**不是分子细胞**。
多细胞 = N 个 agent 粘成团，按团繁殖/死亡。**细胞尺度与个体尺度分开建模块**：
细胞尺度在 `biosphere/cell.py`（类型/栖息约束/分化开关），个体尺度在
genome/develop/action/resolve，团尺度在 `biosphere/groups.py`。

### 13.2 栖息 = 物理约束（不写标签）

| 细胞类型（两个栖息基因定，可演化） | 准入约束 | 不适配代价 |
|---|---|---|
| 水生 | 所在格 `depth_m > 0` | 代谢罚 ×1（上岸搁浅） |
| 陆生 | 所在格 `soil_water ≥ world.soil_water_min` | 代谢罚 ×1（脱水） |
| 两栖 | 两者皆可 | 代谢罚 ×0.5（各有代价） |
| 未定准（两基因 < 0.5） | 都能待 | 代谢罚 ×1（全程付费） |

阈值 `soil_water_min` 是**世界侧参数**（`WorldView.soil_water_min`，
合成世界默认 2.0；真实世界由 A 线降水/土壤水耦合经
`adapt_real_world(..., soil_water_min=)` 传入）⇒ 与 A 线真降水的耦合点
已留接口。⇒ 登陆/下水是由字段约束 + 选择塑形的演化事件。

### 13.3 双稳态分化（迟滞开关）

`diff_state ∈ {0,1}`：邻体数 ≥ 3 ⇒ 翻 1；≤ 1 ⇒ 翻 0；中间保持（迟滞 =
双稳态）。state=1（体细胞态）表达全额粘附，state=0（生殖/游动态）只表达
50% 粘附 ⇒ 分化有机械后果（分工：体细胞粘团、生殖细胞游动）。
开关阈值目前是常量（W4b 可基因化）。

### 13.4 团机制

- **团**：占用图连通片（相邻即成团）；并查集判定、按最小 uid 编号；
- **团共享能量**：每 tick 每团建池，成员按 `0.30 × share_drive` 比例注入、
  池子平分；注入比例 < 1/n 者**净蹭团**（cheater）；
- **按团繁殖（组选择）**：团总能量 ≥ `80 × 团大小`（人均门槛 = 个体繁殖
  门槛均量级，不给团退化性优势）⇒ 成本团员平摊、基因组取团内最高能量者
  （生殖系）、每次 2 个子代落团邻空格；
- **粘附**：MOVE 贴团奖励（`2.0 × adhesion_eff`）+ 粘附个体（表达后
  ≥ 0.5）**硬过滤**（团在时只贴团移动）。

### 13.5 判据（四件套，全进 metrics JSON `groups` 段）

| 判据 | 读数 | 零模型预期 |
|---|---|---|
| 团大小分布 | `size_histogram_final` / `max_size_final` / `mean_size_multi_final` | 全混合 ⇒ 最大团 ≤ 4 |
| 团内分化（表达方差） | `within_var_final`（attack/share/adhesion/metabolism 四参数） | 全混合 ⇒ ≈ 0 |
| 作弊者频率 | `cheaters.frequency`（公共池净蹭 > 1.0 能量） | 关组选择 ⇒ 池消失 ⇒ 频率不可定义（n_members=0） |
| 谱系存活 | `lineage_survival.fraction`（有活后代的根占比） | — |

另：**团运动学**（`kinematics`）：焦点团（每 tick 最大多体团）质心位移
速度 + 形态半径。**团速 = 质心位移（格/tick），不用单细胞速度解释**；
全混合臂的 20 格/tick 是"焦点团身份每 tick 翻搅"的假象，非 locomotion
（结论页已注）。

### 13.6 四个零模型（`W4Config` 组合表达）

| 零模型 | 配置 | 语义 |
|---|---|---|
| Z1 关粘附 | `adhesion=False` | 贴团奖励 + 硬过滤全关（团只剩偶遇） |
| Z2 关组选择 | `group_reproduction=False` | 无团繁殖（只个体繁殖） |
| Z3 全混合 | `full_mixing=True` | 每 tick 全局重排 ⇒ 空间结构归零 |
| Z4 关信号 | `signal=False` | SIGNAL 意图 -inf 硬压 + 信号层不写 |

### 13.7 可恢复协议（>1000 tick 跑批）

`biosphere/checkpoint.py`：跑批每 500 tick 写 `ckpt_<tag>_t<N>.pkl`
（pickle 全模拟器：种群/rng 态/资源/信号层/日志；原子落盘）+ 追加
`<tag>.run.log` 进度行。恢复 = 同子命令 `--resume <pkl>`；**续跑与直跑
逐位一致**（`test_bio_w4.py::TestCheckpoint` 钉住）。CLI：`--ckpt-dir`
传目录即用；不传则 **ticks>1000 自动开**（目录
`results/biosphere/ckpt`）、≤1000 不写。

### 13.8 W4 首跑结论（2026-10-18，2000 tick × 2 seed；**已被 B12 核验降级，见 §13.10**）

> 🔴 **B12 核验后降级**：结论 1"多细胞结构真实涌现"**不成立**（降级为
> "拥挤团已出现"）——原团定义（邻接团）被密度混淆。W4b（§13.9）起团分
> 两套，主张只看粘附链团。
- 全混合臂最大团 ≤ 4、时间平均多体团数结构差异显著 ⇒ 空间结构判据可辨；
- 粘附的边际贡献体现在**团大小上限**（49 vs 34）而非普遍性（诚实读数：
   MVP 粘附机制效应中等，W4b 可加"分裂/粘连维持"强化）；
- **分化方差低（0.000–0.024）**：双稳态开关在但尚未塑出可测分工——判据
  在位、读数诚实；
- 作弊者频率 0.38–0.65：无惩罚机制的公共品博弈默认漏容；
- 谱系存活 0.02–0.10：多数创始谱系灭绝（组选择尚未形成优势谱系）。

### 13.9 W4b：两套团定义（B12；机制与主张切到粘附链团）

| 镜头 | 定义 | 用途 |
|---|---|---|
| **邻接团** | 占用图连通片（相邻即成团） | **仅作对照**（密度混淆根源：拥挤本身成团） |
| **粘附链团** | 仅按**粘附边**（相邻**且两端都粘**，表达后 adhesion ≥ 0.5、受分化调制）连通的连通片 | **所有主张**（团大小/组繁殖/作弊者/运动学） |

B12 起：团共享能量、按团繁殖、粘附硬过滤、双稳态分化的邻体数
——**一律跑在粘附链团上**；邻接团仅进 metrics 的 `adjacency_control`
段。读数键：`groups.adhesion_chain`（主张）/ `groups.adjacency_control`
（对照）。

**新增判据读数**（`groups.adhesion_chain`）：
- `chain_ratio_mean`：粘附链团比例（多体链团个体占比，时间平均）；
- `adhesion_trajectory`：粘附基因频率轨迹（start/mid/end + 全程均值：
  原始基因均值/方差、表达均值、粘附个体占比、链团比例）；
- `repro_composition`：组繁殖构成（事件数、高/低共享桶、
  **每合格团·tick 的繁殖率**、共享注入比、生殖系来源）、
  `children_survival_by_bucket`（高/低共享团子代的终局存活）；
- `cheaters`：作弊者（净蹭 > margin）——**含已死者流水**（修生存偏差）
  + `per_final_group`（每团作弊者占比）。

### 13.10 W4b 预注册判据（2026-10-18 跑批前写定；实现 `biosphere/criteria.py`）

**生态门（先过门再判读）**
1. 单 seed 过门 = 终局存活 > 0 且全程无连续 500 tick 的 0 存活空窗；
2. 一臂过门 = 过门 seed > 半数（5 seed ⇒ ≥3/5）；
3. **不过门 ⇒ 该臂写"生态门未过、不可判"，不做机制结论。**

**组选择判据（"多细胞是选择出来的"）**
- 比较 `full` vs `关组选择`（Z2），逐 seed 配对，两读数：
  **粘附链团比例**（`chain_ratio_mean`）与 **粘附基因频率**
  （`adhesion_trajectory.whole_run_mean.adhesion_gene_mean`）；
- 方向 = full 高于 Z2；
- **要求 ≥4/5 seed 两项都方向一致** ⇒ "已建立"；否则**如实写"未建立"**
  （不用 p 值表演、不润色）。

**规模**：五臂 × 5 seed（41–45）× 3000 tick；>1000 tick 走 checkpoint
协议（§13.7）。

### 13.11 W4b 判定（2026-10-19）：**未建立**

五臂生态门 **5/5 全过**（可判读），但预注册判据
（§13.10）**方向一致数不足**：粘附链团比例 1/5、粘附基因频率 2/5，
both 1/5（需 ≥4/5）⇒ **"多细胞是选择出来的"未建立**（不润色）。

结构性原因（非读数噪声）：
1. 2/5 seed 上 `full` 从未触发组繁殖 ⇒ `full` 与 `关组选择` **是同一场
   模拟**（数字逐位相同）⇒ 判据只有 3 个 seed 有区分度；
2. 组繁殖稀疏（3000 tick 内 full 仅 3/5 seed 有事件：74/12/28）⇒ 选择压
   维不住粘附；实测**粘附基因频率在 4/5 seed 上全程下降**（≈0.0 → −0.8~−1.5）；
3. "高共享团"几乎是空桶：`ratio_net = SHARE_BASE×mean×n > 1` 需 n≥4，
   而高共享 = 净奉献 ⇒ 达标团少（s42：高共享合格团·tick 13 vs 低共享 656）。

⇒ **B11"多细胞涌现"结论维持冻结/不成立**；下一轮（W4c/机制侧）若要成立，
需改**机制参数**（粘附门槛/共享激励/团能量上限）并**重走预注册**——
不得改标签阈值或判据方向来"过门"。

**W4b 过程修掉的两个真 bug（判据与阈值未动）**：
1. **分化死锁**：分化的邻体数原取自粘附链团 ⇒ 非粘附细胞恒为单体 ⇒
   永不翻到 state=1 ⇒ 表达永远减半 ⇒ 链团永不成形。修复 = **感知取邻接**
   （细胞能看到什么），链团只作主张镜头；修复后粘附占比 2% → 12–62%；
2. **普查口径**：合格团·tick 原按扣费后余额统计 ⇒ 高共享团（净奉献者）
   系统性漏记 ⇒ "高共享繁殖率恒 0"假读数。修复 = 普查记在繁殖**决策点**
   （扣费前）。

### 13.12 W4c 预注册（2026-10-19 跑批前写定；限一轮，每组只改一个旋钮）

针对 W4b 三条结构性原因，每组一个机制旋钮（**其余参数不动**）：

| 组 | 旋钮（`W4Config` 字段） | 档位 | 假设（W4b 读数推出） |
|---|---|---|---|
| G1 组繁殖门槛 | `group_repro_thresh_per_capita` | {40, **80**, 160} | 门槛过高 ⇒ 组繁殖稀疏（W4b：仅 3/5 seed 有事件）⇒ 降门槛应增加事件、给粘附选择压 |
| G2 亲缘结构 | `kin_cohesion`（bool） | {关, **开**} | 混团稀释合作、粘附被负选择 ⇒ 开 = ①MOVE 计数加权贴团奖励（`KIN_COHESION_WEIGHT × adhesion_eff × 候选格链团邻居数`）②粘附个体抖动 ×0.5（降扩散） |
| G3 粘附收益 | `adhesion_benefit`（float） | {0=关, **0.5=中**, 1.0=强} | 粘附只有成本没有收益 ⇒ 群体觅食分红：每 tick 每多体链团（大小 n）均分 `strength × GROUP_FORAGE_BASE(0.1) × n` 能量（计入 `energy_in`，世界侧流入） |

**实验矩阵**（5 seed(41–45) × 3000 tick；`--ckpt-every 1000`）：
`thr40_full/thr40_z2`、`thr80_full/thr80_z2`（= W4b baseline）、
`thr160_full/thr160_z2`、`kin_full/kin_z2`、`benMid_full/benMid_z2`、
`benStrong_full/benStrong_z2`（**12 臂**；每臂的 z2 = 同参数下
`group_reproduction=False`）。W4b 的四个零模型臂本轮不重跑（判据不需要）。

**生态门**（同 W4b §13.10）：单 seed 终局存活 > 0 且无连续 500 tick 空窗；
一臂多数（≥3/5）过门才判读；不过 ⇒ 写"生态门未过、不可判"。

**判据（同 W4b，逐组判定）**：每组内，`*_full` 的**粘附链团比例**与
**粘附基因频率**须双双高于同参数 `*_z2`，且 **≥4/5 seed 方向一致** ⇒
该组"已建立"；否则"未建立"。

**升级边界（硬）**：本轮 = 机制参数修正的**唯一一轮**。若 12 臂全部
"未建立" ⇒ 直接写负结果 + 一页"转 C3 方案"（单细胞层发送者获益通道，
测交流/合作），**不许连续调参刷阳性**、不许改判据方向/种子集/标签阈值。

### 13.13 W4c 判定（2026-10-19）：**6/6 全部未建立 ⇒ 转 C3**

12 臂生态门 **5/5 全过**（可判读），但 6 条判据无一满足 ≥4/5：

| 组 | chain 方向 | adhesion 方向 | both | 判定 |
|---|---|---|---|---|
| G1 门槛 40 / 80 / 160 | 1/5、1/5、1/5 | 2/5、2/5、1/5 | 全 1/5 | 未建立 |
| G2 亲缘结构 | 3/5 | 1/5 | 1/5 | 未建立 |
| G3 收益 中 / 强 | **4/5** / 1/5 | 3/5 / 3/5 | 2/5 / 1/5 | 未建立 |

机制侧真信号：**G1 假设成立**（组繁殖事件总数 thr160=5 → thr80=114 →
thr40=309）；G2/G3 对链团比例有部分正方向。但**无一能把粘附基因频率
扭成上行**（各臂仍 4/5 seed 下降）。

根因（机制诊断）：① 组繁殖**不传粘**——子代基因组取自团内最高**能量**者
（非最高粘附），且子代 `diff_state=0` 出生 ⇒ 表达减半 ⇒ 不粘 ⇒ 扩散；
② 成本每 tick、收益稀奢；③ 无组繁殖事件的 seed 上 full ≡ z2（判据零贡献）。

⇒ **"多细胞是选择出来的"在现行参数域内不成立**（两轮预注册、10 seed 全量）。
按派工令停止调参；**转 C3 方案**（`内部汇报/转进-C3方案-20261019.md`，
单细胞层发送者获益通道；待 项目所有者 裁批）。W4 线若要再启，第一刀应切
"子代继承"（diff_state 继承 + 粘附加权生殖系），但须**新预注册轮**。

### 13.14 C3 v2 预注册（B14.3，2026-10-19 跑批前写定；owner 已批口径）

**机制（只写物理，不写语义）**：新增可遗传 ``badge ∈ K=4``（连续基因
离散化）、``emit_rate``、``response[K]``（signed：正=倾向攻击该徽章
载体、负=倾向远离）、``strength``（进攻击威力）、``learning_rate``
（打斗结果更新 response；**首轮默认关**）。信号发射**耗能**
（`signal_emit_cost_base × emit_rate`）并写入 K 通道徽章层；接收方按
自身 ``response[badge]`` 反应。**无任何"信号→攻击欲"直规则**；
"威慑/友善"只允许从「打斗代价（B14.2 反击风险）→ 选择/学习 →
response 符号」中涌现。

**实验矩阵**（5 seed(41–45) × 3000 tick；>1000 ⇒ checkpoint 每 1000）：
`c3_full`（完整档，记忆关）｜`c3_no_strength`（负对照：strength 不进
威力）｜`c3_badge_reshuffle`（负对照：子代徽章每代重随机）｜
`c3_memory`（观察臂：开记忆）。**判据只对 full 臂下**。

**生态门**（同前）：单 seed 终局存活 > 0 且无 500-tick 空窗；一臂多数
（≥3/5）过门才判读；不过 ⇒ 写"生态门未过、不可判"。

**判据（逐条，预注册）**：
1. **MI(badge, strength) > 0**（逐 seed 报；徽章与强度须相关——否则
   徽章不携带"实力信息"，后面两条无从谈起）；
2. **response 符号对齐 ≥4/5 seed**：各族徽章的 mean response 与其载体
   mean strength 呈**负相关**（强者徽章被回避、弱者徽章被追击）；
3. **强度三分位 × 发射的被攻击率 ≥4/5 seed**：`attacked_emit["high"] <
   attacked_emit["low"]`（强者带可见徽章被攻击率低于弱者）——同时报
   `attacked_all`（全体）与发射/非发射的对比，防"最弱优先"规则混淆。

**负对照预期**：`c3_no_strength` 与 `c3_badge_reshuffle` 的 MI 应显著
低于 full（否则 full 的 MI 是噪声/遗传漂变）；`c3_memory` 仅观察。

**升级边界**：若全部未建立 ⇒ 如实写负结果 + 原因，**不连续调参**；
下一机制轮须重走预注册。

---

## 十三·B15 诊断轮（2026-10-19；预注册 §13.15/§13.16）

> 背景：B14 C3 v2 首轮"未建立"，核验件判"瓶颈在统计功效 + 判据判别力"。
> 本轮定位 = **诊断，不加新机制**：先证明因果链存在（oracle），再量功效，
> 再修判据，最后一页决策。不许本轮顺手调选择压。

### 13.15 B15.3 判据修正（预注册，先写后跑）

针对核验件诊断的三条（判据①负对照同值 / 徽章单斑化 / reshuffle 破坏生态）：

1. **判据①改谱系条件**：弃全局 MI(badge, strength)，改为
   **谱系内徽章标签置换检验**——在每个谱系 root 内打乱 badge 标签
   （strength/root 不动），重算"按 root 加权的层内 MI"得置换分布；
   观测值的单侧 p < 0.05 且 full 与负对照臂**有差别**才判"徽章携带
   实力信息"。全局 MI 只作参考列（`old_mi_global`）。
   动因：全局 MI 高可能只是谱系间差异（奠基者/漂变）——B14 的
   no_strength 负对照同样 >0 即此因。
2. **读数加 K_eff**：有效徽章多样性 = exp(徽章分布 Shannon 熵)；
   K_eff ≤ 2 ⇒ 符号对齐类判据统计功效不足，读数须注明。
3. **reshuffle 对照退役**：B14 实证 4/5 seed 灭绝（生态破坏 ⇒ 不能干净
   当"识别不可能"对照）；其角色由**谱系内标签置换**接替（非破坏性：
   只重排标签、不动生态与能量）。
4. **重判方式**：用 B14 终局 checkpoint（`ckpt_b14_*_t3000.pkl`）恢复
   ⇒ 个体级 (badge, strength, root) ⇒ 新判据。**不重跑**（恢复=那场
   模拟的终局，逐位一致有测试钉住）。
5. **判读**：若新判据下 full 与 no_strength 仍无差别 ⇒ "徽章-强度关联"
   是中性遗传结构 ⇒ C3 v2 判据①不可用，写进决策页。

### 13.16 B15.1/B15.2 诊断（预注册）

**B15.1 oracle 正对照**（3 臂 × 5 seed(41–45) × 1000 tick）：
- `oracle_response`：每 tick 把所有个体 response_state 覆写为"按徽章
  载体真实 strength 设符号"（强徽章⇒回避、弱徽章⇒倾向攻）+ SIGNAL
  保底得分（`oracle_signal_floor=0.5`，防无人发射）；
- `oracle_emitter`：只允许 strength ≥ 0.5 者发射（弱者 SIGNAL 硬压
  −inf；强者 SIGNAL 保底 10 分）；
- `natural`：B14 c3_full 同参数复跑（同 tick 口径对照）。
- **判据③ oracle 式**：`attacked_rate(by_emitting).emitting <
  non_emitting`（发射者被攻击率低于非发射者；≥4/5 seed 方向一致）。
- **判读三分支**：oracle 也不出 ⇒ 因果链断（写清断点）；oracle 出而
  自然不出 ⇒ 选择压问题（进 B15.4 决策的"继续"分支）；两者都出 ⇒
  功效问题（B15.2 量化）。

**B15.2 功效分析**（N ∈ {100, 500, 2000} × {c3_full, c3_no_strength}
× 10 seed(41–50) × 1000 tick；世界随 N 扩：32×64 / 32×64 / 64×128）：
- 效应量②：徽章对 (载体 mean strength, mean response) 的 Kendall
  一致度 ∈ [−1,1]（0 = 无对齐；负 = 威慑方向）；
- 效应量③：`attacked_emit["high"] − attacked_emit["low"]`（负 = 威慑）；
- CI：对 seed 的 bootstrap 95% 区间（10000 次重抽样）；
- 交付：效应量 vs N 曲线 + "要多少 N 才可能检出"的判读（CI 是否
  跨 0、方向是否随 N 稳定）。

### 13.17 B16 捕食-猎物军备竞赛（2026-10-20；owner 定线）

> 背景：C3 威慑线经 B15 诊断判"因果链断在攻击端"⇒ owner 裁定**停止**（`内部汇报/决策-C3威慑线-20261019.md`），
> 转**捕食-猎物军备竞赛**（收益 = 直接能量转移；B14.2 打斗/反击物理已就位）。
> 完整预注册文本：`docs/预注册/预注册-B16-捕食猎物军备竞赛-20261020.md`（**先提交、后跑批**）。

**纪律（B15 教训直接复用）**：oracle 先行（证明因果链）→ 预注册文本先提交 → 才跑批；
负对照必须保持生态可行；未建立就如实写，不连续调参。

**B16.1 oracle 正对照**（3 臂 × 5 seed(41–45) × 2000 tick；性状钉死 0/1，**不跑演化**）：
- `b16_oracle_defense`：半数 defense=1 / 半数 =0（uid 奇偶）；主读数 = 被掠食危害 `killed / 个体·tick`；
- `b16_oracle_attack`：半数 attack_drive=1 / 半数 =0；主读数 = 攻击净能量 `loot − 成本 − 反击`（每个体终身）；
- `b16_natural`：同参数不钉（对照）。
- 判读：≥4/5 seed 方向一致 + 对 seed 的 bootstrap 95% CI。

**B16.2/B16.3 军备竞赛**（4 臂 × 5 seed × 5000 tick；不手派角色，初始方差 = 随机基因组 + `init_trait_spread=2.0`）：
- 臂：`b16_full` / `b16_no_loot`（`attack_loot_frac=0`）/ `b16_no_defense`（`defense_mit=0` 且 `counter_defense_scale=0`）
  / `b16_no_counter`（只关反击，观察臂）。
- ⚠️ **口径更正**：派工原文"关 counter ⇒ defense 无收益"不成立——**减伤才是 defense 的主收益通道**
  （反击需 `8·defense − power > 2`，自然档基本不触发）⇒ 负对照取"减伤+反击全关"。
- 五条判据（≥4/5 seed 方向一致）：① defense 均值上移；② attack_drive 均值上移；③ 高防御者/谱系存活更好；
  ④ 两性状同升或周期波动；⑤ 终局 defense 二型化系数 BC > 0.555。
- 生态门：终局存活 > 0 且无 ≥500 tick 空窗，≥3/5 过门才判读。

**B16.0 并行基建**（`biosphere/parallel.py`）：进程池（Windows spawn）跑 (臂, seed) 作业，
结果合并 + manifest（workers/wall/吞吐/逐作业耗时）；**并行 ≡ 串行逐位**（测试钉 sha），实测 5.51×。
