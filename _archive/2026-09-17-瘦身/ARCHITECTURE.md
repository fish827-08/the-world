# Digital Life 架构导读（the-world /digital\_life）

> 本文档写给 "想看懂这个项目、参与维护和开发" 的人。
> 它不是替代项目自带文档，而是一条从零开始的理解路径：读完本文，你应该能回答 "这个项目在做什么、代码怎么组织、一个 tick 里发生了什么、想改东西该动哪里"。
> 详细手册、维护纪律与研究记录分别见 
>
> `digital_life/MANUAL.md`
>
> 、
>
> `digital_life/PROJECT_MAINTENANCE_PROMPT.md`
>
> 、
>
> `digital_life/RESEARCH.md`
>
> 。



***

## 0. 一句话认识项目

这是一个**人工生命（Artificial Life）实验系统**，不是游戏、不是聊天机器人。

在一个二维网格世界上，一群由 "基因" 决定的数字生命个体要活下去：它们移动、觅食、消耗能量、繁殖、变异、死亡。程序员**只定义物理 / 能量 / 遗传等基础规则**，不定义 "什么生命最优秀、谁该成为捕食者"—— 行为应该由自然选择自己涌现出来。

当前阶段：**Stage 2 — Evolution Observatory**，目标是证明 "环境变化能引起群体统计特征发生稳定、可解释、可重复的跨世代变化"（已达成，毕业状态 READY，2026-09-06）。

技术栈：Python ≥3.12 + NumPy + pydantic，测试用 pytest（151 项）。无 GUI，纯 headless 命令行运行。



***

## 1. 核心哲学：程序只定义规则，不定义结果

这是整个项目最重要的约束，所有架构决策都从它出发。

**程序员只定义 9 类基础规则：**



1. 物理规则（世界大小、边界是否环形）

2. 能量规则（代谢消耗、进食增益、能量上限）

3. 感知规则（个体能看到什么 —— 当前只有 "自己格子 + 8 邻格"）

4. 行为接口（每 tick 做什么：代谢→进食→移动→年龄推进）

5. 遗传规则（基因如何解码成性状）

6. 变异规则（基因如何随机扰动）

7. 生命周期（年龄推进、死亡判定）

8. 繁殖机制（无性繁殖、能量对半、世代 + 1）

9. 环境变化（资源再生、空间异质性）

**程序员永远不直接定义：**



* 什么生命最优秀 / 最聪明

* 谁应该成为捕食者

* 谁应该合作 / 形成社会 / 产生语言

* 生命最终应该变成什么

**两条硬性红线：**



* 全程禁止 LLM / 人工 fitness / 预设物种类别 —— 不要为了 "看起来智能" 提前加入

* 每个新功能必须回答：Why / Evidence / Stage / Impact / Reproducibility / Documentation / Testing，答不上来就不加

**核心循环（理论模型，项目最终形态）：**



```
WORLD → SENSOR → BRAIN → ACTION → ENERGY → SURVIVAL → REPRODUCTION → MUTATION → NEW ORGANISM → WORLD
```

当前代码只实现了其中 "没有感知 / 大脑" 的子集（SENSOR 退化为直接读自己格子，BRAIN 退化为基因决定的固定行为概率）。



***

## 2. 目录总览：每个包是干什么的



```
digital\_life/

├── core/           # ① 生命内核：与"规则无关"的纯数据与契约（最底层）

│   ├── genome.py       # Genome：定长连续基因链（float64 一维数组）容器

│   ├── genetics.py     # 基因→性状解码契约 + 变异/交叉纯函数 + TRAIT\_TABLE

│   ├── organism.py     # Organism：个体（位置/能量/基因/表现型/生命周期/谱系）

│   ├── lifecycle.py    # Lifecycle：年龄推进、死亡判定、死因枚举

│   └── metabolism.py   # 能量收支纯函数（代谢/进食/钳制）

├── world/          # ② 世界：个体生存的空间与资源

│   ├── resource.py     # ResourceField：网格资源场（向量化再生/消费/斑块异质）

│   ├── terrain.py      # TerrainField：通行性布尔网格（v1 全通）

│   ├── environment.py  # Environment：EnvironmentView 协议实现（torus/查询/8 邻格）

│   └── world.py        # World：装配以上三个子对象，对外暴露整体语义

├── evolution/      # ③ 演化：把"个体与世界"变成"种群与选择"

│   ├── population.py   # Population：种群容器，每 tick 总账（行动→繁殖→死亡）

│   ├── mutation.py     # 变异策略包装（genome → 变异 genome）

│   ├── reproduction.py # 无性繁殖：子代 = 亲代基因 + 变异，能量对半

│   └── selection.py    # 选择压判定：繁殖门槛 / 死亡判定（纯涌现，无 fitness）

├── simulation/     # ④ 调度：确定性主循环与配置

│   ├── config.py       # SimConfig 全量参数（唯一事实来源，pydantic 校验）

│   ├── engine.py       # SimulationEngine：对象引擎（AoS，逐个体循环）

│   ├── vecengine.py    # VecEngine：数组化高速引擎（SoA，规则等价，\~90× 提速）

│   └── tick.py         # TickStats：单 tick 的不可变统计快照

├── observatory/    # ⑤ 观测台（Stage 2 的实验系统）

│   ├── statistics.py   # 纯函数统计聚合 → GenerationStats

│   ├── observer.py     # EvolutionObserver：世代轴 + 时间轴双触发采样

│   ├── experiment.py   # ExperimentSpec/Run/Runner + 5 类实验矩阵 + 种子派生

│   └── \_\_main\_\_.py     # \`py -m observatory\` 无头 CLI

├── persistence/    # ⑥ 持久化

│   └── io.py           # 实验结果 JSON/CSV 落盘与回读 + survey 摘要

├── brain/          # ⑦ 预留：感知→决策（Stage 3 才实现，当前空包）

├── visualization/  # ⑧ 预留：pygame 可视化（当前空包）

├── main.py             # Stage 1 无头 CLI（单次模拟摘要）

├── tests/              # 151 个确定性测试

├── results/            # 实验输出（运行产物，不入 git）

└── pyproject.toml      # 打包/依赖/pytest 配置
```

**依赖方向是单向的，只允许上层依赖下层，禁止反向：**



```
core ← world ← evolution ← simulation ← main

&#x20;                     ↘ observatory ← persistence ← (CLI)
```

也就是说：`core` 不知道 `world` 的存在；`world` 不依赖 `evolution`；观测层（observatory）从不引入 "适应度" 概念。这条单向依赖是你理解 "改哪里会波及哪里" 的关键 —— 改下层会影响上层，改上层不影响下层。



***

## 3. 一个 tick 里发生了什么（顺序契约）

**这是整个项目最核心的代码语义，任何改动都不能打破这个顺序**（破坏 RNG 调用顺序 = 破坏可复现性 = 最高级别红线）。



```
每个 tick（SimulationEngine.\_advance\_one\_tick / VecEngine.\_advance\_one\_tick）：

&#x20; 1\) world.tick\_regrowth()        → 资源再生（全格 += regrowth\_rate，封顶 capacity）

&#x20; 2\) population.update(rng)       → 对每个存活个体（快照迭代）依次：

&#x20;       a. 代谢扣费        energy -= base\_metabolism × metabolism\_mult

&#x20;       b. 若 energy ≤ 0   本 tick 不再进食/移动（死亡判定交给引擎）

&#x20;       c. 进食            taken = env.consume\_resource(x, y, eat\_amount)

&#x20;                         energy = clamp(energy + taken × eat\_efficiency, max\_energy)

&#x20;       d. 移动            if rng.random() < move\_prob 且 energy ≥ move\_cost：

&#x20;                           random\_neighbor(8 邻格，torus 包装，避开不可通行格)

&#x20;                         energy -= move\_cost

&#x20;       e. 年龄 +1

&#x20;       f. 死亡判定        先饿死（energy ≤ 0）后老死（age ≥ life\_span）

&#x20;       g. 繁殖判定        存活 且 种群 < max\_count 且 energy ≥ repro\_fraction × max\_energy：

&#x20;                         子代 = 亲代基因 + 变异；energy 对半扣；generation+1；parent\_id=亲代

&#x20; 3\) 清理尸体（记录死因）→ 产出 TickStats → append 历史（有界模式下环形裁剪）
```

要点：



* **本 tick 新生的子代不参与本次行动**，下个 tick 才动。

* **子代出生在亲代所在格**（允许同格叠放；领地 / 碰撞规则留给未来生态位阶段）。

* 同一配置 + 同一种子 → 逐 tick 世界演化**完全相同**（确定性是硬保证）。



***

## 4. 双引擎设计（最容易困惑、也最关键的设计）

同一个模拟有两套实现，**规则等价、接口一致（duck-typed），可随时切换**：



|        | SimulationEngine（engine.py）  | VecEngine（vecengine.py）                                 |
| ------ | ---------------------------- | ------------------------------------------------------- |
| 数据布局   | AoS：`dict[id → Organism]` 对象 | SoA：定长 NumPy 数组（x/y/energy/genes/age/generation/parent） |
| 每 tick | 逐个体 Python 循环                | 全批量向量化（代谢 / 进食 / 移动 / 衰老 / 死亡 / 繁殖）                     |
| 速度     | \~150 tick/s                 | \~13,500 tick/s（实测～90× 提速）                              |
| 用途     | 单步精确参照、行为语义基准                | 长程实验（万代 / 百万 tick）默认首选                                  |

**为什么有两套？** 长程实验（如 10,000 代 ≈ 1,330 万 tick）用对象引擎要跑数小时，VecEngine 把整个种群变成 NumPy 数组批量运算，20 分钟内跑完。它们共享 `SimConfig`、`World`、`TickStats`，`observatory` 挂上去不用改一行代码。

**已知的刻意差异**（统计等价而非逐位一致，有测试覆盖）：



1. 同格多点取食：VecEngine 用 "均分封顶"（顺序无关），对象引擎是先到先得

2. 移动方向：VecEngine 均匀抽 8 方向，对象引擎逐次抽 (dx,dy) 直到非零

3. RNG 流整体不同（批量 vs 逐个体采样）→ 同种子不逐位复现，仅长期统计一致

**热路径铁律**：不得在 `VecEngine._step_population` 内引入逐个体 Python 循环或对象创建；观测采样走 `population.organisms()`（按需重建，约每百 tick 一次）。



***

## 5. 基因与性状：16 个基因，只有 4 个起作用

**Genome**（`core/genome.py`）：定长 16 个 float64 基因的一维数组，纯数据容器，只负责存储 / 复制 / 比较。

**Genetics**（`core/genetics.py`）：定义 "基因如何解码成性状"（TRAIT\_TABLE）以及变异 / 交叉纯函数。

基因值 ∈ \[0,1]，线性映射到性状区间：



| 基因位 | 性状               | 区间           | 影响                       |
| --- | ---------------- | ------------ | ------------------------ |
| 0   | move\_prob       | \[0, 1]      | 每 tick 移动概率（觅食 vs 定居的权衡） |
| 1   | metabolism\_mult | \[0.5, 2.0]  | 代谢倍率（省能耗 vs 能力上限）        |
| 2   | repro\_fraction  | \[0.25, 0.9] | 繁殖门槛比例（多生 vs 稳健）         |
| 3   | life\_span       | \[200, 4000] | 寿命 tick 数（短命早生代 vs 长寿晚育） |

其余 12 个基因是**中性漂移区**：不被解码使用，但仍参与变异 —— 为将来复杂性状保留进化底物。

**关键设计**：基因和性状的映射是 "结构契约"，放在 `core/genetics.py` 而不是配置里；替换基因表达方式 = 换掉这个模块的 `decode`，其余模块不感知。



***

## 6. 能量经济学 = 自然选择（"进化" 的定义位置）

**这个项目没有 fitness 函数。** "什么是最优个体" 不由代码定义，而由环境参数定义。

选择压完全由两条涌现规则构成（`evolution/selection.py`）：



1. **能攒够能量者繁殖**：energy ≥ repro\_fraction × max\_energy（门槛由基因决定）

2. **攒不够者被淘汰**（饿死）或 **寿命到限被淘汰**（老死）

这就是 "干预的杠杆"：你改环境参数（资源多少 / 再生快慢 / 世界大小），就改变了选择压力，群体就会朝不同方向演化。实验证明的收敛方向包括：低代谢省能、低繁殖门槛早繁殖、长寿扩大繁殖窗口、高移动概率觅食 —— 全部是选择压涌现的结果，没有一个被写死在代码里。



***

## 7. 配置系统：干预的第一入口

所有数值参数集中在 `simulation/config.py` 的 `SimConfig`（pydantic 校验）。**改参数 = 不写代码的干预**。



```
SimConfig

├── seed          # 随机种子；None = 系统熵（不可复现）

├── world         # width/height（默认 128×128）、wrap（torus 环形边界）

├── resources     # capacity/initial\_fill/regrowth\_rate/patchiness/patch\_count

├── organisms     # initial\_energy/max\_energy/base\_metabolism/move\_cost/eat\_amount/eat\_efficiency

├── genome        # gene\_count/mutation\_rate/mutation\_sigma/crossover\_rate

├── population    # initial\_count/max\_count

└── simulation    # ticks/stop\_on\_extinction/log\_interval/history\_limit
```

常用干预示例：



| 想观察       | 改什么                                   | 预期效应                          |
| --------- | ------------------------------------- | ----------------------------- |
| 食物变少迫使进化  | regrowth\_rate 0.02→0.005             | 代谢更低、更早繁殖（pressure\_low 已验证）  |
| 饥饿危机      | regrowth\_rate →0.001                 | 灭绝动力学（pressure\_critical 已验证） |
| 空间不公造就生态位 | initial\_fill 0.08/0.8，patchiness 0.8 | 稀疏 / 富饶 / 斑块，迁移 vs 定居         |
| 遗传多样性变化   | mutation\_rate 0.05→0.005/0.3         | 漂变 vs 变异供给                    |
| 结果可重复性    | 换 --seed                              | repeated\_seeds 方差分析          |
| 跑得快一点     | 世界调小（64×64）                           | tick 提速，但密度上升                 |

⚠️ 注意：`crossover_rate` 默认 0.5 是**死参数**—— 无性繁殖从未调用 `crossover`（有性繁殖留待 Stage 5），文档 / 论文里不要引用它。



***

## 8. 观测台与实验系统（Stage 2 的核心）

Stage 2 的目标是**证明自然选择正在发生**。三个组件：

**① statistics.py — 观测点口径（纯函数）**

每个观测点（`GenerationStats`，CSV 一行的字段）包含：种群数、平均年龄 / 能量、世代号分布、4 个性状的均值 / 标准差、基因组多样性（平均每位点 Simpson 杂合度 \[0,1]）、唯一基因型数。

**② observer.py — 采样触发器（EvolutionObserver）**

两个触发器保证时间轴与世代轴都连续：



1. 种群 `max_generation` 前进时采样（世代轴连续）

2. 距上次采样 ≥ tick\_interval tick 时兜底采样（覆盖世代停滞 / 灭绝）

出生 / 死亡率按窗口累计 ÷ span\_ticks，per-tick 单位，跨实验可比。

**③ experiment.py — 实验矩阵（ExperimentRunner）**



* 确定性：`seed = SeedSequence([base_seed, group_id, seed_index])` → 同命令行必然同结果

* 5 类实验：baseline（默认 1000 代长程）/resource\_pressure ×2 /resource\_distribution ×3 /mutation\_rate ×2 /repeated\_seeds ×3

* 停止条件：世代达标 ∨ 引擎结束（跑满 tick / 灭绝）∨ tick 硬上限（防失控）

* 结果摘要：起 / 中 / 末三点种群、多样性 drift、4 性状 start→mid→end drift、末态出生率 / 死亡率

**输出文件**（每个实验一个目录）：



* `manifest.json` — 实验元数据 + 完整配置 + 汇总

* `generations.csv` — 每代一行观测点（宽表）

* `generations.json` — CSV 的 JSON 镜像

* `__survey__.json` / `__survey__.md` — 批量汇总



***

## 9. Stage 路线：项目现在在哪，接下来去哪

项目采用严格的 Stage 系统，**代码写完 ≠ 阶段完成**，只有验收条件满足才能升级：



```
Stage 0  Architecture         基础架构（已验收）

Stage 1  Minimal Life         最小生命闭环（已验收）

Stage 2  Evolution Observatory 证明自然选择在发生（当前，毕业 READY 2026-09-06）

Stage 3  Adaptive Brain       引入感知/决策（Sensors → Brain → Action）

Stage 4  Ecological Evolution 生态演化（多资源/风险/生态位涌现）

Stage 5  Communication & Social Evolution  通信与社会（有代价的信号）

Stage 6  Open-ended Evolution 开放进化（基因结构本身可进化）
```

**Stage 2 毕业证据**（详见 `RESEARCH.md` R-1/R-2）：



* mutation 梯度 → 多样性终值单调（low 0.30 < normal 0.53 < high 0.70）

* 资源临界 → 群体灭绝；资源低压 → 种群收缩到 76

* 万代基线：10,000 代 19 分钟跑完，收敛后稳态 1 万代不漂移

* repeated seeds 3 次趋势一致

* 四性状收敛方向全部符合 "生存 / 繁殖成功" 解释（低代谢、低门槛、长寿、高移动）

**当前 Do Not Implement Yet**：Brain / Sensor（须等 Stage 3 验收交接）、LLM / 人工 fitness / 预设物种类别（全程禁止）、复杂 GUI。



***

## 10. 日常命令速查



```
\# 在 digital\_life/ 目录下执行

\# ① 单次模拟（Stage 1 入口，种子固定、确定性）

py -m main                        # 默认配置：seed=42，1000 tick

py -m main --ticks 5000 --seed 7  # 覆盖 tick 数与种子

\# ② 实验矩阵（Stage 2 入口，默认对象引擎；--vec 用数组化高速引擎）

py -m observatory --out results/s2                    # 默认：baseline 1000 代 + 全部对照 250 代

py -m observatory --names baseline --generations 1000 # 只跑 baseline（精简档）

py -m observatory --names baseline --generations 10000 --vec  # 完整 1 万代档（约 30–60 分钟）

py -m observatory --group resource\_pressure            # 只跑某组

py -m observatory --names pressure\_low --seed 7        # 换 base seed（跨 seed 复现）

\# ③ 测试

py -m pytest -q                  # 全量（151 个，确定性，\~2 分钟）

py -m pytest -q tests/test\_vecengine.py     # 只跑数组引擎（等价性/确定性/性能哨兵）

py -m pytest -q tests/test\_observatory.py   # 只跑观测台
```



***

## 11. 参与开发：改哪里、怎么改、红线

### 11.1 文档地图（一个事实一个来源）



| 文件                              | 职责                                 |
| ------------------------------- | ---------------------------------- |
| README.md                       | 面向人类的项目哲学摘要                        |
| AGENT.md                        | AI 导航：文档分工 / 双引擎 / 硬约定 / 红线        |
| MANUAL.md                       | 完整使用手册：模块地图 / 配置表 / 干预指南 / 测试 / 诊断 |
| RESEARCH.md                     | 研究记录：实验的 Observed/Expected、证据等级    |
| .ai/current-stage.md            | 当前 Stage 状态的唯一事实来源                 |
| AI\_DEVELOPMENT\_PROMPT.md      | 开发哲学、Stage 路线与验收标准                 |
| PROJECT\_MAINTENANCE\_PROMPT.md | 维护纪律（一致性 / 实验诚实 / Stage 评审）        |

### 11.2 想改什么，动哪一层



| 想达成的改动         | 应该改哪一层                                       | 关键注意                                   |
| -------------- | -------------------------------------------- | -------------------------------------- |
| 数值标定           | simulation/config.py                         | 只改默认值，别动字段名                            |
| 加一个性状          | core/genetics.py（TRAIT\_TABLE）+ config       | 手册 6.5 节 5 步法                          |
| 换变异策略          | core/genetics.mutate 或 evolution/mutation.py | 保持纯函数签名                                |
| 换繁殖方式（有性）      | evolution/reproduction.py                    | 保留 Offspring 语义                        |
| 改行为决策          | core/organism.py::Organism.step              | 保持 RNG 调用次数不变量                         |
| 加环境压力（毒区 / 季节） | world/ + resource.py/tick\_regrowth          | 保持 tick\_regrowth 先于 population.update |
| 加真・感知 / 决策     | brain/（当前空包）                                 | 通过 EnvironmentView 协议接入                |
| 加新实验类别         | observatory/experiment.py + \_GROUP\_IDS     | 新组须登记 int id                           |
| 加新的观测指标        | observatory/statistics.py                    | 纯函数聚合，JSON 安全                          |

### 11.3 六条铁律（改代码前必读）



1. **不要改变每 tick 的 RNG 调用顺序 / 次数**（否则全历史复现失效）

2. 随机性必须来自显式传入的 `rng`，禁止 `random`/`np.random` 全局

3. 新规则先写成纯函数（入参出参，无全局状态），再让带状态对象去调用

4. 不向 `core` 层引入对 `world/simulation` 的依赖（依赖方向单向）

5. 每次改动先跑对应测试，大改动全量回归

6. 不要为 "看起来智能" 提前加 LLM / 神经网络 /fitness—— 先证明涌现

### 11.4 已知坑（改项目前必读）



* `history_limit=0`（默认）时引擎保留全部 tick 历史，百万 tick 级会涨到数 GB—— 长程实验必须开 `history_limit>0`（实验 runner 已自动设为 4096）

* `pyproject.toml` 的 packages 列表没有 `observatory` 与 `persistence` 是历史遗漏；`py -m observatory` 在仓库根运行时无需安装即可用，将来 `pip install` 打包需补上

* 世代推进速率实测～850–1,500 tick / 代：1,000 代（对象引擎）≈ 40–45 分钟，长程实验请用 `--vec`

* 测试都是确定性的（固定种子构造），任何一次红 = 你破坏了不变量，不是 flaky



***

## 12. 一个快速理解全貌的阅读顺序

如果你只有 20 分钟，按这个顺序读：



1. 本文件（10 分钟）→ 建立全局图景

2. `README.md`（2 分钟）→ 项目哲学

3. `MANUAL.md` 第 3 节（5 分钟）→ "你不知道就改不对的三件事"（tick 顺序契约 / 基因解码 / 能量经济学）

4. `simulation/engine.py`（3 分钟）→ 看真实的 tick 驱动代码

之后遇到具体问题时再按文档地图查对应文件。