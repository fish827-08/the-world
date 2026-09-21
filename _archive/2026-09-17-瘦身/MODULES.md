# 模块实现说明（MODULES）

> 本文档是**各模块实现细节的唯一事实来源**：每个文件、每个类、每个方法
> 的通俗解释都记录在这里。架构决策（为什么这样设计）见 PROJECT-DESCRIPTION.md，
> 进度与检查清单见 PROGRESS.md。

---

## 模块一：球面世界（SphereWorld 世界模块）

### 概览

| 文件 | 角色 | 通俗一句 |
|------|------|---------|
| `world/sphere_world.py` | 土地 | 世界长什么样：格子怎么排、哪格大哪格小、每个格子和谁相邻 |
| `world/light_and_temperature.py` | 天气 | 阳光和温度：什么时候白天、哪里冷哪里热 |
| `world/resource_field.py` | 食物 | 吃的：哪格有多少食物、会慢慢长回来、冷的地方长得慢 |

依赖链：`resource_field → light_and_temperature → sphere_world`（食物要看天气，天气要在地上的格子里）。

---

### 文件 1：`world/sphere_world.py` —— 土地（网格拓扑）

**核心概念**
- 把 60 行纬度 × 120 列经度网格"糊"成一个球面。纬度越高的格子越靠近两极，格子**越小**（面积 ≈ cos 纬度）。
- 两个极点：上极（row 0）和下极（row 59）。极点处经度坍缩——整个极点只有"一格"（所有经度槽位都算同一个位置）。
- 每个格子有一个平铺编号 `flat`（0~7199），`flat = row×120 + col`，方便引擎一锅端处理。

**Class `SphereWorld`**

实例属性（全部字段说明）：
| 字段 | 通俗解释 |
|------|---------|
| `rows` | 行数（纬度方向），默认 60 |
| `cols` | 列数（经度方向），默认 120 |
| `_n` | 总格子数 = 60×120 = 7200 |
| `_lat` | 每行中心纬度（弧度），预计算好的表 |
| `_area` | 每行面积权重（近赤道≈1，极点≈0.026），预计算好的表 |
| `_pole_top` | 上极点行号 = 0 |
| `_pole_bottom` | 下极点行号 = 59 |
| `_nb_table` | 邻居快查表：(7200, 8) 的 int64 数组，第 f 行就是格 f 的 8 个邻居（构造时一次性算好） |
| `_pole_nb` | 极点邻居表：(2, 120) 数组，上/下极点各一行——相邻纬度带整行的 120 个格子 |

| 方法 | 通俗解释 |
|------|---------|
| `__init__(rows=60, cols=120)` | 开工：把格子排好，预计算出纬度和面积两张表，记住两个极点在哪，并**顺手把全部格子的邻居关系写进快查表**（构造后想查邻居都是读表，不再现算） |
| `n_cells` | 问总共有多少格子（7200） |
| `flat_to_rc(flat)` | 把格子编号拆成（行, 列）。例：编号 125 → (1, 5) |
| `rc_to_flat(row, col)` | 把（行, 列）合回格子编号。绕圈：col=-1 会变成 119；若在极点行，无论 col 是多少都归并到 col 0（因为极点只有一格） |
| `latitude_of(row)` | 问某行的纬度（弧度，-90°~+90°） |
| `cell_area(flat)` | 问格子多大（面积权重）：赤道附近≈1，极点≈0.026（只有赤道的 1/38） |
| `is_pole(flat)` | 问这格是不是极点格（上极/下极整行都是 True） |
| `neighbors(flat)` | 问相邻格子：普通格 8 个（上下左右+对角，经度能绕圈）；极点格有 120 个（能一步走到相邻纬度带任意经度）。**查表返回**：普通格直接 `_nb_table[f]` 取一行，极点格从 `_pole_nb` 拿整行——比原来每次现算坐标快约一个量级 |
| `__repr__` | 打印摘要，如 "SphereWorld(rows=60, cols=120, cells=7200)" |

---

### 文件 2：`world/light_and_temperature.py` —— 天气（光照温度）

**核心概念**
- 太阳固定不动，世界自己转 → 转到太阳那边的格子是白天，转过去的这边是晚上。昼夜是"经度方向"扫过去的。
- 光照 = 纬度顶 × 昼夜顶：赤道正午光照 1（最亮），极地永远接近 0（天然极夜/极昼）。
- 温度 = 纬度基温 + 光照×小温差。极地永远冷；夜晚只比白天**冷一点点**（温差刻意设小）。
- 生物会跟着温度"行动变快/变慢"（活性因子），低温时移动、发育都更慢（引擎用）。

**Class `LightAndTemperature`**

实例属性（全部字段说明）：
| 字段 | 通俗解释 |
|------|---------|
| `world` | 属于哪个网格（借用它的纬度和经度换算） |
| `rotation_period` | 转一圈要多少个 tick（默认 2400 = 白天夜晚各半） |
| `tilt_rad` | 黄道倾角。本阶段固定 0（没四季），留作未来扩展 |
| `lat_base_ref` | 纬度光照敏感度（越大极地越暗越冷） |
| `t_equator` | 赤道基温（默认 30） |
| `t_pole` | 极点基温（默认 -20，恒冷） |
| `day_boost` | 昼夜温差幅度（默认 6，正午比深夜高 6 度） |
| `_daily_cos` | 预计算的"经度→光照"对照表（不用每次现算） |

| 方法 | 通俗解释 |
|------|---------|
| `__init__(world, rotation_period=2400, ...)` | 开工：记住网格和各档温度参数，预计算昼夜对照表 |
| `sun_longitude(tick)` | 问这个时间点太阳照在哪个经度（弧度 0~2π） |
| `illumination(flat, tick)` | 问格子现在多亮（0~1）：赤道正午=1，晚上=0，极地≈0 |
| `base_temperature(flat)` | 问格子"只看纬度"的基本温度（不看昼夜） |
| `temperature(flat, tick)` | 问格子现在实际温度 = 基温 + 光照×6（所以晚上只冷一点） |
| `activity_factor(flat, tick)` | 问生物在这里"活跃度"（0~1）：温度≥10° 全速行动，≤-10° 基本冻住，中间慢慢过渡 |
| `__repr__` | 打印摘要 |

---

### 文件 3：`world/resource_field.py` —— 食物（资源场）

**核心概念**
- 每格食物有上限（容量），随面积：赤道格子大 → 粮仓大（≈40）；极点格子小 → 粮仓小（≈1）。所以极地"养不活多少生物"。
- 食物随时间慢慢长回来（再生），但**冷的地方长得慢**：热带正午最快，极地/深夜几乎不恢复。
- 生物吃掉食物，同格多只时**均分**，绝不欠账。

**Class `ResourceField`**

实例属性（全部字段说明）：
| 字段 | 通俗解释 |
|------|---------|
| `world` | 属于哪个网格（提供容量缩放所需的面积） |
| `lt` | 光照温度场（再生时要查温度） |
| `capacity_per_area` | 每单位面积的食物上限（默认 40） |
| `regrowth_rate` | 基准恢复速度：温度合适时每 tick 每格涨多少食物 |
| `temp_sensitivity` | 再生对温度的依赖（0=不care温度，越大越在意） |
| `_grid` | 每格当前食物存量（一条 7200 长的一维数组，按 flat 取） |
| `_capacity` | 每格食物上限（构造时按面积算好，之后只读） |

| 方法 | 通俗解释 |
|------|---------|
| `__init__(world, lt, ...)` | 开工：按面积算好每格上限，每格先填一半食物 |
| `capacity_at(flat)` | 问：这格粮仓多大？ |
| `amount_at(flat)` | 问：这格还剩多少食物？ |
| `consume(flat, amount)` | 一只生物吃这格：吃想要的量，不够就吃到空，不欠账 |
| `consume_many(flats, amount)` | 一串生物同时吃（批量版）。同格多只均分存量，绝不吃出负数 |
| `regrow(tick)` | 全世界食物长一点：每格 += 恢复量，顶到容量为止；冷的地方恢复量打折 |
| `_regrowth_amount(tick)` | （内部）算出本 tick 每格该长多少 = 基准率 × 温度因子 |
| `snapshot()` | 拍一张 60×120 的食物分布图（拷贝，不改原数据，给可视化） |
| `__repr__` | 打印摘要 |

**温度因子规则**（`_regrowth_amount` 内）：因子 = (温度+20)/20，夹在 [0,1]。
温度 ≥0° → 满速；-20° → 停摆；中间线性过渡。

---

### 模块一的"一句话总结"

> 土地（SphereWorld）决定"哪格多大、邻居是谁"；天气（LightAndTemperature）
> 决定"哪里亮、哪里冷"；食物（ResourceField）决定"哪格有多少吃的"。
> 三者合起来：赤道既是格子大、又暖和、食物多、长得快；
> 极地格子小、又冷、食物少、长得慢——生态梯度**不靠写死规则，靠环境涌现**。

### 模块一测试点（照此验证）

1. `cell_area(极点) ≈ 0.026`，`cell_area(赤道) ≈ 1`（面积梯度）
2. `rc_to_flat(0, 任意col) → 0`（极点坍缩）
3. `neighbors(普通格)` 返回 8 个；`neighbors(极点格)` 返回 120 个
4. `illumination(赤道正午)=1`、`illumination(赤道深夜)=0`
5. `temperature(深夜) = temperature(正午) - 6`（小温差）
6. `consume_many` 同格多只后存量 ≥ 0（不欠账）

### 基础时间与资源概念（和主人对齐过的理解）

**tick 是什么**
- tick 是数字世界的最小时间单位，打个比方就是现实世界的"一秒"。
- "这一秒里数字世界发生了什么变化"，就是引擎跑一次 `step()` 执行的整套流程：
  资源再生 → （光合）代谢转化 → 维持消耗 →（进食）→（移动）→ 年龄 → 死亡 → 繁殖。
- 一切规则、速度、周期都以 tick 为刻度，**它是抽象单位，不映射现实秒数**。

**世界自转一圈要多久**
- 默认 `rotation_period = 2400 tick`：白天 1200 tick + 黑夜 1200 tick。
- 太阳子午线每 tick 移动 360°÷2400 = 0.15° 经度，在球面上扫出一个昼夜。
- 注意：默认 `simulation.ticks = 1000`，跑一次默认模拟连"半天"都不到。

**整个世界食物有限**
- 每格容量 = 面积权重 × `capacity_per_area(40)`。赤道格容量≈40，极点格≈1.0。
- 全球总容量 ≈ 4π × 40 ≈ 500 单位，初始填一半 ≈ 250。总量有限，不会无限。

**极地食物不会无限铺满**
- 双重保险：① 再生用 `np.minimum(容量, 存量+生长量)` 严格封顶，涨到容量就停；
  ② 再生被低温压制，极点恒温 ≈ -18.6°C → 再生因子 ≈ 0.07，每 tick 只长 ≈0.03。
- 极端情形只是"极地那一点点容量慢慢长满 1.0 就停"，不会爆发。

---

## 模块二：球面世界引擎（Simulation 引擎层）

### 概览

| 文件 | 角色 | 通俗一句 |
|------|------|---------|
| `simulation/config.py` | 总参数本 | 所有数值参数集中一处，改规则=改配置，不改代码 |
| `simulation/tick.py` | 相机 | 每 tick 结束拍一张世界快照（多少生物、死了几个、还剩多少食物） |
| `simulation/sphere_engine.py` | 导演 | 每个 tick 指挥所有生物：进食、消化、移动、衰老、死亡、繁殖 |
| `core/lifecycle.py` | 字典 | 死亡原因枚举（饿死/老死），引擎统计死因用 |

依赖链：`sphere_engine → light_and_temperature / resource_field / sphere_world`（引擎站在模块一上面）、`sphere_engine → tick / core.lifecycle`（引擎产出快照和死因统计）。

### 文件 1：`simulation/config.py` —— 总参数本

**核心概念**
- 全部数值参数（网格/光照/资源/生物/基因/种群/运行时长）都用 Python 内置 dataclass 存放。
- "配置即复现"：同一份配置 + 同一种子 → 完全相同的模拟。
- 为什么不用 pydantic：本项目要接 Rust 热核，尽量减少外部依赖。

**Class `SimConfig`**（顶层配置，嵌套各子配置）
| 子配置 | 通俗解释 |
|--------|---------|
| `WorldConfig` | 球面网格几行几列（默认 60×120） |
| `LightConfig` | 自转周期、赤道/极点温度、昼夜温差 |
| `ResourceConfig` | 食物容量、再生速度、初始填充 |
| `OrganismConfig` | 能量收支、进食速率、光合产能、恒温维持费 |
| `GenomeConfig` | 基因链长度、变异率、变异幅度 |
| `PopulationConfig` | 初始个体数、种群硬上限 |
| `SimulationConfig` | 跑多少 tick、种群归零是否提前停 |

方法：`to_dict()` 导出 / `from_dict()` 重建 / `fingerprint()` 比对两份配置是否一致。

### 文件 2：`simulation/tick.py` —— 相机（TickStats）

| 字段 | 通俗解释 |
|------|---------|
| `tick` | 第几个 tick |
| `population` | 当前存活生物数 |
| `born` / `died` | 本 tick 出生 / 死亡数 |
| `deaths_by_cause` | 按死因统计（饿死 / 老死） |
| `total_energy` | 全体生物可用能量之和 |
| `total_resource` | 全世界剩余食物总量 |

`to_dict()`：转成 JSON 安全的普通字典（死因枚举转字符串），供存档与逐 tick 等价对比。

### 文件 3：`simulation/sphere_engine.py` —— 导演（SphereEngine）

**核心概念**
- 所有个体用"大数据表"（NumPy 数组）同时推进：位置、能量、胃、基因、年龄各占一列，一个 tick 全部算完，比逐个体循环快几十倍。
- 位置用 `flat`（格子编号）表示，经度环绕、极点坍缩全由模块一内部处理，引擎只管"搬格子"。
- 每个 tick 内部的固定流程（0~9）：
  - 0) 温度响应：冷血（g9=0）随环境；恒温（g9=1）低温不减速但多付维持费
  - 1) 光合收入（g8）：按所在格光照获得少量能量（产能刻意小）
  - 2) 代谢转化（g1）：胃里的食物 → 能量。同样食物给同样能量，温度只影响速度
  - 3) 维持消耗：基础价 × 年龄需求倍率（幼体×1.6 在长身体 / 成年×1 / 老年×1.4 器官退化）+ 恒温个体额外费
  - 4) 进食（g4/g5/g10）：吃进胃（饱食度封顶）；有觅食基因的自己格不够吃邻格
  - 5) 移动（g0 决定走不走，g6/温度决定费多少能，g13 决定往同伴多还是少的邻格走）
  - 6) 年龄 +1
  - 7) 死亡判定：饿死（能量≤0）→ 老死（年龄≥寿命，g3）
  - 8) 繁殖（g2 阈值、g7 投入比例、g12 冷却间隔、**成熟年龄**（未长大不能生））
  - 9) 清理尸体

**Class `SphereEngine` 常用接口**
| 方法/属性 | 通俗解释 |
|-----------|---------|
| `step()` | 推进 1 tick，返回 TickStats 快照 |
| `run(ticks)` | 连续跑到指定 tick 或自然结束，返回历史快照列表 |
| `tick` | 当前走到第几个 tick |
| `alive_count()` | 当前有多少生物活着 |
| `population_view()` | 常驻只读统计（数量/最大世代/总能量/胃里食物总量） |
| `extinct` / `finished` | 种群灭绝 / 运行结束状态 |

### 文件 4：`core/lifecycle.py` —— 死亡原因字典（DeathCause）

**核心概念**
- 就一个枚举 `DeathCause`，两个取值：`STARVATION`（饿死：能量 ≤ 0）、`OLD_AGE`（老死：年龄 ≥ 寿命，寿命由基因 g3 决定）。
- 为什么单独成文件：死因标记是"生命规则层"的契约，比引擎更底层；引擎只负责往 `TickStats.deaths_by_cause` 账本里填它（`Counter[DeathCause]`）。单一职责，引擎不自己发明字符串。
- 与旧项目 digital_life 的关系：旧版的 Lifecycle 类还管年龄推进、存活状态等，新球面版这些逻辑已向量化进 `SphereEngine` 的数组操作，这里只留下"死因枚举"这份契约。

### 基因表（基因位 → 性状，定义在引擎热路径）

基因链默认 16 位定长（与旧项目兼容），每个基因取值 [0,1]，变异时整体震荡。当前已定义 g0~g13（g14~g15 预留未来扩展）：

| 位 | 性状 | 映射公式（g 表示基因值） | 通俗解释 |
|----|------|--------------------------|---------|
| g0 | 移动概率 | 概率 = g | 每天有多少几率迈一步（自己决定） |
| g1 | 代谢倍率 | 0.5 + g×1.5 | 消化/维持的倍率：高 = 快吃快耗 |
| g2 | 繁殖阈值 | (0.25 + g×0.65) × 能量上限 | 攒多少能量才肯生 |
| g3 | 寿命 | 一昼夜 × (1 + g×7) | 活多少个 tick 老死；**最短整整一昼夜(2400)、最长八昼夜**，保证生物经历得起完整昼夜 |
| g4 | 进食量倍率 | 0.5 + g | 每 tick 能吃多少（抢食能力） |
| g5 | 饱食度上限 | (0.5 + g×1.5) × 基础胃容量 | 胃多大：能存多少待消化食物 |
| g6 | 移动能耗倍率 | 0.5 + g | 走路费不费劲（在"冷更费能"之上） |
| g7 | 繁殖投入比例 | 0.3 + g×0.4 | 把多少比例能量/胃粮分给子代 |
| g8 | 光合利用 | 收入 = 光照 × g × 0.1 | 白天按光照晒出少量能量（极地≈0） |
| g9 | 恒温指数 | 抗寒度 = g | 1=恒温低温不减速（月付维持费）；0=冷血 |
| g10 | 邻格觅食 | g ≥ 0.5 能吃邻格 | 自己格不够吃时补吃一格外邻格 |
| g11 | 温度偏好 | 偏好温度 = -10 + g×50 | 冷血个体只在自己偏好的温度附近才满速，偏离越远消化越慢（恒温者免疫） |
| g12 | 繁殖冷却 | 间隔 = g × 60 tick | 生完一胎要休息这么久才能再生（冷却没到攒再多能量也不生） |
| g13 | 群居性 | g ≥ 0.5 聚群 / g < 0.5 避群 | 走路时优先挑同伴多的邻格（聚群）还是最冷清的邻格（避群），g≈0.5 随机 |

> 对照参考：The Bibites 有 28 个基因（食性、脂肪储能、器官分配等）。
> 经评估（见 PROGRESS.md 决策表）：**已接入** 光合（g8）、恒温（g9）、邻格觅食（g10）、
> 温度偏好（g11）、繁殖冷却（g12）、群居性（g13）——全部挂靠既有机制，无需新子系统；
> **暂缓**：食性/捕食（需要"尸体+伤害"新机制）、脂肪储能（需要存储池）、
> 感知/交流（独立感知子系统，等引擎稳定后再评）、器官分配（器官系统）。

### 生命周期年龄（成熟 / 能量需求随年龄）

生物不是一出生就能生，能量需求也不是恒定不变的：

| 阶段 | 判定 | 维持费倍率 | 能否繁衍 |
|------|------|-----------|---------|
| 幼体 | 年龄 < 成熟年龄 | ×1.6（在长身体，吃得多耗得多） | ❌ |
| 成年 | 成熟年龄 ≤ 年龄 < 老年年龄 | ×1 | ✅ |
| 老年 | 年龄 ≥ 老年年龄 | ×1.4（器官退化，维持费上升） | ✅（但耗时难攒够能量） |

- **成熟年龄 = 寿命 × `maturity_fraction`（默认 15%）**；**老年年龄 = 寿命 × `senile_fraction`（默认 75%）**。
- 寿命由基因 g3 决定（**一昼夜 × (1 + g3×7)**，最短 1 昼夜、最长 8 昼夜）→ **寿命长的物种成熟和老化都更晚，自然形成"速生速死"与"晚熟长寿"两种生存策略**；且所有生物至少活满一昼夜，能对昼夜节奏做出反应。
- 防止的现象：新生命一出生就能疯狂繁衍，让种群在前几代就爆发。

### 扩展约定（为未来预留的接缝）

系统成熟后要加"新基因 / 世界地形 / 其他因素"时，按下面的接缝扩展，不需要动整体结构：

**加一个生物基因**（基因链 16 位，g14~g15 预留）
1. 占用一个空位（当前是 g14、g15；如不够再把 `GenomeConfig.gene_count` 调大，变异量会跟着位宽走）；
2. 把"位 → 行为"映射写进 `sphere_engine.py` 热路径里（现有基因都这样，见文件头注释）；
3. 新基因必须**挂靠既有机制**的某一步（0~9 流程编号就是现成的缝：温度响应/光合/代谢/维持/进食/移动/衰老/死亡/繁殖/清理），不新开子系统；
4. 在本基因表加一行 + 在 `tests/test_engine_lifecycle.py` 加一条机制单测。

**加世界地形/环境因素**（如海拔、水系、季节）
1. 世界层（`SphereWorld`）管"格子形状"，新增地形只在它上面加维度（如海拔高度表），邻居/面积逻辑不变；
2. `LightAndTemperature` / `ResourceField` 是"环境提供者"：新因素只要实现成"输入棋盘状态 → 输出光照/温度/食物容量或再生"，引擎不必改——改为给它们加参数（`LightConfig`/`ResourceConfig`）；
3. 接入后照旧在引擎 step 流程里的 0/2/3 步引用新环境量。

**加时间/阶段因素**（如季节、日夜长短变化）
- 引擎流程 0~9 编号即插桩位；新增阶段要么插进某一步、要么加上编号 11+，同时记得同步 `TickStats` 快照与观测端，避免"代码更新了、统计没跟上"。

### 测试与实验文件（模块一、二配套）

| 文件 | 角色 | 通俗一句 |
|------|------|---------|
| `tests/conftest.py` | 路径接线 | 让 pytest 里的 `import simulation/world/core` 能找到项目根，不用装包 |
| `tests/test_engine_lifecycle.py` | 引擎机制单测 | 饿死/老死/成熟门槛/年龄能耗/繁殖冷却等一条条规则单独验证 |
| `tests/test_sphere_world.py` | 世界模块回归 | 预计算邻居表 vs 逐格参照算法在全部格子逐位一致（含极点/环绕特殊路径） |
| `tests/test_sim_core.py` | Rust 对拍（函数级） | 同一份输入 Python/Rust 各跑一遍，逐位比较 |
| `tests/test_sim_core_engine.py` | Rust 对拍（引擎级） | 同种子两个引擎（开关开/关）逐 tick TickStats 与内部数组逐位相等 |
| `experiments/quick_check_age.py` | 快速长程体检 | 4000+ tick 跑一遍：看种群是否灭绝、世代是否推进、寿命基因多样性是否保留 |
| `experiments/benchmark_sim_core.py` | 性能基准 | 同种子双引擎跑 N tick 计时 + 位级一致性校验，回答"Rust 到底快多少" |

## 模块三：Rust 加速核（Sim-core）

> 定位：**不改任何玩法**，只把引擎里"真正发热的批量数值运算"下沉到 Rust
> （PyO3 扩展 `sim_core`），换一个更快的算盘。行为必须与纯 Python 引擎**逐位一致**。

### 概览

| 项 | 结论 |
|----|------|
| 产物 | `sim_core/`（maturin 项目，本地扩展装进 `.venv`） |
| 已移植 | 3.2 `regrow`（资源场再生）、3.3 `step_vectors`（种群数值管线 stage1/stage2）、3.5 `consume_many`（资源场批量消耗） |
| 开关 | `SimulationConfig.use_sim_core`（默认 **False**，保持原 Python 路径） |
| 对拍保证 | 函数级（同一份输入 Python/Rust 各跑一遍逐位比较）+ 引擎级（同种子逐 tick TickStats 相等） |

### 文件清单（整个 `sim_core/` 目录）

| 文件 | 角色 | 通俗一句 |
|------|------|---------|
| `Cargo.toml` | 工程清单 | 告诉 Rust 编译器这就是一个 pyo3 扩展（pyo3/numpy 依赖、cdylib 产物） |
| `pyproject.toml` | 打包清单 | 定义 `.venv` 里装的是名为 sim-core 的包，构建走 maturin |
| `src/lib.rs` | 胶水 | Rust ⇄ Python 的接口层：校验数组长度/索引越界，把 numpy 数组切给纯 Rust 函数 |
| `src/regrow.rs` | 数值核（3.2） | 食物再生：每格按温度涨一点，涨到容量为止 |
| `src/step_vectors.rs` | 数值核（3.3） | 种群管线：stage1 光合/代谢/维持、stage2 移动扣费/衰老/死亡/冷却/繁殖候选 |
| `src/consume.rs` | 数值核（3.5） | 进食结算：一批生物同时吃，同格均分存量、绝不欠账 |

### 文件 1：`sim_core/src/regrow.rs` —— 资源场再生（3.2）

- 语义与 `ResourceField.regrow` 逐位等价：`factor = clamp((温度+20)/20,0,1)^sens`，`存量 = min(容量, 存量 + 再生率×factor)`。
- `temp_sensitivity==1.0`（默认）跳过 powf → 可逐位相等断言；非 1 用 1e-9 容差。
- 纯 f64 四则 + min，无超越函数；自带 3 个单元测试（容量封顶 / 寒冷停摆 / 敏感度缩放）。

### 文件 2：`sim_core/src/step_vectors.rs` —— 种群数值管线（3.3）

**为什么分两段（stage1 / stage2）**
- 引擎一个 tick 内部步骤 1~3（光合/代谢/维持）与 5~8（移动扣费/年龄/死亡/冷却/繁殖候选）全是**确定性数值**；
- 中间夹着的"进食（步骤 4）、移动抽样、觅食目标选择、变异"依赖 RNG，且必须按 Python 原顺序消费随机数（同种子同结果）；
- 所以拆成两个 Rust 函数，RNG 缺口留给 Python：`stage1` 在进食前调用，`stage2` 在进食 + RNG 抽签后调用。

| 函数 | 对应引擎步骤 | 就地更新 | 输出 |
|------|------------|---------|------|
| `step_vectors_stage1` | 1 光合 / 2 代谢 / 3 维持 | energy、stomach | — |
| `step_vectors_stage2` | 5 移动扣费 / 6 年龄+1 / 7 死亡 / 8 冷却+繁殖候选 | energy、age、cooldown | out_moved/out_starved/out_expired/out_repro 四个掩码 |

- 数值纪律：只用 f64 四则 + min/max，**禁 exp/pow/sqrt** → 与 numpy 逐位一致。
- 边界：（a）年龄用"推进前"判成熟、用"推进后"判老死（与引擎捕获时机一致）；（b）移动"付得起才走"，扣费在 Rust 内完成（Python 路径在目标选择后扣，同一批个体、同样金额）。

### 文件 3：`sim_core/src/consume.rs` —— 进食结算（3.5）

- 语义与 `ResourceField.consume_many` 逐位等价（引擎第 4 步"进食"的双路径之一）：
  - 先数每格几只同时吃（整数计数）；
  - 每只"实吃量" = min(该格存量, 想吃的量 × 该格只数) ÷ 该格只数 → **同格多只均分**；
  - 按原数组顺序逐只从存量里扣（与 numpy `np.subtract.at` 的 unbuffered 累减**同序**），总量绝不为负；
- 与 numpy 逐位一致的三个关键细节：
  1. `min` 用 `a < b ? a : b`（`np.minimum` 的"相等取 b"语义，含 ±0.0）；
  2. `÷ 只数` 先把只数 `as f64` 再除（numpy 的 float64/int64 提升路径相同）；
  3. **先全部算完再逐只扣**（先读原始存量快照，与 `avail = grid[flats]` 一次性快照一致）。
- 自带 6 个单元测试（单只吃饱/存量不足/同格均分/同格不同量/存量归零/吃不完）。

### 文件 4：`sim_core/src/lib.rs` —— 绑定层（胶水）

- 三个 `#[pyfunction]`，各自先做防御性校验再下沉：
  - `regrow`：校验（grid, capacity, temperature）长度一致；
  - `consume_many`：校验 lengths 一致，且**所有 flat 索引落在 [0, n_cells) 内**（越界直接报 ValueError，绝不进索引运算——防 Rust slice 越界 panic/UB）；
  - `step_vectors_stage1/2`：校验各数组长度与 genes 行数一致、genes 需 C 连续。
- 全部就地更新；`#[pymodule] fn sim_core` 统一注册，Python 侧 `import sim_core` 即拿到这 4 个函数。

### 引擎接入（use_sim_core 开关）

`SphereEngine` 双路径：
- `use_sim_core=False`（默认）：原有 Python 数值代码原样跑；
- `use_sim_core=True`：资源再生走 `regrow`，步骤 1~3 调 `stage1`、步骤 5~8 调 `stage2`、步骤 4 进食（含邻格觅食的补吃）走 `consume_many`；所有 RNG 决策（进食对象选择、移动抽样、觅食目标、变异）仍走 Python，**RNG 消费顺序不变**。

开启方式：

```python
from simulation.config import SimConfig
cfg = SimConfig(seed=42)
cfg.simulation.use_sim_core = True   # 需要先构建：sim_core/ 下运行 .venv\Scripts\python -m maturin develop
```

### 对拍测试

| 测试文件 | 覆盖 |
|---------|------|
| `tests/test_sim_core.py` | regrow 逐位/容差对拍（3 例）；step_vectors 函数级 bitwise 对拍（8 例）+ 边界与长度校验（3 例）；consume_many 函数级 bitwise 对拍（4 个随机场景 + 4 个边界 + 1 例输入校验 = 9 例） |
| `tests/test_sim_core_engine.py` | 引擎级：3 个种子逐 tick TickStats 相等 + 最终内部数组逐位相等 + 资源网格逐位相等（4 例） |

### 性能基准（3.4 结论 → 邻居表优化后）

`experiments/benchmark_sim_core.py`：同种子双引擎跑 N tick 对比耗时并做位级一致性校验。

**第一轮（Rust 下沉，邻居未优化，2000 个体 × 300 tick，seed=42）**：rust 3.043s vs python 3.078s = **1.01x**，行为位级一致（REGRESSION CHECK PASS）。复跑确认 python 3.030s vs rust 3.049s = **0.99x**。
结论：已下沉数值段无净收益 —— 瓶颈在 per-individual 邻居循环（`SphereWorld.neighbors()` 被逐个体调用 ~19 万次、占 ~77% 时间）。

**第二轮（预计算邻居表，同一基准参数）**：python 3.030s → **0.615s**（2.049 ms/tick）、rust 3.049s → **0.569s**（1.896 ms/tick）= **约 5x 提速**；双引擎位级一致，REGRESSION CHECK PASS。
方法：`SphereWorld` 构造时一次性算好全网格邻居表 `_nb_table`(7200,8) + `_pole_nb`(2,120)，`neighbors()` 改为读表返回，消除循环里每次 flat_to_rc/clip/rc_to_flat 的 numpy 反复开销；纯 Python 改动、零 Rust 依赖、行为零变化（`tests/test_sphere_world.py` 全网格逐位对拍兜底）。
当前瓶颈：Rust 相对 Python 仅 1.08x——剩余大头是 Python 侧 RNG 消费与逐 tick 开销，数值段/邻居段已足够便宜。

---

## 模块四：观察台适配 + 快照桥（observatory / persistence）

> 定位：**观察**（离线统计落盘）+ **直播**（实时快照推流）两件事。
> 统计口径 = 14 基因位 + 3 派生 trait（基因 + 派生，人确认过的口径）；
> 快照推送 = 每 100 tick（架构定稿约定，broker 节拍可配）。

### 概览

| 文件 | 角色 | 通俗一句 |
|------|------|---------|
| `observatory/traits.py` | 统计口径 | 把基因串翻译成"看得懂的表现型"：14 个基因位 + 3 个派生指标（寿命/代谢倍率/成熟年龄） |
| `observatory/statistics.py` | 聚合 | 从引擎数组直接算一个"观测点"：trait 均值/标准差、基因组多样性、世代分布 |
| `observatory/observer.py` | 采样器 | 以"世代"为轴拍照：新世代出现拍一张，长期不出新世代按节拍兜底拍 |
| `observatory/experiment.py` | 调度 | 确定性实验：规格→派生种子→跑引擎→汇总（ExperimentsRunner + 5 类预置实验） |
| `observatory/__main__.py` | CLI | `py -m observatory`：命令行跑实验矩阵，结果落盘 results/ |
| `persistence/io.py` | 落盘 | 每个实验一个目录：manifest.json + generations.csv/json + 批量 survey |
| `observatory/broker.py` | 快照桥 | 每 N tick 采一份世界快照，WebSocket 推给前端（含个体明细） |

依赖链：`statistics → traits`；`observer → statistics`；`experiment → observer → engine`；
`broker → engine`（独立于实验管线，可单独 `py -m observatory.broker` 跑直播）。

### 文件 1：`observatory/traits.py` —— trait 表（统计口径）

- `GENE_TRAIT_NAMES`：14 个基因位的语义名（g0 移动概率 … g13 群居性，与引擎 `_genes` 列一一对应）。
- `_DERIVED_TRAITS`：3 个派生指标，**与引擎同一公式**（防止统计与行为脱节）：
  - `life_span = 一昼夜 × (1 + g3×7)`（= 引擎 `_lifespan()`）；
  - `metabolic_mult = 0.5 + g1×1.5`（= 引擎代谢倍率）；
  - `maturity_age = 寿命 × 0.15`（成熟年龄）。
- `decode_trait_matrix(genes, day_length)` → (n, 17) 表现型矩阵（观察台专用，不参与演化）。

### 文件 2：`observatory/statistics.py` —— 纯函数聚合

- `GenerationStats`：一个观测点的全部统计（全字段 JSON 安全）：trait 均值/标准差、多样性（平均每位点 Simpson 杂合度）、唯一基因型、世代分布对数桶。
- `generation_statistics(engine)`：**直接从引擎内部数组聚合**（`_age/_energy/_generation/_genes`）→ 零对象创建、O(N) numpy，与旧版"遍历 Organism 对象"相比快且与 SoA 引擎天然匹配。空种群返回全零合法观测点。

### 文件 3：`observatory/observer.py` —— EvolutionObserver

- 两个触发器：① 种群 max_generation 前进（新世代出现）；② tick_interval 兜底采样 → 时间轴、世代轴都连续。
- 窗口聚合（出生/死亡计数由调用方逐 tick 喂入）不读引擎历史 → 配合有界历史模式（history_limit=4096）跑百万 tick 实验不依赖被裁剪历史。
- `GenerationSample.flatten()`：CSV 一行（trait 列按 `trait_<name>_mean/_std` 字母序固定列序，直方图只进 JSON）。

### 文件 4：`observatory/experiment.py` —— 实验调度

- `ExperimentSpec/Run/Result`：规格（名字/组/描述/overrides/世代数/tick 上限）→ 确定性派生种子：`SeedSequence([base_seed, group_id, seed_index])`，同参数必复现同结果。
- `build_config()`：组装球面 `SimConfig`（嵌套 dataclass），overrides 按子配置分组浅合并（如 `{"genome": {"mutation_rate": 0.3}}`），未覆盖键保默认。
- `build_plan()`：5 类预置实验（baseline 长程 + pressure/distribution/mutation 短程对照 + repeated_seeds 方差分析）；**球面资源场暂不支持斑块分布 → distribution 组只保留 sparse/rich 均匀对照（诚实标注，不预设机制）**。
- `run_single`：手动逐 tick 循环（每 tick 后按引擎终止条件置位，与 `engine.run()` 语义对齐）；停止条件 = 世代达标 / 引擎自然结束（含灭绝）/ tick 硬上限。
- 引擎选择：球面只有 `SphereEngine`（数组化 SoA）；`use_sim_core=True` 走 Rust 数值管线（统计口径不因此改变）。

### 文件 5：`observatory/__main__.py` —— CLI

- `--rows/--cols`（默认 60×120）、`--sim-core`（Rust 开关）、`--names/--group`（只跑部分实验）、`--generations`（baseline 长程世代数，10,000 完整档）。
- 输出：`--out` 目录，每实验一个子目录（manifest + generations 表）+ 顶层 `__survey__.json/.md` 汇总。

### 文件 6：`persistence/io.py` —— 落盘

- 约定：每实验一个目录 = 名字；`manifest.json`（元数据+完整配置+汇总）、`generations.csv`（宽表）、`generations.json`（镜像）。
- `save_survey/save_survey_markdown`：批量汇总（性状漂变 start→end 人类可读摘要）；`load_manifest/load_generations` 供读取。

### 文件 7：`observatory/broker.py` —— 快照桥（WebSocket 实时直播）

**接口**
| 成员 | 说明 |
|------|------|
| `SnapshotBroker(engine, interval=100, max_snapshots=4096)` | 快照泵：`pump(ticks, observer)` 推进引擎并按节拍采集；`snapshots` 有界环形标量快照；`snapshot(include_individuals)` 现时刻一份 JSON 安全快照 |
| `serve(broker, host, port, ticks)` | 异步上下文管理器：泵在后台线程跑，快照经队列逐条广播给所有客户端 |
| `py -m observatory.broker` | 独立直播入口（`--host/--port/--rows/--cols/--count/--ticks/--interval/--sim-core`） |

**快照内容**：`tick / population / max_generation / total_energy / total_resource / mean_age / mean_energy` + （广播版）`individuals`：每个个体 `{id, flat, energy, generation, age}` —— 前端渲染直接消费。
**两个设计点**：
1. 环形缓冲只存标量版（5000 个体 × 4096 份明细会 OOM），明细只在广播时带；
2. 新客户端连上先补发最新标量快照（落点晚也能立刻看到画面）。

**线程模型**：引擎/rng/ndarray 全程留在泵线程（无跨线程共享）；泵 → `queue.Queue` → asyncio 主循环逐条 `json.dumps` 广播；末条哨兵 `_END_MARK` 通知停机。

### 模块四测试

| 测试文件 | 覆盖 |
|---------|------|
| `tests/test_observatory.py` | trait 表口径（17 列/派生公式）；空种群全零观测点；observer 采样连续性；run_single 冒烟/确定性复现/overrides 合并/灭绝专项；config roundtrip；持久化 roundtrip（manifest+CSV） |
| `tests/test_broker.py` | 节拍采集（tick 100/200/300）；环形缓冲有界；JSON 可序列化；广播带个体明细；WebSocket 端到端推流（兜底+推流 ≥2 份）；无客户端也正常跑完 |
---

## 模块五：基因系统扩展性（simulation/genes.py + sim_core/src/genes.rs）

> C3 基因扩展性（G1~G4），2026-09-07 实现。目标：新增基因位有标准化流程，
> Python↔Rust 双写不漂移，预留位有明确接管规则，元数据供观察台消费。

### G1：加基因五步曲（新增基因位的标准流程）

新增一个基因位（例如 g24 新性状）必须按以下顺序执行，缺一不可：

1. **Python 注册表追加**：`simulation/genes.py` 的 `Gene` 枚举末尾追加成员
   （`NEW_TRAIT = 24`），同步更新 `GENE_SEMANTICS`（功能说明）、
   `GENE_META`（mutation_scale + selection_direction）、`GENE_WIRED`（如已接线）。
   `GENE_COUNT` 自动 = len(Gene)，无需手动改。

2. **Rust 常量追加**：`sim_core/src/genes.rs` 末尾追加同名常量
   （`pub const G_NEW_TRAIT: usize = 24;`），注释写清语义。
   **必须与 Python 侧值完全一致**，否则 `validate_gene_wiring()` 报错。

3. **lib.rs 校验列表追加**：`sim_core/src/lib.rs` 的 `validate_gene_wiring` 函数中
   `rust_all` 向量末尾追加 `("G_NEW_TRAIT", G_NEW_TRAIT)`。
   这是漂移检测的关键——漏掉这一步，该基因的双写漂移不会被发现。

4. **引擎消费**：在 `simulation/sphere_engine.py` 对应步骤中用 `Gene.NEW_TRAIT`
   （而非裸数字 24）消费该基因。如需 Rust 下沉，在对应 Rust 模块中用
   `crate::genes::G_NEW_TRAIT`。

5. **测试 + 验证**：
   - 跑 `tests/test_genes_registry.py`（18 项，含 24 基因逐个篡改检测）；
   - 跑全量 `pytest tests/` 确认无回归；
   - 如改了 Rust，重新编译 `sim_core.so`；
   - 更新 `PROGRESS.md` 记录新基因。

**禁止**：直接改已有基因的索引值（会破坏所有存档和对拍）；
在枚举中间插入新基因（必须末尾追加）；只改 Python 不改 Rust（漂移）。

### G2：双写漂移检测（validate_gene_wiring）

- `simulation/genes.py::validate_gene_wiring()` → 调用 Rust 侧
  `sim_core.validate_gene_wiring(py_names, py_values)`，逐位对照 24 个基因的
  (name, value)，返回不一致列表 `[(name, rust_val, py_val), ...]`。
- 空列表 = 全部一致；非空 = 存在漂移。
- `tests/test_genes_registry.py::TestGeneWiringValidation` 覆盖：
  正常无漂移 / 改值检测 / 改名检测 / 数量不匹配 / **24 基因逐个篡改都能被检测**。
- 引擎初始化时（use_sim_core=True）应调用此函数，漂移则直接报错（防静默错误）。

### G3：基因元数据（GENE_META）

`GENE_META: tuple[GeneMeta, ...]`，与 `GENE_SEMANTICS` 平行，index = 列索引。
每基因两个字段（纯扩展，不改变引擎行为，供观察台/分析工具消费）：

| 字段 | 类型 | 说明 |
|------|------|------|
| `mutation_scale` | float | 突变标准差（高斯突变 σ）。核心基因（代谢/寿命）保守（0.02~0.05），探索性基因（信号/感知/群居）宽松（0.08~0.10），预留位默认 0.10 |
| `selection_direction` | int | 选择方向标注：+1=高值正向选择，-1=低值负向选择，0=中性/环境依赖。仅作观察台标注，引擎不强制（选择压来自生态动力学） |

访问函数：`gene_meta(idx) -> GeneMeta`，越界抛 IndexError。

### G4：预留位接管规则

当前 24 位基因中，**已接线 18 位**（g0~g16, g19），**预留位 6 位**：

| 预留位 | 语义名 | 建议用途 | 优先级 |
|--------|--------|---------|--------|
| g17 | DIET | 食性（肉食/植食/杂食连续谱） | 高（L8 地形+捕食扩展时用） |
| g18 | DEFENSE | 防御（被攻击时的减伤/逃跑概率） | 高（L4 捕食扩展时用） |
| g20 | HEDONISM | 享乐敏感度（愉悦度系统的响应增益） | 中（愉悦度扩展时用） |
| g21 | PROCESSING | 处理位（认知/决策延迟，类工作记忆容量） | 中（语言涌现扩展时用） |
| g22 | TRUST_GENE | 信任阈值（对信号的初始信任/学习率） | 中（L5 信任扩展时用） |
| g23 | RESERVED | 通用预留 | 低（最后使用） |

**接管规则**：
1. **优先复用预留位，耗尽前不扩 gene_count**。新增性状时先检查上表，
   有合适的预留位就接管（改语义名+接线），不要直接扩到 25 位。
2. **接管 = 改语义 + 接线 + 更新元数据**：把 `GENE_SEMANTICS` 从"（预留）"
   改成实际功能，加入 `GENE_WIRED`，调整 `GENE_META` 的 mutation_scale。
   索引值不变（存档兼容）。
3. **6 个预留位全部耗尽后才允许扩 gene_count**（24→25）。扩位时：
   - 必须在 `Gene` 枚举末尾追加（不在中间插入）；
   - 同步 Rust 常量 + lib.rs 校验列表；
   - **触发存档格式升级提示**：快照文件的 `gene_count` 元数据会变化，
     `load_snapshot` 已校验 gene_count，旧快照无法直接加载（需迁移脚本）；
   - 在 `PROGRESS.md` 记录扩位事件和兼容性说明。
4. **预留位在未接管前仍参与进化**（观察台跟踪漂移），但引擎不消费，
   因此对适应度无直接影响（中性漂变）。
