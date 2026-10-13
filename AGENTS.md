# AGENTS.md — the-world 项目代理指南（工程口径）

> 供 AI 代理/协作者快速了解**代码约束、构建命令和关键流程**。
> 详细模块说明以代码为准（旧版 `MODULES.md` 手册已归档至 `_archive/2026-09-17-瘦身/`；项目全景见 `HISTORY.md`）。
>
> 🏛️ **2026-10-10 起：AI 团队模式退役**——开发由 owner（fish）直接下令，不再走讨论板/任务卡流程。
> 团队期协作章程 `AGENT.md`、`_share/`、`collab-toolkit/`、`_eval/` 等团队文件已整体归档至
> `_archive/2026-10-10-退役团队-归档/`（总索引：`_archive/INDEX.md`）。本文件（工程指南）在现行模式下继续有效。
>
> 仓库已于 2026-09-13 迁移：主工作树 = `the-world/` 根，远端 `gitee` = `the-world`；
> 实验数据在独立仓库 `the-world/the-world-data/`（**不进主仓**）。
>
> 🆕 **新 AI / 新协作者接手：先读根目录 `ONBOARDING.md`**——10 分钟跑通 + 架构地图 + 术语表 + 红线 + 加机制/跑实验流程。

## 项目概述

数字生命模拟系统：球面网格（等距圆柱投影），Python 引擎 + Rust（PyO3）加速核。
⚠️ **网格口径**：`WorldConfig` 默认 **60×120（7,200 格）**只是代码默认值与微型冒烟档；
**正式实验档 = 480×960（460,800 格），必须显式 `--device s2`**（R5.6：吃默认值 ⇒ 与历史读数不可比）。
真源：`tools/device_presets.py:22`｜口径出处：`docs/设计总档案/AI-设计总档案-v1.4-定稿.md` §0/§8.5。
目标：在有限死规则内让生命自发涌现智能与语言（不使用 LLM）。

## 技术栈

- Python 3.12 + NumPy
- Rust 1.81+ + PyO3（sim_core 扩展）
- pytest（测试）

## 构建命令

```bash
# 编译 Rust 扩展（release）
export PATH="$HOME/.cargo/bin:$PATH"
export CARGO_TARGET_DIR=/tmp/sim_core_target
export LIBRARY_PATH=/opt/python3.12/lib
cd sim_core && cargo build --release -j1
cp /tmp/sim_core_target/release/libsim_core.so ../sim_core.so

# 跑测试
cd .. && python3 -m pytest tests/ -q --ignore=tests/test_broker.py
```

> 🖥️ **本机（Windows，轻舟机位）配方见 `sim_core/BUILD.md`**：用 `maturin develop --release` 而非 cargo build（cargo 不在 bash PATH；venv 路径含中文须前置 `PYTHONIOENCODING=utf-8`）。本机全量测试：`.venv/Scripts/python.exe -m pytest tests/ -q`（~10 分钟；`test_broker.py` 6 例依赖 fastapi/uvicorn/websockets，环境缺依赖时用上方 `--ignore`）。

## 关键约束

1. **双路径逐位对拍**：所有 Rust 下沉模块必须与 Python 参考实现逐位一致
   （函数级对拍 + 引擎级对拍）。RNG 消费顺序必须一致（预生成随机数传入 Rust）。
2. **基因索引不裸写**：用 `Gene.MOVE_PROB` 而非 `0`。Python↔Rust 双写由
   `validate_gene_wiring()` 校验，漂移则报错。
3. **独立分支**：新功能在 feature 分支开发，不直接碰 main。
4. **快照兼容**：`save_snapshot`/`load_snapshot` 保存 gene_count 和配置指纹，
   扩 gene_count 会破坏旧快照兼容性。

## 加基因五步曲（C3 G1）

新增基因位必须按顺序执行（详见 `simulation/genes.py` 注册表与 `tests/test_genes_registry.py`；旧版 `MODULES.md` 模块五手册已归档）：

1. Python `Gene` 枚举末尾追加 + `GENE_SEMANTICS` + `GENE_META` + `GENE_WIRED`
2. Rust `sim_core/src/genes.rs` 追加同名常量
3. `lib.rs` 的 `validate_gene_wiring` 中 `rust_all` 追加
4. 引擎用 `Gene.NEW_TRAIT` 消费（Rust 侧用 `crate::genes::G_NEW_TRAIT`）
5. 跑 `tests/test_genes_registry.py` + 全量测试

**禁止**：改已有基因索引、在枚举中间插入、只改 Python 不改 Rust。

## 预留位接管规则（C3 G4）

6 个预留位（g17/g18/g20/g21/g22/g23）优先复用，耗尽前不扩 gene_count。
扩位时触发存档格式升级提示（细节见代码注释；旧版手册已归档）。

⚠️ **预留位已全部接管（2026-09-29，13.11 收尾）**：g17 = 追猎倾向（§13.2）｜g18 = 机动性（§13.2）｜
g20/g21 = ARS 双模式觅食（§14.9）｜g22 = 记忆权重乘子（§13.11）｜g23 = 迁移偏置（§13.8）
⇒ **下次新增基因位必须扩 `gene_count`**，并按归档章程 §十三（`_archive/2026-10-10-退役团队-归档/AGENT.md`）触发存档格式升级提示。

## D2 信息结构开发约定（2026-09-09）

D2 是语言涌现的核心重构，分支 `feat/info-structure`。四大机制：学习瓶颈 / 任意性码本 / 信息不对称 / Steels 对齐。

### 开发规则

1. **渐进式开启**：禁止一次性开启全部机制验证。必须按 学习瓶颈+Steels → +码本 → +信息不对称 顺序，每步确认生态稳定。
2. **D2 启用时走 Python 路径**：`use_sim_core=False`。Rust 侧暂未实现码本/学习瓶颈/softmax，后续逐步下沉，每步双路径对拍。
3. **配置总开关**：`InfoStructureConfig.enabled=False`（默认），关闭时行为与旧版完全一致。新增参数必须有默认值且 `from_dict` 旧存档回退。
4. **快照版本**：D2 后 `SNAPSHOT_VERSION=3`，新增 `codebook`/`learning_count`。旧快照加载时回退默认值。
5. **码本是 uint8 (N,16)**：初始恒等映射 `codebook[state]=state`。Steels 对齐用离散概率替换（不能直接浮点运算）。
6. **唯一验收门**：g15 无手调补贴从 0.5 上升（3 seed 一致）。其他指标（cult_div/trust/signal_density）为辅助观察。
7. **生态稳定前置门（R3，2026-09-12 裁定）**：g15 判读前必须先过生态门——至少 ≥5 seed 中**多数**能存活至目标 tick（N 稳定 >0 持续 50k tick）方可进入 g15 判读；灭绝率过高时 D2 gate **视为不可执行**（R3 裁定原文见归档 `_archive/2026-10-10-退役团队-归档/_share/路线共识.md`）。

### 实验工具

- `experiments/run_d2_experiment.py`：D2 参数调优专用脚本，支持四大机制独立开关 + 全部参数 CLI 透传 + 自动 manifest + CSV。
- 详细调优方案见归档件 `_archive/2026-10-10-退役团队-归档/docs-旧档/评估-EVAL/评估-EVAL-D2参数调优实验指南-20260909.md`。

### 禁止事项

- 禁止在 D2 启用时调用 Rust 路径的 `step_movement`/`signal_emit`/`reproduce_batch`（会忽略码本/学习瓶颈）。
- 禁止把码本当浮点数组做 EWMA（uint8 离散映射，用概率替换）。
- 🔴 禁止把 `perception_radius` 改成 4/8 以外的值（`config.py:316` 硬断言只许 `(4,8)`，**传 6 会 F1 硬失败**）。
  ⚠️ 语义澄清（内评 09-19 更正，原文档写"4/6/8 三档"**已过期**）：**`perception_radius` 是"邻居个数"不是"距离"**
  —— `4 = Von Neumann（上下左右 4 格）`、`8 = Moore（含对角 8 格）`；**两者感知范围都是 1 格**。
  另：球面拓扑下"方形窗口"不良定义（极点处 `neighbors(0)` 返回 120 格）⇒ 若要扩大感知，
  须改用**球面距离阈值**，不能直译 9×9（详见 `docs/设计文档/设计-多位编码v0-20260919.md`）。

## 条件表 C1–C5（项目纪律；R2 采纳，替代原"语言涌现五要素"）

> 原"五要素"是实现要素堆叠、无一项经实验验证必要性。改为**按证伪成本排序**的条件表。
> **纪律**：此后每个实验/战役**必须按本表逐条对照报告**——本实验在检验哪一条件、
> 其余条件是否被无意削弱。违反（如靠"加资源"让生态好看）即视为抹掉被测条件。

| # | 条件 | 语义 | 当前状态 | 验证方式 |
|---|------|------|---------|---------|
| **C1** | 信息不对称 | 感知半径 < 世界尺度 + 感知噪声 | ⚠️ **尚未被检验** —— 原判"已实测主导生态崩溃"**已作废**：该判定基于缺陷代码（`nb[:4]` 取半平面，R12 定性为 bug fix、R19 整批作废）；反证见归档《战略定位-20260912-语言涌现可达性》（`_archive/2026-10-10-退役团队-归档/docs-旧档/决策与评审/`）§三·补（修正后 `vn_r4=120 ≥ off_r8=105`） | 开关 `info_structure.perception_radius`(4/8)、`perception_noise`；**须用 `7eeaa5f` 之后代码重测** |
| **C2** | 共识成本 > 0 | 学习瓶颈 + 任意性码本 | ⚠️ D2 已实现，但 **W-F3 判定只实现半个条件**（缺 MDL 紧凑性偏差） | 学习瓶颈开关 + 码本熵/紧凑性指标 |
| **C3** | 发送者获益通道 | payoff 耦合（信号对发射者有私利） | ❌ **完全缺失**；用 **R2 声誉权重**验证 | `info_structure.reputation_weight`（默认 0=关闭）；判读**须先过生态门**（R6） |
| **C4** | 环境信息价值 | 稀有事件 + 时空异质 | ⚠️ 斑块已做；季节未做 | 斑块/季节开关 + 信息收益指标 |
| **C5** | 社会复杂度 | 亲缘/互惠/群体协调 | ⚠️ 仅弱群居 g13；**文献支持最强**（社会脑/马基雅维利） | 群居/互惠机制 + 社会结构指标 |

**优先级（证伪成本序）**：C1 → C3 → C2 → C4 → C5。

### 生态稳定前置门（R3/R6，硬门）

g15 任何判读**之前**必须先过生态门：**≥5 seed 中多数存活至目标 tick，且 N 稳定 >0 持续 50k tick**。
生态门不通过 ⇒ D2 gate 与 C3（R2 声誉权重）**均视为不可执行**，不得下任何正面/负面结论。

---

## 测试状态（**2026-10-13 实测**；全量 `tests/` ≈9 分钟）

- **全量（标准命令，`--ignore=tests/test_broker.py`）**：**1377 passed / 52 failed / 9 skipped**（`[实测]` 2026-10-13，537 s）。含 broker 档 = 1383/52/9（`test_broker.py` 6/6 绿，需 fastapi/uvicorn/websockets）。
- **52 红全部为存量环境族、非代码回归**（逐名核过）：`test_f_r21_presets` ×36 + `test_r278_p3_device_flush` ×7（含 1 例 ULP 最末位）+ `test_r287_s3_sample_resume` ×4 + `test_221_snap_shard` ×3 + `test_r98_nonascii_print` ×1 + `test_f_r10_f_r12` ×1。
  根因：子进程 stdout 写 UTF-8，父进程 `subprocess.run(text=True)` 未指定 encoding ⇒ 中文 Windows 按 locale **GBK 解码** ⇒ `UnicodeDecodeError`（环境问题，非代码问题）。
- 另有 1 例并发 flake：`test_r110_criteria.py::test_concurrent_start_exactly_one_wins`（负载下偶发；触发时总数变 1376/53/9，隔离复跑绿）。
- **对账纪律**：与失败**集合**逐名对比（零新增才算过），不看总数（pass 数随测试增长）；新写 subprocess 测试两侧钉 UTF-8（env `PYTHONIOENCODING=utf-8` + `run(encoding="utf-8")`）。
- 快速子集（≈2 秒，49 例）：`test_env_field.py` + `test_sim_core.py` + `test_genes_registry.py` ⇒ 49 passed。更多口径见 `ONBOARDING.md` §1.4。

## 当前阶段（**2026-10-13 更正**；活文档：`docs/项目状态.md` §三）

> ⚠️ 本节原为 2026-09-13 版本（P0' 生态稳定性修复为唯一阻塞）；随**纯斑块世界线**（2026-10-11 拍板）与运行模式切换（owner 直领），主线已更换，旧 P0' 表述作废。**查在办线一律以 `docs/项目状态.md` 为准**（下面是 2026-10-13 快照）。

**在办线**：

1. 🔝 **接手开发指南 / 项目记忆（owner 令第一优先）**——让任意无历史的新 AI 只看仓库即可接手；产物 = `ONBOARDING.md` + `docs/技术文档/技术-架构地图-20261013.md` + `docs/技术文档/技术-历史教训与关键决策-20261013.md`。传达件 `docs/汇报/话事人传达-20261013-C裁定与接手开发任务.md`。
2. **纯斑块世界线**（无生物、底层规则自组织）——四臂批（s2/2000t）跑毕，交裁项 owner 裁定 **C**：本批不下单边结论；下一批 amend 预注册（fixed-k 主判据 + 明示 Magnus 纬度混杂 + 去混杂臂）**排在接手指南之后**。汇报 `docs/汇报/汇报-20261011-纯斑块世界四臂.md`｜预注册 `docs/预注册/预注册-纯斑块世界-四臂-20261011.md`。
3. **实验主线 S0′**（ENERGY-CLOSE 合成闭环口径）——⏸ 三裁点已定档，接续方式待 owner 示意。
4. **可视化 v1**——已实施完毕（分支 `feat/viz-v1-globe`），**owner 审过后才并 main**（v0 已在 main）。

**纪律锚点**：对外表述一律锚定《战略定位》的 **L0–L4 判据阶梯**（原文见归档 `_archive/2026-10-10-退役团队-归档/docs-旧档/决策与评审/战略定位-20260912-语言涌现可达性.md`）；未过判据的不得写成"涌现"。
