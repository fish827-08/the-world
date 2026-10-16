# ONBOARDING.md — 接手开发主入口（零对话历史可读）

> 建立：2026-10-13（owner 直令「项目记忆工程」；让任意无历史的新 AI/新人只看仓库即可接手）。
> **读法（按顺序）**：本文件（跑通 + 术语 + 红线 + 流程）→
> `docs/技术文档/技术-架构地图-20261013.md`（代码在哪）→
> `docs/技术文档/技术-历史教训与关键决策-20261013.md`（为什么这么设计、踩过哪些坑）。
> **唯一权威源**：设计 = `docs/设计总档案/AI-设计总档案-v1.4-定稿.md`；状态 = `docs/项目状态.md`；
> 其余文档与本两处冲突时以两处为准。本文件只放「代码看不出来的东西」，能从代码直接读的不抄。

---

## 0. 三十秒定位

**数字生命模拟系统**：球面网格（等距圆柱投影）+ Python 引擎 + Rust(PyO3) 加速核。
目标：在有限死规则内让生命**自发涌现**智能与语言（**不使用 LLM**）。
正式实验档 = **480×960（460,800 格），必须显式 `--device s2`**；吃代码默认值（60×120 微型档）⇒ 与历史读数不可比（R5.6）。装置真源 `tools/device_presets.py:45`。

**2026-10-13 时点，两条在办线**（现状以 `docs/项目状态.md` 为准）：

| 线 | 状态 |
|---|---|
| **纯斑块世界**（底层规则自组织：无生物 → 地形 → 河流） | 四臂批已跑（s2/2000t/seed42）；owner 裁定 **C**：本批不下单边结论（Magnus 纬度混杂 + fair-k 塌缩）；下一批 = amend 预注册后去混杂再跑 |
| **S0′ 守恒闭环**（ENERGY-CLOSE） | 合成档先证「账对」；引擎世界全链守恒审计**未闭环**；接续待 owner 示意 |

**代码规模**（体检件 `docs/技术文档/技术-代码体量体检与结构建议-20261013.md`）：核心 Python ≈1.6 万行 + Rust ≈3 千行；`experiments/` ≈2.9 万行（大量历史一次性脚本，旧线已归档进 `experiments/archive/`）。

目录速览：

| 目录 | 是什么 |
|---|---|
| `simulation/` | 引擎（`sphere_engine.py` 6967 行巨石 = 主力；`config.py` 全部配置；`genes.py` 基因注册表） |
| `world/` | 环境场与几何（球面网格/光照温度/资源/信号/气味/ENV 场/亚格）；**自足：只 import `world/`** |
| `core/` | 生命周期枚举等最底层 |
| `observatory/` | 实验矩阵调度、统计采样、实时快照 broker |
| `persistence/` | 实验产物落盘/装载（manifest/survey） |
| `viz/` | 数据包导出器 + web 前端（v0 在 main；v1 在独立工作树待审） |
| `s0_kernel/` | **零引擎依赖**的通道驱动迷你世界（守恒断言 + 双臂 harness，S0′ 线） |
| `sim_core/` | Rust 加速核（PyO3；数值热路径下沉） |
| `experiments/` | 实验探针脚本（新脚本放这里；旧线已入 `experiments/archive/`） |
| `tools/` | 跨批工具（装置预设/审计/判读/守卫；旧线已入 `tools/archive/`） |
| `tests/` | pytest 全量测试 |
| `docs/` | 文档（先读 `docs/README.md` 索引） |
| `the-world-data/` | **独立数据仓**（实验数据 .npz/.csv，`git clone` 单独拉，不入主仓） |

---

## 1. 十分钟跑通（逐条可复制；以下命令均已实测）

### 1.1 拉仓

```bash
# gitee 为权威源；github 为 main 镜像（只同步 main）
git clone https://gitee.com/little-fishy/the-world.git && cd the-world
# （可选）实验数据独立仓——判读历史批、看 viz 数据时才需要
git clone https://gitee.com/little-fishy/the-world-data.git
```

### 1.2 建环境

```bash
# Windows ⚠️ 本机 bash 里没有 `python`/`python3` ⇒ 用 Windows 启动器 py（`py -0` 可列已装版本）
py -3.14 -m venv .venv
.venv/Scripts/python.exe -m pip install numpy pytest maturin   # maturin = §1.3 编译 Rust 扩展用
# Linux / macOS：
python3 -m venv .venv && .venv/bin/python -m pip install numpy pytest maturin
```

Python ≥3.10（`sim_core/pyproject.toml` requires-python）；本机实况 **3.14.7** + numpy 2.5.x（未 pin：2.5.3/2.5.4 均实测）+ pytest 9.1.1。
（`pytest-asyncio`/`fastapi`/`uvicorn`/`websockets` 只有 `tests/test_broker.py` 用——该文件在标准全量命令里被 `--ignore`，可先不装。）

### 1.3 编译 Rust 扩展（跑全量测试前必做）

```bash
# 【Windows 本机】`which cargo` 为空时先补 PATH（2026-10-13 实测一般已在 PATH，此行兜底）
export PATH="$HOME/.cargo/bin:$PATH"
cd sim_core
PYTHONIOENCODING=utf-8 ../.venv/Scripts/python.exe -m maturin develop --release
cd ..
.venv/Scripts/python.exe -c "import sim_core; print(sim_core.__file__)"   # 验证
```

完整说明（含受限环境绕行配方）见 `sim_core/BUILD.md`。
不编译也能跑纯 Python 路径（`use_sim_core=False` 是默认），但全量测试会因 `tests/test_sim_core.py` 硬失败而挂（有意设计，防静默跳过）。

### 1.4 跑测试

```bash
# 快速自检（约 4 秒，72 例：ENV 场 + Rust 对拍 + 基因注册表）
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m pytest tests/test_env_field.py tests/test_sim_core.py tests/test_genes_registry.py -q

# 全量（本机约 11 分钟）：预期 1499 passed / 1 failed / 9 skipped（唯一红 = T2b，见下）
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m pytest tests/ -q --ignore=tests/test_broker.py
```

⏱️ 全量 9–11 分钟（537–674 s 实测窗口），若工具/子代理单命令有 ~10 分钟超时上限，用**分段跑法**（对账时把两段的失败集**合并**看）：

```bash
F=$(ls tests/test_*.py | grep -v test_broker); H=$(( $(echo "$F" | wc -l) / 2 ))
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m pytest $(echo "$F" | head -n $H) -q
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m pytest $(echo "$F" | tail -n +$((H+1))) -q
```

**存量六族 52 红已于 2026-10-16 清零并合入 main**（merge `7263619`，owner 10-16 批准；`[实测]` 默认模式 1499/1/9、UTF-8 模式同为 1 failed；逐名明细见 `docs/汇报/汇报-20261016-测试红族清零.md`）：

- 解码族 50 例（`test_f_r21_presets` ×36 + `test_r278_p3_device_flush` ×6 + `test_r287_s3_sample_resume` ×4 + `test_221_snap_shard` ×3 + `test_f_r10_f_r12` ×1）：父进程 `subprocess.run(text=True)` 未指定 encoding ⇒ 在中文 Windows 上按 locale **GBK 解码**子进程的 UTF-8 输出（Python 3.14 表现为 reader thread 解码崩、流截断 / `r.stdout=None`）。修法 = **双钉 UTF-8**（env `PYTHONIOENCODING=utf-8` + `run(encoding="utf-8")`）——**新写 subprocess 测试照此双钉**，勿只钉一半。
- `test_r98_nonascii_print` ×1：16 个文件 print 含 GBK 不可编码字符且无 reconfigure 调用点 ⇒ 按 R216 裁定一模式补入口兜底。
- 唯一遗留红 = `test_r278_p3_device_flush.py::test_T2b_s2_default_csv_equals_baseline`（1–2 ULP）：夹具系异地生成、本机 venv 不可复现（四时代代码输出逐位一致，见汇报 §五），A/B/C 已交 owner，**未动容差**。
- 另 1 个并发 flake：`test_r110_criteria.py::test_concurrent_start_exactly_one_wins`（负载下偶发；隔离复跑绿；锁写窗口根因已定位，修它要动 batch_runner 语义 ⇒ 维持现状）。

**对账纪律**：只看「失败集合是否与上述清单一致、零新增」，不看 pass 数绝对值（pass 随测试增长）。在 UTF-8 locale 的 Linux/macOS 上无 GBK 族问题；本机复核实测：解码族转绿的真条件是 `PYTHONUTF8=1`（UTF-8 模式），仅 `PYTHONIOENCODING=utf-8` 不够（它只改流编码、不改 `subprocess` 的 locale 解码）。

### 1.5 最小实验

```bash
# A. 无生物纯斑块世界（arm 必填：1 现状 / 2 置换零模型 / 3 地形 / 4 全规则；experiments/run_bare_world.py:50）
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -X utf8 experiments/run_bare_world.py \
    --arm 4 --rows 96 --cols 192 --patches 40 --ticks 100 --sample 25 --out _rerun_logs/bare_smoke
# 预期：打印 [arm4 全规则(处理)] ticks=100 ... 及 env: w_mean=... river_frac=...（本机 0.4s）

# B. 有生物微世界（50 tick，纯 Python 路径；本机 0.3s）
PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -X utf8 -c "
from simulation.config import SimConfig
from simulation.sphere_engine import SphereEngine
stats = SphereEngine(SimConfig(seed=1)).run(50)
print('ticks:', len(stats), '| pop:', stats[-1].population)"
# 预期：ticks: 50 | pop: 157
```

说明：小装置自检才传小 `--rows/--cols`（**非 480×960 会在 stderr 打「非 s2 正式档」提示**）；**正式批必须 `--device s2`**（忘传 = 静默换装置，R278 防呆会回显）。
⚠️ 无生物小档 100 tick 的读数**全静态是预期**（初始饱和 + 无消耗者；`[实测]` `stock_sum` 恒定、`sat_mean=1.0`）——它只证明管线通、零科学信息；自检完请核对 manifest 的 `device_applied`/`world.rows`（显式小档忘改回此前无回显，2026-10-13 起有 stderr 提示兜底）。
`run_bare_world` 默认输出到 `the-world-data/bare_world/`（数据仓）；自检请指到 `_rerun_logs/`（已 gitignore）。

### 1.6 常见报错速查

| 症状 | 原因 | 解法 |
|---|---|---|
| `python: command not found`（bash 内） | 本机 PATH 里没有 `python`/`python3` | 用 Windows 启动器：`py -3.14`（`py -0` 列已装版本）；建好 venv 后一律走 `.venv/Scripts/python.exe` |
| `No module named maturin` | 新建/更换 venv 后未装构建器 | `.venv/Scripts/python.exe -m pip install maturin`（§1.2 已含） |
| `UnicodeDecodeError: 'gbk' codec ...` / 控制台乱码 | 中文 Windows 控制台编码 | 跑脚本前 `PYTHONIOENCODING=utf-8`（py≥3.12 可加 `-X utf8`）；新脚本入口照抄 `experiments/run_bare_world.py:204-208` 的 stdout reconfigure；**自写 subprocess 测试两端钉 UTF-8**（env + `run(encoding="utf-8")`），否则给本机加伪失败 |
| `which cargo` 为空 | cargo 不在 bash PATH（实际在 `C:\Users\<你>\.cargo\bin`） | 用绝对路径或 `export PATH="$HOME/.cargo/bin:$PATH"` |
| maturin 在 pip 输出处 GBK 解码中止 | venv 路径含中文（本机 `C:\Users\圣羽\...`） | `PYTHONIOENCODING=utf-8` 前置（`sim_core/BUILD.md §2`） |
| `failed to run the Python interpreter ... (os error 231)` | 受限/沙箱环境里 pyo3-build-config 无法 spawn Python | 只做编译自检：`PYO3_NO_PYTHON=1 cargo check --release --features pyo3/extension-module,pyo3/abi3-py314`（不产 .pyd；`sim_core/BUILD.md §3`） |
| 收集期硬失败「sim_core 扩展未安装」 | 没编译扩展 | 做 §1.3 |
| Linux 构建链接错误 | 缺库路径 | `LIBRARY_PATH=/opt/python3.12/lib`（AGENTS.md 构建命令） |
| 全量测试 1 failed（T2b ULP，A/B/C 交 owner） | 唯一遗留红，非代码回归 | 见 §1.4 清单与对账纪律 |

---

## 2. 架构地图（代码在哪、数据怎么流）

→ 完整版：`docs/技术文档/技术-架构地图-20261013.md`。三句话摘要：

1. **引擎是唯一状态机**：`SphereEngine.step()`（`simulation/sphere_engine.py:2921`）→ `_advance_one_tick`（:2934，固定八段顺序）→ `_step_population`（:3383，11 段）；每 tick 读哪些场、写哪些状态，见地图 §2.3/§2.4 两张表。
2. **依赖方向单向**：`world/` 自足（只 import `world/`）；`simulation/` 依赖 `world/` + `core/` + `sim_core`；`observatory/`、`viz/` 依赖 `simulation/`；`s0_kernel/` **零引擎依赖**（独立复刻，供守恒断言）。
3. **数据流**：`SimConfig`（一切开关，`simulation/config.py:1532`）→ 引擎 → 场状态 → 落盘（快照 `.npz` 走 `save_snapshot` `:6375`；批产物 CSV+manifest 走探针脚本）→ 判读脚本（readout）→ 汇报。
   实验数据一律入独立仓 `the-world-data/`，**不进主仓**。

---

## 3. 核心术语表（一词一条，以代码为准）

| 词 | 含义 | 锚点 |
|---|---|---|
| **tick** | 引擎最小步进；一 tick = `_advance_one_tick` 的固定八段顺序（资源再生→尸体→信号→气味→种群行动→rd 轮作回写→能量封顶→观测） | `simulation/sphere_engine.py:2934`；步序表见架构地图 §2.3 |
| **装置档 device s2** | 正式实验的命名装置预设（480×960｜patches 1700｜rgm 1.195｜背景纯零产能）；不传 `--device` 会吃代码默认微型档 ⇒ 读数不可比（R5.6） | `tools/device_presets.py:45` |
| **斑块 patch / 背景产能** | `distribution="patchy"`：资源只产在少量斑块格；背景格产能为零（`bg_production_zero=True`，s2 口径）。掩码由独立流生成 | `world/resource_dynamics.py:206`；`simulation/config.py`（`resources.*`） |
| **基因位 gene** | 个体基因组按 index 索引（g0–gN）；新增必须走「加基因五步曲」（Python+Rust 双写），禁裸写索引 | `simulation/genes.py:30`（`Gene`）、`:64`（`GENE_COUNT`）；AGENTS.md |
| **记忆 memv2** | 记忆 v2 总开关（朝向梯度版；默认关=False 时逐位不变） | `simulation/config.py:635` |
| **场 field** | 环境空间张量：`smell` 气味（4 通道 food/prey/risk/kin）、`signal` 信号（D2 码本）、`env` ENV 场（地形→水分→产能因子） | `world/smell_field.py:99`、`world/signal_field.py:19`、`world/env_field.py:112` |
| **rd（资源动态）** | 斑块可耗竭/轮作（休耕、反荒漠化、容量重算）；开 rd = 资源动态 | `world/resource_dynamics.py:67`；开关 `simulation/config.py:1206` |
| **sparse_fields** | 惰性场结算总开关（默认关）；开档 = 子集结算，逐位等价是硬前提 | `simulation/config.py:426` |
| **臂 arm** | 同批对照/处理分组；纯斑块四臂 = 1 现状 / 2 置换零模型 / 3 地形 / 4 全规则（只动规则开关） | `experiments/run_bare_world.py:50-59` |
| **seed** | 随机种子；跨臂同 seed ⇒ 配对可比 | `observatory/experiment.py:86`（derive_seed） |
| **manifest** | 批产物元数据（git 号/装置/计时/probe）——判读与复盘的第一手证据 | `experiments/run_bare_world.py:165-187` |
| **readout（判读读数）** | 对批产出按预注册判据做判读的读数脚本/口径，与「跑批脚本」分离以防自证 | 范例 `tools/tier1_judge.py`、`tools/verify_k.py` |
| **C7 逐位一致** | Python↔Rust 同配置逐位对拍（回归红线；不放宽、不用 float32——放宽=能力归零证明，float32=行为分岔） | `docs/设计总档案/AI-设计总档案-v1.4-定稿.md:424-425`；现状钉子 `tests/test_energy_probe.py:150`（digest `(574887, 11266.746993)`） |
| **守恒审计** | 独立账本审计能量流：`measured − predicted = 残差`，`close()` 是唯一会合点（禁同账自证）；现役载体 = S0′ 合成闭环 | `s0_kernel/books.py`（模块 docstring）、`tests/test_s0_kernel.py` |
| **快照 / 配置指纹** | `save_snapshot`/`load_snapshot` 存 gene_count + 配置指纹；扩 gene_count = 破坏旧快照兼容（升级提示） | `simulation/sphere_engine.py:6375`、`:6590` |
| **L0–L4** | 信号涌现判据阶梯；未过判据**不得写「涌现」**。原文在归档 | `_archive/2026-10-10-退役团队-归档/docs-旧档/决策与评审/战略定位-20260912-语言涌现可达性.md` |

---

## 4. 红线与约束（违反即视为无效改动/回滚）

1. **禁 GPU**：GPU/float32 破坏逐位对拍（C7 是回归红线）。R9.3：「C7 不放宽、不用 float32」`docs/设计总档案/AI-设计总档案-v1.4-定稿.md:424-425`。
2. **禁外部随机数**：随机性只用 `CountingRNG`（禁 `np.random` 全局、禁 `random` 模块）；跨 Python↔Rust 的随机数必须在 Python 侧预生成按序传入，RNG 消费顺序一致。`simulation/provenance.py:35`。
3. **基因索引不裸写**：用 `Gene.MOVE_PROB` 而非 `0`；加基因走五步曲（Python+Rust 双写 + `validate_gene_wiring` 校验）。预留位已全接管 ⇒ 下次新增必须扩 `gene_count` 并触发存档格式升级提示（AGENTS.md）。
4. **双路径逐位对拍**：所有 Rust 下沉必须与 Python 参考实现逐位一致（函数级 + 引擎级两层）。
5. **新机制默认关 + 关档逐位等价**：新 config 开关默认关；默认路径行为逐位不变（回滚点=一条钉死 digest 的测试）。
6. **数据不进主仓**：`.npz`/`.csv`/批产物 → 独立仓 `the-world-data/`；主仓 `.gitignore` 已挡。
7. **未过判据不写「涌现」**（L0–L4）；**对照不干净时不下结论**（2026-10-13 裁定 C 即范例：Magnus 纬度混杂 + fair-k 塌缩 ⇒ 两读并列，不宣布地形有效/无效）。
8. **快照兼容**：`gene_count`/配置指纹随存；扩展会破坏旧快照兼容（触发升级流程）。
9. **文件与 Git 纪律**：文件只许放 `the-world/` 内（域外会被清）；临时文件用系统 `%TEMP%` 用完即删；提交用明确 pathspec（禁 `git add .`）；多会话共享 worktree 时用 `git commit --only <pathspec...>`（覆盖重命名新旧两侧）；🔴 **禁 `git worktree prune`**；批跑期间不动被 import 的代码（F-R9）。
10. **机时墙（R228）**：本机批默认 23:00 止、24:00 断电；跑批前报预算。新机制改动前先跑 1 run × 2000 tick 性能冒烟（C1）。

---

## 5. 加一个新机制的标准流程（样板：`env_field` / `s0_kernel`）

1. **设计稿 → owner 拍板**：拍板前零代码（纯斑块世界线、viz v1 都是这样走过来的）。
2. **独立模块 + config 组**：新场放 `world/`（或独立包），配置放 `simulation/config.py` 的独立 dataclass（列进 `SimConfig`），**默认关**；新开关经 `asdict` 自动进 `fingerprint()` ⇒ 跨档续跑被拦。
3. **构造期 fail-loud 互斥守卫**：不满足前提的组合直接报错（禁静默降级）。样板：`EnvFieldConfig` 的 `enabled ∧ use_sim_core ⇒ NotImplementedError` 等守卫（`simulation/config.py:1453` 附近 + 引擎构造）。
4. **关档逐位等价测试**：默认关时旧 digest 不动。样板：`tests/test_env_field.py::test_e1_env_sens0_bit_equivalent`。
5. **变异检查 / 零模型**：证明机制「真的在起作用」而不是摆设——置换零模型、确定性、守卫、结构不变量。样板：`tests/test_env_field.py` E1–E8（置换保直方图、路由树严格下降、因子单调等）。
6. **单测 + 小冒烟**：1 run × ≤2000 tick；性能冒烟（1.10 开/关 ms/tick 中位比门禁）。
7. **预注册**：判据/零模型/统计口径**跑前锁死**（含固定 k、去混杂设计），见 `docs/预注册/`；范例 `docs/预注册/预注册-纯斑块世界-四臂-20261011.md`。
8. **跑批 → 判读 → 汇报**：见 §6。

**最小改动清单**（依据 env_field 样板，防漏登记——新 AI 实测易在多文件登记环节耗时）：

- **动文件**：① 机制本体（`world/` 新模块或现有文件）② `simulation/config.py`（新 config 组/字段：默认关 + 列入 `SimConfig` + `from_dict` 旧档缺键回退）③ 引擎接线（如需要，默认关分支）④ `tests/test_<机制>.py`（关档逐位等价 + 机制生效 + fail-loud 守卫，≥3 例）。
- **必跑（按序）**：① 靶向单测 ② 快速子集（§1.4，72 例）③ C7 钉子 `tests/test_energy_probe.py`（digest 不动）④ `tools/tick_denomination_audit.py` + `tools/config_fallback_audit.py`（rc=0；bool 字段自动豁免登记）⑤ 全量（唯一红 T2b 对账零新增）⑥ 性能冒烟 1 run × ≤2000 tick。
- **提交**：新分支提交（不碰 main）；提交信息写明「**指纹变、行为不变**」（新键进 `fingerprint()` 时）与「默认关逐位等价」。

---

## 6. 跑实验 / 判读流程

1. **预注册先于跑批锁死**（有的批甚至先于代码合并）：判据、零模型、统计口径、固定参数（如 fixed-k）写进 `docs/预注册/`，此后不得事后改。
2. **跑批**：探针脚本（`experiments/`）+ 显式 `--device s2`；产物 = CSV + manifest（git 号/装置/计时/probe 全留痕）。
3. **数据入数据仓**：`the-world-data/` 内同类目录（如 `bare_world/`），独立 commit+push。
4. **判读（readout）**：独立读数脚本按预注册判据逐条对照（跑批与判读脚本分离）；「判负出口」照预注册文本执行（一般是上呈候裁，不写「证伪」）。
5. **汇报**：`docs/汇报/`（一报一文件；结论先行 → 证据分级 `[实测]`/`[读数]`/`[推断]` → 待决策项顶置）；状态变化原位更新 `docs/项目状态.md`。
6. **对照不干净时**：不下单边结论，两读并列呈裁——2026-10-13 裁定 C 是标准范例（`docs/汇报/话事人传达-20261013-C裁定与接手开发任务.md` §一）。

---

## 7. 历史教训库（7 条速览；详版=现象→根因→防线 见 ADR 文档）

> 详版：`docs/技术文档/技术-历史教训与关键决策-20261013.md`。

| # | 现象 | 根因 | 现在的防线 |
|---|---|---|---|
| 1 | 「K=2 留守/出走双稳态」曾被当涌现 | 背景倍率**抽奖 bug**：`from_field` 用背景第 0 格反推全局倍率，该格恰好低产能的概率 ≈40% ⇒ 装置随机漂移 | R320 撤绿洲带（bg 0.0/0.0，抽奖从参数上消失）+ R328 `from_field` 直读显式字段 + `tools/check_bg_lottery.py` 守卫。`tools/device_presets.py:34-44` |
| 2 | 置换零模型（arm2）不干净 | Magnus 公式把纬度压印进水分场 ⇒ H 置换后仍留纬度暗通道 | 预注册显式声明混杂 + 下一批去混杂设计（关 base_t 梯度或先对 \|lat\| 回归取残差）。`docs/汇报/汇报-20261011-纯斑块世界四臂.md` |
| 3 | 对账脚本静默错一位 | `W(g)` 在 `g=gate` 有**跳跃间断**（严格 `g>gate`；引用「W(0.3)=1.072」实为右极限，真值 0.49） | `gate_inclusive` 显式暴露边界约定（默认 False=引擎字面语义）。`s0_kernel/arms.py:19-22`；`docs/预注册/落笔-双臂S0-H2-20261009.md:128` |
| 4 | ETA/成本线性外推翻车 | 性能随种群增大减速（短跑 2.9 ms/tick，20k 实测 ≈12.2 ms/tick，慢 ~4×） | 预算按实测 tick 成本+种群曲线报，不按短跑外推；批前性能冒烟。归档 `_archive/2026-10-10-退役团队-归档/_share/讨论板.md:11411` |
| 5 | ~50 个测试红看着像代码回归 | 中文 Windows locale + subprocess 未钉 UTF-8 的**环境族** | 对账只看失败集合零新增；新测试两端钉 UTF-8。§1.4 清单 |
| 6 | `kin` 通道被当真实亲缘读 | 曾是 ones 占位假实现（无真实亲缘计算） | 现况如实：`kin` 源=全体存活个体（与 prey 同源），join 响应由 g13 基因门控 ⇒ **不得当「亲缘识别」读**。`simulation/sphere_engine.py:2877/:2898/:3253-3258` |
| 7 | 旧实验批次结论不可复现 | 默认值回落（忘传装置参数）/半硬编码/抽奖 bug 混入 | 装置预设防呆（`--device` fail-loud + 显式优先回显）+ R5.6 显式档纪律 ⇒ **引用任何历史结论前先查其装置与有效性** |

---

## 8. 关键决策记录（6 条速览；详版见 ADR 文档）

1. **撤 patch_mask 人为倍率层**：斑块/群系「不人为规定」（fish 原话）——撤掉用二值掩码乘倍率的做法，改由底层规则（地形→水分→产能因子）自组织；两条红线：掩码退出行为路、群系标签永不进判据前置。`docs/设计文档/设计-ENV-FIELD-物理场与格子产能-20261008.md` §二/§六（ENV 场已落地、默认关，整线在纯斑块世界线推进中）。
2. **纯斑块世界先行**：先做无生物的环境底层规则（地形/河流/斑块自组织），再放生物——避免生物行为掩盖底层规则信号。`docs/设计文档/设计-纯斑块世界-底层规则自组织-20261011.md`。
3. **禁 GPU**：GPU/float32 破坏逐位对拍（C7 回归红线）；当前规模数据传输开销 > 计算收益。`docs/技术文档/L7d-Rayon并行化技术文档.md:184`；`docs/设计总档案:425`（R9.3）。
4. **坚持逐位一致（C7）**：Python↔Rust 同配置逐位对拍是「能力不归零」的证明基础，宁慢不放宽。
5. **多角色团队 → owner 直领**：2026-10-10 起 AI 团队模式退役（讨论板/任务卡/派单全停），owner 直接下令；团队文件整体归档 `_archive/2026-10-10-退役团队-归档/`。`AGENTS.md` 头部；`docs/项目状态.md` §二。
6. **守恒闭合先行**：「账对不对」先于「生态活不活、信号涌没涌现」——S0′ 用零引擎依赖合成档先证逐 tick 恒等式；引擎世界的全链守恒审计未闭环前，相关结论须标注账目前提。`s0_kernel/` + `tests/test_s0_kernel.py`；`docs/预注册/落笔-双臂S0-H2-20261009.md`；`docs/项目状态.md`。

---

## 9. 文档地图（含过期标注）

| 想知道什么 | 去哪读 | 状态 |
|---|---|---|
| 接手入口 / 跑通 / 红线 | **本文件** | 🟢 2026-10-13 |
| 代码在哪（架构/步序/边界） | `docs/技术文档/技术-架构地图-20261013.md` | 🟢 |
| 为什么这么设计 + 踩过的坑 | `docs/技术文档/技术-历史教训与关键决策-20261013.md` | 🟢 |
| 设计唯一依据 | `docs/设计总档案/AI-设计总档案-v1.4-定稿.md`（+ fish 大白话版） | 🟢 唯一权威 |
| 项目状态（活文档） | `docs/项目状态.md` | 🟢 唯一权威 |
| 工程约束与构建 | 根 `AGENTS.md` | 🟢（测试状态/当前阶段两节以本文件 §1.4 与项目状态为准） |
| 实验流水 | `docs/实验台账.md` + `docs/实验记录.md` | 🟢 活文档 |
| 预注册（跑前锁死判据） | `docs/预注册/` | 🟢 |
| 给 owner 的汇报 | `docs/汇报/`（含 README 约定） | 🟢 |
| 体检/结构建议 | `docs/技术文档/技术-代码体量体检与结构建议-20261013.md` | 🟢 |
| 历史文档（判读/评估/团队件） | `_archive/INDEX.md` → `_archive/2026-10-10-退役团队-归档/` | 🗄️ 只读留痕 |
| 设计稿 / 规格（机制级） | `docs/设计文档/`、`docs/规格与机制/` | 🟡 逐件看日期，旧稿口径可能已过期 |
| ⚠️ 根 `PROJECT-DESCRIPTION.md` | — | 🔴 **已过期**（2026-09 口径，写「217 passed」；勿引用） |
| ⚠️ 根 `HISTORY.md` | — | 🟡 项目发展史（止于 2026-09-17，团队时代路径已归档） |
| ⚠️ `docs/设计总档案/_archive_评审过程_v1.0-v1.3/` | — | 🗄️ 评审过程附件；**以 v1.4 为准** |

---

## 附 A. 验收自测四题（本指南的硬标准）

新 AI（零对话历史）只拿本仓库，应能独立完成：

1. **构建并跑通测试** → §1.1–1.4（含识别「53 红=存量环境族」不误判为回归）。
2. **跑通最小 bare_world 实验** → §1.5 A（能解释 arm 1/2/3/4 各是什么、为什么小装置不能冒充正式批）。
3. **定位「某生物某 tick 在哪段代码进食」** → 架构地图 §2.4 的第 3.5) 段（`consume_many` 双路径），顺着 `_step_population` 行段表找。
4. **合规提交一个默认关的最小新机制（或小修复）并补测试** → §5 流程 + env_field 样板（config 组默认关 + 关档逐位等价测试 + fail-loud 守卫）。

## 附 B. 维护约定

- 本文件随主线演进**原位更新**；新增红线/教训必须同步进 §4/§7（并更新 ADR 详版）。
- 本文件与架构地图的锚点会随重构漂移（尤其拆 `sphere_engine.py` 巨石后）——改代码动了行段，顺手修这里。
