# the-world

[English](README.en.md) | 中文

**确定性的球面数字生命仿真平台 —— Python 编排、Rust/PyO3 数值核与版本化科研记录。**

状态：**已结项的研究项目（2026-10-24 结项），维护态**。
许可：**Apache-2.0**。本项目不主张已出现语言、智能或「L0–L4」成功。

---

## 这是什么

`the-world` 在经纬球面上模拟一个最小数字生命种群（正式档 **480 × 960 = 460,800 格**）。
世界提供地形、季节性能量平衡气候、资源/信号/气味场与亚格移动；个体只有小型实数基因与
固定行为公式，**仿真回路不使用适应度函数，也不使用 LLM**。研究问题是：仅靠选择压力，
信号使用、记忆等行为结构能否自发出现并稳定。

项目已于 **2026-10-24 结项**。这里公开全部结果，包括失败——因为失败本身也是结果。

## 已经确立什么、没有确立什么

| 问题 | 结论 |
|---|---|
| 类语言信号的端到端涌现（L0–L4） | **未确立**。大气判据门 J1、J2 均 **FAIL**；J3 端到端耦合检验因 P0 前置未过而 **NOT-RUN（未执行）**，**零耦合数据**。 |
| 记忆/气味机制是否改变觅食 | **机制层有效**。四项机制层正结果如下；按裁定**不作为涌现证据**。 |
| 世界是否确定性可复现 | **是**。Python 与 Rust 双路径逐位一致，冻结 digest 可复算。 |

机制层正结果（口径与限定见 [docs/公开/02-科研账本-20261024.md](docs/公开/02-科研账本-20261024.md)）：

- **记忆扩大斑块覆盖**：S3 记忆批（跑在 s2 装置上）24/24 seed 减少未访斑块格（on−off 中位 −2.92pp）。
- **记忆提高承载量**：旗舰 60k 同 seed 配对 K 4/4 正向；前 3 组 +23.1%~+176.1%，seed 172
  为 82 → 18,634（分母接近灭绝，比值失真，应读绝对值）。更宽的旗舰 8 seed 集（A 支 5 + B 支 3）
  K 比值 1.231–227.244，其中 seed 172 的比值同样受该分母影响。B 侧主门未过（另记）。
- **气味使零食物沙漠可穿越**：批 A 主门 23/24。另有四通道新装置 E-048A 的 A 门 24/24，
  判据与装置均不同，**不得混用**。
- **记忆扩大个体活动范围**：4/4 seed，>5 格移动占比升至 4.7–7.5 倍。

负结果同样入账：g22 记忆权重扰动中性、g22 演化级纪元不足、LTO-1 未达检查点、旗舰 60k
B 主门未过、C 批自然档 fail、K 收敛仅 3/5、链式/黏附探测多数为负、多细胞（W4b）与威慑/
军备竞赛线（C3/B16）为负或功效不足、J1 FAIL、J2 FAIL。

结构性发现（最可复用的产出）：

1. **信号边际价值为零是构造性的**——信号内容是接收者本可直读的发射者状态，发射纯付出，
   选择因此关闭信号。
2. **表达力天花板**——有限维实数基因 + 固定行为公式只能张成有限维行为流形，与语言所需的
   离散组合无限性结构性不匹配。
3. **K=2「双稳态」是装置 bug**——由背景倍率抽奖造成，并非被测机制的效应。

## 架构

```text
world/            球面网格、光照/温度、资源/信号/气味场
simulation/       引擎与配置（单一状态机、确定性步进）
biosphere/        最小个体层（基因 → 发育 → 控制器）
sim_core/         Rust/PyO3 热路径，与 Python 参考实现逐位一致
atmo/             大气耦合接口（默认关闭）
atmo_bridge/      大气重映射/桥接辅助
atmo_core/        Rust 干 Held–Suarez 核（P1 FAIL 后冻结）
atmo_julia/       可选 SpeedyWeather.jl 集成（外部 EUPL-1.2 依赖）
s0_kernel/        零依赖、通道驱动的守恒测试参考世界
observatory/      实验矩阵、统计、WebSocket 快照
viz/              快照导出 + Vite/React/three.js 回放前端
benchmarks/       大气判据、指标、签门件
experiments/      一次性实验运行器（历史；部分需适配）
tools/            可复现探针、digest 与审计工具
tests/            三层 pytest 测试套
```

## 快速开始

需要 Python ≥ 3.10（已在 3.12 与 3.14 测试）、Rust 工具链与 `maturin`。

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 编译主线测试所需的 Rust 扩展
cd sim_core && maturin develop --release && cd ..

# 档②主线测试（该命令读数为 1,580 passed / 0 failed / 10 skipped / 48 deselected）
python -m pytest tests/ -q -m "not legacy and not slow" --ignore=tests/test_broker.py

# 小装置确定性冒烟（不是正式 480x960 档）
python -X utf8 experiments/run_bare_world.py --arm 4 --rows 96 --cols 192 --patches 40 \
    --ticks 100 --sample 25 --out _rerun_logs/bare_smoke
```

上面的 `--ignore` 排除了 6 个可选的 broker 测试：该命令读数为 **1,580 passed / 0 failed /
10 skipped / 48 deselected**（1,638 收集）。不忽略且装有 `fastapi`、`uvicorn`、`websockets`
时，同一集合读数为 **1,586 passed**（1,644 收集）。2026-10-24 复跑：空载 **111.4 s**；
共享机并发负载下同期实测 **130.5 s**（忽略 broker）/ **175.4 s**（含 broker），会超出
150 s 预算线——耗时随负载变化，不是性能保证。结项记录为 **109.36 s** 与 **140.23 s**
（1,644 收集口径）。

Windows：激活命令改为 `.venv\Scripts\activate`；若控制台代码页不是 UTF-8，命令前加
`PYTHONIOENCODING=utf-8`。

确定性校验（必须复现冻结 digest）：

```bash
python tools/env_field_digest.py --device s2 --mode classic   # 31d6acce…4072
python tools/env_field_digest.py --device s2 --mode default   # db9ccbb3…d295
```

优化路径需要编译扩展；`tests/test_broker.py` 还需要 `fastapi`、`uvicorn`、`websockets`。

## 数据与范围

实验数据（`.npz`、`.csv`、快照、图）**不在本仓**，存放在独立数据仓。本仓是**装置**，
不是证据包；证据读数与文件锚点汇总在 `docs/公开/` 的公开报告中。

本发行物是结项后的**精选快照**：包含源代码、测试、设计/预注册/签门件与公开报告；
内部运营材料（派工、团队管理、原始跑批日志、归档垃圾）不包含在内。

## 文档

- [docs/公开/README.md](docs/公开/README.md) — 公开报告索引与一分钟结论
- [01-最终汇报](docs/公开/01-最终汇报-20261024.md) — 结项汇报：结果、边界与教训
- [02-科研账本](docs/公开/02-科研账本-20261024.md) — 正负结果逐条汇集与证据等级
- [03-工程与复现](docs/公开/03-工程与复现-20261024.md) — 工程资产、构建、测试与复现
- [04-方法与纪律](docs/公开/04-方法与纪律-20261024.md) — 预注册、评审与审计纪律
- [05-大气线技术纪要](docs/公开/05-大气线技术纪要-20261024.md) — 大气判据门记录（J1/J2/J3）
- [06-发布与表述边界](docs/公开/06-发布与表述边界-20261024.md) — 可以写什么、不能写什么

## 许可与第三方组件

原创代码与文档采用 **Apache License 2.0**（见 `LICENSE`）。第三方组件不随仓分发，
由使用者自行安装并遵循各自许可。特别说明：

- **SpeedyWeather.jl**（可选大气集成）为 **EUPL-1.2**，不是 Apache-2.0；它只是外部依赖，
  不在本发行物中分发。
- **exoplasim**（GPL-2.0）**不属于本发行物**；对应的可选实验脚本已从公开快照中排除。
- Rust crates、npm 包与 Python 依赖均按各自许可使用；本仓声明的直接依赖为
  MIT/BSD/Apache-2.0 兼容许可。传递依赖与可选依赖须按其自身许可文本核对。

详见 `NOTICE` 与 `THIRD-PARTY-NOTICES.md`。

## 引用

引用本项目请使用 `CITATION.cff` 中的元数据：

> the-world contributors (2026). *the-world: a deterministic spherical digital-life simulation
> platform* (Version 1.0.0) [Computer software]. https://github.com/fish827-08/the-world

## 维护

项目已**结项**，仓库按只读精神维护：不新开实验、不重开 J1/J2 门、不重启 J3（已归档
NOT-RUN）、不追溯修改已冻结的判据与阈值。重启须另立预注册并经 owner 新裁定，且不得
重新解释本仓记录的结果。
