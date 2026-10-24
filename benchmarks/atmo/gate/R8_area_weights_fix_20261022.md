# R-8 修正：`area_weights` 精确带边（ACCEPTED · 已冻门栈）

> **状态**：**ACCEPTED（评审 2026-10-21 独立核验后合入 main，merge `1eac6dd`）**；门栈六文件哈希见台账 §二十五 25.3。
> 原提交为 2026-10-21 夜间 PROPOSED；文件名按派工 `…_20261022`。
> **不追溯 J1**（J1 目标网格等 μ，v1 中点重建精确，FAIL 定案不变，`gate/J1_signoff_20261021.md`）。
> **阈值/id/desc 一字未改**：`criteria.py` sha256 仍为 `f80dd80…1ac52`。
> **分支**：`feat/atmo-r8`，基线 `acbae4a`（= `d6604c1` + 派工/台账文档；代码与 `d6604c1` 相同）。

## 一、问题与独立复现

v1 `area_weights` 用**行中心相邻中点**反推 μ 带边：对等 μ / 等 Δφ 网格精确，对 **Gaussian 原生网格只是近似**
（带边最多偏 ~0.7°）。后果：全球 ps 均值有**时变伪偏差** ⇒ CONS-MASS 伪漂移 1e−6 量级。

在 A1 Plan B 失败复现目录 `_rerun_logs/20261021-planb/repro-a`（32×64 Gauss 节点，30 快照）上独立复算
（`lat` 与 `asin(leggauss(32))` 最大差 2.84e−14°；`repro-a` day11 起 ps 含 NaN ⇒ 全期 `mass_drift`=NaN，
下表取到最后一个全有限日）：

| day | v1 中点带边（冻结口径） | 精确 Gauss 权（本次修正） |
|---|---|---|
| 1（全球均值 Pa） | **100000.045094** | **100000.000000** |
| 3 | 1.0111e−06 | 4.5256e−14 |
| 5 | 2.1107e−06 | 9.0658e−14 |
| 7 | 3.1401e−06 | 1.3606e−13 |
| 9 | 3.6376e−06 | 1.8132e−13 |

与台账 §23.3（评审独立复算）逐项一致。

## 二、修正设计

| 位置 | 改动 |
|---|---|
| `metrics.area_weights(lat_deg, mu_edges=None)` | 给 `mu_edges` 时返回 `np.diff(mu_edges)`；校验：1D、长度 `nlat+1`、全有限、严格升序、落在 `[-1,1]`，违规 `ValueError`（fail-loud）。**不给时逐位走 v1** |
| `conservation.ps_global_mean_series(ps, lat_deg, mu_edges=None)` | 透传给 `area_weights` |
| `conservation.mass_drift(ps, lat_deg, mu_edges=None)` | 透传给 `ps_global_mean_series` |
| `run_hs.py` coords 导出 | 增 `mu_edges=np.asarray(g["mu_edges"])`（`atmo_core.grid_spec` 已提供：`-1 + 2·cumsum(area)`，`area`=Gauss 权/2） |
| `suites.evaluate` CONS-MASS | `mu_edges = coords.get("mu_edges")` 透传；缺键（旧 run 目录）⇒ `None` ⇒ v1 **逐位不变**；行 note 记 `weights=mu_edges` / `weights=v1(midpoint)` |

⚠️ `atmo_core/` **未改动**（`grid_spec` 本来就导出 `mu_edges`，`atmo_core/src/grid.rs:124-133,257`）。

## 三、默认路径逐位不变（证明）

从基线 `acbae4a:benchmarks/atmo/metrics.py` 抽出 v1 函数独立执行，与新 `area_weights(lat)` 对比
（`np.array_equal`，含单行特例）——**全部逐位相同**：

| 网格 | 结果 |
|---|---|
| 等 μ 目标 48 行 | 逐位 |
| 等 Δφ 61 行 | 逐位 |
| Gaussian 原生 32 行 | 逐位 |
| Gaussian 原生 48 行 | 逐位 |
| 单行 `[10.0]` | 逐位（`[2.0]`） |
| 随机 37 行（seed 0） | 逐位 |

测试内固化同一证明：`tests/test_atmo_metrics.py::test_area_weights_default_path_bitwise_v1_three_grids`
（内联 v1 公式 + `np.array_equal`），不依赖 git 历史。

## 四、测试

`tests/test_atmo_metrics.py`（+5）：

1. `test_area_weights_default_path_bitwise_v1_three_grids` — 三网格默认路径逐位 + 单行特例。
2. `test_area_weights_mu_edges_match_gauss_weights` — 32 节点 `diff(mu_edges)` == `leggauss(32)` 权（atol 1e−15，Σ=2）。
3. `test_area_weights_mu_edges_match_atmo_core_grid` — **金标**：`area_weights(g["lat_deg"], mu_edges=g["mu_edges"])/2`
   == `atmo_core.grid_spec(32,64,8)["area"]`（atol 1e−15）；扩展未编译时显式 `importorskip(reason=…)`。
4. `test_area_weights_mu_edges_fail_loud` — 五种非法输入（长度/维度/NaN/非升序/越界）均 `ValueError`。
5. `test_mass_drift_gaussian_exact_weights_remove_pseudo_drift` — 合成 P2 振幅渐增场（Gauss 求积对 deg 2 精确）：
   v1 **3.612e−06**，精确带边 **1.46e−16**（断言区间 3.0–4.2e−6 / <1e−15）。

`tests/test_atmo_suites.py`（+1）：

6. `test_cons_mass_threads_mu_edges_old_dir_falls_back` — 同一 Gauss 32 节点 run 写两版 coords：
   缺 `mu_edges` ⇒ CONS-MASS **逐位等于** `mass_drift(ps, lat)`（3.612e−06）；带 `mu_edges` ⇒ **逐位等于**
   `mass_drift(ps, lat, mu_edges=…)`（<1e−15）；note 分别为 `weights=v1(midpoint)` / `weights=mu_edges`。

读数：`tests/test_atmo_*.py` **184 passed / 2.49 s**（原 178，+6）。

## 五、变更文件哈希（sha256, LF 归一）

| 文件 | 基线 `acbae4a` | 本分支 |
|---|---|---|
| `criteria.py` | `f80dd80…1ac52` | **`f80dd80…1ac52`（未改）** |
| `loader.py` | `18c2f09…47df7af` | **`18c2f09…47df7af`（未改）** |
| `metrics.py` | `b3172a4…e67033` | `aaa0032b88d436fab6e0ba5d8f80ae59375fa9234946baf27c963b79524ca0c2` |
| `conservation.py` | `310d726…3ccfc2` | `8ffbb0a522f52c15516150adb8811ba5cc9bec96ae8deae2d4b7c75f371d5eb8` |
| `run_hs.py` | `222d83e…43096` | `e1c069071a9d75deecdaf066f002157893e9a35d27d93820e1b6d4e3c6607fb1` |
| `suites.py` | `cb5f768…44c9e8` | `c204ca387f9030f2026a3ebe17e04789ca0893b4da52881f6fa36cfcee737c83` |

（`criteria.py`/`loader.py` 的当前工作树字节含 CRLF 属检出转换；上表按 `git show`/LF 归一统计，与台账 §24.5 同口径。）

## 六、对 Plan B2 门的影响（使用说明）

- 合入后跑出的 `coords.npz` **必含** `mu_edges`；CONS-MASS 自动走精确权，读数可与 `diag.mass_daily` 同量级比对
  （台账 §24.3 签门要求）。
- **旧目录**（无 `mu_edges`）行为与 v1 逐位相同 ⇒ 历史读数口径不变，不会因本次修正跳变。
- 15 条 v1 判据、阈值、`criteria.py` 均未动；R-7(c) v2 **不在**本门使用。
- 本次修正**不改变** Plan B2 的稳定性结论（2Δx 截断是另一回事）。

## 七、未改动 / 开放项（供评审判）

1. `water_closure` / `energy_closure`（湿套件）仍走 v1 中点带边；若未来在 Gaussian 原生网格上判
   CONS-WATER/CONS-ENERGY，同一伪偏差会进入面积平均。本派工范围只到 CONS-MASS，**未扩权修改**，列开放项。
2. `area_mean` 未加 `mu_edges` 参数（无调用方需要）；如湿套件跟进，一并接。
3. 本修正不重建历史 run（J1 不追溯；Plan B 未进门槛）。

## 八、复现命令（本机）

```bash
# 测试（AV-r8 工作树）
PYTHONIOENCODING=utf-8 ../../.venv/Scripts/python.exe -m pytest tests/test_atmo_*.py -q
# 真实数据复算（repro-a）
PYTHONIOENCODING=utf-8 ../../.venv/Scripts/python.exe - <<'PY'
import json, numpy as np
from pathlib import Path
from benchmarks.atmo import conservation
rd = Path("../..<worktree>/_rerun_logs/20261021-planb/repro-a")
lat = np.load(rd/"coords.npz")["lat"].astype(np.float64)
ps = np.stack([np.load(p)["ps"].astype(np.float64) for p in sorted(rd.glob("snap_*.npz"))])
nodes, gw = np.polynomial.legendre.leggauss(lat.size)
edges = np.concatenate(([-1.0], -1.0 + 2.0*np.cumsum(gw/2.0)[:-1], [1.0]))
print(conservation.mass_drift(ps[:9], lat), conservation.mass_drift(ps[:9], lat, mu_edges=edges))
PY
```
