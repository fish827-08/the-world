# J2 湿场 remap 扩展提案（PROPOSED · 待评审批）

> **状态**：**PROPOSED（2026-10-22）**，未合入 main、未进冻门栈。按 J2_prereg §6 的
> 「R-8 同模式 + 冻门栈 bump」流程走评审：**核验默认路径逐位、测试、阈值零改动**后合入，
> 再由评审登记新的冻门栈（main hash + 六文件 sha256 + 映射件新哈希）。
> **分支**：`feat/atmo-j2`（worktree `<worktree>`），基线 main `cee19fd`。
> **判据阈值/id/desc 一字未改**：`criteria.py` sha256 仍为 `f80dd80…1ac52`；
> `loader.py`/`metrics.py`/`conservation.py`/`run_hs.py`/`suites.py` 与 main 逐位相同（§六）。
> 本件只改**输入映射与契约层**，不碰任何生产代码。

## 一、问题：J1 冻结映射件在湿臂上的四个硬阻塞

J2 的真源 manifest（A2 `_rerun_logs/20261022-A2-J2/native_geom/manifest.json`，只读参考）：
`benchmark=aquaplanet`、`grid.nlayers=8`、`sigma_half=[0,.05,.14,.26,.40,.55,.70,.85,1]`、
`export.snap_fields_3d=[u,v,T,q]`、`snap_fields_2d=[ps,sst,pr,evap,sw_toa,lw_out]`、
`forcing={}`、`ic.name=j2aqua_v2`、`slab_heat_capacity=2.092e8`、`radiation`/`moist_scheme` 非空。

| # | 现有 `jia_remap.py`（J1 冻结版）行为 | 湿臂后果 |
|---|---|---|
| 1 | 字段集固定 `u,v,T,ps,lat,lon` | 转换产物缺 `q/pr/evap/sw_toa/lw_out/sst` ⇒ `loader.load_run` 直接拒（`快照缺场`）；自证也不覆盖湿场 |
| 2 | 无条件要求 HS94 十二键 + 协议三键；`ic.name ∈ {iso_w1,teq_noise}` | A2 湿 manifest（`forcing={}`、`ic.name=j2aqua_v2`）在入口被误拒 |
| 3 | `sigma_edges` 由 `sigma` 相邻中点反推 | **与真层界不符**：J2 阶梯下中点反推 = `[0,.06,.1475,.265,.4025,.55,.70,.85,1]`，真值 = `[0,.05,.14,.26,.40,.55,.70,.85,1]` ⇒ 最底层厚度 **0.06 vs 0.05（+20%）**，直接偏置 CONS-WATER 柱水积分 |
| 4 | `sigma_edges` 重建式不读 manifest 层界、`nlayers` 只用于 reshape | J2 后续 16–24 层无法按真层界积分；层厚错配不报错 |

## 二、设计（R-8 同模式：默认路径逐位不变 + 显式湿分支）

| 位置 | 改动 |
|---|---|
| 常量 | 增 `WET_SNAP_FIELDS_3D=(u,v,T,q)`、`WET_SNAP_FIELDS_2D=(ps,pr,evap,sw_toa,lw_out,sst)`、`WET_SNAP_FIELDS`、`BENCHMARKS=("held_suarez","aquaplanet")`；`SNAP_FIELDS` 原样保留 |
| `_fields_for(mf)` | benchmark → (3D 字段, 2D 字段)；未知 benchmark ⇒ `ValueError` |
| `_validate_manifest` | 先做原通用块校验；`benchmark=aquaplanet` 走湿分支（`forcing` 为对象即可空；`ic.name`/`ic.sha256` 非空；调 `_validate_wet_manifest`）后返回；`held_suarez` 保序执行原 HS94/协议/IC 检查，**一字不改** |
| `_validate_wet_manifest` | 必含 `slab_heat_capacity`（有限正数）/`radiation`（非空串）/`moist_scheme`（非空对象）；`grid.nlayers≥1`；`sigma_half` 一维 `nlev+1`、有限、严格增、首尾 0/1（±1e-12）、中点与 `sigma` 一致（1e-9）；`export.pr_units`/`evap_units == "kg m-2 s-1"`；可选 `snap_fields_3d/2d` 按**集合**比对（A2 的 2D 顺序与契约顺序不同，不比顺序） |
| `_read_native_snap` | 按 `_fields_for` 读 `f3+f2+lat/lon`；3D reshape `(npoints,nlayers)`、2D/坐标校验长度；`log(Pa)→exp` 不变 |
| `native_sha256(d, fields=SNAP_FIELDS)` | 目录形态按字段序哈希；npz 形态仍哈希整文件字节；**默认参数 = 原行为** |
| `convert_native_run` | `coords.sigma_edges`：湿臂 **直接取 `manifest.grid.sigma_half`**，干燥臂仍走原中点式（逐位不变）；快照 payload 按 `f3+f2` 写；meta 湿臂增 `moist_scheme` 与 `mapping.sigma_edges_source`；报告增 `benchmark/fields_3d/fields_2d` |
| **未改** | `remap_field`/`mu_bands`/`target_grid`/`native_area_mean`/`native_cfl_max`/`bounds_excess`/`area_mean_target`/`ring_of`/`materialize_ic`（干燥 IC 协议）/`jia_smoke.py`；**不导出 `mu_edges`**（目标仍等 Δμ ⇒ v1 中点权精确，P1′ 的 CONS-MASS 路径不动） |

## 三、默认路径逐位不变（真实 J1′ 数据证明）

输入 = A1′ J1′ 原生真跑 `_rerun_logs/20261021-A1-J1prime/native_run1`（46 快照）。
改前（main `cee19fd` 的冻结转换器）与改后（本分支）各产出一个 loader run 目录，逐文件比对：

| 项 | 读数 |
|---|---|
| 文件集合 | 48 / 48（coords + meta + 46 snap） |
| 数组逐位（`tobytes()`，含 dtype/shape/键序） | **0 差异** |
| `meta.json` 字节 | **完全相同** |
| `cfl_native_max` | 0.04905326421078971（前后相同） |
| `mu_coverage_defect` | 2.6678139831372082e-15（前后相同） |
| `self_check_bad` | 0（前后相同） |

> 干燥臂不读 `sigma_half` 也有显式钉住：`test_dry_path_ignores_sigma_half` 给干燥 manifest
> 塞入错误的 `sigma_half`，转换仍按原中点式产出。

## 四、湿臂实测（合成场 × A2 真实几何）

几何取自 A2 `native_geom`（只读）：**3168 点 × 8 层**、48 环、∣lat∣max=87.1591°、
`sigma_half=[0,.05,.14,.26,.40,.55,.70,.85,1]`。场为解析构造 + 固定种子噪声（3 快照；
合成场读数无物理意义，只作**管线可执行性**证据）：

| 项 | 读数 |
|---|---|
| 转换 | rc=0，`self_check_bad=0`，`mu_coverage_defect=2.6678e-15` |
| `coords.sigma_edges` | **与 `manifest.grid.sigma_half` 逐位相等**；与中点反推**不相等** |
| loader | `wet=True`；3D `(8,48,224)`、2D `(48,224)`，10 字段齐全 |
| `suites.evaluate(AQUAPLANET)` | **15 行全部可执行**（CONS-NAN/MASS/CFL/POLAR-AMP 等在合成场上 PASS/FAIL 混合；AQ-ITCZ/CONS-WATER/CONS-ENERGY 的 FAIL 来自合成场不满足物理，不是转换缺陷） |
| 可变 nlev | 5 层几何（`sigma_half=[0,.1,.3,.6,.85,1]`）转换 + loader 通过，形状 `(5,8,32)` |

## 五、测试

`tests/test_atmo_jia_remap.py` **+10 例（36 → 46）**，`46 passed / 0.48 s`：

1. `test_fields_for_routes_by_benchmark` — 干/湿/未知 benchmark 路由。
2. `test_wet_convert_loader_valid_and_sigma_half_is_source` — 湿产物 loader 合法、σ 层界来自 `sigma_half`、meta 登记 `moist_scheme`。
3. `test_wet_variable_nlev` — 5 层几何端到端（不写死 8）。
4. `test_wet_all_fields_conserved` — 10 个湿场逐场 native↔target 面积均值一致（1e-11 相对）。
5. `test_wet_flux_sign_passthrough` — `pr` 负值原样透传（不取绝对值/不改号）。
6. `test_wet_manifest_rejections` ×10 parametrize — 缺 `sigma_half`/`moist_scheme`/`radiation`/`slab`、错 `pr_units`/`evap_units`、`sigma` 与 `sigma_half` 不匹配、空 IC sha、`nlayers=0`、未知 benchmark。
7. `test_wet_rejects_missing_field` — 快照缺 `q` 入口即挡（`缺场`）。
8. `test_dry_path_ignores_sigma_half` — 干燥臂不读 `sigma_half`（J1 冻结兼容）。

组读数：`tests/test_atmo_*.py` **201 passed / 2.71 s**；档② `-m "not legacy and not slow"`
**1573 passed / 10 skipped / 48 deselected / 110.81 s**，失败集合 **∅**（r110 本次未触发）。

## 六、变更文件哈希（sha256，LF 归一）与冻门栈

| 文件 | main `cee19fd` | 本分支 | 判定 |
|---|---|---|---|
| `criteria.py` | `f80dd8029843c73fc09b5dda65020e849c1e8527a89f44cfb5413ddb4a81ac52` | 同左 | **未改（阈值/desc）** |
| `loader.py` | `18c2f094fa85c7a05b8de6dad8b07ca658421b72ff9e878727234292947df7af` | 同左 | **未改** |
| `metrics.py` | `aaa0032b88d436fab6e0ba5d8f80ae59375fa9234946baf27c963b79524ca0c2` | 同左 | **未改** |
| `conservation.py` | `8ffbb0a522f52c15516150adb8811ba5cc9bec96ae8deae2d4b7c75f371d5eb8` | 同左 | **未改** |
| `run_hs.py` | `e1c069071a9d75deecdaf066f002157893e9a35d27d93820e1b6d4e3c6607fb1` | 同左 | **未改** |
| `suites.py` | `c204ca387f9030f2026a3ebe17e04789ca0893b4da52881f6fa36cfcee737c83` | 同左 | **未改** |
| `jia_remap.py`（映射件，非判据栈） | `bae272a86b24cec59deac8c97dd57ee89d0c9c2c2b7126c4c770c275129d2bf0` | **`627cf7255fa88b1a519493c41c40cd27860cb197e55c2a2b727e8b3ac4ddfd6c`** | 改（本提案） |
| `tests/test_atmo_jia_remap.py` | `3fe2a8bb048981044cc24b6f8d7590f0b00f758577014f8aed74dca039f81a36` | `28718bed3b7694e675fae4e2ccd0a7062300061a1e5ae6157abf9de1f3a707fb` | 改（+10 例） |
| `jia_smoke.py`（冻结件） | `5300a7cf44c997ac763de94b12680af47720326863665eceb92496d534b30e1f` | 同左 | **未改** |

## 七、评审合入后的冻门栈 bump（建议操作清单）

1. 评审核验本件 → 合入 main（AV 侧不 rebase）。
2. 台账新登记：**main 新 hash + 六文件 sha256（均不变）+ 映射件 `627cf72…`**；
   `gate/J2_prereg_20261021.md` §冻门栈表加一行映射件哈希（阈值表不动）。
3. `gate/J1_mapping_freeze.md` 状态区补一句指路："J2 湿臂扩展见
   `gate/J2_wet_remap_extension_20261022.md`；J1 干燥臂仍按本件"。
4. 合入后 A2 门跑只认新冻门栈；J1/J1′ 历史读数不追溯。

## 八、未做 / 开放项（供评审判）

1. **q / 通量重映射的口径**：映射逐层/逐场保**面积**均值；CONS-WATER 在目标网格上用 `ps` 加权积分。
   `ps≈1e5 Pa` 近均匀 ⇒ 差异二阶小，但**未加质量权修正**（加了就是新的映射自由度，须单独立项）。
2. **无垂直重映射**：`q` 必须与 `T` 同一 σ 阶梯；A2 若改 16–24 层，转换器随 manifest 走，
   但跨阶梯比对不支持。
3. `moist_scheme` **只登记不校验语义**（AQ-FORCING 只查存在性）；方案等价性仍由 A2 自证 + 评审抽核。
4. **湿臂 IC 独立复核不在本件**：转换器只要求 `ic.name`/`ic.sha256` 非空并抄进 `meta.initial_state`；
   AV 独立材料化 `j2aqua_v2` 并比对 sha 是 J2 判读流水线的下一步（等 A2 的 IC 规格冻结后做）。
5. `pr/evap` 符号**原样透传**：若 A2 的符号约定与判据解释不一致，属判读发现，不在映射层"修好"。
6. 不导出 `mu_edges`（目标等 Δμ，v1 中点权精确）；若未来目标改 Gaussian，按 J2_prereg §6 另立
   R-8 同模式件 + 再 bump 冻门栈。
7. 转换器不检查 46 快照/120 天等运行规格（那是判读前置检查的职责，loader/suites/run 侧已有 fail-loud）。

## 九、复现命令（本机）

```bash
# 干燥臂逐位（改前基线已存于 _rerun_logs/20261022-AV-j2/dry_before）
PYTHONIOENCODING=utf-8 ../../.venv/Scripts/python.exe -m benchmarks.atmo.jia_remap \
  --native ../..<worktree>/_rerun_logs/20261021-A1-J1prime/native_run1 \
  --out _rerun_logs/20261022-AV-j2/dry_after2
# 湿臂合成场（真实 A2 几何 3168×8）
PYTHONIOENCODING=utf-8 ../../.venv/Scripts/python.exe -m benchmarks.atmo.jia_remap \
  --native _rerun_logs/20261022-AV-j2/wet_realgeom/native \
  --out _rerun_logs/20261022-AV-j2/wet_realgeom/run
# 测试
PYTHONIOENCODING=utf-8 ../../.venv/Scripts/python.exe -m pytest tests/test_atmo_jia_remap.py -q
PYTHONIOENCODING=utf-8 ../../.venv/Scripts/python.exe -m pytest tests/test_atmo_*.py -q
```
