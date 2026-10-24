# J2 Track A · reader 接口变更单（FROZEN · 随 J2 rev2 冻结）

> **状态**：**FROZEN（2026-10-22，评审冻结；原 PROPOSED 返工版经端到端复验通过）**。
> 按执行件 §2.2「AV 侧（reader）」实现；评审台账 §36.4 退回后已返工；
> 评审核验 PASS，已在 J2 rev2 冻门栈登记新哈希（见 `gate/J2_prereg_rev2_20261022.md` §六/§十）。
> **依据**：`内部汇报/裁定执行-20261022-J2-rev2路线.md` §2.2（reader）/§2.5（更正）/§2.4（冻门栈影响）；
> 台账 §34/§35/§36；J2 预注册 rev1.1 `gate/J2_prereg_20261021.md` 与 rev2 `gate/J2_prereg_rev2_20261022.md`。
> **分工**：A2 负责 runner（区间累积导出 + 60 天双模式冒烟）；本件是判定侧 reader 的第二半。
> **分支**：`feat/atmo-tracka`（worktree `<worktree>`），基线 main `af247ec`。
>
> **返工记录（2026-10-22 晚，评审 §36.4）**：原交付 `2b8fe67` 被退回三项——
> ① `CONS-WATER` 把 `w_ser`/`days` 跟着切到 `[1:]`，比较窗错开一个区间；
> ② interval 数据缺登记键时静默按 instant 口径读；
> ③ 未拦截五键错放在 `manifest.export`（A2 `run_j2.py` 旧版）。
> 本版：W/days 改**全窗**；`flux_valid_from_snap≠2` 与 export 错位均 **fail-loud**；
> 接口单更正；补 3 条测试并让切片测试能真正识别 W 窗口错位。

## 一、为什么要改（仪器问题，不是判据问题）

旧口径 CONS-WATER 用快照上的**瞬时** `pr`/`evap` 做时间平均。实测采样噪声：E−P 序列
std 0.0706 mm/d、ac1 0.0248 ⇒ N=46 时标准误 ≈ **2.92e-3 相对 P**，**高于阈值 1e-3**；
实测 2.921e-3/6.213e-3 与噪声底同量级 ⇒ 无法区分「模型水账不闭合」与「采样噪声」。
Track A 让 A2 导出**区间平均率**；reader 必须按新数据语义取数，否则判据仍吃错窗口。

**改的是数据窗口，不是判据**：阈值、id、desc、PASS/FAIL 逻辑一字未动。

## 二、改动（2 个文件）

### 2.1 `jia_remap.py`：五个登记键白名单透传（顶层）+ 错位拒绝

```python
FLUX_META_KEYS = suites.FLUX_META_KEYS   # 单源：切片与透传共用
...
if wet:
    misplaced = [k for k in FLUX_META_KEYS if k in export and k not in mf]
    if misplaced:
        raise ValueError(f"flux 登记键只出现在 manifest.export 而顶层缺失: {misplaced}")
    for k in FLUX_META_KEYS:
        if k in mf:
            meta[k] = mf[k]
```

- **仅湿臂**（`benchmark=aquaplanet`）且 **manifest 顶层存在该键**时才写入目标 `meta.json`。
- **五键必须在顶层**；只出现在 `manifest.export` 而顶层缺失 ⇒ `ValueError`
  （专防评审 §36.4 缺陷 1：A2 旧版把五键写进 `manifest["export"]`，reader 静默丢键）。
- 干臂、湿臂 instantaneous 模式（无这些键）⇒ 目标 meta **不出现**这些键 ⇒ 产物逐位不变（§四）。
- A2 corrected 冒烟另带 `flux_last_interval_s`（区间长度诊断）；**不在契约五键内，reader 不搬运**
  （判据不读；如需进入目标 meta，另开接口变更单）。
- 本件只搬登记字段，不解释语义、不校验数值、不改判据。

### 2.2 `suites.py`：`flux_semantics=="interval_mean"` 时 pr/evap 类判据从第 2 个快照起（W 全窗）

```python
export_meta = meta.get("export")
if isinstance(export_meta, dict):
    misplaced = [k for k in FLUX_META_KEYS if k in export_meta and k not in meta]
    if misplaced:
        raise ValueError(f"flux 登记键错放在 meta['export'] 而顶层缺失: {misplaced}")
flux_semantics = meta.get("flux_semantics")
if flux_semantics not in (None, "instant", "interval_mean"):
    raise ValueError(f"未知 flux_semantics={flux_semantics!r}（只许 instant / interval_mean）")
if flux_semantics == "interval_mean":
    if meta.get("flux_valid_from_snap") != 2:
        raise ValueError("flux_semantics=interval_mean 需 flux_valid_from_snap=2（1-based）")
    flux_off = 1
else:
    flux_off = 0
pr_flux = stack("pr")[flux_off:]
evap_flux = stack("evap")[flux_off:]
...
wc = conservation.water_closure(pr_flux, evap_flux, w_ser, days, lat)
```

| 判据 | 取数变化 |
|---|---|
| `AQ-ITCZ-LAT` / `AQ-HEMI-DIFF` / `AQ-ITCZ-P` / `AQ-EP-SUBTROP` | 经 `pr_zm`/`evap_zm` 间接取 `pr_flux`/`evap_flux`（第 2..N 个快照） |
| `CONS-WATER` | `water_closure(pr_flux, evap_flux, w_ser, days, lat)` —— P/E 覆盖 `(day0, dayN]`（第 2..N 快照），柱水差 ΔW 用**全窗** `w_ser`/`days`（day0→dayN）。**不得**把 W/days 跟着切到 `[1:]`，否则比较窗错开一个区间（评审 §36.3/§36.4） |
| `CONS-WATER` note | 增 `flux=instant|interval_mean`（可追溯；instant 路径的 measured 值不变） |
| 能量判据（`CONS-ENERGY` 等） | **不动**：仍用末半段 `stack("sw_toa")[half:]` 等；A2 只累积 pr/evap，能量语义是否切换由评审在冒烟后裁（§36.5：rev2 仍用瞬时 canonical SW/LW） |
| 其他判据 | 不读 pr/evap，路径不变 |

**设计约束（评审请重点核）**：

- **offset 固定为 1，不读 `flux_valid_from_snap` 当偏移量**。判据取数窗口由判据栈写死，
  不由 manifest 字段驱动（避免输入侧自由度直接移动判据窗口）；该键仅**校验必须为 2** 并透传登记。
  `interval_mean` 但 `flux_valid_from_snap≠2`（含缺失）⇒ `ValueError`，不静默改窗口。
- `flux_semantics` 缺省或 `"instant"` ⇒ 与 main 逐位相同的旧口径。
- 未知值 ⇒ `ValueError`（数据契约边界 fail-loud，不静默按 instant 处理）。
- 五键只出现在 `meta["export"]` 而顶层缺失 ⇒ `ValueError`（与 `jia_remap` 同源检查，
  防转换器与判据器两侧任一环节错位）。

### 2.3 未改（冻结面）

`criteria.py`/`metrics.py`/`conservation.py`/`loader.py`/`run_hs.py` **零改动**；
阈值/id/desc 零改动；套件 verdict 逻辑零改动。

## 三、哈希（LF 归一 sha256）

| 文件 | main `af247ec`（变更前） | 本分支（返工后） | 性质 |
|---|---|---|---|
| `benchmarks/atmo/suites.py` | `c204ca387f9030f2026a3ebe17e04789ca0893b4da52881f6fa36cfcee737c83` | `07a5cb603865395078c8156429fb9f25d8c635f341755dd77a33bae72fdc37a5` | flux 语义切片 + W 全窗 + fail-loud |
| `benchmarks/atmo/jia_remap.py` | `627cf7255fa88b1a519493c41c40cd27860cb197e55c2a2b727e8b3ac4ddfd6c` | `2159b3739ce97cf1d98e287a33ba999d1e26e1c4b0f56dca778c8515774fd512` | meta 白名单透传 + export 错位拒绝 |
| `tests/test_atmo_jia_remap.py` | `28718bed3b7694e675fae4e2ccd0a7062300061a1e5ae6157abf9de1f3a707fb` | `42e835126b88751734b131db72c4f081590e01503642d9eb594265a0bd5dd55b` | +5 例（返工 +1） |
| `tests/test_atmo_suites.py` | `37ac93f2c16a39e48805fe59a6311d31a4baa82089f3c04e14bcfa64d5bc0db0` | `a27554ef2d2d0702a76fc6cfe457a8eade108fe7bc45158cdc1d1299a2dc4b39` | +4 例（返工 +2，其一参数化 2 例） |

**冻结五文件不变**（与 main 逐位相同）：

| 文件 | sha256（LF 归一） |
|---|---|
| `criteria.py` | `f80dd8029843c73fc09b5dda65020e849c1e8527a89f44cfb5413ddb4a81ac52` |
| `loader.py` | `18c2f094fa85c7a05b8de6dad8b07ca658421b72ff9e878727234292947df7af` |
| `metrics.py` | `aaa0032b88d436fab6e0ba5d8f80ae59375fa9234946baf27c963b79524ca0c2` |
| `conservation.py` | `8ffbb0a522f52c15516150adb8811ba5cc9bec96ae8deae2d4b7c75f371d5eb8` |
| `run_hs.py` | `e1c069071a9d75deecdaf066f002157893e9a35d27d93820e1b6d4e3c6607fb1` |

## 四、干臂逐位不变（真实 J1′ 46 快照）

输入 `<worktree>/_rerun_logs/20261021-A1-J1prime/native_run1`（46 快照）。
改前 = main 冻结转换器产物；改后 = 本分支转换器产物；逐文件比对：

| 项 | 读数 |
|---|---|
| 文件集合 | 48 / 48（`coords.npz` + `meta.json` + 46 `snap_*.npz`） |
| 数组逐位（`tobytes()`，含 dtype/shape/键序） | **0 差异** |
| `meta.json` 字节 | **完全相同** |
| 转换自证 | `self_check_bad=0`；`mu_coverage_defect=2.6678139831372082e-15`；`cfl_native_max=0.04905326421078971` |

证据脚本与输出：`_rerun_logs/20261022-AV-tracka/compare_dry.py`、`compare_dry.json`
（`{'files': 48, 'differing': 0, 'examples': [], 'meta_bytes_equal': True}`）。

另有两例显式钉住：

- `test_dry_ignores_flux_meta_keys_bitwise` —— 干 manifest **塞入**五个 flux 键，
  目标 `meta.json`/`coords.npz`/`snap_00001.npz` 仍与不带键的转换**逐字节相同**；
- `test_wet_flux_meta_absent_when_not_declared` —— 湿 manifest 未声明 ⇒ 目标 meta 不出现这些键。

## 五、测试（主交付 +6，返工 +3；两文件现 62 例）

`tests/test_atmo_jia_remap.py`（50 例；主交付 +4，返工 +1）：

1. `test_wet_flux_meta_passthrough` —— 五键逐键精确透传；
2. `test_wet_flux_meta_absent_when_not_declared` —— 未声明 ⇒ 无键；
3. `test_dry_ignores_flux_meta_keys_bitwise` —— 干臂逐位不变（§四）；
4. `test_wet_manifest_rejects_flux_keys_nested_in_export`（**返工**）—— 五键只在
   `manifest.export` 而顶层缺失 ⇒ `ValueError`（真实复现评审 §36.4 缺陷 1）；
5. 既有 45 例全部保持。

`tests/test_atmo_suites.py`（12 例；主交付 +2，返工 +2）：

6. `test_aqua_interval_flux_uses_snapshots_from_second`（**返工重写**）——
   合成 10 快照，首快照 `pr=0`、`evap` 极端非对称；`q_scales=[1]+[2]*9` 令 W 在首快照后翻倍。
   `interval_mean` run 与丢首快照 control 在 `AQ-ITCZ-LAT/AQ-HEMI-DIFF/AQ-ITCZ-P/AQ-EP-SUBTROP`
   四条上逐位相等；`CONS-WATER` 与**直接全窗重算** `water_closure(pr[1:], evap[1:], w_ser, days)` 逐位相等，
   且与错切窗 `w_ser[1:]/days[1:]` 的读数显著不同（错切窗 `dw_dt=0`，正好掩盖 W 跃变）；
   `instant` run 在 `AQ-ITCZ-P`/`CONS-WATER` 上读数不同 ⇒ 切片非空转；
7. `test_aqua_interval_requires_valid_from_snap_2`（**返工**，参数化 2 例）——
   缺失或 `=3` ⇒ `ValueError`（`flux_valid_from_snap`）；
8. `test_aqua_flux_keys_nested_in_export_rejected`（**返工**）—— 五键只在 `meta['export']`
   而顶层缺失 ⇒ `ValueError`；
9. `test_aqua_unknown_flux_semantics_rejected` —— `flux_semantics="cumulative"` ⇒ `ValueError`。

**实测**（本机，Windows）：

| 范围 | 读数 |
|---|---|
| `tests/test_atmo_jia_remap.py` | **50 passed / 0.59 s** |
| `tests/test_atmo_suites.py` | **12 passed / 0.62 s** |
| atmo 全组（10 文件） | **210 passed / 2.74 s** |
| 档② `-m "not legacy and not slow"` | **1582 passed / 10 skipped / 48 deselected / 121.17 s**，失败集合 **∅**（r110 未触发） |

## 六、A2 输入契约（reader 依赖，不满足 ⇒ 拒绝/报 owner）

1. `flux_mode=:interval` 时 manifest 声明 `flux_semantics="interval_mean"`、
   `flux_interval_s=snap_every_day*86400`、`flux_accumulation="rate×dt per step; Float64 accumulator"`、
   `flux_valid_from_snap=2`、`flux_first_interval_s`（实际首区间长度）。
2. 快照 i（i≥2）对应区间 `(day_{i−1}, day_i)`；首快照区间从 step 0 起，只作占位。
3. `pr`/`evap` 仍为 `kg m⁻² s⁻¹`，符号原样（不取绝对值、不改号）。
4. 大气状态场与 `:instant` 模式逐位相同（导出模式不影响物理）——由 A2 冒烟证明。

## 七、真实 A2 数据核对（2026-10-22 晚）

**旧布局硬拒**（`trackA_smoke_prev_25d186f`，五键在 `manifest.export`）：
新转换器对 `interval` 抛 `ValueError: flux 登记键只出现在 manifest.export 而顶层缺失:
['flux_semantics', 'flux_interval_s', 'flux_accumulation', 'flux_valid_from_snap',
'flux_first_interval_s']`；`instant` 正常转换，目标 meta 无 flux 键（0 键）。
证据 `_rerun_logs/20261022-AV-tracka/realdata_check.json`。

**corrected 冒烟**（`trackA_smoke/{instant,interval}`；A2 `2a49ce6`；60 天 / 60 快照；五键顶层）：
本 reader 独立转换（未用 A2 `trackA_smoke/remap`），`self_check_bad=0`，目标 meta 五键逐值 = 源 manifest。
CONS-WATER 实测（**只报读数**；60 天不是门窗口，不判 J2）：

| 口径 | measured | residual (kg m-2 s-1) | dw/dt (kg m-2 s-1) |
|---|---:|---:|---:|
| interval，W/days **全窗**（本 reader） | **1.4379578373678941e-3** | −5.616578550457868e-08 | −2.1384608349037356e-07 |
| interval，W/days 错切 `[1:]`（对照，非门件） | 1.4325699735580082e-3 | −5.595533872011116e-08 | −2.1405653027484108e-07 |
| instant（旧口径） | 1.3878085098131723e-3 | −5.423480473528041e-08 | −2.1384608349037356e-07 |

三项与 A2 `trackA_summary.json` 的 `aligned_p1_Wfull` / `spec_literal` / `old_instant_all`
**逐位一致**（独立转换路径）。W 序列范围两模式一致：**30.935898657356802–32.42234505962794 kg m-2**。
证据 `_rerun_logs/20261022-AV-tracka/realdata_read.json`。

两点边界：

- A2 manifest 另带 `flux_last_interval_s`（第六键，区间长度诊断）；契约仍为五键，
  reader **不搬运**该键，判据不读（已在 §2.1 声明，非静默丢弃）。
- 60 天读数 >1e-3 **不构成闭合失败结论**（判据定义在门 120 天 / 46 快照）；
  本件不据此调阈值、不判 PASS/FAIL。

## 八、复现命令（AV-j2 worktree 根）

```bash
PYTHONIOENCODING=utf-8 ../../.venv/Scripts/python.exe -m pytest \
    tests/test_atmo_jia_remap.py tests/test_atmo_suites.py -q

PYTHONIOENCODING=utf-8 ../../.venv/Scripts/python.exe \
    _rerun_logs/20261022-AV-tracka/compare_dry.py \
    _rerun_logs/20261022-AV-j2/dry_before \
    _rerun_logs/20261022-AV-tracka/dry_after

PYTHONIOENCODING=utf-8 ../../.venv/Scripts/python.exe -m pytest tests/ -q \
    -m "not legacy and not slow"
```

## 九、评审核验清单

1. §三 四个变更文件哈希与本文一致；五个冻结文件哈希与 main 相同；
2. 干臂 48/48 逐位（可复跑 §八 命令）；
3. 切片仅在 `flux_semantics=="interval_mean"` 时发生，`instant`/缺省路径与 main 逐位一致；
4. `interval_mean` 但 `flux_valid_from_snap≠2`（含缺失）⇒ `ValueError`；五键只在
   `meta["export"]`/`manifest.export` 而顶层缺失 ⇒ `ValueError`（两侧同源检查）；
5. `CONS-WATER` 用全窗 `w_ser`/`days`；测试能区分全窗与错切窗（合成 W 跃变 + 直接重算）；
6. 能量判据路径零改动（rev2 仍用瞬时 canonical SW/LW，§36.5）；
7. 阈值/id/desc 零改动；
8. 真实 corrected 冒烟读数：interval 全窗 **1.4379578373678941e-3**（与 A2 `trackA_summary.json`
   的 `aligned_p1_Wfull` 逐位一致）；旧布局 interval 被硬拒；
9. 冻结权在评审/owner，本件不自冻；本件不判 J2 PASS/FAIL。

## 十、评审冻结记录（**FROZEN**，2026-10-22）

**冻结人**：评审（总审）｜**依据**：本件 §九 九项核验 + 评审独立端到端复验（`_rerun_logs/20261022-review-tracka-verify/`）。
**状态变更**：`PROPOSED` → **`FROZEN`**，随 `gate/J2_prereg_rev2_20261022.md` 一同生效。

### 10.1 独立复验读数（评审，非 AV 报告转抄）

- corrected interval 冒烟经本 reader 独立转换：目标 `meta` 五键逐值 = 源 manifest，`self_check_bad=0`；
  `CONS-WATER`（全窗 W）**1.4379578373678941e-3**，与直接全窗重算逐位一致；
  错切窗对照 **1.4325699735580082e-3**；instant **1.3878085098131723e-3**；note 均带 `flux=` 标签。
- 旧布局（五键在 `manifest.export`）interval ⇒ `ValueError` 硬拒。
- A2 重冒烟 vs 返工前：10 场 ×60 快照 ×2 模式 **1,200/1,200 逐字节相同**；npz **120/120**；
  5 个区间调试数组逐位相同；`config_hash` 不变。
- 干臂真实 J1′ 46 快照：**48/48 文件逐字节相同**。
- 四个变更文件哈希与本件 §三一致；五个冻结文件与 main 同哈希。
- 档②（合并后 main `f926d10`）：**1582 passed / 0 failed / 10 skipped / 48 deselected，115.72 s**，失败集合 **∅**（r110 未触发）。AV 分支预跑曾出现 1 例 r110（隔离复跑 1 passed），按已知并发 flake 单名对账。

### 10.2 合入与哈希登记

- 本分支 `feat/atmo-tracka` @ `7593b2c` 已由评审合入 main，merge **`f926d10`**（冲突仅 `内部团队文档/任务队列.md`，取主版后评审更新 AV 行）。
- rev2 冻门栈登记：`suites.py` `07a5cb60…37a5`、`jia_remap.py` `2159b373…d512`；五个冻结文件哈希不变。
  完整表见 `gate/J2_prereg_rev2_20261022.md` §六。
- 本件冻结**不改**阈值/id/desc，不构成 J2 判读、不宣布 PASS/FAIL；后续任何改动 = 新提交 + revision。
