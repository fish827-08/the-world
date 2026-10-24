# A3 大气↔世界耦合层 · 设计页（A3.1）

> 线：A3（atmo 耦合）｜日期：2026-10-20｜基线 `main = eae404c`｜分支 `feat/atmo-coupling`
> 依据：**`atmo/INTERFACE.md` FROZEN v1.0.0**（唯一接口依据，7 条 OPEN 见其 §11）
> 评审台账：`内部技术文档/技术-评审清单与台账-atmo线-20261020.md`
> 本文只定 A3 的口径与接线；**不改** `atmo_core/`（A1）、`atmo/physics/`（A2）、`benchmarks/`（AV）、`world/`、`simulation/`。
> 跨线需要的改动集中在 §9《接口变更单提案》，批准前不动。

---

## 0. 一句话与范围

A3 只做三件事：**把世界边界场灌进 atmo、按世界 tick 推进 atmo、把 atmo 出口场送回世界**。
不实现动力（A1）、不实现物理（A2）；A1/A2 未实现的接口一律按 §8 签名写桩并**显式报未实现**，不返回假数据。

```
世界 tick t ──(每 k tick)──► 边界场 (h/mask/alb/sst/seaice) ──重映射──► atmo_core.step(dt_step)
     ▲                                                                        │
     └────────── 出口场 (t2m/precip/u10/v10/cloud/sw/lw) ◄──重映射 ◄──────────┘
                         │
                         ├─ t2m → ClimateSource.temperature_c（生物层/NPP）
                         └─ precip → EnvField.pr 空间型（A17/A18 构造期换源通路）
```

**默认关**：`AtmoConfig.enabled=False`（`weather_source="off"`）⇒ 不建 atmo、不重映射、不改任何世界场；
关档世界 digest 必须与基线逐位一致（`classic 31d6acce…4072` / `default db9ccbb3…d295`）。
`atmo/coupling/` 只依赖 numpy；关档即使被 import 也无副作用。

---

## 1. 时钟：M2′（tick 整数倍）

### 1.1 口径（INTERFACE §7.3，总审 OPEN-1 裁定）

```
k      = floor(dt_atmo / tick_seconds)        # 每 k 个世界 tick 交换一次边界
n_sub  = 1
dt_step = k * tick_seconds                    # 必须 ≤ dt_atmo
```

- 交换相位：**世界 tick `t` 满足 `t % k == 0`**（t=0 先装边界；t=k, 2k, … 交换）。
- atmo 时钟：`time_s = (t // k) * dt_step`，**整数倍 ⇒ 零漂移**（不使用浮点累加）。
- `k` 由 `tick_seconds` 与 `dt_atmo` **在构造期一次算定**，进 manifest；运行期不改。
- `dt_atmo > tick_seconds` 恒成立（36/360 两档都满足）；`step_mode="tick_split"`（M1）只留兼容开关，
  **不默认**（OPEN-1）。

### 1.2 两档读数（`[算术]`，年长按世界口径 365 d）

| `tick_seconds` | `dt_atmo` | **k** | `dt_step` | 交换/年（365 d） | 交换/年（365.25 d，INTERFACE 默认） |
|---:|---:|---:|---:|---:|---:|
| **36** | 900 | **25** | **900 s** | **35,040** | **35,064** |
| **360** | 900 | **2** | **720 s** | **43,800** | **43,830** |

> 🔴 **勘误提请**：INTERFACE §7.3 表注"36 s ⇒ 1.5 万步/年量级"与公式不符——按公式实算
> `31,536,000 / 900 = 35,040`（≈3.5 万步/年）。A3 以**公式**为准，建议总审在 v1.0.1 出勘误；
> 该条不影响任何字段/单位/索引语义，**不是**接口变更。

### 1.3 年长与季节相位（世界族参数）

`orbit` 从世界派生，保证季节相位不各写一份（INTERFACE §9.3 "世界侧对应"）：

| orbit 键 | 取值 | 说明 |
|---|---|---|
| `year_s` | `(light.season_period if >0 else time_scale.year_ticks) × tick_seconds` | 世界一年 = 365 d（`year_ticks=rotation_period×365`）；开季节时跟 `season_period` |
| `rotation_period_s` | `light.rotation_period × tick_seconds` | 36 s 档 = 86,400 s；360 s 档 = 86,400 s |
| `obliquity_rad` | `light.tilt_rad` | 世界无季节（默认 0）⇒ atmo 也先无季节；**`follow_world_season=False`** 时才用 INTERFACE 默认 23.4393° |
| `equinox_offset_s` | 0.0（北半球春分 = t=0） | 世界 `LightAndTemperature` 季节相位同源约定 |
| `eccentricity` | 0.0167（INTERFACE 默认） | 世界无对应字段 |

> `follow_world_season=True`（默认）是**世界族一致性**优先：世界的"一年/倾角"只有一个真源。
> 代价：世界默认 `tilt_rad=0` ⇒ atmo 首版无季节。要做"真实季节"必须同时改世界族参数，不在 A3 偷改。

### 1.4 确定性

- 不用 RNG；交换相位由整数 `t % k` 决定。
- 同 seed 两次 ⇒ 边界场、atmo 输入、出口场**逐位一致**（A3.3 单测钉死）。
- 重映射权重建一次、缓存；不在热路径重复构造。

---

## 2. 世界 → atmo 边界场（INTERFACE §5）

| atmo 键 | 世界来源（真源） | 派生口径 | 重映射 | 频率 |
|---|---|---|---|---|
| `h` (m) | `EnvField.h_m`、`sea_level_m`、`land_mask` | `h = where(land_mask, max(h_m − sea_level_m, 0), 0)`。**海洋恒 0**（§5）；湖/河暂按陆地 | 世界 480×960 → T21/T31，保守面积权 | 静态（每交换步重读同值） |
| `mask` (0..1) | `EnvField.land_mask`（bool） | `float(land_mask)`；**湖/河算陆地**（v0，写入风险） | 同上 | 静态 |
| `alb` (1) | `SURFACE_THERMOPHYSICS[surface_type][0]` | 按 `surface_type` 查表（陆 0.17 / 海 0.06 / 湖河 0.06） | 同上（面积权） | 静态 v0 |
| `sst` (K) | `LightAndTemperature.base_temperature(cells)`，仅海洋 | `K = °C + 273.15`。**只有海洋格有权重**：同时重映射 `ocean_mask`，`sst = remap(T_sea·ocean) / remap(ocean)`（分母=0 的 atmo 格用全球海温均值） | 同上（分子/分母两次） | 每交换步（v0 值静态） |
| `seaice` (1) | 常量 0 | INTERFACE §5 v0 允许恒 0 | — | 每交换步 |

**为什么这些来源**：
- 世界唯一有**米制海拔**的是 `EnvField.h_m`（C0b 开档）；`EnvField.h` 是 [0,1] 归一地形，**不能**当米用。
- 世界唯一有**海陆**的是 `EnvField.land_mask`（C0b 开档）⇒ 天气模式**硬前提** `env_field.c0b_enabled=True`。
- 世界**没有** SST/海冰/反照率字段：`alb` 用已有的 `SURFACE_THERMOPHYSICS` 查表；`sst` 用世界光照温度剖面（**占位，不是海表能量平衡**）；`seaice` 恒 0。
  ⇒ A2 陆面/海表能量平衡上线后必须换源（走变更单，MINOR）。

**重映射方向**：源 = 世界规则网格（升序南→北、经度 0..360），目标 = T21/T31 Gaussian。
**源面积权 = 世界 `cell_area`（cosφ 行权，归一化）**——与世界全局平均同一把尺。

---

## 3. 重映射（A3.2 骨架）

### 3.1 网格坐标

- **世界**：`row 0 = 最南`，`lat_r = −90 + (r+0.5)·180/rows`；`col 0 = 0°E`，`lon_c = c·360/cols`。
  面积权 = `SphereWorld.cell_area`（行常数 `cosφ`），归一化后 `Σ area = 1`。
- **atmo T21/T31**：`atmo_core.grid_spec()` 的 `lat_deg`（升序南→北）、`lon_deg`（0..360）、`area`（=`dmu/2`，`Σ=1`）。
- 🔴 **纬序守卫**：`Grid` 构造时只接受升序；若输入北→南（外部表常见）⇒ **翻转 lat 与 area 并置
  `flipped_from_descending=True`**，`RemapPair` 自动翻转场行。禁止静默南北镜像（A17 已踩坑）。
- 🔴 **atmo `lat_edges_deg` 不能用作面积边**：实测 T21 首行 `lat_edges_deg[0]=−90°, [1]=−82.4969°`
  是 **μ 中点边**，对应 μ 宽 0.00856；而该行真实高斯面积权 `2×area=0.0070186`（差 **22%**）。
  保守重映射的面积边必须由 **`cumsum(area)` 在 μ 空间生成**（`μ_0=−1, μ_j=−1+2Σ_{<j}area, μ_n=+1`），
  两端钉 ±90°。`lat_edges_deg` 只用于绘图/CFL。

### 3.2 权重口径（保守）

- 纬向：源/目标都用**各自声明的 row area**生成 μ 边界 ⇒ 交叠长度矩阵，按目标行归一化。
  ⇒ 对任意场 `f`：`Σ_j src.area_j·f_j = Σ_i dst.area_i·(Wf)_i`（机器精度）。
- 经向：周期角度交叠，口径同 `atmo_bridge.remap.lon_weights`（A17），跨 0/360 拆段；目标行归一化。
  T21 64 列 ↔ 世界 960 列 = 15 整除；T31 96 ↔ 960 = 10 整除 ⇒ 无插值损失。
- 二维 = 纬向矩阵左乘 + 经向矩阵右乘（可分离），不做逐格求交。
- 常值场：每行权重和为 1 ⇒ 偏差 ≤1e-12（目标 1e-15 量级）。
- 极区：Gaussian 无极点行；**纬向常值场**重映射后仍为经向常数、无 NaN（波状场只保证有限）。

### 3.3 A3.2 单测（`tests/test_atmo_coupling.py`，tier=fast）

1. 常值场双向偏差 ≤1e-12（T21 与 T31 参数化）。
2. 随机场双向**全球面积权均值**守恒 ≤1e-12。
3. 跨接缝（源经度错开 1/4 格 ⇒ 源首格跨接缝）：权重落在目标首/次列（0.75/0.25）、
   行和=列和=1（源格被完整分配）；波状场 + 纬向常值场过极点：无 NaN/Inf 且常值场保持经向常数。
4. `world.land_mask` → T21 再回世界：面积权陆地比例一致 ≤1e-12（真 `EnvField` 60×120）。
5. 北→南纬序输入：`flipped_from_descending=True`，输出与升序输入**逐位一致**。
6. 规则经纬网格上，纬/经权重与 `atmo_bridge.remap` 对拍 ≤1e-13（机器精度；纬度边界计算路径不同，非逐位）。
7. 非法输入 fail-loud：面积和≠1、纬度非单调、场形状不符 ⇒ `ValueError`。

---

## 4. atmo → 世界出口场（INTERFACE §10）

| atmo 键 | 单位 | 换算 | 世界消费点 | 状态 |
|---|---|---|---|---|
| `t2m` | K | `°C = K − 273.15` | ① `ClimateSource.temperature_c(flat,tick)`：生物 g11 温度偏好/冷惩罚（每 tick）、A20 动态 NPP（若开）；② **资源再生温度不走此接口**（见 §9-IC1） | 注入 `ClimateSource`（A16b 通路） |
| `precip` | kg·m⁻²·s⁻¹ | 报告用 `mm/yr = ×31,557,600`；**进世界只取空间型**（`keep_total`） | `EnvField.pr`：构造期 `precip_override=`（A18.3）；运行期 `set_precip_pattern()`（A17.3） | A2 有非零降水才激活；v0 `precip=0` ⇒ 不注入并记 `weather_precip_active=false` |
| `u10,v10` | m/s | 无 | 世界当前**无消费点**（`climate.wind()` 未被调用） | 只导出/进 probe |
| `cloud` | 1 | 无 | 无 | 只进 probe |
| `sw_sfc,lw_sfc` | W/m² | 无 | 无 | 只进 probe；A1 占位值必须标 `placeholder=true`，**不可作结论** |

**时间插值**：v0 = **零阶保持**（用最近一次交换的场），确定性、无额外状态；不做浮点时间插值（A3.4 评估后再定）。

### 4.1 为什么降水走 `keep_total`（不改全球水量）

`EnvField.pr` 是 **Magnus proxy 量纲**（hPa 量级），不是 mm/yr；其下游 `net = pr − q·et_frac` 也在这个量纲上。
把 atmo 的 mm/yr 原值直接写进 `pr` 会**静默改口径**（NPP 的 `precip_half` 同样按 proxy 量纲定）。
因此天气模式：

1. 构造期 `EnvField(..., precip_override=atmo_precip_pattern, precip_policy="keep_total")`；
2. 运行期 `set_precip_pattern(..., keep_total=True)`；
3. **只取空间型，全球总量仍由世界 proxy 自己的均值钉住**（不是 990 mm/yr 锚）。

### 4.2 🔴 撤掉预设降水锚点（A3.4 验收）

天气模式必须**同时**满足下列三条，否则构造期 fail-loud：

| 检查 | 为什么 |
|---|---|
| `config.atmosphere.enabled == False` | `world/atmosphere.py` 的 **P̄=990 mm/yr 锚**（`_P_ANCHOR_KG_M2_S`）只在 C4a-1 开档时生效；与 atmo 源同时开 = 两条降水口径 |
| `config.env_field.c1_enabled == False` | **C1 手画降水带**（副热带干槽 `band`）会乘在 `pr` 上；天气模式下降水空间型来自 atmo |
| `config.env_field.enabled == True` ∧ `c0b_enabled == True` | 没有 `pr`/`land_mask`/`h_m` 就无法灌边界；不静默退化成"均匀降水/全球一种地表" |

验收证据（A3.4 汇报必附）：
```bash
grep -rn "_P_ANCHOR_KG_M2_S\|990.0 / _SECONDS_PER_YEAR" world/atmosphere.py
grep -rn "c1_enabled" simulation/config.py world/env_field.py
# 运行期：eng.atmosphere is None ∧ env_field.pr_source == "atmo" ∧ env_field.band is None
```

---

## 5. 配置（`AtmoConfig`，默认关）

> 位置：`atmo/coupling/config.py`。**v0 不挂 `SimConfig`**（跨线改动见 §9-IC2）；由实验驱动显式构造。

| 组 | 键 | 默认 | 说明 |
|---|---|---|---|
| 总开关 | `enabled` | **False** | False ⇒ 耦合层不构造、世界逐位不变 |
| | `weather_source` | `"off"` | `"off"` \| `"atmo"`；`"atmo"` ⇒ 必须 `enabled=True` |
| `grid`（§9.1） | `kind` | `"gaussian"` | T21/T31 都支持（OPEN-4） |
| | `nlat/nlon/nlev` | 32/64/8 | L8 先跑通（OPEN-6） |
| `time`（§9.2） | `dt_atmo` | 900.0 | 600–900 |
| | `step_mode` | `"tick_multiple"` | M2′；`"tick_split"` 仅兼容 |
| `orbit`（§9.3） | 见 §1.3 | 世界派生 | `follow_world_season=True` |
| `held_suarez`（§9.4） | HS94 标准值 | A1 透传 | 不在此调参 |
| 耦合 | `boundary_every_k` | 由时钟算 | 只读派生值，不允许手改 |
| | `precip_refresh` | `"every_exchange"` | `"off"` ⇒ 只在构造期换源 |
| | `sst_mode` | `"world_light_base"` | 占位；`"atmo_t2m_feedback"` 留 A2 |

---

## 6. 默认关与 manifest 字段清单

### 6.1 默认关保证

- `enabled=False` ⇒ `SphereEngine(config, climate_source=None)`（默认 `EnvFieldSource`）、不 import 耦合层、不建 atmo、不调用重映射。
- 关档 digest 两档必须逐位：`classic 31d6acce…4072` / `default db9ccbb3…d295`（`tools/env_field_digest.py --device s2`）。

### 6.2 manifest（实验输出，A3.5 起）

```json
{
  "atmo": {
    "enabled": true,
    "weather_source": "atmo",
    "interface_version": "1.0.0",
    "core_version": "0.1.0",
    "grid": {"kind": "gaussian", "nlat": 32, "nlon": 64, "nlev": 8},
    "time": {"step_mode": "tick_multiple", "dt_atmo_s": 900.0,
             "tick_seconds": 36, "k_ticks": 25, "n_sub": 1,
             "dt_step_s": 900.0, "exchanges_per_year": 35040,
             "year_s": 31536000.0},
    "orbit": {"year_s": 31536000.0, "rotation_period_s": 86400.0,
              "obliquity_rad": 0.0, "eccentricity": 0.0167,
              "equinox_offset_s": 0.0, "follow_world_season": true},
    "boundary": {"h": "env_field.h_m - sea_level_m (ocean=0)",
                 "mask": "env_field.land_mask",
                 "alb": "SURFACE_THERMOPHYSICS[surface_type][0]",
                 "sst": "light.base_temperature (ocean only; placeholder)",
                 "seaice": "zero"},
    "exports": {"t2m_consumer": "climate.temperature_c",
                "precip_policy": "keep_total",
                "precip_refresh": "every_exchange",
                "weather_precip_active": true,
                "wind_consumed": false, "cloud_consumed": false,
                "sw_lw_consumed": false},
    "guards": {"world_atmosphere_enabled": false, "c1_enabled": false,
               "env_field_enabled": true, "c0b_enabled": true},
    "digest": {"atmo_initial": "...", "atmo_final": "..."}
  }
}
```

同时沿用既有 `time_scale` 节（`tick_seconds/rotation_period/year_ticks`），**不重复定义**。

---

## 7. 接口落差的显式处理（A3.3 桩）

- `atmo_core` 当前 `INTERFACE_VERSION="0.1.0-draft"`，且 `grid_spec(nlat,nlon,nlev,kind,rotation_period_s)`
  比 INTERFACE §8.2 多一个参数 ⇒ A3 启动断言**接受 `0.1.0-draft`/`1.0.0` 两种**并优先按关键字调用；
  A1 交 `contract()` 后收紧为字面量。**不自行改 atmo_core**。
- `ac.step` / `export`（§8.3/§10）未实现 ⇒ A3.3 定义**协议桩** `AtmoCoreLike`（同 §8 签名），
  测试用 fake core；A1 实现后零改动切换。
- `ac.step` 原地更新（OPEN-5）⇒ A3 **不持有跨步别名**：每次交换后把出口场 copy 到世界侧缓存。

---

## 8. 测试与验收（A3.2–A3.5）

| 项 | 判据 | 文件 |
|---|---|---|
| A3.2 重映射 | §3.3 七条 | `tests/test_atmo_coupling.py` |
| A3.3 时钟 | N=10,000 tick 后 `time_s == (t//k)·dt_step` 逐位；同 seed 两次边界/出口逐位 | 同上 |
| A3.4 关档 | 两档 digest 逐位不变；`weather_source="atmo"` 时三条 guard 全过 + `pr_source=="atmo"` + `band is None` | 同上 + `tools/env_field_digest.py` |
| A3.5 demo | s1（60×120）/200 tick：t2m 进世界、precip 空间型进 `pr`、NPP 随动；出冒烟读数 | `experiments/atmo_coupling_demo.py` |

测试预算：新用例目标 **≤2 s**（档②当前余量 ~12 s）；单例 >2 s 加 `slow` 并说明。

---

## 9. 接口变更单提案（批准前不动）

| # | 改什么 | 为什么 | 影响 |
|---|---|---|---|
| **IC1** | `simulation/sphere_engine.py` 资源再生温度改走 `self.climate.temperature_c(...)`（Python 与 Rust `regrow*` 两条路径） | 当前 `ClimateSource.temperature_c` 只覆盖生物层 + 动态 NPP；资源再生仍读 `LightAndTemperature` ⇒ "t2m 替换世界温度"只完成一半 | 开气候源时资源再生随天气；默认 `EnvFieldSource` 透传 ⇒ 逐位等价 |
| **IC2** | `SimConfig` 新增 `atmo: AtmoConfig`（默认关） | 让 A3 配置进指纹/manifest/旧档回退 | 指纹变、行为不变（默认关） |
| **IC3** | 引擎在耦合节拍调用气候源的可选钩子（如 `refresh(tick)` 或新 `on_coupling_step`） | 运行期推送降水/NPP 需要世界侧调用点；当前 `refresh()` 文档明确"引擎不调用" | 默认源 no-op ⇒ 逐位等价 |
| **IC4** | `world` 新增 SST/海冰/反照率场（或 A2 海表能量平衡） | §2 的 `sst` 目前是世界光照剖面占位，`seaice=0` | 物理口径升级，MINOR |
| **IC*（勘误）** | INTERFACE §7.3 "1.5 万步/年"→"3.5 万步/年" | 公式实算 35,040 | 文本勘误，无代码影响 |

> 首轮（A3.1+A3.2）不依赖 IC1–IC4；A3.5 demo 用**实验驱动循环**（`eng.step()` + 每 k tick 手动交换）
> 绕开 IC3，把"链路能跑"与"生产接线"分开验收。

---

## 10. 风险与未做

1. **A2 物理未就位**：`precip/cloud/sw/lw` v0 为 0/占位 ⇒ A3.4/A3.5 只能证链路，不能证天气真实性。
2. **A1 step 未实现**：A3.3 用 fake core；A1 实现后需重跑同一套时钟/确定性用例。
3. **SST 占位**：世界光照基温当 SST，无海表热容 ⇒ 海温不响应大气；A2 上线后换源。
4. **温度替换不完整**：资源再生路径见 IC1；v0 只覆盖生物层 + 动态 NPP，汇报必须如实写。
5. **降水只换空间型**（`keep_total`）：全球水量仍由世界 proxy 均值定；若将来要"atmo 总量说了算"（`table_total`），
   必须先解决 proxy 量纲 ↔ mm/yr 的换算口径（单列变更单，不在 A3 偷改）。
6. **Gaussian 面积边**：必须用 `cumsum(area)`，不能用 `lat_edges_deg`（§3.1，差 22%）。
7. **极端经度网格**：经度要求升序 `[0,360)`；外部表负经度需先归一（A3.2 明确报错，不静默排序）。

---

*起草：A3（2026-10-20）。基线 `eae404c`。等总审审阅后开工 A3.3/A3.4。*
