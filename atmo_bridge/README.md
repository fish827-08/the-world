# atmo_bridge —— 外部 GCM 的离线气候表桥接

> 建立：2026-10-18（A 线 W1）｜定位：**薄层**。GCM 在独立进程/机器上跑，本包只做
> 「表 ↔ 世界网格」的搬运与重映射。

**现役引擎 = ExoPlaSim 3.4.2（GPLv2）**。SpeedyWeather.jl 因 Julia 包服务器 175 B/s **不可行**被否
（2026-10-18 owner 裁定）；两个引擎都只能"离线出表、不内嵌"。

## 为什么是"离线表"而不是"直接耦合"

| 原因 | 说明 |
|---|---|
| **许可证** | SpeedyWeather.jl 是 **EUPL-1.2**（copyleft）。跑它、只发布**输出数据**不触发义务；把引擎打进发行包就要开源。离线表是唯一既能用又不用开源的路（见 `内部技术文档/技术-A3-SpeedyWeather许可证核实-20261018.md`） |
| **分辨率** | T31（≈400 km）远粗于世界 0.375°（≈42 km）。逐 tick 耦合没有意义，只会把粗场当真值 |
| **确定性** | 世界要求 L0 逐位可复现；GCM 自己带随机/并行归约差异。**表是死的**，世界才是可复现的 |

## 文件

| 文件 | 作用 |
|---|---|
| `remap.py` | 规则经纬网格之间的**保守面积权重映射**（纬度 sin 面积、经度角度交叠，行和归一）。守恒实测 relerr ≤ 4e-16 |
| `__init__.py` | 导出 `build_weights / remap_field / remap_fields` |

配套（不在本包内）：

| 位置 | 作用 |
|---|---|
| `experiments/run_exoplasim_t21.py` | 云端机上跑 ExoPlaSim 地球基准（T21/L10），量墙钟（**现役引擎**） |
| `experiments/export_exoplasim_table.py` | ExoPlaSim netCDF → `npz-climatology-v0` + manifest（sha256/字段极值） |
| `experiments/make_gcm_table.py` | CSV → `npz-climatology-v0`（备用路径：任何能吐 CSV 的 GCM） |
| `experiments/export_speedyweather_t31.jl` | SpeedyWeather 模板（**已弃用**：Julia 包下载 175 B/s，见派工 2026-10-18 引擎裁定） |
| `experiments/run_gcm_table_smoke.py` | 「世界读表」冒烟（默认合成表；`--table` 可真表） |
| `experiments/run_gcm_compare.py` | 世界气候态 vs GCM 表 三条纬向剖面对照 |
| `world/climate.py:GCMTableSource` | 世界侧只读该表的实现（`ClimateSource` 协议） |

## 表格式（`npz-climatology-v0`）

```
lat (nlat,)  lon (nlon,)              度；lon 用 0..360
temperature_c (nlat,nlon) 或 (12,nlat,nlon)   ℃
precip_mm_yr  (…同上…)                      mm/yr
u, v          (…同上…)                      m/s（东正 / 北正），可缺
meta_json     字符串：source/grid/months/units
```

## 纪律

1. **不守恒的东西不许进世界**：降水/水汽一律走保守重映射，禁双线性（会把水凭空造出来）。
2. **不猜参数**：12 月表必须显式给 `year_ticks`（世界"一年"多长）；没给就 `NotImplementedError`。
3. **不假装分辨率**：T31 只提供大尺度背景；480×960 上的地形细节来自 `env_field` 的降尺度，两笔账分开记。
4. **不 vendor GCM 源码**：本仓不存 SpeedyWeather 代码。
