# A2 大气物理包（`atmo/physics/`）

单柱可独立运行的大气物理包：太阳几何 → 短波/长波 → 湿力学 → 云 → 降水 →
对流 → 地表，外加不依赖 A1 动力核心的 `column` 驱动与能量/水量账本。
`driver.py` 提供 A1 sigma 布局 `(nlev, nlat, nlon)` 的多柱驱动（层序翻转、
sigma 层压、外部 `toa_sw/cosz`），供 A3 在 A1 步之间调用。

> **太阳几何归属（`atmo/INTERFACE.md` FROZEN v1.0.2 §1）**：正式耦合的
> 太阳几何/TOA 短波由 A1 提供；本包 `solar.py` 仅作列模式与 A1↔A2 对拍参考，
> 不得作为世界天气的第二个真源。

## 快速开始

在 A2 worktree 根目录（`atmo/` 的上一级）执行：

```bash
# 模块自检 + 三档单柱 RCE（slab 海洋 / 固定 SST / 陆地），结果写 JSON
PYTHONIOENCODING=utf-8 \
  "<repo>/.venv/Scripts/python.exe" \
  -m atmo.physics.selfcheck --out _rerun_logs/atmo_physics_selfcheck.json

# 快速档（60+5 天；固定 SST/陆地 20+5 天）
... -m atmo.physics.selfcheck --quick

# 只跑模块级检查
... -m atmo.physics.selfcheck --skip-rce
```

最小代码用法：

```python
from atmo.physics import AtmoPhysicsConfig, ColumnModel

model = ColumnModel(AtmoPhysicsConfig())
diag = model.step()          # 一步 dt=900 s，返回 StepDiagnostics
summary = model.run(days=30, spinup_days=300)
print(summary["means"]["toa_net"], summary["system_closure_wm2"])
```

## 模块与公式出处（一行一条）

| 文件 | 内容 | 公式/参数出处 |
|---|---|---|
| `solar.py` | **参考实现**：太阳赤纬、日地距离、天顶角、三种柱模式（global_mean/fixed/geometry） | Hartmann, *Global Physical Climatology* 2nd ed. §2.3（δ、a/r、μ0、日平均、S0/4） |
| `radiation.py` | 短波 delta-Eddington 双流；长波逐层灰体（可选 OLR=A+B·T 对照档） | Liou 2nd ed. Ch. 6–7；Petty 2nd ed. Ch. 13；Toon et al. (1989), *JGR* 94 |
| `thermo.py` | Clausius–Clapeyron、饱和比湿、潜热、饱和调整、液相蒸发 | Rogers & Yau 3rd ed. §2.3–2.4；Buck (1981), *JAM* 20 |
| `clouds.py` | 诊断云量、云水路径、Slingo 云光学厚度、未饱和云水蒸发 | Slingo (1987), *QJRMS* 113；Slingo (1989), *QJRMS* 115 |
| `precip.py` | Kessler 自动转化/碰并、稳态雨通量落出与再蒸发 | Kessler (1969), *Meteor. Monogr.* 10(32) |
| `convection.py` | MSE 中性湿对流调整（Betts–Miller 硬调整极限），产加热/增湿/降水 | Betts & Miller (1986), *QJRMS* 112；Manabe & Wetherald (1967) |
| `surface.py` | 整体公式感热/潜热、风驱动蒸发、slab 海洋、1–2 层土壤隐式热传导 | Stull (1988) 边界层 §5；Bonan (2019) Ch. 5–7 |
| `column.py` | 单柱状态、逐步物理编排、能量/水量账本、RCE 运行器 | —（装置层，不引入新公式） |
| `driver.py` | A1 sigma 布局多柱驱动：层序翻转、`p_edges=ps·sigma_half`、A2 自有态（qc/Ts/soil_T）、面积加权账本 | —（映射/装置层，不引入新公式） |
| `config.py` | 全部可调参数（dataclass，`to_dict`/`from_dict`） | 各字段行内注释 |
| `constants.py` | 物理常数/地球装置量（SI，逐条注明出处） | CODATA 2018、WMO、AMS/ISO 2533 等 |
| `selfcheck.py` | 34 项模块检查 + 三档 RCE + 判据门 | 判据来自量纲/文献量级，先写后跑 |

## 状态接口（接口冻结前的列模式 stub）

数组顺序统一为**地表 → 顶层**（与 `pressure_edges_pa` 递减一致）：

- `ColumnState.p_edges`：`(nlev+1,)` 气压界面 [Pa]，`p_edges[0]` 为地表。
- `ColumnState.T / qv / qc`：`(nlev,)` 层平均温度 [K]、比湿/云水 [kg/kg 干空气]。
- `ColumnState.Ts`：表层温度 [K]；`soil_T`：土壤层温 [K]；`u/v` 可选。
- 辐射：`ShortwaveFluxes` / `LongwaveFluxes` 的界面通量均为 `(nlev+1,)`，
  向下/向上为正；`toa_in/out` 为水平面通量 [W/m²]。
- 地表湍流：`sensible`/`latent`/`evaporation` **向上为正**。
- 长波 `surface_net_down` 为地面净向下 [W/m²]（向下为正）。
- 账本：`StepDiagnostics.energy_residual` [J/m²/step]、
  `water_residual` [kg/m²/step]；`ColumnModel.run()` 另给
  `system_closure_wm2`（TOA − dE/dt）、`atmosphere_closure_wm2`、
  `surface_closure_wm2`、`dts_dt_k_per_day`。

## A1 布局驱动（`driver.py`，联调前置件）

```python
from atmo.physics import AtmoPhysicsConfig, AtmoPhysicsDriver

drv = AtmoPhysicsDriver(AtmoPhysicsConfig(), grid)   # grid = atmo_core.grid_spec() 子集
aux = drv.make_state(st, forcing)                    # 构造 A2 自有态
diag = drv.step(st, forcing, aux, 900.0,
                toa_sw=toa_sw, cosz=cosz)            # 必传 A1 的太阳几何
fields = drv.export()                                # precip/cloud/sw_sfc/lw_sfc
```

硬口径：

- **层序**：A1 `k=0` 在模式顶 ⇔ 柱索引 `0` 在地表；`T[7] ↔ T[0]`。
- **层压**：柱界面 `p_edges[0]=ps`、`p_edges[k]=ps·sigma_half[nlev-k]`，
  顶界面用 `driver.top_pressure_pa`（默认 1 Pa）代替 sigma 的 0。
- **太阳**：`toa_sw`/`cosz` 必须来自 A1 `ac.solar()`；本驱动不自行兜底。
- **A2 自有态**：`qc`、`Ts`、`soil_T`（A1 状态无这三项）。
- **海陆**：`mask ≥ driver.land_mask_threshold` 按陆面，否则海面；v0 不做部分陆地混合。
- **海洋模式**：`prescribed`（Ts 固定为 `sst`，外部热库）或 `slab`（混合层参与储能）。
- **出口**：`precip` kg·m⁻²·s⁻¹、`cloud` 0–1、`sw_sfc` 地面净短波、
  `lw_sfc` 地面净向下长波 [W/m²]。
- **账本**：逐柱沿用 `ColumnModel`；全球平均 = `Σ_j area_j·(1/nlon)Σ_i`。

## 参数与账本

- 可调参数全部在 `config.py`；默认值可原样进 manifest（`to_dict()`），
  `from_dict()` 对缺失字段回退默认值。
- 物理常数在 `constants.py`，属于写死账本 🔵 保留类；
  参数入账提案见 `内部汇报/变更单-20261020-A2-物理参数入账.md`。

## 验证读数（2026-10-20，最终代码）

模块级 `34/34 PASS`（0.3 s）；三档 RCE 总判定 PASS。摘要：

| 档 | Ts 起→末 [K] | dTs/dt [K/d] | TOA [W/m²] | OLR | P [mm/d] | E [mm/d] | 系统闭合 [W/m²] |
|---|---|---|---|---|---|---|---|
| slab 海洋（300+30 d） | 283.78→283.43 | −0.0119 | −31.49 | 90.65 | 0.68 | 0.72 | +3.8e-7 |
| 固定 SST（60+10 d） | 288.15→288.15 | 0 | −33.89 | 86.08 | 2.10 | 2.12 | 大气 −2.8e-7 |
| 陆地（60+10 d） | 266.31→264.59 | −0.172 | −13.67 | 65.23 | 0.12 | 0.04 | 地表 +1.4e-7 |

能量账本单步最大残差 ≤ 0.035 J/m²（等效 <1e-12 K），水量账本 ≤ 1e-14 kg/m²/step，
无 NaN；同配置两次 20 步逐位一致。完整读数见
`_rerun_logs/atmo_physics_selfcheck.json` 与
`内部汇报/汇报-20261020-A2-大气物理包.md`。

## 已知限制

1. **无冰相**：云水/降水全按液态处理；`qc_hold` 与降水不受温度相态约束。
2. **单柱无大尺度动力/下沉**：slab 档会饱和成满云盖（cloud=1.0），
   TOA 不趋零；"辐射平衡"按**系统能量闭合 + 准平衡漂移**判定，
   不按 `TOA≈0` 判定。
3. **陆地无土壤湿度库**：`land_evap_beta` 固定 ⇒ 干燥/偏冷态；
   陆地档只验收守恒，不验收气候合理性门。
4. **云-辐射用网格平均凝结物**：无云量分区/云内光学厚度加权；
   随机重叠仅用于诊断总云量。
5. **水汽松弛档默认关**：目标温度必须固定；用当前 T 作目标会正反馈发散
   （实测数天内 NaN）。
6. **与 A1 冻结接口的映射**（`atmo/INTERFACE.md` v1.0.2）：`driver.py` 已显式
   处理层序（A1 `k=0` 顶 ⇔ 柱 0 地表）与 sigma 层压；单柱 `column.py` 仍是固定
   8 层气压（索引 0 地表），两者不要混用同一数组约定。A3 只负责世界↔atmo 重映射与
   出口组装，不重复做本驱动的层序换算。
7. **部分陆地未混合**：`land_mask_threshold` 二选一，海岸带格点会整格按陆/海处理。
8. **A1 动力未就位**：驱动只用合成 A1 状态验证过；`ac.step` 仍是桩，
   端到端水行星尚未跑（等 A1 补交）。
9. **简化辐射**：宽带水汽吸收系数按晴空吸收/OLR 标定，无臭氧/气溶胶/
   光谱分带；对流为硬调整极限，无卷入/下沉气流。
