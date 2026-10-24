# AV 验证线 benchmark 套件（atmo）

> 角色：独立裁判。**不写生产代码**；只在本目录与 `tests/test_atmo_*` 写。
> 纪律：判据先提交（commit 时间戳为证）后跑；阈值来自文献/基准，不按结果回调；
> 报告只写实测值；不调参救结果。协议见 `内部汇报/管理-模型分配与审核协议-20261020.md`。

## 文件

| 文件 | 职责 |
|---|---|
| `criteria.py` | 冻结判据注册表（阈值+文献锚；唯一判定真源） |
| `metrics.py` | 指标库（纯 numpy：急流/ITCZ/EKE/CFL/极区/权重积分） |
| `conservation.py` | 守恒与确定性检查（质量/水/能量账、逐位对比） |
| `loader.py` | 输入契约（meta.json + coords.npz + snap_*.npz；只读） |
| `suites.py` | 两套件定义（Held–Suarez / aquaplanet）+ 判据执行 |
| `run.py` | CLI 运行器（JSON 报告 → `_rerun_logs/`） |
| `audit.py` | 写死审计门（扫描生产代码硬编码场/预设图案） |
| `reference.py` | 只读对照裁判（ExoPlaSim/观测气候态 npz 对比；**绝不作为模型输入**） |
| `jia_remap.py` | **甲方案原生 → loader 契约**的映射与协议件（J1 冻结；映射权归裁判，A1 只交原生场） |
| `jia_smoke.py` | 甲规模合成两臂证据（T32/L8、3168 点、float32、ps=ln Pa）：只验"管线能否出读数"，数值无物理意义 |
| `gate/P1_gate_template.md` | 《乙方案 P1 能力门》报告模板 |
| `gate/J1_mapping_freeze.md` | 《J1 映射与协议冻结》：映射方法逐条理由 + 15 条判据可评估性表 + 待 owner 裁定项 R-1…R-5 |
| `out/` | 本地产物（参考气候态数据等，二进制不进主仓） |

## 输入契约（A1/A2/A3 交付物须按此落盘）

见 `loader.py` docstring。要点：`meta.json`（benchmark/dt_atmo_s/days/snap_days/obliquity_deg/seed/config_hash/cfl_reported；湿另需 slab_heat_capacity/radiation）+ `coords.npz`（lat/lon + plev_hpa 或 sigma+sigma_edges）+ `snap_*.npz`（u,v,T,ps；湿另需 q,pr,evap,sw_toa,lw_out,sst，SI 单位）。数组 `(nlev,nlat,nlon)` C 序。
契约不符 ⇒ 运行器报 `INPUT_CONTRACT_ERROR` 退出 1——这是判例，不是要 AV 迁移适配。

**甲方案（SpeedyWeather）另走原生契约**：Float32 + octahedral 约化高斯（每环点数随纬度变、无极点）与上面的规则 `(nlev,nlat,nlon)` 不同构。A1 只交**原生场 + manifest**，由 `jia_remap.py` 冻结映射并产出 loader 契约 run 目录，判据/阈值一律不动。契约要点（全表见 `gate/J1_mapping_freeze.md` §2）：`manifest.json` 必含 `MANIFEST_KEYS`（含 `ic`/`env`）、`grid` 含 `npoints/nlayers/sigma/truncation`、`export` 声明 `dtype/ps_units/field_shape/byte_order`（`ps` 若为 `log(Pa)`，升 float64 后才 exp）、HS 套件另需 `forcing` 登记 HS94 十二键**加** `tmax_eq_K/tmin_eq_K/ps0_Pa`（协议值，实测所用，不许估填）。

## 复现命令（在仓库根或 worktree 根）

```bash
PY=<repo>/.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8

# 单元测试（合成场解析答案，秒级）
$PY -m pytest tests/test_atmo_criteria.py tests/test_atmo_metrics.py tests/test_atmo_conservation.py tests/test_atmo_audit.py tests/test_atmo_solar.py tests/test_atmo_jia_remap.py -v

# 甲规模合成两臂证据（映射能否产出 15 条读数；数值无物理意义）
$PY -m benchmarks.atmo.jia_smoke --out _rerun_logs/20261021-AV-j1-mapping

# 甲原生 run → loader 契约 run（A1 交付后；映射/阈值不动，只改数据）
$PY -m benchmarks.atmo.jia_remap --native <A1原生目录> --out <run目录> --rep <自证报告.json>
# AV 冻结 IC 供 A1 施加（先由 A1 交网格几何；A1 须把返回的 sha256 回填 manifest.ic）
$PY -m benchmarks.atmo.jia_remap --native <A1几何/原生目录> --ic iso_w1 --ic-out <原生目录>/ic_iso_w1.npz --ic-seed 0

# Held-Suarez 基准（A1 交付后）
$PY -m benchmarks.atmo.run --suite held_suarez --run-dir <A1产出目录> --run-dir-b <同seed第二次运行>

# 水行星湿基准（A2 交付后）
$PY -m benchmarks.atmo.run --suite aquaplanet --run-dir <目录> --run-dir-b <目录>

# 写死审计门（有 HIGH 命中即退出 1）
$PY -m benchmarks.atmo.audit atmo_core atmo
```

## 判据总览

- **CONS-\***（两套件通用）：质量漂移 ≤1e-6；水账闭合相对残差 ≤1e-3；能量账闭合 ≤1 W/m²（仅湿）；AV 独立重算 CFL ≤1；无 NaN/Inf；极区振幅比 ≤3 且末 20% 斜率 ≤0.01 m/s/d；同 seed 两次逐位一致。
- **HS-\***（Held & Suarez 1994）：强迫参数逐项对表（容差 1%）；急流 u_max ∈ [15,40] m/s、|lat| ∈ [20,60]°；近地面热带东风 u ∈ [−8,+2]；热带/极区地面温、热带对流层上层温三窗带宽；中纬瞬变 EKE ≥1 m²/s² 且中纬/热带比 ≥1.5。
- **AQ-\***（Neale & Hoskins 2000 框架 + GPCP 量级）：ITCZ 纬度 ∈ [−5,+5]°（obliquity=0）、峰值 ∈ [4,20] mm/d；副热带 E−P > 0；TOA 净通量 ∈ [−5,+5] W/m²；SST 趋势 ≤0.01 K/d；半球差 ≤2 mm/d；meta 登记辐射/slab 热容。

细节与出处逐条见 `criteria.py`。带均为宽松 sanity 带，细判走 `reference.py` 只读对照（ExoPlaSim/观测气候态，数据放本地 `out/reference/`，不进主仓、不作输入）。
