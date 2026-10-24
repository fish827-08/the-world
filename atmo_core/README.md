# atmo_core —— 大气动力核心（乙方案）

> **契约**：`../atmo/INTERFACE.md`（**FROZEN v1.0.2**）。
> **分支**：`feat/atmo-core` ｜ **写作用域**：本目录（+ 契约只读；改契约走变更单）。

球面干动力核心。职责单一：**给定网格、状态、边界场、步长 → 推进一个确定性子步**。
不含辐射/湿/云/降水/对流（A2）、不含世界耦合（A3）。

---

## 1. 当前状态（2026-10-20 第 1 天）

| 模块 | 文件 | 状态 |
|---|---|---|
| 确定性数学（sin/cos/atan/asin） | `src/detmath.rs` | ✅ |
| 网格（Gaussian / 规则、`mu_edges`、`cosbar`、科氏、sigma） | `src/grid.rs` | ✅（v1.0.1 的 `mu_edges` 已补） |
| 状态（分配 / 布局校验 / FNV-1a digest） | `src/state.rs` | ✅ |
| 球面算子（散度/涡度/梯度/拉普拉斯） | `src/ops.rs` | ✅ W1（全球平均散度精确为零） |
| 太阳几何（轨道 → cosZ → 大气顶短波） | `src/solar.rs` | ✅ W1（纯解析，禁表） |
| 出口场 `t2m/u10/v10` | `src/export.rs` | ✅（v1.0.2） |
| 2 层浅水 + RK3 + 极区滤波 + CFL | `src/shallow.rs` | ✅ W1（独立入口 `shallow_water_step`） |
| 极区**纬向截断**滤波（按当地 CFL 定 m_max） | `src/shallow.rs` | ✅ 默认 `filter_mode=2, cfl_target=0.2` |
| 8 层原始方程 `step()`（HS94 + 海绵 + 极区滤波） | `src/prim.rs` | ✅ W2（**dt≤288 s**，见 §5.2） |
| 极区滤波/扩散公共件 | `src/filt.rs` | ✅ |

**没实现的东西会抛 `NotImplementedError` 并写明交付周次**（不返回假数据）。

---

## 2. 构建（本机 Windows）

```bash
cd atmo_core
export PATH="$HOME/.cargo/bin:$PATH"                       # ① 不设 ⇒ cargo 找不到
export PYO3_PYTHON="<repo>/.venv/Scripts/python.exe"               # ② 不设 ⇒ "no Python 3.x interpreter found"
export TEMP="<repo><worktree>/_rerun_logs/tmp"; export TMP="$TEMP"; export TMPDIR="$TEMP"  # ③ 沙箱下 rustc 写临时目录
PYTHONIOENCODING=utf-8 ../../.venv/Scripts/python.exe -m maturin build --release -o ../_rerun_logs/wheels
```

> ⚠️ 三条都是**实测踩过的坑**（2026-10-20）：venv 路径含中文 ⇒ 必须 `PYTHONIOENCODING=utf-8`；
> `TEMP` 指到 `D://AppData//Local//Temp` 时 rustc 报 `拒绝访问 (os error 5)`，必须改到仓库内。

只做编译自检（不产出 `.pyd`，最快）：

```bash
PYTHONIOENCODING=utf-8 cargo check
```

单元测试（纯 Rust，不需要 Python 解释器）：

```bash
cargo test
```

---

## 3. 最小用法

```python
import atmo_core as ac
import numpy as np

print(ac.__version__, ac.INTERFACE_VERSION)

g = ac.grid_spec(nlat=32, nlon=64, nlev=8)     # T21 / L8
assert abs(float(g["area"].sum()) - 1.0) < 1e-14
print(g["lat_deg"][:3], g["lat_deg"][-3:])      # −85.76 … +85.76（升序）
print(g["coriolis"][15:17])                     # 赤道附近 ≈ 0

st = ac.make_state(g, T0=288.0, ps0=101325.0)
print(st["u"].shape, st["ps"].shape)            # (8,32,64) (32,64)
print(ac.digest(st))                            # 16 位十六进制

core = ac.AtmoCore(32, 64, 8, 86400.0)
# core.step(st, forcing, 900.0)                 # W1 起可用（当前抛 NotImplementedError）
```

---

## 4. 为什么有 `src/detmath.rs`（不是重复造轮子）

`f64::sin/cos/atan/asin` 调用**系统 libm**：Windows（MSVC CRT）与 Linux（glibc）在最后 1–2 ulp
不保证相同。本项目要求「CPU L0 逐位」（`atmo/INTERFACE.md` §0/§8.6、内部工程指南 约束 1），
所以核心路径里的超越函数由 `detmath` 提供——它只用 IEEE-754 **正确舍入**的 `+ − × ÷ sqrt`，
因此**同一输入在任何平台输出逐位相同**。

精度：实测 |Δ| ≤ 2e-15（对角度量 ≈ 1e-13 度）。**精度不是首要指标，可复现才是。**

🔴 **禁止**在 `atmo_core` 的确定性路径直接调用 `f64::sin/cos/atan/asin/exp/ln/powf`。
`exp/ln` 由 A2 按同一模式补（或走接口变更单）。

---

## 5. 确定性契约（摘要）

| 项 | 规定 |
|---|---|
| 归约顺序 | 固定 `k → j → i`；禁止无序并行归约（rayon 只允许"逐格独立、无归约"的循环） |
| FMA | **禁 `mul_add`**（不同 target 会走不同路径） |
| 编译 | 禁 `-C target-cpu=native`、禁 fast-math（Rust 无 fast-math，但慎用 unsafe 的 SIMD intrinsic） |
| digest | FNV-1a 64，覆盖 `u,v,w,T,ps,q`（`p` 是诊断量，不入） |
| 跨平台 | Windows / Linux x86-64 应逐位一致（同 opt-level 同 target） |

---

## 5.1 极区滤波（W1 实测结论）

Gaussian 网格极行 `dx=46.2 km` ⇒ 显式积分在极区必然违反 CFL（`c_ext=198 m/s, dt=900` ⇒ **3.86**）。
实测（H=4000, dt=900, 600 步上限）：

| 方案 | 存活 |
|---|---|
| 无滤波 | 7 步（0.1 天） |
| 纬向 1-2-1 ×4 | 28 步（0.3 天） |
| 纬向 1-2-1 ×20 | 600 步（6.2 天，跑满） |
| **纬向截断 `cfl_target=0.5`** | 17 步 |
| **纬向截断 `cfl_target=0.2`** | **572 步（6.0 天）** |

⇒ 默认 `filter_mode=2, cfl_target=0.2`：按 `m_max(φ)=⌊0.2·2πa·cosφ/(c·dt)⌋` 只砍当地违反 CFL 的波数，
**不动低波数物理模态**（1-2-1 会一起压）。README 的这条是 W2 的时间步前提。

## 5.2 原始方程的时间步（W2 实测）

| dt | 结果 |
|---|---|
| 900 s | **NaN @ 40 步**（0.4 天） |
| 600 s | NaN @ 289 步（2.0 天） |
| **288 s** | **稳跑 3000 步（10 天）**：T[263.9, 278.7] K、umax 24.5 m/s、CFL 0.04、质量漂移 −2.2e-13 |

⇒ 显式方案的可用步长 ≈ **288 s**（tick=36 s ⇒ k=8），不是契约 §7.1 的 900 s。
另外三个必需项（缺一个就炸）：
1. **`nu = 1e7 m²/s`** 水平扩散（1e6 时模式顶赤道附近 2Δx 模 ~3.4 天炸）；
2. **垂直平流用迎风**（中心差激发 2Δσ 计算模，dt=288 也活不过 599 步）；
3. **顶部海绵层**（σ<0.1，Rayleigh 1/(0.5 天)）。

## 6. 不做（明确边界）

- 湿物理 / 辐射 / 云 / 降水 / 对流 → **A2**
- atmo↔世界重映射 / 时钟 / 边界场灌入 → **A3**
- benchmark 与判据 → **AV**
- 甲方案（SpeedyWeather.jl / ExoPlaSim 在线耦合）→ 不是本线

---

*2026-10-20 建立。改动本文件不改变接口契约；接口只认 `atmo/INTERFACE.md`。*
