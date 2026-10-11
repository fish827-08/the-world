# viz/ — 可视化（v1 · 三维球体）

把实验运行变成"**看得到、可复查**"的本地 Web 应用：世界回放 + 指标仪表盘。

> **本文档针对 v1（三维球体，分支 `feat/viz-v1-globe`），全程 Windows PowerShell 口径。**
> - **v0 vs v1**：v0 是旧平面网格版（设计稿 `docs/设计文档/设计-可视化v0-20261010.md`）；v1 在其基础上改为 three.js 三维球体（设计稿 `docs/设计文档/设计-可视化v1-三维球体-20261011.md`）。**你现在要跑的是 v1**，照本文即可。
> - 🔒 接口契约：`viz/CONTRACT.md`（v0 冻结件，改字段需三侧同步）。
> - ⚠️ PowerShell **不支持 `&&`** 作语句分隔符，一律用 `;`。
> - 本文所有命令、端口、环境变量均已实测确证，照抄即可。

---

## 术语速查（第一次接触先看这里）

| 术语 | 白话解释 |
|---|---|
| **球体 / globe** | v1 前端渲染的可旋转三维地球，世界的每一格贴在球面上。 |
| **datapack（数据包）** | 一次实验的**可视化产物目录**：帧序列 + 实体 + `meta.json`（+ 可选 `series.csv` 曲线）。前端读它来回放世界。 |
| **s2** | 正式实验档的**装置**代号 = **480×960 网格（460,800 格）**。纪律 R5.6：跑正式实验必须显式 `--device s2`；吃默认值（60×120）会与历史读数不可比。 |
| **stride** | 采样间隔：每多少个 tick 存一帧（`--stride 20` = 每 20 tick 一帧）。 |
| **supervise（路径）** | 后端"托管跑实验"的运行路径：按 stride 把帧**增量**写进 `<data_root>\live\<run_id>\`，前端 live 模式轮询 `meta.json` 增量拉帧。 |
| **broker** | 一条**独立的 WebSocket 标量直播管道**（每 N tick 推一份世界状态 JSON）。⚠️ **看球体 / 浏览数据包用不到它**。 |
| **worktree** | Git 工作树。本 v1 代码在 `.worktrees\VIZ\`（分支 `feat/viz-v1-globe`），主仓在 `the-world\` 根。 |
| **简牍 / 丹青 / 营造** | 项目内部分工线名：简牍=数据管线，丹青=渲染，营造=应用+图表。见根目录 `AGENTS.md` 与 `viz/CONTRACT.md`。 |
| **`.venv` 的 python** | 主树虚拟环境的 Python 解释器。本机 `python` **不在 PATH**，所有 Python 命令都用它的绝对路径（见《全局前置约定》）。 |

---

## 目录结构

| 目录 | 线 | 干什么 |
|---|---|---|
| `exporter/` | 简牍（数据管线） | 快照 → 数据包（帧序列 + 实体 + meta） |
| `web/` | 丹青（渲染）/ 营造（应用+图表） | Vite+React+TS：三维球体世界回放 + ECharts 仪表盘 |

---

## 🚀 最短路径：只想看球体（3 步，单终端）

> **无需后端、无需导出数据包、无需快照。**
> 前端**自带样例数据包**（`viz\web\public\datapack\`，随代码打包），`/` 会自动加载它。
> `node_modules` 已装好，不用 `npm install`。

打开**一个** PowerShell 窗口，逐行执行：

```powershell
cd c:\Users\圣羽\Desktop\temp\tempCode\the-world\.worktrees\VIZ\viz\web
npm run dev -- --host 127.0.0.1 --port 5180 --strictPort
```

启动后终端会打印实际监听地址。浏览器打开：

```
http://127.0.0.1:5180/
```

**应看到一个可旋转的三维球体**（自带样例数据）。到此最短路径完成 ✅。

> - `-- --host 127.0.0.1 --port 5180 --strictPort`：`--` 之后是**透传给 vite 的参数**——强制 IPv4、固定端口 5180。为什么需要，见《前端 dev server 详解 · IPv4/IPv6》。
> - 省事写法：直接 `npm run dev`（不带参数）——vite 默认只监听 IPv6 `[::1]`，浏览器开 **`http://localhost:5173/`** 即可（**默认端口是 5173**，被占会自动顺延 5174、5175…，**以终端打印为准**）。

---

## 完整路径：跑实验 / 看真实 s2 数据包

要看主树数据仓里的**真实 s2 数据包**、或从控制台**发起实验**，在最短路径之上再加两件事：① 另开一个终端启后端 API；② 给前端设数据仓环境变量。

### 启动顺序与终端安排

| 步骤 | 终端 | 干什么 |
|---|---|---|
| 1 | 终端 A | 启**后端 API**（`observatory.api`，端口 8000）——供前端 `/api` 代理与实验控制 |
| 2 | 终端 B | 设 `THE_WORLD_DATA_ROOT` / `THE_WORLD_API_ORIGIN`，启**前端 dev server** |

> - 顺序建议**先后端、再前端**（前端会把 `/api` 代理到 8000）。
> - **两个服务各占一个终端窗口**，别在同一窗口先后跑（会互相顶掉）。
> - 只浏览数据包不发起实验时后端可省，但看真实 s2 包必须设 `THE_WORLD_DATA_ROOT`。

### 终端 A — 启后端 API

```powershell
cd c:\Users\圣羽\Desktop\temp\tempCode\the-world\.worktrees\VIZ
c:\Users\圣羽\Desktop\temp\tempCode\the-world\.venv\Scripts\python.exe -m observatory.api --host 127.0.0.1 --port 8000 --data-root c:\Users\圣羽\Desktop\temp\tempCode\the-world\the-world-data
```

自检：浏览器开 `http://127.0.0.1:8000/health`（应返回 JSON）；交互文档在 `http://127.0.0.1:8000/docs`。

### 终端 B — 设环境变量 + 启前端

```powershell
cd c:\Users\圣羽\Desktop\temp\tempCode\the-world\.worktrees\VIZ\viz\web
$env:THE_WORLD_DATA_ROOT="c:\Users\圣羽\Desktop\temp\tempCode\the-world\the-world-data"
$env:THE_WORLD_API_ORIGIN="http://127.0.0.1:8000"
npm run dev -- --host 127.0.0.1 --port 5180 --strictPort
```

打开真实 s2 样例包（该包已确认存在于数据仓 `the-world-data\viz\v1_s2_sample\`）：

```
http://127.0.0.1:5180/?datapack=/data/viz/v1_s2_sample/
```

从控制台发起实验（⚠️ 本分支落后 main，当前会 fail-loud 报 `script_not_found`，见《已知坑 · 3》）：

```
http://127.0.0.1:5180/?tab=console
```

---

## ✅ 成功长什么样 & 排查表

**成功现象：**
- 开 `/` → **可旋转的三维球体**（拖拽旋转、滚轮缩放）。
- 开 `/?datapack=/data/viz/v1_s2_sample/` → 球体加载真实 s2 数据。
- 开 `/?tab=console` → **实验控制台面板**（本分支发起实验报错属已知坑，见《已知坑 · 3》）。

**没成功？对照排查：**

| 现象 | 大概率原因 | 怎么修 |
|---|---|---|
| 页面**空白** / 球体不出来 | 看真实包时 `THE_WORLD_DATA_ROOT` 没设或指错 | 在**当前**终端重设（会话级！），确认指向主树 `the-world-data`；只看自带样例则无需该变量 |
| 浏览器**连接被拒 / 打不开** | vite 默认只听 IPv6 `[::1]`，你用了 `http://127.0.0.1:` | 二选一：① 改用 `http://localhost:<端口>/` 或 `http://[::1]:<端口>/`；② 启动加 `-- --host 127.0.0.1`（配 `--port 5180 --strictPort` 固定端口） |
| 端口和文档对不上 / 打不开 | dev server 实际端口 ≠ 你以为的 | **以终端打印的监听地址为准**（默认 5173，被占顺延） |
| `/data/...` 返回 **404** | 数据仓路径不对，或目标包不存在 | 核对 `THE_WORLD_DATA_ROOT` 指向 `the-world-data`；确认包目录真实存在（如 `the-world-data\viz\v1_s2_sample\`） |
| 控制台发起实验报 `script_not_found` | 本分支缺 `experiments/run_bare_world.py` | 已知坑，见《已知坑 · 3》（合并 main 后自愈） |

---

## 全局前置约定（每台机器 / 每个新窗口都要注意）

1. **python 不在 PATH** —— 所有 Python 命令一律用主树 venv 的**绝对路径**：
   `c:\Users\圣羽\Desktop\temp\tempCode\the-world\.venv\Scripts\python.exe`
2. **`$env:` 环境变量是"会话级"的** —— 只在**设置它的那个 PowerShell 窗口**里有效。**每新开一个窗口都要重设**（`THE_WORLD_DATA_ROOT` / `THE_WORLD_API_ORIGIN` / `PYTHONUTF8` 皆然）。
3. **跑测试前先设 UTF-8**（规避 Windows GBK 解码假红）：
   ```powershell
   $env:PYTHONUTF8="1"
   ```
4. **数据仓在主树** —— 可视化数据仓 `the-world-data\` 位于主树
   `c:\Users\圣羽\Desktop\temp\tempCode\the-world\the-world-data`；本 worktree 通过 `--data-root`（后端）/ `THE_WORLD_DATA_ROOT`（前端）指过去即可看到真实包。

---

## 前端 dev server 详解

工作目录（`node_modules` 已装）：`c:\Users\圣羽\Desktop\temp\tempCode\the-world\.worktrees\VIZ\viz\web`

### 环境变量（node 侧）

| 变量 | 值 | 作用 |
|---|---|---|
| `THE_WORLD_DATA_ROOT` | `c:\Users\圣羽\Desktop\temp\tempCode\the-world\the-world-data` | vite 的 `serveDataRoot` 中间件把 URL 里的 `/data/` 映射到它 |
| `THE_WORLD_API_ORIGIN` | `http://127.0.0.1:8000` | 前端 `/api` 请求的代理目标（后端 API） |

前端 `src/app/config.ts` 的三个默认值**保持不动、无需改**：
`VITE_API_BASE=/api`、`VITE_DATA_BASE=/data`、`VITE_LIVE_DATA_BASE=/data/live/`。

### IPv4 / IPv6（连接被拒的根因）

vite 默认只监听 IPv6 `[::1]`，直接用 `http://127.0.0.1:<端口>` 可能"连接被拒"。两种解法：
- **改用 localhost**：`http://localhost:<端口>/`（或 `http://[::1]:<端口>/`）——无需任何启动参数。
- **强制 IPv4**：启动加 `-- --host 127.0.0.1`，之后就能用 `http://127.0.0.1:<端口>/`。

### 端口口径（统一说明）

> **深链里的端口以你 dev server 实际监听为准**：默认 **5173**，被占会**自动顺延**（5173→5174→5175…）；启动加 `--port 5180 --strictPort` 可**固定为 5180**（被占直接报错，不顺延）。本文深链示例统一用 `5180` 举例；若你用默认端口，把 `5180` 换成终端实际打印的端口即可。

### 深链（三个 query 参数可自由组合）

| URL（示例用 5180） | 打开 |
|---|---|
| `http://127.0.0.1:5180/` | 默认球体，自动加载自带样例 `/datapack/` |
| `http://127.0.0.1:5180/?view=flat` | 平面视图 |
| `http://127.0.0.1:5180/?tab=console` | 实验控制台 |
| `http://127.0.0.1:5180/?datapack=/data/viz/v1_s2_sample/` | 指定真实 s2 数据包（需设 `THE_WORLD_DATA_ROOT`） |

组合示例：`/?view=flat&tab=console&datapack=/data/viz/v1_s2_sample/`

---

## 后端 API 详解（实验控制，REST）

从 **worktree 目录**运行主树 venv 的 python：

```powershell
cd c:\Users\圣羽\Desktop\temp\tempCode\the-world\.worktrees\VIZ
c:\Users\圣羽\Desktop\temp\tempCode\the-world\.venv\Scripts\python.exe -m observatory.api --host 127.0.0.1 --port 8000 --data-root c:\Users\圣羽\Desktop\temp\tempCode\the-world\the-world-data
```

- 自检：`GET http://127.0.0.1:8000/health`；交互文档：`http://127.0.0.1:8000/docs`。
- **装置白名单当前只有 `s2`**（=480×960 正式档；纪律 R5.6：吃默认值 ⇒ 与历史读数不可比）。

---

## 后端 broker（WebSocket 标量直播，可选）

> **看球体 / 浏览数据包用不到它**——仅当需要实时标量直播时才启。

```powershell
c:\Users\圣羽\Desktop\temp\tempCode\the-world\.venv\Scripts\python.exe -m observatory.broker --port 8765 --interval 100
```

- `--interval 100` 的单位是 **tick**：模拟每走 100 个 tick 推一份世界状态 JSON（快照节拍，默认 100，必须 ≥1）。
- 它是**独立于前端主链路的管道**，非启动必需。

---

## 数据包导出 / 校验（要自己造数据包时才用）

> 只想看球体**跳过本节**——前端已自带样例数据。要把**自己的实验快照**变成可回放数据包时才用。

**快照 `<snap>.npz` 从哪来？** 它是**实验/引擎的产物**：跑实验时由引擎 `save_snapshot` 落盘的 `.npz`（通常在实验输出目录 / `_rerun_logs\`）。拿到路径填给 `--snapshot` 即可。

```powershell
# 导出：快照 → 数据包（帧序列 + 实体 + meta + series）
c:\Users\圣羽\Desktop\temp\tempCode\the-world\.venv\Scripts\python.exe viz\exporter\export_datapack.py --snapshot <snap>.npz --out <dir> --ticks 200 --stride 20 --channels resource,energy --series auto

# 校验数据包完整性
c:\Users\圣羽\Desktop\temp\tempCode\the-world\.venv\Scripts\python.exe viz\exporter\check_datapack.py <dir>
```

- `--ticks 200`：从快照续跑 200 tick（`0` = 仅导出快照当前帧）。
- `--stride 20`：每 20 tick 存一帧。
- `--series auto`：自动扫描快照同目录的 `generations.csv` / `arm*.csv`，复制为 `series.csv`（前端曲线用）。
- 产物放 `the-world-data\`（或 `_rerun_logs\`），**不入主仓**。
- **实时数据包**由后端 supervise 路径按 stride 增量写到 `<data_root>\live\<run_id>\`（v0 格式）；前端 live 模式轮询 `meta.json` 增量拉帧。

---

## 生产部署（首次本地跑通可跳过）

- 后端**只提供 JSON**（REST），不托管数据帧字节。
- 数据帧字节在 **dev** 下由 vite `serveDataRoot` 中间件经 `/data/` 直供；
  **生产环境需自备静态服务或反向代理**来托管 `data_root`（vite dev 中间件不用于生产）。

---

## 已知坑（务必对照）

1. **python 不在 PATH**：一律用主树 `.venv\Scripts\python.exe` 绝对路径。
2. **`$env:` 变量是会话级**：每个新 PowerShell 窗口都要重设；跑测试前设 `$env:PYTHONUTF8="1"` 规避 GBK 解码假红。
3. **分支落后 main**：`.worktrees\VIZ`（`feat/viz-v1-globe`）缺 `experiments/run_bare_world.py` 与 `the-world-data/`，
   故 `GET /health` 返回 `default_script_exists=false`，从控制台发起新实验会 **fail-loud** 报 `script_not_found`；
   **把 main 合并进本分支后即自动可用**（可视化 / 数据包浏览不受影响）。
   数据仓在主树，用 `--data-root` 指过去即可看到真实包。
4. **生产部署**：见上节——数据帧字节需自备静态服务 / 反向代理。
5. **契约 `viz/CONTRACT.md` v0 为冻结件**：改字段需**三侧同步**（简牍 + 丹青/营造 + 后端），走评审流程。

---

## 纪律

- 导出器**只读**引擎公开状态 / 快照格式；要新字段 → 走引擎快照正常流程，不在本线改 `simulation/`。
- 契约改动 = 三侧同步（简牍 + 丹青/营造 + 后端），由轻舟评审。

---

> **Legacy（v0 / 非 Windows）**：旧 bash 口径的快速开始（`python viz/exporter/...`、`cd viz/web && npm install && npm run dev`）**不适用于本机 Windows 环境**——`python` 不在 PATH、PowerShell 不认 `&&`、依赖已装无需 `npm install`。请一律改用上方《最短路径》与《完整路径》。前端构建产物仍用 `npm run build`（在 `viz\web` 目录下执行）。
