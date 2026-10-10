# viz/ — 可视化（v0）

把实验运行变成"**看得到、可复查**"的本地 Web 应用：世界回放 + 指标仪表盘。
设计稿：`docs/设计文档/设计-可视化v0-20261010.md`｜🔒 接口：`viz/CONTRACT.md`（先读）。

## 结构

| 目录 | 线 | 干什么 |
|---|---|---|
| `exporter/` | 简牍（数据管线） | 快照 → 数据包（帧序列 + 实体 + meta） |
| `web/` | 丹青（渲染）/ 营造（应用+图表） | Vite+React+TS：Canvas 世界回放 + ECharts 仪表盘 |

## 快速开始

```bash
# 1) 导出数据包（示例：小世界快照续跑 200 tick，每 20 tick 一帧）
python viz/exporter/export_datapack.py \
  --snapshot <快照>.npz --out <输出目录> --ticks 200 --stride 20

# 2) 校验数据包
python viz/exporter/check_datapack.py <输出目录>

# 3) 前端（开发）
cd viz/web && npm install && npm run dev

# 4) 前端（构建）
npm run build
```

数据包**不入主仓**：产物放 `the-world-data/`（或 `_rerun_logs/`）。

## 纪律

- 导出器**只读**引擎公开状态/快照格式；要新字段 → 走引擎快照正常流程，不在本线改 `simulation/`。
- 契约改动 = 双侧同步（简牍 + 丹青/营造），由轻舟评审。
