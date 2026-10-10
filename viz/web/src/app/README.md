# src/app/ — 应用与图表（营造线）

职责：应用骨架 + 数据加载 + ECharts 仪表盘。

- **数据加载**：选择/输入数据包目录 → 读 `meta.json` + 按需拉帧（`frames/ch_*/NNNNN.bin`、`entities/*.f32`、`series.csv`）。开发期数据包放 `web/public/datapack/`（构建产物不入 git）。
- **状态**（Zustand）：`frameIdx / playing / fps / layer / datapackUrl`。
- **图表**：ECharts 读 `series.csv`（列随实验：tick,pop,global_sat,abs_food,…）；与播放器联动（当前 tick 指示线）。
- **壳工程**：`App.tsx` 组合世界画布（丹青的 `src/render/`）+ 控制条 + 图表面板。
- 类型：`src/contract.ts`（契约 v0 的 TS 镜像，改动须随 `viz/CONTRACT.md` 同步）。

接口契约：`viz/CONTRACT.md` 🔒。先读它再动工。
