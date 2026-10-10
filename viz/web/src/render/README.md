# src/render/ — 世界渲染层（丹青线）

职责：把数据包渲染成可看的球面世界。

- **入口部件**：`WorldCanvas`（props：`meta: Meta`、`frame: GridFrame`、`entities: EntityFrame`、`layer: string`）。
- **技术**：Canvas2D + ImageData；每帧把 `frames/ch_<key>/NNNNN.bin`（uint8, rows*cols）按 `scale` 拉伸成 RGBA 位图直写。
- **播放器**（与营造的 store 配合）：播放/暂停/步进/seek；帧号 ↔ tick = `meta.tick.start + k * meta.tick.stride`。
- **实体叠加**：`entities/NNNNN.f32`（交错 7×f32）→ 亮点/圆点；位置 `sub_r/sub_c`（+配置 subdiv 细化，缺数据回退格中心），颜色/尺寸按 `energy`（可用 `meta.entities.norm`）。
- 行 0 = 北极端；`flat = r*cols + c`；经度环绕（列方向绘制注意接缝）。

接口契约：`viz/CONTRACT.md` 🔒。先读它再动工。
