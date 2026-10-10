# src/render/ — 世界渲染层（丹青线）

职责：把数据包渲染成可看的球面世界。接口契约：`viz/CONTRACT.md` 🔒（v0 修订1）。

## 部件

| 文件 | 说明 |
|---|---|
| `WorldCanvas.tsx` | 入口部件（Canvas2D + ImageData 逐帧直写） |
| `palette.ts` | 调色板纯函数（零依赖，便于测试复用） |

### `WorldCanvas` props（冻结签名）

```ts
interface WorldCanvasProps {
  meta: Meta;                     // src/contract.ts
  grid: Uint8Array | null;        // rows*cols 当前图层帧（u8，行主序，行 0 = 北极端）
  entities: Float32Array | null;  // 交错 7 列 [flat,sub_r,sub_c,energy,age,generation,mode]
  scale: { p_lo: number; p_hi: number };  // 当前图层标定（meta.channels[].scale）
  cellPx?: number;                // 每格像素；缺省按容器宽度自动取整
}
```

### 行为要点

- **图层帧**：u8（导出端已按 scale 量化）→ 近似 viridis LUT → ImageData 直写离屏位图
  → `drawImage` 整数倍放大（`image-rendering: pixelated` 保锐利）。
  `scale` 不参与绘制（契约 §3：量化已含标定），签名保留仅为契约完整。
- **实体位置**：`flat → (r,c)`；亚格 `sub_r/sub_c` 按 `meta.entities.subdiv` 归一
  （`row = sub_r/subdiv`，格单位）。**帧级一致性校验**（`sub/subdiv` 必须与 flat 推的 r/c 一致）
  不过则全帧回退**格中心**——实测 subpos 关档续跑的 sub 列与 flat 漂移
  （官方样例包 3240 实体/帧：f0 不一致率 0% → f1 88.2% → f50 99.6%），直读会错位；
  `subdiv` 缺失或无亚格数据直接回退格中心（契约 §4 口径，不做推断）。
- **实体着色/大小**：按 `energy` 用 `meta.entities.norm.energy` 归一（缺标定 ⇒ 当帧 min/max 兜底）；
  暖色带（低 = 暗红橙 → 高 = 亮黄白），半径随能量 0.3–1.0 格。
- **自适应**：ResizeObserver 监听容器宽；经度环绕：跨列缝的实体点在对侧补绘。
- **空数据安全**：`grid`/`entities` 为 null、0 字节（`Float32Array(0)`）、长度不足、
  越界索引、NaN 均不崩（纯背景 `#0d0d12`）。
- **已知限制（v0）**：不做 dpr 细化（M4 打磨项）。

### `palette` 导出

`viridisRgb(t)`（t∈[0,1] → RGB，六次多项式拟合，通道误差 ≲ 4/255）、
`viridisLut()`（256 级 RGBA 查找表，长度 1024）、
`energyRgb(t)` / `energyColorCss(t)`（能量暖色带，1/64 量化缓存）。

## 使用示例（营造集成）

```tsx
import { useEffect, useState } from "react";
import type { Meta } from "../contract";
import { WorldCanvas } from "../render/WorldCanvas";

const [meta, setMeta] = useState<Meta | null>(null);
const [grid, setGrid] = useState<Uint8Array | null>(null);
const [entities, setEntities] = useState<Float32Array | null>(null);

useEffect(() => {
  (async () => {
    setMeta(await (await fetch("/datapack/meta.json")).json());
    setGrid(new Uint8Array(
      await (await fetch("/datapack/frames/ch_resource/00050.bin")).arrayBuffer(),
    ));
    setEntities(new Float32Array(
      await (await fetch("/datapack/entities/00050.f32")).arrayBuffer(),
    )); // 空个体帧 = 0 字节 ⇒ Float32Array(0)，与 null 等价
  })();
}, []);

meta && (
  <WorldCanvas
    meta={meta}
    grid={grid}
    entities={entities}
    scale={meta.channels[0].scale}
    cellPx={2}
  />
);
```

- 播放器帧号 ↔ tick：`frameToTick(meta, k)`（`src/contract.ts`）。
- 依赖：仅 react + 平台 API（Canvas2D / ResizeObserver），无新增依赖。
