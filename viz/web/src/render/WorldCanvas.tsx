/**
 * WorldCanvas —— 世界帧渲染（Canvas2D + ImageData 直写）。
 *
 * 图层帧 = 导出端按 scale 量化后的 u8（行主序，行 0 = 北极端）⇒ 直接查
 * viridis LUT 上色；`scale` 不参与绘制（契约 §3：量化已含标定），仅为签名完整保留。
 * 实体叠加：flat → (r,c)；`sub_r/sub_c` 亚格绝对坐标（`meta.entities.subdiv`，契约 v0 修订1），
 * 帧级一致性校验不过（如 subpos 关档续跑的漂移列）⇒ 回退格中心；`subdiv` 缺失同此回退。
 */
import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import type { JSX } from "react";
import type { Meta } from "../contract";
import { energyColorCss, viridisLut } from "./palette";

const LUT = viridisLut();
const BG = "#0d0d12";
const TAU = Math.PI * 2;

export interface WorldCanvasProps {
  meta: Meta;
  /** rows*cols 当前图层帧（uint8，行主序，行 0 = 北极端）；null/空 = 无图层 */
  grid: Uint8Array | null;
  /** 交错 7 列 [flat,sub_r,sub_c,energy,age,generation,mode]；null/空 = 无实体 */
  entities: Float32Array | null;
  /** 当前图层标定（meta.channels[].scale）；u8 帧已量化 ⇒ 绘制不消费 */
  scale: { p_lo: number; p_hi: number };
  /** 每格像素；缺省按容器宽度自动取整 */
  cellPx?: number;
}

export function WorldCanvas({
  meta,
  grid,
  entities,
  cellPx: cellPxProp,
}: WorldCanvasProps): JSX.Element {
  const rows = Math.max(0, Math.floor(meta.world.rows));
  const cols = Math.max(0, Math.floor(meta.world.cols));

  const hostRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const offRef = useRef<HTMLCanvasElement | null>(null);
  const [hostWidth, setHostWidth] = useState(0);

  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;
    const measure = () => setHostWidth(host.clientWidth);
    measure();
    if (typeof ResizeObserver === "undefined") {
      window.addEventListener("resize", measure);
      return () => window.removeEventListener("resize", measure);
    }
    const ro = new ResizeObserver(measure);
    ro.observe(host);
    return () => ro.disconnect();
  }, []);

  const cellPx = useMemo(() => {
    const auto = hostWidth > 0 && cols > 0 ? Math.floor(hostWidth / cols) : 1;
    const raw = cellPxProp ?? auto;
    return Number.isFinite(raw) ? Math.max(1, Math.floor(raw)) : Math.max(1, auto);
  }, [cellPxProp, hostWidth, cols]);

  useLayoutEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || rows <= 0 || cols <= 0) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const W = cols * cellPx;
    const H = rows * cellPx;

    ctx.imageSmoothingEnabled = false;
    ctx.fillStyle = BG;
    ctx.fillRect(0, 0, W, H);

    if (grid && grid.length > 0) {
      const off = offScreen(offRef, rows, cols);
      const octx = off.getContext("2d");
      if (octx) {
        const img = octx.createImageData(cols, rows);
        const d = img.data;
        const n = rows * cols;
        const m = Math.min(grid.length, n);
        for (let i = 0; i < m; i++) {
          const o = grid[i] * 4;
          const p = i * 4;
          d[p] = LUT[o];
          d[p + 1] = LUT[o + 1];
          d[p + 2] = LUT[o + 2];
          d[p + 3] = 255;
        }
        octx.putImageData(img, 0, 0);
        ctx.drawImage(off, 0, 0, cols, rows, 0, 0, W, H);
      }
    }

    drawEntities(ctx, entities, meta, rows, cols, cellPx);
  }, [meta, grid, entities, cellPx, rows, cols]);

  return (
    <div ref={hostRef} style={{ width: "100%", overflow: "hidden" }}>
      <canvas
        ref={canvasRef}
        width={Math.max(1, cols * cellPx)}
        height={Math.max(1, rows * cellPx)}
        role="img"
        aria-label={`世界渲染 ${rows}×${cols}`}
        style={{
          display: "block",
          width: cols > 0 ? cols * cellPx : undefined,
          height: rows > 0 ? rows * cellPx : undefined,
          imageRendering: "pixelated",
        }}
      />
    </div>
  );
}

function offScreen(
  ref: { current: HTMLCanvasElement | null },
  rows: number,
  cols: number,
): HTMLCanvasElement {
  let c = ref.current;
  if (!c) {
    c = document.createElement("canvas");
    ref.current = c;
  }
  if (c.width !== cols) c.width = cols;
  if (c.height !== rows) c.height = rows;
  return c;
}

function drawEntities(
  ctx: CanvasRenderingContext2D,
  entities: Float32Array | null,
  meta: Meta,
  rows: number,
  cols: number,
  cellPx: number,
): void {
  if (!entities || entities.length < 7) return;
  const nEnt = Math.floor(entities.length / 7);
  const W = cols * cellPx;
  const H = rows * cellPx;

  const norm = meta.entities?.norm?.energy;
  let lo = 0;
  let hi = 1;
  if (norm && Number.isFinite(norm.p_lo) && Number.isFinite(norm.p_hi) && norm.p_hi > norm.p_lo) {
    lo = norm.p_lo;
    hi = norm.p_hi;
  } else {
    let mn = Infinity;
    let mx = -Infinity;
    for (let i = 0; i < nEnt; i++) {
      const e = entities[i * 7 + 3];
      if (Number.isFinite(e)) {
        if (e < mn) mn = e;
        if (e > mx) mx = e;
      }
    }
    if (Number.isFinite(mn) && mx > mn) {
      lo = mn;
      hi = mx;
    }
  }
  const span = hi - lo;

  const subdiv = resolveSubdiv(entities, nEnt, cols, meta.entities.subdiv);

  for (let i = 0; i < nEnt; i++) {
    const b = i * 7;
    const flat = entities[b];
    if (!Number.isFinite(flat)) continue;
    const row = Math.floor(flat / cols);
    const col = flat - row * cols;
    if (row < 0 || row >= rows || col < 0 || col >= cols) continue;

    let x: number;
    let y: number;
    const sr = entities[b + 1];
    const sc = entities[b + 2];
    if (subdiv > 0 && Number.isFinite(sr) && Number.isFinite(sc)) {
      y = (sr / subdiv) * cellPx;
      x = (sc / subdiv) * cellPx;
    } else {
      x = (col + 0.5) * cellPx;
      y = (row + 0.5) * cellPx;
    }
    if (y < -cellPx || y > H + cellPx) continue;

    const e = entities[b + 3];
    let t = 0.5;
    if (Number.isFinite(e) && span > 0) t = (e - lo) / span;
    t = t < 0 ? 0 : t > 1 ? 1 : t;

    const r = Math.max(cellPx * (0.3 + 0.7 * t), 0.5);
    ctx.fillStyle = energyColorCss(t);
    dot(ctx, x, y, r, W);
  }
}

function dot(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  r: number,
  W: number,
): void {
  ctx.beginPath();
  ctx.arc(x, y, r, 0, TAU);
  ctx.fill();
  if (x < r) {
    ctx.beginPath();
    ctx.arc(x + W, y, r, 0, TAU);
    ctx.fill();
  } else if (x > W - r) {
    ctx.beginPath();
    ctx.arc(x - W, y, r, 0, TAU);
    ctx.fill();
  }
}

/**
 * 解析本帧 subdiv（契约 v0 修订1 §4）：值取自 `meta.entities.subdiv`；
 * 缺失 或 无亚格数据 ⇒ 0 = 回退格中心。声明 subdiv 时做帧级一致性校验 ——
 * 判据 `sub_r // subdiv == floor(flat/cols)` 且 `sub_c // subdiv == col`
 * （同一实体的两种表示）；不过（实测：subpos 关档续跑时 sub 列与 flat 漂移）
 * ⇒ 一律 0 = 全帧回退格中心，不允许部分实体按亚格绘制。
 */
function resolveSubdiv(
  entities: Float32Array,
  nEnt: number,
  cols: number,
  metaSubdiv: number | undefined,
): number {
  if (typeof metaSubdiv !== "number" || !Number.isFinite(metaSubdiv) || metaSubdiv < 1) {
    return 0;
  }
  let total = 0;
  let bad = 0;
  for (let i = 0; i < nEnt; i++) {
    const b = i * 7;
    const sr = entities[b + 1];
    const sc = entities[b + 2];
    if (sr === 0 && sc === 0) continue;
    const flat = entities[b];
    if (!Number.isFinite(flat) || !Number.isFinite(sr) || !Number.isFinite(sc)) continue;
    const row = Math.floor(flat / cols);
    const col = flat - row * cols;
    total++;
    if (Math.floor(sr / metaSubdiv) !== row || Math.floor(sc / metaSubdiv) !== col) bad++;
  }
  if (total === 0) return 0;
  return bad === 0 ? metaSubdiv : 0;
}

export default WorldCanvas;
