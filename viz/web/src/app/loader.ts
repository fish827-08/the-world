/**
 * 数据包加载器 —— 契约 v0（viz/CONTRACT.md 唯一接口）。
 *
 * 布局：meta.json / frames/ch_<key>/NNNNN.bin / entities/NNNNN.f32 / series.csv（可选）
 * 帧数据按需拉取；所有 fetch 支持 AbortSignal（播放/拖动时取消过期请求）。
 */
import type { Meta } from "../contract";

/** 帧文件命名：5 位零填充（≥100000 帧时自然溢出位数，契约 v0 无超帧包） */
export const frameFileName = (k: number): string => String(k).padStart(5, "0");

const withSlash = (base: string): string => (base.endsWith("/") ? base : `${base}/`);

async function fetchOrThrow(url: string, what: string, signal?: AbortSignal): Promise<Response> {
  const res = await fetch(url, { signal });
  if (!res.ok) throw new Error(`${what} 加载失败：HTTP ${res.status}（${url}）`);
  // dev/preview 静态服务对缺失路径会 SPA 兜底返回 index.html ⇒ 必须挡掉，防把 HTML 当数据
  if ((res.headers.get("content-type") ?? "").includes("text/html")) {
    throw new Error(`${what} 不存在：服务器返回 HTML 兜底页（${url}）`);
  }
  return res;
}

/** meta.json → Meta；不认识的大版本报错，未知字段忽略（契约 §0） */
export async function loadMeta(base: string, signal?: AbortSignal): Promise<Meta> {
  const res = await fetchOrThrow(`${withSlash(base)}meta.json`, "meta.json", signal);
  const meta = (await res.json()) as Meta;
  if (!String(meta.contract_version ?? "").startsWith("v0")) {
    throw new Error(`数据包契约版本不支持：${meta.contract_version}（前端支持 v0）`);
  }
  return meta;
}

/** frames/ch_<key>/NNNNN.bin → Uint8Array（行主序 rows*cols） */
export async function loadGridFrame(
  base: string,
  key: string,
  k: number,
  signal?: AbortSignal,
): Promise<Uint8Array> {
  const url = `${withSlash(base)}frames/ch_${key}/${frameFileName(k)}.bin`;
  const res = await fetchOrThrow(url, `通道帧 ${key}/${frameFileName(k)}`, signal);
  return new Uint8Array(await res.arrayBuffer());
}

/** 实体交错缓冲步长：flat, sub_r, sub_c, energy, age, generation, mode */
export const ENTITY_STRIDE = 7;

/** entities/NNNNN.f32 → Float32Array（小端 7×f32 交错；DataView 显式小端读取） */
export async function loadEntities(
  base: string,
  k: number,
  signal?: AbortSignal,
): Promise<Float32Array> {
  const url = `${withSlash(base)}entities/${frameFileName(k)}.f32`;
  const res = await fetchOrThrow(url, `实体帧 ${frameFileName(k)}`, signal);
  const buf = await res.arrayBuffer();
  const dv = new DataView(buf);
  const n = Math.floor(buf.byteLength / 4);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i += 1) out[i] = dv.getFloat32(i * 4, true);
  return out;
}

export interface SeriesData {
  columns: string[];
  rows: number[][];
}

/** series.csv（可选）→ SeriesData；缺失（404 或 SPA 兜底页）⇒ null（不是错误） */
export async function loadSeries(base: string, signal?: AbortSignal): Promise<SeriesData | null> {
  const res = await fetch(`${withSlash(base)}series.csv`, { signal });
  if (res.status === 404) return null;
  if (!res.ok) throw new Error(`series.csv 加载失败：HTTP ${res.status}`);
  if ((res.headers.get("content-type") ?? "").includes("text/html")) return null;
  return parseSeriesCsv(await res.text());
}

/** 解析逐 tick 序列 CSV（表头为准；空字段 → NaN，图表自动断线） */
export function parseSeriesCsv(text: string): SeriesData {
  const lines = text.split(/\r?\n/).filter((l) => l.trim().length > 0);
  // 表头须是逗号分隔多列（单字段行按"非 CSV"处理，防 HTML/文本兜底内容混入）
  if (lines.length === 0 || !lines[0].includes(",")) return { columns: [], rows: [] };
  const columns = lines[0].split(",").map((c) => c.trim());
  const rows = lines.slice(1).map((l) => {
    const parts = l.split(",");
    return columns.map((_, i) => {
      const t = (parts[i] ?? "").trim();
      return t === "" ? NaN : Number(t);
    });
  });
  return { columns, rows };
}

/** 取某列全部值（列不存在 ⇒ 空数组） */
export function seriesColumn(s: SeriesData, name: string): number[] {
  const i = s.columns.indexOf(name);
  if (i < 0) return [];
  return s.rows.map((r) => r[i]);
}
