/**
 * 前端运行期配置（单一真源）。
 *
 * 一律走**同源相对路径**，由 vite.config.ts 的开发代理转发：
 *   - `/api/*`  → observatory HTTP API（http://127.0.0.1:8000）
 *   - `/data/*` → `<data_root>/*`（dev 中间件直供；生产环境需自备静态服务）
 *
 * 可用 Vite 环境变量覆盖（`.env.local` 或 shell）：
 *   VITE_API_BASE       默认 `/api`
 *   VITE_DATA_BASE      默认 `/data`
 *   VITE_LIVE_DATA_BASE 默认 `/data/live/`（实时监控轮询 `<data_root>/live/<run_id>/`）
 */

/** 去掉尾斜杠，便于拼接 `/${path}` */
const trimSlash = (s: string): string => s.replace(/\/+$/, "");

const env = (key: string, fallback: string): string => {
  const v = import.meta.env[key];
  return typeof v === "string" && v.length > 0 ? v : fallback;
};

/** observatory API 基址（同源代理，默认 `/api`） */
export const API_BASE: string = trimSlash(env("VITE_API_BASE", "/api"));

/** 数据仓静态基址（dev 中间件把 `<data_root>` 挂到这里，默认 `/data`） */
export const DATA_BASE: string = trimSlash(env("VITE_DATA_BASE", "/data"));

/** 实时数据包基址：`<data_root>/live/<run_id>/`（默认 `/data/live/`） */
export const LIVE_DATA_BASE: string = (() => {
  const raw = env("VITE_LIVE_DATA_BASE", `${DATA_BASE}/live`);
  return raw.endsWith("/") ? raw : `${raw}/`;
})();

/** 由 run_id 拼出实时数据包 URL 前缀（以 `/` 结尾，喂给 loader） */
export const livePackUrl = (runId: string): string => `${LIVE_DATA_BASE}${runId}/`;

/** 由 datapack（kind+name）拼出静态 URL 前缀（以 `/` 结尾） */
export const dataPackUrl = (kind: string, name: string): string =>
  `${DATA_BASE}/${kind}/${name}/`;
