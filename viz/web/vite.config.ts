import { fileURLToPath } from "node:url";
import nodePath from "node:path";
import fs from "node:fs";
import type { ServerResponse } from "node:http";
import react from "@vitejs/plugin-react";
import { defineConfig, type Plugin } from "vite";

// https://vite.dev/config/

/** vite.config.ts 所在目录（viz/web） */
const configDir = fileURLToPath(new URL(".", import.meta.url));

/** observatory API 源（可用 THE_WORLD_API_ORIGIN 覆盖） */
const API_ORIGIN = process.env.THE_WORLD_API_ORIGIN ?? "http://127.0.0.1:8000";

/**
 * 数据仓根：THE_WORLD_DATA_ROOT 优先，缺省 `<worktree_root>/the-world-data`。
 * 与后端 `observatory/api.py:get_data_root()` 同口径（缺省 repo_root/the-world-data）。
 * ⚠️ VIZ worktree 分支可能没有该目录（落后 main）——中间件对缺失目录优雅 404。
 */
const DATA_ROOT = nodePath.resolve(
  process.env.THE_WORLD_DATA_ROOT ?? nodePath.join(configDir, "..", "..", "the-world-data"),
);

/** URL 前缀 → 本地数据仓静态服务（dev/preview 专用；生产环境需自备静态服务或反代） */
const DATA_URL_PREFIX = "/data/";

const CONTENT_TYPES: Record<string, string> = {
  ".json": "application/json; charset=utf-8",
  ".csv": "text/csv; charset=utf-8",
  ".bin": "application/octet-stream",
  ".f32": "application/octet-stream",
  ".npz": "application/octet-stream",
  ".log": "text/plain; charset=utf-8",
  ".txt": "text/plain; charset=utf-8",
};

/**
 * 把 `<data_root>` 挂到 `/data/`：实时监控（A4）轮询 `/data/live/<run_id>/meta.json`
 * 与增量帧、数据包浏览器加载 `/data/{viz,live}/<name>/` 都走这里。
 * 后端 API **不提供静态文件服务**（契约只有 JSON 端点），故由 dev server 直供字节。
 */
function handleDataRequest(
  url: string | undefined,
  res: ServerResponse,
  next: (err?: unknown) => void,
): void {
  const u = url ?? "";
  if (!u.startsWith(DATA_URL_PREFIX)) {
    next();
    return;
  }
  let rel: string;
  try {
    rel = decodeURIComponent(u.slice(DATA_URL_PREFIX.length).split("?")[0]);
  } catch {
    next();
    return;
  }
  const abs = nodePath.resolve(DATA_ROOT, rel);
  // 路径穿越守卫：解析后必须仍在 DATA_ROOT 之内
  if (abs !== DATA_ROOT && !abs.startsWith(DATA_ROOT + nodePath.sep)) {
    res.statusCode = 403;
    res.end("Forbidden");
    return;
  }
  fs.stat(abs, (err, st) => {
    if (err || !st.isFile()) {
      // /data/ 下的缺失一律显式 404——绝不回落到 SPA index.html，防把 HTML 当数据
      res.statusCode = 404;
      res.setHeader("Content-Type", "text/plain; charset=utf-8");
      res.end("Not found");
      return;
    }
    const ext = nodePath.extname(abs).toLowerCase();
    res.setHeader("Content-Type", CONTENT_TYPES[ext] ?? "application/octet-stream");
    res.setHeader("Content-Length", st.size);
    // 实时数据逐帧增长，禁用缓存以免读到旧帧/旧 meta
    res.setHeader("Cache-Control", "no-store, max-age=0");
    res.statusCode = 200;
    const stream = fs.createReadStream(abs);
    stream.on("error", () => {
      res.statusCode = 500;
      res.end("Read error");
    });
    stream.pipe(res);
  });
}

function serveDataRoot(): Plugin {
  return {
    name: "the-world-serve-data-root",
    configureServer(server) {
      server.middlewares.use((req, res, next) => handleDataRequest(req.url, res, next));
    },
    configurePreviewServer(server) {
      server.middlewares.use((req, res, next) => handleDataRequest(req.url, res, next));
    },
  };
}

export default defineConfig({
  plugins: [react(), serveDataRoot()],
  server: {
    proxy: {
      // 前端一律调同源 /api/*，代理转发到 observatory（去掉 /api 前缀）
      "/api": {
        target: API_ORIGIN,
        changeOrigin: true,
        rewrite: (p) => p.replace(/^\/api/, ""),
      },
    },
  },
});
