/**
 * 控制台展示层纯函数（格式化 / 状态映射）。与组件分离，便于复用与静态检查。
 */
import type { EndedReason, RunStatus } from "../app/api";

export const STATUS_LABEL: Record<RunStatus, string> = {
  pending: "排队",
  running: "运行中",
  finished: "完成",
  failed: "失败",
};

export const ENDED_REASON_LABEL: Record<EndedReason, string> = {
  exit_ok: "正常退出",
  exit_nonzero: "非零退出",
  cancelled: "已取消",
  spawn_error: "启动失败",
  unknown_after_restart: "重启后状态未知",
};

/** 状态 → LED class 后缀 */
export const ledClass = (s: RunStatus | string | null | undefined): string => {
  switch (s) {
    case "running":
      return "led-run";
    case "finished":
      return "led-ok";
    case "pending":
      return "led-pending";
    case "failed":
      return "led-failed";
    default:
      return "led-idle";
  }
};

export const badgeClass = (s: RunStatus | string): string => `badge badge-${s}`;

export function fmtBytes(n: number | null | undefined): string {
  if (n === null || n === undefined || !Number.isFinite(n)) return "—";
  const u = ["B", "KB", "MB", "GB", "TB"];
  let v = n;
  let i = 0;
  while (v >= 1024 && i < u.length - 1) {
    v /= 1024;
    i += 1;
  }
  return `${i === 0 ? v.toFixed(0) : v.toFixed(v >= 100 ? 0 : 1)} ${u[i]}`;
}

export function fmtNum(n: number | null | undefined): string {
  if (n === null || n === undefined || !Number.isFinite(n)) return "—";
  return n.toLocaleString("en-US");
}

/** ISO8601 → HH:MM:SS（UTC，仪器读数用） */
export function clockUTC(iso: string | null | undefined): string {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  const p = (x: number): string => String(x).padStart(2, "0");
  return `${p(d.getUTCHours())}:${p(d.getUTCMinutes())}:${p(d.getUTCSeconds())}`;
}

/** ISO8601 → "x 秒前 / x 分前"（相对时间，台账用） */
export function timeAgo(iso: string | null | undefined, nowMs = Date.now()): string {
  if (!iso) return "—";
  const t = new Date(iso).getTime();
  if (Number.isNaN(t)) return iso;
  const s = Math.max(0, Math.round((nowMs - t) / 1000));
  if (s < 5) return "刚刚";
  if (s < 60) return `${s} 秒前`;
  const m = Math.round(s / 60);
  if (m < 60) return `${m} 分前`;
  const h = Math.round(m / 60);
  if (h < 24) return `${h} 小时前`;
  return `${Math.round(h / 24)} 天前`;
}

/** 日志行着色：含 error/traceback/exception ⇒ err；warn/⚠ ⇒ warn */
export function logLineClass(line: string): string {
  const l = line.toLowerCase();
  if (l.includes("error") || l.includes("traceback") || l.includes("exception") || l.includes("failed"))
    return "ln err";
  if (l.includes("warn") || l.includes("⚠")) return "ln warn";
  return "ln";
}

/** 把 device.fields 值格式化为短字符串（人眼复核） */
export function fmtFieldValue(v: unknown): string {
  if (v === null || v === undefined) return "—";
  if (typeof v === "number") return Number.isInteger(v) ? String(v) : v.toFixed(4).replace(/0+$/, "");
  if (typeof v === "boolean") return v ? "true" : "false";
  return String(v);
}
