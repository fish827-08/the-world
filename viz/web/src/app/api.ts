/**
 * observatory HTTP API 客户端（v1 契约，类型化）。
 *
 * 约定（见任务契约 / observatory/api.py）：
 *   - 一律同源 `/api/*`（vite 代理转发到 127.0.0.1:8000）。
 *   - 所有非 2xx 返回**统一错误信封** `{error_code, message, detail}`；
 *     422 的 `detail.fields = [{field, msg, type}]`（请求体 extra=forbid）。
 *   - 响应 JSON / UTF-8 / ISO8601 UTC。
 *
 * 🔴 语义红线：POST /snapshots 的 live 快照与 tick 推进**非原子**，仅供取帧，
 *    **不得当实验判据数据**（UI 侧须显著提示）。
 */
import { API_BASE } from "./config";

/* ------------------------------------------------------------------ */
/* 错误信封                                                             */
/* ------------------------------------------------------------------ */

export interface ValidationField {
  field: string;
  msg: string;
  type: string;
}

export interface ErrorEnvelope {
  error_code: string;
  message: string;
  detail: unknown;
}

/** 带机器可读 error_code 的 API 错误（前端只认这一套形状） */
export class ApiError extends Error {
  readonly status: number;
  readonly errorCode: string;
  readonly detail: unknown;

  constructor(status: number, envelope: ErrorEnvelope) {
    super(envelope.message || `HTTP ${status}`);
    this.name = "ApiError";
    this.status = status;
    this.errorCode = envelope.error_code || "unknown";
    this.detail = envelope.detail;
  }

  /** 422 逐字段提示（非 422 或无 fields ⇒ 空数组） */
  get fields(): ValidationField[] {
    const d = this.detail as { fields?: ValidationField[] } | null;
    return Array.isArray(d?.fields) ? d!.fields! : [];
  }
}

const isApiError = (e: unknown): e is ApiError => e instanceof ApiError;

/** 把任意异常归一成一句人话（网络错误 / ApiError / 其它） */
export function describeError(e: unknown): string {
  if (isApiError(e)) {
    const tail =
      e.errorCode && e.errorCode !== "unknown" ? `（${e.errorCode}）` : "";
    return `${e.message}${tail}`;
  }
  if (e instanceof Error) return e.message;
  return String(e);
}

/* ------------------------------------------------------------------ */
/* 契约类型                                                             */
/* ------------------------------------------------------------------ */

export type RunStatus = "pending" | "running" | "finished" | "failed";

export type EndedReason =
  | "exit_ok"
  | "exit_nonzero"
  | "cancelled"
  | "spawn_error"
  | "unknown_after_restart";

export type SnapshotMode = "live" | "inproc";

export interface Health {
  service: string;
  api_version: string;
  status: string;
  server_time: string;
  python: string;
  repo_root: string;
  data_root: string;
  data_root_exists: boolean;
  default_script: string;
  default_script_exists: boolean;
  devices: string[];
  registered_runs: number;
  endpoints: string[];
}

/** 装置预设字段（R278：rows/cols/patches/pop/rgm/… 须回显给人眼复核） */
export type DeviceFields = Record<string, string | number | boolean | null>;

export interface DeviceInfo {
  name: string;
  fields: DeviceFields;
}

export interface DevicesResponse {
  source: string;
  count: number;
  devices: DeviceInfo[];
  note: string;
}

export interface SnapshotRecord {
  path: string;
  rel_path: string;
  tick: number | null;
  size_bytes: number;
  mode: SnapshotMode | string;
  created_at: string;
  config_fingerprint: string | null;
  gene_count: number | null;
  snapshot_version: number | null;
}

export interface Run {
  run_id: string;
  name: string;
  device: string;
  seed: number;
  ticks: number;
  arm: number | null;
  script: string;
  args: string[];
  command: string[];
  live: boolean;
  out_dir: string;
  log_path: string;
  status: RunStatus;
  pid: number | null;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  returncode: number | null;
  ended_reason: EndedReason | null;
  note: string | null;
  snapshots: SnapshotRecord[];
  log_tail: string[] | null;
}

export interface RunListResponse {
  count: number;
  total: number;
  by_status: Partial<Record<RunStatus, number>>;
  server_time: string;
  runs: Run[];
}

export interface ExperimentCreateBody {
  device: string;
  seed: number;
  ticks: number;
  arm?: number | null;
  name?: string | null;
  script?: string;
  extra_args?: string[];
  /** 结构化参数（键须在 GET /param-schema 白名单内；留空 = 用脚本默认） */
  params?: Record<string, number | boolean | string>;
  live?: boolean;
}

export interface SnapshotCreateBody {
  run_id: string;
  tick?: number | null;
  wait_s?: number;
}

/** POST /snapshots 响应（**只有元数据**，绝不回 npz 内容） */
export interface SnapshotMeta {
  run_id?: string;
  mode: SnapshotMode | string;
  request_id?: string;
  path: string;
  rel_path: string;
  tick: number | null;
  size_bytes: number;
  config_fingerprint: string | null;
  gene_count: number | null;
  snapshot_version: number | null;
  created_at: string;
  engine_tick_at_request?: number | null;
}

export interface DataPackStats {
  n_files: number;
  size_bytes: number;
  n_frame_files: number;
  n_entity_files: number;
  has_series_csv: boolean;
}

export type DataPackKind = "viz" | "live";

export interface DataPack {
  id: string;
  kind: DataPackKind;
  name: string;
  path: string;
  meta_source: string | null;
  /** 契约 v0 原文（或 run.json / *.manifest.json）；include_meta=false ⇒ null */
  meta: Record<string, unknown> | null;
  contract_version: string | null;
  world: { rows: number; cols: number; n_cells: number | null } | null;
  tick: {
    start: number | null;
    end: number | null;
    stride: number | null;
    n_frames: number | null;
  } | null;
  stats: DataPackStats | null;
  run_status: RunStatus | null;
  modified_at: string | null;
  error: string | null;
}

export interface DataPackIndex {
  data_root: string;
  data_root_exists: boolean;
  generated_at: string;
  count: number;
  packs: DataPack[];
}

/* ------------------------------------------------------------------ */
/* 传输层                                                               */
/* ------------------------------------------------------------------ */

async function parseEnvelope(res: Response): Promise<ErrorEnvelope> {
  let raw: unknown = null;
  try {
    raw = await res.json();
  } catch {
    raw = null;
  }
  if (raw && typeof raw === "object" && "error_code" in raw) {
    const r = raw as Partial<ErrorEnvelope>;
    return {
      error_code: String(r.error_code ?? "unknown"),
      message: String(r.message ?? `HTTP ${res.status}`),
      detail: r.detail ?? null,
    };
  }
  // FastAPI 原生 422 兜底（detail 是数组）——归一到同一信封
  if (raw && typeof raw === "object" && "detail" in raw) {
    const d = (raw as { detail: unknown }).detail;
    const fields = Array.isArray(d)
      ? d.map((x) => {
          const o = (x ?? {}) as Record<string, unknown>;
          const loc = Array.isArray(o.loc)
            ? o.loc.filter((s) => s !== "body").map(String)
            : [];
          return {
            field: loc.join("."),
            msg: String(o.msg ?? ""),
            type: String(o.type ?? ""),
          };
        })
      : undefined;
    return {
      error_code: "invalid_request",
      message: `请求被拒绝：HTTP ${res.status}`,
      detail: fields ? { fields } : d,
    };
  }
  return {
    error_code: "http_error",
    message: `HTTP ${res.status} ${res.statusText}`.trim(),
    detail: typeof raw === "string" ? raw : null,
  };
}

async function request<T>(
  path: string,
  init?: RequestInit & { json?: unknown },
): Promise<T> {
  const url = `${API_BASE}${path}`;
  const { json, ...rest } = init ?? {};
  const headers = new Headers(rest.headers);
  let body = rest.body;
  if (json !== undefined) {
    headers.set("Content-Type", "application/json");
    body = JSON.stringify(json);
  }
  let res: Response;
  try {
    res = await fetch(url, { ...rest, headers, body });
  } catch (e) {
    // 网络层失败（后端没起 / 代理未通）——归一成 ApiError 便于 UI 统一展示
    throw new ApiError(0, {
      error_code: "network_error",
      message: `无法连接 API（${url}）：${e instanceof Error ? e.message : String(e)}`,
      detail: null,
    });
  }
  if (!res.ok) throw new ApiError(res.status, await parseEnvelope(res));
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

const qs = (params: object): string => {
  const sp = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) {
    if (v !== undefined && v !== null) sp.set(k, String(v));
  }
  const s = sp.toString();
  return s ? `?${s}` : "";
};

/* ------------------------------------------------------------------ */
/* 端点                                                                 */
/* ------------------------------------------------------------------ */

export const getHealth = (signal?: AbortSignal): Promise<Health> =>
  request<Health>("/health", { signal });

export const getDevices = (signal?: AbortSignal): Promise<DevicesResponse> =>
  request<DevicesResponse>("/devices", { signal });

export const createExperiment = (
  body: ExperimentCreateBody,
  signal?: AbortSignal,
): Promise<Run> =>
  request<Run>("/experiments", { method: "POST", json: body, signal });

export interface ListExperimentsParams {
  status?: RunStatus;
  include_log?: boolean;
  limit?: number;
}

export const listExperiments = (
  params: ListExperimentsParams = {},
  signal?: AbortSignal,
): Promise<RunListResponse> =>
  request<RunListResponse>(`/experiments${qs(params)}`, { signal });

export const getExperiment = (
  runId: string,
  tailLines = 40,
  signal?: AbortSignal,
): Promise<Run> =>
  request<Run>(`/experiments/${encodeURIComponent(runId)}${qs({ tail_lines: tailLines })}`, {
    signal,
  });

export const cancelExperiment = (
  runId: string,
  graceS?: number,
  signal?: AbortSignal,
): Promise<Run> =>
  request<Run>(
    `/experiments/${encodeURIComponent(runId)}/cancel${qs({ grace_s: graceS })}`,
    { method: "POST", signal },
  );

export const createSnapshot = (
  body: SnapshotCreateBody,
  signal?: AbortSignal,
): Promise<SnapshotMeta> =>
  request<SnapshotMeta>("/snapshots", { method: "POST", json: body, signal });

export interface DataPacksParams {
  kind?: "all" | DataPackKind;
  include_meta?: boolean;
  stats?: boolean;
}

export const getDatapacks = (
  params: DataPacksParams = {},
  signal?: AbortSignal,
): Promise<DataPackIndex> =>
  request<DataPackIndex>(`/datapacks${qs(params)}`, { signal });

/* ---------------- 实验参数 schema（启动表单“高级参数”渲染依据） -------- */

export type ParamType = "int" | "float" | "bool" | "choice";

/** GET /param-schema 单个参数项（default 恒为 null ⇒ 留空不传旗标，用脚本默认） */
export interface ParamSchemaItem {
  key: string;
  /** 脚本旗标（kebab-case，如 --diff-du-dv） */
  flag: string;
  label: string;
  type: ParamType;
  default: number | boolean | string | null;
  min: number | null;
  max: number | null;
  choices: string[] | null;
  unit: string | null;
  group: string;
  /** 如“覆盖默认将产生新配置，读数不与历史可比” */
  warning: string | null;
}

/** 参数分组元数据（collapsed = 前端默认折叠，如自发分化组） */
export interface ParamGroupMeta {
  key: string;
  label: string;
  note: string | null;
  collapsed: boolean;
}

/** GET /param-schema 响应（未知 script ⇒ groups/params 均空，不报错） */
export interface ParamSchema {
  script: string;
  script_exists: boolean;
  groups: ParamGroupMeta[];
  params: ParamSchemaItem[];
}

export const getParamSchema = (
  script: string,
  signal?: AbortSignal,
): Promise<ParamSchema> =>
  request<ParamSchema>(`/param-schema${qs({ script })}`, { signal });

/* ---------------- 快照加载（快照 npz → 导出 v0 数据包 → 载入回放） -------- */

/** GET /snapshots 列表项（snapshot 为 data_root 相对路径） */
export interface SnapshotEntry {
  snapshot: string;
  size_bytes: number;
  tick: number | null;
  run_id: string | null;
  modified_at: string | null;
}

/** GET /snapshots 响应（skipped = 被排除的非快照/不可读 npz 数） */
export interface SnapshotIndex {
  data_root: string;
  count: number;
  skipped: number;
  snapshots: SnapshotEntry[];
}

export interface SnapshotExportBody {
  /** data_root 相对路径，须 .npz 结尾（后端防穿越校验） */
  snapshot: string;
  /** 数据包名（slug 化，≤64）；缺省 = UTC 时间戳 */
  name?: string | null;
  /** 续跑 tick 数；0 = 仅当前帧（静态单帧查看）。默认 0 */
  ticks?: number;
  /** 采样间隔 tick/帧。默认 20 */
  stride?: number;
  /** 导出通道（resource/energy/fruit）。默认 resource,energy */
  channels?: string[];
}

/** POST /snapshots/export 响应（pack 可直接经 /data/viz/<name>/ 载入） */
export interface SnapshotExportResult {
  pack_id: string;
  name: string;
  path: string;
  n_frames: number;
  rows: number;
  cols: number;
  size_bytes: number;
}

export const listSnapshots = (
  limit?: number,
  signal?: AbortSignal,
): Promise<SnapshotIndex> =>
  request<SnapshotIndex>(`/snapshots${qs({ limit })}`, { signal });

export const exportSnapshot = (
  body: SnapshotExportBody,
  signal?: AbortSignal,
): Promise<SnapshotExportResult> =>
  request<SnapshotExportResult>("/snapshots/export", {
    method: "POST",
    json: body,
    signal,
  });
