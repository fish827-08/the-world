/**
 * 实验控制台状态（Zustand）—— A3 控制台 / A4 实时监控发现 / A5 多 run 对比数据源。
 *
 * 轮询：控制台挂载时 `setPolling(true)` 起 1s 循环——
 *   1) `GET /experiments`（列表 + by_status，廉价）；
 *   2) 对 running/pending 与展开中的 run 各打 `GET /experiments/{id}?tail_lines`（契约轮询建议）。
 * 竞态：所有请求带自增 token，只有最新一轮落状态；卸载 `setPolling(false)` 清定时器。
 */
import { create } from "zustand";
import {
  ApiError,
  cancelExperiment,
  createExperiment,
  createSnapshot,
  getDatapacks,
  getDevices,
  getExperiment,
  getHealth,
  listExperiments,
  listSnapshots,
  type DataPack,
  type DeviceInfo,
  type Health,
  type Run,
  type RunStatus,
  type SnapshotEntry,
  type SnapshotMeta,
} from "../app/api";
import { loadSeries, type SeriesData } from "../app/loader";

/* ------------------------------------------------------------------ */
/* A5：trait 漂移 + 多 run 对比 的规整形状                              */
/* ------------------------------------------------------------------ */

/** 一条 trait 漂移曲线（归一后）：name + [tick, value][] */
export interface TraitSeries {
  name: string;
  points: [number, number][];
}

/** 多 run 对比：一条来自某数据包/run 的序列 */
export interface CompareEntry {
  label: string;
  color: string;
  url: string;
  data: SeriesData | null;
  error: string | null;
  loading: boolean;
}

export const COMPARE_COLORS = [
  "#4cc9f0",
  "#ffd166",
  "#57cc99",
  "#ff6b6b",
  "#c792ea",
  "#f78c6c",
  "#82aaff",
  "#addb67",
];

/**
 * 把 `manifest.summary.trait_drift` 归一成 TraitSeries[]。
 * 容忍多种形状：{trait:number[]} / {trait:{tick,value}} / [{trait,tick,value}]。
 * 认不出 ⇒ 空数组（"列存在即渲染"，不硬造）。
 */
export function normalizeTraitDrift(raw: unknown): TraitSeries[] {
  if (!raw) return [];
  const out: TraitSeries[] = [];
  const pushNums = (name: string, ticks: number[], vals: number[]): void => {
    const pts: [number, number][] = [];
    for (let i = 0; i < vals.length; i += 1) {
      const v = vals[i];
      if (Number.isFinite(v)) pts.push([Number.isFinite(ticks[i]) ? ticks[i] : i, v]);
    }
    if (pts.length) out.push({ name, points: pts });
  };
  if (Array.isArray(raw)) {
    // [{trait,tick,value}] 长表
    const byTrait = new Map<string, { t: number[]; v: number[] }>();
    for (const row of raw) {
      const o = row as Record<string, unknown>;
      const name = String(o.trait ?? o.name ?? o.gene ?? "trait");
      const t = Number(o.tick ?? o.t ?? NaN);
      const v = Number(o.value ?? o.val ?? o.mean ?? NaN);
      const e = byTrait.get(name) ?? { t: [], v: [] };
      e.t.push(t);
      e.v.push(v);
      byTrait.set(name, e);
    }
    for (const [name, e] of byTrait) pushNums(name, e.t, e.v);
    return out;
  }
  if (typeof raw === "object") {
    for (const [name, val] of Object.entries(raw as Record<string, unknown>)) {
      if (Array.isArray(val)) {
        const nums = val.map(Number);
        if (nums.every((n) => Number.isFinite(n))) {
          pushNums(name, nums.map((_, i) => i), nums);
        } else {
          // [{tick,value}]
          const t = val.map((x) => Number((x as Record<string, unknown>)?.tick ?? NaN));
          const v = val.map((x) => Number((x as Record<string, unknown>)?.value ?? NaN));
          pushNums(name, t, v);
        }
      } else if (val && typeof val === "object") {
        const o = val as Record<string, unknown>;
        const t = Array.isArray(o.tick) ? o.tick.map(Number) : [];
        const v = Array.isArray(o.value) ? o.value.map(Number) : [];
        if (v.length) pushNums(name, t.length === v.length ? t : v.map((_, i) => i), v);
      } else if (typeof val === "number" && Number.isFinite(val)) {
        out.push({ name, points: [[0, val]] });
      }
    }
  }
  return out;
}

/** 从任意 meta（契约 v0 / manifest / run.json）里挖 summary.trait_drift */
export function extractTraitDrift(meta: unknown): TraitSeries[] | null {
  if (!meta || typeof meta !== "object") return null;
  const summary = (meta as Record<string, unknown>).summary;
  if (!summary || typeof summary !== "object") return null;
  const td = (summary as Record<string, unknown>).trait_drift;
  if (td === undefined) return null;
  const norm = normalizeTraitDrift(td);
  return norm.length ? norm : null;
}

/* ------------------------------------------------------------------ */
/* Store                                                               */
/* ------------------------------------------------------------------ */

export interface ConsoleState {
  /* 健康 / 装置 */
  health: Health | null;
  apiUp: boolean;
  healthError: string | null;
  devices: DeviceInfo[];
  devicesNote: string;
  devicesSource: string;

  /* 运行台账 */
  runs: Run[];
  byStatus: Partial<Record<RunStatus, number>>;
  runsTotal: number;
  details: Record<string, Run>;
  expanded: string | null;
  listError: string | null;

  /* 启动实验 */
  submitting: boolean;
  submitError: ApiError | null;
  lastCreatedRunId: string | null;

  /* 操作（cancel） */
  busyRunId: string | null;
  actionError: string | null;

  /* 快照 */
  snapBusy: boolean;
  snapResult: SnapshotMeta | null;
  snapError: ApiError | null;

  /* 数据包索引 */
  datapacks: DataPack[];
  datapacksError: string | null;
  datapacksLoading: boolean;
  dataRootExists: boolean | null;

  /* A6 快照加载（GET /snapshots） */
  snapshots: SnapshotEntry[] | null;
  snapshotsSkipped: number;
  snapshotsError: ApiError | null;
  snapshotsLoading: boolean;

  /* A5 对比 / trait 漂移 */
  compare: CompareEntry[];
  traitDrift: TraitSeries[] | null;

  polling: boolean;

  /* actions */
  bootstrap: () => Promise<void>;
  refreshHealth: () => Promise<void>;
  refreshDevices: () => Promise<void>;
  refreshList: () => Promise<void>;
  refreshDetail: (runId: string) => Promise<void>;
  refreshDatapacks: () => Promise<void>;
  refreshSnapshots: () => Promise<void>;
  setPolling: (on: boolean) => void;
  toggleExpand: (runId: string) => void;
  submitExperiment: (body: Parameters<typeof createExperiment>[0]) => Promise<Run | null>;
  cancelRun: (runId: string) => Promise<void>;
  takeSnapshot: (runId: string, tick?: number | null) => Promise<void>;
  clearSnapshot: () => void;
  addCompare: (label: string, url: string) => Promise<void>;
  removeCompare: (label: string) => void;
  clearCompare: () => void;
  setTraitDrift: (meta: unknown) => void;
}

let pollTimer: ReturnType<typeof setInterval> | null = null;
let listToken = 0;
let bootstrapped = false;

const errText = (e: unknown): string =>
  e instanceof Error ? e.message : String(e);

export const useConsole = create<ConsoleState>()((set, get) => {
  async function pollOnce(): Promise<void> {
    const my = ++listToken;
    try {
      const data = await listExperiments({ limit: 300 });
      if (my !== listToken) return;
      set({
        runs: data.runs,
        byStatus: data.by_status,
        runsTotal: data.total,
        listError: null,
      });
      // 对活跃 run + 展开 run 拉详情（带 log_tail）
      const expanded = get().expanded;
      const active = data.runs.filter(
        (r) => r.status === "running" || r.status === "pending" || r.run_id === expanded,
      );
      await Promise.all(
        active.slice(0, 10).map((r) =>
          getExperiment(r.run_id, 60)
            .then((d) => {
              if (my !== listToken) return;
              set((s) => ({ details: { ...s.details, [d.run_id]: d } }));
            })
            .catch(() => undefined),
        ),
      );
    } catch (e) {
      if (my !== listToken) return;
      set({ listError: errText(e), apiUp: false });
    }
  }

  return {
    health: null,
    apiUp: false,
    healthError: null,
    devices: [],
    devicesNote: "",
    devicesSource: "",
    runs: [],
    byStatus: {},
    runsTotal: 0,
    details: {},
    expanded: null,
    listError: null,
    submitting: false,
    submitError: null,
    lastCreatedRunId: null,
    busyRunId: null,
    actionError: null,
    snapBusy: false,
    snapResult: null,
    snapError: null,
    datapacks: [],
    datapacksError: null,
    datapacksLoading: false,
    dataRootExists: null,
    snapshots: null,
    snapshotsSkipped: 0,
    snapshotsError: null,
    snapshotsLoading: false,
    compare: [],
    traitDrift: null,
    polling: false,

    refreshHealth: async () => {
      try {
        const h = await getHealth();
        set({ health: h, apiUp: true, healthError: null });
      } catch (e) {
        set({ apiUp: false, healthError: errText(e) });
      }
    },

    refreshDevices: async () => {
      try {
        const d = await getDevices();
        set({ devices: d.devices, devicesNote: d.note, devicesSource: d.source });
      } catch (e) {
        set({ devicesNote: errText(e) });
      }
    },

    refreshList: async () => {
      await pollOnce();
    },

    refreshDetail: async (runId) => {
      try {
        const d = await getExperiment(runId, 120);
        set((s) => ({ details: { ...s.details, [d.run_id]: d } }));
      } catch {
        /* 忽略：列表轮询会兜底 */
      }
    },

    refreshDatapacks: async () => {
      set({ datapacksLoading: true });
      try {
        const idx = await getDatapacks({ kind: "all", include_meta: true, stats: true });
        set({
          datapacks: idx.packs,
          dataRootExists: idx.data_root_exists,
          datapacksError: null,
          datapacksLoading: false,
        });
      } catch (e) {
        set({ datapacksError: errText(e), datapacksLoading: false });
      }
    },

    refreshSnapshots: async () => {
      set({ snapshotsLoading: true });
      try {
        const idx = await listSnapshots();
        set({
          snapshots: idx.snapshots,
          snapshotsSkipped: idx.skipped,
          snapshotsError: null,
          snapshotsLoading: false,
        });
      } catch (e) {
        set({
          snapshotsError:
            e instanceof ApiError
              ? e
              : new ApiError(0, {
                  error_code: "network_error",
                  message: e instanceof Error ? e.message : String(e),
                  detail: null,
                }),
          snapshotsLoading: false,
        });
      }
    },

    bootstrap: async () => {
      const g = get();
      await Promise.all([
              g.refreshHealth(),
              g.refreshDevices(),
              g.refreshDatapacks(),
              g.refreshSnapshots(),
            ]);
      await pollOnce();
      bootstrapped = true;
    },

    setPolling: (on) => {
      if (on) {
        if (pollTimer === null) {
          void pollOnce();
          pollTimer = setInterval(() => void pollOnce(), 1000);
        }
        set({ polling: true });
      } else if (pollTimer !== null) {
        clearInterval(pollTimer);
        pollTimer = null;
        set({ polling: false });
      }
    },

    toggleExpand: (runId) => {
      const cur = get().expanded;
      const next = cur === runId ? null : runId;
      set({ expanded: next });
      if (next) void get().refreshDetail(next);
    },

    submitExperiment: async (body) => {
      set({ submitting: true, submitError: null });
      try {
        const run = await createExperiment(body);
        set((s) => ({
          submitting: false,
          lastCreatedRunId: run.run_id,
          expanded: run.run_id,
          details: { ...s.details, [run.run_id]: run },
          runs: [run, ...s.runs.filter((r) => r.run_id !== run.run_id)],
        }));
        void get().refreshList();
        return run;
      } catch (e) {
        set({ submitting: false, submitError: e instanceof ApiError ? e : new ApiError(0, { error_code: "network_error", message: errText(e), detail: null }) });
        return null;
      }
    },

    cancelRun: async (runId) => {
      set({ busyRunId: runId, actionError: null });
      try {
        const run = await cancelExperiment(runId);
        set((s) => ({
          busyRunId: null,
          details: { ...s.details, [run.run_id]: run },
          runs: s.runs.map((r) => (r.run_id === run.run_id ? run : r)),
        }));
      } catch (e) {
        set({ busyRunId: null, actionError: errText(e) });
      }
    },

    takeSnapshot: async (runId, tick) => {
      set({ snapBusy: true, snapError: null });
      try {
        const meta = await createSnapshot({ run_id: runId, tick: tick ?? null });
        set({ snapBusy: false, snapResult: meta });
        void get().refreshDetail(runId);
      } catch (e) {
        set({
          snapBusy: false,
          snapError:
            e instanceof ApiError
              ? e
              : new ApiError(0, { error_code: "network_error", message: errText(e), detail: null }),
        });
      }
    },

    clearSnapshot: () => set({ snapResult: null, snapError: null }),

    addCompare: async (label, url) => {
      const color = COMPARE_COLORS[get().compare.length % COMPARE_COLORS.length];
      set((s) => ({
        compare: [
          ...s.compare.filter((c) => c.label !== label),
          { label, color, url, data: null, error: null, loading: true },
        ],
      }));
      try {
        const data = await loadSeries(url);
        set((s) => ({
          compare: s.compare.map((c) =>
            c.label === label
              ? { ...c, data, loading: false, error: data ? null : "无 series.csv" }
              : c,
          ),
        }));
      } catch (e) {
        set((s) => ({
          compare: s.compare.map((c) =>
            c.label === label ? { ...c, loading: false, error: errText(e) } : c,
          ),
        }));
      }
    },

    removeCompare: (label) =>
      set((s) => ({ compare: s.compare.filter((c) => c.label !== label) })),

    clearCompare: () => set({ compare: [] }),

    setTraitDrift: (meta) => set({ traitDrift: extractTraitDrift(meta) }),
  };
});

/** 供组件判断是否已初始化（避免 StrictMode 双跑 bootstrap） */
export const isBootstrapped = (): boolean => bootstrapped;
