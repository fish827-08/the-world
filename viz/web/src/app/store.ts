/**
 * 应用状态（Zustand）—— 数据包 / 播放器 / 图层 / 实时监控（A4）。
 *
 * 离线回放：`load(url)` 拉契约 v0 数据包；setInterval 按 fps 推进帧（到末帧回绕）。
 * 实时监控：`loadLive(runId)` 轮询 `<data_root>/live/<run_id>/meta.json`（no-store），
 *   发现 `tick.n_frames` 增长即**增量拉取新帧**并跟随最新；数据包尚未出现 ⇒ `waiting`，
 *   帧文件尚未写出 ⇒ 静默跳过、下轮再试（优雅处理帧数增长）。
 * 竞态：单调整数 token（帧）+ liveToken（轮询）+ AbortController —— 只有最新请求落状态。
 */
import { create } from "zustand";
import { frameToTick, type Meta } from "../contract";
import { livePackUrl } from "./config";
import {
  loadEntities,
  loadGridFrame,
  loadMeta,
  loadSeries,
  type SeriesData,
} from "./loader";

export const DEFAULT_DATAPACK_URL = "/datapack/";
export const FPS_CHOICES = [1, 2, 5, 10, 15, 30];
/** 实时监控轮询间隔（ms）——meta + 增量帧 + series */
export const LIVE_POLL_MS = 1500;

export type LiveStatus = "idle" | "waiting" | "streaming" | "error";

export interface VizState {
  /** 数据包 URL 前缀（成功加载后提交；失败不改） */
  datapackUrl: string;
  meta: Meta | null;
  series: SeriesData | null;
  frameIdx: number;
  /** 当前帧图层栅格（uint8, rows*cols） */
  grid: Uint8Array | null;
  /** 当前帧实体交错缓冲（7×f32） */
  entities: Float32Array | null;
  playing: boolean;
  fps: number;
  /** 通道 key（meta.channels 之一） */
  layer: string;
  loading: boolean;
  error: string | null;

  /* ---- A4 实时监控 ---- */
  liveRunId: string | null;
  liveActive: boolean;
  /** 跟随最新帧（用户拖动滑条 ⇒ false；"回到最新" ⇒ true） */
  liveFollow: boolean;
  /** 已发现的帧数（meta.tick.n_frames） */
  liveFrames: number;
  liveStatus: LiveStatus;
  liveError: string | null;

  load: (url?: string) => Promise<void>;
  seek: (k: number) => void;
  step: (delta: number) => void;
  togglePlaying: () => void;
  setFps: (fps: number) => void;
  setLayer: (key: string) => void;

  loadLive: (runId: string) => Promise<void>;
  stopLive: () => void;
  setLiveFollow: (on: boolean) => void;
  followLatest: () => void;
}

let frameAbort: AbortController | null = null;
let timer: ReturnType<typeof setInterval> | null = null;
let liveTimer: ReturnType<typeof setInterval> | null = null;
let token = 0;
let liveToken = 0;

const clampFrame = (meta: Meta, k: number): number =>
  Math.max(0, Math.min(meta.tick.n_frames - 1, k));

const errMsg = (e: unknown): string => (e instanceof Error ? e.message : String(e));

const isAbort = (e: unknown): boolean =>
  e instanceof DOMException && e.name === "AbortError";

export const useViz = create<VizState>()((set, get) => {
  function stopTimer(): void {
    if (timer !== null) {
      clearInterval(timer);
      timer = null;
    }
  }

  function startTimer(): void {
    stopTimer();
    timer = setInterval(() => {
      const { meta, frameIdx, playing } = get();
      if (!meta || !playing) return;
      get().seek((frameIdx + 1) % meta.tick.n_frames);
    }, Math.max(16, Math.round(1000 / get().fps)));
  }

  function stopLiveTimer(): void {
    if (liveTimer !== null) {
      clearInterval(liveTimer);
      liveTimer = null;
    }
    liveToken += 1; // 作废在途轮询
  }

  /** 当前帧数据基址：实时模式走 live 包，否则走离线 datapackUrl */
  function activeBase(): string {
    const { liveActive, liveRunId, datapackUrl } = get();
    return liveActive && liveRunId ? livePackUrl(liveRunId) : datapackUrl;
  }

  /** 实时模式禁缓存（拿增长后的帧/meta） */
  function activeCache(): RequestCache | undefined {
    return get().liveActive ? "no-store" : undefined;
  }

  async function loadFrameData(k: number): Promise<void> {
    const { meta, layer } = get();
    if (!meta) return;
    const base = activeBase();
    const cache = activeCache();
    const my = ++token;
    frameAbort?.abort();
    const ac = new AbortController();
    frameAbort = ac;
    try {
      const [grid, entities] = await Promise.all([
        loadGridFrame(base, layer, k, ac.signal, cache),
        loadEntities(base, k, ac.signal, cache),
      ]);
      if (my !== token) return;
      set({ grid, entities });
    } catch (e) {
      if (my !== token || isAbort(e)) return;
      set({ error: `帧数据加载失败：${errMsg(e)}` });
    }
  }

  /** 一轮实时轮询：meta（no-store）→ 帧数增长则增量拉帧 → series 增量 */
  async function pollLive(): Promise<void> {
    const st = get();
    if (!st.liveActive || !st.liveRunId) return;
    const runId = st.liveRunId;
    const base = livePackUrl(runId);
    const my = ++liveToken;

    let meta: Meta;
    try {
      meta = await loadMeta(base, undefined, "no-store");
    } catch (e) {
      if (my !== liveToken || isAbort(e)) return;
      // 数据包尚未出现（实验刚起、导出器还没写 meta.json）⇒ waiting，保持轮询
      set({ liveStatus: "waiting", loading: false, liveError: null });
      return;
    }
    if (my !== liveToken) return;
    const g2 = get();
    if (!g2.liveActive || g2.liveRunId !== runId) return;

    const n = meta.tick.n_frames;
    const cur = g2.meta;
    const metaChanged =
      !cur ||
      cur.tick.n_frames !== n ||
      cur.tick.start !== meta.tick.start ||
      cur.tick.end !== meta.tick.end ||
      cur.tick.stride !== meta.tick.stride ||
      cur.channels.length !== meta.channels.length;

    if (metaChanged) {
      const wantLayer = g2.layer;
      const layer = meta.channels.some((c) => c.key === wantLayer)
        ? wantLayer
        : (meta.channels[0]?.key ?? "");
      set({
        meta,
        layer,
        liveFrames: n,
        liveStatus: n > 0 ? "streaming" : "waiting",
        loading: false,
        error: null,
        liveError: null,
      });
    } else {
      set({ liveFrames: n, liveStatus: n > 0 ? "streaming" : "waiting" });
    }
    if (n <= 0) return;

    const g3 = get();
    const targetK = clampFrame(meta, g3.liveFollow ? n - 1 : g3.frameIdx);
    const needFrame = g3.grid === null || targetK !== g3.frameIdx || metaChanged;
    if (needFrame) {
      frameAbort?.abort();
      const ac = new AbortController();
      frameAbort = ac;
      try {
        const [grid, entities] = await Promise.all([
          loadGridFrame(base, get().layer, targetK, ac.signal, "no-store"),
          loadEntities(base, targetK, ac.signal, "no-store"),
        ]);
        if (my !== liveToken || get().liveRunId !== runId) return;
        set({ frameIdx: targetK, grid, entities, loading: false });
      } catch (e) {
        // 帧文件尚未写出 / 瞬态失败 ⇒ 静默等下一轮（不打断监控）
        if (my !== liveToken || isAbort(e)) return;
      }
    }

    // series.csv 增量（可选，缺失 ⇒ null，不报错）
    try {
      const series = await loadSeries(base, undefined, "no-store");
      if (my === liveToken && get().liveRunId === runId) set({ series });
    } catch {
      /* 可选文件，忽略 */
    }
  }

  return {
    datapackUrl: DEFAULT_DATAPACK_URL,
    meta: null,
    series: null,
    frameIdx: 0,
    grid: null,
    entities: null,
    playing: false,
    fps: 10,
    layer: "",
    loading: false,
    error: null,

    liveRunId: null,
    liveActive: false,
    liveFollow: true,
    liveFrames: 0,
    liveStatus: "idle",
    liveError: null,

    load: async (url) => {
      // 切离线包 ⇒ 停实时监控
      stopLiveTimer();
      set({ liveActive: false, liveRunId: null, liveStatus: "idle", liveFollow: true });

      const cur = get();
      const raw = (url ?? cur.datapackUrl).trim();
      const normalized = raw.endsWith("/") ? raw : `${raw}/`;
      if (cur.meta && normalized === cur.datapackUrl && !cur.error) return;

      const my = ++token;
      frameAbort?.abort();
      stopTimer();
      set({ loading: true, error: null, playing: false });
      try {
        const meta = await loadMeta(normalized);
        if (my !== token) return;
        const layer = meta.channels[0]?.key ?? "";
        const [grid, entities, series] = await Promise.all([
          loadGridFrame(normalized, layer, 0),
          loadEntities(normalized, 0),
          loadSeries(normalized).catch((e: unknown): null => {
            if (!isAbort(e)) console.warn("series.csv 解析失败（可选文件）：", e);
            return null;
          }),
        ]);
        if (my !== token) return;
        set({
          datapackUrl: normalized,
          meta,
          layer,
          frameIdx: 0,
          grid,
          entities,
          series,
          loading: false,
        });
      } catch (e) {
        if (my !== token || isAbort(e)) return;
        set({ loading: false, error: errMsg(e) });
      }
    },

    seek: (k) => {
      const { meta, frameIdx, grid, liveActive } = get();
      if (!meta) return;
      const kk = clampFrame(meta, k);
      // 实时模式下用户手动拖动 ⇒ 取消"跟随最新"
      if (liveActive && kk !== meta.tick.n_frames - 1) set({ liveFollow: false });
      if (kk === frameIdx && grid !== null) return;
      set({ frameIdx: kk });
      void loadFrameData(kk);
    },

    step: (delta) => {
      const { meta, frameIdx } = get();
      if (!meta) return;
      get().seek(clampFrame(meta, frameIdx + delta));
    },

    togglePlaying: () => {
      const { meta, playing } = get();
      if (!meta) return;
      if (playing) {
        stopTimer();
        set({ playing: false });
      } else {
        set({ playing: true });
        startTimer();
      }
    },

    setFps: (fps) => {
      const f = Math.max(1, Math.min(60, Math.round(fps)));
      set({ fps: f });
      if (get().playing) startTimer();
    },

    setLayer: (key) => {
      const { meta, layer, frameIdx } = get();
      if (!meta || key === layer) return;
      set({ layer: key });
      void loadFrameData(frameIdx);
    },

    loadLive: async (runId) => {
      stopTimer();
      frameAbort?.abort();
      stopLiveTimer();
      set({
        liveActive: true,
        liveRunId: runId,
        liveFollow: true,
        liveStatus: "waiting",
        liveFrames: 0,
        liveError: null,
        loading: true,
        error: null,
        playing: false,
      });
      await pollLive();
      // 期间可能已切走（另一次 loadLive / load）⇒ 只有仍是本次 run 才起定时器
      if (get().liveRunId !== runId || !get().liveActive) return;
      if (liveTimer === null) {
        liveTimer = setInterval(() => void pollLive(), LIVE_POLL_MS);
      }
    },

    stopLive: () => {
      stopLiveTimer();
      set({
        liveActive: false,
        liveRunId: null,
        liveStatus: "idle",
        liveFollow: true,
        liveError: null,
      });
    },

    setLiveFollow: (on) => set({ liveFollow: on }),

    followLatest: () => {
      const { meta, liveActive } = get();
      if (!meta || !liveActive) return;
      const k = clampFrame(meta, meta.tick.n_frames - 1);
      set({ liveFollow: true, frameIdx: k });
      void loadFrameData(k);
    },
  };
});

/** 当前帧对应 tick（无数据包 ⇒ null） */
export const useCurrentTick = (): number | null => {
  const meta = useViz((s) => s.meta);
  const frameIdx = useViz((s) => s.frameIdx);
  return meta ? frameToTick(meta, frameIdx) : null;
};
