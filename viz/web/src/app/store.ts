/**
 * 应用状态（Zustand）—— 数据包 / 播放器 / 图层。
 *
 * 播放：setInterval 按 fps 推进帧（到末帧回绕）；帧号 ↔ tick 用 contract.frameToTick。
 * 竞态：单调整数 token + AbortController —— 快速拖动/切换图层时只有最新请求落状态。
 */
import { create } from "zustand";
import { frameToTick, type Meta } from "../contract";
import {
  loadEntities,
  loadGridFrame,
  loadMeta,
  loadSeries,
  type SeriesData,
} from "./loader";

export const DEFAULT_DATAPACK_URL = "/datapack/";
export const FPS_CHOICES = [1, 2, 5, 10, 15, 30];

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

  load: (url?: string) => Promise<void>;
  seek: (k: number) => void;
  step: (delta: number) => void;
  togglePlaying: () => void;
  setFps: (fps: number) => void;
  setLayer: (key: string) => void;
}

let frameAbort: AbortController | null = null;
let timer: ReturnType<typeof setInterval> | null = null;
let token = 0;

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

  async function loadFrameData(k: number): Promise<void> {
    const { meta, layer, datapackUrl } = get();
    if (!meta) return;
    const my = ++token;
    frameAbort?.abort();
    const ac = new AbortController();
    frameAbort = ac;
    try {
      const [grid, entities] = await Promise.all([
        loadGridFrame(datapackUrl, layer, k, ac.signal),
        loadEntities(datapackUrl, k, ac.signal),
      ]);
      if (my !== token) return;
      set({ grid, entities });
    } catch (e) {
      if (my !== token || isAbort(e)) return;
      set({ error: `帧数据加载失败：${errMsg(e)}` });
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

    load: async (url) => {
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
      const { meta, frameIdx, grid } = get();
      if (!meta) return;
      const kk = clampFrame(meta, k);
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
  };
});

/** 当前帧对应 tick（无数据包 ⇒ null） */
export const useCurrentTick = (): number | null => {
  const meta = useViz((s) => s.meta);
  const frameIdx = useViz((s) => s.frameIdx);
  return meta ? frameToTick(meta, frameIdx) : null;
};
