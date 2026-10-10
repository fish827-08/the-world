/**
 * 契约 v0 的 TS 镜像 —— 与 viz/CONTRACT.md 同步（🔒 改动须双侧更新）。
 */

export interface ChannelScale {
  mode: "quantile";
  p_lo: number;
  p_hi: number;
  min: number;
  max: number;
  q_lo: number;
  q_hi: number;
}

export interface ChannelMeta {
  key: string;
  label: string;
  kind: "grid";
  scale: ChannelScale;
  cmap: string;
}

export interface EntitiesMeta {
  columns: ["flat", "sub_r", "sub_c", "energy", "age", "generation", "mode"];
  dtype: "float32";
  /** 亚格细分数：`row = sub_r/subdiv`、`col = sub_c/subdiv`（格单位）；缺失 ⇒ 回退 flat 格中心 */
  subdiv?: number;
  norm: Record<string, { p_lo: number; p_hi: number }>;
}

export interface Meta {
  contract_version: string;
  world: { rows: number; cols: number; n_cells: number; projection: "equirect" };
  tick: { start: number; end: number; stride: number; n_frames: number };
  channels: ChannelMeta[];
  entities: EntitiesMeta;
  source: {
    snapshot: string;
    snapshot_tick: number;
    config_fingerprint: string;
    engine: "python" | "rust";
  };
  generator: { tool: string; version: string; created_at: string };
}

/** 帧号 → tick（k*stride 与导出器约定一致） */
export const frameToTick = (meta: Meta, k: number): number =>
  meta.tick.start + k * meta.tick.stride;
