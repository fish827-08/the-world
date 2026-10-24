/**
 * A5 · 序列指标元数据（纯函数，与组件分离便于静态检查 / 复用）。
 *
 * 职责：
 *  1) 从 series.csv 表头**自动发现数值列**（不硬编码列集）；
 *  2) **列名别名映射**做展示（population|N|pop→"种群" 等）；
 *  3) 为 **C0 能量守恒残差**与**斑块类型占比**预留指标位（列存在即渲染、置顶固定）；
 *  4) 抽列 (x=tick, y=值) 供 echarts。
 */
import type { SeriesData } from "./loader";

/** 非指标列（坐标/标识，不作为曲线） */
export const NON_METRIC_COLS = new Set(["tick", "seed", "arm", "step", "frame", "index"]);

/**
 * 列名别名映射（展示用中文名）。键为**小写**列名；查表时先原样、再小写。
 * 覆盖任务书列出的别名 + 常见同义；未命中 ⇒ humanize（下划线转空格）。
 */
export const ALIAS: Record<string, string> = {
  // 种群
  population: "种群",
  pop: "种群",
  n: "种群",
  n_pop: "种群",
  num_agents: "种群",
  agent_count: "种群",
  // 能量
  total_energy: "能量",
  energy: "能量",
  sum_energy: "能量",
  energy_sum: "能量",
  mean_energy: "均能量",
  avg_energy: "均能量",
  // 资源
  stock_sum: "资源",
  sum_resource: "资源",
  resource: "资源",
  resources: "资源",
  total_resource: "资源",
  abs_food: "食物",
  abs_cap: "承载",
  global_sat: "饱和度",
  // 年龄 / 世代
  mean_age: "均龄",
  avg_age: "均龄",
  max_generation: "最大世代",
  max_gen: "最大世代",
  generation: "世代",
  // 生死率
  birth_rate: "出生率",
  births: "出生率",
  birth: "出生率",
  death_rate: "死亡率",
  deaths: "死亡率",
  death: "死亡率",
  // 多样性 / 信号
  genome_diversity: "基因组多样性",
  diversity: "基因组多样性",
  gene_diversity: "基因组多样性",
  signal_density: "信号密度",
  signal: "信号密度",
  trust: "信任",
  // 斑块
  n_patches: "斑块数",
  patch_count: "斑块数",
  // 性能
  ms_per_tick: "ms/tick",
  // C0 能量守恒残差（预留指标位）
  energy_residual_c0: "能量残差(C0)",
  energy_residual: "能量残差(C0)",
  c0_energy_residual: "能量残差(C0)",
  energy_conservation_residual: "能量残差(C0)",
};

/** 把未命中别名的列名 humanize：下划线→空格，保留原大小写 */
function humanize(col: string): string {
  return col.replace(/_/g, " ").trim();
}

/** 列 → 展示名（别名优先，未命中 humanize） */
export function labelOf(col: string): string {
  if (ALIAS[col]) return ALIAS[col];
  const lc = col.toLowerCase();
  if (ALIAS[lc]) return ALIAS[lc];
  return humanize(col);
}

/**
 * 预留指标位检测（C0 能量守恒残差 / 斑块类型占比）。
 * 命中 ⇒ 该列**固定置顶**展示（Track B 产出即渲染），并归入对应分组标签。
 */
export interface ReservedSlot {
  group: string;
  test: RegExp;
}

export const RESERVED_SLOTS: ReservedSlot[] = [
  { group: "C0 能量守恒残差", test: /(energy[_-]?residual|residual[_-]?c0|conservation)/i },
  { group: "斑块类型占比", test: /(patch[_-]?.*(frac|ratio|share)|(frac|ratio|share)[_-]?.*patch)/i },
  { group: "占比 / 分数", test: /(_frac$|_ratio$|_share$|^frac_)/i },
];

/** 命中的预留分组名（未命中 ⇒ null） */
export function reservedGroup(col: string): string | null {
  for (const s of RESERVED_SLOTS) if (s.test.test(col)) return s.group;
  return null;
}

export const isReserved = (col: string): boolean => reservedGroup(col) !== null;

/** 默认优先展示列（存在才启用）；其余数值列按表头顺序补位 */
export const METRIC_PRIORITY = [
  "population",
  "pop",
  "total_energy",
  "stock_sum",
  "sum_resource",
  "mean_age",
  "max_generation",
  "birth_rate",
  "death_rate",
  "genome_diversity",
  "signal_density",
  "global_sat",
  "abs_food",
  "abs_cap",
  "n_patches",
  "mean_energy",
  "ms_per_tick",
  "trust",
  "max_gen",
];

/** 判断某列是否为数值列（前若干行出现有限数即认） */
function isNumericColumn(s: SeriesData, idx: number): boolean {
  const n = Math.min(8, s.rows.length);
  for (let r = 0; r < n; r += 1) if (Number.isFinite(s.rows[r][idx])) return true;
  return false;
}

/** 自动发现数值列（排除 tick/seed/arm 等标识列），保持表头顺序 */
export function discoverNumericColumns(s: SeriesData | null): string[] {
  if (!s) return [];
  const out: string[] = [];
  s.columns.forEach((c, i) => {
    if (NON_METRIC_COLS.has(c.toLowerCase())) return;
    if (isNumericColumn(s, i)) out.push(c);
  });
  return out;
}

/**
 * 排序：预留指标位置顶 → 优先列 → 其余（表头序）。
 * 返回用于 chips 展示的列顺序。
 */
export function orderColumns(cols: string[]): string[] {
  const reserved = cols.filter(isReserved);
  const pri = METRIC_PRIORITY.filter((m) => cols.includes(m) && !isReserved(m));
  const rest = cols.filter((c) => !reserved.includes(c) && !pri.includes(c));
  return [...reserved, ...pri, ...rest];
}

/** 抽某列 (x=tick 或行号, y=值)；列不存在 ⇒ 空 */
export function colXY(s: SeriesData, col: string): { x: number[]; y: number[] } {
  const ti = s.columns.findIndex((c) => c.toLowerCase() === "tick");
  const ci = s.columns.indexOf(col);
  if (ci < 0) return { x: [], y: [] };
  const x = s.rows.map((r, i) => (ti >= 0 && Number.isFinite(r[ti]) ? r[ti] : i));
  const y = s.rows.map((r) => r[ci]);
  return { x, y };
}

/** 是否含某列（大小写不敏感） */
export function hasColumn(s: SeriesData | null, col: string): boolean {
  if (!s) return false;
  const lc = col.toLowerCase();
  return s.columns.some((c) => c.toLowerCase() === lc);
}
