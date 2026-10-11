/**
 * A5 · 曲线分析仪表盘（ECharts）。
 *
 * 能力（对齐任务书）：
 *  1) 从 series.csv **自动发现数值列**（metrics.discoverNumericColumns，不硬编码列集）；
 *  2) **列名别名映射**做展示（metrics.labelOf：population|N|pop→"种群" 等）；
 *  3) 保留：多选 chips / markLine tick 联动 / dataZoom / LTTB 抽样；
 *  4) 新增：**trait 漂移视图**（useConsole.traitDrift，来自 manifest.summary.trait_drift）；
 *  5) 新增：**多 run 对比**（useConsole.compare，叠加多个数据包/run 的同一列序列）；
 *  6) 为 **C0 能量守恒残差**与**斑块类型占比**预留指标位（metrics.isReserved：列存在即置顶渲染）。
 *
 * 数据源：主序列 = useViz.series（当前离线包 / 实时流）；对比序列 = useConsole.compare。
 */
import { useEffect, useMemo, useRef, useState } from "react";
import type { JSX } from "react";
import * as echarts from "echarts/core";
import { LineChart } from "echarts/charts";
import {
  DataZoomComponent,
  GridComponent,
  LegendComponent,
  MarkLineComponent,
  TooltipComponent,
} from "echarts/components";
import { CanvasRenderer } from "echarts/renderers";
import type { EChartsCoreOption } from "echarts/core";
import { useCurrentTick, useViz } from "./store";
import { useConsole, type TraitSeries } from "./console";
import { dataPackUrl } from "./config";
import {
  colXY,
  discoverNumericColumns,
  isReserved,
  labelOf,
  orderColumns,
  reservedGroup,
} from "./metrics";

echarts.use([
  LineChart,
  GridComponent,
  TooltipComponent,
  MarkLineComponent,
  DataZoomComponent,
  LegendComponent,
  CanvasRenderer,
]);

const MAX_CHIPS = 18;
const DEFAULT_PICKS = 2;
const PRIMARY_COLOR = "#ffd166";

function fmt(v: number | null): string {
  if (v === null || !Number.isFinite(v)) return "—";
  const a = Math.abs(v);
  if (a >= 1000) return v.toLocaleString("en-US", { maximumFractionDigits: 0 });
  if (a >= 1) return v.toFixed(2);
  return v.toPrecision(3);
}

/** tick ≤ t 的最近样本值（x 已按 tick 升序） */
function valueAt(x: number[], y: number[], t: number | null): number | null {
  if (t === null || x.length === 0) return null;
  let lo = 0;
  let hi = x.length - 1;
  let ans = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (x[mid] <= t) {
      ans = mid;
      lo = mid + 1;
    } else {
      hi = mid - 1;
    }
  }
  if (ans < 0) return null;
  const v = y[ans];
  return Number.isFinite(v) ? v : null;
}

function markLineOf(tick: number) {
  return {
    symbol: "none",
    silent: true,
    lineStyle: { color: PRIMARY_COLOR, width: 1, type: "dashed" },
    label: {
      show: true,
      formatter: String(tick),
      position: "insideEndTop",
      fontSize: 10,
      color: PRIMARY_COLOR,
    },
    data: [{ xAxis: tick }],
  };
}

/** 一条待绘曲线 */
interface Line {
  name: string;
  color: string;
  x: number[];
  y: number[];
  mark?: boolean; // 是否携带当前 tick 指示线（仅主序列）
}

function makeOption(lines: Line[], markTick: number | null): EChartsCoreOption {
  const multi = lines.length > 1;
  return {
    animation: false,
    grid: { left: 56, right: 14, top: multi ? 22 : 10, bottom: 20 },
    legend: multi
      ? {
          show: true,
          top: 0,
          right: 4,
          itemWidth: 12,
          itemHeight: 8,
          textStyle: { color: "#8b93a3", fontSize: 10 },
          data: lines.map((l) => l.name),
        }
      : { show: false },
    tooltip: { trigger: "axis", confine: true },
    xAxis: { type: "value", min: "dataMin", max: "dataMax" },
    yAxis: { type: "value", scale: true },
    dataZoom: [{ type: "inside", filterMode: "none" }],
    series: lines.map((l) => ({
      id: l.name,
      name: l.name,
      type: "line",
      showSymbol: false,
      sampling: "lttb",
      lineStyle: { width: 1.3, color: l.color },
      itemStyle: { color: l.color },
      data: l.x.map((xv, i) => [xv, l.y[i]]),
      markLine: l.mark && markTick !== null ? markLineOf(markTick) : undefined,
    })),
  };
}

/* ------------------------------------------------------------------ */
/* 通用折线卡（主序列 + 对比叠加）                                       */
/* ------------------------------------------------------------------ */

interface MetricChartProps {
  label: string;
  lines: Line[];
  markTick: number | null;
  reserved?: boolean;
  group?: string | null;
  onRemove?: () => void;
}

function LineChartCard({ label, lines, markTick, reserved, group, onRemove }: MetricChartProps): JSX.Element {
  const plotRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<ReturnType<typeof echarts.init> | null>(null);
  const markTickRef = useRef(markTick);
  const primary = lines[0];

  useEffect(() => {
    markTickRef.current = markTick;
  }, [markTick]);

  useEffect(() => {
    const el = plotRef.current;
    if (!el) return;
    const chart = echarts.init(el);
    chartRef.current = chart;
    const ro = new ResizeObserver(() => chart.resize());
    ro.observe(el);
    return () => {
      ro.disconnect();
      chart.dispose();
      chartRef.current = null;
    };
  }, []);

  // lines 变化：整图重建（带入当前指示线）
  useEffect(() => {
    chartRef.current?.setOption(makeOption(lines, markTickRef.current), { notMerge: true });
  }, [lines]);

  // 仅 tick 变化：合并更新主序列指示线
  useEffect(() => {
    if (markTick === null || !primary) return;
    chartRef.current?.setOption({
      series: [{ id: primary.name, markLine: markLineOf(markTick) }],
    });
  }, [markTick, primary]);

  const latest = primary ? valueAt(primary.x, primary.y, markTick) : null;

  return (
    <div className={`metric-card${reserved ? " metric-reserved" : ""}`}>
      <div className="metric-head">
        <span className="metric-name">{label}</span>
        {reserved ? <span className="rsv-tag">预留 · {group}</span> : null}
        <span className="metric-value">{fmt(latest)}</span>
        {onRemove ? (
          <button type="button" className="metric-close" onClick={onRemove} title="移除该指标">
            ×
          </button>
        ) : null}
      </div>
      <div className="metric-plot" ref={plotRef} />
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* trait 漂移视图                                                       */
/* ------------------------------------------------------------------ */

const TRAIT_COLORS = ["#4cc9f0", "#ffd166", "#57cc99", "#c792ea", "#f78c6c", "#82aaff"];

function TraitDriftChart({ data, markTick }: { data: TraitSeries[]; markTick: number | null }): JSX.Element {
  const lines = useMemo<Line[]>(
    () =>
      data.map((d, i) => ({
        name: d.name,
        color: TRAIT_COLORS[i % TRAIT_COLORS.length],
        x: d.points.map((p) => p[0]),
        y: d.points.map((p) => p[1]),
        mark: i === 0,
      })),
    [data],
  );
  return <LineChartCard label="trait 漂移" lines={lines} markTick={markTick} />;
}

/* ------------------------------------------------------------------ */
/* 多 run 对比管理器                                                    */
/* ------------------------------------------------------------------ */

function CompareManager(): JSX.Element {
  const datapacks = useConsole((s) => s.datapacks);
  const compare = useConsole((s) => s.compare);
  const addCompare = useConsole((s) => s.addCompare);
  const removeCompare = useConsole((s) => s.removeCompare);
  const clearCompare = useConsole((s) => s.clearCompare);

  const addable = useMemo(
    () =>
      datapacks.filter(
        (p) => !p.error && p.stats?.has_series_csv && !compare.some((c) => c.label === p.name),
      ),
    [datapacks, compare],
  );

  return (
    <div className="cmp-mgr">
      <div className="cmp-hd">
        <span>多 run 对比</span>
        {compare.length > 0 && (
          <button type="button" className="metric-close" onClick={clearCompare} title="清空对比">
            ×
          </button>
        )}
      </div>
      {compare.length > 0 && (
        <div className="cmp-list">
          {compare.map((c) => (
            <span className="cmp-entry" key={c.label} style={{ borderColor: c.color }}>
              <i className="cmp-sw" style={{ background: c.color }} />
              <span className="cmp-label">{c.label}</span>
              {c.loading ? (
                <span className="cmp-state">…</span>
              ) : c.error ? (
                <span className="cmp-state bad" title={c.error}>
                  ✕
                </span>
              ) : (
                <span className="cmp-state">{c.data ? `${c.data.rows.length}行` : "空"}</span>
              )}
              <button
                type="button"
                className="metric-close"
                onClick={() => removeCompare(c.label)}
                title="移除"
              >
                ×
              </button>
            </span>
          ))}
        </div>
      )}
      {addable.length > 0 ? (
        <div className="cmp-add">
          <span className="cmp-add-hd">叠加数据包：</span>
          {addable.slice(0, 12).map((p) => (
            <button
              key={p.id}
              type="button"
              className="chip"
              onClick={() => void addCompare(p.name, dataPackUrl(p.kind, p.name))}
              title={`${p.kind}/${p.name}`}
            >
              + {p.name}
            </button>
          ))}
        </div>
      ) : (
        compare.length === 0 && (
          <div className="cmp-empty">
            暂无可叠加的数据包（需含 series.csv）。在「控制台 › 数据包」刷新索引后再试。
          </div>
        )
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* 主组件                                                               */
/* ------------------------------------------------------------------ */

export function SeriesChart(): JSX.Element {
  const series = useViz((s) => s.series);
  const tick = useCurrentTick();
  const compare = useConsole((s) => s.compare);
  const traitDrift = useConsole((s) => s.traitDrift);
  const [selected, setSelected] = useState<string[]>([]);

  const numericColumns = useMemo(() => discoverNumericColumns(series), [series]);
  const ordered = useMemo(() => orderColumns(numericColumns), [numericColumns]);
  const reservedCols = useMemo(() => ordered.filter(isReserved), [ordered]);
  const chipColumns = useMemo(() => ordered.slice(0, MAX_CHIPS), [ordered]);

  // 列集变化 ⇒ 重置为「预留列 + 默认前几项」（渲染期校正，替代 setState-in-effect）
  const [prevChips, setPrevChips] = useState(chipColumns);
  if (prevChips !== chipColumns) {
    setPrevChips(chipColumns);
    const nonReserved = chipColumns.filter((c) => !isReserved(c));
    setSelected([...reservedCols, ...nonReserved.slice(0, DEFAULT_PICKS)]);
  }

  const shown = useMemo(() => {
    const set = new Set(selected);
    for (const r of reservedCols) set.add(r); // 预留位强制展示
    return ordered.filter((c) => set.has(c));
  }, [ordered, selected, reservedCols]);

  // 主序列 + 对比叠加：每列一组 lines
  const linesByCol = useMemo(() => {
    const m = new Map<string, Line[]>();
    if (!series) return m;
    for (const col of shown) {
      const p = colXY(series, col);
      const lines: Line[] = [
        { name: labelOf(col), color: PRIMARY_COLOR, x: p.x, y: p.y, mark: true },
      ];
      for (const c of compare) {
        if (!c.data) continue;
        const q = colXY(c.data, col);
        if (q.x.length === 0) continue;
        lines.push({ name: `${labelOf(col)} · ${c.label}`, color: c.color, x: q.x, y: q.y });
      }
      m.set(col, lines);
    }
    return m;
  }, [series, shown, compare]);

  const toggle = (name: string): void => {
    if (isReserved(name)) return; // 预留位不可移除
    setSelected((sel) => (sel.includes(name) ? sel.filter((s) => s !== name) : [...sel, name]));
  };

  const hasData = !!series && series.rows.length > 0;

  return (
    <div className="charts-panel">
      <div className="panel-head">
        <h2>曲线分析</h2>
        <span className="dim small">
          {hasData
            ? `${series!.rows.length} 行 · ${numericColumns.length} 数值列 · tick ${tick ?? "—"}`
            : "series.csv（可选）"}
        </span>
      </div>

      <CompareManager />

      {!hasData ? (
        <div className="panel-empty">
          数据包无 series.csv 或为空 —— 序列仪表盘不可用（数据包仍可播放）。
          {traitDrift && traitDrift.length > 0 ? " trait 漂移视图仍可用（见下）。" : ""}
        </div>
      ) : (
        <>
          <div className="chips">
            {chipColumns.map((c) => (
              <button
                key={c}
                type="button"
                className={`chip ${shown.includes(c) ? "on" : ""}${isReserved(c) ? " chip-rsv" : ""}`}
                onClick={() => toggle(c)}
                title={isReserved(c) ? `${c}（预留指标位，固定展示）` : c}
              >
                {labelOf(c)}
                {labelOf(c) !== c ? <span className="chip-raw"> {c}</span> : null}
              </button>
            ))}
            {numericColumns.length > MAX_CHIPS && (
              <span className="dim small">…共 {numericColumns.length} 个数值列</span>
            )}
          </div>
          <div className="charts">
            {shown.map((col) => (
              <LineChartCard
                key={col}
                label={labelOf(col)}
                lines={linesByCol.get(col) ?? []}
                markTick={tick}
                reserved={isReserved(col)}
                group={reservedGroup(col)}
                onRemove={isReserved(col) ? undefined : () => toggle(col)}
              />
            ))}
            {shown.length === 0 && (
              <div className="panel-empty">未选择指标（点上方列名启用）</div>
            )}
          </div>
        </>
      )}

      {traitDrift && traitDrift.length > 0 ? (
        <div className="charts" style={{ borderTop: "1px solid var(--line)" }}>
          <TraitDriftChart data={traitDrift} markTick={tick} />
        </div>
      ) : null}
    </div>
  );
}
