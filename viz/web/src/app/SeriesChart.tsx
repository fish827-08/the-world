/**
 * ECharts 仪表盘 —— 读 series.csv（可选文件），每个指标一张独立 y 轴小折线图，
 * 与播放器联动：黄色虚线 = 当前 tick。
 * 按需引入（LineChart + Grid/Tooltip/MarkLine/DataZoom）。
 */
import { useEffect, useMemo, useRef, useState } from "react";
import * as echarts from "echarts/core";
import { LineChart } from "echarts/charts";
import {
  DataZoomComponent,
  GridComponent,
  MarkLineComponent,
  TooltipComponent,
} from "echarts/components";
import { CanvasRenderer } from "echarts/renderers";
import type { EChartsCoreOption } from "echarts/core";
import { seriesColumn } from "./loader";
import { useCurrentTick, useViz } from "./store";

echarts.use([
  LineChart,
  GridComponent,
  TooltipComponent,
  MarkLineComponent,
  DataZoomComponent,
  CanvasRenderer,
]);

/** 默认优先展示的列（存在才启用）；其余数值列按表头顺序补位 */
const METRIC_PRIORITY = [
  "pop",
  "global_sat",
  "abs_food",
  "abs_cap",
  "n_patches",
  "mean_energy",
  "sum_resource",
  "ms_per_tick",
  "trust",
  "max_gen",
];
const NON_METRIC_COLS = new Set(["tick", "seed", "arm"]);
const MAX_CHIPS = 14;
const DEFAULT_PICKS = 2;

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
    lineStyle: { color: "#ffd166", width: 1, type: "dashed" },
    label: {
      show: true,
      formatter: String(tick),
      position: "insideEndTop",
      fontSize: 10,
      color: "#ffd166",
    },
    data: [{ xAxis: tick }],
  };
}

function makeOption(name: string, x: number[], y: number[], markTick: number | null): EChartsCoreOption {
  return {
    animation: false,
    grid: { left: 56, right: 14, top: 10, bottom: 20 },
    tooltip: { trigger: "axis", confine: true },
    xAxis: { type: "value", min: "dataMin", max: "dataMax" },
    yAxis: { type: "value", scale: true },
    dataZoom: [{ type: "inside", filterMode: "none" }],
    series: [
      {
        id: name,
        name,
        type: "line",
        showSymbol: false,
        sampling: "lttb",
        lineStyle: { width: 1.2 },
        data: x.map((xv, i) => [xv, y[i]]),
        markLine: markTick !== null ? markLineOf(markTick) : undefined,
      },
    ],
  };
}

interface MetricChartProps {
  name: string;
  x: number[];
  y: number[];
  markTick: number | null;
  onRemove: () => void;
}

function MetricChart({ name, x, y, markTick, onRemove }: MetricChartProps) {
  const plotRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<ReturnType<typeof echarts.init> | null>(null);
  const markTickRef = useRef(markTick);

  // 经 ref 把最新 tick 带入重建 effect（重建不随 tick 触发，避免每 tick 整图重建）
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

  // 数据/名称变化：整图重建（经 markTickRef 带入当前指示线）
  useEffect(() => {
    chartRef.current?.setOption(makeOption(name, x, y, markTickRef.current), { notMerge: true });
  }, [name, x, y]);

  // 仅 tick 变化：合并更新指示线（series 按 id 匹配）
  useEffect(() => {
    if (markTick === null) return;
    chartRef.current?.setOption({ series: [{ id: name, markLine: markLineOf(markTick) }] });
  }, [name, markTick]);

  const latest = valueAt(x, y, markTick);

  return (
    <div className="metric-card">
      <div className="metric-head">
        <span className="metric-name">{name}</span>
        <span className="metric-value">{fmt(latest)}</span>
        <button type="button" className="metric-close" onClick={onRemove} title="移除该指标">
          ×
        </button>
      </div>
      <div className="metric-plot" ref={plotRef} />
    </div>
  );
}

export function SeriesChart() {
  const series = useViz((s) => s.series);
  const tick = useCurrentTick();
  const [selected, setSelected] = useState<string[]>([]);

  const numericColumns = useMemo(() => {
    if (!series) return [];
    const idx = new Map(series.columns.map((c, i) => [c, i]));
    return series.columns.filter((c) => {
      if (NON_METRIC_COLS.has(c)) return false;
      const i = idx.get(c)!;
      for (let r = 0; r < Math.min(5, series.rows.length); r += 1) {
        if (Number.isFinite(series.rows[r][i])) return true;
      }
      return false;
    });
  }, [series]);

  const chipColumns = useMemo(() => {
    const pri = METRIC_PRIORITY.filter((m) => numericColumns.includes(m));
    const rest = numericColumns.filter((c) => !pri.includes(c));
    return [...pri, ...rest].slice(0, MAX_CHIPS);
  }, [numericColumns]);

  // 列集变化（装载/切换数据包）⇒ 重置为默认前几项（渲染期校正，替代 setState-in-effect）
  const [prevChips, setPrevChips] = useState(chipColumns);
  if (prevChips !== chipColumns) {
    setPrevChips(chipColumns);
    setSelected(chipColumns.slice(0, DEFAULT_PICKS));
  }

  const shown = useMemo(
    () => numericColumns.filter((c) => selected.includes(c)),
    [numericColumns, selected],
  );

  const xValues = useMemo((): number[] => {
    if (!series) return [];
    const ti = series.columns.indexOf("tick");
    return series.rows.map((r, i) => (ti >= 0 ? r[ti] : i));
  }, [series]);

  const yMap = useMemo(() => {
    const m = new Map<string, number[]>();
    if (series) for (const c of shown) m.set(c, seriesColumn(series, c));
    return m;
  }, [series, shown]);

  const toggle = (name: string) => {
    setSelected((sel) =>
      sel.includes(name) ? sel.filter((s) => s !== name) : [...sel, name],
    );
  };

  return (
    <div className="charts-panel">
      <div className="panel-head">
        <h2>指标仪表盘</h2>
        <span className="dim small">
          {series ? `${series.rows.length} 行 · 当前 tick ${tick ?? "—"}` : "series.csv（可选）"}
        </span>
      </div>
      {!series || series.rows.length === 0 ? (
        <div className="panel-empty">
          数据包无 series.csv 或为空 —— 序列仪表盘不可用（数据包仍可播放）。
        </div>
      ) : (
        <>
          <div className="chips">
            {chipColumns.map((c) => (
              <button
                key={c}
                type="button"
                className={`chip ${selected.includes(c) ? "on" : ""}`}
                onClick={() => toggle(c)}
              >
                {c}
              </button>
            ))}
            {numericColumns.length > MAX_CHIPS && (
              <span className="dim small">…共 {numericColumns.length} 个数值列</span>
            )}
          </div>
          <div className="charts">
            {shown.map((m) => (
              <MetricChart
                key={m}
                name={m}
                x={xValues}
                y={yMap.get(m) ?? []}
                markTick={tick}
                onRemove={() => toggle(m)}
              />
            ))}
            {shown.length === 0 && <div className="panel-empty">未选择指标（点上方列名启用）</div>}
          </div>
        </>
      )}
    </div>
  );
}
