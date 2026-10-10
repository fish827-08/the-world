/**
 * 应用壳：左侧世界画布（球体 WorldGlobe / 平面 WorldCanvas，可切换）+ 控制条；右侧 ECharts 仪表盘。
 * 数据包地址默认 /datapack/（开发期放 public/datapack/）。
 */
import { useEffect, useState } from "react";
import { ControlsBar } from "./app/ControlsBar";
import { SeriesChart } from "./app/SeriesChart";
import { WorldCanvas } from "./render/WorldCanvas";
import { WorldGlobe } from "./render/WorldGlobe";
import { DEFAULT_DATAPACK_URL, useViz } from "./app/store";
import "./App.css";

function MetaPills() {
  const meta = useViz((s) => s.meta);
  if (!meta) return null;
  return (
    <div className="pills">
      <span className="pill">契约 {meta.contract_version}</span>
      <span className="pill">
        世界 {meta.world.rows}×{meta.world.cols}
      </span>
      <span className="pill">
        tick {meta.tick.start}→{meta.tick.end} · {meta.tick.n_frames} 帧
      </span>
      <span className="pill">
        {meta.channels.map((c) => c.label).join(" / ")}
      </span>
      <span className="pill">引擎 {meta.source.engine}</span>
      <span className="pill" title={meta.generator.created_at}>
        生成 {meta.generator.tool}
      </span>
    </div>
  );
}

function App() {
  const load = useViz((s) => s.load);
  const meta = useViz((s) => s.meta);
  const loading = useViz((s) => s.loading);
  const error = useViz((s) => s.error);
  const grid = useViz((s) => s.grid);
  const entities = useViz((s) => s.entities);
  const datapackUrl = useViz((s) => s.datapackUrl);
  const layer = useViz((s) => s.layer);

  const [urlInput, setUrlInput] = useState(
    () => new URLSearchParams(window.location.search).get("datapack") ?? DEFAULT_DATAPACK_URL,
  );
  /** 世界读法：球体（v1，默认）/ 平面（v0 地图读法，保留作对照）；`?view=flat` 可深链 */
  const [view, setView] = useState<"globe" | "flat">(() =>
    new URLSearchParams(window.location.search).get("view") === "flat" ? "flat" : "globe",
  );

  useEffect(() => {
    // ?datapack=/path/ 可深链切换数据包（缺省 /datapack/）
    void load(new URLSearchParams(window.location.search).get("datapack") ?? undefined);
  }, [load]);

  return (
    <div className="viz-app">
      <header className="viz-header">
        <div className="brand">
          <span className="logo">the-world</span>
          <span className="sub">可视化 v1 · 三维球体 + 世界回放 + 指标仪表盘</span>
        </div>
        <form
          className="url-form"
          onSubmit={(e) => {
            e.preventDefault();
            void load(urlInput);
          }}
        >
          <input
            value={urlInput}
            onChange={(e) => setUrlInput(e.target.value)}
            spellCheck={false}
            placeholder={datapackUrl}
            title="数据包 URL 前缀（默认 /datapack/）"
          />
          <button type="submit" disabled={loading}>
            {loading ? "加载中…" : "加载"}
          </button>
        </form>
      </header>
      <MetaPills />
      {error && (
        <div className="error-banner">
          {error}
          <span className="dim small">
            　检查路径：开发期数据包放 viz/web/public/datapack/，或输入其他 URL 前缀
          </span>
        </div>
      )}
      <main className="viz-main">
        <section className="viz-world">
          <div className="stage-wrap">
            {meta ? (
              <div className={view === "globe" ? "stage stage-globe" : "stage"}>
                {view === "globe" ? (
                  <WorldGlobe
                    meta={meta}
                    grid={grid}
                    entities={entities}
                    scale={meta.channels.find((c) => c.key === layer)?.scale ?? { p_lo: 0, p_hi: 1 }}
                  />
                ) : (
                  <WorldCanvas
                    meta={meta}
                    grid={grid}
                    entities={entities}
                    scale={meta.channels.find((c) => c.key === layer)?.scale ?? { p_lo: 0, p_hi: 1 }}
                  />
                )}
              </div>
            ) : (
              <div className="stage">
                <div className="empty">
                  {loading ? (
                    "加载数据包中…"
                  ) : (
                    <>
                      无数据包。把数据包放入 <code>viz/web/public/datapack/</code>
                      <br />
                      （默认地址 /datapack/），或在上方输入其他 URL 前缀后点"加载"。
                    </>
                  )}
                </div>
              </div>
            )}
            {meta && (
              <div className="view-toggle" role="group" aria-label="视图切换">
                <button
                  type="button"
                  className={view === "globe" ? "on" : undefined}
                  onClick={() => setView("globe")}
                  title="三维球体读法（看整体对形）"
                >
                  球体
                </button>
                <button
                  type="button"
                  className={view === "flat" ? "on" : undefined}
                  onClick={() => setView("flat")}
                  title="平面地图读法（查具体格值）"
                >
                  平面
                </button>
              </div>
            )}
          </div>
          <ControlsBar />
        </section>
        <aside className="viz-side">
          <SeriesChart />
        </aside>
      </main>
    </div>
  );
}

export default App;
