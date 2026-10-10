/**
 * 应用壳：左侧世界画布（丹青 render/WorldCanvas）+ 控制条；右侧 ECharts 仪表盘。
 * 数据包地址默认 /datapack/（开发期放 public/datapack/）。
 */
import { useEffect, useState } from "react";
import { ControlsBar } from "./app/ControlsBar";
import { SeriesChart } from "./app/SeriesChart";
import { WorldCanvas } from "./render/WorldCanvas";
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

  const [urlInput, setUrlInput] = useState(DEFAULT_DATAPACK_URL);

  useEffect(() => {
    void load();
  }, [load]);

  return (
    <div className="viz-app">
      <header className="viz-header">
        <div className="brand">
          <span className="logo">the-world</span>
          <span className="sub">可视化 v0 · 世界回放 + 指标仪表盘</span>
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
          {meta ? (
            <div className="stage">
              <WorldCanvas
                meta={meta}
                grid={grid}
                entities={entities}
                scale={meta.channels.find((c) => c.key === layer)?.scale ?? { p_lo: 0, p_hi: 1 }}
              />
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
