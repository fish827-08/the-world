/**
 * 应用壳：顶栏页签切换「世界 | 控制台」。
 *  · 世界：左侧世界画布（球体 WorldGlobe / 平面 WorldCanvas，可切换）+ 控制条；右侧 A5 曲线分析。
 *  · 控制台：A3 实验控制台 / A4 实时监控 / 数据包浏览（ExperimentConsole）。
 * 数据包地址默认 /datapack/（开发期放 public/datapack/），或经控制台数据包浏览器载入 /data/*。
 * ⚠ 球体渲染（WorldGlobe）与球/平切换、离线回放保持原样，仅在外层加页签条件渲染。
 */
import { useEffect, useState } from "react";
import { ControlsBar } from "./app/ControlsBar";
import { SeriesChart } from "./app/SeriesChart";
import { WorldCanvas } from "./render/WorldCanvas";
import { WorldGlobe } from "./render/WorldGlobe";
import { DEFAULT_DATAPACK_URL, useViz } from "./app/store";
import { ExperimentConsole } from "./console/ExperimentConsole";
import "./App.css";
import "./console/console.css";

type Tab = "world" | "console";

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
  /** 顶栏页签：世界 | 控制台；`?tab=console` 可深链 */
  const [tab, setTab] = useState<Tab>(() =>
    new URLSearchParams(window.location.search).get("tab") === "console" ? "console" : "world",
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
          <span className="sub">观测台 · 世界回放 + 实验控制台</span>
        </div>
        <div className="tabbar" role="tablist" aria-label="主视图">
          <button
            type="button"
            role="tab"
            aria-selected={tab === "world"}
            className={tab === "world" ? "tab on" : "tab"}
            onClick={() => setTab("world")}
          >
            世界
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={tab === "console"}
            className={tab === "console" ? "tab on" : "tab"}
            onClick={() => setTab("console")}
          >
            控制台
          </button>
        </div>
        {tab === "world" && (
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
        )}
      </header>

      {tab === "world" ? (
        <>
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
        </>
      ) : (
        <ExperimentConsole />
      )}
    </div>
  );
}

export default App;
