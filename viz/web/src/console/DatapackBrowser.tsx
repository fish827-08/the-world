/**
 * A4 · 数据包浏览器。
 * 映射契约：GET /datapacks?kind=all&include_meta&stats → 索引（坏包不 500，逐包 error 标注）。
 *   点击 viz 包 → store.load(`/data/viz/<name>/`) 走既有离线回放（喂世界视图 + SeriesChart）。
 *   live 包同样可载入（静态帧序列回放），但实时监控请走 A4 LiveMonitor（增量接流）。
 */
import { useEffect } from "react";
import type { JSX } from "react";
import type { DataPack } from "../app/api";
import { dataPackUrl } from "../app/config";
import { useConsole } from "../app/console";
import { DEFAULT_DATAPACK_URL, useViz } from "../app/store";
import { ErrorBox, Panel } from "./bits";
import { clockUTC, fmtBytes, fmtNum } from "./format";

function packUrl(p: DataPack): string {
  return dataPackUrl(p.kind, p.name);
}

function PackRow({
  pack,
  active,
  onLoad,
}: {
  pack: DataPack;
  active: boolean;
  onLoad: () => void;
}): JSX.Element {
  const world = pack.world;
  const tick = pack.tick;
  const stats = pack.stats;
  const sub = [
    world ? `${fmtNum(world.rows)}×${fmtNum(world.cols)}` : null,
    tick && tick.n_frames !== null ? `${fmtNum(tick.n_frames)} 帧` : null,
    tick && tick.start !== null && tick.end !== null ? `tick ${tick.start}→${tick.end}` : null,
    pack.contract_version,
    stats ? fmtBytes(stats.size_bytes) : null,
    stats && stats.has_series_csv ? "series✓" : null,
    pack.run_status,
    pack.modified_at ? clockUTC(pack.modified_at) : null,
  ]
    .filter(Boolean)
    .join(" · ");

  return (
    <button
      type="button"
      className={`pack-item${active ? " active" : ""}`}
      onClick={onLoad}
      disabled={!!pack.error}
      title={
        pack.error
          ? pack.path
          : `${pack.path}\nrows×cols：格网行×列（n_cells = 总格数）\nN 帧：包含的帧数\ntick a→b：覆盖 tick 段（含 stride 采样间隔）\nchannels：帧内图层（resource = 资源场 / energy = 能量场）\nentities：实体层（生物个体）\ncontract_version：数据包契约版本\nsize：包大小\nseries✓：含曲线数据 series.csv（缺 = 曲线区为空，属正常）`
      }
    >
      <span
        className={`kind-tag kind-${pack.kind}`}
        title={
          pack.kind === "viz"
            ? "viz 包：完整可视化数据包（离线回放 / 曲线分析用）"
            : "live 包：supervise 实时运行增量写出的数据包（可静态回放；实时观看请用「实时监控」）"
        }
      >
        {pack.kind}
      </span>
      <span className="pk-main">
        <span className="pk-name">{pack.name}</span>
        <span className="pk-sub">
          {pack.error ? <span className="pack-err">✕ {pack.error}</span> : sub || "（无元数据）"}
        </span>
      </span>
      {active && <span className="mono small" style={{ color: "var(--accent)" }}>●</span>}
    </button>
  );
}

export function DatapackBrowser(): JSX.Element {
  const datapacks = useConsole((s) => s.datapacks);
  const datapacksError = useConsole((s) => s.datapacksError);
  const datapacksLoading = useConsole((s) => s.datapacksLoading);
  const dataRootExists = useConsole((s) => s.dataRootExists);
  const refreshDatapacks = useConsole((s) => s.refreshDatapacks);
  const setTraitDrift = useConsole((s) => s.setTraitDrift);

  const datapackUrl = useViz((s) => s.datapackUrl);
  const liveActive = useViz((s) => s.liveActive);
  const load = useViz((s) => s.load);

  useEffect(() => {
    if (datapacks.length === 0 && !datapacksLoading) void refreshDatapacks();
  }, [datapacks.length, datapacksLoading, refreshDatapacks]);

  const viz = datapacks.filter((p) => p.kind === "viz");
  const live = datapacks.filter((p) => p.kind === "live");

  return (
    <Panel
      idx="A4"
      title="数据包"
      tag={`GET /datapacks · ${datapacks.length}`}
      actions={
        <button
          type="button"
          className="btn btn-ghost btn-sm"
          onClick={() => void refreshDatapacks()}
          disabled={datapacksLoading}
          title="重新扫描 data_root，刷新数据包索引（坏包不报 500，逐包标注错误）"
        >
          {datapacksLoading ? (
            <>
              <span className="spin" /> 刷新中
            </>
          ) : (
            "↻ 刷新"
          )}
        </button>
      }
      style={{ ["--d" as string]: "240ms" }}
      className="reveal"
    >
      <div className="field-hint" style={{ marginBottom: 8 }}>
        viz = 完整可视化包（离线回放 / 曲线分析）；live = supervise 实时运行增量写出的包
        （此处可静态回放，实时观看用「实时监控」）。
      </div>
      {dataRootExists === false && (
        <div className="empty-note">
          <code>data_root</code> 不存在（本 worktree 缺 <code>the-world-data/</code>）。
          启动 live 实验并落盘后，这里会出现数据包。
        </div>
      )}
      <ErrorBox error={datapacksError} />

      <div className="pack-list" style={{ marginTop: datapacksError ? 10 : 0 }}>
        {datapacks.length === 0 && !datapacksLoading && !datapacksError ? (
          <div className="empty-note">
            尚无数据包。默认世界视图仍会尝试载入 <code>{DEFAULT_DATAPACK_URL}</code>。
          </div>
        ) : (
          <>
            {viz.map((p) => (
              <PackRow
                key={p.id}
                pack={p}
                active={!liveActive && datapackUrl === packUrl(p)}
                onLoad={() => {
                  setTraitDrift(p.meta);
                  void load(packUrl(p));
                }}
              />
            ))}
            {live.map((p) => (
              <PackRow
                key={p.id}
                pack={p}
                active={!liveActive && datapackUrl === packUrl(p)}
                onLoad={() => {
                  setTraitDrift(p.meta);
                  void load(packUrl(p));
                }}
              />
            ))}
          </>
        )}
      </div>
    </Panel>
  );
}
