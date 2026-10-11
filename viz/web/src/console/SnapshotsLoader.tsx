/**
 * A6 · 快照加载面板（快照 → v0 数据包 → 世界视图回放）。
 * 映射契约：GET /snapshots → data_root 下可加载的引擎快照 npz（相对路径/tick/大小/修改时间）；
 *   POST /snapshots/export {snapshot, ticks?, stride?} → 服务端 subprocess 导出（可能数秒），
 *   成功后刷新数据包列表并 store.load(`/data/viz/<name>/`) 载入世界视图。
 * 高级设置：ticks（0=仅当前帧 ⇒ 静态单帧查看）/ stride（默认 20）。
 * 失败（400 invalid_snapshot / 500 snapshot_export_failed / 超时）走统一错误信封。
 */
import { useState } from "react";
import type { JSX } from "react";
import { ApiError, describeError, exportSnapshot } from "../app/api";
import type { SnapshotEntry } from "../app/api";
import { dataPackUrl } from "../app/config";
import { useConsole } from "../app/console";
import { useViz } from "../app/store";
import { ErrorBox, OkBox, Panel } from "./bits";
import { fmtBytes, fmtNum, timeAgo } from "./format";

/** rel 路径取尾段展示（全路径进 title） */
const tail = (rel: string): string => rel.split("/").pop() ?? rel;

function SnapshotRow({
  entry,
  busy,
  disabled,
  onExport,
}: {
  entry: SnapshotEntry;
  busy: boolean;
  disabled: boolean;
  onExport: () => void;
}): JSX.Element {
  return (
    <div className="snap-row" style={{ justifyContent: "flex-start" }}>
      <span className="tk" title={`${entry.snapshot}（相对 data_root 的路径）`}>
        {tail(entry.snapshot)}
      </span>
      <span className="dim2" title={`tick ${entry.tick ?? "未知"}：快照对应的引擎步数`}>
        tick {entry.tick === null ? "—" : fmtNum(entry.tick)}
      </span>
      <span className="sz" title="快照 npz 文件大小">
        {fmtBytes(entry.size_bytes)}
      </span>
      <span className="dim2">{timeAgo(entry.modified_at)}</span>
      <span className="dim2" title={entry.run_id ?? "不在 live/<run_id>/ 下（散落在 data_root 根部的快照）"}>
        {entry.run_id ? `run:${entry.run_id.slice(-10)}` : "散落"}
      </span>
      <span className="grow" />
      <button
        type="button"
        className="btn btn-ghost btn-sm"
        onClick={onExport}
        disabled={disabled}
        title={`快照 → 导出为 v0 数据包 → 自动载入世界视图（服务端续跑采样，可能需数秒至数十秒）\n${entry.snapshot}`}
      >
        {busy ? (
          <>
            <span className="spin" /> 导出中…
          </>
        ) : (
          "导出并载入"
        )}
      </button>
    </div>
  );
}

export function SnapshotsLoader({
  onGoWorld,
}: {
  /** 导出并载入成功后调用（App 下放：自动切回“世界”页签看球面回放） */
  onGoWorld?: () => void;
}): JSX.Element {
  // 列表来自 store（bootstrap 拉首轮；扫描按钮手动刷新），导出动作组件内自管
  const snapshots = useConsole((s) => s.snapshots);
  const skipped = useConsole((s) => s.snapshotsSkipped);
  const listError = useConsole((s) => s.snapshotsError);
  const loading = useConsole((s) => s.snapshotsLoading);
  const refreshSnapshots = useConsole((s) => s.refreshSnapshots);
  const refreshDatapacks = useConsole((s) => s.refreshDatapacks);

  const load = useViz((s) => s.load);

  const [exporting, setExporting] = useState<string | null>(null);
  const [exportError, setExportError] = useState<ApiError | string | null>(null);
  const [doneName, setDoneName] = useState<string | null>(null);
  const [advanced, setAdvanced] = useState(false);
  const [ticks, setTicks] = useState("0");
  const [stride, setStride] = useState("20");

  const entries = snapshots ?? [];

  async function exportAndLoad(entry: SnapshotEntry): Promise<void> {
    setExporting(entry.snapshot);
    setExportError(null);
    setDoneName(null);
    try {
      const t = Math.max(0, Math.floor(Number(ticks) || 0));
      const st = Math.max(1, Math.floor(Number(stride) || 20));
      const res = await exportSnapshot({ snapshot: entry.snapshot, ticks: t, stride: st });
      await refreshDatapacks();
      setDoneName(res.name);
      await load(dataPackUrl("viz", res.name));
      onGoWorld?.(); // 载入成功 ⇒ 自动切到世界页签（深链/手动切换不受影响）
    } catch (e) {
      setExportError(e instanceof ApiError ? e : describeError(e));
    } finally {
      setExporting(null);
    }
  }

  return (
    <Panel
      idx="A6"
      title="快照加载"
      tag={`GET /snapshots · ${entries.length}`}
      actions={
        <button
          type="button"
          className="btn btn-ghost btn-sm"
          onClick={() => void refreshSnapshots()}
          disabled={loading || exporting !== null}
          title="重新扫描 data_root 下的引擎快照 npz 列表"
        >
          {loading ? (
            <>
              <span className="spin" /> 扫描中
            </>
          ) : (
            "↻ 扫描"
          )}
        </button>
      }
      style={{ ["--d" as string]: "180ms" }}
      className="reveal"
    >
      <div className="row wrap" style={{ gap: 8 }}>
        <span className="dim2 small">
          把 <code>data_root</code> 下的引擎快照 npz 导出为 v0 数据包并直接载入世界视图。
        </span>
      </div>
      <div className="row wrap" style={{ gap: 8, marginTop: 8 }}>
        <button
          type="button"
          className="btn btn-ghost btn-sm"
          onClick={() => setAdvanced((v) => !v)}
        >
          {advanced ? "▾ 高级设置" : "▸ 高级设置"}
        </button>
        <span className="dim2 small">
          导出在服务端续跑引擎并采样，可能需要数秒至数十秒。
        </span>
      </div>

      {advanced && (
        <div className="row wrap" style={{ marginTop: 8, gap: 14 }}>
          <div className="field">
            <span className="field-label">ticks 续跑</span>
            <input
              className="input tnum"
              type="number"
              min={0}
              step={1}
              value={ticks}
              onChange={(e) => setTicks(e.target.value)}
              disabled={exporting !== null}
              title="导出时让引擎从快照继续跑多少 tick；0 = 不续跑，仅当前帧静态查看"
            />
            <span className="field-hint">0=仅当前帧（静态单帧查看）</span>
          </div>
          <div className="field">
            <span className="field-label">stride 采样</span>
            <input
              className="input tnum"
              type="number"
              min={1}
              step={1}
              value={stride}
              onChange={(e) => setStride(e.target.value)}
              disabled={exporting !== null}
              title="续跑时每 N tick 采一帧（stride = 20 即每 20 tick 留一帧）"
            />
            <span className="field-hint">每 N tick 采一帧（帧数 = ticks/stride + 1）</span>
          </div>
        </div>
      )}

      <ErrorBox error={listError} />
      <ErrorBox error={exportError} />

      {!exportError && doneName && (
        <OkBox>
          已导出 <b>{doneName}</b> 并载入世界视图。
        </OkBox>
      )}

      {entries.length === 0 && !loading && !listError ? (
        <div className="empty-note">
          <code>data_root</code> 下暂无引擎快照 npz（运行 live 实验并取快照后，这里会出现可加载项）。
        </div>
      ) : (
        <div className="snap-hist" style={{ marginTop: 10 }}>
          {entries.map((s) => (
            <SnapshotRow
              key={s.snapshot}
              entry={s}
              busy={exporting === s.snapshot}
              disabled={exporting !== null}
              onExport={() => void exportAndLoad(s)}
            />
          ))}
        </div>
      )}

      {skipped > 0 && (
        <div className="dim2 small" style={{ marginTop: 8 }}>
          已跳过 {fmtNum(skipped)} 个非快照 npz（缺 <code>snapshot_version</code> 键的实验数据 / 损坏文件）。
        </div>
      )}
    </Panel>
  );
}
