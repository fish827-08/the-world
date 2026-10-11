/**
 * A3 · 快照面板。
 * 映射契约：POST /snapshots {run_id, tick?} → 200 **只回元数据**（绝不回 npz 内容）。
 *   展示 path/tick/size_bytes/config_fingerprint/gene_count/snapshot_version/mode。
 * 🔴 语义红线：live 快照与 tick 推进**非原子**、仅供取帧，**不得当实验判据数据**——显著提示。
 * 失败（409 not_running/snapshot_unsupported/snapshot_timeout、500 snapshot_failed）走错误信封。
 */
import { useMemo, useState } from "react";
import type { JSX } from "react";
import type { SnapshotMeta } from "../app/api";
import { useConsole } from "../app/console";
import { ErrorBox, Panel, StatusLed } from "./bits";
import { clockUTC, fmtBytes, fmtNum } from "./format";

function MetaGrid({ meta }: { meta: SnapshotMeta }): JSX.Element {
  // 每项 = [键, 值, 白话悬浮说明]；说明在 .snap-cell 的 title 上
  const cells: [string, string, string?][] = [
    ["mode", String(meta.mode), "快照方式：live = 运行中实时取帧（非原子）；inproc = 引擎内同步取帧"],
    ["tick", meta.tick === null || meta.tick === undefined ? "—（未标注）" : fmtNum(meta.tick), "快照标注的 tick（取快照时留空 = 引擎当前 tick）"],
    ["size", fmtBytes(meta.size_bytes), "快照 npz 文件大小"],
    ["gene_count", meta.gene_count === null || meta.gene_count === undefined ? "—" : fmtNum(meta.gene_count), "基因位数：个体基因组的长度"],
    ["snapshot_version", meta.snapshot_version === null || meta.snapshot_version === undefined ? "—" : String(meta.snapshot_version), "快照格式版本（契约 v0）"],
    ["config_fingerprint", meta.config_fingerprint ?? "—", "配置指纹：校验快照与启动配置一致（不匹配 = 配置被改过）"],
    ["created_at", clockUTC(meta.created_at), "快照创建时间（UTC）"],
    ...(meta.request_id
      ? ([
          ["request_id", meta.request_id, "本次取快照请求的唯一 ID（排查用）"],
        ] as [string, string, string?][])
      : []),
    ...(meta.engine_tick_at_request !== null && meta.engine_tick_at_request !== undefined
      ? ([
          ["engine_tick@req", fmtNum(meta.engine_tick_at_request), "取快照请求时引擎实际推进到的 tick（live 快照非原子：可能与标注 tick 有偏差）"],
        ] as [string, string, string?][])
      : []),
  ];
  return (
    <div className="snap-grid">
      {cells.map(([k, v, hint]) => (
        <div className="snap-cell" key={k} title={hint}>
          <div className="k">{k}</div>
          <div className="v" title={v}>
            {v}
          </div>
        </div>
      ))}
      <div className="snap-cell" style={{ gridColumn: "1 / -1" }}>
        <div className="k">path</div>
        <div className="v">{meta.path}</div>
      </div>
      <div className="snap-cell" style={{ gridColumn: "1 / -1" }}>
        <div className="k">rel_path</div>
        <div className="v">{meta.rel_path}</div>
      </div>
    </div>
  );
}

export function SnapshotPanel(): JSX.Element {
  const runs = useConsole((s) => s.runs);
  const expanded = useConsole((s) => s.expanded);
  const snapBusy = useConsole((s) => s.snapBusy);
  const snapResult = useConsole((s) => s.snapResult);
  const snapError = useConsole((s) => s.snapError);
  const takeSnapshot = useConsole((s) => s.takeSnapshot);
  const clearSnapshot = useConsole((s) => s.clearSnapshot);

  // 可快照的 run：优先 running/pending（live 快照），也允许对已结束的取（后端会 409 判定）
  const snapshottable = useMemo(
    () => runs.filter((r) => r.live || r.status === "running" || r.status === "pending"),
    [runs],
  );
  const pool = snapshottable.length > 0 ? snapshottable : runs;

  const [runId, setRunId] = useState("");
  const [tick, setTick] = useState("");

  // 默认选中：展开的 run > 运行中的 run > 第一个可快照 run（渲染期校正，避免 setState-in-effect）
  if (!runId || !pool.some((r) => r.run_id === runId)) {
    const prefer =
      (expanded && pool.find((r) => r.run_id === expanded)?.run_id) ||
      pool.find((r) => r.status === "running")?.run_id ||
      pool[0]?.run_id ||
      "";
    if (prefer !== runId) setRunId(prefer);
  }

  const selectedRun = runs.find((r) => r.run_id === runId) ?? null;

  function submit(): void {
    if (!runId) return;
    const t = tick.trim();
    if (t !== "") {
      const n = Number(t);
      if (!Number.isInteger(n) || n < 0) return; // 客户端挡一下
      void takeSnapshot(runId, n);
    } else {
      void takeSnapshot(runId, null);
    }
  }

  return (
    <Panel
      idx="A3"
      title="保存快照"
      tag="POST /snapshots"
      style={{ ["--d" as string]: "120ms" }}
      className="reveal"
    >
      <div className="snap-form">
        <div className="field">
          <span className="field-label" title="选择对哪个 run 取快照（优先列出 live / 运行中的 run）">
            目标 run
          </span>
          <select
            className="select"
            value={runId}
            onChange={(e) => setRunId(e.target.value)}
            disabled={snapBusy || pool.length === 0}
            title="选择对哪个 run 取快照（优先列出 live / 运行中的 run）"
          >
            {pool.length === 0 && <option value="">（无可快照 run）</option>}
            {pool.map((r) => (
              <option key={r.run_id} value={r.run_id}>
                {r.name || r.run_id} · {r.device}
              </option>
            ))}
          </select>
        </div>
        <div className="field">
          <span className="field-label">
            tick 标注 <span className="opt">可选</span>
          </span>
          <input
            className="input tnum"
            type="number"
            min={0}
            step={1}
            placeholder="留空 = 引擎当前 tick"
            value={tick}
            onChange={(e) => setTick(e.target.value)}
            disabled={snapBusy || !runId}
            title="给快照打 tick 标签：留空 = 引擎当前 tick。仅是标注，不会让引擎回退"
          />
          <span className="field-hint">仅标注，不回退引擎</span>
        </div>
        <button
          type="button"
          className="btn btn-primary"
          onClick={submit}
          disabled={snapBusy || !runId}
          title="POST /snapshots：向运行中的实验要一帧世界状态；只回元数据，绝不回 npz 内容"
        >
          {snapBusy ? (
            <>
              <span className="spin" /> 快照中…
            </>
          ) : (
            <>◉ 取快照</>
          )}
        </button>
      </div>

      {selectedRun && (
        <div className="row" style={{ marginTop: 10 }}>
          <StatusLed status={selectedRun.status} />
          <span className="mono small dim2">
            {selectedRun.run_id} · live={String(selectedRun.live)} · status={selectedRun.status}
          </span>
        </div>
      )}

      {/* 🔴 语义红线：始终显示 */}
      <div className="redline">
        <span className="rl-ico">▲</span>
        <div className="rl-tx">
          <b>语义红线</b>：live 快照与引擎 tick 推进<b>非原子</b>，取到的帧可能跨越相邻 tick——
          <b>仅供取帧 / 可视化</b>，<b>不得</b>当作实验判据数据。判据请走已落盘的 series / 帧序列。
          本面板<b>只展示元数据，绝不下载 npz 内容</b>。
        </div>
      </div>

      <ErrorBox error={snapError} />

      {snapResult && !snapError ? (
        <>
          <div className="row" style={{ marginTop: 12 }}>
            <span className="field-label" style={{ margin: 0 }}>
              快照元数据
            </span>
            <span className="grow" />
            <button type="button" className="btn btn-ghost btn-sm" onClick={clearSnapshot}>
              清除
            </button>
          </div>
          <MetaGrid meta={snapResult} />
        </>
      ) : null}

      {selectedRun && selectedRun.snapshots && selectedRun.snapshots.length > 0 ? (
        <>
          <div className="log-hd" style={{ marginTop: 14 }}>
            <span>历史快照</span>
            <span className="dim2">（{selectedRun.snapshots.length}）</span>
          </div>
          <div className="snap-hist">
            {selectedRun.snapshots
              .slice()
              .reverse()
              .map((s, i) => (
                <div className="snap-row" key={`${s.created_at}-${i}`}>
                  <span className="tk">tick {s.tick ?? "—"}</span>
                  <span>{s.mode}</span>
                  <span className="dim2">{clockUTC(s.created_at)}</span>
                  <span className="dim2">v{s.snapshot_version ?? "—"}</span>
                  <span className="dim2">g{s.gene_count ?? "—"}</span>
                  <span className="sz">{fmtBytes(s.size_bytes)}</span>
                </div>
              ))}
          </div>
        </>
      ) : null}
    </Panel>
  );
}
