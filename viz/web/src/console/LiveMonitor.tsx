/**
 * A4 · 实时监控。
 * 映射契约：给定 run_id，store.loadLive 轮询 `<data_root>/live/<run_id>/meta.json`（no-store），
 *   发现 tick.n_frames 增长即**增量拉取新帧**喂给世界视图（WorldGlobe/Canvas 读同一 useViz）。
 *   数据包尚未出现 ⇒ waiting（保持轮询）；帧文件尚未写出 ⇒ 静默跳过、下轮再试。
 * 本面板只负责「选 run / 起停监控 / 跟随最新 / 读数」，画面渲染在「世界」页签。
 */
import { useMemo, useState } from "react";
import type { JSX } from "react";
import { LIVE_POLL_MS, useCurrentTick, useViz } from "../app/store";
import { useConsole } from "../app/console";
import { Panel, StatusLed } from "./bits";
import { fmtNum } from "./format";

const LIVE_STATUS_LABEL: Record<string, string> = {
  idle: "未监控",
  waiting: "等待数据包…",
  streaming: "实时流",
  error: "错误",
};

/** 状态读数悬浮白话（live-stats「状态」格 title 用） */
const LIVE_STATUS_HINT: Record<string, string> = {
  idle: "未开始监控：在上方选 run 并点「开始监控」",
  waiting: "等待首帧：正在轮询 live/<run_id>/meta.json，数据包一出现即自动接流",
  streaming: "实时流：已连上数据包，新帧到达即喂给世界视图",
  error: "出错：见下方错误信息",
};

export function LiveMonitor(): JSX.Element {
  const runs = useConsole((s) => s.runs);
  const details = useConsole((s) => s.details);

  const liveActive = useViz((s) => s.liveActive);
  const liveRunId = useViz((s) => s.liveRunId);
  const liveStatus = useViz((s) => s.liveStatus);
  const liveFrames = useViz((s) => s.liveFrames);
  const liveFollow = useViz((s) => s.liveFollow);
  const liveError = useViz((s) => s.liveError);
  const meta = useViz((s) => s.meta);
  const frameIdx = useViz((s) => s.frameIdx);
  const layer = useViz((s) => s.layer);
  const loadLive = useViz((s) => s.loadLive);
  const stopLive = useViz((s) => s.stopLive);
  const followLatest = useViz((s) => s.followLatest);
  const setLiveFollow = useViz((s) => s.setLiveFollow);
  const tick = useCurrentTick();

  // 候选：live 的 run（能产 live 数据包）优先，回退运行中的 run
  const candidates = useMemo(() => {
    const live = runs.filter((r) => r.live);
    const active = runs.filter((r) => r.status === "running" || r.status === "pending");
    const seen = new Set<string>();
    return [...live, ...active].filter((r) => {
      if (seen.has(r.run_id)) return false;
      seen.add(r.run_id);
      return true;
    });
  }, [runs]);

  const [picked, setPicked] = useState("");
  const [prevLive, setPrevLive] = useState<string | null>(liveRunId);

  // liveRunId 变化时同步选中；否则保证 picked 在候选内（渲染期校正，避免 setState-in-effect）
  if (prevLive !== liveRunId) {
    setPrevLive(liveRunId);
    if (liveRunId) setPicked(liveRunId);
  }
  const defaultPick =
    candidates.find((r) => r.status === "running")?.run_id ?? candidates[0]?.run_id ?? "";
  if (!liveRunId && (!picked || !candidates.some((r) => r.run_id === picked))) {
    if (defaultPick !== picked) setPicked(defaultPick);
  }

  const pickedRun =
    runs.find((r) => r.run_id === picked) ?? details[picked] ?? null;

  function start(): void {
    if (!picked) return;
    void loadLive(picked);
  }

  const monitoring = liveActive && liveRunId === picked;
  const nFrames = meta?.tick.n_frames ?? liveFrames;
  const pct =
    nFrames > 0 && liveFrames > 0
      ? Math.max(2, Math.min(100, Math.round((frameIdx + 1) / nFrames * 100)))
      : 0;

  return (
    <Panel
      idx="A4"
      title="实时监控"
      cyan
      scan={liveStatus === "streaming"}
      tag={`${Math.round(LIVE_POLL_MS / 100) / 10}s 轮询`}
      style={{ ["--d" as string]: "180ms" }}
      className="reveal"
    >
      <div className="snap-form">
        <div className="field">
          <span className="field-label">监控 run</span>
          <select
            className="select"
            value={picked}
            onChange={(e) => setPicked(e.target.value)}
            disabled={candidates.length === 0}
          >
            {candidates.length === 0 && <option value="">（无 live run）</option>}
            {candidates.map((r) => (
              <option key={r.run_id} value={r.run_id}>
                {r.name || r.run_id} · {r.device} · {r.status}
              </option>
            ))}
          </select>
          <span className="field-hint">
            {pickedRun
              ? `live=${String(pickedRun.live)} · 轮询 /data/live/${picked}/meta.json`
              : "先启动一个 live 实验"}
          </span>
        </div>
        <button
          type="button"
          className={monitoring ? "btn btn-danger" : "btn btn-cyan"}
          onClick={() => (monitoring ? stopLive() : start())}
          disabled={!picked}
        >
          {monitoring ? "■ 停止监控" : "▶ 开始监控"}
        </button>
      </div>

      <div className="live-stats" style={{ marginTop: 12 }}>
        <div className="lstat">
          <div className="k">状态</div>
          <div className="v" style={{ fontSize: 13 }} title={LIVE_STATUS_HINT[liveStatus]}>
            {LIVE_STATUS_LABEL[liveStatus] ?? liveStatus}
          </div>
        </div>
        <div className="lstat">
          <div className="k">帧数</div>
          <div
            className="v"
            title="已拉取的实时帧数：随实验运行不断增长（每帧 = 引擎按 stride 存出的一张世界快照）"
          >
            {fmtNum(liveFrames)}
          </div>
        </div>
        <div className="lstat">
          <div className="k">当前 tick</div>
          <div className="v amber">{tick === null ? "—" : fmtNum(tick)}</div>
        </div>
        <div className="lstat">
          <div className="k">图层</div>
          <div className="v" style={{ fontSize: 13 }}>
            {layer || "—"}
          </div>
        </div>
      </div>

      <div className="frame-bar">
        <i style={{ width: `${pct}%` }} />
      </div>

      <div className="row wrap" style={{ marginTop: 12 }}>
        <label className="switch">
          <input
            type="checkbox"
            checked={liveFollow}
            onChange={(e) => {
              setLiveFollow(e.target.checked);
              if (e.target.checked) followLatest();
            }}
            disabled={!liveActive}
          />
          <span className="track" />
          <span className="sw-label">跟随最新帧</span>
        </label>
        <button
          type="button"
          className="btn btn-ghost btn-sm"
          onClick={followLatest}
          disabled={!liveActive}
        >
          ⤒ 回到最新
        </button>
        <span className="grow" />
        <StatusLed status={liveStatus === "streaming" ? "running" : liveStatus === "error" ? "failed" : "pending"} />
        <span className="mono small dim2">
          {liveActive
            ? `帧 ${frameIdx + 1}/${nFrames || "?"} · ${liveFollow ? "跟随" : "已暂停跟随"}`
            : "未在监控"}
        </span>
      </div>

      {liveStatus === "waiting" && (
        <div className="empty-note" style={{ marginTop: 8 }}>
          实验刚启动 / 导出器尚未写出 <code>meta.json</code>，或该分支缺
          <code>the-world-data/live/</code>。保持轮询中——数据包一出现即自动接流。
        </div>
      )}
      {liveError && (
        <div className="err-box" style={{ marginTop: 8 }}>
          <div className="eb-hd">监控错误</div>
          <div className="eb-msg">{liveError}</div>
        </div>
      )}
      {liveStatus === "streaming" && (
        <div className="empty-note" style={{ marginTop: 8 }}>
          切到「世界」页签查看实时球体；帧数增长时视图自动跟随最新 tick。
        </div>
      )}
    </Panel>
  );
}
