/**
 * A3 · 运行台账。
 * 映射契约：GET /experiments（by_status 计数 + runs）；对 running/pending/展开 run 每 1s
 *   由 store.setPolling 轮询 GET /experiments/{id}?tail_lines（status/ended_reason/returncode/log_tail）；
 *   Cancel → POST /experiments/{id}/cancel（已结束 ⇒ 后端 409 not_running，走 actionError）。
 * 卡片点击展开显示 log_tail（logLineClass 着色）。
 */
import type { JSX } from "react";
import type { Run, RunStatus } from "../app/api";
import { useConsole } from "../app/console";
import { ErrorBox, Panel, StatusBadge, StatusLed } from "./bits";
import {
  ENDED_REASON_LABEL,
  clockUTC,
  fmtBytes,
  fmtNum,
  logLineClass,
  timeAgo,
} from "./format";

const BY_STATUS_ORDER: RunStatus[] = ["running", "pending", "finished", "failed"];

/** 台账里优先用详情（带 log_tail），回退列表项 */
function pickRun(runs: Run[], details: Record<string, Run>, runId: string): Run {
  const d = details[runId];
  return d ?? runs.find((r) => r.run_id === runId) ?? ({} as Run);
}

function RunSpec({ run }: { run: Run }): JSX.Element {
  return (
    <div className="run-spec">
      <span className="dev">{run.device}</span>
      <span>seed {run.seed}</span>
      <span>ticks {fmtNum(run.ticks)}</span>
      {run.arm !== null && run.arm !== undefined ? <span>arm {run.arm}</span> : null}
      {run.live ? <span>live</span> : null}
    </div>
  );
}

function RunBody({
  run,
  busy,
  onCancel,
}: {
  run: Run;
  busy: boolean;
  onCancel: () => void;
}): JSX.Element {
  const active = run.status === "running" || run.status === "pending";
  const log = run.log_tail ?? null;
  return (
    <div className="run-body">
      <dl className="kv">
        <dt>状态</dt>
        <dd>
          <StatusBadge status={run.status} />
          {run.ended_reason ? (
            <span className="dim2" style={{ marginLeft: 8 }}>
              {ENDED_REASON_LABEL[run.ended_reason] ?? run.ended_reason}
            </span>
          ) : null}
        </dd>
        <dt>返回码</dt>
        <dd className={run.returncode === 0 ? "ok" : run.returncode !== null ? "bad" : ""}>
          {run.returncode === null || run.returncode === undefined ? "—" : run.returncode}
        </dd>
        <dt>PID</dt>
        <dd>{run.pid ?? "—"}</dd>
        <dt>创建</dt>
        <dd>
          {clockUTC(run.created_at)} <span className="dim2">（{timeAgo(run.created_at)}）</span>
        </dd>
        <dt>开始 / 结束</dt>
        <dd>
          {clockUTC(run.started_at)} → {clockUTC(run.finished_at)}
        </dd>
        <dt>脚本</dt>
        <dd>{run.script || "—"}</dd>
        <dt>输出目录</dt>
        <dd title={run.out_dir}>{run.out_dir || "—"}</dd>
        <dt>快照</dt>
        <dd>
          {run.snapshots && run.snapshots.length > 0
            ? `${run.snapshots.length} 个（最近 ${fmtBytes(run.snapshots[run.snapshots.length - 1].size_bytes)}）`
            : "无"}
        </dd>
        {run.note ? (
          <>
            <dt>备注</dt>
            <dd>{run.note}</dd>
          </>
        ) : null}
      </dl>

      <div className="run-actions">
        <button
          type="button"
          className="btn btn-danger btn-sm"
          disabled={!active || busy}
          onClick={onCancel}
          title={active ? "POST /experiments/{id}/cancel" : "仅运行中/排队可取消"}
        >
          {busy ? (
            <>
              <span className="spin" /> 取消中…
            </>
          ) : (
            <>✕ 取消运行</>
          )}
        </button>
        {!active && <span className="field-hint mono">已结束（409 not_running）</span>}
      </div>

      <div>
        <div className="log-hd">
          <span>log_tail</span>
          <span className="dim2">（最近 {log ? log.length : 0} 行 · 每 1s 刷新）</span>
        </div>
        <div className="log-well">
          {log && log.length > 0 ? (
            log.map((line, i) => (
              <span className={logLineClass(line)} key={i}>
                {line}
              </span>
            ))
          ) : (
            <span className="log-empty">（暂无日志输出）</span>
          )}
        </div>
      </div>
    </div>
  );
}

export function RunList(): JSX.Element {
  const runs = useConsole((s) => s.runs);
  const byStatus = useConsole((s) => s.byStatus);
  const runsTotal = useConsole((s) => s.runsTotal);
  const details = useConsole((s) => s.details);
  const expanded = useConsole((s) => s.expanded);
  const busyRunId = useConsole((s) => s.busyRunId);
  const actionError = useConsole((s) => s.actionError);
  const listError = useConsole((s) => s.listError);
  const polling = useConsole((s) => s.polling);
  const toggleExpand = useConsole((s) => s.toggleExpand);
  const cancelRun = useConsole((s) => s.cancelRun);

  return (
    <Panel
      idx="A3"
      title="运行台账"
      tag={polling ? "轮询 1s ●" : "已暂停"}
      style={{ ["--d" as string]: "60ms" }}
      className="reveal"
    >
      <div className="by-status">
        {BY_STATUS_ORDER.map((s) => (
          <span className={`bs-item bs-${s}`} key={s}>
            <StatusLed status={s} />
            {s}
            <b>{byStatus[s] ?? 0}</b>
          </span>
        ))}
        <span className="bs-item">
          总计 <b>{runsTotal}</b>
        </span>
      </div>

      {listError && (
        <ErrorBox error={listError} />
      )}

      <div className="run-list" style={{ marginTop: 12 }}>
        {runs.length === 0 && !listError ? (
          <div className="empty-note">
            尚无实验。用左侧「启动实验」提交一个 run（POST /experiments → 201），
            台账会每 1s 轮询其状态与日志。
          </div>
        ) : (
          runs.map((r) => {
            const full = pickRun(runs, details, r.run_id);
            const isOpen = expanded === r.run_id;
            return (
              <div
                className={`run-card st-${r.status}${isOpen ? " selected" : ""}`}
                key={r.run_id}
              >
                <div
                  className="run-hd"
                  role="button"
                  tabIndex={0}
                  onClick={() => toggleExpand(r.run_id)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" || e.key === " ") {
                      e.preventDefault();
                      toggleExpand(r.run_id);
                    }
                  }}
                >
                  <StatusLed status={r.status} />
                  <div className="run-title">
                    <div className="run-name">{r.name || "(未命名)"}</div>
                    <div className="run-id">{r.run_id}</div>
                  </div>
                  <RunSpec run={full} />
                  <span className="dim2 mono small">{isOpen ? "▾" : "▸"}</span>
                </div>
                {isOpen ? (
                  <RunBody
                    run={full}
                    busy={busyRunId === r.run_id}
                    onCancel={() => void cancelRun(r.run_id)}
                  />
                ) : null}
              </div>
            );
          })
        )}
      </div>

      {actionError ? <ErrorBox error={actionError} /> : null}
    </Panel>
  );
}
