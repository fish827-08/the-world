/**
 * A3/A4 · 实验控制台布局容器。
 * 生命周期：挂载时 bootstrap（health + devices + datapacks + 首轮列表）并起 1s 台账轮询；
 *   卸载停轮询（清定时器）。isBootstrapped 防 StrictMode 双跑重复请求。
 * 布局：左列（启动实验 / 保存快照 / 快照加载）· 右列（运行台账 / 实时监控 / 数据包）。
 * 顶部健康条：GET /health 关键读数（api_version / data_root / registered_runs）。
 */
import { useEffect } from "react";
import type { JSX } from "react";
import { useConsole, isBootstrapped } from "../app/console";
import { DatapackBrowser } from "./DatapackBrowser";
import { LaunchForm } from "./LaunchForm";
import { LiveMonitor } from "./LiveMonitor";
import { RunList } from "./RunList";
import { SnapshotPanel } from "./SnapshotPanel";
import { SnapshotsLoader } from "./SnapshotsLoader";
import { StatusLed } from "./bits";
import { fmtNum } from "./format";

function HealthStrip(): JSX.Element {
  const health = useConsole((s) => s.health);
  const apiUp = useConsole((s) => s.apiUp);
  const healthError = useConsole((s) => s.healthError);
  const polling = useConsole((s) => s.polling);

  if (!health) {
    return (
      <div className="api-stat">
        <StatusLed status={apiUp ? "running" : "failed"} />
        {apiUp ? "连接中…" : healthError ? "API 不可达" : "API —"}
      </div>
    );
  }
  return (
    <div className="row wrap" style={{ gap: 8 }}>
      <span className="api-stat">
        <StatusLed status={apiUp ? (polling ? "running" : "finished") : "failed"} />
        {health.service} v{health.api_version}
      </span>
      <span className="api-stat">
        <StatusLed status={health.data_root_exists ? "finished" : "failed"} />
        data_root {health.data_root_exists ? "✓" : "✕ 缺失"}
      </span>
      <span className="api-stat">
        runs <b style={{ color: "var(--accent-2)" }}>{fmtNum(health.registered_runs)}</b>
      </span>
      <span className="api-stat">
        devices <b style={{ color: "var(--accent-2)" }}>{health.devices.length}</b>
      </span>
      {!health.default_script_exists && (
        <span className="api-stat" style={{ borderColor: "rgba(255,107,107,0.4)", color: "var(--failed)" }}>
          ⚠ 默认脚本缺失
        </span>
      )}
    </div>
  );
}

export function ExperimentConsole(): JSX.Element {
  const bootstrap = useConsole((s) => s.bootstrap);
  const setPolling = useConsole((s) => s.setPolling);

  useEffect(() => {
    if (!isBootstrapped()) void bootstrap();
    setPolling(true);
    return () => setPolling(false);
  }, [bootstrap, setPolling]);

  return (
    <div className="console">
      <div style={{ maxWidth: 1680, margin: "0 auto 14px" }}>
        <HealthStrip />
      </div>
      <div className="console-grid">
        <div className="col-left">
          <LaunchForm />
          <SnapshotPanel />
          <SnapshotsLoader />
        </div>
        <div className="col-right">
          <RunList />
          <LiveMonitor />
          <DatapackBrowser />
        </div>
      </div>
    </div>
  );
}
