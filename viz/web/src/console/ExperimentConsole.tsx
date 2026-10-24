/**
 * A3/A4 · 实验控制台布局容器。
 * 生命周期：挂载时 bootstrap（health + devices + datapacks + 首轮列表）并起 1s 台账轮询；
 *   卸载停轮询（清定时器）。isBootstrapped 防 StrictMode 双跑重复请求。
 * 布局：左列（启动实验 / 保存快照 / 快照加载）· 右列（运行台账 / 实时监控 / 数据包）。
 * 顶部健康条：GET /health 关键读数（api_version / data_root / registered_runs）。
 */
import { useEffect, useState } from "react";
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

/** 术语速查条目（口径与 viz/README.md「术语速查」表同源） */
const GLOSSARY: [string, string][] = [
  ["s2", "正式实验档装置：480×960 = 460,800 格球面网格（世界规模基准）。"],
  ["datapack", "数据包：实验的可视化产物目录（帧序列 + 实体 + meta.json，可选 series.csv），位于 data_root 的 viz/ 或 live/ 下。"],
  ["stride", "采样间隔：每多少 tick 存一帧（stride=20 即每 20 tick 留一帧）。"],
  ["supervise", "后端托管运行：实验进程由 API 按 stride 增量写 live/<run_id>/ 数据包，前端可实时观看。"],
  ["live", "实时模式：表单 live 开关 + live 数据包 + 「实时监控」增量接流的总称。"],
  ["契约 v0", "数据包 / 快照 / 曲线的格式契约（viz/CONTRACT.md，冻结件），读写均按此对齐。"],
];

/** 顶部可折叠「术语速查」面板（默认收起，不占空间；让不懂项目的人也能对照着用） */
function GlossPanel(): JSX.Element {
  const [open, setOpen] = useState(false);
  return (
    <div className="gloss">
      <button
        type="button"
        className="gloss-hd"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
      >
        <span className="gloss-title">{open ? "▾" : "▸"} 术语速查</span>
        <span className="gloss-sum">
          s2 · datapack · stride · supervise · live · 契约 v0
        </span>
      </button>
      {open && (
        <div className="gloss-bd">
          {GLOSSARY.map(([k, v]) => (
            <div className="gloss-item" key={k}>
              <span className="gloss-k">{k}</span>
              <span className="gloss-v">{v}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export function ExperimentConsole({
  onGoWorld,
}: {
  /** 自动切回世界页签（App 下放的 setTab 能力）：快照“导出并载入”成功、
   *  启动实验成功（live）两处共用同一回调 */
  onGoWorld?: () => void;
}): JSX.Element {
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
        <div style={{ marginTop: 10 }}>
          <GlossPanel />
        </div>
      </div>
      <div className="console-grid">
        <div className="col-left">
          <LaunchForm onGoWorld={onGoWorld} />
          <SnapshotPanel />
          <SnapshotsLoader onGoWorld={onGoWorld} />
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
