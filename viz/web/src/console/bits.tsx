/**
 * 控制台共享展示件（全部为组件，符合 react/only-export-components）。
 * mission-control 视觉：HUD 角标面板 / 状态 LED / 徽章 / 错误信封盒。
 */
import type { CSSProperties, JSX, ReactNode } from "react";
import { ApiError } from "../app/api";
import type { RunStatus } from "../app/api";
import { STATUS_HINT, STATUS_LABEL, badgeClass, ledClass } from "./format";

export function Panel({
  idx,
  title,
  tag,
  cyan,
  actions,
  children,
  className,
  bodyClass,
  style,
  scan,
}: {
  idx?: string;
  title: string;
  tag?: ReactNode;
  cyan?: boolean;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
  bodyClass?: string;
  style?: CSSProperties;
  scan?: boolean;
}): JSX.Element {
  const cls = ["panel", cyan ? "panel-cyan" : "", scan ? "live-scan" : "", className ?? ""]
    .filter(Boolean)
    .join(" ");
  return (
    <section className={cls} style={style}>
      <header className="panel-hd">
        <h3>
          {idx ? <span className="idx">{idx}</span> : null}
          {title}
        </h3>
        {tag ? <span className="panel-tag">{tag}</span> : null}
        <span className="spacer" />
        {actions}
      </header>
      <div className={bodyClass ?? "panel-bd"}>{children}</div>
    </section>
  );
}

export function StatusLed({ status }: { status: RunStatus | string | null }): JSX.Element {
  return <span className={`led ${ledClass(status)}`} aria-hidden="true" />;
}

export function StatusBadge({ status }: { status: RunStatus | string }): JSX.Element {
  const label = STATUS_LABEL[status as RunStatus] ?? String(status);
  return (
    <span
      className={badgeClass(status)}
      title={STATUS_HINT[status as RunStatus] ?? `${label}：run 状态`}
    >
      <StatusLed status={status} />
      {label}
    </span>
  );
}

/** 极简悬浮提示（纯 CSS，无新依赖）：问号圆点，hover / 键盘聚焦显示气泡（样式见 console.css .tip） */
export function Tip({ text }: { text: string }): JSX.Element {
  return (
    <span className="tip" tabIndex={0}>
      <span aria-hidden="true">?</span>
      <span className="tip-bd" role="tooltip">
        {text}
      </span>
    </span>
  );
}

/** 统一错误信封展示：error_code + message + （422）逐字段 */
export function ErrorBox({ error }: { error: ApiError | string | null }): JSX.Element | null {
  if (!error) return null;
  if (typeof error === "string") {
    return (
      <div className="err-box" role="alert">
        <div className="eb-hd">请求失败</div>
        <div className="eb-msg">{error}</div>
      </div>
    );
  }
  const fields = error.fields;
  return (
    <div className="err-box" role="alert">
      <div className="eb-hd">
        <span>请求被拒绝</span>
        <span className="eb-code">{error.errorCode}</span>
        {error.status ? <span className="eb-code">HTTP {error.status}</span> : null}
      </div>
      <div className="eb-msg">{error.message}</div>
      {fields.length > 0 && (
        <div className="eb-fields">
          {fields.map((f, i) => (
            <div className="eb-field" key={`${f.field}-${i}`}>
              <b>{f.field || "(body)"}</b>：{f.msg}
              {f.type ? <span className="dim2"> [{f.type}]</span> : null}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export function OkBox({ children }: { children: ReactNode }): JSX.Element {
  return (
    <div className="ok-box" role="status">
      {children}
    </div>
  );
}
