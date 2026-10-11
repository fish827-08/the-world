/**
 * A3 · 启动实验表单。
 * 映射契约：device(GET /devices 白名单 + fields 回显 R278) / seed≥0 / ticks≥1 /
 *   arm?≥1 / name?≤64 / script?(默认 /health.default_script) / extra_args? / live。
 * 提交 → POST /experiments（201）。失败走统一错误信封：422 逐字段、400 unknown_device/script_not_found。
 */
import { useMemo, useState } from "react";
import type { FormEvent, JSX } from "react";
import type { ExperimentCreateBody } from "../app/api";
import { useConsole } from "../app/console";
import { ErrorBox, OkBox, Panel } from "./bits";
import { fmtFieldValue } from "./format";

export function LaunchForm(): JSX.Element {
  const devices = useConsole((s) => s.devices);
  const devicesNote = useConsole((s) => s.devicesNote);
  const health = useConsole((s) => s.health);
  const submitting = useConsole((s) => s.submitting);
  const submitError = useConsole((s) => s.submitError);
  const lastCreatedRunId = useConsole((s) => s.lastCreatedRunId);
  const submitExperiment = useConsole((s) => s.submitExperiment);

  const [device, setDevice] = useState("");
  const [seed, setSeed] = useState("42");
  const [ticks, setTicks] = useState("60000");
  const [arm, setArm] = useState("");
  const [name, setName] = useState("");
  const [script, setScript] = useState("");
  const [extraArgs, setExtraArgs] = useState("");
  const [live, setLive] = useState(true);
  const [errs, setErrs] = useState<Record<string, string>>({});
  const [scriptSeeded, setScriptSeeded] = useState(false);

  // 装置档默认选白名单第一个（异步到达后回填一次）——渲染期校正，避免 setState-in-effect
  if (device === "" && devices.length > 0) setDevice(devices[0].name);

  // script 预置为 /health.default_script（仅一次；用户可改）
  if (health && !scriptSeeded) {
    setScriptSeeded(true);
    setScript(health.default_script);
  }

  const selectedDevice = devices.find((d) => d.name === device) ?? null;
  const defaultScript = health?.default_script ?? "experiments/run_bare_world.py";

  // 服务端 422 字段集合 → 标记对应输入为 invalid
  const serverFields = useMemo(() => {
    const set = new Set<string>();
    if (submitError) for (const f of submitError.fields) set.add(f.field);
    return set;
  }, [submitError]);

  const inv = (k: string): string => (errs[k] || serverFields.has(k) ? " invalid" : "");

  function validate(): Record<string, string> {
    const v: Record<string, string> = {};
    if (!device) v.device = "请选择装置档";
    const sd = Number(seed);
    if (seed.trim() === "" || !Number.isInteger(sd) || sd < 0) v.seed = "seed 须为整数 ≥ 0";
    const tk = Number(ticks);
    if (ticks.trim() === "" || !Number.isInteger(tk) || tk < 1) v.ticks = "ticks 须为整数 ≥ 1";
    if (arm.trim() !== "") {
      const a = Number(arm);
      if (!Number.isInteger(a) || a < 1) v.arm = "arm 须为整数 ≥ 1（或留空）";
    }
    if (name.trim().length > 64) v.name = "name ≤ 64 字符";
    return v;
  }

  async function submit(): Promise<void> {
    const v = validate();
    setErrs(v);
    if (Object.keys(v).length > 0) return;
    const extra = extraArgs.trim().length > 0 ? extraArgs.trim().split(/\s+/) : [];
    const body: ExperimentCreateBody = {
      device,
      seed: Number(seed),
      ticks: Number(ticks),
      live,
      ...(arm.trim() ? { arm: Number(arm) } : {}),
      ...(name.trim() ? { name: name.trim() } : {}),
      ...(script.trim() ? { script: script.trim() } : {}),
      ...(extra.length ? { extra_args: extra } : {}),
    };
    await submitExperiment(body);
  }

  return (
    <Panel
      idx="A3"
      title="启动实验"
      cyan
      tag={health ? `api ${health.api_version}` : "api —"}
      style={{ ["--d" as string]: "0ms" }}
      className="reveal"
    >
      <form
        className="form-grid"
        noValidate
        onSubmit={(e: FormEvent<HTMLFormElement>) => {
          e.preventDefault();
          void submit();
        }}
      >
        <div className="field span2">
          <span className="field-label">
            装置档 device <span className="req">*</span>
          </span>
          <select
            className={`select${inv("device")}`}
            value={device}
            onChange={(e) => setDevice(e.target.value)}
            disabled={submitting}
          >
            {devices.length === 0 && <option value="">（加载白名单中…）</option>}
            {devices.map((d) => (
              <option key={d.name} value={d.name}>
                {d.name}
              </option>
            ))}
          </select>
          {errs.device && <span className="field-err">{errs.device}</span>}
          <span className="field-hint">
            {devicesNote || "只接受白名单内的 device 名（纪律 R5.6，绝不静默回落）"}
          </span>
        </div>

        {selectedDevice && (
          <div className="device-echo">
            <div className="de-hd">◆ 装置字段回显 · 供人眼复核（{selectedDevice.name}）</div>
            <div className="device-fields">
              {Object.entries(selectedDevice.fields).map(([k, val]) => (
                <span className="df" key={k}>
                  <span className="k">{k}</span>
                  <span className="v">{fmtFieldValue(val)}</span>
                </span>
              ))}
            </div>
          </div>
        )}

        <div className="field">
          <span className="field-label">
            随机种子 seed <span className="req">*</span>
          </span>
          <input
            className={`input tnum${inv("seed")}`}
            type="number"
            min={0}
            step={1}
            value={seed}
            onChange={(e) => setSeed(e.target.value)}
            disabled={submitting}
          />
          {errs.seed && <span className="field-err">{errs.seed}</span>}
        </div>

        <div className="field">
          <span className="field-label">
            tick 数 ticks <span className="req">*</span>
          </span>
          <input
            className={`input tnum${inv("ticks")}`}
            type="number"
            min={1}
            /* step 不可用 1000：min=1 时原生合法值集合为 1+1000k，默认 60000
             * 判 stepMismatch ⇒ 浏览器 :invalid → a11y 树 Invalid=true（未提交就标红）。
             * step="any" 撤掉原生 step 校验，合法性交由本组件 validate()（提交时才红） */
            step="any"
            value={ticks}
            onChange={(e) => setTicks(e.target.value)}
            disabled={submitting}
          />
          {errs.ticks && <span className="field-err">{errs.ticks}</span>}
        </div>

        <div className="field">
          <span className="field-label">
            实验臂 arm <span className="opt">可选</span>
          </span>
          <input
            className={`input tnum${inv("arm")}`}
            type="number"
            min={1}
            step={1}
            placeholder="留空 = 不传"
            value={arm}
            onChange={(e) => setArm(e.target.value)}
            disabled={submitting}
          />
          {errs.arm && <span className="field-err">{errs.arm}</span>}
        </div>

        <div className="field">
          <span className="field-label">
            名称 name <span className="opt">可选</span>
          </span>
          <input
            className={`input${inv("name")}`}
            maxLength={64}
            placeholder="<device>-arm<arm>"
            value={name}
            onChange={(e) => setName(e.target.value)}
            disabled={submitting}
          />
          {errs.name && <span className="field-err">{errs.name}</span>}
        </div>

        <div className="field span2">
          <span className="field-label">
            脚本 script <span className="opt">可选</span>
          </span>
          <input
            className={`input${inv("script")}`}
            spellCheck={false}
            placeholder={defaultScript}
            value={script}
            onChange={(e) => setScript(e.target.value)}
            disabled={submitting}
          />
          <span className="field-hint">
            相对 repo root 的 <code>.py</code>；默认 <code>{defaultScript}</code>
            {health && !health.default_script_exists && (
              <span style={{ color: "var(--failed)" }}>
                {" "}
                ⚠ 默认脚本在本分支不存在（提交将 400 script_not_found）
              </span>
            )}
          </span>
        </div>

        <div className="field span2">
          <span className="field-label">
            额外参数 extra_args <span className="opt">可选</span>
          </span>
          <input
            className="input"
            spellCheck={false}
            placeholder="空格分隔，原样追加，如：--sample 20"
            value={extraArgs}
            onChange={(e) => setExtraArgs(e.target.value)}
            disabled={submitting}
          />
        </div>

        <div className="field span2">
          <label className="switch">
            <input
              type="checkbox"
              checked={live}
              onChange={(e) => setLive(e.target.checked)}
              disabled={submitting}
            />
            <span className="track" />
            <span className="sw-label">live 实时监控（supervise 包装，具备快照 / 取帧能力）</span>
          </label>
        </div>

        <div className="form-actions">
          <button
            className="btn btn-primary"
            type="submit"
            disabled={submitting || devices.length === 0}
          >
            {submitting ? (
              <>
                <span className="spin" /> 启动中…
              </>
            ) : (
              <>▶ 启动实验</>
            )}
          </button>
          <span className="field-hint mono">POST /experiments → 201</span>
        </div>
      </form>

      <ErrorBox error={submitError} />
      {lastCreatedRunId && !submitError && (
        <OkBox>
          已启动 run <code>{lastCreatedRunId}</code> —— 见运行台账（每 1s 轮询状态与日志）。
        </OkBox>
      )}
    </Panel>
  );
}
