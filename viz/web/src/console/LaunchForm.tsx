/**
 * A3 · 启动实验表单。
 * 映射契约：device(GET /devices 白名单 + fields 回显 R278) / seed≥0 / ticks≥1 /
 *   arm∈[1,4]（固定四臂下拉，默认 1）/ name?≤64 / script?(默认 /health.default_script) /
 *   extra_args? / params? / live。
 * 提交 → POST /experiments（201）。失败走统一错误信封：422 逐字段、400 unknown_device/script_not_found。
 * 启动成功且 live=true ⇒ 自动切世界页签（onGoWorld，与快照面板复用同一回调）并
 *   loadLive(run_id) 进入实时观看；失败 / live=false 不跳转。
 */
import { useEffect, useMemo, useState } from "react";
import type { FormEvent, JSX } from "react";
import { getParamSchema } from "../app/api";
import type { ExperimentCreateBody, ParamSchema, ParamSchemaItem } from "../app/api";
import { useConsole } from "../app/console";
import { useViz } from "../app/store";
import { ErrorBox, OkBox, Panel, Tip } from "./bits";
import { fmtFieldValue } from "./format";

/** 装置回显字段白话（供人眼复核；未知键不提示） */
const DF_HINT: Record<string, string> = {
  rows: "格网行数：世界被划分成多少行",
  cols: "格网列数：世界被划分成多少列",
  patches: "斑块数：初始资源斑块的个数",
  pop: "种群数量：初始生物个体总数",
  rgm: "再生增益乘子：资源再生速率的倍率",
  bg_low_prod_frac: "低产能背景占比：低产能格占全图的比例",
  bg_low_cap_mult: "低产能容量乘子：低产能格的资源容量倍率",
};

/** 高级参数白话（键名对齐 GET /param-schema；diff 组口径以脚本报告为准 → 用后端 warning，不在此编） */
const PARAM_HINT: Record<string, string> = {
  pop: "生物数量：世界里生物个体的总数（0 = 纯场无生物）。",
  sun_budget: "太阳能量预算总量：系统可供分配的光照能量总和。",
  stress_rate: "非生物胁迫强度：环境对生物的损伤强度。",
  abiotic_budget: "能量守恒账本开关：开启后逐笔记录能量收支，守恒可审计。",
  area_weighted: "面积权加权：统计按格网真实面积加权（球面高纬格更小）。",
  abiotic_stress: "胁迫损伤：开启非生物胁迫对生物的损伤项。",
  patch_mode: "斑块生成方式：资源斑块的空间布局算法（seed_dilate=种子扩散 / field_threshold=场阈值）。",
  patch_coverage: "斑块覆盖比例：资源斑块占全图面积的比例（0–1 开区间）。",
  env: "地形环境场：叠加地形环境（对资源与移动的空间影响）。",
};

export function LaunchForm({
  onGoWorld,
}: {
  /** 启动成功（live）后自动切世界页签（App 经 ExperimentConsole 下放，与快照面板复用同一回调） */
  onGoWorld?: () => void;
}): JSX.Element {
  const devices = useConsole((s) => s.devices);
  const devicesNote = useConsole((s) => s.devicesNote);
  const health = useConsole((s) => s.health);
  const submitting = useConsole((s) => s.submitting);
  const submitError = useConsole((s) => s.submitError);
  const lastCreatedRunId = useConsole((s) => s.lastCreatedRunId);
  const submitExperiment = useConsole((s) => s.submitExperiment);
  const loadLive = useViz((s) => s.loadLive);

  const [device, setDevice] = useState("");
  const [seed, setSeed] = useState("42");
  const [ticks, setTicks] = useState("60000");
  const [arm, setArm] = useState("1");
  const [name, setName] = useState("");
  const [script, setScript] = useState("");
  const [extraArgs, setExtraArgs] = useState("");
  const [live, setLive] = useState(true);
  const [errs, setErrs] = useState<Record<string, string>>({});
  const [scriptSeeded, setScriptSeeded] = useState(false);
  // 高级参数（GET /param-schema）：编辑态数字/下拉存字符串（留空=不传），开关存布尔
  const [schema, setSchema] = useState<ParamSchema | null>(null);
  const [schemaError, setSchemaError] = useState<string | null>(null);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [pvals, setPvals] = useState<Record<string, string | boolean>>({});
  const [openGroups, setOpenGroups] = useState<Record<string, boolean>>({});

  // 装置档默认选白名单第一个（异步到达后回填一次）——渲染期校正，避免 setState-in-effect
  if (device === "" && devices.length > 0) setDevice(devices[0].name);

  // script 预置为 /health.default_script（仅一次；用户可改）
  if (health && !scriptSeeded) {
    setScriptSeeded(true);
    setScript(health.default_script);
  }

  const selectedDevice = devices.find((d) => d.name === device) ?? null;
  const defaultScript = health?.default_script ?? "experiments/run_bare_world.py";

  // 高级参数：按当前 script 拉参数 schema（未知 script ⇒ 空组，不报错）；
  // setState 全部在异步回调内（避免 set-state-in-effect）；script 连续输入时用 abort 防竞态
  useEffect(() => {
    const ctrl = new AbortController();
    getParamSchema(script.trim() || defaultScript, ctrl.signal)
      .then((s) => {
        if (ctrl.signal.aborted) return;
        setSchema(s);
        setSchemaError(null);
      })
      .catch((e: unknown) => {
        if (ctrl.signal.aborted) return;
        setSchema(null);
        setSchemaError(e instanceof Error ? e.message : String(e));
      });
    return () => ctrl.abort();
  }, [script, defaultScript]);

  // 服务端 422 字段集合 → 标记对应输入为 invalid
  const serverFields = useMemo(() => {
    const set = new Set<string>();
    if (submitError) for (const f of submitError.fields) set.add(f.field);
    return set;
  }, [submitError]);

  const inv = (k: string): string => (errs[k] || serverFields.has(k) ? " invalid" : "");

  /** 高级参数编辑态 → 提交体 params（留空/未勾选 = 用脚本默认 = 不传键，后端翻译成旗标） */
  function collectParams(): Record<string, number | boolean | string> {
    const out: Record<string, number | boolean | string> = {};
    if (!schema) return out;
    for (const p of schema.params) {
      const raw = pvals[p.key];
      if (raw === undefined || raw === "") continue;
      if (p.type === "bool") {
        if (raw === true) out[p.key] = true; // false → 不追加旗标（与后端翻译规则一致）
        continue;
      }
      if (p.type === "choice") {
        out[p.key] = String(raw);
        continue;
      }
      const n = Number(raw);
      if (Number.isFinite(n)) out[p.key] = n;
    }
    return out;
  }

  function validate(): Record<string, string> {
    const v: Record<string, string> = {};
    if (!device) v.device = "请选择装置档";
    const sd = Number(seed);
    if (seed.trim() === "" || !Number.isInteger(sd) || sd < 0) v.seed = "seed 须为整数 ≥ 0";
    const tk = Number(ticks);
    if (ticks.trim() === "" || !Number.isInteger(tk) || tk < 1) v.ticks = "ticks 须为整数 ≥ 1";
    if (name.trim().length > 64) v.name = "name ≤ 64 字符";
    // 高级参数本地预检（与后端 translate_params 同规则；后端仍兜底 422）
    for (const p of schema?.params ?? []) {
      const k = `params.${p.key}`;
      const raw = pvals[p.key];
      if (raw === undefined || raw === "" || p.type === "bool") continue;
      if (p.type === "choice") {
        if (!p.choices?.includes(String(raw))) v[k] = "非法选项";
        continue;
      }
      const n = Number(raw);
      if (!Number.isFinite(n)) {
        v[k] = "须为数字";
      } else if (p.key === "sun_budget" && n <= 0) {
        v[k] = "须 > 0";
      } else if (p.key === "patch_coverage" && !(0 < n && n < 1)) {
        v[k] = "须 0 < x < 1";
      } else if (p.min !== null && n < p.min) {
        v[k] = `须 ≥ ${p.min}`;
      } else if (p.max !== null && n > p.max) {
        v[k] = `须 ≤ ${p.max}`;
      }
    }
    return v;
  }

  async function submit(): Promise<void> {
    const v = validate();
    setErrs(v);
    if (Object.keys(v).length > 0) return;
    const extra = extraArgs.trim().length > 0 ? extraArgs.trim().split(/\s+/) : [];
    const params = collectParams();
    const body: ExperimentCreateBody = {
      device,
      seed: Number(seed),
      ticks: Number(ticks),
      live,
      arm: Number(arm), // 固定四臂下拉（1–4），恒传
      ...(name.trim() ? { name: name.trim() } : {}),
      ...(script.trim() ? { script: script.trim() } : {}),
      ...(extra.length ? { extra_args: extra } : {}),
      ...(Object.keys(params).length ? { params } : {}),
    };
    const run = await submitExperiment(body);
    // 启动成功且 live ⇒ 自动切世界页签并进入实时观看（失败 / live=false 不跳）
    if (run && live) {
      onGoWorld?.();
      void loadLive(run.run_id);
    }
  }

  /** 高级参数单项渲染（数字 / 开关 / 下拉三态；title 携带后端 warning） */
  function renderParam(p: ParamSchemaItem): JSX.Element {
    const k = `params.${p.key}`;
    const val = pvals[p.key];
    // 悬浮说明 = 白话 hint + 后端 warning（两者都有则合并在一个 title 里）
    const hint = PARAM_HINT[p.key];
    const title = hint && p.warning ? `${hint}\n⚠ ${p.warning}` : (p.warning ?? hint);
    if (p.type === "bool") {
      return (
        <div className="field span2" key={p.key}>
          <label className="switch" title={title}>
            <input
              type="checkbox"
              checked={val === true}
              onChange={(e) => setPvals((m) => ({ ...m, [p.key]: e.target.checked }))}
              disabled={submitting}
            />
            <span className="track" />
            <span className="sw-label">
              {p.label} — 勾选追加 <code>{p.flag}</code>
            </span>
          </label>
        </div>
      );
    }
    if (p.type === "choice") {
      return (
        <div className="field" key={p.key}>
          <span className="field-label">
            {p.label} <span className="opt">可选</span>
          </span>
          <select
            className={`select${inv(k)}`}
            value={typeof val === "string" ? val : ""}
            onChange={(e) => setPvals((m) => ({ ...m, [p.key]: e.target.value }))}
            disabled={submitting}
            title={title}
          >
            <option value="">默认（不传）</option>
            {(p.choices ?? []).map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>
          {errs[k] && <span className="field-err">{errs[k]}</span>}
        </div>
      );
    }
    return (
      <div className="field" key={p.key}>
        <span className="field-label">
          {p.label}
          {p.unit ? `（${p.unit}）` : ""} <span className="opt">可选</span>
        </span>
        <input
          className={`input tnum${inv(k)}`}
          type="number"
          min={p.min ?? undefined}
          max={p.max ?? undefined}
          step={p.type === "int" ? 1 : "any"}
          placeholder="默认（不传）"
          value={typeof val === "string" ? val : ""}
          onChange={(e) => setPvals((m) => ({ ...m, [p.key]: e.target.value }))}
          disabled={submitting}
          title={title}
        />
        {errs[k] && <span className="field-err">{errs[k]}</span>}
      </div>
    );
  }

  const hasOverride = Object.keys(collectParams()).length > 0;

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
            <Tip text="装置档 = 世界规模预设（白名单制）。S2 为正式实验档：480×960 ≈ 46 万格球面网格；小档用于快速试跑。" />
          </span>
          <select
            className={`select${inv("device")}`}
            value={device}
            onChange={(e) => setDevice(e.target.value)}
            disabled={submitting}
            title="装置档 = 世界规模预设；S2（正式实验档）= 480×960 ≈ 46 万格球面网格"
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
                <span className="df" key={k} title={DF_HINT[k]}>
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
            <Tip text="随机种子：同 seed + 同参数 ⇒ 实验结果可复现（对照实验的锚点）。" />
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
            <Tip text="模拟步数：1 tick = 世界推进一小步。下方注有实测速率供估算时长。" />
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
          <span className="field-hint">
            实测速率：s2 纯场约 13ms/tick ⇒ 60000 tick ≈ 十几分钟
          </span>
        </div>

        <div className="field">
          <span className="field-label">
            实验臂 arm <span className="req">*</span>
          </span>
          <select
            className={`select${inv("arm")}`}
            value={arm}
            onChange={(e) => setArm(e.target.value)}
            disabled={submitting}
          >
            <option value="1">1 · 现状对照（默认）</option>
            <option value="2">2 · 置换零模型</option>
            <option value="3">3 · 地形</option>
            <option value="4">4 · 全规则</option>
          </select>
          <span className="field-hint">
            实验臂 ARM（1=现状对照，2=置换零模型，3=地形，4=全规则；默认 1）
          </span>
        </div>

        <div className="field">
          <span className="field-label">
            名称 name <span className="opt">可选</span>
            <Tip text="实验名称：台账 / 数据包列表里显示的名字；留空自动生成 <device>-arm<arm>。" />
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
            <Tip text="运行哪个实验脚本：不同脚本接受不同的参数（高级参数区按当前脚本给出 schema）。" />
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
          <span className="field-hint">
            额外命令行参数：空格分隔，原样追加在脚本命令行末尾（高级参数区填写的项由后端翻译成旗标）
          </span>
        </div>

        <div className="field span2">
          <div className="adv-hd">
            <button
              type="button"
              className="btn btn-ghost btn-sm"
              onClick={() => setShowAdvanced((v) => !v)}
              disabled={submitting}
            >
              {showAdvanced ? "▾ 高级参数" : "▸ 高级参数"}
            </button>
            {hasOverride && (
              <span className="warn-badge" role="status">
                ⚠ 覆盖默认将产生新配置，读数不与历史可比
              </span>
            )}
            <span className="field-hint">
              {schemaError
                ? "schema 加载失败（高级参数不可用，不影响其余字段）"
                : schema
                  ? `GET /param-schema · ${schema.params.length} 项 · 留空 = 用脚本默认`
                  : "参数 schema 加载中…"}
            </span>
          </div>
        </div>

        {showAdvanced &&
          schema &&
          schema.groups.map((g) => {
            const items = schema.params.filter((p) => p.group === g.key);
            if (items.length === 0) return null;
            const open = openGroups[g.key] ?? !g.collapsed;
            return (
              <div
                key={g.key}
                className={`param-group${g.collapsed ? " param-group-dim" : ""}`}
                data-collapsed={!open}
              >
                <button
                  type="button"
                  className="param-group-hd"
                  onClick={() => setOpenGroups((m) => ({ ...m, [g.key]: !open }))}
                  title={g.note ?? g.label}
                >
                  <span className="pg-label">
                    {open ? "▾" : "▸"} {g.label}
                  </span>
                  {g.note && <span className="pg-note">{g.note}</span>}
                </button>
                {open && <div className="param-grid">{items.map(renderParam)}</div>}
              </div>
            );
          })}

        <div className="field span2">
          <label
            className="switch"
            title="勾选后以 supervise 方式托管运行：按 stride 增量写 live/<run_id>/ 数据包，具备实时帧导出与快照能力；启动成功后自动切到世界页签实时观看"
          >
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
            title="POST /experiments：创建 run 并入台账（每 1s 轮询状态与日志）；勾选 live 时启动成功自动切到世界页签"
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
