/**
 * 控制条：播放/暂停、步进、帧滑条、tick 读数、fps、图层切换。
 */
import { frameToTick } from "../contract";
import { FPS_CHOICES, useViz } from "./store";

export function ControlsBar() {
  const meta = useViz((s) => s.meta);
  const frameIdx = useViz((s) => s.frameIdx);
  const playing = useViz((s) => s.playing);
  const fps = useViz((s) => s.fps);
  const layer = useViz((s) => s.layer);
  const togglePlaying = useViz((s) => s.togglePlaying);
  const step = useViz((s) => s.step);
  const seek = useViz((s) => s.seek);
  const setFps = useViz((s) => s.setFps);
  const setLayer = useViz((s) => s.setLayer);

  if (!meta) return null;
  const n = meta.tick.n_frames;
  const tick = frameToTick(meta, frameIdx);

  return (
    <div className="controls">
      <button
        type="button"
        onClick={() => step(-1)}
        disabled={frameIdx <= 0}
        title="上一帧"
      >
        ◀
      </button>
      <button
        type="button"
        className="play"
        onClick={togglePlaying}
        title={playing ? "暂停" : "播放"}
      >
        {playing ? "❚❚" : "▶"}
      </button>
      <button
        type="button"
        onClick={() => step(1)}
        disabled={frameIdx >= n - 1}
        title="下一帧"
      >
        ▶
      </button>
      <input
        className="frame-slider"
        type="range"
        min={0}
        max={n - 1}
        value={frameIdx}
        onChange={(e) => seek(Number(e.target.value))}
        title="帧滑条"
      />
      <span className="tick-readout">
        tick <b>{tick}</b> <span className="dim">（帧 {frameIdx + 1}/{n}）</span>
      </span>
      <label className="ctl">
        速度
        <select value={fps} onChange={(e) => setFps(Number(e.target.value))}>
          {FPS_CHOICES.map((f) => (
            <option key={f} value={f}>
              {f} fps
            </option>
          ))}
        </select>
      </label>
      <label className="ctl">
        图层
        <select value={layer} onChange={(e) => setLayer(e.target.value)}>
          {meta.channels.map((c) => (
            <option key={c.key} value={c.key}>
              {c.label}
            </option>
          ))}
        </select>
      </label>
      <span className="dim small">
        tick {meta.tick.start}→{meta.tick.end} · stride {meta.tick.stride}
      </span>
    </div>
  );
}
