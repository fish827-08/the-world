# CONTRACT.md — 可视化数据契约 v0（导出器 ↔ 前端 唯一接口）

> 🔒 **冻结件**（2026-10-10 v0）：改动须轻舟评审 + 两侧同步；简牍（导出）/丹青（渲染）/营造（应用）以此为准。
> 上游设计稿：`docs/设计文档/设计-可视化v0-20261010.md`。

## 0. 版本

- `contract_version = "v0"`（`meta.json` 必含）。前端：不认识的字段**忽略**；不认识的大版本**报错**。

## 1. 数据包布局

```
<datapack>/
├─ meta.json
├─ frames/
│  └─ ch_<key>/
│     └─ 00000.bin, 00001.bin, …        # 每通道每帧一个
├─ entities/
│  └─ 00000.f32, 00001.f32, …           # 每帧一个（可为空个体帧 ⇒ 0 字节）
└─ series.csv                            # 可选（有实验序列时随包提供）
```

## 2. `meta.json` 模式

```json
{
  "contract_version": "v0",
  "world":   {"rows": 60, "cols": 120, "n_cells": 7200, "projection": "equirect"},
  "tick":    {"start": 300000, "end": 300200, "stride": 20, "n_frames": 11},
  "channels": [
    {"key": "resource", "label": "资源场", "kind": "grid",
     "scale": {"mode": "quantile", "p_lo": 0.0, "p_hi": 4.5, "min": 0.0, "max": 9.1, "q_lo": 0.005, "q_hi": 0.995},
     "cmap": "viridis"}
  ],
  "entities": {"columns": ["flat", "sub_r", "sub_c", "energy", "age", "generation", "mode"],
               "dtype": "float32",
               "norm": {"energy": {"p_lo": 1.2, "p_hi": 40.0}, "age": {"p_lo": 0.0, "p_hi": 1200.0}}},
  "source":  {"snapshot": "…npz 路径", "snapshot_tick": 300000,
              "config_fingerprint": "…", "engine": "python|rust"},
  "generator": {"tool": "viz/exporter", "version": "v0", "created_at": "ISO8601"}
}
```

## 3. `frames/ch_<key>/NNNNN.bin`

- 行主序 `uint8`，长度 = `rows*cols`；帧号从 0 起，5 位零填充。
- 球面等距圆柱投影 ⇒ 可直视为 `rows×cols` 位图（行 0 = 北极端；`flat = r*cols + c`）。
- 值映射：`u8 = round(clip((v - scale.p_lo) / (scale.p_hi - scale.p_lo), 0, 1) * 255)`；
  标定（分位裁剪）由导出器整段算出并记入 meta ⇒ **帧间稳定不闪**；**只进显示，不改数**。

## 4. `entities/NNNNN.f32`

- 小端 `float32` **交错**记录，每实体 **7** 个值：`flat, sub_r, sub_c, energy, age, generation, mode`（**原始值**，非归一）。
- 实体序 = id 升序；行数 = `file_size / 28`（帧间可变，个体数逐帧不同）。
- 前端叠加：位置优先用 `sub_r/sub_c` + 配置 `subdiv` 细化到亚格（无亚格数据时回退格中心）；着色建议用 `meta.entities.norm`。
- ⚠️ 旧快照无亚格/模式键时，载入侧回退零值 ⇒ 零值语义 = "无数据/关档"。

## 5. `series.csv`

- 直接沿用实验侧逐 tick CSV（表头为准，如 `tick,pop,global_sat,abs_food,abs_cap,ms_per_tick,n_patches,…`）；导出器**不重排、不改列**。

## 6. 通道 keys（v0 首批）

| key | 来源（引擎状态） | 说明 |
|---|---|---|
| `resource` | `eng.resources._grid` | 资源场 |
| `energy` | `eng._energy` 经 `flat` 聚合（`np.bincount`，空=0） | 种群能量密度 |
| `fruit` | `eng._fruit_grid`（可选） | 果实场 |

后续（corpse/smell/signal…）以**新 key 追加**，未知 key 前端忽略。

## 7. 生成方式（导出器 CLI）

```bash
python viz/exporter/export_datapack.py --snapshot X.npz --out DIR \
    [--ticks N] [--stride K] [--channels resource,energy] [--series PATH] [--force-python]
```

- `--ticks 0` ⇒ 仅导出快照当前 tick 单帧（冒烟）。
- `--ticks N` ⇒ 从快照起**确定性续跑** N tick（引擎 Python 路径；`--force-python` 翻转 `use_sim_core`），每 K tick 采一帧。复跑可复现。
- 内存守卫：整段帧以 float32 缓冲算分位；超限（默认 512 MB）报错并提示调大 `--stride`。

## 8. 校验

`python viz/exporter/check_datapack.py DIR`：meta 一致性 / 帧文件计数与长度 / 实体文件对齐（%28）/ 抽样 NaN 检查；退出码 0=过、2=失败。
