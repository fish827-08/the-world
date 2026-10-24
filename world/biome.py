"""群系分类（A20.2）：从气候量**派生**每格"属于哪类群系"，不写死"这里是什么"。

口径（阈值全部可配，进 `BiomeConfig` 与账本）
-------------------------------------------
输入四个环境量（都可以从世界/气候源直接拿到）：
  * `temp_c`         年均温（℃）—— GCM 表的 `temperature_c` 或 `EnvField.temp`
  * `precip_mm_yr`   年降水（mm/yr）—— **必须是 mm/yr**；`EnvField.pr` 是 proxy 量纲，不能直接喂
  * `season_amp_c`   季节振幅（月均温的 max−min，℃；可选，缺省 0）
  * `aridity`        干旱指数代理（可选；缺省用 precip/PET 的简化式）

判定顺序（先极端、后过渡；**顺序本身是口径**，改顺序 = 改分类）：
  1. `temp < ice_t`                      → 冰盖/极地荒漠（ice）
  2. `temp < tundra_t`                   → 苔原（tundra）
  3. `precip < desert_p`                 → 荒漠（desert）
  4. `temp > tropical_t ∧ precip > rainforest_p` → 热带雨林（tropical_rainforest）
  5. `temp > tropical_t`                 → 热带季风/稀树草原（savanna）
  6. `temp > temperate_t ∧ season_amp > continental_amp` → 温带大陆性（temperate_continental）
  7. `temp > temperate_t`                → 温带海洋性（temperate_oceanic）
  8. 其余                                 → 寒温带针叶林（boreal）

⚠️ 这是**分类器**不是"生物群区模型"：阈值是人类给的（Köppen 量级），所以阈值敏感性必须一起报。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: 群系编号（**只增不改**：改顺序会作废旧读数）
BIOME_NAMES: tuple[str, ...] = (
    "ice", "tundra", "desert", "tropical_rainforest", "savanna",
    "temperate_continental", "temperate_oceanic", "boreal",
)


@dataclass
class BiomeThresholds:
    """分类阈值（世界族参数；默认取 Köppen 量级，**未校准**）。"""

    ice_t: float = -10.0             # ℃：低于此视为冰盖/极地荒漠
    tundra_t: float = -2.0           # ℃：低于此视为苔原
    desert_p: float = 250.0          # mm/yr：低于此视为荒漠
    rainforest_p: float = 1800.0     # mm/yr：热带雨林的水分门限
    tropical_t: float = 20.0         # ℃：热带门限
    temperate_t: float = 8.0         # ℃：温带门限
    continental_amp: float = 12.0    # ℃：季节振幅超过此值算大陆性

    def __post_init__(self) -> None:
        assert self.ice_t < self.tundra_t < self.temperate_t < self.tropical_t, \
            "温度阈值必须严格递增（ice < tundra < temperate < tropical）"
        assert 0.0 < self.desert_p < self.rainforest_p, "降水阈值必须 0 < desert < rainforest"
        assert self.continental_amp > 0.0, "continental_amp > 0"


def classify(temp_c, precip_mm_yr, *, season_amp_c=None,
             thresholds: BiomeThresholds | None = None) -> np.ndarray:
    """返回 int8 群系编号数组（与输入同形状），编号见 `BIOME_NAMES`。"""
    th = thresholds or BiomeThresholds()
    t = np.asarray(temp_c, dtype=np.float64)
    p = np.asarray(precip_mm_yr, dtype=np.float64)
    if t.shape != p.shape:
        raise ValueError(f"temp {t.shape} 与 precip {p.shape} 形状不一致")
    if not (np.isfinite(t).all() and np.isfinite(p).all()):
        raise ValueError("输入含 NaN/Inf —— 群系分类拒绝用坏数据")
    if float(p.min()) < 0.0:
        raise ValueError("降水不得为负")
    amp = np.zeros_like(t) if season_amp_c is None else np.asarray(season_amp_c, dtype=np.float64)
    if amp.shape != t.shape:
        raise ValueError(f"season_amp {amp.shape} 与 temp {t.shape} 不一致")

    out = np.full(t.shape, BIOME_NAMES.index("boreal"), dtype=np.int8)
    out = np.where(t > th.temperate_t, BIOME_NAMES.index("temperate_oceanic"), out)
    out = np.where((t > th.temperate_t) & (amp > th.continental_amp),
                   BIOME_NAMES.index("temperate_continental"), out)
    out = np.where(t > th.tropical_t, BIOME_NAMES.index("savanna"), out)
    out = np.where((t > th.tropical_t) & (p > th.rainforest_p),
                   BIOME_NAMES.index("tropical_rainforest"), out)
    out = np.where(p < th.desert_p, BIOME_NAMES.index("desert"), out)
    out = np.where(t < th.tundra_t, BIOME_NAMES.index("tundra"), out)
    out = np.where(t < th.ice_t, BIOME_NAMES.index("ice"), out)
    return out


def biome_stats(biome: np.ndarray, areas=None) -> dict:
    """覆盖率统计（给了 `areas` 就是**面积权**，否则按格数）。"""
    b = np.asarray(biome).ravel()
    b = np.where(b < 0, 0, b)        # -1（海洋/未分类）归到 0 槽但在 counts 里额外扣除
    w = None if areas is None else np.asarray(areas, dtype=np.float64).ravel()
    if w is not None and w.size != b.size:
        raise ValueError(f"areas 长度 {w.size} ≠ 格数 {b.size}")
    counts = np.bincount(b.astype(np.int64), minlength=len(BIOME_NAMES))
    if w is not None:
        counts = np.bincount(b.astype(np.int64), weights=w, minlength=len(BIOME_NAMES))
    n_ocean = 0.0
    tot = float(counts.sum())
    frac = counts / tot if tot > 0 else counts * 0.0
    return {
        "names": list(BIOME_NAMES),
        "counts": {name: float(c) for name, c in zip(BIOME_NAMES, counts)},
        "frac": {name: round(float(f), 6) for name, f in zip(BIOME_NAMES, frac)},
        "dominant": BIOME_NAMES[int(np.argmax(frac))] if tot > 0 else None,
    }
