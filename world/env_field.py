"""ENV 场：地形 → 气候态水分 → 格子产能因子（「纯斑块世界」底层规则线）。

口径来源
--------
* `docs/设计文档/设计-纯斑块世界-底层规则自组织-20261011.md`（§十五 四臂协议）；
* `docs/设计文档/设计-ENV-FIELD-物理场与格子产能-20261008.md`（场定义 H/W/P、归一化
  条款、守卫与算路口径）。

流水线（全部**构造期**算一次 ⇒ 每 tick 零成本；不进快照 —— 从 config 确定性重建）
----------------------------------------------------------------------------------
    H  ← 3D 值噪声 fBm（球面采样；独立 rng 流，**不消费引擎 RNG**）
    T  = base_temperature(φ) − lapse_c×H            （气候态温度，°C）
    q  = 6.11×exp(17.27T/(237.3+T))                 （Magnus 饱和水汽压，hPa）
    Pr = q×(1 + orographic×H)                       （抬升增雨）
    Et = et_frac×q                                  （蒸散）
    A  = D8 汇流累积 max(Pr−Et, 0)                    （priority-flood 路由树；极点行 = 出水口）
    W  = max(Pr−Et, 0) × (1 + runoff_gain×A/(A+A_ref))   （汇流带更湿；A_ref = 99 分位）
    f_g = u^ws / 面积权均值            （再生因子；u = W/W_ref，W_ref = W 的面积权均值）
    f_c = clip(u,0,k_max)^cs / 面积权均值（容量因子）

🔴 两条写死的口径（防"数字没出处"）
------------------------------------
* **水只进 W，W 只进因子**：全链路**没有**任何 `if 河道: 产能×k` 分支 ——
  "河 = 肥沃"不是硬编码，是 f 的结果；`river_mask` 只是**读数命名层**
  （不进产能输入、不作判据前置；砚 69d6e0c §三-3 五道防退化闸的引擎侧落点）。
* **归一化条款**（ENV-FIELD §二-5 硬要求）：两个因子都按**面积权均值 = 1** 归一
  ⇒ Σ容量**逐位守恒**（f_c 乘在容量上、面积权均值恰为 1）、Σ名义再生零阶（面积权）
  守恒 ⇒ K 尺度不被连续场整体抬高/压低（以免抹掉 C4 的"稀缺"前提）。

已明写的近似（实现层声明，不许扩读）
------------------------------------
* 拓扑用 `world._nb_table`（真 8 邻居，极点行 `-1` 占位）—— 比 ENV-FIELD §三-2 的
  "(ii) 粗网格五点"更严格；**极点行**在平滑里跳过、在汇流里当**出水口**（水离场）；
* 填洼 = **priority-flood**（Barnes 2014 的堆式单遍变体，`filled[j] = max(H[j], filled[开堆者]+eps)`，
  eps = 1e-6）。旧"抬升至 min 邻居+eps"的向量化近似需要 O(盆地深度) 轮迭代
  （60×120 冒烟实测 200 轮不收敛 ⇒ 换掉，不许静默留余洼）。
  不可达格（邻居图不连通）⇒ 直接炸（fail-loud），不静默把余洼当汇。
* 路由（接收者）= **`filled` 面的 D8 最陡下降**（Barnes 2014 标准用法）：非洼格
  `filled == H` ⇒ 即地形最陡下降；洼内沿 eps 梯度 ⇒ 流向溢流口。
  无环：每格的开堆者邻元 `filled` 严格更小 ⇒ 最陡下降链严格下降。
* **填洼的实测语义**（`[实测]` s2/seed 42，2026-10-11 诊断）：闭环球面 + 极点=唯一出水口
  ⇒ 低于**极环高度**（两规范出水槽均为 0.4577）的低地连成整体"极海"，`filled` 近似常值面
  （H<0.4577 内拟合抬升面 `filled−H ≈ 0.4535 − 0.98×H`，R²=0.985；被抬升格 50.8%，
  连通块 98.7% 单块 ⇒ 海平面型而非局部碗型）。A 的语义 = **流向两极的汇流树**。
  这是设计（极点=出水口）x 未侵蚀噪声地形的**结构性后果**，不是数值缺陷；
  若将来要"少海/无海"世界，动地形或极环高度即可，**不动路由链**。
"""
from __future__ import annotations

import heapq

import numpy as np
from numpy.typing import NDArray

from world.sphere_world import SphereWorld
from world.light_and_temperature import LightAndTemperature

#: Magnus 公式常数（Tetens 系数；T 为 °C，输出 hPa）
_MAGNUS_A = 17.27
_MAGNUS_B = 237.3
_MAGNUS_C = 6.11
#: 温度下钳（防 |T| 极端时除零；本项目 t_equator/t_pole 默认远在安全域内）
_T_FLOOR = -60.0
#: fBm 独立流常数（与 salt 组合出流名；防同 seed 不同机制撞流）
_STREAM_NOISE = 0xE17F
_STREAM_SHUFFLE = 0x5A17
#: priority-flood 抬升步长（保 filled 沿接收者链严格下降 ⇒ 路由树无环）
_FLOOD_EPS = 1e-6


def _value_noise_3d(rng, p: NDArray, lattice: int, freq: float) -> NDArray:
    """单八度 3D 值噪声：p (3,n) ∈ [-1,1]³ → (n,)，格点值 standard normal。

    三线性插值 + smoothstep 权重；格点索引按 `lattice` 取模（周期格点 ⇒ 无接缝）。
    """
    grid = rng.standard_normal((lattice, lattice, lattice))
    x = (p[0] + 1.0) * 0.5 * (lattice * freq)
    y = (p[1] + 1.0) * 0.5 * (lattice * freq)
    z = (p[2] + 1.0) * 0.5 * (lattice * freq)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    z0 = np.floor(z).astype(np.int64)
    tx = x - x0
    ty = y - y0
    tz = z - z0
    tx = tx * tx * (3.0 - 2.0 * tx)
    ty = ty * ty * (3.0 - 2.0 * ty)
    tz = tz * tz * (3.0 - 2.0 * tz)
    x1 = (x0 + 1) % lattice
    y1 = (y0 + 1) % lattice
    z1 = (z0 + 1) % lattice
    x0 %= lattice
    y0 %= lattice
    z0 %= lattice
    c000 = grid[x0, y0, z0]
    c100 = grid[x1, y0, z0]
    c010 = grid[x0, y1, z0]
    c110 = grid[x1, y1, z0]
    c001 = grid[x0, y0, z1]
    c101 = grid[x1, y0, z1]
    c011 = grid[x0, y1, z1]
    c111 = grid[x1, y1, z1]
    x00 = c000 + (c100 - c000) * tx
    x10 = c010 + (c110 - c010) * tx
    x01 = c001 + (c101 - c001) * tx
    x11 = c011 + (c111 - c011) * tx
    y0v = x00 + (x10 - x00) * ty
    y1v = x01 + (x11 - x01) * ty
    return y0v + (y1v - y0v) * tz


class EnvField:
    """一次构造、全静态的地形—水分—产能因子场（引擎构造期建好，之后只读）。

    实例属性（`__slots__` 全部字段）
    ------------------------------
    world, lt : 网格与光温场（只读引用）。
    cfg : `EnvFieldConfig`（参数契约）。
    h : NDArray[float64] (n_cells,) —— 地形 H ∈ [0,1]（已含 shuffle 置换，若开）。
    h_routed : NDArray[float64] (n_cells,) —— 路由面（平滑 + 填洼后的 H；只用于汇流）。
    temp : NDArray[float64] (n_cells,) —— 气候态温度 T = base − lapse×H（°C）。
    w : NDArray[float64] (n_cells,) —— 水分场 W ≥ 0。
    acc : NDArray[float64] (n_cells,) —— 汇流累积 A（自产 yield 已含）。
    river_mask : NDArray[bool] (n_cells,) —— 汇流带（A ≥ 98 分位；**读数命名层**）。
    filled : NDArray[float64] (n_cells,) —— 填洼面（priority-flood 输出；只进路由，不进物理量）。
    receivers : NDArray[int64] (n_cells,) —— 路由树父节点（`filled` 面 D8 最陡下降；-1 = 出水口）。
    pit_first, flood_filled : int —— 平滑面局部极小值数（洼地首数）/ 被淹没抬升格数（诊断）。
    """

    __slots__ = (
        "world", "lt", "cfg",
        "h", "h_routed", "temp", "w", "acc", "river_mask", "receivers",
        "w_ref", "pit_first", "flood_filled", "filled",
        "_gf", "_cf",
    )

    def __init__(self, world: SphereWorld, lt: LightAndTemperature,
                 cfg, base_seed: int = 42) -> None:
        self.world = world
        self.lt = lt
        self.cfg = cfg
        n = world.n_cells
        cells = np.arange(n, dtype=np.int64)

        # ---- 1) 地形 H：3D 值噪声 fBm（独立流；球面坐标采样 ⇒ 无接缝/极点奇点）----
        rows, cols = world.rows, world.cols
        lat_cell = np.repeat(np.asarray(world._lat, dtype=np.float64), cols)
        lon = 2.0 * np.pi * np.arange(cols, dtype=np.float64) / cols
        lon_cell = np.tile(lon, rows)
        cos_lat = np.cos(lat_cell)
        p = np.stack([cos_lat * np.cos(lon_cell),
                      cos_lat * np.sin(lon_cell),
                      np.sin(lat_cell)])                 # (3, n) 单位球面点
        rng = np.random.default_rng([int(base_seed), int(cfg.salt), _STREAM_NOISE])
        h = np.zeros(n, dtype=np.float64)
        amp, freq, norm = 1.0, 1.0, 0.0
        for _ in range(int(cfg.terrain_octaves)):
            h += amp * _value_noise_3d(rng, p, int(cfg.terrain_lattice), freq)
            norm += amp
            amp *= 0.5
            freq *= 2.0
        h /= norm                                            # 八度归一：H 幅值 ~N(0, σ)
        lo, hi = float(h.min()), float(h.max())
        if hi - lo < 1e-12:
            raise FloatingPointError("ENV 地形退化为常值（幅值 ~0）—— 参数异常，中止")
        h = (h - lo) / (hi - lo)                             # → [0,1]
        if cfg.shuffle:
            # 🔴 置换零模型（对照臂）：保 H 直方图、毁空间结构；独立流（不扰动噪声流）
            rng_sh = np.random.default_rng([int(base_seed), int(cfg.salt), _STREAM_SHUFFLE])
            h = h[rng_sh.permutation(n)]
        self.h = h

        # ---- 2) 气候态温度与水量平衡（Magnus）----
        base_t = lt.base_temperature(cells)                  # 纬度基温（年气候态，°C）
        temp = base_t - float(cfg.lapse_c) * h
        np.clip(temp, _T_FLOOR, None, out=temp)              # 🔴 防极端配置除零
        self.temp = temp
        q = _MAGNUS_C * np.exp(_MAGNUS_A * temp / (_MAGNUS_B + temp))
        net = q * (1.0 + float(cfg.orographic) * h - float(cfg.et_frac))
        yield_ = np.maximum(net, 0.0)                        # 径流产额（土壤赤字 ⇒ 无径流）

        # ---- 3) 路由面：平滑（真 8 邻居）+ priority-flood 填洼（Barnes 2014）----
        nb = world._nb_table                                 # (n, 8)，极点行 = -1
        valid = nb >= 0
        safe_nb = np.where(valid, nb, 0)
        cnt = valid.sum(axis=1).astype(np.float64)
        pole_rows = cnt == 0.0                               # 极点行（真拓扑邻居 = 整行）

        h_r = h.copy()
        if int(cfg.terrain_smooth) > 0:
            alpha = 0.5
            for _ in range(int(cfg.terrain_smooth)):
                mean_nb = h_r[safe_nb].sum(axis=1) / np.maximum(cnt, 1.0)
                new = (1.0 - alpha) * h_r + alpha * mean_nb
                new[pole_rows] = h_r[pole_rows]              # 极点行跳过（邻居 = 整行，见模块头）
                h_r = new

        # priority-flood：堆式单遍（极点行 = 出水口）。不变量（每次开堆）：
        #   filled[j] = max(H[j], filled[开堆者] + eps) ≥ filled[开堆者] + eps
        # ⇒ 沿"接收者 = 开堆者"的链，filled **严格下降** ⇒ 树无环、无需迭代收敛
        #   （旧"抬升至 min 邻居+eps"需 O(盆地深度) 轮，60×120 实测 200 轮不收敛 ⇒ 弃）。
        h_list = h_r.tolist()
        nb_l = nb.tolist()
        # 极点行在 `_nb_table` 是 -1 占位（真邻居 = 相邻纬度带整行，见 `sphere_world`）；
        # 但带内格的 `_nb_table` 只指回**规范槽 col 0**（rc_to_flat 极点坍缩）。
        # ⇒ 遍历表**只给规范槽**（0 与 (rows-1)*cols）补出边 = 与接收者邻接表严格同图；
        #   若给全部极槽补出边，带内格可能被"非规范槽"打开，而该槽不在它的 nb 邻接里
        #   ⇒ 最陡下降无严格下界 ⇒ 实测出环（60×120 cycle at 120）。
        #   其余极槽照旧当出水口（recv=-1、无出边、空转会弹出）。
        nb_l[0] = world._pole_nb[0].tolist()
        nb_l[(world.rows - 1) * world.cols] = world._pole_nb[1].tolist()
        done = bytearray(n)
        filled_l = h_list[:]
        heap: list = []
        for i in np.flatnonzero(pole_rows).tolist():
            done[i] = 1
            heap.append((h_list[i], i))
        heapq.heapify(heap)
        while heap:
            fv, i = heapq.heappop(heap)
            for j in nb_l[i]:
                if j >= 0 and not done[j]:
                    done[j] = 1
                    hv = h_list[j]
                    nv = fv + _FLOOD_EPS
                    if hv > nv:
                        nv = hv
                    filled_l[j] = nv
                    heapq.heappush(heap, (nv, j))
        n_done = int(sum(done))
        if n_done < n:
            raise FloatingPointError(
                f"ENV 路由：priority-flood 仅覆盖 {n_done}/{n} 格（邻居图不连通？）——"
                f"中止（不静默把余洼当汇）。"
            )
        filled = np.asarray(filled_l, dtype=np.float64)
        del nb_l, h_list, filled_l, done

        # 路由 = **filled 面的 D8 最陡下降**（Barnes 2014 的标准出口用法）：
        #   非洼格 filled == H ⇒ 就是地形最陡下降；洼内沿 eps 梯度 ⇒ 流向溢流口（湖面出流）。
        #   无环证明：每格 j 的开堆者 o 满足 filled[o] < filled[j]（严格，见上）⇒
        #   argmin 邻居的 filled < filled[j] ⇒ 沿接收者链严格下降 ⇒ 无环。
        filled_nb = filled[safe_nb]
        filled_nb = np.where(valid, filled_nb, np.inf)
        jmin = np.argmin(filled_nb, axis=1)                  # 平手取首列 ⇒ 确定性
        row_i = np.arange(n)
        receivers = np.where(valid[row_i, jmin], nb[row_i, jmin], -1).astype(np.int64)

        # 诊断：平滑面局部极小值数（"洼地首数"，填洼前的计数）
        v_nb = h_r[safe_nb]
        v_nb = np.where(valid, v_nb, np.inf)
        pit_first = int(((h_r < v_nb.min(axis=1)) & (~pole_rows)).sum())
        self.pit_first = pit_first
        self.flood_filled = int(np.count_nonzero(filled > h_r + 1e-12))
        self.h_routed = h_r
        self.filled = filled
        self.receivers = receivers

        # ---- 4) D8 汇流累积 A：按路由面 filled 降序（接收者恒更低 ⇒ 无环、无需求拓扑序）----
        #   纯 Python 标量循环（~0.3 s @460k，构造期一次性）；顺序 = filled 降序，
        #   处理到 i 时其全部上游已加完 ⇒ a[i] 定格后再摊给它唯一的接收者。
        order = np.argsort(-filled, kind="stable").tolist()
        recv_l = receivers.tolist()
        a_l = yield_.tolist()
        for i in order:
            r = recv_l[i]
            if r >= 0:
                a_l[r] += a_l[i]
        acc = np.asarray(a_l, dtype=np.float64)
        self.acc = acc
        a_ref = float(np.percentile(acc, 99.0))
        if a_ref <= 0.0:
            a_ref = 1.0                                      # 全零场兜底（配置异常，防 0/0）
        amp_r = 1.0 + float(cfg.runoff_gain) * acc / (acc + a_ref)
        self.w = yield_ * amp_r
        self.river_mask = acc >= float(np.percentile(acc, 98.0))

        # ---- 5) 因子（按面积权均值 = 1 归一；sensitivity=0 ⇒ None = 整块跳过）----
        areas = world.cell_area(cells)
        self.w_ref = float(np.sum(self.w * areas) / np.sum(areas))
        if not np.isfinite(self.w_ref) or self.w_ref <= 0.0:
            raise FloatingPointError(
                f"ENV 水分场 W 的面积权均值 = {self.w_ref!r}（应 >0）—— "
                f"et_frac/orographic/lapse_c 组合使竞量衡全负？参数异常，中止（不静默）。"
            )
        self._gf = None
        self._cf = None

    # ---- 因子出口（None = 此项关闭 ⇒ 引擎/资源场整块跳过 = 逐位等价）----------

    def _factor(self, sens: float, clip_hi) -> NDArray[np.float64] | None:
        if sens <= 0.0:
            return None
        u = self.w / self.w_ref
        if clip_hi is not None:
            u = np.clip(u, 0.0, clip_hi)
        f = np.power(u, sens)
        areas = self.world.cell_area(np.arange(self.world.n_cells, dtype=np.int64))
        m = float(np.sum(f * areas) / np.sum(areas))
        if m > 1e-12:                                        # 面积权均值归一（防 0/0）
            f = f / m
        return f

    def growth_factor(self) -> NDArray[np.float64] | None:
        """再生因子 f_g = u^water_sensitivity / 面积权均值；`water_sensitivity=0` ⇒ None。"""
        if self.cfg.water_sensitivity <= 0.0:
            return None
        if self._gf is None:
            self._gf = self._factor(float(self.cfg.water_sensitivity), None)
        return self._gf

    def capacity_factor(self) -> NDArray[np.float64] | None:
        """容量因子 f_c = clip(u,0,k_max)^cap_sensitivity / 面积权均值；`cap_sensitivity=0` ⇒ None。"""
        if self.cfg.cap_sensitivity <= 0.0:
            return None
        if self._cf is None:
            self._cf = self._factor(float(self.cfg.cap_sensitivity), float(self.cfg.k_max))
        return self._cf

    # ---- 诊断（读侧：probe/判读图用；**不进任何判据前置**）----------------------

    def probe(self) -> dict:
        """场统计快照（probe/manifest 用；纯读数，不影响动力学）。"""
        n = self.world.n_cells
        areas = self.world.cell_area(np.arange(n, dtype=np.int64))
        a_sum = float(np.sum(areas))
        u = self.w / self.w_ref
        out = {
            "salt": int(self.cfg.salt), "shuffle": bool(self.cfg.shuffle),
            "h_mean": float(np.sum(self.h * areas) / a_sum),
            "h_std": float(np.sqrt(np.sum((self.h - np.sum(self.h * areas) / a_sum) ** 2
                                          * areas) / a_sum)),
            "temp_min": float(self.temp.min()), "temp_max": float(self.temp.max()),
            "w_mean": self.w_ref, "w_p50": float(np.percentile(self.w, 50.0)),
            "w_p99": float(np.percentile(self.w, 99.0)), "w_max": float(self.w.max()),
            "u_p50": float(np.percentile(u, 50.0)), "u_p99": float(np.percentile(u, 99.0)),
            "u_max": float(u.max()),
            "acc_p50": float(np.percentile(self.acc, 50.0)),
            "acc_p99": float(np.percentile(self.acc, 99.0)),
            "river_cells": int(self.river_mask.sum()),
            "river_frac": float(self.river_mask.sum()) / n,
            "pit_first": self.pit_first, "flood_filled": self.flood_filled,
        }
        gf, cf = self.growth_factor(), self.capacity_factor()
        for tag, f in (("gf", gf), ("cf", cf)):
            if f is None:
                out[f"{tag}_on"] = False
            else:
                out[f"{tag}_on"] = True
                out[f"{tag}_min"] = float(f.min())
                out[f"{tag}_p99"] = float(np.percentile(f, 99.0))
                out[f"{tag}_max"] = float(f.max())
        return out

    def __repr__(self) -> str:  # pragma: no cover - 诊断用
        return (f"EnvField(cells={self.world.n_cells}, salt={self.cfg.salt}, "
                f"shuffle={self.cfg.shuffle}, pits={self.pit_first}/filled{self.flood_filled}, "
                f"rivers={float(self.river_mask.mean()):.3f})")
