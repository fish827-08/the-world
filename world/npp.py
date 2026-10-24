"""NPP（净初级生产力）场与"斑块 = NPP 超阈值连通区"的派生统计（A13 / A16a）。

口径（v1，按审核线裁定 A1 的三条件）
----------------------------------
* **只用世界里真实存在的场**：光照（几何年代均辐照代理 ∝ cosφ）、温度（`EnvField.temp`）、
  降水（`EnvField.pr`，气候态 proxy）。
  🔴 **不使用 soil_water / 养分**——世界里没有这两个场（土壤水是 S1 计划、养分不存在）。
  干旱指数用 `P/(P+PET)` 代理（PET = `pet_base + pet_coef·max(T,0)`），**不当真实蒸散**。
* NPP 只作**容量/再生的乘性因子**（与既有 `env_growth_factor`/`env_capacity_factor` 同一条通路）
  ⇒ 关档（`enabled=False`）时本模块**不被调用**，旧路径逐位等价。
* "斑块"是**派生量**：`NPP > quantile(NPP, 1-threshold_frac)` 的 4-邻接连通区域（经度环向连通）。
  只进 probe/CSV 统计，**不参与容量计算**（容量只看 NPP 本身）。

⚠️ 参数**未校准**（文献锚缺失）：默认值只是"合理起点"，第一次用必须做敏感性扫描，
   不许把参数选择当成世界差异（变更单 §七-1 已写明）。
"""
from __future__ import annotations

import numpy as np


#: 允许的 NPP 形式（A18.1 / 写死审计 H3）
#: * `multiplicative`：现状。四项连乘（任一为 0 则 NPP=0，"限制因子可互相补偿"）。
#: * `liebig`：Liebig 最小因子律 `min(光, 温, 水, 干旱)`（**不可补偿**：最缺的那项说了算）。
#: * `light_temp_only`：只吃光温（把"水"完全交给别处）——用来量化"水项到底改变了什么"。
NPP_FORMS: tuple[str, ...] = ("multiplicative", "liebig", "light_temp_only")


def compute_npp(temp_c: np.ndarray, precip_proxy: np.ndarray, lat_rad: np.ndarray,
                *, form: str = "multiplicative",
                light_exponent: float = 1.0, temp_opt_c: float = 25.0,
                temp_width_c: float = 12.0, precip_half: float = 15.0,
                pet_base: float = 20.0, pet_coef: float = 2.0,
                light_floor: float = 0.02) -> np.ndarray:
    """返回 (n_cells,) 的 NPP 场，归一化到 [0, 1]（全球最大值 = 1）。

    参数均为**代理量纲**（不是真实 W/m²、mm/yr）；只用于相对分布。
    """
    lat = np.asarray(lat_rad, dtype=np.float64)
    t = np.asarray(temp_c, dtype=np.float64)
    p = np.asarray(precip_proxy, dtype=np.float64)
    if t.shape != lat.shape or p.shape != lat.shape:
        raise ValueError(f"形状必须一致：temp{t.shape} precip{p.shape} lat{lat.shape}")

    light = np.clip(np.cos(lat), light_floor, None) ** float(light_exponent)
    ft = np.exp(-((t - float(temp_opt_c)) / float(temp_width_c)) ** 2)
    fp = p / (p + float(precip_half))
    pet = float(pet_base) + float(pet_coef) * np.clip(t, 0.0, None)
    arid = p / (p + pet)

    if form == "multiplicative":
        raw = light * ft * fp * arid
    elif form == "liebig":
        # Liebig 最小因子：四项里最小的那个决定产量（不可补偿）
        raw = np.minimum(np.minimum(light, ft), np.minimum(fp, arid))
    elif form == "light_temp_only":
        raw = light * ft           # 刻意不看水：用来做"水项贡献"的边际对照
    else:
        raise ValueError(f"未知 NPP 形式 {form!r}（允许：{NPP_FORMS}）")
    m = float(raw.mean())
    if not np.isfinite(m) or m <= 0.0:
        raise FloatingPointError("NPP 场全 0 或非有限 —— 检查输入场（温度/降水代理）")
    # 🔴 按**全球均值**归一（均值 = 1），不是按最大值：
    #   容量 = base × NPP ⇒ 均值 1 才保住"全球总容量与 uniform 档同量级"的量纲锚
    #   （按最大值归一会让 90% 的格接近 0 ⇒ 世界大半不可居住，属口径错误）。
    return raw / m


def label_components(mask: np.ndarray, *, wrap_lon: bool = True) -> tuple[np.ndarray, int]:
    """4-邻接连通分量（经度环向可连通）。返回 (labels, n)，未选中的格 = -1。

    纯 numpy + 并查集（不引 scipy）；一次构造期调用，460k 格 ≈ 亚秒级。
    """
    m = np.asarray(mask, dtype=bool)
    if m.ndim != 2:
        raise ValueError(f"mask 必须是 2D，收到 {m.shape}")
    rows, cols = m.shape
    labels = -np.ones(m.shape, dtype=np.int64)
    n_on = int(m.sum())
    if n_on == 0:
        return labels, 0
    idx = np.arange(n_on, dtype=np.int64).reshape(-1)
    labels[m] = idx
    parent = np.arange(n_on, dtype=np.int64)

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    def _edges(pa: np.ndarray, pb: np.ndarray) -> None:
        ok = (pa >= 0) & (pb >= 0)
        for a, b in zip(pa[ok].ravel(), pb[ok].ravel()):
            union(int(a), int(b))

    _edges(labels[:, :-1], labels[:, 1:])          # 东西向
    _edges(labels[:-1, :], labels[1:, :])          # 南北向
    if wrap_lon and cols > 1:
        _edges(labels[:, -1], labels[:, 0])        # 0/360 接缝

    roots = np.array([find(i) for i in range(n_on)], dtype=np.int64)
    uniq, inv = np.unique(roots, return_inverse=True)
    labels[m] = inv
    return labels, int(uniq.size)


def box_mean(a: np.ndarray, window: int, *, wrap_lon: bool = True) -> np.ndarray:
    """(rows, cols) 的**矩形窗均值**（可分离两次 cumsum；经度环向、纬度边缘夹取）。

    A17.4：局部化阈值要用"局地平滑值"做基准。窗口按奇数取（`window=9` ⇒ 半径 4）。
    纯 numpy，无 scipy；480×960 实测 ≈ 10 ms。
    """
    arr = np.asarray(a, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"box_mean 只接受 2D，收到 {arr.shape}")
    w = max(int(window), 1)
    if w % 2 == 0:
        w += 1
    if w == 1:
        return arr.copy()
    r = w // 2
    rows, cols = arr.shape

    pad = np.pad(arr, ((r, r), (0, 0)), mode="edge")
    cs = np.cumsum(pad, axis=0)
    cs = np.vstack([np.zeros((1, cols)), cs])
    out = (cs[w:] - cs[:-w]) / float(w)                    # (rows, cols)

    if wrap_lon and cols > 1:
        p2 = np.concatenate([out[:, cols - r:], out, out[:, :r]], axis=1)
    else:
        p2 = np.pad(out, ((0, 0), (r, r)), mode="edge")
    cs2 = np.cumsum(p2, axis=1)
    cs2 = np.hstack([np.zeros((rows, 1)), cs2])
    return (cs2[:, w:] - cs2[:, :-w]) / float(w)


def patch_mask(npp: np.ndarray, *, mode: str = "local", threshold_frac: float = 0.35,
               local_window: int = 9, local_delta: float = 0.10,
               land_mask: np.ndarray | None = None) -> np.ndarray:
    """"斑块"判定掩码（A17.4）。

    * `mode="local"`（默认）：`NPP > 局地平滑(NPP) × (1+δ)` —— 只挑**相对周边的高值**，
      平滑气候场（纬度带）因此不会整条入选（A16c 的"环球带"退化的修法）。
    * `mode="global"`：旧口径 `NPP > quantile(NPP, 1-θ)`（保留以便对照，**不推荐**）。
    * `land_mask`：给了就只在陆地上找斑块（海洋是零容量，不该算斑块）。
    """
    npp2 = np.asarray(npp, dtype=np.float64)
    if npp2.ndim != 2:
        raise ValueError(f"npp 必须是 2D，收到 {npp2.shape}")
    if mode not in ("local", "global"):
        raise ValueError(f"patch mode 只能是 local / global，收到 {mode!r}")
    if mode == "local":
        base = box_mean(npp2, int(local_window), wrap_lon=True)
        mask = npp2 > base * (1.0 + float(local_delta))
    else:
        thr = float(np.quantile(npp2, 1.0 - float(threshold_frac)))
        mask = npp2 > thr
    if land_mask is not None:
        lm = np.asarray(land_mask, dtype=bool)
        if lm.shape != npp2.shape:
            raise ValueError(f"land_mask 形状 {lm.shape} ≠ npp {npp2.shape}")
        mask &= lm
    return mask


def patch_stats(npp: np.ndarray, rows: int, cols: int, cell_area_deg2: np.ndarray,
                *, threshold_frac: float = 0.35, wrap_lon: bool = True,
                land_mask: np.ndarray | None = None, mode: str = "local",
                local_window: int = 9, local_delta: float = 0.10,
                min_cells: int = 3) -> dict:
    """把"斑块"统计成可进 probe/CSV 的一行（A13 建立 / A17.4 改造）。

    统计量：块数 / 覆盖率（占全球、占陆地）/ 面积分布（格）/ 等效半径（度、km）/
    **最大块占比**（`patch_max_share`，判断是否退化成"一条带"的关键指标）。
    """
    npp2 = np.asarray(npp, dtype=np.float64).reshape(rows, cols)
    land = None if land_mask is None else np.asarray(land_mask, dtype=bool).reshape(rows, cols)
    mask = patch_mask(npp2, mode=mode, threshold_frac=threshold_frac,
                      local_window=local_window, local_delta=local_delta, land_mask=land)
    labels, n = label_components(mask, wrap_lon=wrap_lon)

    areas = np.asarray(cell_area_deg2, dtype=np.float64).reshape(rows, cols)
    out = {
        "patch_mode": mode,
        "patch_threshold_frac": float(threshold_frac),
        "local_window": int(local_window),
        "local_delta": float(local_delta),
        "npp_threshold": (float(np.quantile(npp2, 1.0 - float(threshold_frac)))
                          if mode == "global"
                          else float(np.mean(box_mean(npp2, int(local_window))
                                             * (1.0 + float(local_delta))))),
        "n_patches": int(n),
        "patch_cover_frac": float(mask.sum()) / float(rows * cols),
        "patch_cover_frac_land": (float(mask.sum()) / float(max(int(land.sum()), 1))
                                  if land is not None else float(mask.sum()) / float(rows * cols)),
        "patch_area_cells_p50": 0.0,
        "patch_area_cells_max": 0,
        "patch_radius_deg_mean": 0.0,
        "patch_radius_km_mean": 0.0,
        "patch_max_share": 0.0,
        "land_frac": (float(land.mean()) if land is not None else 1.0),
        "patch_min_cells": int(min_cells),
    }
    if n == 0:
        return out

    counts = np.bincount(labels[mask], minlength=n)
    # 小块剔除（<min_cells 格）：直接从 labels 抹掉，不重标号（省一次连通域计算）
    keep = counts >= max(int(min_cells), 1)
    if not np.all(keep):
        drop = ~keep
        drop_ids = np.flatnonzero(drop)
        if drop_ids.size:
            mask = mask & ~np.isin(labels, drop_ids)
            labels = np.where(np.isin(labels, drop_ids), -1, labels)
        counts = counts[keep]
        n = int(counts.size)
    out["n_patches"] = int(n)
    out["patch_cover_frac"] = float(mask.sum()) / float(rows * cols)
    if land is not None:
        out["patch_cover_frac_land"] = float(mask.sum()) / float(max(int(land.sum()), 1))
    if n == 0 or counts.size == 0:
        return out

    sizes_deg2 = np.bincount(labels[mask], weights=areas[mask],
                             minlength=int(labels.max()) + 1)
    sizes_deg2 = sizes_deg2[sizes_deg2 > 0]
    live = counts[counts > 0]
    out["patch_area_cells_p50"] = float(np.median(live))
    out["patch_area_cells_max"] = int(np.max(live))
    out["patch_max_share"] = float(np.max(live) / np.sum(live))
    # 面积单位 = 平方度 ⇒ 半径（度）→ km（×111.32，球面平均；气候统计够用）
    out["patch_radius_deg_mean"] = float(np.mean(np.sqrt(sizes_deg2 / np.pi)))
    out["patch_radius_km_mean"] = out["patch_radius_deg_mean"] * 111.32
    return out
