"""纯斑块世界四臂判读（read_bare_world.py）——观察侧，零重跑。

口径真源 = 预注册 `docs/预注册/预注册-纯斑块世界-四臂-20261011.md`（§2 全套）；
本脚本只读 `the-world-data/bare_world/arm{N}_s{seed}.{npz,csv,manifest.json}` 产物，
不改引擎、不重跑。脚本偏差 = 脚本 bug，口径以预注册为准。

用法
----
  python experiments/read_bare_world.py --seed 42
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ARMS = (1, 2, 3, 4)


# ---------------------------------------------------------------- 几何/斑块

def _lat_deg(rows: int) -> np.ndarray:
    """行纬度（度），与 SphereWorld._lat 同式。"""
    return np.degrees(-np.pi / 2 + (np.arange(rows) + 0.5) * (np.pi / rows))


def _components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """4 邻域连通域（经度 wrap；极行 {0, rows-1} 垂直豁免）。

    返回 (labels[rows,cols] int32，-1=背景, n_components)。min-标签传播，收敛=稳定。
    """
    rows, cols = mask.shape
    lab = np.where(mask, np.arange(rows * cols, dtype=np.int64).reshape(rows, cols), -1)
    neg = np.full((1, cols), -1, dtype=np.int64)
    nbig = np.int64(rows * cols)                     # 无效邻居哨兵（大于一切有效标签）
    for _ in range(2000):
        up = np.vstack([lab[1:], neg])       # up[r] = lab[r+1]（末行无上邻）
        dn = np.vstack([neg, lab[:-1]])      # dn[r] = lab[r-1]（首行无下邻）
        lf = np.roll(lab, 1, axis=1)         # 经度 wrap
        rt = np.roll(lab, -1, axis=1)
        prop = nbig
        for nb in (up, dn, lf, rt):
            prop = np.minimum(prop, np.where(nb >= 0, nb, nbig))
        new = np.where(mask, np.minimum(lab, prop), -1)
        if np.array_equal(new, lab):
            break
        lab = new
    uniq = np.unique(lab[lab >= 0])
    relabel = np.full(rows * cols, -1, dtype=np.int32)
    for new_id, old in enumerate(uniq):
        relabel[old] = new_id
    return relabel[lab.ravel()].reshape(rows, cols), int(uniq.size)


def _patch_stats(plab: np.ndarray, cap: np.ndarray, u: np.ndarray | None,
                 lat_deg_row: np.ndarray) -> dict[str, np.ndarray]:
    """每斑块 5 特征（预注册 §2.2）。u=None ⇒ 常数 1.0 口径。"""
    n = plab.size
    pid = plab.ravel()
    m = pid >= 0
    pid = pid[m]
    p = int(pid.max()) + 1
    area_w = np.cos(np.radians(np.repeat(lat_deg_row, plab.shape[1])))[m]
    abslat = np.abs(np.repeat(lat_deg_row, plab.shape[1]))[m]
    capv = cap.ravel()[m]
    uv = np.ones_like(capv) if u is None else u.ravel()[m]

    cnt = np.bincount(pid, minlength=p).astype(np.float64)
    w_sum = np.bincount(pid, weights=area_w, minlength=p)
    f1 = np.log10(cnt)
    f2 = np.bincount(pid, weights=abslat * area_w, minlength=p) / w_sum
    cap_sum = np.bincount(pid, weights=capv, minlength=p)
    cap_sq = np.bincount(pid, weights=capv * capv, minlength=p)
    f3 = cap_sum / cnt
    f4 = np.bincount(pid, weights=uv, minlength=p) / cnt
    var = np.maximum(cap_sq / cnt - f3 * f3, 0.0)
    f5 = np.sqrt(var) / np.maximum(f3, 1e-300)
    return {"f1_log_area": f1, "f2_abslat": f2, "f3_cap_mean": f3,
            "f4_u_mean": f4, "f5_cap_cv": f5, "area": cnt}


# ---------------------------------------------------------------- 聚类（手写）

def _kmeans_pp(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n = X.shape[0]
    centers = np.empty((k, X.shape[1]), dtype=np.float64)
    centers[0] = X[rng.integers(n)]
    d2 = ((X - centers[0]) ** 2).sum(axis=1)
    for j in range(1, k):
        total = d2.sum()
        if total <= 0.0:
            centers[j] = X[rng.integers(n)]
        else:
            centers[j] = X[rng.choice(n, p=d2 / total)]
        d2 = np.minimum(d2, ((X - centers[j]) ** 2).sum(axis=1))
    return centers


def _kmeans(X: np.ndarray, k: int, rng: np.random.Generator,
            n_init: int = 10, max_iter: int = 300) -> tuple[np.ndarray, float]:
    best_lab, best_inertia = None, np.inf
    for _ in range(n_init):
        C = _kmeans_pp(X, k, rng)
        lab = np.full(X.shape[0], -1, dtype=np.int64)
        for _ in range(max_iter):
            d2 = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
            new = d2.argmin(axis=1)
            if np.array_equal(new, lab):
                break
            lab = new
            for j in range(k):
                sel = lab == j
                if sel.any():
                    C[j] = X[sel].mean(axis=0)
                else:                                   # 空簇：重挂到最远点
                    far = ((X - C[lab]) ** 2).sum(axis=1).argmax()
                    C[j] = X[far]
        inertia = float(((X - C[lab]) ** 2).sum())
        if inertia < best_inertia:
            best_inertia, best_lab = inertia, lab.copy()
    return best_lab, best_inertia


def _silhouette(X: np.ndarray, lab: np.ndarray) -> float:
    n = X.shape[0]
    D = np.sqrt(np.maximum(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2), 0.0))
    np.fill_diagonal(D, 0.0)
    ks = np.unique(lab)
    means = np.zeros((n, ks.size))
    for j, kv in enumerate(ks):
        sel = lab == kv
        cnt = int(sel.sum())
        means[:, j] = D[:, sel].sum(axis=1) / max(cnt - 1, 1)
    own = np.array([np.where(ks == l)[0][0] for l in lab])
    a = means[np.arange(n), own]
    means[np.arange(n), own] = np.inf
    b = means.min(axis=1)
    sizes = {int(kv): int((lab == kv).sum()) for kv in ks}
    s = (b - a) / np.maximum(np.maximum(a, b), 1e-300)
    s[np.array([sizes[int(l)] for l in lab]) < 2] = 0.0   # 单点簇 ⇒ s=0
    return float(s.mean())


def _ari(a: np.ndarray, b: np.ndarray) -> float:
    ua, ia = np.unique(a, return_inverse=True)
    ub, ib = np.unique(b, return_inverse=True)
    tab = np.zeros((ua.size, ub.size), dtype=np.float64)
    np.add.at(tab, (ia, ib), 1.0)
    n = float(a.size)
    comb2 = lambda x: x * (x - 1.0) / 2.0            # noqa: E731
    sum_c = comb2(tab).sum()
    a_c = comb2(tab.sum(axis=1)).sum()
    b_c = comb2(tab.sum(axis=0)).sum()
    exp = a_c * b_c / comb2(n)
    mx = (a_c + b_c) / 2.0
    return float((sum_c - exp) / (mx - exp)) if mx > exp else 1.0


# ---------------------------------------------------------------- 空间连贯

def _adjacent_pairs(plab: np.ndarray, d: int = 2) -> np.ndarray:
    """近邻斑块对（预注册 §2.4 + 修订留痕：Chebyshev ≤ d 的跨域格对）。

    🔴 勘误（2026-10-11，见预注册修订留痕）：斑块=4 连通域 ⇒ 不同连通域**按定义**
    永不格级相邻 ⇒ 原「4 邻邻接」口径退化（分母 0）。改判：跨域格对距离 Chebyshev ≤ d
    （d=2 主报 / d=3 敏感性）。经度 wrap；行向不 wrap（极行豁免）。
    """
    rows, cols = plab.shape
    pairs = []
    for dr in range(-d, d + 1):
        for dc in range(-d, d + 1):
            if dr == 0 and dc == 0:
                continue
            rolled = np.roll(plab, -dc, axis=1)          # 经度 wrap
            sh = np.full_like(plab, -1)
            r0, r1 = max(0, -dr), min(rows, rows - dr)   # 行向裁剪（不 wrap）
            sh[r0:r1] = rolled[r0 + dr:r1 + dr]
            m = (plab >= 0) & (sh >= 0) & (plab != sh)
            a, b = plab[m], sh[m]
            pairs.append(np.stack([np.minimum(a, b), np.maximum(a, b)], axis=1))
    if not pairs:
        return np.zeros((0, 2), dtype=plab.dtype)
    return np.unique(np.concatenate(pairs, axis=0), axis=0)


def _purity(lab_patch: np.ndarray, pairs: np.ndarray) -> float:
    if pairs.size == 0:
        return float("nan")
    same = lab_patch[pairs[:, 0]] == lab_patch[pairs[:, 1]]
    return float(same.mean())


def _purity_null(lab_patch: np.ndarray, pairs: np.ndarray,
                 draws: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    p = lab_patch.size
    vals = np.empty(draws)
    for i in range(draws):
        perm = rng.permutation(p)
        lp = lab_patch[perm]
        vals[i] = (lp[pairs[:, 0]] == lp[pairs[:, 1]]).mean()
    return float(vals.mean()), float(vals.std()), float(vals.max())


# ---------------------------------------------------------------- PNG（手写）

def _write_png(path: Path, rgb: np.ndarray) -> None:
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
           + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))
    path.write_bytes(png)


_DARK = np.array([10, 12, 18], dtype=np.float64)
_ANCHORS = np.array([[30, 45, 110], [25, 120, 160], [45, 180, 140],
                     [150, 215, 85], [252, 232, 95]], dtype=np.float64)


def _cmap(v01: np.ndarray, bg_mask: np.ndarray) -> np.ndarray:
    v = np.clip(v01, 0.0, 1.0) * (_ANCHORS.shape[0] - 1)
    lo = np.clip(np.floor(v).astype(int), 0, _ANCHORS.shape[0] - 2)
    t = (v - lo)[..., None]
    rgb = _ANCHORS[lo] * (1 - t) + _ANCHORS[lo + 1] * t
    rgb[bg_mask] = _DARK
    return np.clip(rgb, 0, 255).astype(np.uint8)


_LABEL_COLORS = np.array([
    [225, 80, 70], [250, 165, 60], [90, 200, 160],
    [80, 140, 235], [180, 105, 220], [140, 210, 80],
], dtype=np.uint8)


def _label_map(plab: np.ndarray, lab_patch: np.ndarray) -> np.ndarray:
    flat = plab.ravel()
    lab_of_cell = np.where(flat >= 0, lab_patch[np.clip(flat, 0, None)],
                           -1)
    rgb = np.empty((plab.size, 3), dtype=np.uint8)
    bg = lab_of_cell < 0
    rgb[bg] = _DARK.astype(np.uint8)
    ok = ~bg
    rgb[ok] = _LABEL_COLORS[lab_of_cell[ok] % _LABEL_COLORS.shape[0]]
    return rgb.reshape(*plab.shape, 3)


# ---------------------------------------------------------------- 主流程

def _load(data_dir: Path, arm: int, seed: int) -> dict:
    stem = data_dir / f"arm{arm}_s{seed}"
    npz = np.load(str(stem) + ".npz")
    man = json.loads(Path(str(stem) + ".manifest.json").read_text(encoding="utf-8"))
    csv = np.genfromtxt(str(stem) + ".csv", delimiter=",", names=True)
    return {"npz": npz, "man": man, "csv": csv}


def _csv_flatness(csv) -> dict:
    names = [n for n in csv.dtype.names if n not in ("arm", "seed", "tick")]
    out = {}
    for n in names:
        col = np.atleast_1d(csv[n]).astype(np.float64)
        if col.size >= 2:
            out[n] = {"min": float(col.min()), "max": float(col.max()),
                      "flat": bool(np.ptp(col) == 0.0)}
    return out


def main(argv=None) -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="四臂判读（预注册口径）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-dir",
                    default=str(ROOT / "the-world-data" / "bare_world"))
    args = ap.parse_args(argv)

    data_dir = Path(args.data_dir)
    runs = {a: _load(data_dir, a, args.seed) for a in ARMS}
    rows, cols = (runs[1]["man"]["world"]["rows"], runs[1]["man"]["world"]["cols"])
    lat_row = _lat_deg(rows)
    n_cells = rows * cols
    area1 = np.cos(np.radians(np.repeat(lat_row, cols)))

    report: dict = {"seed": args.seed, "rows": rows, "cols": cols, "arms": {}}

    # ---- 几何一致性 + 口径 B 恒等（预注册 §1/§2.1 断言；npz 全为扁平 (n_cells,)）----
    pm_flat = runs[1]["npz"]["patch_mask"].astype(bool)
    pm0 = pm_flat.reshape(rows, cols)
    caps = {}
    for a in ARMS:
        pm = runs[a]["npz"]["patch_mask"].astype(bool)
        if not np.array_equal(pm, pm_flat):
            raise SystemExit(f"[STOP] arm{a} patch_mask 与 arm1 不一致——破几何一致性，停读照报")
    plab, n_patch = _components(pm0)
    for a in ARMS:
        cap = runs[a]["npz"]["capacity"]
        caps[a] = cap
        if not np.array_equal(cap > 0.0, pm_flat):
            raise SystemExit(f"[STOP] arm{a} (cap>0) != patch_mask——口径 B 恒等破例，停读照报")
    report["n_patch_components"] = n_patch
    report["patch_cells"] = int(pm0.sum())

    # ---- CSV 平线检查（预注册 §3.1；非平 = 发现事件，照报不停）----
    report["csv_flatness"] = {a: _csv_flatness(runs[a]["csv"]) for a in ARMS}
    flat_all = all(v["flat"] for a in ARMS
                   for v in report["csv_flatness"][a].values())
    report["csv_all_flat"] = bool(flat_all)

    # ---- C-c 完整性：三快照容量逐位相同（预注册 §2.4）----
    cc = {}
    for a in ARMS:
        keys = sorted((k for k in runs[a]["npz"].files if k.startswith("stock_t")),
                      key=lambda k: int(k.split("_t")[1]))
        snaps = [runs[a]["npz"][k] for k in keys]
        cc[a] = bool(all(np.array_equal(snaps[0], s) for s in snaps[1:])) if len(snaps) >= 2 else None
    report["snapshot_identity"] = cc

    # ---- 供给记账（预注册 §2.5）----
    cap_sum = {a: float(caps[a].sum()) for a in ARMS}
    capw_sum = {a: float((caps[a] * area1).sum()) for a in ARMS}
    report["supply"] = {
        "cap_sum": cap_sum,
        "cap_area_sum": capw_sum,
        "cap_sum_pct_vs_arm1": {a: (cap_sum[a] / cap_sum[1] - 1.0) * 100.0 for a in ARMS},
        "cap_area_pct_vs_arm1": {a: (capw_sum[a] / capw_sum[1] - 1.0) * 100.0 for a in ARMS},
    }

    # ---- 每臂特征/聚类/轮廓/连贯（预注册 §2.2–2.4）----
    pairs_by_d = {d: _adjacent_pairs(plab, d) for d in (2, 3)}
    labels_at_kstar = {}
    feats_all = {}
    for a in ARMS:
        u = None
        if "w" in runs[a]["npz"].files:
            # w_ref 不在 npz 里 ⇒ 用 manifest env_probe.w_mean（= env_field.w_ref，面积权均值）
            wref = runs[a]["man"].get("env_probe", {}).get("w_mean")
            if wref is None:
                raise SystemExit(f"[STOP] arm{a} 缺 env_probe.w_mean，无法算 u")
            u = runs[a]["npz"]["w"] / float(wref)
        feats = _patch_stats(plab, caps[a], u, lat_row)
        feats_all[a] = feats
        X_raw = np.stack([feats[f] for f in
                          ("f1_log_area", "f2_abslat", "f3_cap_mean",
                           "f4_u_mean", "f5_cap_cv")], axis=1)
        mu = X_raw.mean(axis=0)
        sd = X_raw.std(axis=0)
        X = np.where(sd > 1e-12, (X_raw - mu) / np.where(sd > 1e-12, sd, 1.0), 0.0)
        rng = np.random.default_rng(0)
        sil = {}
        labels_by_k = {}
        for k in range(2, 7):
            lab, _ = _kmeans(X, k, rng)
            labels_by_k[k] = lab
            sil[k] = _silhouette(X, lab)
        kstar = max(sil, key=sil.get)
        labels_at_kstar[a] = labels_by_k[kstar]
        sens = {k: _ari(labels_by_k[k], labels_by_k[kstar]) for k in labels_by_k}
        per_d = {}
        for d, pr in pairs_by_d.items():
            if pr.size == 0:
                per_d[d] = {"n_pairs": 0}
                continue
            obs = _purity(labels_by_k[kstar], pr)
            nul_m, nul_s, nul_x = _purity_null(labels_by_k[kstar], pr)
            z = (obs - nul_m) / nul_s if nul_s > 1e-300 else float("inf")
            per_d[d] = {"n_pairs": int(pr.shape[0]), "purity_obs": obs,
                        "null_mean": nul_m, "null_std": nul_s, "null_max": nul_x,
                        "purity_z": z, "purity_z_gt_3": bool(z > 3.0)}
        report["arms"][a] = {
            "arm_name": runs[a]["man"]["arm_name"],
            "silhouette": sil, "kstar": int(kstar),
            "sensitivity_ari": sens,
            "coherence": per_d,
            "cluster_sizes": np.bincount(labels_at_kstar[a]).tolist(),
            "feat_sd": {f: float(s) for f, s in zip(
                ("f1_log_area", "f2_abslat", "f3_cap_mean", "f4_u_mean", "f5_cap_cv"), sd)},
        }

    # ---- 跨臂 ARI 矩阵（预注册 §2.4 C-d 部分）----
    ari_mat = {}
    for a in ARMS:
        for b in ARMS:
            ari_mat[f"{a}-{b}"] = _ari(labels_at_kstar[a], labels_at_kstar[b])
    report["ari_matrix"] = ari_mat

    # ---- 图（预注册 §4）----
    figs = data_dir / "figs"
    figs.mkdir(parents=True, exist_ok=True)
    cap_lo = min(float(caps[a][pm_flat].min()) for a in ARMS)
    cap_hi = max(float(caps[a][pm_flat].max()) for a in ARMS)
    cap_panels = [_cmap(((caps[a] - cap_lo) / max(cap_hi - cap_lo, 1e-300))
                        .reshape(rows, cols), ~pm0)
                  for a in ARMS]
    _write_png(figs / "fig1_capacity_4arms.png", np.concatenate(cap_panels, axis=0))

    lab_panels = [_label_map(plab, labels_at_kstar[a]) for a in ARMS]
    _write_png(figs / "fig4_clusters_4arms.png", np.concatenate(lab_panels, axis=0))

    if "h" in runs[4]["npz"].files:
        e = runs[4]["npz"]
        h = e["h"].reshape(rows, cols)
        w = e["w"].reshape(rows, cols)
        acc = e["acc"].reshape(rows, cols)
        rm = e["river_mask"].reshape(rows, cols).astype(bool)
        h01 = (h - h.min()) / max(float(np.ptp(h)), 1e-300)
        w01 = (w - w.min()) / max(float(np.ptp(w)), 1e-300)
        a01 = np.log10(1.0 + acc)
        a01 = (a01 - a01.min()) / max(float(np.ptp(a01)), 1e-300)
        r01 = np.where(rm, 1.0, 0.0)
        panel = np.concatenate([_cmap(h01, np.zeros_like(pm0)),
                                _cmap(w01, np.zeros_like(pm0)),
                                _cmap(a01, np.zeros_like(pm0)),
                                _cmap(r01, ~rm)], axis=0)
        _write_png(figs / "fig2_arm4_env_fields.png", panel)

    out = data_dir / f"readout_s{args.seed}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- 控制台摘要 ----
    print(f"n_patch={n_patch} cells={report['patch_cells']} csv_all_flat={report['csv_all_flat']}")
    print(f"snapshot_identity: {cc}")
    print("supply:", json.dumps(report["supply"]["cap_area_pct_vs_arm1"], ensure_ascii=False))
    for a in ARMS:
        r = report["arms"][a]
        c2 = r["coherence"].get(2, {})
        c3 = r["coherence"].get(3, {})
        fmt = lambda c: (f"n={c.get('n_pairs')} pur={c.get('purity_obs', float('nan')):.4f} "
                         f"z={c.get('purity_z', float('nan')):.2f}" if c.get("n_pairs") else "n/a")
        print(f"[arm{a} {r['arm_name']}] k*={r['kstar']} sil={r['silhouette']} "
              f"sizes={r['cluster_sizes']}")
        print(f"    C-b d=2: {fmt(c2)} | d=3: {fmt(c3)}")
    print("ARI matrix:")
    for a in ARMS:
        print("  ", " ".join(f"{ari_mat[f'{a}-{b}']:+.3f}" for b in ARMS))
    print(f"-> {out}")
    print(f"-> figs: {figs}")


if __name__ == "__main__":
    main()
