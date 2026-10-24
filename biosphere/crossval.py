"""biosphere.crossval：行为标签的**无标签交叉验证**（H1 去写死第三件）。

做法（与标签法并列输出，不替换标签法）：
1. **个体轨迹特征**：每个 uid 一条特征向量（平均步长/平均最近邻距/
   各动作率/能量均值与方差/存活时长）；
2. **轨迹聚类**：确定性 KMeans（k 可配；种子化初始化，无 sklearn 依赖）；
3. **互信息**：标签 ↔ 聚类簇、标签 ↔ 各原始特征的 MI（bits）；
4. **口径判定**：某标签人数 > 0 但其与最优特征的 MI ≈ 0 ⇒ 记"口径分歧"
   （标签说有事、原始量说无事，或反之），如实并列，不裁决谁对。

纪律：本模块**不修改**标签定义与阈值；只提供第二把尺子。
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from biosphere.metrics import BEHAVIOR_LABELS, BehaviorLabeler

#: 口径分歧判据：MI 低于此值（bits）记"弱/分歧"
MI_AGREEMENT_MIN: float = 0.05
#: 默认聚类数
DEFAULT_K: int = 5


def uid_features(log, world=None) -> Tuple[List[int], np.ndarray, List[str]]:
    """逐个体轨迹特征（uid 序）。

    特征（8 维）：平均步长、平均最近邻距、move/eat/attack/share/signal 率、
    能量均值（另返回特征名）。
    """
    ticks = sorted({f.tick for f in log.frames})
    t_index = {t: i for i, t in enumerate(ticks)}
    n_t = len(ticks)
    cells: Dict[int, Dict[int, int]] = {}
    nn: Dict[int, Dict[int, float]] = {}
    energy: Dict[int, Dict[int, float]] = {}
    for f in log.frames:
        i = t_index[f.tick]
        for j in range(f.uids.shape[0]):
            uid = int(f.uids[j])
            cells.setdefault(uid, {})[i] = int(f.cells[j])
            nn.setdefault(uid, {})[i] = float(f.nn_dist[j])
            energy.setdefault(uid, {})[i] = float(f.energies[j])
    counts: Dict[int, Dict[str, int]] = {}
    for e in log.events:
        d = counts.setdefault(e.actor_id, {})
        d[e.kind] = d.get(e.kind, 0) + 1
    uids = sorted(cells)
    feats: List[List[float]] = []
    names = ["mean_speed", "mean_nn", "rate_move", "rate_eat", "rate_attack",
             "rate_share", "rate_signal", "mean_energy"]
    for uid in uids:
        ts = sorted(cells[uid])
        steps = 0
        for a, b in zip(ts, ts[1:]):
            if b == a + 1:
                ca, cb = cells[uid][a], cells[uid][b]
                steps += abs(ca // world.cols - cb // world.cols) if world else 0
                if world:
                    dc = abs((ca % world.cols) - (cb % world.cols))
                    dc = min(dc, world.cols - dc)
                    steps += dc
        span = max(1, ts[-1] - ts[0] + 1)
        nn_vals = [v for v in nn[uid].values() if np.isfinite(v)]
        en_vals = list(energy[uid].values())
        c = counts.get(uid, {})
        feats.append([
            steps / span,
            (math.fsum(nn_vals) / len(nn_vals)) if nn_vals else 99.0,
            c.get("move", 0) / span,
            c.get("eat", 0) / span,
            c.get("attack", 0) / span,
            c.get("share", 0) / span,
            c.get("signal", 0) / span,
            (math.fsum(en_vals) / len(en_vals)) if en_vals else 0.0,
        ])
    return uids, np.array(feats, dtype=np.float64), names


def kmeans(X: np.ndarray, k: int = DEFAULT_K, iters: int = 60,
           seed: int = 0) -> np.ndarray:
    """确定性 KMeans（Lloyd；kmeans++ 风格初始化用种子化 rng）。"""
    n = X.shape[0]
    if n == 0:
        return np.array([], dtype=np.int64)
    k = min(k, n)
    rng = np.random.default_rng(seed)
    # 标准化（防能量均值压倒其它维）
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-12] = 1.0
    Z = (X - mu) / sd
    # kmeans++ 初始化
    centers = [Z[int(rng.integers(n))]]
    for _ in range(1, k):
        d2 = np.min(
            [np.sum((Z - c) ** 2, axis=1) for c in centers], axis=0)
        total = float(d2.sum())
        probs = d2 / total if total > 0 else np.full(n, 1.0 / n)
        centers.append(Z[int(rng.choice(n, p=probs))])
    C = np.array(centers)
    labels = np.zeros(n, dtype=np.int64)
    for _ in range(iters):
        dist = np.stack([np.sum((Z - c) ** 2, axis=1) for c in C])
        new = np.argmin(dist, axis=0)
        if np.array_equal(new, labels):
            break
        labels = new
        for j in range(k):
            members = Z[labels == j]
            if members.shape[0]:
                C[j] = members.mean(axis=0)
    return labels


def mutual_info(x: Sequence[int], y: Sequence[int]) -> float:
    """离散互信息（bits；确定性）。"""
    n = len(x)
    if n == 0:
        return 0.0
    px: Dict[int, int] = {}
    py: Dict[int, int] = {}
    pxy: Dict[Tuple[int, int], int] = {}
    for a, b in zip(x, y):
        px[a] = px.get(a, 0) + 1
        py[b] = py.get(b, 0) + 1
        pxy[(a, b)] = pxy.get((a, b), 0) + 1
    mi = 0.0
    for (a, b), c in pxy.items():
        p_ab = c / n
        mi += p_ab * math.log2(p_ab / ((px[a] / n) * (py[b] / n)))
    return mi


def label_free_crossval(log, world, thresholds=None,
                        k: int = DEFAULT_K) -> Dict:
    """无标签交叉验证：聚类簇 vs 行为标签 vs 原始特征。

    返回每标签：人数、与聚类簇的 MI、与最相关特征的 MI、口径判定。
    """
    uids, X, names = uid_features(log, world)
    if not uids:
        return {"enabled": False, "n_individuals": 0, "labels": {},
                "note": "无个体"}
    clusters = kmeans(X, k=k, seed=0)
    labeler = BehaviorLabeler(log, thresholds)
    labels_all = labeler.label_all(world)
    out: Dict[str, Dict] = {}
    for lab in BEHAVIOR_LABELS:
        y = [1 if labels_all.get(u, {}).get(lab) else 0 for u in uids]
        n_on = sum(y)
        mi_cluster = mutual_info(clusters.tolist(), y)
        mi_feats = {names[j]: mutual_info(X[:, j].astype(int).tolist(), y)
                    for j in range(X.shape[1])}
        best_feat = max(mi_feats, key=mi_feats.get) if mi_feats else ""
        best_mi = mi_feats[best_feat] if best_feat else 0.0
        if n_on == 0:
            verdict = "无该标签（不判）"
        elif best_mi >= MI_AGREEMENT_MIN or mi_cluster >= MI_AGREEMENT_MIN:
            verdict = "一致（原始量支持标签）"
        else:
            verdict = "口径分歧（标签计数>0 但原始量 MI≈0）"
        out[lab] = {
            "n_labeled": n_on,
            "mi_with_clusters": mi_cluster,
            "mi_best_feature": best_mi,
            "mi_best_feature_name": best_feat,
            "mi_features": mi_feats,
            "verdict": verdict,
        }
    return {
        "enabled": True,
        "n_individuals": len(uids),
        "k": k,
        "mi_agreement_min": MI_AGREEMENT_MIN,
        "labels": out,
        "note": "无标签第二把尺子；与标签法并列，不替换、不裁决",
    }


__all__ = [
    "MI_AGREEMENT_MIN", "DEFAULT_K", "uid_features", "kmeans",
    "mutual_info", "label_free_crossval",
]
