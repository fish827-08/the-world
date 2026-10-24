"""biosphere.criteria：W4b 预注册判据与生态门（纯函数，可单测）。
**预注册（B12 派工，跑批前写定）**
---------------------------------
生态门（先过门再判读）：

- 每 seed 过门 = 终局存活 > 0 且全程无连续 500 tick 的 0 存活空窗；
- 一臂过门 = 过门 seed ≥ 多数（≥3/5）；
- **不过门 ⇒ 该臂写"生态门未过、不可判"，不做任何机制结论。**

组选择判据（"多细胞是选择出来的"）：

- 比较臂 = `关组选择`（Z2）vs `full`，逐 seed 配对；
- 两个读数：**粘附链团比例**（时间平均，多体链团个体占比）与
  **粘附基因频率**（全程均值，原始基因槽位）；
- 方向 = full 高于 Z2；
- **要求 ≥4/5 seed 两项都方向一致**（不用 p 值表演）⇒ 才算"已建立"；
  否则如实写"未建立"。
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

#: 生态门：多数定义（过门 seed ≥ ceil(n/2)+? 用显式比例）
GATE_MAJORITY_FRACTION: float = 0.5
#: 生态门：允许的最大连续零存活窗口（tick）
GATE_EMPTY_WINDOW: int = 500


def _longest_empty_window(alive_series: Sequence[int]) -> int:
    best = cur = 0
    for n in alive_series:
        if n <= 0:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def seed_gate(alive_series: Sequence[int],
              empty_window: int = GATE_EMPTY_WINDOW) -> Dict:
    """单 seed 生态门：终局存活 > 0 且无超限零存活窗口。"""
    final = int(alive_series[-1]) if len(alive_series) else 0
    longest = _longest_empty_window(alive_series)
    return {"final_alive": final, "longest_empty_window": longest,
            "passed": bool(final > 0 and longest < empty_window)}


def evaluate_ecological_gate(per_seed: Dict[int, Sequence[int]],
                             empty_window: int = GATE_EMPTY_WINDOW
                             ) -> Dict:
    """一臂的生态门：多数 seed 过门才判读。"""
    gates = {seed: seed_gate(series, empty_window)
             for seed, series in per_seed.items()}
    n_pass = sum(1 for g in gates.values() if g["passed"])
    n_total = len(gates)
    majority = n_total > 0 and n_pass / n_total > GATE_MAJORITY_FRACTION
    return {"n_total": n_total, "n_passed": n_pass, "majority_passed":
            majority, "per_seed": gates,
            "verdict": "过门，可判读" if majority else "生态门未过、不可判"}


def evaluate_group_selection_criterion(full_rows: Iterable[Dict],
                                        z2_rows: Iterable[Dict],
                                        min_consistent: int = 4) -> Dict:
    """预注册判据：full 的粘附链团比例与粘附基因频率须**高于**关组选择臂。

    Args:
        full_rows: full 臂逐 seed 行（须含 ``seed``、``chain_ratio_mean``、
            ``adhesion_gene_mean``）。
        z2_rows: 关组选择臂同构行。
        min_consistent: 方向一致的 seed 数下限（派工口径 ≥4/5）。
    """
    full = {int(r["seed"]): r for r in full_rows}
    z2 = {int(r["seed"]): r for r in z2_rows}
    seeds = sorted(set(full) & set(z2))
    chain_ok, adh_ok, both_ok = [], [], []
    detail = []
    for s in seeds:
        f, g = full[s], z2[s]
        c_ok = float(f["chain_ratio_mean"]) > float(g["chain_ratio_mean"])
        a_ok = float(f["adhesion_gene_mean"]) > float(g["adhesion_gene_mean"])
        chain_ok.append(c_ok)
        adh_ok.append(a_ok)
        both_ok.append(bool(c_ok and a_ok))
        detail.append({
            "seed": s,
            "full_chain_ratio": float(f["chain_ratio_mean"]),
            "z2_chain_ratio": float(g["chain_ratio_mean"]),
            "full_adhesion_gene": float(f["adhesion_gene_mean"]),
            "z2_adhesion_gene": float(g["adhesion_gene_mean"]),
            "chain_direction_ok": c_ok, "adhesion_direction_ok": a_ok,
        })
    n = len(seeds)
    n_chain = sum(chain_ok)
    n_adh = sum(adh_ok)
    n_both = sum(both_ok)
    established = (n > 0 and n_both >= min_consistent
                   and n_chain >= min_consistent and n_adh >= min_consistent)
    return {
        "n_seeds": n,
        "min_consistent_required": min_consistent,
        "n_chain_direction_ok": n_chain,
        "n_adhesion_direction_ok": n_adh,
        "n_both_direction_ok": n_both,
        "established": established,
        "verdict": "已建立" if established else "未建立",
        "per_seed": detail,
    }


__all__ = [
    "GATE_MAJORITY_FRACTION", "GATE_EMPTY_WINDOW",
    "seed_gate", "evaluate_ecological_gate",
    "evaluate_group_selection_criterion",
]

# ── B15.3 谱系条件判据（预注册 §13.15） ──────────────────────────────

def lineage_root_map(births: List[Tuple[int, int, int]]) -> Dict[int, int]:
    """uid → 谱系 root（沿出生流水回溯到 parent<0 的 founder；founder 自身）。"""
    # parent<0 = 无父母（founder/放置者）⇒ 自身即 root；只收录有父母的边
    parent_of = {c: p for p, c, _cell in births if p >= 0}
    roots: Dict[int, int] = {}

    def _root(uid: int) -> int:
        seen = []
        cur = uid
        while cur in parent_of and cur not in roots:
            seen.append(cur)
            cur = parent_of[cur]
        r = roots.get(cur, cur)
        for u in seen:
            roots[u] = r
        return r

    for uid in list(parent_of):
        _root(uid)
    return roots


def badge_keff(hist: Dict[int, int]) -> float:
    """有效徽章多样性 K_eff = exp( Shannon 熵 )（1 = 单斑化；K = 均匀）。"""
    total = sum(hist.values())
    if total <= 0:
        return 0.0
    ent = 0.0
    for n in hist.values():
        if n > 0:
            p = n / total
            ent -= p * math.log(p)
    return math.exp(ent)


def _mi_counts(xs: List[int], ys: List[int]) -> float:
    """互信息（bits；小样本；两端离散）。"""
    n = len(xs)
    if n == 0:
        return 0.0
    joint: Dict[Tuple[int, int], int] = {}
    cx: Dict[int, int] = {}
    cy: Dict[int, int] = {}
    for x, y in zip(xs, ys):
        joint[(x, y)] = joint.get((x, y), 0) + 1
        cx[x] = cx.get(x, 0) + 1
        cy[y] = cy.get(y, 0) + 1
    mi = 0.0
    for (x, y), c in joint.items():
        pxy = c / n
        mi += pxy * math.log2(pxy / ((cx[x] / n) * (cy[y] / n)))
    return mi


def lineage_conditional_mi(records: List[Dict]) -> Dict:
    """谱系条件 MI：按 root 分层，层内 MI(badge, strength 四分位)。

    返回 {"mi_weighted": 加权均值, "per_root": {root: mi}, "n_roots": k}。
    权重 = 层内样本数（大谱系说话响）。这抵消"全局 MI 被谱系间差异
    （漂变/奠基者效应）灌水"——B14 判据①失效的根因。
    """
    by_root: Dict[int, List[Tuple[int, int]]] = {}
    for r in records:
        by_root.setdefault(r["root"], []).append((r["badge"], r["sbin"]))
    per_root: Dict[int, float] = {}
    total_w = 0
    acc = 0.0
    for root, pairs in by_root.items():
        if len(pairs) < 4:            # 层内样本不足 ⇒ 不可估
            continue
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        mi = _mi_counts(xs, ys)
        per_root[root] = mi
        acc += mi * len(pairs)
        total_w += len(pairs)
    return {
        "mi_weighted": (acc / total_w) if total_w else 0.0,
        "per_root": per_root,
        "n_roots_estimated": len(per_root),
    }


def lineage_label_permutation(records: List[Dict], n_perm: int = 2000,
                              seed: int = 11) -> Dict:
    """谱系内徽章标签置换检验（B15.3 判据①新式）。

    在每个谱系 root 内打乱 badge 标签（strength 与 root 不动）⇒ 重算
    加权谱系内 MI ⇒ 置换分布。观测 MI 的 z 与单侧 p（大 = 正向关联）。
    负对照同式：reshuffle 臂的徽章本就在谱系内随机 ⇒ p 应 ≈ 0.5 附近。
    """
    rng = np.random.default_rng(seed)
    by_root: Dict[int, List[Tuple[int, int]]] = {}
    for r in records:
        by_root.setdefault(r["root"], []).append((r["badge"], r["sbin"]))
    usable = {k: v for k, v in by_root.items() if len(v) >= 4}
    if not usable:
        return {"observed": 0.0, "z": 0.0, "p_one_sided": 1.0, "n_perm": 0}

    def _weighted_mi(assign: Dict[int, List[int]]) -> float:
        acc = 0.0
        w = 0
        for root, pairs in usable.items():
            bs = assign[root]
            xs = list(bs)
            ys = [p[1] for p in pairs]
            mi = _mi_counts(xs, ys)
            acc += mi * len(pairs)
            w += len(pairs)
        return acc / w

    base = {root: [p[0] for p in pairs] for root, pairs in usable.items()}
    observed = _weighted_mi(base)
    null = []
    for _ in range(n_perm):
        permuted = {}
        for root, pairs in usable.items():
            labels = list(base[root])
            rng.shuffle(labels)
            permuted[root] = labels
        null.append(_weighted_mi(permuted))
    arr = np.asarray(null, dtype=np.float64)
    sd = float(arr.std())
    z = ((observed - float(arr.mean())) / sd) if sd > 1e-12 else 0.0
    p = float((np.sum(arr >= observed) + 1) / (n_perm + 1))
    return {
        "observed": observed,
        "null_mean": float(arr.mean()),
        "null_sd": sd,
        "z": z,
        "p_one_sided": p,
        "n_perm": n_perm,
    }


def individual_records(sim) -> List[Dict]:
    """从（恢复的）模拟器终局种群提个体级判据记录。

    badge/strength 走 develop；root 沿 log.births 回溯；sbin = strength
    四分位（0..3，全种群分位 ⇒ 跨臂可比）。
    """
    from biosphere.develop import develop
    inds = sim.population.individuals
    roots = lineage_root_map(list(sim.log.births))
    strengths = [develop(ind.genome).strength for ind in inds]
    if strengths:
        qs = np.quantile(strengths, [0.25, 0.5, 0.75])
    else:
        qs = np.array([0.0, 0.0, 0.0])
    out: List[Dict] = []
    for ind, s in zip(inds, strengths):
        c = develop(ind.genome)
        sbin = int(np.searchsorted(qs, s, side="right"))
        out.append({
            "uid": ind.uid,
            "badge": int(c.badge),
            "strength": float(s),
            "sbin": sbin,
            "root": roots.get(ind.uid, ind.uid),
            "parent": ind.parent,
        })
    return out
