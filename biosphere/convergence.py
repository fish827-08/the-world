"""biosphere.convergence：B19b 观测 K 收敛判定（纯函数）。

预注册：``docs/预注册/预注册-B19b-K收敛-20261021.md`` §二/§三/§四。

口径（写死，不得在判读后改）::

    W = 500, eps = 0.10, N = 3, t0 = 500, T_max = 6000

- ``P[s]`` = 第 ``s`` 步后的存活数（``s=1..T_max``），``P[0]`` = 初始。
- ``M(s) = mean(P[s-W+1 .. s])``；``r(s) = |M(s) - M(s-W)| / M(s-W)``。
- 评估点 ``s = t0+W, t0+2W, ..., T_max``。
- 从 ``T_max`` 向前的最大合格后缀长度 ≥ N ⇒ 已收敛；``T_K`` = 后缀最早评估点。
- ``K = mean(P[T_max-W+1 .. T_max])``，只对已收敛 seed 报。

本模块**不读 RNG、不读文件**；输入为 ``P[0..T_max]``，输出可逐项由序列重算。
"""
from __future__ import annotations

import math
import statistics
from typing import Dict, List, Mapping, Optional, Sequence

#: B19b 五个判据数（预注册 §三）
K_W: int = 500
K_EPS: float = 0.10
K_N: int = 3
K_T0: int = 500
K_TMAX: int = 6000

STATUS_CONVERGED = "converged"
STATUS_NOT_CONVERGED = "not_converged"
STATUS_EXTINCT = "extinct"

CONCLUSION_GLOBAL = "global_convergence"
CONCLUSION_MAJORITY = "majority_convergence_not_global"
CONCLUSION_NOT_GLOBAL = "not_global_convergence"


def _window_mean(P: Sequence[int], end: int, W: int) -> float:
    """``mean(P[end-W+1 .. end])``（闭区间，end ≥ W）。"""
    start = end - W + 1
    return math.fsum(float(P[i]) for i in range(start, end + 1)) / float(W)


def _window_values(P: Sequence[int], end: int, W: int) -> List[float]:
    start = end - W + 1
    return [float(P[i]) for i in range(start, end + 1)]


def _verdict_params(W: int, eps: float, N: int, t0: int, t_max: int) -> Dict:
    return {
        "W": int(W), "eps": float(eps), "N": int(N),
        "t0": int(t0), "t_max": int(t_max),
    }


def k_convergence_verdict(
    P: Sequence[int],
    W: int = K_W,
    eps: float = K_EPS,
    N: int = K_N,
    t0: int = K_T0,
    t_max: int = K_TMAX,
) -> Dict:
    """单个生存序列的 B19b 收敛判定（预注册 §3.2 逐字实现）。

    Args:
        P: ``P[0..t_max]``；``P[0]`` 为初始种群，``P[s]`` 为第 s 步后存活数。
        W/eps/N/t0/t_max: 预注册五个判据数（默认 = B19b 固定值）。

    Returns:
        dict：``status`` / ``T_K`` / ``K`` / ``final_window_mean`` / ``std`` /
        ``rel_std`` / ``r_values`` / ``suffix_len`` / ``eval_points`` /
        ``extinct_at`` / ``edge_confirmation`` 等。

    Raises:
        ValueError: 序列长度不足、参数非法、或评估点未对齐 ``T_max``。
    """
    if W <= 0:
        raise ValueError(f"W 必须 > 0（收到 {W}）")
    if not (0.0 < eps < 1.0):
        raise ValueError(f"eps 必须在 (0,1)（收到 {eps}）")
    if N < 1:
        raise ValueError(f"N 必须 ≥ 1（收到 {N}）")
    if t0 < W:
        raise ValueError(f"t0 必须 ≥ W（t0={t0}, W={W}）")
    if t_max < t0 + W:
        raise ValueError(f"t_max 必须 ≥ t0+W（t_max={t_max}）")
    if (t_max - t0) % W != 0:
        raise ValueError(
            f"评估点必须对齐 T_max：(t_max-t0)={t_max - t0} 不是 W={W} 的整数倍")
    if len(P) < t_max + 1:
        raise ValueError(
            f"P 长度 {len(P)} < t_max+1={t_max + 1}（需要 P[0..t_max]）")

    params = _verdict_params(W, eps, N, t0, t_max)
    eval_points = list(range(t0 + W, t_max + 1, W))
    final_values = _window_values(P, t_max, W)
    final_mean = math.fsum(final_values) / float(W)
    final_std = statistics.stdev(final_values) if W >= 2 else 0.0
    rel_std = (final_std / final_mean) if final_mean > 0.0 else None

    base = {
        "params": params,
        "eval_points": [int(s) for s in eval_points],
        "final_window_mean": float(final_mean),
        "final_window_std": float(final_std),
        "final_window_rel_std": rel_std,
        "series_len": int(len(P)),
    }

    zero = next((s for s in range(1, t_max + 1) if int(P[s]) == 0), None)
    if zero is not None:
        # 灭绝：不计算 r/K，只给灭绝前最后一个完整窗口均值（若存在）。
        pre_mean = None
        if zero - 1 >= 1:
            end = zero - 1
            start = max(1, end - W + 1)
            n = end - start + 1
            pre_mean = math.fsum(float(P[i]) for i in range(start, end + 1)) / n
        out = dict(base)
        out.update({
            "status": STATUS_EXTINCT,
            "T_K": None,
            "K": None,
            "std": None,
            "rel_std": None,
            "r_values": None,
            "suffix_len": 0,
            "extinct_at": int(zero),
            "pre_extinction_window_mean": pre_mean,
            "edge_confirmation": False,
        })
        return out

    # 评估点的 M(s) 与 M(s-W)；分母为 0 按灭绝处理（非负计数下等价于窗口
    # 全零，上面的灭绝扫描已覆盖；此分支只作防御）。
    m_cache: Dict[int, float] = {}

    def _m(end: int) -> float:
        if end not in m_cache:
            m_cache[end] = _window_mean(P, end, W)
        return m_cache[end]

    r_values: List[float] = []
    for s in eval_points:
        prev = _m(s - W)
        if prev <= 0.0:
            out = dict(base)
            out.update({
                "status": STATUS_EXTINCT,
                "T_K": None, "K": None, "std": None, "rel_std": None,
                "r_values": None, "suffix_len": 0,
                "extinct_at": None,
                "pre_extinction_window_mean": None,
                "edge_confirmation": False,
            })
            return out
        r_values.append(abs(_m(s) - prev) / prev)

    suffix_len = 0
    for r in reversed(r_values):
        if r < eps:
            suffix_len += 1
        else:
            break

    converged = suffix_len >= N
    T_K = int(eval_points[-suffix_len]) if converged else None
    edge = bool(converged and suffix_len == N)

    out = dict(base)
    out.update({
        "status": STATUS_CONVERGED if converged else STATUS_NOT_CONVERGED,
        "T_K": T_K,
        "K": float(final_mean) if converged else None,
        "std": float(final_std) if converged else None,
        "rel_std": rel_std if converged else None,
        "r_values": [float(r) for r in r_values],
        "suffix_len": int(suffix_len),
        "extinct_at": None,
        "pre_extinction_window_mean": None,
        "edge_confirmation": edge,
    })
    return out


def batch_k_convergence_verdict(
    series_by_seed: Mapping[int, Sequence[int]],
    W: int = K_W,
    eps: float = K_EPS,
    N: int = K_N,
    t0: int = K_T0,
    t_max: int = K_TMAX,
) -> Dict:
    """整批判定（预注册 §4.2）：逐 seed 判定 + 跨 seed K 汇总。

    结论只由**已收敛 seed 数**决定：
    ``n_total`` / ``n_total-1`` / 其余 ⇒ 全局 / 多数（非全局）/ 未全局。
    任一种子灭绝都会使整批不可能为“全局收敛”（灭绝不计入已收敛）。
    """
    per_seed: Dict[int, Dict] = {}
    for seed in sorted(int(s) for s in series_by_seed):
        per_seed[seed] = k_convergence_verdict(
            series_by_seed[seed], W=W, eps=eps, N=N, t0=t0, t_max=t_max)

    seeds = sorted(per_seed)
    n_total = len(seeds)
    converged_seeds = [s for s in seeds
                      if per_seed[s]["status"] == STATUS_CONVERGED]
    not_converged_seeds = [s for s in seeds
                          if per_seed[s]["status"] == STATUS_NOT_CONVERGED]
    extinct_seeds = [s for s in seeds
                     if per_seed[s]["status"] == STATUS_EXTINCT]
    n_converged = len(converged_seeds)

    if n_total > 0 and n_converged == n_total:
        conclusion = CONCLUSION_GLOBAL
    elif n_total > 0 and n_converged == n_total - 1:
        conclusion = CONCLUSION_MAJORITY
    else:
        conclusion = CONCLUSION_NOT_GLOBAL

    ks = [per_seed[s]["K"] for s in converged_seeds]
    cross = None
    if ks:
        cross = {
            "n": len(ks),
            "seeds": converged_seeds,
            "min": float(min(ks)),
            "median": float(statistics.median(ks)),
            "max": float(max(ks)),
        }

    return {
        "params": _verdict_params(W, eps, N, t0, t_max),
        "per_seed": per_seed,
        "n_total": int(n_total),
        "n_converged": int(n_converged),
        "n_not_converged": len(not_converged_seeds),
        "n_extinct": len(extinct_seeds),
        "converged_seeds": converged_seeds,
        "not_converged_seeds": not_converged_seeds,
        "extinct_seeds": extinct_seeds,
        "edge_confirmation_seeds": [s for s in converged_seeds
                                    if per_seed[s]["edge_confirmation"]],
        "conclusion": conclusion,
        "K_cross_seed": cross,
    }


__all__ = [
    "K_W", "K_EPS", "K_N", "K_T0", "K_TMAX",
    "STATUS_CONVERGED", "STATUS_NOT_CONVERGED", "STATUS_EXTINCT",
    "CONCLUSION_GLOBAL", "CONCLUSION_MAJORITY", "CONCLUSION_NOT_GLOBAL",
    "k_convergence_verdict", "batch_k_convergence_verdict",
]
