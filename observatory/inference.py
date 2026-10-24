"""observatory.inference：最小统计推断工具（D-7，置换检验 + Bootstrap CI）。

职责
----
- 两组独立样本的中位数差置换检验（permutation test），报告 p 值；
- 中位数差的 Bootstrap 置信区间（CI）；
- 统一入口 compare_groups(a, b) → {median_diff, p_value, ci_low, ci_high, n_a, n_b}。

设计原则
--------
- 最小依赖：仅 numpy，不引入 scipy.stats（环境可能无 scipy）；
- 可复现：固定 random_state 时结果完全一致；
- 保守默认：双侧检验，95% CI，10000 次置换/重采样；
- 空安全：空数组/单元素返回 NaN 并给出警告，不抛异常。

用法
----
    from observatory.inference import compare_groups
    result = compare_groups(treatment, control, n_perm=10000, random_state=42)
    print(f"median_diff={result['median_diff']:.3f}, p={result['p_value']:.4f}, "
          f"CI=[{result['ci_low']:.3f}, {result['ci_high']:.3f}]")
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class InferenceResult:
    """两组比较的推断结果。"""
    median_diff: float          # median(a) - median(b)
    p_value: float              # 双侧置换检验 p 值
    ci_low: float               # Bootstrap 95% CI 下界
    ci_high: float              # Bootstrap 95% CI 上界
    median_a: float             # a 组中位数
    median_b: float             # b 组中位数
    n_a: int                    # a 组样本量
    n_b: int                    # b 组样本量
    n_perm: int                 # 置换次数
    n_boot: int                 # Bootstrap 次数

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        """一行可读摘要。"""
        return (
            f"median_diff={self.median_diff:.4f}, p={self.p_value:.4f}, "
            f"95%CI=[{self.ci_low:.4f}, {self.ci_high:.4f}], "
            f"n_a={self.n_a}, n_b={self.n_b}"
        )


def _median(x: np.ndarray) -> float:
    """安全中位数：空数组返回 NaN。"""
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        return float("nan")
    return float(np.median(x))


def permutation_test(
    a: np.ndarray,
    b: np.ndarray,
    n_perm: int = 10000,
    random_state: int | None = 42,
) -> float:
    """两组中位数差的双侧置换检验。

    原假设 H0：两组来自同一分布（中位数差 = 0）。
    统计量：observed = |median(a) - median(b)|。
    p 值：置换分布中 >= observed 的比例（加 1 修正，避免 p=0）。

    参数
    ----
    a, b : 两组样本（1D array-like）
    n_perm : 置换次数（默认 10000）
    random_state : 随机种子（None=不固定）

    返回
    ----
    p_value : float（0~1）；样本量不足时返回 NaN
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()

    if len(a) < 2 or len(b) < 2:
        return float("nan")

    rng = np.random.default_rng(random_state)
    observed = abs(np.median(a) - np.median(b))
    combined = np.concatenate([a, b])
    n_a = len(a)

    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(combined)
        perm_a = perm[:n_a]
        perm_b = perm[n_a:]
        if abs(np.median(perm_a) - np.median(perm_b)) >= observed:
            count += 1

    # 加 1 修正（包含观测值本身），避免 p=0
    return float((count + 1) / (n_perm + 1))


def bootstrap_ci(
    a: np.ndarray,
    b: np.ndarray,
    n_boot: int = 10000,
    ci: float = 0.95,
    random_state: int | None = 42,
) -> tuple[float, float]:
    """中位数差的 Bootstrap 置信区间（百分位法）。

    参数
    ----
    a, b : 两组样本
    n_boot : Bootstrap 重采样次数（默认 10000）
    ci : 置信水平（默认 0.95）
    random_state : 随机种子

    返回
    ----
    (ci_low, ci_high) : 置信区间上下界；样本量不足时返回 (NaN, NaN)
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()

    if len(a) < 2 or len(b) < 2:
        return (float("nan"), float("nan"))

    rng = np.random.default_rng(random_state)
    diffs = np.empty(n_boot, dtype=np.float64)

    for i in range(n_boot):
        sample_a = rng.choice(a, size=len(a), replace=True)
        sample_b = rng.choice(b, size=len(b), replace=True)
        diffs[i] = np.median(sample_a) - np.median(sample_b)

    alpha = (1 - ci) / 2
    ci_low = float(np.percentile(diffs, alpha * 100))
    ci_high = float(np.percentile(diffs, (1 - alpha) * 100))
    return (ci_low, ci_high)


def compare_groups(
    a: np.ndarray,
    b: np.ndarray,
    n_perm: int = 10000,
    n_boot: int = 10000,
    ci: float = 0.95,
    random_state: int | None = 42,
) -> InferenceResult:
    """两组比较的统一入口：中位数差 + 置换 p 值 + Bootstrap CI。

    参数
    ----
    a : 处理组/实验组样本
    b : 对照组样本
    n_perm : 置换检验次数（默认 10000）
    n_boot : Bootstrap 次数（默认 10000）
    ci : 置信水平（默认 0.95）
    random_state : 随机种子（默认 42，可复现）

    返回
    ----
    InferenceResult : 包含 median_diff / p_value / ci_low / ci_high / 样本量等

    示例
    ----
    >>> import numpy as np
    >>> a = np.random.default_rng(1).normal(0.5, 0.1, 50)
    >>> b = np.random.default_rng(2).normal(0.3, 0.1, 50)
    >>> r = compare_groups(a, b, n_perm=1000, n_boot=1000)
    >>> r.p_value < 0.05
    True
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()

    median_a = _median(a)
    median_b = _median(b)
    median_diff = median_a - median_b if not (np.isnan(median_a) or np.isnan(median_b)) else float("nan")

    p_value = permutation_test(a, b, n_perm=n_perm, random_state=random_state)
    ci_low, ci_high = bootstrap_ci(a, b, n_boot=n_boot, ci=ci, random_state=random_state)

    return InferenceResult(
        median_diff=median_diff,
        p_value=p_value,
        ci_low=ci_low,
        ci_high=ci_high,
        median_a=median_a,
        median_b=median_b,
        n_a=int(len(a)),
        n_b=int(len(b)),
        n_perm=n_perm,
        n_boot=n_boot,
    )
