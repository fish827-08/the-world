"""`s0_kernel.arms` —— 双臂 S0 的**获取函数层**（S0-kernel-b 机制面）。

归属与边界
----------
- A 臂＝现公式（旧纪元账面形）最小移植；B 臂＝**镜 b5a7526 定形**（R383 设计权，逐字照抄）：
  `W(g,n) = W₀(g) + α_W·f(g,n)`，其中 `W₀(g)=p(g)·s₀(g)·E_prey·r`（整块去 gate、无两池、无素食/成本腿）、
  `f(g,n) = −p(g)·s₀(g)·E_prey·r·ν(g,n)`（**A1 线性折扣**，λ 吸进 α_W），
  `s₀·(1−α_W·ν) ≥ 0` clamp 保物理性；A2/A3 由镜弃于 S0（归 S1 稳健性扫描）⇒ 本模块**只认 A1**，
  传别的形状即 raise（不替设计做主，也不给静默默认）。
- 判读面（谷深 D／峰位／出不出）**不在本模块**（R225/R394④ 禁自实现判据，聚合走冻结判据判读脚本）。
- 零行为面：本包**不被 `simulation/` 导入**（守卫见 `tests/test_s0_kernel.py`）⇒ 现役引擎逐位等价门不受影响。
- 不消费 RNG：全模块纯函数，同输入逐位同输出（`tests/test_s0_arms.py` 有 AST 级 import 守卫）。
- 参数单一来源：八个捕食系数一律从 `PredationConfig()` 读，**禁裸写**（AGENTS.md 关键约束；
  变异靶 M-D/M-E 证明"改 config 而 W 不动"会被测出来）。

对账锚（E-034／R197，`docs/实验记录.md:54` ＋ `docs/设计文档/证伪-双峰在现行捕食公式下不可能-20260924.md` §3.4）
-----------------------------------------------------------------------------------------------
k=2 默认档五读数与 cost×300 的谷读数按 6 位小数逐位钉死（`tests/test_s0_arms.py`）。
🔴 **边界歧义实测**（本模块发现，已上板）：原脚本 mask 用严格 `g > gate`，而 `g = gate` 恰是格点
⇒ 文档里「`g=.3 → 1.0720`」实际是**右极限**（`0.3+1e-9` 得 1.0720，`0.3` 得 0.49）。
⇒ `W` 在 `g=gate` 有跳跃间断，任何拿 `W(gate)` 对账的脚本都会**静默错一位**；本模块用
`gate_inclusive` 显式暴露该约定，默认 `False`（＝引擎字面语义 `:4057`/`:4129`）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from simulation.config import PredationConfig

#: 预注册红线（草案 §五：「两臂皆否 ⇒ S0 尺度不可裁决」必须在跑之前写进代码，不是事后解释）。
INCONCLUSIVE_EXIT = "两臂皆否 ⇒ S0 尺度不可裁决（此出口在跑批前已声明，不得事后追加）"


def declare_inconclusive_exit() -> str:
    """返回预注册不可裁决出口的原文（供 manifest／判读脚本引用，禁止改写）。"""
    return INCONCLUSIVE_EXIT


#: 镜 b5a7526：S0 主判只走 A1；A2（污染差分）／A3（Red Queen 崩盘风险）留 S1 ⇒ 不接受，也不静默替代。
F_FORMS = ("A1",)
#: 竞争核宽度：**待标定批锁**（镜 §四-2：单值进 manifest，不作 S0 扫描维度）。
SIGMA_C_PROVISIONAL = 0.1


@dataclass(frozen=True)
class Landscape:
    """E-034 解析档的**归一化选择**（不是引擎配置，勿当 config 默认读）。

    真源 = `experiments/analyze_bimodal_fitness.py:find_valley` 默认参数
    （`f_veg=1.0, e_self=150.0, e_prey=150.0, hunger=0.5`）⇒ 对账锚必须用同一组值。
    """

    f_veg: float = 1.0            # 素食端基准收益 F（只进 A 臂；B 臂按镜形不含素食腿）
    e_self: float = 150.0         # 自身能量 E（"E 固定"近似＝无频率依赖的根因位）
    e_prey: float = 150.0         # 猎物能量 E_prey
    hunger: float = 0.5           # 饥饿度 h ∈ [0,1]
    sigma_c: float = SIGMA_C_PROVISIONAL   # 竞争核 σ_C（待标定批锁）
    n_ref: float = 100.0          # `nu_proxy` 的密度基准（纯函数层 S-B 核查用替身）

    def __post_init__(self) -> None:
        if not 0.0 <= self.hunger <= 1.0:
            raise ValueError(f"hunger 必须 ∈ [0,1]，实得 {self.hunger}")
        if self.e_self < 0.0 or self.e_prey < 0.0 or self.sigma_c <= 0.0 or self.n_ref <= 0.0:
            raise ValueError("e_self/e_prey 非负，sigma_c/n_ref 必须 > 0")


def default_predation() -> PredationConfig:
    """捕食系数唯一来源（禁裸写 0.2/0.3/0.5/0.1/0.9/0.4/0.1/0.0）。"""
    return PredationConfig()


def _check_g(g: float) -> None:
    if not 0.0 <= g <= 1.0:
        raise ValueError(f"g 必须在有界原始轴 [0,1]（v1.1.3 读数轴禁分位/秩变换），实得 {g}")


# ------------------------------------------------------------------ 共用构件（与引擎同式）

def attack_p(g: float, pred: PredationConfig, land: Landscape) -> float:
    """出手概率 `p(g) = g · attack_prob_coef · hunger`（引擎 `:4699` 同形）。"""
    return g * pred.attack_prob_coef * land.hunger


def success_s(g: float, pred: PredationConfig, land: Landscape) -> float:
    """成功率 `s(g) = clip(E/(E+E_prey)·(0.5+g·γ), floor, ceil)`（引擎 `:4821-4826` 同形＝镜的 `s₀`）。"""
    ratio = land.e_self / max(land.e_self + land.e_prey, 1e-9)
    return min(max(ratio * (0.5 + g * pred.success_gene_gain),
                   pred.success_floor), pred.success_ceil)


def _forage(g: float, k: float, land: Landscape) -> float:
    """素食端 `(1-g)^k · F`（引擎 `forage_tradeoff_k` 消费点 `:3602`；**只进 A 臂**）。"""
    return (1.0 - g) ** k * land.f_veg


def _pred_net(g: float, pred: PredationConfig, land: Landscape, cost_scale: float) -> float:
    """净捕食收益 `p·s·E_prey·r − p·c`（掠得腿取**账面** `transfer_ratio × E_prey`）。"""
    p = attack_p(g, pred, land)
    gain = p * success_s(g, pred, land) * land.e_prey * pred.transfer_ratio
    return gain - p * pred.attack_cost * cost_scale


# ------------------------------------------------------------------ A 臂：现公式（旧纪元）

def w_a(g: float, *, k: float | None = None, cost_scale: float = 1.0,
        pred: PredationConfig | None = None, land: Landscape | None = None,
        gate_inclusive: bool = False) -> float:
    """A 臂 `W(g) = (1-g)^k·F + 𝟙[g>gate]·[p·s·E_prey·r − p·c]`（R8.1 证伪文档口径）。

    🔴 纪元条：掠得腿是**旧纪元账面转移**，不含步B 的补扣减、也不含"整份转移"会签形态
    （混进来＝同时污 A/B 差分与跨纪元比，砚 B1／R320）。
    `k=None` ⇒ 取引擎默认 `forage_tradeoff_k`（S0 档 = 0.0，H2 落笔 #10 定参）。
    """
    pred = pred or default_predation()
    land = land or Landscape()
    kk = pred.forage_tradeoff_k if k is None else k
    _check_g(g)
    if cost_scale < 0.0:
        raise ValueError(f"cost_scale 非负，实得 {cost_scale}")
    attack = g > pred.attack_gene_gate
    if gate_inclusive and g == pred.attack_gene_gate:
        attack = True
    return _forage(g, kk, land) + (_pred_net(g, pred, land, cost_scale) if attack else 0.0)


# ------------------------------------------------------------------ B 臂：镜 b5a7526 定形

def w_b0(g: float, *, pred: PredationConfig | None = None,
         land: Landscape | None = None) -> float:
    """B 臂 `W₀(g) = p(g)·s₀(g)·E_prey·r`：**整块去 gate、无两池、无 sigmoid**（镜 §一/§三）。

    镜的去 gate 三条理由里工程上最硬的一条：`α_W=0` 支要成为**单变量零对照**，前提是 gate 也一起拿掉
    （sigmoid 会在 α_W=0 时仍留软 gate ⇒ 分岔无法归因到 `W_mr`）。素食腿与成本腿按镜原文式子**不入 B 臂**。
    """
    pred = pred or default_predation()
    land = land or Landscape()
    _check_g(g)
    return attack_p(g, pred, land) * success_s(g, pred, land) * land.e_prey * pred.transfer_ratio


def nu_local(g_self: float, g_neighbors, sigma_c: float | None = None) -> float:
    """`ν(g,n)` ＝**同类表型近邻竞争密度**（镜 §一：竞争核 σ_C 加权邻域内同表型密度）。

    `ν = mean_j exp(−(g_self−g_j)²/(2σ_C²))` ∈ [0,1]，**局部口径**——不是全场总数、也不是
    gate-defined attacker 占比（gate 已去，`f_hi` 无定义）。`sigma_c=None` ⇒ 取 `Landscape().sigma_c`
    （= 待标定批锁的占位单值，manifest 必须回显实际用值）。
    """
    _check_g(g_self)
    sc = Landscape().sigma_c if sigma_c is None else float(sigma_c)
    if sc <= 0.0:
        raise ValueError(f"sigma_c 必须 > 0，实得 {sc}")
    nb = [float(x) for x in g_neighbors]
    if not nb:
        return 0.0
    acc = 0.0
    for gj in nb:
        d = (g_self - gj) / sc
        acc += math.exp(-0.5 * d * d)
    return min(max(acc / len(nb), 0.0), 1.0)


def nu_proxy(n: float, *, n_ref: float | None = None) -> float:
    """纯函数层的**密度替身** `ν = min(n/n_ref, 1)`：只给 S-B 数值核查当单调映射用。

    🔴 正式批读数的 ν 走 `nu_local`（镜定义）；本替身不得进 harness 出货路径（`run_arm` 会显式区分）。
    """
    if n < 0.0:
        raise ValueError(f"n（近邻竞争者数）非负，实得 {n}")
    ref = Landscape().n_ref if n_ref is None else float(n_ref)
    if ref <= 0.0:
        raise ValueError(f"n_ref 必须 > 0，实得 {ref}")
    return min(n / ref, 1.0)


def f_a1(g: float, nu: float, *, pred: PredationConfig | None = None,
         land: Landscape | None = None) -> float:
    """A1 线性折扣形状 `f(g,n) = −p·s₀·E_prey·r·ν`（镜：与加法形 `W₀+α_W·f` 1:1 同构）。"""
    if not 0.0 <= nu <= 1.0:
        raise ValueError(f"ν 是 [0,1] 的密度比例，实得 {nu}")
    return -w_b0(g, pred=pred, land=land) * nu


def f_gn(g: float, nu: float, form: str, *, pred: PredationConfig | None = None,
         land: Landscape | None = None) -> float:
    """按名派发形状；**只认 A1**，其余（含 `None`/`""`）一律 raise。

    镜弃 A2/A3 于 S0（A2 混进涌现统计项污染 A/B 归因；A3 非线性饱和把两件事糊一起＋Red Queen 风险）
    ⇒ 这里给默认值＝替设计做主，给静默替代＝把 S1 的形混进 S0 主判，两者都禁。
    """
    if form != "A1":
        raise ValueError(
            f"B 臂 f(g,n) 形状必须是 'A1'，实得 {form!r}——A1 由镜 b5a7526 定形（R383 设计权），"
            "A2/A3 已判归 S1 稳健性扫描，不在 S0 主判；要改形请走上游设计权，不在代码里静默换。"
        )
    return f_a1(g, nu, pred=pred, land=land)


def w_b(g: float, nu: float, alpha_W: float, *, f_form: str = "A1",
        pred: PredationConfig | None = None, land: Landscape | None = None) -> float:
    """B 臂 `W(g,n) = W₀(g) + α_W·f(g,n)`，即镜式 `p·s₀·E·r·(1 − α_W·ν)`，**clamp ≥ 0**。

    🔴 `α_W == 0.0` ⇒ **短路返回 W₀**（`f` 不求值）：零对照"构造上不该出谷"由代码路径保证；
    写成"算完乘 0"数值上一样、语义上假绿（镜 §三-3 的隔离失效正是这一族，变异靶 M-B 钉住）。
    """
    _check_g(g)
    if not 0.0 <= nu <= 1.0:
        raise ValueError(f"ν 是 [0,1] 的密度比例，实得 {nu}")
    if alpha_W is None or not math.isfinite(alpha_W) or alpha_W < 0.0:
        raise ValueError(
            f"alpha_W 必须是有限非负实数（负值＝正频率依赖，与设计稿 λ≥0 反向；要换方向走设计权），"
            f"实得 {alpha_W!r}"
        )
    w0 = w_b0(g, pred=pred, land=land)
    if alpha_W == 0.0:
        return w0
    return max(0.0, w0 + alpha_W * f_gn(g, nu, f_form, pred=pred or default_predation(),
                                        land=land or Landscape()))


def clamp_is_active(g: float, nu: float, alpha_W: float, **kw) -> bool:
    """镜 §四-1 的守卫件：clamp 是否把该点截到 0（截断点上 `∂²W/∂g∂n` 会局部失效）。

    S-B 数值核查须先扫一边峰位落点是否压在 clamp 界上——**这条判读归砚**，本函数只把可检测性交出去。
    """
    pred = kw.get("pred") or default_predation()
    land = kw.get("land") or Landscape()
    return (w_b0(g, pred=pred, land=land) * (1.0 - alpha_W * nu)) < 0.0


def w_mn(g: float, nu: float, alpha_W: float, *, h: float = 1e-3, **kw) -> float:
    """`∂²W/∂g∂ν`（中心差分，纯 py——H3 禁 scipy）；S-B 的 ν 面口径。"""
    if h <= 0.0:
        raise ValueError(f"差分步长必须 > 0，实得 {h}")
    gp, gm = min(g + h, 1.0), max(g - h, 0.0)
    np_, nm = min(nu + h, 1.0), max(nu - h, 0.0)
    dg, dn = (gp - gm), (np_ - nm)
    if dg <= 0.0 or dn <= 0.0:
        raise ValueError(f"差分退化（g={g}, ν={nu}, h={h}）⇒ 混合偏导不可算")
    return (w_b(gp, np_, alpha_W, **kw) - w_b(gm, np_, alpha_W, **kw)
            - w_b(gp, nm, alpha_W, **kw) + w_b(gm, nm, alpha_W, **kw)) / (dg * dn)


def w_mr(g: float, n: float, alpha_W: float, *, h: float = 1e-3,
         nu_fn=nu_proxy, nu_kw: dict | None = None, **kw) -> float:
    """S-B 结构自证的数值面：`∂²W/∂g∂n`（镜解析式 `= −α_W·∂²[p·s₀·E·ν]/∂g∂n`）。

    ν 由 `nu_fn` 注入（默认 `nu_proxy` 供零机时核查；harness 侧换 `nu_local` 的密度口径）。
    `α_W=0` ⇒ 恒 0（构造上没有 n 依赖）；`α_W>0` 且 ν 随 g/n 变 ⇒ 非 0 ⇒ S-B 必过。
    """
    if h <= 0.0:
        raise ValueError(f"差分步长必须 > 0，实得 {h}")
    nk = nu_kw or {}
    gp, gm = min(g + h, 1.0), max(g - h, 0.0)
    np_, nm = n + h, max(n - h, 0.0)
    dg, dn = (gp - gm), (np_ - nm)
    if dg <= 0.0 or dn <= 0.0:
        raise ValueError(f"差分退化（g={g}, n={n}, h={h}）⇒ W_mr 不可算")
    return (w_b(gp, nu_fn(np_, **nk), alpha_W, **kw) - w_b(gm, nu_fn(np_, **nk), alpha_W, **kw)
            - w_b(gp, nu_fn(nm, **nk), alpha_W, **kw) + w_b(gm, nu_fn(nm, **nk), alpha_W, **kw)
            ) / (dg * dn)


# ------------------------------------------------------------------ #10 谷窗（H2 定参的机读形）

#: R197 原文口径（`docs/实验记录.md:54`；砚/PI 引用的是这一条）
WINDOW_QUOTED_R197 = (250.0, 500.0)
#: 本模块 `has_interior_valley` 实测（k=0、E-034 归一化档、"两端皆高于内部极小"判据）
#: ⇒ **下沿比原文早 50×**（200× 已有真谷），上沿 450×（原文 500×）
WINDOW_MEASURED = (200.0, 450.0)
#: 守卫用**保守并集**：宁可在窗外的低侧多拦一刀，也不让真谷漏进主 2×2
WINDOW_GUARD = (min(WINDOW_QUOTED_R197[0], WINDOW_MEASURED[0]),
                max(WINDOW_QUOTED_R197[1], WINDOW_MEASURED[1]))


def is_inside_valley_window(attack_cost: float, base_cost: float | None = None,
                            window: tuple[float, float] = WINDOW_GUARD) -> bool:
    """H2 §二 的机读判据：`attack_cost / base ∈ window` ⇒ 落 R197 谷窗**内**。

    `base_cost=None` ⇒ 取引擎默认 `attack_cost`（0.1）作 ×1 基准，与 E-034 倍率口径一致。
    🔴 默认窗取**原文 ∪ 实测**的保守并集 `[200,500]`：实测下沿 200×（原文 250×）⇒ 若按 250× 设门槛，
    `[200,250)` 这段真谷会漏拦 ⇒ 建议 C-A11 触发阈值取 **20.0**（=200×0.1）而非 25.0（已上板勘正）。
    """
    base = (default_predation().attack_cost if base_cost is None else base_cost)
    if base <= 0.0:
        raise ValueError(f"基准 cost 必须 > 0，实得 {base}")
    return window[0] <= (attack_cost / base) <= window[1]


def has_interior_valley(*, k: float = 0.0, cost_scale: float = 1.0, n_grid: int = 2001,
                        tol: float = 0.0, pred: PredationConfig | None = None,
                        land: Landscape | None = None) -> bool:
    """A 臂 `W(g)` 是否存在**内部**极小（破坏性选择的必要条件 `W(0) > W(谷) < W(1)`）。

    这是 #10 声明的可复算形：`cost_scale=1` 必须 False、`cost_scale=300` 必须 True。
    🔴 用**形态判据**（两端皆高于内部最小）而非梯度阈：R197 推翻 E-034 的那一条正是
    "绝对梯度阈把光滑极小静默丢弃"（`slope_l<-1e-6 and slope_r>1e-6`）⇒ 不许再用回去。
    """
    if n_grid < 5:
        raise ValueError("n_grid ≥ 5")
    ws = [w_a(i / (n_grid - 1), k=k, cost_scale=cost_scale, pred=pred, land=land,
             gate_inclusive=True) for i in range(n_grid)]
    interior_min = min(ws[1:-1])
    return interior_min < ws[0] - tol and interior_min < ws[-1] - tol


# ------------------------------------------------------------------ 双臂可比性（机制面，非判据）

def assert_arms_comparable(spec_a: dict, spec_b: dict, *, measured_channel: str,
                           extra_allowed: tuple[str, ...] = ()) -> None:
    """草案 §五 的硬校验：两臂**只允许**在 `measured_channel`（＋臂标签等显式许可键）上不同。

    防"顺手多改一个旋钮"＝把被测条件抹掉（AGENTS.md 条件表纪律）；差异或键集合不齐 ⇒ raise 并列清单。
    """
    if measured_channel not in spec_a or measured_channel not in spec_b:
        raise KeyError(f"两臂 spec 必须都含被测通道 {measured_channel!r}")
    allow = {measured_channel, "arm", *extra_allowed}
    keys = set(spec_a) | set(spec_b)
    diffs = sorted(k for k in keys - allow if spec_a.get(k) != spec_b.get(k))
    missing = sorted(keys - set(spec_a)) + sorted(keys - set(spec_b))
    if diffs or missing:
        raise ValueError(
            f"双臂不可比：非被测通道差异 {diffs}，键集合不齐 {missing}"
            "⇒ 两臂差只允许落在一个通道上（草案 §五）"
        )


def arm_specs(seed: int, *, alpha_W: float, **overrides) -> tuple[dict, dict]:
    """同 seed 派生两臂 spec（双流由 harness 侧派生，本函数不起 RNG、不消费 RNG）。

    🔴 A 臂 `alpha_W = None`（该项**不存在**），B 臂 `alpha_W = 值`（可为 0.0＝Z2 零对照）——
    `None` 与 `0.0` 是两个命题，不许混记（H2 落笔 §一.3；同探针"未观测≠0"律，砚⑥）。
    """
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError(f"seed 必须是非负整数（bool 是 int 子类，也算配置错误），实得 {seed!r}")
    base = {"seed": seed, "landscape": Landscape(), "cost_scale": 1.0, "f_form": "A1"}
    base.update(overrides)
    a = dict(base, arm="A", alpha_W=None, f_form=None)
    b = dict(base, arm="B", alpha_W=alpha_W)
    assert_arms_comparable(a, b, measured_channel="alpha_W", extra_allowed=("f_form",))
    return a, b
