"""`s0_kernel.arms` 的靶向测试（S0-kernel-b 机制面）。

四类钉法：
1. **历史对账锚**——E-034/R197 两组解析读数按 6 位小数逐位钉死（移植错 ⇒ 红，不放过）；
2. **镜定形照抄**（b5a7526）——B 臂 `W₀=p·s₀·E·r`（整块去 gate、无素食/成本腿）、A1 线性折扣、
   `s₀(1−α_W ν)` clamp≥0、`α_W=0` 单变量零对照 ⇒ 每条一钉；
3. **fail-loud**——形状非 A1、ν 越界、`α_W` 非有限/负、差分退化、双臂多改旋钮，全部当场 raise；
4. **不消费 RNG／零行为面**——AST 级 import 守卫 ＋ 主引擎不 import `s0_kernel` 的子进程实证。

变异靶（改错必红）：M-A 删 A 臂 gate mask／M-B α_W=0 去掉短路／M-C 非 A1 静默替代／
M-D·M-E 系数裸写／M-F B 臂误加回 gate／M-G clamp 去掉／M-H ν 用全场代替近邻。
"""
from __future__ import annotations

import math

import pytest

from s0_kernel.arms import (F_FORMS, INCONCLUSIVE_EXIT, SIGMA_C_PROVISIONAL, WINDOW_GUARD,
                            WINDOW_MEASURED, WINDOW_QUOTED_R197, Landscape, arm_specs,
                            assert_arms_comparable, clamp_is_active, default_predation, f_a1,
                            f_gn, has_interior_valley, is_inside_valley_window, nu_local,
                            nu_proxy, w_a, w_b, w_b0, w_mn, w_mr)

ANCHOR = 6      # 与引擎 digest 同用 6 位小数约定


# ---------------------------------------------------------------- ① A 臂历史对账锚（E-034／R197）

def test_e034_k2_readings_match_documented_anchors() -> None:
    """证伪文档 §3.4 的 k=2 五读数（1.0000／1.0720／1.3700／1.8680／2.9900）逐位复现。"""
    got = [round(w_a(g, k=2.0), ANCHOR) for g in (0.0, 0.3, 0.5, 0.7, 1.0)]
    # g=0.3 == gate ⇒ 严格 `>` 下不入场（0.49）；文档报的 1.0720 是**右极限**（见模块 docstring）
    assert got == [1.0, 0.49, 1.37, 1.868, 2.99], got


def test_e034_gate_is_a_jump_discontinuity() -> None:
    """边界歧义钉死：`W(gate)` 两侧差 0.582 ⇒ 拿 `g=gate` 对账的脚本必然静默错一位。"""
    lo, hi = w_a(0.3 - 1e-9, k=2.0), w_a(0.3 + 1e-9, k=2.0)
    assert round(hi, ANCHOR) == 1.072 and round(lo, ANCHOR) == 0.49
    assert round(hi - lo, ANCHOR) == 0.582
    assert round(w_a(0.3, k=2.0, gate_inclusive=True), ANCHOR) == 1.072


def test_r197_cost_x300_valley_readings() -> None:
    """R197 谷读数：k=0、cost×300 ⇒ `W(0)=W(1)=1.0000`、中点 `0.6250`。"""
    assert round(w_a(0.0, k=0.0, cost_scale=300.0), ANCHOR) == 1.0
    assert round(w_a(0.5, k=0.0, cost_scale=300.0), ANCHOR) == 0.625
    assert round(w_a(1.0, k=0.0, cost_scale=300.0), ANCHOR) == 1.0


def test_valley_window_measured_edges_versus_quoted() -> None:
    """#10 的可复算形：实测窗 **[200,450]**；原文 [250,500] ⇒ 下沿早 50×，守卫取并集。"""
    assert has_interior_valley(k=0.0, cost_scale=1.0) is False     # S0 默认档：窗外（H2 定参）
    assert has_interior_valley(k=0.0, cost_scale=195.0) is False
    assert has_interior_valley(k=0.0, cost_scale=200.0) is True    # 实测下沿（原文写 250）
    assert has_interior_valley(k=0.0, cost_scale=300.0) is True    # R197 实测有谷
    assert has_interior_valley(k=0.0, cost_scale=445.0) is True
    assert has_interior_valley(k=0.0, cost_scale=500.0) is False   # 高成本把 g=1 端压成全局极小
    assert WINDOW_QUOTED_R197 == (250.0, 500.0)
    assert WINDOW_MEASURED == (200.0, 450.0)
    assert WINDOW_GUARD == (200.0, 500.0)


def test_is_inside_valley_window_uses_engine_default_as_x1() -> None:
    """机读判据（H2 §二）：基准＝引擎默认 `attack_cost`，A 臂 ×1 ⇒ 窗外。"""
    base = default_predation().attack_cost
    assert base == 0.1 and default_predation().forage_tradeoff_k == 0.0
    assert is_inside_valley_window(base) is False
    assert is_inside_valley_window(base * 199) is False
    assert is_inside_valley_window(base * 200) is True             # 实测下沿＝绝对 20.0
    assert is_inside_valley_window(base * 250) is True             # 原文下沿
    assert is_inside_valley_window(base * 500) is True
    assert is_inside_valley_window(base * 501) is False
    with pytest.raises(ValueError, match="基准 cost"):
        is_inside_valley_window(base, base_cost=0.0)


# ---------------------------------------------------------------- ② B 臂＝镜 b5a7526 定形

def test_w_b0_is_gate_free_monotone_and_excludes_forage_and_cost() -> None:
    """`W₀ = p·s₀·E_prey·r`：无 gate 跳变、随 g 单调增、**不含**素食腿与成本腿（镜原文式子）。"""
    land, pred = Landscape(), default_predation()
    manual = (0.6 * pred.attack_prob_coef * land.hunger) * (
        (land.e_self / (land.e_self + land.e_prey)) * (0.5 + 0.6 * pred.success_gene_gain)
        * land.e_prey * pred.transfer_ratio)
    assert abs(w_b0(0.6) - manual) < 1e-15
    vals = [w_b0(i / 200.0) for i in range(201)]
    assert all(b >= a - 1e-15 for a, b in zip(vals, vals[1:])), "W₀ 必须单调不降"
    # 🔴 变异靶 M-F：B 臂误加回 gate ⇒ 下面两条同时红
    assert abs(w_b0(0.3 + 1e-9) - w_b0(0.3 - 1e-9)) < 1e-6      # gate 处不跳
    assert w_b0(0.0) == 0.0                                     # 0.3 不是入场券
    assert w_b0(0.3) < w_b0(0.31)


def test_w_b_equals_mirror_formula_and_isomorphic_to_additive_form() -> None:
    """`W₀+α_W·f_A1` 与镜式 `p·s₀·E·r·(1−α_W ν)` 1:1 同构（线性折扣展开）。"""
    for g, nu, aw in [(0.6, 0.3, 1.0), (0.8, 0.05, 0.4), (0.45, 0.9, 1.0), (1.0, 1.0, 0.7)]:
        w0 = w_b0(g)
        assert abs(w_b(g, nu, aw) - max(0.0, w0 + aw * f_a1(g, nu))) < 1e-15
        assert abs(w_b(g, nu, aw) - max(0.0, w0 * (1.0 - aw * nu))) < 1e-15


def test_clamp_keeps_W_nonnegative_and_is_detectable() -> None:
    """🔴 变异靶 M-G：`s₀(1−α_W ν) ≥ 0` 的 clamp 被去掉 ⇒ 负适应度，本条红。"""
    g, nu, aw = 0.9, 0.95, 2.0                       # α_W·ν = 1.9 ⇒ 未 clamp 会为负
    assert clamp_is_active(g, nu, aw) is True
    assert w_b(g, nu, aw) == 0.0
    assert w_b(g, 0.2, 1.0) > 0.0 and clamp_is_active(g, 0.2, 1.0) is False
    assert all(w_b(g / 10.0, nu / 10.0, 3.0) >= 0.0 for g in range(11) for nu in range(11))


def test_alpha_W_zero_is_clean_zero_control() -> None:
    """`α_W=0` ⇒ 逐位等于 `W₀`（Z2；镜 §三-3：整块去 gate 是让 α_W=0 成为单变量对照的前提）。"""
    for g in (0.0, 0.25, 0.5, 0.75, 1.0):
        for nu in (0.0, 0.5, 1.0):
            assert w_b(g, nu, 0.0) == w_b0(g)
    assert round(w_mn(0.6, 0.5, 0.0), 12) == 0.0
    assert round(w_mr(0.6, 30.0, 0.0), 12) == 0.0     # 构造上没有 n 依赖


def test_alpha_W_zero_short_circuits_f_entirely() -> None:
    """🔴 变异靶 M-B：零对照必须**不求值 f**（"算完乘 0"数值一样、语义假绿 ⇒ 只有哨兵抓得到）。"""
    import s0_kernel.arms as mod

    def boom(*a, **kw):
        raise AssertionError("α_W=0 时 f_gn 被调用 ⇒ 短路语义丢了")

    orig, mod.f_gn = mod.f_gn, boom
    try:
        assert w_b(0.4, 0.7, 0.0) == w_b0(0.4)
        assert w_b(0.4, 0.7, 0.0, f_form="A1") == w_b0(0.4)
    finally:
        mod.f_gn = orig


def test_f_a1_is_negative_feedback_and_linear_in_alpha_W() -> None:
    """A1 ⇒ ν 增则 W 降（负频率依赖），且 α_W 只线性缩放该项。"""
    assert f_a1(0.6, 0.0) == 0.0 and f_a1(0.6, 0.5) < 0.0
    seq = [w_b(0.6, nu, 1.0) for nu in (0.0, 0.2, 0.5, 0.8, 1.0)]
    assert all(b <= a + 1e-15 for a, b in zip(seq, seq[1:])), seq
    once, twice = w_b(0.6, 0.3, 1.0) - w_b0(0.6), w_b(0.6, 0.3, 2.0) - w_b0(0.6)
    assert abs(twice - 2.0 * once) < 1e-15


def test_nu_local_is_neighbor_kernel_density_not_global_count() -> None:
    """🔴 变异靶 M-H：ν 若改用全场总数／attacker 占比 ⇒ 本条红（镜 §一：局部竞争核口径）。"""
    same = [0.6] * 8
    far = [0.6 + 0.4 * ((-1) ** i) for i in range(8)]        # 表型远离自身
    assert nu_local(0.6, same) == pytest.approx(1.0, abs=1e-12)
    assert nu_local(0.6, far) < 0.05
    assert nu_local(0.6, []) == 0.0
    mixed = same + far                                       # 一半同类一半远类 ⇒ ν≈0.5
    assert nu_local(0.6, mixed) == pytest.approx(0.5, abs=2e-3)   # 远类核尾贡献一点点（>0）
    assert nu_local(0.6, mixed) < nu_local(0.6, same)
    assert nu_local(0.6, mixed) > 0.0
    assert nu_local(0.6, far) > 0.0                            # 核是连续的，不是硬阈值
    with pytest.raises(ValueError, match="sigma_c"):
        nu_local(0.5, [0.4], sigma_c=0.0)
    assert SIGMA_C_PROVISIONAL > 0.0     # 待标定批锁的单值；manifest 须回显实际用值


def test_nu_proxy_bounds_and_monotone() -> None:
    """`nu_proxy` 只作 S-B 零机时核查的单调替身（不得进 harness 出货路径）；越界即 raise。"""
    assert nu_proxy(0.0) == 0.0
    assert nu_proxy(50.0) == pytest.approx(0.5)
    assert nu_proxy(1e6) == 1.0                               # ν 是比例，上限截在 1
    for bad in (-1.0, -0.001):
        with pytest.raises(ValueError, match="非负"):
            nu_proxy(bad)
    with pytest.raises(ValueError, match="n_ref"):
        nu_proxy(1.0, n_ref=0.0)


# ---------------------------------------------------------------- ③ W_mr／S-B 数值面

def test_w_mn_and_w_mr_sign_and_s_b_passability() -> None:
    """S-B 数值面可用：`∂²W/∂g∂ν < 0`（α_W>0）；`∂²W/∂g∂n ≠ 0` 与镜解析式同号。"""
    v = w_mn(0.6, 0.4, 1.0)
    assert math.isfinite(v) and v < 0.0
    assert abs(w_mn(0.6, 0.4, 2.0) - 2.0 * v) < 1e-9
    m = w_mr(0.6, 40.0, 1.0)
    assert math.isfinite(m) and m < 0.0
    assert abs(w_mr(0.3, 40.0, 1.0)) < abs(w_mr(0.9, 40.0, 1.0))   # 稀有表型竞争更弱（DD99 形状）


def test_w_mr_clamp_region_reports_degeneracy() -> None:
    """镜 §四-1：clamp 截断点上混合偏导局部归零 ⇒ `clamp_is_active` 交砚做 S-B 前置扫描。"""
    assert clamp_is_active(0.95, 0.99, 2.0) is True
    assert w_mn(0.95, 0.99, 2.0) == 0.0              # 被压平 ⇒ 该点无数值曲率
    assert w_mn(0.95, 0.30, 1.0) < 0.0               # 未截断点正常非零


@pytest.mark.parametrize("h", [0.0, -1.0])
def test_w_mr_degenerate_step_raises(h) -> None:
    with pytest.raises(ValueError, match="差分"):
        w_mr(0.5, 10.0, 1.0, h=h)
    with pytest.raises(ValueError, match="差分"):
        w_mn(0.5, 0.5, 1.0, h=h)


# ---------------------------------------------------------------- ④ fail-loud 面

def test_f_form_only_A1_no_silent_substitute() -> None:
    """🔴 变异靶 M-C：给非 A1 找"最近的默认"静默替代 ⇒ 本条红（A2/A3 判归 S1，镜 §二）。"""
    assert F_FORMS == ("A1",)
    for bad in ("A2", "A3", "", None, "a1"):
        with pytest.raises(ValueError, match="A1"):
            f_gn(0.5, 0.3, bad)
        with pytest.raises(ValueError, match="A1"):
            w_b(0.5, 0.3, 1.0, f_form=bad)


@pytest.mark.parametrize("g", [-0.1, 1.5])
def test_g_axis_is_bounded_raw(g) -> None:
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        w_a(g)
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        w_b0(g)
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        w_b(g, 0.5, 1.0)


@pytest.mark.parametrize("nu", [-0.01, 1.01, float("nan")])
def test_nu_must_be_a_ratio(nu) -> None:
    with pytest.raises(ValueError, match="ν"):
        w_b(0.5, nu, 1.0)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), None, -1.0])
def test_alpha_W_must_be_finite_nonneg(bad) -> None:
    with pytest.raises(ValueError, match="alpha_W"):
        w_b(0.5, 0.3, bad)


def test_cost_scale_and_landscape_validate() -> None:
    with pytest.raises(ValueError, match="cost_scale"):
        w_a(0.5, cost_scale=-1.0)
    with pytest.raises(ValueError, match="hunger"):
        Landscape(hunger=1.5)
    with pytest.raises(ValueError, match="sigma_c"):
        Landscape(sigma_c=0.0)


# ---------------------------------------------------------------- ⑤ 系数单一来源

@pytest.mark.parametrize("field,bumped", [
    ("attack_prob_coef", 0.35), ("attack_gene_gate", 0.7), ("success_gene_gain", 0.8),
    ("success_floor", 0.5), ("success_ceil", 0.35), ("transfer_ratio", 0.5),
    ("attack_cost", 0.05), ("forage_tradeoff_k", 1.0),
])
def test_coefficients_are_read_not_hardcoded(monkeypatch, field, bumped) -> None:
    """🔴 变异靶 M-D/M-E：八个系数任一个被裸写 ⇒ 对应那档红（改 config 而 W 不动＝裸写）。"""
    import s0_kernel.arms as mod
    from simulation.config import PredationConfig

    base_a, base_b = w_a(0.6), w_b0(0.6)
    if getattr(PredationConfig(), field) == bumped:
        pytest.fail(f"档本身无效：bumped={bumped} 与默认值相同")

    def bumped_pred() -> PredationConfig:
        q = PredationConfig()
        setattr(q, field, bumped)
        return q

    monkeypatch.setattr(mod, "default_predation", bumped_pred)
    assert w_a(0.6) != base_a, f"{field} 被裸写（A 臂 W 不吃 config）"
    if field in ("attack_cost", "forage_tradeoff_k", "attack_gene_gate"):
        # 镜的 W₀ 只含 p·s₀·E·r：成本腿／素食腿／gate 不入 B 臂是**设计事实**，不是裸写
        assert w_b0(0.6) == base_b, f"{field} 竟进了 W₀ ⇒ B 臂不再是镜 b5a7526 的形"
    else:
        assert w_b0(0.6) != base_b, f"{field} 被裸写（B 臂 W₀ 腿不吃 config）"


def test_default_config_matches_h2_declared_parameters() -> None:
    """H2 #10 定参的守卫：引擎默认被改动 ⇒ 本例红 ⇒ 强制重算谷窗声明与倍率基准。"""
    q = default_predation()
    assert (q.attack_cost, q.forage_tradeoff_k) == (0.1, 0.0)
    assert (q.attack_prob_coef, q.attack_gene_gate, q.success_gene_gain) == (0.2, 0.3, 0.5)
    assert (q.success_floor, q.success_ceil, q.transfer_ratio) == (0.1, 0.9, 0.4)


# ---------------------------------------------------------------- ⑥ 双臂可比性／出货纪律

def test_arms_comparable_rejects_extra_knob() -> None:
    """草案 §五：两臂只许差一个被测通道；多改一个旋钮当场红（AGENTS.md 条件表纪律）。"""
    a, b = arm_specs(207, alpha_W=1.0)
    assert a["alpha_W"] is None and b["alpha_W"] == 1.0     # A 臂记 None，不记 0.0（H2 §一.3）
    assert a["f_form"] is None and a["seed"] == b["seed"] == 207
    with pytest.raises(ValueError, match="不可比"):
        assert_arms_comparable(a, dict(b, cost_scale=2.0), measured_channel="alpha_W",
                               extra_allowed=("f_form",))
    with pytest.raises(ValueError, match="不可比"):
        assert_arms_comparable(a, dict(b, landscape=Landscape(sigma_c=0.2)),
                               measured_channel="alpha_W", extra_allowed=("f_form",))
    with pytest.raises(KeyError):
        assert_arms_comparable(a, b, measured_channel="not_a_channel")
    a2, b2 = arm_specs(207, alpha_W=0.0)                    # Z2 对照批与运行批只差 α_W 的值
    assert b2["alpha_W"] == 0.0 and a2["alpha_W"] is None


def test_arm_specs_rejects_bad_seed() -> None:
    for bad in (-1, "7", 2.5, True):
        with pytest.raises(ValueError, match="seed"):
            arm_specs(bad, alpha_W=1.0)


def test_inconclusive_exit_is_declared_in_code() -> None:
    """「两臂皆否 ⇒ S0 尺度不可裁决」跑前就在代码里，不是事后解释（草案 §五 预注册红线）。"""
    from s0_kernel.arms import declare_inconclusive_exit
    assert declare_inconclusive_exit() == INCONCLUSIVE_EXIT
    assert "不可裁决" in INCONCLUSIVE_EXIT and "跑批前" in INCONCLUSIVE_EXIT


def test_arms_module_is_deterministic_and_rng_free() -> None:
    """不消费 RNG：AST 级守卫（docstring 里提 random 不算违规，只看 import 节点）。"""
    import ast
    import s0_kernel.arms as mod

    seq = [w_a(g / 20.0, k=1.0) for g in range(21)]
    assert seq == [w_a(g / 20.0, k=1.0) for g in range(21)]
    seqb = [w_b(0.7, nu / 10.0, 1.0) for nu in range(11)]
    assert seqb == [w_b(0.7, nu / 10.0, 1.0) for nu in range(11)]

    tree = ast.parse(open(mod.__file__, encoding="utf-8").read())
    got = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            got |= {al.name for al in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            got.add(node.module)
    assert got <= {"__future__", "math", "dataclasses", "simulation.config"}, sorted(got)


def test_main_engine_does_not_import_s0_kernel() -> None:
    """零行为面实证：干净子进程里 import 主引擎后，`s0_kernel` 不在 `sys.modules`。"""
    import subprocess
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    r = subprocess.run([sys.executable, "-c",
                        "import simulation.sphere_engine, sys;"
                        "print('s0_kernel' in sys.modules)"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace",
                       cwd=str(root), timeout=300)
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip().endswith("False"), r.stdout
