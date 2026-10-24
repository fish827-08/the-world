"""Pre-Flight 检查工具（R61-C3/C4 + R78-3 + C5）的纯函数单测。

纪律（R47 精神）：**仪器自己也要被验证**——本工具的作用是在实验前拦截静默配置事故；
若它自己的比较/断言逻辑有错，就会变成"又一件坏仪器"。
"""
import importlib.util
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location(
    "preflight_check", ROOT / "experiments" / "preflight_check.py"
)
pf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pf)


def _rows(pairs):
    return [{"tick": str(t), **{k: str(v) for k, v in d.items()}} for t, d in pairs]


def test_compare_trajectories_identical_detected():
    """F-R9 症状：逐字段完全一致 ⇒ 必须被判为"完全一致"（失败信号）。"""
    a = _rows([(1000, {"N": 300, "g14": 0.5, "g15": 0.4, "trust": 0.5, "max_gen": 3}),
               (2000, {"N": 500, "g14": 0.5, "g15": 0.4, "trust": 0.5, "max_gen": 5})])
    b = [dict(r) for r in a]
    identical, first = pf.compare_trajectories(a, b)
    assert identical is True and first is None


def test_compare_trajectories_divergence_detected():
    a = _rows([(1000, {"N": 300, "g14": 0.5, "g15": 0.4, "trust": 0.5, "max_gen": 3}),
               (2000, {"N": 500, "g14": 0.5, "g15": 0.4, "trust": 0.5, "max_gen": 5})])
    b = _rows([(1000, {"N": 300, "g14": 0.5, "g15": 0.4, "trust": 0.5, "max_gen": 3}),
               (2000, {"N": 501, "g14": 0.5, "g15": 0.4, "trust": 0.5, "max_gen": 5})])
    identical, first = pf.compare_trajectories(a, b)
    assert identical is False and first == 2000


def test_compare_trajectories_length_mismatch_is_difference():
    a = _rows([(1000, {"N": 1, "g14": 0, "g15": 0, "trust": 0, "max_gen": 0})])
    b = _rows([(1000, {"N": 1, "g14": 0, "g15": 0, "trust": 0, "max_gen": 0}),
               (2000, {"N": 2, "g14": 0, "g15": 0, "trust": 0, "max_gen": 1})])
    identical, first = pf.compare_trajectories(a, b)
    assert identical is False and first is None


def test_seed_sensitivity_r78_3():
    """R78-3：std≡0 ⇒ 拒判；有差异 ⇒ 通过。"""
    ok, sd = pf.seed_sensitivity([0.42, 0.42, 0.42])
    assert ok is False and sd == 0.0
    ok2, sd2 = pf.seed_sensitivity([0.42, 0.51, 0.47])
    assert ok2 is True and sd2 > 0
    # 样本不足 ⇒ 不能断言敏感（保守：判失败）
    ok3, _ = pf.seed_sensitivity([0.42])
    assert ok3 is False


def test_readback_problems_catches_f_r9_symptom():
    """F-R9 症状：control 臂若读回 arbitrary_codebook=True ⇒ 必须报问题。"""
    bad = {"arbitrary_codebook": True, "neutral_genes": False,
           "signal_disabled": False, "oracle_enabled": False}
    probs = pf.readback_problems("control", bad)
    assert any("arbitrary_codebook" in p for p in probs)

    good = dict(bad, arbitrary_codebook=False)
    assert pf.readback_problems("control", good) == []
    # 未知臂名 ⇒ 报错而不是静默通过
    assert pf.readback_problems("nope", good) != []


def test_expected_switches_covers_runner_arm_choices():
    """防漂移守卫：runner 的 --arm choices 每新增一个臂，本表必须同步（否则读回核对静默失效）。"""
    src = (ROOT / "experiments" / "a4_verify_capacity.py").read_text(encoding="utf-8")
    m = re.search(r'--arm",\s*choices=\(([^)]*)\)', src)
    assert m, "未能解析 runner 的 --arm choices（若实现改了，请同步本测试）"
    choices = [c.strip().strip('"\'') for c in m.group(1).split(",") if c.strip()]
    assert set(choices) == set(pf.EXPECTED_SWITCHES), (
        f"runner 臂集 {sorted(choices)} 与 preflight 表 {sorted(pf.EXPECTED_SWITCHES)} 不一致"
    )


# ---------------------------------------------------------------- C5 规格自洽检查
# 事故原型（D-24 / G-A 不过）：V-1 把 donation 定为 0.05 而 SIGNAL_COST 是 0.1
# ⇒ return_ratio 结构性上限 = 0.5 < 1.0 ⇒ "保本"语义在数学上不可达。
# C4 全过（每个开关都确实生效了），但规格自相矛盾 —— 这正是 C5 要拦的。

def test_c5_catches_donation_below_signal_cost():
    """D-24 的根因：oracle 启用 + donation(0.05) < SIGNAL_COST(0.1) ⇒ 必须报"保本不可达"。"""
    sw = {"oracle_enabled": True, "donation": 0.05}
    probs = pf.spec_consistency_problems(sw)
    assert len(probs) == 1, probs
    assert "保本" in probs[0], probs


def test_c5_catches_at_exactly_half_and_just_below():
    """0.5×SIGNAL_COST 与"刚好差一点"的档位都要被拦（不许浮点擦边通过）。"""
    for d in (0.05, 0.0999999, 0.0):
        probs = pf.spec_consistency_problems({"oracle_enabled": True, "donation": d})
        assert probs, f"donation={d} 应被拦下"


def test_c5_passes_at_break_even_and_above():
    """保本线（donation == SIGNAL_COST）与反解档（≈0.674）都应通过。"""
    for d in (pf.SIGNAL_COST, 0.674, 0.7, 2.0):
        assert pf.spec_consistency_problems({"oracle_enabled": True, "donation": d}) == [], d


def test_c5_ignores_donation_when_oracle_disabled():
    """oracle 关闭时 donation 不生效 ⇒ 不得据此拦批（C-6：关闭时行为逐位一致）。"""
    assert pf.spec_consistency_problems(
        {"oracle_enabled": False, "donation": 0.05}) == []


def test_c5_flags_negative_donation():
    probs = pf.spec_consistency_problems({"oracle_enabled": True, "donation": -0.1})
    assert any("负" in p for p in probs), probs


def test_c5_no_donation_key_means_not_applicable():
    """非 oracle 批的 switches 里没有 donation ⇒ 无从检查，应静默通过（不得误报）。"""
    assert pf.spec_consistency_problems(
        {"arbitrary_codebook": True, "neutral_genes": False}) == []


def test_c5_tolerates_non_numeric_donation():
    """脏数据不得让工具崩 ⇒ 应降级为一条问题。"""
    probs = pf.spec_consistency_problems({"oracle_enabled": True, "donation": "abc"})
    assert probs and "不是数值" in probs[0]


def test_signal_cost_ceiling_algebra():
    """结构性上限 = donation/SIGNAL_COST（与生态无关的纯代数上界）。"""
    assert pf.signal_cost_ceiling(0.05) == pytest.approx(0.5)
    assert pf.signal_cost_ceiling(0.1) == pytest.approx(1.0)
    assert pf.signal_cost_ceiling(0.674) > 1.0
    # 除零保护
    assert pf.signal_cost_ceiling(0.05, signal_cost=0.0) == float("inf")


def test_breakeven_neutral_boundary_is_ratio_one():
    """🔴 保本中性边界（联网线 2026-09-15 预警；Lewis 原型对照）：

    `ratio = donation/SIGNAL_COST == 1.0` ⇒ sender 期望净收益 = 0 ⇒ **选择差 = 0** ⇒ 漂变主导
    ⇒ 跨 seed CI **结构性易含 0** ⇒ 该臂**不能**当阳性对照（阳性要求 ratio 严格 > 1）。

    本测试把这个"边界值"钉死：`donation == SIGNAL_COST` 恰好落在 ratio=1。
    """
    assert pf.signal_cost_ceiling(pf.SIGNAL_COST) == pytest.approx(1.0)
    # 保本档可通过硬断言（它是合法配置），但低于建议的阳性裕度 ⇒ 应由 advisory 提示
    assert pf.spec_consistency_problems(
        {"oracle_enabled": True, "donation": pf.SIGNAL_COST}) == []
    assert pf.POSITIVE_CONTROL_MIN_RATIO > 1.0


def test_c5_waiver_allows_declared_sub_breakeven():
    """显式逃生阀：声明要研究"补偿不足区制"时放行（须留痕，不得静默）。

    语义（与 config.py 的分工）：断言仍然硬失败（`from_dict` 须能忠实回放旧快照），
    **放行只体现在 Pre-Flight 读回**——即本函数看到 switches 里带了该字段就放过。
    """
    assert pf.spec_consistency_problems(
        {"oracle_enabled": True, "donation": 0.05, "allow_non_breakeven": True}) == []


# ---------------------------------------------------------------- C5 自身静默失效回归
# 🔴 2026-09-15 实跑验收抓到的**本函数自己的**静默失效（正是本工具要防的错型）：
#    `summary.switches` 的键名是 **`oracle_donation`**（带前缀），不是 `donation`。
#    原实现 `if d is None: return []` ⇒ 对 D-24 的 oracle 三臂**静默报 ✅**（本该报"保本不可达"）。
#    以下两条把这个错型钉死。

def test_c5_reads_prefixed_key_from_real_switches():
    """必须能读 `oracle_donation`（真接口键名），否则 C5 对真实产物静默失效。"""
    sw = {"oracle_enabled": True, "oracle_donation": 0.05}
    probs = pf.spec_consistency_problems(sw)
    assert probs, "读不到 oracle_donation ⇒ C5 静默通过（历史真实事故）"
    assert "保本" in probs[0]
    # 无前缀的旧写法也要兼容（防某一侧改了键名）
    assert pf.spec_consistency_problems({"oracle_enabled": True, "donation": 0.05})


def test_c5_does_not_silently_pass_when_donation_unreadable():
    """启用 oracle 却读不到 donation ⇒ 必须**报违规**（"读不到" ≠ "值合法"）。"""
    probs = pf.spec_consistency_problems({"oracle_enabled": True})
    assert probs, "启用 oracle 但 donation 缺失 ⇒ 不得静默通过"
    assert any("读不到" in p for p in probs)


def test_c5_signal_cost_matches_engine():
    """🔴 C5 的核心前提：preflight 的 SIGNAL_COST 必须与引擎真源一致。

    若引擎改了 SIGNAL_COST 而这里没同步，C5 会用**错误的保本线**做断言
    ⇒ 静默失效（正是本工具要防的错型）。故必须硬核对。
    """
    # 🔴 C5 的真源已收拢为 `simulation/config.py:SIGNAL_COST`（2026-09-15）：
    #    此前同源值散在 4 处（引擎两分支局部字面量 / oracle.EMISSION_COST / preflight 本地抄写）
    #    —— 那本身就是"改一处漏一处"的温床。现在只允许一个真源。
    cfg_src = (ROOT / "simulation" / "config.py").read_text(encoding="utf-8")
    m = re.search(r"^SIGNAL_COST\s*:\s*float\s*=\s*([0-9.]+)", cfg_src, re.M)
    assert m, "未能从 config.py 解析 SIGNAL_COST（若实现改了，请同步本测试）"
    assert float(m.group(1)) == pytest.approx(pf.SIGNAL_COST), (
        f"config.SIGNAL_COST={m.group(1)} 与 preflight {pf.SIGNAL_COST} 不一致"
    )
    # 引擎与 oracle 侧不得再有字面量副本（防退化回"多处声明"）
    for f in ("simulation/sphere_engine.py", "simulation/oracle.py"):
        src = (ROOT / f).read_text(encoding="utf-8")
        assert not re.search(r"^\s*SIGNAL_COST\s*=\s*0\.1", src, re.M), f"{f} 出现字面量副本"
        assert not re.search(r"^EMISSION_COST\s*=\s*0\.1", src, re.M), f"{f} 出现 EMISSION_COST 字面量"


def test_preflight_report_mentions_c5():
    """CLI 的检查流程里必须真的接线了 C5（防"函数写了但没被调用"）。"""
    src = (ROOT / "experiments" / "preflight_check.py").read_text(encoding="utf-8")
    assert "spec_consistency_problems(" in src.split("def main(")[1], (
        "main() 未调用 spec_consistency_problems ⇒ C5 未接线"
    )
    assert "C5 规格自洽" in src


# ---- R78-3 结构性恒定豁免（2026-09-15 实跑抓到的假警报）----

def test_zero_arm_g15_constant_is_expected_not_failure():
    """`zero` 臂冻结 g14/g15 ⇒ g15 恒定是**设计意图**，不得判"拒判"。"""
    ok, exempt, sd, msgs = pf.seed_sensitivity_verdict("zero", "g15_final", [0.5, 0.5, 0.5])
    assert ok is True and exempt is True and msgs == []
    assert sd == 0.0


def test_zero_arm_g15_varying_means_freeze_broken():
    """反向自检：该恒定的指标若**有变化** ⇒ 说明冻结没生效 ⇒ 必须失败。"""
    ok, exempt, _sd, msgs = pf.seed_sensitivity_verdict("zero", "g15_final", [0.5, 0.6, 0.5])
    assert ok is False and exempt is True
    assert any("冻结未生效" in m for m in msgs)


def test_zero_arm_N_still_checked_normally():
    """zero 臂的 N **不是**结构性恒定（生态仍随种子变化）⇒ 按常规判。"""
    ok, exempt, _sd, msgs = pf.seed_sensitivity_verdict("zero", "N_final", [10.0, 10.0, 10.0])
    assert ok is False and exempt is False and msgs
    ok2, exempt2, _sd2, _ = pf.seed_sensitivity_verdict("zero", "N_final", [10.0, 12.0, 9.0])
    assert ok2 is True and exempt2 is False


def test_main_arm_constant_still_rejected():
    """非豁免臂的指标 std≡0 ⇒ 仍按 R78-3 拒判（豁免不得扩大化）。"""
    ok, exempt, _sd, msgs = pf.seed_sensitivity_verdict("main", "g15_final", [0.4, 0.4, 0.4])
    assert ok is False and exempt is False
    assert any("拒判" in m for m in msgs)


def test_structurally_constant_table_covers_only_zero_genes():
    """豁免表只允许"由配置冻结"的项，且只限 zero 臂（防被滥用成免检）。"""
    assert set(pf.STRUCTURALLY_CONSTANT) == {"zero"}
    assert pf.STRUCTURALLY_CONSTANT["zero"] == {"g15_final", "g14_final"}
    # 任何其他臂/指标都不得豁免
    for arm in ("main", "control", "sigoff", "oracle"):
        for metric in ("g15_final", "g14_final", "N_final"):
            assert not pf.is_structurally_constant(arm, metric)


def test_no_signal_cost_literal_anywhere_in_controlled_tree():
    """🔴 防退化：字面量副本守卫必须覆盖**全受控目录**，不能只列两个文件名。

    理由：原守卫（上一条测试）只扫 `simulation/sphere_engine.py` 与
    `simulation/oracle.py` ⇒ **新文件（如新增 experiment 脚本）可以悄悄引入
    `SIGNAL_COST = 0.1` 副本而不被拦** —— 这正是"检查工具自带的静默漏洞"
    （2026-09-15 本人在写 `experiments/d27_dose_fit.py` 时**亲手撞上**：
    第一版确实写了字面量，靠人工 grep 才发现）。
    本测试把范围扩到受控代码目录下的全部 `.py`。
    """
    pat = re.compile(r"^\s*(SIGNAL_COST|EMISSION_COST)\s*=\s*[0-9.]+", re.M)
    allow = {"simulation/config.py"}          # 唯一真源（注释：C5）
    bad = []
    for pkg in ("simulation", "observatory", "experiments", "world", "core"):
        base = ROOT / pkg
        if not base.is_dir():
            continue
        for p in base.rglob("*.py"):
            rel = p.relative_to(ROOT).as_posix()
            if rel in allow or "__pycache__" in p.parts:
                continue
            if pat.search(p.read_text(encoding="utf-8")):
                bad.append(rel)
    assert not bad, (
        f"以下文件出现 SIGNAL_COST/EMISSION_COST 字面量副本（应 import 单一真源）：{bad}"
    )
